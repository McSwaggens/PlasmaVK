/* Resource churn: what streaming engines and render graphs do every frame -
 * create buffers, images, views and samplers, bind them to memory, upload,
 * use them, and destroy them a few frames later, once the GPU is done with
 * them (as deferred deletion queues do). Every frame's new image views also
 * go into a descriptor array, as engines add new textures to their texture
 * tables.
 *
 *   textures_dedicated     8 BC7 textures per frame (128..1024, full mips),
 *                          each with its own VkDeviceMemory (dedicated, as
 *                          VMA does for large textures and simple engines
 *                          for all), a view, and uploads of every mip from a
 *                          staging ring
 *   textures_suballoc      the same, bound at offsets of big blocks (VMA)
 *   textures_suballoc_ws4k the same with 4000 other textures and views alive
 *                          (a game's material set)
 *   buffers_dedicated      32 buffers per frame: 16 host-visible staging
 *                          buffers (4..64 KB, mapped and written) copied
 *                          into 16 device-local ones, each with its own memory
 *   buffers_dedicated_256k the same with 64..512 KB buffers
 *   buffers_suballoc       buffers_dedicated bound at offsets of big blocks
 *   rendertargets_aliased  a render graph's 6 transient 1280x720 targets (5
 *                          colors and a depth) with views, created per frame
 *                          in a memory block that is reused every 3 frames,
 *                          and cleared in one render pass instance
 *   views_x128             128 one-mip, one-layer views per frame of a
 *                          persistent 12-mip, 6-layer texture (mip
 *                          generation, cube faces, shadow cascades)
 *   samplers_x16           16 samplers per frame (the same 16 parameter sets)
 *   load_oneshot_x4        load-time uploads as samples and simple engines do
 *                          them: per texture a staging buffer of its own, the
 *                          image, one command buffer submitted and waited
 *                          for, the staging buffer freed, then a view and a
 *                          sampler
 *   bufferviews_x64        64 texel buffer views per frame of slices of one
 *                          buffer (D3D-on-Vulkan layers renaming buffers)
 *
 * The _named variants also give every new resource a VK_EXT_debug_utils
 * name, as engines that name everything do. The instance enables the
 * extension wherever it is offered (for every configuration); without it
 * they are skipped.
 *
 * cpu_us is the frame's CPU time; ns_per_op has each call's mean cost
 * (write_descriptor's per descriptor written: one vkUpdateDescriptorSets
 * writes all of a frame's new views) and resource_us_per_frame their sum.
 * The first frame's uploads, copies, clears or views are read back and
 * checked (samplers and buffer views are not used by any command, so there
 * is nothing to check).
 */
#include "bench_vk.h"

#define RING 3 /* resources die RING frames after creation */
#define MAX_PER_FRAME 128
#define STAGING_SLOT (16u << 20)
#define BLOCK_BYTES (64u << 20)
#define READBACK_BYTES (4u << 20)
#define CHECK_VIEW 13 /* views_x128: the view the check renders through (layer 1, mip 2) */

enum kind { K_TEXTURES, K_BUFFERS, K_TARGETS, K_VIEWS, K_SAMPLERS, K_ONESHOT, K_BUFFER_VIEWS };

struct config {
    const char *name;
    enum kind kind;
    int dedicated;
    int working_set; /* textures (and views) alive besides the churn */
    int per_frame;
    int frames;
    int named;
    uint32_t buffer_kb[2]; /* K_BUFFERS: sizes from [0] to [1] KB */
};

static const struct config configs[] = {
    { "textures_dedicated", K_TEXTURES, 1, 0, 8, 300, 0, { 0, 0 } },
    { "textures_suballoc", K_TEXTURES, 0, 0, 8, 300, 0, { 0, 0 } },
    { "textures_suballoc_ws4k", K_TEXTURES, 0, 4000, 8, 300, 0, { 0, 0 } },
    { "buffers_dedicated", K_BUFFERS, 1, 0, 32, 300, 0, { 4, 64 } },
    { "buffers_dedicated_256k", K_BUFFERS, 1, 0, 32, 300, 0, { 64, 512 } },
    { "buffers_suballoc", K_BUFFERS, 0, 0, 32, 300, 0, { 4, 64 } },
    { "rendertargets_aliased", K_TARGETS, 0, 0, 6, 300, 0, { 0, 0 } },
    { "views_x128", K_VIEWS, 0, 0, 128, 300, 0, { 0, 0 } },
    { "samplers_x16", K_SAMPLERS, 0, 0, 16, 300, 0, { 0, 0 } },
    { "load_oneshot_x4", K_ONESHOT, 1, 0, 4, 200, 0, { 0, 0 } },
    { "bufferviews_x64", K_BUFFER_VIEWS, 0, 0, 64, 300, 0, { 0, 0 } },
    /* Last: the configurations above run after the same ones as when these were skipped. */
    { "textures_suballoc_named", K_TEXTURES, 0, 0, 8, 300, 1, { 0, 0 } },
    { "rendertargets_aliased_named", K_TARGETS, 0, 0, 6, 300, 1, { 0, 0 } },
};

enum op {
    OP_CREATE_IMAGE, OP_IMAGE_REQS, OP_ALLOC, OP_BIND_IMAGE, OP_CREATE_VIEW, OP_CREATE_BUFFER, OP_BUFFER_REQS,
    OP_BIND_BUFFER, OP_MAP, OP_DESTROY_VIEW, OP_DESTROY_IMAGE, OP_DESTROY_BUFFER, OP_FREE, OP_CREATE_SAMPLER,
    OP_DESTROY_SAMPLER, OP_WRITE_DESC, OP_ONESHOT, OP_CREATE_BVIEW, OP_DESTROY_BVIEW, OP_NAME, OP_COUNT
};
static const char *op_names[OP_COUNT] = {
    "create_image", "image_reqs", "allocate_memory", "bind_image", "create_view", "create_buffer", "buffer_reqs",
    "bind_buffer", "map_memory", "destroy_view", "destroy_image", "destroy_buffer", "free_memory", "create_sampler",
    "destroy_sampler", "write_descriptor", "oneshot_submit_wait", "create_buffer_view", "destroy_buffer_view",
    "set_object_name",
};

static int debug_names; /* VK_EXT_debug_utils enabled: the _named configurations can run */

struct res {
    VkImage image;
    VkImageView view;
    VkBuffer buffer;
    VkDeviceMemory memory; /* dedicated only */
    VkSampler sampler;
    VkBufferView buffer_view;
};

/* What one frame created, destroyed RING frames later. */
struct ring_set {
    struct res r[2 * MAX_PER_FRAME];
    int n;
    VkDeviceSize block_used, block2_used;
};

struct scene {
    struct bench_vk *vk;
    const struct config *cfg;
    VkFormat tex_format;
    uint32_t tex_block_bytes; /* 16: BC7; 0: RGBA8 (4-byte texels) */
    uint32_t host_type, device_type;
    struct ring_set sets[RING];
    VkDeviceMemory blocks[RING], blocks2[RING]; /* suballoc: device-local (and host-visible for buffers) */
    void *block2_map[RING];
    struct bench_buffer staging; /* RING slots of STAGING_SLOT */
    struct bench_buffer readback;
    struct bench_image big;    /* views_x128 */
    struct bench_buffer texel; /* bufferviews_x64 */
    VkDeviceMemory ws_memory;  /* working set */
    VkImage *ws_images;
    VkImageView *ws_views;
    int ws_count;
    /* Every frame's views go into the descriptor set of its ring slot. */
    VkDescriptorSetLayout set_layout;
    VkDescriptorPool desc_pool;
    VkDescriptorSet desc_sets[RING];
    VkSampler sampler;
    VkCommandPool oneshot_pool; /* load_oneshot */
    VkCommandBuffer oneshot_cmd;
    VkFence oneshot_fence;
    /* Timing */
    uint32_t first_measured; /* frame index */
    int measuring;
    double op_us[OP_COUNT];
    long op_n[OP_COUNT];
    unsigned name_serial;
    uint64_t rng;
    /* Check: what the first frame wrote to the readback buffer */
    VkDeviceSize check_offset; /* textures: the upload's offset in the staging buffer */
    uint32_t check_size;       /* textures: mip 0 size */
    uint32_t check_value;      /* buffers: the byte; render targets: the RGBA8 texel */
};

#define TIMED(s, op, call)                              \
    do {                                                \
        double t0_ = bench_now_us();                    \
        call;                                           \
        if ((s)->measuring) {                           \
            (s)->op_us[op] += bench_now_us() - t0_;     \
            (s)->op_n[op]++;                            \
        }                                               \
    } while (0)

/* _named: every new resource gets a name. */
static void name_object(struct scene *s, VkObjectType type, uint64_t handle, const char *what)
{
    VkDebugUtilsObjectNameInfoEXT info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
        .objectType = type,
        .objectHandle = handle,
    };
    char name[64];
    VkResult r;
    if (!s->cfg->named)
        return;
    snprintf(name, sizeof(name), "%s %u", what, s->name_serial++);
    info.pObjectName = name;
    TIMED(s, OP_NAME, r = vkd.SetDebugUtilsObjectNameEXT(s->vk->device, &info));
    BENCH_VK(r);
}

static uint32_t mip_count(uint32_t size)
{
    uint32_t m = 1;
    while (size > 1) {
        size >>= 1;
        m++;
    }
    return m;
}

/* Bytes of one mip of the streamed texture format. */
static VkDeviceSize mip_bytes(const struct scene *s, uint32_t size, uint32_t mip)
{
    uint32_t w = size >> mip ? size >> mip : 1;
    if (s->tex_block_bytes) {
        uint32_t blocks = (w + 3) / 4;
        return (VkDeviceSize)blocks * blocks * s->tex_block_bytes;
    }
    return (VkDeviceSize)w * w * 4;
}

static VkDeviceSize align_up(VkDeviceSize x, VkDeviceSize a)
{
    return (x + a - 1) / a * a;
}

/* Memory requirements the way VMA asks for them (dedicated preference). */
static void image_reqs(struct scene *s, VkImage image, VkMemoryRequirements *out)
{
    VkMemoryDedicatedRequirements dedicated = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
    VkMemoryRequirements2 req = { .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated };
    VkImageMemoryRequirementsInfo2 info = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
                                            .image = image };
    TIMED(s, OP_IMAGE_REQS, vkd.GetImageMemoryRequirements2(s->vk->device, &info, &req));
    *out = req.memoryRequirements;
}

static void buffer_reqs(struct scene *s, VkBuffer buffer, VkMemoryRequirements *out)
{
    VkMemoryDedicatedRequirements dedicated = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
    VkMemoryRequirements2 req = { .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated };
    VkBufferMemoryRequirementsInfo2 info = { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
                                             .buffer = buffer };
    TIMED(s, OP_BUFFER_REQS, vkd.GetBufferMemoryRequirements2(s->vk->device, &info, &req));
    *out = req.memoryRequirements;
}

static void check_type(const VkMemoryRequirements *req, uint32_t type)
{
    if (!(req->memoryTypeBits & (1u << type)))
        bench_fatal("memory type %u cannot hold the resource (memoryTypeBits 0x%x)", type, req->memoryTypeBits);
}

static VkDeviceMemory alloc_dedicated(struct scene *s, const VkMemoryRequirements *req, uint32_t type, VkImage image,
                                      VkBuffer buffer)
{
    VkMemoryDedicatedAllocateInfo dedicated = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                                .image = image, .buffer = buffer };
    VkMemoryAllocateInfo info = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &dedicated,
                                  .allocationSize = req->size, .memoryTypeIndex = type };
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkResult r;
    check_type(req, type);
    TIMED(s, OP_ALLOC, r = vkd.AllocateMemory(s->vk->device, &info, NULL, &memory));
    BENCH_VK(r);
    return memory;
}

static VkDeviceMemory alloc_block(struct scene *s, VkDeviceSize size, uint32_t type)
{
    VkMemoryAllocateInfo info = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = size,
                                  .memoryTypeIndex = type };
    VkDeviceMemory memory = VK_NULL_HANDLE;
    BENCH_VK(vkd.AllocateMemory(s->vk->device, &info, NULL, &memory));
    return memory;
}

static void destroy_set(struct scene *s, struct ring_set *set)
{
    VkDevice dev = s->vk->device;
    int i;
    for (i = 0; i < set->n; i++) {
        struct res *r = &set->r[i];
        if (r->view)
            TIMED(s, OP_DESTROY_VIEW, vkd.DestroyImageView(dev, r->view, NULL));
        if (r->image)
            TIMED(s, OP_DESTROY_IMAGE, vkd.DestroyImage(dev, r->image, NULL));
        if (r->buffer)
            TIMED(s, OP_DESTROY_BUFFER, vkd.DestroyBuffer(dev, r->buffer, NULL));
        if (r->memory)
            TIMED(s, OP_FREE, vkd.FreeMemory(dev, r->memory, NULL));
        if (r->sampler)
            TIMED(s, OP_DESTROY_SAMPLER, vkd.DestroySampler(dev, r->sampler, NULL));
        if (r->buffer_view)
            TIMED(s, OP_DESTROY_BVIEW, vkd.DestroyBufferView(dev, r->buffer_view, NULL));
        memset(r, 0, sizeof(*r));
    }
    set->n = 0;
    set->block_used = set->block2_used = 0;
}

static VkImage create_image(struct scene *s, VkFormat format, uint32_t w, uint32_t h, uint32_t mips, uint32_t layers,
                            VkImageUsageFlags usage)
{
    VkImageCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = { w, h, 1 },
        .mipLevels = mips,
        .arrayLayers = layers,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage image = VK_NULL_HANDLE;
    VkResult r;
    TIMED(s, OP_CREATE_IMAGE, r = vkd.CreateImage(s->vk->device, &info, NULL, &image));
    BENCH_VK(r);
    name_object(s, VK_OBJECT_TYPE_IMAGE, (uint64_t)(uintptr_t)image, "image");
    return image;
}

static VkImageView create_view(struct scene *s, VkImage image, VkFormat format, VkImageAspectFlags aspect,
                               uint32_t base_mip, uint32_t mips, uint32_t base_layer)
{
    VkImageViewCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = { aspect, base_mip, mips, base_layer, 1 },
    };
    VkImageView view = VK_NULL_HANDLE;
    VkResult r;
    TIMED(s, OP_CREATE_VIEW, r = vkd.CreateImageView(s->vk->device, &info, NULL, &view));
    BENCH_VK(r);
    name_object(s, VK_OBJECT_TYPE_IMAGE_VIEW, (uint64_t)(uintptr_t)view, "view");
    return view;
}

/* Binds an image to memory of its own (returned), or at the set's next
 * offset in block. */
static VkDeviceMemory bind_image(struct scene *s, struct ring_set *set, VkDeviceMemory block, VkImage image,
                                 int dedicated)
{
    VkMemoryRequirements req;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkResult r;
    image_reqs(s, image, &req);
    if (dedicated) {
        memory = alloc_dedicated(s, &req, s->device_type, image, VK_NULL_HANDLE);
        TIMED(s, OP_BIND_IMAGE, r = vkd.BindImageMemory(s->vk->device, image, memory, 0));
    } else {
        VkDeviceSize offset = align_up(set->block_used, req.alignment);
        check_type(&req, s->device_type);
        if (offset + req.size > BLOCK_BYTES)
            bench_fatal("a frame's images do not fit in a %u MB block", BLOCK_BYTES >> 20);
        set->block_used = offset + req.size;
        TIMED(s, OP_BIND_IMAGE, r = vkd.BindImageMemory(s->vk->device, image, block, offset));
    }
    BENCH_VK(r);
    return memory;
}

/* Records a copy of one subresource (in TRANSFER_SRC_OPTIMAL, written by
 * src_stage) to the start of the readback buffer, made visible to the host. */
static void copy_to_readback(struct scene *s, VkCommandBuffer cmd, VkImage image, uint32_t mip, uint32_t layer,
                             uint32_t size)
{
    VkBufferImageCopy region = {
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1 },
        .imageExtent = { size, size, 1 },
    };
    vkd.CmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s->readback.buffer, 1, &region);
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
}

static VkImageMemoryBarrier2 layout_barrier(VkImage image, VkImageAspectFlags aspect, uint32_t mip, uint32_t mips,
                                            uint32_t layer, VkImageLayout old_layout, VkImageLayout new_layout,
                                            VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                            VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access)
{
    VkImageMemoryBarrier2 b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = { aspect, mip, mips, layer, 1 },
    };
    return b;
}

static void tex_frame(struct scene *s, const struct bench_frame *f, struct ring_set *set, uint32_t ring)
{
    static const uint32_t sizes[8] = { 128, 256, 128, 512, 256, 1024, 128, 256 };
    VkImageMemoryBarrier2 barriers[MAX_PER_FRAME];
    VkDeviceSize staging = (VkDeviceSize)ring * STAGING_SLOT;
    int i, first = set->n, n = s->cfg->per_frame;

    for (i = 0; i < n; i++) {
        struct res *r = &set->r[set->n++];
        uint32_t size = sizes[(i + f->index) % 8], mips = mip_count(size);
        r->image = create_image(s, s->tex_format, size, size, mips, 1,
                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        r->memory = bind_image(s, set, s->blocks[ring], r->image, s->cfg->dedicated);
        r->view = create_view(s, r->image, s->tex_format, VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0);
        barriers[i] = layout_barrier(r->image, VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
                                     VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    }
    bench_image_barriers(f->cmd, (uint32_t)n, barriers);
    for (i = 0; i < n; i++) {
        struct res *r = &set->r[first + i];
        uint32_t size = sizes[(i + f->index) % 8], mips = mip_count(size), m;
        VkBufferImageCopy regions[16];
        memset(regions, 0, sizeof(regions));
        for (m = 0; m < mips; m++) {
            uint32_t w = size >> m ? size >> m : 1;
            regions[m].bufferOffset = staging;
            regions[m].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            regions[m].imageSubresource.mipLevel = m;
            regions[m].imageSubresource.layerCount = 1;
            regions[m].imageExtent.width = w;
            regions[m].imageExtent.height = w;
            regions[m].imageExtent.depth = 1;
            if (f->verify && i == 0 && m == 0) {
                s->check_offset = staging;
                s->check_size = size;
            }
            staging = align_up(staging + mip_bytes(s, size, m), 16);
        }
        vkd.CmdCopyBufferToImage(f->cmd, s->staging.buffer, r->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mips,
                                 regions);
        /* The check reads the first texture back instead of sampling it. */
        barriers[i] = layout_barrier(r->image, VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     f->verify && i == 0 ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                                         : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
                                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
    }
    bench_image_barriers(f->cmd, (uint32_t)n, barriers);
    if (f->verify)
        copy_to_readback(s, f->cmd, set->r[first].image, 0, 0, s->check_size);
}

static void buf_frame(struct scene *s, const struct bench_frame *f, struct ring_set *set, uint32_t ring)
{
    const VkDevice dev = s->vk->device;
    const uint32_t *kb = s->cfg->buffer_kb;
    VkBufferCreateInfo info = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    int i, n = s->cfg->per_frame / 2;

    for (i = 0; i < n; i++) {
        struct res *src = &set->r[set->n++], *dst = &set->r[set->n++];
        VkDeviceSize size = (VkDeviceSize)(kb[0] + bench_rand(&s->rng) % (kb[1] - kb[0] + 1)) << 10;
        VkDeviceSize src_offset = 0, dst_offset = 0;
        VkMemoryRequirements req;
        VkDeviceMemory src_memory, dst_memory;
        VkBufferCopy copy = { 0, 0, size };
        uint8_t *map;
        VkResult r;

        /* Staging: host-visible, mapped and filled. */
        info.size = size;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        TIMED(s, OP_CREATE_BUFFER, r = vkd.CreateBuffer(dev, &info, NULL, &src->buffer));
        BENCH_VK(r);
        name_object(s, VK_OBJECT_TYPE_BUFFER, (uint64_t)(uintptr_t)src->buffer, "staging");
        buffer_reqs(s, src->buffer, &req);
        if (s->cfg->dedicated) {
            src_memory = src->memory = alloc_dedicated(s, &req, s->host_type, VK_NULL_HANDLE, src->buffer);
            TIMED(s, OP_MAP, r = vkd.MapMemory(dev, src_memory, 0, VK_WHOLE_SIZE, 0, (void **)&map));
            BENCH_VK(r);
        } else {
            check_type(&req, s->host_type);
            src_memory = s->blocks2[ring];
            src_offset = align_up(set->block2_used, req.alignment);
            set->block2_used = src_offset + req.size;
            map = (uint8_t *)s->block2_map[ring] + src_offset;
        }
        TIMED(s, OP_BIND_BUFFER, r = vkd.BindBufferMemory(dev, src->buffer, src_memory, src_offset));
        BENCH_VK(r);
        memset(map, (int)((f->index + (uint32_t)i) & 0xFF), (size_t)(size < 16384 ? size : 16384)); /* the data */

        /* Its device-local destination: a vertex, index or storage buffer. */
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        TIMED(s, OP_CREATE_BUFFER, r = vkd.CreateBuffer(dev, &info, NULL, &dst->buffer));
        BENCH_VK(r);
        name_object(s, VK_OBJECT_TYPE_BUFFER, (uint64_t)(uintptr_t)dst->buffer, "vertices");
        buffer_reqs(s, dst->buffer, &req);
        if (s->cfg->dedicated) {
            dst_memory = dst->memory = alloc_dedicated(s, &req, s->device_type, VK_NULL_HANDLE, dst->buffer);
        } else {
            check_type(&req, s->device_type);
            dst_memory = s->blocks[ring];
            dst_offset = align_up(set->block_used, req.alignment);
            set->block_used = dst_offset + req.size;
        }
        TIMED(s, OP_BIND_BUFFER, r = vkd.BindBufferMemory(dev, dst->buffer, dst_memory, dst_offset));
        BENCH_VK(r);
        vkd.CmdCopyBuffer(f->cmd, src->buffer, dst->buffer, 1, &copy);
        if (f->verify && i == 0) {
            VkBufferCopy back = { 0, 0, 4096 };
            bench_memory_barrier(f->cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            vkd.CmdCopyBuffer(f->cmd, dst->buffer, s->readback.buffer, 1, &back);
            bench_memory_barrier(f->cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
            s->check_value = (f->index + (uint32_t)i) & 0xFF;
        }
    }
    bench_memory_barrier(f->cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
}

static void rt_frame(struct scene *s, const struct bench_frame *f, struct ring_set *set, uint32_t ring)
{
    static const VkFormat colors[5] = { VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM,
                                        VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_R16G16_SFLOAT,
                                        VK_FORMAT_R8_UNORM };
    float red = (float)(1 + f->index % 5) / 5.0f; /* exact in UNORM8: 51 per step */
    VkRenderingAttachmentInfo attachments[5], depth;
    VkImageMemoryBarrier2 barriers[6];
    VkRenderingInfo info;
    int i, first = set->n;

    memset(attachments, 0, sizeof(attachments));
    memset(&depth, 0, sizeof(depth));
    for (i = 0; i < 6; i++) {
        struct res *r = &set->r[set->n++];
        int is_depth = i == 5;
        VkFormat format = is_depth ? VK_FORMAT_D32_SFLOAT : colors[i];
        VkImageAspectFlags aspect = is_depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        if (is_depth)
            usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        else
            usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (i == 1) /* written by compute too, and read back by the check */
            usage |= VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        r->image = create_image(s, format, 1280, 720, 1, 1, usage);
        bind_image(s, set, s->blocks[ring], r->image, 0);
        r->view = create_view(s, r->image, format, aspect, 0, 1, 0);
        barriers[i] = layout_barrier(r->image, aspect, 0, 1, 0, VK_IMAGE_LAYOUT_UNDEFINED,
                                     is_depth ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                                              : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                                     is_depth ? VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
                                              : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     is_depth ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                                              : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        if (is_depth) {
            depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth.imageView = r->view;
            depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth.clearValue.depthStencil.depth = 1.0f;
        } else {
            attachments[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            attachments[i].imageView = r->view;
            attachments[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachments[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[i].clearValue.color.float32[0] = red;
            attachments[i].clearValue.color.float32[1] = 1.0f;
            attachments[i].clearValue.color.float32[3] = 1.0f;
        }
    }
    bench_image_barriers(f->cmd, 6, barriers);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    info.renderArea.extent.width = 1280;
    info.renderArea.extent.height = 720;
    info.layerCount = 1;
    info.colorAttachmentCount = 5;
    info.pColorAttachments = attachments;
    info.pDepthAttachment = &depth;
    vkd.CmdBeginRendering(f->cmd, &info);
    vkd.CmdEndRendering(f->cmd);
    bench_memory_barrier(f->cmd,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    if (f->verify) { /* a 720x720 corner of the RGBA8 target */
        barriers[0] = layout_barrier(set->r[first + 1].image, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                                     VK_ACCESS_2_TRANSFER_READ_BIT);
        bench_image_barriers(f->cmd, 1, barriers);
        copy_to_readback(s, f->cmd, set->r[first + 1].image, 0, 0, 720);
        s->check_size = 720;
        s->check_value = bench_rgba8((uint32_t)(red * 255.0f + 0.5f), 255, 0, 255);
    }
}

static void view_frame(struct scene *s, const struct bench_frame *f, struct ring_set *set)
{
    int i, first = set->n;
    for (i = 0; i < s->cfg->per_frame; i++) {
        struct res *r = &set->r[set->n++];
        uint32_t layer = (uint32_t)i % 6, mip = ((uint32_t)i / 6) % 12;
        r->view = create_view(s, s->big.image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, layer);
    }
    if (f->verify) { /* render through one of the views: it must reach its mip and layer */
        uint32_t layer = CHECK_VIEW % 6, mip = (CHECK_VIEW / 6) % 12, size = s->big.width >> mip;
        VkRenderingAttachmentInfo color = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = set->r[first + CHECK_VIEW].view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue.color.float32 = { 0.2f, 0.4f, 0.6f, 1.0f },
        };
        VkRenderingInfo info = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = { { 0, 0 }, { size, size } },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color,
        };
        VkImageMemoryBarrier2 b = layout_barrier(s->big.image, VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, layer,
                                                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                 VK_PIPELINE_STAGE_2_NONE, 0,
                                                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        bench_image_barriers(f->cmd, 1, &b);
        vkd.CmdBeginRendering(f->cmd, &info);
        vkd.CmdEndRendering(f->cmd);
        b = layout_barrier(s->big.image, VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, layer,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        bench_image_barriers(f->cmd, 1, &b);
        copy_to_readback(s, f->cmd, s->big.image, mip, layer, size);
        s->check_size = size;
        s->check_value = bench_rgba8(51, 102, 153, 255);
    }
}

static void sampler_frame(struct scene *s, struct ring_set *set)
{
    int i;
    for (i = 0; i < s->cfg->per_frame; i++) {
        struct res *r = &set->r[set->n++];
        /* Wrap or clamp, linear or nearest, and four LOD clamps (no LOD bias:
         * portability implementations may not support it). */
        VkSamplerCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = i & 2 ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
            .minFilter = i & 2 ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = i & 1 ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = i & 1 ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .minLod = (float)(i / 4),
            .maxLod = VK_LOD_CLAMP_NONE,
        };
        VkResult res;
        TIMED(s, OP_CREATE_SAMPLER, res = vkd.CreateSampler(s->vk->device, &info, NULL, &r->sampler));
        BENCH_VK(res);
    }
}

static void oneshot_frame(struct scene *s, const struct bench_frame *f, struct ring_set *set)
{
    const VkDevice dev = s->vk->device;
    int i;
    for (i = 0; i < s->cfg->per_frame; i++) {
        struct res *r = &set->r[set->n++], *smp = &set->r[set->n++];
        uint32_t size = 256, mips = mip_count(size), m;
        VkDeviceSize bytes = 0, offset = 0;
        VkBufferCreateInfo buffer_info = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                           .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                           .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
        VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                           .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                                .pCommandBuffers = &s->oneshot_cmd };
        VkSamplerCreateInfo sampler_info = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                             .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
                                             .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR, .maxLod = (float)mips };
        int check = f->verify && i == 0;
        VkBufferImageCopy regions[16];
        VkImageMemoryBarrier2 b;
        VkMemoryRequirements req;
        VkBuffer staging;
        VkDeviceMemory staging_memory;
        void *map;
        VkResult res;
        double t0;

        for (m = 0; m < mips; m++)
            bytes = align_up(bytes + mip_bytes(s, size, m), 16);
        buffer_info.size = bytes;
        TIMED(s, OP_CREATE_BUFFER, res = vkd.CreateBuffer(dev, &buffer_info, NULL, &staging));
        BENCH_VK(res);
        buffer_reqs(s, staging, &req);
        staging_memory = alloc_dedicated(s, &req, s->host_type, VK_NULL_HANDLE, staging);
        TIMED(s, OP_MAP, res = vkd.MapMemory(dev, staging_memory, 0, VK_WHOLE_SIZE, 0, &map));
        BENCH_VK(res);
        memset(map, 0x5A, (size_t)bytes);
        TIMED(s, OP_BIND_BUFFER, res = vkd.BindBufferMemory(dev, staging, staging_memory, 0));
        BENCH_VK(res);
        r->image = create_image(s, s->tex_format, size, size, mips, 1,
                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        r->memory = bind_image(s, set, VK_NULL_HANDLE, r->image, 1);

        t0 = bench_now_us();
        BENCH_VK(vkd.ResetCommandPool(dev, s->oneshot_pool, 0));
        BENCH_VK(vkd.BeginCommandBuffer(s->oneshot_cmd, &begin));
        b = layout_barrier(r->image, VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        bench_image_barriers(s->oneshot_cmd, 1, &b);
        memset(regions, 0, sizeof(regions));
        for (m = 0; m < mips; m++) {
            uint32_t w = size >> m ? size >> m : 1;
            regions[m].bufferOffset = offset;
            regions[m].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            regions[m].imageSubresource.mipLevel = m;
            regions[m].imageSubresource.layerCount = 1;
            regions[m].imageExtent.width = regions[m].imageExtent.height = w;
            regions[m].imageExtent.depth = 1;
            offset = align_up(offset + mip_bytes(s, size, m), 16);
        }
        vkd.CmdCopyBufferToImage(s->oneshot_cmd, staging, r->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mips,
                                 regions);
        if (check) { /* read mip 0 back before it goes to the shaders */
            b = layout_barrier(r->image, VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                               VK_ACCESS_2_TRANSFER_READ_BIT);
            bench_image_barriers(s->oneshot_cmd, 1, &b);
            copy_to_readback(s, s->oneshot_cmd, r->image, 0, 0, size);
            s->check_size = size;
        }
        b = layout_barrier(r->image, VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0,
                           check ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        bench_image_barriers(s->oneshot_cmd, 1, &b);
        BENCH_VK(vkd.EndCommandBuffer(s->oneshot_cmd));
        BENCH_VK(vkd.ResetFences(dev, 1, &s->oneshot_fence));
        BENCH_VK(vkd.QueueSubmit(s->vk->queue, 1, &submit, s->oneshot_fence));
        bench_fence_wait(s->vk, s->oneshot_fence);
        if (s->measuring) {
            s->op_us[OP_ONESHOT] += bench_now_us() - t0;
            s->op_n[OP_ONESHOT]++;
        }
        TIMED(s, OP_DESTROY_BUFFER, vkd.DestroyBuffer(dev, staging, NULL));
        TIMED(s, OP_FREE, vkd.FreeMemory(dev, staging_memory, NULL));
        r->view = create_view(s, r->image, s->tex_format, VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0);
        TIMED(s, OP_CREATE_SAMPLER, res = vkd.CreateSampler(dev, &sampler_info, NULL, &smp->sampler));
        BENCH_VK(res);
    }
}

static void buffer_view_frame(struct scene *s, const struct bench_frame *f, struct ring_set *set)
{
    int i;
    for (i = 0; i < s->cfg->per_frame; i++) {
        struct res *r = &set->r[set->n++];
        VkBufferViewCreateInfo info = { .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
                                        .buffer = s->texel.buffer,
                                        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                                        .offset = (VkDeviceSize)((f->index * 64u + (uint32_t)i) % 1024u) << 16,
                                        .range = 65536 };
        VkResult res;
        TIMED(s, OP_CREATE_BVIEW, res = vkd.CreateBufferView(s->vk->device, &info, NULL, &r->buffer_view));
        BENCH_VK(res);
    }
}

static void record(void *user, const struct bench_frame *f)
{
    struct scene *s = user;
    uint32_t ring = f->index % RING;
    struct ring_set *set = &s->sets[ring];
    VkDescriptorImageInfo images[2 * MAX_PER_FRAME];
    VkBufferView buffer_views[MAX_PER_FRAME];
    VkWriteDescriptorSet write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    uint32_t n = 0;
    int i;

    s->measuring = f->index >= s->first_measured;
    destroy_set(s, set); /* created RING frames ago: the GPU is done with it */
    switch (s->cfg->kind) {
    case K_TEXTURES:
        tex_frame(s, f, set, ring);
        break;
    case K_BUFFERS:
        buf_frame(s, f, set, ring);
        break;
    case K_TARGETS:
        rt_frame(s, f, set, ring);
        break;
    case K_VIEWS:
        view_frame(s, f, set);
        break;
    case K_SAMPLERS:
        sampler_frame(s, set);
        break;
    case K_ONESHOT:
        oneshot_frame(s, f, set);
        break;
    case K_BUFFER_VIEWS:
        buffer_view_frame(s, f, set);
        break;
    }
    if (!s->set_layout)
        return;
    for (i = 0; i < set->n; i++) {
        if (set->r[i].buffer_view)
            buffer_views[n++] = set->r[i].buffer_view;
        if (set->r[i].view) {
            images[n].sampler = s->sampler;
            images[n].imageView = set->r[i].view;
            images[n].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            n++;
        }
    }
    write.dstSet = s->desc_sets[ring];
    write.descriptorCount = n;
    write.descriptorType = s->cfg->kind == K_BUFFER_VIEWS ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
                                                          : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = images;
    write.pTexelBufferView = buffer_views;
    {
        double t0 = bench_now_us();
        vkd.UpdateDescriptorSets(s->vk->device, 1, &write, 0, NULL);
        if (s->measuring) {
            s->op_us[OP_WRITE_DESC] += bench_now_us() - t0;
            s->op_n[OP_WRITE_DESC] += n;
        }
    }
}

/* Compares what the first frame read back. */
static int verify(struct scene *s)
{
    const uint8_t *got = s->readback.map;
    size_t i, n;
    switch (s->cfg->kind) {
    case K_TEXTURES: {
        const uint8_t *want = (const uint8_t *)s->staging.map + s->check_offset;
        if (memcmp(got, want, (size_t)mip_bytes(s, s->check_size, 0)) != 0) {
            bench_error(s->cfg->name, "texture readback differs from the upload");
            return 0;
        }
        return 1;
    }
    case K_ONESHOT:
        n = (size_t)mip_bytes(s, s->check_size, 0);
        for (i = 0; i < n; i++)
            if (got[i] != 0x5A) {
                bench_error(s->cfg->name, "texture byte %zu: got 0x%02x, want 0x5a", i, got[i]);
                return 0;
            }
        return 1;
    case K_BUFFERS:
        for (i = 0; i < 4096; i++)
            if (got[i] != s->check_value) {
                bench_error(s->cfg->name, "buffer byte %zu: got %u, want %u", i, got[i], s->check_value);
                return 0;
            }
        return 1;
    case K_TARGETS:
    case K_VIEWS: {
        const uint32_t *texels = s->readback.map;
        n = (size_t)s->check_size * s->check_size;
        for (i = 0; i < n; i++)
            if (texels[i] != s->check_value) {
                bench_error(s->cfg->name, "texel %zu: got 0x%08x, want 0x%08x", i, texels[i], s->check_value);
                return 0;
            }
        return 1;
    }
    default: /* samplers and buffer views: nothing uses them */
        return 1;
    }
}

static void run(struct bench_vk *vk, const struct config *cfg)
{
    struct scene *s = bench_alloc(sizeof(*s));
    struct bench_loop loop;
    struct bench_json j;
    VkMemoryRequirements req;
    VkFormatProperties props;
    int i, frames = bench_frames(cfg->frames), ok = 1, first;
    double total = 0.0;

    s->vk = vk;
    s->cfg = cfg;
    s->rng = 12345;
    s->first_measured = UINT32_MAX;
    vkGetPhysicalDeviceFormatProperties(vk->physical_device, VK_FORMAT_BC7_UNORM_BLOCK, &props);
    if ((props.optimalTilingFeatures & (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT)) ==
        (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT)) {
        s->tex_format = VK_FORMAT_BC7_UNORM_BLOCK;
        s->tex_block_bytes = 16;
    } else {
        s->tex_format = VK_FORMAT_R8G8B8A8_UNORM;
    }
    s->host_type = bench_memory_type(vk, ~0u, BENCH_HOST, 0, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
    s->device_type = bench_memory_type(vk, ~0u, BENCH_DEVICE_LOCAL, 0,
                                       VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (s->device_type == UINT32_MAX)
        s->device_type = bench_memory_type(vk, ~0u, BENCH_DEVICE_LOCAL, 0, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
    if (s->host_type == UINT32_MAX || s->device_type == UINT32_MAX)
        bench_fatal("no host-visible or no device-local memory type");

    /* What the kind streams from or into. */
    if (cfg->kind == K_TEXTURES) {
        uint8_t *p;
        size_t k;
        bench_buffer_create(vk, &s->staging, (VkDeviceSize)RING * STAGING_SLOT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            BENCH_HOST, 0);
        p = s->staging.map;
        for (k = 0; k < (size_t)RING * STAGING_SLOT; k++)
            p[k] = (uint8_t)(k * 2654435761u >> 13);
    }
    bench_buffer_create(vk, &s->readback, READBACK_BYTES, VK_BUFFER_USAGE_TRANSFER_DST_BIT, BENCH_HOST,
                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (!cfg->dedicated && (cfg->kind == K_TEXTURES || cfg->kind == K_BUFFERS || cfg->kind == K_TARGETS)) {
        for (i = 0; i < RING; i++) {
            s->blocks[i] = alloc_block(s, BLOCK_BYTES, s->device_type);
            if (cfg->kind == K_BUFFERS) {
                s->blocks2[i] = alloc_block(s, BLOCK_BYTES, s->host_type);
                BENCH_VK(vkd.MapMemory(vk->device, s->blocks2[i], 0, VK_WHOLE_SIZE, 0, &s->block2_map[i]));
            }
        }
    }
    if (cfg->kind == K_VIEWS) { /* 2048x2048, 12 mips, 6 layers */
        s->big.image = create_image(s, VK_FORMAT_R8G8B8A8_UNORM, 2048, 2048, 12, 6,
                                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        s->big.width = 2048;
        vkd.GetImageMemoryRequirements(vk->device, s->big.image, &req);
        check_type(&req, s->device_type);
        s->big.memory = alloc_block(s, req.size, s->device_type);
        BENCH_VK(vkd.BindImageMemory(vk->device, s->big.image, s->big.memory, 0));
    }
    if (cfg->working_set) { /* 64x64 RGBA8 textures with full mips, in one block */
        VkDeviceSize used = 0, stride;
        VkImage probe = create_image(s, VK_FORMAT_R8G8B8A8_UNORM, 64, 64, 7, 1,
                                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        vkd.GetImageMemoryRequirements(vk->device, probe, &req);
        vkd.DestroyImage(vk->device, probe, NULL);
        check_type(&req, s->device_type);
        stride = align_up(req.size, req.alignment);
        s->ws_memory = alloc_block(s, stride * (VkDeviceSize)cfg->working_set, s->device_type);
        s->ws_images = bench_alloc(sizeof(VkImage) * (size_t)cfg->working_set);
        s->ws_views = bench_alloc(sizeof(VkImageView) * (size_t)cfg->working_set);
        for (i = 0; i < cfg->working_set; i++) {
            s->ws_images[i] = create_image(s, VK_FORMAT_R8G8B8A8_UNORM, 64, 64, 7, 1,
                                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            BENCH_VK(vkd.BindImageMemory(vk->device, s->ws_images[i], s->ws_memory, used));
            used += stride;
            s->ws_views[i] = create_view(s, s->ws_images[i], VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                         7, 0);
        }
        s->ws_count = cfg->working_set;
    }
    if (cfg->kind == K_ONESHOT) {
        VkCommandPoolCreateInfo pool_info = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                              .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                              .queueFamilyIndex = vk->queue_family };
        VkCommandBufferAllocateInfo alloc = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                              .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
        VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        BENCH_VK(vkd.CreateCommandPool(vk->device, &pool_info, NULL, &s->oneshot_pool));
        alloc.commandPool = s->oneshot_pool;
        BENCH_VK(vkd.AllocateCommandBuffers(vk->device, &alloc, &s->oneshot_cmd));
        BENCH_VK(vkd.CreateFence(vk->device, &fence_info, NULL, &s->oneshot_fence));
    }
    if (cfg->kind == K_BUFFER_VIEWS)
        bench_buffer_create(vk, &s->texel, 64u << 20,
                            VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT, 0,
                            BENCH_DEVICE_LOCAL);
    if (cfg->kind != K_BUFFERS && cfg->kind != K_SAMPLERS) { /* the descriptor arrays */
        VkDescriptorType type = cfg->kind == K_BUFFER_VIEWS ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
                                                            : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        VkDescriptorSetLayoutBinding binding = { 0, type, MAX_PER_FRAME, VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
        VkDescriptorPoolSize pool_size = { type, MAX_PER_FRAME * RING };
        VkDescriptorPoolCreateInfo pool_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                 .maxSets = RING, .poolSizeCount = 1, .pPoolSizes = &pool_size };
        VkDescriptorSetLayout layouts[RING];
        VkDescriptorSetAllocateInfo alloc = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                              .descriptorSetCount = RING, .pSetLayouts = layouts };
        VkSamplerCreateInfo sampler_info = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                             .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
                                             .maxLod = VK_LOD_CLAMP_NONE };
        s->set_layout = bench_set_layout(vk, 1, &binding, 0);
        BENCH_VK(vkd.CreateDescriptorPool(vk->device, &pool_info, NULL, &s->desc_pool));
        for (i = 0; i < RING; i++)
            layouts[i] = s->set_layout;
        alloc.descriptorPool = s->desc_pool;
        BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &alloc, s->desc_sets));
        BENCH_VK(vkd.CreateSampler(vk->device, &sampler_info, NULL, &s->sampler));
    }

    /* The checked first frame, warmup, then the measured frames. */
    bench_loop_init(&loop, vk, frames);
    bench_loop_verify(&loop, record, s);
    ok = verify(s);
    if (ok) {
        s->first_measured = loop.next_index + 20;
        bench_loop_run(&loop, record, s, 20, frames);
        s->measuring = 0;

        bench_json_begin(&j, cfg->name);
        bench_json_int(&j, "per_frame", cfg->per_frame);
        bench_json_int(&j, "working_set", cfg->working_set);
        bench_loop_report(&loop, &j);
        bench_json_printf(&j, ",\"ns_per_op\":{");
        for (i = 0, first = 1; i < OP_COUNT; i++) {
            if (!s->op_n[i])
                continue;
            bench_json_printf(&j, "%s\"%s\":%.1f", first ? "" : ",", op_names[i],
                              s->op_us[i] * 1000.0 / (double)s->op_n[i]);
            total += s->op_us[i];
            first = 0;
        }
        bench_json_printf(&j, "}");
        bench_json_num(&j, "resource_us_per_frame", total / frames);
        bench_json_end(&j);
    }

    bench_loop_destroy(&loop);
    for (i = 0; i < RING; i++)
        destroy_set(s, &s->sets[i]);
    for (i = 0; i < s->ws_count; i++) {
        vkd.DestroyImageView(vk->device, s->ws_views[i], NULL);
        vkd.DestroyImage(vk->device, s->ws_images[i], NULL);
    }
    free(s->ws_images);
    free(s->ws_views);
    if (s->ws_memory)
        vkd.FreeMemory(vk->device, s->ws_memory, NULL);
    if (s->big.image) {
        vkd.DestroyImage(vk->device, s->big.image, NULL);
        vkd.FreeMemory(vk->device, s->big.memory, NULL);
    }
    for (i = 0; i < RING; i++) {
        if (s->blocks[i])
            vkd.FreeMemory(vk->device, s->blocks[i], NULL);
        if (s->blocks2[i])
            vkd.FreeMemory(vk->device, s->blocks2[i], NULL);
    }
    bench_buffer_destroy(vk, &s->staging);
    bench_buffer_destroy(vk, &s->readback);
    bench_buffer_destroy(vk, &s->texel);
    if (s->oneshot_pool) {
        vkd.DestroyFence(vk->device, s->oneshot_fence, NULL);
        vkd.DestroyCommandPool(vk->device, s->oneshot_pool, NULL);
    }
    if (s->set_layout) {
        vkd.DestroyDescriptorPool(vk->device, s->desc_pool, NULL);
        vkd.DestroyDescriptorSetLayout(vk->device, s->set_layout, NULL);
        vkd.DestroySampler(vk->device, s->sampler, NULL);
    }
    free(s);
}

int main(void)
{
    struct bench_vk vk;
    struct bench_vk_wants wants;
    size_t i;

    bench_begin("resource_churn");
    memset(&wants, 0, sizeof(wants));
    wants.instance_extensions[0] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    bench_vk_init_with(&vk, BENCH_NEED_SYNC2 | BENCH_NEED_DYNAMIC_RENDERING, &wants);
    debug_names = bench_vk_extension(&vk, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) && vkd.SetDebugUtilsObjectNameEXT;
    for (i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
        if (!bench_selected(configs[i].name))
            continue;
        bench_config = configs[i].name;
        if (configs[i].named && !debug_names)
            bench_skip(configs[i].name, "VK_EXT_debug_utils not supported");
        else
            run(&vk, &configs[i]);
        bench_config = NULL;
    }
    bench_vk_destroy(&vk);
    return bench_exit_code();
}
