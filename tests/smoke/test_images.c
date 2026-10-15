/* Formats, images, views, buffer views, samplers, image transfer commands
 * and host image copy (sections 7, 11, 14 and 15), checked texel by texel.
 * Sampling is checked with a Metal kernel that reads the views and samplers
 * the way descriptors will (by MTLResourceID in an argument table). */
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <math.h>
#include <string.h>

#include "../pvk_test.h"

static VkInstance instance;
static VkPhysicalDevice pd;
static VkDevice device;
static VkQueue queue;
static VkCommandPool pool;
static VkFence fence;
static uint32_t private_type = UINT32_MAX; /* device-local only */
static uint32_t cached_type = UINT32_MAX;  /* host-visible, not coherent */
static uint32_t shared_type = UINT32_MAX;  /* host-visible and coherent */
static uint32_t lazy_type = UINT32_MAX;    /* lazily allocated */

struct buf {
    VkBuffer buffer;
    VkDeviceMemory memory;
    unsigned char *data;
    VkDeviceSize size;
};

struct img {
    VkImage image;
    VkDeviceMemory memory;
    VkImageCreateInfo info;
};

static void setup(void)
{
    VkPhysicalDeviceMemoryProperties mem;
    VkInstanceCreateInfo instance_info;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo device_info;
    VkCommandPoolCreateInfo pool_info;
    VkFenceCreateInfo fence_info;
    float priority = 1.0f;
    uint32_t count = 1, i;

    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    REQUIRE_VK(vkCreateInstance(&instance_info, NULL, &instance));
    REQUIRE_VK(vkEnumeratePhysicalDevices(instance, &count, &pd));
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    REQUIRE_VK(vkCreateDevice(pd, &device_info, NULL, &device));
    vkGetDeviceQueue(device, 0, 0, &queue);
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    for (i = 0; i < mem.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mem.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) && lazy_type == UINT32_MAX)
            lazy_type = i;
        else if ((f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) && shared_type == UINT32_MAX)
            shared_type = i;
        else if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && cached_type == UINT32_MAX)
            cached_type = i;
        else if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && private_type == UINT32_MAX)
            private_type = i;
    }
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    REQUIRE_VK(vkCreateCommandPool(device, &pool_info, NULL, &pool));
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    REQUIRE_VK(vkCreateFence(device, &fence_info, NULL, &fence));
}

static void teardown(void)
{
    vkDestroyFence(device, fence, NULL);
    vkDestroyCommandPool(device, pool, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
}

static struct buf make_buf(VkDeviceSize size, uint32_t type, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo info;
    VkMemoryAllocateInfo alloc;
    VkMemoryRequirements req;
    struct buf b;
    void *p = NULL;

    memset(&b, 0, sizeof(b));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage;
    if (vkCreateBuffer(device, &info, NULL, &b.buffer) != VK_SUCCESS)
        return b;
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    if (vkAllocateMemory(device, &alloc, NULL, &b.memory) != VK_SUCCESS)
        return b;
    vkBindBufferMemory(device, b.buffer, b.memory, 0);
    if (type != private_type && vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS) {
        b.data = p;
        memset(b.data, 0, (size_t)size);
    }
    b.size = size;
    return b;
}

static void free_buf(struct buf *b)
{
    vkDestroyBuffer(device, b->buffer, NULL);
    vkFreeMemory(device, b->memory, NULL);
    memset(b, 0, sizeof(*b));
}

static VkImageCreateInfo image_info(VkImageType type, VkFormat format, uint32_t w, uint32_t h, uint32_t d,
                                    uint32_t levels, uint32_t layers, VkImageUsageFlags usage)
{
    VkImageCreateInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = type;
    info.format = format;
    info.extent.width = w;
    info.extent.height = h;
    info.extent.depth = d;
    info.mipLevels = levels;
    info.arrayLayers = layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    return info;
}

/* Creates and binds an image in memory of the given type (UINT32_MAX: the
 * first type it allows). */
static struct img make_image_in(const VkImageCreateInfo *info, uint32_t type)
{
    VkMemoryAllocateInfo alloc;
    VkMemoryRequirements req;
    struct img m;
    uint32_t i;

    memset(&m, 0, sizeof(m));
    m.info = *info;
    if (vkCreateImage(device, info, NULL, &m.image) != VK_SUCCESS)
        return m;
    vkGetImageMemoryRequirements(device, m.image, &req);
    if (type == UINT32_MAX)
        for (i = 0; i < 32 && type == UINT32_MAX; i++)
            if (req.memoryTypeBits & (1u << i))
                type = i;
    if (!(req.memoryTypeBits & (1u << type))) {
        PVK_TEST_FAIL("memory type %u not allowed (bits %x)", type, req.memoryTypeBits);
        return m;
    }
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    if (vkAllocateMemory(device, &alloc, NULL, &m.memory) != VK_SUCCESS)
        return m;
    if (vkBindImageMemory(device, m.image, m.memory, 0) != VK_SUCCESS)
        PVK_TEST_FAIL("vkBindImageMemory failed");
    return m;
}

static struct img make_image(const VkImageCreateInfo *info)
{
    return make_image_in(info, private_type);
}

static void free_image(struct img *m)
{
    vkDestroyImage(device, m->image, NULL);
    vkFreeMemory(device, m->memory, NULL);
    memset(m, 0, sizeof(*m));
}

static id texture_of(VkImage image)
{
    return ((struct pvk_image *)image)->mtl_texture;
}

static VkCommandBuffer begin(void)
{
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferBeginInfo info;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &alloc, &cb) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &info);
    return cb;
}

static VkResult finish(VkCommandBuffer cb)
{
    VkSubmitInfo submit;
    VkResult r = vkEndCommandBuffer(cb);
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    vkResetFences(device, 1, &fence);
    if (r == VK_SUCCESS)
        r = vkQueueSubmit(queue, 1, &submit, fence);
    if (r == VK_SUCCESS)
        r = vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ull);
    vkFreeCommandBuffers(device, pool, 1, &cb);
    return r;
}

/* Everything before against everything after, as an image barrier with a
 * layout change (which has nothing to do on Metal). */
static void barrier(VkCommandBuffer cb, VkImage image)
{
    VkImageMemoryBarrier2 ib;
    VkMemoryBarrier2 mb;
    VkDependencyInfo dep;
    memset(&ib, 0, sizeof(ib));
    ib.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    ib.srcStageMask = ib.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    ib.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    ib.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = image;
    ib.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ib.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    ib.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_HOST_READ_BIT;
    memset(&dep, 0, sizeof(dep));
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    dep.imageMemoryBarrierCount = image ? 1 : 0;
    dep.pImageMemoryBarriers = &ib;
    vkCmdPipelineBarrier2(cb, &dep);
}

static VkBufferImageCopy region(VkImageAspectFlags aspect, uint32_t level, uint32_t layer, uint32_t layers,
                                int32_t x, int32_t y, int32_t z, uint32_t w, uint32_t h, uint32_t d)
{
    VkBufferImageCopy r;
    memset(&r, 0, sizeof(r));
    r.imageSubresource.aspectMask = aspect;
    r.imageSubresource.mipLevel = level;
    r.imageSubresource.baseArrayLayer = layer;
    r.imageSubresource.layerCount = layers;
    r.imageOffset.x = x;
    r.imageOffset.y = y;
    r.imageOffset.z = z;
    r.imageExtent.width = w;
    r.imageExtent.height = h;
    r.imageExtent.depth = d;
    return r;
}

static void fill_pattern(unsigned char *p, size_t n, unsigned seed)
{
    size_t i;
    for (i = 0; i < n; i++)
        p[i] = (unsigned char)((i * 131u + seed * 7u + (i >> 8) * 17u) & 0xff);
}

static int ct_is_pow2_(VkDeviceSize x)
{
    return x && !(x & (x - 1));
}

/* Metal object properties, for checking what was made. */
static unsigned long msg_ulong(id object, const char *selector)
{
    return PVK_MSG(unsigned long, (id, SEL))(object, sel_registerName(selector));
}

/* ---- Formats ---- */

static VkFormatFeatureFlags2 features3(VkFormat format, int which) /* 0 optimal, 1 linear, 2 buffer */
{
    VkFormatProperties3 p3;
    VkFormatProperties2 p2;
    memset(&p3, 0, sizeof(p3));
    p3.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3;
    memset(&p2, 0, sizeof(p2));
    p2.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
    p2.pNext = &p3;
    vkGetPhysicalDeviceFormatProperties2(pd, format, &p2);
    return which == 0 ? p3.optimalTilingFeatures : which == 1 ? p3.linearTilingFeatures : p3.bufferFeatures;
}

/* Spot checks of the table; the conformance suite checks the required
 * features of every format. */
static void test_format_properties(void)
{
    VkFormatFeatureFlags2 f;
    VkFormatProperties p;

    f = features3(VK_FORMAT_R8G8B8A8_UNORM, 0);
    CHECK((f & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT) && (f & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT) &&
          (f & VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT) && (f & VK_FORMAT_FEATURE_2_BLIT_DST_BIT) &&
          (f & VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT) && (f & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT));
    CHECK(features3(VK_FORMAT_R8G8B8A8_UNORM, 1) & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT);
    CHECK(features3(VK_FORMAT_R8G8B8A8_UNORM, 2) & VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT);
    /* sRGB: no storage */
    f = features3(VK_FORMAT_R8G8B8A8_SRGB, 0);
    CHECK((f & VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT) && !(f & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT));
    /* swizzled: sampled and transfers only */
    f = features3(VK_FORMAT_B4G4R4A4_UNORM_PACK16, 0);
    CHECK((f & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT) && !(f & VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT));
    /* integer: no filtering, storage atomics on R32 */
    f = features3(VK_FORMAT_R32_UINT, 0);
    CHECK((f & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT) && !(f & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT));
    CHECK(features3(VK_FORMAT_R32_UINT, 2) & VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT);
    /* depth */
    f = features3(VK_FORMAT_D32_SFLOAT_S8_UINT, 0);
    CHECK((f & VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT) && (f & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT));
    CHECK(features3(VK_FORMAT_D32_SFLOAT_S8_UINT, 1) == 0 && features3(VK_FORMAT_D32_SFLOAT_S8_UINT, 2) == 0);
    CHECK(features3(VK_FORMAT_D24_UNORM_S8_UINT, 0) == 0); /* none on Apple silicon */
    /* compressed */
    CHECK(features3(VK_FORMAT_ASTC_4x4_UNORM_BLOCK, 0) & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
    CHECK(features3(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 0) & VK_FORMAT_FEATURE_2_BLIT_SRC_BIT);
    CHECK(features3(VK_FORMAT_ASTC_4x4_UNORM_BLOCK, 1) == 0);
    /* vertex-only */
    CHECK(features3(VK_FORMAT_R32G32B32_SFLOAT, 0) == 0 && features3(VK_FORMAT_R32G32B32_SFLOAT, 2) ==
                                                                  VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT);
    CHECK(features3(VK_FORMAT_UNDEFINED, 0) == 0 && features3(VK_FORMAT_R64_UINT, 2) == 0);
    /* vkGetPhysicalDeviceFormatProperties: the 32-bit part */
    vkGetPhysicalDeviceFormatProperties(pd, VK_FORMAT_R8G8B8A8_UNORM, &p);
    CHECK(p.optimalTilingFeatures == (VkFormatFeatureFlags)(features3(VK_FORMAT_R8G8B8A8_UNORM, 0) & 0x7FFFFFFFu));
    /* BC follows supportsBCTextureCompression (and Apple9) */
    CHECK(((features3(VK_FORMAT_BC7_UNORM_BLOCK, 0) & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT) != 0) ==
          ((struct pvk_physical_device *)pd)->bc_formats);
}

/* Every image format's Metal pixel format can be made in every image type
 * vkGetPhysicalDeviceImageFormatProperties accepts, with every usage its
 * features allow (MTL_DEBUG_LAYER=1 validates the descriptors). */
static void test_all_formats_create(void)
{
    static const VkImageType types[] = { VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_3D };
    static const VkImageUsageFlags usages[] = {
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_IMAGE_USAGE_HOST_TRANSFER_BIT,
    };
    unsigned created = 0, f, t, u;
    for (f = 1; f < 1000470002u; f = f == VK_FORMAT_ASTC_12x12_SRGB_BLOCK ? 1000066000u
                                     : f == 1000066013u                     ? 1000340000u
                                     : f == 1000340001u                     ? 1000470000u
                                                                            : f + 1) {
        for (t = 0; t < PVK_ARRAY_SIZE(types); t++) {
            VkImageUsageFlags usage = 0;
            VkImageFormatProperties props;
            VkImageCreateInfo info;
            struct img m;
            for (u = 0; u < PVK_ARRAY_SIZE(usages); u++)
                if (vkGetPhysicalDeviceImageFormatProperties(pd, (VkFormat)f, types[t], VK_IMAGE_TILING_OPTIMAL,
                                                             usages[u], 0, &props) == VK_SUCCESS)
                    usage |= usages[u];
            if (!usage)
                continue;
            CHECK(vkGetPhysicalDeviceImageFormatProperties(pd, (VkFormat)f, types[t], VK_IMAGE_TILING_OPTIMAL, usage, 0,
                                                           &props) == VK_SUCCESS);
            info = image_info(types[t], (VkFormat)f, 8, types[t] == VK_IMAGE_TYPE_1D ? 1 : 8,
                              types[t] == VK_IMAGE_TYPE_3D ? 4 : 1, props.maxMipLevels < 2 ? 1 : 2, 1, usage);
            m = make_image_in(&info, (usage & VK_IMAGE_USAGE_HOST_TRANSFER_BIT) ? shared_type : private_type);
            if (!texture_of(m.image))
                PVK_TEST_FAIL("format %u type %u usage %x: no texture", f, types[t], usage);
            else
                created++;
            free_image(&m);
        }
    }
    CHECK(created > 200);
}

static void test_image_format_properties(void)
{
    VkPhysicalDeviceImageFormatInfo2 info;
    VkImageFormatProperties2 props;
    VkHostImageCopyDevicePerformanceQuery perf;
    VkImageFormatProperties p;

    CHECK(vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 0,
                                                   &p) == VK_SUCCESS);
    CHECK(p.maxExtent.width >= 16384 && p.maxMipLevels == 15 && p.maxArrayLayers >= 2048 &&
          (p.sampleCounts & VK_SAMPLE_COUNT_4_BIT) && p.maxResourceSize >= (1ull << 31));
    /* storage images are single-sample; integer formats too where sampled */
    CHECK(vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                   VK_IMAGE_USAGE_STORAGE_BIT, 0, &p) == VK_SUCCESS &&
          p.sampleCounts == VK_SAMPLE_COUNT_1_BIT);
    CHECK(vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_R32_UINT, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                   VK_IMAGE_USAGE_SAMPLED_BIT, 0, &p) == VK_SUCCESS &&
          p.sampleCounts == VK_SAMPLE_COUNT_1_BIT);
    /* 1D images get mipmaps (2D textures underneath); 3D has one layer */
    CHECK(vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_R8_UNORM, VK_IMAGE_TYPE_1D, VK_IMAGE_TILING_OPTIMAL,
                                                   VK_IMAGE_USAGE_SAMPLED_BIT, 0, &p) == VK_SUCCESS &&
          p.maxMipLevels == 15 && p.maxExtent.height == 1);
    CHECK(vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_R8_UNORM, VK_IMAGE_TYPE_3D, VK_IMAGE_TILING_OPTIMAL,
                                                   VK_IMAGE_USAGE_SAMPLED_BIT, 0, &p) == VK_SUCCESS &&
          p.maxArrayLayers == 1 && p.maxExtent.depth == 2048 && p.maxMipLevels == 12);
    /* not supported: depth 3D, ETC2 3D, linear depth, partially resident
     * combined depth/stencil (test_sparse.c), sparse linear, depth storage */
    CHECK(vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_D32_SFLOAT, VK_IMAGE_TYPE_3D, VK_IMAGE_TILING_OPTIMAL,
                                                   VK_IMAGE_USAGE_SAMPLED_BIT, 0, &p) == VK_ERROR_FORMAT_NOT_SUPPORTED);
    CHECK(vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, VK_IMAGE_TYPE_3D,
                                                   VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT, 0,
                                                   &p) == VK_ERROR_FORMAT_NOT_SUPPORTED);
    CHECK(vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_D32_SFLOAT, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_LINEAR,
                                                   VK_IMAGE_USAGE_SAMPLED_BIT, 0, &p) == VK_ERROR_FORMAT_NOT_SUPPORTED);
    CHECK(vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_TYPE_2D,
                                                   VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT,
                                                   VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
                                                   &p) == VK_ERROR_FORMAT_NOT_SUPPORTED);
    CHECK(vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_R8_UNORM, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_LINEAR,
                                                   VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_CREATE_SPARSE_BINDING_BIT,
                                                   &p) == VK_ERROR_FORMAT_NOT_SUPPORTED);
    CHECK(vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_D16_UNORM, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                   VK_IMAGE_USAGE_STORAGE_BIT, 0, &p) == VK_ERROR_FORMAT_NOT_SUPPORTED);
    /* linear: 2D, one level, one layer, one sample */
    CHECK(vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_LINEAR,
                                                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0,
                                                   &p) == VK_SUCCESS &&
          p.maxMipLevels == 1 && p.maxArrayLayers == 1 && p.sampleCounts == VK_SAMPLE_COUNT_1_BIT);
    /* HOST_TRANSFER changes nothing with SAMPLED; the performance query */
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
    info.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    info.type = VK_IMAGE_TYPE_2D;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_HOST_TRANSFER_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    memset(&perf, 0, sizeof(perf));
    perf.sType = VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY;
    memset(&props, 0, sizeof(props));
    props.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
    props.pNext = &perf;
    CHECK(vkGetPhysicalDeviceImageFormatProperties2(pd, &info, &props) == VK_SUCCESS);
    CHECK(props.imageFormatProperties.sampleCounts & VK_SAMPLE_COUNT_4_BIT);
    CHECK(!perf.identicalMemoryLayout);
}

/* ---- Creation, memory requirements, binding ---- */

static void test_create_kinds(void)
{
    struct {
        VkImageCreateInfo info;
        unsigned long mtl_type;
        uint32_t type;
    } k[10];
    unsigned i, n = 0;
    VkMemoryRequirements req, req2;
    VkMemoryRequirements2 dreq;
    VkDeviceImageMemoryRequirements dinfo;

    k[n].info = image_info(VK_IMAGE_TYPE_1D, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, 1, 7, 1, VK_IMAGE_USAGE_SAMPLED_BIT);
    k[n++].mtl_type = PVK_MTLTextureType2D;
    k[n].info = image_info(VK_IMAGE_TYPE_1D, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, 1, 1, 3, VK_IMAGE_USAGE_SAMPLED_BIT);
    k[n++].mtl_type = PVK_MTLTextureType2DArray;
    k[n].info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, 33, 17, 1, 6, 1,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    k[n++].mtl_type = PVK_MTLTextureType2D;
    k[n].info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8_UNORM, 16, 16, 1, 1, 5, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    k[n++].mtl_type = PVK_MTLTextureType2DArray;
    k[n].info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_SRGB, 32, 32, 1, 6, 12, VK_IMAGE_USAGE_SAMPLED_BIT);
    k[n].info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    k[n++].mtl_type = PVK_MTLTextureType2DArray;
    k[n].info = image_info(VK_IMAGE_TYPE_3D, VK_FORMAT_R32_SFLOAT, 16, 8, 4, 5, 1, VK_IMAGE_USAGE_SAMPLED_BIT);
    k[n++].mtl_type = PVK_MTLTextureType3D;
    k[n].info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 1, 1,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    k[n].info.samples = VK_SAMPLE_COUNT_4_BIT;
    k[n++].mtl_type = PVK_MTLTextureType2DMultisample;
    k[n].info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_D32_SFLOAT_S8_UINT, 16, 16, 1, 1, 2,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    k[n].info.samples = VK_SAMPLE_COUNT_4_BIT;
    k[n++].mtl_type = PVK_MTLTextureType2DMultisampleArray;
    k[n].info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 20, 20, 1, 5, 2, VK_IMAGE_USAGE_SAMPLED_BIT);
    k[n++].mtl_type = PVK_MTLTextureType2DArray;
    k[n].info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_ASTC_8x8_SRGB_BLOCK, 100, 60, 1, 7, 1, VK_IMAGE_USAGE_SAMPLED_BIT);
    k[n++].mtl_type = PVK_MTLTextureType2D;

    for (i = 0; i < n; i++) {
        uint32_t t;
        VkImage image;
        if (k[i].info.format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK && !((struct pvk_physical_device *)pd)->bc_formats)
            continue;
        REQUIRE_VK(vkCreateImage(device, &k[i].info, NULL, &image));
        vkGetImageMemoryRequirements(device, image, &req);
        CHECK(req.size > 0 && ct_is_pow2_(req.alignment));
        memset(&dinfo, 0, sizeof(dinfo));
        dinfo.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS;
        dinfo.pCreateInfo = &k[i].info;
        memset(&dreq, 0, sizeof(dreq));
        dreq.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
        vkGetDeviceImageMemoryRequirements(device, &dinfo, &dreq);
        CHECK(memcmp(&dreq.memoryRequirements, &req, sizeof(req)) == 0);
        CHECK(req.memoryTypeBits == ((1u << private_type) | (1u << cached_type) | (1u << shared_type)));
        vkDestroyImage(device, image, NULL);
        /* bound in each memory type it allows */
        for (t = 0; t < 32; t++) {
            struct img m;
            id tex;
            if (!(req.memoryTypeBits & (1u << t)))
                continue;
            m = make_image_in(&k[i].info, t);
            tex = texture_of(m.image);
            CHECK(tex != NULL);
            if (tex) {
                CHECK(msg_ulong(tex, "textureType") == k[i].mtl_type);
                CHECK(msg_ulong(tex, "mipmapLevelCount") == k[i].info.mipLevels);
                CHECK(msg_ulong(tex, "sampleCount") == (unsigned long)k[i].info.samples);
                CHECK(msg_ulong(tex, "storageMode") ==
                      (t == private_type ? PVK_MTLStorageModePrivate : PVK_MTLStorageModeShared));
                vkGetImageMemoryRequirements(device, m.image, &req2);
                CHECK(memcmp(&req, &req2, sizeof(req)) == 0);
            }
            free_image(&m);
        }
    }

    /* Two images in one allocation, at the second's alignment. */
    {
        VkImageCreateInfo info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, 1, 1,
                                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        VkMemoryAllocateInfo alloc;
        VkDeviceMemory mem;
        VkImage a, b;
        VkBindImageMemoryInfo binds[2];
        VkBindMemoryStatus status[2];
        VkResult results[2] = { VK_ERROR_UNKNOWN, VK_ERROR_UNKNOWN };
        REQUIRE_VK(vkCreateImage(device, &info, NULL, &a));
        REQUIRE_VK(vkCreateImage(device, &info, NULL, &b));
        vkGetImageMemoryRequirements(device, a, &req);
        memset(&alloc, 0, sizeof(alloc));
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = req.size * 2;
        alloc.memoryTypeIndex = private_type;
        REQUIRE_VK(vkAllocateMemory(device, &alloc, NULL, &mem));
        memset(binds, 0, sizeof(binds));
        memset(status, 0, sizeof(status));
        for (i = 0; i < 2; i++) {
            status[i].sType = VK_STRUCTURE_TYPE_BIND_MEMORY_STATUS;
            status[i].pResult = &results[i];
            binds[i].sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
            binds[i].pNext = &status[i];
            binds[i].image = i ? b : a;
            binds[i].memory = mem;
            binds[i].memoryOffset = i * req.size;
        }
        CHECK_VK(vkBindImageMemory2(device, 2, binds), VK_SUCCESS);
        CHECK(results[0] == VK_SUCCESS && results[1] == VK_SUCCESS);
        CHECK(texture_of(a) && texture_of(b));
        vkDestroyImage(device, a, NULL);
        vkDestroyImage(device, b, NULL);
        vkFreeMemory(device, mem, NULL);
    }
}

/* Resources > Resource Memory Association: linear and HOST_TRANSFER images
 * take host-visible memory, transient attachments may take lazily allocated
 * memory (a memoryless texture), nothing else may. */
static void test_memory_types(void)
{
    uint32_t host = (1u << cached_type) | (1u << shared_type), all = host | (1u << private_type);
    VkImageCreateInfo info;
    VkMemoryRequirements req;
    VkImage image;
    struct img m;
    VkDeviceSize committed = 1;

    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 1, 1, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    info.tiling = VK_IMAGE_TILING_LINEAR;
    REQUIRE_VK(vkCreateImage(device, &info, NULL, &image));
    vkGetImageMemoryRequirements(device, image, &req);
    CHECK(req.memoryTypeBits == host);
    vkDestroyImage(device, image, NULL);

    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 1, 1,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_HOST_TRANSFER_BIT);
    REQUIRE_VK(vkCreateImage(device, &info, NULL, &image));
    vkGetImageMemoryRequirements(device, image, &req);
    CHECK(req.memoryTypeBits == host);
    vkDestroyImage(device, image, NULL);

    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, 1, 1,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT);
    REQUIRE_VK(vkCreateImage(device, &info, NULL, &image));
    vkGetImageMemoryRequirements(device, image, &req);
    CHECK(req.memoryTypeBits == (all | (1u << lazy_type)));
    vkDestroyImage(device, image, NULL);
    m = make_image_in(&info, lazy_type);
    CHECK(texture_of(m.image) && msg_ulong(texture_of(m.image), "storageMode") == PVK_MTLStorageModeMemoryless);
    vkGetDeviceMemoryCommitment(device, m.memory, &committed);
    CHECK(committed == 0);
    {
        VkImageViewCreateInfo vi;
        VkImageView view;
        memset(&vi, 0, sizeof(vi));
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = m.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = info.format;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        CHECK_VK(vkCreateImageView(device, &vi, NULL, &view), VK_SUCCESS);
        vkDestroyImageView(device, view, NULL);
    }
    free_image(&m);
    /* the same transient image in ordinary memory is an ordinary texture */
    m = make_image_in(&info, private_type);
    CHECK(texture_of(m.image) && msg_ulong(texture_of(m.image), "storageMode") == PVK_MTLStorageModePrivate);
    free_image(&m);
}

/* ---- Buffer <-> image copies ---- */

/* Uploads `up` from a buffer of patterned bytes, reads the region back
 * tightly packed at another offset, and compares block row by block row.
 * block: bytes per texel block; bw, bh: block size in texels. */
static void roundtrip(const VkImageCreateInfo *info, uint32_t type, const VkBufferImageCopy *up, uint32_t block,
                      uint32_t bw, uint32_t bh, const char *what)
{
    struct img m = make_image_in(info, type);
    VkBufferImageCopy down = *up;
    uint32_t rl = up->bufferRowLength ? up->bufferRowLength : up->imageExtent.width;
    uint32_t ih = up->bufferImageHeight ? up->bufferImageHeight : up->imageExtent.height;
    uint32_t brow = (rl + bw - 1) / bw, bimg = (ih + bh - 1) / bh;
    uint32_t wb = (up->imageExtent.width + bw - 1) / bw, hb = (up->imageExtent.height + bh - 1) / bh;
    uint32_t layers = up->imageSubresource.layerCount, d = up->imageExtent.depth, slices = layers > d ? layers : d;
    VkDeviceSize in_size = up->bufferOffset + (VkDeviceSize)brow * bimg * slices * block;
    VkDeviceSize out_size = 64 + (VkDeviceSize)wb * hb * slices * block;
    struct buf src = make_buf(in_size, shared_type, 0), dst = make_buf(out_size, shared_type, 0);
    VkCommandBuffer cb;
    uint32_t z, y, bad = 0;

    if (!texture_of(m.image) || !src.data || !dst.data) {
        PVK_TEST_FAIL("%s: setup failed", what);
        goto out;
    }
    fill_pattern(src.data, (size_t)in_size, info->format);
    down.bufferOffset = 64;
    down.bufferRowLength = 0;
    down.bufferImageHeight = 0;
    cb = begin();
    vkCmdCopyBufferToImage(cb, src.buffer, m.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, up);
    barrier(cb, m.image);
    vkCmdCopyImageToBuffer(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.buffer, 1, &down);
    barrier(cb, VK_NULL_HANDLE);
    if (finish(cb) != VK_SUCCESS) {
        PVK_TEST_FAIL("%s: submit failed", what);
        goto out;
    }
    for (z = 0; z < slices; z++)
        for (y = 0; y < hb; y++)
            if (memcmp(dst.data + 64 + ((VkDeviceSize)z * hb + y) * wb * block,
                       src.data + up->bufferOffset + ((VkDeviceSize)z * bimg + y) * brow * block, (size_t)wb * block))
                bad++;
    if (bad)
        PVK_TEST_FAIL("%s: %u of %u block rows differ", what, bad, slices * hb);
out:
    free_buf(&src);
    free_buf(&dst);
    free_image(&m);
}

static void test_copy_buffer_image(void)
{
    VkImageCreateInfo info;
    VkBufferImageCopy r;
    uint32_t types[3], t;
    types[0] = private_type;
    types[1] = cached_type;
    types[2] = shared_type;

    /* 2D, a sub-rectangle of a mip level of two layers, padded rows and images */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 37, 21, 1, 4, 3,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    for (t = 0; t < 3; t++) {
        r = region(VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 2, 3, 2, 0, 13, 7, 1);
        r.bufferOffset = 12;
        r.bufferRowLength = 20;
        r.bufferImageHeight = 9;
        roundtrip(&info, types[t], &r, 4, 1, 1, "2D array level 1");
    }
    /* whole level 0, tightly packed, 16-bit float */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, 64, 32, 1, 1, 1,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 64, 32, 1);
    roundtrip(&info, private_type, &r, 8, 1, 1, "2D RGBA16F");
    /* packed float formats keep every bit (lossless compression would write
     * their values in canonical form) */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 64, 64, 1, 1, 1,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 64, 64, 1);
    roundtrip(&info, private_type, &r, 4, 1, 1, "2D RGB9E5");
    info.format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    roundtrip(&info, private_type, &r, 4, 1, 1, "2D B10G11R11");
    /* 3D: slices of a level */
    info = image_info(VK_IMAGE_TYPE_3D, VK_FORMAT_R32_SFLOAT, 16, 12, 8, 3, 1,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 2, 1, 3, 10, 9, 4);
    r.bufferRowLength = 12;
    r.bufferImageHeight = 10;
    roundtrip(&info, private_type, &r, 4, 1, 1, "3D slices");
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1, 0, 0, 0, 8, 6, 4);
    roundtrip(&info, private_type, &r, 4, 1, 1, "3D level 1");
    /* 1D array, cube */
    info = image_info(VK_IMAGE_TYPE_1D, VK_FORMAT_R8_UINT, 100, 1, 1, 2, 4,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 3, 5, 0, 0, 90, 1, 1);
    roundtrip(&info, private_type, &r, 1, 1, 1, "1D array");
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_UINT, 8, 8, 1, 1, 6,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 6, 0, 0, 0, 8, 8, 1);
    roundtrip(&info, private_type, &r, 16, 1, 1, "cube faces");
    /* rows further apart than Metal's bytesPerRow limit (16384 texels) */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8_UNORM, 8, 4, 1, 1, 2,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 2, 0, 0, 0, 8, 4, 1);
    r.bufferRowLength = 20000;
    roundtrip(&info, private_type, &r, 2, 1, 1, "wide rows");
    /* depth and stencil aspects; depth values in [0, 1] */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_D32_SFLOAT, 16, 16, 1, 1, 1,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    {
        struct img m = make_image(&info);
        struct buf src = make_buf(16 * 16 * 4, shared_type, 0), dst = make_buf(16 * 16 * 4, shared_type, 0);
        VkCommandBuffer cb;
        float *f = (float *)(void *)src.data;
        unsigned i;
        for (i = 0; i < 256; i++)
            f[i] = (float)i / 255.0f;
        r = region(VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1, 0, 0, 0, 16, 16, 1);
        cb = begin();
        vkCmdCopyBufferToImage(cb, src.buffer, m.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
        barrier(cb, m.image);
        vkCmdCopyImageToBuffer(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.buffer, 1, &r);
        barrier(cb, VK_NULL_HANDLE);
        CHECK_VK(finish(cb), VK_SUCCESS);
        CHECK(memcmp(src.data, dst.data, 1024) == 0);
        free_buf(&src);
        free_buf(&dst);
        free_image(&m);
    }
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_D16_UNORM, 20, 10, 1, 2, 2,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    r = region(VK_IMAGE_ASPECT_DEPTH_BIT, 1, 0, 2, 0, 0, 0, 10, 5, 1);
    roundtrip(&info, private_type, &r, 2, 1, 1, "D16");
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_S8_UINT, 20, 10, 1, 1, 1,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    r = region(VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1, 4, 4, 0, 16, 6, 1);
    roundtrip(&info, private_type, &r, 1, 1, 1, "S8");
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_D32_SFLOAT_S8_UINT, 16, 16, 1, 1, 2,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    r = region(VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 2, 0, 0, 0, 16, 16, 1);
    roundtrip(&info, private_type, &r, 1, 1, 1, "D32S8 stencil");
    /* compressed: whole blocks, and partial edge blocks */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_ASTC_6x5_UNORM_BLOCK, 26, 23, 1, 2, 1,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 26, 23, 1);
    roundtrip(&info, private_type, &r, 16, 6, 5, "ASTC 6x5 whole level");
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 12, 10, 0, 14, 13, 1);
    roundtrip(&info, private_type, &r, 16, 6, 5, "ASTC 6x5 edge blocks");
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, 16, 16, 1, 1, 1,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 4, 0, 0, 8, 16, 1);
    r.bufferRowLength = 16;
    roundtrip(&info, private_type, &r, 8, 4, 4, "ETC2");
    if (((struct pvk_physical_device *)pd)->bc_formats) {
        info = image_info(VK_IMAGE_TYPE_3D, VK_FORMAT_BC7_UNORM_BLOCK, 16, 8, 4, 1, 1,
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 16, 8, 4);
        roundtrip(&info, private_type, &r, 16, 4, 4, "BC7 3D");
    }
}

/* ---- Image -> image copies ---- */

static void upload(VkImage image, const VkBufferImageCopy *r, const struct buf *src)
{
    VkCommandBuffer cb = begin();
    vkCmdCopyBufferToImage(cb, src->buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, r);
    barrier(cb, image);
    CHECK_VK(finish(cb), VK_SUCCESS);
}

static void download(VkImage image, const VkBufferImageCopy *r, const struct buf *dst)
{
    VkCommandBuffer cb = begin();
    barrier(cb, image);
    vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst->buffer, 1, r);
    barrier(cb, VK_NULL_HANDLE);
    CHECK_VK(finish(cb), VK_SUCCESS);
}

static VkImageCopy image_copy(VkImageAspectFlags aspect, uint32_t sl, uint32_t slayer, int32_t sx, int32_t sy, int32_t sz,
                              uint32_t dl, uint32_t dlayer, int32_t dx, int32_t dy, int32_t dz, uint32_t layers,
                              uint32_t w, uint32_t h, uint32_t d)
{
    VkImageCopy c;
    memset(&c, 0, sizeof(c));
    c.srcSubresource.aspectMask = c.dstSubresource.aspectMask = aspect;
    c.srcSubresource.mipLevel = sl;
    c.srcSubresource.baseArrayLayer = slayer;
    c.srcSubresource.layerCount = layers;
    c.dstSubresource.mipLevel = dl;
    c.dstSubresource.baseArrayLayer = dlayer;
    c.dstSubresource.layerCount = layers;
    c.srcOffset.x = sx;
    c.srcOffset.y = sy;
    c.srcOffset.z = sz;
    c.dstOffset.x = dx;
    c.dstOffset.y = dy;
    c.dstOffset.z = dz;
    c.extent.width = w;
    c.extent.height = h;
    c.extent.depth = d;
    return c;
}

static void test_copy_image(void)
{
    VkImageUsageFlags xfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    struct buf src = make_buf(1 << 16, shared_type, 0), dst = make_buf(1 << 16, shared_type, 0);
    VkImageCreateInfo ia, ib;
    struct img a, b;
    VkBufferImageCopy r;
    VkImageCopy c;
    VkCommandBuffer cb;
    uint32_t y, bad, l;

    REQUIRE(src.data && dst.data);
    fill_pattern(src.data, 1 << 16, 3);

    /* Size-compatible formats, sub-rectangles of mip levels and layers. */
    ia = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32_UINT, 32, 32, 1, 2, 3, xfer);
    ib = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 1, 4, xfer);
    a = make_image(&ia);
    b = make_image(&ib);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 3, 0, 0, 0, 16, 16, 1);
    upload(a.image, &r, &src);
    c = image_copy(VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 2, 3, 0, 0, 2, 5, 6, 0, 2, 9, 7, 1);
    cb = begin();
    vkCmdCopyImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    CHECK_VK(finish(cb), VK_SUCCESS);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 2, 2, 0, 0, 0, 16, 16, 1);
    download(b.image, &r, &dst);
    for (bad = 0, y = 0; y < 7; y++) /* layer 2+l of b, at (5,6) = layer 1+l of a's level 1 at (2,3) */
        for (l = 0; l < 2; l++)
            if (memcmp(dst.data + ((l * 16 + 6 + y) * 16 + 5) * 4, src.data + (((l + 1) * 16 + 3 + y) * 16 + 2) * 4, 9 * 4))
                bad++;
    CHECK(bad == 0);
    free_image(&a);
    free_image(&b);

    /* Compressed <-> uncompressed of the same block size (ETC2 RGB8 and
     * RG32_UINT, 8 B): extents in source texels. */
    ia = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 32, 32, 1, 1, 1, xfer);
    ib = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32_UINT, 8, 8, 1, 1, 1, xfer);
    a = make_image(&ia);
    b = make_image(&ib);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 32, 32, 1);
    upload(a.image, &r, &src);
    c = image_copy(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 8, 4, 0, 0, 0, 1, 2, 0, 1, 16, 12, 1);
    cb = begin();
    vkCmdCopyImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    CHECK_VK(finish(cb), VK_SUCCESS);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 8, 8, 1);
    download(b.image, &r, &dst);
    for (bad = 0, y = 0; y < 3; y++) /* blocks (2,1).. of a -> texels (1,2).. of b, 4 x 3 */
        if (memcmp(dst.data + ((2 + y) * 8 + 1) * 8, src.data + ((1 + y) * 8 + 2) * 8, 4 * 8))
            bad++;
    CHECK(bad == 0);
    c = image_copy(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 2, 0, 0, 0, 16, 16, 0, 1, 4, 3, 1); /* and back */
    cb = begin();
    vkCmdCopyImage(cb, b.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, a.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    CHECK_VK(finish(cb), VK_SUCCESS);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 32, 32, 1);
    download(a.image, &r, &dst);
    for (bad = 0, y = 0; y < 3; y++) /* blocks (4,4).. of a hold the same blocks again */
        if (memcmp(dst.data + ((4 + y) * 8 + 4) * 8, src.data + ((1 + y) * 8 + 2) * 8, 4 * 8))
            bad++;
    CHECK(bad == 0);
    free_image(&a);
    free_image(&b);

    /* 3D slices <-> 2D array layers */
    ia = image_info(VK_IMAGE_TYPE_3D, VK_FORMAT_R16_UINT, 16, 16, 8, 1, 1, xfer);
    ib = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R16_UINT, 16, 16, 1, 1, 6, xfer);
    a = make_image(&ia);
    b = make_image(&ib);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 16, 16, 8);
    upload(a.image, &r, &src);
    c = image_copy(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 16, 16, 4);
    c.dstSubresource.layerCount = 4; /* slices 2..5 -> layers 1..4 */
    cb = begin();
    vkCmdCopyImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    barrier(cb, b.image);
    c = image_copy(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 1, 16, 16, 1); /* layer 1 -> slice 6 */
    vkCmdCopyImage(cb, b.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, a.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    CHECK_VK(finish(cb), VK_SUCCESS);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 4, 0, 0, 0, 16, 16, 1);
    download(b.image, &r, &dst);
    CHECK(memcmp(dst.data, src.data + 2 * 512, 4 * 512) == 0);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 6, 16, 16, 1);
    download(a.image, &r, &dst);
    CHECK(memcmp(dst.data, src.data + 2 * 512, 512) == 0);
    free_image(&a);
    free_image(&b);

    /* One aspect of a combined depth/stencil image (through a buffer). */
    ia = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_D32_SFLOAT_S8_UINT, 16, 16, 1, 1, 1, xfer);
    a = make_image(&ia);
    b = make_image(&ia);
    r = region(VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1, 0, 0, 0, 16, 16, 1);
    upload(a.image, &r, &src);
    memset(dst.data, 0, 256);
    upload(b.image, &r, &dst); /* b's stencil: zero */
    c = image_copy(VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 16, 16, 1);
    cb = begin();
    vkCmdCopyImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    CHECK_VK(finish(cb), VK_SUCCESS);
    download(b.image, &r, &dst);
    CHECK(memcmp(dst.data, src.data, 256) == 0);
    free_image(&a);
    free_image(&b);

    /* Multisampled images: a whole subresource, compared after resolving. */
    ia = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 1, 1, xfer | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    ia.samples = VK_SAMPLE_COUNT_4_BIT;
    a = make_image(&ia);
    b = make_image(&ia);
    {
        VkClearColorValue color;
        VkImageSubresourceRange range;
        VkImageResolve res;
        struct img single;
        ib = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 1, 1, xfer);
        single = make_image(&ib);
        memset(&color, 0, sizeof(color));
        color.float32[0] = 1.0f;
        color.float32[2] = 0.5f;
        memset(&range, 0, sizeof(range));
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.levelCount = range.layerCount = 1;
        memset(&res, 0, sizeof(res));
        res.srcSubresource.aspectMask = res.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        res.srcSubresource.layerCount = res.dstSubresource.layerCount = 1;
        res.extent.width = res.extent.height = 16;
        res.extent.depth = 1;
        c = image_copy(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 16, 16, 1);
        cb = begin();
        vkCmdClearColorImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
        barrier(cb, a.image);
        vkCmdCopyImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        barrier(cb, b.image);
        vkCmdResolveImage(cb, b.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, single.image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &res);
        CHECK_VK(finish(cb), VK_SUCCESS);
        r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 16, 16, 1);
        download(single.image, &r, &dst);
        CHECK(dst.data[0] == 0xff && dst.data[1] == 0 && (dst.data[2] == 0x80 || dst.data[2] == 0x7f) &&
              dst.data[3] == 0 && memcmp(dst.data, dst.data + 255 * 4, 4) == 0);
        free_image(&single);
    }
    free_image(&a);
    free_image(&b);
    free_buf(&src);
    free_buf(&dst);
}

/* ---- Clears ---- */

static void clear_range(VkImageSubresourceRange *range, VkImageAspectFlags aspect, uint32_t level, uint32_t levels,
                        uint32_t layer, uint32_t layers)
{
    memset(range, 0, sizeof(*range));
    range->aspectMask = aspect;
    range->baseMipLevel = level;
    range->levelCount = levels;
    range->baseArrayLayer = layer;
    range->layerCount = layers;
}

/* Every texel block of a run equals the given bytes. */
static int all_texels(const unsigned char *p, size_t count, const void *texel, size_t size)
{
    size_t i;
    for (i = 0; i < count; i++)
        if (memcmp(p + i * size, texel, size)) {
            fprintf(stderr, "  texel %zu differs\n", i);
            return 0;
        }
    return 1;
}

static void test_clear_color(void)
{
    VkImageUsageFlags xfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    struct buf dst = make_buf(1 << 16, shared_type, 0);
    VkImageSubresourceRange ranges[2];
    VkClearColorValue color;
    VkImageCreateInfo info;
    VkBufferImageCopy r;
    VkCommandBuffer cb;
    struct img m;
    unsigned char want[16];

    REQUIRE(dst.data);
    /* unorm: levels 1-2 of layers 1-3 of an array; the rest keeps the first clear */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 3, 5, xfer);
    m = make_image(&info);
    memset(&color, 0, sizeof(color));
    clear_range(&ranges[0], VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS);
    clear_range(&ranges[1], VK_IMAGE_ASPECT_COLOR_BIT, 1, 2, 1, 3);
    cb = begin();
    vkCmdClearColorImage(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &ranges[0]);
    barrier(cb, m.image);
    color.float32[0] = 1.0f;
    color.float32[1] = 0.25f;
    color.float32[2] = 2.0f; /* clamped */
    color.float32[3] = 0.0f;
    vkCmdClearColorImage(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &ranges[1]);
    CHECK_VK(finish(cb), VK_SUCCESS);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 5, 0, 0, 0, 8, 8, 1);
    download(m.image, &r, &dst);
    want[0] = 0xff;
    want[1] = 0x40;
    want[2] = 0xff;
    want[3] = 0;
    CHECK(all_texels(dst.data + 64 * 4, 3 * 64, want, 4));
    memset(want, 0, 4);
    CHECK(all_texels(dst.data, 64, want, 4) && all_texels(dst.data + 4 * 64 * 4, 64, want, 4));
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 5, 0, 0, 0, 16, 16, 1);
    download(m.image, &r, &dst);
    CHECK(all_texels(dst.data, 5 * 256, want, 4));
    free_image(&m);

    /* integers keep their low bits (Clear Values) */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SINT, 8, 8, 1, 1, 1, xfer);
    m = make_image(&info);
    color.int32[0] = -1;
    color.int32[1] = 0x12345;
    color.int32[2] = -32768;
    color.int32[3] = 70000;
    clear_range(&ranges[0], VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1);
    cb = begin();
    vkCmdClearColorImage(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &ranges[0]);
    CHECK_VK(finish(cb), VK_SUCCESS);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 8, 8, 1);
    download(m.image, &r, &dst);
    {
        int16_t w[4];
        w[0] = -1;
        w[1] = 0x2345;
        w[2] = -32768;
        w[3] = (int16_t)(70000 - 65536);
        CHECK(all_texels(dst.data, 64, w, 8));
    }
    free_image(&m);
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_UINT, 4, 4, 1, 1, 1, xfer);
    m = make_image(&info);
    color.uint32[0] = 0xffffffffu;
    color.uint32[1] = 0x80000001u;
    color.uint32[2] = 0;
    color.uint32[3] = 123456789u;
    cb = begin();
    vkCmdClearColorImage(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &ranges[0]);
    CHECK_VK(finish(cb), VK_SUCCESS);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 4, 4, 1);
    download(m.image, &r, &dst);
    CHECK(all_texels(dst.data, 16, color.uint32, 16));
    free_image(&m);

    /* a swizzled format: the Vulkan components land in the right bits */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_B4G4R4A4_UNORM_PACK16, 4, 4, 1, 1, 1, xfer);
    m = make_image(&info);
    color.float32[0] = 1.0f;         /* R: bits 4-7 */
    color.float32[1] = 0.0f;         /* G: bits 8-11 */
    color.float32[2] = 0.0f;         /* B: bits 12-15 */
    color.float32[3] = 1.0f / 15.0f; /* A: bits 0-3 */
    cb = begin();
    vkCmdClearColorImage(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &ranges[0]);
    CHECK_VK(finish(cb), VK_SUCCESS);
    download(m.image, &r, &dst);
    {
        uint16_t w = 0x00f1;
        CHECK(all_texels(dst.data, 16, &w, 2));
    }
    free_image(&m);

    /* sRGB: linear clear values are encoded */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8_SRGB, 4, 4, 1, 1, 1, xfer);
    m = make_image(&info);
    color.float32[0] = 0.5f;
    cb = begin();
    vkCmdClearColorImage(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &ranges[0]);
    CHECK_VK(finish(cb), VK_SUCCESS);
    download(m.image, &r, &dst);
    CHECK(dst.data[0] == 0xbc && dst.data[15] == 0xbc);
    free_image(&m);

    /* 3D: every slice of level 1 */
    info = image_info(VK_IMAGE_TYPE_3D, VK_FORMAT_R32_SFLOAT, 8, 8, 8, 2, 1, xfer);
    m = make_image(&info);
    color.float32[0] = 3.5f;
    clear_range(&ranges[0], VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 0, 1);
    cb = begin();
    vkCmdClearColorImage(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &ranges[0]);
    CHECK_VK(finish(cb), VK_SUCCESS);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1, 0, 0, 0, 4, 4, 4);
    download(m.image, &r, &dst);
    CHECK(all_texels(dst.data, 64, &color.float32[0], 4));
    free_image(&m);
    free_buf(&dst);
}

static void test_clear_depth_stencil(void)
{
    VkImageUsageFlags xfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    struct buf dst = make_buf(1 << 16, shared_type, 0);
    VkImageSubresourceRange range;
    VkClearDepthStencilValue v;
    VkImageCreateInfo info;
    VkBufferImageCopy r;
    VkCommandBuffer cb;
    struct img m;
    float d;
    unsigned char s;

    REQUIRE(dst.data);
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_D32_SFLOAT_S8_UINT, 16, 16, 1, 2, 3,
                      xfer | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    m = make_image(&info);
    v.depth = 0.25f;
    v.stencil = 0x1a7; /* the low 8 bits */
    clear_range(&range, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 2, 0, 3);
    cb = begin();
    vkCmdClearDepthStencilImage(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
    barrier(cb, m.image);
    v.depth = 1.0f; /* depth only, of layer 1 */
    v.stencil = 0;
    clear_range(&range, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 1, 1);
    vkCmdClearDepthStencilImage(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
    barrier(cb, m.image);
    v.stencil = 9; /* stencil only, of layer 2 */
    clear_range(&range, VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 2, 1);
    vkCmdClearDepthStencilImage(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
    CHECK_VK(finish(cb), VK_SUCCESS);
    r = region(VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 3, 0, 0, 0, 16, 16, 1);
    download(m.image, &r, &dst);
    d = 0.25f;
    CHECK(all_texels(dst.data, 256, &d, 4) && all_texels(dst.data + 2 * 1024, 256, &d, 4));
    d = 1.0f;
    CHECK(all_texels(dst.data + 1024, 256, &d, 4));
    r = region(VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 3, 0, 0, 0, 16, 16, 1);
    download(m.image, &r, &dst);
    s = 0xa7;
    CHECK(all_texels(dst.data, 512, &s, 1));
    s = 9;
    CHECK(all_texels(dst.data + 512, 256, &s, 1));
    r = region(VK_IMAGE_ASPECT_STENCIL_BIT, 1, 0, 1, 0, 0, 0, 8, 8, 1);
    download(m.image, &r, &dst);
    s = 0xa7;
    CHECK(all_texels(dst.data, 64, &s, 1));
    free_image(&m);

    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_D16_UNORM, 8, 8, 1, 1, 1, xfer);
    m = make_image(&info);
    v.depth = 0.5f;
    clear_range(&range, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
    cb = begin();
    vkCmdClearDepthStencilImage(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
    CHECK_VK(finish(cb), VK_SUCCESS);
    r = region(VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1, 0, 0, 0, 8, 8, 1);
    download(m.image, &r, &dst);
    {
        uint16_t w;
        memcpy(&w, dst.data, 2);
        CHECK(w == 0x8000 || w == 0x7fff);
        CHECK(all_texels(dst.data, 64, &w, 2));
    }
    free_image(&m);
    free_buf(&dst);
}

/* ---- Blits ---- */

static struct img image_with(VkImageType type, VkFormat format, uint32_t w, uint32_t h, uint32_t d, uint32_t levels,
                             uint32_t layers, const void *texels, size_t size)
{
    VkImageCreateInfo info = image_info(type, format, w, h, d, levels, layers,
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                            VK_IMAGE_USAGE_SAMPLED_BIT);
    struct img m = make_image(&info);
    if (texels) {
        struct buf b = make_buf(size, shared_type, 0);
        VkBufferImageCopy r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers, 0, 0, 0, w, h, d);
        memcpy(b.data, texels, size);
        upload(m.image, &r, &b);
        free_buf(&b);
    }
    return m;
}

static VkImageBlit blit_region(uint32_t sl, uint32_t slayer, int32_t sx0, int32_t sy0, int32_t sz0, int32_t sx1, int32_t sy1,
                               int32_t sz1, uint32_t dl, uint32_t dlayer, int32_t dx0, int32_t dy0, int32_t dz0, int32_t dx1,
                               int32_t dy1, int32_t dz1, uint32_t layers)
{
    VkImageBlit b;
    memset(&b, 0, sizeof(b));
    b.srcSubresource.aspectMask = b.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.srcSubresource.mipLevel = sl;
    b.srcSubresource.baseArrayLayer = slayer;
    b.srcSubresource.layerCount = layers;
    b.dstSubresource.mipLevel = dl;
    b.dstSubresource.baseArrayLayer = dlayer;
    b.dstSubresource.layerCount = layers;
    b.srcOffsets[0].x = sx0;
    b.srcOffsets[0].y = sy0;
    b.srcOffsets[0].z = sz0;
    b.srcOffsets[1].x = sx1;
    b.srcOffsets[1].y = sy1;
    b.srcOffsets[1].z = sz1;
    b.dstOffsets[0].x = dx0;
    b.dstOffsets[0].y = dy0;
    b.dstOffsets[0].z = dz0;
    b.dstOffsets[1].x = dx1;
    b.dstOffsets[1].y = dy1;
    b.dstOffsets[1].z = dz1;
    return b;
}

static void blit(struct img *src, struct img *dst, const VkImageBlit *b, VkFilter filter)
{
    VkCommandBuffer cb = begin();
    vkCmdBlitImage(cb, src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                   b, filter);
    CHECK_VK(finish(cb), VK_SUCCESS);
}

static int near(int a, int b)
{
    return a - b <= 1 && b - a <= 1;
}

static void test_blit(void)
{
    struct buf out = make_buf(1 << 16, shared_type, 0);
    uint8_t src4[4 * 4 * 4], *o;
    uint32_t x, y, c;
    struct img a, b;
    VkImageBlit r;
    VkBufferImageCopy dl;

    REQUIRE(out.data);
    o = out.data;
    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++)
            for (c = 0; c < 4; c++)
                src4[(y * 4 + x) * 4 + c] = (uint8_t)(16 * (y * 4 + x) + 4 * c);
    a = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 1, 1, src4, sizeof(src4));

    /* 2x down, linear: each destination texel averages a 2x2 block */
    b = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 2, 2, 1, 1, 1, NULL, 0);
    r = blit_region(0, 0, 0, 0, 0, 4, 4, 1, 0, 0, 0, 0, 0, 2, 2, 1, 1);
    blit(&a, &b, &r, VK_FILTER_LINEAR);
    dl = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 2, 2, 1);
    download(b.image, &dl, &out);
    for (y = 0; y < 2; y++)
        for (x = 0; x < 2; x++)
            for (c = 0; c < 4; c++) {
                int sum = 0, want, k;
                for (k = 0; k < 4; k++)
                    sum += src4[(((2 * y + k / 2) * 4) + 2 * x + k % 2) * 4 + c];
                want = sum / 4;
                CHECK(near(o[(y * 2 + x) * 4 + c], want));
            }
    /* 2x down, nearest: texel floor(2x + 1) */
    blit(&a, &b, &r, VK_FILTER_NEAREST);
    download(b.image, &dl, &out);
    for (y = 0; y < 2; y++)
        for (x = 0; x < 2; x++)
            CHECK(memcmp(o + (y * 2 + x) * 4, src4 + ((2 * y + 1) * 4 + 2 * x + 1) * 4, 4) == 0);
    free_image(&b);

    /* 2x up with a horizontal flip, nearest */
    b = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 8, 8, 1, 1, 1, NULL, 0);
    r = blit_region(0, 0, 0, 0, 0, 4, 4, 1, 0, 0, 8, 0, 0, 0, 8, 1, 1);
    blit(&a, &b, &r, VK_FILTER_NEAREST);
    dl = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 8, 8, 1);
    download(b.image, &dl, &out);
    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++)
            CHECK(memcmp(o + (y * 8 + x) * 4, src4 + ((y / 2) * 4 + (7 - x) / 2) * 4, 4) == 0);
    /* a sub-rectangle into a sub-rectangle leaves the rest alone */
    r = blit_region(0, 0, 1, 1, 0, 3, 3, 1, 0, 0, 5, 5, 0, 7, 7, 1, 1);
    blit(&a, &b, &r, VK_FILTER_NEAREST);
    download(b.image, &dl, &out);
    CHECK(memcmp(o + (5 * 8 + 5) * 4, src4 + (1 * 4 + 1) * 4, 4) == 0);
    CHECK(memcmp(o + (6 * 8 + 6) * 4, src4 + (2 * 4 + 2) * 4, 4) == 0);
    CHECK(memcmp(o + (4 * 8 + 4) * 4, src4 + (2 * 4 + 1) * 4, 4) == 0); /* the flipped blit's */
    free_image(&b);

    /* formats: UNORM -> SFLOAT; sRGB decodes */
    b = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, 4, 4, 1, 1, 1, NULL, 0);
    r = blit_region(0, 0, 0, 0, 0, 4, 4, 1, 0, 0, 0, 0, 0, 4, 4, 1, 1);
    blit(&a, &b, &r, VK_FILTER_NEAREST);
    dl = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 4, 4, 1);
    download(b.image, &dl, &out);
    {
        uint16_t h;
        memcpy(&h, o + 5 * 8 + 2, 2); /* texel 5, green: 84 / 255 as a half */
        CHECK(h >= 0x3542 && h <= 0x3548); /* 0.3294 */
    }
    free_image(&b);
    free_image(&a);

    /* integers: nearest, exact; a 3D source and destination, linear */
    {
        uint32_t ints[2 * 2 * 4] = { 1, 2, 3, 4, 0xffffffffu, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0x80000000u };
        a = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_UINT, 2, 2, 1, 1, 1, ints, sizeof(ints));
        b = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_UINT, 4, 4, 1, 1, 1, NULL, 0);
        r = blit_region(0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 4, 4, 1, 1);
        blit(&a, &b, &r, VK_FILTER_NEAREST);
        download(b.image, &dl, &out);
        CHECK(memcmp(o + 3 * 16, ints + 4, 16) == 0 && memcmp(o + 15 * 16, ints + 12, 16) == 0);
        free_image(&a);
        free_image(&b);
    }
    {
        float f[4 * 4 * 4];
        unsigned i;
        for (i = 0; i < 64; i++)
            f[i] = (float)i;
        a = image_with(VK_IMAGE_TYPE_3D, VK_FORMAT_R32_SFLOAT, 4, 4, 4, 1, 1, f, sizeof(f));
        b = image_with(VK_IMAGE_TYPE_3D, VK_FORMAT_R32_SFLOAT, 2, 2, 2, 1, 1, NULL, 0);
        r = blit_region(0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 2, 2, 2, 1);
        blit(&a, &b, &r, VK_FILTER_LINEAR);
        dl = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 2, 2, 2);
        download(b.image, &dl, &out);
        for (i = 0; i < 8; i++) {
            /* the mean of a 2x2x2 block: x0 + 0.5, y0 * 4 + 2, z0 * 16 + 8 */
            unsigned x0 = 2 * (i & 1), y0 = 2 * ((i >> 1) & 1), z0 = 2 * (i >> 2);
            float want = (float)(z0 * 16 + y0 * 4 + x0) + 0.5f + 2.0f + 8.0f, got;
            memcpy(&got, o + i * 4, 4);
            CHECK(fabsf(got - want) < 0.01f);
        }
        free_image(&a);
        free_image(&b);
    }

    /* a swizzled source format keeps its components */
    {
        uint16_t t = 0x1234; /* B4G4R4A4: B=1 G=2 R=3 A=4 */
        uint16_t texels[4];
        texels[0] = texels[1] = texels[2] = texels[3] = t;
        a = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_B4G4R4A4_UNORM_PACK16, 2, 2, 1, 1, 1, texels, sizeof(texels));
        b = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 2, 2, 1, 1, 1, NULL, 0);
        r = blit_region(0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 2, 2, 1, 1);
        blit(&a, &b, &r, VK_FILTER_NEAREST);
        dl = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 2, 2, 1);
        download(b.image, &dl, &out);
        CHECK(o[0] == 0x33 && o[1] == 0x22 && o[2] == 0x11 && o[3] == 0x44);
        free_image(&a);
        free_image(&b);
    }

    /* unscaled within one format (a copy): exact texels whatever the filter,
     * sRGB too, at offsets, in array layers; the rest stays */
    {
        uint8_t px[4 * 4 * 2 * 4], fill[4 * 4 * 2 * 4];
        uint32_t l;
        for (x = 0; x < sizeof(px); x++)
            px[x] = (uint8_t)(x * 7 + 3);
        memset(fill, 0xee, sizeof(fill));
        a = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_SRGB, 4, 4, 1, 1, 2, px, sizeof(px));
        b = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_SRGB, 4, 4, 1, 1, 2, fill, sizeof(fill));
        r = blit_region(0, 0, 1, 0, 0, 3, 2, 1, 0, 0, 0, 2, 0, 2, 4, 1, 2); /* (1, 0)-(3, 2) to (0, 2)-(2, 4) */
        blit(&a, &b, &r, VK_FILTER_LINEAR);
        dl = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 2, 0, 0, 0, 4, 4, 1);
        download(b.image, &dl, &out);
        for (l = 0; l < 2; l++)
            for (y = 0; y < 4; y++)
                for (x = 0; x < 4; x++) {
                    const uint8_t *got = o + ((l * 4 + y) * 4 + x) * 4;
                    if (x < 2 && y >= 2)
                        CHECK(memcmp(got, px + ((l * 4 + y - 2) * 4 + x + 1) * 4, 4) == 0);
                    else
                        CHECK(memcmp(got, fill, 4) == 0);
                }
        free_image(&a);
        free_image(&b);
    }

    /* a mip chain, level by level, and array layers */
    {
        uint8_t px[8 * 8 * 2 * 4];
        VkCommandBuffer cb;
        uint32_t l;
        for (x = 0; x < sizeof(px); x++)
            px[x] = (uint8_t)(x & 0xfc);
        a = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 8, 8, 1, 4, 2, px, sizeof(px));
        cb = begin();
        for (l = 1; l < 4; l++) {
            r = blit_region(l - 1, 0, 0, 0, 0, 8 >> (l - 1), 8 >> (l - 1), 1, l, 0, 0, 0, 0, 8 >> l, 8 >> l, 1, 2);
            vkCmdBlitImage(cb, a.image, VK_IMAGE_LAYOUT_GENERAL, a.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r, VK_FILTER_LINEAR);
            barrier(cb, a.image);
        }
        CHECK_VK(finish(cb), VK_SUCCESS);
        dl = region(VK_IMAGE_ASPECT_COLOR_BIT, 3, 0, 2, 0, 0, 0, 1, 1, 1);
        download(a.image, &dl, &out);
        for (l = 0; l < 2; l++) /* level 3: the mean of each layer */
            for (c = 0; c < 4; c++) {
                unsigned sum = 0;
                for (x = 0; x < 64; x++)
                    sum += px[(l * 64 + x) * 4 + c];
                CHECK(o[l * 4 + c] + 2 >= sum / 64 && o[l * 4 + c] <= sum / 64 + 2);
            }
        free_image(&a);
    }
    free_buf(&out);
}

/* ---- Resolves ---- */

static void test_resolve(void)
{
    VkImageUsageFlags xfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    struct buf out = make_buf(1 << 16, shared_type, 0);
    VkImageCreateInfo info;
    VkImageSubresourceRange range;
    VkClearColorValue color;
    VkImageResolve res;
    VkBufferImageCopy dl;
    VkCommandBuffer cb;
    struct img ms, single;
    uint32_t x, y;

    REQUIRE(out.data);
    /* float: two layers, a region at an offset */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, 16, 16, 1, 1, 2,
                      xfer | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    info.samples = VK_SAMPLE_COUNT_4_BIT;
    ms = make_image(&info);
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, 16, 16, 1, 1, 3, xfer);
    single = make_image(&info);
    memset(&color, 0, sizeof(color));
    clear_range(&range, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS);
    memset(&res, 0, sizeof(res));
    res.srcSubresource.aspectMask = res.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    res.srcSubresource.layerCount = res.dstSubresource.layerCount = 2;
    res.dstSubresource.baseArrayLayer = 1;
    res.srcOffset.x = 2;
    res.srcOffset.y = 3;
    res.dstOffset.x = 8;
    res.dstOffset.y = 4;
    res.extent.width = 6;
    res.extent.height = 5;
    res.extent.depth = 1;
    cb = begin();
    vkCmdClearColorImage(cb, single.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
    color.float32[0] = 0.5f;
    color.float32[1] = -2.0f;
    color.float32[2] = 8.0f;
    color.float32[3] = 1.0f;
    vkCmdClearColorImage(cb, ms.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
    barrier(cb, VK_NULL_HANDLE);
    vkCmdResolveImage(cb, ms.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, single.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      1, &res);
    CHECK_VK(finish(cb), VK_SUCCESS);
    dl = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 3, 0, 0, 0, 16, 16, 1);
    download(single.image, &dl, &out);
    {
        uint16_t want[4] = { 0x3800, 0xc000, 0x4800, 0x3c00 }, zero[4] = { 0, 0, 0, 0 };
        uint32_t l;
        for (l = 0; l < 3; l++)
            for (y = 0; y < 16; y++)
                for (x = 0; x < 16; x++) {
                    int inside = l >= 1 && x >= 8 && x < 14 && y >= 4 && y < 9;
                    if (memcmp(out.data + ((l * 16 + y) * 16 + x) * 8, inside ? want : zero, 8)) {
                        PVK_TEST_FAIL("layer %u texel (%u, %u)", l, x, y);
                        l = 3;
                        y = 16;
                        break;
                    }
                }
    }
    free_image(&ms);
    free_image(&single);

    /* integers: sample 0 */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32_UINT, 8, 8, 1, 1, 1, xfer | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    info.samples = VK_SAMPLE_COUNT_4_BIT;
    ms = make_image(&info);
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32_UINT, 8, 8, 1, 1, 1, xfer);
    single = make_image(&info);
    color.uint32[0] = 0xdeadbeefu;
    clear_range(&range, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1);
    res.srcSubresource.layerCount = res.dstSubresource.layerCount = 1;
    res.dstSubresource.baseArrayLayer = 0;
    res.srcOffset.x = res.srcOffset.y = res.dstOffset.x = res.dstOffset.y = 0;
    res.extent.width = res.extent.height = 8;
    cb = begin();
    vkCmdClearColorImage(cb, ms.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
    barrier(cb, ms.image);
    vkCmdResolveImage(cb, ms.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, single.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      1, &res);
    CHECK_VK(finish(cb), VK_SUCCESS);
    dl = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 8, 8, 1);
    download(single.image, &dl, &out);
    CHECK(all_texels(out.data, 64, &color.uint32[0], 4));
    free_image(&single);

    /* into a 3D image: depth plane dstOffset.z */
    info = image_info(VK_IMAGE_TYPE_3D, VK_FORMAT_R32_UINT, 8, 8, 4, 1, 1, xfer);
    single = make_image(&info);
    {
        VkClearColorValue zero;
        uint32_t z;
        memset(&zero, 0, sizeof(zero));
        res.dstOffset.z = 2;
        cb = begin();
        vkCmdClearColorImage(cb, single.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
        barrier(cb, single.image);
        vkCmdResolveImage(cb, ms.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, single.image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &res);
        CHECK_VK(finish(cb), VK_SUCCESS);
        dl = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 8, 8, 4);
        download(single.image, &dl, &out);
        for (z = 0; z < 4; z++)
            if (!all_texels(out.data + z * 256, 64, z == 2 ? &color.uint32[0] : &zero.uint32[0], 4))
                PVK_TEST_FAIL("depth plane %u", z);
    }
    free_image(&ms);
    free_image(&single);
    free_buf(&out);
}

/* ---- Sampling through Metal ----
 *
 * A kernel reads a view (and a sampler) bound by MTLResourceID in an
 * argument table, as descriptor sets will bind them, at coordinates from a
 * buffer, and writes float4 results. */

static const char harness_msl[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "#define K(name, T, body) kernel void name(T t [[texture(0)]], sampler s [[sampler(0)]],\\\n"
    "    device float4 *out [[buffer(0)]], constant float4 *in [[buffer(1)]], uint i [[thread_position_in_grid]])\\\n"
    "    { float4 c = in[i]; out[i] = body; }\n"
    "K(k2d, texture2d<float>, t.sample(s, c.xy, level(c.w)))\n"
    "K(k2d_read, texture2d<float>, t.read(uint2(c.xy), uint(c.w)))\n"
    "K(k2d_uint, texture2d<uint>, float4(t.read(uint2(c.xy), uint(c.w))))\n"
    "K(k2darray, texture2d_array<float>, t.sample(s, c.xy, uint(c.z), level(c.w)))\n"
    "K(kcube, texturecube<float>, t.sample(s, c.xyz, level(c.w)))\n"
    "K(kcubearray, texturecube_array<float>, t.sample(s, c.xyz, uint(c.w), level(0)))\n"
    "K(k3d, texture3d<float>, t.sample(s, c.xyz, level(c.w)))\n"
    "K(kdepth, depth2d<float>, float4(t.sample_compare(s, c.xy, c.z, level(0))))\n"
    "K(kdepth_read, depth2d<float>, float4(t.read(uint2(c.xy))))\n"
    "K(kbuffer, texture_buffer<float>, t.read(uint(c.x)))\n"
    "K(kbuffer_uint, texture_buffer<uint>, float4(t.read(uint(c.x))))\n";

struct harness_pipeline {
    const char *entry;
    id pso;
};
static struct harness_pipeline harness_pipelines[16];
static id harness_queue, harness_event, harness_table;
static uint64_t harness_serial;

static int harness(const char *entry, id texture, VkSampler sampler, const float *coords, unsigned n, float *out)
{
    struct pvk_device *dev = (struct pvk_device *)device;
    id mtl = dev->physical_device->mtl_device, pso = NULL, alloc, cb, enc;
    struct buf in = make_buf(16 * n, shared_type, 0), res = make_buf(16 * n, shared_type, 0);
    pvk_MTLSize groups, threads;
    unsigned i;
    char err[256];

    for (i = 0; i < 16 && harness_pipelines[i].entry; i++)
        if (!strcmp(harness_pipelines[i].entry, entry))
            pso = harness_pipelines[i].pso;
    if (!pso && i < 16) {
        pso = pvk_compile_compute(dev, harness_msl, entry, 1, err, sizeof(err));
        if (!pso)
            fprintf(stderr, "harness %s: %s\n", entry, err);
        harness_pipelines[i].entry = entry;
        harness_pipelines[i].pso = pso;
    }
    if (!harness_queue) {
        id desc = pvk_MTL4ArgumentTableDescriptor_new();
        harness_queue = pvk_MTLDevice_newMTL4CommandQueue(mtl);
        pvk_MTL4CommandQueue_addResidencySet(harness_queue, dev->residency_set);
        harness_event = pvk_MTLDevice_newSharedEvent(mtl);
        pvk_MTL4ArgumentTableDescriptor_setMaxBufferBindCount(desc, 2);
        pvk_MTL4ArgumentTableDescriptor_setMaxTextureBindCount(desc, 1);
        PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setMaxSamplerStateBindCount:"), 1);
        harness_table = pvk_MTLDevice_newArgumentTableWithDescriptor_error(mtl, desc, NULL);
        pvk_release(desc);
    }
    if (!pso || !in.data || !res.data || !harness_table) {
        free_buf(&in);
        free_buf(&res);
        return 0;
    }
    memcpy(in.data, coords, 16 * n);
    alloc = pvk_MTLDevice_newCommandAllocator(mtl);
    cb = pvk_MTLDevice_newCommandBuffer(mtl);
    pvk_MTL4CommandBuffer_beginCommandBufferWithAllocator(cb, alloc);
    enc = pvk_MTL4CommandBuffer_computeCommandEncoder(cb);
    pvk_MTL4ComputeCommandEncoder_setComputePipelineState(enc, pso);
    pvk_MTL4ComputeCommandEncoder_setArgumentTable(enc, harness_table);
    pvk_MTL4ArgumentTable_setAddress_atIndex(harness_table, ((struct pvk_buffer *)res.buffer)->gpu_address, 0);
    pvk_MTL4ArgumentTable_setAddress_atIndex(harness_table, ((struct pvk_buffer *)in.buffer)->gpu_address, 1);
    pvk_MTL4ArgumentTable_setTexture_atIndex(harness_table, pvk_MTLTexture_gpuResourceID(texture), 0);
    if (sampler)
        PVK_MSG(void, (id, SEL, pvk_MTLResourceID, unsigned long))(
            harness_table, sel_registerName("setSamplerState:atIndex:"),
            pvk_MTLSamplerState_gpuResourceID(((struct pvk_sampler *)sampler)->mtl_sampler), 0);
    groups.width = n;
    groups.height = groups.depth = 1;
    threads.width = threads.height = threads.depth = 1;
    pvk_MTL4ComputeCommandEncoder_dispatchThreadgroups_threadsPerThreadgroup(enc, groups, threads);
    pvk_MTL4CommandEncoder_endEncoding(enc);
    pvk_MTL4CommandBuffer_endCommandBuffer(cb);
    pvk_MTL4CommandQueue_commit_count(harness_queue, &cb, 1);
    pvk_MTL4CommandQueue_signalEvent_value(harness_queue, harness_event, ++harness_serial);
    pvk_MTLSharedEvent_waitUntilSignaledValue_timeoutMS(harness_event, harness_serial, 10000);
    memcpy(out, res.data, 16 * n);
    pvk_release(cb);
    pvk_release(alloc);
    free_buf(&in);
    free_buf(&res);
    return 1;
}

static void harness_done(void)
{
    unsigned i;
    for (i = 0; i < 16; i++)
        pvk_release(harness_pipelines[i].pso);
    pvk_release(harness_table);
    pvk_release(harness_event);
    pvk_release(harness_queue);
}

static VkImageView make_view(VkImage image, VkImageViewType type, VkFormat format, VkImageAspectFlags aspect,
                             uint32_t level, uint32_t levels, uint32_t layer, uint32_t layers, VkComponentMapping comp)
{
    VkImageViewCreateInfo vi;
    VkImageView view = VK_NULL_HANDLE;
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = image;
    vi.viewType = type;
    vi.format = format;
    vi.components = comp;
    vi.subresourceRange.aspectMask = aspect;
    vi.subresourceRange.baseMipLevel = level;
    vi.subresourceRange.levelCount = levels;
    vi.subresourceRange.baseArrayLayer = layer;
    vi.subresourceRange.layerCount = layers;
    if (vkCreateImageView(device, &vi, NULL, &view) != VK_SUCCESS)
        PVK_TEST_FAIL("vkCreateImageView failed");
    return view;
}

static VkComponentMapping identity_mapping(void)
{
    VkComponentMapping m;
    memset(&m, 0, sizeof(m)); /* VK_COMPONENT_SWIZZLE_IDENTITY */
    return m;
}

static id view_texture(VkImageView view)
{
    return view ? ((struct pvk_image_view *)view)->mtl_texture : NULL;
}

static int close4(const float *got, float r, float g, float b, float a)
{
    float eps = 1.0f / 512.0f;
    if (fabsf(got[0] - r) > eps || fabsf(got[1] - g) > eps || fabsf(got[2] - b) > eps || fabsf(got[3] - a) > eps) {
        fprintf(stderr, "  got (%g %g %g %g), want (%g %g %g %g)\n", got[0], got[1], got[2], got[3], r, g, b, a);
        return 0;
    }
    return 1;
}

static VkSampler make_sampler(VkFilter filter, VkSamplerAddressMode mode, VkBorderColor border, float bias, float min_lod,
                              float max_lod, VkBool32 unnormalized)
{
    VkSamplerCreateInfo si;
    VkSampler s = VK_NULL_HANDLE;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = si.minFilter = filter;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = mode;
    si.mipLodBias = bias;
    si.minLod = min_lod;
    si.maxLod = max_lod;
    si.borderColor = border;
    si.unnormalizedCoordinates = unnormalized;
    if (vkCreateSampler(device, &si, NULL, &s) != VK_SUCCESS)
        PVK_TEST_FAIL("vkCreateSampler failed");
    return s;
}

/* View swizzles, the format swizzles of section 7, and BC1_RGB's alpha. */
static void test_view_swizzle(void)
{
    uint8_t px[4] = { 10, 20, 30, 40 };
    float coords[4] = { 0.5f, 0.5f, 0, 0 }, out[4];
    VkComponentMapping m;
    struct img a;
    VkImageView v;

    a = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, 1, 1, px, 4);
    m.r = VK_COMPONENT_SWIZZLE_B;
    m.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    m.b = VK_COMPONENT_SWIZZLE_ZERO;
    m.a = VK_COMPONENT_SWIZZLE_ONE;
    v = make_view(a.image, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1, m);
    CHECK(harness("k2d_read", view_texture(v), VK_NULL_HANDLE, coords, 1, out) &&
          close4(out, 30 / 255.0f, 20 / 255.0f, 0, 1));
    vkDestroyImageView(device, v, NULL);
    free_image(&a);

    {
        static const struct {
            VkFormat format;
            uint16_t texel;
            float r, g, b, a;
        } f[] = {
            { VK_FORMAT_B4G4R4A4_UNORM_PACK16, 0x1234, 3 / 15.0f, 2 / 15.0f, 1 / 15.0f, 4 / 15.0f },
            { VK_FORMAT_R4G4B4A4_UNORM_PACK16, 0x1234, 1 / 15.0f, 2 / 15.0f, 3 / 15.0f, 4 / 15.0f },
            { VK_FORMAT_A4R4G4B4_UNORM_PACK16, 0x1234, 2 / 15.0f, 3 / 15.0f, 4 / 15.0f, 1 / 15.0f },
            { VK_FORMAT_A4B4G4R4_UNORM_PACK16, 0x1234, 4 / 15.0f, 3 / 15.0f, 2 / 15.0f, 1 / 15.0f },
            { VK_FORMAT_B5G6R5_UNORM_PACK16, (3u << 11) | (5u << 5) | 7u, 7 / 31.0f, 5 / 63.0f, 3 / 31.0f, 1 },
            { VK_FORMAT_R5G6B5_UNORM_PACK16, (3u << 11) | (5u << 5) | 7u, 3 / 31.0f, 5 / 63.0f, 7 / 31.0f, 1 },
            { VK_FORMAT_A1B5G5R5_UNORM_PACK16, 0x8000u | (3u << 10) | (5u << 5) | 7u, 7 / 31.0f, 5 / 31.0f, 3 / 31.0f, 1 },
            { VK_FORMAT_B5G5R5A1_UNORM_PACK16, (3u << 11) | (5u << 6) | (7u << 1), 7 / 31.0f, 5 / 31.0f, 3 / 31.0f, 0 },
            { VK_FORMAT_A1R5G5B5_UNORM_PACK16, (3u << 10) | (5u << 5) | 7u, 3 / 31.0f, 5 / 31.0f, 7 / 31.0f, 0 },
        };
        unsigned i;
        for (i = 0; i < PVK_ARRAY_SIZE(f); i++) {
            a = image_with(VK_IMAGE_TYPE_2D, f[i].format, 1, 1, 1, 1, 1, &f[i].texel, 2);
            v = make_view(a.image, VK_IMAGE_VIEW_TYPE_2D, f[i].format, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                          identity_mapping());
            if (!harness("k2d_read", view_texture(v), VK_NULL_HANDLE, coords, 1, out) ||
                !close4(out, f[i].r, f[i].g, f[i].b, f[i].a))
                PVK_TEST_FAIL("format %d", (int)f[i].format);
            vkDestroyImageView(device, v, NULL);
            free_image(&a);
        }
    }

    /* BC1 in 3-color mode (color0 <= color1), index 3: transparent black;
     * BC1_RGB reads alpha one. */
    if (((struct pvk_physical_device *)pd)->bc_formats) {
        uint8_t block[8] = { 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
        a = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_BC1_RGB_UNORM_BLOCK, 4, 4, 1, 1, 1, block, 8);
        v = make_view(a.image, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                      identity_mapping());
        CHECK(harness("k2d_read", view_texture(v), VK_NULL_HANDLE, coords, 1, out) && close4(out, 0, 0, 0, 1));
        vkDestroyImageView(device, v, NULL);
        free_image(&a);
        a = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 4, 4, 1, 1, 1, block, 8);
        v = make_view(a.image, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                      identity_mapping());
        CHECK(harness("k2d_read", view_texture(v), VK_NULL_HANDLE, coords, 1, out) && close4(out, 0, 0, 0, 0));
        vkDestroyImageView(device, v, NULL);
        free_image(&a);
    }
}

/* View types and subresource ranges. */
static void test_view_types(void)
{
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkSampler nearest = make_sampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                     VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, 0, 0, VK_LOD_CLAMP_NONE, VK_FALSE);
    VkImageSubresourceRange range;
    VkClearColorValue color;
    VkImageCreateInfo info;
    VkCommandBuffer cb;
    VkImageView v;
    struct img a;
    float out[4 * 6];
    unsigned i;

    /* 1D, as a 2D texture of height 1, with mipmaps */
    {
        uint8_t px[8 * 4];
        float c[2 * 4] = { 0.3f, 0.5f, 0, 0, 0.9f, 0.5f, 0, 0 };
        for (i = 0; i < 32; i++)
            px[i] = (uint8_t)(i * 8);
        a = image_with(VK_IMAGE_TYPE_1D, VK_FORMAT_R8G8B8A8_UNORM, 8, 1, 1, 4, 1, px, sizeof(px));
        v = make_view(a.image, VK_IMAGE_VIEW_TYPE_1D, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 0, 4, 0, 1,
                      identity_mapping());
        CHECK(msg_ulong(view_texture(v), "textureType") == PVK_MTLTextureType2D &&
              msg_ulong(view_texture(v), "height") == 1 && msg_ulong(view_texture(v), "mipmapLevelCount") == 4);
        CHECK(harness("k2d", view_texture(v), nearest, c, 2, out) &&
              close4(out, px[8] / 255.0f, px[9] / 255.0f, px[10] / 255.0f, px[11] / 255.0f) &&
              close4(out + 4, px[28] / 255.0f, px[29] / 255.0f, px[30] / 255.0f, px[31] / 255.0f));
        vkDestroyImageView(device, v, NULL);
        v = make_view(a.image, VK_IMAGE_VIEW_TYPE_1D_ARRAY, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 1, 2, 0, 1,
                      identity_mapping());
        CHECK(msg_ulong(view_texture(v), "textureType") == PVK_MTLTextureType2DArray &&
              msg_ulong(view_texture(v), "mipmapLevelCount") == 2);
        vkDestroyImageView(device, v, NULL);
        free_image(&a);
    }

    /* cube and cube array: faces cleared to their index */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, 4, 4, 1, 1, 12, usage);
    info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    a = make_image(&info);
    cb = begin();
    memset(&color, 0, sizeof(color));
    for (i = 0; i < 12; i++) {
        color.float32[0] = (float)i;
        clear_range(&range, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, i, 1);
        vkCmdClearColorImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
    }
    CHECK_VK(finish(cb), VK_SUCCESS);
    {
        float dirs[6 * 4] = { 1, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0 };
        v = make_view(a.image, VK_IMAGE_VIEW_TYPE_CUBE, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 6, 6,
                      identity_mapping());
        CHECK(msg_ulong(view_texture(v), "textureType") == PVK_MTLTextureTypeCube);
        CHECK(harness("kcube", view_texture(v), nearest, dirs, 6, out));
        for (i = 0; i < 6; i++)
            CHECK(out[i * 4] == (float)(6 + i));
        vkDestroyImageView(device, v, NULL);
        for (i = 0; i < 6; i++)
            dirs[i * 4 + 3] = 1; /* the second cube */
        v = make_view(a.image, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 12,
                      identity_mapping());
        CHECK(harness("kcubearray", view_texture(v), nearest, dirs, 6, out));
        for (i = 0; i < 6; i++)
            CHECK(out[i * 4] == (float)(6 + i));
        vkDestroyImageView(device, v, NULL);
        /* a 2D array view of layers 3-4, a 2D view of layer 11 */
        {
            float c[2 * 4] = { 0.5f, 0.5f, 0, 0, 0.5f, 0.5f, 1, 0 };
            v = make_view(a.image, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 3,
                          2, identity_mapping());
            CHECK(harness("k2darray", view_texture(v), nearest, c, 2, out) && out[0] == 3.0f && out[4] == 4.0f);
            vkDestroyImageView(device, v, NULL);
            v = make_view(a.image, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 11, 1,
                          identity_mapping());
            CHECK(harness("k2d", view_texture(v), nearest, c, 1, out) && out[0] == 11.0f);
            vkDestroyImageView(device, v, NULL);
        }
    }
    free_image(&a);

    /* mip ranges: level 0 of a view of levels 1.. is the image's level 1 */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, 8, 8, 1, 4, 1, usage);
    a = make_image(&info);
    cb = begin();
    for (i = 0; i < 4; i++) {
        color.float32[0] = (float)(10 + i);
        clear_range(&range, VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1);
        vkCmdClearColorImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
    }
    CHECK_VK(finish(cb), VK_SUCCESS);
    {
        float c[3 * 4] = { 0.5f, 0.5f, 0, 0, 0.5f, 0.5f, 0, 1, 0.5f, 0.5f, 0, 2 };
        v = make_view(a.image, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 1, 3, 0, 1,
                      identity_mapping());
        CHECK(harness("k2d", view_texture(v), nearest, c, 3, out) && out[0] == 11.0f && out[4] == 12.0f &&
              out[8] == 13.0f);
        vkDestroyImageView(device, v, NULL);
    }
    free_image(&a);

    /* 3D, and a 2D view of a 3D image (2D_ARRAY_COMPATIBLE): an attachment
     * view without a texture of its own */
    {
        float f[2 * 2 * 4], c[2 * 4] = { 0.25f, 0.25f, 0.125f, 0, 0.75f, 0.75f, 0.875f, 0 };
        for (i = 0; i < 16; i++)
            f[i] = (float)i;
        info = image_info(VK_IMAGE_TYPE_3D, VK_FORMAT_R32_SFLOAT, 2, 2, 4, 1, 1, usage | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        info.flags = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
        a = make_image(&info);
        {
            struct buf b = make_buf(sizeof(f), shared_type, 0);
            VkBufferImageCopy r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 2, 2, 4);
            memcpy(b.data, f, sizeof(f));
            upload(a.image, &r, &b);
            free_buf(&b);
        }
        v = make_view(a.image, VK_IMAGE_VIEW_TYPE_3D, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                      identity_mapping());
        CHECK(harness("k3d", view_texture(v), nearest, c, 2, out) && out[0] == 0.0f && out[4] == 15.0f);
        vkDestroyImageView(device, v, NULL);
        v = make_view(a.image, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 1, 2,
                      identity_mapping());
        CHECK(v && ((struct pvk_image_view *)v)->depth_plane == 1 &&
              msg_ulong(view_texture(v), "textureType") == PVK_MTLTextureType3D);
        vkDestroyImageView(device, v, NULL);
        free_image(&a);
    }
    vkDestroySampler(device, nearest, NULL);
}

/* Depth and stencil aspect views of a combined format; comparison samplers. */
static void test_depth_stencil_views(void)
{
    VkImageCreateInfo info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_D32_SFLOAT_S8_UINT, 4, 4, 1, 1, 1,
                                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    struct img a = make_image(&info);
    VkImageSubresourceRange range;
    VkClearDepthStencilValue v;
    VkSamplerCreateInfo si;
    VkSampler less, greater;
    VkImageView dv, sv;
    VkCommandBuffer cb;
    float c[2 * 4] = { 0.5f, 0.5f, 0.25f, 0, 0.5f, 0.5f, 0.75f, 0 }, out[8];
    float r[4] = { 1, 1, 0, 0 };

    v.depth = 0.5f;
    v.stencil = 77;
    clear_range(&range, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1);
    cb = begin();
    vkCmdClearDepthStencilImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
    CHECK_VK(finish(cb), VK_SUCCESS);
    dv = make_view(a.image, VK_IMAGE_VIEW_TYPE_2D, info.format, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1, identity_mapping());
    sv = make_view(a.image, VK_IMAGE_VIEW_TYPE_2D, info.format, VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1, identity_mapping());
    CHECK(msg_ulong(view_texture(sv), "pixelFormat") == PVK_MTLPixelFormatX32_Stencil8);
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.compareEnable = VK_TRUE;
    si.compareOp = VK_COMPARE_OP_LESS; /* reference < texel passes */
    si.maxLod = VK_LOD_CLAMP_NONE;
    CHECK_VK(vkCreateSampler(device, &si, NULL, &less), VK_SUCCESS);
    si.compareOp = VK_COMPARE_OP_GREATER;
    CHECK_VK(vkCreateSampler(device, &si, NULL, &greater), VK_SUCCESS);
    CHECK(harness("kdepth", view_texture(dv), less, c, 2, out) && out[0] == 1.0f && out[4] == 0.0f);
    CHECK(harness("kdepth", view_texture(dv), greater, c, 2, out) && out[0] == 0.0f && out[4] == 1.0f);
    CHECK(harness("kdepth_read", view_texture(dv), VK_NULL_HANDLE, c, 1, out) && out[0] == 0.5f);
    CHECK(harness("k2d_uint", view_texture(sv), VK_NULL_HANDLE, r, 1, out) && out[0] == 77.0f && out[1] == 0.0f &&
          out[3] == 1.0f);
    /* read through a color type: (D, 0, 0, 1) as in Vulkan, not Metal's (D, D, D, 1) */
    CHECK(harness("k2d_read", view_texture(dv), VK_NULL_HANDLE, r, 1, out) && close4(out, 0.5f, 0, 0, 1));
    vkDestroySampler(device, less, NULL);
    vkDestroySampler(device, greater, NULL);
    vkDestroyImageView(device, dv, NULL);
    vkDestroyImageView(device, sv, NULL);
    free_image(&a);
}

/* A view of all of an image in its format, texture type and channels is
 * the image's texture, also as an attachment, and outlives the image; a
 * view of a level, of a layer, of another type or with a channel mapping
 * gets one of its own, as does a depth view shaders may read (it carries
 * (D, 0, 0, 1)). A depth view no shader reads carries no mapping: it is
 * its image's texture too. */
static void test_view_reuse(void)
{
    VkImageCreateInfo ci = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 3, 2,
                                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    VkImageCreateInfo si = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_D32_SFLOAT, 4, 4, 1, 1, 1,
                                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    VkImageCreateInfo ai = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_D32_SFLOAT, 4, 4, 1, 1, 1,
                                      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    struct img c = make_image(&ci), s = make_image(&si), a = make_image(&ai);
    VkComponentMapping rgba = identity_mapping(), swap = identity_mapping();
    VkImageView whole, explicit_rgba, level, layer, swapped, depth, attachment;
    id texture = texture_of(c.image);

    rgba.r = VK_COMPONENT_SWIZZLE_R;
    rgba.g = VK_COMPONENT_SWIZZLE_G;
    rgba.b = VK_COMPONENT_SWIZZLE_B;
    rgba.a = VK_COMPONENT_SWIZZLE_A;
    swap.r = VK_COMPONENT_SWIZZLE_B;
    swap.b = VK_COMPONENT_SWIZZLE_R;
    whole = make_view(c.image, VK_IMAGE_VIEW_TYPE_2D_ARRAY, ci.format, VK_IMAGE_ASPECT_COLOR_BIT, 0, 3, 0, 2,
                      identity_mapping());
    explicit_rgba = make_view(c.image, VK_IMAGE_VIEW_TYPE_2D_ARRAY, ci.format, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                              VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS, rgba);
    level = make_view(c.image, VK_IMAGE_VIEW_TYPE_2D_ARRAY, ci.format, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 0, 2,
                      identity_mapping());
    layer = make_view(c.image, VK_IMAGE_VIEW_TYPE_2D, ci.format, VK_IMAGE_ASPECT_COLOR_BIT, 0, 3, 1, 1,
                      identity_mapping());
    swapped = make_view(c.image, VK_IMAGE_VIEW_TYPE_2D_ARRAY, ci.format, VK_IMAGE_ASPECT_COLOR_BIT, 0, 3, 0, 2, swap);
    depth = make_view(s.image, VK_IMAGE_VIEW_TYPE_2D, si.format, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                      identity_mapping());
    attachment = make_view(a.image, VK_IMAGE_VIEW_TYPE_2D, ai.format, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                           identity_mapping());
    REQUIRE(texture && whole && explicit_rgba && level && layer && swapped && depth && attachment);
    CHECK(view_texture(whole) == texture && ((struct pvk_image_view *)whole)->mtl_attachment == texture);
    CHECK(view_texture(explicit_rgba) == texture);
    CHECK(view_texture(level) != texture && view_texture(layer) != texture && view_texture(swapped) != texture);
    CHECK(view_texture(depth) != texture_of(s.image));
    CHECK(view_texture(attachment) == texture_of(a.image) &&
          ((struct pvk_image_view *)attachment)->mtl_attachment == texture_of(a.image));
    free_image(&c); /* the views keep the texture */
    CHECK(msg_ulong(view_texture(whole), "arrayLength") == 2 &&
          msg_ulong(view_texture(whole), "mipmapLevelCount") == 3);
    vkDestroyImageView(device, whole, NULL);
    vkDestroyImageView(device, explicit_rgba, NULL);
    vkDestroyImageView(device, level, NULL);
    vkDestroyImageView(device, layer, NULL);
    vkDestroyImageView(device, swapped, NULL);
    vkDestroyImageView(device, depth, NULL);
    vkDestroyImageView(device, attachment, NULL);
    free_image(&s);
    free_image(&a);
}

/* Sampler parameters as Metal applies them. */
static void test_samplers(void)
{
    float px[2 * 2 * 4] = { 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1 }; /* R G / B W */
    struct img a = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, 2, 2, 1, 1, 1, px, sizeof(px));
    VkImageView v = make_view(a.image, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                              0, 1, identity_mapping());
    float c[4] = { -0.75f, 0.25f, 0, 0 }, out[8];
    VkSampler s, s2;
    struct {
        VkSamplerAddressMode mode;
        int texel; /* -1: border */
    } modes[] = {
        { VK_SAMPLER_ADDRESS_MODE_REPEAT, 0 },          /* -0.75 -> 0.25 */
        { VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT, 1 }, /* -> 0.75 */
        { VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0 },
        { VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE, 1 },
        { VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, -1 },
    };
    unsigned i;

    for (i = 0; i < PVK_ARRAY_SIZE(modes); i++) {
        s = make_sampler(VK_FILTER_NEAREST, modes[i].mode, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK, 0, 0, 0, VK_FALSE);
        CHECK(harness("k2d", view_texture(v), s, c, 1, out));
        if (modes[i].texel < 0)
            CHECK(close4(out, 0, 0, 0, 1));
        else
            CHECK(close4(out, px[modes[i].texel * 4], px[modes[i].texel * 4 + 1], px[modes[i].texel * 4 + 2], 1));
        vkDestroySampler(device, s, NULL);
    }
    s = make_sampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, 0,
                     0, 0, VK_FALSE);
    CHECK(harness("k2d", view_texture(v), s, c, 1, out) && close4(out, 0, 0, 0, 0));
    vkDestroySampler(device, s, NULL);
    s = make_sampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE, 0, 0, 0,
                     VK_FALSE);
    CHECK(harness("k2d", view_texture(v), s, c, 1, out) && close4(out, 1, 1, 1, 1));
    vkDestroySampler(device, s, NULL);
    /* linear filtering at the center: the mean of the four texels */
    c[0] = c[1] = 0.5f;
    s = make_sampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK, 0, 0, 0,
                     VK_FALSE);
    CHECK(harness("k2d", view_texture(v), s, c, 1, out) && close4(out, 0.5f, 0.5f, 0.5f, 1));
    vkDestroySampler(device, s, NULL);
    /* unnormalized coordinates */
    c[0] = 1.5f;
    c[1] = 0.5f;
    s = make_sampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK, 0, 0, 0,
                     VK_TRUE);
    CHECK(harness("k2d", view_texture(v), s, c, 1, out) && close4(out, 0, 1, 0, 1));
    vkDestroySampler(device, s, NULL);
    vkDestroyImageView(device, v, NULL);
    free_image(&a);

    /* LOD bias moves explicit LODs too (Vulkan: lod = explicit + bias) */
    {
        VkImageCreateInfo info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, 4, 4, 1, 3, 1,
                                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        VkImageSubresourceRange range;
        VkClearColorValue color;
        VkCommandBuffer cb;
        float l[2 * 4] = { 0.5f, 0.5f, 0, 0, 0.5f, 0.5f, 0, 1 };
        a = make_image(&info);
        cb = begin();
        memset(&color, 0, sizeof(color));
        for (i = 0; i < 3; i++) {
            color.float32[0] = (float)i;
            clear_range(&range, VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1);
            vkCmdClearColorImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
        }
        CHECK_VK(finish(cb), VK_SUCCESS);
        v = make_view(a.image, VK_IMAGE_VIEW_TYPE_2D, info.format, VK_IMAGE_ASPECT_COLOR_BIT, 0, 3, 0, 1, identity_mapping());
        s = make_sampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK, 1.0f, 0,
                         VK_LOD_CLAMP_NONE, VK_FALSE);
        CHECK(harness("k2d", view_texture(v), s, l, 2, out) && out[0] == 1.0f && out[4] == 2.0f);
        vkDestroySampler(device, s, NULL);
        /* minLod/maxLod clamp explicit LODs too (Metal's header says they do
         * not; M4 Max does, as Vulkan requires) */
        s = make_sampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK, 0, 1.0f,
                         1.0f, VK_FALSE);
        l[4 + 3] = 2;
        CHECK(harness("k2d", view_texture(v), s, l, 2, out) && out[0] == 1.0f && out[4] == 1.0f);
        vkDestroySampler(device, s, NULL);
        vkDestroyImageView(device, v, NULL);
        free_image(&a);
    }

    /* equal samplers share one MTLSamplerState; anisotropy and compare build */
    s = make_sampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_BORDER_COLOR_INT_OPAQUE_WHITE, 0.5f, 0, 4, VK_FALSE);
    s2 = make_sampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_BORDER_COLOR_INT_OPAQUE_WHITE, 0.5f, 0, 4, VK_FALSE);
    CHECK(s && s2 && ((struct pvk_sampler *)s)->mtl_sampler == ((struct pvk_sampler *)s2)->mtl_sampler);
    vkDestroySampler(device, s, NULL);
    CHECK(((struct pvk_sampler *)s2)->mtl_sampler && ((struct pvk_sampler *)s2)->state->refs == 1);
    {
        VkSamplerCreateInfo si;
        VkSamplerReductionModeCreateInfo red;
        memset(&si, 0, sizeof(si));
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.anisotropyEnable = VK_TRUE;
        si.maxAnisotropy = 16.0f;
        si.compareEnable = VK_TRUE;
        si.compareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
        si.maxLod = VK_LOD_CLAMP_NONE;
        si.mipLodBias = 100.0f; /* beyond maxSamplerLodBias: clamped */
        CHECK_VK(vkCreateSampler(device, &si, NULL, &s), VK_SUCCESS);
        CHECK(((struct pvk_sampler *)s)->mtl_sampler != ((struct pvk_sampler *)s2)->mtl_sampler);
        vkDestroySampler(device, s, NULL);
        if (((struct pvk_physical_device *)pd)->filter_minmax) {
            memset(&red, 0, sizeof(red));
            red.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO;
            red.reductionMode = VK_SAMPLER_REDUCTION_MODE_MIN;
            si.pNext = &red;
            si.anisotropyEnable = VK_FALSE;
            si.compareEnable = VK_FALSE;
            CHECK_VK(vkCreateSampler(device, &si, NULL, &s), VK_SUCCESS);
            vkDestroySampler(device, s, NULL);
        }
    }
    vkDestroySampler(device, s2, NULL);
}

/* Metal sampler behaviors worked around: its opaque black border reads
 * (0, 0, 0, 0) from A8Unorm, so A8_UNORM views take the sampler's opaque
 * white twin, (0, 0, 0, 1) there; R5G5B5A1, where no Metal border reads
 * opaque black, is not sampled; and Metal picks the filter by the LOD before
 * minLod and maxLod clamp it, so a range on one side of 0 takes one filter. */
static void test_sampler_quirks(void)
{
    uint8_t a8[4] = { 64, 64, 64, 64 };
    float px[2 * 2 * 4] = { 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1 }; /* R G / B W */
    struct img a = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_A8_UNORM, 2, 2, 1, 1, 1, a8, sizeof(a8));
    struct img b = image_with(VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, 2, 2, 1, 1, 1, px, sizeof(px));
    VkImageView va = make_view(a.image, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                               identity_mapping());
    VkImageView vb = make_view(b.image, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                               1, 0, 1, identity_mapping());
    VkSampler s = make_sampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
                               0, 0, 0, VK_FALSE);
    float c[4] = { -0.75f, 0.25f, 0, 0 }, out[8];
    struct pvk_sampler twin;
    VkFormatProperties fp;
    VkSamplerCreateInfo si;

    twin = *(struct pvk_sampler *)s;
    twin.mtl_sampler = twin.mtl_alpha;
    CHECK(twin.mtl_alpha && twin.mtl_alpha != ((struct pvk_sampler *)s)->mtl_sampler);
    CHECK(harness("k2d", view_texture(va), (VkSampler)&twin, c, 1, out) && close4(out, 0, 0, 0, 1));
    vkDestroySampler(device, s, NULL);

    vkGetPhysicalDeviceFormatProperties(pd, VK_FORMAT_R5G5B5A1_UNORM_PACK16, &fp);
    CHECK(!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
          (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT));

    /* minFilter LINEAR, magFilter NEAREST at the center, level 0: minLod
     * 0.01 minifies (the mean), maxLod 0 magnifies even at level 2 (W) */
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.minFilter = VK_FILTER_LINEAR;
    si.magFilter = VK_FILTER_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.minLod = 0.01f;
    si.maxLod = VK_LOD_CLAMP_NONE;
    CHECK_VK(vkCreateSampler(device, &si, NULL, &s), VK_SUCCESS);
    c[0] = c[1] = 0.5f;
    CHECK(harness("k2d", view_texture(vb), s, c, 1, out) && close4(out, 0.5f, 0.5f, 0.5f, 1));
    vkDestroySampler(device, s, NULL);
    si.minLod = si.maxLod = 0.0f;
    CHECK_VK(vkCreateSampler(device, &si, NULL, &s), VK_SUCCESS);
    c[3] = 2.0f;
    CHECK(harness("k2d", view_texture(vb), s, c, 1, out) && close4(out, 1, 1, 1, 1));
    vkDestroySampler(device, s, NULL);
    vkDestroyImageView(device, va, NULL);
    vkDestroyImageView(device, vb, NULL);
    free_image(&a);
    free_image(&b);
}

/* Texel buffer views: texture buffers at the view's offset. */
static void test_buffer_views(void)
{
    struct buf b = make_buf(4096, shared_type,
                            VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT);
    VkBufferViewCreateInfo bi;
    VkBufferUsageFlags2CreateInfo usage2;
    VkBufferView v;
    float c[3 * 4] = { 0, 0, 0, 0, 1, 0, 0, 0, 15, 0, 0, 0 }, out[12];
    unsigned i;

    REQUIRE(b.data);
    for (i = 0; i < 4096; i++)
        b.data[i] = (unsigned char)i;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
    bi.buffer = b.buffer;
    bi.format = VK_FORMAT_R8G8B8A8_UNORM;
    bi.offset = 64;
    bi.range = 64;
    REQUIRE_VK(vkCreateBufferView(device, &bi, NULL, &v));
    CHECK(msg_ulong(((struct pvk_buffer_view *)v)->mtl_texture, "textureType") == PVK_MTLTextureTypeTextureBuffer &&
          msg_ulong(((struct pvk_buffer_view *)v)->mtl_texture, "width") == 16);
    CHECK(harness("kbuffer", ((struct pvk_buffer_view *)v)->mtl_texture, VK_NULL_HANDLE, c, 3, out) &&
          close4(out, 64 / 255.0f, 65 / 255.0f, 66 / 255.0f, 67 / 255.0f) &&
          close4(out + 4, 68 / 255.0f, 69 / 255.0f, 70 / 255.0f, 71 / 255.0f) &&
          close4(out + 8, 124 / 255.0f, 125 / 255.0f, 126 / 255.0f, 127 / 255.0f));
    vkDestroyBufferView(device, v, NULL);
    bi.format = VK_FORMAT_R32_UINT; /* atomics-capable */
    bi.offset = 1024;
    bi.range = VK_WHOLE_SIZE;
    REQUIRE_VK(vkCreateBufferView(device, &bi, NULL, &v));
    CHECK(msg_ulong(((struct pvk_buffer_view *)v)->mtl_texture, "width") == (4096 - 1024) / 4);
    CHECK(harness("kbuffer_uint", ((struct pvk_buffer_view *)v)->mtl_texture, VK_NULL_HANDLE, c, 2, out) &&
          out[0] == (float)0x03020100u && out[4] == (float)0x07060504u);
    vkDestroyBufferView(device, v, NULL);
    bi.format = VK_FORMAT_B8G8R8A8_UNORM; /* uniform texel buffers only: the view's usage says so */
    bi.offset = 0;
    bi.range = 256;
    memset(&usage2, 0, sizeof(usage2));
    usage2.sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO;
    usage2.usage = VK_BUFFER_USAGE_2_UNIFORM_TEXEL_BUFFER_BIT;
    bi.pNext = &usage2;
    REQUIRE_VK(vkCreateBufferView(device, &bi, NULL, &v));
    CHECK(harness("kbuffer", ((struct pvk_buffer_view *)v)->mtl_texture, VK_NULL_HANDLE, c, 1, out) &&
          close4(out, 2 / 255.0f, 1 / 255.0f, 0, 3 / 255.0f));
    vkDestroyBufferView(device, v, NULL);
    free_buf(&b);
}

/* ---- Host image copy ---- */

static void test_host_copy(void)
{
    VkImageUsageFlags usage = VK_IMAGE_USAGE_HOST_TRANSFER_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    unsigned char *in = malloc(1 << 16), *out = malloc(1 << 16);
    VkImageCreateInfo info;
    VkMemoryToImageCopy up;
    VkImageToMemoryCopy down;
    VkCopyMemoryToImageInfo ui;
    VkCopyImageToMemoryInfo di;
    VkHostImageLayoutTransitionInfo tr;
    struct img a, b;
    uint32_t y, l, bad;

    REQUIRE(in && out);
    fill_pattern(in, 1 << 16, 9);
    memset(&up, 0, sizeof(up));
    up.sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY;
    memset(&down, 0, sizeof(down));
    down.sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY;
    memset(&ui, 0, sizeof(ui));
    ui.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO;
    ui.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ui.regionCount = 1;
    ui.pRegions = &up;
    memset(&di, 0, sizeof(di));
    di.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO;
    di.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
    di.regionCount = 1;
    di.pRegions = &down;

    /* a sub-rectangle of level 1 of layers 1-2, padded rows; then tightly */
    info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 32, 32, 1, 3, 3, usage);
    a = make_image_in(&info, shared_type);
    memset(&tr, 0, sizeof(tr));
    tr.sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO;
    tr.image = a.image;
    tr.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    tr.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    tr.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    tr.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    tr.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
    CHECK_VK(vkTransitionImageLayout(device, 1, &tr), VK_SUCCESS);
    up.pHostPointer = in;
    up.memoryRowLength = 12;
    up.memoryImageHeight = 8;
    up.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    up.imageSubresource.mipLevel = 1;
    up.imageSubresource.baseArrayLayer = 1;
    up.imageSubresource.layerCount = 2;
    up.imageOffset.x = 3;
    up.imageOffset.y = 5;
    up.imageExtent.width = 10;
    up.imageExtent.height = 7;
    up.imageExtent.depth = 1;
    ui.dstImage = a.image;
    CHECK_VK(vkCopyMemoryToImage(device, &ui), VK_SUCCESS);
    down.pHostPointer = out;
    down.imageSubresource = up.imageSubresource;
    down.imageOffset = up.imageOffset;
    down.imageExtent = up.imageExtent;
    di.srcImage = a.image;
    CHECK_VK(vkCopyImageToMemory(device, &di), VK_SUCCESS);
    for (bad = 0, l = 0; l < 2; l++)
        for (y = 0; y < 7; y++)
            if (memcmp(out + ((l * 7 + y) * 10) * 4, in + ((l * 8 + y) * 12) * 4, 40))
                bad++;
    CHECK(bad == 0);

    /* the GPU sees host-written texels, and the host sees GPU-written ones */
    {
        struct buf g = make_buf(4096, shared_type, 0);
        VkBufferImageCopy r = region(VK_IMAGE_ASPECT_COLOR_BIT, 1, 2, 1, 3, 5, 0, 10, 7, 1);
        VkImageSubresourceRange range;
        VkClearColorValue color;
        VkCommandBuffer cb;
        download(a.image, &r, &g);
        for (bad = 0, y = 0; y < 7; y++)
            if (memcmp(g.data + y * 40, in + ((8 + y) * 12) * 4, 40))
                bad++;
        CHECK(bad == 0);
        memset(&color, 0, sizeof(color));
        color.float32[1] = 1.0f;
        clear_range(&range, VK_IMAGE_ASPECT_COLOR_BIT, 2, 1, 0, 1);
        cb = begin();
        vkCmdClearColorImage(cb, a.image, VK_IMAGE_LAYOUT_GENERAL, &color, 1, &range);
        barrier(cb, a.image);
        CHECK_VK(finish(cb), VK_SUCCESS);
        free_buf(&g);
    }
    /* MEMCPY: the whole subresource, tightly packed (VkSubresourceHostMemcpySize) */
    {
        VkImageSubresource2 sub;
        VkSubresourceHostMemcpySize size;
        VkSubresourceLayout2 layout;
        uint8_t green[4] = { 0, 0xff, 0, 0 };
        memset(&sub, 0, sizeof(sub));
        sub.sType = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2;
        sub.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        sub.imageSubresource.mipLevel = 2;
        memset(&size, 0, sizeof(size));
        size.sType = VK_STRUCTURE_TYPE_SUBRESOURCE_HOST_MEMCPY_SIZE;
        memset(&layout, 0, sizeof(layout));
        layout.sType = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2;
        layout.pNext = &size;
        vkGetImageSubresourceLayout2(device, a.image, &sub, &layout);
        CHECK(size.size == 8 * 8 * 4);
        memset(&down, 0, sizeof(down));
        down.sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY;
        down.pHostPointer = out;
        down.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        down.imageSubresource.mipLevel = 2;
        down.imageSubresource.layerCount = 1;
        down.imageExtent.width = down.imageExtent.height = 8;
        down.imageExtent.depth = 1;
        di.flags = VK_HOST_IMAGE_COPY_MEMCPY_BIT;
        CHECK_VK(vkCopyImageToMemory(device, &di), VK_SUCCESS);
        CHECK(all_texels(out, 64, green, 4));
        di.flags = 0;
    }
    /* image to image, into a second image */
    b = make_image_in(&info, cached_type);
    {
        VkCopyImageToImageInfo ii;
        VkImageCopy2 c;
        memset(&c, 0, sizeof(c));
        c.sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2;
        c.srcSubresource = up.imageSubresource;
        c.dstSubresource = up.imageSubresource;
        c.dstSubresource.mipLevel = 0;
        c.srcOffset = up.imageOffset;
        c.dstOffset.x = 20;
        c.dstOffset.y = 1;
        c.extent = up.imageExtent;
        memset(&ii, 0, sizeof(ii));
        ii.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_IMAGE_INFO;
        ii.srcImage = a.image;
        ii.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ii.dstImage = b.image;
        ii.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ii.regionCount = 1;
        ii.pRegions = &c;
        CHECK_VK(vkCopyImageToImage(device, &ii), VK_SUCCESS);
        memset(&down, 0, sizeof(down));
        down.sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY;
        down.pHostPointer = out;
        down.imageSubresource = c.dstSubresource;
        down.imageOffset = c.dstOffset;
        down.imageExtent = c.extent;
        di.srcImage = b.image;
        CHECK_VK(vkCopyImageToMemory(device, &di), VK_SUCCESS);
        for (bad = 0, l = 0; l < 2; l++)
            for (y = 0; y < 7; y++)
                if (memcmp(out + ((l * 7 + y) * 10) * 4, in + ((l * 8 + y) * 12) * 4, 40))
                    bad++;
        CHECK(bad == 0);
    }
    free_image(&a);
    free_image(&b);

    /* 3D, compressed, and the aspects of a combined depth/stencil format */
    {
        static const struct {
            VkImageType type;
            VkFormat format;
            uint32_t w, h, d, bytes;
            VkImageAspectFlags aspect;
        } k[] = {
            { VK_IMAGE_TYPE_3D, VK_FORMAT_R16G16_SFLOAT, 8, 4, 5, 4 * 8 * 4 * 5, VK_IMAGE_ASPECT_COLOR_BIT },
            { VK_IMAGE_TYPE_2D, VK_FORMAT_ASTC_8x8_UNORM_BLOCK, 24, 16, 1, 3 * 2 * 16, VK_IMAGE_ASPECT_COLOR_BIT },
            { VK_IMAGE_TYPE_2D, VK_FORMAT_D32_SFLOAT_S8_UINT, 8, 8, 1, 64, VK_IMAGE_ASPECT_STENCIL_BIT },
            { VK_IMAGE_TYPE_2D, VK_FORMAT_D16_UNORM, 8, 8, 1, 128, VK_IMAGE_ASPECT_DEPTH_BIT },
            { VK_IMAGE_TYPE_2D, VK_FORMAT_S8_UINT, 8, 8, 1, 64, VK_IMAGE_ASPECT_STENCIL_BIT },
        };
        unsigned i;
        for (i = 0; i < PVK_ARRAY_SIZE(k); i++) {
            info = image_info(k[i].type, k[i].format, k[i].w, k[i].h, k[i].d, 1, 1, usage);
            a = make_image_in(&info, shared_type);
            memset(&up, 0, sizeof(up));
            up.sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY;
            up.pHostPointer = in;
            up.imageSubresource.aspectMask = k[i].aspect;
            up.imageSubresource.layerCount = 1;
            up.imageExtent.width = k[i].w;
            up.imageExtent.height = k[i].h;
            up.imageExtent.depth = k[i].d;
            ui.dstImage = a.image;
            CHECK_VK(vkCopyMemoryToImage(device, &ui), VK_SUCCESS);
            memset(out, 0, k[i].bytes);
            memset(&down, 0, sizeof(down));
            down.sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY;
            down.pHostPointer = out;
            down.imageSubresource = up.imageSubresource;
            down.imageExtent = up.imageExtent;
            di.srcImage = a.image;
            CHECK_VK(vkCopyImageToMemory(device, &di), VK_SUCCESS);
            if (memcmp(in, out, k[i].bytes))
                PVK_TEST_FAIL("host copy of format %d", (int)k[i].format);
            free_image(&a);
        }
    }
    /* image to image, both aspects of a combined depth/stencil format in one region */
    {
        static const VkImageAspectFlags aspects[2] = { VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_STENCIL_BIT };
        float depth[64];
        VkCopyImageToImageInfo ii;
        VkImageCopy2 c;
        unsigned i;
        for (i = 0; i < 64; i++)
            depth[i] = (float)i / 64.0f;
        info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_D32_SFLOAT_S8_UINT, 8, 8, 1, 1, 1, usage);
        a = make_image_in(&info, shared_type);
        b = make_image_in(&info, shared_type);
        for (i = 0; i < 2; i++) {
            memset(&up, 0, sizeof(up));
            up.sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY;
            up.pHostPointer = i ? (void *)in : (void *)depth;
            up.imageSubresource.aspectMask = aspects[i];
            up.imageSubresource.layerCount = 1;
            up.imageExtent.width = up.imageExtent.height = 8;
            up.imageExtent.depth = 1;
            ui.dstImage = a.image;
            CHECK_VK(vkCopyMemoryToImage(device, &ui), VK_SUCCESS);
        }
        memset(&c, 0, sizeof(c));
        c.sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2;
        c.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        c.srcSubresource.layerCount = 1;
        c.dstSubresource = c.srcSubresource;
        c.extent.width = c.extent.height = 8;
        c.extent.depth = 1;
        memset(&ii, 0, sizeof(ii));
        ii.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_IMAGE_INFO;
        ii.srcImage = a.image;
        ii.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ii.dstImage = b.image;
        ii.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ii.regionCount = 1;
        ii.pRegions = &c;
        CHECK_VK(vkCopyImageToImage(device, &ii), VK_SUCCESS);
        for (i = 0; i < 2; i++) {
            memset(out, 0, 256);
            memset(&down, 0, sizeof(down));
            down.sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY;
            down.pHostPointer = out;
            down.imageSubresource.aspectMask = aspects[i];
            down.imageSubresource.layerCount = 1;
            down.imageExtent.width = down.imageExtent.height = 8;
            down.imageExtent.depth = 1;
            di.srcImage = b.image;
            CHECK_VK(vkCopyImageToMemory(device, &di), VK_SUCCESS);
            if (memcmp(i ? (const void *)in : (const void *)depth, out, i ? 64 : 256))
                PVK_TEST_FAIL("image to image host copy: the %s", i ? "stencil" : "depth");
        }
        free_image(&a);
        free_image(&b);
    }
    free(in);
    free(out);
}

/* Texels the host writes into a PREINITIALIZED optimal image's memory read
 * the same through host copies as through the GPU: 32x32 (memory pooled) and
 * 512x512 (memory of its own). */
static void test_host_copy_preinitialized(void)
{
    static const uint32_t sizes[2] = { 32, 512 };
    unsigned char *out = malloc(512 * 512 * 4);
    uint32_t s;

    REQUIRE(out);
    for (s = 0; s < 2; s++) {
        uint32_t n = sizes[s];
        VkImageCreateInfo info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, n, n, 1, 1, 1,
                                            VK_IMAGE_USAGE_HOST_TRANSFER_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        VkBufferImageCopy r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, n, n, 1);
        VkImageToMemoryCopy down;
        VkCopyImageToMemoryInfo di;
        struct buf g = make_buf((VkDeviceSize)n * n * 4, shared_type, 0);
        struct img a;
        void *p;

        info.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
        a = make_image_in(&info, shared_type);
        REQUIRE_VK(vkMapMemory(device, a.memory, 0, VK_WHOLE_SIZE, 0, &p));
        fill_pattern(p, (size_t)n * n * 4, 11);
        vkUnmapMemory(device, a.memory);
        memset(&down, 0, sizeof(down));
        down.sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY;
        down.pHostPointer = out;
        down.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        down.imageSubresource.layerCount = 1;
        down.imageExtent.width = down.imageExtent.height = n;
        down.imageExtent.depth = 1;
        memset(&di, 0, sizeof(di));
        di.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO;
        di.srcImage = a.image;
        di.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        di.regionCount = 1;
        di.pRegions = &down;
        CHECK_VK(vkCopyImageToMemory(device, &di), VK_SUCCESS);
        download(a.image, &r, &g);
        if (memcmp(out, g.data, (size_t)n * n * 4))
            PVK_TEST_FAIL("%ux%u: host copy and GPU copy differ", n, n);
        free_image(&a);
        free_buf(&g);
    }
    free(out);
}

/* ---- Linear images ---- */

static void test_linear(void)
{
    VkImageCreateInfo info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 13, 5, 1, 1, 1,
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    VkDeviceImageSubresourceInfo dinfo;
    VkImageSubresource2 sub2;
    VkSubresourceLayout2 layout2;
    VkImageSubresource sub;
    VkSubresourceLayout layout;
    struct buf out = make_buf(4096, shared_type, 0);
    VkBufferImageCopy r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 13, 5, 1);
    struct img a;
    unsigned char *p;
    uint32_t y;

    info.tiling = VK_IMAGE_TILING_LINEAR;
    info.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
    a = make_image_in(&info, shared_type);
    REQUIRE(texture_of(a.image) && out.data);
    memset(&sub, 0, sizeof(sub));
    sub.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkGetImageSubresourceLayout(device, a.image, &sub, &layout);
    CHECK(layout.rowPitch >= 13 * 4 && layout.rowPitch % 16 == 0 && layout.size >= layout.rowPitch * 5);
    memset(&sub2, 0, sizeof(sub2));
    sub2.sType = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2;
    sub2.imageSubresource = sub;
    memset(&layout2, 0, sizeof(layout2));
    layout2.sType = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2;
    memset(&dinfo, 0, sizeof(dinfo));
    dinfo.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_SUBRESOURCE_INFO;
    dinfo.pCreateInfo = &info;
    dinfo.pSubresource = &sub2;
    vkGetDeviceImageSubresourceLayout(device, &dinfo, &layout2);
    CHECK(memcmp(&layout2.subresourceLayout, &layout, sizeof(layout)) == 0);
    REQUIRE_VK(vkMapMemory(device, a.memory, 0, VK_WHOLE_SIZE, 0, (void **)&p));
    for (y = 0; y < 5; y++)
        fill_pattern(p + layout.offset + y * layout.rowPitch, 13 * 4, y + 1);
    download(a.image, &r, &out);
    for (y = 0; y < 5; y++)
        CHECK(memcmp(out.data + y * 13 * 4, p + layout.offset + y * layout.rowPitch, 13 * 4) == 0);
    free_buf(&out);
    free_image(&a);
}

/* ---- Ordering across encoders ----
 *
 * Clears and blits run in render encoders between compute encoders; a
 * barrier must order all of them, also from one submission to the next. */
static void test_ordering(void)
{
    VkImageUsageFlags xfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageCreateInfo info = image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32_UINT, 256, 256, 1, 1, 1, xfer);
    struct img a = make_image(&info), b = make_image(&info);
    struct buf out = make_buf(256 * 256 * 4, shared_type, 0);
    VkImageSubresourceRange range;
    VkClearColorValue color;
    VkBufferImageCopy r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 256, 256, 1);
    VkImageCopy c = image_copy(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 256, 256, 1);
    VkImageBlit bl = blit_region(0, 0, 0, 0, 0, 256, 256, 1, 0, 0, 0, 0, 0, 256, 256, 1, 1);
    VkCommandBuffer cb;
    unsigned round;

    REQUIRE(out.data);
    memset(&color, 0, sizeof(color));
    clear_range(&range, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1);
    for (round = 1; round <= 20; round++) {
        uint32_t want = round * 1000;
        color.uint32[0] = want;
        cb = begin();
        vkCmdClearColorImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range); /* render */
        barrier(cb, a.image);
        vkCmdCopyImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                       &c); /* compute */
        barrier(cb, b.image);
        color.uint32[0] = 1;
        vkCmdClearColorImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
        barrier(cb, a.image);
        vkCmdBlitImage(cb, b.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, a.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                       &bl, VK_FILTER_NEAREST); /* render */
        barrier(cb, a.image);                    /* the last command: reaches the next submission */
        CHECK_VK(finish(cb), VK_SUCCESS);
        cb = begin();
        vkCmdCopyImageToBuffer(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.buffer, 1, &r);
        barrier(cb, VK_NULL_HANDLE);
        CHECK_VK(finish(cb), VK_SUCCESS);
        if (!all_texels(out.data, 256 * 256, &want, 4)) {
            PVK_TEST_FAIL("round %u", round);
            break;
        }
    }
    free_buf(&out);
    free_image(&a);
    free_image(&b);
}

int main(void)
{
    setup();
    if (!device)
        return 1;
    RUN(test_format_properties);
    RUN(test_image_format_properties);
    RUN(test_all_formats_create);
    RUN(test_create_kinds);
    RUN(test_memory_types);
    RUN(test_copy_buffer_image);
    RUN(test_copy_image);
    RUN(test_clear_color);
    RUN(test_clear_depth_stencil);
    RUN(test_blit);
    RUN(test_resolve);
    RUN(test_view_swizzle);
    RUN(test_view_types);
    RUN(test_depth_stencil_views);
    RUN(test_view_reuse);
    RUN(test_samplers);
    RUN(test_sampler_quirks);
    RUN(test_buffer_views);
    RUN(test_host_copy);
    RUN(test_host_copy_preinitialized);
    RUN(test_linear);
    RUN(test_ordering);
    harness_done();
    teardown();
    return TEST_RESULT();
}
