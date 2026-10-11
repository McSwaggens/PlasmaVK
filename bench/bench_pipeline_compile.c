/* Pipeline compile: the stutter an application sees when it creates
 * pipelines. 256 distinct graphics pipelines (2 vertex shaders x an "uber"
 * fragment shader specialized by material, iteration count and feature bits,
 * x blend/cull/depth state) and 128 distinct compute pipelines (the same
 * paths, specialized by workgroup size too). Each vkCreate*Pipelines call
 * creates one pipeline and is timed on its own; the lines report the
 * distribution (p99 and max are the hitches).
 *
 *   cold_*           fresh device, empty VkPipelineCache. A per-run salt
 *                    (a specialization constant that changes the code) keeps
 *                    caches from earlier runs - the driver's or the OS's -
 *                    from serving any pipeline
 *   warm_cache_*     a second fresh device, the VkPipelineCache initialized
 *                    from the cold run's vkGetPipelineCacheData: the next
 *                    launch of an application that saves its cache
 *   no_cache_*       a third fresh device, no VkPipelineCache, same
 *                    pipelines: whatever caching the implementation does
 *                    on its own
 *   cold_parallel4_* a new salt, created from 4 threads sharing one cache
 *
 * Every pipeline of every phase is used once - each graphics pipeline draws
 * its own 4x4 cell, each compute pipeline fills its own slot - and the
 * results are compared exactly with the shaders evaluated on the CPU.
 */
#include <pthread.h>
#include <unistd.h>

#include "bench_vk.h"
#include "bench_shaders.h"

#define GFX 256
#define COMPUTE 128
#define CELL 4
#define GRID 16 /* cells per row */
#define SIZE (GRID * CELL)
#define COMPUTE_SLOT 256 /* result words per compute pipeline */
#define THREADS 4

static const uint32_t iteration_counts[4] = { 1, 3, 6, 12 };
static const uint32_t local_sizes[4] = { 32, 64, 128, 256 };

/* Specialization data: pc_uber.frag (ids 0-3), pc_uber.comp (ids 0-4). */
struct uber_spec {
    uint32_t material, iterations, salt, features, local_size;
};

static const VkSpecializationMapEntry uber_entries[5] = {
    { 0, offsetof(struct uber_spec, material), sizeof(uint32_t) },
    { 1, offsetof(struct uber_spec, iterations), sizeof(uint32_t) },
    { 2, offsetof(struct uber_spec, salt), sizeof(uint32_t) },
    { 3, offsetof(struct uber_spec, features), sizeof(uint32_t) },
    { 4, offsetof(struct uber_spec, local_size), sizeof(uint32_t) },
};
static const VkSpecializationMapEntry vs_entry = { 0, 0, sizeof(uint32_t) };

struct gfx_variant {
    struct uber_spec spec;
    uint32_t vs_seed;
    int mesh_vs, blend, cull, depth_test;
};

static struct gfx_variant gfx_variant(int i, uint32_t salt)
{
    struct gfx_variant v;
    memset(&v, 0, sizeof(v));
    v.spec.material = (uint32_t)i & 7;
    v.spec.iterations = iteration_counts[(i >> 3) & 3];
    v.spec.features = (uint32_t)(i >> 5) & 7;
    v.spec.salt = salt;
    v.vs_seed = (uint32_t)i * 2654435761u;
    v.mesh_vs = (i >> 1) & 1;
    v.blend = (i >> 2) & 1;
    v.cull = (i >> 4) & 1;
    v.depth_test = (i >> 6) & 1;
    return v;
}

static struct uber_spec compute_variant(int i, uint32_t salt)
{
    struct uber_spec s;
    s.material = (uint32_t)i & 7;
    s.local_size = local_sizes[(i >> 3) & 3];
    s.iterations = iteration_counts[(i >> 5) & 3];
    s.features = (uint32_t)(i * 3) & 7;
    s.salt = salt;
    return s;
}

/* ---- The shaders on the CPU (pc_*.vert, pc_uber.frag/comp) ---- */

static uint32_t rotl(uint32_t x, uint32_t r)
{
    return (x << r) | (x >> (32u - r));
}

static uint32_t hash2(uint32_t a, uint32_t b)
{
    uint32_t h = (a * 0x8DA6B343u) ^ (b * 0xD8163841u);
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return h;
}

static uint32_t bit_reverse(uint32_t x)
{
    uint32_t r = 0;
    int i;
    for (i = 0; i < 32; i++, x >>= 1)
        r = (r << 1) | (x & 1u);
    return r;
}

static uint32_t bit_count(uint32_t x)
{
    uint32_t n = 0;
    for (; x; x &= x - 1)
        n++;
    return n;
}

static uint32_t find_msb(uint32_t x) /* x != 0 */
{
    uint32_t n = 31;
    while (!(x >> n))
        n--;
    return n;
}

static uint32_t uber(const struct uber_spec *s, uint32_t seed)
{
    uint32_t h = seed ^ s->salt, n = s->iterations, i, k;
    int pass;

    switch (s->material) {
    case 0:
        for (i = 0; i < n; i++) {
            h ^= h << 13;
            h ^= h >> 17;
            h ^= h << 5;
        }
        break;
    case 1:
        for (i = 0; i < n; i++)
            h = rotl(h * 1664525u + 1013904223u, 7u);
        break;
    case 2:
        break; /* murmur: below */
    case 3: {
        uint32_t v[4] = { h, h ^ 0xA5A5A5A5u, h * 3u, h + 7u }, t[4];
        for (i = 0; i < n; i++) {
            for (k = 0; k < 4; k++)
                t[k] = (v[(k + 1) & 3] * 0x01000193u) ^ v[k];
            memcpy(v, t, sizeof(v));
        }
        h = v[0] ^ v[1] ^ v[2] ^ v[3];
        break;
    }
    case 4:
        for (i = 0; i < n; i++)
            for (k = 0; k < 8; k++)
                h = (h >> 1) ^ (0xEDB88320u & (0u - (h & 1u)));
        break;
    case 5:
        for (i = 0; i < n; i++)
            h = bit_reverse(h) + bit_count(h) * 0x9E3779B9u + find_msb(h | 1u);
        break;
    case 6: {
        uint32_t v[8], r = 0;
        for (k = 0; k < 8; k++) {
            h = h * 747796405u + 2891336453u;
            v[k] = h >> 8;
        }
        for (i = 0; i < n; i++) {
            for (k = 0; k < 8; k += 2) {
                uint32_t a = v[k] < v[k + 1] ? v[k] : v[k + 1], b = v[k] < v[k + 1] ? v[k + 1] : v[k];
                v[k] = a;
                v[k + 1] = b;
            }
            for (k = 1; k < 7; k += 2) {
                uint32_t a = v[k] < v[k + 1] ? v[k] : v[k + 1], b = v[k] < v[k + 1] ? v[k + 1] : v[k];
                v[k] = a;
                v[k + 1] = b;
            }
            v[i & 7] ^= v[(i + 3) & 7] * 0x27D4EB2Du;
        }
        for (k = 0; k < 8; k++)
            r = r * 31u + v[k];
        h = r;
        break;
    }
    default: {
        uint32_t x = h & 0xFFFFu, y = h >> 16;
        for (i = 0; i < n; i++) {
            uint32_t cx = x >> 4, cy = y >> 4, fx = x & 15u, fy = y & 15u;
            uint32_t top = (hash2(cx, cy) & 0xFFFFu) * (16u - fx) + (hash2(cx + 1u, cy) & 0xFFFFu) * fx;
            uint32_t bottom = (hash2(cx, cy + 1u) & 0xFFFFu) * (16u - fx) + (hash2(cx + 1u, cy + 1u) & 0xFFFFu) * fx;
            uint32_t value = top * (16u - fy) + bottom * fy;
            x = (x * 5u + value) & 0xFFFFu;
            y = (y * 3u + (value >> 8)) & 0xFFFFu;
            h ^= value + i;
        }
        break;
    }
    }
    /* path_murmur: material 2, and feature bit 0 (on h ^ 0x51). */
    for (pass = 0; pass < 2; pass++) {
        if (pass == 0 ? s->material != 2 : !(s->features & 1u))
            continue;
        if (pass == 1)
            h ^= 0x51u;
        for (i = 0; i < n; i++) {
            h ^= h >> 16;
            h *= 0x85EBCA6Bu;
            h ^= h >> 13;
            h *= 0xC2B2AE35u;
            h ^= h >> 16;
            h += i;
        }
    }
    if (s->features & 2u)
        h = ~h;
    if (s->features & 4u)
        h = rotl(h, 11u) * 0x9E3779B1u;
    return h;
}

static uint32_t vs_out_seed(const struct gfx_variant *v)
{
    return v->mesh_vs ? (v->vs_seed ^ 0x5BD1E995u) * 0x9E3779B9u : v->vs_seed * 0x9E3779B9u + 0x7F4A7C15u;
}

/* ---- One phase: a fresh device creating every pipeline ---- */

struct phase {
    const char *name;
    struct bench_vk vk;
    uint32_t salt;
    VkShaderModule vs_quad, vs_mesh, fs, cs;
    VkPipelineLayout gfx_layout, cs_layout;
    VkDescriptorSetLayout cs_set_layout;
    VkPipelineCache cache;
    struct gfx_variant gfx_variants[GFX];
    struct uber_spec cs_specs[COMPUTE];
    VkSpecializationInfo fs_spec[GFX], vs_spec[GFX], cs_spec[COMPUTE];
    struct bench_gfx_state gfx_states[GFX];
    VkComputePipelineCreateInfo cs_infos[COMPUTE];
    VkPipeline gfx[GFX], compute[COMPUTE];
    double gfx_us[GFX], cs_us[COMPUTE];
};

static void phase_prepare(struct phase *p, const void *cache_data, size_t cache_size, int use_cache)
{
    struct bench_vk *vk = &p->vk;
    VkDescriptorSetLayoutBinding binding = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    struct bench_gfx_desc d;
    int i;

    p->vs_quad = BENCH_SHADER(vk, pc_quad_vert);
    p->vs_mesh = BENCH_SHADER(vk, pc_mesh_vert);
    p->fs = BENCH_SHADER(vk, pc_uber_frag);
    p->cs = BENCH_SHADER(vk, pc_uber_comp);
    p->gfx_layout = bench_pipeline_layout(vk, 0, NULL, 4 * sizeof(float), VK_SHADER_STAGE_VERTEX_BIT);
    p->cs_set_layout = bench_set_layout(vk, 1, &binding, 0);
    p->cs_layout = bench_pipeline_layout(vk, 1, &p->cs_set_layout, sizeof(uint32_t), VK_SHADER_STAGE_COMPUTE_BIT);
    if (use_cache) {
        VkPipelineCacheCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .initialDataSize = cache_size,
            .pInitialData = cache_data,
        };
        BENCH_VK(vkd.CreatePipelineCache(vk->device, &info, NULL, &p->cache));
    }

    for (i = 0; i < GFX; i++) {
        struct gfx_variant *v = &p->gfx_variants[i];
        *v = gfx_variant(i, p->salt);
        p->fs_spec[i] = (VkSpecializationInfo){ 4, uber_entries, sizeof(v->spec), &v->spec };
        p->vs_spec[i] = (VkSpecializationInfo){ 1, &vs_entry, sizeof(uint32_t), &v->vs_seed };
        memset(&d, 0, sizeof(d));
        d.vs = v->mesh_vs ? p->vs_mesh : p->vs_quad;
        d.fs = p->fs;
        d.vs_spec = &p->vs_spec[i];
        d.fs_spec = &p->fs_spec[i];
        d.layout = p->gfx_layout;
        d.color_count = 1;
        d.color_formats[0] = VK_FORMAT_R8G8B8A8_UNORM;
        d.depth_format = VK_FORMAT_D32_SFLOAT;
        d.depth_test = d.depth_write = v->depth_test;
        d.depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
        d.blend = v->blend;
        d.cull = v->cull ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
        d.front_face = VK_FRONT_FACE_CLOCKWISE; /* the quads' winding in framebuffer space */
        if (v->mesh_vs) {
            d.binding_count = 1;
            d.bindings[0] = (VkVertexInputBindingDescription){ 0, 2 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX };
            d.attribute_count = 1;
            d.attributes[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 };
        }
        bench_gfx_state_init(&p->gfx_states[i], &d);
    }
    for (i = 0; i < COMPUTE; i++) {
        p->cs_specs[i] = compute_variant(i, p->salt);
        p->cs_spec[i] = (VkSpecializationInfo){ 5, uber_entries, sizeof(struct uber_spec), &p->cs_specs[i] };
        p->cs_infos[i] = bench_compute_info(p->cs, &p->cs_spec[i], p->cs_layout);
    }
}

/* Creates pipelines first, first + stride, ... and times each call. */
static void create_range(struct phase *p, int first, int stride)
{
    VkDevice dev = p->vk.device;
    int i;
    for (i = first; i < GFX; i += stride) {
        double t0 = bench_now_us();
        BENCH_VK(vkd.CreateGraphicsPipelines(dev, p->cache, 1, &p->gfx_states[i].info, NULL, &p->gfx[i]));
        p->gfx_us[i] = bench_now_us() - t0;
    }
    for (i = first; i < COMPUTE; i += stride) {
        double t0 = bench_now_us();
        BENCH_VK(vkd.CreateComputePipelines(dev, p->cache, 1, &p->cs_infos[i], NULL, &p->compute[i]));
        p->cs_us[i] = bench_now_us() - t0;
    }
}

struct worker {
    struct phase *p;
    int index;
};

static void *worker_main(void *arg)
{
    struct worker *w = arg;
    create_range(w->p, w->index, THREADS);
    return NULL;
}

/* Returns wall time in microseconds. */
static double create_all(struct phase *p, int threads)
{
    double t0 = bench_now_us();
    if (threads <= 1) {
        create_range(p, 0, 1);
    } else {
        pthread_t ids[THREADS];
        struct worker workers[THREADS];
        int i;
        for (i = 0; i < THREADS; i++) {
            workers[i].p = p;
            workers[i].index = i;
            if (pthread_create(&ids[i], NULL, worker_main, &workers[i]) != 0)
                bench_fatal("pthread_create failed");
        }
        for (i = 0; i < THREADS; i++)
            pthread_join(ids[i], NULL);
    }
    return bench_now_us() - t0;
}

/* Draws a cell with every graphics pipeline and runs every compute pipeline,
 * then checks both against the CPU. */
static int phase_verify(struct phase *p, const char *gfx_config, const char *cs_config)
{
    struct bench_vk *vk = &p->vk;
    static const float corners[6][2] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };
    struct bench_image target, depth;
    struct bench_buffer vertices, results, readback;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    VkDescriptorSetAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorSetCount = 1,
        .pSetLayouts = &p->cs_set_layout,
    };
    VkDescriptorBufferInfo buffer_info;
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_info,
    };
    VkRenderingAttachmentInfo depth_attachment = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    };
    VkRenderingAttachmentInfo color_attachment = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    VkRenderingInfo rendering = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = { { 0, 0 }, { SIZE, SIZE } },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
        .pDepthAttachment = &depth_attachment,
    };
    VkBufferCopy copy = { 0, SIZE * SIZE * 4, (VkDeviceSize)COMPUTE * COMPUTE_SLOT * 4 };
    VkDeviceSize zero = 0;
    VkCommandBuffer cmd;
    uint32_t *want, *got;
    int i, ok;

    bench_image_create(vk, &target, VK_FORMAT_R8G8B8A8_UNORM, SIZE, SIZE, 1, VK_SAMPLE_COUNT_1_BIT,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0);
    bench_image_create(vk, &depth, VK_FORMAT_D32_SFLOAT, SIZE, SIZE, 1, VK_SAMPLE_COUNT_1_BIT,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 0);
    bench_buffer_create(vk, &vertices, sizeof(corners), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &vertices, 0, corners, sizeof(corners));
    bench_buffer_create(vk, &results, copy.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        0, BENCH_DEVICE_LOCAL);
    bench_buffer_create(vk, &readback, SIZE * SIZE * 4 + copy.size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, BENCH_HOST,
                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    BENCH_VK(vkd.CreateDescriptorPool(vk->device, &pool_info, NULL, &pool));
    alloc.descriptorPool = pool;
    BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &alloc, &set));
    buffer_info = (VkDescriptorBufferInfo){ results.buffer, 0, VK_WHOLE_SIZE };
    write.dstSet = set;
    vkd.UpdateDescriptorSets(vk->device, 1, &write, 0, NULL);

    cmd = bench_cmd_begin(vk);
    bench_image_barrier(cmd, &target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    bench_image_barrier(cmd, &depth, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_NONE, 0,
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    color_attachment.imageView = target.view;
    depth_attachment.imageView = depth.view;
    depth_attachment.clearValue.depthStencil.depth = 1.0f;
    vkd.CmdBeginRendering(cmd, &rendering);
    bench_cmd_viewport(cmd, SIZE, SIZE);
    vkd.CmdBindVertexBuffers(cmd, 0, 1, &vertices.buffer, &zero);
    for (i = 0; i < GFX; i++) {
        struct bench_rect r = { (i % GRID) * CELL, (i / GRID) * CELL, (i % GRID + 1) * CELL, (i / GRID + 1) * CELL, 0 };
        float rect[4];
        bench_rect_ndc(&r, SIZE, SIZE, 0, 0, rect);
        vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p->gfx[i]);
        vkd.CmdPushConstants(cmd, p->gfx_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(rect), rect);
        vkd.CmdDraw(cmd, 6, 1, 0, 0);
    }
    vkd.CmdEndRendering(cmd);
    bench_cmd_copy_image_to_buffer(cmd, &target, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, readback.buffer);
    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->cs_layout, 0, 1, &set, 0, NULL);
    for (i = 0; i < COMPUTE; i++) {
        uint32_t base = (uint32_t)i * COMPUTE_SLOT;
        vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->compute[i]);
        vkd.CmdPushConstants(cmd, p->cs_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(base), &base);
        vkd.CmdDispatch(cmd, 1, 1, 1);
    }
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    vkd.CmdCopyBuffer(cmd, results.buffer, readback.buffer, 1, &copy);
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    bench_cmd_finish(vk, cmd);

    /* Graphics: every pixel of every cell. */
    want = bench_alloc(sizeof(uint32_t) * SIZE * SIZE);
    for (i = 0; i < GFX; i++) {
        const struct gfx_variant *v = &p->gfx_variants[i];
        uint32_t seed = vs_out_seed(v);
        int x, y;
        for (y = (i / GRID) * CELL; y < (i / GRID + 1) * CELL; y++)
            for (x = (i % GRID) * CELL; x < (i % GRID + 1) * CELL; x++)
                want[y * SIZE + x] = uber(&v->spec, seed ^ ((uint32_t)x * 73856093u) ^ ((uint32_t)y * 19349663u)) |
                                     0xFF000000u;
    }
    ok = bench_check_rgba8(gfx_config, readback.map, want, SIZE, SIZE);
    free(want);

    /* Compute: each slot's first local_size words. */
    got = (uint32_t *)((char *)readback.map + SIZE * SIZE * 4);
    for (i = 0; i < COMPUTE && ok; i++) {
        const struct uber_spec *s = &p->cs_specs[i];
        uint32_t lid;
        for (lid = 0; lid < s->local_size; lid++) {
            uint32_t w = uber(s, lid * 0x9E3779B9u + s->local_size), g = got[i * COMPUTE_SLOT + lid];
            if (g != w) {
                bench_error(cs_config, "pipeline %d, invocation %u: got 0x%08x, want 0x%08x", i, lid, g, w);
                ok = 0;
                break;
            }
        }
    }

    vkd.DestroyDescriptorPool(vk->device, pool, NULL);
    bench_buffer_destroy(vk, &vertices);
    bench_buffer_destroy(vk, &results);
    bench_buffer_destroy(vk, &readback);
    bench_image_destroy(vk, &target);
    bench_image_destroy(vk, &depth);
    return ok;
}

static void report(struct phase *p, const char *config, int compute, const char *cache, size_t cache_bytes,
                   int threads, double wall_us)
{
    double *samples = compute ? p->cs_us : p->gfx_us;
    int i, n = compute ? COMPUTE : GFX;
    double total = 0.0;
    struct bench_json j;

    for (i = 0; i < n; i++)
        total += samples[i];
    bench_json_begin(&j, config);
    bench_json_int(&j, "pipelines", n);
    bench_json_str(&j, "cache", cache);
    if (cache_bytes)
        bench_json_int(&j, "cache_bytes", (long long)cache_bytes);
    bench_json_int(&j, "threads", threads);
    bench_json_stats(&j, "create_us", samples, n);
    bench_json_num(&j, "total_ms", total / 1000.0);
    if (threads > 1)
        bench_json_num(&j, "wall_ms", wall_us / 1000.0); /* both kinds of pipeline */
    bench_json_end(&j);
}

static uint32_t new_salt(void)
{
    static uint32_t counter;
    uint64_t t = mach_absolute_time();
    return ((uint32_t)t ^ (uint32_t)(t >> 32) ^ ((uint32_t)getpid() << 16) ^ (++counter * 0x9E3779B9u)) | 1u;
}

/* Runs one phase on a fresh device; returns its pipeline cache data when
 * cache_out is set (caller frees). */
static void run_phase(const char *name, uint32_t salt, int use_cache, const void *cache_in, size_t cache_in_size,
                      int threads, void **cache_out, size_t *cache_out_size)
{
    struct phase *p = bench_alloc(sizeof(*p));
    char gfx_config[64], cs_config[64];
    const char *cache_kind = !use_cache ? "none" : cache_in_size ? "loaded" : "empty";
    double wall;
    int i;

    snprintf(gfx_config, sizeof(gfx_config), "%s_graphics", name);
    snprintf(cs_config, sizeof(cs_config), "%s_compute", name);
    bench_config = gfx_config;
    p->name = name;
    p->salt = salt;
    bench_vk_init(&p->vk, BENCH_NEED_DYNAMIC_RENDERING | BENCH_NEED_SYNC2 | BENCH_NEED_MAINTENANCE4);
    phase_prepare(p, cache_in, cache_in_size, use_cache);
    wall = create_all(p, threads);

    if (phase_verify(p, gfx_config, cs_config)) {
        if (bench_selected(gfx_config))
            report(p, gfx_config, 0, cache_kind, cache_in_size, threads, wall);
        if (bench_selected(cs_config))
            report(p, cs_config, 1, cache_kind, cache_in_size, threads, wall);
    }
    if (cache_out && p->cache) {
        size_t size = 0;
        void *data = NULL;
        if (vkd.GetPipelineCacheData(p->vk.device, p->cache, &size, NULL) == VK_SUCCESS && size) {
            data = bench_alloc(size);
            if (vkd.GetPipelineCacheData(p->vk.device, p->cache, &size, data) != VK_SUCCESS) {
                free(data);
                data = NULL;
                size = 0;
            }
        }
        *cache_out = data;
        *cache_out_size = size;
    }

    vkd.DeviceWaitIdle(p->vk.device);
    for (i = 0; i < GFX; i++)
        vkd.DestroyPipeline(p->vk.device, p->gfx[i], NULL);
    for (i = 0; i < COMPUTE; i++)
        vkd.DestroyPipeline(p->vk.device, p->compute[i], NULL);
    if (p->cache)
        vkd.DestroyPipelineCache(p->vk.device, p->cache, NULL);
    vkd.DestroyPipelineLayout(p->vk.device, p->gfx_layout, NULL);
    vkd.DestroyPipelineLayout(p->vk.device, p->cs_layout, NULL);
    vkd.DestroyDescriptorSetLayout(p->vk.device, p->cs_set_layout, NULL);
    vkd.DestroyShaderModule(p->vk.device, p->vs_quad, NULL);
    vkd.DestroyShaderModule(p->vk.device, p->vs_mesh, NULL);
    vkd.DestroyShaderModule(p->vk.device, p->fs, NULL);
    vkd.DestroyShaderModule(p->vk.device, p->cs, NULL);
    bench_vk_destroy(&p->vk);
    bench_config = NULL;
    free(p);
}

static int phase_selected(const char *name)
{
    char config[64];
    snprintf(config, sizeof(config), "%s_graphics", name);
    if (bench_selected(config))
        return 1;
    snprintf(config, sizeof(config), "%s_compute", name);
    return bench_selected(config);
}

int main(void)
{
    uint32_t salt = new_salt();
    void *cache = NULL;
    size_t cache_size = 0;

    bench_begin("pipeline_compile");
    if (phase_selected("cold") || phase_selected("warm_cache") || phase_selected("no_cache"))
        run_phase("cold", salt, 1, NULL, 0, 1, &cache, &cache_size);
    if (phase_selected("warm_cache")) {
        if (cache_size)
            run_phase("warm_cache", salt, 1, cache, cache_size, 1, NULL, NULL);
        else
            bench_skip("warm_cache", "vkGetPipelineCacheData returned no data");
    }
    if (phase_selected("no_cache"))
        run_phase("no_cache", salt, 0, NULL, 0, 1, NULL, NULL);
    if (phase_selected("cold_parallel4"))
        run_phase("cold_parallel4", new_salt(), 1, NULL, 0, THREADS, NULL, NULL);
    free(cache);
    return bench_exit_code();
}
