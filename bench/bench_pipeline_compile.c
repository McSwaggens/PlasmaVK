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
 *   batch_cold_*     a new salt, every pipeline of a kind in one
 *                    vkCreate*Pipelines call (as engines do at load); the
 *                    lines report the call's wall time
 *
 * Every pipeline of every phase is used once - each graphics pipeline draws
 * its own 4x4 cell, each compute pipeline fills its own slot - and the
 * results are compared exactly with the shaders evaluated on the CPU.
 *
 * The phases below use a textured material shader instead (pc_material.frag:
 * two textures and a uniform block, feature switches as specialization
 * constants). Engines create pipelines this way: few shaders, many states,
 * no VkPipelineCache. One call per pipeline, VK_NULL_HANDLE cache:
 *
 *   permutations_cold     fresh device, new salt: 16 shader pairs (2 vertex
 *                         shaders x 8 feature sets) x 32 states (blend, cull,
 *                         depth attachment, color format, write mask), 512
 *                         pipelines, pair by pair. Each pipeline is counted
 *                         in the first of these buckets that names something
 *                         no earlier pipeline of the phase had:
 *                           new_fragment  a fragment variant (8; the very
 *                                         first also brings vertex shader A)
 *                           new_vertex    a vertex shader (1: B's first)
 *                           new_state     an output state of a known
 *                                         variant: blend, color format,
 *                                         write mask, depth format (120).
 *                                         Apple GPUs build blending and the
 *                                         output conversion into the
 *                                         fragment binary, so drivers
 *                                         compile the variant again for
 *                                         some of these
 *                           new_pair      a pairing of known shaders (7)
 *                           repeat        nothing new (376; cull is encoder
 *                                         state)
 *                         Each bucket reports its _us statistics and its _ms
 *                         total. A bucket in which a driver compiles only
 *                         some pipelines is bimodal, so compare the totals
 *   permutations_recreate the same device: all 512 destroyed and created
 *                         again (a renderer restart or settings change)
 *   permutations_warm     a fresh device, the same salt: the next launch,
 *                         with whatever the OS shader cache kept
 *   spec_variants_cold    fresh device, new salt: 16 pipelines with the same
 *                         shaders and state that differ only in the values of
 *                         specialization constants
 *
 * The salt also replaces a constant in both vertex shaders (the depth they
 * output), so their modules are new in every run too.
 * These draw with every pipeline too; nearest sampling at texel centres lets
 * the CPU reproduce the pixels within 1 per channel.
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
    VkGraphicsPipelineCreateInfo gfx_infos[GFX]; /* batch creation */
    VkPipeline gfx[GFX], compute[COMPUTE];
    double gfx_us[GFX], cs_us[COMPUTE];
    double gfx_batch_us, cs_batch_us;
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

/* Creates every pipeline of a kind in one call and times the call. */
static void create_batch(struct phase *p)
{
    VkDevice dev = p->vk.device;
    double t0;
    int i;
    for (i = 0; i < GFX; i++)
        p->gfx_infos[i] = p->gfx_states[i].info;
    t0 = bench_now_us();
    BENCH_VK(vkd.CreateGraphicsPipelines(dev, p->cache, GFX, p->gfx_infos, NULL, p->gfx));
    p->gfx_batch_us = bench_now_us() - t0;
    t0 = bench_now_us();
    BENCH_VK(vkd.CreateComputePipelines(dev, p->cache, COMPUTE, p->cs_infos, NULL, p->compute));
    p->cs_batch_us = bench_now_us() - t0;
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

/* Returns wall time in microseconds. threads 0: one call per kind. */
static double create_all(struct phase *p, int threads)
{
    double t0 = bench_now_us();
    if (threads == 0) {
        create_batch(p);
    } else if (threads == 1) {
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
    if (threads == 0) { /* one call: no per-pipeline times */
        total = compute ? p->cs_batch_us : p->gfx_batch_us;
        bench_json_int(&j, "threads", 1);
        bench_json_int(&j, "calls", 1);
        bench_json_num(&j, "total_ms", total / 1000.0);
        bench_json_num(&j, "ms_per_pipeline", total / 1000.0 / n);
        bench_json_end(&j);
        return;
    }
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
 * cache_out is set (caller frees). threads 0: batch creation. */
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

/* ---- Material permutations (pc_mat_*.vert, pc_material.frag) ---- */

#define MAT_PAIRS 16  /* 2 vertex shaders x 8 feature sets */
#define MAT_STATES 32 /* the STATE_* bits */
#define MAT_MAX (MAT_PAIRS * MAT_STATES)
#define MAT_SPEC_VARIANTS 16
#define TEX 64      /* material textures: TEX x TEX RGBA8, nearest, repeat */
#define GROUP_W 64  /* one target per (color format, depth) group of pipelines */
#define GROUP_H 32
#define GROUP_COLS (GROUP_W / CELL)
#define GROUPS 4

enum {
    STATE_BLEND = 1,
    STATE_CULL = 2,
    STATE_DEPTH = 4,          /* D32 attachment, test and write; otherwise none */
    STATE_BGRA = 8,           /* B8G8R8A8 color attachment; otherwise R8G8B8A8 */
    STATE_NO_ALPHA_WRITE = 16 /* color write mask RGB */
};

static const float mat_tint[3] = { 0.75f, 0.5f, 1.0f };
static const float mat_emissive[3] = { 0.125f, 0.0625f, 0.25f };
static const float mat_clear[4] = { 64.0f / 255.0f, 128.0f / 255.0f, 192.0f / 255.0f, 1.0f };

/* pc_material.frag's specialization constants. */
struct mat_spec {
    uint32_t features, salt;
};

static const VkSpecializationMapEntry mat_entries[2] = {
    { 0, offsetof(struct mat_spec, features), sizeof(uint32_t) },
    { 1, offsetof(struct mat_spec, salt), sizeof(uint32_t) },
};

struct mat_pipeline {
    struct mat_spec spec;
    int mesh_vs, state;
};

struct mat_phase {
    struct bench_vk vk;
    int count;
    VkShaderModule vs_quad, vs_mesh, fs;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkSampler sampler;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    struct bench_image albedo, detail;
    struct bench_buffer material;
    uint32_t albedo_texels[TEX * TEX], detail_texels[TEX * TEX];
    struct mat_pipeline info[MAT_MAX];
    VkSpecializationInfo spec[MAT_MAX];
    struct bench_gfx_state states[MAT_MAX];
    VkPipeline pipelines[MAT_MAX];
    double us[MAT_MAX];
};

static int mat_group(int state)
{
    return ((state & STATE_DEPTH) ? 1 : 0) | ((state & STATE_BGRA) ? 2 : 0);
}

/* A material vertex shader whose depth constant (0.5, its only OpConstant of
 * that value) is replaced by one derived from the salt, so every run
 * compiles a module that no cache has seen. No pixel depends on the depth:
 * each cell is drawn once, over a depth cleared to 1. A specialization
 * constant would do the same, but drivers that specialize Metal functions
 * at pipeline creation (MoltenVK) would then pay that for every pipeline,
 * which vertex shaders without specialization constants do not cost. */
static VkShaderModule mat_vertex_shader(struct bench_vk *vk, const uint32_t *code, size_t size, uint32_t salt)
{
    uint32_t *words = bench_alloc(size), depth_bits;
    float depth = 0.25f + (float)(salt & 0xFFFFu) / 131072.0f;
    size_t i, n = size / sizeof(uint32_t), patched = 0;
    VkShaderModule module;

    memcpy(words, code, size);
    memcpy(&depth_bits, &depth, sizeof(depth_bits));
    for (i = 5; i < n && words[i] >> 16; i += words[i] >> 16) /* after the header, instruction by instruction */
        if (words[i] == (4u << 16 | 43u) && i + 3 < n && words[i + 3] == 0x3F000000u) { /* OpConstant 0.5 */
            words[i + 3] = depth_bits;
            patched++;
        }
    if (patched != 1)
        bench_fatal("material vertex shader: %zu depth constants, want 1", patched);
    module = bench_shader_module(vk, words, size);
    free(words);
    return module;
}

/* permutations: MAT_MAX pipelines, pair-major (pair = vertex shader x
 * feature set, then its 32 states); otherwise MAT_SPEC_VARIANTS pipelines
 * with one vertex shader and state, features 0-15. */
static void mat_setup(struct mat_phase *m, uint32_t salt, int permutations)
{
    struct bench_vk *vk = &m->vk;
    VkDescriptorSetLayoutBinding bindings[3] = {
        { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
    };
    VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    };
    VkDescriptorPoolSize pool_sizes[2] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
    };
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes,
    };
    VkDescriptorSetAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorSetCount = 1,
    };
    VkDescriptorImageInfo images[2];
    VkDescriptorBufferInfo buffer_info;
    VkWriteDescriptorSet writes[3];
    uint64_t seed = 0x3A7E21A1ull;
    float *ubo;
    int i;

    bench_vk_init(vk, BENCH_NEED_DYNAMIC_RENDERING | BENCH_NEED_SYNC2);
    m->vs_quad = mat_vertex_shader(vk, pc_mat_quad_vert, sizeof(pc_mat_quad_vert), salt);
    m->vs_mesh = mat_vertex_shader(vk, pc_mat_mesh_vert, sizeof(pc_mat_mesh_vert), salt);
    m->fs = BENCH_SHADER(vk, pc_material_frag);
    m->set_layout = bench_set_layout(vk, 3, bindings, 0);
    m->layout = bench_pipeline_layout(vk, 1, &m->set_layout, 8 * sizeof(float), VK_SHADER_STAGE_VERTEX_BIT);
    BENCH_VK(vkd.CreateSampler(vk->device, &sampler_info, NULL, &m->sampler));

    /* Albedo alpha is either well above or well below the alpha test. */
    for (i = 0; i < TEX * TEX; i++) {
        m->albedo_texels[i] = (bench_rand(&seed) & 0x00FFFFFFu) | ((bench_rand(&seed) & 1u) ? 0xFF000000u : 0x40000000u);
        m->detail_texels[i] = bench_rand(&seed) | 0xFF000000u;
    }
    bench_image_create(vk, &m->albedo, VK_FORMAT_R8G8B8A8_UNORM, TEX, TEX, 1, VK_SAMPLE_COUNT_1_BIT,
                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0);
    bench_image_create(vk, &m->detail, VK_FORMAT_R8G8B8A8_UNORM, TEX, TEX, 1, VK_SAMPLE_COUNT_1_BIT,
                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0);
    bench_image_upload(vk, &m->albedo, m->albedo_texels, sizeof(m->albedo_texels),
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    bench_image_upload(vk, &m->detail, m->detail_texels, sizeof(m->detail_texels),
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    bench_buffer_create(vk, &m->material, 8 * sizeof(float), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, BENCH_HOST, 0);
    ubo = m->material.map;
    memcpy(ubo, mat_tint, sizeof(mat_tint));
    ubo[3] = 1.0f;
    memcpy(ubo + 4, mat_emissive, sizeof(mat_emissive));
    ubo[7] = 0.0f;

    BENCH_VK(vkd.CreateDescriptorPool(vk->device, &pool_info, NULL, &m->pool));
    alloc.descriptorPool = m->pool;
    alloc.pSetLayouts = &m->set_layout;
    BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &alloc, &m->set));
    images[0] = (VkDescriptorImageInfo){ m->sampler, m->albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    images[1] = (VkDescriptorImageInfo){ m->sampler, m->detail.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    buffer_info = (VkDescriptorBufferInfo){ m->material.buffer, 0, VK_WHOLE_SIZE };
    for (i = 0; i < 3; i++) {
        memset(&writes[i], 0, sizeof(writes[i]));
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m->set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = bindings[i].descriptorType;
        if (i < 2)
            writes[i].pImageInfo = &images[i];
        else
            writes[i].pBufferInfo = &buffer_info;
    }
    vkd.UpdateDescriptorSets(vk->device, 3, writes, 0, NULL);

    m->count = permutations ? MAT_MAX : MAT_SPEC_VARIANTS;
    for (i = 0; i < m->count; i++) {
        struct mat_pipeline *p = &m->info[i];
        struct bench_gfx_desc d;
        if (permutations) {
            int pair = i / MAT_STATES;
            p->mesh_vs = pair / 8;
            p->spec.features = (uint32_t)pair % 8;
            p->state = i % MAT_STATES;
        } else {
            p->mesh_vs = 0;
            p->spec.features = (uint32_t)i;
            p->state = 0;
        }
        p->spec.salt = salt;
        m->spec[i] = (VkSpecializationInfo){ 2, mat_entries, sizeof(p->spec), &p->spec };
        memset(&d, 0, sizeof(d));
        d.vs = p->mesh_vs ? m->vs_mesh : m->vs_quad;
        d.fs = m->fs;
        d.fs_spec = &m->spec[i];
        d.layout = m->layout;
        d.color_count = 1;
        d.color_formats[0] = (p->state & STATE_BGRA) ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
        if (p->state & STATE_DEPTH) {
            d.depth_format = VK_FORMAT_D32_SFLOAT;
            d.depth_test = d.depth_write = 1;
            d.depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
        }
        d.blend = (p->state & STATE_BLEND) != 0;
        d.cull = (p->state & STATE_CULL) ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
        d.front_face = VK_FRONT_FACE_CLOCKWISE; /* the quads' winding in framebuffer space */
        if (p->mesh_vs) {
            d.binding_count = 1;
            d.bindings[0] = (VkVertexInputBindingDescription){ 0, 2 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX };
            d.attribute_count = 1;
            d.attributes[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 };
        }
        bench_gfx_state_init(&m->states[i], &d);
        if (p->state & STATE_NO_ALPHA_WRITE)
            m->states[i].attachments[0].colorWriteMask &= ~(VkColorComponentFlags)VK_COLOR_COMPONENT_A_BIT;
    }
}

static void mat_create(struct mat_phase *m)
{
    VkDevice dev = m->vk.device;
    int i;
    for (i = 0; i < m->count; i++) {
        double t0 = bench_now_us();
        BENCH_VK(vkd.CreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &m->states[i].info, NULL, &m->pipelines[i]));
        m->us[i] = bench_now_us() - t0;
    }
}

static void mat_destroy_pipelines(struct mat_phase *m)
{
    int i;
    vkd.DeviceWaitIdle(m->vk.device);
    for (i = 0; i < m->count; i++) {
        vkd.DestroyPipeline(m->vk.device, m->pipelines[i], NULL);
        m->pipelines[i] = VK_NULL_HANDLE;
    }
}

static void mat_teardown(struct mat_phase *m)
{
    struct bench_vk *vk = &m->vk;
    mat_destroy_pipelines(m);
    vkd.DestroyDescriptorPool(vk->device, m->pool, NULL);
    vkd.DestroyPipelineLayout(vk->device, m->layout, NULL);
    vkd.DestroyDescriptorSetLayout(vk->device, m->set_layout, NULL);
    vkd.DestroySampler(vk->device, m->sampler, NULL);
    vkd.DestroyShaderModule(vk->device, m->vs_quad, NULL);
    vkd.DestroyShaderModule(vk->device, m->vs_mesh, NULL);
    vkd.DestroyShaderModule(vk->device, m->fs, NULL);
    bench_image_destroy(vk, &m->albedo);
    bench_image_destroy(vk, &m->detail);
    bench_buffer_destroy(vk, &m->material);
    bench_vk_destroy(vk);
}

/* pc_material.frag on the CPU, as stored in the pipeline's color format. */
static uint32_t mat_pixel(const struct mat_phase *m, const struct mat_pipeline *p, int x, int y)
{
    uint32_t sx = p->spec.salt & 63u, sy = (p->spec.salt >> 6) & 63u, features = p->spec.features;
    uint32_t a = m->albedo_texels[((uint32_t)y + sy) % TEX * TEX + ((uint32_t)x + sx) % TEX];
    uint32_t d = m->detail_texels[(y >> 1) * TEX + (x >> 1)];
    uint32_t out = 0xFF000000u;
    float c[3];
    int k;

    if ((features & 8u) && (a >> 24) < 128u) {
        for (k = 0; k < 3; k++)
            c[k] = mat_clear[k];
    } else {
        for (k = 0; k < 3; k++) {
            c[k] = (float)((a >> (8 * k)) & 0xFF) / 255.0f;
            if (features & 1u)
                c[k] *= (float)((d >> (8 * k)) & 0xFF) / 255.0f * 2.0f;
            if (features & 2u)
                c[k] *= mat_tint[k];
            if (features & 4u)
                c[k] += mat_emissive[k];
        }
    }
    for (k = 0; k < 3; k++) {
        float v = c[k] < 0.0f ? 0.0f : c[k] > 1.0f ? 1.0f : c[k];
        out |= (uint32_t)(v * 255.0f + 0.5f) << (8 * ((p->state & STATE_BGRA) ? 2 - k : k));
    }
    return out;
}

/* Draws a cell with every pipeline, one target per (color format, depth)
 * group, and checks the pixels against the CPU. */
static int mat_verify(struct mat_phase *m, const char *config)
{
    struct bench_vk *vk = &m->vk;
    static const float corners[6][2] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };
    struct bench_image targets[GROUPS], depth;
    struct bench_buffer vertices, readback[GROUPS];
    VkDeviceSize zero = 0;
    VkCommandBuffer cmd;
    uint32_t *want;
    int g, i, ok = 1;

    bench_image_create(vk, &depth, VK_FORMAT_D32_SFLOAT, GROUP_W, GROUP_H, 1, VK_SAMPLE_COUNT_1_BIT,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 0);
    bench_buffer_create(vk, &vertices, sizeof(corners), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &vertices, 0, corners, sizeof(corners));
    for (g = 0; g < GROUPS; g++) {
        bench_image_create(vk, &targets[g], (g & 2) ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R8G8B8A8_UNORM, GROUP_W,
                           GROUP_H, 1, VK_SAMPLE_COUNT_1_BIT,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0);
        bench_buffer_create(vk, &readback[g], GROUP_W * GROUP_H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, BENCH_HOST,
                            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    }

    cmd = bench_cmd_begin(vk);
    for (g = 0; g < GROUPS; g++) {
        VkRenderingAttachmentInfo color_attachment = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = targets[g].view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        };
        VkRenderingAttachmentInfo depth_attachment = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = depth.view,
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        };
        VkRenderingInfo rendering = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = { { 0, 0 }, { GROUP_W, GROUP_H } },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_attachment,
            .pDepthAttachment = (g & 1) ? &depth_attachment : NULL,
        };
        int cell = 0;

        memcpy(color_attachment.clearValue.color.float32, mat_clear, sizeof(mat_clear));
        depth_attachment.clearValue.depthStencil.depth = 1.0f;
        bench_image_barrier(cmd, &targets[g], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        if (g & 1)
            bench_image_barrier(cmd, &depth, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        vkd.CmdBeginRendering(cmd, &rendering);
        bench_cmd_viewport(cmd, GROUP_W, GROUP_H);
        vkd.CmdBindVertexBuffers(cmd, 0, 1, &vertices.buffer, &zero);
        vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m->layout, 0, 1, &m->set, 0, NULL);
        for (i = 0; i < m->count; i++) {
            struct bench_rect r;
            float push[8];
            if (mat_group(m->info[i].state) != g)
                continue;
            r.x0 = (cell % GROUP_COLS) * CELL;
            r.y0 = (cell / GROUP_COLS) * CELL;
            r.x1 = r.x0 + CELL;
            r.y1 = r.y0 + CELL;
            cell++;
            bench_rect_ndc(&r, GROUP_W, GROUP_H, 0, 0, push);
            push[4] = (float)r.x0 / TEX;
            push[5] = (float)r.y0 / TEX;
            push[6] = (float)r.x1 / TEX;
            push[7] = (float)r.y1 / TEX;
            vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m->pipelines[i]);
            vkd.CmdPushConstants(cmd, m->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), push);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        vkd.CmdEndRendering(cmd);
        bench_cmd_copy_image_to_buffer(cmd, &targets[g], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, readback[g].buffer);
    }
    bench_cmd_finish(vk, cmd);

    want = bench_alloc(sizeof(uint32_t) * GROUP_W * GROUP_H);
    for (g = 0; g < GROUPS && ok; g++) {
        int cell = 0, x, y;
        uint32_t clear = 0xFF000000u;
        for (i = 0; i < 3; i++)
            clear |= (uint32_t)(mat_clear[i] * 255.0f + 0.5f) << (8 * ((g & 2) ? 2 - i : i));
        for (i = 0; i < GROUP_W * GROUP_H; i++)
            want[i] = clear;
        for (i = 0; i < m->count; i++) {
            int x0, y0;
            if (mat_group(m->info[i].state) != g)
                continue;
            x0 = (cell % GROUP_COLS) * CELL;
            y0 = (cell / GROUP_COLS) * CELL;
            cell++;
            for (y = y0; y < y0 + CELL; y++)
                for (x = x0; x < x0 + CELL; x++)
                    want[y * GROUP_W + x] = mat_pixel(m, &m->info[i], x, y);
        }
        ok = bench_check_rgba8_tolerance(config, readback[g].map, want, GROUP_W, GROUP_H, 1);
    }
    free(want);

    for (g = 0; g < GROUPS; g++) {
        bench_image_destroy(vk, &targets[g]);
        bench_buffer_destroy(vk, &readback[g]);
    }
    bench_image_destroy(vk, &depth);
    bench_buffer_destroy(vk, &vertices);
    return ok;
}

enum { NEW_FRAGMENT, NEW_VERTEX, NEW_STATE, NEW_PAIR, REPEAT, BUCKETS };

static const char *const bucket_names[BUCKETS] = { "new_fragment", "new_vertex", "new_state", "new_pair", "repeat" };

/* The first of the buckets that names something no earlier pipeline of the
 * phase had. A fragment variant's output state is what Apple GPUs build into
 * its binary: blending, the color format and write mask, and (for some
 * drivers) the depth format; cull mode is not part of it. */
static void mat_buckets(const struct mat_phase *m, int *bucket)
{
    unsigned char seen_fragment[MAT_SPEC_VARIANTS] = { 0 }, seen_vertex[2] = { 0 };
    unsigned char seen_state[MAT_SPEC_VARIANTS * MAT_STATES] = { 0 }, seen_pair[2 * MAT_SPEC_VARIANTS] = { 0 };
    int i;

    for (i = 0; i < m->count; i++) {
        const struct mat_pipeline *p = &m->info[i];
        int fragment = (int)p->spec.features, state = fragment * MAT_STATES + (p->state & ~STATE_CULL);
        int pair = p->mesh_vs * MAT_SPEC_VARIANTS + fragment;
        bucket[i] = !seen_fragment[fragment]  ? NEW_FRAGMENT
                    : !seen_vertex[p->mesh_vs] ? NEW_VERTEX
                    : !seen_state[state]       ? NEW_STATE
                    : !seen_pair[pair]         ? NEW_PAIR
                                               : REPEAT;
        seen_fragment[fragment] = seen_vertex[p->mesh_vs] = seen_state[state] = seen_pair[pair] = 1;
    }
}

static void mat_report(struct mat_phase *m, const char *config, int permutations)
{
    double samples[BUCKETS][MAT_MAX], sums[BUCKETS] = { 0 }, total = 0.0;
    int bucket[MAT_MAX], counts[BUCKETS] = { 0 }, i;
    struct bench_json j;
    char key[64];

    mat_buckets(m, bucket);
    for (i = 0; i < m->count; i++) {
        total += m->us[i];
        sums[bucket[i]] += m->us[i];
        samples[bucket[i]][counts[bucket[i]]++] = m->us[i];
    }
    bench_json_begin(&j, config);
    bench_json_int(&j, "pipelines", m->count);
    bench_json_int(&j, "vertex_shaders", permutations ? 2 : 1);
    bench_json_int(&j, "fragment_variants", permutations ? 8 : MAT_SPEC_VARIANTS);
    bench_json_int(&j, "states", permutations ? MAT_STATES : 1);
    bench_json_str(&j, "cache", "none");
    bench_json_stats(&j, "create_us", m->us, m->count);
    for (i = 0; permutations && i < BUCKETS; i++) {
        snprintf(key, sizeof(key), "%s_us", bucket_names[i]);
        bench_json_stats(&j, key, samples[i], counts[i]);
        snprintf(key, sizeof(key), "%s_ms", bucket_names[i]);
        bench_json_num(&j, key, sums[i] / 1000.0);
    }
    bench_json_num(&j, "total_ms", total / 1000.0);
    bench_json_end(&j);
}

/* permutations_cold and _recreate on one device, then _warm on a fresh one
 * with the same salt. */
static void run_permutations(void)
{
    int cold = bench_selected("permutations_cold"), recreate = bench_selected("permutations_recreate");
    int warm = bench_selected("permutations_warm");
    uint32_t salt = new_salt();
    struct mat_phase *m;

    if (!cold && !recreate && !warm)
        return;
    m = bench_alloc(sizeof(*m));
    bench_config = "permutations_cold";
    mat_setup(m, salt, 1);
    mat_create(m);
    if (mat_verify(m, "permutations_cold") && cold)
        mat_report(m, "permutations_cold", 1);
    if (recreate) {
        bench_config = "permutations_recreate";
        mat_destroy_pipelines(m);
        mat_create(m);
        if (mat_verify(m, "permutations_recreate"))
            mat_report(m, "permutations_recreate", 1);
    }
    mat_teardown(m);
    if (warm) {
        memset(m, 0, sizeof(*m));
        bench_config = "permutations_warm";
        mat_setup(m, salt, 1);
        mat_create(m);
        if (mat_verify(m, "permutations_warm"))
            mat_report(m, "permutations_warm", 1);
        mat_teardown(m);
    }
    bench_config = NULL;
    free(m);
}

static void run_spec_variants(void)
{
    struct mat_phase *m;
    if (!bench_selected("spec_variants_cold"))
        return;
    m = bench_alloc(sizeof(*m));
    bench_config = "spec_variants_cold";
    mat_setup(m, new_salt(), 0);
    mat_create(m);
    if (mat_verify(m, "spec_variants_cold"))
        mat_report(m, "spec_variants_cold", 0);
    mat_teardown(m);
    bench_config = NULL;
    free(m);
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
    if (phase_selected("batch_cold"))
        run_phase("batch_cold", new_salt(), 1, NULL, 0, 0, NULL, NULL);
    free(cache);
    run_permutations();
    run_spec_variants();
    return bench_exit_code();
}
