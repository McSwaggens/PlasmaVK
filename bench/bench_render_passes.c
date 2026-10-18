/* Render passes: a deferred frame with dynamic rendering. A G-buffer pass
 * (albedo RGBA8, normal RGBA16F, emissive RGBA16F, depth D32) draws ~2000
 * overlapping objects; a lighting pass shades every pixel with 32 point
 * lights into an RGBA16F target. On Apple GPUs this is mostly a question of
 * what leaves tile memory, which the variants change:
 *
 *   clear_store      G-buffer CLEAR/STORE, sampled by the lighting pass
 *   load_store       G-buffer LOAD/STORE (last frame's contents reloaded)
 *   dontcare_store   G-buffer DONT_CARE/STORE (fully covered, so legal)
 *   msaa4_resolve    4x MSAA G-buffer, STORE_OP_DONT_CARE + resolve; the
 *                    MSAA images are transient (lazily allocated, memoryless)
 *   msaa4_store      the same but the MSAA images are stored too
 *   local_read       one rendering instance: the lighting reads the G-buffer
 *                    as input attachments (dynamic rendering local read,
 *                    core in 1.4) and the G-buffer is never stored
 *
 * GPU time per frame is the number to watch. The first frame of each
 * configuration is read back and compared with a CPU shading of the scene.
 */
#include <math.h>

#include "bench_vk.h"
#include "bench_shaders.h"

#define RECTS 2000 /* plus a background covering the screen */
#define LIGHTS 32
#define GBUFFERS 3
#define OUTPUT_FORMAT VK_FORMAT_R16G16B16A16_SFLOAT
#define DEPTH_FORMAT VK_FORMAT_D32_SFLOAT

static const VkFormat gbuffer_formats[GBUFFERS] = {
    VK_FORMAT_R8G8B8A8_UNORM,      /* albedo */
    VK_FORMAT_R16G16B16A16_SFLOAT, /* normal, roughness */
    VK_FORMAT_R16G16B16A16_SFLOAT, /* emissive */
};

enum mode { MODE_TWO_PASS, MODE_MSAA, MODE_LOCAL_READ };

struct config {
    const char *name;
    enum mode mode;
    uint32_t width, height;
    VkAttachmentLoadOp gbuffer_load; /* two-pass */
    VkAttachmentStoreOp msaa_store;  /* MSAA */
    int frames;
};

static const struct config configs[] = {
    { "clear_store_1080p", MODE_TWO_PASS, 1920, 1080, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, 100 },
    { "load_store_1080p", MODE_TWO_PASS, 1920, 1080, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, 100 },
    { "dontcare_store_1080p", MODE_TWO_PASS, 1920, 1080, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE, 100 },
    { "msaa4_resolve_1080p", MODE_MSAA, 1920, 1080, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE, 100 },
    { "msaa4_store_1080p", MODE_MSAA, 1920, 1080, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, 100 },
    { "local_read_1080p", MODE_LOCAL_READ, 1920, 1080, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE, 100 },
    { "clear_store_2160p", MODE_TWO_PASS, 3840, 2160, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, 60 },
    { "local_read_2160p", MODE_LOCAL_READ, 3840, 2160, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE, 60 },
};

/* gbuffer.vert's per-instance attributes. */
struct instance {
    float rect[4];
    float normal[4];
    float emissive[4];
    uint32_t albedo;
    float depth;
};

/* lighting.frag's uniform block (std140). */
struct lights_ubo {
    float pos_radius[LIGHTS][4];
    float color[LIGHTS][4];
    float inv_size[4];
    uint32_t count, pad[3];
};

struct object {
    int x0, y0, x1, y1;
    float depth;
    uint32_t albedo;
    float normal[3], emissive[3];
};

struct scene {
    struct bench_vk *vk;
    const struct config *cfg;
    uint32_t width, height; /* of the current resources */
    struct object objects[RECTS + 1];
    struct lights_ubo lights;
    float *reference; /* RGB per pixel at width x height */

    /* Per resolution. */
    struct bench_image gbuffer[GBUFFERS], depth;       /* single-sample, sampled */
    struct bench_image ms_gbuffer[GBUFFERS], ms_depth; /* 4x MSAA */
    struct bench_image lr_gbuffer[GBUFFERS], lr_depth; /* local read, transient */
    struct bench_image output[BENCH_FRAMES_IN_FLIGHT];
    struct bench_buffer instances, readback;
    int memoryless; /* the configuration's transient images are lazily allocated */

    struct bench_buffer ubo;
    VkSampler sampler;
    VkDescriptorSetLayout set_layout, lr_set_layout;
    VkPipelineLayout layout, lr_layout;
    VkDescriptorPool pool;
    VkDescriptorSet set, lr_set;
    VkPipeline gbuffer_1x, gbuffer_4x, lighting, gbuffer_lr, lighting_lr;
};

#define COLOR_OUT VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
#define FRAG_TESTS (VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
#define FRAG_SHADER VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
#define ATTACHMENT_WRITES (VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)

/* Into an attachment layout, after earlier frames' attachment writes,
 * resolves and shader reads of the image. */
static VkImageMemoryBarrier2 to_attachment(const struct bench_image *img, VkImageLayout old_layout,
                                           VkImageLayout layout)
{
    int depth = (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
    return bench_image_barrier_info(img, old_layout, layout, COLOR_OUT | FRAG_TESTS | FRAG_SHADER,
                                    ATTACHMENT_WRITES, depth ? FRAG_TESTS | COLOR_OUT : COLOR_OUT,
                                    depth ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
                                          : VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
}

/* From an attachment (or resolve) layout to sampled by the lighting pass. */
static VkImageMemoryBarrier2 to_sampled(const struct bench_image *img, VkImageLayout old_layout)
{
    return bench_image_barrier_info(img, old_layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    COLOR_OUT | FRAG_TESTS, ATTACHMENT_WRITES, FRAG_SHADER,
                                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

static VkRenderingAttachmentInfo attachment(const struct bench_image *img, VkImageLayout layout,
                                            VkAttachmentLoadOp load, VkAttachmentStoreOp store)
{
    VkRenderingAttachmentInfo a = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = img->view,
        .imageLayout = layout,
        .loadOp = load,
        .storeOp = store,
    };
    if (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)
        a.clearValue.depthStencil.depth = 1.0f;
    return a;
}

static void draw_gbuffer(struct scene *s, VkCommandBuffer cmd, VkPipeline pipeline)
{
    VkDeviceSize zero = 0;
    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkd.CmdBindVertexBuffers(cmd, 0, 1, &s->instances.buffer, &zero);
    vkd.CmdDraw(cmd, 6, RECTS + 1, 0, 0);
}

static void draw_lighting(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout,
                          VkDescriptorSet set)
{
    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
    vkd.CmdDraw(cmd, 3, 1, 0, 0);
}

static void record_two_pass(struct scene *s, const struct bench_frame *f)
{
    const struct config *cfg = s->cfg;
    VkCommandBuffer cmd = f->cmd;
    int msaa = cfg->mode == MODE_MSAA;
    VkImageMemoryBarrier2 barriers[2 * (GBUFFERS + 1) + 1];
    VkRenderingAttachmentInfo colors[GBUFFERS], depth, out;
    VkRenderingInfo info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = { { 0, 0 }, { s->width, s->height } },
        .layerCount = 1,
        .colorAttachmentCount = GBUFFERS,
        .pColorAttachments = colors,
        .pDepthAttachment = &depth,
    };
    VkImageLayout old_layout = cfg->gbuffer_load == VK_ATTACHMENT_LOAD_OP_LOAD ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                                               : VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t n = 0;
    int k;

    /* G-buffer pass. */
    for (k = 0; k < GBUFFERS; k++)
        barriers[n++] = to_attachment(&s->gbuffer[k], old_layout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    barriers[n++] = to_attachment(&s->depth, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    if (msaa) {
        for (k = 0; k < GBUFFERS; k++)
            barriers[n++] = to_attachment(&s->ms_gbuffer[k], VK_IMAGE_LAYOUT_UNDEFINED,
                                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        barriers[n++] = to_attachment(&s->ms_depth, VK_IMAGE_LAYOUT_UNDEFINED,
                                      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    }
    bench_image_barriers(cmd, n, barriers);
    for (k = 0; k < GBUFFERS; k++) {
        if (msaa) {
            colors[k] = attachment(&s->ms_gbuffer[k], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_ATTACHMENT_LOAD_OP_CLEAR, cfg->msaa_store);
            colors[k].resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            colors[k].resolveImageView = s->gbuffer[k].view;
            colors[k].resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        } else {
            colors[k] = attachment(&s->gbuffer[k], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, cfg->gbuffer_load,
                                   VK_ATTACHMENT_STORE_OP_STORE);
        }
    }
    if (msaa) {
        depth = attachment(&s->ms_depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_CLEAR,
                           cfg->msaa_store);
        depth.resolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
        depth.resolveImageView = s->depth.view;
        depth.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    } else {
        depth = attachment(&s->depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_CLEAR,
                           VK_ATTACHMENT_STORE_OP_STORE);
    }
    vkd.CmdBeginRendering(cmd, &info);
    bench_cmd_viewport(cmd, s->width, s->height);
    draw_gbuffer(s, cmd, msaa ? s->gbuffer_4x : s->gbuffer_1x);
    vkd.CmdEndRendering(cmd);

    /* Lighting pass, sampling the (resolved) G-buffer. */
    n = 0;
    for (k = 0; k < GBUFFERS; k++)
        barriers[n++] = to_sampled(&s->gbuffer[k], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    barriers[n++] = to_sampled(&s->depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    barriers[n++] = to_attachment(&s->output[f->slot], VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    barriers[n - 1].srcStageMask |= VK_PIPELINE_STAGE_2_COPY_BIT;
    bench_image_barriers(cmd, n, barriers);
    out = attachment(&s->output[f->slot], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                     VK_ATTACHMENT_STORE_OP_STORE);
    info.colorAttachmentCount = 1;
    info.pColorAttachments = &out;
    info.pDepthAttachment = NULL;
    vkd.CmdBeginRendering(cmd, &info);
    bench_cmd_viewport(cmd, s->width, s->height);
    draw_lighting(cmd, s->lighting, s->layout, s->set);
    vkd.CmdEndRendering(cmd);
}

static void record_local_read(struct scene *s, const struct bench_frame *f)
{
    static const uint32_t gbuffer_locations[GBUFFERS + 1] = { 0, 1, 2, VK_ATTACHMENT_UNUSED };
    static const uint32_t lighting_locations[GBUFFERS + 1] = { VK_ATTACHMENT_UNUSED, VK_ATTACHMENT_UNUSED,
                                                               VK_ATTACHMENT_UNUSED, 0 };
    static const uint32_t input_indices[GBUFFERS + 1] = { 0, 1, 2, VK_ATTACHMENT_UNUSED };
    static const uint32_t depth_input_index = GBUFFERS;
    VkCommandBuffer cmd = f->cmd;
    VkImageMemoryBarrier2 barriers[GBUFFERS + 2];
    VkRenderingAttachmentInfo colors[GBUFFERS + 1], depth;
    VkRenderingInfo info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = { { 0, 0 }, { s->width, s->height } },
        .layerCount = 1,
        .colorAttachmentCount = GBUFFERS + 1,
        .pColorAttachments = colors,
        .pDepthAttachment = &depth,
    };
    VkRenderingAttachmentLocationInfo locations = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO,
        .colorAttachmentCount = GBUFFERS + 1,
        .pColorAttachmentLocations = gbuffer_locations,
    };
    VkRenderingInputAttachmentIndexInfo inputs = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO,
        .colorAttachmentCount = GBUFFERS + 1,
        .pColorAttachmentInputIndices = input_indices,
        .pDepthInputAttachmentIndex = &depth_input_index,
    };
    VkMemoryBarrier2 tile_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = COLOR_OUT | FRAG_TESTS,
        .srcAccessMask = ATTACHMENT_WRITES,
        .dstStageMask = FRAG_SHADER,
        .dstAccessMask = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT,
    };
    VkDependencyInfo dep = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &tile_barrier,
    };
    int k;

    for (k = 0; k < GBUFFERS; k++)
        barriers[k] = to_attachment(&s->lr_gbuffer[k], VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ);
    barriers[GBUFFERS] = to_attachment(&s->lr_depth, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ);
    barriers[GBUFFERS + 1] = to_attachment(&s->output[f->slot], VK_IMAGE_LAYOUT_UNDEFINED,
                                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    barriers[GBUFFERS + 1].srcStageMask |= VK_PIPELINE_STAGE_2_COPY_BIT;
    bench_image_barriers(cmd, GBUFFERS + 2, barriers);

    for (k = 0; k < GBUFFERS; k++)
        colors[k] = attachment(&s->lr_gbuffer[k], VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ, VK_ATTACHMENT_LOAD_OP_CLEAR,
                               VK_ATTACHMENT_STORE_OP_DONT_CARE);
    colors[GBUFFERS] = attachment(&s->output[f->slot], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE);
    depth = attachment(&s->lr_depth, VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ, VK_ATTACHMENT_LOAD_OP_CLEAR,
                       VK_ATTACHMENT_STORE_OP_DONT_CARE);
    vkd.CmdBeginRendering(cmd, &info);
    bench_cmd_viewport(cmd, s->width, s->height);
    vkd.CmdSetRenderingAttachmentLocations(cmd, &locations);
    draw_gbuffer(s, cmd, s->gbuffer_lr);
    vkd.CmdPipelineBarrier2(cmd, &dep);
    locations.pColorAttachmentLocations = lighting_locations;
    vkd.CmdSetRenderingAttachmentLocations(cmd, &locations);
    vkd.CmdSetRenderingInputAttachmentIndices(cmd, &inputs);
    draw_lighting(cmd, s->lighting_lr, s->lr_layout, s->lr_set);
    vkd.CmdEndRendering(cmd);
}

static void record(void *user, const struct bench_frame *f)
{
    struct scene *s = user;
    if (s->cfg->mode == MODE_LOCAL_READ)
        record_local_read(s, f);
    else
        record_two_pass(s, f);
    if (f->verify)
        bench_cmd_copy_image_to_buffer(f->cmd, &s->output[f->slot], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                       COLOR_OUT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, s->readback.buffer);
}

/* ---- Scene and CPU reference ---- */

static float dyadic(uint64_t *seed, int lo, int hi, int denominator)
{
    return (float)bench_rand_range(seed, lo, hi + 1) / (float)denominator;
}

/* Objects in pixels for a width x height target. Normals and emissive have
 * few significant bits, so the RGBA16F G-buffer stores them exactly. */
static void make_scene(struct scene *s, uint32_t width, uint32_t height)
{
    uint64_t seed = 0x5CE7E5CEull;
    int scale = (int)width / 1920 > 0 ? (int)width / 1920 : 1;
    int i;

    for (i = 0; i <= RECTS; i++) {
        struct object *o = &s->objects[i];
        if (i == 0) {
            o->x0 = o->y0 = 0;
            o->x1 = (int)width;
            o->y1 = (int)height;
            o->depth = 0.999f;
            o->albedo = bench_rgba8(128, 128, 128, 255);
            o->normal[0] = o->normal[1] = 0.0f;
            o->normal[2] = -1.0f;
            o->emissive[0] = o->emissive[1] = o->emissive[2] = 0.0f;
            continue;
        }
        {
            int w = bench_rand_range(&seed, 16, 161) * scale, h = bench_rand_range(&seed, 16, 161) * scale;
            o->x0 = bench_rand_range(&seed, 0, (int)width - w);
            o->y0 = bench_rand_range(&seed, 0, (int)height - h);
            o->x1 = o->x0 + w;
            o->y1 = o->y0 + h;
        }
        /* Distinct depths in (0.05, 0.95), so the depth test has one answer. */
        o->depth = 0.05f + 0.9f * (float)((i * 7919) % RECTS) / (float)RECTS;
        o->albedo = bench_rand(&seed) | 0xFF000000u;
        o->normal[0] = dyadic(&seed, -2, 2, 4);
        o->normal[1] = dyadic(&seed, -2, 2, 4);
        o->normal[2] = -dyadic(&seed, 2, 4, 4);
        if (bench_rand(&seed) % 8 == 0) {
            o->emissive[0] = dyadic(&seed, 0, 8, 64);
            o->emissive[1] = dyadic(&seed, 0, 8, 64);
            o->emissive[2] = dyadic(&seed, 0, 8, 64);
        } else {
            o->emissive[0] = o->emissive[1] = o->emissive[2] = 0.0f;
        }
    }

    memset(&s->lights, 0, sizeof(s->lights));
    for (i = 0; i < LIGHTS; i++) {
        s->lights.pos_radius[i][0] = (float)(bench_rand(&seed) % 1024) / 1024.0f;
        s->lights.pos_radius[i][1] = (float)(bench_rand(&seed) % 1024) / 1024.0f;
        s->lights.pos_radius[i][2] = -0.25f;
        s->lights.pos_radius[i][3] = 0.25f + (float)(bench_rand(&seed) % 256) / 640.0f;
        s->lights.color[i][0] = 0.1f + (float)(bench_rand(&seed) % 256) / 512.0f;
        s->lights.color[i][1] = 0.1f + (float)(bench_rand(&seed) % 256) / 512.0f;
        s->lights.color[i][2] = 0.1f + (float)(bench_rand(&seed) % 256) / 512.0f;
    }
    s->lights.inv_size[0] = 1.0f / (float)width;
    s->lights.inv_size[1] = 1.0f / (float)height;
    s->lights.count = LIGHTS;
}

/* lighting.frag, on the CPU. */
static void make_reference(struct scene *s)
{
    uint32_t w = s->width, h = s->height, x, y;
    int *top = bench_alloc(sizeof(int) * w * h);
    float *zbuf = bench_alloc(sizeof(float) * w * h);
    int i, l;

    for (x = 0; x < w * h; x++)
        zbuf[x] = 1.0f;
    for (i = 0; i <= RECTS; i++) {
        const struct object *o = &s->objects[i];
        int px, py;
        for (py = o->y0; py < o->y1; py++)
            for (px = o->x0; px < o->x1; px++) {
                size_t p = (size_t)py * w + (size_t)px;
                if (o->depth < zbuf[p]) {
                    zbuf[p] = o->depth;
                    top[p] = i;
                }
            }
    }
    free(s->reference);
    s->reference = bench_alloc(sizeof(float) * 3 * w * h);
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            size_t p = (size_t)y * w + x;
            const struct object *o = &s->objects[top[p]];
            float albedo[3], n[3], c[3], pos[3], len;
            for (i = 0; i < 3; i++) {
                albedo[i] = (float)((o->albedo >> (8 * i)) & 0xFF) / 255.0f;
                c[i] = o->emissive[i];
            }
            len = sqrtf(o->normal[0] * o->normal[0] + o->normal[1] * o->normal[1] + o->normal[2] * o->normal[2]);
            for (i = 0; i < 3; i++)
                n[i] = o->normal[i] / len;
            pos[0] = ((float)x + 0.5f) * s->lights.inv_size[0];
            pos[1] = ((float)y + 0.5f) * s->lights.inv_size[1];
            pos[2] = o->depth;
            for (l = 0; l < LIGHTS; l++) {
                const float *lp = s->lights.pos_radius[l];
                float d[3] = { lp[0] - pos[0], lp[1] - pos[1], lp[2] - pos[2] };
                float dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                float atten = fmaxf(0.0f, 1.0f - dist2 / (lp[3] * lp[3]));
                float ndl = fmaxf(0.0f, (n[0] * d[0] + n[1] * d[1] + n[2] * d[2]) / sqrtf(fmaxf(dist2, 1e-8f)));
                for (i = 0; i < 3; i++)
                    c[i] += albedo[i] * s->lights.color[l][i] * (ndl * atten * atten);
            }
            memcpy(&s->reference[3 * p], c, sizeof(c));
        }
    }
    free(top);
    free(zbuf);
}

static int verify(struct scene *s)
{
    const uint16_t *got = s->readback.map;
    size_t p, n = (size_t)s->width * s->height, bad = 0, first = 0;
    int c;

    for (p = 0; p < n; p++) {
        int ok = bench_half_to_float(got[4 * p + 3]) == 1.0f;
        for (c = 0; c < 3 && ok; c++) {
            float g = bench_half_to_float(got[4 * p + c]), w = s->reference[3 * p + c];
            ok = fabsf(g - w) <= 3e-3f + 5e-3f * fabsf(w);
        }
        if (!ok && bad++ == 0)
            first = p;
    }
    if (!bad)
        return 1;
    bench_error(s->cfg->name, "%zu of %zu pixels wrong; first at (%zu,%zu): got (%g,%g,%g,%g), want (%g,%g,%g,1)",
                bad, n, first % s->width, first / s->width, bench_half_to_float(got[4 * first]),
                bench_half_to_float(got[4 * first + 1]), bench_half_to_float(got[4 * first + 2]),
                bench_half_to_float(got[4 * first + 3]), s->reference[3 * first], s->reference[3 * first + 1],
                s->reference[3 * first + 2]);
    return 0;
}

/* ---- Resources ---- */

static void destroy_targets(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    int k;
    vkd.DeviceWaitIdle(vk->device);
    for (k = 0; k < GBUFFERS; k++) {
        bench_image_destroy(vk, &s->gbuffer[k]);
        bench_image_destroy(vk, &s->ms_gbuffer[k]);
        bench_image_destroy(vk, &s->lr_gbuffer[k]);
    }
    bench_image_destroy(vk, &s->depth);
    bench_image_destroy(vk, &s->ms_depth);
    bench_image_destroy(vk, &s->lr_depth);
    for (k = 0; k < BENCH_FRAMES_IN_FLIGHT; k++)
        bench_image_destroy(vk, &s->output[k]);
    bench_buffer_destroy(vk, &s->instances);
    bench_buffer_destroy(vk, &s->readback);
}

/* (Re)creates everything sized by the configuration. */
static void create_targets(struct scene *s, const struct config *cfg)
{
    struct bench_vk *vk = s->vk;
    uint32_t w = cfg->width, h = cfg->height;
    VkImageUsageFlags color = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VkImageUsageFlags depth = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    int ms_transient = cfg->msaa_store == VK_ATTACHMENT_STORE_OP_DONT_CARE;
    VkDescriptorImageInfo images[GBUFFERS + 1];
    VkDescriptorBufferInfo ubo = { VK_NULL_HANDLE, 0, sizeof(struct lights_ubo) };
    VkWriteDescriptorSet writes[GBUFFERS + 2];
    struct instance *inst;
    VkCommandBuffer cmd;
    int k;

    destroy_targets(s);
    s->width = w;
    s->height = h;
    s->memoryless = 0;
    for (k = 0; k < GBUFFERS; k++)
        bench_image_create(vk, &s->gbuffer[k], gbuffer_formats[k], w, h, 1, VK_SAMPLE_COUNT_1_BIT,
                           color | VK_IMAGE_USAGE_SAMPLED_BIT, 0);
    bench_image_create(vk, &s->depth, DEPTH_FORMAT, w, h, 1, VK_SAMPLE_COUNT_1_BIT, depth | VK_IMAGE_USAGE_SAMPLED_BIT, 0);
    if (cfg->mode == MODE_MSAA) {
        for (k = 0; k < GBUFFERS; k++)
            bench_image_create(vk, &s->ms_gbuffer[k], gbuffer_formats[k], w, h, 1, VK_SAMPLE_COUNT_4_BIT, color,
                               ms_transient);
        bench_image_create(vk, &s->ms_depth, DEPTH_FORMAT, w, h, 1, VK_SAMPLE_COUNT_4_BIT, depth, ms_transient);
        s->memoryless = ms_transient && s->ms_depth.lazy && s->ms_gbuffer[0].lazy;
    }
    if (cfg->mode == MODE_LOCAL_READ) {
        for (k = 0; k < GBUFFERS; k++)
            bench_image_create(vk, &s->lr_gbuffer[k], gbuffer_formats[k], w, h, 1, VK_SAMPLE_COUNT_1_BIT,
                               color | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, 1);
        bench_image_create(vk, &s->lr_depth, DEPTH_FORMAT, w, h, 1, VK_SAMPLE_COUNT_1_BIT,
                           depth | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, 1);
        s->memoryless = s->lr_depth.lazy && s->lr_gbuffer[0].lazy;
    }
    for (k = 0; k < BENCH_FRAMES_IN_FLIGHT; k++)
        bench_image_create(vk, &s->output[k], OUTPUT_FORMAT, w, h, 1, VK_SAMPLE_COUNT_1_BIT,
                           color | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0);
    bench_buffer_create(vk, &s->readback, (VkDeviceSize)w * h * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, BENCH_HOST,
                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

    /* load_store reads last frame's G-buffer, so it must start defined. */
    cmd = bench_cmd_begin(vk);
    for (k = 0; k < GBUFFERS; k++)
        bench_image_barrier(cmd, &s->gbuffer[k], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_NONE, 0, FRAG_SHADER, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    bench_cmd_finish(vk, cmd);

    /* The scene at this resolution. */
    make_scene(s, w, h);
    inst = bench_alloc(sizeof(*inst) * (RECTS + 1));
    for (k = 0; k <= RECTS; k++) {
        const struct object *o = &s->objects[k];
        struct bench_rect r = { o->x0, o->y0, o->x1, o->y1, 0 };
        bench_rect_ndc(&r, (int)w, (int)h, 0, 0, inst[k].rect);
        memcpy(inst[k].normal, o->normal, sizeof(o->normal));
        inst[k].normal[3] = 0.5f;
        memcpy(inst[k].emissive, o->emissive, sizeof(o->emissive));
        inst[k].emissive[3] = 1.0f;
        inst[k].albedo = o->albedo;
        inst[k].depth = o->depth;
    }
    bench_buffer_create(vk, &s->instances, sizeof(*inst) * (RECTS + 1),
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->instances, 0, inst, sizeof(*inst) * (RECTS + 1));
    free(inst);
    bench_buffer_upload(vk, &s->ubo, 0, &s->lights, sizeof(s->lights));
    make_reference(s);

    /* Descriptors pointing at the new images. */
    ubo.buffer = s->ubo.buffer;
    for (k = 0; k < GBUFFERS + 2; k++) {
        memset(&writes[k], 0, sizeof(writes[k]));
        writes[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[k].dstBinding = (uint32_t)k;
        writes[k].descriptorCount = 1;
        writes[k].pImageInfo = &images[k < GBUFFERS + 1 ? k : 0];
    }
    writes[GBUFFERS + 1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[GBUFFERS + 1].pImageInfo = NULL;
    writes[GBUFFERS + 1].pBufferInfo = &ubo;
    for (k = 0; k <= GBUFFERS; k++) {
        const struct bench_image *img = k < GBUFFERS ? &s->gbuffer[k] : &s->depth;
        images[k].sampler = s->sampler;
        images[k].imageView = img->view;
        images[k].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        writes[k].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[k].dstSet = s->set;
    }
    writes[GBUFFERS + 1].dstSet = s->set;
    vkd.UpdateDescriptorSets(vk->device, GBUFFERS + 2, writes, 0, NULL);
    if (cfg->mode == MODE_LOCAL_READ) {
        for (k = 0; k <= GBUFFERS; k++) {
            const struct bench_image *img = k < GBUFFERS ? &s->lr_gbuffer[k] : &s->lr_depth;
            images[k].sampler = VK_NULL_HANDLE;
            images[k].imageView = img->view;
            images[k].imageLayout = VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ;
            writes[k].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            writes[k].dstSet = s->lr_set;
        }
        writes[GBUFFERS + 1].dstSet = s->lr_set;
        vkd.UpdateDescriptorSets(vk->device, GBUFFERS + 2, writes, 0, NULL);
    }
}

static VkPipeline gbuffer_pipeline(struct scene *s, VkSampleCountFlagBits samples, int local_read)
{
    static const uint32_t locations[GBUFFERS + 1] = { 0, 1, 2, VK_ATTACHMENT_UNUSED };
    VkRenderingAttachmentLocationInfo location_info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO,
        .colorAttachmentCount = GBUFFERS + 1,
        .pColorAttachmentLocations = locations,
    };
    struct bench_vk *vk = s->vk;
    struct bench_gfx_desc d;
    VkPipeline p;
    int k;

    memset(&d, 0, sizeof(d));
    d.vs = BENCH_SHADER(vk, gbuffer_vert);
    d.fs = BENCH_SHADER(vk, gbuffer_frag);
    d.layout = s->layout;
    d.color_count = GBUFFERS;
    for (k = 0; k < GBUFFERS; k++)
        d.color_formats[k] = gbuffer_formats[k];
    if (local_read) {
        d.color_count = GBUFFERS + 1;
        d.color_formats[GBUFFERS] = OUTPUT_FORMAT;
        d.rendering_next = &location_info;
    }
    d.depth_format = DEPTH_FORMAT;
    d.samples = samples;
    d.depth_test = d.depth_write = 1;
    d.depth_compare = VK_COMPARE_OP_LESS;
    d.binding_count = 1;
    d.bindings[0] = (VkVertexInputBindingDescription){ 0, sizeof(struct instance), VK_VERTEX_INPUT_RATE_INSTANCE };
    d.attribute_count = 5;
    d.attributes[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(struct instance, rect) };
    d.attributes[1] = (VkVertexInputAttributeDescription){ 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(struct instance, normal) };
    d.attributes[2] = (VkVertexInputAttributeDescription){ 2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(struct instance, emissive) };
    d.attributes[3] = (VkVertexInputAttributeDescription){ 3, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(struct instance, albedo) };
    d.attributes[4] = (VkVertexInputAttributeDescription){ 4, 0, VK_FORMAT_R32_SFLOAT, offsetof(struct instance, depth) };
    p = bench_gfx_pipeline(vk, &d);
    vkd.DestroyShaderModule(vk->device, d.vs, NULL);
    vkd.DestroyShaderModule(vk->device, d.fs, NULL);
    return p;
}

static VkPipeline lighting_pipeline(struct scene *s, int local_read)
{
    static const uint32_t locations[GBUFFERS + 1] = { VK_ATTACHMENT_UNUSED, VK_ATTACHMENT_UNUSED,
                                                      VK_ATTACHMENT_UNUSED, 0 };
    static const uint32_t input_indices[GBUFFERS + 1] = { 0, 1, 2, VK_ATTACHMENT_UNUSED };
    static const uint32_t depth_input_index = GBUFFERS;
    VkRenderingInputAttachmentIndexInfo input_info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO,
        .colorAttachmentCount = GBUFFERS + 1,
        .pColorAttachmentInputIndices = input_indices,
        .pDepthInputAttachmentIndex = &depth_input_index,
    };
    VkRenderingAttachmentLocationInfo location_info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO,
        .pNext = &input_info,
        .colorAttachmentCount = GBUFFERS + 1,
        .pColorAttachmentLocations = locations,
    };
    struct bench_vk *vk = s->vk;
    struct bench_gfx_desc d;
    VkPipeline p;
    int k;

    memset(&d, 0, sizeof(d));
    d.vs = BENCH_SHADER(vk, fullscreen_vert);
    d.color_count = 1;
    d.color_formats[0] = OUTPUT_FORMAT;
    if (local_read) {
        d.fs = BENCH_SHADER(vk, lighting_local_frag);
        d.layout = s->lr_layout;
        d.color_count = GBUFFERS + 1;
        for (k = 0; k < GBUFFERS; k++)
            d.color_formats[k] = gbuffer_formats[k];
        d.color_formats[GBUFFERS] = OUTPUT_FORMAT;
        d.depth_format = DEPTH_FORMAT;
        d.rendering_next = &location_info;
    } else {
        d.fs = BENCH_SHADER(vk, lighting_frag);
        d.layout = s->layout;
    }
    p = bench_gfx_pipeline(vk, &d);
    vkd.DestroyShaderModule(vk->device, d.vs, NULL);
    vkd.DestroyShaderModule(vk->device, d.fs, NULL);
    return p;
}

static void setup(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDescriptorSetLayoutBinding bindings[GBUFFERS + 2];
    VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    };
    VkDescriptorPoolSize sizes[3] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, GBUFFERS + 1 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, GBUFFERS + 1 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 },
    };
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = 3,
        .pPoolSizes = sizes,
    };
    VkDescriptorSetAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorSetCount = 1,
    };
    int k;

    BENCH_VK(vkd.CreateSampler(vk->device, &sampler_info, NULL, &s->sampler));
    bench_buffer_create(vk, &s->ubo, sizeof(struct lights_ubo),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, BENCH_DEVICE_LOCAL);
    for (k = 0; k < GBUFFERS + 2; k++) {
        bindings[k].binding = (uint32_t)k;
        bindings[k].descriptorType = k <= GBUFFERS ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                   : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[k].descriptorCount = 1;
        bindings[k].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[k].pImmutableSamplers = NULL;
    }
    s->set_layout = bench_set_layout(vk, GBUFFERS + 2, bindings, 0);
    s->layout = bench_pipeline_layout(vk, 1, &s->set_layout, 0, 0);
    BENCH_VK(vkd.CreateDescriptorPool(vk->device, &pool_info, NULL, &s->pool));
    alloc.descriptorPool = s->pool;
    alloc.pSetLayouts = &s->set_layout;
    BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &alloc, &s->set));
    s->gbuffer_1x = gbuffer_pipeline(s, VK_SAMPLE_COUNT_1_BIT, 0);
    s->lighting = lighting_pipeline(s, 0);
    if (vk->props.limits.framebufferColorSampleCounts & vk->props.limits.framebufferDepthSampleCounts &
        VK_SAMPLE_COUNT_4_BIT)
        s->gbuffer_4x = gbuffer_pipeline(s, VK_SAMPLE_COUNT_4_BIT, 0);
    if (vk->local_read && vk->local_read_depth && vkd.CmdSetRenderingAttachmentLocations &&
        vkd.CmdSetRenderingInputAttachmentIndices) {
        for (k = 0; k <= GBUFFERS; k++)
            bindings[k].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        s->lr_set_layout = bench_set_layout(vk, GBUFFERS + 2, bindings, 0);
        s->lr_layout = bench_pipeline_layout(vk, 1, &s->lr_set_layout, 0, 0);
        alloc.pSetLayouts = &s->lr_set_layout;
        BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &alloc, &s->lr_set));
        s->gbuffer_lr = gbuffer_pipeline(s, VK_SAMPLE_COUNT_1_BIT, 1);
        s->lighting_lr = lighting_pipeline(s, 1);
    }
}

static void teardown(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDevice dev = vk->device;
    destroy_targets(s);
    vkd.DestroyPipeline(dev, s->gbuffer_1x, NULL);
    vkd.DestroyPipeline(dev, s->lighting, NULL);
    if (s->gbuffer_4x)
        vkd.DestroyPipeline(dev, s->gbuffer_4x, NULL);
    if (s->gbuffer_lr) {
        vkd.DestroyPipeline(dev, s->gbuffer_lr, NULL);
        vkd.DestroyPipeline(dev, s->lighting_lr, NULL);
        vkd.DestroyPipelineLayout(dev, s->lr_layout, NULL);
        vkd.DestroyDescriptorSetLayout(dev, s->lr_set_layout, NULL);
    }
    vkd.DestroyDescriptorPool(dev, s->pool, NULL);
    vkd.DestroyPipelineLayout(dev, s->layout, NULL);
    vkd.DestroyDescriptorSetLayout(dev, s->set_layout, NULL);
    vkd.DestroySampler(dev, s->sampler, NULL);
    bench_buffer_destroy(vk, &s->ubo);
    free(s->reference);
}

static const char *op_name(int load, int op)
{
    if (load)
        return op == VK_ATTACHMENT_LOAD_OP_LOAD ? "load" : op == VK_ATTACHMENT_LOAD_OP_CLEAR ? "clear" : "dont_care";
    return op == VK_ATTACHMENT_STORE_OP_STORE ? "store" : "dont_care";
}

int main(void)
{
    struct bench_vk vk;
    struct bench_loop loop;
    struct scene *s;
    size_t c;

    bench_begin("render_passes");
    bench_vk_init(&vk, BENCH_NEED_DYNAMIC_RENDERING | BENCH_NEED_SYNC2 | BENCH_WANT_LOCAL_READ);
    s = bench_alloc(sizeof(*s));
    s->vk = &vk;
    setup(s);
    bench_loop_init(&loop, &vk, 1000);

    for (c = 0; c < sizeof(configs) / sizeof(configs[0]); c++) {
        const struct config *cfg = &configs[c];
        int frames = bench_frames(cfg->frames);
        struct bench_json j;

        if (!bench_selected(cfg->name))
            continue;
        bench_config = cfg->name;
        if (cfg->mode == MODE_MSAA && !s->gbuffer_4x) {
            bench_skip(cfg->name, "4x MSAA color+depth attachments not supported");
            continue;
        }
        if (cfg->mode == MODE_LOCAL_READ && !s->gbuffer_lr) {
            bench_skip(cfg->name, "dynamicRenderingLocalRead (with depth input attachments) not supported");
            continue;
        }
        s->cfg = cfg;
        create_targets(s, cfg);
        bench_loop_verify(&loop, record, s);
        if (!verify(s))
            continue;
        bench_loop_run(&loop, record, s, 10, frames < 1000 ? frames : 1000);

        bench_json_begin(&j, cfg->name);
        bench_json_int(&j, "width", cfg->width);
        bench_json_int(&j, "height", cfg->height);
        bench_json_int(&j, "samples", cfg->mode == MODE_MSAA ? 4 : 1);
        bench_json_int(&j, "objects", RECTS + 1);
        bench_json_int(&j, "lights", LIGHTS);
        bench_json_str(&j, "gbuffer_load", op_name(1, cfg->mode == MODE_TWO_PASS ? (int)cfg->gbuffer_load
                                                                                    : (int)VK_ATTACHMENT_LOAD_OP_CLEAR));
        bench_json_str(&j, "gbuffer_store", op_name(0, cfg->mode == MODE_TWO_PASS ? (int)VK_ATTACHMENT_STORE_OP_STORE
                                                                                     : (int)cfg->msaa_store));
        if (cfg->mode != MODE_TWO_PASS)
            bench_json_bool(&j, "memoryless", s->memoryless);
        bench_loop_report(&loop, &j);
        bench_json_end(&j);
    }
    bench_config = NULL;

    bench_loop_destroy(&loop);
    teardown(s);
    free(s);
    bench_vk_destroy(&vk);
    return bench_exit_code();
}
