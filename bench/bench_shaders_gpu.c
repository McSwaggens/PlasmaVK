/* Shader-bound GPU passes: fragment shaders whose cost is their own code, in
 * the shapes engines write them. One full-screen pass per frame into an
 * RGBA16F target; gpu_us is the number to watch.
 *
 *   pom_*          parallax occlusion mapping: a height-field ray march in a
 *                  helper that returns from inside its loop (the way POM,
 *                  screen-space reflections, SDF marching and PCSS blocker
 *                  searches are written), then normal-mapped lighting
 *   deferred_*     a deferred-lighting composite: G-buffer reads, early
 *                  returns for a debug view and for sky before a loop over
 *                  32 lights
 *   forward_pbr_*  forward PBR over a mesh of 30-pixel triangles: three
 *                  material maps, 5x5 PCF over a cascaded shadow-map array,
 *                  32 lights read from a storage buffer (each one copied,
 *                  those out of range skipped), image-based lighting
 *
 * 1080p and 2160p. The first frame of each configuration is read back and
 * compared with the shaders evaluated on the CPU, within fp16 rounding (at
 * 2160p every other pixel in each direction). The content keeps the image
 * continuous - smooth textures, lights that fade to zero at their range, a
 * height field shallow enough that the ray march finds the same surface
 * whichever step it stops at - so a rounding difference cannot flip a pixel.
 */
#include <math.h>

#include "bench_vk.h"
#include "bench_shaders.h"

#define OUTPUT_FORMAT VK_FORMAT_R16G16B16A16_SFLOAT
#define WARMUP 300 /* frames: short passes need the GPU clock ramped up */
#define LIGHTS 32
#define PI_F 3.14159265f

enum kind { KIND_POM, KIND_DEFERRED, KIND_PBR };

struct config {
    const char *name;
    enum kind kind;
    uint32_t width, height;
    int frames;
};

static const struct config configs[] = {
    { "pom_1080p", KIND_POM, 1920, 1080, 200 },
    { "pom_2160p", KIND_POM, 3840, 2160, 100 },
    { "deferred_1080p", KIND_DEFERRED, 1920, 1080, 200 },
    { "deferred_2160p", KIND_DEFERRED, 3840, 2160, 100 },
    { "forward_pbr_1080p", KIND_PBR, 1920, 1080, 200 },
    { "forward_pbr_2160p", KIND_PBR, 3840, 2160, 100 },
};

/* ==========================================================================
 * Small vector math, shared by the CPU references
 * ========================================================================== */

static float dot3(const float *a, const float *b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void normalize3(float *v)
{
    float s = 1.0f / sqrtf(dot3(v, v));
    v[0] *= s;
    v[1] *= s;
    v[2] *= s;
}

static float clampf(float x, float lo, float hi)
{
    return x < lo ? lo : x > hi ? hi : x;
}

/* Exact for 0 and the normal numbers with at most 11 significant bits used
 * here (k / 64). */
static uint16_t float_to_half(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    if ((bits & 0x7FFFFFFFu) == 0)
        return (uint16_t)((bits >> 16) & 0x8000u);
    return (uint16_t)(((bits >> 16) & 0x8000u) | ((((bits >> 23) & 0xFFu) - 112u) << 10) | ((bits >> 13) & 0x3FFu));
}

static uint32_t unorm8(float x)
{
    return (uint32_t)(clampf(x, 0.0f, 1.0f) * 255.0f + 0.5f);
}

/* An RGBA8 texture's texels on the CPU, sampled like a LINEAR sampler with
 * REPEAT (or CLAMP_TO_EDGE) addressing at mip 0. */
struct cpu_texture {
    const uint32_t *texels;
    int width, height, repeat;
};

static int wrap(int i, int n, int repeat)
{
    if (repeat)
        return ((i % n) + n) % n;
    return i < 0 ? 0 : i >= n ? n - 1 : i;
}

static void sample_bilinear(const struct cpu_texture *t, float u, float v, float out[4])
{
    float x = u * (float)t->width - 0.5f, y = v * (float)t->height - 0.5f;
    float fx = floorf(x), fy = floorf(y), ax = x - fx, ay = y - fy;
    int x0 = wrap((int)fx, t->width, t->repeat), x1 = wrap((int)fx + 1, t->width, t->repeat);
    int y0 = wrap((int)fy, t->height, t->repeat), y1 = wrap((int)fy + 1, t->height, t->repeat);
    uint32_t a = t->texels[y0 * t->width + x0], b = t->texels[y0 * t->width + x1];
    uint32_t c = t->texels[y1 * t->width + x0], d = t->texels[y1 * t->width + x1];
    int k;
    for (k = 0; k < 4; k++) {
        float top = (float)((a >> (8 * k)) & 0xFF) * (1.0f - ax) + (float)((b >> (8 * k)) & 0xFF) * ax;
        float bottom = (float)((c >> (8 * k)) & 0xFF) * (1.0f - ax) + (float)((d >> (8 * k)) & 0xFF) * ax;
        out[k] = (top * (1.0f - ay) + bottom * ay) / 255.0f;
    }
}

/* ==========================================================================
 * Resources
 * ========================================================================== */

/* A sampled image with all its layers in one view. */
struct texture {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
};

/* Creates the image and fills every layer of mip 0 from data (layers packed
 * one after another), leaving it SHADER_READ_ONLY. */
static void texture_create(struct bench_vk *vk, struct texture *t, VkFormat format, uint32_t width, uint32_t height,
                           uint32_t layers, VkImageViewType view_type, const void *data, VkDeviceSize size)
{
    VkImageAspectFlags aspect = bench_format_aspect(format);
    VkImageCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = view_type == VK_IMAGE_VIEW_TYPE_CUBE ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = { width, height, 1 },
        .mipLevels = 1,
        .arrayLayers = layers,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImageViewCreateInfo view = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .viewType = view_type,
        .format = format,
        .subresourceRange = { aspect, 0, 1, 0, layers },
    };
    VkBufferImageCopy region = {
        .imageSubresource = { aspect, 0, 0, layers },
        .imageExtent = { width, height, 1 },
    };
    VkImageMemoryBarrier2 barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
        .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .subresourceRange = { aspect, 0, 1, 0, layers },
    };
    struct bench_buffer staging;
    VkMemoryRequirements req;
    VkCommandBuffer cmd;

    memset(t, 0, sizeof(*t));
    BENCH_VK(vkd.CreateImage(vk->device, &info, NULL, &t->image));
    vkd.GetImageMemoryRequirements(vk->device, t->image, &req);
    t->memory = bench_allocate(vk, &req, 0, BENCH_DEVICE_LOCAL, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, NULL);
    BENCH_VK(vkd.BindImageMemory(vk->device, t->image, t->memory, 0));
    view.image = t->image;
    BENCH_VK(vkd.CreateImageView(vk->device, &view, NULL, &t->view));

    bench_buffer_create(vk, &staging, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, BENCH_HOST, 0);
    memcpy(staging.map, data, (size_t)size);
    cmd = bench_cmd_begin(vk);
    barrier.image = t->image;
    bench_image_barriers(cmd, 1, &barrier);
    vkd.CmdCopyBufferToImage(cmd, staging.buffer, t->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bench_image_barriers(cmd, 1, &barrier);
    bench_cmd_finish(vk, cmd);
    bench_buffer_destroy(vk, &staging);
}

static void texture_destroy(struct bench_vk *vk, struct texture *t)
{
    if (t->view)
        vkd.DestroyImageView(vk->device, t->view, NULL);
    if (t->image)
        vkd.DestroyImage(vk->device, t->image, NULL);
    if (t->memory)
        vkd.FreeMemory(vk->device, t->memory, NULL);
    memset(t, 0, sizeof(*t));
}

static VkSampler make_sampler(struct bench_vk *vk, VkFilter filter, VkSamplerAddressMode address, int compare)
{
    VkSampler sampler = VK_NULL_HANDLE;
    VkSamplerCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = filter,
        .minFilter = filter,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = address,
        .addressModeV = address,
        .addressModeW = address,
        .compareEnable = compare ? VK_TRUE : VK_FALSE,
        .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
    };
    BENCH_VK(vkd.CreateSampler(vk->device, &info, NULL, &sampler));
    return sampler;
}

static void write_image(struct bench_vk *vk, VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                        VkImageView view, VkSampler sampler)
{
    VkDescriptorImageInfo image = { sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = type,
        .pImageInfo = &image,
    };
    vkd.UpdateDescriptorSets(vk->device, 1, &write, 0, NULL);
}

static void write_buffer(struct bench_vk *vk, VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                         const struct bench_buffer *b)
{
    VkDescriptorBufferInfo buffer = { b->buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = type,
        .pBufferInfo = &buffer,
    };
    vkd.UpdateDescriptorSets(vk->device, 1, &write, 0, NULL);
}

/* A host-visible uniform, storage or vertex buffer holding data. */
static void buffer_with(struct bench_vk *vk, struct bench_buffer *b, VkBufferUsageFlags usage, const void *data,
                        size_t size)
{
    bench_buffer_create(vk, b, size, usage, BENCH_HOST, 0);
    memcpy(b->map, data, size);
}

/* ==========================================================================
 * The scene of one configuration
 * ========================================================================== */

#define MAX_TEXTURES 8
#define MAX_BUFFERS 4
#define MAX_SAMPLERS 3

struct scene {
    struct bench_vk *vk;
    const struct config *cfg;
    uint32_t width, height;
    int stride; /* the reference covers every stride-th pixel in each direction */

    struct bench_image target;
    struct bench_buffer readback;
    struct texture textures[MAX_TEXTURES];
    struct bench_buffer buffers[MAX_BUFFERS];
    VkSampler samplers[MAX_SAMPLERS];
    VkDescriptorSetLayout set_layouts[2];
    VkPipelineLayout layout;
    VkDescriptorPool pool;
    VkDescriptorSet sets[2];
    uint32_t set_count;
    VkShaderModule vs, fs;
    VkPipeline pipeline;
    VkBuffer vertex_buffer; /* forward PBR */
    uint32_t vertex_count;

    float *reference; /* RGB per checked pixel */
    uint32_t *cpu[4]; /* texels the reference samples */
};

static void scene_layouts(struct scene *s, uint32_t set_count, const VkDescriptorSetLayoutBinding *const *bindings,
                          const uint32_t *counts)
{
    struct bench_vk *vk = s->vk;
    VkDescriptorPoolSize sizes[5] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8 },
        { VK_DESCRIPTOR_TYPE_SAMPLER, 4 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 },
    };
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = 5,
        .pPoolSizes = sizes,
    };
    VkDescriptorSetAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorSetCount = set_count,
        .pSetLayouts = s->set_layouts,
    };
    uint32_t i;

    for (i = 0; i < set_count; i++)
        s->set_layouts[i] = bench_set_layout(vk, counts[i], bindings[i], 0);
    s->set_count = set_count;
    s->layout = bench_pipeline_layout(vk, set_count, s->set_layouts, 0, 0);
    BENCH_VK(vkd.CreateDescriptorPool(vk->device, &pool_info, NULL, &s->pool));
    alloc.descriptorPool = s->pool;
    BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &alloc, s->sets));
}

static void scene_pipeline(struct scene *s, struct bench_gfx_desc *d)
{
    d->layout = s->layout;
    d->color_count = 1;
    d->color_formats[0] = OUTPUT_FORMAT;
    s->pipeline = bench_gfx_pipeline(s->vk, d);
}

/* ---- pom_*: sg_pom.frag ---- */

#define POM_TEX 256

/* sg_pom.frag's uniform block. */
struct pom_params {
    float view[4], light_dir[4], pom[4];
};

static const struct pom_params pom_params = {
    { 1.6f, 0.9f, 0.5f, 0.0f },
    { 0.3f, 0.4f, 0.8660254f, 0.0f },
    { 0.06f, 8.0f, 32.0f, 4.0f }, /* height scale, min and max layers, repeats */
};

/* The height field: low frequencies only. Its steepest slope times the
 * march step stays below the layer depth, so near-ties in the march resolve
 * to the same intersection. */
static float pom_height(float u, float v)
{
    return 0.5f + 0.25f * sinf(2.0f * PI_F * u) * cosf(4.0f * PI_F * v) + 0.12f * sinf(2.0f * PI_F * (3.0f * u + 2.0f * v));
}

static void pom_setup(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDescriptorSetLayoutBinding bindings[4] = {
        { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
    };
    const VkDescriptorSetLayoutBinding *sets[1] = { bindings };
    uint32_t counts[1] = { 4 };
    struct bench_gfx_desc d;
    int i, x, y;

    for (i = 0; i < 3; i++)
        s->cpu[i] = bench_alloc(sizeof(uint32_t) * POM_TEX * POM_TEX);
    for (y = 0; y < POM_TEX; y++) {
        for (x = 0; x < POM_TEX; x++) {
            float u = ((float)x + 0.5f) / POM_TEX, v = ((float)y + 0.5f) / POM_TEX, e = 1.0f / 1024.0f;
            float n[3] = { -(pom_height(u + e, v) - pom_height(u - e, v)) / (2.0f * e) * 0.05f,
                           -(pom_height(u, v + e) - pom_height(u, v - e)) / (2.0f * e) * 0.05f, 1.0f };
            normalize3(n);
            s->cpu[0][y * POM_TEX + x] = bench_rgba8(unorm8(pom_height(u, v)), 0, 0, 255);
            s->cpu[1][y * POM_TEX + x] = bench_rgba8(unorm8(0.5f + 0.35f * sinf(2.0f * PI_F * u)),
                                                     unorm8(0.5f + 0.35f * sinf(2.0f * PI_F * (v + 0.3f))),
                                                     unorm8(0.5f + 0.35f * cosf(2.0f * PI_F * (u + v))), 255);
            s->cpu[2][y * POM_TEX + x] = bench_rgba8(unorm8(n[0] * 0.5f + 0.5f), unorm8(n[1] * 0.5f + 0.5f),
                                                     unorm8(n[2] * 0.5f + 0.5f), 255);
        }
    }
    for (i = 0; i < 3; i++)
        texture_create(vk, &s->textures[i], VK_FORMAT_R8G8B8A8_UNORM, POM_TEX, POM_TEX, 1, VK_IMAGE_VIEW_TYPE_2D,
                       s->cpu[i], sizeof(uint32_t) * POM_TEX * POM_TEX);
    buffer_with(vk, &s->buffers[0], VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &pom_params, sizeof(pom_params));
    s->samplers[0] = make_sampler(vk, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, 0);

    scene_layouts(s, 1, sets, counts);
    for (i = 0; i < 3; i++)
        write_image(vk, s->sets[0], (uint32_t)i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s->textures[i].view,
                    s->samplers[0]);
    write_buffer(vk, s->sets[0], 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &s->buffers[0]);

    memset(&d, 0, sizeof(d));
    d.vs = s->vs = BENCH_SHADER(vk, sg_fullscreen_vert);
    d.fs = s->fs = BENCH_SHADER(vk, sg_pom_frag);
    scene_pipeline(s, &d);
    s->vertex_count = 3;
}

static void pom_shade(const struct scene *s, float u, float v, float out[3])
{
    const struct pom_params *p = &pom_params;
    struct cpu_texture height = { s->cpu[0], POM_TEX, POM_TEX, 1 }, color = { s->cpu[1], POM_TEX, POM_TEX, 1 };
    struct cpu_texture normal = { s->cpu[2], POM_TEX, POM_TEX, 1 };
    float view[3] = { (u - 0.5f) * p->view[0], (v - 0.5f) * p->view[1], p->view[2] };
    float t[2], delta[2], texel[4], n[3], layers, layer_depth, depth = 0.0f, h, ndl;
    int i, k;

    normalize3(view);
    t[0] = u * p->pom[3];
    t[1] = v * p->pom[3];
    layers = p->pom[2] * (1.0f - fabsf(view[2])) + p->pom[1] * fabsf(view[2]);
    layer_depth = 1.0f / layers;
    delta[0] = view[0] / view[2] * p->pom[0] / layers;
    delta[1] = view[1] / view[2] * p->pom[0] / layers;
    sample_bilinear(&height, t[0], t[1], texel);
    h = 1.0f - texel[0];
    for (i = 0; i < 64; i++) {
        if (depth >= h) {
            float prev[2] = { t[0] + delta[0], t[1] + delta[1] }, after = h - depth, before, w;
            sample_bilinear(&height, prev[0], prev[1], texel);
            before = 1.0f - texel[0] - depth + layer_depth;
            w = after / (after - before);
            t[0] = t[0] * (1.0f - w) + prev[0] * w;
            t[1] = t[1] * (1.0f - w) + prev[1] * w;
            break;
        }
        t[0] -= delta[0];
        t[1] -= delta[1];
        sample_bilinear(&height, t[0], t[1], texel);
        h = 1.0f - texel[0];
        depth += layer_depth;
    }
    sample_bilinear(&normal, t[0], t[1], texel);
    for (k = 0; k < 3; k++)
        n[k] = texel[k] * 2.0f - 1.0f;
    normalize3(n);
    ndl = dot3(n, p->light_dir);
    ndl = ndl > 0.0f ? ndl : 0.0f;
    sample_bilinear(&color, t[0], t[1], texel);
    for (k = 0; k < 3; k++)
        out[k] = texel[k] * (0.1f + 0.9f * ndl);
}

/* ---- deferred_*: sg_deferred.frag ---- */

/* sg_deferred.frag's uniform block (std140). */
struct deferred_frame {
    float inv_view_proj[16]; /* column-major */
    float camera[4], sky[4], ambient[4];
    int32_t debug_view, light_count, pad[2];
    float light_pos[64][4]; /* xyz, w radius */
    float light_color[64][4];
};

/* The inverse of a perspective projection (60 degree vertical field of view,
 * depth 0..1, near 0.1, far 100) followed by a camera at camera[]. */
static void deferred_matrix(struct deferred_frame *f, float aspect)
{
    float tan_half = 0.57735027f, near_z = 0.1f, far_z = 100.0f;
    float a = far_z / (near_z - far_z), b = near_z * far_z / (near_z - far_z);
    float *m = f->inv_view_proj;
    memset(m, 0, sizeof(f->inv_view_proj));
    /* Clip (x, y, z, 1) to view (x aspect tan, y tan, -1, (z + a) / b). */
    m[0] = aspect * tan_half;
    m[5] = tan_half;
    m[11] = 1.0f / b;
    m[14] = -1.0f;
    m[15] = a / b;
    /* Then the camera translation: world = view + camera w. */
    m[8] = f->camera[0] / b;
    m[9] = f->camera[1] / b;
    m[10] = f->camera[2] / b;
    m[12] = f->camera[0] * m[15];
    m[13] = f->camera[1] * m[15];
    m[14] += f->camera[2] * m[15];
}

static void deferred_setup(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDescriptorSetLayoutBinding bindings[4] = {
        { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
    };
    const VkDescriptorSetLayoutBinding *sets[1] = { bindings };
    uint32_t counts[1] = { 4 };
    uint32_t w = s->width, h = s->height, x, y;
    size_t n = (size_t)w * h;
    struct deferred_frame *f = bench_alloc(sizeof(*f));
    uint16_t *normals = bench_alloc(sizeof(uint16_t) * 4 * n);
    uint64_t seed = 0xDEFE22EDull;
    struct bench_gfx_desc d;
    int i;

    /* G-buffer: random albedo and normals (sampled at texel centres with a
     * NEAREST sampler, so any content is exact), a sky band above a wavy
     * horizon, and depth receding towards it. */
    s->cpu[0] = bench_alloc(sizeof(uint32_t) * n);
    s->cpu[1] = bench_alloc(sizeof(float) * 4 * n); /* normals as floats */
    s->cpu[2] = bench_alloc(sizeof(float) * n);     /* depth */
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            size_t p = (size_t)y * w + x;
            float u = ((float)x + 0.5f) / (float)w, v = ((float)y + 0.5f) / (float)h;
            float horizon = 0.25f + 0.05f * sinf(4.0f * PI_F * u);
            float *normal = (float *)s->cpu[1] + 4 * p;
            int k;
            s->cpu[0][p] = (bench_rand(&seed) & 0x00FFFFFFu) | ((bench_rand(&seed) & 0x7Fu) << 24);
            normal[0] = (float)bench_rand_range(&seed, -32, 33) / 64.0f;
            normal[1] = (float)bench_rand_range(&seed, -32, 33) / 64.0f;
            normal[2] = (float)bench_rand_range(&seed, -64, -15) / 64.0f;
            normal[3] = 0.0f;
            for (k = 0; k < 4; k++)
                normals[4 * p + (size_t)k] = float_to_half(normal[k]);
            ((float *)s->cpu[2])[p] = v < horizon ? 1.0f
                                                  : 0.97f + 0.025f * (1.0f - v) + (float)(bench_rand(&seed) & 63) / 65536.0f;
        }
    }
    texture_create(vk, &s->textures[0], VK_FORMAT_R8G8B8A8_UNORM, w, h, 1, VK_IMAGE_VIEW_TYPE_2D, s->cpu[0],
                   sizeof(uint32_t) * n);
    texture_create(vk, &s->textures[1], VK_FORMAT_R16G16B16A16_SFLOAT, w, h, 1, VK_IMAGE_VIEW_TYPE_2D, normals,
                   sizeof(uint16_t) * 4 * n);
    texture_create(vk, &s->textures[2], VK_FORMAT_R32_SFLOAT, w, h, 1, VK_IMAGE_VIEW_TYPE_2D, s->cpu[2],
                   sizeof(float) * n);
    free(normals);

    f->camera[0] = 0.0f;
    f->camera[1] = 1.5f;
    f->camera[2] = 0.0f;
    f->sky[0] = 0.4f;
    f->sky[1] = 0.6f;
    f->sky[2] = 0.9f;
    f->sky[3] = 1.0f;
    f->ambient[0] = f->ambient[1] = 0.05f;
    f->ambient[2] = 0.06f;
    f->light_count = LIGHTS;
    deferred_matrix(f, (float)w / (float)h);
    for (i = 0; i < LIGHTS; i++) {
        f->light_pos[i][0] = (float)bench_rand_range(&seed, -600, 601) / 100.0f;
        f->light_pos[i][1] = (float)bench_rand_range(&seed, -50, 351) / 100.0f;
        f->light_pos[i][2] = -(float)bench_rand_range(&seed, 300, 1501) / 100.0f;
        f->light_pos[i][3] = (float)bench_rand_range(&seed, 300, 701) / 100.0f;
        f->light_color[i][0] = (float)bench_rand_range(&seed, 20, 101) / 100.0f;
        f->light_color[i][1] = (float)bench_rand_range(&seed, 20, 101) / 100.0f;
        f->light_color[i][2] = (float)bench_rand_range(&seed, 20, 101) / 100.0f;
    }
    buffer_with(vk, &s->buffers[0], VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, f, sizeof(*f));
    s->cpu[3] = (uint32_t *)f;
    s->samplers[0] = make_sampler(vk, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0);

    scene_layouts(s, 1, sets, counts);
    for (i = 0; i < 3; i++)
        write_image(vk, s->sets[0], (uint32_t)i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s->textures[i].view,
                    s->samplers[0]);
    write_buffer(vk, s->sets[0], 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &s->buffers[0]);

    memset(&d, 0, sizeof(d));
    d.vs = s->vs = BENCH_SHADER(vk, sg_fullscreen_vert);
    d.fs = s->fs = BENCH_SHADER(vk, sg_deferred_frag);
    scene_pipeline(s, &d);
    s->vertex_count = 3;
}

static void deferred_shade(const struct scene *s, uint32_t x, uint32_t y, float out[3])
{
    const struct deferred_frame *f = (const struct deferred_frame *)s->cpu[3];
    size_t p = (size_t)y * s->width + x;
    uint32_t albedo_texel = s->cpu[0][p];
    const float *normal = (const float *)s->cpu[1] + 4 * p, *m = f->inv_view_proj;
    float depth = ((const float *)s->cpu[2])[p];
    float clip[4] = { ((float)x + 0.5f) / (float)s->width * 2.0f - 1.0f, ((float)y + 0.5f) / (float)s->height * 2.0f - 1.0f,
                      depth, 1.0f };
    float albedo[4], world[4], pos[3], n[3], v[3];
    int i, k;

    if (depth >= 1.0f) {
        memcpy(out, f->sky, sizeof(float) * 3);
        return;
    }
    bench_unorm4(albedo_texel, albedo);
    for (k = 0; k < 4; k++)
        world[k] = m[k] * clip[0] + m[4 + k] * clip[1] + m[8 + k] * clip[2] + m[12 + k] * clip[3];
    for (k = 0; k < 3; k++) {
        pos[k] = world[k] / world[3];
        n[k] = normal[k];
        v[k] = f->camera[k] - pos[k];
        out[k] = albedo[k] * f->ambient[k];
    }
    normalize3(n);
    normalize3(v);
    for (i = 0; i < f->light_count; i++) {
        float l[3], hv[3], dist2, r2, atten, ndl, ndh, s_inv;
        for (k = 0; k < 3; k++)
            l[k] = f->light_pos[i][k] - pos[k];
        dist2 = dot3(l, l);
        r2 = f->light_pos[i][3] * f->light_pos[i][3];
        if (dist2 > r2)
            continue;
        s_inv = 1.0f / sqrtf(dist2);
        for (k = 0; k < 3; k++) {
            l[k] *= s_inv;
            hv[k] = l[k] + v[k];
        }
        normalize3(hv);
        atten = 1.0f - dist2 / r2;
        atten *= atten;
        ndl = fmaxf(dot3(n, l), 0.0f);
        ndh = fmaxf(dot3(n, hv), 0.0f);
        for (k = 0; k < 3; k++)
            out[k] += (albedo[k] * ndl + albedo[3] * powf(ndh, 32.0f)) * f->light_color[i][k] * atten;
    }
}

/* ---- forward_pbr_*: sg_pbr.vert, sg_pbr.frag ---- */

#define PBR_TEX 512
#define PBR_LUT 64
#define PBR_CELL 30 /* pixels per mesh cell */

/* sg_pbr.*'s uniform blocks and light (std140, std430). */
struct pbr_frame {
    float shadow_view_proj[4][16];
    float cascade_splits[4];
    float world_scale[4];
    float camera_pos[3];
    uint32_t light_count;
    float sun_dir[3];
    float exposure;
    float sun_color[3];
    float pad;
};

struct pbr_material {
    float base_color[4];
    float metallic, roughness, normal_scale, alpha_cutoff;
};

struct pbr_light {
    float position[3], radius;
    float color[3], intensity;
    float direction[3], spot_cos;
};

struct pbr_vertex {
    float position[3], normal[3], tangent[4], uv[2];
};

static const float pbr_irradiance[3] = { 0.2f, 0.25f, 0.3f };
static const float pbr_prefiltered[3] = { 0.5f, 0.45f, 0.4f };

/* The mesh's attributes are affine in NDC, so interpolation over any
 * triangulation gives the CPU's values at each pixel. */
static void pbr_attributes(float nx, float ny, struct pbr_vertex *v)
{
    v->position[0] = nx;
    v->position[1] = ny;
    v->position[2] = 0.3f + 0.1f * nx + 0.05f * ny;
    v->normal[0] = 0.15f * nx;
    v->normal[1] = 0.1f * ny;
    v->normal[2] = -1.0f;
    v->tangent[0] = 1.0f;
    v->tangent[1] = v->tangent[2] = 0.0f;
    v->tangent[3] = 1.0f;
    v->uv[0] = (nx * 0.5f + 0.5f) * 3.0f;
    v->uv[1] = (ny * 0.5f + 0.5f) * 1.6875f;
}

static void pbr_setup(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDescriptorSetLayoutBinding set0[8] = {
        { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL }, /* immutable: portability */
        { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 6, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
    };
    VkDescriptorSetLayoutBinding set1[5] = {
        { 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
    };
    const VkDescriptorSetLayoutBinding *sets[2] = { set0, set1 };
    uint32_t counts[2] = { 8, 5 };
    struct pbr_frame frame;
    struct pbr_material material = { { 1.0f, 1.0f, 1.0f, 1.0f }, 1.0f, 1.0f, 1.0f, 0.5f };
    struct pbr_light *lights = bench_alloc(sizeof(struct pbr_light) * LIGHTS);
    uint32_t *data = bench_alloc(sizeof(uint32_t) * PBR_TEX * PBR_TEX * 6);
    float *depth = bench_alloc(sizeof(float) * 512 * 512 * 4);
    uint32_t cols = s->width / PBR_CELL, rows = s->height / PBR_CELL, cx, cy, k;
    struct pbr_vertex *vertices = bench_alloc(sizeof(struct pbr_vertex) * 6 * cols * rows);
    uint64_t seed = 0xBB0F0A5Eull;
    struct bench_gfx_desc d;
    int i, x, y;

    /* Material maps: smooth, so bilinear filtering is all the CPU needs. */
    for (i = 0; i < 3; i++)
        s->cpu[i] = bench_alloc(sizeof(uint32_t) * PBR_TEX * PBR_TEX);
    for (y = 0; y < PBR_TEX; y++) {
        for (x = 0; x < PBR_TEX; x++) {
            float u = ((float)x + 0.5f) / PBR_TEX, v = ((float)y + 0.5f) / PBR_TEX;
            float n[3] = { 0.3f * sinf(2.0f * PI_F * (2.0f * u + v)), 0.3f * cosf(2.0f * PI_F * (u - 2.0f * v)), 1.0f };
            normalize3(n);
            s->cpu[0][y * PBR_TEX + x] = bench_rgba8(unorm8(0.55f + 0.3f * sinf(2.0f * PI_F * u)),
                                                     unorm8(0.5f + 0.3f * sinf(2.0f * PI_F * (v + 0.25f))),
                                                     unorm8(0.5f + 0.25f * cosf(2.0f * PI_F * (u + 2.0f * v))), 255);
            s->cpu[1][y * PBR_TEX + x] = bench_rgba8(unorm8(n[0] * 0.5f + 0.5f), unorm8(n[1] * 0.5f + 0.5f),
                                                     unorm8(n[2] * 0.5f + 0.5f), 255);
            s->cpu[2][y * PBR_TEX + x] = bench_rgba8(255, unorm8(0.675f + 0.225f * sinf(2.0f * PI_F * (u + v))),
                                                     unorm8(0.5f + 0.5f * sinf(2.0f * PI_F * (u - v))), 255);
        }
    }
    s->cpu[3] = bench_alloc(sizeof(uint32_t) * PBR_LUT * PBR_LUT);
    for (y = 0; y < PBR_LUT; y++)
        for (x = 0; x < PBR_LUT; x++) {
            float u = ((float)x + 0.5f) / PBR_LUT, v = ((float)y + 0.5f) / PBR_LUT;
            s->cpu[3][y * PBR_LUT + x] = bench_rgba8(unorm8(0.95f - 0.5f * v * (1.0f - 0.4f * u)),
                                                     unorm8(0.02f + 0.1f * (1.0f - u) * v), 0, 255);
        }
    for (i = 0; i < 3; i++)
        texture_create(vk, &s->textures[i], VK_FORMAT_R8G8B8A8_UNORM, PBR_TEX, PBR_TEX, 1, VK_IMAGE_VIEW_TYPE_2D,
                       s->cpu[i], sizeof(uint32_t) * PBR_TEX * PBR_TEX);
    texture_create(vk, &s->textures[3], VK_FORMAT_R8G8B8A8_UNORM, PBR_LUT, PBR_LUT, 1, VK_IMAGE_VIEW_TYPE_2D, s->cpu[3],
                   sizeof(uint32_t) * PBR_LUT * PBR_LUT);
    /* Environment cubes and the shadow map: uniform, so their sampling
     * costs the GPU the same and gives the CPU exact values. Everything is
     * lit (stored depth 1.0). */
    for (k = 0; k < 16 * 16 * 6; k++)
        data[k] = bench_rgba8(unorm8(pbr_irradiance[0]), unorm8(pbr_irradiance[1]), unorm8(pbr_irradiance[2]), 255);
    texture_create(vk, &s->textures[4], VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 6, VK_IMAGE_VIEW_TYPE_CUBE, data,
                   sizeof(uint32_t) * 16 * 16 * 6);
    for (k = 0; k < 64 * 64 * 6; k++)
        data[k] = bench_rgba8(unorm8(pbr_prefiltered[0]), unorm8(pbr_prefiltered[1]), unorm8(pbr_prefiltered[2]), 255);
    texture_create(vk, &s->textures[5], VK_FORMAT_R8G8B8A8_UNORM, 64, 64, 6, VK_IMAGE_VIEW_TYPE_CUBE, data,
                   sizeof(uint32_t) * 64 * 64 * 6);
    for (k = 0; k < 512 * 512 * 4; k++)
        depth[k] = 1.0f;
    texture_create(vk, &s->textures[6], VK_FORMAT_D32_SFLOAT, 512, 512, 4, VK_IMAGE_VIEW_TYPE_2D_ARRAY, depth,
                   sizeof(float) * 512 * 512 * 4);
    free(data);
    free(depth);

    memset(&frame, 0, sizeof(frame));
    for (i = 0; i < 4; i++)
        for (k = 0; k < 4; k++)
            frame.shadow_view_proj[i][k * 5] = 1.0f;
    frame.cascade_splits[0] = 1.0f;
    frame.cascade_splits[1] = 2.0f;
    frame.cascade_splits[2] = 3.0f;
    frame.world_scale[0] = 4.0f;
    frame.world_scale[1] = 2.25f;
    frame.world_scale[2] = 2.0f;
    frame.world_scale[3] = 1.5f;
    frame.camera_pos[2] = -4.0f;
    frame.light_count = LIGHTS;
    frame.sun_dir[0] = 0.3f;
    frame.sun_dir[1] = -0.5f;
    frame.sun_dir[2] = 0.8f;
    normalize3(frame.sun_dir);
    frame.exposure = 1.0f;
    frame.sun_color[0] = 2.0f;
    frame.sun_color[1] = 1.9f;
    frame.sun_color[2] = 1.7f;
    for (i = 0; i < LIGHTS; i++) {
        struct pbr_light *l = &lights[i];
        l->position[0] = (float)bench_rand_range(&seed, -450, 451) / 100.0f;
        l->position[1] = (float)bench_rand_range(&seed, -250, 251) / 100.0f;
        l->position[2] = -(float)bench_rand_range(&seed, 30, 151) / 100.0f;
        l->radius = (float)bench_rand_range(&seed, 120, 281) / 100.0f;
        for (k = 0; k < 3; k++)
            l->color[k] = (float)bench_rand_range(&seed, 30, 101) / 100.0f;
        l->intensity = (float)bench_rand_range(&seed, 100, 401) / 100.0f;
        l->direction[0] = (float)bench_rand_range(&seed, -30, 31) / 100.0f;
        l->direction[1] = (float)bench_rand_range(&seed, -30, 31) / 100.0f;
        l->direction[2] = 1.0f;
        normalize3(l->direction);
        l->spot_cos = (i & 1) ? -1.0f : (float)bench_rand_range(&seed, 60, 86) / 100.0f; /* omni, or a spot */
    }
    buffer_with(vk, &s->buffers[0], VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &frame, sizeof(frame));
    buffer_with(vk, &s->buffers[1], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, lights, sizeof(struct pbr_light) * LIGHTS);
    buffer_with(vk, &s->buffers[2], VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &material, sizeof(material));
    s->cpu[3] = realloc(s->cpu[3], sizeof(uint32_t) * PBR_LUT * PBR_LUT + sizeof(frame) + sizeof(material) +
                                       sizeof(struct pbr_light) * LIGHTS);
    if (!s->cpu[3])
        bench_fatal("out of memory");
    memcpy((char *)s->cpu[3] + sizeof(uint32_t) * PBR_LUT * PBR_LUT, &frame, sizeof(frame));
    memcpy((char *)s->cpu[3] + sizeof(uint32_t) * PBR_LUT * PBR_LUT + sizeof(frame), &material, sizeof(material));
    memcpy((char *)s->cpu[3] + sizeof(uint32_t) * PBR_LUT * PBR_LUT + sizeof(frame) + sizeof(material), lights,
           sizeof(struct pbr_light) * LIGHTS);
    free(lights);

    /* The mesh: two triangles per PBR_CELL-pixel cell, wound alike. */
    for (cy = 0; cy < rows; cy++)
        for (cx = 0; cx < cols; cx++) {
            static const int corner_x[6] = { 0, 1, 0, 1, 1, 0 }, corner_y[6] = { 0, 0, 1, 0, 1, 1 };
            for (k = 0; k < 6; k++)
                pbr_attributes((float)(cx + (uint32_t)corner_x[k]) / (float)cols * 2.0f - 1.0f,
                               (float)(cy + (uint32_t)corner_y[k]) / (float)rows * 2.0f - 1.0f,
                               &vertices[(cy * cols + cx) * 6 + k]);
        }
    s->vertex_count = 6 * cols * rows;
    buffer_with(vk, &s->buffers[3], VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices,
                sizeof(struct pbr_vertex) * s->vertex_count);
    s->vertex_buffer = s->buffers[3].buffer;
    free(vertices);

    s->samplers[0] = make_sampler(vk, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 1);
    s->samplers[1] = make_sampler(vk, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0);
    s->samplers[2] = make_sampler(vk, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, 0);
    set0[2].pImmutableSamplers = &s->samplers[0];
    scene_layouts(s, 2, sets, counts);
    write_buffer(vk, s->sets[0], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &s->buffers[0]);
    write_image(vk, s->sets[0], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s->textures[6].view, VK_NULL_HANDLE);
    write_image(vk, s->sets[0], 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s->textures[4].view, VK_NULL_HANDLE);
    write_image(vk, s->sets[0], 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s->textures[5].view, VK_NULL_HANDLE);
    write_image(vk, s->sets[0], 5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s->textures[3].view, VK_NULL_HANDLE);
    write_image(vk, s->sets[0], 6, VK_DESCRIPTOR_TYPE_SAMPLER, VK_NULL_HANDLE, s->samplers[1]);
    write_buffer(vk, s->sets[0], 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &s->buffers[1]);
    for (i = 0; i < 3; i++)
        write_image(vk, s->sets[1], (uint32_t)i, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s->textures[i].view, VK_NULL_HANDLE);
    write_image(vk, s->sets[1], 3, VK_DESCRIPTOR_TYPE_SAMPLER, VK_NULL_HANDLE, s->samplers[2]);
    write_buffer(vk, s->sets[1], 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &s->buffers[2]);

    memset(&d, 0, sizeof(d));
    d.vs = s->vs = BENCH_SHADER(vk, sg_pbr_vert);
    d.fs = s->fs = BENCH_SHADER(vk, sg_pbr_frag);
    d.binding_count = 1;
    d.bindings[0] = (VkVertexInputBindingDescription){ 0, sizeof(struct pbr_vertex), VK_VERTEX_INPUT_RATE_VERTEX };
    d.attribute_count = 4;
    d.attributes[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32B32_SFLOAT,
                                                           offsetof(struct pbr_vertex, position) };
    d.attributes[1] = (VkVertexInputAttributeDescription){ 1, 0, VK_FORMAT_R32G32B32_SFLOAT,
                                                           offsetof(struct pbr_vertex, normal) };
    d.attributes[2] = (VkVertexInputAttributeDescription){ 2, 0, VK_FORMAT_R32G32B32A32_SFLOAT,
                                                           offsetof(struct pbr_vertex, tangent) };
    d.attributes[3] = (VkVertexInputAttributeDescription){ 3, 0, VK_FORMAT_R32G32_SFLOAT,
                                                           offsetof(struct pbr_vertex, uv) };
    scene_pipeline(s, &d);
}

static void pbr_light(const float n[3], const float v[3], const float l[3], const float radiance[3],
                      const float albedo[3], float metallic, float roughness, float out[3])
{
    float h[3] = { v[0] + l[0], v[1] + l[1], v[2] + l[2] };
    float ndl, ndv, ndh, vdh, a, a2, dd, dggx, kk, vis, fresnel_pow;
    int k;

    normalize3(h);
    ndl = clampf(dot3(n, l), 0.0f, 1.0f);
    ndv = clampf(dot3(n, v), 0.0f, 1.0f) + 1e-4f;
    ndh = clampf(dot3(n, h), 0.0f, 1.0f);
    vdh = clampf(dot3(v, h), 0.0f, 1.0f);
    a = roughness * roughness;
    a2 = a * a;
    dd = ndh * ndh * (a2 - 1.0f) + 1.0f;
    dggx = a2 / (PI_F * dd * dd);
    kk = a * 0.5f;
    vis = (ndv / (ndv * (1.0f - kk) + kk)) * (ndl / (ndl * (1.0f - kk) + kk));
    fresnel_pow = powf(1.0f - vdh, 5.0f);
    for (k = 0; k < 3; k++) {
        float f0 = 0.04f * (1.0f - metallic) + albedo[k] * metallic;
        float f = f0 + (1.0f - f0) * fresnel_pow;
        float spec = dggx * vis * f / 4.0f;
        float kd = (1.0f - f) * (1.0f - metallic);
        out[k] += (kd * albedo[k] / PI_F + spec) * radiance[k] * ndl;
    }
}

static void pbr_shade(const struct scene *s, float u, float v, float out[3])
{
    const char *extra = (const char *)s->cpu[3] + sizeof(uint32_t) * PBR_LUT * PBR_LUT;
    const struct pbr_frame *frame = (const struct pbr_frame *)extra;
    const struct pbr_material *material = (const struct pbr_material *)(extra + sizeof(*frame));
    const struct pbr_light *lights = (const struct pbr_light *)(extra + sizeof(*frame) + sizeof(*material));
    struct cpu_texture albedo_tex = { s->cpu[0], PBR_TEX, PBR_TEX, 1 }, normal_tex = { s->cpu[1], PBR_TEX, PBR_TEX, 1 };
    struct cpu_texture orm_tex = { s->cpu[2], PBR_TEX, PBR_TEX, 1 }, lut = { s->cpu[3], PBR_LUT, PBR_LUT, 0 };
    struct pbr_vertex a;
    float base[4], orm[4], tn[4], brdf[4], world[3], n0[3], t[3], b[3], n[3], view[3], sun[3], color[3] = { 0, 0, 0 };
    float roughness, metallic, ndv, td;
    uint32_t i;
    int k;

    pbr_attributes(u * 2.0f - 1.0f, v * 2.0f - 1.0f, &a);
    world[0] = a.position[0] * frame->world_scale[0];
    world[1] = a.position[1] * frame->world_scale[1];
    world[2] = a.position[2];
    sample_bilinear(&albedo_tex, a.uv[0], a.uv[1], base);
    for (k = 0; k < 4; k++)
        base[k] *= material->base_color[k];
    sample_bilinear(&orm_tex, a.uv[0], a.uv[1], orm);
    roughness = orm[1] * material->roughness;
    metallic = orm[2] * material->metallic;
    sample_bilinear(&normal_tex, a.uv[0], a.uv[1], tn);
    for (k = 0; k < 3; k++)
        tn[k] = tn[k] * 2.0f - 1.0f;
    tn[0] *= material->normal_scale;
    tn[1] *= material->normal_scale;
    memcpy(n0, a.normal, sizeof(n0));
    normalize3(n0);
    td = dot3(n0, a.tangent);
    for (k = 0; k < 3; k++)
        t[k] = a.tangent[k] - n0[k] * td;
    normalize3(t);
    b[0] = (n0[1] * t[2] - n0[2] * t[1]) * a.tangent[3];
    b[1] = (n0[2] * t[0] - n0[0] * t[2]) * a.tangent[3];
    b[2] = (n0[0] * t[1] - n0[1] * t[0]) * a.tangent[3];
    for (k = 0; k < 3; k++) {
        n[k] = t[k] * tn[0] + b[k] * tn[1] + n0[k] * tn[2];
        view[k] = frame->camera_pos[k] - world[k];
        sun[k] = -frame->sun_dir[k];
    }
    normalize3(n);
    normalize3(view);
    pbr_light(n, view, sun, frame->sun_color, base, metallic, roughness, color); /* shadow factor 1: all lit */
    for (i = 0; i < frame->light_count; i++) {
        const struct pbr_light *light = &lights[i];
        float lv[3], l[3], radiance[3], d2, r2, atten, spot, s_inv;
        for (k = 0; k < 3; k++)
            lv[k] = light->position[k] - world[k];
        d2 = dot3(lv, lv);
        r2 = light->radius * light->radius;
        if (d2 > r2)
            continue;
        s_inv = 1.0f / sqrtf(d2);
        for (k = 0; k < 3; k++)
            l[k] = lv[k] * s_inv;
        atten = clampf(1.0f - d2 / r2, 0.0f, 1.0f);
        atten *= atten;
        spot = clampf((-dot3(l, light->direction) - light->spot_cos) * 8.0f, 0.0f, 1.0f);
        for (k = 0; k < 3; k++)
            radiance[k] = light->color[k] * light->intensity * atten * spot;
        pbr_light(n, view, l, radiance, base, metallic, roughness, color);
    }
    ndv = clampf(dot3(n, view), 0.0f, 1.0f);
    sample_bilinear(&lut, ndv, roughness, brdf);
    for (k = 0; k < 3; k++) {
        float f0 = 0.04f * (1.0f - metallic) + base[k] * metallic;
        float spec = (float)unorm8(pbr_prefiltered[k]) / 255.0f * (f0 * brdf[0] + brdf[1]);
        float diffuse = (float)unorm8(pbr_irradiance[k]) / 255.0f * base[k] * (1.0f - metallic);
        out[k] = (color[k] + diffuse + spec) * frame->exposure;
    }
}

/* ==========================================================================
 * Frames, reference, verification
 * ========================================================================== */

static void record(void *user, const struct bench_frame *f)
{
    struct scene *s = user;
    static const float clear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    VkDeviceSize zero = 0;

    bench_cmd_begin_color(f->cmd, &s->target, clear);
    vkd.CmdBindPipeline(f->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->pipeline);
    vkd.CmdBindDescriptorSets(f->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->layout, 0, s->set_count, s->sets, 0, NULL);
    if (s->vertex_buffer)
        vkd.CmdBindVertexBuffers(f->cmd, 0, 1, &s->vertex_buffer, &zero);
    vkd.CmdDraw(f->cmd, s->vertex_count, 1, 0, 0);
    vkd.CmdEndRendering(f->cmd);
    if (f->verify)
        bench_cmd_copy_image_to_buffer(f->cmd, &s->target, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, s->readback.buffer);
}

static void make_reference(struct scene *s)
{
    uint32_t x, y, cols = (s->width + (uint32_t)s->stride - 1) / (uint32_t)s->stride;
    s->reference = bench_alloc(sizeof(float) * 3 * cols * ((s->height + (uint32_t)s->stride - 1) / (uint32_t)s->stride));
    for (y = 0; y < s->height; y += (uint32_t)s->stride)
        for (x = 0; x < s->width; x += (uint32_t)s->stride) {
            float *out = &s->reference[3 * ((y / (uint32_t)s->stride) * cols + x / (uint32_t)s->stride)];
            float u = ((float)x + 0.5f) / (float)s->width, v = ((float)y + 0.5f) / (float)s->height;
            if (s->cfg->kind == KIND_POM)
                pom_shade(s, u, v, out);
            else if (s->cfg->kind == KIND_DEFERRED)
                deferred_shade(s, x, y, out);
            else
                pbr_shade(s, u, v, out);
        }
}

static int verify(struct scene *s)
{
    const uint16_t *got = s->readback.map;
    uint32_t x, y, cols = (s->width + (uint32_t)s->stride - 1) / (uint32_t)s->stride;
    size_t bad = 0, n = 0, first = 0;
    const float *want = NULL;

    for (y = 0; y < s->height; y += (uint32_t)s->stride)
        for (x = 0; x < s->width; x += (uint32_t)s->stride) {
            size_t p = (size_t)y * s->width + x;
            const float *w = &s->reference[3 * ((y / (uint32_t)s->stride) * cols + x / (uint32_t)s->stride)];
            int ok = bench_half_to_float(got[4 * p + 3]) == 1.0f, c;
            for (c = 0; c < 3 && ok; c++) {
                float g = bench_half_to_float(got[4 * p + (size_t)c]);
                ok = fabsf(g - w[c]) <= 4e-3f + 5e-3f * fabsf(w[c]);
            }
            n++;
            if (!ok && bad++ == 0) {
                first = p;
                want = w;
            }
        }
    if (!bad)
        return 1;
    bench_error(s->cfg->name, "%zu of %zu pixels wrong; first at (%zu,%zu): got (%g,%g,%g,%g), want (%g,%g,%g,1)",
                bad, n, first % s->width, first / s->width, bench_half_to_float(got[4 * first]),
                bench_half_to_float(got[4 * first + 1]), bench_half_to_float(got[4 * first + 2]), bench_half_to_float(got[4 * first + 3]),
                want[0], want[1], want[2]);
    return 0;
}

static void scene_setup(struct scene *s, struct bench_vk *vk, const struct config *cfg)
{
    memset(s, 0, sizeof(*s));
    s->vk = vk;
    s->cfg = cfg;
    s->width = cfg->width;
    s->height = cfg->height;
    s->stride = cfg->width > 1920 ? 2 : 1;
    bench_image_create(vk, &s->target, OUTPUT_FORMAT, s->width, s->height, 1, VK_SAMPLE_COUNT_1_BIT,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0);
    bench_buffer_create(vk, &s->readback, (VkDeviceSize)s->width * s->height * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        BENCH_HOST, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (cfg->kind == KIND_POM)
        pom_setup(s);
    else if (cfg->kind == KIND_DEFERRED)
        deferred_setup(s);
    else
        pbr_setup(s);
}

static void scene_teardown(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDevice dev = vk->device;
    int i;

    vkd.DeviceWaitIdle(dev);
    vkd.DestroyPipeline(dev, s->pipeline, NULL);
    vkd.DestroyShaderModule(dev, s->vs, NULL);
    vkd.DestroyShaderModule(dev, s->fs, NULL);
    vkd.DestroyDescriptorPool(dev, s->pool, NULL);
    vkd.DestroyPipelineLayout(dev, s->layout, NULL);
    for (i = 0; i < (int)s->set_count; i++)
        vkd.DestroyDescriptorSetLayout(dev, s->set_layouts[i], NULL);
    for (i = 0; i < MAX_SAMPLERS; i++)
        if (s->samplers[i])
            vkd.DestroySampler(dev, s->samplers[i], NULL);
    for (i = 0; i < MAX_TEXTURES; i++)
        texture_destroy(vk, &s->textures[i]);
    for (i = 0; i < MAX_BUFFERS; i++)
        if (s->buffers[i].buffer)
            bench_buffer_destroy(vk, &s->buffers[i]);
    for (i = 0; i < 4; i++)
        free(s->cpu[i]);
    bench_image_destroy(vk, &s->target);
    bench_buffer_destroy(vk, &s->readback);
    free(s->reference);
    memset(s, 0, sizeof(*s));
}

int main(void)
{
    static const char *kinds[] = { "pom", "deferred", "forward_pbr" };
    struct bench_vk vk;
    struct bench_loop loop;
    struct scene *s;
    size_t c;

    bench_begin("shaders_gpu");
    bench_vk_init(&vk, BENCH_NEED_DYNAMIC_RENDERING | BENCH_NEED_SYNC2);
    bench_loop_init(&loop, &vk, 1000);
    s = bench_alloc(sizeof(*s));

    for (c = 0; c < sizeof(configs) / sizeof(configs[0]); c++) {
        const struct config *cfg = &configs[c];
        int frames = bench_frames(cfg->frames);
        struct bench_json j;

        if (!bench_selected(cfg->name))
            continue;
        bench_config = cfg->name;
        scene_setup(s, &vk, cfg);
        make_reference(s);
        bench_loop_verify(&loop, record, s);
        if (verify(s)) {
            bench_loop_run(&loop, record, s, WARMUP, frames < 1000 ? frames : 1000);
            bench_json_begin(&j, cfg->name);
            bench_json_str(&j, "shader", kinds[cfg->kind]);
            bench_json_int(&j, "width", cfg->width);
            bench_json_int(&j, "height", cfg->height);
            if (cfg->kind != KIND_POM)
                bench_json_int(&j, "lights", LIGHTS);
            if (cfg->kind == KIND_PBR)
                bench_json_int(&j, "triangles", s->vertex_count / 3);
            bench_loop_report(&loop, &j);
            bench_json_end(&j);
        }
        scene_teardown(s);
    }
    bench_config = NULL;

    free(s);
    bench_loop_destroy(&loop);
    bench_vk_destroy(&vk);
    return bench_exit_code();
}
