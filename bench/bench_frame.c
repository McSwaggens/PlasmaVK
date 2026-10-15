/* Frame: one engine-like deferred frame, the mix of work a game submits,
 * instead of one pattern at a time. Per frame, in order:
 *
 *   streaming    object constants (3000 x 64 B into a dynamic uniform
 *                buffer), UI vertices and indices (330 KB) and 4 texture
 *                tiles (staging ring + vkCmdCopyBufferToImage), all into the
 *                frame's slot
 *   descriptors  the slot's pool reset, 14 transient sets allocated and
 *                written (culling, lighting, bloom, tonemap)
 *   shadows      4 cascades, each a rendering instance into a layer of a
 *                2048^2 D32 array: a depth-only pipeline (no fragment shader,
 *                depth bias), 1000 draws with a dynamic offset each
 *   G-buffer     albedo RGBA8, normal and emissive RGBA16F, D32: 3000 draws
 *                sorted by pipeline (4 specializations of one shader: plain,
 *                dithered discard, two-sided, emissive) and material (64 of
 *                4 textures, changing every 10 draws: set bind and fragment
 *                push constants), a dynamic offset per draw
 *   culling      tiled light culling (compute, 16x16 tiles, shared atomics)
 *                of 256 lights
 *   lighting     fullscreen: G-buffer, shadow cascades (2x2 PCF through an
 *                immutable comparison sampler), the tile's lights -> RGBA16F
 *   bloom        6 downsampling and 5 additive upsampling passes
 *   final        tonemap, then 600 UI draws (a scissor each, streamed indexed
 *                geometry, a glyph atlas) into the slot's RGBA8 image
 *
 * That is 7613 draws, 18 rendering instances and a dispatch. Configurations
 * (1080p unless named):
 *
 *   st                     one thread records everything
 *   mt4_secondary          4 worker threads record the cascades and 4 chunks
 *                          of the G-buffer as secondaries that the primary
 *                          executes; the main thread streams meanwhile
 *   mt4_primary            4 worker threads record 4 primaries (uploads and
 *                          cascades 0-1, cascades 2-3, G-buffer, the rest),
 *                          submitted together, as D3D12-style renderers do
 *   st_profiled            2 timestamps around every pass, read a frame
 *                          later like an engine's GPU profiler (gpu_*_us)
 *   st_bgcompile           a background thread meanwhile creates G-buffer
 *                          pipelines with fresh specialization constants
 *                          (bg_compiles finished while measuring, create_us)
 *   st_2160p               GPU-bound: the frame at 3840x2160
 *   st_dynamic_depth_bounds  a device with depthBounds enabled whose depth
 *                          pipelines make the bounds test dynamic (and off),
 *                          as DXVK, vkd3d-proton, Zink and EDS engines do
 *
 * cpu_us is the main thread's critical path: streaming, descriptors, its
 * recording and waiting for the workers, and the submit. Multithreaded
 * configurations add worker_cpu_us (CPU time of all workers in a frame) and
 * record_wall_us (from waking them to the last one finishing).
 *
 * The first frame of each configuration is read back and compared with a
 * CPU rendering: the G-buffer albedo and the tiles' light lists exactly, the
 * final image within 2/255 per channel (float lighting, bloom and tonemap;
 * UI glyphs exactly). The scene is built so that no comparison is near a
 * decision boundary: shadow texels and depths keep margins, lights are
 * placed so that no tile's culling test is within 1e-5 of the radius. At
 * 1080p the multithreaded and background-compile frames must also equal the
 * single-threaded one bit for bit.
 */
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#include "bench_vk.h"
#include "bench_shaders.h"

#define OBJECTS 3000
#define CASCADES 4
#define CASCADE_DRAWS 1000
#define SHADOW_SIZE 2048
#define MATERIALS 64
#define TEXTURES 16
#define TEX_SIZE 64
#define LIGHTS 256
#define TILE_LIGHTS 63 /* per tile, plus the count: 64 words */
#define BLOOM_LEVELS 6
#define WIDGETS 600
#define WIDGET_QUADS 6 /* glyphs per label */
#define GLYPH_W 10
#define GLYPH_H 16
#define GLYPH_ADVANCE 11
#define LABEL_W 60 /* the scissor: the last glyph is cut */
#define FONT_SIZE 256
#define STREAMED_TILES 4
#define THREADS 4
#define GBUFFERS 3
#define TRANSIENT_SETS (3 + 2 * BLOOM_LEVELS - 1) /* culling, lighting, tonemap, bloom */
#define REGIONS (CASCADES + 3 + 2 * BLOOM_LEVELS - 1 + 1) /* timed passes and the dispatch */
#define DRAWS (CASCADES * CASCADE_DRAWS + OBJECTS + WIDGETS + 2 + 2 * BLOOM_LEVELS - 1)
#define PASSES (CASCADES + 3 + 2 * BLOOM_LEVELS - 1)
#define MAX_FRAMES 1000
#define WARMUP_US 200000.0
#define EXPOSURE 1.0f
#define BLOOM_STRENGTH 0.2f

static const VkFormat gbuffer_formats[GBUFFERS] = {
    VK_FORMAT_R8G8B8A8_UNORM,      /* albedo */
    VK_FORMAT_R16G16B16A16_SFLOAT, /* normal, roughness */
    VK_FORMAT_R16G16B16A16_SFLOAT, /* emissive */
};
#define DEPTH_FORMAT VK_FORMAT_D32_SFLOAT
#define HDR_FORMAT VK_FORMAT_R16G16B16A16_SFLOAT
#define FINAL_FORMAT VK_FORMAT_R8G8B8A8_UNORM

enum mode { MODE_ST, MODE_SECONDARY, MODE_PRIMARY };

struct config {
    const char *name;
    enum mode mode;
    uint32_t width, height;
    int profiled, bgcompile, depth_bounds;
    int frames;
};

static const struct config configs[] = {
    { "st_1080p", MODE_ST, 1920, 1080, 0, 0, 0, 200 },
    { "mt4_secondary_1080p", MODE_SECONDARY, 1920, 1080, 0, 0, 0, 200 },
    { "mt4_primary_1080p", MODE_PRIMARY, 1920, 1080, 0, 0, 0, 200 },
    { "st_profiled_1080p", MODE_ST, 1920, 1080, 1, 0, 0, 200 },
    { "st_bgcompile_1080p", MODE_ST, 1920, 1080, 0, 1, 0, 200 },
    { "st_2160p", MODE_ST, 3840, 2160, 0, 0, 0, 100 },
    { "st_dynamic_depth_bounds_1080p", MODE_ST, 1920, 1080, 0, 0, 1, 200 },
};
#define CONFIGS (sizeof(configs) / sizeof(configs[0]))

/* The profiled regions, in recording order, by group. */
static const char *const group_names[] = { "gpu_shadow_us", "gpu_gbuffer_us", "gpu_cull_us",
                                           "gpu_light_us", "gpu_bloom_us", "gpu_final_us" };
#define GROUPS 6

static int region_group(int region)
{
    if (region < CASCADES)
        return 0;
    if (region < CASCADES + 3)
        return region - CASCADES + 1; /* G-buffer, culling, lighting */
    return region == REGIONS - 1 ? 5 : 4;
}

/* frame_object.vert's uniform block. */
struct object_ubo {
    float rect[4];         /* x0 y0 x1 y1 in NDC */
    float normal[4];       /* xyz, w roughness */
    float emissive[4];
    float albedo_depth[4]; /* rgb, depth */
};

/* frame_ui.vert's vertices. */
struct ui_vertex {
    float pos[2], uv[2];
    uint32_t rgba;
};

/* The lights' storage buffer (std430). */
struct lights_ssbo {
    float pos_radius[LIGHTS][4];
    float color[LIGHTS][4];
};

/* ==========================================================================
 * The scene and its CPU rendering
 * ========================================================================== */

struct object {
    int x0, y0, x1, y1; /* pixels at 1080p */
    int pipeline, material;
    struct object_ubo ubo;
};

struct material {
    int textures[4];
    float offset[2]; /* texels */
};

struct widget {
    int x, y; /* pixels at 1080p */
    uint32_t rgba;
};

/* The CPU rendering at one resolution. */
struct reference {
    uint32_t width, height, tiles_x, tiles_y;
    int16_t *top;      /* the object seen at each pixel, -1: none */
    uint32_t *albedo;  /* G-buffer albedo */
    uint32_t *tiles;   /* 64 words per tile: count, indices */
    uint32_t *final;   /* final image */
    uint8_t *glyph;    /* 1: a UI glyph pixel (exact) */
    int complete;      /* tiles, final and glyph computed */
};

struct world {
    struct object objects[OBJECTS];
    struct material materials[MATERIALS];
    uint32_t texels[TEXTURES][TEX_SIZE * TEX_SIZE];
    uint8_t font[FONT_SIZE * FONT_SIZE]; /* 0 or 255 */
    struct widget widgets[WIDGETS];
    struct lights_ssbo lights;
    struct reference refs[2]; /* 1080p, 2160p */
};

static float half_to_float(uint16_t h)
{
    int exponent = (h >> 10) & 0x1F, mantissa = h & 0x3FF;
    float v;
    if (exponent == 0)
        v = ldexpf((float)mantissa, -24);
    else if (exponent == 31)
        v = mantissa ? NAN : INFINITY;
    else
        v = ldexpf((float)(mantissa | 0x400), exponent - 25);
    return (h & 0x8000) ? -v : v;
}

/* Round to nearest even, as the GPU stores RGBA16F. */
static uint16_t half_from_float(float f)
{
    uint32_t x, sign, mantissa, h, rest, halfway, shift;
    int exponent;
    memcpy(&x, &f, sizeof(x));
    sign = (x >> 16) & 0x8000u;
    mantissa = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF)
        return (uint16_t)(sign | 0x7C00u | (mantissa ? 0x200u : 0u));
    exponent = (int)((x >> 23) & 0xFF) - 127 + 15;
    if (exponent >= 31)
        return (uint16_t)(sign | 0x7C00u);
    if (exponent <= 0) { /* subnormal */
        if (exponent < -10)
            return (uint16_t)sign;
        mantissa |= 0x800000u;
        shift = (uint32_t)(14 - exponent);
        h = mantissa >> shift;
        rest = mantissa & ((1u << shift) - 1u);
        halfway = 1u << (shift - 1u);
    } else {
        h = ((uint32_t)exponent << 10) | (mantissa >> 13);
        rest = mantissa & 0x1FFFu;
        halfway = 0x1000u;
    }
    if (rest > halfway || (rest == halfway && (h & 1u)))
        h++; /* may carry into the exponent, which is right */
    return (uint16_t)(sign | h);
}

static float rounded_half(float f)
{
    return half_to_float(half_from_float(f));
}

static uint32_t unorm8(float f)
{
    float v = floorf(f * 255.0f + 0.5f);
    return v <= 0.0f ? 0u : v >= 255.0f ? 255u : (uint32_t)v;
}

static float fclamp(float v, float lo, float hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

static int iclamp(int v, int lo, int hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

/* Depths are distinct and an odd number of half steps from 0.25, 0.5 and
 * 0.75 (the cascade split) and from each other minus the shadow offset
 * 0.002 (= 7.5 steps). */
static float object_depth(int i)
{
    int k = (i * 7919) % OBJECTS;
    return 0.1f + 0.8f * ((float)k + 0.25f) / (float)OBJECTS;
}

static float dyadic(uint64_t *seed, int lo, int hi, int denominator)
{
    return (float)bench_rand_range(seed, lo, hi + 1) / (float)denominator;
}

static void make_world(struct world *w)
{
    struct bench_rect *rects = bench_alloc(sizeof(*rects) * OBJECTS);
    uint64_t seed = 0xF4A3E5EEDull;
    int i, k;

    bench_rect_scene(rects, OBJECTS, 1920, 1080, 120, 16, seed);
    for (i = 0; i < OBJECTS; i++) {
        struct object *o = &w->objects[i];
        float c[4];
        o->x0 = rects[i].x0;
        o->y0 = rects[i].y0;
        o->x1 = rects[i].x1;
        o->y1 = rects[i].y1;
        o->pipeline = i * 4 / OBJECTS;
        o->material = (i / 10) % MATERIALS;
        bench_rect_ndc(&rects[i], 1920, 1080, 0, 0, o->ubo.rect);
        bench_unorm4(rects[i].rgba, c);
        o->ubo.normal[0] = dyadic(&seed, -2, 2, 8);
        o->ubo.normal[1] = dyadic(&seed, -2, 2, 8);
        o->ubo.normal[2] = -1.0f;
        o->ubo.normal[3] = dyadic(&seed, 1, 4, 4);
        if (bench_rand(&seed) % 6 == 0) {
            o->ubo.emissive[0] = dyadic(&seed, 0, 16, 16);
            o->ubo.emissive[1] = dyadic(&seed, 0, 16, 16);
            o->ubo.emissive[2] = dyadic(&seed, 0, 16, 16);
        }
        o->ubo.albedo_depth[0] = c[0];
        o->ubo.albedo_depth[1] = c[1];
        o->ubo.albedo_depth[2] = c[2];
        o->ubo.albedo_depth[3] = object_depth(i);
    }
    free(rects);
    for (i = 0; i < MATERIALS; i++) {
        for (k = 0; k < 4; k++)
            w->materials[i].textures[k] = (i * 4 + k * 5) % TEXTURES;
        w->materials[i].offset[0] = (float)((i * 23) % TEX_SIZE);
        w->materials[i].offset[1] = (float)((i * 41) % TEX_SIZE);
    }
    for (i = 0; i < TEXTURES; i++)
        for (k = 0; k < TEX_SIZE * TEX_SIZE; k++)
            w->texels[i][k] = bench_rand(&seed);
    for (k = 0; k < FONT_SIZE * FONT_SIZE; k++)
        w->font[k] = (bench_rand(&seed) & 1) ? 255 : 0;
    for (i = 0; i < WIDGETS; i++) {
        w->widgets[i].x = bench_rand_range(&seed, 0, 1920 - 2 * LABEL_W);
        w->widgets[i].y = bench_rand_range(&seed, 0, 1080 - 2 * GLYPH_H);
        w->widgets[i].rgba = bench_rand(&seed) | 0xFF000000u;
    }
}

/* frame_material.frag at pixel (x, y) of object o: the albedo texel, and
 * the normal and emissive as stored (RGBA16F). */
static uint32_t gbuffer_texel(const struct world *w, const struct object *o, int x, int y, float normal[3],
                              float emissive[3])
{
    const struct material *m = &w->materials[o->material];
    int t = ((y + (int)m->offset[1]) & (TEX_SIZE - 1)) * TEX_SIZE + ((x + (int)m->offset[0]) & (TEX_SIZE - 1));
    float a[4], n[4], r[4], e[4], scale = o->pipeline == 3 ? 2.0f : 1.0f;
    uint32_t albedo = 0;
    int c;
    bench_unorm4(w->texels[m->textures[0]][t], a);
    bench_unorm4(w->texels[m->textures[1]][t], n);
    bench_unorm4(w->texels[m->textures[2]][t], r);
    bench_unorm4(w->texels[m->textures[3]][t], e);
    for (c = 0; c < 4; c++)
        albedo |= unorm8((c < 3 ? o->ubo.albedo_depth[c] : 1.0f) * a[c]) << (8 * c);
    for (c = 0; c < 3; c++) {
        normal[c] = rounded_half(o->ubo.normal[c] + n[c] - 0.5f);
        emissive[c] = rounded_half(o->ubo.emissive[c] * e[c] * scale);
    }
    return albedo;
}

static float pixel_depth(const struct world *w, const struct reference *r, size_t p)
{
    return r->top[p] < 0 ? 1.0f : w->objects[r->top[p]].ubo.albedo_depth[3];
}

/* The G-buffer pass: which object each pixel shows. */
static void reference_gbuffer(const struct world *w, struct reference *r, uint32_t width, uint32_t height)
{
    int scale = (int)(width / 1920);
    size_t n = (size_t)width * height, p;
    float *zbuf = bench_alloc(sizeof(float) * n);
    int i, x, y;

    r->width = width;
    r->height = height;
    r->tiles_x = (width + 15) / 16;
    r->tiles_y = (height + 15) / 16;
    r->top = bench_alloc(sizeof(int16_t) * n);
    for (p = 0; p < n; p++) {
        zbuf[p] = 1.0f;
        r->top[p] = -1;
    }
    for (i = 0; i < OBJECTS; i++) {
        const struct object *o = &w->objects[i];
        for (y = o->y0 * scale; y < o->y1 * scale; y++)
            for (x = o->x0 * scale; x < o->x1 * scale; x++) {
                p = (size_t)y * width + (size_t)x;
                if (o->pipeline == 1 && ((x ^ y) & 4))
                    continue; /* dithered */
                if (o->ubo.albedo_depth[3] < zbuf[p]) {
                    zbuf[p] = o->ubo.albedo_depth[3];
                    r->top[p] = (int16_t)i;
                }
            }
    }
    free(zbuf);
}

/* frame_cull.comp's box for tile (tx, ty): x, y in [0,1], z the depth range. */
static void tile_box(const struct world *w, const struct reference *r, uint32_t tx, uint32_t ty, float lo[3],
                     float hi[3])
{
    float inv_w = 1.0f / (float)r->width, inv_h = 1.0f / (float)r->height;
    uint32_t x, y;
    lo[0] = (float)(tx * 16) * inv_w;
    lo[1] = (float)(ty * 16) * inv_h;
    hi[0] = (float)(tx * 16 + 16) * inv_w;
    hi[1] = (float)(ty * 16 + 16) * inv_h;
    lo[2] = 1.0f;
    hi[2] = 0.0f;
    for (y = ty * 16; y < ty * 16 + 16; y++)
        for (x = tx * 16; x < tx * 16 + 16; x++) {
            uint32_t cx = x < r->width ? x : r->width - 1, cy = y < r->height ? y : r->height - 1;
            float z = pixel_depth(w, r, (size_t)cy * r->width + cx);
            lo[2] = z < lo[2] ? z : lo[2];
            hi[2] = z > hi[2] ? z : hi[2];
        }
}

static float box_distance2(const float lo[3], const float hi[3], const float *l)
{
    float d2 = 0.0f;
    int c;
    for (c = 0; c < 3; c++) {
        float d = fclamp(l[c], lo[c], hi[c]) - l[c];
        d2 += d * d;
    }
    return d2;
}

/* 256 point lights in front of the scene. Each is placed so that, at both
 * resolutions, every tile is clearly inside or outside its radius. */
static void make_lights(struct world *w)
{
    uint64_t seed = 0x11647ull;
    float *boxes[2];
    uint32_t counts[2], g, t;
    int i;

    for (g = 0; g < 2; g++) {
        const struct reference *r = &w->refs[g];
        counts[g] = r->tiles_x * r->tiles_y;
        boxes[g] = bench_alloc(sizeof(float) * 6 * counts[g]);
        for (t = 0; t < counts[g]; t++)
            tile_box(w, r, t % r->tiles_x, t / r->tiles_x, &boxes[g][6 * t], &boxes[g][6 * t + 3]);
    }
    for (i = 0; i < LIGHTS; i++) {
        float *l = w->lights.pos_radius[i];
        int ok = 0;
        while (!ok) {
            l[0] = (float)(bench_rand(&seed) % 4096) / 4096.0f;
            l[1] = (float)(bench_rand(&seed) % 4096) / 4096.0f;
            l[2] = 0.45f * (float)(bench_rand(&seed) % 4096) / 4096.0f;
            l[3] = 0.08f + 0.14f * (float)(bench_rand(&seed) % 4096) / 4096.0f;
            ok = 1;
            for (g = 0; g < 2 && ok; g++)
                for (t = 0; t < counts[g] && ok; t++) {
                    double d = sqrt((double)box_distance2(&boxes[g][6 * t], &boxes[g][6 * t + 3], l));
                    ok = fabs(d - (double)l[3]) > 1e-5;
                }
        }
        w->lights.color[i][0] = dyadic(&seed, 8, 32, 32);
        w->lights.color[i][1] = dyadic(&seed, 8, 32, 32);
        w->lights.color[i][2] = dyadic(&seed, 8, 32, 32);
    }
    free(boxes[0]);
    free(boxes[1]);
}

/* texture() through a linear clamp-to-edge sampler, of an RGB float image. */
static void bilinear(const float *img, int width, int height, float u, float v, float out[3])
{
    float x = u * (float)width - 0.5f, y = v * (float)height - 0.5f;
    float fx = floorf(x), fy = floorf(y), ax = x - fx, ay = y - fy;
    int x0 = iclamp((int)fx, 0, width - 1), x1 = iclamp((int)fx + 1, 0, width - 1);
    int y0 = iclamp((int)fy, 0, height - 1), y1 = iclamp((int)fy + 1, 0, height - 1);
    int c;
    for (c = 0; c < 3; c++) {
        float top = img[3 * ((size_t)y0 * width + x0) + c] * (1.0f - ax) + img[3 * ((size_t)y0 * width + x1) + c] * ax;
        float bottom = img[3 * ((size_t)y1 * width + x0) + c] * (1.0f - ax) + img[3 * ((size_t)y1 * width + x1) + c] * ax;
        out[c] = top * (1.0f - ay) + bottom * ay;
    }
}

/* frame_bloom.frag into a dst_w x dst_h level. */
static void bloom_texel(const float *src, int src_w, int src_h, int dst_w, int dst_h, int x, int y, float out[3])
{
    float inv_w = 1.0f / (float)dst_w, inv_h = 1.0f / (float)dst_h;
    float u = ((float)x + 0.5f) * inv_w, v = ((float)y + 0.5f) * inv_h, du = 0.5f * inv_w, dv = 0.5f * inv_h;
    float t[4][3];
    int c;
    bilinear(src, src_w, src_h, u - du, v - dv, t[0]);
    bilinear(src, src_w, src_h, u + du, v - dv, t[1]);
    bilinear(src, src_w, src_h, u - du, v + dv, t[2]);
    bilinear(src, src_w, src_h, u + du, v + dv, t[3]);
    for (c = 0; c < 3; c++)
        out[c] = 0.25f * (t[0][c] + t[1][c] + t[2][c] + t[3][c]);
}

/* A label's glyph q, in pixels at scale, and its cell in the atlas. */
static void widget_quad(const struct widget *wd, int i, int q, int scale, int rect[4], int cell[2])
{
    int g = (i * 7 + q * 3) % 100;
    rect[0] = (wd->x + q * GLYPH_ADVANCE) * scale;
    rect[1] = wd->y * scale;
    rect[2] = rect[0] + GLYPH_W * scale;
    rect[3] = rect[1] + GLYPH_H * scale;
    cell[0] = (g % 20) * (GLYPH_W + 2);
    cell[1] = (g / 20) * GLYPH_H;
}

/* Tiles, lighting, bloom, tonemap and UI at r's resolution. */
static void reference_complete(const struct world *w, struct reference *r)
{
    uint32_t width = r->width, height = r->height, x, y, t;
    size_t n = (size_t)width * height, p;
    int scale = (int)(width / 1920), lw[BLOOM_LEVELS], lh[BLOOM_LEVELS], l, c, i, q;
    float inv_w = 1.0f / (float)width, inv_h = 1.0f / (float)height;
    float *hdr = bench_alloc(sizeof(float) * 3 * n), *levels[BLOOM_LEVELS], *map;
    uint8_t *lit = bench_alloc(n); /* shadow taps passed, 0-4 */

    r->albedo = bench_alloc(sizeof(uint32_t) * n);
    for (p = 0; p < n; p++) {
        float normal[3], emissive[3];
        r->albedo[p] = r->top[p] < 0 ? 0
                                     : gbuffer_texel(w, &w->objects[r->top[p]], (int)(p % width), (int)(p / width),
                                                     normal, emissive);
    }

    /* Light culling. */
    r->tiles = bench_alloc(sizeof(uint32_t) * 64 * r->tiles_x * r->tiles_y);
    for (t = 0; t < r->tiles_x * r->tiles_y; t++) {
        float lo[3], hi[3];
        uint32_t count = 0;
        tile_box(w, r, t % r->tiles_x, t / r->tiles_x, lo, hi);
        for (i = 0; i < LIGHTS; i++) {
            const float *lp = w->lights.pos_radius[i];
            if (box_distance2(lo, hi, lp) <= lp[3] * lp[3] && count++ < TILE_LIGHTS)
                r->tiles[64 * t + count] = (uint32_t)i;
        }
        r->tiles[64 * t] = count < TILE_LIGHTS ? count : TILE_LIGHTS;
    }

    /* Shadows: one cascade's map at a time, for the pixels that use it. */
    map = bench_alloc(sizeof(float) * SHADOW_SIZE * SHADOW_SIZE);
    for (l = 0; l < CASCADES; l++) {
        double s = ldexp(1.0, -l);
        int k, tx, ty;
        for (p = 0; p < (size_t)SHADOW_SIZE * SHADOW_SIZE; p++)
            map[p] = 1.0f;
        for (k = 0; k < CASCADE_DRAWS; k++) {
            const struct object *o = &w->objects[(l * 750 + k) % OBJECTS];
            /* Texel centres inside the rectangle (no edge is within 1/120
             * texel of one). */
            int x0 = (int)ceil(((double)o->ubo.rect[0] * s + 1.0) * SHADOW_SIZE / 2 - 0.5);
            int x1 = (int)ceil(((double)o->ubo.rect[2] * s + 1.0) * SHADOW_SIZE / 2 - 0.5);
            int y0 = (int)ceil(((double)o->ubo.rect[1] * s + 1.0) * SHADOW_SIZE / 2 - 0.5);
            int y1 = (int)ceil(((double)o->ubo.rect[3] * s + 1.0) * SHADOW_SIZE / 2 - 0.5);
            for (ty = y0; ty < y1; ty++)
                for (tx = x0; tx < x1; tx++)
                    if (o->ubo.albedo_depth[3] < map[ty * SHADOW_SIZE + tx])
                        map[ty * SHADOW_SIZE + tx] = o->ubo.albedo_depth[3];
        }
        for (y = 0; y < height; y++)
            for (x = 0; x < width; x++) {
                float z, su, sv, scale_c = (float)s;
                p = (size_t)y * width + x;
                z = pixel_depth(w, r, p);
                if ((int)(z * 4.0f < 3.0f ? z * 4.0f : 3.0f) != l)
                    continue;
                su = (((float)x + 0.5f) * inv_w - 0.5f) * scale_c + (0.5f + 0.25f / 2048.0f);
                sv = (((float)y + 0.5f) * inv_h - 0.5f) * scale_c + (0.5f + 0.25f / 2048.0f);
                lit[p] = 0;
                for (k = 0; k < 4; k++) {
                    tx = iclamp((int)floorf((su + (float)(k & 1) / 2048.0f) * 2048.0f), 0, SHADOW_SIZE - 1);
                    ty = iclamp((int)floorf((sv + (float)(k >> 1) / 2048.0f) * 2048.0f), 0, SHADOW_SIZE - 1);
                    lit[p] += z - 0.002f <= map[ty * SHADOW_SIZE + tx];
                }
            }
    }
    free(map);

    /* Lighting (frame_light.frag), stored as RGBA16F. */
    for (y = 0; y < height; y++)
        for (x = 0; x < width; x++) {
            const uint32_t *tile;
            float albedo[4], nrm[3] = { 0.0f, 0.0f, -1.0f }, col[3] = { 0.0f, 0.0f, 0.0f }, pos[3], len, sh;
            p = (size_t)y * width + x;
            tile = &r->tiles[64 * ((y / 16) * r->tiles_x + x / 16)];
            bench_unorm4(r->albedo[p], albedo);
            if (r->top[p] >= 0)
                gbuffer_texel(w, &w->objects[r->top[p]], (int)x, (int)y, nrm, col);
            len = sqrtf(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);
            for (c = 0; c < 3; c++)
                nrm[c] /= len;
            pos[0] = ((float)x + 0.5f) * inv_w;
            pos[1] = ((float)y + 0.5f) * inv_h;
            pos[2] = pixel_depth(w, r, p);
            for (t = 0; t < tile[0]; t++) {
                const float *lp = w->lights.pos_radius[tile[1 + t]], *lc = w->lights.color[tile[1 + t]];
                float d[3] = { lp[0] - pos[0], lp[1] - pos[1], lp[2] - pos[2] };
                float dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                float atten = fmaxf(0.0f, 1.0f - dist2 / (lp[3] * lp[3]));
                float ndl = fmaxf(0.0f, (nrm[0] * d[0] + nrm[1] * d[1] + nrm[2] * d[2]) / sqrtf(fmaxf(dist2, 1e-8f)));
                for (c = 0; c < 3; c++)
                    col[c] += albedo[c] * lc[c] * (ndl * atten * atten);
            }
            sh = 0.25f * (float)lit[p];
            for (c = 0; c < 3; c++)
                hdr[3 * p + c] = rounded_half(col[c] * (0.25f + 0.75f * sh));
        }
    free(lit);

    /* Bloom: down, then additive up. */
    for (l = 0; l < BLOOM_LEVELS; l++) {
        lw[l] = (int)width >> (l + 1);
        lh[l] = (int)height >> (l + 1);
        levels[l] = bench_alloc(sizeof(float) * 3 * (size_t)lw[l] * (size_t)lh[l]);
        for (y = 0; y < (uint32_t)lh[l]; y++)
            for (x = 0; x < (uint32_t)lw[l]; x++) {
                float v[3];
                if (l == 0)
                    bloom_texel(hdr, (int)width, (int)height, lw[l], lh[l], (int)x, (int)y, v);
                else
                    bloom_texel(levels[l - 1], lw[l - 1], lh[l - 1], lw[l], lh[l], (int)x, (int)y, v);
                for (c = 0; c < 3; c++)
                    levels[l][3 * ((size_t)y * lw[l] + x) + c] = rounded_half(v[c]);
            }
    }
    for (l = BLOOM_LEVELS - 2; l >= 0; l--)
        for (y = 0; y < (uint32_t)lh[l]; y++)
            for (x = 0; x < (uint32_t)lw[l]; x++) {
                float v[3];
                float *dst = &levels[l][3 * ((size_t)y * lw[l] + x)];
                bloom_texel(levels[l + 1], lw[l + 1], lh[l + 1], lw[l], lh[l], (int)x, (int)y, v);
                for (c = 0; c < 3; c++)
                    dst[c] = rounded_half(dst[c] + v[c]);
            }

    /* Tonemap, then the UI over it. */
    r->final = bench_alloc(sizeof(uint32_t) * n);
    r->glyph = bench_alloc(n);
    for (y = 0; y < height; y++)
        for (x = 0; x < width; x++) {
            float b[3];
            uint32_t rgba = 0xFF000000u;
            p = (size_t)y * width + x;
            bilinear(levels[0], lw[0], lh[0], ((float)x + 0.5f) * inv_w, ((float)y + 0.5f) * inv_h, b);
            for (c = 0; c < 3; c++) {
                float v = 1.0f - expf(-(hdr[3 * p + c] + BLOOM_STRENGTH * b[c]) * EXPOSURE);
                rgba |= unorm8(powf(v, 1.0f / 2.2f)) << (8 * c);
            }
            r->final[p] = rgba;
        }
    for (i = 0; i < WIDGETS; i++) {
        const struct widget *wd = &w->widgets[i];
        int clip = (wd->x + LABEL_W) * scale; /* the scissor's right edge */
        for (q = 0; q < WIDGET_QUADS; q++) {
            int rect[4], cell[2], px, py;
            widget_quad(wd, i, q, scale, rect, cell);
            for (py = rect[1]; py < rect[3]; py++)
                for (px = rect[0]; px < rect[2] && px < clip; px++) {
                    int u = cell[0] + (2 * (px - rect[0]) + 1) / (2 * scale);
                    int v = cell[1] + (2 * (py - rect[1]) + 1) / (2 * scale);
                    p = (size_t)py * width + (size_t)px;
                    if (w->font[v * FONT_SIZE + u]) {
                        r->final[p] = wd->rgba;
                        r->glyph[p] = 1;
                    } else { /* blended with alpha 0: color kept, alpha replaced */
                        r->final[p] &= 0x00FFFFFFu;
                    }
                }
        }
    }
    for (l = 0; l < BLOOM_LEVELS; l++)
        free(levels[l]);
    free(hdr);
    r->complete = 1;
}

static void reference_free(struct reference *r)
{
    free(r->top);
    free(r->albedo);
    free(r->tiles);
    free(r->final);
    free(r->glyph);
    memset(r, 0, sizeof(*r));
}

/* ==========================================================================
 * Device resources
 * ========================================================================== */

#define MAX_BG 4096

struct worker {
    struct scene *s;
    int id;
};

struct scene {
    struct bench_vk *vk;
    struct world *world;
    const struct config *cfg;
    struct reference *ref;
    int dynamic_depth_bounds; /* depth pipelines make the bounds test dynamic */

    /* Sized by the resolution. */
    uint32_t width, height, tiles_x, tiles_y;
    struct bench_image gbuffer[GBUFFERS], depth, hdr, bloom[BLOOM_LEVELS], final[BENCH_FRAMES_IN_FLIGHT];
    struct bench_buffer tiles, readback_final, readback_albedo, readback_tiles;
    struct ui_vertex *ui_vertices; /* this resolution's UI, copied every frame */
    uint16_t *ui_indices;
    VkRect2D ui_scissors[WIDGETS];

    /* Fixed. */
    VkImage shadow;
    VkDeviceMemory shadow_memory;
    VkImageView shadow_layers[CASCADES], shadow_array;
    struct bench_image textures[TEXTURES], font;
    struct bench_buffer objects, lights, ui_vb[BENCH_FRAMES_IN_FLIGHT], ui_ib[BENCH_FRAMES_IN_FLIGHT];
    struct bench_buffer staging[BENCH_FRAMES_IN_FLIGHT];
    VkDeviceSize object_stride;
    VkSampler nearest, repeat, linear, compare;
    VkDescriptorSetLayout material_layout, object_layout, light_layout, cull_layout, tex1_layout, tex2_layout;
    VkDescriptorPool pool, frame_pools[BENCH_FRAMES_IN_FLIGHT];
    VkDescriptorSet materials[MATERIALS], object_set, font_set;
    VkDescriptorSet frame_sets[BENCH_FRAMES_IN_FLIGHT][TRANSIENT_SETS]; /* see write_frame_sets */
    VkPipelineLayout scene_playout, light_playout, cull_playout, bloom_playout, tonemap_playout, ui_playout;
    VkPipeline shadow_pipeline, gbuffer_pipelines[4], cull_pipeline, light_pipeline, down_pipeline, up_pipeline;
    VkPipeline tonemap_pipeline, ui_pipeline;
    struct bench_gfx_state *bg_state; /* a G-buffer pipeline, for background compiles */

    /* Worker threads and their secondaries. */
    VkCommandPool secondary_pools[BENCH_FRAMES_IN_FLIGHT][THREADS];
    VkCommandBuffer shadow_cmds[BENCH_FRAMES_IN_FLIGHT][THREADS], gbuffer_cmds[BENCH_FRAMES_IN_FLIGHT][THREADS];
    pthread_t threads[THREADS];
    struct worker workers[THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t start, done;
    unsigned generation;
    int finished, quit;
    const struct bench_frame *job;
    double start_us, end_us[THREADS], cpu_us[THREADS];

    /* Background compiles. */
    pthread_t bg_thread;
    int bg_stop, bg_count;
    uint32_t bg_salt;
    double bg_end_us[MAX_BG], bg_create_us[MAX_BG];

    /* Profiling: 2 timestamps per region and slot. */
    VkQueryPool profile;
    int profiling;
    uint32_t region;
    uint32_t profiled_frame[BENCH_FRAMES_IN_FLIGHT]; /* frame index + 1 in the slot's queries, or 0 */

    /* The measured frames. */
    int measuring;
    uint32_t measure_from;      /* the first one's index */
    double measure_start_us, measure_end_us;
    int group_count, mt_count;
    double group_us[GROUPS][MAX_FRAMES], worker_cpu_us[MAX_FRAMES], record_wall_us[MAX_FRAMES];
    uint32_t *st_final; /* st_1080p's image, which the other 1080p frames must equal */
};

#define COLOR_OUT VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
#define FRAG_TESTS (VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
#define FRAG_SHADER VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
#define DEPTH_RW (VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
#define SAMPLED VK_ACCESS_2_SHADER_SAMPLED_READ_BIT

static double thread_cpu_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return (double)t.tv_sec * 1e6 + (double)t.tv_nsec / 1e3;
}

static uint32_t object_offset(const struct scene *s, uint32_t slot, int i)
{
    return (uint32_t)(((VkDeviceSize)slot * OBJECTS + (VkDeviceSize)i) * s->object_stride);
}

/* ---- Recording ---- */

/* The profiled configuration's timestamps around region s->region. */
static void stamp(struct scene *s, VkCommandBuffer cmd, uint32_t slot, int end)
{
    if (!s->profiling)
        return;
    vkd.CmdWriteTimestamp2(cmd, end ? VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                           s->profile, slot * 2 * REGIONS + 2 * s->region + (uint32_t)end);
    s->region += (uint32_t)end;
}

/* The slot's previous profiled frame, if measured, into group_us. */
static void collect_profile(struct scene *s, uint32_t slot)
{
    uint64_t ts[2 * REGIONS];
    double groups[GROUPS] = { 0 };
    int k;
    if (!s->profiled_frame[slot] || s->profiled_frame[slot] - 1 < s->measure_from || !s->measuring ||
        s->group_count >= MAX_FRAMES)
        return;
    s->profiled_frame[slot] = 0;
    if (vkd.GetQueryPoolResults(s->vk->device, s->profile, slot * 2 * REGIONS, 2 * REGIONS, sizeof(ts), ts,
                                sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        return;
    for (k = 0; k < REGIONS; k++)
        groups[region_group(k)] += (double)(ts[2 * k + 1] - ts[2 * k]) * s->vk->timestamp_period / 1000.0;
    for (k = 0; k < GROUPS; k++)
        s->group_us[k][s->group_count] = groups[k];
    s->group_count++;
}

static VkImageMemoryBarrier2 shadow_barrier(const struct scene *s, VkImageLayout from, VkImageLayout to,
                                            VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                            VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access)
{
    VkImageMemoryBarrier2 b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = s->shadow,
        .subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, CASCADES },
    };
    return b;
}

/* Streaming, on the CPU: the slot's object constants (one field changes
 * every frame), UI geometry and texture tiles. */
static void stream(struct scene *s, const struct bench_frame *f)
{
    const struct world *w = s->world;
    char *objects = (char *)s->objects.map;
    int i;
    for (i = 0; i < OBJECTS; i++) {
        struct object_ubo *o = (struct object_ubo *)(objects + object_offset(s, f->slot, i));
        *o = w->objects[i].ubo;
        o->emissive[3] = (float)(f->index & 7); /* not shaded */
    }
    memcpy(s->ui_vb[f->slot].map, s->ui_vertices, sizeof(struct ui_vertex) * WIDGETS * WIDGET_QUADS * 4);
    memcpy(s->ui_ib[f->slot].map, s->ui_indices, sizeof(uint16_t) * WIDGETS * WIDGET_QUADS * 6);
    for (i = 0; i < STREAMED_TILES; i++)
        memcpy((char *)s->staging[f->slot].map + (size_t)i * TEX_SIZE * TEX_SIZE * 4,
               w->texels[(f->index * STREAMED_TILES + (uint32_t)i) % TEXTURES], TEX_SIZE * TEX_SIZE * 4);
}

/* The streamed texture tiles, from the slot's staging buffer. */
static void record_uploads(struct scene *s, VkCommandBuffer cmd, const struct bench_frame *f)
{
    const struct bench_image *tex[STREAMED_TILES];
    VkImageMemoryBarrier2 b[STREAMED_TILES];
    VkBufferImageCopy region = {
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageExtent = { TEX_SIZE, TEX_SIZE, 1 },
    };
    int k;
    for (k = 0; k < STREAMED_TILES; k++) {
        tex[k] = &s->textures[(f->index * STREAMED_TILES + (uint32_t)k) % TEXTURES];
        /* After the last frame's G-buffer sampled it. */
        b[k] = bench_image_barrier_info(tex[k], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, FRAG_SHADER, 0,
                                        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    }
    bench_image_barriers(cmd, STREAMED_TILES, b);
    for (k = 0; k < STREAMED_TILES; k++) {
        region.bufferOffset = (VkDeviceSize)k * TEX_SIZE * TEX_SIZE * 4;
        vkd.CmdCopyBufferToImage(cmd, s->staging[f->slot].buffer, tex[k]->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 1, &region);
    }
    for (k = 0; k < STREAMED_TILES; k++)
        b[k] = bench_image_barrier_info(tex[k], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                                        VK_ACCESS_2_TRANSFER_WRITE_BIT, FRAG_SHADER, SAMPLED);
    bench_image_barriers(cmd, STREAMED_TILES, b);
}

/* The slot's transient descriptor sets, rebuilt every frame: 0 culling,
 * 1 lighting, 2 tonemap, 3.. downsampling into level l, then upsampling. */
static void write_frame_sets(struct scene *s, uint32_t slot)
{
    VkDevice dev = s->vk->device;
    VkDescriptorSetLayout layouts[TRANSIENT_SETS];
    VkDescriptorSetAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s->frame_pools[slot],
        .descriptorSetCount = TRANSIENT_SETS,
        .pSetLayouts = layouts,
    };
    VkDescriptorImageInfo images[24];
    VkDescriptorBufferInfo buffers[2] = { { s->lights.buffer, 0, VK_WHOLE_SIZE }, { s->tiles.buffer, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet writes[32];
    VkDescriptorSet *sets = s->frame_sets[slot];
    uint32_t n = 0, ni = 0, k;

#define FRAME_IMAGE(set_, binding_, view_, sampler_)                                                \
    do {                                                                                            \
        VkDescriptorImageInfo ii_ = { (sampler_), (view_), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL }; \
        VkWriteDescriptorSet w_ = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = (set_), \
                                    .dstBinding = (binding_), .descriptorCount = 1,                \
                                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,   \
                                    .pImageInfo = &images[ni] };                                    \
        images[ni++] = ii_;                                                                         \
        writes[n++] = w_;                                                                           \
    } while (0)
#define FRAME_BUFFER(set_, binding_, info_)                                                         \
    do {                                                                                            \
        VkWriteDescriptorSet w_ = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = (set_), \
                                    .dstBinding = (binding_), .descriptorCount = 1,                \
                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,           \
                                    .pBufferInfo = (info_) };                                       \
        writes[n++] = w_;                                                                           \
    } while (0)

    layouts[0] = s->cull_layout;
    layouts[1] = s->light_layout;
    layouts[2] = s->tex2_layout;
    for (k = 3; k < TRANSIENT_SETS; k++)
        layouts[k] = s->tex1_layout;
    BENCH_VK(vkd.ResetDescriptorPool(dev, s->frame_pools[slot], 0));
    BENCH_VK(vkd.AllocateDescriptorSets(dev, &alloc, sets));
    FRAME_IMAGE(sets[0], 0, s->depth.view, s->nearest);
    FRAME_BUFFER(sets[0], 1, &buffers[0]);
    FRAME_BUFFER(sets[0], 2, &buffers[1]);
    for (k = 0; k < GBUFFERS; k++)
        FRAME_IMAGE(sets[1], k, s->gbuffer[k].view, s->nearest);
    FRAME_IMAGE(sets[1], 3, s->depth.view, s->nearest);
    FRAME_IMAGE(sets[1], 4, s->shadow_array, VK_NULL_HANDLE); /* immutable comparison sampler */
    FRAME_BUFFER(sets[1], 5, &buffers[0]);
    FRAME_BUFFER(sets[1], 6, &buffers[1]);
    FRAME_IMAGE(sets[2], 0, s->hdr.view, s->nearest);
    FRAME_IMAGE(sets[2], 1, s->bloom[0].view, s->linear);
    for (k = 0; k < BLOOM_LEVELS; k++)
        FRAME_IMAGE(sets[3 + k], 0, k == 0 ? s->hdr.view : s->bloom[k - 1].view, s->linear);
    for (k = 0; k + 1 < BLOOM_LEVELS; k++)
        FRAME_IMAGE(sets[3 + BLOOM_LEVELS + k], 0, s->bloom[k + 1].view, s->linear);
#undef FRAME_IMAGE
#undef FRAME_BUFFER
    vkd.UpdateDescriptorSets(dev, n, writes, 0, NULL);
}

static void begin_cascade(struct scene *s, VkCommandBuffer cmd, int c, VkRenderingFlags flags)
{
    VkRenderingAttachmentInfo depth = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = s->shadow_layers[c],
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    VkRenderingInfo info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .flags = flags,
        .renderArea = { { 0, 0 }, { SHADOW_SIZE, SHADOW_SIZE } },
        .layerCount = 1,
        .pDepthAttachment = &depth,
    };
    depth.clearValue.depthStencil.depth = 1.0f;
    vkd.CmdBeginRendering(cmd, &info);
}

/* Cascade c's draws: the screen scaled by 2^-c about its centre. */
static void record_cascade_draws(struct scene *s, VkCommandBuffer cmd, uint32_t slot, int c)
{
    float scale = ldexpf(1.0f, -c), view[4] = { 0.0f, 0.0f, scale, scale };
    int k;
    bench_cmd_viewport(cmd, SHADOW_SIZE, SHADOW_SIZE);
    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->shadow_pipeline);
    if (s->dynamic_depth_bounds)
        vkd.CmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
    vkd.CmdPushConstants(cmd, s->scene_playout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(view), view);
    for (k = 0; k < CASCADE_DRAWS; k++) {
        uint32_t offset = object_offset(s, slot, (c * 750 + k) % OBJECTS);
        vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->scene_playout, 1, 1, &s->object_set, 1,
                                  &offset);
        vkd.CmdDraw(cmd, 6, 1, 0, 0);
    }
}

static void record_cascades(struct scene *s, VkCommandBuffer cmd, uint32_t slot, int first, int count)
{
    int c;
    for (c = first; c < first + count; c++) {
        stamp(s, cmd, slot, 0);
        begin_cascade(s, cmd, c, 0);
        record_cascade_draws(s, cmd, slot, c);
        vkd.CmdEndRendering(cmd);
        stamp(s, cmd, slot, 1);
    }
}

static void begin_gbuffer(struct scene *s, VkCommandBuffer cmd, VkRenderingFlags flags)
{
    VkRenderingAttachmentInfo colors[GBUFFERS], depth = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = s->depth.view,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    VkRenderingInfo info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .flags = flags,
        .renderArea = { { 0, 0 }, { s->width, s->height } },
        .layerCount = 1,
        .colorAttachmentCount = GBUFFERS,
        .pColorAttachments = colors,
        .pDepthAttachment = &depth,
    };
    int k;
    for (k = 0; k < GBUFFERS; k++) {
        VkRenderingAttachmentInfo a = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = s->gbuffer[k].view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        };
        colors[k] = a;
    }
    colors[1].clearValue.color.float32[2] = -1.0f; /* the background's normal */
    depth.clearValue.depthStencil.depth = 1.0f;
    vkd.CmdBeginRendering(cmd, &info);
}

/* Objects [first, first + count), in their order (sorted by pipeline, then
 * material). */
static void record_gbuffer_draws(struct scene *s, VkCommandBuffer cmd, uint32_t slot, int first, int count)
{
    static const float view[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
    const struct world *w = s->world;
    int i, pipeline = -1, material = -1;
    bench_cmd_viewport(cmd, s->width, s->height);
    vkd.CmdPushConstants(cmd, s->scene_playout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(view), view);
    for (i = first; i < first + count; i++) {
        const struct object *o = &w->objects[i];
        uint32_t offset = object_offset(s, slot, i);
        if (o->pipeline != pipeline) {
            vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->gbuffer_pipelines[o->pipeline]);
            if (s->dynamic_depth_bounds && pipeline < 0)
                vkd.CmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
            pipeline = o->pipeline;
        }
        if (o->material != material) {
            float texel_offset[4] = { w->materials[o->material].offset[0], w->materials[o->material].offset[1], 0, 0 };
            material = o->material;
            vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->scene_playout, 0, 1,
                                      &s->materials[material], 0, NULL);
            vkd.CmdPushConstants(cmd, s->scene_playout, VK_SHADER_STAGE_FRAGMENT_BIT, 16, sizeof(texel_offset),
                                 texel_offset);
        }
        vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->scene_playout, 1, 1, &s->object_set, 1,
                                  &offset);
        vkd.CmdDraw(cmd, 6, 1, 0, 0);
    }
}

static void record_shadow_start(struct scene *s, VkCommandBuffer cmd)
{
    /* After the last frame's lighting read the cascades. */
    VkImageMemoryBarrier2 b = shadow_barrier(s, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                             FRAG_SHADER, 0, FRAG_TESTS, DEPTH_RW);
    bench_image_barriers(cmd, 1, &b);
}

static void record_shadow_end(struct scene *s, VkCommandBuffer cmd)
{
    VkImageMemoryBarrier2 b = shadow_barrier(s, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, FRAG_TESTS,
                                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, FRAG_SHADER, SAMPLED);
    bench_image_barriers(cmd, 1, &b);
}

static void record_gbuffer_start(struct scene *s, VkCommandBuffer cmd)
{
    VkImageMemoryBarrier2 b[GBUFFERS + 1];
    int k;
    /* After the last frame's reads (lighting, culling, readback). */
    for (k = 0; k < GBUFFERS; k++)
        b[k] = bench_image_barrier_info(&s->gbuffer[k], VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        FRAG_SHADER | VK_PIPELINE_STAGE_2_COPY_BIT, 0, COLOR_OUT,
                                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    b[GBUFFERS] = bench_image_barrier_info(&s->depth, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                           FRAG_SHADER | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, FRAG_TESTS, DEPTH_RW);
    bench_image_barriers(cmd, GBUFFERS + 1, b);
}

static void record_gbuffer_end(struct scene *s, VkCommandBuffer cmd)
{
    VkImageMemoryBarrier2 b[GBUFFERS + 1];
    int k;
    for (k = 0; k < GBUFFERS; k++)
        b[k] = bench_image_barrier_info(&s->gbuffer[k], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, COLOR_OUT,
                                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, FRAG_SHADER, SAMPLED);
    b[GBUFFERS] = bench_image_barrier_info(&s->depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, FRAG_TESTS,
                                           VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                           FRAG_SHADER | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, SAMPLED);
    bench_image_barriers(cmd, GBUFFERS + 1, b);
}

static void record_gbuffer(struct scene *s, VkCommandBuffer cmd, uint32_t slot)
{
    record_gbuffer_start(s, cmd);
    stamp(s, cmd, slot, 0);
    begin_gbuffer(s, cmd, 0);
    record_gbuffer_draws(s, cmd, slot, 0, OBJECTS);
    vkd.CmdEndRendering(cmd);
    stamp(s, cmd, slot, 1);
    record_gbuffer_end(s, cmd);
}

static void record_cull(struct scene *s, VkCommandBuffer cmd, uint32_t slot)
{
    struct {
        float inv_size[2];
        uint32_t tiles_x, light_count;
    } pc = { { 1.0f / (float)s->width, 1.0f / (float)s->height }, s->tiles_x, LIGHTS };
    /* After the last frame's lighting (and readback) read the tile lists. */
    bench_memory_barrier(cmd, FRAG_SHADER | VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0);
    stamp(s, cmd, slot, 0);
    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->cull_pipeline);
    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->cull_playout, 0, 1, &s->frame_sets[slot][0], 0,
                              NULL);
    vkd.CmdPushConstants(cmd, s->cull_playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkd.CmdDispatch(cmd, s->tiles_x, s->tiles_y, 1);
    stamp(s, cmd, slot, 1);
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, FRAG_SHADER,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
}

/* A rendering instance into one color image. */
static void begin_color(VkCommandBuffer cmd, const struct bench_image *img, VkAttachmentLoadOp load)
{
    VkRenderingAttachmentInfo color = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = img->view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = load,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    VkRenderingInfo info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = { { 0, 0 }, { img->width, img->height } },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color,
    };
    vkd.CmdBeginRendering(cmd, &info);
    bench_cmd_viewport(cmd, img->width, img->height);
}

static void record_light(struct scene *s, VkCommandBuffer cmd, uint32_t slot)
{
    struct {
        float inv_size[2];
        uint32_t tiles_x, pad;
    } pc = { { 1.0f / (float)s->width, 1.0f / (float)s->height }, s->tiles_x, 0 };
    bench_image_barrier(cmd, &s->hdr, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, FRAG_SHADER,
                        0, COLOR_OUT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    stamp(s, cmd, slot, 0);
    begin_color(cmd, &s->hdr, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->light_pipeline);
    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->light_playout, 0, 1, &s->frame_sets[slot][1], 0,
                              NULL);
    vkd.CmdPushConstants(cmd, s->light_playout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    vkd.CmdDraw(cmd, 3, 1, 0, 0);
    vkd.CmdEndRendering(cmd);
    stamp(s, cmd, slot, 1);
    bench_image_barrier(cmd, &s->hdr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        COLOR_OUT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, FRAG_SHADER, SAMPLED);
}

/* One bloom pass into level l: downsampling (contents discarded) or
 * additive upsampling (loaded). */
static void bloom_pass(struct scene *s, VkCommandBuffer cmd, uint32_t slot, int l, int up)
{
    const struct bench_image *dst = &s->bloom[l];
    float pc[4] = { 1.0f / (float)dst->width, 1.0f / (float)dst->height, 0.0f, 0.0f };
    VkDescriptorSet set = s->frame_sets[slot][3 + (up ? BLOOM_LEVELS + l : l)];
    if (up)
        bench_image_barrier(cmd, dst, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            FRAG_SHADER, 0, COLOR_OUT,
                            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    else
        bench_image_barrier(cmd, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, FRAG_SHADER, 0,
                            COLOR_OUT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    stamp(s, cmd, slot, 0);
    begin_color(cmd, dst, up ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, up ? s->up_pipeline : s->down_pipeline);
    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->bloom_playout, 0, 1, &set, 0, NULL);
    vkd.CmdPushConstants(cmd, s->bloom_playout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), pc);
    vkd.CmdDraw(cmd, 3, 1, 0, 0);
    vkd.CmdEndRendering(cmd);
    stamp(s, cmd, slot, 1);
    bench_image_barrier(cmd, dst, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        COLOR_OUT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, FRAG_SHADER, SAMPLED);
}

static void record_bloom(struct scene *s, VkCommandBuffer cmd, uint32_t slot)
{
    int l;
    for (l = 0; l < BLOOM_LEVELS; l++)
        bloom_pass(s, cmd, slot, l, 0);
    for (l = BLOOM_LEVELS - 2; l >= 0; l--)
        bloom_pass(s, cmd, slot, l, 1);
}

static void record_final(struct scene *s, VkCommandBuffer cmd, uint32_t slot)
{
    struct {
        float inv_size[2];
        float exposure, bloom;
    } pc = { { 1.0f / (float)s->width, 1.0f / (float)s->height }, EXPOSURE, BLOOM_STRENGTH };
    float ui_pc[4] = { 2.0f / (float)s->width, 2.0f / (float)s->height, -1.0f, -1.0f };
    const struct bench_image *dst = &s->final[slot];
    VkDeviceSize zero = 0;
    int i;
    /* After this slot's last frame wrote (and the readback read) it. */
    bench_image_barrier(cmd, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        COLOR_OUT | VK_PIPELINE_STAGE_2_COPY_BIT, 0, COLOR_OUT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    stamp(s, cmd, slot, 0);
    begin_color(cmd, dst, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->tonemap_pipeline);
    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->tonemap_playout, 0, 1, &s->frame_sets[slot][2],
                              0, NULL);
    vkd.CmdPushConstants(cmd, s->tonemap_playout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    vkd.CmdDraw(cmd, 3, 1, 0, 0);
    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->ui_pipeline);
    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->ui_playout, 0, 1, &s->font_set, 0, NULL);
    vkd.CmdBindVertexBuffers(cmd, 0, 1, &s->ui_vb[slot].buffer, &zero);
    vkd.CmdBindIndexBuffer(cmd, s->ui_ib[slot].buffer, 0, VK_INDEX_TYPE_UINT16);
    vkd.CmdPushConstants(cmd, s->ui_playout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ui_pc), ui_pc);
    for (i = 0; i < WIDGETS; i++) {
        vkd.CmdSetScissor(cmd, 0, 1, &s->ui_scissors[i]);
        vkd.CmdDrawIndexed(cmd, WIDGET_QUADS * 6, 1, (uint32_t)(i * WIDGET_QUADS * 6), i * WIDGET_QUADS * 4, 0);
    }
    vkd.CmdEndRendering(cmd);
    stamp(s, cmd, slot, 1);
}

/* The verified frame: the final image, the G-buffer albedo, the tile lists. */
static void record_readback(struct scene *s, VkCommandBuffer cmd, uint32_t slot)
{
    VkBufferCopy region = { 0, 0, s->tiles.size };
    bench_cmd_copy_image_to_buffer(cmd, &s->final[slot], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, COLOR_OUT,
                                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, s->readback_final.buffer);
    bench_cmd_copy_image_to_buffer(cmd, &s->gbuffer[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   COLOR_OUT | FRAG_SHADER, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                   s->readback_albedo.buffer);
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    vkd.CmdCopyBuffer(cmd, s->tiles.buffer, s->readback_tiles.buffer, 1, &region);
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                         VK_ACCESS_2_HOST_READ_BIT);
}

/* Culling, lighting, bloom, final (and the readback): after the G-buffer. */
static void record_post(struct scene *s, VkCommandBuffer cmd, const struct bench_frame *f)
{
    record_cull(s, cmd, f->slot);
    record_light(s, cmd, f->slot);
    record_bloom(s, cmd, f->slot);
    record_final(s, cmd, f->slot);
    if (f->verify)
        record_readback(s, cmd, f->slot);
}

static void begin_secondary(VkCommandBuffer cmd, const VkCommandBufferInheritanceRenderingInfo *rendering)
{
    VkCommandBufferInheritanceInfo inheritance = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
        .pNext = rendering,
    };
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
        .pInheritanceInfo = &inheritance,
    };
    BENCH_VK(vkd.BeginCommandBuffer(cmd, &begin));
}

/* Worker id's part of the frame. */
static void worker_record(struct scene *s, int id, const struct bench_frame *f)
{
    uint32_t slot = f->slot;
    if (s->cfg->mode == MODE_SECONDARY) {
        VkCommandBufferInheritanceRenderingInfo shadow = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO,
            .depthAttachmentFormat = DEPTH_FORMAT,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };
        VkCommandBufferInheritanceRenderingInfo gbuffer = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO,
            .colorAttachmentCount = GBUFFERS,
            .pColorAttachmentFormats = gbuffer_formats,
            .depthAttachmentFormat = DEPTH_FORMAT,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };
        BENCH_VK(vkd.ResetCommandPool(s->vk->device, s->secondary_pools[slot][id], 0));
        begin_secondary(s->shadow_cmds[slot][id], &shadow);
        record_cascade_draws(s, s->shadow_cmds[slot][id], slot, id);
        BENCH_VK(vkd.EndCommandBuffer(s->shadow_cmds[slot][id]));
        begin_secondary(s->gbuffer_cmds[slot][id], &gbuffer);
        record_gbuffer_draws(s, s->gbuffer_cmds[slot][id], slot, id * (OBJECTS / THREADS), OBJECTS / THREADS);
        BENCH_VK(vkd.EndCommandBuffer(s->gbuffer_cmds[slot][id]));
        return;
    }
    /* MODE_PRIMARY: primary id of 4. */
    bench_frame_begin(f, (uint32_t)id);
    switch (id) {
    case 0:
        record_uploads(s, f->cmds[0], f);
        record_shadow_start(s, f->cmds[0]);
        record_cascades(s, f->cmds[0], slot, 0, 2);
        break;
    case 1:
        record_cascades(s, f->cmds[1], slot, 2, 2);
        record_shadow_end(s, f->cmds[1]);
        break;
    case 2:
        record_gbuffer(s, f->cmds[2], slot);
        break;
    default:
        record_post(s, f->cmds[3], f);
        break;
    }
    bench_frame_end(f, (uint32_t)id);
}

static void *worker_main(void *arg)
{
    struct worker *wk = arg;
    struct scene *s = wk->s;
    unsigned seen = 0;
    for (;;) {
        const struct bench_frame *f;
        double cpu;
        pthread_mutex_lock(&s->mutex);
        while (s->generation == seen && !s->quit)
            pthread_cond_wait(&s->start, &s->mutex);
        if (s->quit) {
            pthread_mutex_unlock(&s->mutex);
            return NULL;
        }
        seen = s->generation;
        f = s->job;
        pthread_mutex_unlock(&s->mutex);
        cpu = thread_cpu_us();
        worker_record(s, wk->id, f);
        cpu = thread_cpu_us() - cpu;
        pthread_mutex_lock(&s->mutex);
        s->cpu_us[wk->id] = cpu;
        s->end_us[wk->id] = bench_now_us();
        if (++s->finished == THREADS)
            pthread_cond_signal(&s->done);
        pthread_mutex_unlock(&s->mutex);
    }
}

static void workers_start(struct scene *s, const struct bench_frame *f)
{
    pthread_mutex_lock(&s->mutex);
    s->job = f;
    s->finished = 0;
    s->generation++;
    s->start_us = bench_now_us();
    pthread_cond_broadcast(&s->start);
    pthread_mutex_unlock(&s->mutex);
}

static void workers_wait(struct scene *s)
{
    double cpu = 0.0, end = s->start_us;
    int i;
    pthread_mutex_lock(&s->mutex);
    while (s->finished < THREADS)
        pthread_cond_wait(&s->done, &s->mutex);
    pthread_mutex_unlock(&s->mutex);
    for (i = 0; i < THREADS; i++) {
        cpu += s->cpu_us[i];
        end = s->end_us[i] > end ? s->end_us[i] : end;
    }
    if (s->measuring && s->mt_count < MAX_FRAMES) {
        s->worker_cpu_us[s->mt_count] = cpu;
        s->record_wall_us[s->mt_count++] = end - s->start_us;
    }
}

static void record(void *user, const struct bench_frame *f)
{
    struct scene *s = user;
    VkCommandBuffer cmd = f->cmd;
    int c;

    switch (s->cfg->mode) {
    case MODE_ST:
        stream(s, f);
        write_frame_sets(s, f->slot);
        if (s->profiling) {
            collect_profile(s, f->slot);
            vkd.CmdResetQueryPool(cmd, s->profile, f->slot * 2 * REGIONS, 2 * REGIONS);
            s->region = 0;
        }
        record_uploads(s, cmd, f);
        record_shadow_start(s, cmd);
        record_cascades(s, cmd, f->slot, 0, CASCADES);
        record_shadow_end(s, cmd);
        record_gbuffer(s, cmd, f->slot);
        record_post(s, cmd, f);
        if (s->profiling)
            s->profiled_frame[f->slot] = f->index + 1;
        break;
    case MODE_SECONDARY:
        /* Workers record the cascades and G-buffer while this thread streams. */
        workers_start(s, f);
        stream(s, f);
        write_frame_sets(s, f->slot);
        workers_wait(s);
        record_uploads(s, cmd, f);
        record_shadow_start(s, cmd);
        for (c = 0; c < CASCADES; c++) {
            begin_cascade(s, cmd, c, VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT);
            vkd.CmdExecuteCommands(cmd, 1, &s->shadow_cmds[f->slot][c]);
            vkd.CmdEndRendering(cmd);
        }
        record_shadow_end(s, cmd);
        record_gbuffer_start(s, cmd);
        begin_gbuffer(s, cmd, VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT);
        vkd.CmdExecuteCommands(cmd, THREADS, s->gbuffer_cmds[f->slot]);
        vkd.CmdEndRendering(cmd);
        record_gbuffer_end(s, cmd);
        record_post(s, cmd, f);
        break;
    case MODE_PRIMARY:
        /* The last primary binds the transient sets: write them first. */
        write_frame_sets(s, f->slot);
        workers_start(s, f);
        stream(s, f);
        workers_wait(s);
        break;
    }
}

/* ---- Setup ---- */

static VkShaderModule shader(struct scene *s, const uint32_t *code, size_t size)
{
    return bench_shader_module(s->vk, code, size);
}
#define SHADER(s, array) shader((s), (array), sizeof(array))

static VkDescriptorSetLayout set_layout(struct scene *s, uint32_t count, const VkDescriptorType *types,
                                        VkShaderStageFlags stages, int immutable_compare_binding)
{
    VkDescriptorSetLayoutBinding b[8];
    uint32_t i;
    for (i = 0; i < count; i++) {
        VkDescriptorSetLayoutBinding x = { i, types[i], 1, stages, NULL };
        if ((int)i == immutable_compare_binding)
            x.pImmutableSamplers = &s->compare;
        b[i] = x;
    }
    return bench_set_layout(s->vk, count, b, 0);
}

static VkPipelineLayout pipeline_layout(struct scene *s, uint32_t set_count, const VkDescriptorSetLayout *sets,
                                        uint32_t range_count, const VkPushConstantRange *ranges)
{
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipelineLayoutCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = set_count,
        .pSetLayouts = sets,
        .pushConstantRangeCount = range_count,
        .pPushConstantRanges = ranges,
    };
    BENCH_VK(vkd.CreatePipelineLayout(s->vk->device, &info, NULL, &layout));
    return layout;
}

static VkDescriptorSet allocate_set(struct scene *s, VkDescriptorSetLayout layout)
{
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s->pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };
    BENCH_VK(vkd.AllocateDescriptorSets(s->vk->device, &info, &set));
    return set;
}

static void write_image(struct scene *s, VkDescriptorSet set, uint32_t binding, VkImageView view, VkSampler sampler)
{
    VkDescriptorImageInfo image = { sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet w = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &image,
    };
    vkd.UpdateDescriptorSets(s->vk->device, 1, &w, 0, NULL);
}

static VkSampler sampler(struct scene *s, VkFilter filter, VkSamplerAddressMode address, int compare)
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
    BENCH_VK(vkd.CreateSampler(s->vk->device, &info, NULL, &sampler));
    return sampler;
}

static void create_pipelines(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkShaderModule object_vs = SHADER(s, frame_object_vert), material_fs = SHADER(s, frame_material_frag);
    VkShaderModule fullscreen_vs = SHADER(s, fullscreen_vert), light_fs = SHADER(s, frame_light_frag);
    VkShaderModule bloom_fs = SHADER(s, frame_bloom_frag), tonemap_fs = SHADER(s, frame_tonemap_frag);
    VkShaderModule ui_vs = SHADER(s, frame_ui_vert), ui_fs = SHADER(s, frame_ui_frag);
    VkShaderModule cull_cs = SHADER(s, frame_cull_comp);
    struct bench_gfx_state *st = bench_alloc(sizeof(*st));
    struct bench_gfx_desc d;
    int32_t variant;
    VkSpecializationMapEntry entry = { 0, 0, sizeof(int32_t) };
    VkSpecializationInfo spec = { 1, &entry, sizeof(int32_t), &variant };

    /* Depth-only shadow casters: no fragment shader, depth bias. */
    memset(&d, 0, sizeof(d));
    d.vs = object_vs;
    d.fs = material_fs;
    d.layout = s->scene_playout;
    d.depth_format = DEPTH_FORMAT;
    d.depth_test = d.depth_write = 1;
    d.depth_compare = VK_COMPARE_OP_LESS;
    if (s->dynamic_depth_bounds) {
        d.dynamic_count = 1;
        d.dynamic[0] = VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE;
    }
    bench_gfx_state_init(st, &d);
    st->info.stageCount = 1;
    st->raster.depthBiasEnable = VK_TRUE;
    st->raster.depthBiasConstantFactor = 2.0f;
    st->raster.depthBiasSlopeFactor = 1.5f;
    BENCH_VK(vkd.CreateGraphicsPipelines(vk->device, VK_NULL_HANDLE, 1, &st->info, NULL, &s->shadow_pipeline));

    /* G-buffer: 0 opaque, 1 dithered, 2 two-sided, 3 emissive. Rectangles
     * wind clockwise. */
    d.color_count = GBUFFERS;
    memcpy(d.color_formats, gbuffer_formats, sizeof(gbuffer_formats));
    d.fs_spec = &spec;
    d.front_face = VK_FRONT_FACE_CLOCKWISE;
    for (variant = 0; variant < 4; variant++) {
        d.cull = variant == 2 ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
        s->gbuffer_pipelines[variant] = bench_gfx_pipeline(vk, &d);
    }
    d.cull = VK_CULL_MODE_BACK_BIT;
    s->bg_state = bench_alloc(sizeof(*s->bg_state));
    bench_gfx_state_init(s->bg_state, &d); /* the background thread sets its own specialization */

    memset(&d, 0, sizeof(d));
    d.vs = fullscreen_vs;
    d.fs = light_fs;
    d.layout = s->light_playout;
    d.color_count = 1;
    d.color_formats[0] = HDR_FORMAT;
    s->light_pipeline = bench_gfx_pipeline(vk, &d);
    d.fs = bloom_fs;
    d.layout = s->bloom_playout;
    s->down_pipeline = bench_gfx_pipeline(vk, &d);
    bench_gfx_state_init(st, &d); /* upsampling adds */
    st->attachments[0].blendEnable = VK_TRUE;
    st->attachments[0].srcColorBlendFactor = st->attachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    st->attachments[0].srcAlphaBlendFactor = st->attachments[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    st->attachments[0].colorBlendOp = st->attachments[0].alphaBlendOp = VK_BLEND_OP_ADD;
    BENCH_VK(vkd.CreateGraphicsPipelines(vk->device, VK_NULL_HANDLE, 1, &st->info, NULL, &s->up_pipeline));
    d.fs = tonemap_fs;
    d.layout = s->tonemap_playout;
    d.color_formats[0] = FINAL_FORMAT;
    s->tonemap_pipeline = bench_gfx_pipeline(vk, &d);
    d.vs = ui_vs;
    d.fs = ui_fs;
    d.layout = s->ui_playout;
    d.blend = 1;
    d.binding_count = 1;
    d.bindings[0] = (VkVertexInputBindingDescription){ 0, sizeof(struct ui_vertex), VK_VERTEX_INPUT_RATE_VERTEX };
    d.attribute_count = 3;
    d.attributes[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(struct ui_vertex, pos) };
    d.attributes[1] = (VkVertexInputAttributeDescription){ 1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(struct ui_vertex, uv) };
    d.attributes[2] = (VkVertexInputAttributeDescription){ 2, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(struct ui_vertex, rgba) };
    s->ui_pipeline = bench_gfx_pipeline(vk, &d);
    s->cull_pipeline = bench_compute_pipeline(vk, cull_cs, NULL, s->cull_playout);
    free(st);

    /* The background thread keeps the G-buffer shaders. */
    vkd.DestroyShaderModule(vk->device, fullscreen_vs, NULL);
    vkd.DestroyShaderModule(vk->device, light_fs, NULL);
    vkd.DestroyShaderModule(vk->device, bloom_fs, NULL);
    vkd.DestroyShaderModule(vk->device, tonemap_fs, NULL);
    vkd.DestroyShaderModule(vk->device, ui_vs, NULL);
    vkd.DestroyShaderModule(vk->device, ui_fs, NULL);
    vkd.DestroyShaderModule(vk->device, cull_cs, NULL);
}

static void setup(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    struct world *w = s->world;
    VkDevice dev = vk->device;
    VkDeviceSize align = vk->props.limits.minUniformBufferOffsetAlignment;
    int i, k;

    s->nearest = sampler(s, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0);
    s->repeat = sampler(s, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT, 0);
    s->linear = sampler(s, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0);
    s->compare = sampler(s, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 1);

    /* The cascades: a layered depth image, a view per layer to render and
     * an array view to sample. */
    {
        VkImageCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = DEPTH_FORMAT,
            .extent = { SHADOW_SIZE, SHADOW_SIZE, 1 },
            .mipLevels = 1,
            .arrayLayers = CASCADES,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        };
        VkImageViewCreateInfo view = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = DEPTH_FORMAT,
            .subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 },
        };
        VkMemoryRequirements req;
        BENCH_VK(vkd.CreateImage(dev, &info, NULL, &s->shadow));
        vkd.GetImageMemoryRequirements(dev, s->shadow, &req);
        s->shadow_memory = bench_allocate(vk, &req, 0, BENCH_DEVICE_LOCAL, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, NULL);
        BENCH_VK(vkd.BindImageMemory(dev, s->shadow, s->shadow_memory, 0));
        view.image = s->shadow;
        for (i = 0; i < CASCADES; i++) {
            view.subresourceRange.baseArrayLayer = (uint32_t)i;
            BENCH_VK(vkd.CreateImageView(dev, &view, NULL, &s->shadow_layers[i]));
        }
        view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        view.subresourceRange.baseArrayLayer = 0;
        view.subresourceRange.layerCount = CASCADES;
        BENCH_VK(vkd.CreateImageView(dev, &view, NULL, &s->shadow_array));
    }
    for (i = 0; i < TEXTURES; i++) {
        bench_image_create(vk, &s->textures[i], VK_FORMAT_R8G8B8A8_UNORM, TEX_SIZE, TEX_SIZE, 1, VK_SAMPLE_COUNT_1_BIT,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0);
        bench_image_upload(vk, &s->textures[i], w->texels[i], sizeof(w->texels[i]),
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    bench_image_create(vk, &s->font, VK_FORMAT_R8_UNORM, FONT_SIZE, FONT_SIZE, 1, VK_SAMPLE_COUNT_1_BIT,
                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0);
    bench_image_upload(vk, &s->font, w->font, sizeof(w->font), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    s->object_stride = (sizeof(struct object_ubo) + align - 1) / align * align;
    bench_buffer_create(vk, &s->objects, s->object_stride * OBJECTS * BENCH_FRAMES_IN_FLIGHT,
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, BENCH_HOST, 0);
    bench_buffer_create(vk, &s->lights, sizeof(w->lights),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->lights, 0, &w->lights, sizeof(w->lights));
    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++) {
        bench_buffer_create(vk, &s->ui_vb[i], sizeof(struct ui_vertex) * WIDGETS * WIDGET_QUADS * 4,
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, BENCH_HOST, 0);
        bench_buffer_create(vk, &s->ui_ib[i], sizeof(uint16_t) * WIDGETS * WIDGET_QUADS * 6,
                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT, BENCH_HOST, 0);
        bench_buffer_create(vk, &s->staging[i], (VkDeviceSize)STREAMED_TILES * TEX_SIZE * TEX_SIZE * 4,
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, BENCH_HOST, 0);
    }

    /* Descriptors: materials, the objects' dynamic uniform buffer and the
     * font are persistent; the rest is rebuilt every frame. */
    {
        static const VkDescriptorType cis4[4] = {
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        };
        static const VkDescriptorType object[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC };
        static const VkDescriptorType light[7] = {
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        };
        static const VkDescriptorType cull[3] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
        VkDescriptorPoolSize sizes[2] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MATERIALS * 4 + 1 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1 },
        };
        VkDescriptorPoolSize frame_sizes[2] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 + 1 + 2 + 2 * BLOOM_LEVELS - 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 },
        };
        VkDescriptorPoolCreateInfo pool = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = MATERIALS + 2,
            .poolSizeCount = 2,
            .pPoolSizes = sizes,
        };
        VkDescriptorSetLayout scene_sets[2];
        VkPushConstantRange scene_push[2] = { { VK_SHADER_STAGE_VERTEX_BIT, 0, 16 },
                                              { VK_SHADER_STAGE_FRAGMENT_BIT, 16, 16 } };
        VkPushConstantRange fragment16 = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16 };
        VkPushConstantRange compute16 = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };
        VkPushConstantRange vertex16 = { VK_SHADER_STAGE_VERTEX_BIT, 0, 16 };
        VkDescriptorBufferInfo objects = { VK_NULL_HANDLE, 0, sizeof(struct object_ubo) };
        VkWriteDescriptorSet write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .pBufferInfo = &objects,
        };

        s->material_layout = set_layout(s, 4, cis4, VK_SHADER_STAGE_FRAGMENT_BIT, -1);
        s->object_layout = set_layout(s, 1, object, VK_SHADER_STAGE_VERTEX_BIT, -1);
        s->light_layout = set_layout(s, 7, light, VK_SHADER_STAGE_FRAGMENT_BIT, 4);
        s->cull_layout = set_layout(s, 3, cull, VK_SHADER_STAGE_COMPUTE_BIT, -1);
        s->tex1_layout = set_layout(s, 1, cis4, VK_SHADER_STAGE_FRAGMENT_BIT, -1);
        s->tex2_layout = set_layout(s, 2, cis4, VK_SHADER_STAGE_FRAGMENT_BIT, -1);
        BENCH_VK(vkd.CreateDescriptorPool(dev, &pool, NULL, &s->pool));
        pool.maxSets = TRANSIENT_SETS;
        pool.pPoolSizes = frame_sizes;
        for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++)
            BENCH_VK(vkd.CreateDescriptorPool(dev, &pool, NULL, &s->frame_pools[i]));
        for (i = 0; i < MATERIALS; i++) {
            s->materials[i] = allocate_set(s, s->material_layout);
            for (k = 0; k < 4; k++)
                write_image(s, s->materials[i], (uint32_t)k, s->textures[w->materials[i].textures[k]].view,
                            s->repeat);
        }
        s->object_set = allocate_set(s, s->object_layout);
        objects.buffer = s->objects.buffer;
        write.dstSet = s->object_set;
        vkd.UpdateDescriptorSets(dev, 1, &write, 0, NULL);
        s->font_set = allocate_set(s, s->tex1_layout);
        write_image(s, s->font_set, 0, s->font.view, s->nearest);

        scene_sets[0] = s->material_layout;
        scene_sets[1] = s->object_layout;
        s->scene_playout = pipeline_layout(s, 2, scene_sets, 2, scene_push);
        s->light_playout = pipeline_layout(s, 1, &s->light_layout, 1, &fragment16);
        s->cull_playout = pipeline_layout(s, 1, &s->cull_layout, 1, &compute16);
        s->bloom_playout = pipeline_layout(s, 1, &s->tex1_layout, 1, &fragment16);
        s->tonemap_playout = pipeline_layout(s, 1, &s->tex2_layout, 1, &fragment16);
        s->ui_playout = pipeline_layout(s, 1, &s->tex1_layout, 1, &vertex16);
    }
    create_pipelines(s);

    /* Workers' secondaries, profiling queries. */
    {
        VkCommandPoolCreateInfo pool = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = vk->queue_family,
        };
        VkQueryPoolCreateInfo queries = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2 * REGIONS * BENCH_FRAMES_IN_FLIGHT,
        };
        uint32_t slot;
        for (slot = 0; slot < BENCH_FRAMES_IN_FLIGHT; slot++)
            for (i = 0; i < THREADS; i++) {
                VkCommandBuffer two[2];
                VkCommandBufferAllocateInfo alloc = {
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    .level = VK_COMMAND_BUFFER_LEVEL_SECONDARY,
                    .commandBufferCount = 2,
                };
                BENCH_VK(vkd.CreateCommandPool(dev, &pool, NULL, &s->secondary_pools[slot][i]));
                alloc.commandPool = s->secondary_pools[slot][i];
                BENCH_VK(vkd.AllocateCommandBuffers(dev, &alloc, two));
                s->shadow_cmds[slot][i] = two[0];
                s->gbuffer_cmds[slot][i] = two[1];
            }
        if (vk->timestamp_bits)
            BENCH_VK(vkd.CreateQueryPool(dev, &queries, NULL, &s->profile));
    }

    pthread_mutex_init(&s->mutex, NULL);
    pthread_cond_init(&s->start, NULL);
    pthread_cond_init(&s->done, NULL);
    for (i = 0; i < THREADS; i++) {
        s->workers[i].s = s;
        s->workers[i].id = i;
        if (pthread_create(&s->threads[i], NULL, worker_main, &s->workers[i]) != 0)
            bench_fatal("pthread_create failed");
    }
}

static void destroy_targets(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    int k;
    vkd.DeviceWaitIdle(vk->device);
    for (k = 0; k < GBUFFERS; k++)
        bench_image_destroy(vk, &s->gbuffer[k]);
    bench_image_destroy(vk, &s->depth);
    bench_image_destroy(vk, &s->hdr);
    for (k = 0; k < BLOOM_LEVELS; k++)
        bench_image_destroy(vk, &s->bloom[k]);
    for (k = 0; k < BENCH_FRAMES_IN_FLIGHT; k++)
        bench_image_destroy(vk, &s->final[k]);
    bench_buffer_destroy(vk, &s->tiles);
    bench_buffer_destroy(vk, &s->readback_final);
    bench_buffer_destroy(vk, &s->readback_albedo);
    bench_buffer_destroy(vk, &s->readback_tiles);
    free(s->ui_vertices);
    free(s->ui_indices);
    s->ui_vertices = NULL;
    s->ui_indices = NULL;
    s->width = s->height = 0;
}

/* Everything sized by the resolution, and the UI laid out for it. */
static void create_targets(struct scene *s, uint32_t width, uint32_t height)
{
    struct bench_vk *vk = s->vk;
    VkImageUsageFlags color = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkDeviceSize tiles_size;
    int scale = (int)(width / 1920), i, q, c, k;

    destroy_targets(s);
    s->width = width;
    s->height = height;
    s->tiles_x = (width + 15) / 16;
    s->tiles_y = (height + 15) / 16;
    tiles_size = (VkDeviceSize)s->tiles_x * s->tiles_y * 64 * sizeof(uint32_t);
    for (k = 0; k < GBUFFERS; k++)
        bench_image_create(vk, &s->gbuffer[k], gbuffer_formats[k], width, height, 1, VK_SAMPLE_COUNT_1_BIT,
                           color | (k == 0 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0), 0);
    bench_image_create(vk, &s->depth, DEPTH_FORMAT, width, height, 1, VK_SAMPLE_COUNT_1_BIT,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 0);
    bench_image_create(vk, &s->hdr, HDR_FORMAT, width, height, 1, VK_SAMPLE_COUNT_1_BIT, color, 0);
    for (k = 0; k < BLOOM_LEVELS; k++)
        bench_image_create(vk, &s->bloom[k], HDR_FORMAT, width >> (k + 1), height >> (k + 1), 1,
                           VK_SAMPLE_COUNT_1_BIT, color, 0);
    for (k = 0; k < BENCH_FRAMES_IN_FLIGHT; k++)
        bench_image_create(vk, &s->final[k], FINAL_FORMAT, width, height, 1, VK_SAMPLE_COUNT_1_BIT,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0);
    bench_buffer_create(vk, &s->tiles, tiles_size,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_create(vk, &s->readback_final, (VkDeviceSize)width * height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        BENCH_HOST, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    bench_buffer_create(vk, &s->readback_albedo, (VkDeviceSize)width * height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        BENCH_HOST, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    bench_buffer_create(vk, &s->readback_tiles, tiles_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, BENCH_HOST,
                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

    /* UI quads in pixels; at 2160p everything is twice the size. */
    s->ui_vertices = bench_alloc(sizeof(struct ui_vertex) * WIDGETS * WIDGET_QUADS * 4);
    s->ui_indices = bench_alloc(sizeof(uint16_t) * WIDGETS * WIDGET_QUADS * 6);
    for (i = 0; i < WIDGETS; i++) {
        const struct widget *wd = &s->world->widgets[i];
        VkRect2D scissor = { { wd->x * scale, wd->y * scale }, { LABEL_W * (uint32_t)scale, GLYPH_H * (uint32_t)scale } };
        s->ui_scissors[i] = scissor;
        for (q = 0; q < WIDGET_QUADS; q++) {
            static const uint16_t corners[6] = { 0, 1, 2, 1, 3, 2 };
            struct ui_vertex *v = &s->ui_vertices[(i * WIDGET_QUADS + q) * 4];
            uint16_t *index = &s->ui_indices[(i * WIDGET_QUADS + q) * 6];
            int rect[4], cell[2];
            widget_quad(wd, i, q, scale, rect, cell);
            for (c = 0; c < 4; c++) {
                v[c].pos[0] = (float)((c & 1) ? rect[2] : rect[0]);
                v[c].pos[1] = (float)((c >> 1) ? rect[3] : rect[1]);
                v[c].uv[0] = (float)(cell[0] + (c & 1) * GLYPH_W) / FONT_SIZE;
                v[c].uv[1] = (float)(cell[1] + (c >> 1) * GLYPH_H) / FONT_SIZE;
                v[c].rgba = wd->rgba;
            }
            for (c = 0; c < 6; c++)
                index[c] = (uint16_t)(q * 4 + corners[c]);
        }
    }
}

static void teardown(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDevice dev = vk->device;
    int i;

    pthread_mutex_lock(&s->mutex);
    s->quit = 1;
    pthread_cond_broadcast(&s->start);
    pthread_mutex_unlock(&s->mutex);
    for (i = 0; i < THREADS; i++)
        pthread_join(s->threads[i], NULL);
    pthread_mutex_destroy(&s->mutex);
    pthread_cond_destroy(&s->start);
    pthread_cond_destroy(&s->done);

    destroy_targets(s);
    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++) {
        int k;
        for (k = 0; k < THREADS; k++)
            vkd.DestroyCommandPool(dev, s->secondary_pools[i][k], NULL);
        vkd.DestroyDescriptorPool(dev, s->frame_pools[i], NULL);
        bench_buffer_destroy(vk, &s->ui_vb[i]);
        bench_buffer_destroy(vk, &s->ui_ib[i]);
        bench_buffer_destroy(vk, &s->staging[i]);
    }
    if (s->profile)
        vkd.DestroyQueryPool(dev, s->profile, NULL);
    vkd.DestroyPipeline(dev, s->shadow_pipeline, NULL);
    for (i = 0; i < 4; i++)
        vkd.DestroyPipeline(dev, s->gbuffer_pipelines[i], NULL);
    vkd.DestroyPipeline(dev, s->cull_pipeline, NULL);
    vkd.DestroyPipeline(dev, s->light_pipeline, NULL);
    vkd.DestroyPipeline(dev, s->down_pipeline, NULL);
    vkd.DestroyPipeline(dev, s->up_pipeline, NULL);
    vkd.DestroyPipeline(dev, s->tonemap_pipeline, NULL);
    vkd.DestroyPipeline(dev, s->ui_pipeline, NULL);
    vkd.DestroyShaderModule(dev, s->bg_state->stages[0].module, NULL);
    vkd.DestroyShaderModule(dev, s->bg_state->stages[1].module, NULL);
    free(s->bg_state);
    vkd.DestroyPipelineLayout(dev, s->scene_playout, NULL);
    vkd.DestroyPipelineLayout(dev, s->light_playout, NULL);
    vkd.DestroyPipelineLayout(dev, s->cull_playout, NULL);
    vkd.DestroyPipelineLayout(dev, s->bloom_playout, NULL);
    vkd.DestroyPipelineLayout(dev, s->tonemap_playout, NULL);
    vkd.DestroyPipelineLayout(dev, s->ui_playout, NULL);
    vkd.DestroyDescriptorPool(dev, s->pool, NULL);
    vkd.DestroyDescriptorSetLayout(dev, s->material_layout, NULL);
    vkd.DestroyDescriptorSetLayout(dev, s->object_layout, NULL);
    vkd.DestroyDescriptorSetLayout(dev, s->light_layout, NULL);
    vkd.DestroyDescriptorSetLayout(dev, s->cull_layout, NULL);
    vkd.DestroyDescriptorSetLayout(dev, s->tex1_layout, NULL);
    vkd.DestroyDescriptorSetLayout(dev, s->tex2_layout, NULL);
    bench_buffer_destroy(vk, &s->objects);
    bench_buffer_destroy(vk, &s->lights);
    for (i = 0; i < TEXTURES; i++)
        bench_image_destroy(vk, &s->textures[i]);
    bench_image_destroy(vk, &s->font);
    for (i = 0; i < CASCADES; i++)
        vkd.DestroyImageView(dev, s->shadow_layers[i], NULL);
    vkd.DestroyImageView(dev, s->shadow_array, NULL);
    vkd.DestroyImage(dev, s->shadow, NULL);
    vkd.FreeMemory(dev, s->shadow_memory, NULL);
    vkd.DestroySampler(dev, s->nearest, NULL);
    vkd.DestroySampler(dev, s->repeat, NULL);
    vkd.DestroySampler(dev, s->linear, NULL);
    vkd.DestroySampler(dev, s->compare, NULL);
    free(s->st_final);
}

/* ==========================================================================
 * Background compiles
 * ========================================================================== */

static int bg_stopped(struct scene *s)
{
    int stop;
    pthread_mutex_lock(&s->mutex);
    stop = s->bg_stop;
    pthread_mutex_unlock(&s->mutex);
    return stop;
}

/* G-buffer pipelines, each with a specialization constant no pipeline had
 * before (the per-run salt), so no cache can serve it: what an engine's
 * shader-compile thread does while the game renders. */
static void *bg_main(void *arg)
{
    struct scene *s = arg;
    int32_t value;
    VkSpecializationMapEntry entry = { 0, 0, sizeof(int32_t) };
    VkSpecializationInfo spec = { 1, &entry, sizeof(int32_t), &value };
    uint32_t n = 0;
    s->bg_state->stages[1].pSpecializationInfo = &spec;
    while (!bg_stopped(s)) {
        VkPipeline pipeline = VK_NULL_HANDLE;
        double t0, t1;
        VkResult r;
        value = (int32_t)((s->bg_salt + n++) & 0x3FFFFFFFu) | 16; /* >= 4: shades like variant 0 */
        t0 = bench_now_us();
        r = vkd.CreateGraphicsPipelines(s->vk->device, VK_NULL_HANDLE, 1, &s->bg_state->info, NULL, &pipeline);
        t1 = bench_now_us();
        if (r == VK_SUCCESS)
            vkd.DestroyPipeline(s->vk->device, pipeline, NULL);
        pthread_mutex_lock(&s->mutex);
        if (r == VK_SUCCESS && s->bg_count < MAX_BG) {
            s->bg_end_us[s->bg_count] = t1;
            s->bg_create_us[s->bg_count++] = t1 - t0;
        }
        pthread_mutex_unlock(&s->mutex);
    }
    return NULL;
}

static uint32_t new_salt(void)
{
    uint64_t t = mach_absolute_time();
    return (uint32_t)t ^ (uint32_t)(t >> 32) ^ ((uint32_t)getpid() << 16);
}

/* ==========================================================================
 * Verification and the run
 * ========================================================================== */

static int verify(struct scene *s)
{
    const struct reference *r = s->ref;
    const char *name = s->cfg->name;
    const uint32_t *final = s->readback_final.map, *tiles = s->readback_tiles.map;
    size_t n = (size_t)s->width * s->height, p, bad = 0, first = 0;
    uint32_t t, k;

    if (!bench_check_rgba8(name, s->readback_albedo.map, r->albedo, (int)s->width, (int)s->height))
        return 0;
    for (t = 0; t < s->tiles_x * s->tiles_y; t++)
        for (k = 0; k <= r->tiles[64 * t] && k < 64; k++)
            if (tiles[64 * t + k] != r->tiles[64 * t + k]) {
                bench_error(name, "tile (%u,%u) word %u: got %u, want %u", t % s->tiles_x, t / s->tiles_x, k,
                            tiles[64 * t + k], r->tiles[64 * t + k]);
                return 0;
            }
    for (p = 0; p < n; p++)
        if (final[p] >> 24 != r->final[p] >> 24 ||
            !bench_rgba8_close(final[p], r->final[p], r->glyph[p] ? 0 : 2))
            if (bad++ == 0)
                first = p;
    if (bad) {
        bench_error(name, "%zu of %zu pixels wrong; first at (%zu,%zu): got 0x%08x, want 0x%08x", bad, n,
                    first % s->width, first / s->width, final[first], r->final[first]);
        return 0;
    }
    /* Threads, secondaries, timestamps and compiles must not change a bit. */
    if (s->width == 1920 && !s->dynamic_depth_bounds) {
        if (!s->st_final && strcmp(name, "st_1080p") == 0) {
            s->st_final = bench_alloc(sizeof(uint32_t) * n);
            memcpy(s->st_final, final, sizeof(uint32_t) * n);
        } else if (s->st_final && memcmp(s->st_final, final, sizeof(uint32_t) * n) != 0) {
            for (p = 0; p < n && s->st_final[p] == final[p]; p++)
                ;
            bench_error(name, "differs from st_1080p; first at (%zu,%zu): got 0x%08x, st_1080p 0x%08x",
                        p % s->width, p / s->width, final[p], s->st_final[p]);
            return 0;
        }
    }
    return 1;
}

/* Unmeasured frames for at least WARMUP_US (GPU clocks ramp up), then the
 * measured ones. */
static void run(struct scene *s, struct bench_loop *loop, int frames)
{
    double start = bench_now_us();
    int i;
    loop->count = loop->gpu_count = 0;
    s->measuring = 0;
    for (i = 0; i < 10 || bench_now_us() - start < WARMUP_US; i++)
        bench_loop_frame(loop, record, s, 0, 0);
    s->measure_from = loop->next_index;
    s->group_count = s->mt_count = 0;
    s->measuring = 1;
    s->measure_start_us = bench_now_us();
    for (i = 0; i < frames; i++)
        bench_loop_frame(loop, record, s, 0, 1);
    bench_loop_wait_all(loop);
    s->measure_end_us = bench_now_us();
    if (s->profiling)
        for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++)
            collect_profile(s, (uint32_t)i);
    s->measuring = 0;
}

static void run_config(struct scene *s, struct bench_loop *loop, const struct config *cfg)
{
    int frames = bench_frames(cfg->frames), bg = 0, g;
    double cpu_median;
    struct bench_json j;

    if (frames > MAX_FRAMES)
        frames = MAX_FRAMES;
    s->cfg = cfg;
    if (s->width != cfg->width)
        create_targets(s, cfg->width, cfg->height);
    s->ref = &s->world->refs[cfg->width == 1920 ? 0 : 1];
    if (!s->ref->complete)
        reference_complete(s->world, s->ref);
    s->profiling = cfg->profiled && s->profile;
    if (cfg->profiled && !s->profile) {
        bench_skip(cfg->name, "no timestamps on this queue");
        return;
    }
    memset(s->profiled_frame, 0, sizeof(s->profiled_frame));
    bench_loop_primaries(loop, cfg->mode == MODE_PRIMARY ? THREADS : 1);
    bench_loop_verify(loop, record, s);
    if (!verify(s))
        return;

    if (cfg->bgcompile) {
        s->bg_stop = 0;
        s->bg_count = 0;
        s->bg_salt = new_salt();
        if (pthread_create(&s->bg_thread, NULL, bg_main, s) != 0)
            bench_fatal("pthread_create failed");
        bg = 1;
    }
    run(s, loop, frames);
    if (bg) {
        pthread_mutex_lock(&s->mutex);
        s->bg_stop = 1;
        pthread_mutex_unlock(&s->mutex);
        pthread_join(s->bg_thread, NULL);
    }

    bench_json_begin(&j, cfg->name);
    bench_json_int(&j, "width", cfg->width);
    bench_json_int(&j, "height", cfg->height);
    bench_json_int(&j, "draws", DRAWS);
    bench_json_int(&j, "passes", PASSES);
    bench_json_int(&j, "dispatches", 1);
    bench_json_int(&j, "worker_threads", cfg->mode == MODE_ST ? 0 : THREADS);
    bench_json_int(&j, "primaries", cfg->mode == MODE_PRIMARY ? THREADS : 1);
    cpu_median = bench_loop_cpu_median(loop);
    bench_loop_report(loop, &j);
    bench_json_num(&j, "cpu_ns_per_draw", cpu_median * 1000.0 / DRAWS);
    if (s->mt_count) {
        bench_json_stats(&j, "worker_cpu_us", s->worker_cpu_us, s->mt_count);
        bench_json_stats(&j, "record_wall_us", s->record_wall_us, s->mt_count);
    }
    if (s->profiling)
        for (g = 0; g < GROUPS; g++)
            bench_json_stats(&j, group_names[g], s->group_us[g], s->group_count);
    if (bg) {
        /* Compiles that finished while the frames were measured. */
        double *create = bench_alloc(sizeof(double) * MAX_BG);
        int i, n = 0;
        for (i = 0; i < s->bg_count; i++)
            if (s->bg_end_us[i] >= s->measure_start_us && s->bg_end_us[i] <= s->measure_end_us)
                create[n++] = s->bg_create_us[i];
        bench_json_int(&j, "bg_compiles", n);
        bench_json_num(&j, "bg_compiles_per_s", n / ((s->measure_end_us - s->measure_start_us) / 1e6));
        bench_json_stats(&j, "create_us", create, n);
        free(create);
    }
    bench_json_end(&j);
}

int main(void)
{
    struct world *w = bench_alloc(sizeof(*w));
    int pass;
    size_t c;

    bench_begin("frame");
    make_world(w);
    reference_gbuffer(w, &w->refs[0], 1920, 1080);
    reference_gbuffer(w, &w->refs[1], 3840, 2160);
    make_lights(w);

    /* The depth-bounds configuration runs on a device of its own. */
    for (pass = 0; pass < 2; pass++) {
        struct bench_vk vk;
        struct bench_vk_wants wants;
        struct bench_loop loop;
        struct scene *s;
        int any = 0;

        for (c = 0; c < CONFIGS; c++)
            any |= configs[c].depth_bounds == pass && bench_selected(configs[c].name);
        if (!any)
            continue;
        memset(&wants, 0, sizeof(wants));
        wants.features.core.features.depthBounds = pass ? VK_TRUE : VK_FALSE;
        bench_config = NULL;
        bench_vk_init_with(&vk, BENCH_NEED_DYNAMIC_RENDERING | BENCH_NEED_SYNC2, &wants);
        if (pass && !vk.enabled.core.features.depthBounds) {
            for (c = 0; c < CONFIGS; c++)
                if (configs[c].depth_bounds && bench_selected(configs[c].name))
                    bench_skip(configs[c].name, "depthBounds not supported");
            bench_vk_destroy(&vk);
            continue;
        }
        s = bench_alloc(sizeof(*s));
        s->vk = &vk;
        s->world = w;
        s->dynamic_depth_bounds = pass;
        setup(s);
        bench_loop_init(&loop, &vk, MAX_FRAMES);
        for (c = 0; c < CONFIGS; c++) {
            if (configs[c].depth_bounds != pass || !bench_selected(configs[c].name))
                continue;
            bench_config = configs[c].name;
            run_config(s, &loop, &configs[c]);
        }
        bench_config = NULL;
        bench_loop_destroy(&loop);
        teardown(s);
        free(s);
        bench_vk_destroy(&vk);
    }
    reference_free(&w->refs[0]);
    reference_free(&w->refs[1]);
    free(w);
    return bench_exit_code();
}
