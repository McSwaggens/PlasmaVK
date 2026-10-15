/* Render areas smaller than the attachments, as engines use them:
 *
 *   atlas_16tiles         a shadow atlas updated tile by tile: 16 depth-only
 *                         passes per frame, each into one 1024x1024 tile of an
 *                         8192x8192 D32 atlas (renderArea = the tile,
 *                         CLEAR/STORE), 128 casters per tile; the tiles are
 *                         spread over the whole atlas
 *   separate_16tiles      the same passes into 16 1024x1024 images of their
 *                         own: the cost if only the tiles were touched
 *   dynres70_2160p        dynamic resolution: a scene pass (1000 objects) into
 *                         a 3840x2160 RGBA16F + D32 target at 70% scale,
 *                         renderArea (0,0) 2688x1512; color CLEAR/STORE,
 *                         depth CLEAR/DONT_CARE (only the pass reads it)
 *   dynres70_2160p_exact  the same pass into 2688x1512 images
 *
 * The pairs differ only in the size of the attachments: GPU time is the
 * number to watch. The first frame of each configuration is read back. The
 * render area must hold the CPU reference and everything outside it what it
 * held before, the DONT_CARE depth included: Vulkan's load and store
 * operations only touch the render area.
 */
#include "bench_vk.h"
#include "bench_shaders.h"

#define ATLAS 8192
#define TILE 1024
#define TILES 16
#define CASTERS 128 /* per tile */
#define DYN_W 3840
#define DYN_H 2160
#define AREA_W 2688 /* 70% */
#define AREA_H 1512
#define OBJECTS 1000
#define BORDER 64 /* atlas pixels read back around the verified tile */
#define VERIFY_TILE 5

/* What the passes leave outside the render area must survive: the setup fills it. */
#define OUTSIDE_DEPTH 0.5f
static const float outside_color[4] = { 0.25f, 0.5f, 0.75f, 1.0f };

/* Device commands that bench_vk.h does not load. */
static struct {
    PFN_vkCmdClearColorImage CmdClearColorImage;
    PFN_vkCmdClearDepthStencilImage CmdClearDepthStencilImage;
} vkx;

enum mode { MODE_ATLAS, MODE_SEPARATE, MODE_DYNRES, MODE_DYNRES_EXACT };

struct config {
    const char *name;
    enum mode mode;
    int frames;
};

static const struct config configs[] = {
    { "atlas_16tiles", MODE_ATLAS, 100 },
    { "separate_16tiles", MODE_SEPARATE, 100 },
    { "dynres70_2160p", MODE_DYNRES, 200 },
    { "dynres70_2160p_exact", MODE_DYNRES_EXACT, 200 },
};

/* rect_depth.vert's per-instance attributes. */
struct instance {
    float rect[4];
    uint32_t rgba;
    float depth;
};

struct object {
    struct bench_rect r; /* pixels of the tile or of the render area */
    float depth;         /* distinct within a tile or scene */
};

struct scene {
    struct bench_vk *vk;
    const struct config *cfg;
    struct object casters[TILES][CASTERS], objects[OBJECTS];
    struct bench_image atlas, tiles[TILES];
    struct bench_image color, depth;             /* 3840x2160 */
    struct bench_image color_exact, depth_exact; /* 2688x1512 */
    struct bench_buffer instances, readback;
    VkPipelineLayout layout;
    VkPipeline depth_pipeline, color_pipeline;
};

/* Tile t's corner in the atlas: every other tile of the 8x8 grid. */
static int tile_x(int t)
{
    return (t % 4) * 2 * TILE;
}

static int tile_y(int t)
{
    return (t / 4) * 2 * TILE;
}

/* Distinct depths in (0, 1), shuffled: the nearest object wins regardless of
 * draw order. */
static void make_objects(struct object *objects, int count, int width, int height, int max_size, uint64_t seed)
{
    struct bench_rect *rects = bench_alloc(sizeof(*rects) * (size_t)count);
    int i;
    bench_rect_scene(rects, count, width, height, max_size, 0, seed);
    for (i = 0; i < count; i++) {
        objects[i].r = rects[i];
        objects[i].depth = (float)((i * 7919) % count + 1) / (float)(count + 2);
    }
    free(rects);
}

static void begin_pass(VkCommandBuffer cmd, const struct bench_image *color, const struct bench_image *depth,
                       VkAttachmentStoreOp depth_store, int32_t x, int32_t y, uint32_t w, uint32_t h)
{
    VkRenderingAttachmentInfo c = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue.color.float32 = { 0.0f, 0.0f, 0.0f, 1.0f },
    };
    VkRenderingAttachmentInfo d = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = depth->view,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = depth_store,
        .clearValue.depthStencil = { 1.0f, 0 },
    };
    VkRenderingInfo info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = { { x, y }, { w, h } },
        .layerCount = 1,
        .pDepthAttachment = &d,
    };
    VkViewport viewport = { (float)x, (float)y, (float)w, (float)h, 0.0f, 1.0f };
    VkRect2D scissor = { { x, y }, { w, h } };
    if (color) {
        c.imageView = color->view;
        info.colorAttachmentCount = 1;
        info.pColorAttachments = &c;
    }
    vkd.CmdBeginRendering(cmd, &info);
    vkd.CmdSetViewport(cmd, 0, 1, &viewport);
    vkd.CmdSetScissor(cmd, 0, 1, &scissor);
}

/* Depth attachment whose contents (outside the area) carry over between frames. */
static void depth_to_attachment(VkCommandBuffer cmd, const struct bench_image *img, VkImageLayout from)
{
    bench_image_barrier(cmd, img, from, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
}

/* Shadow maps and the scene color are sampled by the passes that follow. */
static void to_sampled(VkCommandBuffer cmd, const struct bench_image *img, VkImageLayout from)
{
    int depth = img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT;
    bench_image_barrier(cmd, img, from, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        depth ? VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
                              : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        depth ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

/* Records a copy of [x, x+w) x [y, y+h) of img (in layout) to the readback
 * buffer at offset, then returns img to layout. */
static void read_back(struct scene *s, VkCommandBuffer cmd, const struct bench_image *img, VkImageLayout layout,
                      int32_t x, int32_t y, uint32_t w, uint32_t h, VkDeviceSize offset)
{
    VkBufferImageCopy region = {
        .bufferOffset = offset,
        .imageSubresource = { img->aspect, 0, 0, 1 },
        .imageOffset = { x, y, 0 },
        .imageExtent = { w, h, 1 },
    };
    bench_image_barrier(cmd, img, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    vkd.CmdCopyImageToBuffer(cmd, img->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s->readback.buffer, 1, &region);
    bench_image_barrier(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout, VK_PIPELINE_STAGE_2_COPY_BIT, 0,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
}

static void record(void *user, const struct bench_frame *f)
{
    struct scene *s = user;
    VkCommandBuffer cmd = f->cmd;
    VkDeviceSize zero = 0;
    int t;

    vkd.CmdBindVertexBuffers(cmd, 0, 1, &s->instances.buffer, &zero);
    switch (s->cfg->mode) {
    case MODE_ATLAS:
    case MODE_SEPARATE: {
        int atlas = s->cfg->mode == MODE_ATLAS;
        if (atlas)
            depth_to_attachment(cmd, &s->atlas, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        else
            for (t = 0; t < TILES; t++)
                depth_to_attachment(cmd, &s->tiles[t], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->depth_pipeline);
        for (t = 0; t < TILES; t++) {
            if (atlas)
                begin_pass(cmd, NULL, &s->atlas, VK_ATTACHMENT_STORE_OP_STORE, tile_x(t), tile_y(t), TILE, TILE);
            else
                begin_pass(cmd, NULL, &s->tiles[t], VK_ATTACHMENT_STORE_OP_STORE, 0, 0, TILE, TILE);
            vkd.CmdDraw(cmd, 6, CASTERS, 0, (uint32_t)(t * CASTERS));
            vkd.CmdEndRendering(cmd);
        }
        if (atlas)
            to_sampled(cmd, &s->atlas, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        else
            for (t = 0; t < TILES; t++)
                to_sampled(cmd, &s->tiles[t], VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        if (f->verify) {
            if (atlas)
                read_back(s, cmd, &s->atlas, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, tile_x(VERIFY_TILE) - BORDER,
                          tile_y(VERIFY_TILE) - BORDER, TILE + 2 * BORDER, TILE + 2 * BORDER, 0);
            else
                read_back(s, cmd, &s->tiles[VERIFY_TILE], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 0, TILE, TILE,
                          0);
        }
        break;
    }
    case MODE_DYNRES:
    case MODE_DYNRES_EXACT: {
        const struct bench_image *color = s->cfg->mode == MODE_DYNRES ? &s->color : &s->color_exact;
        const struct bench_image *depth = s->cfg->mode == MODE_DYNRES ? &s->depth : &s->depth_exact;
        bench_image_barrier(cmd, color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        depth_to_attachment(cmd, depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->color_pipeline);
        begin_pass(cmd, color, depth, VK_ATTACHMENT_STORE_OP_DONT_CARE, 0, 0, AREA_W, AREA_H);
        vkd.CmdDraw(cmd, 6, OBJECTS, 0, TILES * CASTERS);
        vkd.CmdEndRendering(cmd);
        to_sampled(cmd, color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        if (f->verify) {
            read_back(s, cmd, color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 0, color->width, color->height, 0);
            read_back(s, cmd, depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, 0, 0, depth->width, depth->height,
                      (VkDeviceSize)DYN_W * DYN_H * 8);
        }
        break;
    }
    }
}

/* ---- Verification ---- */

/* IEEE half to float (no NaNs here). */
static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) << 31, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, bits;
    float f;
    if (exp == 0) { /* zero or subnormal */
        f = (float)mant / 16777216.0f;
        return sign ? -f : f;
    }
    bits = sign | ((exp + 112) << 23) | (mant << 13);
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Per-pixel reference of a scene: index of the visible (nearest) object, or -1. */
static int *visible_objects(const struct object *objects, int count, int width, int height)
{
    int *visible = bench_alloc(sizeof(int) * (size_t)width * (size_t)height);
    float *depth = bench_alloc(sizeof(float) * (size_t)width * (size_t)height);
    int i, x, y;
    for (i = 0; i < width * height; i++) {
        visible[i] = -1;
        depth[i] = 1.0f;
    }
    for (i = 0; i < count; i++) {
        const struct bench_rect *r = &objects[i].r;
        for (y = r->y0; y < r->y1; y++)
            for (x = r->x0; x < r->x1; x++)
                if (objects[i].depth < depth[y * width + x]) {
                    depth[y * width + x] = objects[i].depth;
                    visible[y * width + x] = i;
                }
    }
    free(depth);
    return visible;
}

static int verify_tiles(struct scene *s)
{
    const struct config *cfg = s->cfg;
    const float *got = s->readback.map;
    int *visible = visible_objects(s->casters[VERIFY_TILE], CASTERS, TILE, TILE);
    int border = cfg->mode == MODE_ATLAS ? BORDER : 0, size = TILE + 2 * border, x, y, ok = 1;
    for (y = 0; y < size && ok; y++)
        for (x = 0; x < size; x++) {
            int tx = x - border, ty = y - border, inside = tx >= 0 && tx < TILE && ty >= 0 && ty < TILE, v;
            float want = OUTSIDE_DEPTH;
            if (inside) {
                v = visible[ty * TILE + tx];
                want = v < 0 ? 1.0f : s->casters[VERIFY_TILE][v].depth;
            }
            if (got[y * size + x] != want) {
                bench_error(cfg->name, "tile %d, pixel (%d,%d) of the tile: depth %g, want %g%s", VERIFY_TILE, tx, ty,
                            (double)got[y * size + x], (double)want, inside ? "" : " (outside the render area)");
                ok = 0;
                break;
            }
        }
    free(visible);
    return ok;
}

static int verify_dynres(struct scene *s)
{
    const struct config *cfg = s->cfg;
    const struct bench_image *color = cfg->mode == MODE_DYNRES ? &s->color : &s->color_exact;
    int w = (int)color->width, h = (int)color->height, x, y, c, ok = 1;
    const uint16_t *got = s->readback.map;
    const float *depth = (const float *)((const char *)s->readback.map + (size_t)DYN_W * DYN_H * 8);
    int *visible = visible_objects(s->objects, OBJECTS, AREA_W, AREA_H);

    for (y = 0; y < h && ok; y++)
        for (x = 0; x < w && ok; x++) {
            int inside = x < AREA_W && y < AREA_H, v = inside ? visible[y * AREA_W + x] : -1;
            float want[4];
            if (!inside)
                memcpy(want, outside_color, sizeof(want));
            else if (v < 0)
                want[0] = want[1] = want[2] = 0.0f, want[3] = 1.0f;
            else
                bench_unorm4(s->objects[v].r.rgba, want);
            for (c = 0; c < 4; c++) {
                float g = half_to_float(got[((size_t)y * w + x) * 4 + c]);
                /* unorm8 / 255 converted to half: within an ulp (2^-11 below 1),
                 * as the conversion may round toward zero */
                if (g - want[c] > 1.0f / 2048.0f || want[c] - g > 1.0f / 2048.0f) {
                    bench_error(cfg->name, "color at (%d,%d) channel %d: %g, want %g%s", x, y, c, (double)g,
                                (double)want[c], inside ? "" : " (outside the render area)");
                    ok = 0;
                    break;
                }
            }
            if (ok && !inside && depth[(size_t)y * w + x] != OUTSIDE_DEPTH) {
                bench_error(cfg->name, "depth at (%d,%d) outside the render area: %g, want %g", x, y,
                            (double)depth[(size_t)y * w + x], (double)OUTSIDE_DEPTH);
                ok = 0;
            }
        }
    free(visible);
    return ok;
}

static int verify(struct scene *s)
{
    if (s->cfg->mode == MODE_ATLAS || s->cfg->mode == MODE_SEPARATE)
        return verify_tiles(s);
    return verify_dynres(s);
}

/* ---- Setup ---- */

/* Fills img with the outside value and leaves it in layout. */
static void prefill(struct bench_vk *vk, struct bench_image *img, VkImageLayout layout)
{
    VkCommandBuffer cmd = bench_cmd_begin(vk);
    VkImageSubresourceRange range = { img->aspect, 0, 1, 0, 1 };
    bench_image_barrier(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    if (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) {
        VkClearDepthStencilValue value = { OUTSIDE_DEPTH, 0 };
        vkx.CmdClearDepthStencilImage(cmd, img->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, 1, &range);
    } else {
        VkClearColorValue value;
        memcpy(value.float32, outside_color, sizeof(outside_color));
        vkx.CmdClearColorImage(cmd, img->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, 1, &range);
    }
    bench_image_barrier(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layout, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                        VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    bench_cmd_finish(vk, cmd);
}

static void to_instance(const struct object *o, int width, int height, struct instance *inst)
{
    bench_rect_ndc(&o->r, width, height, 0, 0, inst->rect);
    inst->rgba = o->r.rgba;
    inst->depth = o->depth;
}

static void setup(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    const VkImageUsageFlags readable = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const VkImageUsageFlags depth_usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | readable;
    const VkImageUsageFlags color_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | readable;
    const size_t count = TILES * CASTERS + OBJECTS;
    struct instance *inst = bench_alloc(sizeof(*inst) * count);
    struct bench_gfx_desc desc;
    struct bench_gfx_state *state;
    VkShaderModule vs, fs;
    int t, i;

    vkx.CmdClearColorImage = (PFN_vkCmdClearColorImage)vkGetDeviceProcAddr(vk->device, "vkCmdClearColorImage");
    vkx.CmdClearDepthStencilImage =
        (PFN_vkCmdClearDepthStencilImage)vkGetDeviceProcAddr(vk->device, "vkCmdClearDepthStencilImage");
    if (!vkx.CmdClearColorImage || !vkx.CmdClearDepthStencilImage)
        bench_fatal("vkGetDeviceProcAddr returned NULL for the clear commands");

    /* Casters of tile t are instances t * CASTERS..., the scene's follow them. */
    for (t = 0; t < TILES; t++) {
        make_objects(s->casters[t], CASTERS, TILE, TILE, 256, 0x5AD0A71Aull + (uint64_t)t);
        for (i = 0; i < CASTERS; i++)
            to_instance(&s->casters[t][i], TILE, TILE, &inst[t * CASTERS + i]);
    }
    make_objects(s->objects, OBJECTS, AREA_W, AREA_H, 384, 0xD1A3E570ull);
    for (i = 0; i < OBJECTS; i++)
        to_instance(&s->objects[i], AREA_W, AREA_H, &inst[TILES * CASTERS + i]);
    bench_buffer_create(vk, &s->instances, sizeof(*inst) * count,
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->instances, 0, inst, sizeof(*inst) * count);
    free(inst);

    bench_image_create(vk, &s->atlas, VK_FORMAT_D32_SFLOAT, ATLAS, ATLAS, 1, VK_SAMPLE_COUNT_1_BIT, depth_usage, 0);
    prefill(vk, &s->atlas, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (t = 0; t < TILES; t++) {
        bench_image_create(vk, &s->tiles[t], VK_FORMAT_D32_SFLOAT, TILE, TILE, 1, VK_SAMPLE_COUNT_1_BIT, depth_usage,
                           0);
        prefill(vk, &s->tiles[t], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    bench_image_create(vk, &s->color, VK_FORMAT_R16G16B16A16_SFLOAT, DYN_W, DYN_H, 1, VK_SAMPLE_COUNT_1_BIT,
                       color_usage, 0);
    bench_image_create(vk, &s->depth, VK_FORMAT_D32_SFLOAT, DYN_W, DYN_H, 1, VK_SAMPLE_COUNT_1_BIT, depth_usage, 0);
    bench_image_create(vk, &s->color_exact, VK_FORMAT_R16G16B16A16_SFLOAT, AREA_W, AREA_H, 1, VK_SAMPLE_COUNT_1_BIT,
                       color_usage, 0);
    bench_image_create(vk, &s->depth_exact, VK_FORMAT_D32_SFLOAT, AREA_W, AREA_H, 1, VK_SAMPLE_COUNT_1_BIT,
                       depth_usage, 0);
    prefill(vk, &s->color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    prefill(vk, &s->depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    prefill(vk, &s->color_exact, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    prefill(vk, &s->depth_exact, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    bench_buffer_create(vk, &s->readback, (VkDeviceSize)DYN_W * DYN_H * (8 + 4), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        BENCH_HOST, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

    s->layout = bench_pipeline_layout(vk, 0, NULL, 0, 0);
    vs = BENCH_SHADER(vk, rect_depth_vert);
    fs = BENCH_SHADER(vk, rect_flat_frag);
    memset(&desc, 0, sizeof(desc));
    desc.vs = vs;
    desc.layout = s->layout;
    desc.depth_format = VK_FORMAT_D32_SFLOAT;
    desc.depth_test = desc.depth_write = 1;
    desc.depth_compare = VK_COMPARE_OP_LESS;
    desc.binding_count = 1;
    desc.bindings[0] = (VkVertexInputBindingDescription){ 0, sizeof(struct instance), VK_VERTEX_INPUT_RATE_INSTANCE };
    desc.attribute_count = 3;
    desc.attributes[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
    desc.attributes[1] = (VkVertexInputAttributeDescription){ 1, 0, VK_FORMAT_R8G8B8A8_UNORM,
                                                              (uint32_t)offsetof(struct instance, rgba) };
    desc.attributes[2] = (VkVertexInputAttributeDescription){ 2, 0, VK_FORMAT_R32_SFLOAT,
                                                              (uint32_t)offsetof(struct instance, depth) };
    /* Shadow casters: depth only, no fragment shader. */
    state = bench_alloc(sizeof(*state));
    bench_gfx_state_init(state, &desc);
    state->info.stageCount = 1;
    BENCH_VK(vkd.CreateGraphicsPipelines(vk->device, VK_NULL_HANDLE, 1, &state->info, NULL, &s->depth_pipeline));
    free(state);
    desc.fs = fs;
    desc.color_count = 1;
    desc.color_formats[0] = VK_FORMAT_R16G16B16A16_SFLOAT;
    s->color_pipeline = bench_gfx_pipeline(vk, &desc);
    vkd.DestroyShaderModule(vk->device, vs, NULL);
    vkd.DestroyShaderModule(vk->device, fs, NULL);
}

static void teardown(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    int t;
    vkd.DeviceWaitIdle(vk->device);
    vkd.DestroyPipeline(vk->device, s->depth_pipeline, NULL);
    vkd.DestroyPipeline(vk->device, s->color_pipeline, NULL);
    vkd.DestroyPipelineLayout(vk->device, s->layout, NULL);
    bench_image_destroy(vk, &s->atlas);
    for (t = 0; t < TILES; t++)
        bench_image_destroy(vk, &s->tiles[t]);
    bench_image_destroy(vk, &s->color);
    bench_image_destroy(vk, &s->depth);
    bench_image_destroy(vk, &s->color_exact);
    bench_image_destroy(vk, &s->depth_exact);
    bench_buffer_destroy(vk, &s->instances);
    bench_buffer_destroy(vk, &s->readback);
}

int main(void)
{
    struct bench_vk vk;
    struct bench_loop loop;
    struct scene *s;
    size_t c;

    bench_begin("render_area");
    bench_vk_init(&vk, BENCH_NEED_DYNAMIC_RENDERING | BENCH_NEED_SYNC2);
    s = bench_alloc(sizeof(*s));
    s->vk = &vk;
    setup(s);
    bench_loop_init(&loop, &vk, 1000);

    for (c = 0; c < sizeof(configs) / sizeof(configs[0]); c++) {
        const struct config *cfg = &configs[c];
        int frames = bench_frames(cfg->frames), tiles = cfg->mode == MODE_ATLAS || cfg->mode == MODE_SEPARATE;
        struct bench_json j;

        if (!bench_selected(cfg->name))
            continue;
        bench_config = cfg->name;
        s->cfg = cfg;
        bench_loop_verify(&loop, record, s);
        if (!verify(s))
            continue;
        bench_loop_run(&loop, record, s, 30, frames < 1000 ? frames : 1000);

        bench_json_begin(&j, cfg->name);
        switch (cfg->mode) {
        case MODE_ATLAS:
            bench_json_int(&j, "width", ATLAS);
            bench_json_int(&j, "height", ATLAS);
            break;
        case MODE_SEPARATE:
            bench_json_int(&j, "width", TILE);
            bench_json_int(&j, "height", TILE);
            break;
        case MODE_DYNRES:
            bench_json_int(&j, "width", DYN_W);
            bench_json_int(&j, "height", DYN_H);
            break;
        case MODE_DYNRES_EXACT:
            bench_json_int(&j, "width", AREA_W);
            bench_json_int(&j, "height", AREA_H);
            break;
        }
        bench_json_int(&j, "area_width", tiles ? TILE : AREA_W);
        bench_json_int(&j, "area_height", tiles ? TILE : AREA_H);
        bench_json_int(&j, "passes", tiles ? TILES : 1);
        bench_json_int(&j, "objects", tiles ? TILES * CASTERS : OBJECTS);
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
