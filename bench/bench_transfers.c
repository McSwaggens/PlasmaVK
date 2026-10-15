/* Transfers and other non-draw commands, in the patterns engines record them
 * every frame or while loading:
 *
 *   clear3d_128          vkCmdClearColorImage of a 128x128x128 RGBA8 3D image
 *                        (a froxel or voxel volume cleared every frame)
 *   mipgen_2048          vkCmdBlitImage mip chains of 2 2048x2048 RGBA8
 *                        textures, a barrier after each level (sampled images)
 *   mipgen_2048_storage  the same with STORAGE usage on the images (render
 *                        targets that compute passes also write)
 *   mipgen_cube512       the mip chain of a 512x512 RGBA16F cube map, each
 *                        blit all 6 faces (a reflection probe)
 *   mipgen_warp4_512     4 512x512 RGBA8 STORAGE images with 5 levels, level
 *                        by level over all 4, a barrier per level (vkQuake's
 *                        underwater warp textures)
 *   blit_atlas64         64 scaled blits (256x256 to 128x128) into cells of a
 *                        2048x2048 atlas, no barriers between them (sprite,
 *                        glyph and impostor atlases, emulators' scaling)
 *   stream16_512         texture streaming: 16 512x512 sRGB textures, each
 *                        copied from staging into level 0 and its 9 mips
 *                        blitted (the staging data is written once)
 *   fill_x128            128 vkCmdFillBuffer back to back into one 2 MB
 *                        buffer, 16 KB each, alternately 0 and 1.0f (counters,
 *                        indirect arguments, min/max bounds)
 *
 * GPU time is the number to watch. The first frame of each configuration is
 * read back. Clears, copies and fills must be exact; each blitted level must
 * be the 2x2 average of the level above it (as read back) within the
 * rounding of the format: 1 per channel for 8-bit formats (sRGB averaged in
 * linear), 2^-10 relative for half floats.
 */
#include <math.h>

#include "bench_vk.h"

#define BIG 2048          /* mipgen_2048, the atlas */
#define BIG_MIPS 12
#define CUBE 512
#define CUBE_MIPS 10
#define WARP 512
#define WARP_MIPS 5
#define WARPS 4
#define VOLUME 128
#define SPRITE 256
#define CELL 128 /* atlas cells */
#define CELLS ((BIG / CELL) * (BIG / CELL))
#define BLITS 64
#define STREAM 512
#define STREAM_MIPS 10
#define STREAMS 16
#define FILLS 128
#define FILL_BYTES (16u << 10)
#define FILL_FLOAT 0x3F800000u /* 1.0f */

static const float clear3d_color[4] = { 0.2f, 0.4f, 0.6f, 1.0f }; /* 51, 102, 153, 255 */
static const float atlas_clear[4] = { 0.0f, 0.0f, 1.0f, 1.0f };   /* verify frames: cells not blitted */

enum mode { MODE_CLEAR3D, MODE_MIPGEN, MODE_MIPGEN_STORAGE, MODE_CUBE, MODE_WARP, MODE_ATLAS, MODE_STREAM, MODE_FILL };

struct config {
    const char *name;
    enum mode mode;
    int frames;
};

static const struct config configs[] = {
    { "clear3d_128", MODE_CLEAR3D, 100 },
    { "mipgen_2048", MODE_MIPGEN, 200 },
    { "mipgen_2048_storage", MODE_MIPGEN_STORAGE, 200 },
    { "mipgen_cube512", MODE_CUBE, 100 },
    { "mipgen_warp4_512", MODE_WARP, 200 },
    { "blit_atlas64", MODE_ATLAS, 100 },
    { "stream16_512", MODE_STREAM, 60 },
    { "fill_x128", MODE_FILL, 200 },
};

/* Transfer commands that bench_vk.h does not load. */
static struct {
    PFN_vkCmdBlitImage CmdBlitImage;
    PFN_vkCmdClearColorImage CmdClearColorImage;
    PFN_vkCmdFillBuffer CmdFillBuffer;
} vkx;

/* An image with any type, layers and mips; transfers need no views. */
struct img {
    VkImage image;
    VkDeviceMemory memory;
    VkFormat format;
    uint32_t width, height, depth, mips, layers;
};

struct scene {
    struct bench_vk *vk;
    const struct config *cfg;
    struct img volume, mips[2], storage_mips[2], cube, warps[WARPS], atlas, sprite, streams[STREAMS];
    struct bench_buffer staging, fills, readback;
    uint32_t *big_texels, *sprite_texels, *warp_texels, *stream_texels; /* level 0 contents */
    uint16_t *cube_texels;
    uint32_t verify_index;
};

/* ---- Images ---- */

static void img_create(struct bench_vk *vk, struct img *im, VkImageType type, VkFormat format, uint32_t width,
                       uint32_t height, uint32_t depth, uint32_t mips, uint32_t layers, VkImageUsageFlags usage,
                       VkImageCreateFlags flags)
{
    VkImageCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = flags,
        .imageType = type,
        .format = format,
        .extent = { width, height, depth },
        .mipLevels = mips,
        .arrayLayers = layers,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkMemoryRequirements req;

    memset(im, 0, sizeof(*im));
    memset(&req, 0, sizeof(req));
    im->format = format;
    im->width = width;
    im->height = height;
    im->depth = depth;
    im->mips = mips;
    im->layers = layers;
    BENCH_VK(vkd.CreateImage(vk->device, &info, NULL, &im->image));
    vkd.GetImageMemoryRequirements(vk->device, im->image, &req);
    im->memory = bench_allocate(vk, &req, 0, BENCH_DEVICE_LOCAL, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, NULL);
    BENCH_VK(vkd.BindImageMemory(vk->device, im->image, im->memory, 0));
}

static void img_destroy(struct bench_vk *vk, struct img *im)
{
    if (im->image)
        vkd.DestroyImage(vk->device, im->image, NULL);
    if (im->memory)
        vkd.FreeMemory(vk->device, im->memory, NULL);
    memset(im, 0, sizeof(*im));
}

/* Levels [level, level + levels) of every layer. */
static VkImageMemoryBarrier2 img_barrier_info(const struct img *im, uint32_t level, uint32_t levels,
                                              VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 src_stage,
                                              VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                                              VkAccessFlags2 dst_access)
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
        .image = im->image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, level, levels, 0, VK_REMAINING_ARRAY_LAYERS },
    };
    return b;
}

static void img_barrier(VkCommandBuffer cmd, const struct img *im, uint32_t level, uint32_t levels,
                        VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 src_stage,
                        VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access)
{
    VkImageMemoryBarrier2 b =
        img_barrier_info(im, level, levels, from, to, src_stage, src_access, dst_stage, dst_access);
    bench_image_barriers(cmd, 1, &b);
}

static uint32_t level_size(uint32_t size, uint32_t level)
{
    return size >> level ? size >> level : 1;
}

static VkDeviceSize texel_bytes(VkFormat format)
{
    return format == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 4;
}

/* Bytes of one level (all layers) of a 2D image. */
static VkDeviceSize level_bytes(const struct img *im, uint32_t level)
{
    return (VkDeviceSize)level_size(im->width, level) * level_size(im->height, level) * im->layers *
           texel_bytes(im->format);
}

/* Blit of level - 1 into level, all layers, linear filter: 2:1. */
static void blit_level(VkCommandBuffer cmd, const struct img *im, uint32_t level, VkImageLayout src_layout,
                       VkImageLayout dst_layout)
{
    VkImageBlit blit = {
        .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, im->layers },
        .srcOffsets = { { 0, 0, 0 },
                        { (int32_t)level_size(im->width, level - 1), (int32_t)level_size(im->height, level - 1), 1 } },
        .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, im->layers },
        .dstOffsets = { { 0, 0, 0 }, { (int32_t)level_size(im->width, level), (int32_t)level_size(im->height, level), 1 } },
    };
    vkx.CmdBlitImage(cmd, im->image, src_layout, im->image, dst_layout, 1, &blit, VK_FILTER_LINEAR);
}

/* The usual mip generation: level 0 (in from, written by src_stage) is the
 * source; each level is blitted from the one above it, then becomes the
 * next source. The texture ends up sampled. */
static void mipgen(VkCommandBuffer cmd, const struct img *im, VkImageLayout from, VkPipelineStageFlags2 src_stage,
                   VkAccessFlags2 src_access)
{
    uint32_t level;
    img_barrier(cmd, im, 0, 1, from, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, src_stage, src_access,
                VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    img_barrier(cmd, im, 1, im->mips - 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    for (level = 1; level < im->mips; level++) {
        blit_level(cmd, im, level, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        img_barrier(cmd, im, level, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                    VK_ACCESS_2_TRANSFER_READ_BIT);
    }
    img_barrier(cmd, im, 0, im->mips, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

/* Records a copy of levels [0, levels) of im (in layout) into the readback
 * buffer, level after level, each tightly packed. */
static void img_read_back(struct scene *s, VkCommandBuffer cmd, const struct img *im, VkImageLayout layout,
                          uint32_t levels)
{
    VkBufferImageCopy regions[BIG_MIPS];
    VkDeviceSize offset = 0;
    uint32_t level;
    for (level = 0; level < levels; level++) {
        memset(&regions[level], 0, sizeof(regions[level]));
        regions[level].bufferOffset = offset;
        regions[level].imageSubresource = (VkImageSubresourceLayers){ VK_IMAGE_ASPECT_COLOR_BIT, level, 0, im->layers };
        regions[level].imageExtent = (VkExtent3D){ level_size(im->width, level), level_size(im->height, level),
                                                   level_size(im->depth, level) };
        offset += level_bytes(im, level) * level_size(im->depth, level);
    }
    img_barrier(cmd, im, 0, levels, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    vkd.CmdCopyImageToBuffer(cmd, im->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s->readback.buffer, levels, regions);
    img_barrier(cmd, im, 0, levels, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout, VK_PIPELINE_STAGE_2_COPY_BIT, 0,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
}

/* ---- Frames ---- */

static uint32_t atlas_cell(uint32_t frame, uint32_t blit)
{
    return (blit + frame * BLITS) % CELLS;
}

static void record(void *user, const struct bench_frame *f)
{
    struct scene *s = user;
    VkCommandBuffer cmd = f->cmd;
    uint32_t i, level;

    if (f->verify)
        s->verify_index = f->index;
    switch (s->cfg->mode) {
    case MODE_CLEAR3D: {
        VkClearColorValue color;
        VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        memcpy(color.float32, clear3d_color, sizeof(clear3d_color));
        img_barrier(cmd, &s->volume, 0, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT);
        vkx.CmdClearColorImage(cmd, s->volume.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
        img_barrier(cmd, &s->volume, 0, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        if (f->verify)
            img_read_back(s, cmd, &s->volume, VK_IMAGE_LAYOUT_GENERAL, 1);
        break;
    }
    case MODE_MIPGEN:
    case MODE_MIPGEN_STORAGE: {
        struct img *ims = s->cfg->mode == MODE_MIPGEN ? s->mips : s->storage_mips;
        for (i = 0; i < 2; i++)
            mipgen(cmd, &ims[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0);
        if (f->verify)
            img_read_back(s, cmd, &ims[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, BIG_MIPS);
        break;
    }
    case MODE_CUBE:
        mipgen(cmd, &s->cube, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0);
        if (f->verify)
            img_read_back(s, cmd, &s->cube, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, CUBE_MIPS);
        break;
    case MODE_WARP: {
        /* vkQuake's barriers, each one call for all 4 images. There a compute
         * pass has just written level 0 (none here: level 0 keeps its upload),
         * so the chain waits for compute writes; GENERAL throughout. */
        VkImageMemoryBarrier2 b[2 * WARPS];
        for (i = 0; i < WARPS; i++) {
            b[2 * i] = img_barrier_info(&s->warps[i], 0, 1, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                        VK_IMAGE_LAYOUT_GENERAL,
                                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                                        VK_ACCESS_2_TRANSFER_READ_BIT);
            b[2 * i + 1] = img_barrier_info(&s->warps[i], 1, WARP_MIPS - 1, VK_IMAGE_LAYOUT_UNDEFINED,
                                            VK_IMAGE_LAYOUT_GENERAL,
                                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                            0, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        }
        bench_image_barriers(cmd, 2 * WARPS, b);
        for (level = 1; level < WARP_MIPS; level++) {
            for (i = 0; i < WARPS; i++)
                blit_level(cmd, &s->warps[i], level, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
            if (level + 1 < WARP_MIPS)
                bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                     VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        }
        for (i = 0; i < WARPS; i++)
            b[i] = img_barrier_info(&s->warps[i], 0, WARP_MIPS, VK_IMAGE_LAYOUT_GENERAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT,
                                    VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        bench_image_barriers(cmd, WARPS, b);
        if (f->verify)
            img_read_back(s, cmd, &s->warps[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, WARP_MIPS);
        break;
    }
    case MODE_ATLAS:
        img_barrier(cmd, &s->atlas, 0, 1, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                    VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        if (f->verify) { /* known contents outside the cells blitted this frame */
            VkClearColorValue color;
            VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            memcpy(color.float32, atlas_clear, sizeof(atlas_clear));
            vkx.CmdClearColorImage(cmd, s->atlas.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
            bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        }
        for (i = 0; i < BLITS; i++) {
            uint32_t cell = atlas_cell(f->index, i);
            int32_t x = (int32_t)(cell % (BIG / CELL)) * CELL, y = (int32_t)(cell / (BIG / CELL)) * CELL;
            VkImageBlit blit = {
                .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .srcOffsets = { { 0, 0, 0 }, { SPRITE, SPRITE, 1 } },
                .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .dstOffsets = { { x, y, 0 }, { x + CELL, y + CELL, 1 } },
            };
            vkx.CmdBlitImage(cmd, s->sprite.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s->atlas.image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        }
        img_barrier(cmd, &s->atlas, 0, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        if (f->verify)
            img_read_back(s, cmd, &s->atlas, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
        break;
    case MODE_STREAM:
        for (i = 0; i < STREAMS; i++) {
            VkBufferImageCopy region = {
                .bufferOffset = (VkDeviceSize)i * STREAM * STREAM * 4,
                .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .imageExtent = { STREAM, STREAM, 1 },
            };
            img_barrier(cmd, &s->streams[i], 0, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT,
                        VK_ACCESS_2_TRANSFER_WRITE_BIT);
            vkd.CmdCopyBufferToImage(cmd, s->staging.buffer, s->streams[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     1, &region);
            mipgen(cmd, &s->streams[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                   VK_ACCESS_2_TRANSFER_WRITE_BIT);
        }
        if (f->verify)
            img_read_back(s, cmd, &s->streams[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, STREAM_MIPS);
        break;
    case MODE_FILL:
        /* After last frame's fills and the passes that used them. */
        bench_memory_barrier(cmd,
                             VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        for (i = 0; i < FILLS; i++)
            vkx.CmdFillBuffer(cmd, s->fills.buffer, (VkDeviceSize)i * FILL_BYTES, FILL_BYTES, i & 1 ? FILL_FLOAT : 0);
        bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                 VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
        if (f->verify) {
            VkBufferCopy region = { 0, 0, (VkDeviceSize)FILLS * FILL_BYTES };
            bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            vkd.CmdCopyBuffer(cmd, s->fills.buffer, s->readback.buffer, 1, &region);
            bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        }
        break;
    }
}

/* ---- Verification ---- */

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

/* Round to nearest even; normal values and zero only (all this file makes). */
static uint16_t float_to_half(float f)
{
    uint32_t x, sign, mant, h, rest;
    int exp;
    memcpy(&x, &f, sizeof(x));
    sign = (x >> 16) & 0x8000;
    exp = (int)((x >> 23) & 0xFF) - 127 + 15;
    mant = x & 0x7FFFFF;
    if (exp <= 0)
        return (uint16_t)sign;
    h = ((uint32_t)exp << 10) | (mant >> 13);
    rest = mant & 0x1FFF;
    if (rest > 0x1000 || (rest == 0x1000 && (h & 1)))
        h++;
    return (uint16_t)(sign | h);
}

static float srgb_to_linear(float c)
{
    return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
}

static float linear_to_srgb(float l)
{
    return l <= 0.0031308f ? l * 12.92f : 1.055f * powf(l, 1.0f / 2.4f) - 0.055f;
}

/* Channel c of texel i of a level as a float: unorm in [0, 255] (sRGB
 * decoded to linear, in [0, 1]), half floats as they are. */
static float channel(const struct img *im, const void *level, size_t i, int c)
{
    if (im->format == VK_FORMAT_R16G16B16A16_SFLOAT)
        return half_to_float(((const uint16_t *)level)[i * 4 + (size_t)c]);
    if (im->format == VK_FORMAT_R8G8B8A8_SRGB && c < 3)
        return srgb_to_linear((float)((const uint8_t *)level)[i * 4 + (size_t)c] / 255.0f);
    return (float)((const uint8_t *)level)[i * 4 + (size_t)c];
}

/* Whether got (a texel channel as stored) is avg (computed by channel()'s
 * conventions) within the rounding of the format. */
static int channel_close(const struct img *im, float got, float avg, int c)
{
    float d;
    if (im->format == VK_FORMAT_R16G16B16A16_SFLOAT) {
        d = got - avg;
        return (d < 0 ? -d : d) <= (avg < 0 ? -avg : avg) / 1024.0f + 1e-7f;
    }
    if (im->format == VK_FORMAT_R8G8B8A8_SRGB && c < 3) /* compare encoded */
        avg = linear_to_srgb(avg) * 255.0f, got = linear_to_srgb(got) * 255.0f;
    d = got - avg;
    return (d < 0 ? -d : d) <= 1.0f + 1e-3f;
}

/* Each level of a mip chain read back with img_read_back is the 2x2 average
 * of the level above it; level 0 is level0 exactly. */
static int check_chain(const char *config, const struct img *im, const uint8_t *got, const void *level0,
                       uint32_t levels)
{
    const uint8_t *prev = got;
    VkDeviceSize texel = texel_bytes(im->format);
    uint32_t level, layer, x, y;
    int c;

    if (memcmp(got, level0, (size_t)level_bytes(im, 0)) != 0) {
        bench_error(config, "level 0 differs from the uploaded texels");
        return 0;
    }
    for (level = 1; level < levels; level++) {
        const uint8_t *cur = prev + level_bytes(im, level - 1);
        uint32_t w = level_size(im->width, level), h = level_size(im->height, level);
        uint32_t pw = level_size(im->width, level - 1);
        for (layer = 0; layer < im->layers; layer++) {
            const uint8_t *p = prev + (size_t)layer * pw * level_size(im->height, level - 1) * texel;
            const uint8_t *q = cur + (size_t)layer * w * h * texel;
            for (y = 0; y < h; y++)
                for (x = 0; x < w; x++)
                    for (c = 0; c < 4; c++) {
                        size_t i0 = (size_t)(2 * y) * pw + 2 * x, i1 = i0 + pw;
                        float avg = (channel(im, p, i0, c) + channel(im, p, i0 + 1, c) + channel(im, p, i1, c) +
                                     channel(im, p, i1 + 1, c)) / 4.0f;
                        float g = channel(im, q, (size_t)y * w + x, c);
                        if (!channel_close(im, g, avg, c)) {
                            bench_error(config, "level %u layer %u texel (%u,%u) channel %d: %g, want %g", level,
                                        layer, x, y, c, (double)g, (double)avg);
                            return 0;
                        }
                    }
        }
        prev = cur;
    }
    return 1;
}

static int verify(struct scene *s)
{
    const struct config *cfg = s->cfg;
    const uint8_t *got = s->readback.map;
    uint32_t i;

    switch (cfg->mode) {
    case MODE_CLEAR3D: {
        const uint32_t want = bench_rgba8(51, 102, 153, 255);
        for (i = 0; i < VOLUME * VOLUME * VOLUME; i++)
            if (((const uint32_t *)got)[i] != want) {
                bench_error(cfg->name, "texel (%u,%u,%u): 0x%08x, want 0x%08x", i % VOLUME, i / VOLUME % VOLUME,
                            i / (VOLUME * VOLUME), ((const uint32_t *)got)[i], want);
                return 0;
            }
        return 1;
    }
    case MODE_MIPGEN:
        return check_chain(cfg->name, &s->mips[0], got, s->big_texels, BIG_MIPS);
    case MODE_MIPGEN_STORAGE:
        return check_chain(cfg->name, &s->storage_mips[0], got, s->big_texels, BIG_MIPS);
    case MODE_CUBE:
        return check_chain(cfg->name, &s->cube, got, s->cube_texels, CUBE_MIPS);
    case MODE_WARP:
        return check_chain(cfg->name, &s->warps[0], got, s->warp_texels, WARP_MIPS);
    case MODE_STREAM:
        return check_chain(cfg->name, &s->streams[0], got, s->stream_texels, STREAM_MIPS);
    case MODE_ATLAS: {
        /* Cells blitted this frame hold the sprite at half size, the others the clear color. */
        uint8_t blitted[CELLS];
        const uint32_t *atlas = (const uint32_t *)got, clear = bench_rgba8(0, 0, 255, 255);
        uint32_t x, y;
        int c;
        memset(blitted, 0, sizeof(blitted));
        for (i = 0; i < BLITS; i++)
            blitted[atlas_cell(s->verify_index, i)] = 1;
        for (y = 0; y < BIG; y++)
            for (x = 0; x < BIG; x++) {
                uint32_t cell = (y / CELL) * (BIG / CELL) + x / CELL, sx = x % CELL * 2, sy = y % CELL * 2;
                uint32_t texel = atlas[(size_t)y * BIG + x];
                if (!blitted[cell]) {
                    if (texel != clear) {
                        bench_error(cfg->name, "texel (%u,%u) of a cell not blitted: 0x%08x, want 0x%08x", x, y,
                                    texel, clear);
                        return 0;
                    }
                    continue;
                }
                for (c = 0; c < 4; c++) {
                    const uint32_t *sp = s->sprite_texels;
                    int shift = 8 * c;
                    float avg = (float)(((sp[sy * SPRITE + sx] >> shift) & 0xFF) +
                                        ((sp[sy * SPRITE + sx + 1] >> shift) & 0xFF) +
                                        ((sp[(sy + 1) * SPRITE + sx] >> shift) & 0xFF) +
                                        ((sp[(sy + 1) * SPRITE + sx + 1] >> shift) & 0xFF)) / 4.0f;
                    float g = (float)((texel >> shift) & 0xFF), d = g - avg;
                    if ((d < 0 ? -d : d) > 1.0f) {
                        bench_error(cfg->name, "texel (%u,%u) channel %d: %g, want %g", x, y, c, (double)g,
                                    (double)avg);
                        return 0;
                    }
                }
            }
        return 1;
    }
    case MODE_FILL:
        for (i = 0; i < FILLS * FILL_BYTES / 4; i++) {
            uint32_t want = (i / (FILL_BYTES / 4)) & 1 ? FILL_FLOAT : 0;
            if (((const uint32_t *)got)[i] != want) {
                bench_error(cfg->name, "word %u: 0x%08x, want 0x%08x", i, ((const uint32_t *)got)[i], want);
                return 0;
            }
        }
        return 1;
    }
    return 0;
}

/* ---- Setup ---- */

/* Staging: the streamed textures' level 0 data, then room for uploads. */
#define STREAM_BYTES ((VkDeviceSize)STREAMS * STREAM * STREAM * 4)
#define UPLOAD_BYTES ((VkDeviceSize)BIG * BIG * 4)

/* Uploads level 0 (all layers) and leaves every level in layout. */
static void upload(struct scene *s, struct img *im, const void *data, VkImageLayout layout)
{
    struct bench_vk *vk = s->vk;
    VkBufferImageCopy region = {
        .bufferOffset = STREAM_BYTES,
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, im->layers },
        .imageExtent = { im->width, im->height, im->depth },
    };
    VkCommandBuffer cmd;
    memcpy((uint8_t *)s->staging.map + STREAM_BYTES, data, (size_t)level_bytes(im, 0));
    cmd = bench_cmd_begin(vk);
    img_barrier(cmd, im, 0, im->mips, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkd.CmdCopyBufferToImage(cmd, s->staging.buffer, im->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    img_barrier(cmd, im, 0, im->mips, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layout, VK_PIPELINE_STAGE_2_COPY_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
    bench_cmd_finish(vk, cmd);
}

/* Texture-like RGBA8 texels: gradients plus some noise. */
static uint32_t *texels(uint32_t size, uint64_t seed)
{
    uint32_t *t = bench_alloc(sizeof(uint32_t) * size * size), x, y;
    for (y = 0; y < size; y++)
        for (x = 0; x < size; x++) {
            uint32_t n = bench_rand(&seed);
            t[y * size + x] = bench_rgba8((x * 256 / size + (n & 15)) & 255, (y * 256 / size + ((n >> 4) & 15)) & 255,
                                          ((x + y) * 128 / size + ((n >> 8) & 31)) & 255, 255 - ((n >> 16) & 63));
        }
    return t;
}

static void setup(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    const VkImageUsageFlags sampled = VK_IMAGE_USAGE_SAMPLED_BIT;
    uint32_t i, x, y, face;

    vkx.CmdBlitImage = (PFN_vkCmdBlitImage)vkGetDeviceProcAddr(vk->device, "vkCmdBlitImage");
    vkx.CmdClearColorImage = (PFN_vkCmdClearColorImage)vkGetDeviceProcAddr(vk->device, "vkCmdClearColorImage");
    vkx.CmdFillBuffer = (PFN_vkCmdFillBuffer)vkGetDeviceProcAddr(vk->device, "vkCmdFillBuffer");
    if (!vkx.CmdBlitImage || !vkx.CmdClearColorImage || !vkx.CmdFillBuffer)
        bench_fatal("vkGetDeviceProcAddr returned NULL for a transfer command");

    bench_buffer_create(vk, &s->staging, STREAM_BYTES + UPLOAD_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, BENCH_HOST, 0);
    bench_buffer_create(vk, &s->readback, (VkDeviceSize)BIG * BIG * 4 * 4 / 3 + (1u << 20),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT, BENCH_HOST, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    bench_buffer_create(vk, &s->fills, (VkDeviceSize)FILLS * FILL_BYTES,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                        0, BENCH_DEVICE_LOCAL);

    s->big_texels = texels(BIG, 0xB16B16ull);
    s->sprite_texels = texels(SPRITE, 0x5B817Eull);
    s->warp_texels = texels(WARP, 0x3A4Bull);
    s->stream_texels = texels(STREAM, 0x57AEull);
    /* Each streamed texture its own data; the first is stream_texels. */
    for (i = 0; i < STREAMS; i++) {
        uint32_t *t = i ? texels(STREAM, 0x57AEull + i) : s->stream_texels;
        memcpy((uint8_t *)s->staging.map + (size_t)i * STREAM * STREAM * 4, t, (size_t)STREAM * STREAM * 4);
        if (i)
            free(t);
    }
    /* HDR cube faces: values in [0, 4]. */
    s->cube_texels = bench_alloc(sizeof(uint16_t) * 4 * CUBE * CUBE * 6);
    for (face = 0; face < 6; face++)
        for (y = 0; y < CUBE; y++)
            for (x = 0; x < CUBE; x++) {
                uint16_t *t = s->cube_texels + (((size_t)face * CUBE + y) * CUBE + x) * 4;
                t[0] = float_to_half((float)(x + 1) * 4.0f / CUBE);
                t[1] = float_to_half((float)((x * 7 + y * 3) % CUBE + 1) / CUBE);
                t[2] = float_to_half((float)(face + 1) / 6.0f + (float)y / (8.0f * CUBE));
                t[3] = float_to_half(1.0f);
            }

    img_create(vk, &s->volume, VK_IMAGE_TYPE_3D, VK_FORMAT_R8G8B8A8_UNORM, VOLUME, VOLUME, VOLUME, 1, 1,
               VK_IMAGE_USAGE_STORAGE_BIT | sampled, 0);
    for (i = 0; i < 2; i++) {
        img_create(vk, &s->mips[i], VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, BIG, BIG, 1, BIG_MIPS, 1, sampled, 0);
        upload(s, &s->mips[i], s->big_texels, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        img_create(vk, &s->storage_mips[i], VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, BIG, BIG, 1, BIG_MIPS, 1,
                   sampled | VK_IMAGE_USAGE_STORAGE_BIT, 0);
        upload(s, &s->storage_mips[i], s->big_texels, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    img_create(vk, &s->cube, VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, CUBE, CUBE, 1, CUBE_MIPS, 6, sampled,
               VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
    upload(s, &s->cube, s->cube_texels, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (i = 0; i < WARPS; i++) {
        img_create(vk, &s->warps[i], VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, WARP, WARP, 1, WARP_MIPS, 1,
                   sampled | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 0);
        upload(s, &s->warps[i], s->warp_texels, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    img_create(vk, &s->atlas, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, BIG, BIG, 1, 1, 1, sampled, 0);
    upload(s, &s->atlas, s->big_texels, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    img_create(vk, &s->sprite, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, SPRITE, SPRITE, 1, 1, 1, sampled, 0);
    upload(s, &s->sprite, s->sprite_texels, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    for (i = 0; i < STREAMS; i++)
        img_create(vk, &s->streams[i], VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_SRGB, STREAM, STREAM, 1, STREAM_MIPS, 1,
                   sampled, 0);
}

static void teardown(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    uint32_t i;
    vkd.DeviceWaitIdle(vk->device);
    img_destroy(vk, &s->volume);
    for (i = 0; i < 2; i++) {
        img_destroy(vk, &s->mips[i]);
        img_destroy(vk, &s->storage_mips[i]);
    }
    img_destroy(vk, &s->cube);
    for (i = 0; i < WARPS; i++)
        img_destroy(vk, &s->warps[i]);
    img_destroy(vk, &s->atlas);
    img_destroy(vk, &s->sprite);
    for (i = 0; i < STREAMS; i++)
        img_destroy(vk, &s->streams[i]);
    bench_buffer_destroy(vk, &s->staging);
    bench_buffer_destroy(vk, &s->fills);
    bench_buffer_destroy(vk, &s->readback);
    free(s->big_texels);
    free(s->sprite_texels);
    free(s->warp_texels);
    free(s->stream_texels);
    free(s->cube_texels);
}

/* Transfer commands per frame. */
static int commands(const struct config *cfg)
{
    switch (cfg->mode) {
    case MODE_CLEAR3D:
        return 1;
    case MODE_MIPGEN:
    case MODE_MIPGEN_STORAGE:
        return 2 * (BIG_MIPS - 1);
    case MODE_CUBE:
        return CUBE_MIPS - 1;
    case MODE_WARP:
        return WARPS * (WARP_MIPS - 1);
    case MODE_ATLAS:
        return BLITS;
    case MODE_STREAM:
        return STREAMS * STREAM_MIPS; /* a copy and 9 blits each */
    case MODE_FILL:
        return FILLS;
    }
    return 0;
}

int main(void)
{
    struct bench_vk vk;
    struct bench_loop loop;
    struct scene *s;
    size_t c;

    bench_begin("transfers");
    bench_vk_init(&vk, BENCH_NEED_SYNC2);
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
        s->cfg = cfg;
        bench_loop_verify(&loop, record, s);
        if (!verify(s))
            continue;
        bench_loop_run(&loop, record, s, 10, frames < 1000 ? frames : 1000);

        bench_json_begin(&j, cfg->name);
        bench_json_int(&j, "commands", commands(cfg));
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
