/* Chapter "Copy Commands": copies between buffers and images and between
 * images, blits, resolves, and host image copies (Copying Data Between Host
 * Memory and Images), checked texel by texel. Formats and capabilities are
 * the required ones (Formats > Required Format Support) or queried first. */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

#define XFER (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)
#define DST VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
#define SRC VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL

static VkBufferImageCopy region(VkImageAspectFlags aspect, uint32_t level, uint32_t layer, uint32_t layers, int32_t x,
                                int32_t y, int32_t z, uint32_t w, uint32_t h, uint32_t d)
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

/* Uploads `up` from `in` (patterned), reads the same region back tightly
 * packed into `out`, and compares block row by block row. */
static void roundtrip(struct ct *t, struct ct_env *e, const VkImageCreateInfo *info, VkImageAspectFlags aspect,
                      const VkBufferImageCopy *up, uint32_t block, uint32_t bw, uint32_t bh, const char *what)
{
    uint32_t rl = up->bufferRowLength ? up->bufferRowLength : up->imageExtent.width;
    uint32_t ih = up->bufferImageHeight ? up->bufferImageHeight : up->imageExtent.height;
    uint32_t brow = (rl + bw - 1) / bw, bimg = (ih + bh - 1) / bh;
    uint32_t wb = (up->imageExtent.width + bw - 1) / bw, hb = (up->imageExtent.height + bh - 1) / bh;
    uint32_t slices = up->imageSubresource.layerCount > up->imageExtent.depth ? up->imageSubresource.layerCount
                                                                             : up->imageExtent.depth;
    VkDeviceSize in_size = up->bufferOffset + (VkDeviceSize)brow * bimg * slices * block;
    VkBufferImageCopy down = *up;
    struct ct_buffer in, out;
    struct ct_image m;
    VkCommandBuffer cb;
    uint32_t z, y, bad = 0;

    ct_image(t, e, info, 0, &m);
    ct_host_buffer(t, e, in_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &in);
    ct_host_buffer(t, e, 16 + (VkDeviceSize)wb * hb * slices * block, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &out);
    ct_fill_pattern(in.data, (size_t)in_size, (unsigned)info->format);
    down.bufferOffset = 16;
    down.bufferRowLength = down.bufferImageHeight = 0;
    cb = ct_begin_commands(t, e);
    ct_image_barrier(cb, m.image, aspect, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    vkCmdCopyBufferToImage(cb, in.buffer, m.image, DST, 1, up);
    ct_image_barrier(cb, m.image, aspect, DST, SRC);
    vkCmdCopyImageToBuffer(cb, m.image, SRC, out.buffer, 1, &down);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, e, cb);
    for (z = 0; z < slices; z++)
        for (y = 0; y < hb; y++)
            if (memcmp(out.data + 16 + ((VkDeviceSize)z * hb + y) * wb * block,
                       in.data + up->bufferOffset + ((VkDeviceSize)z * bimg + y) * brow * block, (size_t)wb * block))
                bad++;
    CT_CHECKF(t, bad == 0, "%s: %u of %u block rows differ", what, bad, slices * hb);
    ct_buffer_fini(e, &in);
    ct_buffer_fini(e, &out);
    ct_image_fini(e, &m);
}

CT_TEST(copy_buffer_image,
        "Copy Commands > Copying Data Between Buffers and Images: a region (offset, extent, mip level, array "
        "layers) is copied between buffer memory laid out by bufferOffset, bufferRowLength and bufferImageHeight and "
        "the image's texels, both ways, for 1D, 2D array, cube and 3D images",
        "vkCmdCopyBufferToImage vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateImage vkDestroyImage "
        "vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory")
{
    struct ct_env e;
    VkImageCreateInfo info;
    VkBufferImageCopy r;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 37, 21, 1, 4, 3, XFER);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 2, 3, 2, 0, 13, 7, 1);
    r.bufferOffset = 12;
    r.bufferRowLength = 20;
    r.bufferImageHeight = 9;
    roundtrip(t, &e, &info, VK_IMAGE_ASPECT_COLOR_BIT, &r, 4, 1, 1, "2D array, level 1, padded rows");
    info = ct_image_info(VK_IMAGE_TYPE_1D, VK_FORMAT_R16_UINT, 100, 1, 1, 2, 4, XFER);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 3, 5, 0, 0, 90, 1, 1);
    roundtrip(t, &e, &info, VK_IMAGE_ASPECT_COLOR_BIT, &r, 2, 1, 1, "1D array");
    info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_UINT, 8, 8, 1, 1, 6, XFER);
    info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 6, 0, 0, 0, 8, 8, 1);
    roundtrip(t, &e, &info, VK_IMAGE_ASPECT_COLOR_BIT, &r, 16, 1, 1, "cube faces");
    info = ct_image_info(VK_IMAGE_TYPE_3D, VK_FORMAT_R32_SFLOAT, 16, 12, 8, 3, 1, XFER);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 2, 1, 3, 10, 9, 4);
    r.bufferRowLength = 12;
    r.bufferImageHeight = 10;
    roundtrip(t, &e, &info, VK_IMAGE_ASPECT_COLOR_BIT, &r, 4, 1, 1, "3D slices");
    ct_env_fini(&e);
}

CT_TEST(copy_buffer_image_depth_stencil,
        "Copy Commands > Copying Data Between Buffers and Images: depth aspects copy as 2 bytes per texel for "
        "D16_UNORM and 4 for D32_SFLOAT, stencil aspects as 1 byte, the other aspect untouched",
        "vkCmdCopyBufferToImage vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateImage vkDestroyImage "
        "vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory vkGetPhysicalDeviceFormatProperties "
        "vkGetPhysicalDeviceFormatProperties2")
{
    struct ct_env e;
    VkImageCreateInfo info;
    VkBufferImageCopy r;
    VkFormat ds = VK_FORMAT_UNDEFINED;
    VkFormatFeatureFlags2 need = VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
                                 VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    /* D16_UNORM: SAMPLED (so transfers) and DEPTH_STENCIL_ATTACHMENT are required */
    info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_D16_UNORM, 20, 10, 1, 2, 2, XFER);
    r = region(VK_IMAGE_ASPECT_DEPTH_BIT, 1, 0, 2, 0, 0, 0, 10, 5, 1);
    roundtrip(t, &e, &info, VK_IMAGE_ASPECT_DEPTH_BIT, &r, 2, 1, 1, "D16 depth");
    if ((ct_format_features(&e, VK_FORMAT_D24_UNORM_S8_UINT, 0) & need) == need)
        ds = VK_FORMAT_D24_UNORM_S8_UINT;
    if ((ct_format_features(&e, VK_FORMAT_D32_SFLOAT_S8_UINT, 0) & need) == need)
        ds = VK_FORMAT_D32_SFLOAT_S8_UINT;
    if (ds != VK_FORMAT_UNDEFINED) {
        info = ct_image_info(VK_IMAGE_TYPE_2D, ds, 16, 16, 1, 1, 2, XFER);
        r = region(VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 2, 0, 0, 0, 16, 16, 1);
        roundtrip(t, &e, &info, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, &r, 1, 1, 1, "stencil aspect");
    }
    ct_env_fini(&e);
}

/* The first fully supported compressed family's 8- and 16-byte formats. */
static int compressed_formats(const struct ct_env *e, VkFormat *f8, VkFormat *f16, uint32_t *bw, uint32_t *bh)
{
    VkFormatFeatureFlags2 need = VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
    if ((ct_format_features(e, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 0) & need) == need &&
        (ct_format_features(e, VK_FORMAT_BC3_UNORM_BLOCK, 0) & need) == need) {
        *f8 = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        *f16 = VK_FORMAT_BC3_UNORM_BLOCK;
        *bw = *bh = 4;
        return 1;
    }
    if ((ct_format_features(e, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 0) & need) == need &&
        (ct_format_features(e, VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 0) & need) == need) {
        *f8 = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
        *f16 = VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
        *bw = *bh = 4;
        return 1;
    }
    if ((ct_format_features(e, VK_FORMAT_ASTC_4x4_UNORM_BLOCK, 0) & need) == need) {
        *f8 = VK_FORMAT_UNDEFINED;
        *f16 = VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
        *bw = *bh = 4;
        return 1;
    }
    return 0;
}

CT_TEST(copy_buffer_image_compressed,
        "Copy Commands > Copying Data Between Buffers and Images: compressed images copy whole texel blocks, the "
        "row length and image height in texels, and partial blocks only at the image's edge",
        "vkCmdCopyBufferToImage vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateImage vkDestroyImage "
        "vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory vkGetPhysicalDeviceFormatProperties "
        "vkGetPhysicalDeviceFormatProperties2")
{
    struct ct_env e;
    VkImageCreateInfo info;
    VkBufferImageCopy r;
    VkFormat f8, f16;
    uint32_t bw, bh;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    CT_REQUIREF(t, compressed_formats(&e, &f8, &f16, &bw, &bh), "no compressed family with transfers");
    info = ct_image_info(VK_IMAGE_TYPE_2D, f16, 30, 22, 1, 2, 2, XFER);
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 2, 0, 0, 0, 30, 22, 1);
    roundtrip(t, &e, &info, VK_IMAGE_ASPECT_COLOR_BIT, &r, 16, bw, bh, "16-byte blocks, edge blocks");
    r = region(VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 1, 4, 4, 0, 11, 7, 1); /* level 1 is 15x11: up to the edge */
    r.bufferRowLength = 16;
    roundtrip(t, &e, &info, VK_IMAGE_ASPECT_COLOR_BIT, &r, 16, bw, bh, "level 1 to the edge");
    if (f8 != VK_FORMAT_UNDEFINED) {
        info = ct_image_info(VK_IMAGE_TYPE_2D, f8, 16, 16, 1, 1, 1, XFER);
        r = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 4, 8, 0, 8, 8, 1);
        roundtrip(t, &e, &info, VK_IMAGE_ASPECT_COLOR_BIT, &r, 8, bw, bh, "8-byte blocks");
    }
    ct_env_fini(&e);
}

static VkImageCopy image_copy(uint32_t sl, uint32_t slayer, int32_t sx, int32_t sy, int32_t sz, uint32_t dl,
                              uint32_t dlayer, int32_t dx, int32_t dy, int32_t dz, uint32_t layers, uint32_t w, uint32_t h,
                              uint32_t d)
{
    VkImageCopy c;
    memset(&c, 0, sizeof(c));
    c.srcSubresource.aspectMask = c.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
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

/* Fills `a` (all of `up`) from a pattern, copies `c` to `b`, reads `down`
 * of `b`; returns the input in `in` and the result in `out`. */
static void image_to_image(struct ct *t, struct ct_env *e, struct ct_image *a, struct ct_image *b, const VkBufferImageCopy *up,
                           const VkImageCopy *c, const VkBufferImageCopy *down, struct ct_buffer *in, struct ct_buffer *out)
{
    VkCommandBuffer cb = ct_begin_commands(t, e);
    ct_fill_pattern(in->data, (size_t)in->size, 5);
    memset(out->data, 0, (size_t)out->size);
    ct_image_barrier(cb, a->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    ct_image_barrier(cb, b->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    vkCmdCopyBufferToImage(cb, in->buffer, a->image, DST, 1, up);
    ct_image_barrier(cb, a->image, VK_IMAGE_ASPECT_COLOR_BIT, DST, SRC);
    vkCmdCopyImage(cb, a->image, SRC, b->image, DST, 1, c);
    ct_image_barrier(cb, b->image, VK_IMAGE_ASPECT_COLOR_BIT, DST, SRC);
    vkCmdCopyImageToBuffer(cb, b->image, SRC, out->buffer, 1, down);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, e, cb);
}

CT_TEST(copy_image,
        "Copy Commands > Copying Data Between Images: regions of mip levels and array layers are copied between "
        "images, also of size-compatible formats (R32_UINT to R8G8B8A8_UNORM: the same bytes)",
        "vkCmdCopyImage vkCmdCopyBufferToImage vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateImage "
        "vkDestroyImage vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer "
        "vkDestroyBuffer vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory")
{
    struct ct_env e;
    VkImageCreateInfo ia, ib;
    struct ct_image a, b;
    struct ct_buffer in, out;
    VkBufferImageCopy up, down;
    VkImageCopy c;
    uint32_t y, l, bad = 0;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    ia = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32_UINT, 32, 32, 1, 2, 3, XFER);
    ib = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 1, 4, XFER);
    ct_image(t, &e, &ia, 0, &a);
    ct_image(t, &e, &ib, 0, &b);
    ct_host_buffer(t, &e, 16 * 16 * 3 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &in);
    ct_host_buffer(t, &e, 16 * 16 * 2 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &out);
    up = region(VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 3, 0, 0, 0, 16, 16, 1);
    c = image_copy(1, 1, 2, 3, 0, 0, 2, 5, 6, 0, 2, 9, 7, 1);
    down = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 2, 2, 0, 0, 0, 16, 16, 1);
    image_to_image(t, &e, &a, &b, &up, &c, &down, &in, &out);
    for (l = 0; l < 2; l++)
        for (y = 0; y < 7; y++)
            if (memcmp(out.data + ((l * 16 + 6 + y) * 16 + 5) * 4, in.data + (((l + 1) * 16 + 3 + y) * 16 + 2) * 4, 9 * 4))
                bad++;
    CT_CHECKF(t, bad == 0, "%u rows differ", bad);
    ct_buffer_fini(&e, &in);
    ct_buffer_fini(&e, &out);
    ct_image_fini(&e, &a);
    ct_image_fini(&e, &b);
    ct_env_fini(&e);
}

CT_TEST(copy_image_compressed,
        "Copy Commands > Copying Data Between Images: between a compressed and an uncompressed format of the same "
        "texel block size, each block copies to one texel, the extent in source texels",
        "vkCmdCopyImage vkCmdCopyBufferToImage vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateImage "
        "vkDestroyImage vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer "
        "vkDestroyBuffer vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory vkGetPhysicalDeviceFormatProperties "
        "vkGetPhysicalDeviceFormatProperties2")
{
    struct ct_env e;
    VkImageCreateInfo ia, ib;
    struct ct_image a, b;
    struct ct_buffer in, out;
    VkBufferImageCopy up, down;
    VkImageCopy c;
    VkFormat f8, f16;
    uint32_t bw, bh, y, bad = 0;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    CT_REQUIREF(t, compressed_formats(&e, &f8, &f16, &bw, &bh), "no compressed family with transfers");
    /* 16-byte blocks <-> R32G32B32A32_UINT (32 x 32 texels: 8 x 8 blocks) */
    ia = ct_image_info(VK_IMAGE_TYPE_2D, f16, 32, 32, 1, 1, 1, XFER);
    ib = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_UINT, 8, 8, 1, 1, 1, XFER);
    ct_image(t, &e, &ia, 0, &a);
    ct_image(t, &e, &ib, 0, &b);
    ct_host_buffer(t, &e, 8 * 8 * 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &in);
    ct_host_buffer(t, &e, 8 * 8 * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &out);
    up = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 32, 32, 1);
    c = image_copy(0, 0, 8, 4, 0, 0, 0, 1, 2, 0, 1, 16, 12, 1); /* blocks (2,1) + 4x3 -> texels (1,2) */
    down = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 8, 8, 1);
    image_to_image(t, &e, &a, &b, &up, &c, &down, &in, &out);
    for (y = 0; y < 3; y++)
        if (memcmp(out.data + ((2 + y) * 8 + 1) * 16, in.data + ((1 + y) * 8 + 2) * 16, 4 * 16))
            bad++;
    CT_CHECKF(t, bad == 0, "compressed -> uncompressed: %u rows differ", bad);
    ct_image_fini(&e, &a);
    ct_image_fini(&e, &b);
    /* and uncompressed -> compressed: texels (1,2) + 4x3 -> blocks (1,1) */
    ia = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_UINT, 8, 8, 1, 1, 1, XFER);
    ib = ct_image_info(VK_IMAGE_TYPE_2D, f16, 32, 32, 1, 1, 1, XFER);
    ct_image(t, &e, &ia, 0, &a);
    ct_image(t, &e, &ib, 0, &b);
    up = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 8, 8, 1);
    c = image_copy(0, 0, 1, 2, 0, 0, 0, 4, 4, 0, 1, 4, 3, 1);
    down = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 32, 32, 1);
    image_to_image(t, &e, &a, &b, &up, &c, &down, &in, &out);
    for (bad = 0, y = 0; y < 3; y++)
        if (memcmp(out.data + ((1 + y) * 8 + 1) * 16, in.data + ((2 + y) * 8 + 1) * 16, 4 * 16))
            bad++;
    CT_CHECKF(t, bad == 0, "uncompressed -> compressed: %u rows differ", bad);
    ct_buffer_fini(&e, &in);
    ct_buffer_fini(&e, &out);
    ct_image_fini(&e, &a);
    ct_image_fini(&e, &b);
    ct_env_fini(&e);
}

CT_TEST(copy_image_3d_layers,
        "Copy Commands > Copying Data Between Images: with Vulkan 1.1, slices of a 3D image copy to array layers "
        "of a 2D image and back, the depth extent matching the layer count",
        "vkCmdCopyImage vkCmdCopyBufferToImage vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateImage "
        "vkDestroyImage vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer "
        "vkDestroyBuffer vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory")
{
    struct ct_env e;
    VkImageCreateInfo ia, ib;
    struct ct_image a, b;
    struct ct_buffer in, out;
    VkBufferImageCopy up, down;
    VkImageCopy c;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_1)
        ct_skip(t, "3D <-> 2D array copies need Vulkan 1.1 (or VK_KHR_maintenance1)");
    ct_env_device(t, &e, NULL);
    ia = ct_image_info(VK_IMAGE_TYPE_3D, VK_FORMAT_R16_UINT, 16, 16, 8, 1, 1, XFER);
    ib = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R16_UINT, 16, 16, 1, 1, 6, XFER);
    ct_image(t, &e, &ia, 0, &a);
    ct_image(t, &e, &ib, 0, &b);
    ct_host_buffer(t, &e, 16 * 16 * 8 * 2, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &in);
    ct_host_buffer(t, &e, 16 * 16 * 4 * 2, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &out);
    up = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 16, 16, 8);
    c = image_copy(0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 16, 16, 4);
    c.dstSubresource.layerCount = 4; /* slices 2..5 -> layers 1..4 */
    down = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 4, 0, 0, 0, 16, 16, 1);
    image_to_image(t, &e, &a, &b, &up, &c, &down, &in, &out);
    CT_CHECK(t, memcmp(out.data, in.data + 2 * 512, 4 * 512) == 0);
    ct_buffer_fini(&e, &in);
    ct_buffer_fini(&e, &out);
    ct_image_fini(&e, &a);
    ct_image_fini(&e, &b);
    ct_env_fini(&e);
}

static VkImageBlit blit_region(uint32_t sl, int32_t sx0, int32_t sy0, int32_t sx1, int32_t sy1, uint32_t dl, int32_t dx0,
                               int32_t dy0, int32_t dx1, int32_t dy1)
{
    VkImageBlit b;
    memset(&b, 0, sizeof(b));
    b.srcSubresource.aspectMask = b.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.srcSubresource.mipLevel = sl;
    b.srcSubresource.layerCount = b.dstSubresource.layerCount = 1;
    b.dstSubresource.mipLevel = dl;
    b.srcOffsets[0].x = sx0;
    b.srcOffsets[0].y = sy0;
    b.srcOffsets[1].x = sx1;
    b.srcOffsets[1].y = sy1;
    b.srcOffsets[1].z = 1;
    b.dstOffsets[0].x = dx0;
    b.dstOffsets[0].y = dy0;
    b.dstOffsets[1].x = dx1;
    b.dstOffsets[1].y = dy1;
    b.dstOffsets[1].z = 1;
    return b;
}

CT_TEST(blit_image,
        "Copy Commands > Image Copies With Scaling: nearest filtering takes the texel at floor(u) of "
        "u = srcOffset + (x + 0.5 - dstOffset) * scale (for a 2x reduction the odd texels), reversed offsets mirror, "
        "linear filtering of a 2x reduction averages 2x2 texels, integer formats copy exactly",
        "vkCmdBlitImage vkCmdCopyBufferToImage vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateImage "
        "vkDestroyImage vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer "
        "vkDestroyBuffer vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory")
{
    struct ct_env e;
    VkImageCreateInfo info;
    struct ct_image a, b;
    struct ct_buffer in, out;
    VkBufferImageCopy up, down;
    VkImageBlit bl[3];
    VkCommandBuffer cb;
    uint32_t x, y, c;
    int bad = 0;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    /* R8G8B8A8_UNORM: BLIT_SRC, BLIT_DST and SAMPLED_IMAGE_FILTER_LINEAR are required */
    info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 1, 1, XFER);
    ct_image(t, &e, &info, 0, &a);
    info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 8, 8, 1, 1, 1, XFER);
    ct_image(t, &e, &info, 0, &b);
    ct_host_buffer(t, &e, 4 * 4 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &in);
    ct_host_buffer(t, &e, 8 * 8 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &out);
    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++)
            for (c = 0; c < 4; c++)
                in.data[(y * 4 + x) * 4 + c] = (unsigned char)(16 * (y * 4 + x) + 4 * c);
    up = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 4, 4, 1);
    down = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 8, 8, 1);
    bl[0] = blit_region(0, 0, 0, 4, 4, 0, 0, 0, 2, 2); /* 2x down, nearest */
    bl[1] = blit_region(0, 0, 0, 4, 4, 0, 4, 0, 6, 2); /* 2x down, linear */
    bl[2] = blit_region(0, 0, 0, 4, 4, 0, 8, 2, 0, 6); /* 2x up horizontally, mirrored, nearest */
    cb = ct_begin_commands(t, &e);
    ct_image_barrier(cb, a.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    ct_image_barrier(cb, b.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    vkCmdCopyBufferToImage(cb, in.buffer, a.image, DST, 1, &up);
    ct_image_barrier(cb, a.image, VK_IMAGE_ASPECT_COLOR_BIT, DST, SRC);
    vkCmdBlitImage(cb, a.image, SRC, b.image, DST, 1, &bl[0], VK_FILTER_NEAREST);
    vkCmdBlitImage(cb, a.image, SRC, b.image, DST, 1, &bl[1], VK_FILTER_LINEAR);
    vkCmdBlitImage(cb, a.image, SRC, b.image, DST, 1, &bl[2], VK_FILTER_NEAREST);
    ct_image_barrier(cb, b.image, VK_IMAGE_ASPECT_COLOR_BIT, DST, SRC);
    vkCmdCopyImageToBuffer(cb, b.image, SRC, out.buffer, 1, &down);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &e, cb);
    for (y = 0; y < 2; y++)
        for (x = 0; x < 2; x++) {
            if (memcmp(out.data + (y * 8 + x) * 4, in.data + ((2 * y + 1) * 4 + 2 * x + 1) * 4, 4))
                bad |= 1;
            for (c = 0; c < 4; c++) {
                int sum = in.data[((2 * y) * 4 + 2 * x) * 4 + c] + in.data[((2 * y) * 4 + 2 * x + 1) * 4 + c] +
                          in.data[((2 * y + 1) * 4 + 2 * x) * 4 + c] + in.data[((2 * y + 1) * 4 + 2 * x + 1) * 4 + c];
                int got = out.data[(y * 8 + 4 + x) * 4 + c];
                if (got - sum / 4 > 1 || sum / 4 - got > 1)
                    bad |= 2;
            }
        }
    for (y = 2; y < 6; y++)
        for (x = 0; x < 8; x++)
            if (memcmp(out.data + (y * 8 + x) * 4, in.data + ((y - 2) * 4 + (7 - x) / 2) * 4, 4))
                bad |= 4;
    CT_CHECKF(t, !(bad & 1), "nearest reduction");
    CT_CHECKF(t, !(bad & 2), "linear reduction");
    CT_CHECKF(t, !(bad & 4), "mirrored magnification");
    ct_buffer_fini(&e, &in);
    ct_buffer_fini(&e, &out);
    ct_image_fini(&e, &a);
    ct_image_fini(&e, &b);
    ct_env_fini(&e);
}

CT_TEST(blit_image_integer,
        "Copy Commands > Image Copies With Scaling: integer formats blit with nearest filtering to integer formats "
        "of the same signedness, values unchanged (R32G32B32A32_UINT has BLIT_SRC and BLIT_DST)",
        "vkCmdBlitImage vkCmdCopyBufferToImage vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateImage "
        "vkDestroyImage vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer "
        "vkDestroyBuffer vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory")
{
    struct ct_env e;
    VkImageCreateInfo info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_UINT, 2, 2, 1, 1, 1, XFER);
    struct ct_image a, b;
    struct ct_buffer in, out;
    VkBufferImageCopy up, down;
    VkImageBlit bl = blit_region(0, 0, 0, 2, 2, 0, 0, 0, 4, 4);
    VkCommandBuffer cb;
    uint32_t i;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    ct_image(t, &e, &info, 0, &a);
    info.extent.width = info.extent.height = 4;
    ct_image(t, &e, &info, 0, &b);
    ct_host_buffer(t, &e, 2 * 2 * 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &in);
    ct_host_buffer(t, &e, 4 * 4 * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &out);
    for (i = 0; i < 16; i++) {
        uint32_t v = 0xF0000000u + i * 0x01010101u;
        memcpy(in.data + i * 4, &v, 4);
    }
    up = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 2, 2, 1);
    down = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0, 0, 0, 4, 4, 1);
    cb = ct_begin_commands(t, &e);
    ct_image_barrier(cb, a.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    ct_image_barrier(cb, b.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    vkCmdCopyBufferToImage(cb, in.buffer, a.image, DST, 1, &up);
    ct_image_barrier(cb, a.image, VK_IMAGE_ASPECT_COLOR_BIT, DST, SRC);
    vkCmdBlitImage(cb, a.image, SRC, b.image, DST, 1, &bl, VK_FILTER_NEAREST);
    ct_image_barrier(cb, b.image, VK_IMAGE_ASPECT_COLOR_BIT, DST, SRC);
    vkCmdCopyImageToBuffer(cb, b.image, SRC, out.buffer, 1, &down);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &e, cb);
    for (i = 0; i < 16; i++) {
        uint32_t x = i % 4, y = i / 4;
        CT_CHECKF(t, memcmp(out.data + i * 16, in.data + ((y / 2) * 2 + x / 2) * 16, 16) == 0, "texel %u", i);
    }
    ct_buffer_fini(&e, &in);
    ct_buffer_fini(&e, &out);
    ct_image_fini(&e, &a);
    ct_image_fini(&e, &b);
    ct_env_fini(&e);
}

CT_TEST(resolve_image,
        "Copy Commands > Resolving Multisample Images: a region of a multisampled image resolves to a single-sample "
        "image at another offset and layer; samples of equal value resolve to that value",
        "vkCmdResolveImage vkCmdClearColorImage vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateImage "
        "vkDestroyImage vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer "
        "vkDestroyBuffer vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory "
        "vkGetPhysicalDeviceImageFormatProperties")
{
    struct ct_env e;
    VkImageCreateInfo info;
    VkImageFormatProperties p;
    struct ct_image ms, single;
    struct ct_buffer out;
    VkImageSubresourceRange range;
    VkClearColorValue color;
    VkImageResolve res;
    VkBufferImageCopy down;
    VkCommandBuffer cb;
    uint32_t x, y, l;
    const unsigned char want[4] = { 0xff, 0x00, 0x00, 0xff }, zero[4] = { 0, 0, 0, 0 };

    ct_env_init(t, &e);
    if (vkGetPhysicalDeviceImageFormatProperties(e.pd, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | XFER, 0, &p) != VK_SUCCESS ||
        !(p.sampleCounts & VK_SAMPLE_COUNT_4_BIT))
        ct_skip(t, "no 4-sample R8G8B8A8_UNORM images");
    ct_env_device(t, &e, NULL);
    info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 1, 1,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | XFER);
    info.samples = VK_SAMPLE_COUNT_4_BIT;
    ct_image(t, &e, &info, 0, &ms);
    info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 1, 2, XFER);
    ct_image(t, &e, &info, 0, &single);
    ct_host_buffer(t, &e, 16 * 16 * 2 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &out);
    memset(&range, 0, sizeof(range));
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = VK_REMAINING_ARRAY_LAYERS;
    memset(&res, 0, sizeof(res));
    res.srcSubresource.aspectMask = res.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    res.srcSubresource.layerCount = res.dstSubresource.layerCount = 1;
    res.dstSubresource.baseArrayLayer = 1;
    res.srcOffset.x = 2;
    res.srcOffset.y = 3;
    res.dstOffset.x = 8;
    res.dstOffset.y = 4;
    res.extent.width = 6;
    res.extent.height = 5;
    res.extent.depth = 1;
    down = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 2, 0, 0, 0, 16, 16, 1);
    cb = ct_begin_commands(t, &e);
    ct_image_barrier(cb, ms.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    ct_image_barrier(cb, single.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    memset(&color, 0, sizeof(color));
    vkCmdClearColorImage(cb, single.image, DST, &color, 1, &range);
    color.float32[0] = 1.0f;
    color.float32[3] = 1.0f;
    vkCmdClearColorImage(cb, ms.image, DST, &color, 1, &range);
    ct_image_barrier(cb, ms.image, VK_IMAGE_ASPECT_COLOR_BIT, DST, SRC);
    vkCmdResolveImage(cb, ms.image, SRC, single.image, DST, 1, &res);
    ct_image_barrier(cb, single.image, VK_IMAGE_ASPECT_COLOR_BIT, DST, SRC);
    vkCmdCopyImageToBuffer(cb, single.image, SRC, out.buffer, 1, &down);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &e, cb);
    for (l = 0; l < 2; l++)
        for (y = 0; y < 16; y++)
            for (x = 0; x < 16; x++) {
                int inside = l == 1 && x >= 8 && x < 14 && y >= 4 && y < 9;
                if (memcmp(out.data + ((l * 16 + y) * 16 + x) * 4, inside ? want : zero, 4)) {
                    CT_CHECKF(t, 0, "layer %u texel (%u, %u)", l, x, y);
                    l = 2;
                    y = 16;
                    break;
                }
            }
    ct_buffer_fini(&e, &out);
    ct_image_fini(&e, &ms);
    ct_image_fini(&e, &single);
    ct_env_fini(&e);
}

CT_TEST(host_image_copy,
        "Copy Commands > Copying Data Between Host Memory and Images: with hostImageCopy, vkCopyMemoryToImage and "
        "vkCopyImageToMemory copy a region laid out by memoryRowLength and memoryImageHeight; "
        "VK_HOST_IMAGE_COPY_MEMCPY copies whole subresources of VkSubresourceHostMemcpySize bytes; "
        "vkCopyImageToImage copies between images; device copies see the data",
        "vkTransitionImageLayout vkCopyMemoryToImage vkCopyImageToMemory vkCopyImageToImage "
        "vkGetImageSubresourceLayout2 vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateImage vkDestroyImage "
        "vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory vkGetPhysicalDeviceFeatures2 "
        "vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceImageFormatProperties")
{
    struct ct_env e;
    struct ct_features f;
    VkPhysicalDeviceVulkan14Features v14;
    VkPhysicalDeviceHostImageCopyProperties hp;
    VkPhysicalDeviceProperties2 p2;
    VkImageLayout src_layouts[64], dst_layouts[64];
    VkImageFormatProperties ip;
    VkImageCreateInfo info;
    VkHostImageLayoutTransitionInfo tr[2];
    VkMemoryToImageCopy up;
    VkImageToMemoryCopy down;
    VkCopyMemoryToImageInfo ui;
    VkCopyImageToMemoryInfo di;
    VkCopyImageToImageInfo ii;
    VkImageCopy2 c;
    VkImageSubresource2 sub;
    VkSubresourceHostMemcpySize size;
    VkSubresourceLayout2 layout;
    struct ct_image a, b;
    struct ct_buffer gpu;
    VkBufferImageCopy r;
    VkCommandBuffer cb;
    unsigned char *in, *out;
    uint32_t i, l, y, bad = 0;
    int general_src = 0, general_dst = 0;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_4)
        ct_skip(t, "host image copies are Vulkan 1.4");
    ct_get_features(&e, &f, 0);
    if (!f.v14.hostImageCopy)
        ct_skip(t, "hostImageCopy is not supported");
    memset(&hp, 0, sizeof(hp));
    hp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES;
    hp.copySrcLayoutCount = 64;
    hp.pCopySrcLayouts = src_layouts;
    hp.copyDstLayoutCount = 64;
    hp.pCopyDstLayouts = dst_layouts;
    memset(&p2, 0, sizeof(p2));
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &hp;
    vkGetPhysicalDeviceProperties2(e.pd, &p2);
    for (i = 0; i < hp.copySrcLayoutCount && i < 64; i++)
        general_src |= src_layouts[i] == VK_IMAGE_LAYOUT_GENERAL;
    for (i = 0; i < hp.copyDstLayoutCount && i < 64; i++)
        general_dst |= dst_layouts[i] == VK_IMAGE_LAYOUT_GENERAL;
    if (!general_src || !general_dst)
        ct_skip(t, "VK_IMAGE_LAYOUT_GENERAL is not a host copy layout");
    info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 32, 32, 1, 3, 3,
                         VK_IMAGE_USAGE_HOST_TRANSFER_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | XFER);
    if (vkGetPhysicalDeviceImageFormatProperties(e.pd, info.format, info.imageType, info.tiling, info.usage, 0, &ip) !=
        VK_SUCCESS)
        ct_skip(t, "no R8G8B8A8_UNORM images with HOST_TRANSFER usage");
    memset(&v14, 0, sizeof(v14));
    v14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    v14.hostImageCopy = VK_TRUE;
    ct_env_device(t, &e, &v14);
    in = malloc(65536);
    out = malloc(65536);
    CT_REQUIRE(t, in && out);
    ct_fill_pattern(in, 65536, 9);
    ct_image(t, &e, &info, 0, &a);
    ct_image(t, &e, &info, 0, &b);
    memset(tr, 0, sizeof(tr));
    for (i = 0; i < 2; i++) {
        tr[i].sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO;
        tr[i].image = i ? b.image : a.image;
        tr[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        tr[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        tr[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        tr[i].subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        tr[i].subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
    }
    CT_REQUIRE_VK(t, vkTransitionImageLayout(e.device, 2, tr));

    memset(&up, 0, sizeof(up));
    up.sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY;
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
    memset(&ui, 0, sizeof(ui));
    ui.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO;
    ui.dstImage = a.image;
    ui.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ui.regionCount = 1;
    ui.pRegions = &up;
    CT_REQUIRE_VK(t, vkCopyMemoryToImage(e.device, &ui));
    memset(&down, 0, sizeof(down));
    down.sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY;
    down.pHostPointer = out;
    down.imageSubresource = up.imageSubresource;
    down.imageOffset = up.imageOffset;
    down.imageExtent = up.imageExtent;
    memset(&di, 0, sizeof(di));
    di.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO;
    di.srcImage = a.image;
    di.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
    di.regionCount = 1;
    di.pRegions = &down;
    CT_REQUIRE_VK(t, vkCopyImageToMemory(e.device, &di));
    for (l = 0; l < 2; l++)
        for (y = 0; y < 7; y++)
            if (memcmp(out + ((l * 7 + y) * 10) * 4, in + ((l * 8 + y) * 12) * 4, 40))
                bad++;
    CT_CHECKF(t, bad == 0, "memory -> image -> memory: %u rows differ", bad);

    /* image to image */
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
    CT_REQUIRE_VK(t, vkCopyImageToImage(e.device, &ii));
    down.imageSubresource = c.dstSubresource;
    down.imageOffset = c.dstOffset;
    di.srcImage = b.image;
    memset(out, 0, 65536);
    CT_REQUIRE_VK(t, vkCopyImageToMemory(e.device, &di));
    for (bad = 0, l = 0; l < 2; l++)
        for (y = 0; y < 7; y++)
            if (memcmp(out + ((l * 7 + y) * 10) * 4, in + ((l * 8 + y) * 12) * 4, 40))
                bad++;
    CT_CHECKF(t, bad == 0, "image -> image: %u rows differ", bad);

    /* MEMCPY of a whole subresource, back and forth */
    memset(&sub, 0, sizeof(sub));
    sub.sType = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2;
    sub.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    sub.imageSubresource.mipLevel = 2;
    sub.imageSubresource.arrayLayer = 0;
    memset(&size, 0, sizeof(size));
    size.sType = VK_STRUCTURE_TYPE_SUBRESOURCE_HOST_MEMCPY_SIZE;
    memset(&layout, 0, sizeof(layout));
    layout.sType = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2;
    layout.pNext = &size;
    vkGetImageSubresourceLayout2(e.device, a.image, &sub, &layout);
    CT_CHECKF(t, size.size > 0 && size.size <= 65536, "memcpy size %llu", (unsigned long long)size.size);
    if (size.size > 0 && size.size <= 65536) {
        memset(&up, 0, sizeof(up));
        up.sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY;
        up.pHostPointer = in;
        up.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        up.imageSubresource.mipLevel = 2;
        up.imageSubresource.layerCount = 1;
        up.imageExtent.width = up.imageExtent.height = 8;
        up.imageExtent.depth = 1;
        ui.flags = VK_HOST_IMAGE_COPY_MEMCPY_BIT;
        CT_REQUIRE_VK(t, vkCopyMemoryToImage(e.device, &ui));
        down.imageSubresource = up.imageSubresource;
        down.imageOffset = up.imageOffset;
        down.imageExtent = up.imageExtent;
        di.srcImage = a.image;
        di.flags = VK_HOST_IMAGE_COPY_MEMCPY_BIT;
        memset(out, 0, 65536);
        CT_REQUIRE_VK(t, vkCopyImageToMemory(e.device, &di));
        CT_CHECK(t, memcmp(in, out, (size_t)size.size) == 0);
    }

    /* the device sees texels the host wrote: level 1, layer 2 of a */
    ct_host_buffer(t, &e, 10 * 7 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &gpu);
    memset(&r, 0, sizeof(r));
    r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    r.imageSubresource.mipLevel = 1;
    r.imageSubresource.baseArrayLayer = 2;
    r.imageSubresource.layerCount = 1;
    r.imageOffset.x = 3;
    r.imageOffset.y = 5;
    r.imageExtent.width = 10;
    r.imageExtent.height = 7;
    r.imageExtent.depth = 1;
    cb = ct_begin_commands(t, &e);
    ct_image_barrier(cb, a.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, SRC);
    vkCmdCopyImageToBuffer(cb, a.image, SRC, gpu.buffer, 1, &r);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &e, cb);
    for (bad = 0, y = 0; y < 7; y++)
        if (memcmp(gpu.data + y * 40, in + ((8 + y) * 12) * 4, 40))
            bad++;
    CT_CHECKF(t, bad == 0, "device copy of host-written texels: %u rows differ", bad);
    ct_buffer_fini(&e, &gpu);
    ct_image_fini(&e, &a);
    ct_image_fini(&e, &b);
    free(in);
    free(out);
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(copy_buffer_image),    CT_REF(copy_buffer_image_depth_stencil), CT_REF(copy_buffer_image_compressed),
    CT_REF(copy_image),           CT_REF(copy_image_compressed),           CT_REF(copy_image_3d_layers),
    CT_REF(blit_image),           CT_REF(blit_image_integer),              CT_REF(resolve_image),
    CT_REF(host_image_copy),
};

CT_SUITE(copies, tests)
