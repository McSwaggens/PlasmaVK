/* Chapter "Clear Commands": vkCmdClearColorImage and
 * vkCmdClearDepthStencilImage, checked texel by texel, of whole images and
 * of subresource ranges. */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

#define XFER (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)
#define DST VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
#define SRC VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL

static VkImageSubresourceRange range(VkImageAspectFlags aspect, uint32_t level, uint32_t levels, uint32_t layer,
                                     uint32_t layers)
{
    VkImageSubresourceRange r;
    memset(&r, 0, sizeof(r));
    r.aspectMask = aspect;
    r.baseMipLevel = level;
    r.levelCount = levels;
    r.baseArrayLayer = layer;
    r.layerCount = layers;
    return r;
}

static VkBufferImageCopy region(VkImageAspectFlags aspect, uint32_t level, uint32_t layers, uint32_t w, uint32_t h, uint32_t d)
{
    VkBufferImageCopy r;
    memset(&r, 0, sizeof(r));
    r.imageSubresource.aspectMask = aspect;
    r.imageSubresource.mipLevel = level;
    r.imageSubresource.layerCount = layers;
    r.imageExtent.width = w;
    r.imageExtent.height = h;
    r.imageExtent.depth = d;
    return r;
}

/* The first texel of a run that differs from `texel`, or -1. */
static long differs(const unsigned char *p, size_t count, const void *texel, size_t size)
{
    size_t i;
    for (i = 0; i < count; i++)
        if (memcmp(p + i * size, texel, size))
            return (long)i;
    return -1;
}

CT_TEST(clear_color_image,
        "Clear Commands > Clearing Color Images: every texel of the subresource ranges is set to the clear color, "
        "converted to the format (UNORM: float values, clamped to [0, 1]); texels outside the ranges keep their "
        "values; VK_REMAINING_MIP_LEVELS and VK_REMAINING_ARRAY_LAYERS cover the rest of the image",
        "vkCmdClearColorImage vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateImage vkDestroyImage "
        "vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory")
{
    struct ct_env e;
    VkImageCreateInfo info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 3, 5, XFER);
    struct ct_image m;
    struct ct_buffer out;
    VkImageSubresourceRange all = range(VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS);
    VkImageSubresourceRange part = range(VK_IMAGE_ASPECT_COLOR_BIT, 1, 2, 1, 3);
    VkBufferImageCopy l0 = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 5, 16, 16, 1), l1 = region(VK_IMAGE_ASPECT_COLOR_BIT, 1, 5, 8, 8, 1);
    VkClearColorValue color;
    VkCommandBuffer cb;
    const unsigned char want[4] = { 0xff, 0x40, 0xff, 0x00 }, zero[4] = { 0, 0, 0, 0 };

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    ct_image(t, &e, &info, 0, &m);
    ct_host_buffer(t, &e, 16 * 16 * 5 * 4 + 8 * 8 * 5 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &out);
    l1.bufferOffset = 16 * 16 * 5 * 4;
    cb = ct_begin_commands(t, &e);
    ct_image_barrier(cb, m.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    memset(&color, 0, sizeof(color));
    vkCmdClearColorImage(cb, m.image, DST, &color, 1, &all);
    ct_image_barrier(cb, m.image, VK_IMAGE_ASPECT_COLOR_BIT, DST, DST);
    color.float32[0] = 1.0f;
    color.float32[1] = 0.25f;
    color.float32[2] = 2.0f;
    color.float32[3] = -1.0f;
    vkCmdClearColorImage(cb, m.image, DST, &color, 1, &part);
    ct_image_barrier(cb, m.image, VK_IMAGE_ASPECT_COLOR_BIT, DST, SRC);
    vkCmdCopyImageToBuffer(cb, m.image, SRC, out.buffer, 1, &l0);
    vkCmdCopyImageToBuffer(cb, m.image, SRC, out.buffer, 1, &l1);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &e, cb);
    CT_CHECKF(t, differs(out.data, 5 * 256, zero, 4) < 0, "level 0 was cleared by the second range");
    CT_CHECKF(t, differs(out.data + l1.bufferOffset, 64, zero, 4) < 0 &&
                     differs(out.data + l1.bufferOffset + 4 * 64 * 4, 64, zero, 4) < 0,
              "level 1, layers 0 and 4: changed");
    {
        long d = differs(out.data + l1.bufferOffset + 64 * 4, 3 * 64, want, 4);
        CT_CHECKF(t, d < 0, "level 1, layers 1-3: texel %ld is not (255, 64, 255, 0)", d);
    }
    ct_buffer_fini(&e, &out);
    ct_image_fini(&e, &m);
    ct_env_fini(&e);
}

CT_TEST(clear_color_image_integer,
        "Clear Commands > Clear Values: integer images are cleared to the uint32/int32 values of VkClearColorValue "
        "(values representable in the format, here the full 32-bit range)",
        "vkCmdClearColorImage vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateImage vkDestroyImage "
        "vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory")
{
    struct ct_env e;
    VkImageCreateInfo info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_UINT, 8, 8, 1, 1, 1, XFER);
    struct ct_image u, s;
    struct ct_buffer out;
    VkImageSubresourceRange r = range(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1);
    VkBufferImageCopy copy = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 8, 8, 1);
    VkClearColorValue cu, cs;
    VkCommandBuffer cb;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    ct_image(t, &e, &info, 0, &u);
    info.format = VK_FORMAT_R32G32B32A32_SINT;
    ct_image(t, &e, &info, 0, &s);
    ct_host_buffer(t, &e, 2 * 64 * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &out);
    cu.uint32[0] = 0xffffffffu;
    cu.uint32[1] = 0x80000001u;
    cu.uint32[2] = 0;
    cu.uint32[3] = 123456789u;
    cs.int32[0] = (int32_t)0x80000000u;
    cs.int32[1] = -1;
    cs.int32[2] = 0x7fffffff;
    cs.int32[3] = -123456789;
    cb = ct_begin_commands(t, &e);
    ct_image_barrier(cb, u.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    ct_image_barrier(cb, s.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    vkCmdClearColorImage(cb, u.image, DST, &cu, 1, &r);
    vkCmdClearColorImage(cb, s.image, DST, &cs, 1, &r);
    ct_image_barrier(cb, u.image, VK_IMAGE_ASPECT_COLOR_BIT, DST, SRC);
    ct_image_barrier(cb, s.image, VK_IMAGE_ASPECT_COLOR_BIT, DST, SRC);
    vkCmdCopyImageToBuffer(cb, u.image, SRC, out.buffer, 1, &copy);
    copy.bufferOffset = 64 * 16;
    vkCmdCopyImageToBuffer(cb, s.image, SRC, out.buffer, 1, &copy);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &e, cb);
    CT_CHECK(t, differs(out.data, 64, cu.uint32, 16) < 0);
    CT_CHECK(t, differs(out.data + 64 * 16, 64, cs.int32, 16) < 0);
    ct_buffer_fini(&e, &out);
    ct_image_fini(&e, &u);
    ct_image_fini(&e, &s);
    ct_env_fini(&e);
}

CT_TEST(clear_color_image_3d,
        "Clear Commands > Clearing Color Images: clearing a mip level of a 3D image sets every slice of it "
        "(a 3D image has one array layer); other levels keep their values",
        "vkCmdClearColorImage vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateImage vkDestroyImage "
        "vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory")
{
    struct ct_env e;
    VkImageCreateInfo info = ct_image_info(VK_IMAGE_TYPE_3D, VK_FORMAT_R32_SFLOAT, 8, 8, 8, 2, 1, XFER);
    struct ct_image m;
    struct ct_buffer out;
    VkImageSubresourceRange r0 = range(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1), r1 = range(VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 0, 1);
    VkBufferImageCopy c0 = region(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 8, 8, 8), c1 = region(VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 4, 4, 4);
    VkClearColorValue color;
    VkCommandBuffer cb;
    float a = -2.5f, b = 3.5f;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    ct_image(t, &e, &info, 0, &m);
    ct_host_buffer(t, &e, 512 * 4 + 64 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &out);
    c1.bufferOffset = 512 * 4;
    memset(&color, 0, sizeof(color));
    cb = ct_begin_commands(t, &e);
    ct_image_barrier(cb, m.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    color.float32[0] = a;
    vkCmdClearColorImage(cb, m.image, DST, &color, 1, &r0);
    color.float32[0] = b;
    vkCmdClearColorImage(cb, m.image, DST, &color, 1, &r1);
    ct_image_barrier(cb, m.image, VK_IMAGE_ASPECT_COLOR_BIT, DST, SRC);
    vkCmdCopyImageToBuffer(cb, m.image, SRC, out.buffer, 1, &c0);
    vkCmdCopyImageToBuffer(cb, m.image, SRC, out.buffer, 1, &c1);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &e, cb);
    CT_CHECK(t, differs(out.data, 512, &a, 4) < 0);
    CT_CHECK(t, differs(out.data + 512 * 4, 64, &b, 4) < 0);
    ct_buffer_fini(&e, &out);
    ct_image_fini(&e, &m);
    ct_env_fini(&e);
}

CT_TEST(clear_depth_stencil_image,
        "Clear Commands > Clearing Depth/Stencil Images: the aspects in the ranges are set to the depth and stencil "
        "values, the other aspect and the other layers keep theirs (D16_UNORM: 0.5 is 0x7fff or 0x8000)",
        "vkCmdClearDepthStencilImage vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateImage vkDestroyImage "
        "vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory vkGetPhysicalDeviceFormatProperties "
        "vkGetPhysicalDeviceFormatProperties2")
{
    struct ct_env e;
    VkImageCreateInfo info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_D16_UNORM, 16, 16, 1, 1, 2, XFER);
    VkFormatFeatureFlags2 need = VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
    struct ct_image m;
    struct ct_buffer out;
    VkImageSubresourceRange r;
    VkBufferImageCopy c;
    VkClearDepthStencilValue v;
    VkCommandBuffer cb;
    VkFormat ds = VK_FORMAT_UNDEFINED;
    uint16_t d16;
    long d;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    ct_image(t, &e, &info, 0, &m);
    ct_host_buffer(t, &e, 2 * 256 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &out);
    cb = ct_begin_commands(t, &e);
    ct_image_barrier(cb, m.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST);
    v.depth = 1.0f;
    v.stencil = 0;
    r = range(VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 2);
    vkCmdClearDepthStencilImage(cb, m.image, DST, &v, 1, &r);
    ct_image_barrier(cb, m.image, VK_IMAGE_ASPECT_DEPTH_BIT, DST, DST);
    v.depth = 0.5f;
    r = range(VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 1, 1);
    vkCmdClearDepthStencilImage(cb, m.image, DST, &v, 1, &r);
    ct_image_barrier(cb, m.image, VK_IMAGE_ASPECT_DEPTH_BIT, DST, SRC);
    c = region(VK_IMAGE_ASPECT_DEPTH_BIT, 0, 2, 16, 16, 1);
    vkCmdCopyImageToBuffer(cb, m.image, SRC, out.buffer, 1, &c);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &e, cb);
    d16 = 0xffff;
    CT_CHECKF(t, differs(out.data, 256, &d16, 2) < 0, "layer 0 is not 1.0");
    memcpy(&d16, out.data + 512, 2);
    CT_CHECKF(t, (d16 == 0x7fff || d16 == 0x8000) && differs(out.data + 512, 256, &d16, 2) < 0, "layer 1 is not 0.5");
    ct_image_fini(&e, &m);

    if ((ct_format_features(&e, VK_FORMAT_D24_UNORM_S8_UINT, 0) & need) == need)
        ds = VK_FORMAT_D24_UNORM_S8_UINT;
    if ((ct_format_features(&e, VK_FORMAT_D32_SFLOAT_S8_UINT, 0) & need) == need)
        ds = VK_FORMAT_D32_SFLOAT_S8_UINT;
    if (ds != VK_FORMAT_UNDEFINED) {
        unsigned char s;
        info.format = ds;
        ct_image(t, &e, &info, 0, &m);
        memset(out.data, 0, (size_t)out.size);
        cb = ct_begin_commands(t, &e);
        ct_image_barrier(cb, m.image, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, VK_IMAGE_LAYOUT_UNDEFINED, DST);
        v.depth = 0.25f;
        v.stencil = 0x5a;
        r = range(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 2);
        vkCmdClearDepthStencilImage(cb, m.image, DST, &v, 1, &r);
        ct_image_barrier(cb, m.image, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, DST, DST);
        v.depth = 1.0f; /* depth only, of layer 1 */
        v.stencil = 0x33;
        r = range(VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 1, 1);
        vkCmdClearDepthStencilImage(cb, m.image, DST, &v, 1, &r);
        ct_image_barrier(cb, m.image, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, DST, SRC);
        c = region(VK_IMAGE_ASPECT_STENCIL_BIT, 0, 2, 16, 16, 1);
        vkCmdCopyImageToBuffer(cb, m.image, SRC, out.buffer, 1, &c);
        ct_host_read_barrier(cb);
        ct_submit_and_wait(t, &e, cb);
        s = 0x5a;
        d = differs(out.data, 512, &s, 1);
        CT_CHECKF(t, d < 0, "stencil texel %ld changed by a depth-only clear or not cleared", d);
        ct_image_fini(&e, &m);
    }
    ct_buffer_fini(&e, &out);
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(clear_color_image),
    CT_REF(clear_color_image_integer),
    CT_REF(clear_color_image_3d),
    CT_REF(clear_depth_stencil_image),
};

CT_SUITE(clears, tests)
