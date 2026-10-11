/* Chapter "Clear Commands": vkCmdClearColorImage and
 * vkCmdClearDepthStencilImage, checked texel by texel, of whole images and
 * of subresource ranges; vkCmdClearAttachments; vkCmdUpdateBuffer. */
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

static uint32_t expect_cleared_rect(uint32_t x, uint32_t y, const void *ctx)
{
    (void)ctx;
    return x >= 2 && x < 6 && y >= 2 && y < 6 ? CT_RGBA(255, 0, 0, 255) : CT_RGBA(0, 0, 255, 255);
}

static uint16_t expect_depth_rect(uint32_t x, uint32_t y)
{
    return x >= 8 && x < 12 && y >= 8 && y < 12 ? 16384 : 65535; /* 0.25 in the rectangle, else 1 */
}

CT_TEST(clear_attachments,
        "Clear Commands > Clearing Attachments: inside a render pass instance, vkCmdClearAttachments sets the "
        "pixels of the rectangles, in the layers baseArrayLayer .. baseArrayLayer + layerCount - 1, of the "
        "named color attachment and of the depth attachment to the clear values; other pixels and layers keep "
        "their values (a layered framebuffer of 2 layers)",
        CT_RENDER_COMMANDS "vkCreateRenderPass vkDestroyRenderPass vkCreateFramebuffer vkDestroyFramebuffer "
                           "vkCmdBeginRenderPass vkCmdEndRenderPass vkCmdClearAttachments")
{
    struct ct_scene s;
    struct ct_target c, d;
    VkAttachmentDescription a[2];
    VkAttachmentReference color_ref = { 0, VK_IMAGE_LAYOUT_GENERAL }, depth_ref = { 1, VK_IMAGE_LAYOUT_GENERAL };
    VkSubpassDescription sp;
    VkRenderPassCreateInfo rpi;
    VkFramebufferCreateInfo fbi;
    VkRenderPassBeginInfo begin;
    VkRenderPass rp = VK_NULL_HANDLE;
    VkFramebuffer fb = VK_NULL_HANDLE;
    VkImageView views[2];
    VkClearValue clears[2];
    VkClearAttachment atts[2];
    VkClearRect rects[2];
    VkCommandBuffer cb;
    struct ct_box layer0 = { 0, 0, 0, 0, CT_RGBA(0, 0, 255, 255), CT_RGBA(0, 0, 255, 255) };
    uint32_t i;

    ct_scene_init(t, &s);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, CT_SIZE, CT_SIZE, 2, VK_SAMPLE_COUNT_1_BIT,
              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &c);
    ct_target(t, &s.e, VK_FORMAT_D16_UNORM, CT_SIZE, CT_SIZE, 2, VK_SAMPLE_COUNT_1_BIT,
              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, &d);
    memset(a, 0, sizeof(a));
    for (i = 0; i < 2; i++) {
        a[i].format = i ? VK_FORMAT_D16_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
        a[i].samples = VK_SAMPLE_COUNT_1_BIT;
        a[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        a[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        a[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a[i].initialLayout = VK_IMAGE_LAYOUT_GENERAL;
        a[i].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    memset(&sp, 0, sizeof(sp));
    sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp.colorAttachmentCount = 1;
    sp.pColorAttachments = &color_ref;
    sp.pDepthStencilAttachment = &depth_ref;
    memset(&rpi, 0, sizeof(rpi));
    rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpi.attachmentCount = 2;
    rpi.pAttachments = a;
    rpi.subpassCount = 1;
    rpi.pSubpasses = &sp;
    CT_REQUIRE_VK(t, vkCreateRenderPass(s.e.device, &rpi, NULL, &rp));
    views[0] = c.view;
    views[1] = d.view;
    memset(&fbi, 0, sizeof(fbi));
    fbi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbi.renderPass = rp;
    fbi.attachmentCount = 2;
    fbi.pAttachments = views;
    fbi.width = CT_SIZE;
    fbi.height = CT_SIZE;
    fbi.layers = 2;
    CT_REQUIRE_VK(t, vkCreateFramebuffer(s.e.device, &fbi, NULL, &fb));
    memset(clears, 0, sizeof(clears));
    memcpy(clears[0].color.float32, ct_blue, sizeof(ct_blue));
    clears[1].depthStencil.depth = 1.0f;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = rp;
    begin.framebuffer = fb;
    begin.renderArea.extent.width = CT_SIZE;
    begin.renderArea.extent.height = CT_SIZE;
    begin.clearValueCount = 2;
    begin.pClearValues = clears;
    memset(atts, 0, sizeof(atts));
    atts[0].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    atts[0].colorAttachment = 0;
    memcpy(atts[0].clearValue.color.float32, ct_red, sizeof(ct_red));
    atts[1].aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    atts[1].clearValue.depthStencil.depth = 0.25f;
    memset(rects, 0, sizeof(rects));
    rects[0].rect.offset.x = rects[0].rect.offset.y = 2;
    rects[0].rect.extent.width = rects[0].rect.extent.height = 4;
    rects[0].baseArrayLayer = 1;
    rects[0].layerCount = 1;
    rects[1].rect.offset.x = rects[1].rect.offset.y = 8;
    rects[1].rect.extent.width = rects[1].rect.extent.height = 4;
    rects[1].baseArrayLayer = 0;
    rects[1].layerCount = 2;
    cb = ct_begin_commands(t, &s.e);
    vkCmdBeginRenderPass(cb, &begin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdClearAttachments(cb, 1, &atts[0], 1, &rects[0]); /* color: layer 1 only */
    vkCmdClearAttachments(cb, 1, &atts[1], 1, &rects[1]); /* depth: both layers */
    vkCmdEndRenderPass(cb);
    for (i = 0; i < 2; i++) {
        ct_target_read(cb, &c, i, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, i * CT_SIZE * CT_SIZE * 4);
        ct_target_read(cb, &d, i, VK_IMAGE_ASPECT_DEPTH_BIT, s.out.buffer, (2 + i) * CT_SIZE * CT_SIZE * 4);
    }
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_rgba8(t, s.out.data, CT_SIZE, CT_SIZE, ct_expect_box, &layer0, 0, "color layer 0 (not cleared)");
    ct_check_rgba8(t, s.out.data + CT_SIZE * CT_SIZE * 4, CT_SIZE, CT_SIZE, expect_cleared_rect, NULL, 0,
                   "color layer 1");
    ct_check_d16(t, s.out.data + 2 * CT_SIZE * CT_SIZE * 4, CT_SIZE, CT_SIZE, expect_depth_rect, "depth layer 0");
    ct_check_d16(t, s.out.data + 3 * CT_SIZE * CT_SIZE * 4, CT_SIZE, CT_SIZE, expect_depth_rect, "depth layer 1");
    vkDestroyFramebuffer(s.e.device, fb, NULL);
    vkDestroyRenderPass(s.e.device, rp, NULL);
    ct_target_fini(&s.e, &c);
    ct_target_fini(&s.e, &d);
    ct_scene_fini(&s);
}

CT_TEST(update_buffer,
        "Clear Commands > Updating Buffers (vkCmdUpdateBuffer): dataSize bytes (a multiple of 4, up to 65536) "
        "from pData are written at dstOffset (a multiple of 4), the rest of the buffer untouched; the data is "
        "copied from pData when the command is recorded (pData changed and freed before the submission, and the "
        "command buffer submitted twice), also in a secondary command buffer; it is a transfer operation for "
        "synchronization (ordered after a fill and another update by transfer barriers)",
        "vkCmdUpdateBuffer vkCmdFillBuffer vkCmdPipelineBarrier vkCmdExecuteCommands vkCreateCommandPool "
        "vkDestroyCommandPool vkAllocateCommandBuffers vkFreeCommandBuffers vkBeginCommandBuffer vkEndCommandBuffer "
        "vkQueueSubmit vkQueueWaitIdle "
        "vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory vkBindBufferMemory "
        "vkMapMemory vkFreeMemory")
{
    enum { SIZE = 160 * 1024, BIG = 65536, BIG_AT = 1028, OVER_AT = 70000, SECOND_AT = 100000 };
    static const uint32_t small[2] = { 0x01234567u, 0x89abcdefu };
    static const uint32_t first[4] = { 0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u };
    static const uint32_t second[2] = { 0x55555555u, 0x66666666u };
    static const uint32_t third[3] = { 0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu };
    struct ct_env e;
    struct ct_buffer b;
    VkCommandPoolCreateInfo pinfo;
    VkCommandBufferAllocateInfo ainfo;
    VkCommandBufferBeginInfo begin;
    VkCommandBufferInheritanceInfo inherit;
    VkSubmitInfo submit;
    VkCommandPool pool;
    VkCommandBuffer cb, secondary;
    unsigned char *big, *want;
    uint32_t round;
    size_t i;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    ct_host_buffer(t, &e, SIZE, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &b);
    big = malloc(BIG);
    want = malloc(SIZE);
    CT_REQUIRE(t, big && want);
    ct_fill_pattern(big, BIG, 9);
    memset(want, 0xEE, SIZE);
    memcpy(want + 4, small, sizeof(small));
    memcpy(want + BIG_AT, big, BIG);
    memcpy(want + OVER_AT, first, sizeof(first));
    memcpy(want + OVER_AT + 8, second, sizeof(second));
    memcpy(want + SIZE - 4, small, 4);
    memcpy(want + SECOND_AT, third, sizeof(third));

    memset(&pinfo, 0, sizeof(pinfo));
    pinfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pinfo.queueFamilyIndex = e.queue_family;
    CT_REQUIRE_VK(t, vkCreateCommandPool(e.device, &pinfo, NULL, &pool));
    memset(&ainfo, 0, sizeof(ainfo));
    ainfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ainfo.commandPool = pool;
    ainfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ainfo.commandBufferCount = 1;
    CT_REQUIRE_VK(t, vkAllocateCommandBuffers(e.device, &ainfo, &cb));
    ainfo.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    CT_REQUIRE_VK(t, vkAllocateCommandBuffers(e.device, &ainfo, &secondary));
    memset(&inherit, 0, sizeof(inherit));
    inherit.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.pInheritanceInfo = &inherit;
    CT_REQUIRE_VK(t, vkBeginCommandBuffer(secondary, &begin));
    vkCmdUpdateBuffer(secondary, b.buffer, SECOND_AT, sizeof(third), third);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(secondary));
    begin.pInheritanceInfo = NULL;
    CT_REQUIRE_VK(t, vkBeginCommandBuffer(cb, &begin));
    vkCmdFillBuffer(cb, b.buffer, 0, VK_WHOLE_SIZE, 0xEEEEEEEEu);
    ct_transfer_barrier(cb);
    vkCmdExecuteCommands(cb, 1, &secondary);
    vkCmdUpdateBuffer(cb, b.buffer, 4, sizeof(small), small);
    vkCmdUpdateBuffer(cb, b.buffer, BIG_AT, BIG, big);
    vkCmdUpdateBuffer(cb, b.buffer, OVER_AT, sizeof(first), first);
    vkCmdUpdateBuffer(cb, b.buffer, SIZE - 4, 4, small); /* the last word */
    ct_transfer_barrier(cb);
    vkCmdUpdateBuffer(cb, b.buffer, OVER_AT + 8, sizeof(second), second);
    ct_host_read_barrier(cb);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    memset(big, 0, BIG); /* recorded already: neither this nor the free matters */
    free(big);
    for (round = 0; round < 2; round++) {
        memset(b.data, 0, SIZE);
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cb;
        CT_REQUIRE_VK(t, vkQueueSubmit(e.queue, 1, &submit, VK_NULL_HANDLE));
        CT_REQUIRE_VK(t, vkQueueWaitIdle(e.queue));
        for (i = 0; i < SIZE; i++)
            if (b.data[i] != want[i]) {
                ct_fail(t, __FILE__, __LINE__, "submission %u: byte %zu is 0x%02x, expected 0x%02x", round + 1, i,
                        b.data[i], want[i]);
                break;
            }
    }
    vkFreeCommandBuffers(e.device, pool, 1, &cb);
    vkFreeCommandBuffers(e.device, pool, 1, &secondary);
    vkDestroyCommandPool(e.device, pool, NULL);
    free(want);
    ct_buffer_fini(&e, &b);
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(clear_color_image),
    CT_REF(clear_color_image_integer),
    CT_REF(clear_color_image_3d),
    CT_REF(clear_depth_stencil_image),
    CT_REF(clear_attachments),
    CT_REF(update_buffer),
};

CT_SUITE(clears, tests)
