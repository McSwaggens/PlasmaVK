/* Chapter "Fragment Operations": the scissor test, the depth test with
 * every compare op and depth writes off, stencil operations (every op,
 * clamping and wrapping, fail and depth-fail ops, write and compare masks),
 * the depth bounds test, the sample mask (alone and with the SampleMask
 * built-ins) and alpha to one. */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

#define W CT_SIZE
#define H CT_SIZE

#define FO_COMMANDS                                                                                                \
    CT_RENDER_COMMANDS "vkCmdBindPipeline vkCmdDraw vkCmdBeginRendering vkCmdEndRendering "

/*
 * #version 450
 * // SampleMask: red is a quarter per sample of the input coverage, and the
 * // output coverage is the push constants' layer.
 * layout(push_constant) uniform P { vec4 rect; vec4 color; float depth; int layer; int viewport; } pc;
 * layout(location = 0) out vec4 o_color;
 * void main()
 * {
 *     o_color = vec4(float(bitCount(gl_SampleMaskIn[0])) * 0.25, 0.0, 0.0, 1.0);
 *     gl_SampleMask[0] = pc.layer;
 * }
 */
static const uint32_t mask_frag[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000026, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0009000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000f, 0x0000001c,
    0x0000001f, 0x00030010, 0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f635f6f, 0x00726f6c, 0x00060005,
    0x0000000f, 0x535f6c67, 0x6c706d61, 0x73614d65, 0x006e496b, 0x00060005, 0x0000001c, 0x535f6c67,
    0x6c706d61, 0x73614d65, 0x0000006b, 0x00030005, 0x0000001d, 0x00000050, 0x00050006, 0x0000001d,
    0x00000000, 0x74636572, 0x00000000, 0x00050006, 0x0000001d, 0x00000001, 0x6f6c6f63, 0x00000072,
    0x00050006, 0x0000001d, 0x00000002, 0x74706564, 0x00000068, 0x00050006, 0x0000001d, 0x00000003,
    0x6579616c, 0x00000072, 0x00060006, 0x0000001d, 0x00000004, 0x77656976, 0x74726f70, 0x00000000,
    0x00030005, 0x0000001f, 0x00006370, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00040047,
    0x0000000f, 0x0000000b, 0x00000014, 0x00030047, 0x0000000f, 0x0000000e, 0x00040047, 0x0000001c,
    0x0000000b, 0x00000014, 0x00030047, 0x0000001d, 0x00000002, 0x00050048, 0x0000001d, 0x00000000,
    0x00000023, 0x00000000, 0x00050048, 0x0000001d, 0x00000001, 0x00000023, 0x00000010, 0x00050048,
    0x0000001d, 0x00000002, 0x00000023, 0x00000020, 0x00050048, 0x0000001d, 0x00000003, 0x00000023,
    0x00000024, 0x00050048, 0x0000001d, 0x00000004, 0x00000023, 0x00000028, 0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007,
    0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008,
    0x00000009, 0x00000003, 0x00040015, 0x0000000a, 0x00000020, 0x00000001, 0x00040015, 0x0000000b,
    0x00000020, 0x00000000, 0x0004002b, 0x0000000b, 0x0000000c, 0x00000001, 0x0004001c, 0x0000000d,
    0x0000000a, 0x0000000c, 0x00040020, 0x0000000e, 0x00000001, 0x0000000d, 0x0004003b, 0x0000000e,
    0x0000000f, 0x00000001, 0x0004002b, 0x0000000a, 0x00000010, 0x00000000, 0x00040020, 0x00000011,
    0x00000001, 0x0000000a, 0x0004002b, 0x00000006, 0x00000016, 0x3e800000, 0x0004002b, 0x00000006,
    0x00000018, 0x00000000, 0x0004002b, 0x00000006, 0x00000019, 0x3f800000, 0x00040020, 0x0000001b,
    0x00000003, 0x0000000d, 0x0004003b, 0x0000001b, 0x0000001c, 0x00000003, 0x0007001e, 0x0000001d,
    0x00000007, 0x00000007, 0x00000006, 0x0000000a, 0x0000000a, 0x00040020, 0x0000001e, 0x00000009,
    0x0000001d, 0x0004003b, 0x0000001e, 0x0000001f, 0x00000009, 0x0004002b, 0x0000000a, 0x00000020,
    0x00000003, 0x00040020, 0x00000021, 0x00000009, 0x0000000a, 0x00040020, 0x00000024, 0x00000003,
    0x0000000a, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
    0x00050041, 0x00000011, 0x00000012, 0x0000000f, 0x00000010, 0x0004003d, 0x0000000a, 0x00000013,
    0x00000012, 0x000400cd, 0x0000000a, 0x00000014, 0x00000013, 0x0004006f, 0x00000006, 0x00000015,
    0x00000014, 0x00050085, 0x00000006, 0x00000017, 0x00000015, 0x00000016, 0x00070050, 0x00000007,
    0x0000001a, 0x00000017, 0x00000018, 0x00000018, 0x00000019, 0x0003003e, 0x00000009, 0x0000001a,
    0x00050041, 0x00000021, 0x00000022, 0x0000001f, 0x00000020, 0x0004003d, 0x0000000a, 0x00000023,
    0x00000022, 0x00050041, 0x00000024, 0x00000025, 0x0000001c, 0x00000010, 0x0003003e, 0x00000025,
    0x00000023, 0x000100fd, 0x00010038,
};

/* NDC of the left edge of pixel column (or row) p. */
static float edge(uint32_t p)
{
    return (float)p / 8.0f - 1.0f;
}

CT_TEST(scissor_test,
        "Fragment Operations > Scissor Test: fragments outside the scissor rectangle "
        "(VkPipelineViewportStateCreateInfo::pScissors) are discarded",
        FO_COMMANDS)
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkPipeline pipe;
    VkCommandBuffer cb;
    struct ct_box want = { 2, 3, 7, 9, CT_RED, CT_BLACK };

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.scissors[0].offset.x = 2;
    g.scissors[0].offset.y = 3;
    g.scissors[0].extent.width = 5;
    g.scissors[0].extent.height = 6;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, ct_red, 0.0f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want, 0, "scissor (2, 3) 5 x 6");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

/* Which of the rows at depth 0.1 (rows 0..3), 0.2 (4..7, equal to the
 * stored 0.2) and 0.3 (8..11) pass each compare op (bits 0, 1, 2). */
static const uint32_t compare_passes[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

static uint32_t expect_compare(uint32_t x, uint32_t y, const void *ctx)
{
    uint32_t op = x / 2, row = y / 4;
    (void)ctx;
    if (row < 3 && (compare_passes[op] & (1u << row)))
        return CT_RED;
    return CT_BLACK;
}

static uint16_t expect_depth_kept(uint32_t x, uint32_t y)
{
    (void)x;
    (void)y;
    return 13107; /* 0.2: no compare op wrote */
}

CT_TEST(depth_compare_ops,
        "Fragment Operations > Depth Test: the fragment's depth is compared with the stored depth by "
        "depthCompareOp (NEVER, LESS, EQUAL, LESS_OR_EQUAL, GREATER, NOT_EQUAL, GREATER_OR_EQUAL, ALWAYS), the "
        "fragment discarded when the comparison fails; with depthWriteEnable VK_FALSE the stored depth is kept",
        FO_COMMANDS)
{
    struct ct_scene s;
    struct ct_target c, d;
    struct ct_gfx g;
    VkPipeline pipes[8];
    VkCommandBuffer cb;
    uint32_t op, row;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    ct_target(t, &s.e, VK_FORMAT_D16_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              &d);
    for (op = 0; op < 8; op++) {
        ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
        g.rendering.depthAttachmentFormat = VK_FORMAT_D16_UNORM;
        g.ds.depthTestEnable = VK_TRUE;
        g.ds.depthWriteEnable = VK_FALSE;
        g.ds.depthCompareOp = (VkCompareOp)op; /* VK_COMPARE_OP_NEVER .. ALWAYS are 0 .. 7 */
        pipes[op] = ct_gfx_pipeline(t, &s.e, &g);
    }
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, &d, 0.2f); /* 0.2 * 65535 = 13107 exactly */
    for (op = 0; op < 8; op++) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[op]);
        for (row = 0; row < 3; row++)
            ct_draw_rect(cb, s.layout, edge(2 * op), edge(4 * row), edge(2 * op + 2), edge(4 * row + 4), ct_red,
                         0.1f * (float)(row + 1));
    }
    vkCmdEndRendering(cb);
    ct_target_read(cb, &d, 0, VK_IMAGE_ASPECT_DEPTH_BIT, s.out.buffer, W * H * 4);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, expect_compare, NULL, 0,
                   "columns of NEVER .. ALWAYS, rows at depth 0.1, 0.2, 0.3 against 0.2");
    ct_check_d16(t, s.out.data + W * H * 4, W, H, expect_depth_kept, "depth (writes off)");
    for (op = 0; op < 8; op++)
        vkDestroyPipeline(s.e.device, pipes[op], NULL);
    ct_target_fini(&s.e, &d);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

/* The stencil values the stencil test writes: see stencil_ops. */
static uint8_t expect_stencil(uint32_t x, uint32_t y)
{
    static const uint8_t pass_ops[8] = { 0x80, 0x00, 0x3c, 0x81, 0x7f, 0x7f, 0x81, 0x7f };
    static const uint8_t edges[4] = { 0xff, 0x00, 0x00, 0xff };
    static const uint8_t others[5] = { 0x11, 0x7f, 0x8f, 0x7f, 0x80 };
    if (y < 4 && x < 8)
        return pass_ops[x];
    if (y >= 4 && y < 8 && x >= 8 && x < 12)
        return edges[x - 8];
    if (y >= 8 && y < 12 && x < 5)
        return others[x];
    return 0x80;
}

CT_TEST(stencil_ops,
        "Fragment Operations > Stencil Test: the reference and the stored value, both ANDed with compareMask, "
        "are compared by compareOp; failOp applies when the comparison fails, depthFailOp when it passes and the "
        "depth test fails, passOp when both pass; KEEP, ZERO, REPLACE, INCREMENT_AND_CLAMP, DECREMENT_AND_CLAMP, "
        "INVERT, INCREMENT_AND_WRAP and DECREMENT_AND_WRAP update the value, of which only the writeMask bits are "
        "written",
        FO_COMMANDS "vkCmdSetStencilOp vkCmdSetStencilReference vkCmdSetStencilCompareMask vkCmdSetStencilWriteMask")
{
    struct ct_scene s;
    struct ct_target c, ds;
    struct ct_gfx g;
    VkFormat format;
    VkPipeline pipe;
    VkCommandBuffer cb;
    const VkStencilFaceFlags both = VK_STENCIL_FACE_FRONT_AND_BACK;
    uint32_t k, x, y;
    int bad = 0;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    format = ct_depth_stencil_format(&s.e);
    ct_target(t, &s.e, format, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, &ds);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.rendering.depthAttachmentFormat = g.rendering.stencilAttachmentFormat = format;
    g.ds.depthTestEnable = VK_TRUE;
    g.ds.depthCompareOp = VK_COMPARE_OP_LESS;
    g.ds.stencilTestEnable = VK_TRUE;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_STENCIL_OP;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_STENCIL_REFERENCE;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_STENCIL_WRITE_MASK;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    {
        VkRenderingAttachmentInfo ca, da;
        VkRenderingInfo info;
        memset(&ca, 0, sizeof(ca));
        ca.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        ca.imageView = c.view;
        ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        da = ca;
        da.imageView = ds.view;
        da.clearValue.depthStencil.depth = 0.5f;
        da.clearValue.depthStencil.stencil = 0x80;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        info.renderArea.extent.width = W;
        info.renderArea.extent.height = H;
        info.layerCount = 1;
        info.colorAttachmentCount = 1;
        info.pColorAttachments = &ca;
        info.pDepthAttachment = &da;
        info.pStencilAttachment = &da;
        vkCmdBeginRendering(cb, &info);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdSetStencilCompareMask(cb, both, 0xff);
    vkCmdSetStencilWriteMask(cb, both, 0xff);
    /* rows 0..3, column k: passOp k with reference 0x3c on 0x80 */
    vkCmdSetStencilReference(cb, both, 0x3c);
    for (k = 0; k < 8; k++) {
        vkCmdSetStencilOp(cb, both, VK_STENCIL_OP_KEEP, (VkStencilOp)k, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
        ct_draw_rect(cb, s.layout, edge(k), edge(0), edge(k + 1), edge(4), ct_red, 0.25f);
    }
    /* rows 4..7: 0xff (columns 8, 9) and 0 (10, 11), then clamping and wrapping */
    vkCmdSetStencilReference(cb, both, 0xff);
    vkCmdSetStencilOp(cb, both, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
    ct_draw_rect(cb, s.layout, edge(8), edge(4), edge(10), edge(8), ct_red, 0.25f);
    vkCmdSetStencilOp(cb, both, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
    ct_draw_rect(cb, s.layout, edge(10), edge(4), edge(12), edge(8), ct_red, 0.25f);
    for (k = 0; k < 4; k++) {
        static const VkStencilOp ops[4] = { VK_STENCIL_OP_INCREMENT_AND_CLAMP, VK_STENCIL_OP_INCREMENT_AND_WRAP,
                                            VK_STENCIL_OP_DECREMENT_AND_CLAMP, VK_STENCIL_OP_DECREMENT_AND_WRAP };
        vkCmdSetStencilOp(cb, both, VK_STENCIL_OP_KEEP, ops[k], VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
        ct_draw_rect(cb, s.layout, edge(8 + k), edge(4), edge(9 + k), edge(8), ct_red, 0.25f);
    }
    /* rows 8..11 */
    vkCmdSetStencilReference(cb, both, 0x11); /* column 0: compare NEVER, failOp REPLACE */
    vkCmdSetStencilOp(cb, both, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_ZERO, VK_COMPARE_OP_NEVER);
    ct_draw_rect(cb, s.layout, edge(0), edge(8), edge(1), edge(12), ct_red, 0.25f);
    /* column 1: the depth test fails (0.75 against 0.5): depthFailOp INVERT */
    vkCmdSetStencilOp(cb, both, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_INVERT, VK_COMPARE_OP_ALWAYS);
    ct_draw_rect(cb, s.layout, edge(1), edge(8), edge(2), edge(12), ct_red, 0.75f);
    /* column 2: REPLACE 0xff through writeMask 0x0f */
    vkCmdSetStencilReference(cb, both, 0xff);
    vkCmdSetStencilWriteMask(cb, both, 0x0f);
    vkCmdSetStencilOp(cb, both, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
    ct_draw_rect(cb, s.layout, edge(2), edge(8), edge(3), edge(12), ct_red, 0.25f);
    /* columns 3, 4: EQUAL under compareMask 0xf0: 0x8c passes (INVERT), 0x4c fails (KEEP) */
    vkCmdSetStencilWriteMask(cb, both, 0xff);
    vkCmdSetStencilCompareMask(cb, both, 0xf0);
    vkCmdSetStencilOp(cb, both, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_INVERT, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_EQUAL);
    vkCmdSetStencilReference(cb, both, 0x8c);
    ct_draw_rect(cb, s.layout, edge(3), edge(8), edge(4), edge(12), ct_red, 0.25f);
    vkCmdSetStencilReference(cb, both, 0x4c);
    ct_draw_rect(cb, s.layout, edge(4), edge(8), edge(5), edge(12), ct_red, 0.25f);
    vkCmdEndRendering(cb);
    ct_target_read(cb, &ds, 0, VK_IMAGE_ASPECT_STENCIL_BIT, s.out.buffer, 0);
    ct_submit_and_wait(t, &s.e, cb);
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++) {
            uint8_t got = s.out.data[y * W + x], want = expect_stencil(x, y);
            if (got != want && bad++ < 6)
                ct_fail(t, __FILE__, __LINE__, "stencil texel (%u, %u) is 0x%02x, expected 0x%02x", x, y, got, want);
        }
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_target_fini(&s.e, &ds);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(depth_bounds_test,
        "Fragment Operations > Depth Bounds Test: with depthBoundsTestEnable (depthBounds), a sample passes only "
        "if the depth stored in the attachment (not the fragment's) lies in [minDepthBounds, maxDepthBounds]",
        FO_COMMANDS)
{
    struct ct_scene s;
    struct ct_target c, d;
    struct ct_gfx g;
    VkPipeline write, bounded;
    VkCommandBuffer cb;
    struct ct_box want = { W / 2, 0, W, H, CT_RED, CT_BLACK };

    ct_scene_init(t, &s);
    if (!s.f.f2.features.depthBounds)
        ct_skip(t, "depthBounds is not supported");
    ct_scene_color(t, &s, &c);
    ct_target(t, &s.e, VK_FORMAT_D16_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              &d);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.rendering.depthAttachmentFormat = VK_FORMAT_D16_UNORM;
    g.ds.depthTestEnable = g.ds.depthWriteEnable = VK_TRUE;
    g.ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    g.cb.attachmentCount = 1;
    g.blend[0].colorWriteMask = 0;
    write = ct_gfx_pipeline(t, &s.e, &g);
    g.blend[0].colorWriteMask = 0xf;
    g.ds.depthTestEnable = g.ds.depthWriteEnable = VK_FALSE;
    g.ds.depthBoundsTestEnable = VK_TRUE;
    g.ds.minDepthBounds = 0.4f;
    g.ds.maxDepthBounds = 0.6f;
    bounded = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, &d, 0.5f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, write);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 1.0f, NULL, 0.25f); /* left half: 0.25 */
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, bounded);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, ct_red, 0.9f); /* the fragment's 0.9 does not matter */
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want, 0, "drawn where the stored depth is 0.5");
    vkDestroyPipeline(s.e.device, write, NULL);
    vkDestroyPipeline(s.e.device, bounded, NULL);
    ct_target_fini(&s.e, &d);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(sample_mask_test,
        "Fragment Operations > Sample Mask Test: coverage bits whose bit in pSampleMask is 0 are cleared: with 4 "
        "samples, mask 0x1 leaves one sample of a covered pixel and 0x5 two",
        FO_COMMANDS)
{
    struct ct_scene s;
    struct ct_target ms, c;
    struct ct_gfx g;
    VkSampleMask masks[2] = { 0x1, 0x5 };
    VkPipeline pipes[2];
    VkRenderingAttachmentInfo att;
    VkRenderingInfo info;
    VkCommandBuffer cb;
    uint32_t k, want[4] = { CT_RGBA(64, 0, 0, 255), CT_RGBA(128, 0, 0, 255), CT_BLACK, CT_BLACK };

    ct_scene_init(t, &s);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_4_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &ms);
    ct_scene_color(t, &s, &c);
    for (k = 0; k < 2; k++) {
        ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
        g.ms.rasterizationSamples = VK_SAMPLE_COUNT_4_BIT;
        g.ms.pSampleMask = &masks[k];
        pipes[k] = ct_gfx_pipeline(t, &s.e, &g);
    }
    memset(&att, 0, sizeof(att));
    att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    att.imageView = ms.view;
    att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    att.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
    att.resolveImageView = c.view;
    att.resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    memcpy(att.clearValue.color.float32, ct_black, sizeof(ct_black));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    info.renderArea.extent.width = W;
    info.renderArea.extent.height = H;
    info.layerCount = 1;
    info.colorAttachmentCount = 1;
    info.pColorAttachments = &att;
    cb = ct_begin_commands(t, &s.e);
    vkCmdBeginRendering(cb, &info);
    for (k = 0; k < 2; k++) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[k]);
        ct_draw_rect(cb, s.layout, k ? 0.0f : -1.0f, -1.0f, k ? 1.0f : 0.0f, 0.0f, ct_red, 0.0f);
    }
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_quadrants, want, 1, "masks 0x1 (top left) and 0x5 (top right)");
    for (k = 0; k < 2; k++)
        vkDestroyPipeline(s.e.device, pipes[k], NULL);
    ct_target_fini(&s.e, &ms);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(sample_mask_shader_coverage,
        "Fragment Operations > Sample Mask Test, Fragment Shading, Multisample Coverage: the sample mask test "
        "comes before fragment shading, so the SampleMask input holds the coverage it leaves (pSampleMask 0x5 "
        "of 4 samples: 2 bits), and the SampleMask output is ANDed with that coverage (0x3: sample 0 written)",
        FO_COMMANDS)
{
    struct ct_scene s;
    struct ct_target ms, c;
    struct ct_gfx g;
    struct ct_rect_pc pc;
    VkSampleMask mask = 0x5;
    VkShaderModule fs;
    VkPipeline pipe;
    VkRenderingAttachmentInfo att;
    VkRenderingInfo info;
    VkCommandBuffer cb;
    uint32_t want[4] = { CT_RGBA(32, 0, 0, 255), CT_BLACK, CT_BLACK, CT_BLACK };

    ct_scene_init(t, &s);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_4_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &ms);
    ct_scene_color(t, &s, &c);
    fs = ct_shader_module(t, &s.e, mask_frag, sizeof(mask_frag));
    ct_gfx_init(&g, s.vs, fs, s.layout, W, H);
    g.ms.rasterizationSamples = VK_SAMPLE_COUNT_4_BIT;
    g.ms.pSampleMask = &mask;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    memset(&att, 0, sizeof(att));
    att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    att.imageView = ms.view;
    att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    att.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
    att.resolveImageView = c.view;
    att.resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    memcpy(att.clearValue.color.float32, ct_black, sizeof(ct_black));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    info.renderArea.extent.width = W;
    info.renderArea.extent.height = H;
    info.layerCount = 1;
    info.colorAttachmentCount = 1;
    info.pColorAttachments = &att;
    memset(&pc, 0, sizeof(pc));
    pc.rect[0] = pc.rect[1] = -1.0f; /* the top left quadrant */
    pc.layer = 0x3;                  /* the SampleMask output */
    cb = ct_begin_commands(t, &s.e);
    vkCmdBeginRendering(cb, &info);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdPushConstants(cb, s.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    vkCmdDraw(cb, 4, 1, 0, 0);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_quadrants, want, 1,
                   "top left: red 0.5 (2 samples of input coverage) on 1 sample of 4");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_target_fini(&s.e, &ms);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(alpha_to_one,
        "Fragment Operations > Multisample Coverage: with alphaToOneEnable (alphaToOne), each alpha value is "
        "replaced by the maximum representable alpha before blending: alpha 0.25 is stored as 1, and "
        "SRC_ALPHA / ONE_MINUS_SRC_ALPHA blending takes the source whole",
        FO_COMMANDS)
{
    static const float faint[4] = { 1.0f, 0.0f, 0.0f, 0.25f };
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkPipeline plain, blended;
    VkCommandBuffer cb;
    struct ct_box want = { 0, 0, W, H / 2, CT_RED, CT_RED };

    ct_scene_init(t, &s);
    if (!s.f.f2.features.alphaToOne)
        ct_skip(t, "alphaToOne is not supported");
    ct_scene_color(t, &s, &c);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.ms.alphaToOneEnable = VK_TRUE;
    plain = ct_gfx_pipeline(t, &s.e, &g);
    g.blend[0].blendEnable = VK_TRUE;
    g.blend[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    g.blend[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    g.blend[0].colorBlendOp = VK_BLEND_OP_ADD;
    g.blend[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    g.blend[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    g.blend[0].alphaBlendOp = VK_BLEND_OP_ADD;
    blended = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_blue, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, plain);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 0.0f, faint, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, blended);
    ct_draw_rect(cb, s.layout, -1.0f, 0.0f, 1.0f, 1.0f, faint, 0.0f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want, 1, "red with alpha 1 (top: written, bottom: blended)");
    vkDestroyPipeline(s.e.device, plain, NULL);
    vkDestroyPipeline(s.e.device, blended, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

static const struct ct_test *const tests[] = {
    CT_REF(scissor_test),
    CT_REF(depth_compare_ops),
    CT_REF(stencil_ops),
    CT_REF(depth_bounds_test),
    CT_REF(sample_mask_test),
    CT_REF(sample_mask_shader_coverage),
    CT_REF(alpha_to_one),
};

CT_SUITE(fragops, tests)
