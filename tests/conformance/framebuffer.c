/* Chapter "The Framebuffer": blending (factors, operations, constants static
 * and dynamic, independent per attachment, dual source), color write
 * masks, blending of sRGB attachments in linear space, and logic
 * operations. Expected colors are computed from the specification's
 * equations, the destination taken as stored (8 bits). */
#include "ct.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define W CT_SIZE
#define H CT_SIZE

#define FB_COMMANDS                                                                                                \
    CT_RENDER_COMMANDS "vkCmdBindPipeline vkCmdDraw vkCmdBeginRendering vkCmdEndRendering "

/*
 * #version 450
 * // The push constants' color to two attachments.
 * layout(push_constant) uniform P { vec4 rect; vec4 color; float depth; int layer; int viewport; } pc;
 * layout(location = 0) out vec4 o_color0;
 * layout(location = 1) out vec4 o_color1;
 * void main()
 * {
 *     o_color0 = pc.color;
 *     o_color1 = pc.color;
 * }
 */
static const uint32_t two_frag[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000015, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0008000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000d, 0x00000012,
    0x00030010, 0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
    0x6e69616d, 0x00000000, 0x00050005, 0x00000009, 0x6f635f6f, 0x30726f6c, 0x00000000, 0x00030005,
    0x0000000b, 0x00000050, 0x00050006, 0x0000000b, 0x00000000, 0x74636572, 0x00000000, 0x00050006,
    0x0000000b, 0x00000001, 0x6f6c6f63, 0x00000072, 0x00050006, 0x0000000b, 0x00000002, 0x74706564,
    0x00000068, 0x00050006, 0x0000000b, 0x00000003, 0x6579616c, 0x00000072, 0x00060006, 0x0000000b,
    0x00000004, 0x77656976, 0x74726f70, 0x00000000, 0x00030005, 0x0000000d, 0x00006370, 0x00050005,
    0x00000012, 0x6f635f6f, 0x31726f6c, 0x00000000, 0x00040047, 0x00000009, 0x0000001e, 0x00000000,
    0x00030047, 0x0000000b, 0x00000002, 0x00050048, 0x0000000b, 0x00000000, 0x00000023, 0x00000000,
    0x00050048, 0x0000000b, 0x00000001, 0x00000023, 0x00000010, 0x00050048, 0x0000000b, 0x00000002,
    0x00000023, 0x00000020, 0x00050048, 0x0000000b, 0x00000003, 0x00000023, 0x00000024, 0x00050048,
    0x0000000b, 0x00000004, 0x00000023, 0x00000028, 0x00040047, 0x00000012, 0x0000001e, 0x00000001,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
    0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007,
    0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00040015, 0x0000000a, 0x00000020, 0x00000001,
    0x0007001e, 0x0000000b, 0x00000007, 0x00000007, 0x00000006, 0x0000000a, 0x0000000a, 0x00040020,
    0x0000000c, 0x00000009, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000009, 0x0004002b,
    0x0000000a, 0x0000000e, 0x00000001, 0x00040020, 0x0000000f, 0x00000009, 0x00000007, 0x0004003b,
    0x00000008, 0x00000012, 0x00000003, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005, 0x00050041, 0x0000000f, 0x00000010, 0x0000000d, 0x0000000e, 0x0004003d,
    0x00000007, 0x00000011, 0x00000010, 0x0003003e, 0x00000009, 0x00000011, 0x00050041, 0x0000000f,
    0x00000013, 0x0000000d, 0x0000000e, 0x0004003d, 0x00000007, 0x00000014, 0x00000013, 0x0003003e,
    0x00000012, 0x00000014, 0x000100fd, 0x00010038,
};

/*
 * #version 450
 * // Dual-source blending: the color (index 0) and a second source (index 1).
 * layout(push_constant) uniform P { vec4 rect; vec4 color; float depth; int layer; int viewport; } pc;
 * layout(location = 0, index = 0) out vec4 o_color;
 * layout(location = 0, index = 1) out vec4 o_factor;
 * void main()
 * {
 *     o_color = pc.color;
 *     o_factor = vec4(0.5, 0.25, 1.0, 0.0);
 * }
 */
static const uint32_t dual_frag[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000018, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0008000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000d, 0x00000012,
    0x00030010, 0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
    0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f635f6f, 0x00726f6c, 0x00030005, 0x0000000b,
    0x00000050, 0x00050006, 0x0000000b, 0x00000000, 0x74636572, 0x00000000, 0x00050006, 0x0000000b,
    0x00000001, 0x6f6c6f63, 0x00000072, 0x00050006, 0x0000000b, 0x00000002, 0x74706564, 0x00000068,
    0x00050006, 0x0000000b, 0x00000003, 0x6579616c, 0x00000072, 0x00060006, 0x0000000b, 0x00000004,
    0x77656976, 0x74726f70, 0x00000000, 0x00030005, 0x0000000d, 0x00006370, 0x00050005, 0x00000012,
    0x61665f6f, 0x726f7463, 0x00000000, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00040047,
    0x00000009, 0x00000020, 0x00000000, 0x00030047, 0x0000000b, 0x00000002, 0x00050048, 0x0000000b,
    0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000000b, 0x00000001, 0x00000023, 0x00000010,
    0x00050048, 0x0000000b, 0x00000002, 0x00000023, 0x00000020, 0x00050048, 0x0000000b, 0x00000003,
    0x00000023, 0x00000024, 0x00050048, 0x0000000b, 0x00000004, 0x00000023, 0x00000028, 0x00040047,
    0x00000012, 0x0000001e, 0x00000000, 0x00040047, 0x00000012, 0x00000020, 0x00000001, 0x00020013,
    0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017,
    0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b,
    0x00000008, 0x00000009, 0x00000003, 0x00040015, 0x0000000a, 0x00000020, 0x00000001, 0x0007001e,
    0x0000000b, 0x00000007, 0x00000007, 0x00000006, 0x0000000a, 0x0000000a, 0x00040020, 0x0000000c,
    0x00000009, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000009, 0x0004002b, 0x0000000a,
    0x0000000e, 0x00000001, 0x00040020, 0x0000000f, 0x00000009, 0x00000007, 0x0004003b, 0x00000008,
    0x00000012, 0x00000003, 0x0004002b, 0x00000006, 0x00000013, 0x3f000000, 0x0004002b, 0x00000006,
    0x00000014, 0x3e800000, 0x0004002b, 0x00000006, 0x00000015, 0x3f800000, 0x0004002b, 0x00000006,
    0x00000016, 0x00000000, 0x0007002c, 0x00000007, 0x00000017, 0x00000013, 0x00000014, 0x00000015,
    0x00000016, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
    0x00050041, 0x0000000f, 0x00000010, 0x0000000d, 0x0000000e, 0x0004003d, 0x00000007, 0x00000011,
    0x00000010, 0x0003003e, 0x00000009, 0x00000011, 0x0003003e, 0x00000012, 0x00000017, 0x000100fd,
    0x00010038,
};
/*
 * #version 450
 * // The push constants' color, as the bits of unsigned integers.
 * layout(push_constant) uniform P { vec4 rect; vec4 color; float depth; int layer; int viewport; } pc;
 * layout(location = 0) out uvec4 o_color;
 * void main()
 * {
 *     o_color = floatBitsToUint(pc.color);
 * }
 */
static const uint32_t uint_frag[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000015, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000f, 0x00030010,
    0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00040005, 0x00000009, 0x6f635f6f, 0x00726f6c, 0x00030005, 0x0000000d, 0x00000050,
    0x00050006, 0x0000000d, 0x00000000, 0x74636572, 0x00000000, 0x00050006, 0x0000000d, 0x00000001,
    0x6f6c6f63, 0x00000072, 0x00050006, 0x0000000d, 0x00000002, 0x74706564, 0x00000068, 0x00050006,
    0x0000000d, 0x00000003, 0x6579616c, 0x00000072, 0x00060006, 0x0000000d, 0x00000004, 0x77656976,
    0x74726f70, 0x00000000, 0x00030005, 0x0000000f, 0x00006370, 0x00040047, 0x00000009, 0x0000001e,
    0x00000000, 0x00030047, 0x0000000d, 0x00000002, 0x00050048, 0x0000000d, 0x00000000, 0x00000023,
    0x00000000, 0x00050048, 0x0000000d, 0x00000001, 0x00000023, 0x00000010, 0x00050048, 0x0000000d,
    0x00000002, 0x00000023, 0x00000020, 0x00050048, 0x0000000d, 0x00000003, 0x00000023, 0x00000024,
    0x00050048, 0x0000000d, 0x00000004, 0x00000023, 0x00000028, 0x00020013, 0x00000002, 0x00030021,
    0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040017, 0x00000007,
    0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008,
    0x00000009, 0x00000003, 0x00030016, 0x0000000a, 0x00000020, 0x00040017, 0x0000000b, 0x0000000a,
    0x00000004, 0x00040015, 0x0000000c, 0x00000020, 0x00000001, 0x0007001e, 0x0000000d, 0x0000000b,
    0x0000000b, 0x0000000a, 0x0000000c, 0x0000000c, 0x00040020, 0x0000000e, 0x00000009, 0x0000000d,
    0x0004003b, 0x0000000e, 0x0000000f, 0x00000009, 0x0004002b, 0x0000000c, 0x00000010, 0x00000001,
    0x00040020, 0x00000011, 0x00000009, 0x0000000b, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x00050041, 0x00000011, 0x00000012, 0x0000000f, 0x00000010,
    0x0004003d, 0x0000000b, 0x00000013, 0x00000012, 0x0004007c, 0x00000007, 0x00000014, 0x00000013,
    0x0003003e, 0x00000009, 0x00000014, 0x000100fd, 0x00010038,
};

static float unorm8(float v)
{
    v = v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
    return floorf(v * 255.0f + 0.5f) / 255.0f;
}

static uint32_t pack(const float c[4])
{
    uint32_t i, v = 0;
    for (i = 0; i < 4; i++)
        v |= (uint32_t)(unorm8(c[i]) * 255.0f + 0.5f) << (8 * i);
    return v;
}

/* Blend factor f for component i (Framebuffer > Blend Factors). */
static float factor(VkBlendFactor f, uint32_t i, const float s[4], const float d[4], const float k[4])
{
    switch (f) {
    case VK_BLEND_FACTOR_ZERO: return 0.0f;
    case VK_BLEND_FACTOR_ONE: return 1.0f;
    case VK_BLEND_FACTOR_SRC_COLOR: return s[i];
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: return 1.0f - s[i];
    case VK_BLEND_FACTOR_DST_COLOR: return d[i];
    case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR: return 1.0f - d[i];
    case VK_BLEND_FACTOR_SRC_ALPHA: return s[3];
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return 1.0f - s[3];
    case VK_BLEND_FACTOR_DST_ALPHA: return d[3];
    case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return 1.0f - d[3];
    case VK_BLEND_FACTOR_CONSTANT_COLOR: return k[i];
    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR: return 1.0f - k[i];
    case VK_BLEND_FACTOR_CONSTANT_ALPHA: return k[3];
    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: return 1.0f - k[3];
    case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE: return i < 3 ? (s[3] < 1.0f - d[3] ? s[3] : 1.0f - d[3]) : 1.0f;
    default: return 0.0f;
    }
}

static float blend_op(VkBlendOp op, float s, float sf, float d, float df)
{
    switch (op) {
    case VK_BLEND_OP_ADD: return s * sf + d * df;
    case VK_BLEND_OP_SUBTRACT: return s * sf - d * df;
    case VK_BLEND_OP_REVERSE_SUBTRACT: return d * df - s * sf;
    case VK_BLEND_OP_MIN: return s < d ? s : d;
    case VK_BLEND_OP_MAX: return s > d ? s : d;
    default: return 0.0f;
    }
}

struct blend_case {
    const char *name;
    VkBlendFactor sc, dc;
    VkBlendOp oc;
    VkBlendFactor sa, da;
    VkBlendOp oa;
    float src[4], dst[4], k[4];
    int dynamic_constants;
};

static const struct blend_case blend_cases[] = {
    { "SRC_ALPHA, ONE_MINUS_SRC_ALPHA; ONE, ZERO", VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, { 0.8f, 0.2f, 0.4f, 0.6f },
      { 0.2f, 0.4f, 1.0f, 1.0f }, { 0 }, 0 },
    { "ONE, ONE (clamped)", VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE,
      VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD, { 0.6f, 0.6f, 0.2f, 0.4f }, { 0.6f, 0.2f, 0.2f, 0.4f }, { 0 }, 0 },
    { "SUBTRACT", VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_SUBTRACT, VK_BLEND_FACTOR_ONE,
      VK_BLEND_FACTOR_ONE, VK_BLEND_OP_SUBTRACT, { 0.8f, 0.4f, 0.2f, 1.0f }, { 0.2f, 0.6f, 0.2f, 0.4f }, { 0 }, 0 },
    { "REVERSE_SUBTRACT", VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_REVERSE_SUBTRACT,
      VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_REVERSE_SUBTRACT, { 0.2f, 0.4f, 0.6f, 0.4f },
      { 0.6f, 0.4f, 0.2f, 0.8f }, { 0 }, 0 },
    { "MIN", VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_MIN, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO,
      VK_BLEND_OP_MIN, { 0.8f, 0.2f, 0.6f, 0.4f }, { 0.4f, 0.6f, 0.6f, 1.0f }, { 0 }, 0 },
    { "MAX", VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_MAX, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO,
      VK_BLEND_OP_MAX, { 0.8f, 0.2f, 0.6f, 0.4f }, { 0.4f, 0.6f, 0.6f, 1.0f }, { 0 }, 0 },
    { "DST_COLOR, ZERO; DST_ALPHA, ZERO", VK_BLEND_FACTOR_DST_COLOR, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
      VK_BLEND_FACTOR_DST_ALPHA, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, { 0.8f, 0.4f, 1.0f, 0.6f },
      { 0.6f, 0.4f, 0.2f, 0.4f }, { 0 }, 0 },
    { "ONE_MINUS_DST_COLOR, SRC_COLOR; ONE_MINUS_DST_ALPHA, ONE", VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
      VK_BLEND_FACTOR_SRC_COLOR, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, VK_BLEND_FACTOR_ONE,
      VK_BLEND_OP_ADD, { 0.4f, 0.8f, 0.2f, 0.2f }, { 0.2f, 0.6f, 1.0f, 0.4f }, { 0 }, 0 },
    { "CONSTANT_COLOR, ONE_MINUS_CONSTANT_COLOR; CONSTANT_ALPHA, ONE_MINUS_CONSTANT_ALPHA",
      VK_BLEND_FACTOR_CONSTANT_COLOR, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR, VK_BLEND_OP_ADD,
      VK_BLEND_FACTOR_CONSTANT_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA, VK_BLEND_OP_ADD,
      { 1.0f, 0.8f, 0.0f, 1.0f }, { 0.0f, 0.4f, 1.0f, 0.2f }, { 0.2f, 0.4f, 0.6f, 0.8f }, 0 },
    { "the same with vkCmdSetBlendConstants", VK_BLEND_FACTOR_CONSTANT_COLOR,
      VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_CONSTANT_ALPHA,
      VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA, VK_BLEND_OP_ADD, { 1.0f, 0.8f, 0.0f, 1.0f },
      { 0.0f, 0.4f, 1.0f, 0.2f }, { 0.6f, 0.2f, 0.4f, 0.2f }, 1 },
    { "SRC_ALPHA_SATURATE, ONE; ONE, ZERO", VK_BLEND_FACTOR_SRC_ALPHA_SATURATE, VK_BLEND_FACTOR_ONE,
      VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, { 0.8f, 0.8f, 0.8f, 0.6f },
      { 0.2f, 0.2f, 0.2f, 0.6f }, { 0 }, 0 },
};

static uint32_t expect_uniform(uint32_t x, uint32_t y, const void *ctx)
{
    (void)x;
    (void)y;
    return *(const uint32_t *)ctx;
}

CT_TEST(blend_equations,
        "The Framebuffer > Blending: with blendEnable the color is srcColorBlendFactor x source colorBlendOp "
        "dstColorBlendFactor x destination (and the same for alpha with the alpha factors and op), clamped for "
        "UNORM attachments; factors ZERO, ONE, SRC/DST COLOR and ALPHA and their ONE_MINUS forms, CONSTANT_COLOR "
        "and CONSTANT_ALPHA with blendConstants (static, or vkCmdSetBlendConstants with "
        "VK_DYNAMIC_STATE_BLEND_CONSTANTS), SRC_ALPHA_SATURATE; ops ADD, SUBTRACT, REVERSE_SUBTRACT, MIN and MAX "
        "(MIN and MAX ignore the factors)",
        FB_COMMANDS "vkCmdSetBlendConstants")
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkPipeline pipe;
    VkCommandBuffer cb;
    uint32_t k, i;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    for (k = 0; k < CT_ARRAY_SIZE(blend_cases); k++) {
        const struct blend_case *b = &blend_cases[k];
        float d[4], out[4];
        uint32_t want;
        for (i = 0; i < 4; i++)
            d[i] = unorm8(b->dst[i]); /* as stored */
        for (i = 0; i < 4; i++) {
            VkBlendFactor sf = i < 3 ? b->sc : b->sa, df = i < 3 ? b->dc : b->da;
            out[i] = blend_op(i < 3 ? b->oc : b->oa, b->src[i], factor(sf, i, b->src, d, b->k), d[i],
                              factor(df, i, b->src, d, b->k));
        }
        want = pack(out);
        ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
        g.blend[0].blendEnable = VK_TRUE;
        g.blend[0].srcColorBlendFactor = b->sc;
        g.blend[0].dstColorBlendFactor = b->dc;
        g.blend[0].colorBlendOp = b->oc;
        g.blend[0].srcAlphaBlendFactor = b->sa;
        g.blend[0].dstAlphaBlendFactor = b->da;
        g.blend[0].alphaBlendOp = b->oa;
        if (b->dynamic_constants) {
            g.cb.blendConstants[0] = g.cb.blendConstants[1] = 1.0f; /* overridden */
            g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_BLEND_CONSTANTS;
        } else {
            memcpy(g.cb.blendConstants, b->k, sizeof(b->k));
        }
        pipe = ct_gfx_pipeline(t, &s.e, &g);
        cb = ct_begin_commands(t, &s.e);
        ct_begin_rendering(cb, &c, 1, b->dst, NULL, 0.0f);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        if (b->dynamic_constants)
            vkCmdSetBlendConstants(cb, b->k);
        ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, b->src, 0.0f);
        vkCmdEndRendering(cb);
        ct_scene_read(t, &s, cb, &c);
        ct_check_rgba8(t, s.out.data, W, H, expect_uniform, &want, 2, b->name);
        vkDestroyPipeline(s.e.device, pipe, NULL);
    }
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(color_write_mask,
        "The Framebuffer > Color Write Mask: only the components in colorWriteMask are written (here R and B), "
        "the others keep their stored values",
        FB_COMMANDS)
{
    static const float dst[4] = { 0.2f, 0.4f, 0.6f, 0.8f };
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkPipeline pipe;
    VkCommandBuffer cb;
    uint32_t want = CT_RGBA(255, 102, 255, 204);

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.blend[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_B_BIT;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, dst, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, ct_white, 0.0f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, expect_uniform, &want, 0, "R and B written");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(independent_blend,
        "The Framebuffer > Blending: with independentBlend, each color attachment uses its own "
        "VkPipelineColorBlendAttachmentState (here one blending ONE, ONE and one not blending)",
        FB_COMMANDS)
{
    static const float dst[4] = { 0.2f, 0.2f, 0.2f, 0.2f }, src[4] = { 0.4f, 0.4f, 0.4f, 0.4f };
    struct ct_scene s;
    struct ct_target c[2];
    struct ct_gfx g;
    VkShaderModule fs;
    VkPipeline pipe;
    VkCommandBuffer cb;
    uint32_t want0 = CT_RGBA(153, 153, 153, 153), want1 = CT_RGBA(102, 102, 102, 102);

    ct_scene_init(t, &s);
    if (!s.f.f2.features.independentBlend)
        ct_skip(t, "independentBlend is not supported");
    ct_scene_color(t, &s, &c[0]);
    ct_scene_color(t, &s, &c[1]);
    fs = ct_shader_module(t, &s.e, two_frag, sizeof(two_frag));
    ct_gfx_init(&g, s.vs, fs, s.layout, W, H);
    g.colors = 2;
    g.blend[0].blendEnable = VK_TRUE;
    g.blend[0].srcColorBlendFactor = g.blend[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    g.blend[0].srcAlphaBlendFactor = g.blend[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    g.blend[0].colorBlendOp = g.blend[0].alphaBlendOp = VK_BLEND_OP_ADD;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, c, 2, dst, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, src, 0.0f);
    vkCmdEndRendering(cb);
    ct_target_read(cb, &c[1], 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, W * H * 4);
    ct_scene_read(t, &s, cb, &c[0]);
    ct_check_rgba8(t, s.out.data, W, H, expect_uniform, &want0, 1, "attachment 0 (blended)");
    ct_check_rgba8(t, s.out.data + W * H * 4, W, H, expect_uniform, &want1, 1, "attachment 1 (written)");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_target_fini(&s.e, &c[0]);
    ct_target_fini(&s.e, &c[1]);
    ct_scene_fini(&s);
}

CT_TEST(dual_source_blend,
        "The Framebuffer > Dual-Source Blending: with dualSrcBlend, SRC1_COLOR and SRC1_ALPHA factors take the "
        "fragment output of Location 0, Index 1",
        FB_COMMANDS)
{
    static const float src[4] = { 0.8f, 0.4f, 1.0f, 1.0f }, dst[4] = { 0.4f, 0.8f, 0.2f, 1.0f };
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkShaderModule fs;
    VkPipeline pipe;
    VkCommandBuffer cb;
    uint32_t want = CT_RGBA(255, 153, 255, 255); /* src + dst x (0.5, 0.25, 1, 0): (1, 0.6, 1, 1) */

    ct_scene_init(t, &s);
    if (!s.f.f2.features.dualSrcBlend)
        ct_skip(t, "dualSrcBlend is not supported");
    ct_scene_color(t, &s, &c);
    fs = ct_shader_module(t, &s.e, dual_frag, sizeof(dual_frag));
    ct_gfx_init(&g, s.vs, fs, s.layout, W, H);
    g.blend[0].blendEnable = VK_TRUE;
    g.blend[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    g.blend[0].dstColorBlendFactor = VK_BLEND_FACTOR_SRC1_COLOR;
    g.blend[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    g.blend[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_SRC1_ALPHA;
    g.blend[0].colorBlendOp = g.blend[0].alphaBlendOp = VK_BLEND_OP_ADD;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, dst, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, src, 0.0f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, expect_uniform, &want, 1, "color");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

static float srgb_encode(float l)
{
    return l <= 0.0031308f ? 12.92f * l : 1.055f * powf(l, 1.0f / 2.4f) - 0.055f;
}

static float srgb_decode(float e)
{
    return e <= 0.04045f ? e / 12.92f : powf((e + 0.055f) / 1.055f, 2.4f);
}

CT_TEST(srgb_blending,
        "The Framebuffer > Blending, Formats: an sRGB attachment's stored values are converted to linear before "
        "blending and the result converted back to sRGB; values written without blending are encoded too "
        "(R8G8B8A8_SRGB, which must support blending)",
        FB_COMMANDS)
{
    static const float half[4] = { 0.5f, 0.5f, 0.5f, 1.0f }, src[4] = { 1.0f, 1.0f, 1.0f, 0.5f };
    static const float dst[4] = { 0.2f, 0.2f, 0.2f, 1.0f };
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkPipeline plain, blended;
    VkCommandBuffer cb;
    uint32_t e, got, x, y;
    float stored, linear;

    ct_scene_init(t, &s);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_SRGB, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &c);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.color_formats[0] = VK_FORMAT_R8G8B8A8_SRGB;
    plain = ct_gfx_pipeline(t, &s.e, &g);
    g.blend[0].blendEnable = VK_TRUE;
    g.blend[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    g.blend[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    g.blend[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    g.blend[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    g.blend[0].colorBlendOp = g.blend[0].alphaBlendOp = VK_BLEND_OP_ADD;
    blended = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, dst, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, plain);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 0.0f, half, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, blended);
    ct_draw_rect(cb, s.layout, -1.0f, 0.0f, 1.0f, 1.0f, src, 0.0f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    /* top: linear 0.5 encoded; bottom: 0.5 + 0.5 x the stored clear color, decoded */
    stored = floorf(srgb_encode(0.2f) * 255.0f + 0.5f) / 255.0f;
    linear = 0.5f + 0.5f * srgb_decode(stored);
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++) {
            float want = y < H / 2 ? srgb_encode(0.5f) * 255.0f : srgb_encode(linear) * 255.0f;
            uint32_t alpha = y < H / 2 ? 255 : 128; /* alpha is linear: ONE x 0.5 */
            int ok;
            got = ct_texel32(s.out.data, W, x, y);
            e = got & 0xff;
            ok = fabsf((float)e - want) <= 1.0f && (got >> 24) + 1 >= alpha && (got >> 24) <= alpha + 1;
            CT_CHECKF(t, ok, "texel (%u, %u) is %08x: red %u, expected %.1f, alpha %u (%s)", x, y, got, e, want,
                      alpha, y < H / 2 ? "written" : "blended in linear space");
            if (!ok)
                y = H, x = W; /* one message */
        }
    vkDestroyPipeline(s.e.device, plain, NULL);
    vkDestroyPipeline(s.e.device, blended, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(logic_op,
        "The Framebuffer > Logical Operations: with logicOpEnable (logicOp), the color is the bitwise logicOp "
        "of the source (converted to the attachment format) and the stored value, blending skipped "
        "(VK_LOGIC_OP_XOR on R8G8B8A8_UNORM)",
        FB_COMMANDS)
{
    static const float src[4] = { 1.0f, 0.0f, 1.0f, 0.0f };
    float dst[4];
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkPipeline pipe;
    VkCommandBuffer cb;
    uint32_t want = CT_RGBA(0x0f, 0xf0, 0x0f, 0xf0);

    ct_scene_init(t, &s);
    if (!s.f.f2.features.logicOp)
        ct_skip(t, "logicOp is not supported");
    dst[0] = dst[1] = dst[2] = dst[3] = 240.0f / 255.0f;
    ct_scene_color(t, &s, &c);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.cb.logicOpEnable = VK_TRUE;
    g.cb.logicOp = VK_LOGIC_OP_XOR;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, dst, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, src, 0.0f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, expect_uniform, &want, 0, "0xf0 XOR (0xff, 0, 0xff, 0)");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

static uint32_t logic_op_result(VkLogicOp op, uint32_t s, uint32_t d)
{
    switch (op) {
    case VK_LOGIC_OP_CLEAR: return 0;
    case VK_LOGIC_OP_AND: return s & d;
    case VK_LOGIC_OP_AND_REVERSE: return s & ~d;
    case VK_LOGIC_OP_COPY: return s;
    case VK_LOGIC_OP_AND_INVERTED: return ~s & d;
    case VK_LOGIC_OP_NO_OP: return d;
    case VK_LOGIC_OP_XOR: return s ^ d;
    case VK_LOGIC_OP_OR: return s | d;
    case VK_LOGIC_OP_NOR: return ~(s | d);
    case VK_LOGIC_OP_EQUIVALENT: return ~(s ^ d);
    case VK_LOGIC_OP_INVERT: return ~d;
    case VK_LOGIC_OP_OR_REVERSE: return s | ~d;
    case VK_LOGIC_OP_COPY_INVERTED: return ~s;
    case VK_LOGIC_OP_OR_INVERTED: return ~s | d;
    case VK_LOGIC_OP_NAND: return ~(s & d);
    default: return ~0u; /* SET */
    }
}

CT_TEST(logic_op_uint_ops,
        "The Framebuffer > Logical Operations: each of the 16 VkLogicOp values combines the source and the "
        "stored value bitwise as its table defines (logicOp), on R8G8B8A8_UINT, and the color write mask "
        "applies to the result (odd operations write R and B only)",
        FB_COMMANDS)
{
    static const uint32_t src[4] = { 0xcc, 0x55, 0xf0, 0x0f }, dst[4] = { 0xf0, 0x3c, 0xaa, 0x0f };
    float dst_bits[4];
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    struct ct_rect_pc pc;
    VkShaderModule fs;
    VkPipeline pipes[16];
    VkCommandBuffer cb;
    uint32_t k, x, y, ch;
    int bad = 0;

    ct_scene_init(t, &s);
    if (!s.f.f2.features.logicOp)
        ct_skip(t, "logicOp is not supported");
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UINT, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &c);
    fs = ct_shader_module(t, &s.e, uint_frag, sizeof(uint_frag));
    for (k = 0; k < 16; k++) {
        ct_gfx_init(&g, s.vs, fs, s.layout, W, H);
        g.color_formats[0] = VK_FORMAT_R8G8B8A8_UINT;
        g.cb.logicOpEnable = VK_TRUE;
        g.cb.logicOp = (VkLogicOp)k;
        g.blend[0].colorWriteMask = k & 1 ? VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_B_BIT : 0xf;
        pipes[k] = ct_gfx_pipeline(t, &s.e, &g);
    }
    memcpy(dst_bits, dst, sizeof(dst_bits)); /* the clear value's uint32 members */
    memset(&pc, 0, sizeof(pc));
    memcpy(pc.color, src, sizeof(pc.color)); /* the shader writes their bits */
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, dst_bits, NULL, 0.0f);
    for (k = 0; k < 16; k++) { /* operation k in the 4x4 cell at column k % 4, row k / 4 */
        pc.rect[0] = -1.0f + 0.5f * (float)(k % 4);
        pc.rect[1] = -1.0f + 0.5f * (float)(k / 4);
        pc.rect[2] = pc.rect[0] + 0.5f;
        pc.rect[3] = pc.rect[1] + 0.5f;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[k]);
        vkCmdPushConstants(cb, s.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
        vkCmdDraw(cb, 4, 1, 0, 0);
    }
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++) {
            uint32_t op = (y / 4) * 4 + x / 4, got = ct_texel32(s.out.data, W, x, y), want = 0;
            for (ch = 0; ch < 4; ch++) {
                uint32_t v = (op & 1) && (ch == 1 || ch == 3) ? dst[ch] : logic_op_result((VkLogicOp)op, src[ch], dst[ch]);
                want |= (v & 0xffu) << (8 * ch);
            }
            if (got != want && bad++ < 4)
                CT_CHECKF(t, got == want, "texel (%u, %u), operation %u: %08x, expected %08x", x, y, op, got, want);
        }
    CT_CHECKF(t, bad == 0, "%d texels differ", bad);
    for (k = 0; k < 16; k++)
        vkDestroyPipeline(s.e.device, pipes[k], NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

static uint32_t expect_snorm_bits(uint32_t x, uint32_t y, const void *ctx)
{
    (void)x;
    (void)ctx;
    return y < H / 2 ? CT_RGBA(0xff, 0x01, 0x80, 0xc0) : CT_RGBA(0x80, 0x7e, 0xff, 0xbf);
}

CT_TEST(logic_op_snorm_bits,
        "The Framebuffer > Logical Operations: the source is converted to the attachment's format (Fixed-Point "
        "Data Conversions) and the operation applies to the bits, on R8G8B8A8_SNORM too: COPY_INVERTED of "
        "(1, -1, 0, 64/127) stores (0x80, 0x7e, 0xff, 0xbf), and XOR 0x7f of a stored 0x80 (-128) in a later "
        "render pass gives 0xff, read back by copying the image to a buffer",
        FB_COMMANDS)
{
    float clear[4] = { 1.0f, -1.0f, 64.0f / 127.0f, 0.0f }, src[4] = { 1.0f, -1.0f, 0.0f, 64.0f / 127.0f };
    static const float sevens[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkPipeline inverted, xor_pipe;
    VkCommandBuffer cb;

    ct_scene_init(t, &s);
    if (!s.f.f2.features.logicOp)
        ct_skip(t, "logicOp is not supported");
    if (!(ct_format_features(&s.e, VK_FORMAT_R8G8B8A8_SNORM, 0) & VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT))
        ct_skip(t, "R8G8B8A8_SNORM is not a color attachment format");
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_SNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &c);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.color_formats[0] = VK_FORMAT_R8G8B8A8_SNORM;
    g.cb.logicOpEnable = VK_TRUE;
    g.cb.logicOp = VK_LOGIC_OP_COPY_INVERTED;
    inverted = ct_gfx_pipeline(t, &s.e, &g);
    g.cb.logicOp = VK_LOGIC_OP_XOR;
    xor_pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, clear, NULL, 0.0f); /* (0x7f, 0x81, 0x40, 0x00) */
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, inverted);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, src, 0.0f); /* ~(0x7f, 0x81, 0x00, 0x40) */
    vkCmdEndRendering(cb);
    ct_begin_rendering(cb, &c, 1, NULL, NULL, 0.0f); /* loaded */
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, xor_pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 0.0f, sevens, 0.0f); /* top half: ^ 0x7f */
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, expect_snorm_bits, NULL, 0,
                   "top: (0x80, 0x7e, 0xff, 0xbf) ^ 0x7f; bottom: ~(0x7f, 0x81, 0x00, 0x40)");
    vkDestroyPipeline(s.e.device, inverted, NULL);
    vkDestroyPipeline(s.e.device, xor_pipe, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

static const struct ct_test *const tests[] = {
    CT_REF(blend_equations),
    CT_REF(color_write_mask),
    CT_REF(independent_blend),
    CT_REF(dual_source_blend),
    CT_REF(srgb_blending),
    CT_REF(logic_op),
    CT_REF(logic_op_uint_ops),
    CT_REF(logic_op_snorm_bits),
};

CT_SUITE(framebuffer, tests)
