/* Chapters "Fixed-Function Vertex Post-Processing" and "Rasterization": the
 * viewport transformation (offsets, negative heights, depth ranges),
 * clipping and depth clamping, multiple viewports and layers selected by
 * the vertex shader, polygon culling, rasterizer discard, multisample
 * coverage at the standard sample locations, sample shading, wide points,
 * flat (provoking vertex) and smooth interpolation.
 *
 * Shaders: GLSL in the comments, compiled with
 *   glslangValidator -V --target-env vulkan1.3 -x */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

#define W CT_SIZE
#define H CT_SIZE

#define RA_COMMANDS                                                                                                \
    CT_RENDER_COMMANDS "vkCmdBindPipeline vkCmdDraw vkCmdBeginRendering vkCmdEndRendering "

/*
 * #version 450
 * #extension GL_ARB_shader_viewport_layer_array : require
 * // ct_rect_vert's rectangle, to the viewport the push constants name.
 * layout(push_constant) uniform P { vec4 rect; vec4 color; float depth; int layer; int viewport; } pc;
 * void main()
 * {
 *     vec2 c = vec2((gl_VertexIndex & 1) != 0 ? pc.rect.z : pc.rect.x,
 *                   (gl_VertexIndex & 2) != 0 ? pc.rect.w : pc.rect.y);
 *     gl_Position = vec4(c, pc.depth, 1.0);
 *     gl_ViewportIndex = pc.viewport;
 * }
 */
static const uint32_t viewport_vert[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000048, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x00000046, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x00000000, 0x00000001, 0x0009000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000c,
    0x0000001a, 0x00000038, 0x00000043, 0x00030003, 0x00000002, 0x000001c2, 0x000a0004, 0x415f4c47,
    0x735f4252, 0x65646168, 0x69765f72, 0x6f707765, 0x6c5f7472, 0x72657961, 0x7272615f, 0x00007961,
    0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000009, 0x00000063, 0x00060005,
    0x0000000c, 0x565f6c67, 0x65747265, 0x646e4978, 0x00007865, 0x00030005, 0x00000018, 0x00000050,
    0x00050006, 0x00000018, 0x00000000, 0x74636572, 0x00000000, 0x00050006, 0x00000018, 0x00000001,
    0x6f6c6f63, 0x00000072, 0x00050006, 0x00000018, 0x00000002, 0x74706564, 0x00000068, 0x00050006,
    0x00000018, 0x00000003, 0x6579616c, 0x00000072, 0x00060006, 0x00000018, 0x00000004, 0x77656976,
    0x74726f70, 0x00000000, 0x00030005, 0x0000001a, 0x00006370, 0x00060005, 0x00000036, 0x505f6c67,
    0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x00000036, 0x00000000, 0x505f6c67, 0x7469736f,
    0x006e6f69, 0x00070006, 0x00000036, 0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953, 0x00000000,
    0x00070006, 0x00000036, 0x00000002, 0x435f6c67, 0x4470696c, 0x61747369, 0x0065636e, 0x00070006,
    0x00000036, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e, 0x00030005, 0x00000038,
    0x00000000, 0x00070005, 0x00000043, 0x565f6c67, 0x70776569, 0x4974726f, 0x7865646e, 0x00000000,
    0x00040047, 0x0000000c, 0x0000000b, 0x0000002a, 0x00030047, 0x00000018, 0x00000002, 0x00050048,
    0x00000018, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000018, 0x00000001, 0x00000023,
    0x00000010, 0x00050048, 0x00000018, 0x00000002, 0x00000023, 0x00000020, 0x00050048, 0x00000018,
    0x00000003, 0x00000023, 0x00000024, 0x00050048, 0x00000018, 0x00000004, 0x00000023, 0x00000028,
    0x00030047, 0x00000036, 0x00000002, 0x00050048, 0x00000036, 0x00000000, 0x0000000b, 0x00000000,
    0x00050048, 0x00000036, 0x00000001, 0x0000000b, 0x00000001, 0x00050048, 0x00000036, 0x00000002,
    0x0000000b, 0x00000003, 0x00050048, 0x00000036, 0x00000003, 0x0000000b, 0x00000004, 0x00040047,
    0x00000043, 0x0000000b, 0x0000000a, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
    0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000002, 0x00040020,
    0x00000008, 0x00000007, 0x00000007, 0x00040015, 0x0000000a, 0x00000020, 0x00000001, 0x00040020,
    0x0000000b, 0x00000001, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c, 0x00000001, 0x0004002b,
    0x0000000a, 0x0000000e, 0x00000001, 0x0004002b, 0x0000000a, 0x00000010, 0x00000000, 0x00020014,
    0x00000011, 0x00040020, 0x00000013, 0x00000007, 0x00000006, 0x00040017, 0x00000017, 0x00000006,
    0x00000004, 0x0007001e, 0x00000018, 0x00000017, 0x00000017, 0x00000006, 0x0000000a, 0x0000000a,
    0x00040020, 0x00000019, 0x00000009, 0x00000018, 0x0004003b, 0x00000019, 0x0000001a, 0x00000009,
    0x00040015, 0x0000001b, 0x00000020, 0x00000000, 0x0004002b, 0x0000001b, 0x0000001c, 0x00000002,
    0x00040020, 0x0000001d, 0x00000009, 0x00000006, 0x0004002b, 0x0000001b, 0x00000021, 0x00000000,
    0x0004002b, 0x0000000a, 0x00000026, 0x00000002, 0x0004002b, 0x0000001b, 0x0000002c, 0x00000003,
    0x0004002b, 0x0000001b, 0x00000030, 0x00000001, 0x0004001c, 0x00000035, 0x00000006, 0x00000030,
    0x0006001e, 0x00000036, 0x00000017, 0x00000006, 0x00000035, 0x00000035, 0x00040020, 0x00000037,
    0x00000003, 0x00000036, 0x0004003b, 0x00000037, 0x00000038, 0x00000003, 0x0004002b, 0x00000006,
    0x0000003c, 0x3f800000, 0x00040020, 0x00000040, 0x00000003, 0x00000017, 0x00040020, 0x00000042,
    0x00000003, 0x0000000a, 0x0004003b, 0x00000042, 0x00000043, 0x00000003, 0x0004002b, 0x0000000a,
    0x00000044, 0x00000004, 0x00040020, 0x00000045, 0x00000009, 0x0000000a, 0x00050036, 0x00000002,
    0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000008, 0x00000009,
    0x00000007, 0x0004003b, 0x00000013, 0x00000014, 0x00000007, 0x0004003b, 0x00000013, 0x00000029,
    0x00000007, 0x0004003d, 0x0000000a, 0x0000000d, 0x0000000c, 0x000500c7, 0x0000000a, 0x0000000f,
    0x0000000d, 0x0000000e, 0x000500ab, 0x00000011, 0x00000012, 0x0000000f, 0x00000010, 0x000300f7,
    0x00000016, 0x00000000, 0x000400fa, 0x00000012, 0x00000015, 0x00000020, 0x000200f8, 0x00000015,
    0x00060041, 0x0000001d, 0x0000001e, 0x0000001a, 0x00000010, 0x0000001c, 0x0004003d, 0x00000006,
    0x0000001f, 0x0000001e, 0x0003003e, 0x00000014, 0x0000001f, 0x000200f9, 0x00000016, 0x000200f8,
    0x00000020, 0x00060041, 0x0000001d, 0x00000022, 0x0000001a, 0x00000010, 0x00000021, 0x0004003d,
    0x00000006, 0x00000023, 0x00000022, 0x0003003e, 0x00000014, 0x00000023, 0x000200f9, 0x00000016,
    0x000200f8, 0x00000016, 0x0004003d, 0x00000006, 0x00000024, 0x00000014, 0x0004003d, 0x0000000a,
    0x00000025, 0x0000000c, 0x000500c7, 0x0000000a, 0x00000027, 0x00000025, 0x00000026, 0x000500ab,
    0x00000011, 0x00000028, 0x00000027, 0x00000010, 0x000300f7, 0x0000002b, 0x00000000, 0x000400fa,
    0x00000028, 0x0000002a, 0x0000002f, 0x000200f8, 0x0000002a, 0x00060041, 0x0000001d, 0x0000002d,
    0x0000001a, 0x00000010, 0x0000002c, 0x0004003d, 0x00000006, 0x0000002e, 0x0000002d, 0x0003003e,
    0x00000029, 0x0000002e, 0x000200f9, 0x0000002b, 0x000200f8, 0x0000002f, 0x00060041, 0x0000001d,
    0x00000031, 0x0000001a, 0x00000010, 0x00000030, 0x0004003d, 0x00000006, 0x00000032, 0x00000031,
    0x0003003e, 0x00000029, 0x00000032, 0x000200f9, 0x0000002b, 0x000200f8, 0x0000002b, 0x0004003d,
    0x00000006, 0x00000033, 0x00000029, 0x00050050, 0x00000007, 0x00000034, 0x00000024, 0x00000033,
    0x0003003e, 0x00000009, 0x00000034, 0x0004003d, 0x00000007, 0x00000039, 0x00000009, 0x00050041,
    0x0000001d, 0x0000003a, 0x0000001a, 0x00000026, 0x0004003d, 0x00000006, 0x0000003b, 0x0000003a,
    0x00050051, 0x00000006, 0x0000003d, 0x00000039, 0x00000000, 0x00050051, 0x00000006, 0x0000003e,
    0x00000039, 0x00000001, 0x00070050, 0x00000017, 0x0000003f, 0x0000003d, 0x0000003e, 0x0000003b,
    0x0000003c, 0x00050041, 0x00000040, 0x00000041, 0x00000038, 0x00000010, 0x0003003e, 0x00000041,
    0x0000003f, 0x00050041, 0x00000045, 0x00000046, 0x0000001a, 0x00000044, 0x0004003d, 0x0000000a,
    0x00000047, 0x00000046, 0x0003003e, 0x00000043, 0x00000047, 0x000100fd, 0x00010038,
};

/*
 * #version 450
 * #extension GL_ARB_shader_viewport_layer_array : require
 * // ct_rect_vert's rectangle, to the layer the push constants name.
 * layout(push_constant) uniform P { vec4 rect; vec4 color; float depth; int layer; int viewport; } pc;
 * void main()
 * {
 *     vec2 c = vec2((gl_VertexIndex & 1) != 0 ? pc.rect.z : pc.rect.x,
 *                   (gl_VertexIndex & 2) != 0 ? pc.rect.w : pc.rect.y);
 *     gl_Position = vec4(c, pc.depth, 1.0);
 *     gl_Layer = pc.layer;
 * }
 */
static const uint32_t layer_vert[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000048, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x00000045, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x00000000, 0x00000001, 0x0009000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000c,
    0x0000001a, 0x00000038, 0x00000043, 0x00030003, 0x00000002, 0x000001c2, 0x000a0004, 0x415f4c47,
    0x735f4252, 0x65646168, 0x69765f72, 0x6f707765, 0x6c5f7472, 0x72657961, 0x7272615f, 0x00007961,
    0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000009, 0x00000063, 0x00060005,
    0x0000000c, 0x565f6c67, 0x65747265, 0x646e4978, 0x00007865, 0x00030005, 0x00000018, 0x00000050,
    0x00050006, 0x00000018, 0x00000000, 0x74636572, 0x00000000, 0x00050006, 0x00000018, 0x00000001,
    0x6f6c6f63, 0x00000072, 0x00050006, 0x00000018, 0x00000002, 0x74706564, 0x00000068, 0x00050006,
    0x00000018, 0x00000003, 0x6579616c, 0x00000072, 0x00060006, 0x00000018, 0x00000004, 0x77656976,
    0x74726f70, 0x00000000, 0x00030005, 0x0000001a, 0x00006370, 0x00060005, 0x00000036, 0x505f6c67,
    0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x00000036, 0x00000000, 0x505f6c67, 0x7469736f,
    0x006e6f69, 0x00070006, 0x00000036, 0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953, 0x00000000,
    0x00070006, 0x00000036, 0x00000002, 0x435f6c67, 0x4470696c, 0x61747369, 0x0065636e, 0x00070006,
    0x00000036, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e, 0x00030005, 0x00000038,
    0x00000000, 0x00050005, 0x00000043, 0x4c5f6c67, 0x72657961, 0x00000000, 0x00040047, 0x0000000c,
    0x0000000b, 0x0000002a, 0x00030047, 0x00000018, 0x00000002, 0x00050048, 0x00000018, 0x00000000,
    0x00000023, 0x00000000, 0x00050048, 0x00000018, 0x00000001, 0x00000023, 0x00000010, 0x00050048,
    0x00000018, 0x00000002, 0x00000023, 0x00000020, 0x00050048, 0x00000018, 0x00000003, 0x00000023,
    0x00000024, 0x00050048, 0x00000018, 0x00000004, 0x00000023, 0x00000028, 0x00030047, 0x00000036,
    0x00000002, 0x00050048, 0x00000036, 0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x00000036,
    0x00000001, 0x0000000b, 0x00000001, 0x00050048, 0x00000036, 0x00000002, 0x0000000b, 0x00000003,
    0x00050048, 0x00000036, 0x00000003, 0x0000000b, 0x00000004, 0x00040047, 0x00000043, 0x0000000b,
    0x00000009, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
    0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000002, 0x00040020, 0x00000008, 0x00000007,
    0x00000007, 0x00040015, 0x0000000a, 0x00000020, 0x00000001, 0x00040020, 0x0000000b, 0x00000001,
    0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c, 0x00000001, 0x0004002b, 0x0000000a, 0x0000000e,
    0x00000001, 0x0004002b, 0x0000000a, 0x00000010, 0x00000000, 0x00020014, 0x00000011, 0x00040020,
    0x00000013, 0x00000007, 0x00000006, 0x00040017, 0x00000017, 0x00000006, 0x00000004, 0x0007001e,
    0x00000018, 0x00000017, 0x00000017, 0x00000006, 0x0000000a, 0x0000000a, 0x00040020, 0x00000019,
    0x00000009, 0x00000018, 0x0004003b, 0x00000019, 0x0000001a, 0x00000009, 0x00040015, 0x0000001b,
    0x00000020, 0x00000000, 0x0004002b, 0x0000001b, 0x0000001c, 0x00000002, 0x00040020, 0x0000001d,
    0x00000009, 0x00000006, 0x0004002b, 0x0000001b, 0x00000021, 0x00000000, 0x0004002b, 0x0000000a,
    0x00000026, 0x00000002, 0x0004002b, 0x0000001b, 0x0000002c, 0x00000003, 0x0004002b, 0x0000001b,
    0x00000030, 0x00000001, 0x0004001c, 0x00000035, 0x00000006, 0x00000030, 0x0006001e, 0x00000036,
    0x00000017, 0x00000006, 0x00000035, 0x00000035, 0x00040020, 0x00000037, 0x00000003, 0x00000036,
    0x0004003b, 0x00000037, 0x00000038, 0x00000003, 0x0004002b, 0x00000006, 0x0000003c, 0x3f800000,
    0x00040020, 0x00000040, 0x00000003, 0x00000017, 0x00040020, 0x00000042, 0x00000003, 0x0000000a,
    0x0004003b, 0x00000042, 0x00000043, 0x00000003, 0x0004002b, 0x0000000a, 0x00000044, 0x00000003,
    0x00040020, 0x00000045, 0x00000009, 0x0000000a, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000008, 0x00000009, 0x00000007, 0x0004003b,
    0x00000013, 0x00000014, 0x00000007, 0x0004003b, 0x00000013, 0x00000029, 0x00000007, 0x0004003d,
    0x0000000a, 0x0000000d, 0x0000000c, 0x000500c7, 0x0000000a, 0x0000000f, 0x0000000d, 0x0000000e,
    0x000500ab, 0x00000011, 0x00000012, 0x0000000f, 0x00000010, 0x000300f7, 0x00000016, 0x00000000,
    0x000400fa, 0x00000012, 0x00000015, 0x00000020, 0x000200f8, 0x00000015, 0x00060041, 0x0000001d,
    0x0000001e, 0x0000001a, 0x00000010, 0x0000001c, 0x0004003d, 0x00000006, 0x0000001f, 0x0000001e,
    0x0003003e, 0x00000014, 0x0000001f, 0x000200f9, 0x00000016, 0x000200f8, 0x00000020, 0x00060041,
    0x0000001d, 0x00000022, 0x0000001a, 0x00000010, 0x00000021, 0x0004003d, 0x00000006, 0x00000023,
    0x00000022, 0x0003003e, 0x00000014, 0x00000023, 0x000200f9, 0x00000016, 0x000200f8, 0x00000016,
    0x0004003d, 0x00000006, 0x00000024, 0x00000014, 0x0004003d, 0x0000000a, 0x00000025, 0x0000000c,
    0x000500c7, 0x0000000a, 0x00000027, 0x00000025, 0x00000026, 0x000500ab, 0x00000011, 0x00000028,
    0x00000027, 0x00000010, 0x000300f7, 0x0000002b, 0x00000000, 0x000400fa, 0x00000028, 0x0000002a,
    0x0000002f, 0x000200f8, 0x0000002a, 0x00060041, 0x0000001d, 0x0000002d, 0x0000001a, 0x00000010,
    0x0000002c, 0x0004003d, 0x00000006, 0x0000002e, 0x0000002d, 0x0003003e, 0x00000029, 0x0000002e,
    0x000200f9, 0x0000002b, 0x000200f8, 0x0000002f, 0x00060041, 0x0000001d, 0x00000031, 0x0000001a,
    0x00000010, 0x00000030, 0x0004003d, 0x00000006, 0x00000032, 0x00000031, 0x0003003e, 0x00000029,
    0x00000032, 0x000200f9, 0x0000002b, 0x000200f8, 0x0000002b, 0x0004003d, 0x00000006, 0x00000033,
    0x00000029, 0x00050050, 0x00000007, 0x00000034, 0x00000024, 0x00000033, 0x0003003e, 0x00000009,
    0x00000034, 0x0004003d, 0x00000007, 0x00000039, 0x00000009, 0x00050041, 0x0000001d, 0x0000003a,
    0x0000001a, 0x00000026, 0x0004003d, 0x00000006, 0x0000003b, 0x0000003a, 0x00050051, 0x00000006,
    0x0000003d, 0x00000039, 0x00000000, 0x00050051, 0x00000006, 0x0000003e, 0x00000039, 0x00000001,
    0x00070050, 0x00000017, 0x0000003f, 0x0000003d, 0x0000003e, 0x0000003b, 0x0000003c, 0x00050041,
    0x00000040, 0x00000041, 0x00000038, 0x00000010, 0x0003003e, 0x00000041, 0x0000003f, 0x00050041,
    0x00000045, 0x00000046, 0x0000001a, 0x00000044, 0x0004003d, 0x0000000a, 0x00000047, 0x00000046,
    0x0003003e, 0x00000043, 0x00000047, 0x000100fd, 0x00010038,
};

/*
 * #version 450
 * // A position and a color per vertex.
 * layout(location = 0) in vec2 pos;
 * layout(location = 1) in vec4 color;
 * layout(location = 0) out vec4 v_color;
 * void main()
 * {
 *     gl_Position = vec4(pos, 0.0, 1.0);
 *     v_color = color;
 * }
 */
static const uint32_t vcolor_vert[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x0000001f, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0009000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00000012, 0x0000001b,
    0x0000001d, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00060005, 0x0000000b, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x0000000b,
    0x00000000, 0x505f6c67, 0x7469736f, 0x006e6f69, 0x00070006, 0x0000000b, 0x00000001, 0x505f6c67,
    0x746e696f, 0x657a6953, 0x00000000, 0x00070006, 0x0000000b, 0x00000002, 0x435f6c67, 0x4470696c,
    0x61747369, 0x0065636e, 0x00070006, 0x0000000b, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369,
    0x0065636e, 0x00030005, 0x0000000d, 0x00000000, 0x00030005, 0x00000012, 0x00736f70, 0x00040005,
    0x0000001b, 0x6f635f76, 0x00726f6c, 0x00040005, 0x0000001d, 0x6f6c6f63, 0x00000072, 0x00030047,
    0x0000000b, 0x00000002, 0x00050048, 0x0000000b, 0x00000000, 0x0000000b, 0x00000000, 0x00050048,
    0x0000000b, 0x00000001, 0x0000000b, 0x00000001, 0x00050048, 0x0000000b, 0x00000002, 0x0000000b,
    0x00000003, 0x00050048, 0x0000000b, 0x00000003, 0x0000000b, 0x00000004, 0x00040047, 0x00000012,
    0x0000001e, 0x00000000, 0x00040047, 0x0000001b, 0x0000001e, 0x00000000, 0x00040047, 0x0000001d,
    0x0000001e, 0x00000001, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
    0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040015, 0x00000008,
    0x00000020, 0x00000000, 0x0004002b, 0x00000008, 0x00000009, 0x00000001, 0x0004001c, 0x0000000a,
    0x00000006, 0x00000009, 0x0006001e, 0x0000000b, 0x00000007, 0x00000006, 0x0000000a, 0x0000000a,
    0x00040020, 0x0000000c, 0x00000003, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000003,
    0x00040015, 0x0000000e, 0x00000020, 0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000,
    0x00040017, 0x00000010, 0x00000006, 0x00000002, 0x00040020, 0x00000011, 0x00000001, 0x00000010,
    0x0004003b, 0x00000011, 0x00000012, 0x00000001, 0x0004002b, 0x00000006, 0x00000014, 0x00000000,
    0x0004002b, 0x00000006, 0x00000015, 0x3f800000, 0x00040020, 0x00000019, 0x00000003, 0x00000007,
    0x0004003b, 0x00000019, 0x0000001b, 0x00000003, 0x00040020, 0x0000001c, 0x00000001, 0x00000007,
    0x0004003b, 0x0000001c, 0x0000001d, 0x00000001, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x00000010, 0x00000013, 0x00000012, 0x00050051,
    0x00000006, 0x00000016, 0x00000013, 0x00000000, 0x00050051, 0x00000006, 0x00000017, 0x00000013,
    0x00000001, 0x00070050, 0x00000007, 0x00000018, 0x00000016, 0x00000017, 0x00000014, 0x00000015,
    0x00050041, 0x00000019, 0x0000001a, 0x0000000d, 0x0000000f, 0x0003003e, 0x0000001a, 0x00000018,
    0x0004003d, 0x00000007, 0x0000001e, 0x0000001d, 0x0003003e, 0x0000001b, 0x0000001e, 0x000100fd,
    0x00010038,
};

/*
 * #version 450
 * layout(location = 0) in vec4 v_color;
 * layout(location = 0) out vec4 o_color;
 * void main()
 * {
 *     o_color = v_color;
 * }
 */
static const uint32_t smooth_frag[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x0000000d, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000b, 0x00030010,
    0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00040005, 0x00000009, 0x6f635f6f, 0x00726f6c, 0x00040005, 0x0000000b, 0x6f635f76,
    0x00726f6c, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00040047, 0x0000000b, 0x0000001e,
    0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
    0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003,
    0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00040020, 0x0000000a, 0x00000001,
    0x00000007, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x00000007, 0x0000000c, 0x0000000b,
    0x0003003e, 0x00000009, 0x0000000c, 0x000100fd, 0x00010038,
};

/*
 * #version 450
 * layout(location = 0) flat in vec4 v_color;
 * layout(location = 0) out vec4 o_color;
 * void main()
 * {
 *     o_color = v_color;
 * }
 */
static const uint32_t flat_frag[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x0000000d, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000b, 0x00030010,
    0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00040005, 0x00000009, 0x6f635f6f, 0x00726f6c, 0x00040005, 0x0000000b, 0x6f635f76,
    0x00726f6c, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00030047, 0x0000000b, 0x0000000e,
    0x00040047, 0x0000000b, 0x0000001e, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
    0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004,
    0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003,
    0x00040020, 0x0000000a, 0x00000001, 0x00000007, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d,
    0x00000007, 0x0000000c, 0x0000000b, 0x0003003e, 0x00000009, 0x0000000c, 0x000100fd, 0x00010038,
};

/*
 * #version 450
 * // A point of size 4 at the center of the target.
 * void main()
 * {
 *     gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
 *     gl_PointSize = 4.0;
 * }
 */
static const uint32_t bigpoint_vert[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000019, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00030003, 0x00000002,
    0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00060005, 0x0000000b, 0x505f6c67,
    0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x0000000b, 0x00000000, 0x505f6c67, 0x7469736f,
    0x006e6f69, 0x00070006, 0x0000000b, 0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953, 0x00000000,
    0x00070006, 0x0000000b, 0x00000002, 0x435f6c67, 0x4470696c, 0x61747369, 0x0065636e, 0x00070006,
    0x0000000b, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e, 0x00030005, 0x0000000d,
    0x00000000, 0x00030047, 0x0000000b, 0x00000002, 0x00050048, 0x0000000b, 0x00000000, 0x0000000b,
    0x00000000, 0x00050048, 0x0000000b, 0x00000001, 0x0000000b, 0x00000001, 0x00050048, 0x0000000b,
    0x00000002, 0x0000000b, 0x00000003, 0x00050048, 0x0000000b, 0x00000003, 0x0000000b, 0x00000004,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
    0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040015, 0x00000008, 0x00000020, 0x00000000,
    0x0004002b, 0x00000008, 0x00000009, 0x00000001, 0x0004001c, 0x0000000a, 0x00000006, 0x00000009,
    0x0006001e, 0x0000000b, 0x00000007, 0x00000006, 0x0000000a, 0x0000000a, 0x00040020, 0x0000000c,
    0x00000003, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000003, 0x00040015, 0x0000000e,
    0x00000020, 0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000, 0x0004002b, 0x00000006,
    0x00000010, 0x00000000, 0x0004002b, 0x00000006, 0x00000011, 0x3f800000, 0x0007002c, 0x00000007,
    0x00000012, 0x00000010, 0x00000010, 0x00000010, 0x00000011, 0x00040020, 0x00000013, 0x00000003,
    0x00000007, 0x0004002b, 0x0000000e, 0x00000015, 0x00000001, 0x0004002b, 0x00000006, 0x00000016,
    0x40800000, 0x00040020, 0x00000017, 0x00000003, 0x00000006, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041, 0x00000013, 0x00000014, 0x0000000d,
    0x0000000f, 0x0003003e, 0x00000014, 0x00000012, 0x00050041, 0x00000017, 0x00000018, 0x0000000d,
    0x00000015, 0x0003003e, 0x00000018, 0x00000016, 0x000100fd, 0x00010038,
};

/*
 * #version 450
 * // Red for sample 0, black for the others.
 * layout(location = 0) out vec4 o_color;
 * void main()
 * {
 *     o_color = vec4(gl_SampleID == 0 ? 1.0 : 0.0, 0.0, 0.0, 1.0);
 * }
 */
static const uint32_t sample_frag[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000015, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x00000023, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x00000000, 0x00000001, 0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009,
    0x0000000c, 0x00030010, 0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f635f6f, 0x00726f6c, 0x00050005,
    0x0000000c, 0x535f6c67, 0x6c706d61, 0x00444965, 0x00040047, 0x00000009, 0x0000001e, 0x00000000,
    0x00040047, 0x0000000c, 0x0000000b, 0x00000012, 0x00030047, 0x0000000c, 0x0000000e, 0x00020013,
    0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017,
    0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b,
    0x00000008, 0x00000009, 0x00000003, 0x00040015, 0x0000000a, 0x00000020, 0x00000001, 0x00040020,
    0x0000000b, 0x00000001, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c, 0x00000001, 0x0004002b,
    0x0000000a, 0x0000000e, 0x00000000, 0x00020014, 0x0000000f, 0x0004002b, 0x00000006, 0x00000011,
    0x3f800000, 0x0004002b, 0x00000006, 0x00000012, 0x00000000, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x0000000a, 0x0000000d, 0x0000000c,
    0x000500aa, 0x0000000f, 0x00000010, 0x0000000d, 0x0000000e, 0x000600a9, 0x00000006, 0x00000013,
    0x00000010, 0x00000011, 0x00000012, 0x00070050, 0x00000007, 0x00000014, 0x00000013, 0x00000012,
    0x00000012, 0x00000011, 0x0003003e, 0x00000009, 0x00000014, 0x000100fd, 0x00010038,
};
static VkViewport viewport(float x, float y, float w, float h, float min_depth, float max_depth)
{
    VkViewport v;
    v.x = x;
    v.y = y;
    v.width = w;
    v.height = h;
    v.minDepth = min_depth;
    v.maxDepth = max_depth;
    return v;
}

static uint32_t expect_viewports(uint32_t x, uint32_t y, const void *ctx)
{
    (void)ctx;
    if (x >= 4 && x < 12 && y >= 2 && y < 10)
        return CT_RED; /* the NDC square in viewport (4, 2) 8 x 8 */
    if (x < 2 && y >= 8)
        return CT_GREEN; /* NDC x -1..-0.75, y -1..0, through a viewport of height -16: the bottom half */
    return CT_BLACK;
}

CT_TEST(viewport_transform,
        "Fixed-Function Vertex Post-Processing > Controlling the Viewport: framebuffer coordinates are "
        "x = (px / 2) xd + ox, y = (py / 2) yd + oy with ox, oy the viewport's center and px, py its width and "
        "height; a negative height (Vulkan 1.1) flips y",
        RA_COMMANDS)
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkPipeline offset, flipped;
    VkCommandBuffer cb;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.viewports[0] = viewport(4.0f, 2.0f, 8.0f, 8.0f, 0.0f, 1.0f);
    offset = ct_gfx_pipeline(t, &s.e, &g);
    g.viewports[0] = viewport(0.0f, (float)H, (float)W, -(float)H, 0.0f, 1.0f);
    flipped = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, offset);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, ct_red, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, flipped);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, -0.75f, 0.0f, ct_green, 0.0f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, expect_viewports, NULL, 0, "color");
    vkDestroyPipeline(s.e.device, offset, NULL);
    vkDestroyPipeline(s.e.device, flipped, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

static uint16_t expect_depth_range(uint32_t x, uint32_t y)
{
    static const uint16_t z[4] = { 16384, 32768, 49151, 65535 };
    (void)y;
    return z[x / 4];
}

CT_TEST(viewport_depth_range,
        "Fixed-Function Vertex Post-Processing > Controlling the Viewport: framebuffer depth is "
        "zf = pz zd + oz with oz = minDepth and pz = maxDepth - minDepth (depth 0, 0.5 and 1 in [0.25, 0.75])",
        RA_COMMANDS)
{
    struct ct_scene s;
    struct ct_target d;
    struct ct_gfx g;
    VkPipeline pipe;
    VkCommandBuffer cb;

    ct_scene_init(t, &s);
    ct_target(t, &s.e, VK_FORMAT_D16_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              &d);
    ct_gfx_init(&g, s.vs, VK_NULL_HANDLE, s.layout, W, H);
    g.colors = 0;
    g.viewports[0].minDepth = 0.25f;
    g.viewports[0].maxDepth = 0.75f;
    g.rendering.depthAttachmentFormat = VK_FORMAT_D16_UNORM;
    g.ds.depthTestEnable = g.ds.depthWriteEnable = VK_TRUE;
    g.ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, NULL, 0, NULL, &d, 1.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, -0.5f, 1.0f, NULL, 0.0f);
    ct_draw_rect(cb, s.layout, -0.5f, -1.0f, 0.0f, 1.0f, NULL, 0.5f);
    ct_draw_rect(cb, s.layout, 0.0f, -1.0f, 0.5f, 1.0f, NULL, 1.0f);
    vkCmdEndRendering(cb);
    ct_target_read(cb, &d, 0, VK_IMAGE_ASPECT_DEPTH_BIT, s.out.buffer, 0);
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_d16(t, s.out.data, W, H, expect_depth_range, "depth");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_target_fini(&s.e, &d);
    ct_scene_fini(&s);
}

static uint16_t expect_clamped(uint32_t x, uint32_t y)
{
    (void)y;
    return x < W / 2 ? 0 : 65535; /* depth -0.5 and 1.5 clamped to the viewport's 0..1 */
}

CT_TEST(clipping_depth_clamp,
        "Fixed-Function Vertex Post-Processing > Primitive Clipping: primitives outside 0 <= zc <= wc are "
        "clipped unless depth clamping is enabled; with depthClampEnable (depthClamp) they are not clipped in z "
        "and fragment depth is clamped to the viewport's depth range",
        RA_COMMANDS)
{
    struct ct_scene s;
    struct ct_target c, d;
    struct ct_gfx g;
    VkPipeline clipped, clamped = VK_NULL_HANDLE;
    VkCommandBuffer cb;
    struct ct_box nothing = { 0, 0, 0, 0, CT_BLACK, CT_BLACK };

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    ct_target(t, &s.e, VK_FORMAT_D16_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              &d);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.rendering.depthAttachmentFormat = VK_FORMAT_D16_UNORM;
    g.ds.depthTestEnable = g.ds.depthWriteEnable = VK_TRUE;
    g.ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    clipped = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, &d, 0.5f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, clipped);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 1.0f, ct_red, -0.5f);
    ct_draw_rect(cb, s.layout, 0.0f, -1.0f, 1.0f, 1.0f, ct_red, 1.5f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &nothing, 0, "clipped in z: nothing drawn");
    if (s.f.f2.features.depthClamp) {
        g.rs.depthClampEnable = VK_TRUE;
        clamped = ct_gfx_pipeline(t, &s.e, &g);
        cb = ct_begin_commands(t, &s.e);
        ct_begin_rendering(cb, &c, 1, ct_black, &d, 0.5f);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, clamped);
        ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 1.0f, ct_red, -0.5f);
        ct_draw_rect(cb, s.layout, 0.0f, -1.0f, 1.0f, 1.0f, ct_red, 1.5f);
        vkCmdEndRendering(cb);
        ct_target_read(cb, &d, 0, VK_IMAGE_ASPECT_DEPTH_BIT, s.out.buffer, 0);
        ct_submit_and_wait(t, &s.e, cb);
        ct_check_d16(t, s.out.data, W, H, expect_clamped, "clamped depth");
        vkDestroyPipeline(s.e.device, clamped, NULL);
    } else {
        ct_note(t, "depthClamp is not supported: clamping not checked");
    }
    vkDestroyPipeline(s.e.device, clipped, NULL);
    ct_target_fini(&s.e, &d);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(viewport_index_output,
        "Fixed-Function Vertex Post-Processing > Controlling the Viewport: with several viewports "
        "(multiViewport), a vertex shader writing ViewportIndex (shaderOutputViewportIndex, Vulkan 1.2) selects "
        "the viewport and scissor each primitive uses",
        RA_COMMANDS)
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    struct ct_rect_pc pc;
    VkShaderModule vs;
    VkPipeline pipe;
    VkCommandBuffer cb;
    struct ct_box want = { 0, 0, W / 2, H, CT_GREEN, CT_RED };

    ct_scene_init(t, &s);
    if (!s.f.f2.features.multiViewport || !s.f.v12.shaderOutputViewportIndex)
        ct_skip(t, "multiViewport or shaderOutputViewportIndex is not supported");
    ct_scene_color(t, &s, &c);
    vs = ct_shader_module(t, &s.e, viewport_vert, sizeof(viewport_vert));
    ct_gfx_init(&g, vs, s.fs, s.layout, W, H);
    g.viewport_count = 2;
    g.viewports[0] = viewport(0.0f, 0.0f, W / 2, H, 0.0f, 1.0f);
    g.viewports[1] = viewport(W / 2, 0.0f, W / 2, H, 0.0f, 1.0f);
    g.scissors[1] = g.scissors[0];
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    memset(&pc, 0, sizeof(pc));
    pc.rect[0] = pc.rect[1] = -1.0f;
    pc.rect[2] = pc.rect[3] = 1.0f;
    memcpy(pc.color, ct_red, sizeof(pc.color));
    pc.viewport = 1;
    vkCmdPushConstants(cb, s.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    vkCmdDraw(cb, 4, 1, 0, 0);
    memcpy(pc.color, ct_green, sizeof(pc.color));
    pc.viewport = 0;
    vkCmdPushConstants(cb, s.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    vkCmdDraw(cb, 4, 1, 0, 0);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want, 0, "viewport 0 green, viewport 1 red");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(layer_output,
        "Rasterization, Render Pass > Layered Rendering: a vertex shader writing Layer (shaderOutputLayer, "
        "Vulkan 1.2) selects the framebuffer layer each primitive is rasterized into (a dynamic render pass "
        "instance of layerCount 2)",
        RA_COMMANDS)
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    struct ct_rect_pc pc;
    VkRenderingAttachmentInfo att;
    VkRenderingInfo info;
    VkShaderModule vs;
    VkPipeline pipe;
    VkCommandBuffer cb;
    struct ct_box want0 = { 0, 0, W, H / 2, CT_GREEN, CT_BLACK }, want1 = { 0, H / 2, W, H, CT_RED, CT_BLACK };

    ct_scene_init(t, &s);
    if (!s.f.v12.shaderOutputLayer)
        ct_skip(t, "shaderOutputLayer is not supported");
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 2, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &c);
    vs = ct_shader_module(t, &s.e, layer_vert, sizeof(layer_vert));
    ct_gfx_init(&g, vs, s.fs, s.layout, W, H);
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    memset(&att, 0, sizeof(att));
    att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    att.imageView = c.view;
    att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    memcpy(att.clearValue.color.float32, ct_black, sizeof(ct_black));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    info.renderArea.extent.width = W;
    info.renderArea.extent.height = H;
    info.layerCount = 2;
    info.colorAttachmentCount = 1;
    info.pColorAttachments = &att;
    cb = ct_begin_commands(t, &s.e);
    vkCmdBeginRendering(cb, &info);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    memset(&pc, 0, sizeof(pc));
    pc.rect[0] = -1.0f;
    pc.rect[1] = 0.0f;
    pc.rect[2] = pc.rect[3] = 1.0f;
    memcpy(pc.color, ct_red, sizeof(pc.color));
    pc.layer = 1; /* the bottom half of layer 1 */
    vkCmdPushConstants(cb, s.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    vkCmdDraw(cb, 4, 1, 0, 0);
    pc.rect[1] = -1.0f;
    pc.rect[3] = 0.0f;
    memcpy(pc.color, ct_green, sizeof(pc.color));
    pc.layer = 0; /* the top half of layer 0 */
    vkCmdPushConstants(cb, s.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    vkCmdDraw(cb, 4, 1, 0, 0);
    vkCmdEndRendering(cb);
    ct_target_read(cb, &c, 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, 0);
    ct_target_read(cb, &c, 1, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, W * H * 4);
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want0, 0, "layer 0");
    ct_check_rgba8(t, s.out.data + W * H * 4, W, H, ct_expect_box, &want1, 0, "layer 1");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(polygon_culling,
        "Rasterization > Basic Polygon Rasterization: a polygon is front-facing when its signed area in "
        "framebuffer coordinates is positive with VK_FRONT_FACE_COUNTER_CLOCKWISE (negative with "
        "VK_FRONT_FACE_CLOCKWISE); cullMode discards back-facing, front-facing or all polygons",
        RA_COMMANDS)
{
    static const uint32_t want[4] = { CT_RED, CT_BLACK, CT_RED, CT_BLACK };
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkPipeline p[4];
    VkCommandBuffer cb;
    static const VkCullModeFlags cull[4] = { VK_CULL_MODE_NONE, VK_CULL_MODE_BACK_BIT, VK_CULL_MODE_FRONT_BIT,
                                             VK_CULL_MODE_FRONT_AND_BACK };
    uint32_t i;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    for (i = 0; i < 4; i++) {
        ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
        g.rs.cullMode = cull[i];
        g.rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        p[i] = ct_gfx_pipeline(t, &s.e, &g);
    }
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    /* ct_rect_vert's strip is clockwise in framebuffer coordinates (y down):
     * back-facing with VK_FRONT_FACE_COUNTER_CLOCKWISE */
    for (i = 0; i < 4; i++) {
        float x0 = i & 1 ? 0.0f : -1.0f, y0 = i & 2 ? 0.0f : -1.0f;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p[i]);
        ct_draw_rect(cb, s.layout, x0, y0, x0 + 1.0f, y0 + 1.0f, ct_red, 0.0f);
    }
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_quadrants, want, 0, "none, back, front, front and back culled");
    for (i = 0; i < 4; i++)
        vkDestroyPipeline(s.e.device, p[i], NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(rasterizer_discard,
        "Rasterization > Discarding Primitives Before Rasterization: with rasterizerDiscardEnable, primitives "
        "are discarded and nothing is written, while the render pass instance's clear still happens (a "
        "pipeline of a vertex shader only: without fragment shader state, VUID-VkGraphicsPipelineCreateInfo-"
        "pStages-06894)",
        RA_COMMANDS "vkCreateRenderPass vkDestroyRenderPass vkCreateFramebuffer vkDestroyFramebuffer "
                    "vkCmdBeginRenderPass vkCmdEndRenderPass")
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkAttachmentDescription a;
    VkAttachmentReference ref;
    VkSubpassDescription sp;
    VkRenderPassCreateInfo rpi;
    VkFramebufferCreateInfo fbi;
    VkRenderPassBeginInfo begin;
    VkRenderPass rp = VK_NULL_HANDLE;
    VkFramebuffer fb = VK_NULL_HANDLE;
    VkClearValue clear;
    VkPipeline pipe;
    VkCommandBuffer cb;
    struct ct_box want = { 0, 0, 0, 0, CT_BLUE, CT_BLUE };

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    memset(&a, 0, sizeof(a));
    a.format = VK_FORMAT_R8G8B8A8_UNORM;
    a.samples = VK_SAMPLE_COUNT_1_BIT;
    a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    a.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
    a.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    ref.attachment = 0;
    ref.layout = VK_IMAGE_LAYOUT_GENERAL;
    memset(&sp, 0, sizeof(sp));
    sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp.colorAttachmentCount = 1;
    sp.pColorAttachments = &ref;
    memset(&rpi, 0, sizeof(rpi));
    rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpi.attachmentCount = 1;
    rpi.pAttachments = &a;
    rpi.subpassCount = 1;
    rpi.pSubpasses = &sp;
    CT_REQUIRE_VK(t, vkCreateRenderPass(s.e.device, &rpi, NULL, &rp));
    memset(&fbi, 0, sizeof(fbi));
    fbi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbi.renderPass = rp;
    fbi.attachmentCount = 1;
    fbi.pAttachments = &c.view;
    fbi.width = W;
    fbi.height = H;
    fbi.layers = 1;
    CT_REQUIRE_VK(t, vkCreateFramebuffer(s.e.device, &fbi, NULL, &fb));
    ct_gfx_init(&g, s.vs, VK_NULL_HANDLE, s.layout, W, H);
    g.render_pass = rp;
    g.rs.rasterizerDiscardEnable = VK_TRUE;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    memset(&clear, 0, sizeof(clear));
    memcpy(clear.color.float32, ct_blue, sizeof(ct_blue));
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = rp;
    begin.framebuffer = fb;
    begin.renderArea.extent.width = W;
    begin.renderArea.extent.height = H;
    begin.clearValueCount = 1;
    begin.pClearValues = &clear;
    cb = ct_begin_commands(t, &s.e);
    vkCmdBeginRenderPass(cb, &begin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, ct_red, 0.0f);
    vkCmdEndRenderPass(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want, 0, "cleared, nothing drawn");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyFramebuffer(s.e.device, fb, NULL);
    vkDestroyRenderPass(s.e.device, rp, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

static uint32_t expect_coverage(uint32_t x, uint32_t y, const void *ctx)
{
    (void)y;
    (void)ctx;
    return x < 8 ? CT_RED : x == 8 ? CT_RGBA(128, 0, 0, 255) : CT_BLACK;
}

/* 4x multisampled rendering into ms, resolved into c by the instance. */
static void begin_msaa(VkCommandBuffer cb, const struct ct_target *ms, const struct ct_target *c)
{
    VkRenderingAttachmentInfo att;
    VkRenderingInfo info;
    memset(&att, 0, sizeof(att));
    att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    att.imageView = ms->view;
    att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    att.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
    att.resolveImageView = c->view;
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
    vkCmdBeginRendering(cb, &info);
}

CT_TEST(multisample_coverage,
        "Rasterization > Multisampling: a sample is covered when its location is inside the primitive; with "
        "standardSampleLocations, 4 samples sit at (0.375, 0.125), (0.875, 0.375), (0.125, 0.625) and "
        "(0.625, 0.875): a polygon edge through the middle of a pixel covers 2 of them, and the resolve averages "
        "the samples",
        RA_COMMANDS)
{
    struct ct_scene s;
    struct ct_target ms, c;
    struct ct_gfx g;
    VkPipeline pipe;
    VkCommandBuffer cb;

    ct_scene_init(t, &s);
    if (!s.e.props.limits.standardSampleLocations)
        ct_skip(t, "standardSampleLocations is VK_FALSE");
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_4_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &ms);
    ct_scene_color(t, &s, &c);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.ms.rasterizationSamples = VK_SAMPLE_COUNT_4_BIT;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    begin_msaa(cb, &ms, &c);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 8.5f / 8.0f - 1.0f, 1.0f, ct_red, 0.0f); /* to x = 8.5 */
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, expect_coverage, NULL, 1, "resolved color (column 8 half covered)");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_target_fini(&s.e, &ms);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(sample_shading,
        "Rasterization > Sample Shading: with sampleShadingEnable and minSampleShading 1.0 (sampleRateShading), "
        "the fragment shader runs once per sample, SampleId numbering them: a shader writing red for sample 0 "
        "only resolves to a quarter of red",
        RA_COMMANDS)
{
    struct ct_scene s;
    struct ct_target ms, c;
    struct ct_gfx g;
    VkShaderModule fs;
    VkPipeline pipe;
    VkCommandBuffer cb;
    struct ct_box want = { 0, 0, W, H, CT_RGBA(64, 0, 0, 255), CT_BLACK };

    ct_scene_init(t, &s);
    if (!s.f.f2.features.sampleRateShading)
        ct_skip(t, "sampleRateShading is not supported");
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_4_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &ms);
    ct_scene_color(t, &s, &c);
    fs = ct_shader_module(t, &s.e, sample_frag, sizeof(sample_frag));
    ct_gfx_init(&g, s.vs, fs, s.layout, W, H);
    g.ms.rasterizationSamples = VK_SAMPLE_COUNT_4_BIT;
    g.ms.sampleShadingEnable = VK_TRUE;
    g.ms.minSampleShading = 1.0f;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    begin_msaa(cb, &ms, &c);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, NULL, 0.0f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want, 1, "resolved color");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_target_fini(&s.e, &ms);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(wide_points,
        "Rasterization > Points: a point of size s (PointSize, within pointSizeRange; largePoints) covers the "
        "pixels whose centers lie in the s x s square around it",
        RA_COMMANDS)
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkShaderModule vs;
    VkPipeline pipe;
    VkCommandBuffer cb;
    struct ct_box want = { 6, 6, 10, 10, CT_RED, CT_BLACK };

    ct_scene_init(t, &s);
    if (!s.f.f2.features.largePoints || s.e.props.limits.pointSizeRange[1] < 4.0f)
        ct_skip(t, "points of size 4 are not supported");
    ct_scene_color(t, &s, &c);
    vs = ct_shader_module(t, &s.e, bigpoint_vert, sizeof(bigpoint_vert));
    ct_gfx_init(&g, vs, s.fs, s.layout, W, H);
    g.ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, 0.0f, 0.0f, 0.0f, 0.0f, ct_red, 0.0f); /* the color, and 4 points at the center */
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want, 0, "a point of size 4 at (8, 8)");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

struct color_vertex {
    float pos[2];
    float color[4];
};

/* A quad (triangle strip) over the target: left vertices red 0, right
 * vertices red 1; and a triangle list whose triangles' first vertices are
 * green and blue. */
static void interpolation_vertices(struct color_vertex *v)
{
    static const struct color_vertex quad[4] = {
        { { -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f, 1.0f } },
        { { 1.0f, -1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },
        { { -1.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 1.0f } },
        { { 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },
    };
    static const struct color_vertex tris[6] = {
        { { -1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } }, /* provoking: green */
        { { 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },
        { { -1.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } },
        { { 1.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } }, /* provoking: blue */
        { { -1.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },
        { { 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } },
    };
    memcpy(v, quad, sizeof(quad));
    memcpy(v + 4, tris, sizeof(tris));
}

CT_TEST(interpolation_flat_smooth,
        "Rasterization > Basic Polygon Rasterization, Interpolation: smooth inputs are interpolated across the "
        "polygon (a linear ramp at w = 1, sampled at pixel centers); Flat inputs take the value of the provoking "
        "vertex, the first vertex of each triangle of a triangle list",
        RA_COMMANDS "vkCmdBindVertexBuffers")
{
    VkVertexInputBindingDescription binding = { 0, sizeof(struct color_vertex), VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription attrs[2] = {
        { 0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(struct color_vertex, pos) },
        { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(struct color_vertex, color) },
    };
    VkPipelineVertexInputStateCreateInfo vi;
    struct ct_scene s;
    struct ct_target c;
    struct ct_buffer vb;
    struct ct_gfx g;
    VkShaderModule vs, smooth_fs, flat_fs;
    VkPipeline smooth, flat;
    VkCommandBuffer cb;
    VkDeviceSize zero = 0, tri_offset = 4 * sizeof(struct color_vertex);
    uint32_t x, y;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 1024, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb);
    interpolation_vertices((struct color_vertex *)vb.data);
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = 2;
    vi.pVertexAttributeDescriptions = attrs;
    vs = ct_shader_module(t, &s.e, vcolor_vert, sizeof(vcolor_vert));
    smooth_fs = ct_shader_module(t, &s.e, smooth_frag, sizeof(smooth_frag));
    flat_fs = ct_shader_module(t, &s.e, flat_frag, sizeof(flat_frag));
    ct_gfx_init(&g, vs, smooth_fs, s.layout, W, H);
    g.vertex = &vi;
    smooth = ct_gfx_pipeline(t, &s.e, &g);
    g.fs = flat_fs;
    g.ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    flat = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, smooth);
    vkCmdBindVertexBuffers(cb, 0, 1, &vb.buffer, &zero);
    vkCmdDraw(cb, 4, 1, 0, 0);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, flat);
    vkCmdBindVertexBuffers(cb, 0, 1, &vb.buffer, &tri_offset);
    vkCmdDraw(cb, 6, 1, 0, 0);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    for (y = 0; y < H / 2; y++)
        for (x = 0; x < W; x++) {
            uint32_t got = ct_texel32(s.out.data, W, x, y), r = got & 0xff;
            uint32_t want = (uint32_t)(255.0f * ((float)x + 0.5f) / (float)W + 0.5f);
            CT_CHECKF(t, r + 2 >= want && r <= want + 2 && (got >> 8) == 0xff0000u,
                      "smooth: texel (%u, %u) is %08x, expected red %u", x, y, got, want);
        }
    /* the bottom half: triangle (0, 8) (16, 8) (0, 16) green, triangle
     * (16, 16) (0, 16) (16, 8) blue, split by the line x + 2y = 32; pixels
     * near it are not checked */
    for (y = H / 2; y < H; y++)
        for (x = 0; x < W; x++) {
            uint32_t got = ct_texel32(s.out.data, W, x, y);
            float f = ((float)x + 0.5f) + 2.0f * ((float)y + 0.5f) - 32.0f;
            if (f < -2.0f)
                CT_CHECKF(t, got == CT_GREEN, "flat: texel (%u, %u) is %08x, expected green", x, y, got);
            else if (f > 2.0f)
                CT_CHECKF(t, got == CT_BLUE, "flat: texel (%u, %u) is %08x, expected blue", x, y, got);
        }
    vkDestroyPipeline(s.e.device, smooth, NULL);
    vkDestroyPipeline(s.e.device, flat, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    vkDestroyShaderModule(s.e.device, smooth_fs, NULL);
    vkDestroyShaderModule(s.e.device, flat_fs, NULL);
    ct_buffer_fini(&s.e, &vb);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

static const struct ct_test *const tests[] = {
    CT_REF(viewport_transform),
    CT_REF(viewport_depth_range),
    CT_REF(clipping_depth_clamp),
    CT_REF(viewport_index_output),
    CT_REF(layer_output),
    CT_REF(polygon_culling),
    CT_REF(rasterizer_discard),
    CT_REF(multisample_coverage),
    CT_REF(sample_shading),
    CT_REF(wide_points),
    CT_REF(interpolation_flat_smooth),
};

CT_SUITE(rasterization, tests)
