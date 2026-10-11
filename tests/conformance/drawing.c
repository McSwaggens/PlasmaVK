/* Chapters "Drawing Commands" and "Fixed-Function Vertex Processing":
 * vertex and instance numbering of direct, indexed and indirect draws (with
 * counts), index types and primitive restart, primitive topologies, draw
 * parameters (BaseVertex, BaseInstance, DrawIndex), vertex attributes,
 * instance rates with divisors and dynamic binding strides. Most draws are
 * points at pixel centers of a 16 x 16 target, so each vertex lands on one
 * known pixel.
 *
 * Shaders: GLSL in the comments, compiled with
 *   glslangValidator -V --target-env vulkan1.3 -x */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

#define W CT_SIZE
#define H CT_SIZE

#define DR_COMMANDS                                                                                                \
    CT_RENDER_COMMANDS "vkCmdBindPipeline vkCmdDraw vkCmdBeginRendering vkCmdEndRendering vkCmdBindVertexBuffers "

/*
 * #version 450
 * // A point at pixel (VertexIndex, InstanceIndex) of a 16 x 16 target, colored
 * // (VertexIndex, InstanceIndex, 0, 255) / 255.
 * layout(location = 0) out vec4 v_color;
 * void main()
 * {
 *     gl_Position = vec4((float(gl_VertexIndex) + 0.5) / 8.0 - 1.0, (float(gl_InstanceIndex) + 0.5) / 8.0 - 1.0, 0.0, 1.0);
 *     gl_PointSize = 1.0;
 *     v_color = vec4(float(gl_VertexIndex) / 255.0, float(gl_InstanceIndex) / 255.0, 0.0, 1.0);
 * }
 */
static const uint32_t index_vert[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000030, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0009000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00000011, 0x0000001a,
    0x00000027, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00060005, 0x0000000b, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x0000000b,
    0x00000000, 0x505f6c67, 0x7469736f, 0x006e6f69, 0x00070006, 0x0000000b, 0x00000001, 0x505f6c67,
    0x746e696f, 0x657a6953, 0x00000000, 0x00070006, 0x0000000b, 0x00000002, 0x435f6c67, 0x4470696c,
    0x61747369, 0x0065636e, 0x00070006, 0x0000000b, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369,
    0x0065636e, 0x00030005, 0x0000000d, 0x00000000, 0x00060005, 0x00000011, 0x565f6c67, 0x65747265,
    0x646e4978, 0x00007865, 0x00070005, 0x0000001a, 0x495f6c67, 0x6174736e, 0x4965636e, 0x7865646e,
    0x00000000, 0x00040005, 0x00000027, 0x6f635f76, 0x00726f6c, 0x00030047, 0x0000000b, 0x00000002,
    0x00050048, 0x0000000b, 0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x0000000b, 0x00000001,
    0x0000000b, 0x00000001, 0x00050048, 0x0000000b, 0x00000002, 0x0000000b, 0x00000003, 0x00050048,
    0x0000000b, 0x00000003, 0x0000000b, 0x00000004, 0x00040047, 0x00000011, 0x0000000b, 0x0000002a,
    0x00040047, 0x0000001a, 0x0000000b, 0x0000002b, 0x00040047, 0x00000027, 0x0000001e, 0x00000000,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
    0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040015, 0x00000008, 0x00000020, 0x00000000,
    0x0004002b, 0x00000008, 0x00000009, 0x00000001, 0x0004001c, 0x0000000a, 0x00000006, 0x00000009,
    0x0006001e, 0x0000000b, 0x00000007, 0x00000006, 0x0000000a, 0x0000000a, 0x00040020, 0x0000000c,
    0x00000003, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000003, 0x00040015, 0x0000000e,
    0x00000020, 0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000, 0x00040020, 0x00000010,
    0x00000001, 0x0000000e, 0x0004003b, 0x00000010, 0x00000011, 0x00000001, 0x0004002b, 0x00000006,
    0x00000014, 0x3f000000, 0x0004002b, 0x00000006, 0x00000016, 0x41000000, 0x0004002b, 0x00000006,
    0x00000018, 0x3f800000, 0x0004003b, 0x00000010, 0x0000001a, 0x00000001, 0x0004002b, 0x00000006,
    0x00000020, 0x00000000, 0x00040020, 0x00000022, 0x00000003, 0x00000007, 0x0004002b, 0x0000000e,
    0x00000024, 0x00000001, 0x00040020, 0x00000025, 0x00000003, 0x00000006, 0x0004003b, 0x00000022,
    0x00000027, 0x00000003, 0x0004002b, 0x00000006, 0x0000002a, 0x437f0000, 0x00050036, 0x00000002,
    0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x0000000e, 0x00000012,
    0x00000011, 0x0004006f, 0x00000006, 0x00000013, 0x00000012, 0x00050081, 0x00000006, 0x00000015,
    0x00000013, 0x00000014, 0x00050088, 0x00000006, 0x00000017, 0x00000015, 0x00000016, 0x00050083,
    0x00000006, 0x00000019, 0x00000017, 0x00000018, 0x0004003d, 0x0000000e, 0x0000001b, 0x0000001a,
    0x0004006f, 0x00000006, 0x0000001c, 0x0000001b, 0x00050081, 0x00000006, 0x0000001d, 0x0000001c,
    0x00000014, 0x00050088, 0x00000006, 0x0000001e, 0x0000001d, 0x00000016, 0x00050083, 0x00000006,
    0x0000001f, 0x0000001e, 0x00000018, 0x00070050, 0x00000007, 0x00000021, 0x00000019, 0x0000001f,
    0x00000020, 0x00000018, 0x00050041, 0x00000022, 0x00000023, 0x0000000d, 0x0000000f, 0x0003003e,
    0x00000023, 0x00000021, 0x00050041, 0x00000025, 0x00000026, 0x0000000d, 0x00000024, 0x0003003e,
    0x00000026, 0x00000018, 0x0004003d, 0x0000000e, 0x00000028, 0x00000011, 0x0004006f, 0x00000006,
    0x00000029, 0x00000028, 0x00050088, 0x00000006, 0x0000002b, 0x00000029, 0x0000002a, 0x0004003d,
    0x0000000e, 0x0000002c, 0x0000001a, 0x0004006f, 0x00000006, 0x0000002d, 0x0000002c, 0x00050088,
    0x00000006, 0x0000002e, 0x0000002d, 0x0000002a, 0x00070050, 0x00000007, 0x0000002f, 0x0000002b,
    0x0000002e, 0x00000020, 0x00000018, 0x0003003e, 0x00000027, 0x0000002f, 0x000100fd, 0x00010038,
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
 * #version 460
 * // A point at pixel (VertexIndex, InstanceIndex) of a 16 x 16 target, colored
 * // (BaseVertex, BaseInstance, DrawIndex, 255) / 255.
 * layout(location = 0) out vec4 v_color;
 * void main()
 * {
 *     gl_Position = vec4((float(gl_VertexIndex) + 0.5) / 8.0 - 1.0, (float(gl_InstanceIndex) + 0.5) / 8.0 - 1.0, 0.0, 1.0);
 *     gl_PointSize = 1.0;
 *     v_color = vec4(float(gl_BaseVertex) / 255.0, float(gl_BaseInstance) / 255.0, float(gl_DrawID) / 255.0, 1.0);
 * }
 */
static const uint32_t params_vert[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000036, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x0000114b, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x00000000, 0x00000001, 0x000c000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000d,
    0x00000011, 0x0000001a, 0x00000027, 0x00000028, 0x0000002d, 0x00000031, 0x00030003, 0x00000002,
    0x000001cc, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00060005, 0x0000000b, 0x505f6c67,
    0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x0000000b, 0x00000000, 0x505f6c67, 0x7469736f,
    0x006e6f69, 0x00070006, 0x0000000b, 0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953, 0x00000000,
    0x00070006, 0x0000000b, 0x00000002, 0x435f6c67, 0x4470696c, 0x61747369, 0x0065636e, 0x00070006,
    0x0000000b, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e, 0x00030005, 0x0000000d,
    0x00000000, 0x00060005, 0x00000011, 0x565f6c67, 0x65747265, 0x646e4978, 0x00007865, 0x00070005,
    0x0000001a, 0x495f6c67, 0x6174736e, 0x4965636e, 0x7865646e, 0x00000000, 0x00040005, 0x00000027,
    0x6f635f76, 0x00726f6c, 0x00060005, 0x00000028, 0x425f6c67, 0x56657361, 0x65747265, 0x00000078,
    0x00060005, 0x0000002d, 0x425f6c67, 0x49657361, 0x6174736e, 0x0065636e, 0x00050005, 0x00000031,
    0x445f6c67, 0x49776172, 0x00000044, 0x00030047, 0x0000000b, 0x00000002, 0x00050048, 0x0000000b,
    0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x0000000b, 0x00000001, 0x0000000b, 0x00000001,
    0x00050048, 0x0000000b, 0x00000002, 0x0000000b, 0x00000003, 0x00050048, 0x0000000b, 0x00000003,
    0x0000000b, 0x00000004, 0x00040047, 0x00000011, 0x0000000b, 0x0000002a, 0x00040047, 0x0000001a,
    0x0000000b, 0x0000002b, 0x00040047, 0x00000027, 0x0000001e, 0x00000000, 0x00040047, 0x00000028,
    0x0000000b, 0x00001148, 0x00040047, 0x0000002d, 0x0000000b, 0x00001149, 0x00040047, 0x00000031,
    0x0000000b, 0x0000114a, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
    0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040015, 0x00000008,
    0x00000020, 0x00000000, 0x0004002b, 0x00000008, 0x00000009, 0x00000001, 0x0004001c, 0x0000000a,
    0x00000006, 0x00000009, 0x0006001e, 0x0000000b, 0x00000007, 0x00000006, 0x0000000a, 0x0000000a,
    0x00040020, 0x0000000c, 0x00000003, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000003,
    0x00040015, 0x0000000e, 0x00000020, 0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000,
    0x00040020, 0x00000010, 0x00000001, 0x0000000e, 0x0004003b, 0x00000010, 0x00000011, 0x00000001,
    0x0004002b, 0x00000006, 0x00000014, 0x3f000000, 0x0004002b, 0x00000006, 0x00000016, 0x41000000,
    0x0004002b, 0x00000006, 0x00000018, 0x3f800000, 0x0004003b, 0x00000010, 0x0000001a, 0x00000001,
    0x0004002b, 0x00000006, 0x00000020, 0x00000000, 0x00040020, 0x00000022, 0x00000003, 0x00000007,
    0x0004002b, 0x0000000e, 0x00000024, 0x00000001, 0x00040020, 0x00000025, 0x00000003, 0x00000006,
    0x0004003b, 0x00000022, 0x00000027, 0x00000003, 0x0004003b, 0x00000010, 0x00000028, 0x00000001,
    0x0004002b, 0x00000006, 0x0000002b, 0x437f0000, 0x0004003b, 0x00000010, 0x0000002d, 0x00000001,
    0x0004003b, 0x00000010, 0x00000031, 0x00000001, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x0000000e, 0x00000012, 0x00000011, 0x0004006f,
    0x00000006, 0x00000013, 0x00000012, 0x00050081, 0x00000006, 0x00000015, 0x00000013, 0x00000014,
    0x00050088, 0x00000006, 0x00000017, 0x00000015, 0x00000016, 0x00050083, 0x00000006, 0x00000019,
    0x00000017, 0x00000018, 0x0004003d, 0x0000000e, 0x0000001b, 0x0000001a, 0x0004006f, 0x00000006,
    0x0000001c, 0x0000001b, 0x00050081, 0x00000006, 0x0000001d, 0x0000001c, 0x00000014, 0x00050088,
    0x00000006, 0x0000001e, 0x0000001d, 0x00000016, 0x00050083, 0x00000006, 0x0000001f, 0x0000001e,
    0x00000018, 0x00070050, 0x00000007, 0x00000021, 0x00000019, 0x0000001f, 0x00000020, 0x00000018,
    0x00050041, 0x00000022, 0x00000023, 0x0000000d, 0x0000000f, 0x0003003e, 0x00000023, 0x00000021,
    0x00050041, 0x00000025, 0x00000026, 0x0000000d, 0x00000024, 0x0003003e, 0x00000026, 0x00000018,
    0x0004003d, 0x0000000e, 0x00000029, 0x00000028, 0x0004006f, 0x00000006, 0x0000002a, 0x00000029,
    0x00050088, 0x00000006, 0x0000002c, 0x0000002a, 0x0000002b, 0x0004003d, 0x0000000e, 0x0000002e,
    0x0000002d, 0x0004006f, 0x00000006, 0x0000002f, 0x0000002e, 0x00050088, 0x00000006, 0x00000030,
    0x0000002f, 0x0000002b, 0x0004003d, 0x0000000e, 0x00000032, 0x00000031, 0x0004006f, 0x00000006,
    0x00000033, 0x00000032, 0x00050088, 0x00000006, 0x00000034, 0x00000033, 0x0000002b, 0x00070050,
    0x00000007, 0x00000035, 0x0000002c, 0x00000030, 0x00000034, 0x00000018, 0x0003003e, 0x00000027,
    0x00000035, 0x000100fd, 0x00010038,
};

/*
 * #version 450
 * // Positions from a vertex buffer, shifted down one pixel (of 16) per instance.
 * layout(location = 0) in vec2 pos;
 * void main()
 * {
 *     gl_Position = vec4(pos.x, pos.y + float(gl_InstanceIndex) * 0.125, 0.0, 1.0);
 *     gl_PointSize = 1.0;
 * }
 */
static const uint32_t pos_vert[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000028, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0008000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00000012, 0x0000001a,
    0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00060005,
    0x0000000b, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x0000000b, 0x00000000,
    0x505f6c67, 0x7469736f, 0x006e6f69, 0x00070006, 0x0000000b, 0x00000001, 0x505f6c67, 0x746e696f,
    0x657a6953, 0x00000000, 0x00070006, 0x0000000b, 0x00000002, 0x435f6c67, 0x4470696c, 0x61747369,
    0x0065636e, 0x00070006, 0x0000000b, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e,
    0x00030005, 0x0000000d, 0x00000000, 0x00030005, 0x00000012, 0x00736f70, 0x00070005, 0x0000001a,
    0x495f6c67, 0x6174736e, 0x4965636e, 0x7865646e, 0x00000000, 0x00030047, 0x0000000b, 0x00000002,
    0x00050048, 0x0000000b, 0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x0000000b, 0x00000001,
    0x0000000b, 0x00000001, 0x00050048, 0x0000000b, 0x00000002, 0x0000000b, 0x00000003, 0x00050048,
    0x0000000b, 0x00000003, 0x0000000b, 0x00000004, 0x00040047, 0x00000012, 0x0000001e, 0x00000000,
    0x00040047, 0x0000001a, 0x0000000b, 0x0000002b, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
    0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004,
    0x00040015, 0x00000008, 0x00000020, 0x00000000, 0x0004002b, 0x00000008, 0x00000009, 0x00000001,
    0x0004001c, 0x0000000a, 0x00000006, 0x00000009, 0x0006001e, 0x0000000b, 0x00000007, 0x00000006,
    0x0000000a, 0x0000000a, 0x00040020, 0x0000000c, 0x00000003, 0x0000000b, 0x0004003b, 0x0000000c,
    0x0000000d, 0x00000003, 0x00040015, 0x0000000e, 0x00000020, 0x00000001, 0x0004002b, 0x0000000e,
    0x0000000f, 0x00000000, 0x00040017, 0x00000010, 0x00000006, 0x00000002, 0x00040020, 0x00000011,
    0x00000001, 0x00000010, 0x0004003b, 0x00000011, 0x00000012, 0x00000001, 0x0004002b, 0x00000008,
    0x00000013, 0x00000000, 0x00040020, 0x00000014, 0x00000001, 0x00000006, 0x00040020, 0x00000019,
    0x00000001, 0x0000000e, 0x0004003b, 0x00000019, 0x0000001a, 0x00000001, 0x0004002b, 0x00000006,
    0x0000001d, 0x3e000000, 0x0004002b, 0x00000006, 0x00000020, 0x00000000, 0x0004002b, 0x00000006,
    0x00000021, 0x3f800000, 0x00040020, 0x00000023, 0x00000003, 0x00000007, 0x0004002b, 0x0000000e,
    0x00000025, 0x00000001, 0x00040020, 0x00000026, 0x00000003, 0x00000006, 0x00050036, 0x00000002,
    0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041, 0x00000014, 0x00000015,
    0x00000012, 0x00000013, 0x0004003d, 0x00000006, 0x00000016, 0x00000015, 0x00050041, 0x00000014,
    0x00000017, 0x00000012, 0x00000009, 0x0004003d, 0x00000006, 0x00000018, 0x00000017, 0x0004003d,
    0x0000000e, 0x0000001b, 0x0000001a, 0x0004006f, 0x00000006, 0x0000001c, 0x0000001b, 0x00050085,
    0x00000006, 0x0000001e, 0x0000001c, 0x0000001d, 0x00050081, 0x00000006, 0x0000001f, 0x00000018,
    0x0000001e, 0x00070050, 0x00000007, 0x00000022, 0x00000016, 0x0000001f, 0x00000020, 0x00000021,
    0x00050041, 0x00000023, 0x00000024, 0x0000000d, 0x0000000f, 0x0003003e, 0x00000024, 0x00000022,
    0x00050041, 0x00000026, 0x00000027, 0x0000000d, 0x00000025, 0x0003003e, 0x00000027, 0x00000021,
    0x000100fd, 0x00010038,
};

/*
 * #version 450
 * // Vertex attributes: a position, a color and a scale (a one-component
 * // format: the missing components read as 0, 0, 1), one pixel (of 16) down
 * // per instance.
 * layout(location = 0) in vec2 pos;
 * layout(location = 1) in vec4 color;
 * layout(location = 2) in vec4 scale;
 * layout(location = 0) out vec4 v_color;
 * void main()
 * {
 *     gl_Position = vec4(pos.x, pos.y + float(gl_InstanceIndex) * 0.125, 0.0, 1.0);
 *     gl_PointSize = 1.0;
 *     v_color = vec4(color.rgb * scale.x, color.a * scale.w + scale.y + scale.z);
 * }
 */
static const uint32_t attr_vert[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000043, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x000b000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00000012, 0x0000001a,
    0x00000028, 0x0000002a, 0x0000002e, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
    0x6e69616d, 0x00000000, 0x00060005, 0x0000000b, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000,
    0x00060006, 0x0000000b, 0x00000000, 0x505f6c67, 0x7469736f, 0x006e6f69, 0x00070006, 0x0000000b,
    0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953, 0x00000000, 0x00070006, 0x0000000b, 0x00000002,
    0x435f6c67, 0x4470696c, 0x61747369, 0x0065636e, 0x00070006, 0x0000000b, 0x00000003, 0x435f6c67,
    0x446c6c75, 0x61747369, 0x0065636e, 0x00030005, 0x0000000d, 0x00000000, 0x00030005, 0x00000012,
    0x00736f70, 0x00070005, 0x0000001a, 0x495f6c67, 0x6174736e, 0x4965636e, 0x7865646e, 0x00000000,
    0x00040005, 0x00000028, 0x6f635f76, 0x00726f6c, 0x00040005, 0x0000002a, 0x6f6c6f63, 0x00000072,
    0x00040005, 0x0000002e, 0x6c616373, 0x00000065, 0x00030047, 0x0000000b, 0x00000002, 0x00050048,
    0x0000000b, 0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x0000000b, 0x00000001, 0x0000000b,
    0x00000001, 0x00050048, 0x0000000b, 0x00000002, 0x0000000b, 0x00000003, 0x00050048, 0x0000000b,
    0x00000003, 0x0000000b, 0x00000004, 0x00040047, 0x00000012, 0x0000001e, 0x00000000, 0x00040047,
    0x0000001a, 0x0000000b, 0x0000002b, 0x00040047, 0x00000028, 0x0000001e, 0x00000000, 0x00040047,
    0x0000002a, 0x0000001e, 0x00000001, 0x00040047, 0x0000002e, 0x0000001e, 0x00000002, 0x00020013,
    0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017,
    0x00000007, 0x00000006, 0x00000004, 0x00040015, 0x00000008, 0x00000020, 0x00000000, 0x0004002b,
    0x00000008, 0x00000009, 0x00000001, 0x0004001c, 0x0000000a, 0x00000006, 0x00000009, 0x0006001e,
    0x0000000b, 0x00000007, 0x00000006, 0x0000000a, 0x0000000a, 0x00040020, 0x0000000c, 0x00000003,
    0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000003, 0x00040015, 0x0000000e, 0x00000020,
    0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000, 0x00040017, 0x00000010, 0x00000006,
    0x00000002, 0x00040020, 0x00000011, 0x00000001, 0x00000010, 0x0004003b, 0x00000011, 0x00000012,
    0x00000001, 0x0004002b, 0x00000008, 0x00000013, 0x00000000, 0x00040020, 0x00000014, 0x00000001,
    0x00000006, 0x00040020, 0x00000019, 0x00000001, 0x0000000e, 0x0004003b, 0x00000019, 0x0000001a,
    0x00000001, 0x0004002b, 0x00000006, 0x0000001d, 0x3e000000, 0x0004002b, 0x00000006, 0x00000020,
    0x00000000, 0x0004002b, 0x00000006, 0x00000021, 0x3f800000, 0x00040020, 0x00000023, 0x00000003,
    0x00000007, 0x0004002b, 0x0000000e, 0x00000025, 0x00000001, 0x00040020, 0x00000026, 0x00000003,
    0x00000006, 0x0004003b, 0x00000023, 0x00000028, 0x00000003, 0x00040020, 0x00000029, 0x00000001,
    0x00000007, 0x0004003b, 0x00000029, 0x0000002a, 0x00000001, 0x00040017, 0x0000002b, 0x00000006,
    0x00000003, 0x0004003b, 0x00000029, 0x0000002e, 0x00000001, 0x0004002b, 0x00000008, 0x00000032,
    0x00000003, 0x0004002b, 0x00000008, 0x0000003b, 0x00000002, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041, 0x00000014, 0x00000015, 0x00000012,
    0x00000013, 0x0004003d, 0x00000006, 0x00000016, 0x00000015, 0x00050041, 0x00000014, 0x00000017,
    0x00000012, 0x00000009, 0x0004003d, 0x00000006, 0x00000018, 0x00000017, 0x0004003d, 0x0000000e,
    0x0000001b, 0x0000001a, 0x0004006f, 0x00000006, 0x0000001c, 0x0000001b, 0x00050085, 0x00000006,
    0x0000001e, 0x0000001c, 0x0000001d, 0x00050081, 0x00000006, 0x0000001f, 0x00000018, 0x0000001e,
    0x00070050, 0x00000007, 0x00000022, 0x00000016, 0x0000001f, 0x00000020, 0x00000021, 0x00050041,
    0x00000023, 0x00000024, 0x0000000d, 0x0000000f, 0x0003003e, 0x00000024, 0x00000022, 0x00050041,
    0x00000026, 0x00000027, 0x0000000d, 0x00000025, 0x0003003e, 0x00000027, 0x00000021, 0x0004003d,
    0x00000007, 0x0000002c, 0x0000002a, 0x0008004f, 0x0000002b, 0x0000002d, 0x0000002c, 0x0000002c,
    0x00000000, 0x00000001, 0x00000002, 0x00050041, 0x00000014, 0x0000002f, 0x0000002e, 0x00000013,
    0x0004003d, 0x00000006, 0x00000030, 0x0000002f, 0x0005008e, 0x0000002b, 0x00000031, 0x0000002d,
    0x00000030, 0x00050041, 0x00000014, 0x00000033, 0x0000002a, 0x00000032, 0x0004003d, 0x00000006,
    0x00000034, 0x00000033, 0x00050041, 0x00000014, 0x00000035, 0x0000002e, 0x00000032, 0x0004003d,
    0x00000006, 0x00000036, 0x00000035, 0x00050085, 0x00000006, 0x00000037, 0x00000034, 0x00000036,
    0x00050041, 0x00000014, 0x00000038, 0x0000002e, 0x00000009, 0x0004003d, 0x00000006, 0x00000039,
    0x00000038, 0x00050081, 0x00000006, 0x0000003a, 0x00000037, 0x00000039, 0x00050041, 0x00000014,
    0x0000003c, 0x0000002e, 0x0000003b, 0x0004003d, 0x00000006, 0x0000003d, 0x0000003c, 0x00050081,
    0x00000006, 0x0000003e, 0x0000003a, 0x0000003d, 0x00050051, 0x00000006, 0x0000003f, 0x00000031,
    0x00000000, 0x00050051, 0x00000006, 0x00000040, 0x00000031, 0x00000001, 0x00050051, 0x00000006,
    0x00000041, 0x00000031, 0x00000002, 0x00070050, 0x00000007, 0x00000042, 0x0000003f, 0x00000040,
    0x00000041, 0x0000003e, 0x0003003e, 0x00000028, 0x00000042, 0x000100fd, 0x00010038,
};
/* NDC of the center of pixel column (or row) p. */
static float ndc(uint32_t p)
{
    return ((float)p + 0.5f) / 8.0f - 1.0f;
}

/* ct_color_frag's color, without drawing. */
static void set_color(VkCommandBuffer cb, VkPipelineLayout layout, const float color[4])
{
    struct ct_rect_pc pc;
    memset(&pc, 0, sizeof(pc));
    memcpy(pc.color, color, sizeof(pc.color));
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
}

/* Expected points: n pixels with their colors, bg elsewhere. */
struct points {
    uint32_t n, bg;
    uint32_t x[128], y[128], c[128];
};

static void add_point(struct points *p, uint32_t x, uint32_t y, uint32_t c)
{
    uint32_t i;
    for (i = 0; i < p->n; i++)
        if (p->x[i] == x && p->y[i] == y) {
            p->c[i] = c; /* drawn over */
            return;
        }
    if (p->n < 128) {
        p->x[p->n] = x;
        p->y[p->n] = y;
        p->c[p->n++] = c;
    }
}

/* The points index.vert draws for vertices [v0, v0 + count) of instances
 * [i0, i0 + instances). */
static void add_index_points(struct points *p, uint32_t v0, uint32_t count, uint32_t i0, uint32_t instances)
{
    uint32_t v, i;
    for (i = i0; i < i0 + instances; i++)
        for (v = v0; v < v0 + count; v++)
            add_point(p, v, i, CT_RGBA(v, i, 0, 255));
}

static uint32_t expect_points(uint32_t x, uint32_t y, const void *ctx)
{
    const struct points *p = ctx;
    uint32_t i;
    for (i = 0; i < p->n; i++)
        if (p->x[i] == x && p->y[i] == y)
            return p->c[i];
    return p->bg;
}

/* A point-list pipeline of a vertex shader and flat.frag. */
static VkPipeline point_pipeline(struct ct *t, struct ct_scene *s, VkShaderModule vs, VkShaderModule fs,
                                 const VkPipelineVertexInputStateCreateInfo *vertex)
{
    struct ct_gfx g;
    ct_gfx_init(&g, vs, fs, s->layout, W, H);
    g.ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    g.vertex = vertex;
    return ct_gfx_pipeline(t, &s->e, &g);
}

CT_TEST(draw_vertex_instance_index,
        "Drawing Commands > Programmable Primitive Shading (vkCmdDraw): vertexCount vertices numbered from "
        "firstVertex (VertexIndex) are drawn for each of instanceCount instances numbered from firstInstance "
        "(InstanceIndex)",
        DR_COMMANDS)
{
    struct ct_scene s;
    struct ct_target c;
    struct points want;
    VkShaderModule vs, fs;
    VkPipeline pipe;
    VkCommandBuffer cb;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    vs = ct_shader_module(t, &s.e, index_vert, sizeof(index_vert));
    fs = ct_shader_module(t, &s.e, flat_frag, sizeof(flat_frag));
    pipe = point_pipeline(t, &s, vs, fs, NULL);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdDraw(cb, 4, 3, 2, 5);  /* x 2..5, y 5..7 */
    vkCmdDraw(cb, 1, 1, 14, 0); /* x 14, y 0 */
    vkCmdDraw(cb, 0, 4, 0, 0);  /* nothing */
    vkCmdDraw(cb, 3, 0, 0, 0);  /* nothing */
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    memset(&want, 0, sizeof(want));
    want.bg = CT_BLACK;
    add_index_points(&want, 2, 4, 5, 3);
    add_index_points(&want, 14, 1, 0, 1);
    ct_check_rgba8(t, s.out.data, W, H, expect_points, &want, 0, "points");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(draw_indexed,
        "Drawing Commands > Programmable Primitive Shading (vkCmdDrawIndexed): indices are read from the index "
        "buffer at its bound offset plus firstIndex times the index size (VK_INDEX_TYPE_UINT16 and UINT32), "
        "VertexIndex is the index plus vertexOffset, and instances are numbered from firstInstance",
        DR_COMMANDS "vkCmdDrawIndexed vkCmdBindIndexBuffer")
{
    static const uint16_t i16[8] = { 7, 7, 0, 3, 1, 7, 7, 7 };
    static const uint32_t i32[6] = { 9, 9, 9, 2, 0, 4 };
    struct ct_scene s;
    struct ct_target c;
    struct ct_buffer ib;
    struct points want;
    VkShaderModule vs, fs;
    VkPipeline pipe;
    VkCommandBuffer cb;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 256, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &ib);
    memcpy(ib.data, i16, sizeof(i16));
    memcpy(ib.data + 64, i32, sizeof(i32));
    vs = ct_shader_module(t, &s.e, index_vert, sizeof(index_vert));
    fs = ct_shader_module(t, &s.e, flat_frag, sizeof(flat_frag));
    pipe = point_pipeline(t, &s, vs, fs, NULL);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindIndexBuffer(cb, ib.buffer, 2, VK_INDEX_TYPE_UINT16); /* from i16[1] */
    vkCmdDrawIndexed(cb, 3, 2, 1, 5, 1);                         /* i16[2..4] = 0 3 1, + 5: x 5 8 6, y 1..2 */
    vkCmdBindIndexBuffer(cb, ib.buffer, 64, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cb, 3, 1, 3, -0, 9);                         /* i32[3..5] = 2 0 4: x 2 0 4, y 9 */
    vkCmdDrawIndexed(cb, 2, 1, 4, 10, 12);                        /* i32[4..5] = 0 4, + 10: x 10 14, y 12 */
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    memset(&want, 0, sizeof(want));
    want.bg = CT_BLACK;
    add_index_points(&want, 5, 1, 1, 2);
    add_index_points(&want, 8, 1, 1, 2);
    add_index_points(&want, 6, 1, 1, 2);
    add_index_points(&want, 2, 1, 9, 1);
    add_index_points(&want, 0, 1, 9, 1);
    add_index_points(&want, 4, 1, 9, 1);
    add_index_points(&want, 10, 1, 12, 1);
    add_index_points(&want, 14, 1, 12, 1);
    ct_check_rgba8(t, s.out.data, W, H, expect_points, &want, 0, "points");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_buffer_fini(&s.e, &ib);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(draw_indexed_uint8,
        "Drawing Commands > Programmable Primitive Shading: VK_INDEX_TYPE_UINT8 indices are one byte each "
        "(indexTypeUint8, required from Vulkan 1.4)",
        DR_COMMANDS "vkCmdDrawIndexed vkCmdBindIndexBuffer")
{
    static const uint8_t i8[8] = { 9, 3, 1, 12, 0, 9, 9, 9 };
    struct ct_scene s;
    struct ct_target c;
    struct ct_buffer ib;
    struct points want;
    VkShaderModule vs, fs;
    VkPipeline pipe;
    VkCommandBuffer cb;

    ct_scene_init(t, &s);
    if (s.e.version < VK_API_VERSION_1_4)
        ct_skip(t, "VK_INDEX_TYPE_UINT8 is core in Vulkan 1.4");
    CT_REQUIRE(t, s.f.v14.indexTypeUint8);
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 64, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &ib);
    memcpy(ib.data, i8, sizeof(i8));
    vs = ct_shader_module(t, &s.e, index_vert, sizeof(index_vert));
    fs = ct_shader_module(t, &s.e, flat_frag, sizeof(flat_frag));
    pipe = point_pipeline(t, &s, vs, fs, NULL);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindIndexBuffer(cb, ib.buffer, 1, VK_INDEX_TYPE_UINT8);
    vkCmdDrawIndexed(cb, 4, 1, 0, 2, 4); /* i8[1..4] = 3 1 12 0, + 2: x 5 3 14 2, y 4 */
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    memset(&want, 0, sizeof(want));
    want.bg = CT_BLACK;
    add_index_points(&want, 5, 1, 4, 1);
    add_index_points(&want, 3, 1, 4, 1);
    add_index_points(&want, 14, 1, 4, 1);
    add_index_points(&want, 2, 1, 4, 1);
    ct_check_rgba8(t, s.out.data, W, H, expect_points, &want, 0, "points");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_buffer_fini(&s.e, &ib);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

/* Two quads as triangle strips with a restart index between them, and what
 * vertex `restart` (the index's value when restart is off) holds: the top
 * right corner, so strips joined through it cover the top right quadrant. */
static void restart_vertices(float *v, uint32_t restart)
{
    static const float quads[16] = {
        -1.0f, -1.0f, 0.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, /* top left */
        0.0f,  0.0f,  1.0f, 0.0f,  0.0f,  1.0f, 1.0f, 1.0f, /* bottom right */
    };
    memcpy(v, quads, sizeof(quads));
    v[2 * restart] = 1.0f;
    v[2 * restart + 1] = -1.0f;
}

CT_TEST(primitive_restart,
        "Drawing Commands > Primitive Topologies > Primitive Restart: with primitiveRestartEnable, the index "
        "0xFFFF (VK_INDEX_TYPE_UINT16), 0xFFFFFFFF (UINT32) or 0xFF (UINT8, Vulkan 1.4) ends the strip and the "
        "next index starts a new one; without it (VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE, "
        "vkCmdSetPrimitiveRestartEnable) 0xFFFF is an ordinary vertex index",
        DR_COMMANDS "vkCmdDrawIndexed vkCmdBindIndexBuffer vkCmdSetPrimitiveRestartEnable")
{
    static const uint16_t i16[9] = { 0, 1, 2, 3, 0xffff, 4, 5, 6, 7 };
    static const uint32_t i32[9] = { 0, 1, 2, 3, 0xffffffffu, 4, 5, 6, 7 };
    static const uint8_t i8[9] = { 0, 1, 2, 3, 0xff, 4, 5, 6, 7 };
    static const uint32_t restarted[4] = { CT_RED, CT_BLACK, CT_BLACK, CT_RED };
    struct ct_scene s;
    struct ct_target c;
    struct ct_buffer vb, ib;
    struct ct_gfx g;
    VkVertexInputBindingDescription binding = { 0, 8, VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription attr = { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 };
    VkPipelineVertexInputStateCreateInfo vi;
    VkShaderModule vs;
    VkPipeline fixed, dyn;
    VkCommandBuffer cb;
    VkDeviceSize zero = 0;
    uint32_t k, got;
    int uint8 = 0;

    ct_scene_init(t, &s);
    uint8 = s.e.version >= VK_API_VERSION_1_4 && s.f.v14.indexTypeUint8;
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 65536 * 8, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb);
    ct_host_buffer(t, &s.e, 256, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &ib);
    restart_vertices((float *)vb.data, 0xffff);
    memcpy(ib.data, i16, sizeof(i16));
    memcpy(ib.data + 64, i32, sizeof(i32));
    memcpy(ib.data + 128, i8, sizeof(i8));
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions = &attr;
    vs = ct_shader_module(t, &s.e, pos_vert, sizeof(pos_vert));
    ct_gfx_init(&g, vs, s.fs, s.layout, W, H);
    g.vertex = &vi;
    g.ia.primitiveRestartEnable = VK_TRUE;
    fixed = ct_gfx_pipeline(t, &s.e, &g);
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE;
    g.ia.primitiveRestartEnable = VK_FALSE;
    dyn = ct_gfx_pipeline(t, &s.e, &g);
    for (k = 0; k < 4; k++) {
        static const char *const what[4] = { "UINT16", "UINT32", "UINT8", "UINT16, dynamic restart enable" };
        if (k == 2 && !uint8)
            continue;
        cb = ct_begin_commands(t, &s.e);
        ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, k == 3 ? dyn : fixed);
        vkCmdBindVertexBuffers(cb, 0, 1, &vb.buffer, &zero);
        vkCmdBindIndexBuffer(cb, ib.buffer, k == 1 ? 64 : k == 2 ? 128 : 0,
                             k == 1 ? VK_INDEX_TYPE_UINT32 : k == 2 ? VK_INDEX_TYPE_UINT8 : VK_INDEX_TYPE_UINT16);
        set_color(cb, s.layout, ct_red);
        if (k == 3)
            vkCmdSetPrimitiveRestartEnable(cb, VK_TRUE);
        vkCmdDrawIndexed(cb, 9, 1, 0, 0, 0);
        vkCmdEndRendering(cb);
        ct_scene_read(t, &s, cb, &c);
        ct_check_rgba8(t, s.out.data, W, H, ct_expect_quadrants, restarted, 0, what[k]);
    }
    /* restart off: 0xFFFF is vertex 65535, at the top right corner */
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, dyn);
    vkCmdBindVertexBuffers(cb, 0, 1, &vb.buffer, &zero);
    vkCmdBindIndexBuffer(cb, ib.buffer, 0, VK_INDEX_TYPE_UINT16);
    set_color(cb, s.layout, ct_red);
    vkCmdSetPrimitiveRestartEnable(cb, VK_FALSE);
    vkCmdDrawIndexed(cb, 9, 1, 0, 0, 0);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    got = ct_texel32(s.out.data, W, 12, 2);
    CT_CHECKF(t, got == CT_RED, "restart off: texel (12, 2) is %08x, expected red (the strip runs through vertex "
              "0xFFFF)", got);
    got = ct_texel32(s.out.data, W, 0, 15);
    CT_CHECKF(t, got == CT_BLACK, "restart off: texel (0, 15) is %08x, expected black", got);
    vkDestroyPipeline(s.e.device, fixed, NULL);
    vkDestroyPipeline(s.e.device, dyn, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    ct_buffer_fini(&s.e, &vb);
    ct_buffer_fini(&s.e, &ib);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

/* Which pixels a draw covered: a bitmap of non-black texels. */
static int lit(const struct ct_scene *s, uint32_t x, uint32_t y)
{
    return ct_texel32(s->out.data, W, x, y) != CT_BLACK;
}

CT_TEST(primitive_topologies,
        "Drawing Commands > Primitive Topologies: point lists (one point per vertex), line lists (pairs), line "
        "strips (consecutive vertices), triangle lists (triples), triangle strips and triangle fans (the first "
        "vertex shared; not on VK_KHR_portability_subset devices, whose triangleFans feature may be off) "
        "rasterize the expected pixels",
        DR_COMMANDS)
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_buffer vb;
    struct ct_gfx g;
    VkVertexInputBindingDescription binding = { 0, 8, VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription attr = { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 };
    VkPipelineVertexInputStateCreateInfo vi;
    VkShaderModule vs;
    VkPipeline pipes[6];
    VkCommandBuffer cb;
    float *v;
    uint32_t k, x, y;
    static const VkPrimitiveTopology topologies[6] = {
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST,     VK_PRIMITIVE_TOPOLOGY_LINE_LIST,      VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
    };
    int fans;

    ct_scene_init(t, &s);
    fans = !s.e.portability_subset;
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 4096, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb);
    v = (float *)vb.data;
    /* 0: points at pixels (1, 1), (3, 1), (5, 1) */
    v[0] = ndc(1), v[1] = ndc(1), v[2] = ndc(3), v[3] = ndc(1), v[4] = ndc(5), v[5] = ndc(1);
    /* 8: a line list, two horizontal lines in rows 3 and 5, x 1.5 to 14.5 */
    v[16] = ndc(1), v[17] = ndc(3), v[18] = ndc(14), v[19] = ndc(3);
    v[20] = ndc(1), v[21] = ndc(5), v[22] = ndc(14), v[23] = ndc(5);
    /* 16: a line strip: row 7 from x 1.5 to 14.5, then down column 14 to row 13.5 */
    v[32] = ndc(1), v[33] = ndc(7), v[34] = ndc(14), v[35] = ndc(7), v[36] = ndc(14), v[37] = ndc(13);
    /* 24: a triangle list: two triangles making the rectangle x 0..4, y 9..16 */
    v[48] = -1.0f, v[49] = 0.125f, v[50] = -0.5f, v[51] = 0.125f, v[52] = -1.0f, v[53] = 1.0f;
    v[54] = -0.5f, v[55] = 0.125f, v[56] = -0.5f, v[57] = 1.0f, v[58] = -1.0f, v[59] = 1.0f;
    /* 32: a triangle strip: the rectangle x 5..9, y 9..16 */
    v[64] = -0.375f, v[65] = 0.125f, v[66] = 0.125f, v[67] = 0.125f, v[68] = -0.375f, v[69] = 1.0f, v[70] = 0.125f,
    v[71] = 1.0f;
    /* 40: a triangle fan: the rectangle x 10..12, y 9..16, from its top left corner */
    v[80] = 0.25f, v[81] = 0.125f, v[82] = 0.5f, v[83] = 0.125f, v[84] = 0.5f, v[85] = 1.0f, v[86] = 0.25f,
    v[87] = 1.0f;
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions = &attr;
    vs = ct_shader_module(t, &s.e, pos_vert, sizeof(pos_vert));
    memset(pipes, 0, sizeof(pipes));
    for (k = 0; k < 6; k++) {
        if (k == 5 && !fans)
            continue;
        ct_gfx_init(&g, vs, s.fs, s.layout, W, H);
        g.vertex = &vi;
        g.ia.topology = topologies[k];
        pipes[k] = ct_gfx_pipeline(t, &s.e, &g);
    }
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    for (k = 0; k < 6; k++) {
        VkDeviceSize offset = 8 * 8 * k; /* 8 vertices per section */
        static const uint32_t counts[6] = { 3, 4, 3, 6, 4, 4 };
        if (!pipes[k])
            continue;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[k]);
        vkCmdBindVertexBuffers(cb, 0, 1, &vb.buffer, &offset);
        set_color(cb, s.layout, ct_white);
        vkCmdDraw(cb, counts[k], 1, 0, 0);
    }
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    /* points: exactly their pixels in rows 0..2 */
    for (y = 0; y < 3; y++)
        for (x = 0; x < W; x++)
            CT_CHECKF(t, lit(&s, x, y) == (y == 1 && (x == 1 || x == 3 || x == 5)), "points: texel (%u, %u) %s", x,
                      y, lit(&s, x, y) ? "drawn" : "not drawn");
    /* lines: the interior of each segment, nothing in the rows between */
    for (x = 3; x <= 12; x++) {
        CT_CHECKF(t, lit(&s, x, 3) && lit(&s, x, 5), "line list: texel (%u, 3) or (%u, 5) not drawn", x, x);
        CT_CHECKF(t, !lit(&s, x, 4) && !lit(&s, x, 6), "line list: texel (%u, 4) or (%u, 6) drawn", x, x);
        CT_CHECKF(t, lit(&s, x, 7) && !lit(&s, x, 8), "line strip: row 7 not drawn or row 8 drawn at x %u", x);
    }
    for (y = 9; y <= 12; y++)
        CT_CHECKF(t, lit(&s, 14, y) && !lit(&s, 13, y) && !lit(&s, 15, y), "line strip: column 14 at y %u", y);
    /* triangles: the rectangles, and the columns between them empty */
    for (y = 9; y < H; y++) {
        CT_CHECKF(t, lit(&s, 0, y) && lit(&s, 3, y) && !lit(&s, 4, y), "triangle list: row %u", y);
        CT_CHECKF(t, lit(&s, 5, y) && lit(&s, 8, y) && !lit(&s, 9, y), "triangle strip: row %u", y);
        if (fans)
            CT_CHECKF(t, lit(&s, 10, y) && lit(&s, 11, y) && !lit(&s, 12, y), "triangle fan: row %u", y);
    }
    if (!fans)
        ct_note(t, "triangle fans not drawn (VK_KHR_portability_subset)");
    for (k = 0; k < 6; k++)
        if (pipes[k])
            vkDestroyPipeline(s.e.device, pipes[k], NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    ct_buffer_fini(&s.e, &vb);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(draw_indirect,
        "Drawing Commands > Programmable Primitive Shading: vkCmdDrawIndirect and vkCmdDrawIndexedIndirect read "
        "VkDrawIndirectCommand / VkDrawIndexedIndirectCommand from the buffer at offset (drawCount of them, "
        "stride apart, with multiDrawIndirect) and draw as the direct commands with those parameters",
        DR_COMMANDS "vkCmdDrawIndirect vkCmdDrawIndexedIndirect vkCmdBindIndexBuffer")
{
    static const uint16_t indices[6] = { 9, 0, 1, 2, 9, 9 };
    struct ct_scene s;
    struct ct_target c;
    struct ct_buffer buf;
    struct points want;
    VkDrawIndirectCommand d[2] = { { 4, 2, 2, 3 }, { 2, 1, 10, 7 } };
    VkDrawIndexedIndirectCommand x[2] = { { 3, 2, 1, 4, 10 }, { 1, 1, 3, 12, 14 } };
    VkShaderModule vs, fs;
    VkPipeline pipe;
    VkCommandBuffer cb;
    int multi;

    ct_scene_init(t, &s);
    multi = s.f.f2.features.multiDrawIndirect;
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 1024, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &buf);
    memcpy(buf.data + 16, &d[0], sizeof(d[0])); /* stride 32 */
    memcpy(buf.data + 48, &d[1], sizeof(d[1]));
    memcpy(buf.data + 256, &x[0], sizeof(x[0])); /* stride 24 */
    memcpy(buf.data + 280, &x[1], sizeof(x[1]));
    memcpy(buf.data + 512, indices, sizeof(indices));
    vs = ct_shader_module(t, &s.e, index_vert, sizeof(index_vert));
    fs = ct_shader_module(t, &s.e, flat_frag, sizeof(flat_frag));
    pipe = point_pipeline(t, &s, vs, fs, NULL);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindIndexBuffer(cb, buf.buffer, 512, VK_INDEX_TYPE_UINT16);
    if (multi) {
        vkCmdDrawIndirect(cb, buf.buffer, 16, 2, 32);
        vkCmdDrawIndexedIndirect(cb, buf.buffer, 256, 2, 24);
    } else {
        vkCmdDrawIndirect(cb, buf.buffer, 16, 1, 0);
        vkCmdDrawIndirect(cb, buf.buffer, 48, 1, 0);
        vkCmdDrawIndexedIndirect(cb, buf.buffer, 256, 1, 0);
        vkCmdDrawIndexedIndirect(cb, buf.buffer, 280, 1, 0);
    }
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    memset(&want, 0, sizeof(want));
    want.bg = CT_BLACK;
    add_index_points(&want, 2, 4, 3, 2);   /* d[0] */
    add_index_points(&want, 10, 2, 7, 1);  /* d[1] */
    add_index_points(&want, 4, 3, 10, 2);  /* x[0]: indices 0 1 2 + 4 */
    add_index_points(&want, 14, 1, 14, 1); /* x[1]: index 2 + 12 */
    ct_check_rgba8(t, s.out.data, W, H, expect_points, &want, 0, multi ? "points (drawCount 2)" : "points");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_buffer_fini(&s.e, &buf);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(draw_indirect_count,
        "Drawing Commands > Programmable Primitive Shading: vkCmdDrawIndirectCount and "
        "vkCmdDrawIndexedIndirectCount draw min(the count buffer's value, maxDrawCount) commands "
        "(drawIndirectCount, Vulkan 1.2; counts within maxDrawIndirectCount, VUID-vkCmdDrawIndirectCount-countBuffer-02717)",
        DR_COMMANDS "vkCmdDrawIndirectCount vkCmdDrawIndexedIndirectCount vkCmdBindIndexBuffer")
{
    static const uint16_t indices[4] = { 0, 1, 2, 3 };
    struct ct_scene s;
    struct ct_target c;
    struct ct_buffer buf;
    struct points want;
    VkDrawIndirectCommand d[3] = { { 1, 1, 1, 1 }, { 1, 1, 3, 1 }, { 1, 1, 5, 1 } };
    VkDrawIndexedIndirectCommand x[3] = { { 1, 1, 0, 1, 3 }, { 1, 1, 1, 2, 3 }, { 1, 1, 2, 3, 3 } };
    uint32_t counts[4] = { 2, 5, 0, 3 }, lim, i, n;
    VkShaderModule vs, fs;
    VkPipeline pipe;
    VkCommandBuffer cb;

    ct_scene_init(t, &s);
    if (!s.f.v12.drawIndirectCount)
        ct_skip(t, "drawIndirectCount is not supported");
    lim = s.e.props.limits.maxDrawIndirectCount; /* 1 without multiDrawIndirect */
    for (i = 0; i < 4; i++)
        counts[i] = CT_MIN(counts[i], lim);
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 1024, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &buf);
    memcpy(buf.data, d, sizeof(d));
    memcpy(buf.data + 256, x, sizeof(x));
    memcpy(buf.data + 512, counts, sizeof(counts));
    memcpy(buf.data + 768, indices, sizeof(indices));
    /* the same commands one row down, for the second count */
    d[0].firstInstance = d[1].firstInstance = d[2].firstInstance = 2;
    memcpy(buf.data + 128, d, sizeof(d));
    x[0].firstInstance = x[1].firstInstance = x[2].firstInstance = 4;
    memcpy(buf.data + 384, x, sizeof(x));
    vs = ct_shader_module(t, &s.e, index_vert, sizeof(index_vert));
    fs = ct_shader_module(t, &s.e, flat_frag, sizeof(flat_frag));
    pipe = point_pipeline(t, &s, vs, fs, NULL);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindIndexBuffer(cb, buf.buffer, 768, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndirectCount(cb, buf.buffer, 0, buf.buffer, 512, 3, sizeof(VkDrawIndirectCommand));   /* count 2 */
    vkCmdDrawIndirectCount(cb, buf.buffer, 128, buf.buffer, 516, 2, sizeof(VkDrawIndirectCommand)); /* 5, max 2 */
    vkCmdDrawIndexedIndirectCount(cb, buf.buffer, 256, buf.buffer, 520, 3,
                                  sizeof(VkDrawIndexedIndirectCommand)); /* count 0 */
    vkCmdDrawIndexedIndirectCount(cb, buf.buffer, 384, buf.buffer, 524, 3,
                                  sizeof(VkDrawIndexedIndirectCommand)); /* count 3 */
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    memset(&want, 0, sizeof(want));
    want.bg = CT_BLACK;
    for (i = 0; i < counts[0]; i++) /* count 2, maxDrawCount 3 */
        add_index_points(&want, 1 + 2 * i, 1, 1, 1);
    n = CT_MIN(counts[1], 2u); /* count 5, maxDrawCount 2 */
    for (i = 0; i < n; i++)
        add_index_points(&want, 1 + 2 * i, 1, 2, 1);
    for (i = 0; i < counts[3]; i++) /* indexed, count 3: index i + vertexOffset */
        add_index_points(&want, 1 + 2 * i, 1, 4, 1);
    ct_check_rgba8(t, s.out.data, W, H, expect_points, &want, 0, "points");
    if (lim < 5)
        ct_note(t, "counts limited to maxDrawIndirectCount %u", lim);
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_buffer_fini(&s.e, &buf);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(draw_parameters,
        "Drawing Commands, Shader Interfaces > Built-In Variables: BaseVertex is firstVertex (non-indexed draws) "
        "or vertexOffset (indexed draws), BaseInstance is firstInstance, DrawIndex is 0 for a draw that is not "
        "part of a multi-draw; for direct and indirect draws (shaderDrawParameters)",
        DR_COMMANDS "vkCmdDrawIndexed vkCmdDrawIndirect vkCmdDrawIndexedIndirect vkCmdBindIndexBuffer")
{
    static const uint16_t indices[2] = { 0, 1 };
    struct ct_scene s;
    struct ct_target c;
    struct ct_buffer buf;
    struct points want;
    VkDrawIndirectCommand d = { 1, 2, 9, 9 };
    VkDrawIndexedIndirectCommand x = { 2, 1, 0, 12, 13 };
    VkShaderModule vs, fs;
    VkPipeline pipe;
    VkCommandBuffer cb;

    ct_scene_init(t, &s);
    if (!s.f.v11.shaderDrawParameters)
        ct_skip(t, "shaderDrawParameters is not supported");
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 1024, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &buf);
    memcpy(buf.data, &d, sizeof(d));
    memcpy(buf.data + 64, &x, sizeof(x));
    memcpy(buf.data + 128, indices, sizeof(indices));
    vs = ct_shader_module(t, &s.e, params_vert, sizeof(params_vert));
    fs = ct_shader_module(t, &s.e, flat_frag, sizeof(flat_frag));
    pipe = point_pipeline(t, &s, vs, fs, NULL);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindIndexBuffer(cb, buf.buffer, 128, VK_INDEX_TYPE_UINT16);
    vkCmdDraw(cb, 2, 1, 3, 5);              /* x 3..4, y 5 */
    vkCmdDrawIndexed(cb, 2, 1, 0, 6, 2);    /* x 6..7, y 2 */
    vkCmdDrawIndirect(cb, buf.buffer, 0, 1, 0);         /* x 9, y 9..10 */
    vkCmdDrawIndexedIndirect(cb, buf.buffer, 64, 1, 0); /* x 12..13, y 13 */
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    memset(&want, 0, sizeof(want));
    want.bg = CT_BLACK;
    add_point(&want, 3, 5, CT_RGBA(3, 5, 0, 255));
    add_point(&want, 4, 5, CT_RGBA(3, 5, 0, 255));
    add_point(&want, 6, 2, CT_RGBA(6, 2, 0, 255));
    add_point(&want, 7, 2, CT_RGBA(6, 2, 0, 255));
    add_point(&want, 9, 9, CT_RGBA(9, 9, 0, 255));
    add_point(&want, 9, 10, CT_RGBA(9, 9, 0, 255));
    add_point(&want, 12, 13, CT_RGBA(12, 13, 0, 255));
    add_point(&want, 13, 13, CT_RGBA(12, 13, 0, 255));
    ct_check_rgba8(t, s.out.data, W, H, expect_points, &want, 0, "points colored (BaseVertex, BaseInstance, DrawIndex)");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_buffer_fini(&s.e, &buf);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(draw_index_multi_draw,
        "Drawing Commands, Shader Interfaces > Built-In Variables: DrawIndex is the index of the draw within a "
        "multi-draw: vkCmdDrawIndirectCount with drawIndirectCount, vkCmdDrawIndirect with drawCount > 1 with "
        "multiDrawIndirect (shaderDrawParameters; maxDrawIndirectCount of 3 or more)",
        DR_COMMANDS "vkCmdDrawIndirect vkCmdDrawIndirectCount")
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_buffer buf;
    struct points want;
    VkDrawIndirectCommand d[3] = { { 1, 1, 1, 1 }, { 1, 1, 3, 1 }, { 1, 1, 5, 1 } };
    uint32_t count = 3, i;
    VkShaderModule vs, fs;
    VkPipeline pipe;
    VkCommandBuffer cb;
    int multi, counted;

    ct_scene_init(t, &s);
    if (!s.f.v11.shaderDrawParameters)
        ct_skip(t, "shaderDrawParameters is not supported");
    multi = s.f.f2.features.multiDrawIndirect;
    counted = s.f.v12.drawIndirectCount;
    if (!multi && !counted)
        ct_skip(t, "neither multiDrawIndirect nor drawIndirectCount is supported");
    if (s.e.props.limits.maxDrawIndirectCount < 3)
        ct_skip(t, "maxDrawIndirectCount is %u: no multi-draw of 3", s.e.props.limits.maxDrawIndirectCount);
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 1024, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, &buf);
    memcpy(buf.data, d, sizeof(d));
    for (i = 0; i < 3; i++)
        d[i].firstInstance = 6;
    memcpy(buf.data + 128, d, sizeof(d));
    memcpy(buf.data + 256, &count, sizeof(count));
    vs = ct_shader_module(t, &s.e, params_vert, sizeof(params_vert));
    fs = ct_shader_module(t, &s.e, flat_frag, sizeof(flat_frag));
    pipe = point_pipeline(t, &s, vs, fs, NULL);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    if (counted)
        vkCmdDrawIndirectCount(cb, buf.buffer, 0, buf.buffer, 256, 3, sizeof(VkDrawIndirectCommand));
    if (multi)
        vkCmdDrawIndirect(cb, buf.buffer, 128, 3, sizeof(VkDrawIndirectCommand));
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    memset(&want, 0, sizeof(want));
    want.bg = CT_BLACK;
    for (i = 0; i < 3; i++) {
        if (counted)
            add_point(&want, 1 + 2 * i, 1, CT_RGBA(1 + 2 * i, 1, i, 255));
        if (multi)
            add_point(&want, 1 + 2 * i, 6, CT_RGBA(1 + 2 * i, 6, i, 255));
    }
    ct_check_rgba8(t, s.out.data, W, H, expect_points, &want, 0, "points colored (BaseVertex, BaseInstance, DrawIndex)");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_buffer_fini(&s.e, &buf);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(vertex_attributes,
        "Fixed-Function Vertex Processing > Vertex Attributes, Vertex Input Extraction: attributes are read at "
        "their offset within each binding stride and converted from their format (R32G32_SFLOAT, R8G8B8A8_UNORM "
        "normalized to [0, 1], R32_SFLOAT with the missing components (0, 0, 1)); vkCmdBindVertexBuffers offsets "
        "apply; vertices of per-instance bindings advance once per instance",
        DR_COMMANDS)
{
    struct vertex {
        float pos[2];
        uint8_t color[4];
        float scale;
    } verts[3] = {
        { { 0.0f, 0.0f }, { 0, 0, 0, 0 }, 0.0f }, /* skipped by the binding offset */
        { { 0.0f, 0.0f }, { 255, 128, 64, 255 }, 0.5f },
        { { 0.0f, 0.0f }, { 0, 255, 0, 255 }, 1.0f },
    };
    static const uint8_t instance_colors[2][4] = { { 0, 0, 255, 255 }, { 255, 255, 0, 255 } };
    VkVertexInputBindingDescription bindings[2] = {
        { 0, sizeof(struct vertex), VK_VERTEX_INPUT_RATE_VERTEX },
        { 1, 4, VK_VERTEX_INPUT_RATE_INSTANCE },
    };
    VkVertexInputAttributeDescription attrs[3] = {
        { 0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(struct vertex, pos) },
        { 1, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(struct vertex, color) },
        { 2, 0, VK_FORMAT_R32_SFLOAT, offsetof(struct vertex, scale) },
    };
    VkVertexInputAttributeDescription attrs2[3];
    VkPipelineVertexInputStateCreateInfo vi;
    struct ct_scene s;
    struct ct_target c;
    struct ct_buffer vb;
    struct points want;
    VkShaderModule vs, fs;
    VkPipeline per_vertex, per_instance;
    VkBuffer buffers[2];
    VkDeviceSize offsets[2];
    VkCommandBuffer cb;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 1024, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb);
    verts[1].pos[0] = ndc(2), verts[1].pos[1] = ndc(3);
    verts[2].pos[0] = ndc(7), verts[2].pos[1] = ndc(3);
    memcpy(vb.data + 64, verts, sizeof(verts));
    memcpy(vb.data + 512, instance_colors, sizeof(instance_colors));
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = bindings;
    vi.vertexAttributeDescriptionCount = 3;
    vi.pVertexAttributeDescriptions = attrs;
    vs = ct_shader_module(t, &s.e, attr_vert, sizeof(attr_vert));
    fs = ct_shader_module(t, &s.e, flat_frag, sizeof(flat_frag));
    per_vertex = point_pipeline(t, &s, vs, fs, &vi);
    /* color from binding 1 (per instance), position and scale per vertex */
    memcpy(attrs2, attrs, sizeof(attrs));
    attrs2[1].binding = 1;
    attrs2[1].offset = 0;
    vi.vertexBindingDescriptionCount = 2;
    vi.pVertexAttributeDescriptions = attrs2;
    per_instance = point_pipeline(t, &s, vs, fs, &vi);
    buffers[0] = buffers[1] = vb.buffer;
    offsets[0] = 64 + sizeof(struct vertex);
    offsets[1] = 512 - 5 * 4; /* instance-rate elements are numbered from 0, not firstInstance */
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, per_vertex);
    vkCmdBindVertexBuffers(cb, 0, 1, buffers, offsets);
    vkCmdDraw(cb, 2, 1, 0, 0); /* row 3 */
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, per_instance);
    vkCmdBindVertexBuffers(cb, 0, 2, buffers, offsets);
    vkCmdDraw(cb, 2, 2, 0, 5); /* instances 5, 6 of binding 1 elements 0 and 1: rows 8 and 9 */
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    memset(&want, 0, sizeof(want));
    want.bg = CT_BLACK;
    add_point(&want, 2, 3, CT_RGBA(128, 64, 32, 255));
    add_point(&want, 7, 3, CT_RGBA(0, 255, 0, 255));
    add_point(&want, 2, 8, CT_RGBA(0, 0, 128, 255)); /* instance color * scale 0.5 */
    add_point(&want, 7, 8, CT_RGBA(0, 0, 255, 255));
    add_point(&want, 2, 9, CT_RGBA(128, 128, 0, 255));
    add_point(&want, 7, 9, CT_RGBA(255, 255, 0, 255));
    ct_check_rgba8(t, s.out.data, W, H, expect_points, &want, 1, "points");
    vkDestroyPipeline(s.e.device, per_vertex, NULL);
    vkDestroyPipeline(s.e.device, per_instance, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_buffer_fini(&s.e, &vb);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(vertex_attribute_divisor,
        "Fixed-Function Vertex Processing > Vertex Attribute Divisor in Instanced Rendering: with "
        "VkPipelineVertexInputDivisorStateCreateInfo, a per-instance binding with divisor d advances once every "
        "d instances (vertexAttributeInstanceRateDivisor, required from Vulkan 1.4)",
        DR_COMMANDS)
{
    static const uint8_t colors[3][4] = { { 255, 0, 0, 255 }, { 0, 255, 0, 255 }, { 0, 0, 255, 255 } };
    float pos[2];
    VkVertexInputBindingDescription bindings[2] = {
        { 0, 8, VK_VERTEX_INPUT_RATE_VERTEX },
        { 1, 4, VK_VERTEX_INPUT_RATE_INSTANCE },
    };
    VkVertexInputAttributeDescription attrs[3] = {
        { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 },
        { 1, 1, VK_FORMAT_R8G8B8A8_UNORM, 0 },
        { 2, 0, VK_FORMAT_R32_SFLOAT, 8 },
    };
    VkVertexInputBindingDivisorDescription divisor = { 1, 2 };
    VkPipelineVertexInputDivisorStateCreateInfo div;
    VkPipelineVertexInputStateCreateInfo vi;
    struct ct_scene s;
    struct ct_target c;
    struct ct_buffer vb;
    struct points want;
    VkShaderModule vs, fs;
    VkPipeline pipe;
    VkBuffer buffers[2];
    VkDeviceSize offsets[2] = { 0, 256 };
    VkCommandBuffer cb;
    uint32_t i;
    float one = 1.0f;

    ct_scene_init(t, &s);
    if (s.e.version < VK_API_VERSION_1_4)
        ct_skip(t, "vertex attribute divisors are core in Vulkan 1.4");
    CT_REQUIRE(t, s.f.v14.vertexAttributeInstanceRateDivisor);
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 1024, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb);
    bindings[0].stride = 12;
    pos[0] = ndc(4);
    pos[1] = ndc(2);
    memcpy(vb.data, pos, sizeof(pos));
    memcpy(vb.data + 8, &one, sizeof(one));
    memcpy(vb.data + 256, colors, sizeof(colors));
    memset(&div, 0, sizeof(div));
    div.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO;
    div.vertexBindingDivisorCount = 1;
    div.pVertexBindingDivisors = &divisor;
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.pNext = &div;
    vi.vertexBindingDescriptionCount = 2;
    vi.pVertexBindingDescriptions = bindings;
    vi.vertexAttributeDescriptionCount = 3;
    vi.pVertexAttributeDescriptions = attrs;
    vs = ct_shader_module(t, &s.e, attr_vert, sizeof(attr_vert));
    fs = ct_shader_module(t, &s.e, flat_frag, sizeof(flat_frag));
    pipe = point_pipeline(t, &s, vs, fs, &vi);
    buffers[0] = buffers[1] = vb.buffer;
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindVertexBuffers(cb, 0, 2, buffers, offsets);
    vkCmdDraw(cb, 1, 6, 0, 0); /* instances 0..5 in rows 2..7: colors 0 0 1 1 2 2 */
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    memset(&want, 0, sizeof(want));
    want.bg = CT_BLACK;
    for (i = 0; i < 6; i++)
        add_point(&want, 4, 2 + i, CT_RGBA(colors[i / 2][0], colors[i / 2][1], colors[i / 2][2], 255));
    ct_check_rgba8(t, s.out.data, W, H, expect_points, &want, 0, "points of instances 0..5, divisor 2");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_buffer_fini(&s.e, &vb);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(vertex_binding_stride_dynamic,
        "Fixed-Function Vertex Processing > Vertex Input Description, Pipelines > Dynamic State: with "
        "VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE, the strides given to vkCmdBindVertexBuffers2 are used "
        "instead of the pipeline's (Vulkan 1.3)",
        DR_COMMANDS "vkCmdBindVertexBuffers2")
{
    VkVertexInputBindingDescription binding = { 0, 8, VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription attr = { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 };
    VkPipelineVertexInputStateCreateInfo vi;
    struct ct_scene s;
    struct ct_target c;
    struct ct_buffer vb;
    struct ct_gfx g;
    VkShaderModule vs;
    VkPipeline pipe;
    VkCommandBuffer cb;
    VkDeviceSize offset = 0, stride, size = 6 * 8;
    float *v;
    uint32_t k, x, y;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 1024, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb);
    v = (float *)vb.data;
    for (k = 0; k < 6; k++) { /* element k: pixel (k, 2) if even, (k, 5) if odd */
        v[2 * k] = ndc(k);
        v[2 * k + 1] = ndc(k & 1 ? 5 : 2);
    }
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions = &attr;
    vs = ct_shader_module(t, &s.e, pos_vert, sizeof(pos_vert));
    ct_gfx_init(&g, vs, s.fs, s.layout, W, H);
    g.vertex = &vi;
    g.ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    for (k = 0; k < 2; k++) { /* stride 16: elements 0, 2, 4; stride 8: all six */
        stride = k ? 8 : 16;
        cb = ct_begin_commands(t, &s.e);
        ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vkCmdBindVertexBuffers2(cb, 0, 1, &vb.buffer, &offset, &size, &stride);
        set_color(cb, s.layout, ct_white);
        vkCmdDraw(cb, k ? 6 : 3, 1, 0, 0);
        vkCmdEndRendering(cb);
        ct_scene_read(t, &s, cb, &c);
        for (y = 0; y < H; y++)
            for (x = 0; x < W; x++) {
                int drawn = ct_texel32(s.out.data, W, x, y) == CT_WHITE;
                int expected = x < 6 && ((y == 2 && !(x & 1)) || (k && y == 5 && (x & 1)));
                CT_CHECKF(t, drawn == expected, "stride %u: texel (%u, %u) %s", (unsigned)stride, x, y,
                          drawn ? "drawn" : "not drawn");
            }
    }
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, vs, NULL);
    ct_buffer_fini(&s.e, &vb);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

static const struct ct_test *const tests[] = {
    CT_REF(draw_vertex_instance_index),
    CT_REF(draw_indexed),
    CT_REF(draw_indexed_uint8),
    CT_REF(primitive_restart),
    CT_REF(primitive_topologies),
    CT_REF(draw_indirect),
    CT_REF(draw_indirect_count),
    CT_REF(draw_parameters),
    CT_REF(draw_index_multi_draw),
    CT_REF(vertex_attributes),
    CT_REF(vertex_attribute_divisor),
    CT_REF(vertex_binding_stride_dynamic),
};

CT_SUITE(drawing, tests)
