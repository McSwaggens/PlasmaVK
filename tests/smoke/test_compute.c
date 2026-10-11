/* Shader modules, pipeline layouts and compute pipelines (section 14), and
 * dispatches with push constants and buffer device addresses (sections 12
 * and 13).
 *
 * Shaders: GLSL in the comment above each array, compiled with
 *   glslangValidator -V --target-env vulkan1.3 */
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <string.h>

#include "../pvk_test.h"

/*
 * #version 450
 * #extension GL_EXT_buffer_reference : require
 * layout(local_size_x = 64) in;
 * layout(buffer_reference, std430, buffer_reference_align = 4) buffer Words { uint data[]; };
 * layout(push_constant) uniform Push { Words src; Words dst; uint count; uint mul; uint add; } pc;
 * void main()
 * {
 *     uint i = gl_GlobalInvocationID.x;
 *     if (i < pc.count)
 *         pc.dst.data[i] = pc.src.data[i] * pc.mul + pc.add;
 * }
 */
static const uint32_t spv_scale[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000038, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x000014e3, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x000014e4, 0x00000001, 0x0007000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b,
    0x00000016, 0x00060010, 0x00000004, 0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003,
    0x00000002, 0x000001c2, 0x00070004, 0x455f4c47, 0x625f5458, 0x65666675, 0x65725f72, 0x65726566,
    0x0065636e, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008, 0x00000069,
    0x00080005, 0x0000000b, 0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044,
    0x00040005, 0x00000012, 0x68737550, 0x00000000, 0x00040006, 0x00000012, 0x00000000, 0x00637273,
    0x00040006, 0x00000012, 0x00000001, 0x00747364, 0x00050006, 0x00000012, 0x00000002, 0x6e756f63,
    0x00000074, 0x00040006, 0x00000012, 0x00000003, 0x006c756d, 0x00040006, 0x00000012, 0x00000004,
    0x00646461, 0x00040005, 0x00000014, 0x64726f57, 0x00000073, 0x00050006, 0x00000014, 0x00000000,
    0x61746164, 0x00000000, 0x00030005, 0x00000016, 0x00006370, 0x00040047, 0x0000000b, 0x0000000b,
    0x0000001c, 0x00030047, 0x00000012, 0x00000002, 0x00050048, 0x00000012, 0x00000000, 0x00000023,
    0x00000000, 0x00050048, 0x00000012, 0x00000001, 0x00000023, 0x00000008, 0x00050048, 0x00000012,
    0x00000002, 0x00000023, 0x00000010, 0x00050048, 0x00000012, 0x00000003, 0x00000023, 0x00000014,
    0x00050048, 0x00000012, 0x00000004, 0x00000023, 0x00000018, 0x00040047, 0x00000013, 0x00000006,
    0x00000004, 0x00030047, 0x00000014, 0x00000002, 0x00050048, 0x00000014, 0x00000000, 0x00000023,
    0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006,
    0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00040017, 0x00000009,
    0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000009, 0x0004003b, 0x0000000a,
    0x0000000b, 0x00000001, 0x0004002b, 0x00000006, 0x0000000c, 0x00000000, 0x00040020, 0x0000000d,
    0x00000001, 0x00000006, 0x00030027, 0x00000011, 0x000014e5, 0x0007001e, 0x00000012, 0x00000011,
    0x00000011, 0x00000006, 0x00000006, 0x00000006, 0x0003001d, 0x00000013, 0x00000006, 0x0003001e,
    0x00000014, 0x00000013, 0x00040020, 0x00000011, 0x000014e5, 0x00000014, 0x00040020, 0x00000015,
    0x00000009, 0x00000012, 0x0004003b, 0x00000015, 0x00000016, 0x00000009, 0x00040015, 0x00000017,
    0x00000020, 0x00000001, 0x0004002b, 0x00000017, 0x00000018, 0x00000002, 0x00040020, 0x00000019,
    0x00000009, 0x00000006, 0x00020014, 0x0000001c, 0x0004002b, 0x00000017, 0x00000020, 0x00000001,
    0x00040020, 0x00000021, 0x00000009, 0x00000011, 0x0004002b, 0x00000017, 0x00000024, 0x00000000,
    0x00040020, 0x00000029, 0x000014e5, 0x00000006, 0x0004002b, 0x00000017, 0x0000002c, 0x00000003,
    0x0004002b, 0x00000017, 0x00000030, 0x00000004, 0x0004002b, 0x00000006, 0x00000035, 0x00000040,
    0x0004002b, 0x00000006, 0x00000036, 0x00000001, 0x0006002c, 0x00000009, 0x00000037, 0x00000035,
    0x00000036, 0x00000036, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
    0x00000005, 0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x00050041, 0x0000000d, 0x0000000e,
    0x0000000b, 0x0000000c, 0x0004003d, 0x00000006, 0x0000000f, 0x0000000e, 0x0003003e, 0x00000008,
    0x0000000f, 0x0004003d, 0x00000006, 0x00000010, 0x00000008, 0x00050041, 0x00000019, 0x0000001a,
    0x00000016, 0x00000018, 0x0004003d, 0x00000006, 0x0000001b, 0x0000001a, 0x000500b0, 0x0000001c,
    0x0000001d, 0x00000010, 0x0000001b, 0x000300f7, 0x0000001f, 0x00000000, 0x000400fa, 0x0000001d,
    0x0000001e, 0x0000001f, 0x000200f8, 0x0000001e, 0x00050041, 0x00000021, 0x00000022, 0x00000016,
    0x00000020, 0x0004003d, 0x00000011, 0x00000023, 0x00000022, 0x0004003d, 0x00000006, 0x00000025,
    0x00000008, 0x00050041, 0x00000021, 0x00000026, 0x00000016, 0x00000024, 0x0004003d, 0x00000011,
    0x00000027, 0x00000026, 0x0004003d, 0x00000006, 0x00000028, 0x00000008, 0x00060041, 0x00000029,
    0x0000002a, 0x00000027, 0x00000024, 0x00000028, 0x0006003d, 0x00000006, 0x0000002b, 0x0000002a,
    0x00000002, 0x00000004, 0x00050041, 0x00000019, 0x0000002d, 0x00000016, 0x0000002c, 0x0004003d,
    0x00000006, 0x0000002e, 0x0000002d, 0x00050084, 0x00000006, 0x0000002f, 0x0000002b, 0x0000002e,
    0x00050041, 0x00000019, 0x00000031, 0x00000016, 0x00000030, 0x0004003d, 0x00000006, 0x00000032,
    0x00000031, 0x00050080, 0x00000006, 0x00000033, 0x0000002f, 0x00000032, 0x00060041, 0x00000029,
    0x00000034, 0x00000023, 0x00000024, 0x00000025, 0x0005003e, 0x00000034, 0x00000033, 0x00000002,
    0x00000004, 0x000200f9, 0x0000001f, 0x000200f8, 0x0000001f, 0x000100fd, 0x00010038,
};

/*
 * #version 450
 * #extension GL_EXT_buffer_reference : require
 * layout(local_size_x = 4, local_size_y = 2, local_size_z = 2) in;
 * layout(buffer_reference, std430, buffer_reference_align = 16) buffer Cells { uvec4 cell[]; };
 * layout(push_constant) uniform Push { Cells cells; uint width; uint height; } pc;
 * void main()
 * {
 *     uvec3 g = gl_GlobalInvocationID;
 *     pc.cells.cell[(g.z * pc.height + g.y) * pc.width + g.x] = uvec4(gl_WorkGroupID, gl_LocalInvocationIndex + 1u);
 * }
 */
static const uint32_t spv_ids[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x0000003c, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x000014e3, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x000014e4, 0x00000001, 0x0009000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b,
    0x00000013, 0x0000002e, 0x00000031, 0x00060010, 0x00000004, 0x00000011, 0x00000004, 0x00000002,
    0x00000002, 0x00030003, 0x00000002, 0x000001c2, 0x00070004, 0x455f4c47, 0x625f5458, 0x65666675,
    0x65725f72, 0x65726566, 0x0065636e, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005,
    0x00000009, 0x00000067, 0x00080005, 0x0000000b, 0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f,
    0x496e6f69, 0x00000044, 0x00040005, 0x0000000e, 0x68737550, 0x00000000, 0x00050006, 0x0000000e,
    0x00000000, 0x6c6c6563, 0x00000073, 0x00050006, 0x0000000e, 0x00000001, 0x74646977, 0x00000068,
    0x00050006, 0x0000000e, 0x00000002, 0x67696568, 0x00007468, 0x00040005, 0x00000011, 0x6c6c6543,
    0x00000073, 0x00050006, 0x00000011, 0x00000000, 0x6c6c6563, 0x00000000, 0x00030005, 0x00000013,
    0x00006370, 0x00060005, 0x0000002e, 0x575f6c67, 0x476b726f, 0x70756f72, 0x00004449, 0x00080005,
    0x00000031, 0x4c5f6c67, 0x6c61636f, 0x6f766e49, 0x69746163, 0x6e496e6f, 0x00786564, 0x00040047,
    0x0000000b, 0x0000000b, 0x0000001c, 0x00030047, 0x0000000e, 0x00000002, 0x00050048, 0x0000000e,
    0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000000e, 0x00000001, 0x00000023, 0x00000008,
    0x00050048, 0x0000000e, 0x00000002, 0x00000023, 0x0000000c, 0x00040047, 0x00000010, 0x00000006,
    0x00000010, 0x00030047, 0x00000011, 0x00000002, 0x00050048, 0x00000011, 0x00000000, 0x00000023,
    0x00000000, 0x00040047, 0x0000002e, 0x0000000b, 0x0000001a, 0x00040047, 0x00000031, 0x0000000b,
    0x0000001d, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006,
    0x00000020, 0x00000000, 0x00040017, 0x00000007, 0x00000006, 0x00000003, 0x00040020, 0x00000008,
    0x00000007, 0x00000007, 0x00040020, 0x0000000a, 0x00000001, 0x00000007, 0x0004003b, 0x0000000a,
    0x0000000b, 0x00000001, 0x00030027, 0x0000000d, 0x000014e5, 0x0005001e, 0x0000000e, 0x0000000d,
    0x00000006, 0x00000006, 0x00040017, 0x0000000f, 0x00000006, 0x00000004, 0x0003001d, 0x00000010,
    0x0000000f, 0x0003001e, 0x00000011, 0x00000010, 0x00040020, 0x0000000d, 0x000014e5, 0x00000011,
    0x00040020, 0x00000012, 0x00000009, 0x0000000e, 0x0004003b, 0x00000012, 0x00000013, 0x00000009,
    0x00040015, 0x00000014, 0x00000020, 0x00000001, 0x0004002b, 0x00000014, 0x00000015, 0x00000000,
    0x00040020, 0x00000016, 0x00000009, 0x0000000d, 0x0004002b, 0x00000006, 0x00000019, 0x00000002,
    0x00040020, 0x0000001a, 0x00000007, 0x00000006, 0x0004002b, 0x00000014, 0x0000001d, 0x00000002,
    0x00040020, 0x0000001e, 0x00000009, 0x00000006, 0x0004002b, 0x00000006, 0x00000022, 0x00000001,
    0x0004002b, 0x00000014, 0x00000026, 0x00000001, 0x0004002b, 0x00000006, 0x0000002a, 0x00000000,
    0x0004003b, 0x0000000a, 0x0000002e, 0x00000001, 0x00040020, 0x00000030, 0x00000001, 0x00000006,
    0x0004003b, 0x00000030, 0x00000031, 0x00000001, 0x00040020, 0x00000038, 0x000014e5, 0x0000000f,
    0x0004002b, 0x00000006, 0x0000003a, 0x00000004, 0x0006002c, 0x00000007, 0x0000003b, 0x0000003a,
    0x00000019, 0x00000019, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
    0x00000005, 0x0004003b, 0x00000008, 0x00000009, 0x00000007, 0x0004003d, 0x00000007, 0x0000000c,
    0x0000000b, 0x0003003e, 0x00000009, 0x0000000c, 0x00050041, 0x00000016, 0x00000017, 0x00000013,
    0x00000015, 0x0004003d, 0x0000000d, 0x00000018, 0x00000017, 0x00050041, 0x0000001a, 0x0000001b,
    0x00000009, 0x00000019, 0x0004003d, 0x00000006, 0x0000001c, 0x0000001b, 0x00050041, 0x0000001e,
    0x0000001f, 0x00000013, 0x0000001d, 0x0004003d, 0x00000006, 0x00000020, 0x0000001f, 0x00050084,
    0x00000006, 0x00000021, 0x0000001c, 0x00000020, 0x00050041, 0x0000001a, 0x00000023, 0x00000009,
    0x00000022, 0x0004003d, 0x00000006, 0x00000024, 0x00000023, 0x00050080, 0x00000006, 0x00000025,
    0x00000021, 0x00000024, 0x00050041, 0x0000001e, 0x00000027, 0x00000013, 0x00000026, 0x0004003d,
    0x00000006, 0x00000028, 0x00000027, 0x00050084, 0x00000006, 0x00000029, 0x00000025, 0x00000028,
    0x00050041, 0x0000001a, 0x0000002b, 0x00000009, 0x0000002a, 0x0004003d, 0x00000006, 0x0000002c,
    0x0000002b, 0x00050080, 0x00000006, 0x0000002d, 0x00000029, 0x0000002c, 0x0004003d, 0x00000007,
    0x0000002f, 0x0000002e, 0x0004003d, 0x00000006, 0x00000032, 0x00000031, 0x00050080, 0x00000006,
    0x00000033, 0x00000032, 0x00000022, 0x00050051, 0x00000006, 0x00000034, 0x0000002f, 0x00000000,
    0x00050051, 0x00000006, 0x00000035, 0x0000002f, 0x00000001, 0x00050051, 0x00000006, 0x00000036,
    0x0000002f, 0x00000002, 0x00070050, 0x0000000f, 0x00000037, 0x00000034, 0x00000035, 0x00000036,
    0x00000033, 0x00060041, 0x00000038, 0x00000039, 0x00000018, 0x00000015, 0x0000002d, 0x0005003e,
    0x00000039, 0x00000037, 0x00000002, 0x00000010, 0x000100fd, 0x00010038,
};

/*
 * #version 450
 * #extension GL_EXT_buffer_reference : require
 * layout(local_size_x = 1) in;
 * layout(buffer_reference, std430, buffer_reference_align = 4) buffer Args { uint x; uint y; uint z; };
 * layout(push_constant) uniform Push { Args args; uint x; uint y; uint z; } pc;
 * void main()
 * {
 *     pc.args.x = pc.x;
 *     pc.args.y = pc.y;
 *     pc.args.z = pc.z;
 * }
 */
static const uint32_t spv_args[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000026, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x000014e3, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x000014e4, 0x00000001, 0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b,
    0x00060010, 0x00000004, 0x00000011, 0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002,
    0x000001c2, 0x00070004, 0x455f4c47, 0x625f5458, 0x65666675, 0x65725f72, 0x65726566, 0x0065636e,
    0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x00000008, 0x68737550, 0x00000000,
    0x00050006, 0x00000008, 0x00000000, 0x73677261, 0x00000000, 0x00040006, 0x00000008, 0x00000001,
    0x00000078, 0x00040006, 0x00000008, 0x00000002, 0x00000079, 0x00040006, 0x00000008, 0x00000003,
    0x0000007a, 0x00040005, 0x00000009, 0x73677241, 0x00000000, 0x00040006, 0x00000009, 0x00000000,
    0x00000078, 0x00040006, 0x00000009, 0x00000001, 0x00000079, 0x00040006, 0x00000009, 0x00000002,
    0x0000007a, 0x00030005, 0x0000000b, 0x00006370, 0x00030047, 0x00000008, 0x00000002, 0x00050048,
    0x00000008, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000008, 0x00000001, 0x00000023,
    0x00000008, 0x00050048, 0x00000008, 0x00000002, 0x00000023, 0x0000000c, 0x00050048, 0x00000008,
    0x00000003, 0x00000023, 0x00000010, 0x00030047, 0x00000009, 0x00000002, 0x00050048, 0x00000009,
    0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000009, 0x00000001, 0x00000023, 0x00000004,
    0x00050048, 0x00000009, 0x00000002, 0x00000023, 0x00000008, 0x00020013, 0x00000002, 0x00030021,
    0x00000003, 0x00000002, 0x00030027, 0x00000006, 0x000014e5, 0x00040015, 0x00000007, 0x00000020,
    0x00000000, 0x0006001e, 0x00000008, 0x00000006, 0x00000007, 0x00000007, 0x00000007, 0x0005001e,
    0x00000009, 0x00000007, 0x00000007, 0x00000007, 0x00040020, 0x00000006, 0x000014e5, 0x00000009,
    0x00040020, 0x0000000a, 0x00000009, 0x00000008, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000009,
    0x00040015, 0x0000000c, 0x00000020, 0x00000001, 0x0004002b, 0x0000000c, 0x0000000d, 0x00000000,
    0x00040020, 0x0000000e, 0x00000009, 0x00000006, 0x0004002b, 0x0000000c, 0x00000011, 0x00000001,
    0x00040020, 0x00000012, 0x00000009, 0x00000007, 0x00040020, 0x00000015, 0x000014e5, 0x00000007,
    0x0004002b, 0x0000000c, 0x00000019, 0x00000002, 0x0004002b, 0x0000000c, 0x0000001f, 0x00000003,
    0x00040017, 0x00000023, 0x00000007, 0x00000003, 0x0004002b, 0x00000007, 0x00000024, 0x00000001,
    0x0006002c, 0x00000023, 0x00000025, 0x00000024, 0x00000024, 0x00000024, 0x00050036, 0x00000002,
    0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041, 0x0000000e, 0x0000000f,
    0x0000000b, 0x0000000d, 0x0004003d, 0x00000006, 0x00000010, 0x0000000f, 0x00050041, 0x00000012,
    0x00000013, 0x0000000b, 0x00000011, 0x0004003d, 0x00000007, 0x00000014, 0x00000013, 0x00050041,
    0x00000015, 0x00000016, 0x00000010, 0x0000000d, 0x0005003e, 0x00000016, 0x00000014, 0x00000002,
    0x00000004, 0x00050041, 0x0000000e, 0x00000017, 0x0000000b, 0x0000000d, 0x0004003d, 0x00000006,
    0x00000018, 0x00000017, 0x00050041, 0x00000012, 0x0000001a, 0x0000000b, 0x00000019, 0x0004003d,
    0x00000007, 0x0000001b, 0x0000001a, 0x00050041, 0x00000015, 0x0000001c, 0x00000018, 0x00000011,
    0x0005003e, 0x0000001c, 0x0000001b, 0x00000002, 0x00000004, 0x00050041, 0x0000000e, 0x0000001d,
    0x0000000b, 0x0000000d, 0x0004003d, 0x00000006, 0x0000001e, 0x0000001d, 0x00050041, 0x00000012,
    0x00000020, 0x0000000b, 0x0000001f, 0x0004003d, 0x00000007, 0x00000021, 0x00000020, 0x00050041,
    0x00000015, 0x00000022, 0x0000001e, 0x00000019, 0x0005003e, 0x00000022, 0x00000021, 0x00000002,
    0x00000004, 0x000100fd, 0x00010038,
};

/*
 * #version 450
 * #extension GL_EXT_buffer_reference : require
 * layout(local_size_x_id = 0) in;
 * layout(constant_id = 1) const uint SCALE = 2u;
 * layout(buffer_reference, std430, buffer_reference_align = 4) buffer Words { uint data[]; };
 * layout(push_constant) uniform Push { Words dst; } pc;
 * void main()
 * {
 *     pc.dst.data[gl_GlobalInvocationID.x] = gl_LocalInvocationIndex * SCALE + gl_WorkGroupID.x * 100000u;
 * }
 */
static const uint32_t spv_spec[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000029, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x000014e3, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x000014e4, 0x00000001, 0x0009000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000e,
    0x00000016, 0x0000001b, 0x0000001f, 0x0006014b, 0x00000004, 0x00000026, 0x00000007, 0x00000008,
    0x00000008, 0x00030003, 0x00000002, 0x000001c2, 0x00070004, 0x455f4c47, 0x625f5458, 0x65666675,
    0x65725f72, 0x65726566, 0x0065636e, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00040005,
    0x0000000a, 0x68737550, 0x00000000, 0x00040006, 0x0000000a, 0x00000000, 0x00747364, 0x00040005,
    0x0000000c, 0x64726f57, 0x00000073, 0x00050006, 0x0000000c, 0x00000000, 0x61746164, 0x00000000,
    0x00030005, 0x0000000e, 0x00006370, 0x00080005, 0x00000016, 0x475f6c67, 0x61626f6c, 0x766e496c,
    0x7461636f, 0x496e6f69, 0x00000044, 0x00080005, 0x0000001b, 0x4c5f6c67, 0x6c61636f, 0x6f766e49,
    0x69746163, 0x6e496e6f, 0x00786564, 0x00040005, 0x0000001d, 0x4c414353, 0x00000045, 0x00060005,
    0x0000001f, 0x575f6c67, 0x476b726f, 0x70756f72, 0x00004449, 0x00040047, 0x00000007, 0x00000001,
    0x00000000, 0x00030047, 0x0000000a, 0x00000002, 0x00050048, 0x0000000a, 0x00000000, 0x00000023,
    0x00000000, 0x00040047, 0x0000000b, 0x00000006, 0x00000004, 0x00030047, 0x0000000c, 0x00000002,
    0x00050048, 0x0000000c, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000016, 0x0000000b,
    0x0000001c, 0x00040047, 0x0000001b, 0x0000000b, 0x0000001d, 0x00040047, 0x0000001d, 0x00000001,
    0x00000001, 0x00040047, 0x0000001f, 0x0000000b, 0x0000001a, 0x00040047, 0x00000027, 0x00000001,
    0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006,
    0x00000020, 0x00000000, 0x00040032, 0x00000006, 0x00000007, 0x00000001, 0x0004002b, 0x00000006,
    0x00000008, 0x00000001, 0x00030027, 0x00000009, 0x000014e5, 0x0003001e, 0x0000000a, 0x00000009,
    0x0003001d, 0x0000000b, 0x00000006, 0x0003001e, 0x0000000c, 0x0000000b, 0x00040020, 0x00000009,
    0x000014e5, 0x0000000c, 0x00040020, 0x0000000d, 0x00000009, 0x0000000a, 0x0004003b, 0x0000000d,
    0x0000000e, 0x00000009, 0x00040015, 0x0000000f, 0x00000020, 0x00000001, 0x0004002b, 0x0000000f,
    0x00000010, 0x00000000, 0x00040020, 0x00000011, 0x00000009, 0x00000009, 0x00040017, 0x00000014,
    0x00000006, 0x00000003, 0x00040020, 0x00000015, 0x00000001, 0x00000014, 0x0004003b, 0x00000015,
    0x00000016, 0x00000001, 0x0004002b, 0x00000006, 0x00000017, 0x00000000, 0x00040020, 0x00000018,
    0x00000001, 0x00000006, 0x0004003b, 0x00000018, 0x0000001b, 0x00000001, 0x00040032, 0x00000006,
    0x0000001d, 0x00000002, 0x0004003b, 0x00000015, 0x0000001f, 0x00000001, 0x0004002b, 0x00000006,
    0x00000022, 0x000186a0, 0x00040020, 0x00000025, 0x000014e5, 0x00000006, 0x00040032, 0x00000006,
    0x00000027, 0x00000001, 0x00060033, 0x00000014, 0x00000028, 0x00000027, 0x00000008, 0x00000008,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041,
    0x00000011, 0x00000012, 0x0000000e, 0x00000010, 0x0004003d, 0x00000009, 0x00000013, 0x00000012,
    0x00050041, 0x00000018, 0x00000019, 0x00000016, 0x00000017, 0x0004003d, 0x00000006, 0x0000001a,
    0x00000019, 0x0004003d, 0x00000006, 0x0000001c, 0x0000001b, 0x00050084, 0x00000006, 0x0000001e,
    0x0000001c, 0x0000001d, 0x00050041, 0x00000018, 0x00000020, 0x0000001f, 0x00000017, 0x0004003d,
    0x00000006, 0x00000021, 0x00000020, 0x00050084, 0x00000006, 0x00000023, 0x00000021, 0x00000022,
    0x00050080, 0x00000006, 0x00000024, 0x0000001e, 0x00000023, 0x00060041, 0x00000025, 0x00000026,
    0x00000013, 0x00000010, 0x0000001a, 0x0005003e, 0x00000026, 0x00000024, 0x00000002, 0x00000004,
    0x000100fd, 0x00010038,
};

/*
 * #version 450
 * layout(local_size_x = 1) in;
 * layout(std430, set = 0, binding = 0) buffer B { uint v[]; } b;
 * void main()
 * {
 *     b.v[0] = 1u;
 * }
 */
static const uint32_t spv_set[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000012, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000a, 0x00060010, 0x00000004,
    0x00000011, 0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008, 0x00000042, 0x00040006, 0x00000008,
    0x00000000, 0x00000076, 0x00030005, 0x0000000a, 0x00000062, 0x00040047, 0x00000007, 0x00000006,
    0x00000004, 0x00030047, 0x00000008, 0x00000002, 0x00050048, 0x00000008, 0x00000000, 0x00000023,
    0x00000000, 0x00040047, 0x0000000a, 0x00000021, 0x00000000, 0x00040047, 0x0000000a, 0x00000022,
    0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006,
    0x00000020, 0x00000000, 0x0003001d, 0x00000007, 0x00000006, 0x0003001e, 0x00000008, 0x00000007,
    0x00040020, 0x00000009, 0x0000000c, 0x00000008, 0x0004003b, 0x00000009, 0x0000000a, 0x0000000c,
    0x00040015, 0x0000000b, 0x00000020, 0x00000001, 0x0004002b, 0x0000000b, 0x0000000c, 0x00000000,
    0x0004002b, 0x00000006, 0x0000000d, 0x00000001, 0x00040020, 0x0000000e, 0x0000000c, 0x00000006,
    0x00040017, 0x00000010, 0x00000006, 0x00000003, 0x0006002c, 0x00000010, 0x00000011, 0x0000000d,
    0x0000000d, 0x0000000d, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
    0x00000005, 0x00060041, 0x0000000e, 0x0000000f, 0x0000000a, 0x0000000c, 0x0000000c, 0x0003003e,
    0x0000000f, 0x0000000d, 0x000100fd, 0x00010038,
};

/*
 * #version 450
 * #extension GL_EXT_buffer_reference : require
 * layout(local_size_x = 1024) in;
 * layout(buffer_reference, std430, buffer_reference_align = 4) buffer Words { uint data[]; };
 * layout(push_constant) uniform Push { Words dst; } pc;
 * shared uint s[1024];
 * void main()
 * {
 *     uint l = gl_LocalInvocationIndex;
 *     s[l] = l * 3u;
 *     barrier();
 *     pc.dst.data[gl_GlobalInvocationID.x] = s[1023u - l] + gl_WorkGroupID.x;
 * }
 */
static const uint32_t spv_wide[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000036, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x000014e3, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x000014e4, 0x00000001, 0x000a000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000a,
    0x0000000f, 0x0000001d, 0x00000025, 0x0000002e, 0x00060010, 0x00000004, 0x00000011, 0x00000400,
    0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00070004, 0x455f4c47, 0x625f5458,
    0x65666675, 0x65725f72, 0x65726566, 0x0065636e, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00030005, 0x00000008, 0x0000006c, 0x00080005, 0x0000000a, 0x4c5f6c67, 0x6c61636f, 0x6f766e49,
    0x69746163, 0x6e496e6f, 0x00786564, 0x00030005, 0x0000000f, 0x00000073, 0x00040005, 0x00000019,
    0x68737550, 0x00000000, 0x00040006, 0x00000019, 0x00000000, 0x00747364, 0x00040005, 0x0000001b,
    0x64726f57, 0x00000073, 0x00050006, 0x0000001b, 0x00000000, 0x61746164, 0x00000000, 0x00030005,
    0x0000001d, 0x00006370, 0x00080005, 0x00000025, 0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f,
    0x496e6f69, 0x00000044, 0x00060005, 0x0000002e, 0x575f6c67, 0x476b726f, 0x70756f72, 0x00004449,
    0x00040047, 0x0000000a, 0x0000000b, 0x0000001d, 0x00030047, 0x00000019, 0x00000002, 0x00050048,
    0x00000019, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x0000001a, 0x00000006, 0x00000004,
    0x00030047, 0x0000001b, 0x00000002, 0x00050048, 0x0000001b, 0x00000000, 0x00000023, 0x00000000,
    0x00040047, 0x00000025, 0x0000000b, 0x0000001c, 0x00040047, 0x0000002e, 0x0000000b, 0x0000001a,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020,
    0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00040020, 0x00000009, 0x00000001,
    0x00000006, 0x0004003b, 0x00000009, 0x0000000a, 0x00000001, 0x0004002b, 0x00000006, 0x0000000c,
    0x00000400, 0x0004001c, 0x0000000d, 0x00000006, 0x0000000c, 0x00040020, 0x0000000e, 0x00000004,
    0x0000000d, 0x0004003b, 0x0000000e, 0x0000000f, 0x00000004, 0x0004002b, 0x00000006, 0x00000012,
    0x00000003, 0x00040020, 0x00000014, 0x00000004, 0x00000006, 0x0004002b, 0x00000006, 0x00000016,
    0x00000002, 0x0004002b, 0x00000006, 0x00000017, 0x00000108, 0x00030027, 0x00000018, 0x000014e5,
    0x0003001e, 0x00000019, 0x00000018, 0x0003001d, 0x0000001a, 0x00000006, 0x0003001e, 0x0000001b,
    0x0000001a, 0x00040020, 0x00000018, 0x000014e5, 0x0000001b, 0x00040020, 0x0000001c, 0x00000009,
    0x00000019, 0x0004003b, 0x0000001c, 0x0000001d, 0x00000009, 0x00040015, 0x0000001e, 0x00000020,
    0x00000001, 0x0004002b, 0x0000001e, 0x0000001f, 0x00000000, 0x00040020, 0x00000020, 0x00000009,
    0x00000018, 0x00040017, 0x00000023, 0x00000006, 0x00000003, 0x00040020, 0x00000024, 0x00000001,
    0x00000023, 0x0004003b, 0x00000024, 0x00000025, 0x00000001, 0x0004002b, 0x00000006, 0x00000026,
    0x00000000, 0x0004002b, 0x00000006, 0x00000029, 0x000003ff, 0x0004003b, 0x00000024, 0x0000002e,
    0x00000001, 0x00040020, 0x00000032, 0x000014e5, 0x00000006, 0x0004002b, 0x00000006, 0x00000034,
    0x00000001, 0x0006002c, 0x00000023, 0x00000035, 0x0000000c, 0x00000034, 0x00000034, 0x00050036,
    0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007,
    0x00000008, 0x00000007, 0x0004003d, 0x00000006, 0x0000000b, 0x0000000a, 0x0003003e, 0x00000008,
    0x0000000b, 0x0004003d, 0x00000006, 0x00000010, 0x00000008, 0x0004003d, 0x00000006, 0x00000011,
    0x00000008, 0x00050084, 0x00000006, 0x00000013, 0x00000011, 0x00000012, 0x00050041, 0x00000014,
    0x00000015, 0x0000000f, 0x00000010, 0x0003003e, 0x00000015, 0x00000013, 0x000400e0, 0x00000016,
    0x00000016, 0x00000017, 0x00050041, 0x00000020, 0x00000021, 0x0000001d, 0x0000001f, 0x0004003d,
    0x00000018, 0x00000022, 0x00000021, 0x00050041, 0x00000009, 0x00000027, 0x00000025, 0x00000026,
    0x0004003d, 0x00000006, 0x00000028, 0x00000027, 0x0004003d, 0x00000006, 0x0000002a, 0x00000008,
    0x00050082, 0x00000006, 0x0000002b, 0x00000029, 0x0000002a, 0x00050041, 0x00000014, 0x0000002c,
    0x0000000f, 0x0000002b, 0x0004003d, 0x00000006, 0x0000002d, 0x0000002c, 0x00050041, 0x00000009,
    0x0000002f, 0x0000002e, 0x00000026, 0x0004003d, 0x00000006, 0x00000030, 0x0000002f, 0x00050080,
    0x00000006, 0x00000031, 0x0000002d, 0x00000030, 0x00060041, 0x00000032, 0x00000033, 0x00000022,
    0x0000001f, 0x00000028, 0x0005003e, 0x00000033, 0x00000031, 0x00000002, 0x00000004, 0x000100fd,
    0x00010038,
};

/* ---- Setup ---- */

static VkInstance instance;
static VkDevice device;
static VkQueue queue;
static VkCommandPool pool;
static VkFence fence;
static VkPipelineLayout layout; /* 256 bytes of push constants */
static uint32_t shared_type = UINT32_MAX;
static uint32_t private_type = UINT32_MAX;

struct buf {
    VkBuffer buffer;
    VkDeviceMemory memory;
    unsigned char *data; /* NULL for private memory */
    VkDeviceAddress address;
    VkDeviceSize size;
};

static void setup(void)
{
    VkPhysicalDeviceMemoryProperties mem;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceFeatures2 features;
    VkApplicationInfo app;
    VkInstanceCreateInfo instance_info;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo device_info;
    VkCommandPoolCreateInfo pool_info;
    VkFenceCreateInfo fence_info;
    VkPushConstantRange range;
    VkPipelineLayoutCreateInfo layout_info;
    VkPhysicalDevice pd;
    float priority = 1.0f;
    uint32_t count = 1, i;

    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_4;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    REQUIRE_VK(vkCreateInstance(&instance_info, NULL, &instance));
    REQUIRE_VK(vkEnumeratePhysicalDevices(instance, &count, &pd));
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&features, 0, sizeof(features));
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &v12;
    vkGetPhysicalDeviceFeatures2(pd, &features);
    REQUIRE(v12.bufferDeviceAddress);
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.bufferDeviceAddress = VK_TRUE;
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &v12;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    REQUIRE_VK(vkCreateDevice(pd, &device_info, NULL, &device));
    vkGetDeviceQueue(device, 0, 0, &queue);
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    for (i = 0; i < mem.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mem.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) && shared_type == UINT32_MAX)
            shared_type = i;
        if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && private_type == UINT32_MAX)
            private_type = i;
    }
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    REQUIRE_VK(vkCreateCommandPool(device, &pool_info, NULL, &pool));
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    REQUIRE_VK(vkCreateFence(device, &fence_info, NULL, &fence));
    memset(&range, 0, sizeof(range));
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = 256;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    REQUIRE_VK(vkCreatePipelineLayout(device, &layout_info, NULL, &layout));
}

static void teardown(void)
{
    vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroyFence(device, fence, NULL);
    vkDestroyCommandPool(device, pool, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
}

static struct buf make_buf(VkDeviceSize size, uint32_t type)
{
    VkMemoryAllocateFlagsInfo flags;
    VkBufferDeviceAddressInfo address;
    VkBufferCreateInfo info;
    VkMemoryAllocateInfo alloc;
    VkMemoryRequirements req;
    struct buf b;
    void *p = NULL;

    memset(&b, 0, sizeof(b));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (vkCreateBuffer(device, &info, NULL, &b.buffer) != VK_SUCCESS)
        return b;
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    memset(&flags, 0, sizeof(flags));
    flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = &flags;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    if (vkAllocateMemory(device, &alloc, NULL, &b.memory) != VK_SUCCESS)
        return b;
    vkBindBufferMemory(device, b.buffer, b.memory, 0);
    if (type == shared_type && vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS) {
        b.data = p;
        memset(b.data, 0, (size_t)size);
    }
    memset(&address, 0, sizeof(address));
    address.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    address.buffer = b.buffer;
    b.address = vkGetBufferDeviceAddress(device, &address);
    b.size = size;
    return b;
}

static void free_buf(struct buf *b)
{
    vkDestroyBuffer(device, b->buffer, NULL);
    vkFreeMemory(device, b->memory, NULL);
}

static uint32_t word(const struct buf *b, size_t i)
{
    uint32_t w;
    memcpy(&w, b->data + 4 * i, 4);
    return w;
}

/* A compute pipeline with entry point "main"; VK_NULL_HANDLE on failure. */
static VkPipeline make_pipeline_ex(const uint32_t *code, size_t size, VkPipelineCreateFlags flags,
                                   const VkSpecializationInfo *spec, VkResult *result)
{
    VkShaderModuleCreateInfo module_info;
    VkComputePipelineCreateInfo info;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    memset(&module_info, 0, sizeof(module_info));
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = size;
    module_info.pCode = code;
    *result = vkCreateShaderModule(device, &module_info, NULL, &module);
    if (*result != VK_SUCCESS)
        return VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.flags = flags;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.stage.pSpecializationInfo = spec;
    info.layout = layout;
    *result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, NULL, &pipeline);
    vkDestroyShaderModule(device, module, NULL);
    return pipeline;
}

static VkPipeline make_pipeline(const uint32_t *code, size_t size, VkPipelineCreateFlags flags)
{
    VkResult r;
    VkPipeline p = make_pipeline_ex(code, size, flags, NULL, &r);
    if (r != VK_SUCCESS)
        fprintf(stderr, "  vkCreateComputePipelines: %d\n", (int)r);
    return p;
}

static VkCommandBuffer begin(void)
{
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferBeginInfo info;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &alloc, &cb);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &info);
    return cb;
}

static VkResult submit_wait(VkCommandBuffer cb)
{
    VkSubmitInfo submit;
    VkResult r;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    vkResetFences(device, 1, &fence);
    r = vkQueueSubmit(queue, 1, &submit, fence);
    return r != VK_SUCCESS ? r : vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ull);
}

static VkResult finish(VkCommandBuffer cb)
{
    VkResult r = vkEndCommandBuffer(cb);
    if (r == VK_SUCCESS)
        r = submit_wait(cb);
    vkFreeCommandBuffers(device, pool, 1, &cb);
    return r;
}

static void barrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst, VkAccessFlags dst_access)
{
    VkMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cb, src, dst, 0, 1, &b, 0, NULL, 0, NULL);
}

static void host_barrier(VkCommandBuffer cb)
{
    barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_HOST_READ_BIT);
}

/* spv_scale's push constants */
struct scale_push {
    VkDeviceAddress src, dst;
    uint32_t count, mul, add;
};

static void push_scale(VkCommandBuffer cb, VkDeviceAddress src, VkDeviceAddress dst, uint32_t count, uint32_t mul,
                       uint32_t add)
{
    struct scale_push p;
    memset(&p, 0, sizeof(p));
    p.src = src;
    p.dst = dst;
    p.count = count;
    p.mul = mul;
    p.add = add;
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
}

/* ---- Tests ---- */

/* Push constants change between dispatches of one command buffer (each
 * dispatch sees its own), through vkCmdPushConstants2 too, and partial
 * updates keep the other bytes. */
static void test_push_constants(void)
{
    enum { N = 1000 };
    struct buf src = make_buf(4 * N, shared_type), dst = make_buf(4 * 4 * N, shared_type);
    VkPipeline p = make_pipeline(spv_scale, sizeof(spv_scale), 0);
    VkPushConstantsInfo info2;
    struct scale_push sp;
    VkCommandBuffer cb;
    uint32_t i, add = 7;
    int ok = 1;

    REQUIRE(p && src.data && dst.data);
    for (i = 0; i < N; i++)
        memcpy(src.data + 4 * i, &i, 4);
    cb = begin();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    push_scale(cb, src.address, dst.address, N, 2, 1);
    vkCmdDispatch(cb, (N + 63) / 64, 1, 1);
    push_scale(cb, src.address, dst.address + 4 * N, N, 3, 5);
    vkCmdDispatch(cb, (N + 63) / 64, 1, 1);
    memset(&sp, 0, sizeof(sp));
    sp.src = src.address;
    sp.dst = dst.address + 8 * N;
    sp.count = N;
    sp.mul = 10;
    sp.add = 0;
    memset(&info2, 0, sizeof(info2));
    info2.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO;
    info2.layout = layout;
    info2.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    info2.size = sizeof(sp);
    info2.pValues = &sp;
    vkCmdPushConstants2(cb, &info2);
    vkCmdDispatch(cb, (N + 63) / 64, 1, 1);
    /* only dst and add change */
    sp.dst = dst.address + 12 * N;
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, offsetof(struct scale_push, dst), 8, &sp.dst);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, offsetof(struct scale_push, add), 4, &add);
    vkCmdDispatch(cb, (N + 63) / 64, 1, 1);
    host_barrier(cb);
    REQUIRE_VK(finish(cb));
    for (i = 0; i < N && ok; i++) {
        ok = word(&dst, i) == 2 * i + 1 && word(&dst, N + i) == 3 * i + 5 && word(&dst, 2 * N + i) == 10 * i &&
             word(&dst, 3 * N + i) == 10 * i + 7;
        if (!ok)
            fprintf(stderr, "  element %u: %u %u %u %u\n", i, word(&dst, i), word(&dst, N + i), word(&dst, 2 * N + i),
                    word(&dst, 3 * N + i));
    }
    CHECK(ok);
    vkDestroyPipeline(device, p, NULL);
    free_buf(&src);
    free_buf(&dst);
}

/* Dependent dispatches ping-ponging between two buffers in private memory,
 * after a fill by the internal kernel and before a copy: each step needs
 * the barrier before it. */
static void test_dispatch_chain(void)
{
    enum { N = 1u << 20, STEPS = 32 };
    struct buf a = make_buf(4 * N, private_type), b = make_buf(4 * N, private_type), out = make_buf(4 * N, shared_type);
    VkPipeline p = make_pipeline(spv_scale, sizeof(spv_scale), 0);
    VkBufferCopy region;
    VkCommandBuffer cb;
    uint32_t k, i;
    int ok = 1;

    REQUIRE(p && a.buffer && b.buffer && out.data);
    cb = begin();
    vkCmdFillBuffer(cb, a.buffer, 0, VK_WHOLE_SIZE, 0x00010203u);
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    for (k = 0; k < STEPS; k++) {
        if (k)
            barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        push_scale(cb, (k & 1) ? b.address : a.address, (k & 1) ? a.address : b.address, N, 1, k + 1);
        vkCmdDispatch(cb, N / 64, 1, 1);
    }
    barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    memset(&region, 0, sizeof(region));
    region.size = 4 * N;
    vkCmdCopyBuffer(cb, (STEPS & 1) ? b.buffer : a.buffer, out.buffer, 1, &region);
    host_barrier(cb);
    REQUIRE_VK(finish(cb));
    for (i = 0; i < N && ok; i++) {
        ok = word(&out, i) == 0x00010203u + STEPS * (STEPS + 1) / 2;
        if (!ok)
            fprintf(stderr, "  element %u: %08x\n", i, word(&out, i));
    }
    CHECK(ok);
    vkDestroyPipeline(device, p, NULL);
    free_buf(&a);
    free_buf(&b);
    free_buf(&out);
}

/* spv_ids over a width x height x depth grid of uvec4 cells: which cells
 * were written, and with which workgroup and local index. */
static int check_cells(const struct buf *cells, uint32_t width, uint32_t height, uint32_t depth, const uint32_t base[3],
                       const uint32_t count[3])
{
    uint32_t x, y, z;
    for (z = 0; z < depth; z++)
        for (y = 0; y < height; y++)
            for (x = 0; x < width; x++) {
                const unsigned char *c = cells->data + 16 * ((z * height + y) * width + x);
                uint32_t got[4], want[4] = { 0, 0, 0, 0 };
                memcpy(got, c, 16);
                if (x / 4 >= base[0] && x / 4 < base[0] + count[0] && y / 2 >= base[1] &&
                    y / 2 < base[1] + count[1] && z / 2 >= base[2] && z / 2 < base[2] + count[2]) {
                    want[0] = x / 4;
                    want[1] = y / 2;
                    want[2] = z / 2;
                    want[3] = ((z % 2) * 2 + y % 2) * 4 + x % 4 + 1;
                }
                if (memcmp(got, want, 16) != 0) {
                    fprintf(stderr, "  cell (%u, %u, %u) = %u %u %u %u, want %u %u %u %u\n", x, y, z, got[0], got[1],
                            got[2], got[3], want[0], want[1], want[2], want[3]);
                    return 0;
                }
            }
    return 1;
}

struct ids_push {
    VkDeviceAddress cells;
    uint32_t width, height;
};

static void test_dispatch_base(void)
{
    static const uint32_t base[3] = { 1, 2, 1 }, count[3] = { 2, 1, 1 }, zero[3] = { 0, 0, 0 };
    enum { W = 12, H = 6, D = 4 };
    struct buf cells = make_buf(16 * W * H * D, shared_type);
    VkPipeline p = make_pipeline(spv_ids, sizeof(spv_ids), VK_PIPELINE_CREATE_DISPATCH_BASE_BIT);
    struct ids_push push;
    VkCommandBuffer cb;

    REQUIRE(p && cells.data);
    memset(&push, 0, sizeof(push));
    push.cells = cells.address;
    push.width = W;
    push.height = H;
    cb = begin();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatchBase(cb, base[0], base[1], base[2], count[0], count[1], count[2]);
    host_barrier(cb);
    REQUIRE_VK(finish(cb));
    CHECK(check_cells(&cells, W, H, D, base, count));

    /* The same pipeline without a base, then one made without the flag. */
    memset(cells.data, 0, 16 * W * H * D);
    cb = begin();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cb, 3, 3, 2);
    host_barrier(cb);
    REQUIRE_VK(finish(cb));
    {
        static const uint32_t all[3] = { 3, 3, 2 };
        CHECK(check_cells(&cells, W, H, D, zero, all));
    }
    vkDestroyPipeline(device, p, NULL);
    free_buf(&cells);
}

/* Indirect dispatches: arguments from vkCmdUpdateBuffer, from a previous
 * dispatch, and all zero (nothing runs). */
static void test_dispatch_indirect(void)
{
    static const uint32_t zero[3] = { 0, 0, 0 };
    enum { W = 12, H = 6, D = 4 };
    struct buf cells = make_buf(16 * W * H * D, shared_type), args = make_buf(256, private_type);
    VkPipeline ids = make_pipeline(spv_ids, sizeof(spv_ids), 0), gen = make_pipeline(spv_args, sizeof(spv_args), 0);
    uint32_t host_args[3] = { 3, 1, 2 }, none[3] = { 0, 0, 0 };
    struct {
        VkDeviceAddress args;
        uint32_t x, y, z;
    } gen_push;
    struct ids_push push;
    VkCommandBuffer cb;

    REQUIRE(ids && gen && cells.data && args.buffer);
    memset(&push, 0, sizeof(push));
    push.cells = cells.address;
    push.width = W;
    push.height = H;
    cb = begin();
    vkCmdUpdateBuffer(cb, args.buffer, 16, sizeof(host_args), host_args);
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ids);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatchIndirect(cb, args.buffer, 16);
    host_barrier(cb);
    REQUIRE_VK(finish(cb));
    CHECK(check_cells(&cells, W, H, D, zero, host_args));

    memset(cells.data, 0, 16 * W * H * D);
    memset(&gen_push, 0, sizeof(gen_push));
    gen_push.args = args.address + 64;
    gen_push.x = 2;
    gen_push.y = 3;
    gen_push.z = 1;
    cb = begin();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, gen);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gen_push), &gen_push);
    vkCmdDispatch(cb, 1, 1, 1);
    vkCmdUpdateBuffer(cb, args.buffer, 128, sizeof(none), none);
    barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ids);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatchIndirect(cb, args.buffer, 64);
    vkCmdDispatchIndirect(cb, args.buffer, 128); /* 0 x 0 x 0 */
    host_barrier(cb);
    REQUIRE_VK(finish(cb));
    {
        static const uint32_t gen_count[3] = { 2, 3, 1 };
        CHECK(check_cells(&cells, W, H, D, zero, gen_count));
    }
    vkDestroyPipeline(device, ids, NULL);
    vkDestroyPipeline(device, gen, NULL);
    free_buf(&cells);
    free_buf(&args);
}

/* Specialization constants (the workgroup size among them), with the
 * shader given as VkShaderModuleCreateInfo in the stage (no module). */
static void test_specialization(void)
{
    static const VkSpecializationMapEntry entries[2] = { { 0, 0, 4 }, { 1, 4, 4 } };
    static const uint32_t data[2] = { 128, 3 };
    VkSpecializationInfo spec;
    VkShaderModuleCreateInfo module_info;
    VkComputePipelineCreateInfo info;
    VkPipeline p = VK_NULL_HANDLE;
    struct buf dst = make_buf(4 * 512, shared_type);
    VkDeviceAddress address;
    VkCommandBuffer cb;
    uint32_t i;
    int ok = 1;

    REQUIRE(dst.data);
    memset(&spec, 0, sizeof(spec));
    spec.mapEntryCount = 2;
    spec.pMapEntries = entries;
    spec.dataSize = sizeof(data);
    spec.pData = data;
    memset(&module_info, 0, sizeof(module_info));
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = sizeof(spv_spec);
    module_info.pCode = spv_spec;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.pNext = &module_info;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.pName = "main";
    info.stage.pSpecializationInfo = &spec;
    info.layout = layout;
    REQUIRE_VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, NULL, &p));
    address = dst.address;
    cb = begin();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(address), &address);
    vkCmdDispatch(cb, 4, 1, 1);
    host_barrier(cb);
    REQUIRE_VK(finish(cb));
    for (i = 0; i < 512 && ok; i++) {
        ok = word(&dst, i) == (i % 128) * 3 + (i / 128) * 100000;
        if (!ok)
            fprintf(stderr, "  element %u: %u\n", i, word(&dst, i));
    }
    CHECK(ok);
    vkDestroyPipeline(device, p, NULL);
    free_buf(&dst);
}

/* The largest workgroup (maxComputeWorkGroupInvocations = 1024), sharing
 * workgroup memory across a barrier. */
static void test_wide_workgroup(void)
{
    struct buf dst = make_buf(4 * 1024 * 3, shared_type);
    VkPipeline p = make_pipeline(spv_wide, sizeof(spv_wide), 0);
    VkDeviceAddress address = dst.address;
    VkCommandBuffer cb;
    uint32_t g, l;
    int ok = 1;

    REQUIRE(p && dst.data);
    cb = begin();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(address), &address);
    vkCmdDispatch(cb, 3, 1, 1);
    host_barrier(cb);
    REQUIRE_VK(finish(cb));
    for (g = 0; g < 3 && ok; g++)
        for (l = 0; l < 1024 && ok; l++) {
            ok = word(&dst, g * 1024 + l) == (1023 - l) * 3 + g;
            if (!ok)
                fprintf(stderr, "  group %u thread %u: %u\n", g, l, word(&dst, g * 1024 + l));
        }
    CHECK(ok);
    vkDestroyPipeline(device, p, NULL);
    free_buf(&dst);
}

/* One recorded dispatch, submitted again with new input each time. */
static void test_resubmit(void)
{
    enum { N = 4096 };
    struct buf src = make_buf(4 * N, shared_type), dst = make_buf(4 * N, shared_type);
    VkPipeline p = make_pipeline(spv_scale, sizeof(spv_scale), 0);
    VkCommandBuffer cb;
    uint32_t round, i;
    int ok = 1;

    REQUIRE(p && src.data && dst.data);
    cb = begin();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    push_scale(cb, src.address, dst.address, N, 5, 0);
    vkCmdDispatch(cb, N / 64, 1, 1);
    host_barrier(cb);
    REQUIRE_VK(vkEndCommandBuffer(cb));
    for (round = 0; round < 4 && ok; round++) {
        for (i = 0; i < N; i++) {
            uint32_t v = i + round * 1000;
            memcpy(src.data + 4 * i, &v, 4);
        }
        REQUIRE_VK(submit_wait(cb));
        for (i = 0; i < N && ok; i++)
            ok = word(&dst, i) == 5 * (i + round * 1000);
    }
    CHECK(ok);
    vkFreeCommandBuffers(device, pool, 1, &cb);
    vkDestroyPipeline(device, p, NULL);
    free_buf(&src);
    free_buf(&dst);
}

/* Creation failures return VK_NULL_HANDLE for that pipeline only (or the
 * rest too with EARLY_RETURN_ON_FAILURE), and FAIL_ON_PIPELINE_COMPILE_
 * REQUIRED is honoured without a cache. */
static void test_pipeline_errors(void)
{
    static const uint32_t garbage[] = { 0x07230203, 0x00010600, 0, 10, 0, 0xdeadbeef };
    VkShaderModuleCreateInfo module_info;
    VkComputePipelineCreateInfo infos[3];
    VkShaderModule good, bad;
    VkPipeline p[3];
    VkResult r;

    CHECK(make_pipeline_ex(spv_set, sizeof(spv_set), 0, NULL, &r) == VK_NULL_HANDLE);
    CHECK(r == VK_ERROR_UNKNOWN); /* descriptor sets are not implemented yet */
    memset(&module_info, 0, sizeof(module_info));
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = sizeof(spv_scale);
    module_info.pCode = spv_scale;
    REQUIRE_VK(vkCreateShaderModule(device, &module_info, NULL, &good));
    module_info.codeSize = sizeof(garbage);
    module_info.pCode = garbage;
    REQUIRE_VK(vkCreateShaderModule(device, &module_info, NULL, &bad));
    memset(infos, 0, sizeof(infos));
    infos[0].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    infos[0].stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    infos[0].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    infos[0].stage.module = good;
    infos[0].stage.pName = "main";
    infos[0].layout = layout;
    infos[1] = infos[2] = infos[0];
    infos[1].stage.module = bad;

    p[0] = p[1] = p[2] = (VkPipeline)(uintptr_t)1;
    CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 3, infos, NULL, p) == VK_ERROR_UNKNOWN);
    CHECK(p[0] && !p[1] && p[2]);
    vkDestroyPipeline(device, p[0], NULL);
    vkDestroyPipeline(device, p[2], NULL);

    infos[1].flags = VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT;
    p[0] = p[1] = p[2] = (VkPipeline)(uintptr_t)1;
    CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 3, infos, NULL, p) == VK_ERROR_UNKNOWN);
    CHECK(p[0] && !p[1] && !p[2]);
    vkDestroyPipeline(device, p[0], NULL);

    infos[0].flags = VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
    p[0] = (VkPipeline)(uintptr_t)1;
    CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, infos, NULL, p) == VK_PIPELINE_COMPILE_REQUIRED);
    CHECK(!p[0]);

    vkDestroyShaderModule(device, good, NULL);
    vkDestroyShaderModule(device, bad, NULL);
    vkDestroyPipeline(device, VK_NULL_HANDLE, NULL);
    vkDestroyShaderModule(device, VK_NULL_HANDLE, NULL);
    vkDestroyPipelineLayout(device, VK_NULL_HANDLE, NULL);
}

int main(void)
{
    setup();
    if (pvk_test_failures)
        return 1;
    RUN(test_push_constants);
    RUN(test_dispatch_chain);
    RUN(test_dispatch_base);
    RUN(test_dispatch_indirect);
    RUN(test_specialization);
    RUN(test_wide_workgroup);
    RUN(test_resubmit);
    RUN(test_pipeline_errors);
    teardown();
    return TEST_RESULT();
}
