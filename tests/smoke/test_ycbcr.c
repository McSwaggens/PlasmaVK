/* Sampler Y'CbCr conversion (sections 7, 11, 12, 13 and 17): Y'CbCr formats
 * and their properties, multi-planar images (plane copies, disjoint
 * memory, linear layouts, host copies, plane views), and sampling with
 * conversions, checked against a CPU reference of Vulkan's Textures >
 * Sampler Y'CbCr Conversion: in compute shaders through the Vulkan API, and
 * in fragment shaders through the translator and Metal render pipelines
 * (graphics pipelines are not in PlasmaVK yet).
 *
 * Tolerances: nearest sampling reads texels exactly, so results match the
 * reference up to float rounding (TOL_NEAREST). Linear filtering by the GPU
 * (the luma, and the chroma of implicit reconstruction) uses 8-bit weights
 * (subTexelPrecisionBits = 8): each weight may be off by 1/512, so a value
 * filtered across two axes by about 1/256 of the texels' spread (at most 1),
 * which range expansion and model conversion amplify by at most 2.14
 * (narrow-range chroma, 255 / 224, times BT.2020's 1.8814): 0.0084.
 * TOL_LINEAR leaves margin for that. Coordinates where nearest sampling is
 * within 2% of a texel of a boundary (luma or chroma) are not compared:
 * either neighbour is right. */
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../pvk_test.h"
#include "../pvk_msl_test.h"
#include "../pvk_msl_render.h"

#define TOL_NEAREST 0.002f
#define TOL_LINEAR 0.015f

/*
 * #version 450
 * layout(local_size_x = 64) in;
 * layout(set = 0, binding = 0) uniform sampler2D ycbcr;
 * layout(std430, set = 0, binding = 1) readonly buffer Coords { vec4 coords[]; };
 * layout(std430, set = 0, binding = 2) writeonly buffer Results { vec4 results[]; };
 * layout(push_constant) uniform Push { uint count; uint mode; };
 * void main()
 * {
 *     uint i = gl_GlobalInvocationID.x;
 *     if (i >= count)
 *         return;
 *     vec4 c = coords[i];
 *     if (mode == 0u)
 *         results[i] = textureLod(ycbcr, c.xy, c.z);
 *     else
 *         results[i] = textureGrad(ycbcr, c.xy, vec2(c.z, 0.0), vec2(0.0, c.w));
 * }
 */
static const uint32_t spv_sample_comp[] = {
    0x07230203,0x00010000,0x0008000b,0x00000055,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0006000f,0x00000005,0x00000004,0x6e69616d,0x00000000,0x0000000b,0x00060010,0x00000004,
    0x00000011,0x00000040,0x00000001,0x00000001,0x00030003,0x00000002,0x000001c2,0x00040005,
    0x00000004,0x6e69616d,0x00000000,0x00030005,0x00000008,0x00000069,0x00080005,0x0000000b,
    0x475f6c67,0x61626f6c,0x766e496c,0x7461636f,0x496e6f69,0x00000044,0x00040005,0x00000011,
    0x68737550,0x00000000,0x00050006,0x00000011,0x00000000,0x6e756f63,0x00000074,0x00050006,
    0x00000011,0x00000001,0x65646f6d,0x00000000,0x00030005,0x00000013,0x00000000,0x00030005,
    0x00000021,0x00000063,0x00040005,0x00000023,0x726f6f43,0x00007364,0x00050006,0x00000023,
    0x00000000,0x726f6f63,0x00007364,0x00030005,0x00000025,0x00000000,0x00040005,0x00000031,
    0x75736552,0x0073746c,0x00050006,0x00000031,0x00000000,0x75736572,0x0073746c,0x00030005,
    0x00000033,0x00000000,0x00040005,0x00000038,0x63626379,0x00000072,0x00040047,0x0000000b,
    0x0000000b,0x0000001c,0x00030047,0x00000011,0x00000002,0x00050048,0x00000011,0x00000000,
    0x00000023,0x00000000,0x00050048,0x00000011,0x00000001,0x00000023,0x00000004,0x00040047,
    0x00000022,0x00000006,0x00000010,0x00030047,0x00000023,0x00000003,0x00040048,0x00000023,
    0x00000000,0x00000018,0x00050048,0x00000023,0x00000000,0x00000023,0x00000000,0x00030047,
    0x00000025,0x00000018,0x00040047,0x00000025,0x00000021,0x00000001,0x00040047,0x00000025,
    0x00000022,0x00000000,0x00040047,0x00000030,0x00000006,0x00000010,0x00030047,0x00000031,
    0x00000003,0x00040048,0x00000031,0x00000000,0x00000019,0x00050048,0x00000031,0x00000000,
    0x00000023,0x00000000,0x00030047,0x00000033,0x00000019,0x00040047,0x00000033,0x00000021,
    0x00000002,0x00040047,0x00000033,0x00000022,0x00000000,0x00040047,0x00000038,0x00000021,
    0x00000000,0x00040047,0x00000038,0x00000022,0x00000000,0x00040047,0x00000054,0x0000000b,
    0x00000019,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,0x00000006,
    0x00000020,0x00000000,0x00040020,0x00000007,0x00000007,0x00000006,0x00040017,0x00000009,
    0x00000006,0x00000003,0x00040020,0x0000000a,0x00000001,0x00000009,0x0004003b,0x0000000a,
    0x0000000b,0x00000001,0x0004002b,0x00000006,0x0000000c,0x00000000,0x00040020,0x0000000d,
    0x00000001,0x00000006,0x0004001e,0x00000011,0x00000006,0x00000006,0x00040020,0x00000012,
    0x00000009,0x00000011,0x0004003b,0x00000012,0x00000013,0x00000009,0x00040015,0x00000014,
    0x00000020,0x00000001,0x0004002b,0x00000014,0x00000015,0x00000000,0x00040020,0x00000016,
    0x00000009,0x00000006,0x00020014,0x00000019,0x00030016,0x0000001e,0x00000020,0x00040017,
    0x0000001f,0x0000001e,0x00000004,0x00040020,0x00000020,0x00000007,0x0000001f,0x0003001d,
    0x00000022,0x0000001f,0x0003001e,0x00000023,0x00000022,0x00040020,0x00000024,0x00000002,
    0x00000023,0x0004003b,0x00000024,0x00000025,0x00000002,0x00040020,0x00000027,0x00000002,
    0x0000001f,0x0004002b,0x00000014,0x0000002a,0x00000001,0x0003001d,0x00000030,0x0000001f,
    0x0003001e,0x00000031,0x00000030,0x00040020,0x00000032,0x00000002,0x00000031,0x0004003b,
    0x00000032,0x00000033,0x00000002,0x00090019,0x00000035,0x0000001e,0x00000001,0x00000000,
    0x00000000,0x00000000,0x00000001,0x00000000,0x0003001b,0x00000036,0x00000035,0x00040020,
    0x00000037,0x00000000,0x00000036,0x0004003b,0x00000037,0x00000038,0x00000000,0x00040017,
    0x0000003a,0x0000001e,0x00000002,0x0004002b,0x00000006,0x0000003d,0x00000002,0x00040020,
    0x0000003e,0x00000007,0x0000001e,0x0004002b,0x0000001e,0x0000004a,0x00000000,0x0004002b,
    0x00000006,0x0000004c,0x00000003,0x0004002b,0x00000006,0x00000052,0x00000040,0x0004002b,
    0x00000006,0x00000053,0x00000001,0x0006002c,0x00000009,0x00000054,0x00000052,0x00000053,
    0x00000053,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,
    0x0004003b,0x00000007,0x00000008,0x00000007,0x0004003b,0x00000020,0x00000021,0x00000007,
    0x00050041,0x0000000d,0x0000000e,0x0000000b,0x0000000c,0x0004003d,0x00000006,0x0000000f,
    0x0000000e,0x0003003e,0x00000008,0x0000000f,0x0004003d,0x00000006,0x00000010,0x00000008,
    0x00050041,0x00000016,0x00000017,0x00000013,0x00000015,0x0004003d,0x00000006,0x00000018,
    0x00000017,0x000500ae,0x00000019,0x0000001a,0x00000010,0x00000018,0x000300f7,0x0000001c,
    0x00000000,0x000400fa,0x0000001a,0x0000001b,0x0000001c,0x000200f8,0x0000001b,0x000100fd,
    0x000200f8,0x0000001c,0x0004003d,0x00000006,0x00000026,0x00000008,0x00060041,0x00000027,
    0x00000028,0x00000025,0x00000015,0x00000026,0x0004003d,0x0000001f,0x00000029,0x00000028,
    0x0003003e,0x00000021,0x00000029,0x00050041,0x00000016,0x0000002b,0x00000013,0x0000002a,
    0x0004003d,0x00000006,0x0000002c,0x0000002b,0x000500aa,0x00000019,0x0000002d,0x0000002c,
    0x0000000c,0x000300f7,0x0000002f,0x00000000,0x000400fa,0x0000002d,0x0000002e,0x00000043,
    0x000200f8,0x0000002e,0x0004003d,0x00000006,0x00000034,0x00000008,0x0004003d,0x00000036,
    0x00000039,0x00000038,0x0004003d,0x0000001f,0x0000003b,0x00000021,0x0007004f,0x0000003a,
    0x0000003c,0x0000003b,0x0000003b,0x00000000,0x00000001,0x00050041,0x0000003e,0x0000003f,
    0x00000021,0x0000003d,0x0004003d,0x0000001e,0x00000040,0x0000003f,0x00070058,0x0000001f,
    0x00000041,0x00000039,0x0000003c,0x00000002,0x00000040,0x00060041,0x00000027,0x00000042,
    0x00000033,0x00000015,0x00000034,0x0003003e,0x00000042,0x00000041,0x000200f9,0x0000002f,
    0x000200f8,0x00000043,0x0004003d,0x00000006,0x00000044,0x00000008,0x0004003d,0x00000036,
    0x00000045,0x00000038,0x0004003d,0x0000001f,0x00000046,0x00000021,0x0007004f,0x0000003a,
    0x00000047,0x00000046,0x00000046,0x00000000,0x00000001,0x00050041,0x0000003e,0x00000048,
    0x00000021,0x0000003d,0x0004003d,0x0000001e,0x00000049,0x00000048,0x00050050,0x0000003a,
    0x0000004b,0x00000049,0x0000004a,0x00050041,0x0000003e,0x0000004d,0x00000021,0x0000004c,
    0x0004003d,0x0000001e,0x0000004e,0x0000004d,0x00050050,0x0000003a,0x0000004f,0x0000004a,
    0x0000004e,0x00080058,0x0000001f,0x00000050,0x00000045,0x00000047,0x00000004,0x0000004b,
    0x0000004f,0x00060041,0x00000027,0x00000051,0x00000033,0x00000015,0x00000044,0x0003003e,
    0x00000051,0x00000050,0x000200f9,0x0000002f,0x000200f8,0x0000002f,0x000100fd,0x00010038,
};

/*
 * #version 450
 * layout(location = 0) out vec2 uv;
 * void main()
 * {
 *     vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
 *     uv = p;
 *     gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
 * }
 */
static const uint32_t spv_quad_vert[] = {
    0x07230203,0x00010000,0x0008000b,0x0000002e,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0008000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x0000000c,0x00000018,0x00000020,
    0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00030005,
    0x00000009,0x00000070,0x00060005,0x0000000c,0x565f6c67,0x65747265,0x646e4978,0x00007865,
    0x00030005,0x00000018,0x00007675,0x00060005,0x0000001e,0x505f6c67,0x65567265,0x78657472,
    0x00000000,0x00060006,0x0000001e,0x00000000,0x505f6c67,0x7469736f,0x006e6f69,0x00070006,
    0x0000001e,0x00000001,0x505f6c67,0x746e696f,0x657a6953,0x00000000,0x00070006,0x0000001e,
    0x00000002,0x435f6c67,0x4470696c,0x61747369,0x0065636e,0x00070006,0x0000001e,0x00000003,
    0x435f6c67,0x446c6c75,0x61747369,0x0065636e,0x00030005,0x00000020,0x00000000,0x00040047,
    0x0000000c,0x0000000b,0x0000002a,0x00040047,0x00000018,0x0000001e,0x00000000,0x00030047,
    0x0000001e,0x00000002,0x00050048,0x0000001e,0x00000000,0x0000000b,0x00000000,0x00050048,
    0x0000001e,0x00000001,0x0000000b,0x00000001,0x00050048,0x0000001e,0x00000002,0x0000000b,
    0x00000003,0x00050048,0x0000001e,0x00000003,0x0000000b,0x00000004,0x00020013,0x00000002,
    0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,
    0x00000006,0x00000002,0x00040020,0x00000008,0x00000007,0x00000007,0x00040015,0x0000000a,
    0x00000020,0x00000001,0x00040020,0x0000000b,0x00000001,0x0000000a,0x0004003b,0x0000000b,
    0x0000000c,0x00000001,0x0004002b,0x0000000a,0x0000000e,0x00000001,0x0004002b,0x0000000a,
    0x00000010,0x00000002,0x00040020,0x00000017,0x00000003,0x00000007,0x0004003b,0x00000017,
    0x00000018,0x00000003,0x00040017,0x0000001a,0x00000006,0x00000004,0x00040015,0x0000001b,
    0x00000020,0x00000000,0x0004002b,0x0000001b,0x0000001c,0x00000001,0x0004001c,0x0000001d,
    0x00000006,0x0000001c,0x0006001e,0x0000001e,0x0000001a,0x00000006,0x0000001d,0x0000001d,
    0x00040020,0x0000001f,0x00000003,0x0000001e,0x0004003b,0x0000001f,0x00000020,0x00000003,
    0x0004002b,0x0000000a,0x00000021,0x00000000,0x0004002b,0x00000006,0x00000023,0x40000000,
    0x0004002b,0x00000006,0x00000025,0x3f800000,0x0004002b,0x00000006,0x00000028,0x00000000,
    0x00040020,0x0000002c,0x00000003,0x0000001a,0x00050036,0x00000002,0x00000004,0x00000000,
    0x00000003,0x000200f8,0x00000005,0x0004003b,0x00000008,0x00000009,0x00000007,0x0004003d,
    0x0000000a,0x0000000d,0x0000000c,0x000500c4,0x0000000a,0x0000000f,0x0000000d,0x0000000e,
    0x000500c7,0x0000000a,0x00000011,0x0000000f,0x00000010,0x0004006f,0x00000006,0x00000012,
    0x00000011,0x0004003d,0x0000000a,0x00000013,0x0000000c,0x000500c7,0x0000000a,0x00000014,
    0x00000013,0x00000010,0x0004006f,0x00000006,0x00000015,0x00000014,0x00050050,0x00000007,
    0x00000016,0x00000012,0x00000015,0x0003003e,0x00000009,0x00000016,0x0004003d,0x00000007,
    0x00000019,0x00000009,0x0003003e,0x00000018,0x00000019,0x0004003d,0x00000007,0x00000022,
    0x00000009,0x0005008e,0x00000007,0x00000024,0x00000022,0x00000023,0x00050050,0x00000007,
    0x00000026,0x00000025,0x00000025,0x00050083,0x00000007,0x00000027,0x00000024,0x00000026,
    0x00050051,0x00000006,0x00000029,0x00000027,0x00000000,0x00050051,0x00000006,0x0000002a,
    0x00000027,0x00000001,0x00070050,0x0000001a,0x0000002b,0x00000029,0x0000002a,0x00000028,
    0x00000025,0x00050041,0x0000002c,0x0000002d,0x00000020,0x00000021,0x0003003e,0x0000002d,
    0x0000002b,0x000100fd,0x00010038,
};

/*
 * #version 450
 * layout(set = 0, binding = 0) uniform sampler2D ycbcr;
 * layout(location = 0) in vec2 uv;
 * layout(location = 0) out vec4 color;
 * void main()
 * {
 *     color = texture(ycbcr, uv);
 * }
 */
static const uint32_t spv_sample_frag[] = {
    0x07230203,0x00010000,0x0008000b,0x00000014,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0007000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000009,0x00000011,0x00030010,
    0x00000004,0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,
    0x00000000,0x00040005,0x00000009,0x6f6c6f63,0x00000072,0x00040005,0x0000000d,0x63626379,
    0x00000072,0x00030005,0x00000011,0x00007675,0x00040047,0x00000009,0x0000001e,0x00000000,
    0x00040047,0x0000000d,0x00000021,0x00000000,0x00040047,0x0000000d,0x00000022,0x00000000,
    0x00040047,0x00000011,0x0000001e,0x00000000,0x00020013,0x00000002,0x00030021,0x00000003,
    0x00000002,0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,0x00000006,0x00000004,
    0x00040020,0x00000008,0x00000003,0x00000007,0x0004003b,0x00000008,0x00000009,0x00000003,
    0x00090019,0x0000000a,0x00000006,0x00000001,0x00000000,0x00000000,0x00000000,0x00000001,
    0x00000000,0x0003001b,0x0000000b,0x0000000a,0x00040020,0x0000000c,0x00000000,0x0000000b,
    0x0004003b,0x0000000c,0x0000000d,0x00000000,0x00040017,0x0000000f,0x00000006,0x00000002,
    0x00040020,0x00000010,0x00000001,0x0000000f,0x0004003b,0x00000010,0x00000011,0x00000001,
    0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,0x0004003d,
    0x0000000b,0x0000000e,0x0000000d,0x0004003d,0x0000000f,0x00000012,0x00000011,0x00050057,
    0x00000007,0x00000013,0x0000000e,0x00000012,0x0003003e,0x00000009,0x00000013,0x000100fd,
    0x00010038,
};

/* ---- Formats ---- */

struct yfmt {
    VkFormat format;
    const char *name;
    uint32_t planes; /* 1: packed 4:2:2 */
    uint32_t packed; /* 1: G8B8G8R8 (Y in bytes 0, 2), 2: B8G8R8G8 (Y in bytes 1, 3) */
    uint32_t xs, ys; /* chroma subsampling */
    VkFormat plane_formats[3];
    uint32_t plane_bytes[3];
};

static const struct yfmt yfmts[] = {
    { VK_FORMAT_G8B8G8R8_422_UNORM, "G8B8G8R8_422", 1, 1, 1, 0, { VK_FORMAT_UNDEFINED }, { 2 } },
    { VK_FORMAT_B8G8R8G8_422_UNORM, "B8G8R8G8_422", 1, 2, 1, 0, { VK_FORMAT_UNDEFINED }, { 2 } },
    { VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM, "G8_B8_R8_3PLANE_420", 3, 0, 1, 1,
      { VK_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM }, { 1, 1, 1 } },
    { VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, "G8_B8R8_2PLANE_420", 2, 0, 1, 1,
      { VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM }, { 1, 2 } },
    { VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM, "G8_B8_R8_3PLANE_422", 3, 0, 1, 0,
      { VK_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM }, { 1, 1, 1 } },
    { VK_FORMAT_G8_B8R8_2PLANE_422_UNORM, "G8_B8R8_2PLANE_422", 2, 0, 1, 0,
      { VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM }, { 1, 2 } },
    { VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM, "G8_B8_R8_3PLANE_444", 3, 0, 0, 0,
      { VK_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM }, { 1, 1, 1 } },
    { VK_FORMAT_G8_B8R8_2PLANE_444_UNORM, "G8_B8R8_2PLANE_444", 2, 0, 0, 0,
      { VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM }, { 1, 2 } },
};

#define FMT_PACKED_GBGR 0
#define FMT_PACKED_BGRG 1
#define FMT_3PLANE_420 2
#define FMT_2PLANE_420 3
#define FMT_3PLANE_422 4
#define FMT_2PLANE_422 5
#define FMT_3PLANE_444 6
#define FMT_2PLANE_444 7

static const VkImageAspectFlagBits plane_aspects[3] = { VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT,
                                                        VK_IMAGE_ASPECT_PLANE_2_BIT };

/* Plane p of an image of format f, w x h texels: extent and bytes. */
static void plane_extent(const struct yfmt *f, uint32_t p, uint32_t w, uint32_t h, uint32_t *pw, uint32_t *ph)
{
    *pw = p ? w >> f->xs : w;
    *ph = p ? h >> f->ys : h;
}

static VkImageAspectFlagBits plane_aspect(const struct yfmt *f, uint32_t p)
{
    return f->planes == 1 ? VK_IMAGE_ASPECT_COLOR_BIT : plane_aspects[p];
}

/* ---- Vulkan setup ---- */

static VkInstance instance;
static VkPhysicalDevice pd;
static VkDevice device;
static VkQueue queue;
static VkCommandPool cmd_pool;
static VkFence fence;
static uint32_t private_type = UINT32_MAX, shared_type = UINT32_MAX;

struct buf {
    VkBuffer buffer;
    VkDeviceMemory memory;
    unsigned char *data;
    VkDeviceSize size;
};

static void setup(void)
{
    VkPhysicalDeviceMemoryProperties mem;
    VkPhysicalDeviceVulkan11Features v11;
    VkPhysicalDeviceVulkan14Features v14;
    VkInstanceCreateInfo instance_info;
    VkApplicationInfo app;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo device_info;
    VkCommandPoolCreateInfo pool_info;
    VkFenceCreateInfo fence_info;
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
    memset(&v14, 0, sizeof(v14));
    v14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    v14.hostImageCopy = VK_TRUE;
    memset(&v11, 0, sizeof(v11));
    v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    v11.pNext = &v14;
    v11.samplerYcbcrConversion = VK_TRUE;
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &v11;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    REQUIRE_VK(vkCreateDevice(pd, &device_info, NULL, &device));
    vkGetDeviceQueue(device, 0, 0, &queue);
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    for (i = 0; i < mem.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mem.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) && shared_type == UINT32_MAX)
            shared_type = i;
        else if (!(f & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)) &&
                 private_type == UINT32_MAX)
            private_type = i;
    }
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    REQUIRE_VK(vkCreateCommandPool(device, &pool_info, NULL, &cmd_pool));
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    REQUIRE_VK(vkCreateFence(device, &fence_info, NULL, &fence));
}

static void teardown(void)
{
    vkDestroyFence(device, fence, NULL);
    vkDestroyCommandPool(device, cmd_pool, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
}

static struct buf make_buf(VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo info;
    VkMemoryAllocateInfo alloc;
    VkMemoryRequirements req;
    struct buf b;
    void *p = NULL;

    memset(&b, 0, sizeof(b));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage;
    if (vkCreateBuffer(device, &info, NULL, &b.buffer) != VK_SUCCESS)
        return b;
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = shared_type;
    if (vkAllocateMemory(device, &alloc, NULL, &b.memory) != VK_SUCCESS)
        return b;
    vkBindBufferMemory(device, b.buffer, b.memory, 0);
    if (vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS) {
        b.data = p;
        memset(b.data, 0, (size_t)size);
    }
    b.size = size;
    return b;
}

static void free_buf(struct buf *b)
{
    vkDestroyBuffer(device, b->buffer, NULL);
    vkFreeMemory(device, b->memory, NULL);
    memset(b, 0, sizeof(*b));
}

static VkCommandBuffer begin(void)
{
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferBeginInfo bi;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = cmd_pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &alloc, &cb) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

/* All earlier commands against all later ones, host reads included. */
static void barrier(VkCommandBuffer cb)
{
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
}

static VkResult finish(VkCommandBuffer cb)
{
    VkSubmitInfo si;
    VkResult r;
    vkEndCommandBuffer(cb);
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    r = vkQueueSubmit(queue, 1, &si, fence);
    if (r == VK_SUCCESS)
        r = vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ull);
    vkResetFences(device, 1, &fence);
    vkFreeCommandBuffers(device, cmd_pool, 1, &cb);
    return r;
}

static uint32_t first_type(uint32_t bits, uint32_t preferred)
{
    uint32_t i;
    if (preferred != UINT32_MAX && (bits & (1u << preferred)))
        return preferred;
    for (i = 0; i < 32; i++)
        if (bits & (1u << i))
            return i;
    return UINT32_MAX;
}

/* ---- Multi-planar images ---- */

struct yimg {
    const struct yfmt *f;
    VkImage image;
    VkDeviceMemory memory[3]; /* disjoint: one per plane */
    uint32_t w, h;
    VkImageTiling tiling;
    VkImageCreateFlags flags;
};

/* Creates and binds an image of format f (memory: host-visible for linear
 * images, else device-local). */
static bool make_yimg(struct yimg *m, const struct yfmt *f, uint32_t w, uint32_t h, VkImageTiling tiling,
                      VkImageCreateFlags flags, VkImageUsageFlags usage)
{
    VkImageCreateInfo info;
    VkMemoryAllocateInfo alloc;
    VkBindImageMemoryInfo bind[3];
    VkBindImagePlaneMemoryInfo bp[3];
    uint32_t p, n;

    memset(m, 0, sizeof(*m));
    m->f = f;
    m->w = w;
    m->h = h;
    m->tiling = tiling;
    m->flags = flags;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.flags = flags;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = f->format;
    info.extent.width = w;
    info.extent.height = h;
    info.extent.depth = 1;
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = tiling;
    info.usage = usage;
    if (vkCreateImage(device, &info, NULL, &m->image) != VK_SUCCESS)
        return false;
    n = (flags & VK_IMAGE_CREATE_DISJOINT_BIT) ? f->planes : 1;
    for (p = 0; p < n; p++) {
        VkImageMemoryRequirementsInfo2 ri;
        VkImagePlaneMemoryRequirementsInfo pi;
        VkMemoryRequirements2 req;
        memset(&pi, 0, sizeof(pi));
        pi.sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO;
        pi.planeAspect = plane_aspects[p];
        memset(&ri, 0, sizeof(ri));
        ri.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
        ri.pNext = (flags & VK_IMAGE_CREATE_DISJOINT_BIT) ? &pi : NULL;
        ri.image = m->image;
        memset(&req, 0, sizeof(req));
        req.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
        vkGetImageMemoryRequirements2(device, &ri, &req);
        memset(&alloc, 0, sizeof(alloc));
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = req.memoryRequirements.size;
        alloc.memoryTypeIndex = first_type(req.memoryRequirements.memoryTypeBits,
                                           tiling == VK_IMAGE_TILING_LINEAR ? shared_type : private_type);
        if (!req.memoryRequirements.size || vkAllocateMemory(device, &alloc, NULL, &m->memory[p]) != VK_SUCCESS)
            return false;
        memset(&bp[p], 0, sizeof(bp[p]));
        bp[p].sType = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO;
        bp[p].planeAspect = plane_aspects[p];
        memset(&bind[p], 0, sizeof(bind[p]));
        bind[p].sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
        bind[p].pNext = (flags & VK_IMAGE_CREATE_DISJOINT_BIT) ? &bp[p] : NULL;
        bind[p].image = m->image;
        bind[p].memory = m->memory[p];
    }
    return vkBindImageMemory2(device, n, bind) == VK_SUCCESS; /* disjoint: every plane in one call */
}

static void free_yimg(struct yimg *m)
{
    uint32_t p;
    vkDestroyImage(device, m->image, NULL);
    for (p = 0; p < 3; p++)
        vkFreeMemory(device, m->memory[p], NULL);
    memset(m, 0, sizeof(*m));
}

/* Plane data in host memory, tightly packed rows (plane p at offsets[p]). */
struct planes {
    unsigned char *data;
    size_t offsets[3], sizes[3], total;
};

static void planes_init(struct planes *pl, const struct yfmt *f, uint32_t w, uint32_t h, uint32_t seed)
{
    uint32_t p, x = seed * 2654435761u + 12345u;
    size_t i;
    memset(pl, 0, sizeof(*pl));
    for (p = 0; p < f->planes; p++) {
        uint32_t pw, ph;
        plane_extent(f, p, w, h, &pw, &ph);
        pl->offsets[p] = pl->total;
        pl->sizes[p] = (size_t)pw * ph * f->plane_bytes[p];
        pl->total += pl->sizes[p];
    }
    pl->data = malloc(pl->total);
    for (i = 0; pl->data && i < pl->total; i++) {
        x = x * 1103515245u + 12345u;
        pl->data[i] = (unsigned char)(x >> 16);
    }
}

/* Records copies of every plane between a buffer laid out as pl and m. */
static void copy_planes(VkCommandBuffer cb, const struct buf *b, const struct planes *pl, const struct yimg *m,
                        bool to_image)
{
    uint32_t p;
    for (p = 0; p < m->f->planes; p++) {
        VkBufferImageCopy r;
        uint32_t pw, ph;
        plane_extent(m->f, p, m->w, m->h, &pw, &ph);
        memset(&r, 0, sizeof(r));
        r.bufferOffset = pl->offsets[p];
        r.imageSubresource.aspectMask = plane_aspect(m->f, p);
        r.imageSubresource.layerCount = 1;
        r.imageExtent.width = pw;
        r.imageExtent.height = ph;
        r.imageExtent.depth = 1;
        if (to_image)
            vkCmdCopyBufferToImage(cb, b->buffer, m->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
        else
            vkCmdCopyImageToBuffer(cb, m->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b->buffer, 1, &r);
    }
}

/* Uploads pl into every plane of m. */
static bool upload(const struct yimg *m, const struct planes *pl)
{
    struct buf b = make_buf(pl->total, 0);
    VkCommandBuffer cb;
    bool ok;
    if (!b.data)
        return false;
    memcpy(b.data, pl->data, pl->total);
    cb = begin();
    copy_planes(cb, &b, pl, m, true);
    barrier(cb);
    ok = finish(cb) == VK_SUCCESS;
    free_buf(&b);
    return ok;
}

/* Reads every plane of m back and compares with pl. */
static bool readback_matches(const struct yimg *m, const struct planes *pl)
{
    struct buf b = make_buf(pl->total, 0);
    VkCommandBuffer cb;
    bool ok;
    if (!b.data)
        return false;
    memset(b.data, 0xA5, pl->total);
    cb = begin();
    copy_planes(cb, &b, pl, m, false);
    barrier(cb);
    ok = finish(cb) == VK_SUCCESS && memcmp(b.data, pl->data, pl->total) == 0;
    free_buf(&b);
    return ok;
}

/* ---- CPU reference: Textures > Sampler Y'CbCr Conversion ---- */

struct conv {
    VkSamplerYcbcrModelConversion model;
    VkSamplerYcbcrRange range;
    VkComponentMapping components;
    VkChromaLocation x_chroma, y_chroma;
    VkFilter chroma_filter;
    VkBool32 force_explicit;
    VkFilter min_filter, mag_filter;
};

struct ref {
    const struct yfmt *f;
    const struct planes *pl;
    uint32_t w, h;
};

static int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

static float luma_at(const struct ref *r, int i, int j)
{
    i = clampi(i, 0, (int)r->w - 1);
    j = clampi(j, 0, (int)r->h - 1);
    if (r->f->planes == 1)
        return r->pl->data[((size_t)j * r->w + (size_t)i) * 2 + (r->f->packed == 1 ? 0 : 1)] / 255.0f;
    return r->pl->data[(size_t)j * r->w + (size_t)i] / 255.0f;
}

/* (Cb, Cr) of chroma texel (k, l), clamped to the chroma extent. */
static void chroma_at(const struct ref *r, int k, int l, float ch[2])
{
    uint32_t cw = r->w >> r->f->xs, chh = r->h >> r->f->ys;
    const unsigned char *d = r->pl->data;
    k = clampi(k, 0, (int)cw - 1);
    l = clampi(l, 0, (int)chh - 1);
    if (r->f->planes == 1) {
        size_t row = (size_t)l * r->w, c = r->f->packed == 1 ? 1 : 0;
        ch[0] = d[(row + 2 * (size_t)k) * 2 + c] / 255.0f;
        ch[1] = d[(row + 2 * (size_t)k + 1) * 2 + c] / 255.0f;
    } else if (r->f->planes == 3) {
        ch[0] = d[r->pl->offsets[1] + (size_t)l * cw + (size_t)k] / 255.0f;
        ch[1] = d[r->pl->offsets[2] + (size_t)l * cw + (size_t)k] / 255.0f;
    } else {
        ch[0] = d[r->pl->offsets[1] + ((size_t)l * cw + (size_t)k) * 2] / 255.0f;
        ch[1] = d[r->pl->offsets[1] + ((size_t)l * cw + (size_t)k) * 2 + 1] / 255.0f;
    }
}

/* Whether nearest sampling at unnormalized u is within 2% of a texel of a
 * boundary between texels. */
static bool near_edge(float u)
{
    float f = u - floorf(u);
    return f < 0.02f || f > 0.98f;
}

/* Bilinear (or nearest) of the luma or chroma grid at unnormalized (u, v). */
static void filter_grid(const struct ref *r, bool chroma, bool linear, float u, float v, float out[2], bool *ambiguous)
{
    if (!linear) {
        if (near_edge(u) || near_edge(v))
            *ambiguous = true;
        if (chroma) {
            chroma_at(r, (int)floorf(u), (int)floorf(v), out);
        } else {
            out[0] = luma_at(r, (int)floorf(u), (int)floorf(v));
            out[1] = 0.0f;
        }
    } else {
        float bu = u - 0.5f, bv = v - 0.5f, fu = bu - floorf(bu), fv = bv - floorf(bv), t[4][2];
        int i = (int)floorf(bu), j = (int)floorf(bv), n;
        for (n = 0; n < 4; n++) {
            int di = n & 1, dj = n >> 1;
            if (chroma) {
                chroma_at(r, i + di, j + dj, t[n]);
            } else {
                t[n][0] = luma_at(r, i + di, j + dj);
                t[n][1] = 0.0f;
            }
        }
        for (n = 0; n < 2; n++)
            out[n] = (t[0][n] * (1 - fu) + t[1][n] * fu) * (1 - fv) + (t[2][n] * (1 - fu) + t[3][n] * fu) * fv;
    }
}

/* Explicit reconstruction: (R, G, B) at luma texel (i, j). */
static void texel_explicit(const struct ref *r, const struct conv *c, int i, int j, float rgb[3])
{
    float ch[2];
    i = clampi(i, 0, (int)r->w - 1);
    j = clampi(j, 0, (int)r->h - 1);
    if (c->chroma_filter == VK_FILTER_NEAREST || (!r->f->xs && !r->f->ys)) {
        chroma_at(r, i >> r->f->xs, j >> r->f->ys, ch);
    } else {
        float fi = !r->f->xs ? (float)i : c->x_chroma == VK_CHROMA_LOCATION_COSITED_EVEN ? 0.5f * i : 0.5f * (i - 0.5f);
        float fj = !r->f->ys ? (float)j : c->y_chroma == VK_CHROMA_LOCATION_COSITED_EVEN ? 0.5f * j : 0.5f * (j - 0.5f);
        float wi = fi - floorf(fi), wj = fj - floorf(fj), t[4][2];
        int k = (int)floorf(fi), l = (int)floorf(fj), n;
        for (n = 0; n < 4; n++)
            chroma_at(r, k + (n & 1), l + (n >> 1), t[n]);
        for (n = 0; n < 2; n++)
            ch[n] = (t[0][n] * (1 - wi) + t[1][n] * wi) * (1 - wj) + (t[2][n] * (1 - wi) + t[3][n] * wi) * wj;
    }
    rgb[0] = ch[1];
    rgb[1] = luma_at(r, i, j);
    rgb[2] = ch[0];
}

static float component(VkComponentSwizzle s, int identity, const float t[4])
{
    switch (s) {
    case VK_COMPONENT_SWIZZLE_ZERO: return 0.0f;
    case VK_COMPONENT_SWIZZLE_ONE: return 1.0f;
    case VK_COMPONENT_SWIZZLE_R: return t[0];
    case VK_COMPONENT_SWIZZLE_G: return t[1];
    case VK_COMPONENT_SWIZZLE_B: return t[2];
    case VK_COMPONENT_SWIZZLE_A: return t[3];
    default: return t[identity];
    }
}

/* Component mapping, range expansion, model conversion (8-bit). */
static void convert(const struct conv *c, const float t[4], float out[4])
{
    float s[4], y, cb, cr;
    s[0] = component(c->components.r, 0, t);
    s[1] = component(c->components.g, 1, t);
    s[2] = component(c->components.b, 2, t);
    s[3] = component(c->components.a, 3, t);
    if (c->model == VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY) {
        memcpy(out, s, sizeof(s));
        return;
    }
    if (c->range == VK_SAMPLER_YCBCR_RANGE_ITU_FULL) {
        y = s[1];
        cb = s[2] - 128.0f / 255.0f;
        cr = s[0] - 128.0f / 255.0f;
    } else {
        y = (s[1] * 255.0f - 16.0f) / 219.0f;
        cb = (s[2] * 255.0f - 128.0f) / 224.0f;
        cr = (s[0] * 255.0f - 128.0f) / 224.0f;
    }
    out[3] = s[3];
    switch (c->model) {
    case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709:
        out[0] = y + 1.5748f * cr;
        out[1] = y - (0.13397432f / 0.7152f) * cb - (0.33480248f / 0.7152f) * cr;
        out[2] = y + 1.8556f * cb;
        break;
    case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601:
        out[0] = y + 1.402f * cr;
        out[1] = y - (0.202008f / 0.587f) * cb - (0.419198f / 0.587f) * cr;
        out[2] = y + 1.772f * cb;
        break;
    case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020:
        out[0] = y + 1.4746f * cr;
        out[1] = y - (0.11156702f / 0.6780f) * cb - (0.38737742f / 0.6780f) * cr;
        out[2] = y + 1.8814f * cb;
        break;
    default: /* YCbCr identity */
        out[0] = cr;
        out[1] = y;
        out[2] = cb;
        break;
    }
}

/* The sample at normalized (s, t) with LOD lod (the sampler's min/max LOD
 * are 0 and VK_LOD_CLAMP_NONE, its bias 0). */
static void ref_sample(const struct ref *r, const struct conv *c, float s, float t, float lod, float out[4],
                       bool *ambiguous, bool *linear)
{
    bool lin = (lod > 0.0f ? c->min_filter : c->mag_filter) == VK_FILTER_LINEAR;
    bool explicit_rec = c->force_explicit;
    float u = s * r->w, v = t * r->h, rgba[4] = { 0, 0, 0, 1 };

    *linear = lin;
    if (explicit_rec) {
        if (lin) {
            float bu = u - 0.5f, bv = v - 0.5f, fu = bu - floorf(bu), fv = bv - floorf(bv), q[4][3];
            int i = (int)floorf(bu), j = (int)floorf(bv), n, k;
            for (n = 0; n < 4; n++)
                texel_explicit(r, c, i + (n & 1), j + (n >> 1), q[n]);
            for (k = 0; k < 3; k++)
                rgba[k] = (q[0][k] * (1 - fu) + q[1][k] * fu) * (1 - fv) + (q[2][k] * (1 - fu) + q[3][k] * fu) * fv;
        } else {
            if (near_edge(u) || near_edge(v))
                *ambiguous = true;
            texel_explicit(r, c, (int)floorf(u), (int)floorf(v), rgba);
        }
        if (c->chroma_filter == VK_FILTER_LINEAR)
            *linear = true; /* (computed in float either way) */
    } else {
        float g[2], ch[2];
        float cu = !r->f->xs ? u : c->x_chroma == VK_CHROMA_LOCATION_COSITED_EVEN ? 0.5f * (u + 0.5f) : 0.5f * u;
        float cv = !r->f->ys ? v : c->y_chroma == VK_CHROMA_LOCATION_COSITED_EVEN ? 0.5f * (v + 0.5f) : 0.5f * v;
        filter_grid(r, false, lin, u, v, g, ambiguous);
        filter_grid(r, true, c->chroma_filter == VK_FILTER_LINEAR, cu, cv, ch, ambiguous);
        if (c->chroma_filter == VK_FILTER_LINEAR)
            *linear = true;
        rgba[0] = ch[1];
        rgba[1] = g[0];
        rgba[2] = ch[0];
    }
    convert(c, rgba, out);
}

static VkComponentMapping identity_mapping(void)
{
    VkComponentMapping m;
    m.r = m.g = m.b = m.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    return m;
}

/* ---- Sampling through the Vulkan API (compute) ---- */

#define IMG_W 16u
#define IMG_H 8u
#define NCOORDS 128u

/* Coordinates: texel fractions away from luma and chroma boundaries, some
 * outside the image (clamp to edge); LOD from `lod` (mode 0) or gradients
 * of that scale (mode 1). */
static void make_coords(float *coords, uint32_t n, float lod, uint32_t mode)
{
    static const float fr[] = { 0.15f, 0.3f, 0.7f, 0.85f, 0.4f, 0.6f };
    uint32_t i;
    for (i = 0; i < n; i++) {
        float *c = &coords[4 * i];
        int xi = (int)(i % 18) - 1, yi = (int)((i / 18) % 10) - 1;
        c[0] = (xi + fr[i % 6]) / IMG_W;
        c[1] = (yi + fr[(i / 3 + 1) % 6]) / IMG_H;
        if (mode == 0) {
            c[2] = lod;
            c[3] = 0.0f;
        } else { /* textureGrad: (c.z, 0) and (0, c.w); LOD log2(c.z * W) */
            c[2] = exp2f(lod) / IMG_W;
            c[3] = exp2f(lod) / IMG_H;
        }
    }
}

struct sample_case {
    const char *name;
    uint32_t format;     /* index in yfmts */
    struct conv conv;
    VkImageTiling tiling;
    VkImageCreateFlags flags;
    float lod;
    uint32_t mode;       /* 0: textureLod, 1: textureGrad */
};

/* Fills linear image m's planes through mapped memory, at the offsets and
 * row pitches vkGetImageSubresourceLayout gives. */
static bool write_linear(const struct yimg *m, const struct planes *pl)
{
    uint32_t p, row;
    for (p = 0; p < m->f->planes; p++) {
        VkImageSubresource sub;
        VkSubresourceLayout layout;
        uint32_t pw, ph, mem = (m->flags & VK_IMAGE_CREATE_DISJOINT_BIT) ? p : 0;
        unsigned char *dst;
        void *map;
        plane_extent(m->f, p, m->w, m->h, &pw, &ph);
        sub.aspectMask = plane_aspect(m->f, p);
        sub.mipLevel = 0;
        sub.arrayLayer = 0;
        vkGetImageSubresourceLayout(device, m->image, &sub, &layout);
        if (layout.rowPitch < (VkDeviceSize)pw * m->f->plane_bytes[p] || layout.size < layout.rowPitch * ph)
            return false;
        if (vkMapMemory(device, m->memory[mem], 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS)
            return false;
        dst = (unsigned char *)map + layout.offset;
        for (row = 0; row < ph; row++)
            memcpy(dst + layout.rowPitch * row, pl->data + pl->offsets[p] + (size_t)row * pw * m->f->plane_bytes[p],
                   (size_t)pw * m->f->plane_bytes[p]);
        vkUnmapMemory(device, m->memory[mem]);
    }
    return true;
}

static VkSamplerYcbcrConversion make_conversion(const struct yfmt *f, const struct conv *c)
{
    VkSamplerYcbcrConversionCreateInfo info;
    VkSamplerYcbcrConversion conv = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO;
    info.format = f->format;
    info.ycbcrModel = c->model;
    info.ycbcrRange = c->range;
    info.components = c->components;
    info.xChromaOffset = c->x_chroma;
    info.yChromaOffset = c->y_chroma;
    info.chromaFilter = c->chroma_filter;
    info.forceExplicitReconstruction = c->force_explicit;
    if (vkCreateSamplerYcbcrConversion(device, &info, NULL, &conv) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return conv;
}

static VkSampler make_sampler(VkSamplerYcbcrConversion conv, VkFilter min_filter, VkFilter mag_filter)
{
    VkSamplerYcbcrConversionInfo ci;
    VkSamplerCreateInfo info;
    VkSampler s = VK_NULL_HANDLE;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
    ci.conversion = conv;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.pNext = conv ? &ci : NULL;
    info.magFilter = mag_filter;
    info.minFilter = min_filter;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = info.addressModeV = info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.minLod = 0.0f;
    info.maxLod = VK_LOD_CLAMP_NONE;
    vkCreateSampler(device, &info, NULL, &s);
    return s;
}

static VkImageView make_view(VkImage image, VkFormat format, VkImageAspectFlags aspect, VkSamplerYcbcrConversion conv)
{
    VkSamplerYcbcrConversionInfo ci;
    VkImageViewCreateInfo info;
    VkImageView v = VK_NULL_HANDLE;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
    ci.conversion = conv;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.pNext = conv ? &ci : NULL;
    info.image = image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = format;
    info.components = identity_mapping();
    info.subresourceRange.aspectMask = aspect;
    info.subresourceRange.levelCount = 1;
    info.subresourceRange.layerCount = 1;
    vkCreateImageView(device, &info, NULL, &v);
    return v;
}

/* The compute shader sampling `view` through `sampler` (immutable) at
 * coords (NCOORDS vec4) into results. descriptors: combined image sampler
 * descriptors the pool needs for it. */
static bool run_compute(VkImageView view, VkSampler sampler, uint32_t descriptors, const float *coords, uint32_t mode,
                        float *results)
{
    VkDescriptorSetLayoutBinding b[3];
    VkDescriptorSetLayoutCreateInfo li;
    VkPipelineLayoutCreateInfo pli;
    VkPushConstantRange pcr;
    VkShaderModuleCreateInfo smi;
    VkComputePipelineCreateInfo cpi;
    VkDescriptorPoolSize sizes[2];
    VkDescriptorPoolCreateInfo dpi;
    VkDescriptorSetAllocateInfo dai;
    VkDescriptorImageInfo ii;
    VkDescriptorBufferInfo bi[2];
    VkWriteDescriptorSet w[2];
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    struct buf cbuf = make_buf(NCOORDS * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    struct buf rbuf = make_buf(NCOORDS * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    uint32_t push[2] = { NCOORDS, mode };
    bool ok = false;
    VkCommandBuffer cb;

    memset(b, 0, sizeof(b));
    b[0].binding = 0;
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[0].descriptorCount = 1;
    b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[0].pImmutableSamplers = &sampler;
    b[1].binding = 1;
    b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[1].descriptorCount = 1;
    b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[2] = b[1];
    b[2].binding = 2;
    memset(&li, 0, sizeof(li));
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 3;
    li.pBindings = b;
    memset(&pcr, 0, sizeof(pcr));
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size = sizeof(push);
    memset(&smi, 0, sizeof(smi));
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = sizeof(spv_sample_comp);
    smi.pCode = spv_sample_comp;
    if (!cbuf.data || !rbuf.data || vkCreateDescriptorSetLayout(device, &li, NULL, &set_layout) != VK_SUCCESS)
        goto out;
    memset(&pli, 0, sizeof(pli));
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device, &pli, NULL, &layout) != VK_SUCCESS ||
        vkCreateShaderModule(device, &smi, NULL, &module) != VK_SUCCESS)
        goto out;
    memset(&cpi, 0, sizeof(cpi));
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = module;
    cpi.stage.pName = "main";
    cpi.layout = layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, NULL, &pipeline) != VK_SUCCESS) {
        PVK_TEST_FAIL("vkCreateComputePipelines failed");
        goto out;
    }
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = descriptors;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[1].descriptorCount = 2;
    memset(&dpi, 0, sizeof(dpi));
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(device, &dpi, NULL, &dpool) != VK_SUCCESS)
        goto out;
    memset(&dai, 0, sizeof(dai));
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &set_layout;
    if (vkAllocateDescriptorSets(device, &dai, &set) != VK_SUCCESS)
        goto out;
    memset(&ii, 0, sizeof(ii));
    ii.imageView = view;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bi[0].buffer = cbuf.buffer;
    bi[0].offset = 0;
    bi[0].range = VK_WHOLE_SIZE;
    bi[1] = bi[0];
    bi[1].buffer = rbuf.buffer;
    memset(w, 0, sizeof(w));
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = set;
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[0].pImageInfo = &ii;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[1].dstSet = set;
    w[1].dstBinding = 1;
    w[1].descriptorCount = 2;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].pBufferInfo = bi;
    vkUpdateDescriptorSets(device, 2, w, 0, NULL);
    memcpy(cbuf.data, coords, NCOORDS * 16);
    cb = begin();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    vkCmdDispatch(cb, NCOORDS / 64, 1, 1);
    barrier(cb);
    ok = finish(cb) == VK_SUCCESS;
    memcpy(results, rbuf.data, NCOORDS * 16);
out:
    vkDestroyDescriptorPool(device, dpool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, module, NULL);
    vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroyDescriptorSetLayout(device, set_layout, NULL);
    free_buf(&cbuf);
    free_buf(&rbuf);
    return ok;
}

/* Combined image sampler descriptors a format takes. */
static uint32_t descriptor_count(VkFormat format, VkImageTiling tiling, VkImageCreateFlags flags)
{
    VkPhysicalDeviceImageFormatInfo2 info;
    VkSamplerYcbcrConversionImageFormatProperties y;
    VkImageFormatProperties2 props;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
    info.format = format;
    info.type = VK_IMAGE_TYPE_2D;
    info.tiling = tiling;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.flags = flags;
    memset(&y, 0, sizeof(y));
    y.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES;
    memset(&props, 0, sizeof(props));
    props.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
    props.pNext = &y;
    if (vkGetPhysicalDeviceImageFormatProperties2(pd, &info, &props) != VK_SUCCESS)
        return 0;
    return y.combinedImageSamplerDescriptorCount;
}

/* Compares results with the reference; prints the first mismatches. */
static int compare(const char *name, const struct ref *r, const struct conv *c, const float *coords, const float *results,
                   uint32_t n, uint32_t mode)
{
    uint32_t i, k, bad = 0, compared = 0;
    for (i = 0; i < n; i++) {
        const float *co = &coords[4 * i], *got = &results[4 * i];
        float want[4], lod = mode == 0 ? co[2] : log2f(fmaxf(co[2] * IMG_W, co[3] * IMG_H));
        bool ambiguous = false, linear = false;
        ref_sample(r, c, co[0], co[1], lod, want, &ambiguous, &linear);
        if (ambiguous)
            continue;
        compared++;
        for (k = 0; k < 4; k++) {
            if (fabsf(got[k] - want[k]) > (linear ? TOL_LINEAR : TOL_NEAREST) || got[k] != got[k]) {
                if (bad++ < 4)
                    fprintf(stderr, "%s: sample %u at (%.4f, %.4f) lod %.2f: got (%.4f %.4f %.4f %.4f), want (%.4f %.4f %.4f "
                                    "%.4f)\n",
                            name, i, co[0], co[1], lod, got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                break;
            }
        }
    }
    if (compared < n / 2) {
        fprintf(stderr, "%s: only %u of %u samples compared\n", name, compared, n);
        bad++;
    }
    return (int)bad;
}

static void run_case(const struct sample_case *sc)
{
    const struct yfmt *f = &yfmts[sc->format];
    struct yimg m;
    struct planes pl;
    struct ref r;
    VkSamplerYcbcrConversion conv = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    static float coords[NCOORDS * 4], results[NCOORDS * 4];
    uint32_t descriptors = descriptor_count(f->format, sc->tiling, sc->flags);

    planes_init(&pl, f, IMG_W, IMG_H, sc->format + 7u * (uint32_t)sc->conv.model);
    if (!pl.data || !descriptors) {
        PVK_TEST_FAIL("%s: setup", sc->name);
        free(pl.data);
        return;
    }
    if (!make_yimg(&m, f, IMG_W, IMG_H, sc->tiling, sc->flags,
                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        PVK_TEST_FAIL("%s: image", sc->name);
        goto out;
    }
    if (sc->tiling == VK_IMAGE_TILING_LINEAR ? !write_linear(&m, &pl) : !upload(&m, &pl)) {
        PVK_TEST_FAIL("%s: upload", sc->name);
        goto out;
    }
    conv = make_conversion(f, &sc->conv);
    sampler = make_sampler(conv, sc->conv.min_filter, sc->conv.mag_filter);
    view = make_view(m.image, f->format, VK_IMAGE_ASPECT_COLOR_BIT, conv);
    if (!conv || !sampler || !view) {
        PVK_TEST_FAIL("%s: conversion, sampler or view", sc->name);
        goto out;
    }
    make_coords(coords, NCOORDS, sc->lod, sc->mode);
    memset(results, 0, sizeof(results));
    if (!run_compute(view, sampler, descriptors, coords, sc->mode, results)) {
        PVK_TEST_FAIL("%s: dispatch", sc->name);
        goto out;
    }
    r.f = f;
    r.pl = &pl;
    r.w = IMG_W;
    r.h = IMG_H;
    if (compare(sc->name, &r, &sc->conv, coords, results, NCOORDS, sc->mode))
        PVK_TEST_FAIL("%s: wrong samples", sc->name);
out:
    vkDestroyImageView(device, view, NULL);
    vkDestroySampler(device, sampler, NULL);
    vkDestroySamplerYcbcrConversion(device, conv, NULL);
    free_yimg(&m);
    free(pl.data);
}

static struct conv conv_default(void)
{
    struct conv c;
    memset(&c, 0, sizeof(c));
    c.model = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
    c.range = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
    c.components = identity_mapping();
    c.x_chroma = c.y_chroma = VK_CHROMA_LOCATION_COSITED_EVEN;
    c.chroma_filter = VK_FILTER_LINEAR;
    c.min_filter = c.mag_filter = VK_FILTER_LINEAR;
    return c;
}

static struct sample_case scase(const char *name, uint32_t format, struct conv c)
{
    struct sample_case s;
    memset(&s, 0, sizeof(s));
    s.name = name;
    s.format = format;
    s.conv = c;
    s.tiling = VK_IMAGE_TILING_OPTIMAL;
    s.lod = -1.0f;
    return s;
}

/* Every format, implicit and explicit reconstruction. */
static void test_sample_formats(void)
{
    uint32_t i, e;
    char name[96];
    for (i = 0; i < PVK_ARRAY_SIZE(yfmts); i++) {
        for (e = 0; e < 2; e++) {
            struct conv c = conv_default();
            struct sample_case s;
            c.force_explicit = e;
            snprintf(name, sizeof(name), "%s %s", yfmts[i].name, e ? "explicit" : "implicit");
            s = scase(name, i, c);
            run_case(&s);
        }
    }
}

/* Every model and range. */
static void test_sample_models(void)
{
    static const VkSamplerYcbcrModelConversion models[] = {
        VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY, VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY,
        VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709, VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601,
        VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020,
    };
    uint32_t m, r;
    char name[96];
    for (m = 0; m < PVK_ARRAY_SIZE(models); m++) {
        for (r = 0; r < 2; r++) {
            struct conv c = conv_default();
            struct sample_case s;
            c.model = models[m];
            c.range = r ? VK_SAMPLER_YCBCR_RANGE_ITU_NARROW : VK_SAMPLER_YCBCR_RANGE_ITU_FULL;
            snprintf(name, sizeof(name), "3-plane 4:2:0 model %d range %u", (int)models[m], r);
            s = scase(name, FMT_3PLANE_420, c);
            run_case(&s);
        }
    }
}

/* Every chroma location and chroma filter, implicit and explicit, with
 * linear and nearest luma filters, on 4:2:0 and 4:2:2. */
static void test_sample_chroma(void)
{
    static const uint32_t formats[] = { FMT_2PLANE_420, FMT_3PLANE_422, FMT_PACKED_BGRG };
    uint32_t fi, x, y, cf, e, lf;
    char name[128];
    for (fi = 0; fi < PVK_ARRAY_SIZE(formats); fi++)
        for (x = 0; x < 2; x++)
            for (y = 0; y < 2; y++)
                for (cf = 0; cf < 2; cf++)
                    for (e = 0; e < 2; e++)
                        for (lf = 0; lf < 2; lf++) {
                            struct conv c = conv_default();
                            struct sample_case s;
                            if ((!yfmts[formats[fi]].ys && y) || (lf && (x || y)))
                                continue; /* 4:2:2: no y location; nearest luma once */
                            c.x_chroma = x ? VK_CHROMA_LOCATION_MIDPOINT : VK_CHROMA_LOCATION_COSITED_EVEN;
                            c.y_chroma = y ? VK_CHROMA_LOCATION_MIDPOINT : VK_CHROMA_LOCATION_COSITED_EVEN;
                            c.chroma_filter = cf ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
                            c.force_explicit = e;
                            c.min_filter = c.mag_filter = lf ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
                            snprintf(name, sizeof(name), "%s x %s y %s chroma %s %s luma %s", yfmts[formats[fi]].name,
                                     x ? "mid" : "cosited", y ? "mid" : "cosited", cf ? "nearest" : "linear",
                                     e ? "explicit" : "implicit", lf ? "nearest" : "linear");
                            s = scase(name, formats[fi], c);
                            run_case(&s);
                        }
}

/* Separate min and mag filters: the LOD picks one (textureLod and
 * textureGrad), implicit and explicit reconstruction. */
static void test_sample_min_mag(void)
{
    static const uint32_t formats[] = { FMT_3PLANE_420, FMT_PACKED_GBGR };
    uint32_t fi, e, mode, l;
    char name[128];
    for (fi = 0; fi < PVK_ARRAY_SIZE(formats); fi++)
        for (e = 0; e < 2; e++)
            for (mode = 0; mode < 2; mode++)
                for (l = 0; l < 2; l++) {
                    struct conv c = conv_default();
                    struct sample_case s;
                    c.force_explicit = e;
                    c.min_filter = VK_FILTER_LINEAR;
                    c.mag_filter = VK_FILTER_NEAREST;
                    snprintf(name, sizeof(name), "%s min linear mag nearest %s %s lod %s", yfmts[formats[fi]].name,
                             e ? "explicit" : "implicit", mode ? "grad" : "lod", l ? "+1" : "-1");
                    s = scase(name, formats[fi], c);
                    s.mode = mode;
                    s.lod = l ? 1.0f : -1.0f;
                    run_case(&s);
                }
}

/* Component mappings, linear tiling and disjoint planes. */
static void test_sample_mapping_tiling(void)
{
    struct conv c = conv_default();
    struct sample_case s;

    c.model = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY;
    c.components.r = VK_COMPONENT_SWIZZLE_B;
    c.components.b = VK_COMPONENT_SWIZZLE_R;
    c.components.a = VK_COMPONENT_SWIZZLE_ZERO;
    s = scase("3-plane 4:2:0 swapped chroma, alpha zero", FMT_3PLANE_420, c);
    run_case(&s);
    c = conv_default();
    c.components.a = VK_COMPONENT_SWIZZLE_ONE;
    c.components.r = VK_COMPONENT_SWIZZLE_B;
    c.components.b = VK_COMPONENT_SWIZZLE_R;
    s = scase("2-plane 4:2:2 swapped chroma, alpha one", FMT_2PLANE_422, c);
    run_case(&s);

    s = scase("3-plane 4:2:0 linear tiling", FMT_3PLANE_420, conv_default());
    s.tiling = VK_IMAGE_TILING_LINEAR;
    run_case(&s);
    s = scase("2-plane 4:2:0 linear tiling, disjoint", FMT_2PLANE_420, conv_default());
    s.tiling = VK_IMAGE_TILING_LINEAR;
    s.flags = VK_IMAGE_CREATE_DISJOINT_BIT;
    run_case(&s);
    s = scase("packed 4:2:2 linear tiling", FMT_PACKED_GBGR, conv_default());
    s.tiling = VK_IMAGE_TILING_LINEAR;
    run_case(&s);
    s = scase("3-plane 4:4:4 disjoint", FMT_3PLANE_444, conv_default());
    s.flags = VK_IMAGE_CREATE_DISJOINT_BIT;
    run_case(&s);
}

/* ---- Formats, images, copies ---- */

static void test_format_properties(void)
{
    uint32_t i;
    for (i = 0; i < PVK_ARRAY_SIZE(yfmts); i++) {
        const struct yfmt *f = &yfmts[i];
        VkFormatProperties3 p3;
        VkFormatProperties2 p2;
        VkFormatFeatureFlags2 need = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
                                     VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_2_MIDPOINT_CHROMA_SAMPLES_BIT |
                                     VK_FORMAT_FEATURE_2_COSITED_CHROMA_SAMPLES_BIT |
                                     VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT |
                                     VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT;
        VkPhysicalDeviceImageFormatInfo2 info;
        VkImageFormatProperties2 props;
        memset(&p3, 0, sizeof(p3));
        p3.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3;
        memset(&p2, 0, sizeof(p2));
        p2.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
        p2.pNext = &p3;
        vkGetPhysicalDeviceFormatProperties2(pd, f->format, &p2);
        CHECK((p3.optimalTilingFeatures & need) == need);
        CHECK((p3.linearTilingFeatures & need) == need);
        CHECK(p3.bufferFeatures == 0);
        CHECK(!(p3.optimalTilingFeatures & (VK_FORMAT_FEATURE_2_BLIT_SRC_BIT | VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT |
                                            VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT)));
        CHECK(!!(p3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_DISJOINT_BIT) == (f->planes > 1));
        /* implicit, explicit not offered (the CTS reference differs from the spec) */
        CHECK(!(p3.optimalTilingFeatures &
                (VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT |
                 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT)));
        CHECK(descriptor_count(f->format, VK_IMAGE_TILING_OPTIMAL, 0) == f->planes);

        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
        info.format = f->format;
        info.type = VK_IMAGE_TYPE_2D;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        memset(&props, 0, sizeof(props));
        props.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
        CHECK_VK(vkGetPhysicalDeviceImageFormatProperties2(pd, &info, &props), VK_SUCCESS);
        CHECK(props.imageFormatProperties.maxMipLevels == 1 && props.imageFormatProperties.maxArrayLayers == 1 &&
              props.imageFormatProperties.sampleCounts == VK_SAMPLE_COUNT_1_BIT &&
              props.imageFormatProperties.maxExtent.depth == 1);
        info.flags = VK_IMAGE_CREATE_DISJOINT_BIT;
        CHECK_VK(vkGetPhysicalDeviceImageFormatProperties2(pd, &info, &props),
                 f->planes > 1 ? VK_SUCCESS : VK_ERROR_FORMAT_NOT_SUPPORTED);
        info.flags = 0;
        info.type = VK_IMAGE_TYPE_3D;
        CHECK_VK(vkGetPhysicalDeviceImageFormatProperties2(pd, &info, &props), VK_ERROR_FORMAT_NOT_SUPPORTED);
        info.type = VK_IMAGE_TYPE_2D;
        info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        CHECK_VK(vkGetPhysicalDeviceImageFormatProperties2(pd, &info, &props), VK_ERROR_FORMAT_NOT_SUPPORTED);
    }
    /* Plane formats are not Y'CbCr formats */
    CHECK(descriptor_count(VK_FORMAT_R8_UNORM, VK_IMAGE_TILING_OPTIMAL, 0) == 1);
}

/* Plane copies both ways for every format, optimal and linear, plain and
 * disjoint; then image <-> image plane copies. */
static void test_plane_copies(void)
{
    uint32_t i, t, d;
    for (i = 0; i < PVK_ARRAY_SIZE(yfmts); i++)
        for (t = 0; t < 2; t++)
            for (d = 0; d < 2; d++) {
                const struct yfmt *f = &yfmts[i];
                struct yimg m;
                struct planes pl;
                if (d && f->planes == 1)
                    continue;
                planes_init(&pl, f, 32, 16, i * 4 + t * 2 + d);
                if (!make_yimg(&m, f, 32, 16, t ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL,
                               d ? VK_IMAGE_CREATE_DISJOINT_BIT : 0,
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
                    PVK_TEST_FAIL("%s: image (tiling %u, disjoint %u)", f->name, t, d);
                } else {
                    CHECK(upload(&m, &pl));
                    if (!readback_matches(&m, &pl))
                        PVK_TEST_FAIL("%s: planes read back differ (tiling %u, disjoint %u)", f->name, t, d);
                }
                free_yimg(&m);
                free(pl.data);
            }
}

/* Copies plane 1 of a 3-plane image to an R8 image and back into plane 2
 * of another; plane 1 (B8R8) of a 2-plane image to an R8G8 image; a whole
 * plane between two multi-planar images. */
static void test_image_plane_copies(void)
{
    const struct yfmt *f3 = &yfmts[FMT_3PLANE_420], *f2 = &yfmts[FMT_2PLANE_420];
    struct yimg a, b, c;
    struct planes pa, pb;
    VkImageCreateInfo ri;
    VkImage r8 = VK_NULL_HANDLE, rg8 = VK_NULL_HANDLE;
    VkDeviceMemory mem[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkMemoryRequirements req;
    VkMemoryAllocateInfo alloc;
    VkImageCopy cp;
    VkCommandBuffer cb;
    struct buf out = make_buf(4096, 0);
    uint32_t k;

    planes_init(&pa, f3, 32, 16, 1);
    planes_init(&pb, f2, 32, 16, 2);
    REQUIRE(out.data);
    REQUIRE(make_yimg(&a, f3, 32, 16, VK_IMAGE_TILING_OPTIMAL, 0,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
    REQUIRE(make_yimg(&b, f3, 32, 16, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_CREATE_DISJOINT_BIT,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
    REQUIRE(make_yimg(&c, f2, 32, 16, VK_IMAGE_TILING_OPTIMAL, 0,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
    REQUIRE(upload(&a, &pa));
    REQUIRE(upload(&c, &pb));
    memset(&ri, 0, sizeof(ri));
    ri.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ri.imageType = VK_IMAGE_TYPE_2D;
    ri.format = VK_FORMAT_R8_UNORM;
    ri.extent.width = 16;
    ri.extent.height = 8;
    ri.extent.depth = 1;
    ri.mipLevels = ri.arrayLayers = 1;
    ri.samples = VK_SAMPLE_COUNT_1_BIT;
    ri.tiling = VK_IMAGE_TILING_OPTIMAL;
    ri.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    REQUIRE_VK(vkCreateImage(device, &ri, NULL, &r8));
    ri.format = VK_FORMAT_R8G8_UNORM;
    REQUIRE_VK(vkCreateImage(device, &ri, NULL, &rg8));
    for (k = 0; k < 2; k++) {
        vkGetImageMemoryRequirements(device, k ? rg8 : r8, &req);
        memset(&alloc, 0, sizeof(alloc));
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = first_type(req.memoryTypeBits, private_type);
        REQUIRE_VK(vkAllocateMemory(device, &alloc, NULL, &mem[k]));
        REQUIRE_VK(vkBindImageMemory(device, k ? rg8 : r8, mem[k], 0));
    }

    cb = begin();
    memset(&cp, 0, sizeof(cp));
    cp.srcSubresource.layerCount = cp.dstSubresource.layerCount = 1;
    cp.extent.width = 16;
    cp.extent.height = 8;
    cp.extent.depth = 1;
    /* a.plane1 -> r8 -> b.plane2; c.plane1 -> rg8; a.plane0 -> b.plane0 */
    cp.srcSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
    cp.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkCmdCopyImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r8, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    vkCmdCopyImage(cb, c.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rg8, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    barrier(cb);
    cp.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    cp.dstSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_2_BIT;
    vkCmdCopyImage(cb, r8, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    cp.srcSubresource.aspectMask = cp.dstSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
    cp.extent.width = 32;
    cp.extent.height = 16;
    vkCmdCopyImage(cb, a.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    barrier(cb);
    {
        VkBufferImageCopy r[3];
        memset(r, 0, sizeof(r));
        for (k = 0; k < 3; k++) {
            r[k].imageSubresource.layerCount = 1;
            r[k].imageExtent.depth = 1;
        }
        r[0].imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_2_BIT; /* 16 x 8 at 0 */
        r[0].imageExtent.width = 16;
        r[0].imageExtent.height = 8;
        vkCmdCopyImageToBuffer(cb, b.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.buffer, 1, &r[0]);
        r[1].bufferOffset = 1024; /* rg8: 16 x 8 x 2 */
        r[1].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        r[1].imageExtent.width = 16;
        r[1].imageExtent.height = 8;
        vkCmdCopyImageToBuffer(cb, rg8, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.buffer, 1, &r[1]);
        r[2].bufferOffset = 2048; /* b.plane0: 32 x 16 */
        r[2].imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
        r[2].imageExtent.width = 32;
        r[2].imageExtent.height = 16;
        vkCmdCopyImageToBuffer(cb, b.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.buffer, 1, &r[2]);
    }
    barrier(cb);
    CHECK_VK(finish(cb), VK_SUCCESS);
    CHECK(memcmp(out.data, pa.data + pa.offsets[1], 16 * 8) == 0);
    CHECK(memcmp(out.data + 1024, pb.data + pb.offsets[1], 16 * 8 * 2) == 0);
    CHECK(memcmp(out.data + 2048, pa.data, 32 * 16) == 0);

    vkDestroyImage(device, r8, NULL);
    vkDestroyImage(device, rg8, NULL);
    vkFreeMemory(device, mem[0], NULL);
    vkFreeMemory(device, mem[1], NULL);
    free_yimg(&a);
    free_yimg(&b);
    free_yimg(&c);
    free_buf(&out);
    free(pa.data);
    free(pb.data);
}

/* Per-plane memory requirements of disjoint images, and the combined
 * requirements of plain ones; linear layouts of the planes. */
static void test_requirements_layouts(void)
{
    uint32_t i, p;
    for (i = 0; i < PVK_ARRAY_SIZE(yfmts); i++) {
        const struct yfmt *f = &yfmts[i];
        VkImageCreateInfo info;
        VkImage image = VK_NULL_HANDLE;
        VkMemoryRequirements whole;
        VkDeviceSize end = 0;
        if (f->planes == 1)
            continue;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = f->format;
        info.extent.width = 64;
        info.extent.height = 32;
        info.extent.depth = 1;
        info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_LINEAR;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        REQUIRE_VK(vkCreateImage(device, &info, NULL, &image));
        vkGetImageMemoryRequirements(device, image, &whole);
        for (p = 0; p < f->planes; p++) {
            VkImageSubresource sub;
            VkSubresourceLayout l;
            uint32_t pw, ph;
            plane_extent(f, p, 64, 32, &pw, &ph);
            sub.aspectMask = plane_aspects[p];
            sub.mipLevel = sub.arrayLayer = 0;
            vkGetImageSubresourceLayout(device, image, &sub, &l);
            CHECK(l.offset >= end);
            CHECK(l.rowPitch >= (VkDeviceSize)pw * f->plane_bytes[p]);
            CHECK(l.size >= l.rowPitch * (ph - 1) + (VkDeviceSize)pw * f->plane_bytes[p]);
            end = l.offset + l.size;
        }
        CHECK(end <= whole.size);
        vkDestroyImage(device, image, NULL);

        info.flags = VK_IMAGE_CREATE_DISJOINT_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        REQUIRE_VK(vkCreateImage(device, &info, NULL, &image));
        for (p = 0; p < f->planes; p++) {
            VkImagePlaneMemoryRequirementsInfo pi;
            VkImageMemoryRequirementsInfo2 ri;
            VkMemoryDedicatedRequirements ded;
            VkMemoryRequirements2 req;
            VkDeviceImageMemoryRequirements di;
            VkMemoryRequirements2 dreq;
            memset(&pi, 0, sizeof(pi));
            pi.sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO;
            pi.planeAspect = plane_aspects[p];
            memset(&ri, 0, sizeof(ri));
            ri.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
            ri.pNext = &pi;
            ri.image = image;
            memset(&ded, 0, sizeof(ded));
            ded.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
            memset(&req, 0, sizeof(req));
            req.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
            req.pNext = &ded;
            vkGetImageMemoryRequirements2(device, &ri, &req);
            CHECK(req.memoryRequirements.size > 0 && req.memoryRequirements.alignment > 0 &&
                  req.memoryRequirements.memoryTypeBits != 0);
            memset(&di, 0, sizeof(di));
            di.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS;
            di.pCreateInfo = &info;
            di.planeAspect = plane_aspects[p];
            memset(&dreq, 0, sizeof(dreq));
            dreq.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
            vkGetDeviceImageMemoryRequirements(device, &di, &dreq);
            CHECK(dreq.memoryRequirements.size == req.memoryRequirements.size &&
                  dreq.memoryRequirements.alignment == req.memoryRequirements.alignment);
        }
        vkDestroyImage(device, image, NULL);
    }
}

/* Host image copies of planes (VK_IMAGE_USAGE_HOST_TRANSFER_BIT). */
static void test_host_copies(void)
{
    static const uint32_t formats[] = { FMT_3PLANE_420, FMT_2PLANE_422, FMT_PACKED_BGRG };
    uint32_t fi, p;
    for (fi = 0; fi < PVK_ARRAY_SIZE(formats); fi++) {
        const struct yfmt *f = &yfmts[formats[fi]];
        struct yimg m;
        struct planes pl;
        unsigned char *back;
        planes_init(&pl, f, 16, 8, 40 + fi);
        back = calloc(1, pl.total);
        if (!make_yimg(&m, f, 16, 8, VK_IMAGE_TILING_OPTIMAL, 0,
                       VK_IMAGE_USAGE_HOST_TRANSFER_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
            PVK_TEST_FAIL("%s: host transfer image", f->name);
        } else {
            for (p = 0; p < f->planes; p++) {
                VkMemoryToImageCopy to;
                VkImageToMemoryCopy from;
                VkCopyMemoryToImageInfo ti;
                VkCopyImageToMemoryInfo fi2;
                uint32_t pw, ph;
                plane_extent(f, p, 16, 8, &pw, &ph);
                memset(&to, 0, sizeof(to));
                to.sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY;
                to.pHostPointer = pl.data + pl.offsets[p];
                to.imageSubresource.aspectMask = plane_aspect(f, p);
                to.imageSubresource.layerCount = 1;
                to.imageExtent.width = pw;
                to.imageExtent.height = ph;
                to.imageExtent.depth = 1;
                memset(&ti, 0, sizeof(ti));
                ti.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO;
                ti.dstImage = m.image;
                ti.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
                ti.regionCount = 1;
                ti.pRegions = &to;
                CHECK_VK(vkCopyMemoryToImage(device, &ti), VK_SUCCESS);
                memset(&from, 0, sizeof(from));
                from.sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY;
                from.pHostPointer = back + pl.offsets[p];
                from.imageSubresource = to.imageSubresource;
                from.imageExtent = to.imageExtent;
                memset(&fi2, 0, sizeof(fi2));
                fi2.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO;
                fi2.srcImage = m.image;
                fi2.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
                fi2.regionCount = 1;
                fi2.pRegions = &from;
                CHECK_VK(vkCopyImageToMemory(device, &fi2), VK_SUCCESS);
            }
            CHECK(memcmp(back, pl.data, pl.total) == 0);
            CHECK(readback_matches(&m, &pl)); /* the GPU sees the same planes */
        }
        free_yimg(&m);
        free(back);
        free(pl.data);
    }
}

/* Plane views (MUTABLE_FORMAT): plane 1 of a 3-plane image viewed as
 * R8_UNORM and sampled without a conversion reads the plane's texels. */
static void test_plane_views(void)
{
    const struct yfmt *f = &yfmts[FMT_3PLANE_420];
    struct yimg m;
    struct planes pl;
    VkSampler sampler;
    VkImageView view;
    static float coords[NCOORDS * 4], results[NCOORDS * 4];
    uint32_t i;

    planes_init(&pl, f, IMG_W * 2, IMG_H * 2, 77); /* plane 1: IMG_W x IMG_H */
    REQUIRE(pl.data);
    REQUIRE(make_yimg(&m, f, IMG_W * 2, IMG_H * 2, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
    REQUIRE(upload(&m, &pl));
    sampler = make_sampler(VK_NULL_HANDLE, VK_FILTER_NEAREST, VK_FILTER_NEAREST);
    view = make_view(m.image, VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_PLANE_1_BIT, VK_NULL_HANDLE);
    REQUIRE(sampler && view);
    for (i = 0; i < NCOORDS; i++) {
        coords[4 * i] = ((i % IMG_W) + 0.5f) / IMG_W;
        coords[4 * i + 1] = (((i / IMG_W) % IMG_H) + 0.5f) / IMG_H;
        coords[4 * i + 2] = coords[4 * i + 3] = 0.0f;
    }
    CHECK(run_compute(view, sampler, 1, coords, 0, results));
    for (i = 0; i < NCOORDS; i++) {
        float want = pl.data[pl.offsets[1] + ((i / IMG_W) % IMG_H) * IMG_W + i % IMG_W] / 255.0f;
        if (fabsf(results[4 * i] - want) > TOL_NEAREST) {
            PVK_TEST_FAIL("plane view texel %u: got %f, want %f", i, results[4 * i], want);
            break;
        }
    }
    vkDestroyImageView(device, view, NULL);
    vkDestroySampler(device, sampler, NULL);
    free_yimg(&m);
    free(pl.data);
}

/* Descriptor writes, copies and update templates put the planes of a view
 * with a conversion in the 32-byte descriptor (section 11), and keep the
 * immutable sampler. Checked in the set tables. */
static void test_descriptor_updates(void)
{
    const struct yfmt *f = &yfmts[FMT_3PLANE_420];
    struct conv c = conv_default();
    struct yimg m;
    VkSamplerYcbcrConversion conv;
    VkSampler sampler;
    VkImageView view;
    VkDescriptorSetLayoutBinding b;
    VkDescriptorSetLayoutCreateInfo li;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorPoolSize size;
    VkDescriptorPoolCreateInfo dpi;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSetLayout layouts[3];
    VkDescriptorSetAllocateInfo dai;
    VkDescriptorSet sets[3];
    VkDescriptorImageInfo ii;
    VkWriteDescriptorSet w;
    VkCopyDescriptorSet cp;
    VkDescriptorUpdateTemplateEntry entry;
    VkDescriptorUpdateTemplateCreateInfo tci;
    VkDescriptorUpdateTemplate tmpl = VK_NULL_HANDLE;
    pvk_MTLResourceID want[4];
    uint32_t s;

    REQUIRE(make_yimg(&m, f, 16, 8, VK_IMAGE_TILING_OPTIMAL, 0, VK_IMAGE_USAGE_SAMPLED_BIT));
    conv = make_conversion(f, &c);
    sampler = make_sampler(conv, VK_FILTER_LINEAR, VK_FILTER_LINEAR);
    view = make_view(m.image, f->format, VK_IMAGE_ASPECT_COLOR_BIT, conv);
    REQUIRE(conv && sampler && view);
    memset(&b, 0, sizeof(b));
    b.binding = 3;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b.pImmutableSamplers = &sampler;
    memset(&li, 0, sizeof(li));
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 1;
    li.pBindings = &b;
    REQUIRE_VK(vkCreateDescriptorSetLayout(device, &li, NULL, &layout));
    size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    size.descriptorCount = 3 * f->planes;
    memset(&dpi, 0, sizeof(dpi));
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 3;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &size;
    REQUIRE_VK(vkCreateDescriptorPool(device, &dpi, NULL, &dpool));
    layouts[0] = layouts[1] = layouts[2] = layout;
    memset(&dai, 0, sizeof(dai));
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = dpool;
    dai.descriptorSetCount = 3;
    dai.pSetLayouts = layouts;
    REQUIRE_VK(vkAllocateDescriptorSets(device, &dai, sets));
    memset(&ii, 0, sizeof(ii));
    ii.imageView = view;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    memset(&w, 0, sizeof(w));
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = sets[0];
    w.dstBinding = 3;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    memset(&cp, 0, sizeof(cp));
    cp.sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
    cp.srcSet = sets[0];
    cp.srcBinding = 3;
    cp.dstSet = sets[1];
    cp.dstBinding = 3;
    cp.descriptorCount = 1;
    vkUpdateDescriptorSets(device, 1, &w, 0, NULL);
    vkUpdateDescriptorSets(device, 0, NULL, 1, &cp);
    entry.dstBinding = 3;
    entry.dstArrayElement = 0;
    entry.descriptorCount = 1;
    entry.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    entry.offset = 0;
    entry.stride = sizeof(ii);
    memset(&tci, 0, sizeof(tci));
    tci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
    tci.descriptorUpdateEntryCount = 1;
    tci.pDescriptorUpdateEntries = &entry;
    tci.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
    tci.descriptorSetLayout = layout;
    REQUIRE_VK(vkCreateDescriptorUpdateTemplate(device, &tci, NULL, &tmpl));
    vkUpdateDescriptorSetWithTemplate(device, sets[2], tmpl, &ii);
    want[0] = pvk_MTLTexture_gpuResourceID(((struct pvk_image_view *)view)->mtl_texture);
    want[1] = pvk_MTLSamplerState_gpuResourceID(((struct pvk_sampler *)sampler)->mtl_sampler);
    want[2] = pvk_MTLTexture_gpuResourceID(((struct pvk_image_view *)view)->planes[0]);
    want[3] = pvk_MTLTexture_gpuResourceID(((struct pvk_image_view *)view)->planes[1]);
    for (s = 0; s < 3; s++) {
        const struct pvk_descriptor_set *set = (const struct pvk_descriptor_set *)sets[s];
        if (memcmp(set->contents + set->layout->bindings[0].offset, want, sizeof(want)) != 0)
            PVK_TEST_FAIL("set %u (%s): the descriptor is not { plane 0, sampler, plane 1, plane 2 }", s,
                          s == 0 ? "write" : s == 1 ? "copy" : "template");
    }
    vkDestroyDescriptorUpdateTemplate(device, tmpl, NULL);
    vkDestroyDescriptorPool(device, dpool, NULL);
    vkDestroyDescriptorSetLayout(device, layout, NULL);
    vkDestroyImageView(device, view, NULL);
    vkDestroySampler(device, sampler, NULL);
    vkDestroySamplerYcbcrConversion(device, conv, NULL);
    free_yimg(&m);
}

/* Set layouts: combined image samplers with multi-planar conversions take
 * 32 bytes, packed 4:2:2 ones 16. */
static void test_descriptor_layouts(void)
{
    static const uint32_t formats[] = { FMT_3PLANE_420, FMT_2PLANE_444, FMT_PACKED_GBGR };
    uint32_t fi;
    for (fi = 0; fi < PVK_ARRAY_SIZE(formats); fi++) {
        const struct yfmt *f = &yfmts[formats[fi]];
        struct conv c = conv_default();
        VkSamplerYcbcrConversion conv = make_conversion(f, &c);
        VkSampler s = make_sampler(conv, VK_FILTER_LINEAR, VK_FILTER_LINEAR);
        VkSampler two[2];
        VkDescriptorSetLayoutBinding b[2];
        VkDescriptorSetLayoutCreateInfo li;
        VkDescriptorSetLayoutSupport support;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        two[0] = two[1] = s;
        memset(b, 0, sizeof(b));
        b[0].binding = 0;
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[0].descriptorCount = 2;
        b[0].stageFlags = VK_SHADER_STAGE_ALL;
        b[0].pImmutableSamplers = two;
        b[1].binding = 1;
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        b[1].descriptorCount = 1;
        b[1].stageFlags = VK_SHADER_STAGE_ALL;
        memset(&li, 0, sizeof(li));
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 2;
        li.pBindings = b;
        memset(&support, 0, sizeof(support));
        support.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT;
        vkGetDescriptorSetLayoutSupport(device, &li, &support);
        CHECK(support.supported);
        CHECK_VK(vkCreateDescriptorSetLayout(device, &li, NULL, &layout), VK_SUCCESS);
        if (layout) {
            const struct pvk_descriptor_set_layout *l = (const struct pvk_descriptor_set_layout *)layout;
            CHECK(l->bindings[0].stride == (f->planes > 1 ? 32u : 16u));
            CHECK(l->bindings[1].offset == (f->planes > 1 ? 64u : 32u));
            CHECK(l->bindings[0].ycbcr && l->bindings[0].ycbcr[1].planes == f->planes);
        }
        vkDestroyDescriptorSetLayout(device, layout, NULL);
        vkDestroySampler(device, s, NULL);
        vkDestroySamplerYcbcrConversion(device, conv, NULL);
    }
    vkDestroySamplerYcbcrConversion(device, VK_NULL_HANDLE, NULL);
}

/* ---- Sampling in fragment shaders (translator + Metal render pipelines) ---- */

/* A Metal sampler state as section 11 makes a VkSampler with a conversion. */
static id frag_sampler(VkFilter min_filter, VkFilter mag_filter)
{
    id desc = PVK_MSG(id, (id, SEL))((id)objc_getClass("MTLSamplerDescriptor"), sel_registerName("new")), smp;
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setMinFilter:"), min_filter == VK_FILTER_LINEAR);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setMagFilter:"), mag_filter == VK_FILTER_LINEAR);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setMipFilter:"), PVK_MTLSamplerMipFilterNearest);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setSAddressMode:"), PVK_MTLSamplerAddressModeClampToEdge);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setTAddressMode:"), PVK_MTLSamplerAddressModeClampToEdge);
    PVK_MSG(void, (id, SEL, float))(desc, sel_registerName("setLodMaxClamp:"), 1000.0f);
    PVK_MSG(void, (id, SEL, bool))(desc, sel_registerName("setSupportArgumentBuffers:"), true);
    smp = PVK_MSG(id, (id, SEL, id))(mt_device, sel_registerName("newSamplerStateWithDescriptor:"), desc);
    pvk_release(desc);
    return smp;
}

/* The struct pvk_ycbcr section 11 makes from a conversion and a sampler. */
static struct pvk_ycbcr frag_ycbcr(const struct yfmt *f, const struct conv *c)
{
    struct pvk_ycbcr y;
    const VkComponentSwizzle comps[4] = { c->components.r, c->components.g, c->components.b, c->components.a };
    unsigned i;
    memset(&y, 0, sizeof(y));
    y.planes = (uint8_t)f->planes;
    y.packed = (uint8_t)f->packed;
    y.x_shift = (uint8_t)f->xs;
    y.y_shift = (uint8_t)f->ys;
    y.bits = 8;
    y.model = (uint8_t)c->model;
    y.range = (uint8_t)c->range;
    for (i = 0; i < 4; i++)
        y.swizzle[i] = (uint8_t)(comps[i] == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_R + i : comps[i]);
    y.x_chroma = (uint8_t)c->x_chroma;
    y.y_chroma = (uint8_t)c->y_chroma;
    y.chroma_filter = (uint8_t)c->chroma_filter;
    y.explicit_reconstruction = (uint8_t)c->force_explicit;
    y.min_filter = (uint8_t)c->min_filter;
    y.mag_filter = (uint8_t)c->mag_filter;
    y.min_lod = 0.0f;
    y.max_lod = VK_LOD_CLAMP_NONE;
    return y;
}

/* Draws a full-target triangle sampling (implicit LOD) a tw x th target
 * from an IMG_W x IMG_H image, and compares every pixel center. */
static void frag_case(const char *name, uint32_t format, struct conv c, uint32_t tw, uint32_t th)
{
    const struct yfmt *f = &yfmts[format];
    struct pvk_ycbcr y = frag_ycbcr(f, &c);
    struct pvk_msl_binding lb;
    struct pvk_msl_set_layout sl;
    struct pvk_msl_ycbcr_binding yb;
    struct pvk_msl_options fo;
    struct pvk_msl_result vs, fs;
    struct mr_pipeline_desc pd2;
    struct mr_pass pass;
    struct mr_draw draw;
    struct mt_binding bind;
    struct planes pl;
    struct ref r;
    id pso = NULL, target = NULL, smp = NULL, tex[3] = { NULL, NULL, NULL };
    float *pixels = NULL;
    uint32_t p, x, yy, bad = 0, compared = 0;

    planes_init(&pl, f, IMG_W, IMG_H, 100 + format);
    memset(&lb, 0, sizeof(lb));
    lb.binding = 0;
    lb.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    lb.count = 1;
    lb.stride = f->planes > 1 ? 32 : 16;
    sl.bindings = &lb;
    sl.binding_count = 1;
    yb.set = yb.binding = 0;
    yb.conversions = &y;
    memset(&fo, 0, sizeof(fo));
    fo.set_layouts = &sl;
    fo.set_layout_count = 1;
    fo.ycbcr_bindings = &yb;
    fo.ycbcr_binding_count = 1;
    memset(&pd2, 0, sizeof(pd2));
    pd2.colors[0] = PVK_MTLPixelFormatRGBA32Float;
    pd2.ncolors = 1;
    if (!pl.data ||
        !mr_build(spv_quad_vert, PVK_ARRAY_SIZE(spv_quad_vert), NULL, spv_sample_frag, PVK_ARRAY_SIZE(spv_sample_frag), &fo,
                  &pd2, &vs, &fs, &pso)) {
        PVK_TEST_FAIL("%s: pipeline", name);
        free(pl.data);
        return;
    }
    for (p = 0; p < f->planes; p++) {
        uint32_t pw, ph;
        plane_extent(f, p, IMG_W, IMG_H, &pw, &ph);
        tex[p] = mt_texture(PVK_MTLTextureType2D, f->planes == 1 || f->plane_bytes[p] == 2 ? PVK_MTLPixelFormatRG8Unorm
                                                                                          : PVK_MTLPixelFormatR8Unorm,
                            pw, ph, 1, 1, 1, 1);
        mt_upload(tex[p], 0, 0, pw, ph, 1, f->plane_bytes[p], pl.data + pl.offsets[p]);
    }
    smp = frag_sampler(c.min_filter, c.mag_filter);
    target = mr_target(PVK_MTLTextureType2D, PVK_MTLPixelFormatRGBA32Float, tw, th, 1, 1);
    memset(&bind, 0, sizeof(bind));
    bind.texture = tex[0];
    bind.sampler = smp;
    bind.planes[0] = tex[1];
    bind.planes[1] = tex[2];
    memset(&pass, 0, sizeof(pass));
    pass.colors[0] = target;
    pass.ncolors = 1;
    memset(&draw, 0, sizeof(draw));
    draw.pipeline = pso;
    draw.vs = &vs;
    draw.fs = &fs;
    draw.bindings = &bind;
    draw.nbindings = 1;
    draw.primitive = PVK_MTLPrimitiveTypeTriangle;
    draw.vertex_count = 3;
    pixels = calloc((size_t)tw * th * 4, sizeof(float));
    if (!pixels || !mr_render(&pass, &draw, 1)) {
        PVK_TEST_FAIL("%s: render", name);
        goto out;
    }
    mt_download(target, 0, 0, tw, th, 1, 16, pixels);
    r.f = f;
    r.pl = &pl;
    r.w = IMG_W;
    r.h = IMG_H;
    for (yy = 0; yy < th; yy++)
        for (x = 0; x < tw; x++) {
            const float *got = &pixels[(yy * tw + x) * 4];
            float want[4], s = (x + 0.5f) / tw, t = (yy + 0.5f) / th, lod = log2f(fmaxf((float)IMG_W / tw, (float)IMG_H / th));
            bool ambiguous = false, linear = false;
            unsigned k;
            ref_sample(&r, &c, s, t, lod, want, &ambiguous, &linear);
            if (ambiguous)
                continue;
            compared++;
            for (k = 0; k < 4; k++)
                if (fabsf(got[k] - want[k]) > (linear ? TOL_LINEAR : TOL_NEAREST)) {
                    if (bad++ < 4)
                        fprintf(stderr, "%s: pixel (%u, %u): got (%.4f %.4f %.4f %.4f), want (%.4f %.4f %.4f %.4f)\n", name, x,
                                yy, got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                    break;
                }
        }
    if (bad || compared < tw * th / 3) /* 2/3 scaling: a third of the rows and columns are on texel edges */
        PVK_TEST_FAIL("%s: %u wrong of %u pixels compared", name, bad, compared);
out:
    free(pixels);
    for (p = 0; p < 3; p++)
        pvk_release(tex[p]);
    pvk_release(smp);
    pvk_release(target);
    pvk_release(pso);
    pvk_msl_result_free(&vs);
    pvk_msl_result_free(&fs);
    free(pl.data);
}

static void test_fragment(void)
{
    struct conv c = conv_default();
    /* magnified (24 x 12 from 16 x 8) and minified (11 x 5) */
    frag_case("frag 3-plane 4:2:0 implicit, magnified", FMT_3PLANE_420, c, 24, 12);
    frag_case("frag 3-plane 4:2:0 implicit, minified", FMT_3PLANE_420, c, 11, 5);
    c.min_filter = VK_FILTER_LINEAR;
    c.mag_filter = VK_FILTER_NEAREST;
    frag_case("frag 2-plane 4:2:0 implicit min/mag, magnified", FMT_2PLANE_420, c, 24, 12);
    frag_case("frag 2-plane 4:2:0 implicit min/mag, minified", FMT_2PLANE_420, c, 11, 5);
    c.force_explicit = VK_TRUE;
    c.x_chroma = VK_CHROMA_LOCATION_MIDPOINT;
    frag_case("frag 2-plane 4:2:0 explicit min/mag, magnified", FMT_2PLANE_420, c, 24, 12);
    frag_case("frag 2-plane 4:2:0 explicit min/mag, minified", FMT_2PLANE_420, c, 11, 5);
    c = conv_default();
    c.model = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601;
    c.range = VK_SAMPLER_YCBCR_RANGE_ITU_FULL;
    frag_case("frag packed G8B8G8R8 4:2:2", FMT_PACKED_GBGR, c, 24, 12);
    c.chroma_filter = VK_FILTER_NEAREST;
    c.min_filter = c.mag_filter = VK_FILTER_NEAREST;
    frag_case("frag packed B8G8R8G8 4:2:2 nearest", FMT_PACKED_BGRG, c, 24, 12);
}

/* The translator rejects what Y'CbCr sampling does not support: here, a
 * gather on a combined image sampler with a conversion. Translating the
 * compute shader for a packed format and for a 3-plane format succeeds. */
static void test_translator(void)
{
    struct conv c = conv_default();
    struct pvk_ycbcr y = frag_ycbcr(&yfmts[FMT_3PLANE_420], &c);
    struct pvk_msl_binding lb[3];
    struct pvk_msl_set_layout sl;
    struct pvk_msl_ycbcr_binding yb;
    struct pvk_msl_options o;
    struct pvk_msl_result res;
    uint32_t i;
    memset(lb, 0, sizeof(lb));
    for (i = 0; i < 3; i++) { /* offsets 0 (32 bytes), 32, 48 */
        lb[i].binding = i;
        lb[i].type = i ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        lb[i].count = 1;
        lb[i].offset = i ? 16 + 16 * i : 0;
        lb[i].stride = 16;
    }
    lb[0].stride = 32;
    sl.bindings = lb;
    sl.binding_count = 3;
    yb.set = yb.binding = 0;
    yb.conversions = &y;
    memset(&o, 0, sizeof(o));
    o.set_layouts = &sl;
    o.set_layout_count = 1;
    o.ycbcr_bindings = &yb;
    o.ycbcr_binding_count = 1;
    CHECK_VK(pvk_spirv_to_msl(spv_sample_comp, PVK_ARRAY_SIZE(spv_sample_comp), NULL, &o, &res), VK_SUCCESS);
    CHECK(res.msl && strstr(res.msl, "pvk_ycbcr_d b0;") && strstr(res.msl, "pvk_ycbcr_0("));
    pvk_msl_result_free(&res);
    /* a layout whose next binding overlaps the planes */
    lb[1].offset = 16;
    CHECK_VK(pvk_spirv_to_msl(spv_sample_comp, PVK_ARRAY_SIZE(spv_sample_comp), NULL, &o, &res),
             VK_ERROR_INITIALIZATION_FAILED);
    pvk_msl_result_free(&res);
    /* an invalid conversion */
    lb[1].offset = 32;
    y.bits = 3;
    CHECK_VK(pvk_spirv_to_msl(spv_sample_comp, PVK_ARRAY_SIZE(spv_sample_comp), NULL, &o, &res),
             VK_ERROR_INITIALIZATION_FAILED);
    pvk_msl_result_free(&res);
}

int main(void)
{
    setup();
    if (!device || !mt_init()) {
        fprintf(stderr, "no device\n");
        return 1;
    }
    RUN(test_format_properties);
    RUN(test_plane_copies);
    RUN(test_image_plane_copies);
    RUN(test_requirements_layouts);
    RUN(test_host_copies);
    RUN(test_plane_views);
    RUN(test_descriptor_layouts);
    RUN(test_descriptor_updates);
    RUN(test_translator);
    RUN(test_sample_formats);
    RUN(test_sample_models);
    RUN(test_sample_chroma);
    RUN(test_sample_min_mag);
    RUN(test_sample_mapping_tiling);
    RUN(test_fragment);
    mt_shutdown();
    teardown();
    return TEST_RESULT();
}
