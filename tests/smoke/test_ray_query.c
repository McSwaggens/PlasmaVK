/* Acceleration structures (VK_KHR_acceleration_structure: sections 10, 15
 * and 16) and ray queries (VK_KHR_ray_query: section 13, RAY QUERIES): a
 * scene of bottom levels of triangles and of a bounding box, a top level of
 * instances of them, and a compute shader that queries rays through it,
 * whose hits are compared with the same rays traced here on the host.
 *
 * Shaders: GLSL in the comment above each array, compiled with
 *   glslangValidator -V --target-env vulkan1.3 */
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <math.h>
#include <string.h>

#include "../pvk_test.h"

/*
 * #version 460
 * #extension GL_EXT_ray_query : require
 * layout(local_size_x = 64) in;
 * struct Ray { vec3 origin; float tmin; vec3 direction; float tmax; };
 * struct Hit {
 *     float t; uint primitive; uint instance; uint custom;
 *     vec2 bary; uint type; uint front;
 *     vec3 translation; uint geometry;
 *     vec3 object_origin; uint offset;
 * };
 * layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
 * layout(set = 0, binding = 1, std430) readonly buffer Rays { Ray rays[]; };
 * layout(set = 0, binding = 2, std430) writeonly buffer Hits { Hit hits[]; };
 * layout(push_constant) uniform Push { uint count; uint flags; uint mask; uint mode; } pc;
 * void main()
 * {
 *     uint i = gl_GlobalInvocationID.x;
 *     if (i >= pc.count)
 *         return;
 *     Ray r = rays[i];
 *     rayQueryEXT q;
 *     rayQueryInitializeEXT(q, tlas, pc.flags, pc.mask, r.origin, r.tmin, r.direction, r.tmax);
 *     while (rayQueryProceedEXT(q)) {
 *         if (rayQueryGetIntersectionTypeEXT(q, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
 *             // mode 1: as an any-hit shader that ignores odd primitives would
 *             if (pc.mode != 1u || (rayQueryGetIntersectionPrimitiveIndexEXT(q, false) & 1) == 0)
 *                 rayQueryConfirmIntersectionEXT(q);
 *         } else {
 *             // a sphere of radius 0.5 at the middle of the box, in object space
 *             vec3 o = rayQueryGetIntersectionObjectRayOriginEXT(q, false);
 *             vec3 d = rayQueryGetIntersectionObjectRayDirectionEXT(q, false);
 *             float a = dot(d, d), b = dot(o, d), c = dot(o, o) - 0.25;
 *             float disc = b * b - a * c;
 *             float far = r.tmax;
 *             if (rayQueryGetIntersectionTypeEXT(q, true) != gl_RayQueryCommittedIntersectionNoneEXT)
 *                 far = rayQueryGetIntersectionTEXT(q, true);
 *             if (disc >= 0.0) {
 *                 float t = (-b - sqrt(disc)) / a;
 *                 if (t >= rayQueryGetRayTMinEXT(q) && t <= far)
 *                     rayQueryGenerateIntersectionEXT(q, t);
 *             }
 *         }
 *     }
 *     Hit h;
 *     h.t = -1.0;
 *     h.primitive = h.instance = h.custom = h.front = h.geometry = h.offset = 0u;
 *     h.bary = vec2(0.0);
 *     h.translation = h.object_origin = vec3(0.0);
 *     h.type = rayQueryGetIntersectionTypeEXT(q, true);
 *     if (h.type != gl_RayQueryCommittedIntersectionNoneEXT) {
 *         h.t = rayQueryGetIntersectionTEXT(q, true);
 *         h.primitive = uint(rayQueryGetIntersectionPrimitiveIndexEXT(q, true));
 *         h.instance = uint(rayQueryGetIntersectionInstanceIdEXT(q, true));
 *         h.custom = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(q, true));
 *         h.geometry = uint(rayQueryGetIntersectionGeometryIndexEXT(q, true));
 *         h.offset = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(q, true);
 *         h.translation = rayQueryGetIntersectionObjectToWorldEXT(q, true)[3];
 *         h.object_origin = rayQueryGetIntersectionObjectRayOriginEXT(q, true);
 *         if (h.type == gl_RayQueryCommittedIntersectionTriangleEXT) {
 *             h.bary = rayQueryGetIntersectionBarycentricsEXT(q, true);
 *             h.front = rayQueryGetIntersectionFrontFaceEXT(q, true) ? 1u : 0u;
 *         }
 *     }
 *     hits[i] = h;
 * }
 */
static const uint32_t spv_trace[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x000000e4, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x00001178, 0x0006000a, 0x5f565053, 0x5f52484b, 0x5f796172, 0x72657571, 0x00000079, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x000b000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00000013, 0x00000027,
    0x0000002f, 0x00000032, 0x000000dc, 0x00060010, 0x00000004, 0x00000011, 0x00000040, 0x00000001,
    0x00000001, 0x00030003, 0x00000002, 0x000001cc, 0x00060004, 0x455f4c47, 0x725f5458, 0x715f7961,
    0x79726575, 0x00000000, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008,
    0x00000069, 0x00080005, 0x0000000b, 0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69,
    0x00000044, 0x00040005, 0x00000011, 0x68737550, 0x00000000, 0x00050006, 0x00000011, 0x00000000,
    0x6e756f63, 0x00000074, 0x00050006, 0x00000011, 0x00000001, 0x67616c66, 0x00000073, 0x00050006,
    0x00000011, 0x00000002, 0x6b73616d, 0x00000000, 0x00050006, 0x00000011, 0x00000003, 0x65646f6d,
    0x00000000, 0x00030005, 0x00000013, 0x00006370, 0x00030005, 0x00000020, 0x00796152, 0x00050006,
    0x00000020, 0x00000000, 0x6769726f, 0x00006e69, 0x00050006, 0x00000020, 0x00000001, 0x6e696d74,
    0x00000000, 0x00060006, 0x00000020, 0x00000002, 0x65726964, 0x6f697463, 0x0000006e, 0x00050006,
    0x00000020, 0x00000003, 0x78616d74, 0x00000000, 0x00030005, 0x00000022, 0x00000072, 0x00030005,
    0x00000023, 0x00796152, 0x00050006, 0x00000023, 0x00000000, 0x6769726f, 0x00006e69, 0x00050006,
    0x00000023, 0x00000001, 0x6e696d74, 0x00000000, 0x00060006, 0x00000023, 0x00000002, 0x65726964,
    0x6f697463, 0x0000006e, 0x00050006, 0x00000023, 0x00000003, 0x78616d74, 0x00000000, 0x00040005,
    0x00000025, 0x73796152, 0x00000000, 0x00050006, 0x00000025, 0x00000000, 0x73796172, 0x00000000,
    0x00030005, 0x00000027, 0x00000000, 0x00030005, 0x0000002f, 0x00000071, 0x00040005, 0x00000032,
    0x73616c74, 0x00000000, 0x00030005, 0x0000005e, 0x0000006f, 0x00030005, 0x00000060, 0x00000064,
    0x00030005, 0x00000062, 0x00000061, 0x00030005, 0x00000066, 0x00000062, 0x00030005, 0x0000006a,
    0x00000063, 0x00040005, 0x00000070, 0x63736964, 0x00000000, 0x00030005, 0x00000078, 0x00726166,
    0x00030005, 0x00000086, 0x00000074, 0x00030005, 0x00000099, 0x00746948, 0x00040006, 0x00000099,
    0x00000000, 0x00000074, 0x00060006, 0x00000099, 0x00000001, 0x6d697270, 0x76697469, 0x00000065,
    0x00060006, 0x00000099, 0x00000002, 0x74736e69, 0x65636e61, 0x00000000, 0x00050006, 0x00000099,
    0x00000003, 0x74737563, 0x00006d6f, 0x00050006, 0x00000099, 0x00000004, 0x79726162, 0x00000000,
    0x00050006, 0x00000099, 0x00000005, 0x65707974, 0x00000000, 0x00050006, 0x00000099, 0x00000006,
    0x6e6f7266, 0x00000074, 0x00060006, 0x00000099, 0x00000007, 0x6e617274, 0x74616c73, 0x006e6f69,
    0x00060006, 0x00000099, 0x00000008, 0x6d6f6567, 0x79727465, 0x00000000, 0x00070006, 0x00000099,
    0x00000009, 0x656a626f, 0x6f5f7463, 0x69676972, 0x0000006e, 0x00050006, 0x00000099, 0x0000000a,
    0x7366666f, 0x00007465, 0x00030005, 0x0000009b, 0x00000068, 0x00030005, 0x000000d8, 0x00746948,
    0x00040006, 0x000000d8, 0x00000000, 0x00000074, 0x00060006, 0x000000d8, 0x00000001, 0x6d697270,
    0x76697469, 0x00000065, 0x00060006, 0x000000d8, 0x00000002, 0x74736e69, 0x65636e61, 0x00000000,
    0x00050006, 0x000000d8, 0x00000003, 0x74737563, 0x00006d6f, 0x00050006, 0x000000d8, 0x00000004,
    0x79726162, 0x00000000, 0x00050006, 0x000000d8, 0x00000005, 0x65707974, 0x00000000, 0x00050006,
    0x000000d8, 0x00000006, 0x6e6f7266, 0x00000074, 0x00060006, 0x000000d8, 0x00000007, 0x6e617274,
    0x74616c73, 0x006e6f69, 0x00060006, 0x000000d8, 0x00000008, 0x6d6f6567, 0x79727465, 0x00000000,
    0x00070006, 0x000000d8, 0x00000009, 0x656a626f, 0x6f5f7463, 0x69676972, 0x0000006e, 0x00050006,
    0x000000d8, 0x0000000a, 0x7366666f, 0x00007465, 0x00040005, 0x000000da, 0x73746948, 0x00000000,
    0x00050006, 0x000000da, 0x00000000, 0x73746968, 0x00000000, 0x00030005, 0x000000dc, 0x00000000,
    0x00040047, 0x0000000b, 0x0000000b, 0x0000001c, 0x00030047, 0x00000011, 0x00000002, 0x00050048,
    0x00000011, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000011, 0x00000001, 0x00000023,
    0x00000004, 0x00050048, 0x00000011, 0x00000002, 0x00000023, 0x00000008, 0x00050048, 0x00000011,
    0x00000003, 0x00000023, 0x0000000c, 0x00050048, 0x00000023, 0x00000000, 0x00000023, 0x00000000,
    0x00050048, 0x00000023, 0x00000001, 0x00000023, 0x0000000c, 0x00050048, 0x00000023, 0x00000002,
    0x00000023, 0x00000010, 0x00050048, 0x00000023, 0x00000003, 0x00000023, 0x0000001c, 0x00040047,
    0x00000024, 0x00000006, 0x00000020, 0x00030047, 0x00000025, 0x00000002, 0x00040048, 0x00000025,
    0x00000000, 0x00000018, 0x00050048, 0x00000025, 0x00000000, 0x00000023, 0x00000000, 0x00030047,
    0x00000027, 0x00000018, 0x00040047, 0x00000027, 0x00000021, 0x00000001, 0x00040047, 0x00000027,
    0x00000022, 0x00000000, 0x00040047, 0x00000032, 0x00000021, 0x00000000, 0x00040047, 0x00000032,
    0x00000022, 0x00000000, 0x00050048, 0x000000d8, 0x00000000, 0x00000023, 0x00000000, 0x00050048,
    0x000000d8, 0x00000001, 0x00000023, 0x00000004, 0x00050048, 0x000000d8, 0x00000002, 0x00000023,
    0x00000008, 0x00050048, 0x000000d8, 0x00000003, 0x00000023, 0x0000000c, 0x00050048, 0x000000d8,
    0x00000004, 0x00000023, 0x00000010, 0x00050048, 0x000000d8, 0x00000005, 0x00000023, 0x00000018,
    0x00050048, 0x000000d8, 0x00000006, 0x00000023, 0x0000001c, 0x00050048, 0x000000d8, 0x00000007,
    0x00000023, 0x00000020, 0x00050048, 0x000000d8, 0x00000008, 0x00000023, 0x0000002c, 0x00050048,
    0x000000d8, 0x00000009, 0x00000023, 0x00000030, 0x00050048, 0x000000d8, 0x0000000a, 0x00000023,
    0x0000003c, 0x00040047, 0x000000d9, 0x00000006, 0x00000040, 0x00030047, 0x000000da, 0x00000002,
    0x00040048, 0x000000da, 0x00000000, 0x00000019, 0x00050048, 0x000000da, 0x00000000, 0x00000023,
    0x00000000, 0x00030047, 0x000000dc, 0x00000019, 0x00040047, 0x000000dc, 0x00000021, 0x00000002,
    0x00040047, 0x000000dc, 0x00000022, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
    0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000007,
    0x00000006, 0x00040017, 0x00000009, 0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001,
    0x00000009, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b, 0x00000006, 0x0000000c,
    0x00000000, 0x00040020, 0x0000000d, 0x00000001, 0x00000006, 0x0006001e, 0x00000011, 0x00000006,
    0x00000006, 0x00000006, 0x00000006, 0x00040020, 0x00000012, 0x00000009, 0x00000011, 0x0004003b,
    0x00000012, 0x00000013, 0x00000009, 0x00040015, 0x00000014, 0x00000020, 0x00000001, 0x0004002b,
    0x00000014, 0x00000015, 0x00000000, 0x00040020, 0x00000016, 0x00000009, 0x00000006, 0x00020014,
    0x00000019, 0x00030016, 0x0000001e, 0x00000020, 0x00040017, 0x0000001f, 0x0000001e, 0x00000003,
    0x0006001e, 0x00000020, 0x0000001f, 0x0000001e, 0x0000001f, 0x0000001e, 0x00040020, 0x00000021,
    0x00000007, 0x00000020, 0x0006001e, 0x00000023, 0x0000001f, 0x0000001e, 0x0000001f, 0x0000001e,
    0x0003001d, 0x00000024, 0x00000023, 0x0003001e, 0x00000025, 0x00000024, 0x00040020, 0x00000026,
    0x0000000c, 0x00000025, 0x0004003b, 0x00000026, 0x00000027, 0x0000000c, 0x00040020, 0x00000029,
    0x0000000c, 0x00000023, 0x00021178, 0x0000002d, 0x00040020, 0x0000002e, 0x00000006, 0x0000002d,
    0x0004003b, 0x0000002e, 0x0000002f, 0x00000006, 0x000214dd, 0x00000030, 0x00040020, 0x00000031,
    0x00000000, 0x00000030, 0x0004003b, 0x00000031, 0x00000032, 0x00000000, 0x0004002b, 0x00000014,
    0x00000034, 0x00000001, 0x0004002b, 0x00000014, 0x00000037, 0x00000002, 0x00040020, 0x0000003a,
    0x00000007, 0x0000001f, 0x00040020, 0x0000003d, 0x00000007, 0x0000001e, 0x0004002b, 0x00000014,
    0x00000042, 0x00000003, 0x0003002a, 0x00000019, 0x0000004b, 0x0004002b, 0x00000006, 0x00000052,
    0x00000001, 0x0004002b, 0x0000001e, 0x0000006e, 0x3e800000, 0x00030029, 0x00000019, 0x0000007b,
    0x0004002b, 0x0000001e, 0x00000082, 0x00000000, 0x00040017, 0x00000098, 0x0000001e, 0x00000002,
    0x000d001e, 0x00000099, 0x0000001e, 0x00000006, 0x00000006, 0x00000006, 0x00000098, 0x00000006,
    0x00000006, 0x0000001f, 0x00000006, 0x0000001f, 0x00000006, 0x00040020, 0x0000009a, 0x00000007,
    0x00000099, 0x0004002b, 0x0000001e, 0x0000009c, 0xbf800000, 0x0004002b, 0x00000014, 0x0000009e,
    0x00000006, 0x0004002b, 0x00000014, 0x0000009f, 0x00000008, 0x0004002b, 0x00000014, 0x000000a0,
    0x0000000a, 0x0004002b, 0x00000014, 0x000000a7, 0x00000004, 0x0005002c, 0x00000098, 0x000000a8,
    0x00000082, 0x00000082, 0x00040020, 0x000000a9, 0x00000007, 0x00000098, 0x0004002b, 0x00000014,
    0x000000ab, 0x00000007, 0x0004002b, 0x00000014, 0x000000ac, 0x00000009, 0x0006002c, 0x0000001f,
    0x000000ad, 0x00000082, 0x00000082, 0x00000082, 0x0004002b, 0x00000014, 0x000000b0, 0x00000005,
    0x00040018, 0x000000c8, 0x0000001f, 0x00000004, 0x000d001e, 0x000000d8, 0x0000001e, 0x00000006,
    0x00000006, 0x00000006, 0x00000098, 0x00000006, 0x00000006, 0x0000001f, 0x00000006, 0x0000001f,
    0x00000006, 0x0003001d, 0x000000d9, 0x000000d8, 0x0003001e, 0x000000da, 0x000000d9, 0x00040020,
    0x000000db, 0x0000000c, 0x000000da, 0x0004003b, 0x000000db, 0x000000dc, 0x0000000c, 0x00040020,
    0x000000df, 0x0000000c, 0x000000d8, 0x0004002b, 0x00000006, 0x000000e2, 0x00000040, 0x0006002c,
    0x00000009, 0x000000e3, 0x000000e2, 0x00000052, 0x00000052, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x00000008, 0x00000007,
    0x0004003b, 0x00000021, 0x00000022, 0x00000007, 0x0004003b, 0x0000003a, 0x0000005e, 0x00000007,
    0x0004003b, 0x0000003a, 0x00000060, 0x00000007, 0x0004003b, 0x0000003d, 0x00000062, 0x00000007,
    0x0004003b, 0x0000003d, 0x00000066, 0x00000007, 0x0004003b, 0x0000003d, 0x0000006a, 0x00000007,
    0x0004003b, 0x0000003d, 0x00000070, 0x00000007, 0x0004003b, 0x0000003d, 0x00000078, 0x00000007,
    0x0004003b, 0x0000003d, 0x00000086, 0x00000007, 0x0004003b, 0x0000009a, 0x0000009b, 0x00000007,
    0x00050041, 0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c, 0x0004003d, 0x00000006, 0x0000000f,
    0x0000000e, 0x0003003e, 0x00000008, 0x0000000f, 0x0004003d, 0x00000006, 0x00000010, 0x00000008,
    0x00050041, 0x00000016, 0x00000017, 0x00000013, 0x00000015, 0x0004003d, 0x00000006, 0x00000018,
    0x00000017, 0x000500ae, 0x00000019, 0x0000001a, 0x00000010, 0x00000018, 0x000300f7, 0x0000001c,
    0x00000000, 0x000400fa, 0x0000001a, 0x0000001b, 0x0000001c, 0x000200f8, 0x0000001b, 0x000100fd,
    0x000200f8, 0x0000001c, 0x0004003d, 0x00000006, 0x00000028, 0x00000008, 0x00060041, 0x00000029,
    0x0000002a, 0x00000027, 0x00000015, 0x00000028, 0x0004003d, 0x00000023, 0x0000002b, 0x0000002a,
    0x00040190, 0x00000020, 0x0000002c, 0x0000002b, 0x0003003e, 0x00000022, 0x0000002c, 0x0004003d,
    0x00000030, 0x00000033, 0x00000032, 0x00050041, 0x00000016, 0x00000035, 0x00000013, 0x00000034,
    0x0004003d, 0x00000006, 0x00000036, 0x00000035, 0x00050041, 0x00000016, 0x00000038, 0x00000013,
    0x00000037, 0x0004003d, 0x00000006, 0x00000039, 0x00000038, 0x00050041, 0x0000003a, 0x0000003b,
    0x00000022, 0x00000015, 0x0004003d, 0x0000001f, 0x0000003c, 0x0000003b, 0x00050041, 0x0000003d,
    0x0000003e, 0x00000022, 0x00000034, 0x0004003d, 0x0000001e, 0x0000003f, 0x0000003e, 0x00050041,
    0x0000003a, 0x00000040, 0x00000022, 0x00000037, 0x0004003d, 0x0000001f, 0x00000041, 0x00000040,
    0x00050041, 0x0000003d, 0x00000043, 0x00000022, 0x00000042, 0x0004003d, 0x0000001e, 0x00000044,
    0x00000043, 0x00091179, 0x0000002f, 0x00000033, 0x00000036, 0x00000039, 0x0000003c, 0x0000003f,
    0x00000041, 0x00000044, 0x000200f9, 0x00000045, 0x000200f8, 0x00000045, 0x000400f6, 0x00000047,
    0x00000048, 0x00000000, 0x000200f9, 0x00000049, 0x000200f8, 0x00000049, 0x0004117d, 0x00000019,
    0x0000004a, 0x0000002f, 0x000400fa, 0x0000004a, 0x00000046, 0x00000047, 0x000200f8, 0x00000046,
    0x0005117f, 0x00000006, 0x0000004c, 0x0000002f, 0x00000015, 0x000500aa, 0x00000019, 0x0000004d,
    0x0000004c, 0x0000000c, 0x000300f7, 0x0000004f, 0x00000000, 0x000400fa, 0x0000004d, 0x0000004e,
    0x0000005d, 0x000200f8, 0x0000004e, 0x00050041, 0x00000016, 0x00000050, 0x00000013, 0x00000042,
    0x0004003d, 0x00000006, 0x00000051, 0x00000050, 0x000500ab, 0x00000019, 0x00000053, 0x00000051,
    0x00000052, 0x000400a8, 0x00000019, 0x00000054, 0x00000053, 0x000300f7, 0x00000056, 0x00000000,
    0x000400fa, 0x00000054, 0x00000055, 0x00000056, 0x000200f8, 0x00000055, 0x00051787, 0x00000014,
    0x00000057, 0x0000002f, 0x00000015, 0x000500c7, 0x00000014, 0x00000058, 0x00000057, 0x00000034,
    0x000500aa, 0x00000019, 0x00000059, 0x00000058, 0x00000015, 0x000200f9, 0x00000056, 0x000200f8,
    0x00000056, 0x000700f5, 0x00000019, 0x0000005a, 0x00000053, 0x0000004e, 0x00000059, 0x00000055,
    0x000300f7, 0x0000005c, 0x00000000, 0x000400fa, 0x0000005a, 0x0000005b, 0x0000005c, 0x000200f8,
    0x0000005b, 0x0002117c, 0x0000002f, 0x000200f9, 0x0000005c, 0x000200f8, 0x0000005c, 0x000200f9,
    0x0000004f, 0x000200f8, 0x0000005d, 0x0005178c, 0x0000001f, 0x0000005f, 0x0000002f, 0x00000015,
    0x0003003e, 0x0000005e, 0x0000005f, 0x0005178b, 0x0000001f, 0x00000061, 0x0000002f, 0x00000015,
    0x0003003e, 0x00000060, 0x00000061, 0x0004003d, 0x0000001f, 0x00000063, 0x00000060, 0x0004003d,
    0x0000001f, 0x00000064, 0x00000060, 0x00050094, 0x0000001e, 0x00000065, 0x00000063, 0x00000064,
    0x0003003e, 0x00000062, 0x00000065, 0x0004003d, 0x0000001f, 0x00000067, 0x0000005e, 0x0004003d,
    0x0000001f, 0x00000068, 0x00000060, 0x00050094, 0x0000001e, 0x00000069, 0x00000067, 0x00000068,
    0x0003003e, 0x00000066, 0x00000069, 0x0004003d, 0x0000001f, 0x0000006b, 0x0000005e, 0x0004003d,
    0x0000001f, 0x0000006c, 0x0000005e, 0x00050094, 0x0000001e, 0x0000006d, 0x0000006b, 0x0000006c,
    0x00050083, 0x0000001e, 0x0000006f, 0x0000006d, 0x0000006e, 0x0003003e, 0x0000006a, 0x0000006f,
    0x0004003d, 0x0000001e, 0x00000071, 0x00000066, 0x0004003d, 0x0000001e, 0x00000072, 0x00000066,
    0x00050085, 0x0000001e, 0x00000073, 0x00000071, 0x00000072, 0x0004003d, 0x0000001e, 0x00000074,
    0x00000062, 0x0004003d, 0x0000001e, 0x00000075, 0x0000006a, 0x00050085, 0x0000001e, 0x00000076,
    0x00000074, 0x00000075, 0x00050083, 0x0000001e, 0x00000077, 0x00000073, 0x00000076, 0x0003003e,
    0x00000070, 0x00000077, 0x00050041, 0x0000003d, 0x00000079, 0x00000022, 0x00000042, 0x0004003d,
    0x0000001e, 0x0000007a, 0x00000079, 0x0003003e, 0x00000078, 0x0000007a, 0x0005117f, 0x00000006,
    0x0000007c, 0x0000002f, 0x00000034, 0x000500ab, 0x00000019, 0x0000007d, 0x0000007c, 0x0000000c,
    0x000300f7, 0x0000007f, 0x00000000, 0x000400fa, 0x0000007d, 0x0000007e, 0x0000007f, 0x000200f8,
    0x0000007e, 0x00051782, 0x0000001e, 0x00000080, 0x0000002f, 0x00000034, 0x0003003e, 0x00000078,
    0x00000080, 0x000200f9, 0x0000007f, 0x000200f8, 0x0000007f, 0x0004003d, 0x0000001e, 0x00000081,
    0x00000070, 0x000500be, 0x00000019, 0x00000083, 0x00000081, 0x00000082, 0x000300f7, 0x00000085,
    0x00000000, 0x000400fa, 0x00000083, 0x00000084, 0x00000085, 0x000200f8, 0x00000084, 0x0004003d,
    0x0000001e, 0x00000087, 0x00000066, 0x0004007f, 0x0000001e, 0x00000088, 0x00000087, 0x0004003d,
    0x0000001e, 0x00000089, 0x00000070, 0x0006000c, 0x0000001e, 0x0000008a, 0x00000001, 0x0000001f,
    0x00000089, 0x00050083, 0x0000001e, 0x0000008b, 0x00000088, 0x0000008a, 0x0004003d, 0x0000001e,
    0x0000008c, 0x00000062, 0x00050088, 0x0000001e, 0x0000008d, 0x0000008b, 0x0000008c, 0x0003003e,
    0x00000086, 0x0000008d, 0x0004003d, 0x0000001e, 0x0000008e, 0x00000086, 0x00041780, 0x0000001e,
    0x0000008f, 0x0000002f, 0x000500be, 0x00000019, 0x00000090, 0x0000008e, 0x0000008f, 0x0004003d,
    0x0000001e, 0x00000091, 0x00000086, 0x0004003d, 0x0000001e, 0x00000092, 0x00000078, 0x000500bc,
    0x00000019, 0x00000093, 0x00000091, 0x00000092, 0x000500a7, 0x00000019, 0x00000094, 0x00000090,
    0x00000093, 0x000300f7, 0x00000096, 0x00000000, 0x000400fa, 0x00000094, 0x00000095, 0x00000096,
    0x000200f8, 0x00000095, 0x0004003d, 0x0000001e, 0x00000097, 0x00000086, 0x0003117b, 0x0000002f,
    0x00000097, 0x000200f9, 0x00000096, 0x000200f8, 0x00000096, 0x000200f9, 0x00000085, 0x000200f8,
    0x00000085, 0x000200f9, 0x0000004f, 0x000200f8, 0x0000004f, 0x000200f9, 0x00000048, 0x000200f8,
    0x00000048, 0x000200f9, 0x00000045, 0x000200f8, 0x00000047, 0x00050041, 0x0000003d, 0x0000009d,
    0x0000009b, 0x00000015, 0x0003003e, 0x0000009d, 0x0000009c, 0x00050041, 0x00000007, 0x000000a1,
    0x0000009b, 0x000000a0, 0x0003003e, 0x000000a1, 0x0000000c, 0x00050041, 0x00000007, 0x000000a2,
    0x0000009b, 0x0000009f, 0x0003003e, 0x000000a2, 0x0000000c, 0x00050041, 0x00000007, 0x000000a3,
    0x0000009b, 0x0000009e, 0x0003003e, 0x000000a3, 0x0000000c, 0x00050041, 0x00000007, 0x000000a4,
    0x0000009b, 0x00000042, 0x0003003e, 0x000000a4, 0x0000000c, 0x00050041, 0x00000007, 0x000000a5,
    0x0000009b, 0x00000037, 0x0003003e, 0x000000a5, 0x0000000c, 0x00050041, 0x00000007, 0x000000a6,
    0x0000009b, 0x00000034, 0x0003003e, 0x000000a6, 0x0000000c, 0x00050041, 0x000000a9, 0x000000aa,
    0x0000009b, 0x000000a7, 0x0003003e, 0x000000aa, 0x000000a8, 0x00050041, 0x0000003a, 0x000000ae,
    0x0000009b, 0x000000ac, 0x0003003e, 0x000000ae, 0x000000ad, 0x00050041, 0x0000003a, 0x000000af,
    0x0000009b, 0x000000ab, 0x0003003e, 0x000000af, 0x000000ad, 0x0005117f, 0x00000006, 0x000000b1,
    0x0000002f, 0x00000034, 0x00050041, 0x00000007, 0x000000b2, 0x0000009b, 0x000000b0, 0x0003003e,
    0x000000b2, 0x000000b1, 0x00050041, 0x00000007, 0x000000b3, 0x0000009b, 0x000000b0, 0x0004003d,
    0x00000006, 0x000000b4, 0x000000b3, 0x000500ab, 0x00000019, 0x000000b5, 0x000000b4, 0x0000000c,
    0x000300f7, 0x000000b7, 0x00000000, 0x000400fa, 0x000000b5, 0x000000b6, 0x000000b7, 0x000200f8,
    0x000000b6, 0x00051782, 0x0000001e, 0x000000b8, 0x0000002f, 0x00000034, 0x00050041, 0x0000003d,
    0x000000b9, 0x0000009b, 0x00000015, 0x0003003e, 0x000000b9, 0x000000b8, 0x00051787, 0x00000014,
    0x000000ba, 0x0000002f, 0x00000034, 0x0004007c, 0x00000006, 0x000000bb, 0x000000ba, 0x00050041,
    0x00000007, 0x000000bc, 0x0000009b, 0x00000034, 0x0003003e, 0x000000bc, 0x000000bb, 0x00051784,
    0x00000014, 0x000000bd, 0x0000002f, 0x00000034, 0x0004007c, 0x00000006, 0x000000be, 0x000000bd,
    0x00050041, 0x00000007, 0x000000bf, 0x0000009b, 0x00000037, 0x0003003e, 0x000000bf, 0x000000be,
    0x00051783, 0x00000014, 0x000000c0, 0x0000002f, 0x00000034, 0x0004007c, 0x00000006, 0x000000c1,
    0x000000c0, 0x00050041, 0x00000007, 0x000000c2, 0x0000009b, 0x00000042, 0x0003003e, 0x000000c2,
    0x000000c1, 0x00051786, 0x00000014, 0x000000c3, 0x0000002f, 0x00000034, 0x0004007c, 0x00000006,
    0x000000c4, 0x000000c3, 0x00050041, 0x00000007, 0x000000c5, 0x0000009b, 0x0000009f, 0x0003003e,
    0x000000c5, 0x000000c4, 0x00051785, 0x00000006, 0x000000c6, 0x0000002f, 0x00000034, 0x00050041,
    0x00000007, 0x000000c7, 0x0000009b, 0x000000a0, 0x0003003e, 0x000000c7, 0x000000c6, 0x0005178f,
    0x000000c8, 0x000000c9, 0x0000002f, 0x00000034, 0x00050051, 0x0000001f, 0x000000ca, 0x000000c9,
    0x00000003, 0x00050041, 0x0000003a, 0x000000cb, 0x0000009b, 0x000000ab, 0x0003003e, 0x000000cb,
    0x000000ca, 0x0005178c, 0x0000001f, 0x000000cc, 0x0000002f, 0x00000034, 0x00050041, 0x0000003a,
    0x000000cd, 0x0000009b, 0x000000ac, 0x0003003e, 0x000000cd, 0x000000cc, 0x00050041, 0x00000007,
    0x000000ce, 0x0000009b, 0x000000b0, 0x0004003d, 0x00000006, 0x000000cf, 0x000000ce, 0x000500aa,
    0x00000019, 0x000000d0, 0x000000cf, 0x00000052, 0x000300f7, 0x000000d2, 0x00000000, 0x000400fa,
    0x000000d0, 0x000000d1, 0x000000d2, 0x000200f8, 0x000000d1, 0x00051788, 0x00000098, 0x000000d3,
    0x0000002f, 0x00000034, 0x00050041, 0x000000a9, 0x000000d4, 0x0000009b, 0x000000a7, 0x0003003e,
    0x000000d4, 0x000000d3, 0x00051789, 0x00000019, 0x000000d5, 0x0000002f, 0x00000034, 0x000600a9,
    0x00000006, 0x000000d6, 0x000000d5, 0x00000052, 0x0000000c, 0x00050041, 0x00000007, 0x000000d7,
    0x0000009b, 0x0000009e, 0x0003003e, 0x000000d7, 0x000000d6, 0x000200f9, 0x000000d2, 0x000200f8,
    0x000000d2, 0x000200f9, 0x000000b7, 0x000200f8, 0x000000b7, 0x0004003d, 0x00000006, 0x000000dd,
    0x00000008, 0x0004003d, 0x00000099, 0x000000de, 0x0000009b, 0x00060041, 0x000000df, 0x000000e0,
    0x000000dc, 0x00000015, 0x000000dd, 0x00040190, 0x000000d8, 0x000000e1, 0x000000de, 0x0003003e,
    0x000000e0, 0x000000e1, 0x000100fd, 0x00010038,
};

static VkInstance instance;
static VkPhysicalDevice physical_device;
static VkDevice device;
static VkQueue queue;
static VkCommandPool pool;
static VkFence fence;
static VkDescriptorSetLayout set_layout;
static VkPipelineLayout layout;
static VkDescriptorPool descriptor_pool;
static VkPipeline pipeline;
static uint32_t shared_type = UINT32_MAX;
static uint32_t private_type = UINT32_MAX;
static bool supported;

struct buf {
    VkBuffer buffer;
    VkDeviceMemory memory;
    unsigned char *data; /* NULL for private memory */
    VkDeviceAddress address;
    VkDeviceSize size;
};

struct accel {
    VkAccelerationStructureKHR handle;
    struct buf storage;
    VkDeviceAddress address;
};

static void setup(void)
{
    static const char *const extensions[] = { VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                                              VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                              VK_KHR_RAY_QUERY_EXTENSION_NAME };
    VkPhysicalDeviceMemoryProperties mem;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel;
    VkPhysicalDeviceRayQueryFeaturesKHR ray_query;
    VkPhysicalDeviceFeatures2 features;
    VkApplicationInfo app;
    VkInstanceCreateInfo instance_info;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo device_info;
    VkCommandPoolCreateInfo pool_info;
    VkFenceCreateInfo fence_info;
    VkDescriptorSetLayoutBinding bindings[3];
    VkDescriptorSetLayoutCreateInfo set_info;
    VkPushConstantRange range;
    VkPipelineLayoutCreateInfo layout_info;
    VkDescriptorPoolSize sizes[2];
    VkDescriptorPoolCreateInfo dp_info;
    VkShaderModuleCreateInfo module_info;
    VkComputePipelineCreateInfo pipeline_info;
    VkShaderModule module;
    float priority = 1.0f;
    uint32_t count = 1, i;

    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_4;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    REQUIRE_VK(vkCreateInstance(&instance_info, NULL, &instance));
    REQUIRE_VK(vkEnumeratePhysicalDevices(instance, &count, &physical_device));
    memset(&ray_query, 0, sizeof(ray_query));
    ray_query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    memset(&accel, 0, sizeof(accel));
    accel.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accel.pNext = &ray_query;
    memset(&features, 0, sizeof(features));
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &accel;
    vkGetPhysicalDeviceFeatures2(physical_device, &features);
    supported = accel.accelerationStructure && ray_query.rayQuery;
    if (!supported)
        return;

    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.bufferDeviceAddress = VK_TRUE;
    memset(&accel, 0, sizeof(accel));
    accel.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accel.pNext = &v12;
    accel.accelerationStructure = VK_TRUE;
    memset(&ray_query, 0, sizeof(ray_query));
    ray_query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    ray_query.pNext = &accel;
    ray_query.rayQuery = VK_TRUE;
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &ray_query;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = 3;
    device_info.ppEnabledExtensionNames = extensions;
    REQUIRE_VK(vkCreateDevice(physical_device, &device_info, NULL, &device));
    vkGetDeviceQueue(device, 0, 0, &queue);
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem);
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

    memset(bindings, 0, sizeof(bindings));
    for (i = 0; i < 3; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    memset(&set_info, 0, sizeof(set_info));
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.bindingCount = 3;
    set_info.pBindings = bindings;
    REQUIRE_VK(vkCreateDescriptorSetLayout(device, &set_info, NULL, &set_layout));
    memset(&range, 0, sizeof(range));
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = 16;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    REQUIRE_VK(vkCreatePipelineLayout(device, &layout_info, NULL, &layout));
    sizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    sizes[0].descriptorCount = 8;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[1].descriptorCount = 16;
    memset(&dp_info, 0, sizeof(dp_info));
    dp_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_info.maxSets = 8;
    dp_info.poolSizeCount = 2;
    dp_info.pPoolSizes = sizes;
    REQUIRE_VK(vkCreateDescriptorPool(device, &dp_info, NULL, &descriptor_pool));

    memset(&module_info, 0, sizeof(module_info));
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = sizeof(spv_trace);
    module_info.pCode = spv_trace;
    REQUIRE_VK(vkCreateShaderModule(device, &module_info, NULL, &module));
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = layout;
    CHECK_VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline), VK_SUCCESS);
    vkDestroyShaderModule(device, module, NULL);
}

static void teardown(void)
{
    if (supported) {
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyDescriptorPool(device, descriptor_pool, NULL);
        vkDestroyPipelineLayout(device, layout, NULL);
        vkDestroyDescriptorSetLayout(device, set_layout, NULL);
        vkDestroyFence(device, fence, NULL);
        vkDestroyCommandPool(device, pool, NULL);
        vkDestroyDevice(device, NULL);
    }
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
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
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

static VkResult finish(VkCommandBuffer cb)
{
    VkSubmitInfo submit;
    VkResult r = vkEndCommandBuffer(cb);
    if (r == VK_SUCCESS) {
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cb;
        vkResetFences(device, 1, &fence);
        r = vkQueueSubmit(queue, 1, &submit, fence);
        if (r == VK_SUCCESS)
            r = vkWaitForFences(device, 1, &fence, VK_TRUE, 20000000000ull);
    }
    vkFreeCommandBuffers(device, pool, 1, &cb);
    return r;
}

/* What follows waits for the acceleration structure builds before it. */
static void build_barrier(VkCommandBuffer cb, VkPipelineStageFlags dst)
{
    VkMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    b.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, dst, 0, 1, &b, 0, NULL, 0, NULL);
}

/* ---- The scene ----
 *
 * Bottom level "mesh": geometry 0, a grid of 4 x 4 squares over [-2, 2]^2
 * at z = 0, two triangles each with 16-bit indices, the even ones wound
 * counter-clockwise seen from +z and the odd ones clockwise, not opaque;
 * geometry 1, one opaque triangle of half-float vertices without indices,
 * which its transform moves to z = 0.5.
 * Bottom level "ball": one bounding box, [-0.5, 0.5]^3: the shader finds a
 * sphere of radius 0.5 in it.
 * Top level: the instances below. */

#define GRID 4
#define GRID_TRIANGLES (2 * GRID * GRID)
#define MESH_TRIANGLES (GRID_TRIANGLES + 1)

static float grid_vertices[(GRID + 1) * (GRID + 1)][4]; /* stride 16 */
static uint16_t grid_indices[GRID_TRIANGLES][3];
static const uint16_t big_vertices[3][4] = { /* -3, -3, 0; 3, -3, 0; 0, 3, 0 as half floats; stride 8 */
    { 0xc200, 0xc200, 0, 0 }, { 0x4200, 0xc200, 0, 0 }, { 0, 0x4200, 0, 0 },
};
static const float big_positions[3][3] = { { -3, -3, 0 }, { 3, -3, 0 }, { 0, 3, 0 } };
static const VkTransformMatrixKHR big_transform = { { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0.5f } } };

struct instance {
    float m[3][4]; /* object to world */
    uint32_t custom, mask, offset, flags;
    int blas; /* 0 the mesh, 1 the ball, -1 none (inactive) */
};

#define INSTANCES 6
static const struct instance scene_instances[INSTANCES] = {
    { { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 } }, 100, 0x01, 5, 0, 0 },
    /* a quarter turn about z at half the size, its facing flipped */
    { { { 0, -0.5f, 0, 6 }, { 0.5f, 0, 0, 0 }, { 0, 0, 0.5f, 2 } }, 101, 0x02, 7,
      VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR, 0 },
    { { { 1, 0, 0, -6 }, { 0, 1, 0, 0 }, { 0, 0, 1, 1 } }, 102, 0x04, 9, 0, 1 },
    /* in front of everything, of no acceleration structure: inactive */
    { { { 9, 0, 0, 0 }, { 0, 9, 0, 0 }, { 0, 0, 9, -2 } }, 103, 0xff, 3, 0, -1 },
    { { { 2, 0, 0, 0 }, { 0, 2, 0, 6 }, { 0, 0, 2, 0 } }, 104, 0x08, 11, VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR, 1 },
    /* behind the first, three times its size, never opaque and never culled */
    { { { 3, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 0, 3, 3 } }, 105, 0x01, 13,
      VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR | VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR, 0 },
};

static void scene_init(void)
{
    int x, y;
    for (y = 0; y <= GRID; y++)
        for (x = 0; x <= GRID; x++) {
            float *v = grid_vertices[y * (GRID + 1) + x];
            v[0] = (float)x - GRID / 2;
            v[1] = (float)y - GRID / 2;
            v[2] = 0;
            v[3] = 77; /* unread */
        }
    for (y = 0; y < GRID; y++)
        for (x = 0; x < GRID; x++) {
            uint16_t v00 = (uint16_t)(y * (GRID + 1) + x), v10 = (uint16_t)(v00 + 1);
            uint16_t v01 = (uint16_t)(v00 + GRID + 1), v11 = (uint16_t)(v01 + 1);
            uint16_t *t = grid_indices[2 * (y * GRID + x)];
            t[0] = v00, t[1] = v10, t[2] = v01;
            t[3] = v10, t[4] = v01, t[5] = v11;
        }
}

/* Triangle `primitive` of the mesh's geometry g, as built. */
static void mesh_triangle(uint32_t g, uint32_t primitive, double v[3][3])
{
    int k, c;
    for (k = 0; k < 3; k++)
        for (c = 0; c < 3; c++)
            v[k][c] = g ? big_positions[k][c] + big_transform.matrix[c][3] : grid_vertices[grid_indices[primitive][k]][c];
}

/* ---- The host's trace ---- */

struct ray {
    float origin[3], tmin, direction[3], tmax;
};

struct hit {
    float t;
    uint32_t primitive, instance, custom;
    float bary[2];
    uint32_t type, front;
    float translation[3];
    uint32_t geometry;
    float object_origin[3];
    uint32_t offset;
};

#define RAY_OPAQUE 0x1u
#define RAY_NO_OPAQUE 0x2u
#define RAY_CULL_BACK 0x10u
#define RAY_CULL_FRONT 0x20u
#define RAY_CULL_OPAQUE 0x40u
#define RAY_CULL_NO_OPAQUE 0x80u
#define RAY_SKIP_TRIANGLES 0x100u
#define RAY_SKIP_AABBS 0x200u

static void cross(const double a[3], const double b[3], double r[3])
{
    r[0] = a[1] * b[2] - a[2] * b[1];
    r[1] = a[2] * b[0] - a[0] * b[2];
    r[2] = a[0] * b[1] - a[1] * b[0];
}

static double dot(const double a[3], const double b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/* Whether a ray of the flags takes geometry that is opaque or not, with an
 * instance's flags: its own opacity, in *opaque. */
static bool takes(uint32_t flags, uint32_t instance_flags, bool *opaque)
{
    if (instance_flags & VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR)
        *opaque = true;
    if (instance_flags & VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR)
        *opaque = false;
    if (flags & RAY_OPAQUE)
        *opaque = true;
    if (flags & RAY_NO_OPAQUE)
        *opaque = false;
    return !((flags & RAY_CULL_OPAQUE) && *opaque) && !((flags & RAY_CULL_NO_OPAQUE) && !*opaque);
}

/* The ray's closest hit that the shader commits, in *h (type 0: none), with
 * the scene's instances. False: too close to call (the ray passes a
 * triangle's edge, grazes the sphere, or two hits are as far): the GPU may
 * rightly find otherwise. */
static bool host_trace(const struct instance *instances, const struct ray *r, uint32_t flags, uint32_t mask, uint32_t mode,
                       struct hit *h)
{
    double best = r->tmax, second = 1e30;
    bool sure = true;
    uint32_t i, g, p;
    int c;

    memset(h, 0, sizeof(*h));
    h->t = -1;
    for (i = 0; i < INSTANCES; i++) {
        const struct instance *in = &instances[i];
        double m[3][3], inv[3][3], det, o[3], d[3], rel[3];
        if (in->blas < 0 || !(in->mask & mask))
            continue;
        for (c = 0; c < 3; c++) {
            m[c][0] = in->m[c][0], m[c][1] = in->m[c][1], m[c][2] = in->m[c][2];
            rel[c] = (double)r->origin[c] - in->m[c][3];
        }
        det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
              m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
        inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
        inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
        inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
        inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
        inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
        inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
        inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
        inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
        for (c = 0; c < 3; c++) {
            o[c] = inv[c][0] * rel[0] + inv[c][1] * rel[1] + inv[c][2] * rel[2];
            d[c] = inv[c][0] * r->direction[0] + inv[c][1] * r->direction[1] + inv[c][2] * r->direction[2];
        }
        if (in->blas == 1) { /* the sphere */
            double a = dot(d, d), b = dot(o, d), cc = dot(o, o) - 0.25, disc = b * b - a * cc, t;
            bool opaque = false;
            if ((flags & RAY_SKIP_AABBS) || !takes(flags, in->flags, &opaque))
                continue;
            if (fabs(disc) < 1e-3 * a)
                sure = false;
            if (disc < 0)
                continue;
            t = (-b - sqrt(disc)) / a;
            if (t < r->tmin || t > r->tmax)
                continue;
            if (t < best) {
                second = best;
                best = t;
                h->type = 2;
                h->t = (float)t;
                h->primitive = h->geometry = h->front = 0;
                h->bary[0] = h->bary[1] = 0;
            } else if (t < second) {
                second = t;
            } else {
                continue;
            }
            if (best == t) {
                h->instance = i;
                h->custom = in->custom;
                h->offset = in->offset;
                for (c = 0; c < 3; c++) {
                    h->translation[c] = in->m[c][3];
                    h->object_origin[c] = (float)o[c];
                }
            }
            continue;
        }
        if (flags & RAY_SKIP_TRIANGLES)
            continue;
        for (g = 0; g < 2; g++)
            for (p = 0; p < (g ? 1u : (uint32_t)GRID_TRIANGLES); p++) {
                double v[3][3], e1[3], e2[3], n[3], pv[3], tv[3], qv[3], dt, u, w, t;
                bool opaque = g == 1, front;
                if (!takes(flags, in->flags, &opaque))
                    continue;
                mesh_triangle(g, p, v);
                for (c = 0; c < 3; c++) {
                    e1[c] = v[1][c] - v[0][c];
                    e2[c] = v[2][c] - v[0][c];
                    tv[c] = o[c] - v[0][c];
                }
                cross(d, e2, pv);
                dt = dot(e1, pv);
                if (fabs(dt) < 1e-12)
                    continue;
                u = dot(tv, pv) / dt;
                cross(tv, e1, qv);
                w = dot(d, qv) / dt;
                t = dot(e2, qv) / dt;
                if (t <= r->tmin || t >= r->tmax)
                    continue;
                if (fabs(u) < 1e-3 || fabs(w) < 1e-3 || fabs(1 - u - w) < 1e-3)
                    sure = false;
                if (u < 0 || w < 0 || u + w > 1)
                    continue;
                /* front: the vertices counter-clockwise seen from the ray's origin */
                cross(e1, e2, n);
                front = dot(n, d) < 0;
                if (in->flags & VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR)
                    front = !front;
                if (!(in->flags & VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR) &&
                    (((flags & RAY_CULL_BACK) && !front) || ((flags & RAY_CULL_FRONT) && front)))
                    continue;
                if (!opaque && mode == 1 && (p & 1)) /* the shader leaves it */
                    continue;
                if (t >= best) {
                    if (t < second)
                        second = t;
                    continue;
                }
                second = best;
                best = t;
                h->type = 1;
                h->t = (float)t;
                h->primitive = p;
                h->geometry = g;
                h->front = front;
                h->bary[0] = (float)u;
                h->bary[1] = (float)w;
                h->instance = i;
                h->custom = in->custom;
                h->offset = in->offset;
                for (c = 0; c < 3; c++) {
                    h->translation[c] = in->m[c][3];
                    h->object_origin[c] = (float)o[c];
                }
            }
    }
    if (second - best < 1e-3 * (1 + best))
        sure = false;
    return sure;
}

static bool close_to(float a, float b, float tolerance)
{
    return fabsf(a - b) <= tolerance * (1 + fabsf(b));
}

/* ---- Acceleration structures ---- */

static struct accel make_accel(VkAccelerationStructureTypeKHR type, VkDeviceSize size)
{
    VkAccelerationStructureCreateInfoKHR info;
    VkAccelerationStructureDeviceAddressInfoKHR address;
    struct accel a;
    memset(&a, 0, sizeof(a));
    /* in the middle of a buffer of private memory: an offset of 256 bytes */
    a.storage = make_buf(size + 512, private_type);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    info.buffer = a.storage.buffer;
    info.offset = 256;
    info.size = size;
    info.type = type;
    if (vkCreateAccelerationStructureKHR(device, &info, NULL, &a.handle) != VK_SUCCESS)
        return a;
    memset(&address, 0, sizeof(address));
    address.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    address.accelerationStructure = a.handle;
    a.address = vkGetAccelerationStructureDeviceAddressKHR(device, &address);
    return a;
}

static void free_accel(struct accel *a)
{
    vkDestroyAccelerationStructureKHR(device, a->handle, NULL);
    free_buf(&a->storage);
}

/* The scene's buffers and acceleration structures. */
static struct scene {
    struct buf data;     /* vertices, indices, the transform, the box */
    struct buf instances;
    struct buf scratch;
    struct accel mesh, ball, top;
    VkDeviceSize top_update_scratch;
    struct buf rays, hits;
    VkDescriptorSet set;
} scene;

#define DATA_GRID_VERTICES 0u
#define DATA_GRID_INDICES 1024u
#define DATA_BIG_VERTICES 2048u
#define DATA_TRANSFORM 2304u
#define DATA_BOX 2560u
#define RAYS 4096u

static void mesh_geometries(VkAccelerationStructureGeometryKHR g[2])
{
    memset(g, 0, 2 * sizeof(g[0]));
    g[0].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    g[0].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    g[0].geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    g[0].geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    g[0].geometry.triangles.vertexData.deviceAddress = scene.data.address + DATA_GRID_VERTICES;
    g[0].geometry.triangles.vertexStride = 16;
    g[0].geometry.triangles.maxVertex = (GRID + 1) * (GRID + 1) - 1;
    g[0].geometry.triangles.indexType = VK_INDEX_TYPE_UINT16;
    g[0].geometry.triangles.indexData.deviceAddress = scene.data.address + DATA_GRID_INDICES;
    g[1].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    g[1].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    g[1].flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    g[1].geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    g[1].geometry.triangles.vertexFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    g[1].geometry.triangles.vertexData.deviceAddress = scene.data.address + DATA_BIG_VERTICES;
    g[1].geometry.triangles.vertexStride = 8;
    g[1].geometry.triangles.maxVertex = 2;
    g[1].geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
    g[1].geometry.triangles.transformData.deviceAddress = scene.data.address + DATA_TRANSFORM;
}

static void ball_geometry(VkAccelerationStructureGeometryKHR *g)
{
    memset(g, 0, sizeof(*g));
    g->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    g->geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    g->geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    g->geometry.aabbs.data.deviceAddress = scene.data.address + DATA_BOX;
    g->geometry.aabbs.stride = 32;
}

static void top_geometry(VkAccelerationStructureGeometryKHR *g)
{
    memset(g, 0, sizeof(*g));
    g->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    g->geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    g->geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    g->geometry.instances.data.deviceAddress = scene.instances.address;
}

static void build_info(VkAccelerationStructureBuildGeometryInfoKHR *info, VkAccelerationStructureTypeKHR type,
                       VkBuildAccelerationStructureFlagsKHR flags, const VkAccelerationStructureGeometryKHR *g,
                       uint32_t count)
{
    memset(info, 0, sizeof(*info));
    info->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info->type = type;
    info->flags = flags;
    info->mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info->geometryCount = count;
    info->pGeometries = g;
}

/* Writes the scene's instances, of bottom levels at the addresses. */
static void write_instances(const struct instance *instances, VkDeviceAddress mesh, VkDeviceAddress ball)
{
    VkAccelerationStructureInstanceKHR out[INSTANCES];
    uint32_t i;
    memset(out, 0, sizeof(out));
    for (i = 0; i < INSTANCES; i++) {
        memcpy(out[i].transform.matrix, instances[i].m, sizeof(instances[i].m));
        out[i].instanceCustomIndex = instances[i].custom;
        out[i].mask = instances[i].mask;
        out[i].instanceShaderBindingTableRecordOffset = instances[i].offset;
        out[i].flags = instances[i].flags;
        out[i].accelerationStructureReference = instances[i].blas == 0 ? mesh : instances[i].blas == 1 ? ball : 0;
    }
    memcpy(scene.instances.data, out, sizeof(out));
}

static void write_descriptors(VkAccelerationStructureKHR top)
{
    VkWriteDescriptorSetAccelerationStructureKHR accel;
    VkDescriptorBufferInfo buffers[2];
    VkWriteDescriptorSet writes[2];
    memset(&accel, 0, sizeof(accel));
    accel.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    accel.accelerationStructureCount = 1;
    accel.pAccelerationStructures = &top;
    buffers[0].buffer = scene.rays.buffer;
    buffers[0].offset = 0;
    buffers[0].range = VK_WHOLE_SIZE;
    buffers[1].buffer = scene.hits.buffer;
    buffers[1].offset = 0;
    buffers[1].range = VK_WHOLE_SIZE;
    memset(writes, 0, sizeof(writes));
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &accel;
    writes[0].dstSet = scene.set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = scene.set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 2;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = buffers;
    vkUpdateDescriptorSets(device, 2, writes, 0, NULL);
}

static uint32_t lcg(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state >> 8;
}

static float unit(uint32_t *state)
{
    return (float)lcg(state) / 16777216.0f;
}

static void scene_create(void)
{
    static const uint32_t mesh_counts[2] = { GRID_TRIANGLES, 1 }, one = 1, instances = INSTANCES;
    static const float box[8] = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f, 123, 456 };
    VkAccelerationStructureGeometryKHR mesh_g[2], ball_g, top_g;
    VkAccelerationStructureBuildGeometryInfoKHR infos[2], top_info;
    VkAccelerationStructureBuildSizesInfoKHR mesh_sizes, ball_sizes, top_sizes;
    VkAccelerationStructureBuildRangeInfoKHR mesh_ranges[2], ball_range, top_range;
    const VkAccelerationStructureBuildRangeInfoKHR *ranges[2], *top_ranges;
    VkDescriptorSetAllocateInfo set_alloc;
    VkCommandBuffer cb;
    struct ray *rays;
    uint32_t i, seed = 12345;

    scene_init();
    scene.data = make_buf(4096, shared_type);
    scene.instances = make_buf(sizeof(VkAccelerationStructureInstanceKHR) * INSTANCES, shared_type);
    scene.rays = make_buf(sizeof(struct ray) * RAYS, shared_type);
    scene.hits = make_buf(sizeof(struct hit) * RAYS, shared_type);
    REQUIRE(scene.data.data && scene.instances.data && scene.rays.data && scene.hits.data);
    memcpy(scene.data.data + DATA_GRID_VERTICES, grid_vertices, sizeof(grid_vertices));
    memcpy(scene.data.data + DATA_GRID_INDICES, grid_indices, sizeof(grid_indices));
    memcpy(scene.data.data + DATA_BIG_VERTICES, big_vertices, sizeof(big_vertices));
    memcpy(scene.data.data + DATA_TRANSFORM, &big_transform, sizeof(big_transform));
    memcpy(scene.data.data + DATA_BOX, box, sizeof(box));
    rays = (struct ray *)(void *)scene.rays.data;
    for (i = 0; i < RAYS; i++) {
        rays[i].origin[0] = -9 + 18 * unit(&seed);
        rays[i].origin[1] = -5 + 14 * unit(&seed);
        rays[i].origin[2] = -5;
        rays[i].tmin = 0;
        rays[i].direction[0] = 0.2f * unit(&seed) - 0.1f;
        rays[i].direction[1] = 0.2f * unit(&seed) - 0.1f;
        rays[i].direction[2] = 1;
        rays[i].tmax = 100;
    }

    mesh_geometries(mesh_g);
    ball_geometry(&ball_g);
    top_geometry(&top_g);
    build_info(&infos[0], VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
               VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
               mesh_g, 2);
    build_info(&infos[1], VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, 0, &ball_g, 1);
    build_info(&top_info, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
               &top_g, 1);
    memset(&mesh_sizes, 0, sizeof(mesh_sizes));
    mesh_sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    ball_sizes = top_sizes = mesh_sizes;
    vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &infos[0], mesh_counts,
                                            &mesh_sizes);
    vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &infos[1], &one,
                                            &ball_sizes);
    vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &top_info, &instances,
                                            &top_sizes);
    CHECK(mesh_sizes.accelerationStructureSize && mesh_sizes.buildScratchSize);
    CHECK(ball_sizes.accelerationStructureSize && ball_sizes.buildScratchSize);
    CHECK(top_sizes.accelerationStructureSize && top_sizes.buildScratchSize && top_sizes.updateScratchSize);
    scene.top_update_scratch = top_sizes.updateScratchSize;
    scene.mesh = make_accel(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, mesh_sizes.accelerationStructureSize);
    scene.ball = make_accel(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, ball_sizes.accelerationStructureSize);
    scene.top = make_accel(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, top_sizes.accelerationStructureSize);
    REQUIRE(scene.mesh.handle && scene.ball.handle && scene.top.handle);
    /* an acceleration structure's address is its storage's, at its offset */
    CHECK(scene.mesh.address == scene.mesh.storage.address + 256);
    scene.scratch = make_buf(mesh_sizes.buildScratchSize + ball_sizes.buildScratchSize + top_sizes.buildScratchSize +
                                 top_sizes.updateScratchSize + 1024,
                             private_type);
    write_instances(scene_instances, scene.mesh.address, scene.ball.address);

    memset(&set_alloc, 0, sizeof(set_alloc));
    set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_alloc.descriptorPool = descriptor_pool;
    set_alloc.descriptorSetCount = 1;
    set_alloc.pSetLayouts = &set_layout;
    REQUIRE_VK(vkAllocateDescriptorSets(device, &set_alloc, &scene.set));
    write_descriptors(scene.top.handle);

    /* both bottom levels in one command, then the top level behind a barrier */
    infos[0].dstAccelerationStructure = scene.mesh.handle;
    infos[0].scratchData.deviceAddress = scene.scratch.address;
    infos[1].dstAccelerationStructure = scene.ball.handle;
    infos[1].scratchData.deviceAddress = scene.scratch.address + mesh_sizes.buildScratchSize;
    top_info.dstAccelerationStructure = scene.top.handle;
    top_info.scratchData.deviceAddress = infos[1].scratchData.deviceAddress + ball_sizes.buildScratchSize;
    memset(mesh_ranges, 0, sizeof(mesh_ranges));
    mesh_ranges[0].primitiveCount = GRID_TRIANGLES;
    mesh_ranges[1].primitiveCount = 1;
    memset(&ball_range, 0, sizeof(ball_range));
    ball_range.primitiveCount = 1;
    memset(&top_range, 0, sizeof(top_range));
    top_range.primitiveCount = INSTANCES;
    ranges[0] = mesh_ranges;
    ranges[1] = &ball_range;
    top_ranges = &top_range;
    cb = begin();
    vkCmdBuildAccelerationStructuresKHR(cb, 2, infos, ranges);
    build_barrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
    vkCmdBuildAccelerationStructuresKHR(cb, 1, &top_info, &top_ranges);
    CHECK_VK(finish(cb), VK_SUCCESS);
}

static void scene_destroy(void)
{
    free_accel(&scene.top);
    free_accel(&scene.ball);
    free_accel(&scene.mesh);
    free_buf(&scene.scratch);
    free_buf(&scene.hits);
    free_buf(&scene.rays);
    free_buf(&scene.instances);
    free_buf(&scene.data);
}

/* Queries the scene's rays with the shader, and compares what it found with
 * the host's trace of the instances. Returns the rays compared; *kinds gets
 * bit t for each committed type t found. */
static uint32_t trace_and_compare(const struct instance *instances, uint32_t flags, uint32_t mask, uint32_t mode,
                                  uint32_t *kinds)
{
    const struct ray *rays = (const struct ray *)(const void *)scene.rays.data;
    const struct hit *hits = (const struct hit *)(const void *)scene.hits.data;
    uint32_t push[4], i, compared = 0, wrong = 0;
    VkCommandBuffer cb = begin();
    int c;

    push[0] = RAYS;
    push[1] = flags;
    push[2] = mask;
    push[3] = mode;
    memset(scene.hits.data, 0xee, sizeof(struct hit) * RAYS);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &scene.set, 0, NULL);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    vkCmdDispatch(cb, RAYS / 64, 1, 1);
    if (finish(cb) != VK_SUCCESS) {
        PVK_TEST_FAIL("trace: the dispatch failed");
        return 0;
    }
    for (i = 0; i < RAYS; i++) {
        struct hit want;
        const struct hit *got = &hits[i];
        bool same;
        if (!host_trace(instances, &rays[i], flags, mask, mode, &want))
            continue;
        compared++;
        *kinds |= 1u << want.type;
        same = got->type == want.type;
        if (same && want.type) {
            same = close_to(got->t, want.t, 1e-4f) && got->primitive == want.primitive && got->instance == want.instance &&
                   got->custom == want.custom && got->geometry == want.geometry && got->offset == want.offset &&
                   got->front == want.front && fabsf(got->bary[0] - want.bary[0]) < 2e-3f &&
                   fabsf(got->bary[1] - want.bary[1]) < 2e-3f;
            for (c = 0; c < 3; c++)
                same = same && got->translation[c] == want.translation[c] &&
                       close_to(got->object_origin[c], want.object_origin[c], 1e-4f);
        }
        if (!same && wrong++ < 4)
            PVK_TEST_FAIL("flags 0x%x mask 0x%x mode %u ray %u: type %u t %g primitive %u instance %u custom %u geometry %u "
                          "offset %u front %u bary %g %g, expected type %u t %g primitive %u instance %u custom %u geometry "
                          "%u offset %u front %u bary %g %g",
                          flags, mask, mode, i, got->type, (double)got->t, got->primitive, got->instance, got->custom,
                          got->geometry, got->offset, got->front, (double)got->bary[0], (double)got->bary[1], want.type,
                          (double)want.t, want.primitive, want.instance, want.custom, want.geometry, want.offset, want.front,
                          (double)want.bary[0], (double)want.bary[1]);
    }
    if (wrong > 4)
        PVK_TEST_FAIL("flags 0x%x mask 0x%x mode %u: %u rays differ", flags, mask, mode, wrong);
    return compared;
}

/* ---- Tests ---- */

/* The extensions, their features, properties and commands. */
static void test_features(void)
{
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel;
    VkPhysicalDeviceRayQueryFeaturesKHR ray_query;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR properties;
    VkPhysicalDeviceFeatures2 features;
    VkPhysicalDeviceProperties2 properties2;
    VkFormatProperties format;
    VkExtensionProperties extensions[256];
    VkDeferredOperationKHR operation = VK_NULL_HANDLE;
    uint32_t count = 256, i, found = 0;

    CHECK_VK(vkEnumerateDeviceExtensionProperties(physical_device, NULL, &count, extensions), VK_SUCCESS);
    for (i = 0; i < count; i++)
        if (!strcmp(extensions[i].extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) ||
            !strcmp(extensions[i].extensionName, VK_KHR_RAY_QUERY_EXTENSION_NAME) ||
            !strcmp(extensions[i].extensionName, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME))
            found++;
    CHECK(found == 3);

    memset(&ray_query, 0, sizeof(ray_query));
    ray_query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    memset(&accel, 0, sizeof(accel));
    accel.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accel.pNext = &ray_query;
    memset(&features, 0, sizeof(features));
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &accel;
    vkGetPhysicalDeviceFeatures2(physical_device, &features);
    CHECK(accel.accelerationStructure && ray_query.rayQuery);
    CHECK(accel.descriptorBindingAccelerationStructureUpdateAfterBind);
    /* what Metal has nothing for is not reported */
    CHECK(!accel.accelerationStructureCaptureReplay && !accel.accelerationStructureIndirectBuild &&
          !accel.accelerationStructureHostCommands);

    memset(&properties, 0, sizeof(properties));
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    memset(&properties2, 0, sizeof(properties2));
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &properties;
    vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    CHECK(properties.maxGeometryCount >= (1u << 24) - 1 && properties.maxInstanceCount >= (1u << 24) - 1);
    CHECK(properties.maxPrimitiveCount >= (1u << 28) - 1);
    CHECK(properties.maxPerStageDescriptorAccelerationStructures >= 16 &&
          properties.maxDescriptorSetAccelerationStructures >= 16);
    CHECK(properties.maxPerStageDescriptorUpdateAfterBindAccelerationStructures >= 500000 &&
          properties.maxDescriptorSetUpdateAfterBindAccelerationStructures >= 500000);
    CHECK(properties.minAccelerationStructureScratchOffsetAlignment >= 1 &&
          properties.minAccelerationStructureScratchOffsetAlignment <= 256);

    vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R32G32B32_SFLOAT, &format);
    CHECK(format.bufferFeatures & VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR);
    vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16G16B16A16_SNORM, &format);
    CHECK(format.bufferFeatures & VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR);
    vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R8G8B8A8_UNORM, &format);
    CHECK(!(format.bufferFeatures & VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR));

    CHECK(vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR") != NULL);
    CHECK(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR") != NULL);
    CHECK(vkGetDeviceProcAddr(device, "vkCreateDeferredOperationKHR") != NULL);

    /* an operation no command defers on is complete */
    CHECK_VK(vkCreateDeferredOperationKHR(device, NULL, &operation), VK_SUCCESS);
    CHECK(vkGetDeferredOperationMaxConcurrencyKHR(device, operation) == 0);
    CHECK_VK(vkDeferredOperationJoinKHR(device, operation), VK_SUCCESS);
    CHECK_VK(vkGetDeferredOperationResultKHR(device, operation), VK_SUCCESS);
    vkDestroyDeferredOperationKHR(device, operation, NULL);
}

/* Rays through the scene, with ray flags, cull masks and candidates the
 * shader leaves: hit distances, primitives, instances, custom indices,
 * record offsets, barycentrics, facing and transforms as the host finds
 * them. */
static void test_ray_queries(void)
{
    static const struct {
        uint32_t flags, mask, mode;
    } cases[] = {
        { 0, 0xff, 0 },
        { 0, 0xff, 1 },
        { RAY_OPAQUE, 0xff, 1 },
        { RAY_NO_OPAQUE, 0xff, 1 },
        { RAY_CULL_BACK, 0xff, 0 },
        { RAY_CULL_FRONT, 0xff, 0 },
        { RAY_CULL_OPAQUE, 0xff, 0 },
        { RAY_CULL_NO_OPAQUE, 0xff, 0 },
        { RAY_SKIP_TRIANGLES, 0xff, 0 },
        { RAY_SKIP_AABBS, 0xff, 1 },
        { 0, 0x01, 0 },
        { RAY_CULL_BACK | RAY_NO_OPAQUE, 0x06, 1 },
        { 0, 0x00, 0 },
    };
    uint32_t i, kinds = 0;
    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        uint32_t compared = trace_and_compare(scene_instances, cases[i].flags, cases[i].mask, cases[i].mode, &kinds);
        CHECK(compared > RAYS * 3 / 4);
    }
    CHECK(kinds == 7); /* misses, triangles and the sphere */
}

/* An update of the top level in place with an instance moved and another
 * disabled, then back: the hits follow. */
static void test_update(void)
{
    struct instance moved[INSTANCES];
    VkAccelerationStructureGeometryKHR top_g;
    VkAccelerationStructureBuildGeometryInfoKHR info;
    VkAccelerationStructureBuildRangeInfoKHR range;
    const VkAccelerationStructureBuildRangeInfoKHR *ranges = &range;
    uint32_t kinds = 0, pass;

    memcpy(moved, scene_instances, sizeof(moved));
    moved[0].m[0][3] = 0.75f;
    moved[0].m[1][3] = -0.25f;
    moved[2].m[2][3] = 2;
    top_geometry(&top_g);
    build_info(&info, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
               &top_g, 1);
    info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    info.srcAccelerationStructure = info.dstAccelerationStructure = scene.top.handle;
    info.scratchData.deviceAddress = scene.scratch.address;
    memset(&range, 0, sizeof(range));
    range.primitiveCount = INSTANCES;
    for (pass = 0; pass < 2; pass++) {
        const struct instance *instances = pass ? scene_instances : moved;
        VkCommandBuffer cb = begin();
        write_instances(instances, scene.mesh.address, scene.ball.address);
        vkCmdBuildAccelerationStructuresKHR(cb, 1, &info, &ranges);
        CHECK_VK(finish(cb), VK_SUCCESS);
        CHECK(trace_and_compare(instances, 0, 0xff, 0, &kinds) > RAYS * 3 / 4);
    }
}

/* The mesh's compacted size from a query, a compacting copy of it into an
 * acceleration structure of that size, a top level of the copy and a clone
 * of that top level: rays find the same in the clone. */
static void test_compact_and_copy(void)
{
    VkQueryPoolCreateInfo pool_info;
    VkQueryPool queries = VK_NULL_HANDLE;
    VkCopyAccelerationStructureInfoKHR copy;
    VkAccelerationStructureGeometryKHR top_g;
    VkAccelerationStructureBuildGeometryInfoKHR info;
    VkAccelerationStructureBuildRangeInfoKHR range;
    const VkAccelerationStructureBuildRangeInfoKHR *ranges = &range;
    VkAccelerationStructureKHR sized[2];
    struct accel compact, clone;
    VkDeviceSize sizes[2] = { 0, 0 };
    VkCommandBuffer cb;
    uint32_t kinds = 0;

    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    pool_info.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    pool_info.queryCount = 2;
    REQUIRE_VK(vkCreateQueryPool(device, &pool_info, NULL, &queries));
    sized[0] = scene.mesh.handle;
    sized[1] = scene.top.handle;
    cb = begin();
    vkCmdResetQueryPool(cb, queries, 0, 2);
    vkCmdWriteAccelerationStructuresPropertiesKHR(cb, 2, sized, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                                                  queries, 0);
    CHECK_VK(finish(cb), VK_SUCCESS);
    CHECK_VK(vkGetQueryPoolResults(device, queries, 0, 2, sizeof(sizes), sizes, sizeof(sizes[0]),
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
             VK_SUCCESS);
    CHECK(sizes[0] > 0 && sizes[0] <= scene.mesh.storage.size);
    /* a top level's holds its instances' record offsets too */
    CHECK(sizes[1] >= 16 + 4 * INSTANCES && sizes[1] <= scene.top.storage.size);
    vkDestroyQueryPool(device, queries, NULL);

    compact = make_accel(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizes[0]);
    clone = make_accel(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, scene.top.storage.size - 512);
    REQUIRE(compact.handle && clone.handle);
    memset(&copy, 0, sizeof(copy));
    copy.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
    copy.src = scene.mesh.handle;
    copy.dst = compact.handle;
    copy.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
    top_geometry(&top_g);
    build_info(&info, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
               &top_g, 1);
    info.dstAccelerationStructure = scene.top.handle;
    info.scratchData.deviceAddress = scene.scratch.address;
    memset(&range, 0, sizeof(range));
    range.primitiveCount = INSTANCES;
    write_instances(scene_instances, compact.address, scene.ball.address);
    cb = begin();
    vkCmdCopyAccelerationStructureKHR(cb, &copy);
    build_barrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
    vkCmdBuildAccelerationStructuresKHR(cb, 1, &info, &ranges);
    build_barrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
    copy.src = scene.top.handle;
    copy.dst = clone.handle;
    copy.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR;
    vkCmdCopyAccelerationStructureKHR(cb, &copy);
    CHECK_VK(finish(cb), VK_SUCCESS);

    write_descriptors(clone.handle);
    CHECK(trace_and_compare(scene_instances, 0, 0xff, 1, &kinds) > RAYS * 3 / 4);
    CHECK(kinds == 7);

    /* the scene as it was */
    write_descriptors(scene.top.handle);
    write_instances(scene_instances, scene.mesh.address, scene.ball.address);
    cb = begin();
    vkCmdBuildAccelerationStructuresKHR(cb, 1, &info, &ranges);
    CHECK_VK(finish(cb), VK_SUCCESS);
    free_accel(&clone);
    free_accel(&compact);
}

int main(void)
{
    setup();
    if (pvk_test_failures || !supported) {
        printf("%s\n", pvk_test_failures ? "FAIL setup" : "SKIP: the GPU does not trace rays");
        teardown();
        return TEST_RESULT();
    }
    scene_create();
    if (!pvk_test_failures) {
        RUN(test_features);
        RUN(test_ray_queries);
        RUN(test_update);
        RUN(test_compact_and_copy);
    }
    scene_destroy();
    teardown();
    return TEST_RESULT();
}
