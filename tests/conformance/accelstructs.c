/* Chapters "Acceleration Structures" and "Ray Traversal", with the features
 * and limits VK_KHR_acceleration_structure and VK_KHR_ray_query require and
 * the deferred operations of VK_KHR_deferred_host_operations: device builds
 * of triangle, box and instance acceleration structures, updates, compacting
 * and cloning copies, and what a ray query of a compute shader commits,
 * compared ray by ray with values worked out on the host.
 *
 * The scene is flat on purpose, so the expected values are exact arithmetic:
 * every instance is a unit square (two triangles) or a unit box, scaled and
 * moved, and every ray goes straight down.
 *
 * The tests skip on a device without the three extensions. Their commands
 * are resolved with vkGetDeviceProcAddr (a loader need not export them).
 *
 * Shaders: GLSL in the comments, compiled with
 *   glslangValidator -V --target-env vulkan1.2 -x */
#include "ct.h"

#include <stdio.h>

#define RT_COMMANDS                                                                                               \
    "vkGetPhysicalDeviceFeatures2 vkCreateDescriptorSetLayout vkDestroyDescriptorSetLayout "                      \
    "vkCreateDescriptorPool vkDestroyDescriptorPool vkAllocateDescriptorSets vkUpdateDescriptorSets "             \
    "vkCreatePipelineLayout vkDestroyPipelineLayout vkCreateShaderModule vkDestroyShaderModule "                  \
    "vkCreateComputePipelines vkDestroyPipeline vkCmdBindPipeline vkCmdBindDescriptorSets vkCmdPushConstants "    \
    "vkCmdDispatch vkCmdPipelineBarrier vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements "           \
    "vkAllocateMemory vkBindBufferMemory vkMapMemory vkFreeMemory vkGetBufferDeviceAddress "                      \
    "vkGetAccelerationStructureBuildSizesKHR vkCreateAccelerationStructureKHR "                                   \
    "vkDestroyAccelerationStructureKHR vkGetAccelerationStructureDeviceAddressKHR "                               \
    "vkCmdBuildAccelerationStructuresKHR "

/*
 * #version 460
 * #extension GL_EXT_ray_query : require
 * // Ray i of a side x side grid goes straight down from z = 100 over the point
 * // ((i % side) + 0.37, (i / side) + 0.29) * 0.25. Triangle candidates are
 * // confirmed, except those of primitive 1 in mode 1 (as an any-hit shader
 * // ignoring them would); a box candidate becomes a hit on the plane z = 0.5 of
 * // its object space. What the query committed is written out.
 * layout(local_size_x = 16) in;
 * struct Hit {
 *     float t; uint primitive; uint instance; uint custom;
 *     vec2 bary; uint type; uint candidates;
 *     float height; uint pad;
 * };
 * layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
 * layout(set = 0, binding = 1, std430) writeonly buffer Hits { Hit hits[]; };
 * layout(push_constant) uniform Push { uint flags; uint mask; uint mode; uint side; } pc;
 * void main()
 * {
 *     uint i = gl_GlobalInvocationID.x;
 *     if (i >= pc.side * pc.side)
 *         return;
 *     vec3 origin = vec3((float(i % pc.side) + 0.37) * 0.25, (float(i / pc.side) + 0.29) * 0.25, 100.0);
 *     rayQueryEXT q;
 *     rayQueryInitializeEXT(q, tlas, pc.flags, pc.mask, origin, 0.0, vec3(0.0, 0.0, -1.0), 1000.0);
 *     Hit h;
 *     h.candidates = 0u;
 *     while (rayQueryProceedEXT(q)) {
 *         h.candidates++;
 *         if (rayQueryGetIntersectionTypeEXT(q, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
 *             if (pc.mode != 1u || rayQueryGetIntersectionPrimitiveIndexEXT(q, false) == 0)
 *                 rayQueryConfirmIntersectionEXT(q);
 *         } else {
 *             vec3 o = rayQueryGetIntersectionObjectRayOriginEXT(q, false);
 *             vec3 d = rayQueryGetIntersectionObjectRayDirectionEXT(q, false);
 *             rayQueryGenerateIntersectionEXT(q, (o.z - 0.5) / -d.z);
 *         }
 *     }
 *     h.t = -1.0;
 *     h.primitive = h.instance = h.custom = h.pad = 0u;
 *     h.bary = vec2(0.0);
 *     h.height = 0.0;
 *     h.type = rayQueryGetIntersectionTypeEXT(q, true);
 *     if (h.type != gl_RayQueryCommittedIntersectionNoneEXT) {
 *         h.t = rayQueryGetIntersectionTEXT(q, true);
 *         h.primitive = uint(rayQueryGetIntersectionPrimitiveIndexEXT(q, true));
 *         h.instance = uint(rayQueryGetIntersectionInstanceIdEXT(q, true));
 *         h.custom = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(q, true));
 *         h.height = rayQueryGetIntersectionObjectToWorldEXT(q, true)[3].z;
 *         if (h.type == gl_RayQueryCommittedIntersectionTriangleEXT)
 *             h.bary = rayQueryGetIntersectionBarycentricsEXT(q, true);
 *     }
 *     hits[i] = h;
 * }
 */
static const uint32_t spv_trace[] = {
    0x07230203, 0x00010500, 0x0008000b, 0x000000b1, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x00001178, 0x0006000a, 0x5f565053, 0x5f52484b, 0x5f796172, 0x72657571, 0x00000079, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x000a000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00000013, 0x0000003a,
    0x0000003d, 0x000000a9, 0x00060010, 0x00000004, 0x00000011, 0x00000010, 0x00000001, 0x00000001,
    0x00030003, 0x00000002, 0x000001cc, 0x00060004, 0x455f4c47, 0x725f5458, 0x715f7961, 0x79726575,
    0x00000000, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008, 0x00000069,
    0x00080005, 0x0000000b, 0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044,
    0x00040005, 0x00000011, 0x68737550, 0x00000000, 0x00050006, 0x00000011, 0x00000000, 0x67616c66,
    0x00000073, 0x00050006, 0x00000011, 0x00000001, 0x6b73616d, 0x00000000, 0x00050006, 0x00000011,
    0x00000002, 0x65646f6d, 0x00000000, 0x00050006, 0x00000011, 0x00000003, 0x65646973, 0x00000000,
    0x00030005, 0x00000013, 0x00006370, 0x00040005, 0x00000024, 0x6769726f, 0x00006e69, 0x00030005,
    0x0000003a, 0x00000071, 0x00040005, 0x0000003d, 0x73616c74, 0x00000000, 0x00030005, 0x0000004b,
    0x00746948, 0x00040006, 0x0000004b, 0x00000000, 0x00000074, 0x00060006, 0x0000004b, 0x00000001,
    0x6d697270, 0x76697469, 0x00000065, 0x00060006, 0x0000004b, 0x00000002, 0x74736e69, 0x65636e61,
    0x00000000, 0x00050006, 0x0000004b, 0x00000003, 0x74737563, 0x00006d6f, 0x00050006, 0x0000004b,
    0x00000004, 0x79726162, 0x00000000, 0x00050006, 0x0000004b, 0x00000005, 0x65707974, 0x00000000,
    0x00060006, 0x0000004b, 0x00000006, 0x646e6163, 0x74616469, 0x00007365, 0x00050006, 0x0000004b,
    0x00000007, 0x67696568, 0x00007468, 0x00040006, 0x0000004b, 0x00000008, 0x00646170, 0x00030005,
    0x0000004d, 0x00000068, 0x00030005, 0x0000006c, 0x0000006f, 0x00030005, 0x0000006e, 0x00000064,
    0x00030005, 0x000000a5, 0x00746948, 0x00040006, 0x000000a5, 0x00000000, 0x00000074, 0x00060006,
    0x000000a5, 0x00000001, 0x6d697270, 0x76697469, 0x00000065, 0x00060006, 0x000000a5, 0x00000002,
    0x74736e69, 0x65636e61, 0x00000000, 0x00050006, 0x000000a5, 0x00000003, 0x74737563, 0x00006d6f,
    0x00050006, 0x000000a5, 0x00000004, 0x79726162, 0x00000000, 0x00050006, 0x000000a5, 0x00000005,
    0x65707974, 0x00000000, 0x00060006, 0x000000a5, 0x00000006, 0x646e6163, 0x74616469, 0x00007365,
    0x00050006, 0x000000a5, 0x00000007, 0x67696568, 0x00007468, 0x00040006, 0x000000a5, 0x00000008,
    0x00646170, 0x00040005, 0x000000a7, 0x73746948, 0x00000000, 0x00050006, 0x000000a7, 0x00000000,
    0x73746968, 0x00000000, 0x00030005, 0x000000a9, 0x00000000, 0x00040047, 0x0000000b, 0x0000000b,
    0x0000001c, 0x00030047, 0x00000011, 0x00000002, 0x00050048, 0x00000011, 0x00000000, 0x00000023,
    0x00000000, 0x00050048, 0x00000011, 0x00000001, 0x00000023, 0x00000004, 0x00050048, 0x00000011,
    0x00000002, 0x00000023, 0x00000008, 0x00050048, 0x00000011, 0x00000003, 0x00000023, 0x0000000c,
    0x00040047, 0x0000003d, 0x00000021, 0x00000000, 0x00040047, 0x0000003d, 0x00000022, 0x00000000,
    0x00050048, 0x000000a5, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x000000a5, 0x00000001,
    0x00000023, 0x00000004, 0x00050048, 0x000000a5, 0x00000002, 0x00000023, 0x00000008, 0x00050048,
    0x000000a5, 0x00000003, 0x00000023, 0x0000000c, 0x00050048, 0x000000a5, 0x00000004, 0x00000023,
    0x00000010, 0x00050048, 0x000000a5, 0x00000005, 0x00000023, 0x00000018, 0x00050048, 0x000000a5,
    0x00000006, 0x00000023, 0x0000001c, 0x00050048, 0x000000a5, 0x00000007, 0x00000023, 0x00000020,
    0x00050048, 0x000000a5, 0x00000008, 0x00000023, 0x00000024, 0x00040047, 0x000000a6, 0x00000006,
    0x00000028, 0x00030047, 0x000000a7, 0x00000002, 0x00040048, 0x000000a7, 0x00000000, 0x00000019,
    0x00050048, 0x000000a7, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x000000a9, 0x00000019,
    0x00040047, 0x000000a9, 0x00000021, 0x00000001, 0x00040047, 0x000000a9, 0x00000022, 0x00000000,
    0x00040047, 0x000000b0, 0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
    0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000007,
    0x00000006, 0x00040017, 0x00000009, 0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001,
    0x00000009, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b, 0x00000006, 0x0000000c,
    0x00000000, 0x00040020, 0x0000000d, 0x00000001, 0x00000006, 0x0006001e, 0x00000011, 0x00000006,
    0x00000006, 0x00000006, 0x00000006, 0x00040020, 0x00000012, 0x00000009, 0x00000011, 0x0004003b,
    0x00000012, 0x00000013, 0x00000009, 0x00040015, 0x00000014, 0x00000020, 0x00000001, 0x0004002b,
    0x00000014, 0x00000015, 0x00000003, 0x00040020, 0x00000016, 0x00000009, 0x00000006, 0x00020014,
    0x0000001c, 0x00030016, 0x00000021, 0x00000020, 0x00040017, 0x00000022, 0x00000021, 0x00000003,
    0x00040020, 0x00000023, 0x00000007, 0x00000022, 0x0004002b, 0x00000021, 0x0000002a, 0x3ebd70a4,
    0x0004002b, 0x00000021, 0x0000002c, 0x3e800000, 0x0004002b, 0x00000021, 0x00000033, 0x3e947ae1,
    0x0004002b, 0x00000021, 0x00000036, 0x42c80000, 0x00021178, 0x00000038, 0x00040020, 0x00000039,
    0x00000006, 0x00000038, 0x0004003b, 0x00000039, 0x0000003a, 0x00000006, 0x000214dd, 0x0000003b,
    0x00040020, 0x0000003c, 0x00000000, 0x0000003b, 0x0004003b, 0x0000003c, 0x0000003d, 0x00000000,
    0x0004002b, 0x00000014, 0x0000003f, 0x00000000, 0x0004002b, 0x00000014, 0x00000042, 0x00000001,
    0x0004002b, 0x00000021, 0x00000046, 0x00000000, 0x0004002b, 0x00000021, 0x00000047, 0xbf800000,
    0x0006002c, 0x00000022, 0x00000048, 0x00000046, 0x00000046, 0x00000047, 0x0004002b, 0x00000021,
    0x00000049, 0x447a0000, 0x00040017, 0x0000004a, 0x00000021, 0x00000002, 0x000b001e, 0x0000004b,
    0x00000021, 0x00000006, 0x00000006, 0x00000006, 0x0000004a, 0x00000006, 0x00000006, 0x00000021,
    0x00000006, 0x00040020, 0x0000004c, 0x00000007, 0x0000004b, 0x0004002b, 0x00000014, 0x0000004e,
    0x00000006, 0x0003002a, 0x0000001c, 0x00000059, 0x0004002b, 0x00000014, 0x0000005e, 0x00000002,
    0x0004002b, 0x00000006, 0x00000061, 0x00000001, 0x0004002b, 0x00000006, 0x00000070, 0x00000002,
    0x00040020, 0x00000071, 0x00000007, 0x00000021, 0x0004002b, 0x00000021, 0x00000074, 0x3f000000,
    0x0004002b, 0x00000014, 0x0000007b, 0x00000008, 0x0004002b, 0x00000014, 0x00000080, 0x00000004,
    0x0005002c, 0x0000004a, 0x00000081, 0x00000046, 0x00000046, 0x00040020, 0x00000082, 0x00000007,
    0x0000004a, 0x0004002b, 0x00000014, 0x00000084, 0x00000007, 0x0004002b, 0x00000014, 0x00000086,
    0x00000005, 0x00030029, 0x0000001c, 0x00000087, 0x00040018, 0x0000009a, 0x00000022, 0x00000004,
    0x000b001e, 0x000000a5, 0x00000021, 0x00000006, 0x00000006, 0x00000006, 0x0000004a, 0x00000006,
    0x00000006, 0x00000021, 0x00000006, 0x0003001d, 0x000000a6, 0x000000a5, 0x0003001e, 0x000000a7,
    0x000000a6, 0x00040020, 0x000000a8, 0x0000000c, 0x000000a7, 0x0004003b, 0x000000a8, 0x000000a9,
    0x0000000c, 0x00040020, 0x000000ac, 0x0000000c, 0x000000a5, 0x0004002b, 0x00000006, 0x000000af,
    0x00000010, 0x0006002c, 0x00000009, 0x000000b0, 0x000000af, 0x00000061, 0x00000061, 0x00050036,
    0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007,
    0x00000008, 0x00000007, 0x0004003b, 0x00000023, 0x00000024, 0x00000007, 0x0004003b, 0x0000004c,
    0x0000004d, 0x00000007, 0x0004003b, 0x00000023, 0x0000006c, 0x00000007, 0x0004003b, 0x00000023,
    0x0000006e, 0x00000007, 0x00050041, 0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c, 0x0004003d,
    0x00000006, 0x0000000f, 0x0000000e, 0x0003003e, 0x00000008, 0x0000000f, 0x0004003d, 0x00000006,
    0x00000010, 0x00000008, 0x00050041, 0x00000016, 0x00000017, 0x00000013, 0x00000015, 0x0004003d,
    0x00000006, 0x00000018, 0x00000017, 0x00050041, 0x00000016, 0x00000019, 0x00000013, 0x00000015,
    0x0004003d, 0x00000006, 0x0000001a, 0x00000019, 0x00050084, 0x00000006, 0x0000001b, 0x00000018,
    0x0000001a, 0x000500ae, 0x0000001c, 0x0000001d, 0x00000010, 0x0000001b, 0x000300f7, 0x0000001f,
    0x00000000, 0x000400fa, 0x0000001d, 0x0000001e, 0x0000001f, 0x000200f8, 0x0000001e, 0x000100fd,
    0x000200f8, 0x0000001f, 0x0004003d, 0x00000006, 0x00000025, 0x00000008, 0x00050041, 0x00000016,
    0x00000026, 0x00000013, 0x00000015, 0x0004003d, 0x00000006, 0x00000027, 0x00000026, 0x00050089,
    0x00000006, 0x00000028, 0x00000025, 0x00000027, 0x00040070, 0x00000021, 0x00000029, 0x00000028,
    0x00050081, 0x00000021, 0x0000002b, 0x00000029, 0x0000002a, 0x00050085, 0x00000021, 0x0000002d,
    0x0000002b, 0x0000002c, 0x0004003d, 0x00000006, 0x0000002e, 0x00000008, 0x00050041, 0x00000016,
    0x0000002f, 0x00000013, 0x00000015, 0x0004003d, 0x00000006, 0x00000030, 0x0000002f, 0x00050086,
    0x00000006, 0x00000031, 0x0000002e, 0x00000030, 0x00040070, 0x00000021, 0x00000032, 0x00000031,
    0x00050081, 0x00000021, 0x00000034, 0x00000032, 0x00000033, 0x00050085, 0x00000021, 0x00000035,
    0x00000034, 0x0000002c, 0x00060050, 0x00000022, 0x00000037, 0x0000002d, 0x00000035, 0x00000036,
    0x0003003e, 0x00000024, 0x00000037, 0x0004003d, 0x0000003b, 0x0000003e, 0x0000003d, 0x00050041,
    0x00000016, 0x00000040, 0x00000013, 0x0000003f, 0x0004003d, 0x00000006, 0x00000041, 0x00000040,
    0x00050041, 0x00000016, 0x00000043, 0x00000013, 0x00000042, 0x0004003d, 0x00000006, 0x00000044,
    0x00000043, 0x0004003d, 0x00000022, 0x00000045, 0x00000024, 0x00091179, 0x0000003a, 0x0000003e,
    0x00000041, 0x00000044, 0x00000045, 0x00000046, 0x00000048, 0x00000049, 0x00050041, 0x00000007,
    0x0000004f, 0x0000004d, 0x0000004e, 0x0003003e, 0x0000004f, 0x0000000c, 0x000200f9, 0x00000050,
    0x000200f8, 0x00000050, 0x000400f6, 0x00000052, 0x00000053, 0x00000000, 0x000200f9, 0x00000054,
    0x000200f8, 0x00000054, 0x0004117d, 0x0000001c, 0x00000055, 0x0000003a, 0x000400fa, 0x00000055,
    0x00000051, 0x00000052, 0x000200f8, 0x00000051, 0x00050041, 0x00000007, 0x00000056, 0x0000004d,
    0x0000004e, 0x0004003d, 0x00000006, 0x00000057, 0x00000056, 0x00050080, 0x00000006, 0x00000058,
    0x00000057, 0x00000042, 0x0003003e, 0x00000056, 0x00000058, 0x0005117f, 0x00000006, 0x0000005a,
    0x0000003a, 0x0000003f, 0x000500aa, 0x0000001c, 0x0000005b, 0x0000005a, 0x0000000c, 0x000300f7,
    0x0000005d, 0x00000000, 0x000400fa, 0x0000005b, 0x0000005c, 0x0000006b, 0x000200f8, 0x0000005c,
    0x00050041, 0x00000016, 0x0000005f, 0x00000013, 0x0000005e, 0x0004003d, 0x00000006, 0x00000060,
    0x0000005f, 0x000500ab, 0x0000001c, 0x00000062, 0x00000060, 0x00000061, 0x000400a8, 0x0000001c,
    0x00000063, 0x00000062, 0x000300f7, 0x00000065, 0x00000000, 0x000400fa, 0x00000063, 0x00000064,
    0x00000065, 0x000200f8, 0x00000064, 0x00051787, 0x00000014, 0x00000066, 0x0000003a, 0x0000003f,
    0x000500aa, 0x0000001c, 0x00000067, 0x00000066, 0x0000003f, 0x000200f9, 0x00000065, 0x000200f8,
    0x00000065, 0x000700f5, 0x0000001c, 0x00000068, 0x00000062, 0x0000005c, 0x00000067, 0x00000064,
    0x000300f7, 0x0000006a, 0x00000000, 0x000400fa, 0x00000068, 0x00000069, 0x0000006a, 0x000200f8,
    0x00000069, 0x0002117c, 0x0000003a, 0x000200f9, 0x0000006a, 0x000200f8, 0x0000006a, 0x000200f9,
    0x0000005d, 0x000200f8, 0x0000006b, 0x0005178c, 0x00000022, 0x0000006d, 0x0000003a, 0x0000003f,
    0x0003003e, 0x0000006c, 0x0000006d, 0x0005178b, 0x00000022, 0x0000006f, 0x0000003a, 0x0000003f,
    0x0003003e, 0x0000006e, 0x0000006f, 0x00050041, 0x00000071, 0x00000072, 0x0000006c, 0x00000070,
    0x0004003d, 0x00000021, 0x00000073, 0x00000072, 0x00050083, 0x00000021, 0x00000075, 0x00000073,
    0x00000074, 0x00050041, 0x00000071, 0x00000076, 0x0000006e, 0x00000070, 0x0004003d, 0x00000021,
    0x00000077, 0x00000076, 0x0004007f, 0x00000021, 0x00000078, 0x00000077, 0x00050088, 0x00000021,
    0x00000079, 0x00000075, 0x00000078, 0x0003117b, 0x0000003a, 0x00000079, 0x000200f9, 0x0000005d,
    0x000200f8, 0x0000005d, 0x000200f9, 0x00000053, 0x000200f8, 0x00000053, 0x000200f9, 0x00000050,
    0x000200f8, 0x00000052, 0x00050041, 0x00000071, 0x0000007a, 0x0000004d, 0x0000003f, 0x0003003e,
    0x0000007a, 0x00000047, 0x00050041, 0x00000007, 0x0000007c, 0x0000004d, 0x0000007b, 0x0003003e,
    0x0000007c, 0x0000000c, 0x00050041, 0x00000007, 0x0000007d, 0x0000004d, 0x00000015, 0x0003003e,
    0x0000007d, 0x0000000c, 0x00050041, 0x00000007, 0x0000007e, 0x0000004d, 0x0000005e, 0x0003003e,
    0x0000007e, 0x0000000c, 0x00050041, 0x00000007, 0x0000007f, 0x0000004d, 0x00000042, 0x0003003e,
    0x0000007f, 0x0000000c, 0x00050041, 0x00000082, 0x00000083, 0x0000004d, 0x00000080, 0x0003003e,
    0x00000083, 0x00000081, 0x00050041, 0x00000071, 0x00000085, 0x0000004d, 0x00000084, 0x0003003e,
    0x00000085, 0x00000046, 0x0005117f, 0x00000006, 0x00000088, 0x0000003a, 0x00000042, 0x00050041,
    0x00000007, 0x00000089, 0x0000004d, 0x00000086, 0x0003003e, 0x00000089, 0x00000088, 0x00050041,
    0x00000007, 0x0000008a, 0x0000004d, 0x00000086, 0x0004003d, 0x00000006, 0x0000008b, 0x0000008a,
    0x000500ab, 0x0000001c, 0x0000008c, 0x0000008b, 0x0000000c, 0x000300f7, 0x0000008e, 0x00000000,
    0x000400fa, 0x0000008c, 0x0000008d, 0x0000008e, 0x000200f8, 0x0000008d, 0x00051782, 0x00000021,
    0x0000008f, 0x0000003a, 0x00000042, 0x00050041, 0x00000071, 0x00000090, 0x0000004d, 0x0000003f,
    0x0003003e, 0x00000090, 0x0000008f, 0x00051787, 0x00000014, 0x00000091, 0x0000003a, 0x00000042,
    0x0004007c, 0x00000006, 0x00000092, 0x00000091, 0x00050041, 0x00000007, 0x00000093, 0x0000004d,
    0x00000042, 0x0003003e, 0x00000093, 0x00000092, 0x00051784, 0x00000014, 0x00000094, 0x0000003a,
    0x00000042, 0x0004007c, 0x00000006, 0x00000095, 0x00000094, 0x00050041, 0x00000007, 0x00000096,
    0x0000004d, 0x0000005e, 0x0003003e, 0x00000096, 0x00000095, 0x00051783, 0x00000014, 0x00000097,
    0x0000003a, 0x00000042, 0x0004007c, 0x00000006, 0x00000098, 0x00000097, 0x00050041, 0x00000007,
    0x00000099, 0x0000004d, 0x00000015, 0x0003003e, 0x00000099, 0x00000098, 0x0005178f, 0x0000009a,
    0x0000009b, 0x0000003a, 0x00000042, 0x00060051, 0x00000021, 0x0000009c, 0x0000009b, 0x00000003,
    0x00000002, 0x00050041, 0x00000071, 0x0000009d, 0x0000004d, 0x00000084, 0x0003003e, 0x0000009d,
    0x0000009c, 0x00050041, 0x00000007, 0x0000009e, 0x0000004d, 0x00000086, 0x0004003d, 0x00000006,
    0x0000009f, 0x0000009e, 0x000500aa, 0x0000001c, 0x000000a0, 0x0000009f, 0x00000061, 0x000300f7,
    0x000000a2, 0x00000000, 0x000400fa, 0x000000a0, 0x000000a1, 0x000000a2, 0x000200f8, 0x000000a1,
    0x00051788, 0x0000004a, 0x000000a3, 0x0000003a, 0x00000042, 0x00050041, 0x00000082, 0x000000a4,
    0x0000004d, 0x00000080, 0x0003003e, 0x000000a4, 0x000000a3, 0x000200f9, 0x000000a2, 0x000200f8,
    0x000000a2, 0x000200f9, 0x0000008e, 0x000200f8, 0x0000008e, 0x0004003d, 0x00000006, 0x000000aa,
    0x00000008, 0x0004003d, 0x0000004b, 0x000000ab, 0x0000004d, 0x00060041, 0x000000ac, 0x000000ad,
    0x000000a9, 0x0000003f, 0x000000aa, 0x00040190, 0x000000a5, 0x000000ae, 0x000000ab, 0x0003003e,
    0x000000ad, 0x000000ae, 0x000100fd, 0x00010038,
};

#define RT_SIDE 16u /* rays a side: they cover x and y from 0 to 4 */
#define RT_RAYS (RT_SIDE * RT_SIDE)
#define RT_OPAQUE 1u /* gl_RayFlagsOpaqueEXT */

struct rt_hit { /* Hit of the shader */
    float t;
    uint32_t primitive, instance, custom;
    float bary[2];
    uint32_t type, candidates;
    float height;
    uint32_t pad;
};

/* An instance: the unit square or the unit box, scaled, then moved. */
struct rt_instance {
    float scale, x, y, z;
    uint32_t custom, mask, flags;
    int box;
};

#define RT_INSTANCES 4u
static const struct rt_instance rt_scene[RT_INSTANCES] = {
    { 4.0f, 0.0f, 0.0f, 0.0f, 100, 0x01, 0, 0 },
    { 1.0f, 0.5f, 0.5f, 1.0f, 200, 0x02, 0, 0 },
    { 2.0f, 1.0f, 1.0f, 2.0f, 300, 0x04, VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR, 0 },
    { 1.0f, 3.0f, 0.0f, 0.25f, 400, 0x08, 0, 1 },
};

struct rt_fn {
    PFN_vkCreateAccelerationStructureKHR create;
    PFN_vkDestroyAccelerationStructureKHR destroy;
    PFN_vkGetAccelerationStructureBuildSizesKHR sizes;
    PFN_vkGetAccelerationStructureDeviceAddressKHR address;
    PFN_vkCmdBuildAccelerationStructuresKHR build;
    PFN_vkCmdCopyAccelerationStructureKHR copy;
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR write;
};

struct rt_buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    uint8_t *data;
    VkDeviceAddress address;
};

struct rt_accel {
    struct rt_buffer storage;
    VkAccelerationStructureKHR handle;
    VkDeviceAddress address;
    VkDeviceSize size;
};

struct rt {
    struct ct_env e;
    struct rt_fn fn;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    VkShaderModule module;
    VkPipeline pipeline;
    struct rt_buffer input, scratch;
    struct ct_buffer hits;
    struct rt_accel square, box, top;
};

/* Offsets into rt::input */
#define RT_IN_VERTICES 0u
#define RT_IN_BOX 256u
#define RT_IN_INSTANCES 512u
#define RT_IN_SIZE 4096u
#define RT_SCRATCH_SIZE (4u << 20)

static const char *const rt_extensions[] = {
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
};

/* Ends the test as skipped unless the device has the three extensions on
 * Vulkan 1.2 (which has what they depend on in core). */
static void rt_require_extensions(struct ct *t, struct ct_env *e)
{
    size_t i;
    if (e->version < VK_API_VERSION_1_2)
        ct_skip(t, "needs Vulkan 1.2");
    for (i = 0; i < CT_ARRAY_SIZE(rt_extensions); i++)
        if (!ct_device_extension_supported(e->pd, rt_extensions[i]))
            ct_skip(t, "%s is not supported", rt_extensions[i]);
}

static void rt_buffer_init(struct ct *t, struct rt *r, VkDeviceSize size, VkBufferUsageFlags usage,
                           struct rt_buffer *out)
{
    VkBufferCreateInfo info;
    VkMemoryAllocateInfo alloc;
    VkMemoryAllocateFlagsInfo flags;
    VkBufferDeviceAddressInfo address;
    VkMemoryRequirements req;
    void *p = NULL;

    memset(out, 0, sizeof(*out));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    CT_REQUIRE_VK(t, vkCreateBuffer(r->e.device, &info, NULL, &out->buffer));
    vkGetBufferMemoryRequirements(r->e.device, out->buffer, &req);
    memset(&flags, 0, sizeof(flags));
    flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = &flags;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = ct_find_memory_type(&r->e, req.memoryTypeBits,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                VK_MEMORY_PROPERTY_PROTECTED_BIT);
    CT_REQUIREF(t, alloc.memoryTypeIndex != UINT32_MAX, "no host-coherent memory type for a buffer");
    CT_REQUIRE_VK(t, vkAllocateMemory(r->e.device, &alloc, NULL, &out->memory));
    CT_REQUIRE_VK(t, vkBindBufferMemory(r->e.device, out->buffer, out->memory, 0));
    CT_REQUIRE_VK(t, vkMapMemory(r->e.device, out->memory, 0, VK_WHOLE_SIZE, 0, &p));
    out->data = p;
    memset(out->data, 0, (size_t)size);
    memset(&address, 0, sizeof(address));
    address.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    address.buffer = out->buffer;
    out->address = vkGetBufferDeviceAddress(r->e.device, &address);
    CT_REQUIRE(t, out->address != 0);
}

static void rt_buffer_fini(struct rt *r, struct rt_buffer *b)
{
    if (b->buffer)
        vkDestroyBuffer(r->e.device, b->buffer, NULL);
    if (b->memory)
        vkFreeMemory(r->e.device, b->memory, NULL);
    memset(b, 0, sizeof(*b));
}

static void rt_accel_init(struct ct *t, struct rt *r, VkAccelerationStructureTypeKHR type, VkDeviceSize size,
                          struct rt_accel *out)
{
    VkAccelerationStructureCreateInfoKHR info;
    VkAccelerationStructureDeviceAddressInfoKHR address;

    memset(out, 0, sizeof(*out));
    rt_buffer_init(t, r, size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, &out->storage);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    info.buffer = out->storage.buffer;
    info.size = size;
    info.type = type;
    CT_REQUIRE_VK(t, r->fn.create(r->e.device, &info, NULL, &out->handle));
    memset(&address, 0, sizeof(address));
    address.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    address.accelerationStructure = out->handle;
    out->address = r->fn.address(r->e.device, &address);
    out->size = size;
    CT_REQUIRE(t, out->address != 0);
}

static void rt_accel_fini(struct rt *r, struct rt_accel *a)
{
    if (a->handle)
        r->fn.destroy(r->e.device, a->handle, NULL);
    rt_buffer_fini(r, &a->storage);
    memset(a, 0, sizeof(*a));
}

/* The geometry of the square (two triangles of float3 positions), of the
 * box (one VkAabbPositionsKHR) or of count instances; all in rt::input. */
static void rt_geometry(const struct rt *r, int kind, VkAccelerationStructureGeometryKHR *g)
{
    memset(g, 0, sizeof(*g));
    g->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    if (kind == 0) {
        g->geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        g->geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        g->geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        g->geometry.triangles.vertexData.deviceAddress = r->input.address + RT_IN_VERTICES;
        g->geometry.triangles.vertexStride = 12;
        g->geometry.triangles.maxVertex = 5;
        g->geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
        return;
    }
    if (kind == 1) {
        g->geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
        g->geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
        g->geometry.aabbs.data.deviceAddress = r->input.address + RT_IN_BOX;
        g->geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);
        return;
    }
    g->geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    g->geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    g->geometry.instances.data.deviceAddress = r->input.address + RT_IN_INSTANCES;
}

static void rt_build_info(VkAccelerationStructureBuildGeometryInfoKHR *info, VkAccelerationStructureTypeKHR type,
                          VkBuildAccelerationStructureFlagsKHR flags, const VkAccelerationStructureGeometryKHR *g)
{
    memset(info, 0, sizeof(*info));
    info->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info->type = type;
    info->flags = flags;
    info->mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info->geometryCount = 1;
    info->pGeometries = g;
}

/* Makes what an acceleration structure build wrote visible to later builds,
 * copies, queries and compute shaders. */
static void rt_barrier(VkCommandBuffer cb)
{
    VkMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    b.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &b, 0, NULL, 0, NULL);
}

/* Creates an acceleration structure of the size a build of count primitives
 * of g needs and builds it. */
static void rt_build(struct ct *t, struct rt *r, VkAccelerationStructureTypeKHR type,
                     VkBuildAccelerationStructureFlagsKHR flags, const VkAccelerationStructureGeometryKHR *g,
                     uint32_t count, struct rt_accel *out)
{
    VkAccelerationStructureBuildGeometryInfoKHR info;
    VkAccelerationStructureBuildSizesInfoKHR sizes;
    VkAccelerationStructureBuildRangeInfoKHR range;
    const VkAccelerationStructureBuildRangeInfoKHR *ranges = &range;
    VkCommandBuffer cb;

    rt_build_info(&info, type, flags, g);
    memset(&sizes, 0, sizeof(sizes));
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    r->fn.sizes(r->e.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &count, &sizes);
    CT_REQUIREF(t, sizes.accelerationStructureSize > 0 && sizes.buildScratchSize <= RT_SCRATCH_SIZE,
                "sizes of a build of %u primitives: structure %llu, scratch %llu", count,
                (unsigned long long)sizes.accelerationStructureSize, (unsigned long long)sizes.buildScratchSize);
    rt_accel_init(t, r, type, sizes.accelerationStructureSize, out);
    info.dstAccelerationStructure = out->handle;
    info.scratchData.deviceAddress = r->scratch.address;
    memset(&range, 0, sizeof(range));
    range.primitiveCount = count;
    cb = ct_begin_commands(t, &r->e);
    r->fn.build(cb, 1, &info, &ranges);
    rt_barrier(cb);
    ct_submit_and_wait(t, &r->e, cb);
}

/* Writes the instances the top level is built of: the square or the box,
 * scaled, then moved. */
static void rt_write_instances(struct rt *r, const struct rt_instance *instances, VkDeviceAddress square,
                               VkDeviceAddress box)
{
    uint32_t i;
    for (i = 0; i < RT_INSTANCES; i++) {
        const struct rt_instance *s = &instances[i];
        VkAccelerationStructureInstanceKHR v;
        memset(&v, 0, sizeof(v));
        v.transform.matrix[0][0] = v.transform.matrix[1][1] = v.transform.matrix[2][2] = s->scale;
        v.transform.matrix[0][3] = s->x;
        v.transform.matrix[1][3] = s->y;
        v.transform.matrix[2][3] = s->z;
        v.instanceCustomIndex = s->custom;
        v.mask = s->mask;
        v.flags = s->flags;
        v.accelerationStructureReference = s->box ? box : square;
        memcpy(r->input.data + RT_IN_INSTANCES + i * sizeof(v), &v, sizeof(v));
    }
}

static void rt_write_descriptor(struct rt *r, VkAccelerationStructureKHR top)
{
    VkWriteDescriptorSetAccelerationStructureKHR accel;
    VkDescriptorBufferInfo buffer;
    VkWriteDescriptorSet writes[2];

    memset(&accel, 0, sizeof(accel));
    accel.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    accel.accelerationStructureCount = 1;
    accel.pAccelerationStructures = &top;
    memset(writes, 0, sizeof(writes));
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &accel;
    writes[0].dstSet = r->set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    buffer.buffer = r->hits.buffer;
    buffer.offset = 0;
    buffer.range = VK_WHOLE_SIZE;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = r->set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &buffer;
    vkUpdateDescriptorSets(r->e.device, 2, writes, 0, NULL);
}

/* A device with the three extensions and their features, the pipeline of
 * the shader above and the scene: the square, the box and a top level of
 * rt_scene (built with top_flags). */
static void rt_init(struct ct *t, struct rt *r, VkBuildAccelerationStructureFlagsKHR bottom_flags,
                    VkBuildAccelerationStructureFlagsKHR top_flags)
{
    static const float vertices[6][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 } };
    static const VkAabbPositionsKHR box = { 0, 0, 0, 1, 1, 1 };
    VkPhysicalDeviceFeatures2 features;
    VkPhysicalDeviceVulkan12Features v12, have12;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel, have_accel;
    VkPhysicalDeviceRayQueryFeaturesKHR query, have_query;
    VkDescriptorSetLayoutBinding bindings[2];
    VkDescriptorSetLayoutCreateInfo set_info;
    VkPipelineLayoutCreateInfo layout_info;
    VkPushConstantRange push;
    VkDescriptorPoolSize pool_sizes[2];
    VkDescriptorPoolCreateInfo pool_info;
    VkDescriptorSetAllocateInfo set_alloc;
    VkComputePipelineCreateInfo pipeline_info;
    VkAccelerationStructureGeometryKHR g;

    memset(r, 0, sizeof(*r));
    ct_env_init(t, &r->e);
    rt_require_extensions(t, &r->e);

    memset(&have12, 0, sizeof(have12));
    have12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&have_accel, 0, sizeof(have_accel));
    have_accel.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    have_accel.pNext = &have12;
    memset(&have_query, 0, sizeof(have_query));
    have_query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    have_query.pNext = &have_accel;
    memset(&features, 0, sizeof(features));
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &have_query;
    vkGetPhysicalDeviceFeatures2(r->e.pd, &features);
    CT_REQUIREF(t, have12.bufferDeviceAddress && have_accel.accelerationStructure && have_query.rayQuery,
                "bufferDeviceAddress %u, accelerationStructure %u, rayQuery %u: the extensions require all three",
                have12.bufferDeviceAddress, have_accel.accelerationStructure, have_query.rayQuery);

    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.bufferDeviceAddress = VK_TRUE;
    memset(&accel, 0, sizeof(accel));
    accel.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accel.pNext = &v12;
    accel.accelerationStructure = VK_TRUE;
    memset(&query, 0, sizeof(query));
    query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    query.pNext = &accel;
    query.rayQuery = VK_TRUE;
    ct_env_device_ex(t, &r->e, &query, rt_extensions, CT_ARRAY_SIZE(rt_extensions), NULL);

#define RT_RESOLVE(field, name)                                                                                   \
    do {                                                                                                           \
        r->fn.field = (PFN_##name)vkGetDeviceProcAddr(r->e.device, #name);                                       \
        CT_REQUIREF(t, r->fn.field != NULL, "vkGetDeviceProcAddr(device, \"%s\") is NULL with its extension "     \
                                            "enabled",                                                            \
                    #name);                                                                                        \
    } while (0)
    RT_RESOLVE(create, vkCreateAccelerationStructureKHR);
    RT_RESOLVE(destroy, vkDestroyAccelerationStructureKHR);
    RT_RESOLVE(sizes, vkGetAccelerationStructureBuildSizesKHR);
    RT_RESOLVE(address, vkGetAccelerationStructureDeviceAddressKHR);
    RT_RESOLVE(build, vkCmdBuildAccelerationStructuresKHR);
    RT_RESOLVE(copy, vkCmdCopyAccelerationStructureKHR);
    RT_RESOLVE(write, vkCmdWriteAccelerationStructuresPropertiesKHR);
#undef RT_RESOLVE

    memset(bindings, 0, sizeof(bindings));
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    memset(&set_info, 0, sizeof(set_info));
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.bindingCount = 2;
    set_info.pBindings = bindings;
    CT_REQUIRE_VK(t, vkCreateDescriptorSetLayout(r->e.device, &set_info, NULL, &r->set_layout));
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = 16;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &r->set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    CT_REQUIRE_VK(t, vkCreatePipelineLayout(r->e.device, &layout_info, NULL, &r->layout));
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    pool_sizes[0].descriptorCount = 1;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[1].descriptorCount = 1;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = pool_sizes;
    CT_REQUIRE_VK(t, vkCreateDescriptorPool(r->e.device, &pool_info, NULL, &r->pool));
    memset(&set_alloc, 0, sizeof(set_alloc));
    set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_alloc.descriptorPool = r->pool;
    set_alloc.descriptorSetCount = 1;
    set_alloc.pSetLayouts = &r->set_layout;
    CT_REQUIRE_VK(t, vkAllocateDescriptorSets(r->e.device, &set_alloc, &r->set));
    r->module = ct_shader_module(t, &r->e, spv_trace, sizeof(spv_trace));
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = r->module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = r->layout;
    CT_REQUIRE_VK(t, vkCreateComputePipelines(r->e.device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &r->pipeline));

    rt_buffer_init(t, r, RT_IN_SIZE, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, &r->input);
    rt_buffer_init(t, r, RT_SCRATCH_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &r->scratch);
    ct_host_buffer(t, &r->e, RT_RAYS * sizeof(struct rt_hit), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &r->hits);
    memcpy(r->input.data + RT_IN_VERTICES, vertices, sizeof(vertices));
    memcpy(r->input.data + RT_IN_BOX, &box, sizeof(box));

    rt_geometry(r, 0, &g);
    rt_build(t, r, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, bottom_flags, &g, 2, &r->square);
    rt_geometry(r, 1, &g);
    rt_build(t, r, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, bottom_flags, &g, 1, &r->box);
    rt_write_instances(r, rt_scene, r->square.address, r->box.address);
    rt_geometry(r, 2, &g);
    rt_build(t, r, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, top_flags, &g, RT_INSTANCES, &r->top);
    rt_write_descriptor(r, r->top.handle);
}

static void rt_fini(struct rt *r)
{
    rt_accel_fini(r, &r->top);
    rt_accel_fini(r, &r->box);
    rt_accel_fini(r, &r->square);
    ct_buffer_fini(&r->e, &r->hits);
    rt_buffer_fini(r, &r->scratch);
    rt_buffer_fini(r, &r->input);
    vkDestroyPipeline(r->e.device, r->pipeline, NULL);
    vkDestroyShaderModule(r->e.device, r->module, NULL);
    vkDestroyDescriptorPool(r->e.device, r->pool, NULL);
    vkDestroyPipelineLayout(r->e.device, r->layout, NULL);
    vkDestroyDescriptorSetLayout(r->e.device, r->set_layout, NULL);
    ct_env_fini(&r->e);
}

/* What ray i must commit in a scene of instances: the highest surface under
 * it that its mask selects and, in mode 1, whose candidate the shader
 * confirms (primitive 1 of a square is confirmed only where the geometry is
 * opaque, by the instance or by the ray's flags). */
static void rt_expect(const struct rt_instance *instances, uint32_t i, uint32_t flags, uint32_t mask, uint32_t mode,
                      struct rt_hit *h)
{
    double px = ((double)(i % RT_SIDE) + 0.37) * 0.25, py = ((double)(i / RT_SIDE) + 0.29) * 0.25;
    double best = -1.0;
    uint32_t k;

    memset(h, 0, sizeof(*h));
    h->t = -1.0f;
    for (k = 0; k < RT_INSTANCES; k++) {
        const struct rt_instance *s = &instances[k];
        double u = (px - s->x) / s->scale, v = (py - s->y) / s->scale, z = s->z;
        uint32_t primitive = 0, type = 1;
        if (!(s->mask & mask) || u <= 0.0 || v <= 0.0 || u >= 1.0 || v >= 1.0)
            continue;
        if (s->box) {
            z += 0.5 * s->scale;
            type = 2;
        } else {
            int opaque = (s->flags & VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR) || (flags & RT_OPAQUE);
            primitive = u + v < 1.0 ? 0 : 1;
            if (!opaque && mode == 1 && primitive == 1)
                continue;
        }
        if (z <= best)
            continue;
        best = z;
        h->t = (float)(100.0 - z);
        h->primitive = primitive;
        h->instance = k;
        h->custom = s->custom;
        h->type = type;
        h->height = s->z;
        h->bary[0] = h->bary[1] = 0.0f;
        if (type == 1) {
            h->bary[0] = (float)(primitive ? 1.0 - u : u);
            h->bary[1] = (float)(primitive ? 1.0 - v : v);
        }
    }
}

static int rt_near(float a, float b)
{
    float d = a - b;
    return d < 1e-4f && d > -1e-4f;
}

/* Traces the rays and compares every one with rt_expect; returns how many
 * committed an intersection. */
static uint32_t rt_trace(struct ct *t, struct rt *r, const struct rt_instance *instances, uint32_t flags,
                         uint32_t mask, uint32_t mode, const char *what)
{
    uint32_t push[4], i, hit = 0, bad = 0;
    const struct rt_hit *got = (const struct rt_hit *)(const void *)r->hits.data;
    VkCommandBuffer cb = ct_begin_commands(t, &r->e);

    push[0] = flags;
    push[1] = mask;
    push[2] = mode;
    push[3] = RT_SIDE;
    memset(r->hits.data, 0xee, RT_RAYS * sizeof(struct rt_hit));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, r->pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, r->layout, 0, 1, &r->set, 0, NULL);
    vkCmdPushConstants(cb, r->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    vkCmdDispatch(cb, RT_RAYS / 16, 1, 1);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &r->e, cb);
    for (i = 0; i < RT_RAYS; i++) {
        struct rt_hit want;
        rt_expect(instances, i, flags, mask, mode, &want);
        hit += want.type != 0;
        if (got[i].type == want.type && rt_near(got[i].t, want.t) && got[i].primitive == want.primitive &&
            got[i].instance == want.instance && got[i].custom == want.custom &&
            rt_near(got[i].bary[0], want.bary[0]) && rt_near(got[i].bary[1], want.bary[1]) &&
            rt_near(got[i].height, want.height))
            continue;
        if (bad++ < 4)
            ct_fail(t, __FILE__, __LINE__,
                    "%s: ray %u committed type %u t %g primitive %u instance %u custom %u barycentrics (%g, %g) "
                    "instance height %g; expected type %u t %g primitive %u instance %u custom %u barycentrics "
                    "(%g, %g) instance height %g",
                    what, i, got[i].type, (double)got[i].t, got[i].primitive, got[i].instance, got[i].custom,
                    (double)got[i].bary[0], (double)got[i].bary[1], (double)got[i].height, want.type,
                    (double)want.t, want.primitive, want.instance, want.custom, (double)want.bary[0],
                    (double)want.bary[1], (double)want.height);
    }
    CT_CHECKF(t, bad == 0, "%s: %u of %u rays differ", what, bad, RT_RAYS);
    return hit;
}

/* ---- Features and limits ---- */

CT_TEST(accel_feature_requirements,
        "Features > Feature Requirements: accelerationStructure and "
        "descriptorBindingAccelerationStructureUpdateAfterBind if VK_KHR_acceleration_structure is supported "
        "(which also requires bufferDeviceAddress and the descriptor indexing features Vulkan 1.2 requires "
        "with descriptorIndexing), rayQuery if VK_KHR_ray_query is supported",
        "vkGetPhysicalDeviceFeatures2")
{
    struct ct_env e;
    VkPhysicalDeviceFeatures2 features;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel;
    VkPhysicalDeviceRayQueryFeaturesKHR query;

    ct_env_init(t, &e);
    rt_require_extensions(t, &e);
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&accel, 0, sizeof(accel));
    accel.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accel.pNext = &v12;
    memset(&query, 0, sizeof(query));
    query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    query.pNext = &accel;
    memset(&features, 0, sizeof(features));
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &query;
    vkGetPhysicalDeviceFeatures2(e.pd, &features);
    CT_CHECK(t, accel.accelerationStructure == VK_TRUE);
    CT_CHECK(t, accel.descriptorBindingAccelerationStructureUpdateAfterBind == VK_TRUE);
    CT_CHECK(t, query.rayQuery == VK_TRUE);
    CT_CHECK(t, v12.bufferDeviceAddress == VK_TRUE);
    CT_CHECK(t, v12.descriptorIndexing == VK_TRUE);
    CT_CHECK(t, v12.shaderSampledImageArrayNonUniformIndexing == VK_TRUE);
    CT_CHECK(t, v12.shaderStorageBufferArrayNonUniformIndexing == VK_TRUE);
    CT_CHECK(t, v12.descriptorBindingUpdateUnusedWhilePending == VK_TRUE);
    CT_CHECK(t, v12.descriptorBindingPartiallyBound == VK_TRUE);
    CT_CHECK(t, v12.runtimeDescriptorArray == VK_TRUE);
    ct_env_fini(&e);
}

static void rt_get_limits(struct ct *t, struct ct_env *e, VkPhysicalDeviceAccelerationStructurePropertiesKHR *p)
{
    VkPhysicalDeviceProperties2 props;
    ct_env_init(t, e);
    rt_require_extensions(t, e);
    memset(p, 0, sizeof(*p));
    p->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    memset(&props, 0, sizeof(props));
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = p;
    vkGetPhysicalDeviceProperties2(e->pd, &props);
}

CT_TEST(accel_limit_requirements,
        "Limits > Limit Requirements (Required Limits table): maxGeometryCount and maxInstanceCount at least "
        "2^24-1, maxPerStageDescriptorAccelerationStructures and maxDescriptorSetAccelerationStructures at "
        "least 16, their UpdateAfterBind counterparts at least 500000, "
        "minAccelerationStructureScratchOffsetAlignment at most 256 (and a power of two)",
        "vkGetPhysicalDeviceProperties2")
{
    struct ct_env e;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR p;

    rt_get_limits(t, &e, &p);
    CT_CHECKF(t, p.maxGeometryCount >= 16777215u, "maxGeometryCount %llu", (unsigned long long)p.maxGeometryCount);
    CT_CHECKF(t, p.maxInstanceCount >= 16777215u, "maxInstanceCount %llu", (unsigned long long)p.maxInstanceCount);
    CT_CHECKF(t, p.maxPerStageDescriptorAccelerationStructures >= 16, "maxPerStageDescriptorAccelerationStructures %u",
              p.maxPerStageDescriptorAccelerationStructures);
    CT_CHECKF(t, p.maxPerStageDescriptorUpdateAfterBindAccelerationStructures >= 500000,
              "maxPerStageDescriptorUpdateAfterBindAccelerationStructures %u",
              p.maxPerStageDescriptorUpdateAfterBindAccelerationStructures);
    CT_CHECKF(t, p.maxDescriptorSetAccelerationStructures >= 16, "maxDescriptorSetAccelerationStructures %u",
              p.maxDescriptorSetAccelerationStructures);
    CT_CHECKF(t, p.maxDescriptorSetUpdateAfterBindAccelerationStructures >= 500000,
              "maxDescriptorSetUpdateAfterBindAccelerationStructures %u",
              p.maxDescriptorSetUpdateAfterBindAccelerationStructures);
    CT_CHECKF(t, p.minAccelerationStructureScratchOffsetAlignment <= 256 &&
                     ct_is_pow2(p.minAccelerationStructureScratchOffsetAlignment),
              "minAccelerationStructureScratchOffsetAlignment %u", p.minAccelerationStructureScratchOffsetAlignment);
    ct_env_fini(&e);
}

CT_TEST(accel_limit_primitive_count,
        "Limits > Limit Requirements (Required Limits table): maxPrimitiveCount at least 2^29-1",
        "vkGetPhysicalDeviceProperties2")
{
    struct ct_env e;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR p;

    rt_get_limits(t, &e, &p);
    CT_CHECKF(t, p.maxPrimitiveCount >= 536870911u, "maxPrimitiveCount %llu", (unsigned long long)p.maxPrimitiveCount);
    ct_env_fini(&e);
}

/* ---- Deferred operations ---- */

CT_TEST(deferred_operation_result,
        "Deferred Host Operations > Querying Result: \"If no command has been deferred on operation, "
        "vkGetDeferredOperationResultKHR returns VK_SUCCESS\"",
        "vkCreateDeferredOperationKHR vkGetDeferredOperationResultKHR vkDestroyDeferredOperationKHR")
{
    static const char *const extension = VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME;
    struct ct_env e;
    PFN_vkCreateDeferredOperationKHR create;
    PFN_vkGetDeferredOperationResultKHR result;
    PFN_vkDestroyDeferredOperationKHR destroy;
    VkDeferredOperationKHR operation = VK_NULL_HANDLE;

    ct_env_init(t, &e);
    if (!ct_device_extension_supported(e.pd, extension))
        ct_skip(t, "%s is not supported", extension);
    ct_env_device_ex(t, &e, NULL, &extension, 1, NULL);
    create = (PFN_vkCreateDeferredOperationKHR)vkGetDeviceProcAddr(e.device, "vkCreateDeferredOperationKHR");
    result = (PFN_vkGetDeferredOperationResultKHR)vkGetDeviceProcAddr(e.device, "vkGetDeferredOperationResultKHR");
    destroy = (PFN_vkDestroyDeferredOperationKHR)vkGetDeviceProcAddr(e.device, "vkDestroyDeferredOperationKHR");
    CT_REQUIRE(t, create != NULL && result != NULL && destroy != NULL);
    CT_REQUIRE_VK(t, create(e.device, NULL, &operation));
    CT_CHECK(t, operation != VK_NULL_HANDLE);
    CT_CHECK_VK(t, result(e.device, operation), VK_SUCCESS);
    destroy(e.device, operation, NULL);
    ct_env_fini(&e);
}

/* ---- Builds ---- */

CT_TEST(accel_build_sizes_ignored_members,
        "Acceleration Structures > vkGetAccelerationStructureBuildSizesKHR: \"The srcAccelerationStructure, "
        "dstAccelerationStructure, and mode members of pBuildInfo are ignored. Any VkDeviceOrHostAddressKHR or "
        "VkDeviceOrHostAddressConstKHR members of pBuildInfo are ignored by this command, except that the "
        "hostAddress member of VkAccelerationStructureGeometryTrianglesDataKHR::transformData will be examined "
        "to check if it is NULL\"",
        RT_COMMANDS)
{
    struct rt r;
    VkAccelerationStructureGeometryKHR g;
    VkAccelerationStructureBuildGeometryInfoKHR info;
    VkAccelerationStructureBuildSizesInfoKHR a, b;
    uint32_t kind, count = 1000;

    rt_init(t, &r, 0, 0);
    for (kind = 0; kind < 3; kind++) {
        VkAccelerationStructureTypeKHR type =
            kind == 2 ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        rt_geometry(&r, (int)kind, &g);
        if (kind == 0)
            g.geometry.triangles.maxVertex = 3 * count - 1;
        rt_build_info(&info, type, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR, &g);
        memset(&a, 0, sizeof(a));
        a.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        r.fn.sizes(r.e.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &count, &a);
        CT_CHECKF(t, a.accelerationStructureSize > 0 && a.buildScratchSize > 0,
                  "kind %u: structure %llu, build scratch %llu", kind, (unsigned long long)a.accelerationStructureSize,
                  (unsigned long long)a.buildScratchSize);

        /* the ignored members, changed */
        if (kind == 0)
            g.geometry.triangles.vertexData.deviceAddress = 0;
        if (kind == 1)
            g.geometry.aabbs.data.deviceAddress = 0;
        if (kind == 2)
            g.geometry.instances.data.deviceAddress = 0;
        info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        info.srcAccelerationStructure = r.top.handle;
        info.dstAccelerationStructure = r.top.handle;
        info.scratchData.deviceAddress = r.scratch.address;
        memset(&b, 0, sizeof(b));
        b.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        r.fn.sizes(r.e.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &count, &b);
        CT_CHECKF(t, a.accelerationStructureSize == b.accelerationStructureSize &&
                         a.buildScratchSize == b.buildScratchSize && a.updateScratchSize == b.updateScratchSize,
                  "kind %u: sizes %llu/%llu/%llu became %llu/%llu/%llu with other ignored members", kind,
                  (unsigned long long)a.accelerationStructureSize, (unsigned long long)a.buildScratchSize,
                  (unsigned long long)a.updateScratchSize, (unsigned long long)b.accelerationStructureSize,
                  (unsigned long long)b.buildScratchSize, (unsigned long long)b.updateScratchSize);
    }
    rt_fini(&r);
}

/* ---- Ray queries ---- */

CT_TEST(ray_query_committed_hit,
        "Ray Traversal > Ray Closest Hit Determination, Acceleration Structures > "
        "VkAccelerationStructureInstanceKHR: a ray query that confirms every candidate commits the closest "
        "intersection: its T, primitive index, instance index and custom index, barycentrics and "
        "object-to-world transform, through instances that scale and move a triangle geometry and a box "
        "geometry whose candidates the shader turns into hits (on a plane of object space)",
        RT_COMMANDS)
{
    struct rt r;
    uint32_t hit;

    rt_init(t, &r, 0, 0);
    hit = rt_trace(t, &r, rt_scene, 0, 0xff, 0, "every instance");
    CT_CHECKF(t, hit == RT_RAYS, "the floor is under every ray, but only %u of %u are expected to hit", hit, RT_RAYS);
    rt_fini(&r);
}

CT_TEST(ray_query_cull_mask,
        "Ray Traversal > Ray Intersection Culling > Ray Mask Culling: \"Instances can be made invisible to "
        "particular rays based on the value of VkAccelerationStructureInstanceKHR::mask\": an instance is "
        "tested only if mask & cullMask is not zero",
        RT_COMMANDS)
{
    static const uint32_t masks[] = { 0x01, 0x02, 0x04, 0x08, 0x06, 0x0b, 0xf0, 0x00 };
    struct rt r;
    uint32_t i;

    rt_init(t, &r, 0, 0);
    for (i = 0; i < CT_ARRAY_SIZE(masks); i++) {
        char what[32];
        uint32_t hit;
        snprintf(what, sizeof(what), "cull mask 0x%02x", masks[i]);
        hit = rt_trace(t, &r, rt_scene, 0, masks[i], 0, what);
        if (!(masks[i] & 0x0f))
            CT_CHECKF(t, hit == 0, "%s selects no instance", what);
    }
    rt_fini(&r);
}

CT_TEST(ray_query_candidate_confirmation,
        "Ray Traversal > Ray Intersection Confirmation and Ray Intersection Culling > Ray Opacity Culling: "
        "an intersection with a non-opaque triangle is a candidate the ray query commits only if the shader "
        "confirms it; one with an opaque triangle (VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR on the instance, "
        "or the Opaque ray flag) is committed without the shader",
        RT_COMMANDS)
{
    struct rt r;
    const struct rt_hit *got;
    uint32_t i, offered = 0;

    rt_init(t, &r, 0, 0);
    rt_trace(t, &r, rt_scene, 0, 0xff, 1, "primitive 1 of non-opaque squares ignored");
    rt_trace(t, &r, rt_scene, RT_OPAQUE, 0xff, 1, "primitive 1 ignored, but the ray has the Opaque flag");
    /* with the Opaque flag and the box masked out, nothing is left for the shader to decide */
    rt_trace(t, &r, rt_scene, RT_OPAQUE, 0x07, 1, "the Opaque flag over the squares alone");
    got = (const struct rt_hit *)(const void *)r.hits.data;
    for (i = 0; i < RT_RAYS; i++)
        offered += got[i].candidates;
    CT_CHECKF(t, offered == 0, "rays with the Opaque flag over triangles were offered %u candidates", offered);
    rt_fini(&r);
}

/* ---- Updates and copies ---- */

CT_TEST(accel_update_instances,
        "Acceleration Structures > VkAccelerationStructureBuildGeometryInfoKHR: a build of mode "
        "VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR with srcAccelerationStructure equal to "
        "dstAccelerationStructure updates in place an acceleration structure built with "
        "VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR: rays then see the instances' new transforms, "
        "custom indices and masks",
        RT_COMMANDS)
{
    struct rt r;
    struct rt_instance moved[RT_INSTANCES];
    VkAccelerationStructureGeometryKHR g;
    VkAccelerationStructureBuildGeometryInfoKHR info;
    VkAccelerationStructureBuildSizesInfoKHR sizes;
    VkAccelerationStructureBuildRangeInfoKHR range;
    const VkAccelerationStructureBuildRangeInfoKHR *ranges = &range;
    uint32_t count = RT_INSTANCES;
    VkCommandBuffer cb;

    rt_init(t, &r, 0, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);
    rt_trace(t, &r, rt_scene, 0, 0xff, 0, "before the update");

    memcpy(moved, rt_scene, sizeof(moved));
    moved[1].z = 3.0f; /* above instance 2 now */
    moved[1].x = 1.25f;
    moved[1].custom = 250;
    moved[2].scale = 1.5f;
    moved[3].mask = 0x10;
    rt_write_instances(&r, moved, r.square.address, r.box.address);
    rt_geometry(&r, 2, &g);
    rt_build_info(&info, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
                  &g);
    memset(&sizes, 0, sizeof(sizes));
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    r.fn.sizes(r.e.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &count, &sizes);
    CT_REQUIREF(t, sizes.updateScratchSize <= RT_SCRATCH_SIZE, "update scratch %llu",
                (unsigned long long)sizes.updateScratchSize);
    info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    info.srcAccelerationStructure = r.top.handle;
    info.dstAccelerationStructure = r.top.handle;
    info.scratchData.deviceAddress = r.scratch.address;
    memset(&range, 0, sizeof(range));
    range.primitiveCount = RT_INSTANCES;
    cb = ct_begin_commands(t, &r.e);
    r.fn.build(cb, 1, &info, &ranges);
    rt_barrier(cb);
    ct_submit_and_wait(t, &r.e, cb);
    rt_trace(t, &r, moved, 0, 0xff, 0, "after the update");
    rt_trace(t, &r, moved, 0, 0x1a, 1, "after the update, cull mask 0x1a, primitive 1 ignored");
    rt_fini(&r);
}

CT_TEST(accel_compacting_copy,
        "Acceleration Structures > Copying Acceleration Structures: a "
        "VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR query gives the size a "
        "VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR copy of an acceleration structure built with "
        "VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR needs (no more than the size it was created "
        "with); rays through instances of the compacted copies, and through a "
        "VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR copy of the top level, commit what they did through "
        "the sources",
        RT_COMMANDS "vkCmdWriteAccelerationStructuresPropertiesKHR vkCmdCopyAccelerationStructureKHR "
                    "vkCreateQueryPool vkDestroyQueryPool vkCmdResetQueryPool vkGetQueryPoolResults")
{
    struct rt r;
    struct rt_accel compact[2], clone;
    VkQueryPoolCreateInfo pool_info;
    VkQueryPool pool = VK_NULL_HANDLE;
    VkAccelerationStructureKHR sources[2];
    VkDeviceSize sizes[2] = { 0, 0 };
    VkCopyAccelerationStructureInfoKHR copy;
    VkAccelerationStructureGeometryKHR g;
    struct rt_accel top;
    VkCommandBuffer cb;
    uint32_t i;

    rt_init(t, &r, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR, 0);
    sources[0] = r.square.handle;
    sources[1] = r.box.handle;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    pool_info.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    pool_info.queryCount = 2;
    CT_REQUIRE_VK(t, vkCreateQueryPool(r.e.device, &pool_info, NULL, &pool));
    cb = ct_begin_commands(t, &r.e);
    vkCmdResetQueryPool(cb, pool, 0, 2);
    r.fn.write(cb, 2, sources, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, pool, 0);
    ct_submit_and_wait(t, &r.e, cb);
    CT_REQUIRE_VK(t, vkGetQueryPoolResults(r.e.device, pool, 0, 2, sizeof(sizes), sizes, sizeof(sizes[0]),
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    CT_REQUIREF(t, sizes[0] > 0 && sizes[0] <= r.square.size && sizes[1] > 0 && sizes[1] <= r.box.size,
                "compacted sizes %llu and %llu of structures created with %llu and %llu",
                (unsigned long long)sizes[0], (unsigned long long)sizes[1], (unsigned long long)r.square.size,
                (unsigned long long)r.box.size);

    cb = ct_begin_commands(t, &r.e);
    for (i = 0; i < 2; i++) {
        rt_accel_init(t, &r, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizes[i], &compact[i]);
        memset(&copy, 0, sizeof(copy));
        copy.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
        copy.src = sources[i];
        copy.dst = compact[i].handle;
        copy.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
        r.fn.copy(cb, &copy);
    }
    rt_barrier(cb);
    ct_submit_and_wait(t, &r.e, cb);

    /* a top level of the compacted copies, and a clone of it */
    rt_write_instances(&r, rt_scene, compact[0].address, compact[1].address);
    rt_geometry(&r, 2, &g);
    rt_build(t, &r, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, 0, &g, RT_INSTANCES, &top);
    rt_write_descriptor(&r, top.handle);
    rt_trace(t, &r, rt_scene, 0, 0xff, 0, "instances of the compacted copies");
    rt_accel_init(t, &r, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, top.size, &clone);
    memset(&copy, 0, sizeof(copy));
    copy.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
    copy.src = top.handle;
    copy.dst = clone.handle;
    copy.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR;
    cb = ct_begin_commands(t, &r.e);
    r.fn.copy(cb, &copy);
    rt_barrier(cb);
    ct_submit_and_wait(t, &r.e, cb);
    rt_write_descriptor(&r, clone.handle);
    rt_trace(t, &r, rt_scene, 0, 0xff, 1, "the clone of the top level, primitive 1 ignored");

    rt_accel_fini(&r, &clone);
    rt_accel_fini(&r, &top);
    rt_accel_fini(&r, &compact[1]);
    rt_accel_fini(&r, &compact[0]);
    vkDestroyQueryPool(r.e.device, pool, NULL);
    rt_fini(&r);
}

static const struct ct_test *const tests[] = {
    CT_REF(accel_feature_requirements),
    CT_REF(accel_limit_requirements),
    CT_REF(accel_limit_primitive_count),
    CT_REF(deferred_operation_result),
    CT_REF(accel_build_sizes_ignored_members),
    CT_REF(ray_query_committed_hit),
    CT_REF(ray_query_cull_mask),
    CT_REF(ray_query_candidate_confirmation),
    CT_REF(accel_update_instances),
    CT_REF(accel_compacting_copy),
};

CT_SUITE(accelstructs, tests)
