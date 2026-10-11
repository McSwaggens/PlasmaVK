/* Chapter "Dispatching Commands": which workgroups vkCmdDispatchIndirect and
 * vkCmdDispatchBase assemble, and the WorkgroupId, NumWorkgroups and
 * GlobalInvocationId their invocations see, recorded by a compute shader
 * into a grid of cells, one per WorkgroupId.
 *
 * Shaders: GLSL in the comments, compiled with
 *   glslangValidator -V --target-env vulkan1.3 -x */
#include "ct.h"

#define GRID 8                     /* WorkgroupId components recorded: 0 .. 7 */
#define CELLS (GRID * GRID * GRID) /* 4 words each, then the out-of-grid counter */
#define REGION_WORDS (CELLS * 4 + 1)
#define REGION_STRIDE 16384 /* a multiple of minStorageBufferOffsetAlignment (at most 256) */
#define REGIONS 4

#define DP_COMMANDS                                                                                                \
    "vkCreateDescriptorSetLayout vkDestroyDescriptorSetLayout vkCreateDescriptorPool vkDestroyDescriptorPool "    \
    "vkAllocateDescriptorSets vkUpdateDescriptorSets vkCreatePipelineLayout vkDestroyPipelineLayout "             \
    "vkCreateShaderModule vkDestroyShaderModule vkCreateComputePipelines vkDestroyPipeline vkCmdBindPipeline "    \
    "vkCmdBindDescriptorSets vkCmdPipelineBarrier vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements " \
    "vkAllocateMemory vkBindBufferMemory vkMapMemory vkFreeMemory "

/*
 * #version 450
 * // Per workgroup of an 8 x 8 x 8 grid of WorkgroupId values, 4 words: the
 * // number of invocations that ran (atomically), NumWorkgroups and each
 * // invocation's GlobalInvocationId (x | y << 8 | z << 16). Workgroups outside
 * // the grid count in word 2048.
 * layout(local_size_x = 2) in;
 * layout(std430, set = 0, binding = 0) buffer Out { uint w[]; } o;
 * void main()
 * {
 *     uvec3 g = gl_WorkGroupID;
 *     if (g.x < 8u && g.y < 8u && g.z < 8u) {
 *         uint c = ((g.z * 8u + g.y) * 8u + g.x) * 4u;
 *         atomicAdd(o.w[c], 1u);
 *         o.w[c + 1u] = gl_NumWorkGroups.x | (gl_NumWorkGroups.y << 8) | (gl_NumWorkGroups.z << 16);
 *         o.w[c + 2u + gl_LocalInvocationID.x] =
 *             gl_GlobalInvocationID.x | (gl_GlobalInvocationID.y << 8) | (gl_GlobalInvocationID.z << 16);
 *     } else {
 *         atomicAdd(o.w[2048], 1u);
 *     }
 * }
 */
static const uint32_t grid_comp[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000063, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x000a000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00000034, 0x0000003d,
    0x0000004e, 0x00000052, 0x00060010, 0x00000004, 0x00000011, 0x00000002, 0x00000001, 0x00000001,
    0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005,
    0x00000009, 0x00000067, 0x00060005, 0x0000000b, 0x575f6c67, 0x476b726f, 0x70756f72, 0x00004449,
    0x00030005, 0x00000024, 0x00000063, 0x00030005, 0x00000032, 0x0074754f, 0x00040006, 0x00000032,
    0x00000000, 0x00000077, 0x00030005, 0x00000034, 0x0000006f, 0x00070005, 0x0000003d, 0x4e5f6c67,
    0x6f576d75, 0x72476b72, 0x7370756f, 0x00000000, 0x00080005, 0x0000004e, 0x4c5f6c67, 0x6c61636f,
    0x6f766e49, 0x69746163, 0x44496e6f, 0x00000000, 0x00080005, 0x00000052, 0x475f6c67, 0x61626f6c,
    0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00040047, 0x0000000b, 0x0000000b, 0x0000001a,
    0x00040047, 0x00000031, 0x00000006, 0x00000004, 0x00030047, 0x00000032, 0x00000002, 0x00050048,
    0x00000032, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000034, 0x00000021, 0x00000000,
    0x00040047, 0x00000034, 0x00000022, 0x00000000, 0x00040047, 0x0000003d, 0x0000000b, 0x00000018,
    0x00040047, 0x0000004e, 0x0000000b, 0x0000001b, 0x00040047, 0x00000052, 0x0000000b, 0x0000001c,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020,
    0x00000000, 0x00040017, 0x00000007, 0x00000006, 0x00000003, 0x00040020, 0x00000008, 0x00000007,
    0x00000007, 0x00040020, 0x0000000a, 0x00000001, 0x00000007, 0x0004003b, 0x0000000a, 0x0000000b,
    0x00000001, 0x00020014, 0x0000000d, 0x0004002b, 0x00000006, 0x0000000e, 0x00000000, 0x00040020,
    0x0000000f, 0x00000007, 0x00000006, 0x0004002b, 0x00000006, 0x00000012, 0x00000008, 0x0004002b,
    0x00000006, 0x00000016, 0x00000001, 0x0004002b, 0x00000006, 0x0000001d, 0x00000002, 0x0004002b,
    0x00000006, 0x0000002f, 0x00000004, 0x0003001d, 0x00000031, 0x00000006, 0x0003001e, 0x00000032,
    0x00000031, 0x00040020, 0x00000033, 0x0000000c, 0x00000032, 0x0004003b, 0x00000033, 0x00000034,
    0x0000000c, 0x00040015, 0x00000035, 0x00000020, 0x00000001, 0x0004002b, 0x00000035, 0x00000036,
    0x00000000, 0x00040020, 0x00000038, 0x0000000c, 0x00000006, 0x0004003b, 0x0000000a, 0x0000003d,
    0x00000001, 0x00040020, 0x0000003e, 0x00000001, 0x00000006, 0x0004002b, 0x00000035, 0x00000043,
    0x00000008, 0x0004002b, 0x00000035, 0x00000048, 0x00000010, 0x0004003b, 0x0000000a, 0x0000004e,
    0x00000001, 0x0004003b, 0x0000000a, 0x00000052, 0x00000001, 0x0004002b, 0x00000035, 0x0000005f,
    0x00000800, 0x0006002c, 0x00000007, 0x00000062, 0x0000001d, 0x00000016, 0x00000016, 0x00050036,
    0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000008,
    0x00000009, 0x00000007, 0x0004003b, 0x0000000f, 0x00000024, 0x00000007, 0x0004003d, 0x00000007,
    0x0000000c, 0x0000000b, 0x0003003e, 0x00000009, 0x0000000c, 0x00050041, 0x0000000f, 0x00000010,
    0x00000009, 0x0000000e, 0x0004003d, 0x00000006, 0x00000011, 0x00000010, 0x000500b0, 0x0000000d,
    0x00000013, 0x00000011, 0x00000012, 0x000300f7, 0x00000015, 0x00000000, 0x000400fa, 0x00000013,
    0x00000014, 0x00000015, 0x000200f8, 0x00000014, 0x00050041, 0x0000000f, 0x00000017, 0x00000009,
    0x00000016, 0x0004003d, 0x00000006, 0x00000018, 0x00000017, 0x000500b0, 0x0000000d, 0x00000019,
    0x00000018, 0x00000012, 0x000200f9, 0x00000015, 0x000200f8, 0x00000015, 0x000700f5, 0x0000000d,
    0x0000001a, 0x00000013, 0x00000005, 0x00000019, 0x00000014, 0x000300f7, 0x0000001c, 0x00000000,
    0x000400fa, 0x0000001a, 0x0000001b, 0x0000001c, 0x000200f8, 0x0000001b, 0x00050041, 0x0000000f,
    0x0000001e, 0x00000009, 0x0000001d, 0x0004003d, 0x00000006, 0x0000001f, 0x0000001e, 0x000500b0,
    0x0000000d, 0x00000020, 0x0000001f, 0x00000012, 0x000200f9, 0x0000001c, 0x000200f8, 0x0000001c,
    0x000700f5, 0x0000000d, 0x00000021, 0x0000001a, 0x00000015, 0x00000020, 0x0000001b, 0x000300f7,
    0x00000023, 0x00000000, 0x000400fa, 0x00000021, 0x00000022, 0x0000005e, 0x000200f8, 0x00000022,
    0x00050041, 0x0000000f, 0x00000025, 0x00000009, 0x0000001d, 0x0004003d, 0x00000006, 0x00000026,
    0x00000025, 0x00050084, 0x00000006, 0x00000027, 0x00000026, 0x00000012, 0x00050041, 0x0000000f,
    0x00000028, 0x00000009, 0x00000016, 0x0004003d, 0x00000006, 0x00000029, 0x00000028, 0x00050080,
    0x00000006, 0x0000002a, 0x00000027, 0x00000029, 0x00050084, 0x00000006, 0x0000002b, 0x0000002a,
    0x00000012, 0x00050041, 0x0000000f, 0x0000002c, 0x00000009, 0x0000000e, 0x0004003d, 0x00000006,
    0x0000002d, 0x0000002c, 0x00050080, 0x00000006, 0x0000002e, 0x0000002b, 0x0000002d, 0x00050084,
    0x00000006, 0x00000030, 0x0000002e, 0x0000002f, 0x0003003e, 0x00000024, 0x00000030, 0x0004003d,
    0x00000006, 0x00000037, 0x00000024, 0x00060041, 0x00000038, 0x00000039, 0x00000034, 0x00000036,
    0x00000037, 0x000700ea, 0x00000006, 0x0000003a, 0x00000039, 0x00000016, 0x0000000e, 0x00000016,
    0x0004003d, 0x00000006, 0x0000003b, 0x00000024, 0x00050080, 0x00000006, 0x0000003c, 0x0000003b,
    0x00000016, 0x00050041, 0x0000003e, 0x0000003f, 0x0000003d, 0x0000000e, 0x0004003d, 0x00000006,
    0x00000040, 0x0000003f, 0x00050041, 0x0000003e, 0x00000041, 0x0000003d, 0x00000016, 0x0004003d,
    0x00000006, 0x00000042, 0x00000041, 0x000500c4, 0x00000006, 0x00000044, 0x00000042, 0x00000043,
    0x000500c5, 0x00000006, 0x00000045, 0x00000040, 0x00000044, 0x00050041, 0x0000003e, 0x00000046,
    0x0000003d, 0x0000001d, 0x0004003d, 0x00000006, 0x00000047, 0x00000046, 0x000500c4, 0x00000006,
    0x00000049, 0x00000047, 0x00000048, 0x000500c5, 0x00000006, 0x0000004a, 0x00000045, 0x00000049,
    0x00060041, 0x00000038, 0x0000004b, 0x00000034, 0x00000036, 0x0000003c, 0x0003003e, 0x0000004b,
    0x0000004a, 0x0004003d, 0x00000006, 0x0000004c, 0x00000024, 0x00050080, 0x00000006, 0x0000004d,
    0x0000004c, 0x0000001d, 0x00050041, 0x0000003e, 0x0000004f, 0x0000004e, 0x0000000e, 0x0004003d,
    0x00000006, 0x00000050, 0x0000004f, 0x00050080, 0x00000006, 0x00000051, 0x0000004d, 0x00000050,
    0x00050041, 0x0000003e, 0x00000053, 0x00000052, 0x0000000e, 0x0004003d, 0x00000006, 0x00000054,
    0x00000053, 0x00050041, 0x0000003e, 0x00000055, 0x00000052, 0x00000016, 0x0004003d, 0x00000006,
    0x00000056, 0x00000055, 0x000500c4, 0x00000006, 0x00000057, 0x00000056, 0x00000043, 0x000500c5,
    0x00000006, 0x00000058, 0x00000054, 0x00000057, 0x00050041, 0x0000003e, 0x00000059, 0x00000052,
    0x0000001d, 0x0004003d, 0x00000006, 0x0000005a, 0x00000059, 0x000500c4, 0x00000006, 0x0000005b,
    0x0000005a, 0x00000048, 0x000500c5, 0x00000006, 0x0000005c, 0x00000058, 0x0000005b, 0x00060041,
    0x00000038, 0x0000005d, 0x00000034, 0x00000036, 0x00000051, 0x0003003e, 0x0000005d, 0x0000005c,
    0x000200f9, 0x00000023, 0x000200f8, 0x0000005e, 0x00060041, 0x00000038, 0x00000060, 0x00000034,
    0x00000036, 0x0000005f, 0x000700ea, 0x00000006, 0x00000061, 0x00000060, 0x00000016, 0x0000000e,
    0x00000016, 0x000200f9, 0x00000023, 0x000200f8, 0x00000023, 0x000100fd, 0x00010038,
};

struct dp {
    struct ct_env e;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkDescriptorPool pool;
    VkDescriptorSet sets[REGIONS];
    VkPipeline pipeline;
    struct ct_buffer out; /* REGIONS regions of REGION_WORDS words */
};

/* A device, grid_comp as a pipeline created with `flags`, and a descriptor
 * set per region of the result buffer. */
static void dp_init(struct ct *t, struct dp *d, VkPipelineCreateFlags flags)
{
    VkDescriptorSetLayoutBinding binding;
    VkDescriptorSetLayoutCreateInfo dsl;
    VkPipelineLayoutCreateInfo pl;
    VkDescriptorPoolSize size;
    VkDescriptorPoolCreateInfo dp;
    VkDescriptorSetLayout layouts[REGIONS];
    VkDescriptorSetAllocateInfo alloc;
    VkDescriptorBufferInfo infos[REGIONS];
    VkWriteDescriptorSet writes[REGIONS];
    VkComputePipelineCreateInfo info;
    VkShaderModule module;
    VkResult r;
    uint32_t i;

    ct_env_device(t, &d->e, NULL);
    ct_host_buffer(t, &d->e, REGION_STRIDE * REGIONS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &d->out);
    memset(&binding, 0, sizeof(binding));
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    memset(&dsl, 0, sizeof(dsl));
    dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = 1;
    dsl.pBindings = &binding;
    CT_REQUIRE_VK(t, vkCreateDescriptorSetLayout(d->e.device, &dsl, NULL, &d->set_layout));
    memset(&pl, 0, sizeof(pl));
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &d->set_layout;
    CT_REQUIRE_VK(t, vkCreatePipelineLayout(d->e.device, &pl, NULL, &d->layout));
    size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    size.descriptorCount = REGIONS;
    memset(&dp, 0, sizeof(dp));
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = REGIONS;
    dp.poolSizeCount = 1;
    dp.pPoolSizes = &size;
    CT_REQUIRE_VK(t, vkCreateDescriptorPool(d->e.device, &dp, NULL, &d->pool));
    for (i = 0; i < REGIONS; i++)
        layouts[i] = d->set_layout;
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = d->pool;
    alloc.descriptorSetCount = REGIONS;
    alloc.pSetLayouts = layouts;
    CT_REQUIRE_VK(t, vkAllocateDescriptorSets(d->e.device, &alloc, d->sets));
    memset(writes, 0, sizeof(writes));
    for (i = 0; i < REGIONS; i++) {
        infos[i].buffer = d->out.buffer;
        infos[i].offset = (VkDeviceSize)i * REGION_STRIDE;
        infos[i].range = REGION_WORDS * 4;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = d->sets[i];
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(d->e.device, REGIONS, writes, 0, NULL);
    module = ct_shader_module(t, &d->e, grid_comp, sizeof(grid_comp));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.flags = flags;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = d->layout;
    info.basePipelineIndex = -1;
    r = vkCreateComputePipelines(d->e.device, VK_NULL_HANDLE, 1, &info, NULL, &d->pipeline);
    vkDestroyShaderModule(d->e.device, module, NULL);
    CT_REQUIRE_VK(t, r);
}

static void dp_fini(struct dp *d)
{
    vkDestroyPipeline(d->e.device, d->pipeline, NULL);
    vkDestroyDescriptorPool(d->e.device, d->pool, NULL);
    vkDestroyPipelineLayout(d->e.device, d->layout, NULL);
    vkDestroyDescriptorSetLayout(d->e.device, d->set_layout, NULL);
    ct_buffer_fini(&d->e, &d->out);
    ct_env_fini(&d->e);
}

static void dp_bind(VkCommandBuffer cb, const struct dp *d, uint32_t region)
{
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->layout, 0, 1, &d->sets[region], 0, NULL);
}

static uint32_t dp_word(const struct dp *d, uint32_t region, uint32_t i)
{
    uint32_t w;
    memcpy(&w, d->out.data + (size_t)region * REGION_STRIDE + 4 * (size_t)i, 4);
    return w;
}

/* The region holds what a dispatch of workgroups [base, base + count) in
 * each dimension records; count of 0 in any dimension: nothing. */
static void dp_check(struct ct *t, const struct dp *d, uint32_t region, const uint32_t base[3], const uint32_t count[3],
                     const char *what)
{
    uint32_t x, y, z, l, bad = 0;
    uint32_t num = count[0] | (count[1] << 8) | (count[2] << 16);
    for (z = 0; z < GRID; z++)
        for (y = 0; y < GRID; y++)
            for (x = 0; x < GRID; x++) {
                uint32_t c = ((z * GRID + y) * GRID + x) * 4, want[4] = { 0, 0, 0, 0 };
                int in = x >= base[0] && x < base[0] + count[0] && y >= base[1] && y < base[1] + count[1] &&
                         z >= base[2] && z < base[2] + count[2];
                if (in) {
                    want[0] = 2;
                    want[1] = num;
                    for (l = 0; l < 2; l++)
                        want[2 + l] = (x * 2 + l) | (y << 8) | (z << 16);
                }
                for (l = 0; l < 4; l++)
                    if (dp_word(d, region, c + l) != want[l] && bad++ < 4)
                        ct_fail(t, __FILE__, __LINE__,
                                "%s: WorkgroupId (%u, %u, %u): %s is 0x%x, expected 0x%x", what, x, y, z,
                                l == 0 ? "invocations" : l == 1 ? "NumWorkgroups" : "GlobalInvocationId",
                                dp_word(d, region, c + l), want[l]);
            }
    CT_CHECKF(t, dp_word(d, region, CELLS * 4) == 0, "%s: %u invocations had a WorkgroupId outside [0, 8)", what,
              dp_word(d, region, CELLS * 4));
    if (bad > 4)
        ct_fail(t, __FILE__, __LINE__, "%s: %u words differ in all", what, bad);
}

CT_TEST(dispatch_indirect,
        "Dispatching Commands > vkCmdDispatchIndirect: the dispatch takes its x, y, z workgroup counts from the "
        "VkDispatchIndirectCommand at offset in the buffer, read by the device when the command executes (written "
        "earlier in the command buffer by vkCmdUpdateBuffer, after a barrier to VK_ACCESS_INDIRECT_COMMAND_READ_BIT), "
        "with the meaning of vkCmdDispatch's parameters: x * y * z workgroups, none if one is 0; NumWorkgroups "
        "equals them",
        DP_COMMANDS "vkCmdDispatchIndirect vkCmdUpdateBuffer")
{
    static const uint32_t a[3] = { 3, 2, 1 }, zero[3] = { 0, 5, 5 }, before[3] = { 1, 1, 1 }, after[3] = { 2, 2, 2 },
                          deep[3] = { 1, 3, 7 }, origin[3] = { 0, 0, 0 };
    struct dp d;
    struct ct_buffer args;
    VkMemoryBarrier mb;
    VkCommandBuffer cb;

    memset(&d, 0, sizeof(d));
    ct_env_init(t, &d.e);
    dp_init(t, &d, 0);
    ct_host_buffer(t, &d.e, 256, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &args);
    memcpy(args.data, a, sizeof(a));
    memcpy(args.data + 20, zero, sizeof(zero));
    memcpy(args.data + 44, before, sizeof(before));
    memcpy(args.data + 128, deep, sizeof(deep));
    cb = ct_begin_commands(t, &d.e);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipeline);
    dp_bind(cb, &d, 0);
    vkCmdDispatchIndirect(cb, args.buffer, 0);
    dp_bind(cb, &d, 1);
    vkCmdDispatchIndirect(cb, args.buffer, 20);
    vkCmdUpdateBuffer(cb, args.buffer, 44, sizeof(after), after);
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &mb, 0, NULL,
                         0, NULL);
    dp_bind(cb, &d, 2);
    vkCmdDispatchIndirect(cb, args.buffer, 44);
    dp_bind(cb, &d, 3);
    vkCmdDispatchIndirect(cb, args.buffer, 128);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &d.e, cb);
    dp_check(t, &d, 0, origin, a, "(3, 2, 1) at offset 0");
    dp_check(t, &d, 1, origin, zero, "(0, 5, 5) at offset 20");
    dp_check(t, &d, 2, origin, after, "(2, 2, 2) written over (1, 1, 1) by vkCmdUpdateBuffer");
    dp_check(t, &d, 3, origin, deep, "(1, 3, 7) at offset 128");
    ct_buffer_fini(&d.e, &args);
    dp_fini(&d);
}

CT_TEST(dispatch_base,
        "Dispatching Commands > vkCmdDispatchBase (Vulkan 1.1): groupCountX x groupCountY x groupCountZ "
        "workgroups with WorkgroupId in [baseGroup, baseGroup + groupCount) in each component, NumWorkgroups the "
        "group counts, GlobalInvocationId from the WorkgroupId; with a pipeline created with "
        "VK_PIPELINE_CREATE_DISPATCH_BASE_BIT (VUID-vkCmdDispatchBase-baseGroupX-00427); vkCmdDispatch is "
        "vkCmdDispatchBase(0, 0, 0, ...), also after a dispatch with another base",
        DP_COMMANDS "vkCmdDispatchBase vkCmdDispatch")
{
    static const uint32_t b0[3] = { 1, 2, 3 }, c0[3] = { 3, 2, 1 }, origin[3] = { 0, 0, 0 }, c1[3] = { 2, 2, 1 },
                          c2[3] = { 2, 1, 2 }, b3[3] = { 5, 0, 7 }, c3[3] = { 3, 8, 1 };
    struct dp d;
    VkCommandBuffer cb;

    memset(&d, 0, sizeof(d));
    ct_env_init(t, &d.e);
    if (d.e.version < VK_API_VERSION_1_1)
        ct_skip(t, "vkCmdDispatchBase is Vulkan 1.1");
    dp_init(t, &d, VK_PIPELINE_CREATE_DISPATCH_BASE_BIT);
    cb = ct_begin_commands(t, &d.e);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipeline);
    dp_bind(cb, &d, 0);
    vkCmdDispatchBase(cb, b0[0], b0[1], b0[2], c0[0], c0[1], c0[2]);
    dp_bind(cb, &d, 1);
    vkCmdDispatch(cb, c1[0], c1[1], c1[2]);
    dp_bind(cb, &d, 2);
    vkCmdDispatchBase(cb, 0, 0, 0, c2[0], c2[1], c2[2]);
    dp_bind(cb, &d, 3);
    vkCmdDispatchBase(cb, b3[0], b3[1], b3[2], c3[0], c3[1], c3[2]);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &d.e, cb);
    dp_check(t, &d, 0, b0, c0, "base (1, 2, 3), count (3, 2, 1)");
    dp_check(t, &d, 1, origin, c1, "vkCmdDispatch(2, 2, 1) after base (1, 2, 3)");
    dp_check(t, &d, 2, origin, c2, "base 0, count (2, 1, 2)");
    dp_check(t, &d, 3, b3, c3, "base (5, 0, 7), count (3, 8, 1)");
    dp_fini(&d);
}

CT_TEST(dispatch_indirect_after_base,
        "Dispatching Commands > vkCmdDispatchIndirect: it \"behaves similarly to vkCmdDispatch\", which is "
        "vkCmdDispatchBase(0, 0, 0, ...): after a vkCmdDispatchBase with a non-zero base (pipeline created with "
        "VK_PIPELINE_CREATE_DISPATCH_BASE_BIT, Vulkan 1.1), its workgroups' WorkgroupId still start at 0",
        DP_COMMANDS "vkCmdDispatchBase vkCmdDispatchIndirect")
{
    static const uint32_t b0[3] = { 5, 0, 7 }, c0[3] = { 3, 8, 1 }, c1[3] = { 2, 1, 1 }, b2[3] = { 1, 2, 3 },
                          c2[3] = { 1, 1, 1 }, c3[3] = { 1, 3, 7 }, origin[3] = { 0, 0, 0 };
    struct dp d;
    struct ct_buffer args;
    VkCommandBuffer cb;

    memset(&d, 0, sizeof(d));
    ct_env_init(t, &d.e);
    if (d.e.version < VK_API_VERSION_1_1)
        ct_skip(t, "vkCmdDispatchBase is Vulkan 1.1");
    dp_init(t, &d, VK_PIPELINE_CREATE_DISPATCH_BASE_BIT);
    ct_host_buffer(t, &d.e, 64, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, &args);
    memcpy(args.data, c1, sizeof(c1));
    memcpy(args.data + 16, c3, sizeof(c3));
    cb = ct_begin_commands(t, &d.e);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipeline);
    dp_bind(cb, &d, 0);
    vkCmdDispatchBase(cb, b0[0], b0[1], b0[2], c0[0], c0[1], c0[2]);
    dp_bind(cb, &d, 1);
    vkCmdDispatchIndirect(cb, args.buffer, 0);
    dp_bind(cb, &d, 2);
    vkCmdDispatchBase(cb, b2[0], b2[1], b2[2], c2[0], c2[1], c2[2]);
    dp_bind(cb, &d, 3);
    vkCmdDispatchIndirect(cb, args.buffer, 16);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &d.e, cb);
    dp_check(t, &d, 0, b0, c0, "base (5, 0, 7), count (3, 8, 1)");
    dp_check(t, &d, 1, origin, c1, "indirect (2, 1, 1) after base (5, 0, 7)");
    dp_check(t, &d, 2, b2, c2, "base (1, 2, 3), count (1, 1, 1)");
    dp_check(t, &d, 3, origin, c3, "indirect (1, 3, 7) after base (1, 2, 3)");
    ct_buffer_fini(&d.e, &args);
    dp_fini(&d);
}

static const struct ct_test *const tests[] = {
    CT_REF(dispatch_indirect),
    CT_REF(dispatch_base),
    CT_REF(dispatch_indirect_after_base),
};

CT_SUITE(dispatch, tests)
