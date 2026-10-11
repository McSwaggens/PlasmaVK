/* Chapter "Pipelines": graphics pipeline creation (without a fragment
 * shader, specialization, specialization constant operations, descriptor
 * sets), dynamic state (viewports and
 * scissors with and without counts, cull mode, front face, primitive
 * topology, depth and stencil state, rasterizer discard, depth bias),
 * pipeline caches (header, round trip, merging, partial data),
 * VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT and creation
 * feedback.
 *
 * Shaders: GLSL in the comments, compiled with
 *   glslangValidator -V --target-env vulkan1.3 -x */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

#define W CT_SIZE
#define H CT_SIZE

#define PL_COMMANDS CT_RENDER_COMMANDS "vkCmdBindPipeline vkCmdDraw vkCmdBeginRendering vkCmdEndRendering "

/*
 * #version 450
 * // Specialization constants pick the color.
 * layout(constant_id = 0) const float RED = 0.0;
 * layout(constant_id = 1) const uint GREEN = 0u;
 * layout(constant_id = 2) const bool BLUE = false;
 * layout(constant_id = 3) const int ALPHA = 255;
 * layout(location = 0) out vec4 o_color;
 * void main()
 * {
 *     const uint blue = BLUE ? 255u : 0u;
 *     o_color = vec4(RED, float(GREEN) / 255.0, float(blue) / 255.0, float(ALPHA) / 255.0);
 * }
 */
static const uint32_t spec_frag[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x0000001c, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
    0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00040005, 0x00000009, 0x6f635f6f, 0x00726f6c, 0x00030005, 0x0000000a, 0x00444552, 0x00040005,
    0x0000000c, 0x45455247, 0x0000004e, 0x00040005, 0x00000011, 0x45554c42, 0x00000000, 0x00040005,
    0x00000014, 0x65756c62, 0x00000000, 0x00040005, 0x00000018, 0x48504c41, 0x00000041, 0x00040047,
    0x00000009, 0x0000001e, 0x00000000, 0x00040047, 0x0000000a, 0x00000001, 0x00000000, 0x00040047,
    0x0000000c, 0x00000001, 0x00000001, 0x00040047, 0x00000011, 0x00000001, 0x00000002, 0x00040047,
    0x00000018, 0x00000001, 0x00000003, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
    0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020,
    0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00040032,
    0x00000006, 0x0000000a, 0x00000000, 0x00040015, 0x0000000b, 0x00000020, 0x00000000, 0x00040032,
    0x0000000b, 0x0000000c, 0x00000000, 0x0004002b, 0x00000006, 0x0000000e, 0x437f0000, 0x00020014,
    0x00000010, 0x00030031, 0x00000010, 0x00000011, 0x0004002b, 0x0000000b, 0x00000012, 0x000000ff,
    0x0004002b, 0x0000000b, 0x00000013, 0x00000000, 0x00070034, 0x0000000b, 0x00000014, 0x000000a9,
    0x00000011, 0x00000012, 0x00000013, 0x00040015, 0x00000017, 0x00000020, 0x00000001, 0x00040032,
    0x00000017, 0x00000018, 0x000000ff, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005, 0x00040070, 0x00000006, 0x0000000d, 0x0000000c, 0x00050088, 0x00000006,
    0x0000000f, 0x0000000d, 0x0000000e, 0x00040070, 0x00000006, 0x00000015, 0x00000014, 0x00050088,
    0x00000006, 0x00000016, 0x00000015, 0x0000000e, 0x0004006f, 0x00000006, 0x00000019, 0x00000018,
    0x00050088, 0x00000006, 0x0000001a, 0x00000019, 0x0000000e, 0x00070050, 0x00000007, 0x0000001b,
    0x0000000a, 0x0000000f, 0x00000016, 0x0000001a, 0x0003003e, 0x00000009, 0x0000001b, 0x000100fd,
    0x00010038,
};

/*
 * #version 450
 * // The color of a uniform buffer.
 * layout(set = 0, binding = 0) uniform U { vec4 color; } u;
 * layout(location = 0) out vec4 o_color;
 * void main()
 * {
 *     o_color = u.color;
 * }
 */
static const uint32_t ubo_frag[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000012, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000c, 0x00030010,
    0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00040005, 0x00000009, 0x6f635f6f, 0x00726f6c, 0x00030005, 0x0000000a, 0x00000055,
    0x00050006, 0x0000000a, 0x00000000, 0x6f6c6f63, 0x00000072, 0x00030005, 0x0000000c, 0x00000075,
    0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00030047, 0x0000000a, 0x00000002, 0x00050048,
    0x0000000a, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x0000000c, 0x00000021, 0x00000000,
    0x00040047, 0x0000000c, 0x00000022, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
    0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004,
    0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003,
    0x0003001e, 0x0000000a, 0x00000007, 0x00040020, 0x0000000b, 0x00000002, 0x0000000a, 0x0004003b,
    0x0000000b, 0x0000000c, 0x00000002, 0x00040015, 0x0000000d, 0x00000020, 0x00000001, 0x0004002b,
    0x0000000d, 0x0000000e, 0x00000000, 0x00040020, 0x0000000f, 0x00000002, 0x00000007, 0x00050036,
    0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041, 0x0000000f,
    0x00000010, 0x0000000c, 0x0000000e, 0x0004003d, 0x00000007, 0x00000011, 0x00000010, 0x0003003e,
    0x00000009, 0x00000011, 0x000100fd, 0x00010038,
};
/*
 * #version 450
 * // Specialization constant operations on floats: selects of a float and a
 * // vector, a composite extract and a vector shuffle.
 * layout(constant_id = 0) const bool B = false;
 * layout(constant_id = 1) const float F = 0.25;
 * const float f = B ? 1.0 : 0.0;
 * const vec4 c = B ? vec4(F, 0.5, 0.0, 1.0) : vec4(0.0, F, 1.0, 1.0);
 * const float g = c.y;
 * const vec2 sw = c.wx;
 * layout(location = 0) out vec4 o_color;
 * void main()
 * {
 *     o_color = vec4(f, g, sw.y, sw.x);
 * }
 */
static const uint32_t spec_ops_frag[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x0000001d, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
    0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00040005, 0x00000009, 0x6f635f6f, 0x00726f6c, 0x00030005, 0x0000000b, 0x00000042, 0x00030005,
    0x0000000e, 0x00000066, 0x00030005, 0x0000000f, 0x00000046, 0x00030005, 0x00000013, 0x00000063,
    0x00030005, 0x00000016, 0x00000067, 0x00030005, 0x00000018, 0x00007773, 0x00040047, 0x00000009,
    0x0000001e, 0x00000000, 0x00040047, 0x0000000b, 0x00000001, 0x00000000, 0x00040047, 0x0000000f,
    0x00000001, 0x00000001, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
    0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008,
    0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00020014, 0x0000000a,
    0x00030031, 0x0000000a, 0x0000000b, 0x0004002b, 0x00000006, 0x0000000c, 0x3f800000, 0x0004002b,
    0x00000006, 0x0000000d, 0x00000000, 0x00070034, 0x00000006, 0x0000000e, 0x000000a9, 0x0000000b,
    0x0000000c, 0x0000000d, 0x00040032, 0x00000006, 0x0000000f, 0x3e800000, 0x0004002b, 0x00000006,
    0x00000010, 0x3f000000, 0x00070033, 0x00000007, 0x00000011, 0x0000000f, 0x00000010, 0x0000000d,
    0x0000000c, 0x00070033, 0x00000007, 0x00000012, 0x0000000d, 0x0000000f, 0x0000000c, 0x0000000c,
    0x00070034, 0x00000007, 0x00000013, 0x000000a9, 0x0000000b, 0x00000011, 0x00000012, 0x00040015,
    0x00000014, 0x00000020, 0x00000000, 0x0004002b, 0x00000014, 0x00000015, 0x00000001, 0x00060034,
    0x00000006, 0x00000016, 0x00000051, 0x00000013, 0x00000001, 0x00040017, 0x00000017, 0x00000006,
    0x00000002, 0x00080034, 0x00000017, 0x00000018, 0x0000004f, 0x00000013, 0x00000013, 0x00000003,
    0x00000000, 0x00060034, 0x00000006, 0x00000019, 0x00000051, 0x00000018, 0x00000001, 0x0004002b,
    0x00000014, 0x0000001a, 0x00000000, 0x00060034, 0x00000006, 0x0000001b, 0x00000051, 0x00000018,
    0x00000000, 0x00070033, 0x00000007, 0x0000001c, 0x0000000e, 0x00000016, 0x00000019, 0x0000001b,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0003003e,
    0x00000009, 0x0000001c, 0x000100fd, 0x00010038,
};

static uint16_t expect_depth_left(uint32_t x, uint32_t y)
{
    (void)y;
    return x < W / 2 ? 16384 : 65535; /* 0.25 drawn, 1 cleared */
}

CT_TEST(pipeline_without_fragment_shader,
        "Pipelines > Graphics Pipelines: a graphics pipeline may omit the fragment shader stage; rasterization, "
        "the depth test and depth writes still happen",
        PL_COMMANDS)
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
    g.rendering.depthAttachmentFormat = VK_FORMAT_D16_UNORM;
    g.ds.depthTestEnable = g.ds.depthWriteEnable = VK_TRUE;
    g.ds.depthCompareOp = VK_COMPARE_OP_LESS;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, NULL, 0, NULL, &d, 1.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 1.0f, NULL, 0.25f);
    vkCmdEndRendering(cb);
    ct_target_read(cb, &d, 0, VK_IMAGE_ASPECT_DEPTH_BIT, s.out.buffer, 0);
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_d16(t, s.out.data, W, H, expect_depth_left, "depth");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_target_fini(&s.e, &d);
    ct_scene_fini(&s);
}

static uint32_t expect_spec(uint32_t x, uint32_t y, const void *ctx)
{
    (void)ctx;
    (void)y;
    return x < W / 2 ? CT_RGBA(255, 128, 255, 255) : CT_BLACK;
}

CT_TEST(pipeline_specialization,
        "Pipelines > Specialization Constants: VkSpecializationInfo replaces the default values of the "
        "specialization constants it maps (float, uint and bool); unmapped ones keep their defaults",
        PL_COMMANDS)
{
    struct spec {
        float red;
        uint32_t green;
        VkBool32 blue;
    } data = { 1.0f, 128, VK_TRUE };
    VkSpecializationMapEntry map[3] = {
        { 0, offsetof(struct spec, red), sizeof(float) },
        { 1, offsetof(struct spec, green), sizeof(uint32_t) },
        { 2, offsetof(struct spec, blue), sizeof(VkBool32) },
    };
    VkSpecializationInfo info = { 3, map, sizeof(data), &data };
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkShaderModule fs;
    VkPipeline specialized, defaults;
    VkCommandBuffer cb;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    fs = ct_shader_module(t, &s.e, spec_frag, sizeof(spec_frag));
    ct_gfx_init(&g, s.vs, fs, s.layout, W, H);
    defaults = ct_gfx_pipeline(t, &s.e, &g);
    g.fs_spec = &info;
    specialized = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_white, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, specialized);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 1.0f, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, defaults);
    ct_draw_rect(cb, s.layout, 0.0f, -1.0f, 1.0f, 1.0f, NULL, 0.0f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, expect_spec, NULL, 1, "specialized (left), defaults (right)");
    vkDestroyPipeline(s.e.device, specialized, NULL);
    vkDestroyPipeline(s.e.device, defaults, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

static uint32_t expect_spec_ops(uint32_t x, uint32_t y, const void *ctx)
{
    (void)ctx;
    (void)y;
    return x < W / 2 ? CT_RGBA(255, 128, 191, 255) : CT_RGBA(0, 64, 0, 255);
}

CT_TEST(pipeline_specialization_ops,
        "Pipelines > Specialization Constants: specialization constant operations (OpSpecConstantOp) are "
        "evaluated with the specialized values: Select of a float and of a vector, CompositeExtract and "
        "VectorShuffle of floats",
        PL_COMMANDS)
{
    struct spec {
        VkBool32 b;
        float f;
    } data = { VK_TRUE, 0.75f };
    VkSpecializationMapEntry map[2] = {
        { 0, offsetof(struct spec, b), sizeof(VkBool32) },
        { 1, offsetof(struct spec, f), sizeof(float) },
    };
    VkSpecializationInfo info = { 2, map, sizeof(data), &data };
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkShaderModule fs;
    VkPipeline specialized, defaults;
    VkCommandBuffer cb;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    fs = ct_shader_module(t, &s.e, spec_ops_frag, sizeof(spec_ops_frag));
    ct_gfx_init(&g, s.vs, fs, s.layout, W, H);
    defaults = ct_gfx_pipeline(t, &s.e, &g);
    g.fs_spec = &info;
    specialized = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_white, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, specialized);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 1.0f, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, defaults);
    ct_draw_rect(cb, s.layout, 0.0f, -1.0f, 1.0f, 1.0f, NULL, 0.0f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, expect_spec_ops, NULL, 1,
                   "specialized: (1, 0.5, 0.75, 1) (left); defaults: (0, 0.25, 0, 1) (right)");
    vkDestroyPipeline(s.e.device, specialized, NULL);
    vkDestroyPipeline(s.e.device, defaults, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(pipeline_uniform_buffer,
        "Pipelines > Pipeline Layouts, Resource Descriptors: a fragment shader reads a uniform buffer through a "
        "descriptor set bound with vkCmdBindDescriptorSets at the graphics bind point",
        PL_COMMANDS "vkCreateDescriptorSetLayout vkDestroyDescriptorSetLayout vkCreateDescriptorPool "
                    "vkDestroyDescriptorPool vkAllocateDescriptorSets vkUpdateDescriptorSets vkCmdBindDescriptorSets")
{
    static const float color[4] = { 0.25f, 0.5f, 0.75f, 1.0f };
    struct ct_scene s;
    struct ct_target c;
    struct ct_buffer ubo;
    struct ct_gfx g;
    VkDescriptorSetLayoutBinding binding;
    VkDescriptorSetLayoutCreateInfo dsl;
    VkDescriptorPoolSize size;
    VkDescriptorPoolCreateInfo dpi;
    VkDescriptorSetAllocateInfo dsa;
    VkDescriptorBufferInfo bi;
    VkWriteDescriptorSet write;
    VkDescriptorSetLayout set_layout;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    VkPipelineLayout layout;
    VkShaderModule fs;
    VkPipeline pipe;
    VkCommandBuffer cb;
    struct ct_box want = { 0, 0, W, H / 2, CT_RGBA(64, 128, 191, 255), CT_BLACK };

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo);
    memcpy(ubo.data, color, sizeof(color));
    memset(&binding, 0, sizeof(binding));
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    memset(&dsl, 0, sizeof(dsl));
    dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = 1;
    dsl.pBindings = &binding;
    CT_REQUIRE_VK(t, vkCreateDescriptorSetLayout(s.e.device, &dsl, NULL, &set_layout));
    size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    size.descriptorCount = 1;
    memset(&dpi, 0, sizeof(dpi));
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &size;
    CT_REQUIRE_VK(t, vkCreateDescriptorPool(s.e.device, &dpi, NULL, &pool));
    memset(&dsa, 0, sizeof(dsa));
    dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = pool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &set_layout;
    CT_REQUIRE_VK(t, vkAllocateDescriptorSets(s.e.device, &dsa, &set));
    bi.buffer = ubo.buffer;
    bi.offset = 0;
    bi.range = 16;
    memset(&write, 0, sizeof(write));
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &bi;
    vkUpdateDescriptorSets(s.e.device, 1, &write, 0, NULL);
    layout = ct_pipeline_layout(t, &s.e, &set_layout, 1, sizeof(struct ct_rect_pc));
    fs = ct_shader_module(t, &s.e, ubo_frag, sizeof(ubo_frag));
    ct_gfx_init(&g, s.vs, fs, layout, W, H);
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
    ct_draw_rect(cb, layout, -1.0f, -1.0f, 1.0f, 0.0f, NULL, 0.0f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want, 1, "uniform buffer color in the top half");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, fs, NULL);
    vkDestroyPipelineLayout(s.e.device, layout, NULL);
    vkDestroyDescriptorPool(s.e.device, pool, NULL);
    vkDestroyDescriptorSetLayout(s.e.device, set_layout, NULL);
    ct_buffer_fini(&s.e, &ubo);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

static VkViewport viewport(float x, float y, float w, float h)
{
    VkViewport v;
    v.x = x;
    v.y = y;
    v.width = w;
    v.height = h;
    v.minDepth = 0.0f;
    v.maxDepth = 1.0f;
    return v;
}

static VkRect2D rect2d(int32_t x, int32_t y, uint32_t w, uint32_t h)
{
    VkRect2D r;
    r.offset.x = x;
    r.offset.y = y;
    r.extent.width = w;
    r.extent.height = h;
    return r;
}

CT_TEST(dynamic_viewport_scissor,
        "Pipelines > Dynamic State: with VK_DYNAMIC_STATE_VIEWPORT and VK_DYNAMIC_STATE_SCISSOR the pipeline's "
        "values are ignored and vkCmdSetViewport / vkCmdSetScissor apply to later draws; with "
        "VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT and VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT (Vulkan 1.3) "
        "vkCmdSetViewportWithCount / vkCmdSetScissorWithCount set the count too",
        PL_COMMANDS "vkCmdSetViewport vkCmdSetScissor vkCmdSetViewportWithCount vkCmdSetScissorWithCount")
{
    static const uint32_t want[4] = { CT_RED, CT_BLUE, CT_RED, CT_GREEN };
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkPipeline fixed_count, with_count;
    VkCommandBuffer cb;
    VkViewport v;
    VkRect2D r;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.viewports[0] = viewport(4.0f, 4.0f, 2.0f, 2.0f); /* ignored */
    g.scissors[0] = rect2d(0, 0, 1, 1);
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_VIEWPORT;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_SCISSOR;
    fixed_count = ct_gfx_pipeline(t, &s.e, &g);
    g.dynamic_count = 0;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT;
    g.viewport_count = 0;
    with_count = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, fixed_count);
    v = viewport(0.0f, 0.0f, W / 2, H); /* the whole NDC square into the left half */
    r = rect2d(0, 0, W, H);
    vkCmdSetViewport(cb, 0, 1, &v);
    vkCmdSetScissor(cb, 0, 1, &r);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, ct_red, 0.0f);
    v = viewport(0.0f, 0.0f, W, H);
    r = rect2d(W / 2, H / 2, W / 2, H / 2); /* the bottom right quadrant only */
    vkCmdSetViewport(cb, 0, 1, &v);
    vkCmdSetScissor(cb, 0, 1, &r);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, ct_green, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, with_count);
    r = rect2d(W / 2, 0, W / 2, H / 2); /* the top right quadrant only */
    vkCmdSetViewportWithCount(cb, 1, &v);
    vkCmdSetScissorWithCount(cb, 1, &r);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, ct_blue, 0.0f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_quadrants, want, 0, "color");
    vkDestroyPipeline(s.e.device, fixed_count, NULL);
    vkDestroyPipeline(s.e.device, with_count, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(dynamic_cull_front_face_topology,
        "Pipelines > Dynamic State, Rasterization > Basic Polygon Rasterization: with VK_DYNAMIC_STATE_CULL_MODE, "
        "VK_DYNAMIC_STATE_FRONT_FACE and VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY (Vulkan 1.3), vkCmdSetCullMode, "
        "vkCmdSetFrontFace and vkCmdSetPrimitiveTopology apply to later draws: a clockwise polygon (in framebuffer "
        "coordinates) is back-facing with VK_FRONT_FACE_COUNTER_CLOCKWISE and front-facing with "
        "VK_FRONT_FACE_CLOCKWISE; VK_CULL_MODE_FRONT_AND_BACK discards every polygon; six vertices make two "
        "triangles as a list and four as a strip",
        PL_COMMANDS "vkCmdSetCullMode vkCmdSetFrontFace vkCmdSetPrimitiveTopology")
{
    static const uint32_t want[4] = { CT_RED, CT_BLACK, CT_RED, CT_BLACK };
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkPipeline pipe;
    VkCommandBuffer cb;
    uint32_t got;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.rs.cullMode = VK_CULL_MODE_FRONT_AND_BACK; /* overridden */
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_CULL_MODE;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_FRONT_FACE;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdSetPrimitiveTopology(cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
    /* ct_rect_vert's strip runs (x0, y0), (x1, y0), (x0, y1), (x1, y1): with
     * y down in framebuffer coordinates, clockwise */
    vkCmdSetFrontFace(cb, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    vkCmdSetCullMode(cb, VK_CULL_MODE_NONE);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 0.0f, ct_red, 0.0f); /* top left: drawn */
    vkCmdSetCullMode(cb, VK_CULL_MODE_BACK_BIT);
    ct_draw_rect(cb, s.layout, 0.0f, -1.0f, 1.0f, 0.0f, ct_red, 0.0f); /* top right: back-facing, culled */
    vkCmdSetFrontFace(cb, VK_FRONT_FACE_CLOCKWISE);
    ct_draw_rect(cb, s.layout, -1.0f, 0.0f, 0.0f, 1.0f, ct_red, 0.0f); /* bottom left: front-facing, drawn */
    vkCmdSetCullMode(cb, VK_CULL_MODE_FRONT_AND_BACK);
    ct_draw_rect(cb, s.layout, 0.0f, 0.0f, 1.0f, 1.0f, ct_red, 0.0f); /* bottom right: culled */
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_quadrants, want, 0, "cull mode and front face");

    /* six vertices over the top right quadrant: as a list, (v0 v1 v2) and
     * (v3 v4 v5) = (x1 y1, x0 y0, x1 y0) leave the wedge under both diagonals
     * empty; as a strip they cover the quadrant */
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdSetCullMode(cb, VK_CULL_MODE_NONE);
    vkCmdSetFrontFace(cb, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    vkCmdSetPrimitiveTopology(cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    {
        struct ct_rect_pc pc;
        memset(&pc, 0, sizeof(pc));
        pc.rect[0] = 0.0f;
        pc.rect[1] = -1.0f;
        pc.rect[2] = 1.0f;
        pc.rect[3] = 0.0f;
        memcpy(pc.color, ct_green, sizeof(pc.color));
        vkCmdPushConstants(cb, s.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
        vkCmdDraw(cb, 6, 1, 0, 0);
        pc.rect[0] = -1.0f;
        pc.rect[2] = 0.0f;
        vkCmdPushConstants(cb, s.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
        vkCmdSetPrimitiveTopology(cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
        vkCmdDraw(cb, 6, 1, 0, 0);
    }
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    got = ct_texel32(s.out.data, W, 8, 0);
    CT_CHECKF(t, got == CT_GREEN, "triangle list: texel (8, 0) is %08x, expected green", got);
    got = ct_texel32(s.out.data, W, 12, 7);
    CT_CHECKF(t, got == CT_BLACK, "triangle list: texel (12, 7) is %08x, expected black (no triangle covers it)", got);
    got = ct_texel32(s.out.data, W, 4, 7);
    CT_CHECKF(t, got == CT_GREEN, "triangle strip: texel (4, 7) is %08x, expected green", got);
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(dynamic_depth_stencil_state,
        "Pipelines > Dynamic State, Fragment Operations: depth test enable, depth write enable, depth compare op, "
        "stencil test enable, stencil ops, compare mask, write mask and reference set with vkCmdSet* (Vulkan 1.3 "
        "extended dynamic state) decide, draw by draw, which fragments pass and what they write",
        PL_COMMANDS "vkCmdSetDepthTestEnable vkCmdSetDepthWriteEnable vkCmdSetDepthCompareOp "
                    "vkCmdSetStencilTestEnable vkCmdSetStencilOp vkCmdSetStencilReference "
                    "vkCmdSetStencilCompareMask vkCmdSetStencilWriteMask")
{
    static const uint32_t want[4] = { CT_WHITE, CT_WHITE, CT_BLUE, CT_RED };
    struct ct_scene s;
    struct ct_target c, ds;
    struct ct_gfx g;
    VkFormat format;
    VkPipeline pipe;
    VkCommandBuffer cb;
    const VkStencilFaceFlags both = VK_STENCIL_FACE_FRONT_AND_BACK;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    format = ct_depth_stencil_format(&s.e);
    ct_target(t, &s.e, format, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, &ds);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.rendering.depthAttachmentFormat = g.rendering.stencilAttachmentFormat = format;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_DEPTH_COMPARE_OP;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_STENCIL_OP;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_STENCIL_REFERENCE;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_STENCIL_WRITE_MASK;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, &ds, 0.5f); /* stencil cleared to 0 */
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdSetStencilTestEnable(cb, VK_FALSE);
    vkCmdSetStencilOp(cb, both, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
    vkCmdSetStencilCompareMask(cb, both, 0xff);
    vkCmdSetStencilWriteMask(cb, both, 0xff);
    vkCmdSetStencilReference(cb, both, 0);
    /* depth test off: everything red, depth untouched */
    vkCmdSetDepthTestEnable(cb, VK_FALSE);
    vkCmdSetDepthWriteEnable(cb, VK_FALSE);
    vkCmdSetDepthCompareOp(cb, VK_COMPARE_OP_NEVER);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, ct_red, 0.9f);
    /* LESS with writes: the left half at 0.25 */
    vkCmdSetDepthTestEnable(cb, VK_TRUE);
    vkCmdSetDepthWriteEnable(cb, VK_TRUE);
    vkCmdSetDepthCompareOp(cb, VK_COMPARE_OP_LESS);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 1.0f, ct_green, 0.25f);
    /* GREATER at 0.375 passes where 0.25 was written (left), fails on 0.5 */
    vkCmdSetDepthWriteEnable(cb, VK_FALSE);
    vkCmdSetDepthCompareOp(cb, VK_COMPARE_OP_GREATER);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, ct_blue, 0.375f);
    /* stencil: REPLACE with 1 over the top half (depth test off), then
     * EQUAL 1 lets white through there only */
    vkCmdSetDepthTestEnable(cb, VK_FALSE);
    vkCmdSetStencilTestEnable(cb, VK_TRUE);
    vkCmdSetStencilOp(cb, both, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
    vkCmdSetStencilReference(cb, both, 1);
    vkCmdSetStencilWriteMask(cb, both, 0x01);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 0.0f, ct_white, 0.0f);
    vkCmdSetStencilOp(cb, both, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_EQUAL);
    vkCmdSetStencilCompareMask(cb, both, 0x01);
    vkCmdSetStencilReference(cb, both, 0xff); /* & compare mask = 1 */
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, ct_white, 0.0f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_quadrants, want, 0, "color");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_target_fini(&s.e, &ds);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(dynamic_rasterizer_discard_depth_bias,
        "Pipelines > Dynamic State: VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE (vkCmdSetRasterizerDiscardEnable: "
        "primitives are discarded before rasterization) and VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE with "
        "VK_DYNAMIC_STATE_DEPTH_BIAS (vkCmdSetDepthBiasEnable, vkCmdSetDepthBias: o = r x constantFactor is "
        "added to the depth of each fragment, r being at most 2 x 2^-16 for D16_UNORM)",
        PL_COMMANDS "vkCmdSetRasterizerDiscardEnable vkCmdSetDepthBiasEnable vkCmdSetDepthBias")
{
    static const uint32_t want[4] = { CT_GREEN, CT_RED, CT_BLACK, CT_BLACK };
    struct ct_scene s;
    struct ct_target c, d;
    struct ct_gfx g;
    VkPipeline pipe;
    VkCommandBuffer cb;

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    ct_target(t, &s.e, VK_FORMAT_D16_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              &d);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.rendering.depthAttachmentFormat = VK_FORMAT_D16_UNORM;
    g.ds.depthTestEnable = g.ds.depthWriteEnable = VK_TRUE;
    g.ds.depthCompareOp = VK_COMPARE_OP_LESS;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE;
    g.dynamic[g.dynamic_count++] = VK_DYNAMIC_STATE_DEPTH_BIAS;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, &d, 1.0f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdSetDepthBias(cb, 1000.0f, 0.0f, 0.0f);
    /* top half at 0.5 + 1000 r (0.015 to 0.031) */
    vkCmdSetRasterizerDiscardEnable(cb, VK_FALSE);
    vkCmdSetDepthBiasEnable(cb, VK_TRUE);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 0.0f, ct_red, 0.5f);
    /* without bias: 0.505 passes LESS against the biased 0.5, 0.6 does not */
    vkCmdSetDepthBiasEnable(cb, VK_FALSE);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 0.0f, ct_green, 0.505f); /* < 0.515: passes */
    ct_draw_rect(cb, s.layout, 0.0f, -1.0f, 1.0f, 0.0f, ct_blue, 0.6f);     /* > 0.531: fails */
    /* discarded: nothing in the bottom half */
    vkCmdSetRasterizerDiscardEnable(cb, VK_TRUE);
    ct_draw_rect(cb, s.layout, -1.0f, 0.0f, 1.0f, 1.0f, ct_white, 0.0f);
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_quadrants, want, 0, "color");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_target_fini(&s.e, &d);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

/* A graphics pipeline for the cache tests (in a cache, with flags). */
static VkResult cache_pipeline(struct ct_scene *s, VkShaderModule fs, VkPipelineCache cache, VkPipelineCreateFlags flags,
                               VkPipeline *out)
{
    struct ct_gfx g;
    ct_gfx_init(&g, s->vs, fs, s->layout, W, H);
    g.cache = cache;
    g.flags = flags;
    return ct_gfx_create(&s->e, &g, out);
}

static VkPipelineCache new_cache(struct ct *t, struct ct_env *e, const void *data, size_t size)
{
    VkPipelineCacheCreateInfo info;
    VkPipelineCache cache = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    info.initialDataSize = size;
    info.pInitialData = data;
    CT_REQUIRE_VK(t, vkCreatePipelineCache(e->device, &info, NULL, &cache));
    return cache;
}

/* The header of pipeline cache data (Pipeline Cache Header). */
static void check_header(struct ct *t, const struct ct_env *e, const unsigned char *data, size_t size, const char *what)
{
    uint32_t header_size, version, vendor, device;
    CT_REQUIREF(t, size >= 32, "%s: %zu bytes, less than a header", what, size);
    memcpy(&header_size, data, 4);
    memcpy(&version, data + 4, 4);
    memcpy(&vendor, data + 8, 4);
    memcpy(&device, data + 12, 4);
    CT_CHECKF(t, header_size == 32, "%s: headerSize %u, must be 32", what, header_size);
    CT_CHECKF(t, version == VK_PIPELINE_CACHE_HEADER_VERSION_ONE, "%s: headerVersion %u", what, version);
    CT_CHECKF(t, vendor == e->props.vendorID, "%s: vendorID 0x%x, the device's is 0x%x", what, vendor,
              e->props.vendorID);
    CT_CHECKF(t, device == e->props.deviceID, "%s: deviceID 0x%x, the device's is 0x%x", what, device,
              e->props.deviceID);
    CT_CHECKF(t, !memcmp(data + 16, e->props.pipelineCacheUUID, VK_UUID_SIZE),
              "%s: pipelineCacheUUID is not VkPhysicalDeviceProperties::pipelineCacheUUID", what);
}

CT_TEST(pipeline_cache_data,
        "Pipelines > Pipeline Cache: vkGetPipelineCacheData returns the maximum size with pData NULL; the data "
        "begins with a VkPipelineCacheHeaderVersionOne (headerSize 32, VK_PIPELINE_CACHE_HEADER_VERSION_ONE, the "
        "device's vendorID, deviceID and pipelineCacheUUID); two calls retrieve the same data; with less room "
        "than the data, at most that much is written and VK_INCOMPLETE returned, and with less room than the "
        "header nothing is written and 0 returned in pDataSize",
        CT_RENDER_COMMANDS "vkCreatePipelineCache vkDestroyPipelineCache vkGetPipelineCacheData")
{
    struct ct_scene s;
    VkPipelineCache cache;
    VkPipeline pipe;
    unsigned char *a, *b, small[31];
    size_t size = 0, n;
    VkResult r;

    ct_scene_init(t, &s);
    cache = new_cache(t, &s.e, NULL, 0);
    CT_REQUIRE_VK(t, cache_pipeline(&s, s.fs, cache, 0, &pipe));
    CT_REQUIRE_VK(t, vkGetPipelineCacheData(s.e.device, cache, &size, NULL));
    CT_REQUIREF(t, size >= 32, "maximum size %zu: less than the header", size);
    a = malloc(size + 64);
    b = malloc(size + 64);
    CT_REQUIRE(t, a && b);
    memset(a, 0xab, size + 64);
    memset(b, 0xcd, size + 64);
    n = size;
    CT_CHECK_VK(t, vkGetPipelineCacheData(s.e.device, cache, &n, a), VK_SUCCESS);
    CT_CHECKF(t, n <= size, "wrote %zu bytes, more than the %zu returned as the maximum", n, size);
    check_header(t, &s.e, a, n, "data");
    CT_CHECKF(t, ct_all_bytes(a + size, 64, 0xab), "bytes written past pDataSize");
    size = n;
    n = size + 64;
    CT_CHECK_VK(t, vkGetPipelineCacheData(s.e.device, cache, &n, b), VK_SUCCESS);
    CT_CHECKF(t, n == size && !memcmp(a, b, size), "a second call retrieved different data (%zu bytes, then %zu)",
              size, n);
    if (size > 32) {
        n = size - 1;
        memset(b, 0xcd, size + 64);
        r = vkGetPipelineCacheData(s.e.device, cache, &n, b);
        CT_CHECKF(t, r == VK_INCOMPLETE, "with %zu of %zu bytes: %s, expected VK_INCOMPLETE", size - 1, size,
                  ct_result_name(r));
        CT_CHECKF(t, n <= size - 1, "with %zu bytes of room: %zu written", size - 1, n);
        CT_CHECKF(t, ct_all_bytes(b + size - 1, 65, 0xcd), "bytes written past pDataSize");
        if (n >= 32)
            check_header(t, &s.e, b, n, "partial data");
    }
    n = sizeof(small);
    memset(small, 0xcd, sizeof(small));
    r = vkGetPipelineCacheData(s.e.device, cache, &n, small);
    CT_CHECKF(t, r == VK_INCOMPLETE, "with 31 bytes of room: %s, expected VK_INCOMPLETE", ct_result_name(r));
    CT_CHECKF(t, n == 0, "with 31 bytes of room (less than the header): pDataSize %zu, expected 0", n);
    CT_CHECKF(t, ct_all_bytes(small, sizeof(small), 0xcd), "with 31 bytes of room: something was written");
    free(a);
    free(b);
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyPipelineCache(s.e.device, cache, NULL);
    ct_scene_fini(&s);
}

CT_TEST(pipeline_cache_round_trip,
        "Pipelines > Pipeline Cache: data from vkGetPipelineCacheData creates a cache (pInitialData) that "
        "pipelines can be created with; vkMergePipelineCaches merges caches into one whose data again begins "
        "with a valid header",
        PL_COMMANDS "vkCreatePipelineCache vkDestroyPipelineCache vkGetPipelineCacheData vkMergePipelineCaches")
{
    struct ct_scene s;
    struct ct_target c;
    VkPipelineCache first, loaded, merged, sources[2];
    VkPipeline pipes[2];
    VkCommandBuffer cb;
    unsigned char *data;
    size_t size = 0, n;
    uint32_t i;
    struct ct_box want = { 0, 0, W, H, CT_GREEN, CT_BLACK };

    ct_scene_init(t, &s);
    ct_scene_color(t, &s, &c);
    first = new_cache(t, &s.e, NULL, 0);
    CT_REQUIRE_VK(t, cache_pipeline(&s, s.fs, first, 0, &pipes[0]));
    CT_REQUIRE_VK(t, vkGetPipelineCacheData(s.e.device, first, &size, NULL));
    data = malloc(size);
    CT_REQUIRE(t, data != NULL);
    n = size;
    CT_REQUIRE_VK(t, vkGetPipelineCacheData(s.e.device, first, &n, data));
    loaded = new_cache(t, &s.e, data, n);
    CT_REQUIRE_VK(t, cache_pipeline(&s, s.fs, loaded, 0, &pipes[1]));
    merged = new_cache(t, &s.e, NULL, 0);
    sources[0] = first;
    sources[1] = loaded;
    CT_CHECK_VK(t, vkMergePipelineCaches(s.e.device, merged, 2, sources), VK_SUCCESS);
    free(data);
    size = 0;
    CT_REQUIRE_VK(t, vkGetPipelineCacheData(s.e.device, merged, &size, NULL));
    data = malloc(size + 1);
    CT_REQUIRE(t, data != NULL);
    n = size;
    CT_CHECK_VK(t, vkGetPipelineCacheData(s.e.device, merged, &n, data), VK_SUCCESS);
    check_header(t, &s.e, data, n, "merged cache");
    free(data);
    /* both pipelines draw */
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_black, NULL, 0.0f);
    for (i = 0; i < 2; i++) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[i]);
        ct_draw_rect(cb, s.layout, i ? 0.0f : -1.0f, -1.0f, i ? 1.0f : 0.0f, 1.0f, ct_green, 0.0f);
    }
    vkCmdEndRendering(cb);
    ct_scene_read(t, &s, cb, &c);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want, 0, "drawn with both pipelines");
    for (i = 0; i < 2; i++)
        vkDestroyPipeline(s.e.device, pipes[i], NULL);
    vkDestroyPipelineCache(s.e.device, first, NULL);
    vkDestroyPipelineCache(s.e.device, loaded, NULL);
    vkDestroyPipelineCache(s.e.device, merged, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(pipeline_cache_partial_data,
        "Pipelines > Pipeline Cache: data a vkGetPipelineCacheData call returned with VK_INCOMPLETE is valid "
        "initial data too (\"Any data written to pData is valid and can be provided as the pInitialData member "
        "of the VkPipelineCacheCreateInfo structure passed to vkCreatePipelineCache\"): the cache is created and "
        "pipelines can be created with it",
        CT_RENDER_COMMANDS "vkCreatePipelineCache vkDestroyPipelineCache vkGetPipelineCacheData")
{
    struct ct_scene s;
    VkPipelineCache first, partial;
    VkPipeline pipes[2];
    unsigned char *data;
    size_t size = 0, n;
    VkResult r;

    ct_scene_init(t, &s);
    first = new_cache(t, &s.e, NULL, 0);
    CT_REQUIRE_VK(t, cache_pipeline(&s, s.fs, first, 0, &pipes[0]));
    CT_REQUIRE_VK(t, vkGetPipelineCacheData(s.e.device, first, &size, NULL));
    if (size <= 32)
        ct_skip(t, "the cache holds no more than its header (%zu bytes)", size);
    data = malloc(size);
    CT_REQUIRE(t, data != NULL);
    n = size - 1;
    r = vkGetPipelineCacheData(s.e.device, first, &n, data);
    CT_REQUIREF(t, r == VK_INCOMPLETE, "with %zu of %zu bytes: %s", size - 1, size, ct_result_name(r));
    partial = new_cache(t, &s.e, data, n);
    CT_CHECK_VK(t, cache_pipeline(&s, s.fs, partial, 0, &pipes[1]), VK_SUCCESS);
    free(data);
    vkDestroyPipeline(s.e.device, pipes[0], NULL);
    vkDestroyPipeline(s.e.device, pipes[1], NULL);
    vkDestroyPipelineCache(s.e.device, first, NULL);
    vkDestroyPipelineCache(s.e.device, partial, NULL);
    ct_scene_fini(&s);
}

/* Checks the outcome of creating n pipelines with
 * VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT and
 * VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT; returns the number made. */
static uint32_t check_compile_required(struct ct *t, VkResult r, const VkPipeline *p, uint32_t n, const char *what)
{
    uint32_t i, made = 0;
    int failed = 0;
    CT_CHECKF(t, r == VK_SUCCESS || r == VK_PIPELINE_COMPILE_REQUIRED, "%s: %s, expected VK_SUCCESS or "
              "VK_PIPELINE_COMPILE_REQUIRED", what, ct_result_name(r));
    for (i = 0; i < n; i++) {
        if (p[i] == VK_NULL_HANDLE) {
            failed = 1;
            continue;
        }
        made++;
        CT_CHECKF(t, !failed, "%s: pipeline %u was made after one that failed (EARLY_RETURN_ON_FAILURE)", what, i);
    }
    CT_CHECKF(t, (r == VK_SUCCESS) == (made == n), "%s: %s with %u of %u pipelines made", what, ct_result_name(r), made,
              n);
    return made;
}

CT_TEST(pipeline_compile_required,
        "Pipelines > Pipeline Cache, Multiple Pipeline Creation: with "
        "VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT a pipeline that needs compiling is not made: "
        "VK_PIPELINE_COMPILE_REQUIRED and VK_NULL_HANDLE; with VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT the "
        "pipelines after a failed one are VK_NULL_HANDLE too (pipelineCreationCacheControl, required from Vulkan "
        "1.3)",
        CT_RENDER_COMMANDS "vkCreatePipelineCache vkDestroyPipelineCache")
{
    struct ct_scene s;
    struct ct_gfx g;
    VkGraphicsPipelineCreateInfo infos[3];
    VkPipelineShaderStageCreateInfo stages[2];
    VkPipelineVertexInputStateCreateInfo vi;
    VkPipelineViewportStateCreateInfo vp;
    VkPipelineDynamicStateCreateInfo dyn;
    VkPipelineCache cache;
    VkPipeline p[3], made;
    VkResult r;
    uint32_t i, k;

    ct_scene_init(t, &s);
    CT_REQUIRE(t, s.f.v13.pipelineCreationCacheControl);
    cache = new_cache(t, &s.e, NULL, 0);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.rendering.colorAttachmentCount = 1;
    g.rendering.pColorAttachmentFormats = g.color_formats;
    g.cb.attachmentCount = 1;
    g.cb.pAttachments = g.blend;
    memset(stages, 0, sizeof(stages));
    for (i = 0; i < 2; i++) {
        stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[i].stage = i ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT;
        stages[i].module = i ? s.fs : s.vs;
        stages[i].pName = "main";
    }
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    memset(&vp, 0, sizeof(vp));
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = vp.scissorCount = 1;
    vp.pViewports = g.viewports;
    vp.pScissors = g.scissors;
    memset(&dyn, 0, sizeof(dyn));
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    memset(infos, 0, sizeof(infos));
    for (i = 0; i < 3; i++) {
        infos[i].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        infos[i].pNext = &g.rendering;
        infos[i].flags = VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT |
                         VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT;
        infos[i].stageCount = 2;
        infos[i].pStages = stages;
        infos[i].pVertexInputState = &vi;
        infos[i].pInputAssemblyState = &g.ia;
        infos[i].pViewportState = &vp;
        infos[i].pRasterizationState = &g.rs;
        infos[i].pMultisampleState = &g.ms;
        infos[i].pDepthStencilState = &g.ds;
        infos[i].pColorBlendState = &g.cb;
        infos[i].pDynamicState = &dyn;
        infos[i].layout = s.layout;
        infos[i].basePipelineIndex = -1;
    }
    for (k = 0; k < 2; k++) {
        /* k = 0: never compiled; k = 1: after compiling it into the cache */
        for (i = 0; i < 3; i++)
            p[i] = (VkPipeline)(uintptr_t)0x1234; /* overwritten */
        r = vkCreateGraphicsPipelines(s.e.device, cache, 3, infos, NULL, p);
        made = VK_NULL_HANDLE;
        if (check_compile_required(t, r, p, 3, k ? "after compiling" : "never compiled") == 0 && !k) {
            infos[0].flags = 0;
            CT_CHECK_VK(t, vkCreateGraphicsPipelines(s.e.device, cache, 1, infos, NULL, &made), VK_SUCCESS);
            infos[0].flags = infos[1].flags;
        }
        for (i = 0; i < 3; i++)
            if (p[i] != VK_NULL_HANDLE && p[i] != (VkPipeline)(uintptr_t)0x1234)
                vkDestroyPipeline(s.e.device, p[i], NULL);
        if (made)
            vkDestroyPipeline(s.e.device, made, NULL);
    }
    vkDestroyPipelineCache(s.e.device, cache, NULL);
    ct_scene_fini(&s);
}

CT_TEST(pipeline_creation_feedback,
        "Pipelines > Pipeline Creation Feedback: with VkPipelineCreationFeedbackCreateInfo, an implementation "
        "that leaves VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT out of flags sets no other bit; stage feedback is "
        "written for the pipeline's stages only (Vulkan 1.3)",
        CT_RENDER_COMMANDS)
{
    struct ct_scene s;
    struct ct_gfx g;
    VkPipelineCreationFeedback pipe_fb, stage_fb[3];
    VkPipelineCreationFeedbackCreateInfo info;
    VkPipeline pipe;
    uint32_t i;

    ct_scene_init(t, &s);
    memset(&pipe_fb, 0, sizeof(pipe_fb));
    memset(stage_fb, 0, sizeof(stage_fb));
    stage_fb[2].flags = 0x5a5a5a5au;
    stage_fb[2].duration = 0x1234;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO;
    info.pPipelineCreationFeedback = &pipe_fb;
    info.pipelineStageCreationFeedbackCount = 2;
    info.pPipelineStageCreationFeedbacks = stage_fb;
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.next = &info;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    CT_CHECKF(t, (pipe_fb.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) || pipe_fb.flags == 0,
              "pipeline feedback flags 0x%x without VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT", pipe_fb.flags);
    for (i = 0; i < 2; i++)
        CT_CHECKF(t, (stage_fb[i].flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) || stage_fb[i].flags == 0,
                  "stage %u feedback flags 0x%x without VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT", i, stage_fb[i].flags);
    CT_CHECKF(t, stage_fb[2].flags == 0x5a5a5a5au && stage_fb[2].duration == 0x1234,
              "feedback written past pipelineStageCreationFeedbackCount");
    if (!(pipe_fb.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT))
        ct_note(t, "no pipeline creation feedback (allowed)");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_scene_fini(&s);
}

static const struct ct_test *const tests[] = {
    CT_REF(pipeline_without_fragment_shader),
    CT_REF(pipeline_specialization),
    CT_REF(pipeline_specialization_ops),
    CT_REF(pipeline_uniform_buffer),
    CT_REF(dynamic_viewport_scissor),
    CT_REF(dynamic_cull_front_face_topology),
    CT_REF(dynamic_depth_stencil_state),
    CT_REF(dynamic_rasterizer_discard_depth_bias),
    CT_REF(pipeline_cache_data),
    CT_REF(pipeline_cache_round_trip),
    CT_REF(pipeline_cache_partial_data),
    CT_REF(pipeline_compile_required),
    CT_REF(pipeline_creation_feedback),
};

CT_SUITE(pipelines, tests)
