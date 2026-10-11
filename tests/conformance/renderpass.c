/* Chapter "Render Pass": render pass objects (load and store operations,
 * render areas, subpasses with input attachments and dependencies, color
 * and depth resolves, multiview, imageless framebuffers, granularity) and
 * dynamic rendering (resolves, suspending and resuming, multiview, local
 * read), checked texel by texel on 16 x 16 targets.
 *
 * Shaders: GLSL in the comments, compiled with
 *   glslangValidator -V --target-env vulkan1.3 -x */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

#define W CT_SIZE
#define H CT_SIZE
#define RED CT_RED
#define GREEN CT_GREEN
#define BLUE CT_BLUE
#define BLACK CT_BLACK

#define RP_COMMANDS CT_RENDER_COMMANDS "vkCmdBindPipeline vkCmdDraw "

/*
 * #version 450
 * // The input attachment's texel, components reversed (BGRA).
 * layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_in;
 * layout(location = 0) out vec4 o_color;
 * void main()
 * {
 *     o_color = subpassLoad(u_in).bgra;
 * }
 */
static const uint32_t input_frag[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000014, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x00000028, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x00000000, 0x00000001, 0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009,
    0x0000000c, 0x00030010, 0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x00000009, 0x6f635f6f, 0x00726f6c, 0x00040005,
    0x0000000c, 0x6e695f75, 0x00000000, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00040047,
    0x0000000c, 0x00000021, 0x00000000, 0x00040047, 0x0000000c, 0x00000022, 0x00000000, 0x00040047,
    0x0000000c, 0x0000002b, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
    0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020,
    0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00090019,
    0x0000000a, 0x00000006, 0x00000006, 0x00000000, 0x00000000, 0x00000000, 0x00000002, 0x00000000,
    0x00040020, 0x0000000b, 0x00000000, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c, 0x00000000,
    0x00040015, 0x0000000e, 0x00000020, 0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000,
    0x00040017, 0x00000010, 0x0000000e, 0x00000002, 0x0005002c, 0x00000010, 0x00000011, 0x0000000f,
    0x0000000f, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
    0x0004003d, 0x0000000a, 0x0000000d, 0x0000000c, 0x00050062, 0x00000007, 0x00000012, 0x0000000d,
    0x00000011, 0x0009004f, 0x00000007, 0x00000013, 0x00000012, 0x00000012, 0x00000002, 0x00000001,
    0x00000000, 0x00000003, 0x0003003e, 0x00000009, 0x00000013, 0x000100fd, 0x00010038,
};

/*
 * #version 450
 * #extension GL_EXT_multiview : require
 * // Red (ViewIndex + 1) / 4.
 * layout(location = 0) out vec4 o_color;
 * void main()
 * {
 *     o_color = vec4(float(gl_ViewIndex + 1) * 0.25, 0.0, 0.0, 1.0);
 * }
 */
static const uint32_t view_frag[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x00000016, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x00001157, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x00000000, 0x00000001, 0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009,
    0x0000000c, 0x00030010, 0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00060004,
    0x455f4c47, 0x6d5f5458, 0x69746c75, 0x77656976, 0x00000000, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00040005, 0x00000009, 0x6f635f6f, 0x00726f6c, 0x00060005, 0x0000000c, 0x565f6c67,
    0x49776569, 0x7865646e, 0x00000000, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00040047,
    0x0000000c, 0x0000000b, 0x00001158, 0x00030047, 0x0000000c, 0x0000000e, 0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007,
    0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008,
    0x00000009, 0x00000003, 0x00040015, 0x0000000a, 0x00000020, 0x00000001, 0x00040020, 0x0000000b,
    0x00000001, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c, 0x00000001, 0x0004002b, 0x0000000a,
    0x0000000e, 0x00000001, 0x0004002b, 0x00000006, 0x00000011, 0x3e800000, 0x0004002b, 0x00000006,
    0x00000013, 0x00000000, 0x0004002b, 0x00000006, 0x00000014, 0x3f800000, 0x00050036, 0x00000002,
    0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x0000000a, 0x0000000d,
    0x0000000c, 0x00050080, 0x0000000a, 0x0000000f, 0x0000000d, 0x0000000e, 0x0004006f, 0x00000006,
    0x00000010, 0x0000000f, 0x00050085, 0x00000006, 0x00000012, 0x00000010, 0x00000011, 0x00070050,
    0x00000007, 0x00000015, 0x00000012, 0x00000013, 0x00000013, 0x00000014, 0x0003003e, 0x00000009,
    0x00000015, 0x000100fd, 0x00010038,
};
/* A one-subpass render pass (vkCreateRenderPass) with one color attachment
 * in GENERAL layout. */
static VkRenderPass simple_pass(struct ct *t, struct ct_env *e, VkAttachmentLoadOp load, VkAttachmentStoreOp store)
{
    VkAttachmentDescription a;
    VkAttachmentReference ref;
    VkSubpassDescription sp;
    VkRenderPassCreateInfo info;
    VkRenderPass rp = VK_NULL_HANDLE;
    memset(&a, 0, sizeof(a));
    a.format = VK_FORMAT_R8G8B8A8_UNORM;
    a.samples = VK_SAMPLE_COUNT_1_BIT;
    a.loadOp = load;
    a.storeOp = store;
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
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 1;
    info.pAttachments = &a;
    info.subpassCount = 1;
    info.pSubpasses = &sp;
    CT_REQUIRE_VK(t, vkCreateRenderPass(e->device, &info, NULL, &rp));
    return rp;
}

static VkFramebuffer framebuffer(struct ct *t, struct ct_env *e, VkRenderPass rp, const VkImageView *views, uint32_t n,
                                 uint32_t layers)
{
    VkFramebufferCreateInfo info;
    VkFramebuffer fb = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    info.renderPass = rp;
    info.attachmentCount = n;
    info.pAttachments = views;
    info.width = W;
    info.height = H;
    info.layers = layers;
    CT_REQUIRE_VK(t, vkCreateFramebuffer(e->device, &info, NULL, &fb));
    return fb;
}

static void begin_pass(VkCommandBuffer cb, VkRenderPass rp, VkFramebuffer fb, int32_t x, int32_t y, uint32_t w,
                       uint32_t h, const VkClearValue *clears, uint32_t n)
{
    VkRenderPassBeginInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass = rp;
    info.framebuffer = fb;
    info.renderArea.offset.x = x;
    info.renderArea.offset.y = y;
    info.renderArea.extent.width = w;
    info.renderArea.extent.height = h;
    info.clearValueCount = n;
    info.pClearValues = clears;
    vkCmdBeginRenderPass(cb, &info, VK_SUBPASS_CONTENTS_INLINE);
}

static VkClearValue clear_color(const float c[4])
{
    VkClearValue v;
    memset(&v, 0, sizeof(v));
    memcpy(v.color.float32, c, 4 * sizeof(float));
    return v;
}

CT_TEST(render_pass_clear_store,
        "Render Pass > Load Operations, Store Operations: VK_ATTACHMENT_LOAD_OP_CLEAR sets the render area to the "
        "clear value (VkRenderPassBeginInfo::pClearValues) at the start of the subpass, draws in the subpass then "
        "write, and VK_ATTACHMENT_STORE_OP_STORE writes the result to memory",
        RP_COMMANDS "vkCreateRenderPass vkDestroyRenderPass vkCreateFramebuffer vkDestroyFramebuffer "
                    "vkCmdBeginRenderPass vkCmdEndRenderPass")
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkRenderPass rp;
    VkFramebuffer fb;
    VkPipeline pipe;
    VkClearValue clear = clear_color(ct_blue);
    VkCommandBuffer cb;
    struct ct_box want = { 0, 0, 8, H, RED, BLUE };

    ct_scene_init(t, &s);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &c);
    rp = simple_pass(t, &s.e, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
    fb = framebuffer(t, &s.e, rp, &c.view, 1, 1);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.render_pass = rp;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    begin_pass(cb, rp, fb, 0, 0, W, H, &clear, 1);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 1.0f, ct_red, 0.0f);
    vkCmdEndRenderPass(cb);
    ct_target_read(cb, &c, 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, 0);
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want, 0, "color");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyFramebuffer(s.e.device, fb, NULL);
    vkDestroyRenderPass(s.e.device, rp, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(render_pass_load_and_render_area,
        "Render Pass > Load Operations: VK_ATTACHMENT_LOAD_OP_LOAD preserves the previous contents of the render "
        "area; \"load operations only update values within the defined render area\" and \"store operations only "
        "update values within the defined render area\": a CLEAR pass with a smaller render area leaves the pixels "
        "outside it as they were",
        RP_COMMANDS "vkCreateRenderPass vkDestroyRenderPass vkCreateFramebuffer vkDestroyFramebuffer "
                    "vkCmdBeginRenderPass vkCmdEndRenderPass vkCmdClearColorImage")
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkRenderPass load, clear_pass;
    VkFramebuffer fb;
    VkPipeline pipe;
    VkClearValue clear = clear_color(ct_blue);
    VkClearColorValue fill;
    VkImageSubresourceRange range;
    VkCommandBuffer cb;
    struct ct_box want_load = { 0, 0, W, 8, RED, GREEN }, want_area = { 4, 4, 12, 12, BLUE, GREEN };

    ct_scene_init(t, &s);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &c);
    load = simple_pass(t, &s.e, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
    clear_pass = simple_pass(t, &s.e, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
    fb = framebuffer(t, &s.e, load, &c.view, 1, 1); /* compatible with both */
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.render_pass = load;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    memset(&fill, 0, sizeof(fill));
    fill.float32[1] = fill.float32[3] = 1.0f;
    memset(&range, 0, sizeof(range));
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = range.layerCount = 1;

    /* green everywhere, then a LOAD pass draws the top half red */
    cb = ct_begin_commands(t, &s.e);
    vkCmdClearColorImage(cb, c.image.image, VK_IMAGE_LAYOUT_GENERAL, &fill, 1, &range);
    ct_image_barrier(cb, VK_NULL_HANDLE, 0, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    begin_pass(cb, load, fb, 0, 0, W, H, NULL, 0);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 0.0f, ct_red, 0.0f);
    vkCmdEndRenderPass(cb);
    ct_target_read(cb, &c, 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, 0);
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want_load, 0, "LOAD pass");

    /* green everywhere, then a CLEAR pass over (4, 4) - (12, 12) only */
    cb = ct_begin_commands(t, &s.e);
    vkCmdClearColorImage(cb, c.image.image, VK_IMAGE_LAYOUT_GENERAL, &fill, 1, &range);
    ct_image_barrier(cb, VK_NULL_HANDLE, 0, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    begin_pass(cb, clear_pass, fb, 4, 4, 8, 8, &clear, 1);
    vkCmdEndRenderPass(cb);
    ct_target_read(cb, &c, 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, 0);
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want_area, 0, "CLEAR pass with a render area");

    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyFramebuffer(s.e.device, fb, NULL);
    vkDestroyRenderPass(s.e.device, load, NULL);
    vkDestroyRenderPass(s.e.device, clear_pass, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

static uint32_t expect_swizzled(uint32_t x, uint32_t y, const void *ctx)
{
    (void)y;
    (void)ctx;
    return x < 8 ? CT_RGBA(64, 128, 255, 255) : BLACK;
}

CT_TEST(render_pass_input_attachment,
        "Render Pass > Render Pass Objects, Subpass Dependencies: a later subpass reads an attachment an earlier "
        "subpass wrote as an input attachment (subpassLoad), after a dependency from color attachment writes to "
        "input attachment reads; vkCmdNextSubpass2 advances to it (vkCreateRenderPass2)",
        RP_COMMANDS "vkCreateRenderPass2 vkDestroyRenderPass vkCreateFramebuffer vkDestroyFramebuffer "
                    "vkCmdBeginRenderPass2 vkCmdNextSubpass2 vkCmdEndRenderPass2 vkCreateDescriptorSetLayout "
                    "vkDestroyDescriptorSetLayout vkCreateDescriptorPool vkDestroyDescriptorPool "
                    "vkAllocateDescriptorSets vkUpdateDescriptorSets vkCmdBindDescriptorSets")
{
    static const float orange[4] = { 1.0f, 0.5f, 0.25f, 1.0f };
    struct ct_scene s;
    struct ct_target a, b;
    struct ct_gfx g;
    VkAttachmentDescription2 att[2];
    VkAttachmentReference2 a_color, a_input, b_color;
    VkSubpassDescription2 sp[2];
    VkSubpassDependency2 dep;
    VkRenderPassCreateInfo2 info;
    VkRenderPassBeginInfo begin;
    VkSubpassBeginInfo sbi;
    VkSubpassEndInfo sei;
    VkDescriptorSetLayoutBinding binding;
    VkDescriptorSetLayoutCreateInfo dsl;
    VkDescriptorPoolSize size;
    VkDescriptorPoolCreateInfo dpi;
    VkDescriptorSetAllocateInfo dsa;
    VkDescriptorImageInfo image;
    VkWriteDescriptorSet write;
    VkDescriptorSetLayout set_layout;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    VkPipelineLayout layout;
    VkShaderModule input_fs;
    VkRenderPass rp = VK_NULL_HANDLE;
    VkFramebuffer fb;
    VkImageView views[2];
    VkPipeline first, second;
    VkClearValue clears[2];
    VkCommandBuffer cb;
    uint32_t i;
    struct ct_box want_a = { 0, 0, 8, H, CT_RGBA(255, 128, 64, 255), BLACK };

    ct_scene_init(t, &s);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT,
              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, &a);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &b);
    memset(att, 0, sizeof(att));
    for (i = 0; i < 2; i++) {
        att[i].sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
        att[i].format = VK_FORMAT_R8G8B8A8_UNORM;
        att[i].samples = VK_SAMPLE_COUNT_1_BIT;
        att[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[i].initialLayout = VK_IMAGE_LAYOUT_GENERAL;
        att[i].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    memset(&a_color, 0, sizeof(a_color));
    a_color.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
    a_color.attachment = 0;
    a_color.layout = VK_IMAGE_LAYOUT_GENERAL;
    a_color.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    a_input = a_color;
    b_color = a_color;
    b_color.attachment = 1;
    memset(sp, 0, sizeof(sp));
    sp[0].sType = sp[1].sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2;
    sp[0].pipelineBindPoint = sp[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp[0].colorAttachmentCount = 1;
    sp[0].pColorAttachments = &a_color;
    sp[1].inputAttachmentCount = 1;
    sp[1].pInputAttachments = &a_input;
    sp[1].colorAttachmentCount = 1;
    sp[1].pColorAttachments = &b_color;
    memset(&dep, 0, sizeof(dep));
    dep.sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
    dep.srcSubpass = 0;
    dep.dstSubpass = 1;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep.dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
    dep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2;
    info.attachmentCount = 2;
    info.pAttachments = att;
    info.subpassCount = 2;
    info.pSubpasses = sp;
    info.dependencyCount = 1;
    info.pDependencies = &dep;
    CT_REQUIRE_VK(t, vkCreateRenderPass2(s.e.device, &info, NULL, &rp));
    views[0] = a.view;
    views[1] = b.view;
    fb = framebuffer(t, &s.e, rp, views, 2, 1);

    memset(&binding, 0, sizeof(binding));
    binding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    memset(&dsl, 0, sizeof(dsl));
    dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = 1;
    dsl.pBindings = &binding;
    CT_REQUIRE_VK(t, vkCreateDescriptorSetLayout(s.e.device, &dsl, NULL, &set_layout));
    size.type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
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
    memset(&image, 0, sizeof(image));
    image.imageView = a.view;
    image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    memset(&write, 0, sizeof(write));
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    write.pImageInfo = &image;
    vkUpdateDescriptorSets(s.e.device, 1, &write, 0, NULL);
    layout = ct_pipeline_layout(t, &s.e, &set_layout, 1, sizeof(struct ct_rect_pc));
    input_fs = ct_shader_module(t, &s.e, input_frag, sizeof(input_frag));

    ct_gfx_init(&g, s.vs, s.fs, layout, W, H);
    g.render_pass = rp;
    first = ct_gfx_pipeline(t, &s.e, &g);
    g.fs = input_fs;
    g.subpass = 1;
    second = ct_gfx_pipeline(t, &s.e, &g);

    clears[0] = clear_color(ct_black);
    clears[1] = clear_color(ct_green);
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = rp;
    begin.framebuffer = fb;
    begin.renderArea.extent.width = W;
    begin.renderArea.extent.height = H;
    begin.clearValueCount = 2;
    begin.pClearValues = clears;
    memset(&sbi, 0, sizeof(sbi));
    sbi.sType = VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO;
    sbi.contents = VK_SUBPASS_CONTENTS_INLINE;
    memset(&sei, 0, sizeof(sei));
    sei.sType = VK_STRUCTURE_TYPE_SUBPASS_END_INFO;
    cb = ct_begin_commands(t, &s.e);
    vkCmdBeginRenderPass2(cb, &begin, &sbi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, first);
    ct_draw_rect(cb, layout, -1.0f, -1.0f, 0.0f, 1.0f, orange, 0.0f);
    vkCmdNextSubpass2(cb, &sbi, &sei);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, second);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
    ct_draw_rect(cb, layout, -1.0f, -1.0f, 1.0f, 1.0f, NULL, 0.0f);
    vkCmdEndRenderPass2(cb, &sei);
    ct_target_read(cb, &a, 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, 0);
    ct_target_read(cb, &b, 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, W * H * 4);
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want_a, 1, "attachment written by subpass 0");
    ct_check_rgba8(t, s.out.data + W * H * 4, W, H, expect_swizzled, NULL, 1, "subpass 1 (input attachment, BGRA)");

    vkDestroyPipeline(s.e.device, first, NULL);
    vkDestroyPipeline(s.e.device, second, NULL);
    vkDestroyShaderModule(s.e.device, input_fs, NULL);
    vkDestroyPipelineLayout(s.e.device, layout, NULL);
    vkDestroyDescriptorPool(s.e.device, pool, NULL);
    vkDestroyDescriptorSetLayout(s.e.device, set_layout, NULL);
    vkDestroyFramebuffer(s.e.device, fb, NULL);
    vkDestroyRenderPass(s.e.device, rp, NULL);
    ct_target_fini(&s.e, &a);
    ct_target_fini(&s.e, &b);
    ct_scene_fini(&s);
}

static uint16_t expect_depth_quarter(uint32_t x, uint32_t y)
{
    return x < 8 && y < 8 ? 16384 : 65535; /* 0.25 inside the drawn quadrant, else the clear value 1 */
}

CT_TEST(render_pass_resolves,
        "Render Pass > Render Pass Objects: pResolveAttachments resolve a multisampled color attachment (the "
        "average of its samples) and VkSubpassDescriptionDepthStencilResolve a depth attachment "
        "(VK_RESOLVE_MODE_SAMPLE_ZERO_BIT, always in supportedDepthResolveModes) at the end of the subpass, "
        "whether the multisampled attachments are stored or not",
        RP_COMMANDS "vkCreateRenderPass2 vkDestroyRenderPass vkCreateFramebuffer vkDestroyFramebuffer "
                    "vkCmdBeginRenderPass vkCmdEndRenderPass")
{
    struct ct_scene s;
    struct ct_target ms, msd, color, depth;
    struct ct_gfx g;
    VkAttachmentDescription2 att[4];
    VkAttachmentReference2 refs[4];
    VkSubpassDescriptionDepthStencilResolve dsr;
    VkSubpassDescription2 sp;
    VkRenderPassCreateInfo2 info;
    VkRenderPass rp[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkFramebuffer fb;
    VkImageView views[4];
    VkPipeline pipe;
    VkClearValue clears[2];
    VkCommandBuffer cb;
    uint32_t i, k;
    struct ct_box want = { 0, 0, 8, 8, RED, BLACK };

    ct_scene_init(t, &s);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_4_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &ms);
    ct_target(t, &s.e, VK_FORMAT_D16_UNORM, W, H, 1, VK_SAMPLE_COUNT_4_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              &msd);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &color);
    ct_target(t, &s.e, VK_FORMAT_D16_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              &depth);
    for (k = 0; k < 2; k++) { /* the multisampled attachments: DONT_CARE, then STORE */
        memset(att, 0, sizeof(att));
        memset(refs, 0, sizeof(refs));
        for (i = 0; i < 4; i++) {
            att[i].sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
            att[i].format = i & 1 ? VK_FORMAT_D16_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
            att[i].samples = i < 2 ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT;
            att[i].loadOp = i < 2 ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            att[i].storeOp = i < 2 && !k ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
            att[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            att[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            att[i].initialLayout = VK_IMAGE_LAYOUT_GENERAL;
            att[i].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
            refs[i].sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
            refs[i].attachment = i;
            refs[i].layout = VK_IMAGE_LAYOUT_GENERAL;
            refs[i].aspectMask = i & 1 ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        }
        memset(&dsr, 0, sizeof(dsr));
        dsr.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE;
        dsr.depthResolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
        dsr.stencilResolveMode = VK_RESOLVE_MODE_NONE;
        dsr.pDepthStencilResolveAttachment = &refs[3];
        memset(&sp, 0, sizeof(sp));
        sp.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2;
        sp.pNext = &dsr;
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments = &refs[0];
        sp.pResolveAttachments = &refs[2];
        sp.pDepthStencilAttachment = &refs[1];
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2;
        info.attachmentCount = 4;
        info.pAttachments = att;
        info.subpassCount = 1;
        info.pSubpasses = &sp;
        CT_REQUIRE_VK(t, vkCreateRenderPass2(s.e.device, &info, NULL, &rp[k]));
    }
    views[0] = ms.view;
    views[1] = msd.view;
    views[2] = color.view;
    views[3] = depth.view;
    fb = framebuffer(t, &s.e, rp[0], views, 4, 1);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.render_pass = rp[0];
    g.ms.rasterizationSamples = VK_SAMPLE_COUNT_4_BIT;
    g.ds.depthTestEnable = g.ds.depthWriteEnable = VK_TRUE;
    g.ds.depthCompareOp = VK_COMPARE_OP_LESS;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    clears[0] = clear_color(ct_black);
    memset(&clears[1], 0, sizeof(clears[1]));
    clears[1].depthStencil.depth = 1.0f;
    for (k = 0; k < 2; k++) {
        cb = ct_begin_commands(t, &s.e);
        begin_pass(cb, rp[k], fb, 0, 0, W, H, clears, 2);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 0.0f, ct_red, 0.25f);
        vkCmdEndRenderPass(cb);
        ct_target_read(cb, &color, 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, 0);
        ct_target_read(cb, &depth, 0, VK_IMAGE_ASPECT_DEPTH_BIT, s.out.buffer, W * H * 4);
        ct_submit_and_wait(t, &s.e, cb);
        ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want, 0, k ? "resolved color (stored)" : "resolved color");
        ct_check_d16(t, s.out.data + W * H * 4, W, H, expect_depth_quarter,
                     k ? "resolved depth (stored)" : "resolved depth");
    }
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyFramebuffer(s.e.device, fb, NULL);
    vkDestroyRenderPass(s.e.device, rp[0], NULL);
    vkDestroyRenderPass(s.e.device, rp[1], NULL);
    ct_target_fini(&s.e, &ms);
    ct_target_fini(&s.e, &msd);
    ct_target_fini(&s.e, &color);
    ct_target_fini(&s.e, &depth);
    ct_scene_fini(&s);
}

static uint32_t expect_view(uint32_t x, uint32_t y, const void *ctx)
{
    (void)x;
    (void)y;
    return *(const uint32_t *)ctx;
}

CT_TEST(render_pass_multiview,
        "Render Pass > Render Pass Objects (VkRenderPassMultiviewCreateInfo): with a view mask, the subpass's "
        "drawing is broadcast to each view in it, view n rendering to layer n of the attachments, with ViewIndex "
        "the view's index (multiview, required from Vulkan 1.1)",
        RP_COMMANDS "vkCreateRenderPass vkDestroyRenderPass vkCreateFramebuffer vkDestroyFramebuffer "
                    "vkCmdBeginRenderPass vkCmdEndRenderPass")
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkAttachmentDescription a;
    VkAttachmentReference ref;
    VkSubpassDescription sp;
    VkRenderPassMultiviewCreateInfo mv;
    VkRenderPassCreateInfo info;
    VkRenderPass rp = VK_NULL_HANDLE;
    VkFramebuffer fb;
    VkShaderModule view_fs;
    VkPipeline pipe;
    VkClearValue clear = clear_color(ct_black);
    VkCommandBuffer cb;
    const uint32_t mask = 0x5; /* views 0 and 2 */
    uint32_t want0 = CT_RGBA(64, 0, 0, 255), want2 = CT_RGBA(191, 0, 0, 255);

    ct_scene_init(t, &s);
    CT_REQUIRE(t, s.f.v11.multiview); /* required from Vulkan 1.1 */
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 3, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &c);
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
    memset(&mv, 0, sizeof(mv));
    mv.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO;
    mv.subpassCount = 1;
    mv.pViewMasks = &mask;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.pNext = &mv;
    info.attachmentCount = 1;
    info.pAttachments = &a;
    info.subpassCount = 1;
    info.pSubpasses = &sp;
    CT_REQUIRE_VK(t, vkCreateRenderPass(s.e.device, &info, NULL, &rp));
    fb = framebuffer(t, &s.e, rp, &c.view, 1, 1); /* layers 1 with multiview */
    view_fs = ct_shader_module(t, &s.e, view_frag, sizeof(view_frag));
    ct_gfx_init(&g, s.vs, view_fs, s.layout, W, H);
    g.render_pass = rp;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    begin_pass(cb, rp, fb, 0, 0, W, H, &clear, 1);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, NULL, 0.0f);
    vkCmdEndRenderPass(cb);
    ct_target_read(cb, &c, 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, 0);
    ct_target_read(cb, &c, 2, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, W * H * 4);
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_rgba8(t, s.out.data, W, H, expect_view, &want0, 1, "layer 0 (view 0)");
    ct_check_rgba8(t, s.out.data + W * H * 4, W, H, expect_view, &want2, 1, "layer 2 (view 2)");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, view_fs, NULL);
    vkDestroyFramebuffer(s.e.device, fb, NULL);
    vkDestroyRenderPass(s.e.device, rp, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(imageless_framebuffer,
        "Render Pass > Framebuffers: a framebuffer created with VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT describes its "
        "attachments (VkFramebufferAttachmentsCreateInfo); the image views come with "
        "VkRenderPassAttachmentBeginInfo when the render pass begins (imagelessFramebuffer, required from Vulkan "
        "1.2)",
        RP_COMMANDS "vkCreateRenderPass vkDestroyRenderPass vkCreateFramebuffer vkDestroyFramebuffer "
                    "vkCmdBeginRenderPass vkCmdEndRenderPass")
{
    struct ct_scene s;
    struct ct_target c[2];
    struct ct_gfx g;
    VkRenderPass rp;
    VkFramebufferAttachmentImageInfo image;
    VkFramebufferAttachmentsCreateInfo atts;
    VkFramebufferCreateInfo info;
    VkRenderPassAttachmentBeginInfo views;
    VkRenderPassBeginInfo begin;
    VkFramebuffer fb = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipeline pipe;
    VkClearValue clear = clear_color(ct_blue);
    VkCommandBuffer cb;
    uint32_t k;
    struct ct_box want[2] = { { 0, 0, 8, H, RED, BLUE }, { 8, 0, W, H, GREEN, BLUE } };

    ct_scene_init(t, &s);
    CT_REQUIRE(t, s.f.v12.imagelessFramebuffer); /* required from Vulkan 1.2 */
    for (k = 0; k < 2; k++)
        ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &c[k]);
    rp = simple_pass(t, &s.e, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
    memset(&image, 0, sizeof(image));
    image.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO;
    image.usage = c[0].image.info.usage;
    image.width = W;
    image.height = H;
    image.layerCount = 1;
    image.viewFormatCount = 1;
    image.pViewFormats = &format;
    memset(&atts, 0, sizeof(atts));
    atts.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO;
    atts.attachmentImageInfoCount = 1;
    atts.pAttachmentImageInfos = &image;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    info.pNext = &atts;
    info.flags = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT;
    info.renderPass = rp;
    info.attachmentCount = 1;
    info.width = W;
    info.height = H;
    info.layers = 1;
    CT_REQUIRE_VK(t, vkCreateFramebuffer(s.e.device, &info, NULL, &fb));
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.render_pass = rp;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    for (k = 0; k < 2; k++) { /* one framebuffer, two images */
        memset(&views, 0, sizeof(views));
        views.sType = VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO;
        views.attachmentCount = 1;
        views.pAttachments = &c[k].view;
        memset(&begin, 0, sizeof(begin));
        begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.pNext = &views;
        begin.renderPass = rp;
        begin.framebuffer = fb;
        begin.renderArea.extent.width = W;
        begin.renderArea.extent.height = H;
        begin.clearValueCount = 1;
        begin.pClearValues = &clear;
        vkCmdBeginRenderPass(cb, &begin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        if (k)
            ct_draw_rect(cb, s.layout, 0.0f, -1.0f, 1.0f, 1.0f, ct_green, 0.0f);
        else
            ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 1.0f, ct_red, 0.0f);
        vkCmdEndRenderPass(cb);
    }
    ct_target_read(cb, &c[0], 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, 0);
    ct_target_read(cb, &c[1], 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, W * H * 4);
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want[0], 0, "first image");
    ct_check_rgba8(t, s.out.data + W * H * 4, W, H, ct_expect_box, &want[1], 0, "second image");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyFramebuffer(s.e.device, fb, NULL);
    vkDestroyRenderPass(s.e.device, rp, NULL);
    ct_target_fini(&s.e, &c[0]);
    ct_target_fini(&s.e, &c[1]);
    ct_scene_fini(&s);
}

CT_TEST(render_area_granularity,
        "Render Pass > Render Pass Objects (vkGetRenderAreaGranularity) and Dynamic Rendering "
        "(vkGetRenderingAreaGranularity, Vulkan 1.4): the granularity is a non-zero extent in pixels",
        "vkCreateRenderPass vkDestroyRenderPass vkGetRenderAreaGranularity vkGetRenderingAreaGranularity")
{
    struct ct_env e;
    struct ct_features f;
    VkRenderPass rp;
    VkExtent2D g;
    ct_env_render(t, &e, &f);
    rp = simple_pass(t, &e, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
    g.width = g.height = 0;
    vkGetRenderAreaGranularity(e.device, rp, &g);
    CT_CHECKF(t, g.width >= 1 && g.height >= 1, "vkGetRenderAreaGranularity: %u x %u", g.width, g.height);
    vkDestroyRenderPass(e.device, rp, NULL);
    if (e.version >= VK_API_VERSION_1_4) {
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        VkRenderingAreaInfo info;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_RENDERING_AREA_INFO;
        info.colorAttachmentCount = 1;
        info.pColorAttachmentFormats = &format;
        g.width = g.height = 0;
        vkGetRenderingAreaGranularity(e.device, &info, &g);
        CT_CHECKF(t, g.width >= 1 && g.height >= 1, "vkGetRenderingAreaGranularity: %u x %u", g.width, g.height);
    }
    ct_env_fini(&e);
}

static uint16_t expect_depth_left_half(uint32_t x, uint32_t y)
{
    (void)y;
    return x < 8 ? 16384 : 32768; /* 0.25 drawn, 0.5 cleared */
}

CT_TEST(dynamic_rendering_color_depth,
        "Render Pass > Dynamic Render Pass Instances: vkCmdBeginRendering clears color and depth attachments "
        "(VkRenderingAttachmentInfo loadOp and clearValue), draws in the instance pass the depth test, and "
        "VK_ATTACHMENT_STORE_OP_STORE stores both (dynamicRendering, required from Vulkan 1.3)",
        RP_COMMANDS "vkCmdBeginRendering vkCmdEndRendering")
{
    struct ct_scene s;
    struct ct_target c, d;
    struct ct_gfx g;
    VkPipeline pipe;
    VkCommandBuffer cb;
    struct ct_box want = { 0, 0, 8, H, RED, BLUE };

    ct_scene_init(t, &s);
    CT_REQUIRE(t, s.f.v13.dynamicRendering);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &c);
    ct_target(t, &s.e, VK_FORMAT_D16_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              &d);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.rendering.depthAttachmentFormat = VK_FORMAT_D16_UNORM;
    g.ds.depthTestEnable = g.ds.depthWriteEnable = VK_TRUE;
    g.ds.depthCompareOp = VK_COMPARE_OP_LESS;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    ct_begin_rendering(cb, &c, 1, ct_blue, &d, 0.5f);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 1.0f, ct_red, 0.25f);   /* passes: 0.25 < 0.5 */
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, ct_green, 0.75f); /* fails everywhere */
    vkCmdEndRendering(cb);
    ct_target_read(cb, &c, 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, 0);
    ct_target_read(cb, &d, 0, VK_IMAGE_ASPECT_DEPTH_BIT, s.out.buffer, W * H * 4);
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want, 0, "color");
    ct_check_d16(t, s.out.data + W * H * 4, W, H, expect_depth_left_half, "depth");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_target_fini(&s.e, &c);
    ct_target_fini(&s.e, &d);
    ct_scene_fini(&s);
}

CT_TEST(dynamic_rendering_resolves,
        "Render Pass > Dynamic Render Pass Instances: VkRenderingAttachmentInfo resolveMode resolves a "
        "multisampled color attachment (VK_RESOLVE_MODE_AVERAGE_BIT) and depth attachment "
        "(VK_RESOLVE_MODE_SAMPLE_ZERO_BIT) into resolveImageView at the end of the instance, whether the "
        "multisampled attachments are stored or not",
        RP_COMMANDS "vkCmdBeginRendering vkCmdEndRendering")
{
    struct ct_scene s;
    struct ct_target ms, msd, color, depth;
    struct ct_gfx g;
    VkRenderingAttachmentInfo ca, da;
    VkRenderingInfo info;
    VkPipeline pipe;
    VkCommandBuffer cb;
    uint32_t k;
    struct ct_box want = { 0, 0, 8, 8, RED, BLACK };

    ct_scene_init(t, &s);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_4_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &ms);
    ct_target(t, &s.e, VK_FORMAT_D16_UNORM, W, H, 1, VK_SAMPLE_COUNT_4_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              &msd);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &color);
    ct_target(t, &s.e, VK_FORMAT_D16_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              &depth);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.rendering.depthAttachmentFormat = VK_FORMAT_D16_UNORM;
    g.ms.rasterizationSamples = VK_SAMPLE_COUNT_4_BIT;
    g.ds.depthTestEnable = g.ds.depthWriteEnable = VK_TRUE;
    g.ds.depthCompareOp = VK_COMPARE_OP_LESS;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    for (k = 0; k < 2; k++) {
        memset(&ca, 0, sizeof(ca));
        ca.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        ca.imageView = ms.view;
        ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
        ca.resolveImageView = color.view;
        ca.resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ca.storeOp = k ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        memcpy(ca.clearValue.color.float32, ct_black, sizeof(ct_black));
        da = ca;
        da.imageView = msd.view;
        da.resolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
        da.resolveImageView = depth.view;
        memset(&da.clearValue, 0, sizeof(da.clearValue));
        da.clearValue.depthStencil.depth = 1.0f;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        info.renderArea.extent.width = W;
        info.renderArea.extent.height = H;
        info.layerCount = 1;
        info.colorAttachmentCount = 1;
        info.pColorAttachments = &ca;
        info.pDepthAttachment = &da;
        cb = ct_begin_commands(t, &s.e);
        vkCmdBeginRendering(cb, &info);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 0.0f, ct_red, 0.25f);
        vkCmdEndRendering(cb);
        ct_target_read(cb, &color, 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, 0);
        ct_target_read(cb, &depth, 0, VK_IMAGE_ASPECT_DEPTH_BIT, s.out.buffer, W * H * 4);
        ct_submit_and_wait(t, &s.e, cb);
        ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want, 0, k ? "resolved color (stored)" : "resolved color");
        ct_check_d16(t, s.out.data + W * H * 4, W, H, expect_depth_quarter,
                     k ? "resolved depth (stored)" : "resolved depth");
    }
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_target_fini(&s.e, &ms);
    ct_target_fini(&s.e, &msd);
    ct_target_fini(&s.e, &color);
    ct_target_fini(&s.e, &depth);
    ct_scene_fini(&s);
}

CT_TEST(dynamic_rendering_suspend_resume,
        "Render Pass > Dynamic Render Pass Instances: an instance ended with VK_RENDERING_SUSPENDING_BIT is "
        "resumed by the next vkCmdBeginRendering with VK_RENDERING_RESUMING_BIT, here in the next command buffer "
        "of the batch, as one render pass instance: the clear happens once and the draws of both command buffers "
        "land",
        RP_COMMANDS "vkCmdBeginRendering vkCmdEndRendering vkAllocateCommandBuffers vkBeginCommandBuffer "
                    "vkEndCommandBuffer vkFreeCommandBuffers vkQueueSubmit vkCreateFence vkDestroyFence "
                    "vkWaitForFences")
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkRenderingAttachmentInfo att;
    VkRenderingInfo info;
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferBeginInfo bi;
    VkCommandBuffer cbs[2];
    VkSubmitInfo submit;
    VkFenceCreateInfo fi;
    VkFence fence = VK_NULL_HANDLE;
    VkPipeline pipe;
    VkCommandBuffer cb;
    uint32_t k;
    struct ct_box want = { 0, 0, 8, H, RED, GREEN };

    ct_scene_init(t, &s);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &c);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e); /* makes the pool */
    ct_submit_and_wait(t, &s.e, cb);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = s.e.pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 2;
    CT_REQUIRE_VK(t, vkAllocateCommandBuffers(s.e.device, &alloc, cbs));
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
    info.layerCount = 1;
    info.colorAttachmentCount = 1;
    info.pColorAttachments = &att;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    for (k = 0; k < 2; k++) {
        CT_REQUIRE_VK(t, vkBeginCommandBuffer(cbs[k], &bi));
        info.flags = k ? VK_RENDERING_RESUMING_BIT : VK_RENDERING_SUSPENDING_BIT;
        vkCmdBeginRendering(cbs[k], &info);
        vkCmdBindPipeline(cbs[k], VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        if (k)
            ct_draw_rect(cbs[k], s.layout, 0.0f, -1.0f, 1.0f, 1.0f, ct_green, 0.0f);
        else
            ct_draw_rect(cbs[k], s.layout, -1.0f, -1.0f, 0.0f, 1.0f, ct_red, 0.0f);
        vkCmdEndRendering(cbs[k]);
        if (k)
            ct_target_read(cbs[k], &c, 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, 0);
        CT_REQUIRE_VK(t, vkEndCommandBuffer(cbs[k]));
    }
    memset(&fi, 0, sizeof(fi));
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreateFence(s.e.device, &fi, NULL, &fence));
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 2;
    submit.pCommandBuffers = cbs;
    CT_REQUIRE_VK(t, vkQueueSubmit(s.e.queue, 1, &submit, fence));
    CT_REQUIRE_VK(t, vkWaitForFences(s.e.device, 1, &fence, VK_TRUE, 10000000000ull));
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want, 0, "color");
    vkFreeCommandBuffers(s.e.device, s.e.pool, 2, cbs);
    vkDestroyFence(s.e.device, fence, NULL);
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(dynamic_rendering_multiview,
        "Render Pass > Dynamic Render Pass Instances: VkRenderingInfo::viewMask broadcasts the instance's drawing "
        "to each view in the mask, view n rendering to layer n of the attachments (multiview)",
        RP_COMMANDS "vkCmdBeginRendering vkCmdEndRendering")
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    VkRenderingAttachmentInfo att;
    VkRenderingInfo info;
    VkShaderModule view_fs;
    VkPipeline pipe;
    VkCommandBuffer cb;
    uint32_t want0 = CT_RGBA(64, 0, 0, 255), want1 = CT_RGBA(128, 0, 0, 255);

    ct_scene_init(t, &s);
    CT_REQUIRE(t, s.f.v11.multiview);
    ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 2, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              &c);
    view_fs = ct_shader_module(t, &s.e, view_frag, sizeof(view_frag));
    ct_gfx_init(&g, s.vs, view_fs, s.layout, W, H);
    g.rendering.viewMask = 0x3;
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
    info.viewMask = 0x3;
    info.colorAttachmentCount = 1;
    info.pColorAttachments = &att;
    cb = ct_begin_commands(t, &s.e);
    vkCmdBeginRendering(cb, &info);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, NULL, 0.0f);
    vkCmdEndRendering(cb);
    ct_target_read(cb, &c, 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, 0);
    ct_target_read(cb, &c, 1, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, W * H * 4);
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_rgba8(t, s.out.data, W, H, expect_view, &want0, 1, "layer 0 (view 0)");
    ct_check_rgba8(t, s.out.data + W * H * 4, W, H, expect_view, &want1, 1, "layer 1 (view 1)");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    vkDestroyShaderModule(s.e.device, view_fs, NULL);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

/* Two color attachments cleared to black, for the local read tests. */
static void begin_two(VkCommandBuffer cb, const struct ct_target *c)
{
    VkRenderingAttachmentInfo att[2];
    VkRenderingInfo info;
    uint32_t i;
    memset(att, 0, sizeof(att));
    for (i = 0; i < 2; i++) {
        att[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        att[i].imageView = c[i].view;
        att[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        att[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        memcpy(att[i].clearValue.color.float32, ct_black, sizeof(ct_black));
    }
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    info.renderArea.extent.width = W;
    info.renderArea.extent.height = H;
    info.layerCount = 1;
    info.colorAttachmentCount = 2;
    info.pColorAttachments = att;
    vkCmdBeginRendering(cb, &info);
}

CT_TEST(local_read_attachment_locations,
        "Render Pass > Dynamic Rendering Local Read: vkCmdSetRenderingAttachmentLocations maps color attachments "
        "to fragment output locations (matching the bound pipeline's VkRenderingAttachmentLocationInfo); an "
        "attachment mapped to VK_ATTACHMENT_UNUSED is not written (dynamicRenderingLocalRead, required from "
        "Vulkan 1.4)",
        RP_COMMANDS "vkCmdBeginRendering vkCmdEndRendering vkCmdSetRenderingAttachmentLocations")
{
    static const uint32_t locations[2] = { VK_ATTACHMENT_UNUSED, 0 };
    struct ct_scene s;
    struct ct_target c[2];
    struct ct_gfx g;
    VkRenderingAttachmentLocationInfo loc;
    VkPipeline pipe;
    VkCommandBuffer cb;
    struct ct_box want0 = { 0, 0, 0, 0, BLACK, BLACK }, want1 = { 0, 0, 8, H, RED, BLACK };
    uint32_t k;

    ct_scene_init(t, &s);
    if (s.e.version < VK_API_VERSION_1_4)
        ct_skip(t, "dynamic rendering local read is core in Vulkan 1.4");
    CT_REQUIRE(t, s.f.v14.dynamicRenderingLocalRead);
    for (k = 0; k < 2; k++)
        ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &c[k]);
    memset(&loc, 0, sizeof(loc));
    loc.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO;
    loc.colorAttachmentCount = 2;
    loc.pColorAttachmentLocations = locations;
    ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
    g.colors = 2;
    g.next = &loc;
    pipe = ct_gfx_pipeline(t, &s.e, &g);
    cb = ct_begin_commands(t, &s.e);
    begin_two(cb, c);
    vkCmdSetRenderingAttachmentLocations(cb, &loc);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 1.0f, ct_red, 0.0f);
    vkCmdEndRendering(cb);
    ct_target_read(cb, &c[0], 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, 0);
    ct_target_read(cb, &c[1], 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, W * H * 4);
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want0, 0, "attachment 0 (VK_ATTACHMENT_UNUSED)");
    ct_check_rgba8(t, s.out.data + W * H * 4, W, H, ct_expect_box, &want1, 0, "attachment 1 (location 0)");
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_target_fini(&s.e, &c[0]);
    ct_target_fini(&s.e, &c[1]);
    ct_scene_fini(&s);
}

CT_TEST(local_read_input_attachment,
        "Render Pass > Dynamic Rendering Local Read: within one dynamic render pass instance, after a "
        "VK_DEPENDENCY_BY_REGION_BIT barrier from color attachment writes to input attachment reads, a fragment "
        "shader reads what earlier draws wrote to a color attachment through the input attachment index that "
        "vkCmdSetRenderingInputAttachmentIndices gives it (dynamicRenderingLocalRead, required from Vulkan 1.4)",
        RP_COMMANDS "vkCmdBeginRendering vkCmdEndRendering vkCmdSetRenderingAttachmentLocations "
                    "vkCmdSetRenderingInputAttachmentIndices vkCmdPipelineBarrier2 vkCreateDescriptorSetLayout "
                    "vkDestroyDescriptorSetLayout vkCreateDescriptorPool vkDestroyDescriptorPool "
                    "vkAllocateDescriptorSets vkUpdateDescriptorSets vkCmdBindDescriptorSets")
{
    static const float orange[4] = { 1.0f, 0.5f, 0.25f, 1.0f };
    static const uint32_t write_first[2] = { 0, VK_ATTACHMENT_UNUSED }, write_second[2] = { VK_ATTACHMENT_UNUSED, 0 };
    static const uint32_t inputs[2] = { 0, VK_ATTACHMENT_UNUSED };
    struct ct_scene s;
    struct ct_target c[2];
    struct ct_gfx g;
    VkRenderingAttachmentLocationInfo loc1, loc2;
    VkRenderingInputAttachmentIndexInfo idx;
    VkDescriptorSetLayoutBinding binding;
    VkDescriptorSetLayoutCreateInfo dsl;
    VkDescriptorPoolSize size;
    VkDescriptorPoolCreateInfo dpi;
    VkDescriptorSetAllocateInfo dsa;
    VkDescriptorImageInfo image;
    VkWriteDescriptorSet write;
    VkDescriptorSetLayout set_layout;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    VkPipelineLayout layout;
    VkShaderModule input_fs;
    VkMemoryBarrier2 mb;
    VkDependencyInfo dep;
    VkPipeline first, second;
    VkCommandBuffer cb;
    uint32_t k;
    struct ct_box want0 = { 0, 0, 8, H, CT_RGBA(255, 128, 64, 255), BLACK };

    ct_scene_init(t, &s);
    if (s.e.version < VK_API_VERSION_1_4)
        ct_skip(t, "dynamic rendering local read is core in Vulkan 1.4");
    CT_REQUIRE(t, s.f.v14.dynamicRenderingLocalRead);
    for (k = 0; k < 2; k++)
        ct_target(t, &s.e, VK_FORMAT_R8G8B8A8_UNORM, W, H, 1, VK_SAMPLE_COUNT_1_BIT,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, &c[k]);
    memset(&binding, 0, sizeof(binding));
    binding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    memset(&dsl, 0, sizeof(dsl));
    dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = 1;
    dsl.pBindings = &binding;
    CT_REQUIRE_VK(t, vkCreateDescriptorSetLayout(s.e.device, &dsl, NULL, &set_layout));
    size.type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
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
    memset(&image, 0, sizeof(image));
    image.imageView = c[0].view;
    image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    memset(&write, 0, sizeof(write));
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    write.pImageInfo = &image;
    vkUpdateDescriptorSets(s.e.device, 1, &write, 0, NULL);
    layout = ct_pipeline_layout(t, &s.e, &set_layout, 1, sizeof(struct ct_rect_pc));
    input_fs = ct_shader_module(t, &s.e, input_frag, sizeof(input_frag));

    memset(&loc1, 0, sizeof(loc1));
    loc1.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO;
    loc1.colorAttachmentCount = 2;
    loc1.pColorAttachmentLocations = write_first;
    loc2 = loc1;
    loc2.pColorAttachmentLocations = write_second;
    memset(&idx, 0, sizeof(idx));
    idx.sType = VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO;
    idx.colorAttachmentCount = 2;
    idx.pColorAttachmentInputIndices = inputs;
    ct_gfx_init(&g, s.vs, s.fs, layout, W, H);
    g.colors = 2;
    g.next = &loc1;
    first = ct_gfx_pipeline(t, &s.e, &g);
    loc2.pNext = &idx;
    g.fs = input_fs;
    g.next = &loc2;
    second = ct_gfx_pipeline(t, &s.e, &g);

    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    mb.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT;
    memset(&dep, 0, sizeof(dep));
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    cb = ct_begin_commands(t, &s.e);
    begin_two(cb, c);
    vkCmdSetRenderingAttachmentLocations(cb, &loc1);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, first);
    ct_draw_rect(cb, layout, -1.0f, -1.0f, 0.0f, 1.0f, orange, 0.0f);
    vkCmdPipelineBarrier2(cb, &dep);
    vkCmdSetRenderingAttachmentLocations(cb, &loc2);
    vkCmdSetRenderingInputAttachmentIndices(cb, &idx);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, second);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
    ct_draw_rect(cb, layout, -1.0f, -1.0f, 1.0f, 1.0f, NULL, 0.0f);
    vkCmdEndRendering(cb);
    ct_target_read(cb, &c[0], 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, 0);
    ct_target_read(cb, &c[1], 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, W * H * 4);
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want0, 1, "attachment 0 (written, then read)");
    ct_check_rgba8(t, s.out.data + W * H * 4, W, H, expect_swizzled, NULL, 1, "attachment 1 (input attachment, BGRA)");
    vkDestroyPipeline(s.e.device, first, NULL);
    vkDestroyPipeline(s.e.device, second, NULL);
    vkDestroyShaderModule(s.e.device, input_fs, NULL);
    vkDestroyPipelineLayout(s.e.device, layout, NULL);
    vkDestroyDescriptorPool(s.e.device, pool, NULL);
    vkDestroyDescriptorSetLayout(s.e.device, set_layout, NULL);
    ct_target_fini(&s.e, &c[0]);
    ct_target_fini(&s.e, &c[1]);
    ct_scene_fini(&s);
}

CT_TEST(render_pass_next_subpass,
        "Render Pass > Render Pass Commands (vkCmdNextSubpass): the subpass index advances by one and later "
        "commands are recorded for that subpass, inline (VK_SUBPASS_CONTENTS_INLINE) or in secondary command "
        "buffers executed in it (VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS, inheriting the subpass); "
        "attachments a subpass preserves (pPreserveAttachments) keep what an earlier subpass wrote "
        "(vkCreateRenderPass)",
        RP_COMMANDS "vkCreateRenderPass vkDestroyRenderPass vkCreateFramebuffer vkDestroyFramebuffer "
                    "vkCmdBeginRenderPass vkCmdNextSubpass vkCmdEndRenderPass vkAllocateCommandBuffers "
                    "vkBeginCommandBuffer vkEndCommandBuffer vkCmdExecuteCommands vkFreeCommandBuffers")
{
    struct ct_scene s;
    struct ct_target c[3];
    struct ct_gfx g;
    VkAttachmentDescription att[3];
    VkAttachmentReference refs[3];
    uint32_t preserve[2] = { 0, 1 };
    VkSubpassDescription sp[3];
    VkRenderPassCreateInfo info;
    VkRenderPass rp = VK_NULL_HANDLE;
    VkFramebuffer fb;
    VkImageView views[3];
    VkPipeline pipes[3];
    VkClearValue clears[3];
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferInheritanceInfo inherit;
    VkCommandBufferBeginInfo begin;
    VkCommandBuffer cb, second;
    uint32_t i;
    struct ct_box want[3] = {
        { 0, 0, W / 2, H, RED, BLACK },   /* subpass 0: left half */
        { 0, 0, W, H / 2, GREEN, BLACK }, /* subpass 1: top half */
        { 0, H / 2, W, H, BLUE, BLACK },  /* subpass 2 (secondary): bottom half */
    };

    ct_scene_init(t, &s);
    memset(att, 0, sizeof(att));
    memset(sp, 0, sizeof(sp));
    for (i = 0; i < 3; i++) {
        ct_scene_color(t, &s, &c[i]);
        views[i] = c[i].view;
        att[i].format = VK_FORMAT_R8G8B8A8_UNORM;
        att[i].samples = VK_SAMPLE_COUNT_1_BIT;
        att[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[i].initialLayout = VK_IMAGE_LAYOUT_GENERAL;
        att[i].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        refs[i].attachment = i;
        refs[i].layout = VK_IMAGE_LAYOUT_GENERAL;
        sp[i].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp[i].colorAttachmentCount = 1;
        sp[i].pColorAttachments = &refs[i];
        sp[i].preserveAttachmentCount = i; /* the earlier subpasses' attachments */
        sp[i].pPreserveAttachments = i ? preserve : NULL;
        clears[i] = clear_color(ct_black);
    }
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 3;
    info.pAttachments = att;
    info.subpassCount = 3;
    info.pSubpasses = sp;
    CT_REQUIRE_VK(t, vkCreateRenderPass(s.e.device, &info, NULL, &rp));
    fb = framebuffer(t, &s.e, rp, views, 3, 1);
    for (i = 0; i < 3; i++) {
        ct_gfx_init(&g, s.vs, s.fs, s.layout, W, H);
        g.render_pass = rp;
        g.subpass = i;
        pipes[i] = ct_gfx_pipeline(t, &s.e, &g);
    }

    cb = ct_begin_commands(t, &s.e);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = s.e.pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    alloc.commandBufferCount = 1;
    CT_REQUIRE_VK(t, vkAllocateCommandBuffers(s.e.device, &alloc, &second));
    memset(&inherit, 0, sizeof(inherit));
    inherit.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    inherit.renderPass = rp;
    inherit.subpass = 2;
    inherit.framebuffer = fb;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
    begin.pInheritanceInfo = &inherit;
    CT_REQUIRE_VK(t, vkBeginCommandBuffer(second, &begin));
    vkCmdBindPipeline(second, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[2]);
    ct_draw_rect(second, s.layout, -1.0f, 0.0f, 1.0f, 1.0f, ct_blue, 0.0f);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(second));

    begin_pass(cb, rp, fb, 0, 0, W, H, clears, 3);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[0]);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 0.0f, 1.0f, ct_red, 0.0f);
    vkCmdNextSubpass(cb, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[1]);
    ct_draw_rect(cb, s.layout, -1.0f, -1.0f, 1.0f, 0.0f, ct_green, 0.0f);
    vkCmdNextSubpass(cb, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
    vkCmdExecuteCommands(cb, 1, &second);
    vkCmdEndRenderPass(cb);
    for (i = 0; i < 3; i++)
        ct_target_read(cb, &c[i], 0, VK_IMAGE_ASPECT_COLOR_BIT, s.out.buffer, (VkDeviceSize)i * W * H * 4);
    ct_submit_and_wait(t, &s.e, cb);
    ct_check_rgba8(t, s.out.data, W, H, ct_expect_box, &want[0], 0, "attachment of subpass 0 (preserved by 1 and 2)");
    ct_check_rgba8(t, s.out.data + W * H * 4, W, H, ct_expect_box, &want[1], 0,
                   "attachment of subpass 1 (preserved by 2)");
    ct_check_rgba8(t, s.out.data + 2 * W * H * 4, W, H, ct_expect_box, &want[2], 0,
                   "attachment of subpass 2 (secondary command buffer)");

    vkFreeCommandBuffers(s.e.device, s.e.pool, 1, &second);
    for (i = 0; i < 3; i++) {
        vkDestroyPipeline(s.e.device, pipes[i], NULL);
        ct_target_fini(&s.e, &c[i]);
    }
    vkDestroyFramebuffer(s.e.device, fb, NULL);
    vkDestroyRenderPass(s.e.device, rp, NULL);
    ct_scene_fini(&s);
}

CT_TEST(rendering_area_granularity_khr,
        "Render Pass > Dynamic Render Pass Instances (vkGetRenderingAreaGranularityKHR, VK_KHR_maintenance5 on an "
        "apiVersion 1.3 device, the maintenance5 feature enabled): the render area granularity for the "
        "attachment formats of VkRenderingAreaInfo (color, color and depth/stencil, none) is a non-zero extent in "
        "pixels",
        "vkGetRenderingAreaGranularityKHR vkGetDeviceProcAddr vkCreateDevice vkDestroyDevice")
{
    const char *exts[] = { "VK_KHR_maintenance5" };
    VkPhysicalDeviceMaintenance5Features m5;
    PFN_vkGetRenderingAreaGranularityKHR get;
    VkFormat color[2] = { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM };
    VkRenderingAreaInfo info;
    VkExtent2D gran;
    struct ct_env e;
    uint32_t i;

    if (CT_VERSION(ct_instance_version()) < VK_API_VERSION_1_3)
        ct_skip(t, "instance version below 1.3");
    ct_env_init_ex(t, &e, VK_API_VERSION_1_3, NULL, 0);
    if (e.version < VK_API_VERSION_1_3)
        ct_skip(t, "device version below 1.3");
    if (!ct_device_extension_supported(e.pd, "VK_KHR_maintenance5"))
        ct_skip(t, "VK_KHR_maintenance5 is not offered");
    memset(&m5, 0, sizeof(m5));
    m5.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES;
    m5.maintenance5 = VK_TRUE;
    ct_env_device_ex(t, &e, &m5, exts, 1, NULL);
    get = (PFN_vkGetRenderingAreaGranularityKHR)vkGetDeviceProcAddr(e.device, "vkGetRenderingAreaGranularityKHR");
    CT_REQUIREF(t, get != NULL, "vkGetRenderingAreaGranularityKHR: NULL with VK_KHR_maintenance5 enabled");
    for (i = 0; i < 3; i++) {
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_RENDERING_AREA_INFO;
        info.colorAttachmentCount = i == 2 ? 0 : i + 1;
        info.pColorAttachmentFormats = i == 2 ? NULL : color;
        if (i == 1) {
            info.depthAttachmentFormat = ct_depth_stencil_format(&e);
            info.stencilAttachmentFormat = info.depthAttachmentFormat;
        }
        gran.width = gran.height = 0;
        get(e.device, &info, &gran);
        CT_CHECKF(t, gran.width >= 1 && gran.height >= 1, "attachments %u: granularity %u x %u", i, gran.width,
                  gran.height);
    }
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(render_pass_clear_store),
    CT_REF(render_pass_load_and_render_area),
    CT_REF(render_pass_input_attachment),
    CT_REF(render_pass_resolves),
    CT_REF(render_pass_multiview),
    CT_REF(imageless_framebuffer),
    CT_REF(render_area_granularity),
    CT_REF(dynamic_rendering_color_depth),
    CT_REF(dynamic_rendering_resolves),
    CT_REF(dynamic_rendering_suspend_resume),
    CT_REF(dynamic_rendering_multiview),
    CT_REF(local_read_attachment_locations),
    CT_REF(local_read_input_attachment),
    CT_REF(render_pass_next_subpass),
    CT_REF(rendering_area_granularity_khr),
};

CT_SUITE(renderpass, tests)
