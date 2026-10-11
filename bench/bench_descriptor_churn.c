/* Descriptor churn: every draw gets its own descriptors - one uniform buffer
 * (the object's rectangle and color) and four material textures - delivered
 * in each of the ways an engine might:
 *
 *   bind_static      pre-built set per object; vkCmdBindDescriptorSets only
 *   update_sets      per frame vkResetDescriptorPool; per draw
 *                    vkAllocateDescriptorSets + vkUpdateDescriptorSets (5
 *                    writes) + vkCmdBindDescriptorSets
 *   update_template  the same with vkUpdateDescriptorSetWithTemplate
 *   dynamic_offset   one set per material, uniform buffer bound with a
 *                    per-draw dynamic offset
 *   push_descriptor  vkCmdPushDescriptorSet (core in 1.4)
 *   push_template    vkCmdPushDescriptorSetWithTemplate
 *
 * The first frame of each mode is read back and checked against the CPU.
 */
#include "bench_vk.h"
#include "bench_shaders.h"

#define WIDTH 1024
#define HEIGHT 1024
#define OBJECTS 10000
#define TEXTURES 16
#define TEXTURE_SIZE 4
#define MATERIALS 16 /* object i uses material i % MATERIALS */
#define SLOTS 4      /* textures per material */

enum mode { MODE_BIND_STATIC, MODE_UPDATE, MODE_UPDATE_TEMPLATE, MODE_DYNAMIC_OFFSET, MODE_PUSH, MODE_PUSH_TEMPLATE };

struct config {
    const char *name;
    enum mode mode;
    int draws;
    int frames;
};

static const struct config configs[] = {
    { "bind_static_10000", MODE_BIND_STATIC, OBJECTS, 100 },
    { "update_sets_10000", MODE_UPDATE, OBJECTS, 100 },
    { "update_template_10000", MODE_UPDATE_TEMPLATE, OBJECTS, 100 },
    { "dynamic_offset_10000", MODE_DYNAMIC_OFFSET, OBJECTS, 100 },
    { "push_descriptor_10000", MODE_PUSH, OBJECTS, 100 },
    { "push_template_10000", MODE_PUSH_TEMPLATE, OBJECTS, 100 },
};

struct object_ubo {
    float rect[4];
    float color[4];
};

/* What the update templates read. */
struct descriptors {
    VkDescriptorBufferInfo ubo;
    VkDescriptorImageInfo images[SLOTS];
};

struct scene {
    struct bench_vk *vk;
    const struct config *cfg;
    struct bench_rect rects[OBJECTS];
    uint32_t texture_rgba[TEXTURES];
    VkDeviceSize ubo_stride;
    struct bench_buffer ubo, readback;
    struct bench_image targets[BENCH_FRAMES_IN_FLIGHT];
    struct bench_image textures[TEXTURES];
    VkSampler sampler;
    VkDescriptorSetLayout set_layout, dynamic_set_layout, push_set_layout;
    VkPipelineLayout layout, dynamic_layout, push_layout;
    VkPipeline pipeline, dynamic_pipeline, push_pipeline;
    VkDescriptorPool static_pool, frame_pools[BENCH_FRAMES_IN_FLIGHT];
    VkDescriptorSet static_sets[OBJECTS], material_sets[MATERIALS];
    VkDescriptorUpdateTemplate update_template, push_template;
};

static int texture_of(int object, int slot)
{
    return (object % MATERIALS + slot * 5) % TEXTURES;
}

static void object_descriptors(struct scene *s, int i, struct descriptors *d)
{
    int k;
    d->ubo.buffer = s->ubo.buffer;
    d->ubo.offset = (VkDeviceSize)i * s->ubo_stride;
    d->ubo.range = sizeof(struct object_ubo);
    for (k = 0; k < SLOTS; k++) {
        d->images[k].sampler = s->sampler;
        d->images[k].imageView = s->textures[texture_of(i, k)].view;
        d->images[k].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
}

/* The five writes a straightforward engine issues per draw. */
static void object_writes(const struct descriptors *d, VkDescriptorSet set, VkDescriptorType ubo_type,
                          VkWriteDescriptorSet writes[1 + SLOTS])
{
    int k;
    memset(writes, 0, sizeof(VkWriteDescriptorSet) * (1 + SLOTS));
    for (k = 0; k < 1 + SLOTS; k++) {
        writes[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[k].dstSet = set;
        writes[k].dstBinding = (uint32_t)k;
        writes[k].descriptorCount = 1;
        writes[k].descriptorType = k == 0 ? ubo_type : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        if (k == 0)
            writes[k].pBufferInfo = &d->ubo;
        else
            writes[k].pImageInfo = &d->images[k - 1];
    }
}

static void record(void *user, const struct bench_frame *f)
{
    struct scene *s = user;
    struct bench_vk *vk = s->vk;
    const float clear[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    VkCommandBuffer cmd = f->cmd;
    VkPipelineBindPoint gfx = VK_PIPELINE_BIND_POINT_GRAPHICS;
    VkDescriptorSetAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s->frame_pools[f->slot],
        .descriptorSetCount = 1,
        .pSetLayouts = &s->set_layout,
    };
    VkWriteDescriptorSet writes[1 + SLOTS];
    struct descriptors d;
    VkDescriptorSet set;
    int i, n = s->cfg->draws;

    bench_cmd_begin_color(cmd, &s->targets[f->slot], clear);
    switch (s->cfg->mode) {
    case MODE_BIND_STATIC:
        vkd.CmdBindPipeline(cmd, gfx, s->pipeline);
        for (i = 0; i < n; i++) {
            vkd.CmdBindDescriptorSets(cmd, gfx, s->layout, 0, 1, &s->static_sets[i], 0, NULL);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        break;
    case MODE_UPDATE:
    case MODE_UPDATE_TEMPLATE:
        BENCH_VK(vkd.ResetDescriptorPool(vk->device, s->frame_pools[f->slot], 0));
        vkd.CmdBindPipeline(cmd, gfx, s->pipeline);
        for (i = 0; i < n; i++) {
            BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &alloc, &set));
            object_descriptors(s, i, &d);
            if (s->cfg->mode == MODE_UPDATE) {
                object_writes(&d, set, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, writes);
                vkd.UpdateDescriptorSets(vk->device, 1 + SLOTS, writes, 0, NULL);
            } else {
                vkd.UpdateDescriptorSetWithTemplate(vk->device, set, s->update_template, &d);
            }
            vkd.CmdBindDescriptorSets(cmd, gfx, s->layout, 0, 1, &set, 0, NULL);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        break;
    case MODE_DYNAMIC_OFFSET:
        vkd.CmdBindPipeline(cmd, gfx, s->dynamic_pipeline);
        for (i = 0; i < n; i++) {
            uint32_t offset = (uint32_t)((VkDeviceSize)i * s->ubo_stride);
            vkd.CmdBindDescriptorSets(cmd, gfx, s->dynamic_layout, 0, 1, &s->material_sets[i % MATERIALS], 1,
                                      &offset);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        break;
    case MODE_PUSH:
        vkd.CmdBindPipeline(cmd, gfx, s->push_pipeline);
        for (i = 0; i < n; i++) {
            object_descriptors(s, i, &d);
            object_writes(&d, VK_NULL_HANDLE, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, writes);
            vkd.CmdPushDescriptorSet(cmd, gfx, s->push_layout, 0, 1 + SLOTS, writes);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        break;
    case MODE_PUSH_TEMPLATE:
        vkd.CmdBindPipeline(cmd, gfx, s->push_pipeline);
        for (i = 0; i < n; i++) {
            object_descriptors(s, i, &d);
            vkd.CmdPushDescriptorSetWithTemplate(cmd, s->push_template, s->push_layout, 0, &d);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        break;
    }
    vkd.CmdEndRendering(cmd);
    if (f->verify)
        bench_cmd_copy_image_to_buffer(cmd, &s->targets[f->slot], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, s->readback.buffer);
}

static int verify(struct scene *s)
{
    uint32_t *want = bench_alloc(sizeof(uint32_t) * WIDTH * HEIGHT);
    int i, k, c, ok;

    for (i = 0; i < WIDTH * HEIGHT; i++)
        want[i] = bench_rgba8(0, 0, 0, 255);
    for (i = 0; i < s->cfg->draws; i++) {
        const struct bench_rect *r = &s->rects[i];
        uint32_t rgba = 0;
        for (c = 0; c < 4; c++) {
            double v = (double)((r->rgba >> (8 * c)) & 0xFF) / 255.0;
            for (k = 0; k < SLOTS; k++)
                v *= (double)((s->texture_rgba[texture_of(i, k)] >> (8 * c)) & 0xFF) / 255.0;
            rgba |= (uint32_t)(v * 255.0 + 0.5) << (8 * c);
        }
        bench_fill_rect(want, WIDTH, r->x0, r->y0, r->x1, r->y1, rgba);
    }
    /* Five-way float products: allow rounding differences of one step. */
    ok = bench_check_rgba8_tolerance(s->cfg->name, s->readback.map, want, WIDTH, HEIGHT, 1);
    free(want);
    return ok;
}

static VkPipeline make_pipeline(struct scene *s, VkPipelineLayout layout)
{
    struct bench_vk *vk = s->vk;
    struct bench_gfx_desc desc;
    VkPipeline pipeline;
    memset(&desc, 0, sizeof(desc));
    desc.vs = BENCH_SHADER(vk, rect_ubo_vert);
    desc.fs = BENCH_SHADER(vk, rect_material_frag);
    desc.layout = layout;
    desc.color_count = 1;
    desc.color_formats[0] = VK_FORMAT_R8G8B8A8_UNORM;
    pipeline = bench_gfx_pipeline(vk, &desc);
    vkd.DestroyShaderModule(vk->device, desc.vs, NULL);
    vkd.DestroyShaderModule(vk->device, desc.fs, NULL);
    return pipeline;
}

static VkDescriptorPool make_pool(struct bench_vk *vk, uint32_t sets, uint32_t ubos, uint32_t dynamic_ubos)
{
    VkDescriptorPoolSize sizes[3] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubos ? ubos : 1 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, dynamic_ubos ? dynamic_ubos : 1 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sets * SLOTS },
    };
    VkDescriptorPoolCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = sets,
        .poolSizeCount = 3,
        .pPoolSizes = sizes,
    };
    VkDescriptorPool pool = VK_NULL_HANDLE;
    BENCH_VK(vkd.CreateDescriptorPool(vk->device, &info, NULL, &pool));
    return pool;
}

static VkDescriptorUpdateTemplate make_template(struct scene *s, VkDescriptorUpdateTemplateType type,
                                                VkDescriptorSetLayout set_layout, VkPipelineLayout layout)
{
    VkDescriptorUpdateTemplateEntry entries[1 + SLOTS];
    VkDescriptorUpdateTemplateCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO,
        .descriptorUpdateEntryCount = 1 + SLOTS,
        .pDescriptorUpdateEntries = entries,
        .templateType = type,
        .descriptorSetLayout = set_layout,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .pipelineLayout = layout,
        .set = 0,
    };
    VkDescriptorUpdateTemplate tmpl = VK_NULL_HANDLE;
    int k;
    for (k = 0; k < 1 + SLOTS; k++) {
        entries[k].dstBinding = (uint32_t)k;
        entries[k].dstArrayElement = 0;
        entries[k].descriptorCount = 1;
        entries[k].descriptorType = k == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        entries[k].offset = k == 0 ? offsetof(struct descriptors, ubo)
                                   : offsetof(struct descriptors, images) + (size_t)(k - 1) * sizeof(VkDescriptorImageInfo);
        entries[k].stride = 0;
    }
    BENCH_VK(vkd.CreateDescriptorUpdateTemplate(s->vk->device, &info, NULL, &tmpl));
    return tmpl;
}

static void setup(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDeviceSize align = vk->props.limits.minUniformBufferOffsetAlignment;
    VkDescriptorSetLayoutBinding bindings[1 + SLOTS];
    VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxLod = 0.0f,
    };
    struct object_ubo *ubo_data;
    uint32_t texels[TEXTURE_SIZE * TEXTURE_SIZE];
    uint64_t seed = 0x7E57ED5Eull;
    int i, k;

    /* Targets, textures, per-object uniform data. */
    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++)
        bench_image_create(vk, &s->targets[i], VK_FORMAT_R8G8B8A8_UNORM, WIDTH, HEIGHT, 1, VK_SAMPLE_COUNT_1_BIT,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0);
    bench_buffer_create(vk, &s->readback, WIDTH * HEIGHT * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, BENCH_HOST,
                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    for (i = 0; i < TEXTURES; i++) {
        /* Solid colors with every channel in [160, 255]. */
        s->texture_rgba[i] = bench_rgba8(160 + bench_rand(&seed) % 96, 160 + bench_rand(&seed) % 96,
                                         160 + bench_rand(&seed) % 96, 255);
        for (k = 0; k < TEXTURE_SIZE * TEXTURE_SIZE; k++)
            texels[k] = s->texture_rgba[i];
        bench_image_create(vk, &s->textures[i], VK_FORMAT_R8G8B8A8_UNORM, TEXTURE_SIZE, TEXTURE_SIZE, 1,
                           VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0);
        bench_image_upload(vk, &s->textures[i], texels, sizeof(texels), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    BENCH_VK(vkd.CreateSampler(vk->device, &sampler_info, NULL, &s->sampler));

    bench_rect_scene(s->rects, OBJECTS, WIDTH, HEIGHT, 17, 0, 0xDE5C0001ull);
    if (align < 1)
        align = 1;
    s->ubo_stride = (sizeof(struct object_ubo) + align - 1) / align * align;
    ubo_data = bench_alloc((size_t)(s->ubo_stride * OBJECTS));
    for (i = 0; i < OBJECTS; i++) {
        struct object_ubo *o = (struct object_ubo *)((char *)ubo_data + (size_t)(s->ubo_stride * (VkDeviceSize)i));
        bench_rect_ndc(&s->rects[i], WIDTH, HEIGHT, 0, 0, o->rect);
        bench_unorm4(s->rects[i].rgba, o->color);
    }
    bench_buffer_create(vk, &s->ubo, s->ubo_stride * OBJECTS,
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->ubo, 0, ubo_data, s->ubo_stride * OBJECTS);
    free(ubo_data);

    /* Layouts: plain, dynamic uniform buffer, push. */
    for (k = 0; k < 1 + SLOTS; k++) {
        bindings[k].binding = (uint32_t)k;
        bindings[k].descriptorType = k == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[k].descriptorCount = 1;
        bindings[k].stageFlags = k == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[k].pImmutableSamplers = NULL;
    }
    s->set_layout = bench_set_layout(vk, 1 + SLOTS, bindings, 0);
    if (vk->push_descriptor)
        s->push_set_layout = bench_set_layout(vk, 1 + SLOTS, bindings, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT);
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    s->dynamic_set_layout = bench_set_layout(vk, 1 + SLOTS, bindings, 0);
    s->layout = bench_pipeline_layout(vk, 1, &s->set_layout, 0, 0);
    s->dynamic_layout = bench_pipeline_layout(vk, 1, &s->dynamic_set_layout, 0, 0);
    s->pipeline = make_pipeline(s, s->layout);
    s->dynamic_pipeline = make_pipeline(s, s->dynamic_layout);
    if (vk->push_descriptor) {
        s->push_layout = bench_pipeline_layout(vk, 1, &s->push_set_layout, 0, 0);
        s->push_pipeline = make_pipeline(s, s->push_layout);
        s->push_template = make_template(s, VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS, VK_NULL_HANDLE,
                                         s->push_layout);
    }
    s->update_template = make_template(s, VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET, s->set_layout,
                                       VK_NULL_HANDLE);

    /* Pools; the static sets and per-material sets are written once. */
    s->static_pool = make_pool(vk, OBJECTS + MATERIALS, OBJECTS, MATERIALS);
    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++)
        s->frame_pools[i] = make_pool(vk, OBJECTS, OBJECTS, 0);
    for (i = 0; i < OBJECTS + MATERIALS; i++) {
        int material = i >= OBJECTS;
        VkDescriptorSet *set = material ? &s->material_sets[i - OBJECTS] : &s->static_sets[i];
        VkDescriptorSetAllocateInfo alloc = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = s->static_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = material ? &s->dynamic_set_layout : &s->set_layout,
        };
        VkWriteDescriptorSet writes[1 + SLOTS];
        struct descriptors d;
        BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &alloc, set));
        object_descriptors(s, material ? i - OBJECTS : i, &d);
        if (material) {
            d.ubo.offset = 0; /* the dynamic offset selects the object */
            object_writes(&d, *set, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, writes);
        } else {
            object_writes(&d, *set, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, writes);
        }
        vkd.UpdateDescriptorSets(vk->device, 1 + SLOTS, writes, 0, NULL);
    }
}

static void teardown(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDevice dev = vk->device;
    int i;
    vkd.DeviceWaitIdle(dev);
    vkd.DestroyDescriptorPool(dev, s->static_pool, NULL);
    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++) {
        vkd.DestroyDescriptorPool(dev, s->frame_pools[i], NULL);
        bench_image_destroy(vk, &s->targets[i]);
    }
    vkd.DestroyDescriptorUpdateTemplate(dev, s->update_template, NULL);
    if (s->push_template)
        vkd.DestroyDescriptorUpdateTemplate(dev, s->push_template, NULL);
    vkd.DestroyPipeline(dev, s->pipeline, NULL);
    vkd.DestroyPipeline(dev, s->dynamic_pipeline, NULL);
    if (s->push_pipeline)
        vkd.DestroyPipeline(dev, s->push_pipeline, NULL);
    vkd.DestroyPipelineLayout(dev, s->layout, NULL);
    vkd.DestroyPipelineLayout(dev, s->dynamic_layout, NULL);
    if (s->push_layout)
        vkd.DestroyPipelineLayout(dev, s->push_layout, NULL);
    vkd.DestroyDescriptorSetLayout(dev, s->set_layout, NULL);
    vkd.DestroyDescriptorSetLayout(dev, s->dynamic_set_layout, NULL);
    if (s->push_set_layout)
        vkd.DestroyDescriptorSetLayout(dev, s->push_set_layout, NULL);
    vkd.DestroySampler(dev, s->sampler, NULL);
    for (i = 0; i < TEXTURES; i++)
        bench_image_destroy(vk, &s->textures[i]);
    bench_buffer_destroy(vk, &s->ubo);
    bench_buffer_destroy(vk, &s->readback);
}

int main(void)
{
    struct bench_vk vk;
    struct bench_loop loop;
    struct scene *s;
    size_t c;

    bench_begin("descriptor_churn");
    bench_vk_init(&vk, BENCH_NEED_DYNAMIC_RENDERING | BENCH_NEED_SYNC2 | BENCH_WANT_PUSH_DESCRIPTOR);
    s = bench_alloc(sizeof(*s));
    s->vk = &vk;
    setup(s);
    bench_loop_init(&loop, &vk, 1000);

    for (c = 0; c < sizeof(configs) / sizeof(configs[0]); c++) {
        const struct config *cfg = &configs[c];
        int frames = bench_frames(cfg->frames);
        struct bench_json j;
        double cpu_median;

        if (!bench_selected(cfg->name))
            continue;
        bench_config = cfg->name;
        if ((cfg->mode == MODE_PUSH || cfg->mode == MODE_PUSH_TEMPLATE) &&
            (!vk.push_descriptor || !vkd.CmdPushDescriptorSet || !vkd.CmdPushDescriptorSetWithTemplate)) {
            bench_skip(cfg->name, "feature pushDescriptor (Vulkan 1.4) not supported");
            continue;
        }
        s->cfg = cfg;
        bench_loop_verify(&loop, record, s);
        if (!verify(s))
            continue;
        bench_loop_run(&loop, record, s, 10, frames < 1000 ? frames : 1000);
        cpu_median = bench_loop_cpu_median(&loop);

        bench_json_begin(&j, cfg->name);
        bench_json_int(&j, "draws", cfg->draws);
        bench_json_int(&j, "descriptors_per_draw", 1 + SLOTS);
        bench_loop_report(&loop, &j);
        bench_json_num(&j, "cpu_ns_per_draw", cpu_median * 1000.0 / cfg->draws);
        bench_json_end(&j);
    }
    bench_config = NULL;

    bench_loop_destroy(&loop);
    teardown(s);
    free(s);
    bench_vk_destroy(&vk);
    return bench_exit_code();
}
