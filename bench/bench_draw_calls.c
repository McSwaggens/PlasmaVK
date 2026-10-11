/* Draw calls: thousands of small draws per frame into a 1024x1024 target,
 * the CPU-bound pattern of a scene with many objects.
 *
 *   push_constants   per draw: vkCmdPushConstants + vkCmdDraw
 *   indexed_vb       per draw: vkCmdBindVertexBuffers (one of 64 meshes) +
 *                    vkCmdPushConstants + vkCmdDrawIndexed
 *   pipeline_switch  push_constants, plus a different pipeline (one of 4
 *                    materials) every 8 draws
 *   instanced        objects in batches: vkCmdPushConstants (batch offset) +
 *                    vkCmdDraw with per-instance vertex attributes
 *
 * Reports CPU recording + submit time and GPU time per frame. The first
 * frame is read back and compared with a CPU rasterization of the scene.
 */
#include "bench_vk.h"
#include "bench_shaders.h"

#define WIDTH 1024
#define HEIGHT 1024
#define MAX_OBJECTS 50000
#define MESHES 64
#define MATERIALS 4
#define DRAWS_PER_MATERIAL 8

enum mode { MODE_PUSH, MODE_INDEXED_VB, MODE_PIPELINE_SWITCH, MODE_INSTANCED };

struct config {
    const char *name;
    enum mode mode;
    int objects;
    int batch; /* instances per draw */
    int frames;
};

static const struct config configs[] = {
    { "push_constants_1000", MODE_PUSH, 1000, 1, 300 },
    { "push_constants_10000", MODE_PUSH, 10000, 1, 200 },
    { "push_constants_50000", MODE_PUSH, 50000, 1, 60 },
    { "indexed_vb_10000", MODE_INDEXED_VB, 10000, 1, 200 },
    { "pipeline_switch_10000", MODE_PIPELINE_SWITCH, 10000, 1, 200 },
    { "instanced_50000_x64", MODE_INSTANCED, 50000, 64, 200 },
    { "instanced_50000_x1024", MODE_INSTANCED, 50000, 1024, 200 },
};

struct push {
    float rect[4];
    float color[4];
};

struct instance {
    float rect[4];
    uint32_t rgba;
};

struct scene {
    struct bench_vk *vk;
    const struct config *cfg;
    struct bench_rect rects[MAX_OBJECTS];
    struct push push[MAX_OBJECTS];       /* per-draw push constants */
    float batch_offset[MAX_OBJECTS][2];  /* instanced: per-draw push constants */
    struct bench_image targets[BENCH_FRAMES_IN_FLIGHT];
    struct bench_buffer readback, instances, mesh_vertices, mesh_indices;
    VkPipelineLayout layout, instanced_layout;
    VkPipeline push_pipeline, mesh_pipeline, instanced_pipeline;
    VkPipeline materials[MATERIALS];
};

/* Batch b of an instanced draw moves by a few whole pixels. */
static void batch_offset_px(int b, int *dx, int *dy)
{
    *dx = ((b * 7) % 5 - 2) * 4;
    *dy = ((b * 3) % 5 - 2) * 4;
}

static void record(void *user, const struct bench_frame *f)
{
    struct scene *s = user;
    const struct config *cfg = s->cfg;
    const float clear[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    VkCommandBuffer cmd = f->cmd;
    VkDeviceSize zero = 0;
    int i;

    bench_cmd_begin_color(cmd, &s->targets[f->slot], clear);
    switch (cfg->mode) {
    case MODE_PUSH:
        vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->push_pipeline);
        for (i = 0; i < cfg->objects; i++) {
            vkd.CmdPushConstants(cmd, s->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(struct push), &s->push[i]);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        break;
    case MODE_INDEXED_VB:
        vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->mesh_pipeline);
        vkd.CmdBindIndexBuffer(cmd, s->mesh_indices.buffer, 0, VK_INDEX_TYPE_UINT16);
        for (i = 0; i < cfg->objects; i++) {
            VkDeviceSize offset = (VkDeviceSize)(i % MESHES) * 4 * 2 * sizeof(float);
            vkd.CmdBindVertexBuffers(cmd, 0, 1, &s->mesh_vertices.buffer, &offset);
            vkd.CmdPushConstants(cmd, s->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(struct push), &s->push[i]);
            vkd.CmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
        }
        break;
    case MODE_PIPELINE_SWITCH:
        for (i = 0; i < cfg->objects; i++) {
            if (i % DRAWS_PER_MATERIAL == 0)
                vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    s->materials[(i / DRAWS_PER_MATERIAL) % MATERIALS]);
            vkd.CmdPushConstants(cmd, s->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(struct push), &s->push[i]);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        break;
    case MODE_INSTANCED:
        vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->instanced_pipeline);
        vkd.CmdBindVertexBuffers(cmd, 0, 1, &s->instances.buffer, &zero);
        for (i = 0; i * cfg->batch < cfg->objects; i++) {
            int first = i * cfg->batch;
            int count = cfg->objects - first < cfg->batch ? cfg->objects - first : cfg->batch;
            vkd.CmdPushConstants(cmd, s->instanced_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 2,
                                 s->batch_offset[i]);
            vkd.CmdDraw(cmd, 6, (uint32_t)count, 0, (uint32_t)first);
        }
        break;
    }
    vkd.CmdEndRendering(cmd);
    if (f->verify)
        bench_cmd_copy_image_to_buffer(cmd, &s->targets[f->slot], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, s->readback.buffer);
}

/* Channel order of material m (see rect_flat.frag). */
static uint32_t swizzle(uint32_t rgba, int m)
{
    uint32_t r = rgba & 0xFF, g = (rgba >> 8) & 0xFF, b = (rgba >> 16) & 0xFF, a = rgba >> 24;
    switch (m) {
    case 1: return bench_rgba8(g, b, r, a);
    case 2: return bench_rgba8(b, r, g, a);
    case 3: return bench_rgba8(b, g, r, a);
    default: return rgba;
    }
}

static int verify(struct scene *s)
{
    const struct config *cfg = s->cfg;
    uint32_t *want = bench_alloc(sizeof(uint32_t) * WIDTH * HEIGHT);
    int i, ok;

    for (i = 0; i < WIDTH * HEIGHT; i++)
        want[i] = bench_rgba8(0, 0, 0, 255);
    for (i = 0; i < cfg->objects; i++) {
        const struct bench_rect *r = &s->rects[i];
        int dx = 0, dy = 0;
        uint32_t rgba = r->rgba;
        if (cfg->mode == MODE_INSTANCED)
            batch_offset_px(i / cfg->batch, &dx, &dy);
        if (cfg->mode == MODE_PIPELINE_SWITCH)
            rgba = swizzle(rgba, (i / DRAWS_PER_MATERIAL) % MATERIALS);
        bench_fill_rect(want, WIDTH, r->x0 + dx, r->y0 + dy, r->x1 + dx, r->y1 + dy, rgba);
    }
    ok = bench_check_rgba8(cfg->name, s->readback.map, want, WIDTH, HEIGHT);
    free(want);
    return ok;
}

static void setup(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    struct bench_gfx_desc desc;
    VkShaderModule push_vs, mesh_vs, instanced_vs, fs;
    uint32_t material;
    VkSpecializationMapEntry entry = { 0, 0, sizeof(uint32_t) };
    VkSpecializationInfo spec = { 1, &entry, sizeof(uint32_t), &material };
    static const uint16_t indices[6] = { 0, 1, 2, 1, 3, 2 };
    float corners[MESHES][4][2];
    struct instance *inst;
    int i;

    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++)
        bench_image_create(vk, &s->targets[i], VK_FORMAT_R8G8B8A8_UNORM, WIDTH, HEIGHT, 1, VK_SAMPLE_COUNT_1_BIT,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0);
    bench_buffer_create(vk, &s->readback, WIDTH * HEIGHT * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, BENCH_HOST,
                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

    /* The scene, and everything the modes feed it from. */
    bench_rect_scene(s->rects, MAX_OBJECTS, WIDTH, HEIGHT, 17, 16, 0xD4A3C411ull);
    inst = bench_alloc(sizeof(*inst) * MAX_OBJECTS);
    for (i = 0; i < MAX_OBJECTS; i++) {
        int dx, dy;
        bench_rect_ndc(&s->rects[i], WIDTH, HEIGHT, 0, 0, s->push[i].rect);
        bench_unorm4(s->rects[i].rgba, s->push[i].color);
        memcpy(inst[i].rect, s->push[i].rect, sizeof(inst[i].rect));
        inst[i].rgba = s->rects[i].rgba;
        batch_offset_px(i, &dx, &dy);
        s->batch_offset[i][0] = (float)dx * 2.0f / WIDTH;
        s->batch_offset[i][1] = (float)dy * 2.0f / HEIGHT;
    }
    bench_buffer_create(vk, &s->instances, sizeof(*inst) * MAX_OBJECTS,
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->instances, 0, inst, sizeof(*inst) * MAX_OBJECTS);
    free(inst);
    for (i = 0; i < MESHES; i++) {
        int c;
        for (c = 0; c < 4; c++) {
            corners[i][c][0] = (float)(c & 1);
            corners[i][c][1] = (float)(c >> 1);
        }
    }
    bench_buffer_create(vk, &s->mesh_vertices, sizeof(corners),
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->mesh_vertices, 0, corners, sizeof(corners));
    bench_buffer_create(vk, &s->mesh_indices, sizeof(indices),
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->mesh_indices, 0, indices, sizeof(indices));

    /* Pipelines. */
    s->layout = bench_pipeline_layout(vk, 0, NULL, sizeof(struct push), VK_SHADER_STAGE_VERTEX_BIT);
    s->instanced_layout = bench_pipeline_layout(vk, 0, NULL, sizeof(float) * 2, VK_SHADER_STAGE_VERTEX_BIT);
    push_vs = BENCH_SHADER(vk, rect_push_vert);
    mesh_vs = BENCH_SHADER(vk, rect_mesh_vert);
    instanced_vs = BENCH_SHADER(vk, rect_instanced_vert);
    fs = BENCH_SHADER(vk, rect_flat_frag);

    memset(&desc, 0, sizeof(desc));
    desc.vs = push_vs;
    desc.fs = fs;
    desc.fs_spec = &spec;
    desc.layout = s->layout;
    desc.color_count = 1;
    desc.color_formats[0] = VK_FORMAT_R8G8B8A8_UNORM;
    for (material = 0; material < MATERIALS; material++)
        s->materials[material] = bench_gfx_pipeline(vk, &desc);
    s->push_pipeline = s->materials[0];

    desc.vs = mesh_vs;
    desc.fs_spec = NULL;
    desc.binding_count = 1;
    desc.bindings[0] = (VkVertexInputBindingDescription){ 0, 2 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX };
    desc.attribute_count = 1;
    desc.attributes[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 };
    s->mesh_pipeline = bench_gfx_pipeline(vk, &desc);

    desc.vs = instanced_vs;
    desc.layout = s->instanced_layout;
    desc.bindings[0] = (VkVertexInputBindingDescription){ 0, sizeof(struct instance), VK_VERTEX_INPUT_RATE_INSTANCE };
    desc.attribute_count = 2;
    desc.attributes[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
    desc.attributes[1] = (VkVertexInputAttributeDescription){ 1, 0, VK_FORMAT_R8G8B8A8_UNORM,
                                                              (uint32_t)(4 * sizeof(float)) };
    s->instanced_pipeline = bench_gfx_pipeline(vk, &desc);

    vkd.DestroyShaderModule(vk->device, push_vs, NULL);
    vkd.DestroyShaderModule(vk->device, mesh_vs, NULL);
    vkd.DestroyShaderModule(vk->device, instanced_vs, NULL);
    vkd.DestroyShaderModule(vk->device, fs, NULL);
}

static void teardown(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    int i;
    vkd.DeviceWaitIdle(vk->device);
    for (i = 0; i < MATERIALS; i++)
        vkd.DestroyPipeline(vk->device, s->materials[i], NULL);
    vkd.DestroyPipeline(vk->device, s->mesh_pipeline, NULL);
    vkd.DestroyPipeline(vk->device, s->instanced_pipeline, NULL);
    vkd.DestroyPipelineLayout(vk->device, s->layout, NULL);
    vkd.DestroyPipelineLayout(vk->device, s->instanced_layout, NULL);
    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++)
        bench_image_destroy(vk, &s->targets[i]);
    bench_buffer_destroy(vk, &s->readback);
    bench_buffer_destroy(vk, &s->instances);
    bench_buffer_destroy(vk, &s->mesh_vertices);
    bench_buffer_destroy(vk, &s->mesh_indices);
}

int main(void)
{
    struct bench_vk vk;
    struct bench_loop loop;
    struct scene *s;
    size_t c;

    bench_begin("draw_calls");
    bench_vk_init(&vk, BENCH_NEED_DYNAMIC_RENDERING | BENCH_NEED_SYNC2);
    s = bench_alloc(sizeof(*s));
    s->vk = &vk;
    setup(s);
    bench_loop_init(&loop, &vk, 1000);

    for (c = 0; c < sizeof(configs) / sizeof(configs[0]); c++) {
        const struct config *cfg = &configs[c];
        int frames = bench_frames(cfg->frames), draws = (cfg->objects + cfg->batch - 1) / cfg->batch;
        struct bench_json j;
        double cpu_median;

        if (!bench_selected(cfg->name))
            continue;
        bench_config = cfg->name;
        s->cfg = cfg;
        bench_loop_verify(&loop, record, s);
        if (!verify(s))
            continue;
        bench_loop_run(&loop, record, s, 10, frames < 1000 ? frames : 1000);
        cpu_median = bench_loop_cpu_median(&loop);

        bench_json_begin(&j, cfg->name);
        bench_json_int(&j, "objects", cfg->objects);
        bench_json_int(&j, "draws", draws);
        bench_json_int(&j, "instances_per_draw", cfg->batch);
        bench_loop_report(&loop, &j);
        bench_json_num(&j, "cpu_ns_per_draw", cpu_median * 1000.0 / draws);
        bench_json_end(&j);
    }
    bench_config = NULL;

    bench_loop_destroy(&loop);
    teardown(s);
    free(s);
    bench_vk_destroy(&vk);
    return bench_exit_code();
}
