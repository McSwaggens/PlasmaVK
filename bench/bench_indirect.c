/* Indirect draws of a GPU-driven renderer. A culling pass (not timed here:
 * the records are prepared once) leaves draw records in a buffer sized for
 * every object, 65536; each record draws one object, a rectangle picked by
 * firstInstance with per-instance attributes. One indirect call per frame:
 *
 *   mdi_1000              vkCmdDrawIndirect of 1000 compacted records
 *   mdi_1000_of_16384     vkCmdDrawIndirect of 16384 records, the 15384 culled
 *                         ones with instanceCount 0 (culling without compaction)
 *   count_1000_cap1000    vkCmdDrawIndirectCount, 1000 records counted, and
 *                         maxDrawCount 1000
 *   count_1000_cap16384   ... with maxDrawCount 16384, and
 *   count_1000_cap65536   65536: the buffer's capacity, as engines pass it
 *   icount_1000_cap16384  vkCmdDrawIndexedIndirectCount, 1000 of at most 16384
 *   count_10000_cap16384  10000 records counted, at most 16384
 *
 * The count configurations need drawIndirectCount and are skipped without
 * it. Records past the count are valid draws too: drawing any of them would
 * show in the output. The first frame of each configuration is read back and
 * compared with a CPU rendering of the drawn records.
 */
#include "bench_vk.h"
#include "bench_shaders.h"

#define WIDTH 1024
#define HEIGHT 1024
#define RECORDS 65536 /* the record buffers' capacity: one per object */
#define CULLED_DRAWS 16384

enum mode { MODE_MDI, MODE_MDI_CULLED, MODE_COUNT };

struct config {
    const char *name;
    enum mode mode;
    int indexed;
    uint32_t count, max_draws;
    int frames;
};

static const struct config configs[] = {
    { "mdi_1000", MODE_MDI, 0, 1000, 1000, 200 },
    { "mdi_1000_of_16384", MODE_MDI_CULLED, 0, 1000, CULLED_DRAWS, 200 },
    { "count_1000_cap1000", MODE_COUNT, 0, 1000, 1000, 200 },
    { "count_1000_cap16384", MODE_COUNT, 0, 1000, 16384, 200 },
    { "count_1000_cap65536", MODE_COUNT, 0, 1000, RECORDS, 100 },
    { "icount_1000_cap16384", MODE_COUNT, 1, 1000, 16384, 200 },
    { "count_10000_cap16384", MODE_COUNT, 0, 10000, 16384, 200 },
};

/* rect_instanced.vert's per-instance attributes. */
struct instance {
    float rect[4];
    uint32_t rgba;
};

/* The vkCmdDraw(Indexed)IndirectCount commands, which bench_vk.h does not load. */
static struct {
    PFN_vkCmdDrawIndirect CmdDrawIndirect;
    PFN_vkCmdDrawIndexedIndirect CmdDrawIndexedIndirect;
    PFN_vkCmdDrawIndirectCount CmdDrawIndirectCount;
    PFN_vkCmdDrawIndexedIndirectCount CmdDrawIndexedIndirectCount;
} vkx;

static int draw_indirect_count; /* drawIndirectCount enabled */

struct scene {
    struct bench_vk *vk;
    const struct config *cfg;
    struct bench_rect rects[RECORDS];
    struct bench_image targets[BENCH_FRAMES_IN_FLIGHT];
    struct bench_buffer readback, instances, records, indexed_records, culled_records, counts, indices;
    VkPipelineLayout layout;
    VkPipeline pipeline;
};

/* The object compacted record i draws. */
static uint32_t record_object(uint32_t i)
{
    return (i * 7u) % RECORDS;
}

/* Whether record r of the uncompacted (culled) records survived culling: 1000
 * of 16384, spread evenly. */
static int record_visible(uint32_t r)
{
    return r * 1000u / CULLED_DRAWS != (r + 1) * 1000u / CULLED_DRAWS;
}

/* bench_vk_init enables only what every benchmark needs: recreate the device
 * with multiDrawIndirect and drawIndirectFirstInstance, and drawIndirectCount
 * where the device has it. */
static void enable_indirect_features(struct bench_vk *vk)
{
    VkPhysicalDeviceVulkan12Features have12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceFeatures2 have = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &have12 };
    VkPhysicalDeviceVulkan13Features want13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
    };
    VkPhysicalDeviceVulkan12Features want12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                                .pNext = &want13 };
    VkPhysicalDeviceFeatures2 want = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &want12 };
    const char *extensions[1] = { "VK_KHR_portability_subset" };
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = vk->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &want,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = vk->portability_subset ? 1 : 0,
        .ppEnabledExtensionNames = extensions,
    };
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = vk->queue_family,
    };
    VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };

    vkGetPhysicalDeviceFeatures2(vk->physical_device, &have);
    if (!have.features.multiDrawIndirect || !have.features.drawIndirectFirstInstance)
        bench_fatal("multiDrawIndirect and drawIndirectFirstInstance are not supported");
    want.features.multiDrawIndirect = VK_TRUE;
    want.features.drawIndirectFirstInstance = VK_TRUE;
    want12.drawIndirectCount = have12.drawIndirectCount;
    draw_indirect_count = have12.drawIndirectCount != VK_FALSE;

    vkd.DestroyFence(vk->device, vk->fence, NULL);
    vkd.DestroyCommandPool(vk->device, vk->pool, NULL);
    vkd.DestroyDevice(vk->device, NULL);
    vk->device = VK_NULL_HANDLE;
    BENCH_VK(vkCreateDevice(vk->physical_device, &device_info, NULL, &vk->device));
#define LOAD(table, name)                                                                   \
    table.name = (PFN_vk##name)vkGetDeviceProcAddr(vk->device, "vk" #name);                 \
    if (!table.name)                                                                        \
        bench_fatal("vkGetDeviceProcAddr(vk" #name ") returned NULL");
#define LOAD_VKD(name) LOAD(vkd, name)
#define LOAD_OPTIONAL(name) vkd.name = (PFN_vk##name)vkGetDeviceProcAddr(vk->device, "vk" #name);
    BENCH_DEVICE_COMMANDS(LOAD_VKD)
    BENCH_DEVICE_COMMANDS_OPTIONAL(LOAD_OPTIONAL)
    LOAD(vkx, CmdDrawIndirect)
    LOAD(vkx, CmdDrawIndexedIndirect)
    if (draw_indirect_count) {
        LOAD(vkx, CmdDrawIndirectCount)
        LOAD(vkx, CmdDrawIndexedIndirectCount)
    }
#undef LOAD_OPTIONAL
#undef LOAD_VKD
#undef LOAD
    vkd.GetDeviceQueue(vk->device, vk->queue_family, 0, &vk->queue);
    BENCH_VK(vkd.CreateCommandPool(vk->device, &pool_info, NULL, &vk->pool));
    BENCH_VK(vkd.CreateFence(vk->device, &fence_info, NULL, &vk->fence));
}

static void record(void *user, const struct bench_frame *f)
{
    struct scene *s = user;
    const struct config *cfg = s->cfg;
    const float clear[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    const float offset[2] = { 0.0f, 0.0f };
    VkCommandBuffer cmd = f->cmd;
    VkDeviceSize zero = 0;
    /* The count buffer holds 1000 and 10000. */
    VkDeviceSize count_offset = cfg->count == 1000 ? 0 : 4;

    bench_cmd_begin_color(cmd, &s->targets[f->slot], clear);
    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->pipeline);
    vkd.CmdBindVertexBuffers(cmd, 0, 1, &s->instances.buffer, &zero);
    vkd.CmdPushConstants(cmd, s->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(offset), offset);
    if (cfg->indexed)
        vkd.CmdBindIndexBuffer(cmd, s->indices.buffer, 0, VK_INDEX_TYPE_UINT16);
    switch (cfg->mode) {
    case MODE_MDI:
        vkx.CmdDrawIndirect(cmd, s->records.buffer, 0, cfg->count, sizeof(VkDrawIndirectCommand));
        break;
    case MODE_MDI_CULLED:
        vkx.CmdDrawIndirect(cmd, s->culled_records.buffer, 0, cfg->max_draws, sizeof(VkDrawIndirectCommand));
        break;
    case MODE_COUNT:
        if (cfg->indexed)
            vkx.CmdDrawIndexedIndirectCount(cmd, s->indexed_records.buffer, 0, s->counts.buffer, count_offset,
                                            cfg->max_draws, sizeof(VkDrawIndexedIndirectCommand));
        else
            vkx.CmdDrawIndirectCount(cmd, s->records.buffer, 0, s->counts.buffer, count_offset, cfg->max_draws,
                                     sizeof(VkDrawIndirectCommand));
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
    const struct config *cfg = s->cfg;
    uint32_t *want = bench_alloc(sizeof(uint32_t) * WIDTH * HEIGHT), i;
    int ok;
    for (i = 0; i < WIDTH * HEIGHT; i++)
        want[i] = bench_rgba8(0, 0, 0, 255);
    /* Records draw in order: later ones cover earlier ones. */
    for (i = 0; i < (cfg->mode == MODE_MDI_CULLED ? cfg->max_draws : cfg->count); i++) {
        const struct bench_rect *r;
        if (cfg->mode == MODE_MDI_CULLED && !record_visible(i))
            continue;
        r = &s->rects[cfg->mode == MODE_MDI_CULLED ? i : record_object(i)];
        bench_fill_rect(want, WIDTH, r->x0, r->y0, r->x1, r->y1, r->rgba);
    }
    ok = bench_check_rgba8(cfg->name, s->readback.map, want, WIDTH, HEIGHT);
    free(want);
    return ok;
}

static void setup(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    const VkBufferUsageFlags indirect = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    static const uint16_t indices[6] = { 0, 1, 2, 3, 4, 5 };
    const uint32_t counts[2] = { 1000, 10000 };
    struct bench_gfx_desc desc;
    VkShaderModule vs, fs;
    struct instance *inst = bench_alloc(sizeof(*inst) * RECORDS);
    VkDrawIndirectCommand *rec = bench_alloc(sizeof(*rec) * RECORDS);
    VkDrawIndexedIndirectCommand *irec = bench_alloc(sizeof(*irec) * RECORDS);
    uint32_t i;

    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++)
        bench_image_create(vk, &s->targets[i], VK_FORMAT_R8G8B8A8_UNORM, WIDTH, HEIGHT, 1, VK_SAMPLE_COUNT_1_BIT,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0);
    bench_buffer_create(vk, &s->readback, WIDTH * HEIGHT * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, BENCH_HOST,
                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

    /* One small rectangle per object. */
    bench_rect_scene(s->rects, RECORDS, WIDTH, HEIGHT, 17, 16, 0xD4A3C411ull);
    for (i = 0; i < RECORDS; i++) {
        bench_rect_ndc(&s->rects[i], WIDTH, HEIGHT, 0, 0, inst[i].rect);
        inst[i].rgba = s->rects[i].rgba;
    }
    bench_buffer_create(vk, &s->instances, sizeof(*inst) * RECORDS,
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->instances, 0, inst, sizeof(*inst) * RECORDS);

    /* Compacted records: every one a valid draw, also past the count. */
    for (i = 0; i < RECORDS; i++) {
        rec[i].vertexCount = 6;
        rec[i].instanceCount = 1;
        rec[i].firstVertex = 0;
        rec[i].firstInstance = record_object(i);
        irec[i].indexCount = 6;
        irec[i].instanceCount = 1;
        irec[i].firstIndex = 0;
        irec[i].vertexOffset = 0;
        irec[i].firstInstance = record_object(i);
    }
    bench_buffer_create(vk, &s->records, sizeof(*rec) * RECORDS, indirect, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->records, 0, rec, sizeof(*rec) * RECORDS);
    bench_buffer_create(vk, &s->indexed_records, sizeof(*irec) * RECORDS, indirect, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->indexed_records, 0, irec, sizeof(*irec) * RECORDS);
    /* Uncompacted records: object r, culled ones with no instances. */
    for (i = 0; i < CULLED_DRAWS; i++) {
        rec[i].instanceCount = record_visible(i) ? 1 : 0;
        rec[i].firstInstance = i;
    }
    bench_buffer_create(vk, &s->culled_records, sizeof(*rec) * CULLED_DRAWS, indirect, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->culled_records, 0, rec, sizeof(*rec) * CULLED_DRAWS);
    bench_buffer_create(vk, &s->counts, sizeof(counts), indirect, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->counts, 0, counts, sizeof(counts));
    bench_buffer_create(vk, &s->indices, sizeof(indices),
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->indices, 0, indices, sizeof(indices));
    free(inst);
    free(rec);
    free(irec);

    s->layout = bench_pipeline_layout(vk, 0, NULL, sizeof(float) * 2, VK_SHADER_STAGE_VERTEX_BIT);
    vs = BENCH_SHADER(vk, rect_instanced_vert);
    fs = BENCH_SHADER(vk, rect_flat_frag);
    memset(&desc, 0, sizeof(desc));
    desc.vs = vs;
    desc.fs = fs;
    desc.layout = s->layout;
    desc.color_count = 1;
    desc.color_formats[0] = VK_FORMAT_R8G8B8A8_UNORM;
    desc.binding_count = 1;
    desc.bindings[0] = (VkVertexInputBindingDescription){ 0, sizeof(struct instance), VK_VERTEX_INPUT_RATE_INSTANCE };
    desc.attribute_count = 2;
    desc.attributes[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
    desc.attributes[1] = (VkVertexInputAttributeDescription){ 1, 0, VK_FORMAT_R8G8B8A8_UNORM,
                                                              (uint32_t)offsetof(struct instance, rgba) };
    s->pipeline = bench_gfx_pipeline(vk, &desc);
    vkd.DestroyShaderModule(vk->device, vs, NULL);
    vkd.DestroyShaderModule(vk->device, fs, NULL);
}

static void teardown(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    uint32_t i;
    vkd.DeviceWaitIdle(vk->device);
    vkd.DestroyPipeline(vk->device, s->pipeline, NULL);
    vkd.DestroyPipelineLayout(vk->device, s->layout, NULL);
    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++)
        bench_image_destroy(vk, &s->targets[i]);
    bench_buffer_destroy(vk, &s->readback);
    bench_buffer_destroy(vk, &s->instances);
    bench_buffer_destroy(vk, &s->records);
    bench_buffer_destroy(vk, &s->indexed_records);
    bench_buffer_destroy(vk, &s->culled_records);
    bench_buffer_destroy(vk, &s->counts);
    bench_buffer_destroy(vk, &s->indices);
}

int main(void)
{
    struct bench_vk vk;
    struct bench_loop loop;
    struct scene *s;
    size_t c;

    bench_begin("indirect");
    bench_vk_init(&vk, BENCH_NEED_DYNAMIC_RENDERING | BENCH_NEED_SYNC2);
    enable_indirect_features(&vk);
    s = bench_alloc(sizeof(*s));
    s->vk = &vk;
    setup(s);
    bench_loop_init(&loop, &vk, 1000);

    for (c = 0; c < sizeof(configs) / sizeof(configs[0]); c++) {
        const struct config *cfg = &configs[c];
        int frames = bench_frames(cfg->frames);
        struct bench_json j;

        if (!bench_selected(cfg->name))
            continue;
        bench_config = cfg->name;
        if (cfg->mode == MODE_COUNT && !draw_indirect_count) {
            bench_skip(cfg->name, "drawIndirectCount not supported");
            continue;
        }
        s->cfg = cfg;
        bench_loop_verify(&loop, record, s);
        if (!verify(s))
            continue;
        bench_loop_run(&loop, record, s, 10, frames < 1000 ? frames : 1000);

        bench_json_begin(&j, cfg->name);
        bench_json_int(&j, "draws", cfg->count);
        bench_json_int(&j, "max_draws", cfg->max_draws);
        bench_json_bool(&j, "indexed", cfg->indexed);
        bench_loop_report(&loop, &j);
        bench_json_end(&j);
    }
    bench_config = NULL;

    bench_loop_destroy(&loop);
    teardown(s);
    free(s);
    bench_vk_destroy(&vk);
    return bench_exit_code();
}
