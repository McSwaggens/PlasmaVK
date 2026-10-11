/* Compute: many small dispatches, and a large multi-pass prefix sum.
 *
 *   chain_N        N dependent dispatches of 4096 elements (16 workgroups)
 *                  with a pipeline barrier between each - a chain of
 *                  simulation passes; measures dispatch + barrier cost
 *   independent_N  N dispatches on disjoint slices with no barriers between
 *                  them; dispatch cost alone
 *   scan_N         inclusive prefix sum of N uint32s: block scans in shared
 *                  memory, a scan of the block sums per level, then adds
 *                  back down, with barriers between passes; bandwidth-bound
 *
 * The first frame of each configuration starts from known data, is read back
 * and compared with the same computation on the CPU.
 */
#include "bench_vk.h"
#include "bench_shaders.h"

#define SLICE 4096           /* elements per small dispatch */
#define GROUP 256            /* step.comp workgroup size */
#define BLOCK 1024           /* elements per scan workgroup */
#define MAX_DISPATCHES 1000
#define MAX_SCAN (1u << 24)
#define MAX_LEVELS 4

enum mode { MODE_CHAIN, MODE_INDEPENDENT, MODE_SCAN };

struct config {
    const char *name;
    enum mode mode;
    uint32_t size; /* dispatches, or scan elements */
    int frames;
};

static const struct config configs[] = {
    { "chain_100", MODE_CHAIN, 100, 300 },
    { "chain_1000", MODE_CHAIN, 1000, 100 },
    { "independent_1000", MODE_INDEPENDENT, 1000, 100 },
    { "scan_1m", MODE_SCAN, 1u << 20, 200 },
    { "scan_16m", MODE_SCAN, 1u << 24, 100 },
};

struct step_push {
    uint32_t offset, count, step;
};

struct scan_push {
    uint32_t in_offset, out_offset, sums_offset, count, from_src;
};

struct add_push {
    uint32_t data_offset, sums_offset, count;
};

struct level {
    uint32_t offset, count;
};

struct scene {
    struct bench_vk *vk;
    const struct config *cfg;
    struct bench_buffer data, data_init, src, work, readback;
    VkDescriptorSetLayout step_set_layout, scan_set_layout;
    VkPipelineLayout step_layout, scan_layout;
    VkPipeline step, scan_blocks, scan_add;
    VkDescriptorPool pool;
    VkDescriptorSet step_set, scan_set;
    struct level levels[MAX_LEVELS];
    int level_count;
    uint32_t scratch; /* sums of the last level (unused) */
    uint32_t *initial; /* initial contents of data, and of src */
    uint32_t *source;
};

static void compute_barrier(VkCommandBuffer cmd)
{
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
}

static uint32_t groups(uint32_t count, uint32_t per_group)
{
    return (count + per_group - 1) / per_group;
}

static void plan_scan(struct scene *s, uint32_t n)
{
    uint32_t offset = 0;
    s->level_count = 0;
    for (;;) {
        s->levels[s->level_count].offset = offset;
        s->levels[s->level_count].count = n;
        s->level_count++;
        offset += n;
        if (n <= BLOCK)
            break;
        n = groups(n, BLOCK);
    }
    s->scratch = offset;
}

static void record(void *user, const struct bench_frame *f)
{
    struct scene *s = user;
    const struct config *cfg = s->cfg;
    VkCommandBuffer cmd = f->cmd;
    VkPipelineBindPoint bp = VK_PIPELINE_BIND_POINT_COMPUTE;
    uint32_t i;
    int l;

    /* Earlier frames still touching the buffers come first. */
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                             VK_ACCESS_2_TRANSFER_WRITE_BIT);

    if (cfg->mode == MODE_SCAN) {
        vkd.CmdBindDescriptorSets(cmd, bp, s->scan_layout, 0, 1, &s->scan_set, 0, NULL);
        vkd.CmdBindPipeline(cmd, bp, s->scan_blocks);
        for (l = 0; l < s->level_count; l++) {
            struct scan_push p;
            p.in_offset = s->levels[l].offset;
            p.out_offset = s->levels[l].offset;
            p.sums_offset = l + 1 < s->level_count ? s->levels[l + 1].offset : s->scratch;
            p.count = s->levels[l].count;
            p.from_src = l == 0;
            vkd.CmdPushConstants(cmd, s->scan_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
            vkd.CmdDispatch(cmd, groups(p.count, BLOCK), 1, 1);
            compute_barrier(cmd);
        }
        vkd.CmdBindPipeline(cmd, bp, s->scan_add);
        for (l = s->level_count - 2; l >= 0; l--) {
            struct add_push p;
            p.data_offset = s->levels[l].offset;
            p.sums_offset = s->levels[l + 1].offset;
            p.count = s->levels[l].count;
            vkd.CmdPushConstants(cmd, s->scan_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
            vkd.CmdDispatch(cmd, groups(p.count, BLOCK) - 1, 1, 1);
            compute_barrier(cmd);
        }
        if (f->verify) {
            VkBufferCopy region = { 0, 0, (VkDeviceSize)cfg->size * 4 };
            bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            vkd.CmdCopyBuffer(cmd, s->work.buffer, s->readback.buffer, 1, &region);
        }
        return;
    }

    if (f->verify) {
        VkBufferCopy region = { 0, 0, s->data.size };
        vkd.CmdCopyBuffer(cmd, s->data_init.buffer, s->data.buffer, 1, &region);
        bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
    vkd.CmdBindDescriptorSets(cmd, bp, s->step_layout, 0, 1, &s->step_set, 0, NULL);
    vkd.CmdBindPipeline(cmd, bp, s->step);
    for (i = 0; i < cfg->size; i++) {
        struct step_push p;
        p.offset = cfg->mode == MODE_CHAIN ? 0 : i * SLICE;
        p.count = SLICE;
        p.step = i;
        vkd.CmdPushConstants(cmd, s->step_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        vkd.CmdDispatch(cmd, SLICE / GROUP, 1, 1);
        if (cfg->mode == MODE_CHAIN)
            compute_barrier(cmd);
    }
    if (f->verify) {
        VkBufferCopy region = { 0, 0, s->data.size };
        bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        vkd.CmdCopyBuffer(cmd, s->data.buffer, s->readback.buffer, 1, &region);
    }
}

static int verify(struct scene *s)
{
    const struct config *cfg = s->cfg;
    const uint32_t *got = s->readback.map;
    uint32_t n = cfg->mode == MODE_SCAN ? cfg->size : cfg->mode == MODE_CHAIN ? SLICE : cfg->size * SLICE;
    uint32_t *want = bench_alloc(sizeof(uint32_t) * n);
    uint32_t i, k;
    int ok = 1;

    if (cfg->mode == MODE_SCAN) {
        uint32_t sum = 0;
        for (i = 0; i < n; i++)
            want[i] = sum += s->source[i];
    } else {
        memcpy(want, s->initial, sizeof(uint32_t) * n);
        for (k = 0; k < cfg->size; k++) {
            uint32_t base = cfg->mode == MODE_CHAIN ? 0 : k * SLICE;
            for (i = 0; i < SLICE; i++)
                want[base + i] = want[base + i] * 1664525u + 1013904223u + k;
        }
    }
    for (i = 0; i < n; i++) {
        if (got[i] != want[i]) {
            bench_error(cfg->name, "element %u: got %u, want %u", i, got[i], want[i]);
            ok = 0;
            break;
        }
    }
    free(want);
    return ok;
}

static void setup(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkDescriptorSetLayoutBinding bindings[2] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
    };
    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    VkDescriptorBufferInfo infos[3];
    VkWriteDescriptorSet writes[3];
    VkShaderModule module;
    uint64_t seed = 0xC0FFEE11ull;
    VkDeviceSize data_size = (VkDeviceSize)MAX_DISPATCHES * SLICE * 4, work_size;
    uint32_t i;

    /* Small-dispatch data, with a copy of its initial contents. */
    s->initial = bench_alloc((size_t)data_size);
    for (i = 0; i < MAX_DISPATCHES * SLICE; i++)
        s->initial[i] = bench_rand(&seed);
    bench_buffer_create(vk, &s->data, data_size, storage, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_create(vk, &s->data_init, data_size, storage, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->data_init, 0, s->initial, data_size);

    /* Scan input; work holds every level plus a scratch slot. */
    s->source = bench_alloc(sizeof(uint32_t) * MAX_SCAN);
    for (i = 0; i < MAX_SCAN; i++)
        s->source[i] = bench_rand(&seed) & 0xFFFF;
    bench_buffer_create(vk, &s->src, (VkDeviceSize)MAX_SCAN * 4, storage, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->src, 0, s->source, (VkDeviceSize)MAX_SCAN * 4);
    plan_scan(s, MAX_SCAN);
    work_size = ((VkDeviceSize)s->scratch + 1) * 4;
    bench_buffer_create(vk, &s->work, work_size, storage, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_create(vk, &s->readback, data_size > work_size ? data_size : work_size,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT, BENCH_HOST, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

    /* Descriptors. */
    s->step_set_layout = bench_set_layout(vk, 1, bindings, 0);
    s->scan_set_layout = bench_set_layout(vk, 2, bindings, 0);
    s->step_layout = bench_pipeline_layout(vk, 1, &s->step_set_layout, sizeof(struct step_push),
                                           VK_SHADER_STAGE_COMPUTE_BIT);
    s->scan_layout = bench_pipeline_layout(vk, 1, &s->scan_set_layout, sizeof(struct scan_push),
                                           VK_SHADER_STAGE_COMPUTE_BIT);
    BENCH_VK(vkd.CreateDescriptorPool(vk->device, &pool_info, NULL, &s->pool));
    {
        VkDescriptorSetLayout layouts[2] = { s->step_set_layout, s->scan_set_layout };
        VkDescriptorSet sets[2];
        VkDescriptorSetAllocateInfo alloc = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = s->pool,
            .descriptorSetCount = 2,
            .pSetLayouts = layouts,
        };
        BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &alloc, sets));
        s->step_set = sets[0];
        s->scan_set = sets[1];
    }
    infos[0] = (VkDescriptorBufferInfo){ s->data.buffer, 0, VK_WHOLE_SIZE };
    infos[1] = (VkDescriptorBufferInfo){ s->src.buffer, 0, VK_WHOLE_SIZE };
    infos[2] = (VkDescriptorBufferInfo){ s->work.buffer, 0, VK_WHOLE_SIZE };
    for (i = 0; i < 3; i++) {
        memset(&writes[i], 0, sizeof(writes[i]));
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = i == 0 ? s->step_set : s->scan_set;
        writes[i].dstBinding = i == 2 ? 1 : 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkd.UpdateDescriptorSets(vk->device, 3, writes, 0, NULL);

    /* Pipelines. */
    module = BENCH_SHADER(vk, step_comp);
    s->step = bench_compute_pipeline(vk, module, NULL, s->step_layout);
    vkd.DestroyShaderModule(vk->device, module, NULL);
    module = BENCH_SHADER(vk, scan_blocks_comp);
    s->scan_blocks = bench_compute_pipeline(vk, module, NULL, s->scan_layout);
    vkd.DestroyShaderModule(vk->device, module, NULL);
    module = BENCH_SHADER(vk, scan_add_comp);
    s->scan_add = bench_compute_pipeline(vk, module, NULL, s->scan_layout);
    vkd.DestroyShaderModule(vk->device, module, NULL);
}

static void teardown(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDevice dev = vk->device;
    vkd.DeviceWaitIdle(dev);
    vkd.DestroyPipeline(dev, s->step, NULL);
    vkd.DestroyPipeline(dev, s->scan_blocks, NULL);
    vkd.DestroyPipeline(dev, s->scan_add, NULL);
    vkd.DestroyDescriptorPool(dev, s->pool, NULL);
    vkd.DestroyPipelineLayout(dev, s->step_layout, NULL);
    vkd.DestroyPipelineLayout(dev, s->scan_layout, NULL);
    vkd.DestroyDescriptorSetLayout(dev, s->step_set_layout, NULL);
    vkd.DestroyDescriptorSetLayout(dev, s->scan_set_layout, NULL);
    bench_buffer_destroy(vk, &s->data);
    bench_buffer_destroy(vk, &s->data_init);
    bench_buffer_destroy(vk, &s->src);
    bench_buffer_destroy(vk, &s->work);
    bench_buffer_destroy(vk, &s->readback);
    free(s->initial);
    free(s->source);
}

int main(void)
{
    struct bench_vk vk;
    struct bench_loop loop;
    struct scene *s;
    size_t c;

    bench_begin("compute");
    bench_vk_init(&vk, BENCH_NEED_SYNC2);
    s = bench_alloc(sizeof(*s));
    s->vk = &vk;
    setup(s);
    bench_loop_init(&loop, &vk, 1000);

    for (c = 0; c < sizeof(configs) / sizeof(configs[0]); c++) {
        const struct config *cfg = &configs[c];
        int frames = bench_frames(cfg->frames);
        struct bench_json j;
        double cpu_median, gpu_median;
        int dispatches, barriers;

        if (!bench_selected(cfg->name))
            continue;
        bench_config = cfg->name;
        s->cfg = cfg;
        if (cfg->mode == MODE_SCAN)
            plan_scan(s, cfg->size);
        bench_loop_verify(&loop, record, s);
        if (!verify(s))
            continue;
        bench_loop_run(&loop, record, s, 10, frames < 1000 ? frames : 1000);
        cpu_median = bench_loop_cpu_median(&loop);
        gpu_median = bench_loop_gpu_median(&loop);

        if (cfg->mode == MODE_SCAN) {
            dispatches = 2 * s->level_count - 1;
            barriers = dispatches;
        } else {
            dispatches = (int)cfg->size;
            barriers = cfg->mode == MODE_CHAIN ? dispatches : 0;
        }
        bench_json_begin(&j, cfg->name);
        if (cfg->mode == MODE_SCAN)
            bench_json_int(&j, "elements", cfg->size);
        else
            bench_json_int(&j, "elements_per_dispatch", SLICE);
        bench_json_int(&j, "dispatches", dispatches);
        bench_json_int(&j, "barriers", barriers);
        bench_loop_report(&loop, &j);
        if (cfg->mode == MODE_SCAN && gpu_median > 0.0)
            bench_json_num(&j, "gelements_per_s", cfg->size / gpu_median / 1000.0);
        else
            bench_json_num(&j, "cpu_ns_per_dispatch", cpu_median * 1000.0 / dispatches);
        bench_json_end(&j);
    }
    bench_config = NULL;

    bench_loop_destroy(&loop);
    teardown(s);
    free(s);
    bench_vk_destroy(&vk);
    return bench_exit_code();
}
