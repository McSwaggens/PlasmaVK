/* Ray queries: a compute dispatch that traces rays through an acceleration
 * structure of a terrain, as a lighting bake does on the GPU.
 *
 * The terrain is a height field of 768 x 768 squares, two triangles each
 * (1 179 648 triangles, 32-bit indices), in one bottom level with one
 * instance. A frame is one dispatch of 4 194 304 rays, one a thread, each
 * from a vertex of the terrain toward a low sun, the direction a little
 * different for every ray. The shader is Slang, compiled by slangc
 * (bench_shaders.h).
 *
 *   closest_4m   the closest hit: its distance, triangle and barycentrics
 *   shadow_4m    whether anything is hit (the query ends at its first hit),
 *                the bake's shadow ray
 *
 * The first frame of each configuration is read back, and a sample of its
 * rays compared with the same rays traced on the CPU against every
 * triangle; rays that pass too close to an edge to call are left out.
 * Reported: the acceleration structures' build time (once, with its
 * submission and wait), their sizes, and mrays_per_s from the frames' GPU
 * time. Skipped where the device has no VK_KHR_ray_query.
 */
#include "bench_vk.h"
#include "bench_shaders.h"

#include <math.h>

#define GRID 768u
#define VERTICES ((GRID + 1) * (GRID + 1))
#define TRIANGLES (2 * GRID * GRID)
#define RAYS (1u << 22)
#define GROUP 64u
#define SAMPLE 96u /* rays the CPU traces */

struct config {
    const char *name;
    uint32_t mode; /* terrain_rays.slang's */
    int frames;
};

static const struct config configs[] = {
    { "closest_4m", 0, 30 },
    { "shadow_4m", 1, 30 },
};

/* terrain_rays.slang's push constants and results */
struct push {
    uint32_t count, vertices, mode;
    float reach;
    float sun[3];
    float pad;
};

struct hit {
    float t;
    uint32_t primitive;
    float barycentrics[2];
};

struct scene {
    struct bench_vk *vk;
    const struct config *cfg;
    float (*vertices)[4];
    uint32_t *indices;
    struct bench_buffer vertex_buffer, index_buffer, instance_buffer, scratch, bottom_storage, top_storage;
    struct bench_buffer hits, readback;
    VkAccelerationStructureKHR bottom, top;
    VkDeviceSize bottom_size, top_size;
    double build_us;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    VkPipeline pipeline;
    struct push push;
};

static VkDeviceAddress address_of(struct bench_vk *vk, const struct bench_buffer *b)
{
    VkBufferDeviceAddressInfo info = { .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b->buffer };
    return vkd.GetBufferDeviceAddress(vk->device, &info);
}

static void make_terrain(struct scene *s)
{
    uint32_t x, y, n = 0;
    s->vertices = bench_alloc(sizeof(*s->vertices) * VERTICES);
    s->indices = bench_alloc(sizeof(uint32_t) * 3 * TRIANGLES);
    for (y = 0; y <= GRID; y++)
        for (x = 0; x <= GRID; x++) {
            float *v = s->vertices[y * (GRID + 1) + x];
            float fx = (float)x, fy = (float)y;
            v[0] = fx;
            v[1] = fy;
            v[2] = 24.0f * sinf(fx * 0.021f) * cosf(fy * 0.017f) + 9.0f * sinf(fx * 0.083f + fy * 0.061f) +
                   2.5f * cosf(fx * 0.31f - fy * 0.27f);
            v[3] = 1.0f;
        }
    for (y = 0; y < GRID; y++)
        for (x = 0; x < GRID; x++) {
            uint32_t v00 = y * (GRID + 1) + x, v10 = v00 + 1, v01 = v00 + GRID + 1, v11 = v01 + 1;
            s->indices[n++] = v00;
            s->indices[n++] = v10;
            s->indices[n++] = v01;
            s->indices[n++] = v10;
            s->indices[n++] = v11;
            s->indices[n++] = v01;
        }
}

static void setup(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    const VkBufferUsageFlags input = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkAccelerationStructureGeometryKHR triangles = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };
    VkAccelerationStructureGeometryKHR instances = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
    };
    VkAccelerationStructureBuildGeometryInfoKHR bottom_info = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .geometryCount = 1,
        .pGeometries = &triangles,
    };
    VkAccelerationStructureBuildGeometryInfoKHR top_info = bottom_info;
    VkAccelerationStructureBuildSizesInfoKHR bottom_sizes = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
    VkAccelerationStructureBuildSizesInfoKHR top_sizes = bottom_sizes;
    VkAccelerationStructureCreateInfoKHR create = { .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
    VkAccelerationStructureDeviceAddressInfoKHR address = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
    };
    VkAccelerationStructureInstanceKHR instance = {
        .transform = { { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 } } },
        .mask = 0xff,
        .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
    };
    VkAccelerationStructureBuildRangeInfoKHR bottom_range = { .primitiveCount = TRIANGLES };
    VkAccelerationStructureBuildRangeInfoKHR top_range = { .primitiveCount = 1 };
    const VkAccelerationStructureBuildRangeInfoKHR *ranges;
    VkDescriptorSetLayoutBinding bindings[3] = {
        { 0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
    };
    VkDescriptorPoolSize pool_sizes[2] = {
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
    };
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes,
    };
    VkDescriptorSetAllocateInfo set_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorSetCount = 1,
    };
    VkWriteDescriptorSetAccelerationStructureKHR accel_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
    };
    VkDescriptorBufferInfo buffers[2];
    VkWriteDescriptorSet writes[3];
    VkShaderModule module;
    VkCommandBuffer cmd;
    uint32_t count, i;
    double start;

    make_terrain(s);
    bench_buffer_create(vk, &s->vertex_buffer, sizeof(*s->vertices) * VERTICES, input | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        0, BENCH_DEVICE_LOCAL);
    bench_buffer_create(vk, &s->index_buffer, sizeof(uint32_t) * 3 * TRIANGLES, input, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_create(vk, &s->instance_buffer, sizeof(instance), input, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->vertex_buffer, 0, s->vertices, s->vertex_buffer.size);
    bench_buffer_upload(vk, &s->index_buffer, 0, s->indices, s->index_buffer.size);

    triangles.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.geometry.triangles.vertexData.deviceAddress = address_of(vk, &s->vertex_buffer);
    triangles.geometry.triangles.vertexStride = sizeof(*s->vertices);
    triangles.geometry.triangles.maxVertex = VERTICES - 1;
    triangles.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.geometry.triangles.indexData.deviceAddress = address_of(vk, &s->index_buffer);
    instances.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.geometry.instances.data.deviceAddress = address_of(vk, &s->instance_buffer);
    top_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    top_info.pGeometries = &instances;
    count = TRIANGLES;
    vkd.GetAccelerationStructureBuildSizesKHR(vk->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bottom_info,
                                              &count, &bottom_sizes);
    count = 1;
    vkd.GetAccelerationStructureBuildSizesKHR(vk->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &top_info,
                                              &count, &top_sizes);
    s->bottom_size = bottom_sizes.accelerationStructureSize;
    s->top_size = top_sizes.accelerationStructureSize;
    bench_buffer_create(vk, &s->bottom_storage, s->bottom_size, storage, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_create(vk, &s->top_storage, s->top_size, storage, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_create(vk, &s->scratch, bottom_sizes.buildScratchSize + top_sizes.buildScratchSize + 512,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0,
                        BENCH_DEVICE_LOCAL);
    create.buffer = s->bottom_storage.buffer;
    create.size = s->bottom_size;
    create.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    BENCH_VK(vkd.CreateAccelerationStructureKHR(vk->device, &create, NULL, &s->bottom));
    create.buffer = s->top_storage.buffer;
    create.size = s->top_size;
    create.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    BENCH_VK(vkd.CreateAccelerationStructureKHR(vk->device, &create, NULL, &s->top));
    address.accelerationStructure = s->bottom;
    instance.accelerationStructureReference = vkd.GetAccelerationStructureDeviceAddressKHR(vk->device, &address);
    bench_buffer_upload(vk, &s->instance_buffer, 0, &instance, sizeof(instance));

    /* The builds: the bottom level, then the top level of its instance. */
    bottom_info.dstAccelerationStructure = s->bottom;
    bottom_info.scratchData.deviceAddress = address_of(vk, &s->scratch);
    top_info.dstAccelerationStructure = s->top;
    top_info.scratchData.deviceAddress = bottom_info.scratchData.deviceAddress +
                                         ((bottom_sizes.buildScratchSize + 255) & ~(VkDeviceSize)255);
    start = bench_now_us();
    cmd = bench_cmd_begin(vk);
    ranges = &bottom_range;
    vkd.CmdBuildAccelerationStructuresKHR(cmd, 1, &bottom_info, &ranges);
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                         VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    ranges = &top_range;
    vkd.CmdBuildAccelerationStructuresKHR(cmd, 1, &top_info, &ranges);
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    bench_cmd_finish(vk, cmd);
    s->build_us = bench_now_us() - start;

    /* Results, descriptors and the pipeline. */
    bench_buffer_create(vk, &s->hits, sizeof(struct hit) * RAYS,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_create(vk, &s->readback, sizeof(struct hit) * RAYS, VK_BUFFER_USAGE_TRANSFER_DST_BIT, BENCH_HOST,
                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    s->set_layout = bench_set_layout(vk, 3, bindings, 0);
    s->layout = bench_pipeline_layout(vk, 1, &s->set_layout, sizeof(struct push), VK_SHADER_STAGE_COMPUTE_BIT);
    BENCH_VK(vkd.CreateDescriptorPool(vk->device, &pool_info, NULL, &s->pool));
    set_alloc.descriptorPool = s->pool;
    set_alloc.pSetLayouts = &s->set_layout;
    BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &set_alloc, &s->set));
    accel_write.pAccelerationStructures = &s->top;
    buffers[0] = (VkDescriptorBufferInfo){ s->vertex_buffer.buffer, 0, VK_WHOLE_SIZE };
    buffers[1] = (VkDescriptorBufferInfo){ s->hits.buffer, 0, VK_WHOLE_SIZE };
    memset(writes, 0, sizeof(writes));
    for (i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = s->set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        if (i)
            writes[i].pBufferInfo = &buffers[i - 1];
        else
            writes[i].pNext = &accel_write;
    }
    vkd.UpdateDescriptorSets(vk->device, 3, writes, 0, NULL);
    module = BENCH_SHADER(vk, terrain_rays_slang);
    s->pipeline = bench_compute_pipeline(vk, module, NULL, s->layout);
    vkd.DestroyShaderModule(vk->device, module, NULL);

    s->push.count = RAYS;
    s->push.vertices = VERTICES;
    s->push.reach = 4096.0f;
    s->push.sun[0] = 0.75f;
    s->push.sun[1] = 0.40625f;
    s->push.sun[2] = 0.28125f;
}

static void record(void *user, const struct bench_frame *f)
{
    struct scene *s = user;
    VkCommandBuffer cmd = f->cmd;

    /* Earlier frames still writing the hits come first. */
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    s->push.mode = s->cfg->mode;
    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline);
    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->layout, 0, 1, &s->set, 0, NULL);
    vkd.CmdPushConstants(cmd, s->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(s->push), &s->push);
    vkd.CmdDispatch(cmd, RAYS / GROUP, 1, 1);
    if (f->verify) {
        VkBufferCopy region = { 0, 0, s->hits.size };
        bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        vkd.CmdCopyBuffer(cmd, s->hits.buffer, s->readback.buffer, 1, &region);
    }
}

/* Ray i as terrain_rays.slang makes it, in the same float arithmetic. */
static void make_ray(const struct scene *s, uint32_t i, float origin[3], float direction[3])
{
    uint32_t h = i * 2654435761u;
    const float *v = s->vertices[(h >> 4) % VERTICES];
    origin[0] = v[0];
    origin[1] = v[1];
    origin[2] = v[2] + 0.015625f;
    direction[0] = s->push.sun[0] + ((float)(h >> 24) - 127.5f) / 1024.0f;
    direction[1] = s->push.sun[1] + ((float)((h >> 16) & 255u) - 127.5f) / 1024.0f;
    direction[2] = s->push.sun[2] + 0.0f;
}

/* The ray against every triangle: its closest hit in *want (t -1: none),
 * the hits in *hits. Returns 0 when the ray passes too close to an edge of
 * a triangle it reaches, or its two closest hits are too close, to call. */
static int cpu_trace(const struct scene *s, const float origin[3], const float direction[3], struct hit *want,
                     uint32_t *hits)
{
    double o[3] = { origin[0], origin[1], origin[2] }, d[3] = { direction[0], direction[1], direction[2] };
    double best = s->push.reach, second = 1e30;
    uint32_t p;
    int sure = 1, c;

    want->t = -1.0f;
    want->primitive = 0;
    want->barycentrics[0] = want->barycentrics[1] = 0.0f;
    *hits = 0;
    for (p = 0; p < TRIANGLES; p++) {
        const float *v0 = s->vertices[s->indices[3 * p]], *v1 = s->vertices[s->indices[3 * p + 1]];
        const float *v2 = s->vertices[s->indices[3 * p + 2]];
        double e1[3], e2[3], pv[3], tv[3], qv[3], det, u, w, t;
        /* (the triangle lies within one square: most are nowhere near the ray) */
        if ((d[0] > 0 ? v0[0] + 2 < o[0] : v0[0] - 2 > o[0]) || (d[1] > 0 ? v0[1] + 2 < o[1] : v0[1] - 2 > o[1]))
            continue;
        for (c = 0; c < 3; c++) {
            e1[c] = (double)v1[c] - v0[c];
            e2[c] = (double)v2[c] - v0[c];
            tv[c] = o[c] - v0[c];
        }
        pv[0] = d[1] * e2[2] - d[2] * e2[1];
        pv[1] = d[2] * e2[0] - d[0] * e2[2];
        pv[2] = d[0] * e2[1] - d[1] * e2[0];
        det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
        if (fabs(det) < 1e-12)
            continue;
        u = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) / det;
        qv[0] = tv[1] * e1[2] - tv[2] * e1[1];
        qv[1] = tv[2] * e1[0] - tv[0] * e1[2];
        qv[2] = tv[0] * e1[1] - tv[1] * e1[0];
        w = (d[0] * qv[0] + d[1] * qv[1] + d[2] * qv[2]) / det;
        t = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) / det;
        if (t <= 1e-6 || t >= s->push.reach || u < -1e-3 || w < -1e-3 || u + w > 1 + 1e-3)
            continue;
        if (fabs(u) < 1e-3 || fabs(w) < 1e-3 || fabs(1 - u - w) < 1e-3) {
            sure = 0;
            continue;
        }
        ++*hits;
        if (t < best) {
            second = best;
            best = t;
            want->t = (float)t;
            want->primitive = p;
            want->barycentrics[0] = (float)u;
            want->barycentrics[1] = (float)w;
        } else if (t < second) {
            second = t;
        }
    }
    if (second - best < 1e-3 * (1 + best))
        sure = 0;
    return sure;
}

static int verify(struct scene *s, uint32_t *compared, uint32_t *hit_rays)
{
    const struct hit *got = s->readback.map;
    uint32_t k, wrong = 0;

    *compared = *hit_rays = 0;
    for (k = 0; k < SAMPLE; k++) {
        uint32_t i = (uint32_t)(((uint64_t)k * 2654435761u + 12345u) % RAYS), hits;
        float origin[3], direction[3];
        struct hit want;
        int same;
        make_ray(s, i, origin, direction);
        if (!cpu_trace(s, origin, direction, &want, &hits))
            continue;
        ++*compared;
        *hit_rays += hits != 0;
        if (s->cfg->mode == 1)
            same = (got[i].t > 0.0f) == (hits != 0);
        else
            same = (want.t < 0.0f) == (got[i].t < 0.0f) &&
                   (want.t < 0.0f ||
                    (fabsf(got[i].t - want.t) <= 1e-3f * (1.0f + want.t) && got[i].primitive == want.primitive &&
                     fabsf(got[i].barycentrics[0] - want.barycentrics[0]) < 5e-3f &&
                     fabsf(got[i].barycentrics[1] - want.barycentrics[1]) < 5e-3f));
        if (!same && !wrong++)
            bench_error(s->cfg->name, "ray %u: t %g primitive %u barycentrics %g %g, the CPU finds t %g primitive %u "
                                      "barycentrics %g %g (%u hits)",
                        i, (double)got[i].t, got[i].primitive, (double)got[i].barycentrics[0],
                        (double)got[i].barycentrics[1], (double)want.t, want.primitive, (double)want.barycentrics[0],
                        (double)want.barycentrics[1], hits);
    }
    if (!wrong && (*compared < SAMPLE / 2 || !*hit_rays || *hit_rays == *compared)) {
        bench_error(s->cfg->name, "the sample does not test the trace: %u of %u rays compared, %u of them hit",
                    *compared, SAMPLE, *hit_rays);
        return 0;
    }
    return !wrong;
}

static void teardown(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDevice dev = vk->device;
    vkd.DeviceWaitIdle(dev);
    vkd.DestroyPipeline(dev, s->pipeline, NULL);
    vkd.DestroyDescriptorPool(dev, s->pool, NULL);
    vkd.DestroyPipelineLayout(dev, s->layout, NULL);
    vkd.DestroyDescriptorSetLayout(dev, s->set_layout, NULL);
    vkd.DestroyAccelerationStructureKHR(dev, s->top, NULL);
    vkd.DestroyAccelerationStructureKHR(dev, s->bottom, NULL);
    bench_buffer_destroy(vk, &s->readback);
    bench_buffer_destroy(vk, &s->hits);
    bench_buffer_destroy(vk, &s->scratch);
    bench_buffer_destroy(vk, &s->top_storage);
    bench_buffer_destroy(vk, &s->bottom_storage);
    bench_buffer_destroy(vk, &s->instance_buffer);
    bench_buffer_destroy(vk, &s->index_buffer);
    bench_buffer_destroy(vk, &s->vertex_buffer);
    free(s->vertices);
    free(s->indices);
}

int main(void)
{
    struct bench_vk vk;
    struct bench_loop loop;
    struct scene *s;
    size_t c;

    bench_begin("ray_query");
    bench_vk_init(&vk, BENCH_NEED_SYNC2 | BENCH_WANT_RAY_QUERY);
    if (!vk.ray_query) {
        bench_skip(NULL, "no VK_KHR_acceleration_structure and VK_KHR_ray_query");
        bench_vk_destroy(&vk);
        return bench_exit_code();
    }
    s = bench_alloc(sizeof(*s));
    s->vk = &vk;
    setup(s);
    bench_loop_init(&loop, &vk, 1000);

    for (c = 0; c < sizeof(configs) / sizeof(configs[0]); c++) {
        const struct config *cfg = &configs[c];
        int frames = bench_frames(cfg->frames);
        struct bench_json j;
        uint32_t compared, hit_rays;
        double gpu_median;

        if (!bench_selected(cfg->name))
            continue;
        bench_config = cfg->name;
        s->cfg = cfg;
        bench_loop_verify(&loop, record, s);
        if (!verify(s, &compared, &hit_rays))
            continue;
        bench_loop_run(&loop, record, s, 3, frames < 1000 ? frames : 1000);
        gpu_median = bench_loop_gpu_median(&loop);

        bench_json_begin(&j, cfg->name);
        bench_json_int(&j, "triangles", TRIANGLES);
        bench_json_int(&j, "rays", RAYS);
        bench_json_int(&j, "rays_compared", compared);
        bench_json_int(&j, "rays_compared_hit", hit_rays);
        bench_json_num(&j, "build_ms", s->build_us / 1000.0);
        bench_json_int(&j, "bottom_level_bytes", (long long)s->bottom_size);
        bench_json_int(&j, "top_level_bytes", (long long)s->top_size);
        bench_loop_report(&loop, &j);
        if (gpu_median > 0.0)
            bench_json_num(&j, "mrays_per_s", RAYS / gpu_median);
        bench_json_end(&j);
    }
    bench_config = NULL;

    bench_loop_destroy(&loop);
    teardown(s);
    free(s);
    bench_vk_destroy(&vk);
    return bench_exit_code();
}
