/* Query pools (section 14): timestamps, resets, results with every flag,
 * availability, waiting, and copies of results to buffers. */
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <mach/mach_time.h>
#include <pthread.h>
#include <string.h>
#include <time.h>

#include "../pvk_test.h"

static VkInstance instance;
static VkPhysicalDevice pd;
static VkDevice device;
static VkQueue queue, compute_queue;
static VkCommandPool pool, compute_pool;
static uint32_t shared_type = UINT32_MAX, private_type = UINT32_MAX;
static double period; /* ns per tick */
static uint32_t valid_bits;

struct buf {
    VkBuffer buffer;
    VkDeviceMemory memory;
    unsigned char *data;
    VkDeviceSize size;
};

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

static void setup(void)
{
    VkPhysicalDeviceMemoryProperties mem;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    VkPhysicalDeviceProperties props;
    VkQueueFamilyProperties families[2];
    VkInstanceCreateInfo instance_info;
    VkApplicationInfo app;
    VkDeviceQueueCreateInfo queue_info[2];
    VkDeviceCreateInfo device_info;
    VkCommandPoolCreateInfo pool_info;
    float priority = 1.0f;
    uint32_t count = 1, i;

    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_4;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    REQUIRE_VK(vkCreateInstance(&instance_info, NULL, &instance));
    REQUIRE_VK(vkEnumeratePhysicalDevices(instance, &count, &pd));
    vkGetPhysicalDeviceProperties(pd, &props);
    period = props.limits.timestampPeriod;
    count = 2;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, families);
    REQUIRE(count == 2);
    valid_bits = families[0].timestampValidBits;
    CHECK(valid_bits == 64 && families[1].timestampValidBits == 64);
    CHECK(props.limits.timestampComputeAndGraphics);
    /* The GPU's timestamps are mach_absolute_time ticks (section 14). */
    {
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        CHECK(period > 0.99 * tb.numer / tb.denom && period < 1.01 * tb.numer / tb.denom);
    }

    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.hostQueryReset = VK_TRUE;
    v12.timelineSemaphore = VK_TRUE;
    memset(&v13, 0, sizeof(v13));
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13.pNext = &v12;
    v13.synchronization2 = VK_TRUE;
    memset(queue_info, 0, sizeof(queue_info));
    for (i = 0; i < 2; i++) {
        queue_info[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info[i].queueFamilyIndex = i;
        queue_info[i].queueCount = 1;
        queue_info[i].pQueuePriorities = &priority;
    }
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &v13;
    device_info.queueCreateInfoCount = 2;
    device_info.pQueueCreateInfos = queue_info;
    REQUIRE_VK(vkCreateDevice(pd, &device_info, NULL, &device));
    vkGetDeviceQueue(device, 0, 0, &queue);
    vkGetDeviceQueue(device, 1, 0, &compute_queue);
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    for (i = 0; i < mem.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mem.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) && shared_type == UINT32_MAX)
            shared_type = i;
        if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && !(f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) &&
            private_type == UINT32_MAX)
            private_type = i;
    }
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    REQUIRE_VK(vkCreateCommandPool(device, &pool_info, NULL, &pool));
    pool_info.queueFamilyIndex = 1;
    REQUIRE_VK(vkCreateCommandPool(device, &pool_info, NULL, &compute_pool));
}

static void teardown(void)
{
    vkDestroyCommandPool(device, pool, NULL);
    vkDestroyCommandPool(device, compute_pool, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
}

static struct buf make_buf(VkDeviceSize size, uint32_t type)
{
    VkBufferCreateInfo info;
    VkMemoryAllocateInfo alloc;
    VkMemoryRequirements req;
    struct buf b;
    void *p = NULL;
    memset(&b, 0, sizeof(b));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vkCreateBuffer(device, &info, NULL, &b.buffer);
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    vkAllocateMemory(device, &alloc, NULL, &b.memory);
    vkBindBufferMemory(device, b.buffer, b.memory, 0);
    if (type == shared_type && vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS)
        b.data = p;
    b.size = size;
    return b;
}

static void free_buf(struct buf *b)
{
    vkDestroyBuffer(device, b->buffer, NULL);
    vkFreeMemory(device, b->memory, NULL);
}

static VkQueryPool make_pool(VkQueryType type, uint32_t count)
{
    VkQueryPoolCreateInfo info;
    VkQueryPool p = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = type;
    info.queryCount = count;
    CHECK_VK(vkCreateQueryPool(device, &info, NULL, &p), VK_SUCCESS);
    return p;
}

static VkCommandBuffer begin_in(VkCommandPool from, VkCommandBufferLevel level)
{
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferInheritanceInfo inherit;
    VkCommandBufferBeginInfo info;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = from;
    alloc.level = level;
    alloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &alloc, &cb);
    memset(&inherit, 0, sizeof(inherit));
    inherit.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.pInheritanceInfo = level == VK_COMMAND_BUFFER_LEVEL_SECONDARY ? &inherit : NULL;
    vkBeginCommandBuffer(cb, &info);
    return cb;
}

static VkCommandBuffer begin(void)
{
    return begin_in(pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
}

static void host_barrier(VkCommandBuffer cb)
{
    VkMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &b, 0, NULL, 0, NULL);
}

static void transfer_barrier(VkCommandBuffer cb)
{
    VkMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, NULL, 0, NULL);
}

/* Ends, submits on q (waiting for `wait` reaching `value` if set) and waits
 * unless no fence is wanted. */
static VkResult submit_on(VkQueue q, VkCommandBuffer cb, VkSemaphore wait, uint64_t value, VkFence fence)
{
    VkTimelineSemaphoreSubmitInfo values;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo info;
    vkEndCommandBuffer(cb);
    memset(&values, 0, sizeof(values));
    values.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    values.waitSemaphoreValueCount = wait ? 1 : 0;
    values.pWaitSemaphoreValues = &value;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.pNext = &values;
    info.waitSemaphoreCount = wait ? 1 : 0;
    info.pWaitSemaphores = &wait;
    info.pWaitDstStageMask = &stage;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &cb;
    return vkQueueSubmit(q, 1, &info, fence);
}

static void run_on(VkQueue q, VkCommandBuffer cb)
{
    CHECK_VK(submit_on(q, cb, VK_NULL_HANDLE, 0, VK_NULL_HANDLE), VK_SUCCESS);
    CHECK_VK(vkQueueWaitIdle(q), VK_SUCCESS);
}

static void run(VkCommandBuffer cb)
{
    run_on(queue, cb);
}

static VkSemaphore make_timeline(void)
{
    VkSemaphoreTypeCreateInfo type;
    VkSemaphoreCreateInfo info;
    VkSemaphore s = VK_NULL_HANDLE;
    memset(&type, 0, sizeof(type));
    type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    info.pNext = &type;
    vkCreateSemaphore(device, &info, NULL, &s);
    return s;
}

/* Copies of `bytes` between two device-local buffers, `n` times, ordered:
 * some GPU work that takes a while. */
static void record_work(VkCommandBuffer cb, const struct buf *a, const struct buf *b, VkDeviceSize bytes, uint32_t n)
{
    VkBufferCopy region;
    uint32_t i;
    memset(&region, 0, sizeof(region));
    region.size = bytes;
    for (i = 0; i < n; i++) {
        vkCmdCopyBuffer(cb, i & 1 ? b->buffer : a->buffer, i & 1 ? a->buffer : b->buffer, 1, &region);
        transfer_barrier(cb);
    }
}

static VkResult get64(VkQueryPool p, uint32_t first, uint32_t count, uint64_t *out, VkQueryResultFlags flags)
{
    VkDeviceSize stride = (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) ? 16 : 8;
    return vkGetQueryPoolResults(device, p, first, count, (size_t)(stride * count), out, stride,
                                 flags | VK_QUERY_RESULT_64_BIT);
}

/* ---- Timestamps ---- */

/* Around GPU work: available, ordered, and within the CPU's view of the
 * submission (the same clock); not later queries. */
static void test_timestamp_basic(void)
{
    VkQueryPool p = make_pool(VK_QUERY_TYPE_TIMESTAMP, 4);
    struct buf a = make_buf(64u << 20, private_type), b = make_buf(64u << 20, private_type);
    VkCommandBuffer cb = begin();
    uint64_t r[8], t0, t1;
    double gpu_ns, cpu_ns;

    vkResetQueryPool(device, p, 0, 4);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, p, 0);
    record_work(cb, &a, &b, 64u << 20, 16);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 1);
    t0 = mach_absolute_time();
    run(cb);
    t1 = mach_absolute_time();
    memset(r, 0xab, sizeof(r));
    CHECK_VK(get64(p, 0, 4, r, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT), VK_NOT_READY);
    CHECK(r[1] == 1 && r[3] == 1 && r[5] == 0 && r[7] == 0);
    CHECK(r[4] == 0xababababababababull && r[6] == 0xababababababababull);
    CHECK(r[0] <= r[2]);
    CHECK(t0 <= r[0] && r[2] <= t1);
    gpu_ns = (double)(r[2] - r[0]) * period;
    cpu_ns = (double)(t1 - t0) * period;
    CHECK(gpu_ns > 0.0 && gpu_ns <= cpu_ns);
    printf("    16 copies of 64 MiB: %.3f ms on the GPU, %.3f ms submit to idle\n", gpu_ns / 1e6, cpu_ns / 1e6);
    CHECK_VK(get64(p, 0, 2, r, VK_QUERY_RESULT_WAIT_BIT), VK_SUCCESS);
    free_buf(&a);
    free_buf(&b);
    vkDestroyQueryPool(device, p, NULL);
}

/* Bottom-of-pipe timestamps never go backwards in submission order: across
 * command buffers of a batch, batches, submissions, secondaries and queues
 * (each queue compared with itself). */
static void test_timestamp_order(void)
{
    VkQueryPool p = make_pool(VK_QUERY_TYPE_TIMESTAMP, 16);
    struct buf a = make_buf(16u << 20, private_type), b = make_buf(16u << 20, private_type);
    VkCommandBuffer cbs[4], sec;
    VkSubmitInfo info[2];
    uint64_t r[16];
    uint32_t i;

    vkResetQueryPool(device, p, 0, 16);
    for (i = 0; i < 4; i++) {
        cbs[i] = begin();
        vkCmdWriteTimestamp2(cbs[i], VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, p, 2 * i);
        record_work(cbs[i], &a, &b, 16u << 20, 2 + i);
        vkCmdWriteTimestamp2(cbs[i], VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, p, 2 * i + 1);
        vkEndCommandBuffer(cbs[i]);
    }
    memset(info, 0, sizeof(info));
    info[0].sType = info[1].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info[0].commandBufferCount = 2; /* two command buffers in one batch */
    info[0].pCommandBuffers = cbs;
    info[1].commandBufferCount = 1; /* a second batch */
    info[1].pCommandBuffers = &cbs[2];
    CHECK_VK(vkQueueSubmit(queue, 2, info, VK_NULL_HANDLE), VK_SUCCESS);
    CHECK_VK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                                                      .pCommandBuffers = &cbs[3] },
                           VK_NULL_HANDLE),
             VK_SUCCESS);
    /* A secondary command buffer's timestamps, then the primary's. */
    sec = begin_in(pool, VK_COMMAND_BUFFER_LEVEL_SECONDARY);
    vkCmdWriteTimestamp2(sec, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, p, 8);
    record_work(sec, &a, &b, 16u << 20, 2);
    vkCmdWriteTimestamp2(sec, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 9);
    vkEndCommandBuffer(sec);
    cbs[0] = begin();
    vkCmdExecuteCommands(cbs[0], 1, &sec);
    vkCmdWriteTimestamp2(cbs[0], VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 10);
    run(cbs[0]);
    /* The compute family's queue. */
    cbs[1] = begin_in(compute_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    vkCmdWriteTimestamp2(cbs[1], VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, p, 11);
    record_work(cbs[1], &a, &b, 16u << 20, 2);
    vkCmdWriteTimestamp2(cbs[1], VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 12);
    run_on(compute_queue, cbs[1]);

    CHECK_VK(get64(p, 0, 13, r, VK_QUERY_RESULT_WAIT_BIT), VK_SUCCESS);
    for (i = 0; i < 4; i++)
        CHECK(r[2 * i] <= r[2 * i + 1]);
    /* each bottom-of-pipe timestamp after everything before it */
    CHECK(r[1] <= r[3] && r[3] <= r[5] && r[5] <= r[7] && r[7] <= r[9] && r[9] <= r[10]);
    CHECK(r[8] <= r[9] && r[11] <= r[12]);
    CHECK(r[11] >= r[10]); /* submitted after the graphics queue went idle */
    vkFreeCommandBuffers(device, pool, 1, &sec);
    free_buf(&a);
    free_buf(&b);
    vkDestroyQueryPool(device, p, NULL);
}

/* A pool much larger than the 16 ranges an encoder collects, written in one
 * command buffer, interleaved with work and out of order. */
static void test_timestamp_many(void)
{
    enum { N = 4096 };
    VkQueryPool p = make_pool(VK_QUERY_TYPE_TIMESTAMP, N);
    static uint64_t r[N];
    struct buf a = make_buf(1u << 20, private_type), b = make_buf(1u << 20, private_type);
    VkCommandBuffer cb = begin();
    uint32_t i, bad = 0;

    vkCmdResetQueryPool(cb, p, 0, N);
    for (i = 0; i < N / 2; i++) { /* even queries in order, odd ones backwards */
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 2 * i);
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, N - 1 - 2 * i);
        if (i % 512 == 0)
            record_work(cb, &a, &b, 1u << 20, 1);
    }
    run(cb);
    CHECK_VK(get64(p, 0, N, r, 0), VK_SUCCESS);
    for (i = 1; i < N / 2; i++)
        if (r[2 * i] < r[N - 1 - 2 * (i - 1)] || r[N - 1 - 2 * i] < r[2 * i])
            bad++;
    CHECK(bad == 0);
    free_buf(&a);
    free_buf(&b);
    vkDestroyQueryPool(device, p, NULL);
}

/* ---- Results ---- */

#define SENTINEL 0xab

/* Half of 8 queries written: every flag combination and stride. */
static void test_results_flags(void)
{
    VkQueryPool p = make_pool(VK_QUERY_TYPE_TIMESTAMP, 8);
    VkCommandBuffer cb = begin();
    uint64_t ts[8], r64[8 * 3];
    uint32_t r32[8 * 3], i;
    unsigned char raw[8 * 24];

    vkResetQueryPool(device, p, 0, 8);
    for (i = 0; i < 8; i += 2)
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, i);
    run(cb);
    for (i = 0; i < 8; i += 2) {
        CHECK_VK(vkGetQueryPoolResults(device, p, i, 1, 8, &ts[i], 0, VK_QUERY_RESULT_64_BIT), VK_SUCCESS);
        CHECK(ts[i] != 0);
    }

    /* 32-bit, no availability: written ones are the low bits, others untouched. */
    memset(r32, SENTINEL, sizeof(r32));
    CHECK_VK(vkGetQueryPoolResults(device, p, 0, 8, sizeof(r32), r32, 4, 0), VK_NOT_READY);
    for (i = 0; i < 8; i++)
        CHECK(i % 2 ? r32[i] == 0xabababab : r32[i] == (uint32_t)ts[i]);
    /* 32-bit with availability, stride 8 and 12. */
    memset(r32, SENTINEL, sizeof(r32));
    CHECK_VK(vkGetQueryPoolResults(device, p, 0, 8, sizeof(r32), r32, 8, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
             VK_NOT_READY);
    for (i = 0; i < 8; i++) {
        CHECK(r32[2 * i + 1] == (i % 2 ? 0u : 1u));
        CHECK(i % 2 ? r32[2 * i] == 0xabababab : r32[2 * i] == (uint32_t)ts[i]);
    }
    memset(r32, SENTINEL, sizeof(r32));
    CHECK_VK(vkGetQueryPoolResults(device, p, 0, 8, sizeof(r32), r32, 12, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
             VK_NOT_READY);
    for (i = 0; i < 8; i++) {
        CHECK(r32[3 * i + 1] == (i % 2 ? 0u : 1u));
        CHECK(r32[3 * i + 2] == 0xabababab); /* padding */
    }
    /* 64-bit with availability, stride 24. */
    memset(raw, SENTINEL, sizeof(raw));
    CHECK_VK(vkGetQueryPoolResults(device, p, 0, 8, sizeof(raw), raw, 24,
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
             VK_NOT_READY);
    memcpy(r64, raw, sizeof(raw));
    for (i = 0; i < 8; i++) {
        CHECK(r64[3 * i + 1] == (i % 2 ? 0u : 1u));
        CHECK(i % 2 ? r64[3 * i] == 0xababababababababull : r64[3 * i] == ts[i]);
        CHECK(r64[3 * i + 2] == 0xababababababababull);
    }
    /* Only written queries, with WAIT: success; stride 16 for one query each. */
    for (i = 0; i < 8; i += 2) {
        memset(r64, SENTINEL, sizeof(r64));
        CHECK_VK(get64(p, i, 1, r64, VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT), VK_SUCCESS);
        CHECK(r64[0] == ts[i] && r64[1] == 1);
    }
    /* Host reset: unavailable again. */
    vkResetQueryPool(device, p, 0, 4);
    memset(r64, 0, sizeof(r64));
    CHECK_VK(get64(p, 0, 8, r64, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT), VK_NOT_READY);
    for (i = 0; i < 8; i++)
        CHECK(r64[2 * i + 1] == (i >= 4 && i % 2 == 0 ? 1u : 0u));
    vkDestroyQueryPool(device, p, NULL);
}

/* Device resets: before a write in the same command buffer, after one (the
 * query ends unavailable), across submissions; a rewrite after a reset. */
static void test_results_resets(void)
{
    VkQueryPool p = make_pool(VK_QUERY_TYPE_TIMESTAMP, 4);
    VkCommandBuffer cb = begin();
    uint64_t r[8], first;

    vkCmdResetQueryPool(cb, p, 0, 4);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 0);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 1);
    vkCmdResetQueryPool(cb, p, 1, 1); /* undoes the write of query 1 */
    run(cb);
    CHECK_VK(get64(p, 0, 4, r, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT), VK_NOT_READY);
    CHECK(r[1] == 1 && r[3] == 0 && r[5] == 0 && r[7] == 0);
    first = r[0];

    cb = begin();
    vkCmdResetQueryPool(cb, p, 0, 1);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 0);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 1);
    run(cb);
    CHECK_VK(get64(p, 0, 2, r, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT), VK_SUCCESS);
    CHECK(r[1] == 1 && r[3] == 1 && r[0] > first && r[2] >= r[0]);

    cb = begin(); /* a reset alone, in its own submission */
    vkCmdResetQueryPool(cb, p, 0, 4);
    run(cb);
    CHECK_VK(get64(p, 0, 4, r, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT), VK_NOT_READY);
    CHECK(r[1] == 0 && r[3] == 0 && r[5] == 0 && r[7] == 0);
    vkDestroyQueryPool(device, p, NULL);
}

struct release {
    VkSemaphore gate;
    unsigned ms;
};

static void *release_later(void *arg)
{
    struct release *rel = arg;
    VkSemaphoreSignalInfo info;
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = (long)rel->ms * 1000000;
    nanosleep(&ts, NULL);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    info.semaphore = rel->gate;
    info.value = 1;
    vkSignalSemaphore(device, &info);
    return NULL;
}

/* VK_QUERY_RESULT_WAIT_BIT on a submission the host holds back: the call
 * returns once the GPU wrote the queries. */
static void test_results_wait(void)
{
    VkQueryPool p = make_pool(VK_QUERY_TYPE_TIMESTAMP, 2);
    VkSemaphore gate = make_timeline();
    VkCommandBuffer cb = begin();
    struct release rel;
    pthread_t thread;
    uint64_t r[4], t0, dt;

    vkResetQueryPool(device, p, 0, 2);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, p, 0);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, p, 1);
    REQUIRE_VK(submit_on(queue, cb, gate, 1, VK_NULL_HANDLE));
    CHECK_VK(get64(p, 0, 2, r, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT), VK_NOT_READY);
    CHECK(r[1] == 0 && r[3] == 0);
    rel.gate = gate;
    rel.ms = 50;
    t0 = now_ns();
    REQUIRE(pthread_create(&thread, NULL, release_later, &rel) == 0);
    CHECK_VK(get64(p, 0, 2, r, VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT), VK_SUCCESS);
    dt = now_ns() - t0;
    CHECK(dt >= 40000000);
    CHECK(r[1] == 1 && r[3] == 1 && r[0] <= r[2]);
    pthread_join(thread, NULL);
    CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);
    vkDestroySemaphore(device, gate, NULL);
    vkDestroyQueryPool(device, p, NULL);
}

/* Occlusion pools (no queries run without render passes): an unavailable
 * query with PARTIAL reads 0 and is reported unavailable. */
static void test_results_occlusion_partial(void)
{
    VkQueryPool p = make_pool(VK_QUERY_TYPE_OCCLUSION, 3);
    VkCommandBuffer cb = begin();
    uint64_t r[6];
    uint32_t r32[2];

    vkCmdResetQueryPool(cb, p, 0, 3);
    run(cb);
    memset(r, SENTINEL, sizeof(r));
    CHECK_VK(get64(p, 0, 3, r, VK_QUERY_RESULT_PARTIAL_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT), VK_NOT_READY);
    CHECK(r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0 && r[4] == 0 && r[5] == 0);
    memset(r, SENTINEL, sizeof(r));
    CHECK_VK(get64(p, 0, 3, r, 0), VK_NOT_READY);
    CHECK(r[0] == 0xababababababababull && r[2] == 0xababababababababull);
    memset(r32, SENTINEL, sizeof(r32));
    CHECK_VK(vkGetQueryPoolResults(device, p, 2, 1, 8, r32, 8,
                                   VK_QUERY_RESULT_PARTIAL_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
             VK_NOT_READY);
    CHECK(r32[0] == 0 && r32[1] == 0);
    vkDestroyQueryPool(device, p, NULL);
}

/* ---- Copies ---- */

struct copy_case {
    VkQueryResultFlags flags;
    VkDeviceSize offset, stride;
};

/* Each combination of flags, offset and stride, copied in the command buffer
 * that wrote the queries (after a reset there) and in a later one. */
static void test_copy_results(void)
{
    static const struct copy_case cases[] = {
        { 0, 0, 4 },
        { 0, 4, 12 },
        { VK_QUERY_RESULT_64_BIT, 8, 8 },
        { VK_QUERY_RESULT_64_BIT, 16, 24 },
        { VK_QUERY_RESULT_WITH_AVAILABILITY_BIT, 4, 8 },
        { VK_QUERY_RESULT_WITH_AVAILABILITY_BIT, 0, 16 },
        { VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT, 8, 16 },
        { VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT, 0, 32 },
        { VK_QUERY_RESULT_WAIT_BIT, 0, 4 },
        { VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT, 8, 16 },
    };
    enum { N = 5, SLOT = 256 };
    VkQueryPool p = make_pool(VK_QUERY_TYPE_TIMESTAMP, N);
    struct buf out = make_buf(2 * PVK_ARRAY_SIZE(cases) * SLOT, shared_type);
    uint64_t ts[N];
    uint32_t pass, c, i;

    REQUIRE(out.data);
    for (pass = 0; pass < 2; pass++) {
        VkCommandBuffer cb = begin();
        memset(out.data, SENTINEL, (size_t)out.size);
        vkCmdResetQueryPool(cb, p, 0, N);
        for (i = 0; i < N; i++)
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, i);
        if (pass == 1) { /* copied by the next command buffer */
            run(cb);
            cb = begin();
        }
        for (c = 0; c < PVK_ARRAY_SIZE(cases); c++)
            vkCmdCopyQueryPoolResults(cb, p, 0, N, out.buffer, c * SLOT + cases[c].offset, cases[c].stride,
                                      cases[c].flags);
        /* Overlapping copies only touch their own bytes: copy one query
         * into the middle of another case's area afterwards. */
        transfer_barrier(cb);
        vkCmdCopyQueryPoolResults(cb, p, 3, 1, out.buffer, PVK_ARRAY_SIZE(cases) * SLOT, 0, VK_QUERY_RESULT_64_BIT);
        host_barrier(cb);
        run(cb);
        CHECK_VK(get64(p, 0, N, ts, 0), VK_SUCCESS);
        for (c = 0; c < PVK_ARRAY_SIZE(cases); c++) {
            const struct copy_case *k = &cases[c];
            bool wide = (k->flags & VK_QUERY_RESULT_64_BIT) != 0;
            bool avail = (k->flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != 0;
            unsigned char *base = out.data + c * SLOT;
            size_t size = (wide ? 8u : 4u) * (avail ? 2u : 1u), k2;
            for (k2 = 0; k2 < k->offset; k2++)
                CHECK(base[k2] == SENTINEL);
            for (i = 0; i < N; i++) {
                unsigned char *q = base + k->offset + k->stride * i;
                uint64_t v = 0, a = 0;
                memcpy(&v, q, wide ? 8 : 4);
                if (avail)
                    memcpy(&a, q + (wide ? 8 : 4), wide ? 8 : 4);
                CHECK(v == (wide ? ts[i] : (uint32_t)ts[i]));
                CHECK(!avail || a == 1);
                for (k2 = size; k2 < k->stride; k2++)
                    CHECK(q[k2] == SENTINEL);
            }
            CHECK(base[k->offset + k->stride * (N - 1) + size] == SENTINEL);
        }
        {
            uint64_t v;
            memcpy(&v, out.data + PVK_ARRAY_SIZE(cases) * SLOT, 8);
            CHECK(v == ts[3]);
        }
    }
    free_buf(&out);
    vkDestroyQueryPool(device, p, NULL);
}

/* A copy on the compute family's queue, with results written there. */
static void test_copy_compute_queue(void)
{
    VkQueryPool p = make_pool(VK_QUERY_TYPE_TIMESTAMP, 3);
    struct buf out = make_buf(64, shared_type);
    VkCommandBuffer cb = begin_in(compute_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    uint64_t ts[3], v[6];

    REQUIRE(out.data);
    memset(out.data, SENTINEL, 64);
    vkCmdResetQueryPool(cb, p, 0, 3);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 0);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 1);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 2);
    vkCmdCopyQueryPoolResults(cb, p, 0, 3, out.buffer, 0, 16,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT | VK_QUERY_RESULT_WAIT_BIT);
    host_barrier(cb);
    run_on(compute_queue, cb);
    CHECK_VK(get64(p, 0, 3, ts, 0), VK_SUCCESS);
    memcpy(v, out.data, sizeof(v));
    CHECK(v[0] == ts[0] && v[2] == ts[1] && v[4] == ts[2]);
    CHECK(v[1] == 1 && v[3] == 1 && v[5] == 1);
    free_buf(&out);
    vkDestroyQueryPool(device, p, NULL);
}

int main(void)
{
    setup();
    if (!device)
        return 1;
    RUN(test_timestamp_basic);
    RUN(test_timestamp_order);
    RUN(test_timestamp_many);
    RUN(test_results_flags);
    RUN(test_results_resets);
    RUN(test_results_wait);
    RUN(test_results_occlusion_partial);
    RUN(test_copy_results);
    RUN(test_copy_compute_queue);
    teardown();
    return TEST_RESULT();
}
