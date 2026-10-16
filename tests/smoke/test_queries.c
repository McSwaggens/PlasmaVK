/* Query pools (section 14): timestamps, resets, results with every flag,
 * availability, waiting, and copies of results to buffers. */
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <mach/mach_time.h>
#include <pthread.h>
#include <stdlib.h>
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
static bool calibrated; /* VK_KHR_calibrated_timestamps offered (and enabled) */
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
    v13.dynamicRendering = VK_TRUE;
    memset(queue_info, 0, sizeof(queue_info));
    for (i = 0; i < 2; i++) {
        queue_info[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info[i].queueFamilyIndex = i;
        queue_info[i].queueCount = 1;
        queue_info[i].pQueuePriorities = &priority;
    }
    {
        VkExtensionProperties exts[256];
        uint32_t n = 256, e;
        vkEnumerateDeviceExtensionProperties(pd, NULL, &n, exts);
        for (e = 0; e < n; e++)
            calibrated |= strcmp(exts[e].extensionName, VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME) == 0;
    }
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &v13;
    device_info.queueCreateInfoCount = 2;
    device_info.pQueueCreateInfos = queue_info;
    if (calibrated) {
        static const char *const ext = VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
        device_info.enabledExtensionCount = 1;
        device_info.ppEnabledExtensionNames = &ext;
    }
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

/* VK_KHR_calibrated_timestamps: the device domain is the timestamps' clock,
 * read with the host's: a GPU timestamp between two calibrations falls
 * between their device values, and the device and raw host clocks advance
 * alike (within the deviations and a millisecond). */
static void test_calibrated_timestamps(void)
{
    PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR get_domains;
    PFN_vkGetCalibratedTimestampsKHR get_timestamps;
    VkCalibratedTimestampInfoKHR info[2];
    VkTimeDomainKHR domains[8];
    VkQueryPool p;
    VkCommandBuffer cb;
    uint64_t before[2], after[2], dev0 = 0, dev1 = 0, r[2];
    uint32_t n = 8, i, found = 0;
    double host_ns, device_ns;

    CHECK(calibrated);
    if (!calibrated)
        return;
    get_domains = (PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR)vkGetInstanceProcAddr(
        instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR");
    get_timestamps = (PFN_vkGetCalibratedTimestampsKHR)vkGetDeviceProcAddr(device, "vkGetCalibratedTimestampsKHR");
    REQUIRE(get_domains && get_timestamps);
    CHECK_VK(get_domains(pd, &n, domains), VK_SUCCESS);
    for (i = 0; i < n; i++)
        found |= domains[i] == VK_TIME_DOMAIN_DEVICE_KHR ? 1u : domains[i] == VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR ? 2u : 0u;
    CHECK(found == 3);
    memset(info, 0, sizeof(info));
    for (i = 0; i < 2; i++)
        info[i].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR;
    info[0].timeDomain = VK_TIME_DOMAIN_DEVICE_KHR;
    info[1].timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR;
    p = make_pool(VK_QUERY_TYPE_TIMESTAMP, 1);
    vkResetQueryPool(device, p, 0, 1);
    cb = begin();
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 0);
    CHECK_VK(get_timestamps(device, 2, info, before, &dev0), VK_SUCCESS);
    run(cb);
    CHECK_VK(get_timestamps(device, 2, info, after, &dev1), VK_SUCCESS);
    CHECK_VK(get64(p, 0, 1, r, VK_QUERY_RESULT_WAIT_BIT), VK_SUCCESS);
    CHECK(before[0] <= r[0] && r[0] <= after[0]);
    host_ns = (double)(after[1] - before[1]);
    device_ns = (double)(after[0] - before[0]) * period;
    CHECK(host_ns > 0.0 && device_ns > 0.0);
    CHECK(host_ns - device_ns < 1e6 + (double)(dev0 + dev1) && device_ns - host_ns < 1e6 + (double)(dev0 + dev1));
    CHECK(dev0 < 1000000 && dev1 < 1000000);
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

/*
 * #version 450
 * // A small triangle per draw, at a place picked by its first instance.
 * void main()
 * {
 *     uint i = uint(gl_InstanceIndex);
 *     vec2 at = vec2(float(i % 64u), float(i / 64u % 64u)) / 32.0 - 1.0;
 *     vec2 corner = vec2(float(gl_VertexIndex == 1), float(gl_VertexIndex == 2)) / 64.0;
 *     gl_Position = vec4(at + corner, 0.0, 1.0);
 * }
 */
static const uint32_t spv_spread_vs[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x00000040, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0008000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00000022, 0x00000036,
    0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005,
    0x00000008, 0x00000069, 0x00070005, 0x0000000b, 0x495f6c67, 0x6174736e, 0x4965636e, 0x7865646e,
    0x00000000, 0x00030005, 0x00000011, 0x00007461, 0x00040005, 0x00000021, 0x6e726f63, 0x00007265,
    0x00060005, 0x00000022, 0x565f6c67, 0x65747265, 0x646e4978, 0x00007865, 0x00060005, 0x00000034,
    0x505f6c67, 0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x00000034, 0x00000000, 0x505f6c67,
    0x7469736f, 0x006e6f69, 0x00070006, 0x00000034, 0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953,
    0x00000000, 0x00070006, 0x00000034, 0x00000002, 0x435f6c67, 0x4470696c, 0x61747369, 0x0065636e,
    0x00070006, 0x00000034, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e, 0x00030005,
    0x00000036, 0x00000000, 0x00040047, 0x0000000b, 0x0000000b, 0x0000002b, 0x00040047, 0x00000022,
    0x0000000b, 0x0000002a, 0x00030047, 0x00000034, 0x00000002, 0x00050048, 0x00000034, 0x00000000,
    0x0000000b, 0x00000000, 0x00050048, 0x00000034, 0x00000001, 0x0000000b, 0x00000001, 0x00050048,
    0x00000034, 0x00000002, 0x0000000b, 0x00000003, 0x00050048, 0x00000034, 0x00000003, 0x0000000b,
    0x00000004, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006,
    0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00040015, 0x00000009,
    0x00000020, 0x00000001, 0x00040020, 0x0000000a, 0x00000001, 0x00000009, 0x0004003b, 0x0000000a,
    0x0000000b, 0x00000001, 0x00030016, 0x0000000e, 0x00000020, 0x00040017, 0x0000000f, 0x0000000e,
    0x00000002, 0x00040020, 0x00000010, 0x00000007, 0x0000000f, 0x0004002b, 0x00000006, 0x00000013,
    0x00000040, 0x0004002b, 0x0000000e, 0x0000001b, 0x42000000, 0x0004002b, 0x0000000e, 0x0000001e,
    0x3f800000, 0x0004003b, 0x0000000a, 0x00000022, 0x00000001, 0x0004002b, 0x00000009, 0x00000024,
    0x00000001, 0x00020014, 0x00000025, 0x0004002b, 0x0000000e, 0x00000027, 0x00000000, 0x0004002b,
    0x00000009, 0x0000002a, 0x00000002, 0x0004002b, 0x0000000e, 0x0000002e, 0x42800000, 0x00040017,
    0x00000031, 0x0000000e, 0x00000004, 0x0004002b, 0x00000006, 0x00000032, 0x00000001, 0x0004001c,
    0x00000033, 0x0000000e, 0x00000032, 0x0006001e, 0x00000034, 0x00000031, 0x0000000e, 0x00000033,
    0x00000033, 0x00040020, 0x00000035, 0x00000003, 0x00000034, 0x0004003b, 0x00000035, 0x00000036,
    0x00000003, 0x0004002b, 0x00000009, 0x00000037, 0x00000000, 0x00040020, 0x0000003e, 0x00000003,
    0x00000031, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
    0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x0004003b, 0x00000010, 0x00000011, 0x00000007,
    0x0004003b, 0x00000010, 0x00000021, 0x00000007, 0x0004003d, 0x00000009, 0x0000000c, 0x0000000b,
    0x0004007c, 0x00000006, 0x0000000d, 0x0000000c, 0x0003003e, 0x00000008, 0x0000000d, 0x0004003d,
    0x00000006, 0x00000012, 0x00000008, 0x00050089, 0x00000006, 0x00000014, 0x00000012, 0x00000013,
    0x00040070, 0x0000000e, 0x00000015, 0x00000014, 0x0004003d, 0x00000006, 0x00000016, 0x00000008,
    0x00050086, 0x00000006, 0x00000017, 0x00000016, 0x00000013, 0x00050089, 0x00000006, 0x00000018,
    0x00000017, 0x00000013, 0x00040070, 0x0000000e, 0x00000019, 0x00000018, 0x00050050, 0x0000000f,
    0x0000001a, 0x00000015, 0x00000019, 0x00050050, 0x0000000f, 0x0000001c, 0x0000001b, 0x0000001b,
    0x00050088, 0x0000000f, 0x0000001d, 0x0000001a, 0x0000001c, 0x00050050, 0x0000000f, 0x0000001f,
    0x0000001e, 0x0000001e, 0x00050083, 0x0000000f, 0x00000020, 0x0000001d, 0x0000001f, 0x0003003e,
    0x00000011, 0x00000020, 0x0004003d, 0x00000009, 0x00000023, 0x00000022, 0x000500aa, 0x00000025,
    0x00000026, 0x00000023, 0x00000024, 0x000600a9, 0x0000000e, 0x00000028, 0x00000026, 0x0000001e,
    0x00000027, 0x0004003d, 0x00000009, 0x00000029, 0x00000022, 0x000500aa, 0x00000025, 0x0000002b,
    0x00000029, 0x0000002a, 0x000600a9, 0x0000000e, 0x0000002c, 0x0000002b, 0x0000001e, 0x00000027,
    0x00050050, 0x0000000f, 0x0000002d, 0x00000028, 0x0000002c, 0x00050050, 0x0000000f, 0x0000002f,
    0x0000002e, 0x0000002e, 0x00050088, 0x0000000f, 0x00000030, 0x0000002d, 0x0000002f, 0x0003003e,
    0x00000021, 0x00000030, 0x0004003d, 0x0000000f, 0x00000038, 0x00000011, 0x0004003d, 0x0000000f,
    0x00000039, 0x00000021, 0x00050081, 0x0000000f, 0x0000003a, 0x00000038, 0x00000039, 0x00050051,
    0x0000000e, 0x0000003b, 0x0000003a, 0x00000000, 0x00050051, 0x0000000e, 0x0000003c, 0x0000003a,
    0x00000001, 0x00070050, 0x00000031, 0x0000003d, 0x0000003b, 0x0000003c, 0x00000027, 0x0000001e,
    0x00050041, 0x0000003e, 0x0000003f, 0x00000036, 0x00000037, 0x0003003e, 0x0000003f, 0x0000003d,
    0x000100fd, 0x00010038,
};

/*
 * #version 450
 * layout(location = 0) out vec4 color;
 * void main()
 * {
 *     color = vec4(1.0);
 * }
 */
static const uint32_t spv_white_fs[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x0000000c, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
    0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00040005, 0x00000009, 0x6f6c6f63, 0x00000072, 0x00040047, 0x00000009, 0x0000001e, 0x00000000,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
    0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007,
    0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x0004002b, 0x00000006, 0x0000000a, 0x3f800000,
    0x0007002c, 0x00000007, 0x0000000b, 0x0000000a, 0x0000000a, 0x0000000a, 0x0000000a, 0x00050036,
    0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0003003e, 0x00000009,
    0x0000000b, 0x000100fd, 0x00010038,
};

/* A render pass of `draws` small triangles spread over the target, cleared
 * and stored. */
static void record_spread_pass(VkCommandBuffer cb, VkImageView view, uint32_t size, VkPipeline pipeline,
                               uint32_t draws)
{
    VkRenderingAttachmentInfo att;
    VkRenderingInfo info;
    uint32_t i;
    memset(&att, 0, sizeof(att));
    att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    att.imageView = view;
    att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    info.renderArea.extent.width = info.renderArea.extent.height = size;
    info.layerCount = 1;
    info.colorAttachmentCount = 1;
    info.pColorAttachments = &att;
    vkCmdBeginRendering(cb, &info);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    for (i = 0; i < draws; i++)
        vkCmdDraw(cb, 3, 1, 0, i);
    vkCmdEndRendering(cb);
}

/* Timestamps bracket render passes (section 14, TIMESTAMPS). A pass of many
 * draws begins with work that its render encoder's timestamps leave out:
 * written there, a profiler's scopes around two identical passes measured
 * the second one short and the first one with the start of the second in
 * it (without sparseBinding, as on this device: with it every barrier also
 * flushes aliased addresses, and that left less out). Here they must
 * measure alike, after a pass that ran the pipeline once. GPU work of
 * other tests may run in between: the best of a few tries counts. */
static void test_timestamp_passes(void)
{
    enum { SIZE = 1024, DRAWS = 10000, TRIES = 5 };
    VkQueryPool p = make_pool(VK_QUERY_TYPE_TIMESTAMP, 3);
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkShaderModule modules[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkImageCreateInfo ii;
    VkImageViewCreateInfo vi;
    VkMemoryRequirements req;
    VkMemoryAllocateInfo alloc;
    VkShaderModuleCreateInfo si;
    VkPipelineLayoutCreateInfo li;
    VkPipelineShaderStageCreateInfo stages[2];
    VkPipelineVertexInputStateCreateInfo vertex;
    VkPipelineInputAssemblyStateCreateInfo assembly;
    VkViewport viewport;
    VkRect2D scissor;
    VkPipelineViewportStateCreateInfo vp;
    VkPipelineRasterizationStateCreateInfo raster;
    VkPipelineMultisampleStateCreateInfo ms;
    VkPipelineColorBlendAttachmentState blend;
    VkPipelineColorBlendStateCreateInfo cbs;
    VkPipelineRenderingCreateInfo rendering;
    VkGraphicsPipelineCreateInfo gi;
    uint64_t r[3];
    double diff = 1.0;
    uint32_t t, i;

    memset(&ii, 0, sizeof(ii));
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = format;
    ii.extent.width = ii.extent.height = SIZE;
    ii.extent.depth = 1;
    ii.mipLevels = ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    REQUIRE_VK(vkCreateImage(device, &ii, NULL, &image));
    vkGetImageMemoryRequirements(device, image, &req);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = private_type;
    REQUIRE_VK(vkAllocateMemory(device, &alloc, NULL, &memory));
    REQUIRE_VK(vkBindImageMemory(device, image, memory, 0));
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = vi.subresourceRange.layerCount = 1;
    REQUIRE_VK(vkCreateImageView(device, &vi, NULL, &view));

    for (i = 0; i < 2; i++) {
        memset(&si, 0, sizeof(si));
        si.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        si.codeSize = i ? sizeof(spv_white_fs) : sizeof(spv_spread_vs);
        si.pCode = i ? spv_white_fs : spv_spread_vs;
        REQUIRE_VK(vkCreateShaderModule(device, &si, NULL, &modules[i]));
        memset(&stages[i], 0, sizeof(stages[i]));
        stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[i].stage = i ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT;
        stages[i].module = modules[i];
        stages[i].pName = "main";
    }
    memset(&li, 0, sizeof(li));
    li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    REQUIRE_VK(vkCreatePipelineLayout(device, &li, NULL, &layout));
    memset(&vertex, 0, sizeof(vertex));
    vertex.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    memset(&assembly, 0, sizeof(assembly));
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    memset(&viewport, 0, sizeof(viewport));
    viewport.width = viewport.height = (float)SIZE;
    viewport.maxDepth = 1.0f;
    memset(&scissor, 0, sizeof(scissor));
    scissor.extent.width = scissor.extent.height = SIZE;
    memset(&vp, 0, sizeof(vp));
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = vp.scissorCount = 1;
    vp.pViewports = &viewport;
    vp.pScissors = &scissor;
    memset(&raster, 0, sizeof(raster));
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.lineWidth = 1.0f;
    memset(&ms, 0, sizeof(ms));
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    memset(&blend, 0, sizeof(blend));
    blend.colorWriteMask = 0xf;
    memset(&cbs, 0, sizeof(cbs));
    cbs.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbs.attachmentCount = 1;
    cbs.pAttachments = &blend;
    memset(&rendering, 0, sizeof(rendering));
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &format;
    memset(&gi, 0, sizeof(gi));
    gi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gi.pNext = &rendering;
    gi.stageCount = 2;
    gi.pStages = stages;
    gi.pVertexInputState = &vertex;
    gi.pInputAssemblyState = &assembly;
    gi.pViewportState = &vp;
    gi.pRasterizationState = &raster;
    gi.pMultisampleState = &ms;
    gi.pColorBlendState = &cbs;
    gi.layout = layout;
    REQUIRE_VK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gi, NULL, &pipeline));

    for (t = 0; t < TRIES && diff > 0.2; t++) {
        VkCommandBuffer cb = begin();
        int64_t a, b;
        vkCmdResetQueryPool(cb, p, 0, 3);
        record_spread_pass(cb, view, SIZE, pipeline, DRAWS);
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, p, 0);
        record_spread_pass(cb, view, SIZE, pipeline, DRAWS);
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, p, 1);
        record_spread_pass(cb, view, SIZE, pipeline, DRAWS);
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, p, 2);
        run(cb);
        vkFreeCommandBuffers(device, pool, 1, &cb);
        CHECK_VK(get64(p, 0, 3, r, 0), VK_SUCCESS);
        a = (int64_t)(r[1] - r[0]);
        b = (int64_t)(r[2] - r[1]);
        CHECK(a > 0 && b > 0);
        if (a > 0 && b > 0 && (double)(a > b ? a - b : b - a) / (double)(a > b ? a : b) < diff)
            diff = (double)(a > b ? a - b : b - a) / (double)(a > b ? a : b);
        if (diff > 0.2 && t + 1 == TRIES)
            printf("    passes %lld and %lld ticks\n", (long long)a, (long long)b);
    }
    CHECK(diff <= 0.2);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroyShaderModule(device, modules[0], NULL);
    vkDestroyShaderModule(device, modules[1], NULL);
    vkDestroyImageView(device, view, NULL);
    vkDestroyImage(device, image, NULL);
    vkFreeMemory(device, memory, NULL);
    vkDestroyQueryPool(device, p, NULL);
}

static int cmp_i64(const void *a, const void *b)
{
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return x < y ? -1 : x > y;
}

/* Timestamps bracket copies too (section 14, TIMESTAMPS): written at the
 * start of the encoder of the copies after them, they came late in some
 * runs (Metal takes an encoder's timestamps together, at its start or
 * later), and one of two identical chains of copies between three
 * timestamps measured a fraction of the other. GPU work of other tests
 * only makes a chain measure longer: of all tries, the shortest must
 * measure at least 0.4 of the median. */
static void test_timestamp_copies(void)
{
    enum { TRIES = 12, COPIES = 8 };
    VkQueryPool p = make_pool(VK_QUERY_TYPE_TIMESTAMP, 3);
    struct buf a = make_buf(64u << 20, private_type), b = make_buf(64u << 20, private_type);
    int64_t d[2 * TRIES], sorted[2 * TRIES];
    uint64_t r[3];
    uint32_t t;

    for (t = 0; t < TRIES; t++) {
        VkCommandBuffer cb = begin();
        vkCmdResetQueryPool(cb, p, 0, 3);
        record_work(cb, &a, &b, 64u << 20, 2);
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, p, 0);
        record_work(cb, &a, &b, 64u << 20, COPIES);
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, p, 1);
        record_work(cb, &a, &b, 64u << 20, COPIES);
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, p, 2);
        run(cb);
        vkFreeCommandBuffers(device, pool, 1, &cb);
        CHECK_VK(get64(p, 0, 3, r, 0), VK_SUCCESS);
        d[2 * t] = (int64_t)(r[1] - r[0]);
        d[2 * t + 1] = (int64_t)(r[2] - r[1]);
    }
    memcpy(sorted, d, sizeof(d));
    qsort(sorted, 2 * TRIES, sizeof(sorted[0]), cmp_i64);
    CHECK(sorted[0] > 0 && (double)sorted[0] >= 0.4 * (double)sorted[TRIES]);
    if (sorted[0] <= 0 || (double)sorted[0] < 0.4 * (double)sorted[TRIES])
        for (t = 0; t < TRIES; t++)
            printf("    chains %lld and %lld ticks\n", (long long)d[2 * t], (long long)d[2 * t + 1]);
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

/* Whether query q of p is available within 2 s, polled without
 * VK_QUERY_RESULT_WAIT_BIT. */
static bool available_soon(VkQueryPool p, uint32_t q)
{
    uint64_t r[2];
    int i;
    for (i = 0; i < 2000; i++) {
        struct timespec ts;
        if (get64(p, q, 1, r, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS)
            return r[1] == 1;
        ts.tv_sec = 0;
        ts.tv_nsec = 1000000;
        nanosleep(&ts, NULL);
    }
    return false;
}

/* One submission, three batches writing a query each: the second and third
 * wait for the host to signal 1 and 2. A batch's query is available once it
 * is done, not the submission (section 14, TIMESTAMPS): a host waiting for
 * it with VK_QUERY_RESULT_WAIT_BIT before releasing the later batches
 * would wait forever. */
static void test_results_batches(void)
{
    VkQueryPool p = make_pool(VK_QUERY_TYPE_TIMESTAMP, 3);
    VkSemaphore gate = make_timeline();
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkTimelineSemaphoreSubmitInfo values[3];
    VkSemaphoreSignalInfo signal;
    VkCommandBuffer cb[3];
    VkSubmitInfo info[3];
    uint64_t wait[3] = { 0, 1, 2 }, r[6];
    uint32_t i, k;

    vkResetQueryPool(device, p, 0, 3);
    for (i = 0; i < 3; i++) {
        cb[i] = begin();
        vkCmdWriteTimestamp2(cb[i], VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, p, i);
        vkEndCommandBuffer(cb[i]);
        memset(&values[i], 0, sizeof(values[i]));
        values[i].sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        values[i].waitSemaphoreValueCount = i ? 1 : 0;
        values[i].pWaitSemaphoreValues = &wait[i];
        memset(&info[i], 0, sizeof(info[i]));
        info[i].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info[i].pNext = &values[i];
        info[i].waitSemaphoreCount = i ? 1 : 0;
        info[i].pWaitSemaphores = &gate;
        info[i].pWaitDstStageMask = &stage;
        info[i].commandBufferCount = 1;
        info[i].pCommandBuffers = &cb[i];
    }
    REQUIRE_VK(vkQueueSubmit(queue, 3, info, VK_NULL_HANDLE));
    for (i = 0; i < 3; i++) { /* batches up to i released */
        bool ready = available_soon(p, i);
        CHECK(ready);
        CHECK_VK(get64(p, 0, 3, r, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT), i < 2 ? VK_NOT_READY : VK_SUCCESS);
        for (k = 0; k < 3; k++)
            CHECK(r[2 * k + 1] == (k <= i));
        if (ready) /* (else it would not return) */
            CHECK_VK(get64(p, i, 1, r, VK_QUERY_RESULT_WAIT_BIT), VK_SUCCESS);
        memset(&signal, 0, sizeof(signal));
        signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
        signal.semaphore = gate;
        signal.value = i + 1;
        if (i < 2)
            CHECK_VK(vkSignalSemaphore(device, &signal), VK_SUCCESS);
    }
    CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);
    vkDestroySemaphore(device, gate, NULL);
    vkDestroyQueryPool(device, p, NULL);
}

struct late_write {
    VkCommandBuffer cb;
    unsigned ms;
    VkResult result;
};

static void *submit_later(void *arg)
{
    struct late_write *w = arg;
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = (long)w->ms * 1000000;
    nanosleep(&ts, NULL);
    w->result = submit_on(queue, w->cb, VK_NULL_HANDLE, 0, VK_NULL_HANDLE);
    return NULL;
}

/* VK_QUERY_RESULT_WAIT_BIT on queries reset on the host whose write another
 * thread submits only later: the call returns once they are written. */
static void test_results_wait_reset(void)
{
    VkQueryPool p = make_pool(VK_QUERY_TYPE_TIMESTAMP, 2);
    struct late_write w;
    pthread_t thread;
    uint64_t r[4], t0;

    vkResetQueryPool(device, p, 0, 2);
    w.cb = begin();
    vkCmdWriteTimestamp2(w.cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, p, 0);
    vkCmdWriteTimestamp2(w.cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, p, 1);
    w.ms = 50;
    t0 = now_ns();
    REQUIRE(pthread_create(&thread, NULL, submit_later, &w) == 0);
    CHECK_VK(get64(p, 0, 2, r, VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT), VK_SUCCESS);
    CHECK(now_ns() - t0 >= 40000000);
    CHECK(r[1] == 1 && r[3] == 1 && r[0] != 0 && r[0] <= r[2]);
    pthread_join(thread, NULL);
    CHECK_VK(w.result, VK_SUCCESS);
    CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);
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

/* Copies see the queries as of their place in submission order: written,
 * then reset (by the same command buffer, or one of an earlier submission
 * still running), then written again; an earlier submission's write behind
 * long work. */
static void test_copy_order(void)
{
    VkQueryPool p = make_pool(VK_QUERY_TYPE_TIMESTAMP, 3);
    struct buf a = make_buf(16u << 20, private_type), b = make_buf(16u << 20, private_type);
    struct buf out = make_buf(4 * 48, shared_type);
    VkCommandBuffer cb = begin(), cb2;
    uint64_t v[4][6], ts[3];

    REQUIRE(out.data);
    memset(out.data, SENTINEL, (size_t)out.size);
    vkCmdResetQueryPool(cb, p, 0, 3);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 0);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 1);
    vkCmdCopyQueryPoolResults(cb, p, 0, 3, out.buffer, 0, 16,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    vkCmdResetQueryPool(cb, p, 1, 1);
    vkCmdCopyQueryPoolResults(cb, p, 0, 3, out.buffer, 48, 16,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    record_work(cb, &a, &b, 16u << 20, 4);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 2);
    REQUIRE_VK(submit_on(queue, cb, VK_NULL_HANDLE, 0, VK_NULL_HANDLE));
    cb2 = begin(); /* submitted while the first may still run */
    vkCmdCopyQueryPoolResults(cb2, p, 0, 3, out.buffer, 96, 16,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    vkCmdResetQueryPool(cb2, p, 0, 1);
    vkCmdWriteTimestamp2(cb2, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, p, 1);
    vkCmdCopyQueryPoolResults(cb2, p, 0, 3, out.buffer, 144, 16,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    host_barrier(cb2);
    run(cb2);
    memcpy(v, out.data, sizeof(v));
    CHECK(v[0][1] == 1 && v[0][3] == 1 && v[0][5] == 0 && v[0][4] == 0xababababababababull);
    CHECK(v[1][1] == 1 && v[1][3] == 0 && v[1][5] == 0 && v[1][0] == v[0][0]);
    CHECK(v[1][2] == 0xababababababababull);
    CHECK(v[2][1] == 1 && v[2][3] == 0 && v[2][5] == 1 && v[2][0] == v[0][0] && v[2][4] >= v[0][2]);
    CHECK(v[3][1] == 0 && v[3][3] == 1 && v[3][5] == 1 && v[3][4] == v[2][4] && v[3][2] >= v[2][4]);
    CHECK_VK(get64(p, 1, 2, ts + 1, 0), VK_SUCCESS);
    CHECK(ts[1] == v[3][2] && ts[2] == v[3][4]);
    free_buf(&out);
    free_buf(&a);
    free_buf(&b);
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
    RUN(test_calibrated_timestamps);
    RUN(test_timestamp_order);
    RUN(test_timestamp_many);
    RUN(test_timestamp_passes);
    RUN(test_timestamp_copies);
    RUN(test_results_flags);
    RUN(test_results_resets);
    RUN(test_results_wait);
    RUN(test_results_batches);
    RUN(test_results_wait_reset);
    RUN(test_results_occlusion_partial);
    RUN(test_copy_results);
    RUN(test_copy_order);
    RUN(test_copy_compute_queue);
    teardown();
    return TEST_RESULT();
}
