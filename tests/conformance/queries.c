/* Chapter "Queries": query pools, resets on the host and the device, the
 * availability and the results of timestamp queries (vkGetQueryPoolResults
 * with each flag, vkCmdCopyQueryPoolResults), and unavailable occlusion
 * queries. Occlusion queries that count samples need render passes and are
 * not tested here. */
#include "ct.h"

#include <stdio.h>
#include <time.h>

#define SENTINEL 0xab
#define SENTINEL64 0xababababababababull

/* ---- Environment ---- */

struct qt {
    struct ct_env e;
    uint32_t valid_bits; /* timestampValidBits of the queue's family */
    int host_reset;      /* hostQueryReset enabled */
    int timeline;        /* timelineSemaphore enabled */
    int sync2;           /* synchronization2 enabled */
};

/* An instance and a device with hostQueryReset and timelineSemaphore
 * (Vulkan 1.2) and synchronization2 (1.3) enabled when supported. Skips the
 * test if `timestamps` and the queue's family has none. */
static void qt_init(struct ct *t, struct qt *q, int timestamps)
{
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    VkQueueFamilyProperties families[64];
    struct ct_features f;
    uint32_t count = CT_ARRAY_SIZE(families);
    void *chain = NULL;

    memset(q, 0, sizeof(*q));
    ct_env_init(t, &q->e);
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&v13, 0, sizeof(v13));
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    if (q->e.version >= VK_API_VERSION_1_2) {
        ct_get_features(&q->e, &f, 0);
        v12.hostQueryReset = f.v12.hostQueryReset;
        v12.timelineSemaphore = f.v12.timelineSemaphore;
        chain = &v12;
        if (q->e.version >= VK_API_VERSION_1_3) {
            v13.synchronization2 = f.v13.synchronization2;
            v13.pNext = chain;
            chain = &v13;
        }
    }
    q->host_reset = v12.hostQueryReset;
    q->timeline = v12.timelineSemaphore;
    q->sync2 = v13.synchronization2;
    ct_env_device(t, &q->e, chain);
    vkGetPhysicalDeviceQueueFamilyProperties(q->e.pd, &count, families);
    q->valid_bits = families[q->e.queue_family].timestampValidBits;
    if (timestamps && !q->valid_bits)
        ct_skip(t, "the queue family has no timestamps (timestampValidBits 0)");
}

static VkQueryPool qt_pool(struct ct *t, struct qt *q, VkQueryType type, uint32_t count)
{
    VkQueryPoolCreateInfo info;
    VkQueryPool pool = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = type;
    info.queryCount = count;
    CT_REQUIRE_VK(t, vkCreateQueryPool(q->e.device, &info, NULL, &pool));
    return pool;
}

/* A device-local buffer for GPU work (contents irrelevant). */
struct qt_scratch {
    VkBuffer buffer;
    VkDeviceMemory memory;
};

static void qt_scratch(struct ct *t, struct qt *q, VkDeviceSize size, struct qt_scratch *s)
{
    VkBufferCreateInfo info;
    VkMemoryAllocateInfo alloc;
    VkMemoryRequirements req;
    uint32_t type;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    CT_REQUIRE_VK(t, vkCreateBuffer(q->e.device, &info, NULL, &s->buffer));
    vkGetBufferMemoryRequirements(q->e.device, s->buffer, &req);
    type = ct_find_memory_type(&q->e, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
    CT_REQUIRE(t, type != UINT32_MAX);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    CT_REQUIRE_VK(t, vkAllocateMemory(q->e.device, &alloc, NULL, &s->memory));
    CT_REQUIRE_VK(t, vkBindBufferMemory(q->e.device, s->buffer, s->memory, 0));
}

static void qt_scratch_fini(struct qt *q, struct qt_scratch *s)
{
    vkDestroyBuffer(q->e.device, s->buffer, NULL);
    vkFreeMemory(q->e.device, s->memory, NULL);
}

/* `n` ordered copies of `size` bytes back and forth between a and b. */
static void qt_work(VkCommandBuffer cb, const struct qt_scratch *a, const struct qt_scratch *b, VkDeviceSize size,
                    uint32_t n)
{
    VkBufferCopy region;
    uint32_t i;
    memset(&region, 0, sizeof(region));
    region.size = size;
    for (i = 0; i < n; i++) {
        vkCmdCopyBuffer(cb, i & 1 ? b->buffer : a->buffer, i & 1 ? a->buffer : b->buffer, 1, &region);
        ct_transfer_barrier(cb);
    }
}

/* b - a in timestamp ticks, modulo 2^valid_bits. */
static uint64_t qt_ticks(const struct qt *q, uint64_t a, uint64_t b)
{
    uint64_t mask = q->valid_bits >= 64 ? ~0ull : (1ull << q->valid_bits) - 1;
    return (b - a) & mask;
}

/* a <= b for timestamps less than half the counter's range apart. */
static int qt_ordered(const struct qt *q, uint64_t a, uint64_t b)
{
    uint64_t half = q->valid_bits >= 64 ? 1ull << 63 : 1ull << (q->valid_bits - 1);
    return qt_ticks(q, a, b) < half;
}

static uint64_t qt_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

/* Submits and waits for a recorded command buffer without freeing it. */
static void qt_run(struct ct *t, struct qt *q, VkCommandBuffer cb)
{
    ct_submit_and_wait(t, &q->e, cb);
}

/* ---- Pools ---- */

CT_TEST(create_query_pools,
        "Queries > Query Pools: vkCreateQueryPool creates occlusion and timestamp pools of any queryCount > 0 "
        "(VUID-VkQueryPoolCreateInfo-queryCount-02763); vkDestroyQueryPool accepts VK_NULL_HANDLE",
        "vkCreateQueryPool vkDestroyQueryPool")
{
    static const uint32_t counts[] = { 1, 7, 64, 4096 };
    struct qt q;
    uint32_t i;
    qt_init(t, &q, 0);
    for (i = 0; i < CT_ARRAY_SIZE(counts); i++) {
        VkQueryPool a = qt_pool(t, &q, VK_QUERY_TYPE_OCCLUSION, counts[i]);
        VkQueryPool b = qt_pool(t, &q, VK_QUERY_TYPE_TIMESTAMP, counts[i]);
        CT_CHECK(t, a != VK_NULL_HANDLE && b != VK_NULL_HANDLE && a != b);
        vkDestroyQueryPool(q.e.device, a, NULL);
        vkDestroyQueryPool(q.e.device, b, NULL);
    }
    vkDestroyQueryPool(q.e.device, VK_NULL_HANDLE, NULL);
    ct_env_fini(&q.e);
}

/* Checks queries [0, n) of results read with 64-bit values and availability
 * (stride 16) after they were reset: unavailable, no value written. */
static void check_unavailable(struct ct *t, const uint64_t *r, uint32_t n, const char *what)
{
    uint32_t i;
    for (i = 0; i < n; i++) {
        CT_CHECKF(t, r[2 * i] == SENTINEL64, "%s: query %u: a value was written for an unavailable query", what, i);
        CT_CHECKF(t, r[2 * i + 1] == 0, "%s: query %u: availability %llu, expected 0", what, i,
                  (unsigned long long)r[2 * i + 1]);
    }
}

CT_TEST(host_reset_unavailable,
        "Queries > Query Operation: vkResetQueryPool sets queries unavailable; vkGetQueryPoolResults without WAIT or "
        "PARTIAL then writes no values for them and returns VK_NOT_READY, but writes their availability (0) with "
        "VK_QUERY_RESULT_WITH_AVAILABILITY_BIT",
        "vkCreateQueryPool vkResetQueryPool vkGetQueryPoolResults vkDestroyQueryPool")
{
    static const VkQueryType types[] = { VK_QUERY_TYPE_OCCLUSION, VK_QUERY_TYPE_TIMESTAMP };
    struct qt q;
    uint64_t r[8];
    uint32_t i;
    qt_init(t, &q, 0);
    if (q.e.version < VK_API_VERSION_1_2)
        ct_skip(t, "vkResetQueryPool is Vulkan 1.2");
    CT_REQUIREF(t, q.host_reset, "hostQueryReset is required by Vulkan 1.2");
    for (i = 0; i < CT_ARRAY_SIZE(types); i++) {
        VkQueryPool pool = qt_pool(t, &q, types[i], 4);
        vkResetQueryPool(q.e.device, pool, 0, 4);
        memset(r, SENTINEL, sizeof(r));
        CT_CHECK_VK(t,
                    vkGetQueryPoolResults(q.e.device, pool, 0, 4, sizeof(r), r, 16,
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
                    VK_NOT_READY);
        check_unavailable(t, r, 4, types[i] == VK_QUERY_TYPE_OCCLUSION ? "occlusion" : "timestamp");
        memset(r, SENTINEL, sizeof(r));
        CT_CHECK_VK(t, vkGetQueryPoolResults(q.e.device, pool, 1, 2, 16, r, 8, VK_QUERY_RESULT_64_BIT), VK_NOT_READY);
        CT_CHECK(t, r[0] == SENTINEL64 && r[1] == SENTINEL64 && r[2] == SENTINEL64);
        vkDestroyQueryPool(q.e.device, pool, NULL);
    }
    ct_env_fini(&q.e);
}

CT_TEST(cmd_reset_unavailable,
        "Queries > Query Operation: vkCmdResetQueryPool sets queries unavailable when it executes "
        "(vkGetQueryPoolResults then returns VK_NOT_READY and availability 0, and writes no values)",
        "vkCreateQueryPool vkCmdResetQueryPool vkGetQueryPoolResults vkDestroyQueryPool vkQueueSubmit vkWaitForFences")
{
    static const VkQueryType types[] = { VK_QUERY_TYPE_OCCLUSION, VK_QUERY_TYPE_TIMESTAMP };
    struct qt q;
    uint64_t r[6];
    uint32_t i;
    qt_init(t, &q, 0);
    for (i = 0; i < CT_ARRAY_SIZE(types); i++) {
        VkQueryPool pool = qt_pool(t, &q, types[i], 3);
        VkCommandBuffer cb = ct_begin_commands(t, &q.e);
        vkCmdResetQueryPool(cb, pool, 0, 3);
        qt_run(t, &q, cb);
        memset(r, SENTINEL, sizeof(r));
        CT_CHECK_VK(t,
                    vkGetQueryPoolResults(q.e.device, pool, 0, 3, sizeof(r), r, 16,
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
                    VK_NOT_READY);
        check_unavailable(t, r, 3, types[i] == VK_QUERY_TYPE_OCCLUSION ? "occlusion" : "timestamp");
        vkDestroyQueryPool(q.e.device, pool, NULL);
    }
    ct_env_fini(&q.e);
}

CT_TEST(occlusion_partial_unavailable,
        "Queries > Query Operation: with VK_QUERY_RESULT_PARTIAL_BIT, an unavailable query gets an intermediate "
        "value written and, with VK_QUERY_RESULT_WITH_AVAILABILITY_BIT, availability 0 (an occlusion query that "
        "was reset and not begun)",
        "vkCreateQueryPool vkCmdResetQueryPool vkGetQueryPoolResults vkDestroyQueryPool vkQueueSubmit")
{
    struct qt q;
    VkQueryPool pool;
    VkCommandBuffer cb;
    uint64_t r[4];
    uint32_t r32[2];
    VkResult res;
    qt_init(t, &q, 0);
    pool = qt_pool(t, &q, VK_QUERY_TYPE_OCCLUSION, 2);
    cb = ct_begin_commands(t, &q.e);
    vkCmdResetQueryPool(cb, pool, 0, 2);
    qt_run(t, &q, cb);
    memset(r, SENTINEL, sizeof(r));
    res = vkGetQueryPoolResults(q.e.device, pool, 0, 2, sizeof(r), r, 16,
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_PARTIAL_BIT |
                                    VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    CT_CHECKF(t, res == VK_NOT_READY || res == VK_SUCCESS, "vkGetQueryPoolResults returned %s", ct_result_name(res));
    CT_CHECK(t, r[1] == 0 && r[3] == 0);
    CT_CHECKF(t, r[0] != SENTINEL64 && r[2] != SENTINEL64, "PARTIAL wrote no intermediate values");
    memset(r32, SENTINEL, sizeof(r32));
    res = vkGetQueryPoolResults(q.e.device, pool, 1, 1, sizeof(r32), r32, 8,
                                VK_QUERY_RESULT_PARTIAL_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    CT_CHECKF(t, res == VK_NOT_READY || res == VK_SUCCESS, "vkGetQueryPoolResults returned %s", ct_result_name(res));
    CT_CHECK(t, r32[1] == 0 && r32[0] != 0xababababu);
    vkDestroyQueryPool(q.e.device, pool, NULL);
    ct_env_fini(&q.e);
}

/* ---- Timestamps ---- */

CT_TEST(timestamp_available,
        "Queries > Timestamp Queries: a timestamp query written by vkCmdWriteTimestamp is available once the "
        "command completed; vkGetQueryPoolResults returns VK_SUCCESS (with WAIT and without) and availability 1",
        "vkCreateQueryPool vkCmdResetQueryPool vkCmdWriteTimestamp vkGetQueryPoolResults vkDestroyQueryPool "
        "vkQueueSubmit vkWaitForFences")
{
    struct qt q;
    VkQueryPool pool;
    VkCommandBuffer cb;
    uint64_t r[4], w[2];
    qt_init(t, &q, 1);
    pool = qt_pool(t, &q, VK_QUERY_TYPE_TIMESTAMP, 2);
    cb = ct_begin_commands(t, &q.e);
    vkCmdResetQueryPool(cb, pool, 0, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 0);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 1);
    qt_run(t, &q, cb);
    memset(r, SENTINEL, sizeof(r));
    CT_CHECK_VK(t,
                vkGetQueryPoolResults(q.e.device, pool, 0, 2, sizeof(r), r, 16,
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
                VK_SUCCESS);
    CT_CHECK(t, r[1] == 1 && r[3] == 1);
    CT_CHECKF(t, qt_ordered(&q, r[0], r[2]), "bottom-of-pipe timestamp %llu before the top-of-pipe one %llu",
              (unsigned long long)r[2], (unsigned long long)r[0]);
    CT_CHECK_VK(t,
                vkGetQueryPoolResults(q.e.device, pool, 0, 2, sizeof(w), w, 8,
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                VK_SUCCESS);
    CT_CHECK(t, w[0] == r[0] && w[1] == r[2]);
    vkDestroyQueryPool(q.e.device, pool, NULL);
    ct_env_fini(&q.e);
}

CT_TEST(timestamp_order,
        "Synchronization and Cache Control > Execution and Memory Dependencies, Queries > Timestamp Queries: "
        "vkCmdWriteTimestamp at VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT waits for all earlier commands in submission "
        "order, so bottom-of-pipe timestamps do not decrease across commands, command buffers and submissions to a "
        "queue",
        "vkCreateQueryPool vkCmdResetQueryPool vkCmdWriteTimestamp vkCmdCopyBuffer vkCmdPipelineBarrier "
        "vkGetQueryPoolResults vkDestroyQueryPool vkQueueSubmit vkWaitForFences vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements vkAllocateMemory vkBindBufferMemory vkFreeMemory")
{
    struct qt q;
    struct qt_scratch a, b;
    VkQueryPool pool;
    uint64_t r[8];
    uint32_t i, s;
    qt_init(t, &q, 1);
    pool = qt_pool(t, &q, VK_QUERY_TYPE_TIMESTAMP, 8);
    qt_scratch(t, &q, 8 << 20, &a);
    qt_scratch(t, &q, 8 << 20, &b);
    for (s = 0; s < 2; s++) { /* two submissions of four timestamps each */
        VkCommandBuffer cb = ct_begin_commands(t, &q.e);
        vkCmdResetQueryPool(cb, pool, 4 * s, 4);
        for (i = 0; i < 4; i++) {
            qt_work(cb, &a, &b, 8 << 20, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 4 * s + i);
        }
        qt_run(t, &q, cb);
    }
    CT_REQUIRE_VK(t, vkGetQueryPoolResults(q.e.device, pool, 0, 8, sizeof(r), r, 8,
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    for (i = 1; i < 8; i++)
        CT_CHECKF(t, qt_ordered(&q, r[i - 1], r[i]), "timestamp %u (%llu) before timestamp %u (%llu)", i,
                  (unsigned long long)r[i], i - 1, (unsigned long long)r[i - 1]);
    qt_scratch_fini(&q, &a);
    qt_scratch_fini(&q, &b);
    vkDestroyQueryPool(q.e.device, pool, NULL);
    ct_env_fini(&q.e);
}

CT_TEST(timestamp_period,
        "Limits > timestampPeriod: the nanoseconds for a timestamp to be incremented by 1. GPU work between a "
        "top-of-pipe and a bottom-of-pipe timestamp, in nanoseconds, takes no longer than the host sees the "
        "submission take, and is not implausibly short (16 copies of 64 MiB)",
        "vkCreateQueryPool vkCmdResetQueryPool vkCmdWriteTimestamp vkCmdCopyBuffer vkCmdPipelineBarrier "
        "vkGetQueryPoolResults vkDestroyQueryPool vkQueueSubmit vkWaitForFences vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements vkAllocateMemory vkBindBufferMemory vkFreeMemory vkGetPhysicalDeviceProperties")
{
    struct qt q;
    struct qt_scratch a, b;
    VkQueryPool pool;
    VkCommandBuffer cb;
    uint64_t r[2], t0, t1;
    double gpu_ns, cpu_ns;
    qt_init(t, &q, 1);
    CT_REQUIRE(t, q.e.props.limits.timestampPeriod > 0.0f);
    pool = qt_pool(t, &q, VK_QUERY_TYPE_TIMESTAMP, 2);
    qt_scratch(t, &q, 64 << 20, &a);
    qt_scratch(t, &q, 64 << 20, &b);
    cb = ct_begin_commands(t, &q.e);
    vkCmdResetQueryPool(cb, pool, 0, 2);
    ct_transfer_barrier(cb);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 0);
    qt_work(cb, &a, &b, 64 << 20, 16);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 1);
    t0 = qt_now_ns();
    qt_run(t, &q, cb);
    t1 = qt_now_ns();
    CT_REQUIRE_VK(t, vkGetQueryPoolResults(q.e.device, pool, 0, 2, sizeof(r), r, 8,
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    gpu_ns = (double)qt_ticks(&q, r[0], r[1]) * q.e.props.limits.timestampPeriod;
    cpu_ns = (double)(t1 - t0);
    ct_note(t, "GPU %.3f ms, host %.3f ms", gpu_ns / 1e6, cpu_ns / 1e6);
    CT_CHECKF(t, gpu_ns <= cpu_ns + 1e6, "the timestamps span %.3f ms, the host saw %.3f ms", gpu_ns / 1e6,
              cpu_ns / 1e6);
    /* 2 GiB moved: at least 0.2 ms below 10 TB/s. */
    CT_CHECKF(t, gpu_ns >= 2e5, "the timestamps span only %.3f ms for 2 GiB of copies", gpu_ns / 1e6);
    qt_scratch_fini(&q, &a);
    qt_scratch_fini(&q, &b);
    vkDestroyQueryPool(q.e.device, pool, NULL);
    ct_env_fini(&q.e);
}

CT_TEST(timestamp_write2,
        "Queries > Timestamp Queries: vkCmdWriteTimestamp2 (synchronization2) writes timestamps that become "
        "available, bottom-of-pipe ones after earlier ones",
        "vkCreateQueryPool vkResetQueryPool vkCmdWriteTimestamp2 vkGetQueryPoolResults vkDestroyQueryPool "
        "vkQueueSubmit vkWaitForFences")
{
    struct qt q;
    VkQueryPool pool;
    VkCommandBuffer cb;
    uint64_t r[6];
    qt_init(t, &q, 1);
    if (q.e.version < VK_API_VERSION_1_3)
        ct_skip(t, "vkCmdWriteTimestamp2 is Vulkan 1.3");
    CT_REQUIREF(t, q.sync2 && q.host_reset, "synchronization2 and hostQueryReset are required by Vulkan 1.3");
    pool = qt_pool(t, &q, VK_QUERY_TYPE_TIMESTAMP, 3);
    vkResetQueryPool(q.e.device, pool, 0, 3);
    cb = ct_begin_commands(t, &q.e);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, pool, 0);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, pool, 1);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool, 2);
    qt_run(t, &q, cb);
    CT_CHECK_VK(t,
                vkGetQueryPoolResults(q.e.device, pool, 0, 3, sizeof(r), r, 16,
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
                VK_SUCCESS);
    CT_CHECK(t, r[1] == 1 && r[3] == 1 && r[5] == 1);
    CT_CHECK(t, qt_ordered(&q, r[2], r[4]) && qt_ordered(&q, r[0], r[4]));
    vkDestroyQueryPool(q.e.device, pool, NULL);
    ct_env_fini(&q.e);
}

CT_TEST(timestamp_secondary,
        "Command Buffers > Secondary Command Buffer Execution, Queries: timestamps written by a secondary command "
        "buffer become available and precede a bottom-of-pipe timestamp the primary writes after it",
        "vkCreateQueryPool vkCmdResetQueryPool vkCmdWriteTimestamp vkCmdExecuteCommands vkAllocateCommandBuffers "
        "vkBeginCommandBuffer vkEndCommandBuffer vkGetQueryPoolResults vkDestroyQueryPool vkQueueSubmit "
        "vkWaitForFences")
{
    struct qt q;
    VkQueryPool pool;
    VkCommandBuffer cb, sec = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferInheritanceInfo inherit;
    VkCommandBufferBeginInfo begin;
    uint64_t r[3];
    qt_init(t, &q, 1);
    pool = qt_pool(t, &q, VK_QUERY_TYPE_TIMESTAMP, 3);
    cb = ct_begin_commands(t, &q.e); /* makes the pool */
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = q.e.pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    alloc.commandBufferCount = 1;
    CT_REQUIRE_VK(t, vkAllocateCommandBuffers(q.e.device, &alloc, &sec));
    memset(&inherit, 0, sizeof(inherit));
    inherit.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.pInheritanceInfo = &inherit;
    CT_REQUIRE_VK(t, vkBeginCommandBuffer(sec, &begin));
    vkCmdWriteTimestamp(sec, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 0);
    vkCmdWriteTimestamp(sec, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 1);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(sec));
    vkCmdResetQueryPool(cb, pool, 0, 3);
    vkCmdExecuteCommands(cb, 1, &sec);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 2);
    qt_run(t, &q, cb);
    CT_CHECK_VK(t, vkGetQueryPoolResults(q.e.device, pool, 0, 3, sizeof(r), r, 8, VK_QUERY_RESULT_64_BIT), VK_SUCCESS);
    CT_CHECK(t, qt_ordered(&q, r[0], r[1]) && qt_ordered(&q, r[1], r[2]));
    vkFreeCommandBuffers(q.e.device, q.e.pool, 1, &sec);
    vkDestroyQueryPool(q.e.device, pool, NULL);
    ct_env_fini(&q.e);
}

CT_TEST(timestamp_rewrite,
        "Queries > Query Operation: a query reset after it became available (on the host, or by the device in "
        "the command buffer that writes it again) is unavailable, then available with the new, later timestamp",
        "vkCreateQueryPool vkResetQueryPool vkCmdResetQueryPool vkCmdWriteTimestamp vkGetQueryPoolResults "
        "vkDestroyQueryPool vkQueueSubmit vkWaitForFences")
{
    struct qt q;
    VkQueryPool pool;
    VkCommandBuffer cb;
    uint64_t r[2], first;
    uint32_t round;
    qt_init(t, &q, 1);
    pool = qt_pool(t, &q, VK_QUERY_TYPE_TIMESTAMP, 1);
    cb = ct_begin_commands(t, &q.e);
    vkCmdResetQueryPool(cb, pool, 0, 1);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 0);
    qt_run(t, &q, cb);
    CT_REQUIRE_VK(t, vkGetQueryPoolResults(q.e.device, pool, 0, 1, 8, &first, 8, VK_QUERY_RESULT_64_BIT));
    for (round = 0; round < 2; round++) {
        if (round == 0 && q.host_reset) {
            vkResetQueryPool(q.e.device, pool, 0, 1);
            memset(r, SENTINEL, sizeof(r));
            CT_CHECK_VK(t,
                        vkGetQueryPoolResults(q.e.device, pool, 0, 1, sizeof(r), r, 16,
                                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
                        VK_NOT_READY);
            CT_CHECK(t, r[0] == SENTINEL64 && r[1] == 0);
        }
        cb = ct_begin_commands(t, &q.e);
        if (round == 1 || !q.host_reset)
            vkCmdResetQueryPool(cb, pool, 0, 1);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 0);
        qt_run(t, &q, cb);
        CT_CHECK_VK(t,
                    vkGetQueryPoolResults(q.e.device, pool, 0, 1, sizeof(r), r, 16,
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
                    VK_SUCCESS);
        CT_CHECK(t, r[1] == 1);
        CT_CHECKF(t, qt_ordered(&q, first, r[0]) && r[0] != first, "round %u: timestamp %llu, earlier one %llu", round,
                  (unsigned long long)r[0], (unsigned long long)first);
        first = r[0];
    }
    vkDestroyQueryPool(q.e.device, pool, NULL);
    ct_env_fini(&q.e);
}

/* ---- Results ---- */

CT_TEST(results_layout,
        "Queries > vkGetQueryPoolResults: query i's results start at pData + i * stride; 32-bit results are the low "
        "bits of 64-bit ones; with VK_QUERY_RESULT_WITH_AVAILABILITY_BIT the availability follows the value in the "
        "same width; bytes between results are not written; unavailable queries get no value (VK_NOT_READY)",
        "vkCreateQueryPool vkCmdResetQueryPool vkCmdWriteTimestamp vkGetQueryPoolResults vkDestroyQueryPool "
        "vkQueueSubmit vkWaitForFences")
{
    struct qt q;
    VkQueryPool pool;
    VkCommandBuffer cb;
    uint64_t ts[4], r64[4 * 3];
    uint32_t r32[4 * 3], i;
    qt_init(t, &q, 1);
    pool = qt_pool(t, &q, VK_QUERY_TYPE_TIMESTAMP, 4);
    cb = ct_begin_commands(t, &q.e);
    vkCmdResetQueryPool(cb, pool, 0, 4);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 0);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 2);
    qt_run(t, &q, cb);
    CT_REQUIRE_VK(t, vkGetQueryPoolResults(q.e.device, pool, 0, 1, 8, &ts[0], 0, VK_QUERY_RESULT_64_BIT));
    CT_REQUIRE_VK(t, vkGetQueryPoolResults(q.e.device, pool, 2, 1, 8, &ts[2], 0, VK_QUERY_RESULT_64_BIT));

    memset(r64, SENTINEL, sizeof(r64));
    CT_CHECK_VK(t,
                vkGetQueryPoolResults(q.e.device, pool, 0, 4, sizeof(r64), r64, 24,
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
                VK_NOT_READY);
    for (i = 0; i < 4; i++) {
        CT_CHECKF(t, r64[3 * i] == (i % 2 ? SENTINEL64 : ts[i]), "64-bit stride 24: query %u value", i);
        CT_CHECKF(t, r64[3 * i + 1] == (i % 2 ? 0u : 1u), "64-bit stride 24: query %u availability", i);
        CT_CHECKF(t, r64[3 * i + 2] == SENTINEL64, "64-bit stride 24: query %u: padding written", i);
    }
    memset(r32, SENTINEL, sizeof(r32));
    CT_CHECK_VK(t,
                vkGetQueryPoolResults(q.e.device, pool, 0, 4, sizeof(r32), r32, 12,
                                      VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
                VK_NOT_READY);
    for (i = 0; i < 4; i++) {
        CT_CHECKF(t, r32[3 * i] == (i % 2 ? 0xababababu : (uint32_t)ts[i]), "32-bit stride 12: query %u value", i);
        CT_CHECKF(t, r32[3 * i + 1] == (i % 2 ? 0u : 1u), "32-bit stride 12: query %u availability", i);
        CT_CHECKF(t, r32[3 * i + 2] == 0xababababu, "32-bit stride 12: query %u: padding written", i);
    }
    memset(r32, SENTINEL, sizeof(r32));
    CT_CHECK_VK(t, vkGetQueryPoolResults(q.e.device, pool, 0, 4, 16, r32, 4, 0), VK_NOT_READY);
    for (i = 0; i < 4; i++)
        CT_CHECKF(t, r32[i] == (i % 2 ? 0xababababu : (uint32_t)ts[i]), "32-bit stride 4: query %u value", i);
    CT_CHECK(t, r32[4] == 0xababababu);
    vkDestroyQueryPool(q.e.device, pool, NULL);
    ct_env_fini(&q.e);
}

struct qt_release {
    VkDevice device;
    VkSemaphore semaphore;
};

static void *qt_release_later(void *arg)
{
    struct qt_release *rel = arg;
    VkSemaphoreSignalInfo info;
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 50000000;
    nanosleep(&ts, NULL);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    info.semaphore = rel->semaphore;
    info.value = 1;
    vkSignalSemaphore(rel->device, &info);
    return NULL;
}

CT_TEST(results_wait,
        "Queries > vkGetQueryPoolResults: with VK_QUERY_RESULT_WAIT_BIT the command waits until the queries are "
        "available and returns VK_SUCCESS (a submission held back by a timeline semaphore that another thread "
        "signals 50 ms later)",
        "vkCreateQueryPool vkCmdResetQueryPool vkCmdWriteTimestamp vkGetQueryPoolResults vkDestroyQueryPool "
        "vkQueueSubmit vkQueueWaitIdle vkCreateSemaphore vkDestroySemaphore vkSignalSemaphore vkEndCommandBuffer "
        "vkFreeCommandBuffers")
{
    struct qt q;
    VkSemaphoreTypeCreateInfo type;
    VkSemaphoreCreateInfo sem_info;
    VkTimelineSemaphoreSubmitInfo values;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit;
    struct qt_release rel;
    VkSemaphore gate;
    VkQueryPool pool;
    VkCommandBuffer cb;
    pthread_t thread;
    uint64_t r[4], one = 1, t0;
    qt_init(t, &q, 1);
    if (q.e.version < VK_API_VERSION_1_2)
        ct_skip(t, "timeline semaphores are Vulkan 1.2");
    CT_REQUIREF(t, q.timeline, "timelineSemaphore is required by Vulkan 1.2");
    memset(&type, 0, sizeof(type));
    type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    memset(&sem_info, 0, sizeof(sem_info));
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sem_info.pNext = &type;
    CT_REQUIRE_VK(t, vkCreateSemaphore(q.e.device, &sem_info, NULL, &gate));
    pool = qt_pool(t, &q, VK_QUERY_TYPE_TIMESTAMP, 2);
    cb = ct_begin_commands(t, &q.e);
    vkCmdResetQueryPool(cb, pool, 0, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 0);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 1);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    memset(&values, 0, sizeof(values));
    values.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    values.waitSemaphoreValueCount = 1;
    values.pWaitSemaphoreValues = &one;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = &values;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &gate;
    submit.pWaitDstStageMask = &stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    CT_REQUIRE_VK(t, vkQueueSubmit(q.e.queue, 1, &submit, VK_NULL_HANDLE));
    rel.device = q.e.device;
    rel.semaphore = gate;
    t0 = qt_now_ns();
    CT_REQUIRE(t, pthread_create(&thread, NULL, qt_release_later, &rel) == 0);
    memset(r, SENTINEL, sizeof(r));
    CT_CHECK_VK(t,
                vkGetQueryPoolResults(q.e.device, pool, 0, 2, sizeof(r), r, 16,
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT |
                                          VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
                VK_SUCCESS);
    CT_CHECKF(t, qt_now_ns() - t0 >= 40000000, "returned before the GPU could have written the queries");
    CT_CHECK(t, r[1] == 1 && r[3] == 1 && qt_ordered(&q, r[0], r[2]));
    pthread_join(thread, NULL);
    CT_CHECK_VK(t, vkQueueWaitIdle(q.e.queue), VK_SUCCESS);
    vkFreeCommandBuffers(q.e.device, q.e.pool, 1, &cb);
    vkDestroySemaphore(q.e.device, gate, NULL);
    vkDestroyQueryPool(q.e.device, pool, NULL);
    ct_env_fini(&q.e);
}

/* ---- Copies ---- */

struct qt_copy {
    VkQueryResultFlags flags;
    VkDeviceSize offset, stride;
};

/* Copies the results of 4 timestamps with each case of flags, offset and
 * stride, and checks them against vkGetQueryPoolResults. mode 0: in the
 * command buffer that wrote them, right after, every case with
 * VK_QUERY_RESULT_WAIT_BIT; 1: there, after an execution dependency on the
 * writes; 2: in a later submission. */
static void copy_results_mode(struct ct *t, uint32_t mode)
{
    static const struct qt_copy cases[] = {
        { 0, 0, 4 },
        { 0, 4, 12 },
        { VK_QUERY_RESULT_64_BIT, 8, 8 },
        { VK_QUERY_RESULT_64_BIT, 16, 24 },
        { VK_QUERY_RESULT_WITH_AVAILABILITY_BIT, 4, 8 },
        { VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT, 8, 16 },
        { VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT, 0, 40 },
        { VK_QUERY_RESULT_WAIT_BIT, 0, 4 },
        { VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT, 8, 16 },
    };
    enum { N = 4, SLOT = 256 };
    struct qt q;
    struct ct_buffer out;
    VkQueryPool pool;
    VkCommandBuffer cb;
    uint64_t ts[N];
    uint32_t c, i;
    qt_init(t, &q, 1);
    pool = qt_pool(t, &q, VK_QUERY_TYPE_TIMESTAMP, N);
    ct_host_buffer(t, &q.e, CT_ARRAY_SIZE(cases) * SLOT, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &out);
    memset(out.data, SENTINEL, (size_t)out.size);
    cb = ct_begin_commands(t, &q.e);
    vkCmdResetQueryPool(cb, pool, 0, N);
    for (i = 0; i < N; i++)
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, i);
    if (mode == 1)
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0,
                             NULL, 0, NULL);
    if (mode == 2) {
        qt_run(t, &q, cb);
        cb = ct_begin_commands(t, &q.e);
    }
    for (c = 0; c < CT_ARRAY_SIZE(cases); c++)
        vkCmdCopyQueryPoolResults(cb, pool, 0, N, out.buffer, c * SLOT + cases[c].offset, cases[c].stride,
                                  cases[c].flags | (mode == 0 ? VK_QUERY_RESULT_WAIT_BIT : 0));
    ct_host_read_barrier(cb);
    qt_run(t, &q, cb);
    CT_REQUIRE_VK(t, vkGetQueryPoolResults(q.e.device, pool, 0, N, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT));
    for (c = 0; c < CT_ARRAY_SIZE(cases); c++) {
        const struct qt_copy *k = &cases[c];
        int wide = (k->flags & VK_QUERY_RESULT_64_BIT) != 0;
        int avail = (k->flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != 0;
        const unsigned char *base = out.data + c * SLOT;
        size_t size = (wide ? 8u : 4u) * (avail ? 2u : 1u), b;
        int untouched = 1;
        for (b = 0; b < k->offset; b++)
            untouched &= base[b] == SENTINEL;
        for (i = 0; i < N; i++) {
            const unsigned char *p = base + k->offset + k->stride * i;
            uint64_t v = 0, a = 0, want = wide ? ts[i] : (uint32_t)ts[i];
            memcpy(&v, p, wide ? 8 : 4);
            if (avail)
                memcpy(&a, p + (wide ? 8 : 4), wide ? 8 : 4);
            CT_CHECKF(t, v == want, "case %u, query %u: value %llx, expected %llx", c, i, (unsigned long long)v,
                      (unsigned long long)want);
            CT_CHECKF(t, !avail || a == 1, "case %u, query %u: availability %llu", c, i, (unsigned long long)a);
            for (b = size; b < k->stride && i + 1 < N; b++)
                untouched &= p[b] == SENTINEL;
        }
        for (b = k->offset + k->stride * (N - 1) + size; b < SLOT; b++)
            untouched &= base[b] == SENTINEL;
        CT_CHECKF(t, untouched, "case %u: bytes outside the results were written", c);
    }
    ct_buffer_fini(&q.e, &out);
    vkDestroyQueryPool(q.e.device, pool, NULL);
    ct_env_fini(&q.e);
}

#define COPY_COMMANDS                                                                                             \
    "vkCreateQueryPool vkCmdResetQueryPool vkCmdWriteTimestamp vkCmdCopyQueryPoolResults vkCmdPipelineBarrier "   \
    "vkGetQueryPoolResults vkDestroyQueryPool vkQueueSubmit vkWaitForFences vkCreateBuffer vkDestroyBuffer "      \
    "vkGetBufferMemoryRequirements vkAllocateMemory vkBindBufferMemory vkMapMemory vkFreeMemory"

CT_TEST(copy_results,
        "Queries > vkCmdCopyQueryPoolResults: the results of available queries are written to dstBuffer from "
        "dstOffset, query i at i * stride, in 32 or 64 bits, followed by the availability with "
        "VK_QUERY_RESULT_WITH_AVAILABILITY_BIT; the same values vkGetQueryPoolResults returns; nothing else is "
        "written. Copied in a later submission than the one that wrote the queries",
        COPY_COMMANDS)
{
    copy_results_mode(t, 2);
}

CT_TEST(copy_results_same_commands,
        "Queries > vkCmdCopyQueryPoolResults: as copy_results, copied in the command buffer that wrote the queries, "
        "after an execution dependency on the writes (ALL_COMMANDS to TRANSFER)",
        COPY_COMMANDS)
{
    copy_results_mode(t, 1);
}

CT_TEST(copy_results_wait,
        "Queries > vkCmdCopyQueryPoolResults: \"If VK_QUERY_RESULT_WAIT_BIT is set, the implementation will wait for "
        "each query's status to be in the available state before retrieving the numerical results\": copied with "
        "WAIT in the command buffer that wrote the queries, right after the writes (earlier in submission order, "
        "VUID-vkCmdCopyQueryPoolResults-None-08752), the results are those of copy_results",
        COPY_COMMANDS)
{
    copy_results_mode(t, 0);
}

static const struct ct_test *const tests[] = {
    CT_REF(create_query_pools),     CT_REF(host_reset_unavailable), CT_REF(cmd_reset_unavailable),
    CT_REF(occlusion_partial_unavailable), CT_REF(timestamp_available), CT_REF(timestamp_order),
    CT_REF(timestamp_period),       CT_REF(timestamp_write2),       CT_REF(timestamp_secondary),
    CT_REF(timestamp_rewrite),      CT_REF(results_layout),         CT_REF(results_wait),
    CT_REF(copy_results),           CT_REF(copy_results_same_commands), CT_REF(copy_results_wait),
};

CT_SUITE(queries, tests)
