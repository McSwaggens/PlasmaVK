/* Chapter "Command Buffers": resetting command pools and command buffers
 * (what a command buffer holds after it returns to the initial state),
 * trimming pools, vkQueueSubmit2 (batches, semaphores, fences) and the
 * device mask of a command buffer. Command buffers fill regions of a host
 * buffer, so which commands ran shows in which regions changed. */
#include "ct.h"

#include <time.h>

#define REGION 4096 /* bytes per region of the result buffer */
#define REGIONS 4

#define CB_COMMANDS                                                                                                \
    "vkCreateCommandPool vkDestroyCommandPool vkAllocateCommandBuffers vkFreeCommandBuffers vkBeginCommandBuffer " \
    "vkEndCommandBuffer vkCmdFillBuffer vkCmdPipelineBarrier vkQueueSubmit vkQueueWaitIdle vkCreateBuffer "        \
    "vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory vkBindBufferMemory vkMapMemory vkFreeMemory "

static VkCommandPool cb_pool(struct ct *t, struct ct_env *e, VkCommandPoolCreateFlags flags)
{
    VkCommandPoolCreateInfo info;
    VkCommandPool pool = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.flags = flags;
    info.queueFamilyIndex = e->queue_family;
    CT_REQUIRE_VK(t, vkCreateCommandPool(e->device, &info, NULL, &pool));
    return pool;
}

static void cb_alloc(struct ct *t, struct ct_env *e, VkCommandPool pool, uint32_t n, VkCommandBuffer *cbs)
{
    VkCommandBufferAllocateInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = pool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = n;
    CT_REQUIRE_VK(t, vkAllocateCommandBuffers(e->device, &info, cbs));
}

/* Begins (reusable: no ONE_TIME_SUBMIT), fills `region` with `value` and,
 * unless `open`, makes the write visible to the host and ends. */
static void cb_record(struct ct *t, VkCommandBuffer cb, const struct ct_buffer *b, uint32_t region, uint32_t value,
                      int open)
{
    VkCommandBufferBeginInfo begin;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    CT_REQUIRE_VK(t, vkBeginCommandBuffer(cb, &begin));
    vkCmdFillBuffer(cb, b->buffer, (VkDeviceSize)region * REGION, REGION, value);
    if (open)
        return;
    ct_host_read_barrier(cb);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
}

static void cb_submit(struct ct *t, struct ct_env *e, uint32_t n, const VkCommandBuffer *cbs)
{
    VkSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = n;
    submit.pCommandBuffers = cbs;
    CT_REQUIRE_VK(t, vkQueueSubmit(e->queue, 1, &submit, VK_NULL_HANDLE));
    CT_REQUIRE_VK(t, vkQueueWaitIdle(e->queue));
}

/* The words of each region: what they all are, or 0xdeadbeef if they differ. */
static uint32_t cb_region(const struct ct_buffer *b, uint32_t region)
{
    const unsigned char *p = b->data + (size_t)region * REGION;
    uint32_t first, w;
    size_t i;
    memcpy(&first, p, 4);
    for (i = 4; i < REGION; i += 4) {
        memcpy(&w, p + i, 4);
        if (w != first)
            return 0xdeadbeefu;
    }
    return first;
}

/* Every region is as `want` says. */
static void cb_check(struct ct *t, const struct ct_buffer *b, const uint32_t want[REGIONS], const char *what)
{
    uint32_t i;
    for (i = 0; i < REGIONS; i++)
        CT_CHECKF(t, cb_region(b, i) == want[i], "%s: region %u is 0x%08x, expected 0x%08x", what, i,
                  cb_region(b, i), want[i]);
}

CT_TEST(reset_command_pool,
        "Command Buffers > Command Pools (vkResetCommandPool): resetting a pool puts every command buffer "
        "allocated from it in the initial state, executed or only recorded, with flags 0 or "
        "VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT: begun again (valid without RESET_COMMAND_BUFFER_BIT only in "
        "the initial state, VUID-vkBeginCommandBuffer-commandBuffer-00050), they hold only the new commands",
        CB_COMMANDS "vkResetCommandPool")
{
    struct ct_env e;
    struct ct_buffer b;
    VkCommandPool pool;
    VkCommandBuffer cbs[2];
    uint32_t want[REGIONS] = { 0, 0, 0, 0 };
    VkCommandPoolResetFlags flags;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    ct_host_buffer(t, &e, REGION * REGIONS, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &b);
    pool = cb_pool(t, &e, 0);
    cb_alloc(t, &e, pool, 2, cbs);
    cb_record(t, cbs[0], &b, 0, 0x11111111u, 0);
    cb_record(t, cbs[1], &b, 1, 0x22222222u, 0); /* recorded, never submitted */
    cb_submit(t, &e, 1, &cbs[0]);
    want[0] = 0x11111111u;
    cb_check(t, &b, want, "before the reset");
    for (flags = 0; flags <= VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT; flags += VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT) {
        uint32_t v = flags ? 0x55555555u : 0x33333333u;
        memset(b.data, 0, REGION * REGIONS);
        CT_CHECK_VK(t, vkResetCommandPool(e.device, pool, flags), VK_SUCCESS);
        cb_record(t, cbs[0], &b, 2, v, 0);
        cb_record(t, cbs[1], &b, 3, v + 0x11111111u, 0);
        cb_submit(t, &e, 2, cbs);
        want[0] = want[1] = 0;
        want[2] = v;
        want[3] = v + 0x11111111u;
        cb_check(t, &b, want, flags ? "after a reset releasing resources" : "after a reset");
    }
    vkFreeCommandBuffers(e.device, pool, 2, cbs);
    vkDestroyCommandPool(e.device, pool, NULL);
    ct_buffer_fini(&e, &b);
    ct_env_fini(&e);
}

CT_TEST(reset_command_buffer,
        "Command Buffers > Command Buffer Allocation and Management (vkResetCommandBuffer): a command buffer in the recording, "
        "executable or invalid state (not pending) moves to the initial state, with flags 0 or "
        "VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT; its earlier commands are no longer executed (\"must no "
        "longer be accessed\"), and other command buffers of the pool keep theirs",
        CB_COMMANDS "vkResetCommandBuffer")
{
    struct ct_env e;
    struct ct_buffer b;
    VkCommandPool pool;
    VkCommandBuffer cbs[2];
    VkCommandBufferBeginInfo once;
    uint32_t want[REGIONS] = { 0, 0, 0, 0 };

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    ct_host_buffer(t, &e, REGION * REGIONS, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &b);
    pool = cb_pool(t, &e, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    cb_alloc(t, &e, pool, 2, cbs);

    /* executable, never submitted */
    cb_record(t, cbs[0], &b, 0, 0x11111111u, 0);
    CT_CHECK_VK(t, vkResetCommandBuffer(cbs[0], 0), VK_SUCCESS);
    cb_record(t, cbs[0], &b, 1, 0x22222222u, 0);
    cb_record(t, cbs[1], &b, 3, 0x66666666u, 0);
    cb_submit(t, &e, 1, &cbs[0]);
    want[1] = 0x22222222u;
    cb_check(t, &b, want, "reset when executable");

    /* executable again after its submission completed */
    memset(b.data, 0, REGION * REGIONS);
    CT_CHECK_VK(t, vkResetCommandBuffer(cbs[0], VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT), VK_SUCCESS);
    cb_record(t, cbs[0], &b, 2, 0x33333333u, 0);
    cb_submit(t, &e, 1, &cbs[0]);
    want[1] = 0;
    want[2] = 0x33333333u;
    cb_check(t, &b, want, "reset after execution, releasing resources");

    /* recording */
    memset(b.data, 0, REGION * REGIONS);
    cb_record(t, cbs[0], &b, 2, 0x44444444u, 1);
    CT_CHECK_VK(t, vkResetCommandBuffer(cbs[0], 0), VK_SUCCESS);
    cb_record(t, cbs[0], &b, 0, 0x55555555u, 0);
    cb_submit(t, &e, 1, &cbs[0]);
    want[0] = 0x55555555u;
    want[2] = 0;
    cb_check(t, &b, want, "reset while recording");

    /* invalid: recorded with ONE_TIME_SUBMIT and submitted */
    memset(b.data, 0, REGION * REGIONS);
    memset(&once, 0, sizeof(once));
    once.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    once.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CT_REQUIRE_VK(t, vkBeginCommandBuffer(cbs[0], &once));
    vkCmdFillBuffer(cbs[0], b.buffer, 0, REGION, 0x77777777u);
    ct_host_read_barrier(cbs[0]);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cbs[0]));
    cb_submit(t, &e, 1, &cbs[0]);
    CT_CHECK_VK(t, vkResetCommandBuffer(cbs[0], 0), VK_SUCCESS);
    cb_record(t, cbs[0], &b, 1, 0x88888888u, 0);
    memset(b.data, 0, REGION * REGIONS);
    /* with the other command buffer of the pool, recorded at the start */
    cb_submit(t, &e, 2, cbs);
    want[0] = 0;
    want[1] = 0x88888888u;
    want[3] = 0x66666666u;
    cb_check(t, &b, want, "reset when invalid; the pool's other command buffer");

    vkFreeCommandBuffers(e.device, pool, 2, cbs);
    vkDestroyCommandPool(e.device, pool, NULL);
    ct_buffer_fini(&e, &b);
    ct_env_fini(&e);
}

CT_TEST(trim_command_pool,
        "Command Buffers > Command Pools (vkTrimCommandPool, Vulkan 1.1): trimming recycles unused memory and "
        "\"command buffers allocated from the pool are not affected\": command buffers recorded before still "
        "execute their commands, and command buffers can be reset, recorded and allocated after it",
        CB_COMMANDS "vkTrimCommandPool vkResetCommandBuffer")
{
    struct ct_env e;
    struct ct_buffer b;
    VkCommandPool pool, empty;
    VkCommandBuffer cbs[4], extra;
    uint32_t want[REGIONS] = { 0x10101010u, 0x20202020u, 0x30303030u, 0x40404040u };
    uint32_t i;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_1)
        ct_skip(t, "vkTrimCommandPool is Vulkan 1.1");
    ct_env_device(t, &e, NULL);
    ct_host_buffer(t, &e, REGION * REGIONS, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &b);
    empty = cb_pool(t, &e, 0);
    vkTrimCommandPool(e.device, empty, 0); /* nothing allocated */
    vkDestroyCommandPool(e.device, empty, NULL);
    pool = cb_pool(t, &e, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    cb_alloc(t, &e, pool, 4, cbs);
    for (i = 0; i < 4; i++)
        cb_record(t, cbs[i], &b, i, i < 2 ? want[i] : 0xdead0000u + i, 0);
    /* memory left unused: a freed command buffer and a reset one */
    vkFreeCommandBuffers(e.device, pool, 1, &cbs[3]);
    CT_CHECK_VK(t, vkResetCommandBuffer(cbs[2], VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT), VK_SUCCESS);
    for (i = 0; i < 2; i++) {
        vkTrimCommandPool(e.device, pool, 0);
        vkTrimCommandPool(e.device, pool, 0);
        memset(b.data, 0, REGION * REGIONS);
        /* cbs[0] and cbs[1] were recorded before the trims; cbs[2] is in the
         * initial state (then reset by the begin) */
        cb_record(t, cbs[2], &b, 2, want[2], 0);
        cb_alloc(t, &e, pool, 1, &extra);
        cb_record(t, extra, &b, 3, want[3], 0);
        cb_submit(t, &e, 3, cbs);
        cb_submit(t, &e, 1, &extra);
        cb_check(t, &b, want, i ? "after the second trims" : "after the first trims");
        vkFreeCommandBuffers(e.device, pool, 1, &extra);
    }
    vkFreeCommandBuffers(e.device, pool, 3, cbs);
    vkTrimCommandPool(e.device, pool, 0);
    vkDestroyCommandPool(e.device, pool, NULL);
    ct_buffer_fini(&e, &b);
    ct_env_fini(&e);
}

/* ---- vkQueueSubmit2 ---- */

/* A device with timelineSemaphore and synchronization2 (Vulkan 1.3). */
static void cb_env_sync2(struct ct *t, struct ct_env *e)
{
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    struct ct_features f;
    ct_env_init(t, e);
    if (e->version < VK_API_VERSION_1_3)
        ct_skip(t, "vkQueueSubmit2 is Vulkan 1.3");
    ct_get_features(e, &f, 0);
    CT_REQUIREF(t, f.v12.timelineSemaphore && f.v13.synchronization2,
                "timelineSemaphore and synchronization2 are required by Vulkan 1.3");
    memset(&v13, 0, sizeof(v13));
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13.synchronization2 = VK_TRUE;
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.pNext = &v13;
    v12.timelineSemaphore = VK_TRUE;
    ct_env_device(t, e, &v12);
}

static VkSemaphore cb_semaphore(struct ct *t, struct ct_env *e, int timeline, uint64_t initial)
{
    VkSemaphoreTypeCreateInfo type;
    VkSemaphoreCreateInfo info;
    VkSemaphore s = VK_NULL_HANDLE;
    memset(&type, 0, sizeof(type));
    type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type.initialValue = initial;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    info.pNext = timeline ? &type : NULL;
    CT_REQUIRE_VK(t, vkCreateSemaphore(e->device, &info, NULL, &s));
    return s;
}

static VkFence cb_fence(struct ct *t, struct ct_env *e)
{
    VkFenceCreateInfo info;
    VkFence f = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreateFence(e->device, &info, NULL, &f));
    return f;
}

static VkSemaphoreSubmitInfo cb_sem_info(VkSemaphore s, uint64_t value, VkPipelineStageFlags2 stages)
{
    VkSemaphoreSubmitInfo i;
    memset(&i, 0, sizeof(i));
    i.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    i.semaphore = s;
    i.value = value;
    i.stageMask = stages;
    return i;
}

static VkCommandBufferSubmitInfo cb_cb_info(VkCommandBuffer cb, uint32_t device_mask)
{
    VkCommandBufferSubmitInfo i;
    memset(&i, 0, sizeof(i));
    i.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    i.commandBuffer = cb;
    i.deviceMask = device_mask;
    return i;
}

static void cb_copy(VkCommandBuffer cb, VkBuffer src, VkBuffer dst, VkDeviceSize size)
{
    VkBufferCopy region;
    memset(&region, 0, sizeof(region));
    region.size = size;
    vkCmdCopyBuffer(cb, src, dst, 1, &region);
}

CT_TEST(queue_submit2,
        "Command Buffers > Command Buffer Submission (vkQueueSubmit2, synchronization2): batches run their command buffers "
        "(VkCommandBufferSubmitInfo, deviceMask 0 meaning every device, or 1); a batch's semaphore wait "
        "(VkSemaphoreSubmitInfo) orders its commands after the earlier batch's binary semaphore signal, limited to "
        "stageMask; a timeline signal sets its value; a batch waiting on a timeline value the host signals later "
        "does not run before; the fence of a submission of no batches signals after all earlier commands "
        "(Synchronization > Fences: its first synchronization scope)",
        "vkQueueSubmit2 vkCreateSemaphore vkDestroySemaphore vkSignalSemaphore vkGetSemaphoreCounterValue "
        "vkCreateFence vkDestroyFence vkGetFenceStatus vkWaitForFences vkCmdCopyBuffer vkCmdFillBuffer "
        "vkCmdPipelineBarrier vkCreateCommandPool vkDestroyCommandPool vkAllocateCommandBuffers vkFreeCommandBuffers "
        "vkBeginCommandBuffer vkEndCommandBuffer vkQueueWaitIdle vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements vkAllocateMemory vkBindBufferMemory vkMapMemory vkFreeMemory")
{
    struct ct_env e;
    struct ct_buffer src, mid, dst, late;
    VkCommandPool pool;
    VkCommandBuffer cbs[3];
    VkCommandBufferBeginInfo begin;
    VkSemaphore binary, timeline, gate;
    VkSemaphoreSubmitInfo signal_binary, wait_binary, signal_timeline, wait_gate;
    VkCommandBufferSubmitInfo cbi[3];
    VkSemaphoreSignalInfo host_signal;
    VkSubmitInfo2 batches[2];
    VkFence fence, late_fence;
    uint64_t value = 0;
    struct timespec ts;
    uint32_t i;
    int same = 1;

    cb_env_sync2(t, &e);
    ct_host_buffer(t, &e, 1 << 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src);
    ct_host_buffer(t, &e, 1 << 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mid);
    ct_host_buffer(t, &e, 1 << 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst);
    ct_host_buffer(t, &e, REGION, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &late);
    ct_fill_pattern(src.data, 1 << 16, 3);
    pool = cb_pool(t, &e, 0);
    cb_alloc(t, &e, pool, 3, cbs);
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CT_REQUIRE_VK(t, vkBeginCommandBuffer(cbs[0], &begin));
    cb_copy(cbs[0], src.buffer, mid.buffer, 1 << 16);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cbs[0]));
    CT_REQUIRE_VK(t, vkBeginCommandBuffer(cbs[1], &begin));
    cb_copy(cbs[1], mid.buffer, dst.buffer, 1 << 16);
    ct_host_read_barrier(cbs[1]);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cbs[1]));
    CT_REQUIRE_VK(t, vkBeginCommandBuffer(cbs[2], &begin));
    vkCmdFillBuffer(cbs[2], late.buffer, 0, VK_WHOLE_SIZE, 0x1a7e1a7eu);
    ct_host_read_barrier(cbs[2]);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cbs[2]));

    binary = cb_semaphore(t, &e, 0, 0);
    timeline = cb_semaphore(t, &e, 1, 0);
    gate = cb_semaphore(t, &e, 1, 0);
    fence = cb_fence(t, &e);
    late_fence = cb_fence(t, &e);
    signal_binary = cb_sem_info(binary, 0, VK_PIPELINE_STAGE_2_COPY_BIT);
    wait_binary = cb_sem_info(binary, 0, VK_PIPELINE_STAGE_2_COPY_BIT);
    signal_timeline = cb_sem_info(timeline, 5, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    wait_gate = cb_sem_info(gate, 1, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    cbi[0] = cb_cb_info(cbs[0], 0);
    cbi[1] = cb_cb_info(cbs[1], 1);
    cbi[2] = cb_cb_info(cbs[2], 0);
    memset(batches, 0, sizeof(batches));
    for (i = 0; i < 2; i++) {
        batches[i].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        batches[i].commandBufferInfoCount = 1;
        batches[i].pCommandBufferInfos = &cbi[i];
    }
    batches[0].signalSemaphoreInfoCount = 1;
    batches[0].pSignalSemaphoreInfos = &signal_binary;
    batches[1].waitSemaphoreInfoCount = 1;
    batches[1].pWaitSemaphoreInfos = &wait_binary;
    batches[1].signalSemaphoreInfoCount = 1;
    batches[1].pSignalSemaphoreInfos = &signal_timeline;
    CT_REQUIRE_VK(t, vkQueueSubmit2(e.queue, 2, batches, VK_NULL_HANDLE));
    CT_REQUIRE_VK(t, vkQueueSubmit2(e.queue, 0, NULL, fence));
    CT_REQUIRE_VK(t, vkWaitForFences(e.device, 1, &fence, VK_TRUE, 10000000000ull));
    CT_CHECK(t, memcmp(dst.data, src.data, 1 << 16) == 0);
    CT_CHECK_VK(t, vkGetSemaphoreCounterValue(e.device, timeline, &value), VK_SUCCESS);
    CT_CHECKF(t, value == 5, "timeline semaphore value %llu after the signal of 5", (unsigned long long)value);

    /* a batch waiting on a value the host signals later */
    memset(batches, 0, sizeof(batches));
    batches[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    batches[0].waitSemaphoreInfoCount = 1;
    batches[0].pWaitSemaphoreInfos = &wait_gate;
    batches[0].commandBufferInfoCount = 1;
    batches[0].pCommandBufferInfos = &cbi[2];
    CT_REQUIRE_VK(t, vkQueueSubmit2(e.queue, 1, batches, late_fence));
    ts.tv_sec = 0;
    ts.tv_nsec = 20000000;
    nanosleep(&ts, NULL);
    CT_CHECK_VK(t, vkGetFenceStatus(e.device, late_fence), VK_NOT_READY);
    for (i = 0; i < REGION; i++)
        same &= late.data[i] == 0;
    CT_CHECKF(t, same, "the batch ran before its semaphore wait was satisfied");
    memset(&host_signal, 0, sizeof(host_signal));
    host_signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    host_signal.semaphore = gate;
    host_signal.value = 1;
    CT_REQUIRE_VK(t, vkSignalSemaphore(e.device, &host_signal));
    CT_CHECK_VK(t, vkWaitForFences(e.device, 1, &late_fence, VK_TRUE, 10000000000ull), VK_SUCCESS);
    CT_CHECK(t, cb_region(&late, 0) == 0x1a7e1a7eu);

    CT_REQUIRE_VK(t, vkQueueWaitIdle(e.queue));
    vkDestroyFence(e.device, fence, NULL);
    vkDestroyFence(e.device, late_fence, NULL);
    vkDestroySemaphore(e.device, binary, NULL);
    vkDestroySemaphore(e.device, timeline, NULL);
    vkDestroySemaphore(e.device, gate, NULL);
    vkFreeCommandBuffers(e.device, pool, 3, cbs);
    vkDestroyCommandPool(e.device, pool, NULL);
    ct_buffer_fini(&e, &src);
    ct_buffer_fini(&e, &mid);
    ct_buffer_fini(&e, &dst);
    ct_buffer_fini(&e, &late);
    ct_env_fini(&e);
}

/* ---- Device mask ---- */

CT_TEST(set_device_mask,
        "Command Buffers > Command Buffer Device Mask (vkCmdSetDeviceMask, Vulkan 1.1): commands after it execute "
        "on the physical devices whose bits are set: on a device of one physical device (mask 1, within the "
        "VkDeviceGroupCommandBufferBeginInfo::deviceMask the command buffer began with, "
        "VUID-vkCmdSetDeviceMask-deviceMask-00110, or within all devices without it) they execute, before and "
        "after it; VkDeviceGroupSubmitInfo submits it with device mask 1",
        CB_COMMANDS "vkCmdSetDeviceMask")
{
    struct ct_env e;
    struct ct_buffer b;
    VkCommandPool pool;
    VkCommandBuffer cbs[2];
    VkDeviceGroupCommandBufferBeginInfo group_begin;
    VkCommandBufferBeginInfo begin;
    VkDeviceGroupSubmitInfo group_submit;
    VkSubmitInfo submit;
    uint32_t masks[2] = { 1, 1 };
    uint32_t want[REGIONS] = { 0x01010101u, 0x02020202u, 0x03030303u, 0x04040404u };

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_1)
        ct_skip(t, "vkCmdSetDeviceMask is Vulkan 1.1");
    ct_env_device(t, &e, NULL);
    ct_host_buffer(t, &e, REGION * REGIONS, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &b);
    pool = cb_pool(t, &e, 0);
    cb_alloc(t, &e, pool, 2, cbs);
    memset(&group_begin, 0, sizeof(group_begin));
    group_begin.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO;
    group_begin.deviceMask = 1;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.pNext = &group_begin;
    CT_REQUIRE_VK(t, vkBeginCommandBuffer(cbs[0], &begin));
    vkCmdFillBuffer(cbs[0], b.buffer, 0, REGION, want[0]);
    vkCmdSetDeviceMask(cbs[0], 1);
    vkCmdFillBuffer(cbs[0], b.buffer, REGION, REGION, want[1]);
    ct_host_read_barrier(cbs[0]);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cbs[0]));
    begin.pNext = NULL; /* initial device mask: every physical device of the device */
    CT_REQUIRE_VK(t, vkBeginCommandBuffer(cbs[1], &begin));
    vkCmdSetDeviceMask(cbs[1], 1);
    vkCmdFillBuffer(cbs[1], b.buffer, 2 * REGION, REGION, want[2]);
    vkCmdSetDeviceMask(cbs[1], 1);
    vkCmdFillBuffer(cbs[1], b.buffer, 3 * REGION, REGION, want[3]);
    ct_host_read_barrier(cbs[1]);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cbs[1]));
    memset(&group_submit, 0, sizeof(group_submit));
    group_submit.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO;
    group_submit.commandBufferCount = 2;
    group_submit.pCommandBufferDeviceMasks = masks;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = &group_submit;
    submit.commandBufferCount = 2;
    submit.pCommandBuffers = cbs;
    CT_REQUIRE_VK(t, vkQueueSubmit(e.queue, 1, &submit, VK_NULL_HANDLE));
    CT_REQUIRE_VK(t, vkQueueWaitIdle(e.queue));
    cb_check(t, &b, want, "fills before and after vkCmdSetDeviceMask(1)");
    vkFreeCommandBuffers(e.device, pool, 2, cbs);
    vkDestroyCommandPool(e.device, pool, NULL);
    ct_buffer_fini(&e, &b);
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(reset_command_pool), CT_REF(reset_command_buffer), CT_REF(trim_command_pool),
    CT_REF(queue_submit2),      CT_REF(set_device_mask),
};

CT_SUITE(cmdbuffers, tests)
