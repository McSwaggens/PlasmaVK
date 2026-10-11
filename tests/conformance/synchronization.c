/* Chapter "Synchronization and Cache Control". Events: event state on the
 * host and the device, and device waits on events signaled by the host
 * before submission or by the device earlier in submission order (the only
 * waits valid usage allows: VUID-vkCmdWaitEvents2-pEvents-03840, -03841,
 * VUID-vkSetEvent-event-09543), with the memory dependency of the wait.
 * Timeline semaphores: their values and host waits. Wait idle operations. */
#include "ct.h"

#include <time.h>

/* VkPhysicalDevicePortabilitySubsetFeaturesKHR (vulkan_beta.h, beta
 * extensions only): devices with VK_KHR_portability_subset may lack events. */
#define ST_PORTABILITY_SUBSET_FEATURES ((VkStructureType)1000163000)
struct st_portability_features {
    VkStructureType sType;
    void *pNext;
    VkBool32 constantAlphaColorBlendFactors, events, imageViewFormatReinterpretation, imageViewFormatSwizzle,
        imageView2DOn3DImage, multisampleArrayImage, mutableComparisonSamplers, pointPolygons, samplerMipLodBias,
        separateStencilMaskRef, shaderSampleRateInterpolationFunctions, tessellationIsolines, tessellationPointMode,
        triangleFans, vertexAttributeAccessBeyondStride;
};

struct st {
    struct ct_env e;
    int timeline; /* timelineSemaphore enabled */
    int sync2;    /* synchronization2 enabled */
};

/* An instance and a device with timelineSemaphore (Vulkan 1.2) and
 * synchronization2 (1.3) enabled when supported, and on a
 * VK_KHR_portability_subset device its events feature (skipped without:
 * VUID-vkCreateEvent-events-04468); skips below 1.3 if `sync2`. */
static void st_init(struct ct *t, struct st *s, int sync2)
{
    static struct st_portability_features p; /* in the device's chain */
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    struct ct_features f;
    void *chain = NULL;

    memset(s, 0, sizeof(*s));
    ct_env_init(t, &s->e);
    if (sync2 && s->e.version < VK_API_VERSION_1_3)
        ct_skip(t, "synchronization2 is Vulkan 1.3");
    if (s->e.portability_subset) {
        VkPhysicalDeviceFeatures2 f2;
        memset(&p, 0, sizeof(p));
        p.sType = ST_PORTABILITY_SUBSET_FEATURES;
        memset(&f2, 0, sizeof(f2));
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &p;
        vkGetPhysicalDeviceFeatures2(s->e.pd, &f2);
        if (!p.events)
            ct_skip(t, "VK_KHR_portability_subset device without events");
        memset(&p, 0, sizeof(p));
        p.sType = ST_PORTABILITY_SUBSET_FEATURES;
        p.events = VK_TRUE;
        chain = &p;
    }
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&v13, 0, sizeof(v13));
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    if (s->e.version >= VK_API_VERSION_1_2) {
        ct_get_features(&s->e, &f, 0);
        v12.timelineSemaphore = f.v12.timelineSemaphore;
        v12.pNext = chain;
        chain = &v12;
        if (s->e.version >= VK_API_VERSION_1_3) {
            v13.synchronization2 = f.v13.synchronization2;
            v13.pNext = chain;
            chain = &v13;
        }
    }
    s->timeline = v12.timelineSemaphore;
    s->sync2 = v13.synchronization2;
    if (sync2)
        CT_REQUIREF(t, s->sync2, "synchronization2 is required by Vulkan 1.3");
    ct_env_device(t, &s->e, chain);
}

static VkEvent st_event(struct ct *t, struct st *s, VkEventCreateFlags flags)
{
    VkEventCreateInfo info;
    VkEvent ev = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    info.flags = flags;
    CT_REQUIRE_VK(t, vkCreateEvent(s->e.device, &info, NULL, &ev));
    return ev;
}

static void st_full_barrier(VkCommandBuffer cb)
{
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL,
                         0, NULL);
}

static VkMemoryBarrier st_transfer_dependency(void)
{
    VkMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    return b;
}

static void st_copy(VkCommandBuffer cb, const struct ct_buffer *src, const struct ct_buffer *dst)
{
    VkBufferCopy region;
    memset(&region, 0, sizeof(region));
    region.size = src->size;
    vkCmdCopyBuffer(cb, src->buffer, dst->buffer, 1, &region);
}

static int st_all_words(const struct ct_buffer *b, uint32_t want)
{
    VkDeviceSize i;
    for (i = 0; i < b->size; i += 4) {
        uint32_t w;
        memcpy(&w, b->data + i, 4);
        if (w != want)
            return 0;
    }
    return 1;
}

static void st_submit(struct ct *t, struct st *s, uint32_t count, const VkCommandBuffer *cbs)
{
    VkSubmitInfo submit;
    uint32_t i;
    for (i = 0; i < count; i++)
        CT_REQUIRE_VK(t, vkEndCommandBuffer(cbs[i]));
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = count;
    submit.pCommandBuffers = cbs;
    CT_REQUIRE_VK(t, vkQueueSubmit(s->e.queue, 1, &submit, VK_NULL_HANDLE));
}

static void st_idle(struct ct *t, struct st *s)
{
    CT_REQUIRE_VK(t, vkQueueWaitIdle(s->e.queue));
}

/* ---- Event state ---- */

CT_TEST(event_host_state,
        "Synchronization and Cache Control > Events: an event is created unsignaled (VK_EVENT_RESET); vkSetEvent "
        "signals it and vkResetEvent unsignals it, as vkGetEventStatus reports, also when repeated; vkDestroyEvent "
        "accepts VK_NULL_HANDLE",
        "vkCreateEvent vkDestroyEvent vkGetEventStatus vkSetEvent vkResetEvent")
{
    struct st s;
    VkEvent a, b;
    st_init(t, &s, 0);
    a = st_event(t, &s, 0);
    b = st_event(t, &s, 0);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, a), VK_EVENT_RESET);
    CT_CHECK_VK(t, vkSetEvent(s.e.device, a), VK_SUCCESS);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, a), VK_EVENT_SET);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, b), VK_EVENT_RESET); /* events are independent */
    CT_CHECK_VK(t, vkSetEvent(s.e.device, a), VK_SUCCESS);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, a), VK_EVENT_SET);
    CT_CHECK_VK(t, vkResetEvent(s.e.device, a), VK_SUCCESS);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, a), VK_EVENT_RESET);
    CT_CHECK_VK(t, vkResetEvent(s.e.device, a), VK_SUCCESS);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, a), VK_EVENT_RESET);
    vkDestroyEvent(s.e.device, a, NULL);
    vkDestroyEvent(s.e.device, b, NULL);
    vkDestroyEvent(s.e.device, VK_NULL_HANDLE, NULL);
    ct_env_fini(&s.e);
}

CT_TEST(event_device_state,
        "Synchronization and Cache Control > Events: vkCmdSetEvent signals and vkCmdResetEvent unsignals an event "
        "when they execute; separated by an execution dependency (the specification's note: there is no ordering "
        "between them otherwise), the later one decides the state",
        "vkCreateEvent vkDestroyEvent vkGetEventStatus vkSetEvent vkCmdSetEvent vkCmdResetEvent vkCmdPipelineBarrier "
        "vkQueueSubmit vkWaitForFences")
{
    struct st s;
    VkCommandBuffer cb;
    VkEvent ev;
    st_init(t, &s, 0);
    ev = st_event(t, &s, 0);
    cb = ct_begin_commands(t, &s.e);
    vkCmdSetEvent(cb, ev, VK_PIPELINE_STAGE_TRANSFER_BIT);
    ct_submit_and_wait(t, &s.e, cb);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, ev), VK_EVENT_SET);
    cb = ct_begin_commands(t, &s.e);
    vkCmdResetEvent(cb, ev, VK_PIPELINE_STAGE_TRANSFER_BIT);
    ct_submit_and_wait(t, &s.e, cb);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, ev), VK_EVENT_RESET);
    cb = ct_begin_commands(t, &s.e);
    vkCmdSetEvent(cb, ev, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    st_full_barrier(cb);
    vkCmdResetEvent(cb, ev, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    ct_submit_and_wait(t, &s.e, cb);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, ev), VK_EVENT_RESET);
    CT_CHECK_VK(t, vkSetEvent(s.e.device, ev), VK_SUCCESS);
    cb = ct_begin_commands(t, &s.e);
    vkCmdResetEvent(cb, ev, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    st_full_barrier(cb);
    vkCmdSetEvent(cb, ev, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    ct_submit_and_wait(t, &s.e, cb);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, ev), VK_EVENT_SET);
    vkDestroyEvent(s.e.device, ev, NULL);
    ct_env_fini(&s.e);
}

CT_TEST(event_device_state2,
        "Synchronization and Cache Control > Events: vkCmdSetEvent2 signals and vkCmdResetEvent2 unsignals an "
        "event when they execute (synchronization2)",
        "vkCreateEvent vkDestroyEvent vkGetEventStatus vkCmdSetEvent2 vkCmdResetEvent2 vkCmdPipelineBarrier "
        "vkQueueSubmit vkWaitForFences")
{
    struct st s;
    VkMemoryBarrier2 mb;
    VkDependencyInfo dep;
    VkCommandBuffer cb;
    VkEvent ev;
    st_init(t, &s, 1);
    ev = st_event(t, &s, 0);
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    memset(&dep, 0, sizeof(dep));
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    cb = ct_begin_commands(t, &s.e);
    vkCmdSetEvent2(cb, ev, &dep);
    ct_submit_and_wait(t, &s.e, cb);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, ev), VK_EVENT_SET);
    cb = ct_begin_commands(t, &s.e);
    vkCmdResetEvent2(cb, ev, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);
    ct_submit_and_wait(t, &s.e, cb);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, ev), VK_EVENT_RESET);
    cb = ct_begin_commands(t, &s.e);
    vkCmdResetEvent2(cb, ev, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    st_full_barrier(cb);
    vkCmdSetEvent2(cb, ev, &dep);
    ct_submit_and_wait(t, &s.e, cb);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, ev), VK_EVENT_SET);
    vkDestroyEvent(s.e.device, ev, NULL);
    ct_env_fini(&s.e);
}

CT_TEST(event_device_signal_host_sees,
        "Synchronization and Cache Control > Events: an event signaled by vkCmdSetEvent stays unsignaled until "
        "the command executes (its submission is held back by a timeline semaphore), then vkGetEventStatus sees "
        "it signaled while the submission may still be pending, with the earlier transfer's writes done",
        "vkCreateEvent vkDestroyEvent vkGetEventStatus vkCmdSetEvent vkCmdFillBuffer vkCmdPipelineBarrier "
        "vkQueueSubmit vkQueueWaitIdle vkCreateSemaphore vkDestroySemaphore vkSignalSemaphore vkEndCommandBuffer "
        "vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory vkBindBufferMemory "
        "vkMapMemory vkFreeMemory")
{
    struct st s;
    VkSemaphoreTypeCreateInfo type;
    VkSemaphoreCreateInfo sem_info;
    VkSemaphoreSignalInfo signal;
    VkTimelineSemaphoreSubmitInfo values;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit;
    VkMemoryBarrier mb;
    struct ct_buffer buf;
    VkSemaphore gate;
    VkCommandBuffer cb;
    VkEvent ev;
    uint64_t one = 1;
    int polls = 0;
    struct timespec ts;

    st_init(t, &s, 0);
    if (s.e.version < VK_API_VERSION_1_2)
        ct_skip(t, "timeline semaphores are Vulkan 1.2");
    CT_REQUIREF(t, s.timeline, "timelineSemaphore is required by Vulkan 1.2");
    ev = st_event(t, &s, 0);
    ct_host_buffer(t, &s.e, 1 << 20, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buf);
    memset(&type, 0, sizeof(type));
    type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    memset(&sem_info, 0, sizeof(sem_info));
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sem_info.pNext = &type;
    CT_REQUIRE_VK(t, vkCreateSemaphore(s.e.device, &sem_info, NULL, &gate));
    cb = ct_begin_commands(t, &s.e);
    vkCmdFillBuffer(cb, buf.buffer, 0, VK_WHOLE_SIZE, 0x600df00du);
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdSetEvent(cb, ev, VK_PIPELINE_STAGE_TRANSFER_BIT);
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
    CT_REQUIRE_VK(t, vkQueueSubmit(s.e.queue, 1, &submit, VK_NULL_HANDLE));
    ts.tv_sec = 0;
    ts.tv_nsec = 20000000;
    nanosleep(&ts, NULL);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, ev), VK_EVENT_RESET);
    memset(&signal, 0, sizeof(signal));
    signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal.semaphore = gate;
    signal.value = 1;
    CT_REQUIRE_VK(t, vkSignalSemaphore(s.e.device, &signal));
    ts.tv_nsec = 100000;
    while (vkGetEventStatus(s.e.device, ev) == VK_EVENT_RESET && polls++ < 50000) /* 5 s */
        nanosleep(&ts, NULL);
    CT_CHECK_VK(t, vkGetEventStatus(s.e.device, ev), VK_EVENT_SET);
    st_idle(t, &s);
    CT_CHECK(t, st_all_words(&buf, 0x600df00du));
    vkFreeCommandBuffers(s.e.device, s.e.pool, 1, &cb);
    vkDestroySemaphore(s.e.device, gate, NULL);
    ct_buffer_fini(&s.e, &buf);
    vkDestroyEvent(s.e.device, ev, NULL);
    ct_env_fini(&s.e);
}

/* ---- Device waits ---- */

CT_TEST(event_wait_host_signal,
        "Synchronization and Cache Control > Events > vkCmdWaitEvents: a wait on an event the host signaled before "
        "submission, with VK_PIPELINE_STAGE_HOST_BIT as the source stage, completes, and host writes made before "
        "the signal are visible to the commands after the wait",
        "vkCreateEvent vkDestroyEvent vkSetEvent vkCmdWaitEvents vkCmdCopyBuffer vkCmdPipelineBarrier vkQueueSubmit "
        "vkWaitForFences vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory "
        "vkBindBufferMemory vkMapMemory vkFreeMemory")
{
    struct st s;
    struct ct_buffer a, b;
    VkMemoryBarrier mb;
    VkCommandBuffer cb;
    VkEvent ev;
    st_init(t, &s, 0);
    ev = st_event(t, &s, 0);
    ct_host_buffer(t, &s.e, 1 << 20, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &a);
    ct_host_buffer(t, &s.e, 1 << 20, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &b);
    memset(a.data, 0x5a, 1 << 20);
    CT_REQUIRE_VK(t, vkSetEvent(s.e.device, ev));
    cb = ct_begin_commands(t, &s.e);
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdWaitEvents(cb, 1, &ev, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 1, &mb, 0, NULL, 0, NULL);
    st_copy(cb, &a, &b);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &s.e, cb);
    CT_CHECK(t, st_all_words(&b, 0x5a5a5a5au));
    ct_buffer_fini(&s.e, &a);
    ct_buffer_fini(&s.e, &b);
    vkDestroyEvent(s.e.device, ev, NULL);
    ct_env_fini(&s.e);
}

CT_TEST(event_wait_device_signal,
        "Synchronization and Cache Control > Events > vkCmdWaitEvents: a wait on an event signaled by a vkCmdSetEvent "
        "earlier in submission order (the same command buffer, another command buffer of the batch, an earlier "
        "submission) makes the transfer writes before the signal visible to the transfer reads after the wait; "
        "also waiting on two events, one of them signaled by the host",
        "vkCreateEvent vkDestroyEvent vkSetEvent vkResetEvent vkCmdSetEvent vkCmdWaitEvents vkCmdFillBuffer "
        "vkCmdCopyBuffer vkCmdPipelineBarrier vkQueueSubmit vkQueueWaitIdle vkEndCommandBuffer vkGetEventStatus "
        "vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory vkBindBufferMemory "
        "vkMapMemory vkFreeMemory")
{
    struct st s;
    struct ct_buffer a, b;
    VkMemoryBarrier mb = st_transfer_dependency();
    VkEvent ev, host_ev, both[2];
    uint32_t round;
    st_init(t, &s, 0);
    ev = st_event(t, &s, 0);
    host_ev = st_event(t, &s, 0);
    ct_host_buffer(t, &s.e, 4 << 20, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &a);
    ct_host_buffer(t, &s.e, 4 << 20, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &b);
    for (round = 0; round < 4; round++) {
        uint32_t value = 0x11223344u + round;
        VkCommandBuffer cbs[2];
        cbs[0] = ct_begin_commands(t, &s.e);
        vkCmdFillBuffer(cbs[0], a.buffer, 0, VK_WHOLE_SIZE, value);
        vkCmdSetEvent(cbs[0], ev, VK_PIPELINE_STAGE_TRANSFER_BIT);
        if (round == 0) {
            cbs[1] = cbs[0];
        } else {
            if (round >= 2) /* an earlier submission */
                st_submit(t, &s, 1, &cbs[0]);
            cbs[1] = ct_begin_commands(t, &s.e);
        }
        if (round == 3) {
            CT_REQUIRE_VK(t, vkSetEvent(s.e.device, host_ev));
            both[0] = ev;
            both[1] = host_ev;
            vkCmdWaitEvents(cbs[1], 2, both, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 1, &mb, 0, NULL, 0, NULL);
        } else {
            vkCmdWaitEvents(cbs[1], 1, &ev, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 1, &mb, 0,
                            NULL, 0, NULL);
        }
        st_copy(cbs[1], &a, &b);
        ct_host_read_barrier(cbs[1]);
        st_submit(t, &s, round == 1 ? 2 : 1, round == 1 ? cbs : &cbs[1]);
        st_idle(t, &s);
        CT_CHECKF(t, st_all_words(&b, value), "round %u: the copy after the wait missed the fill before the signal",
                  round);
        CT_CHECK_VK(t, vkGetEventStatus(s.e.device, ev), VK_EVENT_SET);
        CT_REQUIRE_VK(t, vkResetEvent(s.e.device, ev));
        vkFreeCommandBuffers(s.e.device, s.e.pool, round == 0 ? 1 : 2, cbs);
    }
    ct_buffer_fini(&s.e, &a);
    ct_buffer_fini(&s.e, &b);
    vkDestroyEvent(s.e.device, ev, NULL);
    vkDestroyEvent(s.e.device, host_ev, NULL);
    ct_env_fini(&s.e);
}

CT_TEST(event_wait2,
        "Synchronization and Cache Control > Events > vkCmdWaitEvents2: a wait on an event signaled by vkCmdSetEvent2 "
        "earlier in submission order with the same VkDependencyInfo (VUID-vkCmdWaitEvents2-pEvents-03838) applies "
        "its memory dependency; also for a VK_EVENT_CREATE_DEVICE_ONLY_BIT event, in the same and in another "
        "command buffer",
        "vkCreateEvent vkDestroyEvent vkCmdSetEvent2 vkCmdResetEvent2 vkCmdWaitEvents2 vkCmdFillBuffer "
        "vkCmdCopyBuffer vkCmdPipelineBarrier vkQueueSubmit vkQueueWaitIdle vkEndCommandBuffer vkCreateBuffer "
        "vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory vkBindBufferMemory vkMapMemory vkFreeMemory")
{
    struct st s;
    struct ct_buffer a, b;
    VkMemoryBarrier2 mb;
    VkDependencyInfo dep;
    VkEvent evs[2];
    uint32_t round;
    st_init(t, &s, 1);
    evs[0] = st_event(t, &s, 0);
    evs[1] = st_event(t, &s, VK_EVENT_CREATE_DEVICE_ONLY_BIT);
    ct_host_buffer(t, &s.e, 4 << 20, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &a);
    ct_host_buffer(t, &s.e, 4 << 20, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &b);
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    mb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    memset(&dep, 0, sizeof(dep));
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    for (round = 0; round < 4; round++) {
        VkEvent ev = evs[round / 2];
        uint32_t value = 0x0badcafeu + round;
        VkCommandBuffer cbs[2];
        cbs[0] = ct_begin_commands(t, &s.e);
        vkCmdFillBuffer(cbs[0], a.buffer, 0, VK_WHOLE_SIZE, value);
        vkCmdSetEvent2(cbs[0], ev, &dep);
        cbs[1] = round % 2 ? ct_begin_commands(t, &s.e) : cbs[0];
        vkCmdWaitEvents2(cbs[1], 1, &ev, &dep);
        st_copy(cbs[1], &a, &b);
        vkCmdResetEvent2(cbs[1], ev, VK_PIPELINE_STAGE_2_COPY_BIT);
        ct_host_read_barrier(cbs[1]);
        st_submit(t, &s, round % 2 ? 2 : 1, cbs);
        st_idle(t, &s);
        CT_CHECKF(t, st_all_words(&b, value), "round %u: the copy after the wait missed the fill before the signal",
                  round);
        vkFreeCommandBuffers(s.e.device, s.e.pool, round % 2 ? 2 : 1, cbs);
    }
    ct_buffer_fini(&s.e, &a);
    ct_buffer_fini(&s.e, &b);
    vkDestroyEvent(s.e.device, evs[0], NULL);
    vkDestroyEvent(s.e.device, evs[1], NULL);
    ct_env_fini(&s.e);
}

/* ---- Timeline semaphores ---- */

/* A device with timelineSemaphore; skips below Vulkan 1.2. */
static void st_init_timeline(struct ct *t, struct ct_env *e)
{
    VkPhysicalDeviceVulkan12Features v12;
    struct ct_features f;
    ct_env_init(t, e);
    if (e->version < VK_API_VERSION_1_2)
        ct_skip(t, "timeline semaphores are Vulkan 1.2");
    ct_get_features(e, &f, 0);
    CT_REQUIREF(t, f.v12.timelineSemaphore, "timelineSemaphore is required by Vulkan 1.2");
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.timelineSemaphore = VK_TRUE;
    ct_env_device(t, e, &v12);
}

static VkSemaphore st_timeline(struct ct *t, struct ct_env *e, uint64_t initial)
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
    info.pNext = &type;
    CT_REQUIRE_VK(t, vkCreateSemaphore(e->device, &info, NULL, &s));
    return s;
}

static uint64_t st_value(struct ct *t, struct ct_env *e, VkSemaphore s)
{
    uint64_t v = 0xdeadbeefdeadbeefull;
    CT_CHECK_VK(t, vkGetSemaphoreCounterValue(e->device, s, &v), VK_SUCCESS);
    return v;
}

static VkResult st_host_signal(struct ct_env *e, VkSemaphore s, uint64_t value)
{
    VkSemaphoreSignalInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    info.semaphore = s;
    info.value = value;
    return vkSignalSemaphore(e->device, &info);
}

/* Submits cb in a batch waiting for `wait` to reach wait_value (if wait is
 * not VK_NULL_HANDLE) and signaling `signal` with signal_value. */
static void st_submit_timeline(struct ct *t, struct ct_env *e, VkCommandBuffer cb, VkSemaphore wait,
                               uint64_t wait_value, VkSemaphore signal, uint64_t signal_value)
{
    VkTimelineSemaphoreSubmitInfo values;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit;
    memset(&values, 0, sizeof(values));
    values.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    values.waitSemaphoreValueCount = wait ? 1 : 0;
    values.pWaitSemaphoreValues = wait ? &wait_value : NULL;
    values.signalSemaphoreValueCount = 1;
    values.pSignalSemaphoreValues = &signal_value;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = &values;
    submit.waitSemaphoreCount = wait ? 1 : 0;
    submit.pWaitSemaphores = wait ? &wait : NULL;
    submit.pWaitDstStageMask = wait ? &stage : NULL;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &signal;
    CT_REQUIRE_VK(t, vkQueueSubmit(e->queue, 1, &submit, VK_NULL_HANDLE));
}

static void st_sleep_ms(long ms)
{
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000;
    nanosleep(&ts, NULL);
}

CT_TEST(semaphore_counter_value,
        "Synchronization and Cache Control > Semaphores > Host Operations on Semaphores (vkGetSemaphoreCounterValue, "
        "Vulkan 1.2): a timeline semaphore's value is VkSemaphoreTypeCreateInfo::initialValue, then the value of "
        "each signal operation: by vkSignalSemaphore at once, by a batch once it executes; not before (a batch "
        "held by a wait on a value the host signals later has not signaled)",
        "vkGetSemaphoreCounterValue vkCreateSemaphore vkDestroySemaphore vkSignalSemaphore vkQueueSubmit "
        "vkQueueWaitIdle vkCmdFillBuffer vkCmdPipelineBarrier vkEndCommandBuffer vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements vkAllocateMemory vkBindBufferMemory vkMapMemory vkFreeMemory")
{
    struct st s;
    struct ct_buffer buf;
    VkSemaphore a, gate, big;
    VkCommandBuffer cb;
    uint64_t v;

    st_init_timeline(t, &s.e);
    a = st_timeline(t, &s.e, 7);
    gate = st_timeline(t, &s.e, 0);
    big = st_timeline(t, &s.e, 0x123456789abcull);
    CT_CHECKF(t, (v = st_value(t, &s.e, a)) == 7, "initial value 7: %llu", (unsigned long long)v);
    CT_CHECKF(t, (v = st_value(t, &s.e, gate)) == 0, "initial value 0: %llu", (unsigned long long)v);
    CT_CHECKF(t, (v = st_value(t, &s.e, big)) == 0x123456789abcull, "initial value 0x123456789abc: 0x%llx",
              (unsigned long long)v);
    CT_REQUIRE_VK(t, st_host_signal(&s.e, a, 9));
    CT_CHECKF(t, (v = st_value(t, &s.e, a)) == 9, "after vkSignalSemaphore(9): %llu", (unsigned long long)v);
    ct_host_buffer(t, &s.e, 4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buf);
    cb = ct_begin_commands(t, &s.e);
    vkCmdFillBuffer(cb, buf.buffer, 0, VK_WHOLE_SIZE, 0x5e5e5e5eu);
    ct_host_read_barrier(cb);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    st_submit_timeline(t, &s.e, cb, gate, 1, a, 12);
    st_sleep_ms(20);
    CT_CHECKF(t, (v = st_value(t, &s.e, a)) == 9, "signaled (%llu) by a batch still waiting for the host",
              (unsigned long long)v);
    CT_REQUIRE_VK(t, st_host_signal(&s.e, gate, 1));
    st_idle(t, &s);
    CT_CHECKF(t, (v = st_value(t, &s.e, a)) == 12, "after the batch's signal of 12: %llu", (unsigned long long)v);
    CT_CHECKF(t, (v = st_value(t, &s.e, gate)) == 1, "after vkSignalSemaphore(1): %llu", (unsigned long long)v);
    CT_CHECK(t, st_all_words(&buf, 0x5e5e5e5eu));
    vkFreeCommandBuffers(s.e.device, s.e.pool, 1, &cb);
    vkDestroySemaphore(s.e.device, a, NULL);
    vkDestroySemaphore(s.e.device, gate, NULL);
    vkDestroySemaphore(s.e.device, big, NULL);
    ct_buffer_fini(&s.e, &buf);
    ct_env_fini(&s.e);
}

static VkResult st_wait(struct ct_env *e, VkSemaphoreWaitFlags flags, uint32_t n, const VkSemaphore *s,
                        const uint64_t *values, uint64_t timeout)
{
    VkSemaphoreWaitInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    info.flags = flags;
    info.semaphoreCount = n;
    info.pSemaphores = s;
    info.pValues = values;
    return vkWaitSemaphores(e->device, &info, timeout);
}

struct st_waiter {
    struct ct_env *e;
    VkSemaphore s;
    uint64_t value;
    VkResult result;
    volatile int done;
};

static void *st_waiter_main(void *p)
{
    struct st_waiter *w = p;
    w->result = st_wait(w->e, 0, 1, &w->s, &w->value, 10000000000ull);
    w->done = 1;
    return NULL;
}

CT_TEST(wait_semaphores,
        "Synchronization and Cache Control > Semaphores > Host Operations on Semaphores (vkWaitSemaphores, Vulkan "
        "1.2): with timeout 0, VK_SUCCESS if every semaphore (VK_SEMAPHORE_WAIT_ANY_BIT: at least one) has reached "
        "its value, else VK_TIMEOUT; an expiring timeout returns VK_TIMEOUT; a wait blocks until a later host signal "
        "satisfies it, and returns once a batch's signal does (the batch's writes then visible to the host)",
        "vkWaitSemaphores vkGetSemaphoreCounterValue vkCreateSemaphore vkDestroySemaphore vkSignalSemaphore "
        "vkQueueSubmit vkQueueWaitIdle vkCmdFillBuffer vkCmdPipelineBarrier vkEndCommandBuffer vkCreateBuffer "
        "vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory vkBindBufferMemory vkMapMemory vkFreeMemory")
{
    struct st s;
    struct ct_buffer buf;
    struct st_waiter w;
    pthread_t thread;
    VkSemaphore sem[2];
    VkCommandBuffer cb;
    uint64_t v[2];

    st_init_timeline(t, &s.e);
    sem[0] = st_timeline(t, &s.e, 5);
    sem[1] = st_timeline(t, &s.e, 0);
    v[0] = 5;
    v[1] = 0;
    CT_CHECK_VK(t, st_wait(&s.e, 0, 2, sem, v, 0), VK_SUCCESS);
    v[0] = 3; /* below the value: reached */
    CT_CHECK_VK(t, st_wait(&s.e, 0, 2, sem, v, 0), VK_SUCCESS);
    v[0] = 5;
    v[1] = 1;
    CT_CHECK_VK(t, st_wait(&s.e, 0, 2, sem, v, 0), VK_TIMEOUT);
    CT_CHECK_VK(t, st_wait(&s.e, 0, 2, sem, v, 1000000), VK_TIMEOUT); /* 1 ms */
    CT_CHECK_VK(t, st_wait(&s.e, VK_SEMAPHORE_WAIT_ANY_BIT, 2, sem, v, 0), VK_SUCCESS);
    v[0] = 6;
    CT_CHECK_VK(t, st_wait(&s.e, VK_SEMAPHORE_WAIT_ANY_BIT, 2, sem, v, 0), VK_TIMEOUT);
    CT_CHECK_VK(t, st_wait(&s.e, VK_SEMAPHORE_WAIT_ANY_BIT, 2, sem, v, 1000000), VK_TIMEOUT);

    /* a host signal ends another thread's wait */
    memset(&w, 0, sizeof(w));
    w.e = &s.e;
    w.s = sem[1];
    w.value = 3;
    CT_REQUIRE(t, pthread_create(&thread, NULL, st_waiter_main, &w) == 0);
    st_sleep_ms(20);
    CT_CHECKF(t, !w.done, "vkWaitSemaphores returned %s before the value was reached", ct_result_name(w.result));
    CT_CHECK_VK(t, st_host_signal(&s.e, sem[1], 3), VK_SUCCESS);
    pthread_join(thread, NULL);
    CT_CHECK_VK(t, w.result, VK_SUCCESS);
    v[0] = 6;
    v[1] = 3;
    CT_CHECK_VK(t, st_wait(&s.e, VK_SEMAPHORE_WAIT_ANY_BIT, 2, sem, v, 0), VK_SUCCESS);
    CT_CHECK_VK(t, st_wait(&s.e, 0, 2, sem, v, 0), VK_TIMEOUT);

    /* a batch's signal */
    ct_host_buffer(t, &s.e, 1 << 20, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buf);
    cb = ct_begin_commands(t, &s.e);
    vkCmdFillBuffer(cb, buf.buffer, 0, VK_WHOLE_SIZE, 0x7a7a7a7au);
    ct_host_read_barrier(cb);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    st_submit_timeline(t, &s.e, cb, VK_NULL_HANDLE, 0, sem[0], 8);
    v[0] = 8;
    CT_CHECK_VK(t, st_wait(&s.e, 0, 1, sem, v, 10000000000ull), VK_SUCCESS);
    CT_CHECK(t, st_value(t, &s.e, sem[0]) == 8);
    CT_CHECK(t, st_all_words(&buf, 0x7a7a7a7au));
    st_idle(t, &s);
    vkFreeCommandBuffers(s.e.device, s.e.pool, 1, &cb);
    vkDestroySemaphore(s.e.device, sem[0], NULL);
    vkDestroySemaphore(s.e.device, sem[1], NULL);
    ct_buffer_fini(&s.e, &buf);
    ct_env_fini(&s.e);
}

/* ---- Wait idle ---- */

#define WI_QUEUES 16
#define WI_SIZE (4 << 20)

struct wi_queue {
    uint32_t family;
    VkQueue queue;
    VkCommandPool pool;
    VkCommandBuffer cb;
    VkFence fence;
    struct ct_buffer buf;
};

CT_TEST(device_wait_idle,
        "Synchronization and Cache Control > Wait Idle Operations (vkDeviceWaitIdle): equivalent to "
        "vkQueueWaitIdle on every queue of the device (waiting for a fence of every earlier submission): after it "
        "returns, the work submitted to each queue (two of a family that has them, one of every other family "
        "with transfers) has completed: their fences are signaled and their writes visible to the host",
        "vkDeviceWaitIdle vkCreateDevice vkDestroyDevice vkGetDeviceQueue vkGetPhysicalDeviceQueueFamilyProperties "
        "vkCreateFence vkDestroyFence vkGetFenceStatus vkQueueSubmit vkCmdFillBuffer vkCmdPipelineBarrier "
        "vkCreateCommandPool vkDestroyCommandPool vkAllocateCommandBuffers vkBeginCommandBuffer vkEndCommandBuffer "
        "vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory vkBindBufferMemory "
        "vkMapMemory vkFreeMemory")
{
    struct ct_env e;
    VkQueueFamilyProperties families[16];
    VkDeviceQueueCreateInfo qinfo[16];
    VkDeviceCreateInfo dinfo;
    const char *names[2];
    float priorities[2] = { 1.0f, 1.0f };
    struct wi_queue q[WI_QUEUES];
    uint32_t family_count = 16, nfam = 0, nq = 0, i, j, k;

    ct_env_init(t, &e);
    vkGetPhysicalDeviceQueueFamilyProperties(e.pd, &family_count, families);
    memset(qinfo, 0, sizeof(qinfo));
    memset(q, 0, sizeof(q));
    for (i = 0; i < family_count && nq < WI_QUEUES; i++) {
        if (!(families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)))
            continue;
        qinfo[nfam].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qinfo[nfam].queueFamilyIndex = i;
        qinfo[nfam].queueCount = CT_MIN(families[i].queueCount, 2);
        qinfo[nfam].pQueuePriorities = priorities;
        for (j = 0; j < qinfo[nfam].queueCount && nq < WI_QUEUES; j++)
            q[nq++].family = i;
        nfam++;
    }
    CT_REQUIRE(t, nq > 0);
    memset(&dinfo, 0, sizeof(dinfo));
    dinfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dinfo.queueCreateInfoCount = nfam;
    dinfo.pQueueCreateInfos = qinfo;
    dinfo.enabledExtensionCount = ct_add_portability_subset(&e, names, 0);
    dinfo.ppEnabledExtensionNames = dinfo.enabledExtensionCount ? names : NULL;
    CT_REQUIRE_VK(t, vkCreateDevice(e.pd, &dinfo, NULL, &e.device));
    for (i = 0; i < nq; i++) {
        VkCommandPoolCreateInfo pinfo;
        VkCommandBufferAllocateInfo ainfo;
        VkCommandBufferBeginInfo begin;
        VkFenceCreateInfo finfo;
        VkSubmitInfo submit;
        uint32_t index = (i > 0 && q[i - 1].family == q[i].family) ? 1 : 0;
        vkGetDeviceQueue(e.device, q[i].family, index, &q[i].queue);
        memset(&pinfo, 0, sizeof(pinfo));
        pinfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pinfo.queueFamilyIndex = q[i].family;
        CT_REQUIRE_VK(t, vkCreateCommandPool(e.device, &pinfo, NULL, &q[i].pool));
        memset(&ainfo, 0, sizeof(ainfo));
        ainfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ainfo.commandPool = q[i].pool;
        ainfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ainfo.commandBufferCount = 1;
        CT_REQUIRE_VK(t, vkAllocateCommandBuffers(e.device, &ainfo, &q[i].cb));
        memset(&finfo, 0, sizeof(finfo));
        finfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        CT_REQUIRE_VK(t, vkCreateFence(e.device, &finfo, NULL, &q[i].fence));
        ct_host_buffer(t, &e, WI_SIZE, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &q[i].buf);
        memset(&begin, 0, sizeof(begin));
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        CT_REQUIRE_VK(t, vkBeginCommandBuffer(q[i].cb, &begin));
        for (k = 0; k < 16; k++) { /* some work; the last fill decides */
            vkCmdFillBuffer(q[i].cb, q[i].buf.buffer, 0, VK_WHOLE_SIZE, 0x01010101u * (k + 1) + i);
            ct_transfer_barrier(q[i].cb);
        }
        ct_host_read_barrier(q[i].cb);
        CT_REQUIRE_VK(t, vkEndCommandBuffer(q[i].cb));
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &q[i].cb;
        CT_REQUIRE_VK(t, vkQueueSubmit(q[i].queue, 1, &submit, q[i].fence));
    }
    CT_CHECK_VK(t, vkDeviceWaitIdle(e.device), VK_SUCCESS);
    for (i = 0; i < nq; i++) {
        CT_CHECKF(t, vkGetFenceStatus(e.device, q[i].fence) == VK_SUCCESS, "queue %u (family %u): fence not signaled",
                  i, q[i].family);
        CT_CHECKF(t, st_all_words(&q[i].buf, 0x10101010u + i), "queue %u (family %u): fills not done", i,
                  q[i].family);
    }
    CT_CHECK_VK(t, vkDeviceWaitIdle(e.device), VK_SUCCESS); /* idle already */
    for (i = 0; i < nq; i++) {
        vkDestroyFence(e.device, q[i].fence, NULL);
        vkDestroyCommandPool(e.device, q[i].pool, NULL);
        ct_buffer_fini(&e, &q[i].buf);
    }
    ct_note(t, "%u queues of %u families", nq, nfam);
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(event_host_state),      CT_REF(event_device_state),     CT_REF(event_device_state2),
    CT_REF(event_device_signal_host_sees), CT_REF(event_wait_host_signal), CT_REF(event_wait_device_signal),
    CT_REF(event_wait2), CT_REF(semaphore_counter_value), CT_REF(wait_semaphores), CT_REF(device_wait_idle),
};

CT_SUITE(synchronization, tests)
