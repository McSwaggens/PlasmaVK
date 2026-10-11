/* Events (section 16): host and device signals and resets, device waits
 * on host and device signals within and across command buffers and
 * submissions. Private data (section 9). The small commands: tool and
 * external handle properties, device masks, vkQueueBindSparse. */
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <string.h>
#include <time.h>

#include "../pvk_test.h"

static VkInstance instance;
static VkPhysicalDevice pd;
static VkDevice device;
static VkQueue queue;
static VkCommandPool pool;
static uint32_t shared_type = UINT32_MAX;

struct buf {
    VkBuffer buffer;
    VkDeviceMemory memory;
    unsigned char *data;
};

static VkDevice create_device(uint32_t reserved_slots)
{
    VkDevicePrivateDataCreateInfo reserve;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo device_info;
    VkDevice d = VK_NULL_HANDLE;
    float priority = 1.0f;

    memset(&reserve, 0, sizeof(reserve));
    reserve.sType = VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO;
    reserve.privateDataSlotRequestCount = reserved_slots;
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.pNext = reserved_slots ? &reserve : NULL;
    v12.timelineSemaphore = VK_TRUE;
    memset(&v13, 0, sizeof(v13));
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13.pNext = &v12;
    v13.synchronization2 = VK_TRUE;
    v13.privateData = VK_TRUE;
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &v13;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    CHECK_VK(vkCreateDevice(pd, &device_info, NULL, &d), VK_SUCCESS);
    return d;
}

static void setup(void)
{
    VkPhysicalDeviceMemoryProperties mem;
    VkInstanceCreateInfo instance_info;
    VkApplicationInfo app;
    VkCommandPoolCreateInfo pool_info;
    uint32_t count = 1, i;

    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_4;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    REQUIRE_VK(vkCreateInstance(&instance_info, NULL, &instance));
    REQUIRE_VK(vkEnumeratePhysicalDevices(instance, &count, &pd));
    device = create_device(0);
    REQUIRE(device);
    vkGetDeviceQueue(device, 0, 0, &queue);
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    for (i = 0; i < mem.memoryTypeCount; i++)
        if ((mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) && shared_type == UINT32_MAX)
            shared_type = i;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    REQUIRE_VK(vkCreateCommandPool(device, &pool_info, NULL, &pool));
}

static void teardown(void)
{
    vkDestroyCommandPool(device, pool, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
}

static struct buf make_buf(VkDeviceSize size)
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
    alloc.memoryTypeIndex = shared_type;
    vkAllocateMemory(device, &alloc, NULL, &b.memory);
    vkBindBufferMemory(device, b.buffer, b.memory, 0);
    if (vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS) {
        b.data = p;
        memset(b.data, 0, (size_t)size);
    }
    return b;
}

static void free_buf(struct buf *b)
{
    vkDestroyBuffer(device, b->buffer, NULL);
    vkFreeMemory(device, b->memory, NULL);
}

static VkEvent make_event(VkEventCreateFlags flags)
{
    VkEventCreateInfo info;
    VkEvent e = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    info.flags = flags;
    CHECK_VK(vkCreateEvent(device, &info, NULL, &e), VK_SUCCESS);
    return e;
}

static VkCommandBuffer begin(void)
{
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferBeginInfo info;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &alloc, &cb);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &info);
    return cb;
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

static void signal_value(VkSemaphore s, uint64_t value)
{
    VkSemaphoreSignalInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    info.semaphore = s;
    info.value = value;
    vkSignalSemaphore(device, &info);
}

/* Ends and submits command buffers as one batch, optionally behind a
 * timeline semaphore value. */
static VkResult submit(uint32_t count, VkCommandBuffer *cbs, VkSemaphore wait, uint64_t value)
{
    VkTimelineSemaphoreSubmitInfo values;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo info;
    uint32_t i;
    for (i = 0; i < count; i++)
        vkEndCommandBuffer(cbs[i]);
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
    info.commandBufferCount = count;
    info.pCommandBuffers = cbs;
    return vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE);
}

static void run(VkCommandBuffer cb)
{
    CHECK_VK(submit(1, &cb, VK_NULL_HANDLE, 0), VK_SUCCESS);
    CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);
}

static VkMemoryBarrier transfer_dependency(void)
{
    VkMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    return b;
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

static void copy(VkCommandBuffer cb, const struct buf *src, const struct buf *dst, VkDeviceSize size)
{
    VkBufferCopy region;
    memset(&region, 0, sizeof(region));
    region.size = size;
    vkCmdCopyBuffer(cb, src->buffer, dst->buffer, 1, &region);
}

static int all_words(const struct buf *b, VkDeviceSize size, uint32_t want)
{
    VkDeviceSize i;
    for (i = 0; i < size; i += 4) {
        uint32_t w;
        memcpy(&w, b->data + i, 4);
        if (w != want)
            return 0;
    }
    return 1;
}

/* ---- Events ---- */

static void test_event_host(void)
{
    VkEvent e = make_event(0);
    CHECK_VK(vkGetEventStatus(device, e), VK_EVENT_RESET);
    CHECK_VK(vkSetEvent(device, e), VK_SUCCESS);
    CHECK_VK(vkGetEventStatus(device, e), VK_EVENT_SET);
    CHECK_VK(vkSetEvent(device, e), VK_SUCCESS);
    CHECK_VK(vkGetEventStatus(device, e), VK_EVENT_SET);
    CHECK_VK(vkResetEvent(device, e), VK_SUCCESS);
    CHECK_VK(vkGetEventStatus(device, e), VK_EVENT_RESET);
    CHECK_VK(vkResetEvent(device, e), VK_SUCCESS);
    CHECK_VK(vkGetEventStatus(device, e), VK_EVENT_RESET);
    vkDestroyEvent(device, e, NULL);
    vkDestroyEvent(device, VK_NULL_HANDLE, NULL);
}

/* Device signals and resets, both versions; set then reset (and the other
 * way round) ordered by a barrier, as the specification asks. */
static void test_event_device_set_reset(void)
{
    VkEvent e = make_event(0), d = make_event(VK_EVENT_CREATE_DEVICE_ONLY_BIT);
    VkDependencyInfo dep;
    VkCommandBuffer cb;

    memset(&dep, 0, sizeof(dep));
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    cb = begin();
    vkCmdSetEvent(cb, e, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    run(cb);
    CHECK_VK(vkGetEventStatus(device, e), VK_EVENT_SET);
    cb = begin();
    vkCmdResetEvent(cb, e, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    run(cb);
    CHECK_VK(vkGetEventStatus(device, e), VK_EVENT_RESET);
    cb = begin();
    vkCmdSetEvent2(cb, e, &dep);
    run(cb);
    CHECK_VK(vkGetEventStatus(device, e), VK_EVENT_SET);
    cb = begin();
    vkCmdResetEvent2(cb, e, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    run(cb);
    CHECK_VK(vkGetEventStatus(device, e), VK_EVENT_RESET);

    cb = begin();
    vkCmdSetEvent(cb, e, VK_PIPELINE_STAGE_TRANSFER_BIT);
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL,
                         0, NULL);
    vkCmdResetEvent(cb, e, VK_PIPELINE_STAGE_TRANSFER_BIT);
    run(cb);
    CHECK_VK(vkGetEventStatus(device, e), VK_EVENT_RESET);
    cb = begin();
    vkCmdResetEvent2(cb, e, VK_PIPELINE_STAGE_2_COPY_BIT);
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL,
                         0, NULL);
    vkCmdSetEvent2(cb, e, &dep);
    run(cb);
    CHECK_VK(vkGetEventStatus(device, e), VK_EVENT_SET);

    /* A device-only event: signaled and waited for on the device. */
    cb = begin();
    vkCmdSetEvent2(cb, d, &dep);
    vkCmdWaitEvents2(cb, 1, &d, &dep);
    vkCmdResetEvent2(cb, d, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    run(cb);
    vkDestroyEvent(device, e, NULL);
    vkDestroyEvent(device, d, NULL);
}

/* The host sees a device signal only once the work before it is done: the
 * submission is held back by a semaphore. */
static void test_event_device_set_host_sees(void)
{
    VkEvent e = make_event(0);
    VkSemaphore gate = make_timeline();
    struct buf a = make_buf(1 << 20);
    VkCommandBuffer cb = begin();
    int polls = 0;

    vkCmdFillBuffer(cb, a.buffer, 0, VK_WHOLE_SIZE, 0x5eed5eedu);
    vkCmdSetEvent(cb, e, VK_PIPELINE_STAGE_TRANSFER_BIT);
    REQUIRE_VK(submit(1, &cb, gate, 1));
    {
        struct timespec ts = { 0, 20000000 };
        nanosleep(&ts, NULL);
    }
    CHECK_VK(vkGetEventStatus(device, e), VK_EVENT_RESET);
    signal_value(gate, 1);
    while (vkGetEventStatus(device, e) != VK_EVENT_SET && polls < 100000) {
        struct timespec ts = { 0, 100000 };
        nanosleep(&ts, NULL);
        polls++;
    }
    CHECK_VK(vkGetEventStatus(device, e), VK_EVENT_SET);
    CHECK(all_words(&a, 1 << 20, 0x5eed5eedu)); /* the fill was done before the signal */
    CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);
    free_buf(&a);
    vkDestroySemaphore(device, gate, NULL);
    vkDestroyEvent(device, e, NULL);
}

/* Device waits: on a host signal made before submission; on device signals
 * in the same command buffer, another command buffer of the batch, another
 * batch and another submission; on several events at once; sync2. The data
 * that the signal follows must be seen after the wait. */
static void test_event_waits(void)
{
    struct buf a = make_buf(4 << 20), b = make_buf(4 << 20);
    VkEvent e = make_event(0), h = make_event(0);
    VkMemoryBarrier mb = transfer_dependency();
    VkMemoryBarrier2 mb2;
    VkDependencyInfo dep;
    VkCommandBuffer cbs[2];
    VkEvent both[2];
    uint32_t round;

    memset(&mb2, 0, sizeof(mb2));
    mb2.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb2.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    mb2.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb2.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    mb2.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    memset(&dep, 0, sizeof(dep));
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb2;

    /* Host signal before submission; host data written before it. */
    {
        VkMemoryBarrier hb;
        memset(&hb, 0, sizeof(hb));
        hb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        hb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        hb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        memset(a.data, 0x11, 4 << 20);
        vkSetEvent(device, h);
        cbs[0] = begin();
        vkCmdWaitEvents(cbs[0], 1, &h, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 1, &hb, 0, NULL, 0,
                        NULL);
        copy(cbs[0], &a, &b, 4 << 20);
        host_barrier(cbs[0]);
        run(cbs[0]);
        CHECK(all_words(&b, 4 << 20, 0x11111111u));
    }

    for (round = 0; round < 5; round++) {
        uint32_t value = 0x01020304u * (round + 1);
        vkCmdFillBuffer(cbs[0] = begin(), a.buffer, 0, VK_WHOLE_SIZE, value);
        if (round == 4)
            vkCmdSetEvent2(cbs[0], e, &dep);
        else
            vkCmdSetEvent(cbs[0], e, VK_PIPELINE_STAGE_TRANSFER_BIT);
        switch (round) {
        case 0: /* same command buffer */
            cbs[1] = cbs[0];
            break;
        case 1: /* two command buffers, one batch */
        case 4:
            cbs[1] = begin();
            break;
        case 2: /* two submissions */
        case 3: /* the second with another event the host signaled too */
            CHECK_VK(submit(1, &cbs[0], VK_NULL_HANDLE, 0), VK_SUCCESS);
            cbs[1] = begin();
            break;
        }
        if (round == 4) {
            vkCmdWaitEvents2(cbs[1], 1, &e, &dep);
        } else if (round == 3) {
            both[0] = e;
            both[1] = h;
            vkCmdWaitEvents(cbs[1], 2, both, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 1, &mb, 0, NULL, 0, NULL);
        } else {
            vkCmdWaitEvents(cbs[1], 1, &e, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 1, &mb, 0,
                            NULL, 0, NULL);
        }
        copy(cbs[1], &a, &b, 4 << 20);
        vkCmdResetEvent(cbs[1], e, VK_PIPELINE_STAGE_TRANSFER_BIT);
        host_barrier(cbs[1]);
        if (round == 1 || round == 4)
            CHECK_VK(submit(2, cbs, VK_NULL_HANDLE, 0), VK_SUCCESS);
        else
            CHECK_VK(submit(1, &cbs[1], VK_NULL_HANDLE, 0), VK_SUCCESS);
        CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);
        CHECK(all_words(&b, 4 << 20, value));
        CHECK_VK(vkGetEventStatus(device, e), VK_EVENT_RESET);
    }
    free_buf(&a);
    free_buf(&b);
    vkDestroyEvent(device, e, NULL);
    vkDestroyEvent(device, h, NULL);
}

/* ---- Private data ---- */

static VkPrivateDataSlot make_slot(VkDevice d)
{
    VkPrivateDataSlotCreateInfo info;
    VkPrivateDataSlot s = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO;
    CHECK_VK(vkCreatePrivateDataSlot(d, &info, NULL, &s), VK_SUCCESS);
    return s;
}

static uint64_t get_data(VkDevice d, VkObjectType type, uint64_t handle, VkPrivateDataSlot slot)
{
    uint64_t v = 0xdeadbeef;
    vkGetPrivateData(d, type, handle, slot, &v);
    return v;
}

static void test_private_data_objects(void)
{
    VkPrivateDataSlot s1 = make_slot(device), s2 = make_slot(device);
    struct buf a = make_buf(256);
    VkEvent e = make_event(0);
    VkSemaphore sem = make_timeline();
    VkCommandBuffer cb = begin();
    struct {
        VkObjectType type;
        uint64_t handle;
    } objs[8];
    uint32_t n = 0, i;

    objs[n].type = VK_OBJECT_TYPE_DEVICE, objs[n++].handle = (uint64_t)(uintptr_t)device;
    objs[n].type = VK_OBJECT_TYPE_QUEUE, objs[n++].handle = (uint64_t)(uintptr_t)queue;
    objs[n].type = VK_OBJECT_TYPE_BUFFER, objs[n++].handle = (uint64_t)(uintptr_t)a.buffer;
    objs[n].type = VK_OBJECT_TYPE_DEVICE_MEMORY, objs[n++].handle = (uint64_t)(uintptr_t)a.memory;
    objs[n].type = VK_OBJECT_TYPE_EVENT, objs[n++].handle = (uint64_t)(uintptr_t)e;
    objs[n].type = VK_OBJECT_TYPE_SEMAPHORE, objs[n++].handle = (uint64_t)(uintptr_t)sem;
    objs[n].type = VK_OBJECT_TYPE_COMMAND_BUFFER, objs[n++].handle = (uint64_t)(uintptr_t)cb;
    objs[n].type = VK_OBJECT_TYPE_PRIVATE_DATA_SLOT, objs[n++].handle = (uint64_t)(uintptr_t)s2;
    for (i = 0; i < n; i++) { /* nothing set: 0 */
        CHECK(get_data(device, objs[i].type, objs[i].handle, s1) == 0);
        CHECK(get_data(device, objs[i].type, objs[i].handle, s2) == 0);
    }
    for (i = 0; i < n; i++) {
        CHECK_VK(vkSetPrivateData(device, objs[i].type, objs[i].handle, s1, 1000 + i), VK_SUCCESS);
        CHECK_VK(vkSetPrivateData(device, objs[i].type, objs[i].handle, s2, ~(uint64_t)i), VK_SUCCESS);
    }
    for (i = 0; i < n; i++) {
        CHECK(get_data(device, objs[i].type, objs[i].handle, s1) == 1000 + i);
        CHECK(get_data(device, objs[i].type, objs[i].handle, s2) == ~(uint64_t)i);
    }
    CHECK_VK(vkSetPrivateData(device, objs[2].type, objs[2].handle, s1, 7), VK_SUCCESS); /* overwrite */
    CHECK(get_data(device, objs[2].type, objs[2].handle, s1) == 7);
    CHECK_VK(vkSetPrivateData(device, objs[2].type, objs[2].handle, s1, 0), VK_SUCCESS); /* 0 clears */
    CHECK(get_data(device, objs[2].type, objs[2].handle, s1) == 0);
    CHECK(get_data(device, objs[2].type, objs[2].handle, s2) == ~(uint64_t)2);

    /* A freed object's data goes with it: a buffer made afterwards (often
     * at the same address) starts at 0. */
    for (i = 0; i < 4; i++) {
        VkBufferCreateInfo info;
        VkBuffer b = VK_NULL_HANDLE;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = 64;
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkCreateBuffer(device, &info, NULL, &b);
        CHECK(get_data(device, VK_OBJECT_TYPE_BUFFER, (uint64_t)(uintptr_t)b, s1) == 0);
        CHECK_VK(vkSetPrivateData(device, VK_OBJECT_TYPE_BUFFER, (uint64_t)(uintptr_t)b, s1, 99 + i), VK_SUCCESS);
        vkDestroyBuffer(device, b, NULL);
    }

    /* Many objects: the table grows. */
    {
        enum { M = 2000 };
        static VkFence fences[M];
        VkFenceCreateInfo info;
        uint32_t bad = 0;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        for (i = 0; i < M; i++) {
            vkCreateFence(device, &info, NULL, &fences[i]);
            CHECK_VK(vkSetPrivateData(device, VK_OBJECT_TYPE_FENCE, (uint64_t)(uintptr_t)fences[i], s2, i * 3 + 1),
                     VK_SUCCESS);
        }
        for (i = 0; i < M; i += 2) /* drop half */
            vkDestroyFence(device, fences[i], NULL);
        for (i = 1; i < M; i += 2)
            bad += get_data(device, VK_OBJECT_TYPE_FENCE, (uint64_t)(uintptr_t)fences[i], s2) != i * 3 + 1;
        CHECK(bad == 0);
        for (i = 1; i < M; i += 2)
            vkDestroyFence(device, fences[i], NULL);
    }
    vkDestroyPrivateDataSlot(device, s1, NULL);
    vkDestroyPrivateDataSlot(device, s2, NULL);
    vkDestroyPrivateDataSlot(device, VK_NULL_HANDLE, NULL);

    /* A new slot (possibly at a freed one's address) starts empty. */
    s1 = make_slot(device);
    for (i = 0; i < n; i++)
        CHECK(get_data(device, objs[i].type, objs[i].handle, s1) == 0);
    vkDestroyPrivateDataSlot(device, s1, NULL);
    vkFreeCommandBuffers(device, pool, 1, &cb);
    free_buf(&a);
    vkDestroyEvent(device, e, NULL);
    vkDestroySemaphore(device, sem, NULL);
}

/* VkDevicePrivateDataCreateInfo: reserved slots work like others, also
 * when created, destroyed and created again, and beyond the reservation. */
static void test_private_data_reserved(void)
{
    VkDevice d = create_device(2);
    VkPrivateDataSlot s[4];
    uint32_t round, i;
    REQUIRE(d);
    for (round = 0; round < 2; round++) {
        for (i = 0; i < 4; i++)
            s[i] = make_slot(d);
        for (i = 0; i < 4; i++) {
            CHECK(get_data(d, VK_OBJECT_TYPE_DEVICE, (uint64_t)(uintptr_t)d, s[i]) == 0);
            CHECK_VK(vkSetPrivateData(d, VK_OBJECT_TYPE_DEVICE, (uint64_t)(uintptr_t)d, s[i], 50 + i), VK_SUCCESS);
        }
        for (i = 0; i < 4; i++)
            CHECK(get_data(d, VK_OBJECT_TYPE_DEVICE, (uint64_t)(uintptr_t)d, s[i]) == 50 + i);
        for (i = 0; i < 4; i++)
            vkDestroyPrivateDataSlot(d, s[i], NULL);
    }
    vkDestroyDevice(d, NULL);
}

/* ---- Small commands ---- */

static void test_small_commands(void)
{
    VkPhysicalDeviceExternalBufferInfo buffer_info;
    VkExternalBufferProperties buffer_props;
    VkPhysicalDeviceExternalFenceInfo fence_info;
    VkExternalFenceProperties fence_props;
    VkPhysicalDeviceExternalSemaphoreInfo sem_info;
    VkExternalSemaphoreProperties sem_props;
    VkPhysicalDeviceToolProperties tool;
    uint32_t count = 5, tools = ((struct pvk_physical_device *)pd)->instance->debug ? 1 : 0; /* the debug layer */

    CHECK_VK(vkGetPhysicalDeviceToolProperties(pd, &count, NULL), VK_SUCCESS);
    CHECK(count == tools);
    count = 1;
    CHECK_VK(vkGetPhysicalDeviceToolProperties(pd, &count, &tool), VK_SUCCESS);
    CHECK(count == tools);

    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    memset(&buffer_props, 0xff, sizeof(buffer_props));
    buffer_props.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    buffer_props.pNext = NULL;
    vkGetPhysicalDeviceExternalBufferProperties(pd, &buffer_info, &buffer_props);
    CHECK(buffer_props.sType == VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES && buffer_props.pNext == NULL);
    CHECK(buffer_props.externalMemoryProperties.externalMemoryFeatures == 0);
    CHECK(buffer_props.externalMemoryProperties.exportFromImportedHandleTypes == 0);
    CHECK(buffer_props.externalMemoryProperties.compatibleHandleTypes == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT);

    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO;
    fence_info.handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
    memset(&fence_props, 0xff, sizeof(fence_props));
    fence_props.sType = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES;
    fence_props.pNext = NULL;
    vkGetPhysicalDeviceExternalFenceProperties(pd, &fence_info, &fence_props);
    CHECK(fence_props.externalFenceFeatures == 0 && fence_props.compatibleHandleTypes == 0 &&
          fence_props.exportFromImportedHandleTypes == 0 && fence_props.pNext == NULL);

    memset(&sem_info, 0, sizeof(sem_info));
    sem_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    sem_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    memset(&sem_props, 0xff, sizeof(sem_props));
    sem_props.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
    sem_props.pNext = NULL;
    vkGetPhysicalDeviceExternalSemaphoreProperties(pd, &sem_info, &sem_props);
    CHECK(sem_props.externalSemaphoreFeatures == 0 && sem_props.compatibleHandleTypes == 0 &&
          sem_props.exportFromImportedHandleTypes == 0 && sem_props.pNext == NULL);

    /* A device mask of 1 changes nothing. */
    {
        struct buf a = make_buf(64);
        VkCommandBuffer cb = begin();
        vkCmdSetDeviceMask(cb, 1);
        vkCmdFillBuffer(cb, a.buffer, 0, 64, 0x77777777u);
        host_barrier(cb);
        run(cb);
        CHECK(all_words(&a, 64, 0x77777777u));
        free_buf(&a);
    }
}

/* vkQueueBindSparse (no queue supports sparse binding, so valid usage forbids
 * calls; PlasmaVK still handles them): an empty call signals its fence; batches
 * wait for and signal semaphores in order with other submissions. */
static void test_bind_sparse(void)
{
    VkSemaphore t = make_timeline();
    VkFenceCreateInfo fence_info;
    VkTimelineSemaphoreSubmitInfo values;
    VkBindSparseInfo bind;
    uint64_t wait = 1, signal = 2, v = 0;
    VkFence f;

    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    REQUIRE_VK(vkCreateFence(device, &fence_info, NULL, &f));
    CHECK_VK(vkQueueBindSparse(queue, 0, NULL, f), VK_SUCCESS);
    CHECK_VK(vkWaitForFences(device, 1, &f, VK_TRUE, 5000000000ull), VK_SUCCESS);
    vkResetFences(device, 1, &f);

    memset(&values, 0, sizeof(values));
    values.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    values.waitSemaphoreValueCount = 1;
    values.pWaitSemaphoreValues = &wait;
    values.signalSemaphoreValueCount = 1;
    values.pSignalSemaphoreValues = &signal;
    memset(&bind, 0, sizeof(bind));
    bind.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
    bind.pNext = &values;
    bind.waitSemaphoreCount = 1;
    bind.pWaitSemaphores = &t;
    bind.signalSemaphoreCount = 1;
    bind.pSignalSemaphores = &t;
    CHECK_VK(vkQueueBindSparse(queue, 1, &bind, f), VK_SUCCESS);
    CHECK_VK(vkWaitForFences(device, 1, &f, VK_TRUE, 1000000), VK_TIMEOUT);
    signal_value(t, 1);
    CHECK_VK(vkWaitForFences(device, 1, &f, VK_TRUE, 5000000000ull), VK_SUCCESS);
    vkGetSemaphoreCounterValue(device, t, &v);
    CHECK(v == 2);
    vkDestroyFence(device, f, NULL);
    vkDestroySemaphore(device, t, NULL);
}

int main(void)
{
    setup();
    if (!device)
        return 1;
    RUN(test_event_host);
    RUN(test_event_device_set_reset);
    RUN(test_event_device_set_host_sees);
    RUN(test_event_waits);
    RUN(test_private_data_objects);
    RUN(test_private_data_reserved);
    RUN(test_small_commands);
    RUN(test_bind_sparse);
    teardown();
    return TEST_RESULT();
}
