/* Fences, binary and timeline semaphores, queue and device idle waits
 * (section 16). The GPU is held back with a timeline semaphore wait that
 * the host signals later. */
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <pthread.h>
#include <string.h>
#include <time.h>

#include "../pvk_test.h"

static VkInstance instance;
static VkDevice device;
static VkQueue queues[2];
static VkCommandPool pool;
static uint32_t shared_type = UINT32_MAX;
static uint32_t private_type = UINT32_MAX;

struct buf {
    VkBuffer buffer;
    VkDeviceMemory memory;
    unsigned char *data;
};

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

static void sleep_ms(unsigned ms)
{
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000;
    nanosleep(&ts, NULL);
}

static void setup(void)
{
    VkPhysicalDeviceMemoryProperties mem;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceFeatures2 features;
    VkInstanceCreateInfo instance_info;
    VkApplicationInfo app;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo device_info;
    VkCommandPoolCreateInfo pool_info;
    VkPhysicalDevice pd;
    float priorities[2] = { 1.0f, 1.0f };
    uint32_t count = 1, i;

    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_4;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    REQUIRE_VK(vkCreateInstance(&instance_info, NULL, &instance));
    REQUIRE_VK(vkEnumeratePhysicalDevices(instance, &count, &pd));

    /* Timeline semaphores are core in 1.2: enabled when reported. */
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&features, 0, sizeof(features));
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &v12;
    vkGetPhysicalDeviceFeatures2(pd, &features);
    i = v12.timelineSemaphore;
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.timelineSemaphore = i ? VK_TRUE : VK_FALSE;

    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 2;
    queue_info.pQueuePriorities = priorities;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &v12;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    REQUIRE_VK(vkCreateDevice(pd, &device_info, NULL, &device));
    vkGetDeviceQueue(device, 0, 0, &queues[0]);
    vkGetDeviceQueue(device, 0, 1, &queues[1]);
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    for (i = 0; i < mem.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mem.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) && shared_type == UINT32_MAX)
            shared_type = i;
        if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && private_type == UINT32_MAX)
            private_type = i;
    }
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
    if (type == shared_type && vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS) {
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

static VkFence make_fence(bool signaled)
{
    VkFenceCreateInfo info;
    VkFence f = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    info.flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0;
    vkCreateFence(device, &info, NULL, &f);
    return f;
}

static VkSemaphore make_semaphore(bool timeline, uint64_t initial)
{
    VkSemaphoreTypeCreateInfo type;
    VkSemaphoreCreateInfo info;
    VkSemaphore s = VK_NULL_HANDLE;
    memset(&type, 0, sizeof(type));
    type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type.semaphoreType = timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY;
    type.initialValue = initial;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    info.pNext = &type;
    vkCreateSemaphore(device, &info, NULL, &s);
    return s;
}

static VkResult wait_value(VkSemaphore s, uint64_t value, uint64_t timeout)
{
    VkSemaphoreWaitInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    info.semaphoreCount = 1;
    info.pSemaphores = &s;
    info.pValues = &value;
    return vkWaitSemaphores(device, &info, timeout);
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

static VkCommandBuffer record_copy(const struct buf *src, const struct buf *dst, VkDeviceSize size)
{
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferBeginInfo info;
    VkMemoryBarrier b;
    VkBufferCopy region;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &alloc, &cb);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkBeginCommandBuffer(cb, &info);
    memset(&region, 0, sizeof(region));
    region.size = size;
    vkCmdCopyBuffer(cb, src->buffer, dst->buffer, 1, &region);
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &b, 0, NULL, 0, NULL);
    vkEndCommandBuffer(cb);
    return cb;
}

static VkCommandBuffer record_fill(const struct buf *dst, uint32_t value)
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
    vkCmdFillBuffer(cb, dst->buffer, 0, VK_WHOLE_SIZE, value);
    vkEndCommandBuffer(cb);
    return cb;
}

/* One batch on queue q: wait for (wait, wait_value), run cb, signal
 * (signal, signal_value); timeline values are ignored for binary ones. */
static VkResult submit(VkQueue q, VkSemaphore wait, uint64_t wait_value_, VkCommandBuffer cb, VkSemaphore signal,
                       uint64_t signal_value_, VkFence f)
{
    VkTimelineSemaphoreSubmitInfo values;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo info;
    memset(&values, 0, sizeof(values));
    values.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    values.waitSemaphoreValueCount = wait ? 1 : 0;
    values.pWaitSemaphoreValues = &wait_value_;
    values.signalSemaphoreValueCount = signal ? 1 : 0;
    values.pSignalSemaphoreValues = &signal_value_;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.pNext = &values;
    info.waitSemaphoreCount = wait ? 1 : 0;
    info.pWaitSemaphores = &wait;
    info.pWaitDstStageMask = &stage;
    info.commandBufferCount = cb ? 1 : 0;
    info.pCommandBuffers = &cb;
    info.signalSemaphoreCount = signal ? 1 : 0;
    info.pSignalSemaphores = &signal;
    return vkQueueSubmit(q, 1, &info, f);
}

static int all_words(const struct buf *b, VkDeviceSize size, uint32_t want)
{
    VkDeviceSize i;
    for (i = 0; i < size; i += 4) {
        uint32_t w;
        memcpy(&w, b->data + i, 4);
        if (w != want) {
            fprintf(stderr, "  word at %llu is %08x, want %08x\n", (unsigned long long)i, w, want);
            return 0;
        }
    }
    return 1;
}

/* ---- Fences ---- */

static void test_fence_states(void)
{
    VkFence a = make_fence(false), b = make_fence(true), both[2];
    CHECK_VK(vkGetFenceStatus(device, a), VK_NOT_READY);
    CHECK_VK(vkWaitForFences(device, 1, &a, VK_TRUE, 0), VK_TIMEOUT);
    CHECK_VK(vkGetFenceStatus(device, b), VK_SUCCESS);
    CHECK_VK(vkWaitForFences(device, 1, &b, VK_TRUE, 0), VK_SUCCESS);
    CHECK_VK(vkWaitForFences(device, 1, &b, VK_TRUE, UINT64_MAX), VK_SUCCESS);
    both[0] = a;
    both[1] = b;
    CHECK_VK(vkWaitForFences(device, 2, both, VK_FALSE, UINT64_MAX), VK_SUCCESS); /* b is enough */
    CHECK_VK(vkWaitForFences(device, 2, both, VK_TRUE, 1000000), VK_TIMEOUT);    /* a never signals */
    CHECK_VK(vkResetFences(device, 1, &b), VK_SUCCESS);
    CHECK_VK(vkGetFenceStatus(device, b), VK_NOT_READY);
    CHECK_VK(submit(queues[0], VK_NULL_HANDLE, 0, VK_NULL_HANDLE, VK_NULL_HANDLE, 0, b), VK_SUCCESS);
    CHECK_VK(vkWaitForFences(device, 1, &b, VK_TRUE, 10000000000ull), VK_SUCCESS);
    CHECK_VK(vkGetFenceStatus(device, b), VK_SUCCESS);
    vkDestroyFence(device, a, NULL);
    vkDestroyFence(device, b, NULL);
    vkDestroyFence(device, VK_NULL_HANDLE, NULL);
}

/* A submission held back by a semaphore: timeouts of 0, 1 ns (rounds up to
 * a millisecond) and a few ms expire; an infinite wait returns once the host
 * lets the GPU go. */
static void test_fence_timeouts(void)
{
    VkSemaphore gate = make_semaphore(true, 0);
    VkFence f = make_fence(false);
    uint64_t t0, dt;

    REQUIRE_VK(submit(queues[0], gate, 1, VK_NULL_HANDLE, VK_NULL_HANDLE, 0, f));
    CHECK_VK(vkGetFenceStatus(device, f), VK_NOT_READY);
    CHECK_VK(vkWaitForFences(device, 1, &f, VK_TRUE, 0), VK_TIMEOUT);
    CHECK_VK(vkWaitForFences(device, 1, &f, VK_FALSE, 1), VK_TIMEOUT);
    t0 = now_ns();
    CHECK_VK(vkWaitForFences(device, 1, &f, VK_TRUE, 5000000), VK_TIMEOUT);
    dt = now_ns() - t0;
    CHECK(dt >= 5000000 && dt < 1000000000);
    signal_value(gate, 1);
    CHECK_VK(vkWaitForFences(device, 1, &f, VK_TRUE, UINT64_MAX), VK_SUCCESS);
    CHECK_VK(vkGetFenceStatus(device, f), VK_SUCCESS);
    CHECK_VK(vkQueueWaitIdle(queues[0]), VK_SUCCESS);
    vkDestroyFence(device, f, NULL);
    vkDestroySemaphore(device, gate, NULL);
}

/* Any/all over fences on different queues, one held back. */
static void test_fence_any_all(void)
{
    VkSemaphore gate = make_semaphore(true, 0);
    VkFence f[2];
    f[0] = make_fence(false);
    f[1] = make_fence(false);
    REQUIRE_VK(submit(queues[0], gate, 1, VK_NULL_HANDLE, VK_NULL_HANDLE, 0, f[0]));
    REQUIRE_VK(submit(queues[1], VK_NULL_HANDLE, 0, VK_NULL_HANDLE, VK_NULL_HANDLE, 0, f[1]));
    CHECK_VK(vkWaitForFences(device, 2, f, VK_FALSE, 10000000000ull), VK_SUCCESS);
    CHECK_VK(vkGetFenceStatus(device, f[1]), VK_SUCCESS);
    CHECK_VK(vkWaitForFences(device, 2, f, VK_TRUE, 2000000), VK_TIMEOUT);
    signal_value(gate, 1);
    CHECK_VK(vkWaitForFences(device, 2, f, VK_TRUE, 10000000000ull), VK_SUCCESS);
    vkDestroyFence(device, f[0], NULL);
    vkDestroyFence(device, f[1], NULL);
    vkDestroySemaphore(device, gate, NULL);
}

struct waiter {
    VkFence fence;
    VkSemaphore semaphore;
    uint64_t value;
    VkResult result;
};

static void *fence_waiter(void *arg)
{
    struct waiter *w = arg;
    w->result = vkWaitForFences(device, 1, &w->fence, VK_TRUE, UINT64_MAX);
    return NULL;
}

static void *semaphore_waiter(void *arg)
{
    struct waiter *w = arg;
    w->result = wait_value(w->semaphore, w->value, UINT64_MAX);
    return NULL;
}

/* Waiting forever on a fence nobody submitted yet: another thread submits. */
static void test_fence_other_thread(void)
{
    struct waiter w;
    pthread_t thread;
    memset(&w, 0, sizeof(w));
    w.fence = make_fence(false);
    w.result = VK_ERROR_UNKNOWN;
    REQUIRE(pthread_create(&thread, NULL, fence_waiter, &w) == 0);
    sleep_ms(20);
    CHECK(w.result == VK_ERROR_UNKNOWN);
    CHECK_VK(submit(queues[0], VK_NULL_HANDLE, 0, VK_NULL_HANDLE, VK_NULL_HANDLE, 0, w.fence), VK_SUCCESS);
    pthread_join(thread, NULL);
    CHECK_VK(w.result, VK_SUCCESS);
    vkDestroyFence(device, w.fence, NULL);
}

/* ---- Timeline semaphores ---- */

static void test_timeline_host(void)
{
    VkSemaphore a = make_semaphore(true, 5), b = make_semaphore(true, 0), both[2];
    VkSemaphoreWaitInfo info;
    uint64_t value = 0, values[2], t0;
    struct waiter w;
    pthread_t thread;

    CHECK_VK(vkGetSemaphoreCounterValue(device, a, &value), VK_SUCCESS);
    CHECK(value == 5);
    signal_value(a, 10);
    CHECK_VK(vkGetSemaphoreCounterValue(device, a, &value), VK_SUCCESS);
    CHECK(value == 10);
    CHECK_VK(wait_value(a, 10, 0), VK_SUCCESS);
    CHECK_VK(wait_value(a, 7, UINT64_MAX), VK_SUCCESS);
    CHECK_VK(wait_value(a, 11, 0), VK_TIMEOUT);
    t0 = now_ns();
    CHECK_VK(wait_value(a, 11, 3000000), VK_TIMEOUT);
    CHECK(now_ns() - t0 >= 3000000);

    both[0] = b;
    both[1] = a;
    values[0] = 1;
    values[1] = 10;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    info.semaphoreCount = 2;
    info.pSemaphores = both;
    info.pValues = values;
    CHECK_VK(vkWaitSemaphores(device, &info, 1000000), VK_TIMEOUT); /* all: b is at 0 */
    info.flags = VK_SEMAPHORE_WAIT_ANY_BIT;
    CHECK_VK(vkWaitSemaphores(device, &info, UINT64_MAX), VK_SUCCESS);

    /* A thread waits forever; this one signals. */
    memset(&w, 0, sizeof(w));
    w.semaphore = b;
    w.value = 3;
    w.result = VK_ERROR_UNKNOWN;
    REQUIRE(pthread_create(&thread, NULL, semaphore_waiter, &w) == 0);
    sleep_ms(10);
    signal_value(b, 2);
    sleep_ms(10);
    CHECK(w.result == VK_ERROR_UNKNOWN);
    signal_value(b, 3);
    pthread_join(thread, NULL);
    CHECK_VK(w.result, VK_SUCCESS);
    vkDestroySemaphore(device, a, NULL);
    vkDestroySemaphore(device, b, NULL);
}

/* GPU work waits for a value the host signals later (wait-before-signal)
 * and signals another value the host waits for. */
static void test_timeline_gpu(void)
{
    struct buf src = make_buf(1 << 20, shared_type), dst = make_buf(1 << 20, shared_type);
    VkSemaphore in = make_semaphore(true, 0), out = make_semaphore(true, 0);
    VkCommandBuffer cb = record_copy(&src, &dst, 1 << 20);
    uint64_t value = 1;

    REQUIRE(src.data && dst.data);
    memset(src.data, 0x42, 1 << 20);
    REQUIRE_VK(submit(queues[0], in, 3, cb, out, 9, VK_NULL_HANDLE));
    sleep_ms(5);
    CHECK_VK(vkGetSemaphoreCounterValue(device, out, &value), VK_SUCCESS);
    CHECK(value == 0);
    CHECK(dst.data[0] == 0 && dst.data[(1 << 20) - 1] == 0);
    CHECK_VK(wait_value(out, 9, 2000000), VK_TIMEOUT);
    signal_value(in, 2); /* not enough */
    CHECK_VK(wait_value(out, 9, 2000000), VK_TIMEOUT);
    signal_value(in, 3);
    CHECK_VK(wait_value(out, 9, 10000000000ull), VK_SUCCESS);
    CHECK_VK(vkGetSemaphoreCounterValue(device, out, &value), VK_SUCCESS);
    CHECK(value == 9);
    CHECK(all_words(&dst, 1 << 20, 0x42424242u));
    CHECK_VK(vkQueueWaitIdle(queues[0]), VK_SUCCESS);
    vkFreeCommandBuffers(device, pool, 1, &cb);
    vkDestroySemaphore(device, in, NULL);
    vkDestroySemaphore(device, out, NULL);
    free_buf(&src);
    free_buf(&dst);
}

/* A wait longer than the ~4 s after which a Metal queue's event wait gives
 * up: the work behind it must still wait for the host. Also an
 * unsignalled semaphore wait followed by more work on the same queue, which
 * must wait too (submission order). */
static void test_timeline_long_wait(void)
{
    struct buf src = make_buf(65536, shared_type), dst = make_buf(65536, shared_type);
    struct buf dst2 = make_buf(65536, shared_type);
    VkSemaphore gate = make_semaphore(true, 0);
    VkCommandBuffer cb = record_copy(&src, &dst, 65536), cb2 = record_copy(&dst, &dst2, 65536);
    VkFence f = make_fence(false);

    REQUIRE(src.data && dst.data && dst2.data);
    memset(src.data, 0x24, 65536);
    REQUIRE_VK(submit(queues[0], gate, 1, cb, VK_NULL_HANDLE, 0, VK_NULL_HANDLE));
    REQUIRE_VK(submit(queues[0], VK_NULL_HANDLE, 0, cb2, VK_NULL_HANDLE, 0, f));
    sleep_ms(4500);
    CHECK(dst.data[0] == 0 && dst2.data[0] == 0);
    CHECK_VK(vkGetFenceStatus(device, f), VK_NOT_READY);
    signal_value(gate, 1);
    CHECK_VK(vkWaitForFences(device, 1, &f, VK_TRUE, 10000000000ull), VK_SUCCESS);
    CHECK(dst.data[0] == 0x24 && dst.data[65535] == 0x24);
    vkFreeCommandBuffers(device, pool, 1, &cb);
    vkFreeCommandBuffers(device, pool, 1, &cb2);
    vkDestroyFence(device, f, NULL);
    vkDestroySemaphore(device, gate, NULL);
    free_buf(&src);
    free_buf(&dst);
    free_buf(&dst2);
}

/* Barrier stages (white box: pvk_barrier_stages), and a barrier's ordering
 * of later encoders lasting until the timeline shows its submission done
 * (pvk_queue_order). */
static void test_barrier_order(void)
{
    struct pvk_queue *q = (struct pvk_queue *)queues[0];
    struct buf a = make_buf(4096, shared_type);
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferBeginInfo info;
    VkMemoryBarrier b;
    VkCommandBuffer cb = VK_NULL_HANDLE, fill;
    VkFence f = make_fence(false);

    CHECK(pvk_barrier_stages(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, true) ==
          (PVK_MTLStageFragment | PVK_MTLStageTile));
    CHECK(pvk_barrier_stages(VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT, false) == (PVK_MTLStageVertex | PVK_MTLStageDispatch));
    CHECK(pvk_barrier_stages(VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, false) == PVK_MTLStageVertex);
    CHECK(pvk_barrier_stages(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, false) == PVK_MTLStageDispatch);
    CHECK(pvk_barrier_stages(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, true) == 0);
    CHECK(pvk_barrier_stages(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, false) == 0);
    CHECK(pvk_barrier_stages(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, false) == PVK_STAGES_ALL);
    CHECK(pvk_barrier_stages(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, true) == PVK_STAGES_ALL);
    CHECK(pvk_barrier_stages(VK_PIPELINE_STAGE_2_TRANSFER_BIT, false) == PVK_STAGES_ALL);
    CHECK(pvk_barrier_stages(VK_PIPELINE_STAGE_2_HOST_BIT, true) == 0);

    REQUIRE(a.data);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    REQUIRE_VK(vkAllocateCommandBuffers(device, &alloc, &cb));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &info);
    vkCmdFillBuffer(cb, a.buffer, 0, VK_WHOLE_SIZE, 1);
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &b, 0, NULL, 0,
                         NULL);
    vkEndCommandBuffer(cb);
    REQUIRE_VK(vkQueueWaitIdle(queues[0]));
    REQUIRE_VK(submit(queues[0], VK_NULL_HANDLE, 0, cb, VK_NULL_HANDLE, 0, f));
    REQUIRE(q->orders.count >= 1);
    CHECK(q->orders.at[q->orders.count - 1].after == PVK_STAGES_ALL &&
          q->orders.at[q->orders.count - 1].before == (PVK_MTLStageFragment | PVK_MTLStageTile));
    REQUIRE_VK(vkWaitForFences(device, 1, &f, VK_TRUE, 10000000000ull));
    fill = record_fill(&a, 2);
    REQUIRE_VK(submit(queues[0], VK_NULL_HANDLE, 0, fill, VK_NULL_HANDLE, 0, VK_NULL_HANDLE)); /* (done: dropped) */
    CHECK(q->orders.count == 0);
    CHECK_VK(vkQueueWaitIdle(queues[0]), VK_SUCCESS);
    CHECK(((uint32_t *)a.data)[0] == 2);
    vkFreeCommandBuffers(device, pool, 1, &cb);
    vkFreeCommandBuffers(device, pool, 1, &fill);
    vkDestroyFence(device, f, NULL);
    free_buf(&a);
}

/* A render pass's dependencies on outside work are barriers before and
 * after its instances (white box: pvk_queue_order); nothing else orders
 * passes that sample what earlier ones rendered. */
static void test_render_pass_external_dependencies(void)
{
    struct pvk_queue *q = (struct pvk_queue *)queues[0];
    VkSubpassDescription subpass;
    VkSubpassDependency deps[2];
    VkRenderPassCreateInfo rp_info;
    VkFramebufferCreateInfo fb_info;
    VkRenderPassBeginInfo begin;
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferBeginInfo info;
    VkRenderPass rp = VK_NULL_HANDLE;
    VkFramebuffer fb = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence f = make_fence(false);
    const pvk_MTLStages fragment = PVK_MTLStageFragment | PVK_MTLStageTile;

    memset(&subpass, 0, sizeof(subpass));
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    memset(deps, 0, sizeof(deps));
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    memset(&rp_info, 0, sizeof(rp_info));
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.subpassCount = 1;
    rp_info.pSubpasses = &subpass;
    rp_info.dependencyCount = 2;
    rp_info.pDependencies = deps;
    REQUIRE_VK(vkCreateRenderPass(device, &rp_info, NULL, &rp));
    memset(&fb_info, 0, sizeof(fb_info));
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass = rp;
    fb_info.width = fb_info.height = 16;
    fb_info.layers = 1;
    REQUIRE_VK(vkCreateFramebuffer(device, &fb_info, NULL, &fb));
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    REQUIRE_VK(vkAllocateCommandBuffers(device, &alloc, &cb));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &info);
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = rp;
    begin.framebuffer = fb;
    begin.renderArea.extent.width = begin.renderArea.extent.height = 16;
    vkCmdBeginRenderPass(cb, &begin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdEndRenderPass(cb);
    vkEndCommandBuffer(cb);
    REQUIRE_VK(vkQueueWaitIdle(queues[0]));
    REQUIRE_VK(submit(queues[0], VK_NULL_HANDLE, 0, cb, VK_NULL_HANDLE, 0, f));
    REQUIRE(q->orders.count >= 1);
    CHECK(q->orders.at[q->orders.count - 1].after == fragment);
    CHECK(q->orders.at[q->orders.count - 1].before == (fragment | PVK_MTLStageDispatch));
    CHECK_VK(vkWaitForFences(device, 1, &f, VK_TRUE, 10000000000ull), VK_SUCCESS);
    vkFreeCommandBuffers(device, pool, 1, &cb);
    vkDestroyFence(device, f, NULL);
    vkDestroyFramebuffer(device, fb, NULL);
    vkDestroyRenderPass(device, rp, NULL);
}

/* ---- Binary semaphores and queues ---- */

/* Nothing waits on the queue's thread: every wait so far was resolved when
 * submitted (white box: section 16). */
static bool no_backlog(VkQueue queue)
{
    struct pvk_queue *q = (struct pvk_queue *)queue;
    bool none;
    pthread_mutex_lock(&q->lock);
    none = q->deferred == NULL;
    pthread_mutex_unlock(&q->lock);
    return none;
}

/* Submissions ordered only by a semaphore (no barriers, no host waits),
 * on one queue and across two, reusing the semaphore each round: the queue's
 * order covers the wait, or the other queue waits on the GPU for the signal
 * the first one issued, never on its thread. */
static void test_binary_semaphores(void)
{
    enum { SIZE = 16u << 20 };
    struct buf a = make_buf(SIZE, private_type), out = make_buf(SIZE, shared_type);
    VkSemaphore sem = make_semaphore(false, 0);
    VkFence f = make_fence(false);
    uint32_t round;

    REQUIRE(a.buffer && out.data);
    for (round = 0; round < 6; round++) {
        uint32_t value = 0x10203040u + round * 0x01010101u;
        VkCommandBuffer fill = record_fill(&a, value), copy = record_copy(&a, &out, SIZE);
        VkQueue second = queues[round & 1]; /* same queue, then the other one */
        REQUIRE_VK(submit(queues[0], VK_NULL_HANDLE, 0, fill, sem, 0, VK_NULL_HANDLE));
        REQUIRE_VK(submit(second, sem, 0, copy, VK_NULL_HANDLE, 0, f));
        CHECK(no_backlog(second));
        REQUIRE_VK(vkWaitForFences(device, 1, &f, VK_TRUE, 10000000000ull));
        CHECK(all_words(&out, SIZE, value));
        vkResetFences(device, 1, &f);
        vkFreeCommandBuffers(device, pool, 1, &fill);
        vkFreeCommandBuffers(device, pool, 1, &copy);
    }
    CHECK_VK(vkDeviceWaitIdle(device), VK_SUCCESS);
    vkDestroyFence(device, f, NULL);
    vkDestroySemaphore(device, sem, NULL);
    free_buf(&a);
    free_buf(&out);
}

/* Batches of one vkQueueSubmit ordered only by semaphores (binary, then
 * timeline): resolved when submitting, by a barrier on the queue. */
static void test_semaphore_same_submit(void)
{
    enum { SIZE = 16u << 20 };
    struct buf a = make_buf(SIZE, private_type), b = make_buf(SIZE, private_type), out = make_buf(SIZE, shared_type);
    VkSemaphore bin = make_semaphore(false, 0), tl = make_semaphore(true, 0);
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkTimelineSemaphoreSubmitInfo values[3];
    uint64_t zero = 0;
    VkFence f = make_fence(false);
    VkSubmitInfo info[3];
    uint32_t round;

    REQUIRE(a.buffer && b.buffer && out.data);
    for (round = 0; round < 4; round++) {
        uint32_t value = 0x7700AA00u + round;
        VkCommandBuffer fill = record_fill(&a, value), copy = record_copy(&a, &b, SIZE), copy2 = record_copy(&b, &out, SIZE);
        uint64_t next = round + 1;
        memset(info, 0, sizeof(info));
        memset(values, 0, sizeof(values));
        info[0].sType = info[1].sType = info[2].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info[0].commandBufferCount = info[1].commandBufferCount = info[2].commandBufferCount = 1;
        info[0].pCommandBuffers = &fill;
        info[0].signalSemaphoreCount = 1;
        info[0].pSignalSemaphores = &bin;
        info[1].waitSemaphoreCount = 1;
        info[1].pWaitSemaphores = &bin;
        info[1].pWaitDstStageMask = &stage;
        info[1].pCommandBuffers = &copy;
        info[1].signalSemaphoreCount = 1;
        info[1].pSignalSemaphores = &tl;
        values[1].sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        values[1].waitSemaphoreValueCount = 1;
        values[1].pWaitSemaphoreValues = &zero; /* ignored: binary */
        values[1].signalSemaphoreValueCount = 1;
        values[1].pSignalSemaphoreValues = &next;
        info[1].pNext = &values[1];
        info[2].waitSemaphoreCount = 1;
        info[2].pWaitSemaphores = &tl;
        info[2].pWaitDstStageMask = &stage;
        info[2].pCommandBuffers = &copy2;
        values[2].sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        values[2].waitSemaphoreValueCount = 1;
        values[2].pWaitSemaphoreValues = &next;
        info[2].pNext = &values[2];
        REQUIRE_VK(vkQueueSubmit(queues[0], 3, info, f));
        REQUIRE_VK(vkWaitForFences(device, 1, &f, VK_TRUE, 10000000000ull));
        CHECK(all_words(&out, SIZE, value));
        vkResetFences(device, 1, &f);
        vkFreeCommandBuffers(device, pool, 1, &fill);
        vkFreeCommandBuffers(device, pool, 1, &copy);
        vkFreeCommandBuffers(device, pool, 1, &copy2);
    }
    vkDestroyFence(device, f, NULL);
    vkDestroySemaphore(device, bin, NULL);
    vkDestroySemaphore(device, tl, NULL);
    free_buf(&a);
    free_buf(&b);
    free_buf(&out);
}

/* vkQueueSubmit2: a timeline chain across both queues, each step waiting
 * for the previous one's value. */
static void test_submit2_chain(void)
{
    enum { SIZE = 4u << 20, STEPS = 6 };
    struct buf a = make_buf(SIZE, private_type), b = make_buf(SIZE, private_type), out = make_buf(SIZE, shared_type);
    VkSemaphore t = make_semaphore(true, 100);
    VkCommandBuffer cbs[STEPS + 1];
    VkSemaphoreSubmitInfo wait, signal;
    VkCommandBufferSubmitInfo cbi;
    VkSubmitInfo2 info;
    uint32_t k;

    REQUIRE(a.buffer && b.buffer && out.data);
    cbs[0] = record_fill(&a, 0x5EED5EEDu);
    for (k = 1; k < STEPS; k++)
        cbs[k] = record_copy((k & 1) ? &a : &b, (k & 1) ? &b : &a, SIZE);
    cbs[STEPS] = record_copy((STEPS & 1) ? &a : &b, &out, SIZE);
    memset(&wait, 0, sizeof(wait));
    wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    wait.semaphore = t;
    wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    signal = wait;
    memset(&cbi, 0, sizeof(cbi));
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    info.waitSemaphoreInfoCount = 1;
    info.pWaitSemaphoreInfos = &wait;
    info.commandBufferInfoCount = 1;
    info.pCommandBufferInfos = &cbi;
    info.signalSemaphoreInfoCount = 1;
    info.pSignalSemaphoreInfos = &signal;
    /* Everything waits for the host's first signal. */
    for (k = 0; k <= STEPS; k++) {
        wait.value = 101 + k;
        signal.value = 102 + k;
        cbi.commandBuffer = cbs[k];
        REQUIRE_VK(vkQueueSubmit2(queues[k & 1], 1, &info, VK_NULL_HANDLE));
    }
    sleep_ms(5);
    CHECK(out.data[0] == 0);
    signal_value(t, 101);
    CHECK_VK(wait_value(t, 102 + STEPS, 10000000000ull), VK_SUCCESS);
    CHECK(all_words(&out, SIZE, 0x5EED5EEDu));
    CHECK_VK(vkDeviceWaitIdle(device), VK_SUCCESS);
    for (k = 0; k <= STEPS; k++)
        vkFreeCommandBuffers(device, pool, 1, &cbs[k]);
    vkDestroySemaphore(device, t, NULL);
    free_buf(&a);
    free_buf(&b);
    free_buf(&out);
}

/* ---- Lazy signals under threads ----
 *
 * A binary semaphore signalled by the last batch of a submission the queue
 * commits at once shows on the host only through the queue's timeline
 * (section 16, pvk_queue_signal). Two threads hand rounds back and forth on
 * two queues through two binary semaphores: P fills round r's part of `a`
 * after waiting T and signals S; C copies that part to `out` after waiting
 * S, signals T and the round's fence. Rounds put either queue, or both,
 * behind a timeline gate a third thread opens after a random delay, so the
 * waits are resolved by the GPU (the other queue issued the signal) or by
 * the queue's thread on the host (a lazy signal's timeline value, or an
 * eager signal from the other queue's thread); a fourth thread waits for
 * every round's fence and checks its part. A wait for a value a lazy signal
 * never sets hangs here. */

enum { LAZY_ROUNDS = 240, LAZY_PART = 65536 };

static struct {
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int signalled, waited;            /* rounds P signalled S for, C waited for S */
    VkSemaphore gate_sem[16];         /* gate requests: opened in order */
    uint64_t gate_value[16];
    uint32_t gate_head, gate_tail;
    bool done;
    VkSemaphore s, t, gates[2];
    uint64_t gate_next[2];
    VkCommandBuffer fill[LAZY_ROUNDS], copy[LAZY_ROUNDS];
    VkFence fences[LAZY_ROUNDS];
    struct buf out;
    int errors[3]; /* P, C, waiter: test macros are not thread safe */
} lazy;

static uint32_t lazy_value(int round)
{
    return 0xC0DE0000u + (uint32_t)round;
}

static void lazy_pause(unsigned *seed)
{
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = (long)(rand_r(seed) % 300) * 1000;
    if (ts.tv_nsec >= 100000)
        nanosleep(&ts, NULL);
}

/* Puts queue q behind its gate (an empty batch waiting for the next value)
 * and asks the gate thread to open it. */
static VkResult lazy_gate(uint32_t q)
{
    VkTimelineSemaphoreSubmitInfo values;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo info;
    uint64_t value = ++lazy.gate_next[q];
    VkResult r;
    memset(&values, 0, sizeof(values));
    values.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    values.waitSemaphoreValueCount = 1;
    values.pWaitSemaphoreValues = &value;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.pNext = &values;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &lazy.gates[q];
    info.pWaitDstStageMask = &stage;
    r = vkQueueSubmit(queues[q], 1, &info, VK_NULL_HANDLE);
    pthread_mutex_lock(&lazy.lock);
    while (lazy.gate_tail - lazy.gate_head == 16)
        pthread_cond_wait(&lazy.cond, &lazy.lock);
    lazy.gate_sem[lazy.gate_tail % 16] = lazy.gates[q];
    lazy.gate_value[lazy.gate_tail % 16] = value;
    lazy.gate_tail++;
    pthread_cond_broadcast(&lazy.cond);
    pthread_mutex_unlock(&lazy.lock);
    return r;
}

static void *lazy_gate_thread(void *arg)
{
    unsigned seed = 7;
    (void)arg;
    pthread_mutex_lock(&lazy.lock);
    for (;;) {
        VkSemaphore sem;
        uint64_t value;
        while (lazy.gate_head == lazy.gate_tail && !lazy.done)
            pthread_cond_wait(&lazy.cond, &lazy.lock);
        if (lazy.gate_head == lazy.gate_tail)
            break;
        sem = lazy.gate_sem[lazy.gate_head % 16];
        value = lazy.gate_value[lazy.gate_head % 16];
        lazy.gate_head++;
        pthread_cond_broadcast(&lazy.cond);
        pthread_mutex_unlock(&lazy.lock);
        lazy_pause(&seed);
        signal_value(sem, value);
        pthread_mutex_lock(&lazy.lock);
    }
    pthread_mutex_unlock(&lazy.lock);
    return NULL;
}

static void *lazy_producer(void *arg)
{
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    unsigned seed = 1;
    int r;
    (void)arg;
    for (r = 0; r < LAZY_ROUNDS; r++) {
        VkSubmitInfo info;
        pthread_mutex_lock(&lazy.lock);
        while (lazy.waited < r - 1) /* T's previous signal is waited for, and S's */
            pthread_cond_wait(&lazy.cond, &lazy.lock);
        pthread_mutex_unlock(&lazy.lock);
        if ((r & 2) && lazy_gate(0) != VK_SUCCESS)
            lazy.errors[0]++;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.waitSemaphoreCount = r ? 1 : 0;
        info.pWaitSemaphores = &lazy.t;
        info.pWaitDstStageMask = &stage;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &lazy.fill[r];
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &lazy.s;
        if (vkQueueSubmit(queues[0], 1, &info, VK_NULL_HANDLE) != VK_SUCCESS)
            lazy.errors[0]++;
        pthread_mutex_lock(&lazy.lock);
        lazy.signalled = r;
        pthread_cond_broadcast(&lazy.cond);
        pthread_mutex_unlock(&lazy.lock);
        lazy_pause(&seed);
    }
    return NULL;
}

static void *lazy_consumer(void *arg)
{
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    unsigned seed = 2;
    int r;
    (void)arg;
    for (r = 0; r < LAZY_ROUNDS; r++) {
        VkSubmitInfo info;
        pthread_mutex_lock(&lazy.lock);
        while (lazy.signalled < r)
            pthread_cond_wait(&lazy.cond, &lazy.lock);
        pthread_mutex_unlock(&lazy.lock);
        if ((r & 1) && lazy_gate(1) != VK_SUCCESS)
            lazy.errors[1]++;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores = &lazy.s;
        info.pWaitDstStageMask = &stage;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &lazy.copy[r];
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &lazy.t;
        if (vkQueueSubmit(queues[1], 1, &info, lazy.fences[r]) != VK_SUCCESS)
            lazy.errors[1]++;
        pthread_mutex_lock(&lazy.lock);
        lazy.waited = r;
        pthread_cond_broadcast(&lazy.cond);
        pthread_mutex_unlock(&lazy.lock);
        lazy_pause(&seed);
    }
    return NULL;
}

static void *lazy_fence_waiter(void *arg)
{
    int r;
    (void)arg;
    for (r = 0; r < LAZY_ROUNDS; r++) {
        struct buf part = lazy.out;
        if (vkWaitForFences(device, 1, &lazy.fences[r], VK_TRUE, 20000000000ull) != VK_SUCCESS) {
            fprintf(stderr, "  round %d: fence not signalled\n", r);
            lazy.errors[2]++;
            break;
        }
        part.data = lazy.out.data + (size_t)r * LAZY_PART;
        if (!all_words(&part, LAZY_PART, lazy_value(r)))
            lazy.errors[2]++;
    }
    return NULL;
}

static void test_lazy_signal_threads(void)
{
    struct buf a = make_buf((VkDeviceSize)LAZY_ROUNDS * LAZY_PART, private_type);
    pthread_t threads[4];
    void *(*fns[4])(void *) = { lazy_gate_thread, lazy_producer, lazy_consumer, lazy_fence_waiter };
    VkBufferCopy region;
    VkMemoryBarrier b;
    int r, i;

    memset(&lazy, 0, sizeof(lazy));
    lazy.out = make_buf((VkDeviceSize)LAZY_ROUNDS * LAZY_PART, shared_type);
    REQUIRE(a.buffer && lazy.out.data);
    REQUIRE(pthread_mutex_init(&lazy.lock, NULL) == 0 && pthread_cond_init(&lazy.cond, NULL) == 0);
    lazy.signalled = lazy.waited = -1;
    lazy.s = make_semaphore(false, 0);
    lazy.t = make_semaphore(false, 0);
    lazy.gates[0] = make_semaphore(true, 0);
    lazy.gates[1] = make_semaphore(true, 0);
    for (r = 0; r < LAZY_ROUNDS; r++) {
        VkCommandBufferAllocateInfo alloc;
        VkCommandBufferBeginInfo info;
        memset(&alloc, 0, sizeof(alloc));
        alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool = pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        REQUIRE_VK(vkAllocateCommandBuffers(device, &alloc, &lazy.fill[r]));
        vkBeginCommandBuffer(lazy.fill[r], &info);
        vkCmdFillBuffer(lazy.fill[r], a.buffer, (VkDeviceSize)r * LAZY_PART, LAZY_PART, lazy_value(r));
        vkEndCommandBuffer(lazy.fill[r]);
        REQUIRE_VK(vkAllocateCommandBuffers(device, &alloc, &lazy.copy[r]));
        vkBeginCommandBuffer(lazy.copy[r], &info);
        memset(&region, 0, sizeof(region));
        region.srcOffset = region.dstOffset = (VkDeviceSize)r * LAZY_PART;
        region.size = LAZY_PART;
        vkCmdCopyBuffer(lazy.copy[r], a.buffer, lazy.out.buffer, 1, &region);
        memset(&b, 0, sizeof(b));
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(lazy.copy[r], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &b, 0, NULL,
                             0, NULL);
        vkEndCommandBuffer(lazy.copy[r]);
        lazy.fences[r] = make_fence(false);
    }
    for (i = 0; i < 4; i++)
        REQUIRE(pthread_create(&threads[i], NULL, fns[i], NULL) == 0);
    for (i = 3; i > 0; i--)
        pthread_join(threads[i], NULL);
    pthread_mutex_lock(&lazy.lock);
    lazy.done = true;
    pthread_cond_broadcast(&lazy.cond);
    pthread_mutex_unlock(&lazy.lock);
    pthread_join(threads[0], NULL);
    CHECK(lazy.errors[0] == 0 && lazy.errors[1] == 0 && lazy.errors[2] == 0);
    CHECK_VK(vkDeviceWaitIdle(device), VK_SUCCESS);
    for (r = 0; r < LAZY_ROUNDS; r++) {
        vkFreeCommandBuffers(device, pool, 1, &lazy.fill[r]);
        vkFreeCommandBuffers(device, pool, 1, &lazy.copy[r]);
        vkDestroyFence(device, lazy.fences[r], NULL);
    }
    vkDestroySemaphore(device, lazy.s, NULL);
    vkDestroySemaphore(device, lazy.t, NULL);
    vkDestroySemaphore(device, lazy.gates[0], NULL);
    vkDestroySemaphore(device, lazy.gates[1], NULL);
    pthread_cond_destroy(&lazy.cond);
    pthread_mutex_destroy(&lazy.lock);
    free_buf(&a);
    free_buf(&lazy.out);
}

/* Many submissions without fences, each filling its own 64 KiB, then idle
 * waits see all of them done. */
static void test_wait_idle(void)
{
    enum { N = 64, PART = 65536 };
    struct buf a = make_buf((VkDeviceSize)N * PART, shared_type);
    VkCommandBuffer cbs[N];
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferBeginInfo info;
    uint32_t k;
    int ok = 1;

    REQUIRE(a.data);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = N;
    REQUIRE_VK(vkAllocateCommandBuffers(device, &alloc, cbs));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    for (k = 0; k < N; k++) {
        vkBeginCommandBuffer(cbs[k], &info);
        vkCmdFillBuffer(cbs[k], a.buffer, (VkDeviceSize)k * PART, PART, 0x01000000u + k);
        vkEndCommandBuffer(cbs[k]);
        REQUIRE_VK(submit(queues[k & 1], VK_NULL_HANDLE, 0, cbs[k], VK_NULL_HANDLE, 0, VK_NULL_HANDLE));
        if (k == N / 2 - 1) {
            uint32_t j;
            CHECK_VK(vkQueueWaitIdle(queues[1]), VK_SUCCESS);
            CHECK_VK(vkQueueWaitIdle(queues[0]), VK_SUCCESS);
            for (j = 0; j <= k && ok; j++) {
                struct buf part = a;
                part.data = a.data + (size_t)j * PART;
                ok = all_words(&part, PART, 0x01000000u + j);
            }
        }
    }
    CHECK_VK(vkDeviceWaitIdle(device), VK_SUCCESS);
    for (k = 0; k < N && ok; k++) {
        struct buf part = a;
        part.data = a.data + (size_t)k * PART;
        ok = all_words(&part, PART, 0x01000000u + k);
    }
    CHECK(ok);
    vkFreeCommandBuffers(device, pool, N, cbs);
    free_buf(&a);
}

int main(void)
{
    setup();
    if (pvk_test_failures)
        return 1;
    RUN(test_fence_states);
    RUN(test_fence_timeouts);
    RUN(test_fence_any_all);
    RUN(test_fence_other_thread);
    RUN(test_timeline_host);
    RUN(test_timeline_gpu);
    RUN(test_timeline_long_wait);
    RUN(test_barrier_order);
    RUN(test_render_pass_external_dependencies);
    RUN(test_binary_semaphores);
    RUN(test_semaphore_same_submit);
    RUN(test_submit2_chain);
    RUN(test_lazy_signal_threads);
    RUN(test_wait_idle);
    teardown();
    return TEST_RESULT();
}
