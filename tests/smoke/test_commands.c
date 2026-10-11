/* Command pools and buffers, transfer commands and pipeline barriers
 * (sections 15 and 16), checked against host-visible memory. */
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <string.h>

#include "../pvk_test.h"

static VkDevice device;
static VkQueue queue;
static VkCommandPool pool;
static VkFence fence;
static uint32_t shared_type = UINT32_MAX;  /* host-visible and coherent */
static uint32_t private_type = UINT32_MAX; /* device-local only */

struct buf {
    VkBuffer buffer;
    VkDeviceMemory memory;
    unsigned char *data; /* NULL for private memory */
    VkDeviceSize size;
};

static VkInstance instance;

static void setup(void)
{
    VkPhysicalDeviceMemoryProperties mem;
    VkInstanceCreateInfo instance_info;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo device_info;
    VkCommandPoolCreateInfo pool_info;
    VkFenceCreateInfo fence_info;
    VkPhysicalDevice pd;
    float priority = 1.0f;
    uint32_t count = 1, i;

    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    REQUIRE_VK(vkCreateInstance(&instance_info, NULL, &instance));
    REQUIRE_VK(vkEnumeratePhysicalDevices(instance, &count, &pd));
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    REQUIRE_VK(vkCreateDevice(pd, &device_info, NULL, &device));
    vkGetDeviceQueue(device, 0, 0, &queue);
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
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    REQUIRE_VK(vkCreateFence(device, &fence_info, NULL, &fence));
}

static void teardown(void)
{
    vkDestroyFence(device, fence, NULL);
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
    if (vkCreateBuffer(device, &info, NULL, &b.buffer) != VK_SUCCESS)
        return b;
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    if (vkAllocateMemory(device, &alloc, NULL, &b.memory) != VK_SUCCESS)
        return b;
    vkBindBufferMemory(device, b.buffer, b.memory, 0);
    if (type == shared_type && vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS) {
        b.data = p;
        memset(b.data, 0, (size_t)size);
    }
    b.size = size;
    return b;
}

static void free_buf(struct buf *b)
{
    vkDestroyBuffer(device, b->buffer, NULL);
    vkFreeMemory(device, b->memory, NULL);
    memset(b, 0, sizeof(*b));
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
    if (vkAllocateCommandBuffers(device, &alloc, &cb) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &info);
    return cb;
}

static VkResult submit_wait(VkCommandBuffer cb)
{
    VkSubmitInfo submit;
    VkResult r;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = cb ? 1 : 0;
    submit.pCommandBuffers = &cb;
    vkResetFences(device, 1, &fence);
    r = vkQueueSubmit(queue, 1, &submit, fence);
    if (r != VK_SUCCESS)
        return r;
    return vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ull);
}

/* End, submit, wait, free. */
static VkResult finish(VkCommandBuffer cb)
{
    VkResult r = vkEndCommandBuffer(cb);
    if (r == VK_SUCCESS)
        r = submit_wait(cb);
    vkFreeCommandBuffers(device, pool, 1, &cb);
    return r;
}

static void barrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst)
{
    VkMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, src, dst, 0, 1, &b, 0, NULL, 0, NULL);
}

static void transfer_barrier(VkCommandBuffer cb)
{
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
}

static int check_words(const struct buf *b, VkDeviceSize from, VkDeviceSize to, uint32_t want)
{
    VkDeviceSize i;
    for (i = from; i < to; i += 4) {
        uint32_t w;
        memcpy(&w, b->data + i, 4);
        if (w != want) {
            fprintf(stderr, "  word at %llu is %08x, want %08x\n", (unsigned long long)i, w, want);
            return 0;
        }
    }
    return 1;
}

/* ---- Fill ---- */

/* A byte pattern (Metal's fill) and a word pattern (the internal kernel),
 * at offsets, with VK_WHOLE_SIZE rounding down to whole words. */
static void test_fill(void)
{
    struct buf b = make_buf(65536 + 6, shared_type);
    VkCommandBuffer cb;
    REQUIRE(b.data);
    memset(b.data, 0xEE, (size_t)b.size);
    cb = begin();
    vkCmdFillBuffer(cb, b.buffer, 0, VK_WHOLE_SIZE, 0x5A5A5A5Au);
    transfer_barrier(cb);
    vkCmdFillBuffer(cb, b.buffer, 4096, 8192, 0x12345678u);
    vkCmdFillBuffer(cb, b.buffer, 32768, 4, 0xA5A5A5A5u);
    vkCmdFillBuffer(cb, b.buffer, 40000, 400, 0xCAFEF00Du);
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
    REQUIRE_VK(finish(cb));
    CHECK(check_words(&b, 0, 4096, 0x5A5A5A5Au));
    CHECK(check_words(&b, 4096, 12288, 0x12345678u));
    CHECK(check_words(&b, 12288, 32768, 0x5A5A5A5Au));
    CHECK(check_words(&b, 32768, 32772, 0xA5A5A5A5u));
    CHECK(check_words(&b, 32772, 40000, 0x5A5A5A5Au));
    CHECK(check_words(&b, 40000, 40400, 0xCAFEF00Du));
    CHECK(check_words(&b, 40400, 65536 + 4, 0x5A5A5A5Au));
    CHECK(b.data[65536 + 4] == 0xEE && b.data[65536 + 5] == 0xEE); /* past the last whole word */
    free_buf(&b);
}

/* Fills of a large buffer: several kernel dispatches' worth is not needed,
 * but many threadgroups and a tail. */
static void test_fill_large(void)
{
    VkDeviceSize size = 64u << 20;
    struct buf b = make_buf(size, shared_type);
    VkCommandBuffer cb;
    REQUIRE(b.data);
    cb = begin();
    vkCmdFillBuffer(cb, b.buffer, 4, size - 12, 0x01020304u);
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
    REQUIRE_VK(finish(cb));
    CHECK(check_words(&b, 0, 4, 0));
    CHECK(check_words(&b, 4, size - 8, 0x01020304u));
    CHECK(check_words(&b, size - 8, size, 0));
    free_buf(&b);
}

/* ---- Copy and update ---- */

/* Byte-granular regions (Vulkan has no alignment rule for copies), several
 * per command, through private memory and back, and vkCmdCopyBuffer2. */
static void test_copy(void)
{
    struct buf src = make_buf(4096, shared_type), mid = make_buf(4096, private_type), dst = make_buf(4096, shared_type);
    VkBufferCopy regions[3];
    VkBufferCopy2 regions2[2];
    VkCopyBufferInfo2 info2;
    VkCommandBuffer cb;
    uint32_t i;
    int ok = 1;

    REQUIRE(src.data && mid.buffer && dst.data);
    for (i = 0; i < 4096; i++)
        src.data[i] = (unsigned char)(i * 7 + 3);
    memset(dst.data, 0xEE, 4096);
    memset(regions, 0, sizeof(regions));
    regions[0].size = 4096; /* src -> mid, whole */
    cb = begin();
    vkCmdCopyBuffer(cb, src.buffer, mid.buffer, 1, regions);
    transfer_barrier(cb);
    regions[0].srcOffset = 3;
    regions[0].dstOffset = 1001;
    regions[0].size = 13;
    regions[1].srcOffset = 100;
    regions[1].dstOffset = 2000;
    regions[1].size = 1000;
    regions[2].srcOffset = 4095;
    regions[2].dstOffset = 0;
    regions[2].size = 1;
    vkCmdCopyBuffer(cb, mid.buffer, dst.buffer, 3, regions);
    memset(regions2, 0, sizeof(regions2));
    regions2[0].sType = regions2[1].sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
    regions2[0].srcOffset = 10;
    regions2[0].dstOffset = 3500;
    regions2[0].size = 7;
    regions2[1].srcOffset = 0;
    regions2[1].dstOffset = 3600;
    regions2[1].size = 496;
    memset(&info2, 0, sizeof(info2));
    info2.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
    info2.srcBuffer = mid.buffer;
    info2.dstBuffer = dst.buffer;
    info2.regionCount = 2;
    info2.pRegions = regions2;
    vkCmdCopyBuffer2(cb, &info2);
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
    REQUIRE_VK(finish(cb));
    for (i = 0; i < 4096; i++) {
        unsigned char want = 0xEE;
        if (i >= 1001 && i < 1014)
            want = (unsigned char)((i - 1001 + 3) * 7 + 3);
        else if (i >= 2000 && i < 3000)
            want = (unsigned char)((i - 2000 + 100) * 7 + 3);
        else if (i == 0)
            want = (unsigned char)(4095 * 7 + 3);
        else if (i >= 3500 && i < 3507)
            want = (unsigned char)((i - 3500 + 10) * 7 + 3);
        else if (i >= 3600)
            want = (unsigned char)((i - 3600) * 7 + 3);
        if (dst.data[i] != want && ok) {
            fprintf(stderr, "  dst[%u] = %02x, want %02x\n", i, dst.data[i], want);
            ok = 0;
        }
    }
    CHECK(ok);
    free_buf(&src);
    free_buf(&mid);
    free_buf(&dst);
}

/* Updates go through the upload ring: small, the 64 KiB maximum, and many
 * to one place in order (the last one wins). */
static void test_update(void)
{
    static uint32_t big[16384];
    struct buf b = make_buf(65536 + 256, shared_type);
    VkCommandBuffer cb;
    uint32_t i, v;
    REQUIRE(b.data);
    for (i = 0; i < 16384; i++)
        big[i] = i * 2654435761u;
    cb = begin();
    vkCmdUpdateBuffer(cb, b.buffer, 256, sizeof(big), big);
    for (i = 0; i < 100; i++) {
        transfer_barrier(cb);
        vkCmdUpdateBuffer(cb, b.buffer, 4, 4, &i);
    }
    v = 0xABCDEF01u;
    vkCmdUpdateBuffer(cb, b.buffer, 252, 4, &v);
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
    REQUIRE_VK(finish(cb));
    CHECK(memcmp(b.data + 256, big, sizeof(big)) == 0);
    CHECK(check_words(&b, 4, 8, 99));
    CHECK(check_words(&b, 252, 256, 0xABCDEF01u));
    CHECK(check_words(&b, 0, 4, 0) && check_words(&b, 8, 252, 0));
    free_buf(&b);
}

/* ---- Barriers ---- */

/* A chain of dependent transfers, each step reading what the previous one
 * wrote: without working barriers (Metal 4 orders nothing by itself) the
 * steps overlap and stale data travels down the chain. Mixes blits and the
 * fill kernel, and private memory. */
static void test_barrier_chain(void)
{
    enum { STEPS = 48, SIZE = 4u << 20 };
    struct buf a = make_buf(SIZE, private_type), b = make_buf(SIZE, private_type), out = make_buf(SIZE, shared_type);
    VkBufferCopy region;
    VkCommandBuffer cb;
    uint32_t k;

    REQUIRE(a.buffer && b.buffer && out.data);
    memset(&region, 0, sizeof(region));
    region.size = SIZE;
    cb = begin();
    vkCmdFillBuffer(cb, a.buffer, 0, VK_WHOLE_SIZE, 0x11223344u);
    for (k = 0; k < STEPS; k++) {
        transfer_barrier(cb);
        if (k % 8 == 7) { /* overwrite, then keep copying the new value */
            vkCmdFillBuffer(cb, (k & 1) ? b.buffer : a.buffer, 0, VK_WHOLE_SIZE, 0x0A0B0C00u + k);
            transfer_barrier(cb);
        }
        vkCmdCopyBuffer(cb, (k & 1) ? b.buffer : a.buffer, (k & 1) ? a.buffer : b.buffer, 1, &region);
    }
    transfer_barrier(cb);
    vkCmdCopyBuffer(cb, (STEPS & 1) ? b.buffer : a.buffer, out.buffer, 1, &region);
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
    REQUIRE_VK(finish(cb));
    CHECK(check_words(&out, 0, SIZE, 0x0A0B0C00u + 47));
    free_buf(&a);
    free_buf(&b);
    free_buf(&out);
}

/* A barrier orders against everything later in submission order, also in
 * the next submission (producer side), and one in a later submission orders
 * against earlier ones (consumer side). The synchronization2 form too. */
static void test_barrier_across_submits(void)
{
    enum { SIZE = 8u << 20 };
    struct buf a = make_buf(SIZE, private_type), a2 = make_buf(SIZE, private_type);
    struct buf out = make_buf(SIZE, shared_type), out2 = make_buf(SIZE, shared_type);
    VkCommandBuffer cb1, cb2, cb3, cb4;
    VkSubmitInfo submits[4];
    VkBufferCopy region;
    VkMemoryBarrier2 b2;
    VkDependencyInfo dep;
    uint32_t round;

    REQUIRE(a.buffer && a2.buffer && out.data && out2.data);
    memset(&region, 0, sizeof(region));
    region.size = SIZE;
    memset(&b2, 0, sizeof(b2));
    b2.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    b2.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT;
    b2.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b2.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b2.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    memset(&dep, 0, sizeof(dep));
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &b2;
    for (round = 0; round < 4; round++) {
        uint32_t value = 0x01010101u * (round + 1) + 0x00102030u, value2 = value ^ 0x0F0F0F0Fu;
        cb1 = begin(); /* fill, then a barrier at the very end */
        vkCmdFillBuffer(cb1, a.buffer, 0, VK_WHOLE_SIZE, value);
        transfer_barrier(cb1);
        vkEndCommandBuffer(cb1);
        cb2 = begin(); /* no barrier of its own: ordered by cb1's */
        vkCmdCopyBuffer(cb2, a.buffer, out.buffer, 1, &region);
        vkEndCommandBuffer(cb2);
        cb3 = begin(); /* fill without any barrier */
        vkCmdFillBuffer(cb3, a2.buffer, 0, VK_WHOLE_SIZE, value2);
        vkEndCommandBuffer(cb3);
        cb4 = begin(); /* barrier first: orders after cb3's fill */
        vkCmdPipelineBarrier2(cb4, &dep);
        vkCmdCopyBuffer(cb4, a2.buffer, out2.buffer, 1, &region);
        barrier(cb4, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
        vkEndCommandBuffer(cb4);
        memset(submits, 0, sizeof(submits));
        submits[0].sType = submits[1].sType = submits[2].sType = submits[3].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submits[0].commandBufferCount = submits[1].commandBufferCount = 1;
        submits[2].commandBufferCount = submits[3].commandBufferCount = 1;
        submits[0].pCommandBuffers = &cb1;
        submits[1].pCommandBuffers = &cb2;
        submits[2].pCommandBuffers = &cb3;
        submits[3].pCommandBuffers = &cb4;
        vkResetFences(device, 1, &fence);
        REQUIRE_VK(vkQueueSubmit(queue, 1, &submits[0], VK_NULL_HANDLE));
        REQUIRE_VK(vkQueueSubmit(queue, 3, &submits[1], fence));
        REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ull));
        CHECK(check_words(&out, 0, SIZE, value));
        CHECK(check_words(&out2, 0, SIZE, value2));
        vkFreeCommandBuffers(device, pool, 1, &cb1);
        vkFreeCommandBuffers(device, pool, 1, &cb2);
        vkFreeCommandBuffers(device, pool, 1, &cb3);
        vkFreeCommandBuffers(device, pool, 1, &cb4);
    }
    free_buf(&a);
    free_buf(&a2);
    free_buf(&out);
    free_buf(&out2);
}

/* ---- Command buffer lifecycle ---- */

/* Recorded once, submitted again and again: each submission reads the
 * memory as it is then. */
static void test_resubmit(void)
{
    struct buf src = make_buf(4096, shared_type), dst = make_buf(4096, shared_type);
    VkBufferCopy region;
    VkCommandBuffer cb;
    uint32_t round, value = 77;

    REQUIRE(src.data && dst.data);
    memset(&region, 0, sizeof(region));
    region.size = 4096;
    cb = begin();
    vkCmdCopyBuffer(cb, src.buffer, dst.buffer, 1, &region);
    transfer_barrier(cb);
    vkCmdUpdateBuffer(cb, dst.buffer, 0, 4, &value);
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
    REQUIRE_VK(vkEndCommandBuffer(cb));
    for (round = 0; round < 5; round++) {
        memset(src.data, (int)(round + 1), 4096);
        REQUIRE_VK(submit_wait(cb));
        CHECK(check_words(&dst, 0, 4, 77));
        CHECK(check_words(&dst, 4, 4096, 0x01010101u * (round + 1)));
    }
    vkFreeCommandBuffers(device, pool, 1, &cb);
    free_buf(&src);
    free_buf(&dst);
}

/* SIMULTANEOUS_USE: one command buffer in two batches of one submission. */
static void test_simultaneous(void)
{
    struct buf b = make_buf(1024, shared_type);
    VkCommandBufferBeginInfo info;
    VkSubmitInfo submits[2];
    VkCommandBuffer cb;
    uint32_t one = 1;

    REQUIRE(b.data);
    cb = begin();
    REQUIRE_VK(vkEndCommandBuffer(cb)); /* executable: a recording one must not be begun (the debug layer's 00049) */
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    REQUIRE_VK(vkBeginCommandBuffer(cb, &info)); /* re-begin: implicit reset */
    vkCmdUpdateBuffer(cb, b.buffer, 0, 4, &one);
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
    REQUIRE_VK(vkEndCommandBuffer(cb));
    memset(submits, 0, sizeof(submits));
    submits[0].sType = submits[1].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submits[0].commandBufferCount = submits[1].commandBufferCount = 1;
    submits[0].pCommandBuffers = submits[1].pCommandBuffers = &cb;
    vkResetFences(device, 1, &fence);
    REQUIRE_VK(vkQueueSubmit(queue, 2, submits, fence));
    REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ull));
    CHECK(check_words(&b, 0, 4, 1));
    vkFreeCommandBuffers(device, pool, 1, &cb);
    free_buf(&b);
}

/* Reset paths: vkResetCommandBuffer, vkResetCommandPool (with and without
 * releasing), vkTrimCommandPool; buffers re-record and still work. */
static void test_pool_reset(void)
{
    struct buf b = make_buf(4096, shared_type);
    VkCommandPoolCreateInfo pool_info;
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferBeginInfo begin_info;
    VkCommandBuffer cbs[3];
    VkCommandPool p;
    uint32_t i, round;

    REQUIRE(b.data);
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    REQUIRE_VK(vkCreateCommandPool(device, &pool_info, NULL, &p));
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = p;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 3;
    REQUIRE_VK(vkAllocateCommandBuffers(device, &alloc, cbs));
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    for (round = 0; round < 4; round++) {
        for (i = 0; i < 3; i++) {
            uint32_t v = round * 10 + i;
            REQUIRE_VK(vkBeginCommandBuffer(cbs[i], &begin_info));
            vkCmdUpdateBuffer(cbs[i], b.buffer, 4 * i, 4, &v);
            barrier(cbs[i], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
            REQUIRE_VK(vkEndCommandBuffer(cbs[i]));
            REQUIRE_VK(submit_wait(cbs[i]));
            CHECK(check_words(&b, 4 * i, 4 * i + 4, v));
        }
        switch (round) {
        case 0:
            CHECK_VK(vkResetCommandPool(device, p, 0), VK_SUCCESS);
            break;
        case 1:
            CHECK_VK(vkResetCommandPool(device, p, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT), VK_SUCCESS);
            break;
        case 2:
            for (i = 0; i < 3; i++)
                CHECK_VK(vkResetCommandBuffer(cbs[i], i ? VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT : 0), VK_SUCCESS);
            vkTrimCommandPool(device, p, 0);
            break;
        default:
            break;
        }
    }
    vkFreeCommandBuffers(device, p, 2, cbs); /* the third goes with the pool */
    vkDestroyCommandPool(device, p, NULL);
    free_buf(&b);
}

/* More packets than one storage block holds. */
static void test_many_commands(void)
{
    enum { N = 3000 };
    struct buf b = make_buf(4 * N, shared_type);
    VkCommandBuffer cb;
    uint32_t i;
    int ok = 1;
    REQUIRE(b.data);
    cb = begin();
    for (i = 0; i < N; i++)
        vkCmdFillBuffer(cb, b.buffer, 4 * i, 4, i * 0x01000193u);
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
    REQUIRE_VK(finish(cb));
    for (i = 0; i < N && ok; i++)
        ok = check_words(&b, 4 * i, 4 * i + 4, i * 0x01000193u);
    CHECK(ok);
    free_buf(&b);
}

/* Secondary command buffers run where vkCmdExecuteCommands is recorded. */
static void test_secondary(void)
{
    struct buf b = make_buf(1024, shared_type);
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferInheritanceInfo inherit;
    VkCommandBufferBeginInfo info;
    VkCommandBuffer sec[2], cb;
    uint32_t v = 5;

    REQUIRE(b.data);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    alloc.commandBufferCount = 2;
    REQUIRE_VK(vkAllocateCommandBuffers(device, &alloc, sec));
    memset(&inherit, 0, sizeof(inherit));
    inherit.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.pInheritanceInfo = &inherit;
    REQUIRE_VK(vkBeginCommandBuffer(sec[0], &info));
    vkCmdFillBuffer(sec[0], b.buffer, 0, 512, 0x33333333u);
    REQUIRE_VK(vkEndCommandBuffer(sec[0]));
    REQUIRE_VK(vkBeginCommandBuffer(sec[1], &info));
    vkCmdUpdateBuffer(sec[1], b.buffer, 0, 4, &v);
    REQUIRE_VK(vkEndCommandBuffer(sec[1]));
    cb = begin();
    vkCmdExecuteCommands(cb, 1, &sec[0]);
    transfer_barrier(cb);
    vkCmdExecuteCommands(cb, 1, &sec[1]);
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
    REQUIRE_VK(finish(cb));
    CHECK(check_words(&b, 0, 4, 5));
    CHECK(check_words(&b, 4, 512, 0x33333333u));
    CHECK(check_words(&b, 512, 1024, 0));
    vkFreeCommandBuffers(device, pool, 2, sec);
    free_buf(&b);
}

/* Submissions without command buffers, empty command buffers and no
 * batches at all still signal the fence after earlier work. */
static void test_empty_submits(void)
{
    VkCommandBuffer cb = begin();
    REQUIRE_VK(vkEndCommandBuffer(cb));
    CHECK_VK(submit_wait(cb), VK_SUCCESS);
    CHECK_VK(submit_wait(VK_NULL_HANDLE), VK_SUCCESS);
    vkResetFences(device, 1, &fence);
    CHECK_VK(vkQueueSubmit(queue, 0, NULL, fence), VK_SUCCESS);
    CHECK_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ull), VK_SUCCESS);
    CHECK_VK(vkQueueSubmit2(queue, 0, NULL, VK_NULL_HANDLE), VK_SUCCESS);
    CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);
    vkFreeCommandBuffers(device, pool, 1, &cb);
}

/* ---- Allocation callbacks ---- */

struct counter {
    long live, total;
};

static void *VKAPI_PTR count_alloc(void *user, size_t size, size_t alignment, VkSystemAllocationScope scope)
{
    struct counter *c = user;
    void *p;
    (void)scope;
    if (alignment > 16)
        return NULL;
    p = malloc(size);
    if (p) {
        c->live++;
        c->total++;
    }
    return p;
}

static void *VKAPI_PTR count_realloc(void *user, void *original, size_t size, size_t alignment,
                                     VkSystemAllocationScope scope)
{
    (void)user;
    (void)original;
    (void)size;
    (void)alignment;
    (void)scope;
    return NULL; /* PlasmaVK never reallocates */
}

static void VKAPI_PTR count_free(void *user, void *memory)
{
    struct counter *c = user;
    if (memory)
        c->live--;
    free(memory);
}

/* A pool's command buffers allocate through the pool's callbacks, and
 * destroying the pool frees everything. */
static void test_pool_allocator(void)
{
    struct counter c = { 0, 0 };
    VkAllocationCallbacks cbs;
    VkCommandPoolCreateInfo info;
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferBeginInfo begin_info;
    struct buf b = make_buf(65536, shared_type);
    VkCommandBuffer cb[4];
    VkCommandPool p;
    uint32_t i, k;

    REQUIRE(b.data);
    memset(&cbs, 0, sizeof(cbs));
    cbs.pUserData = &c;
    cbs.pfnAllocation = count_alloc;
    cbs.pfnReallocation = count_realloc;
    cbs.pfnFree = count_free;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    REQUIRE_VK(vkCreateCommandPool(device, &info, &cbs, &p));
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = p;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 4;
    REQUIRE_VK(vkAllocateCommandBuffers(device, &alloc, cb));
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    for (i = 0; i < 4; i++) {
        REQUIRE_VK(vkBeginCommandBuffer(cb[i], &begin_info));
        for (k = 0; k < 200; k++)
            vkCmdUpdateBuffer(cb[i], b.buffer, 0, 256, b.data); /* > one block */
        REQUIRE_VK(vkEndCommandBuffer(cb[i]));
    }
    CHECK(c.total > 5);
    vkFreeCommandBuffers(device, p, 1, cb);
    vkDestroyCommandPool(device, p, &cbs);
    CHECK(c.live == 0);
    free_buf(&b);
}

int main(void)
{
    setup();
    if (pvk_test_failures)
        return 1;
    RUN(test_fill);
    RUN(test_fill_large);
    RUN(test_copy);
    RUN(test_update);
    RUN(test_barrier_chain);
    RUN(test_barrier_across_submits);
    RUN(test_resubmit);
    RUN(test_simultaneous);
    RUN(test_pool_reset);
    RUN(test_many_commands);
    RUN(test_secondary);
    RUN(test_empty_submits);
    RUN(test_pool_allocator);
    teardown();
    return TEST_RESULT();
}
