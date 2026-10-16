/* VkDeviceMemory, VkBuffer, mapping, binding and device addresses. */
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <string.h>

#include "../pvk_test.h"

static VkInstance instance;
static VkPhysicalDevice physical_device;
static VkDevice device;
static VkQueue queue;
static VkPhysicalDeviceLimits limits;
static uint32_t private_type = UINT32_MAX; /* device-local, not host-visible */
static uint32_t shared_type = UINT32_MAX;  /* host-visible and coherent */

static void setup(void)
{
    VkPhysicalDeviceMemoryProperties mem;
    VkPhysicalDeviceProperties props;
    VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkInstanceCreateInfo instance_info;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo device_info;
    float priority = 1.0f;
    uint32_t count = 1, i;

    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    REQUIRE_VK(vkCreateInstance(&instance_info, NULL, &instance));
    REQUIRE_VK(vkEnumeratePhysicalDevices(instance, &count, &physical_device));
    REQUIRE(count == 1);

    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    REQUIRE_VK(vkCreateDevice(physical_device, &device_info, NULL, &device));
    vkGetDeviceQueue(device, 0, 0, &queue);

    vkGetPhysicalDeviceProperties(physical_device, &props);
    limits = props.limits;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem);
    for (i = 0; i < mem.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mem.memoryTypes[i].propertyFlags;
        if ((f & host) == host && shared_type == UINT32_MAX)
            shared_type = i;
        if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            private_type == UINT32_MAX)
            private_type = i;
    }
}

static VkResult allocate(uint32_t type, VkDeviceSize size, const VkAllocationCallbacks *alloc, VkDeviceMemory *memory)
{
    VkMemoryAllocateInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize = size;
    info.memoryTypeIndex = type;
    return vkAllocateMemory(device, &info, alloc, memory);
}

static VkResult create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, const VkAllocationCallbacks *alloc,
                              VkBuffer *buffer)
{
    VkBufferCreateInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    return vkCreateBuffer(device, &info, alloc, buffer);
}

static VkDeviceAddress address_of(VkBuffer buffer)
{
    VkBufferDeviceAddressInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(device, &info);
}

static int is_pow2(VkDeviceSize x)
{
    return x && !(x & (x - 1));
}

/* ---- Allocation callbacks that count live allocations ---- */

struct counter {
    int live;
    int total;
};

static void *VKAPI_PTR count_alloc(void *user, size_t size, size_t alignment, VkSystemAllocationScope scope)
{
    struct counter *c = user;
    void *p;
    (void)scope;
    if (alignment > 16)
        return NULL; /* malloc's guarantee; PlasmaVK asks for 16 */
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
    struct counter *c = user;
    (void)alignment;
    (void)scope;
    if (!original)
        return count_alloc(user, size, alignment, scope);
    if (size == 0) {
        free(original);
        c->live--;
        return NULL;
    }
    return realloc(original, size);
}

static void VKAPI_PTR count_free(void *user, void *memory)
{
    struct counter *c = user;
    if (!memory)
        return;
    free(memory);
    c->live--;
}

static VkAllocationCallbacks counting_callbacks(struct counter *c)
{
    VkAllocationCallbacks cb;
    memset(&cb, 0, sizeof(cb));
    cb.pUserData = c;
    cb.pfnAllocation = count_alloc;
    cb.pfnReallocation = count_realloc;
    cb.pfnFree = count_free;
    return cb;
}

/* ---- Raw Metal 4, to check what the GPU sees ---- */

#define MSG0(R, obj, sel) PVK_MSG(R, (id, SEL))(obj, sel_registerName(sel))

/* Copies between two MTLBuffers on the VkQueue's MTL4CommandQueue and waits.
 * Resources reach the GPU only through the device's residency set, which
 * PlasmaVK commits before each submission: so must this. */
static bool gpu_copy(id src, unsigned long src_offset, id dst, unsigned long dst_offset, unsigned long size)
{
    void *pool = objc_autoreleasePoolPush(); /* the encoder is autoreleased */
    id mtl = ((struct pvk_device *)device)->physical_device->mtl_device;
    id mtl_queue = ((struct pvk_queue *)queue)->mtl_queue;
    id allocator = MSG0(id, mtl, "newCommandAllocator");
    id cb = MSG0(id, mtl, "newCommandBuffer");
    id event = MSG0(id, mtl, "newSharedEvent");
    id encoder;
    bool done = false;

    if (allocator && cb && event) {
        PVK_MSG(void, (id, SEL, id))(cb, sel_registerName("beginCommandBufferWithAllocator:"), allocator);
        encoder = MSG0(id, cb, "computeCommandEncoder");
        PVK_MSG(void, (id, SEL, id, unsigned long, id, unsigned long, unsigned long))(
            encoder, sel_registerName("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
            src, src_offset, dst, dst_offset, size);
        MSG0(void, encoder, "endEncoding");
        MSG0(void, cb, "endCommandBuffer");
        pvk_device_commit_residency((struct pvk_device *)device);
        PVK_MSG(void, (id, SEL, const id *, unsigned long))(mtl_queue, sel_registerName("commit:count:"), &cb, 1);
        PVK_MSG(void, (id, SEL, id, uint64_t))(mtl_queue, sel_registerName("signalEvent:value:"), event, 1);
        done = PVK_MSG(bool, (id, SEL, uint64_t, uint64_t))(
            event, sel_registerName("waitUntilSignaledValue:timeoutMS:"), 1, 5000);
    }
    pvk_release(event);
    pvk_release(cb);
    pvk_release(allocator);
    objc_autoreleasePoolPop(pool);
    return done;
}

/* ---- Tests ---- */

static void test_limits(void)
{
    const struct pvk_physical_device *pd = ((struct pvk_device *)device)->physical_device;
    VkDeviceSize heap = pd->memory_properties.memoryHeaps[0].size, limit, remaining;
    VkPhysicalDeviceMaintenance3Properties m3;
    VkPhysicalDeviceMaintenance4Properties m4;
    VkPhysicalDeviceProperties2 props2;

    CHECK(limits.maxMemoryAllocationCount >= 4096);
    CHECK(limits.maxStorageBufferRange >= (1u << 27));
    CHECK(limits.maxUniformBufferRange >= 65536);
    CHECK(is_pow2(limits.minStorageBufferOffsetAlignment) && limits.minStorageBufferOffsetAlignment <= 256);
    CHECK(is_pow2(limits.minUniformBufferOffsetAlignment) && limits.minUniformBufferOffsetAlignment <= 256);
    CHECK(is_pow2(limits.minTexelBufferOffsetAlignment) && limits.minTexelBufferOffsetAlignment <= 256);
    CHECK(is_pow2(limits.minMemoryMapAlignment) && limits.minMemoryMapAlignment >= 64);
    CHECK(is_pow2(limits.nonCoherentAtomSize) && limits.nonCoherentAtomSize <= 256);
    CHECK(limits.bufferImageGranularity >= 1 && limits.bufferImageGranularity <= 131072);

    memset(&m3, 0, sizeof(m3));
    m3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
    memset(&m4, 0, sizeof(m4));
    m4.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES;
    m3.pNext = &m4;
    memset(&props2, 0, sizeof(props2));
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &m3;
    vkGetPhysicalDeviceProperties2(physical_device, &props2);
    CHECK(m3.maxMemoryAllocationSize >= (1u << 30)); /* the spec's minimum */
    CHECK(m4.maxBufferSize >= (1u << 30));

    /* The heap is the recommended working set, or the process's memory
     * limit if less (section 7, Memory heap), and no allocation is larger
     * (but for Vulkan's minimum). */
    if (pvk_process_limit(&limit, &remaining))
        CHECK(heap <= limit && heap <= pvk_MTLDevice_recommendedMaxWorkingSetSize(pd->mtl_device));
    else
        CHECK(heap == pvk_MTLDevice_recommendedMaxWorkingSetSize(pd->mtl_device)); /* macOS */
    CHECK(m3.maxMemoryAllocationSize <= heap || m3.maxMemoryAllocationSize == 1u << 30);
}

static void test_allocate_free(void)
{
    static const VkDeviceSize sizes[] = { 1, 256, 4096, 16384, 100000, (1u << 20) + 3 };
    uint32_t types[2];
    unsigned t, s;

    REQUIRE(private_type != UINT32_MAX && shared_type != UINT32_MAX);
    types[0] = private_type;
    types[1] = shared_type;
    for (t = 0; t < 2; t++) {
        for (s = 0; s < PVK_ARRAY_SIZE(sizes); s++) {
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkDeviceSize committed = 0;
            CHECK_VK(allocate(types[t], sizes[s], NULL, &memory), VK_SUCCESS);
            if (!memory)
                continue;
            vkGetDeviceMemoryCommitment(device, memory, &committed);
            CHECK(committed >= sizes[s]);
            CHECK(((struct pvk_device_memory *)memory)->gpu_address != 0);
            vkFreeMemory(device, memory, NULL);
        }
    }
    vkFreeMemory(device, VK_NULL_HANDLE, NULL);
    vkDestroyBuffer(device, VK_NULL_HANDLE, NULL);
}

static void test_map(void)
{
    const VkDeviceSize size = 65536, offset = 4096;
    VkDeviceMemory memory;
    VkMemoryMapInfo map_info;
    VkMemoryUnmapInfo unmap_info;
    VkMappedMemoryRange range;
    unsigned char *p, *q;
    void *v = NULL;
    VkDeviceSize i;
    int same = 1;

    REQUIRE_VK(allocate(shared_type, size, NULL, &memory));

    REQUIRE_VK(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &v));
    p = v;
    CHECK(((uintptr_t)p % limits.minMemoryMapAlignment) == 0);
    for (i = 0; i < size; i++)
        p[i] = (unsigned char)(i * 7 + 3);
    memset(&range, 0, sizeof(range));
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.size = VK_WHOLE_SIZE;
    CHECK_VK(vkFlushMappedMemoryRanges(device, 1, &range), VK_SUCCESS);
    vkUnmapMemory(device, memory);

    /* Mapping at an offset sees the same bytes. */
    REQUIRE_VK(vkMapMemory(device, memory, offset, 1024, 0, &v));
    q = v;
    for (i = 0; i < 1024; i++)
        same &= q[i] == (unsigned char)((i + offset) * 7 + 3);
    CHECK(same);
    q[0] = 0xAB;
    vkUnmapMemory(device, memory);

    /* vkMapMemory2 at an offset sees the write made through the last map. */
    memset(&map_info, 0, sizeof(map_info));
    map_info.sType = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    map_info.memory = memory;
    map_info.offset = offset - 16;
    map_info.size = VK_WHOLE_SIZE;
    REQUIRE_VK(vkMapMemory2(device, &map_info, &v));
    q = v;
    CHECK(q[16] == 0xAB);
    CHECK(q[15] == (unsigned char)((offset - 1) * 7 + 3));
    CHECK_VK(vkInvalidateMappedMemoryRanges(device, 1, &range), VK_SUCCESS);
    memset(&unmap_info, 0, sizeof(unmap_info));
    unmap_info.sType = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO;
    unmap_info.memory = memory;
    CHECK_VK(vkUnmapMemory2(device, &unmap_info), VK_SUCCESS);

    vkFreeMemory(device, memory, NULL);
}

static void test_map_private_fails(void)
{
    VkDeviceMemory memory;
    VkMemoryMapInfo map_info;
    void *v = (void *)1;

    REQUIRE_VK(allocate(private_type, 4096, NULL, &memory));
    CHECK_VK(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &v), VK_ERROR_MEMORY_MAP_FAILED);
    CHECK(v == NULL);
    memset(&map_info, 0, sizeof(map_info));
    map_info.sType = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    map_info.memory = memory;
    map_info.size = VK_WHOLE_SIZE;
    CHECK_VK(vkMapMemory2(device, &map_info, &v), VK_ERROR_MEMORY_MAP_FAILED);
    vkFreeMemory(device, memory, NULL);
}

static void test_buffer_requirements(void)
{
    static const VkDeviceSize sizes[] = { 1, 4, 17, 256, 1000, 65536, 1u << 20 };
    static const VkBufferUsageFlags usages[] = {
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
    };
    uint32_t both = (1u << private_type) | (1u << shared_type);
    unsigned s, u;

    for (s = 0; s < PVK_ARRAY_SIZE(sizes); s++) {
        for (u = 0; u < PVK_ARRAY_SIZE(usages); u++) {
            VkBufferCreateInfo info;
            VkBuffer buffer;
            VkMemoryRequirements req;
            VkMemoryRequirements2 req2, dev_req;
            VkMemoryDedicatedRequirements dedicated;
            VkBufferMemoryRequirementsInfo2 req_info;
            VkDeviceBufferMemoryRequirements dev_info;

            REQUIRE_VK(create_buffer(sizes[s], usages[u], NULL, &buffer));
            vkGetBufferMemoryRequirements(device, buffer, &req);
            CHECK(req.size >= sizes[s]);
            CHECK(is_pow2(req.alignment));
            CHECK(req.size % req.alignment == 0);
            CHECK((req.memoryTypeBits & both) == both);
            if (usages[u] & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
                CHECK(req.alignment % limits.minUniformBufferOffsetAlignment == 0);
            if (usages[u] & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
                CHECK(req.alignment % limits.minStorageBufferOffsetAlignment == 0);
            if (usages[u] & VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT)
                CHECK(req.alignment % limits.minTexelBufferOffsetAlignment == 0);

            memset(&dedicated, 0, sizeof(dedicated));
            dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
            dedicated.prefersDedicatedAllocation = VK_TRUE;
            dedicated.requiresDedicatedAllocation = VK_TRUE;
            memset(&req2, 0, sizeof(req2));
            req2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
            req2.pNext = &dedicated;
            memset(&req_info, 0, sizeof(req_info));
            req_info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
            req_info.buffer = buffer;
            vkGetBufferMemoryRequirements2(device, &req_info, &req2);
            CHECK(memcmp(&req2.memoryRequirements, &req, sizeof(req)) == 0);
            CHECK(!dedicated.prefersDedicatedAllocation && !dedicated.requiresDedicatedAllocation);

            memset(&info, 0, sizeof(info));
            info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            info.size = sizes[s];
            info.usage = usages[u];
            memset(&dev_info, 0, sizeof(dev_info));
            dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS;
            dev_info.pCreateInfo = &info;
            memset(&dev_req, 0, sizeof(dev_req));
            dev_req.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
            vkGetDeviceBufferMemoryRequirements(device, &dev_info, &dev_req);
            CHECK(memcmp(&dev_req.memoryRequirements, &req, sizeof(req)) == 0);

            vkDestroyBuffer(device, buffer, NULL);
        }
    }
}

static void test_usage_flags2(void)
{
    VkBufferUsageFlags2CreateInfo usage2;
    VkBufferCreateInfo info;
    VkBuffer buffer;

    memset(&usage2, 0, sizeof(usage2));
    usage2.sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO;
    usage2.usage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.pNext = &usage2;
    info.size = 64;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; /* ignored: usage2 wins */
    REQUIRE_VK(vkCreateBuffer(device, &info, NULL, &buffer));
    CHECK(((struct pvk_buffer *)buffer)->usage == usage2.usage);
    vkDestroyBuffer(device, buffer, NULL);
}

static void test_bind_and_addresses(void)
{
    enum { N = 4 };
    VkBuffer buffers[N];
    VkDeviceSize offsets[N];
    VkDeviceAddress addresses[N];
    VkMemoryRequirements req;
    VkBindBufferMemoryInfo binds[2];
    VkBindMemoryStatus status;
    VkResult status_result = VK_ERROR_UNKNOWN;
    VkDeviceMemory memory;
    VkDeviceAddress base;
    VkBufferDeviceAddressInfo addr_info;
    VkDeviceMemoryOpaqueCaptureAddressInfo mem_addr_info;
    unsigned i, j;

    for (i = 0; i < N; i++)
        REQUIRE_VK(create_buffer(1000, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                 NULL, &buffers[i]));
    vkGetBufferMemoryRequirements(device, buffers[0], &req);
    REQUIRE_VK(allocate(shared_type, req.size * N + req.alignment * 8, NULL, &memory));
    base = ((struct pvk_device_memory *)memory)->gpu_address;
    CHECK(base != 0);

    /* Spread out, not packed, and not in order. */
    for (i = 0; i < N; i++)
        offsets[i] = (N - 1 - i) * (req.size + req.alignment);

    REQUIRE_VK(vkBindBufferMemory(device, buffers[0], memory, offsets[0]));
    REQUIRE_VK(vkBindBufferMemory(device, buffers[1], memory, offsets[1]));
    memset(binds, 0, sizeof(binds));
    memset(&status, 0, sizeof(status));
    status.sType = VK_STRUCTURE_TYPE_BIND_MEMORY_STATUS;
    status.pResult = &status_result;
    for (i = 0; i < 2; i++) {
        binds[i].sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
        binds[i].buffer = buffers[2 + i];
        binds[i].memory = memory;
        binds[i].memoryOffset = offsets[2 + i];
    }
    binds[1].pNext = &status;
    REQUIRE_VK(vkBindBufferMemory2(device, 2, binds));
    CHECK(status_result == VK_SUCCESS);

    for (i = 0; i < N; i++) {
        addresses[i] = address_of(buffers[i]);
        CHECK(addresses[i] == base + offsets[i]);
        CHECK(addresses[i] % req.alignment == 0);
        for (j = 0; j < i; j++)
            CHECK(addresses[i] != addresses[j]);
    }

    /* Capture/replay is unsupported: no opaque addresses. */
    memset(&addr_info, 0, sizeof(addr_info));
    addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.buffer = buffers[0];
    CHECK(vkGetBufferOpaqueCaptureAddress(device, &addr_info) == 0);
    memset(&mem_addr_info, 0, sizeof(mem_addr_info));
    mem_addr_info.sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO;
    mem_addr_info.memory = memory;
    CHECK(vkGetDeviceMemoryOpaqueCaptureAddress(device, &mem_addr_info) == 0);

    for (i = 0; i < N; i++)
        vkDestroyBuffer(device, buffers[i], NULL);
    vkFreeMemory(device, memory, NULL);
}

static void test_oversized_allocation(void)
{
    VkPhysicalDeviceMaintenance3Properties m3;
    VkPhysicalDeviceProperties2 props2;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    struct counter c = { 0, 0 };
    VkAllocationCallbacks cb = counting_callbacks(&c);

    memset(&m3, 0, sizeof(m3));
    m3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
    memset(&props2, 0, sizeof(props2));
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &m3;
    vkGetPhysicalDeviceProperties2(physical_device, &props2);

    CHECK_VK(allocate(private_type, m3.maxMemoryAllocationSize + 1, &cb, &memory), VK_ERROR_OUT_OF_DEVICE_MEMORY);
    CHECK_VK(allocate(shared_type, (VkDeviceSize)1 << 62, &cb, &memory), VK_ERROR_OUT_OF_DEVICE_MEMORY);
    CHECK(memory == VK_NULL_HANDLE);
    CHECK(c.live == 0);

    /* The device still works afterwards. */
    CHECK_VK(allocate(private_type, 4096, &cb, &memory), VK_SUCCESS);
    vkFreeMemory(device, memory, &cb);
    CHECK(c.live == 0);
}

static void test_allocation_callbacks(void)
{
    struct counter c = { 0, 0 };
    VkAllocationCallbacks cb = counting_callbacks(&c);
    VkDeviceMemory memory[2];
    VkBuffer buffer;

    REQUIRE_VK(allocate(private_type, 65536, &cb, &memory[0]));
    REQUIRE_VK(allocate(shared_type, 65536, &cb, &memory[1]));
    REQUIRE_VK(create_buffer(4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &cb, &buffer));
    CHECK(c.live == 3);
    REQUIRE_VK(vkBindBufferMemory(device, buffer, memory[1], 0));
    vkDestroyBuffer(device, buffer, &cb);
    vkFreeMemory(device, memory[0], &cb);
    vkFreeMemory(device, memory[1], &cb);
    CHECK(c.total == 3);
    CHECK(c.live == 0);
}

/* shared -> private -> shared through the GPU, between buffers bound at
 * non-zero offsets: proves both memory types are resident and addressed as
 * the VkBuffers say. */
static void test_gpu_sees_memory(void)
{
    const VkDeviceSize size = 65536, src_off = 4096, mid_off = 8192, dst_off = 12288;
    VkDeviceMemory src_mem, mid_mem, dst_mem;
    VkBuffer src, mid, dst;
    struct pvk_buffer *b;
    unsigned char *s, *d;
    void *v;
    VkDeviceSize i;
    int same = 1, untouched = 1;

    REQUIRE_VK(allocate(shared_type, src_off + size, NULL, &src_mem));
    REQUIRE_VK(allocate(private_type, mid_off + size, NULL, &mid_mem));
    REQUIRE_VK(allocate(shared_type, dst_off + size + 4096, NULL, &dst_mem));
    REQUIRE_VK(create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, NULL, &src));
    REQUIRE_VK(create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, NULL, &mid));
    REQUIRE_VK(create_buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, NULL, &dst));
    REQUIRE_VK(vkBindBufferMemory(device, src, src_mem, src_off));
    REQUIRE_VK(vkBindBufferMemory(device, mid, mid_mem, mid_off));
    REQUIRE_VK(vkBindBufferMemory(device, dst, dst_mem, dst_off));

    REQUIRE_VK(vkMapMemory(device, src_mem, src_off, size, 0, &v));
    s = v;
    for (i = 0; i < size; i++)
        s[i] = (unsigned char)(i * 13 + 5);
    REQUIRE_VK(vkMapMemory(device, dst_mem, 0, VK_WHOLE_SIZE, 0, &v));
    d = v;
    memset(d, 0xEE, (size_t)(dst_off + size + 4096));

    /* Copy with the memory's MTLBuffer at each VkBuffer's offset. */
    b = (struct pvk_buffer *)src;
    CHECK(gpu_copy(b->memory->mtl_buffer, b->offset, ((struct pvk_buffer *)mid)->memory->mtl_buffer,
                   ((struct pvk_buffer *)mid)->offset, size));
    b = (struct pvk_buffer *)mid;
    CHECK(gpu_copy(b->memory->mtl_buffer, b->offset, ((struct pvk_buffer *)dst)->memory->mtl_buffer,
                   ((struct pvk_buffer *)dst)->offset, size));

    for (i = 0; i < size; i++)
        same &= d[dst_off + i] == (unsigned char)(i * 13 + 5);
    for (i = 0; i < dst_off; i++)
        untouched &= d[i] == 0xEE;
    for (i = dst_off + size; i < dst_off + size + 4096; i++)
        untouched &= d[i] == 0xEE;
    CHECK(same);
    CHECK(untouched);

    vkDestroyBuffer(device, src, NULL);
    vkDestroyBuffer(device, mid, NULL);
    vkDestroyBuffer(device, dst, NULL);
    vkFreeMemory(device, src_mem, NULL);
    vkFreeMemory(device, mid_mem, NULL);
    vkFreeMemory(device, dst_mem, NULL);
}

/* Small allocations are blocks of shared heaps (section 10): each block is
 * aligned to its power-of-two size, none overlap, mapped pointers keep
 * minMemoryMapAlignment, large allocations keep heaps of their own, and
 * empty heaps go but one per type. */
#define SMALL_COUNT 300

static uint32_t chunk_count(uint32_t type)
{
    struct pvk_memory_chunk *c;
    uint32_t n = 0;
    for (c = ((struct pvk_device *)device)->chunks[type]; c; c = c->next)
        n++;
    return n;
}

static VkDeviceSize small_size(unsigned i)
{
    static const VkDeviceSize sizes[] = { 1, 100, 4096, 16384, 16385, 65536, 100000, 262144, 524288 };
    return sizes[i % PVK_ARRAY_SIZE(sizes)];
}

static void fill(VkDeviceMemory memory, VkDeviceSize size, unsigned seed)
{
    unsigned char *p = (unsigned char *)((struct pvk_device_memory *)memory)->contents;
    VkDeviceSize i;
    for (i = 0; i < size; i++)
        p[i] = (unsigned char)(seed * 31 + i * 7);
}

static int filled(VkDeviceMemory memory, VkDeviceSize size, unsigned seed)
{
    const unsigned char *p = (const unsigned char *)((struct pvk_device_memory *)memory)->contents;
    VkDeviceSize i;
    for (i = 0; i < size; i++)
        if (p[i] != (unsigned char)(seed * 31 + i * 7))
            return 0;
    return 1;
}

static void test_small_allocations(void)
{
    VkDeviceMemory memories[SMALL_COUNT], large;
    const struct pvk_device_memory *m, *o;
    int pooled = 1, aligned = 1, apart = 1, intact = 1;
    unsigned i, j;
    void *v;

    REQUIRE(shared_type != UINT32_MAX);
    memset(memories, 0, sizeof(memories));
    for (i = 0; i < SMALL_COUNT; i++) {
        VkDeviceSize block = 16384;
        REQUIRE_VK(allocate(shared_type, small_size(i), NULL, &memories[i]));
        m = (const struct pvk_device_memory *)memories[i];
        while (block < small_size(i))
            block *= 2;
        pooled &= m->chunk != NULL && m->mtl_heap == m->chunk->heap;
        aligned &= m->heap_offset % block == 0 && m->heap_offset + block <= PVK_POOL_CHUNK;
        aligned &= (uintptr_t)m->contents % limits.minMemoryMapAlignment == 0;
        fill(memories[i], small_size(i), i);
    }
    for (i = 0; i < SMALL_COUNT; i++) {
        m = (const struct pvk_device_memory *)memories[i];
        intact &= filled(memories[i], small_size(i), i);
        for (j = i + 1; j < SMALL_COUNT; j++) {
            o = (const struct pvk_device_memory *)memories[j];
            if (o->chunk == m->chunk && o->heap_offset < m->heap_offset + m->size &&
                m->heap_offset < o->heap_offset + o->size)
                apart = 0;
        }
    }
    CHECK(pooled);
    CHECK(aligned);
    CHECK(apart);
    CHECK(intact);
    CHECK(chunk_count(shared_type) > 1);

    /* Freed blocks merge and are handed out again. */
    for (i = 0; i < SMALL_COUNT; i += 2) {
        vkFreeMemory(device, memories[i], NULL);
        memories[i] = VK_NULL_HANDLE;
    }
    for (i = 0; i < SMALL_COUNT; i += 2) {
        REQUIRE_VK(allocate(shared_type, small_size(i + 3), NULL, &memories[i]));
        fill(memories[i], small_size(i + 3), i + 3);
    }
    intact = 1;
    for (i = 0; i < SMALL_COUNT; i++)
        intact &= filled(memories[i], small_size(i % 2 ? i : i + 3), i % 2 ? i : i + 3);
    CHECK(intact);

    REQUIRE_VK(allocate(shared_type, PVK_POOL_LARGEST + 1, NULL, &large));
    m = (const struct pvk_device_memory *)large;
    CHECK(!m->chunk && m->heap_offset == 0 && m->mtl_heap);
    CHECK_VK(vkMapMemory(device, large, 0, VK_WHOLE_SIZE, 0, &v), VK_SUCCESS);
    vkFreeMemory(device, large, NULL);

    for (i = 0; i < SMALL_COUNT; i++)
        vkFreeMemory(device, memories[i], NULL);
    CHECK(chunk_count(shared_type) == 1);
}

/* The GPU copies between blocks of a private and a shared chunk. */
static void test_small_allocations_on_gpu(void)
{
    const VkDeviceSize size = 20000;
    VkDeviceMemory src, mid, dst;
    const struct pvk_device_memory *s, *t, *d;

    REQUIRE_VK(allocate(shared_type, size, NULL, &src));
    REQUIRE_VK(allocate(private_type, size, NULL, &mid));
    REQUIRE_VK(allocate(shared_type, size, NULL, &dst));
    s = (const struct pvk_device_memory *)src;
    t = (const struct pvk_device_memory *)mid;
    d = (const struct pvk_device_memory *)dst;
    CHECK(s->chunk && t->chunk && d->chunk);
    fill(src, size, 9);
    memset(d->contents, 0, (size_t)size);
    CHECK(gpu_copy(s->mtl_buffer, 0, t->mtl_buffer, 0, (unsigned long)size));
    CHECK(gpu_copy(t->mtl_buffer, 0, d->mtl_buffer, 0, (unsigned long)size));
    CHECK(filled(dst, size, 9));
    vkFreeMemory(device, src, NULL);
    vkFreeMemory(device, mid, NULL);
    vkFreeMemory(device, dst, NULL);
}

/* Freed heaps of their own stay resident in the device's cache (section 10,
 * HEAP CACHE): an allocation of their type and about their size takes one,
 * with a buffer of its own size that the GPU sees; the cache keeps to its
 * limits, lets go of heaps freed a while ago and of all of them under
 * memory pressure; heaps it doesn't keep leave the residency set at once. */
static VkDeviceMemory cached_alloc(uint32_t type, VkDeviceSize size, id *heap)
{
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (allocate(type, size, NULL, &memory) == VK_SUCCESS)
        *heap = ((struct pvk_device_memory *)memory)->mtl_heap;
    return memory;
}

static void test_heap_cache(void)
{
    struct pvk_device *d = (struct pvk_device *)device;
    const VkDeviceSize size = (2u << 20) + 4096, fifth = d->heap_cache_limit / 5;
    VkDeviceMemory a, b, c, src, many[PVK_HEAP_CACHE_SLOTS + 4];
    id heap, other, heaps[PVK_HEAP_CACHE_SLOTS + 4];
    const struct pvk_device_memory *m;
    int same = 1;
    unsigned i;

    REQUIRE(d->heap_cache_limit > 0 && d->heap_cache_limit <= PVK_HEAP_CACHE_BYTES && size < d->heap_cache_limit / 4);
    /* The system's memory pressure is simulated below: the real one stops
     * changing the cache first. */
    CHECK(d->memory_source != NULL);
    pvk_memory_unwatch_pressure(d);
    pvk_heap_cache_pressure(d, PVK_DISPATCH_MEMORYPRESSURE_NORMAL);
    pvk_heap_cache_flush(d, true, true);
    CHECK(d->heap_cache_count == 0 && d->heap_cache_bytes == 0);

    /* Taken again by memory of the same type, but not of another one or of
     * another size class. */
    a = cached_alloc(shared_type, size, &heap);
    REQUIRE(a);
    vkFreeMemory(device, a, NULL);
    CHECK(d->heap_cache_count == 1 && d->heap_cache[0].heap == heap && d->heap_cache_bytes >= size);
    b = cached_alloc(private_type, size, &other);
    REQUIRE(b);
    CHECK(other != heap && d->heap_cache_count == 1);
    c = cached_alloc(shared_type, size + size / 4, &other);
    REQUIRE(c);
    CHECK(other != heap && d->heap_cache_count == 1);
    vkFreeMemory(device, c, NULL);
    CHECK(d->heap_cache_count == 2 && d->heap_cache[0].heap == other);
    a = cached_alloc(shared_type, size - 65536, &other); /* smaller, within an eighth */
    REQUIRE(a);
    m = (const struct pvk_device_memory *)a;
    CHECK(other == heap && d->heap_cache_count == 1 && !m->chunk);
    CHECK(MSG0(unsigned long, m->mtl_buffer, "length") == size - 65536);
    CHECK(m->gpu_address == MSG0(uint64_t, m->mtl_buffer, "gpuAddress") && m->contents);

    /* The GPU sees it, resident from before. */
    REQUIRE_VK(allocate(shared_type, 20000, NULL, &src));
    fill(src, 20000, 5);
    memset(m->contents, 0, (size_t)(size - 65536));
    CHECK(gpu_copy(((struct pvk_device_memory *)src)->mtl_buffer, 0, ((struct pvk_device_memory *)b)->mtl_buffer, 0,
                   20000));
    CHECK(gpu_copy(((struct pvk_device_memory *)b)->mtl_buffer, 0, m->mtl_buffer, 8192, 20000));
    for (i = 0; i < 20000; i++)
        same &= ((const unsigned char *)m->contents)[8192 + i] == (unsigned char)(5 * 31 + i * 7);
    CHECK(same);
    vkFreeMemory(device, src, NULL);
    vkFreeMemory(device, a, NULL);
    vkFreeMemory(device, b, NULL);

    /* At most PVK_HEAP_CACHE_SLOTS heaps, newest first... */
    pvk_heap_cache_flush(d, true, true);
    for (i = 0; i < PVK_ARRAY_SIZE(many); i++)
        many[i] = cached_alloc(shared_type, size + i * 65536, &heaps[i]);
    for (i = 0; i < PVK_ARRAY_SIZE(many); i++)
        vkFreeMemory(device, many[i], NULL);
    CHECK(d->heap_cache_count == PVK_HEAP_CACHE_SLOTS && d->heap_cache[0].heap == heaps[PVK_ARRAY_SIZE(many) - 1]);
    /* ...of heap_cache_limit bytes, none over a quarter of it. */
    for (i = 0; i < 6; i++)
        many[i] = cached_alloc(private_type, fifth, &heaps[i]);
    for (i = 0; i < 6; i++)
        vkFreeMemory(device, many[i], NULL);
    CHECK(d->heap_cache_bytes <= d->heap_cache_limit && d->heap_cache[0].heap == heaps[5]);
    a = cached_alloc(private_type, d->heap_cache_limit / 4 + 65536, &heap);
    REQUIRE(a);
    i = d->heap_cache_count;
    vkFreeMemory(device, a, NULL);
    CHECK(d->heap_cache_count == i && d->heap_cache[0].heap != heap);
    CHECK(d->residency_changes == 0); /* it left the residency set at once */

    /* Heaps freed over a second ago leave at the next free. */
    for (i = 0; i < d->heap_cache_count; i++)
        d->heap_cache[i].freed -= 2 * (uint64_t)PVK_HEAP_CACHE_AGE;
    a = cached_alloc(private_type, (1u << 20) + 4096, &heap);
    REQUIRE(a);
    vkFreeMemory(device, a, NULL);
    CHECK(d->heap_cache_count == 1 && d->heap_cache[0].heap == heap);
    /* ...and at the next submission, when frees stop. */
    d->heap_cache[0].freed -= 2 * (uint64_t)PVK_HEAP_CACHE_AGE;
    CHECK(vkQueueSubmit(queue, 0, NULL, VK_NULL_HANDLE) == VK_SUCCESS);
    CHECK(d->heap_cache_count == 0 && d->heap_cache_bytes == 0 && d->residency_changes == 0);

    /* Memory pressure empties the cache, and frees keep nothing until it is
     * normal again. */
    a = cached_alloc(private_type, size, &heap);
    b = cached_alloc(private_type, size, &other);
    REQUIRE(a && b);
    vkFreeMemory(device, a, NULL);
    CHECK(d->heap_cache_count == 1 && d->heap_cache[0].heap == heap);
    pvk_heap_cache_pressure(d, PVK_DISPATCH_MEMORYPRESSURE_WARN);
    CHECK(d->heap_cache_count == 0 && d->heap_cache_bytes == 0 && d->residency_changes == 0);
    vkFreeMemory(device, b, NULL);
    CHECK(d->heap_cache_count == 0 && d->residency_changes == 0);
    pvk_heap_cache_pressure(d, PVK_DISPATCH_MEMORYPRESSURE_NORMAL);
    a = cached_alloc(private_type, size, &heap);
    REQUIRE(a);
    vkFreeMemory(device, a, NULL);
    CHECK(d->heap_cache_count == 1 && d->heap_cache[0].heap == heap);
}

int main(void)
{
    setup();
    if (!device)
        return 1;
    RUN(test_limits);
    RUN(test_allocate_free);
    RUN(test_map);
    RUN(test_map_private_fails);
    RUN(test_buffer_requirements);
    RUN(test_usage_flags2);
    RUN(test_bind_and_addresses);
    RUN(test_oversized_allocation);
    RUN(test_allocation_callbacks);
    RUN(test_gpu_sees_memory);
    RUN(test_small_allocations);
    RUN(test_small_allocations_on_gpu);
    RUN(test_heap_cache);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return TEST_RESULT();
}
