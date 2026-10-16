/* Chapter "Memory Allocation": device memory properties, allocation,
 * mapping, flushing. Host allocation callbacks are tested where objects are
 * created (initialization.c, devsandqueues.c, here and resources.c). */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

#define HOST_COHERENT (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

/* Types a plain allocation may use without enabling features. */
#define SPECIAL_TYPES                                                                                            \
    (VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD |                             \
     VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD)

static VkResult allocate(const struct ct_env *e, uint32_t type, VkDeviceSize size, const void *pnext,
                         const VkAllocationCallbacks *alloc, VkDeviceMemory *memory)
{
    VkMemoryAllocateInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.pNext = pnext;
    info.allocationSize = size;
    info.memoryTypeIndex = type;
    return vkAllocateMemory(e->device, &info, alloc, memory);
}

static int heap_is_tile(const struct ct_env *e, uint32_t type)
{
    return (e->mem.memoryHeaps[e->mem.memoryTypes[type].heapIndex].flags & VK_MEMORY_HEAP_TILE_MEMORY_BIT_QCOM) != 0;
}

/* Can a plain vkAllocateMemory use this type? */
static int plain_type(const struct ct_env *e, uint32_t type)
{
    return !(e->mem.memoryTypes[type].propertyFlags & SPECIAL_TYPES) && !heap_is_tile(e, type);
}

static VkDeviceSize heap_size(const struct ct_env *e, uint32_t type)
{
    return e->mem.memoryHeaps[e->mem.memoryTypes[type].heapIndex].size;
}

/* ==========================================================================
 * Device memory properties
 * ========================================================================== */

CT_TEST(memory_counts,
        "Memory Allocation > Device Memory Properties: memoryTypeCount is at most VK_MAX_MEMORY_TYPES and "
        "memoryHeapCount at most VK_MAX_MEMORY_HEAPS; there is at least one of each",
        "vkGetPhysicalDeviceMemoryProperties")
{
    struct ct_env e;
    ct_env_init(t, &e);
    CT_CHECKF(t, e.mem.memoryTypeCount >= 1 && e.mem.memoryTypeCount <= VK_MAX_MEMORY_TYPES, "memoryTypeCount %u",
              e.mem.memoryTypeCount);
    CT_CHECKF(t, e.mem.memoryHeapCount >= 1 && e.mem.memoryHeapCount <= VK_MAX_MEMORY_HEAPS, "memoryHeapCount %u",
              e.mem.memoryHeapCount);
    ct_env_fini(&e);
}

CT_TEST(memory_heap_device_local,
        "Memory Allocation > Device Memory Properties: at least one heap must include "
        "VK_MEMORY_HEAP_DEVICE_LOCAL_BIT",
        "vkGetPhysicalDeviceMemoryProperties")
{
    struct ct_env e;
    uint32_t i;
    int found = 0;
    ct_env_init(t, &e);
    for (i = 0; i < e.mem.memoryHeapCount && i < VK_MAX_MEMORY_HEAPS; i++)
        found |= (e.mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
    CT_CHECK(t, found);
    ct_env_fini(&e);
}

CT_TEST(memory_heap_size,
        "Memory Allocation > Device Memory Properties: VkMemoryHeap::size is the total memory in the heap, which "
        "allocations consume; every reported heap has a non-zero size",
        "vkGetPhysicalDeviceMemoryProperties")
{
    struct ct_env e;
    uint32_t i;
    ct_env_init(t, &e);
    for (i = 0; i < e.mem.memoryHeapCount && i < VK_MAX_MEMORY_HEAPS; i++)
        CT_CHECKF(t, e.mem.memoryHeaps[i].size > 0, "heap %u has size 0", i);
    ct_env_fini(&e);
}

CT_TEST(memory_type_heap_index,
        "Memory Allocation > VkMemoryType: heapIndex must be less than memoryHeapCount",
        "vkGetPhysicalDeviceMemoryProperties")
{
    struct ct_env e;
    uint32_t i;
    ct_env_init(t, &e);
    for (i = 0; i < e.mem.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; i++)
        CT_CHECKF(t, e.mem.memoryTypes[i].heapIndex < e.mem.memoryHeapCount, "type %u: heapIndex %u of %u heaps", i,
                  e.mem.memoryTypes[i].heapIndex, e.mem.memoryHeapCount);
    ct_env_fini(&e);
}

CT_TEST(memory_type_flags_allowed,
        "Memory Allocation > Device Memory Properties: each memory type's propertyFlags must be one of the listed "
        "combinations ([memory-device-bitmask-list])",
        "vkGetPhysicalDeviceMemoryProperties")
{
    enum {
        DL = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        HC = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        CA = VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        LA = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT,
        PR = VK_MEMORY_PROPERTY_PROTECTED_BIT,
        DC = VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD,
        DU = VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD,
        RD = VK_MEMORY_PROPERTY_RDMA_CAPABLE_BIT_NV
    };
    static const VkMemoryPropertyFlags allowed[] = {
        0,
        HV | HC,
        HV | CA,
        HV | CA | HC,
        DL,
        DL | HV | HC,
        DL | HV | CA,
        DL | HV | CA | HC,
        DL | LA,
        PR,
        PR | DL,
        HV | HC | DC,
        HV | CA | HC | DC,
        DL | DC,
        DL | HV | HC | DC,
        DL | HV | CA | HC | DC,
        HV | HC | DC | DU,
        HV | CA | HC | DC | DU,
        DL | DC | DU,
        DL | HV | HC | DC | DU,
        DL | HV | CA | HC | DC | DU,
        DL | RD,
    };
    struct ct_env e;
    uint32_t i;
    size_t k;
    ct_env_init(t, &e);
    for (i = 0; i < e.mem.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; i++) {
        int ok = 0;
        for (k = 0; k < CT_ARRAY_SIZE(allowed); k++)
            ok |= e.mem.memoryTypes[i].propertyFlags == allowed[k];
        CT_CHECKF(t, ok, "type %u: propertyFlags 0x%x is not an allowed combination", i,
                  e.mem.memoryTypes[i].propertyFlags);
    }
    ct_env_fini(&e);
}

CT_TEST(memory_type_host_coherent,
        "Memory Allocation > Device Memory Properties: there must be at least one memory type with both "
        "VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT and VK_MEMORY_PROPERTY_HOST_COHERENT_BIT",
        "vkGetPhysicalDeviceMemoryProperties")
{
    struct ct_env e;
    ct_env_init(t, &e);
    CT_CHECK(t, ct_find_memory_type(&e, ~0u, HOST_COHERENT, 0) != UINT32_MAX);
    ct_env_fini(&e);
}

CT_TEST(memory_type_device_local,
        "Memory Allocation > Device Memory Properties: there must be at least one memory type with "
        "VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT",
        "vkGetPhysicalDeviceMemoryProperties")
{
    struct ct_env e;
    ct_env_init(t, &e);
    CT_CHECK(t, ct_find_memory_type(&e, ~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0) != UINT32_MAX);
    ct_env_fini(&e);
}

CT_TEST(memory_type_device_local_heap,
        "Memory Allocation > VkMemoryPropertyFlagBits: VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT is set if and only if "
        "the type's heap has VK_MEMORY_HEAP_DEVICE_LOCAL_BIT",
        "vkGetPhysicalDeviceMemoryProperties")
{
    struct ct_env e;
    uint32_t i;
    ct_env_init(t, &e);
    for (i = 0; i < e.mem.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; i++) {
        uint32_t h = e.mem.memoryTypes[i].heapIndex;
        int type_local = (e.mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        int heap_local;
        if (h >= e.mem.memoryHeapCount)
            continue;
        heap_local = (e.mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        CT_CHECKF(t, type_local == heap_local, "type %u: DEVICE_LOCAL %d, heap %u DEVICE_LOCAL %d", i, type_local, h,
                  heap_local);
    }
    ct_env_fini(&e);
}

CT_TEST(memory_type_order,
        "Memory Allocation > Device Memory Properties: a memory type whose propertyFlags are a strict subset of "
        "another's must be at a lower index; device coherent/uncached types come after types without them",
        "vkGetPhysicalDeviceMemoryProperties")
{
    const VkMemoryPropertyFlags amd = VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
    struct ct_env e;
    uint32_t x, y;
    ct_env_init(t, &e);
    for (x = 0; x < e.mem.memoryTypeCount && x < VK_MAX_MEMORY_TYPES; x++)
        for (y = 0; y < x; y++) {
            VkMemoryPropertyFlags fx = e.mem.memoryTypes[x].propertyFlags, fy = e.mem.memoryTypes[y].propertyFlags;
            CT_CHECKF(t, !((fx & fy) == fx && fx != fy), "type %u (0x%x) is a strict subset of type %u (0x%x) but "
                      "comes later", x, fx, y, fy);
            CT_CHECKF(t, !((fy & amd) && !(fx & amd)), "type %u (0x%x) without device coherent/uncached comes after "
                      "type %u (0x%x)", x, fx, y, fy);
        }
    ct_env_fini(&e);
}

CT_TEST(memory_type_exclusive_bits,
        "Memory Allocation > VkMemoryPropertyFlagBits: no type has both LAZILY_ALLOCATED and HOST_VISIBLE, nor "
        "PROTECTED with any host bit",
        "vkGetPhysicalDeviceMemoryProperties")
{
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    struct ct_env e;
    uint32_t i;
    ct_env_init(t, &e);
    for (i = 0; i < e.mem.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; i++) {
        VkMemoryPropertyFlags f = e.mem.memoryTypes[i].propertyFlags;
        CT_CHECKF(t, !((f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) && (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)),
                  "type %u: lazily allocated and host visible", i);
        CT_CHECKF(t, !((f & VK_MEMORY_PROPERTY_PROTECTED_BIT) && (f & host)), "type %u: protected with host bits", i);
    }
    ct_env_fini(&e);
}

CT_TEST(memory_properties2,
        "Memory Allocation > Device Memory Properties: VkPhysicalDeviceMemoryProperties2::memoryProperties is "
        "populated with the same values as vkGetPhysicalDeviceMemoryProperties",
        "vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties2")
{
    struct ct_env e;
    VkPhysicalDeviceMemoryProperties m;
    VkPhysicalDeviceMemoryProperties2 m2;
    uint32_t i;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_1)
        ct_skip(t, "Vulkan 1.1 is not available");
    memset(&m, 0x00, sizeof(m));
    vkGetPhysicalDeviceMemoryProperties(e.pd, &m);
    memset(&m2, 0xFF, sizeof(m2));
    m2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    m2.pNext = NULL;
    vkGetPhysicalDeviceMemoryProperties2(e.pd, &m2);
    CT_REQUIRE(t, m.memoryTypeCount == m2.memoryProperties.memoryTypeCount);
    CT_REQUIRE(t, m.memoryHeapCount == m2.memoryProperties.memoryHeapCount);
    for (i = 0; i < m.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; i++) {
        CT_CHECKF(t, m.memoryTypes[i].propertyFlags == m2.memoryProperties.memoryTypes[i].propertyFlags,
                  "type %u flags", i);
        CT_CHECKF(t, m.memoryTypes[i].heapIndex == m2.memoryProperties.memoryTypes[i].heapIndex, "type %u heap", i);
    }
    for (i = 0; i < m.memoryHeapCount && i < VK_MAX_MEMORY_HEAPS; i++) {
        CT_CHECKF(t, m.memoryHeaps[i].size == m2.memoryProperties.memoryHeaps[i].size, "heap %u size", i);
        CT_CHECKF(t, m.memoryHeaps[i].flags == m2.memoryProperties.memoryHeaps[i].flags, "heap %u flags", i);
    }
    ct_env_fini(&e);
}

CT_TEST(memory_budget,
        "Memory Allocation > Device Memory Properties (VK_EXT_memory_budget): heapBudget and heapUsage are zero for "
        "elements at or above memoryHeapCount; heapBudget is non-zero below it, and at most VkMemoryHeap::size",
        "vkGetPhysicalDeviceMemoryProperties2")
{
    struct ct_env e;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT b;
    VkPhysicalDeviceMemoryProperties2 m2;
    uint32_t i;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_1)
        ct_skip(t, "Vulkan 1.1 is not available");
    if (!ct_device_extension_supported(e.pd, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME))
        ct_skip(t, "VK_EXT_memory_budget is not supported");
    memset(&b, 0xFF, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    b.pNext = NULL;
    memset(&m2, 0, sizeof(m2));
    m2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    m2.pNext = &b;
    vkGetPhysicalDeviceMemoryProperties2(e.pd, &m2);
    for (i = 0; i < VK_MAX_MEMORY_HEAPS; i++) {
        if (i < m2.memoryProperties.memoryHeapCount)
            CT_CHECKF(t, b.heapBudget[i] > 0 && b.heapBudget[i] <= m2.memoryProperties.memoryHeaps[i].size,
                      "heap %u: budget %llu, size %llu", i, (unsigned long long)b.heapBudget[i],
                      (unsigned long long)m2.memoryProperties.memoryHeaps[i].size);
        else
            CT_CHECKF(t, b.heapBudget[i] == 0 && b.heapUsage[i] == 0, "unused heap %u: budget %llu, usage %llu", i,
                      (unsigned long long)b.heapBudget[i], (unsigned long long)b.heapUsage[i]);
    }
    ct_env_fini(&e);
}

/* ==========================================================================
 * Allocation
 * ========================================================================== */

CT_TEST(allocate_every_type,
        "Memory Allocation > Device Memory Allocation: vkAllocateMemory returns a memory object of each memory "
        "type for sizes within the heap (no substitution: the requested parameters are used)",
        "vkAllocateMemory vkFreeMemory")
{
    static const VkDeviceSize sizes[] = { 1, 64, 4096, 65536, (1u << 20) + 3 };
    struct ct_env e;
    uint32_t type;
    size_t s;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    for (type = 0; type < e.mem.memoryTypeCount; type++) {
        if (!plain_type(&e, type))
            continue;
        for (s = 0; s < CT_ARRAY_SIZE(sizes); s++) {
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkResult r;
            if (sizes[s] > heap_size(&e, type))
                continue;
            r = allocate(&e, type, sizes[s], NULL, NULL, &memory);
            CT_CHECKF(t, r == VK_SUCCESS && memory != VK_NULL_HANDLE, "type %u, %llu bytes: %s", type,
                      (unsigned long long)sizes[s], ct_result_name(r));
            vkFreeMemory(e.device, memory, NULL);
        }
    }
    ct_env_fini(&e);
}

CT_TEST(allocate_many,
        "Memory Allocation > Device Memory Allocation: maxMemoryAllocationCount allocations (at least the "
        "required 4096) can exist simultaneously; each is a distinct object",
        "vkAllocateMemory vkFreeMemory")
{
    struct ct_env e;
    VkDeviceMemory *memory;
    uint32_t type, n, i, made = 0;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    type = ct_find_memory_type(&e, ~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, SPECIAL_TYPES);
    if (type == UINT32_MAX)
        type = ct_find_memory_type(&e, ~0u, 0, SPECIAL_TYPES);
    CT_REQUIRE(t, type != UINT32_MAX);
    n = CT_MIN(e.props.limits.maxMemoryAllocationCount, 4096);
    CT_REQUIREF(t, n > 0, "maxMemoryAllocationCount is 0");
    memory = calloc(n, sizeof(*memory));
    CT_REQUIRE(t, memory != NULL);
    for (i = 0; i < n; i++) {
        VkResult r = allocate(&e, type, 256, NULL, NULL, &memory[i]);
        if (r != VK_SUCCESS) {
            ct_fail(t, __FILE__, __LINE__, "allocation %u of %u: %s", i, n, ct_result_name(r));
            break;
        }
        made++;
    }
    for (i = 1; i < made && i < 64; i++)
        CT_CHECK(t, memory[i] != memory[i - 1]);
    for (i = 0; i < made; i++)
        vkFreeMemory(e.device, memory[i], NULL);
    free(memory);
    ct_env_fini(&e);
}

CT_TEST(free_memory_null,
        "Memory Allocation > Device Memory Allocation: vkFreeMemory accepts VK_NULL_HANDLE (implicit valid usage: "
        "\"if memory is not VK_NULL_HANDLE\") and does nothing",
        "vkFreeMemory")
{
    struct ct_env e;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    vkFreeMemory(e.device, VK_NULL_HANDLE, NULL);
    ct_env_fini(&e);
}

CT_TEST(allocate_too_large,
        "Limits > maxMemoryAllocationSize: an allocationSize above maxMemoryAllocationSize (within the heap) "
        "succeeds or fails with VK_ERROR_OUT_OF_DEVICE_MEMORY; failures from such platform limits must return "
        "VK_ERROR_OUT_OF_DEVICE_MEMORY (Device Memory Allocation)",
        "vkAllocateMemory vkFreeMemory vkGetPhysicalDeviceProperties2")
{
    struct ct_env e;
    VkPhysicalDeviceMaintenance3Properties m3;
    VkPhysicalDeviceProperties2 p2;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t type, i;
    VkDeviceSize size;
    VkResult r;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_1)
        ct_skip(t, "maxMemoryAllocationSize is Vulkan 1.1");
    memset(&m3, 0, sizeof(m3));
    m3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
    memset(&p2, 0, sizeof(p2));
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &m3;
    vkGetPhysicalDeviceProperties2(e.pd, &p2);
    CT_REQUIRE(t, m3.maxMemoryAllocationSize > 0);
    size = m3.maxMemoryAllocationSize + 1;
    type = UINT32_MAX;
    for (i = 0; i < e.mem.memoryTypeCount && type == UINT32_MAX; i++)
        if (plain_type(&e, i) && !(e.mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) &&
            heap_size(&e, i) >= size)
            type = i;
    if (type == UINT32_MAX)
        ct_skip(t, "no heap is larger than maxMemoryAllocationSize (%llu)",
                (unsigned long long)m3.maxMemoryAllocationSize);
    ct_env_device(t, &e, NULL);
    r = allocate(&e, type, size, NULL, NULL, &memory);
    CT_CHECKF(t, r == VK_SUCCESS || r == VK_ERROR_OUT_OF_DEVICE_MEMORY, "%llu bytes of type %u: %s",
              (unsigned long long)size, type, ct_result_name(r));
    if (r == VK_SUCCESS)
        vkFreeMemory(e.device, memory, NULL);
    /* The device still works. */
    memory = VK_NULL_HANDLE;
    CT_CHECK_VK(t, allocate(&e, type, 4096, NULL, NULL, &memory), VK_SUCCESS);
    vkFreeMemory(e.device, memory, NULL);
    ct_env_fini(&e);
}

CT_TEST(memory_allocation_callbacks,
        "Memory Allocation > Host Memory: host memory a VkDeviceMemory needs comes from the allocator given to "
        "vkAllocateMemory (else the device's), following the callback rules, and vkFreeMemory releases it",
        "vkAllocateMemory vkFreeMemory")
{
    struct ct_env e;
    struct ct_allocator a;
    VkDeviceMemory memory[4];
    uint32_t type, i;
    VkResult r;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    type = ct_find_memory_type(&e, ~0u, HOST_COHERENT, 0);
    CT_REQUIRE(t, type != UINT32_MAX);
    ct_allocator_init(&a, -1);
    for (i = 0; i < 4; i++) {
        CT_CALL_VK(&a, r, allocate(&e, type, 4096u << i, NULL, &a.cb, &memory[i]));
        CT_REQUIRE_VK(t, r);
    }
    for (i = 0; i < 4; i++)
        CT_CALL(&a, vkFreeMemory(e.device, memory[i], &a.cb));
    CT_CHECKF(t, a.live == 0, "%ld allocations still live after vkFreeMemory", a.live);
    ct_allocator_check(t, &a, "memory");
    ct_allocator_fini(&a);
    ct_env_fini(&e);
}

CT_TEST(memory_allocation_failure,
        "Memory Allocation > Host Memory: when pfnAllocation returns NULL, vkAllocateMemory either continues or "
        "returns VK_ERROR_OUT_OF_HOST_MEMORY, frees what it allocated, and leaves the device usable",
        "vkAllocateMemory vkFreeMemory")
{
    struct ct_env e;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t type;
    long n;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    type = ct_find_memory_type(&e, ~0u, HOST_COHERENT, 0);
    CT_REQUIRE(t, type != UINT32_MAX);
    for (n = 0; n < 200; n++) {
        struct ct_allocator a;
        VkResult r;
        int failed;
        ct_allocator_init(&a, n);
        memory = VK_NULL_HANDLE;
        CT_CALL_VK(&a, r, allocate(&e, type, 65536, NULL, &a.cb, &memory));
        CT_CHECKF(t, r == VK_SUCCESS || r == VK_ERROR_OUT_OF_HOST_MEMORY, "allocation %ld failing: %s", n,
                  ct_result_name(r));
        if (r == VK_SUCCESS)
            CT_CALL(&a, vkFreeMemory(e.device, memory, &a.cb));
        CT_CHECKF(t, a.live == 0, "allocation %ld failing: %ld host allocations leaked", n, a.live);
        ct_allocator_check(t, &a, "memory");
        failed = a.failed > 0;
        ct_allocator_fini(&a);
        if (t->failures || !failed)
            break;
    }
    memory = VK_NULL_HANDLE;
    CT_CHECK_VK(t, allocate(&e, type, 65536, NULL, NULL, &memory), VK_SUCCESS);
    vkFreeMemory(e.device, memory, NULL);
    ct_env_fini(&e);
}

CT_TEST(allocate_dedicated,
        "Memory Allocation > VkMemoryDedicatedAllocateInfo: memory dedicated to a buffer (allocationSize equal to "
        "its VkMemoryRequirements::size) can be allocated and bound at offset 0",
        "vkAllocateMemory vkFreeMemory vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements "
        "vkBindBufferMemory")
{
    struct ct_env e;
    VkBufferCreateInfo info;
    VkMemoryDedicatedAllocateInfo dedicated;
    VkMemoryRequirements req;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t type;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_1)
        ct_skip(t, "dedicated allocations are Vulkan 1.1");
    ct_env_device(t, &e, NULL);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = 100000;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    CT_REQUIRE_VK(t, vkCreateBuffer(e.device, &info, NULL, &buffer));
    vkGetBufferMemoryRequirements(e.device, buffer, &req);
    type = ct_find_memory_type(&e, req.memoryTypeBits, 0, SPECIAL_TYPES);
    CT_REQUIRE(t, type != UINT32_MAX);
    memset(&dedicated, 0, sizeof(dedicated));
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.buffer = buffer;
    CT_REQUIRE_VK(t, allocate(&e, type, req.size, &dedicated, NULL, &memory));
    CT_CHECK_VK(t, vkBindBufferMemory(e.device, buffer, memory, 0), VK_SUCCESS);
    vkDestroyBuffer(e.device, buffer, NULL);
    vkFreeMemory(e.device, memory, NULL);
    ct_env_fini(&e);
}

CT_TEST(memory_commitment,
        "Memory Allocation > Lazily Allocated Memory: vkGetDeviceMemoryCommitment on memory of a "
        "LAZILY_ALLOCATED type returns a committed size between zero and the allocation size",
        "vkGetDeviceMemoryCommitment vkAllocateMemory vkFreeMemory")
{
    struct ct_env e;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize committed = ~(VkDeviceSize)0;
    uint32_t type;

    ct_env_init(t, &e);
    type = ct_find_memory_type(&e, ~0u, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, SPECIAL_TYPES);
    if (type == UINT32_MAX)
        ct_skip(t, "no LAZILY_ALLOCATED memory type (vkGetDeviceMemoryCommitment is only valid on one)");
    ct_env_device(t, &e, NULL);
    CT_REQUIRE_VK(t, allocate(&e, type, 1u << 20, NULL, NULL, &memory));
    vkGetDeviceMemoryCommitment(e.device, memory, &committed);
    CT_CHECKF(t, committed <= (1u << 20), "committed %llu of %u bytes", (unsigned long long)committed, 1u << 20);
    vkFreeMemory(e.device, memory, NULL);
    ct_env_fini(&e);
}

/* ==========================================================================
 * Host access
 * ========================================================================== */

static VkDeviceSize align_up(VkDeviceSize x, VkDeviceSize a)
{
    return (x + a - 1) / a * a;
}

/* Maps [offset, offset + size) and checks the pointer rule. */
static unsigned char *map(struct ct *t, const struct ct_env *e, VkDeviceMemory memory, VkDeviceSize offset,
                          VkDeviceSize size)
{
    void *p = NULL;
    VkResult r = vkMapMemory(e->device, memory, offset, size, 0, &p);
    CT_REQUIREF(t, r == VK_SUCCESS && p != NULL, "vkMapMemory(offset %llu, size %llu): %s, pointer %p",
                (unsigned long long)offset, (unsigned long long)size, ct_result_name(r), p);
    CT_REQUIREF(t, e->props.limits.minMemoryMapAlignment > 0, "minMemoryMapAlignment is 0");
    CT_CHECKF(t, ((uintptr_t)p - (uintptr_t)offset) % e->props.limits.minMemoryMapAlignment == 0,
              "vkMapMemory(offset %llu) returned %p: pointer - offset is not a multiple of minMemoryMapAlignment %zu",
              (unsigned long long)offset, p, e->props.limits.minMemoryMapAlignment);
    return p;
}

static VkMappedMemoryRange range(VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size)
{
    VkMappedMemoryRange r;
    memset(&r, 0, sizeof(r));
    r.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    r.memory = memory;
    r.offset = offset;
    r.size = size;
    return r;
}

CT_TEST(map_offsets,
        "Memory Allocation > Host Access: vkMapMemory returns a pointer to the mapped range whose value minus "
        "offset is aligned to minMemoryMapAlignment, at any offset and for VK_WHOLE_SIZE; every host-visible type; "
        "host writes are seen through later mappings",
        "vkMapMemory vkUnmapMemory vkAllocateMemory vkFreeMemory vkFlushMappedMemoryRanges "
        "vkInvalidateMappedMemoryRanges")
{
    struct ct_env e;
    const VkDeviceSize size = 65536 + 3;
    static const VkDeviceSize offsets[] = { 0, 1, 7, 64, 4095, 4096, 65536, 65536 + 2 };
    uint32_t type, tested = 0;

    ct_env_init(t, &e);
    CT_REQUIREF(t, ct_is_pow2(e.props.limits.minMemoryMapAlignment), "minMemoryMapAlignment %zu",
                e.props.limits.minMemoryMapAlignment);
    CT_REQUIREF(t, ct_is_pow2(e.props.limits.nonCoherentAtomSize), "nonCoherentAtomSize %llu",
                (unsigned long long)e.props.limits.nonCoherentAtomSize);
    ct_env_device(t, &e, NULL);
    for (type = 0; type < e.mem.memoryTypeCount; type++) {
        VkMemoryPropertyFlags f = e.mem.memoryTypes[type].propertyFlags;
        int coherent = (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkMappedMemoryRange whole = range(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
        unsigned char *p;
        VkDeviceSize i;
        size_t k;
        int same = 1;

        if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) || !plain_type(&e, type))
            continue;
        tested++;
        CT_REQUIRE_VK(t, allocate(&e, type, size, NULL, NULL, &memory));
        whole.memory = memory;
        p = map(t, &e, memory, 0, VK_WHOLE_SIZE);
        for (i = 0; i < size; i++)
            p[i] = (unsigned char)(i * 7 + type);
        if (!coherent)
            CT_CHECK_VK(t, vkFlushMappedMemoryRanges(e.device, 1, &whole), VK_SUCCESS);
        vkUnmapMemory(e.device, memory);

        for (k = 0; k < CT_ARRAY_SIZE(offsets); k++) {
            VkDeviceSize off = offsets[k];
            VkDeviceSize len = size - off;
            unsigned char *q = map(t, &e, memory, off, k % 2 ? VK_WHOLE_SIZE : len);
            if (!coherent) {
                VkDeviceSize start = align_up(off, e.props.limits.nonCoherentAtomSize);
                if (start < size) {
                    VkMappedMemoryRange rg = range(memory, start, VK_WHOLE_SIZE);
                    CT_CHECK_VK(t, vkInvalidateMappedMemoryRanges(e.device, 1, &rg), VK_SUCCESS);
                }
            }
            for (i = 0; i < len; i++)
                same &= q[i] == (unsigned char)((i + off) * 7 + type);
            CT_CHECKF(t, same, "type %u: mapping at offset %llu does not show the bytes written at that offset", type,
                      (unsigned long long)off);
            vkUnmapMemory(e.device, memory);
        }
        vkFreeMemory(e.device, memory, NULL);
    }
    CT_CHECK(t, tested > 0);
    ct_env_fini(&e);
}

CT_TEST(map_memory2,
        "Memory Allocation > Host Access: vkMapMemory2 behaves as vkMapMemory (pointer minus offset aligned to "
        "minMemoryMapAlignment, same bytes) and vkUnmapMemory2 unmaps with VK_SUCCESS",
        "vkMapMemory2 vkUnmapMemory2 vkMapMemory vkUnmapMemory vkAllocateMemory vkFreeMemory")
{
    struct ct_env e;
    VkMemoryMapInfo info;
    VkMemoryUnmapInfo unmap;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    unsigned char *p;
    void *q = NULL;
    uint32_t type, i;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_4)
        ct_skip(t, "vkMapMemory2 is Vulkan 1.4");
    ct_env_device(t, &e, NULL);
    type = ct_find_memory_type(&e, ~0u, HOST_COHERENT, SPECIAL_TYPES);
    CT_REQUIRE(t, type != UINT32_MAX);
    CT_REQUIRE_VK(t, allocate(&e, type, 8192, NULL, NULL, &memory));
    p = map(t, &e, memory, 0, VK_WHOLE_SIZE);
    for (i = 0; i < 8192; i++)
        p[i] = (unsigned char)(i ^ 0x5A);
    vkUnmapMemory(e.device, memory);

    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    info.memory = memory;
    info.offset = 3001;
    info.size = VK_WHOLE_SIZE;
    CT_REQUIRE_VK(t, vkMapMemory2(e.device, &info, &q));
    CT_REQUIRE(t, q != NULL);
    CT_CHECKF(t, ((uintptr_t)q - 3001) % e.props.limits.minMemoryMapAlignment == 0,
              "pointer %p minus offset 3001 is not aligned to minMemoryMapAlignment", q);
    CT_CHECK(t, ((unsigned char *)q)[0] == (unsigned char)(3001 ^ 0x5A));
    CT_CHECK(t, ((unsigned char *)q)[8191 - 3001] == (unsigned char)(8191 ^ 0x5A));
    memset(&unmap, 0, sizeof(unmap));
    unmap.sType = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO;
    unmap.memory = memory;
    CT_CHECK_VK(t, vkUnmapMemory2(e.device, &unmap), VK_SUCCESS);
    /* Unmapped: it can be mapped again. */
    info.offset = 0;
    info.size = 16;
    CT_CHECK_VK(t, vkMapMemory2(e.device, &info, &q), VK_SUCCESS);
    CT_CHECK_VK(t, vkUnmapMemory2(e.device, &unmap), VK_SUCCESS);
    vkFreeMemory(e.device, memory, NULL);
    ct_env_fini(&e);
}

CT_TEST(flush_invalidate,
        "Memory Allocation > Host Access: vkFlushMappedMemoryRanges and vkInvalidateMappedMemoryRanges accept "
        "ranges within the current mapping at multiples of nonCoherentAtomSize, several at once and VK_WHOLE_SIZE, "
        "and return VK_SUCCESS",
        "vkFlushMappedMemoryRanges vkInvalidateMappedMemoryRanges vkMapMemory vkUnmapMemory vkAllocateMemory "
        "vkFreeMemory")
{
    struct ct_env e;
    uint32_t type, tested = 0;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    for (type = 0; type < e.mem.memoryTypeCount; type++) {
        VkDeviceSize atom = e.props.limits.nonCoherentAtomSize, size;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkMappedMemoryRange r[3];
        unsigned char *p;

        if (!(e.mem.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) || !plain_type(&e, type))
            continue;
        CT_REQUIREF(t, ct_is_pow2(atom), "nonCoherentAtomSize %llu", (unsigned long long)atom);
        tested++;
        size = atom * 8 + 5; /* ends off the atom grid: the last range runs to the end */
        CT_REQUIRE_VK(t, allocate(&e, type, size, NULL, NULL, &memory));
        p = map(t, &e, memory, 0, VK_WHOLE_SIZE);
        memset(p, 0x3C, (size_t)size);
        r[0] = range(memory, 0, VK_WHOLE_SIZE);
        CT_CHECK_VK(t, vkFlushMappedMemoryRanges(e.device, 1, r), VK_SUCCESS);
        CT_CHECK_VK(t, vkInvalidateMappedMemoryRanges(e.device, 1, r), VK_SUCCESS);
        r[0] = range(memory, 0, atom);
        r[1] = range(memory, atom * 2, atom * 3);
        r[2] = range(memory, atom * 6, size - atom * 6);
        CT_CHECK_VK(t, vkFlushMappedMemoryRanges(e.device, 3, r), VK_SUCCESS);
        CT_CHECK_VK(t, vkInvalidateMappedMemoryRanges(e.device, 3, r), VK_SUCCESS);
        r[0] = range(memory, atom * 4, VK_WHOLE_SIZE);
        CT_CHECK_VK(t, vkFlushMappedMemoryRanges(e.device, 1, r), VK_SUCCESS);
        CT_CHECK_VK(t, vkInvalidateMappedMemoryRanges(e.device, 1, r), VK_SUCCESS);
        CT_CHECK(t, p[0] == 0x3C && p[size - 1] == 0x3C);
        vkUnmapMemory(e.device, memory);

        /* A mapping that starts mid-allocation, flushed with VK_WHOLE_SIZE. */
        p = map(t, &e, memory, atom * 2, VK_WHOLE_SIZE);
        r[0] = range(memory, atom * 2, VK_WHOLE_SIZE);
        CT_CHECK_VK(t, vkFlushMappedMemoryRanges(e.device, 1, r), VK_SUCCESS);
        CT_CHECK_VK(t, vkInvalidateMappedMemoryRanges(e.device, 1, r), VK_SUCCESS);
        vkUnmapMemory(e.device, memory);
        vkFreeMemory(e.device, memory, NULL);
    }
    CT_CHECK(t, tested > 0);
    ct_env_fini(&e);
}

CT_TEST(map_repeatedly,
        "Memory Allocation > Host Access: memory can be mapped and unmapped repeatedly; each vkMapMemory succeeds "
        "with an aligned pointer",
        "vkMapMemory vkUnmapMemory vkAllocateMemory vkFreeMemory")
{
    struct ct_env e;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t type, i;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    type = ct_find_memory_type(&e, ~0u, HOST_COHERENT, SPECIAL_TYPES);
    CT_REQUIRE(t, type != UINT32_MAX);
    CT_REQUIRE_VK(t, allocate(&e, type, 1u << 20, NULL, NULL, &memory));
    for (i = 0; i < 500 && !t->failures; i++) {
        unsigned char *p = map(t, &e, memory, (i * 4099u) % (1u << 20), VK_WHOLE_SIZE);
        p[0] = (unsigned char)i;
        vkUnmapMemory(e.device, memory);
    }
    vkFreeMemory(e.device, memory, NULL);
    ct_env_fini(&e);
}

CT_TEST(free_while_mapped,
        "Memory Allocation > Device Memory Allocation: if a memory object is mapped when freed it is implicitly "
        "unmapped; allocation and mapping keep working afterwards",
        "vkFreeMemory vkMapMemory vkUnmapMemory vkAllocateMemory")
{
    struct ct_env e;
    uint32_t type, i;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    type = ct_find_memory_type(&e, ~0u, HOST_COHERENT, SPECIAL_TYPES);
    CT_REQUIRE(t, type != UINT32_MAX);
    for (i = 0; i < 8; i++) {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        unsigned char *p;
        CT_REQUIRE_VK(t, allocate(&e, type, 65536, NULL, NULL, &memory));
        p = map(t, &e, memory, 0, VK_WHOLE_SIZE);
        memset(p, (int)i, 65536);
        vkFreeMemory(e.device, memory, NULL);
    }
    ct_env_fini(&e);
}

/* ==========================================================================
 * Host and device access together (needs command buffers)
 * ========================================================================== */

#define GPU_COMMANDS                                                                                             \
    "vkCreateCommandPool vkDestroyCommandPool vkAllocateCommandBuffers vkFreeCommandBuffers "                    \
    "vkBeginCommandBuffer vkEndCommandBuffer vkCreateFence vkDestroyFence vkResetFences vkQueueSubmit "         \
    "vkWaitForFences vkCmdPipelineBarrier vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements "        \
    "vkBindBufferMemory vkAllocateMemory vkFreeMemory vkMapMemory vkUnmapMemory"

/* A buffer of `size` bytes bound at offset 0 of fresh memory of `type`. */
static VkBuffer buffer_in(struct ct *t, const struct ct_env *e, uint32_t type, VkDeviceSize size,
                          VkBufferUsageFlags usage, VkDeviceMemory *memory)
{
    VkBufferCreateInfo info;
    VkMemoryRequirements req;
    VkBuffer buffer = VK_NULL_HANDLE;

    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    CT_REQUIRE_VK(t, vkCreateBuffer(e->device, &info, NULL, &buffer));
    vkGetBufferMemoryRequirements(e->device, buffer, &req);
    CT_REQUIREF(t, req.memoryTypeBits & (1u << type), "memory type %u cannot hold a transfer buffer", type);
    CT_REQUIRE_VK(t, allocate(e, type, req.size, NULL, NULL, memory));
    CT_REQUIRE_VK(t, vkBindBufferMemory(e->device, buffer, *memory, 0));
    return buffer;
}

static uint32_t transfer_types(const struct ct_env *e)
{
    VkBufferCreateInfo info;
    VkMemoryRequirements req;
    VkBuffer buffer = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = 65536;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(e->device, &info, NULL, &buffer) != VK_SUCCESS)
        return 0;
    vkGetBufferMemoryRequirements(e->device, buffer, &req);
    vkDestroyBuffer(e->device, buffer, NULL);
    return req.memoryTypeBits;
}

CT_TEST(coherent_device_writes,
        "Memory Allocation > VkMemoryPropertyFlagBits: HOST_COHERENT memory needs no "
        "vkInvalidateMappedMemoryRanges: device writes made available to the host by a barrier "
        "(dstAccessMask HOST_READ) and waited for with a fence are seen by host reads of the mapping",
        "vkCmdFillBuffer " GPU_COMMANDS)
{
    struct ct_env e;
    uint32_t bits, type, tested = 0;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    bits = transfer_types(&e);
    for (type = 0; type < e.mem.memoryTypeCount; type++) {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkCommandBuffer cb;
        VkBuffer buffer;
        unsigned char *p;
        void *v = NULL;
        uint32_t i;
        int ok = 1;

        if (!(bits & (1u << type)) || (e.mem.memoryTypes[type].propertyFlags & HOST_COHERENT) != HOST_COHERENT ||
            !plain_type(&e, type))
            continue;
        tested++;
        buffer = buffer_in(t, &e, type, 65536, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &memory);
        CT_REQUIRE_VK(t, vkMapMemory(e.device, memory, 0, VK_WHOLE_SIZE, 0, &v));
        p = v;
        memset(p, 0, 65536);
        cb = ct_begin_commands(t, &e);
        vkCmdFillBuffer(cb, buffer, 0, VK_WHOLE_SIZE, 0xA5A5A5A5u);
        ct_transfer_barrier(cb);
        vkCmdFillBuffer(cb, buffer, 4096, 8192, 0x5A5A5A5Au);
        ct_host_read_barrier(cb);
        ct_submit_and_wait(t, &e, cb);
        for (i = 0; i < 65536; i++)
            ok &= p[i] == (i >= 4096 && i < 12288 ? 0x5A : 0xA5);
        CT_CHECKF(t, ok, "type %u: the host does not see the device's writes", type);
        vkUnmapMemory(e.device, memory);
        vkDestroyBuffer(e.device, buffer, NULL);
        vkFreeMemory(e.device, memory, NULL);
    }
    CT_CHECK(t, tested > 0);
    ct_env_fini(&e);
}

CT_TEST(coherent_host_writes,
        "Memory Allocation > VkMemoryPropertyFlagBits: HOST_COHERENT memory needs no vkFlushMappedMemoryRanges: "
        "host writes before vkQueueSubmit are seen by the device (here copied through device-local memory and "
        "back)",
        "vkCmdCopyBuffer " GPU_COMMANDS)
{
    struct ct_env e;
    VkDeviceMemory src_mem = VK_NULL_HANDLE, mid_mem = VK_NULL_HANDLE, dst_mem = VK_NULL_HANDLE;
    VkBuffer src, mid, dst;
    VkBufferCopy region;
    VkCommandBuffer cb;
    uint32_t bits, host, local, i;
    unsigned char *s, *d;
    void *v = NULL;
    int ok = 1;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    bits = transfer_types(&e);
    host = ct_find_memory_type(&e, bits, HOST_COHERENT, SPECIAL_TYPES);
    local = ct_find_memory_type(&e, bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                SPECIAL_TYPES | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (local == UINT32_MAX)
        local = ct_find_memory_type(&e, bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, SPECIAL_TYPES);
    CT_REQUIRE(t, host != UINT32_MAX && local != UINT32_MAX);
    src = buffer_in(t, &e, host, 65536, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src_mem);
    mid = buffer_in(t, &e, local, 65536, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mid_mem);
    dst = buffer_in(t, &e, host, 65536, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst_mem);
    CT_REQUIRE_VK(t, vkMapMemory(e.device, src_mem, 0, VK_WHOLE_SIZE, 0, &v));
    s = v;
    for (i = 0; i < 65536; i++)
        s[i] = (unsigned char)(i * 13 + 7);
    CT_REQUIRE_VK(t, vkMapMemory(e.device, dst_mem, 0, VK_WHOLE_SIZE, 0, &v));
    d = v;
    memset(d, 0, 65536);

    memset(&region, 0, sizeof(region));
    region.size = 65536;
    cb = ct_begin_commands(t, &e);
    vkCmdCopyBuffer(cb, src, mid, 1, &region);
    ct_transfer_barrier(cb);
    vkCmdCopyBuffer(cb, mid, dst, 1, &region);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &e, cb);
    for (i = 0; i < 65536; i++)
        ok &= d[i] == (unsigned char)(i * 13 + 7);
    CT_CHECKF(t, ok, "bytes written by the host (type %u) did not come back through type %u", host, local);
    vkDestroyBuffer(e.device, src, NULL);
    vkDestroyBuffer(e.device, mid, NULL);
    vkDestroyBuffer(e.device, dst, NULL);
    vkFreeMemory(e.device, src_mem, NULL);
    vkFreeMemory(e.device, mid_mem, NULL);
    vkFreeMemory(e.device, dst_mem, NULL);
    ct_env_fini(&e);
}

CT_TEST(noncoherent_flush_invalidate_gpu,
        "Memory Allocation > Host Access: for memory without HOST_COHERENT, vkFlushMappedMemoryRanges makes host "
        "writes available to the device and vkInvalidateMappedMemoryRanges makes device writes visible to the "
        "host",
        "vkCmdCopyBuffer vkFlushMappedMemoryRanges vkInvalidateMappedMemoryRanges " GPU_COMMANDS)
{
    struct ct_env e;
    uint32_t bits, type, tested = 0;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    bits = transfer_types(&e);
    for (type = 0; type < e.mem.memoryTypeCount; type++) {
        VkMemoryPropertyFlags f = e.mem.memoryTypes[type].propertyFlags;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkMappedMemoryRange whole = range(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
        VkBufferCopy region;
        VkCommandBuffer cb;
        VkBuffer buffer;
        unsigned char *p;
        void *v = NULL;
        uint32_t i;
        int ok = 1;

        if (!(bits & (1u << type)) || !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ||
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) || !plain_type(&e, type))
            continue;
        tested++;
        buffer = buffer_in(t, &e, type, 65536, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           &memory);
        whole.memory = memory;
        CT_REQUIRE_VK(t, vkMapMemory(e.device, memory, 0, VK_WHOLE_SIZE, 0, &v));
        p = v;
        for (i = 0; i < 32768; i++)
            p[i] = (unsigned char)(i * 3 + 1);
        CT_REQUIRE_VK(t, vkFlushMappedMemoryRanges(e.device, 1, &whole));
        memset(&region, 0, sizeof(region));
        region.dstOffset = 32768;
        region.size = 32768;
        cb = ct_begin_commands(t, &e);
        vkCmdCopyBuffer(cb, buffer, buffer, 1, &region);
        ct_host_read_barrier(cb);
        ct_submit_and_wait(t, &e, cb);
        CT_REQUIRE_VK(t, vkInvalidateMappedMemoryRanges(e.device, 1, &whole));
        for (i = 0; i < 32768; i++)
            ok &= p[32768 + i] == (unsigned char)(i * 3 + 1);
        CT_CHECKF(t, ok, "type %u: flushed host writes, copied by the device, are not seen after invalidation", type);
        vkUnmapMemory(e.device, memory);
        vkDestroyBuffer(e.device, buffer, NULL);
        vkFreeMemory(e.device, memory, NULL);
    }
    if (!tested)
        ct_skip(t, "every host-visible memory type is coherent");
    ct_env_fini(&e);
}

/* ==========================================================================
 * Peer memory
 * ========================================================================== */

CT_TEST(peer_memory_features,
        "Memory Allocation > Peer Memory Features (vkGetDeviceGroupPeerMemoryFeatures, Vulkan 1.1): on a device "
        "made of a group of two or more physical devices, for every pair of distinct device indices "
        "(VUID-vkGetDeviceGroupPeerMemoryFeatures-localDeviceIndex-00694), VK_PEER_MEMORY_FEATURE_COPY_DST_BIT is "
        "supported for every heap that is not device-local and for at least one device-local heap",
        "vkGetDeviceGroupPeerMemoryFeatures vkEnumeratePhysicalDeviceGroups vkGetPhysicalDeviceMemoryProperties "
        "vkCreateDevice vkDestroyDevice")
{
    struct ct_env e;
    VkPhysicalDeviceGroupProperties groups[16];
    VkPhysicalDeviceMemoryProperties mem;
    VkDeviceGroupDeviceCreateInfo group_info;
    VkDeviceQueueCreateInfo queue;
    VkDeviceCreateInfo info;
    const char *names[2];
    float priority = 1.0f;
    uint32_t count = 16, g, i, l, r, h;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_1)
        ct_skip(t, "device groups are Vulkan 1.1");
    for (i = 0; i < 16; i++) {
        memset(&groups[i], 0, sizeof(groups[i]));
        groups[i].sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES;
    }
    CT_REQUIRE(t, vkEnumeratePhysicalDeviceGroups(e.instance, &count, groups) >= 0);
    for (g = 0; g < count && groups[g].physicalDeviceCount < 2; g++)
        ;
    if (g == count)
        ct_skip(t, "every device group has one physical device: no two distinct device indices");
    vkGetPhysicalDeviceMemoryProperties(groups[g].physicalDevices[0], &mem);
    memset(&group_info, 0, sizeof(group_info));
    group_info.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO;
    group_info.physicalDeviceCount = groups[g].physicalDeviceCount;
    group_info.pPhysicalDevices = groups[g].physicalDevices;
    memset(&queue, 0, sizeof(queue));
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    e.pd = groups[g].physicalDevices[0];
    e.portability_subset = ct_device_extension_supported(e.pd, "VK_KHR_portability_subset");
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.pNext = &group_info;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue;
    info.enabledExtensionCount = ct_add_portability_subset(&e, names, 0);
    info.ppEnabledExtensionNames = info.enabledExtensionCount ? names : NULL;
    CT_REQUIRE_VK(t, vkCreateDevice(e.pd, &info, NULL, &e.device));
    for (l = 0; l < groups[g].physicalDeviceCount; l++)
        for (r = 0; r < groups[g].physicalDeviceCount; r++) {
            int device_local_dst = 0;
            if (l == r)
                continue;
            for (h = 0; h < mem.memoryHeapCount; h++) {
                VkPeerMemoryFeatureFlags f = 0;
                vkGetDeviceGroupPeerMemoryFeatures(e.device, h, l, r, &f);
                if (mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    device_local_dst |= (f & VK_PEER_MEMORY_FEATURE_COPY_DST_BIT) != 0;
                else
                    CT_CHECKF(t, f & VK_PEER_MEMORY_FEATURE_COPY_DST_BIT,
                              "heap %u (host local), device %u accessing %u's memory: features 0x%x", h, l, r, f);
            }
            CT_CHECKF(t, device_local_dst, "device %u accessing %u's memory: no device-local heap with COPY_DST", l, r);
        }
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(memory_counts),
    CT_REF(memory_heap_device_local),
    CT_REF(memory_heap_size),
    CT_REF(memory_type_heap_index),
    CT_REF(memory_type_flags_allowed),
    CT_REF(memory_type_host_coherent),
    CT_REF(memory_type_device_local),
    CT_REF(memory_type_device_local_heap),
    CT_REF(memory_type_order),
    CT_REF(memory_type_exclusive_bits),
    CT_REF(memory_properties2),
    CT_REF(memory_budget),
    CT_REF(allocate_every_type),
    CT_REF(allocate_many),
    CT_REF(free_memory_null),
    CT_REF(allocate_too_large),
    CT_REF(memory_allocation_callbacks),
    CT_REF(memory_allocation_failure),
    CT_REF(allocate_dedicated),
    CT_REF(memory_commitment),
    CT_REF(map_offsets),
    CT_REF(map_memory2),
    CT_REF(flush_invalidate),
    CT_REF(map_repeatedly),
    CT_REF(free_while_mapped),
    CT_REF(coherent_device_writes),
    CT_REF(coherent_host_writes),
    CT_REF(noncoherent_flush_invalidate_gpu),
    CT_REF(peer_memory_features),
};

CT_SUITE(memory, tests)
