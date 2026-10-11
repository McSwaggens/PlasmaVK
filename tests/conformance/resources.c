/* Chapter "Resource Creation": buffers, their memory requirements, memory
 * binding and buffer device addresses. */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

#define HOST_COHERENT (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
#define SPECIAL_TYPES                                                                                            \
    (VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD |                             \
     VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)

/* Vulkan 1.0 usages that need no feature. */
static const VkBufferUsageFlags usages[] = {
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
    VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
        VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
};

static const VkDeviceSize sizes[] = { 1, 3, 16, 255, 4097, 65536, (1u << 24) + 1 };

static VkBufferCreateInfo buffer_info(VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    return info;
}

static VkBuffer make_buffer(struct ct *t, const struct ct_env *e, VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo info = buffer_info(size, usage);
    VkBuffer buffer = VK_NULL_HANDLE;
    VkResult r = vkCreateBuffer(e->device, &info, NULL, &buffer);
    CT_REQUIREF(t, r == VK_SUCCESS && buffer != VK_NULL_HANDLE, "vkCreateBuffer(size %llu, usage 0x%x): %s",
                (unsigned long long)size, usage, ct_result_name(r));
    return buffer;
}

static VkMemoryRequirements requirements(const struct ct_env *e, VkBuffer buffer)
{
    VkMemoryRequirements req;
    memset(&req, 0xA5, sizeof(req));
    vkGetBufferMemoryRequirements(e->device, buffer, &req);
    return req;
}

static VkDeviceMemory allocate(struct ct *t, const struct ct_env *e, uint32_t type, VkDeviceSize size,
                               VkMemoryAllocateFlags flags)
{
    VkMemoryAllocateFlagsInfo flags_info;
    VkMemoryAllocateInfo info;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    memset(&flags_info, 0, sizeof(flags_info));
    flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags_info.flags = flags;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.pNext = flags ? &flags_info : NULL;
    info.allocationSize = size;
    info.memoryTypeIndex = type;
    CT_REQUIRE_VK(t, vkAllocateMemory(e->device, &info, NULL, &memory));
    return memory;
}

/* The offset alignment usage requires of requirements (Resource Memory
 * Association: alignment satisfies the descriptor offset alignments). */
static VkDeviceSize usage_alignment(const struct ct_env *e, VkBufferUsageFlags usage)
{
    const VkPhysicalDeviceLimits *l = &e->props.limits;
    VkDeviceSize a = 1;
    if (usage & (VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT))
        a = a > l->minTexelBufferOffsetAlignment ? a : l->minTexelBufferOffsetAlignment;
    if (usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
        a = a > l->minUniformBufferOffsetAlignment ? a : l->minUniformBufferOffsetAlignment;
    if (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        a = a > l->minStorageBufferOffsetAlignment ? a : l->minStorageBufferOffsetAlignment;
    return a;
}

/* A device with a Vulkan 1.2/1.3 feature enabled. A feature that Vulkan
 * requires for the device's version is a failure when missing, otherwise a
 * skip. */
enum feature { BUFFER_DEVICE_ADDRESS, MAINTENANCE4 };

static void env_with_feature(struct ct *t, struct ct_env *e, enum feature which)
{
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    struct ct_features f;
    const char *name = which == BUFFER_DEVICE_ADDRESS ? "bufferDeviceAddress" : "maintenance4";
    uint32_t introduced = which == BUFFER_DEVICE_ADDRESS ? VK_API_VERSION_1_2 : VK_API_VERSION_1_3;
    VkBool32 have;

    ct_env_init(t, e);
    if (e->version < introduced)
        ct_skip(t, "%s is a Vulkan %u.%u feature; the device offers %u.%u", name, VK_API_VERSION_MAJOR(introduced),
                VK_API_VERSION_MINOR(introduced), VK_API_VERSION_MAJOR(e->version), VK_API_VERSION_MINOR(e->version));
    ct_get_features(e, &f, 0);
    have = which == BUFFER_DEVICE_ADDRESS ? f.v12.bufferDeviceAddress : f.v13.maintenance4;
    if (!have) {
        if (e->version >= VK_API_VERSION_1_3) {
            ct_fail(t, __FILE__, __LINE__, "%s is not supported, but Vulkan 1.3 requires it (Features > Feature "
                    "Requirements); this test cannot run", name);
            ct_abort(t);
        }
        ct_skip(t, "%s is not supported", name);
    }
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&v13, 0, sizeof(v13));
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    if (which == BUFFER_DEVICE_ADDRESS)
        v12.bufferDeviceAddress = VK_TRUE;
    else
        v13.maintenance4 = VK_TRUE;
    ct_env_device(t, e, which == BUFFER_DEVICE_ADDRESS ? (const void *)&v12 : (const void *)&v13);
}

/* ==========================================================================
 * Buffers
 * ========================================================================== */

CT_TEST(create_buffer,
        "Resource Creation > Buffers: vkCreateBuffer creates buffers of any non-zero size up to maxBufferSize with "
        "any valid usage; vkDestroyBuffer destroys them",
        "vkCreateBuffer vkDestroyBuffer")
{
    struct ct_env e;
    size_t s, u;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    for (s = 0; s < CT_ARRAY_SIZE(sizes); s++)
        for (u = 0; u < CT_ARRAY_SIZE(usages); u++) {
            VkBuffer buffer = make_buffer(t, &e, sizes[s], usages[u]);
            vkDestroyBuffer(e.device, buffer, NULL);
        }
    ct_env_fini(&e);
}

CT_TEST(create_buffer_concurrent,
        "Resource Creation > Buffers: a buffer can be shared VK_SHARING_MODE_CONCURRENT between two or more "
        "distinct queue families",
        "vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements")
{
    struct ct_env e;
    VkBufferCreateInfo info = buffer_info(4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    uint32_t families[2] = { 0, 1 }, count = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkMemoryRequirements req;

    ct_env_init(t, &e);
    vkGetPhysicalDeviceQueueFamilyProperties(e.pd, &count, NULL);
    if (count < 2)
        ct_skip(t, "one queue family: concurrent sharing needs two (VUID-VkBufferCreateInfo-sharingMode-00914)");
    ct_env_device(t, &e, NULL);
    info.sharingMode = VK_SHARING_MODE_CONCURRENT;
    info.queueFamilyIndexCount = 2;
    info.pQueueFamilyIndices = families;
    CT_REQUIRE_VK(t, vkCreateBuffer(e.device, &info, NULL, &buffer));
    req = requirements(&e, buffer);
    CT_CHECK(t, req.memoryTypeBits != 0 && req.size >= 4096);
    vkDestroyBuffer(e.device, buffer, NULL);
    ct_env_fini(&e);
}

CT_TEST(destroy_buffer_null,
        "Resource Creation > Buffers: vkDestroyBuffer accepts VK_NULL_HANDLE (implicit valid usage: \"if buffer "
        "is not VK_NULL_HANDLE\") and does nothing",
        "vkDestroyBuffer")
{
    struct ct_env e;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    vkDestroyBuffer(e.device, VK_NULL_HANDLE, NULL);
    ct_env_fini(&e);
}

CT_TEST(buffer_allocation_callbacks,
        "Memory Allocation > Host Memory: host memory a buffer needs comes from the allocator given to "
        "vkCreateBuffer (else the device's), following the callback rules, and vkDestroyBuffer releases it",
        "vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements")
{
    struct ct_env e;
    struct ct_allocator a;
    VkBuffer buffers[8];
    uint32_t i;
    VkResult r;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    ct_allocator_init(&a, -1);
    for (i = 0; i < 8; i++) {
        VkBufferCreateInfo info = buffer_info(1000u * (i + 1), usages[i]);
        CT_CALL_VK(&a, r, vkCreateBuffer(e.device, &info, &a.cb, &buffers[i]));
        CT_REQUIRE_VK(t, r);
    }
    for (i = 0; i < 8; i++)
        CT_CALL(&a, vkDestroyBuffer(e.device, buffers[i], &a.cb));
    CT_CHECKF(t, a.live == 0, "%ld allocations still live after vkDestroyBuffer", a.live);
    ct_allocator_check(t, &a, "buffer");
    ct_allocator_fini(&a);
    ct_env_fini(&e);
}

CT_TEST(buffer_allocation_failure,
        "Memory Allocation > Host Memory: when pfnAllocation returns NULL, vkCreateBuffer either continues or "
        "returns VK_ERROR_OUT_OF_HOST_MEMORY, frees what it allocated, and the device keeps working",
        "vkCreateBuffer vkDestroyBuffer")
{
    struct ct_env e;
    VkBufferCreateInfo info = buffer_info(65536, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkBuffer buffer;
    long n;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    for (n = 0; n < 200; n++) {
        struct ct_allocator a;
        VkResult r;
        int failed;
        ct_allocator_init(&a, n);
        buffer = VK_NULL_HANDLE;
        CT_CALL_VK(&a, r, vkCreateBuffer(e.device, &info, &a.cb, &buffer));
        CT_CHECKF(t, r == VK_SUCCESS || r == VK_ERROR_OUT_OF_HOST_MEMORY, "allocation %ld failing: %s", n,
                  ct_result_name(r));
        if (r == VK_SUCCESS)
            CT_CALL(&a, vkDestroyBuffer(e.device, buffer, &a.cb));
        CT_CHECKF(t, a.live == 0, "allocation %ld failing: %ld host allocations leaked", n, a.live);
        ct_allocator_check(t, &a, "buffer");
        failed = a.failed > 0;
        ct_allocator_fini(&a);
        if (t->failures || !failed)
            break;
    }
    buffer = make_buffer(t, &e, 4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    vkDestroyBuffer(e.device, buffer, NULL);
    ct_env_fini(&e);
}

/* ==========================================================================
 * Memory requirements
 * ========================================================================== */

CT_TEST(requirements_memory_types,
        "Resource Creation > Resource Memory Association: memoryTypeBits has at least one bit, includes a "
        "HOST_VISIBLE and HOST_COHERENT type (non-sparse, unprotected buffer) and a DEVICE_LOCAL type, and never a "
        "LAZILY_ALLOCATED type; bit i set only for existing types",
        "vkGetBufferMemoryRequirements vkCreateBuffer vkDestroyBuffer")
{
    struct ct_env e;
    size_t s, u;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    for (s = 0; s < CT_ARRAY_SIZE(sizes); s++)
        for (u = 0; u < CT_ARRAY_SIZE(usages); u++) {
            VkBuffer buffer = make_buffer(t, &e, sizes[s], usages[u]);
            VkMemoryRequirements req = requirements(&e, buffer);
            uint32_t bits = req.memoryTypeBits, i;
            CT_CHECKF(t, bits != 0, "usage 0x%x: memoryTypeBits 0", usages[u]);
            CT_CHECKF(t, e.mem.memoryTypeCount >= 32 || !(bits >> e.mem.memoryTypeCount),
                      "usage 0x%x: memoryTypeBits 0x%x names types beyond memoryTypeCount %u", usages[u], bits,
                      e.mem.memoryTypeCount);
            CT_CHECKF(t, ct_find_memory_type(&e, bits, HOST_COHERENT, 0) != UINT32_MAX,
                      "usage 0x%x: memoryTypeBits 0x%x has no host-visible coherent type", usages[u], bits);
            CT_CHECKF(t, ct_find_memory_type(&e, bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0) != UINT32_MAX,
                      "usage 0x%x: memoryTypeBits 0x%x has no device-local type", usages[u], bits);
            for (i = 0; i < e.mem.memoryTypeCount; i++)
                if (bits & (1u << i))
                    CT_CHECKF(t, !(e.mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT),
                              "usage 0x%x: buffer may use lazily allocated type %u", usages[u], i);
            vkDestroyBuffer(e.device, buffer, NULL);
        }
    ct_env_fini(&e);
}

CT_TEST(requirements_alignment,
        "Resource Creation > Resource Memory Association: alignment is a power of two and a multiple of "
        "minTexelBufferOffsetAlignment, minUniformBufferOffsetAlignment and minStorageBufferOffsetAlignment for "
        "texel, uniform and storage usage",
        "vkGetBufferMemoryRequirements vkCreateBuffer vkDestroyBuffer")
{
    struct ct_env e;
    size_t s, u;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    for (s = 0; s < CT_ARRAY_SIZE(sizes); s++)
        for (u = 0; u < CT_ARRAY_SIZE(usages); u++) {
            VkBuffer buffer = make_buffer(t, &e, sizes[s], usages[u]);
            VkMemoryRequirements req = requirements(&e, buffer);
            VkDeviceSize need = usage_alignment(&e, usages[u]);
            CT_CHECKF(t, ct_is_pow2(req.alignment), "usage 0x%x: alignment %llu", usages[u],
                      (unsigned long long)req.alignment);
            CT_CHECKF(t, need && req.alignment % need == 0, "usage 0x%x: alignment %llu is not a multiple of %llu",
                      usages[u], (unsigned long long)req.alignment, (unsigned long long)need);
            vkDestroyBuffer(e.device, buffer, NULL);
        }
    ct_env_fini(&e);
}

CT_TEST(requirements_size,
        "Resource Creation > Resource Memory Association: VkMemoryRequirements::size is the allocation size the "
        "buffer needs, so it covers VkBufferCreateInfo::size (Buffer Device Addresses: the whole range accesses "
        "bound memory)",
        "vkGetBufferMemoryRequirements vkCreateBuffer vkDestroyBuffer")
{
    struct ct_env e;
    size_t s, u;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    for (s = 0; s < CT_ARRAY_SIZE(sizes); s++)
        for (u = 0; u < CT_ARRAY_SIZE(usages); u++) {
            VkBuffer buffer = make_buffer(t, &e, sizes[s], usages[u]);
            VkMemoryRequirements req = requirements(&e, buffer);
            CT_CHECKF(t, req.size >= sizes[s], "size %llu, usage 0x%x: requirement %llu", (unsigned long long)sizes[s],
                      usages[u], (unsigned long long)req.size);
            vkDestroyBuffer(e.device, buffer, NULL);
        }
    ct_env_fini(&e);
}

CT_TEST(requirements_identical,
        "Resource Creation > Resource Memory Association: memoryTypeBits and alignment are identical for all "
        "buffers with the same usage and flags, and size identical for identical create infos",
        "vkGetBufferMemoryRequirements vkCreateBuffer vkDestroyBuffer")
{
    struct ct_env e;
    size_t s, u;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    for (u = 0; u < CT_ARRAY_SIZE(usages); u++) {
        VkMemoryRequirements first;
        for (s = 0; s < CT_ARRAY_SIZE(sizes); s++) {
            VkBuffer a = make_buffer(t, &e, sizes[s], usages[u]);
            VkBuffer b = make_buffer(t, &e, sizes[s], usages[u]);
            VkMemoryRequirements ra = requirements(&e, a), rb = requirements(&e, b);
            if (s == 0)
                first = ra;
            CT_CHECKF(t, ra.size == rb.size && ra.alignment == rb.alignment && ra.memoryTypeBits == rb.memoryTypeBits,
                      "usage 0x%x size %llu: two identical buffers have different requirements", usages[u],
                      (unsigned long long)sizes[s]);
            CT_CHECKF(t, ra.memoryTypeBits == first.memoryTypeBits, "usage 0x%x: memoryTypeBits 0x%x for size %llu, "
                      "0x%x for size %llu", usages[u], ra.memoryTypeBits, (unsigned long long)sizes[s],
                      first.memoryTypeBits, (unsigned long long)sizes[0]);
            CT_CHECKF(t, ra.alignment == first.alignment, "usage 0x%x: alignment %llu for size %llu, %llu for size "
                      "%llu", usages[u], (unsigned long long)ra.alignment, (unsigned long long)sizes[s],
                      (unsigned long long)first.alignment, (unsigned long long)sizes[0]);
            vkDestroyBuffer(e.device, a, NULL);
            vkDestroyBuffer(e.device, b, NULL);
        }
    }
    ct_env_fini(&e);
}

CT_TEST(requirements_usage_subset,
        "Resource Creation > Resource Memory Association: if the usage bits of usage2 are a subset of usage1's, "
        "the memoryTypeBits for usage1 are a subset of those for usage2",
        "vkGetBufferMemoryRequirements vkCreateBuffer vkDestroyBuffer")
{
    struct ct_env e;
    uint32_t bits[CT_ARRAY_SIZE(usages)];
    size_t a, b;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    for (a = 0; a < CT_ARRAY_SIZE(usages); a++) {
        VkBuffer buffer = make_buffer(t, &e, 1024, usages[a]);
        bits[a] = requirements(&e, buffer).memoryTypeBits;
        vkDestroyBuffer(e.device, buffer, NULL);
    }
    for (a = 0; a < CT_ARRAY_SIZE(usages); a++)
        for (b = 0; b < CT_ARRAY_SIZE(usages); b++)
            if ((usages[b] & usages[a]) == usages[b]) /* usage b is a subset of usage a */
                CT_CHECKF(t, (bits[a] & bits[b]) == bits[a],
                          "usage 0x%x allows types 0x%x, not a subset of 0x%x allowed for its subset 0x%x", usages[a],
                          bits[a], bits[b], usages[b]);
    ct_env_fini(&e);
}

CT_TEST(requirements2,
        "Resource Creation > Resource Memory Association: vkGetBufferMemoryRequirements2 returns the requirements "
        "vkGetBufferMemoryRequirements returns; VkMemoryDedicatedRequirements::requiresDedicatedAllocation is "
        "VK_FALSE for a non-external buffer and implies prefersDedicatedAllocation",
        "vkGetBufferMemoryRequirements2 vkGetBufferMemoryRequirements vkCreateBuffer vkDestroyBuffer")
{
    struct ct_env e;
    size_t s, u;
    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_1)
        ct_skip(t, "vkGetBufferMemoryRequirements2 is Vulkan 1.1");
    ct_env_device(t, &e, NULL);
    for (s = 0; s < CT_ARRAY_SIZE(sizes); s++)
        for (u = 0; u < CT_ARRAY_SIZE(usages); u++) {
            VkBuffer buffer = make_buffer(t, &e, sizes[s], usages[u]);
            VkMemoryRequirements req = requirements(&e, buffer);
            VkBufferMemoryRequirementsInfo2 info;
            VkMemoryDedicatedRequirements dedicated;
            VkMemoryRequirements2 req2;

            memset(&info, 0, sizeof(info));
            info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
            info.buffer = buffer;
            memset(&dedicated, 0xA5, sizeof(dedicated));
            dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
            dedicated.pNext = NULL;
            memset(&req2, 0x5A, sizeof(req2));
            req2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
            req2.pNext = &dedicated;
            vkGetBufferMemoryRequirements2(e.device, &info, &req2);
            CT_CHECKF(t, req2.memoryRequirements.size == req.size &&
                             req2.memoryRequirements.alignment == req.alignment &&
                             req2.memoryRequirements.memoryTypeBits == req.memoryTypeBits,
                      "usage 0x%x size %llu: requirements differ", usages[u], (unsigned long long)sizes[s]);
            CT_CHECKF(t, dedicated.requiresDedicatedAllocation == VK_FALSE,
                      "usage 0x%x: requiresDedicatedAllocation is %u", usages[u], dedicated.requiresDedicatedAllocation);
            CT_CHECKF(t, dedicated.prefersDedicatedAllocation == VK_FALSE ||
                             dedicated.prefersDedicatedAllocation == VK_TRUE,
                      "usage 0x%x: prefersDedicatedAllocation is %u", usages[u], dedicated.prefersDedicatedAllocation);
            CT_CHECK(t, req2.pNext == &dedicated && dedicated.pNext == NULL);
            vkDestroyBuffer(e.device, buffer, NULL);
        }
    ct_env_fini(&e);
}

CT_TEST(device_buffer_requirements,
        "Resource Creation > Resource Memory Association: with maintenance4 enabled, "
        "vkGetDeviceBufferMemoryRequirements returns the requirements vkGetBufferMemoryRequirements2 returns for "
        "a buffer created with the same VkBufferCreateInfo (dedicated requirements included)",
        "vkGetDeviceBufferMemoryRequirements vkGetBufferMemoryRequirements2 vkCreateBuffer vkDestroyBuffer")
{
    struct ct_env e;
    size_t s, u;
    env_with_feature(t, &e, MAINTENANCE4);
    for (s = 0; s < CT_ARRAY_SIZE(sizes); s++)
        for (u = 0; u < CT_ARRAY_SIZE(usages); u++) {
            VkBufferCreateInfo create = buffer_info(sizes[s], usages[u]);
            VkBuffer buffer = make_buffer(t, &e, sizes[s], usages[u]);
            VkBufferMemoryRequirementsInfo2 info;
            VkDeviceBufferMemoryRequirements dev_info;
            VkMemoryDedicatedRequirements ded_a, ded_b;
            VkMemoryRequirements2 a, b;

            memset(&info, 0, sizeof(info));
            info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
            info.buffer = buffer;
            memset(&ded_a, 0, sizeof(ded_a));
            ded_a.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
            ded_b = ded_a;
            memset(&a, 0, sizeof(a));
            a.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
            b = a;
            a.pNext = &ded_a;
            b.pNext = &ded_b;
            memset(&dev_info, 0, sizeof(dev_info));
            dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS;
            dev_info.pCreateInfo = &create;
            vkGetBufferMemoryRequirements2(e.device, &info, &a);
            vkGetDeviceBufferMemoryRequirements(e.device, &dev_info, &b);
            CT_CHECKF(t, a.memoryRequirements.size == b.memoryRequirements.size &&
                             a.memoryRequirements.alignment == b.memoryRequirements.alignment &&
                             a.memoryRequirements.memoryTypeBits == b.memoryRequirements.memoryTypeBits,
                      "size %llu usage 0x%x: buffer {%llu, %llu, 0x%x}, create info {%llu, %llu, 0x%x}",
                      (unsigned long long)sizes[s], usages[u], (unsigned long long)a.memoryRequirements.size,
                      (unsigned long long)a.memoryRequirements.alignment, a.memoryRequirements.memoryTypeBits,
                      (unsigned long long)b.memoryRequirements.size,
                      (unsigned long long)b.memoryRequirements.alignment, b.memoryRequirements.memoryTypeBits);
            CT_CHECK(t, ded_a.prefersDedicatedAllocation == ded_b.prefersDedicatedAllocation &&
                            ded_a.requiresDedicatedAllocation == ded_b.requiresDedicatedAllocation);
            vkDestroyBuffer(e.device, buffer, NULL);
        }
    ct_env_fini(&e);
}

CT_TEST(requirements_maintenance4_size,
        "Resource Creation > Resource Memory Association: with maintenance4 enabled, a buffer's size requirement "
        "is never greater than VkBufferCreateInfo::size aligned to the alignment requirement, nor greater than "
        "that of a larger buffer with otherwise identical parameters",
        "vkGetBufferMemoryRequirements vkCreateBuffer vkDestroyBuffer")
{
    struct ct_env e;
    size_t s, u;
    env_with_feature(t, &e, MAINTENANCE4);
    for (u = 0; u < CT_ARRAY_SIZE(usages); u++) {
        VkDeviceSize previous = 0;
        for (s = 0; s < CT_ARRAY_SIZE(sizes); s++) {
            VkBuffer buffer = make_buffer(t, &e, sizes[s], usages[u]);
            VkMemoryRequirements req = requirements(&e, buffer);
            VkDeviceSize aligned = req.alignment ? (sizes[s] + req.alignment - 1) / req.alignment * req.alignment : 0;
            CT_CHECKF(t, req.size <= aligned, "size %llu usage 0x%x: requirement %llu exceeds %llu",
                      (unsigned long long)sizes[s], usages[u], (unsigned long long)req.size,
                      (unsigned long long)aligned);
            CT_CHECKF(t, req.size >= previous, "usage 0x%x: size %llu needs %llu, less than a smaller buffer's %llu",
                      usages[u], (unsigned long long)sizes[s], (unsigned long long)req.size,
                      (unsigned long long)previous);
            previous = req.size;
            vkDestroyBuffer(e.device, buffer, NULL);
        }
    }
    ct_env_fini(&e);
}

/* ==========================================================================
 * Binding
 * ========================================================================== */

CT_TEST(bind_buffer_memory,
        "Resource Creation > Resource Memory Association: vkBindBufferMemory binds buffers at any offset that is "
        "a multiple of the alignment with the size requirement fitting, in memory of any type allowed by "
        "memoryTypeBits, and returns VK_SUCCESS",
        "vkBindBufferMemory vkAllocateMemory vkFreeMemory vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements")
{
    struct ct_env e;
    VkBuffer probe;
    VkMemoryRequirements req;
    uint32_t type;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    probe = make_buffer(t, &e, 1000, usages[CT_ARRAY_SIZE(usages) - 1]);
    req = requirements(&e, probe);
    vkDestroyBuffer(e.device, probe, NULL);
    CT_REQUIRE(t, ct_is_pow2(req.alignment) && req.size >= 1000);
    for (type = 0; type < e.mem.memoryTypeCount; type++) {
        enum { N = 5 };
        VkBuffer buffers[N];
        VkDeviceSize slot = req.size + req.alignment, memsize = slot * N + req.alignment * 3, offsets[N];
        VkDeviceMemory memory;
        uint32_t i;

        if (!(req.memoryTypeBits & (1u << type)) || (e.mem.memoryTypes[type].propertyFlags & SPECIAL_TYPES))
            continue;
        memory = allocate(t, &e, type, memsize, 0);
        /* Out of order, with gaps, the last one ending at the end of memory. */
        for (i = 0; i < N - 1; i++)
            offsets[i] = (N - 2 - i) * slot / req.alignment * req.alignment;
        offsets[N - 1] = (memsize - req.size) / req.alignment * req.alignment;
        for (i = 0; i < N; i++) {
            buffers[i] = make_buffer(t, &e, 1000, usages[CT_ARRAY_SIZE(usages) - 1]);
            CT_CHECK_VK(t, vkBindBufferMemory(e.device, buffers[i], memory, offsets[i]), VK_SUCCESS);
        }
        for (i = 0; i < N; i++)
            vkDestroyBuffer(e.device, buffers[i], NULL);
        vkFreeMemory(e.device, memory, NULL);
    }
    ct_env_fini(&e);
}

CT_TEST(bind_buffer_memory_mapped,
        "Resource Creation > Resource Memory Association: binding a buffer does not disturb the memory's "
        "contents or its mapping (host-visible memory bound while mapped keeps the bytes written)",
        "vkBindBufferMemory vkMapMemory vkUnmapMemory vkAllocateMemory vkFreeMemory vkCreateBuffer "
        "vkDestroyBuffer vkGetBufferMemoryRequirements")
{
    struct ct_env e;
    VkBuffer buffer;
    VkMemoryRequirements req;
    VkDeviceMemory memory;
    uint32_t type, i;
    unsigned char *p;
    void *v = NULL;
    int same = 1;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    buffer = make_buffer(t, &e, 65536, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    req = requirements(&e, buffer);
    type = ct_find_memory_type(&e, req.memoryTypeBits, HOST_COHERENT, SPECIAL_TYPES);
    CT_REQUIRE(t, type != UINT32_MAX);
    memory = allocate(t, &e, type, req.size + req.alignment, 0);
    CT_REQUIRE_VK(t, vkMapMemory(e.device, memory, 0, VK_WHOLE_SIZE, 0, &v));
    p = v;
    for (i = 0; i < 65536; i++)
        p[i] = (unsigned char)(i * 31);
    CT_CHECK_VK(t, vkBindBufferMemory(e.device, buffer, memory, req.alignment), VK_SUCCESS);
    for (i = 0; i < 65536; i++)
        same &= p[i] == (unsigned char)(i * 31);
    CT_CHECK(t, same);
    vkUnmapMemory(e.device, memory);
    vkDestroyBuffer(e.device, buffer, NULL);
    vkFreeMemory(e.device, memory, NULL);
    ct_env_fini(&e);
}

CT_TEST(bind_buffer_memory2,
        "Resource Creation > Resource Memory Association: vkBindBufferMemory2 binds several buffers in one call "
        "and returns VK_SUCCESS; a VkBindMemoryStatus in a bind's pNext receives that bind's result (Vulkan 1.4)",
        "vkBindBufferMemory2 vkAllocateMemory vkFreeMemory vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements")
{
    enum { N = 4 };
    struct ct_env e;
    VkBuffer buffers[N];
    VkBindBufferMemoryInfo binds[N];
    VkBindMemoryStatus status[N];
    VkResult results[N];
    VkMemoryRequirements req;
    VkDeviceMemory memory;
    VkDeviceSize slot;
    uint32_t type, i;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_1)
        ct_skip(t, "vkBindBufferMemory2 is Vulkan 1.1");
    ct_env_device(t, &e, NULL);
    for (i = 0; i < N; i++)
        buffers[i] = make_buffer(t, &e, 3000, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    req = requirements(&e, buffers[0]);
    CT_REQUIRE(t, ct_is_pow2(req.alignment));
    type = ct_find_memory_type(&e, req.memoryTypeBits, 0, SPECIAL_TYPES);
    CT_REQUIRE(t, type != UINT32_MAX);
    slot = (req.size + req.alignment - 1) / req.alignment * req.alignment;
    memory = allocate(t, &e, type, slot * N, 0);
    for (i = 0; i < N; i++) {
        memset(&binds[i], 0, sizeof(binds[i]));
        binds[i].sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
        binds[i].buffer = buffers[i];
        binds[i].memory = memory;
        binds[i].memoryOffset = slot * (N - 1 - i);
        memset(&status[i], 0, sizeof(status[i]));
        status[i].sType = VK_STRUCTURE_TYPE_BIND_MEMORY_STATUS;
        status[i].pResult = &results[i];
        results[i] = VK_ERROR_UNKNOWN;
        if (e.version >= VK_API_VERSION_1_4 && i % 2 == 0)
            binds[i].pNext = &status[i];
    }
    CT_CHECK_VK(t, vkBindBufferMemory2(e.device, N, binds), VK_SUCCESS);
    if (e.version >= VK_API_VERSION_1_4)
        for (i = 0; i < N; i += 2)
            CT_CHECKF(t, results[i] == VK_SUCCESS, "VkBindMemoryStatus of bind %u: %s", i, ct_result_name(results[i]));
    for (i = 0; i < N; i++)
        vkDestroyBuffer(e.device, buffers[i], NULL);
    vkFreeMemory(e.device, memory, NULL);
    ct_env_fini(&e);
}

/* ==========================================================================
 * Usage flags 2
 * ========================================================================== */

CT_TEST(buffer_usage_flags2,
        "Resource Creation > Buffers: with VkBufferUsageFlags2CreateInfo in pNext, VkBufferCreateInfo::usage is "
        "ignored (it may be 0) and the effective usage is VkBufferUsageFlags2CreateInfo::usage: requirements match "
        "a buffer created with that usage directly",
        "vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements")
{
    struct ct_env e;
    size_t u;
    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_4)
        ct_skip(t, "VkBufferUsageFlags2CreateInfo is Vulkan 1.4");
    ct_env_device(t, &e, NULL);
    for (u = 0; u < CT_ARRAY_SIZE(usages); u++) {
        static const VkBufferUsageFlags legacy[] = { 0, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT };
        VkBuffer direct = make_buffer(t, &e, 5000, usages[u]);
        VkMemoryRequirements want = requirements(&e, direct);
        size_t k;
        for (k = 0; k < CT_ARRAY_SIZE(legacy); k++) {
            VkBufferUsageFlags2CreateInfo usage2;
            VkBufferCreateInfo info = buffer_info(5000, legacy[k]);
            VkBuffer buffer = VK_NULL_HANDLE;
            VkMemoryRequirements got;
            memset(&usage2, 0, sizeof(usage2));
            usage2.sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO;
            usage2.usage = usages[u];
            info.pNext = &usage2;
            CT_REQUIRE_VK(t, vkCreateBuffer(e.device, &info, NULL, &buffer));
            got = requirements(&e, buffer);
            CT_CHECKF(t, got.size == want.size && got.alignment == want.alignment &&
                             got.memoryTypeBits == want.memoryTypeBits,
                      "usage2 0x%x with legacy usage 0x%x: requirements differ from usage 0x%x", usages[u], legacy[k],
                      usages[u]);
            vkDestroyBuffer(e.device, buffer, NULL);
        }
        vkDestroyBuffer(e.device, direct, NULL);
    }
    ct_env_fini(&e);
}

/* ==========================================================================
 * Buffer device addresses
 * ========================================================================== */

static VkDeviceAddress address_of(const struct ct_env *e, VkBuffer buffer)
{
    VkBufferDeviceAddressInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(e->device, &info);
}

CT_TEST(buffer_device_address,
        "Resource Creation > Buffer Device Addresses: vkGetBufferDeviceAddress never returns 0; the address minus "
        "the bind offset is aligned to the alignment requirement (and the storage/uniform offset alignments); "
        "buffers bound to disjoint memory have disjoint address ranges; the address does not change",
        "vkGetBufferDeviceAddress vkBindBufferMemory vkAllocateMemory vkFreeMemory vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements")
{
    enum { N = 6 };
    static const VkDeviceSize buffer_sizes[N] = { 1, 1000, 4096, 65536, 17, 300000 };
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    struct ct_env e;
    VkBuffer buffers[N];
    VkMemoryRequirements req[N];
    VkDeviceSize offsets[N], end = 0, align;
    VkDeviceAddress address[N];
    VkDeviceMemory memory[2];
    uint32_t type, bits = ~0u, i, j;

    env_with_feature(t, &e, BUFFER_DEVICE_ADDRESS);
    for (i = 0; i < N; i++) {
        buffers[i] = make_buffer(t, &e, buffer_sizes[i], usage);
        req[i] = requirements(&e, buffers[i]);
        bits &= req[i].memoryTypeBits;
    }
    type = ct_find_memory_type(&e, bits, 0, SPECIAL_TYPES);
    CT_REQUIRE(t, type != UINT32_MAX);
    /* Buffers 0..N-2 in one allocation, reverse order with gaps; N-1 alone. */
    for (i = N - 1; i-- > 0;) {
        end = (end + req[i].alignment * 2 - 1) / req[i].alignment * req[i].alignment;
        offsets[i] = end;
        end += req[i].size;
    }
    offsets[N - 1] = 0;
    memory[0] = allocate(t, &e, type, end, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
    memory[1] = allocate(t, &e, type, req[N - 1].size, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
    for (i = 0; i < N; i++)
        CT_REQUIRE_VK(t, vkBindBufferMemory(e.device, buffers[i], memory[i == N - 1], offsets[i]));

    align = usage_alignment(&e, usage);
    for (i = 0; i < N; i++) {
        VkDeviceSize a = req[i].alignment > align ? req[i].alignment : align;
        address[i] = address_of(&e, buffers[i]);
        CT_CHECKF(t, address[i] != 0, "buffer %u: address 0", i);
        CT_CHECKF(t, a && (address[i] - offsets[i]) % a == 0, "buffer %u: address 0x%llx - offset %llu is not "
                  "aligned to %llu", i, (unsigned long long)address[i], (unsigned long long)offsets[i],
                  (unsigned long long)a);
        CT_CHECKF(t, address_of(&e, buffers[i]) == address[i], "buffer %u: address changed between queries", i);
        for (j = 0; j < i; j++)
            CT_CHECKF(t, address[i] + buffer_sizes[i] <= address[j] || address[j] + buffer_sizes[j] <= address[i],
                      "buffers %u [0x%llx, +%llu) and %u [0x%llx, +%llu) overlap", i, (unsigned long long)address[i],
                      (unsigned long long)buffer_sizes[i], j, (unsigned long long)address[j],
                      (unsigned long long)buffer_sizes[j]);
    }
    for (i = 0; i < N; i++)
        vkDestroyBuffer(e.device, buffers[i], NULL);
    vkFreeMemory(e.device, memory[0], NULL);
    vkFreeMemory(e.device, memory[1], NULL);
    ct_env_fini(&e);
}

CT_TEST(buffer_device_address_usage2,
        "Resource Creation > Buffers: the effective usage from VkBufferUsageFlags2CreateInfo counts for device "
        "addresses: VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT there (legacy usage without it) makes the buffer's "
        "address queryable and non-zero",
        "vkGetBufferDeviceAddress vkCreateBuffer vkDestroyBuffer vkBindBufferMemory vkAllocateMemory vkFreeMemory "
        "vkGetBufferMemoryRequirements")
{
    struct ct_env e;
    VkBufferUsageFlags2CreateInfo usage2;
    VkBufferCreateInfo info = buffer_info(4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    VkBuffer buffer = VK_NULL_HANDLE;
    VkMemoryRequirements req;
    VkDeviceMemory memory;
    uint32_t type;

    env_with_feature(t, &e, BUFFER_DEVICE_ADDRESS);
    if (e.version < VK_API_VERSION_1_4)
        ct_skip(t, "VkBufferUsageFlags2CreateInfo is Vulkan 1.4");
    memset(&usage2, 0, sizeof(usage2));
    usage2.sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO;
    usage2.usage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT;
    info.pNext = &usage2;
    CT_REQUIRE_VK(t, vkCreateBuffer(e.device, &info, NULL, &buffer));
    req = requirements(&e, buffer);
    type = ct_find_memory_type(&e, req.memoryTypeBits, 0, SPECIAL_TYPES);
    CT_REQUIRE(t, type != UINT32_MAX);
    memory = allocate(t, &e, type, req.size, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
    CT_REQUIRE_VK(t, vkBindBufferMemory(e.device, buffer, memory, 0));
    CT_CHECK(t, address_of(&e, buffer) != 0);
    vkDestroyBuffer(e.device, buffer, NULL);
    vkFreeMemory(e.device, memory, NULL);
    ct_env_fini(&e);
}

CT_TEST(buffer_capture_replay,
        "Resource Creation > Buffer Device Addresses: with bufferDeviceAddressCaptureReplay, a buffer created with "
        "VkBufferOpaqueCaptureAddressCreateInfo::opaqueCaptureAddress reports that opaque address from "
        "vkGetBufferOpaqueCaptureAddress (or creation fails with VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS)",
        "vkGetBufferOpaqueCaptureAddress vkGetDeviceMemoryOpaqueCaptureAddress vkGetBufferDeviceAddress "
        "vkCreateBuffer vkDestroyBuffer vkAllocateMemory vkFreeMemory vkBindBufferMemory vkGetBufferMemoryRequirements")
{
    struct ct_env e;
    struct ct_features f;
    VkPhysicalDeviceVulkan12Features v12;
    VkBufferCreateInfo info = buffer_info(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    VkBufferOpaqueCaptureAddressCreateInfo replay;
    VkBufferDeviceAddressInfo addr_info;
    VkDeviceMemoryOpaqueCaptureAddressInfo mem_info;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkMemoryRequirements req;
    VkDeviceMemory memory;
    uint64_t captured;
    uint32_t type;
    VkResult r;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_2)
        ct_skip(t, "capture/replay is Vulkan 1.2");
    ct_get_features(&e, &f, 0);
    if (!f.v12.bufferDeviceAddress || !f.v12.bufferDeviceAddressCaptureReplay)
        ct_skip(t, "bufferDeviceAddressCaptureReplay is not supported");
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.bufferDeviceAddress = VK_TRUE;
    v12.bufferDeviceAddressCaptureReplay = VK_TRUE;
    ct_env_device(t, &e, &v12);

    info.flags = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT;
    CT_REQUIRE_VK(t, vkCreateBuffer(e.device, &info, NULL, &buffer));
    req = requirements(&e, buffer);
    type = ct_find_memory_type(&e, req.memoryTypeBits, 0, SPECIAL_TYPES);
    CT_REQUIRE(t, type != UINT32_MAX);
    memory = allocate(t, &e, type, req.size,
                      VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT | VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT);
    CT_REQUIRE_VK(t, vkBindBufferMemory(e.device, buffer, memory, 0));
    memset(&addr_info, 0, sizeof(addr_info));
    addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.buffer = buffer;
    captured = vkGetBufferOpaqueCaptureAddress(e.device, &addr_info);
    memset(&mem_info, 0, sizeof(mem_info));
    mem_info.sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO;
    mem_info.memory = memory;
    (void)vkGetDeviceMemoryOpaqueCaptureAddress(e.device, &mem_info);
    vkDestroyBuffer(e.device, buffer, NULL);

    memset(&replay, 0, sizeof(replay));
    replay.sType = VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO;
    replay.opaqueCaptureAddress = captured;
    info.pNext = captured ? &replay : NULL;
    buffer = VK_NULL_HANDLE;
    r = vkCreateBuffer(e.device, &info, NULL, &buffer);
    CT_CHECKF(t, r == VK_SUCCESS || r == VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS, "replay: %s", ct_result_name(r));
    if (r == VK_SUCCESS) {
        addr_info.buffer = buffer;
        if (captured)
            CT_CHECK(t, vkGetBufferOpaqueCaptureAddress(e.device, &addr_info) == captured);
        vkDestroyBuffer(e.device, buffer, NULL);
    }
    vkFreeMemory(e.device, memory, NULL);
    ct_env_fini(&e);
}

/* ==========================================================================
 * What the device sees (needs command buffers and, for addresses, a shader)
 * ========================================================================== */

#define GPU_COMMANDS                                                                                             \
    "vkCreateCommandPool vkDestroyCommandPool vkAllocateCommandBuffers vkFreeCommandBuffers "                    \
    "vkBeginCommandBuffer vkEndCommandBuffer vkCreateFence vkDestroyFence vkResetFences vkQueueSubmit "         \
    "vkWaitForFences vkCmdPipelineBarrier vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements "        \
    "vkBindBufferMemory vkAllocateMemory vkFreeMemory vkMapMemory vkUnmapMemory"

CT_TEST(bound_range_on_device,
        "Resource Creation > Resource Memory Association: the bytes of memory from memoryOffset on are bound to "
        "the buffer: transfer writes to two buffers bound at different offsets of one allocation land exactly "
        "in their ranges",
        "vkCmdFillBuffer " GPU_COMMANDS)
{
    struct ct_env e;
    VkBuffer a, b;
    VkMemoryRequirements req;
    VkDeviceMemory memory;
    VkDeviceSize off_a, off_b = 0, size;
    VkCommandBuffer cb;
    uint32_t type;
    unsigned char *p;
    void *v = NULL;
    VkDeviceSize i;
    int ok = 1;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    a = make_buffer(t, &e, 1000, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    b = make_buffer(t, &e, 1000, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    req = requirements(&e, a);
    CT_REQUIRE(t, ct_is_pow2(req.alignment));
    type = ct_find_memory_type(&e, req.memoryTypeBits, HOST_COHERENT, SPECIAL_TYPES);
    CT_REQUIRE(t, type != UINT32_MAX);
    off_a = (req.size + req.alignment - 1) / req.alignment * req.alignment + req.alignment;
    size = off_a + req.size + req.alignment;
    memory = allocate(t, &e, type, size, 0);
    CT_REQUIRE_VK(t, vkBindBufferMemory(e.device, a, memory, off_a));
    CT_REQUIRE_VK(t, vkBindBufferMemory(e.device, b, memory, off_b));
    CT_REQUIRE_VK(t, vkMapMemory(e.device, memory, 0, VK_WHOLE_SIZE, 0, &v));
    p = v;
    memset(p, 0xEE, (size_t)size);

    cb = ct_begin_commands(t, &e);
    vkCmdFillBuffer(cb, a, 0, VK_WHOLE_SIZE, 0x11111111u);
    vkCmdFillBuffer(cb, b, 0, VK_WHOLE_SIZE, 0x22222222u);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &e, cb);
    for (i = 0; i < size; i++) {
        unsigned char want = i >= off_a && i < off_a + 1000 ? 0x11 : i >= off_b && i < off_b + 1000 ? 0x22 : 0xEE;
        if (p[i] != want && ok) {
            ct_fail(t, __FILE__, __LINE__, "memory byte %llu is 0x%02x, expected 0x%02x (buffer A at %llu, B at %llu)",
                    (unsigned long long)i, p[i], want, (unsigned long long)off_a, (unsigned long long)off_b);
            ok = 0;
        }
    }
    vkUnmapMemory(e.device, memory);
    vkDestroyBuffer(e.device, a, NULL);
    vkDestroyBuffer(e.device, b, NULL);
    vkFreeMemory(e.device, memory, NULL);
    ct_env_fini(&e);
}

/* Compute shader writing through a buffer device address (glslangValidator
 * -V --target-env vulkan1.2, SPIR-V 1.5):
 *
 *   #version 450
 *   #extension GL_EXT_buffer_reference : require
 *   layout(local_size_x = 64) in;
 *   layout(buffer_reference, std430, buffer_reference_align = 4) buffer Words { uint data[]; };
 *   layout(push_constant) uniform Push { Words dst; uint count; uint value; } pc;
 *   void main()
 *   {
 *       uint i = gl_GlobalInvocationID.x;
 *       if (i < pc.count)
 *           pc.dst.data[i] = pc.value + i;
 *   }
 */
static const uint32_t address_shader[] = {
    0x07230203, 0x00010500, 0x0008000b, 0x0000002f, 0x00000000, 0x00020011, 0x00000001, 0x00020011, 0x000014e3,
    0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x000014e4, 0x00000001,
    0x0007000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00000016, 0x00060010, 0x00000004,
    0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00070004, 0x455f4c47,
    0x625f5458, 0x65666675, 0x65725f72, 0x65726566, 0x0065636e, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00030005, 0x00000008, 0x00000069, 0x00080005, 0x0000000b, 0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f,
    0x496e6f69, 0x00000044, 0x00040005, 0x00000012, 0x68737550, 0x00000000, 0x00040006, 0x00000012, 0x00000000,
    0x00747364, 0x00050006, 0x00000012, 0x00000001, 0x6e756f63, 0x00000074, 0x00050006, 0x00000012, 0x00000002,
    0x756c6176, 0x00000065, 0x00040005, 0x00000014, 0x64726f57, 0x00000073, 0x00050006, 0x00000014, 0x00000000,
    0x61746164, 0x00000000, 0x00030005, 0x00000016, 0x00006370, 0x00040047, 0x0000000b, 0x0000000b, 0x0000001c,
    0x00030047, 0x00000012, 0x00000002, 0x00050048, 0x00000012, 0x00000000, 0x00000023, 0x00000000, 0x00050048,
    0x00000012, 0x00000001, 0x00000023, 0x00000008, 0x00050048, 0x00000012, 0x00000002, 0x00000023, 0x0000000c,
    0x00040047, 0x00000013, 0x00000006, 0x00000004, 0x00030047, 0x00000014, 0x00000002, 0x00050048, 0x00000014,
    0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x0000002e, 0x0000000b, 0x00000019, 0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040020, 0x00000007,
    0x00000007, 0x00000006, 0x00040017, 0x00000009, 0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001,
    0x00000009, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b, 0x00000006, 0x0000000c, 0x00000000,
    0x00040020, 0x0000000d, 0x00000001, 0x00000006, 0x00030027, 0x00000011, 0x000014e5, 0x0005001e, 0x00000012,
    0x00000011, 0x00000006, 0x00000006, 0x0003001d, 0x00000013, 0x00000006, 0x0003001e, 0x00000014, 0x00000013,
    0x00040020, 0x00000011, 0x000014e5, 0x00000014, 0x00040020, 0x00000015, 0x00000009, 0x00000012, 0x0004003b,
    0x00000015, 0x00000016, 0x00000009, 0x00040015, 0x00000017, 0x00000020, 0x00000001, 0x0004002b, 0x00000017,
    0x00000018, 0x00000001, 0x00040020, 0x00000019, 0x00000009, 0x00000006, 0x00020014, 0x0000001c, 0x0004002b,
    0x00000017, 0x00000020, 0x00000000, 0x00040020, 0x00000021, 0x00000009, 0x00000011, 0x0004002b, 0x00000017,
    0x00000025, 0x00000002, 0x00040020, 0x0000002a, 0x000014e5, 0x00000006, 0x0004002b, 0x00000006, 0x0000002c,
    0x00000040, 0x0004002b, 0x00000006, 0x0000002d, 0x00000001, 0x0006002c, 0x00000009, 0x0000002e, 0x0000002c,
    0x0000002d, 0x0000002d, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
    0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x00050041, 0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c,
    0x0004003d, 0x00000006, 0x0000000f, 0x0000000e, 0x0003003e, 0x00000008, 0x0000000f, 0x0004003d, 0x00000006,
    0x00000010, 0x00000008, 0x00050041, 0x00000019, 0x0000001a, 0x00000016, 0x00000018, 0x0004003d, 0x00000006,
    0x0000001b, 0x0000001a, 0x000500b0, 0x0000001c, 0x0000001d, 0x00000010, 0x0000001b, 0x000300f7, 0x0000001f,
    0x00000000, 0x000400fa, 0x0000001d, 0x0000001e, 0x0000001f, 0x000200f8, 0x0000001e, 0x00050041, 0x00000021,
    0x00000022, 0x00000016, 0x00000020, 0x0004003d, 0x00000011, 0x00000023, 0x00000022, 0x0004003d, 0x00000006,
    0x00000024, 0x00000008, 0x00050041, 0x00000019, 0x00000026, 0x00000016, 0x00000025, 0x0004003d, 0x00000006,
    0x00000027, 0x00000026, 0x0004003d, 0x00000006, 0x00000028, 0x00000008, 0x00050080, 0x00000006, 0x00000029,
    0x00000027, 0x00000028, 0x00060041, 0x0000002a, 0x0000002b, 0x00000023, 0x00000020, 0x00000024, 0x0005003e,
    0x0000002b, 0x00000029, 0x00000002, 0x00000004, 0x000200f9, 0x0000001f, 0x000200f8, 0x0000001f, 0x000100fd,
    0x00010038,
};

struct address_push {
    VkDeviceAddress address;
    uint32_t count;
    uint32_t value;
};

CT_TEST(buffer_device_address_shader,
        "Resource Creation > Buffer Device Addresses: addresses in [address, address + size) access the memory "
        "bound to the buffer: a compute shader writing through the address of a buffer bound at a non-zero "
        "offset changes exactly that buffer's bytes",
        "vkGetBufferDeviceAddress vkCreateShaderModule vkDestroyShaderModule vkCreatePipelineLayout "
        "vkDestroyPipelineLayout vkCreateComputePipelines vkDestroyPipeline vkCmdBindPipeline vkCmdPushConstants "
        "vkCmdDispatch " GPU_COMMANDS)
{
    enum { WORDS = 1000 };
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    struct ct_env e;
    VkBuffer target, neighbour;
    VkMemoryRequirements req;
    VkDeviceMemory memory;
    VkDeviceSize off_target, size;
    VkShaderModuleCreateInfo module_info;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPushConstantRange push_range;
    VkPipelineLayoutCreateInfo layout_info;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkComputePipelineCreateInfo pipeline_info;
    VkPipeline pipeline = VK_NULL_HANDLE;
    struct address_push push;
    VkCommandBuffer cb;
    uint32_t type, *words;
    unsigned char *p;
    void *v = NULL;
    VkDeviceSize i;
    int ok = 1;

    env_with_feature(t, &e, BUFFER_DEVICE_ADDRESS);
    target = make_buffer(t, &e, WORDS * 4, usage);
    neighbour = make_buffer(t, &e, WORDS * 4, usage);
    req = requirements(&e, target);
    CT_REQUIRE(t, ct_is_pow2(req.alignment));
    type = ct_find_memory_type(&e, req.memoryTypeBits, HOST_COHERENT, SPECIAL_TYPES);
    CT_REQUIRE(t, type != UINT32_MAX);
    off_target = (req.size + req.alignment - 1) / req.alignment * req.alignment;
    size = off_target + req.size;
    memory = allocate(t, &e, type, size, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
    CT_REQUIRE_VK(t, vkBindBufferMemory(e.device, neighbour, memory, 0));
    CT_REQUIRE_VK(t, vkBindBufferMemory(e.device, target, memory, off_target));
    CT_REQUIRE_VK(t, vkMapMemory(e.device, memory, 0, VK_WHOLE_SIZE, 0, &v));
    p = v;
    memset(p, 0xEE, (size_t)size);

    memset(&module_info, 0, sizeof(module_info));
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = sizeof(address_shader);
    module_info.pCode = address_shader;
    CT_REQUIRE_VK(t, vkCreateShaderModule(e.device, &module_info, NULL, &module));
    memset(&push_range, 0, sizeof(push_range));
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.size = sizeof(push);
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;
    CT_REQUIRE_VK(t, vkCreatePipelineLayout(e.device, &layout_info, NULL, &layout));
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = layout;
    CT_REQUIRE_VK(t, vkCreateComputePipelines(e.device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline));

    push.address = address_of(&e, target);
    push.count = WORDS;
    push.value = 0x1000;
    CT_REQUIRE(t, push.address != 0);
    cb = ct_begin_commands(t, &e);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cb, (WORDS + 63) / 64, 1, 1);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, &e, cb);

    words = (uint32_t *)(p + off_target);
    for (i = 0; i < WORDS && ok; i++)
        if (words[i] != 0x1000 + i) {
            ct_fail(t, __FILE__, __LINE__, "word %llu of the target buffer is 0x%x, expected 0x%llx",
                    (unsigned long long)i, words[i], (unsigned long long)(0x1000 + i));
            ok = 0;
        }
    for (i = 0; i < size && ok; i++)
        if ((i < off_target || i >= off_target + WORDS * 4) && p[i] != 0xEE) {
            ct_fail(t, __FILE__, __LINE__, "memory byte %llu outside the target buffer was written",
                    (unsigned long long)i);
            ok = 0;
        }
    vkDestroyPipeline(e.device, pipeline, NULL);
    vkDestroyPipelineLayout(e.device, layout, NULL);
    vkDestroyShaderModule(e.device, module, NULL);
    vkUnmapMemory(e.device, memory);
    vkDestroyBuffer(e.device, target, NULL);
    vkDestroyBuffer(e.device, neighbour, NULL);
    vkFreeMemory(e.device, memory, NULL);
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(create_buffer),
    CT_REF(create_buffer_concurrent),
    CT_REF(destroy_buffer_null),
    CT_REF(buffer_allocation_callbacks),
    CT_REF(buffer_allocation_failure),
    CT_REF(requirements_memory_types),
    CT_REF(requirements_alignment),
    CT_REF(requirements_size),
    CT_REF(requirements_identical),
    CT_REF(requirements_usage_subset),
    CT_REF(requirements2),
    CT_REF(device_buffer_requirements),
    CT_REF(requirements_maintenance4_size),
    CT_REF(bind_buffer_memory),
    CT_REF(bind_buffer_memory_mapped),
    CT_REF(bind_buffer_memory2),
    CT_REF(buffer_usage_flags2),
    CT_REF(buffer_device_address),
    CT_REF(buffer_device_address_usage2),
    CT_REF(buffer_capture_replay),
    CT_REF(bound_range_on_device),
    CT_REF(buffer_device_address_shader),
};

CT_SUITE(resources, tests)
