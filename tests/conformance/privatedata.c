/* Chapter "Private Data": slots, vkSetPrivateData and vkGetPrivateData on
 * objects of several types, the value of data never set, reserved slots
 * (VkDevicePrivateDataCreateInfo), allocation callbacks, and
 * VK_EXT_private_data at apiVersion 1.0. */
#include "ct.h"

/* An instance and a device with privateData enabled (VUID-vkCreatePrivateDataSlot-
 * privateData-04564); `reserve` slots requested with VkDevicePrivateDataCreateInfo
 * if not 0. Skips below Vulkan 1.3. */
static void pt_init(struct ct *t, struct ct_env *e, uint32_t reserve, const VkAllocationCallbacks *alloc)
{
    VkDevicePrivateDataCreateInfo slots;
    VkPhysicalDeviceVulkan13Features v13;
    struct ct_features f;

    ct_env_init(t, e);
    if (e->version < VK_API_VERSION_1_3)
        ct_skip(t, "private data is Vulkan 1.3");
    ct_get_features(e, &f, 0);
    CT_REQUIREF(t, f.v13.privateData, "privateData is required by Vulkan 1.3");
    memset(&slots, 0, sizeof(slots));
    slots.sType = VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO;
    slots.privateDataSlotRequestCount = reserve;
    memset(&v13, 0, sizeof(v13));
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13.pNext = reserve ? &slots : NULL;
    v13.privateData = VK_TRUE;
    ct_env_device_ex(t, e, &v13, NULL, 0, alloc);
}

static VkPrivateDataSlot pt_slot(struct ct *t, struct ct_env *e, const VkAllocationCallbacks *alloc)
{
    VkPrivateDataSlotCreateInfo info;
    VkPrivateDataSlot slot = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreatePrivateDataSlot(e->device, &info, alloc, &slot));
    return slot;
}

static uint64_t pt_get(struct ct_env *e, VkObjectType type, uint64_t handle, VkPrivateDataSlot slot)
{
    uint64_t v = 0x5555555555555555ull;
    vkGetPrivateData(e->device, type, handle, slot, &v);
    return v;
}

#define PT_HANDLE(h) ((uint64_t)(uintptr_t)(h))

CT_TEST(private_data_slots,
        "Private Data: vkCreatePrivateDataSlot creates distinct slots, which vkDestroyPrivateDataSlot destroys",
        "vkCreatePrivateDataSlot vkDestroyPrivateDataSlot")
{
    struct ct_env e;
    VkPrivateDataSlot s[8];
    uint32_t i, j;
    pt_init(t, &e, 0, NULL);
    for (i = 0; i < 8; i++)
        s[i] = pt_slot(t, &e, NULL);
    for (i = 0; i < 8; i++)
        for (j = i + 1; j < 8; j++)
            CT_CHECK(t, s[i] != s[j]);
    for (i = 0; i < 8; i++)
        vkDestroyPrivateDataSlot(e.device, s[i], NULL);
    ct_env_fini(&e);
}

CT_TEST(private_data_slot_destroy_null,
        "Private Data > vkDestroyPrivateDataSlot: privateDataSlot may be VK_NULL_HANDLE "
        "(VUID-vkDestroyPrivateDataSlot-privateDataSlot-parameter: must be a valid handle if not VK_NULL_HANDLE)",
        "vkDestroyPrivateDataSlot")
{
    struct ct_env e;
    pt_init(t, &e, 0, NULL);
    vkDestroyPrivateDataSlot(e.device, VK_NULL_HANDLE, NULL);
    ct_env_fini(&e);
}

CT_TEST(private_data_objects,
        "Private Data: vkSetPrivateData stores a value per slot and object, of any object type (here a device, a "
        "queue, a buffer, memory, a fence, a semaphore, an event, a query pool, a sampler, a command pool and buffer, "
        "a private data slot) that vkGetPrivateData returns; 0 is returned for an object of a slot never set, "
        "slots are independent, and a later set replaces the value",
        "vkCreatePrivateDataSlot vkDestroyPrivateDataSlot vkSetPrivateData vkGetPrivateData vkCreateBuffer "
        "vkDestroyBuffer vkAllocateMemory vkFreeMemory vkCreateFence vkDestroyFence vkCreateSemaphore "
        "vkDestroySemaphore vkCreateEvent vkDestroyEvent vkCreateQueryPool vkDestroyQueryPool vkCreateSampler "
        "vkDestroySampler vkCreateCommandPool vkDestroyCommandPool vkAllocateCommandBuffers")
{
    struct ct_env e;
    VkPrivateDataSlot s1, s2;
    VkBufferCreateInfo bi;
    VkMemoryAllocateInfo mi;
    VkFenceCreateInfo fi;
    VkSemaphoreCreateInfo si;
    VkEventCreateInfo ei;
    VkQueryPoolCreateInfo qi;
    VkSamplerCreateInfo smi;
    VkCommandPoolCreateInfo pi;
    VkCommandBufferAllocateInfo ci;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkFence fence;
    VkSemaphore semaphore;
    VkEvent event = VK_NULL_HANDLE;
    VkQueryPool query_pool;
    VkSampler sampler;
    VkCommandPool pool;
    VkCommandBuffer cb;
    struct {
        VkObjectType type;
        uint64_t handle;
    } o[12];
    uint32_t n = 0, i;
    int events = 1;

    pt_init(t, &e, 0, NULL);
    s1 = pt_slot(t, &e, NULL);
    s2 = pt_slot(t, &e, NULL);
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = 256;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    CT_REQUIRE_VK(t, vkCreateBuffer(e.device, &bi, NULL, &buffer));
    memset(&mi, 0, sizeof(mi));
    mi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mi.allocationSize = 4096;
    mi.memoryTypeIndex = 0;
    CT_REQUIRE_VK(t, vkAllocateMemory(e.device, &mi, NULL, &memory));
    memset(&fi, 0, sizeof(fi));
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreateFence(e.device, &fi, NULL, &fence));
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreateSemaphore(e.device, &si, NULL, &semaphore));
    if (e.portability_subset)
        events = 0; /* may lack events (VUID-vkCreateEvent-events-04468); not needed here */
    if (events) {
        memset(&ei, 0, sizeof(ei));
        ei.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
        CT_REQUIRE_VK(t, vkCreateEvent(e.device, &ei, NULL, &event));
    }
    memset(&qi, 0, sizeof(qi));
    qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qi.queryType = VK_QUERY_TYPE_OCCLUSION;
    qi.queryCount = 1;
    CT_REQUIRE_VK(t, vkCreateQueryPool(e.device, &qi, NULL, &query_pool));
    memset(&smi, 0, sizeof(smi));
    smi.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    smi.maxLod = 1.0f;
    CT_REQUIRE_VK(t, vkCreateSampler(e.device, &smi, NULL, &sampler));
    memset(&pi, 0, sizeof(pi));
    pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.queueFamilyIndex = e.queue_family;
    CT_REQUIRE_VK(t, vkCreateCommandPool(e.device, &pi, NULL, &pool));
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ci.commandPool = pool;
    ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ci.commandBufferCount = 1;
    CT_REQUIRE_VK(t, vkAllocateCommandBuffers(e.device, &ci, &cb));

    o[n].type = VK_OBJECT_TYPE_DEVICE, o[n++].handle = PT_HANDLE(e.device);
    o[n].type = VK_OBJECT_TYPE_QUEUE, o[n++].handle = PT_HANDLE(e.queue);
    o[n].type = VK_OBJECT_TYPE_BUFFER, o[n++].handle = PT_HANDLE(buffer);
    o[n].type = VK_OBJECT_TYPE_DEVICE_MEMORY, o[n++].handle = PT_HANDLE(memory);
    o[n].type = VK_OBJECT_TYPE_FENCE, o[n++].handle = PT_HANDLE(fence);
    o[n].type = VK_OBJECT_TYPE_SEMAPHORE, o[n++].handle = PT_HANDLE(semaphore);
    if (events)
        o[n].type = VK_OBJECT_TYPE_EVENT, o[n++].handle = PT_HANDLE(event);
    o[n].type = VK_OBJECT_TYPE_QUERY_POOL, o[n++].handle = PT_HANDLE(query_pool);
    o[n].type = VK_OBJECT_TYPE_SAMPLER, o[n++].handle = PT_HANDLE(sampler);
    o[n].type = VK_OBJECT_TYPE_COMMAND_POOL, o[n++].handle = PT_HANDLE(pool);
    o[n].type = VK_OBJECT_TYPE_COMMAND_BUFFER, o[n++].handle = PT_HANDLE(cb);
    o[n].type = VK_OBJECT_TYPE_PRIVATE_DATA_SLOT, o[n++].handle = PT_HANDLE(s2);

    for (i = 0; i < n; i++)
        CT_CHECKF(t, pt_get(&e, o[i].type, o[i].handle, s1) == 0, "object %u: data never set is not 0", i);
    for (i = 0; i < n; i++)
        CT_CHECK_VK(t, vkSetPrivateData(e.device, o[i].type, o[i].handle, s1, 0x100 + i), VK_SUCCESS);
    for (i = 0; i < n; i += 2)
        CT_CHECK_VK(t, vkSetPrivateData(e.device, o[i].type, o[i].handle, s2, ~(uint64_t)i), VK_SUCCESS);
    for (i = 0; i < n; i++) {
        CT_CHECKF(t, pt_get(&e, o[i].type, o[i].handle, s1) == 0x100 + i, "object %u, slot 1", i);
        CT_CHECKF(t, pt_get(&e, o[i].type, o[i].handle, s2) == (i % 2 ? 0 : ~(uint64_t)i), "object %u, slot 2", i);
    }
    CT_CHECK_VK(t, vkSetPrivateData(e.device, o[2].type, o[2].handle, s1, 42), VK_SUCCESS);
    CT_CHECK(t, pt_get(&e, o[2].type, o[2].handle, s1) == 42);
    CT_CHECK_VK(t, vkSetPrivateData(e.device, o[2].type, o[2].handle, s1, 0), VK_SUCCESS);
    CT_CHECK(t, pt_get(&e, o[2].type, o[2].handle, s1) == 0);
    CT_CHECK(t, pt_get(&e, o[2].type, o[2].handle, s2) == ~(uint64_t)2);

    vkFreeCommandBuffers(e.device, pool, 1, &cb);
    vkDestroyCommandPool(e.device, pool, NULL);
    vkDestroySampler(e.device, sampler, NULL);
    vkDestroyQueryPool(e.device, query_pool, NULL);
    if (events)
        vkDestroyEvent(e.device, event, NULL);
    vkDestroySemaphore(e.device, semaphore, NULL);
    vkDestroyFence(e.device, fence, NULL);
    vkFreeMemory(e.device, memory, NULL);
    vkDestroyBuffer(e.device, buffer, NULL);
    vkDestroyPrivateDataSlot(e.device, s1, NULL);
    vkDestroyPrivateDataSlot(e.device, s2, NULL);
    ct_env_fini(&e);
}

CT_TEST(private_data_new_objects,
        "Private Data > vkGetPrivateData: 0 is written in the absence of a previous vkSetPrivateData for the object: "
        "objects created after others with data set were destroyed (allocators often reuse their addresses, so "
        "their handles) read 0, as do slots created after others were destroyed",
        "vkCreatePrivateDataSlot vkDestroyPrivateDataSlot vkSetPrivateData vkGetPrivateData vkCreateFence "
        "vkDestroyFence")
{
    struct ct_env e;
    VkPrivateDataSlot slot, other;
    VkFenceCreateInfo fi;
    uint32_t i, stale = 0;
    pt_init(t, &e, 0, NULL);
    slot = pt_slot(t, &e, NULL);
    memset(&fi, 0, sizeof(fi));
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    for (i = 0; i < 64; i++) {
        VkFence f;
        CT_REQUIRE_VK(t, vkCreateFence(e.device, &fi, NULL, &f));
        stale += pt_get(&e, VK_OBJECT_TYPE_FENCE, PT_HANDLE(f), slot) != 0;
        CT_CHECK_VK(t, vkSetPrivateData(e.device, VK_OBJECT_TYPE_FENCE, PT_HANDLE(f), slot, 1000 + i), VK_SUCCESS);
        vkDestroyFence(e.device, f, NULL);
    }
    CT_CHECKF(t, stale == 0, "%u of 64 new fences read the data of an earlier, destroyed one", stale);
    CT_CHECK_VK(t, vkSetPrivateData(e.device, VK_OBJECT_TYPE_DEVICE, PT_HANDLE(e.device), slot, 7), VK_SUCCESS);
    vkDestroyPrivateDataSlot(e.device, slot, NULL);
    for (i = 0, stale = 0; i < 16; i++) {
        other = pt_slot(t, &e, NULL);
        stale += pt_get(&e, VK_OBJECT_TYPE_DEVICE, PT_HANDLE(e.device), other) != 0;
        CT_CHECK_VK(t, vkSetPrivateData(e.device, VK_OBJECT_TYPE_DEVICE, PT_HANDLE(e.device), other, 9), VK_SUCCESS);
        vkDestroyPrivateDataSlot(e.device, other, NULL);
    }
    CT_CHECKF(t, stale == 0, "%u of 16 new slots read the data of an earlier, destroyed one", stale);
    ct_env_fini(&e);
}

CT_TEST(private_data_many,
        "Private Data: values set on many objects (2000 fences) in one slot are all returned, also after half of "
        "the objects were destroyed",
        "vkCreatePrivateDataSlot vkDestroyPrivateDataSlot vkSetPrivateData vkGetPrivateData vkCreateFence "
        "vkDestroyFence")
{
    enum { N = 2000 };
    static VkFence fences[N];
    struct ct_env e;
    VkPrivateDataSlot slot;
    VkFenceCreateInfo fi;
    uint32_t i, bad = 0;
    pt_init(t, &e, 0, NULL);
    slot = pt_slot(t, &e, NULL);
    memset(&fi, 0, sizeof(fi));
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    for (i = 0; i < N; i++) {
        CT_REQUIRE_VK(t, vkCreateFence(e.device, &fi, NULL, &fences[i]));
        CT_REQUIRE_VK(t, vkSetPrivateData(e.device, VK_OBJECT_TYPE_FENCE, PT_HANDLE(fences[i]), slot, 3ull * i + 1));
    }
    for (i = 0; i < N; i++)
        bad += pt_get(&e, VK_OBJECT_TYPE_FENCE, PT_HANDLE(fences[i]), slot) != 3ull * i + 1;
    for (i = 0; i < N; i += 2)
        vkDestroyFence(e.device, fences[i], NULL);
    for (i = 1; i < N; i += 2)
        bad += pt_get(&e, VK_OBJECT_TYPE_FENCE, PT_HANDLE(fences[i]), slot) != 3ull * i + 1;
    CT_CHECKF(t, bad == 0, "%u wrong values", bad);
    for (i = 1; i < N; i += 2)
        vkDestroyFence(e.device, fences[i], NULL);
    vkDestroyPrivateDataSlot(e.device, slot, NULL);
    ct_env_fini(&e);
}

CT_TEST(private_data_reserved,
        "Private Data > VkDevicePrivateDataCreateInfo: a device created with privateDataSlotRequestCount slots "
        "reserved creates those and more slots, which work as any other, also when destroyed and created again",
        "vkCreateDevice vkCreatePrivateDataSlot vkDestroyPrivateDataSlot vkSetPrivateData vkGetPrivateData")
{
    struct ct_env e;
    VkPrivateDataSlot s[5];
    uint32_t round, i;
    pt_init(t, &e, 3, NULL);
    for (round = 0; round < 2; round++) {
        for (i = 0; i < 5; i++)
            s[i] = pt_slot(t, &e, NULL);
        for (i = 0; i < 5; i++) {
            CT_CHECK(t, pt_get(&e, VK_OBJECT_TYPE_QUEUE, PT_HANDLE(e.queue), s[i]) == 0);
            CT_CHECK_VK(t, vkSetPrivateData(e.device, VK_OBJECT_TYPE_QUEUE, PT_HANDLE(e.queue), s[i], 77 + i),
                        VK_SUCCESS);
        }
        for (i = 0; i < 5; i++)
            CT_CHECK(t, pt_get(&e, VK_OBJECT_TYPE_QUEUE, PT_HANDLE(e.queue), s[i]) == 77 + i);
        for (i = 0; i < 5; i++)
            vkDestroyPrivateDataSlot(e.device, s[i], NULL);
    }
    ct_env_fini(&e);
}

CT_TEST(private_data_allocator,
        "Private Data, Memory Allocation > Host Memory: a slot created with allocation callbacks allocates and "
        "frees through them correctly (valid alignment and scope, from within Vulkan commands, all freed)",
        "vkCreatePrivateDataSlot vkDestroyPrivateDataSlot vkSetPrivateData vkGetPrivateData vkCreateFence "
        "vkDestroyFence")
{
    struct ct_allocator a;
    struct ct_env e;
    VkPrivateDataSlotCreateInfo info;
    VkPrivateDataSlot slot = VK_NULL_HANDLE;
    VkFenceCreateInfo fi;
    VkFence fences[100];
    VkResult r;
    uint32_t i;
    pt_init(t, &e, 0, NULL);
    ct_allocator_init(&a, -1);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO;
    CT_CALL_VK(&a, r, vkCreatePrivateDataSlot(e.device, &info, &a.cb, &slot));
    CT_REQUIRE_VK(t, r);
    memset(&fi, 0, sizeof(fi));
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    for (i = 0; i < 100; i++) {
        CT_REQUIRE_VK(t, vkCreateFence(e.device, &fi, NULL, &fences[i]));
        CT_CALL_VK(&a, r, vkSetPrivateData(e.device, VK_OBJECT_TYPE_FENCE, PT_HANDLE(fences[i]), slot, i + 1));
        CT_CHECK_VK(t, r, VK_SUCCESS);
    }
    for (i = 0; i < 100; i++) {
        uint64_t v = 0;
        CT_CALL(&a, vkGetPrivateData(e.device, VK_OBJECT_TYPE_FENCE, PT_HANDLE(fences[i]), slot, &v));
        CT_CHECK(t, v == i + 1);
    }
    CT_CALL(&a, vkDestroyPrivateDataSlot(e.device, slot, &a.cb));
    for (i = 0; i < 100; i++)
        vkDestroyFence(e.device, fences[i], NULL);
    ct_allocator_check(t, &a, "private data slot");
    CT_CHECKF(t, a.live == 0, "%ld allocations not freed", a.live);
    ct_allocator_fini(&a);
    ct_env_fini(&e);
}

CT_TEST(private_data_extension,
        "Private Data with VK_EXT_private_data (promoted to Vulkan 1.3): on an apiVersion 1.0 instance with "
        "VK_KHR_get_physical_device_properties2 (the extension's dependency below Vulkan 1.1), privateData is "
        "supported (Features > Feature Requirements: required with the extension), a device with the extension and "
        "VkPhysicalDevicePrivateDataFeaturesEXT::privateData can be created, vkGetDeviceProcAddr returns its "
        "commands (Initialization > Command Function Pointers: enabled extension commands), and they store data",
        "vkGetPhysicalDeviceFeatures2KHR vkCreateDevice vkCreatePrivateDataSlotEXT vkSetPrivateDataEXT "
        "vkGetPrivateDataEXT vkDestroyPrivateDataSlotEXT vkDestroyDevice")
{
    const char *instance_exts[] = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };
    const char *device_exts[] = { VK_EXT_PRIVATE_DATA_EXTENSION_NAME };
    PFN_vkGetPhysicalDeviceFeatures2KHR get_features;
    PFN_vkCreatePrivateDataSlotEXT create;
    PFN_vkDestroyPrivateDataSlotEXT destroy;
    PFN_vkSetPrivateDataEXT set;
    PFN_vkGetPrivateDataEXT get;
    VkPhysicalDevicePrivateDataFeaturesEXT private_data;
    VkPhysicalDeviceFeatures2KHR features;
    VkPrivateDataSlotCreateInfoEXT info;
    VkPrivateDataSlotEXT slot = VK_NULL_HANDLE;
    struct ct_env e;
    uint64_t v;

    if (!ct_instance_extension_supported(instance_exts[0]))
        ct_skip(t, "VK_KHR_get_physical_device_properties2 is not offered");
    ct_env_init_ex(t, &e, VK_API_VERSION_1_0, instance_exts, 1);
    if (!ct_device_extension_supported(e.pd, device_exts[0]))
        ct_skip(t, "VK_EXT_private_data is not offered");
    get_features =
        (PFN_vkGetPhysicalDeviceFeatures2KHR)vkGetInstanceProcAddr(e.instance, "vkGetPhysicalDeviceFeatures2KHR");
    CT_REQUIREF(t, get_features != NULL, "vkGetPhysicalDeviceFeatures2KHR is NULL with its extension enabled");
    memset(&private_data, 0, sizeof(private_data));
    private_data.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES_EXT;
    memset(&features, 0, sizeof(features));
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR;
    features.pNext = &private_data;
    get_features(e.pd, &features);
    CT_REQUIREF(t, private_data.privateData == VK_TRUE, "privateData is %u with VK_EXT_private_data offered",
                private_data.privateData);
    ct_env_device_ex(t, &e, &private_data, device_exts, 1, NULL);

    create = (PFN_vkCreatePrivateDataSlotEXT)vkGetDeviceProcAddr(e.device, "vkCreatePrivateDataSlotEXT");
    destroy = (PFN_vkDestroyPrivateDataSlotEXT)vkGetDeviceProcAddr(e.device, "vkDestroyPrivateDataSlotEXT");
    set = (PFN_vkSetPrivateDataEXT)vkGetDeviceProcAddr(e.device, "vkSetPrivateDataEXT");
    get = (PFN_vkGetPrivateDataEXT)vkGetDeviceProcAddr(e.device, "vkGetPrivateDataEXT");
    CT_REQUIREF(t, create && destroy && set && get, "a command of VK_EXT_private_data is NULL with it enabled");
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO_EXT;
    CT_REQUIRE_VK(t, create(e.device, &info, NULL, &slot));
    CT_CHECK_VK(t, set(e.device, VK_OBJECT_TYPE_DEVICE, PT_HANDLE(e.device), slot, 0x123456789ull), VK_SUCCESS);
    v = 0;
    get(e.device, VK_OBJECT_TYPE_DEVICE, PT_HANDLE(e.device), slot, &v);
    CT_CHECK(t, v == 0x123456789ull);
    v = 1;
    get(e.device, VK_OBJECT_TYPE_QUEUE, PT_HANDLE(e.queue), slot, &v);
    CT_CHECKF(t, v == 0, "data never set reads %llu, not 0", (unsigned long long)v);
    destroy(e.device, slot, NULL);
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(private_data_slots),    CT_REF(private_data_slot_destroy_null), CT_REF(private_data_objects),
    CT_REF(private_data_new_objects), CT_REF(private_data_many),         CT_REF(private_data_reserved),
    CT_REF(private_data_allocator), CT_REF(private_data_extension),
};

CT_SUITE(privatedata, tests)
