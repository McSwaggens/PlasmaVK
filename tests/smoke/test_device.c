#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <string.h>

#include "../pvk_test.h"

static VkInstance instance;
static VkPhysicalDevice physical_device;

static void setup(void)
{
    /* VK_KHR_swapchain, enabled below, needs VK_KHR_surface on the instance. */
    const char *extensions[] = { VK_KHR_SURFACE_EXTENSION_NAME };
    VkInstanceCreateInfo info;
    uint32_t count = 1;

    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.enabledExtensionCount = 1;
    info.ppEnabledExtensionNames = extensions;
    REQUIRE_VK(vkCreateInstance(&info, NULL, &instance));
    REQUIRE_VK(vkEnumeratePhysicalDevices(instance, &count, &physical_device));
    REQUIRE(count == 1);
}

static void test_queue_families(void)
{
    VkQueueFamilyProperties props[4];
    VkPhysicalDeviceFeatures features;
    VkQueueFlags sparse;
    uint32_t count = 0;

    vkGetPhysicalDeviceFeatures(physical_device, &features);
    sparse = features.sparseBinding ? VK_QUEUE_SPARSE_BINDING_BIT : 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, NULL);
    REQUIRE(count == 2);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, props);
    CHECK(props[0].queueFlags & VK_QUEUE_GRAPHICS_BIT);
    CHECK(props[0].queueFlags & VK_QUEUE_COMPUTE_BIT);
    CHECK(props[0].queueCount >= 1);
    /* Family 1: async compute, transfer implied (no explicit TRANSFER_BIT).
     * Both families bind sparse resources where the sparse features are. */
    CHECK(props[1].queueFlags == (VK_QUEUE_COMPUTE_BIT | sparse));
    CHECK((props[0].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) == sparse);
    CHECK(props[1].queueCount >= 1);
}

/* A queue of the compute family runs work like any other. */
static void test_compute_family_queue(void)
{
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info[2];
    VkDeviceCreateInfo info;
    VkDevice device;
    VkQueue graphics = VK_NULL_HANDLE, compute = VK_NULL_HANDLE;

    memset(queue_info, 0, sizeof(queue_info));
    queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info[0].queueFamilyIndex = 0;
    queue_info[0].queueCount = 1;
    queue_info[0].pQueuePriorities = &priority;
    queue_info[1] = queue_info[0];
    queue_info[1].queueFamilyIndex = 1;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 2;
    info.pQueueCreateInfos = queue_info;

    REQUIRE_VK(vkCreateDevice(physical_device, &info, NULL, &device));
    vkGetDeviceQueue(device, 0, 0, &graphics);
    vkGetDeviceQueue(device, 1, 0, &compute);
    CHECK(graphics && compute && graphics != compute);
    CHECK_VK(vkQueueWaitIdle(compute), VK_SUCCESS);
    vkDestroyDevice(device, NULL);
}

/* The lazily allocated type reserves nothing and is not offered to buffers. */
static void test_lazily_allocated_memory(void)
{
    float priority = 1.0f;
    VkPhysicalDeviceMemoryProperties mem;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo info;
    VkMemoryAllocateInfo alloc;
    VkBufferCreateInfo buffer_info;
    VkMemoryRequirements req;
    VkDevice device;
    VkDeviceMemory memory;
    VkBuffer buffer;
    VkDeviceSize committed = 1;
    uint32_t i, lazy = UINT32_MAX;

    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem);
    for (i = 0; i < mem.memoryTypeCount; i++)
        if (mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)
            lazy = i;
    REQUIRE(lazy != UINT32_MAX);
    CHECK(!(mem.memoryTypes[lazy].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));

    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue_info;
    REQUIRE_VK(vkCreateDevice(physical_device, &info, NULL, &device));

    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = 1u << 20;
    alloc.memoryTypeIndex = lazy;
    CHECK_VK(vkAllocateMemory(device, &alloc, NULL, &memory), VK_SUCCESS);
    vkGetDeviceMemoryCommitment(device, memory, &committed);
    CHECK(committed == 0);
    vkFreeMemory(device, memory, NULL);

    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = 4096;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    REQUIRE_VK(vkCreateBuffer(device, &buffer_info, NULL, &buffer));
    vkGetBufferMemoryRequirements(device, buffer, &req);
    CHECK(!(req.memoryTypeBits & (1u << lazy)));
    CHECK(req.memoryTypeBits != 0);
    vkDestroyBuffer(device, buffer, NULL);
    vkDestroyDevice(device, NULL);
}

static void test_memory_properties(void)
{
    VkPhysicalDeviceMemoryProperties mem;
    VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t i;
    int found_host = 0;

    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem);
    REQUIRE(mem.memoryHeapCount >= 1);
    CHECK(mem.memoryHeaps[0].size > 0);
    for (i = 0; i < mem.memoryTypeCount; i++)
        if ((mem.memoryTypes[i].propertyFlags & host) == host)
            found_host = 1;
    CHECK(found_host); /* required by the spec */
}

static void test_device_and_queues(void)
{
    float priorities[2] = { 1.0f, 0.5f };
    const char *extensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo info;
    VkDevice device;
    VkQueue q0 = VK_NULL_HANDLE, q1 = VK_NULL_HANDLE, missing = VK_NULL_HANDLE;

    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 2;
    queue_info.pQueuePriorities = priorities;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue_info;
    info.enabledExtensionCount = 1;
    info.ppEnabledExtensionNames = extensions;

    REQUIRE_VK(vkCreateDevice(physical_device, &info, NULL, &device));
    vkGetDeviceQueue(device, 0, 0, &q0);
    vkGetDeviceQueue(device, 0, 1, &q1);
    vkGetDeviceQueue(device, 0, 2, &missing);
    CHECK(q0 && q1 && q0 != q1);
    CHECK(missing == VK_NULL_HANDLE);

    CHECK(vkGetDeviceProcAddr(device, "vkCmdDraw") == (PFN_vkVoidFunction)vkCmdDraw);
    CHECK(vkGetDeviceProcAddr(device, "vkCreateInstance") == NULL);
    CHECK(vkGetDeviceProcAddr(device, "vkGetPhysicalDeviceProperties") == NULL);
    vkDestroyDevice(device, NULL);
}

/* A device with the extensions named and the feature chain; VK_NULL_HANDLE
 * if vkCreateDevice fails (*result says why). */
static VkDevice create_device(const char *const *extensions, uint32_t count, const void *pnext, VkResult *result)
{
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo info;
    VkDevice device = VK_NULL_HANDLE;

    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.pNext = pnext;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue_info;
    info.enabledExtensionCount = count;
    info.ppEnabledExtensionNames = extensions;
    *result = vkCreateDevice(physical_device, &info, NULL, &device);
    return *result == VK_SUCCESS ? device : VK_NULL_HANDLE;
}

static int listed(const VkExtensionProperties *props, uint32_t count, const char *name)
{
    uint32_t i;
    for (i = 0; i < count; i++)
        if (strcmp(props[i].extensionName, name) == 0)
            return 1;
    return 0;
}

/* Every extension promoted to Vulkan 1.1-1.4 is listed, unless it needs a
 * feature the device lacks, and all of them can be enabled at once. */
static void test_promoted_extensions(void)
{
    static const char *const always[] = {
        "VK_KHR_16bit_storage", "VK_KHR_bind_memory2", "VK_KHR_dedicated_allocation", "VK_KHR_descriptor_update_template",
        "VK_KHR_device_group", "VK_KHR_external_memory", "VK_KHR_maintenance1", "VK_KHR_maintenance2",
        "VK_KHR_maintenance3", "VK_KHR_multiview", "VK_KHR_sampler_ycbcr_conversion", "VK_KHR_variable_pointers",
        "VK_KHR_8bit_storage", "VK_KHR_buffer_device_address", "VK_KHR_create_renderpass2",
        "VK_KHR_shader_float16_int8", "VK_KHR_spirv_1_4", "VK_KHR_timeline_semaphore", "VK_EXT_host_query_reset",
        "VK_EXT_scalar_block_layout", "VK_EXT_4444_formats", "VK_EXT_extended_dynamic_state",
        "VK_EXT_extended_dynamic_state2", "VK_EXT_private_data", "VK_EXT_texel_buffer_alignment",
        "VK_EXT_tooling_info", "VK_KHR_copy_commands2", "VK_KHR_dynamic_rendering", "VK_KHR_maintenance4",
        "VK_KHR_synchronization2", "VK_EXT_host_image_copy", "VK_EXT_pipeline_robustness",
        "VK_KHR_dynamic_rendering_local_read", "VK_KHR_global_priority", "VK_KHR_index_type_uint8",
        "VK_KHR_line_rasterization", "VK_KHR_load_store_op_none", "VK_KHR_maintenance5", "VK_KHR_maintenance6",
        "VK_KHR_map_memory2", "VK_KHR_push_descriptor", "VK_KHR_vertex_attribute_divisor", "VK_KHR_swapchain",
    };
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan14Features v14;
    VkPhysicalDeviceFeatures2 features;
    VkExtensionProperties props[PVK_DEVICE_EXTENSION_COUNT];
    const char *names[PVK_DEVICE_EXTENSION_COUNT];
    uint32_t count = PVK_DEVICE_EXTENSION_COUNT, i;
    VkResult result;
    VkDevice device;

    memset(&v14, 0, sizeof(v14));
    v14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.pNext = &v14;
    memset(&features, 0, sizeof(features));
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &v12;
    vkGetPhysicalDeviceFeatures2(physical_device, &features);

    REQUIRE_VK(vkEnumerateDeviceExtensionProperties(physical_device, NULL, &count, props));
    for (i = 0; i < PVK_ARRAY_SIZE(always); i++)
        if (!listed(props, count, always[i]))
            PVK_TEST_FAIL("%s is not listed", always[i]);
    CHECK(listed(props, count, "VK_KHR_shader_atomic_int64") == (v12.shaderBufferInt64Atomics == VK_TRUE));
    CHECK(listed(props, count, "VK_EXT_sampler_filter_minmax") == (v12.samplerFilterMinmax == VK_TRUE));
    CHECK(listed(props, count, "VK_EXT_pipeline_protected_access") == (v14.pipelineProtectedAccess == VK_TRUE));
    CHECK(!listed(props, count, "VK_EXT_ycbcr_2plane_444_formats"));
    CHECK(!listed(props, count, "VK_KHR_shader_float16_int8") == !(v12.shaderFloat16 || v12.shaderInt8));

    for (i = 0; i < count; i++)
        names[i] = props[i].extensionName;
    device = create_device(names, count, NULL, &result);
    CHECK_VK(result, VK_SUCCESS);
    if (device)
        vkDestroyDevice(device, NULL);
}

/* The extensions' own feature structs, where core has no members: what
 * PlasmaVK has is reported, and vkCreateDevice refuses the rest. */
static void test_promoted_extension_features(void)
{
    const char *extensions[] = { "VK_EXT_extended_dynamic_state2" };
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT eds;
    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT eds2;
    VkPhysicalDeviceTexelBufferAlignmentFeaturesEXT tba;
    VkPhysicalDevice4444FormatsFeaturesEXT f4444;
    VkPhysicalDeviceFeatures2 features;
    VkResult result;
    VkDevice device;

    memset(&eds, 0xFF, sizeof(eds));
    eds.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT;
    eds.pNext = NULL;
    memset(&eds2, 0xFF, sizeof(eds2));
    eds2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT;
    eds2.pNext = &eds;
    memset(&tba, 0xFF, sizeof(tba));
    tba.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_FEATURES_EXT;
    tba.pNext = &eds2;
    memset(&f4444, 0xFF, sizeof(f4444));
    f4444.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_4444_FORMATS_FEATURES_EXT;
    f4444.pNext = &tba;
    memset(&features, 0, sizeof(features));
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &f4444;
    vkGetPhysicalDeviceFeatures2(physical_device, &features);
    CHECK(eds.extendedDynamicState == VK_TRUE);
    CHECK(eds2.extendedDynamicState2 == VK_TRUE);
    CHECK(eds2.extendedDynamicState2LogicOp == VK_FALSE);
    CHECK(eds2.extendedDynamicState2PatchControlPoints == VK_FALSE);
    CHECK(tba.texelBufferAlignment == VK_TRUE);
    CHECK(f4444.formatA4R4G4B4 == VK_TRUE && f4444.formatA4B4G4R4 == VK_TRUE);

    /* Asking for the supported ones works, for a logic op state not. */
    eds2.pNext = NULL;
    eds2.extendedDynamicState2LogicOp = VK_FALSE;
    device = create_device(extensions, 1, &eds2, &result);
    CHECK_VK(result, VK_SUCCESS);
    if (device) {
        CHECK(vkGetDeviceProcAddr(device, "vkCmdSetLogicOpEXT") == (PFN_vkVoidFunction)vkCmdSetLogicOpEXT);
        CHECK(vkGetDeviceProcAddr(device, "vkCmdSetPatchControlPointsEXT") != NULL);
        CHECK(vkGetDeviceProcAddr(device, "vkCmdSetPrimitiveRestartEnableEXT") ==
              (PFN_vkVoidFunction)vkCmdSetPrimitiveRestartEnable);
        vkDestroyDevice(device, NULL);
    }
    eds2.extendedDynamicState2LogicOp = VK_TRUE;
    CHECK(create_device(extensions, 1, &eds2, &result) == VK_NULL_HANDLE);
    CHECK_VK(result, VK_ERROR_FEATURE_NOT_PRESENT);
}

/* vkGetDeviceProcAddr returns the KHR/EXT names of core commands only when
 * their extension is enabled, and the commands an extension adds with
 * another one (or with a core version: the instance asked for 1.0) only
 * with that one. */
static void test_promoted_extension_commands(void)
{
    static const struct {
        const char *extensions[2];
        const char *command;
        PFN_vkVoidFunction fn; /* NULL: not returned */
    } cases[] = {
        { { NULL }, "vkCreatePrivateDataSlotEXT", NULL },
        { { "VK_EXT_private_data" }, "vkCreatePrivateDataSlotEXT", (PFN_vkVoidFunction)vkCreatePrivateDataSlot },
        { { "VK_EXT_private_data" }, "vkCreatePrivateDataSlot", NULL }, /* Vulkan 1.3, beyond apiVersion 1.0 */
        { { "VK_KHR_dynamic_rendering" }, "vkCmdBeginRenderingKHR", (PFN_vkVoidFunction)vkCmdBeginRendering },
        { { "VK_KHR_synchronization2" }, "vkQueueSubmit2KHR", (PFN_vkVoidFunction)vkQueueSubmit2 },
        { { "VK_KHR_synchronization2" }, "vkCmdBeginRenderingKHR", NULL },
        { { "VK_KHR_maintenance1" }, "vkTrimCommandPoolKHR", (PFN_vkVoidFunction)vkTrimCommandPool },
        { { "VK_EXT_host_query_reset" }, "vkResetQueryPoolEXT", (PFN_vkVoidFunction)vkResetQueryPool },
        { { "VK_KHR_push_descriptor" }, "vkCmdPushDescriptorSetKHR", (PFN_vkVoidFunction)vkCmdPushDescriptorSet },
        /* VK_KHR_push_descriptor's template command needs Vulkan 1.1 or VK_KHR_descriptor_update_template */
        { { "VK_KHR_push_descriptor" }, "vkCmdPushDescriptorSetWithTemplateKHR", NULL },
        { { "VK_KHR_push_descriptor", "VK_KHR_descriptor_update_template" }, "vkCmdPushDescriptorSetWithTemplateKHR",
          (PFN_vkVoidFunction)vkCmdPushDescriptorSetWithTemplate },
        /* VK_KHR_maintenance6's push descriptor commands need VK_KHR_push_descriptor */
        { { "VK_KHR_maintenance6" }, "vkCmdPushConstants2KHR", (PFN_vkVoidFunction)vkCmdPushConstants2 },
        { { "VK_KHR_maintenance6" }, "vkCmdPushDescriptorSet2KHR", NULL },
        { { "VK_KHR_maintenance6", "VK_KHR_push_descriptor" }, "vkCmdPushDescriptorSet2KHR",
          (PFN_vkVoidFunction)vkCmdPushDescriptorSet2 },
        /* Device group commands: VK_KHR_swapchain's need Vulkan 1.1;
         * VK_KHR_device_group's VK_KHR_surface (enabled) or VK_KHR_swapchain */
        { { "VK_KHR_swapchain" }, "vkAcquireNextImage2KHR", NULL },
        { { "VK_KHR_swapchain" }, "vkGetDeviceGroupPresentCapabilitiesKHR", NULL },
        { { "VK_KHR_device_group" }, "vkGetDeviceGroupPresentCapabilitiesKHR",
          (PFN_vkVoidFunction)vkGetDeviceGroupPresentCapabilitiesKHR },
        { { "VK_KHR_device_group" }, "vkAcquireNextImage2KHR", NULL },
        { { "VK_KHR_device_group", "VK_KHR_swapchain" }, "vkAcquireNextImage2KHR",
          (PFN_vkVoidFunction)vkAcquireNextImage2KHR },
        { { "VK_KHR_device_group" }, "vkCmdDispatchBaseKHR", (PFN_vkVoidFunction)vkCmdDispatchBase },
        /* Instance-level commands: never */
        { { "VK_EXT_tooling_info" }, "vkGetPhysicalDeviceToolPropertiesEXT", NULL },
    };
    size_t i;

    for (i = 0; i < PVK_ARRAY_SIZE(cases); i++) {
        uint32_t n = cases[i].extensions[1] ? 2 : cases[i].extensions[0] ? 1 : 0;
        VkResult result;
        VkDevice device = create_device(cases[i].extensions, n, NULL, &result);
        if (!device) {
            PVK_TEST_FAIL("vkCreateDevice with %s: %d", n ? cases[i].extensions[0] : "no extension", (int)result);
            continue;
        }
        if (vkGetDeviceProcAddr(device, cases[i].command) != cases[i].fn)
            PVK_TEST_FAIL("vkGetDeviceProcAddr(\"%s\") with %s%s%s", cases[i].command,
                          n ? cases[i].extensions[0] : "no extension", n > 1 ? " and " : "",
                          n > 1 ? cases[i].extensions[1] : "");
        vkDestroyDevice(device, NULL);
    }
    /* vkGetInstanceProcAddr: the instance extension aliases only when enabled
     * (setup enabled VK_KHR_surface alone), device extensions' when available. */
    CHECK(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2KHR") == NULL);
    CHECK(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceToolPropertiesEXT") ==
          (PFN_vkVoidFunction)vkGetPhysicalDeviceToolProperties);
    CHECK(vkGetInstanceProcAddr(instance, "vkCmdPushDescriptorSet2KHR") == (PFN_vkVoidFunction)vkCmdPushDescriptorSet2);
}

static void test_unsupported_feature(void)
{
    float priority = 1.0f;
    VkPhysicalDeviceFeatures features;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo info;
    VkDevice device;

    vkGetPhysicalDeviceFeatures(physical_device, &features);
    if (features.geometryShader)
        return; /* nothing to test once geometry shaders land */
    features.geometryShader = VK_TRUE;

    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue_info;
    info.pEnabledFeatures = &features;
    CHECK_VK(vkCreateDevice(physical_device, &info, NULL, &device), VK_ERROR_FEATURE_NOT_PRESENT);
}

int main(void)
{
    setup();
    if (!physical_device)
        return 1;
    RUN(test_queue_families);
    RUN(test_compute_family_queue);
    RUN(test_lazily_allocated_memory);
    RUN(test_memory_properties);
    RUN(test_device_and_queues);
    RUN(test_promoted_extensions);
    RUN(test_promoted_extension_features);
    RUN(test_promoted_extension_commands);
    RUN(test_unsupported_feature);
    vkDestroyInstance(instance, NULL);
    return TEST_RESULT();
}
