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

/* The driver's identification (VkPhysicalDeviceDriverProperties):
 * conformanceVersion is the Khronos CTS release PlasmaVK was last tested
 * against with every must-pass case passing but driver_id_match, which
 * needs a driver ID registered with Khronos (driverID is still 0). */
static void test_driver_properties(void)
{
    VkPhysicalDeviceDriverProperties drv;
    VkPhysicalDeviceProperties2 props;

    memset(&drv, 0, sizeof(drv));
    drv.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    memset(&props, 0, sizeof(props));
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &drv;
    vkGetPhysicalDeviceProperties2(physical_device, &props);
    CHECK(strcmp(drv.driverName, "PlasmaVK") == 0);
    CHECK(drv.conformanceVersion.major == 1 && drv.conformanceVersion.minor == 4 &&
          drv.conformanceVersion.subminor == 7 && drv.conformanceVersion.patch == 0);
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

/* An internal function of a blit library (NULL: none). */
static id blit_function(id library, const char *name, const char *suffix)
{
    char full[64];
    void *pool = objc_autoreleasePoolPush();
    id f;
    snprintf(full, sizeof(full), "%s%s", name, suffix);
    f = pvk_MTLLibrary_newFunctionWithName(library, pvk_NSString_stringWithUTF8String(full));
    objc_autoreleasePoolPop(pool);
    return f;
}

/* The internal shaders are compiled on the device's own thread, without any
 * use (white box: pvk_device_warm), also by two devices at once (the blit
 * libraries' locks are global) and with a first use meanwhile; a device
 * destroyed at once stops it. Each blit function (also layered) and kernel
 * is in the library it is looked up in. */
static void test_internal_shaders_ahead(void)
{
    VkResult result;
    VkDevice devices[2], quick;
    unsigned i, k;
    id f;

    for (k = 0; k < 2; k++) {
        devices[k] = create_device(NULL, 0, NULL, &result);
        REQUIRE(devices[k]);
    }
    CHECK(pvk_blit_library((struct pvk_device *)devices[1], PVK_BLIT_LIB_CCLEAR) != NULL);
    for (k = 0; k < 2; k++) {
        struct pvk_device *d = (struct pvk_device *)devices[k];
        CHECK(d->warm_running);
        pthread_join(d->warm_thread, NULL); /* (it ends once all are made) */
        d->warm_running = false;
        CHECK(d->fill_pipeline && d->query_copy_pipeline && d->ds_copy_library);
        for (i = 0; i < PVK_PP_KERNEL_COUNT; i++)
            CHECK(d->prepass_pipelines[i] != NULL);
        for (i = 0; i < PVK_BLIT_LIB_COUNT; i++)
            REQUIRE(d->blit_libraries[i] != NULL);
        for (i = 0; i < PVK_BLIT_FUNCTION_COUNT; i++) {
            CHECK((f = blit_function(d->blit_libraries[pvk_blit_function_library(i)], pvk_blit_function_names[i], "")));
            pvk_release(f);
            if (pvk_blit_reads_layer(i)) {
                CHECK((f = blit_function(d->blit_libraries[pvk_blit_function_library(i)], pvk_blit_function_names[i],
                                         "_l")));
                pvk_release(f);
            }
        }
        for (i = 0; i < PVK_ARRAY_SIZE(pvk_blit_kernel_names); i++) {
            CHECK((f = blit_function(d->blit_libraries[pvk_blit_kernel_library(i)], pvk_blit_kernel_names[i], "")));
            pvk_release(f);
        }
    }
    for (k = 0; k < 2; k++)
        vkDestroyDevice(devices[k], NULL);
    for (i = 0; i < 4; i++) {
        quick = create_device(NULL, 0, NULL, &result);
        REQUIRE(quick);
        vkDestroyDevice(quick, NULL);
    }
}

/* An allocator that counts the calls made on threads other than the one
 * that made it, and the allocations still live. */
struct thread_allocator {
    VkAllocationCallbacks cb;
    pthread_t thread;
    int foreign; /* atomic */
    long live;   /* atomic */
};

static void thread_allocator_note(struct thread_allocator *a)
{
    if (!pthread_equal(pthread_self(), a->thread))
        __atomic_add_fetch(&a->foreign, 1, __ATOMIC_RELAXED);
}

static void *VKAPI_PTR thread_alloc(void *user, size_t size, size_t alignment, VkSystemAllocationScope scope)
{
    struct thread_allocator *a = user;
    void *p = NULL;
    (void)scope;
    thread_allocator_note(a);
    if (posix_memalign(&p, alignment < sizeof(void *) ? sizeof(void *) : alignment, size) != 0)
        return NULL;
    __atomic_add_fetch(&a->live, 1, __ATOMIC_RELAXED);
    return p;
}

static void *VKAPI_PTR thread_realloc(void *user, void *original, size_t size, size_t alignment,
                                      VkSystemAllocationScope scope)
{
    struct thread_allocator *a = user;
    void *p;
    (void)alignment;
    (void)scope;
    thread_allocator_note(a);
    if (!original)
        return thread_alloc(user, size, alignment, scope);
    if (!size) {
        free(original);
        __atomic_sub_fetch(&a->live, 1, __ATOMIC_RELAXED);
        return NULL;
    }
    p = realloc(original, size); /* (16-byte aligned: all PlasmaVK asks) */
    return p;
}

static void VKAPI_PTR thread_free(void *user, void *memory)
{
    struct thread_allocator *a = user;
    thread_allocator_note(a);
    if (memory) {
        free(memory);
        __atomic_sub_fetch(&a->live, 1, __ATOMIC_RELAXED);
    }
}

/* A message longer than pvk_vmessage's stack buffer, as a lost device's
 * (pvk_commit_feedback, on Metal's thread) with a long error description. */
static void *long_message(void *device)
{
    char text[2048];
    memset(text, 'x', sizeof(text) - 1);
    text[sizeof(text) - 1] = '\0';
    pvk_logd(device, PVK_LOG_DEBUG, "PlasmaVK-test", VK_OBJECT_TYPE_DEVICE, 0, "%s", text);
    return NULL;
}

/* The device's own threads and Metal's never call the application's
 * allocator, only the thread running a command may (Memory Allocation >
 * Host Memory): the internal shaders compiled ahead (white box:
 * pvk_device_warm, waited for before vkDestroyDevice stops it) and messages
 * sent from other threads take system memory. */
static void test_internal_threads_allocator(void)
{
    float priority = 1.0f;
    VkInstanceCreateInfo instance_info;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo info;
    VkInstance own_instance = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    struct thread_allocator a;
    struct pvk_device *d;
    pthread_t sender;
    uint32_t count = 1;

    memset(&a, 0, sizeof(a));
    a.cb.pUserData = &a;
    a.cb.pfnAllocation = thread_alloc;
    a.cb.pfnReallocation = thread_realloc;
    a.cb.pfnFree = thread_free;
    a.thread = pthread_self();
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    REQUIRE_VK(vkCreateInstance(&instance_info, &a.cb, &own_instance));
    vkEnumeratePhysicalDevices(own_instance, &count, &pd);
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue_info;
    if (pd && vkCreateDevice(pd, &info, &a.cb, &device) == VK_SUCCESS) {
        d = (struct pvk_device *)device;
        CHECK(d->warm_running);
        if (d->warm_running) {
            pthread_join(d->warm_thread, NULL); /* (it ends once all are made) */
            d->warm_running = false;
        }
        CHECK(d->blit_libraries[PVK_BLIT_LIB_PRESENT] && d->fill_pipeline && d->ds_copy_library);
        CHECK(pthread_create(&sender, NULL, long_message, d) == 0 && pthread_join(sender, NULL) == 0);
        vkDestroyDevice(device, &a.cb);
    } else {
        PVK_TEST_FAIL("no device made with allocation callbacks");
    }
    vkDestroyInstance(own_instance, &a.cb);
    if (a.foreign)
        PVK_TEST_FAIL("%d allocator calls on the device's own threads", a.foreign);
    CHECK(a.live == 0);
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

/* The queue family reports one global priority, MEDIUM: with
 * globalPriorityQuery, a queue asking for another fails device creation;
 * without it, the priority is a hint. */
static void test_global_priority(void)
{
    static const VkQueueGlobalPriority asked[] = { VK_QUEUE_GLOBAL_PRIORITY_LOW, VK_QUEUE_GLOBAL_PRIORITY_MEDIUM,
                                                   VK_QUEUE_GLOBAL_PRIORITY_HIGH };
    float priority = 1.0f;
    VkQueueFamilyGlobalPriorityProperties gp;
    VkQueueFamilyProperties2 family;
    VkPhysicalDeviceGlobalPriorityQueryFeatures query;
    VkDeviceQueueGlobalPriorityCreateInfo global;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo info;
    VkDevice device;
    uint32_t count = 1, i, q;

    memset(&gp, 0, sizeof(gp));
    gp.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES;
    memset(&family, 0, sizeof(family));
    family.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
    family.pNext = &gp;
    vkGetPhysicalDeviceQueueFamilyProperties2(physical_device, &count, &family);
    CHECK(gp.priorityCount == 1 && gp.priorities[0] == VK_QUEUE_GLOBAL_PRIORITY_MEDIUM);

    for (q = 0; q < 2; q++)
        for (i = 0; i < sizeof(asked) / sizeof(asked[0]); i++) {
            VkResult want = q && asked[i] != VK_QUEUE_GLOBAL_PRIORITY_MEDIUM ? VK_ERROR_INITIALIZATION_FAILED
                                                                             : VK_SUCCESS;
            memset(&query, 0, sizeof(query));
            query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES;
            query.globalPriorityQuery = q ? VK_TRUE : VK_FALSE;
            memset(&global, 0, sizeof(global));
            global.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO;
            global.globalPriority = asked[i];
            memset(&queue_info, 0, sizeof(queue_info));
            queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_info.pNext = &global;
            queue_info.queueCount = 1;
            queue_info.pQueuePriorities = &priority;
            memset(&info, 0, sizeof(info));
            info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            info.pNext = &query;
            info.queueCreateInfoCount = 1;
            info.pQueueCreateInfos = &queue_info;
            device = VK_NULL_HANDLE;
            CHECK_VK(vkCreateDevice(physical_device, &info, NULL, &device), want);
            if (device)
                vkDestroyDevice(device, NULL);
        }
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
    RUN(test_driver_properties);
    RUN(test_device_and_queues);
    RUN(test_internal_shaders_ahead);
    RUN(test_internal_threads_allocator);
    RUN(test_promoted_extensions);
    RUN(test_promoted_extension_features);
    RUN(test_promoted_extension_commands);
    RUN(test_unsupported_feature);
    RUN(test_global_priority);
    vkDestroyInstance(instance, NULL);
    return TEST_RESULT();
}
