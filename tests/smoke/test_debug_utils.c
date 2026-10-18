/* VK_EXT_debug_utils inside PlasmaVK ("Debug messages" after section 6,
 * sections 8, 9, 15 and 16): PlasmaVK's own diagnostics reach messengers,
 * also the ones chained to VkInstanceCreateInfo; object names become Metal
 * labels; label regions and the debug layer's groups become Metal debug
 * groups, observed by intercepting the Metal methods (the Objective-C
 * runtime lets a C program replace a method's implementation); the debug
 * layer is listed, enabled and reported. */
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <stdlib.h>
#include <string.h>

#include "../pvk_test.h"

/* ---- Messages ---- */

#define GOT_MAX 32

struct got {
    int count;
    char ids[GOT_MAX][96];
    char text[GOT_MAX][256];
    VkDebugUtilsMessageSeverityFlagBitsEXT severity[GOT_MAX];
    VkDebugUtilsMessageTypeFlagsEXT types[GOT_MAX];
    int32_t number[GOT_MAX];
    uint32_t objects[GOT_MAX];
    uint64_t object0[GOT_MAX];
    char name0[GOT_MAX][64];
};

static VkBool32 VKAPI_PTR record(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT types,
                                 const VkDebugUtilsMessengerCallbackDataEXT *data, void *user)
{
    struct got *g = user;
    int i = g->count++;
    if (i >= GOT_MAX)
        return VK_FALSE;
    snprintf(g->ids[i], sizeof(g->ids[i]), "%s", data->pMessageIdName ? data->pMessageIdName : "");
    snprintf(g->text[i], sizeof(g->text[i]), "%s", data->pMessage ? data->pMessage : "");
    g->severity[i] = severity;
    g->types[i] = types;
    g->number[i] = data->messageIdNumber;
    g->objects[i] = data->objectCount;
    g->object0[i] = data->objectCount ? data->pObjects[0].objectHandle : 0;
    snprintf(g->name0[i], sizeof(g->name0[i]), "%s",
             data->objectCount && data->pObjects[0].pObjectName ? data->pObjects[0].pObjectName : "");
    return VK_FALSE;
}

static int find(const struct got *g, const char *id)
{
    int i;
    for (i = 0; i < g->count && i < GOT_MAX; i++)
        if (strcmp(g->ids[i], id) == 0)
            return i;
    return -1;
}

static VkDebugUtilsMessengerCreateInfoEXT messenger_info(struct got *g, VkDebugUtilsMessageSeverityFlagsEXT severities)
{
    VkDebugUtilsMessengerCreateInfoEXT info;
    memset(g, 0, sizeof(*g));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity = severities;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = record;
    info.pUserData = g;
    return info;
}

#define ALL_SEVERITIES                                                                                           \
    (VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |              \
     VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)

/* An instance with VK_EXT_debug_utils, the debug layer if asked, and pNext. */
static VkResult create_instance(bool layer, const void *pnext, const char *extra_extension, VkInstance *instance)
{
    const char *extensions[2] = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME, extra_extension };
    const char *layers[1] = { "VK_LAYER_PLASMAVK_debug" };
    VkApplicationInfo app;
    VkInstanceCreateInfo info;
    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_4;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pNext = pnext;
    info.pApplicationInfo = &app;
    info.enabledExtensionCount = extra_extension ? 2 : 1;
    info.ppEnabledExtensionNames = extensions;
    info.enabledLayerCount = layer ? 1 : 0;
    info.ppEnabledLayerNames = layers;
    return vkCreateInstance(&info, NULL, instance);
}

static VkResult create_device(VkInstance instance, const VkPhysicalDeviceFeatures *features, VkDevice *device)
{
    VkPhysicalDevice pd;
    uint32_t count = 1;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue;
    VkDeviceCreateInfo info;
    vkEnumeratePhysicalDevices(instance, &count, &pd);
    memset(&queue, 0, sizeof(queue));
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue;
    info.pEnabledFeatures = features;
    return vkCreateDevice(pd, &info, NULL, device);
}

static void set_name(VkDevice device, VkObjectType type, uint64_t handle, const char *name)
{
    VkDebugUtilsObjectNameInfoEXT info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = type;
    info.objectHandle = handle;
    info.pObjectName = name;
    CHECK_VK(vkSetDebugUtilsObjectNameEXT(device, &info), VK_SUCCESS);
}

#define H(x) ((uint64_t)(uintptr_t)(x))

/* Messengers chained to VkInstanceCreateInfo see vkCreateInstance's and
 * vkDestroyInstance's messages, filtered. */
static void test_create_info_messages(void)
{
    VkDebugUtilsMessengerCreateInfoEXT chained, errors, extra;
    VkDebugUtilsMessengerEXT leaked;
    struct got all, only_errors, persistent;
    VkInstance instance = VK_NULL_HANDLE;
    int i;

    chained = messenger_info(&all, ALL_SEVERITIES);
    errors = messenger_info(&only_errors, VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
    chained.pNext = &errors;
    CHECK_VK(create_instance(false, &chained, "VK_KHR_not_a_real_extension", &instance), VK_ERROR_EXTENSION_NOT_PRESENT);
    i = find(&all, "PlasmaVK-vkCreateInstance-extension-not-present");
    REQUIRE(i >= 0);
    CHECK(all.severity[i] == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT);
    CHECK(all.types[i] == VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT);
    CHECK(strstr(all.text[i], "VK_KHR_not_a_real_extension") != NULL);
    CHECK(all.number[i] == pvk_message_number("PlasmaVK-vkCreateInstance-extension-not-present"));
    CHECK(only_errors.count == 0); /* a warning */

    chained = messenger_info(&all, ALL_SEVERITIES);
    {
        const char *layers[] = { "VK_LAYER_PLASMAVK_debug", "VK_LAYER_KHRONOS_validation" };
        VkInstanceCreateInfo info;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pNext = &chained;
        info.enabledLayerCount = 2;
        info.ppEnabledLayerNames = layers;
        CHECK_VK(vkCreateInstance(&info, NULL, &instance), VK_ERROR_LAYER_NOT_PRESENT);
        CHECK(find(&all, "PlasmaVK-vkCreateInstance-layer-not-present") >= 0);
    }

    /* With the debug layer: its announcement in vkCreateInstance, and a leak
     * reported in vkDestroyInstance, both to the chained messenger only. */
    chained = messenger_info(&all, ALL_SEVERITIES);
    chained.pNext = NULL;
    REQUIRE_VK(create_instance(true, &chained, NULL, &instance));
    i = find(&all, "PlasmaVK-debug-layer");
    CHECK(i >= 0 && all.severity[i] == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT);
    extra = messenger_info(&persistent, ALL_SEVERITIES);
    REQUIRE_VK(vkCreateDebugUtilsMessengerEXT(instance, &extra, NULL, &leaked));
    all.count = 0;
    pvk_logi((struct pvk_instance *)instance, PVK_LOG_WARN, "PlasmaVK-smoke", "between create and destroy");
    CHECK(all.count == 0);              /* the chained messenger is not active */
    CHECK(find(&persistent, "PlasmaVK-smoke") >= 0);
    vkDestroyInstance(instance, NULL); /* the leaked messenger: reported, and freed */
    i = find(&all, "VUID-vkDestroyInstance-instance-00629");
    CHECK(i >= 0 && all.types[i] == VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT && all.object0[i] == H(leaked));
}

/* PlasmaVK's diagnostics: an instance's go to its messengers, unowned ones to
 * every instance's, with the severity of their level and named objects. */
static void test_internal_messages(void)
{
    VkDebugUtilsMessengerCreateInfoEXT info;
    VkDebugUtilsMessengerEXT messenger;
    VkPhysicalDeviceFeatures supported, wanted;
    VkPhysicalDevice pd;
    VkBufferCreateInfo bi;
    VkBuffer buffer;
    VkInstance instance;
    VkDevice device = VK_NULL_HANDLE;
    VkBool32 *s = (VkBool32 *)&supported, *w = (VkBool32 *)&wanted;
    struct got g;
    uint32_t count = 1, k;
    int i;

    REQUIRE_VK(create_instance(false, NULL, NULL, &instance));
    info = messenger_info(&g, ALL_SEVERITIES);
    REQUIRE_VK(vkCreateDebugUtilsMessengerEXT(instance, &info, NULL, &messenger));
    vkEnumeratePhysicalDevices(instance, &count, &pd);
    vkGetPhysicalDeviceFeatures(pd, &supported);
    memset(&wanted, 0, sizeof(wanted));
    for (k = 0; k < sizeof(supported) / sizeof(VkBool32) && s[k]; k++)
        ;
    REQUIRE(k < sizeof(supported) / sizeof(VkBool32)); /* some core feature is unsupported */
    w[k] = VK_TRUE;
    CHECK_VK(create_device(instance, &wanted, &device), VK_ERROR_FEATURE_NOT_PRESENT);
    i = find(&g, "PlasmaVK-vkCreateDevice-feature-not-present");
    CHECK(i >= 0 && g.severity[i] == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT);

    g.count = 0;
    pvk_logi(NULL, PVK_LOG_ERROR, "PlasmaVK-smoke-unowned", "an unowned error");
    pvk_logi(NULL, PVK_LOG_DEBUG, "PlasmaVK-smoke-unowned", "unowned debug output");
    i = find(&g, "PlasmaVK-smoke-unowned");
    CHECK(i >= 0 && g.severity[i] == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT &&
          strcmp(g.text[i], "an unowned error") == 0);
    CHECK(g.count == 2 && g.severity[1] == VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT);

    REQUIRE_VK(create_device(instance, NULL, &device));
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = 256;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    REQUIRE_VK(vkCreateBuffer(device, &bi, NULL, &buffer));
    set_name(device, VK_OBJECT_TYPE_BUFFER, H(buffer), "the buffer");
    g.count = 0;
    pvk_vuid((struct pvk_device *)device, "VUID-smoke", VK_OBJECT_TYPE_BUFFER, H(buffer), "about %s", "a buffer");
    REQUIRE(g.count == 1);
    CHECK(strcmp(g.ids[0], "VUID-smoke") == 0 && g.types[0] == VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT);
    CHECK(g.objects[0] == 1 && g.object0[0] == H(buffer) && strcmp(g.name0[0], "the buffer") == 0);
    vkDestroyBuffer(device, buffer, NULL); /* its name goes with it */
    CHECK(!pvk_name_get(H(buffer), g.name0[0], sizeof(g.name0[0])));
    vkDestroyDevice(device, NULL);
    vkDestroyDebugUtilsMessengerEXT(instance, messenger, NULL);
    vkDestroyInstance(instance, NULL);
}

/* A handle with a name or tags, and one with private data, count in their
 * marks' buckets until the entries go: renaming adds nothing, freeing the
 * object or destroying the slot takes them out. pvk_free and pvk_name_get
 * skip the tables (and their locks) for handles whose bucket is 0. */
static void test_marks(void)
{
    VkPrivateDataSlotCreateInfo slot_info;
    VkPhysicalDeviceVulkan13Features v13;
    VkPhysicalDeviceFeatures2 features;
    VkPrivateDataSlot slot;
    VkBufferCreateInfo bi;
    VkBuffer buffer, other;
    VkInstance instance;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice pd;
    VkDeviceQueueCreateInfo queue;
    VkDeviceCreateInfo info;
    unsigned char names, data, other_data;
    uint32_t count = 1;
    float priority = 1.0f;
    char text[8];

    REQUIRE_VK(create_instance(false, NULL, NULL, &instance));
    vkEnumeratePhysicalDevices(instance, &count, &pd);
    memset(&v13, 0, sizeof(v13));
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13.privateData = VK_TRUE;
    memset(&features, 0, sizeof(features));
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &v13;
    memset(&queue, 0, sizeof(queue));
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.pNext = &features;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue;
    REQUIRE_VK(vkCreateDevice(pd, &info, NULL, &device));
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = 256;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    memset(&slot_info, 0, sizeof(slot_info));
    slot_info.sType = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO;
    REQUIRE_VK(vkCreateBuffer(device, &bi, NULL, &buffer));
    REQUIRE_VK(vkCreateBuffer(device, &bi, NULL, &other));
    REQUIRE_VK(vkCreatePrivateDataSlot(device, &slot_info, NULL, &slot));
    names = *pvk_mark(pvk_names_marks, H(buffer));
    data = *pvk_mark(pvk_private_data_marks, H(buffer));

    CHECK(!pvk_name_get(H(buffer), text, sizeof(text)));
    set_name(device, VK_OBJECT_TYPE_BUFFER, H(buffer), "a");
    set_name(device, VK_OBJECT_TYPE_BUFFER, H(buffer), "b");
    CHECK(*pvk_mark(pvk_names_marks, H(buffer)) == names + 1);
    CHECK(pvk_name_get(H(buffer), text, sizeof(text)) && strcmp(text, "b") == 0);
    CHECK_VK(vkSetPrivateData(device, VK_OBJECT_TYPE_BUFFER, H(buffer), slot, 7), VK_SUCCESS);
    CHECK_VK(vkSetPrivateData(device, VK_OBJECT_TYPE_BUFFER, H(buffer), slot, 0), VK_SUCCESS); /* deleted */
    CHECK(*pvk_mark(pvk_private_data_marks, H(buffer)) == data);
    CHECK_VK(vkSetPrivateData(device, VK_OBJECT_TYPE_BUFFER, H(buffer), slot, 7), VK_SUCCESS);
    CHECK(*pvk_mark(pvk_private_data_marks, H(buffer)) == data + 1);
    vkDestroyBuffer(device, buffer, NULL);
    CHECK(*pvk_mark(pvk_names_marks, H(buffer)) == names && *pvk_mark(pvk_private_data_marks, H(buffer)) == data);

    other_data = *pvk_mark(pvk_private_data_marks, H(other));
    CHECK_VK(vkSetPrivateData(device, VK_OBJECT_TYPE_BUFFER, H(other), slot, 9), VK_SUCCESS);
    CHECK(*pvk_mark(pvk_private_data_marks, H(other)) == other_data + 1);
    vkDestroyPrivateDataSlot(device, slot, NULL);
    CHECK(*pvk_mark(pvk_private_data_marks, H(other)) == other_data);
    vkDestroyBuffer(device, other, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
}

/* ---- Metal labels of named objects ---- */

/*
 * #version 450
 * layout(local_size_x = 1) in;
 * void main()
 * {
 * }
 */
static const uint32_t spv_empty[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x0000000a, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0005000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x00060010, 0x00000004, 0x00000011,
    0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
    0x6e69616d, 0x00000000, 0x00040047, 0x00000009, 0x0000000b, 0x00000019, 0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040017,
    0x00000007, 0x00000006, 0x00000003, 0x0004002b, 0x00000006, 0x00000008, 0x00000001, 0x0006002c,
    0x00000007, 0x00000009, 0x00000008, 0x00000008, 0x00000008, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x000100fd, 0x00010038,
};

static const char *label_of(id label)
{
    return label ? pvk_ns_utf8(label) : "(nil)";
}

static uint32_t memory_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags flags)
{
    VkPhysicalDeviceMemoryProperties mp;
    uint32_t i;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags &&
            !(mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT))
            return i;
    return 0;
}

static void test_names_metal_labels(void)
{
    VkInstance instance;
    VkDevice device;
    VkPhysicalDevice pd;
    VkQueue queue;
    VkMemoryAllocateInfo mai;
    VkMemoryRequirements req;
    VkDeviceMemory memory, image_memory;
    VkImageCreateInfo ii;
    VkImage image;
    VkImageViewCreateInfo vi;
    VkImageView view;
    VkSemaphoreCreateInfo si;
    VkSemaphore sem;
    VkQueryPoolCreateInfo qi;
    VkQueryPool qp, tsp;
    struct pvk_image_view *pview;
    uint32_t count = 1;
    void *pool = objc_autoreleasePoolPush();

    REQUIRE_VK(create_instance(false, NULL, NULL, &instance));
    REQUIRE_VK(create_device(instance, NULL, &device));
    vkEnumeratePhysicalDevices(instance, &count, &pd);
    vkGetDeviceQueue(device, 0, 0, &queue);

    /* Queues: Metal fixes the label at creation */
    CHECK(strcmp(label_of(pvk_MTL4CommandQueue_label(((struct pvk_queue *)queue)->mtl_queue)), "VkQueue 0.0") == 0);

    memset(&mai, 0, sizeof(mai));
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = 65536; /* small: a block of a shared heap, which keeps its label */
    mai.memoryTypeIndex = memory_type(pd, ~0u, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    REQUIRE_VK(vkAllocateMemory(device, &mai, NULL, &memory));
    set_name(device, VK_OBJECT_TYPE_DEVICE_MEMORY, H(memory), "small");
    CHECK(strcmp(label_of(pvk_MTLHeap_label(((struct pvk_device_memory *)memory)->mtl_heap)),
                 "PlasmaVK small VkDeviceMemory") == 0);
    CHECK(strcmp(label_of(pvk_MTLResource_label(((struct pvk_device_memory *)memory)->mtl_buffer)), "small") == 0);
    vkFreeMemory(device, memory, NULL);
    mai.allocationSize = 1u << 20; /* a heap of its own */
    REQUIRE_VK(vkAllocateMemory(device, &mai, NULL, &memory));
    set_name(device, VK_OBJECT_TYPE_DEVICE_MEMORY, H(memory), "pool of buffers");
    CHECK(strcmp(label_of(pvk_MTLHeap_label(((struct pvk_device_memory *)memory)->mtl_heap)), "pool of buffers") == 0);
    CHECK(strcmp(label_of(pvk_MTLResource_label(((struct pvk_device_memory *)memory)->mtl_buffer)), "pool of buffers") == 0);
    set_name(device, VK_OBJECT_TYPE_DEVICE_MEMORY, H(memory), NULL); /* removed */
    CHECK(pvk_MTLHeap_label(((struct pvk_device_memory *)memory)->mtl_heap) == NULL);

    /* An image named before it is bound: labeled when its texture is made */
    memset(&ii, 0, sizeof(ii));
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent.width = ii.extent.height = 16;
    ii.extent.depth = 1;
    ii.mipLevels = ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ii.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    REQUIRE_VK(vkCreateImage(device, &ii, NULL, &image));
    set_name(device, VK_OBJECT_TYPE_IMAGE, H(image), "albedo");
    vkGetImageMemoryRequirements(device, image, &req);
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memory_type(pd, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    REQUIRE_VK(vkAllocateMemory(device, &mai, NULL, &image_memory));
    REQUIRE_VK(vkBindImageMemory(device, image, image_memory, 0));
    CHECK(strcmp(label_of(pvk_MTLResource_label(((struct pvk_image *)image)->mtl_texture)), "albedo") == 0);
    set_name(device, VK_OBJECT_TYPE_IMAGE, H(image), "albedo map");
    CHECK(strcmp(label_of(pvk_MTLResource_label(((struct pvk_image *)image)->mtl_texture)), "albedo map") == 0);

    /* A view of another format has a texture of its own: it gets the view's
     * name, the image keeps its own */
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_R8G8B8A8_UINT;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = vi.subresourceRange.layerCount = 1;
    REQUIRE_VK(vkCreateImageView(device, &vi, NULL, &view));
    pview = (struct pvk_image_view *)view;
    set_name(device, VK_OBJECT_TYPE_IMAGE_VIEW, H(view), "albedo as uint");
    if (pview->mtl_texture != ((struct pvk_image *)image)->mtl_texture)
        CHECK(strcmp(label_of(pvk_MTLResource_label(pview->mtl_texture)), "albedo as uint") == 0);
    CHECK(strcmp(label_of(pvk_MTLResource_label(((struct pvk_image *)image)->mtl_texture)), "albedo map") == 0);

    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    REQUIRE_VK(vkCreateSemaphore(device, &si, NULL, &sem));
    set_name(device, VK_OBJECT_TYPE_SEMAPHORE, H(sem), "frame done");
    CHECK(strcmp(label_of(pvk_MTLEvent_label(((struct pvk_semaphore *)sem)->event)), "frame done") == 0);

    memset(&qi, 0, sizeof(qi));
    qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qi.queryType = VK_QUERY_TYPE_OCCLUSION;
    qi.queryCount = 4;
    REQUIRE_VK(vkCreateQueryPool(device, &qi, NULL, &qp));
    set_name(device, VK_OBJECT_TYPE_QUERY_POOL, H(qp), "occlusion");
    CHECK(strcmp(label_of(pvk_MTLResource_label(((struct pvk_query_pool *)qp)->buffer)), "occlusion") == 0);
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP; /* a counter heap */
    REQUIRE_VK(vkCreateQueryPool(device, &qi, NULL, &tsp));
    set_name(device, VK_OBJECT_TYPE_QUERY_POOL, H(tsp), "pass times");
    CHECK(strcmp(label_of(pvk_MTL4CounterHeap_label(((struct pvk_query_pool *)tsp)->heap)), "pass times") == 0);

    /* Pipeline states: labeled at creation after their shader modules */
    {
        VkShaderModuleCreateInfo smi;
        VkShaderModule module;
        VkPipelineLayoutCreateInfo pli;
        VkPipelineLayout layout;
        VkComputePipelineCreateInfo ci;
        VkPipeline pipeline;
        memset(&smi, 0, sizeof(smi));
        smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = sizeof(spv_empty);
        smi.pCode = spv_empty;
        REQUIRE_VK(vkCreateShaderModule(device, &smi, NULL, &module));
        set_name(device, VK_OBJECT_TYPE_SHADER_MODULE, H(module), "empty kernel");
        memset(&pli, 0, sizeof(pli));
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        REQUIRE_VK(vkCreatePipelineLayout(device, &pli, NULL, &layout));
        memset(&ci, 0, sizeof(ci));
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = module;
        ci.stage.pName = "main";
        ci.layout = layout;
        REQUIRE_VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, NULL, &pipeline));
        CHECK(strcmp(label_of(pvk_MTLComputePipelineState_label(((struct pvk_pipeline *)pipeline)->mtl_pipeline)),
                     "VkShaderModule 'empty kernel' main") == 0);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyPipelineLayout(device, layout, NULL);
        vkDestroyShaderModule(device, module, NULL);
    }

    vkDestroyQueryPool(device, qp, NULL);
    vkDestroyQueryPool(device, tsp, NULL);
    vkDestroySemaphore(device, sem, NULL);
    vkDestroyImageView(device, view, NULL);
    vkDestroyImage(device, image, NULL);
    vkFreeMemory(device, image_memory, NULL);
    vkFreeMemory(device, memory, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    objc_autoreleasePoolPop(pool);
}

/* ---- Metal debug groups, observed ---- */

#define LOG_MAX 2048

static struct {
    char kind[12];
    char text[160];
} trace[LOG_MAX];
static int trace_count;
static struct {
    id enc;
    int depth;
} depths[64];
static int unbalanced; /* encoders ended with debug groups open, or popped below zero */

static void trace_add(const char *kind, id string)
{
    if (trace_count < LOG_MAX) {
        snprintf(trace[trace_count].kind, sizeof(trace[0].kind), "%s", kind);
        snprintf(trace[trace_count].text, sizeof(trace[0].text), "%s", string ? pvk_ns_utf8(string) : "");
    }
    trace_count++;
}

/* The depth slot of an encoder (made on first use); -1 if the table is full. */
static int slot_of(id enc)
{
    int i, free_slot = -1;
    for (i = 0; i < 64; i++) {
        if (depths[i].enc == enc)
            return i;
        if (!depths[i].enc && free_slot < 0)
            free_slot = i;
    }
    if (free_slot >= 0) {
        depths[free_slot].enc = enc;
        depths[free_slot].depth = 0;
    }
    return free_slot;
}

static struct {
    Method method;
    IMP original;
} hooks[32];
static int hook_count;

static IMP original_of(id self, SEL cmd)
{
    Method m = class_getInstanceMethod(object_getClass(self), cmd);
    int i;
    for (i = 0; i < hook_count; i++)
        if (hooks[i].method == m)
            return hooks[i].original;
    return NULL;
}

static void hook(Class cls, const char *selector, IMP replacement)
{
    Method m = class_getInstanceMethod(cls, sel_registerName(selector));
    if (!m || method_getImplementation(m) == replacement || hook_count == 32)
        return;
    hooks[hook_count].method = m;
    hooks[hook_count].original = method_setImplementation(m, replacement);
    hook_count++;
}

static void enc_push(id self, SEL cmd, id string)
{
    int s = slot_of(self);
    trace_add("push", string);
    if (s >= 0)
        depths[s].depth++;
    ((void (*)(id, SEL, id))original_of(self, cmd))(self, cmd, string);
}

static void enc_pop(id self, SEL cmd)
{
    int s = slot_of(self);
    trace_add("pop", NULL);
    if (s >= 0 && --depths[s].depth < 0)
        unbalanced++;
    ((void (*)(id, SEL))original_of(self, cmd))(self, cmd);
}

static void enc_signpost(id self, SEL cmd, id string)
{
    trace_add("signpost", string);
    ((void (*)(id, SEL, id))original_of(self, cmd))(self, cmd, string);
}

static void enc_label(id self, SEL cmd, id string)
{
    trace_add("enclabel", string);
    ((void (*)(id, SEL, id))original_of(self, cmd))(self, cmd, string);
}

static void enc_end(id self, SEL cmd)
{
    int s = slot_of(self);
    trace_add("end", NULL);
    if (s >= 0) {
        if (depths[s].depth)
            unbalanced++;
        depths[s].enc = NULL;
    }
    ((void (*)(id, SEL))original_of(self, cmd))(self, cmd);
}

static void cb_push(id self, SEL cmd, id string)
{
    trace_add("cbpush", string);
    ((void (*)(id, SEL, id))original_of(self, cmd))(self, cmd, string);
}

static void cb_pop(id self, SEL cmd)
{
    trace_add("cbpop", NULL);
    ((void (*)(id, SEL))original_of(self, cmd))(self, cmd);
}

static void cb_label(id self, SEL cmd, id string)
{
    trace_add("cblabel", string);
    ((void (*)(id, SEL, id))original_of(self, cmd))(self, cmd, string);
}

static int trace_find(const char *kind, const char *prefix, int from)
{
    int i;
    for (i = from; i < trace_count && i < LOG_MAX; i++)
        if (strcmp(trace[i].kind, kind) == 0 && strncmp(trace[i].text, prefix, strlen(prefix)) == 0)
            return i;
    return -1;
}

static int trace_count_of(const char *kind, const char *text)
{
    int i, n = 0;
    for (i = 0; i < trace_count && i < LOG_MAX; i++)
        n += strcmp(trace[i].kind, kind) == 0 && strcmp(trace[i].text, text) == 0;
    return n;
}

/* Intercepts the debug group methods of Metal 4's command buffer and
 * compute and render encoder classes (found by making one of each). */
static void install_hooks(VkQueue queue, VkImage image)
{
    struct pvk_queue *q = (struct pvk_queue *)queue;
    struct pvk_submission *s = pvk_queue_get_submission(q);
    void *pool = objc_autoreleasePoolPush();
    id enc, pass, att;
    Class classes[3];
    int i;

    if (!s)
        return;
    pvk_MTL4CommandBuffer_beginCommandBufferWithAllocator(s->cb, s->allocator);
    enc = pvk_MTL4CommandBuffer_computeCommandEncoder(s->cb);
    classes[0] = object_getClass(enc);
    pvk_MTL4CommandEncoder_endEncoding(enc);
    pass = pvk_MTL4RenderPassDescriptor_new();
    att = pvk_MTLRenderPassColorAttachmentDescriptorArray_objectAtIndexedSubscript(
        pvk_MTL4RenderPassDescriptor_colorAttachments(pass), 0);
    pvk_MTLRenderPassAttachmentDescriptor_setTexture(att, ((struct pvk_image *)image)->mtl_texture);
    pvk_MTLRenderPassAttachmentDescriptor_setLoadAction(att, PVK_MTLLoadActionDontCare);
    pvk_MTLRenderPassAttachmentDescriptor_setStoreAction(att, PVK_MTLStoreActionStore);
    enc = pvk_MTL4CommandBuffer_renderCommandEncoderWithDescriptor(s->cb, pass);
    classes[1] = object_getClass(enc);
    pvk_MTL4CommandEncoder_endEncoding(enc);
    pvk_release(pass);
    pvk_MTL4CommandBuffer_endCommandBuffer(s->cb);
    classes[2] = object_getClass(s->cb);
    pvk_queue_put_submissions(q, s);
    objc_autoreleasePoolPop(pool);
    for (i = 0; i < 2; i++) {
        hook(classes[i], "pushDebugGroup:", (IMP)enc_push);
        hook(classes[i], "popDebugGroup", (IMP)enc_pop);
        hook(classes[i], "insertDebugSignpost:", (IMP)enc_signpost);
        hook(classes[i], "setLabel:", (IMP)enc_label);
        hook(classes[i], "endEncoding", (IMP)enc_end);
    }
    hook(classes[2], "pushDebugGroup:", (IMP)cb_push);
    hook(classes[2], "popDebugGroup", (IMP)cb_pop);
    hook(classes[2], "setLabel:", (IMP)cb_label);
}

/* Label regions and the debug layer's groups as Metal debug groups: nested,
 * balanced in every encoder, open across encoders, command buffers and
 * submissions; the queue's as command buffer groups; signposts; labels of
 * command buffers and encoders. */
static void test_labels_metal(void)
{
    VkInstance instance;
    VkDevice device;
    VkPhysicalDevice pd;
    VkQueue queue;
    VkCommandPoolCreateInfo cpi;
    VkCommandPool cpool;
    VkCommandBufferAllocateInfo cai;
    VkCommandBufferBeginInfo begin;
    VkCommandBuffer cbs[2];
    VkBufferCreateInfo bi;
    VkBuffer buffer;
    VkMemoryAllocateInfo mai;
    VkMemoryRequirements req;
    VkDeviceMemory buffer_memory, image_memory;
    VkImageCreateInfo ii;
    VkImage image;
    VkImageViewCreateInfo vi;
    VkImageView view;
    VkRenderingAttachmentInfo color;
    VkRenderingInfo ri;
    VkClearColorValue clear;
    VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkDebugUtilsLabelEXT outer, frame, point, inner;
    VkSubmitInfo submit;
    uint32_t count = 1, *words;
    int i, at;

    REQUIRE_VK(create_instance(true, NULL, NULL, &instance));
    REQUIRE_VK(create_device(instance, NULL, &device));
    vkEnumeratePhysicalDevices(instance, &count, &pd);
    vkGetDeviceQueue(device, 0, 0, &queue);
    set_name(device, VK_OBJECT_TYPE_QUEUE, H(queue), "gfx");

    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = 4096;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    REQUIRE_VK(vkCreateBuffer(device, &bi, NULL, &buffer));
    vkGetBufferMemoryRequirements(device, buffer, &req);
    memset(&mai, 0, sizeof(mai));
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memory_type(pd, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    REQUIRE_VK(vkAllocateMemory(device, &mai, NULL, &buffer_memory));
    REQUIRE_VK(vkBindBufferMemory(device, buffer, buffer_memory, 0));
    set_name(device, VK_OBJECT_TYPE_BUFFER, H(buffer), "dst");

    memset(&ii, 0, sizeof(ii));
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent.width = ii.extent.height = 16;
    ii.extent.depth = 1;
    ii.mipLevels = ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    REQUIRE_VK(vkCreateImage(device, &ii, NULL, &image));
    vkGetImageMemoryRequirements(device, image, &req);
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memory_type(pd, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    REQUIRE_VK(vkAllocateMemory(device, &mai, NULL, &image_memory));
    REQUIRE_VK(vkBindImageMemory(device, image, image_memory, 0));
    set_name(device, VK_OBJECT_TYPE_IMAGE, H(image), "color");
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange = range;
    REQUIRE_VK(vkCreateImageView(device, &vi, NULL, &view));
    set_name(device, VK_OBJECT_TYPE_IMAGE_VIEW, H(view), "color view");

    install_hooks(queue, image);
    REQUIRE(hook_count >= 8);

    memset(&cpi, 0, sizeof(cpi));
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    REQUIRE_VK(vkCreateCommandPool(device, &cpi, NULL, &cpool));
    memset(&cai, 0, sizeof(cai));
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = cpool;
    cai.commandBufferCount = 2;
    REQUIRE_VK(vkAllocateCommandBuffers(device, &cai, cbs));
    set_name(device, VK_OBJECT_TYPE_COMMAND_BUFFER, H(cbs[0]), "main");
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    memset(&outer, 0, sizeof(outer));
    outer.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    frame = point = inner = outer;
    outer.pLabelName = "outer";
    frame.pLabelName = "frame";
    point.pLabelName = "point";
    inner.pLabelName = "inner";

    /* cbs[0]: outer { fill, clear, rendering, point }, a region left open;
     * cbs[1]: a copy, the end of both */
    REQUIRE_VK(vkBeginCommandBuffer(cbs[0], &begin));
    vkCmdBeginDebugUtilsLabelEXT(cbs[0], &outer);
    vkCmdFillBuffer(cbs[0], buffer, 0, 1024, 0x01020304u);
    memset(&clear, 0, sizeof(clear));
    vkCmdClearColorImage(cbs[0], image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
    memset(&color, 0, sizeof(color));
    color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color.imageView = view;
    color.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    memset(&ri, 0, sizeof(ri));
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.extent.width = ri.renderArea.extent.height = 16;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    vkCmdBeginRendering(cbs[0], &ri);
    vkCmdInsertDebugUtilsLabelEXT(cbs[0], &point);
    vkCmdEndRendering(cbs[0]);
    vkCmdBeginDebugUtilsLabelEXT(cbs[0], &inner);
    REQUIRE_VK(vkEndCommandBuffer(cbs[0]));
    REQUIRE_VK(vkBeginCommandBuffer(cbs[1], &begin));
    vkCmdCopyBuffer(cbs[1], buffer, buffer, 1, &(VkBufferCopy){ 0, 2048, 1024 });
    vkCmdEndDebugUtilsLabelEXT(cbs[1]); /* closes "inner"; "outer" stays open */
    REQUIRE_VK(vkEndCommandBuffer(cbs[1]));

    trace_count = 0;
    vkQueueBeginDebugUtilsLabelEXT(queue, &frame);
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 2;
    submit.pCommandBuffers = cbs;
    REQUIRE_VK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    vkQueueEndDebugUtilsLabelEXT(queue);
    CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);

    CHECK(unbalanced == 0);
    CHECK(trace_find("cblabel", "VkQueue 'gfx' #", 0) >= 0);
    at = trace_find("cblabel", "VkQueue 'gfx' #", 0);
    CHECK(at >= 0 && strstr(trace[at].text, "VkCommandBuffer 'main'") != NULL);
    CHECK(trace_count_of("cbpush", "frame") == 1 && trace_count_of("cbpop", "") == 1);
    CHECK(trace_find("push", "VkCommandBuffer 'main'", 0) >= 0);
    CHECK(trace_find("push", "vkCmdFillBuffer: VkBuffer 'dst'", 0) >= 0);
    CHECK(trace_find("push", "vkCmdClearColorImage: VkImage 'color'", 0) >= 0);
    CHECK(trace_find("push", "vkCmdBeginRendering: VkImageView 'color view'", 0) >= 0);
    CHECK(trace_find("enclabel", "vkCmdBeginRendering: VkImageView 'color view'", 0) >= 0);
    CHECK(trace_find("push", "vkCmdCopyBuffer: VkBuffer 'dst', VkBuffer 'dst'", 0) >= 0);
    CHECK(trace_find("signpost", "point", 0) >= 0);
    /* "outer" is a group of every encoder from the fill to the copy (at least
     * the fill's, the clear's, the render pass's and the copy's) */
    CHECK(trace_count_of("push", "outer") >= 3);
    CHECK(trace_count_of("push", "inner") >= 1);
    /* The copy's encoder (pushes after the encoder before it ended) nests
     * the regions in the order they began: "outer", then "inner" */
    at = trace_find("push", "vkCmdCopyBuffer", 0);
    for (i = at; i > 0 && strcmp(trace[i].kind, "end") != 0; i--)
        ;
    CHECK(at > 0 && trace_find("push", "outer", i) >= 0 &&
          trace_find("push", "outer", i) < trace_find("push", "inner", i) && trace_find("push", "inner", i) < at);
    words = NULL;
    REQUIRE_VK(vkMapMemory(device, buffer_memory, 0, VK_WHOLE_SIZE, 0, (void **)&words));
    CHECK(words[0] == 0x01020304u && words[255] == 0x01020304u && words[512] == 0x01020304u &&
          words[767] == 0x01020304u);
    vkUnmapMemory(device, buffer_memory);

    /* The next submission: "outer" is still open, and ends */
    trace_count = 0;
    REQUIRE_VK(vkResetCommandBuffer(cbs[0], 0));
    REQUIRE_VK(vkBeginCommandBuffer(cbs[0], &begin));
    vkCmdFillBuffer(cbs[0], buffer, 0, 64, 7);
    vkCmdEndDebugUtilsLabelEXT(cbs[0]);
    vkCmdFillBuffer(cbs[0], buffer, 64, 64, 8);
    REQUIRE_VK(vkEndCommandBuffer(cbs[0]));
    submit.commandBufferCount = 1;
    REQUIRE_VK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);
    CHECK(unbalanced == 0);
    CHECK(trace_count_of("push", "outer") == 1); /* its text survived its command buffer's reset */
    CHECK(trace_count_of("cbpush", "frame") == 0);
    CHECK(((struct pvk_queue *)queue)->cmd_labels.count == 0);

    vkFreeCommandBuffers(device, cpool, 2, cbs);
    vkDestroyCommandPool(device, cpool, NULL);
    vkDestroyImageView(device, view, NULL);
    vkDestroyImage(device, image, NULL);
    vkFreeMemory(device, image_memory, NULL);
    vkDestroyBuffer(device, buffer, NULL);
    vkFreeMemory(device, buffer_memory, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
}

/* ---- The debug layer: listed, enabled, reported ---- */

static void test_layer(void)
{
    VkLayerProperties layers[2];
    VkExtensionProperties exts[4];
    VkPhysicalDeviceToolProperties tool;
    VkPhysicalDevice pd;
    VkInstance instance;
    uint32_t count = 2, n;

    REQUIRE_VK(vkEnumerateInstanceLayerProperties(&count, layers));
    REQUIRE(count == 1);
    CHECK(strcmp(layers[0].layerName, "VK_LAYER_PLASMAVK_debug") == 0);
    CHECK(layers[0].specVersion == PVK_API_VERSION && layers[0].implementationVersion == PLASMAVK_VERSION);
    n = 4;
    CHECK_VK(vkEnumerateInstanceExtensionProperties("VK_LAYER_PLASMAVK_debug", &n, exts), VK_SUCCESS);
    CHECK(n == 1 && strcmp(exts[0].extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0);

    /* Without the layer: no device layers, no tools (unless PLASMAVK_DEBUG
     * enables it anyway) */
    REQUIRE_VK(create_instance(false, NULL, NULL, &instance));
    count = 1;
    vkEnumeratePhysicalDevices(instance, &count, &pd);
    if (pvk_debug_env) {
        CHECK(((struct pvk_instance *)instance)->debug);
        vkDestroyInstance(instance, NULL);
        return;
    }
    CHECK(!((struct pvk_instance *)instance)->debug);
    n = 2;
    CHECK_VK(vkEnumerateDeviceLayerProperties(pd, &n, layers), VK_SUCCESS);
    CHECK(n == 0);
    n = 1;
    CHECK_VK(vkGetPhysicalDeviceToolProperties(pd, &n, &tool), VK_SUCCESS);
    CHECK(n == 0);
    n = 4;
    CHECK_VK(vkEnumerateDeviceExtensionProperties(pd, "VK_LAYER_PLASMAVK_debug", &n, exts), VK_SUCCESS);
    CHECK(n == 0);
    vkDestroyInstance(instance, NULL);

    /* With it */
    REQUIRE_VK(create_instance(true, NULL, NULL, &instance));
    count = 1;
    vkEnumeratePhysicalDevices(instance, &count, &pd);
    CHECK(((struct pvk_instance *)instance)->debug);
    n = 2;
    CHECK_VK(vkEnumerateDeviceLayerProperties(pd, &n, layers), VK_SUCCESS);
    CHECK(n == 1 && strcmp(layers[0].layerName, "VK_LAYER_PLASMAVK_debug") == 0);
    n = 0;
    CHECK_VK(vkEnumerateDeviceLayerProperties(pd, &n, layers), VK_INCOMPLETE);
    memset(&tool, 0, sizeof(tool));
    tool.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES;
    n = 1;
    CHECK_VK(vkGetPhysicalDeviceToolProperties(pd, &n, &tool), VK_SUCCESS);
    CHECK(n == 1 && strcmp(tool.layer, "VK_LAYER_PLASMAVK_debug") == 0);
    CHECK(tool.purposes & VK_TOOL_PURPOSE_VALIDATION_BIT);
    CHECK(tool.sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES);
    vkDestroyInstance(instance, NULL);
}

/* The faster paths PlasmaVK leaves are reported once per cause as INFO
 * messages of the performance type: a batch with enough work for parallel
 * encoding whose command buffer records a label region is encoded in
 * order, said once, not again for the next such batch. */
static void test_fallback_messages(void)
{
    VkInstance instance;
    VkDevice device;
    VkQueue queue;
    VkDebugUtilsMessengerCreateInfoEXT info;
    VkDebugUtilsMessengerEXT messenger;
    VkCommandPoolCreateInfo cpi;
    VkCommandPool cpool;
    VkCommandBufferAllocateInfo cai;
    VkCommandBufferBeginInfo begin;
    VkCommandBuffer cbs[2];
    VkViewport viewport = { 0, 0, 16, 16, 0, 1 };
    VkDebugUtilsLabelEXT label;
    VkSubmitInfo submit;
    struct got g;
    int round, i, k, found;

    REQUIRE_VK(create_instance(false, NULL, NULL, &instance));
    REQUIRE_VK(create_device(instance, NULL, &device));
    vkGetDeviceQueue(device, 0, 0, &queue);
    info = messenger_info(&g, ALL_SEVERITIES); /* (after the device: its messages are not the point) */
    REQUIRE_VK(vkCreateDebugUtilsMessengerEXT(instance, &info, NULL, &messenger));
    memset(&cpi, 0, sizeof(cpi));
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    REQUIRE_VK(vkCreateCommandPool(device, &cpi, NULL, &cpool));
    memset(&cai, 0, sizeof(cai));
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = cpool;
    cai.commandBufferCount = 2;
    REQUIRE_VK(vkAllocateCommandBuffers(device, &cai, cbs));
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    memset(&label, 0, sizeof(label));
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = "frame";
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 2;
    submit.pCommandBuffers = cbs;
    for (round = 0; round < 2; round++) {
        CHECK_VK(vkResetCommandPool(device, cpool, 0), VK_SUCCESS);
        for (k = 0; k < 2; k++) { /* 130 packets each: two jobs' worth (PVK_PAR_PACKETS) */
            CHECK_VK(vkBeginCommandBuffer(cbs[k], &begin), VK_SUCCESS);
            if (k == 0)
                vkCmdBeginDebugUtilsLabelEXT(cbs[k], &label);
            for (i = 0; i < 130; i++)
                vkCmdSetViewport(cbs[k], 0, 1, &viewport);
            if (k == 0)
                vkCmdEndDebugUtilsLabelEXT(cbs[k]);
            CHECK_VK(vkEndCommandBuffer(cbs[k]), VK_SUCCESS);
        }
        CHECK_VK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), VK_SUCCESS);
        CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);
        for (i = 0, found = 0; i < g.count && i < GOT_MAX; i++)
            if (strcmp(g.ids[i], "PlasmaVK-parallel-encoding") == 0) {
                found++;
                CHECK(g.severity[i] == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT);
                CHECK(g.types[i] == VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT);
                CHECK(strstr(g.text[i], "label regions") != NULL);
                CHECK(g.objects[i] == 1 && g.object0[i] == H(device));
            }
        CHECK(found == 1); /* (once, also after the second batch) */
    }
    vkDestroyCommandPool(device, cpool, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyDebugUtilsMessengerEXT(instance, messenger, NULL);
    vkDestroyInstance(instance, NULL);
}

int main(void)
{
    RUN(test_create_info_messages);
    RUN(test_internal_messages);
    RUN(test_marks);
    RUN(test_names_metal_labels);
    RUN(test_labels_metal);
    RUN(test_layer);
    RUN(test_fallback_messages);
    return TEST_RESULT();
}
