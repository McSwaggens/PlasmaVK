/* Window system integration (section 18): Metal surfaces, swapchains,
 * acquire and present. The CAMetalLayers are not in a window: such layers
 * still vend drawables, so this runs headless. Presented contents are read
 * back from the drawables, which a CAMetalLayer subclass records. */
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <string.h>
#include <time.h>

#include "../pvk_test.h"

static VkInstance instance;
static VkPhysicalDevice pd;
static VkDevice device;
static VkQueue queues[2]; /* family 0 (graphics), family 1 (compute) */
static VkCommandPool pools[2];
static uint32_t shared_type = UINT32_MAX;

/* ---- Layers ---- */

static Class test_layer_class; /* CAMetalLayer subclass recording nextDrawable */
static id last_drawable;       /* retained */
static double grow_next;       /* added to the drawableSize at the next nextDrawable: a view resizing meanwhile */

static id test_next_drawable(id self, SEL cmd)
{
    struct objc_super super_ = { self, (Class)objc_getClass("CAMetalLayer") };
    id drawable;
    if (grow_next != 0) {
        pvk_CGSize size = pvk_CAMetalLayer_drawableSize(self);
        size.width += grow_next;
        size.height += grow_next;
        pvk_CAMetalLayer_setDrawableSize(self, size);
        grow_next = 0;
    }
    drawable = ((id(*)(struct objc_super *, SEL))objc_msgSendSuper)(&super_, cmd);
    pvk_release(last_drawable);
    last_drawable = pvk_retain(drawable);
    return drawable;
}

static id make_layer(double w, double h, double scale)
{
    pvk_CGRect frame;
    id layer;
    if (!test_layer_class) {
        test_layer_class = objc_allocateClassPair((Class)objc_getClass("CAMetalLayer"), "PVKTestMetalLayer", 0);
        class_addMethod(test_layer_class, sel_registerName("nextDrawable"), (IMP)test_next_drawable, "@@:");
        objc_registerClassPair(test_layer_class);
    }
    layer = PVK_MSG(id, (id, SEL))((id)test_layer_class, sel_registerName("new"));
    frame.origin.x = frame.origin.y = 0;
    frame.size.width = w;
    frame.size.height = h;
    PVK_MSG(void, (id, SEL, pvk_CGRect))(layer, sel_registerName("setFrame:"), frame);
    PVK_MSG(void, (id, SEL, pvk_CGFloat))(layer, sel_registerName("setContentsScale:"), scale);
    return layer;
}

static VkSurfaceKHR make_surface(id layer)
{
    VkMetalSurfaceCreateInfoEXT info;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
    info.pLayer = layer;
    if (vkCreateMetalSurfaceEXT(instance, &info, NULL, &surface) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return surface;
}

/* ---- Setup ---- */

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

static void setup(void)
{
    const char *instance_exts[] = { VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_METAL_SURFACE_EXTENSION_NAME,
                                    VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME };
    const char *device_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    VkPhysicalDeviceMemoryProperties mem;
    VkInstanceCreateInfo instance_info;
    VkApplicationInfo app;
    VkDeviceQueueCreateInfo queue_info[2];
    VkDeviceCreateInfo device_info;
    VkCommandPoolCreateInfo pool_info;
    float priority = 1.0f;
    uint32_t count = 1, i;

    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_4;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    instance_info.enabledExtensionCount = 4;
    instance_info.ppEnabledExtensionNames = instance_exts;
    REQUIRE_VK(vkCreateInstance(&instance_info, NULL, &instance));
    REQUIRE_VK(vkEnumeratePhysicalDevices(instance, &count, &pd));

    memset(queue_info, 0, sizeof(queue_info));
    for (i = 0; i < 2; i++) {
        queue_info[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info[i].queueFamilyIndex = i;
        queue_info[i].queueCount = 1;
        queue_info[i].pQueuePriorities = &priority;
    }
    memset(&v13, 0, sizeof(v13));
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13.dynamicRendering = VK_TRUE;
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.pNext = &v13;
    v12.timelineSemaphore = VK_TRUE;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &v12;
    device_info.queueCreateInfoCount = 2;
    device_info.pQueueCreateInfos = queue_info;
    device_info.enabledExtensionCount = 1;
    device_info.ppEnabledExtensionNames = device_exts;
    REQUIRE_VK(vkCreateDevice(pd, &device_info, NULL, &device));
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    for (i = 0; i < mem.memoryTypeCount && shared_type == UINT32_MAX; i++)
        if (mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            shared_type = i;
    for (i = 0; i < 2; i++) {
        vkGetDeviceQueue(device, i, 0, &queues[i]);
        memset(&pool_info, 0, sizeof(pool_info));
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = i;
        REQUIRE_VK(vkCreateCommandPool(device, &pool_info, NULL, &pools[i]));
    }
}

static void teardown(void)
{
    vkDestroyCommandPool(device, pools[0], NULL);
    vkDestroyCommandPool(device, pools[1], NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    pvk_release(last_drawable);
    last_drawable = NULL;
}

static VkSwapchainCreateInfoKHR swapchain_info(VkSurfaceKHR surface, VkFormat format, VkColorSpaceKHR space,
                                               uint32_t w, uint32_t h, uint32_t images)
{
    VkSwapchainCreateInfoKHR info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface = surface;
    info.minImageCount = images;
    info.imageFormat = format;
    info.imageColorSpace = space;
    info.imageExtent.width = w;
    info.imageExtent.height = h;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    info.clipped = VK_TRUE;
    return info;
}

static VkCommandBuffer begin(uint32_t family)
{
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferBeginInfo info;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = pools[family];
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &alloc, &cb) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &info);
    return cb;
}

static void barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to)
{
    VkImageMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL,
                         1, &b);
}

/* Records a clear of the image to `color` (UNDEFINED -> PRESENT_SRC). */
static void record_clear(VkCommandBuffer cb, VkImage image, float r, float g, float b, float a)
{
    VkClearColorValue color;
    VkImageSubresourceRange range;
    memset(&range, 0, sizeof(range));
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;
    color.float32[0] = r;
    color.float32[1] = g;
    color.float32[2] = b;
    color.float32[3] = a;
    barrier(cb, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdClearColorImage(cb, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
    barrier(cb, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
}

static VkResult submit(uint32_t family, VkCommandBuffer cb, VkSemaphore wait, VkSemaphore signal, VkFence fence)
{
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo info;
    vkEndCommandBuffer(cb);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.waitSemaphoreCount = wait ? 1 : 0;
    info.pWaitSemaphores = &wait;
    info.pWaitDstStageMask = &stage;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &cb;
    info.signalSemaphoreCount = signal ? 1 : 0;
    info.pSignalSemaphores = &signal;
    return vkQueueSubmit(queues[family], 1, &info, fence);
}

static VkResult present(uint32_t family, VkSwapchainKHR swapchain, uint32_t index, VkSemaphore wait)
{
    VkPresentInfoKHR info;
    VkResult result = VK_ERROR_UNKNOWN, r;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = wait ? 1 : 0;
    info.pWaitSemaphores = &wait;
    info.swapchainCount = 1;
    info.pSwapchains = &swapchain;
    info.pImageIndices = &index;
    info.pResults = &result;
    r = vkQueuePresentKHR(queues[family], &info);
    if (r != result)
        PVK_TEST_FAIL("vkQueuePresentKHR returned %d, pResults %d", (int)r, (int)result);
    return r;
}

static VkSemaphore make_semaphore(VkSemaphoreType type)
{
    VkSemaphoreTypeCreateInfo t;
    VkSemaphoreCreateInfo info;
    VkSemaphore s = VK_NULL_HANDLE;
    memset(&t, 0, sizeof(t));
    t.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    t.semaphoreType = type;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    info.pNext = &t;
    vkCreateSemaphore(device, &info, NULL, &s);
    return s;
}

static VkFence make_fence(void)
{
    VkFenceCreateInfo info;
    VkFence f = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device, &info, NULL, &f);
    return f;
}

/* Nothing waits on the queue's thread: every wait so far was resolved
 * when submitted (white box: section 16). */
static bool no_backlog(uint32_t family)
{
    struct pvk_queue *q = (struct pvk_queue *)queues[family];
    bool none;
    pthread_mutex_lock(&q->lock);
    none = q->deferred == NULL;
    pthread_mutex_unlock(&q->lock);
    return none;
}

/* ---- Tests ---- */

static void test_surface_queries(void)
{
    id layer = make_layer(64, 48, 2.0), bare = make_layer(0, 0, 1.0);
    VkSurfaceKHR surface = make_surface(layer), unsized = make_surface(bare);
    VkSurfaceCapabilitiesKHR caps;
    VkSurfaceFormatKHR formats[64];
    VkPresentModeKHR modes[8];
    VkDeviceGroupPresentCapabilitiesKHR group;
    VkDeviceGroupPresentModeFlagsKHR group_modes = 0;
    VkRect2D rect;
    VkBool32 supported;
    uint32_t count, i, fifo = 0, immediate = 0;

    REQUIRE(surface && unsized);
    for (i = 0; i < 2; i++) {
        supported = VK_FALSE;
        CHECK_VK(vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &supported), VK_SUCCESS);
        CHECK(supported == VK_TRUE);
    }
    CHECK_VK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps), VK_SUCCESS);
    CHECK(caps.minImageCount == 2 && caps.maxImageCount == 3);
    CHECK(caps.currentExtent.width == 128 && caps.currentExtent.height == 96);
    CHECK(caps.minImageExtent.width == 1 && caps.maxImageExtent.width >= 16384);
    CHECK(caps.maxImageArrayLayers == 1);
    CHECK(caps.currentTransform == VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR);
    CHECK(caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
    CHECK(caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    /* A layer with no area: the swapchain decides. */
    CHECK_VK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, unsized, &caps), VK_SUCCESS);
    CHECK(caps.currentExtent.width == 0xFFFFFFFFu && caps.currentExtent.height == 0xFFFFFFFFu);

    /* VK_EXT_swapchain_colorspace is enabled: extended color spaces too. */
    count = 64;
    CHECK_VK(vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &count, formats), VK_SUCCESS);
    CHECK(count > 5);
    CHECK(formats[0].format == VK_FORMAT_B8G8R8A8_UNORM && formats[0].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);
    for (i = 0, supported = 0; i < count; i++)
        supported |= formats[i].format == VK_FORMAT_R16G16B16A16_SFLOAT &&
                     formats[i].colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT;
    CHECK(supported);
    count = 2;
    CHECK_VK(vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &count, formats), VK_INCOMPLETE);
    CHECK(count == 2);

    count = 8;
    CHECK_VK(vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &count, modes), VK_SUCCESS);
    for (i = 0; i < count; i++) {
        fifo |= modes[i] == VK_PRESENT_MODE_FIFO_KHR;
        immediate |= modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
    CHECK(fifo && immediate); /* macOS: displaySyncEnabled */

    count = 1;
    CHECK_VK(vkGetPhysicalDevicePresentRectanglesKHR(pd, surface, &count, &rect), VK_SUCCESS);
    CHECK(count == 1 && rect.offset.x == 0 && rect.extent.width == 128 && rect.extent.height == 96);
    CHECK_VK(vkGetPhysicalDevicePresentRectanglesKHR(pd, unsized, &count, &rect), VK_SUCCESS);
    CHECK(rect.extent.width == 0xFFFFFFFFu);

    memset(&group, 0, sizeof(group));
    group.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_CAPABILITIES_KHR;
    CHECK_VK(vkGetDeviceGroupPresentCapabilitiesKHR(device, &group), VK_SUCCESS);
    CHECK(group.presentMask[0] == 1 && group.presentMask[1] == 0);
    CHECK(group.modes == VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR);
    CHECK_VK(vkGetDeviceGroupSurfacePresentModesKHR(device, surface, &group_modes), VK_SUCCESS);
    CHECK(group_modes == VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR);

    vkDestroySurfaceKHR(instance, surface, NULL);
    vkDestroySurfaceKHR(instance, unsized, NULL);
    vkDestroySurfaceKHR(instance, VK_NULL_HANDLE, NULL);
    pvk_release(layer);
    pvk_release(bare);
}

/* Without VK_EXT_swapchain_colorspace only sRGB; WSI commands follow the
 * enabled extensions in vkGet*ProcAddr. */
static void test_extension_scoping(void)
{
    const char *surface_ext[] = { VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_METAL_SURFACE_EXTENSION_NAME };
    VkInstanceCreateInfo info;
    VkDeviceQueueCreateInfo queue;
    VkDeviceCreateInfo dinfo;
    VkSurfaceFormatKHR formats[64];
    VkMetalSurfaceCreateInfoEXT sinfo;
    VkInstance inst;
    VkPhysicalDevice phys;
    VkSurfaceKHR surface;
    VkDevice dev;
    id layer = make_layer(16, 16, 1.0);
    float priority = 1.0f;
    uint32_t count = 1, i;

    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    REQUIRE_VK(vkCreateInstance(&info, NULL, &inst));
    CHECK(vkGetInstanceProcAddr(inst, "vkCreateMetalSurfaceEXT") == NULL);
    CHECK(vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceSurfaceFormatsKHR") == NULL);
    CHECK(vkGetInstanceProcAddr(inst, "vkCreateSwapchainKHR") == NULL); /* VK_KHR_swapchain needs VK_KHR_surface */
    vkDestroyInstance(inst, NULL);

    info.enabledExtensionCount = 2;
    info.ppEnabledExtensionNames = surface_ext;
    REQUIRE_VK(vkCreateInstance(&info, NULL, &inst)); /* apiVersion 1.0 */
    CHECK(vkGetInstanceProcAddr(inst, "vkCreateMetalSurfaceEXT") != NULL);
    CHECK(vkGetInstanceProcAddr(inst, "vkQueuePresentKHR") != NULL);
    REQUIRE_VK(vkEnumeratePhysicalDevices(inst, &count, &phys));
    memset(&sinfo, 0, sizeof(sinfo));
    sinfo.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
    sinfo.pLayer = layer;
    REQUIRE_VK(vkCreateMetalSurfaceEXT(inst, &sinfo, NULL, &surface));
    count = 64;
    CHECK_VK(vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &count, formats), VK_SUCCESS);
    CHECK(count == 5);
    for (i = 0; i < count; i++)
        CHECK(formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);

    memset(&queue, 0, sizeof(queue));
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    memset(&dinfo, 0, sizeof(dinfo));
    dinfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dinfo.queueCreateInfoCount = 1;
    dinfo.pQueueCreateInfos = &queue;
    REQUIRE_VK(vkCreateDevice(phys, &dinfo, NULL, &dev));
    CHECK(vkGetDeviceProcAddr(dev, "vkQueuePresentKHR") == NULL); /* VK_KHR_swapchain not enabled */
    vkDestroyDevice(dev, NULL);
    dinfo.enabledExtensionCount = 1;
    dinfo.ppEnabledExtensionNames = (const char *const[]){ VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    REQUIRE_VK(vkCreateDevice(phys, &dinfo, NULL, &dev));
    CHECK(vkGetDeviceProcAddr(dev, "vkQueuePresentKHR") != NULL);
    CHECK(vkGetDeviceProcAddr(dev, "vkAcquireNextImageKHR") != NULL);
    CHECK(vkGetDeviceProcAddr(dev, "vkAcquireNextImage2KHR") == NULL); /* needs Vulkan 1.1 */
    vkDestroyDevice(dev, NULL);
    vkDestroySurfaceKHR(inst, surface, NULL);
    vkDestroyInstance(inst, NULL);
    pvk_release(layer);
}

/* Frames with acquire semaphores and fences; images keep their handles;
 * the layer is configured as asked. */
static void test_present_cycles(void)
{
    id layer = make_layer(32, 32, 1.0);
    VkSurfaceKHR surface = make_surface(layer);
    VkSwapchainCreateInfoKHR info = swapchain_info(surface, VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                                                   32, 32, 3);
    VkSwapchainKHR swapchain;
    VkImage images[4], again[4];
    VkSemaphore acquired = make_semaphore(VK_SEMAPHORE_TYPE_BINARY), rendered = make_semaphore(VK_SEMAPHORE_TYPE_BINARY);
    VkFence fence = make_fence(), done = make_fence();
    uint32_t count = 4, n = 0, frame, index;
    pvk_CGSize size;

    REQUIRE(surface);
    REQUIRE_VK(vkCreateSwapchainKHR(device, &info, NULL, &swapchain));
    REQUIRE_VK(vkGetSwapchainImagesKHR(device, swapchain, &count, images));
    CHECK(count == 3);
    n = 1;
    CHECK_VK(vkGetSwapchainImagesKHR(device, swapchain, &n, again), VK_INCOMPLETE);
    CHECK(n == 1 && again[0] == images[0]);
    size = pvk_CAMetalLayer_drawableSize(layer);
    CHECK(size.width == 32 && size.height == 32);
    CHECK(pvk_CAMetalLayer_pixelFormat(layer) == PVK_MTLPixelFormatBGRA8Unorm_sRGB);
    CHECK(PVK_MSG(unsigned long, (id, SEL))(layer, sel_registerName("maximumDrawableCount")) == 3);
    CHECK(PVK_MSG(bool, (id, SEL))(layer, sel_registerName("framebufferOnly"))); /* presents draw into them */
    CHECK(PVK_MSG(bool, (id, SEL))(layer, sel_registerName("displaySyncEnabled")));

    for (frame = 0; frame < 30; frame++) {
        VkCommandBuffer cb;
        uint32_t i;
        index = UINT32_MAX;
        REQUIRE_VK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, acquired, fence, &index));
        REQUIRE(index < count);
        REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
        REQUIRE_VK(vkResetFences(device, 1, &fence));
        cb = begin(0);
        REQUIRE(cb);
        record_clear(cb, images[index], (float)frame / 30.0f, 0.5f, 0.25f, 1.0f);
        REQUIRE_VK(submit(0, cb, acquired, rendered, frame == 29 ? done : VK_NULL_HANDLE));
        CHECK_VK(present(0, swapchain, index, rendered), VK_SUCCESS);
        /* The present's render pass keeps no drawable from the layer's pool. */
        CHECK(!PVK_MSG(id, (id, SEL))(((struct pvk_swapchain *)swapchain)->present_attachment,
                                      sel_registerName("texture")));
        n = 4;
        vkGetSwapchainImagesKHR(device, swapchain, &n, again);
        for (i = 0; i < count; i++)
            CHECK(again[i] == images[i]);
    }
    CHECK_VK(vkWaitForFences(device, 1, &done, VK_TRUE, 2000000000ull), VK_SUCCESS);
    CHECK_VK(vkQueueWaitIdle(queues[0]), VK_SUCCESS);
    vkResetCommandPool(device, pools[0], 0);
    vkDestroySwapchainKHR(device, swapchain, NULL);
    vkDestroySwapchainKHR(device, VK_NULL_HANDLE, NULL);
    vkDestroySemaphore(device, acquired, NULL);
    vkDestroySemaphore(device, rendered, NULL);
    vkDestroyFence(device, fence, NULL);
    vkDestroyFence(device, done, NULL);
    vkDestroySurfaceKHR(instance, surface, NULL);
    pvk_release(layer);
}

/* VK_EXT_headless_surface: no layer, an extent the swapchain decides, FIFO
 * and IMMEDIATE; one swapchain per surface; frames acquired, rendered and
 * presented as with a window, never out of date. */
static void test_headless(void)
{
    VkHeadlessSurfaceCreateInfoEXT hi;
    VkSurfaceKHR surface = VK_NULL_HANDLE, other = VK_NULL_HANDLE;
    VkSurfaceCapabilitiesKHR caps;
    VkPresentModeKHR modes[4];
    VkSwapchainCreateInfoKHR info;
    VkSwapchainKHR swapchain, second;
    VkImage images[4];
    VkSemaphore acquired = make_semaphore(VK_SEMAPHORE_TYPE_BINARY), rendered = make_semaphore(VK_SEMAPHORE_TYPE_BINARY);
    VkFence fence = make_fence(), done = make_fence();
    uint32_t count = 4, frame, index;
    VkBool32 supported = VK_FALSE;

    memset(&hi, 0, sizeof(hi));
    hi.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
    REQUIRE_VK(vkCreateHeadlessSurfaceEXT(instance, &hi, NULL, &surface));
    REQUIRE_VK(vkCreateHeadlessSurfaceEXT(instance, &hi, NULL, &other));
    CHECK_VK(vkGetPhysicalDeviceSurfaceSupportKHR(pd, 0, surface, &supported), VK_SUCCESS);
    CHECK(supported);
    REQUIRE_VK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps));
    CHECK(caps.currentExtent.width == 0xFFFFFFFFu && caps.currentExtent.height == 0xFFFFFFFFu);
    CHECK(caps.minImageCount >= 1 && caps.maxImageCount >= caps.minImageCount);
    count = 4;
    REQUIRE_VK(vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &count, modes));
    CHECK(count == 2 && modes[0] == VK_PRESENT_MODE_FIFO_KHR && modes[1] == VK_PRESENT_MODE_IMMEDIATE_KHR);
    info = swapchain_info(surface, VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, 48, 24, 2);
    REQUIRE_VK(vkCreateSwapchainKHR(device, &info, NULL, &swapchain));
    CHECK_VK(vkCreateSwapchainKHR(device, &info, NULL, &second), VK_ERROR_NATIVE_WINDOW_IN_USE_KHR);
    info.surface = other; /* another headless surface takes one of its own */
    CHECK_VK(vkCreateSwapchainKHR(device, &info, NULL, &second), VK_SUCCESS);
    vkDestroySwapchainKHR(device, second, NULL);
    count = 4;
    REQUIRE_VK(vkGetSwapchainImagesKHR(device, swapchain, &count, images));
    for (frame = 0; frame < 20; frame++) {
        VkCommandBuffer cb;
        index = UINT32_MAX;
        REQUIRE_VK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, acquired, fence, &index));
        REQUIRE(index < count);
        REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
        REQUIRE_VK(vkResetFences(device, 1, &fence));
        cb = begin(0);
        REQUIRE(cb);
        record_clear(cb, images[index], 0.25f, 0.5f, (float)frame / 20.0f, 1.0f);
        REQUIRE_VK(submit(0, cb, acquired, rendered, frame == 19 ? done : VK_NULL_HANDLE));
        CHECK(no_backlog(0)); /* the acquire's signal is a present's on this queue */
        CHECK_VK(present(0, swapchain, index, rendered), VK_SUCCESS);
        CHECK(no_backlog(0)); /* rendered: a submission's on this queue */
    }
    CHECK_VK(vkWaitForFences(device, 1, &done, VK_TRUE, 2000000000ull), VK_SUCCESS);
    CHECK_VK(vkQueueWaitIdle(queues[0]), VK_SUCCESS);
    vkResetCommandPool(device, pools[0], 0);
    vkDestroySwapchainKHR(device, swapchain, NULL);
    vkDestroySemaphore(device, acquired, NULL);
    vkDestroySemaphore(device, rendered, NULL);
    vkDestroyFence(device, fence, NULL);
    vkDestroyFence(device, done, NULL);
    vkDestroySurfaceKHR(instance, surface, NULL);
    vkDestroySurfaceKHR(instance, other, NULL);
}

/* What reaches the drawable is the image, texel for texel (a present draws
 * it, reading texels as floats): a buffer copied into the image, presented,
 * read back from the drawable's texture. grow: the drawable is that much
 * larger (or smaller) than the image, the layer resized after the present
 * found it in date: the drawable's corner is the image's. */
static void check_contents(VkFormat format, VkColorSpaceKHR space, uint32_t texel_bytes, uint32_t family, int grow)
{
    const uint32_t w = 40, h = 24, rw = grow < 0 ? w - (uint32_t)-grow : w, rh = grow < 0 ? h - (uint32_t)-grow : h;
    id layer = make_layer(w, h, 1.0);
    VkSurfaceKHR surface = make_surface(layer);
    VkSwapchainCreateInfoKHR info = swapchain_info(surface, format, space, w, h, 2);
    VkBufferCreateInfo binfo;
    VkMemoryAllocateInfo alloc;
    VkMemoryRequirements req;
    VkBufferImageCopy region;
    VkSwapchainKHR swapchain;
    VkImage images[3];
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkFence fence = make_fence();
    VkSemaphore rendered = make_semaphore(VK_SEMAPHORE_TYPE_BINARY);
    VkCommandBuffer cb;
    unsigned char *data = NULL, *readback;
    uint32_t count = 3, index, i, y;
    pvk_MTLRegion r;

    REQUIRE(surface);
    REQUIRE_VK(vkCreateSwapchainKHR(device, &info, NULL, &swapchain));
    REQUIRE_VK(vkGetSwapchainImagesKHR(device, swapchain, &count, images));
    /* The swapchain's drawables are framebufferOnly, which the CPU cannot
     * read; these may be (the present draws into either kind). */
    PVK_MSG(void, (id, SEL, bool))(layer, sel_registerName("setFramebufferOnly:"), false);
    memset(&binfo, 0, sizeof(binfo));
    binfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    binfo.size = (VkDeviceSize)w * h * texel_bytes;
    binfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    REQUIRE_VK(vkCreateBuffer(device, &binfo, NULL, &buffer));
    vkGetBufferMemoryRequirements(device, buffer, &req);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = shared_type;
    REQUIRE_VK(vkAllocateMemory(device, &alloc, NULL, &memory));
    REQUIRE_VK(vkBindBufferMemory(device, buffer, memory, 0));
    REQUIRE_VK(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, (void **)&data));
    for (i = 0; i < w * h * texel_bytes; i++)
        data[i] = (unsigned char)(i * 7 + i / 13);
    if (format == VK_FORMAT_R16G16B16A16_SFLOAT) /* finite halves: exponents below 31 */
        for (i = 1; i < w * h * texel_bytes; i += 2)
            data[i] &= 0x3F;

    REQUIRE_VK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, VK_NULL_HANDLE, fence, &index));
    REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
    cb = begin(family);
    REQUIRE(cb);
    memset(&region, 0, sizeof(region));
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width = w;
    region.imageExtent.height = h;
    region.imageExtent.depth = 1;
    barrier(cb, images[index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdCopyBufferToImage(cb, buffer, images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    barrier(cb, images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    REQUIRE_VK(submit(family, cb, VK_NULL_HANDLE, rendered, VK_NULL_HANDLE));
    pvk_release(last_drawable);
    last_drawable = NULL;
    grow_next = grow;
    CHECK_VK(present(family, swapchain, index, rendered), VK_SUCCESS);
    CHECK_VK(vkQueueWaitIdle(queues[family]), VK_SUCCESS);
    REQUIRE(last_drawable != NULL);
    CHECK(pvk_MTLTexture_width(pvk_CAMetalDrawable_texture(last_drawable)) == (unsigned long)((int)w + grow));
    readback = calloc(1, rw * rh * texel_bytes);
    REQUIRE(readback);
    memset(&r, 0, sizeof(r));
    r.size.width = rw;
    r.size.height = rh;
    r.size.depth = 1;
    pvk_MTLTexture_getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
        pvk_CAMetalDrawable_texture(last_drawable), readback, rw * texel_bytes, rw * rh * texel_bytes, r, 0, 0);
    for (y = 0; y < rh; y++)
        CHECK(memcmp(readback + y * rw * texel_bytes, data + y * w * texel_bytes, rw * texel_bytes) == 0);
    free(readback);
    pvk_release(last_drawable);
    last_drawable = NULL;

    vkResetCommandPool(device, pools[family], 0);
    vkDestroySwapchainKHR(device, swapchain, NULL);
    vkDestroyBuffer(device, buffer, NULL);
    vkFreeMemory(device, memory, NULL);
    vkDestroyFence(device, fence, NULL);
    vkDestroySemaphore(device, rendered, NULL);
    vkDestroySurfaceKHR(instance, surface, NULL);
    pvk_release(layer);
}

static void test_present_contents(void)
{
    check_contents(VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, 4, 0, 0);
    check_contents(VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, 4, 0, 0);
    check_contents(VK_FORMAT_A2R10G10B10_UNORM_PACK32, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, 4, 0, 0);
    check_contents(VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT, 8, 0, 0);
    /* the compute family presents */
    check_contents(VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, 4, 1, 0);
    /* drawables larger and smaller than the image */
    check_contents(VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, 4, 0, 8);
    check_contents(VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, 4, 0, -8);
}

/* Resizing: bounds -> suboptimal, drawableSize -> out of date; recreation
 * with oldSwapchain; presenting an image of the retired swapchain. */
static void test_resize(void)
{
    id layer = make_layer(32, 32, 1.0);
    VkSurfaceKHR surface = make_surface(layer);
    VkSwapchainCreateInfoKHR info = swapchain_info(surface, VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                                                   32, 32, 2);
    VkSwapchainKHR swapchain, newer, third;
    VkImage images[3];
    VkFence fence = make_fence();
    VkCommandBuffer cb;
    uint32_t index, held, count = 3;
    pvk_CGRect frame;
    pvk_CGSize size;

    REQUIRE(surface);
    REQUIRE_VK(vkCreateSwapchainKHR(device, &info, NULL, &swapchain));
    REQUIRE_VK(vkGetSwapchainImagesKHR(device, swapchain, &count, images));
    /* The window grows: presents still work, scaled. */
    memset(&frame, 0, sizeof(frame));
    frame.size.width = 50;
    frame.size.height = 40;
    PVK_MSG(void, (id, SEL, pvk_CGRect))(layer, sel_registerName("setFrame:"), frame);
    CHECK_VK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, VK_NULL_HANDLE, fence, &index), VK_SUBOPTIMAL_KHR);
    REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
    vkResetFences(device, 1, &fence);
    cb = begin(0);
    record_clear(cb, images[index], 1, 0, 0, 1);
    REQUIRE_VK(submit(0, cb, VK_NULL_HANDLE, VK_NULL_HANDLE, fence));
    REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
    vkResetFences(device, 1, &fence);
    CHECK_VK(present(0, swapchain, index, VK_NULL_HANDLE), VK_SUBOPTIMAL_KHR);

    /* Hold an image, then a view resizes the drawables: out of date. */
    CHECK_VK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, VK_NULL_HANDLE, fence, &held), VK_SUBOPTIMAL_KHR);
    REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
    vkResetFences(device, 1, &fence);
    size.width = 50;
    size.height = 40;
    pvk_CAMetalLayer_setDrawableSize(layer, size);
    CHECK_VK(vkAcquireNextImageKHR(device, swapchain, 0, VK_NULL_HANDLE, fence, &index), VK_ERROR_OUT_OF_DATE_KHR);

    /* Without oldSwapchain the layer is in use; with it, a new swapchain. */
    info.imageExtent.width = 50;
    info.imageExtent.height = 40;
    CHECK_VK(vkCreateSwapchainKHR(device, &info, NULL, &newer), VK_ERROR_NATIVE_WINDOW_IN_USE_KHR);
    info.oldSwapchain = swapchain;
    REQUIRE_VK(vkCreateSwapchainKHR(device, &info, NULL, &newer));
    CHECK_VK(vkAcquireNextImageKHR(device, newer, UINT64_MAX, VK_NULL_HANDLE, fence, &index), VK_SUCCESS);
    REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
    vkResetFences(device, 1, &fence);
    count = 3;
    REQUIRE_VK(vkGetSwapchainImagesKHR(device, newer, &count, images));
    cb = begin(0);
    record_clear(cb, images[index], 0, 1, 0, 1);
    REQUIRE_VK(submit(0, cb, VK_NULL_HANDLE, VK_NULL_HANDLE, fence));
    REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
    vkResetFences(device, 1, &fence);
    CHECK_VK(present(0, newer, index, VK_NULL_HANDLE), VK_SUCCESS);
    /* The image held from the retired swapchain no longer fits the layer. */
    CHECK_VK(present(0, swapchain, held, VK_NULL_HANDLE), VK_ERROR_OUT_OF_DATE_KHR);
    vkDestroySwapchainKHR(device, swapchain, NULL);

    /* A second retirement, at the same size: no layer change. */
    info.oldSwapchain = newer;
    REQUIRE_VK(vkCreateSwapchainKHR(device, &info, NULL, &third));
    CHECK_VK(vkAcquireNextImageKHR(device, third, UINT64_MAX, VK_NULL_HANDLE, fence, &index), VK_SUCCESS);
    REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
    CHECK_VK(vkQueueWaitIdle(queues[0]), VK_SUCCESS);
    vkResetCommandPool(device, pools[0], 0);
    vkDestroySwapchainKHR(device, newer, NULL);
    vkDestroySwapchainKHR(device, third, NULL); /* with an image acquired: allowed */
    vkDestroyFence(device, fence, NULL);
    vkDestroySurfaceKHR(instance, surface, NULL);
    pvk_release(layer);
}

/* Timeouts: with every image held nothing comes back; a presented image is
 * returned at once, its fence and semaphore signalled when the copy is done. */
static void test_acquire_timeouts(void)
{
    id layer = make_layer(16, 16, 1.0);
    VkSurfaceKHR surface = make_surface(layer);
    VkSwapchainCreateInfoKHR info = swapchain_info(surface, VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                                                   16, 16, 2);
    VkSwapchainKHR swapchain;
    VkFence fences[3];
    VkSemaphore sem = make_semaphore(VK_SEMAPHORE_TYPE_BINARY);
    VkImage images[2];
    VkCommandBuffer cb;
    uint32_t idx[3], count = 2, i, index;
    uint64_t start;

    REQUIRE(surface);
    for (i = 0; i < 3; i++)
        fences[i] = make_fence();
    REQUIRE_VK(vkCreateSwapchainKHR(device, &info, NULL, &swapchain));
    REQUIRE_VK(vkGetSwapchainImagesKHR(device, swapchain, &count, images));
    for (i = 0; i < 2; i++) {
        CHECK_VK(vkAcquireNextImageKHR(device, swapchain, 0, VK_NULL_HANDLE, fences[i], &idx[i]), VK_SUCCESS);
        CHECK_VK(vkWaitForFences(device, 1, &fences[i], VK_TRUE, 0), VK_SUCCESS); /* never presented: free */
    }
    CHECK(idx[0] != idx[1]);
    CHECK_VK(vkAcquireNextImageKHR(device, swapchain, 0, VK_NULL_HANDLE, fences[2], &index), VK_NOT_READY);
    start = now_ns();
    CHECK_VK(vkAcquireNextImageKHR(device, swapchain, 3000000, VK_NULL_HANDLE, fences[2], &index), VK_TIMEOUT);
    CHECK(now_ns() - start >= 3000000);
    CHECK_VK(vkGetFenceStatus(device, fences[2]), VK_NOT_READY); /* untouched */

    /* Present both; acquire again right away: images on their way back. */
    cb = begin(0);
    record_clear(cb, images[idx[0]], 0, 0, 1, 1);
    record_clear(cb, images[idx[1]], 0, 0, 1, 1);
    REQUIRE_VK(submit(0, cb, VK_NULL_HANDLE, VK_NULL_HANDLE, fences[2]));
    REQUIRE_VK(vkWaitForFences(device, 1, &fences[2], VK_TRUE, 2000000000ull));
    vkResetFences(device, 3, fences);
    CHECK_VK(present(0, swapchain, idx[0], VK_NULL_HANDLE), VK_SUCCESS);
    CHECK_VK(present(0, swapchain, idx[1], VK_NULL_HANDLE), VK_SUCCESS);
    CHECK_VK(vkAcquireNextImageKHR(device, swapchain, 0, VK_NULL_HANDLE, fences[0], &index), VK_SUCCESS);
    CHECK(index == idx[0]); /* presented first */
    CHECK_VK(vkAcquireNextImageKHR(device, swapchain, 0, sem, VK_NULL_HANDLE, &index), VK_SUCCESS);
    CHECK(index == idx[1]);
    CHECK_VK(vkWaitForFences(device, 1, &fences[0], VK_TRUE, 2000000000ull), VK_SUCCESS);
    /* The semaphore: a submission waits for it. */
    cb = begin(0);
    record_clear(cb, images[idx[1]], 1, 1, 1, 1);
    REQUIRE_VK(submit(0, cb, sem, VK_NULL_HANDLE, fences[1]));
    CHECK_VK(vkWaitForFences(device, 1, &fences[1], VK_TRUE, 2000000000ull), VK_SUCCESS);

    CHECK_VK(vkQueueWaitIdle(queues[0]), VK_SUCCESS);
    vkResetCommandPool(device, pools[0], 0);
    vkDestroySwapchainKHR(device, swapchain, NULL);
    for (i = 0; i < 3; i++)
        vkDestroyFence(device, fences[i], NULL);
    vkDestroySemaphore(device, sem, NULL);
    vkDestroySurfaceKHR(instance, surface, NULL);
    pvk_release(layer);
}

/* A present whose semaphore is signalled late (behind a timeline wait the
 * host resolves): the queue's thread presents. Acquiring the image back
 * returns before that; its fence waits for the copy. */
static void test_deferred_present(void)
{
    id layer = make_layer(16, 16, 1.0);
    VkSurfaceKHR surface = make_surface(layer);
    VkSwapchainCreateInfoKHR info = swapchain_info(surface, VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                                                   16, 16, 2);
    VkSemaphore timeline = make_semaphore(VK_SEMAPHORE_TYPE_TIMELINE), rendered = make_semaphore(VK_SEMAPHORE_TYPE_BINARY);
    VkTimelineSemaphoreSubmitInfo values;
    VkSemaphoreSignalInfo signal;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit_info;
    VkSwapchainKHR swapchain;
    VkImage images[2];
    VkFence fence = make_fence();
    VkCommandBuffer cb;
    uint64_t one = 1, zero = 0;
    uint32_t count = 2, index, other, again;

    REQUIRE(surface);
    REQUIRE_VK(vkCreateSwapchainKHR(device, &info, NULL, &swapchain));
    REQUIRE_VK(vkGetSwapchainImagesKHR(device, swapchain, &count, images));
    REQUIRE_VK(vkAcquireNextImageKHR(device, swapchain, 0, VK_NULL_HANDLE, fence, &index));
    REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
    vkResetFences(device, 1, &fence);
    cb = begin(0);
    record_clear(cb, images[index], 1, 0, 1, 1);
    vkEndCommandBuffer(cb);
    memset(&values, 0, sizeof(values));
    values.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    values.waitSemaphoreValueCount = 1;
    values.pWaitSemaphoreValues = &one;
    values.signalSemaphoreValueCount = 1;
    values.pSignalSemaphoreValues = &zero;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.pNext = &values;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &timeline;
    submit_info.pWaitDstStageMask = &stage;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cb;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &rendered;
    REQUIRE_VK(vkQueueSubmit(queues[0], 1, &submit_info, VK_NULL_HANDLE));
    CHECK_VK(present(0, swapchain, index, rendered), VK_SUCCESS);
    /* The other image, then this one again: it is not back yet. */
    REQUIRE_VK(vkAcquireNextImageKHR(device, swapchain, 0, VK_NULL_HANDLE, fence, &other));
    CHECK(other != index);
    vkResetFences(device, 1, &fence);
    CHECK_VK(vkAcquireNextImageKHR(device, swapchain, 0, VK_NULL_HANDLE, fence, &again), VK_SUCCESS);
    CHECK(again == index);
    CHECK_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 20000000), VK_TIMEOUT);
    memset(&signal, 0, sizeof(signal));
    signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal.semaphore = timeline;
    signal.value = 1;
    REQUIRE_VK(vkSignalSemaphore(device, &signal));
    CHECK_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull), VK_SUCCESS);
    CHECK_VK(vkQueueWaitIdle(queues[0]), VK_SUCCESS);
    vkResetCommandPool(device, pools[0], 0);
    vkDestroySwapchainKHR(device, swapchain, NULL);
    vkDestroySemaphore(device, timeline, NULL);
    vkDestroySemaphore(device, rendered, NULL);
    vkDestroyFence(device, fence, NULL);
    vkDestroySurfaceKHR(instance, surface, NULL);
    pvk_release(layer);
}

/* A frame loop that never waits on the host: acquire, clear, present, two
 * images. First on one queue, where every wait is covered by its order
 * (nothing reaches the queue's thread), then rendering and presenting on
 * alternating queues (acquires released on the other queue). The last
 * drawable shows the last frame. */
static void test_frame_loop(void)
{
    enum { FRAMES = 16, W = 256, H = 256 };
    id layer = make_layer(W, H, 1.0);
    VkSurfaceKHR surface = make_surface(layer);
    VkSwapchainCreateInfoKHR info = swapchain_info(surface, VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                                                   W, H, 2);
    VkSemaphore acquired[FRAMES], rendered[FRAMES];
    VkSwapchainKHR swapchain;
    VkImage images[2];
    unsigned char texel[4] = { 0, 0, 0, 0 };
    uint32_t count = 2, frame, index;
    pvk_MTLRegion r;

    REQUIRE(surface);
    REQUIRE_VK(vkCreateSwapchainKHR(device, &info, NULL, &swapchain));
    REQUIRE_VK(vkGetSwapchainImagesKHR(device, swapchain, &count, images));
    PVK_MSG(void, (id, SEL, bool))(layer, sel_registerName("setFramebufferOnly:"), false); /* read back */
    for (frame = 0; frame < FRAMES; frame++) {
        bool one_queue = frame < FRAMES / 2;
        uint32_t render = one_queue ? 0 : frame & 1, show = one_queue ? 0 : (frame >> 1) & 1;
        float t = (float)frame / (FRAMES - 1);
        VkCommandBuffer cb;
        acquired[frame] = make_semaphore(VK_SEMAPHORE_TYPE_BINARY);
        rendered[frame] = make_semaphore(VK_SEMAPHORE_TYPE_BINARY);
        REQUIRE_VK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, acquired[frame], VK_NULL_HANDLE, &index));
        cb = begin(render);
        REQUIRE(cb);
        record_clear(cb, images[index], 0.0f, t, 1.0f - t, 1.0f);
        REQUIRE_VK(submit(render, cb, acquired[frame], rendered[frame], VK_NULL_HANDLE));
        if (one_queue)
            CHECK(no_backlog(0));
        CHECK_VK(present(show, swapchain, index, rendered[frame]), VK_SUCCESS);
        if (one_queue)
            CHECK(no_backlog(0));
    }
    CHECK_VK(vkDeviceWaitIdle(device), VK_SUCCESS);
    REQUIRE(last_drawable != NULL);
    memset(&r, 0, sizeof(r));
    r.origin.x = W / 2;
    r.origin.y = H / 2;
    r.size.width = r.size.height = r.size.depth = 1;
    pvk_MTLTexture_getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
        pvk_CAMetalDrawable_texture(last_drawable), texel, 4, 4, r, 0, 0);
    CHECK(texel[0] == 0 && texel[1] == 255 && texel[2] == 0 && texel[3] == 255); /* BGRA */
    pvk_release(last_drawable);
    last_drawable = NULL;
    vkResetCommandPool(device, pools[0], 0);
    vkResetCommandPool(device, pools[1], 0);
    vkDestroySwapchainKHR(device, swapchain, NULL);
    for (frame = 0; frame < FRAMES; frame++) {
        vkDestroySemaphore(device, acquired[frame], NULL);
        vkDestroySemaphore(device, rendered[frame], NULL);
    }
    vkDestroySurfaceKHR(instance, surface, NULL);
    pvk_release(layer);
}

/* ---- Drawables only render passes read ---- */

/* Runs copies of contents on a command queue of its own and waits for
 * them (white box: section 16's pvk_wsi_encode_copies, or with fetch its
 * render pass for each, pvk_wsi_fetch_copy); residency: the layer's set,
 * for drawables' textures (NULL: none). */
static void run_copies(const struct pvk_wsi_copy *copies, uint32_t count, id residency, bool fetch)
{
    struct pvk_device *d = (struct pvk_device *)device;
    id mtl = d->physical_device->mtl_device, queue = pvk_MTLDevice_newMTL4CommandQueue(mtl);
    id allocator = pvk_MTLDevice_newCommandAllocator(mtl), cb = pvk_MTLDevice_newCommandBuffer(mtl);
    id event = pvk_MTLDevice_newSharedEvent(mtl);
    uint32_t i;

    if (queue && allocator && cb && event) {
        pvk_MTL4CommandQueue_addResidencySet(queue, d->residency_set);
        pvk_device_commit_residency(d);
        pvk_MTL4CommandBuffer_beginCommandBufferWithAllocator(cb, allocator);
        if (residency)
            pvk_MTL4CommandBuffer_useResidencySet(cb, residency);
        if (fetch)
            for (i = 0; i < count; i++)
                CHECK(pvk_wsi_fetch_copy(d, cb, &copies[i]));
        else
            CHECK(pvk_wsi_encode_copies(d, cb, copies, count));
        pvk_MTL4CommandBuffer_endCommandBuffer(cb);
        pvk_MTL4CommandQueue_commit_count(queue, &cb, 1);
        pvk_MTL4CommandQueue_signalEvent_value(queue, event, 1);
        CHECK(pvk_MTLSharedEvent_waitUntilSignaledValue_timeoutMS(event, 1, 5000));
    } else {
        PVK_TEST_FAIL("no command queue for copies");
    }
    pvk_release(event);
    pvk_release(cb);
    pvk_release(allocator);
    pvk_release(queue);
}

/* A w x h texture that can be a render target, in memory the CPU reads,
 * resident (the device's set). */
static id shared_texture(pvk_MTLPixelFormat format, uint32_t w, uint32_t h)
{
    struct pvk_device *d = (struct pvk_device *)device;
    id desc = pvk_MTLTextureDescriptor_new(), t = NULL;
    if (desc) {
        pvk_MTLTextureDescriptor_setPixelFormat(desc, format);
        pvk_MTLTextureDescriptor_setWidth(desc, w);
        pvk_MTLTextureDescriptor_setHeight(desc, h);
        pvk_MTLTextureDescriptor_setUsage(desc, PVK_MTLTextureUsageRenderTarget);
        pvk_MTLTextureDescriptor_setResourceOptions(desc, PVK_MTLResourceStorageModeShared);
        t = pvk_MTLDevice_newTextureWithDescriptor(d->physical_device->mtl_device, desc);
        pvk_release(desc);
    }
    if (t)
        pvk_device_make_resident(d, t);
    return t;
}

static void free_shared_texture(id t)
{
    pvk_device_evict((struct pvk_device *)device, t);
    pvk_release(t);
}

/* The texels of a drawable's texture of w x h, 4 bytes each. A
 * framebufferOnly one (images that are only color attachments render to
 * those) is read by no CPU, copy or shader, only by a render pass: one
 * copies it into a shared texture first, as PlasmaVK's copies of contents
 * do. */
static void read_drawable(id drawable, id layer, void *texels, uint32_t w, uint32_t h)
{
    id texture = pvk_CAMetalDrawable_texture(drawable), readable = texture;
    pvk_MTLRegion r;
    if (pvk_MTLTexture_isFramebufferOnly(texture)) {
        struct pvk_wsi_copy c;
        REQUIRE(pvk_MTLTexture_width(texture) == w && pvk_MTLTexture_height(texture) == h);
        readable = shared_texture(pvk_MTLTexture_pixelFormat(texture), w, h);
        REQUIRE(readable);
        c.src = texture;
        c.dst = readable;
        c.residency = NULL;
        run_copies(&c, 1, pvk_CAMetalLayer_residencySet(layer), false);
    }
    memset(&r, 0, sizeof(r));
    r.size.width = w;
    r.size.height = h;
    r.size.depth = 1;
    pvk_MTLTexture_getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(readable, texels, w * 4, w * h * 4,
                                                                                   r, 0, 0);
    if (readable != texture)
        free_shared_texture(readable);
}

/* Copies of contents through a render pass (pvk_wsi_fetch_copy: the
 * fragments read the source attachment's pixel as floats and write them)
 * are exact in every swapchain format, for texels of every value. */
static void test_fetch_copies(void)
{
    static const struct {
        pvk_MTLPixelFormat format;
        uint32_t bytes;
    } formats[] = {
        { PVK_MTLPixelFormatBGRA8Unorm, 4 },    { PVK_MTLPixelFormatBGRA8Unorm_sRGB, 4 },
        { PVK_MTLPixelFormatRGBA16Float, 8 },   { PVK_MTLPixelFormatRGB10A2Unorm, 4 },
        { PVK_MTLPixelFormatBGR10A2Unorm, 4 },
    };
    enum { W = 64, H = 32 };
    static unsigned char data[W * H * 8], readback[W * H * 8];
    uint32_t f, i, y;
    pvk_MTLRegion r;

    memset(&r, 0, sizeof(r));
    r.size.width = W;
    r.size.height = H;
    r.size.depth = 1;
    for (f = 0; f < sizeof(formats) / sizeof(formats[0]); f++) {
        uint32_t row = W * formats[f].bytes;
        struct pvk_wsi_copy c;
        c.src = shared_texture(formats[f].format, W, H);
        c.dst = shared_texture(formats[f].format, W, H);
        c.residency = NULL;
        REQUIRE(c.src && c.dst);
        for (i = 0; i < row * H; i++)
            data[i] = (unsigned char)(i * 7 + i / 13 + f);
        if (formats[f].bytes == 8) /* finite halves of either sign: exponents up to 15 */
            for (i = 1; i < row * H; i += 2)
                data[i] &= 0xBF;
        pvk_MTLTexture_replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(c.src, r, 0, 0, data, row,
                                                                                           row * H);
        run_copies(&c, 1, NULL, true);
        pvk_MTLTexture_getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(c.dst, readback, row, row * H, r,
                                                                                       0, 0);
        for (y = 0; y < H; y++)
            if (memcmp(readback + y * row, data + y * row, row) != 0) {
                PVK_TEST_FAIL("pixel format %lu: row %u differs after a copy", formats[f].format, y);
                break;
            }
        pvk_MTLTexture_getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(c.src, readback, row, row * H, r,
                                                                                       0, 0);
        CHECK(memcmp(readback, data, row * H) == 0); /* the source as it was */
        free_shared_texture(c.src);
        free_shared_texture(c.dst);
    }
}

/* Images without sampled, storage or input attachment usage render to the
 * drawables they are presented with (section 18, SWAPCHAIN IMAGES), and
 * keep their contents from a present to their next acquire, whatever
 * drawable that renders to next: frames that discard the contents, that
 * need them and that leave the image alone, in turns, each presented
 * drawable read back. With transfers: discarding by a transition from
 * UNDEFINED, a LOAD_OP_CLEAR render pass or a scaled blit into all of it
 * (as the engine presents), reading by a copy to a buffer (or a
 * LOAD_OP_LOAD render pass, then a copy), each read checked. Without, the
 * images only color attachments (their drawables framebufferOnly, which
 * render passes copy): discarding by a transition from UNDEFINED or a
 * LOAD_OP_CLEAR or DONT_CARE render pass, needing the contents for a
 * LOAD_OP_LOAD render pass that clears half of it or a render pass over
 * half of it, the other half checked in the drawable. Every lazy-th acquire
 * has a finite timeout, so the first use (or the present) takes the
 * drawable, whose IOSurface may have other contents: lazy 1 has their
 * copies in most frames. */
static void check_direct(uint32_t images, uint32_t family, uint32_t lazy_every, bool transfers)
{
    enum { W = 16, H = 8, FRAMES = 40 };
    id layer = make_layer(W, H, 1.0);
    VkSurfaceKHR surface = make_surface(layer);
    VkSwapchainCreateInfoKHR info = swapchain_info(surface, VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                                                   W, H, images);
    VkBufferCreateInfo binfo;
    VkMemoryAllocateInfo alloc;
    VkMemoryRequirements req;
    VkBufferImageCopy region;
    VkImageViewCreateInfo vinfo;
    VkSwapchainKHR swapchain;
    VkImage image[3], source;
    VkImageView view[3];
    VkImageCreateInfo sinfo;
    VkBuffer buffer;
    VkDeviceMemory memory, source_memory;
    VkFence fence = make_fence();
    VkSemaphore acquired = make_semaphore(VK_SEMAPHORE_TYPE_BINARY), rendered = make_semaphore(VK_SEMAPHORE_TYPE_BINARY);
    uint32_t want[3][W * H]; /* each image's contents (BGRA bytes), 0: undefined */
    uint32_t count = 3, frame, index, i, *data = NULL, texels[W * H];

    memset(want, 0, sizeof(want));
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (transfers)
        info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.clipped = VK_FALSE;
    REQUIRE(surface);
    REQUIRE_VK(vkCreateSwapchainKHR(device, &info, NULL, &swapchain));
    CHECK(((struct pvk_swapchain *)swapchain)->direct); /* (white box) */
    CHECK(PVK_MSG(bool, (id, SEL))(layer, sel_registerName("framebufferOnly")) == !transfers);
    REQUIRE_VK(vkGetSwapchainImagesKHR(device, swapchain, &count, image));
    REQUIRE(count == images);
    for (i = 0; i < count; i++) {
        memset(&vinfo, 0, sizeof(vinfo));
        vinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vinfo.image = image[i];
        vinfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vinfo.format = VK_FORMAT_B8G8R8A8_UNORM;
        vinfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vinfo.subresourceRange.levelCount = 1;
        vinfo.subresourceRange.layerCount = 1;
        REQUIRE_VK(vkCreateImageView(device, &vinfo, NULL, &view[i]));
    }
    CHECK(((struct pvk_swapchain *)swapchain)->direct); /* (the views are the images' textures) */
    memset(&binfo, 0, sizeof(binfo));
    binfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    binfo.size = sizeof(texels);
    binfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    REQUIRE_VK(vkCreateBuffer(device, &binfo, NULL, &buffer));
    vkGetBufferMemoryRequirements(device, buffer, &req);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = shared_type;
    REQUIRE_VK(vkAllocateMemory(device, &alloc, NULL, &memory));
    REQUIRE_VK(vkBindBufferMemory(device, buffer, memory, 0));
    REQUIRE_VK(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, (void **)&data));
    memset(&region, 0, sizeof(region));
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width = W;
    region.imageExtent.height = H;
    region.imageExtent.depth = 1;
    memset(&sinfo, 0, sizeof(sinfo)); /* the blits' source: twice the size, one color */
    sinfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    sinfo.imageType = VK_IMAGE_TYPE_2D;
    sinfo.format = VK_FORMAT_B8G8R8A8_UNORM;
    sinfo.extent.width = 2 * W;
    sinfo.extent.height = 2 * H;
    sinfo.extent.depth = 1;
    sinfo.mipLevels = sinfo.arrayLayers = 1;
    sinfo.samples = VK_SAMPLE_COUNT_1_BIT;
    sinfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    REQUIRE_VK(vkCreateImage(device, &sinfo, NULL, &source));
    vkGetImageMemoryRequirements(device, source, &req);
    alloc.allocationSize = req.size;
    for (alloc.memoryTypeIndex = 0; !(req.memoryTypeBits & (1u << alloc.memoryTypeIndex)); alloc.memoryTypeIndex++)
        ;
    REQUIRE_VK(vkAllocateMemory(device, &alloc, NULL, &source_memory));
    REQUIRE_VK(vkBindImageMemory(device, source, source_memory, 0));

    for (frame = 0; frame < FRAMES; frame++) {
        uint32_t mode = frame % 6, next = 0xff000000u | (frame * 0x050b11u + 0x102030u), k;
        uint32_t before[W * H]; /* the image's contents before the frame */
        bool use = true, read = false, lazy = frame % lazy_every == lazy_every - 1;
        VkClearColorValue clear;
        VkImageSubresourceRange range;
        VkCommandBuffer cb;

        clear.float32[0] = (float)((next >> 16) & 255) / 255.0f; /* (BGRA bytes: R is byte 2) */
        clear.float32[1] = (float)((next >> 8) & 255) / 255.0f;
        clear.float32[2] = (float)(next & 255) / 255.0f;
        clear.float32[3] = 1.0f;
        memset(&range, 0, sizeof(range));
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.levelCount = 1;
        range.layerCount = 1;
        pvk_release(last_drawable);
        last_drawable = NULL;
        REQUIRE_VK(vkAcquireNextImageKHR(device, swapchain, lazy ? 2000000000ull : UINT64_MAX, acquired, VK_NULL_HANDLE,
                                         &index));
        CHECK(lazy == (last_drawable == NULL));
        memcpy(before, want[index], sizeof(before));
        cb = begin(family);
        REQUIRE(cb);
        if (mode == 2 && before[0]) { /* left alone: presented as it was */
            use = false;
        } else if (!transfers) { /* render passes only */
            VkRenderingAttachmentInfo att;
            VkRenderingInfo ri;
            VkClearAttachment ca;
            VkClearRect rect;
            memset(&att, 0, sizeof(att));
            att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            att.imageView = view[index];
            att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            /* 0: discards (a transition from UNDEFINED), then clears all; 1: loads, clears the left half;
             * 4: clears the top half (the render area); 5: discards (DONT_CARE), clears all; 3: clears all */
            att.loadOp = mode == 0 || mode == 1 ? VK_ATTACHMENT_LOAD_OP_LOAD
                         : mode == 5         ? VK_ATTACHMENT_LOAD_OP_DONT_CARE
                                             : VK_ATTACHMENT_LOAD_OP_CLEAR;
            att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att.clearValue.color = clear;
            memset(&ri, 0, sizeof(ri));
            ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea.extent.width = W;
            ri.renderArea.extent.height = mode == 4 ? H / 2 : H;
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &att;
            memset(&ca, 0, sizeof(ca));
            ca.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ca.clearValue.color = clear;
            memset(&rect, 0, sizeof(rect));
            rect.rect.extent.width = mode == 1 ? W / 2 : W;
            rect.rect.extent.height = H;
            rect.layerCount = 1;
            barrier(cb, image[index], mode == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            vkCmdBeginRendering(cb, &ri);
            if (att.loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR)
                vkCmdClearAttachments(cb, 1, &ca, 1, &rect);
            vkCmdEndRendering(cb);
            barrier(cb, image[index], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
            for (k = 0; k < W * H; k++)
                if ((mode != 1 || k % W < W / 2) && (mode != 4 || k / W < H / 2))
                    want[index][k] = next;
        } else if (mode == 0) { /* discards: a clear after a transition from UNDEFINED */
            barrier(cb, image[index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            vkCmdClearColorImage(cb, image[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
            barrier(cb, image[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        } else if (mode == 1 || mode == 4) { /* reads: a copy (or a LOAD_OP_LOAD pass, then a copy), then a clear */
            if (mode == 4) {
                VkRenderingAttachmentInfo att;
                VkRenderingInfo ri;
                memset(&att, 0, sizeof(att));
                att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                att.imageView = view[index];
                att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                memset(&ri, 0, sizeof(ri));
                ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                ri.renderArea.extent.width = W;
                ri.renderArea.extent.height = H;
                ri.layerCount = 1;
                ri.colorAttachmentCount = 1;
                ri.pColorAttachments = &att;
                barrier(cb, image[index], VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                vkCmdBeginRendering(cb, &ri);
                vkCmdEndRendering(cb);
                barrier(cb, image[index], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            } else {
                barrier(cb, image[index], VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            }
            vkCmdCopyImageToBuffer(cb, image[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
            barrier(cb, image[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            vkCmdClearColorImage(cb, image[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
            barrier(cb, image[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
            read = true;
        } else if (mode == 5) { /* discards: a scaled blit into all of it */
            VkImageBlit blit;
            barrier(cb, source, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            vkCmdClearColorImage(cb, source, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
            barrier(cb, source, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            barrier(cb, image[index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            memset(&blit, 0, sizeof(blit));
            blit.srcSubresource = region.imageSubresource;
            blit.srcOffsets[1].x = 2 * W;
            blit.srcOffsets[1].y = 2 * H;
            blit.srcOffsets[1].z = 1;
            blit.dstSubresource = region.imageSubresource;
            blit.dstOffsets[1].x = W;
            blit.dstOffsets[1].y = H;
            blit.dstOffsets[1].z = 1;
            vkCmdBlitImage(cb, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image[index],
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
            barrier(cb, image[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        } else { /* discards: a LOAD_OP_CLEAR render pass */
            VkRenderingAttachmentInfo att;
            VkRenderingInfo ri;
            memset(&att, 0, sizeof(att));
            att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            att.imageView = view[index];
            att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att.clearValue.color = clear;
            memset(&ri, 0, sizeof(ri));
            ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea.extent.width = W;
            ri.renderArea.extent.height = H;
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &att;
            barrier(cb, image[index], VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            vkCmdBeginRendering(cb, &ri);
            vkCmdEndRendering(cb);
            barrier(cb, image[index], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        }
        if (transfers && use) /* (every frame that uses the image writes all of it) */
            for (k = 0; k < W * H; k++)
                want[index][k] = next;
        REQUIRE_VK(submit(family, cb, acquired, rendered, fence)); /* (empty when the image is left alone) */
        if (lazy)
            CHECK(use == (last_drawable != NULL));
        CHECK_VK(present(family, swapchain, index, rendered), VK_SUCCESS);
        REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
        REQUIRE_VK(vkResetFences(device, 1, &fence));
        CHECK_VK(vkQueueWaitIdle(queues[family]), VK_SUCCESS);
        if (read)
            for (k = 0; k < W * H; k++)
                if (before[k] && data[k] != before[k]) {
                    PVK_TEST_FAIL("frame %u (%u images, mode %u): image %u read 0x%08x at %u, not 0x%08x", frame,
                                  images, mode, index, data[k], k, before[k]);
                    break;
                }
        REQUIRE(last_drawable != NULL);
        read_drawable(last_drawable, layer, texels, W, H);
        for (k = 0; k < W * H; k++)
            if (want[index][k] && texels[k] != want[index][k]) {
                PVK_TEST_FAIL("frame %u (%u images, mode %u%s): image %u presented 0x%08x at %u, not 0x%08x", frame,
                              images, mode, transfers ? "" : ", color attachment only", index, texels[k], k,
                              want[index][k]);
                break;
            }
        pvk_release(last_drawable);
        last_drawable = NULL;
        vkResetCommandPool(device, pools[family], 0);
        /* read back once presented, not acquired (invalid usage, but the CTS's colorspace tests do) */
        if (transfers && mode == 0) {
            cb = begin(family);
            REQUIRE(cb);
            vkCmdCopyImageToBuffer(cb, image[index], VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, buffer, 1, &region);
            REQUIRE_VK(submit(family, cb, VK_NULL_HANDLE, VK_NULL_HANDLE, fence));
            REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
            REQUIRE_VK(vkResetFences(device, 1, &fence));
            for (k = 0; k < W * H; k++)
                if (data[k] != want[index][k]) {
                    PVK_TEST_FAIL("frame %u (%u images): image %u read 0x%08x at %u once presented, not 0x%08x", frame,
                                  images, index, data[k], k, want[index][k]);
                    break;
                }
            vkResetCommandPool(device, pools[family], 0);
        }
    }

    vkDestroySwapchainKHR(device, swapchain, NULL);
    for (i = 0; i < count; i++)
        vkDestroyImageView(device, view[i], NULL);
    vkDestroyImage(device, source, NULL);
    vkFreeMemory(device, source_memory, NULL);
    vkDestroyBuffer(device, buffer, NULL);
    vkFreeMemory(device, memory, NULL);
    vkDestroyFence(device, fence, NULL);
    vkDestroySemaphore(device, acquired, NULL);
    vkDestroySemaphore(device, rendered, NULL);
    vkDestroySurfaceKHR(instance, surface, NULL);
    pvk_release(layer);
}

static void test_direct(void)
{
    check_direct(2, 0, 3, true);
    check_direct(3, 0, 3, true);
    check_direct(2, 1, 3, true); /* the compute family */
    check_direct(2, 0, 1, true);
    check_direct(3, 0, 1, true);
    check_direct(2, 0, 3, false);
    check_direct(3, 0, 3, false);
    check_direct(2, 1, 3, false);
    check_direct(2, 0, 1, false);
    check_direct(3, 0, 1, false);
}

/* Transfers alone write the drawables, no render pass before them: copies
 * from a buffer into the images, of a layer that had no device before the
 * swapchain (the drawables' residency set comes with it). */
static void test_transfers_into_drawables(void)
{
    enum { W = 16, H = 8, FRAMES = 8 };
    id layer = make_layer(W, H, 1.0);
    VkSurfaceKHR surface = make_surface(layer);
    VkSwapchainCreateInfoKHR info = swapchain_info(surface, VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                                                   W, H, 2);
    VkSemaphore acquired = make_semaphore(VK_SEMAPHORE_TYPE_BINARY), rendered = make_semaphore(VK_SEMAPHORE_TYPE_BINARY);
    VkFence fence = make_fence();
    VkBufferCreateInfo binfo;
    VkMemoryAllocateInfo alloc;
    VkMemoryRequirements req;
    VkBufferImageCopy region;
    VkSwapchainKHR swapchain;
    VkDeviceMemory memory;
    VkBuffer buffer;
    VkImage images[2];
    uint32_t count = 2, frame, index, k, *data = NULL, texels[W * H];

    REQUIRE(surface);
    REQUIRE(pvk_CAMetalLayer_device(layer) == NULL);
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    REQUIRE_VK(vkCreateSwapchainKHR(device, &info, NULL, &swapchain));
    CHECK(((struct pvk_swapchain *)swapchain)->direct && ((struct pvk_swapchain *)swapchain)->residency); /* (white box) */
    REQUIRE_VK(vkGetSwapchainImagesKHR(device, swapchain, &count, images));
    memset(&binfo, 0, sizeof(binfo));
    binfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    binfo.size = sizeof(texels);
    binfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    REQUIRE_VK(vkCreateBuffer(device, &binfo, NULL, &buffer));
    vkGetBufferMemoryRequirements(device, buffer, &req);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = shared_type;
    REQUIRE_VK(vkAllocateMemory(device, &alloc, NULL, &memory));
    REQUIRE_VK(vkBindBufferMemory(device, buffer, memory, 0));
    REQUIRE_VK(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, (void **)&data));
    memset(&region, 0, sizeof(region));
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width = W;
    region.imageExtent.height = H;
    region.imageExtent.depth = 1;
    for (frame = 0; frame < FRAMES; frame++) {
        uint32_t want = 0xff000000u | (frame * 0x0b1d2fu + 0x405060u);
        VkCommandBuffer cb;
        pvk_release(last_drawable);
        last_drawable = NULL;
        REQUIRE_VK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &index));
        for (k = 0; k < W * H; k++)
            data[k] = want;
        cb = begin(0);
        REQUIRE(cb);
        barrier(cb, images[index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(cb, buffer, images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        barrier(cb, images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        REQUIRE_VK(submit(0, cb, acquired, rendered, fence));
        CHECK_VK(present(0, swapchain, index, rendered), VK_SUCCESS);
        REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
        REQUIRE_VK(vkResetFences(device, 1, &fence));
        CHECK_VK(vkQueueWaitIdle(queues[0]), VK_SUCCESS);
        REQUIRE(last_drawable != NULL);
        read_drawable(last_drawable, layer, texels, W, H);
        for (k = 0; k < W * H; k++)
            if (texels[k] != want) {
                PVK_TEST_FAIL("frame %u: image %u presented 0x%08x at %u, not 0x%08x", frame, index, texels[k], k, want);
                break;
            }
        pvk_release(last_drawable);
        last_drawable = NULL;
        vkResetCommandPool(device, pools[0], 0);
    }
    vkDestroySwapchainKHR(device, swapchain, NULL);
    vkDestroyBuffer(device, buffer, NULL);
    vkFreeMemory(device, memory, NULL);
    vkDestroyFence(device, fence, NULL);
    vkDestroySemaphore(device, acquired, NULL);
    vkDestroySemaphore(device, rendered, NULL);
    vkDestroySurfaceKHR(instance, surface, NULL);
    pvk_release(layer);
}

/* Present modes, color spaces, composite alpha and two swapchains in one
 * present (vkAcquireNextImage2KHR for one of them). */
static void test_layer_configuration(void)
{
    id layers[2] = { make_layer(16, 16, 1.0), make_layer(24, 8, 1.0) };
    VkSurfaceKHR surfaces[2] = { make_surface(layers[0]), make_surface(layers[1]) };
    VkSwapchainCreateInfoKHR info;
    VkAcquireNextImageInfoKHR acquire;
    VkPresentInfoKHR present_info;
    VkSwapchainKHR swapchains[2];
    VkImage images[2][3];
    VkFence fence = make_fence();
    VkResult results[2] = { VK_ERROR_UNKNOWN, VK_ERROR_UNKNOWN };
    VkCommandBuffer cb;
    uint32_t indices[2], count, i;

    REQUIRE(surfaces[0] && surfaces[1]);
    info = swapchain_info(surfaces[0], VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT, 16, 16, 3);
    info.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
    REQUIRE_VK(vkCreateSwapchainKHR(device, &info, NULL, &swapchains[0]));
    CHECK(!PVK_MSG(bool, (id, SEL))(layers[0], sel_registerName("displaySyncEnabled")));
    CHECK(PVK_MSG(bool, (id, SEL))(layers[0], sel_registerName("wantsExtendedDynamicRangeContent")));
    CHECK(PVK_MSG(void *, (id, SEL))(layers[0], sel_registerName("colorspace")) != NULL);
    CHECK(!PVK_MSG(bool, (id, SEL))(layers[0], sel_registerName("isOpaque")));
    info = swapchain_info(surfaces[1], VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_PASS_THROUGH_EXT, 24, 8, 2);
    REQUIRE_VK(vkCreateSwapchainKHR(device, &info, NULL, &swapchains[1]));
    CHECK(PVK_MSG(void *, (id, SEL))(layers[1], sel_registerName("colorspace")) == NULL);
    CHECK(PVK_MSG(bool, (id, SEL))(layers[1], sel_registerName("isOpaque")));

    cb = begin(0);
    for (i = 0; i < 2; i++) {
        count = 3;
        REQUIRE_VK(vkGetSwapchainImagesKHR(device, swapchains[i], &count, images[i]));
    }
    REQUIRE_VK(vkAcquireNextImageKHR(device, swapchains[0], UINT64_MAX, VK_NULL_HANDLE, fence, &indices[0]));
    REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
    vkResetFences(device, 1, &fence);
    memset(&acquire, 0, sizeof(acquire));
    acquire.sType = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR;
    acquire.swapchain = swapchains[1];
    acquire.timeout = UINT64_MAX;
    acquire.fence = fence;
    acquire.deviceMask = 1;
    REQUIRE_VK(vkAcquireNextImage2KHR(device, &acquire, &indices[1]));
    REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
    vkResetFences(device, 1, &fence);
    record_clear(cb, images[0][indices[0]], 0.5f, 0.5f, 0.5f, 0.5f);
    record_clear(cb, images[1][indices[1]], 0.5f, 0.5f, 0.5f, 1.0f);
    REQUIRE_VK(submit(0, cb, VK_NULL_HANDLE, VK_NULL_HANDLE, fence));
    REQUIRE_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull));
    memset(&present_info, 0, sizeof(present_info));
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.swapchainCount = 2;
    present_info.pSwapchains = swapchains;
    present_info.pImageIndices = indices;
    present_info.pResults = results;
    CHECK_VK(vkQueuePresentKHR(queues[0], &present_info), VK_SUCCESS);
    CHECK(results[0] == VK_SUCCESS && results[1] == VK_SUCCESS);
    CHECK_VK(vkQueueWaitIdle(queues[0]), VK_SUCCESS);
    vkResetCommandPool(device, pools[0], 0);
    for (i = 0; i < 2; i++) {
        vkDestroySwapchainKHR(device, swapchains[i], NULL);
        vkDestroySurfaceKHR(instance, surfaces[i], NULL);
        pvk_release(layers[i]);
    }
    vkDestroyFence(device, fence, NULL);
}

/* Lazy signals and presentation under threads (section 16's
 * pvk_queue_signal): a render thread acquires from a headless swapchain,
 * clears on queue 0 and signals the frame's binary semaphore lazily (its
 * submission's last batch: the host sees it through queue 0's timeline); a
 * present thread presents each frame waiting on it, from queue 0 (its
 * order), from queue 1 (the GPU waits for the semaphore's twin), or from
 * queue 1 held back by a timeline gate it opens after a random delay (the
 * queue's thread waits on the host for queue 0's timeline); a third thread
 * waits for every acquire's fence. Acquires of images not yet released make
 * queue 0 wait for queue 1 on its thread too. A wait for a value a lazy
 * signal never sets hangs here. */

enum { LAZY_FRAMES = 120 };

static struct {
    pthread_mutex_t lock, swapchain_lock, queue0_lock;
    pthread_cond_t cond;
    int rendered, presented; /* the last frame rendered (submitted), presented */
    VkSwapchainKHR swapchain;
    VkImage images[4];
    uint32_t index[LAZY_FRAMES];
    VkSemaphore acquired[LAZY_FRAMES], rendered_sem[LAZY_FRAMES], gate;
    VkFence fences[LAZY_FRAMES];
    int errors[3]; /* render, present, waiter: test macros are not thread safe */
} lz;

static void lz_wait_for(int *counter, int value)
{
    pthread_mutex_lock(&lz.lock);
    while (*counter < value)
        pthread_cond_wait(&lz.cond, &lz.lock);
    pthread_mutex_unlock(&lz.lock);
}

static void lz_post(int *counter, int value)
{
    pthread_mutex_lock(&lz.lock);
    *counter = value;
    pthread_cond_broadcast(&lz.cond);
    pthread_mutex_unlock(&lz.lock);
}

static void *lz_render(void *arg)
{
    int k;
    (void)arg;
    for (k = 0; k < LAZY_FRAMES; k++) {
        VkCommandBuffer cb;
        VkResult r;
        lz_wait_for(&lz.presented, k - 2); /* never holds every image */
        pthread_mutex_lock(&lz.swapchain_lock);
        r = vkAcquireNextImageKHR(device, lz.swapchain, UINT64_MAX, lz.acquired[k], lz.fences[k], &lz.index[k]);
        pthread_mutex_unlock(&lz.swapchain_lock);
        cb = begin(0);
        if (r != VK_SUCCESS || !cb) {
            lz.errors[0]++;
            break;
        }
        record_clear(cb, lz.images[lz.index[k]], 0.0f, (float)k / LAZY_FRAMES, 1.0f, 1.0f);
        pthread_mutex_lock(&lz.queue0_lock);
        if (submit(0, cb, lz.acquired[k], lz.rendered_sem[k], VK_NULL_HANDLE) != VK_SUCCESS)
            lz.errors[0]++;
        pthread_mutex_unlock(&lz.queue0_lock);
        lz_post(&lz.rendered, k);
    }
    lz_post(&lz.rendered, LAZY_FRAMES); /* (on failure: let the present thread go) */
    return NULL;
}

static void *lz_present(void *arg)
{
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    unsigned seed = 3;
    uint64_t gate_value = 0;
    int k;
    (void)arg;
    for (k = 0; k < LAZY_FRAMES && !lz.errors[0]; k++) {
        uint32_t q = k % 3 == 0 ? 0 : 1;
        bool gated = k % 3 == 2;
        VkPresentInfoKHR info;
        VkResult r;
        lz_wait_for(&lz.rendered, k);
        if (lz.errors[0])
            break;
        if (gated) {
            VkTimelineSemaphoreSubmitInfo values;
            VkSubmitInfo gate;
            gate_value++;
            memset(&values, 0, sizeof(values));
            values.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            values.waitSemaphoreValueCount = 1;
            values.pWaitSemaphoreValues = &gate_value;
            memset(&gate, 0, sizeof(gate));
            gate.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            gate.pNext = &values;
            gate.waitSemaphoreCount = 1;
            gate.pWaitSemaphores = &lz.gate;
            gate.pWaitDstStageMask = &stage;
            if (vkQueueSubmit(queues[1], 1, &gate, VK_NULL_HANDLE) != VK_SUCCESS)
                lz.errors[1]++;
        }
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores = &lz.rendered_sem[k];
        info.swapchainCount = 1;
        info.pSwapchains = &lz.swapchain;
        info.pImageIndices = &lz.index[k];
        if (q == 0)
            pthread_mutex_lock(&lz.queue0_lock);
        pthread_mutex_lock(&lz.swapchain_lock);
        r = vkQueuePresentKHR(queues[q], &info);
        pthread_mutex_unlock(&lz.swapchain_lock);
        if (q == 0)
            pthread_mutex_unlock(&lz.queue0_lock);
        if (r != VK_SUCCESS)
            lz.errors[1]++;
        if (gated) {
            VkSemaphoreSignalInfo signal;
            struct timespec ts;
            ts.tv_sec = 0;
            ts.tv_nsec = (long)(rand_r(&seed) % 500) * 1000;
            nanosleep(&ts, NULL);
            memset(&signal, 0, sizeof(signal));
            signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
            signal.semaphore = lz.gate;
            signal.value = gate_value;
            vkSignalSemaphore(device, &signal);
        }
        lz_post(&lz.presented, k);
    }
    lz_post(&lz.presented, LAZY_FRAMES);
    return NULL;
}

static void *lz_waiter(void *arg)
{
    int k;
    (void)arg;
    for (k = 0; k < LAZY_FRAMES; k++) {
        lz_wait_for(&lz.rendered, k); /* (the acquire set the fence up) */
        if (lz.errors[0])
            break;
        if (vkWaitForFences(device, 1, &lz.fences[k], VK_TRUE, 20000000000ull) != VK_SUCCESS) {
            fprintf(stderr, "  frame %d: acquire fence not signalled\n", k);
            lz.errors[2]++;
            break;
        }
    }
    return NULL;
}

static void test_lazy_present_threads(void)
{
    VkHeadlessSurfaceCreateInfoEXT hi;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainCreateInfoKHR info;
    pthread_t threads[3];
    void *(*fns[3])(void *) = { lz_render, lz_present, lz_waiter };
    uint32_t count = 4, families[2] = { 0, 1 };
    int k, i;

    memset(&lz, 0, sizeof(lz));
    REQUIRE(pthread_mutex_init(&lz.lock, NULL) == 0 && pthread_cond_init(&lz.cond, NULL) == 0);
    REQUIRE(pthread_mutex_init(&lz.swapchain_lock, NULL) == 0 && pthread_mutex_init(&lz.queue0_lock, NULL) == 0);
    lz.rendered = lz.presented = -1;
    memset(&hi, 0, sizeof(hi));
    hi.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
    REQUIRE_VK(vkCreateHeadlessSurfaceEXT(instance, &hi, NULL, &surface));
    info = swapchain_info(surface, VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, 64, 64, 3);
    info.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    info.imageSharingMode = VK_SHARING_MODE_CONCURRENT; /* rendered on queue 0, presented from either */
    info.queueFamilyIndexCount = 2;
    info.pQueueFamilyIndices = families;
    REQUIRE_VK(vkCreateSwapchainKHR(device, &info, NULL, &lz.swapchain));
    REQUIRE_VK(vkGetSwapchainImagesKHR(device, lz.swapchain, &count, lz.images));
    REQUIRE(count == 3);
    lz.gate = make_semaphore(VK_SEMAPHORE_TYPE_TIMELINE);
    for (k = 0; k < LAZY_FRAMES; k++) {
        lz.acquired[k] = make_semaphore(VK_SEMAPHORE_TYPE_BINARY);
        lz.rendered_sem[k] = make_semaphore(VK_SEMAPHORE_TYPE_BINARY);
        lz.fences[k] = make_fence();
        REQUIRE(lz.acquired[k] && lz.rendered_sem[k] && lz.fences[k]);
    }
    for (i = 0; i < 3; i++)
        REQUIRE(pthread_create(&threads[i], NULL, fns[i], NULL) == 0);
    for (i = 0; i < 3; i++)
        pthread_join(threads[i], NULL);
    CHECK(lz.errors[0] == 0 && lz.errors[1] == 0 && lz.errors[2] == 0);
    CHECK_VK(vkDeviceWaitIdle(device), VK_SUCCESS);
    for (k = 0; k < LAZY_FRAMES; k++) {
        CHECK_VK(vkGetFenceStatus(device, lz.fences[k]), VK_SUCCESS);
        vkDestroySemaphore(device, lz.acquired[k], NULL);
        vkDestroySemaphore(device, lz.rendered_sem[k], NULL);
        vkDestroyFence(device, lz.fences[k], NULL);
    }
    vkResetCommandPool(device, pools[0], 0);
    vkDestroySemaphore(device, lz.gate, NULL);
    vkDestroySwapchainKHR(device, lz.swapchain, NULL);
    vkDestroySurfaceKHR(instance, surface, NULL);
    pthread_cond_destroy(&lz.cond);
    pthread_mutex_destroy(&lz.lock);
    pthread_mutex_destroy(&lz.swapchain_lock);
    pthread_mutex_destroy(&lz.queue0_lock);
}

int main(void)
{
    void *pool = objc_autoreleasePoolPush();
    setup();
    RUN(test_surface_queries);
    RUN(test_extension_scoping);
    RUN(test_present_cycles);
    RUN(test_present_contents);
    RUN(test_resize);
    RUN(test_acquire_timeouts);
    RUN(test_deferred_present);
    RUN(test_frame_loop);
    RUN(test_fetch_copies);
    RUN(test_direct);
    RUN(test_transfers_into_drawables);
    RUN(test_layer_configuration);
    RUN(test_headless);
    RUN(test_lazy_present_threads);
    teardown();
    objc_autoreleasePoolPop(pool);
    return TEST_RESULT();
}
