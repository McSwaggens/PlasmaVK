/* Window System Integration (WSI): VK_KHR_surface on CAMetalLayers
 * (VK_EXT_metal_surface) and VK_KHR_swapchain.
 *
 * The layers are made through the Objective-C runtime in plain C and are in
 * no window (CAMetalLayer vends drawables all the same), so the tests run
 * headless. Each is a plain CAMetalLayer with bounds and contentsScale 1. */
#include "ct.h"

#include <objc/message.h>
#include <objc/runtime.h>
#include <stdlib.h>
#include <time.h>

/* CAMetalLayer and the runtime without link flags: on PlasmaVK the
 * implementation brings them, through the loader nothing does. */
__asm__(".linker_option \"-framework\", \"QuartzCore\"\n"
        ".linker_option \"-lobjc\"\n");

/* Exported by libobjc but absent from its public headers. */
extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *pool);

#define WSI_MAX(a, b) ((a) > (b) ? (a) : (b))

/* ---- Layers ---- */

struct wsi_rect {
    double x, y, w, h; /* CGRect */
};

static id wsi_send(id object, const char *selector)
{
    return ((id(*)(id, SEL))objc_msgSend)(object, sel_registerName(selector));
}

/* A new CAMetalLayer (retained) w x h points large, contentsScale 1. */
static void *wsi_layer(double w, double h)
{
    Class cls = (Class)objc_getClass("CAMetalLayer");
    struct wsi_rect frame = { 0, 0, w, h };
    id layer;
    if (!cls)
        return NULL;
    layer = wsi_send((id)cls, "new");
    ((void (*)(id, SEL, struct wsi_rect))objc_msgSend)(layer, sel_registerName("setFrame:"), frame);
    ((void (*)(id, SEL, double))objc_msgSend)(layer, sel_registerName("setContentsScale:"), 1.0);
    return layer;
}

static void wsi_release(void *layer)
{
    if (layer)
        wsi_send((id)layer, "release");
}

/* ---- Environment ---- */

struct wsi {
    struct ct_env e;
    void *pool;  /* autorelease pool */
    void *layer; /* CAMetalLayer */
    VkSurfaceKHR surface;
    VkSurfaceCapabilitiesKHR caps;
    VkSurfaceFormatKHR format; /* the first one reported */
    VkImageUsageFlags usage;   /* COLOR_ATTACHMENT, and TRANSFER_DST/SRC where the surface and format allow */
};

static int wsi_extensions_offered(void)
{
    return ct_instance_extension_supported(VK_KHR_SURFACE_EXTENSION_NAME) &&
           ct_instance_extension_supported(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
}

/* Instance with VK_KHR_surface and VK_EXT_metal_surface, a surface on a
 * new layer w x h large; with `device`, a device with VK_KHR_swapchain whose
 * queue can present to the surface. Skips if any of it is not offered. */
static void wsi_init(struct ct *t, struct wsi *w, double width, double height, int device)
{
    static const char *const instance_exts[] = { VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_METAL_SURFACE_EXTENSION_NAME };
    static const char *const device_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkMetalSurfaceCreateInfoEXT info;
    VkSurfaceFormatKHR formats[256];
    VkImageFormatProperties props;
    VkBool32 supported = VK_FALSE;
    uint32_t count = CT_ARRAY_SIZE(formats);

    memset(w, 0, sizeof(*w));
    if (!wsi_extensions_offered())
        ct_skip(t, "VK_KHR_surface and VK_EXT_metal_surface are not offered");
    w->pool = objc_autoreleasePoolPush();
    ct_env_init_ex(t, &w->e, 0, instance_exts, 2);
    w->layer = wsi_layer(width, height);
    CT_REQUIRE(t, w->layer != NULL);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
    info.pLayer = w->layer;
    CT_REQUIRE_VK(t, vkCreateMetalSurfaceEXT(w->e.instance, &info, NULL, &w->surface));
    CT_REQUIRE_VK(t, vkGetPhysicalDeviceSurfaceCapabilitiesKHR(w->e.pd, w->surface, &w->caps));
    CT_REQUIRE(t, vkGetPhysicalDeviceSurfaceFormatsKHR(w->e.pd, w->surface, &count, formats) >= 0 && count > 0);
    w->format = formats[0];
    w->usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (vkGetPhysicalDeviceImageFormatProperties(w->e.pd, w->format.format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                 w->caps.supportedUsageFlags &
                                                     (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
                                                 0, &props) == VK_SUCCESS)
        w->usage = w->caps.supportedUsageFlags & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (!device)
        return;
    if (!ct_device_extension_supported(w->e.pd, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
        ct_skip(t, "VK_KHR_swapchain is not offered");
    ct_env_device_ex(t, &w->e, NULL, device_exts, 1, NULL);
    CT_REQUIRE_VK(t, vkGetPhysicalDeviceSurfaceSupportKHR(w->e.pd, w->e.queue_family, w->surface, &supported));
    if (!supported)
        ct_skip(t, "queue family %u cannot present to the surface", w->e.queue_family);
}

static void wsi_fini(struct wsi *w)
{
    if (w->e.device)
        vkDeviceWaitIdle(w->e.device);
    if (w->surface)
        vkDestroySurfaceKHR(w->e.instance, w->surface, NULL);
    ct_env_fini(&w->e);
    wsi_release(w->layer);
    if (w->pool)
        objc_autoreleasePoolPop(w->pool);
    memset(w, 0, sizeof(*w));
}

static VkCompositeAlphaFlagBitsKHR wsi_composite_alpha(const VkSurfaceCapabilitiesKHR *caps)
{
    VkCompositeAlphaFlagsKHR bits = caps->supportedCompositeAlpha;
    if (bits & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
        return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    return (VkCompositeAlphaFlagBitsKHR)(bits & ~(bits - 1)); /* the lowest bit set */
}

/* A valid VkSwapchainCreateInfoKHR from the surface's capabilities: the
 * first format, the current extent (or w->e's 64 x 64 when the swapchain
 * decides), FIFO. */
static VkSwapchainCreateInfoKHR wsi_swapchain_info(const struct wsi *w, VkSurfaceKHR surface, VkSwapchainKHR old)
{
    VkSwapchainCreateInfoKHR info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface = surface;
    info.minImageCount = w->caps.minImageCount;
    info.imageFormat = w->format.format;
    info.imageColorSpace = w->format.colorSpace;
    info.imageExtent = w->caps.currentExtent;
    if (info.imageExtent.width == 0xFFFFFFFFu) {
        info.imageExtent.width = CT_MIN(WSI_MAX(64u, w->caps.minImageExtent.width), w->caps.maxImageExtent.width);
        info.imageExtent.height = CT_MIN(WSI_MAX(64u, w->caps.minImageExtent.height), w->caps.maxImageExtent.height);
    }
    info.imageArrayLayers = 1;
    info.imageUsage = w->usage;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = w->caps.currentTransform;
    info.compositeAlpha = wsi_composite_alpha(&w->caps);
    info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    info.clipped = VK_TRUE;
    info.oldSwapchain = old;
    return info;
}

static VkSemaphore wsi_semaphore(struct ct *t, struct wsi *w)
{
    VkSemaphoreCreateInfo info;
    VkSemaphore s;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreateSemaphore(w->e.device, &info, NULL, &s));
    return s;
}

static VkFence wsi_fence(struct ct *t, struct wsi *w)
{
    VkFenceCreateInfo info;
    VkFence f;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreateFence(w->e.device, &info, NULL, &f));
    return f;
}

static void wsi_wait_fence(struct ct *t, struct wsi *w, VkFence fence)
{
    VkResult r = vkWaitForFences(w->e.device, 1, &fence, VK_TRUE, 10000000000ull);
    CT_REQUIRE_VK(t, r);
    CT_REQUIRE_VK(t, vkResetFences(w->e.device, 1, &fence));
}

/* Records UNDEFINED -> PRESENT_SRC for the image, clearing it to `color`
 * on the way when the swapchain has TRANSFER_DST. */
static void wsi_record_frame(struct wsi *w, VkCommandBuffer cb, VkImage image, const float color[4])
{
    VkImageSubresourceRange range;
    VkClearColorValue clear;
    memset(&range, 0, sizeof(range));
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;
    if (!(w->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        ct_image_barrier(cb, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        return;
    }
    memcpy(clear.float32, color, sizeof(clear.float32));
    ct_image_barrier(cb, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdClearColorImage(cb, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
    ct_image_barrier(cb, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
}

/* Ends and submits cb, waiting for `wait` (if any) and signalling `signal`
 * (if any) and `fence`. */
static void wsi_submit(struct ct *t, struct wsi *w, VkCommandBuffer cb, VkSemaphore wait, VkSemaphore signal,
                       VkFence fence)
{
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit;
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = wait ? 1 : 0;
    submit.pWaitSemaphores = &wait;
    submit.pWaitDstStageMask = &stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    submit.signalSemaphoreCount = signal ? 1 : 0;
    submit.pSignalSemaphores = &signal;
    CT_REQUIRE_VK(t, vkQueueSubmit(w->e.queue, 1, &submit, fence));
}

static VkResult wsi_present(struct wsi *w, VkSwapchainKHR swapchain, uint32_t index, VkSemaphore wait,
                            VkResult *per_swapchain)
{
    VkPresentInfoKHR info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = wait ? 1 : 0;
    info.pWaitSemaphores = &wait;
    info.swapchainCount = 1;
    info.pSwapchains = &swapchain;
    info.pImageIndices = &index;
    info.pResults = per_swapchain;
    return vkQueuePresentKHR(w->e.queue, &info);
}

static int wsi_presentable(VkResult r)
{
    return r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR;
}

/* Makes the image presentable (fence-synchronized) and presents it. */
static VkResult wsi_present_image(struct ct *t, struct wsi *w, VkSwapchainKHR swapchain, VkImage image,
                                  uint32_t index, VkFence fence)
{
    static const float color[4] = { 0.0f, 0.5f, 1.0f, 1.0f };
    VkCommandBuffer cb = ct_begin_commands(t, &w->e);
    wsi_record_frame(w, cb, image, color);
    wsi_submit(t, w, cb, VK_NULL_HANDLE, VK_NULL_HANDLE, fence);
    wsi_wait_fence(t, w, fence);
    vkFreeCommandBuffers(w->e.device, w->e.pool, 1, &cb);
    return wsi_present(w, swapchain, index, VK_NULL_HANDLE, NULL);
}

static VkImage *wsi_images(struct ct *t, struct wsi *w, VkSwapchainKHR swapchain, uint32_t *count)
{
    VkImage *images;
    CT_REQUIRE_VK(t, vkGetSwapchainImagesKHR(w->e.device, swapchain, count, NULL));
    CT_REQUIRE(t, *count > 0);
    images = calloc(*count, sizeof(*images));
    CT_REQUIRE(t, images != NULL);
    CT_REQUIRE_VK(t, vkGetSwapchainImagesKHR(w->e.device, swapchain, count, images));
    return images;
}

/* ---- Surfaces ---- */

CT_TEST(surface_create_destroy,
        "Window System Integration > Metal Platform: vkCreateMetalSurfaceEXT makes a VkSurfaceKHR for a "
        "CAMetalLayer; vkDestroySurfaceKHR destroys it and ignores VK_NULL_HANDLE (Surface Queries: two surfaces "
        "of distinct layers are distinct handles)",
        "vkCreateMetalSurfaceEXT vkDestroySurfaceKHR")
{
    static const char *const exts[] = { VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_METAL_SURFACE_EXTENSION_NAME };
    struct ct_env e;
    VkMetalSurfaceCreateInfoEXT info;
    VkSurfaceKHR a = VK_NULL_HANDLE, b = VK_NULL_HANDLE;
    void *pool, *la, *lb;

    if (!wsi_extensions_offered())
        ct_skip(t, "VK_KHR_surface and VK_EXT_metal_surface are not offered");
    pool = objc_autoreleasePoolPush();
    ct_env_init_ex(t, &e, 0, exts, 2);
    la = wsi_layer(32, 32);
    lb = wsi_layer(16, 16);
    CT_REQUIRE(t, la && lb);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
    info.pLayer = la;
    CT_REQUIRE_VK(t, vkCreateMetalSurfaceEXT(e.instance, &info, NULL, &a));
    info.pLayer = lb;
    CT_REQUIRE_VK(t, vkCreateMetalSurfaceEXT(e.instance, &info, NULL, &b));
    CT_CHECK(t, a != VK_NULL_HANDLE && b != VK_NULL_HANDLE && a != b);
    vkDestroySurfaceKHR(e.instance, a, NULL);
    vkDestroySurfaceKHR(e.instance, b, NULL);
    vkDestroySurfaceKHR(e.instance, VK_NULL_HANDLE, NULL);
    ct_env_fini(&e);
    wsi_release(la);
    wsi_release(lb);
    objc_autoreleasePoolPop(pool);
}

CT_TEST(surface_allocation_callbacks,
        "Memory Allocation > Host Memory: host memory of a surface comes from the allocator given to "
        "vkCreateMetalSurfaceEXT, following the callback rules, and is all freed by vkDestroySurfaceKHR",
        "vkCreateMetalSurfaceEXT vkDestroySurfaceKHR")
{
    static const char *const exts[] = { VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_METAL_SURFACE_EXTENSION_NAME };
    struct ct_env e;
    struct ct_allocator a;
    VkMetalSurfaceCreateInfoEXT info;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkResult r;
    void *pool, *layer;

    if (!wsi_extensions_offered())
        ct_skip(t, "VK_KHR_surface and VK_EXT_metal_surface are not offered");
    pool = objc_autoreleasePoolPush();
    ct_env_init_ex(t, &e, 0, exts, 2);
    layer = wsi_layer(32, 32);
    CT_REQUIRE(t, layer != NULL);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
    info.pLayer = layer;
    ct_allocator_init(&a, -1);
    CT_CALL_VK(&a, r, vkCreateMetalSurfaceEXT(e.instance, &info, &a.cb, &surface));
    CT_REQUIRE_VK(t, r);
    CT_CALL(&a, vkDestroySurfaceKHR(e.instance, surface, &a.cb));
    CT_CHECKF(t, a.live == 0, "%ld allocations still live after vkDestroySurfaceKHR", a.live);
    ct_allocator_check(t, &a, "surface");
    ct_allocator_fini(&a);
    ct_env_fini(&e);
    wsi_release(layer);
    objc_autoreleasePoolPop(pool);
}

CT_TEST(surface_support,
        "Window System Integration > Querying for WSI Support: vkGetPhysicalDeviceSurfaceSupportKHR answers "
        "VK_TRUE or VK_FALSE for every queue family of the physical device",
        "vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceQueueFamilyProperties")
{
    struct wsi w;
    uint32_t count = 0, i;
    wsi_init(t, &w, 32, 32, 0);
    vkGetPhysicalDeviceQueueFamilyProperties(w.e.pd, &count, NULL);
    CT_REQUIRE(t, count > 0);
    for (i = 0; i < count; i++) {
        VkBool32 supported = 0xCDu;
        CT_CHECK_VK(t, vkGetPhysicalDeviceSurfaceSupportKHR(w.e.pd, i, w.surface, &supported), VK_SUCCESS);
        CT_CHECKF(t, supported == VK_TRUE || supported == VK_FALSE, "family %u: pSupported is %u", i, supported);
    }
    wsi_fini(&w);
}

static int wsi_one_bit(uint32_t x)
{
    return x && !(x & (x - 1));
}

CT_TEST(surface_capabilities,
        "Window System Integration > Surface Queries > Surface Capabilities: minImageCount >= 1, maxImageCount 0 "
        "or >= minImageCount, minImageExtent <= currentExtent <= maxImageExtent unless currentExtent is "
        "(0xFFFFFFFF, 0xFFFFFFFF), maxImageArrayLayers >= 1, at least one supported transform with "
        "currentTransform one of them (one bit), at least one composite alpha mode, and supportedUsageFlags must "
        "include VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT",
        "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")
{
    struct wsi w;
    const VkSurfaceCapabilitiesKHR *c;
    wsi_init(t, &w, 64, 48, 0);
    c = &w.caps;
    CT_CHECK(t, c->minImageCount >= 1);
    CT_CHECK(t, c->maxImageCount == 0 || c->maxImageCount >= c->minImageCount);
    if (c->currentExtent.width == 0xFFFFFFFFu || c->currentExtent.height == 0xFFFFFFFFu) {
        CT_CHECKF(t, c->currentExtent.width == c->currentExtent.height, "currentExtent %ux%u: half special",
                  c->currentExtent.width, c->currentExtent.height);
    } else {
        CT_CHECK(t, c->minImageExtent.width <= c->currentExtent.width &&
                        c->minImageExtent.height <= c->currentExtent.height);
        CT_CHECK(t, c->currentExtent.width <= c->maxImageExtent.width &&
                        c->currentExtent.height <= c->maxImageExtent.height);
    }
    CT_CHECK(t, c->minImageExtent.width <= c->maxImageExtent.width && c->minImageExtent.height <= c->maxImageExtent.height);
    CT_CHECK(t, c->maxImageArrayLayers >= 1);
    CT_CHECK(t, c->supportedTransforms != 0);
    CT_CHECK(t, wsi_one_bit((uint32_t)c->currentTransform) && (c->supportedTransforms & c->currentTransform));
    CT_CHECK(t, c->supportedCompositeAlpha != 0);
    CT_CHECK(t, (c->supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0);
    wsi_fini(&w);
}

struct wsi_query {
    VkPhysicalDevice pd;
    VkSurfaceKHR surface;
};

static VkResult wsi_enum_formats(void *ctx, uint32_t *count, void *items)
{
    const struct wsi_query *q = ctx;
    return vkGetPhysicalDeviceSurfaceFormatsKHR(q->pd, q->surface, count, items);
}

static VkResult wsi_enum_modes(void *ctx, uint32_t *count, void *items)
{
    const struct wsi_query *q = ctx;
    return vkGetPhysicalDeviceSurfacePresentModesKHR(q->pd, q->surface, count, items);
}

static VkResult wsi_enum_rects(void *ctx, uint32_t *count, void *items)
{
    const struct wsi_query *q = ctx;
    return vkGetPhysicalDevicePresentRectanglesKHR(q->pd, q->surface, count, items);
}

static int wsi_same_format(const void *a, const void *b)
{
    const VkSurfaceFormatKHR *x = a, *y = b;
    return x->format == y->format && x->colorSpace == y->colorSpace;
}

static int wsi_same_mode(const void *a, const void *b)
{
    return *(const VkPresentModeKHR *)a == *(const VkPresentModeKHR *)b;
}

static int wsi_same_rect(const void *a, const void *b)
{
    return memcmp(a, b, sizeof(VkRect2D)) == 0;
}

/* The UNORM/SRGB pairs of renderable formats (Formats). */
static const VkFormat wsi_srgb_pairs[][2] = {
    { VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SRGB },
    { VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SRGB },
    { VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8_SRGB },
    { VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_B8G8R8_SRGB },
    { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB },
    { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB },
    { VK_FORMAT_A8B8G8R8_UNORM_PACK32, VK_FORMAT_A8B8G8R8_SRGB_PACK32 },
};

CT_TEST(surface_formats,
        "Window System Integration > Surface Queries > Surface Format Support: at least one format pair; no "
        "VK_FORMAT_UNDEFINED; an SRGB_NONLINEAR entry with a UNORM (SRGB) format that has an SRGB (UNORM) "
        "counterpart must come with an entry of that counterpart; the VK_INCOMPLETE enumeration protocol",
        "vkGetPhysicalDeviceSurfaceFormatsKHR")
{
    struct wsi w;
    struct wsi_query q;
    VkSurfaceFormatKHR *formats;
    uint32_t count = 0, i, j, k;
    wsi_init(t, &w, 32, 32, 0);
    q.pd = w.e.pd;
    q.surface = w.surface;
    CT_REQUIRE_VK(t, vkGetPhysicalDeviceSurfaceFormatsKHR(w.e.pd, w.surface, &count, NULL));
    CT_REQUIRE(t, count >= 1);
    formats = calloc(count, sizeof(*formats));
    CT_REQUIRE(t, formats != NULL);
    CT_REQUIRE_VK(t, vkGetPhysicalDeviceSurfaceFormatsKHR(w.e.pd, w.surface, &count, formats));
    for (i = 0; i < count; i++) {
        CT_CHECKF(t, formats[i].format != VK_FORMAT_UNDEFINED, "entry %u is VK_FORMAT_UNDEFINED", i);
        if (formats[i].colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            continue;
        for (k = 0; k < CT_ARRAY_SIZE(wsi_srgb_pairs); k++) {
            int side = formats[i].format == wsi_srgb_pairs[k][0] ? 1 : formats[i].format == wsi_srgb_pairs[k][1] ? 0 : -1;
            int found = 0;
            if (side < 0)
                continue;
            for (j = 0; j < count; j++)
                found |= formats[j].format == wsi_srgb_pairs[k][side] &&
                         formats[j].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
            CT_CHECKF(t, found, "format %d in SRGB_NONLINEAR without its counterpart %d", (int)formats[i].format,
                      (int)wsi_srgb_pairs[k][side]);
        }
    }
    ct_check_incomplete(t, wsi_enum_formats, &q, sizeof(VkSurfaceFormatKHR), wsi_same_format, "surface formats");
    free(formats);
    wsi_fini(&w);
}

CT_TEST(surface_present_modes,
        "Window System Integration > Surface Queries > Surface Presentation Mode Support: "
        "VK_PRESENT_MODE_FIFO_KHR is required to be supported; the VK_INCOMPLETE enumeration protocol",
        "vkGetPhysicalDeviceSurfacePresentModesKHR")
{
    struct wsi w;
    struct wsi_query q;
    VkPresentModeKHR modes[32];
    uint32_t count = CT_ARRAY_SIZE(modes), i;
    int fifo = 0;
    wsi_init(t, &w, 32, 32, 0);
    q.pd = w.e.pd;
    q.surface = w.surface;
    CT_REQUIRE(t, vkGetPhysicalDeviceSurfacePresentModesKHR(w.e.pd, w.surface, &count, modes) >= 0);
    for (i = 0; i < count; i++)
        fifo |= modes[i] == VK_PRESENT_MODE_FIFO_KHR;
    CT_CHECK(t, fifo);
    ct_check_incomplete(t, wsi_enum_modes, &q, sizeof(VkPresentModeKHR), wsi_same_mode, "present modes");
    wsi_fini(&w);
}

CT_TEST(present_rectangles,
        "Window System Integration > Device Group Queries: the rectangles of "
        "vkGetPhysicalDevicePresentRectanglesKHR must not overlap; the VK_INCOMPLETE enumeration protocol "
        "(Vulkan 1.1: VK_KHR_swapchain's device group commands)",
        "vkGetPhysicalDevicePresentRectanglesKHR")
{
    struct wsi w;
    struct wsi_query q;
    VkRect2D rects[64];
    uint32_t count = CT_ARRAY_SIZE(rects), i, j;
    wsi_init(t, &w, 32, 32, 0);
    if (w.e.version < VK_API_VERSION_1_1)
        ct_skip(t, "Vulkan 1.0");
    q.pd = w.e.pd;
    q.surface = w.surface;
    CT_REQUIRE(t, vkGetPhysicalDevicePresentRectanglesKHR(w.e.pd, w.surface, &count, rects) >= 0);
    for (i = 0; i < count; i++)
        for (j = i + 1; j < count; j++) {
            int64_t ax = rects[i].offset.x, ay = rects[i].offset.y, bx = rects[j].offset.x, by = rects[j].offset.y;
            int disjoint = ax + (int64_t)rects[i].extent.width <= bx || bx + (int64_t)rects[j].extent.width <= ax ||
                           ay + (int64_t)rects[i].extent.height <= by || by + (int64_t)rects[j].extent.height <= ay;
            CT_CHECKF(t, disjoint, "rectangles %u and %u overlap", i, j);
        }
    ct_check_incomplete(t, wsi_enum_rects, &q, sizeof(VkRect2D), wsi_same_rect, "present rectangles");
    wsi_fini(&w);
}

CT_TEST(device_group_present_modes,
        "Window System Integration > Device Group Queries: VkDeviceGroupPresentCapabilitiesKHR::modes always has "
        "VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR; presentMask[i] non-zero must have bit i, and is zero beyond "
        "the group's devices; vkGetDeviceGroupSurfacePresentModesKHR's modes must be a subset of those "
        "(Vulkan 1.1)",
        "vkGetDeviceGroupPresentCapabilitiesKHR vkGetDeviceGroupSurfacePresentModesKHR vkEnumeratePhysicalDeviceGroups")
{
    struct wsi w;
    VkDeviceGroupPresentCapabilitiesKHR caps;
    VkDeviceGroupPresentModeFlagsKHR modes = 0;
    uint32_t i;
    wsi_init(t, &w, 32, 32, 1);
    if (w.e.version < VK_API_VERSION_1_1)
        ct_skip(t, "Vulkan 1.0");
    memset(&caps, 0xCD, sizeof(caps));
    caps.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_CAPABILITIES_KHR;
    caps.pNext = NULL;
    CT_REQUIRE_VK(t, vkGetDeviceGroupPresentCapabilitiesKHR(w.e.device, &caps));
    CT_CHECK(t, (caps.modes & VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR) != 0);
    /* ct_env_device_ex makes a device of one physical device: element 0. */
    for (i = 0; i < VK_MAX_DEVICE_GROUP_SIZE; i++) {
        if (i > 0)
            CT_CHECKF(t, caps.presentMask[i] == 0, "presentMask[%u] = 0x%x for a device of one physical device", i,
                      caps.presentMask[i]);
        else if (caps.presentMask[i])
            CT_CHECKF(t, caps.presentMask[i] & 1u, "presentMask[0] = 0x%x lacks bit 0", caps.presentMask[0]);
    }
    CT_REQUIRE_VK(t, vkGetDeviceGroupSurfacePresentModesKHR(w.e.device, w.surface, &modes));
    CT_CHECKF(t, (modes & ~caps.modes) == 0, "surface modes 0x%x, device modes 0x%x", modes, caps.modes);
    wsi_fini(&w);
}

/* ---- Swapchains ---- */

static int wsi_same_image(const void *a, const void *b)
{
    return *(const VkImage *)a == *(const VkImage *)b;
}

struct wsi_images_query {
    VkDevice device;
    VkSwapchainKHR swapchain;
};

static VkResult wsi_enum_images(void *ctx, uint32_t *count, void *items)
{
    const struct wsi_images_query *q = ctx;
    return vkGetSwapchainImagesKHR(q->device, q->swapchain, count, items);
}

CT_TEST(swapchain_create_destroy,
        "Window System Integration > WSI Swapchain: a swapchain has at least minImageCount presentable images "
        "(VkSwapchainCreateInfoKHR::minImageCount: at least that many or creation fails), "
        "vkGetSwapchainImagesKHR follows the VK_INCOMPLETE protocol and returns the same images each time; "
        "vkDestroySwapchainKHR ignores VK_NULL_HANDLE",
        "vkCreateSwapchainKHR vkGetSwapchainImagesKHR vkDestroySwapchainKHR")
{
    struct wsi w;
    struct wsi_images_query q;
    VkSwapchainCreateInfoKHR info;
    VkSwapchainKHR swapchain;
    VkImage *images, *again;
    uint32_t count = 0, n, i;
    wsi_init(t, &w, 32, 32, 1);
    info = wsi_swapchain_info(&w, w.surface, VK_NULL_HANDLE);
    CT_REQUIRE_VK(t, vkCreateSwapchainKHR(w.e.device, &info, NULL, &swapchain));
    images = wsi_images(t, &w, swapchain, &count);
    CT_CHECKF(t, count >= info.minImageCount, "%u images, minImageCount %u", count, info.minImageCount);
    n = count;
    again = wsi_images(t, &w, swapchain, &n);
    CT_CHECK(t, n == count);
    for (i = 0; i < count && i < n; i++)
        CT_CHECKF(t, images[i] == again[i], "image %u changed between two queries", i);
    for (i = 0; i < count; i++)
        CT_CHECK(t, images[i] != VK_NULL_HANDLE);
    q.device = w.e.device;
    q.swapchain = swapchain;
    ct_check_incomplete(t, wsi_enum_images, &q, sizeof(VkImage), wsi_same_image, "swapchain images");
    vkDestroySwapchainKHR(w.e.device, swapchain, NULL);
    vkDestroySwapchainKHR(w.e.device, VK_NULL_HANDLE, NULL);
    free(images);
    free(again);
    wsi_fini(&w);
}

CT_TEST(swapchain_allocation_callbacks,
        "Memory Allocation > Host Memory: host memory of a swapchain comes from the allocator given to "
        "vkCreateSwapchainKHR, following the callback rules, and is all freed by vkDestroySwapchainKHR",
        "vkCreateSwapchainKHR vkDestroySwapchainKHR vkGetSwapchainImagesKHR")
{
    struct wsi w;
    struct ct_allocator a;
    VkSwapchainCreateInfoKHR info;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkImage images[16];
    uint32_t count = CT_ARRAY_SIZE(images);
    VkResult r;
    wsi_init(t, &w, 32, 32, 1);
    info = wsi_swapchain_info(&w, w.surface, VK_NULL_HANDLE);
    ct_allocator_init(&a, -1);
    CT_CALL_VK(&a, r, vkCreateSwapchainKHR(w.e.device, &info, &a.cb, &swapchain));
    CT_REQUIRE_VK(t, r);
    CT_CALL_VK(&a, r, vkGetSwapchainImagesKHR(w.e.device, swapchain, &count, images));
    CT_CHECK(t, r == VK_SUCCESS || r == VK_INCOMPLETE);
    CT_CALL(&a, vkDestroySwapchainKHR(w.e.device, swapchain, &a.cb));
    CT_CHECKF(t, a.live == 0, "%ld allocations still live after vkDestroySwapchainKHR", a.live);
    ct_allocator_check(t, &a, "swapchain");
    ct_allocator_fini(&a);
    wsi_fini(&w);
}

/* A second swapchain for the layer of `surface` while one exists:
 * VK_ERROR_NATIVE_WINDOW_IN_USE_KHR, and none made. Once the first is
 * destroyed, the layer takes a swapchain again. */
static void wsi_check_in_use(struct ct *t, struct wsi *w, VkSurfaceKHR surface)
{
    VkSwapchainCreateInfoKHR info = wsi_swapchain_info(w, w->surface, VK_NULL_HANDLE);
    VkSwapchainKHR swapchain, other = VK_NULL_HANDLE;
    VkResult r;
    CT_REQUIRE_VK(t, vkCreateSwapchainKHR(w->e.device, &info, NULL, &swapchain));
    info.surface = surface;
    r = vkCreateSwapchainKHR(w->e.device, &info, NULL, &other);
    CT_CHECKF(t, r == VK_ERROR_NATIVE_WINDOW_IN_USE_KHR, "second swapchain: %s, expected %s", ct_result_name(r),
              ct_result_name(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR));
    if (r == VK_SUCCESS)
        vkDestroySwapchainKHR(w->e.device, other, NULL);
    vkDestroySwapchainKHR(w->e.device, swapchain, NULL);
    other = VK_NULL_HANDLE;
    CT_CHECK_VK(t, vkCreateSwapchainKHR(w->e.device, &info, NULL, &other), VK_SUCCESS);
    if (other)
        vkDestroySwapchainKHR(w->e.device, other, NULL);
}

CT_TEST(native_window_in_use,
        "Window System Integration > WSI Swapchain (vkCreateSwapchainKHR): if oldSwapchain is VK_NULL_HANDLE and "
        "the native window referred to by the surface is already associated with a Vulkan swapchain, "
        "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR must be returned (a second swapchain of the same surface)",
        "vkCreateSwapchainKHR vkDestroySwapchainKHR")
{
    struct wsi w;
    wsi_init(t, &w, 32, 32, 1);
    wsi_check_in_use(t, &w, w.surface);
    wsi_fini(&w);
}

CT_TEST(native_window_in_use_other_surface,
        "Window System Integration > WSI Swapchain (vkCreateSwapchainKHR): if oldSwapchain is VK_NULL_HANDLE and "
        "the native window referred to by the surface is already associated with a Vulkan swapchain, "
        "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR must be returned (a swapchain of another surface of the same "
        "CAMetalLayer: Metal Platform, the layer is the native window)",
        "vkCreateMetalSurfaceEXT vkCreateSwapchainKHR vkDestroySwapchainKHR vkDestroySurfaceKHR")
{
    struct wsi w;
    VkMetalSurfaceCreateInfoEXT sinfo;
    VkSurfaceKHR second;
    wsi_init(t, &w, 32, 32, 1);
    memset(&sinfo, 0, sizeof(sinfo));
    sinfo.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
    sinfo.pLayer = w.layer;
    CT_REQUIRE_VK(t, vkCreateMetalSurfaceEXT(w.e.instance, &sinfo, NULL, &second));
    wsi_check_in_use(t, &w, second);
    vkDestroySurfaceKHR(w.e.instance, second, NULL);
    wsi_fini(&w);
}

CT_TEST(acquire_present_frames,
        "Window System Integration > WSI Swapchain: images acquired with a semaphore, rendered after waiting for "
        "it and presented after waiting for the render's semaphore, frame after frame; each present succeeds "
        "(VK_SUCCESS or VK_SUBOPTIMAL_KHR) with pResults set to the swapchain's result, and acquired indices "
        "are below the image count",
        "vkCreateSwapchainKHR vkGetSwapchainImagesKHR vkAcquireNextImageKHR vkQueuePresentKHR vkDestroySwapchainKHR "
        "vkCreateSemaphore vkDestroySemaphore vkQueueSubmit vkCmdPipelineBarrier vkCmdClearColorImage")
{
    struct wsi w;
    VkSwapchainCreateInfoKHR info;
    VkSwapchainKHR swapchain;
    VkSemaphore acquired, *rendered;
    VkFence fence;
    VkImage *images;
    uint32_t count = 0, frame, i;
    wsi_init(t, &w, 32, 32, 1);
    info = wsi_swapchain_info(&w, w.surface, VK_NULL_HANDLE);
    CT_REQUIRE_VK(t, vkCreateSwapchainKHR(w.e.device, &info, NULL, &swapchain));
    images = wsi_images(t, &w, swapchain, &count);
    acquired = wsi_semaphore(t, &w);
    fence = wsi_fence(t, &w);
    rendered = calloc(count, sizeof(*rendered)); /* one per image: reusable once the image comes back */
    CT_REQUIRE(t, rendered != NULL);
    for (i = 0; i < count; i++)
        rendered[i] = wsi_semaphore(t, &w);
    for (frame = 0; frame < 12; frame++) {
        float color[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
        VkResult r, per = VK_RESULT_MAX_ENUM;
        VkCommandBuffer cb;
        uint32_t index = UINT32_MAX;
        r = vkAcquireNextImageKHR(w.e.device, swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &index);
        CT_REQUIREF(t, wsi_presentable(r), "frame %u: vkAcquireNextImageKHR returned %s", frame, ct_result_name(r));
        CT_REQUIREF(t, index < count, "frame %u: image index %u of %u", frame, index, count);
        color[frame % 3] = 1.0f;
        cb = ct_begin_commands(t, &w.e);
        wsi_record_frame(&w, cb, images[index], color);
        wsi_submit(t, &w, cb, acquired, rendered[index], fence);
        r = wsi_present(&w, swapchain, index, rendered[index], &per);
        CT_CHECKF(t, wsi_presentable(r), "frame %u: vkQueuePresentKHR returned %s", frame, ct_result_name(r));
        CT_CHECKF(t, per == r, "frame %u: pResults[0] %s, returned %s", frame, ct_result_name(per), ct_result_name(r));
        wsi_wait_fence(t, &w, fence); /* `acquired` is free again */
        vkFreeCommandBuffers(w.e.device, w.e.pool, 1, &cb);
    }
    CT_CHECK_VK(t, vkQueueWaitIdle(w.e.queue), VK_SUCCESS);
    vkDestroySwapchainKHR(w.e.device, swapchain, NULL);
    for (i = 0; i < count; i++)
        vkDestroySemaphore(w.e.device, rendered[i], NULL);
    vkDestroySemaphore(w.e.device, acquired, NULL);
    vkDestroyFence(w.e.device, fence, NULL);
    free(rendered);
    free(images);
    wsi_fini(&w);
}

CT_TEST(acquire_fence,
        "Window System Integration > WSI Swapchain: an image acquired with only a fence is available once the "
        "fence signals; images presented come back to vkAcquireNextImageKHR",
        "vkCreateSwapchainKHR vkGetSwapchainImagesKHR vkAcquireNextImageKHR vkQueuePresentKHR vkDestroySwapchainKHR "
        "vkWaitForFences vkResetFences vkQueueSubmit vkCmdPipelineBarrier vkCmdClearColorImage")
{
    struct wsi w;
    VkSwapchainCreateInfoKHR info;
    VkSwapchainKHR swapchain;
    VkFence fence;
    VkImage *images;
    uint32_t count = 0, frame;
    wsi_init(t, &w, 32, 32, 1);
    info = wsi_swapchain_info(&w, w.surface, VK_NULL_HANDLE);
    CT_REQUIRE_VK(t, vkCreateSwapchainKHR(w.e.device, &info, NULL, &swapchain));
    images = wsi_images(t, &w, swapchain, &count);
    fence = wsi_fence(t, &w);
    for (frame = 0; frame < 2 * count + 2; frame++) {
        uint32_t index = UINT32_MAX;
        VkResult r = vkAcquireNextImageKHR(w.e.device, swapchain, UINT64_MAX, VK_NULL_HANDLE, fence, &index);
        CT_REQUIREF(t, wsi_presentable(r), "frame %u: vkAcquireNextImageKHR returned %s", frame, ct_result_name(r));
        CT_REQUIRE(t, index < count);
        wsi_wait_fence(t, &w, fence);
        r = wsi_present_image(t, &w, swapchain, images[index], index, fence);
        CT_CHECKF(t, wsi_presentable(r), "frame %u: vkQueuePresentKHR returned %s", frame, ct_result_name(r));
    }
    CT_CHECK_VK(t, vkQueueWaitIdle(w.e.queue), VK_SUCCESS);
    vkDestroySwapchainKHR(w.e.device, swapchain, NULL);
    vkDestroyFence(w.e.device, fence, NULL);
    free(images);
    wsi_fini(&w);
}

static uint64_t wsi_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

CT_TEST(acquire_timeouts,
        "Window System Integration > WSI Swapchain (vkAcquireNextImageKHR): with timeout 0 the command does not "
        "wait and VK_NOT_READY means no image was available; VK_TIMEOUT means the (non-zero) timeout expired; up "
        "to imageCount - minImageCount + 1 images can be acquired; if no image is acquired, semaphore and fence "
        "are unaffected",
        "vkCreateSwapchainKHR vkGetSwapchainImagesKHR vkAcquireNextImageKHR vkDestroySwapchainKHR "
        "vkGetFenceStatus vkWaitForFences vkResetFences vkGetPhysicalDeviceSurfaceCapabilitiesKHR")
{
    struct wsi w;
    VkSwapchainCreateInfoKHR info;
    VkSwapchainKHR swapchain;
    VkFence fences[2];
    VkImage *images;
    uint32_t count = 0, i, held, index;
    uint64_t start;
    VkResult r;
    wsi_init(t, &w, 32, 32, 1);
    info = wsi_swapchain_info(&w, w.surface, VK_NULL_HANDLE);
    CT_REQUIRE_VK(t, vkCreateSwapchainKHR(w.e.device, &info, NULL, &swapchain));
    images = wsi_images(t, &w, swapchain, &count);
    fences[0] = wsi_fence(t, &w);
    fences[1] = wsi_fence(t, &w);
    /* Forward progress: these acquisitions must return in finite time. */
    held = count - w.caps.minImageCount + 1;
    for (i = 0; i < held; i++) {
        r = vkAcquireNextImageKHR(w.e.device, swapchain, UINT64_MAX, VK_NULL_HANDLE, fences[0], &index);
        CT_REQUIREF(t, wsi_presentable(r), "acquisition %u of %u: %s", i + 1, held, ct_result_name(r));
        wsi_wait_fence(t, &w, fences[0]);
    }
    /* More: an image or none, by the right code, and the fence untouched if none. */
    r = vkAcquireNextImageKHR(w.e.device, swapchain, 0, VK_NULL_HANDLE, fences[1], &index);
    CT_CHECKF(t, wsi_presentable(r) || r == VK_NOT_READY, "timeout 0: %s", ct_result_name(r));
    if (r == VK_NOT_READY) {
        CT_CHECK_VK(t, vkGetFenceStatus(w.e.device, fences[1]), VK_NOT_READY);
        start = wsi_now_ns();
        r = vkAcquireNextImageKHR(w.e.device, swapchain, 2000000, VK_NULL_HANDLE, fences[1], &index);
        CT_CHECKF(t, wsi_presentable(r) || r == VK_TIMEOUT, "timeout 2 ms: %s", ct_result_name(r));
        if (r == VK_TIMEOUT) {
            CT_CHECKF(t, wsi_now_ns() - start >= 2000000, "VK_TIMEOUT after %llu ns of a 2 ms timeout",
                      (unsigned long long)(wsi_now_ns() - start));
            CT_CHECK_VK(t, vkGetFenceStatus(w.e.device, fences[1]), VK_NOT_READY);
        }
    }
    if (wsi_presentable(r))
        CT_CHECK_VK(t, vkWaitForFences(w.e.device, 1, &fences[1], VK_TRUE, 10000000000ull), VK_SUCCESS);
    CT_CHECK_VK(t, vkDeviceWaitIdle(w.e.device), VK_SUCCESS);
    vkDestroySwapchainKHR(w.e.device, swapchain, NULL);
    vkDestroyFence(w.e.device, fences[0], NULL);
    vkDestroyFence(w.e.device, fences[1], NULL);
    free(images);
    wsi_fini(&w);
}

CT_TEST(acquire_next_image2,
        "Window System Integration > WSI Swapchain: vkAcquireNextImage2KHR with deviceMask 1 (a device of one "
        "physical device) acquires an image as vkAcquireNextImageKHR does (Vulkan 1.1)",
        "vkCreateSwapchainKHR vkGetSwapchainImagesKHR vkAcquireNextImage2KHR vkQueuePresentKHR vkDestroySwapchainKHR "
        "vkWaitForFences vkResetFences vkQueueSubmit vkCmdPipelineBarrier vkCmdClearColorImage")
{
    struct wsi w;
    VkSwapchainCreateInfoKHR info;
    VkAcquireNextImageInfoKHR acquire;
    VkSwapchainKHR swapchain;
    VkFence fence;
    VkImage *images;
    uint32_t count = 0, index = UINT32_MAX;
    VkResult r;
    wsi_init(t, &w, 32, 32, 1);
    if (w.e.version < VK_API_VERSION_1_1)
        ct_skip(t, "Vulkan 1.0");
    info = wsi_swapchain_info(&w, w.surface, VK_NULL_HANDLE);
    CT_REQUIRE_VK(t, vkCreateSwapchainKHR(w.e.device, &info, NULL, &swapchain));
    images = wsi_images(t, &w, swapchain, &count);
    fence = wsi_fence(t, &w);
    memset(&acquire, 0, sizeof(acquire));
    acquire.sType = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR;
    acquire.swapchain = swapchain;
    acquire.timeout = UINT64_MAX;
    acquire.fence = fence;
    acquire.deviceMask = 1;
    r = vkAcquireNextImage2KHR(w.e.device, &acquire, &index);
    CT_REQUIREF(t, wsi_presentable(r), "vkAcquireNextImage2KHR returned %s", ct_result_name(r));
    CT_REQUIRE(t, index < count);
    wsi_wait_fence(t, &w, fence);
    r = wsi_present_image(t, &w, swapchain, images[index], index, fence);
    CT_CHECKF(t, wsi_presentable(r), "vkQueuePresentKHR returned %s", ct_result_name(r));
    CT_CHECK_VK(t, vkQueueWaitIdle(w.e.queue), VK_SUCCESS);
    vkDestroySwapchainKHR(w.e.device, swapchain, NULL);
    vkDestroyFence(w.e.device, fence, NULL);
    free(images);
    wsi_fini(&w);
}

CT_TEST(swapchain_recreate,
        "Window System Integration > WSI Swapchain: creating a swapchain with oldSwapchain retires it; images "
        "already acquired from the retired swapchain can still be presented (VK_SUCCESS, VK_SUBOPTIMAL_KHR, or "
        "VK_ERROR_OUT_OF_DATE_KHR if it went out of date); the new swapchain acquires and presents; both are "
        "destroyed",
        "vkCreateSwapchainKHR vkGetSwapchainImagesKHR vkAcquireNextImageKHR vkQueuePresentKHR vkDestroySwapchainKHR "
        "vkWaitForFences vkResetFences vkQueueSubmit vkCmdPipelineBarrier vkCmdClearColorImage")
{
    struct wsi w;
    VkSwapchainCreateInfoKHR info;
    VkSwapchainKHR old, swapchain;
    VkFence fence;
    VkImage *old_images, *images;
    uint32_t old_count = 0, count = 0, held = UINT32_MAX, index = UINT32_MAX;
    VkResult r;
    wsi_init(t, &w, 32, 32, 1);
    info = wsi_swapchain_info(&w, w.surface, VK_NULL_HANDLE);
    CT_REQUIRE_VK(t, vkCreateSwapchainKHR(w.e.device, &info, NULL, &old));
    old_images = wsi_images(t, &w, old, &old_count);
    fence = wsi_fence(t, &w);
    r = vkAcquireNextImageKHR(w.e.device, old, UINT64_MAX, VK_NULL_HANDLE, fence, &held);
    CT_REQUIREF(t, wsi_presentable(r), "vkAcquireNextImageKHR returned %s", ct_result_name(r));
    wsi_wait_fence(t, &w, fence);

    info.oldSwapchain = old;
    CT_REQUIRE_VK(t, vkCreateSwapchainKHR(w.e.device, &info, NULL, &swapchain));
    r = wsi_present_image(t, &w, old, old_images[held], held, fence);
    CT_CHECKF(t, wsi_presentable(r) || r == VK_ERROR_OUT_OF_DATE_KHR, "present to the retired swapchain: %s",
              ct_result_name(r));
    images = wsi_images(t, &w, swapchain, &count);
    r = vkAcquireNextImageKHR(w.e.device, swapchain, UINT64_MAX, VK_NULL_HANDLE, fence, &index);
    CT_REQUIREF(t, wsi_presentable(r), "new swapchain: vkAcquireNextImageKHR returned %s", ct_result_name(r));
    CT_REQUIRE(t, index < count);
    wsi_wait_fence(t, &w, fence);
    r = wsi_present_image(t, &w, swapchain, images[index], index, fence);
    CT_CHECKF(t, wsi_presentable(r), "new swapchain: vkQueuePresentKHR returned %s", ct_result_name(r));
    CT_CHECK_VK(t, vkQueueWaitIdle(w.e.queue), VK_SUCCESS);
    vkDestroySwapchainKHR(w.e.device, old, NULL);
    vkDestroySwapchainKHR(w.e.device, swapchain, NULL);
    vkDestroyFence(w.e.device, fence, NULL);
    free(old_images);
    free(images);
    wsi_fini(&w);
}

CT_TEST(present_two_swapchains,
        "Window System Integration > WSI Swapchain (vkQueuePresentKHR): one present of two swapchains (two "
        "layers) sets each pResults entry to that swapchain's result; the return value is VK_SUCCESS or "
        "VK_SUBOPTIMAL_KHR when both are",
        "vkCreateMetalSurfaceEXT vkDestroySurfaceKHR vkCreateSwapchainKHR vkGetSwapchainImagesKHR "
        "vkAcquireNextImageKHR vkQueuePresentKHR vkDestroySwapchainKHR vkWaitForFences vkResetFences "
        "vkQueueSubmit vkCmdPipelineBarrier vkCmdClearColorImage vkGetPhysicalDeviceSurfaceSupportKHR "
        "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")
{
    static const float color[4] = { 1.0f, 1.0f, 0.0f, 1.0f };
    struct wsi w;
    struct wsi w2; /* the second surface's capabilities (w2.e unused) */
    VkMetalSurfaceCreateInfoEXT sinfo;
    VkSwapchainCreateInfoKHR info;
    VkSwapchainKHR swapchains[2];
    VkSurfaceKHR second;
    VkPresentInfoKHR present;
    VkResult results[2] = { VK_RESULT_MAX_ENUM, VK_RESULT_MAX_ENUM }, r;
    VkImage *images[2];
    VkFence fence;
    VkBool32 supported = VK_FALSE;
    VkCommandBuffer cb;
    uint32_t counts[2] = { 0, 0 }, indices[2], i;
    void *layer2;
    wsi_init(t, &w, 32, 32, 1);
    layer2 = wsi_layer(48, 16);
    CT_REQUIRE(t, layer2 != NULL);
    memset(&sinfo, 0, sizeof(sinfo));
    sinfo.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
    sinfo.pLayer = layer2;
    CT_REQUIRE_VK(t, vkCreateMetalSurfaceEXT(w.e.instance, &sinfo, NULL, &second));
    CT_REQUIRE_VK(t, vkGetPhysicalDeviceSurfaceSupportKHR(w.e.pd, w.e.queue_family, second, &supported));
    if (!supported)
        ct_skip(t, "queue family %u cannot present to the second surface", w.e.queue_family);
    w2 = w;
    CT_REQUIRE_VK(t, vkGetPhysicalDeviceSurfaceCapabilitiesKHR(w.e.pd, second, &w2.caps));
    i = 1;
    r = vkGetPhysicalDeviceSurfaceFormatsKHR(w.e.pd, second, &i, &w2.format);
    CT_REQUIRE(t, (r == VK_SUCCESS || r == VK_INCOMPLETE) && i == 1);
    w2.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    fence = wsi_fence(t, &w);
    info = wsi_swapchain_info(&w, w.surface, VK_NULL_HANDLE);
    CT_REQUIRE_VK(t, vkCreateSwapchainKHR(w.e.device, &info, NULL, &swapchains[0]));
    info = wsi_swapchain_info(&w2, second, VK_NULL_HANDLE);
    CT_REQUIRE_VK(t, vkCreateSwapchainKHR(w.e.device, &info, NULL, &swapchains[1]));
    cb = ct_begin_commands(t, &w.e);
    for (i = 0; i < 2; i++) {
        images[i] = wsi_images(t, &w, swapchains[i], &counts[i]);
        r = vkAcquireNextImageKHR(w.e.device, swapchains[i], UINT64_MAX, VK_NULL_HANDLE, fence, &indices[i]);
        CT_REQUIREF(t, wsi_presentable(r), "swapchain %u: vkAcquireNextImageKHR returned %s", i, ct_result_name(r));
        CT_REQUIRE(t, indices[i] < counts[i]);
        wsi_wait_fence(t, &w, fence);
        wsi_record_frame(i ? &w2 : &w, cb, images[i][indices[i]], color);
    }
    wsi_submit(t, &w, cb, VK_NULL_HANDLE, VK_NULL_HANDLE, fence);
    wsi_wait_fence(t, &w, fence);
    memset(&present, 0, sizeof(present));
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.swapchainCount = 2;
    present.pSwapchains = swapchains;
    present.pImageIndices = indices;
    present.pResults = results;
    r = vkQueuePresentKHR(w.e.queue, &present);
    CT_CHECKF(t, wsi_presentable(results[0]) && wsi_presentable(results[1]), "pResults: %s, %s",
              ct_result_name(results[0]), ct_result_name(results[1]));
    CT_CHECKF(t, r == (results[0] == VK_SUBOPTIMAL_KHR || results[1] == VK_SUBOPTIMAL_KHR ? VK_SUBOPTIMAL_KHR : VK_SUCCESS),
              "returned %s for pResults %s, %s", ct_result_name(r), ct_result_name(results[0]), ct_result_name(results[1]));
    CT_CHECK_VK(t, vkQueueWaitIdle(w.e.queue), VK_SUCCESS);
    for (i = 0; i < 2; i++) {
        vkDestroySwapchainKHR(w.e.device, swapchains[i], NULL);
        free(images[i]);
    }
    vkDestroyFence(w.e.device, fence, NULL);
    vkDestroySurfaceKHR(w.e.instance, second, NULL);
    wsi_release(layer2);
    wsi_fini(&w);
}

CT_TEST(swapchain_image_transfers,
        "Window System Integration > WSI Swapchain: presentable images are images of the swapchain's format and "
        "usage (Presentable Images): cleared as a transfer destination, copied to a buffer as a transfer source, "
        "their texels hold the clear color (B8G8R8A8_UNORM swapchain, when the surface offers it with "
        "TRANSFER_SRC and TRANSFER_DST usage)",
        "vkCreateSwapchainKHR vkGetSwapchainImagesKHR vkAcquireNextImageKHR vkQueuePresentKHR vkDestroySwapchainKHR "
        "vkGetPhysicalDeviceSurfaceFormatsKHR vkWaitForFences vkResetFences vkQueueSubmit vkCmdPipelineBarrier "
        "vkCmdClearColorImage vkCmdCopyImageToBuffer")
{
    static const float color[4] = { 0.0f, 1.0f, 0.0f, 1.0f }; /* B G R A bytes: 00 FF 00 FF */
    static const unsigned char texel[4] = { 0x00, 0xFF, 0x00, 0xFF };
    struct wsi w;
    VkSurfaceFormatKHR formats[256];
    VkSwapchainCreateInfoKHR info;
    VkSwapchainKHR swapchain;
    VkBufferImageCopy region;
    struct ct_buffer buffer;
    VkCommandBuffer cb;
    VkFence fence;
    VkImage *images;
    uint32_t count = CT_ARRAY_SIZE(formats), i, index = UINT32_MAX, bad = 0;
    int found = 0;
    VkResult r;
    wsi_init(t, &w, 16, 16, 1);
    CT_REQUIRE(t, vkGetPhysicalDeviceSurfaceFormatsKHR(w.e.pd, w.surface, &count, formats) >= 0);
    for (i = 0; i < count; i++)
        found |= formats[i].format == VK_FORMAT_B8G8R8A8_UNORM &&
                 formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    if (!found)
        ct_skip(t, "no B8G8R8A8_UNORM / SRGB_NONLINEAR surface format");
    w.format.format = VK_FORMAT_B8G8R8A8_UNORM;
    w.format.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    w.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if ((w.caps.supportedUsageFlags & w.usage) != w.usage)
        ct_skip(t, "the surface does not offer TRANSFER_SRC and TRANSFER_DST usage");
    info = wsi_swapchain_info(&w, w.surface, VK_NULL_HANDLE);
    CT_REQUIRE_VK(t, vkCreateSwapchainKHR(w.e.device, &info, NULL, &swapchain));
    images = wsi_images(t, &w, swapchain, &count);
    fence = wsi_fence(t, &w);
    r = vkAcquireNextImageKHR(w.e.device, swapchain, UINT64_MAX, VK_NULL_HANDLE, fence, &index);
    CT_REQUIREF(t, wsi_presentable(r), "vkAcquireNextImageKHR returned %s", ct_result_name(r));
    wsi_wait_fence(t, &w, fence);
    ct_host_buffer(t, &w.e, (VkDeviceSize)info.imageExtent.width * info.imageExtent.height * 4,
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer);
    memset(buffer.data, 0x55, (size_t)buffer.size);
    cb = ct_begin_commands(t, &w.e);
    wsi_record_frame(&w, cb, images[index], color); /* ends in PRESENT_SRC */
    ct_image_barrier(cb, images[index], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    memset(&region, 0, sizeof(region));
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width = info.imageExtent.width;
    region.imageExtent.height = info.imageExtent.height;
    region.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(cb, images[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer.buffer, 1, &region);
    ct_image_barrier(cb, images[index], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    ct_host_read_barrier(cb);
    wsi_submit(t, &w, cb, VK_NULL_HANDLE, VK_NULL_HANDLE, fence);
    wsi_wait_fence(t, &w, fence);
    for (i = 0; i < info.imageExtent.width * info.imageExtent.height; i++)
        bad += memcmp(buffer.data + 4 * i, texel, 4) != 0;
    CT_CHECKF(t, bad == 0, "%u of %u texels are not the clear color", bad,
              info.imageExtent.width * info.imageExtent.height);
    r = wsi_present(&w, swapchain, index, VK_NULL_HANDLE, NULL);
    CT_CHECKF(t, wsi_presentable(r), "vkQueuePresentKHR returned %s", ct_result_name(r));
    CT_CHECK_VK(t, vkQueueWaitIdle(w.e.queue), VK_SUCCESS);
    vkDestroySwapchainKHR(w.e.device, swapchain, NULL);
    vkDestroyFence(w.e.device, fence, NULL);
    ct_buffer_fini(&w.e, &buffer);
    free(images);
    wsi_fini(&w);
}

CT_TEST(swapchain_image_alias,
        "Window System Integration > WSI Swapchain (VK_KHR_swapchain with Vulkan 1.1): an image made with "
        "VkImageSwapchainCreateInfoKHR and bound with VkBindImageMemorySwapchainInfoKHR to an acquired "
        "swapchain image uses that image's memory: a clear through it is what the swapchain image holds "
        "(B8G8R8A8_UNORM swapchain with TRANSFER_SRC and TRANSFER_DST usage)",
        "vkCreateSwapchainKHR vkGetSwapchainImagesKHR vkAcquireNextImageKHR vkQueuePresentKHR vkDestroySwapchainKHR "
        "vkCreateImage vkBindImageMemory2 vkDestroyImage vkWaitForFences vkResetFences vkQueueSubmit "
        "vkCmdPipelineBarrier vkCmdClearColorImage vkCmdCopyImageToBuffer")
{
    static const float color[4] = { 1.0f, 0.0f, 1.0f, 0.0f }; /* B G R A bytes: FF 00 FF 00 */
    static const unsigned char texel[4] = { 0xFF, 0x00, 0xFF, 0x00 };
    struct wsi w;
    VkSurfaceFormatKHR formats[256];
    VkSwapchainCreateInfoKHR info;
    VkImageSwapchainCreateInfoKHR image_swapchain;
    VkBindImageMemorySwapchainInfoKHR bind_swapchain;
    VkBindImageMemoryInfo bind;
    VkImageCreateInfo image_info;
    VkSwapchainKHR swapchain;
    VkBufferImageCopy region;
    struct ct_buffer buffer;
    VkCommandBuffer cb;
    VkImage *images, alias;
    VkFence fence;
    uint32_t count = CT_ARRAY_SIZE(formats), i, index = UINT32_MAX, bad = 0;
    int found = 0;
    VkResult r;
    wsi_init(t, &w, 16, 16, 1);
    if (w.e.version < VK_API_VERSION_1_1)
        ct_skip(t, "Vulkan 1.0");
    CT_REQUIRE(t, vkGetPhysicalDeviceSurfaceFormatsKHR(w.e.pd, w.surface, &count, formats) >= 0);
    for (i = 0; i < count; i++)
        found |= formats[i].format == VK_FORMAT_B8G8R8A8_UNORM &&
                 formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    if (!found)
        ct_skip(t, "no B8G8R8A8_UNORM / SRGB_NONLINEAR surface format");
    w.format.format = VK_FORMAT_B8G8R8A8_UNORM;
    w.format.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    w.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if ((w.caps.supportedUsageFlags & w.usage) != w.usage)
        ct_skip(t, "the surface does not offer TRANSFER_SRC and TRANSFER_DST usage");
    info = wsi_swapchain_info(&w, w.surface, VK_NULL_HANDLE);
    CT_REQUIRE_VK(t, vkCreateSwapchainKHR(w.e.device, &info, NULL, &swapchain));
    images = wsi_images(t, &w, swapchain, &count);
    fence = wsi_fence(t, &w);
    r = vkAcquireNextImageKHR(w.e.device, swapchain, UINT64_MAX, VK_NULL_HANDLE, fence, &index);
    CT_REQUIRE(t, wsi_presentable(r) && index < count);
    wsi_wait_fence(t, &w, fence);

    /* The swapchain's implied image creation parameters. */
    memset(&image_swapchain, 0, sizeof(image_swapchain));
    image_swapchain.sType = VK_STRUCTURE_TYPE_IMAGE_SWAPCHAIN_CREATE_INFO_KHR;
    image_swapchain.swapchain = swapchain;
    image_info = ct_image_info(VK_IMAGE_TYPE_2D, info.imageFormat, info.imageExtent.width, info.imageExtent.height, 1,
                               1, 1, info.imageUsage);
    image_info.pNext = &image_swapchain;
    CT_REQUIRE_VK(t, vkCreateImage(w.e.device, &image_info, NULL, &alias));
    memset(&bind_swapchain, 0, sizeof(bind_swapchain));
    bind_swapchain.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHR;
    bind_swapchain.swapchain = swapchain;
    bind_swapchain.imageIndex = index;
    memset(&bind, 0, sizeof(bind));
    bind.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
    bind.pNext = &bind_swapchain;
    bind.image = alias;
    bind.memory = VK_NULL_HANDLE;
    CT_REQUIRE_VK(t, vkBindImageMemory2(w.e.device, 1, &bind));

    ct_host_buffer(t, &w.e, (VkDeviceSize)info.imageExtent.width * info.imageExtent.height * 4,
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer);
    memset(buffer.data, 0x55, (size_t)buffer.size);
    cb = ct_begin_commands(t, &w.e);
    wsi_record_frame(&w, cb, alias, color); /* clears the alias */
    ct_image_barrier(cb, images[index], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    memset(&region, 0, sizeof(region));
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width = info.imageExtent.width;
    region.imageExtent.height = info.imageExtent.height;
    region.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(cb, images[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer.buffer, 1, &region);
    ct_image_barrier(cb, images[index], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    ct_host_read_barrier(cb);
    wsi_submit(t, &w, cb, VK_NULL_HANDLE, VK_NULL_HANDLE, fence);
    wsi_wait_fence(t, &w, fence);
    for (i = 0; i < info.imageExtent.width * info.imageExtent.height; i++)
        bad += memcmp(buffer.data + 4 * i, texel, 4) != 0;
    CT_CHECKF(t, bad == 0, "%u of %u texels of the swapchain image are not the color cleared through the alias", bad,
              info.imageExtent.width * info.imageExtent.height);
    r = wsi_present(&w, swapchain, index, VK_NULL_HANDLE, NULL);
    CT_CHECKF(t, wsi_presentable(r), "vkQueuePresentKHR returned %s", ct_result_name(r));
    CT_CHECK_VK(t, vkQueueWaitIdle(w.e.queue), VK_SUCCESS);
    vkDestroyImage(w.e.device, alias, NULL);
    vkDestroySwapchainKHR(w.e.device, swapchain, NULL);
    vkDestroyFence(w.e.device, fence, NULL);
    ct_buffer_fini(&w.e, &buffer);
    free(images);
    wsi_fini(&w);
}

static const struct ct_test *const tests[] = {
    CT_REF(surface_create_destroy),
    CT_REF(surface_allocation_callbacks),
    CT_REF(surface_support),
    CT_REF(surface_capabilities),
    CT_REF(surface_formats),
    CT_REF(surface_present_modes),
    CT_REF(present_rectangles),
    CT_REF(device_group_present_modes),
    CT_REF(swapchain_create_destroy),
    CT_REF(swapchain_allocation_callbacks),
    CT_REF(native_window_in_use),
    CT_REF(native_window_in_use_other_surface),
    CT_REF(acquire_present_frames),
    CT_REF(acquire_fence),
    CT_REF(acquire_timeouts),
    CT_REF(acquire_next_image2),
    CT_REF(swapchain_recreate),
    CT_REF(present_two_swapchains),
    CT_REF(swapchain_image_transfers),
    CT_REF(swapchain_image_alias),
};

CT_SUITE(wsi, tests)
