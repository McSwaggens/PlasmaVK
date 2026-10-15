/* Startup: what an application pays before its first frame.
 *
 *   process_first     a fresh process per iteration (posix_spawn of this
 *                     program): exec, vkCreateInstance, the physical-device
 *                     queries and vkCreateDevice, a headless surface and
 *                     swapchain, then one cleared frame submitted, presented
 *                     and waited for. Each phase is timed in the child; the
 *                     headline is to_first_frame_us, from posix_spawn to the
 *                     frame's fence. first_submit_us is the part of the
 *                     frame until vkQueueSubmit returns (command pool,
 *                     acquire, recording, the submit). Of the two
 *                     unmeasured launches before them, the first also
 *                     copies one pixel of its frame to a host buffer and
 *                     checks it.
 *   process_first_cg  the same, but the child initializes CoreGraphics
 *                     (CGMainDisplayID) before Vulkan, as every windowed
 *                     application (AppKit, SDL, GLFW) has by then. Work that
 *                     CoreGraphics does once per process is then already
 *                     paid, and coregraphics_us shows what it cost.
 *   instance_device   instance + device creation and teardown repeated in
 *                     one process: after the first, the process-wide work
 *                     (frameworks, the Metal device) is done.
 *
 * Like a real application it enumerates device extensions to find (and
 * enable) VK_KHR_portability_subset.
 */
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <math.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include "pvk_bench.h"

extern char **environ;

#define FRAME_WIDTH 1920
#define FRAME_HEIGHT 1080

/* ==========================================================================
 * The child: one launch, timed phase by phase
 * ========================================================================== */

enum {
    T_MAIN,
    T_COREGRAPHICS,
    T_INSTANCE,
    T_DEVICE,
    T_SWAPCHAIN,
    T_SUBMIT,
    T_FRAME,
    T_TEARDOWN,
    T_COUNT
};

/* Absolute times (bench_now_us: mach_absolute_time, the same clock in every
 * process) of the end of each phase, printed as one line for the parent. */
static double child_t[T_COUNT];

/* Child exit codes besides 0. A failure or a wrong pixel is an error; an
 * unsupported surface is a skip. */
enum { CHILD_FAILED = 1, CHILD_UNSUPPORTED = 2, CHILD_WRONG = 3 };

#define CHILD_CHECK(call)                                                       \
    do {                                                                        \
        if ((call) != VK_SUCCESS) {                                             \
            fprintf(stderr, "bench_startup child: %s failed\n", #call);         \
            return CHILD_FAILED;                                                \
        }                                                                       \
    } while (0)

/* The clear color as an 8-bit swapchain format stores it, as 0xAABBGGRR
 * in memory order; 0 for any other format. */
static uint32_t child_expected_pixel(VkFormat format, const float rgba[4])
{
    int srgb = format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_R8G8B8A8_SRGB;
    int bgra = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
    uint32_t out = 0;
    int k;
    if (!srgb && !bgra && format != VK_FORMAT_R8G8B8A8_UNORM)
        return 0;
    for (k = 0; k < 4; k++) {
        float v = rgba[k];
        if (srgb && k < 3)
            v = v <= 0.0031308f ? v * 12.92f : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
        out |= (uint32_t)(v * 255.0f + 0.5f) << (8 * (bgra && k < 3 ? 2 - k : k));
    }
    return out;
}

static PFN_vkVoidFunction child_instance_proc(VkInstance instance, const char *name)
{
    PFN_vkVoidFunction fn = vkGetInstanceProcAddr(instance, name);
    if (!fn)
        fprintf(stderr, "bench_startup child: vkGetInstanceProcAddr(%s) returned NULL\n", name);
    return fn;
}

static PFN_vkVoidFunction child_device_proc(VkDevice device, const char *name)
{
    PFN_vkVoidFunction fn = vkGetDeviceProcAddr(device, name);
    if (!fn)
        fprintf(stderr, "bench_startup child: vkGetDeviceProcAddr(%s) returned NULL\n", name);
    return fn;
}

/* CGMainDisplayID through dlopen, so the program itself never links
 * CoreGraphics. */
static int child_init_coregraphics(void)
{
    void *cg = dlopen("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics", RTLD_LAZY);
    void *sym = cg ? dlsym(cg, "CGMainDisplayID") : NULL;
    uint32_t (*main_display)(void);
    if (!sym)
        return 0;
    memcpy(&main_display, &sym, sizeof(main_display));
    return main_display() != 0;
}

/* verify: also copy the middle pixel of the frame to a host buffer and
 * check it. */
static int child_main(int coregraphics, int verify)
{
    const char *instance_exts[3];
    const char *device_exts[2];
    VkExtensionProperties exts[512];
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "plasmavk-bench",
        .apiVersion = VK_API_VERSION_1_4,
    };
    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
        .ppEnabledExtensionNames = instance_exts,
    };
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceVulkan13Features features13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    VkPhysicalDeviceFeatures2 features = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features13 };
    VkQueueFamilyProperties families[16];
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .ppEnabledExtensionNames = device_exts,
    };
    VkHeadlessSurfaceCreateInfoEXT surface_info = { .sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT };
    VkSwapchainCreateInfoKHR swapchain_info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = VK_PRESENT_MODE_FIFO_KHR,
        .clipped = VK_TRUE,
    };
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    };
    VkCommandBufferAllocateInfo cmd_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VkSemaphoreCreateInfo semaphore_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkImageMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    VkClearColorValue clear = { { 0.25f, 0.5f, 0.75f, 1.0f } };
    VkBufferCreateInfo readback_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = sizeof(uint32_t),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    VkMemoryAllocateInfo readback_alloc = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    VkBufferImageCopy pixel_copy = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                     .imageExtent = { 1, 1, 1 } };
    VkBufferMemoryBarrier readback_barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .size = VK_WHOLE_SIZE,
    };
    VkPhysicalDeviceMemoryProperties memory_props;
    VkMemoryRequirements requirements;
    VkBuffer readback = VK_NULL_HANDLE;
    VkDeviceMemory readback_memory = VK_NULL_HANDLE;
    void *readback_map = NULL;
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .signalSemaphoreCount = 1,
    };
    VkPresentInfoKHR present = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .swapchainCount = 1,
    };
    PFN_vkCreateHeadlessSurfaceEXT create_surface;
    PFN_vkDestroySurfaceKHR destroy_surface;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR surface_support;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR surface_caps;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR surface_formats;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR surface_modes;
    PFN_vkCreateSwapchainKHR create_swapchain;
    PFN_vkDestroySwapchainKHR destroy_swapchain;
    PFN_vkGetSwapchainImagesKHR swapchain_images;
    PFN_vkAcquireNextImageKHR acquire;
    PFN_vkQueuePresentKHR queue_present;
    VkSurfaceCapabilitiesKHR caps;
    VkSurfaceFormatKHR formats[64];
    VkPresentModeKHR modes[8];
    VkImage images[8];
    VkInstance instance;
    VkPhysicalDevice pd;
    VkDevice device;
    VkQueue queue;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    VkCommandPool pool;
    VkCommandBuffer cmd;
    VkSemaphore acquired, rendered;
    VkFence fence;
    VkBool32 supported = VK_FALSE;
    uint32_t count, i, ext_count = 512, image_index = 0, pixel = 0, want = 0;
    int has_portability = 0, has_headless = 0;

    if (coregraphics && !child_init_coregraphics()) {
        fprintf(stderr, "bench_startup child: CoreGraphics did not initialize\n");
        return CHILD_FAILED;
    }
    child_t[T_COREGRAPHICS] = bench_now_us();

    /* Instance: the surface extensions, and portability enumeration so the
     * loader exposes MoltenVK too. */
    if (vkEnumerateInstanceExtensionProperties(NULL, &ext_count, exts) < 0)
        ext_count = 0;
    for (i = 0; i < ext_count; i++) {
        has_portability |= strcmp(exts[i].extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
        has_headless |= strcmp(exts[i].extensionName, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME) == 0;
    }
    if (!has_headless) {
        fprintf(stderr, "bench_startup child: no VK_EXT_headless_surface\n");
        return CHILD_UNSUPPORTED;
    }
    instance_exts[instance_info.enabledExtensionCount++] = VK_KHR_SURFACE_EXTENSION_NAME;
    instance_exts[instance_info.enabledExtensionCount++] = VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME;
    if (has_portability) {
        instance_exts[instance_info.enabledExtensionCount++] = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
        instance_info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    CHILD_CHECK(vkCreateInstance(&instance_info, NULL, &instance));
    child_t[T_INSTANCE] = bench_now_us();

    /* Device: the queries an engine makes before it, then the device. */
    count = 1;
    if (vkEnumeratePhysicalDevices(instance, &count, &pd) < 0 || count == 0)
        return CHILD_FAILED;
    vkGetPhysicalDeviceProperties(pd, &props);
    vkGetPhysicalDeviceFeatures2(pd, &features);
    count = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, families);
    for (i = 0; i < count && !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT); i++)
        ;
    if (i == count)
        return CHILD_FAILED;
    queue_info.queueFamilyIndex = pool_info.queueFamilyIndex = i;
    ext_count = 512;
    if (vkEnumerateDeviceExtensionProperties(pd, NULL, &ext_count, exts) < 0)
        ext_count = 0;
    device_exts[device_info.enabledExtensionCount++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    for (i = 0; i < ext_count; i++)
        if (strcmp(exts[i].extensionName, "VK_KHR_portability_subset") == 0)
            device_exts[device_info.enabledExtensionCount++] = "VK_KHR_portability_subset";
    CHILD_CHECK(vkCreateDevice(pd, &device_info, NULL, &device));
    vkGetDeviceQueue(device, queue_info.queueFamilyIndex, 0, &queue);
    child_t[T_DEVICE] = bench_now_us();

    /* Surface and swapchain. */
    create_surface = (PFN_vkCreateHeadlessSurfaceEXT)child_instance_proc(instance, "vkCreateHeadlessSurfaceEXT");
    destroy_surface = (PFN_vkDestroySurfaceKHR)child_instance_proc(instance, "vkDestroySurfaceKHR");
    surface_support = (PFN_vkGetPhysicalDeviceSurfaceSupportKHR)child_instance_proc(
        instance, "vkGetPhysicalDeviceSurfaceSupportKHR");
    surface_caps = (PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)child_instance_proc(
        instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    surface_formats = (PFN_vkGetPhysicalDeviceSurfaceFormatsKHR)child_instance_proc(
        instance, "vkGetPhysicalDeviceSurfaceFormatsKHR");
    surface_modes = (PFN_vkGetPhysicalDeviceSurfacePresentModesKHR)child_instance_proc(
        instance, "vkGetPhysicalDeviceSurfacePresentModesKHR");
    create_swapchain = (PFN_vkCreateSwapchainKHR)child_device_proc(device, "vkCreateSwapchainKHR");
    destroy_swapchain = (PFN_vkDestroySwapchainKHR)child_device_proc(device, "vkDestroySwapchainKHR");
    swapchain_images = (PFN_vkGetSwapchainImagesKHR)child_device_proc(device, "vkGetSwapchainImagesKHR");
    acquire = (PFN_vkAcquireNextImageKHR)child_device_proc(device, "vkAcquireNextImageKHR");
    queue_present = (PFN_vkQueuePresentKHR)child_device_proc(device, "vkQueuePresentKHR");
    if (!create_surface || !destroy_surface || !surface_support || !surface_caps || !surface_formats ||
        !surface_modes || !create_swapchain || !destroy_swapchain || !swapchain_images || !acquire || !queue_present)
        return CHILD_FAILED;
    CHILD_CHECK(create_surface(instance, &surface_info, NULL, &surface));
    CHILD_CHECK(surface_support(pd, queue_info.queueFamilyIndex, surface, &supported));
    CHILD_CHECK(surface_caps(pd, surface, &caps));
    count = 64;
    if (surface_formats(pd, surface, &count, formats) < 0 || count == 0)
        return CHILD_FAILED;
    for (i = 0; i < count && !want; i++)
        if ((want = child_expected_pixel(formats[i].format, clear.float32)) != 0) {
            swapchain_info.imageFormat = formats[i].format;
            swapchain_info.imageColorSpace = formats[i].colorSpace;
        }
    if (verify)
        swapchain_info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (!supported || !want || (caps.supportedUsageFlags & swapchain_info.imageUsage) != swapchain_info.imageUsage) {
        fprintf(stderr, "bench_startup child: the headless surface has no 8-bit RGBA format that transfers "
                        "can clear and read\n");
        return CHILD_UNSUPPORTED;
    }
    swapchain_info.surface = surface;
    swapchain_info.minImageCount = caps.minImageCount > 2 ? caps.minImageCount : 2;
    if (caps.maxImageCount && swapchain_info.minImageCount > caps.maxImageCount)
        swapchain_info.minImageCount = caps.maxImageCount;
    swapchain_info.imageExtent.width = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent.width : FRAME_WIDTH;
    swapchain_info.imageExtent.height = caps.currentExtent.height != UINT32_MAX ? caps.currentExtent.height : FRAME_HEIGHT;
    count = 8;
    if (surface_modes(pd, surface, &count, modes) >= 0)
        for (i = 0; i < count; i++)
            if (modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)
                swapchain_info.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR; /* no pacing, as in the harness */
    CHILD_CHECK(create_swapchain(device, &swapchain_info, NULL, &swapchain));
    count = 8;
    CHILD_CHECK(swapchain_images(device, swapchain, &count, images));
    child_t[T_SWAPCHAIN] = bench_now_us();

    /* The first frame: acquire, clear, present, wait. */
    if (verify) { /* and a host buffer for its middle pixel */
        CHILD_CHECK(vkCreateBuffer(device, &readback_info, NULL, &readback));
        vkGetBufferMemoryRequirements(device, readback, &requirements);
        vkGetPhysicalDeviceMemoryProperties(pd, &memory_props);
        for (i = 0; i < memory_props.memoryTypeCount; i++)
            if ((requirements.memoryTypeBits >> i & 1u) &&
                (memory_props.memoryTypes[i].propertyFlags &
                 (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                break;
        if (i == memory_props.memoryTypeCount)
            return CHILD_FAILED;
        readback_alloc.allocationSize = requirements.size;
        readback_alloc.memoryTypeIndex = i;
        CHILD_CHECK(vkAllocateMemory(device, &readback_alloc, NULL, &readback_memory));
        CHILD_CHECK(vkBindBufferMemory(device, readback, readback_memory, 0));
        CHILD_CHECK(vkMapMemory(device, readback_memory, 0, VK_WHOLE_SIZE, 0, &readback_map));
        pixel_copy.imageOffset.x = (int32_t)swapchain_info.imageExtent.width / 2;
        pixel_copy.imageOffset.y = (int32_t)swapchain_info.imageExtent.height / 2;
        readback_barrier.buffer = readback;
    }
    CHILD_CHECK(vkCreateCommandPool(device, &pool_info, NULL, &pool));
    cmd_info.commandPool = pool;
    CHILD_CHECK(vkAllocateCommandBuffers(device, &cmd_info, &cmd));
    CHILD_CHECK(vkCreateSemaphore(device, &semaphore_info, NULL, &acquired));
    CHILD_CHECK(vkCreateSemaphore(device, &semaphore_info, NULL, &rendered));
    CHILD_CHECK(vkCreateFence(device, &fence_info, NULL, &fence));
    CHILD_CHECK(acquire(device, swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &image_index));
    CHILD_CHECK(vkBeginCommandBuffer(cmd, &begin));
    barrier.image = images[image_index];
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1,
                         &barrier);
    vkCmdClearColorImage(cmd, images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1,
                         &barrier.subresourceRange);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    if (verify) {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                             1, &barrier);
        vkCmdCopyImageToBuffer(cmd, images[image_index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1,
                               &pixel_copy);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1,
                             &readback_barrier, 0, NULL);
        barrier.srcAccessMask = 0;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }
    barrier.dstAccessMask = 0;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0,
                         NULL, 1, &barrier);
    CHILD_CHECK(vkEndCommandBuffer(cmd));
    submit.pWaitSemaphores = &acquired;
    submit.pCommandBuffers = &cmd;
    submit.pSignalSemaphores = &rendered;
    CHILD_CHECK(vkQueueSubmit(queue, 1, &submit, fence));
    child_t[T_SUBMIT] = bench_now_us();
    present.pWaitSemaphores = &rendered;
    present.pSwapchains = &swapchain;
    present.pImageIndices = &image_index;
    CHILD_CHECK(queue_present(queue, &present));
    CHILD_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ull));
    child_t[T_FRAME] = bench_now_us();
    if (verify)
        memcpy(&pixel, readback_map, sizeof(pixel));

    vkDeviceWaitIdle(device);
    vkDestroyFence(device, fence, NULL);
    vkDestroySemaphore(device, acquired, NULL);
    vkDestroySemaphore(device, rendered, NULL);
    vkDestroyCommandPool(device, pool, NULL);
    if (verify) {
        vkDestroyBuffer(device, readback, NULL);
        vkFreeMemory(device, readback_memory, NULL);
    }
    destroy_swapchain(device, swapchain, NULL);
    destroy_surface(instance, surface, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    child_t[T_TEARDOWN] = bench_now_us();

    /* Within 1 per channel: the clear's float-to-unorm rounding. */
    for (i = 0; verify && i < 32; i += 8) {
        int d = (int)(pixel >> i & 0xFFu) - (int)(want >> i & 0xFFu);
        if (d < -1 || d > 1) {
            printf("error\tthe first frame's pixel (%d, %d) is 0x%08x, want 0x%08x (VkFormat %d)\n",
                   pixel_copy.imageOffset.x, pixel_copy.imageOffset.y, pixel, want, (int)swapchain_info.imageFormat);
            return CHILD_WRONG;
        }
    }
    printf("%s", props.deviceName);
    for (i = 0; i < T_COUNT; i++)
        printf("\t%.3f", child_t[i]);
    printf("\n");
    return 0;
}

/* ==========================================================================
 * The parent: spawns the children and reports
 * ========================================================================== */

struct launch {
    double spawn, exit; /* in the parent */
    double t[T_COUNT];  /* in the child */
};

enum { LAUNCHED, LAUNCH_SKIP, LAUNCH_ERROR };

/* Runs one child; returns LAUNCHED, or LAUNCH_SKIP or LAUNCH_ERROR with the
 * reason in why. A child that fails, crashes or presents a wrong frame is an
 * error; only an unsupported headless surface is a skip. */
static int launch(const char *self, int coregraphics, int verify, struct launch *l, char *why, size_t why_size)
{
    char *args[5];
    char out[1024], device[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    posix_spawn_file_actions_t actions;
    pid_t pid;
    int fds[2], status = 0, i;
    size_t len = 0;
    ssize_t n;
    const char *line;

    args[0] = (char *)self;
    args[1] = "--child";
    args[2] = coregraphics ? "coregraphics" : "plain";
    args[3] = verify ? "verify" : NULL;
    args[4] = NULL;
    if (pipe(fds) != 0) {
        snprintf(why, why_size, "pipe failed");
        return LAUNCH_SKIP;
    }
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, fds[0]);
    posix_spawn_file_actions_addclose(&actions, fds[1]);
    l->spawn = bench_now_us();
    status = posix_spawn(&pid, self, &actions, NULL, args, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(fds[1]);
    if (status != 0) {
        close(fds[0]);
        snprintf(why, why_size, "posix_spawn failed");
        return LAUNCH_SKIP;
    }
    while (len + 1 < sizeof(out) && (n = read(fds[0], out + len, sizeof(out) - 1 - len)) > 0)
        len += (size_t)n;
    out[len] = '\0';
    close(fds[0]);
    if (waitpid(pid, &status, 0) != pid) {
        snprintf(why, why_size, "waitpid failed");
        return LAUNCH_SKIP;
    }
    l->exit = bench_now_us();
    if (WIFSIGNALED(status)) {
        snprintf(why, why_size, "the child process was killed by signal %d", WTERMSIG(status));
        return LAUNCH_ERROR;
    }
    if (!WIFEXITED(status)) {
        snprintf(why, why_size, "the child process did not exit");
        return LAUNCH_ERROR;
    }
    switch (WEXITSTATUS(status)) {
    case 0:
        break;
    case CHILD_UNSUPPORTED:
        snprintf(why, why_size, "VK_EXT_headless_surface with an 8-bit RGBA format and transfer-source and "
                                "transfer-destination swapchain images not supported");
        return LAUNCH_SKIP;
    case CHILD_WRONG: /* "error\treason" */
        line = strchr(out, '\t');
        snprintf(why, why_size, "%s", line ? line + 1 : "the first frame is wrong");
        why[strcspn(why, "\n")] = '\0';
        return LAUNCH_ERROR;
    default:
        snprintf(why, why_size, "the child process failed with exit status %d (its stderr says why)",
                 WEXITSTATUS(status));
        return LAUNCH_ERROR;
    }

    /* "device name\tt0\tt1..." */
    line = strchr(out, '\t');
    if (!line || (size_t)(line - out) >= sizeof(device)) {
        snprintf(why, why_size, "the child printed no timings");
        return LAUNCH_ERROR;
    }
    memcpy(device, out, (size_t)(line - out));
    device[line - out] = '\0';
    if (!bench_device[0])
        bench_set_device(device);
    for (i = 0; i < T_COUNT; i++) {
        char *end;
        l->t[i] = strtod(line + 1, &end);
        if (end == line + 1) {
            snprintf(why, why_size, "the child printed no timings");
            return LAUNCH_ERROR;
        }
        line = end;
    }
    return LAUNCHED;
}

#define SERIES 10 /* the measurements per launch */

static void process_first(const char *config, int coregraphics, int iterations)
{
    char self[4096];
    uint32_t size = sizeof(self);
    struct launch l;
    double *s[SERIES];
    struct bench_json j;
    char why[512];
    int i, k, outcome = LAUNCHED;

    if (!bench_selected(config))
        return;
    if (_NSGetExecutablePath(self, &size) != 0) {
        bench_skip(config, "_NSGetExecutablePath failed");
        return;
    }
    for (k = 0; k < SERIES; k++)
        s[k] = calloc((size_t)iterations, sizeof(double));
    /* Two unmeasured launches: the first after a build fills the OS shader
     * cache, which an installed application finds warm. The first also
     * checks its frame. */
    for (i = -2; i < iterations && outcome == LAUNCHED; i++) {
        outcome = launch(self, coregraphics, i == -2, &l, why, sizeof(why));
        if (outcome != LAUNCHED || i < 0)
            continue;
        s[0][i] = l.t[T_MAIN] - l.spawn;
        s[1][i] = l.t[T_COREGRAPHICS] - l.t[T_MAIN];
        s[2][i] = l.t[T_INSTANCE] - l.t[T_COREGRAPHICS];
        s[3][i] = l.t[T_DEVICE] - l.t[T_INSTANCE];
        s[4][i] = l.t[T_SWAPCHAIN] - l.t[T_DEVICE];
        s[5][i] = l.t[T_FRAME] - l.t[T_SWAPCHAIN];
        s[6][i] = l.t[T_FRAME] - l.spawn;
        s[7][i] = l.t[T_TEARDOWN] - l.t[T_FRAME];
        s[8][i] = l.exit - l.spawn;
        s[9][i] = l.t[T_SUBMIT] - l.t[T_SWAPCHAIN];
    }
    if (outcome == LAUNCH_SKIP) {
        bench_skip(config, "%s", why);
    } else if (outcome == LAUNCH_ERROR) {
        bench_error(config, "%s", why);
    } else {
        bench_json_begin(&j, config);
        bench_json_int(&j, "iterations", iterations);
        bench_json_bool(&j, "coregraphics_initialized", coregraphics);
        bench_json_int(&j, "width", FRAME_WIDTH);
        bench_json_int(&j, "height", FRAME_HEIGHT);
        bench_json_stats(&j, "exec_us", s[0], iterations);
        if (coregraphics)
            bench_json_stats(&j, "coregraphics_us", s[1], iterations);
        bench_json_stats(&j, "instance_us", s[2], iterations);
        bench_json_stats(&j, "device_us", s[3], iterations);
        bench_json_stats(&j, "swapchain_us", s[4], iterations);
        bench_json_stats(&j, "first_frame_us", s[5], iterations);
        bench_json_stats(&j, "first_submit_us", s[9], iterations);
        bench_json_stats(&j, "to_first_frame_us", s[6], iterations);
        bench_json_stats(&j, "teardown_us", s[7], iterations);
        bench_json_stats(&j, "process_us", s[8], iterations);
        bench_json_end(&j);
    }
    for (k = 0; k < SERIES; k++)
        free(s[k]);
}

/* ==========================================================================
 * instance_device: creation and teardown in one process
 * ========================================================================== */

static const char *startup(void *user)
{
    const char *instance_exts[] = { VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME };
    const char *device_exts[1];
    VkExtensionProperties exts[512];
    VkApplicationInfo app;
    VkPhysicalDeviceProperties props;
    float priority = 1.0f;
    VkInstanceCreateInfo instance_info;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo device_info;
    VkInstance instance;
    VkPhysicalDevice pd;
    VkDevice device;
    uint32_t count = 1, ext_count = 512, i;
    (void)user;

    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "plasmavk-bench";
    app.apiVersion = VK_API_VERSION_1_4;

    /* The portability flag lets the loader expose MoltenVK too. */
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    instance_info.pApplicationInfo = &app;
    instance_info.enabledExtensionCount = 1;
    instance_info.ppEnabledExtensionNames = instance_exts;
    if (vkCreateInstance(&instance_info, NULL, &instance) != VK_SUCCESS)
        return "vkCreateInstance failed";
    if (vkEnumeratePhysicalDevices(instance, &count, &pd) < 0 || count == 0) {
        vkDestroyInstance(instance, NULL);
        return "no physical device";
    }
    if (!bench_device[0]) {
        memset(&props, 0, sizeof(props));
        vkGetPhysicalDeviceProperties(pd, &props);
        bench_set_device(props.deviceName);
    }

    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.ppEnabledExtensionNames = device_exts;
    if (vkEnumerateDeviceExtensionProperties(pd, NULL, &ext_count, exts) < 0)
        ext_count = 0;
    for (i = 0; i < ext_count; i++)
        if (strcmp(exts[i].extensionName, "VK_KHR_portability_subset") == 0)
            device_exts[device_info.enabledExtensionCount++] = "VK_KHR_portability_subset";
    if (vkCreateDevice(pd, &device_info, NULL, &device) != VK_SUCCESS) {
        vkDestroyInstance(instance, NULL);
        return "vkCreateDevice failed";
    }

    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return NULL;
}

int main(int argc, char **argv)
{
    if ((argc == 3 || argc == 4) && strcmp(argv[1], "--child") == 0) {
        child_t[T_MAIN] = bench_now_us();
        return child_main(strcmp(argv[2], "coregraphics") == 0, argc == 4 && strcmp(argv[3], "verify") == 0);
    }
    bench_begin("startup");
    /* Fresh processes first, before this one has touched Vulkan or Metal. */
    process_first("process_first", 0, bench_frames(20));
    process_first("process_first_cg", 1, bench_frames(20));
    bench_run("instance_device", 5, bench_frames(100), startup, NULL);
    return bench_exit_code();
}
