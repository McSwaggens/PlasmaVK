/* Presentation: the frame loop of a windowed game, presenting to a real
 * CAMetalLayer through VK_EXT_metal_surface. The layer is in no window, so
 * nothing appears on screen, but presents take the windowed path: drawables,
 * -nextDrawable, and whatever the implementation does to get the image into
 * a drawable. The other benchmarks never present, and the macro harness
 * presents to headless surfaces, which have no drawables.
 *
 * Per frame: acquire; one command buffer that renders a scene (2000
 * overlapping rectangles) into an RGBA16F target and tonemaps it into the
 * swapchain image; submit (waiting for the acquire, signalling the present's
 * semaphore); present; then a command buffer with only a timestamp, to see
 * when the present's GPU work is done.
 *
 *   layer_1080p, layer_2160p, layer_2880p   1920x1080, 3840x2160 and
 *                                           5120x2880 swapchains on the layer
 *   headless_1080p, headless_2160p,         the same frames presented to a
 *   headless_2880p                          VK_EXT_headless_surface: no
 *                                           drawable, the control
 *
 * Reported per frame:
 *   gpu_us          the frame's command buffer (timestamps at its start and end)
 *   present_gpu_us  from the end of that command buffer to the timestamp after
 *                   the present: the GPU time the present takes from the
 *                   frame, plus anything else the implementation runs between
 *                   the two submissions, such as resolving the frame's
 *                   timestamps; the headless control has all but the present
 *   acquire_us, cpu_us (reset, record and vkQueueSubmit), present_us
 *                   CPU time of the calls
 *
 * A layer outside a window releases drawables at the display's rate, so the
 * loop runs at 60 frames per second, as a vsynced game does: present then
 * never waits for a drawable. present_gpu_us only counts frames whose marker
 * was submitted while their command buffer could still be running (less CPU
 * time from vkQueueSubmit to the marker's submission than gpu_us), so the
 * GPU had no reason to go idle in between; present_frames says how many.
 * At this pace the GPU is busy only a few percent of the time and may not
 * clock up as in a GPU-bound game, so absolute GPU times are higher than
 * there: compare a layer configuration with its headless control (the
 * difference is what presenting to the layer adds) rather than absolute
 * values across implementations. A configuration whose acquire and present
 * take over 100 ms on 3 frames is skipped: the layer is not handing out
 * drawables (the display is asleep, say), and its numbers would be stalls.
 * The first frame of each configuration is read back from the swapchain
 * image and compared with a CPU tonemapping of the scene (1 per channel).
 */
#include <objc/message.h>
#include <objc/runtime.h>
#include <time.h>

#include "bench_vk.h"
#include "bench_shaders.h"

/* CAMetalLayer, and the Objective-C runtime to make one (the PlasmaVK build
 * gets them through plasmavk.h already; the loader build needs them here). */
__asm__(".linker_option \"-framework\", \"QuartzCore\"\n"
        ".linker_option \"-lobjc\"\n");

#define RECTS 2000
#define WARMUP 30
#define FRAME_INTERVAL_US 16666.7 /* 60 Hz */
#define STALL_US 100000.0         /* acquire + present of a frame the layer stalled */
#define STALLS 3
#define SCENE_FORMAT VK_FORMAT_R16G16B16A16_SFLOAT
#define SWAPCHAIN_FORMAT VK_FORMAT_B8G8R8A8_UNORM

struct config {
    const char *name;
    int layer; /* 0: headless surface */
    uint32_t width, height;
    int frames;
};

static const struct config configs[] = {
    { "layer_1080p", 1, 1920, 1080, 150 },
    { "layer_2160p", 1, 3840, 2160, 150 },
    { "layer_2880p", 1, 5120, 2880, 150 },
    { "headless_1080p", 0, 1920, 1080, 150 },
    { "headless_2160p", 0, 3840, 2160, 150 },
    { "headless_2880p", 0, 5120, 2880, 150 },
};

/* rect_instanced.vert's per-instance attributes: HDR colors. */
struct instance {
    float rect[4];
    float color[4];
};

/* Instance commands, which bench_vk.h does not load. */
static struct {
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR GetPhysicalDeviceSurfaceSupportKHR;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR GetPhysicalDeviceSurfaceCapabilitiesKHR;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR GetPhysicalDeviceSurfaceFormatsKHR;
    PFN_vkCreateMetalSurfaceEXT CreateMetalSurfaceEXT;
    PFN_vkCreateHeadlessSurfaceEXT CreateHeadlessSurfaceEXT;
    PFN_vkDestroySurfaceKHR DestroySurfaceKHR;
} vkx;

static int have_headless; /* VK_EXT_headless_surface enabled */

/* ---- Instance and device ---- */

/* bench_vk_init with the surface extensions on the instance and
 * VK_KHR_swapchain on the device. */
static void init(struct bench_vk *vk)
{
    struct bench_vk_wants wants;

    memset(&wants, 0, sizeof(wants));
    wants.instance_extensions[0] = VK_KHR_SURFACE_EXTENSION_NAME;
    wants.instance_extensions[1] = VK_EXT_METAL_SURFACE_EXTENSION_NAME;
    wants.instance_extensions[2] = VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME;
    wants.device_extensions[0] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    bench_vk_init_with(vk, BENCH_NEED_DYNAMIC_RENDERING | BENCH_NEED_SYNC2, &wants);
    if (!bench_vk_extension(vk, VK_KHR_SURFACE_EXTENSION_NAME) ||
        !bench_vk_extension(vk, VK_EXT_METAL_SURFACE_EXTENSION_NAME))
        bench_fatal("VK_KHR_surface or VK_EXT_metal_surface not supported");
    if (!bench_vk_extension(vk, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
        bench_fatal("VK_KHR_swapchain not supported");
    if (!vkd.CreateSwapchainKHR || !vkd.DestroySwapchainKHR || !vkd.GetSwapchainImagesKHR ||
        !vkd.AcquireNextImageKHR || !vkd.QueuePresentKHR)
        bench_fatal("vkGetDeviceProcAddr returned NULL for a VK_KHR_swapchain command");
    have_headless = bench_vk_extension(vk, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
    if (!vk->timestamp_bits)
        bench_fatal("the queue has no timestamps");
#define LOAD_INSTANCE(name)                                                                 \
    vkx.name = (PFN_vk##name)vkGetInstanceProcAddr(vk->instance, "vk" #name);               \
    if (!vkx.name)                                                                          \
        bench_fatal("vkGetInstanceProcAddr(vk" #name ") returned NULL");
    LOAD_INSTANCE(GetPhysicalDeviceSurfaceSupportKHR)
    LOAD_INSTANCE(GetPhysicalDeviceSurfaceCapabilitiesKHR)
    LOAD_INSTANCE(GetPhysicalDeviceSurfaceFormatsKHR)
    LOAD_INSTANCE(CreateMetalSurfaceEXT)
    LOAD_INSTANCE(DestroySurfaceKHR)
    if (have_headless) {
        LOAD_INSTANCE(CreateHeadlessSurfaceEXT)
    }
#undef LOAD_INSTANCE
}

/* ---- The layer ---- */

struct rect_points {
    double x, y, width, height; /* CGRect */
};

/* A CAMetalLayer of width x height pixels (contentsScale 1), as a view of
 * that size would have; in no window. */
static id layer_create(uint32_t width, uint32_t height)
{
    id cls = (id)objc_getClass("CAMetalLayer"), layer;
    struct rect_points bounds = { 0.0, 0.0, (double)width, (double)height };
    if (!cls)
        bench_fatal("no CAMetalLayer class");
    layer = ((id (*)(id, SEL))objc_msgSend)(cls, sel_registerName("alloc"));
    layer = ((id (*)(id, SEL))objc_msgSend)(layer, sel_registerName("init"));
    if (!layer)
        bench_fatal("could not create a CAMetalLayer");
    ((void (*)(id, SEL, struct rect_points))objc_msgSend)(layer, sel_registerName("setBounds:"), bounds);
    ((void (*)(id, SEL, double))objc_msgSend)(layer, sel_registerName("setContentsScale:"), 1.0);
    return layer;
}

static void layer_release(id layer)
{
    ((void (*)(id, SEL))objc_msgSend)(layer, sel_registerName("release"));
}

/* ---- Frames ---- */

#define SLOTS 2   /* frames in flight */
#define QUERIES 3 /* per slot: frame start, frame end, after the present */

struct scene {
    struct bench_vk *vk;
    const struct config *cfg;
    /* for every configuration */
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkPipeline scene_pipeline, tonemap_pipeline;
    VkSampler sampler;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet set;
    VkQueryPool queries;
    VkCommandPool pools[SLOTS];
    VkCommandBuffer frame_cmds[SLOTS], marker_cmds[SLOTS];
    VkFence frame_fences[SLOTS], marker_fences[SLOTS];
    VkSemaphore acquired[SLOTS];
    /* per configuration */
    id layer;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    uint32_t image_count;
    VkImage images[8];
    VkImageView views[8];
    VkSemaphore rendered[8]; /* per image: signalled by the frame, waited for by its present */
    struct bench_image target; /* the HDR scene */
    struct bench_buffer instances, readback;
    struct bench_rect rects[RECTS];
    float colors[RECTS][4];
};

static void sleep_until(double t_us)
{
    double now = bench_now_us();
    if (t_us > now) {
        struct timespec ts;
        double wait = t_us - now;
        ts.tv_sec = (time_t)(wait / 1e6);
        ts.tv_nsec = (long)((wait - (double)ts.tv_sec * 1e6) * 1000.0);
        nanosleep(&ts, NULL);
    }
}

static void record_frame(struct scene *s, VkCommandBuffer cmd, uint32_t slot, uint32_t index, int verify)
{
    const struct config *cfg = s->cfg;
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VkRenderingAttachmentInfo scene_color = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = s->target.view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue.color.float32 = { 0.0625f, 0.125f, 0.25f, 1.0f },
    };
    VkRenderingAttachmentInfo swap_color = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = s->views[index],
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, /* every pixel is written */
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    VkRenderingInfo rendering = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = { { 0, 0 }, { cfg->width, cfg->height } },
        .layerCount = 1,
        .colorAttachmentCount = 1,
    };
    struct bench_image swap_image;
    const float offset[2] = { 0.0f, 0.0f };
    VkDeviceSize zero = 0;

    /* bench_image_barrier needs only these */
    memset(&swap_image, 0, sizeof(swap_image));
    swap_image.image = s->images[index];
    swap_image.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    swap_image.width = cfg->width;
    swap_image.height = cfg->height;

    BENCH_VK(vkd.BeginCommandBuffer(cmd, &begin));
    vkd.CmdResetQueryPool(cmd, s->queries, QUERIES * slot, QUERIES);
    vkd.CmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, s->queries, QUERIES * slot);

    /* The scene, into the HDR target (last frame's tonemapping read it). */
    bench_image_barrier(cmd, &s->target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    rendering.pColorAttachments = &scene_color;
    vkd.CmdBeginRendering(cmd, &rendering);
    bench_cmd_viewport(cmd, cfg->width, cfg->height);
    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->scene_pipeline);
    vkd.CmdBindVertexBuffers(cmd, 0, 1, &s->instances.buffer, &zero);
    vkd.CmdPushConstants(cmd, s->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(offset), offset);
    vkd.CmdDraw(cmd, 6, RECTS, 0, 0);
    vkd.CmdEndRendering(cmd);
    bench_image_barrier(cmd, &s->target, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    /* Tonemapping into the swapchain image (the acquire's wait is at color output). */
    bench_image_barrier(cmd, &swap_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    rendering.pColorAttachments = &swap_color;
    vkd.CmdBeginRendering(cmd, &rendering);
    bench_cmd_viewport(cmd, cfg->width, cfg->height);
    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->tonemap_pipeline);
    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s->layout, 0, 1, &s->set, 0, NULL);
    vkd.CmdDraw(cmd, 3, 1, 0, 0);
    vkd.CmdEndRendering(cmd);
    if (verify)
        bench_cmd_copy_image_to_buffer(cmd, &swap_image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, s->readback.buffer);
    bench_image_barrier(cmd, &swap_image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_NONE, 0);
    vkd.CmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, s->queries, QUERIES * slot + 1);
    BENCH_VK(vkd.EndCommandBuffer(cmd));
}

/* Tonemapped scene as B8G8R8A8_UNORM, within 1 per channel. */
static int verify(struct scene *s)
{
    const struct config *cfg = s->cfg;
    const uint8_t *got = s->readback.map;
    int w = (int)cfg->width, h = (int)cfg->height, x, y, c, i;
    int *top = bench_alloc(sizeof(int) * (size_t)w * (size_t)h), ok = 1;
    const float clear[3] = { 0.0625f, 0.125f, 0.25f };

    for (i = 0; i < w * h; i++)
        top[i] = -1;
    for (i = 0; i < RECTS; i++) /* later rectangles cover earlier ones */
        for (y = s->rects[i].y0; y < s->rects[i].y1; y++)
            for (x = s->rects[i].x0; x < s->rects[i].x1; x++)
                top[y * w + x] = i;
    for (y = 0; y < h && ok; y++)
        for (x = 0; x < w && ok; x++) {
            int r = top[y * w + x];
            for (c = 0; c < 3; c++) {
                float v = r < 0 ? clear[c] : s->colors[r][c];
                float want = v / (1.0f + v) * 255.0f;
                float g = (float)got[((size_t)y * w + x) * 4 + (size_t)(2 - c)]; /* BGRA */
                if (g - want > 1.0f || want - g > 1.0f) {
                    bench_error(cfg->name, "pixel (%d,%d) channel %d: %g, want %g", x, y, c, (double)g,
                                (double)want);
                    ok = 0;
                    break;
                }
            }
        }
    free(top);
    return ok;
}

/* ---- Per configuration ---- */

static void surface_destroy(struct scene *s)
{
    vkx.DestroySurfaceKHR(s->vk->instance, s->surface, NULL);
    s->surface = VK_NULL_HANDLE;
    if (s->layer)
        layer_release(s->layer);
    s->layer = NULL;
}

static int config_begin(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    const struct config *cfg = s->cfg;
    VkSurfaceCapabilitiesKHR caps;
    VkBool32 supported = VK_FALSE;
    VkSwapchainCreateInfoKHR info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .imageFormat = SWAPCHAIN_FORMAT,
        .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .imageExtent = { cfg->width, cfg->height },
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .presentMode = VK_PRESENT_MODE_FIFO_KHR,
        .clipped = VK_TRUE,
    };
    VkSemaphoreCreateInfo semaphore_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkSurfaceFormatKHR *formats;
    struct instance *inst;
    VkDescriptorImageInfo scene = { s->sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = s->set,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &scene,
    };
    uint64_t seed = 0x9E5E47ull + cfg->width;
    uint32_t count = 0, i;
    int c;

    if (cfg->layer) {
        VkMetalSurfaceCreateInfoEXT surface_info = { .sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT };
        s->layer = layer_create(cfg->width, cfg->height);
        surface_info.pLayer = (const CAMetalLayer *)s->layer;
        BENCH_VK(vkx.CreateMetalSurfaceEXT(vk->instance, &surface_info, NULL, &s->surface));
    } else {
        VkHeadlessSurfaceCreateInfoEXT surface_info = { .sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT };
        if (!have_headless) {
            bench_skip(cfg->name, "VK_EXT_headless_surface not supported");
            return 0;
        }
        BENCH_VK(vkx.CreateHeadlessSurfaceEXT(vk->instance, &surface_info, NULL, &s->surface));
    }
    BENCH_VK(vkx.GetPhysicalDeviceSurfaceSupportKHR(vk->physical_device, vk->queue_family, s->surface, &supported));
    if (!supported)
        bench_fatal("the queue cannot present to the surface");
    BENCH_VK(vkx.GetPhysicalDeviceSurfaceCapabilitiesKHR(vk->physical_device, s->surface, &caps));
    if ((caps.supportedUsageFlags & info.imageUsage) != info.imageUsage)
        bench_fatal("swapchain images cannot be color attachments and transfer sources");
    if (caps.currentExtent.width != 0xFFFFFFFFu &&
        (caps.currentExtent.width != cfg->width || caps.currentExtent.height != cfg->height))
        bench_fatal("the surface is %ux%u, not %ux%u", caps.currentExtent.width, caps.currentExtent.height,
                    cfg->width, cfg->height);
    if (cfg->width < caps.minImageExtent.width || cfg->width > caps.maxImageExtent.width ||
        cfg->height < caps.minImageExtent.height || cfg->height > caps.maxImageExtent.height) {
        bench_skip(cfg->name, "swapchains on the surface are %ux%u to %ux%u", caps.minImageExtent.width,
                   caps.minImageExtent.height, caps.maxImageExtent.width, caps.maxImageExtent.height);
        surface_destroy(s);
        return 0;
    }
    BENCH_VK(vkx.GetPhysicalDeviceSurfaceFormatsKHR(vk->physical_device, s->surface, &count, NULL));
    formats = bench_alloc(sizeof(*formats) * (count + 1));
    BENCH_VK(vkx.GetPhysicalDeviceSurfaceFormatsKHR(vk->physical_device, s->surface, &count, formats));
    for (i = 0; i < count; i++)
        if (formats[i].format == info.imageFormat && formats[i].colorSpace == info.imageColorSpace)
            break;
    free(formats);
    if (i == count) {
        bench_skip(cfg->name, "the surface has no B8G8R8A8_UNORM swapchains in SRGB_NONLINEAR");
        surface_destroy(s);
        return 0;
    }
    info.surface = s->surface;
    info.minImageCount = caps.minImageCount > 3 ? caps.minImageCount : 3;
    if (caps.maxImageCount && info.minImageCount > caps.maxImageCount)
        info.minImageCount = caps.maxImageCount;
    info.preTransform = caps.currentTransform;
    info.compositeAlpha = caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                              ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                              : (VkCompositeAlphaFlagBitsKHR)(caps.supportedCompositeAlpha &
                                                               (~caps.supportedCompositeAlpha + 1));
    BENCH_VK(vkd.CreateSwapchainKHR(vk->device, &info, NULL, &s->swapchain));
    s->image_count = 8;
    BENCH_VK(vkd.GetSwapchainImagesKHR(vk->device, s->swapchain, &s->image_count, s->images));
    for (i = 0; i < s->image_count; i++) {
        VkImageViewCreateInfo view = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = s->images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = SWAPCHAIN_FORMAT,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        BENCH_VK(vkd.CreateImageView(vk->device, &view, NULL, &s->views[i]));
        BENCH_VK(vkd.CreateSemaphore(vk->device, &semaphore_info, NULL, &s->rendered[i]));
    }

    /* The scene at this resolution: HDR colors in [0, 4), exact in half floats. */
    inst = bench_alloc(sizeof(*inst) * RECTS);
    bench_image_create(vk, &s->target, SCENE_FORMAT, cfg->width, cfg->height, 1, VK_SAMPLE_COUNT_1_BIT,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 0);
    scene.imageView = s->target.view;
    vkd.UpdateDescriptorSets(vk->device, 1, &write, 0, NULL);
    bench_rect_scene(s->rects, RECTS, (int)cfg->width, (int)cfg->height, (int)cfg->width / 8, 0, seed);
    for (i = 0; i < RECTS; i++) {
        bench_rect_ndc(&s->rects[i], (int)cfg->width, (int)cfg->height, 0, 0, inst[i].rect);
        for (c = 0; c < 3; c++)
            s->colors[i][c] = inst[i].color[c] = (float)(bench_rand(&seed) % 1024) / 256.0f;
        s->colors[i][3] = inst[i].color[3] = 1.0f;
    }
    bench_buffer_create(vk, &s->instances, sizeof(*inst) * RECTS,
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->instances, 0, inst, sizeof(*inst) * RECTS);
    bench_buffer_create(vk, &s->readback, (VkDeviceSize)cfg->width * cfg->height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        BENCH_HOST, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    free(inst);
    return 1;
}

static void config_end(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    uint32_t i;
    vkd.DeviceWaitIdle(vk->device);
    for (i = 0; i < s->image_count; i++) {
        vkd.DestroyImageView(vk->device, s->views[i], NULL);
        vkd.DestroySemaphore(vk->device, s->rendered[i], NULL);
    }
    vkd.DestroySwapchainKHR(vk->device, s->swapchain, NULL);
    surface_destroy(s);
    bench_image_destroy(vk, &s->target);
    bench_buffer_destroy(vk, &s->instances);
    bench_buffer_destroy(vk, &s->readback);
}

/* What a configuration measured, per frame. */
struct samples {
    int n, n_gpu, n_present;
    double *acquire_us, *cpu_us, *present_us, *gpu_us, *present_gpu_us;
    double *marker_delay; /* CPU time from the frame's vkQueueSubmit to the marker's submission */
};

/* GPU times of measured frame `sample`, from its slot's timestamps. The
 * present's counts only when the GPU cannot have idled before it: the marker
 * was submitted within the frame's own GPU time. */
static void collect(struct scene *s, struct samples *m, uint32_t slot, int sample)
{
    struct bench_vk *vk = s->vk;
    uint64_t ts[QUERIES], mask = vk->timestamp_bits >= 64 ? ~0ull : (1ull << vk->timestamp_bits) - 1;
    double gpu;
    if (vkd.GetQueryPoolResults(vk->device, s->queries, QUERIES * slot, QUERIES, sizeof(ts), ts, sizeof(uint64_t),
                                VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        return;
    gpu = (double)((ts[1] - ts[0]) & mask) * vk->timestamp_period / 1000.0;
    m->gpu_us[m->n_gpu++] = gpu;
    if (m->marker_delay[sample] < gpu)
        m->present_gpu_us[m->n_present++] = (double)((ts[2] - ts[1]) & mask) * vk->timestamp_period / 1000.0;
}

/* Runs a configuration: one verified frame, warm-up, then frames measured. */
static void run(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    const struct config *cfg = s->cfg;
    int frames = bench_frames(cfg->frames), total, f;
    int busy[SLOTS] = { 0, 0 }, sample_of[SLOTS] = { -1, -1 }, stalls = 0;
    double last_start = 0.0;
    struct samples m;
    struct bench_json j;

    frames = frames < 1000 ? frames : 1000;
    total = 1 + WARMUP + frames;
    memset(&m, 0, sizeof(m));
    m.acquire_us = bench_alloc(sizeof(double) * (size_t)frames);
    m.cpu_us = bench_alloc(sizeof(double) * (size_t)frames);
    m.present_us = bench_alloc(sizeof(double) * (size_t)frames);
    m.gpu_us = bench_alloc(sizeof(double) * (size_t)frames);
    m.present_gpu_us = bench_alloc(sizeof(double) * (size_t)frames);
    m.marker_delay = bench_alloc(sizeof(double) * (size_t)frames);

    for (f = 0; f < total; f++) {
        uint32_t slot = (uint32_t)f % SLOTS, index = 0;
        int measured = f > WARMUP, verify_frame = f == 0;
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &s->acquired[slot],
            .pWaitDstStageMask = &wait_stage,
            .commandBufferCount = 1,
            .pCommandBuffers = &s->frame_cmds[slot],
            .signalSemaphoreCount = 1,
        };
        VkSubmitInfo marker = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &s->marker_cmds[slot],
        };
        VkPresentInfoKHR present = {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .swapchainCount = 1,
            .pSwapchains = &s->swapchain,
            .pImageIndices = &index,
        };
        VkCommandBufferBeginInfo begin = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VkResult r;
        double t0, t1, t2, t3, t4, t5;

        /* A vsynced game's pace, after the slot's previous frame. */
        if (f > 1)
            sleep_until(last_start + FRAME_INTERVAL_US);
        last_start = bench_now_us();
        if (busy[slot]) {
            VkFence fences[2] = { s->frame_fences[slot], s->marker_fences[slot] };
            bench_fence_wait(vk, fences[0]);
            bench_fence_wait(vk, fences[1]);
            BENCH_VK(vkd.ResetFences(vk->device, 2, fences));
            busy[slot] = 0;
            if (sample_of[slot] >= 0)
                collect(s, &m, slot, sample_of[slot]);
            sample_of[slot] = -1;
        }

        t0 = bench_now_us();
        r = vkd.AcquireNextImageKHR(vk->device, s->swapchain, UINT64_MAX, s->acquired[slot], VK_NULL_HANDLE, &index);
        if (r < 0)
            bench_vk_fail(r, "vkAcquireNextImageKHR");
        t1 = bench_now_us();
        BENCH_VK(vkd.ResetCommandPool(vk->device, s->pools[slot], 0));
        record_frame(s, s->frame_cmds[slot], slot, index, verify_frame);
        submit.pSignalSemaphores = &s->rendered[index];
        t2 = bench_now_us();
        BENCH_VK(vkd.QueueSubmit(vk->queue, 1, &submit, s->frame_fences[slot]));
        t3 = bench_now_us();
        present.pWaitSemaphores = &s->rendered[index];
        r = vkd.QueuePresentKHR(vk->queue, &present);
        if (r < 0)
            bench_vk_fail(r, "vkQueuePresentKHR");
        t4 = bench_now_us();
        BENCH_VK(vkd.BeginCommandBuffer(s->marker_cmds[slot], &begin));
        vkd.CmdWriteTimestamp2(s->marker_cmds[slot], VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, s->queries,
                               QUERIES * slot + 2);
        BENCH_VK(vkd.EndCommandBuffer(s->marker_cmds[slot]));
        BENCH_VK(vkd.QueueSubmit(vk->queue, 1, &marker, s->marker_fences[slot]));
        t5 = bench_now_us();
        busy[slot] = 1;
        if ((t1 - t0) + (t4 - t3) > STALL_US && ++stalls == STALLS) {
            bench_skip(cfg->name, "acquire and present took over %g ms on %d frames: presentation stalled "
                       "(display asleep?)", STALL_US / 1000.0, STALLS);
            break;
        }

        if (verify_frame) {
            bench_fence_wait(vk, s->frame_fences[slot]);
            if (!verify(s))
                break;
        }
        if (measured) {
            m.acquire_us[m.n] = t1 - t0;
            m.cpu_us[m.n] = t3 - t1;
            m.present_us[m.n] = t4 - t3;
            m.marker_delay[m.n] = t5 - t2;
            sample_of[slot] = m.n++;
        }
    }
    vkd.DeviceWaitIdle(vk->device);
    for (f = 0; f < SLOTS; f++)
        if (sample_of[f] >= 0)
            collect(s, &m, (uint32_t)f, sample_of[f]);
    /* Unsignalled for the next configuration. */
    BENCH_VK(vkd.ResetFences(vk->device, SLOTS, s->frame_fences));
    BENCH_VK(vkd.ResetFences(vk->device, SLOTS, s->marker_fences));

    if (m.n == frames && stalls < STALLS) {
        bench_json_begin(&j, cfg->name);
        bench_json_str(&j, "surface", cfg->layer ? "metal_layer" : "headless");
        bench_json_int(&j, "width", cfg->width);
        bench_json_int(&j, "height", cfg->height);
        bench_json_int(&j, "images", s->image_count);
        bench_json_int(&j, "frames", m.n);
        bench_json_int(&j, "present_frames", m.n_present);
        bench_json_stats(&j, "acquire_us", m.acquire_us, m.n);
        bench_json_stats(&j, "cpu_us", m.cpu_us, m.n);
        bench_json_stats(&j, "present_us", m.present_us, m.n);
        bench_json_stats(&j, "gpu_us", m.gpu_us, m.n_gpu);
        bench_json_stats(&j, "present_gpu_us", m.present_gpu_us, m.n_present);
        bench_json_end(&j);
    }
    free(m.acquire_us);
    free(m.cpu_us);
    free(m.present_us);
    free(m.gpu_us);
    free(m.present_gpu_us);
    free(m.marker_delay);
}

/* ---- Setup ---- */

static void setup(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDescriptorSetLayoutBinding binding = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                             VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
    VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    VkDescriptorSetAllocateInfo set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorSetCount = 1,
    };
    VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    };
    VkQueryPoolCreateInfo query_info = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = QUERIES * SLOTS,
    };
    VkCommandPoolCreateInfo cmd_pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = vk->queue_family,
    };
    VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkSemaphoreCreateInfo semaphore_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    struct bench_gfx_desc desc;
    VkShaderModule rect_vs, flat_fs, fullscreen_vs, tonemap_fs;
    uint32_t i;

    s->set_layout = bench_set_layout(vk, 1, &binding, 0);
    s->layout = bench_pipeline_layout(vk, 1, &s->set_layout, sizeof(float) * 2, VK_SHADER_STAGE_VERTEX_BIT);
    BENCH_VK(vkd.CreateSampler(vk->device, &sampler_info, NULL, &s->sampler));
    BENCH_VK(vkd.CreateDescriptorPool(vk->device, &pool_info, NULL, &s->descriptor_pool));
    set_info.descriptorPool = s->descriptor_pool;
    set_info.pSetLayouts = &s->set_layout;
    BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &set_info, &s->set));
    BENCH_VK(vkd.CreateQueryPool(vk->device, &query_info, NULL, &s->queries));
    for (i = 0; i < SLOTS; i++) {
        VkCommandBufferAllocateInfo alloc = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        BENCH_VK(vkd.CreateCommandPool(vk->device, &cmd_pool_info, NULL, &s->pools[i]));
        alloc.commandPool = s->pools[i];
        BENCH_VK(vkd.AllocateCommandBuffers(vk->device, &alloc, &s->frame_cmds[i]));
        BENCH_VK(vkd.AllocateCommandBuffers(vk->device, &alloc, &s->marker_cmds[i]));
        BENCH_VK(vkd.CreateFence(vk->device, &fence_info, NULL, &s->frame_fences[i]));
        BENCH_VK(vkd.CreateFence(vk->device, &fence_info, NULL, &s->marker_fences[i]));
        BENCH_VK(vkd.CreateSemaphore(vk->device, &semaphore_info, NULL, &s->acquired[i]));
    }

    rect_vs = BENCH_SHADER(vk, rect_instanced_vert);
    flat_fs = BENCH_SHADER(vk, rect_flat_frag);
    fullscreen_vs = BENCH_SHADER(vk, fullscreen_vert);
    tonemap_fs = BENCH_SHADER(vk, tonemap_frag);
    memset(&desc, 0, sizeof(desc));
    desc.vs = rect_vs;
    desc.fs = flat_fs;
    desc.layout = s->layout;
    desc.color_count = 1;
    desc.color_formats[0] = SCENE_FORMAT;
    desc.binding_count = 1;
    desc.bindings[0] = (VkVertexInputBindingDescription){ 0, sizeof(struct instance), VK_VERTEX_INPUT_RATE_INSTANCE };
    desc.attribute_count = 2;
    desc.attributes[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
    desc.attributes[1] = (VkVertexInputAttributeDescription){ 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT,
                                                              (uint32_t)offsetof(struct instance, color) };
    s->scene_pipeline = bench_gfx_pipeline(vk, &desc);
    memset(&desc, 0, sizeof(desc));
    desc.vs = fullscreen_vs;
    desc.fs = tonemap_fs;
    desc.layout = s->layout;
    desc.color_count = 1;
    desc.color_formats[0] = SWAPCHAIN_FORMAT;
    s->tonemap_pipeline = bench_gfx_pipeline(vk, &desc);
    vkd.DestroyShaderModule(vk->device, rect_vs, NULL);
    vkd.DestroyShaderModule(vk->device, flat_fs, NULL);
    vkd.DestroyShaderModule(vk->device, fullscreen_vs, NULL);
    vkd.DestroyShaderModule(vk->device, tonemap_fs, NULL);
}

static void teardown(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    uint32_t i;
    vkd.DeviceWaitIdle(vk->device);
    for (i = 0; i < SLOTS; i++) {
        vkd.DestroyFence(vk->device, s->frame_fences[i], NULL);
        vkd.DestroyFence(vk->device, s->marker_fences[i], NULL);
        vkd.DestroySemaphore(vk->device, s->acquired[i], NULL);
        vkd.DestroyCommandPool(vk->device, s->pools[i], NULL);
    }
    vkd.DestroyQueryPool(vk->device, s->queries, NULL);
    vkd.DestroyPipeline(vk->device, s->scene_pipeline, NULL);
    vkd.DestroyPipeline(vk->device, s->tonemap_pipeline, NULL);
    vkd.DestroyPipelineLayout(vk->device, s->layout, NULL);
    vkd.DestroyDescriptorPool(vk->device, s->descriptor_pool, NULL);
    vkd.DestroyDescriptorSetLayout(vk->device, s->set_layout, NULL);
    vkd.DestroySampler(vk->device, s->sampler, NULL);
}

int main(void)
{
    struct bench_vk vk;
    struct scene *s;
    size_t c;

    bench_begin("present");
    init(&vk);
    s = bench_alloc(sizeof(*s));
    s->vk = &vk;
    setup(s);
    for (c = 0; c < sizeof(configs) / sizeof(configs[0]); c++) {
        if (!bench_selected(configs[c].name))
            continue;
        bench_config = configs[c].name;
        s->cfg = &configs[c];
        if (!config_begin(s))
            continue;
        run(s);
        config_end(s);
    }
    bench_config = NULL;
    teardown(s);
    free(s);
    bench_vk_destroy(&vk);
    return bench_exit_code();
}
