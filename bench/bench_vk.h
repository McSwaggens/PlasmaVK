/* Shared Vulkan setup for the benchmarks: instance and device creation,
 * memory, images, pipelines, one-shot commands, readback, and the frame loop
 * that measures CPU and GPU time per frame.
 *
 * Device-level commands are called through `vkd`, a table filled from
 * vkGetDeviceProcAddr, the way engines (and volk) do it. That measures the
 * driver itself in both build modes rather than the loader's trampolines.
 *
 * Failure policy: any Vulkan call that fails prints a "skipped" line naming
 * the call and exits 0 (BENCH_VK); a GPU hang (fence timeout) or device loss
 * prints an "error" line and exits 1. Configurations that run but produce
 * wrong output report bench_error() and keep going.
 */
#ifndef BENCH_VK_H
#define BENCH_VK_H

#include "pvk_bench.h"

#define BENCH_FRAMES_IN_FLIGHT 2
#define BENCH_TIMEOUT_NS 10000000000ull /* 10 s: anything longer is a hang */
#define BENCH_HOST (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
#define BENCH_DEVICE_LOCAL VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT

/* The configuration being run, for skip and error lines (NULL: setup). */
static const char *bench_config;

/* ==========================================================================
 * Failure handling
 * ========================================================================== */

static inline const char *bench_vk_result_name(VkResult r)
{
    switch (r) {
#define BENCH_RESULT(name) case name: return #name;
    BENCH_RESULT(VK_SUCCESS)
    BENCH_RESULT(VK_NOT_READY)
    BENCH_RESULT(VK_TIMEOUT)
    BENCH_RESULT(VK_INCOMPLETE)
    BENCH_RESULT(VK_ERROR_OUT_OF_HOST_MEMORY)
    BENCH_RESULT(VK_ERROR_OUT_OF_DEVICE_MEMORY)
    BENCH_RESULT(VK_ERROR_INITIALIZATION_FAILED)
    BENCH_RESULT(VK_ERROR_DEVICE_LOST)
    BENCH_RESULT(VK_ERROR_MEMORY_MAP_FAILED)
    BENCH_RESULT(VK_ERROR_LAYER_NOT_PRESENT)
    BENCH_RESULT(VK_ERROR_EXTENSION_NOT_PRESENT)
    BENCH_RESULT(VK_ERROR_FEATURE_NOT_PRESENT)
    BENCH_RESULT(VK_ERROR_INCOMPATIBLE_DRIVER)
    BENCH_RESULT(VK_ERROR_TOO_MANY_OBJECTS)
    BENCH_RESULT(VK_ERROR_FORMAT_NOT_SUPPORTED)
    BENCH_RESULT(VK_ERROR_FRAGMENTED_POOL)
    BENCH_RESULT(VK_ERROR_UNKNOWN)
    BENCH_RESULT(VK_ERROR_OUT_OF_POOL_MEMORY)
    BENCH_RESULT(VK_ERROR_INVALID_SHADER_NV)
    BENCH_RESULT(VK_PIPELINE_COMPILE_REQUIRED)
#undef BENCH_RESULT
    default: return "VkResult";
    }
}

/* Skips the rest of the benchmark: prints the reason and exits 0. */
static inline void bench_fatal(const char *fmt, ...) __attribute__((format(printf, 1, 2), noreturn));
static inline void bench_fatal(const char *fmt, ...)
{
    char reason[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(reason, sizeof(reason), fmt, args);
    va_end(args);
    bench_skip(bench_config, "%s", reason);
    exit(0);
}

/* "vkd.CreateBuffer(device, ...)" -> "vkCreateBuffer" */
static inline void bench_vk_fail(VkResult r, const char *call) __attribute__((noreturn));
static inline void bench_vk_fail(VkResult r, const char *call)
{
    char name[96];
    size_t n = 0;
    if (strncmp(call, "vkd.", 4) == 0) {
        memcpy(name, "vk", 2);
        n = 2;
        call += 4;
    }
    while (*call && *call != '(' && n + 1 < sizeof(name))
        name[n++] = *call++;
    name[n] = '\0';
    if (r == VK_ERROR_DEVICE_LOST) {
        bench_error(bench_config, "%s: VK_ERROR_DEVICE_LOST", name);
        exit(1);
    }
    bench_fatal("%s: %s (%d)", name, bench_vk_result_name(r), (int)r);
}

#define BENCH_VK(call)                              \
    do {                                            \
        VkResult bench_vk_r_ = (call);              \
        if (bench_vk_r_ != VK_SUCCESS)              \
            bench_vk_fail(bench_vk_r_, #call);      \
    } while (0)

/* malloc that never returns NULL. */
static inline void *bench_alloc(size_t size)
{
    void *p = calloc(1, size ? size : 1);
    if (!p) {
        fprintf(stderr, "%s: out of memory (%zu bytes)\n", bench_name, size);
        exit(1);
    }
    return p;
}

/* ==========================================================================
 * Device commands
 * ========================================================================== */

/* Vulkan 1.3 core: every device the benchmarks run on has them. */
#define BENCH_DEVICE_COMMANDS(X)                                                    \
    X(DestroyDevice) X(GetDeviceQueue) X(DeviceWaitIdle) X(QueueSubmit)             \
    X(QueueSubmit2) X(QueueWaitIdle)                                                \
    X(CreateCommandPool) X(DestroyCommandPool) X(ResetCommandPool)                  \
    X(AllocateCommandBuffers) X(FreeCommandBuffers) X(BeginCommandBuffer)           \
    X(EndCommandBuffer) X(ResetCommandBuffer) X(CreateFence) X(DestroyFence)        \
    X(WaitForFences) X(ResetFences) X(CreateSemaphore) X(DestroySemaphore)          \
    X(WaitSemaphores) X(SignalSemaphore) X(GetSemaphoreCounterValue)                \
    X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements)                 \
    X(GetBufferDeviceAddress) X(CreateBufferView) X(DestroyBufferView)              \
    X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) X(MapMemory)                \
    X(UnmapMemory) X(FlushMappedMemoryRanges)                                       \
    X(CreateImage) X(DestroyImage) X(GetImageMemoryRequirements) X(BindImageMemory) \
    X(CreateImageView) X(DestroyImageView) X(CreateSampler) X(DestroySampler)       \
    X(CreateShaderModule) X(DestroyShaderModule) X(CreatePipelineLayout)            \
    X(DestroyPipelineLayout) X(CreateGraphicsPipelines) X(CreateComputePipelines)   \
    X(DestroyPipeline) X(CreatePipelineCache) X(DestroyPipelineCache)               \
    X(GetPipelineCacheData) X(CreateDescriptorSetLayout)                            \
    X(DestroyDescriptorSetLayout) X(CreateDescriptorPool) X(DestroyDescriptorPool)  \
    X(ResetDescriptorPool) X(AllocateDescriptorSets) X(FreeDescriptorSets)          \
    X(UpdateDescriptorSets) X(CreateDescriptorUpdateTemplate)                       \
    X(DestroyDescriptorUpdateTemplate) X(UpdateDescriptorSetWithTemplate)           \
    X(CreateQueryPool) X(DestroyQueryPool) X(GetQueryPoolResults) X(ResetQueryPool) \
    X(CreateRenderPass2) X(DestroyRenderPass) X(CreateFramebuffer)                  \
    X(DestroyFramebuffer) X(CmdBeginRenderPass2) X(CmdNextSubpass2)                 \
    X(CmdEndRenderPass2) X(CmdBeginRendering) X(CmdEndRendering)                    \
    X(CmdExecuteCommands) X(CmdBindPipeline) X(CmdSetViewport) X(CmdSetScissor)     \
    X(CmdSetViewportWithCount) X(CmdSetScissorWithCount) X(CmdSetDepthBias)         \
    X(CmdSetDepthBounds) X(CmdSetBlendConstants) X(CmdSetStencilReference)          \
    X(CmdSetCullMode) X(CmdSetFrontFace) X(CmdSetPrimitiveTopology)                 \
    X(CmdSetDepthTestEnable) X(CmdSetDepthWriteEnable) X(CmdSetDepthCompareOp)      \
    X(CmdSetDepthBoundsTestEnable) X(CmdSetStencilTestEnable) X(CmdSetStencilOp)    \
    X(CmdSetDepthBiasEnable) X(CmdSetRasterizerDiscardEnable)                       \
    X(CmdSetPrimitiveRestartEnable) X(CmdPushConstants) X(CmdBindDescriptorSets)    \
    X(CmdBindVertexBuffers) X(CmdBindVertexBuffers2) X(CmdBindIndexBuffer)          \
    X(CmdDraw) X(CmdDrawIndexed) X(CmdDrawIndirect) X(CmdDrawIndexedIndirect)       \
    X(CmdDispatch) X(CmdDispatchIndirect) X(CmdPipelineBarrier2)                    \
    X(CmdCopyBuffer) X(CmdCopyImage) X(CmdBlitImage) X(CmdResolveImage)             \
    X(CmdCopyBufferToImage) X(CmdCopyImageToBuffer) X(CmdFillBuffer)                \
    X(CmdUpdateBuffer) X(CmdClearColorImage) X(CmdClearDepthStencilImage)           \
    X(CmdClearAttachments) X(CmdResetQueryPool) X(CmdBeginQuery) X(CmdEndQuery)     \
    X(CmdCopyQueryPoolResults) X(CmdWriteTimestamp2)

/* NULL when the device lacks them; benchmarks check before use. Vulkan 1.4,
 * the indirect-count draws (some 1.3 portability drivers leave them out) and
 * VK_KHR_swapchain (when enabled, see struct bench_vk_wants). */
#define BENCH_DEVICE_COMMANDS_OPTIONAL(X)                                           \
    X(CmdPushDescriptorSet) X(CmdPushDescriptorSetWithTemplate)                     \
    X(CmdSetRenderingAttachmentLocations) X(CmdSetRenderingInputAttachmentIndices)  \
    X(CmdDrawIndirectCount) X(CmdDrawIndexedIndirectCount)                          \
    X(CreateSwapchainKHR) X(DestroySwapchainKHR) X(GetSwapchainImagesKHR)           \
    X(AcquireNextImageKHR) X(QueuePresentKHR)

static struct bench_device_commands {
#define BENCH_DECLARE(name) PFN_vk##name name;
    BENCH_DEVICE_COMMANDS(BENCH_DECLARE)
    BENCH_DEVICE_COMMANDS_OPTIONAL(BENCH_DECLARE)
#undef BENCH_DECLARE
} vkd;

/* ==========================================================================
 * Instance and device
 * ========================================================================== */

enum {
    BENCH_NEED_DYNAMIC_RENDERING = 1 << 0,
    BENCH_NEED_SYNC2 = 1 << 1,
    BENCH_NEED_MAINTENANCE4 = 1 << 2,    /* SPIR-V LocalSizeId (local_size_x_id) */
    BENCH_WANT_PUSH_DESCRIPTOR = 1 << 3, /* optional: see bench_vk.push_descriptor */
    BENCH_WANT_LOCAL_READ = 1 << 4       /* optional: see bench_vk.local_read */
};

/* The core features of Vulkan 1.0 to 1.4. */
struct bench_features {
    VkPhysicalDeviceFeatures2 core;
    VkPhysicalDeviceVulkan11Features v11;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    VkPhysicalDeviceVulkan14Features v14;
};

/* Sets the sTypes and chains core -> v11 -> v12 -> v13 (-> v14 on a 1.4
 * device) for vkGetPhysicalDeviceFeatures2 or VkDeviceCreateInfo. The
 * struct then points into itself: it must not be copied while chained. */
static inline void bench_features_chain(struct bench_features *f, uint32_t minor)
{
    f->core.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f->v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f->v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f->v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f->v14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    f->core.pNext = &f->v11;
    f->v11.pNext = &f->v12;
    f->v12.pNext = &f->v13;
    f->v13.pNext = minor >= 4 ? &f->v14 : NULL;
    f->v14.pNext = NULL;
}

static inline void bench_features_unchain(struct bench_features *f)
{
    f->core.pNext = f->v11.pNext = f->v12.pNext = f->v13.pNext = f->v14.pNext = NULL;
}

/* Every VkBool32 of the five structs, as runs: first member and count. */
#define BENCH_FEATURE_RUNS 5
static inline void bench_features_runs(struct bench_features *f, VkBool32 *first[BENCH_FEATURE_RUNS],
                                       size_t count[BENCH_FEATURE_RUNS])
{
#define BENCH_RUN(i, member, type, from, to)                                                     \
    first[i] = &f->member.from;                                                                  \
    count[i] = (offsetof(type, to) - offsetof(type, from)) / sizeof(VkBool32) + 1;
    BENCH_RUN(0, core.features, VkPhysicalDeviceFeatures, robustBufferAccess, inheritedQueries)
    BENCH_RUN(1, v11, VkPhysicalDeviceVulkan11Features, storageBuffer16BitAccess, shaderDrawParameters)
    BENCH_RUN(2, v12, VkPhysicalDeviceVulkan12Features, samplerMirrorClampToEdge, subgroupBroadcastDynamicId)
    BENCH_RUN(3, v13, VkPhysicalDeviceVulkan13Features, robustImageAccess, maintenance4)
    BENCH_RUN(4, v14, VkPhysicalDeviceVulkan14Features, globalPriorityQuery, pushDescriptor)
#undef BENCH_RUN
}

/* dst &= mask, feature by feature. */
static inline void bench_features_and(struct bench_features *dst, struct bench_features *mask)
{
    VkBool32 *d[BENCH_FEATURE_RUNS], *m[BENCH_FEATURE_RUNS];
    size_t n[BENCH_FEATURE_RUNS], i, k;
    bench_features_runs(dst, d, n);
    bench_features_runs(mask, m, n);
    for (i = 0; i < BENCH_FEATURE_RUNS; i++)
        for (k = 0; k < n[i]; k++)
            d[i][k] = d[i][k] && m[i][k] ? VK_TRUE : VK_FALSE;
}

/* What a benchmark can use if the implementation has it (bench_vk_init_with).
 * Each wanted feature and extension it supports is enabled; check
 * bench_vk.enabled and bench_vk_extension before use. */
struct bench_vk_wants {
    struct bench_features features;     /* VK_TRUE: wanted */
    const char *instance_extensions[8]; /* unused entries NULL */
    const char *device_extensions[8];
};

#define BENCH_MAX_EXTENSIONS 17 /* 8 + 8 wanted, VK_KHR_portability_subset */

struct bench_vk {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    VkCommandPool pool;  /* one-shot commands */
    VkFence fence;       /* one-shot commands */
    uint32_t timestamp_bits; /* 0: no GPU timing */
    double timestamp_period; /* ns per tick */
    int portability_subset;
    int push_descriptor;  /* pushDescriptor enabled */
    int local_read;       /* dynamicRenderingLocalRead enabled */
    int local_read_depth; /* ... and depth can be read as an input attachment */
    int all_features;     /* BENCH_ALL_FEATURES: enabled is everything supported */
    struct bench_features supported, enabled; /* not chained */
    const char *extensions[BENCH_MAX_EXTENSIONS]; /* enabled instance and device extensions */
    uint32_t extension_count;
};

static inline int bench_has_extension(const VkExtensionProperties *exts, uint32_t count, const char *name)
{
    uint32_t i;
    for (i = 0; i < count; i++)
        if (strcmp(exts[i].extensionName, name) == 0)
            return 1;
    return 0;
}

/* Whether an instance or device extension was enabled. */
static inline int bench_vk_extension(const struct bench_vk *vk, const char *name)
{
    uint32_t i;
    for (i = 0; i < vk->extension_count; i++)
        if (strcmp(vk->extensions[i], name) == 0)
            return 1;
    return 0;
}

/* Adds the wanted extensions that avail has to names (and vk->extensions). */
static inline uint32_t bench_vk_pick_extensions(struct bench_vk *vk, const VkExtensionProperties *avail,
                                                uint32_t count, const char *const *wanted, size_t n_wanted,
                                                const char **names, uint32_t n)
{
    size_t i;
    for (i = 0; i < n_wanted; i++) {
        if (!wanted[i] || !bench_has_extension(avail, count, wanted[i]) ||
            vk->extension_count == BENCH_MAX_EXTENSIONS)
            continue;
        names[n++] = wanted[i];
        vk->extensions[vk->extension_count++] = wanted[i];
    }
    return n;
}

static inline void bench_vk_create_instance(struct bench_vk *vk, const struct bench_vk_wants *wants)
{
    const char *exts[9];
    VkExtensionProperties *avail;
    uint32_t count = 0, n_exts = 0;
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "plasmavk-bench",
        .pEngineName = "plasmavk-bench",
        .apiVersion = VK_API_VERSION_1_4,
    };
    VkInstanceCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
        .ppEnabledExtensionNames = exts,
    };

    if (vkEnumerateInstanceExtensionProperties(NULL, &count, NULL) < 0)
        count = 0;
    avail = bench_alloc(sizeof(*avail) * (count + 1));
    if (count && vkEnumerateInstanceExtensionProperties(NULL, &count, avail) < 0)
        count = 0;
    /* Lets the loader expose portability drivers such as MoltenVK. */
    if (bench_has_extension(avail, count, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        exts[n_exts++] = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
        info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    if (wants)
        n_exts = bench_vk_pick_extensions(vk, avail, count, wants->instance_extensions, 8, exts, n_exts);
    free(avail);
    info.enabledExtensionCount = n_exts;
    BENCH_VK(vkCreateInstance(&info, NULL, &vk->instance));
}

/* Creates everything a benchmark needs, or prints "skipped" and exits.
 * needs: BENCH_NEED_* (required) and BENCH_WANT_* (optional) flags; wants
 * (may be NULL): further optional features and extensions. With
 * BENCH_ALL_FEATURES set, every supported feature is enabled instead. */
static inline void bench_vk_init_with(struct bench_vk *vk, unsigned needs, const struct bench_vk_wants *wants)
{
    struct bench_features have, want;
    VkQueueFamilyProperties families[16];
    VkExtensionProperties *exts;
    const char *device_exts[9];
    uint32_t count = 0, i, n_device_exts = 0, api, minor;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &want.core,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .ppEnabledExtensionNames = device_exts,
    };

    memset(vk, 0, sizeof(*vk));
    bench_vk_create_instance(vk, wants);

    count = 1;
    if (vkEnumeratePhysicalDevices(vk->instance, &count, &vk->physical_device) < 0 || count == 0)
        bench_fatal("no Vulkan physical device");
    vkGetPhysicalDeviceProperties(vk->physical_device, &vk->props);
    vkGetPhysicalDeviceMemoryProperties(vk->physical_device, &vk->mem);
    bench_set_device(vk->props.deviceName);

    api = vk->props.apiVersion;
    if (VK_API_VERSION_MAJOR(api) == 1 && VK_API_VERSION_MINOR(api) < 3)
        bench_fatal("device reports Vulkan 1.%u; the benchmarks need 1.3", VK_API_VERSION_MINOR(api));
    minor = VK_API_VERSION_MINOR(api);
    memset(&have, 0, sizeof(have));
    bench_features_chain(&have, minor);
    vkGetPhysicalDeviceFeatures2(vk->physical_device, &have.core);
    bench_features_unchain(&have);

    {
        char missing[256] = "";
        if ((needs & BENCH_NEED_DYNAMIC_RENDERING) && !have.v13.dynamicRendering)
            strcat(missing, " dynamicRendering");
        if ((needs & BENCH_NEED_SYNC2) && !have.v13.synchronization2)
            strcat(missing, " synchronization2");
        if ((needs & BENCH_NEED_MAINTENANCE4) && !have.v13.maintenance4)
            strcat(missing, " maintenance4");
        if (missing[0])
            bench_fatal("Vulkan 1.3 features not supported:%s", missing);
    }
    /* Everything, or what was asked for and is there. */
    vk->all_features = bench_applied.all_features = bench_all_features();
    if (vk->all_features) {
        want = have;
    } else {
        if (wants)
            want = wants->features;
        else
            memset(&want, 0, sizeof(want));
        if (needs & BENCH_NEED_DYNAMIC_RENDERING)
            want.v13.dynamicRendering = VK_TRUE;
        if (needs & BENCH_NEED_SYNC2)
            want.v13.synchronization2 = VK_TRUE;
        if (needs & BENCH_NEED_MAINTENANCE4)
            want.v13.maintenance4 = VK_TRUE;
        if (needs & BENCH_WANT_PUSH_DESCRIPTOR)
            want.v14.pushDescriptor = VK_TRUE;
        if (needs & BENCH_WANT_LOCAL_READ)
            want.v14.dynamicRenderingLocalRead = VK_TRUE;
        bench_features_and(&want, &have);
    }
    if (minor < 4)
        memset(&want.v14, 0, sizeof(want.v14));
    vk->push_descriptor = (needs & BENCH_WANT_PUSH_DESCRIPTOR) && want.v14.pushDescriptor;
    vk->local_read = (needs & BENCH_WANT_LOCAL_READ) && want.v14.dynamicRenderingLocalRead;
    if (vk->local_read) {
        VkPhysicalDeviceVulkan14Properties props14 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES };
        VkPhysicalDeviceProperties2 props2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &props14 };
        vkGetPhysicalDeviceProperties2(vk->physical_device, &props2);
        vk->local_read_depth = props14.dynamicRenderingLocalReadDepthStencilAttachments != VK_FALSE;
    }
    vk->supported = have;
    vk->enabled = want;
    bench_features_chain(&want, minor);

    /* One queue that does everything. */
    count = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(vk->physical_device, &count, families);
    for (i = 0; i < count; i++)
        if ((families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
            break;
    if (i == count)
        bench_fatal("no queue family with graphics and compute");
    vk->queue_family = i;
    queue_info.queueFamilyIndex = i;
    vk->timestamp_bits = families[i].timestampValidBits;
    vk->timestamp_period = vk->props.limits.timestampPeriod;
    if (vk->timestamp_period <= 0.0)
        vk->timestamp_bits = 0;

    /* Portability implementations must have VK_KHR_portability_subset enabled. */
    count = 0;
    if (vkEnumerateDeviceExtensionProperties(vk->physical_device, NULL, &count, NULL) < 0)
        count = 0;
    exts = bench_alloc(sizeof(*exts) * (count + 1));
    if (count && vkEnumerateDeviceExtensionProperties(vk->physical_device, NULL, &count, exts) < 0)
        count = 0;
    if (bench_has_extension(exts, count, "VK_KHR_portability_subset")) {
        device_exts[n_device_exts++] = "VK_KHR_portability_subset";
        vk->extensions[vk->extension_count++] = "VK_KHR_portability_subset";
        vk->portability_subset = 1;
    }
    if (wants)
        n_device_exts = bench_vk_pick_extensions(vk, exts, count, wants->device_extensions, 8, device_exts,
                                                 n_device_exts);
    free(exts);
    device_info.enabledExtensionCount = n_device_exts;
    BENCH_VK(vkCreateDevice(vk->physical_device, &device_info, NULL, &vk->device));

#define BENCH_LOAD(name)                                                            \
    vkd.name = (PFN_vk##name)vkGetDeviceProcAddr(vk->device, "vk" #name);           \
    if (!vkd.name)                                                                  \
        bench_fatal("vkGetDeviceProcAddr(vk" #name ") returned NULL");
    BENCH_DEVICE_COMMANDS(BENCH_LOAD)
#undef BENCH_LOAD
#define BENCH_LOAD_OPTIONAL(name) vkd.name = (PFN_vk##name)vkGetDeviceProcAddr(vk->device, "vk" #name);
    BENCH_DEVICE_COMMANDS_OPTIONAL(BENCH_LOAD_OPTIONAL)
#undef BENCH_LOAD_OPTIONAL

    vkd.GetDeviceQueue(vk->device, vk->queue_family, 0, &vk->queue);
    if (!vk->queue)
        bench_fatal("vkGetDeviceQueue returned no queue");
    {
        VkCommandPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = vk->queue_family,
        };
        VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        BENCH_VK(vkd.CreateCommandPool(vk->device, &pool_info, NULL, &vk->pool));
        BENCH_VK(vkd.CreateFence(vk->device, &fence_info, NULL, &vk->fence));
    }
}

static inline void bench_vk_init(struct bench_vk *vk, unsigned needs)
{
    bench_vk_init_with(vk, needs, NULL);
}

static inline void bench_vk_destroy(struct bench_vk *vk)
{
    if (vk->device) {
        vkd.DeviceWaitIdle(vk->device);
        vkd.DestroyFence(vk->device, vk->fence, NULL);
        vkd.DestroyCommandPool(vk->device, vk->pool, NULL);
        vkd.DestroyDevice(vk->device, NULL);
    }
    if (vk->instance)
        vkDestroyInstance(vk->instance, NULL);
    memset(vk, 0, sizeof(*vk));
}

/* ==========================================================================
 * Memory, buffers and images
 * ========================================================================== */

/* A memory type with all of required, preferably also preferred, and none of
 * avoid. UINT32_MAX if there is none. */
static inline uint32_t bench_memory_type(const struct bench_vk *vk, uint32_t type_bits,
                                         VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred,
                                         VkMemoryPropertyFlags avoid)
{
    uint32_t pass, i;
    for (pass = 0; pass < 2; pass++) {
        VkMemoryPropertyFlags want = pass == 0 ? required | preferred : required;
        for (i = 0; i < vk->mem.memoryTypeCount; i++) {
            VkMemoryPropertyFlags flags = vk->mem.memoryTypes[i].propertyFlags;
            if ((type_bits & (1u << i)) && (flags & want) == want && !(flags & avoid))
                return i;
        }
    }
    return UINT32_MAX;
}

static inline VkDeviceMemory bench_allocate(struct bench_vk *vk, const VkMemoryRequirements *req,
                                            VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred,
                                            VkMemoryPropertyFlags avoid, VkMemoryPropertyFlags *flags)
{
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t type = bench_memory_type(vk, req->memoryTypeBits, required, preferred, avoid);
    VkMemoryAllocateInfo info = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req->size };

    if (req->size == 0 || type == UINT32_MAX)
        bench_fatal("no memory type for %llu bytes (memoryTypeBits 0x%x, flags 0x%x)",
                    (unsigned long long)req->size, req->memoryTypeBits, required);
    info.memoryTypeIndex = type;
    BENCH_VK(vkd.AllocateMemory(vk->device, &info, NULL, &memory));
    if (flags)
        *flags = vk->mem.memoryTypes[type].propertyFlags;
    return memory;
}

struct bench_buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
    VkMemoryPropertyFlags flags;
    void *map; /* persistently mapped when host visible */
};

static inline void bench_buffer_create(struct bench_vk *vk, struct bench_buffer *b, VkDeviceSize size,
                                       VkBufferUsageFlags usage, VkMemoryPropertyFlags required,
                                       VkMemoryPropertyFlags preferred)
{
    VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkMemoryRequirements req;

    memset(b, 0, sizeof(*b));
    memset(&req, 0, sizeof(req));
    BENCH_VK(vkd.CreateBuffer(vk->device, &info, NULL, &b->buffer));
    vkd.GetBufferMemoryRequirements(vk->device, b->buffer, &req);
    b->memory = bench_allocate(vk, &req, required, preferred, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, &b->flags);
    BENCH_VK(vkd.BindBufferMemory(vk->device, b->buffer, b->memory, 0));
    b->size = size;
    if (b->flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        BENCH_VK(vkd.MapMemory(vk->device, b->memory, 0, VK_WHOLE_SIZE, 0, &b->map));
        if (!b->map)
            bench_fatal("vkMapMemory returned NULL");
    }
}

static inline void bench_buffer_destroy(struct bench_vk *vk, struct bench_buffer *b)
{
    if (b->buffer)
        vkd.DestroyBuffer(vk->device, b->buffer, NULL);
    if (b->memory)
        vkd.FreeMemory(vk->device, b->memory, NULL); /* unmaps */
    memset(b, 0, sizeof(*b));
}

struct bench_image {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
    VkFormat format;
    VkImageAspectFlags aspect;
    uint32_t width, height, mips;
    VkSampleCountFlagBits samples;
    int lazy; /* backed by lazily allocated (memoryless) memory */
};

static inline VkImageAspectFlags bench_format_aspect(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

/* A 2D image with a view of all its mips. transient: an attachment that
 * never leaves tile memory (lazily allocated memory when available). */
static inline void bench_image_create(struct bench_vk *vk, struct bench_image *img, VkFormat format,
                                      uint32_t width, uint32_t height, uint32_t mips,
                                      VkSampleCountFlagBits samples, VkImageUsageFlags usage, int transient)
{
    VkImageCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = { width, height, 1 },
        .mipLevels = mips,
        .arrayLayers = 1,
        .samples = samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage | (transient ? VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT : 0),
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImageViewCreateInfo view = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
    };
    VkMemoryRequirements req;
    VkMemoryPropertyFlags flags = 0;

    memset(img, 0, sizeof(*img));
    memset(&req, 0, sizeof(req));
    img->format = format;
    img->aspect = bench_format_aspect(format);
    img->width = width;
    img->height = height;
    img->mips = mips;
    img->samples = samples;
    BENCH_VK(vkd.CreateImage(vk->device, &info, NULL, &img->image));
    vkd.GetImageMemoryRequirements(vk->device, img->image, &req);
    if (transient)
        img->memory = bench_allocate(vk, &req, 0, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, 0, &flags);
    else
        img->memory = bench_allocate(vk, &req, 0, BENCH_DEVICE_LOCAL, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, &flags);
    img->lazy = (flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0;
    BENCH_VK(vkd.BindImageMemory(vk->device, img->image, img->memory, 0));
    view.image = img->image;
    view.subresourceRange.aspectMask = img->aspect & ~(VkImageAspectFlags)VK_IMAGE_ASPECT_STENCIL_BIT;
    view.subresourceRange.levelCount = mips;
    view.subresourceRange.layerCount = 1;
    BENCH_VK(vkd.CreateImageView(vk->device, &view, NULL, &img->view));
}

static inline void bench_image_destroy(struct bench_vk *vk, struct bench_image *img)
{
    if (img->view)
        vkd.DestroyImageView(vk->device, img->view, NULL);
    if (img->image)
        vkd.DestroyImage(vk->device, img->image, NULL);
    if (img->memory)
        vkd.FreeMemory(vk->device, img->memory, NULL);
    memset(img, 0, sizeof(*img));
}

/* ==========================================================================
 * Barriers
 * ========================================================================== */

static inline void bench_memory_barrier(VkCommandBuffer cmd, VkPipelineStageFlags2 src_stage,
                                        VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                                        VkAccessFlags2 dst_access)
{
    VkMemoryBarrier2 barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
    };
    VkDependencyInfo dep = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    vkd.CmdPipelineBarrier2(cmd, &dep);
}

static inline VkImageMemoryBarrier2 bench_image_barrier_info(const struct bench_image *img, VkImageLayout old_layout,
                                                             VkImageLayout new_layout,
                                                             VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                                             VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access)
{
    VkImageMemoryBarrier2 b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img->image,
        .subresourceRange = { img->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, 1 },
    };
    return b;
}

static inline void bench_image_barriers(VkCommandBuffer cmd, uint32_t count, const VkImageMemoryBarrier2 *barriers)
{
    VkDependencyInfo dep = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = count,
        .pImageMemoryBarriers = barriers,
    };
    vkd.CmdPipelineBarrier2(cmd, &dep);
}

static inline void bench_image_barrier(VkCommandBuffer cmd, const struct bench_image *img, VkImageLayout old_layout,
                                       VkImageLayout new_layout, VkPipelineStageFlags2 src_stage,
                                       VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                                       VkAccessFlags2 dst_access)
{
    VkImageMemoryBarrier2 b = bench_image_barrier_info(img, old_layout, new_layout, src_stage, src_access,
                                                       dst_stage, dst_access);
    bench_image_barriers(cmd, 1, &b);
}

/* ==========================================================================
 * One-shot commands, uploads and readback
 * ========================================================================== */

static inline void bench_fence_wait(struct bench_vk *vk, VkFence fence)
{
    VkResult r = vkd.WaitForFences(vk->device, 1, &fence, VK_TRUE, BENCH_TIMEOUT_NS);
    if (r == VK_TIMEOUT) {
        bench_error(bench_config, "GPU work did not complete within %llu s (vkWaitForFences timed out)",
                    BENCH_TIMEOUT_NS / 1000000000ull);
        exit(1);
    }
    if (r != VK_SUCCESS)
        bench_vk_fail(r, "vkWaitForFences");
}

static inline VkCommandBuffer bench_cmd_begin(struct bench_vk *vk)
{
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk->pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    BENCH_VK(vkd.AllocateCommandBuffers(vk->device, &alloc, &cmd));
    BENCH_VK(vkd.BeginCommandBuffer(cmd, &begin));
    return cmd;
}

/* Ends, submits and waits for cmd, then frees it. */
static inline void bench_cmd_finish(struct bench_vk *vk, VkCommandBuffer cmd)
{
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    BENCH_VK(vkd.EndCommandBuffer(cmd));
    BENCH_VK(vkd.QueueSubmit(vk->queue, 1, &submit, vk->fence));
    bench_fence_wait(vk, vk->fence);
    BENCH_VK(vkd.ResetFences(vk->device, 1, &vk->fence));
    vkd.FreeCommandBuffers(vk->device, vk->pool, 1, &cmd);
}

/* Copies size bytes into dst at offset through a staging buffer, and makes
 * them visible to every later command. */
static inline void bench_buffer_upload(struct bench_vk *vk, struct bench_buffer *dst, VkDeviceSize offset,
                                       const void *data, VkDeviceSize size)
{
    struct bench_buffer staging;
    VkBufferCopy region = { 0, offset, size };
    VkCommandBuffer cmd;

    bench_buffer_create(vk, &staging, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, BENCH_HOST, 0);
    memcpy(staging.map, data, (size_t)size);
    cmd = bench_cmd_begin(vk);
    vkd.CmdCopyBuffer(cmd, staging.buffer, dst->buffer, 1, &region);
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    bench_cmd_finish(vk, cmd);
    bench_buffer_destroy(vk, &staging);
}

/* Fills mip 0 of a single-sample color image (tightly packed texels) and
 * leaves it in final_layout. */
static inline void bench_image_upload(struct bench_vk *vk, struct bench_image *img, const void *data,
                                      VkDeviceSize size, VkImageLayout final_layout)
{
    struct bench_buffer staging;
    VkBufferImageCopy region = {
        .imageSubresource = { img->aspect, 0, 0, 1 },
        .imageExtent = { img->width, img->height, 1 },
    };
    VkCommandBuffer cmd;

    bench_buffer_create(vk, &staging, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, BENCH_HOST, 0);
    memcpy(staging.map, data, (size_t)size);
    cmd = bench_cmd_begin(vk);
    bench_image_barrier(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkd.CmdCopyBufferToImage(cmd, staging.buffer, img->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    bench_image_barrier(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, final_layout,
                        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
    bench_cmd_finish(vk, cmd);
    bench_buffer_destroy(vk, &staging);
}

/* Records a copy of mip 0 of img (in layout, written by src_stage) into dst,
 * then returns img to layout. */
static inline void bench_cmd_copy_image_to_buffer(VkCommandBuffer cmd, const struct bench_image *img,
                                                  VkImageLayout layout, VkPipelineStageFlags2 src_stage,
                                                  VkAccessFlags2 src_access, VkBuffer dst)
{
    VkBufferImageCopy region = {
        .imageSubresource = { img->aspect, 0, 0, 1 },
        .imageExtent = { img->width, img->height, 1 },
    };
    bench_image_barrier(cmd, img, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, src_stage, src_access,
                        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    vkd.CmdCopyImageToBuffer(cmd, img->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, 1, &region);
    bench_image_barrier(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout, VK_PIPELINE_STAGE_2_COPY_BIT, 0,
                        src_stage, src_access);
    bench_memory_barrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
}

/* ==========================================================================
 * Shaders and pipelines
 * ========================================================================== */

static inline VkShaderModule bench_shader_module(struct bench_vk *vk, const uint32_t *code, size_t size)
{
    VkShaderModule module = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = code,
    };
    BENCH_VK(vkd.CreateShaderModule(vk->device, &info, NULL, &module));
    return module;
}
#define BENCH_SHADER(vk, array) bench_shader_module((vk), (array), sizeof(array))

static inline VkDescriptorSetLayout bench_set_layout(struct bench_vk *vk, uint32_t count,
                                                     const VkDescriptorSetLayoutBinding *bindings,
                                                     VkDescriptorSetLayoutCreateFlags flags)
{
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayoutCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = flags,
        .bindingCount = count,
        .pBindings = bindings,
    };
    BENCH_VK(vkd.CreateDescriptorSetLayout(vk->device, &info, NULL, &layout));
    return layout;
}

static inline VkPipelineLayout bench_pipeline_layout(struct bench_vk *vk, uint32_t set_count,
                                                     const VkDescriptorSetLayout *sets, uint32_t push_size,
                                                     VkShaderStageFlags push_stages)
{
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPushConstantRange range = { push_stages, 0, push_size };
    VkPipelineLayoutCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = set_count,
        .pSetLayouts = sets,
        .pushConstantRangeCount = push_size ? 1 : 0,
        .pPushConstantRanges = &range,
    };
    BENCH_VK(vkd.CreatePipelineLayout(vk->device, &info, NULL, &layout));
    return layout;
}

/* What varies between the graphics pipelines the benchmarks build. Viewport
 * and scissor are always dynamic, plus any in dynamic[]; rendering is always
 * dynamic rendering. */
struct bench_gfx_desc {
    VkShaderModule vs, fs;
    const VkSpecializationInfo *vs_spec, *fs_spec;
    VkPipelineLayout layout;
    uint32_t color_count;
    VkFormat color_formats[8];
    VkFormat depth_format;
    VkSampleCountFlagBits samples;     /* 0 means 1 */
    int depth_test, depth_write;
    VkCompareOp depth_compare;
    int blend;                         /* premultiplied-style alpha blending */
    VkCullModeFlags cull;
    VkFrontFace front_face;
    VkPrimitiveTopology topology;      /* 0 means triangle list */
    uint32_t binding_count, attribute_count;
    VkVertexInputBindingDescription bindings[4];
    VkVertexInputAttributeDescription attributes[8];
    uint32_t dynamic_count;
    VkDynamicState dynamic[16];        /* more dynamic states */
    const void *rendering_next;        /* chained after VkPipelineRenderingCreateInfo */
};

/* Create-info storage for one graphics pipeline. It points into itself, so
 * it must not be copied or moved after bench_gfx_state_init. */
struct bench_gfx_state {
    struct bench_gfx_desc desc;
    VkPipelineShaderStageCreateInfo stages[2];
    VkPipelineVertexInputStateCreateInfo vertex_input;
    VkPipelineInputAssemblyStateCreateInfo input_assembly;
    VkPipelineViewportStateCreateInfo viewport;
    VkPipelineRasterizationStateCreateInfo raster;
    VkPipelineMultisampleStateCreateInfo multisample;
    VkPipelineDepthStencilStateCreateInfo depth;
    VkPipelineColorBlendAttachmentState attachments[8];
    VkPipelineColorBlendStateCreateInfo blend;
    VkDynamicState dynamic_states[18];
    VkPipelineDynamicStateCreateInfo dynamic;
    VkPipelineRenderingCreateInfo rendering;
    VkGraphicsPipelineCreateInfo info;
};

static inline void bench_gfx_state_init(struct bench_gfx_state *s, const struct bench_gfx_desc *d)
{
    uint32_t i;

    memset(s, 0, sizeof(*s));
    s->desc = *d;
    d = &s->desc;

    s->stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    s->stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    s->stages[0].module = d->vs;
    s->stages[0].pName = "main";
    s->stages[0].pSpecializationInfo = d->vs_spec;
    s->stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    s->stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    s->stages[1].module = d->fs;
    s->stages[1].pName = "main";
    s->stages[1].pSpecializationInfo = d->fs_spec;

    s->vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    s->vertex_input.vertexBindingDescriptionCount = d->binding_count;
    s->vertex_input.pVertexBindingDescriptions = d->bindings;
    s->vertex_input.vertexAttributeDescriptionCount = d->attribute_count;
    s->vertex_input.pVertexAttributeDescriptions = d->attributes;

    s->input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    s->input_assembly.topology = d->topology ? d->topology : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    s->viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    s->viewport.viewportCount = 1;
    s->viewport.scissorCount = 1;

    s->raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    s->raster.polygonMode = VK_POLYGON_MODE_FILL;
    s->raster.cullMode = d->cull;
    s->raster.frontFace = d->front_face;
    s->raster.lineWidth = 1.0f;

    s->multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    s->multisample.rasterizationSamples = d->samples ? d->samples : VK_SAMPLE_COUNT_1_BIT;

    s->depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    s->depth.depthTestEnable = d->depth_test ? VK_TRUE : VK_FALSE;
    s->depth.depthWriteEnable = d->depth_write ? VK_TRUE : VK_FALSE;
    s->depth.depthCompareOp = d->depth_test ? d->depth_compare : VK_COMPARE_OP_ALWAYS;
    s->depth.maxDepthBounds = 1.0f;

    for (i = 0; i < d->color_count; i++) {
        VkPipelineColorBlendAttachmentState *a = &s->attachments[i];
        a->colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        if (d->blend) {
            a->blendEnable = VK_TRUE;
            a->srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            a->dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            a->colorBlendOp = VK_BLEND_OP_ADD;
            a->srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            a->dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            a->alphaBlendOp = VK_BLEND_OP_ADD;
        }
    }
    s->blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    s->blend.attachmentCount = d->color_count;
    s->blend.pAttachments = s->attachments;

    s->dynamic_states[0] = VK_DYNAMIC_STATE_VIEWPORT;
    s->dynamic_states[1] = VK_DYNAMIC_STATE_SCISSOR;
    for (i = 0; i < d->dynamic_count && i < 16; i++)
        s->dynamic_states[2 + i] = d->dynamic[i];
    s->dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    s->dynamic.dynamicStateCount = 2 + i;
    s->dynamic.pDynamicStates = s->dynamic_states;

    s->rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    s->rendering.pNext = d->rendering_next;
    s->rendering.colorAttachmentCount = d->color_count;
    s->rendering.pColorAttachmentFormats = d->color_formats;
    s->rendering.depthAttachmentFormat = d->depth_format;

    s->info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    s->info.pNext = &s->rendering;
    s->info.stageCount = 2;
    s->info.pStages = s->stages;
    s->info.pVertexInputState = &s->vertex_input;
    s->info.pInputAssemblyState = &s->input_assembly;
    s->info.pViewportState = &s->viewport;
    s->info.pRasterizationState = &s->raster;
    s->info.pMultisampleState = &s->multisample;
    s->info.pDepthStencilState = &s->depth;
    s->info.pColorBlendState = &s->blend;
    s->info.pDynamicState = &s->dynamic;
    s->info.layout = d->layout;
}

static inline VkPipeline bench_gfx_pipeline(struct bench_vk *vk, const struct bench_gfx_desc *d)
{
    struct bench_gfx_state *s = bench_alloc(sizeof(*s));
    VkPipeline pipeline = VK_NULL_HANDLE;
    bench_gfx_state_init(s, d);
    BENCH_VK(vkd.CreateGraphicsPipelines(vk->device, VK_NULL_HANDLE, 1, &s->info, NULL, &pipeline));
    free(s);
    return pipeline;
}

static inline VkComputePipelineCreateInfo bench_compute_info(VkShaderModule module, const VkSpecializationInfo *spec,
                                                             VkPipelineLayout layout)
{
    VkComputePipelineCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = module,
            .pName = "main",
            .pSpecializationInfo = spec,
        },
        .layout = layout,
    };
    return info;
}

static inline VkPipeline bench_compute_pipeline(struct bench_vk *vk, VkShaderModule module,
                                                const VkSpecializationInfo *spec, VkPipelineLayout layout)
{
    VkComputePipelineCreateInfo info = bench_compute_info(module, spec, layout);
    VkPipeline pipeline = VK_NULL_HANDLE;
    BENCH_VK(vkd.CreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &info, NULL, &pipeline));
    return pipeline;
}

static inline void bench_cmd_viewport(VkCommandBuffer cmd, uint32_t width, uint32_t height)
{
    VkViewport viewport = { 0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f };
    VkRect2D scissor = { { 0, 0 }, { width, height } };
    vkd.CmdSetViewport(cmd, 0, 1, &viewport);
    vkd.CmdSetScissor(cmd, 0, 1, &scissor);
}

/* Starts dynamic rendering into a single-sample color target, clearing it
 * (its previous contents are discarded, as with a freshly acquired image). */
static inline void bench_cmd_begin_color(VkCommandBuffer cmd, const struct bench_image *target, const float clear[4])
{
    VkRenderingAttachmentInfo color = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = target->view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    VkRenderingInfo info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = { { 0, 0 }, { target->width, target->height } },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color,
    };
    memcpy(color.clearValue.color.float32, clear, sizeof(float) * 4);
    bench_image_barrier(cmd, target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    vkd.CmdBeginRendering(cmd, &info);
    bench_cmd_viewport(cmd, target->width, target->height);
}

/* ==========================================================================
 * Frame loop
 *
 * Frames are recorded into per-slot command pools with BENCH_FRAMES_IN_FLIGHT
 * frames in flight, like a renderer. Per measured frame:
 *   cpu_us     reset + record + end + submit (not the wait for a free slot)
 *   submit_us  vkQueueSubmit alone
 *   gpu_us     timestamps at the start and end of the frame (none with
 *              BENCH_GPU_TIMING=0)
 *   frame_us   time between consecutive submits: throughput, CPU or GPU bound
 *
 * A frame is one primary command buffer, which the loop begins and ends
 * around the benchmark's record function. After bench_loop_primaries(loop,
 * n) it is n primaries that one vkQueueSubmit submits in order, the way
 * D3D12-style renderers submit command lists recorded on several threads.
 * Each has its own command pool, and the record function begins and ends
 * each one (bench_frame_begin, bench_frame_end) on the thread that records
 * it, all before it returns.
 * ========================================================================== */

#define BENCH_MAX_PRIMARIES 8

struct bench_frame {
    VkCommandBuffer cmd; /* the frame's command buffer (several: cmds[0]) */
    uint32_t index;      /* frame number, from 0 */
    uint32_t slot;       /* frame-in-flight slot, for per-slot resources */
    int verify;          /* also record the readback the benchmark checks */
    uint32_t primaries;  /* 1, or as set by bench_loop_primaries */
    VkCommandBuffer cmds[BENCH_MAX_PRIMARIES]; /* submitted in this order */
    VkCommandPool pools[BENCH_MAX_PRIMARIES];  /* cmds[i]'s, reset by bench_frame_begin */
    VkDevice device;
    VkQueryPool queries; /* the loop's timestamps, or VK_NULL_HANDLE */
};

typedef void (*bench_record_fn)(void *user, const struct bench_frame *frame);

/* Starts recording primary i of the frame; the first one also starts the
 * frame's GPU timing. Any thread, one per command buffer. */
static inline void bench_frame_begin(const struct bench_frame *f, uint32_t i)
{
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    BENCH_VK(vkd.ResetCommandPool(f->device, f->pools[i], 0));
    BENCH_VK(vkd.BeginCommandBuffer(f->cmds[i], &begin));
    if (i == 0 && f->queries) {
        vkd.CmdResetQueryPool(f->cmds[0], f->queries, 2 * f->slot, 2);
        vkd.CmdWriteTimestamp2(f->cmds[0], VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, f->queries, 2 * f->slot);
    }
}

/* Ends primary i; the last one first ends the frame's GPU timing. */
static inline void bench_frame_end(const struct bench_frame *f, uint32_t i)
{
    if (i + 1 == f->primaries && f->queries)
        vkd.CmdWriteTimestamp2(f->cmds[i], VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f->queries, 2 * f->slot + 1);
    BENCH_VK(vkd.EndCommandBuffer(f->cmds[i]));
}

struct bench_loop {
    struct bench_vk *vk;
    VkCommandPool pools[BENCH_FRAMES_IN_FLIGHT][BENCH_MAX_PRIMARIES];
    VkCommandBuffer cmds[BENCH_FRAMES_IN_FLIGHT][BENCH_MAX_PRIMARIES];
    uint32_t primaries, allocated; /* per frame, and per slot so far */
    VkFence fences[BENCH_FRAMES_IN_FLIGHT];
    int busy[BENCH_FRAMES_IN_FLIGHT];
    int sample_of[BENCH_FRAMES_IN_FLIGHT]; /* measured frame in the slot, or -1 */
    VkQueryPool queries;
    uint32_t next_index;
    int capacity, count, gpu_count;
    double last_submit;
    double *cpu, *submit, *gpu, *frame;
};

/* Primary i of every slot, each in its own pool. */
static inline void bench_loop_allocate(struct bench_loop *loop, uint32_t i)
{
    struct bench_vk *vk = loop->vk;
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = vk->queue_family,
    };
    uint32_t slot;
    for (slot = 0; slot < BENCH_FRAMES_IN_FLIGHT; slot++) {
        VkCommandBufferAllocateInfo alloc = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        BENCH_VK(vkd.CreateCommandPool(vk->device, &pool_info, NULL, &loop->pools[slot][i]));
        alloc.commandPool = loop->pools[slot][i];
        BENCH_VK(vkd.AllocateCommandBuffers(vk->device, &alloc, &loop->cmds[slot][i]));
    }
}

static inline void bench_loop_init(struct bench_loop *loop, struct bench_vk *vk, int max_frames)
{
    VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    uint32_t i;

    memset(loop, 0, sizeof(*loop));
    loop->vk = vk;
    loop->capacity = max_frames;
    loop->cpu = bench_alloc(sizeof(double) * (size_t)max_frames);
    loop->submit = bench_alloc(sizeof(double) * (size_t)max_frames);
    loop->gpu = bench_alloc(sizeof(double) * (size_t)max_frames);
    loop->frame = bench_alloc(sizeof(double) * (size_t)max_frames);
    bench_loop_allocate(loop, 0);
    loop->primaries = loop->allocated = 1;
    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++) {
        BENCH_VK(vkd.CreateFence(vk->device, &fence_info, NULL, &loop->fences[i]));
        loop->sample_of[i] = -1;
    }
    /* GPU timing is optional: without timestamps the line has no gpu_us. */
    bench_applied.no_gpu_timing = !bench_gpu_timing();
    if (vk->timestamp_bits && !bench_applied.no_gpu_timing) {
        VkQueryPoolCreateInfo query_info = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2 * BENCH_FRAMES_IN_FLIGHT,
        };
        if (vkd.CreateQueryPool(vk->device, &query_info, NULL, &loop->queries) != VK_SUCCESS)
            loop->queries = VK_NULL_HANDLE;
    }
}

/* Waits for the slot's previous frame and collects its GPU time. */
static inline void bench_loop_retire(struct bench_loop *loop, uint32_t slot)
{
    struct bench_vk *vk = loop->vk;
    uint64_t ts[2] = { 0, 0 };

    if (!loop->busy[slot])
        return;
    bench_fence_wait(vk, loop->fences[slot]);
    loop->busy[slot] = 0;
    if (loop->sample_of[slot] >= 0 && loop->queries &&
        vkd.GetQueryPoolResults(vk->device, loop->queries, 2 * slot, 2, sizeof(ts), ts, sizeof(uint64_t),
                                VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        uint64_t mask = vk->timestamp_bits >= 64 ? ~0ull : (1ull << vk->timestamp_bits) - 1;
        uint64_t ticks = (ts[1] - ts[0]) & mask;
        loop->gpu[loop->gpu_count++] = (double)ticks * vk->timestamp_period / 1000.0;
    }
    loop->sample_of[slot] = -1;
}

static inline void bench_loop_wait_all(struct bench_loop *loop)
{
    uint32_t i;
    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++)
        bench_loop_retire(loop, i);
}

/* Frames from now on are n primary command buffers (1 to
 * BENCH_MAX_PRIMARIES), recorded as the section comment says. */
static inline void bench_loop_primaries(struct bench_loop *loop, uint32_t n)
{
    if (n < 1 || n > BENCH_MAX_PRIMARIES)
        bench_fatal("bench_loop_primaries(%u): 1 to %d", n, BENCH_MAX_PRIMARIES);
    bench_loop_wait_all(loop);
    for (; loop->allocated < n; loop->allocated++)
        bench_loop_allocate(loop, loop->allocated);
    loop->primaries = n;
}

/* Records and submits one frame. measured: store its timings. */
static inline void bench_loop_frame(struct bench_loop *loop, bench_record_fn record, void *user, int verify,
                                    int measured)
{
    struct bench_vk *vk = loop->vk;
    uint32_t slot = loop->next_index % BENCH_FRAMES_IN_FLIGHT, i;
    VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
    struct bench_frame frame;
    double t0, t1, t2;

    bench_loop_retire(loop, slot);
    t0 = bench_now_us();
    frame.index = loop->next_index++;
    frame.slot = slot;
    frame.verify = verify;
    frame.primaries = loop->primaries;
    for (i = 0; i < frame.primaries; i++) {
        frame.cmds[i] = loop->cmds[slot][i];
        frame.pools[i] = loop->pools[slot][i];
    }
    frame.cmd = frame.cmds[0];
    frame.device = vk->device;
    frame.queries = loop->queries;
    BENCH_VK(vkd.ResetFences(vk->device, 1, &loop->fences[slot]));
    if (frame.primaries == 1) {
        bench_frame_begin(&frame, 0);
        record(user, &frame);
        bench_frame_end(&frame, 0);
    } else {
        record(user, &frame); /* begins and ends every primary */
    }
    submit.commandBufferCount = frame.primaries;
    submit.pCommandBuffers = frame.cmds;
    t1 = bench_now_us();
    BENCH_VK(vkd.QueueSubmit(vk->queue, 1, &submit, loop->fences[slot]));
    t2 = bench_now_us();
    loop->busy[slot] = 1;

    if (measured && loop->count < loop->capacity) {
        loop->cpu[loop->count] = t2 - t0;
        loop->submit[loop->count] = t2 - t1;
        loop->frame[loop->count] = t2 - loop->last_submit;
        loop->sample_of[slot] = loop->count++;
    }
    loop->last_submit = t2;
}

/* Runs one frame with verify set and waits for it. */
static inline void bench_loop_verify(struct bench_loop *loop, bench_record_fn record, void *user)
{
    bench_loop_frame(loop, record, user, 1, 0);
    bench_loop_wait_all(loop);
}

/* warmup unmeasured frames, then frames measured ones (results kept until
 * the next run). */
static inline void bench_loop_run(struct bench_loop *loop, bench_record_fn record, void *user, int warmup,
                                  int frames)
{
    int i;
    loop->count = loop->gpu_count = 0;
    for (i = 0; i < warmup; i++)
        bench_loop_frame(loop, record, user, 0, 0);
    for (i = 0; i < frames; i++)
        bench_loop_frame(loop, record, user, 0, 1);
    bench_loop_wait_all(loop);
}

/* Appends "frames" and the timing statistics of the last run. */
static inline void bench_loop_report(struct bench_loop *loop, struct bench_json *j)
{
    bench_json_int(j, "frames", loop->count);
    bench_json_stats(j, "cpu_us", loop->cpu, loop->count);
    bench_json_stats(j, "submit_us", loop->submit, loop->count);
    bench_json_stats(j, "gpu_us", loop->gpu, loop->gpu_count);
    bench_json_stats(j, "frame_us", loop->frame, loop->count);
}

/* Medians of the last run in microseconds (GPU: 0 without timestamps).
 * Call before bench_loop_report, which sorts the samples. */
static inline double bench_loop_cpu_median(const struct bench_loop *loop)
{
    return bench_median(loop->cpu, loop->count);
}

static inline double bench_loop_gpu_median(const struct bench_loop *loop)
{
    return bench_median(loop->gpu, loop->gpu_count);
}

static inline void bench_loop_destroy(struct bench_loop *loop)
{
    struct bench_vk *vk = loop->vk;
    uint32_t i, k;
    bench_loop_wait_all(loop);
    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++) {
        vkd.DestroyFence(vk->device, loop->fences[i], NULL);
        for (k = 0; k < loop->allocated; k++)
            vkd.DestroyCommandPool(vk->device, loop->pools[i][k], NULL);
    }
    if (loop->queries)
        vkd.DestroyQueryPool(vk->device, loop->queries, NULL);
    free(loop->cpu);
    free(loop->submit);
    free(loop->gpu);
    free(loop->frame);
    memset(loop, 0, sizeof(*loop));
}

/* ==========================================================================
 * CPU references
 * ========================================================================== */

/* VK_FORMAT_R8G8B8A8_UNORM texel as stored in memory. */
static inline uint32_t bench_rgba8(uint32_t r, uint32_t g, uint32_t b, uint32_t a)
{
    return r | g << 8 | b << 16 | a << 24;
}

/* A pixel-aligned rectangle [x0,x1) x [y0,y1) with an RGBA8 color. Edges on
 * pixel boundaries never touch a sample position, so every implementation
 * must rasterize them identically (with or without MSAA). */
struct bench_rect {
    int x0, y0, x1, y1;
    uint32_t rgba;
};

/* count deterministic rectangles of 2..max_size-1 pixels per side, at least
 * margin pixels from the edges of a width x height target. */
static inline void bench_rect_scene(struct bench_rect *rects, int count, int width, int height, int max_size,
                                    int margin, uint64_t seed)
{
    int i;
    for (i = 0; i < count; i++) {
        struct bench_rect *r = &rects[i];
        int w = bench_rand_range(&seed, 2, max_size), h = bench_rand_range(&seed, 2, max_size);
        r->x0 = bench_rand_range(&seed, margin, width - margin - w);
        r->y0 = bench_rand_range(&seed, margin, height - margin - h);
        r->x1 = r->x0 + w;
        r->y1 = r->y0 + h;
        r->rgba = bench_rand(&seed) | 0xFF000000u; /* opaque */
    }
}

/* Pixel coordinates to NDC; exact in float for power-of-two sizes. */
static inline void bench_rect_ndc(const struct bench_rect *r, int width, int height, int dx, int dy, float out[4])
{
    out[0] = (float)(r->x0 + dx) * 2.0f / (float)width - 1.0f;
    out[1] = (float)(r->y0 + dy) * 2.0f / (float)height - 1.0f;
    out[2] = (float)(r->x1 + dx) * 2.0f / (float)width - 1.0f;
    out[3] = (float)(r->y1 + dy) * 2.0f / (float)height - 1.0f;
}

/* RGBA8 texel as four normalized floats (each c / 255). */
static inline void bench_unorm4(uint32_t rgba, float out[4])
{
    int i;
    for (i = 0; i < 4; i++)
        out[i] = (float)((rgba >> (8 * i)) & 0xFF) / 255.0f;
}

/* Fills [x0,x1) x [y0,y1) of a width-wide RGBA8 image. */
static inline void bench_fill_rect(uint32_t *pixels, int width, int x0, int y0, int x1, int y1, uint32_t rgba)
{
    int x, y;
    for (y = y0; y < y1; y++)
        for (x = x0; x < x1; x++)
            pixels[(size_t)y * (size_t)width + (size_t)x] = rgba;
}

static inline int bench_rgba8_close(uint32_t a, uint32_t b, int tolerance)
{
    int i;
    for (i = 0; i < 32; i += 8) {
        int d = (int)((a >> i) & 0xFF) - (int)((b >> i) & 0xFF);
        if (d > tolerance || d < -tolerance)
            return 0;
    }
    return 1;
}

/* Compares RGBA8 images, allowing each channel to be off by tolerance; on
 * mismatch reports the first bad texel and returns 0. */
static inline int bench_check_rgba8_tolerance(const char *config, const uint32_t *got, const uint32_t *want,
                                              int width, int height, int tolerance)
{
    size_t i, n = (size_t)width * (size_t)height, bad = 0, first = 0;
    for (i = 0; i < n; i++)
        if (!bench_rgba8_close(got[i], want[i], tolerance) && bad++ == 0)
            first = i;
    if (!bad)
        return 1;
    bench_error(config, "%zu of %zu pixels wrong; first at (%d,%d): got 0x%08x, want 0x%08x", bad, n,
                (int)(first % (size_t)width), (int)(first / (size_t)width), got[first], want[first]);
    return 0;
}

/* Exact comparison. */
static inline int bench_check_rgba8(const char *config, const uint32_t *got, const uint32_t *want, int width,
                                    int height)
{
    return bench_check_rgba8_tolerance(config, got, want, width, height, 0);
}

#endif /* BENCH_VK_H */
