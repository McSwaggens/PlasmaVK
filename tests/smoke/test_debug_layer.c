/* The debug layer's checks (section 9, DEBUG LAYER CHECKS): each fires with
 * its VUID on a deliberate error, valid usage reports nothing, and a check
 * whose command would crash skips it. The layer is enabled here through
 * PLASMAVK_DEBUG=1 (set before the first Vulkan call), as for an
 * application that cannot be changed. */
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <stdlib.h>
#include <string.h>

#include "../pvk_test.h"

/* ---- What the messenger saw ---- */

#define SEEN_MAX 64

static struct {
    int count;
    char vuids[SEEN_MAX][128];
    uint64_t objects[SEEN_MAX];
} seen;
static VkBool32 skip_requested; /* the callback's return value */

static VkBool32 VKAPI_PTR record(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT types,
                                 const VkDebugUtilsMessengerCallbackDataEXT *data, void *user)
{
    (void)severity;
    (void)user;
    if (!(types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT))
        return VK_FALSE;
    if (seen.count < SEEN_MAX) {
        snprintf(seen.vuids[seen.count], sizeof(seen.vuids[0]), "%s", data->pMessageIdName);
        seen.objects[seen.count] = data->objectCount ? data->pObjects[0].objectHandle : 0;
    }
    seen.count++;
    return skip_requested;
}

/* The VUID was reported (and only it, since the last expect). */
#define EXPECT(vuid) expect_(vuid, __LINE__)
static void expect_(const char *vuid, int line)
{
    int i, found = 0;
    for (i = 0; i < seen.count && i < SEEN_MAX; i++)
        found |= strcmp(seen.vuids[i], vuid) == 0;
    if (!found || seen.count != 1) {
        fprintf(stderr, "test_debug_layer.c:%d: expected %s alone, saw %d:", line, vuid, seen.count);
        for (i = 0; i < seen.count && i < SEEN_MAX; i++)
            fprintf(stderr, " %s", seen.vuids[i]);
        fputc('\n', stderr);
        pvk_test_failures++;
    }
    seen.count = 0;
}

#define EXPECT_NONE() EXPECT_COUNT(0)
#define EXPECT_COUNT(n)                                                                                          \
    do {                                                                                                         \
        if (seen.count != (n)) {                                                                                 \
            PVK_TEST_FAIL("expected %d validation messages, saw %d (first: %s)", (n), seen.count,                \
                          seen.count ? seen.vuids[0] : "-");                                                     \
        }                                                                                                        \
        seen.count = 0;                                                                                          \
    } while (0)

/* ---- Setup: an instance (the layer implicit), a messenger, two devices ---- */

static VkInstance instance;
static VkDebugUtilsMessengerEXT messenger;
static VkPhysicalDevice pd;

static void make_instance(void)
{
    const char *ext = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    VkApplicationInfo app;
    VkInstanceCreateInfo info;
    VkDebugUtilsMessengerCreateInfoEXT mi;
    uint32_t count = 1;

    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_4;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &app;
    info.enabledExtensionCount = 1;
    info.ppEnabledExtensionNames = &ext;
    if (vkCreateInstance(&info, NULL, &instance) != VK_SUCCESS)
        abort();
    memset(&mi, 0, sizeof(mi));
    mi.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
    mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    mi.pfnUserCallback = record;
    if (vkCreateDebugUtilsMessengerEXT(instance, &mi, NULL, &messenger) != VK_SUCCESS)
        abort();
    vkEnumeratePhysicalDevices(instance, &count, &pd);
}

/* A device with a queue of each family and the features given (NULL: none). */
static VkDevice make_device(const void *features2, const VkAllocationCallbacks *alloc)
{
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queues[2];
    VkDeviceCreateInfo info;
    VkDevice device = VK_NULL_HANDLE;
    memset(queues, 0, sizeof(queues));
    queues[0].sType = queues[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queues[0].queueCount = queues[1].queueCount = 1;
    queues[0].pQueuePriorities = queues[1].pQueuePriorities = &priority;
    queues[1].queueFamilyIndex = 1;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.pNext = features2;
    info.queueCreateInfoCount = 2;
    info.pQueueCreateInfos = queues;
    if (vkCreateDevice(pd, &info, alloc, &device) != VK_SUCCESS)
        abort();
    return device;
}

static uint32_t memory_type(VkMemoryPropertyFlags flags, VkMemoryPropertyFlags excluded)
{
    VkPhysicalDeviceMemoryProperties mp;
    uint32_t i;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (i = 0; i < mp.memoryTypeCount; i++)
        if ((mp.memoryTypes[i].propertyFlags & flags) == flags && !(mp.memoryTypes[i].propertyFlags & excluded) &&
            !(mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT))
            return i;
    return 0;
}

static VkBuffer make_buffer(VkDevice device, const VkAllocationCallbacks *alloc)
{
    VkBufferCreateInfo bi;
    VkBuffer buffer = VK_NULL_HANDLE;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = 4096;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
               VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    vkCreateBuffer(device, &bi, alloc, &buffer);
    return buffer;
}

static VkDeviceMemory make_memory(VkDevice device, VkMemoryPropertyFlags flags, VkMemoryPropertyFlags excluded)
{
    VkMemoryAllocateInfo mai;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    memset(&mai, 0, sizeof(mai));
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = 65536;
    mai.memoryTypeIndex = memory_type(flags, excluded);
    vkAllocateMemory(device, &mai, NULL, &memory);
    return memory;
}

static VkImage make_image(VkDevice device, const VkAllocationCallbacks *alloc)
{
    VkImageCreateInfo ii;
    VkImage image = VK_NULL_HANDLE;
    memset(&ii, 0, sizeof(ii));
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent.width = ii.extent.height = 16;
    ii.extent.depth = 1;
    ii.mipLevels = ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    vkCreateImage(device, &ii, alloc, &image);
    return image;
}

static VkCommandPool make_pool(VkDevice device, uint32_t family, VkCommandPoolCreateFlags flags)
{
    VkCommandPoolCreateInfo info;
    VkCommandPool pool = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.flags = flags;
    info.queueFamilyIndex = family;
    vkCreateCommandPool(device, &info, NULL, &pool);
    return pool;
}

static VkCommandBuffer make_cb(VkDevice device, VkCommandPool pool, VkCommandBufferLevel level)
{
    VkCommandBufferAllocateInfo info;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = pool;
    info.level = level;
    info.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &info, &cb);
    return cb;
}

static VkResult begin(VkCommandBuffer cb, VkCommandBufferUsageFlags flags)
{
    VkCommandBufferInheritanceInfo inherit;
    VkCommandBufferBeginInfo info;
    memset(&inherit, 0, sizeof(inherit));
    inherit.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags = flags;
    info.pInheritanceInfo = &inherit;
    return vkBeginCommandBuffer(cb, &info);
}

static VkResult submit(VkQueue queue, VkCommandBuffer cb, VkFence fence)
{
    VkSubmitInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.commandBufferCount = cb ? 1 : 0;
    info.pCommandBuffers = &cb;
    return vkQueueSubmit(queue, 1, &info, fence);
}

static VkFence make_fence(VkDevice device, bool signaled)
{
    VkFenceCreateInfo info;
    VkFence fence = VK_NULL_HANDLE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    info.flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0;
    vkCreateFence(device, &info, NULL, &fence);
    return fence;
}

static VkSemaphore make_semaphore(VkDevice device, bool timeline)
{
    VkSemaphoreTypeCreateInfo type;
    VkSemaphoreCreateInfo info;
    VkSemaphore sem = VK_NULL_HANDLE;
    memset(&type, 0, sizeof(type));
    type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    info.pNext = timeline ? &type : NULL;
    vkCreateSemaphore(device, &info, NULL, &sem);
    return sem;
}

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

static VkPipeline make_pipeline(VkDevice device, VkPipelineLayout *layout, const VkAllocationCallbacks *alloc)
{
    VkShaderModuleCreateInfo smi;
    VkShaderModule module;
    VkPipelineLayoutCreateInfo pli;
    VkComputePipelineCreateInfo ci;
    VkPipeline pipeline = VK_NULL_HANDLE;
    memset(&smi, 0, sizeof(smi));
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = sizeof(spv_empty);
    smi.pCode = spv_empty;
    vkCreateShaderModule(device, &smi, NULL, &module);
    memset(&pli, 0, sizeof(pli));
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    vkCreatePipelineLayout(device, &pli, NULL, layout);
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = *layout;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, alloc, &pipeline);
    vkDestroyShaderModule(device, module, NULL);
    return pipeline;
}

/* Allocation callbacks over malloc, counting live blocks. */
static int live_blocks;

static void *VKAPI_PTR a_alloc(void *user, size_t size, size_t align, VkSystemAllocationScope scope)
{
    void *p = NULL;
    (void)user;
    (void)scope;
    if (posix_memalign(&p, align < sizeof(void *) ? sizeof(void *) : align, size) != 0)
        return NULL;
    live_blocks++;
    return p;
}

static void *VKAPI_PTR a_realloc(void *user, void *old, size_t size, size_t align, VkSystemAllocationScope scope)
{
    (void)user, (void)old, (void)size, (void)align, (void)scope;
    return NULL;
}

static void VKAPI_PTR a_free(void *user, void *p)
{
    (void)user;
    if (p) {
        live_blocks--;
        free(p);
    }
}

static const VkAllocationCallbacks callbacks = { NULL, a_alloc, a_realloc, a_free, NULL, NULL };

#define H(x) ((uint64_t)(uintptr_t)(x))

/* A device with timelineSemaphore enabled (the tests that wait on one). */
static VkDevice make_timeline_device(void)
{
    VkPhysicalDeviceVulkan12Features v12;
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.timelineSemaphore = VK_TRUE;
    return make_device(&v12, NULL);
}

/* ---- Tests ---- */

static void test_layer_from_environment(void)
{
    VkLayerProperties layer;
    uint32_t n = 1;
    CHECK(((struct pvk_instance *)instance)->debug);
    CHECK_VK(vkEnumerateDeviceLayerProperties(pd, &n, &layer), VK_SUCCESS);
    CHECK(n == 1 && strcmp(layer.layerName, "VK_LAYER_PLASMAVK_debug") == 0);
}

/* A valid workload under the layer reports nothing. */
static void test_valid_usage_is_quiet(void)
{
    VkDevice device = make_timeline_device();
    VkQueue queue;
    VkBuffer buffer = make_buffer(device, NULL);
    VkDeviceMemory memory = make_memory(device, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
    VkImage image = make_image(device, &callbacks);
    VkDeviceMemory image_memory = make_memory(device, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VkCommandPool pool = make_pool(device, 0, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    VkCommandBuffer cb = make_cb(device, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    VkFence fence = make_fence(device, false);
    VkSemaphore sem = make_semaphore(device, true);
    VkPipelineLayout layout;
    VkPipeline pipeline = make_pipeline(device, &layout, NULL);
    VkDebugUtilsLabelEXT label;
    VkSemaphoreSubmitInfo signal;
    VkCommandBufferSubmitInfo cbi;
    VkSubmitInfo2 s2;
    void *data;
    int k;

    vkGetDeviceQueue(device, 0, 0, &queue);
    CHECK_VK(vkBindBufferMemory(device, buffer, memory, 0), VK_SUCCESS);
    CHECK_VK(vkBindImageMemory(device, image, image_memory, 0), VK_SUCCESS);
    CHECK_VK(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &data), VK_SUCCESS);
    vkUnmapMemory(device, memory);
    memset(&label, 0, sizeof(label));
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = "valid";
    for (k = 0; k < 2; k++) { /* recorded twice: the implicit reset of a resettable pool's command buffer */
        CHECK_VK(begin(cb, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT), VK_SUCCESS);
        vkCmdBeginDebugUtilsLabelEXT(cb, &label);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdDispatch(cb, 1, 1, 1);
        vkCmdFillBuffer(cb, buffer, 0, 256, 7);
        vkCmdEndDebugUtilsLabelEXT(cb);
        CHECK_VK(vkEndCommandBuffer(cb), VK_SUCCESS);
        CHECK_VK(submit(queue, cb, fence), VK_SUCCESS);
        CHECK_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), VK_SUCCESS);
        CHECK_VK(vkResetFences(device, 1, &fence), VK_SUCCESS);
    }
    CHECK_VK(begin(cb, 0), VK_SUCCESS);
    CHECK_VK(vkEndCommandBuffer(cb), VK_SUCCESS);
    memset(&cbi, 0, sizeof(cbi));
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cbi.commandBuffer = cb;
    memset(&signal, 0, sizeof(signal));
    signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signal.semaphore = sem;
    signal.value = 1;
    memset(&s2, 0, sizeof(s2));
    s2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    s2.commandBufferInfoCount = 1;
    s2.pCommandBufferInfos = &cbi;
    s2.signalSemaphoreInfoCount = 1;
    s2.pSignalSemaphoreInfos = &signal;
    CHECK_VK(vkQueueSubmit2(queue, 1, &s2, fence), VK_SUCCESS);
    CHECK_VK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), VK_SUCCESS);
    vkQueueBeginDebugUtilsLabelEXT(queue, &label);
    vkQueueEndDebugUtilsLabelEXT(queue);

    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroySemaphore(device, sem, NULL);
    vkDestroyFence(device, fence, NULL);
    vkFreeCommandBuffers(device, pool, 1, &cb);
    vkDestroyCommandPool(device, pool, NULL);
    vkDestroyImage(device, image, &callbacks);
    vkFreeMemory(device, image_memory, NULL);
    vkDestroyBuffer(device, buffer, NULL);
    vkFreeMemory(device, memory, NULL);
    vkDestroyDevice(device, NULL);
    EXPECT_NONE();
    CHECK(live_blocks == 0);
}

/* Destroyed handles: destroying again or using one is reported and skipped
 * (with ASan, touching the freed object would be an error). */
static void test_destroyed_handles(void)
{
    VkDevice device = make_device(NULL, NULL), other = make_device(NULL, NULL);
    VkQueue queue;
    VkBuffer buffer = make_buffer(device, NULL), foreign = make_buffer(other, NULL);
    VkDeviceMemory memory = make_memory(device, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
    VkImage image = make_image(device, NULL);
    VkSamplerCreateInfo si;
    VkSampler sampler;
    VkFence fence = make_fence(device, false);
    VkSemaphore sem = make_semaphore(device, false);
    VkPipelineLayout layout;
    VkPipeline pipeline = make_pipeline(device, &layout, NULL);
    VkCommandPool pool = make_pool(device, 0, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    VkCommandPool pool2 = make_pool(device, 0, 0);
    VkCommandBuffer cb = make_cb(device, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY), dead_cb;
    VkDeviceSize offset = 0;
    VkSubmitInfo info;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cbi;
    VkSubmitInfo2 s2;
    void *data;

    vkGetDeviceQueue(device, 0, 0, &queue);
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    vkCreateSampler(device, &si, NULL, &sampler);
    EXPECT_NONE();

    vkDestroyBuffer(device, foreign, NULL);
    EXPECT("VUID-vkDestroyBuffer-buffer-parent");
    vkDestroyBuffer(other, foreign, NULL);
    EXPECT_NONE();

    vkDestroyBuffer(device, buffer, NULL);
    vkDestroyBuffer(device, buffer, NULL);
    EXPECT("VUID-vkDestroyBuffer-buffer-parameter");
    CHECK(seen.objects[0] == H(buffer));
    vkCmdCopyBuffer(cb, buffer, buffer, 1, &(VkBufferCopy){ 0, 0, 4 });
    EXPECT("VUID-vkCmdCopyBuffer-srcBuffer-parameter");
    vkCmdBindVertexBuffers(cb, 0, 1, &buffer, &offset);
    EXPECT("VUID-vkCmdBindVertexBuffers-pBuffers-parameter");
    vkCmdBindIndexBuffer(cb, buffer, 0, VK_INDEX_TYPE_UINT16);
    EXPECT("VUID-vkCmdBindIndexBuffer-buffer-parameter");
    buffer = make_buffer(device, NULL);
    vkFreeMemory(device, memory, NULL);
    CHECK_VK(vkBindBufferMemory(device, buffer, memory, 0), VK_ERROR_VALIDATION_FAILED_EXT);
    EXPECT("VUID-vkBindBufferMemory-memory-parameter");
    CHECK_VK(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &data), VK_ERROR_VALIDATION_FAILED_EXT);
    EXPECT("VUID-vkMapMemory-memory-parameter");
    vkFreeMemory(device, memory, NULL);
    EXPECT("VUID-vkFreeMemory-memory-parameter");
    vkDestroyBuffer(device, buffer, NULL);

    vkDestroyImage(device, image, NULL);
    vkDestroyImage(device, image, NULL);
    EXPECT("VUID-vkDestroyImage-image-parameter");
    vkDestroySampler(device, sampler, NULL);
    vkDestroySampler(device, sampler, NULL);
    EXPECT("VUID-vkDestroySampler-sampler-parameter");

    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    EXPECT("VUID-vkDestroyPipeline-pipeline-parameter");
    CHECK_VK(begin(cb, 0), VK_SUCCESS);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    EXPECT("VUID-vkCmdBindPipeline-pipeline-parameter");
    CHECK_VK(vkEndCommandBuffer(cb), VK_SUCCESS);
    vkDestroyPipelineLayout(device, layout, NULL);

    dead_cb = make_cb(device, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    vkFreeCommandBuffers(device, pool2, 1, &dead_cb);
    EXPECT("VUID-vkFreeCommandBuffers-pCommandBuffers-parent");
    vkFreeCommandBuffers(device, pool, 1, &dead_cb);
    vkFreeCommandBuffers(device, pool, 1, &dead_cb);
    EXPECT("VUID-vkFreeCommandBuffers-pCommandBuffers-00048");
    CHECK_VK(submit(queue, dead_cb, VK_NULL_HANDLE), VK_ERROR_VALIDATION_FAILED_EXT);
    EXPECT("VUID-VkSubmitInfo-pCommandBuffers-parameter");
    memset(&cbi, 0, sizeof(cbi));
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cbi.commandBuffer = dead_cb;
    memset(&s2, 0, sizeof(s2));
    s2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    s2.commandBufferInfoCount = 1;
    s2.pCommandBufferInfos = &cbi;
    CHECK_VK(vkQueueSubmit2(queue, 1, &s2, VK_NULL_HANDLE), VK_ERROR_VALIDATION_FAILED_EXT);
    EXPECT("VUID-VkCommandBufferSubmitInfo-commandBuffer-parameter");

    vkDestroySemaphore(device, sem, NULL);
    vkDestroySemaphore(device, sem, NULL);
    EXPECT("VUID-vkDestroySemaphore-semaphore-parameter");
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &sem;
    info.pWaitDstStageMask = &stage;
    CHECK_VK(vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE), VK_ERROR_VALIDATION_FAILED_EXT);
    EXPECT("VUID-VkSubmitInfo-pWaitSemaphores-parameter");
    vkDestroyFence(device, fence, NULL);
    vkDestroyFence(device, fence, NULL);
    EXPECT("VUID-vkDestroyFence-fence-parameter");
    CHECK_VK(submit(queue, cb, fence), VK_ERROR_VALIDATION_FAILED_EXT);
    EXPECT("VUID-vkQueueSubmit-fence-parameter");

    vkFreeCommandBuffers(device, pool, 1, &cb);
    vkDestroyCommandPool(device, pool, NULL);
    vkDestroyCommandPool(device, pool2, NULL);
    vkDestroyDevice(other, NULL);
    vkDestroyDevice(device, NULL);
    EXPECT_NONE();
}

/* Mismatched allocation callbacks: reported, the destruction skipped (the
 * object is destroyed afterwards with the right ones). */
static void test_allocation_callbacks(void)
{
    VkDevice device = make_device(NULL, NULL), dev2;
    VkBuffer plain = make_buffer(device, NULL), with = make_buffer(device, &callbacks);
    VkImage image = make_image(device, &callbacks);
    VkPipelineLayout layout;
    VkPipeline pipeline = make_pipeline(device, &layout, &callbacks);
    VkDebugUtilsMessengerCreateInfoEXT mi;
    VkDebugUtilsMessengerEXT m;

    vkDestroyBuffer(device, plain, &callbacks);
    EXPECT("VUID-vkDestroyBuffer-buffer-00924");
    vkDestroyBuffer(device, plain, NULL);
    vkDestroyBuffer(device, with, NULL);
    EXPECT("VUID-vkDestroyBuffer-buffer-00923");
    vkDestroyBuffer(device, with, &callbacks);
    vkDestroyImage(device, image, NULL);
    EXPECT("VUID-vkDestroyImage-image-01001");
    vkDestroyImage(device, image, &callbacks);
    vkDestroyPipeline(device, pipeline, NULL);
    EXPECT("VUID-vkDestroyPipeline-pipeline-00766");
    vkDestroyPipeline(device, pipeline, &callbacks);
    vkDestroyPipelineLayout(device, layout, NULL);
    EXPECT_NONE();
    CHECK(live_blocks == 0);

    memset(&mi, 0, sizeof(mi));
    mi.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
    mi.pfnUserCallback = record;
    CHECK_VK(vkCreateDebugUtilsMessengerEXT(instance, &mi, &callbacks, &m), VK_SUCCESS);
    vkDestroyDebugUtilsMessengerEXT(instance, m, NULL);
    EXPECT("VUID-vkDestroyDebugUtilsMessengerEXT-messenger-01915");
    vkDestroyDebugUtilsMessengerEXT(instance, m, &callbacks);
    vkDestroyDebugUtilsMessengerEXT(instance, m, &callbacks);
    EXPECT("VUID-vkDestroyDebugUtilsMessengerEXT-messenger-parameter");
    CHECK(live_blocks == 0);

    vkDestroyDevice(device, &callbacks);
    EXPECT("VUID-vkDestroyDevice-device-00380");
    dev2 = make_device(NULL, &callbacks);
    vkDestroyDevice(dev2, NULL);
    EXPECT("VUID-vkDestroyDevice-device-00379");
    CHECK(live_blocks == 0);
}

/* Objects left at vkDestroyDevice and vkDestroyInstance. */
static void test_leaks(void)
{
    VkDevice device = make_device(NULL, NULL);
    VkBuffer buffer = make_buffer(device, NULL);
    VkInstance other;
    VkInstanceCreateInfo info;
    VkDebugUtilsMessengerCreateInfoEXT chained, mi;
    VkDebugUtilsMessengerEXT leaked;
    const char *ext = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;

    vkDestroyDevice(device, NULL); /* the buffer: reported (and leaked) */
    CHECK(seen.count == 1 && seen.objects[0] == H(buffer));
    EXPECT("VUID-vkDestroyDevice-device-05137");

    memset(&chained, 0, sizeof(chained));
    chained.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    chained.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    chained.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    chained.pfnUserCallback = record;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pNext = &chained;
    info.enabledExtensionCount = 1;
    info.ppEnabledExtensionNames = &ext;
    REQUIRE_VK(vkCreateInstance(&info, NULL, &other));
    mi = chained;
    mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT; /* not validation: does not record */
    REQUIRE_VK(vkCreateDebugUtilsMessengerEXT(other, &mi, NULL, &leaked));
    vkDestroyInstance(other, &callbacks);
    /* to the chained messenger: the callbacks, then the messenger left */
    CHECK(seen.count == 2 && strcmp(seen.vuids[0], "VUID-vkDestroyInstance-instance-00631") == 0 &&
          strcmp(seen.vuids[1], "VUID-vkDestroyInstance-instance-00629") == 0 && seen.objects[1] == H(leaked));
    seen.count = 0;
}

static void test_command_buffer_states(void)
{
    VkDevice device = make_device(NULL, NULL);
    VkCommandPool resettable = make_pool(device, 0, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    VkCommandPool plain = make_pool(device, 0, 0);
    VkCommandBuffer cb = make_cb(device, resettable, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    VkCommandBuffer once = make_cb(device, plain, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    VkCommandBuffer sec = make_cb(device, resettable, VK_COMMAND_BUFFER_LEVEL_SECONDARY);
    VkDebugUtilsLabelEXT label;
    VkRenderingInfo ri;

    CHECK_VK(vkEndCommandBuffer(cb), VK_SUCCESS); /* not recording: reported, the state kept (not skipped) */
    EXPECT("VUID-vkEndCommandBuffer-commandBuffer-00059");
    CHECK_VK(vkResetCommandBuffer(cb, 0), VK_SUCCESS);
    CHECK_VK(begin(cb, 0), VK_SUCCESS);
    CHECK_VK(begin(cb, 0), VK_SUCCESS);
    EXPECT("VUID-vkBeginCommandBuffer-commandBuffer-00049");
    memset(&ri, 0, sizeof(ri));
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.extent.width = ri.renderArea.extent.height = 16;
    ri.layerCount = 1;
    vkCmdBeginRendering(cb, &ri);
    CHECK_VK(vkEndCommandBuffer(cb), VK_SUCCESS);
    EXPECT("VUID-vkEndCommandBuffer-commandBuffer-00060");

    CHECK_VK(begin(once, 0), VK_SUCCESS);
    CHECK_VK(vkEndCommandBuffer(once), VK_SUCCESS);
    CHECK_VK(begin(once, 0), VK_SUCCESS);
    EXPECT("VUID-vkBeginCommandBuffer-commandBuffer-00050");
    CHECK_VK(vkEndCommandBuffer(once), VK_SUCCESS);
    CHECK_VK(vkResetCommandBuffer(once, 0), VK_SUCCESS);
    EXPECT("VUID-vkResetCommandBuffer-commandBuffer-00046");

    memset(&label, 0, sizeof(label));
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = "secondary";
    CHECK_VK(begin(sec, 0), VK_SUCCESS);
    vkCmdEndDebugUtilsLabelEXT(sec);
    EXPECT("VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-01913");
    vkCmdBeginDebugUtilsLabelEXT(sec, &label);
    CHECK_VK(vkEndCommandBuffer(sec), VK_SUCCESS);
    EXPECT("VUID-vkEndCommandBuffer-commandBuffer-01815");

    vkDestroyCommandPool(device, resettable, NULL);
    vkDestroyCommandPool(device, plain, NULL);
    vkDestroyDevice(device, NULL);
    EXPECT_NONE();
}

static void test_submit_checks(void)
{
    VkDevice device = make_timeline_device();
    VkQueue queue;
    VkCommandPool pool = make_pool(device, 0, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    VkCommandPool compute_pool = make_pool(device, 1, 0);
    VkCommandBuffer cb = make_cb(device, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    VkCommandBuffer sec = make_cb(device, pool, VK_COMMAND_BUFFER_LEVEL_SECONDARY);
    VkCommandBuffer compute = make_cb(device, compute_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    VkFence signaled = make_fence(device, true), fence = make_fence(device, false);
    VkSemaphore timeline = make_semaphore(device, true);
    VkTimelineSemaphoreSubmitInfo values;
    VkSemaphoreSignalInfo signal;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cbi;
    VkSubmitInfo2 s2;
    VkSubmitInfo info;
    uint64_t one = 1;

    vkGetDeviceQueue(device, 0, 0, &queue);
    CHECK_VK(submit(queue, cb, VK_NULL_HANDLE), VK_ERROR_VALIDATION_FAILED_EXT);
    EXPECT("VUID-vkQueueSubmit-pCommandBuffers-00070");
    memset(&cbi, 0, sizeof(cbi));
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cbi.commandBuffer = cb;
    memset(&s2, 0, sizeof(s2));
    s2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    s2.commandBufferInfoCount = 1;
    s2.pCommandBufferInfos = &cbi;
    CHECK_VK(vkQueueSubmit2(queue, 1, &s2, VK_NULL_HANDLE), VK_ERROR_VALIDATION_FAILED_EXT);
    EXPECT("VUID-vkQueueSubmit2-commandBuffer-03874");

    CHECK_VK(begin(cb, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT), VK_SUCCESS);
    CHECK_VK(vkEndCommandBuffer(cb), VK_SUCCESS);
    CHECK_VK(submit(queue, cb, VK_NULL_HANDLE), VK_SUCCESS);
    CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);
    EXPECT_NONE();
    CHECK_VK(submit(queue, cb, VK_NULL_HANDLE), VK_ERROR_VALIDATION_FAILED_EXT); /* one-time: invalid now */
    EXPECT("VUID-vkQueueSubmit-pCommandBuffers-00070");

    CHECK_VK(begin(sec, 0), VK_SUCCESS);
    CHECK_VK(vkEndCommandBuffer(sec), VK_SUCCESS);
    CHECK_VK(submit(queue, sec, VK_NULL_HANDLE), VK_SUCCESS);
    EXPECT("VUID-VkSubmitInfo-pCommandBuffers-00075");
    CHECK_VK(begin(compute, 0), VK_SUCCESS);
    CHECK_VK(vkEndCommandBuffer(compute), VK_SUCCESS);
    CHECK_VK(submit(queue, compute, VK_NULL_HANDLE), VK_SUCCESS);
    EXPECT("VUID-vkQueueSubmit-pCommandBuffers-00074");
    CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);

    CHECK_VK(submit(queue, VK_NULL_HANDLE, signaled), VK_SUCCESS);
    EXPECT("VUID-vkQueueSubmit-fence-00063");
    CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);

    /* A submission held back by a timeline wait keeps its fence pending */
    memset(&values, 0, sizeof(values));
    values.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    values.waitSemaphoreValueCount = 1;
    values.pWaitSemaphoreValues = &one;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.pNext = &values;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &timeline;
    info.pWaitDstStageMask = &stage;
    CHECK_VK(vkQueueSubmit(queue, 1, &info, fence), VK_SUCCESS);
    CHECK_VK(submit(queue, VK_NULL_HANDLE, fence), VK_SUCCESS);
    EXPECT("VUID-vkQueueSubmit-fence-00064");
    memset(&signal, 0, sizeof(signal));
    signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal.semaphore = timeline;
    signal.value = 1;
    CHECK_VK(vkSignalSemaphore(device, &signal), VK_SUCCESS);
    CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);

    vkDestroySemaphore(device, timeline, NULL);
    vkDestroyFence(device, fence, NULL);
    vkDestroyFence(device, signaled, NULL);
    vkDestroyCommandPool(device, compute_pool, NULL);
    vkDestroyCommandPool(device, pool, NULL);
    vkDestroyDevice(device, NULL);
    EXPECT_NONE();
}

static void test_label_checks(void)
{
    VkDevice device = make_device(NULL, NULL);
    VkQueue queue;
    VkCommandPool pool = make_pool(device, 0, 0);
    VkCommandBuffer cb = make_cb(device, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    vkGetDeviceQueue(device, 0, 0, &queue);
    vkQueueEndDebugUtilsLabelEXT(queue);
    EXPECT("VUID-vkQueueEndDebugUtilsLabelEXT-None-01911");
    CHECK_VK(begin(cb, 0), VK_SUCCESS);
    vkCmdEndDebugUtilsLabelEXT(cb); /* a primary: it may close a region of an earlier command buffer */
    CHECK_VK(vkEndCommandBuffer(cb), VK_SUCCESS);
    EXPECT_NONE();
    CHECK_VK(submit(queue, cb, VK_NULL_HANDLE), VK_SUCCESS); /* none is open on the queue */
    EXPECT("VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-01912");
    CHECK_VK(vkQueueWaitIdle(queue), VK_SUCCESS);
    vkDestroyCommandPool(device, pool, NULL);
    vkDestroyDevice(device, NULL);
    EXPECT_NONE();
}

static void test_feature_checks(void)
{
    VkDevice device = make_device(NULL, NULL);
    VkPhysicalDeviceFeatures supported;
    VkPhysicalDeviceFeatures2 f2;
    VkPhysicalDeviceVulkan12Features v12;
    VkDevice full;
    VkCommandPool pool = make_pool(device, 0, 0);
    VkCommandBuffer cb = make_cb(device, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    VkBuffer buffer = make_buffer(device, NULL);
    VkDeviceMemory memory = make_memory(device, 0, 0);
    VkSamplerCreateInfo si;
    VkSampler sampler = VK_NULL_HANDLE;
    VkBufferDeviceAddressInfo bda;
    VkBool32 count;

    vkBindBufferMemory(device, buffer, memory, 0);
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.anisotropyEnable = VK_TRUE;
    si.maxAnisotropy = 2.0f;
    CHECK_VK(vkCreateSampler(device, &si, NULL, &sampler), VK_SUCCESS); /* reported, made */
    EXPECT("VUID-VkSamplerCreateInfo-anisotropyEnable-01070");
    vkDestroySampler(device, sampler, NULL);

    /* recorded, never submitted */
    CHECK_VK(begin(cb, 0), VK_SUCCESS);
    vkCmdDrawIndirect(cb, buffer, 0, 2, 16);
    EXPECT("VUID-vkCmdDrawIndirect-drawCount-02718");
    vkCmdDrawIndexedIndirect(cb, buffer, 0, 2, 20);
    EXPECT("VUID-vkCmdDrawIndexedIndirect-drawCount-02718");
    vkCmdDrawIndirectCount(cb, buffer, 0, buffer, 256, 4, 16);
    EXPECT("VUID-vkCmdDrawIndirectCount-None-04445");
    vkCmdDrawIndexedIndirectCount(cb, buffer, 0, buffer, 256, 4, 20);
    EXPECT("VUID-vkCmdDrawIndexedIndirectCount-None-04445");
    vkCmdSetLineWidth(cb, 2.0f);
    EXPECT("VUID-vkCmdSetLineWidth-lineWidth-00788");
    vkCmdSetLineWidth(cb, 1.0f);
    EXPECT_NONE();
    vkEndCommandBuffer(cb);
    memset(&bda, 0, sizeof(bda));
    bda.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bda.buffer = buffer;
    (void)vkGetBufferDeviceAddress(device, &bda);
    EXPECT("VUID-vkGetBufferDeviceAddress-bufferDeviceAddress-03324");
    vkDestroyBuffer(device, buffer, NULL);
    vkFreeMemory(device, memory, NULL);
    vkDestroyCommandPool(device, pool, NULL);
    vkDestroyDevice(device, NULL);

    /* With the features enabled (those supported): nothing to report */
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    memset(&f2, 0, sizeof(f2));
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &v12;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    supported = f2.features;
    memset(&f2.features, 0, sizeof(f2.features));
    f2.features.samplerAnisotropy = supported.samplerAnisotropy;
    f2.features.multiDrawIndirect = supported.multiDrawIndirect;
    count = v12.drawIndirectCount;
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.drawIndirectCount = count;
    full = make_device(&f2, NULL);
    pool = make_pool(full, 0, 0);
    cb = make_cb(full, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    buffer = make_buffer(full, NULL);
    memory = make_memory(full, 0, 0);
    vkBindBufferMemory(full, buffer, memory, 0);
    if (supported.samplerAnisotropy) {
        CHECK_VK(vkCreateSampler(full, &si, NULL, &sampler), VK_SUCCESS);
        vkDestroySampler(full, sampler, NULL);
    }
    CHECK_VK(begin(cb, 0), VK_SUCCESS);
    if (supported.multiDrawIndirect)
        vkCmdDrawIndirect(cb, buffer, 0, 2, 16);
    if (v12.drawIndirectCount)
        vkCmdDrawIndirectCount(cb, buffer, 0, buffer, 256, 4, 16);
    vkEndCommandBuffer(cb);
    EXPECT_NONE();
    vkDestroyBuffer(full, buffer, NULL);
    vkFreeMemory(full, memory, NULL);
    vkDestroyCommandPool(full, pool, NULL);
    vkDestroyDevice(full, NULL);
    EXPECT_NONE();
}

static void test_memory_checks(void)
{
    VkDevice device = make_device(NULL, NULL);
    VkDeviceMemory host = make_memory(device, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
    VkDeviceMemory local = make_memory(device, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VkBuffer buffer = make_buffer(device, NULL);
    VkImage image = make_image(device, NULL);
    VkDeviceMemory image_memory = make_memory(device, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VkBindBufferMemoryInfo bbi;
    VkMemoryMapInfo mmi;
    void *data;

    CHECK_VK(vkMapMemory(device, host, 0, VK_WHOLE_SIZE, 0, &data), VK_SUCCESS);
    CHECK_VK(vkMapMemory(device, host, 0, VK_WHOLE_SIZE, 0, &data), VK_SUCCESS);
    EXPECT("VUID-vkMapMemory-memory-00678");
    memset(&mmi, 0, sizeof(mmi));
    mmi.sType = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    mmi.memory = host;
    mmi.size = VK_WHOLE_SIZE;
    CHECK_VK(vkMapMemory2(device, &mmi, &data), VK_SUCCESS);
    EXPECT("VUID-VkMemoryMapInfo-memory-07958");
    vkUnmapMemory(device, host);
    vkUnmapMemory(device, host);
    EXPECT("VUID-vkUnmapMemory-memory-00689");
    CHECK_VK(vkMapMemory(device, local, 0, VK_WHOLE_SIZE, 0, &data), VK_ERROR_MEMORY_MAP_FAILED);
    EXPECT("VUID-vkMapMemory-memory-00682");

    CHECK_VK(vkBindBufferMemory(device, buffer, host, 0), VK_SUCCESS);
    CHECK_VK(vkBindBufferMemory(device, buffer, host, 0), VK_SUCCESS);
    EXPECT("VUID-vkBindBufferMemory-buffer-07459");
    memset(&bbi, 0, sizeof(bbi));
    bbi.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
    bbi.buffer = buffer;
    bbi.memory = host;
    CHECK_VK(vkBindBufferMemory2(device, 1, &bbi), VK_SUCCESS);
    EXPECT("VUID-VkBindBufferMemoryInfo-buffer-07459");
    CHECK_VK(vkBindImageMemory(device, image, image_memory, 0), VK_SUCCESS);
    CHECK_VK(vkBindImageMemory(device, image, image_memory, 0), VK_SUCCESS);
    EXPECT("VUID-vkBindImageMemory-image-07460");

    vkDestroyImage(device, image, NULL);
    vkDestroyBuffer(device, buffer, NULL);
    vkFreeMemory(device, image_memory, NULL);
    vkFreeMemory(device, local, NULL);
    vkFreeMemory(device, host, NULL);
    vkDestroyDevice(device, NULL);
    EXPECT_NONE();
}

static void test_name_checks(void)
{
    VkDevice device = make_device(NULL, NULL), other = make_device(NULL, NULL);
    VkBuffer buffer = make_buffer(device, NULL), foreign = make_buffer(other, NULL);
    VkDebugUtilsObjectNameInfoEXT info;
    VkDebugUtilsObjectTagInfoEXT tag;
    const char data = 1;

    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = VK_OBJECT_TYPE_UNKNOWN;
    info.objectHandle = H(buffer);
    info.pObjectName = "name";
    CHECK_VK(vkSetDebugUtilsObjectNameEXT(device, &info), VK_SUCCESS);
    EXPECT("VUID-vkSetDebugUtilsObjectNameEXT-pNameInfo-02587");
    info.objectType = VK_OBJECT_TYPE_BUFFER;
    info.objectHandle = 0;
    CHECK_VK(vkSetDebugUtilsObjectNameEXT(device, &info), VK_SUCCESS);
    EXPECT("VUID-vkSetDebugUtilsObjectNameEXT-pNameInfo-02588");
    info.objectHandle = H(foreign);
    CHECK_VK(vkSetDebugUtilsObjectNameEXT(device, &info), VK_SUCCESS);
    EXPECT("VUID-vkSetDebugUtilsObjectNameEXT-pNameInfo-07874");
    info.objectType = VK_OBJECT_TYPE_IMAGE; /* a buffer as an image: never touched */
    info.objectHandle = H(buffer);
    CHECK_VK(vkSetDebugUtilsObjectNameEXT(device, &info), VK_ERROR_VALIDATION_FAILED_EXT);
    EXPECT("VUID-VkDebugUtilsObjectNameInfoEXT-objectType-02590");
    vkDestroyBuffer(device, buffer, NULL);
    info.objectType = VK_OBJECT_TYPE_BUFFER;
    CHECK_VK(vkSetDebugUtilsObjectNameEXT(device, &info), VK_ERROR_VALIDATION_FAILED_EXT);
    EXPECT("VUID-VkDebugUtilsObjectNameInfoEXT-objectType-02590");
    memset(&tag, 0, sizeof(tag));
    tag.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_TAG_INFO_EXT;
    tag.objectType = VK_OBJECT_TYPE_UNKNOWN;
    tag.objectHandle = H(device);
    tag.tagSize = 1;
    tag.pTag = &data;
    CHECK_VK(vkSetDebugUtilsObjectTagEXT(device, &tag), VK_SUCCESS);
    EXPECT("VUID-VkDebugUtilsObjectTagInfoEXT-objectType-01908");

    vkDestroyBuffer(other, foreign, NULL);
    vkDestroyDevice(other, NULL);
    vkDestroyDevice(device, NULL);
    EXPECT_NONE();
}

/* A callback returning VK_TRUE makes the layer skip the command it reports. */
static void test_callback_skips(void)
{
    VkDevice device = make_device(NULL, NULL);
    VkCommandPool plain = make_pool(device, 0, 0);
    VkCommandBuffer cb = make_cb(device, plain, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    CHECK_VK(begin(cb, 0), VK_SUCCESS);
    CHECK_VK(vkEndCommandBuffer(cb), VK_SUCCESS);
    skip_requested = VK_TRUE;
    CHECK_VK(vkResetCommandBuffer(cb, 0), VK_ERROR_VALIDATION_FAILED_EXT);
    skip_requested = VK_FALSE;
    EXPECT("VUID-vkResetCommandBuffer-commandBuffer-00046");
    CHECK(((struct pvk_command_buffer *)cb)->state == PVK_CB_EXECUTABLE); /* not reset */
    vkDestroyCommandPool(device, plain, NULL);
    vkDestroyDevice(device, NULL);
    EXPECT_NONE();
}

int main(void)
{
    setenv("PLASMAVK_DEBUG", "1", 1); /* before PlasmaVK reads its environment */
    make_instance();
    RUN(test_layer_from_environment);
    RUN(test_valid_usage_is_quiet);
    RUN(test_destroyed_handles);
    RUN(test_allocation_callbacks);
    RUN(test_leaks);
    RUN(test_command_buffer_states);
    RUN(test_submit_checks);
    RUN(test_label_checks);
    RUN(test_feature_checks);
    RUN(test_memory_checks);
    RUN(test_name_checks);
    RUN(test_callback_skips);
    vkDestroyDebugUtilsMessengerEXT(instance, messenger, NULL);
    vkDestroyInstance(instance, NULL);
    return TEST_RESULT();
}
