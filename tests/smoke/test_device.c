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
    RUN(test_unsupported_feature);
    vkDestroyInstance(instance, NULL);
    return TEST_RESULT();
}
