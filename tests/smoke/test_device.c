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
    uint32_t count = 0;

    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, NULL);
    REQUIRE(count == 1);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, props);
    CHECK(props[0].queueFlags & VK_QUEUE_GRAPHICS_BIT);
    CHECK(props[0].queueFlags & VK_QUEUE_COMPUTE_BIT);
    CHECK(props[0].queueCount >= 1);
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
    RUN(test_memory_properties);
    RUN(test_device_and_queues);
    RUN(test_unsupported_feature);
    vkDestroyInstance(instance, NULL);
    return TEST_RESULT();
}
