/* Startup: create an instance and a device, then tear both down - the cost
 * an application pays before it can render its first frame. */
#include "pvk_bench.h"

static int startup(void *user)
{
    const char *instance_exts[] = { VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME };
    float priority = 1.0f;
    VkInstanceCreateInfo instance_info;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo device_info;
    VkInstance instance;
    VkPhysicalDevice pd;
    VkDevice device;
    uint32_t count = 1;
    (void)user;

    /* The portability flag lets the loader expose MoltenVK too. */
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    instance_info.enabledExtensionCount = 1;
    instance_info.ppEnabledExtensionNames = instance_exts;
    if (vkCreateInstance(&instance_info, NULL, &instance) != VK_SUCCESS)
        return 1;
    if (vkEnumeratePhysicalDevices(instance, &count, &pd) < 0 || count == 0) {
        vkDestroyInstance(instance, NULL);
        return 1;
    }

    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    if (vkCreateDevice(pd, &device_info, NULL, &device) != VK_SUCCESS) {
        vkDestroyInstance(instance, NULL);
        return 1;
    }

    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}

int main(void)
{
    return bench_run("startup", 5, 100, startup, NULL);
}
