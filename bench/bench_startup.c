/* Startup: create an instance and a device, then tear both down - the cost
 * an application pays before it can render its first frame. Like a real
 * application it enumerates device extensions to find (and enable)
 * VK_KHR_portability_subset. */
#include "pvk_bench.h"

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

int main(void)
{
    bench_begin("startup");
    bench_run("instance_device", 5, bench_frames(100), startup, NULL);
    return bench_exit_code();
}
