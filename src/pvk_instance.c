/* Global commands and VkInstance. */
#include "pvk_internal.h"

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceVersion(uint32_t *pApiVersion)
{
    *pApiVersion = PVK_API_VERSION;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *pLayerName,
                                                                      uint32_t *pPropertyCount,
                                                                      VkExtensionProperties *pProperties)
{
    if (pLayerName)
        return VK_ERROR_LAYER_NOT_PRESENT;
    return pvk_enumerate(pvk_instance_extensions, sizeof(VkExtensionProperties),
                         PVK_INSTANCE_EXTENSION_COUNT, pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pPropertyCount,
                                                                  VkLayerProperties *pProperties)
{
    (void)pProperties;
    *pPropertyCount = 0; /* layers need a loader */
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                                                const VkAllocationCallbacks *pAllocator,
                                                VkInstance *pInstance)
{
    struct pvk_instance *instance;
    VkResult result;
    uint32_t i;

    if (pCreateInfo->enabledLayerCount > 0) {
        pvk_log(PVK_LOG_WARN, "layers are not supported without a loader (requested %s)",
                pCreateInfo->ppEnabledLayerNames[0]);
        return VK_ERROR_LAYER_NOT_PRESENT;
    }

    instance = pvk_alloc(pAllocator, sizeof(*instance), VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
    if (!instance)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (pAllocator) {
        instance->alloc = *pAllocator;
        instance->has_alloc = true;
    }

    for (i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
        const char *name = pCreateInfo->ppEnabledExtensionNames[i];
        int ext = pvk_find_extension(pvk_instance_extensions, PVK_INSTANCE_EXTENSION_COUNT, name);
        if (ext < 0) {
            pvk_log(PVK_LOG_WARN, "unsupported instance extension %s", name);
            pvk_free(pAllocator, instance);
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
        instance->extensions[ext] = true;
    }

    instance->api_version = VK_API_VERSION_1_0;
    if (pCreateInfo->pApplicationInfo && pCreateInfo->pApplicationInfo->apiVersion)
        instance->api_version = pCreateInfo->pApplicationInfo->apiVersion;

    pvk_metal_init();
    result = pvk_physical_device_create(instance);
    if (result != VK_SUCCESS) {
        pvk_free(pAllocator, instance);
        return result;
    }

    *pInstance = (VkInstance)instance;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance _instance, const VkAllocationCallbacks *pAllocator)
{
    struct pvk_instance *instance = (struct pvk_instance *)_instance;
    (void)pAllocator; /* must match creation; the stored copy is authoritative */
    if (!instance)
        return;
    pvk_physical_device_destroy(instance->physical_device);
    pvk_free(pvk_instance_alloc(instance), instance);
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(VkInstance _instance,
                                                          uint32_t *pPhysicalDeviceCount,
                                                          VkPhysicalDevice *pPhysicalDevices)
{
    struct pvk_instance *instance = (struct pvk_instance *)_instance;
    VkPhysicalDevice handle = (VkPhysicalDevice)instance->physical_device;
    return pvk_enumerate(&handle, sizeof(handle), instance->physical_device ? 1 : 0,
                         pPhysicalDeviceCount, pPhysicalDevices);
}
