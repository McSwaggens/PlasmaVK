/* VkPhysicalDevice: one per instance, backed by the system Metal 4 GPU. */
#include "pvk_internal.h"

static void pvk_physical_device_init_properties(struct pvk_physical_device *pd)
{
    VkPhysicalDeviceProperties *p = &pd->properties;
    VkPhysicalDeviceMemoryProperties *m = &pd->memory_properties;

    p->apiVersion = PVK_API_VERSION;
    p->driverVersion = PLASMAVK_VERSION;
    p->vendorID = PVK_VENDOR_ID_APPLE;
    p->deviceID = 0; /* TODO: derive from the GPU family */
    p->deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    snprintf(p->deviceName, sizeof(p->deviceName), "%s", pvk_mtl_device_name(pd->mtl_device));
    memcpy(p->pipelineCacheUUID, "PlasmaVK", 8); /* TODO: include version and GPU family */
    /* TODO: limits and sparseProperties, filled in as the subsystems that
     * honour them land. Zero until then. */

    /* Apple GPUs have unified memory: one heap, private and shared types. */
    m->memoryHeapCount = 1;
    m->memoryHeaps[0].size = PVK_MSG(uint64_t, (id, SEL))(pd->mtl_device, pvk_sel.recommendedMaxWorkingSetSize);
    m->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    m->memoryTypeCount = 2;
    m->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT; /* MTLStorageModePrivate */
    m->memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | /* MTLStorageModeShared */
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                      VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
}

static VkResult pvk_physical_device_create(struct pvk_instance *instance)
{
    struct pvk_physical_device *pd;
    id mtl = MTLCreateSystemDefaultDevice();

    if (!mtl) {
        pvk_log(PVK_LOG_INFO, "no Metal device; exposing no physical devices");
        return VK_SUCCESS;
    }
    if (!pvk_mtl_device_supports_family(mtl, PVK_MTL_GPU_FAMILY_METAL4)) {
        pvk_log(PVK_LOG_WARN, "%s does not support Metal 4; exposing no physical devices",
                pvk_mtl_device_name(mtl));
        pvk_release(mtl);
        return VK_SUCCESS;
    }

    pd = pvk_alloc(pvk_instance_alloc(instance), sizeof(*pd), VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
    if (!pd) {
        pvk_release(mtl);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    pd->instance = instance;
    pd->mtl_device = mtl;
    pvk_physical_device_init_properties(pd);
    instance->physical_device = pd;
    return VK_SUCCESS;
}

static void pvk_physical_device_destroy(struct pvk_physical_device *pd)
{
    if (!pd)
        return;
    pvk_release(pd->mtl_device);
    pvk_free(pvk_instance_alloc(pd->instance), pd);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(VkPhysicalDevice physicalDevice,
                                                         VkPhysicalDeviceProperties *pProperties)
{
    *pProperties = ((struct pvk_physical_device *)physicalDevice)->properties;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(VkPhysicalDevice physicalDevice,
                                                          VkPhysicalDeviceProperties2 *pProperties)
{
    /* TODO: fill pNext structures */
    vkGetPhysicalDeviceProperties(physicalDevice, &pProperties->properties);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures(VkPhysicalDevice physicalDevice,
                                                       VkPhysicalDeviceFeatures *pFeatures)
{
    (void)physicalDevice;
    memset(pFeatures, 0, sizeof(*pFeatures)); /* TODO: report features as they land */
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice physicalDevice,
                                                        VkPhysicalDeviceFeatures2 *pFeatures)
{
    /* TODO: fill pNext structures */
    vkGetPhysicalDeviceFeatures(physicalDevice, &pFeatures->features);
}

static const VkQueueFamilyProperties pvk_queue_family = {
    VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
    PVK_MAX_QUEUES,
    64,        /* timestampValidBits */
    { 1, 1, 1 } /* minImageTransferGranularity */
};

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice physicalDevice,
                                                                    uint32_t *pQueueFamilyPropertyCount,
                                                                    VkQueueFamilyProperties *pQueueFamilyProperties)
{
    (void)physicalDevice;
    pvk_enumerate(&pvk_queue_family, sizeof(pvk_queue_family), 1,
                  pQueueFamilyPropertyCount, pQueueFamilyProperties);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties2(VkPhysicalDevice physicalDevice,
                                                                     uint32_t *pQueueFamilyPropertyCount,
                                                                     VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
    (void)physicalDevice;
    if (!pQueueFamilyProperties) {
        *pQueueFamilyPropertyCount = 1;
        return;
    }
    if (*pQueueFamilyPropertyCount == 0)
        return;
    *pQueueFamilyPropertyCount = 1;
    pQueueFamilyProperties[0].queueFamilyProperties = pvk_queue_family; /* TODO: pNext */
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice physicalDevice,
                                                               VkPhysicalDeviceMemoryProperties *pMemoryProperties)
{
    *pMemoryProperties = ((struct pvk_physical_device *)physicalDevice)->memory_properties;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties2(VkPhysicalDevice physicalDevice,
                                                                VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
    /* TODO: VkPhysicalDeviceMemoryBudgetPropertiesEXT */
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &pMemoryProperties->memoryProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                                                    const char *pLayerName,
                                                                    uint32_t *pPropertyCount,
                                                                    VkExtensionProperties *pProperties)
{
    (void)physicalDevice;
    if (pLayerName)
        return VK_ERROR_LAYER_NOT_PRESENT;
    return pvk_enumerate(pvk_device_extensions, sizeof(VkExtensionProperties),
                         PVK_DEVICE_EXTENSION_COUNT, pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice,
                                                                uint32_t *pPropertyCount,
                                                                VkLayerProperties *pProperties)
{
    (void)physicalDevice;
    (void)pProperties;
    *pPropertyCount = 0;
    return VK_SUCCESS;
}
