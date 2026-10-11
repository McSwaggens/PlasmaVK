/* VkDevice and VkQueue. Each VkQueue is an MTL4CommandQueue. */
#include "pvk_internal.h"

/* VkPhysicalDeviceFeatures is a struct of VkBool32. */
static bool pvk_features_supported(struct pvk_physical_device *pd, const VkPhysicalDeviceFeatures *wanted)
{
    VkPhysicalDeviceFeatures have;
    const VkBool32 *w = (const VkBool32 *)wanted;
    const VkBool32 *h = (const VkBool32 *)&have;
    size_t i;

    vkGetPhysicalDeviceFeatures((VkPhysicalDevice)pd, &have);
    for (i = 0; i < sizeof(have) / sizeof(VkBool32); i++)
        if (w[i] && !h[i])
            return false;
    return true;
}

static void pvk_device_free(struct pvk_device *device)
{
    uint32_t i;
    for (i = 0; i < device->queue_count; i++)
        pvk_release(device->queues[i].mtl_queue);
    pvk_free(pvk_device_alloc(device), device);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice physicalDevice,
                                              const VkDeviceCreateInfo *pCreateInfo,
                                              const VkAllocationCallbacks *pAllocator,
                                              VkDevice *pDevice)
{
    struct pvk_physical_device *pd = (struct pvk_physical_device *)physicalDevice;
    const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : pvk_instance_alloc(pd->instance);
    const VkBaseInStructure *ext;
    struct pvk_device *device;
    uint32_t i, j;

    /* Device layers are deprecated; enabledLayerCount is ignored. */
    if (pCreateInfo->pEnabledFeatures && !pvk_features_supported(pd, pCreateInfo->pEnabledFeatures))
        return VK_ERROR_FEATURE_NOT_PRESENT;
    for (ext = pCreateInfo->pNext; ext; ext = ext->pNext) {
        if (ext->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 &&
            !pvk_features_supported(pd, &((const VkPhysicalDeviceFeatures2 *)ext)->features))
            return VK_ERROR_FEATURE_NOT_PRESENT;
        /* TODO: validate the other feature structs */
    }

    device = pvk_alloc(alloc, sizeof(*device), VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
    if (!device)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    device->physical_device = pd;
    if (alloc) {
        device->alloc = *alloc;
        device->has_alloc = true;
    }

    for (i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
        const char *name = pCreateInfo->ppEnabledExtensionNames[i];
        int index = pvk_find_extension(pvk_device_extensions, PVK_DEVICE_EXTENSION_COUNT, name);
        if (index < 0) {
            pvk_log(PVK_LOG_WARN, "unsupported device extension %s", name);
            pvk_device_free(device);
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
        device->extensions[index] = true;
    }

    for (i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
        const VkDeviceQueueCreateInfo *info = &pCreateInfo->pQueueCreateInfos[i];
        for (j = 0; j < info->queueCount; j++) {
            struct pvk_queue *queue;
            if (device->queue_count == PVK_MAX_QUEUES) {
                pvk_device_free(device);
                return VK_ERROR_INITIALIZATION_FAILED;
            }
            queue = &device->queues[device->queue_count];
            queue->mtl_queue = PVK_MSG(id, (id, SEL))(pd->mtl_device, pvk_sel.newMTL4CommandQueue);
            if (!queue->mtl_queue) {
                pvk_device_free(device);
                return VK_ERROR_INITIALIZATION_FAILED;
            }
            queue->device = device;
            queue->family_index = info->queueFamilyIndex;
            queue->index = j;
            queue->flags = info->flags;
            device->queue_count++;
        }
    }

    *pDevice = (VkDevice)device;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
    (void)pAllocator; /* must match creation; the stored copy is authoritative */
    if (_device)
        pvk_device_free((struct pvk_device *)_device);
}

VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue2(VkDevice _device, const VkDeviceQueueInfo2 *pQueueInfo,
                                             VkQueue *pQueue)
{
    struct pvk_device *device = (struct pvk_device *)_device;
    uint32_t i;

    *pQueue = VK_NULL_HANDLE;
    for (i = 0; i < device->queue_count; i++) {
        struct pvk_queue *q = &device->queues[i];
        if (q->family_index == pQueueInfo->queueFamilyIndex && q->index == pQueueInfo->queueIndex &&
            q->flags == pQueueInfo->flags) {
            *pQueue = (VkQueue)q;
            return;
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex,
                                            VkQueue *pQueue)
{
    VkDeviceQueueInfo2 info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2;
    info.queueFamilyIndex = queueFamilyIndex;
    info.queueIndex = queueIndex;
    vkGetDeviceQueue2(device, &info, pQueue);
}
