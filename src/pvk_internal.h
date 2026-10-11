/* Shared internals of the PlasmaVK implementation: object layouts, logging,
 * allocation. The whole implementation is one translation unit, so every
 * helper is static. */
#ifndef PVK_INTERNAL_H
#define PVK_INTERNAL_H

#include "plasmavk.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pvk_objc.h"
#include "pvk_metal.h"
#include "gen/pvk_prototypes.h"
#include "gen/pvk_extensions.h"

#define PVK_API_VERSION VK_MAKE_API_VERSION(0, 1, 4, VK_HEADER_VERSION)
#define PVK_VENDOR_ID_APPLE 0x106B
#define PVK_MAX_QUEUES 4
#define PVK_ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

/* ---- Logging ---- */

enum pvk_log_level { PVK_LOG_ERROR, PVK_LOG_WARN, PVK_LOG_INFO, PVK_LOG_DEBUG };

static int pvk_log_threshold = PVK_LOG_WARN;
static pthread_once_t pvk_log_once = PTHREAD_ONCE_INIT;

static void pvk_log_init(void)
{
    static const char *const names[] = { "error", "warn", "info", "debug" };
    const char *env = getenv("PLASMAVK_LOG");
    unsigned i;
    for (i = 0; env && i < PVK_ARRAY_SIZE(names); i++)
        if (strcmp(env, names[i]) == 0)
            pvk_log_threshold = (int)i;
}

static void pvk_log(enum pvk_log_level level, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void pvk_log(enum pvk_log_level level, const char *fmt, ...)
{
    static const char *const names[] = { "error", "warn", "info", "debug" };
    va_list args;
    pthread_once(&pvk_log_once, pvk_log_init);
    if ((int)level > pvk_log_threshold)
        return;
    fprintf(stderr, "plasmavk: %s: ", names[level]);
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
}

/* Body of every generated stub (src/gen/pvk_stubs.h). */
#ifdef PLASMAVK_ABORT_ON_UNIMPLEMENTED
#define PVK_UNIMPLEMENTED() (pvk_log(PVK_LOG_ERROR, "%s is not implemented", __func__), abort())
#else
#define PVK_UNIMPLEMENTED()                                              \
    do {                                                                 \
        static int pvk_warned_;                                          \
        if (!pvk_warned_) {                                              \
            pvk_warned_ = 1;                                             \
            pvk_log(PVK_LOG_WARN, "%s is not implemented", __func__);    \
        }                                                                \
    } while (0)
#endif

/* ---- Allocation ---- */

/* Zeroed allocation through the application's callbacks, if any. */
static void *pvk_alloc(const VkAllocationCallbacks *alloc, size_t size, VkSystemAllocationScope scope)
{
    void *p;
    if (!alloc)
        return calloc(1, size);
    p = alloc->pfnAllocation(alloc->pUserData, size, 16, scope);
    if (p)
        memset(p, 0, size);
    return p;
}

static void pvk_free(const VkAllocationCallbacks *alloc, void *p)
{
    if (!p)
        return;
    if (alloc)
        alloc->pfnFree(alloc->pUserData, p);
    else
        free(p);
}

/* ---- Objects ----
 * Handles are pointers to these structs. Without a loader, dispatchable
 * handles need no dispatch-table header. */

struct pvk_instance {
    VkAllocationCallbacks alloc;
    bool has_alloc;
    uint32_t api_version;
    bool extensions[PVK_INSTANCE_EXTENSION_COUNT];
    struct pvk_physical_device *physical_device; /* NULL: no Metal 4 GPU */
};

struct pvk_physical_device {
    struct pvk_instance *instance;
    id mtl_device;
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memory_properties;
};

struct pvk_queue {
    struct pvk_device *device;
    id mtl_queue; /* MTL4CommandQueue */
    uint32_t family_index;
    uint32_t index;
    VkDeviceQueueCreateFlags flags;
};

struct pvk_device {
    struct pvk_physical_device *physical_device;
    VkAllocationCallbacks alloc;
    bool has_alloc;
    bool extensions[PVK_DEVICE_EXTENSION_COUNT];
    uint32_t queue_count;
    struct pvk_queue queues[PVK_MAX_QUEUES];
};

static inline const VkAllocationCallbacks *pvk_instance_alloc(const struct pvk_instance *instance)
{
    return instance->has_alloc ? &instance->alloc : NULL;
}

static inline const VkAllocationCallbacks *pvk_device_alloc(const struct pvk_device *device)
{
    return device->has_alloc ? &device->alloc : NULL;
}

/* ---- Helpers ---- */

static inline int pvk_find_extension(const VkExtensionProperties *list, uint32_t count, const char *name)
{
    uint32_t i;
    for (i = 0; i < count; i++)
        if (strcmp(list[i].extensionName, name) == 0)
            return (int)i;
    return -1;
}

/* The standard two-call enumeration (count query, then fill). */
static VkResult pvk_enumerate(const void *items, size_t item_size, uint32_t count,
                              uint32_t *pCount, void *pOut)
{
    uint32_t n;
    if (!pOut) {
        *pCount = count;
        return VK_SUCCESS;
    }
    n = *pCount < count ? *pCount : count;
    if (n)
        memcpy(pOut, items, n * item_size);
    *pCount = n;
    return n < count ? VK_INCOMPLETE : VK_SUCCESS;
}

enum pvk_scope { PVK_SCOPE_GLOBAL, PVK_SCOPE_INSTANCE, PVK_SCOPE_DEVICE };

struct pvk_entrypoint {
    const char *name;
    PFN_vkVoidFunction fn;
    enum pvk_scope scope;
};

/* Defined in pvk_physical_device.c. */
static VkResult pvk_physical_device_create(struct pvk_instance *instance);
static void pvk_physical_device_destroy(struct pvk_physical_device *pd);

#endif /* PVK_INTERNAL_H */
