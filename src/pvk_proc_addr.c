/* vkGetInstanceProcAddr / vkGetDeviceProcAddr over the generated table. */
#include "pvk_internal.h"
#include "gen/pvk_entrypoints.h"

static int pvk_entrypoint_compare(const void *name, const void *entry)
{
    return strcmp((const char *)name, ((const struct pvk_entrypoint *)entry)->name);
}

static const struct pvk_entrypoint *pvk_entrypoint_find(const char *name)
{
    if (!name)
        return NULL;
    return bsearch(name, pvk_entrypoints, PVK_ARRAY_SIZE(pvk_entrypoints), sizeof(pvk_entrypoints[0]),
                   pvk_entrypoint_compare);
}

/* TODO: return NULL for commands of extensions the instance/device did not
 * enable and of core versions above the requested apiVersion. */

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *pName)
{
    const struct pvk_entrypoint *e = pvk_entrypoint_find(pName);
    if (!e || (!instance && e->scope != PVK_SCOPE_GLOBAL))
        return NULL;
    return e->fn;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *pName)
{
    const struct pvk_entrypoint *e = pvk_entrypoint_find(pName);
    (void)device;
    if (!e || e->scope != PVK_SCOPE_DEVICE)
        return NULL;
    return e->fn;
}
