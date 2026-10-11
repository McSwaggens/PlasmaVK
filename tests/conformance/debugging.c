/* Chapter "Debugging", section "Active Tooling Information": the tools
 * active on a physical device (none on a driver used directly; a layer such
 * as the validation layer reports itself). */
#include "ct.h"

#include <stdlib.h>

#define DG_GUARD 0xA5

static int dg_same_tool(const VkPhysicalDeviceToolProperties *x, const VkPhysicalDeviceToolProperties *y)
{
    return memcmp(x->name, y->name, sizeof(x->name)) == 0 && memcmp(x->version, y->version, sizeof(x->version)) == 0 &&
           x->purposes == y->purposes && memcmp(x->description, y->description, sizeof(x->description)) == 0 &&
           memcmp(x->layer, y->layer, sizeof(x->layer)) == 0;
}

static int dg_terminated(const char *s, size_t n)
{
    return memchr(s, '\0', n) != NULL;
}

/* n structures with sType and pNext set, the rest the guard pattern. */
static void dg_prepare(VkPhysicalDeviceToolProperties *p, uint32_t n)
{
    uint32_t i;
    for (i = 0; i < n; i++) {
        memset(&p[i], DG_GUARD, sizeof(p[i]));
        p[i].sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES;
        p[i].pNext = NULL;
    }
}

static int dg_untouched(const VkPhysicalDeviceToolProperties *p)
{
    return p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES && p->pNext == NULL &&
           ct_all_bytes(p->name, sizeof(p->name), DG_GUARD) && ct_all_bytes(p->layer, sizeof(p->layer), DG_GUARD) &&
           ct_all_bytes(&p->purposes, sizeof(p->purposes), DG_GUARD);
}

CT_TEST(tool_properties,
        "Debugging > Active Tooling Information (vkGetPhysicalDeviceToolProperties, Vulkan 1.3): the count of "
        "active tools, then with an array the count overwritten with the number written, at most pToolCount "
        "(VK_INCOMPLETE with fewer, VK_SUCCESS with room to spare); name, version, description and layer are "
        "null-terminated UTF-8 strings within their arrays; the application's sType and pNext are kept",
        "vkGetPhysicalDeviceToolProperties")
{
    struct ct_env e;
    VkPhysicalDeviceToolProperties *full, *part;
    uint32_t count = 0, n, i, j, k;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_3)
        ct_skip(t, "vkGetPhysicalDeviceToolProperties is Vulkan 1.3");
    CT_REQUIRE_VK(t, vkGetPhysicalDeviceToolProperties(e.pd, &count, NULL));
    full = calloc(count + 1, sizeof(*full));
    part = calloc(count + 1, sizeof(*part));
    CT_REQUIRE(t, full && part);
    dg_prepare(full, count + 1);
    n = count + 1;
    CT_CHECK_VK(t, vkGetPhysicalDeviceToolProperties(e.pd, &n, full), VK_SUCCESS);
    CT_CHECKF(t, n == count, "%u tools counted, then %u written with room for %u", count, n, count + 1);
    CT_CHECKF(t, dg_untouched(&full[count]), "wrote past the last tool");
    for (i = 0; i < n && i < count; i++) {
        const VkPhysicalDeviceToolProperties *p = &full[i];
        CT_CHECKF(t, p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES && p->pNext == NULL,
                  "tool %u: sType or pNext changed", i);
        CT_CHECKF(t, dg_terminated(p->name, sizeof(p->name)) && dg_terminated(p->version, sizeof(p->version)) &&
                         dg_terminated(p->description, sizeof(p->description)) &&
                         dg_terminated(p->layer, sizeof(p->layer)),
                  "tool %u: a string is not null-terminated", i);
        if (dg_terminated(p->name, sizeof(p->name)) && dg_terminated(p->layer, sizeof(p->layer)))
            ct_note(t, "tool %u: \"%s\" (layer \"%s\"), purposes 0x%x", i, p->name, p->layer, p->purposes);
    }
    for (k = 0; k < count; k++) {
        dg_prepare(part, count + 1);
        n = k;
        CT_CHECK_VK(t, vkGetPhysicalDeviceToolProperties(e.pd, &n, part), VK_INCOMPLETE);
        CT_CHECKF(t, n == k, "room for %u of %u tools: count returned %u", k, count, n);
        CT_CHECKF(t, dg_untouched(&part[k]), "room for %u tools: wrote tool %u", k, k);
        for (i = 0; i < n && i < k; i++) {
            int found = 0;
            for (j = 0; j < count; j++)
                found |= dg_same_tool(&part[i], &full[j]);
            CT_CHECKF(t, found, "room for %u tools: tool %u is not one of the full list", k, i);
        }
    }
    free(full);
    free(part);
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(tool_properties),
};

CT_SUITE(debugging, tests)
