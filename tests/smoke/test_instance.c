#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <stdlib.h>
#include <string.h>

#include "../pvk_test.h"

static VkInstance create_instance(void)
{
    VkApplicationInfo app;
    VkInstanceCreateInfo info;
    VkInstance instance = VK_NULL_HANDLE;

    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_4;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &app;
    if (vkCreateInstance(&info, NULL, &instance) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return instance;
}

static void test_version(void)
{
    uint32_t version = 0;
    CHECK_VK(vkEnumerateInstanceVersion(&version), VK_SUCCESS);
    CHECK(VK_API_VERSION_MAJOR(version) == 1 && VK_API_VERSION_MINOR(version) == 4);
}

static int has_extension(const VkExtensionProperties *props, uint32_t count, const char *name)
{
    uint32_t i;
    for (i = 0; i < count; i++)
        if (strcmp(props[i].extensionName, name) == 0)
            return 1;
    return 0;
}

static void test_instance_extensions(void)
{
    VkExtensionProperties props[32];
    uint32_t count = 0, one = 1;

    REQUIRE_VK(vkEnumerateInstanceExtensionProperties(NULL, &count, NULL));
    REQUIRE(count >= 2 && count <= 32);
    CHECK_VK(vkEnumerateInstanceExtensionProperties(NULL, &one, props), VK_INCOMPLETE);
    CHECK(one == 1);
    REQUIRE_VK(vkEnumerateInstanceExtensionProperties(NULL, &count, props));
    CHECK(has_extension(props, count, VK_KHR_SURFACE_EXTENSION_NAME));
    CHECK(has_extension(props, count, VK_EXT_METAL_SURFACE_EXTENSION_NAME));
    /* The instance extensions promoted to Vulkan 1.1 */
    CHECK(has_extension(props, count, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME));
    CHECK(has_extension(props, count, VK_KHR_DEVICE_GROUP_CREATION_EXTENSION_NAME));
    CHECK(has_extension(props, count, VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME));
    CHECK(has_extension(props, count, VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME));
    CHECK(has_extension(props, count, VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME));
    CHECK_VK(vkEnumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation", &count, NULL),
             VK_ERROR_LAYER_NOT_PRESENT);
}

/* The hand-maintained extension tables in plasmavk.h (designated
 * initializers): every slot is filled, each name once. */
static void check_extension_table(const VkExtensionProperties *props, uint32_t count, const char *what)
{
    uint32_t i, j;
    for (i = 0; i < count; i++) {
        if (strncmp(props[i].extensionName, "VK_", 3) != 0 || props[i].specVersion == 0)
            PVK_TEST_FAIL("%s extension %u is not filled in", what, i);
        for (j = 0; j < i; j++)
            if (strcmp(props[i].extensionName, props[j].extensionName) == 0)
                PVK_TEST_FAIL("%s extension %s is listed twice", what, props[i].extensionName);
    }
}

static void test_extension_tables(void)
{
    check_extension_table(pvk_instance_extensions, PVK_INSTANCE_EXTENSION_COUNT, "instance");
    check_extension_table(pvk_device_extensions, PVK_DEVICE_EXTENSION_COUNT, "device");
}

static void test_unsupported_extension(void)
{
    const char *names[] = { "VK_KHR_not_a_real_extension" };
    VkInstanceCreateInfo info;
    VkInstance instance;

    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.enabledExtensionCount = 1;
    info.ppEnabledExtensionNames = names;
    CHECK_VK(vkCreateInstance(&info, NULL, &instance), VK_ERROR_EXTENSION_NOT_PRESENT);
}

static void test_physical_device(void)
{
    VkInstance instance = create_instance();
    VkPhysicalDevice pd;
    VkPhysicalDeviceProperties props;
    uint32_t count = 0;
    id system;

    REQUIRE(instance);
    CHECK_VK(vkEnumeratePhysicalDevices(instance, &count, NULL), VK_SUCCESS);
    CHECK(count == 1);
    count = 1;
    CHECK_VK(vkEnumeratePhysicalDevices(instance, &count, &pd), VK_SUCCESS);
    if (count == 1) {
        vkGetPhysicalDeviceProperties(pd, &props);
        CHECK(props.vendorID == 0x106B);
        CHECK(props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
        CHECK(VK_API_VERSION_MINOR(props.apiVersion) == 4);
        CHECK(props.limits.maxComputeWorkGroupSize[0] >= 1024); /* struct return through objc_msgSend */
        CHECK(strncmp(props.deviceName, "Apple", 5) == 0);
        printf("  device: %s\n", props.deviceName);
        /* the system default device (white box: found through
         * MTLCopyAllDevices on macOS, pvk_system_default_device) */
        system = pvk_MTLCreateSystemDefaultDevice();
        CHECK(system && ((struct pvk_physical_device *)pd)->mtl_device == system);
        pvk_release(system);
    }
    vkDestroyInstance(instance, NULL);
}

static void test_proc_addr(void)
{
    VkInstance instance = create_instance();
    REQUIRE(instance);

    CHECK(vkGetInstanceProcAddr(NULL, "vkCreateInstance") == (PFN_vkVoidFunction)vkCreateInstance);
    CHECK(vkGetInstanceProcAddr(NULL, "vkGetInstanceProcAddr") == (PFN_vkVoidFunction)vkGetInstanceProcAddr);
    CHECK(vkGetInstanceProcAddr(NULL, "vkCreateDevice") == NULL);
    CHECK(vkGetInstanceProcAddr(instance, "vkCreateDevice") == (PFN_vkVoidFunction)vkCreateDevice);
    CHECK(vkGetInstanceProcAddr(instance, "vkCmdDraw") != NULL);
    CHECK(vkGetInstanceProcAddr(instance, "vkNotARealCommand") == NULL);
    CHECK(vkGetInstanceProcAddr(instance, NULL) == NULL);
    vkDestroyInstance(instance, NULL);
}

/* The hand-maintained table in plasmavk.h: alphabetical (so a command's
 * several entries, the ways it comes, are next to each other), with one
 * function and scope per command, extension tags in range, and with every
 * instance extension enabled every entry but the global commands resolves
 * through vkGetInstanceProcAddr. */
static void test_entrypoint_table(void)
{
    VkExtensionProperties props[PVK_INSTANCE_EXTENSION_COUNT];
    const char *extensions[PVK_INSTANCE_EXTENSION_COUNT];
    VkInstanceCreateInfo info;
    VkInstance instance = VK_NULL_HANDLE;
    uint32_t count = PVK_INSTANCE_EXTENSION_COUNT;
    size_t i;

    REQUIRE_VK(vkEnumerateInstanceExtensionProperties(NULL, &count, props));
    for (i = 0; i < count; i++)
        extensions[i] = props[i].extensionName;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.enabledExtensionCount = count;
    info.ppEnabledExtensionNames = extensions;
    REQUIRE_VK(vkCreateInstance(&info, NULL, &instance));

    for (i = 0; i < PVK_ARRAY_SIZE(pvk_entrypoints); i++) {
        const struct pvk_entrypoint *e = &pvk_entrypoints[i];
        int global = PVK_SCOPE(e->kind) == PVK_SCOPE_GLOBAL && e->fn != (PFN_vkVoidFunction)vkGetInstanceProcAddr;
        CHECK(e->fn != NULL);
        if (vkGetInstanceProcAddr(instance, e->name) != (global ? NULL : e->fn))
            PVK_TEST_FAIL("vkGetInstanceProcAddr(instance, \"%s\")", e->name);
        if (PVK_KIND_IEXT(e->kind) >= PVK_INSTANCE_EXTENSION_COUNT || PVK_KIND_DEXT(e->kind) >= PVK_DEVICE_EXTENSION_COUNT ||
            PVK_KIND_DEXT2(e->kind) >= PVK_DEVICE_EXTENSION_COUNT)
            PVK_TEST_FAIL("%s: extension out of range", e->name);
        if (i > 0) {
            const struct pvk_entrypoint *p = &pvk_entrypoints[i - 1];
            int order = strcmp(p->name, e->name);
            if (order > 0)
                PVK_TEST_FAIL("%s is not in alphabetical order", e->name);
            if (order == 0 && (p->fn != e->fn || PVK_SCOPE(p->kind) != PVK_SCOPE(e->kind) || p->kind == e->kind))
                PVK_TEST_FAIL("entries of %s differ in function or scope, or are the same", e->name);
        }
    }
    CHECK(PVK_ARRAY_SIZE(pvk_entrypoints) >= 363);
    vkDestroyInstance(instance, NULL);
}

/* ---- Allocation callbacks are honoured and balanced ---- */

static int live_allocations;

static void *VKAPI_PTR test_alloc(void *user, size_t size, size_t align, VkSystemAllocationScope scope)
{
    void *p = NULL;
    (void)user;
    (void)scope;
    if (posix_memalign(&p, align < sizeof(void *) ? sizeof(void *) : align, size) != 0)
        return NULL;
    live_allocations++;
    return p;
}

static void *VKAPI_PTR test_realloc(void *user, void *old, size_t size, size_t align, VkSystemAllocationScope scope)
{
    (void)user, (void)old, (void)size, (void)align, (void)scope;
    return NULL; /* PlasmaVK never reallocates (yet) */
}

static void VKAPI_PTR test_free(void *user, void *p)
{
    (void)user;
    if (p) {
        live_allocations--;
        free(p);
    }
}

static void test_allocator(void)
{
    VkAllocationCallbacks callbacks;
    VkInstanceCreateInfo info;
    VkInstance instance;

    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.pfnAllocation = test_alloc;
    callbacks.pfnReallocation = test_realloc;
    callbacks.pfnFree = test_free;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

    REQUIRE_VK(vkCreateInstance(&info, &callbacks, &instance));
    CHECK(live_allocations > 0);
    vkDestroyInstance(instance, &callbacks);
    CHECK(live_allocations == 0);
}

int main(void)
{
    RUN(test_version);
    RUN(test_instance_extensions);
    RUN(test_extension_tables);
    RUN(test_unsupported_extension);
    RUN(test_physical_device);
    RUN(test_proc_addr);
    RUN(test_entrypoint_table);
    RUN(test_allocator);
    return TEST_RESULT();
}
