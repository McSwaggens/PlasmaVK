#define PLASMAVK_IMPLEMENTATION
#include "plasmavk.h"

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
    VkExtensionProperties props[16];
    uint32_t count = 0, one = 1;

    REQUIRE_VK(vkEnumerateInstanceExtensionProperties(NULL, &count, NULL));
    REQUIRE(count >= 2 && count <= 16);
    CHECK_VK(vkEnumerateInstanceExtensionProperties(NULL, &one, props), VK_INCOMPLETE);
    CHECK(one == 1);
    REQUIRE_VK(vkEnumerateInstanceExtensionProperties(NULL, &count, props));
    CHECK(has_extension(props, count, VK_KHR_SURFACE_EXTENSION_NAME));
    CHECK(has_extension(props, count, VK_EXT_METAL_SURFACE_EXTENSION_NAME));
    CHECK_VK(vkEnumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation", &count, NULL),
             VK_ERROR_LAYER_NOT_PRESENT);
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
        CHECK(strncmp(props.deviceName, "Apple", 5) == 0);
        printf("  device: %s\n", props.deviceName);
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
    RUN(test_unsupported_extension);
    RUN(test_physical_device);
    RUN(test_proc_addr);
    RUN(test_allocator);
    return TEST_RESULT();
}
