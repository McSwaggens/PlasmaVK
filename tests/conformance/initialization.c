/* Chapter "Initialization": command function pointers, instances. */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

/* ---- Command function pointers ---- */

static const char *const global_commands[] = {
    "vkEnumerateInstanceVersion",
    "vkEnumerateInstanceExtensionProperties",
    "vkEnumerateInstanceLayerProperties",
    "vkCreateInstance",
};

/* Names no Vulkan implementation defines. */
static const char *const unknown_names[] = {
    "vkNotARealCommand", "vkCreateInstanceX", "vkcreatedevice", "VkCreateDevice", "vkCreateDevice ", "",
    "vk", "vkGetInstanceProcAddrKHR",
};

CT_TEST(enumerate_instance_version,
        "Initialization > Instances: vkEnumerateInstanceVersion, a Vulkan 1.1 command, reports the instance-level "
        "version (at least 1.1, variant 0) and returns VK_SUCCESS; vkGetInstanceProcAddr(NULL) resolves it",
        "vkEnumerateInstanceVersion vkGetInstanceProcAddr")
{
    PFN_vkEnumerateInstanceVersion fn;
    uint32_t version = 0, again = 0;

    CT_REQUIRE_VK(t, vkEnumerateInstanceVersion(&version));
    CT_CHECKF(t, VK_API_VERSION_VARIANT(version) == 0, "variant %u, not Vulkan (0)", VK_API_VERSION_VARIANT(version));
    CT_CHECKF(t, CT_VERSION(version) >= VK_API_VERSION_1_1, "instance version %u.%u is below 1.1",
              VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version));
    fn = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceVersion");
    CT_REQUIRE(t, fn != NULL);
    CT_REQUIRE_VK(t, fn(&again));
    CT_CHECK(t, again == version);
}

CT_TEST(gipa_global_commands_null_instance,
        "Initialization > Command Function Pointers: vkGetInstanceProcAddr(NULL, global command) returns a valid "
        "function pointer for each global command (vkGetInstanceProcAddr behavior table)",
        "vkGetInstanceProcAddr vkEnumerateInstanceVersion vkEnumerateInstanceExtensionProperties "
        "vkEnumerateInstanceLayerProperties vkCreateInstance vkDestroyInstance")
{
    PFN_vkEnumerateInstanceExtensionProperties enum_ext;
    PFN_vkEnumerateInstanceLayerProperties enum_layers;
    PFN_vkCreateInstance create;
    VkInstanceCreateInfo info;
    VkInstance instance = VK_NULL_HANDLE;
    const char *portability[] = { VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME };
    uint32_t count = 0;
    size_t i;

    for (i = 0; i < CT_ARRAY_SIZE(global_commands); i++)
        CT_CHECKF(t, vkGetInstanceProcAddr(NULL, global_commands[i]) != NULL, "%s: NULL", global_commands[i]);

    /* The pointers work. */
    enum_ext = (PFN_vkEnumerateInstanceExtensionProperties)vkGetInstanceProcAddr(
        NULL, "vkEnumerateInstanceExtensionProperties");
    enum_layers = (PFN_vkEnumerateInstanceLayerProperties)vkGetInstanceProcAddr(NULL,
                                                                                 "vkEnumerateInstanceLayerProperties");
    create = (PFN_vkCreateInstance)vkGetInstanceProcAddr(NULL, "vkCreateInstance");
    CT_REQUIRE(t, enum_ext && enum_layers && create);
    CT_CHECK_VK(t, enum_ext(NULL, &count, NULL), VK_SUCCESS);
    CT_CHECK_VK(t, enum_layers(&count, NULL), VK_SUCCESS);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    if (ct_instance_extension_supported(portability[0])) {
        info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        info.enabledExtensionCount = 1;
        info.ppEnabledExtensionNames = portability;
    }
    CT_CHECK_VK(t, create(&info, NULL, &instance), VK_SUCCESS);
    vkDestroyInstance(instance, NULL);
}

CT_TEST(gipa_self_null_instance,
        "Initialization > Command Function Pointers: starting with Vulkan 1.2, vkGetInstanceProcAddr resolves "
        "itself with a NULL instance (vkGetInstanceProcAddr behavior table, footnote 5)",
        "vkGetInstanceProcAddr vkEnumerateInstanceVersion")
{
    PFN_vkGetInstanceProcAddr gipa;
    if (CT_VERSION(ct_instance_version()) < VK_API_VERSION_1_2)
        ct_skip(t, "instance version below 1.2");
    gipa = (PFN_vkGetInstanceProcAddr)vkGetInstanceProcAddr(NULL, "vkGetInstanceProcAddr");
    CT_REQUIRE(t, gipa != NULL);
    CT_CHECK(t, gipa(NULL, "vkEnumerateInstanceVersion") != NULL);
}

CT_TEST(gipa_null_instance_other_commands,
        "Initialization > Command Function Pointers: vkGetInstanceProcAddr(NULL, name) returns NULL for every "
        "name other than the global commands and vkGetInstanceProcAddr (\"any other case\" row)",
        "vkGetInstanceProcAddr")
{
    size_t i, j;
    for (i = 0; i < ct_command_count; i++) {
        const struct ct_command *c = &ct_commands[i];
        int global = 0;
        for (j = 0; j < CT_ARRAY_SIZE(global_commands); j++)
            global |= strcmp(c->name, global_commands[j]) == 0;
        if (global || strcmp(c->name, "vkGetInstanceProcAddr") == 0)
            continue;
        CT_CHECKF(t, vkGetInstanceProcAddr(NULL, c->name) == NULL, "vkGetInstanceProcAddr(NULL, \"%s\") is not NULL",
                  c->name);
    }
    for (i = 0; i < CT_ARRAY_SIZE(unknown_names); i++)
        CT_CHECKF(t, vkGetInstanceProcAddr(NULL, unknown_names[i]) == NULL,
                  "vkGetInstanceProcAddr(NULL, \"%s\") is not NULL", unknown_names[i]);
}

CT_TEST(gipa_core_commands,
        "Initialization > Command Function Pointers: vkGetInstanceProcAddr(instance, name) returns a valid "
        "pointer for vkGetInstanceProcAddr and every core dispatchable command of the instance's version "
        "(vkGetInstanceProcAddr behavior table)",
        "vkGetInstanceProcAddr vkGetPhysicalDeviceProperties vkEnumeratePhysicalDevices")
{
    struct ct_env e;
    PFN_vkEnumeratePhysicalDevices enumerate;
    uint32_t count = 0;
    size_t i;

    ct_env_init(t, &e);
    CT_CHECK(t, vkGetInstanceProcAddr(e.instance, "vkGetInstanceProcAddr") != NULL);
    for (i = 0; i < ct_command_count; i++) {
        const struct ct_command *c = &ct_commands[i];
        uint32_t version = ct_command_version(c);
        if (!version || c->level == CT_LEVEL_GLOBAL || version > e.app_version || version > e.instance_version)
            continue;
        CT_CHECKF(t, vkGetInstanceProcAddr(e.instance, c->name) != NULL, "%s (Vulkan %s): NULL", c->name, c->origin);
    }
    enumerate = (PFN_vkEnumeratePhysicalDevices)vkGetInstanceProcAddr(e.instance, "vkEnumeratePhysicalDevices");
    CT_REQUIRE(t, enumerate != NULL);
    CT_CHECK_VK(t, enumerate(e.instance, &count, NULL), VK_SUCCESS);
    CT_CHECK(t, count > 0);
    ct_env_fini(&e);
}

CT_TEST(gipa_global_commands_with_instance,
        "Initialization > Command Function Pointers: with a non-NULL instance, the table gives a pointer only "
        "for vkGetInstanceProcAddr, dispatchable commands and extension commands; global commands fall under "
        "\"any other case\" and return NULL",
        "vkGetInstanceProcAddr")
{
    struct ct_env e;
    size_t i;
    ct_env_init(t, &e);
    for (i = 0; i < CT_ARRAY_SIZE(global_commands); i++)
        CT_CHECKF(t, vkGetInstanceProcAddr(e.instance, global_commands[i]) == NULL,
                  "vkGetInstanceProcAddr(instance, \"%s\") is not NULL", global_commands[i]);
    ct_env_fini(&e);
}

CT_TEST(gipa_unknown_names,
        "Initialization > Command Function Pointers: vkGetInstanceProcAddr(instance, name) returns NULL for "
        "names that are not Vulkan commands (\"any other case\" row)",
        "vkGetInstanceProcAddr")
{
    struct ct_env e;
    size_t i;
    ct_env_init(t, &e);
    for (i = 0; i < CT_ARRAY_SIZE(unknown_names); i++)
        CT_CHECKF(t, vkGetInstanceProcAddr(e.instance, unknown_names[i]) == NULL,
                  "vkGetInstanceProcAddr(instance, \"%s\") is not NULL", unknown_names[i]);
    ct_env_fini(&e);
}

/* Instance extensions in ct_commands that the implementation offers. */
static uint32_t offered_instance_extensions(const char **names, uint32_t max)
{
    uint32_t n = 0;
    size_t i, j;
    for (i = 0; i < ct_command_count; i++) {
        const struct ct_command *c = &ct_commands[i];
        int seen = 0;
        if (ct_command_version(c) || c->level == CT_LEVEL_DEVICE)
            continue;
        if (strcmp(c->origin, "VK_KHR_swapchain") == 0)
            continue; /* a device extension with a physical-device-level command */
        for (j = 0; j < n; j++)
            seen |= strcmp(names[j], c->origin) == 0;
        if (!seen && n < max && ct_instance_extension_supported(c->origin))
            names[n++] = c->origin;
    }
    return n;
}

CT_TEST(gipa_enabled_instance_extension,
        "Initialization > Command Function Pointers: vkGetInstanceProcAddr(instance, name) returns a valid "
        "pointer for the commands of instance extensions enabled for instance",
        "vkGetInstanceProcAddr")
{
    const char *names[16];
    struct ct_env e;
    uint32_t n = offered_instance_extensions(names, 16), k;
    size_t i;

    if (!n)
        ct_skip(t, "no instance extension with commands in the command list is offered");
    ct_env_init_ex(t, &e, 0, names, n);
    for (k = 0; k < n; k++) {
        for (i = 0; i < ct_command_count; i++)
            if (strcmp(ct_commands[i].origin, names[k]) == 0)
                CT_CHECKF(t, vkGetInstanceProcAddr(e.instance, ct_commands[i].name) != NULL, "%s (%s enabled): NULL",
                          ct_commands[i].name, names[k]);
    }
    ct_env_fini(&e);
}

CT_TEST(gipa_disabled_instance_extension,
        "Initialization > Command Function Pointers: vkGetInstanceProcAddr(instance, name) returns NULL for the "
        "commands of instance extensions not enabled for instance (only enabled ones are listed in the table)",
        "vkGetInstanceProcAddr")
{
    struct ct_env e;
    size_t i;
    ct_env_init(t, &e); /* enables no extension but, if offered, portability enumeration */
    for (i = 0; i < ct_command_count; i++) {
        const struct ct_command *c = &ct_commands[i];
        if (ct_command_version(c) || strcmp(c->origin, "VK_KHR_swapchain") == 0)
            continue;
        CT_CHECKF(t, vkGetInstanceProcAddr(e.instance, c->name) == NULL, "%s (%s not enabled) is not NULL", c->name,
                  c->origin);
    }
    ct_env_fini(&e);
}

CT_TEST(gipa_available_device_extension,
        "Initialization > Command Function Pointers: vkGetInstanceProcAddr(instance, name) returns a valid "
        "pointer for the commands of device extensions available on a physical device of instance",
        "vkGetInstanceProcAddr")
{
    const char *names[16];
    struct ct_env e;
    uint32_t n = offered_instance_extensions(names, 16);
    size_t i;

    /* VK_KHR_swapchain counts as available only with VK_KHR_surface enabled. */
    ct_env_init_ex(t, &e, 0, names, n);
    if (!ct_device_extension_supported(e.pd, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
        ct_skip(t, "VK_KHR_swapchain is not available");
    for (i = 0; i < ct_command_count; i++) {
        const struct ct_command *c = &ct_commands[i];
        if (strcmp(c->origin, "VK_KHR_swapchain") != 0)
            continue;
        CT_CHECKF(t, vkGetInstanceProcAddr(e.instance, c->name) != NULL, "%s: NULL", c->name);
    }
    ct_env_fini(&e);
}

CT_TEST(gdpa_core_device_commands,
        "Initialization > Command Function Pointers: vkGetDeviceProcAddr returns a valid pointer for every "
        "device-level command of the core version requested in VkApplicationInfo::apiVersion (footnote 2), for "
        "each requested version 1.0 to 1.4",
        "vkGetDeviceProcAddr vkGetDeviceQueue vkCreateDevice vkDestroyDevice")
{
    static const uint32_t versions[] = { VK_API_VERSION_1_0, VK_API_VERSION_1_1, VK_API_VERSION_1_2,
                                         VK_API_VERSION_1_3, VK_API_VERSION_1_4 };
    size_t v, i;

    for (v = 0; v < CT_ARRAY_SIZE(versions); v++) {
        struct ct_env e;
        PFN_vkGetDeviceQueue get_queue;
        VkQueue queue = VK_NULL_HANDLE;
        if (versions[v] > CT_VERSION(ct_instance_version()))
            break;
        ct_env_init_ex(t, &e, versions[v], NULL, 0);
        ct_env_device(t, &e, NULL);
        for (i = 0; i < ct_command_count; i++) {
            const struct ct_command *c = &ct_commands[i];
            uint32_t version = ct_command_version(c);
            if (!version || c->level != CT_LEVEL_DEVICE || version > e.version)
                continue;
            CT_CHECKF(t, vkGetDeviceProcAddr(e.device, c->name) != NULL, "%s (Vulkan %s, apiVersion %u.%u): NULL",
                      c->name, c->origin, VK_API_VERSION_MAJOR(versions[v]), VK_API_VERSION_MINOR(versions[v]));
        }
        get_queue = (PFN_vkGetDeviceQueue)vkGetDeviceProcAddr(e.device, "vkGetDeviceQueue");
        CT_REQUIRE(t, get_queue != NULL);
        get_queue(e.device, e.queue_family, 0, &queue);
        CT_CHECK(t, queue == e.queue);
        ct_env_fini(&e);
    }
}

CT_TEST(gdpa_non_device_commands,
        "Initialization > Command Function Pointers: vkGetDeviceProcAddr returns NULL for global, instance-level "
        "and physical-device-level commands (device-level excludes them, footnote 3; \"any other case\")",
        "vkGetDeviceProcAddr")
{
    struct ct_env e;
    size_t i;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    for (i = 0; i < ct_command_count; i++) {
        const struct ct_command *c = &ct_commands[i];
        if (c->level == CT_LEVEL_DEVICE)
            continue;
        CT_CHECKF(t, vkGetDeviceProcAddr(e.device, c->name) == NULL, "vkGetDeviceProcAddr(device, \"%s\") is not NULL",
                  c->name);
    }
    ct_env_fini(&e);
}

CT_TEST(gdpa_unknown_names,
        "Initialization > Command Function Pointers: vkGetDeviceProcAddr returns NULL for names that are not "
        "Vulkan commands (\"any other case\")",
        "vkGetDeviceProcAddr")
{
    struct ct_env e;
    size_t i;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    for (i = 0; i < CT_ARRAY_SIZE(unknown_names); i++)
        CT_CHECKF(t, vkGetDeviceProcAddr(e.device, unknown_names[i]) == NULL,
                  "vkGetDeviceProcAddr(device, \"%s\") is not NULL", unknown_names[i]);
    ct_env_fini(&e);
}

CT_TEST(gdpa_disabled_device_extension,
        "Initialization > Command Function Pointers: vkGetDeviceProcAddr returns NULL for the commands of device "
        "extensions not enabled on the device (only enabled ones are listed; \"any other case\")",
        "vkGetDeviceProcAddr")
{
    struct ct_env e;
    size_t i;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL); /* no extension but, if offered, VK_KHR_portability_subset */
    for (i = 0; i < ct_command_count; i++) {
        const struct ct_command *c = &ct_commands[i];
        if (ct_command_version(c) || c->level != CT_LEVEL_DEVICE)
            continue;
        CT_CHECKF(t, vkGetDeviceProcAddr(e.device, c->name) == NULL, "%s (%s not enabled) is not NULL", c->name,
                  c->origin);
    }
    ct_env_fini(&e);
}

CT_TEST(gdpa_enabled_device_extension,
        "Initialization > Command Function Pointers: vkGetDeviceProcAddr returns a valid pointer for the "
        "device-level commands of enabled device extensions",
        "vkGetDeviceProcAddr vkCreateDevice vkDestroyDevice")
{
    const char *instance_exts[] = { VK_KHR_SURFACE_EXTENSION_NAME };
    const char *device_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    struct ct_env e;
    size_t i;

    if (!ct_instance_extension_supported(VK_KHR_SURFACE_EXTENSION_NAME))
        ct_skip(t, "VK_KHR_surface is not offered");
    ct_env_init_ex(t, &e, 0, instance_exts, 1);
    if (!ct_device_extension_supported(e.pd, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
        ct_skip(t, "VK_KHR_swapchain is not offered");
    ct_env_device_ex(t, &e, NULL, device_exts, 1, NULL);
    for (i = 0; i < ct_command_count; i++) {
        const struct ct_command *c = &ct_commands[i];
        /* The device-group commands of VK_KHR_swapchain need Vulkan 1.1. */
        if (strcmp(c->origin, "VK_KHR_swapchain") != 0 || c->level != CT_LEVEL_DEVICE)
            continue;
        if (e.version < VK_API_VERSION_1_1 && (strstr(c->name, "DeviceGroup") || strstr(c->name, "Image2")))
            continue;
        CT_CHECKF(t, vkGetDeviceProcAddr(e.device, c->name) != NULL, "%s: NULL", c->name);
    }
    ct_env_fini(&e);
}

CT_TEST(gdpa_beyond_requested_version,
        "Initialization > Command Function Pointers: with the maintenance5 feature enabled, vkGetDeviceProcAddr "
        "returns NULL for core commands beyond VkApplicationInfo::apiVersion (footnote 2)",
        "vkGetDeviceProcAddr")
{
    const char *exts[] = { "VK_KHR_maintenance5" };
    VkPhysicalDeviceMaintenance5Features m5;
    struct ct_env e;
    size_t i;

    if (CT_VERSION(ct_instance_version()) < VK_API_VERSION_1_3)
        ct_skip(t, "instance version below 1.3");
    ct_env_init_ex(t, &e, VK_API_VERSION_1_3, NULL, 0);
    if (e.version < VK_API_VERSION_1_3)
        ct_skip(t, "device version below 1.3");
    if (!ct_device_extension_supported(e.pd, "VK_KHR_maintenance5"))
        ct_skip(t, "VK_KHR_maintenance5 is not offered (needed to enable maintenance5 at apiVersion 1.3)");
    memset(&m5, 0, sizeof(m5));
    m5.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES;
    m5.maintenance5 = VK_TRUE;
    ct_env_device_ex(t, &e, &m5, exts, 1, NULL);
    for (i = 0; i < ct_command_count; i++) {
        const struct ct_command *c = &ct_commands[i];
        if (c->level != CT_LEVEL_DEVICE || ct_command_version(c) != VK_API_VERSION_1_4)
            continue;
        CT_CHECKF(t, vkGetDeviceProcAddr(e.device, c->name) == NULL, "%s (Vulkan 1.4, apiVersion 1.3) is not NULL",
                  c->name);
    }
    ct_env_fini(&e);
}

/* ---- Instances ---- */

CT_TEST(create_instance_null_application_info,
        "Initialization > Instances: VkInstanceCreateInfo::pApplicationInfo may be NULL (equivalent to apiVersion "
        "1.0); vkCreateInstance succeeds",
        "vkCreateInstance vkDestroyInstance vkEnumeratePhysicalDevices")
{
    VkInstanceCreateInfo info;
    VkInstance instance = VK_NULL_HANDLE;
    const char *portability[] = { VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME };
    uint32_t count = 0;

    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    if (ct_instance_extension_supported(portability[0])) {
        info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        info.enabledExtensionCount = 1;
        info.ppEnabledExtensionNames = portability;
    }
    CT_REQUIRE_VK(t, vkCreateInstance(&info, NULL, &instance));
    CT_CHECK(t, instance != VK_NULL_HANDLE);
    CT_CHECK_VK(t, vkEnumeratePhysicalDevices(instance, &count, NULL), VK_SUCCESS);
    CT_CHECK(t, count > 0);
    vkDestroyInstance(instance, NULL);
}

CT_TEST(create_instance_api_versions,
        "Initialization > Instances: VkApplicationInfo::apiVersion 0 is ignored, and implementations supporting "
        "Vulkan 1.1 or later must not return VK_ERROR_INCOMPATIBLE_DRIVER for any apiVersion (the patch is ignored)",
        "vkCreateInstance vkDestroyInstance")
{
    static const uint32_t versions[] = {
        0,
        VK_API_VERSION_1_0,
        VK_API_VERSION_1_1,
        VK_API_VERSION_1_2,
        VK_API_VERSION_1_3,
        VK_API_VERSION_1_4,
        VK_MAKE_API_VERSION(0, 1, 4, 4095),
        VK_MAKE_API_VERSION(0, 1, 5, 0),
        VK_MAKE_API_VERSION(0, 1, 1023, 0),
        VK_MAKE_API_VERSION(0, 2, 0, 0),
    };
    size_t i;
    if (CT_VERSION(ct_instance_version()) < VK_API_VERSION_1_1)
        ct_skip(t, "a Vulkan 1.0 instance may reject versions above 1.0");
    for (i = 0; i < CT_ARRAY_SIZE(versions); i++) {
        VkInstance instance = VK_NULL_HANDLE;
        VkResult r = ct_create_instance(versions[i], NULL, 0, NULL, &instance);
        CT_CHECKF(t, r == VK_SUCCESS, "apiVersion %u.%u.%u: %s", VK_API_VERSION_MAJOR(versions[i]),
                  VK_API_VERSION_MINOR(versions[i]), VK_API_VERSION_PATCH(versions[i]), ct_result_name(r));
        if (r == VK_SUCCESS)
            vkDestroyInstance(instance, NULL);
    }
}

CT_TEST(create_instance_application_info_fields,
        "Initialization > Instances: VkApplicationInfo names may be NULL or strings and versions any value; "
        "they only inform the implementation",
        "vkCreateInstance vkDestroyInstance")
{
    VkApplicationInfo app;
    VkInstanceCreateInfo info;
    VkInstance instance = VK_NULL_HANDLE;
    const char *portability[] = { VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME };
    int with_names;

    for (with_names = 0; with_names < 2; with_names++) {
        memset(&app, 0, sizeof(app));
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = with_names ? "app \xc3\xa9t\xc3\xa9" : NULL; /* UTF-8 */
        app.applicationVersion = with_names ? 0xFFFFFFFFu : 0;
        app.pEngineName = with_names ? "engine" : NULL;
        app.engineVersion = with_names ? 12345 : 0;
        app.apiVersion = CT_MIN(VK_API_VERSION_1_4, CT_VERSION(ct_instance_version()));
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pApplicationInfo = &app;
        if (ct_instance_extension_supported(portability[0])) {
            info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
            info.enabledExtensionCount = 1;
            info.ppEnabledExtensionNames = portability;
        }
        CT_CHECK_VK(t, vkCreateInstance(&info, NULL, &instance), VK_SUCCESS);
        vkDestroyInstance(instance, NULL);
        instance = VK_NULL_HANDLE;
    }
}

CT_TEST(create_instance_unsupported_extension,
        "Initialization > Instances: if any requested extension is not supported, vkCreateInstance must return "
        "VK_ERROR_EXTENSION_NOT_PRESENT",
        "vkCreateInstance")
{
    static const char *const bogus[] = { "VK_KHR_not_a_real_extension", "", "VK_KHR_SURFACE",
                                         VK_KHR_SWAPCHAIN_EXTENSION_NAME /* a device extension */ };
    size_t i;
    for (i = 0; i < CT_ARRAY_SIZE(bogus); i++) {
        VkInstance instance = VK_NULL_HANDLE;
        VkResult r = ct_create_instance(VK_API_VERSION_1_0, &bogus[i], 1, NULL, &instance);
        CT_CHECKF(t, r == VK_ERROR_EXTENSION_NOT_PRESENT, "extension \"%s\": %s", bogus[i], ct_result_name(r));
        if (r == VK_SUCCESS)
            vkDestroyInstance(instance, NULL);
    }
}

CT_TEST(create_instance_unsupported_layer,
        "Initialization > Instances: vkCreateInstance verifies that the requested layers exist; if not, it "
        "returns VK_ERROR_LAYER_NOT_PRESENT",
        "vkCreateInstance")
{
    const char *layers[] = { "VK_LAYER_PLASMAVK_not_a_real_layer" };
    VkInstanceCreateInfo info;
    VkInstance instance = VK_NULL_HANDLE;
    VkResult r;

    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.enabledLayerCount = 1;
    info.ppEnabledLayerNames = layers;
    r = vkCreateInstance(&info, NULL, &instance);
    CT_CHECKF(t, r == VK_ERROR_LAYER_NOT_PRESENT, "vkCreateInstance returned %s", ct_result_name(r));
    if (r == VK_SUCCESS)
        vkDestroyInstance(instance, NULL);
}

CT_TEST(destroy_instance_null,
        "Initialization > Instances: vkDestroyInstance accepts a NULL instance (implicit valid usage: \"if "
        "instance is not NULL\") and does nothing",
        "vkDestroyInstance")
{
    (void)t;
    vkDestroyInstance(VK_NULL_HANDLE, NULL);
}

CT_TEST(instance_allocation_callbacks,
        "Memory Allocation > Host Memory: host memory an instance needs comes from the allocator given to "
        "vkCreateInstance, with power-of-two alignments, valid scopes, on the calling thread, during commands, "
        "and is all freed by vkDestroyInstance",
        "vkCreateInstance vkDestroyInstance vkEnumeratePhysicalDevices vkGetPhysicalDeviceProperties")
{
    struct ct_allocator a;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice pds[8];
    VkPhysicalDeviceProperties props;
    uint32_t count = 8, version = CT_MIN(VK_API_VERSION_1_4, CT_VERSION(ct_instance_version()));
    VkResult r;

    ct_allocator_init(&a, -1);
    CT_CALL_VK(&a, r, ct_create_instance(version, NULL, 0, &a.cb, &instance));
    CT_REQUIRE_VK(t, r);
    CT_CHECKF(t, a.total > 0, "vkCreateInstance made no allocation through pAllocator: an instance needs host "
              "memory, which must come from the most specific allocator");
    ct_note(t, "vkCreateInstance made %ld allocations", a.total);
    CT_CALL_VK(&a, r, vkEnumeratePhysicalDevices(instance, &count, pds));
    CT_CHECK(t, r == VK_SUCCESS || r == VK_INCOMPLETE);
    if (count)
        CT_CALL(&a, vkGetPhysicalDeviceProperties(pds[0], &props));
    CT_CALL(&a, vkDestroyInstance(instance, &a.cb));
    CT_CHECKF(t, a.live == 0, "%ld allocations still live after vkDestroyInstance", a.live);
    ct_allocator_check(t, &a, "instance");
    ct_allocator_fini(&a);
}

CT_TEST(instance_allocation_failure,
        "Memory Allocation > Host Memory: when pfnAllocation returns NULL, vkCreateInstance either continues or "
        "returns VK_ERROR_OUT_OF_HOST_MEMORY, and frees everything it allocated (failing every allocation from "
        "the n-th on, for each of the first 128 n and a spread of the rest)",
        "vkCreateInstance vkDestroyInstance")
{
    uint32_t version = CT_MIN(VK_API_VERSION_1_4, CT_VERSION(ct_instance_version()));
    struct ct_allocator a;
    VkInstance instance = VK_NULL_HANDLE;
    long points[256], total;
    size_t k, n;
    VkResult r;

    ct_allocator_init(&a, -1);
    CT_CALL_VK(&a, r, ct_create_instance(version, NULL, 0, &a.cb, &instance));
    CT_REQUIRE_VK(t, r);
    total = a.total;
    CT_CALL(&a, vkDestroyInstance(instance, &a.cb));
    ct_allocator_fini(&a);
    n = ct_failure_points(total, points, CT_ARRAY_SIZE(points));
    for (k = 0; k < n && !t->failures; k++) {
        ct_allocator_init(&a, points[k]);
        instance = VK_NULL_HANDLE;
        CT_CALL_VK(&a, r, ct_create_instance(version, NULL, 0, &a.cb, &instance));
        CT_CHECKF(t, r == VK_SUCCESS || r == VK_ERROR_OUT_OF_HOST_MEMORY, "allocations from %ld on failing: %s",
                  points[k], ct_result_name(r));
        if (r == VK_SUCCESS)
            CT_CALL(&a, vkDestroyInstance(instance, &a.cb));
        CT_CHECKF(t, a.live == 0, "allocations from %ld on failing: %ld allocations leaked", points[k], a.live);
        ct_allocator_check(t, &a, "instance");
        ct_allocator_fini(&a);
    }
}

static const struct ct_test *const tests[] = {
    CT_REF(enumerate_instance_version),
    CT_REF(gipa_global_commands_null_instance),
    CT_REF(gipa_self_null_instance),
    CT_REF(gipa_null_instance_other_commands),
    CT_REF(gipa_core_commands),
    CT_REF(gipa_global_commands_with_instance),
    CT_REF(gipa_unknown_names),
    CT_REF(gipa_enabled_instance_extension),
    CT_REF(gipa_disabled_instance_extension),
    CT_REF(gipa_available_device_extension),
    CT_REF(gdpa_core_device_commands),
    CT_REF(gdpa_non_device_commands),
    CT_REF(gdpa_unknown_names),
    CT_REF(gdpa_disabled_device_extension),
    CT_REF(gdpa_enabled_device_extension),
    CT_REF(gdpa_beyond_requested_version),
    CT_REF(create_instance_null_application_info),
    CT_REF(create_instance_api_versions),
    CT_REF(create_instance_application_info_fields),
    CT_REF(create_instance_unsupported_extension),
    CT_REF(create_instance_unsupported_layer),
    CT_REF(destroy_instance_null),
    CT_REF(instance_allocation_callbacks),
    CT_REF(instance_allocation_failure),
};

CT_SUITE(initialization, tests)
