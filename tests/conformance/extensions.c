/* Chapter "Layers & Extensions" (Extending Vulkan): enumerating layers and
 * extensions, and extension dependencies. */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

/* A string field must hold a NUL within its array. */
static int terminated(const char *s, size_t size)
{
    return memchr(s, '\0', size) != NULL;
}

/* Checks the extension list rules shared by instance and device queries. */
static void check_extension_list(struct ct *t, const VkExtensionProperties *props, uint32_t count, const char *what)
{
    uint32_t i, j;
    for (i = 0; i < count; i++) {
        CT_CHECKF(t, terminated(props[i].extensionName, VK_MAX_EXTENSION_NAME_SIZE),
                  "%s extension %u: extensionName is not NUL-terminated within VK_MAX_EXTENSION_NAME_SIZE", what, i);
        if (!terminated(props[i].extensionName, VK_MAX_EXTENSION_NAME_SIZE))
            continue;
        CT_CHECKF(t, props[i].extensionName[0] != '\0', "%s extension %u has an empty name", what, i);
        for (j = 0; j < i; j++)
            CT_CHECKF(t, strcmp(props[i].extensionName, props[j].extensionName) != 0, "%s extension %s listed twice",
                      what, props[i].extensionName);
    }
}

/* Field-wise: bytes after a string's NUL are not specified. */
static int same_extension(const void *a, const void *b)
{
    const VkExtensionProperties *x = a, *y = b;
    return strncmp(x->extensionName, y->extensionName, VK_MAX_EXTENSION_NAME_SIZE) == 0 &&
           x->specVersion == y->specVersion;
}

static int same_layer(const void *a, const void *b)
{
    const VkLayerProperties *x = a, *y = b;
    return strncmp(x->layerName, y->layerName, VK_MAX_EXTENSION_NAME_SIZE) == 0 && x->specVersion == y->specVersion &&
           x->implementationVersion == y->implementationVersion &&
           strncmp(x->description, y->description, VK_MAX_DESCRIPTION_SIZE) == 0;
}

#define check_incomplete(t, fn, ctx, size, what)                                                                   \
    ct_check_incomplete(t, fn, ctx, size, (size) == sizeof(VkExtensionProperties) ? same_extension : same_layer, what)

static VkResult enum_instance_extensions(void *ctx, uint32_t *count, void *items)
{
    return vkEnumerateInstanceExtensionProperties(ctx, count, items);
}

static VkResult enum_instance_layers(void *ctx, uint32_t *count, void *items)
{
    (void)ctx;
    return vkEnumerateInstanceLayerProperties(count, items);
}

struct device_ctx {
    VkPhysicalDevice pd;
    const char *layer;
};

static VkResult enum_device_extensions(void *ctx, uint32_t *count, void *items)
{
    struct device_ctx *d = ctx;
    return vkEnumerateDeviceExtensionProperties(d->pd, d->layer, count, items);
}

static VkResult enum_device_layers(void *ctx, uint32_t *count, void *items)
{
    struct device_ctx *d = ctx;
    return vkEnumerateDeviceLayerProperties(d->pd, count, items);
}

static VkExtensionProperties *get_instance_extensions(struct ct *t, const char *layer, uint32_t *count)
{
    VkExtensionProperties *props;
    *count = 0;
    CT_REQUIRE_VK(t, vkEnumerateInstanceExtensionProperties(layer, count, NULL));
    props = calloc(*count + 1, sizeof(*props));
    CT_REQUIRE(t, props != NULL);
    CT_REQUIRE_VK(t, vkEnumerateInstanceExtensionProperties(layer, count, props));
    return props;
}

static VkLayerProperties *get_instance_layers(struct ct *t, uint32_t *count)
{
    VkLayerProperties *props;
    *count = 0;
    CT_REQUIRE_VK(t, vkEnumerateInstanceLayerProperties(count, NULL));
    props = calloc(*count + 1, sizeof(*props));
    CT_REQUIRE(t, props != NULL);
    CT_REQUIRE_VK(t, vkEnumerateInstanceLayerProperties(count, props));
    return props;
}

static VkExtensionProperties *get_device_extensions(struct ct *t, VkPhysicalDevice pd, uint32_t *count)
{
    VkExtensionProperties *props;
    *count = 0;
    CT_REQUIRE_VK(t, vkEnumerateDeviceExtensionProperties(pd, NULL, count, NULL));
    props = calloc(*count + 1, sizeof(*props));
    CT_REQUIRE(t, props != NULL);
    CT_REQUIRE_VK(t, vkEnumerateDeviceExtensionProperties(pd, NULL, count, props));
    return props;
}

/* ---- Instance ---- */

CT_TEST(instance_extensions,
        "Layers & Extensions > Instance Extensions: vkEnumerateInstanceExtensionProperties returns the count, then "
        "properties whose extensionName is a NUL-terminated name, each extension once",
        "vkEnumerateInstanceExtensionProperties")
{
    uint32_t count;
    VkExtensionProperties *props = get_instance_extensions(t, NULL, &count);
    check_extension_list(t, props, count, "instance");
    free(props);
}

CT_TEST(instance_extensions_incomplete,
        "Layers & Extensions > Instance Extensions: with pPropertyCount less than the number available, at most "
        "pPropertyCount structures are written and VK_INCOMPLETE is returned; on return the count is the number "
        "written",
        "vkEnumerateInstanceExtensionProperties")
{
    check_incomplete(t, enum_instance_extensions, NULL, sizeof(VkExtensionProperties), "instance extensions");
}

CT_TEST(instance_extensions_missing_layer,
        "Layers & Extensions > Instance Extensions: naming a layer that is not present in pLayerName fails with "
        "VK_ERROR_LAYER_NOT_PRESENT",
        "vkEnumerateInstanceExtensionProperties")
{
    uint32_t count = 0;
    VkExtensionProperties props[4];
    CT_CHECK_VK(t, vkEnumerateInstanceExtensionProperties("VK_LAYER_PLASMAVK_not_a_real_layer", &count, NULL),
                VK_ERROR_LAYER_NOT_PRESENT);
    count = 4;
    CT_CHECK_VK(t, vkEnumerateInstanceExtensionProperties("VK_LAYER_PLASMAVK_not_a_real_layer", &count, props),
                VK_ERROR_LAYER_NOT_PRESENT);
}

CT_TEST(instance_layer_extensions,
        "Layers & Extensions > Instance Extensions: when pLayerName names an available layer, the instance "
        "extensions provided by that layer are returned (same list rules)",
        "vkEnumerateInstanceExtensionProperties vkEnumerateInstanceLayerProperties")
{
    uint32_t layers, i;
    VkLayerProperties *layer = get_instance_layers(t, &layers);
    if (!layers)
        ct_skip(t, "no layers are available");
    for (i = 0; i < layers; i++) {
        uint32_t count;
        VkExtensionProperties *props = get_instance_extensions(t, layer[i].layerName, &count);
        check_extension_list(t, props, count, layer[i].layerName);
        check_incomplete(t, enum_instance_extensions, layer[i].layerName, sizeof(VkExtensionProperties),
                         layer[i].layerName);
        free(props);
    }
    free(layer);
}

CT_TEST(instance_layers,
        "Layers & Extensions > Layers: vkEnumerateInstanceLayerProperties returns layers whose layerName and "
        "description are NUL-terminated strings, each layer once, with the VK_INCOMPLETE protocol",
        "vkEnumerateInstanceLayerProperties")
{
    uint32_t count, i, j;
    VkLayerProperties *props = get_instance_layers(t, &count);
    for (i = 0; i < count; i++) {
        CT_CHECKF(t, terminated(props[i].layerName, VK_MAX_EXTENSION_NAME_SIZE), "layer %u: unterminated name", i);
        CT_CHECKF(t, terminated(props[i].description, VK_MAX_DESCRIPTION_SIZE), "layer %u: unterminated description",
                  i);
        for (j = 0; j < i; j++)
            CT_CHECKF(t, strncmp(props[i].layerName, props[j].layerName, VK_MAX_EXTENSION_NAME_SIZE) != 0,
                      "layer listed twice");
    }
    check_incomplete(t, enum_instance_layers, NULL, sizeof(VkLayerProperties), "instance layers");
    free(props);
}

CT_TEST(instance_extensions_enable_all,
        "Layers & Extensions: implementations must not advertise extensions that cannot be enabled together, and "
        "the required extensions of a supported extension are supported; enabling every offered instance "
        "extension at once succeeds",
        "vkEnumerateInstanceExtensionProperties vkCreateInstance vkDestroyInstance")
{
    uint32_t count, i;
    VkExtensionProperties *props = get_instance_extensions(t, NULL, &count);
    const char **names = calloc(count + 1, sizeof(*names));
    VkInstance instance = VK_NULL_HANDLE;

    CT_REQUIRE(t, names != NULL);
    for (i = 0; i < count; i++)
        names[i] = props[i].extensionName;
    CT_CHECK_VK(t, ct_create_instance(CT_MIN(VK_API_VERSION_1_4, CT_VERSION(ct_instance_version())), names, count,
                                      NULL, &instance),
                VK_SUCCESS);
    vkDestroyInstance(instance, NULL);
    free(names);
    free(props);
}

/* ---- Device ---- */

CT_TEST(device_extensions,
        "Layers & Extensions > Device Extensions: vkEnumerateDeviceExtensionProperties returns the count, then "
        "properties whose extensionName is a NUL-terminated name, each extension once",
        "vkEnumerateDeviceExtensionProperties")
{
    struct ct_env e;
    uint32_t count;
    VkExtensionProperties *props;
    ct_env_init(t, &e);
    props = get_device_extensions(t, e.pd, &count);
    check_extension_list(t, props, count, "device");
    free(props);
    ct_env_fini(&e);
}

CT_TEST(device_extensions_incomplete,
        "Layers & Extensions > Device Extensions: pPropertyCount is treated as for instance extensions: fewer "
        "slots than extensions write that many and return VK_INCOMPLETE",
        "vkEnumerateDeviceExtensionProperties")
{
    struct ct_env e;
    struct device_ctx d;
    ct_env_init(t, &e);
    d.pd = e.pd;
    d.layer = NULL;
    check_incomplete(t, enum_device_extensions, &d, sizeof(VkExtensionProperties), "device extensions");
    ct_env_fini(&e);
}

/* The spec doesn't say what an absent layer name returns: with pLayerName
 * "the device extensions provided by that layer are returned" (an absent
 * layer provides none), and VK_ERROR_LAYER_NOT_PRESENT ("a requested layer is
 * not present") is a listed failure code. Either answer is conformant; what
 * is not is reporting extensions for a layer that doesn't exist. (An earlier
 * version required VK_ERROR_LAYER_NOT_PRESENT, which no "must" supports: the
 * Khronos loader answers VK_SUCCESS with no extensions.) */
CT_TEST(device_extensions_missing_layer,
        "Layers & Extensions > Device Extensions: naming a layer that is not present in pLayerName returns no "
        "extensions: either VK_ERROR_LAYER_NOT_PRESENT or VK_SUCCESS with a count of zero",
        "vkEnumerateDeviceExtensionProperties")
{
    struct ct_env e;
    uint32_t count = 12345;
    VkResult r;
    ct_env_init(t, &e);
    r = vkEnumerateDeviceExtensionProperties(e.pd, "VK_LAYER_PLASMAVK_not_a_real_layer", &count, NULL);
    CT_CHECKF(t, r == VK_ERROR_LAYER_NOT_PRESENT || (r == VK_SUCCESS && count == 0),
              "returned %s with a count of %u", ct_result_name(r), count);
    ct_env_fini(&e);
}

CT_TEST(device_layers,
        "Layers & Extensions > Layers: vkEnumerateDeviceLayerProperties lists exactly the layers enabled for the "
        "instance, with the same properties as vkEnumerateInstanceLayerProperties, and the VK_INCOMPLETE protocol",
        "vkEnumerateDeviceLayerProperties vkEnumerateInstanceLayerProperties")
{
    struct ct_env e;
    struct device_ctx d;
    VkLayerProperties *instance_layers, *device_layers;
    uint32_t instance_count, count = 0, i, j;

    instance_layers = get_instance_layers(t, &instance_count);
    ct_env_init(t, &e); /* no layer enabled explicitly */
    CT_REQUIRE_VK(t, vkEnumerateDeviceLayerProperties(e.pd, &count, NULL));
    device_layers = calloc(count + 1, sizeof(*device_layers));
    CT_REQUIRE(t, device_layers != NULL);
    CT_REQUIRE_VK(t, vkEnumerateDeviceLayerProperties(e.pd, &count, device_layers));
    /* Only implicitly enabled layers can be active. */
    for (i = 0; i < count; i++) {
        int found = 0;
        for (j = 0; j < instance_count; j++)
            found |= same_layer(&device_layers[i], &instance_layers[j]);
        CT_CHECKF(t, found, "device layer %u is not an instance layer with the same properties", i);
    }
    d.pd = e.pd;
    d.layer = NULL;
    check_incomplete(t, enum_device_layers, &d, sizeof(VkLayerProperties), "device layers");
    free(device_layers);
    free(instance_layers);
    ct_env_fini(&e);
}

/* Device extensions that depend on an instance extension in every way they
 * can be enabled at Vulkan 1.4. Derived from the depends="" attributes of
 * vk.xml 1.4.363: an instance extension promoted to core (e.g.
 * VK_KHR_get_physical_device_properties2) counts as satisfied. */
static const char *const needs_instance_extension[] = {
    "VK_AMD_display_native_hdr",
    "VK_EXT_debug_marker",
    "VK_EXT_device_address_binding_report",
    "VK_EXT_display_control",
    "VK_EXT_full_screen_exclusive",
    "VK_EXT_hdr_metadata",
    "VK_EXT_image_compression_control_swapchain",
    "VK_EXT_multisampled_render_to_swapchain",
    "VK_EXT_present_mode_fifo_latest_ready",
    "VK_EXT_present_timing",
    "VK_EXT_swapchain_maintenance1",
    "VK_GGP_frame_token",
    "VK_GOOGLE_display_timing",
    "VK_HUAWEI_hdr_vivid",
    "VK_KHR_display_swapchain",
    "VK_KHR_incremental_present",
    "VK_KHR_present_id",
    "VK_KHR_present_id2",
    "VK_KHR_present_mode_fifo_latest_ready",
    "VK_KHR_present_wait",
    "VK_KHR_present_wait2",
    "VK_KHR_shared_presentable_image",
    "VK_KHR_swapchain",
    "VK_KHR_swapchain_maintenance1",
    "VK_KHR_swapchain_mutable_format",
    "VK_NV_acquire_winrt_display",
    "VK_NV_external_memory",
    "VK_NV_external_memory_win32",
    "VK_NV_low_latency2",
    "VK_NV_present_barrier",
    "VK_NV_win32_keyed_mutex",
};

CT_TEST(device_extension_needs_instance_extension,
        "Layers & Extensions > Extension Dependencies: a device extension with an instance extension dependency "
        "not enabled by vkCreateInstance is unsupported and must not be returned by "
        "vkEnumerateDeviceExtensionProperties (e.g. VK_KHR_swapchain without VK_KHR_surface)",
        "vkEnumerateDeviceExtensionProperties")
{
    struct ct_env e;
    uint32_t count, i;
    size_t j;
    VkExtensionProperties *props;

    ct_env_init(t, &e); /* no instance extension but, if offered, portability enumeration */
    props = get_device_extensions(t, e.pd, &count);
    for (i = 0; i < count; i++)
        for (j = 0; j < CT_ARRAY_SIZE(needs_instance_extension); j++)
            CT_CHECKF(t, strcmp(props[i].extensionName, needs_instance_extension[j]) != 0,
                      "%s is offered although the instance enabled none of the instance extensions it needs",
                      props[i].extensionName);
    free(props);
    ct_env_fini(&e);
}

CT_TEST(device_extension_with_instance_extension,
        "Layers & Extensions > Extension Dependencies: with VK_KHR_surface enabled on the instance, a device "
        "offering VK_KHR_swapchain lists it and it can be enabled",
        "vkEnumerateDeviceExtensionProperties vkCreateDevice vkDestroyDevice")
{
    const char *instance_exts[] = { VK_KHR_SURFACE_EXTENSION_NAME };
    const char *device_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    struct ct_env e;

    if (!ct_instance_extension_supported(VK_KHR_SURFACE_EXTENSION_NAME))
        ct_skip(t, "VK_KHR_surface is not offered");
    ct_env_init_ex(t, &e, 0, instance_exts, 1);
    if (!ct_device_extension_supported(e.pd, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
        ct_skip(t, "VK_KHR_swapchain is not offered");
    ct_env_device_ex(t, &e, NULL, device_exts, 1, NULL);
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(instance_extensions),
    CT_REF(instance_extensions_incomplete),
    CT_REF(instance_extensions_missing_layer),
    CT_REF(instance_layer_extensions),
    CT_REF(instance_layers),
    CT_REF(instance_extensions_enable_all),
    CT_REF(device_extensions),
    CT_REF(device_extensions_incomplete),
    CT_REF(device_extensions_missing_layer),
    CT_REF(device_layers),
    CT_REF(device_extension_needs_instance_extension),
    CT_REF(device_extension_with_instance_extension),
};

CT_SUITE(extensions, tests)
