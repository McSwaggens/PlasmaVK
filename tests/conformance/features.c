/* Chapter "Features" > Feature Requirements: the features every Vulkan
 * graphics implementation must support, per core version. One test per
 * feature, so the baseline tracks each.
 *
 * Source: the <require> feature lists of the VK_VERSION_1_0 .. VK_VERSION_1_4
 * blocks of vk.xml 1.4.363, from which the specification generates its
 * Feature Requirements list. A feature can be required unconditionally from
 * one version and conditionally from an earlier one (when an extension is
 * offered or another feature is supported). */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

enum where { F10, F11, F12, F13, F14, F_NONE };

struct ref {
    enum where where;
    size_t offset;
};

struct requirement {
    const char *name;
    struct ref feature;
    uint32_t always;            /* required from this version unconditionally (0: never) */
    uint32_t when;              /* required from this version if (0: never) ... */
    const char *when_ext;       /* ... this device extension is offered, or */
    struct ref when_feature[2]; /* ... one of these features is supported */
    const char *when_text;
};

#define R10(m) { F10, offsetof(VkPhysicalDeviceFeatures, m) }
#define R11(m) { F11, offsetof(VkPhysicalDeviceVulkan11Features, m) }
#define R12(m) { F12, offsetof(VkPhysicalDeviceVulkan12Features, m) }
#define R13(m) { F13, offsetof(VkPhysicalDeviceVulkan13Features, m) }
#define R14(m) { F14, offsetof(VkPhysicalDeviceVulkan14Features, m) }
#define NO_REF { F_NONE, 0 }

#define V10 VK_API_VERSION_1_0
#define V11 VK_API_VERSION_1_1
#define V12 VK_API_VERSION_1_2
#define V13 VK_API_VERSION_1_3
#define V14 VK_API_VERSION_1_4

/* Required from version v. */
#define ALWAYS(s, m, v) { #m, s(m), v, 0, NULL, { NO_REF, NO_REF }, NULL }
/* Required from `always` (0: never), and from `when` if extension ext is offered. */
#define IF_EXT(s, m, always, when, ext) { #m, s(m), always, when, ext, { NO_REF, NO_REF }, ext " is offered" }
/* Required from `always` (0: never), and from `when` if feature fm (in struct fs) is supported. */
#define IF_FEATURE(s, m, always, when, fs, fm)                                                                    \
    { #m, s(m), always, when, NULL, { fs(fm), NO_REF }, #fm " is supported" }

static const struct requirement requirements[] = {
    /* Vulkan 1.0 */
    ALWAYS(R10, robustBufferAccess, V10),
    /* Vulkan 1.1 */
    ALWAYS(R11, multiview, V11),
    IF_FEATURE(R11, storageBuffer16BitAccess, V14, V11, R11, uniformAndStorageBuffer16BitAccess),
    IF_EXT(R11, shaderDrawParameters, 0, V11, "VK_KHR_shader_draw_parameters"),
    /* Vulkan 1.2 */
    ALWAYS(R12, subgroupBroadcastDynamicId, V12),
    ALWAYS(R12, imagelessFramebuffer, V12),
    ALWAYS(R12, uniformBufferStandardLayout, V12),
    ALWAYS(R12, shaderSubgroupExtendedTypes, V12),
    ALWAYS(R12, separateDepthStencilLayouts, V12),
    ALWAYS(R12, hostQueryReset, V12),
    ALWAYS(R12, timelineSemaphore, V12),
    IF_EXT(R12, samplerMirrorClampToEdge, V14, V12, "VK_KHR_sampler_mirror_clamp_to_edge"),
    IF_EXT(R12, drawIndirectCount, 0, V12, "VK_KHR_draw_indirect_count"),
    IF_FEATURE(R12, storageBuffer8BitAccess, V14, V12, R12, uniformAndStorageBuffer8BitAccess),
    { "shaderInt64", R10(shaderInt64), 0, V12, NULL, { R12(shaderSharedInt64Atomics), R12(shaderBufferInt64Atomics) },
      "shaderSharedInt64Atomics or shaderBufferInt64Atomics is supported" },
    IF_EXT(R12, descriptorIndexing, 0, V12, "VK_EXT_descriptor_indexing"),
    IF_FEATURE(R10, shaderSampledImageArrayDynamicIndexing, V14, V12, R12, descriptorIndexing),
    IF_FEATURE(R10, shaderStorageBufferArrayDynamicIndexing, V14, V12, R12, descriptorIndexing),
    IF_FEATURE(R12, shaderUniformTexelBufferArrayDynamicIndexing, V14, V12, R12, descriptorIndexing),
    IF_FEATURE(R12, shaderStorageTexelBufferArrayDynamicIndexing, V14, V12, R12, descriptorIndexing),
    IF_FEATURE(R12, shaderSampledImageArrayNonUniformIndexing, 0, V12, R12, descriptorIndexing),
    IF_FEATURE(R12, shaderStorageBufferArrayNonUniformIndexing, 0, V12, R12, descriptorIndexing),
    IF_FEATURE(R12, shaderUniformTexelBufferArrayNonUniformIndexing, 0, V12, R12, descriptorIndexing),
    IF_FEATURE(R12, descriptorBindingSampledImageUpdateAfterBind, 0, V12, R12, descriptorIndexing),
    IF_FEATURE(R12, descriptorBindingStorageImageUpdateAfterBind, 0, V12, R12, descriptorIndexing),
    IF_FEATURE(R12, descriptorBindingStorageBufferUpdateAfterBind, 0, V12, R12, descriptorIndexing),
    IF_FEATURE(R12, descriptorBindingUniformTexelBufferUpdateAfterBind, 0, V12, R12, descriptorIndexing),
    IF_FEATURE(R12, descriptorBindingStorageTexelBufferUpdateAfterBind, 0, V12, R12, descriptorIndexing),
    IF_FEATURE(R12, descriptorBindingUpdateUnusedWhilePending, 0, V12, R12, descriptorIndexing),
    IF_FEATURE(R12, descriptorBindingPartiallyBound, 0, V12, R12, descriptorIndexing),
    IF_FEATURE(R12, runtimeDescriptorArray, 0, V12, R12, descriptorIndexing),
    IF_EXT(R12, shaderOutputViewportIndex, 0, V12, "VK_EXT_shader_viewport_index_layer"),
    IF_EXT(R12, shaderOutputLayer, 0, V12, "VK_EXT_shader_viewport_index_layer"),
    IF_EXT(R12, samplerFilterMinmax, 0, V12, "VK_EXT_sampler_filter_minmax"),
    /* Vulkan 1.3 */
    ALWAYS(R13, shaderTerminateInvocation, V13),
    ALWAYS(R13, shaderDemoteToHelperInvocation, V13),
    ALWAYS(R13, privateData, V13),
    ALWAYS(R13, pipelineCreationCacheControl, V13),
    ALWAYS(R13, synchronization2, V13),
    ALWAYS(R13, shaderZeroInitializeWorkgroupMemory, V13),
    ALWAYS(R13, robustImageAccess, V13),
    ALWAYS(R13, subgroupSizeControl, V13),
    ALWAYS(R13, computeFullSubgroups, V13),
    ALWAYS(R13, dynamicRendering, V13),
    ALWAYS(R13, shaderIntegerDotProduct, V13),
    ALWAYS(R13, maintenance4, V13),
    ALWAYS(R12, vulkanMemoryModel, V13),
    ALWAYS(R12, vulkanMemoryModelDeviceScope, V13),
    ALWAYS(R13, inlineUniformBlock, V13),
    ALWAYS(R12, bufferDeviceAddress, V13),
    { "descriptorBindingInlineUniformBlockUpdateAfterBind", R13(descriptorBindingInlineUniformBlockUpdateAfterBind),
      0, V13, "VK_EXT_descriptor_indexing", { R12(descriptorIndexing), NO_REF },
      "VK_EXT_descriptor_indexing is offered or descriptorIndexing is supported" },
    /* Vulkan 1.4 */
    ALWAYS(R10, fullDrawIndexUint32, V14),
    ALWAYS(R10, imageCubeArray, V14),
    ALWAYS(R10, independentBlend, V14),
    ALWAYS(R10, sampleRateShading, V14),
    ALWAYS(R10, drawIndirectFirstInstance, V14),
    ALWAYS(R10, depthClamp, V14),
    ALWAYS(R10, depthBiasClamp, V14),
    ALWAYS(R10, samplerAnisotropy, V14),
    ALWAYS(R10, fragmentStoresAndAtomics, V14),
    ALWAYS(R10, shaderStorageImageExtendedFormats, V14),
    ALWAYS(R10, shaderUniformBufferArrayDynamicIndexing, V14),
    ALWAYS(R10, shaderStorageImageArrayDynamicIndexing, V14),
    ALWAYS(R10, shaderImageGatherExtended, V14),
    ALWAYS(R10, shaderInt16, V14),
    ALWAYS(R10, largePoints, V14),
    ALWAYS(R11, samplerYcbcrConversion, V14),
    ALWAYS(R11, variablePointers, V14),
    ALWAYS(R11, variablePointersStorageBuffer, V14),
    ALWAYS(R12, scalarBlockLayout, V14),
    ALWAYS(R12, shaderInt8, V14),
    ALWAYS(R14, globalPriorityQuery, V14),
    ALWAYS(R14, shaderSubgroupRotate, V14),
    ALWAYS(R14, shaderSubgroupRotateClustered, V14),
    ALWAYS(R14, shaderFloatControls2, V14),
    ALWAYS(R14, shaderExpectAssume, V14),
    ALWAYS(R14, bresenhamLines, V14),
    ALWAYS(R14, vertexAttributeInstanceRateDivisor, V14),
    ALWAYS(R14, indexTypeUint8, V14),
    ALWAYS(R14, maintenance5, V14),
    ALWAYS(R14, pushDescriptor, V14),
    ALWAYS(R14, dynamicRenderingLocalRead, V14),
    ALWAYS(R14, maintenance6, V14),
    ALWAYS(R14, pipelineRobustness, V14),
    IF_FEATURE(R14, pipelineProtectedAccess, 0, V14, R11, protectedMemory),
};

static const VkBool32 *feature_at(const struct ct_features *f, struct ref r)
{
    const char *base;
    switch (r.where) {
    case F10:
        base = (const char *)&f->f2.features;
        break;
    case F11:
        base = (const char *)&f->v11;
        break;
    case F12:
        base = (const char *)&f->v12;
        break;
    case F13:
        base = (const char *)&f->v13;
        break;
    case F14:
        base = (const char *)&f->v14;
        break;
    default:
        return NULL;
    }
    return (const VkBool32 *)(base + r.offset);
}

/* The version whose feature query includes the structure. */
static uint32_t struct_version(enum where w)
{
    switch (w) {
    case F10:
        return VK_API_VERSION_1_0;
    case F11: /* VkPhysicalDeviceVulkan11Features arrived with 1.2 */
    case F12:
        return VK_API_VERSION_1_2;
    case F13:
        return VK_API_VERSION_1_3;
    default:
        return VK_API_VERSION_1_4;
    }
}

static int supported(const struct ct_env *e, const struct ct_features *f, struct ref r)
{
    const VkBool32 *b;
    if (r.where == F_NONE || e->version < struct_version(r.where))
        return 0;
    b = feature_at(f, r);
    return b && *b == VK_TRUE;
}

static void check_required_feature(struct ct *t)
{
    const struct requirement *req = ct_param(t);
    struct ct_env e;
    struct ct_features a, b;
    const VkBool32 *va, *vb;
    int required;

    ct_env_init(t, &e);
    if (!e.graphics)
        ct_skip(t, "no graphics queue family: these are the requirements of graphics implementations");
    ct_get_features(&e, &a, 0x00);
    ct_get_features(&e, &b, 0xFF);
    required = req->always && e.version >= req->always;
    if (!required && req->when && e.version >= req->when)
        required = (req->when_ext && ct_device_extension_supported(e.pd, req->when_ext)) ||
                   supported(&e, &a, req->when_feature[0]) || supported(&e, &a, req->when_feature[1]);
    if (!required) {
        if (req->when && e.version >= req->when)
            ct_skip(t, "at Vulkan %u.%u required only if %s", VK_API_VERSION_MAJOR(e.version),
                    VK_API_VERSION_MINOR(e.version), req->when_text);
        ct_skip(t, "not required at Vulkan %u.%u", VK_API_VERSION_MAJOR(e.version), VK_API_VERSION_MINOR(e.version));
    }
    if (e.version < struct_version(req->feature.where))
        ct_skip(t, "the feature's structure needs Vulkan %u.%u",
                VK_API_VERSION_MAJOR(struct_version(req->feature.where)),
                VK_API_VERSION_MINOR(struct_version(req->feature.where)));
    va = feature_at(&a, req->feature);
    vb = feature_at(&b, req->feature);
    CT_CHECKF(t, *va == *vb, "%s is not written by vkGetPhysicalDeviceFeatures2 (pre-filled 0x00: %u, 0xFF: %u)",
              req->name, *va, *vb);
    CT_CHECKF(t, *va == VK_TRUE, "%s is %u: not supported", req->name, *va);
    ct_env_fini(&e);
}

CT_TEST(hostImageCopy_single_family,
        "Features > Feature Requirements: if Vulkan 1.4 is supported, a queue family supports graphics, and no "
        "additional queue families support VK_QUEUE_TRANSFER_BIT, the hostImageCopy feature must be supported",
        "vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceQueueFamilyProperties")
{
    struct ct_env e;
    struct ct_features f;
    VkQueueFamilyProperties fam[64];
    uint32_t count = 64, i;
    int graphics = 0, other_transfer = 0;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_4)
        ct_skip(t, "the rule applies to Vulkan 1.4");
    vkGetPhysicalDeviceQueueFamilyProperties(e.pd, &count, fam);
    for (i = 0; i < count; i++) {
        if (fam[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            graphics = 1;
        else if (fam[i].queueFlags & VK_QUEUE_TRANSFER_BIT)
            other_transfer = 1;
    }
    if (!graphics || other_transfer)
        ct_skip(t, "a separate transfer queue family exists (or no graphics family)");
    ct_get_features(&e, &f, 0);
    CT_CHECKF(t, f.v14.hostImageCopy == VK_TRUE, "hostImageCopy is %u", f.v14.hostImageCopy);
    ct_env_fini(&e);
}

static const struct ct_test *const fixed_tests[] = {
    CT_REF(hostImageCopy_single_family),
};

void ct_suite_features(struct ct_registry *r);
void ct_suite_features(struct ct_registry *r)
{
    static struct ct_test tests[CT_ARRAY_SIZE(requirements)];
    static const struct ct_test *ptrs[CT_ARRAY_SIZE(requirements)];
    static char specs[CT_ARRAY_SIZE(requirements)][320];
    size_t i;

    for (i = 0; i < CT_ARRAY_SIZE(requirements); i++) {
        const struct requirement *q = &requirements[i];
        int n = snprintf(specs[i], sizeof(specs[i]), "Features > Feature Requirements: Vulkan graphics "
                         "implementations must support %s", q->name);
        if (q->always && n > 0 && (size_t)n < sizeof(specs[i]))
            n += snprintf(specs[i] + n, sizeof(specs[i]) - (size_t)n, " from Vulkan %u.%u",
                          VK_API_VERSION_MAJOR(q->always), VK_API_VERSION_MINOR(q->always));
        if (q->when && n > 0 && (size_t)n < sizeof(specs[i]))
            snprintf(specs[i] + n, sizeof(specs[i]) - (size_t)n, "%s from Vulkan %u.%u if %s", q->always ? ", and" : "",
                     VK_API_VERSION_MAJOR(q->when), VK_API_VERSION_MINOR(q->when), q->when_text);
        tests[i].name = q->name;
        tests[i].fn = check_required_feature;
        tests[i].spec = specs[i];
        tests[i].commands = "vkGetPhysicalDeviceFeatures2";
        tests[i].param = q;
        ptrs[i] = &tests[i];
    }
    ct_register(r, "features", ptrs, CT_ARRAY_SIZE(ptrs));
    ct_register(r, "features", fixed_tests, CT_ARRAY_SIZE(fixed_tests));
}
