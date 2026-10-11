/* Chapter "Devices and Queues": physical devices and their properties,
 * queue families, logical devices and queues. Feature queries live here too
 * (vkGetPhysicalDeviceFeatures2 consistency); the features Vulkan requires
 * are checked in features.c. */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

#define FILL_A 0x00
#define FILL_B 0xFF

/* a.f == b.f for two structs; reports the member. */
#define EQ(a, b, f)                                                                                               \
    CT_CHECKF(t, (a).f == (b).f, "%s: %llu vs %llu (differs between the structures, or not written)", #f,        \
              (unsigned long long)(a).f, (unsigned long long)(b).f)
#define EQ2(x, y)                                                                                                 \
    CT_CHECKF(t, (x) == (y), "%s vs %s: %llu vs %llu (differs, or not written)", #x, #y, (unsigned long long)(x), \
              (unsigned long long)(y))
#define EQ_MEM(a, b, f) CT_CHECKF(t, memcmp((a).f, (b).f, sizeof((a).f)) == 0, "%s differs, or not written", #f)

static int is_bool(VkBool32 b)
{
    return b == VK_TRUE || b == VK_FALSE;
}

static void check_bools(struct ct *t, const void *first, size_t count, const char *what)
{
    const VkBool32 *b = first;
    size_t i;
    for (i = 0; i < count; i++)
        CT_CHECKF(t, is_bool(b[i]), "%s member %zu is %u, neither VK_TRUE nor VK_FALSE (or not written)", what, i,
                  b[i]);
}

static void require_version(struct ct *t, const struct ct_env *e, uint32_t version, const char *what)
{
    if (e->version < version)
        ct_skip(t, "%s needs Vulkan %u.%u; the device or instance offers %u.%u", what, VK_API_VERSION_MAJOR(version),
                VK_API_VERSION_MINOR(version), VK_API_VERSION_MAJOR(e->version), VK_API_VERSION_MINOR(e->version));
}

/* ==========================================================================
 * Physical devices
 * ========================================================================== */

static VkResult enum_physical_devices(void *ctx, uint32_t *count, void *items)
{
    return vkEnumeratePhysicalDevices(*(VkInstance *)ctx, count, items);
}

static int same_handle(const void *a, const void *b)
{
    return *(const VkPhysicalDevice *)a == *(const VkPhysicalDevice *)b;
}

CT_TEST(enumerate_physical_devices,
        "Devices and Queues > Physical Devices: vkEnumeratePhysicalDevices returns the count, then that many "
        "distinct handles; at least one physical device exists, since some queue family of some physical device "
        "must support graphics, compute or video (VkQueueFlagBits)",
        "vkEnumeratePhysicalDevices")
{
    struct ct_env e;
    VkPhysicalDevice pds[16], again[16];
    uint32_t count = 0, n, i, j;

    ct_env_init(t, &e);
    CT_REQUIRE_VK(t, vkEnumeratePhysicalDevices(e.instance, &count, NULL));
    CT_REQUIRE(t, count >= 1);
    n = CT_MIN(count, 16);
    CT_CHECK_VK(t, vkEnumeratePhysicalDevices(e.instance, &n, pds), count <= 16 ? VK_SUCCESS : VK_INCOMPLETE);
    for (i = 0; i < n; i++) {
        CT_CHECKF(t, pds[i] != VK_NULL_HANDLE, "handle %u is NULL", i);
        for (j = 0; j < i; j++)
            CT_CHECKF(t, pds[i] != pds[j], "handles %u and %u are equal", i, j);
    }
    n = CT_MIN(count, 16);
    vkEnumeratePhysicalDevices(e.instance, &n, again);
    CT_CHECKF(t, memcmp(pds, again, n * sizeof(pds[0])) == 0, "two enumerations returned different handles");
    ct_env_fini(&e);
}

CT_TEST(enumerate_physical_devices_incomplete,
        "Devices and Queues > Physical Devices: if pPhysicalDeviceCount is less than the number available, at most "
        "that many handles are written and VK_INCOMPLETE is returned",
        "vkEnumeratePhysicalDevices")
{
    struct ct_env e;
    ct_env_init(t, &e);
    ct_check_incomplete(t, enum_physical_devices, &e.instance, sizeof(VkPhysicalDevice), same_handle,
                        "vkEnumeratePhysicalDevices");
    ct_env_fini(&e);
}

CT_TEST(enumerate_physical_device_groups,
        "Devices and Queues > Devices: vkEnumeratePhysicalDeviceGroups puts every physical device in exactly one "
        "group; a group of one device has subsetAllocation VK_FALSE; VK_INCOMPLETE protocol",
        "vkEnumeratePhysicalDeviceGroups vkEnumeratePhysicalDevices")
{
    struct ct_env e;
    VkPhysicalDeviceGroupProperties groups[8];
    VkPhysicalDevice pds[16];
    uint32_t count = 0, n, pd_count = 16, i, j, k;
    VkResult r;

    ct_env_init(t, &e);
    if (e.instance_version < VK_API_VERSION_1_1 || e.app_version < VK_API_VERSION_1_1)
        ct_skip(t, "instance version below 1.1");
    CT_REQUIRE_VK(t, vkEnumeratePhysicalDevices(e.instance, &pd_count, pds));
    CT_REQUIRE_VK(t, vkEnumeratePhysicalDeviceGroups(e.instance, &count, NULL));
    CT_REQUIRE(t, count >= 1 && count <= 8);
    for (i = 0; i < 8; i++) {
        memset(&groups[i], 0xA5, sizeof(groups[i]));
        groups[i].sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES;
        groups[i].pNext = NULL;
    }
    n = count;
    CT_REQUIRE_VK(t, vkEnumeratePhysicalDeviceGroups(e.instance, &n, groups));
    CT_CHECK(t, n == count);
    for (k = 0; k < pd_count; k++) {
        uint32_t seen = 0;
        for (i = 0; i < n; i++)
            for (j = 0; j < groups[i].physicalDeviceCount && j < VK_MAX_DEVICE_GROUP_SIZE; j++)
                seen += groups[i].physicalDevices[j] == pds[k];
        CT_CHECKF(t, seen == 1, "physical device %u is in %u groups", k, seen);
    }
    for (i = 0; i < n; i++) {
        CT_CHECKF(t, groups[i].physicalDeviceCount >= 1 && groups[i].physicalDeviceCount <= VK_MAX_DEVICE_GROUP_SIZE,
                  "group %u has %u devices", i, groups[i].physicalDeviceCount);
        CT_CHECKF(t, is_bool(groups[i].subsetAllocation), "group %u: subsetAllocation not a VkBool32", i);
        if (groups[i].physicalDeviceCount == 1)
            CT_CHECKF(t, groups[i].subsetAllocation == VK_FALSE, "group %u of one device has subsetAllocation", i);
        CT_CHECK(t, groups[i].sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES && !groups[i].pNext);
    }
    /* VK_INCOMPLETE */
    n = count - 1;
    r = vkEnumeratePhysicalDeviceGroups(e.instance, &n, groups);
    CT_CHECKF(t, r == VK_INCOMPLETE, "room for %u of %u groups: %s", count - 1, count, ct_result_name(r));
    CT_CHECK(t, n == count - 1);
    ct_env_fini(&e);
}

/* Lead bytes and continuation bytes line up (no overlong/surrogate checks). */
static int utf8_valid(const char *str)
{
    const unsigned char *s = (const unsigned char *)str;
    while (*s) {
        unsigned extra = *s < 0x80 ? 0 : (*s & 0xE0) == 0xC0 ? 1 : (*s & 0xF0) == 0xE0 ? 2 : (*s & 0xF8) == 0xF0 ? 3 : 4;
        unsigned k;
        if (extra == 4)
            return 0;
        for (k = 1; k <= extra; k++)
            if ((s[k] & 0xC0) != 0x80) /* also stops at the NUL */
                return 0;
        s += 1 + extra;
    }
    return 1;
}

static int known_vendor(uint32_t id)
{
    if (id >= 0x10000)
        return id >= VK_VENDOR_ID_KHRONOS && id <= VK_VENDOR_ID_MOBILEYE;
    return id != 0 && id != 0xFFFF; /* a PCI vendor ID */
}

CT_TEST(physical_device_properties,
        "Devices and Queues > Physical Devices: VkPhysicalDeviceProperties: apiVersion of the Vulkan variant and "
        "at least 1.0; vendorID a PCI vendor ID (upper bits zero) or a Khronos vendor ID; deviceType a "
        "VkPhysicalDeviceType; deviceName a NUL-terminated UTF-8 name",
        "vkGetPhysicalDeviceProperties")
{
    struct ct_env e;
    VkPhysicalDeviceProperties p;
    size_t len;

    ct_env_init(t, &e);
    memset(&p, 0xA5, sizeof(p));
    vkGetPhysicalDeviceProperties(e.pd, &p);
    CT_CHECKF(t, VK_API_VERSION_VARIANT(p.apiVersion) == 0, "apiVersion variant %u", VK_API_VERSION_VARIANT(p.apiVersion));
    CT_CHECKF(t, CT_VERSION(p.apiVersion) >= VK_API_VERSION_1_0 && VK_API_VERSION_MAJOR(p.apiVersion) == 1,
              "apiVersion %u.%u", VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion));
    CT_CHECKF(t, known_vendor(p.vendorID), "vendorID 0x%x is neither a PCI vendor ID nor a Khronos vendor ID",
              p.vendorID);
    CT_CHECKF(t, p.deviceType >= VK_PHYSICAL_DEVICE_TYPE_OTHER && p.deviceType <= VK_PHYSICAL_DEVICE_TYPE_CPU,
              "deviceType %d", (int)p.deviceType);
    CT_REQUIREF(t, memchr(p.deviceName, '\0', VK_MAX_PHYSICAL_DEVICE_NAME_SIZE) != NULL,
                "deviceName is not NUL-terminated");
    len = strlen(p.deviceName);
    CT_CHECKF(t, len > 0, "deviceName is empty");
    CT_CHECKF(t, utf8_valid(p.deviceName), "deviceName \"%s\" is not well-formed UTF-8", p.deviceName);
    ct_env_fini(&e);
}

CT_TEST(physical_device_properties_stable,
        "Devices and Queues > Physical Devices: properties describe the physical device: the same device "
        "reports the same VkPhysicalDeviceProperties from another instance (identifiers, pipelineCacheUUID)",
        "vkGetPhysicalDeviceProperties")
{
    struct ct_env a, b;
    ct_env_init(t, &a);
    ct_env_init(t, &b);
    EQ(a.props, b.props, apiVersion);
    EQ(a.props, b.props, driverVersion);
    EQ(a.props, b.props, vendorID);
    EQ(a.props, b.props, deviceID);
    EQ(a.props, b.props, deviceType);
    CT_CHECK(t, strncmp(a.props.deviceName, b.props.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE) == 0);
    EQ_MEM(a.props, b.props, pipelineCacheUUID);
    ct_env_fini(&b);
    ct_env_fini(&a);
}

CT_TEST(properties2_core,
        "Devices and Queues > Physical Devices: VkPhysicalDeviceProperties2::properties is written with the same "
        "values as vkGetPhysicalDeviceProperties writes",
        "vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties")
{
    struct ct_env e;
    VkPhysicalDeviceProperties p;
    VkPhysicalDeviceProperties2 p2;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_1, "vkGetPhysicalDeviceProperties2");
    memset(&p, 0, sizeof(p));
    memset(&p2, 0, sizeof(p2));
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkGetPhysicalDeviceProperties(e.pd, &p);
    vkGetPhysicalDeviceProperties2(e.pd, &p2);
    CT_CHECK(t, memcmp(&p, &p2.properties, sizeof(p)) == 0);
    CT_CHECK(t, p2.sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 && p2.pNext == NULL);
    ct_env_fini(&e);
}

/* Queries one property structure on its own, pre-filled with `fill`. */
static void query_property(const struct ct_env *e, void *s, size_t size, VkStructureType type, unsigned char fill)
{
    VkPhysicalDeviceProperties2 p2;
    VkBaseOutStructure *base = s;
    memset(s, fill, size);
    base->sType = type;
    base->pNext = NULL;
    memset(&p2, 0, sizeof(p2));
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = s;
    vkGetPhysicalDeviceProperties2(e->pd, &p2);
}

#define QUERY(e, s, type, fill) query_property(e, &(s), sizeof(s), type, fill)

CT_TEST(properties_vulkan11,
        "Devices and Queues > VkPhysicalDeviceVulkan11Properties: its members have the same values as the "
        "corresponding members of VkPhysicalDeviceIDProperties, SubgroupProperties, PointClippingProperties, "
        "MultiviewProperties, ProtectedMemoryProperties and Maintenance3Properties; deviceNodeMask has one bit "
        "when deviceLUIDValid",
        "vkGetPhysicalDeviceProperties2")
{
    struct ct_env e;
    struct ct_properties p;
    VkPhysicalDeviceIDProperties id;
    VkPhysicalDeviceSubgroupProperties sub;
    VkPhysicalDevicePointClippingProperties clip;
    VkPhysicalDeviceMultiviewProperties mv;
    VkPhysicalDeviceProtectedMemoryProperties prot;
    VkPhysicalDeviceMaintenance3Properties m3;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_2, "VkPhysicalDeviceVulkan11Properties");
    ct_get_properties(&e, &p, FILL_A);
    QUERY(&e, id, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES, FILL_B);
    QUERY(&e, sub, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, FILL_B);
    QUERY(&e, clip, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES, FILL_B);
    QUERY(&e, mv, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES, FILL_B);
    QUERY(&e, prot, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_PROPERTIES, FILL_B);
    QUERY(&e, m3, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES, FILL_B);

    EQ_MEM(p.v11, id, deviceUUID);
    EQ_MEM(p.v11, id, driverUUID);
    EQ(p.v11, id, deviceLUIDValid);
    if (p.v11.deviceLUIDValid) {
        EQ_MEM(p.v11, id, deviceLUID);
        EQ(p.v11, id, deviceNodeMask);
        CT_CHECKF(t, ct_is_pow2(id.deviceNodeMask), "deviceLUIDValid but deviceNodeMask 0x%x has not exactly one bit",
                  id.deviceNodeMask);
    }
    CT_CHECK(t, is_bool(id.deviceLUIDValid));
    EQ2(p.v11.subgroupSize, sub.subgroupSize);
    EQ2(p.v11.subgroupSupportedStages, sub.supportedStages);
    EQ2(p.v11.subgroupSupportedOperations, sub.supportedOperations);
    EQ2(p.v11.subgroupQuadOperationsInAllStages, sub.quadOperationsInAllStages);
    EQ(p.v11, clip, pointClippingBehavior);
    CT_CHECKF(t, clip.pointClippingBehavior == VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES ||
                     clip.pointClippingBehavior == VK_POINT_CLIPPING_BEHAVIOR_USER_CLIP_PLANES_ONLY,
              "pointClippingBehavior %d", (int)clip.pointClippingBehavior);
    EQ(p.v11, mv, maxMultiviewViewCount);
    EQ(p.v11, mv, maxMultiviewInstanceIndex);
    EQ(p.v11, prot, protectedNoFault);
    EQ(p.v11, m3, maxPerSetDescriptors);
    EQ(p.v11, m3, maxMemoryAllocationSize);
    ct_env_fini(&e);
}

CT_TEST(properties_vulkan12,
        "Devices and Queues > VkPhysicalDeviceVulkan12Properties: its members have the same values as the "
        "corresponding members of VkPhysicalDeviceDriverProperties, FloatControlsProperties, "
        "DescriptorIndexingProperties, DepthStencilResolveProperties, SamplerFilterMinmaxProperties and "
        "TimelineSemaphoreProperties; driverName and driverInfo are NUL-terminated",
        "vkGetPhysicalDeviceProperties2")
{
    struct ct_env e;
    struct ct_properties p, q;
    VkPhysicalDeviceDriverProperties drv;
    VkPhysicalDeviceFloatControlsProperties fc;
    VkPhysicalDeviceDescriptorIndexingProperties di;
    VkPhysicalDeviceDepthStencilResolveProperties dsr;
    VkPhysicalDeviceSamplerFilterMinmaxProperties mm;
    VkPhysicalDeviceTimelineSemaphoreProperties ts;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_2, "VkPhysicalDeviceVulkan12Properties");
    ct_get_properties(&e, &p, FILL_A);
    ct_get_properties(&e, &q, FILL_B);
    QUERY(&e, drv, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, FILL_B);
    QUERY(&e, fc, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES, FILL_B);
    QUERY(&e, di, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES, FILL_B);
    QUERY(&e, dsr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES, FILL_B);
    QUERY(&e, mm, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES, FILL_B);
    QUERY(&e, ts, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES, FILL_B);

    EQ(p.v12, drv, driverID);
    CT_CHECK(t, memchr(drv.driverName, '\0', VK_MAX_DRIVER_NAME_SIZE) != NULL);
    CT_CHECK(t, memchr(drv.driverInfo, '\0', VK_MAX_DRIVER_INFO_SIZE) != NULL);
    CT_CHECK(t, strncmp(p.v12.driverName, drv.driverName, VK_MAX_DRIVER_NAME_SIZE) == 0);
    CT_CHECK(t, strncmp(p.v12.driverInfo, drv.driverInfo, VK_MAX_DRIVER_INFO_SIZE) == 0);
    EQ(p.v12.conformanceVersion, drv.conformanceVersion, major);
    EQ(p.v12.conformanceVersion, drv.conformanceVersion, minor);
    EQ(p.v12.conformanceVersion, drv.conformanceVersion, subminor);
    EQ(p.v12.conformanceVersion, drv.conformanceVersion, patch);

    EQ(p.v12, fc, denormBehaviorIndependence);
    EQ(p.v12, fc, roundingModeIndependence);
    EQ(p.v12, fc, shaderSignedZeroInfNanPreserveFloat16);
    EQ(p.v12, fc, shaderSignedZeroInfNanPreserveFloat32);
    EQ(p.v12, fc, shaderSignedZeroInfNanPreserveFloat64);
    EQ(p.v12, fc, shaderDenormPreserveFloat16);
    EQ(p.v12, fc, shaderDenormPreserveFloat32);
    EQ(p.v12, fc, shaderDenormPreserveFloat64);
    EQ(p.v12, fc, shaderDenormFlushToZeroFloat16);
    EQ(p.v12, fc, shaderDenormFlushToZeroFloat32);
    EQ(p.v12, fc, shaderDenormFlushToZeroFloat64);
    EQ(p.v12, fc, shaderRoundingModeRTEFloat16);
    EQ(p.v12, fc, shaderRoundingModeRTEFloat32);
    EQ(p.v12, fc, shaderRoundingModeRTEFloat64);
    EQ(p.v12, fc, shaderRoundingModeRTZFloat16);
    EQ(p.v12, fc, shaderRoundingModeRTZFloat32);
    EQ(p.v12, fc, shaderRoundingModeRTZFloat64);

    EQ(p.v12, di, maxUpdateAfterBindDescriptorsInAllPools);
    EQ(p.v12, di, shaderUniformBufferArrayNonUniformIndexingNative);
    EQ(p.v12, di, shaderSampledImageArrayNonUniformIndexingNative);
    EQ(p.v12, di, shaderStorageBufferArrayNonUniformIndexingNative);
    EQ(p.v12, di, shaderStorageImageArrayNonUniformIndexingNative);
    EQ(p.v12, di, shaderInputAttachmentArrayNonUniformIndexingNative);
    EQ(p.v12, di, robustBufferAccessUpdateAfterBind);
    EQ(p.v12, di, quadDivergentImplicitLod);
    EQ(p.v12, di, maxPerStageDescriptorUpdateAfterBindSamplers);
    EQ(p.v12, di, maxPerStageDescriptorUpdateAfterBindUniformBuffers);
    EQ(p.v12, di, maxPerStageDescriptorUpdateAfterBindStorageBuffers);
    EQ(p.v12, di, maxPerStageDescriptorUpdateAfterBindSampledImages);
    EQ(p.v12, di, maxPerStageDescriptorUpdateAfterBindStorageImages);
    EQ(p.v12, di, maxPerStageDescriptorUpdateAfterBindInputAttachments);
    EQ(p.v12, di, maxPerStageUpdateAfterBindResources);
    EQ(p.v12, di, maxDescriptorSetUpdateAfterBindSamplers);
    EQ(p.v12, di, maxDescriptorSetUpdateAfterBindUniformBuffers);
    EQ(p.v12, di, maxDescriptorSetUpdateAfterBindUniformBuffersDynamic);
    EQ(p.v12, di, maxDescriptorSetUpdateAfterBindStorageBuffers);
    EQ(p.v12, di, maxDescriptorSetUpdateAfterBindStorageBuffersDynamic);
    EQ(p.v12, di, maxDescriptorSetUpdateAfterBindSampledImages);
    EQ(p.v12, di, maxDescriptorSetUpdateAfterBindStorageImages);
    EQ(p.v12, di, maxDescriptorSetUpdateAfterBindInputAttachments);

    EQ(p.v12, dsr, supportedDepthResolveModes);
    EQ(p.v12, dsr, supportedStencilResolveModes);
    EQ(p.v12, dsr, independentResolveNone);
    EQ(p.v12, dsr, independentResolve);
    EQ(p.v12, mm, filterMinmaxSingleComponentFormats);
    EQ(p.v12, mm, filterMinmaxImageComponentMapping);
    EQ(p.v12, ts, maxTimelineSemaphoreValueDifference);
    EQ(p.v12, q.v12, framebufferIntegerColorSampleCounts);
    ct_env_fini(&e);
}

CT_TEST(properties_vulkan13,
        "Devices and Queues > VkPhysicalDeviceVulkan13Properties: its members have the same values as the "
        "corresponding members of VkPhysicalDeviceSubgroupSizeControlProperties, InlineUniformBlockProperties, "
        "ShaderIntegerDotProductProperties, TexelBufferAlignmentProperties and Maintenance4Properties",
        "vkGetPhysicalDeviceProperties2")
{
    struct ct_env e;
    struct ct_properties p, q;
    VkPhysicalDeviceSubgroupSizeControlProperties ssc;
    VkPhysicalDeviceInlineUniformBlockProperties iub;
    VkPhysicalDeviceShaderIntegerDotProductProperties dot;
    VkPhysicalDeviceTexelBufferAlignmentProperties tba;
    VkPhysicalDeviceMaintenance4Properties m4;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_3, "VkPhysicalDeviceVulkan13Properties");
    ct_get_properties(&e, &p, FILL_A);
    ct_get_properties(&e, &q, FILL_B);
    QUERY(&e, ssc, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES, FILL_B);
    QUERY(&e, iub, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES, FILL_B);
    QUERY(&e, dot, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES, FILL_B);
    QUERY(&e, tba, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES, FILL_B);
    QUERY(&e, m4, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES, FILL_B);

    EQ(p.v13, ssc, minSubgroupSize);
    EQ(p.v13, ssc, maxSubgroupSize);
    EQ(p.v13, ssc, maxComputeWorkgroupSubgroups);
    EQ(p.v13, ssc, requiredSubgroupSizeStages);
    EQ(p.v13, iub, maxInlineUniformBlockSize);
    EQ(p.v13, iub, maxPerStageDescriptorInlineUniformBlocks);
    EQ(p.v13, iub, maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks);
    EQ(p.v13, iub, maxDescriptorSetInlineUniformBlocks);
    EQ(p.v13, iub, maxDescriptorSetUpdateAfterBindInlineUniformBlocks);
    EQ(p.v13, q.v13, maxInlineUniformTotalSize);
    EQ(p.v13, dot, integerDotProduct8BitUnsignedAccelerated);
    EQ(p.v13, dot, integerDotProduct8BitSignedAccelerated);
    EQ(p.v13, dot, integerDotProduct8BitMixedSignednessAccelerated);
    EQ(p.v13, dot, integerDotProduct4x8BitPackedUnsignedAccelerated);
    EQ(p.v13, dot, integerDotProduct4x8BitPackedSignedAccelerated);
    EQ(p.v13, dot, integerDotProduct4x8BitPackedMixedSignednessAccelerated);
    EQ(p.v13, dot, integerDotProduct16BitUnsignedAccelerated);
    EQ(p.v13, dot, integerDotProduct16BitSignedAccelerated);
    EQ(p.v13, dot, integerDotProduct16BitMixedSignednessAccelerated);
    EQ(p.v13, dot, integerDotProduct32BitUnsignedAccelerated);
    EQ(p.v13, dot, integerDotProduct32BitSignedAccelerated);
    EQ(p.v13, dot, integerDotProduct32BitMixedSignednessAccelerated);
    EQ(p.v13, dot, integerDotProduct64BitUnsignedAccelerated);
    EQ(p.v13, dot, integerDotProduct64BitSignedAccelerated);
    EQ(p.v13, dot, integerDotProduct64BitMixedSignednessAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating8BitUnsignedAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating8BitSignedAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating16BitUnsignedAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating16BitSignedAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating32BitUnsignedAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating32BitSignedAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating64BitUnsignedAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating64BitSignedAccelerated);
    EQ(p.v13, dot, integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated);
    EQ(p.v13, tba, storageTexelBufferOffsetAlignmentBytes);
    EQ(p.v13, tba, storageTexelBufferOffsetSingleTexelAlignment);
    EQ(p.v13, tba, uniformTexelBufferOffsetAlignmentBytes);
    EQ(p.v13, tba, uniformTexelBufferOffsetSingleTexelAlignment);
    EQ(p.v13, m4, maxBufferSize);
    ct_env_fini(&e);
}

CT_TEST(properties_vulkan14,
        "Devices and Queues > VkPhysicalDeviceVulkan14Properties: its members have the same values as the "
        "corresponding members of VkPhysicalDeviceLineRasterizationProperties, VertexAttributeDivisorProperties, "
        "PushDescriptorProperties, Maintenance5Properties, Maintenance6Properties, PipelineRobustnessProperties "
        "and HostImageCopyProperties",
        "vkGetPhysicalDeviceProperties2")
{
    struct ct_env e;
    struct ct_properties p, q;
    VkPhysicalDeviceLineRasterizationProperties lr;
    VkPhysicalDeviceVertexAttributeDivisorProperties vad;
    VkPhysicalDevicePushDescriptorProperties pd;
    VkPhysicalDeviceMaintenance5Properties m5;
    VkPhysicalDeviceMaintenance6Properties m6;
    VkPhysicalDevicePipelineRobustnessProperties pr;
    VkPhysicalDeviceHostImageCopyProperties hic;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_4, "VkPhysicalDeviceVulkan14Properties");
    ct_get_properties(&e, &p, FILL_A);
    ct_get_properties(&e, &q, FILL_B);
    QUERY(&e, lr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES, FILL_B);
    QUERY(&e, vad, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES, FILL_B);
    QUERY(&e, pd, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES, FILL_B);
    QUERY(&e, m5, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_PROPERTIES, FILL_B);
    QUERY(&e, m6, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_PROPERTIES, FILL_B);
    QUERY(&e, pr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_PROPERTIES, FILL_B);
    memset(&hic, FILL_B, sizeof(hic));
    hic.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES;
    hic.pNext = NULL;
    hic.copySrcLayoutCount = 0;
    hic.pCopySrcLayouts = NULL;
    hic.copyDstLayoutCount = 0;
    hic.pCopyDstLayouts = NULL;
    {
        VkPhysicalDeviceProperties2 p2;
        memset(&p2, 0, sizeof(p2));
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &hic;
        vkGetPhysicalDeviceProperties2(e.pd, &p2);
    }

    EQ(p.v14, lr, lineSubPixelPrecisionBits);
    EQ(p.v14, vad, maxVertexAttribDivisor);
    EQ(p.v14, vad, supportsNonZeroFirstInstance);
    EQ(p.v14, pd, maxPushDescriptors);
    EQ(p.v14, q.v14, dynamicRenderingLocalReadDepthStencilAttachments);
    EQ(p.v14, q.v14, dynamicRenderingLocalReadMultisampledAttachments);
    EQ(p.v14, m5, earlyFragmentMultisampleCoverageAfterSampleCounting);
    EQ(p.v14, m5, earlyFragmentSampleMaskTestBeforeSampleCounting);
    EQ(p.v14, m5, depthStencilSwizzleOneSupport);
    EQ(p.v14, m5, polygonModePointSize);
    EQ(p.v14, m5, nonStrictSinglePixelWideLinesUseParallelogram);
    EQ(p.v14, m5, nonStrictWideLinesUseParallelogram);
    EQ(p.v14, m6, blockTexelViewCompatibleMultipleLayers);
    EQ(p.v14, m6, maxCombinedImageSamplerDescriptorCount);
    EQ(p.v14, m6, fragmentShadingRateClampCombinerInputs);
    EQ(p.v14, pr, defaultRobustnessStorageBuffers);
    EQ(p.v14, pr, defaultRobustnessUniformBuffers);
    EQ(p.v14, pr, defaultRobustnessVertexInputs);
    EQ(p.v14, pr, defaultRobustnessImages);
    EQ(p.v14, hic, copySrcLayoutCount);
    EQ(p.v14, hic, copyDstLayoutCount);
    EQ_MEM(p.v14, hic, optimalTilingLayoutUUID);
    EQ(p.v14, hic, identicalMemoryTypeRequirements);
    CT_CHECK(t, p.v14.pCopySrcLayouts == NULL && p.v14.pCopyDstLayouts == NULL);
    ct_env_fini(&e);
}

/* ==========================================================================
 * Features (consistency; requirements are in features.c)
 * ========================================================================== */

static void query_feature(const struct ct_env *e, void *s, size_t size, VkStructureType type, unsigned char fill)
{
    VkPhysicalDeviceFeatures2 f2;
    VkBaseOutStructure *base = s;
    memset(s, fill, size);
    base->sType = type;
    base->pNext = NULL;
    memset(&f2, 0, sizeof(f2));
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = s;
    vkGetPhysicalDeviceFeatures2(e->pd, &f2);
}

#define QUERY_F(e, s, type, fill) query_feature(e, &(s), sizeof(s), type, fill)

/* VkBool32 members of a VkPhysicalDeviceVulkan1xFeatures (sizeof would count
 * the tail padding). */
#define FIRST_BOOL offsetof(VkPhysicalDeviceVulkan11Features, storageBuffer16BitAccess)
#define BOOL_COUNT(type, last) ((offsetof(type, last) - FIRST_BOOL) / sizeof(VkBool32) + 1)
#define V11_BOOLS BOOL_COUNT(VkPhysicalDeviceVulkan11Features, shaderDrawParameters)
#define V12_BOOLS BOOL_COUNT(VkPhysicalDeviceVulkan12Features, subgroupBroadcastDynamicId)
#define V13_BOOLS BOOL_COUNT(VkPhysicalDeviceVulkan13Features, maintenance4)
#define V14_BOOLS BOOL_COUNT(VkPhysicalDeviceVulkan14Features, pushDescriptor)
#define BOOLS(s, n) check_bools(t, (const char *)&(s) + FIRST_BOOL, n, #s)

CT_TEST(features2_core,
        "Features: vkGetPhysicalDeviceFeatures2 reports VkPhysicalDeviceFeatures as vkGetPhysicalDeviceFeatures "
        "does; each member is VK_TRUE or VK_FALSE",
        "vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures2")
{
    struct ct_env e;
    VkPhysicalDeviceFeatures f;
    VkPhysicalDeviceFeatures2 f2;

    ct_env_init(t, &e);
    memset(&f, FILL_B, sizeof(f));
    vkGetPhysicalDeviceFeatures(e.pd, &f);
    check_bools(t, &f, sizeof(f) / sizeof(VkBool32), "VkPhysicalDeviceFeatures");
    require_version(t, &e, VK_API_VERSION_1_1, "vkGetPhysicalDeviceFeatures2");
    memset(&f2, FILL_A, sizeof(f2));
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = NULL;
    vkGetPhysicalDeviceFeatures2(e.pd, &f2);
    CT_CHECK(t, memcmp(&f, &f2.features, sizeof(f)) == 0);
    ct_env_fini(&e);
}

CT_TEST(features_vulkan11,
        "Features: VkPhysicalDeviceVulkan11Features members are VkBool32 values reporting the same features as "
        "VkPhysicalDevice16BitStorageFeatures, MultiviewFeatures, VariablePointersFeatures, "
        "ProtectedMemoryFeatures, SamplerYcbcrConversionFeatures and ShaderDrawParametersFeatures",
        "vkGetPhysicalDeviceFeatures2")
{
    struct ct_env e;
    struct ct_features f;
    VkPhysicalDevice16BitStorageFeatures s16;
    VkPhysicalDeviceMultiviewFeatures mv;
    VkPhysicalDeviceVariablePointersFeatures vp;
    VkPhysicalDeviceProtectedMemoryFeatures pm;
    VkPhysicalDeviceSamplerYcbcrConversionFeatures ycbcr;
    VkPhysicalDeviceShaderDrawParametersFeatures dp;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_2, "VkPhysicalDeviceVulkan11Features");
    ct_get_features(&e, &f, FILL_A);
    QUERY_F(&e, s16, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, FILL_B);
    QUERY_F(&e, mv, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES, FILL_B);
    QUERY_F(&e, vp, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES, FILL_B);
    QUERY_F(&e, pm, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES, FILL_B);
    QUERY_F(&e, ycbcr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES, FILL_B);
    QUERY_F(&e, dp, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES, FILL_B);
    BOOLS(f.v11, V11_BOOLS);
    EQ(f.v11, s16, storageBuffer16BitAccess);
    EQ(f.v11, s16, uniformAndStorageBuffer16BitAccess);
    EQ(f.v11, s16, storagePushConstant16);
    EQ(f.v11, s16, storageInputOutput16);
    EQ(f.v11, mv, multiview);
    EQ(f.v11, mv, multiviewGeometryShader);
    EQ(f.v11, mv, multiviewTessellationShader);
    EQ(f.v11, vp, variablePointersStorageBuffer);
    EQ(f.v11, vp, variablePointers);
    EQ(f.v11, pm, protectedMemory);
    EQ(f.v11, ycbcr, samplerYcbcrConversion);
    EQ(f.v11, dp, shaderDrawParameters);
    ct_env_fini(&e);
}

CT_TEST(features_vulkan12,
        "Features: VkPhysicalDeviceVulkan12Features members are VkBool32 values reporting the same features as "
        "the corresponding Vulkan 1.2 feature structures (8BitStorage, ShaderAtomicInt64, ShaderFloat16Int8, "
        "DescriptorIndexing, ScalarBlockLayout, ImagelessFramebuffer, UniformBufferStandardLayout, "
        "ShaderSubgroupExtendedTypes, SeparateDepthStencilLayouts, HostQueryReset, TimelineSemaphore, "
        "BufferDeviceAddress, VulkanMemoryModel)",
        "vkGetPhysicalDeviceFeatures2")
{
    struct ct_env e;
    struct ct_features f, g;
    VkPhysicalDevice8BitStorageFeatures s8;
    VkPhysicalDeviceShaderAtomicInt64Features a64;
    VkPhysicalDeviceShaderFloat16Int8Features f16;
    VkPhysicalDeviceDescriptorIndexingFeatures di;
    VkPhysicalDeviceScalarBlockLayoutFeatures sbl;
    VkPhysicalDeviceImagelessFramebufferFeatures ifb;
    VkPhysicalDeviceUniformBufferStandardLayoutFeatures ubsl;
    VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures set;
    VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures sdsl;
    VkPhysicalDeviceHostQueryResetFeatures hqr;
    VkPhysicalDeviceTimelineSemaphoreFeatures tsem;
    VkPhysicalDeviceBufferDeviceAddressFeatures bda;
    VkPhysicalDeviceVulkanMemoryModelFeatures vmm;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_2, "VkPhysicalDeviceVulkan12Features");
    ct_get_features(&e, &f, FILL_A);
    ct_get_features(&e, &g, FILL_B);
    QUERY_F(&e, s8, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, FILL_B);
    QUERY_F(&e, a64, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES, FILL_B);
    QUERY_F(&e, f16, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, FILL_B);
    QUERY_F(&e, di, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES, FILL_B);
    QUERY_F(&e, sbl, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES, FILL_B);
    QUERY_F(&e, ifb, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES, FILL_B);
    QUERY_F(&e, ubsl, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES, FILL_B);
    QUERY_F(&e, set, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES, FILL_B);
    QUERY_F(&e, sdsl, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES, FILL_B);
    QUERY_F(&e, hqr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES, FILL_B);
    QUERY_F(&e, tsem, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, FILL_B);
    QUERY_F(&e, bda, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, FILL_B);
    QUERY_F(&e, vmm, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES, FILL_B);
    BOOLS(f.v12, V12_BOOLS);
    EQ(f.v12, g.v12, samplerMirrorClampToEdge);
    EQ(f.v12, g.v12, drawIndirectCount);
    EQ(f.v12, s8, storageBuffer8BitAccess);
    EQ(f.v12, s8, uniformAndStorageBuffer8BitAccess);
    EQ(f.v12, s8, storagePushConstant8);
    EQ(f.v12, a64, shaderBufferInt64Atomics);
    EQ(f.v12, a64, shaderSharedInt64Atomics);
    EQ(f.v12, f16, shaderFloat16);
    EQ(f.v12, f16, shaderInt8);
    EQ(f.v12, g.v12, descriptorIndexing);
    EQ(f.v12, di, shaderInputAttachmentArrayDynamicIndexing);
    EQ(f.v12, di, shaderUniformTexelBufferArrayDynamicIndexing);
    EQ(f.v12, di, shaderStorageTexelBufferArrayDynamicIndexing);
    EQ(f.v12, di, shaderUniformBufferArrayNonUniformIndexing);
    EQ(f.v12, di, shaderSampledImageArrayNonUniformIndexing);
    EQ(f.v12, di, shaderStorageBufferArrayNonUniformIndexing);
    EQ(f.v12, di, shaderStorageImageArrayNonUniformIndexing);
    EQ(f.v12, di, shaderInputAttachmentArrayNonUniformIndexing);
    EQ(f.v12, di, shaderUniformTexelBufferArrayNonUniformIndexing);
    EQ(f.v12, di, shaderStorageTexelBufferArrayNonUniformIndexing);
    EQ(f.v12, di, descriptorBindingUniformBufferUpdateAfterBind);
    EQ(f.v12, di, descriptorBindingSampledImageUpdateAfterBind);
    EQ(f.v12, di, descriptorBindingStorageImageUpdateAfterBind);
    EQ(f.v12, di, descriptorBindingStorageBufferUpdateAfterBind);
    EQ(f.v12, di, descriptorBindingUniformTexelBufferUpdateAfterBind);
    EQ(f.v12, di, descriptorBindingStorageTexelBufferUpdateAfterBind);
    EQ(f.v12, di, descriptorBindingUpdateUnusedWhilePending);
    EQ(f.v12, di, descriptorBindingPartiallyBound);
    EQ(f.v12, di, descriptorBindingVariableDescriptorCount);
    EQ(f.v12, di, runtimeDescriptorArray);
    EQ(f.v12, g.v12, samplerFilterMinmax);
    EQ(f.v12, sbl, scalarBlockLayout);
    EQ(f.v12, ifb, imagelessFramebuffer);
    EQ(f.v12, ubsl, uniformBufferStandardLayout);
    EQ(f.v12, set, shaderSubgroupExtendedTypes);
    EQ(f.v12, sdsl, separateDepthStencilLayouts);
    EQ(f.v12, hqr, hostQueryReset);
    EQ(f.v12, tsem, timelineSemaphore);
    EQ(f.v12, bda, bufferDeviceAddress);
    EQ(f.v12, bda, bufferDeviceAddressCaptureReplay);
    EQ(f.v12, bda, bufferDeviceAddressMultiDevice);
    EQ(f.v12, vmm, vulkanMemoryModel);
    EQ(f.v12, vmm, vulkanMemoryModelDeviceScope);
    EQ(f.v12, vmm, vulkanMemoryModelAvailabilityVisibilityChains);
    EQ(f.v12, g.v12, shaderOutputViewportIndex);
    EQ(f.v12, g.v12, shaderOutputLayer);
    EQ(f.v12, g.v12, subgroupBroadcastDynamicId);
    ct_env_fini(&e);
}

CT_TEST(features_vulkan13,
        "Features: VkPhysicalDeviceVulkan13Features members are VkBool32 values reporting the same features as "
        "the corresponding Vulkan 1.3 feature structures",
        "vkGetPhysicalDeviceFeatures2")
{
    struct ct_env e;
    struct ct_features f;
    VkPhysicalDeviceImageRobustnessFeatures ir;
    VkPhysicalDeviceInlineUniformBlockFeatures iub;
    VkPhysicalDevicePipelineCreationCacheControlFeatures pccc;
    VkPhysicalDevicePrivateDataFeatures pdata;
    VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures demote;
    VkPhysicalDeviceShaderTerminateInvocationFeatures term;
    VkPhysicalDeviceSubgroupSizeControlFeatures ssc;
    VkPhysicalDeviceSynchronization2Features sync2;
    VkPhysicalDeviceTextureCompressionASTCHDRFeatures astc;
    VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures zero;
    VkPhysicalDeviceDynamicRenderingFeatures dr;
    VkPhysicalDeviceShaderIntegerDotProductFeatures dot;
    VkPhysicalDeviceMaintenance4Features m4;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_3, "VkPhysicalDeviceVulkan13Features");
    ct_get_features(&e, &f, FILL_A);
    QUERY_F(&e, ir, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES, FILL_B);
    QUERY_F(&e, iub, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES, FILL_B);
    QUERY_F(&e, pccc, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES, FILL_B);
    QUERY_F(&e, pdata, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES, FILL_B);
    QUERY_F(&e, demote, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES, FILL_B);
    QUERY_F(&e, term, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES, FILL_B);
    QUERY_F(&e, ssc, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, FILL_B);
    QUERY_F(&e, sync2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, FILL_B);
    QUERY_F(&e, astc, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES, FILL_B);
    QUERY_F(&e, zero, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES, FILL_B);
    QUERY_F(&e, dr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, FILL_B);
    QUERY_F(&e, dot, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, FILL_B);
    QUERY_F(&e, m4, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES, FILL_B);
    BOOLS(f.v13, V13_BOOLS);
    EQ(f.v13, ir, robustImageAccess);
    EQ(f.v13, iub, inlineUniformBlock);
    EQ(f.v13, iub, descriptorBindingInlineUniformBlockUpdateAfterBind);
    EQ(f.v13, pccc, pipelineCreationCacheControl);
    EQ(f.v13, pdata, privateData);
    EQ(f.v13, demote, shaderDemoteToHelperInvocation);
    EQ(f.v13, term, shaderTerminateInvocation);
    EQ(f.v13, ssc, subgroupSizeControl);
    EQ(f.v13, ssc, computeFullSubgroups);
    EQ(f.v13, sync2, synchronization2);
    EQ(f.v13, astc, textureCompressionASTC_HDR);
    EQ(f.v13, zero, shaderZeroInitializeWorkgroupMemory);
    EQ(f.v13, dr, dynamicRendering);
    EQ(f.v13, dot, shaderIntegerDotProduct);
    EQ(f.v13, m4, maintenance4);
    ct_env_fini(&e);
}

CT_TEST(features_vulkan14,
        "Features: VkPhysicalDeviceVulkan14Features members are VkBool32 values reporting the same features as "
        "the corresponding Vulkan 1.4 feature structures",
        "vkGetPhysicalDeviceFeatures2")
{
    struct ct_env e;
    struct ct_features f, g;
    VkPhysicalDeviceGlobalPriorityQueryFeatures gpq;
    VkPhysicalDeviceShaderSubgroupRotateFeatures rot;
    VkPhysicalDeviceShaderFloatControls2Features fc2;
    VkPhysicalDeviceShaderExpectAssumeFeatures ea;
    VkPhysicalDeviceLineRasterizationFeatures lr;
    VkPhysicalDeviceVertexAttributeDivisorFeatures vad;
    VkPhysicalDeviceIndexTypeUint8Features u8;
    VkPhysicalDeviceDynamicRenderingLocalReadFeatures drlr;
    VkPhysicalDeviceMaintenance5Features m5;
    VkPhysicalDeviceMaintenance6Features m6;
    VkPhysicalDevicePipelineProtectedAccessFeatures ppa;
    VkPhysicalDevicePipelineRobustnessFeatures pr;
    VkPhysicalDeviceHostImageCopyFeatures hic;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_4, "VkPhysicalDeviceVulkan14Features");
    ct_get_features(&e, &f, FILL_A);
    ct_get_features(&e, &g, FILL_B);
    QUERY_F(&e, gpq, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES, FILL_B);
    QUERY_F(&e, rot, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES, FILL_B);
    QUERY_F(&e, fc2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES, FILL_B);
    QUERY_F(&e, ea, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_EXPECT_ASSUME_FEATURES, FILL_B);
    QUERY_F(&e, lr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES, FILL_B);
    QUERY_F(&e, vad, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES, FILL_B);
    QUERY_F(&e, u8, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES, FILL_B);
    QUERY_F(&e, drlr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES, FILL_B);
    QUERY_F(&e, m5, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES, FILL_B);
    QUERY_F(&e, m6, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES, FILL_B);
    QUERY_F(&e, ppa, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_PROTECTED_ACCESS_FEATURES, FILL_B);
    QUERY_F(&e, pr, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES, FILL_B);
    QUERY_F(&e, hic, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES, FILL_B);
    BOOLS(f.v14, V14_BOOLS);
    EQ(f.v14, gpq, globalPriorityQuery);
    EQ(f.v14, rot, shaderSubgroupRotate);
    EQ(f.v14, rot, shaderSubgroupRotateClustered);
    EQ(f.v14, fc2, shaderFloatControls2);
    EQ(f.v14, ea, shaderExpectAssume);
    EQ(f.v14, lr, rectangularLines);
    EQ(f.v14, lr, bresenhamLines);
    EQ(f.v14, lr, smoothLines);
    EQ(f.v14, lr, stippledRectangularLines);
    EQ(f.v14, lr, stippledBresenhamLines);
    EQ(f.v14, lr, stippledSmoothLines);
    EQ(f.v14, vad, vertexAttributeInstanceRateDivisor);
    EQ(f.v14, vad, vertexAttributeInstanceRateZeroDivisor);
    EQ(f.v14, u8, indexTypeUint8);
    EQ(f.v14, drlr, dynamicRenderingLocalRead);
    EQ(f.v14, m5, maintenance5);
    EQ(f.v14, m6, maintenance6);
    EQ(f.v14, ppa, pipelineProtectedAccess);
    EQ(f.v14, pr, pipelineRobustness);
    EQ(f.v14, hic, hostImageCopy);
    EQ(f.v14, g.v14, pushDescriptor);
    ct_env_fini(&e);
}

/* ==========================================================================
 * Queue families
 * ========================================================================== */

static uint32_t get_families(const struct ct_env *e, VkQueueFamilyProperties *out, uint32_t max)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(e->pd, &count, NULL);
    if (count > max)
        count = max;
    vkGetPhysicalDeviceQueueFamilyProperties(e->pd, &count, out);
    return count;
}

CT_TEST(queue_families,
        "Devices and Queues > Queue Family Properties: at least one family; each with at least one queue; "
        "timestampValidBits 0 or 36..64; families supporting graphics or compute report "
        "minImageTransferGranularity (1,1,1), others (0,0,0) or powers of two; if any family supports graphics "
        "one supports graphics and compute; some family supports graphics, compute or video",
        "vkGetPhysicalDeviceQueueFamilyProperties")
{
    struct ct_env e;
    VkQueueFamilyProperties fam[64];
    uint32_t count, i;
    int graphics = 0, graphics_compute = 0, work = 0;

    ct_env_init(t, &e);
    count = get_families(&e, fam, 64);
    CT_REQUIRE(t, count >= 1);
    for (i = 0; i < count; i++) {
        VkQueueFlags f = fam[i].queueFlags;
        VkExtent3D g = fam[i].minImageTransferGranularity;
        CT_CHECKF(t, fam[i].queueCount >= 1, "family %u has no queue", i);
        CT_CHECKF(t, fam[i].timestampValidBits == 0 ||
                         (fam[i].timestampValidBits >= 36 && fam[i].timestampValidBits <= 64),
                  "family %u: timestampValidBits %u", i, fam[i].timestampValidBits);
        if (f & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
            CT_CHECKF(t, g.width == 1 && g.height == 1 && g.depth == 1,
                      "family %u supports graphics/compute but minImageTransferGranularity is (%u,%u,%u)", i, g.width,
                      g.height, g.depth);
        else
            CT_CHECKF(t, (g.width == 0 && g.height == 0 && g.depth == 0) ||
                             (ct_is_pow2(g.width) && ct_is_pow2(g.height) && ct_is_pow2(g.depth)),
                      "family %u: minImageTransferGranularity (%u,%u,%u)", i, g.width, g.height, g.depth);
        graphics |= (f & VK_QUEUE_GRAPHICS_BIT) != 0;
        graphics_compute |= (f & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
                            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
        work |= (f & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_VIDEO_DECODE_BIT_KHR |
                      VK_QUEUE_VIDEO_ENCODE_BIT_KHR)) != 0;
    }
    CT_CHECKF(t, work, "no family supports graphics, compute or video");
    if (graphics)
        CT_CHECKF(t, graphics_compute, "families support graphics but none supports graphics and compute");
    ct_env_fini(&e);
}

CT_TEST(queue_families_sparse_protected,
        "Devices and Queues > VkQueueFlagBits: if any sparse feature is supported a family supports "
        "VK_QUEUE_SPARSE_BINDING_BIT; if protectedMemory is supported a family supports VK_QUEUE_PROTECTED_BIT, "
        "and one supports graphics, compute and protected operations",
        "vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures2")
{
    struct ct_env e;
    struct ct_features f;
    VkQueueFamilyProperties fam[64];
    const VkPhysicalDeviceFeatures *c;
    uint32_t count, i;
    int sparse_feature, sparse_family = 0, protected_family = 0, gcp = 0;
    VkBool32 protected_memory = VK_FALSE;

    ct_env_init(t, &e);
    ct_get_features(&e, &f, 0);
    c = &f.f2.features;
    sparse_feature = c->sparseBinding || c->sparseResidencyBuffer || c->sparseResidencyImage2D ||
                     c->sparseResidencyImage3D || c->sparseResidency2Samples || c->sparseResidency4Samples ||
                     c->sparseResidency8Samples || c->sparseResidency16Samples || c->sparseResidencyAliased;
    if (e.version >= VK_API_VERSION_1_2)
        protected_memory = f.v11.protectedMemory;
    count = get_families(&e, fam, 64);
    for (i = 0; i < count; i++) {
        VkQueueFlags q = fam[i].queueFlags;
        sparse_family |= (q & VK_QUEUE_SPARSE_BINDING_BIT) != 0;
        protected_family |= (q & VK_QUEUE_PROTECTED_BIT) != 0;
        gcp |= (q & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_PROTECTED_BIT)) ==
               (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_PROTECTED_BIT);
    }
    if (!sparse_feature && !protected_memory)
        ct_skip(t, "neither sparse resources nor protected memory are supported");
    if (sparse_feature)
        CT_CHECKF(t, sparse_family, "a sparse feature is supported but no family supports sparse binding");
    if (protected_memory) {
        CT_CHECKF(t, protected_family, "protectedMemory is supported but no family has VK_QUEUE_PROTECTED_BIT");
        CT_CHECKF(t, gcp, "protectedMemory is supported but no family supports graphics, compute and protected");
    }
    ct_env_fini(&e);
}

CT_TEST(queue_families_incomplete,
        "Devices and Queues > Physical Devices: with pQueueFamilyPropertyCount less than the number of families, "
        "at most that many structures are written and the count is set to the number written",
        "vkGetPhysicalDeviceQueueFamilyProperties")
{
    struct ct_env e;
    VkQueueFamilyProperties all[65], part[65];
    uint32_t count = 0, k;

    ct_env_init(t, &e);
    vkGetPhysicalDeviceQueueFamilyProperties(e.pd, &count, NULL);
    CT_REQUIRE(t, count >= 1 && count <= 64);
    vkGetPhysicalDeviceQueueFamilyProperties(e.pd, &count, all);
    for (k = 0; k <= count; k++) {
        uint32_t n = k;
        memset(part, 0xA5, sizeof(part));
        vkGetPhysicalDeviceQueueFamilyProperties(e.pd, &n, part);
        CT_CHECKF(t, n == k, "room for %u of %u: count returned %u", k, count, n);
        CT_CHECKF(t, memcmp(part, all, k * sizeof(all[0])) == 0, "room for %u: the families differ", k);
        CT_CHECKF(t, ct_all_bytes(&part[k], sizeof(part[k]), 0xA5), "room for %u: wrote family %u", k, k);
    }
    {
        uint32_t n = count + 1;
        vkGetPhysicalDeviceQueueFamilyProperties(e.pd, &n, part);
        CT_CHECKF(t, n == count, "room for %u of %u: count returned %u", count + 1, count, n);
    }
    ct_env_fini(&e);
}

CT_TEST(queue_families2,
        "Devices and Queues > Physical Devices: vkGetPhysicalDeviceQueueFamilyProperties2 returns the same "
        "families, with VkQueueFamilyProperties2::queueFamilyProperties populated with the same values; fewer "
        "slots write that many",
        "vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties2")
{
    struct ct_env e;
    VkQueueFamilyProperties all[64];
    VkQueueFamilyProperties2 p2[65];
    uint32_t count = 0, n = 0, i;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_1, "vkGetPhysicalDeviceQueueFamilyProperties2");
    count = get_families(&e, all, 64);
    vkGetPhysicalDeviceQueueFamilyProperties2(e.pd, &n, NULL);
    CT_REQUIRE(t, n == count);
    for (i = 0; i < 65; i++) {
        memset(&p2[i], 0xA5, sizeof(p2[i]));
        p2[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        p2[i].pNext = NULL;
    }
    n = count + 1;
    vkGetPhysicalDeviceQueueFamilyProperties2(e.pd, &n, p2);
    CT_CHECKF(t, n == count, "count %u, expected %u", n, count);
    for (i = 0; i < count; i++) {
        CT_CHECKF(t, memcmp(&p2[i].queueFamilyProperties, &all[i], sizeof(all[i])) == 0, "family %u differs", i);
        CT_CHECK(t, p2[i].sType == VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2 && p2[i].pNext == NULL);
    }
    CT_CHECKF(t, ct_all_bytes(&p2[count].queueFamilyProperties, sizeof(p2[count].queueFamilyProperties), 0xA5),
              "wrote past the family count");
    if (count > 1) {
        n = count - 1;
        vkGetPhysicalDeviceQueueFamilyProperties2(e.pd, &n, p2);
        CT_CHECK(t, n == count - 1);
    }
    n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties2(e.pd, &n, p2);
    CT_CHECK(t, n == 0);
    ct_env_fini(&e);
}

CT_TEST(queue_family_global_priority,
        "Devices and Queues > VkQueueFamilyGlobalPriorityProperties: priorityCount is greater than 0 (at most "
        "VK_MAX_GLOBAL_PRIORITY_SIZE) and the priorities are a continuous ascending sequence of "
        "VkQueueGlobalPriority values without duplicates",
        "vkGetPhysicalDeviceQueueFamilyProperties2")
{
    static const VkQueueGlobalPriority order[] = { VK_QUEUE_GLOBAL_PRIORITY_LOW, VK_QUEUE_GLOBAL_PRIORITY_MEDIUM,
                                                   VK_QUEUE_GLOBAL_PRIORITY_HIGH, VK_QUEUE_GLOBAL_PRIORITY_REALTIME };
    struct ct_env e;
    VkQueueFamilyProperties2 p2[64];
    VkQueueFamilyGlobalPriorityProperties gp[64];
    uint32_t count = 64, i, j, k;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_4, "VkQueueFamilyGlobalPriorityProperties");
    for (i = 0; i < 64; i++) {
        memset(&gp[i], 0xA5, sizeof(gp[i]));
        gp[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES;
        gp[i].pNext = NULL;
        memset(&p2[i], 0, sizeof(p2[i]));
        p2[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        p2[i].pNext = &gp[i];
    }
    vkGetPhysicalDeviceQueueFamilyProperties2(e.pd, &count, p2);
    for (i = 0; i < count; i++) {
        uint32_t n = gp[i].priorityCount;
        CT_CHECKF(t, n >= 1 && n <= VK_MAX_GLOBAL_PRIORITY_SIZE, "family %u: priorityCount %u", i, n);
        if (n < 1 || n > VK_MAX_GLOBAL_PRIORITY_SIZE)
            continue;
        for (k = 0; k < CT_ARRAY_SIZE(order) && order[k] != gp[i].priorities[0]; k++)
            ;
        CT_CHECKF(t, k < CT_ARRAY_SIZE(order), "family %u: priorities[0] %d is not a VkQueueGlobalPriority", i,
                  (int)gp[i].priorities[0]);
        for (j = 1; j < n && k < CT_ARRAY_SIZE(order); j++)
            CT_CHECKF(t, k + j < CT_ARRAY_SIZE(order) && gp[i].priorities[j] == order[k + j],
                      "family %u: priorities[%u] %d breaks the continuous ascending sequence", i, j,
                      (int)gp[i].priorities[j]);
    }
    ct_env_fini(&e);
}

/* ==========================================================================
 * Devices and queues
 * ========================================================================== */

static VkResult create_device(const struct ct_env *e, const VkDeviceQueueCreateInfo *queues, uint32_t queue_count,
                              const void *pnext, const VkPhysicalDeviceFeatures *features, const char *const *exts,
                              uint32_t ext_count, const VkAllocationCallbacks *alloc, VkDevice *device)
{
    const char *names[64];
    VkDeviceCreateInfo info;
    uint32_t i, n = 0;
    for (i = 0; i < ext_count && n < 62; i++)
        names[n++] = exts[i];
    n = ct_add_portability_subset(e, names, n);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.pNext = pnext;
    info.queueCreateInfoCount = queue_count;
    info.pQueueCreateInfos = queues;
    info.enabledExtensionCount = n;
    info.ppEnabledExtensionNames = n ? names : NULL;
    info.pEnabledFeatures = features;
    return vkCreateDevice(e->pd, &info, alloc, device);
}

static VkDeviceQueueCreateInfo one_queue(uint32_t family, const float *priority)
{
    VkDeviceQueueCreateInfo q;
    memset(&q, 0, sizeof(q));
    q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    q.queueFamilyIndex = family;
    q.queueCount = 1;
    q.pQueuePriorities = priority;
    return q;
}

CT_TEST(create_device_every_family,
        "Devices and Queues > Queue Creation: a device can be created with every queue of a family, priorities "
        "spread over [0.0, 1.0]; vkGetDeviceQueue returns a distinct, stable handle for each created queue",
        "vkCreateDevice vkGetDeviceQueue vkDestroyDevice vkGetPhysicalDeviceQueueFamilyProperties")
{
    struct ct_env e;
    VkQueueFamilyProperties fam[64];
    uint32_t count, f;

    ct_env_init(t, &e);
    count = get_families(&e, fam, 64);
    for (f = 0; f < count; f++) {
        float priorities[64];
        VkQueue queues[64];
        VkDeviceQueueCreateInfo q = one_queue(f, priorities);
        VkDevice device = VK_NULL_HANDLE;
        uint32_t n = CT_MIN(fam[f].queueCount, 64), i, j;
        VkResult r;

        for (i = 0; i < n; i++)
            priorities[i] = n > 1 ? (float)i / (float)(n - 1) : 0.5f;
        q.queueCount = n;
        r = create_device(&e, &q, 1, NULL, NULL, NULL, 0, NULL, &device);
        CT_CHECKF(t, r == VK_SUCCESS, "family %u with %u queues: %s", f, n, ct_result_name(r));
        if (r != VK_SUCCESS)
            continue;
        for (i = 0; i < n; i++) {
            VkQueue again = VK_NULL_HANDLE;
            queues[i] = VK_NULL_HANDLE;
            vkGetDeviceQueue(device, f, i, &queues[i]);
            vkGetDeviceQueue(device, f, i, &again);
            CT_CHECKF(t, queues[i] != VK_NULL_HANDLE, "family %u queue %u is NULL", f, i);
            CT_CHECKF(t, again == queues[i], "family %u queue %u: two queries differ", f, i);
            for (j = 0; j < i; j++)
                CT_CHECKF(t, queues[i] != queues[j], "family %u: queues %u and %u share a handle", f, i, j);
        }
        vkDestroyDevice(device, NULL);
    }
    ct_env_fini(&e);
}

CT_TEST(create_device_all_families,
        "Devices and Queues > Queue Creation: one device can create queues from every family at once (distinct "
        "queueFamilyIndex per VkDeviceQueueCreateInfo, VUID-VkDeviceCreateInfo-queueFamilyIndex-02802)",
        "vkCreateDevice vkGetDeviceQueue vkDestroyDevice")
{
    struct ct_env e;
    VkQueueFamilyProperties fam[64];
    VkDeviceQueueCreateInfo q[64];
    float priority = 1.0f;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t count, f;

    ct_env_init(t, &e);
    count = get_families(&e, fam, 64);
    for (f = 0; f < count; f++)
        q[f] = one_queue(f, &priority);
    CT_REQUIRE_VK(t, create_device(&e, q, count, NULL, NULL, NULL, 0, NULL, &device));
    for (f = 0; f < count; f++) {
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, f, 0, &queue);
        CT_CHECKF(t, queue != VK_NULL_HANDLE, "family %u: NULL queue", f);
    }
    vkDestroyDevice(device, NULL);
    ct_env_fini(&e);
}

CT_TEST(create_device_twice,
        "Devices and Queues > Device Creation: multiple logical devices can be created from the same physical "
        "device",
        "vkCreateDevice vkDestroyDevice vkGetDeviceQueue")
{
    struct ct_env e;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo q = one_queue(0, &priority);
    VkDevice a = VK_NULL_HANDLE, b = VK_NULL_HANDLE;
    VkQueue qa = VK_NULL_HANDLE, qb = VK_NULL_HANDLE;

    ct_env_init(t, &e);
    CT_REQUIRE_VK(t, create_device(&e, &q, 1, NULL, NULL, NULL, 0, NULL, &a));
    CT_REQUIRE_VK(t, create_device(&e, &q, 1, NULL, NULL, NULL, 0, NULL, &b));
    CT_CHECK(t, a != b);
    vkGetDeviceQueue(a, 0, 0, &qa);
    vkGetDeviceQueue(b, 0, 0, &qb);
    CT_CHECK(t, qa && qb && qa != qb);
    vkDestroyDevice(a, NULL);
    vkDestroyDevice(b, NULL);
    ct_env_fini(&e);
}

CT_TEST(get_device_queue2,
        "Devices and Queues > Queue Creation: vkGetDeviceQueue2 with flags 0 returns the queue vkGetDeviceQueue "
        "returns for the same family and index",
        "vkGetDeviceQueue2 vkGetDeviceQueue vkCreateDevice vkDestroyDevice")
{
    struct ct_env e;
    VkQueueFamilyProperties fam[64];
    float priorities[2] = { 1.0f, 0.0f };
    VkDeviceQueueCreateInfo q;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t i, n;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_1, "vkGetDeviceQueue2");
    get_families(&e, fam, 64);
    q = one_queue(0, priorities);
    q.queueCount = n = CT_MIN(fam[0].queueCount, 2);
    CT_REQUIRE_VK(t, create_device(&e, &q, 1, NULL, NULL, NULL, 0, NULL, &device));
    for (i = 0; i < n; i++) {
        VkDeviceQueueInfo2 info;
        VkQueue a = VK_NULL_HANDLE, b = VK_NULL_HANDLE;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2;
        info.queueFamilyIndex = 0;
        info.queueIndex = i;
        vkGetDeviceQueue(device, 0, i, &a);
        vkGetDeviceQueue2(device, &info, &b);
        CT_CHECKF(t, a != VK_NULL_HANDLE && a == b, "queue %u: vkGetDeviceQueue %p, vkGetDeviceQueue2 %p", i,
                  (void *)a, (void *)b);
    }
    vkDestroyDevice(device, NULL);
    ct_env_fini(&e);
}

CT_TEST(get_device_queue2_protected,
        "Devices and Queues > Queue Creation: queues created with VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT are "
        "retrieved with vkGetDeviceQueue2 and the same flags, separately from unprotected queues",
        "vkGetDeviceQueue2 vkGetDeviceQueue vkCreateDevice vkDestroyDevice")
{
    struct ct_env e;
    struct ct_features f;
    VkQueueFamilyProperties fam[64];
    VkPhysicalDeviceProtectedMemoryFeatures pm;
    VkDeviceQueueCreateInfo q[2];
    VkDeviceQueueInfo2 info;
    float priority = 1.0f;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue plain = VK_NULL_HANDLE, prot = VK_NULL_HANDLE;
    uint32_t count, family = UINT32_MAX, i;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_2, "protected memory feature query");
    ct_get_features(&e, &f, 0);
    if (!f.v11.protectedMemory)
        ct_skip(t, "protectedMemory is not supported");
    count = get_families(&e, fam, 64);
    for (i = 0; i < count && family == UINT32_MAX; i++)
        if (fam[i].queueFlags & VK_QUEUE_PROTECTED_BIT)
            family = i;
    CT_REQUIRE(t, family != UINT32_MAX);
    memset(&pm, 0, sizeof(pm));
    pm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES;
    pm.protectedMemory = VK_TRUE;
    q[0] = one_queue(family, &priority);
    q[1] = one_queue(family, &priority);
    q[1].flags = VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT;
    CT_REQUIRE_VK(t, create_device(&e, q, fam[family].queueCount >= 2 ? 2 : 1, &pm, NULL, NULL, 0, NULL, &device));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2;
    info.queueFamilyIndex = family;
    vkGetDeviceQueue2(device, &info, &plain);
    CT_CHECK(t, plain != VK_NULL_HANDLE);
    if (fam[family].queueCount >= 2) {
        info.flags = VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT;
        vkGetDeviceQueue2(device, &info, &prot);
        CT_CHECK(t, prot != VK_NULL_HANDLE && prot != plain);
    }
    vkDestroyDevice(device, NULL);
    ct_env_fini(&e);
}

CT_TEST(destroy_device_null,
        "Devices and Queues > Device Destruction: vkDestroyDevice accepts a NULL device (implicit valid usage: "
        "\"if device is not NULL\") and does nothing",
        "vkDestroyDevice")
{
    (void)t;
    vkDestroyDevice(VK_NULL_HANDLE, NULL);
}

/* Requesting an unsupported feature must fail with
 * VK_ERROR_FEATURE_NOT_PRESENT. Each feature is requested alone (with the
 * features valid usage makes it depend on). */

/* Member names, in declaration order, for messages. */
static const char *const core_feature_names[] = {
    "robustBufferAccess", "fullDrawIndexUint32", "imageCubeArray", "independentBlend", "geometryShader",
    "tessellationShader", "sampleRateShading", "dualSrcBlend", "logicOp", "multiDrawIndirect",
    "drawIndirectFirstInstance", "depthClamp", "depthBiasClamp", "fillModeNonSolid", "depthBounds", "wideLines",
    "largePoints", "alphaToOne", "multiViewport", "samplerAnisotropy", "textureCompressionETC2",
    "textureCompressionASTC_LDR", "textureCompressionBC", "occlusionQueryPrecise", "pipelineStatisticsQuery",
    "vertexPipelineStoresAndAtomics", "fragmentStoresAndAtomics", "shaderTessellationAndGeometryPointSize",
    "shaderImageGatherExtended", "shaderStorageImageExtendedFormats", "shaderStorageImageMultisample",
    "shaderStorageImageReadWithoutFormat", "shaderStorageImageWriteWithoutFormat",
    "shaderUniformBufferArrayDynamicIndexing", "shaderSampledImageArrayDynamicIndexing",
    "shaderStorageBufferArrayDynamicIndexing", "shaderStorageImageArrayDynamicIndexing", "shaderClipDistance",
    "shaderCullDistance", "shaderFloat64", "shaderInt64", "shaderInt16", "shaderResourceResidency",
    "shaderResourceMinLod", "sparseBinding", "sparseResidencyBuffer", "sparseResidencyImage2D",
    "sparseResidencyImage3D", "sparseResidency2Samples", "sparseResidency4Samples", "sparseResidency8Samples",
    "sparseResidency16Samples", "sparseResidencyAliased", "variableMultisampleRate", "inheritedQueries",
};
static const char *const v11_feature_names[] = {
    "storageBuffer16BitAccess", "uniformAndStorageBuffer16BitAccess", "storagePushConstant16",
    "storageInputOutput16", "multiview", "multiviewGeometryShader", "multiviewTessellationShader",
    "variablePointersStorageBuffer", "variablePointers", "protectedMemory", "samplerYcbcrConversion",
    "shaderDrawParameters",
};
static const char *const v12_feature_names[] = {
    "samplerMirrorClampToEdge", "drawIndirectCount", "storageBuffer8BitAccess", "uniformAndStorageBuffer8BitAccess",
    "storagePushConstant8", "shaderBufferInt64Atomics", "shaderSharedInt64Atomics", "shaderFloat16", "shaderInt8",
    "descriptorIndexing", "shaderInputAttachmentArrayDynamicIndexing", "shaderUniformTexelBufferArrayDynamicIndexing",
    "shaderStorageTexelBufferArrayDynamicIndexing", "shaderUniformBufferArrayNonUniformIndexing",
    "shaderSampledImageArrayNonUniformIndexing", "shaderStorageBufferArrayNonUniformIndexing",
    "shaderStorageImageArrayNonUniformIndexing", "shaderInputAttachmentArrayNonUniformIndexing",
    "shaderUniformTexelBufferArrayNonUniformIndexing", "shaderStorageTexelBufferArrayNonUniformIndexing",
    "descriptorBindingUniformBufferUpdateAfterBind", "descriptorBindingSampledImageUpdateAfterBind",
    "descriptorBindingStorageImageUpdateAfterBind", "descriptorBindingStorageBufferUpdateAfterBind",
    "descriptorBindingUniformTexelBufferUpdateAfterBind", "descriptorBindingStorageTexelBufferUpdateAfterBind",
    "descriptorBindingUpdateUnusedWhilePending", "descriptorBindingPartiallyBound",
    "descriptorBindingVariableDescriptorCount", "runtimeDescriptorArray", "samplerFilterMinmax", "scalarBlockLayout",
    "imagelessFramebuffer", "uniformBufferStandardLayout", "shaderSubgroupExtendedTypes",
    "separateDepthStencilLayouts", "hostQueryReset", "timelineSemaphore", "bufferDeviceAddress",
    "bufferDeviceAddressCaptureReplay", "bufferDeviceAddressMultiDevice", "vulkanMemoryModel",
    "vulkanMemoryModelDeviceScope", "vulkanMemoryModelAvailabilityVisibilityChains", "shaderOutputViewportIndex",
    "shaderOutputLayer", "subgroupBroadcastDynamicId",
};
static const char *const v13_feature_names[] = {
    "robustImageAccess", "inlineUniformBlock", "descriptorBindingInlineUniformBlockUpdateAfterBind",
    "pipelineCreationCacheControl", "privateData", "shaderDemoteToHelperInvocation", "shaderTerminateInvocation",
    "subgroupSizeControl", "computeFullSubgroups", "synchronization2", "textureCompressionASTC_HDR",
    "shaderZeroInitializeWorkgroupMemory", "dynamicRendering", "shaderIntegerDotProduct", "maintenance4",
};
static const char *const v14_feature_names[] = {
    "globalPriorityQuery", "shaderSubgroupRotate", "shaderSubgroupRotateClustered", "shaderFloatControls2",
    "shaderExpectAssume", "rectangularLines", "bresenhamLines", "smoothLines", "stippledRectangularLines",
    "stippledBresenhamLines", "stippledSmoothLines", "vertexAttributeInstanceRateDivisor",
    "vertexAttributeInstanceRateZeroDivisor", "indexTypeUint8", "dynamicRenderingLocalRead", "maintenance5",
    "maintenance6", "pipelineProtectedAccess", "pipelineRobustness", "hostImageCopy", "pushDescriptor",
};

typedef char core_names_match[CT_ARRAY_SIZE(core_feature_names) == sizeof(VkPhysicalDeviceFeatures) / 4 ? 1 : -1];
typedef char v11_names_match[CT_ARRAY_SIZE(v11_feature_names) == V11_BOOLS ? 1 : -1];
typedef char v12_names_match[CT_ARRAY_SIZE(v12_feature_names) == V12_BOOLS ? 1 : -1];
typedef char v13_names_match[CT_ARRAY_SIZE(v13_feature_names) == V13_BOOLS ? 1 : -1];
typedef char v14_names_match[CT_ARRAY_SIZE(v14_feature_names) == V14_BOOLS ? 1 : -1];

static VkBool32 *bool_at(void *s, size_t i)
{
    return (VkBool32 *)((char *)s + FIRST_BOOL) + i;
}

CT_TEST(create_device_unsupported_core_feature,
        "Devices and Queues > Device Creation: if any requested feature is not supported, vkCreateDevice must "
        "return VK_ERROR_FEATURE_NOT_PRESENT (each unsupported VkPhysicalDeviceFeatures member, through "
        "pEnabledFeatures and through VkPhysicalDeviceFeatures2)",
        "vkCreateDevice vkGetPhysicalDeviceFeatures")
{
    struct ct_env e;
    VkPhysicalDeviceFeatures have;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo q = one_queue(0, &priority);
    size_t i, tested = 0;

    ct_env_init(t, &e);
    vkGetPhysicalDeviceFeatures(e.pd, &have);
    for (i = 0; i < sizeof(have) / sizeof(VkBool32); i++) {
        VkPhysicalDeviceFeatures want;
        VkPhysicalDeviceFeatures2 f2;
        VkDevice device = VK_NULL_HANDLE;
        VkResult r;
        if (((const VkBool32 *)&have)[i])
            continue;
        tested++;
        memset(&want, 0, sizeof(want));
        ((VkBool32 *)&want)[i] = VK_TRUE;
        r = create_device(&e, &q, 1, NULL, &want, NULL, 0, NULL, &device);
        CT_CHECKF(t, r == VK_ERROR_FEATURE_NOT_PRESENT, "%s (pEnabledFeatures): %s", core_feature_names[i],
                  ct_result_name(r));
        if (r == VK_SUCCESS)
            vkDestroyDevice(device, NULL);
        if (e.version < VK_API_VERSION_1_1)
            continue;
        memset(&f2, 0, sizeof(f2));
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.features = want;
        device = VK_NULL_HANDLE;
        r = create_device(&e, &q, 1, &f2, NULL, NULL, 0, NULL, &device);
        CT_CHECKF(t, r == VK_ERROR_FEATURE_NOT_PRESENT, "%s (VkPhysicalDeviceFeatures2): %s", core_feature_names[i],
                  ct_result_name(r));
        if (r == VK_SUCCESS)
            vkDestroyDevice(device, NULL);
    }
    if (!tested)
        ct_skip(t, "every VkPhysicalDeviceFeatures member is supported");
    ct_env_fini(&e);
}

CT_TEST(create_device_unsupported_feature_chain,
        "Devices and Queues > Device Creation: if any requested feature is not supported, vkCreateDevice must "
        "return VK_ERROR_FEATURE_NOT_PRESENT (each unsupported member of VkPhysicalDeviceVulkan11/12/13/14Features "
        "in the pNext chain)",
        "vkCreateDevice vkGetPhysicalDeviceFeatures2")
{
    struct ct_env e;
    struct ct_features have;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo q = one_queue(0, &priority);
    struct {
        void *have;
        const char *const *names;
        size_t count;
        VkStructureType type;
        uint32_t version;
    } structs[4];
    size_t s, i, tested = 0;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_2, "VkPhysicalDeviceVulkan11Features");
    ct_get_features(&e, &have, 0);
    structs[0].have = &have.v11;
    structs[0].names = v11_feature_names;
    structs[0].count = V11_BOOLS;
    structs[0].type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    structs[0].version = VK_API_VERSION_1_2;
    structs[1].have = &have.v12;
    structs[1].names = v12_feature_names;
    structs[1].count = V12_BOOLS;
    structs[1].type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    structs[1].version = VK_API_VERSION_1_2;
    structs[2].have = &have.v13;
    structs[2].names = v13_feature_names;
    structs[2].count = V13_BOOLS;
    structs[2].type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    structs[2].version = VK_API_VERSION_1_3;
    structs[3].have = &have.v14;
    structs[3].names = v14_feature_names;
    structs[3].count = V14_BOOLS;
    structs[3].type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    structs[3].version = VK_API_VERSION_1_4;

    for (s = 0; s < 4; s++) {
        if (e.version < structs[s].version)
            continue;
        for (i = 0; i < structs[s].count; i++) {
            union {
                VkPhysicalDeviceVulkan11Features v11;
                VkPhysicalDeviceVulkan12Features v12;
                VkPhysicalDeviceVulkan13Features v13;
                VkPhysicalDeviceVulkan14Features v14;
            } want;
            VkDevice device = VK_NULL_HANDLE;
            VkResult r;
            if (*bool_at(structs[s].have, i))
                continue;
            tested++;
            memset(&want, 0, sizeof(want));
            ((VkBaseOutStructure *)&want)->sType = structs[s].type;
            *bool_at(&want, i) = VK_TRUE;
            /* Valid usage: multiview{Geometry,Tessellation}Shader need multiview,
             * variablePointers needs variablePointersStorageBuffer. */
            if (s == 0) {
                if (want.v11.multiviewGeometryShader || want.v11.multiviewTessellationShader)
                    want.v11.multiview = VK_TRUE;
                if (want.v11.variablePointers)
                    want.v11.variablePointersStorageBuffer = VK_TRUE;
            }
            r = create_device(&e, &q, 1, &want, NULL, NULL, 0, NULL, &device);
            CT_CHECKF(t, r == VK_ERROR_FEATURE_NOT_PRESENT, "%s (Vulkan 1.%zu features): %s", structs[s].names[i],
                      s + 1, ct_result_name(r));
            if (r == VK_SUCCESS)
                vkDestroyDevice(device, NULL);
        }
    }
    if (!tested)
        ct_skip(t, "every Vulkan 1.1-1.4 feature is supported");
    ct_env_fini(&e);
}

/* The feature structs of the extensions promoted to Vulkan 1.1-1.4, with
 * their first and last VkBool32 member and the version from which a feature
 * chain may include them. */
struct promoted_features {
    const char *name;
    VkStructureType type;
    size_t first, last;
    uint32_t version;
};

#define PROMOTED(T, type, first, last, v) { #T, type, offsetof(T, first), offsetof(T, last), v }

static const struct promoted_features promoted_features[] = {
    PROMOTED(VkPhysicalDevice16BitStorageFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
             storageBuffer16BitAccess, storageInputOutput16, VK_API_VERSION_1_1),
    PROMOTED(VkPhysicalDeviceMultiviewFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES, multiview,
             multiviewTessellationShader, VK_API_VERSION_1_1),
    PROMOTED(VkPhysicalDeviceVariablePointersFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES,
             variablePointersStorageBuffer, variablePointers, VK_API_VERSION_1_1),
    PROMOTED(VkPhysicalDeviceProtectedMemoryFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES,
             protectedMemory, protectedMemory, VK_API_VERSION_1_1),
    PROMOTED(VkPhysicalDeviceSamplerYcbcrConversionFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES, samplerYcbcrConversion,
             samplerYcbcrConversion, VK_API_VERSION_1_1),
    PROMOTED(VkPhysicalDeviceShaderDrawParametersFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES, shaderDrawParameters,
             shaderDrawParameters, VK_API_VERSION_1_1),
    PROMOTED(VkPhysicalDevice8BitStorageFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
             storageBuffer8BitAccess, storagePushConstant8, VK_API_VERSION_1_2),
    PROMOTED(VkPhysicalDeviceShaderAtomicInt64Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
             shaderBufferInt64Atomics, shaderSharedInt64Atomics, VK_API_VERSION_1_2),
    PROMOTED(VkPhysicalDeviceShaderFloat16Int8Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
             shaderFloat16, shaderInt8, VK_API_VERSION_1_2),
    PROMOTED(VkPhysicalDeviceDescriptorIndexingFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES, shaderInputAttachmentArrayDynamicIndexing,
             runtimeDescriptorArray, VK_API_VERSION_1_2),
    PROMOTED(VkPhysicalDeviceScalarBlockLayoutFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES,
             scalarBlockLayout, scalarBlockLayout, VK_API_VERSION_1_2),
    PROMOTED(VkPhysicalDeviceImagelessFramebufferFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES, imagelessFramebuffer,
             imagelessFramebuffer, VK_API_VERSION_1_2),
    PROMOTED(VkPhysicalDeviceUniformBufferStandardLayoutFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES, uniformBufferStandardLayout,
             uniformBufferStandardLayout, VK_API_VERSION_1_2),
    PROMOTED(VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES, shaderSubgroupExtendedTypes,
             shaderSubgroupExtendedTypes, VK_API_VERSION_1_2),
    PROMOTED(VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES, separateDepthStencilLayouts,
             separateDepthStencilLayouts, VK_API_VERSION_1_2),
    PROMOTED(VkPhysicalDeviceHostQueryResetFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES,
             hostQueryReset, hostQueryReset, VK_API_VERSION_1_2),
    PROMOTED(VkPhysicalDeviceTimelineSemaphoreFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
             timelineSemaphore, timelineSemaphore, VK_API_VERSION_1_2),
    PROMOTED(VkPhysicalDeviceBufferDeviceAddressFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, bufferDeviceAddress,
             bufferDeviceAddressMultiDevice, VK_API_VERSION_1_2),
    PROMOTED(VkPhysicalDeviceVulkanMemoryModelFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES,
             vulkanMemoryModel, vulkanMemoryModelAvailabilityVisibilityChains, VK_API_VERSION_1_2),
    PROMOTED(VkPhysicalDeviceImageRobustnessFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES,
             robustImageAccess, robustImageAccess, VK_API_VERSION_1_3),
    PROMOTED(VkPhysicalDeviceInlineUniformBlockFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES, inlineUniformBlock,
             descriptorBindingInlineUniformBlockUpdateAfterBind, VK_API_VERSION_1_3),
    PROMOTED(VkPhysicalDevicePipelineCreationCacheControlFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES, pipelineCreationCacheControl,
             pipelineCreationCacheControl, VK_API_VERSION_1_3),
    PROMOTED(VkPhysicalDevicePrivateDataFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES, privateData,
             privateData, VK_API_VERSION_1_3),
    PROMOTED(VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES,
             shaderDemoteToHelperInvocation, shaderDemoteToHelperInvocation, VK_API_VERSION_1_3),
    PROMOTED(VkPhysicalDeviceShaderTerminateInvocationFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES, shaderTerminateInvocation,
             shaderTerminateInvocation, VK_API_VERSION_1_3),
    PROMOTED(VkPhysicalDeviceSubgroupSizeControlFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, subgroupSizeControl,
             computeFullSubgroups, VK_API_VERSION_1_3),
    PROMOTED(VkPhysicalDeviceSynchronization2Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
             synchronization2, synchronization2, VK_API_VERSION_1_3),
    PROMOTED(VkPhysicalDeviceTextureCompressionASTCHDRFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES, textureCompressionASTC_HDR,
             textureCompressionASTC_HDR, VK_API_VERSION_1_3),
    PROMOTED(VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES,
             shaderZeroInitializeWorkgroupMemory, shaderZeroInitializeWorkgroupMemory, VK_API_VERSION_1_3),
    PROMOTED(VkPhysicalDeviceDynamicRenderingFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
             dynamicRendering, dynamicRendering, VK_API_VERSION_1_3),
    PROMOTED(VkPhysicalDeviceShaderIntegerDotProductFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, shaderIntegerDotProduct,
             shaderIntegerDotProduct, VK_API_VERSION_1_3),
    PROMOTED(VkPhysicalDeviceMaintenance4Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES,
             maintenance4, maintenance4, VK_API_VERSION_1_3),
    PROMOTED(VkPhysicalDeviceGlobalPriorityQueryFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES, globalPriorityQuery,
             globalPriorityQuery, VK_API_VERSION_1_4),
    PROMOTED(VkPhysicalDeviceShaderSubgroupRotateFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES, shaderSubgroupRotate,
             shaderSubgroupRotateClustered, VK_API_VERSION_1_4),
    PROMOTED(VkPhysicalDeviceShaderFloatControls2Features,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES, shaderFloatControls2,
             shaderFloatControls2, VK_API_VERSION_1_4),
    PROMOTED(VkPhysicalDeviceShaderExpectAssumeFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_EXPECT_ASSUME_FEATURES, shaderExpectAssume, shaderExpectAssume,
             VK_API_VERSION_1_4),
    PROMOTED(VkPhysicalDeviceLineRasterizationFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES, rectangularLines, stippledSmoothLines,
             VK_API_VERSION_1_4),
    PROMOTED(VkPhysicalDeviceVertexAttributeDivisorFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES, vertexAttributeInstanceRateDivisor,
             vertexAttributeInstanceRateZeroDivisor, VK_API_VERSION_1_4),
    PROMOTED(VkPhysicalDeviceIndexTypeUint8Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES,
             indexTypeUint8, indexTypeUint8, VK_API_VERSION_1_4),
    PROMOTED(VkPhysicalDeviceDynamicRenderingLocalReadFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES, dynamicRenderingLocalRead,
             dynamicRenderingLocalRead, VK_API_VERSION_1_4),
    PROMOTED(VkPhysicalDeviceMaintenance5Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES,
             maintenance5, maintenance5, VK_API_VERSION_1_4),
    PROMOTED(VkPhysicalDeviceMaintenance6Features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES,
             maintenance6, maintenance6, VK_API_VERSION_1_4),
    PROMOTED(VkPhysicalDevicePipelineProtectedAccessFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_PROTECTED_ACCESS_FEATURES, pipelineProtectedAccess,
             pipelineProtectedAccess, VK_API_VERSION_1_4),
    PROMOTED(VkPhysicalDevicePipelineRobustnessFeatures,
             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES, pipelineRobustness, pipelineRobustness,
             VK_API_VERSION_1_4),
    PROMOTED(VkPhysicalDeviceHostImageCopyFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES,
             hostImageCopy, hostImageCopy, VK_API_VERSION_1_4),
};

CT_TEST(create_device_unsupported_promoted_feature,
        "Devices and Queues > Device Creation: if any requested feature is not supported, vkCreateDevice must "
        "return VK_ERROR_FEATURE_NOT_PRESENT (each unsupported member of the feature structs of extensions "
        "promoted to Vulkan 1.1-1.4, each struct alone in the pNext chain)",
        "vkCreateDevice vkGetPhysicalDeviceFeatures2")
{
    struct ct_env e;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo q = one_queue(0, &priority);
    size_t s, i, tested = 0;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_1, "vkGetPhysicalDeviceFeatures2");
    for (s = 0; s < CT_ARRAY_SIZE(promoted_features); s++) {
        const struct promoted_features *p = &promoted_features[s];
        size_t size = p->last + sizeof(VkBool32), count = (p->last - p->first) / sizeof(VkBool32) + 1;
        union {
            VkBaseOutStructure base;
            unsigned char bytes[256];
        } have, want;
        if (e.version < p->version)
            continue;
        query_feature(&e, &have, size, p->type, 0);
        for (i = 0; i < count; i++) {
            VkDevice device = VK_NULL_HANDLE;
            VkResult r;
            if (*(const VkBool32 *)(have.bytes + p->first + i * sizeof(VkBool32)))
                continue;
            tested++;
            memset(&want, 0, sizeof(want));
            want.base.sType = p->type;
            *(VkBool32 *)(want.bytes + p->first + i * sizeof(VkBool32)) = VK_TRUE;
            /* Valid usage: multiview{Geometry,Tessellation}Shader need multiview,
             * variablePointers needs variablePointersStorageBuffer (the first
             * member of each struct). */
            if ((p->type == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES ||
                 p->type == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES) && i > 0)
                *(VkBool32 *)(want.bytes + p->first) = VK_TRUE;
            r = create_device(&e, &q, 1, &want, NULL, NULL, 0, NULL, &device);
            CT_CHECKF(t, r == VK_ERROR_FEATURE_NOT_PRESENT, "%s member %zu: %s", p->name, i, ct_result_name(r));
            if (r == VK_SUCCESS)
                vkDestroyDevice(device, NULL);
        }
    }
    if (!tested)
        ct_skip(t, "every feature of the promoted feature structs is supported");
    ct_env_fini(&e);
}

CT_TEST(create_device_all_features,
        "Devices and Queues > Device Creation: vkCreateDevice succeeds when every feature "
        "vkGetPhysicalDeviceFeatures2 reports (VkPhysicalDeviceFeatures2 and VkPhysicalDeviceVulkan11-14Features) "
        "is enabled: requested features are verified to be supported, and supported ones can be enabled",
        "vkCreateDevice vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures vkDestroyDevice vkGetDeviceQueue")
{
    struct ct_env e;
    struct ct_features f;

    ct_env_init(t, &e);
    ct_get_features(&e, &f, 0);
    if (e.version < VK_API_VERSION_1_1) {
        VkPhysicalDeviceFeatures2 *f2 = &f.f2;
        /* No VkPhysicalDeviceFeatures2 below 1.1: use pEnabledFeatures. */
        float priority = 1.0f;
        VkDeviceQueueCreateInfo q = one_queue(0, &priority);
        VkDevice device = VK_NULL_HANDLE;
        CT_REQUIRE_VK(t, create_device(&e, &q, 1, NULL, &f2->features, NULL, 0, NULL, &device));
        vkDestroyDevice(device, NULL);
    } else {
        ct_env_device(t, &e, &f.f2);
    }
    ct_env_fini(&e);
}

CT_TEST(host_image_copy_layouts,
        "Devices and Queues > VkPhysicalDeviceHostImageCopyProperties: with pCopySrcLayouts / pCopyDstLayouts "
        "NULL the number of layouts is returned in copySrcLayoutCount / copyDstLayoutCount; otherwise the count is "
        "the array's size on input and the number of layouts written on output; VkPhysicalDeviceVulkan14Properties "
        "reports the same lists",
        "vkGetPhysicalDeviceProperties2")
{
    struct ct_env e;
    int which;

    ct_env_init(t, &e);
    require_version(t, &e, VK_API_VERSION_1_4, "VkPhysicalDeviceHostImageCopyProperties");
    for (which = 0; which < 2; which++) { /* source, then destination layouts */
        VkImageLayout full[257], part[257], v14[257];
        uint32_t total, n, k, i, j;
        VkPhysicalDeviceHostImageCopyProperties hic;
        VkPhysicalDeviceVulkan14Properties p14;
        VkPhysicalDeviceProperties2 p2;
        uint32_t *count = which ? &hic.copyDstLayoutCount : &hic.copySrcLayoutCount;
        VkImageLayout **layouts = which ? &hic.pCopyDstLayouts : &hic.pCopySrcLayouts;
        const char *what = which ? "copyDstLayoutCount" : "copySrcLayoutCount";

        memset(&hic, 0, sizeof(hic));
        hic.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES;
        memset(&p2, 0, sizeof(p2));
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &hic;
        vkGetPhysicalDeviceProperties2(e.pd, &p2);
        total = *count;
        CT_REQUIRE(t, total <= 256);
        if (!total)
            continue;

        /* Room for all, and one more. */
        memset(full, 0xA5, sizeof(full));
        *count = total + 1;
        *layouts = full;
        vkGetPhysicalDeviceProperties2(e.pd, &p2);
        CT_CHECKF(t, *count == total, "%s: room for %u of %u, %u returned", what, total + 1, total, *count);
        CT_CHECKF(t, ct_all_bytes(&full[total], sizeof(full[0]), 0xA5), "%s: wrote past the last layout", what);
        for (i = 0; i < total; i++)
            CT_CHECKF(t, !ct_all_bytes(&full[i], sizeof(full[i]), 0xA5), "%s: layout %u is not written", what, i);

        /* Fewer slots: that many written, nothing past them. */
        for (k = 0; k < total; k++) {
            memset(part, 0xA5, sizeof(part));
            *count = k;
            *layouts = part;
            vkGetPhysicalDeviceProperties2(e.pd, &p2);
            CT_CHECKF(t, *count == k, "%s: room for %u of %u, %u returned", what, k, total, *count);
            CT_CHECKF(t, ct_all_bytes(&part[k], sizeof(part[0]), 0xA5), "%s: room for %u, wrote layout %u", what, k,
                      k);
            for (i = 0; i < k; i++) {
                int found = 0;
                for (j = 0; j < total; j++)
                    found |= part[i] == full[j];
                CT_CHECKF(t, found, "%s: room for %u, layout %u (%d) is not in the full list", what, k, i,
                          (int)part[i]);
            }
        }

        /* The same list through VkPhysicalDeviceVulkan14Properties. */
        memset(&p14, 0, sizeof(p14));
        p14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES;
        memset(v14, 0xA5, sizeof(v14));
        if (which) {
            p14.copyDstLayoutCount = 256;
            p14.pCopyDstLayouts = v14;
        } else {
            p14.copySrcLayoutCount = 256;
            p14.pCopySrcLayouts = v14;
        }
        p2.pNext = &p14;
        vkGetPhysicalDeviceProperties2(e.pd, &p2);
        n = which ? p14.copyDstLayoutCount : p14.copySrcLayoutCount;
        CT_CHECKF(t, n == total, "%s: %u in VkPhysicalDeviceVulkan14Properties, %u in "
                  "VkPhysicalDeviceHostImageCopyProperties", what, n, total);
        for (i = 0; i < n && i < total; i++) {
            int found = 0;
            for (j = 0; j < total; j++)
                found |= v14[i] == full[j];
            CT_CHECKF(t, found, "%s: VkPhysicalDeviceVulkan14Properties lists layout %d, "
                      "VkPhysicalDeviceHostImageCopyProperties does not", what, (int)v14[i]);
        }
    }
    ct_env_fini(&e);
}

CT_TEST(create_device_unsupported_extension,
        "Devices and Queues > Device Creation: if any requested extension is not supported, vkCreateDevice must "
        "return VK_ERROR_EXTENSION_NOT_PRESENT",
        "vkCreateDevice")
{
    static const char *const bogus[] = { "VK_KHR_not_a_real_extension", "", "vk_khr_swapchain",
                                         VK_KHR_SURFACE_EXTENSION_NAME /* an instance extension */ };
    struct ct_env e;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo q = one_queue(0, &priority);
    size_t i;

    ct_env_init(t, &e);
    for (i = 0; i < CT_ARRAY_SIZE(bogus); i++) {
        VkDevice device = VK_NULL_HANDLE;
        VkResult r = create_device(&e, &q, 1, NULL, NULL, &bogus[i], 1, NULL, &device);
        CT_CHECKF(t, r == VK_ERROR_EXTENSION_NOT_PRESENT, "extension \"%s\": %s", bogus[i], ct_result_name(r));
        if (r == VK_SUCCESS)
            vkDestroyDevice(device, NULL);
    }
    ct_env_fini(&e);
}

CT_TEST(device_allocation_callbacks,
        "Memory Allocation > Host Memory: host memory a device needs comes from the allocator given to "
        "vkCreateDevice, following the callback rules, and is all freed by vkDestroyDevice",
        "vkCreateDevice vkDestroyDevice vkGetDeviceQueue")
{
    struct ct_env e;
    struct ct_allocator a;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo q = one_queue(0, &priority);
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkResult r;

    ct_env_init(t, &e);
    ct_allocator_init(&a, -1);
    CT_CALL_VK(&a, r, create_device(&e, &q, 1, NULL, NULL, NULL, 0, &a.cb, &device));
    CT_REQUIRE_VK(t, r);
    CT_CHECKF(t, a.total > 0, "vkCreateDevice made no allocation through pAllocator: a device needs host memory, "
              "which must come from the most specific allocator");
    CT_CALL(&a, vkGetDeviceQueue(device, 0, 0, &queue));
    CT_CHECK(t, queue != VK_NULL_HANDLE);
    CT_CALL(&a, vkDestroyDevice(device, &a.cb));
    CT_CHECKF(t, a.live == 0, "%ld allocations still live after vkDestroyDevice", a.live);
    ct_allocator_check(t, &a, "device");
    ct_allocator_fini(&a);
    ct_env_fini(&e);
}

CT_TEST(device_allocation_failure,
        "Memory Allocation > Host Memory: when pfnAllocation returns NULL, vkCreateDevice either continues or "
        "returns VK_ERROR_OUT_OF_HOST_MEMORY, and frees everything it allocated (failing every allocation from the "
        "n-th on, for each of the first 128 n and a spread of the rest)",
        "vkCreateDevice vkDestroyDevice")
{
    struct ct_env e;
    struct ct_allocator a;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo q = one_queue(0, &priority);
    VkDevice device = VK_NULL_HANDLE;
    long points[256];
    size_t k, n;
    VkResult r;

    ct_env_init(t, &e);
    ct_allocator_init(&a, -1);
    CT_CALL_VK(&a, r, create_device(&e, &q, 1, NULL, NULL, NULL, 0, &a.cb, &device));
    CT_REQUIRE_VK(t, r);
    n = ct_failure_points(a.total, points, CT_ARRAY_SIZE(points));
    CT_CALL(&a, vkDestroyDevice(device, &a.cb));
    ct_allocator_fini(&a);
    for (k = 0; k < n && !t->failures; k++) {
        ct_allocator_init(&a, points[k]);
        device = VK_NULL_HANDLE;
        CT_CALL_VK(&a, r, create_device(&e, &q, 1, NULL, NULL, NULL, 0, &a.cb, &device));
        CT_CHECKF(t, r == VK_SUCCESS || r == VK_ERROR_OUT_OF_HOST_MEMORY, "allocations from %ld on failing: %s",
                  points[k], ct_result_name(r));
        if (r == VK_SUCCESS)
            CT_CALL(&a, vkDestroyDevice(device, &a.cb));
        CT_CHECKF(t, a.live == 0, "allocations from %ld on failing: %ld allocations leaked", points[k], a.live);
        ct_allocator_check(t, &a, "device");
        ct_allocator_fini(&a);
    }
    /* The physical device still works. */
    ct_env_device(t, &e, NULL);
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(enumerate_physical_devices),
    CT_REF(enumerate_physical_devices_incomplete),
    CT_REF(enumerate_physical_device_groups),
    CT_REF(physical_device_properties),
    CT_REF(physical_device_properties_stable),
    CT_REF(properties2_core),
    CT_REF(properties_vulkan11),
    CT_REF(properties_vulkan12),
    CT_REF(properties_vulkan13),
    CT_REF(properties_vulkan14),
    CT_REF(features2_core),
    CT_REF(features_vulkan11),
    CT_REF(features_vulkan12),
    CT_REF(features_vulkan13),
    CT_REF(features_vulkan14),
    CT_REF(queue_families),
    CT_REF(queue_families_sparse_protected),
    CT_REF(queue_families_incomplete),
    CT_REF(queue_families2),
    CT_REF(queue_family_global_priority),
    CT_REF(create_device_every_family),
    CT_REF(create_device_all_families),
    CT_REF(create_device_twice),
    CT_REF(get_device_queue2),
    CT_REF(get_device_queue2_protected),
    CT_REF(destroy_device_null),
    CT_REF(create_device_unsupported_core_feature),
    CT_REF(create_device_unsupported_feature_chain),
    CT_REF(create_device_unsupported_promoted_feature),
    CT_REF(create_device_all_features),
    CT_REF(host_image_copy_layouts),
    CT_REF(create_device_unsupported_extension),
    CT_REF(device_allocation_callbacks),
    CT_REF(device_allocation_failure),
};

CT_SUITE(devsandqueues, tests)
