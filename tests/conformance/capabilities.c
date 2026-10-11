/* Chapter "Additional Capabilities": vkGetPhysicalDeviceImageFormatProperties
 * and its rules (extents, mip levels, layers, sample counts, resource size,
 * monotonicity, host image copy), the sparse image format query, and the
 * external memory, semaphore and fence handle type queries. */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

/* A spread of formats PlasmaVK and other drivers commonly support. */
static const VkFormat formats[] = {
    VK_FORMAT_R8_UNORM,           VK_FORMAT_R8G8B8A8_UNORM,     VK_FORMAT_R8G8B8A8_SRGB,  VK_FORMAT_B8G8R8A8_UNORM,
    VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_UINT,          VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_FORMAT_R5G6B5_UNORM_PACK16,
    VK_FORMAT_D16_UNORM,          VK_FORMAT_D32_SFLOAT,         VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT,
    VK_FORMAT_S8_UINT,            VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
    VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
};

static const VkImageUsageFlags usages[] = {
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT,          VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    VK_IMAGE_USAGE_SAMPLED_BIT,               VK_IMAGE_USAGE_STORAGE_BIT,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
};

static uint32_t full_chain(uint32_t w, uint32_t h, uint32_t d)
{
    uint32_t m = w > h ? w : h, n = 1;
    if (d > m)
        m = d;
    while (m > 1) {
        m >>= 1;
        n++;
    }
    return n;
}

static VkResult query(const struct ct_env *e, VkFormat f, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage,
                      VkImageCreateFlags flags, VkImageFormatProperties *p)
{
    memset(p, 0xA5, sizeof(*p));
    return vkGetPhysicalDeviceImageFormatProperties(e->pd, f, type, tiling, usage, flags, p);
}

CT_TEST(image_format_extents,
        "Capabilities > VkImageFormatProperties and Allowed Extent Values Based on Image Type: for optimal tiling, "
        "maxExtent is at least maxImageDimension1D/2D/Cube/3D with 1 in unused dimensions, maxMipLevels is the "
        "complete chain of maxExtent, maxArrayLayers is at least maxImageArrayLayers except for 3D, and "
        "maxResourceSize is at least 2^31",
        "vkGetPhysicalDeviceProperties vkGetPhysicalDeviceImageFormatProperties")
{
    static const struct {
        VkImageType type;
        VkImageCreateFlags flags;
    } kinds[] = {
        { VK_IMAGE_TYPE_1D, 0 },
        { VK_IMAGE_TYPE_2D, 0 },
        { VK_IMAGE_TYPE_2D, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT },
        { VK_IMAGE_TYPE_3D, 0 },
    };
    struct ct_env e;
    const VkPhysicalDeviceLimits *l;
    unsigned f, k, u, checked = 0;

    ct_env_init(t, &e);
    l = &e.props.limits;
    for (f = 0; f < CT_ARRAY_SIZE(formats); f++) {
        for (k = 0; k < CT_ARRAY_SIZE(kinds); k++) {
            for (u = 0; u < CT_ARRAY_SIZE(usages); u++) {
                VkImageFormatProperties p;
                uint32_t want_w, want_h, want_d;
                if (query(&e, formats[f], kinds[k].type, VK_IMAGE_TILING_OPTIMAL, usages[u], kinds[k].flags, &p) !=
                    VK_SUCCESS)
                    continue;
                checked++;
                want_w = kinds[k].type == VK_IMAGE_TYPE_1D ? l->maxImageDimension1D
                         : kinds[k].type == VK_IMAGE_TYPE_3D ? l->maxImageDimension3D
                         : kinds[k].flags                    ? l->maxImageDimensionCube
                                                             : l->maxImageDimension2D;
                want_h = kinds[k].type == VK_IMAGE_TYPE_1D ? 1 : want_w;
                want_d = kinds[k].type == VK_IMAGE_TYPE_3D ? want_w : 1;
                CT_CHECKF(t, p.maxExtent.width >= want_w && p.maxExtent.height >= want_h && p.maxExtent.depth >= want_d &&
                                 (kinds[k].type != VK_IMAGE_TYPE_1D || (p.maxExtent.height == 1 && p.maxExtent.depth == 1)) &&
                                 (kinds[k].type == VK_IMAGE_TYPE_3D || p.maxExtent.depth == 1),
                          "format %d type %d usage 0x%x: maxExtent %ux%ux%u", (int)formats[f], kinds[k].type, usages[u],
                          p.maxExtent.width, p.maxExtent.height, p.maxExtent.depth);
                CT_CHECKF(t, p.maxMipLevels == full_chain(p.maxExtent.width, p.maxExtent.height, p.maxExtent.depth),
                          "format %d type %d usage 0x%x: maxMipLevels %u for %ux%ux%u", (int)formats[f], kinds[k].type,
                          usages[u], p.maxMipLevels, p.maxExtent.width, p.maxExtent.height, p.maxExtent.depth);
                if (kinds[k].type != VK_IMAGE_TYPE_3D)
                    CT_CHECKF(t, p.maxArrayLayers >= l->maxImageArrayLayers, "format %d type %d: maxArrayLayers %u",
                              (int)formats[f], kinds[k].type, p.maxArrayLayers);
                else
                    CT_CHECK(t, p.maxArrayLayers >= 1);
                CT_CHECKF(t, p.maxResourceSize >= (1ull << 31), "maxResourceSize %llu",
                          (unsigned long long)p.maxResourceSize);
                CT_CHECK(t, p.sampleCounts & VK_SAMPLE_COUNT_1_BIT);
            }
        }
    }
    CT_CHECK(t, checked > 0);
    ct_env_fini(&e);
}

CT_TEST(image_format_sample_counts,
        "Capabilities > Supported Sample Counts: sampleCounts is VK_SAMPLE_COUNT_1_BIT for linear tiling, non-2D "
        "images, cube-compatible images and formats without attachment features; otherwise for each usage bit a "
        "superset of the matching VkPhysicalDeviceLimits sample counts, intersected",
        "vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceFormatProperties "
        "vkGetPhysicalDeviceFormatProperties2 vkGetPhysicalDeviceImageFormatProperties")
{
    struct ct_env e;
    struct ct_properties props;
    const VkPhysicalDeviceLimits *l;
    unsigned f, u;

    ct_env_init(t, &e);
    l = &e.props.limits;
    if (e.version >= VK_API_VERSION_1_2)
        ct_get_properties(&e, &props, 0);
    for (f = 0; f < CT_ARRAY_SIZE(formats); f++) {
        VkFormatFeatureFlags2 opt = ct_format_features(&e, formats[f], 0);
        int depth = formats[f] == VK_FORMAT_D16_UNORM || formats[f] == VK_FORMAT_D32_SFLOAT ||
                    formats[f] == VK_FORMAT_D24_UNORM_S8_UINT || formats[f] == VK_FORMAT_D32_SFLOAT_S8_UINT;
        int stencil = formats[f] == VK_FORMAT_S8_UINT || formats[f] == VK_FORMAT_D24_UNORM_S8_UINT ||
                      formats[f] == VK_FORMAT_D32_SFLOAT_S8_UINT;
        int integer = formats[f] == VK_FORMAT_R32_UINT;
        int attach = (opt & (VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT)) != 0;
        for (u = 0; u < CT_ARRAY_SIZE(usages); u++) {
            VkImageFormatProperties p;
            VkSampleCountFlags want = 0;
            int limited = 0;
            if (query(&e, formats[f], VK_IMAGE_TYPE_3D, VK_IMAGE_TILING_OPTIMAL, usages[u], 0, &p) == VK_SUCCESS)
                CT_CHECKF(t, p.sampleCounts == VK_SAMPLE_COUNT_1_BIT, "format %d 3D usage 0x%x: 0x%x", (int)formats[f],
                          usages[u], p.sampleCounts);
            if (query(&e, formats[f], VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_LINEAR, usages[u], 0, &p) == VK_SUCCESS)
                CT_CHECKF(t, p.sampleCounts == VK_SAMPLE_COUNT_1_BIT, "format %d linear usage 0x%x: 0x%x",
                          (int)formats[f], usages[u], p.sampleCounts);
            if (query(&e, formats[f], VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usages[u],
                      VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, &p) == VK_SUCCESS)
                CT_CHECKF(t, p.sampleCounts == VK_SAMPLE_COUNT_1_BIT, "format %d cube usage 0x%x: 0x%x", (int)formats[f],
                          usages[u], p.sampleCounts);
            if (query(&e, formats[f], VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usages[u], 0, &p) != VK_SUCCESS)
                continue;
            if (!attach) {
                CT_CHECKF(t, p.sampleCounts == VK_SAMPLE_COUNT_1_BIT, "format %d without attachment features: 0x%x",
                          (int)formats[f], p.sampleCounts);
                continue;
            }
            want = ~0u;
            if (usages[u] == VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
                limited = 1;
                want = integer ? (e.version >= VK_API_VERSION_1_2 ? props.v12.framebufferIntegerColorSampleCounts
                                                                  : VK_SAMPLE_COUNT_1_BIT)
                               : l->framebufferColorSampleCounts;
            } else if (usages[u] == VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                limited = 1;
                if (depth)
                    want &= l->framebufferDepthSampleCounts;
                if (stencil)
                    want &= l->framebufferStencilSampleCounts;
            } else if (usages[u] == VK_IMAGE_USAGE_SAMPLED_BIT) {
                limited = 1;
                if (depth)
                    want &= l->sampledImageDepthSampleCounts;
                else if (integer)
                    want &= l->sampledImageIntegerSampleCounts;
                else if (!stencil)
                    want &= l->sampledImageColorSampleCounts;
                else
                    limited = 0; /* stencil-only: no limit names it */
            } else if (usages[u] == VK_IMAGE_USAGE_STORAGE_BIT) {
                limited = 1;
                want = l->storageImageSampleCounts;
            }
            if (limited)
                CT_CHECKF(t, (p.sampleCounts & want) == want, "format %d usage 0x%x: sampleCounts 0x%x, limits say 0x%x",
                          (int)formats[f], usages[u], p.sampleCounts, want);
        }
    }
    ct_env_fini(&e);
}

CT_TEST(image_format_monotonic,
        "Capabilities > vkGetPhysicalDeviceImageFormatProperties: the limitations for usage1/flags1 must be no more "
        "strict than for usage2/flags2 when usage1 and flags1 are subsets of usage2 and flags2",
        "vkGetPhysicalDeviceImageFormatProperties")
{
    static const VkImageCreateFlags flag_sets[] = { 0, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
                                                    VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT };
    struct ct_env e;
    unsigned f, u, v, fl;
    ct_env_init(t, &e);
    for (f = 0; f < CT_ARRAY_SIZE(formats); f++) {
        for (fl = 0; fl < CT_ARRAY_SIZE(flag_sets); fl++) {
            for (u = 0; u < CT_ARRAY_SIZE(usages); u++) {
                for (v = 0; v < CT_ARRAY_SIZE(usages); v++) {
                    VkImageFormatProperties small, big;
                    VkImageUsageFlags u1 = usages[u], u2 = usages[u] | usages[v];
                    if (query(&e, formats[f], VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, u2, flag_sets[fl], &big) !=
                        VK_SUCCESS)
                        continue;
                    if (query(&e, formats[f], VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, u1, flag_sets[fl], &small) !=
                        VK_SUCCESS) {
                        CT_CHECKF(t, 0, "format %d: usage 0x%x flags 0x%x supported, usage 0x%x not", (int)formats[f], u2,
                                  flag_sets[fl], u1);
                        continue;
                    }
                    CT_CHECKF(t, small.maxExtent.width >= big.maxExtent.width &&
                                     small.maxExtent.height >= big.maxExtent.height &&
                                     small.maxExtent.depth >= big.maxExtent.depth && small.maxMipLevels >= big.maxMipLevels &&
                                     small.maxArrayLayers >= big.maxArrayLayers &&
                                     (small.sampleCounts & big.sampleCounts) == big.sampleCounts &&
                                     small.maxResourceSize >= big.maxResourceSize,
                              "format %d flags 0x%x: usage 0x%x is more limited than usage 0x%x", (int)formats[f],
                              flag_sets[fl], u1, u2);
                }
            }
        }
    }
    ct_env_fini(&e);
}

CT_TEST(image_format_host_transfer,
        "Capabilities > vkGetPhysicalDeviceImageFormatProperties: with hostImageCopy, for usages with SAMPLED and no "
        "sparse flags, adding VK_IMAGE_USAGE_HOST_TRANSFER_BIT must give identical results",
        "vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceImageFormatProperties")
{
    static const VkImageType types[] = { VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_3D };
    struct ct_env e;
    struct ct_features feats;
    unsigned f, u, k, tl;
    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_4)
        ct_skip(t, "hostImageCopy is a Vulkan 1.4 feature");
    ct_get_features(&e, &feats, 0);
    if (!feats.v14.hostImageCopy)
        ct_skip(t, "hostImageCopy is not supported");
    for (f = 0; f < CT_ARRAY_SIZE(formats); f++)
        for (k = 0; k < CT_ARRAY_SIZE(types); k++)
            for (tl = 0; tl < 2; tl++)
                for (u = 0; u < CT_ARRAY_SIZE(usages); u++) {
                    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | usages[u];
                    VkImageTiling tiling = tl ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
                    VkImageFormatProperties a, b;
                    VkResult ra = query(&e, formats[f], types[k], tiling, usage, 0, &a);
                    VkResult rb = query(&e, formats[f], types[k], tiling, usage | VK_IMAGE_USAGE_HOST_TRANSFER_BIT, 0, &b);
                    CT_CHECKF(t, ra == rb && (ra != VK_SUCCESS || memcmp(&a, &b, sizeof(a)) == 0),
                              "format %d type %d tiling %d usage 0x%x: %s / %s with HOST_TRANSFER", (int)formats[f],
                              types[k], tiling, usage, ct_result_name(ra), ct_result_name(rb));
                }
    ct_env_fini(&e);
}

CT_TEST(image_format_properties2,
        "Capabilities > vkGetPhysicalDeviceImageFormatProperties2: imageFormatProperties equals the result of "
        "vkGetPhysicalDeviceImageFormatProperties; without an external handle type VkExternalImageFormatProperties "
        "reports no external memory features; VkHostImageCopyDevicePerformanceQuery (with HOST_TRANSFER usage): identicalMemoryLayout "
        "implies optimalDeviceAccess",
        "vkGetPhysicalDeviceImageFormatProperties vkGetPhysicalDeviceImageFormatProperties2 vkGetPhysicalDeviceFeatures2")
{
    struct ct_env e;
    struct ct_features feats;
    unsigned f, u, host = 0;
    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_1)
        ct_skip(t, "vkGetPhysicalDeviceImageFormatProperties2 needs Vulkan 1.1");
    if (e.version >= VK_API_VERSION_1_4) {
        ct_get_features(&e, &feats, 0);
        host = feats.v14.hostImageCopy;
    }
    for (f = 0; f < CT_ARRAY_SIZE(formats); f++) {
        for (u = 0; u < CT_ARRAY_SIZE(usages) + host; u++) {
            /* the last round: SAMPLED | HOST_TRANSFER, with the performance query */
            VkImageUsageFlags usage = u < CT_ARRAY_SIZE(usages) ? usages[u]
                                                                : VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_HOST_TRANSFER_BIT;
            VkPhysicalDeviceImageFormatInfo2 info;
            VkPhysicalDeviceExternalImageFormatInfo ext_info;
            VkImageFormatProperties2 p2;
            VkExternalImageFormatProperties ext;
            VkHostImageCopyDevicePerformanceQuery perf;
            VkImageFormatProperties p1;
            VkResult r1, r2;

            memset(&ext_info, 0, sizeof(ext_info));
            ext_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
            memset(&info, 0, sizeof(info));
            info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
            info.pNext = &ext_info;
            info.format = formats[f];
            info.type = VK_IMAGE_TYPE_2D;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = usage;
            memset(&perf, 0xA5, sizeof(perf));
            perf.sType = VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY;
            perf.pNext = NULL;
            memset(&ext, 0xA5, sizeof(ext));
            ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;
            ext.pNext = (usage & VK_IMAGE_USAGE_HOST_TRANSFER_BIT) ? &perf : NULL;
            memset(&p2, 0xA5, sizeof(p2));
            p2.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
            p2.pNext = &ext;
            r1 = query(&e, formats[f], VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usage, 0, &p1);
            r2 = vkGetPhysicalDeviceImageFormatProperties2(e.pd, &info, &p2);
            CT_CHECKF(t, r1 == r2, "format %d usage 0x%x: %s / %s", (int)formats[f], usage, ct_result_name(r1),
                      ct_result_name(r2));
            if (r1 != VK_SUCCESS || r2 != VK_SUCCESS)
                continue;
            CT_CHECKF(t, memcmp(&p1, &p2.imageFormatProperties, sizeof(p1)) == 0, "format %d usage 0x%x: properties differ",
                      (int)formats[f], usage);
            CT_CHECK(t, ext.externalMemoryProperties.externalMemoryFeatures == 0);
            if (usage & VK_IMAGE_USAGE_HOST_TRANSFER_BIT)
                CT_CHECK(t, (perf.optimalDeviceAccess == VK_TRUE || perf.optimalDeviceAccess == VK_FALSE) &&
                                (perf.identicalMemoryLayout == VK_FALSE || perf.optimalDeviceAccess == VK_TRUE));
        }
    }
    ct_env_fini(&e);
}

CT_TEST(sparse_image_format_properties,
        "Sparse Resources > vkGetPhysicalDeviceSparseImageFormatProperties: if VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT "
        "is not supported for the arguments, pPropertyCount is zero (here: sparseResidencyImage2D not supported)",
        "vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceSparseImageFormatProperties "
        "vkGetPhysicalDeviceSparseImageFormatProperties2")
{
    struct ct_env e;
    struct ct_features f;
    uint32_t count = 77;
    ct_env_init(t, &e);
    ct_get_features(&e, &f, 0);
    if (f.f2.features.sparseResidencyImage2D)
        ct_skip(t, "sparseResidencyImage2D is supported");
    vkGetPhysicalDeviceSparseImageFormatProperties(e.pd, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D, VK_SAMPLE_COUNT_1_BIT,
                                                   VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_TILING_OPTIMAL, &count, NULL);
    CT_CHECK(t, count == 0);
    if (e.version >= VK_API_VERSION_1_1) {
        VkPhysicalDeviceSparseImageFormatInfo2 info;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2;
        info.format = VK_FORMAT_R8G8B8A8_UNORM;
        info.type = VK_IMAGE_TYPE_2D;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        count = 77;
        vkGetPhysicalDeviceSparseImageFormatProperties2(e.pd, &info, &count, NULL);
        CT_CHECK(t, count == 0);
    }
    ct_env_fini(&e);
}

CT_TEST(ycbcr_descriptor_count,
        "Capabilities > VkSamplerYcbcrConversionImageFormatProperties: for a supported format requiring a sampler "
        "Y'CbCr conversion, combinedImageSamplerDescriptorCount is at least 1 and, at Vulkan 1.4, at most "
        "maxCombinedImageSamplerDescriptorCount (Limits: the most any such format uses); the 2D sampled images of "
        "G8_B8_R8_3PLANE_420_UNORM and G8_B8R8_2PLANE_420_UNORM, required with samplerYcbcrConversion (Formats > "
        "Required Format Support), are supported",
        "vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceFormatProperties "
        "vkGetPhysicalDeviceFormatProperties2 vkGetPhysicalDeviceImageFormatProperties2")
{
    static VkFormat ycbcr[64];
    struct ct_env e;
    struct ct_properties props;
    VkPhysicalDeviceSamplerYcbcrConversionFeatures yf;
    VkPhysicalDeviceFeatures2 f2;
    size_t n, i;
    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_1)
        ct_skip(t, "sampler Y'CbCr conversion needs Vulkan 1.1");
    memset(&yf, 0, sizeof(yf));
    yf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES;
    memset(&f2, 0, sizeof(f2));
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &yf;
    vkGetPhysicalDeviceFeatures2(e.pd, &f2);
    if (!yf.samplerYcbcrConversion)
        ct_skip(t, "samplerYcbcrConversion is not supported");
    if (e.version >= VK_API_VERSION_1_4)
        ct_get_properties(&e, &props, 0);
    n = ct_ycbcr_formats(ycbcr, CT_ARRAY_SIZE(ycbcr));
    for (i = 0; i < n; i++) {
        VkPhysicalDeviceImageFormatInfo2 info;
        VkSamplerYcbcrConversionImageFormatProperties yp;
        VkImageFormatProperties2 p2;
        VkResult r;
        int required = ycbcr[i] == VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM || ycbcr[i] == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
        if (!(ct_format_features(&e, ycbcr[i], 0) & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT) && !required)
            continue;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
        info.format = ycbcr[i];
        info.type = VK_IMAGE_TYPE_2D;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        memset(&yp, 0xA5, sizeof(yp));
        yp.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES;
        yp.pNext = NULL;
        memset(&p2, 0, sizeof(p2));
        p2.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
        p2.pNext = &yp;
        r = vkGetPhysicalDeviceImageFormatProperties2(e.pd, &info, &p2);
        if (r != VK_SUCCESS) {
            CT_CHECKF(t, !required, "format %d: 2D sampled images: %s", (int)ycbcr[i], ct_result_name(r));
            continue;
        }
        CT_CHECKF(t, yp.combinedImageSamplerDescriptorCount >= 1, "format %d: combinedImageSamplerDescriptorCount %u",
                  (int)ycbcr[i], yp.combinedImageSamplerDescriptorCount);
        if (e.version >= VK_API_VERSION_1_4)
            CT_CHECKF(t, yp.combinedImageSamplerDescriptorCount <= props.v14.maxCombinedImageSamplerDescriptorCount,
                      "format %d: combinedImageSamplerDescriptorCount %u > maxCombinedImageSamplerDescriptorCount %u",
                      (int)ycbcr[i], yp.combinedImageSamplerDescriptorCount,
                      props.v14.maxCombinedImageSamplerDescriptorCount);
    }
    ct_env_fini(&e);
}

/* ---- External memory, semaphore and fence capabilities ---- */

/* A handle type bit no extension defines (bit 30; bit 31 is never a flag
 * bit, Fundamentals > Valid Usage for Flags). */
#define UNDEFINED_HANDLE_TYPE 0x40000000u

/* An instance and device of Vulkan 1.1 or later (timelineSemaphore enabled
 * from 1.2); *maintenance5: the feature is supported (Vulkan 1.4), which
 * makes any value valid in a physical device query (Fundamentals > Valid
 * Usage for Enumerated Types). */
static void external_env(struct ct *t, struct ct_env *e, int *maintenance5)
{
    struct ct_features f;
    VkPhysicalDeviceVulkan12Features v12;
    ct_env_init(t, e);
    if (e->version < VK_API_VERSION_1_1)
        ct_skip(t, "external handle capabilities are Vulkan 1.1");
    ct_get_features(e, &f, 0);
    *maintenance5 = e->version >= VK_API_VERSION_1_4 && f.v14.maintenance5;
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.timelineSemaphore = f.v12.timelineSemaphore;
    ct_env_device(t, e, e->version >= VK_API_VERSION_1_2 ? &v12 : NULL);
}

static void query_buffer(const struct ct_env *e, VkBufferUsageFlags usage, VkExternalMemoryHandleTypeFlagBits type,
                         unsigned char fill, VkExternalBufferProperties *p)
{
    VkPhysicalDeviceExternalBufferInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    info.usage = usage;
    info.handleType = type;
    memset(p, fill, sizeof(*p));
    p->sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    p->pNext = NULL;
    vkGetPhysicalDeviceExternalBufferProperties(e->pd, &info, p);
}

CT_TEST(external_buffer_properties,
        "Additional Capabilities > Additional Buffer Capabilities (vkGetPhysicalDeviceExternalBufferProperties, "
        "Vulkan 1.1): for each core memory handle type and several usages, the properties are written (the same "
        "whatever the output held before: queries are invariant, Fundamentals) with sType and pNext kept; a "
        "supported type's compatibleHandleTypes includes it (External Memory Handle Types: \"must include at "
        "least handleType\") and an exportable type can be given to VkExternalMemoryBufferCreateInfo; with "
        "maintenance5, a handle type no extension defines reports no features",
        "vkGetPhysicalDeviceExternalBufferProperties vkGetPhysicalDeviceFeatures2 vkCreateBuffer vkDestroyBuffer")
{
    static const VkBufferUsageFlags usage[4] = {
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
    };
    struct ct_env e;
    VkExternalBufferProperties a, b;
    uint32_t type, u;
    int maintenance5, supported = 0;

    external_env(t, &e, &maintenance5);
    for (type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT; type <= VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
         type <<= 1)
        for (u = 0; u < 4; u++) {
            const VkExternalMemoryProperties *m = &a.externalMemoryProperties;
            query_buffer(&e, usage[u], (VkExternalMemoryHandleTypeFlagBits)type, 0x00, &a);
            query_buffer(&e, usage[u], (VkExternalMemoryHandleTypeFlagBits)type, 0xff, &b);
            CT_CHECKF(t, a.sType == VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES && a.pNext == NULL,
                      "handle type 0x%x: sType or pNext changed", type);
            CT_CHECKF(t, memcmp(m, &b.externalMemoryProperties, sizeof(*m)) == 0,
                      "handle type 0x%x, usage 0x%x: the properties depend on what the structure held", type,
                      usage[u]);
            if (!(m->externalMemoryFeatures &
                  (VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT | VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)))
                continue;
            supported++;
            CT_CHECKF(t, m->compatibleHandleTypes & type, "handle type 0x%x: compatibleHandleTypes 0x%x lacks it",
                      type, m->compatibleHandleTypes);
            if (m->externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) {
                VkExternalMemoryBufferCreateInfo ext;
                VkBufferCreateInfo info;
                VkBuffer buffer = VK_NULL_HANDLE;
                memset(&ext, 0, sizeof(ext));
                ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
                ext.handleTypes = type;
                memset(&info, 0, sizeof(info));
                info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                info.pNext = &ext;
                info.size = 4096;
                info.usage = usage[u];
                CT_CHECK_VK(t, vkCreateBuffer(e.device, &info, NULL, &buffer), VK_SUCCESS);
                vkDestroyBuffer(e.device, buffer, NULL);
            }
        }
    if (maintenance5) {
        query_buffer(&e, usage[0], (VkExternalMemoryHandleTypeFlagBits)UNDEFINED_HANDLE_TYPE, 0xff, &a);
        CT_CHECKF(t, a.externalMemoryProperties.externalMemoryFeatures == 0,
                  "an undefined handle type: externalMemoryFeatures 0x%x", a.externalMemoryProperties.externalMemoryFeatures);
    }
    ct_note(t, "%d supported (handle type, usage) pairs", supported);
    ct_env_fini(&e);
}

static void query_semaphore(const struct ct_env *e, VkSemaphoreType stype, VkExternalSemaphoreHandleTypeFlagBits type,
                            unsigned char fill, VkExternalSemaphoreProperties *p)
{
    VkSemaphoreTypeCreateInfo type_info;
    VkPhysicalDeviceExternalSemaphoreInfo info;
    memset(&type_info, 0, sizeof(type_info));
    type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_info.semaphoreType = stype;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    info.pNext = e->version >= VK_API_VERSION_1_2 ? &type_info : NULL;
    info.handleType = type;
    memset(p, fill, sizeof(*p));
    p->sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
    p->pNext = NULL;
    vkGetPhysicalDeviceExternalSemaphoreProperties(e->pd, &info, p);
}

CT_TEST(external_semaphore_properties,
        "Additional Capabilities > Optional Semaphore Capabilities (vkGetPhysicalDeviceExternalSemaphoreProperties, "
        "Vulkan 1.1): for each core semaphore handle type, binary and (Vulkan 1.2) timeline, the properties are "
        "written (invariant whatever the output held) with sType and pNext kept; an exportable type can be given "
        "to VkExportSemaphoreCreateInfo (VUID-VkExportSemaphoreCreateInfo-handleTypes-01124); with maintenance5, a "
        "handle type no extension defines is unsupported: externalSemaphoreFeatures is zero",
        "vkGetPhysicalDeviceExternalSemaphoreProperties vkGetPhysicalDeviceFeatures2 vkCreateSemaphore "
        "vkDestroySemaphore")
{
    struct ct_env e;
    VkExternalSemaphoreProperties a, b;
    uint32_t type, k;
    int maintenance5, supported = 0;

    external_env(t, &e, &maintenance5);
    for (k = 0; k < (e.version >= VK_API_VERSION_1_2 ? 2u : 1u); k++) {
        VkSemaphoreType stype = k ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY;
        for (type = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT; type <= VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
             type <<= 1) {
            query_semaphore(&e, stype, (VkExternalSemaphoreHandleTypeFlagBits)type, 0x00, &a);
            query_semaphore(&e, stype, (VkExternalSemaphoreHandleTypeFlagBits)type, 0xff, &b);
            CT_CHECKF(t, a.sType == VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES && a.pNext == NULL,
                      "handle type 0x%x: sType or pNext changed", type);
            CT_CHECKF(t, a.exportFromImportedHandleTypes == b.exportFromImportedHandleTypes &&
                             a.compatibleHandleTypes == b.compatibleHandleTypes &&
                             a.externalSemaphoreFeatures == b.externalSemaphoreFeatures,
                      "%s, handle type 0x%x: the properties depend on what the structure held",
                      k ? "timeline" : "binary", type);
            if (a.externalSemaphoreFeatures)
                supported++;
            if (a.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) {
                VkSemaphoreTypeCreateInfo type_info;
                VkExportSemaphoreCreateInfo export_info;
                VkSemaphoreCreateInfo info;
                VkSemaphore s = VK_NULL_HANDLE;
                memset(&type_info, 0, sizeof(type_info));
                type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
                type_info.semaphoreType = stype;
                memset(&export_info, 0, sizeof(export_info));
                export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
                export_info.pNext = k ? &type_info : NULL;
                export_info.handleTypes = type;
                memset(&info, 0, sizeof(info));
                info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
                info.pNext = &export_info;
                CT_CHECK_VK(t, vkCreateSemaphore(e.device, &info, NULL, &s), VK_SUCCESS);
                vkDestroySemaphore(e.device, s, NULL);
            }
        }
    }
    if (maintenance5) {
        query_semaphore(&e, VK_SEMAPHORE_TYPE_BINARY, (VkExternalSemaphoreHandleTypeFlagBits)UNDEFINED_HANDLE_TYPE, 0xff,
                        &a);
        CT_CHECKF(t, a.externalSemaphoreFeatures == 0, "an undefined handle type: externalSemaphoreFeatures 0x%x",
                  a.externalSemaphoreFeatures);
    }
    ct_note(t, "%d supported (semaphore type, handle type) pairs", supported);
    ct_env_fini(&e);
}

static void query_fence(const struct ct_env *e, VkExternalFenceHandleTypeFlagBits type, unsigned char fill,
                        VkExternalFenceProperties *p)
{
    VkPhysicalDeviceExternalFenceInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO;
    info.handleType = type;
    memset(p, fill, sizeof(*p));
    p->sType = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES;
    p->pNext = NULL;
    vkGetPhysicalDeviceExternalFenceProperties(e->pd, &info, p);
}

CT_TEST(external_fence_properties,
        "Additional Capabilities > Optional Fence Capabilities (vkGetPhysicalDeviceExternalFenceProperties, Vulkan "
        "1.1): for each core fence handle type, the properties are written (invariant whatever the output held) "
        "with sType and pNext kept; an exportable type can be given to VkExportFenceCreateInfo "
        "(VUID-VkExportFenceCreateInfo-handleTypes-01446); with maintenance5, a handle type no extension defines "
        "is unsupported: externalFenceFeatures is zero",
        "vkGetPhysicalDeviceExternalFenceProperties vkGetPhysicalDeviceFeatures2 vkCreateFence vkDestroyFence")
{
    struct ct_env e;
    VkExternalFenceProperties a, b;
    uint32_t type;
    int maintenance5, supported = 0;

    external_env(t, &e, &maintenance5);
    for (type = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT; type <= VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
         type <<= 1) {
        query_fence(&e, (VkExternalFenceHandleTypeFlagBits)type, 0x00, &a);
        query_fence(&e, (VkExternalFenceHandleTypeFlagBits)type, 0xff, &b);
        CT_CHECKF(t, a.sType == VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES && a.pNext == NULL,
                  "handle type 0x%x: sType or pNext changed", type);
        CT_CHECKF(t, a.exportFromImportedHandleTypes == b.exportFromImportedHandleTypes &&
                         a.compatibleHandleTypes == b.compatibleHandleTypes &&
                         a.externalFenceFeatures == b.externalFenceFeatures,
                  "handle type 0x%x: the properties depend on what the structure held", type);
        if (a.externalFenceFeatures)
            supported++;
        if (a.externalFenceFeatures & VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT) {
            VkExportFenceCreateInfo export_info;
            VkFenceCreateInfo info;
            VkFence f = VK_NULL_HANDLE;
            memset(&export_info, 0, sizeof(export_info));
            export_info.sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO;
            export_info.handleTypes = type;
            memset(&info, 0, sizeof(info));
            info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            info.pNext = &export_info;
            CT_CHECK_VK(t, vkCreateFence(e.device, &info, NULL, &f), VK_SUCCESS);
            vkDestroyFence(e.device, f, NULL);
        }
    }
    if (maintenance5) {
        query_fence(&e, (VkExternalFenceHandleTypeFlagBits)UNDEFINED_HANDLE_TYPE, 0xff, &a);
        CT_CHECKF(t, a.externalFenceFeatures == 0, "an undefined handle type: externalFenceFeatures 0x%x",
                  a.externalFenceFeatures);
    }
    ct_note(t, "%d supported handle types", supported);
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(image_format_extents),       CT_REF(image_format_sample_counts), CT_REF(image_format_monotonic),
    CT_REF(image_format_host_transfer), CT_REF(image_format_properties2),   CT_REF(sparse_image_format_properties),
    CT_REF(ycbcr_descriptor_count),     CT_REF(external_buffer_properties), CT_REF(external_semaphore_properties),
    CT_REF(external_fence_properties),
};

CT_SUITE(capabilities, tests)
