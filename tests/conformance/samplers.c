/* Chapter "Samplers": sampler creation with every parameter the core API
 * allows (and the features some need), the maxSamplerAllocationCount
 * guarantee, and destruction. Filtering results are checked by the smoke
 * tests; here, valid samplers must be created. */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

static VkSamplerCreateInfo sampler_info(void)
{
    VkSamplerCreateInfo si;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.maxLod = VK_LOD_CLAMP_NONE;
    si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    return si;
}

static void create_destroy(struct ct *t, struct ct_env *e, const VkSamplerCreateInfo *si, const char *what)
{
    VkSampler s = VK_NULL_HANDLE;
    VkResult r = vkCreateSampler(e->device, si, NULL, &s);
    CT_CHECKF(t, r == VK_SUCCESS && s != VK_NULL_HANDLE, "%s: %s", what, ct_result_name(r));
    if (r == VK_SUCCESS)
        vkDestroySampler(e->device, s, NULL);
}

CT_TEST(create_sampler_parameters,
        "Samplers > VkSamplerCreateInfo: samplers with each filter, mipmap mode, address mode (mirror-clamp-to-edge "
        "with samplerMirrorClampToEdge), border color, compare operation, LOD range and LOD bias up to maxSamplerLodBias (not with VK_KHR_portability_subset) are "
        "created",
        "vkCreateSampler vkDestroySampler vkGetPhysicalDeviceFeatures2")
{
    static const VkSamplerAddressMode modes[] = {
        VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
    };
    static const VkBorderColor borders[] = {
        VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, VK_BORDER_COLOR_INT_TRANSPARENT_BLACK,
        VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,      VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,      VK_BORDER_COLOR_INT_OPAQUE_WHITE,
    };
    struct ct_env e;
    struct ct_features f;
    VkPhysicalDeviceVulkan12Features v12;
    VkSamplerCreateInfo si;
    unsigned i, mirror_clamp;
    int op;

    ct_env_init(t, &e);
    ct_get_features(&e, &f, 0);
    mirror_clamp = e.version >= VK_API_VERSION_1_2 && f.v12.samplerMirrorClampToEdge;
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.samplerMirrorClampToEdge = VK_TRUE;
    ct_env_device(t, &e, mirror_clamp ? &v12 : NULL);
    for (i = 0; i < 4; i++) {
        si = sampler_info();
        si.magFilter = (i & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        si.minFilter = (i & 2) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        si.mipmapMode = (i & 1) ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
        create_destroy(t, &e, &si, "filters");
    }
    for (i = 0; i < CT_ARRAY_SIZE(modes); i++) {
        if (modes[i] == VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE && !mirror_clamp)
            continue;
        si = sampler_info();
        si.addressModeU = modes[i];
        si.addressModeV = modes[(i + 1) % CT_ARRAY_SIZE(modes)] == VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE &&
                                  !mirror_clamp
                              ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                              : modes[(i + 1) % CT_ARRAY_SIZE(modes)];
        si.addressModeW = modes[i];
        create_destroy(t, &e, &si, "address modes");
    }
    for (i = 0; i < CT_ARRAY_SIZE(borders); i++) {
        si = sampler_info();
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.borderColor = borders[i];
        create_destroy(t, &e, &si, "border colors");
    }
    for (op = VK_COMPARE_OP_NEVER; op <= VK_COMPARE_OP_ALWAYS; op++) {
        si = sampler_info();
        si.compareEnable = VK_TRUE;
        si.compareOp = (VkCompareOp)op;
        create_destroy(t, &e, &si, "compare");
    }
    si = sampler_info();
    si.minLod = 1.5f;
    si.maxLod = 3.25f;
    create_destroy(t, &e, &si, "LOD range");
    if (!e.portability_subset) { /* samplerMipLodBias may be VK_FALSE (VUID-VkSamplerCreateInfo-samplerMipLodBias-04467) */
        si.mipLodBias = e.props.limits.maxSamplerLodBias;
        create_destroy(t, &e, &si, "largest bias");
        si.mipLodBias = -e.props.limits.maxSamplerLodBias;
        create_destroy(t, &e, &si, "most negative bias");
    }
    ct_env_fini(&e);
}

CT_TEST(create_sampler_anisotropy,
        "Samplers > VkSamplerCreateInfo: with samplerAnisotropy enabled, samplers with maxAnisotropy from 1 to "
        "VkPhysicalDeviceLimits::maxSamplerAnisotropy are created (VUID-VkSamplerCreateInfo-anisotropyEnable-01071)",
        "vkCreateSampler vkDestroySampler vkGetPhysicalDeviceFeatures2")
{
    struct ct_env e;
    struct ct_features f;
    VkPhysicalDeviceFeatures2 enable;
    VkSamplerCreateInfo si;
    float a;

    ct_env_init(t, &e);
    ct_get_features(&e, &f, 0);
    if (!f.f2.features.samplerAnisotropy)
        ct_skip(t, "samplerAnisotropy is not supported");
    memset(&enable, 0, sizeof(enable));
    enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    enable.features.samplerAnisotropy = VK_TRUE;
    ct_env_device(t, &e, &enable);
    for (a = 1.0f; a <= e.props.limits.maxSamplerAnisotropy; a *= 2.0f) {
        si = sampler_info();
        si.anisotropyEnable = VK_TRUE;
        si.maxAnisotropy = a;
        create_destroy(t, &e, &si, "anisotropy");
    }
    si = sampler_info();
    si.anisotropyEnable = VK_TRUE;
    si.maxAnisotropy = e.props.limits.maxSamplerAnisotropy;
    create_destroy(t, &e, &si, "largest anisotropy");
    ct_env_fini(&e);
}

CT_TEST(create_sampler_unnormalized,
        "Samplers > VkSamplerCreateInfo: a sampler with unnormalizedCoordinates (equal filters, nearest mipmap "
        "mode, LODs 0, clamp-to-edge or border addressing, no anisotropy or comparison) is created",
        "vkCreateSampler vkDestroySampler")
{
    struct ct_env e;
    VkSamplerCreateInfo si;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    si = sampler_info();
    si.unnormalizedCoordinates = VK_TRUE;
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.minLod = si.maxLod = 0.0f;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    create_destroy(t, &e, &si, "unnormalized, linear");
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    create_destroy(t, &e, &si, "unnormalized, nearest, border");
    ct_env_fini(&e);
}

CT_TEST(create_sampler_reduction,
        "Samplers > VkSamplerReductionModeCreateInfo: with samplerFilterMinmax enabled, min and max reduction "
        "samplers are created (Vulkan 1.2 or VK_EXT_sampler_filter_minmax)",
        "vkCreateSampler vkDestroySampler vkGetPhysicalDeviceFeatures2")
{
    struct ct_env e;
    struct ct_features f;
    VkPhysicalDeviceVulkan12Features v12;
    VkSamplerReductionModeCreateInfo red;
    VkSamplerCreateInfo si;
    int m;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_2)
        ct_skip(t, "samplerFilterMinmax is a Vulkan 1.2 feature");
    ct_get_features(&e, &f, 0);
    if (!f.v12.samplerFilterMinmax)
        ct_skip(t, "samplerFilterMinmax is not supported");
    memset(&v12, 0, sizeof(v12));
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.samplerFilterMinmax = VK_TRUE;
    ct_env_device(t, &e, &v12);
    for (m = VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE; m <= VK_SAMPLER_REDUCTION_MODE_MAX; m++) {
        memset(&red, 0, sizeof(red));
        red.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO;
        red.reductionMode = (VkSamplerReductionMode)m;
        si = sampler_info();
        si.pNext = &red;
        create_destroy(t, &e, &si, "reduction");
    }
    ct_env_fini(&e);
}

CT_TEST(sampler_allocation_count,
        "Limits > maxSamplerAllocationCount: at least 4000 samplers can exist at once (here 4000 distinct and "
        "4000 equal ones)",
        "vkCreateSampler vkDestroySampler")
{
    enum { N = 4000 };
    struct ct_env e;
    VkSampler *s = calloc(N, sizeof(*s));
    VkSamplerCreateInfo si;
    unsigned i, made = 0, round;

    CT_REQUIRE(t, s != NULL);
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    CT_CHECK(t, e.props.limits.maxSamplerAllocationCount >= N);
    for (round = 0; round < 2; round++) {
        for (made = 0, i = 0; i < N; i++) {
            si = sampler_info();
            if (round == 0) { /* distinct: LOD clamps of quarter steps */
                si.minLod = (float)(i % 64) * 0.25f;
                si.maxLod = si.minLod + (float)(i / 64) * 0.25f;
            }
            if (vkCreateSampler(e.device, &si, NULL, &s[i]) != VK_SUCCESS)
                break;
            made++;
        }
        CT_CHECKF(t, made == N, "%s samplers: created %u of %u", round ? "equal" : "distinct", made, N);
        for (i = 0; i < made; i++)
            vkDestroySampler(e.device, s[i], NULL);
    }
    free(s);
    ct_env_fini(&e);
}

CT_TEST(destroy_sampler_null,
        "Samplers > vkDestroySampler: sampler may be VK_NULL_HANDLE (VUID-vkDestroySampler-sampler-parameter: "
        "must be a valid handle if not VK_NULL_HANDLE)",
        "vkDestroySampler")
{
    struct ct_env e;
    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    vkDestroySampler(e.device, VK_NULL_HANDLE, NULL);
    ct_env_fini(&e);
}

/* ---- Sampler Y'CbCr conversion ---- */

static VkSamplerYcbcrConversionCreateInfo conversion_info(VkFormat format)
{
    VkSamplerYcbcrConversionCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO;
    ci.format = format;
    ci.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
    ci.ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
    ci.xChromaOffset = ci.yChromaOffset = VK_CHROMA_LOCATION_MIDPOINT;
    ci.chromaFilter = VK_FILTER_NEAREST;
    return ci;
}

/* A sampler valid with a Y'CbCr conversion (VUID-VkSamplerCreateInfo-addressModeU-01646, -minFilter-01645):
 * clamp to edge, no anisotropy, normalized coordinates, filters of the
 * conversion's chroma filter unless separate filters are supported. */
static VkSamplerCreateInfo ycbcr_sampler_info(const VkSamplerYcbcrConversionInfo *ci, VkFilter filter)
{
    VkSamplerCreateInfo si = sampler_info();
    si.pNext = ci;
    si.magFilter = si.minFilter = filter;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    return si;
}

CT_TEST(create_ycbcr_conversion,
        "Samplers > Sampler Y'CbCr Conversion: with samplerYcbcrConversion, conversions of G8_B8_R8_3PLANE_420_UNORM "
        "with every model and range, the chroma locations and filters, and forceExplicitReconstruction its format "
        "features allow (VUID-VkSamplerYcbcrConversionCreateInfo-xChromaOffset-01651, -chromaFilter-01657, "
        "-forceExplicitReconstruction-01656) are created, and samplers, image views and descriptor set layouts "
        "(immutable samplers) made with them; vkDestroySamplerYcbcrConversion accepts VK_NULL_HANDLE",
        "vkCreateSamplerYcbcrConversion vkDestroySamplerYcbcrConversion vkCreateSampler vkDestroySampler "
        "vkCreateImageView vkDestroyImageView vkCreateDescriptorSetLayout vkDestroyDescriptorSetLayout "
        "vkGetDescriptorSetLayoutSupport vkCreateImage vkDestroyImage vkGetImageMemoryRequirements vkAllocateMemory "
        "vkBindImageMemory vkFreeMemory vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFormatProperties "
        "vkGetPhysicalDeviceFormatProperties2")
{
    static const VkSamplerYcbcrModelConversion models[] = {
        VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY, VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY,
        VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709,    VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601,
        VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020,
    };
    const VkFormat format = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
    struct ct_env e;
    struct ct_image img;
    VkImageCreateInfo ii;
    VkFormatFeatureFlags2 ff;
    unsigned m, r, x, y, f, fe, n = 0;

    ct_env_init(t, &e);
    ct_env_device_ycbcr(t, &e);
    ff = ct_format_features(&e, format, 0);
    CT_REQUIREF(t, ff & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT, "G8_B8_R8_3PLANE_420_UNORM cannot be sampled");
    ii = ct_image_info(VK_IMAGE_TYPE_2D, format, 16, 8, 1, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT);
    ct_image(t, &e, &ii, 0, &img);
    for (m = 0; m < CT_ARRAY_SIZE(models); m++)
        for (r = 0; r < 2; r++)
            for (x = 0; x < 2; x++)
                for (y = 0; y < 2; y++)
                    for (f = 0; f < 2; f++)
                        for (fe = 0; fe < 2; fe++) {
                            VkSamplerYcbcrConversionCreateInfo ci = conversion_info(format);
                            VkSamplerYcbcrConversion conv = VK_NULL_HANDLE;
                            VkSamplerYcbcrConversionInfo info;
                            VkSamplerCreateInfo si;
                            VkSampler sampler = VK_NULL_HANDLE;
                            VkImageViewCreateInfo vi;
                            VkImageView view = VK_NULL_HANDLE;
                            VkDescriptorSetLayoutBinding b;
                            VkDescriptorSetLayoutCreateInfo li;
                            VkDescriptorSetLayoutSupport support;
                            VkDescriptorSetLayout layout = VK_NULL_HANDLE;
                            VkFormatFeatureFlags2 loc = x ? VK_FORMAT_FEATURE_2_COSITED_CHROMA_SAMPLES_BIT
                                                          : VK_FORMAT_FEATURE_2_MIDPOINT_CHROMA_SAMPLES_BIT;
                            VkFormatFeatureFlags2 yloc = y ? VK_FORMAT_FEATURE_2_COSITED_CHROMA_SAMPLES_BIT
                                                           : VK_FORMAT_FEATURE_2_MIDPOINT_CHROMA_SAMPLES_BIT;
                            VkResult res;
                            if (!(ff & loc) || !(ff & yloc) ||
                                (f && !(ff & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT)) ||
                                (f && !(ff & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) ||
                                (fe && !(ff & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT)))
                                continue;
                            ci.ycbcrModel = models[m];
                            ci.ycbcrRange = r ? VK_SAMPLER_YCBCR_RANGE_ITU_FULL : VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
                            ci.xChromaOffset = x ? VK_CHROMA_LOCATION_COSITED_EVEN : VK_CHROMA_LOCATION_MIDPOINT;
                            ci.yChromaOffset = y ? VK_CHROMA_LOCATION_COSITED_EVEN : VK_CHROMA_LOCATION_MIDPOINT;
                            ci.chromaFilter = f ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
                            ci.forceExplicitReconstruction = fe;
                            res = vkCreateSamplerYcbcrConversion(e.device, &ci, NULL, &conv);
                            CT_CHECKF(t, res == VK_SUCCESS && conv, "model %d range %u locations %u/%u filter %u explicit %u: %s",
                                      (int)models[m], r, x, y, f, fe, ct_result_name(res));
                            if (res != VK_SUCCESS)
                                continue;
                            n++;
                            memset(&info, 0, sizeof(info));
                            info.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
                            info.conversion = conv;
                            si = ycbcr_sampler_info(&info, ci.chromaFilter);
                            CT_CHECK_VK(t, vkCreateSampler(e.device, &si, NULL, &sampler), VK_SUCCESS);
                            memset(&vi, 0, sizeof(vi));
                            vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                            vi.pNext = &info;
                            vi.image = img.image;
                            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
                            vi.format = format;
                            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                            vi.subresourceRange.levelCount = 1;
                            vi.subresourceRange.layerCount = 1;
                            CT_CHECK_VK(t, vkCreateImageView(e.device, &vi, NULL, &view), VK_SUCCESS);
                            memset(&b, 0, sizeof(b));
                            b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                            b.descriptorCount = 1;
                            b.stageFlags = VK_SHADER_STAGE_ALL;
                            b.pImmutableSamplers = &sampler;
                            memset(&li, 0, sizeof(li));
                            li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                            li.bindingCount = sampler ? 1 : 0;
                            li.pBindings = &b;
                            memset(&support, 0, sizeof(support));
                            support.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT;
                            vkGetDescriptorSetLayoutSupport(e.device, &li, &support);
                            CT_CHECK(t, support.supported == VK_TRUE);
                            CT_CHECK_VK(t, vkCreateDescriptorSetLayout(e.device, &li, NULL, &layout), VK_SUCCESS);
                            if (layout)
                                vkDestroyDescriptorSetLayout(e.device, layout, NULL);
                            if (view)
                                vkDestroyImageView(e.device, view, NULL);
                            if (sampler)
                                vkDestroySampler(e.device, sampler, NULL);
                            vkDestroySamplerYcbcrConversion(e.device, conv, NULL);
                        }
    CT_CHECKF(t, n >= 10, "only %u conversions made", n);
    vkDestroySamplerYcbcrConversion(e.device, VK_NULL_HANDLE, NULL);
    ct_image_fini(&e, &img);
    ct_env_fini(&e);
}

/*
 * #version 450
 * layout(local_size_x = 64) in;
 * layout(set = 0, binding = 0) uniform sampler2D ycbcr;
 * layout(std430, set = 0, binding = 1) writeonly buffer Results { vec4 results[]; };
 * layout(push_constant) uniform Push { uint width; uint height; };
 * void main()
 * {
 *     uint i = gl_GlobalInvocationID.x;
 *     if (i >= width * height)
 *         return;
 *     vec2 c = (vec2(i % width, i / width) + 0.5) / vec2(width, height);
 *     results[i] = textureLod(ycbcr, c, 0.0);
 * }
 */
static const uint32_t spv_texel_centers[] = {
    0x07230203,0x00010000,0x0008000b,0x0000004f,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0006000f,0x00000005,0x00000004,0x6e69616d,0x00000000,0x0000000b,0x00060010,0x00000004,
    0x00000011,0x00000040,0x00000001,0x00000001,0x00030003,0x00000002,0x000001c2,0x00040005,
    0x00000004,0x6e69616d,0x00000000,0x00030005,0x00000008,0x00000069,0x00080005,0x0000000b,
    0x475f6c67,0x61626f6c,0x766e496c,0x7461636f,0x496e6f69,0x00000044,0x00040005,0x00000011,
    0x68737550,0x00000000,0x00050006,0x00000011,0x00000000,0x74646977,0x00000068,0x00050006,
    0x00000011,0x00000001,0x67696568,0x00007468,0x00030005,0x00000013,0x00000000,0x00030005,
    0x00000025,0x00000063,0x00040005,0x0000003e,0x75736552,0x0073746c,0x00050006,0x0000003e,
    0x00000000,0x75736572,0x0073746c,0x00030005,0x00000040,0x00000000,0x00040005,0x00000045,
    0x63626379,0x00000072,0x00040047,0x0000000b,0x0000000b,0x0000001c,0x00030047,0x00000011,
    0x00000002,0x00050048,0x00000011,0x00000000,0x00000023,0x00000000,0x00050048,0x00000011,
    0x00000001,0x00000023,0x00000004,0x00040047,0x0000003d,0x00000006,0x00000010,0x00030047,
    0x0000003e,0x00000003,0x00040048,0x0000003e,0x00000000,0x00000019,0x00050048,0x0000003e,
    0x00000000,0x00000023,0x00000000,0x00030047,0x00000040,0x00000019,0x00040047,0x00000040,
    0x00000021,0x00000001,0x00040047,0x00000040,0x00000022,0x00000000,0x00040047,0x00000045,
    0x00000021,0x00000000,0x00040047,0x00000045,0x00000022,0x00000000,0x00040047,0x0000004e,
    0x0000000b,0x00000019,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00040015,
    0x00000006,0x00000020,0x00000000,0x00040020,0x00000007,0x00000007,0x00000006,0x00040017,
    0x00000009,0x00000006,0x00000003,0x00040020,0x0000000a,0x00000001,0x00000009,0x0004003b,
    0x0000000a,0x0000000b,0x00000001,0x0004002b,0x00000006,0x0000000c,0x00000000,0x00040020,
    0x0000000d,0x00000001,0x00000006,0x0004001e,0x00000011,0x00000006,0x00000006,0x00040020,
    0x00000012,0x00000009,0x00000011,0x0004003b,0x00000012,0x00000013,0x00000009,0x00040015,
    0x00000014,0x00000020,0x00000001,0x0004002b,0x00000014,0x00000015,0x00000000,0x00040020,
    0x00000016,0x00000009,0x00000006,0x0004002b,0x00000014,0x00000019,0x00000001,0x00020014,
    0x0000001d,0x00030016,0x00000022,0x00000020,0x00040017,0x00000023,0x00000022,0x00000002,
    0x00040020,0x00000024,0x00000007,0x00000023,0x0004002b,0x00000022,0x00000031,0x3f000000,
    0x00040017,0x0000003c,0x00000022,0x00000004,0x0003001d,0x0000003d,0x0000003c,0x0003001e,
    0x0000003e,0x0000003d,0x00040020,0x0000003f,0x00000002,0x0000003e,0x0004003b,0x0000003f,
    0x00000040,0x00000002,0x00090019,0x00000042,0x00000022,0x00000001,0x00000000,0x00000000,
    0x00000000,0x00000001,0x00000000,0x0003001b,0x00000043,0x00000042,0x00040020,0x00000044,
    0x00000000,0x00000043,0x0004003b,0x00000044,0x00000045,0x00000000,0x0004002b,0x00000022,
    0x00000048,0x00000000,0x00040020,0x0000004a,0x00000002,0x0000003c,0x0004002b,0x00000006,
    0x0000004c,0x00000040,0x0004002b,0x00000006,0x0000004d,0x00000001,0x0006002c,0x00000009,
    0x0000004e,0x0000004c,0x0000004d,0x0000004d,0x00050036,0x00000002,0x00000004,0x00000000,
    0x00000003,0x000200f8,0x00000005,0x0004003b,0x00000007,0x00000008,0x00000007,0x0004003b,
    0x00000024,0x00000025,0x00000007,0x00050041,0x0000000d,0x0000000e,0x0000000b,0x0000000c,
    0x0004003d,0x00000006,0x0000000f,0x0000000e,0x0003003e,0x00000008,0x0000000f,0x0004003d,
    0x00000006,0x00000010,0x00000008,0x00050041,0x00000016,0x00000017,0x00000013,0x00000015,
    0x0004003d,0x00000006,0x00000018,0x00000017,0x00050041,0x00000016,0x0000001a,0x00000013,
    0x00000019,0x0004003d,0x00000006,0x0000001b,0x0000001a,0x00050084,0x00000006,0x0000001c,
    0x00000018,0x0000001b,0x000500ae,0x0000001d,0x0000001e,0x00000010,0x0000001c,0x000300f7,
    0x00000020,0x00000000,0x000400fa,0x0000001e,0x0000001f,0x00000020,0x000200f8,0x0000001f,
    0x000100fd,0x000200f8,0x00000020,0x0004003d,0x00000006,0x00000026,0x00000008,0x00050041,
    0x00000016,0x00000027,0x00000013,0x00000015,0x0004003d,0x00000006,0x00000028,0x00000027,
    0x00050089,0x00000006,0x00000029,0x00000026,0x00000028,0x00040070,0x00000022,0x0000002a,
    0x00000029,0x0004003d,0x00000006,0x0000002b,0x00000008,0x00050041,0x00000016,0x0000002c,
    0x00000013,0x00000015,0x0004003d,0x00000006,0x0000002d,0x0000002c,0x00050086,0x00000006,
    0x0000002e,0x0000002b,0x0000002d,0x00040070,0x00000022,0x0000002f,0x0000002e,0x00050050,
    0x00000023,0x00000030,0x0000002a,0x0000002f,0x00050050,0x00000023,0x00000032,0x00000031,
    0x00000031,0x00050081,0x00000023,0x00000033,0x00000030,0x00000032,0x00050041,0x00000016,
    0x00000034,0x00000013,0x00000015,0x0004003d,0x00000006,0x00000035,0x00000034,0x00040070,
    0x00000022,0x00000036,0x00000035,0x00050041,0x00000016,0x00000037,0x00000013,0x00000019,
    0x0004003d,0x00000006,0x00000038,0x00000037,0x00040070,0x00000022,0x00000039,0x00000038,
    0x00050050,0x00000023,0x0000003a,0x00000036,0x00000039,0x00050088,0x00000023,0x0000003b,
    0x00000033,0x0000003a,0x0003003e,0x00000025,0x0000003b,0x0004003d,0x00000006,0x00000041,
    0x00000008,0x0004003d,0x00000043,0x00000046,0x00000045,0x0004003d,0x00000023,0x00000047,
    0x00000025,0x00070058,0x0000003c,0x00000049,0x00000046,0x00000047,0x00000002,0x00000048,
    0x00060041,0x0000004a,0x0000004b,0x00000040,0x00000015,0x00000041,0x0003003e,0x0000004b,
    0x00000049,0x000100fd,0x00010038,
};

/* Samples every texel center of `view` (W x H) through the immutable
 * `sampler` with the shader above into out (RGBA floats). */
static void sample_centers(struct ct *t, struct ct_env *e, VkImageView view, VkSampler sampler, uint32_t descriptors,
                           uint32_t w, uint32_t h, float *out)
{
    VkDescriptorSetLayoutBinding b[2];
    VkDescriptorSetLayoutCreateInfo li;
    VkPushConstantRange pcr;
    VkPipelineLayoutCreateInfo pli;
    VkShaderModuleCreateInfo smi;
    VkComputePipelineCreateInfo cpi;
    VkDescriptorPoolSize sizes[2];
    VkDescriptorPoolCreateInfo dpi;
    VkDescriptorSetAllocateInfo dai;
    VkDescriptorImageInfo ii;
    VkDescriptorBufferInfo bi;
    VkWriteDescriptorSet wr[2];
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandBuffer cb;
    struct ct_buffer results;
    uint32_t push[2];

    push[0] = w;
    push[1] = h;
    memset(b, 0, sizeof(b));
    b[0].binding = 0;
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[0].descriptorCount = 1;
    b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[0].pImmutableSamplers = &sampler;
    b[1].binding = 1;
    b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[1].descriptorCount = 1;
    b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    memset(&li, 0, sizeof(li));
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 2;
    li.pBindings = b;
    CT_REQUIRE_VK(t, vkCreateDescriptorSetLayout(e->device, &li, NULL, &set_layout));
    memset(&pcr, 0, sizeof(pcr));
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size = sizeof(push);
    memset(&pli, 0, sizeof(pli));
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    CT_REQUIRE_VK(t, vkCreatePipelineLayout(e->device, &pli, NULL, &layout));
    memset(&smi, 0, sizeof(smi));
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = sizeof(spv_texel_centers);
    smi.pCode = spv_texel_centers;
    CT_REQUIRE_VK(t, vkCreateShaderModule(e->device, &smi, NULL, &module));
    memset(&cpi, 0, sizeof(cpi));
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = module;
    cpi.stage.pName = "main";
    cpi.layout = layout;
    CT_REQUIRE_VK(t, vkCreateComputePipelines(e->device, VK_NULL_HANDLE, 1, &cpi, NULL, &pipeline));
    /* Descriptors.. combinedImageSamplerDescriptorCount of them for the one
     * combined image sampler (Descriptor Pools: VkDescriptorPoolSize) */
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = descriptors;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[1].descriptorCount = 1;
    memset(&dpi, 0, sizeof(dpi));
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = sizes;
    CT_REQUIRE_VK(t, vkCreateDescriptorPool(e->device, &dpi, NULL, &pool));
    memset(&dai, 0, sizeof(dai));
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &set_layout;
    CT_REQUIRE_VK(t, vkAllocateDescriptorSets(e->device, &dai, &set));
    ct_host_buffer(t, e, (VkDeviceSize)w * h * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &results);
    memset(&ii, 0, sizeof(ii));
    ii.imageView = view;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bi.buffer = results.buffer;
    bi.offset = 0;
    bi.range = VK_WHOLE_SIZE;
    memset(wr, 0, sizeof(wr));
    wr[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr[0].dstSet = set;
    wr[0].dstBinding = 0;
    wr[0].descriptorCount = 1;
    wr[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[0].pImageInfo = &ii;
    wr[1] = wr[0];
    wr[1].dstBinding = 1;
    wr[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[1].pImageInfo = NULL;
    wr[1].pBufferInfo = &bi;
    vkUpdateDescriptorSets(e->device, 2, wr, 0, NULL);
    cb = ct_begin_commands(t, e);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    vkCmdDispatch(cb, (w * h + 63) / 64, 1, 1);
    ct_host_read_barrier(cb);
    ct_submit_and_wait(t, e, cb);
    memcpy(out, results.data, (size_t)w * h * 16);
    ct_buffer_fini(e, &results);
    vkDestroyDescriptorPool(e->device, pool, NULL);
    vkDestroyPipeline(e->device, pipeline, NULL);
    vkDestroyShaderModule(e->device, module, NULL);
    vkDestroyPipelineLayout(e->device, layout, NULL);
    vkDestroyDescriptorSetLayout(e->device, set_layout, NULL);
}

CT_TEST(ycbcr_sample_texels,
        "Samplers > Sampler Y'CbCr Conversion and Textures > Sampler Y'CbCr Conversion: sampling a "
        "G8_B8_R8_3PLANE_420_UNORM image at its texel centers with nearest filters and midpoint chroma (or "
        "forced explicit reconstruction) gives each luma texel (i, j) the chroma of texel (i/2, j/2) (Chroma "
        "Reconstruction); RGB_IDENTITY returns those values unmodified, YCbCr models range-expand (ITU full and "
        "narrow) and convert them by their matrices (BT.709, BT.601: Sampler Y'CbCr Model Conversion), alpha 1",
        "vkCreateSamplerYcbcrConversion vkDestroySamplerYcbcrConversion vkCreateSampler vkDestroySampler "
        "vkCreateImageView vkDestroyImageView vkCreateDescriptorSetLayout vkDestroyDescriptorSetLayout "
        "vkCreatePipelineLayout vkDestroyPipelineLayout vkCreateShaderModule vkDestroyShaderModule "
        "vkCreateComputePipelines vkDestroyPipeline vkCreateDescriptorPool vkDestroyDescriptorPool "
        "vkAllocateDescriptorSets vkUpdateDescriptorSets vkCmdBindPipeline vkCmdBindDescriptorSets vkCmdPushConstants "
        "vkCmdDispatch vkCmdCopyBufferToImage vkCmdPipelineBarrier vkCreateImage vkDestroyImage "
        "vkGetImageMemoryRequirements vkAllocateMemory vkBindImageMemory vkFreeMemory vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements vkBindBufferMemory vkMapMemory vkGetPhysicalDeviceFeatures2 "
        "vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties2 vkGetPhysicalDeviceImageFormatProperties2")
{
    enum { W = 16, H = 8 };
    static const struct {
        VkSamplerYcbcrModelConversion model;
        VkSamplerYcbcrRange range;
    } cases[] = {
        { VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY, VK_SAMPLER_YCBCR_RANGE_ITU_FULL },
        { VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY, VK_SAMPLER_YCBCR_RANGE_ITU_FULL },
        { VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709, VK_SAMPLER_YCBCR_RANGE_ITU_NARROW },
        { VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601, VK_SAMPLER_YCBCR_RANGE_ITU_FULL },
    };
    const VkFormat format = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
    static float out[W * H * 4];
    unsigned char planes[W * H + 2 * (W / 2) * (H / 2)];
    struct ct_env e;
    struct ct_image img;
    struct ct_buffer up;
    VkImageCreateInfo ii;
    VkFormatFeatureFlags2 ff;
    VkCommandBuffer cb;
    VkBufferImageCopy r[3];
    VkPhysicalDeviceImageFormatInfo2 fi;
    VkSamplerYcbcrConversionImageFormatProperties yp;
    VkImageFormatProperties2 fp;
    unsigned c, p, i, j, bad = 0;
    int midpoint, forced;

    ct_env_init(t, &e);
    ct_env_device_ycbcr(t, &e);
    ff = ct_format_features(&e, format, 0);
    CT_REQUIREF(t, (ff & (VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT)) ==
                       (VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT),
                "G8_B8_R8_3PLANE_420_UNORM lacks SAMPLED_IMAGE or TRANSFER_DST");
    /* Nearest implicit reconstruction may use midpoint locations whatever the
     * conversion says; with cosited chroma the centers of odd texels would
     * fall on chroma texel edges, so cosited-only formats force explicit
     * reconstruction (where nearest takes texel i/2 at any location). */
    midpoint = (ff & VK_FORMAT_FEATURE_2_MIDPOINT_CHROMA_SAMPLES_BIT) != 0;
    forced = !midpoint;
    if (forced && !(ff & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT))
        ct_skip(t, "cosited chroma only, without forceable explicit reconstruction");
    memset(&fi, 0, sizeof(fi));
    fi.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
    fi.format = format;
    fi.type = VK_IMAGE_TYPE_2D;
    fi.tiling = VK_IMAGE_TILING_OPTIMAL;
    fi.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    memset(&yp, 0, sizeof(yp));
    yp.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES;
    memset(&fp, 0, sizeof(fp));
    fp.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
    fp.pNext = &yp;
    CT_REQUIRE_VK(t, vkGetPhysicalDeviceImageFormatProperties2(e.pd, &fi, &fp));

    ii = ct_image_info(VK_IMAGE_TYPE_2D, format, W, H, 1, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    ct_image(t, &e, &ii, 0, &img);
    ct_host_buffer(t, &e, sizeof(planes), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &up);
    ct_fill_pattern(planes, sizeof(planes), 5);
    memcpy(up.data, planes, sizeof(planes));
    memset(r, 0, sizeof(r));
    for (p = 0; p < 3; p++) {
        r[p].bufferOffset = p == 0 ? 0 : W * H + (p - 1) * (W / 2) * (H / 2);
        r[p].imageSubresource.aspectMask = p == 0   ? VK_IMAGE_ASPECT_PLANE_0_BIT
                                           : p == 1 ? VK_IMAGE_ASPECT_PLANE_1_BIT
                                                    : VK_IMAGE_ASPECT_PLANE_2_BIT;
        r[p].imageSubresource.layerCount = 1;
        r[p].imageExtent.width = p ? W / 2 : W;
        r[p].imageExtent.height = p ? H / 2 : H;
        r[p].imageExtent.depth = 1;
    }
    cb = ct_begin_commands(t, &e);
    ct_image_barrier(cb, img.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdCopyBufferToImage(cb, up.buffer, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 3, r);
    ct_image_barrier(cb, img.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ct_submit_and_wait(t, &e, cb);

    for (c = 0; c < CT_ARRAY_SIZE(cases); c++) {
        VkSamplerYcbcrConversionCreateInfo ci = conversion_info(format);
        VkSamplerYcbcrConversion conv = VK_NULL_HANDLE;
        VkSamplerYcbcrConversionInfo info;
        VkSamplerCreateInfo si;
        VkSampler sampler = VK_NULL_HANDLE;
        VkImageViewCreateInfo vi;
        VkImageView view = VK_NULL_HANDLE;
        ci.ycbcrModel = cases[c].model;
        ci.ycbcrRange = cases[c].range;
        ci.xChromaOffset = ci.yChromaOffset = midpoint ? VK_CHROMA_LOCATION_MIDPOINT : VK_CHROMA_LOCATION_COSITED_EVEN;
        ci.forceExplicitReconstruction = (VkBool32)forced;
        CT_REQUIRE_VK(t, vkCreateSamplerYcbcrConversion(e.device, &ci, NULL, &conv));
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
        info.conversion = conv;
        si = ycbcr_sampler_info(&info, VK_FILTER_NEAREST);
        CT_REQUIRE_VK(t, vkCreateSampler(e.device, &si, NULL, &sampler));
        memset(&vi, 0, sizeof(vi));
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.pNext = &info;
        vi.image = img.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        CT_REQUIRE_VK(t, vkCreateImageView(e.device, &vi, NULL, &view));
        memset(out, 0, sizeof(out));
        sample_centers(t, &e, view, sampler, yp.combinedImageSamplerDescriptorCount, W, H, out);
        for (j = 0; j < H; j++) {
            for (i = 0; i < W; i++) {
                const float *got = &out[(j * W + i) * 4];
                float yv = planes[j * W + i] / 255.0f;
                float cb_ = planes[W * H + (j / 2) * (W / 2) + i / 2] / 255.0f;
                float cr = planes[W * H + (W / 2) * (H / 2) + (j / 2) * (W / 2) + i / 2] / 255.0f;
                float want[4];
                unsigned k;
                if (cases[c].model != VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY) {
                    if (cases[c].range == VK_SAMPLER_YCBCR_RANGE_ITU_FULL) {
                        cb_ -= 128.0f / 255.0f;
                        cr -= 128.0f / 255.0f;
                    } else {
                        yv = (yv * 255.0f - 16.0f) / 219.0f;
                        cb_ = (cb_ * 255.0f - 128.0f) / 224.0f;
                        cr = (cr * 255.0f - 128.0f) / 224.0f;
                    }
                }
                want[3] = 1.0f;
                switch (cases[c].model) {
                case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709:
                    want[0] = yv + 1.5748f * cr;
                    want[1] = yv - (0.13397432f / 0.7152f) * cb_ - (0.33480248f / 0.7152f) * cr;
                    want[2] = yv + 1.8556f * cb_;
                    break;
                case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601:
                    want[0] = yv + 1.402f * cr;
                    want[1] = yv - (0.202008f / 0.587f) * cb_ - (0.419198f / 0.587f) * cr;
                    want[2] = yv + 1.772f * cb_;
                    break;
                default: /* identities: R = Cr, G = Y, B = Cb */
                    want[0] = cr;
                    want[1] = yv;
                    want[2] = cb_;
                    break;
                }
                for (k = 0; k < 4; k++) {
                    /* texels pass exactly; conversions get 1/255 for their arithmetic */
                    float tol = cases[c].model == VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY ? 0.002f : 1.0f / 255.0f;
                    if (!(got[k] >= want[k] - tol && got[k] <= want[k] + tol)) {
                        if (bad++ < 4)
                            ct_note(t, "model %d range %d texel (%u, %u): got (%.4f, %.4f, %.4f, %.4f), want (%.4f, %.4f, "
                                       "%.4f, %.4f)",
                                    (int)cases[c].model, (int)cases[c].range, i, j, got[0], got[1], got[2], got[3], want[0],
                                    want[1], want[2], want[3]);
                        break;
                    }
                }
            }
        }
        vkDestroyImageView(e.device, view, NULL);
        vkDestroySampler(e.device, sampler, NULL);
        vkDestroySamplerYcbcrConversion(e.device, conv, NULL);
    }
    CT_CHECKF(t, bad == 0, "%u texels differ", bad);
    ct_buffer_fini(&e, &up);
    ct_image_fini(&e, &img);
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(create_sampler_parameters), CT_REF(create_sampler_anisotropy), CT_REF(create_sampler_unnormalized),
    CT_REF(create_sampler_reduction),  CT_REF(sampler_allocation_count),  CT_REF(destroy_sampler_null),
    CT_REF(create_ycbcr_conversion),   CT_REF(ycbcr_sample_texels),
};

CT_SUITE(samplers, tests)
