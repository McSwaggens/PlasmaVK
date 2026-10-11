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

static const struct ct_test *const tests[] = {
    CT_REF(create_sampler_parameters), CT_REF(create_sampler_anisotropy), CT_REF(create_sampler_unnormalized),
    CT_REF(create_sampler_reduction),  CT_REF(sampler_allocation_count),  CT_REF(destroy_sampler_null),
};

CT_SUITE(samplers, tests)
