/* Chapter "Limits" > Limit Requirements: every limit of the Required Limits
 * table against its required minimum/maximum, one test per limit so the
 * baseline tracks each. A device reporting Vulkan 1.4 must meet the "Vulkan
 * 1.4" column where the table has one, otherwise the "Vulkan Core" column.
 * A limit tied to an optional feature uses the table's "Unsupported Limit"
 * when the feature is not supported (no requirement if that is "-").
 *
 * Every limit is also checked to be written: the properties are queried
 * twice, pre-filled with 0x00 and 0xFF, and must agree.
 *
 * Rows of the table not tested: those of extensions PlasmaVK does not
 * advertise; timestampPeriod, strictLines and filterMinmax* (no required
 * value); shaderRoundingModeRTEFloat16/32 and maxSubgroupSize (required only
 * by the Roadmap profiles, whose columns are not tested); the UpdateAfterBind
 * input attachment and dynamic buffer limits (footnotes 13 and 14 allow 0).
 * Footnote 9 ("UpdateAfterBind limit >= the non-UpdateAfterBind limit") is
 * checked where the pairing is one-to-one. */
#include "ct.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum lwhere { W_LIM, W_V11, W_V12, W_V13, W_V14, W_NONE };
enum ltype { T_U32, T_I32, T_U64, T_SZ, T_F32, T_B32, T_FLAGS };
enum lkind {
    K_MIN,    /* value >= required */
    K_MAX,    /* value <= required */
    K_BITS,   /* value has every bit of required */
    K_TRUE,   /* VK_TRUE where required is 1 */
    K_NSTAGE, /* value >= n * required (core) and >= v14 (1.4); n = shader stages supported (footnote 8) */
    K_POW2    /* only: a power of two */
};
enum gwhere { G10, G11, G12, G13, G_NONE };

struct gref {
    enum gwhere where;
    size_t offset;
    const char *name;
    uint32_t required; /* the feature is required from this version (0: optional), so its limits apply */
};

struct limit {
    const char *name;
    enum lwhere where;
    size_t offset;
    enum ltype type;
    enum lkind kind;
    double core, v14;       /* NONE: no requirement in that column (v14 NONE: same as core) */
    struct gref gate;       /* the feature the limit depends on */
    double unsupported;     /* the requirement when the gate feature is unsupported (NONE: none) */
    int pow2;               /* "The value must be a power of two" */
    enum lwhere pair_where; /* footnote 9: >= this limit when the gate feature is supported */
    size_t pair_offset;
};

#define NONE (-1e300)

typedef VkPhysicalDeviceLimits S_LIM;
typedef VkPhysicalDeviceVulkan11Properties S_V11;
typedef VkPhysicalDeviceVulkan12Properties S_V12;
typedef VkPhysicalDeviceVulkan13Properties S_V13;
typedef VkPhysicalDeviceVulkan14Properties S_V14;

#define NOGATE { G_NONE, 0, NULL, 0 }
#define G10(m) { G10, offsetof(VkPhysicalDeviceFeatures, m), #m, 0 }
#define G12(m) { G12, offsetof(VkPhysicalDeviceVulkan12Features, m), #m, 0 }
#define G13(m) { G13, offsetof(VkPhysicalDeviceVulkan13Features, m), #m, 0 }
/* Gates on features Feature Requirements makes mandatory from a version. */
#define V13 VK_API_VERSION_1_3
#define V14 VK_API_VERSION_1_4
#define G10R(m, v) { G10, offsetof(VkPhysicalDeviceFeatures, m), #m, v }
#define G13R(m, v) { G13, offsetof(VkPhysicalDeviceVulkan13Features, m), #m, v }

/* Plain row; P: power of two too; G: gated by a feature; A: array element. */
#define ROW(w, m, ty, k, c, v) { #m, W_##w, offsetof(S_##w, m), T_##ty, K_##k, c, v, NOGATE, NONE, 0, W_NONE, 0 }
#define ROWP(w, m, ty, k, c, v) { #m, W_##w, offsetof(S_##w, m), T_##ty, K_##k, c, v, NOGATE, NONE, 1, W_NONE, 0 }
#define ROWG(w, m, ty, k, c, v, g, u) { #m, W_##w, offsetof(S_##w, m), T_##ty, K_##k, c, v, g, u, 0, W_NONE, 0 }
#define ROWGP(w, m, ty, k, c, v, g, pw, pm)                                                                          \
    { #m, W_##w, offsetof(S_##w, m), T_##ty, K_##k, c, v, g, NONE, 0, W_##pw, offsetof(S_##pw, pm) }
#define ROWA(w, m, i, ty, k, c, v) { #m "[" #i "]", W_##w, offsetof(S_##w, m[i]), T_##ty, K_##k, c, v, NOGATE, NONE, 0, W_NONE, 0 }

#define S1 VK_SAMPLE_COUNT_1_BIT
#define S14 (VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT)
#define UAB_SAMPLED G12(descriptorBindingSampledImageUpdateAfterBind)
#define UAB_STORAGE_IMAGE G12(descriptorBindingStorageImageUpdateAfterBind)
#define UAB_UNIFORM G12(descriptorBindingUniformBufferUpdateAfterBind)
#define UAB_STORAGE G12(descriptorBindingStorageBufferUpdateAfterBind)

static const struct limit limits[] = {
    ROW(LIM, maxImageDimension1D, U32, MIN, 4096, 8192),
    ROW(LIM, maxImageDimension2D, U32, MIN, 4096, 8192),
    ROW(LIM, maxImageDimension3D, U32, MIN, 256, 512),
    ROW(LIM, maxImageDimensionCube, U32, MIN, 4096, 8192),
    ROW(LIM, maxImageArrayLayers, U32, MIN, 256, 2048),
    ROW(LIM, maxTexelBufferElements, U32, MIN, 65536, NONE),
    ROW(LIM, maxUniformBufferRange, U32, MIN, 16384, 65536),
    ROW(LIM, maxStorageBufferRange, U32, MIN, 134217728.0, NONE),
    ROW(LIM, maxPushConstantsSize, U32, MIN, 128, 256),
    ROW(LIM, maxMemoryAllocationCount, U32, MIN, 4096, NONE),
    ROW(LIM, maxSamplerAllocationCount, U32, MIN, 4000, NONE),
    ROW(LIM, bufferImageGranularity, U64, MAX, 131072, 4096),
    ROWG(LIM, sparseAddressSpaceSize, U64, MIN, 2147483648.0, NONE, G10(sparseBinding), 0),
    ROW(LIM, maxBoundDescriptorSets, U32, MIN, 4, 7),
    ROW(LIM, maxPerStageDescriptorSamplers, U32, MIN, 16, NONE),
    ROW(LIM, maxPerStageDescriptorUniformBuffers, U32, MIN, 12, 15),
    ROW(LIM, maxPerStageDescriptorStorageBuffers, U32, MIN, 4, NONE),
    ROW(LIM, maxPerStageDescriptorSampledImages, U32, MIN, 16, NONE),
    ROW(LIM, maxPerStageDescriptorStorageImages, U32, MIN, 4, NONE),
    ROW(LIM, maxPerStageDescriptorInputAttachments, U32, MIN, 4, NONE),
    ROW(LIM, maxDescriptorSetSamplers, U32, NSTAGE, 16, NONE),
    ROW(LIM, maxDescriptorSetUniformBuffers, U32, NSTAGE, 12, 90),
    ROW(LIM, maxDescriptorSetUniformBuffersDynamic, U32, MIN, 8, NONE),
    ROW(LIM, maxDescriptorSetStorageBuffers, U32, NSTAGE, 4, 96),
    ROW(LIM, maxDescriptorSetStorageBuffersDynamic, U32, MIN, 4, NONE),
    ROW(LIM, maxDescriptorSetSampledImages, U32, NSTAGE, 16, NONE),
    ROW(LIM, maxDescriptorSetStorageImages, U32, NSTAGE, 4, 144),
    ROW(LIM, maxDescriptorSetInputAttachments, U32, MIN, 4, NONE),
    ROW(LIM, maxVertexInputAttributes, U32, MIN, 16, NONE),
    ROW(LIM, maxVertexInputBindings, U32, MIN, 16, NONE), /* 8 with VK_KHR_portability_subset: see check */
    ROW(LIM, maxVertexInputAttributeOffset, U32, MIN, 2047, NONE),
    ROW(LIM, maxVertexInputBindingStride, U32, MIN, 2048, NONE),
    ROW(LIM, maxVertexOutputComponents, U32, MIN, 64, NONE),
    ROWG(LIM, maxTessellationGenerationLevel, U32, MIN, 64, NONE, G10(tessellationShader), 0),
    ROWG(LIM, maxTessellationPatchSize, U32, MIN, 32, NONE, G10(tessellationShader), 0),
    ROWG(LIM, maxTessellationControlPerVertexInputComponents, U32, MIN, 64, NONE, G10(tessellationShader), 0),
    ROWG(LIM, maxTessellationControlPerVertexOutputComponents, U32, MIN, 64, NONE, G10(tessellationShader), 0),
    ROWG(LIM, maxTessellationControlPerPatchOutputComponents, U32, MIN, 120, NONE, G10(tessellationShader), 0),
    ROWG(LIM, maxTessellationControlTotalOutputComponents, U32, MIN, 2048, NONE, G10(tessellationShader), 0),
    ROWG(LIM, maxTessellationEvaluationInputComponents, U32, MIN, 64, NONE, G10(tessellationShader), 0),
    ROWG(LIM, maxTessellationEvaluationOutputComponents, U32, MIN, 64, NONE, G10(tessellationShader), 0),
    ROWG(LIM, maxGeometryShaderInvocations, U32, MIN, 32, NONE, G10(geometryShader), 0),
    ROWG(LIM, maxGeometryInputComponents, U32, MIN, 64, NONE, G10(geometryShader), 0),
    ROWG(LIM, maxGeometryOutputComponents, U32, MIN, 64, NONE, G10(geometryShader), 0),
    ROWG(LIM, maxGeometryOutputVertices, U32, MIN, 256, NONE, G10(geometryShader), 0),
    ROWG(LIM, maxGeometryTotalOutputComponents, U32, MIN, 1024, NONE, G10(geometryShader), 0),
    ROW(LIM, maxFragmentInputComponents, U32, MIN, 64, NONE),
    ROW(LIM, maxFragmentOutputAttachments, U32, MIN, 4, NONE),
    ROWG(LIM, maxFragmentDualSrcAttachments, U32, MIN, 1, NONE, G10(dualSrcBlend), 0),
    ROW(LIM, maxFragmentCombinedOutputResources, U32, MIN, 4, 16),
    ROW(LIM, maxComputeSharedMemorySize, U32, MIN, 16384, NONE),
    ROWA(LIM, maxComputeWorkGroupCount, 0, U32, MIN, 65535, NONE),
    ROWA(LIM, maxComputeWorkGroupCount, 1, U32, MIN, 65535, NONE),
    ROWA(LIM, maxComputeWorkGroupCount, 2, U32, MIN, 65535, NONE),
    ROW(LIM, maxComputeWorkGroupInvocations, U32, MIN, 128, 256),
    ROWA(LIM, maxComputeWorkGroupSize, 0, U32, MIN, 128, 256),
    ROWA(LIM, maxComputeWorkGroupSize, 1, U32, MIN, 128, 256),
    ROWA(LIM, maxComputeWorkGroupSize, 2, U32, MIN, 64, 64),
    ROW(LIM, subPixelPrecisionBits, U32, MIN, 4, NONE),
    ROW(LIM, subTexelPrecisionBits, U32, MIN, 4, 8),
    ROW(LIM, mipmapPrecisionBits, U32, MIN, 4, 6),
    ROWG(LIM, maxDrawIndexedIndexValue, U32, MIN, 4294967295.0, NONE, G10R(fullDrawIndexUint32, V14), 16777215.0),
    ROWG(LIM, maxDrawIndirectCount, U32, MIN, 65535, NONE, G10(multiDrawIndirect), 1),
    ROW(LIM, maxSamplerLodBias, F32, MIN, 2, 14),
    ROWG(LIM, maxSamplerAnisotropy, F32, MIN, 16, NONE, G10R(samplerAnisotropy, V14), 1),
    ROWG(LIM, maxViewports, U32, MIN, 16, NONE, G10(multiViewport), 1),
    ROWA(LIM, maxViewportDimensions, 0, U32, MIN, 4096, 7680),
    ROWA(LIM, maxViewportDimensions, 1, U32, MIN, 4096, 7680),
    ROWA(LIM, viewportBoundsRange, 0, F32, MAX, -8192, -15360),
    ROWA(LIM, viewportBoundsRange, 1, F32, MIN, 8191, 15359),
    ROW(LIM, viewportSubPixelBits, U32, MIN, 0, NONE),
    ROWP(LIM, minMemoryMapAlignment, SZ, MIN, 64, NONE),
    ROWP(LIM, minTexelBufferOffsetAlignment, U64, MAX, 256, NONE),
    ROWP(LIM, minUniformBufferOffsetAlignment, U64, MAX, 256, NONE),
    ROWP(LIM, minStorageBufferOffsetAlignment, U64, MAX, 256, NONE),
    ROW(LIM, minTexelOffset, I32, MAX, -8, NONE),
    ROW(LIM, maxTexelOffset, U32, MIN, 7, NONE),
    ROWG(LIM, minTexelGatherOffset, I32, MAX, -8, NONE, G10R(shaderImageGatherExtended, V14), 0),
    ROWG(LIM, maxTexelGatherOffset, U32, MIN, 7, NONE, G10R(shaderImageGatherExtended, V14), 0),
    ROWG(LIM, minInterpolationOffset, F32, MAX, -0.5, NONE, G10R(sampleRateShading, V14), 0),
    ROWG(LIM, subPixelInterpolationOffsetBits, U32, MIN, 4, NONE, G10R(sampleRateShading, V14), 0),
    ROW(LIM, maxFramebufferWidth, U32, MIN, 4096, 7680),
    ROW(LIM, maxFramebufferHeight, U32, MIN, 4096, 7680),
    ROW(LIM, maxFramebufferLayers, U32, MIN, 256, NONE),
    ROW(LIM, framebufferColorSampleCounts, FLAGS, BITS, S14, NONE),
    ROW(LIM, framebufferDepthSampleCounts, FLAGS, BITS, S14, NONE),
    ROW(LIM, framebufferStencilSampleCounts, FLAGS, BITS, S14, NONE),
    ROW(LIM, framebufferNoAttachmentsSampleCounts, FLAGS, BITS, S14, NONE),
    ROW(LIM, maxColorAttachments, U32, MIN, 4, 8),
    ROW(LIM, sampledImageColorSampleCounts, FLAGS, BITS, S14, NONE),
    ROW(LIM, sampledImageIntegerSampleCounts, FLAGS, BITS, S1, NONE),
    ROW(LIM, sampledImageDepthSampleCounts, FLAGS, BITS, S14, NONE),
    ROW(LIM, sampledImageStencilSampleCounts, FLAGS, BITS, S14, NONE),
    ROWG(LIM, storageImageSampleCounts, FLAGS, BITS, S14, NONE, G10(shaderStorageImageMultisample), S1),
    ROW(LIM, maxSampleMaskWords, U32, MIN, 1, NONE),
    ROW(LIM, timestampComputeAndGraphics, B32, TRUE, NONE, 1),
    ROWG(LIM, maxClipDistances, U32, MIN, 8, NONE, G10(shaderClipDistance), 0),
    ROWG(LIM, maxCullDistances, U32, MIN, 8, NONE, G10(shaderCullDistance), 0),
    ROWG(LIM, maxCombinedClipAndCullDistances, U32, MIN, 8, NONE, G10(shaderCullDistance), 0),
    ROW(LIM, discreteQueuePriorities, U32, MIN, 2, NONE),
    ROWG(LIM, pointSizeGranularity, F32, MAX, 1.0, 0.125, G10R(largePoints, V14), 0.0),
    ROWG(LIM, lineWidthGranularity, F32, MAX, 1.0, 0.5, G10(wideLines), 0.0),
    ROW(LIM, standardSampleLocations, B32, TRUE, NONE, 1),
    ROWP(LIM, optimalBufferCopyOffsetAlignment, U64, POW2, NONE, NONE),
    ROWP(LIM, optimalBufferCopyRowPitchAlignment, U64, POW2, NONE, NONE),
    ROWP(LIM, nonCoherentAtomSize, U64, MAX, 256, NONE),
    /* Vulkan 1.1 */
    ROWP(V11, subgroupSize, U32, MIN, 1, NONE),
    ROW(V11, subgroupSupportedStages, FLAGS, BITS, VK_SHADER_STAGE_COMPUTE_BIT, NONE),
    ROW(V11, subgroupSupportedOperations, FLAGS, BITS, VK_SUBGROUP_FEATURE_BASIC_BIT, NONE),
    ROW(V11, maxMultiviewViewCount, U32, MIN, 6, NONE),
    ROW(V11, maxMultiviewInstanceIndex, U32, MIN, 134217727.0, NONE),
    ROW(V11, maxPerSetDescriptors, U32, MIN, 1024, NONE),
    ROW(V11, maxMemoryAllocationSize, U64, MIN, 1073741824.0, NONE),
    /* Vulkan 1.2 */
    ROW(V12, shaderSignedZeroInfNanPreserveFloat16, B32, TRUE, NONE, 1),
    ROW(V12, shaderSignedZeroInfNanPreserveFloat32, B32, TRUE, NONE, 1),
    ROW(V12, maxTimelineSemaphoreValueDifference, U64, MIN, 2147483647.0, NONE),
    ROW(V12, framebufferIntegerColorSampleCounts, FLAGS, BITS, S1, NONE),
    ROWG(V12, maxUpdateAfterBindDescriptorsInAllPools, U32, MIN, 500000, NONE, G12(descriptorIndexing), NONE),
    ROWGP(V12, maxPerStageDescriptorUpdateAfterBindSamplers, U32, MIN, 500000, NONE, UAB_SAMPLED, LIM,
          maxPerStageDescriptorSamplers),
    ROWGP(V12, maxPerStageDescriptorUpdateAfterBindUniformBuffers, U32, MIN, 12, NONE, UAB_UNIFORM, LIM,
          maxPerStageDescriptorUniformBuffers),
    ROWGP(V12, maxPerStageDescriptorUpdateAfterBindStorageBuffers, U32, MIN, 500000, NONE, UAB_STORAGE, LIM,
          maxPerStageDescriptorStorageBuffers),
    ROWGP(V12, maxPerStageDescriptorUpdateAfterBindSampledImages, U32, MIN, 500000, NONE, UAB_SAMPLED, LIM,
          maxPerStageDescriptorSampledImages),
    ROWGP(V12, maxPerStageDescriptorUpdateAfterBindStorageImages, U32, MIN, 500000, NONE, UAB_STORAGE_IMAGE, LIM,
          maxPerStageDescriptorStorageImages),
    ROWG(V12, maxPerStageUpdateAfterBindResources, U32, MIN, 500000, NONE, G12(descriptorIndexing), NONE),
    ROWGP(V12, maxDescriptorSetUpdateAfterBindSamplers, U32, MIN, 500000, NONE, UAB_SAMPLED, LIM,
          maxDescriptorSetSamplers),
    ROWGP(V12, maxDescriptorSetUpdateAfterBindUniformBuffers, U32, NSTAGE, 12, NONE, UAB_UNIFORM, LIM,
          maxDescriptorSetUniformBuffers),
    ROWGP(V12, maxDescriptorSetUpdateAfterBindStorageBuffers, U32, MIN, 500000, NONE, UAB_STORAGE, LIM,
          maxDescriptorSetStorageBuffers),
    ROWGP(V12, maxDescriptorSetUpdateAfterBindSampledImages, U32, MIN, 500000, NONE, UAB_SAMPLED, LIM,
          maxDescriptorSetSampledImages),
    ROWGP(V12, maxDescriptorSetUpdateAfterBindStorageImages, U32, MIN, 500000, NONE, UAB_STORAGE_IMAGE, LIM,
          maxDescriptorSetStorageImages),
    /* Vulkan 1.3 */
    ROWG(V13, maxInlineUniformBlockSize, U32, MIN, 256, NONE, G13R(inlineUniformBlock, V13), NONE),
    ROWG(V13, maxPerStageDescriptorInlineUniformBlocks, U32, MIN, 4, NONE, G13R(inlineUniformBlock, V13), NONE),
    ROWGP(V13, maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks, U32, MIN, 4, NONE,
          G13(descriptorBindingInlineUniformBlockUpdateAfterBind), V13, maxPerStageDescriptorInlineUniformBlocks),
    ROWG(V13, maxDescriptorSetInlineUniformBlocks, U32, MIN, 4, NONE, G13R(inlineUniformBlock, V13), NONE),
    ROWGP(V13, maxDescriptorSetUpdateAfterBindInlineUniformBlocks, U32, MIN, 4, NONE,
          G13(descriptorBindingInlineUniformBlockUpdateAfterBind), V13, maxDescriptorSetInlineUniformBlocks),
    ROWG(V13, maxInlineUniformTotalSize, U32, MIN, 256, NONE, G13R(inlineUniformBlock, V13), NONE),
    ROWG(V13, maxBufferSize, U64, MIN, 1073741824.0, NONE, G13R(maintenance4, V13), NONE),
    /* Vulkan 1.4 */
    ROW(V14, lineSubPixelPrecisionBits, U32, MIN, 4, NONE),
    ROW(V14, maxVertexAttribDivisor, U32, MIN, 65535, NONE),
    ROW(V14, maxPushDescriptors, U32, MIN, 32, NONE),
};

/* ---- Reading values ---- */

static const char *struct_base(const struct ct_properties *p, enum lwhere w)
{
    switch (w) {
    case W_LIM:
        return (const char *)&p->p2.properties.limits;
    case W_V11:
        return (const char *)&p->v11;
    case W_V12:
        return (const char *)&p->v12;
    case W_V13:
        return (const char *)&p->v13;
    default:
        return (const char *)&p->v14;
    }
}

static uint32_t struct_version(enum lwhere w)
{
    switch (w) {
    case W_LIM:
        return VK_API_VERSION_1_0;
    case W_V11: /* VkPhysicalDeviceVulkan11Properties arrived with 1.2 */
    case W_V12:
        return VK_API_VERSION_1_2;
    case W_V13:
        return VK_API_VERSION_1_3;
    default:
        return VK_API_VERSION_1_4;
    }
}

static size_t type_size(enum ltype t)
{
    switch (t) {
    case T_U64:
        return sizeof(uint64_t);
    case T_SZ:
        return sizeof(size_t);
    default:
        return 4;
    }
}

static double read_value(const void *p, enum ltype t)
{
    switch (t) {
    case T_U32:
    case T_FLAGS:
    case T_B32: {
        uint32_t v;
        memcpy(&v, p, sizeof(v));
        return v;
    }
    case T_I32: {
        int32_t v;
        memcpy(&v, p, sizeof(v));
        return v;
    }
    case T_U64: {
        uint64_t v;
        memcpy(&v, p, sizeof(v));
        return (double)v;
    }
    case T_SZ: {
        size_t v;
        memcpy(&v, p, sizeof(v));
        return (double)v;
    }
    default: {
        float v;
        memcpy(&v, p, sizeof(v));
        return v;
    }
    }
}

static uint64_t read_bits(const void *p, enum ltype t)
{
    if (t == T_U64) {
        uint64_t v;
        memcpy(&v, p, sizeof(v));
        return v;
    }
    if (t == T_SZ) {
        size_t v;
        memcpy(&v, p, sizeof(v));
        return v;
    }
    {
        uint32_t v;
        memcpy(&v, p, sizeof(v));
        return v;
    }
}

/* Whether the gate feature's "supported" limits apply: it is supported, or
 * the device's version requires it. */
static int gate_supported(const struct ct_env *e, const struct ct_features *f, struct gref g)
{
    const char *base;
    VkBool32 b;
    if (g.required && e->version >= g.required)
        return 1;
    switch (g.where) {
    case G10:
        base = (const char *)&f->f2.features;
        break;
    case G11:
        if (e->version < VK_API_VERSION_1_2)
            return 0;
        base = (const char *)&f->v11;
        break;
    case G12:
        if (e->version < VK_API_VERSION_1_2)
            return 0;
        base = (const char *)&f->v12;
        break;
    case G13:
        if (e->version < VK_API_VERSION_1_3)
            return 0;
        base = (const char *)&f->v13;
        break;
    default:
        return 1;
    }
    memcpy(&b, base + g.offset, sizeof(b));
    return b == VK_TRUE;
}

/* Shader stages supported, for footnote 8: vertex, fragment and compute, plus
 * tessellation control and evaluation, plus geometry. */
static unsigned stage_count(const struct ct_features *f)
{
    return 3 + (f->f2.features.tessellationShader ? 2 : 0) + (f->f2.features.geometryShader ? 1 : 0);
}

struct context {
    struct ct_env e;
    struct ct_properties a, b;
    struct ct_features f;
    int v14;
};

static void load(struct ct *t, struct context *c, uint32_t needs)
{
    ct_env_init(t, &c->e);
    if (c->e.version < needs)
        ct_skip(t, "the limit's structure needs Vulkan %u.%u; the device offers %u.%u", VK_API_VERSION_MAJOR(needs),
                VK_API_VERSION_MINOR(needs), VK_API_VERSION_MAJOR(c->e.version), VK_API_VERSION_MINOR(c->e.version));
    ct_get_properties(&c->e, &c->a, 0x00);
    ct_get_properties(&c->e, &c->b, 0xFF);
    ct_get_features(&c->e, &c->f, 0x00);
    c->v14 = c->e.version >= VK_API_VERSION_1_4;
}

/* Reads a limit, failing if the two pre-filled queries disagree. */
static double value_of(struct ct *t, const struct context *c, enum lwhere w, size_t offset, enum ltype type,
                       const char *name)
{
    const char *pa = struct_base(&c->a, w) + offset, *pb = struct_base(&c->b, w) + offset;
    CT_CHECKF(t, memcmp(pa, pb, type_size(type)) == 0,
              "%s is not written (queries pre-filled with 0x00 and 0xFF return %g and %g)", name,
              read_value(pa, type), read_value(pb, type));
    return read_value(pa, type);
}

static void check_limit(struct ct *t)
{
    const struct limit *l = ct_param(t);
    struct context c;
    double value, required;
    int gated_off;

    load(t, &c, struct_version(l->where));
    value = value_of(t, &c, l->where, l->offset, l->type, l->name);
    required = c.v14 && l->v14 != NONE ? l->v14 : l->core;
    gated_off = l->gate.where != G_NONE && !gate_supported(&c.e, &c.f, l->gate);
    if (gated_off)
        required = l->unsupported;
    if (l->kind != K_TRUE && !strcmp(l->name, "maxVertexInputBindings") && c.e.portability_subset)
        required = 8; /* footnote 10: with VK_KHR_portability_subset enabled */

    if (l->pow2) {
        uint64_t bits = read_bits(struct_base(&c.a, l->where) + l->offset, l->type);
        CT_CHECKF(t, ct_is_pow2(bits), "%s is %llu, not a power of two", l->name, (unsigned long long)bits);
    }
    if (required == NONE) {
        if (l->pow2)
            return;
        if (gated_off)
            ct_skip(t, "%s is not supported: no requirement", l->gate.name);
        ct_skip(t, "no requirement at Vulkan %u.%u", VK_API_VERSION_MAJOR(c.e.version), VK_API_VERSION_MINOR(c.e.version));
    }
    switch (l->kind) {
    case K_MIN:
        CT_CHECKF(t, value >= required, "%s is %.9g, required at least %.9g", l->name, value, required);
        break;
    case K_MAX:
        CT_CHECKF(t, value <= required, "%s is %.9g, required at most %.9g", l->name, value, required);
        break;
    case K_BITS: {
        uint64_t have = read_bits(struct_base(&c.a, l->where) + l->offset, l->type), need = (uint64_t)required;
        CT_CHECKF(t, (have & need) == need, "%s is 0x%llx, required to include 0x%llx", l->name,
                  (unsigned long long)have, (unsigned long long)need);
        break;
    }
    case K_TRUE:
        CT_CHECKF(t, value == VK_TRUE, "%s is %g, required VK_TRUE", l->name, value);
        break;
    case K_NSTAGE: {
        unsigned n = stage_count(&c.f);
        double need = l->core * n;
        if (c.v14 && l->v14 != NONE && l->v14 > need)
            need = l->v14;
        CT_CHECKF(t, value >= need, "%s is %.9g, required at least %.9g (%u stages x %g%s)", l->name, value, need, n,
                  l->core, c.v14 && l->v14 != NONE ? ", Vulkan 1.4 column" : "");
        break;
    }
    default:
        break;
    }
    if (l->pair_where != W_NONE && !gated_off) {
        double other = value_of(t, &c, l->pair_where, l->pair_offset, l->type, "the non-UpdateAfterBind limit");
        CT_CHECKF(t, value >= other, "%s is %.9g, below the corresponding limit %.9g (footnote 9)", l->name, value,
                  other);
    }
    ct_env_fini(&c.e);
}

/* ---- Limits whose requirement depends on other limits ---- */

CT_TEST(maxPerStageResources,
        "Limits > Limit Requirements: maxPerStageResources is at least the smaller of 128 and the sum of the "
        "per-stage uniform buffer, storage buffer, sampled image, storage image, input attachment and color "
        "attachment limits (footnote 2); at least 200 for Vulkan 1.4",
        "vkGetPhysicalDeviceProperties2")
{
    struct context c;
    const VkPhysicalDeviceLimits *l;
    double sum, need, value;

    load(t, &c, VK_API_VERSION_1_0);
    l = &c.a.p2.properties.limits;
    value = value_of(t, &c, W_LIM, offsetof(VkPhysicalDeviceLimits, maxPerStageResources), T_U32,
                     "maxPerStageResources");
    sum = (double)l->maxPerStageDescriptorUniformBuffers + l->maxPerStageDescriptorStorageBuffers +
          l->maxPerStageDescriptorSampledImages + l->maxPerStageDescriptorStorageImages +
          l->maxPerStageDescriptorInputAttachments + l->maxColorAttachments;
    need = c.v14 ? 200 : (sum < 128 ? sum : 128);
    CT_CHECKF(t, value >= need, "maxPerStageResources is %g, required at least %g", value, need);
    ct_env_fini(&c.e);
}

CT_TEST(maxInterpolationOffset,
        "Limits > Limit Requirements: maxInterpolationOffset is at least 0.5 - 1 ULP, the ULP being "
        "2^-subPixelInterpolationOffsetBits (footnote 5), when sampleRateShading is supported; else at least 0.0",
        "vkGetPhysicalDeviceProperties2")
{
    struct context c;
    double value, bits, need;

    load(t, &c, VK_API_VERSION_1_0);
    value = value_of(t, &c, W_LIM, offsetof(VkPhysicalDeviceLimits, maxInterpolationOffset), T_F32,
                     "maxInterpolationOffset");
    bits = c.a.p2.properties.limits.subPixelInterpolationOffsetBits;
    need = c.f.f2.features.sampleRateShading || c.v14 ? 0.5 - pow(2.0, -(bits < 4 ? 4 : bits)) : 0.0;
    CT_CHECKF(t, value >= need, "maxInterpolationOffset is %g, required at least %g", value, need);
    ct_env_fini(&c.e);
}

CT_TEST(pointSizeRange,
        "Limits > Limit Requirements: pointSizeRange is at least (1.0, 64.0 - pointSizeGranularity) with "
        "largePoints, (1.0, 256.0 - pointSizeGranularity) for Vulkan 1.4 (footnote 6); (1.0, 1.0) without",
        "vkGetPhysicalDeviceProperties2")
{
    struct context c;
    double lo, hi, gran, need;

    load(t, &c, VK_API_VERSION_1_0);
    lo = value_of(t, &c, W_LIM, offsetof(VkPhysicalDeviceLimits, pointSizeRange[0]), T_F32, "pointSizeRange[0]");
    hi = value_of(t, &c, W_LIM, offsetof(VkPhysicalDeviceLimits, pointSizeRange[1]), T_F32, "pointSizeRange[1]");
    gran = c.a.p2.properties.limits.pointSizeGranularity;
    need = c.f.f2.features.largePoints || c.v14 ? (c.v14 ? 256.0 : 64.0) - gran : 1.0;
    CT_CHECKF(t, lo <= 1.0, "pointSizeRange[0] is %g, required at most 1.0", lo);
    CT_CHECKF(t, hi >= need, "pointSizeRange[1] is %g, required at least %g", hi, need);
    ct_env_fini(&c.e);
}

CT_TEST(lineWidthRange,
        "Limits > Limit Requirements: lineWidthRange is at least (1.0, 8.0 - lineWidthGranularity) with wideLines "
        "(footnote 7); (1.0, 1.0) without",
        "vkGetPhysicalDeviceProperties2")
{
    struct context c;
    double lo, hi, need;

    load(t, &c, VK_API_VERSION_1_0);
    lo = value_of(t, &c, W_LIM, offsetof(VkPhysicalDeviceLimits, lineWidthRange[0]), T_F32, "lineWidthRange[0]");
    hi = value_of(t, &c, W_LIM, offsetof(VkPhysicalDeviceLimits, lineWidthRange[1]), T_F32, "lineWidthRange[1]");
    need = c.f.f2.features.wideLines ? 8.0 - c.a.p2.properties.limits.lineWidthGranularity : 1.0;
    CT_CHECKF(t, lo <= 1.0, "lineWidthRange[0] is %g, required at most 1.0", lo);
    CT_CHECKF(t, hi >= need, "lineWidthRange[1] is %g, required at least %g", hi, need);
    ct_env_fini(&c.e);
}

CT_TEST(viewportBoundsRange_vs_maxViewportDimensions,
        "Limits > viewportBoundsRange: the range must be at least [-2 x size, 2 x size - 1], size being the "
        "larger maxViewportDimensions",
        "vkGetPhysicalDeviceProperties2")
{
    struct context c;
    const VkPhysicalDeviceLimits *l;
    double size;

    load(t, &c, VK_API_VERSION_1_0);
    l = &c.a.p2.properties.limits;
    size = l->maxViewportDimensions[0] > l->maxViewportDimensions[1] ? l->maxViewportDimensions[0]
                                                                      : l->maxViewportDimensions[1];
    CT_CHECKF(t, l->viewportBoundsRange[0] <= -2 * size, "viewportBoundsRange[0] %g > %g", l->viewportBoundsRange[0],
              -2 * size);
    CT_CHECKF(t, l->viewportBoundsRange[1] >= 2 * size - 1, "viewportBoundsRange[1] %g < %g", l->viewportBoundsRange[1],
              2 * size - 1);
    ct_env_fini(&c.e);
}

CT_TEST(maxViewportDimensions_vs_framebuffer,
        "Limits > maxViewportDimensions: must be at least the largest image that can be created and used as a "
        "framebuffer attachment (min(maxImageDimension2D, maxFramebufferWidth/Height))",
        "vkGetPhysicalDeviceProperties2")
{
    struct context c;
    const VkPhysicalDeviceLimits *l;
    uint32_t w, h;

    load(t, &c, VK_API_VERSION_1_0);
    l = &c.a.p2.properties.limits;
    w = CT_MIN(l->maxImageDimension2D, l->maxFramebufferWidth);
    h = CT_MIN(l->maxImageDimension2D, l->maxFramebufferHeight);
    CT_CHECKF(t, l->maxViewportDimensions[0] >= w, "maxViewportDimensions[0] %u < %u", l->maxViewportDimensions[0], w);
    CT_CHECKF(t, l->maxViewportDimensions[1] >= h, "maxViewportDimensions[1] %u < %u", l->maxViewportDimensions[1], h);
    ct_env_fini(&c.e);
}

static const struct ct_test *const fixed_tests[] = {
    CT_REF(maxPerStageResources),
    CT_REF(maxInterpolationOffset),
    CT_REF(pointSizeRange),
    CT_REF(lineWidthRange),
    CT_REF(viewportBoundsRange_vs_maxViewportDimensions),
    CT_REF(maxViewportDimensions_vs_framebuffer),
};

static void describe(const struct limit *l, char *out, size_t size)
{
    static const char *const kinds[] = { "at least", "at most", "includes", "is VK_TRUE", "at least n stages x", "" };
    int n;
    if (l->kind == K_TRUE)
        n = snprintf(out, size, "Limits > Limit Requirements: %s must be VK_TRUE for Vulkan 1.4", l->name);
    else if (l->kind == K_POW2)
        n = snprintf(out, size, "Limits: %s must be a power of two", l->name);
    else {
        n = snprintf(out, size, "Limits > Limit Requirements: %s %s %.9g", l->name, kinds[l->kind], l->core);
        if (n > 0 && (size_t)n < size && l->v14 != NONE)
            n += snprintf(out + n, size - (size_t)n, " (Vulkan 1.4: %.9g)", l->v14);
    }
    if (n > 0 && (size_t)n < size && l->gate.where != G_NONE)
        n += l->unsupported == NONE
                 ? snprintf(out + n, size - (size_t)n, ", if %s is supported", l->gate.name)
                 : snprintf(out + n, size - (size_t)n, "; %.9g without %s", l->unsupported, l->gate.name);
    if (n > 0 && (size_t)n < size && l->gate.required)
        n += snprintf(out + n, size - (size_t)n, " (%s is required from Vulkan 1.%u)", l->gate.name,
                      VK_API_VERSION_MINOR(l->gate.required));
    if (n > 0 && (size_t)n < size && l->pow2 && l->kind != K_POW2)
        n += snprintf(out + n, size - (size_t)n, "; a power of two");
    if (n > 0 && (size_t)n < size && l->pair_where != W_NONE)
        snprintf(out + n, size - (size_t)n, "; at least the non-UpdateAfterBind limit (footnote 9)");
}

void ct_suite_limits(struct ct_registry *r);
void ct_suite_limits(struct ct_registry *r)
{
    static struct ct_test tests[CT_ARRAY_SIZE(limits)];
    static const struct ct_test *ptrs[CT_ARRAY_SIZE(limits)];
    static char specs[CT_ARRAY_SIZE(limits)][256];
    size_t i;

    for (i = 0; i < CT_ARRAY_SIZE(limits); i++) {
        describe(&limits[i], specs[i], sizeof(specs[i]));
        tests[i].name = limits[i].name;
        tests[i].fn = check_limit;
        tests[i].spec = specs[i];
        tests[i].commands = "vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceFeatures2";
        tests[i].param = &limits[i];
        ptrs[i] = &tests[i];
    }
    ct_register(r, "limits", ptrs, CT_ARRAY_SIZE(ptrs));
    ct_register(r, "limits", fixed_tests, CT_ARRAY_SIZE(fixed_tests));
}
