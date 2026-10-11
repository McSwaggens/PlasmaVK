/* Chapter "Formats": Required Format Support (one test per format of its
 * tables), and the chapter's other rules on format features.
 *
 * Source: chapters/formats.adoc of Vulkan-Docs (1.4.3xx), the tables
 * "Mandatory Format Support: ..." and the sections after them. Columns:
 * SAMPLED_IMAGE, BLIT_SRC, SAMPLED_IMAGE_FILTER_LINEAR, STORAGE_IMAGE,
 * STORAGE_IMAGE_ATOMIC, COLOR_ATTACHMENT, BLIT_DST, COLOR_ATTACHMENT_BLEND,
 * DEPTH_STENCIL_ATTACHMENT (optimalTilingFeatures) and VERTEX_BUFFER,
 * UNIFORM_TEXEL_BUFFER, STORAGE_TEXEL_BUFFER (bufferFeatures). Cells marked
 * "if shaderStorageImageExtendedFormats" are the `ext` storage column. */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

#define S VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT
#define BS VK_FORMAT_FEATURE_2_BLIT_SRC_BIT
#define FL VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT
#define ST VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT
#define SA VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT
#define CA VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT
#define BD VK_FORMAT_FEATURE_2_BLIT_DST_BIT
#define BL VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT
#define DS VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT
#define VB VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT
#define UT VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT
#define STB VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT

#define T1 1u /* VK_IMAGE_TYPE_1D */
#define T2 2u /* VK_IMAGE_TYPE_2D (and cube) */
#define T3 4u /* VK_IMAGE_TYPE_3D */
#define TALL (T1 | T2 | T3)

struct row {
    const char *name;
    VkFormat format;
    VkFormatFeatureFlags2 optimal, buffer;
    int ext;        /* STORAGE_IMAGE if shaderStorageImageExtendedFormats */
    unsigned types; /* image types the optimal features must hold for */
};

#define ROW(f, opt, buf, ext) { #f, VK_FORMAT_##f, opt, buf, ext, TALL }
#define DROW(f, opt) { #f, VK_FORMAT_##f, opt, 0, 0, T2 }

static const struct row rows[] = {
    /* Sub-byte components */
    ROW(B4G4R4A4_UNORM_PACK16, S | BS | FL, 0, 0),
    ROW(R5G6B5_UNORM_PACK16, S | BS | FL | CA | BD | BL, 0, 0),
    ROW(A1R5G5B5_UNORM_PACK16, S | BS | FL | CA | BD | BL, 0, 0),
    /* 1-3 byte-sized components */
    ROW(R8_UNORM, S | BS | FL | CA | BD | BL, VB | UT, 1),
    ROW(R8_SNORM, S | BS | FL, VB | UT, 1),
    ROW(R8_UINT, S | BS | CA | BD, VB | UT, 1),
    ROW(R8_SINT, S | BS | CA | BD, VB | UT, 1),
    ROW(R8G8_UNORM, S | BS | FL | CA | BD | BL, VB | UT, 1),
    ROW(R8G8_SNORM, S | BS | FL, VB | UT, 1),
    ROW(R8G8_UINT, S | BS | CA | BD, VB | UT, 1),
    ROW(R8G8_SINT, S | BS | CA | BD, VB | UT, 1),
    /* 4 byte-sized components */
    ROW(R8G8B8A8_UNORM, S | BS | FL | ST | CA | BD | BL, VB | UT | STB, 0),
    ROW(R8G8B8A8_SNORM, S | BS | FL | ST, VB | UT | STB, 0),
    ROW(R8G8B8A8_UINT, S | BS | ST | CA | BD, VB | UT | STB, 0),
    ROW(R8G8B8A8_SINT, S | BS | ST | CA | BD, VB | UT | STB, 0),
    ROW(R8G8B8A8_SRGB, S | BS | FL | CA | BD | BL, 0, 0),
    ROW(B8G8R8A8_UNORM, S | BS | FL | CA | BD | BL, VB | UT, 0),
    ROW(B8G8R8A8_SRGB, S | BS | FL | CA | BD | BL, 0, 0),
    ROW(A8B8G8R8_UNORM_PACK32, S | BS | FL | CA | BD | BL, VB | UT | STB, 0),
    ROW(A8B8G8R8_SNORM_PACK32, S | BS | FL, VB | UT | STB, 0),
    ROW(A8B8G8R8_UINT_PACK32, S | BS | CA | BD, VB | UT | STB, 0),
    ROW(A8B8G8R8_SINT_PACK32, S | BS | CA | BD, VB | UT | STB, 0),
    ROW(A8B8G8R8_SRGB_PACK32, S | BS | FL | CA | BD | BL, 0, 0),
    /* 10-, 12- and 14-bit components */
    ROW(A2B10G10R10_UNORM_PACK32, S | BS | FL | CA | BD | BL, VB | UT, 1),
    ROW(A2B10G10R10_UINT_PACK32, S | BS | CA | BD, UT, 1),
    /* 16-bit components */
    ROW(R16_UNORM, 0, VB, 1),
    ROW(R16_SNORM, 0, VB, 1),
    ROW(R16_UINT, S | BS | CA | BD, VB | UT, 1),
    ROW(R16_SINT, S | BS | CA | BD, VB | UT, 1),
    ROW(R16_SFLOAT, S | BS | FL | CA | BD | BL, VB | UT, 1),
    ROW(R16G16_UNORM, 0, VB, 1),
    ROW(R16G16_SNORM, 0, VB, 1),
    ROW(R16G16_UINT, S | BS | CA | BD, VB | UT, 1),
    ROW(R16G16_SINT, S | BS | CA | BD, VB | UT, 1),
    ROW(R16G16_SFLOAT, S | BS | FL | CA | BD | BL, VB | UT, 1),
    ROW(R16G16B16A16_UNORM, 0, VB, 1),
    ROW(R16G16B16A16_SNORM, 0, VB, 1),
    ROW(R16G16B16A16_UINT, S | BS | ST | CA | BD, VB | UT | STB, 0),
    ROW(R16G16B16A16_SINT, S | BS | ST | CA | BD, VB | UT | STB, 0),
    ROW(R16G16B16A16_SFLOAT, S | BS | FL | ST | CA | BD | BL, VB | UT | STB, 0),
    /* 32-bit components */
    ROW(R32_UINT, S | BS | ST | SA | CA | BD, VB | UT | STB, 0),
    ROW(R32_SINT, S | BS | ST | SA | CA | BD, VB | UT | STB, 0),
    ROW(R32_SFLOAT, S | BS | ST | CA | BD, VB | UT | STB, 0),
    ROW(R32G32_UINT, S | BS | ST | CA | BD, VB | UT | STB, 0),
    ROW(R32G32_SINT, S | BS | ST | CA | BD, VB | UT | STB, 0),
    ROW(R32G32_SFLOAT, S | BS | ST | CA | BD, VB | UT | STB, 0),
    ROW(R32G32B32_UINT, 0, VB, 0),
    ROW(R32G32B32_SINT, 0, VB, 0),
    ROW(R32G32B32_SFLOAT, 0, VB, 0),
    ROW(R32G32B32A32_UINT, S | BS | ST | CA | BD, VB | UT | STB, 0),
    ROW(R32G32B32A32_SINT, S | BS | ST | CA | BD, VB | UT | STB, 0),
    ROW(R32G32B32A32_SFLOAT, S | BS | ST | CA | BD, VB | UT | STB, 0),
    /* 64-bit/uneven components */
    ROW(B10G11R11_UFLOAT_PACK32, S | BS | FL, UT, 1),
    ROW(E5B9G9R9_UFLOAT_PACK32, S | BS | FL, 0, 0),
    /* Depth/stencil, VK_IMAGE_TYPE_2D (DEPTH_STENCIL_ATTACHMENT of the
     * other formats: depth_stencil_attachment_one_of) */
    DROW(D16_UNORM, S | BS | DS),
    DROW(D32_SFLOAT, S | BS),
};

static const char *feature_name(VkFormatFeatureFlags2 bit)
{
    static const struct {
        VkFormatFeatureFlags2 bit;
        const char *name;
    } names[] = {
        { S, "SAMPLED_IMAGE" },
        { BS, "BLIT_SRC" },
        { FL, "SAMPLED_IMAGE_FILTER_LINEAR" },
        { ST, "STORAGE_IMAGE" },
        { SA, "STORAGE_IMAGE_ATOMIC" },
        { CA, "COLOR_ATTACHMENT" },
        { BD, "BLIT_DST" },
        { BL, "COLOR_ATTACHMENT_BLEND" },
        { DS, "DEPTH_STENCIL_ATTACHMENT" },
        { VB, "VERTEX_BUFFER" },
        { UT, "UNIFORM_TEXEL_BUFFER" },
        { STB, "STORAGE_TEXEL_BUFFER" },
        { VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT, "TRANSFER_SRC" },
        { VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT, "TRANSFER_DST" },
    };
    size_t i;
    for (i = 0; i < CT_ARRAY_SIZE(names); i++)
        if (names[i].bit == bit)
            return names[i].name;
    return "?";
}

static void check_bits(struct ct *t, const char *format, const char *which, VkFormatFeatureFlags2 have,
                       VkFormatFeatureFlags2 want)
{
    VkFormatFeatureFlags2 missing = want & ~have, bit;
    for (bit = 1; missing && bit; bit <<= 1)
        if (missing & bit)
            ct_fail(t, __FILE__, __LINE__, "%s: %s lacks %s", format, which, feature_name(bit));
}

/* The usages the optimal features allow, each of which an image of every
 * listed type must accept (Required Format Support: "these features must be
 * supported for every VkImageType (including arrayed and cube variants)"). */
static void check_image_types(struct ct *t, const struct ct_env *e, VkFormat format, const char *name,
                              VkFormatFeatureFlags2 features, unsigned types)
{
    static const struct {
        VkFormatFeatureFlags2 feature;
        VkImageUsageFlags usage;
    } map[] = {
        { VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT, VK_IMAGE_USAGE_SAMPLED_BIT },
        { VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT, VK_IMAGE_USAGE_STORAGE_BIT },
        { VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT },
        { VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT },
        { VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT, VK_IMAGE_USAGE_TRANSFER_SRC_BIT },
        { VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT, VK_IMAGE_USAGE_TRANSFER_DST_BIT },
    };
    static const struct {
        unsigned type_bit;
        VkImageType type;
        VkImageCreateFlags flags;
        const char *what;
    } kinds[] = {
        { T1, VK_IMAGE_TYPE_1D, 0, "1D" },
        { T2, VK_IMAGE_TYPE_2D, 0, "2D" },
        { T2, VK_IMAGE_TYPE_2D, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, "cube" },
        { T3, VK_IMAGE_TYPE_3D, 0, "3D" },
    };
    unsigned i, k;
    for (k = 0; k < CT_ARRAY_SIZE(kinds); k++) {
        if (!(types & kinds[k].type_bit))
            continue;
        for (i = 0; i < CT_ARRAY_SIZE(map); i++) {
            VkImageFormatProperties p;
            VkResult r;
            if (!(features & map[i].feature))
                continue;
            /* attachments of 3D images are 2D views of them: not required */
            if (kinds[k].type == VK_IMAGE_TYPE_3D && (map[i].usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                                                      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)))
                continue;
            r = vkGetPhysicalDeviceImageFormatProperties(e->pd, format, kinds[k].type, VK_IMAGE_TILING_OPTIMAL,
                                                         map[i].usage, kinds[k].flags, &p);
            CT_CHECKF(t, r == VK_SUCCESS, "%s: %s image with usage 0x%x: %s", name, kinds[k].what, map[i].usage,
                      ct_result_name(r));
            if (r == VK_SUCCESS && kinds[k].flags)
                CT_CHECKF(t, p.maxArrayLayers >= 6, "%s: cube maxArrayLayers %u", name, p.maxArrayLayers);
        }
    }
}

static void check_required_format(struct ct *t)
{
    const struct row *row = ct_param(t);
    struct ct_env e;
    struct ct_features f;
    VkFormatFeatureFlags2 optimal, buffer, want;

    ct_env_init(t, &e);
    if (!e.graphics)
        ct_skip(t, "no graphics queue family: these are the requirements of graphics implementations");
    ct_get_features(&e, &f, 0);
    optimal = ct_format_features(&e, row->format, 0);
    buffer = ct_format_features(&e, row->format, 2);
    want = row->optimal;
    if (row->ext && f.f2.features.shaderStorageImageExtendedFormats)
        want |= ST;
    /* Vulkan 1.1: required SAMPLED_IMAGE implies TRANSFER_SRC and TRANSFER_DST */
    if ((want & S) && e.version >= VK_API_VERSION_1_1)
        want |= VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
    check_bits(t, row->name, "optimalTilingFeatures", optimal, want);
    check_bits(t, row->name, "bufferFeatures", buffer, row->buffer);
    if (row->types == T2 && row->buffer == 0 && (row->optimal & (S | DS))) /* depth/stencil rows */
        CT_CHECKF(t, buffer == 0, "%s: bufferFeatures 0x%llx, must be none for depth/stencil formats", row->name,
                  (unsigned long long)buffer);
    check_image_types(t, &e, row->format, row->name, want & optimal, row->types);
    ct_env_fini(&e);
}

/* ---- The chapter's other rules ---- */

static const VkFormat depth_stencil_formats[] = {
    VK_FORMAT_D16_UNORM,         VK_FORMAT_X8_D24_UNORM_PACK32, VK_FORMAT_D32_SFLOAT,        VK_FORMAT_S8_UINT,
    VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT,   VK_FORMAT_D32_SFLOAT_S8_UINT,
};

CT_TEST(depth_stencil_attachment_one_of,
        "Formats > Required Format Support: DEPTH_STENCIL_ATTACHMENT must be supported for at least one of "
        "X8_D24_UNORM_PACK32 and D32_SFLOAT, and for at least one of D24_UNORM_S8_UINT and D32_SFLOAT_S8_UINT; "
        "bufferFeatures must not support any features for depth/stencil formats",
        "vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties2")
{
    struct ct_env e;
    unsigned i;
    ct_env_init(t, &e);
    if (!e.graphics)
        ct_skip(t, "no graphics queue family");
    CT_CHECK(t, (ct_format_features(&e, VK_FORMAT_X8_D24_UNORM_PACK32, 0) | ct_format_features(&e, VK_FORMAT_D32_SFLOAT, 0)) &
                    DS);
    CT_CHECK(t, (ct_format_features(&e, VK_FORMAT_D24_UNORM_S8_UINT, 0) |
                 ct_format_features(&e, VK_FORMAT_D32_SFLOAT_S8_UINT, 0)) & DS);
    for (i = 0; i < CT_ARRAY_SIZE(depth_stencil_formats); i++)
        CT_CHECKF(t, ct_format_features(&e, depth_stencil_formats[i], 2) == 0, "format %d has bufferFeatures",
                  (int)depth_stencil_formats[i]);
    ct_env_fini(&e);
}

static const VkFormat bc_formats[] = {
    VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_BC1_RGB_SRGB_BLOCK, VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
    VK_FORMAT_BC1_RGBA_SRGB_BLOCK, VK_FORMAT_BC2_UNORM_BLOCK,    VK_FORMAT_BC2_SRGB_BLOCK,
    VK_FORMAT_BC3_UNORM_BLOCK,     VK_FORMAT_BC3_SRGB_BLOCK,     VK_FORMAT_BC4_UNORM_BLOCK,
    VK_FORMAT_BC4_SNORM_BLOCK,     VK_FORMAT_BC5_UNORM_BLOCK,    VK_FORMAT_BC5_SNORM_BLOCK,
    VK_FORMAT_BC6H_UFLOAT_BLOCK,   VK_FORMAT_BC6H_SFLOAT_BLOCK,  VK_FORMAT_BC7_UNORM_BLOCK,
    VK_FORMAT_BC7_SRGB_BLOCK,
};
static const VkFormat etc_formats[] = {
    VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,   VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,   VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
    VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,  VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
    VK_FORMAT_EAC_R11_UNORM_BLOCK,       VK_FORMAT_EAC_R11_SNORM_BLOCK,      VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
    VK_FORMAT_EAC_R11G11_SNORM_BLOCK,
};
#define ASTC_LDR_FIRST VK_FORMAT_ASTC_4x4_UNORM_BLOCK
#define ASTC_LDR_LAST VK_FORMAT_ASTC_12x12_SRGB_BLOCK

static int family_supported(const struct ct_env *e, const VkFormat *list, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++)
        if ((ct_format_features(e, list[i], 0) & (S | BS | FL)) != (S | BS | FL))
            return 0;
    return 1;
}

static int astc_ldr_supported(const struct ct_env *e)
{
    int f;
    for (f = ASTC_LDR_FIRST; f <= ASTC_LDR_LAST; f++)
        if ((ct_format_features(e, (VkFormat)f, 0) & (S | BS | FL)) != (S | BS | FL))
            return 0;
    return 1;
}

CT_TEST(compressed_family_one_of,
        "Formats > Required Format Support: SAMPLED_IMAGE, BLIT_SRC and SAMPLED_IMAGE_FILTER_LINEAR must be "
        "supported in optimalTilingFeatures for all the formats of at least one of the BC, ETC2/EAC and ASTC LDR tables",
        "vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties2")
{
    struct ct_env e;
    ct_env_init(t, &e);
    if (!e.graphics)
        ct_skip(t, "no graphics queue family");
    CT_CHECKF(t, family_supported(&e, bc_formats, CT_ARRAY_SIZE(bc_formats)) ||
                     family_supported(&e, etc_formats, CT_ARRAY_SIZE(etc_formats)) || astc_ldr_supported(&e),
              "no compressed family is fully supported");
    ct_env_fini(&e);
}

CT_TEST(compression_features_match_formats,
        "Features > textureCompressionETC2/ASTC_LDR/BC (and ASTC_HDR): if the feature is supported, SAMPLED_IMAGE, "
        "BLIT_SRC and SAMPLED_IMAGE_FILTER_LINEAR must be supported in optimalTilingFeatures for all its formats, "
        "and 2D images of them created (3D too for BC); at least one of the three LDR features is supported",
        "vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties2 "
        "vkGetPhysicalDeviceImageFormatProperties")
{
    struct ct_env e;
    struct ct_features f;
    VkImageFormatProperties p;
    size_t i;
    int k;

    ct_env_init(t, &e);
    if (!e.graphics)
        ct_skip(t, "no graphics queue family");
    ct_get_features(&e, &f, 0);
    CT_CHECK(t, f.f2.features.textureCompressionBC || f.f2.features.textureCompressionETC2 ||
                    f.f2.features.textureCompressionASTC_LDR);
    if (f.f2.features.textureCompressionBC) {
        CT_CHECK(t, family_supported(&e, bc_formats, CT_ARRAY_SIZE(bc_formats)));
        for (i = 0; i < CT_ARRAY_SIZE(bc_formats); i++) {
            CT_CHECKF(t, vkGetPhysicalDeviceImageFormatProperties(e.pd, bc_formats[i], VK_IMAGE_TYPE_2D,
                                                                  VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT,
                                                                  0, &p) == VK_SUCCESS,
                      "BC format %d: no 2D images", (int)bc_formats[i]);
            CT_CHECKF(t, vkGetPhysicalDeviceImageFormatProperties(e.pd, bc_formats[i], VK_IMAGE_TYPE_3D,
                                                                  VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT,
                                                                  0, &p) == VK_SUCCESS,
                      "BC format %d: no 3D images", (int)bc_formats[i]);
        }
    }
    if (f.f2.features.textureCompressionETC2) {
        CT_CHECK(t, family_supported(&e, etc_formats, CT_ARRAY_SIZE(etc_formats)));
        for (i = 0; i < CT_ARRAY_SIZE(etc_formats); i++)
            CT_CHECKF(t, vkGetPhysicalDeviceImageFormatProperties(e.pd, etc_formats[i], VK_IMAGE_TYPE_2D,
                                                                  VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT,
                                                                  0, &p) == VK_SUCCESS,
                      "ETC2/EAC format %d: no 2D images", (int)etc_formats[i]);
    }
    if (f.f2.features.textureCompressionASTC_LDR) {
        CT_CHECK(t, astc_ldr_supported(&e));
        for (k = ASTC_LDR_FIRST; k <= ASTC_LDR_LAST; k++)
            CT_CHECKF(t, vkGetPhysicalDeviceImageFormatProperties(e.pd, (VkFormat)k, VK_IMAGE_TYPE_2D,
                                                                  VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT,
                                                                  0, &p) == VK_SUCCESS,
                      "ASTC format %d: no 2D images", k);
    }
    if (e.version >= VK_API_VERSION_1_3 && f.v13.textureCompressionASTC_HDR) {
        for (k = VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK; k <= VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK; k++)
            CT_CHECKF(t, (ct_format_features(&e, (VkFormat)k, 0) & (S | BS | FL)) == (S | BS | FL),
                      "ASTC HDR format %d", k);
    }
    ct_env_fini(&e);
}

/* Every format value of core Vulkan 1.4 (the 1.0 range, Y'CbCr, 4444,
 * ASTC HDR, A1B5G5R5, A8). */
static size_t all_formats(VkFormat *out, size_t max)
{
    static const struct {
        int first, last;
    } ranges[] = {
        { 1, VK_FORMAT_ASTC_12x12_SRGB_BLOCK },
        { VK_FORMAT_G8B8G8R8_422_UNORM, VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM },
        { VK_FORMAT_G8_B8R8_2PLANE_444_UNORM, VK_FORMAT_G16_B16R16_2PLANE_444_UNORM },
        { VK_FORMAT_A4R4G4B4_UNORM_PACK16, VK_FORMAT_A4B4G4R4_UNORM_PACK16 },
        { VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK, VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK },
        { VK_FORMAT_A1B5G5R5_UNORM_PACK16, VK_FORMAT_A8_UNORM },
    };
    size_t n = 0, r;
    int f;
    for (r = 0; r < CT_ARRAY_SIZE(ranges); r++)
        for (f = ranges[r].first; f <= ranges[r].last && n < max; f++)
            out[n++] = (VkFormat)f;
    return n;
}

static int is_depth(VkFormat f)
{
    return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_X8_D24_UNORM_PACK32 || f == VK_FORMAT_D32_SFLOAT ||
           f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

static int is_color(VkFormat f)
{
    return !is_depth(f) && f != VK_FORMAT_S8_UINT;
}

CT_TEST(format_properties_consistent,
        "Formats > Format Properties: vkGetPhysicalDeviceFormatProperties2::formatProperties equals "
        "vkGetPhysicalDeviceFormatProperties, and VkFormatProperties3 holds the same flags in the 31 bits "
        "VkFormatFeatureFlagBits defines (derived: VkFormatFeatureFlagBits2 shares their values)",
        "vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties2")
{
    static VkFormat formats[400];
    struct ct_env e;
    size_t n, i;
    ct_env_init(t, &e);
    n = all_formats(formats, CT_ARRAY_SIZE(formats));
    for (i = 0; i < n; i++) {
        VkFormatProperties p1;
        VkFormatProperties2 p2;
        VkFormatProperties3 p3;
        memset(&p1, 0xA5, sizeof(p1));
        vkGetPhysicalDeviceFormatProperties(e.pd, formats[i], &p1);
        memset(&p3, 0, sizeof(p3));
        p3.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3;
        memset(&p2, 0, sizeof(p2));
        p2.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
        p2.pNext = e.version >= VK_API_VERSION_1_3 ? &p3 : NULL;
        vkGetPhysicalDeviceFormatProperties2(e.pd, formats[i], &p2);
        CT_CHECKF(t, memcmp(&p1, &p2.formatProperties, sizeof(p1)) == 0, "format %d: Properties2 differs",
                  (int)formats[i]);
        if (e.version >= VK_API_VERSION_1_3)
            CT_CHECKF(t, (p3.optimalTilingFeatures & 0x7FFFFFFFull) == (p1.optimalTilingFeatures & 0x7FFFFFFFu) &&
                             (p3.linearTilingFeatures & 0x7FFFFFFFull) == (p1.linearTilingFeatures & 0x7FFFFFFFu) &&
                             (p3.bufferFeatures & 0x7FFFFFFFull) == (p1.bufferFeatures & 0x7FFFFFFFu),
                      "format %d: VkFormatProperties 0x%x/0x%x/0x%x, VkFormatProperties3 0x%llx/0x%llx/0x%llx",
                      (int)formats[i], p1.optimalTilingFeatures, p1.linearTilingFeatures, p1.bufferFeatures,
                      (unsigned long long)p3.optimalTilingFeatures, (unsigned long long)p3.linearTilingFeatures,
                      (unsigned long long)p3.bufferFeatures);
    }
    ct_env_fini(&e);
}

CT_TEST(depth_comparison,
        "Formats > Depth Comparison Format Support: with Vulkan 1.3, a depth/stencil format with a depth component "
        "supporting SAMPLED_IMAGE must support SAMPLED_IMAGE_DEPTH_COMPARISON",
        "vkGetPhysicalDeviceFormatProperties2")
{
    struct ct_env e;
    unsigned i, w;
    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_3)
        ct_skip(t, "the rule applies from Vulkan 1.3");
    for (i = 0; i < CT_ARRAY_SIZE(depth_stencil_formats); i++) {
        if (!is_depth(depth_stencil_formats[i]))
            continue;
        for (w = 0; w < 2; w++) {
            VkFormatFeatureFlags2 f = ct_format_features(&e, depth_stencil_formats[i], (int)w);
            if (f & S)
                CT_CHECKF(t, f & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT, "format %d (%s tiling)",
                          (int)depth_stencil_formats[i], w ? "linear" : "optimal");
        }
    }
    ct_env_fini(&e);
}

CT_TEST(host_image_transfer,
        "Formats > Required Format Support: with hostImageCopy, a color format supporting SAMPLED_IMAGE in "
        "optimalTilingFeatures or linearTilingFeatures must support HOST_IMAGE_TRANSFER there too",
        "vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFormatProperties2")
{
    static VkFormat formats[400];
    struct ct_env e;
    struct ct_features f;
    size_t n, i;
    unsigned w;
    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_4)
        ct_skip(t, "hostImageCopy is a Vulkan 1.4 feature");
    ct_get_features(&e, &f, 0);
    if (!f.v14.hostImageCopy)
        ct_skip(t, "hostImageCopy is not supported");
    n = all_formats(formats, CT_ARRAY_SIZE(formats));
    for (i = 0; i < n; i++) {
        if (!is_color(formats[i]))
            continue;
        for (w = 0; w < 2; w++) {
            VkFormatFeatureFlags2 ff = ct_format_features(&e, formats[i], (int)w);
            if (ff & S)
                CT_CHECKF(t, ff & VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT, "format %d (%s tiling)", (int)formats[i],
                          w ? "linear" : "optimal");
        }
    }
    ct_env_fini(&e);
}

/* Formats > SPIR-V and Vulkan Image Format Compatibility */
static const VkFormat spirv_formats[] = {
    VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM,
    VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
    VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_R16G16_UNORM,
    VK_FORMAT_R8G8_UNORM, VK_FORMAT_R16_UNORM, VK_FORMAT_R8_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16_SNORM,
    VK_FORMAT_R8G8_SNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R32G32B32A32_SINT,
    VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R32_SINT, VK_FORMAT_R32G32_SINT,
    VK_FORMAT_R16G16_SINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R16_SINT, VK_FORMAT_R8_SINT, VK_FORMAT_R32G32B32A32_UINT,
    VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R32_UINT, VK_FORMAT_A2B10G10R10_UINT_PACK32,
    VK_FORMAT_R32G32_UINT, VK_FORMAT_R16G16_UINT, VK_FORMAT_R8G8_UINT, VK_FORMAT_R16_UINT, VK_FORMAT_R8_UINT,
    VK_FORMAT_R64_UINT, VK_FORMAT_R64_SINT,
};

CT_TEST(storage_without_format,
        "Formats > Formats Without Shader Storage Format: with Vulkan 1.3, a format supporting neither "
        "STORAGE_READ_WITHOUT_FORMAT nor STORAGE_WRITE_WITHOUT_FORMAT must not report STORAGE_IMAGE or "
        "STORAGE_TEXEL_BUFFER unless it is listed in the SPIR-V and Vulkan Image Format Compatibility table",
        "vkGetPhysicalDeviceFormatProperties2")
{
    static VkFormat formats[400];
    struct ct_env e;
    size_t n, i, k;
    unsigned w;
    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_3)
        ct_skip(t, "the rule applies from Vulkan 1.3");
    n = all_formats(formats, CT_ARRAY_SIZE(formats));
    for (i = 0; i < n; i++) {
        int listed = 0;
        for (k = 0; k < CT_ARRAY_SIZE(spirv_formats); k++)
            listed |= spirv_formats[k] == formats[i];
        if (listed)
            continue;
        for (w = 0; w < 3; w++) {
            VkFormatFeatureFlags2 ff = ct_format_features(&e, formats[i], (int)w);
            if (ff & (VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT | VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT))
                continue;
            CT_CHECKF(t, !(ff & (ST | STB)), "format %d reports storage (%s) without a format token or without-format access",
                      (int)formats[i], w == 2 ? "buffer" : w ? "linear" : "optimal");
        }
    }
    ct_env_fini(&e);
}

CT_TEST(filter_minmax_formats,
        "Limits > filterMinmaxSingleComponentFormats: if VK_TRUE, R8_UNORM, R8_SNORM, R16_UNORM, R16_SNORM, "
        "R16_SFLOAT, R32_SFLOAT and the depth formats must support SAMPLED_IMAGE_FILTER_MINMAX with optimal tiling "
        "if they support SAMPLED_IMAGE",
        "vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceFormatProperties2")
{
    static const VkFormat list[] = {
        VK_FORMAT_R8_UNORM,         VK_FORMAT_R8_SNORM,          VK_FORMAT_R16_UNORM,         VK_FORMAT_R16_SNORM,
        VK_FORMAT_R16_SFLOAT,       VK_FORMAT_R32_SFLOAT,        VK_FORMAT_D16_UNORM,         VK_FORMAT_X8_D24_UNORM_PACK32,
        VK_FORMAT_D32_SFLOAT,       VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT,
    };
    struct ct_env e;
    struct ct_properties p;
    unsigned i;
    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_2)
        ct_skip(t, "VkPhysicalDeviceVulkan12Properties needs Vulkan 1.2");
    ct_get_properties(&e, &p, 0);
    if (!p.v12.filterMinmaxSingleComponentFormats)
        ct_skip(t, "filterMinmaxSingleComponentFormats is VK_FALSE");
    for (i = 0; i < CT_ARRAY_SIZE(list); i++) {
        VkFormatFeatureFlags2 f = ct_format_features(&e, list[i], 0);
        if (f & S)
            CT_CHECKF(t, f & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_MINMAX_BIT, "format %d", (int)list[i]);
    }
    ct_env_fini(&e);
}

/* Whether the physical device supports samplerYcbcrConversion (Vulkan 1.1). */
static int ycbcr_supported(const struct ct_env *e)
{
    VkPhysicalDeviceSamplerYcbcrConversionFeatures y;
    VkPhysicalDeviceFeatures2 f2;
    if (e->version < VK_API_VERSION_1_1)
        return 0;
    memset(&y, 0, sizeof(y));
    y.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES;
    memset(&f2, 0, sizeof(f2));
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &y;
    vkGetPhysicalDeviceFeatures2(e->pd, &f2);
    return y.samplerYcbcrConversion != VK_FALSE;
}

CT_TEST(ycbcr_required_formats,
        "Formats > Required Format Support (formats requiring sampler Y'CbCr conversion): with "
        "samplerYcbcrConversion, G8_B8_R8_3PLANE_420_UNORM and G8_B8R8_2PLANE_420_UNORM must support SAMPLED_IMAGE, "
        "TRANSFER_SRC, TRANSFER_DST and a chroma location (MIDPOINT_CHROMA_SAMPLES or COSITED_CHROMA_SAMPLES) in "
        "optimalTilingFeatures",
        "vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties2")
{
    static const VkFormat required[] = { VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM };
    struct ct_env e;
    unsigned i;
    ct_env_init(t, &e);
    if (!ycbcr_supported(&e))
        ct_skip(t, "samplerYcbcrConversion is not supported");
    for (i = 0; i < CT_ARRAY_SIZE(required); i++) {
        VkFormatFeatureFlags2 f = ct_format_features(&e, required[i], 0);
        VkFormatFeatureFlags2 need = S | VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
        CT_CHECKF(t, (f & need) == need, "format %d: optimalTilingFeatures 0x%llx lack 0x%llx", (int)required[i],
                  (unsigned long long)f, (unsigned long long)(need & ~f));
        CT_CHECKF(t, f & (VK_FORMAT_FEATURE_2_MIDPOINT_CHROMA_SAMPLES_BIT | VK_FORMAT_FEATURE_2_COSITED_CHROMA_SAMPLES_BIT),
                  "format %d: no chroma location", (int)required[i]);
    }
    ct_env_fini(&e);
}

CT_TEST(ycbcr_disjoint_multiplanar,
        "Formats > Format Feature Flags: VK_FORMAT_FEATURE_DISJOINT_BIT specifies that a multi-planar image can be "
        "made disjoint; an implementation must not set it for single-plane formats",
        "vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties2")
{
    static VkFormat formats[400];
    struct ct_env e;
    size_t n, i;
    unsigned w;
    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_1)
        ct_skip(t, "VK_FORMAT_FEATURE_DISJOINT_BIT is Vulkan 1.1");
    n = all_formats(formats, CT_ARRAY_SIZE(formats));
    for (i = 0; i < n; i++)
        for (w = 0; w < 3; w++)
            if (ct_ycbcr_planes(formats[i]) < 2)
                CT_CHECKF(t, !(ct_format_features(&e, formats[i], (int)w) & VK_FORMAT_FEATURE_2_DISJOINT_BIT),
                          "single-plane format %d reports DISJOINT (%s features)", (int)formats[i],
                          w == 2 ? "buffer" : w ? "linear" : "optimal");
    ct_env_fini(&e);
}

static const struct ct_test *const fixed_tests[] = {
    CT_REF(depth_stencil_attachment_one_of), CT_REF(compressed_family_one_of), CT_REF(compression_features_match_formats),
    CT_REF(format_properties_consistent),    CT_REF(depth_comparison),         CT_REF(host_image_transfer),
    CT_REF(storage_without_format),          CT_REF(filter_minmax_formats),    CT_REF(ycbcr_required_formats),
    CT_REF(ycbcr_disjoint_multiplanar),
};

void ct_suite_formats(struct ct_registry *r);
void ct_suite_formats(struct ct_registry *r)
{
    static struct ct_test tests[CT_ARRAY_SIZE(rows)];
    static const struct ct_test *ptrs[CT_ARRAY_SIZE(rows)];
    static char specs[CT_ARRAY_SIZE(rows)][256];
    size_t i;

    for (i = 0; i < CT_ARRAY_SIZE(rows); i++) {
        snprintf(specs[i], sizeof(specs[i]),
                 "Formats > Required Format Support: %s must support the features its table row marks, for every "
                 "image type the table covers",
                 rows[i].name);
        tests[i].name = rows[i].name;
        tests[i].fn = check_required_format;
        tests[i].spec = specs[i];
        tests[i].commands = "vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFormatProperties "
                            "vkGetPhysicalDeviceFormatProperties2 vkGetPhysicalDeviceImageFormatProperties";
        tests[i].param = &rows[i];
        ptrs[i] = &tests[i];
    }
    ct_register(r, "formats", ptrs, CT_ARRAY_SIZE(ptrs));
    ct_register(r, "formats", fixed_tests, CT_ARRAY_SIZE(fixed_tests));
}
