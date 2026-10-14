/* PLASMAVK_QUAD_MERGING (CONFIGURATION): a build that lets Apple GPUs pack
 * fragments of two triangles into one 2x2 quad. Fragment shaders using
 * derivatives don't read their barycentric coordinates, pipeline cache data
 * doesn't cross configurations, and the device claims no conformance. The
 * default build's side: test_device.c (driver properties) and
 * test_graphics.c (test_derivatives_per_triangle).
 *
 * Shaders (test_graphics.c's): GLSL in the comment above each array,
 * compiled with glslangValidator -V. */
#define PLASMAVK_QUAD_MERGING
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <string.h>

#include "../pvk_test.h"
#include "../pvk_msl_test.h"
#include "../pvk_msl_render.h"

/*
 * #version 450
 * layout(location = 0) flat out float v_tri;
 * void main()
 * {
 *     const vec2 p[4] = vec2[](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
 *     const float w[4] = float[](1.0, 1.25, 0.75, 1.5);
 *     gl_Position = vec4(p[gl_VertexIndex] * w[gl_VertexIndex], 0.0, w[gl_VertexIndex]);
 *     v_tri = float(gl_VertexIndex);
 * }
 */
static const uint32_t spv_quad_w_vs[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x0000003c, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0008000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x0000001b, 0x00000039,
    0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00060005,
    0x0000000b, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x0000000b, 0x00000000,
    0x505f6c67, 0x7469736f, 0x006e6f69, 0x00070006, 0x0000000b, 0x00000001, 0x505f6c67, 0x746e696f,
    0x657a6953, 0x00000000, 0x00070006, 0x0000000b, 0x00000002, 0x435f6c67, 0x4470696c, 0x61747369,
    0x0065636e, 0x00070006, 0x0000000b, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e,
    0x00030005, 0x0000000d, 0x00000000, 0x00060005, 0x0000001b, 0x565f6c67, 0x65747265, 0x646e4978,
    0x00007865, 0x00050005, 0x0000001e, 0x65646e69, 0x6c626178, 0x00000065, 0x00050005, 0x00000029,
    0x65646e69, 0x6c626178, 0x00000065, 0x00050005, 0x00000030, 0x65646e69, 0x6c626178, 0x00000065,
    0x00040005, 0x00000039, 0x72745f76, 0x00000069, 0x00030047, 0x0000000b, 0x00000002, 0x00050048,
    0x0000000b, 0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x0000000b, 0x00000001, 0x0000000b,
    0x00000001, 0x00050048, 0x0000000b, 0x00000002, 0x0000000b, 0x00000003, 0x00050048, 0x0000000b,
    0x00000003, 0x0000000b, 0x00000004, 0x00040047, 0x0000001b, 0x0000000b, 0x0000002a, 0x00030047,
    0x00000039, 0x0000000e, 0x00040047, 0x00000039, 0x0000001e, 0x00000000, 0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007,
    0x00000006, 0x00000004, 0x00040015, 0x00000008, 0x00000020, 0x00000000, 0x0004002b, 0x00000008,
    0x00000009, 0x00000001, 0x0004001c, 0x0000000a, 0x00000006, 0x00000009, 0x0006001e, 0x0000000b,
    0x00000007, 0x00000006, 0x0000000a, 0x0000000a, 0x00040020, 0x0000000c, 0x00000003, 0x0000000b,
    0x0004003b, 0x0000000c, 0x0000000d, 0x00000003, 0x00040015, 0x0000000e, 0x00000020, 0x00000001,
    0x0004002b, 0x0000000e, 0x0000000f, 0x00000000, 0x00040017, 0x00000010, 0x00000006, 0x00000002,
    0x0004002b, 0x00000008, 0x00000011, 0x00000004, 0x0004001c, 0x00000012, 0x00000010, 0x00000011,
    0x0004002b, 0x00000006, 0x00000013, 0xbf800000, 0x0005002c, 0x00000010, 0x00000014, 0x00000013,
    0x00000013, 0x0004002b, 0x00000006, 0x00000015, 0x3f800000, 0x0005002c, 0x00000010, 0x00000016,
    0x00000015, 0x00000013, 0x0005002c, 0x00000010, 0x00000017, 0x00000013, 0x00000015, 0x0005002c,
    0x00000010, 0x00000018, 0x00000015, 0x00000015, 0x0007002c, 0x00000012, 0x00000019, 0x00000014,
    0x00000016, 0x00000017, 0x00000018, 0x00040020, 0x0000001a, 0x00000001, 0x0000000e, 0x0004003b,
    0x0000001a, 0x0000001b, 0x00000001, 0x00040020, 0x0000001d, 0x00000007, 0x00000012, 0x00040020,
    0x0000001f, 0x00000007, 0x00000010, 0x0004001c, 0x00000022, 0x00000006, 0x00000011, 0x0004002b,
    0x00000006, 0x00000023, 0x3fa00000, 0x0004002b, 0x00000006, 0x00000024, 0x3f400000, 0x0004002b,
    0x00000006, 0x00000025, 0x3fc00000, 0x0007002c, 0x00000022, 0x00000026, 0x00000015, 0x00000023,
    0x00000024, 0x00000025, 0x00040020, 0x00000028, 0x00000007, 0x00000022, 0x00040020, 0x0000002a,
    0x00000007, 0x00000006, 0x0004002b, 0x00000006, 0x0000002e, 0x00000000, 0x00040020, 0x00000036,
    0x00000003, 0x00000007, 0x00040020, 0x00000038, 0x00000003, 0x00000006, 0x0004003b, 0x00000038,
    0x00000039, 0x00000003, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
    0x00000005, 0x0004003b, 0x0000001d, 0x0000001e, 0x00000007, 0x0004003b, 0x00000028, 0x00000029,
    0x00000007, 0x0004003b, 0x00000028, 0x00000030, 0x00000007, 0x0004003d, 0x0000000e, 0x0000001c,
    0x0000001b, 0x0003003e, 0x0000001e, 0x00000019, 0x00050041, 0x0000001f, 0x00000020, 0x0000001e,
    0x0000001c, 0x0004003d, 0x00000010, 0x00000021, 0x00000020, 0x0004003d, 0x0000000e, 0x00000027,
    0x0000001b, 0x0003003e, 0x00000029, 0x00000026, 0x00050041, 0x0000002a, 0x0000002b, 0x00000029,
    0x00000027, 0x0004003d, 0x00000006, 0x0000002c, 0x0000002b, 0x0005008e, 0x00000010, 0x0000002d,
    0x00000021, 0x0000002c, 0x0004003d, 0x0000000e, 0x0000002f, 0x0000001b, 0x0003003e, 0x00000030,
    0x00000026, 0x00050041, 0x0000002a, 0x00000031, 0x00000030, 0x0000002f, 0x0004003d, 0x00000006,
    0x00000032, 0x00000031, 0x00050051, 0x00000006, 0x00000033, 0x0000002d, 0x00000000, 0x00050051,
    0x00000006, 0x00000034, 0x0000002d, 0x00000001, 0x00070050, 0x00000007, 0x00000035, 0x00000033,
    0x00000034, 0x0000002e, 0x00000032, 0x00050041, 0x00000036, 0x00000037, 0x0000000d, 0x0000000f,
    0x0003003e, 0x00000037, 0x00000035, 0x0004003d, 0x0000000e, 0x0000003a, 0x0000001b, 0x0004006f,
    0x00000006, 0x0000003b, 0x0000003a, 0x0003003e, 0x00000039, 0x0000003b, 0x000100fd, 0x00010038,
};

/*
 * #version 450
 * layout(location = 0) flat in float v_tri;
 * layout(location = 0) out vec4 o;
 * void main()
 * {
 *     o = vec4(dFdx(v_tri), dFdy(v_tri), v_tri, 1.0);
 * }
 */
static const uint32_t spv_flat_derivs_fs[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x00000013, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000b, 0x00030010,
    0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00030005, 0x00000009, 0x0000006f, 0x00040005, 0x0000000b, 0x72745f76, 0x00000069,
    0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00030047, 0x0000000b, 0x0000000e, 0x00040047,
    0x0000000b, 0x0000001e, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
    0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020,
    0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00040020,
    0x0000000a, 0x00000001, 0x00000006, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b,
    0x00000006, 0x00000011, 0x3f800000, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005, 0x0004003d, 0x00000006, 0x0000000c, 0x0000000b, 0x000400cf, 0x00000006,
    0x0000000d, 0x0000000c, 0x0004003d, 0x00000006, 0x0000000e, 0x0000000b, 0x000400d0, 0x00000006,
    0x0000000f, 0x0000000e, 0x0004003d, 0x00000006, 0x00000010, 0x0000000b, 0x00070050, 0x00000007,
    0x00000012, 0x0000000d, 0x0000000f, 0x00000010, 0x00000011, 0x0003003e, 0x00000009, 0x00000012,
    0x000100fd, 0x00010038,
};

/* conformanceVersion 0.0.0.0 and a driverInfo that says why; a pipeline
 * cache UUID of its own. */
static void test_driver_properties(void)
{
    VkPhysicalDeviceDriverProperties drv;
    VkPhysicalDeviceProperties2 props;
    VkInstanceCreateInfo info;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    uint8_t uuid[VK_UUID_SIZE];
    char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE + 32];
    uint32_t count = 1;

    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    REQUIRE_VK(vkCreateInstance(&info, NULL, &instance));
    CHECK_VK(vkEnumeratePhysicalDevices(instance, &count, &physical_device), VK_SUCCESS);
    if (count == 1) {
        memset(&drv, 0, sizeof(drv));
        drv.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        memset(&props, 0, sizeof(props));
        props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props.pNext = &drv;
        vkGetPhysicalDeviceProperties2(physical_device, &props);
        CHECK(strcmp(drv.driverName, "PlasmaVK") == 0);
        CHECK(drv.conformanceVersion.major == 0 && drv.conformanceVersion.minor == 0 &&
              drv.conformanceVersion.subminor == 0 && drv.conformanceVersion.patch == 0);
        CHECK(strstr(drv.driverInfo, "PLASMAVK_QUAD_MERGING, not conformant") != NULL);
        /* the default build's UUID (section 7) */
        snprintf(name, sizeof(name), "%s %d.%d.%d format %d", props.properties.deviceName, PLASMAVK_VERSION_MAJOR,
                 PLASMAVK_VERSION_MINOR, PLASMAVK_VERSION_PATCH, PVK_PIPELINE_CACHE_FORMAT);
        pvk_make_uuid(uuid, "PlasmaVK pipeline cache", name);
        CHECK(memcmp(uuid, props.properties.pipelineCacheUUID, VK_UUID_SIZE) != 0);
    }
    vkDestroyInstance(instance, NULL);
}

/* The fragment shader takes no [[barycentric_coord]] for its derivatives,
 * and compiles. */
static void test_no_barycentric_read(void)
{
    struct pvk_msl_result vs, fs;
    id function;
    VkResult r = pvk_spirv_to_msl_graphics(spv_quad_w_vs, PVK_ARRAY_SIZE(spv_quad_w_vs), NULL, NULL,
                                           spv_flat_derivs_fs, PVK_ARRAY_SIZE(spv_flat_derivs_fs), NULL, NULL,
                                           &vs, &fs);
    REQUIRE(r == VK_SUCCESS);
    CHECK(strstr(fs.msl, "dfdx") != NULL);
    CHECK(strstr(fs.msl, "barycentric_coord") == NULL);
    function = mr_function(fs.msl, fs.entry_point);
    CHECK(function != NULL);
    pvk_release(function);
    pvk_msl_result_free(&vs);
    pvk_msl_result_free(&fs);
}

int main(void)
{
    if (!mt_init())
        return 1;
    RUN(test_driver_properties);
    RUN(test_no_barycentric_read);
    mt_shutdown();
    return TEST_RESULT();
}
