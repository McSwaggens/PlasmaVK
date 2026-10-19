/* PLASMAVK_FAST_MATH (CONFIGURATION): a build that compiles shaders in
 * Vulkan's default floating-point environment with MTLMathModeFast. Shaders
 * with float controls (here NoContraction) keep MTLMathModeSafe, pipeline
 * cache data crosses configurations (the MSL is the same) and the device
 * claims no conformance. The default build's side: test_device.c (driver
 * properties) and test_spirv_msl_caps.c (test_math_mode).
 *
 * Shaders (test_spirv_msl_caps.c's): GLSL in the comment above each array,
 * compiled with glslangValidator -V --target-env vulkan1.3. */
#define PLASMAVK_FAST_MATH
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <string.h>

#include "../pvk_test.h"
#include "../pvk_msl_test.h"

/*
 * #version 450
 * layout(local_size_x = 1) in;
 * layout(std430, binding = 0) buffer B { float a, b, c, r; } buf;
 * void main()
 * {
 *     buf.r = buf.a * buf.b + buf.c;
 * }
 */
static const uint32_t spv_math_plain[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x0000001d, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00060010, 0x00000004,
    0x00000011, 0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000007, 0x00000042, 0x00040006, 0x00000007,
    0x00000000, 0x00000061, 0x00040006, 0x00000007, 0x00000001, 0x00000062, 0x00040006, 0x00000007,
    0x00000002, 0x00000063, 0x00040006, 0x00000007, 0x00000003, 0x00000072, 0x00030005, 0x00000009,
    0x00667562, 0x00030047, 0x00000007, 0x00000002, 0x00050048, 0x00000007, 0x00000000, 0x00000023,
    0x00000000, 0x00050048, 0x00000007, 0x00000001, 0x00000023, 0x00000004, 0x00050048, 0x00000007,
    0x00000002, 0x00000023, 0x00000008, 0x00050048, 0x00000007, 0x00000003, 0x00000023, 0x0000000c,
    0x00040047, 0x00000009, 0x00000021, 0x00000000, 0x00040047, 0x00000009, 0x00000022, 0x00000000,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
    0x0006001e, 0x00000007, 0x00000006, 0x00000006, 0x00000006, 0x00000006, 0x00040020, 0x00000008,
    0x0000000c, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x0000000c, 0x00040015, 0x0000000a,
    0x00000020, 0x00000001, 0x0004002b, 0x0000000a, 0x0000000b, 0x00000003, 0x0004002b, 0x0000000a,
    0x0000000c, 0x00000000, 0x00040020, 0x0000000d, 0x0000000c, 0x00000006, 0x0004002b, 0x0000000a,
    0x00000010, 0x00000001, 0x0004002b, 0x0000000a, 0x00000014, 0x00000002, 0x00040015, 0x00000019,
    0x00000020, 0x00000000, 0x00040017, 0x0000001a, 0x00000019, 0x00000003, 0x0004002b, 0x00000019,
    0x0000001b, 0x00000001, 0x0006002c, 0x0000001a, 0x0000001c, 0x0000001b, 0x0000001b, 0x0000001b,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041,
    0x0000000d, 0x0000000e, 0x00000009, 0x0000000c, 0x0004003d, 0x00000006, 0x0000000f, 0x0000000e,
    0x00050041, 0x0000000d, 0x00000011, 0x00000009, 0x00000010, 0x0004003d, 0x00000006, 0x00000012,
    0x00000011, 0x00050085, 0x00000006, 0x00000013, 0x0000000f, 0x00000012, 0x00050041, 0x0000000d,
    0x00000015, 0x00000009, 0x00000014, 0x0004003d, 0x00000006, 0x00000016, 0x00000015, 0x00050081,
    0x00000006, 0x00000017, 0x00000013, 0x00000016, 0x00050041, 0x0000000d, 0x00000018, 0x00000009,
    0x0000000b, 0x0003003e, 0x00000018, 0x00000017, 0x000100fd, 0x00010038,
};

/*
 * #version 450
 * layout(local_size_x = 1) in;
 * layout(std430, binding = 0) buffer B { float a, b, c, r, s; } buf;
 * void main()
 * {
 *     precise float t = buf.a * buf.b + buf.c;
 *     buf.r = t;
 *     buf.s = buf.a * buf.b + buf.c;
 * }
 */
static const uint32_t spv_math_precise[] = {
    0x07230203, 0x00010600, 0x0008000b, 0x0000002a, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00060010, 0x00000004,
    0x00000011, 0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008, 0x00000074, 0x00030005, 0x00000009,
    0x00000042, 0x00040006, 0x00000009, 0x00000000, 0x00000061, 0x00040006, 0x00000009, 0x00000001,
    0x00000062, 0x00040006, 0x00000009, 0x00000002, 0x00000063, 0x00040006, 0x00000009, 0x00000003,
    0x00000072, 0x00040006, 0x00000009, 0x00000004, 0x00000073, 0x00030005, 0x0000000b, 0x00667562,
    0x00030047, 0x00000009, 0x00000002, 0x00050048, 0x00000009, 0x00000000, 0x00000023, 0x00000000,
    0x00050048, 0x00000009, 0x00000001, 0x00000023, 0x00000004, 0x00050048, 0x00000009, 0x00000002,
    0x00000023, 0x00000008, 0x00050048, 0x00000009, 0x00000003, 0x00000023, 0x0000000c, 0x00050048,
    0x00000009, 0x00000004, 0x00000023, 0x00000010, 0x00040047, 0x0000000b, 0x00000021, 0x00000000,
    0x00040047, 0x0000000b, 0x00000022, 0x00000000, 0x00030047, 0x00000014, 0x0000002a, 0x00030047,
    0x00000018, 0x0000002a, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
    0x00000006, 0x00000020, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x0007001e, 0x00000009,
    0x00000006, 0x00000006, 0x00000006, 0x00000006, 0x00000006, 0x00040020, 0x0000000a, 0x0000000c,
    0x00000009, 0x0004003b, 0x0000000a, 0x0000000b, 0x0000000c, 0x00040015, 0x0000000c, 0x00000020,
    0x00000001, 0x0004002b, 0x0000000c, 0x0000000d, 0x00000000, 0x00040020, 0x0000000e, 0x0000000c,
    0x00000006, 0x0004002b, 0x0000000c, 0x00000011, 0x00000001, 0x0004002b, 0x0000000c, 0x00000015,
    0x00000002, 0x0004002b, 0x0000000c, 0x00000019, 0x00000003, 0x0004002b, 0x0000000c, 0x0000001c,
    0x00000004, 0x00040015, 0x00000026, 0x00000020, 0x00000000, 0x00040017, 0x00000027, 0x00000026,
    0x00000003, 0x0004002b, 0x00000026, 0x00000028, 0x00000001, 0x0006002c, 0x00000027, 0x00000029,
    0x00000028, 0x00000028, 0x00000028, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x00050041, 0x0000000e,
    0x0000000f, 0x0000000b, 0x0000000d, 0x0004003d, 0x00000006, 0x00000010, 0x0000000f, 0x00050041,
    0x0000000e, 0x00000012, 0x0000000b, 0x00000011, 0x0004003d, 0x00000006, 0x00000013, 0x00000012,
    0x00050085, 0x00000006, 0x00000014, 0x00000010, 0x00000013, 0x00050041, 0x0000000e, 0x00000016,
    0x0000000b, 0x00000015, 0x0004003d, 0x00000006, 0x00000017, 0x00000016, 0x00050081, 0x00000006,
    0x00000018, 0x00000014, 0x00000017, 0x0003003e, 0x00000008, 0x00000018, 0x0004003d, 0x00000006,
    0x0000001a, 0x00000008, 0x00050041, 0x0000000e, 0x0000001b, 0x0000000b, 0x00000019, 0x0003003e,
    0x0000001b, 0x0000001a, 0x00050041, 0x0000000e, 0x0000001d, 0x0000000b, 0x0000000d, 0x0004003d,
    0x00000006, 0x0000001e, 0x0000001d, 0x00050041, 0x0000000e, 0x0000001f, 0x0000000b, 0x00000011,
    0x0004003d, 0x00000006, 0x00000020, 0x0000001f, 0x00050085, 0x00000006, 0x00000021, 0x0000001e,
    0x00000020, 0x00050041, 0x0000000e, 0x00000022, 0x0000000b, 0x00000015, 0x0004003d, 0x00000006,
    0x00000023, 0x00000022, 0x00050081, 0x00000006, 0x00000024, 0x00000021, 0x00000023, 0x00050041,
    0x0000000e, 0x00000025, 0x0000000b, 0x0000001c, 0x0003003e, 0x00000025, 0x00000024, 0x000100fd,
    0x00010038,
};

/* conformanceVersion 0.0.0.0 and a driverInfo that says why; the default
 * build's pipeline cache UUID. */
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
        CHECK(drv.conformanceVersion.major == 0 && drv.conformanceVersion.minor == 0 &&
              drv.conformanceVersion.subminor == 0 && drv.conformanceVersion.patch == 0);
        CHECK(strstr(drv.driverInfo, "(PLASMAVK_FAST_MATH, not conformant)") != NULL);
        /* the default build's UUID (section 7) */
        snprintf(name, sizeof(name), "%s %d.%d.%d format %d", props.properties.deviceName, PLASMAVK_VERSION_MAJOR,
                 PLASMAVK_VERSION_MINOR, PLASMAVK_VERSION_PATCH, PVK_PIPELINE_CACHE_FORMAT);
        pvk_make_uuid(uuid, "PlasmaVK pipeline cache", name);
        CHECK(memcmp(uuid, props.properties.pipelineCacheUUID, VK_UUID_SIZE) == 0);
    }
    vkDestroyInstance(instance, NULL);
}

/* The math mode each module compiles with, which Metal accepts and runs:
 * fast (with fast functions) for the plain shader, safe for the precise
 * one; a * b + c is exact either way. */
static void test_math_mode(void)
{
    static const struct { const uint32_t *words; size_t count; long mode; } cases[] = {
        { spv_math_plain, PVK_ARRAY_SIZE(spv_math_plain), PVK_MTLMathModeFast },
        { spv_math_precise, PVK_ARRAY_SIZE(spv_math_precise), PVK_MTLMathModeSafe },
    };
    uint32_t i;
    for (i = 0; i < PVK_ARRAY_SIZE(cases); i++) {
        static const float init[5] = { 2.0f, 3.0f, 4.0f, 0.0f, 0.0f };
        void *pool = objc_autoreleasePoolPush();
        struct pvk_msl_result res;
        struct mt_buffer buf;
        struct mt_binding b;
        id options, library, function = NULL, pso = NULL, error = NULL;
        const float *o;

        REQUIRE(mt_translate(cases[i].words, cases[i].count, NULL, &res));
        options = pvk_msl_compile_options(res.msl, false);
        REQUIRE(options != NULL);
        CHECK(PVK_MSG(long, (id, SEL))(options, sel_registerName("mathMode")) == cases[i].mode);
        if (cases[i].mode == PVK_MTLMathModeFast)
            CHECK(PVK_MSG(long, (id, SEL))(options, sel_registerName("mathFloatingPointFunctions")) ==
                  PVK_MTLMathFloatingPointFunctionsFast);
        library = pvk_MTLDevice_newLibraryWithSource_options_error(mt_device, mt_string(res.msl), options, &error);
        if (!library)
            fprintf(stderr, "case %u: %s\n", i, mt_error_text(error));
        CHECK(library != NULL);
        if (library)
            function = PVK_MSG(id, (id, SEL, id))(library, sel_registerName("newFunctionWithName:"),
                                                  mt_string(res.entry_point));
        if (function)
            pso = PVK_MSG(id, (id, SEL, id, id *))(mt_device, sel_registerName("newComputePipelineStateWithFunction:error:"),
                                                   function, &error);
        CHECK(pso != NULL);
        if (pso) {
            buf = mt_buffer(sizeof(init), init, sizeof(init));
            memset(&b, 0, sizeof(b));
            b.buf = &buf;
            CHECK(mt_dispatch(pso, &res, &b, 1, NULL, 0, 1, 1, 1));
            o = buf.data;
            CHECK(o[3] == 10.0f);
            mt_buffer_free(&buf);
        }
        pvk_release(pso);
        pvk_release(function);
        pvk_release(library);
        pvk_release(options);
        pvk_msl_result_free(&res);
        objc_autoreleasePoolPop(pool);
    }
}

int main(void)
{
    if (!mt_init())
        return 1;
    RUN(test_driver_properties);
    RUN(test_math_mode);
    mt_shutdown();
    return TEST_RESULT();
}
