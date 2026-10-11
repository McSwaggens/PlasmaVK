/* PlasmaVK conformance tests: the framework.
 *
 * Black box: this header and every test include plasmavk.h WITHOUT
 * PLASMAVK_IMPLEMENTATION, so only the public Vulkan API is visible. The
 * implementation comes from the link: PlasmaVK's own translation unit
 * (tests/compile/impl.c) or the system Vulkan loader and any driver.
 *
 * A test is a function declared with CT_TEST, naming the requirement it checks
 * (spec chapter > section, and the VUID or the normative "must") and the
 * Vulkan commands it exercises. Each chapter file lists its tests at the
 * bottom with CT_SUITE. See README.md.
 */
#ifndef CT_H
#define CT_H

#include "../../plasmavk.h"

#include <pthread.h>
#include <setjmp.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ---- Tests and suites ---- */

struct ct;

struct ct_test {
    const char *name;     /* unique within its suite */
    void (*fn)(struct ct *t);
    const char *spec;     /* the requirement: "Chapter > Section: ..." */
    const char *commands; /* space-separated Vulkan commands exercised */
    const void *param;    /* for table-driven tests; see ct_param() */
};

/* Defines a test. The body follows as a block:
 *
 *     CT_TEST(my_test, "Chapter > Section: what must hold (VUID-...)", "vkFoo vkBar")
 *     {
 *         ...
 *     }
 *
 * An unlisted test is an unused static: the build fails until it is added to
 * the file's CT_SUITE. */
#define CT_TEST(name_, spec_, commands_)                                                      \
    static void name_(struct ct *t);                                                          \
    static const struct ct_test name_##_test = { #name_, name_, spec_, commands_, NULL };     \
    static void name_(struct ct *t)

#define CT_REF(name_) (&name_##_test)

struct ct_registry;
void ct_register(struct ct_registry *r, const char *suite, const struct ct_test *const *tests, size_t count);

/* Defines ct_suite_<name>(), which the runner calls (see ct.c). */
#define CT_SUITE(name_, tests_)                                                               \
    void ct_suite_##name_(struct ct_registry *r);                                             \
    void ct_suite_##name_(struct ct_registry *r)                                              \
    {                                                                                         \
        ct_register(r, #name_, tests_, sizeof(tests_) / sizeof((tests_)[0]));                 \
    }

#define CT_ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

/* ---- Results ---- */

struct ct {
    const struct ct_test *test;
    int failures;
    int skipped;
    jmp_buf jump; /* CT_REQUIRE and CT_SKIP leave the test through here */
};

void ct_fail(struct ct *t, const char *file, int line, const char *fmt, ...) __attribute__((format(printf, 4, 5)));
void ct_abort(struct ct *t) __attribute__((noreturn));
void ct_skip(struct ct *t, const char *fmt, ...) __attribute__((format(printf, 2, 3), noreturn));
void ct_note(struct ct *t, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
const void *ct_param(const struct ct *t);
const char *ct_result_name(VkResult r);

/* CHECK records a failure and continues; REQUIRE also ends the test. The
 * plain forms print the condition; the F forms take a printf message. */
#define CT_CHECK(t, ...)                                                                      \
    do {                                                                                      \
        if (!(__VA_ARGS__))                                                                   \
            ct_fail(t, __FILE__, __LINE__, "%s", #__VA_ARGS__);                               \
    } while (0)

#define CT_CHECKF(t, cond, ...)                                                               \
    do {                                                                                      \
        if (!(cond))                                                                          \
            ct_fail(t, __FILE__, __LINE__, __VA_ARGS__);                                      \
    } while (0)

#define CT_REQUIRE(t, ...)                                                                    \
    do {                                                                                      \
        if (!(__VA_ARGS__)) {                                                                 \
            ct_fail(t, __FILE__, __LINE__, "%s", #__VA_ARGS__);                               \
            ct_abort(t);                                                                      \
        }                                                                                     \
    } while (0)

#define CT_REQUIREF(t, cond, ...)                                                             \
    do {                                                                                      \
        if (!(cond)) {                                                                        \
            ct_fail(t, __FILE__, __LINE__, __VA_ARGS__);                                      \
            ct_abort(t);                                                                      \
        }                                                                                     \
    } while (0)

/* A VkResult must equal an expected value. */
#define CT_CHECK_VK(t, expr, expected)                                                        \
    do {                                                                                      \
        VkResult ct_r_ = (expr);                                                              \
        if (ct_r_ != (expected))                                                              \
            ct_fail(t, __FILE__, __LINE__, "%s returned %s, expected %s", #expr,              \
                    ct_result_name(ct_r_), ct_result_name(expected));                         \
    } while (0)

#define CT_REQUIRE_VK(t, expr)                                                                \
    do {                                                                                      \
        VkResult ct_r_ = (expr);                                                              \
        if (ct_r_ != VK_SUCCESS) {                                                            \
            ct_fail(t, __FILE__, __LINE__, "%s returned %s", #expr, ct_result_name(ct_r_));   \
            ct_abort(t);                                                                      \
        }                                                                                     \
    } while (0)

/* ---- Versions ---- */

#define CT_VERSION(v) VK_MAKE_API_VERSION(0, VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v), 0)
#define CT_MIN(a, b) ((a) < (b) ? (a) : (b))

/* ---- Environment: one instance, physical device and device per test ---- */

struct ct_env {
    uint32_t instance_version; /* vkEnumerateInstanceVersion, patch dropped */
    uint32_t app_version;      /* VkApplicationInfo::apiVersion requested */
    VkInstance instance;
    VkPhysicalDevice pd;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem;
    uint32_t version;          /* device-level version usable: min(app, device), patch dropped */
    int portability_subset;    /* the device advertises VK_KHR_portability_subset */
    int graphics;              /* some queue family supports graphics */
    VkDevice device;           /* after ct_env_device() */
    uint32_t queue_family;
    VkQueue queue;
    VkCommandPool pool;        /* made by ct_begin_commands() */
    VkFence fence;
};

/* Instance (apiVersion min(1.4, instance version) unless app_version is
 * non-zero), with portability enumeration when offered, and the physical
 * device selected by $CT_DEVICE_INDEX (default 0). Ends the test on failure. */
void ct_env_init(struct ct *t, struct ct_env *e);
void ct_env_init_ex(struct ct *t, struct ct_env *e, uint32_t app_version, const char *const *extensions,
                    uint32_t extension_count);
/* A device with one queue of a graphics (else compute, else any) family;
 * pnext is an optional feature chain (e.g. VkPhysicalDeviceFeatures2). */
void ct_env_device(struct ct *t, struct ct_env *e, const void *pnext);
void ct_env_device_ex(struct ct *t, struct ct_env *e, const void *pnext, const char *const *extensions,
                      uint32_t extension_count, const VkAllocationCallbacks *alloc);
void ct_env_fini(struct ct_env *e);

/* GPU work on e->queue: a primary command buffer in the recording state,
 * then submitted and waited for with a fence (failing the test after 10 s).
 * ct_host_read_barrier makes transfer and compute writes available and
 * visible to host reads. */
VkCommandBuffer ct_begin_commands(struct ct *t, struct ct_env *e);
void ct_submit_and_wait(struct ct *t, struct ct_env *e, VkCommandBuffer cb);
void ct_host_read_barrier(VkCommandBuffer cb);
void ct_transfer_barrier(VkCommandBuffer cb);

/* ---- Images and host buffers (on e->device) ---- */

/* A host-visible, host-coherent buffer, mapped; ends the test on failure. */
struct ct_buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    unsigned char *data;
    VkDeviceSize size;
};
void ct_host_buffer(struct ct *t, struct ct_env *e, VkDeviceSize size, VkBufferUsageFlags usage, struct ct_buffer *out);
void ct_buffer_fini(struct ct_env *e, struct ct_buffer *b);

/* An image bound at offset 0 to memory of the first type its requirements
 * allow that has the `required` properties (and no protected or lazily
 * allocated ones); ends the test on failure. */
struct ct_image {
    VkImage image;
    VkDeviceMemory memory;
    VkImageCreateInfo info;
};
VkImageCreateInfo ct_image_info(VkImageType type, VkFormat format, uint32_t w, uint32_t h, uint32_t d, uint32_t levels,
                                uint32_t layers, VkImageUsageFlags usage);
void ct_image(struct ct *t, struct ct_env *e, const VkImageCreateInfo *info, VkMemoryPropertyFlags required,
              struct ct_image *out);
void ct_image_fini(struct ct_env *e, struct ct_image *m);
/* All commands before against all after, with an image layout transition of
 * every subresource of `aspect` (image may be VK_NULL_HANDLE: memory only). */
void ct_image_barrier(VkCommandBuffer cb, VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to);
/* optimalTilingFeatures (or linear, buffer) of VkFormatProperties3, or of
 * VkFormatProperties on Vulkan 1.0-1.2. */
VkFormatFeatureFlags2 ct_format_features(const struct ct_env *e, VkFormat format, int which); /* 0 optimal 1 linear 2 buffer */
/* A pattern of bytes for texel data. */
void ct_fill_pattern(unsigned char *p, size_t n, unsigned seed);

/* vkCreateInstance with portability enumeration added when offered. */
VkResult ct_create_instance(uint32_t app_version, const char *const *extensions, uint32_t extension_count,
                            const VkAllocationCallbacks *alloc, VkInstance *instance);
/* Appends VK_KHR_portability_subset to names[] (room for one more) when the
 * device advertises it (VUID-VkDeviceCreateInfo-pProperties-04451). */
uint32_t ct_add_portability_subset(const struct ct_env *e, const char **names, uint32_t count);

uint32_t ct_instance_version(void);
int ct_instance_extension_supported(const char *name);
int ct_device_extension_supported(VkPhysicalDevice pd, const char *name);
uint32_t ct_find_memory_type(const struct ct_env *e, uint32_t type_bits, VkMemoryPropertyFlags required,
                             VkMemoryPropertyFlags excluded);
int ct_is_pow2(uint64_t x);
int ct_all_bytes(const void *p, size_t n, unsigned char value);

/* The two-call enumeration protocol of a command returning VkResult
 * (e.g. vkEnumerateInstanceExtensionProperties): a count query, then with
 * room for fewer than all items exactly that many are written, the count is
 * set to that number and VK_INCOMPLETE returned; with room to spare,
 * VK_SUCCESS and the total. Items written must be items of the full list
 * (compared with `same`); nothing past the count may be written. */
typedef VkResult (*ct_enumerate_fn)(void *ctx, uint32_t *count, void *items);
typedef int (*ct_same_fn)(const void *a, const void *b);
void ct_check_incomplete(struct ct *t, ct_enumerate_fn fn, void *ctx, size_t item_size, ct_same_fn same,
                         const char *what);

/* Feature and property chains for the versions e->version supports. Output
 * structures are pre-filled with `fill` (the pNext/sType are set afterwards)
 * so tests can detect members the implementation did not write. */
struct ct_features {
    VkPhysicalDeviceFeatures2 f2;
    VkPhysicalDeviceVulkan11Features v11;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    VkPhysicalDeviceVulkan14Features v14;
};
void ct_get_features(const struct ct_env *e, struct ct_features *f, unsigned char fill);

struct ct_properties {
    VkPhysicalDeviceProperties2 p2;
    VkPhysicalDeviceVulkan11Properties v11;
    VkPhysicalDeviceVulkan12Properties v12;
    VkPhysicalDeviceVulkan13Properties v13;
    VkPhysicalDeviceVulkan14Properties v14;
};
void ct_get_properties(const struct ct_env *e, struct ct_properties *p, unsigned char fill);

/* ---- Allocation callbacks that check the implementation ----
 *
 * Counts and validates every call (Memory Allocation > Host Memory): the
 * alignment is a power of two, the scope is valid, the call comes from the
 * test's thread and from inside a Vulkan command (wrap calls in CT_CALL),
 * frees and reallocations name live blocks. With fail_at = n, every
 * allocation from the n-th (0-based) on returns NULL, as when memory runs
 * out. */
struct ct_block;
struct ct_allocator {
    VkAllocationCallbacks cb;
    long live, total, failed;
    long fail_at; /* -1: never */
    int bad_alignment, bad_scope, wrong_thread, outside_command, unknown_free;
    int in_command;
    pthread_t thread;
    struct ct_block *blocks;
};
void ct_allocator_init(struct ct_allocator *a, long fail_at);
/* Which allocations to fail, for an operation that makes `total`: each of
 * the first 128, then about 100 spread over the rest, and the last 8. */
size_t ct_failure_points(long total, long *points, size_t max);
void ct_allocator_check(struct ct *t, const struct ct_allocator *a, const char *what);
void ct_allocator_fini(struct ct_allocator *a);

#define CT_CALL(alloc, ...)                                                                   \
    ((alloc)->in_command = 1, (__VA_ARGS__), (alloc)->in_command = 0)
#define CT_CALL_VK(alloc, result, ...)                                                        \
    do {                                                                                      \
        (alloc)->in_command = 1;                                                              \
        (result) = (__VA_ARGS__);                                                             \
        (alloc)->in_command = 0;                                                              \
    } while (0)

/* ---- Vulkan commands (commands.c), for coverage ---- */

enum ct_level { CT_LEVEL_GLOBAL, CT_LEVEL_INSTANCE, CT_LEVEL_PHYSICAL, CT_LEVEL_DEVICE };

struct ct_command {
    const char *name;
    const char *origin; /* "1.0" .. "1.4", or the extension's name */
    enum ct_level level;
};

extern const struct ct_command ct_commands[];
extern const size_t ct_command_count;

/* VK_API_VERSION_1_x of a core origin, or 0 for an extension. */
uint32_t ct_command_version(const struct ct_command *c);

#endif /* CT_H */
