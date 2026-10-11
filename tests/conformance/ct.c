/* PlasmaVK conformance tests: the runner and shared helpers.
 *
 *   ct [options] [pattern...]
 *
 * Runs every test whose full name ("suite.test") matches a pattern: a glob
 * if it has * or ?, a substring otherwise. Each test runs in its own process
 * (a crash or hang is that test's FAIL) and prints one PASS/FAIL/SKIP line.
 *
 *   -l, --list         list the tests and their requirements, run nothing
 *   -c, --coverage     report which Vulkan commands the selected tests exercise
 *   -b, --baseline F   F lists tests expected to pass: fail only on regressions
 *                      and report newly passing tests
 *   -j N               run N tests at a time (default: $CT_JOBS, else the CPU count)
 *   -t, --timeout S    per-test timeout in seconds (default 120)
 *   -v, --verbose      also show the output of passing tests
 *   --in-process       run in this process (debugging; a crash ends the run)
 *
 * Environment: CT_DEVICE_INDEX picks the physical device (default 0).
 */
#define _DARWIN_C_SOURCE
#include "ct.h"

#include <errno.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <mach-o/dyld.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

/* ==========================================================================
 * Registry
 * ========================================================================== */

struct ct_entry {
    const char *suite;
    const struct ct_test *test;
    char name[128];
};

struct ct_registry {
    struct ct_entry *entries;
    size_t count, cap;
};

void ct_register(struct ct_registry *r, const char *suite, const struct ct_test *const *tests, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++) {
        struct ct_entry *e;
        if (r->count == r->cap) {
            r->cap = r->cap ? r->cap * 2 : 256;
            r->entries = realloc(r->entries, r->cap * sizeof(*r->entries));
            if (!r->entries) {
                fprintf(stderr, "ct: out of memory\n");
                exit(2);
            }
        }
        e = &r->entries[r->count++];
        e->suite = suite;
        e->test = tests[i];
        snprintf(e->name, sizeof(e->name), "%s.%s", suite, tests[i]->name);
    }
}

/* One line per chapter file. */
void ct_suite_initialization(struct ct_registry *r);
void ct_suite_devsandqueues(struct ct_registry *r);
void ct_suite_memory(struct ct_registry *r);
void ct_suite_resources(struct ct_registry *r);
void ct_suite_extensions(struct ct_registry *r);
void ct_suite_features(struct ct_registry *r);
void ct_suite_limits(struct ct_registry *r);
void ct_suite_formats(struct ct_registry *r);
void ct_suite_capabilities(struct ct_registry *r);
void ct_suite_samplers(struct ct_registry *r);
void ct_suite_copies(struct ct_registry *r);
void ct_suite_clears(struct ct_registry *r);
void ct_suite_descriptorsets(struct ct_registry *r);
void ct_suite_queries(struct ct_registry *r);
void ct_suite_synchronization(struct ct_registry *r);
void ct_suite_privatedata(struct ct_registry *r);
void ct_suite_wsi(struct ct_registry *r);

static void ct_register_all(struct ct_registry *r)
{
    ct_suite_initialization(r);
    ct_suite_devsandqueues(r);
    ct_suite_memory(r);
    ct_suite_resources(r);
    ct_suite_extensions(r);
    ct_suite_features(r);
    ct_suite_limits(r);
    ct_suite_formats(r);
    ct_suite_capabilities(r);
    ct_suite_samplers(r);
    ct_suite_copies(r);
    ct_suite_clears(r);
    ct_suite_descriptorsets(r);
    ct_suite_queries(r);
    ct_suite_synchronization(r);
    ct_suite_privatedata(r);
    ct_suite_wsi(r);
}

/* ==========================================================================
 * Results
 * ========================================================================== */

#define CT_MAX_MESSAGES 24 /* failure lines shown per test */

void ct_fail(struct ct *t, const char *file, int line, const char *fmt, ...)
{
    va_list args;
    const char *base = strrchr(file, '/');
    if (++t->failures > CT_MAX_MESSAGES)
        return;
    printf("%s:%d: ", base ? base + 1 : file, line);
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    putchar('\n');
    fflush(stdout);
}

void ct_abort(struct ct *t)
{
    longjmp(t->jump, 1);
}

void ct_skip(struct ct *t, const char *fmt, ...)
{
    va_list args;
    t->skipped = 1;
    printf("skipped: ");
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    putchar('\n');
    fflush(stdout);
    longjmp(t->jump, 1);
}

void ct_note(struct ct *t, const char *fmt, ...)
{
    va_list args;
    (void)t;
    printf("note: ");
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    putchar('\n');
    fflush(stdout);
}

const void *ct_param(const struct ct *t)
{
    return t->test->param;
}

const char *ct_result_name(VkResult r)
{
    switch (r) {
#define CT_RESULT(x) case x: return #x;
    CT_RESULT(VK_SUCCESS)
    CT_RESULT(VK_NOT_READY)
    CT_RESULT(VK_TIMEOUT)
    CT_RESULT(VK_EVENT_SET)
    CT_RESULT(VK_EVENT_RESET)
    CT_RESULT(VK_INCOMPLETE)
    CT_RESULT(VK_ERROR_OUT_OF_HOST_MEMORY)
    CT_RESULT(VK_ERROR_OUT_OF_DEVICE_MEMORY)
    CT_RESULT(VK_ERROR_INITIALIZATION_FAILED)
    CT_RESULT(VK_ERROR_DEVICE_LOST)
    CT_RESULT(VK_ERROR_MEMORY_MAP_FAILED)
    CT_RESULT(VK_ERROR_LAYER_NOT_PRESENT)
    CT_RESULT(VK_ERROR_EXTENSION_NOT_PRESENT)
    CT_RESULT(VK_ERROR_FEATURE_NOT_PRESENT)
    CT_RESULT(VK_ERROR_INCOMPATIBLE_DRIVER)
    CT_RESULT(VK_ERROR_TOO_MANY_OBJECTS)
    CT_RESULT(VK_ERROR_FORMAT_NOT_SUPPORTED)
    CT_RESULT(VK_ERROR_FRAGMENTED_POOL)
    CT_RESULT(VK_ERROR_UNKNOWN)
    CT_RESULT(VK_ERROR_VALIDATION_FAILED)
    CT_RESULT(VK_ERROR_OUT_OF_POOL_MEMORY)
    CT_RESULT(VK_ERROR_INVALID_EXTERNAL_HANDLE)
    CT_RESULT(VK_ERROR_FRAGMENTATION)
    CT_RESULT(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS)
    CT_RESULT(VK_PIPELINE_COMPILE_REQUIRED)
    CT_RESULT(VK_ERROR_NOT_PERMITTED)
    CT_RESULT(VK_SUBOPTIMAL_KHR)
    CT_RESULT(VK_ERROR_OUT_OF_DATE_KHR)
    CT_RESULT(VK_ERROR_SURFACE_LOST_KHR)
    CT_RESULT(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR)
#undef CT_RESULT
    default:
        break;
    }
    {
        static char buf[32];
        snprintf(buf, sizeof(buf), "VkResult(%d)", (int)r);
        return buf;
    }
}

/* ==========================================================================
 * Environment helpers
 * ========================================================================== */

int ct_is_pow2(uint64_t x)
{
    return x && !(x & (x - 1));
}

int ct_all_bytes(const void *p, size_t n, unsigned char value)
{
    const unsigned char *b = p;
    size_t i;
    for (i = 0; i < n; i++)
        if (b[i] != value)
            return 0;
    return 1;
}

#define CT_GUARD 0xA5

void ct_check_incomplete(struct ct *t, ct_enumerate_fn fn, void *ctx, size_t item_size, ct_same_fn same,
                         const char *what)
{
    uint32_t total = 0, n, k;
    unsigned char *full, *part;

    CT_REQUIRE_VK(t, fn(ctx, &total, NULL));
    full = calloc(total + 1, item_size);
    part = malloc((total + 1) * item_size);
    CT_REQUIRE(t, full && part);
    n = total;
    CT_REQUIRE_VK(t, fn(ctx, &n, full));
    CT_CHECKF(t, n == total, "%s: count %u, then %u written", what, total, n);

    /* Room to spare: VK_SUCCESS and the total. */
    n = total + 1;
    memset(part, CT_GUARD, (total + 1) * item_size);
    CT_CHECK_VK(t, fn(ctx, &n, part), VK_SUCCESS);
    CT_CHECKF(t, n == total, "%s: room for %u, count returned %u, expected %u", what, total + 1, n, total);
    CT_CHECKF(t, ct_all_bytes(part + total * item_size, item_size, CT_GUARD), "%s: wrote past the last item", what);

    for (k = 0; k < total; k++) {
        uint32_t i, j;
        n = k;
        memset(part, CT_GUARD, (total + 1) * item_size);
        CT_CHECK_VK(t, fn(ctx, &n, part), VK_INCOMPLETE);
        CT_CHECKF(t, n == k, "%s: room for %u of %u, count returned %u", what, k, total, n);
        CT_CHECKF(t, ct_all_bytes(part + k * item_size, item_size, CT_GUARD), "%s: room for %u, wrote item %u",
                  what, k, k);
        for (i = 0; i < n && i < k; i++) {
            int found = 0;
            for (j = 0; j < total; j++)
                found |= same(part + i * item_size, full + j * item_size);
            CT_CHECKF(t, found, "%s: with room for %u, item %u is not one of the full list", what, k, i);
        }
    }
    free(full);
    free(part);
}

uint32_t ct_instance_version(void)
{
    uint32_t version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion(&version) != VK_SUCCESS)
        return VK_API_VERSION_1_0;
    return version;
}

int ct_instance_extension_supported(const char *name)
{
    VkExtensionProperties *props;
    uint32_t count = 0, i;
    int found = 0;
    if (vkEnumerateInstanceExtensionProperties(NULL, &count, NULL) != VK_SUCCESS || !count)
        return 0;
    props = calloc(count, sizeof(*props));
    if (props && vkEnumerateInstanceExtensionProperties(NULL, &count, props) >= 0)
        for (i = 0; i < count; i++)
            found |= strcmp(props[i].extensionName, name) == 0;
    free(props);
    return found;
}

int ct_device_extension_supported(VkPhysicalDevice pd, const char *name)
{
    VkExtensionProperties *props;
    uint32_t count = 0, i;
    int found = 0;
    if (vkEnumerateDeviceExtensionProperties(pd, NULL, &count, NULL) != VK_SUCCESS || !count)
        return 0;
    props = calloc(count, sizeof(*props));
    if (props && vkEnumerateDeviceExtensionProperties(pd, NULL, &count, props) >= 0)
        for (i = 0; i < count; i++)
            found |= strcmp(props[i].extensionName, name) == 0;
    free(props);
    return found;
}

VkResult ct_create_instance(uint32_t app_version, const char *const *extensions, uint32_t extension_count,
                            const VkAllocationCallbacks *alloc, VkInstance *instance)
{
    const char *names[64];
    VkApplicationInfo app;
    VkInstanceCreateInfo info;
    uint32_t i, count = 0;
    int portability = 0;

    for (i = 0; i < extension_count && count < 63; i++) {
        names[count++] = extensions[i];
        portability |= strcmp(extensions[i], VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
    }
    /* Lets the loader expose portability drivers such as MoltenVK. */
    if (!portability && ct_instance_extension_supported(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        names[count++] = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
        portability = 1;
    }
    /* Below Vulkan 1.1, VK_KHR_portability_subset (which such devices make
     * mandatory) needs VK_KHR_get_physical_device_properties2 on the instance
     * (VUID-vkCreateDevice-ppEnabledExtensionNames-01387). */
    if (portability && CT_VERSION(app_version) < VK_API_VERSION_1_1 && count < 64 &&
        ct_instance_extension_supported(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
        for (i = 0; i < count; i++)
            if (strcmp(names[i], VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0)
                break;
        if (i == count)
            names[count++] = VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;
    }

    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "plasmavk-conformance";
    app.applicationVersion = 1;
    app.apiVersion = app_version;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    if (portability)
        info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    info.pApplicationInfo = &app;
    info.enabledExtensionCount = count;
    info.ppEnabledExtensionNames = count ? names : NULL;
    return vkCreateInstance(&info, alloc, instance);
}

uint32_t ct_add_portability_subset(const struct ct_env *e, const char **names, uint32_t count)
{
    uint32_t i;
    if (!e->portability_subset)
        return count;
    for (i = 0; i < count; i++)
        if (strcmp(names[i], "VK_KHR_portability_subset") == 0)
            return count;
    names[count] = "VK_KHR_portability_subset";
    return count + 1;
}

static uint32_t ct_device_index(void)
{
    const char *s = getenv("CT_DEVICE_INDEX");
    return s ? (uint32_t)strtoul(s, NULL, 10) : 0;
}

void ct_env_init_ex(struct ct *t, struct ct_env *e, uint32_t app_version, const char *const *extensions,
                    uint32_t extension_count)
{
    VkPhysicalDevice *pds;
    VkQueueFamilyProperties families[64];
    uint32_t count = 0, index = ct_device_index(), i;

    memset(e, 0, sizeof(*e));
    e->instance_version = CT_VERSION(ct_instance_version());
    e->app_version = app_version ? app_version : CT_MIN(VK_API_VERSION_1_4, e->instance_version);
    CT_REQUIRE_VK(t, ct_create_instance(e->app_version, extensions, extension_count, NULL, &e->instance));
    CT_REQUIRE_VK(t, vkEnumeratePhysicalDevices(e->instance, &count, NULL));
    CT_REQUIREF(t, count > index, "physical device %u requested, %u present", index, count);
    pds = calloc(count, sizeof(*pds));
    CT_REQUIRE(t, pds != NULL);
    CT_REQUIRE_VK(t, vkEnumeratePhysicalDevices(e->instance, &count, pds));
    e->pd = pds[index];
    free(pds);
    vkGetPhysicalDeviceProperties(e->pd, &e->props);
    vkGetPhysicalDeviceMemoryProperties(e->pd, &e->mem);
    e->version = CT_MIN(CT_VERSION(e->app_version), CT_VERSION(e->props.apiVersion));
    e->portability_subset = ct_device_extension_supported(e->pd, "VK_KHR_portability_subset");
    count = CT_ARRAY_SIZE(families);
    vkGetPhysicalDeviceQueueFamilyProperties(e->pd, &count, families);
    for (i = 0; i < count; i++)
        e->graphics |= (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
}

void ct_env_init(struct ct *t, struct ct_env *e)
{
    ct_env_init_ex(t, e, 0, NULL, 0);
}

void ct_env_device_ex(struct ct *t, struct ct_env *e, const void *pnext, const char *const *extensions,
                      uint32_t extension_count, const VkAllocationCallbacks *alloc)
{
    VkQueueFamilyProperties families[64];
    const char *names[64];
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue;
    VkDeviceCreateInfo info;
    uint32_t count = CT_ARRAY_SIZE(families), i, family = UINT32_MAX, n = 0;

    vkGetPhysicalDeviceQueueFamilyProperties(e->pd, &count, families);
    CT_REQUIRE(t, count > 0);
    for (i = 0; i < count && family == UINT32_MAX; i++)
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            family = i;
    for (i = 0; i < count && family == UINT32_MAX; i++)
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            family = i;
    if (family == UINT32_MAX)
        family = 0;

    for (i = 0; i < extension_count && n < 62; i++)
        names[n++] = extensions[i];
    n = ct_add_portability_subset(e, names, n);

    memset(&queue, 0, sizeof(queue));
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueFamilyIndex = family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.pNext = pnext;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue;
    info.enabledExtensionCount = n;
    info.ppEnabledExtensionNames = n ? names : NULL;
    CT_REQUIRE_VK(t, vkCreateDevice(e->pd, &info, alloc, &e->device));
    e->queue_family = family;
    vkGetDeviceQueue(e->device, family, 0, &e->queue);
    CT_REQUIRE(t, e->queue != VK_NULL_HANDLE);
}

void ct_env_device(struct ct *t, struct ct_env *e, const void *pnext)
{
    ct_env_device_ex(t, e, pnext, NULL, 0, NULL);
}

VkCommandBuffer ct_begin_commands(struct ct *t, struct ct_env *e)
{
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferBeginInfo begin;
    VkCommandBuffer cb = VK_NULL_HANDLE;

    if (!e->pool) {
        VkCommandPoolCreateInfo info;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        info.queueFamilyIndex = e->queue_family;
        CT_REQUIRE_VK(t, vkCreateCommandPool(e->device, &info, NULL, &e->pool));
    }
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = e->pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    CT_REQUIRE_VK(t, vkAllocateCommandBuffers(e->device, &alloc, &cb));
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CT_REQUIRE_VK(t, vkBeginCommandBuffer(cb, &begin));
    return cb;
}

void ct_submit_and_wait(struct ct *t, struct ct_env *e, VkCommandBuffer cb)
{
    VkSubmitInfo submit;
    VkResult r;

    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    if (!e->fence) {
        VkFenceCreateInfo info;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        CT_REQUIRE_VK(t, vkCreateFence(e->device, &info, NULL, &e->fence));
    } else {
        CT_REQUIRE_VK(t, vkResetFences(e->device, 1, &e->fence));
    }
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    CT_REQUIRE_VK(t, vkQueueSubmit(e->queue, 1, &submit, e->fence));
    r = vkWaitForFences(e->device, 1, &e->fence, VK_TRUE, 10000000000ull);
    CT_REQUIREF(t, r == VK_SUCCESS, "vkWaitForFences: %s (GPU work not done after 10 s)", ct_result_name(r));
    vkFreeCommandBuffers(e->device, e->pool, 1, &cb);
}

static void ct_barrier(VkCommandBuffer cb, VkPipelineStageFlags dst_stage, VkAccessFlags dst_access)
{
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, dst_stage, 0, 1,
                         &barrier, 0, NULL, 0, NULL);
}

void ct_host_read_barrier(VkCommandBuffer cb)
{
    ct_barrier(cb, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
}

void ct_transfer_barrier(VkCommandBuffer cb)
{
    ct_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
}

void ct_env_fini(struct ct_env *e)
{
    if (e->fence)
        vkDestroyFence(e->device, e->fence, NULL);
    if (e->pool)
        vkDestroyCommandPool(e->device, e->pool, NULL);
    if (e->device)
        vkDestroyDevice(e->device, NULL);
    if (e->instance)
        vkDestroyInstance(e->instance, NULL);
    memset(e, 0, sizeof(*e));
}

void ct_host_buffer(struct ct *t, struct ct_env *e, VkDeviceSize size, VkBufferUsageFlags usage, struct ct_buffer *out)
{
    VkBufferCreateInfo info;
    VkMemoryAllocateInfo alloc;
    VkMemoryRequirements req;
    uint32_t type;
    void *p = NULL;

    memset(out, 0, sizeof(*out));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    CT_REQUIRE_VK(t, vkCreateBuffer(e->device, &info, NULL, &out->buffer));
    vkGetBufferMemoryRequirements(e->device, out->buffer, &req);
    type = ct_find_memory_type(e, req.memoryTypeBits,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               VK_MEMORY_PROPERTY_PROTECTED_BIT);
    CT_REQUIREF(t, type != UINT32_MAX, "no host-coherent memory type for a buffer");
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    CT_REQUIRE_VK(t, vkAllocateMemory(e->device, &alloc, NULL, &out->memory));
    CT_REQUIRE_VK(t, vkBindBufferMemory(e->device, out->buffer, out->memory, 0));
    CT_REQUIRE_VK(t, vkMapMemory(e->device, out->memory, 0, VK_WHOLE_SIZE, 0, &p));
    out->data = p;
    out->size = size;
    memset(out->data, 0, (size_t)size);
}

void ct_buffer_fini(struct ct_env *e, struct ct_buffer *b)
{
    if (b->buffer)
        vkDestroyBuffer(e->device, b->buffer, NULL);
    if (b->memory)
        vkFreeMemory(e->device, b->memory, NULL);
    memset(b, 0, sizeof(*b));
}

VkImageCreateInfo ct_image_info(VkImageType type, VkFormat format, uint32_t w, uint32_t h, uint32_t d, uint32_t levels,
                                uint32_t layers, VkImageUsageFlags usage)
{
    VkImageCreateInfo info;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = type;
    info.format = format;
    info.extent.width = w;
    info.extent.height = h;
    info.extent.depth = d;
    info.mipLevels = levels;
    info.arrayLayers = layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    return info;
}

void ct_image(struct ct *t, struct ct_env *e, const VkImageCreateInfo *info, VkMemoryPropertyFlags required,
              struct ct_image *out)
{
    VkMemoryAllocateInfo alloc;
    VkMemoryRequirements req;
    uint32_t type;

    memset(out, 0, sizeof(*out));
    out->info = *info;
    CT_REQUIRE_VK(t, vkCreateImage(e->device, info, NULL, &out->image));
    vkGetImageMemoryRequirements(e->device, out->image, &req);
    type = ct_find_memory_type(e, req.memoryTypeBits, required,
                               VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
    CT_REQUIREF(t, type != UINT32_MAX, "no memory type with properties 0x%x for the image (bits 0x%x)", required,
                req.memoryTypeBits);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    CT_REQUIRE_VK(t, vkAllocateMemory(e->device, &alloc, NULL, &out->memory));
    CT_REQUIRE_VK(t, vkBindImageMemory(e->device, out->image, out->memory, 0));
}

void ct_image_fini(struct ct_env *e, struct ct_image *m)
{
    if (m->image)
        vkDestroyImage(e->device, m->image, NULL);
    if (m->memory)
        vkFreeMemory(e->device, m->memory, NULL);
    memset(m, 0, sizeof(*m));
}

void ct_image_barrier(VkCommandBuffer cb, VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to)
{
    VkImageMemoryBarrier ib;
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    memset(&ib, 0, sizeof(ib));
    ib.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    ib.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    ib.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    ib.oldLayout = from;
    ib.newLayout = to;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = image;
    ib.subresourceRange.aspectMask = aspect;
    ib.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    ib.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb, 0, NULL, image ? 1 : 0, &ib);
}

VkFormatFeatureFlags2 ct_format_features(const struct ct_env *e, VkFormat format, int which)
{
    VkFormatProperties3 p3;
    VkFormatProperties2 p2;
    if (e->version < VK_API_VERSION_1_3) {
        VkFormatProperties p;
        vkGetPhysicalDeviceFormatProperties(e->pd, format, &p);
        return which == 0 ? p.optimalTilingFeatures : which == 1 ? p.linearTilingFeatures : p.bufferFeatures;
    }
    memset(&p3, 0, sizeof(p3));
    p3.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3;
    memset(&p2, 0, sizeof(p2));
    p2.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
    p2.pNext = &p3;
    vkGetPhysicalDeviceFormatProperties2(e->pd, format, &p2);
    return which == 0 ? p3.optimalTilingFeatures : which == 1 ? p3.linearTilingFeatures : p3.bufferFeatures;
}

void ct_fill_pattern(unsigned char *p, size_t n, unsigned seed)
{
    size_t i;
    for (i = 0; i < n; i++)
        p[i] = (unsigned char)((i * 131u + seed * 7u + (i >> 8) * 17u) & 0xff);
}

uint32_t ct_find_memory_type(const struct ct_env *e, uint32_t type_bits, VkMemoryPropertyFlags required,
                             VkMemoryPropertyFlags excluded)
{
    uint32_t i;
    for (i = 0; i < e->mem.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = e->mem.memoryTypes[i].propertyFlags;
        if ((type_bits & (1u << i)) && (f & required) == required && !(f & excluded))
            return i;
    }
    return UINT32_MAX;
}

void ct_get_features(const struct ct_env *e, struct ct_features *f, unsigned char fill)
{
    memset(f, fill, sizeof(*f));
    f->f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f->f2.pNext = NULL;
    f->v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f->v11.pNext = NULL;
    f->v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f->v12.pNext = NULL;
    f->v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f->v13.pNext = NULL;
    f->v14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    f->v14.pNext = NULL;
    if (e->version < VK_API_VERSION_1_1) {
        vkGetPhysicalDeviceFeatures(e->pd, &f->f2.features);
        return;
    }
    /* VkPhysicalDeviceVulkan11Features arrived with Vulkan 1.2. */
    if (e->version >= VK_API_VERSION_1_2) {
        f->f2.pNext = &f->v11;
        f->v11.pNext = &f->v12;
    }
    if (e->version >= VK_API_VERSION_1_3)
        f->v12.pNext = &f->v13;
    if (e->version >= VK_API_VERSION_1_4)
        f->v13.pNext = &f->v14;
    vkGetPhysicalDeviceFeatures2(e->pd, &f->f2);
}

void ct_get_properties(const struct ct_env *e, struct ct_properties *p, unsigned char fill)
{
    memset(p, fill, sizeof(*p));
    p->p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p->p2.pNext = NULL;
    p->v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
    p->v11.pNext = NULL;
    p->v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
    p->v12.pNext = NULL;
    p->v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
    p->v13.pNext = NULL;
    p->v14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES;
    p->v14.pNext = NULL;
    /* Inputs of the two-call idiom for the host image copy layouts. */
    p->v14.copySrcLayoutCount = 0;
    p->v14.pCopySrcLayouts = NULL;
    p->v14.copyDstLayoutCount = 0;
    p->v14.pCopyDstLayouts = NULL;
    if (e->version < VK_API_VERSION_1_1) {
        vkGetPhysicalDeviceProperties(e->pd, &p->p2.properties);
        return;
    }
    /* VkPhysicalDeviceVulkan11Properties arrived with Vulkan 1.2. */
    if (e->version >= VK_API_VERSION_1_2) {
        p->p2.pNext = &p->v11;
        p->v11.pNext = &p->v12;
    }
    if (e->version >= VK_API_VERSION_1_3)
        p->v12.pNext = &p->v13;
    if (e->version >= VK_API_VERSION_1_4)
        p->v13.pNext = &p->v14;
    vkGetPhysicalDeviceProperties2(e->pd, &p->p2);
}

uint32_t ct_command_version(const struct ct_command *c)
{
    if (c->origin[0] < '0' || c->origin[0] > '9')
        return 0;
    return VK_MAKE_API_VERSION(0, (uint32_t)(c->origin[0] - '0'), (uint32_t)(c->origin[2] - '0'), 0);
}

/* ==========================================================================
 * Checking allocator
 * ========================================================================== */

struct ct_block {
    void *p;
    size_t size;
    struct ct_block *next;
};

static void ct_allocator_note(struct ct_allocator *a, size_t alignment, VkSystemAllocationScope scope)
{
    if (!pthread_equal(pthread_self(), a->thread))
        a->wrong_thread++;
    if (!a->in_command)
        a->outside_command++;
    if (!ct_is_pow2(alignment))
        a->bad_alignment++;
    if ((unsigned)scope > (unsigned)VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE)
        a->bad_scope++;
}

static void *ct_allocator_take(struct ct_allocator *a, size_t size, size_t alignment)
{
    struct ct_block *b;
    void *p = NULL;
    long index = a->total++;

    if (a->fail_at >= 0 && index >= a->fail_at) {
        a->failed++;
        return NULL;
    }
    if (!ct_is_pow2(alignment))
        alignment = 16;
    if (alignment < sizeof(void *))
        alignment = sizeof(void *);
    b = malloc(sizeof(*b));
    if (!b || posix_memalign(&p, alignment, size ? size : 1) != 0) {
        free(b);
        return NULL;
    }
    b->p = p;
    b->size = size;
    b->next = a->blocks;
    a->blocks = b;
    a->live++;
    return p;
}

static struct ct_block **ct_allocator_find(struct ct_allocator *a, void *p)
{
    struct ct_block **b;
    for (b = &a->blocks; *b; b = &(*b)->next)
        if ((*b)->p == p)
            return b;
    return NULL;
}

static void ct_allocator_release(struct ct_allocator *a, void *p)
{
    struct ct_block **b = ct_allocator_find(a, p);
    struct ct_block *dead;
    if (!b) {
        a->unknown_free++;
        return;
    }
    dead = *b;
    *b = dead->next;
    free(dead->p);
    free(dead);
    a->live--;
}

static void *VKAPI_PTR ct_alloc_fn(void *user, size_t size, size_t alignment, VkSystemAllocationScope scope)
{
    struct ct_allocator *a = user;
    ct_allocator_note(a, alignment, scope);
    return ct_allocator_take(a, size, alignment);
}

static void *VKAPI_PTR ct_realloc_fn(void *user, void *original, size_t size, size_t alignment,
                                     VkSystemAllocationScope scope)
{
    struct ct_allocator *a = user;
    struct ct_block **b;
    void *p;

    ct_allocator_note(a, alignment, scope);
    if (!original)
        return ct_allocator_take(a, size, alignment);
    if (size == 0) {
        ct_allocator_release(a, original);
        return NULL;
    }
    b = ct_allocator_find(a, original);
    if (!b) {
        a->unknown_free++;
        return NULL;
    }
    p = ct_allocator_take(a, size, alignment);
    if (!p)
        return NULL; /* the original stays valid */
    b = ct_allocator_find(a, original);
    memcpy(p, original, (*b)->size < size ? (*b)->size : size);
    ct_allocator_release(a, original);
    return p;
}

static void VKAPI_PTR ct_free_fn(void *user, void *memory)
{
    struct ct_allocator *a = user;
    if (!pthread_equal(pthread_self(), a->thread))
        a->wrong_thread++;
    if (!a->in_command)
        a->outside_command++;
    if (memory)
        ct_allocator_release(a, memory);
}

void ct_allocator_init(struct ct_allocator *a, long fail_at)
{
    memset(a, 0, sizeof(*a));
    a->cb.pUserData = a;
    a->cb.pfnAllocation = ct_alloc_fn;
    a->cb.pfnReallocation = ct_realloc_fn;
    a->cb.pfnFree = ct_free_fn;
    a->fail_at = fail_at;
    a->thread = pthread_self();
}

size_t ct_failure_points(long total, long *points, size_t max)
{
    size_t n = 0;
    long i, step = total > 128 ? (total - 128) / 100 + 1 : 1;
    for (i = 0; i < total && n < max; i++)
        if (i < 128 || i >= total - 8 || (i - 128) % step == 0)
            points[n++] = i;
    return n;
}

void ct_allocator_check(struct ct *t, const struct ct_allocator *a, const char *what)
{
    CT_CHECKF(t, !a->bad_alignment, "%s: %d allocations with an alignment that is not a power of two "
              "(PFN_vkAllocationFunction: alignment must be a power of two)", what, a->bad_alignment);
    CT_CHECKF(t, !a->bad_scope, "%s: %d allocations with an invalid VkSystemAllocationScope", what, a->bad_scope);
    CT_CHECKF(t, !a->wrong_thread, "%s: %d callbacks from another thread (Memory Allocation > Host Memory: "
              "only from the thread that called the provoking command)", what, a->wrong_thread);
    CT_CHECKF(t, !a->outside_command, "%s: %d callbacks outside any Vulkan command (Memory Allocation > Host "
              "Memory: only during the execution of an API command)", what, a->outside_command);
    CT_CHECKF(t, !a->unknown_free, "%s: %d frees/reallocations of memory the allocator did not return "
              "(PFN_vkFreeFunction: pMemory must have been allocated by pfnAllocation or pfnReallocation)",
              what, a->unknown_free);
}

void ct_allocator_fini(struct ct_allocator *a)
{
    while (a->blocks) {
        struct ct_block *b = a->blocks;
        a->blocks = b->next;
        free(b); /* not b->p: a leaked block may still be referenced */
    }
}

/* ==========================================================================
 * Running tests
 * ========================================================================== */

enum ct_status { CT_PASS, CT_FAIL, CT_SKIP };

struct ct_options {
    int list, coverage, verbose, in_process;
    unsigned jobs, timeout;
    const char *baseline;
    const struct ct_baseline *expected; /* the loaded baseline, if any */
    const char **patterns;
    int pattern_count;
};

struct ct_result {
    enum ct_status status;
    char *output;
    size_t len, cap;
    char reason[96]; /* for failures the test did not report: crash, timeout */
    double seconds;
    int done;
};

/* Runs a test in this process and returns its status. */
static enum ct_status ct_run_here(const struct ct_test *test)
{
    struct ct t;
    memset(&t, 0, sizeof(t));
    t.test = test;
    if (setjmp(t.jump) == 0)
        test->fn(&t);
    if (t.failures > CT_MAX_MESSAGES)
        printf("(%d more failures not shown)\n", t.failures - CT_MAX_MESSAGES);
    fflush(stdout);
    fflush(stderr);
    if (t.skipped && !t.failures)
        return CT_SKIP;
    return t.failures ? CT_FAIL : CT_PASS;
}

static void ct_append(struct ct_result *r, const char *data, size_t len)
{
    if (r->len + len + 1 > r->cap) {
        size_t cap = r->cap ? r->cap * 2 : 1024;
        while (cap < r->len + len + 1)
            cap *= 2;
        r->output = realloc(r->output, cap);
        if (!r->output) {
            fprintf(stderr, "ct: out of memory\n");
            exit(2);
        }
        r->cap = cap;
    }
    memcpy(r->output + r->len, data, len);
    r->len += len;
    r->output[r->len] = '\0';
}

/* Each test runs in a new process: this executable, started again with
 * "--child NAME". Not fork() alone: on macOS, Metal's shader compiler is an
 * XPC service, unreachable from a child that forked without exec. The child's
 * stdout and stderr go to the returned fd. */
static char ct_exe[4096];

static pid_t ct_spawn(const char *name, unsigned timeout, int *fd)
{
    posix_spawn_file_actions_t actions;
    char timeout_arg[16];
    char *args[6];
    int fds[2];
    pid_t pid;
    int err;

    if (pipe(fds) != 0)
        return -1;
    fcntl(fds[0], F_SETFD, FD_CLOEXEC); /* parallel children must not inherit each other's pipes */
    fcntl(fds[1], F_SETFD, FD_CLOEXEC);
    snprintf(timeout_arg, sizeof(timeout_arg), "%u", timeout);
    args[0] = ct_exe;
    args[1] = (char *)"--child";
    args[2] = (char *)name;
    args[3] = (char *)"--timeout";
    args[4] = timeout_arg;
    args[5] = NULL;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, fds[1], STDERR_FILENO);
    err = posix_spawn(&pid, ct_exe, &actions, NULL, args, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(fds[1]);
    if (err) {
        close(fds[0]);
        errno = err;
        return -1;
    }
    *fd = fds[0];
    return pid;
}

static void ct_banner_here(void);

/* The child side of ct_spawn. */
static int ct_child(const struct ct_registry *reg, const char *name, unsigned timeout)
{
    size_t i;
    setvbuf(stdout, NULL, _IONBF, 0);
    if (timeout)
        alarm(timeout);
    if (strcmp(name, "@banner") == 0) {
        ct_banner_here();
        return 0;
    }
    for (i = 0; i < reg->count; i++)
        if (strcmp(reg->entries[i].name, name) == 0) {
            switch (ct_run_here(reg->entries[i].test)) {
            case CT_PASS:
                return 0;
            case CT_SKIP:
                return 2;
            default:
                return 1;
            }
        }
    fprintf(stderr, "ct: no test named %s\n", name);
    return 3;
}

static void ct_decode(struct ct_result *r, int status, unsigned timeout)
{
    if (WIFEXITED(status)) {
        switch (WEXITSTATUS(status)) {
        case 0:
            r->status = CT_PASS;
            break;
        case 2:
            r->status = CT_SKIP;
            break;
        case 1:
            r->status = CT_FAIL;
            break;
        default:
            r->status = CT_FAIL;
            snprintf(r->reason, sizeof(r->reason), "exited with status %d", WEXITSTATUS(status));
            break;
        }
    } else if (WIFSIGNALED(status)) {
        r->status = CT_FAIL;
        if (WTERMSIG(status) == SIGALRM)
            snprintf(r->reason, sizeof(r->reason), "timed out after %u s", timeout);
        else
            snprintf(r->reason, sizeof(r->reason), "crashed: %s", strsignal(WTERMSIG(status)));
    } else {
        r->status = CT_FAIL;
        snprintf(r->reason, sizeof(r->reason), "ended abnormally");
    }
}

static void ct_print_indented(const char *text)
{
    const char *line = text;
    while (line && *line) {
        const char *end = strchr(line, '\n');
        size_t len = end ? (size_t)(end - line) : strlen(line);
        printf("    %.*s\n", (int)len, line);
        line = end ? end + 1 : NULL;
    }
}

struct ct_baseline;
static int ct_in_baseline(const struct ct_baseline *b, const char *name);

/* With a baseline, failures outside it are expected: one line, no details
 * (unless verbose). */
static void ct_print_result(const struct ct_entry *e, const struct ct_result *r, int verbose,
                            const struct ct_baseline *baseline)
{
    static const char *const names[] = { "PASS", "FAIL", "SKIP" };
    int expected = r->status == CT_FAIL && baseline && !ct_in_baseline(baseline, e->name);
    printf("%s %s%s", names[r->status], e->name, expected && !verbose ? " (expected)" : "");
    if (r->seconds >= 1.0)
        printf(" [%.1f s]", r->seconds);
    putchar('\n');
    if (expected && !verbose) {
        fflush(stdout);
        return;
    }
    if (r->status == CT_FAIL)
        printf("    requirement: %s\n", e->test->spec);
    if (r->reason[0])
        printf("    %s\n", r->reason);
    if (r->output && (r->status != CT_PASS || verbose))
        ct_print_indented(r->output);
    fflush(stdout);
}

static double ct_now(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

struct ct_job {
    pid_t pid;
    int fd;
    size_t index;
    double start;
};

static void ct_run_all(const struct ct_entry *const *sel, size_t n, struct ct_result *results,
                       const struct ct_options *o)
{
    struct ct_job *jobs = calloc(o->jobs, sizeof(*jobs));
    struct pollfd *fds = calloc(o->jobs, sizeof(*fds));
    size_t next = 0, printed = 0;
    unsigned active = 0, i;

    if (!jobs || !fds) {
        fprintf(stderr, "ct: out of memory\n");
        exit(2);
    }
    while (printed < n) {
        while (active < o->jobs && next < n) {
            struct ct_job *j = &jobs[active];
            j->index = next++;
            j->start = ct_now();
            j->pid = ct_spawn(sel[j->index]->name, o->timeout, &j->fd);
            if (j->pid < 0) {
                results[j->index].status = CT_FAIL;
                snprintf(results[j->index].reason, sizeof(results[j->index].reason), "cannot start: %s",
                         strerror(errno));
                results[j->index].done = 1;
                continue;
            }
            active++;
        }
        if (active) {
            for (i = 0; i < active; i++) {
                fds[i].fd = jobs[i].fd;
                fds[i].events = POLLIN;
                fds[i].revents = 0;
            }
            if (poll(fds, active, -1) < 0 && errno != EINTR) {
                perror("ct: poll");
                exit(2);
            }
            for (i = 0; i < active; i++) {
                char buf[4096];
                ssize_t got;
                if (!fds[i].revents)
                    continue;
                got = read(jobs[i].fd, buf, sizeof(buf));
                if (got > 0) {
                    ct_append(&results[jobs[i].index], buf, (size_t)got);
                } else if (got == 0 || errno != EINTR) {
                    int status = 0;
                    close(jobs[i].fd);
                    while (waitpid(jobs[i].pid, &status, 0) < 0 && errno == EINTR)
                        ;
                    ct_decode(&results[jobs[i].index], status, o->timeout);
                    results[jobs[i].index].seconds = ct_now() - jobs[i].start;
                    results[jobs[i].index].done = 1;
                    jobs[i] = jobs[active - 1];
                    fds[i] = fds[active - 1];
                    active--;
                    i--;
                }
            }
        }
        while (printed < n && results[printed].done) {
            ct_print_result(sel[printed], &results[printed], o->verbose, o->expected);
            printed++;
        }
    }
    free(jobs);
    free(fds);
}

/* ==========================================================================
 * Driver banner
 * ========================================================================== */

static void ct_banner_here(void)
{
    struct ct_test test;
    struct ct t;
    struct ct_env e;
    memset(&test, 0, sizeof(test));
    test.name = "banner";
    memset(&t, 0, sizeof(t));
    t.test = &test;
    if (setjmp(t.jump) == 0) {
        uint32_t v = ct_instance_version();
        printf("# instance version %u.%u.%u\n", VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v),
               VK_API_VERSION_PATCH(v));
        ct_env_init(&t, &e);
        printf("# device %u: %s, apiVersion %u.%u.%u, vendor 0x%x, device 0x%x\n", ct_device_index(),
               e.props.deviceName, VK_API_VERSION_MAJOR(e.props.apiVersion),
               VK_API_VERSION_MINOR(e.props.apiVersion), VK_API_VERSION_PATCH(e.props.apiVersion),
               e.props.vendorID, e.props.deviceID);
        if (e.version >= VK_API_VERSION_1_2) {
            struct ct_properties p;
            ct_get_properties(&e, &p, 0);
            p.v12.driverName[VK_MAX_DRIVER_NAME_SIZE - 1] = '\0';
            p.v12.driverInfo[VK_MAX_DRIVER_INFO_SIZE - 1] = '\0';
            printf("# driver: %s %s\n", p.v12.driverName, p.v12.driverInfo);
        }
        ct_env_fini(&e);
    }
}

static void ct_banner(unsigned timeout)
{
    struct ct_result r;
    int fd = -1, status = 0;
    char buf[4096];
    ssize_t got;
    pid_t pid = ct_spawn("@banner", timeout, &fd);

    if (pid < 0)
        return;
    memset(&r, 0, sizeof(r));
    while ((got = read(fd, buf, sizeof(buf))) > 0 || (got < 0 && errno == EINTR))
        if (got > 0)
            ct_append(&r, buf, (size_t)got);
    close(fd);
    waitpid(pid, &status, 0);
    if (r.output)
        fputs(r.output, stdout);
    free(r.output);
}

/* ==========================================================================
 * Coverage
 * ========================================================================== */

static const struct ct_command *ct_find_command(const char *name, size_t len)
{
    size_t i;
    for (i = 0; i < ct_command_count; i++)
        if (strlen(ct_commands[i].name) == len && strncmp(ct_commands[i].name, name, len) == 0)
            return &ct_commands[i];
    return NULL;
}

/* Calls fn for each command a test declares; returns the number of unknown
 * names (printed). */
static int ct_each_command(const struct ct_entry *e, void (*fn)(const struct ct_command *, void *), void *user)
{
    const char *s = e->test->commands;
    int unknown = 0;
    while (s && *s) {
        size_t len;
        const struct ct_command *c;
        while (*s == ' ')
            s++;
        len = strcspn(s, " ");
        if (!len)
            break;
        c = ct_find_command(s, len);
        if (c) {
            if (fn)
                fn(c, user);
        } else {
            fprintf(stderr, "ct: %s declares unknown command '%.*s'\n", e->name, (int)len, s);
            unknown++;
        }
        s += len;
    }
    return unknown;
}

static int ct_is_version(const char *origin)
{
    return origin[0] >= '0' && origin[0] <= '9';
}

static void ct_mark(const struct ct_command *c, void *user)
{
    ((unsigned char *)user)[c - ct_commands] = 1;
}

static void ct_coverage(const struct ct_entry *const *sel, size_t n)
{
    unsigned char *covered = calloc(ct_command_count, 1);
    const char *origins[32];
    size_t i, j, origin_count = 0, total = 0;

    if (!covered)
        return;
    for (i = 0; i < n; i++)
        ct_each_command(sel[i], ct_mark, covered);
    for (i = 0; i < ct_command_count; i++) {
        for (j = 0; j < origin_count; j++)
            if (strcmp(origins[j], ct_commands[i].origin) == 0)
                break;
        if (j == origin_count && origin_count < CT_ARRAY_SIZE(origins))
            origins[origin_count++] = ct_commands[i].origin;
        total += covered[i];
    }
    printf("Command coverage: %zu of %zu commands (%.0f%%)\n", total, ct_command_count,
           100.0 * (double)total / (double)ct_command_count);
    for (j = 0; j < origin_count; j++) {
        size_t have = 0, all = 0;
        char label[64];
        for (i = 0; i < ct_command_count; i++)
            if (strcmp(ct_commands[i].origin, origins[j]) == 0) {
                all++;
                have += covered[i];
            }
        snprintf(label, sizeof(label), "%s%s", ct_is_version(origins[j]) ? "Vulkan " : "", origins[j]);
        printf("  %-24s %3zu / %3zu\n", label, have, all);
    }
    printf("Not covered:\n");
    for (j = 0; j < origin_count; j++) {
        size_t col;
        printf("  %s%s:", ct_is_version(origins[j]) ? "Vulkan " : "", origins[j]);
        col = 10 + strlen(origins[j]);
        for (i = 0; i < ct_command_count; i++) {
            size_t len;
            if (covered[i] || strcmp(ct_commands[i].origin, origins[j]) != 0)
                continue;
            len = strlen(ct_commands[i].name) + 1;
            if (col + len > 100) {
                printf("\n   ");
                col = 3;
            }
            printf(" %s", ct_commands[i].name);
            col += len;
        }
        printf("\n");
    }
    free(covered);
}

/* ==========================================================================
 * Baseline
 * ========================================================================== */

struct ct_baseline {
    char **names;
    size_t count;
};

static int ct_load_baseline(const char *path, struct ct_baseline *b)
{
    FILE *f = fopen(path, "r");
    char line[256];
    size_t cap = 0;

    memset(b, 0, sizeof(*b));
    if (!f) {
        fprintf(stderr, "ct: cannot open baseline %s: %s\n", path, strerror(errno));
        return -1;
    }
    while (fgets(line, sizeof(line), f)) {
        char *s = line, *end;
        char *hash = strchr(s, '#');
        if (hash)
            *hash = '\0';
        while (*s == ' ' || *s == '\t')
            s++;
        end = s + strlen(s);
        while (end > s && (end[-1] == '\n' || end[-1] == '\r' || end[-1] == ' ' || end[-1] == '\t'))
            *--end = '\0';
        if (!*s)
            continue;
        if (b->count == cap) {
            cap = cap ? cap * 2 : 256;
            b->names = realloc(b->names, cap * sizeof(*b->names));
            if (!b->names) {
                fclose(f);
                return -1;
            }
        }
        b->names[b->count] = malloc(strlen(s) + 1);
        if (!b->names[b->count]) {
            fclose(f);
            return -1;
        }
        strcpy(b->names[b->count++], s);
    }
    fclose(f);
    return 0;
}

static int ct_in_baseline(const struct ct_baseline *b, const char *name)
{
    size_t i;
    for (i = 0; i < b->count; i++)
        if (strcmp(b->names[i], name) == 0)
            return 1;
    return 0;
}

/* Returns the number of problems: regressions and unknown names. */
static int ct_compare_baseline(const struct ct_baseline *b, const struct ct_registry *reg,
                               const struct ct_entry *const *sel, const struct ct_result *results, size_t n)
{
    size_t i, j, regressions = 0, fixed = 0, unknown = 0;

    for (i = 0; i < b->count; i++) {
        for (j = 0; j < reg->count; j++)
            if (strcmp(reg->entries[j].name, b->names[i]) == 0)
                break;
        if (j == reg->count) {
            if (!unknown++)
                printf("\nUnknown tests in the baseline (renamed or removed? edit the baseline):\n");
            printf("  %s\n", b->names[i]);
        }
    }
    for (i = 0; i < n; i++)
        if (ct_in_baseline(b, sel[i]->name) && results[i].status != CT_PASS) {
            if (!regressions++)
                printf("\nREGRESSIONS (in the baseline, no longer passing):\n");
            printf("  %s\n", sel[i]->name);
        }
    for (i = 0; i < n; i++)
        if (!ct_in_baseline(b, sel[i]->name) && results[i].status == CT_PASS) {
            if (!fixed++)
                printf("\nNewly passing (add these lines to the baseline):\n");
            printf("%s\n", sel[i]->name);
        }
    printf("\nBaseline: %zu regressions, %zu newly passing, %zu unknown names\n", regressions, fixed, unknown);
    return (int)(regressions + unknown);
}

/* ==========================================================================
 * main
 * ========================================================================== */

static int ct_selected(const struct ct_options *o, const char *name)
{
    int i;
    if (!o->pattern_count)
        return 1;
    for (i = 0; i < o->pattern_count; i++) {
        const char *p = o->patterns[i];
        if (strpbrk(p, "*?[") ? fnmatch(p, name, 0) == 0 : strstr(name, p) != NULL)
            return 1;
    }
    return 0;
}

static void ct_usage(void)
{
    fprintf(stderr, "usage: ct [-l] [-c] [-b baseline] [-j jobs] [-t seconds] [-v] [--in-process] [pattern...]\n");
}

int main(int argc, char **argv)
{
    struct ct_registry reg;
    struct ct_options o;
    struct ct_baseline baseline;
    const struct ct_entry **sel;
    struct ct_result *results;
    size_t i, j, n = 0, counts[3] = { 0, 0, 0 };
    int a, unknown = 0, problems = 0;
    double start;
    const char *jobs = getenv("CT_JOBS"), *child = NULL;
    uint32_t exe_size = sizeof(ct_exe);

    memset(&reg, 0, sizeof(reg));
    memset(&o, 0, sizeof(o));
    o.jobs = jobs ? (unsigned)strtoul(jobs, NULL, 10) : (unsigned)sysconf(_SC_NPROCESSORS_ONLN);
    o.timeout = 120;
    o.patterns = calloc((size_t)argc, sizeof(*o.patterns));
    if (_NSGetExecutablePath(ct_exe, &exe_size) != 0)
        snprintf(ct_exe, sizeof(ct_exe), "%s", argv[0]);
    for (a = 1; a < argc; a++) {
        const char *s = argv[a];
        if (!strcmp(s, "--child") && a + 1 < argc)
            child = argv[++a];
        else if (!strcmp(s, "-l") || !strcmp(s, "--list"))
            o.list = 1;
        else if (!strcmp(s, "-c") || !strcmp(s, "--coverage"))
            o.coverage = 1;
        else if (!strcmp(s, "-v") || !strcmp(s, "--verbose"))
            o.verbose = 1;
        else if (!strcmp(s, "--in-process"))
            o.in_process = 1;
        else if ((!strcmp(s, "-b") || !strcmp(s, "--baseline")) && a + 1 < argc)
            o.baseline = argv[++a];
        else if (!strcmp(s, "-j") && a + 1 < argc)
            o.jobs = (unsigned)strtoul(argv[++a], NULL, 10);
        else if ((!strcmp(s, "-t") || !strcmp(s, "--timeout")) && a + 1 < argc)
            o.timeout = (unsigned)strtoul(argv[++a], NULL, 10);
        else if (s[0] == '-') {
            ct_usage();
            return 2;
        } else
            o.patterns[o.pattern_count++] = s;
    }
    if (!o.jobs)
        o.jobs = 1;

    ct_register_all(&reg);
    for (i = 0; i < reg.count; i++) {
        for (j = 0; j < i; j++)
            if (!strcmp(reg.entries[i].name, reg.entries[j].name)) {
                fprintf(stderr, "ct: duplicate test %s\n", reg.entries[i].name);
                unknown++;
            }
        unknown += ct_each_command(&reg.entries[i], NULL, NULL);
    }
    if (unknown)
        return 2;
    if (child)
        return ct_child(&reg, child, o.timeout);

    sel = calloc(reg.count ? reg.count : 1, sizeof(*sel));
    results = calloc(reg.count ? reg.count : 1, sizeof(*results));
    if (!sel || !results)
        return 2;
    for (i = 0; i < reg.count; i++)
        if (ct_selected(&o, reg.entries[i].name))
            sel[n++] = &reg.entries[i];

    if (o.list) {
        for (i = 0; i < n; i++)
            printf("%s\n    %s\n    uses: %s\n", sel[i]->name, sel[i]->test->spec, sel[i]->test->commands);
        return 0;
    }
    if (o.coverage) {
        ct_coverage(sel, n);
        return 0;
    }
    if (o.baseline) {
        if (ct_load_baseline(o.baseline, &baseline) != 0)
            return 2;
        o.expected = &baseline;
    }

    start = ct_now();
    if (o.in_process) {
        for (i = 0; i < n; i++) {
            printf("---- %s\n", sel[i]->name);
            fflush(stdout);
            results[i].status = ct_run_here(sel[i]->test);
            ct_print_result(sel[i], &results[i], 1, NULL);
        }
    } else {
        ct_banner(o.timeout);
        ct_run_all(sel, n, results, &o);
    }
    for (i = 0; i < n; i++)
        counts[results[i].status]++;
    printf("\n%zu tests: %zu passed, %zu failed, %zu skipped (%.1f s)\n", n, counts[CT_PASS], counts[CT_FAIL],
           counts[CT_SKIP], ct_now() - start);

    if (o.baseline)
        problems = ct_compare_baseline(&baseline, &reg, sel, results, n);
    else
        problems = (int)counts[CT_FAIL];
    return problems ? 1 : 0;
}
