/* Running translated compute shaders on Metal, for the translator tests.
 *
 * Include after plasmavk.h (with PLASMAVK_IMPLEMENTATION) and pvk_test.h.
 * Uses the legacy Metal 3 compute encoder through raw messages; buffers
 * reached only through the set tables' GPU addresses are made resident with
 * useResource:usage:. */
#ifndef PVK_MSL_TEST_H
#define PVK_MSL_TEST_H

#include <stdio.h>
#include <string.h>

static id mt_device, mt_queue;

static bool mt_init(void)
{
    mt_device = pvk_MTLCreateSystemDefaultDevice();
    if (!mt_device)
        return false;
    mt_queue = PVK_MSG(id, (id, SEL))(mt_device, sel_registerName("newCommandQueue"));
    return mt_queue != NULL;
}

static void mt_shutdown(void)
{
    pvk_release(mt_queue);
    pvk_release(mt_device);
}

static id mt_string(const char *s)
{
    return PVK_MSG(id, (id, SEL, const char *))((id)objc_getClass("NSString"), sel_registerName("stringWithUTF8String:"),
                                                s);
}

static const char *mt_error_text(id error)
{
    return error ? pvk_ns_utf8(PVK_MSG(id, (id, SEL))(error, sel_registerName("localizedDescription"))) : "(no error)";
}

static void mt_print_numbered(const char *src)
{
    int line = 1;
    fprintf(stderr, "%4d  ", line);
    for (; *src; src++) {
        fputc(*src, stderr);
        if (*src == '\n' && src[1])
            fprintf(stderr, "%4d  ", ++line);
    }
    fputc('\n', stderr);
}

/* Translates SPIR-V; prints the error on failure. */
static bool mt_translate(const uint32_t *words, size_t count, const struct pvk_msl_options *options,
                         struct pvk_msl_result *res)
{
    VkResult r = pvk_spirv_to_msl(words, count, NULL, options, res);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "translation failed (%d): %s\n", (int)r, res->error);
        return false;
    }
    return true;
}

/* Compiles the MSL and builds a compute pipeline (retained), NULL on failure
 * after printing the compiler error and the numbered source. */
static id mt_pipeline(const struct pvk_msl_result *res)
{
    void *pool = objc_autoreleasePoolPush();
    id options, library, function, pipeline = NULL, error = NULL;

    options = PVK_MSG(id, (id, SEL))((id)objc_getClass("MTLCompileOptions"), sel_registerName("new"));
    PVK_MSG(void, (id, SEL, long))(options, sel_registerName("setMathMode:"), PVK_MTLMathModeSafe);
    library = PVK_MSG(id, (id, SEL, id, id, id *))(mt_device, sel_registerName("newLibraryWithSource:options:error:"),
                                                   mt_string(res->msl), options, &error);
    pvk_release(options);
    if (!library) {
        fprintf(stderr, "MSL compilation failed:\n%s\n", mt_error_text(error));
        mt_print_numbered(res->msl);
        objc_autoreleasePoolPop(pool);
        return NULL;
    }
    function = PVK_MSG(id, (id, SEL, id))(library, sel_registerName("newFunctionWithName:"), mt_string(res->entry_point));
    if (function) {
        error = NULL;
        pipeline = PVK_MSG(id, (id, SEL, id, id *))(mt_device, sel_registerName("newComputePipelineStateWithFunction:error:"),
                                                    function, &error);
        if (!pipeline)
            fprintf(stderr, "pipeline creation failed: %s\n", mt_error_text(error));
    } else {
        fprintf(stderr, "kernel %s not found in the library\n", res->entry_point);
    }
    pvk_release(function);
    pvk_release(library);
    objc_autoreleasePoolPop(pool);
    return pipeline;
}

struct mt_buffer {
    id buffer;
    void *data;
    uint64_t address;
    size_t size;
};

/* A shared-storage buffer, zero-filled, then initialized from init (if any). */
static struct mt_buffer mt_buffer(size_t size, const void *init, size_t init_size)
{
    struct mt_buffer b;
    memset(&b, 0, sizeof(b));
    b.size = size;
    b.buffer = PVK_MSG(id, (id, SEL, unsigned long, unsigned long))(mt_device, sel_registerName("newBufferWithLength:options:"),
                                                                    size, 0 /* shared */);
    if (!b.buffer)
        return b;
    b.data = PVK_MSG(void *, (id, SEL))(b.buffer, sel_registerName("contents"));
    b.address = PVK_MSG(uint64_t, (id, SEL))(b.buffer, sel_registerName("gpuAddress"));
    memset(b.data, 0, size);
    if (init)
        memcpy(b.data, init, init_size < size ? init_size : size);
    return b;
}

static void mt_buffer_free(struct mt_buffer *b)
{
    pvk_release(b->buffer);
    memset(b, 0, sizeof(*b));
}

/* One descriptor: element `index` of (set, binding) is buf + offset. */
struct mt_binding {
    uint32_t set, binding, index;
    struct mt_buffer *buf;
    uint64_t offset;
};

/* Fills the set tables from the reflection, binds everything and runs
 * groups[0..2] threadgroups of res->workgroup_size threads. */
static bool mt_dispatch(id pipeline, const struct pvk_msl_result *res, const struct mt_binding *bindings,
                        uint32_t nbindings, const void *push, size_t push_size, uint32_t gx, uint32_t gy, uint32_t gz)
{
    struct mt_buffer sets[PVK_MSL_MAX_SETS], sizes;
    void *pool = objc_autoreleasePoolPush();
    id cb, enc;
    pvk_MTLSize groups, threads;
    uint32_t s, i, j;
    unsigned long status;
    bool ok = true;

    memset(sets, 0, sizeof(sets));
    memset(&sizes, 0, sizeof(sizes));
    for (s = 0; s < PVK_MSL_MAX_SETS; s++) {
        size_t size = 8;
        if (!(res->set_mask & (1u << s)))
            continue;
        for (i = 0; i < res->resource_count; i++)
            if (res->resources[i].set == s && res->resources[i].offset + 8u * res->resources[i].count > size)
                size = res->resources[i].offset + 8u * res->resources[i].count;
        sets[s] = mt_buffer(size, NULL, 0);
    }
    if (res->buffer_size_count)
        sizes = mt_buffer(8u * res->buffer_size_count, NULL, 0);
    for (i = 0; i < nbindings; i++) {
        const struct mt_binding *b = &bindings[i];
        for (j = 0; j < res->resource_count; j++) {
            const struct pvk_msl_resource *r = &res->resources[j];
            if (r->set != b->set || r->binding != b->binding || b->index >= r->count)
                continue;
            ((uint64_t *)sets[b->set].data)[(r->offset / 8u) + b->index] = b->buf->address + b->offset;
            if (r->size_index >= 0)
                ((uint64_t *)sizes.data)[r->size_index + (int32_t)b->index] = b->buf->size - b->offset;
        }
    }

    cb = PVK_MSG(id, (id, SEL))(mt_queue, sel_registerName("commandBuffer"));
    enc = PVK_MSG(id, (id, SEL))(cb, sel_registerName("computeCommandEncoder"));
    PVK_MSG(void, (id, SEL, id))(enc, sel_registerName("setComputePipelineState:"), pipeline);
    for (s = 0; s < PVK_MSL_MAX_SETS; s++)
        if (sets[s].buffer)
            PVK_MSG(void, (id, SEL, id, unsigned long, unsigned long))(enc, sel_registerName("setBuffer:offset:atIndex:"),
                                                                       sets[s].buffer, 0, s);
    if (sizes.buffer)
        PVK_MSG(void, (id, SEL, id, unsigned long, unsigned long))(enc, sel_registerName("setBuffer:offset:atIndex:"),
                                                                   sizes.buffer, 0, res->buffer_sizes_buffer);
    if (push && push_size)
        PVK_MSG(void, (id, SEL, const void *, unsigned long, unsigned long))(enc, sel_registerName("setBytes:length:atIndex:"),
                                                                             push, push_size, res->push_constant_buffer);
    for (i = 0; i < nbindings; i++)
        PVK_MSG(void, (id, SEL, id, unsigned long))(enc, sel_registerName("useResource:usage:"), bindings[i].buf->buffer,
                                                    PVK_MTLResourceUsageRead | PVK_MTLResourceUsageWrite);
    groups.width = gx;
    groups.height = gy;
    groups.depth = gz;
    threads.width = res->workgroup_size[0];
    threads.height = res->workgroup_size[1];
    threads.depth = res->workgroup_size[2];
    PVK_MSG(void, (id, SEL, pvk_MTLSize, pvk_MTLSize))(enc, sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"),
                                                       groups, threads);
    PVK_MSG(void, (id, SEL))(enc, sel_registerName("endEncoding"));
    PVK_MSG(void, (id, SEL))(cb, sel_registerName("commit"));
    PVK_MSG(void, (id, SEL))(cb, sel_registerName("waitUntilCompleted"));
    status = PVK_MSG(unsigned long, (id, SEL))(cb, sel_registerName("status"));
    if (status != PVK_MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "command buffer failed: %s\n", mt_error_text(PVK_MSG(id, (id, SEL))(cb, sel_registerName("error"))));
        ok = false;
    }
    objc_autoreleasePoolPop(pool);
    for (s = 0; s < PVK_MSL_MAX_SETS; s++)
        mt_buffer_free(&sets[s]);
    mt_buffer_free(&sizes);
    return ok;
}

/* Translate + compile; on success *pipeline is retained and res must be freed. */
#define MT_BUILD(words, options, res, pipeline)                                      \
    do {                                                                             \
        REQUIRE(mt_translate((words), PVK_ARRAY_SIZE(words), (options), &(res)));    \
        (pipeline) = mt_pipeline(&(res));                                            \
        if (!(pipeline))                                                             \
            pvk_msl_result_free(&(res));                                             \
        REQUIRE((pipeline) != NULL);                                                 \
    } while (0)

static bool mt_near(double got, double want, double tolerance)
{
    double d = got - want, m = want < 0 ? -want : want;
    if (d < 0)
        d = -d;
    return d <= tolerance * (m > 1.0 ? m : 1.0);
}

#endif /* PVK_MSL_TEST_H */
