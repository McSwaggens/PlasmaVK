/* Running translated compute shaders on Metal, for the translator tests.
 *
 * Include after plasmavk.h (with PLASMAVK_IMPLEMENTATION) and pvk_test.h.
 * Uses the legacy Metal 3 compute encoder through raw messages. Descriptor
 * set tables are filled in the section 11 format from the translator's
 * reflection; buffers and textures reached only through them are made
 * resident with useResource:usage:. Textures are shared-storage (CPU
 * writable), samplers support argument buffers. Functions are static inline
 * so tests may use any subset (pvk_msl_render.h builds on them). */
#ifndef PVK_MSL_TEST_H
#define PVK_MSL_TEST_H

#include <stdio.h>
#include <string.h>

static id mt_device, mt_queue;

static inline bool mt_init(void)
{
    mt_device = pvk_MTLCreateSystemDefaultDevice();
    if (!mt_device)
        return false;
    mt_queue = PVK_MSG(id, (id, SEL))(mt_device, sel_registerName("newCommandQueue"));
    return mt_queue != NULL;
}

static inline void mt_shutdown(void)
{
    pvk_release(mt_queue);
    pvk_release(mt_device);
}

static inline id mt_string(const char *s)
{
    return PVK_MSG(id, (id, SEL, const char *))((id)objc_getClass("NSString"), sel_registerName("stringWithUTF8String:"),
                                                s);
}

static inline const char *mt_error_text(id error)
{
    return error ? pvk_ns_utf8(PVK_MSG(id, (id, SEL))(error, sel_registerName("localizedDescription"))) : "(no error)";
}

static inline void mt_print_numbered(const char *src)
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
static inline bool mt_translate(const uint32_t *words, size_t count, const struct pvk_msl_options *options,
                                struct pvk_msl_result *res)
{
    VkResult r = pvk_spirv_to_msl(words, count, NULL, options, res);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "translation failed (%d): %s\n", (int)r, res->error);
        return false;
    }
    return true;
}

/* Compiles the MSL as section 17 does (MSL 3.2, safe math) and builds a
 * compute pipeline (retained), NULL on failure after printing the compiler
 * error and the numbered source. */
static inline id mt_pipeline(const struct pvk_msl_result *res)
{
    void *pool = objc_autoreleasePoolPush();
    id options, library, function, pipeline = NULL, error = NULL;

    options = PVK_MSG(id, (id, SEL))((id)objc_getClass("MTLCompileOptions"), sel_registerName("new"));
    PVK_MSG(void, (id, SEL, long))(options, sel_registerName("setMathMode:"), PVK_MTLMathModeSafe);
    PVK_MSG(void, (id, SEL, unsigned long))(options, sel_registerName("setLanguageVersion:"), PVK_MTLLanguageVersion3_2);
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
static inline struct mt_buffer mt_buffer(size_t size, const void *init, size_t init_size)
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

static inline void mt_buffer_free(struct mt_buffer *b)
{
    pvk_release(b->buffer);
    memset(b, 0, sizeof(*b));
}

/* ---- Textures and samplers ---- */

/* A shared-storage texture: type, format, width/height/depth, array length
 * (cube arrays: cubes), mip levels, sample count. */
static inline id mt_texture(pvk_MTLTextureType type, pvk_MTLPixelFormat format, unsigned long w, unsigned long h,
                            unsigned long d, unsigned long layers, unsigned long mips, unsigned long samples)
{
    id desc = PVK_MSG(id, (id, SEL))((id)objc_getClass("MTLTextureDescriptor"), sel_registerName("new")), tex;
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setTextureType:"), type);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setPixelFormat:"), format);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setWidth:"), w);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setHeight:"), h);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setDepth:"), d);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setArrayLength:"), layers);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setMipmapLevelCount:"), mips);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setSampleCount:"), samples);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setStorageMode:"),
                                            samples > 1 ? PVK_MTLStorageModePrivate : PVK_MTLStorageModeShared);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setUsage:"),
                                            PVK_MTLTextureUsageShaderRead | PVK_MTLTextureUsageShaderWrite |
                                                (samples > 1 ? PVK_MTLTextureUsageRenderTarget : 0) |
                                                ((format == PVK_MTLPixelFormatR32Uint || format == PVK_MTLPixelFormatR32Sint)
                                                     ? PVK_MTLTextureUsageShaderAtomic
                                                     : 0));
    tex = PVK_MSG(id, (id, SEL, id))(mt_device, sel_registerName("newTextureWithDescriptor:"), desc);
    pvk_release(desc);
    return tex;
}

/* Uploads one image (w x h x d texels of `bpp` bytes) to mip level / slice. */
static inline void mt_upload(id tex, unsigned long level, unsigned long slice, unsigned long w, unsigned long h,
                             unsigned long d, unsigned long bpp, const void *data)
{
    pvk_MTLRegion r;
    memset(&r, 0, sizeof(r));
    r.size.width = w;
    r.size.height = h;
    r.size.depth = d;
    PVK_MSG(void, (id, SEL, pvk_MTLRegion, unsigned long, unsigned long, const void *, unsigned long, unsigned long))(
        tex, sel_registerName("replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:"), r, level, slice,
        data, w * bpp, w * h * bpp);
}

static inline void mt_download(id tex, unsigned long level, unsigned long slice, unsigned long w, unsigned long h,
                               unsigned long d, unsigned long bpp, void *data)
{
    pvk_MTLRegion r;
    memset(&r, 0, sizeof(r));
    r.size.width = w;
    r.size.height = h;
    r.size.depth = d;
    PVK_MSG(void, (id, SEL, void *, unsigned long, unsigned long, pvk_MTLRegion, unsigned long, unsigned long))(
        tex, sel_registerName("getBytes:bytesPerRow:bytesPerImage:fromRegion:mipmapLevel:slice:"), data, w * bpp,
        w * h * bpp, r, level, slice);
}

/* A texture buffer over buf (offset 0), width texels. */
static inline id mt_texture_buffer(struct mt_buffer *buf, pvk_MTLPixelFormat format, unsigned long width, unsigned long bpp)
{
    id desc = PVK_MSG(id, (id, SEL, unsigned long, unsigned long, unsigned long, unsigned long))(
        (id)objc_getClass("MTLTextureDescriptor"),
        sel_registerName("textureBufferDescriptorWithPixelFormat:width:resourceOptions:usage:"), format, width, 0,
        PVK_MTLTextureUsageShaderRead | PVK_MTLTextureUsageShaderWrite |
            ((format == PVK_MTLPixelFormatR32Uint || format == PVK_MTLPixelFormatR32Sint) ? PVK_MTLTextureUsageShaderAtomic
                                                                                         : 0));
    return PVK_MSG(id, (id, SEL, id, unsigned long, unsigned long))(
        buf->buffer, sel_registerName("newTextureWithDescriptor:offset:bytesPerRow:"), desc, 0, width * bpp);
}

/* Fills every sample of a multisampled color texture with a clear color. */
static inline void mt_clear(id tex, double r, double g, double b, double a)
{
    void *pool = objc_autoreleasePoolPush();
    id rp = PVK_MSG(id, (id, SEL))((id)objc_getClass("MTLRenderPassDescriptor"), sel_registerName("renderPassDescriptor"));
    id att = PVK_MSG(id, (id, SEL, unsigned long))(
        PVK_MSG(id, (id, SEL))(rp, sel_registerName("colorAttachments")), sel_registerName("objectAtIndexedSubscript:"), 0);
    struct { double r, g, b, a; } color = { r, g, b, a };
    id cb, enc;
    PVK_MSG(void, (id, SEL, id))(att, sel_registerName("setTexture:"), tex);
    PVK_MSG(void, (id, SEL, unsigned long))(att, sel_registerName("setLoadAction:"), 2 /* clear */);
    PVK_MSG(void, (id, SEL, unsigned long))(att, sel_registerName("setStoreAction:"), 1 /* store */);
    PVK_MSG(void, (id, SEL, double, double, double, double))(att, sel_registerName("setClearColor:"), color.r, color.g,
                                                             color.b, color.a);
    cb = PVK_MSG(id, (id, SEL))(mt_queue, sel_registerName("commandBuffer"));
    enc = PVK_MSG(id, (id, SEL, id))(cb, sel_registerName("renderCommandEncoderWithDescriptor:"), rp);
    PVK_MSG(void, (id, SEL))(enc, sel_registerName("endEncoding"));
    PVK_MSG(void, (id, SEL))(cb, sel_registerName("commit"));
    PVK_MSG(void, (id, SEL))(cb, sel_registerName("waitUntilCompleted"));
    objc_autoreleasePoolPop(pool);
}

struct mt_sampler_desc {
    pvk_MTLSamplerMinMagFilter filter;  /* min and mag */
    unsigned long mip;                  /* MTLSamplerMipFilter: 0 not mipmapped, 1 nearest, 2 linear */
    pvk_MTLSamplerAddressMode address;  /* s, t, r */
    bool unnormalized;
    pvk_MTLCompareFunction compare;     /* 0 (never): no comparison */
};

static inline id mt_sampler(const struct mt_sampler_desc *s)
{
    id desc = PVK_MSG(id, (id, SEL))((id)objc_getClass("MTLSamplerDescriptor"), sel_registerName("new")), smp;
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setMinFilter:"), s->filter);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setMagFilter:"), s->filter);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setMipFilter:"), s->mip);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setSAddressMode:"), s->address);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setTAddressMode:"), s->address);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setRAddressMode:"), s->address);
    PVK_MSG(void, (id, SEL, bool))(desc, sel_registerName("setNormalizedCoordinates:"), !s->unnormalized);
    if (s->compare)
        PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setCompareFunction:"), s->compare);
    PVK_MSG(void, (id, SEL, bool))(desc, sel_registerName("setSupportArgumentBuffers:"), true);
    smp = PVK_MSG(id, (id, SEL, id))(mt_device, sel_registerName("newSamplerStateWithDescriptor:"), desc);
    pvk_release(desc);
    return smp;
}

/* ---- Descriptors ---- */

/* One descriptor: element `index` of (set, binding). Buffers are buf +
 * offset with `range` bytes (0: to the end); textures and samplers are
 * Metal objects (both for a combined image sampler; planes 1 and 2 of a
 * multi-planar Y'CbCr image in planes, for 32-byte descriptors); inline
 * uniform blocks copy `bytes` to the binding's offset + index. */
struct mt_binding {
    uint32_t set, binding, index;
    struct mt_buffer *buf;
    uint64_t offset, range;
    id texture, sampler;
    const void *bytes;
    size_t nbytes;
    id planes[2];
};

static inline const struct pvk_msl_resource *mt_resource(const struct pvk_msl_result *res, uint32_t set, uint32_t binding)
{
    uint32_t i;
    for (i = 0; i < res->resource_count; i++)
        if (res->resources[i].set == set && res->resources[i].binding == binding)
            return &res->resources[i];
    return NULL;
}

/* Creates and fills the set tables (sets[N] for each set N the shader uses)
 * from the reflection. Free them with mt_buffer_free. */
static inline void mt_fill_sets(const struct pvk_msl_result *res, const struct mt_binding *bindings, uint32_t nbindings,
                                struct mt_buffer sets[PVK_MSL_MAX_SETS])
{
    uint32_t s, i;

    memset(sets, 0, sizeof(struct mt_buffer) * PVK_MSL_MAX_SETS);
    for (s = 0; s < PVK_MSL_MAX_SETS; s++) {
        size_t size = 16;
        if (!(res->set_mask & (1u << s)))
            continue;
        for (i = 0; i < res->resource_count; i++) {
            const struct pvk_msl_resource *r = &res->resources[i];
            size_t end = r->type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK ? r->offset + r->count
                                                                             : r->offset + (size_t)r->stride * r->count;
            if (r->set == s && end > size)
                size = end;
        }
        for (i = 0; i < nbindings; i++) { /* variable counts: as many as bound */
            const struct pvk_msl_resource *r = mt_resource(res, s, bindings[i].binding);
            if (bindings[i].set == s && r && r->offset + (size_t)r->stride * (bindings[i].index + 1) > size)
                size = r->offset + (size_t)r->stride * (bindings[i].index + 1);
        }
        sets[s] = mt_buffer(size, NULL, 0);
    }
    for (i = 0; i < nbindings; i++) {
        const struct mt_binding *b = &bindings[i];
        const struct pvk_msl_resource *r = mt_resource(res, b->set, b->binding);
        unsigned char *p;
        if (!r || b->set >= PVK_MSL_MAX_SETS || !sets[b->set].data)
            continue; /* the shader does not use it: only made resident */
        p = (unsigned char *)sets[b->set].data + r->offset + (size_t)r->stride * b->index;
        switch (r->type) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
            uint64_t desc[2];
            desc[0] = b->buf->address + b->offset;
            desc[1] = b->range ? b->range : b->buf->size - b->offset;
            memcpy(p, desc, 16);
            break;
        }
        case VK_DESCRIPTOR_TYPE_SAMPLER: {
            pvk_MTLResourceID rid = pvk_MTLSamplerState_gpuResourceID(b->sampler);
            memcpy(p, &rid, 8);
            break;
        }
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
            pvk_MTLResourceID t = pvk_MTLTexture_gpuResourceID(b->texture), sm = pvk_MTLSamplerState_gpuResourceID(b->sampler);
            unsigned k;
            memcpy(p, &t, 8);
            memcpy(p + 8, &sm, 8);
            for (k = 0; k < 2 && r->stride >= 32; k++) { /* Y'CbCr: { plane 0, sampler, plane 1, plane 2 } */
                pvk_MTLResourceID pl = { 0 };
                if (b->planes[k])
                    pl = pvk_MTLTexture_gpuResourceID(b->planes[k]);
                memcpy(p + 16 + 8 * k, &pl, 8);
            }
            break;
        }
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: { /* { image, residency texture: the image itself } */
            pvk_MTLResourceID t = pvk_MTLTexture_gpuResourceID(b->texture);
            memcpy(p, &t, 8);
            memcpy(p + 8, &t, 8);
            break;
        }
        case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
            memcpy((unsigned char *)sets[b->set].data + r->offset + b->index, b->bytes, b->nbytes);
            break;
        default: { /* images, texel buffers */
            pvk_MTLResourceID t = pvk_MTLTexture_gpuResourceID(b->texture);
            memcpy(p, &t, 8);
            break;
        }
        }
    }
}

/* Fills the set tables from the reflection, binds everything and runs
 * groups[0..2] threadgroups of res->workgroup_size threads. */
static inline bool mt_dispatch(id pipeline, const struct pvk_msl_result *res, const struct mt_binding *bindings,
                               uint32_t nbindings, const void *push, size_t push_size, uint32_t gx, uint32_t gy,
                               uint32_t gz)
{
    struct mt_buffer sets[PVK_MSL_MAX_SETS];
    void *pool = objc_autoreleasePoolPush();
    id cb, enc;
    pvk_MTLSize groups, threads;
    uint32_t s, i;
    unsigned long status;
    bool ok = true;

    mt_fill_sets(res, bindings, nbindings, sets);
    cb = PVK_MSG(id, (id, SEL))(mt_queue, sel_registerName("commandBuffer"));
    enc = PVK_MSG(id, (id, SEL))(cb, sel_registerName("computeCommandEncoder"));
    PVK_MSG(void, (id, SEL, id))(enc, sel_registerName("setComputePipelineState:"), pipeline);
    for (s = 0; s < PVK_MSL_MAX_SETS; s++)
        if (sets[s].buffer)
            PVK_MSG(void, (id, SEL, id, unsigned long, unsigned long))(enc, sel_registerName("setBuffer:offset:atIndex:"),
                                                                       sets[s].buffer, 0, s);
    if (push && push_size)
        PVK_MSG(void, (id, SEL, const void *, unsigned long, unsigned long))(enc, sel_registerName("setBytes:length:atIndex:"),
                                                                             push, push_size, res->push_constant_buffer);
    for (i = 0; i < nbindings; i++) {
        if (bindings[i].buf)
            PVK_MSG(void, (id, SEL, id, unsigned long))(enc, sel_registerName("useResource:usage:"), bindings[i].buf->buffer,
                                                        PVK_MTLResourceUsageRead | PVK_MTLResourceUsageWrite);
        if (bindings[i].texture)
            PVK_MSG(void, (id, SEL, id, unsigned long))(enc, sel_registerName("useResource:usage:"), bindings[i].texture,
                                                        PVK_MTLResourceUsageRead | PVK_MTLResourceUsageWrite);
        for (s = 0; s < 2; s++)
            if (bindings[i].planes[s])
                PVK_MSG(void, (id, SEL, id, unsigned long))(enc, sel_registerName("useResource:usage:"),
                                                            bindings[i].planes[s], PVK_MTLResourceUsageRead);
    }
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

static inline bool mt_near(double got, double want, double tolerance)
{
    double d = got - want, m = want < 0 ? -want : want;
    if (d < 0)
        d = -d;
    return d <= tolerance * (m > 1.0 ? m : 1.0);
}

#endif /* PVK_MSL_TEST_H */
