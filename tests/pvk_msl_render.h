/* Drawing with translated vertex and fragment shaders on Metal, for the
 * graphics translator tests.
 *
 * Include after pvk_msl_test.h. Uses the legacy Metal 3 render encoder
 * through raw messages. Each stage gets its own descriptor set tables, filled
 * from its own reflection (mt_fill_sets); the vertex buffer table is built
 * in the format the translator documents (struct pvk_msl_vertex_buffer). */
#ifndef PVK_MSL_RENDER_H
#define PVK_MSL_RENDER_H

/* Compiles MSL as section 17 does (MSL 3.2, safe math) and returns the named
 * function (retained), NULL after printing the error and the source. */
static inline id mr_function(const char *msl, const char *name)
{
    void *pool = objc_autoreleasePoolPush();
    id options, library, function = NULL, error = NULL;

    options = PVK_MSG(id, (id, SEL))((id)objc_getClass("MTLCompileOptions"), sel_registerName("new"));
    PVK_MSG(void, (id, SEL, long))(options, sel_registerName("setMathMode:"), PVK_MTLMathModeSafe);
    PVK_MSG(void, (id, SEL, unsigned long))(options, sel_registerName("setLanguageVersion:"), PVK_MTLLanguageVersion3_2);
    library = PVK_MSG(id, (id, SEL, id, id, id *))(mt_device, sel_registerName("newLibraryWithSource:options:error:"),
                                                   mt_string(msl), options, &error);
    pvk_release(options);
    if (!library) {
        fprintf(stderr, "MSL compilation failed:\n%s\n", mt_error_text(error));
        mt_print_numbered(msl);
    } else {
        function = PVK_MSG(id, (id, SEL, id))(library, sel_registerName("newFunctionWithName:"), mt_string(name));
        if (!function)
            fprintf(stderr, "function %s not found in the library\n", name);
        pvk_release(library);
    }
    objc_autoreleasePoolPop(pool);
    return function;
}

/* A render target texture (also readable and writable by shaders). Shared
 * storage when single-sampled, so mt_download can read it. */
static inline id mr_target(pvk_MTLTextureType type, pvk_MTLPixelFormat format, unsigned long w, unsigned long h,
                           unsigned long layers, unsigned long samples)
{
    id desc = PVK_MSG(id, (id, SEL))((id)objc_getClass("MTLTextureDescriptor"), sel_registerName("new")), tex;
    bool depth = format == PVK_MTLPixelFormatDepth32Float || format == PVK_MTLPixelFormatDepth16Unorm ||
                 format == PVK_MTLPixelFormatDepth32Float_Stencil8;
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setTextureType:"), type);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setPixelFormat:"), format);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setWidth:"), w);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setHeight:"), h);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setArrayLength:"), layers);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setSampleCount:"), samples);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setStorageMode:"),
                                            samples > 1 || depth ? PVK_MTLStorageModePrivate : PVK_MTLStorageModeShared);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setUsage:"),
                                            PVK_MTLTextureUsageShaderRead | PVK_MTLTextureUsageRenderTarget |
                                                (samples > 1 || depth ? 0 : PVK_MTLTextureUsageShaderWrite));
    tex = PVK_MSG(id, (id, SEL, id))(mt_device, sel_registerName("newTextureWithDescriptor:"), desc);
    pvk_release(desc);
    return tex;
}

struct mr_blend {
    bool enable;
    pvk_MTLBlendFactor src_rgb, dst_rgb, src_alpha, dst_alpha;
    pvk_MTLColorWriteMask write_mask; /* 0: all */
};

struct mr_pipeline_desc {
    id vertex, fragment;
    pvk_MTLPixelFormat colors[8];
    uint32_t ncolors;
    pvk_MTLPixelFormat depth;
    unsigned long samples;                       /* 0: 1 */
    struct mr_blend blend[8];
    pvk_MTLPrimitiveTopologyClass topology;      /* needed for layered rendering */
    unsigned long amplification;                 /* maxVertexAmplificationCount, 0: none */
};

/* A render pipeline state (retained), NULL after printing the error. */
static inline id mr_pipeline(const struct mr_pipeline_desc *d)
{
    void *pool = objc_autoreleasePoolPush();
    id desc = PVK_MSG(id, (id, SEL))((id)objc_getClass("MTLRenderPipelineDescriptor"), sel_registerName("new"));
    id atts = PVK_MSG(id, (id, SEL))(desc, sel_registerName("colorAttachments")), pso, error = NULL;
    uint32_t i;

    PVK_MSG(void, (id, SEL, id))(desc, sel_registerName("setVertexFunction:"), d->vertex);
    PVK_MSG(void, (id, SEL, id))(desc, sel_registerName("setFragmentFunction:"), d->fragment);
    for (i = 0; i < d->ncolors; i++) {
        id att = PVK_MSG(id, (id, SEL, unsigned long))(atts, sel_registerName("objectAtIndexedSubscript:"), i);
        const struct mr_blend *b = &d->blend[i];
        PVK_MSG(void, (id, SEL, unsigned long))(att, sel_registerName("setPixelFormat:"), d->colors[i]);
        if (b->write_mask)
            PVK_MSG(void, (id, SEL, unsigned long))(att, sel_registerName("setWriteMask:"), b->write_mask);
        if (!b->enable)
            continue;
        PVK_MSG(void, (id, SEL, bool))(att, sel_registerName("setBlendingEnabled:"), true);
        PVK_MSG(void, (id, SEL, unsigned long))(att, sel_registerName("setSourceRGBBlendFactor:"), b->src_rgb);
        PVK_MSG(void, (id, SEL, unsigned long))(att, sel_registerName("setDestinationRGBBlendFactor:"), b->dst_rgb);
        PVK_MSG(void, (id, SEL, unsigned long))(att, sel_registerName("setSourceAlphaBlendFactor:"), b->src_alpha);
        PVK_MSG(void, (id, SEL, unsigned long))(att, sel_registerName("setDestinationAlphaBlendFactor:"), b->dst_alpha);
    }
    if (d->depth)
        PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setDepthAttachmentPixelFormat:"), d->depth);
    PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setRasterSampleCount:"), d->samples ? d->samples : 1);
    if (d->topology)
        PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setInputPrimitiveTopology:"), d->topology);
    if (d->amplification)
        PVK_MSG(void, (id, SEL, unsigned long))(desc, sel_registerName("setMaxVertexAmplificationCount:"), d->amplification);
    pso = PVK_MSG(id, (id, SEL, id, id *))(mt_device, sel_registerName("newRenderPipelineStateWithDescriptor:error:"), desc,
                                           &error);
    if (!pso)
        fprintf(stderr, "render pipeline creation failed: %s\n", mt_error_text(error));
    pvk_release(desc);
    objc_autoreleasePoolPop(pool);
    return pso;
}

/* Translates both stages (printing errors), compiles them and builds the
 * pipeline described by d (its vertex/fragment are filled in). On success
 * *pso is retained and vs/fs must be freed. */
static inline bool mr_build(const uint32_t *vwords, size_t vcount, const struct pvk_msl_options *vopt,
                            const uint32_t *fwords, size_t fcount, const struct pvk_msl_options *fopt,
                            struct mr_pipeline_desc *d, struct pvk_msl_result *vs, struct pvk_msl_result *fs, id *pso)
{
    VkResult r = pvk_spirv_to_msl_graphics(vwords, vcount, NULL, vopt, fwords, fcount, NULL, fopt, vs, fs);
    *pso = NULL;
    if (r != VK_SUCCESS) {
        fprintf(stderr, "translation failed (%d): %s%s\n", (int)r, vs->error, fs->error);
        pvk_msl_result_free(vs);
        pvk_msl_result_free(fs);
        return false;
    }
    d->vertex = mr_function(vs->msl, vs->entry_point);
    d->fragment = fcount ? mr_function(fs->msl, fs->entry_point) : NULL;
    if (d->vertex && (d->fragment || !fcount))
        *pso = mr_pipeline(d);
    if (!*pso) {
        fprintf(stderr, "vertex MSL:\n");
        mt_print_numbered(vs->msl);
        if (fcount) {
            fprintf(stderr, "fragment MSL:\n");
            mt_print_numbered(fs->msl);
        }
    }
    pvk_release(d->vertex);
    pvk_release(d->fragment);
    d->vertex = d->fragment = NULL;
    if (!*pso) {
        pvk_msl_result_free(vs);
        pvk_msl_result_free(fs);
    }
    return *pso != NULL;
}

struct mr_pass {
    id colors[8];
    uint32_t ncolors;
    float clear[8][4];
    bool load[8];      /* keep the contents instead of clearing */
    id resolve[8];     /* multisample resolve targets */
    id depth;
    double clear_depth;
    unsigned long layers; /* renderTargetArrayLength (layered rendering) */
};

struct mr_vertex_buffer {
    struct mt_buffer *buf;
    uint64_t offset, size; /* size 0: to the end */
    uint32_t stride;       /* used by dynamic strides */
};

struct mr_draw {
    id pipeline;
    const struct pvk_msl_result *vs, *fs;
    const struct mt_binding *bindings; /* set contents, applied to each stage's reflection */
    uint32_t nbindings;
    const void *push;
    size_t push_size;
    struct mr_vertex_buffer vb[16];   /* by binding */
    pvk_MTLViewport viewports[16];    /* none: the whole first target */
    uint32_t nviewports;
    pvk_MTLPrimitiveType primitive;
    uint32_t vertex_start, vertex_count, instance_count, base_instance;
    struct mt_buffer *index_buffer;   /* indexed draw */
    pvk_MTLIndexType index_type;
    uint64_t index_offset;
    int32_t base_vertex;
    uint32_t amplification;           /* vertex amplification count, 0: none */
    pvk_MTLVertexAmplificationViewMapping views[8];
    pvk_MTLCullMode cull;
    pvk_MTLWinding winding;
    pvk_MTLCompareFunction depth_compare; /* 0 (never): no depth state */
    bool depth_write;
    float blend_color[4];
    uint32_t draw_index;              /* DrawIndex (bound at PVK_MSL_DRAW_INDEX_BUFFER) */
};

static inline void mr_bind_stage(id enc, const struct pvk_msl_result *res, const struct mr_draw *d, bool fragment,
                                 struct mt_buffer sets[PVK_MSL_MAX_SETS])
{
    const char *buffer_sel = fragment ? "setFragmentBuffer:offset:atIndex:" : "setVertexBuffer:offset:atIndex:";
    const char *bytes_sel = fragment ? "setFragmentBytes:length:atIndex:" : "setVertexBytes:length:atIndex:";
    uint32_t s;
    mt_fill_sets(res, d->bindings, d->nbindings, sets);
    for (s = 0; s < PVK_MSL_MAX_SETS; s++)
        if (sets[s].buffer)
            PVK_MSG(void, (id, SEL, id, unsigned long, unsigned long))(enc, sel_registerName(buffer_sel), sets[s].buffer, 0,
                                                                       s);
    if (d->push && d->push_size)
        PVK_MSG(void, (id, SEL, const void *, unsigned long, unsigned long))(enc, sel_registerName(bytes_sel), d->push,
                                                                             d->push_size, res->push_constant_buffer);
}

#define MR_MAX_DRAWS 16

/* One render pass with the given draws (at most MR_MAX_DRAWS), waited for. */
static inline bool mr_render(const struct mr_pass *pass, const struct mr_draw *draws, uint32_t ndraws)
{
    void *pool;
    id rp, atts, cb, enc;
    struct mt_buffer sets[2 * PVK_MSL_MAX_SETS * MR_MAX_DRAWS];
    uint32_t i, k, k2, nsets = 0;
    unsigned long status, w = 0, h = 0;
    bool ok = true;

    if (ndraws > MR_MAX_DRAWS)
        return false;
    pool = objc_autoreleasePoolPush();
    rp = PVK_MSG(id, (id, SEL))((id)objc_getClass("MTLRenderPassDescriptor"), sel_registerName("renderPassDescriptor"));
    atts = PVK_MSG(id, (id, SEL))(rp, sel_registerName("colorAttachments"));
    memset(sets, 0, sizeof(sets));
    for (i = 0; i < pass->ncolors; i++) {
        id att = PVK_MSG(id, (id, SEL, unsigned long))(atts, sel_registerName("objectAtIndexedSubscript:"), i);
        PVK_MSG(void, (id, SEL, id))(att, sel_registerName("setTexture:"), pass->colors[i]);
        PVK_MSG(void, (id, SEL, unsigned long))(att, sel_registerName("setLoadAction:"),
                                                pass->load[i] ? PVK_MTLLoadActionLoad : PVK_MTLLoadActionClear);
        PVK_MSG(void, (id, SEL, double, double, double, double))(att, sel_registerName("setClearColor:"), pass->clear[i][0],
                                                                 pass->clear[i][1], pass->clear[i][2], pass->clear[i][3]);
        if (pass->resolve[i]) {
            PVK_MSG(void, (id, SEL, id))(att, sel_registerName("setResolveTexture:"), pass->resolve[i]);
            PVK_MSG(void, (id, SEL, unsigned long))(att, sel_registerName("setStoreAction:"),
                                                    PVK_MTLStoreActionStoreAndMultisampleResolve);
        } else {
            PVK_MSG(void, (id, SEL, unsigned long))(att, sel_registerName("setStoreAction:"), PVK_MTLStoreActionStore);
        }
        if (!w) {
            w = PVK_MSG(unsigned long, (id, SEL))(pass->colors[i], sel_registerName("width"));
            h = PVK_MSG(unsigned long, (id, SEL))(pass->colors[i], sel_registerName("height"));
        }
    }
    if (pass->depth) {
        id att = PVK_MSG(id, (id, SEL))(rp, sel_registerName("depthAttachment"));
        PVK_MSG(void, (id, SEL, id))(att, sel_registerName("setTexture:"), pass->depth);
        PVK_MSG(void, (id, SEL, unsigned long))(att, sel_registerName("setLoadAction:"), PVK_MTLLoadActionClear);
        PVK_MSG(void, (id, SEL, unsigned long))(att, sel_registerName("setStoreAction:"), PVK_MTLStoreActionStore);
        PVK_MSG(void, (id, SEL, double))(att, sel_registerName("setClearDepth:"), pass->clear_depth);
        if (!w) {
            w = PVK_MSG(unsigned long, (id, SEL))(pass->depth, sel_registerName("width"));
            h = PVK_MSG(unsigned long, (id, SEL))(pass->depth, sel_registerName("height"));
        }
    }
    if (pass->layers)
        PVK_MSG(void, (id, SEL, unsigned long))(rp, sel_registerName("setRenderTargetArrayLength:"), pass->layers);

    cb = PVK_MSG(id, (id, SEL))(mt_queue, sel_registerName("commandBuffer"));
    enc = PVK_MSG(id, (id, SEL, id))(cb, sel_registerName("renderCommandEncoderWithDescriptor:"), rp);
    for (k = 0; k < ndraws; k++) {
        const struct mr_draw *d = &draws[k];
        struct pvk_msl_vertex_buffer table[16];
        pvk_MTLViewport full;
        PVK_MSG(void, (id, SEL, id))(enc, sel_registerName("setRenderPipelineState:"), d->pipeline);
        if (d->nviewports) {
            PVK_MSG(void, (id, SEL, const pvk_MTLViewport *, unsigned long))(enc, sel_registerName("setViewports:count:"),
                                                                            d->viewports, d->nviewports);
        } else {
            full.originX = full.originY = 0;
            full.width = (double)w;
            full.height = (double)h;
            full.znear = 0;
            full.zfar = 1;
            PVK_MSG(void, (id, SEL, pvk_MTLViewport))(enc, sel_registerName("setViewport:"), full);
        }
        PVK_MSG(void, (id, SEL, unsigned long))(enc, sel_registerName("setCullMode:"), d->cull);
        PVK_MSG(void, (id, SEL, unsigned long))(enc, sel_registerName("setFrontFacingWinding:"), d->winding);
        PVK_MSG(void, (id, SEL, float, float, float, float))(enc, sel_registerName("setBlendColorRed:green:blue:alpha:"),
                                                             d->blend_color[0], d->blend_color[1], d->blend_color[2],
                                                             d->blend_color[3]);
        if (d->depth_compare) {
            id dd = PVK_MSG(id, (id, SEL))((id)objc_getClass("MTLDepthStencilDescriptor"), sel_registerName("new")), ds;
            PVK_MSG(void, (id, SEL, unsigned long))(dd, sel_registerName("setDepthCompareFunction:"), d->depth_compare);
            PVK_MSG(void, (id, SEL, bool))(dd, sel_registerName("setDepthWriteEnabled:"), d->depth_write);
            ds = PVK_MSG(id, (id, SEL, id))(mt_device, sel_registerName("newDepthStencilStateWithDescriptor:"), dd);
            PVK_MSG(void, (id, SEL, id))(enc, sel_registerName("setDepthStencilState:"), ds);
            pvk_release(ds);
            pvk_release(dd);
        }
        if (d->vs) {
            mr_bind_stage(enc, d->vs, d, false, &sets[nsets]);
            nsets += PVK_MSL_MAX_SETS;
            memset(table, 0, sizeof(table));
            for (i = 0; i < 16; i++) {
                const struct mr_vertex_buffer *v = &d->vb[i];
                if (!v->buf)
                    continue;
                table[i].address = v->buf->address + v->offset;
                table[i].size = v->size ? v->size : v->buf->size - v->offset;
                table[i].stride = v->stride;
                PVK_MSG(void, (id, SEL, id, unsigned long, unsigned long))(enc, sel_registerName("useResource:usage:stages:"),
                                                                           v->buf->buffer, PVK_MTLResourceUsageRead,
                                                                           PVK_MTLRenderStageVertex);
            }
            if (d->vs->vertex_input_count)
                PVK_MSG(void, (id, SEL, const void *, unsigned long, unsigned long))(
                    enc, sel_registerName("setVertexBytes:length:atIndex:"), table, sizeof(table),
                    d->vs->vertex_table_buffer);
            if (d->vs->builtins & PVK_MSL_BUILTIN_DRAW_INDEX)
                PVK_MSG(void, (id, SEL, const void *, unsigned long, unsigned long))(
                    enc, sel_registerName("setVertexBytes:length:atIndex:"), &d->draw_index, sizeof(d->draw_index),
                    PVK_MSL_DRAW_INDEX_BUFFER);
        }
        if (d->fs) {
            mr_bind_stage(enc, d->fs, d, true, &sets[nsets]);
            nsets += PVK_MSL_MAX_SETS;
        }
        for (i = 0; i < d->nbindings; i++) {
            unsigned long stages = PVK_MTLRenderStageVertex | PVK_MTLRenderStageFragment;
            if (d->bindings[i].buf)
                PVK_MSG(void, (id, SEL, id, unsigned long, unsigned long))(enc, sel_registerName("useResource:usage:stages:"),
                                                                           d->bindings[i].buf->buffer,
                                                                           PVK_MTLResourceUsageRead | PVK_MTLResourceUsageWrite,
                                                                           stages);
            if (d->bindings[i].texture)
                PVK_MSG(void, (id, SEL, id, unsigned long, unsigned long))(enc, sel_registerName("useResource:usage:stages:"),
                                                                           d->bindings[i].texture,
                                                                           PVK_MTLResourceUsageRead | PVK_MTLResourceUsageWrite,
                                                                           stages);
            for (k2 = 0; k2 < 2; k2++) /* Y'CbCr planes */
                if (d->bindings[i].planes[k2])
                    PVK_MSG(void, (id, SEL, id, unsigned long, unsigned long))(
                        enc, sel_registerName("useResource:usage:stages:"), d->bindings[i].planes[k2],
                        PVK_MTLResourceUsageRead, stages);
        }
        if (d->amplification)
            PVK_MSG(void, (id, SEL, unsigned long, const pvk_MTLVertexAmplificationViewMapping *))(
                enc, sel_registerName("setVertexAmplificationCount:viewMappings:"), d->amplification, d->views);
        if (d->index_buffer)
            PVK_MSG(void, (id, SEL, unsigned long, unsigned long, unsigned long, id, unsigned long, unsigned long, long,
                           unsigned long))(
                enc, sel_registerName("drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:"
                                      "baseVertex:baseInstance:"),
                d->primitive, d->vertex_count, d->index_type, d->index_buffer->buffer, d->index_offset,
                d->instance_count ? d->instance_count : 1, d->base_vertex, d->base_instance);
        else
            PVK_MSG(void, (id, SEL, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long))(
                enc, sel_registerName("drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:"), d->primitive,
                d->vertex_start, d->vertex_count, d->instance_count ? d->instance_count : 1, d->base_instance);
    }
    PVK_MSG(void, (id, SEL))(enc, sel_registerName("endEncoding"));
    PVK_MSG(void, (id, SEL))(cb, sel_registerName("commit"));
    PVK_MSG(void, (id, SEL))(cb, sel_registerName("waitUntilCompleted"));
    status = PVK_MSG(unsigned long, (id, SEL))(cb, sel_registerName("status"));
    if (status != PVK_MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "command buffer failed: %s\n", mt_error_text(PVK_MSG(id, (id, SEL))(cb, sel_registerName("error"))));
        ok = false;
    }
    objc_autoreleasePoolPop(pool);
    for (i = 0; i < nsets; i++)
        mt_buffer_free(&sets[i]);
    return ok;
}

/* Reads every sample of a multisampled float color texture (w x h x n) into
 * out[(y * w + x) * n + s] (RGBA). */
static inline bool mr_read_samples(id tex, unsigned long w, unsigned long h, unsigned long n, float *out)
{
    static const char msl[] =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void read_ms(texture2d_ms<float> t [[texture(0)]], device float4* o [[buffer(0)]],\n"
        "                    constant uint2& wn [[buffer(1)]], uint2 p [[thread_position_in_grid]])\n"
        "{\n"
        "    if (p.x >= t.get_width() || p.y >= t.get_height()) return;\n"
        "    for (uint s = 0; s < wn.y; s++) o[(p.y * wn.x + p.x) * wn.y + s] = t.read(p, s);\n"
        "}\n";
    void *pool = objc_autoreleasePoolPush();
    id fn = mr_function(msl, "read_ms"), pso, error = NULL, cb, enc;
    struct mt_buffer buf = mt_buffer(w * h * n * 16, NULL, 0);
    uint32_t wn[2];
    pvk_MTLSize groups, threads;
    bool ok = fn != NULL;
    wn[0] = (uint32_t)w;
    wn[1] = (uint32_t)n;
    pso = fn ? PVK_MSG(id, (id, SEL, id, id *))(mt_device, sel_registerName("newComputePipelineStateWithFunction:error:"),
                                                 fn, &error)
             : NULL;
    if (pso) {
        cb = PVK_MSG(id, (id, SEL))(mt_queue, sel_registerName("commandBuffer"));
        enc = PVK_MSG(id, (id, SEL))(cb, sel_registerName("computeCommandEncoder"));
        PVK_MSG(void, (id, SEL, id))(enc, sel_registerName("setComputePipelineState:"), pso);
        PVK_MSG(void, (id, SEL, id, unsigned long))(enc, sel_registerName("setTexture:atIndex:"), tex, 0);
        PVK_MSG(void, (id, SEL, id, unsigned long, unsigned long))(enc, sel_registerName("setBuffer:offset:atIndex:"),
                                                                   buf.buffer, 0, 0);
        PVK_MSG(void, (id, SEL, const void *, unsigned long, unsigned long))(enc, sel_registerName("setBytes:length:atIndex:"),
                                                                             wn, sizeof(wn), 1);
        groups.width = (w + 7) / 8;
        groups.height = (h + 7) / 8;
        groups.depth = 1;
        threads.width = threads.height = 8;
        threads.depth = 1;
        PVK_MSG(void, (id, SEL, pvk_MTLSize, pvk_MTLSize))(enc, sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"),
                                                           groups, threads);
        PVK_MSG(void, (id, SEL))(enc, sel_registerName("endEncoding"));
        PVK_MSG(void, (id, SEL))(cb, sel_registerName("commit"));
        PVK_MSG(void, (id, SEL))(cb, sel_registerName("waitUntilCompleted"));
        memcpy(out, buf.data, w * h * n * 16);
    } else {
        ok = false;
    }
    pvk_release(pso);
    pvk_release(fn);
    mt_buffer_free(&buf);
    objc_autoreleasePoolPop(pool);
    return ok;
}

#endif /* PVK_MSL_RENDER_H */
