/* Descriptor churn: every draw gets its own descriptors - one uniform buffer
 * (the object's rectangle and color) and four material textures - delivered
 * in each of the ways an engine might:
 *
 *   bind_static      pre-built set per object; vkCmdBindDescriptorSets only
 *   update_sets      per frame vkResetDescriptorPool; per draw
 *                    vkAllocateDescriptorSets + vkUpdateDescriptorSets (5
 *                    writes) + vkCmdBindDescriptorSets
 *   update_template  the same with vkUpdateDescriptorSetWithTemplate
 *   dynamic_offset   one set per material, uniform buffer bound with a
 *                    per-draw dynamic offset
 *   push_descriptor  vkCmdPushDescriptorSet (core in 1.4)
 *   push_template    vkCmdPushDescriptorSetWithTemplate
 *   multi_frequency  sets by update frequency, as engines lay them out: set 0
 *                    per frame (a uniform buffer the CPU rewrites), set 1 per
 *                    material (bound when the material changes, every 8
 *                    draws), set 2 per object (one set, bound every draw
 *                    with a dynamic offset)
 *   update_sets_mt4  update_sets recorded by 4 threads, each into its own
 *                    secondary command buffer from its own descriptor and
 *                    command pools (sharing one set layout); the frame
 *                    executes the 4 secondaries
 *
 * The first frame of each mode is read back and checked against the CPU.
 *
 * The pool_ and array_ configurations time descriptor management on the CPU
 * alone, outside any frame, and then draw with the sets they leave to check
 * them:
 *
 *   pool_free_churn  a FREE_DESCRIPTOR_SET pool kept full with 4096 sets:
 *                    free a random one, allocate a replacement (streamed
 *                    textures and materials, per-texture sets, UI images)
 *   pool_alloc_batch one vkAllocateDescriptorSets of 256 sets, then
 *                    vkResetDescriptorPool
 *   pool_create      vkCreateDescriptorPool (256 sets) + vkDestroyDescriptorPool,
 *                    as allocators that grow by adding pools do
 *   array_write      one-element writes at random indices of a bindless
 *                    texture table (update after bind, partially bound) of
 *                    65536 textures, or 1024 (within the per-set limits of
 *                    more implementations); skipped where the device does
 *                    not support the table
 *   pool_threads_N   N threads, each with its own pool, allocate and write
 *                    256 sets of one shared layout, then reset the pool, as
 *                    the per-thread transient pools of a job system do
 *
 * The _named variants first give one object (the sampler) a name with
 * VK_EXT_debug_utils, as development builds of engines name everything.
 * They are skipped when the instance does not have the extension enabled.
 */
#include <pthread.h>

#include "bench_vk.h"
#include "bench_shaders.h"

#define WIDTH 1024
#define HEIGHT 1024
#define OBJECTS 10000
#define TEXTURES 16
#define TEXTURE_SIZE 4
#define MATERIALS 16 /* object i uses material i % MATERIALS */
#define SLOTS 4      /* textures per material */
#define DRAWS_PER_MATERIAL 8 /* multi_frequency: object i uses material (i / 8) % MATERIALS */
#define MAX_THREADS 8
#define RECORD_THREADS 4 /* update_sets_mt */
#define THREAD_SETS 256  /* pool_threads: sets per pool cycle */
#define THREAD_CYCLES 16 /* pool_threads: cycles per thread per round */
#define CHECK_ELEMENTS 64 /* array_write: objects drawn with textures copied from the array */

enum mode {
    MODE_BIND_STATIC, MODE_UPDATE, MODE_UPDATE_TEMPLATE, MODE_DYNAMIC_OFFSET, MODE_PUSH, MODE_PUSH_TEMPLATE,
    MODE_MULTI_FREQUENCY, MODE_UPDATE_THREADS,
    /* CPU only, then a check frame (MODE_CHECK) */
    MODE_POOL_FREE_CHURN, MODE_POOL_ALLOC_BATCH, MODE_POOL_CREATE, MODE_ARRAY_WRITE, MODE_POOL_THREADS,
    MODE_CHECK /* draws object i with check_sets[i] */
};

struct config {
    const char *name;
    enum mode mode;
    int draws;   /* CPU only: the sets (array elements) the pool (array) holds */
    int frames;  /* CPU only: measured rounds */
    int threads; /* recording or allocating threads */
    int named;   /* an object has a VK_EXT_debug_utils name */
};

static const struct config configs[] = {
    { "bind_static_10000", MODE_BIND_STATIC, OBJECTS, 100, 0, 0 },
    { "update_sets_10000", MODE_UPDATE, OBJECTS, 100, 0, 0 },
    { "update_template_10000", MODE_UPDATE_TEMPLATE, OBJECTS, 100, 0, 0 },
    { "dynamic_offset_10000", MODE_DYNAMIC_OFFSET, OBJECTS, 100, 0, 0 },
    { "push_descriptor_10000", MODE_PUSH, OBJECTS, 100, 0, 0 },
    { "push_template_10000", MODE_PUSH_TEMPLATE, OBJECTS, 100, 0, 0 },
    { "multi_frequency_10000", MODE_MULTI_FREQUENCY, OBJECTS, 100, 0, 0 },
    { "update_sets_mt4_10000", MODE_UPDATE_THREADS, OBJECTS, 100, RECORD_THREADS, 0 },
    { "update_sets_mt4_named_10000", MODE_UPDATE_THREADS, OBJECTS, 100, RECORD_THREADS, 1 },
    { "pool_free_churn_4096", MODE_POOL_FREE_CHURN, 4096, 200, 0, 0 },
    { "pool_alloc_batch_256", MODE_POOL_ALLOC_BATCH, 256, 200, 0, 0 },
    { "pool_create_256", MODE_POOL_CREATE, 256, 100, 0, 0 },
    { "array_write_1024", MODE_ARRAY_WRITE, 1024, 200, 0, 0 },
    { "array_write_65536", MODE_ARRAY_WRITE, 65536, 200, 0, 0 },
    { "pool_threads_1", MODE_POOL_THREADS, THREAD_SETS, 20, 1, 0 },
    { "pool_threads_2", MODE_POOL_THREADS, THREAD_SETS, 20, 2, 0 },
    { "pool_threads_4", MODE_POOL_THREADS, THREAD_SETS, 20, 4, 0 },
    { "pool_threads_8", MODE_POOL_THREADS, THREAD_SETS, 20, 8, 0 },
    { "pool_threads_8_named", MODE_POOL_THREADS, THREAD_SETS, 20, 8, 1 },
};

struct object_ubo {
    float rect[4];
    float color[4];
};

/* What the update templates read. */
struct descriptors {
    VkDescriptorBufferInfo ubo;
    VkDescriptorImageInfo images[SLOTS];
};

/* multi_frequency's per-frame data (set 0). */
static const float frame_tint[4] = { 1.0f, 0.5f, 0.75f, 1.0f };

/* Worker threads that run one function together, as a job system's do. */
struct scene;
typedef void (*team_fn)(struct scene *s, int thread);

struct team_thread {
    struct team *team;
    int index;
    pthread_t thread;
};

struct team {
    struct scene *s;
    pthread_mutex_t lock;
    pthread_cond_t start, done;
    struct team_thread threads[MAX_THREADS];
    int count;
    unsigned round; /* bumped to start every thread on fn */
    int busy;       /* threads still in the round */
    int quit;
    team_fn fn;
};

struct scene {
    struct bench_vk *vk;
    const struct config *cfg;
    struct bench_rect rects[OBJECTS];
    uint32_t texture_rgba[TEXTURES];
    VkDeviceSize ubo_stride, frame_stride;
    struct bench_buffer ubo, readback, frame_ubo;
    struct bench_image targets[BENCH_FRAMES_IN_FLIGHT];
    struct bench_image textures[TEXTURES];
    VkSampler sampler;
    VkDescriptorSetLayout set_layout, dynamic_set_layout, push_set_layout;
    VkPipelineLayout layout, dynamic_layout, push_layout;
    VkPipeline pipeline, dynamic_pipeline, push_pipeline;
    VkDescriptorPool static_pool, frame_pools[BENCH_FRAMES_IN_FLIGHT];
    VkDescriptorSet static_sets[OBJECTS], material_sets[MATERIALS];
    VkDescriptorUpdateTemplate update_template, push_template;
    /* multi_frequency: set 0 per frame in flight, set 1 per material, set 2 */
    VkDescriptorSetLayout mf_set_layouts[3];
    VkPipelineLayout mf_layout;
    VkPipeline mf_pipeline;
    VkDescriptorPool mf_pool;
    VkDescriptorSet mf_frame_sets[BENCH_FRAMES_IN_FLIGHT], mf_material_sets[MATERIALS], mf_object_set;
    /* update_sets_mt: per thread and frame in flight */
    struct team team;
    VkDescriptorPool thread_pools[RECORD_THREADS][BENCH_FRAMES_IN_FLIGHT];
    VkCommandPool thread_cmd_pools[RECORD_THREADS][BENCH_FRAMES_IN_FLIGHT];
    VkCommandBuffer thread_cmds[RECORD_THREADS][BENCH_FRAMES_IN_FLIGHT];
    uint32_t slot; /* the frame being recorded */
    /* CPU-only configurations: per thread pools, and the sets the check frame draws */
    VkDescriptorPool cpu_pools[MAX_THREADS];
    VkDescriptorSet check_sets[OBJECTS];
    VkPipelineLayout check_layout; /* what the check frame draws with: layout and pipeline, or */
    VkPipeline check_pipeline;     /* (array_write) their update-after-bind twins */
};

/* Not in bench_vk.h's table. */
static PFN_vkSetDebugUtilsObjectNameEXT SetDebugUtilsObjectName; /* NULL: VK_EXT_debug_utils not enabled */

static int material_of(const struct config *cfg, int object)
{
    return cfg->mode == MODE_MULTI_FREQUENCY ? (object / DRAWS_PER_MATERIAL) % MATERIALS : object % MATERIALS;
}

static int texture_of(int material, int slot)
{
    return (material + slot * 5) % TEXTURES;
}

/* array_write's table: n textures, written while bound, with only the
 * elements in use valid, as engines make bindless tables. */
struct table_layout {
    VkDescriptorBindingFlags flags;
    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags;
    VkDescriptorSetLayoutBinding binding;
    VkDescriptorSetLayoutCreateInfo info;
};

static void table_layout(struct table_layout *t, uint32_t n)
{
    memset(t, 0, sizeof(*t));
    t->flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    t->binding_flags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    t->binding_flags.bindingCount = 1;
    t->binding_flags.pBindingFlags = &t->flags;
    t->binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    t->binding.descriptorCount = n;
    t->binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    t->info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    t->info.pNext = &t->binding_flags;
    t->info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    t->info.bindingCount = 1;
    t->info.pBindings = &t->binding;
}

/* An object's set: its uniform buffer, then its material's textures. */
static void object_bindings(VkDescriptorSetLayoutBinding bindings[1 + SLOTS])
{
    int k;
    for (k = 0; k < 1 + SLOTS; k++) {
        bindings[k].binding = (uint32_t)k;
        bindings[k].descriptorType = k == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[k].descriptorCount = 1;
        bindings[k].stageFlags = k == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[k].pImmutableSamplers = NULL;
    }
}

static void object_descriptors(struct scene *s, int i, struct descriptors *d)
{
    int k;
    d->ubo.buffer = s->ubo.buffer;
    d->ubo.offset = (VkDeviceSize)i * s->ubo_stride;
    d->ubo.range = sizeof(struct object_ubo);
    for (k = 0; k < SLOTS; k++) {
        d->images[k].sampler = s->sampler;
        d->images[k].imageView = s->textures[texture_of(i % MATERIALS, k)].view;
        d->images[k].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
}

/* The five writes a straightforward engine issues per draw. */
static void object_writes(const struct descriptors *d, VkDescriptorSet set, VkDescriptorType ubo_type,
                          VkWriteDescriptorSet writes[1 + SLOTS])
{
    int k;
    memset(writes, 0, sizeof(VkWriteDescriptorSet) * (1 + SLOTS));
    for (k = 0; k < 1 + SLOTS; k++) {
        writes[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[k].dstSet = set;
        writes[k].dstBinding = (uint32_t)k;
        writes[k].descriptorCount = 1;
        writes[k].descriptorType = k == 0 ? ubo_type : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        if (k == 0)
            writes[k].pBufferInfo = &d->ubo;
        else
            writes[k].pImageInfo = &d->images[k - 1];
    }
}

/* Writes object i's descriptors into set. */
static void write_object(struct scene *s, VkDescriptorSet set, int i)
{
    VkWriteDescriptorSet writes[1 + SLOTS];
    struct descriptors d;
    object_descriptors(s, i, &d);
    object_writes(&d, set, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, writes);
    vkd.UpdateDescriptorSets(s->vk->device, 1 + SLOTS, writes, 0, NULL);
}

/* ==========================================================================
 * Threads
 * ========================================================================== */

static void *team_main(void *arg)
{
    struct team_thread *self = arg;
    struct team *t = self->team;
    unsigned seen = 0;

    for (;;) {
        team_fn fn;
        int quit;
        pthread_mutex_lock(&t->lock);
        while (t->round == seen && !t->quit)
            pthread_cond_wait(&t->start, &t->lock);
        seen = t->round;
        fn = t->fn;
        quit = t->quit;
        pthread_mutex_unlock(&t->lock);
        if (quit)
            return NULL;
        fn(t->s, self->index);
        pthread_mutex_lock(&t->lock);
        if (--t->busy == 0)
            pthread_cond_signal(&t->done);
        pthread_mutex_unlock(&t->lock);
    }
}

static void team_start(struct team *t, struct scene *s, int count)
{
    int i;
    memset(t, 0, sizeof(*t));
    t->s = s;
    t->count = count;
    pthread_mutex_init(&t->lock, NULL);
    pthread_cond_init(&t->start, NULL);
    pthread_cond_init(&t->done, NULL);
    for (i = 0; i < count; i++) {
        t->threads[i].team = t;
        t->threads[i].index = i;
        if (pthread_create(&t->threads[i].thread, NULL, team_main, &t->threads[i]) != 0)
            bench_fatal("pthread_create failed");
    }
}

/* Runs fn(s, i) on every thread i and waits for all of them. */
static void team_run(struct team *t, team_fn fn)
{
    pthread_mutex_lock(&t->lock);
    t->fn = fn;
    t->busy = t->count;
    t->round++;
    pthread_cond_broadcast(&t->start);
    while (t->busy)
        pthread_cond_wait(&t->done, &t->lock);
    pthread_mutex_unlock(&t->lock);
}

static void team_stop(struct team *t)
{
    int i;
    pthread_mutex_lock(&t->lock);
    t->quit = 1;
    pthread_cond_broadcast(&t->start);
    pthread_mutex_unlock(&t->lock);
    for (i = 0; i < t->count; i++)
        pthread_join(t->threads[i].thread, NULL);
    pthread_mutex_destroy(&t->lock);
    pthread_cond_destroy(&t->start);
    pthread_cond_destroy(&t->done);
}

/* ==========================================================================
 * Frames
 * ========================================================================== */

/* update_sets_mt: thread t records its share of the objects. */
static void record_thread(struct scene *s, int t)
{
    struct bench_vk *vk = s->vk;
    VkPipelineBindPoint gfx = VK_PIPELINE_BIND_POINT_GRAPHICS;
    VkCommandBuffer cmd = s->thread_cmds[t][s->slot];
    VkDescriptorPool pool = s->thread_pools[t][s->slot];
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkCommandBufferInheritanceRenderingInfo rendering = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &format,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    VkCommandBufferInheritanceInfo inheritance = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
        .pNext = &rendering,
    };
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
        .pInheritanceInfo = &inheritance,
    };
    VkDescriptorSetAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &s->set_layout,
    };
    int n = s->cfg->draws, threads = s->cfg->threads;
    int i = (int)((long long)n * t / threads), end = (int)((long long)n * (t + 1) / threads);

    BENCH_VK(vkd.ResetDescriptorPool(vk->device, pool, 0));
    BENCH_VK(vkd.ResetCommandPool(vk->device, s->thread_cmd_pools[t][s->slot], 0));
    BENCH_VK(vkd.BeginCommandBuffer(cmd, &begin));
    bench_cmd_viewport(cmd, WIDTH, HEIGHT);
    vkd.CmdBindPipeline(cmd, gfx, s->pipeline);
    for (; i < end; i++) {
        VkDescriptorSet set;
        BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &alloc, &set));
        write_object(s, set, i);
        vkd.CmdBindDescriptorSets(cmd, gfx, s->layout, 0, 1, &set, 0, NULL);
        vkd.CmdDraw(cmd, 6, 1, 0, 0);
    }
    BENCH_VK(vkd.EndCommandBuffer(cmd));
}

static void record(void *user, const struct bench_frame *f)
{
    struct scene *s = user;
    struct bench_vk *vk = s->vk;
    const float clear[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    VkCommandBuffer cmd = f->cmd;
    VkPipelineBindPoint gfx = VK_PIPELINE_BIND_POINT_GRAPHICS;
    VkDescriptorSetAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s->frame_pools[f->slot],
        .descriptorSetCount = 1,
        .pSetLayouts = &s->set_layout,
    };
    VkWriteDescriptorSet writes[1 + SLOTS];
    struct descriptors d;
    VkDescriptorSet set;
    int i, n = s->cfg->draws;

    if (s->cfg->mode == MODE_UPDATE_THREADS) {
        /* The threads record the secondaries the render pass instance runs. */
        const struct bench_image *target = &s->targets[f->slot];
        VkRenderingAttachmentInfo color = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = target->view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        };
        VkRenderingInfo info = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .flags = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT,
            .renderArea = { { 0, 0 }, { target->width, target->height } },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color,
        };
        VkCommandBuffer secondaries[RECORD_THREADS];
        s->slot = f->slot;
        team_run(&s->team, record_thread);
        for (i = 0; i < s->cfg->threads; i++)
            secondaries[i] = s->thread_cmds[i][f->slot];
        memcpy(color.clearValue.color.float32, clear, sizeof(float) * 4);
        bench_image_barrier(cmd, target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
                            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        vkd.CmdBeginRendering(cmd, &info);
        vkd.CmdExecuteCommands(cmd, (uint32_t)s->cfg->threads, secondaries);
    } else {
        bench_cmd_begin_color(cmd, &s->targets[f->slot], clear);
    }
    switch (s->cfg->mode) {
    case MODE_BIND_STATIC:
        vkd.CmdBindPipeline(cmd, gfx, s->pipeline);
        for (i = 0; i < n; i++) {
            vkd.CmdBindDescriptorSets(cmd, gfx, s->layout, 0, 1, &s->static_sets[i], 0, NULL);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        break;
    case MODE_UPDATE:
    case MODE_UPDATE_TEMPLATE:
        BENCH_VK(vkd.ResetDescriptorPool(vk->device, s->frame_pools[f->slot], 0));
        vkd.CmdBindPipeline(cmd, gfx, s->pipeline);
        for (i = 0; i < n; i++) {
            BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &alloc, &set));
            object_descriptors(s, i, &d);
            if (s->cfg->mode == MODE_UPDATE) {
                object_writes(&d, set, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, writes);
                vkd.UpdateDescriptorSets(vk->device, 1 + SLOTS, writes, 0, NULL);
            } else {
                vkd.UpdateDescriptorSetWithTemplate(vk->device, set, s->update_template, &d);
            }
            vkd.CmdBindDescriptorSets(cmd, gfx, s->layout, 0, 1, &set, 0, NULL);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        break;
    case MODE_DYNAMIC_OFFSET:
        vkd.CmdBindPipeline(cmd, gfx, s->dynamic_pipeline);
        for (i = 0; i < n; i++) {
            uint32_t offset = (uint32_t)((VkDeviceSize)i * s->ubo_stride);
            vkd.CmdBindDescriptorSets(cmd, gfx, s->dynamic_layout, 0, 1, &s->material_sets[i % MATERIALS], 1,
                                      &offset);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        break;
    case MODE_PUSH:
        vkd.CmdBindPipeline(cmd, gfx, s->push_pipeline);
        for (i = 0; i < n; i++) {
            object_descriptors(s, i, &d);
            object_writes(&d, VK_NULL_HANDLE, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, writes);
            vkd.CmdPushDescriptorSet(cmd, gfx, s->push_layout, 0, 1 + SLOTS, writes);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        break;
    case MODE_PUSH_TEMPLATE:
        vkd.CmdBindPipeline(cmd, gfx, s->push_pipeline);
        for (i = 0; i < n; i++) {
            object_descriptors(s, i, &d);
            vkd.CmdPushDescriptorSetWithTemplate(cmd, s->push_template, s->push_layout, 0, &d);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        break;
    case MODE_MULTI_FREQUENCY:
        memcpy((char *)s->frame_ubo.map + f->slot * s->frame_stride, frame_tint, sizeof(frame_tint));
        vkd.CmdBindPipeline(cmd, gfx, s->mf_pipeline);
        vkd.CmdBindDescriptorSets(cmd, gfx, s->mf_layout, 0, 1, &s->mf_frame_sets[f->slot], 0, NULL);
        for (i = 0; i < n; i++) {
            uint32_t offset = (uint32_t)((VkDeviceSize)i * s->ubo_stride);
            if (i % DRAWS_PER_MATERIAL == 0)
                vkd.CmdBindDescriptorSets(cmd, gfx, s->mf_layout, 1, 1, &s->mf_material_sets[material_of(s->cfg, i)],
                                          0, NULL);
            vkd.CmdBindDescriptorSets(cmd, gfx, s->mf_layout, 2, 1, &s->mf_object_set, 1, &offset);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        break;
    case MODE_CHECK:
        vkd.CmdBindPipeline(cmd, gfx, s->check_pipeline);
        for (i = 0; i < n; i++) {
            vkd.CmdBindDescriptorSets(cmd, gfx, s->check_layout, 0, 1, &s->check_sets[i], 0, NULL);
            vkd.CmdDraw(cmd, 6, 1, 0, 0);
        }
        break;
    default: /* MODE_UPDATE_THREADS: recorded above */
        break;
    }
    vkd.CmdEndRendering(cmd);
    if (f->verify)
        bench_cmd_copy_image_to_buffer(cmd, &s->targets[f->slot], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, s->readback.buffer);
}

static int verify(struct scene *s)
{
    uint32_t *want = bench_alloc(sizeof(uint32_t) * WIDTH * HEIGHT);
    int i, k, c, ok;

    for (i = 0; i < WIDTH * HEIGHT; i++)
        want[i] = bench_rgba8(0, 0, 0, 255);
    for (i = 0; i < s->cfg->draws; i++) {
        const struct bench_rect *r = &s->rects[i];
        int material = material_of(s->cfg, i);
        uint32_t rgba = 0;
        for (c = 0; c < 4; c++) {
            double v = (double)((r->rgba >> (8 * c)) & 0xFF) / 255.0;
            for (k = 0; k < SLOTS; k++)
                v *= (double)((s->texture_rgba[texture_of(material, k)] >> (8 * c)) & 0xFF) / 255.0;
            if (s->cfg->mode == MODE_MULTI_FREQUENCY)
                v *= frame_tint[c];
            rgba |= (uint32_t)(v * 255.0 + 0.5) << (8 * c);
        }
        bench_fill_rect(want, WIDTH, r->x0, r->y0, r->x1, r->y1, rgba);
    }
    /* Five-way float products: allow rounding differences of one step. */
    ok = bench_check_rgba8_tolerance(s->cfg->name, s->readback.map, want, WIDTH, HEIGHT, 1);
    free(want);
    return ok;
}

/* ==========================================================================
 * CPU-only configurations
 * ========================================================================== */

static VkDescriptorPool make_pool(struct bench_vk *vk, uint32_t sets, uint32_t ubos, uint32_t dynamic_ubos);
static VkPipeline make_pipeline(struct scene *s, VkPipelineLayout layout, const uint32_t *vs, size_t vs_size,
                                const uint32_t *fs, size_t fs_size);
#define MAKE_PIPELINE(s, layout, vs, fs) make_pipeline((s), (layout), (vs), sizeof(vs), (fs), sizeof(fs))

/* pool_threads: thread t allocates and writes THREAD_SETS sets and resets its
 * pool, THREAD_CYCLES times. */
static void pool_thread(struct scene *s, int t)
{
    VkDevice dev = s->vk->device;
    VkDescriptorSetAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s->cpu_pools[t],
        .descriptorSetCount = 1,
        .pSetLayouts = &s->set_layout,
    };
    int cycle, i;

    for (cycle = 0; cycle < THREAD_CYCLES; cycle++) {
        for (i = 0; i < THREAD_SETS; i++) {
            VkDescriptorSet set;
            BENCH_VK(vkd.AllocateDescriptorSets(dev, &alloc, &set));
            write_object(s, set, t * THREAD_SETS + i);
        }
        BENCH_VK(vkd.ResetDescriptorPool(dev, s->cpu_pools[t], 0));
    }
}

/* pool_threads, afterwards: one more cycle, kept for the check frame. */
static void pool_thread_keep(struct scene *s, int t)
{
    VkDescriptorSetAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s->cpu_pools[t],
        .descriptorSetCount = 1,
        .pSetLayouts = &s->set_layout,
    };
    int i;
    for (i = 0; i < THREAD_SETS; i++) {
        VkDescriptorSet *set = &s->check_sets[t * THREAD_SETS + i];
        BENCH_VK(vkd.AllocateDescriptorSets(s->vk->device, &alloc, set));
        write_object(s, *set, t * THREAD_SETS + i);
    }
}

/* Runs a CPU-only configuration: warmup and measured rounds, then a check
 * frame that draws with the sets it left in check_sets. */
static void run_cpu(struct scene *s, struct bench_loop *loop)
{
    const struct config *cfg = s->cfg;
    struct bench_vk *vk = s->vk;
    VkDevice dev = vk->device;
    int rounds = bench_frames(cfg->frames), warmup = cfg->mode == MODE_POOL_THREADS ? 2 : 10;
    int n = cfg->draws, r, i, k, ops = 0, checks = 0;
    double *samples = bench_alloc(sizeof(double) * (size_t)rounds);
    VkDescriptorSetLayout *layouts = bench_alloc(sizeof(VkDescriptorSetLayout) * (size_t)n);
    VkDescriptorSetAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorSetCount = 1,
        .pSetLayouts = layouts,
    };
    VkDescriptorPool pool = VK_NULL_HANDLE, check_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout array_layout = VK_NULL_HANDLE, check_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet array_set = VK_NULL_HANDLE;
    VkDescriptorImageInfo image = { s->sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &image,
    };
    uint64_t seed = 0x5E75EED5ull;
    struct config check;
    struct bench_json j;

    for (i = 0; i < n; i++)
        layouts[i] = s->set_layout;
    s->check_layout = s->layout;
    s->check_pipeline = s->pipeline;

    /* Setup. */
    switch (cfg->mode) {
    case MODE_POOL_FREE_CHURN: {
        VkDescriptorPoolSize sizes[2] = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, (uint32_t)n },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, (uint32_t)n * SLOTS },
        };
        VkDescriptorPoolCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = (uint32_t)n,
            .poolSizeCount = 2,
            .pPoolSizes = sizes,
        };
        BENCH_VK(vkd.CreateDescriptorPool(dev, &info, NULL, &pool));
        alloc.descriptorPool = pool;
        for (i = 0; i < n; i++)
            BENCH_VK(vkd.AllocateDescriptorSets(dev, &alloc, &s->check_sets[i]));
        ops = 1000;
        break;
    }
    case MODE_POOL_ALLOC_BATCH:
        pool = make_pool(vk, (uint32_t)n, (uint32_t)n, 0);
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = (uint32_t)n;
        ops = 10 * n;
        break;
    case MODE_POOL_CREATE:
        ops = 50;
        break;
    case MODE_ARRAY_WRITE: {
        struct table_layout table;
        VkDescriptorPoolSize size = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, (uint32_t)n };
        VkDescriptorPoolCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &size,
        };
        table_layout(&table, (uint32_t)n);
        BENCH_VK(vkd.CreateDescriptorSetLayout(dev, &table.info, NULL, &array_layout));
        BENCH_VK(vkd.CreateDescriptorPool(dev, &info, NULL, &pool));
        alloc.descriptorPool = pool;
        alloc.pSetLayouts = &array_layout;
        BENCH_VK(vkd.AllocateDescriptorSets(dev, &alloc, &array_set));
        write.dstSet = array_set;
        ops = 1000;
        break;
    }
    case MODE_POOL_THREADS:
        for (i = 0; i < cfg->threads; i++)
            s->cpu_pools[i] = make_pool(vk, THREAD_SETS, THREAD_SETS, 0);
        team_start(&s->team, s, cfg->threads);
        ops = THREAD_CYCLES * THREAD_SETS;
        break;
    default:
        break;
    }

    /* Rounds. */
    for (r = -warmup; r < rounds; r++) {
        double t0 = bench_now_us();
        switch (cfg->mode) {
        case MODE_POOL_FREE_CHURN:
            for (k = 0; k < ops; k++) {
                VkDescriptorSet *set = &s->check_sets[bench_rand(&seed) % (uint32_t)n];
                BENCH_VK(vkd.FreeDescriptorSets(dev, pool, 1, set));
                BENCH_VK(vkd.AllocateDescriptorSets(dev, &alloc, set));
            }
            break;
        case MODE_POOL_ALLOC_BATCH:
            for (k = 0; k < 10; k++) {
                BENCH_VK(vkd.AllocateDescriptorSets(dev, &alloc, s->check_sets));
                BENCH_VK(vkd.ResetDescriptorPool(dev, pool, 0));
            }
            break;
        case MODE_POOL_CREATE:
            for (k = 0; k < ops; k++)
                vkd.DestroyDescriptorPool(dev, make_pool(vk, (uint32_t)n, (uint32_t)n, 0), NULL);
            break;
        case MODE_ARRAY_WRITE:
            for (k = 0; k < ops; k++) {
                write.dstArrayElement = bench_rand(&seed) % (uint32_t)n;
                image.imageView = s->textures[k % TEXTURES].view;
                vkd.UpdateDescriptorSets(dev, 1, &write, 0, NULL);
            }
            break;
        case MODE_POOL_THREADS:
            team_run(&s->team, pool_thread);
            break;
        default:
            break;
        }
        if (r >= 0)
            samples[r] = bench_now_us() - t0;
    }

    /* The sets to check: object i's descriptors in check_sets[i]. */
    switch (cfg->mode) {
    case MODE_POOL_FREE_CHURN: /* every set the pool holds, written after the churn */
        checks = n;
        for (i = 0; i < checks; i++)
            write_object(s, s->check_sets[i], i);
        break;
    case MODE_POOL_ALLOC_BATCH:
        checks = n;
        BENCH_VK(vkd.AllocateDescriptorSets(dev, &alloc, s->check_sets));
        for (i = 0; i < checks; i++)
            write_object(s, s->check_sets[i], i);
        break;
    case MODE_POOL_CREATE: /* a pool made the same way, used */
        checks = n;
        pool = make_pool(vk, (uint32_t)n, (uint32_t)n, 0);
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = (uint32_t)n;
        BENCH_VK(vkd.AllocateDescriptorSets(dev, &alloc, s->check_sets));
        for (i = 0; i < checks; i++)
            write_object(s, s->check_sets[i], i);
        break;
    case MODE_ARRAY_WRITE: {
        /* Texture t at element t * 4093 % n, then object i's textures copied
         * from the table into its set. Copies out of an update-after-bind
         * set go into one: the check sets have set_layout's bindings in an
         * update-after-bind layout and pool, drawn with a pipeline of their
         * own. */
        VkDescriptorSetLayoutBinding bindings[1 + SLOTS];
        VkDescriptorPoolSize sizes[2] = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, CHECK_ELEMENTS },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, CHECK_ELEMENTS * SLOTS },
        };
        VkDescriptorPoolCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
            .maxSets = CHECK_ELEMENTS,
            .poolSizeCount = 2,
            .pPoolSizes = sizes,
        };
        VkCopyDescriptorSet copies[SLOTS];
        checks = CHECK_ELEMENTS;
        for (i = 0; i < TEXTURES; i++) {
            write.dstArrayElement = (uint32_t)((long long)i * 4093 % n);
            image.imageView = s->textures[i].view;
            vkd.UpdateDescriptorSets(dev, 1, &write, 0, NULL);
        }
        object_bindings(bindings);
        check_set_layout = bench_set_layout(vk, 1 + SLOTS, bindings,
                                            VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT);
        s->check_layout = bench_pipeline_layout(vk, 1, &check_set_layout, 0, 0);
        s->check_pipeline = MAKE_PIPELINE(s, s->check_layout, rect_ubo_vert, rect_material_frag);
        BENCH_VK(vkd.CreateDescriptorPool(dev, &info, NULL, &check_pool));
        for (i = 0; i < CHECK_ELEMENTS; i++)
            layouts[i] = check_set_layout;
        alloc.descriptorPool = check_pool;
        alloc.descriptorSetCount = CHECK_ELEMENTS;
        alloc.pSetLayouts = layouts;
        BENCH_VK(vkd.AllocateDescriptorSets(dev, &alloc, s->check_sets));
        for (i = 0; i < checks; i++) {
            VkWriteDescriptorSet writes[1 + SLOTS]; /* only the uniform buffer's is used */
            struct descriptors d;
            object_descriptors(s, i, &d);
            object_writes(&d, s->check_sets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, writes);
            memset(copies, 0, sizeof(copies));
            for (k = 0; k < SLOTS; k++) {
                copies[k].sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
                copies[k].srcSet = array_set;
                copies[k].srcArrayElement = (uint32_t)((long long)texture_of(i % MATERIALS, k) * 4093 % n);
                copies[k].dstSet = s->check_sets[i];
                copies[k].dstBinding = (uint32_t)(1 + k);
                copies[k].descriptorCount = 1;
            }
            vkd.UpdateDescriptorSets(dev, 1, writes, SLOTS, copies);
        }
        break;
    }
    case MODE_POOL_THREADS:
        checks = cfg->threads * THREAD_SETS;
        team_run(&s->team, pool_thread_keep);
        break;
    default:
        break;
    }
    check = *cfg;
    check.mode = MODE_CHECK;
    check.draws = checks;
    s->cfg = &check;
    bench_loop_verify(loop, record, s);
    if (verify(s)) {
        double median = bench_median(samples, rounds);
        bench_json_begin(&j, cfg->name);
        bench_json_int(&j, cfg->mode == MODE_ARRAY_WRITE ? "array_size" : "sets", n);
        if (cfg->mode == MODE_POOL_THREADS)
            bench_json_int(&j, "threads", cfg->threads);
        bench_json_int(&j, "rounds", rounds);
        bench_json_int(&j, "ops_per_round", ops);
        bench_json_stats(&j, "round_us", samples, rounds);
        bench_json_num(&j, "ns_per_op", median * 1000.0 / ops);
        bench_json_end(&j);
    }
    s->cfg = cfg;

    if (cfg->mode == MODE_POOL_THREADS) {
        team_stop(&s->team);
        for (i = 0; i < cfg->threads; i++)
            vkd.DestroyDescriptorPool(dev, s->cpu_pools[i], NULL);
    }
    if (pool)
        vkd.DestroyDescriptorPool(dev, pool, NULL);
    if (check_pool)
        vkd.DestroyDescriptorPool(dev, check_pool, NULL);
    if (array_layout)
        vkd.DestroyDescriptorSetLayout(dev, array_layout, NULL);
    if (check_set_layout) {
        vkd.DestroyPipeline(dev, s->check_pipeline, NULL);
        vkd.DestroyPipelineLayout(dev, s->check_layout, NULL);
        vkd.DestroyDescriptorSetLayout(dev, check_set_layout, NULL);
    }
    free(layouts);
    free(samples);
}

/* ==========================================================================
 * Setup
 * ========================================================================== */

static VkPipeline make_pipeline(struct scene *s, VkPipelineLayout layout, const uint32_t *vs, size_t vs_size,
                                const uint32_t *fs, size_t fs_size)
{
    struct bench_vk *vk = s->vk;
    struct bench_gfx_desc desc;
    VkPipeline pipeline;
    memset(&desc, 0, sizeof(desc));
    desc.vs = bench_shader_module(vk, vs, vs_size);
    desc.fs = bench_shader_module(vk, fs, fs_size);
    desc.layout = layout;
    desc.color_count = 1;
    desc.color_formats[0] = VK_FORMAT_R8G8B8A8_UNORM;
    pipeline = bench_gfx_pipeline(vk, &desc);
    vkd.DestroyShaderModule(vk->device, desc.vs, NULL);
    vkd.DestroyShaderModule(vk->device, desc.fs, NULL);
    return pipeline;
}

static VkDescriptorPool make_pool(struct bench_vk *vk, uint32_t sets, uint32_t ubos, uint32_t dynamic_ubos)
{
    VkDescriptorPoolSize sizes[3] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubos ? ubos : 1 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, dynamic_ubos ? dynamic_ubos : 1 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sets * SLOTS },
    };
    VkDescriptorPoolCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = sets,
        .poolSizeCount = 3,
        .pPoolSizes = sizes,
    };
    VkDescriptorPool pool = VK_NULL_HANDLE;
    BENCH_VK(vkd.CreateDescriptorPool(vk->device, &info, NULL, &pool));
    return pool;
}

static VkDescriptorUpdateTemplate make_template(struct scene *s, VkDescriptorUpdateTemplateType type,
                                                VkDescriptorSetLayout set_layout, VkPipelineLayout layout)
{
    VkDescriptorUpdateTemplateEntry entries[1 + SLOTS];
    VkDescriptorUpdateTemplateCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO,
        .descriptorUpdateEntryCount = 1 + SLOTS,
        .pDescriptorUpdateEntries = entries,
        .templateType = type,
        .descriptorSetLayout = set_layout,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .pipelineLayout = layout,
        .set = 0,
    };
    VkDescriptorUpdateTemplate tmpl = VK_NULL_HANDLE;
    int k;
    for (k = 0; k < 1 + SLOTS; k++) {
        entries[k].dstBinding = (uint32_t)k;
        entries[k].dstArrayElement = 0;
        entries[k].descriptorCount = 1;
        entries[k].descriptorType = k == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        entries[k].offset = k == 0 ? offsetof(struct descriptors, ubo)
                                   : offsetof(struct descriptors, images) + (size_t)(k - 1) * sizeof(VkDescriptorImageInfo);
        entries[k].stride = 0;
    }
    BENCH_VK(vkd.CreateDescriptorUpdateTemplate(s->vk->device, &info, NULL, &tmpl));
    return tmpl;
}

/* multi_frequency's layouts, pipeline and sets. */
static void setup_multi_frequency(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDeviceSize align = vk->props.limits.minUniformBufferOffsetAlignment;
    VkDescriptorSetLayoutBinding bindings[SLOTS];
    VkDescriptorPoolSize sizes[3] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, BENCH_FRAMES_IN_FLIGHT },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MATERIALS * SLOTS },
    };
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = BENCH_FRAMES_IN_FLIGHT + MATERIALS + 1,
        .poolSizeCount = 3,
        .pPoolSizes = sizes,
    };
    VkDescriptorSetAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorSetCount = 1,
    };
    VkDescriptorBufferInfo buffer;
    VkDescriptorImageInfo images[SLOTS];
    VkWriteDescriptorSet writes[SLOTS];
    int i, k;

    for (k = 0; k < SLOTS; k++) {
        bindings[k].binding = (uint32_t)k;
        bindings[k].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[k].descriptorCount = 1;
        bindings[k].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[k].pImmutableSamplers = NULL;
    }
    s->mf_set_layouts[1] = bench_set_layout(vk, SLOTS, bindings, 0);
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    s->mf_set_layouts[0] = bench_set_layout(vk, 1, bindings, 0);
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    s->mf_set_layouts[2] = bench_set_layout(vk, 1, bindings, 0);
    s->mf_layout = bench_pipeline_layout(vk, 3, s->mf_set_layouts, 0, 0);
    s->mf_pipeline = MAKE_PIPELINE(s, s->mf_layout, rect_object_vert, rect_frame_material_frag);

    /* Per frame in flight: its slot of a host-visible buffer. */
    if (align < 1)
        align = 1;
    s->frame_stride = (sizeof(frame_tint) + align - 1) / align * align;
    bench_buffer_create(vk, &s->frame_ubo, s->frame_stride * BENCH_FRAMES_IN_FLIGHT,
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, BENCH_HOST, 0);
    BENCH_VK(vkd.CreateDescriptorPool(vk->device, &pool_info, NULL, &s->mf_pool));
    alloc.descriptorPool = s->mf_pool;
    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT + MATERIALS + 1; i++) {
        int material = i - BENCH_FRAMES_IN_FLIGHT;
        VkDescriptorSet *set = i < BENCH_FRAMES_IN_FLIGHT ? &s->mf_frame_sets[i]
                               : material < MATERIALS     ? &s->mf_material_sets[material]
                                                          : &s->mf_object_set;
        alloc.pSetLayouts = &s->mf_set_layouts[i < BENCH_FRAMES_IN_FLIGHT ? 0 : material < MATERIALS ? 1 : 2];
        BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &alloc, set));
        memset(writes, 0, sizeof(writes));
        for (k = 0; k < SLOTS; k++) {
            writes[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[k].dstSet = *set;
            writes[k].dstBinding = (uint32_t)k;
            writes[k].descriptorCount = 1;
            writes[k].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[k].pImageInfo = &images[k];
            writes[k].pBufferInfo = &buffer;
            images[k].sampler = s->sampler;
            images[k].imageView = s->textures[texture_of(material < 0 ? 0 : material % MATERIALS, k)].view;
            images[k].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        if (material < 0) { /* frame */
            buffer.buffer = s->frame_ubo.buffer;
            buffer.offset = (VkDeviceSize)i * s->frame_stride;
            buffer.range = sizeof(frame_tint);
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        } else if (material == MATERIALS) { /* object */
            buffer.buffer = s->ubo.buffer;
            buffer.offset = 0; /* the dynamic offset selects the object */
            buffer.range = sizeof(struct object_ubo);
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        }
        vkd.UpdateDescriptorSets(vk->device, material >= 0 && material < MATERIALS ? SLOTS : 1, writes, 0, NULL);
    }
}

/* update_sets_mt: per thread and frame in flight, a descriptor pool and a
 * command pool with one secondary. */
static void setup_threads(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = vk->queue_family,
    };
    VkCommandBufferAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .level = VK_COMMAND_BUFFER_LEVEL_SECONDARY,
        .commandBufferCount = 1,
    };
    int t, slot;
    for (t = 0; t < RECORD_THREADS; t++)
        for (slot = 0; slot < BENCH_FRAMES_IN_FLIGHT; slot++) {
            s->thread_pools[t][slot] = make_pool(vk, OBJECTS, OBJECTS, 0);
            BENCH_VK(vkd.CreateCommandPool(vk->device, &pool_info, NULL, &s->thread_cmd_pools[t][slot]));
            alloc.commandPool = s->thread_cmd_pools[t][slot];
            BENCH_VK(vkd.AllocateCommandBuffers(vk->device, &alloc, &s->thread_cmds[t][slot]));
        }
}

static void setup(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDeviceSize align = vk->props.limits.minUniformBufferOffsetAlignment;
    VkDescriptorSetLayoutBinding bindings[1 + SLOTS];
    VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxLod = 0.0f,
    };
    struct object_ubo *ubo_data;
    uint32_t texels[TEXTURE_SIZE * TEXTURE_SIZE];
    uint64_t seed = 0x7E57ED5Eull;
    int i, k;

    /* Targets, textures, per-object uniform data. */
    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++)
        bench_image_create(vk, &s->targets[i], VK_FORMAT_R8G8B8A8_UNORM, WIDTH, HEIGHT, 1, VK_SAMPLE_COUNT_1_BIT,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0);
    bench_buffer_create(vk, &s->readback, WIDTH * HEIGHT * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, BENCH_HOST,
                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    for (i = 0; i < TEXTURES; i++) {
        /* Solid colors with every channel in [160, 255]. */
        s->texture_rgba[i] = bench_rgba8(160 + bench_rand(&seed) % 96, 160 + bench_rand(&seed) % 96,
                                         160 + bench_rand(&seed) % 96, 255);
        for (k = 0; k < TEXTURE_SIZE * TEXTURE_SIZE; k++)
            texels[k] = s->texture_rgba[i];
        bench_image_create(vk, &s->textures[i], VK_FORMAT_R8G8B8A8_UNORM, TEXTURE_SIZE, TEXTURE_SIZE, 1,
                           VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0);
        bench_image_upload(vk, &s->textures[i], texels, sizeof(texels), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    BENCH_VK(vkd.CreateSampler(vk->device, &sampler_info, NULL, &s->sampler));

    bench_rect_scene(s->rects, OBJECTS, WIDTH, HEIGHT, 17, 0, 0xDE5C0001ull);
    if (align < 1)
        align = 1;
    s->ubo_stride = (sizeof(struct object_ubo) + align - 1) / align * align;
    ubo_data = bench_alloc((size_t)(s->ubo_stride * OBJECTS));
    for (i = 0; i < OBJECTS; i++) {
        struct object_ubo *o = (struct object_ubo *)((char *)ubo_data + (size_t)(s->ubo_stride * (VkDeviceSize)i));
        bench_rect_ndc(&s->rects[i], WIDTH, HEIGHT, 0, 0, o->rect);
        bench_unorm4(s->rects[i].rgba, o->color);
    }
    bench_buffer_create(vk, &s->ubo, s->ubo_stride * OBJECTS,
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, BENCH_DEVICE_LOCAL);
    bench_buffer_upload(vk, &s->ubo, 0, ubo_data, s->ubo_stride * OBJECTS);
    free(ubo_data);

    /* Layouts: plain, dynamic uniform buffer, push. */
    object_bindings(bindings);
    s->set_layout = bench_set_layout(vk, 1 + SLOTS, bindings, 0);
    if (vk->push_descriptor)
        s->push_set_layout = bench_set_layout(vk, 1 + SLOTS, bindings, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT);
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    s->dynamic_set_layout = bench_set_layout(vk, 1 + SLOTS, bindings, 0);
    s->layout = bench_pipeline_layout(vk, 1, &s->set_layout, 0, 0);
    s->dynamic_layout = bench_pipeline_layout(vk, 1, &s->dynamic_set_layout, 0, 0);
    s->pipeline = MAKE_PIPELINE(s, s->layout, rect_ubo_vert, rect_material_frag);
    s->dynamic_pipeline = MAKE_PIPELINE(s, s->dynamic_layout, rect_ubo_vert, rect_material_frag);
    if (vk->push_descriptor) {
        s->push_layout = bench_pipeline_layout(vk, 1, &s->push_set_layout, 0, 0);
        s->push_pipeline = MAKE_PIPELINE(s, s->push_layout, rect_ubo_vert, rect_material_frag);
        s->push_template = make_template(s, VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS, VK_NULL_HANDLE,
                                         s->push_layout);
    }
    s->update_template = make_template(s, VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET, s->set_layout,
                                       VK_NULL_HANDLE);

    /* Pools; the static sets and per-material sets are written once. */
    s->static_pool = make_pool(vk, OBJECTS + MATERIALS, OBJECTS, MATERIALS);
    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++)
        s->frame_pools[i] = make_pool(vk, OBJECTS, OBJECTS, 0);
    for (i = 0; i < OBJECTS + MATERIALS; i++) {
        int material = i >= OBJECTS;
        VkDescriptorSet *set = material ? &s->material_sets[i - OBJECTS] : &s->static_sets[i];
        VkDescriptorSetAllocateInfo alloc = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = s->static_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = material ? &s->dynamic_set_layout : &s->set_layout,
        };
        VkWriteDescriptorSet writes[1 + SLOTS];
        struct descriptors d;
        BENCH_VK(vkd.AllocateDescriptorSets(vk->device, &alloc, set));
        object_descriptors(s, material ? i - OBJECTS : i, &d);
        if (material) {
            d.ubo.offset = 0; /* the dynamic offset selects the object */
            object_writes(&d, *set, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, writes);
        } else {
            object_writes(&d, *set, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, writes);
        }
        vkd.UpdateDescriptorSets(vk->device, 1 + SLOTS, writes, 0, NULL);
    }

    setup_multi_frequency(s);
    setup_threads(s);
}

static void teardown(struct scene *s)
{
    struct bench_vk *vk = s->vk;
    VkDevice dev = vk->device;
    int i, slot;
    vkd.DeviceWaitIdle(dev);
    for (i = 0; i < RECORD_THREADS; i++)
        for (slot = 0; slot < BENCH_FRAMES_IN_FLIGHT; slot++) {
            vkd.DestroyDescriptorPool(dev, s->thread_pools[i][slot], NULL);
            vkd.DestroyCommandPool(dev, s->thread_cmd_pools[i][slot], NULL);
        }
    vkd.DestroyDescriptorPool(dev, s->mf_pool, NULL);
    vkd.DestroyPipeline(dev, s->mf_pipeline, NULL);
    vkd.DestroyPipelineLayout(dev, s->mf_layout, NULL);
    for (i = 0; i < 3; i++)
        vkd.DestroyDescriptorSetLayout(dev, s->mf_set_layouts[i], NULL);
    bench_buffer_destroy(vk, &s->frame_ubo);
    vkd.DestroyDescriptorPool(dev, s->static_pool, NULL);
    for (i = 0; i < BENCH_FRAMES_IN_FLIGHT; i++) {
        vkd.DestroyDescriptorPool(dev, s->frame_pools[i], NULL);
        bench_image_destroy(vk, &s->targets[i]);
    }
    vkd.DestroyDescriptorUpdateTemplate(dev, s->update_template, NULL);
    if (s->push_template)
        vkd.DestroyDescriptorUpdateTemplate(dev, s->push_template, NULL);
    vkd.DestroyPipeline(dev, s->pipeline, NULL);
    vkd.DestroyPipeline(dev, s->dynamic_pipeline, NULL);
    if (s->push_pipeline)
        vkd.DestroyPipeline(dev, s->push_pipeline, NULL);
    vkd.DestroyPipelineLayout(dev, s->layout, NULL);
    vkd.DestroyPipelineLayout(dev, s->dynamic_layout, NULL);
    if (s->push_layout)
        vkd.DestroyPipelineLayout(dev, s->push_layout, NULL);
    vkd.DestroyDescriptorSetLayout(dev, s->set_layout, NULL);
    vkd.DestroyDescriptorSetLayout(dev, s->dynamic_set_layout, NULL);
    if (s->push_set_layout)
        vkd.DestroyDescriptorSetLayout(dev, s->push_set_layout, NULL);
    vkd.DestroySampler(dev, s->sampler, NULL);
    for (i = 0; i < TEXTURES; i++)
        bench_image_destroy(vk, &s->textures[i]);
    bench_buffer_destroy(vk, &s->ubo);
    bench_buffer_destroy(vk, &s->readback);
}

/* Whether the device has array_write's table, or prints why not. */
static int array_supported(struct bench_vk *vk, const struct config *cfg)
{
    VkPhysicalDeviceVulkan12Properties p12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES };
    VkPhysicalDeviceProperties2 props = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &p12 };
    VkDescriptorSetLayoutSupport support = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT };
    struct table_layout table;
    uint32_t limit;

    if (!vk->update_after_bind) {
        bench_skip(cfg->name, "features descriptorBindingSampledImageUpdateAfterBind and "
                              "descriptorBindingPartiallyBound not supported");
        return 0;
    }
    vkGetPhysicalDeviceProperties2(vk->physical_device, &props);
    limit = p12.maxDescriptorSetUpdateAfterBindSamplers;
    if (p12.maxDescriptorSetUpdateAfterBindSampledImages < limit)
        limit = p12.maxDescriptorSetUpdateAfterBindSampledImages;
    if (p12.maxPerStageDescriptorUpdateAfterBindSamplers < limit)
        limit = p12.maxPerStageDescriptorUpdateAfterBindSamplers;
    if (p12.maxPerStageDescriptorUpdateAfterBindSampledImages < limit)
        limit = p12.maxPerStageDescriptorUpdateAfterBindSampledImages;
    if (p12.maxPerStageUpdateAfterBindResources < limit)
        limit = p12.maxPerStageUpdateAfterBindResources;
    if (limit < (uint32_t)cfg->draws) {
        bench_skip(cfg->name, "update-after-bind sampler and sampled image limits allow %u, not %d", limit,
                   cfg->draws);
        return 0;
    }
    /* Beyond maxPerSetDescriptors, a layout may still be supported. */
    table_layout(&table, (uint32_t)cfg->draws);
    vkd.GetDescriptorSetLayoutSupport(vk->device, &table.info, &support);
    if (!support.supported) {
        bench_skip(cfg->name, "vkGetDescriptorSetLayoutSupport: a table of %d textures is not supported",
                   cfg->draws);
        return 0;
    }
    return 1;
}

/* Names (name) or unnames (NULL) the sampler. */
static void name_sampler(struct scene *s, const char *name)
{
    VkDebugUtilsObjectNameInfoEXT info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
        .objectType = VK_OBJECT_TYPE_SAMPLER,
        .objectHandle = (uint64_t)(uintptr_t)s->sampler,
        .pObjectName = name,
    };
    BENCH_VK(SetDebugUtilsObjectName(s->vk->device, &info));
}

int main(void)
{
    struct bench_vk vk;
    struct bench_loop loop;
    struct scene *s;
    size_t c;

    bench_begin("descriptor_churn");
    bench_vk_init(&vk, BENCH_NEED_DYNAMIC_RENDERING | BENCH_NEED_SYNC2 | BENCH_WANT_PUSH_DESCRIPTOR |
                           BENCH_WANT_UPDATE_AFTER_BIND);
    SetDebugUtilsObjectName =
        (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(vk.device, "vkSetDebugUtilsObjectNameEXT");
    s = bench_alloc(sizeof(*s));
    s->vk = &vk;
    setup(s);
    bench_loop_init(&loop, &vk, 1000);

    for (c = 0; c < sizeof(configs) / sizeof(configs[0]); c++) {
        const struct config *cfg = &configs[c];
        int frames = bench_frames(cfg->frames);
        struct bench_json j;
        double cpu_median;

        if (!bench_selected(cfg->name))
            continue;
        bench_config = cfg->name;
        if ((cfg->mode == MODE_PUSH || cfg->mode == MODE_PUSH_TEMPLATE) &&
            (!vk.push_descriptor || !vkd.CmdPushDescriptorSet || !vkd.CmdPushDescriptorSetWithTemplate)) {
            bench_skip(cfg->name, "feature pushDescriptor (Vulkan 1.4) not supported");
            continue;
        }
        if (cfg->named && !SetDebugUtilsObjectName) {
            bench_skip(cfg->name, "VK_EXT_debug_utils not enabled on the instance");
            continue;
        }
        if (cfg->mode == MODE_ARRAY_WRITE && !array_supported(&vk, cfg))
            continue;
        s->cfg = cfg;
        if (cfg->named)
            name_sampler(s, "material sampler");
        if (cfg->mode >= MODE_POOL_FREE_CHURN) {
            run_cpu(s, &loop);
        } else {
            if (cfg->mode == MODE_UPDATE_THREADS)
                team_start(&s->team, s, cfg->threads);
            bench_loop_verify(&loop, record, s);
            if (verify(s)) {
                bench_loop_run(&loop, record, s, 10, frames < 1000 ? frames : 1000);
                cpu_median = bench_loop_cpu_median(&loop);

                bench_json_begin(&j, cfg->name);
                bench_json_int(&j, "draws", cfg->draws);
                bench_json_int(&j, "descriptors_per_draw", 1 + SLOTS);
                if (cfg->threads)
                    bench_json_int(&j, "threads", cfg->threads);
                bench_loop_report(&loop, &j);
                bench_json_num(&j, "cpu_ns_per_draw", cpu_median * 1000.0 / cfg->draws);
                bench_json_end(&j);
            }
            if (cfg->mode == MODE_UPDATE_THREADS)
                team_stop(&s->team);
        }
        if (cfg->named)
            name_sampler(s, NULL);
    }
    bench_config = NULL;

    bench_loop_destroy(&loop);
    teardown(s);
    free(s);
    bench_vk_destroy(&vk);
    return bench_exit_code();
}
