/* C99 bridge to the Objective-C runtime.
 *
 * Metal is an Objective-C API. Instead of compiling Objective-C, PlasmaVK
 * talks to the runtime directly: methods are invoked through objc_msgSend
 * cast to the exact function type, and blocks are built by hand following
 * the Apple Block ABI, so the implementation needs neither ObjC nor -fblocks.
 */
#ifndef PVK_OBJC_H
#define PVK_OBJC_H

#include <objc/message.h>
#include <objc/runtime.h>

/* Ask the linker for the frameworks we use (LC_LINKER_OPTION), so the
 * application needs no -framework flags. Unused ones are dropped by ld. */
__asm__(".linker_option \"-framework\", \"Metal\"\n"
        ".linker_option \"-framework\", \"QuartzCore\"\n"
        ".linker_option \"-framework\", \"Foundation\"\n"
        ".linker_option \"-lobjc\"\n");

/* Exported by libobjc but absent from its public headers. */
extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *pool);

/* Typed message send:
 *     PVK_MSG(ReturnType, (id, SEL, ArgTypes...))(receiver, selector, args...)
 * objc_msgSend must always be called through the callee's exact type. */
#define PVK_MSG(R, PARAMS) ((R(*) PARAMS)objc_msgSend)

/* ---- Blocks (https://clang.llvm.org/docs/Block-ABI-Apple.html) ---- */

extern void *_NSConcreteStackBlock[32];

struct pvk_block_descriptor {
    unsigned long reserved;
    unsigned long size;
};

/* A block taking one object argument, e.g. Metal's completion handlers.
 * Captures are plain data, so the runtime's Block_copy is a memmove and no
 * copy/dispose helpers are needed. */
struct pvk_block {
    void *isa;
    int flags;
    int reserved;
    void (*invoke)(struct pvk_block *self, id arg);
    const struct pvk_block_descriptor *descriptor;
    /* captures */
    void (*fn)(void *user, id arg);
    void *user;
};

static inline void pvk_block_invoke(struct pvk_block *self, id arg)
{
    self->fn(self->user, arg);
}

static const struct pvk_block_descriptor pvk_block_descriptor = { 0, sizeof(struct pvk_block) };

/* Returns a stack block. Pass its address straight to an API that copies the
 * block (every Metal handler-taking method does); do not keep it around. */
static inline struct pvk_block pvk_block_make(void (*fn)(void *user, id arg), void *user)
{
    struct pvk_block block;
    block.isa = _NSConcreteStackBlock;
    block.flags = 0;
    block.reserved = 0;
    block.invoke = pvk_block_invoke;
    block.descriptor = &pvk_block_descriptor;
    block.fn = fn;
    block.user = user;
    return block;
}

#endif /* PVK_OBJC_H */
