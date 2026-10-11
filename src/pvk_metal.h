/* C declarations for the parts of Metal and Foundation PlasmaVK uses.
 *
 * Everything is pvk_/PVK_-prefixed so a translation unit that also imports
 * <Metal/Metal.h> (an Objective-C file including the implementation) does
 * not clash with Apple's names. Values come from the macOS/iOS 26 SDK.
 */
#ifndef PVK_METAL_H
#define PVK_METAL_H

#include <pthread.h>
#include <stdint.h>

#include "pvk_objc.h"

extern id MTLCreateSystemDefaultDevice(void);

/* MTLGPUFamily */
enum {
    PVK_MTL_GPU_FAMILY_APPLE7 = 1007, /* M1 / A14 */
    PVK_MTL_GPU_FAMILY_METAL4 = 5002,
};

/* Every selector PlasmaVK sends, registered once (pvk_metal_init) and then
 * read from pvk_sel. Keep sorted by field name. */
#define PVK_SELECTORS(X)                                                   \
    X(UTF8String, "UTF8String")                                            \
    X(name, "name")                                                        \
    X(newMTL4CommandQueue, "newMTL4CommandQueue")                          \
    X(recommendedMaxWorkingSetSize, "recommendedMaxWorkingSetSize")        \
    X(release, "release")                                                  \
    X(supportsFamily, "supportsFamily:")

#define PVK_SEL_FIELD(field, string) SEL field;
static struct {
    PVK_SELECTORS(PVK_SEL_FIELD)
} pvk_sel;
#undef PVK_SEL_FIELD

static pthread_once_t pvk_metal_once = PTHREAD_ONCE_INIT;

static void pvk_metal_init_once(void)
{
#define PVK_SEL_INIT(field, string) pvk_sel.field = sel_registerName(string);
    PVK_SELECTORS(PVK_SEL_INIT)
#undef PVK_SEL_INIT
}

static void pvk_metal_init(void)
{
    pthread_once(&pvk_metal_once, pvk_metal_init_once);
}

/* ---- Typed wrappers for common messages ---- */

static inline void pvk_release(id object)
{
    if (object)
        PVK_MSG(void, (id, SEL))(object, pvk_sel.release);
}

static inline const char *pvk_ns_utf8(id string)
{
    return PVK_MSG(const char *, (id, SEL))(string, pvk_sel.UTF8String);
}

static inline const char *pvk_mtl_device_name(id device)
{
    return pvk_ns_utf8(PVK_MSG(id, (id, SEL))(device, pvk_sel.name));
}

static inline int pvk_mtl_device_supports_family(id device, long family)
{
    return PVK_MSG(BOOL, (id, SEL, long))(device, pvk_sel.supportsFamily, family) ? 1 : 0;
}

#endif /* PVK_METAL_H */
