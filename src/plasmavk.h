/*
 * PlasmaVK - Vulkan on Apple Metal in a single C99 header.
 *
 * USAGE
 *   In exactly one C file:
 *
 *       #define PLASMAVK_IMPLEMENTATION
 *       #include "plasmavk.h"
 *
 *   Everywhere else, include "plasmavk.h" (or <vulkan/vulkan.h>) as usual and
 *   call vk* functions directly. There is no loader, no library and no extra
 *   linker flags: the header asks the linker for Metal and libobjc itself.
 *
 * REQUIREMENTS
 *   macOS 26+ or iOS 26+ on a Metal 4 GPU (Apple M1 / A14 or newer).
 *   The implementation must be compiled as C (C99 or later), or as
 *   Objective-C without ARC.
 *
 * CONFIGURATION (define before including the implementation)
 *   PLASMAVK_ABORT_ON_UNIMPLEMENTED  abort() when an unimplemented command is
 *                                    called, instead of logging once.
 *
 * ENVIRONMENT
 *   PLASMAVK_LOG=error|warn|info|debug   log verbosity (default: warn)
 *
 * LICENSE
 *   MIT. See the end of this file.
 */
#ifndef PLASMAVK_H
#define PLASMAVK_H

#if !defined(__APPLE__)
#error "PlasmaVK only supports Apple platforms (macOS/iOS 26+)."
#endif

#include "vulkan/vulkan_core.h"
#include "vulkan/vulkan_metal.h"

#define PLASMAVK_VERSION_MAJOR 0
#define PLASMAVK_VERSION_MINOR 1
#define PLASMAVK_VERSION_PATCH 0
#define PLASMAVK_VERSION \
    VK_MAKE_API_VERSION(0, PLASMAVK_VERSION_MAJOR, PLASMAVK_VERSION_MINOR, PLASMAVK_VERSION_PATCH)

/* Oldest Vulkan header PlasmaVK compiles against when the application
 * includes its own <vulkan/vulkan.h> before this file. */
#define PLASMAVK_MIN_HEADER_VERSION 363

#if VK_HEADER_VERSION < PLASMAVK_MIN_HEADER_VERSION
#error "PlasmaVK needs Vulkan headers 1.4.363 or newer; include plasmavk.h before other Vulkan headers."
#endif

#endif /* PLASMAVK_H */

#if defined(PLASMAVK_IMPLEMENTATION) && !defined(PLASMAVK_IMPLEMENTATION_INCLUDED)
#define PLASMAVK_IMPLEMENTATION_INCLUDED

#if defined(__cplusplus)
#error "Compile the PlasmaVK implementation in a C file."
#endif
#if defined(__OBJC__) && __has_feature(objc_arc)
#error "Compile the PlasmaVK implementation without ARC (-fno-objc-arc) or as plain C."
#endif

#include "pvk_internal.h"
#include "pvk_instance.c"
#include "pvk_physical_device.c"
#include "pvk_device.c"
#include "gen/pvk_stubs.h"
#include "pvk_proc_addr.c"

#endif /* PLASMAVK_IMPLEMENTATION */
