// An Objective-C file that imports Apple's Metal headers and also compiles
// the PlasmaVK implementation: the pvk_-prefixed bindings must not clash.
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"
