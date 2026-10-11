# PlasmaVK

Vulkan on Apple Metal in one C99 header, `plasmavk.h`. Everything is inside
it, including the Vulkan API headers. No loader, no library, no build step,
no dependencies.

```c
// In exactly one .c file:
#define PLASMAVK_IMPLEMENTATION
#include "plasmavk.h"
```

Everywhere else, include `plasmavk.h` and call `vk*` as usual. You don't need
linker flags: the header links Metal itself.

**Status:** early. Instances, devices and queues, memory and buffers, command
buffers with transfer and compute commands, queue submission, fences, binary
and timeline semaphores, and compute pipelines (SPIR-V translated to MSL,
push constants and buffer device addresses) work. Formats (every core format
Metal can hold, compressed BC/ETC2/ASTC included), images, image and buffer
views, samplers, image copies, blits, clears and resolves, and host image
copies work. Descriptor sets work: every descriptor type, dynamic offsets,
inline uniform blocks, push descriptors, update templates and descriptor
indexing (update after bind, partially bound, variable-count and runtime
arrays, non-uniform indexing). Window system integration works:
VK_EXT_metal_surface surfaces on a CAMetalLayer, VK_KHR_swapchain (FIFO and
immediate presentation, sRGB and, with VK_EXT_swapchain_colorspace, extended
and HDR color spaces). Graphics is next; every other Vulkan 1.4
command exists but logs "not implemented". The target is full Vulkan 1.4 plus
the Roadmap 2026 profile, measured by the Khronos conformance suite.

## Requirements

- macOS 26+ or iOS 26+ on a Metal 4 GPU (Apple M1 / A14 or newer).
- The implementation must be compiled as C, or as Objective-C without ARC.

## Repository layout

| Path | What |
|---|---|
| `plasmavk.h` | The library. It is the only source file and is edited directly; its header comment lists its sections. |
| `tests/smoke/` | Fast C99 tests. Each one includes the header with no link flags. |
| `tests/conformance/` | Black-box Vulkan conformance tests, spec-cited, one file per spec chapter; also run on MoltenVK or any driver. |
| `tests/compile/` | Compile and link checks: C dialects, C++, ObjC (with and without `<Metal/Metal.h>`), iOS, two translation units. |
| `bench/` | Realistic-workload benchmarks, comparable against MoltenVK and KosmicKrisp. |

## Testing

```bash
make test       # smoke tests (ASan + UBSan)
make check      # everything a user hits, then the smoke and conformance tests
make conformance-mvk  # the conformance tests on MoltenVK (see tests/conformance/README.md)
make bench      # benchmarks on PlasmaVK
make bench-mvk  # the same benchmarks on MoltenVK (Homebrew molten-vk + vulkan-loader)
```

Testing needs Xcode (clang, macOS/iOS 26 SDKs). Users need none of this.

## License

MIT. The embedded Khronos Vulkan headers are used under their MIT option.
Both licenses are at the end of `plasmavk.h` and in `LICENSE`.
