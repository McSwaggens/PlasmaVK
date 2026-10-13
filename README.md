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

**Status:** every Vulkan 1.4 command is implemented on Metal 4 (no stubs left),
along with the extensions PlasmaVK advertises: surfaces and swapchains on
CAMetalLayer, debug utils, and, as Vulkan 1.4 drivers list them, the
extensions promoted to Vulkan 1.1-1.4 whose features it has (sampler min/max
filtering only on Apple10 GPUs), with their KHR/EXT command names; and a
built-in debug layer (below). Sparse resources (sparse binding; partially resident buffers and
2D, 3D and multisampled images; aliasing) map onto Metal 4's placement sparse
buffers and textures (macOS/iOS 26.4). SPIR-V compute, vertex and fragment
shaders are translated to MSL inside the header, including subgroups,
robustness, Y'CbCr sampling, sparse residency codes and buffer device
addresses. PlasmaVK passes its own black-box conformance suite
(`tests/conformance/`, every Vulkan 1.4 command covered) with no failures;
the only skipped tests need hardware Apple GPUs lack, like protected memory,
multi-GPU device groups or 64-bit shader atomics. On an M4 Max (macOS 27.2) it
passes every must-pass case of the Khronos CTS 1.4.7.0 except
`api.driver_properties.driver_id_match`, which needs a driver ID registered
with Khronos. It has not been submitted for Khronos conformance, and
performance work has only started: shaders without float controls compile in
Metal's relaxed math mode (Vulkan's default floating-point environment), the
others in safe math.

## Debugging

There is no loader, so the Khronos layers can't run on PlasmaVK; it has its
own instead. `VK_EXT_debug_utils` is always offered: messengers receive
PlasmaVK's diagnostics, and object names and label regions become Metal labels
and debug groups in Xcode's GPU capture. The built-in layer
`VK_LAYER_PLASMAVK_debug` (enable it in `ppEnabledLayerNames`, or set
`PLASMAVK_DEBUG=1` for an application you can't change) prints every
diagnostic, names Metal debug groups after each command buffer, render pass,
draw, dispatch and transfer, and checks common errors (destroyed handles,
command buffer states, mismatched allocation callbacks, missing features,
leaks), reporting them with their VUIDs. The header comment's DEBUGGING part
has the details.

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
