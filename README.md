# PlasmaVK

Vulkan 1.4 on Apple Metal in one C99 header. No loader, no library, no build
step, no dependencies: the Vulkan API headers are inside `plasmavk.h`, and it
links Metal itself.

```c
// In exactly one .c file:
#define PLASMAVK_IMPLEMENTATION
#include "plasmavk.h"
```

Everywhere else, include `plasmavk.h` and call `vk*` as usual.

## Status

- Every Vulkan 1.4 command, on Metal 4. SPIR-V compute, vertex and fragment
  shaders are translated to MSL inside the header.
- Surfaces and swapchains on `CAMetalLayer`, headless surfaces,
  `VK_EXT_debug_utils`, the extensions promoted to Vulkan 1.1-1.4, and sparse
  resources (macOS/iOS 26.4+).
- Passes every must-pass test of the Khronos CTS 1.4.7.0 except
  `api.driver_properties.driver_id_match`, which needs a driver ID registered
  with Khronos. Not submitted for Khronos conformance.

## Requirements

- macOS 26+ or iOS 26+ on Apple M1 / A14 or newer.
- Compile the implementation as C, or as Objective-C without ARC.

## Options

- `PLASMAVK_QUAD_MERGING`: define it before `PLASMAVK_IMPLEMENTATION` to let
  the GPU pack two triangles into one 2x2 quad, as MoltenVK does. Shaders that
  use derivatives run faster, but derivatives along shared edges may mix both
  triangles. Not conformant: `conformanceVersion` reports 0.0.0.0.
- `VK_LAYER_PLASMAVK_debug` (or `PLASMAVK_DEBUG=1`): the built-in debug layer,
  since there is no loader for the Khronos layers. It checks common errors,
  reporting their VUIDs, and names Metal debug groups for Xcode's GPU capture.

The comment at the top of `plasmavk.h` documents the rest.

## Testing

```bash
make test    # smoke tests (ASan + UBSan)
make check   # compile checks, smoke tests and the conformance suite
make bench   # benchmarks; make bench-mvk runs them on MoltenVK
```

Needs Xcode with the macOS and iOS 26 SDKs. See
[`tests/conformance/`](tests/conformance/README.md) and
[`bench/`](bench/README.md) for more.

## License

MIT. The embedded Khronos Vulkan headers are used under their MIT option.
