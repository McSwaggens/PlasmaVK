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

**Status:** early. Instance, physical device, device and queue creation work.
Every other Vulkan 1.4 command exists but logs "not implemented". The target
is full Vulkan 1.4 plus the Roadmap 2026 profile, measured by the Khronos
conformance suite.

## Requirements

- macOS 26+ or iOS 26+ on a Metal 4 GPU (Apple M1 / A14 or newer).
- The implementation must be compiled as C, or as Objective-C without ARC.

## Repository layout

| Path | What |
|---|---|
| `plasmavk.h` | The library. It is the only source file and is edited directly; its header comment lists its sections. |
| `tests/smoke/` | Fast C99 tests. Each one includes the header with no link flags. |
| `tests/compile/` | Compile and link checks: C dialects, C++, ObjC (with and without `<Metal/Metal.h>`), iOS, two translation units. |
| `tests/cts/` | Khronos conformance suite (VK-GL-CTS) integration (planned). |
| `bench/` | Realistic-workload benchmarks, comparable against MoltenVK and KosmicKrisp. |

## Testing

```bash
make test       # smoke tests (ASan + UBSan)
make check      # everything a user hits, then the smoke tests
make bench      # benchmarks on PlasmaVK
make bench-mvk  # the same benchmarks on MoltenVK (Homebrew molten-vk + vulkan-loader)
```

Testing needs Xcode (clang, macOS/iOS 26 SDKs). Users need none of this.

## License

MIT. The embedded Khronos Vulkan headers are used under their MIT option.
Both licenses are at the end of `plasmavk.h` and in `LICENSE`.
