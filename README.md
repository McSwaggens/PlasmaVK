# PlasmaVK

Vulkan on Apple Metal in one C99 header. No loader, no library to build and
no dependencies.

```c
// In exactly one .c file:
#define PLASMAVK_IMPLEMENTATION
#include "plasmavk.h"
```

Everywhere else, include `plasmavk.h` and call `vk*` as usual. You don't need
linker flags: the header links Metal itself.

**Status:** early scaffold. Instance, physical device, device and queue
creation work. Every other Vulkan 1.4 command exists but logs "not
implemented". The target is full Vulkan 1.4 plus the Roadmap 2026 profile,
measured by the Khronos conformance suite.

## Requirements

- macOS 26+ or iOS 26+ on a Metal 4 GPU (Apple M1 / A14 or newer).
- The implementation must be compiled as C, or as Objective-C without ARC.

## Repository layout

| Path | What |
|---|---|
| `plasmavk.h` | **Generated** single header, committed. This is what users take. |
| `src/` | The implementation, split by subsystem. `src/plasmavk.h` is the root. |
| `src/gen/` | **Generated** from `vk.xml`: prototypes, extension tables, stubs, proc-addr table. |
| `src/pvk_extensions.txt` | Extensions PlasmaVK advertises. |
| `tools/` | C99 dev tools: `gen_vk.c` (vk.xml codegen) and `amalgamate.c` (builds the single header). |
| `external/vulkan-headers/` | Pinned Khronos headers and `vk.xml` (1.4.363), embedded into `plasmavk.h`. |
| `tests/smoke/` | Fast C99 tests. Each one includes the header with no link flags. |
| `tests/compile/` | Compile and link checks: C dialects, C++, ObjC, iOS, two translation units. |
| `tests/cts/` | Khronos conformance suite (VK-GL-CTS) integration (planned). |
| `bench/` | Realistic-workload benchmarks, comparable against MoltenVK and KosmicKrisp. |

## Development

```bash
make          # regenerate src/gen/ and plasmavk.h
make test     # smoke tests against src/ (ASan + UBSan)
make check    # everything a user hits, against the amalgamated plasmavk.h
make bench    # benchmarks on PlasmaVK
make bench-mvk  # the same benchmarks on MoltenVK (Homebrew molten-vk + vulkan-loader)
```

Development needs only Xcode (clang, macOS/iOS 26 SDKs); the generators are
C99 tools built by `make`. Users need none of this.

## License

MIT. The embedded Khronos Vulkan headers are used under their MIT option.
