# PlasmaVK

Single-header, dependency-free C99 Vulkan implementation on Metal 4 (macOS/iOS 26+).
Users `#define PLASMAVK_IMPLEMENTATION` + `#include "plasmavk.h"`. No loader, no dylib, no ICD mode.

## Rules
- `plasmavk.h` is the only source file. Edit it directly. Its header comment lists the numbered sections.
- No tools, generators, build steps or scripts of any language in the repo (no Python, no codegen).
  Everything in `plasmavk.h` is hand-written or pasted in (Vulkan headers, Metal declarations).
  The exception is small shell scripts for running work on the dev machine, like `bench/gpulock.sh`.
- C99 only (`-std=c99 -pedantic -Werror`). No Objective-C, no `-fblocks`, no third-party code.
- Never require link flags: frameworks are autolinked via `.linker_option` (section 3).
- All implementation symbols are `static` except the `vk*` entry points.
- Code may only call `vk*` functions defined above it (keeps `VK_NO_PROTOTYPES` builds working).
- Adding a command (new extension): list the extension in section 6, write the command in its subsystem
  section, and give it an entry in section 19's table (alphabetical; scope, core version, extension).
- Metal: names are `pvk_`/`PVK_`-prefixed (`pvk_MTLSize`, `PVK_MTLPixelFormatRGBA8Unorm`) so an ObjC TU
  importing `<Metal/Metal.h>` doesn't clash. Add message wrappers by hand in section 4 following the
  existing pattern, matching Apple's SDK declaration exactly (object pointers -> `id`, blocks -> `void *`
  holding a `struct pvk_block *`, `NSUInteger` -> `unsigned long`, `BOOL` -> `bool`). New structs get a
  `PVK_ASSERT_SIZE`.
- Tests in the repo are our own (the Khronos CTS runs from an external checkout, never vendored). SPIR-V test inputs are compiled once from GLSL with the Vulkan SDK's
  `glslangValidator` and pasted as `uint32_t` arrays with the GLSL source in a comment beside them.

## Verify
- The Mac is shared by several sessions and agents, and timings are only valid when nothing else runs.
  Wrap work in `~/bin/gpulock` (it waits for the lock, then runs the command): the main checkout's
  `bench/gpulock.sh`, one lock (`~/.gpulock`) for every codebase on this Mac, the engine's too. Call it by that
  path, not a worktree's copy, which may be older. Exclusive runs queue first come, first served.
  - `~/bin/gpulock shared make -j32 check`: builds and tests; any number at once.
  - `~/bin/gpulock excl CMD`: benchmarks, profiles, anything timed or GPU-heavy; runs alone.
  - `~/bin/gpulock cts deqp-vk ...`: the CTS; one at a time.
  - `~/bin/gpulock status`: who holds or waits. Exit code 75 means the wait timed out: retry the same command
    from the same directory within 2 minutes and an exclusive run keeps its place in the queue.
- `make -j8 test`: smoke tests (ASan/UBSan), in parallel.
- `make -j8 check`: all C dialects, ObjC with/without `<Metal/Metal.h>`, `VK_NO_PROTOTYPES`, C++, two-TU link, iOS, smoke tests, then the conformance suite (`tests/conformance`).
