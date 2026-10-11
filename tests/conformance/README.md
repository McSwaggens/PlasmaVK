# Conformance tests

PlasmaVK's own Vulkan conformance suite: black-box tests of the public Vulkan
API, each naming the specification requirement it checks. The same sources
run on PlasmaVK and, through the system Vulkan loader, on any driver
(MoltenVK, KosmicKrisp, ...), so a test that fails on PlasmaVK can be checked
against another implementation.

Unlike `tests/smoke`, these tests include `plasmavk.h` without
`PLASMAVK_IMPLEMENTATION` and see nothing but the Vulkan API. On PlasmaVK the
implementation is linked from a separate translation unit
(`tests/compile/impl.c`).

## Layout

One file per chapter of the Vulkan specification:

| File | Chapter | Tests |
|---|---|---|
| `initialization.c` | Initialization: `vkGet*ProcAddr`, instances | 24 |
| `devsandqueues.c` | Devices and Queues: physical devices, properties, features queries, queue families, devices, queues | 31 |
| `memory.c` | Memory Allocation: memory properties, allocation, mapping, flushing, host/device visibility | 27 |
| `resources.c` | Resource Creation: buffers, memory requirements, binding, device addresses | 22 |
| `extensions.c` | Layers & Extensions: enumeration, extension dependencies | 12 |
| `features.c` | Features > Feature Requirements: one test per required feature | 86 |
| `limits.c` | Limits > Limit Requirements: one test per limit of the Required Limits table | 145 |

Framework: `ct.h` (test macros, helpers), `ct.c` (runner, helpers),
`commands.c` (every Vulkan 1.4 core command and the commands of PlasmaVK's
advertised extensions, for coverage and the `vkGet*ProcAddr` tests),
`baseline.txt` (tests PlasmaVK passes).

## Running

```bash
make conformance                  # PlasmaVK (ASan + UBSan), against baseline.txt; part of make check
make conformance-mvk              # MoltenVK (Homebrew molten-vk + vulkan-loader)
make conformance-icd ICD=/path/to/driver_icd.json   # any driver, e.g. KosmicKrisp
make conformance-mvk CONF_ARGS='memory. -v'         # select tests, show all output
```

`CONF_ARGS` goes to the runner (`build/conformance/ct-plasmavk` or
`ct-loader`, which can also be run directly):

- `pattern...`: run tests whose full name (`chapter.test`) matches; a glob if
  it has `*`, `?` or `[`, a substring otherwise.
- `-l`: list tests with their requirement and the commands they use.
- `-c`: command coverage of the selected tests.
- `-b FILE`: compare against a baseline (see below).
- `-v`: show the output of every test; `-j N`: parallel tests (default: the CPU count,
  or `$CT_JOBS`); `-t S`: per-test timeout (default 120 s).
- `--in-process`: run tests in the runner's process, for a debugger.
- `CT_DEVICE_INDEX=n` picks the physical device (default 0).

Each test runs in a new process (the runner re-executes itself with
`--child NAME`), so a crash or a hang is that test's FAIL and the next test
starts clean, with its own instance and device. Not `fork()` alone: Metal's
shader compiler is an XPC service, unreachable from a forked child that did not
exec. Output: one `PASS`/`FAIL`/`SKIP` line per test, with the requirement and
the failed checks under a FAIL, then a summary.

**SKIP** means the device legitimately lacks something optional (a feature,
an extension, a second queue family, a memory type) or the requirement does
not apply to its version. A missing *required* behaviour is a FAIL, including
when a test cannot proceed because a feature its Vulkan version requires is
not reported (e.g. `bufferDeviceAddress` on a 1.3+ device).

## Baseline

`baseline.txt` lists the tests PlasmaVK passes today. With `-b`, the runner
fails only on regressions (a listed test that does not pass), lists tests that
pass but are not listed, and rejects names that no longer exist. Failures
outside the baseline print as `FAIL name (expected)` without details; run with
`-v` (or without `-b`) for those.

When work on PlasmaVK makes tests pass, `make conformance` prints them under
"Newly passing": check them and add the lines to `baseline.txt` by hand. Never
remove a line to silence a regression; fix the regression.

## Adding a test

In the file of the chapter the requirement comes from:

```c
CT_TEST(buffer_size_requirement,
        "Resource Creation > Resource Memory Association: VkMemoryRequirements::size covers "
        "VkBufferCreateInfo::size",
        "vkCreateBuffer vkGetBufferMemoryRequirements vkDestroyBuffer")
{
    struct ct_env e;
    ct_env_init(t, &e);       /* instance + physical device; ends the test on failure */
    ct_env_device(t, &e, NULL); /* device + queue */
    ...
    CT_CHECK(t, req.size >= 1000);                         /* record and continue */
    CT_REQUIREF(t, type != UINT32_MAX, "no host memory");  /* record and end the test */
    if (!optional_thing)
        ct_skip(t, "optional thing not supported");
    ct_env_fini(&e);
}
```

and add `CT_REF(buffer_size_requirement)` to the `tests[]` list at the bottom
of the file (an unlisted test is an unused static and breaks the build).

- **The requirement string** names the chapter and section and the VUID or
  the normative "must" the test derives from, paraphrased. If the requirement
  is derived rather than stated (e.g. from the definition of an error code),
  say so.
- **The commands string** lists every Vulkan command the test *calls*
  (resolving a pointer with `vkGet*ProcAddr` does not count). Names are checked
  against `commands.c` at startup.
- **No invalid usage.** Tests must respect every valid usage rule; a driver may
  do anything with invalid input. Run new tests on MoltenVK under the Khronos
  validation layer:
  `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation make conformance-mvk CONF_ARGS=-v`.
- Helpers in `ct.h`: environments (`ct_env_*`), version gating (`e.version`
  is the version both the instance and the device support), feature and
  property chains pre-filled with a byte pattern (`ct_get_features`,
  `ct_get_properties`; querying twice with different fills detects members an
  implementation does not write), the `VK_INCOMPLETE` protocol
  (`ct_check_incomplete`), a checking allocator with failure injection
  (`ct_allocator_*`, `CT_CALL`), and GPU submission (`ct_begin_commands`,
  `ct_submit_and_wait`, barriers).
- Table-driven chapters (`features.c`, `limits.c`) register one test per row:
  add a row.
- A new chapter file defines `CT_SUITE(name, tests)` and gets one line in
  `ct_register_all()` in `ct.c`.
- SPIR-V: write GLSL, compile once with `glslangValidator -V -x`, paste the
  words as a `static const uint32_t` array with the GLSL in a comment.

## Coverage

`ct -c` reports which commands the selected tests call, against every Vulkan
1.4 core command plus those of VK_KHR_surface, VK_EXT_metal_surface,
VK_KHR_portability_enumeration and VK_KHR_swapchain, and lists the rest. When
PlasmaVK advertises another extension, add its commands to `commands.c`.

## Known deviations of other implementations

Failures on MoltenVK 1.4.2 (Homebrew, with vulkan-loader 1.4.363), checked
against the specification; the tests are right:

- `extensions.device_extension_needs_instance_extension`: MoltenVK lists
  `VK_KHR_swapchain` (and `VK_KHR_present_id`, `VK_KHR_present_wait`,
  `VK_KHR_incremental_present`, `VK_KHR_swapchain_maintenance1`,
  `VK_EXT_hdr_metadata`, `VK_EXT_debug_marker`, ...) for a physical device of
  an instance that did not enable `VK_KHR_surface` / `VK_EXT_debug_report`.
  Layers & Extensions > Extension Dependencies: "Any device extension that has
  an instance extension dependency that is not enabled by vkCreateInstance is
  considered to be unsupported, hence it must not be returned by
  vkEnumerateDeviceExtensionProperties". (PlasmaVK has the same bug with
  `VK_KHR_swapchain`.)
- `extensions.device_extensions_missing_layer`: a loader deviation, not
  MoltenVK's: the loader answers `vkEnumerateDeviceExtensionProperties` with a
  `pLayerName` itself and returns `VK_SUCCESS` and no extensions for a layer
  that does not exist, where it returns `VK_ERROR_LAYER_NOT_PRESENT` for the
  instance query. The expectation is derived from the command's
  `VK_ERROR_LAYER_NOT_PRESENT` result ("a requested layer is not present"),
  not from an explicit "must".

Running under the validation layer on MoltenVK reports no valid usage errors
from the tests. It reports `UNASSIGNED-vkAllocateMemory-maxMemoryAllocationSize`
for `memory.allocate_too_large`, which deliberately exceeds that limit: the
specification allows such an allocation to succeed or to fail with
`VK_ERROR_OUT_OF_DEVICE_MEMORY` (Limits > maxMemoryAllocationSize), and the
test accepts either. `initialization.instance_allocation_failure` crashes
inside the validation layer, which does not survive injected allocation
failures; without the layer it passes.
