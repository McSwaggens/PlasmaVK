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
| `initialization.c` | Initialization: `vkGet*ProcAddr`, instances | 25 |
| `devsandqueues.c` | Devices and Queues: physical devices, properties, features queries, queue families, devices, queues | 34 |
| `memory.c` | Memory Allocation: memory properties, allocation, mapping, flushing, host/device visibility | 27 |
| `resources.c` | Resource Creation: buffers, images, memory requirements, binding, device addresses, subresource layouts, image and buffer views | 29 |
| `extensions.c` | Layers & Extensions: enumeration, extension dependencies | 12 |
| `features.c` | Features > Feature Requirements: one test per required feature | 86 |
| `limits.c` | Limits > Limit Requirements: one test per limit of the Required Limits table | 145 |
| `formats.c` | Formats: one test per format of the Required Format Support tables (features, image types), and the chapter's other rules | 64 |
| `capabilities.c` | Additional Capabilities: image format properties (extents, mip levels, layers, sample counts, monotonicity, host image copy), sparse image formats | 6 |
| `samplers.c` | Samplers: creation with every parameter, maxSamplerAllocationCount | 6 |
| `copies.c` | Copy Commands: buffer/image and image/image copies, blits, resolves, host image copies, texel by texel | 10 |
| `clears.c` | Clear Commands: color and depth/stencil image clears, texel by texel | 4 |
| `descriptorsets.c` | Resource Descriptors > Descriptor Sets: set layouts and support queries, pools and allocation, writes, copies, inline uniform blocks, update templates, dynamic offsets, push descriptors, update after bind, variable-count and partially bound bindings, every descriptor type read by a compute shader | 19 |
| `queries.c` | Queries: pools, host and device resets, timestamp availability, order and period, `vkCmdWriteTimestamp2`, secondaries, results layout with every flag, waiting, unavailable occlusion queries with PARTIAL, `vkCmdCopyQueryPoolResults` (later submission, same command buffer, WAIT) | 15 |
| `synchronization.c` | Synchronization and Cache Control > Events: host and device state, the host seeing a device signal, device waits on host and device signals (same command buffer, batch, earlier submission, several events, synchronization2, device-only events) | 7 |
| `privatedata.c` | Private Data: slots, data on objects of many types, data of new objects and slots, many objects, reserved slots, allocation callbacks | 7 |

Framework: `ct.h` (test macros, helpers), `ct.c` (runner, helpers),
`commands.c` (every Vulkan 1.4 core command and the commands of PlasmaVK's
advertised extensions, for coverage and the `vkGet*ProcAddr` tests),
`baseline.txt` (tests PlasmaVK passes), `baseline-moltenvk.txt` (tests MoltenVK passes).

## Running

```bash
make conformance                  # PlasmaVK (ASan + UBSan), against baseline.txt; part of make check
make conformance-mvk              # MoltenVK (Homebrew molten-vk + vulkan-loader), against baseline-moltenvk.txt
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
  `ct_submit_and_wait`, barriers), images and mapped buffers (`ct_image`,
  `ct_host_buffer`, `ct_image_barrier` with layout transitions) and format
  features (`ct_format_features`).
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
  vkEnumerateDeviceExtensionProperties". For the same reason
  `initialization.gipa_unavailable_device_extension` skips on MoltenVK.
- `devsandqueues.host_image_copy_layouts`: given an array, MoltenVK writes the
  layouts but leaves `copySrcLayoutCount` / `copyDstLayoutCount` as the array
  size instead of the number written, in both
  `VkPhysicalDeviceHostImageCopyProperties` and
  `VkPhysicalDeviceVulkan14Properties`. The specification: "copySrcLayoutCount
  must be set by the application to the number of elements in the
  pCopySrcLayouts array, and on return the variable is overwritten with the
  number of values actually written to pCopySrcLayouts" (and the same for the
  destination list).
- `queries.copy_results_same_commands`, `queries.copy_results_wait`
  (intermittent): `vkCmdCopyQueryPoolResults` in the command buffer that wrote
  the timestamps copies 0 for some of them (with availability 1), after an
  `ALL_COMMANDS` to `TRANSFER` execution dependency on the writes, or with
  `VK_QUERY_RESULT_WAIT_BIT` ("the implementation will wait for each query's
  status to be in the available state before retrieving the numerical
  results"). Copied by a later submission (`queries.copy_results`) they are
  right; under the validation layer, which changes the timing, both pass.
- `synchronization.event_device_state`: in one command buffer, `vkCmdSetEvent`,
  a pipeline barrier with `VK_PIPELINE_STAGE_ALL_COMMANDS_BIT`, then
  `vkCmdResetEvent` leaves the event signaled (and reset, barrier, set leaves
  it unsignaled). The specification (vkCmdResetEvent2) leaves set and reset
  unordered only without such a dependency: "Another execution dependency
  (e.g. a pipeline barrier or semaphore with
  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) is needed to prevent such a race
  condition." In separate submissions both work.
- `privatedata.private_data_slot_destroy_null`: `vkDestroyPrivateDataSlot`
  with `VK_NULL_HANDLE` crashes (`MVKDevice::destroyPrivateDataSlot`), which
  VUID-vkDestroyPrivateDataSlot-privateDataSlot-parameter allows ("If
  privateDataSlot is not VK_NULL_HANDLE, ...").
- `privatedata.private_data_new_objects`: an object created after one with
  private data was destroyed, at the same address, reads that data.
  vkGetPrivateData: "0 will be written in the absence of a previous call to
  vkSetPrivateData using the object specified by objectHandle".

`baseline-moltenvk.txt` lists every other test, which passes on MoltenVK;
`make conformance-mvk` fails if one of those regresses.

`extensions.device_extensions_missing_layer` used to require
`VK_ERROR_LAYER_NOT_PRESENT` for an absent layer and failed through the loader,
which answers `VK_SUCCESS` with no extensions. No "must" supports that
expectation (the command's page only lists the error code), so the test was
corrected to accept either answer, as long as no extensions are reported.

## Tests PlasmaVK skips

Each skip below is an optional capability PlasmaVK does not offer; none hides
a required behaviour.

| Tests | Why |
|---|---|
| `devsandqueues.get_device_queue2_protected`, `features.pipelineProtectedAccess`, `devsandqueues.queue_families_sparse_protected` | Apple GPUs have no protected memory; sparse resources are not implemented. |
| `resources.buffer_capture_replay` | Metal cannot place a resource at a requested GPU address, so `bufferDeviceAddressCaptureReplay` is unsupported. |
| `extensions.instance_layer_extensions` | No layers: PlasmaVK is linked directly, without a loader. |
| `features.shaderInt64` | Required only with 64-bit shader atomics, which Apple7/Apple8 GPUs lack. |
| `features.shaderDrawParameters`, `features.drawIndirectCount`, `features.shaderOutputViewportIndex`, `features.shaderOutputLayer` | Required only with their extensions, which need graphics pipelines (not implemented yet). |
| `features.samplerFilterMinmax`, `formats.filter_minmax_formats`, `samplers.create_sampler_reduction` | Required only with VK_EXT_sampler_filter_minmax, which PlasmaVK offers from Apple10 (M5, A19): Metal's `MTLSamplerDescriptor.reductionMode` is an Apple10 feature (FST "Sampler minimum and maximum reduction"), and on Apple9 (M4 Max) it is accepted and ignored, minimum and maximum sampling as the weighted average. A shader cannot emulate it: a sampler descriptor is only its MTLResourceID. |

Running under the validation layer on MoltenVK reports no valid usage errors
from the tests. Image view and sampler tests avoid component swizzles, format
reinterpretation and LOD bias on devices with `VK_KHR_portability_subset`,
whose features may forbid them (MoltenVK's do). It reports `UNASSIGNED-vkAllocateMemory-maxMemoryAllocationSize`
for `memory.allocate_too_large`, which deliberately exceeds that limit: the
specification allows such an allocation to succeed or to fail with
`VK_ERROR_OUT_OF_DEVICE_MEMORY` (Limits > maxMemoryAllocationSize), and the
test accepts either. It warns (`WARNING-VkDescriptorSetAllocateInfo-maxSet`,
not a valid usage rule) that `descriptorsets.pool_out_of_memory` allocates
more sets than maxSets, which that test does on purpose: Resource Descriptors >
Allocation of Descriptor Sets says such an allocation "may fail", and the test
accepts success or `VK_ERROR_OUT_OF_POOL_MEMORY`. It reports
`VUID-VkComputePipelineCreateInfo-layout-parameter` ("references deleted
object VkDescriptorSetLayout") for `descriptorsets.set_layout_lifetime`,
which creates a pipeline with a pipeline layout whose set layout was destroyed.
That is valid: Fundamentals > Object Lifetime says "a VkDescriptorSetLayout
object passed as a parameter to create another object is not further
accessed by that object after the duration of the command it is passed
into", so the pipeline layout stays a valid handle (the test binds a set of a
second, identically defined layout that is still alive).
`initialization.instance_allocation_failure` crashes
inside the validation layer, which does not survive injected allocation
failures; without the layer it passes.
