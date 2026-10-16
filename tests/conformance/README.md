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
| `initialization.c` | Initialization: `vkGet*ProcAddr` (also of the KHR/EXT names of promoted extensions' commands), instances | 25 |
| `devsandqueues.c` | Devices and Queues: physical devices, properties, features queries, queue families, devices, queues | 34 |
| `cmdbuffers.c` | Command Buffers: resetting pools and command buffers (what they hold afterwards), trimming pools, `vkQueueSubmit2` (batches, binary and timeline semaphores, fences), the device mask | 5 |
| `memory.c` | Memory Allocation: memory properties (also the budget of VK_EXT_memory_budget), allocation, mapping, flushing, host/device visibility, peer memory features | 29 |
| `resources.c` | Resource Creation: buffers, images, memory requirements, binding, device addresses, subresource layouts, image and buffer views, multi-planar images (disjoint planes, linear plane layouts), the subresource layout queries of VK_KHR_maintenance5 on a 1.3 device | 32 |
| `extensions.c` | Layers & Extensions: enumeration, extension dependencies, enabling a layer (with the instance extensions it provides), device layers of an instance with a layer enabled, the device extensions of each layer | 16 |
| `features.c` | Features > Feature Requirements: one test per required feature (also those extensions promoted to core require, in core structures or their own), the format features of VK_EXT_4444_formats | 95 |
| `limits.c` | Limits > Limit Requirements: one test per limit of the Required Limits table | 145 |
| `formats.c` | Formats: one test per format of the Required Format Support tables (features, image types), the required Y'CbCr formats, and the chapter's other rules | 66 |
| `capabilities.c` | Additional Capabilities: image format properties (extents, mip levels, layers, sample counts, monotonicity, host image copy, Y'CbCr descriptor counts), sparse image formats, external memory, semaphore and fence handle types | 10 |
| `samplers.c` | Samplers: creation with every parameter, maxSamplerAllocationCount, sampler Y'CbCr conversions (creation, sampling texel by texel) | 8 |
| `copies.c` | Copy Commands: buffer/image and image/image copies (also of multi-planar image planes), blits, resolves, host image copies, texel by texel; buffer copies and the copies, blits and resolves of Vulkan 1.3 (`vkCmdCopyBuffer2` ...) | 17 |
| `clears.c` | Clear Commands: color and depth/stencil image clears, texel by texel; `vkCmdClearAttachments` (rectangles, layers of a layered framebuffer); `vkCmdUpdateBuffer` | 6 |
| `descriptorsets.c` | Resource Descriptors > Descriptor Sets: set layouts and support queries, pools and allocation, writes, copies, inline uniform blocks, update templates, dynamic offsets, push descriptors, update after bind, variable-count and partially bound bindings, every descriptor type read by a compute shader; push constant updates (`vkCmdPushConstants2`) | 20 |
| `queries.c` | Queries: pools, host and device resets, timestamp availability, order and period, `vkCmdWriteTimestamp2`, secondaries, results layout with every flag, waiting, unavailable occlusion queries with PARTIAL, `vkCmdCopyQueryPoolResults` (later submission, same command buffer, WAIT), occlusion query counts (precise, occluded, no draws, several draws) | 16 |
| `synchronization.c` | Synchronization and Cache Control. Events: host and device state, the host seeing a device signal, device waits on host and device signals (same command buffer, batch, earlier submission, several events, synchronization2, device-only events); timeline semaphore values and host waits; `vkDeviceWaitIdle` | 10 |
| `privatedata.c` | Private Data: slots, data on objects of many types, data of new objects and slots, many objects, reserved slots, allocation callbacks, VK_EXT_private_data on an apiVersion 1.0 instance | 8 |
| `renderpass.c` | Render Pass: load and store operations, render areas, subpasses with an input attachment and a dependency (`vkCreateRenderPass2`, `vkCmdNextSubpass2`), depth and stencil read as input attachments (reference aspect masks), color and depth resolves (multisampled attachments stored or not), multiview, imageless framebuffers, render area granularity (also `vkGetRenderingAreaGranularityKHR` on a 1.3 device), `vkCmdNextSubpass` (inline and secondary contents, preserved attachments); dynamic rendering: color and depth, resolves, suspending and resuming across command buffers, multiview, local read (attachment locations, input attachment indices, depth and stencil input attachment indices) | 17 |
| `pipelines.c` | Pipelines: no fragment shader, specialization constants (and operations on them), a uniform buffer through a descriptor set; dynamic state (viewports and scissors with and without counts, cull mode, front face, topology, depth and stencil state, rasterizer discard, depth bias); pipeline caches (header, `VK_INCOMPLETE`, partial data, round trip, merging), `FAIL_ON_PIPELINE_COMPILE_REQUIRED` with `EARLY_RETURN_ON_FAILURE`, creation feedback | 13 |
| `drawing.c` | Drawing Commands and Fixed-Function Vertex Processing: vertex and instance numbering of direct, indexed (UINT16, UINT32, UINT8) and indirect draws, draw counts, primitive restart (static and dynamic), every primitive topology, draw parameters (BaseVertex, BaseInstance, DrawIndex), vertex attribute formats, offsets and instance rates, divisors, dynamic binding strides; `vkCmdBindIndexBuffer2` ranges; the KHR aliases of VK_KHR_maintenance5 and VK_KHR_draw_indirect_count on a 1.3 device | 15 |
| `rasterization.c` | Fixed-Function Vertex Post-Processing and Rasterization: viewport transform (offsets, negative heights, depth ranges), clipping and depth clamping, ViewportIndex and Layer from the vertex shader, culling, rasterizer discard, coverage at the standard sample locations, sample shading, wide points, flat and smooth interpolation, dynamic line width and line stipple | 13 |
| `fragops.c` | Fragment Operations: scissor test, every depth compare op, every stencil op (clamping, wrapping, fail and depth-fail ops, write and compare masks), depth bounds (static and dynamic), sample mask (alone, and with the SampleMask built-ins), alpha to one | 8 |
| `framebuffer.c` | The Framebuffer: blend factors and operations (constants static and dynamic), color write masks, independent blending, dual-source blending, sRGB blending in linear space, logic operations (XOR on UNORM, every operation on UINT with a write mask, the bits of SNORM) | 8 |
| `dispatch.c` | Dispatching Commands: `vkCmdDispatchIndirect` (parameters read at execution, zero counts), `vkCmdDispatchBase` (WorkgroupId, NumWorkgroups, GlobalInvocationId), no base carried over | 3 |
| `sparsemem.c` | Sparse Resources: memory requirements of sparse buffers and images (none without sparse residency), sparse image format properties and standard block shapes, mip tail rules; `vkQueueBindSparse` on a sparse binding queue: buffer blocks (residency, rebinding, unbinding, aliasing), image blocks, mip tails, fully resident images bound opaquely, 3D images, the order of batches; unbound regions where `residencyNonResidentStrict` | 14 |
| `wsi.c` | Window System Integration: Metal surfaces, surface queries (support, capabilities, formats, present modes, present rectangles, device group modes), swapchains (images, allocation callbacks, `VK_ERROR_NATIVE_WINDOW_IN_USE_KHR`, recreation with `oldSwapchain`, aliasing a swapchain image), acquire and present (semaphores, fences, timeouts, two swapchains in one present), swapchain images as transfer source and destination | 20 |
| `debugging.c` | Debugging > Active Tooling Information: `vkGetPhysicalDeviceToolProperties` (count protocol, strings); Debug Utilities (VK_EXT_debug_utils): messenger filtering by severity and type, callback data passed through, destruction, allocation callbacks, messengers chained to `VkInstanceCreateInfo` (active only in `vkCreateInstance` and `vkDestroyInstance`), names of objects of every type (also instance-level ones and a messenger) and tags, queue labels, command buffer labels (nested, across render pass instances, command buffers of a batch and submissions, in secondaries), which `vkGet*ProcAddr` returns its commands | 11 |

Framework: `ct.h` (test macros, helpers; for the graphics chapters
offscreen render targets, a pipeline description with defaults, rectangle
shaders and texel checks), `ct.c` (runner, helpers),
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
VK_KHR_portability_enumeration, VK_KHR_swapchain, VK_EXT_debug_utils and the
extensions promoted to Vulkan 1.1-1.4 (their commands are the core ones
under KHR/EXT names, which the tests mostly reach through `vkGet*ProcAddr`
rather than call), and lists the rest (VK_EXT_swapchain_colorspace has no
commands). When PlasmaVK advertises another extension, add its commands to
`commands.c` (and, for a device extension, its name to
`ct_origin_is_device_extension`; for a command vk.xml adds only with another
extension or a core version, a row to `ct_command_condition`).
VK_EXT_debug_utils is an instance extension with device-level commands
(object names, labels): the `vkGet*ProcAddr` tests treat them as instance
extension commands, which `vkGetDeviceProcAddr` returns when the instance
enabled the extension.

The layer tests of `extensions.c` enable the first listed of the layers known
to need no configuration (`VK_LAYER_PLASMAVK_debug`, PlasmaVK's built-in
layer, or through a loader `VK_LAYER_KHRONOS_validation`) and skip if none is
listed: enabling arbitrary layers (capture, API dump) would change the test's
process.

`wsi.c` makes its CAMetalLayers through the Objective-C runtime (plain C,
`.linker_option` for QuartzCore and libobjc, so neither build needs link
flags). The layers are in no window: CAMetalLayer vends drawables all the
same, so the chapter runs headless.

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

- `wsi.native_window_in_use_other_surface`: with a swapchain on one
  VkSurfaceKHR of a CAMetalLayer, MoltenVK creates a second swapchain for
  another VkSurfaceKHR of the same layer (it refuses one for the same
  surface). vkCreateSwapchainKHR: "If oldSwapchain is VK_NULL_HANDLE, and the
  native window referred to by pCreateInfo->surface is already associated with
  a Vulkan swapchain, VK_ERROR_NATIVE_WINDOW_IN_USE_KHR must be returned"; on
  the Metal platform the native window is the layer, which both surfaces refer
  to.
- `drawing.primitive_restart`: with primitive restart disabled (dynamically,
  `vkCmdSetPrimitiveRestartEnable(VK_FALSE)`), index 0xFFFF of a
  `VK_INDEX_TYPE_UINT16` triangle strip still restarts the strip instead of
  naming vertex 65535 (Metal always restarts strips on the maximum index).
  VkPipelineInputAssemblyStateCreateInfo: "primitiveRestartEnable controls
  whether a special vertex index value is treated as restarting the assembly
  of primitives". With restart enabled the test passes.
- `dispatch.dispatch_indirect_after_base`: after `vkCmdDispatchBase` with a
  non-zero base, a `vkCmdDispatchIndirect` in the same command buffer keeps
  that base: its WorkgroupId values start at the earlier base instead of 0
  (`vkCmdDispatch` after the same command starts at 0). vkCmdDispatchIndirect
  "behaves similarly to vkCmdDispatch except that the parameters are read by
  the device from a buffer during execution", and "vkCmdDispatch is
  equivalent to vkCmdDispatchBase(0,0,0,groupCountX,groupCountY,groupCountZ)".
- `debugging.debug_utils_messenger_allocator`: `vkDestroyDebugUtilsMessengerEXT`
  given the allocation callbacks the messenger was created with (one
  allocation through them) calls their `pfnFree` twice: once for that
  allocation, once with memory `pfnAllocation` did not return (in the
  loader's `debug_utils_DestroyDebugUtilsMessengerEXT`). PFN_vkFreeFunction:
  "If pMemory is non-NULL, it must be a pointer previously allocated by
  pfnAllocation or pfnReallocation."
- `debugging.debug_utils_object_name_messenger`: `vkSetDebugUtilsObjectNameEXT`
  naming a `VkDebugUtilsMessengerEXT` (`VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT`)
  crashes in `MVKVulkanAPIObject::setDebugName`: the loader passes MoltenVK
  the handle it gave the application, which is the loader's own object. The
  call is valid: VUID-VkDebugUtilsObjectNameInfoEXT-objectType-02590 asks for
  "a valid Vulkan handle of the type associated with objectType as defined in
  the VkObjectType and Vulkan Handle Relationship table", which pairs
  `VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT` with `VkDebugUtilsMessengerEXT`.
  Every other object type is named without trouble
  (`debugging.debug_utils_object_names`).

The graphics chapters' tests also skip on MoltenVK where it lacks a feature
(`drawIndirectCount`, `depthBounds`, `logicOp`: the logic operation tests), and draw no triangle fans on
it (`VK_KHR_portability_subset`, whose `triangleFans` may be off); and
without `VK_KHR_draw_indirect_count`, `drawing.draw_indirect_count_khr`.
Sparse resources, wide lines, stippled lines, depth bounds and external
handle types are not offered by MoltenVK either: their tests check what is
valid without them (sparse requirements of non-sparse images, line width
1.0, stippling disabled, the depth bounds test disabled, the handle type
queries) and skip the rest.

`baseline-moltenvk.txt` lists every other test, which passes on MoltenVK;
`make conformance-mvk` fails if one of those regresses.

`extensions.device_extensions_missing_layer` used to require
`VK_ERROR_LAYER_NOT_PRESENT` for an absent layer and failed through the loader,
which answers `VK_SUCCESS` with no extensions. No "must" supports that
expectation (the command's page only lists the error code), so the test was
corrected to accept either answer, as long as no extensions are reported.

## Tests PlasmaVK skips

Each skip below is an optional capability PlasmaVK does not offer, or a check
of the case without a capability PlasmaVK has; none hides a required
behaviour.

| Tests | Why |
|---|---|
| `devsandqueues.get_device_queue2_protected`, `features.pipelineProtectedAccess` | Apple GPUs have no protected memory. |
| `memory.peer_memory_features` | Every device group has one physical device: `vkGetDeviceGroupPeerMemoryFeatures` has no valid call (its local and remote device indices must differ). |
| `resources.buffer_capture_replay` | Metal cannot place a resource at a requested GPU address, so `bufferDeviceAddressCaptureReplay` is unsupported. |
| `features.shaderInt64`, `features.shaderBufferInt64Atomics` | Required only with 64-bit shader atomics (VK_KHR_shader_atomic_int64), which Apple7/Apple8 GPUs lack. |
| `features.ycbcr2plane444Formats` | VK_EXT_ycbcr_2plane_444_formats is not offered: of the four formats its feature promises, PlasmaVK has only the 8-bit one. |
| `features.samplerFilterMinmax`, `formats.filter_minmax_formats`, `samplers.create_sampler_reduction` | Required only with VK_EXT_sampler_filter_minmax, which PlasmaVK offers from Apple10 (M5, A19): Metal's `MTLSamplerDescriptor.reductionMode` is an Apple10 feature (FST "Sampler minimum and maximum reduction"), and on Apple9 (M4 Max) it is accepted and ignored, minimum and maximum sampling as the weighted average. A shader cannot emulate it: a sampler descriptor is only its MTLResourceID. |

## Tests PlasmaVK fails

None: every test that does not skip is in `baseline.txt`.

Running under the validation layer on MoltenVK reports no valid usage errors
from the tests. (MoltenVK has no sparse features, so the sparse binding tests
of `sparsemem.c` skip there; the validation layer cannot run on PlasmaVK,
which has no loader.) On PlasmaVK the suite also runs under its own debug
layer, `PLASMAVK_DEBUG=1 make conformance`, whose checks report nothing; they
found that the sparse binding tests ended their command buffers twice
(`vkEndCommandBuffer` before `ct_submit_and_wait`, which ends them:
VUID-vkEndCommandBuffer-commandBuffer-00059), since fixed.
Image view and sampler tests avoid component swizzles, format
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
failures; without the layer it passes. `debugging.tool_properties` fails under
the validation layer, which reports itself as an active tool but answers a
count of 0 with a non-NULL array by setting the count to 1 and returning
`VK_SUCCESS`, where Fundamentals > Array Results requires that only as many
elements be written as the count allows, the count be overwritten with the
number written, and `VK_INCOMPLETE` be returned; without the layer there are no
tools and it passes. `initialization.gdpa_disabled_device_extension` and
`debugging.debug_utils_proc_addr` fail under the validation layer, which
provides VK_EXT_debug_utils itself and returns its label commands from
`vkGetDeviceProcAddr` although the instance did not enable the extension
("any other case" of the vkGetDeviceProcAddr table: `NULL`); without the layer
both pass. With synchronization validation as well
(`VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true`), the only further reports are
`SYNC-HAZARD-READ-AFTER-WRITE` for `synchronization.event_wait_device_signal`,
whose copies wait (vkCmdWaitEvents) on events set by fills in another command
buffer, a dependency synchronization validation does not follow.
