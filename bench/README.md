# PlasmaVK benchmarks

Headless programs that time Vulkan in realistic scenarios. Each one builds
two ways from the same source:

- PlasmaVK compiled in (default), the way users ship it.
- `-DBENCH_USE_LOADER`, which uses the system Vulkan loader, so the same
  workload runs on MoltenVK or KosmicKrisp.

Each benchmark checks its own output once per configuration. It reads the
result back and compares it with a CPU reference, so a broken implementation
reports an error instead of fast-but-wrong numbers.

| Program | What it measures | Watch |
|---|---|---|
| `bench_frame` | A whole engine-like frame: streaming, shadows, G-buffer, compute, lighting, bloom and UI, recorded by one thread or four | `cpu_us`, `frame_us`, `gpu_us` |
| `bench_present` | Presenting to a real `CAMetalLayer` (drawables), against a headless surface | `present_gpu_us`, `present_us` |
| `bench_startup` | What an application pays before its first frame, in fresh processes | `to_first_frame_us` and its phases |
| `bench_draw_calls` | Recording and submitting thousands of small draws, and the GPU side of per-draw data | `cpu_ns_per_draw` |
| `bench_indirect` | One indirect draw call of a GPU-driven renderer, with and without a count buffer | `cpu_us`, `gpu_us` |
| `bench_descriptor_churn` | Delivering per-draw descriptors in each way an engine might, and managing pools and sets | `cpu_ns_per_draw`, `ns_per_op` |
| `bench_resource_churn` | Creating and destroying buffers, images, views and samplers every frame | `ns_per_op`, `resource_us_per_frame` |
| `bench_upload_streaming` | Per-frame uploads of constants, geometry and textures | `cpu_us`, `cpu_gb_per_s` |
| `bench_pipeline_compile` | Pipeline creation: the stutter of each call, with and without caches | `create_us` p99 and max, `total_ms` |
| `bench_render_passes` | A deferred frame whose variants change what leaves tile memory | `gpu_us` |
| `bench_render_area` | Render areas smaller than their attachments | `gpu_us` |
| `bench_transfers` | Clears, blit mip chains, atlas blits, fills, an upscale and a resolve, as frames and level loads record them | `gpu_us` |
| `bench_compute` | Small dispatches with and without barriers, and a bandwidth-bound prefix sum | `cpu_ns_per_dispatch`, `gelements_per_s` |
| `bench_shaders_gpu` | Full-screen passes whose cost is their fragment shader's own code | `gpu_us` |

## Configurations

Each benchmark runs all of its configurations unless `BENCH_FILTER` picks
some. The comment at the top of each source file has the details.

`bench_frame`: one deferred frame of 7613 draws in 18 rendering instances.
Per frame it streams object constants, UI geometry and texture tiles,
allocates its descriptor sets from a per-frame pool, renders 4 shadow
cascades (a depth-only pipeline, 1000 draws each) and a G-buffer (3000
draws, 4 pipelines, 64 materials, a dynamic offset per draw), culls 256
lights into 16x16-pixel tiles in compute, lights the frame with shadow
lookups, runs 11 bloom passes, tonemaps, and makes 600 UI draws with a
scissor each.
- `st_1080p`: one thread records everything.
- `mt4_secondary_1080p`: 4 worker threads record the cascades and the
  G-buffer as secondaries that the primary executes.
- `mt4_primary_1080p`: 4 worker threads record 4 primaries, submitted
  together, as D3D12-style renderers do.
- `st_profiled_1080p`: timestamps around every pass, read a frame later
  like an engine's GPU profiler.
- `st_bgcompile_1080p`: a background thread creates G-buffer pipelines with
  fresh specialization constants meanwhile.
- `st_2160p`: the frame at 3840x2160, GPU-bound.
- `st_dynamic_depth_bounds_1080p`: a device with `depthBounds` enabled whose
  depth pipelines make the bounds test dynamic (and turn it off), as
  translation layers and extended-dynamic-state engines do. Skipped without
  `depthBounds`.

`bench_present`: the frame loop of a vsynced game at 60 frames per second.
Each frame renders 2000 rectangles into an RGBA16F target, tonemaps it into
the swapchain image and presents.
- `layer_1080p`, `layer_2160p`, `layer_2880p`: a swapchain on a
  `CAMetalLayer` through `VK_EXT_metal_surface`. The layer is in no window,
  so nothing is shown, but presents take the windowed path. A
  configuration whose layer stops handing out drawables (the display
  asleep) is skipped.
- `headless_1080p`, `headless_2160p`, `headless_2880p`: the same frames
  presented to a `VK_EXT_headless_surface`, the control.

`bench_startup`:
- `process_first`: a fresh process per iteration (`posix_spawn` of the
  benchmark itself) creates an instance, a device, a headless surface and
  swapchain, then presents one cleared frame and waits for it. The first of
  two unmeasured launches also checks a pixel of its frame.
- `process_first_cg`: the same, with CoreGraphics initialized first, as in
  any windowed application.
- `instance_device`: instance and device creation and teardown repeated in
  one process.

`bench_draw_calls`: small draws into a 1024x1024 target.
- `push_constants_1000`, `_10000`, `_50000`: per draw `vkCmdPushConstants`
  and `vkCmdDraw`.
- `indexed_vb_10000`: per draw a vertex buffer bind (one of 64 meshes), push
  constants and `vkCmdDrawIndexed`.
- `pipeline_switch_10000`: push constants and a different pipeline (one of 4
  materials) every 8 draws.
- `instanced_50000_x64`, `instanced_50000_x1024`: 50000 objects in
  instanced batches of 64 or 1024.
- `rebind_10000`: per draw the object's whole state again, as renderers that
  do not track state send it: viewport, scissor, the same pipeline, push
  constants, and its mesh's vertex and index buffers (one of 64).
- `secondary_x4_10000`, `secondary_x1_2000`: `rebind`'s draws in 4 large
  secondaries, or one per object as in Sascha Willems' multithreading
  sample, all recorded on the submitting thread.
- `vs_heavy_1000`: 1000 draws of 6144 vertices, all but six clipped away,
  so the GPU time is vertex shading that reads per-draw data.

`bench_indirect`: one indirect call per frame, from a record buffer sized
for 65536 objects. Needs `multiDrawIndirect` and `drawIndirectFirstInstance`;
the count configurations also need `drawIndirectCount` and are skipped
without it.
- `mdi_1000`: `vkCmdDrawIndirect` of 1000 compacted records.
- `mdi_1000_of_16384`: 16384 records whose culled ones have `instanceCount`
  0 (culling without compaction).
- `count_1000_cap1000`, `count_1000_cap16384`, `count_1000_cap65536`,
  `count_10000_cap16384`: `vkCmdDrawIndirectCount` of 1000 or 10000 counted
  records with `maxDrawCount` up to the buffer's capacity, as engines pass
  it.
- `icount_1000_cap16384`: the same with `vkCmdDrawIndexedIndirectCount`.

`bench_descriptor_churn`: 10000 draws a frame, each with its own uniform
buffer and 4 textures.
- `bind_static_10000`: a pre-built set per object, bound.
- `update_sets_10000`: `vkAllocateDescriptorSets` and
  `vkUpdateDescriptorSets` per draw, the pool reset per frame.
- `update_template_10000`: the same with update templates.
- `dynamic_offset_10000`: a set per material, the uniform buffer at a
  dynamic offset per draw.
- `push_descriptor_10000`, `push_template_10000`: push descriptors, plain
  and with a template. Skipped without `pushDescriptor`.
- `multi_frequency_10000`: sets by update frequency: set 0 per frame, set 1
  per material (every 8 draws), set 2 per object with a dynamic offset.
- `update_sets_mt4_10000`: `update_sets` recorded by 4 threads into
  secondaries, each from its own descriptor and command pools sharing one
  set layout.
- `pool_free_churn_4096`: a `FREE_DESCRIPTOR_SET` pool kept full with 4096
  sets; a random set is freed and another allocated.
- `pool_alloc_batch_256`: one allocation of 256 sets, then a pool reset.
- `pool_create_256`: a pool of 256 sets created and destroyed.
- `array_write_1024`, `array_write_65536`: one-element writes at random
  indices of a bindless texture table (update after bind, partially bound).
  Skipped where the device does not support a table of that size.
- `pool_threads_1`, `_2`, `_4`, `_8`: each thread allocates and writes 256
  sets from its own pool, then resets it.
- `update_sets_mt4_named_10000`, `pool_threads_8_named`: the same after one
  object (the sampler) has been given a `VK_EXT_debug_utils` name, as
  development builds of engines name everything. The instance enables the
  extension for every configuration where it is offered; these two run
  last.

`bench_resource_churn`: resources created, used and destroyed a few frames
later, as streaming engines and render graphs do; every frame's new views
also go into a descriptor array.
- `textures_dedicated`: 8 BC7 textures a frame (128 to 1024, full mips),
  each with memory of its own, a view, and every mip uploaded.
- `textures_suballoc`: the same bound at offsets of large blocks.
- `textures_suballoc_ws4k`: the same with 4000 other textures alive.
- `buffers_dedicated`, `buffers_dedicated_256k`: 16 staging buffers a frame
  (4 to 64 KB, or 64 to 512 KB) written and copied into 16 device-local
  ones, each with memory of its own.
- `buffers_suballoc`: `buffers_dedicated` bound at offsets of large blocks.
- `rendertargets_aliased`: a render graph's 6 transient 1280x720 targets
  created each frame in a memory block reused every 3 frames, cleared in one
  render pass instance.
- `views_x128`: 128 single-mip, single-layer views a frame of a persistent
  texture.
- `samplers_x16`: 16 samplers a frame.
- `load_oneshot_x4`: load-time uploads, each with its own staging buffer
  and a command buffer submitted and waited for.
- `bufferviews_x64`: 64 texel buffer views a frame.
- `textures_suballoc_named`, `rendertargets_aliased_named`: the same with a
  `VK_EXT_debug_utils` name on every new resource, run last as in
  `descriptor_churn`.

`bench_upload_streaming`: CPU time includes writing the data.
- `buffer_4k_x256`: 256 4 KB updates through a staging ring, a copy each.
- `buffer_bulk_32m`: one 32 MB copy.
- `image_tiles_x64`: 64 128x128 tiles into an atlas.
- `image_mips_2048`: a 2048x2048 texture with its 12 mips.
- `direct_4k_x256`: the 256 updates written straight into host-visible
  device-local memory.
- `map_flush_4k_x256`: map, write, flush each range and unmap per frame,
  into non-coherent memory where there is some, then the copies.

`bench_pipeline_compile`: each `vkCreate*Pipelines` call is timed on its
own. The first phases create 256 graphics and 128 compute pipelines of
distinct code, each phase as `<phase>_graphics` and `<phase>_compute`:
- `cold`: a fresh device and an empty `VkPipelineCache`.
- `warm_cache`: a fresh device whose cache starts from the cold phase's
  `vkGetPipelineCacheData`.
- `no_cache`: a fresh device without a cache.
- `cold_parallel4`: 4 threads sharing one cache.
- `batch_cold`: every pipeline of a kind in one call.

The material phases create pipelines the way engines do: a textured
material shader, few shaders and many states, one call per pipeline and no
cache.
- `permutations_cold`: 16 shader pairs (2 vertex shaders, 8 fragment
  variants) times 32 states (blend, cull, depth attachment, color format,
  write mask), 512 pipelines on a fresh device. Each pipeline is counted in
  the first bucket that names something no earlier pipeline had: a new
  fragment variant, a new vertex shader, a new output state of a known
  variant (Apple GPUs build blending and output conversion into the
  fragment binary), a new pairing of known shaders, or a repeat.
- `permutations_recreate`: all 512 destroyed and created again on the same
  device.
- `permutations_warm`: a fresh device, the same shaders, with whatever the
  OS shader cache kept.
- `spec_variants_cold`: 16 pipelines that differ only in specialization
  constants.

The cache phases keep an application `VkPipelineCache` the way engines that
save it as it grows do (Godot 4 after every pipeline): after each creation,
under the application's own lock, a size query, and a save whenever the
data has grown by 3 MB. 32 shader pairs (2 vertex shaders, 16 fragment
variants) times 64 states (blend, color format, write mask), 2048
pipelines.
- `cache_polled_cold`: a fresh device and an empty cache.
- `cache_polled_warm`: a fresh device whose cache starts from the cold
  phase's final data: the next launch.
- `cache_polled_cold_parallel4`, `cache_polled_warm_parallel4`: the same
  from 4 threads sharing the cache and the lock.

`bench_render_passes`: a G-buffer pass (RGBA8, 2x RGBA16F, D32, about 2000
objects), then 32-light shading.
- `clear_store_1080p`, `load_store_1080p`, `dontcare_store_1080p`: the
  G-buffer's load operation.
- `msaa4_resolve_1080p`: a 4x MSAA G-buffer resolved and not stored, its
  images transient (memoryless).
- `msaa4_store_1080p`: the same with the MSAA images stored.
- `local_read_1080p`: one rendering instance whose lighting reads the
  G-buffer as input attachments (dynamic rendering local read); the
  G-buffer is never stored. Skipped without `dynamicRenderingLocalRead`.
- `clear_store_2160p`, `local_read_2160p`: the same at 3840x2160.

`bench_render_area`: each next to the same passes into images of exactly the
area's size. The check also reads back what lies outside the area, which
must survive.
- `atlas_16tiles`: a shadow atlas updated tile by tile, 16 depth-only passes
  a frame into 1024x1024 tiles of an 8192x8192 D32 atlas.
- `separate_16tiles`: the same passes into 16 images of their own.
- `dynres70_2160p`: dynamic resolution, a scene pass into the top-left 70%
  of a 3840x2160 RGBA16F and D32 target, depth `DONT_CARE`.
- `dynres70_2160p_exact`: the same pass into 2688x1512 images.

`bench_transfers`:
- `clear3d_128`: a 128x128x128 3D image cleared every frame.
- `mipgen_2048`, `mipgen_2048_storage`: `vkCmdBlitImage` mip chains of 2048x2048
  RGBA8 textures, a barrier after each level; the second with `STORAGE`
  usage.
- `mipgen_cube512`: the mip chain of a 512x512 RGBA16F cube map.
- `mipgen_warp4_512`: 4 512x512 textures level by level, barriers batched
  as vkQuake records them.
- `blit_atlas64`: 64 scaled blits into an atlas, no barriers between them.
- `stream16_512`: 16 sRGB textures streamed in: a copy into level 0, then 9
  blitted mips each.
- `fill_x128`: 128 `vkCmdFillBuffer` back to back into one buffer.
- `upscale_2160p`: render scaling, a linear `vkCmdBlitImage` of a
  2560x1440 RGBA16F scene into a 3840x2160 BGRA8 image.
- `resolve_1080p`: `vkCmdResolveImage` of a 4x 1920x1080 RGBA16F target.

`bench_compute`:
- `chain_100`, `chain_1000`: dependent dispatches of 4096 elements with a
  barrier between each.
- `independent_1000`: dispatches on disjoint slices, no barriers.
- `scan_1m`, `scan_16m`: a multi-pass inclusive prefix sum of 1M or 16M
  `uint`s.

`bench_shaders_gpu`: one full-screen pass a frame into RGBA16F.
- `pom_1080p`, `pom_2160p`: parallax occlusion mapping, a ray march in a
  helper that returns from inside its loop.
- `deferred_1080p`, `deferred_2160p`: a deferred-lighting composite with
  early returns before a 32-light loop.
- `forward_pbr_1080p`, `forward_pbr_2160p`: forward PBR over a mesh: three
  material maps, 5x5 PCF over a cascade array, 32 lights from a storage
  buffer and image-based lighting.

## Running

```sh
make -j8 bench        # all benchmarks on PlasmaVK
make -j8 bench-mvk    # all benchmarks on MoltenVK (Homebrew vulkan-loader + molten-vk)
```

Both targets build every `bench/bench_*.c` (in parallel with `-j`; a
binary is rebuilt when its source, a `bench/` header or `plasmavk.h`
changes), then run them one at a time. They keep going after an error and
exit non-zero if any benchmark reported one. To run a single benchmark, use
the binary:

```sh
./build/bench/bench_draw_calls
VK_DRIVER_FILES=$(brew --prefix)/etc/vulkan/icd.d/MoltenVK_icd.json BENCH_IMPL=moltenvk \
    ./build/bench-loader/bench_draw_calls
```

The benchmarks read these environment variables:

| Variable | Effect |
|---|---|
| `BENCH_FRAMES=n` | Measured frames per configuration (rounds for the `pool_` and `array_` configurations, launches or iterations for `startup`). Each benchmark has its own defaults, sized for a run of a few seconds; most take at most 1000. `pipeline_compile` always creates its fixed set. |
| `BENCH_FILTER=s` | Run only the configurations whose name contains `s` (for example `push_`, `msaa`, `cold`). |
| `BENCH_IMPL=name` | The `impl` label of every line (default `plasmavk`, or `loader` in loader builds), to tell drivers or two PlasmaVK builds apart. |
| `BENCH_GPU_TIMING=0` | The frame loop writes no timestamps, so lines have no `gpu_us`. The two timestamps and their query reset can cost an implementation GPU time of their own, so use this when `frame_us` should be the application's frame alone. Lines say `"gpu_timing":false`. |
| `BENCH_ALL_FEATURES=1` | Create the device with every core feature it supports, as DXVK, vkd3d-proton and Zink do, instead of only what the benchmark uses. Some features change an implementation's paths: `sparseBinding` its allocations, `depthBounds` and `dynamicRenderingLocalRead` its depth-writing pipelines, `robustBufferAccess` its shaders. Lines say `"all_features":true`. `startup` creates its own devices and ignores this. |

While working on a benchmark, run it under the validation layers with sync
validation turned on (they are found through the same loader):
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true`,
and on PlasmaVK with its debug layer, `PLASMAVK_DEBUG=1`. Every benchmark
is clean under both.

## Comparing implementations

The loader build runs on whichever driver `VK_DRIVER_FILES` names. Use the
same binaries for each driver and give each run its own `impl` label:

```sh
make bench-mvk > moltenvk.jsonl        # also builds build/bench-loader/*
make bench > plasmavk.jsonl
for b in build/bench-loader/bench_*; do
    VK_DRIVER_FILES=/path/to/kosmickrisp_icd.json BENCH_IMPL=kosmickrisp $b
done > kosmickrisp.jsonl
```

Point `VK_DRIVER_FILES` at KosmicKrisp's ICD JSON from Mesa once it is
installed. To compare two versions of `plasmavk.h`, build the benchmarks
against each and label their runs with `BENCH_IMPL`. Every result line
carries `bench`, `config`, `impl` and `device`, so you can merge the files
and compare them, for example with jq:

```sh
cat *.jsonl | jq -r 'select(.cpu_us) | [.bench, .config, .impl, .cpu_us.median, .gpu_us.median] | @tsv'
```

Run comparisons on an idle machine, plugged in, and alternate the builds
over several runs. GPU clocks ramp, so `gpu_us` medians and minimums are
steadier than means. Some configurations depend on what ran before them in
the same process (thread scaling, for one): compare them run the same way,
or alone with `BENCH_FILTER`.

## Output

Each measured configuration prints one JSON line on stdout. Diagnostics go
to stderr.

```json
{"bench":"draw_calls","impl":"moltenvk","device":"Apple M4 Max","config":"push_constants_10000",
 "objects":10000,"draws":10000,"instances_per_draw":1,"frames":200,
 "cpu_us":{"min":949.67,"median":1012.63,"p99":1105.21,"max":1130.54,"mean":1018.02},
 "submit_us":{...},"gpu_us":{...},"frame_us":{...},"cpu_ns_per_draw":101.263}
```

(Shown wrapped here; the real output is one line.)

- Always present: `bench`, `impl`, `device` (when known), `config`. Also
  `gpu_timing` and `all_features` when the benchmark applied those options.
- Parameters such as `draws`, `width` or `bytes_per_frame` are plain fields
  that vary by benchmark.
- Timings are statistics objects in microseconds with `min`, `median`, `p99`,
  `max` and `mean`. The frame loop that most benchmarks share reports:
  - `cpu_us`: resetting the command pool, recording, ending and
    `vkQueueSubmit`, plus per-frame CPU work such as upload memcpys. It
    excludes the wait for a free frame slot.
  - `submit_us`: `vkQueueSubmit` alone. MoltenVK does much of its encoding
    here, so compare `cpu_us` across implementations.
  - `gpu_us`: from timestamp queries at the start and end of the command
    buffer. It is missing when the queue has no timestamps or
    `timestampPeriod` is 0. On PlasmaVK every timestamp drains the GPU,
    whatever stage it names (the work before it finishes, then the work
    after it starts; `plasmavk.h` section 14), so `gpu_us` is the drained
    time of the frame's work alone, while MoltenVK samples the stage asked
    for in a pipelined queue: the two are not comparable with each other,
    only each with itself across builds. Compare `frame_us` between
    implementations.
  - `frame_us`: the time between consecutive submits, with 2 frames in
    flight. This is throughput, whichever of CPU or GPU is the bottleneck.
- Derived numbers, all computed from medians: `cpu_ns_per_draw`
  (`draw_calls`, `descriptor_churn`, `frame`), `cpu_ns_per_dispatch` and
  `gelements_per_s` (`compute`), `cpu_gb_per_s` and `gpu_gb_per_s`
  (`upload_streaming`).
- `frame`: `cpu_us` is the main thread's critical path, including waiting
  for its workers. The multithreaded configurations add `worker_cpu_us`
  (the CPU time of all workers in a frame) and `record_wall_us` (from waking
  the workers to the last one finishing). `st_profiled` adds the GPU time of
  each group of passes: `gpu_shadow_us`, `gpu_gbuffer_us`, `gpu_cull_us`,
  `gpu_light_us`, `gpu_bloom_us` and `gpu_final_us`. `st_bgcompile` adds the
  pipelines that finished compiling while the frames were measured
  (`bg_compiles`, `bg_compiles_per_s`) and their `create_us`.
- `present` has its own loop. `acquire_us`, `cpu_us` (resetting, recording
  and `vkQueueSubmit`) and `present_us` are the CPU time of the calls;
  `gpu_us` is the frame's command buffer. `present_gpu_us` runs from the end
  of that command buffer to a timestamp submitted after the present: what
  presenting adds, such as a copy into the drawable. It counts only frames
  whose timestamp was submitted while the command buffer could still be
  running, so that the GPU did not go idle in between; `present_frames` says
  how many. At this pace the GPU is mostly idle and may not clock up, so
  absolute times run higher than in a GPU-bound game: compare a layer
  configuration with its headless control.
- `startup`: `instance_device` reports `wall_us` per iteration. The
  `process_first` configurations report each phase: `exec_us`,
  `coregraphics_us`, `instance_us`, `device_us`, `swapchain_us`,
  `first_frame_us` and `teardown_us`; `first_submit_us`, the first frame
  until `vkQueueSubmit` returns; `to_first_frame_us`, from `posix_spawn` to
  the first frame's fence; and `process_us`, until the process exited. The
  program has a bundle identifier, so Metal keeps its shader cache apart
  from other programs', as an installed application's, and the unmeasured
  launches fill it. Without one, the first render pass of a process opens
  the cache that every program without a bundle identifier shares: after
  conformance runs, hundreds of milliseconds, by tens more or less with the
  build's code signature.
- `pipeline_compile`: `create_us` per pipeline and `total_ms`, their sum;
  `cold_parallel4` adds `wall_ms`, and `batch_cold` reports only the call's
  `total_ms` and `ms_per_pipeline`. `permutations_` add per-bucket
  statistics and totals: `new_fragment_us` and `new_fragment_ms`,
  `new_vertex_`, `new_state_`, `new_pair_` and `repeat_`. Where a driver
  compiles only some of a bucket's pipelines, the bucket's median hides
  them, so compare its total. `cache_polled_` report `pipeline_us`, what
  the engine's call costs (the creation, waiting for the lock, the size
  query and any save), and each part: `create_us`, `query_us`, `saves` and
  their `save_ms`; then `total_ms` (the creations), `wall_ms`,
  `cache_bytes` (the final size), `merge_ms` (`vkMergePipelineCaches` of
  the final cache into an empty one) and, when warm, `load_ms`
  (`vkCreatePipelineCache` with the data).
- `descriptor_churn`'s `pool_` and `array_` configurations report
  `round_us` (the wall time of a round of `ops_per_round` operations, on
  every thread at once for `pool_threads_N`) and `ns_per_op`. An operation
  is a free and an allocate (`pool_free_churn`), a set of a batch
  allocation and its share of the reset (`pool_alloc_batch`), a pool create
  and destroy (`pool_create`), a one-element write (`array_write`), or a set
  allocated and written on each thread (`pool_threads_N`: constant across N
  when threads scale).
- `resource_churn`: `ns_per_op`, an object with the mean time of each kind
  of call made during the measured frames (create, bind, map, destroy,
  free, name, descriptor write...), and `resource_us_per_frame`, their total
  per frame. `write_descriptor` is per descriptor written: one
  `vkUpdateDescriptorSets` writes all of a frame's new views.

A configuration that cannot run prints `skipped` with the reason instead of
results, and the program exits 0:

```json
{"bench":"draw_calls","impl":"plasmavk","device":"Apple M4 Max","skipped":"Vulkan 1.3 features not supported: dynamicRendering synchronization2"}
{"bench":"compute","impl":"plasmavk","device":"Apple M4 Max","skipped":"vkCreateCommandPool: VK_ERROR_UNKNOWN (-13)"}
```

A failed Vulkan call makes the rest of that program skip. A missing
optional feature or extension skips only the configurations that need it.

A configuration whose output is wrong prints `error` and no numbers. So does
a GPU hang, detected by a 10 s fence timeout. The program then exits 1, after
its other configurations have run:

```json
{"bench":"compute","impl":"plasmavk","device":"Apple M4 Max","config":"scan_1m","error":"element 1024: got 523, want 91772"}
```

## How the benchmarks are built

- Shared code lives in three headers:
  - `pvk_bench.h`: timing, statistics, JSON, options.
  - `bench_vk.h`: device setup, memory, pipelines, barriers, readback, and
    the frame loop.
  - `bench_shaders.h`: the SPIR-V, with each array's GLSL source in the
    comment above it. To change a shader, edit the GLSL, compile it with
    `glslangValidator -V --target-env vulkan1.3`, and paste the words back.
- The benchmarks require Vulkan 1.3 with `dynamicRendering` and
  `synchronization2` (`compute`, `transfers` and `upload_streaming` need only
  the latter). `pipeline_compile` also needs `maintenance4`, because the
  SPIR-V uses `LocalSizeId`; `indirect` needs `multiDrawIndirect` and
  `drawIndirectFirstInstance`; `present` needs `VK_KHR_surface`,
  `VK_EXT_metal_surface`, `VK_KHR_swapchain` and timestamps. Everything else
  is optional, per configuration: push descriptors, dynamic rendering local
  read, 4x MSAA, depth bounds, `drawIndirectCount`, update-after-bind
  texture tables, `VK_EXT_headless_surface` and `VK_EXT_debug_utils`.
  `VK_KHR_portability_subset` is enabled whenever the device exposes it.
- Device-level commands are called through pointers from
  `vkGetDeviceProcAddr` (`vkd.CmdDraw(...)`), as engines do, so loader
  builds measure the driver, not the loader's trampolines. `bench_vk.h`
  loads every device command the benchmarks use. `present` loads the
  surface commands (instance commands) itself, and `startup` everything,
  since it times instance and device creation from scratch.
- Optional features and extensions go through `bench_vk_init_with`: each one
  the implementation supports is enabled, and `bench_vk.enabled` and
  `bench_vk_extension` say which were. The frame loop records one primary
  command buffer per frame, or (after `bench_loop_primaries`) several,
  recorded on separate threads and submitted together.
- Workloads are deterministic: fixed seeds, fixed scenes, and CPU references
  that are exact wherever possible. The exceptions use tolerances: float
  lighting in `render_passes` and `shaders_gpu` (fp16 rounding), the 5-way
  texture product in `descriptor_churn` and the material shader in
  `pipeline_compile` (±1 per channel), blitted mips in `transfers` (the
  rounding of the format) and its upscale (±1), the tonemapped frame in
  `present` (±1), and
  `frame`'s final image, lit, bloomed and tonemapped in float (±2; its
  G-buffer, light lists and UI glyphs are exact).
- `pipeline_compile` mixes a per-run salt into the shader code: into
  specialization constants of the fragment and compute shaders, and in the
  material phases into a constant of the vertex shaders' SPIR-V (the
  original phases' vertex shaders vary per pipeline but not per run). That
  way no earlier run's driver or OS cache of the specialized code can serve
  the "cold" pipelines. The CPU reference uses the same salt. A driver that
  keeps specialization constants as Metal function constants (MoltenVK,
  PlasmaVK) compiles the same MSL for those shaders in every run, so their
  front-end compile (MSL to AIR) can still be served from an earlier run;
  what stays cold for it is the specialization and the back end.
- `gpu_us` is only as good as the implementation's timestamps. MoltenVK
  sometimes reports implausible minimums, far below the median, most often
  in transfer-only frames. Compare medians, and cross-check with
  `frame_us`. PlasmaVK takes each timestamp once all earlier work of the
  queue is done and starts the work after it only then (`plasmavk.h`
  section 14): a full drain at each timestamp, so its `gpu_us` is the
  frame's own GPU time with no overlap with the frames around it, and the
  two timestamps cost the frame the overlap it would have had. MoltenVK's
  `gpu_us` is a pipelined frame's. A lower `gpu_us` on one is not a faster
  frame than the other's: compare `gpu_us` only between PlasmaVK builds,
  and `frame_us` between implementations.
