# PlasmaVK benchmarks

Headless programs that time Vulkan in realistic scenarios. Each one builds
two ways from the same source:

- PlasmaVK compiled in (default), the way users ship it.
- `-DBENCH_USE_LOADER`, which uses the system Vulkan loader, so the same
  workload runs on MoltenVK or KosmicKrisp.

Each benchmark checks its own output once per configuration. It reads the
result back and compares it with a CPU reference, so a broken implementation
reports an error instead of fast-but-wrong numbers.

| Program | What it measures |
|---|---|
| `bench_startup` | Instance + device creation and teardown: the cost before the first frame. |
| `bench_draw_calls` | CPU recording + submit cost and GPU time for thousands of small draws per frame (1k–50k), with push constants per draw. Configs: plain draws, indexed draws that bind a vertex buffer per draw, a pipeline switch every 8 draws, instanced batches of 64 or 1024, draws that each set their whole state again (`rebind`: viewport, scissor, the same pipeline, push constants, and their mesh's vertex and index buffers, one of 64), the same draws in secondary command buffers (4 large ones, or one per object as in Sascha's multithreading sample), and `vs_heavy`: 1000 draws of 6144 vertices, all but six of them clipped away, so `gpu_us` is vertex shading that reads per-draw data. Reports `cpu_ns_per_draw`. |
| `bench_descriptor_churn` | 10k draws/frame, each with its own descriptors (1 uniform buffer + 4 textures), delivered eight ways: pre-built sets (bind only), `vkAllocateDescriptorSets` + `vkUpdateDescriptorSets`, update templates, dynamic offsets, push descriptors, push templates, sets by update frequency (`multi_frequency`: set 0 per frame, set 1 per material every 8 draws, set 2 per object with a dynamic offset), and allocate + update recorded by 4 threads into secondaries from per-thread pools that share one set layout (`update_sets_mt4`). The `pool_` and `array_` configs time descriptor management alone (`ns_per_op`): a full `FREE_DESCRIPTOR_SET` pool of 4096 sets where a random set is freed and reallocated, batch allocation of 256 sets, pool creation, one-element writes into a bindless table of 65536 or 1024 textures (update after bind, partially bound), and 1, 2, 4 and 8 threads with a pool each; each then draws with the sets it leaves to check them. `_named` variants give one object a `VK_EXT_debug_utils` name first. |
| `bench_resource_churn` | Resources created and destroyed every frame, as streaming engines and render graphs do: textures with dedicated memory or suballocated from blocks (also with 4000 other textures alive), staging and device-local buffers, a render graph's transient targets on reused memory, single-mip/layer views, samplers, load-time one-shot uploads, and texel buffer views. Reports `ns_per_op` per call (create, bind, map, destroy, free, descriptor write...) and `resource_us_per_frame` next to the frame timings. `_named` variants name every new resource with `VK_EXT_debug_utils`. |
| `bench_compute` | Chains of 100 or 1000 small dispatches with a barrier between each, 1000 independent dispatches without barriers, and a multi-pass inclusive prefix sum of 1M and 16M `uint`s (reports `gelements_per_s`). |
| `bench_render_passes` | A deferred frame with dynamic rendering: a G-buffer (RGBA8 + 2x RGBA16F + D32, ~2000 objects) and then 32-light shading. The variants change what leaves tile memory: G-buffer `CLEAR`/`LOAD`/`DONT_CARE`; 4x MSAA with resolve, with the MSAA images transient (memoryless) or stored; and a single-pass version that reads the G-buffer as input attachments (dynamic rendering local read) and never stores it. 1080p and 4K. GPU time is the number to watch. |
| `bench_render_area` | Render areas smaller than the attachments: a shadow atlas updated tile by tile (16 depth-only passes per frame, each into a 1024² tile of an 8192² D32 atlas) and dynamic resolution (a scene pass into the top-left 70% of a 4K RGBA16F + D32 target, depth `DONT_CARE`). Each next to the same passes into images of exactly the area's size. GPU time is the number to watch. The check also reads back what lies outside the area, which must survive. |
| `bench_pipeline_compile` | The time to create each of 256 graphics and 128 compute pipelines (distinct shaders, specialization constants and fixed-function state), one call per pipeline. The spread is the stutter: watch p99 and max. Phases: `cold`, `warm_cache` (a fresh device given the cold run's `vkGetPipelineCacheData`), `no_cache`, and `cold_parallel4` (4 threads). |
| `bench_frame` | One engine-like deferred frame, the mix a game submits instead of one pattern at a time: streamed object constants, UI geometry and texture tiles; a per-frame descriptor pool; 4 shadow cascades (a depth-only pipeline, 1000 draws each); a G-buffer (3000 draws, 4 pipelines, 64 materials, a dynamic offset per draw); tiled light culling in compute; tiled lighting with shadow lookups; 11 bloom passes; tonemap and 600 scissored UI draws. 7613 draws in 18 rendering instances. Configs: one thread; 4 threads recording secondaries (`mt4_secondary`); 4 threads recording primaries submitted together (`mt4_primary`); timestamps around every pass, read a frame later like an engine's GPU profiler (`st_profiled`); a thread compiling pipelines meanwhile (`st_bgcompile`); 3840x2160; and a device with depth bounds enabled and made dynamic (and off), as translation layers do (`st_dynamic_depth_bounds`). |
| `bench_upload_streaming` | Per-frame uploads: 256 x 4 KB buffer updates through a staging ring, one 32 MB bulk copy, 64 texture tiles into an atlas, a full 2048² mip chain, the same updates written directly into host-visible device-local memory, and map / flush / unmap per frame. CPU time includes writing the data. |

## Running

```sh
make bench        # all benchmarks on PlasmaVK
make bench-mvk    # all benchmarks on MoltenVK (Homebrew vulkan-loader + molten-vk)
```

Both targets run every benchmark, even after an error, and exit non-zero if
any benchmark reported one. To run a single benchmark, use the binary:

```sh
./build/bench/bench_draw_calls
VK_DRIVER_FILES=$(brew --prefix)/etc/vulkan/icd.d/MoltenVK_icd.json BENCH_IMPL=moltenvk \
    ./build/bench-loader/bench_draw_calls
```

The benchmarks read these environment variables:

| Variable | Effect |
|---|---|
| `BENCH_FRAMES=n` | Measured frames (or `startup` iterations) per configuration, at most 1000. Each benchmark has its own defaults, sized for a run of a few seconds. `pipeline_compile` always creates its fixed set. |
| `BENCH_FILTER=s` | Run only the configurations whose name contains `s` (for example `push_`, `msaa`, `cold`). |
| `BENCH_IMPL=name` | Sets the `impl` label in loader builds. |
| `BENCH_GPU_TIMING=0` | The frame loop writes no timestamps, so lines have no `gpu_us`. The two timestamps and their query reset can cost an implementation GPU time of their own, so use this when `frame_us` should be the application's frame alone. Lines say `"gpu_timing":false`. |
| `BENCH_ALL_FEATURES=1` | Create the device with every core feature it supports, as DXVK, vkd3d-proton and Zink do, instead of only what the benchmark uses. Some features change an implementation's paths: `sparseBinding` its allocations, `depthBounds` and `dynamicRenderingLocalRead` its depth-writing pipelines, `robustBufferAccess` its shaders. Lines say `"all_features":true`. `startup` creates its own device and ignores this. |

While working on a benchmark, run it under the validation layers with sync
validation turned on (they are found through the same loader):
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true`.
Every benchmark is clean under both on MoltenVK.

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
installed. Every result line carries `bench`, `config`, `impl` and `device`,
so you can merge the files and compare them, for example with jq:

```sh
cat *.jsonl | jq -r 'select(.cpu_us) | [.bench, .config, .impl, .cpu_us.median, .gpu_us.median] | @tsv'
```

Run comparisons on an idle machine, plugged in. GPU clocks ramp, so
`gpu_us` medians and minimums are steadier than means.

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
  `max` and `mean`. In the frame-based benchmarks:
  - `cpu_us`: resetting the command pool, recording, ending and
    `vkQueueSubmit`, plus per-frame CPU work such as upload memcpys. It
    excludes the wait for a free frame slot.
  - `submit_us`: `vkQueueSubmit` alone. MoltenVK does much of its encoding
    here, so compare `cpu_us` across implementations.
  - `gpu_us`: from timestamp queries at the start and end of the command
    buffer. It is missing when the queue has no timestamps or
    `timestampPeriod` is 0.
  - `frame_us`: the time between consecutive submits, with 2 frames in
    flight. This is throughput, whichever of CPU or GPU is the bottleneck.
- `startup` reports `wall_us` per iteration. `pipeline_compile` reports
  `create_us` per pipeline plus `total_ms` (and `wall_ms` for the threaded
  phase).
- `frame`'s `cpu_us` is the main thread's critical path, including waiting
  for its workers. The multithreaded configurations add `worker_cpu_us`
  (the CPU time of all workers in a frame) and `record_wall_us`;
  `st_profiled` adds the GPU time of each group of passes (`gpu_shadow_us`,
  `gpu_gbuffer_us`, `gpu_cull_us`, `gpu_light_us`, `gpu_bloom_us`,
  `gpu_final_us`); `st_bgcompile` adds the pipelines compiled while the
  frames were measured (`bg_compiles`, `bg_compiles_per_s`) and their
  `create_us`.
- The `pool_` and `array_` configurations of `descriptor_churn` report
  `round_us` (the wall time of a round of `ops_per_round` operations, on
  every thread at once for `pool_threads_N`) and `ns_per_op`. An operation
  is a free + allocate (`pool_free_churn`), a set of a batch allocation and
  its share of the reset (`pool_alloc_batch`), a pool create + destroy
  (`pool_create`), a one-element write (`array_write`), and a set allocated
  and written on each thread (`pool_threads_N`: constant across N when
  threads scale).
- `resource_churn` adds `ns_per_op`, an object with the mean time of each
  kind of call made during the measured frames, and `resource_us_per_frame`,
  their total per frame. `write_descriptor` is per descriptor written: one
  `vkUpdateDescriptorSets` writes all of a frame's new views.
- Derived numbers, all computed from medians: `cpu_ns_per_draw`,
  `cpu_ns_per_dispatch`, `gelements_per_s`, `cpu_gb_per_s`, `gpu_gb_per_s`.

A configuration that cannot run prints `skipped` with the reason instead of
results, and the program exits 0:

```json
{"bench":"draw_calls","impl":"plasmavk","device":"Apple M4 Max","skipped":"Vulkan 1.3 features not supported: dynamicRendering synchronization2"}
{"bench":"compute","impl":"plasmavk","device":"Apple M4 Max","skipped":"vkCreateCommandPool: VK_ERROR_UNKNOWN (-13)"}
```

A failed Vulkan call makes the rest of that program skip. An unsupported
optional feature (MSAA, push descriptors, local read) skips only its
configuration. So does a `_named` configuration when the instance does not
enable `VK_EXT_debug_utils` (`vkGetDeviceProcAddr` has no
`vkSetDebugUtilsObjectNameEXT`).

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
  `synchronization2` (`compute` and `upload_streaming` need only the
  latter). `pipeline_compile` also needs `maintenance4`, because the SPIR-V
  uses `LocalSizeId`. Push descriptors, dynamic rendering local read, 4x
  MSAA and depth bounds are optional, per configuration.
  `VK_KHR_portability_subset` is enabled whenever the device exposes it.
- Device-level commands are called through pointers from
  `vkGetDeviceProcAddr` (`vkd.CmdDraw(...)`), as engines do; a benchmark
  that needs commands `bench_vk.h` does not load loads them the same way.
  Loader builds therefore measure the driver, not the loader's trampolines.
- Optional features and extensions go through `bench_vk_init_with`: each one
  the device supports is enabled, and `bench_vk.enabled` says which were.
  The frame loop records one primary command buffer per frame, or (after
  `bench_loop_primaries`) several, recorded on separate threads and
  submitted together.
- Workloads are deterministic: fixed seeds, fixed scenes, and CPU references
  that are exact wherever possible. The exceptions use tolerances: float
  lighting in `render_passes`, the 5-way texture product in
  `descriptor_churn` (±1 per channel), and `frame`'s final image, lit,
  bloomed and tonemapped in float (±2; its G-buffer, light lists and UI
  glyphs are exact).
- `pipeline_compile` mixes a per-run salt into a specialization constant,
  which changes the shader code. That way no earlier run's driver or OS
  shader cache can serve the "cold" pipelines. The CPU reference uses the
  same salt.
- `gpu_us` is only as good as the implementation's timestamps. MoltenVK
  sometimes reports implausible minimums, such as a few µs for a 22 MB
  `image_mips_2048` copy or for `chain_100`, whose median is about 2 ms.
  This happens most with transfer-only frames. Compare medians, and
  cross-check with `frame_us`. PlasmaVK takes every timestamp once all
  earlier work of the queue is done (`plasmavk.h` section 14), so its
  `gpu_us` is the frame's own GPU time, not counting earlier frames.
