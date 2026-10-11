# Conformance (VK-GL-CTS)

The Khronos Vulkan CTS (`deqp-vk`, tag `vulkan-cts-1.4.7.0` or newer) is the
compatibility oracle. Its main-branch mustpass list is roughly 2-3M cases.

## Plan (not yet implemented)

1. **Fetch and build** VK-GL-CTS with CMake (`-DDEQP_TARGET=osx`), building
   only the `deqp-vk` target.
2. **Point it at PlasmaVK without a loader.** `deqp-vk` dlopens the Vulkan
   library at runtime. A *test-only* dylib built from `tests/compile/impl.c`
   exports the `vk*` symbols. Placing it first on the library search path makes
   `deqp-vk` use PlasmaVK directly. Users never see this dylib.
   TODO: confirm the library name and path `deqp-vk` loads on macOS, or whether
   it accepts a library-path option.
3. **Curated subsets** per subsystem (`dEQP-VK.api.*`, `dEQP-VK.memory.*`, ...),
   with a pass/fail baseline committed here. `make cts-<group>` fails on any
   regression from the baseline.
4. **Full nightly run**, with pass rate tracked over time.
