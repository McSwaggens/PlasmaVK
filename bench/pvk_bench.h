/* Benchmark harness: timing, statistics, options and JSON output.
 *
 * Every benchmark prints one JSON line per measured configuration on stdout
 * (diagnostics go to stderr):
 *
 *   {"bench":"draw_calls","impl":"moltenvk","device":"Apple M4 Max",
 *    "config":"push_constants","draws":10000,"frames":200,
 *    "cpu_us":{"min":..,"median":..,"p99":..,"max":..,"mean":..},"gpu_us":{..},..}
 *
 * A configuration that cannot run prints "skipped" instead of results, and
 * one whose output fails verification prints "error":
 *
 *   {"bench":"compute","impl":"plasmavk","skipped":"vkCreateCommandPool: VK_ERROR_UNKNOWN"}
 *   {"bench":"compute","impl":"plasmavk","config":"scan","error":"element 17: got 5, want 9"}
 *
 * Skips exit 0. Errors make the program exit 1 once its other configurations
 * have run.
 *
 * Build modes:
 *   default            PlasmaVK compiled into the benchmark (how users ship it)
 *   -DBENCH_USE_LOADER the system Vulkan loader, so the same workload runs on
 *                      MoltenVK or KosmicKrisp (pick one with VK_DRIVER_FILES)
 *
 * Environment:
 *   BENCH_IMPL           the "impl" label in loader builds (default "loader")
 *   BENCH_FRAMES         measured frames (or iterations) per configuration
 *   BENCH_FILTER         only run configurations whose name contains this string
 *   BENCH_GPU_TIMING=0   no timestamps in the frame loop, so no gpu_us: the
 *                        frame as the application recorded it, for frame_us
 *                        (lines say "gpu_timing":false)
 *   BENCH_ALL_FEATURES=1 create the device with every core feature it
 *                        supports, as DXVK, vkd3d-proton and Zink do (lines
 *                        say "all_features":true)
 */
#ifndef PVK_BENCH_H
#define PVK_BENCH_H

#ifndef BENCH_USE_LOADER
#define PLASMAVK_IMPLEMENTATION /* otherwise: declarations only, linked to the loader */
#endif
#include "../plasmavk.h"

#include <mach/mach_time.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Identity ---- */

static const char *bench_name = "bench";
static char bench_device[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
static int bench_status; /* exit code: 1 once any configuration failed verification */

/* The options a benchmark applied (bench_vk.h sets these where it applies
 * them; a benchmark that never reads an option is not marked with it). */
static struct {
    int no_gpu_timing; /* BENCH_GPU_TIMING=0 in the frame loop */
    int all_features;  /* BENCH_ALL_FEATURES=1 at device creation */
} bench_applied;

static inline void bench_begin(const char *name)
{
    bench_name = name;
}

static inline void bench_set_device(const char *device_name)
{
    snprintf(bench_device, sizeof(bench_device), "%s", device_name);
}

static inline const char *bench_impl(void)
{
#ifdef BENCH_USE_LOADER
    const char *name = getenv("BENCH_IMPL");
    return name ? name : "loader";
#else
    return "plasmavk";
#endif
}

static inline int bench_exit_code(void)
{
    return bench_status;
}

/* ---- Time ---- */

static inline double bench_now_us(void)
{
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom)
        mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * timebase.numer / timebase.denom / 1000.0;
}

/* ---- Options ---- */

/* BENCH_FRAMES, or the benchmark's own default. */
static inline int bench_frames(int fallback)
{
    const char *env = getenv("BENCH_FRAMES");
    int n = env ? atoi(env) : 0;
    return n > 0 ? n : fallback;
}

static inline int bench_selected(const char *config)
{
    const char *filter = getenv("BENCH_FILTER");
    return !filter || !*filter || strstr(config, filter) != NULL;
}

/* An environment variable as a flag: unset or empty gives fallback, "0"
 * gives 0, anything else 1. */
static inline int bench_env_flag(const char *name, int fallback)
{
    const char *env = getenv(name);
    if (!env || !*env)
        return fallback;
    return strcmp(env, "0") != 0;
}

static inline int bench_gpu_timing(void)
{
    return bench_env_flag("BENCH_GPU_TIMING", 1);
}

static inline int bench_all_features(void)
{
    return bench_env_flag("BENCH_ALL_FEATURES", 0);
}

/* ---- Deterministic random numbers (splitmix64) ---- */

static inline uint32_t bench_rand(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (uint32_t)((z ^ (z >> 31)) >> 32);
}

/* Uniform in [lo, hi). */
static inline int bench_rand_range(uint64_t *state, int lo, int hi)
{
    return lo + (int)(bench_rand(state) % (uint32_t)(hi - lo));
}

/* ---- Statistics ---- */

struct bench_stats {
    int n;
    double min, median, p99, max, mean;
};

static inline int bench_compare(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Sorts samples in place. */
static inline struct bench_stats bench_stats_compute(double *samples, int n)
{
    struct bench_stats s;
    double sum = 0.0;
    int i;

    memset(&s, 0, sizeof(s));
    if (n <= 0)
        return s;
    qsort(samples, (size_t)n, sizeof(double), bench_compare);
    for (i = 0; i < n; i++)
        sum += samples[i];
    s.n = n;
    s.min = samples[0];
    s.median = samples[n / 2];
    s.p99 = samples[(n * 99) / 100 < n ? (n * 99) / 100 : n - 1];
    s.max = samples[n - 1];
    s.mean = sum / n;
    return s;
}

/* Median without disturbing the samples' order. */
static inline double bench_median(const double *samples, int n)
{
    double *copy, median;
    if (n <= 0)
        return 0.0;
    copy = malloc(sizeof(double) * (size_t)n);
    if (!copy)
        return 0.0;
    memcpy(copy, samples, sizeof(double) * (size_t)n);
    median = bench_stats_compute(copy, n).median;
    free(copy);
    return median;
}

/* ---- JSON lines ---- */

struct bench_json {
    char buf[4096];
    size_t len;
};

static inline void bench_json_printf(struct bench_json *j, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static inline void bench_json_printf(struct bench_json *j, const char *fmt, ...)
{
    va_list args;
    int n;
    if (j->len >= sizeof(j->buf))
        return;
    va_start(args, fmt);
    n = vsnprintf(j->buf + j->len, sizeof(j->buf) - j->len, fmt, args);
    va_end(args);
    if (n > 0)
        j->len += (size_t)n;
    if (j->len >= sizeof(j->buf))
        j->len = sizeof(j->buf) - 1;
}

/* Appends "s" as a JSON string. */
static inline void bench_json_quote(struct bench_json *j, const char *s)
{
    bench_json_printf(j, "\"");
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\')
            bench_json_printf(j, "\\%c", c);
        else if (c < 0x20)
            bench_json_printf(j, "\\u%04x", c);
        else
            bench_json_printf(j, "%c", c);
    }
    bench_json_printf(j, "\"");
}

static inline void bench_json_str(struct bench_json *j, const char *key, const char *value)
{
    bench_json_printf(j, ",\"%s\":", key);
    bench_json_quote(j, value);
}

static inline void bench_json_int(struct bench_json *j, const char *key, long long value)
{
    bench_json_printf(j, ",\"%s\":%lld", key, value);
}

static inline void bench_json_num(struct bench_json *j, const char *key, double value)
{
    bench_json_printf(j, ",\"%s\":%.3f", key, value);
}

static inline void bench_json_bool(struct bench_json *j, const char *key, int value)
{
    bench_json_printf(j, ",\"%s\":%s", key, value ? "true" : "false");
}

/* ,"key":{"min":..,"median":..,"p99":..,"max":..,"mean":..} (sorts samples) */
static inline void bench_json_stats(struct bench_json *j, const char *key, double *samples, int n)
{
    struct bench_stats s = bench_stats_compute(samples, n);
    if (n <= 0)
        return;
    bench_json_printf(j, ",\"%s\":{\"min\":%.2f,\"median\":%.2f,\"p99\":%.2f,\"max\":%.2f,\"mean\":%.2f}",
                      key, s.min, s.median, s.p99, s.max, s.mean);
}

/* {"bench":..,"impl":..[,"device":..][,"config":..], plus the options that
 * change what is measured when the benchmark applied them. */
static inline void bench_json_begin(struct bench_json *j, const char *config)
{
    j->len = 0;
    bench_json_printf(j, "{\"bench\":");
    bench_json_quote(j, bench_name);
    bench_json_str(j, "impl", bench_impl());
    if (bench_device[0])
        bench_json_str(j, "device", bench_device);
    if (config)
        bench_json_str(j, "config", config);
    if (bench_applied.no_gpu_timing)
        bench_json_bool(j, "gpu_timing", 0);
    if (bench_applied.all_features)
        bench_json_bool(j, "all_features", 1);
}

static inline void bench_json_end(struct bench_json *j)
{
    bench_json_printf(j, "}");
    fputs(j->buf, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

/* ---- Outcomes ---- */

static inline void bench_outcome(const char *config, const char *key, const char *fmt, va_list args)
{
    struct bench_json j;
    char reason[1024];
    vsnprintf(reason, sizeof(reason), fmt, args);
    bench_json_begin(&j, config);
    bench_json_str(&j, key, reason);
    bench_json_end(&j);
}

/* The configuration (NULL: the whole benchmark) cannot run here. */
static inline void bench_skip(const char *config, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static inline void bench_skip(const char *config, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    bench_outcome(config, "skipped", fmt, args);
    va_end(args);
}

/* The configuration ran but produced wrong results; no numbers are reported. */
static inline void bench_error(const char *config, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static inline void bench_error(const char *config, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    bench_outcome(config, "error", fmt, args);
    va_end(args);
    bench_status = 1;
}

/* ---- Simple repeated measurement ---- */

/* Runs fn warmup + iterations times and reports per-iteration wall time.
 * fn returns 0 on success, or a reason the benchmark cannot run. */
static inline int bench_run(const char *config, int warmup, int iterations,
                            const char *(*fn)(void *user), void *user)
{
    double *samples = malloc(sizeof(double) * (size_t)iterations);
    struct bench_json j;
    const char *failed = NULL;
    int i;

    if (!samples)
        return 1;
    for (i = 0; i < warmup && !failed; i++)
        failed = fn(user);
    for (i = 0; i < iterations && !failed; i++) {
        double start = bench_now_us();
        failed = fn(user);
        samples[i] = bench_now_us() - start;
    }
    if (failed) {
        bench_skip(config, "%s", failed);
    } else {
        bench_json_begin(&j, config);
        bench_json_int(&j, "iterations", iterations);
        bench_json_stats(&j, "wall_us", samples, iterations);
        bench_json_end(&j);
    }
    free(samples);
    return 0;
}

#endif /* PVK_BENCH_H */
