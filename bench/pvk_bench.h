/* Minimal benchmark harness. Each benchmark prints one JSON line:
 *   {"bench":"startup","impl":"plasmavk","iterations":200,"min_us":..,"median_us":..,"p99_us":..,"mean_us":..}
 *
 * Build modes:
 *   default            PlasmaVK compiled into the benchmark (how users ship it)
 *   -DBENCH_USE_LOADER the system Vulkan loader, so the same workload runs on
 *                      MoltenVK or KosmicKrisp (pick one with VK_DRIVER_FILES)
 */
#ifndef PVK_BENCH_H
#define PVK_BENCH_H

#ifndef BENCH_USE_LOADER
#define PLASMAVK_IMPLEMENTATION /* otherwise: declarations only, linked to the loader */
#endif
#include "../plasmavk.h"

#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *bench_impl(void)
{
#ifdef BENCH_USE_LOADER
    const char *name = getenv("BENCH_IMPL");
    return name ? name : "loader";
#else
    return "plasmavk";
#endif
}

static double bench_now_us(void)
{
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom)
        mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * timebase.numer / timebase.denom / 1000.0;
}

static int bench_compare(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Runs fn warmup + iterations times and reports per-iteration wall time. */
static int bench_run(const char *name, int warmup, int iterations, int (*fn)(void *user), void *user)
{
    double *samples = malloc(sizeof(double) * (size_t)iterations);
    double sum = 0.0;
    int i;

    if (!samples)
        return 1;
    for (i = 0; i < warmup; i++)
        if (fn(user) != 0)
            goto fail;
    for (i = 0; i < iterations; i++) {
        double start = bench_now_us();
        if (fn(user) != 0)
            goto fail;
        samples[i] = bench_now_us() - start;
        sum += samples[i];
    }
    qsort(samples, (size_t)iterations, sizeof(double), bench_compare);
    printf("{\"bench\":\"%s\",\"impl\":\"%s\",\"iterations\":%d,"
           "\"min_us\":%.2f,\"median_us\":%.2f,\"p99_us\":%.2f,\"mean_us\":%.2f}\n",
           name, bench_impl(), iterations, samples[0], samples[iterations / 2],
           samples[(iterations * 99) / 100], sum / iterations);
    free(samples);
    return 0;
fail:
    fprintf(stderr, "bench %s: iteration failed\n", name);
    free(samples);
    return 1;
}

#endif /* PVK_BENCH_H */
