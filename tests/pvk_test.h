/* Minimal test harness for PlasmaVK smoke tests. */
#ifndef PVK_TEST_H
#define PVK_TEST_H

#include <stdio.h>

/* An Info.plist section: with a bundle identifier, Metal caches the tests'
 * pipelines in files of their own ($DARWIN_USER_CACHE_DIR/plasmavk.smoke).
 * Programs without one share $DARWIN_USER_CACHE_DIR/com.apple.metal, each
 * process taking the first file no other process holds; the conformance
 * runs fill those to gigabytes, and opening a file of them grows its index
 * with realloc, which AddressSanitizer makes take minutes (it copies). */
static const char pvk_test_info_plist[] __attribute__((used, section("__TEXT,__info_plist"))) =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<plist version=\"1.0\"><dict><key>CFBundleIdentifier</key><string>plasmavk.smoke</string></dict></plist>\n";

static int pvk_test_failures;

#define PVK_TEST_FAIL(...)                                    \
    do {                                                      \
        fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);       \
        fprintf(stderr, __VA_ARGS__);                         \
        fputc('\n', stderr);                                  \
        pvk_test_failures++;                                  \
    } while (0)

/* CHECK records a failure and continues; REQUIRE also returns from the test. */
#define CHECK(cond)                                           \
    do {                                                      \
        if (!(cond))                                          \
            PVK_TEST_FAIL("CHECK failed: %s", #cond);         \
    } while (0)

#define REQUIRE(cond)                                         \
    do {                                                      \
        if (!(cond)) {                                        \
            PVK_TEST_FAIL("REQUIRE failed: %s", #cond);       \
            return;                                           \
        }                                                     \
    } while (0)

#define CHECK_VK(expr, expected)                                                        \
    do {                                                                                \
        VkResult pvk_r_ = (expr);                                                       \
        if (pvk_r_ != (expected))                                                       \
            PVK_TEST_FAIL("%s returned %d, expected %s", #expr, (int)pvk_r_, #expected); \
    } while (0)

#define REQUIRE_VK(expr)                                                                \
    do {                                                                                \
        VkResult pvk_r_ = (expr);                                                       \
        if (pvk_r_ != VK_SUCCESS) {                                                     \
            PVK_TEST_FAIL("%s returned %d", #expr, (int)pvk_r_);                        \
            return;                                                                     \
        }                                                                               \
    } while (0)

#define RUN(test)                                                                       \
    do {                                                                                \
        int pvk_before_ = pvk_test_failures;                                            \
        test();                                                                         \
        printf("%s %s\n", pvk_test_failures == pvk_before_ ? "PASS" : "FAIL", #test);   \
    } while (0)

#define TEST_RESULT() (pvk_test_failures ? 1 : 0)

#endif /* PVK_TEST_H */
