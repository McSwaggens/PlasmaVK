/* Minimal test harness for PlasmaVK smoke tests. */
#ifndef PVK_TEST_H
#define PVK_TEST_H

#include <stdio.h>

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
