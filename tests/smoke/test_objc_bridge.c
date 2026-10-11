/* The C99 Objective-C bridge: typed messaging and hand-built blocks. */
#define PLASMAVK_IMPLEMENTATION
#include "../../plasmavk.h"

#include <time.h>

#include "../pvk_test.h"

static void test_device_messaging(void)
{
    id device = pvk_MTLCreateSystemDefaultDevice();

    REQUIRE(device != NULL);
    CHECK(pvk_ns_utf8(pvk_MTLDevice_name(device))[0] != '\0');
    CHECK(pvk_MTLDevice_supportsFamily(device, PVK_MTLGPUFamilyApple7));
    CHECK(pvk_MTLDevice_supportsFamily(device, PVK_MTLGPUFamilyMetal4));
    CHECK(pvk_MTLDevice_recommendedMaxWorkingSetSize(device) > 0);
    pvk_release(device);
}

struct completion {
    volatile int fired;
    id command_buffer;
};

static void on_completed(void *user, id command_buffer)
{
    struct completion *c = user;
    c->command_buffer = command_buffer;
    c->fired = 1;
}

/* A stack block with captured state survives Metal's Block_copy and is
 * invoked on Metal's completion thread with the right captures. */
static void test_stack_block(void)
{
    struct completion done = { 0, NULL };
    struct pvk_block block;
    struct timespec tick = { 0, 1000000 };
    id device, queue, cb;
    int waited_ms;
    void *pool = objc_autoreleasePoolPush(); /* commandBuffer is autoreleased */

    device = pvk_MTLCreateSystemDefaultDevice();
    REQUIRE(device != NULL);
    queue = PVK_MSG(id, (id, SEL))(device, sel_registerName("newCommandQueue"));
    REQUIRE(queue != NULL);
    cb = PVK_MSG(id, (id, SEL))(queue, sel_registerName("commandBuffer"));
    REQUIRE(cb != NULL);

    block = pvk_block_make(on_completed, &done);
    PVK_MSG(void, (id, SEL, void *))(cb, sel_registerName("addCompletedHandler:"), &block);
    PVK_MSG(void, (id, SEL))(cb, sel_registerName("commit"));
    PVK_MSG(void, (id, SEL))(cb, sel_registerName("waitUntilCompleted"));

    for (waited_ms = 0; !done.fired && waited_ms < 2000; waited_ms++)
        nanosleep(&tick, NULL);
    CHECK(done.fired);
    CHECK(done.command_buffer == cb);

    objc_autoreleasePoolPop(pool);
    pvk_release(queue);
    pvk_release(device);
}

int main(void)
{
    RUN(test_device_messaging);
    RUN(test_stack_block);
    return TEST_RESULT();
}
