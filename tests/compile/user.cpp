// The declarations part of the header must be usable from C++.
#include "../../plasmavk.h"

uint32_t plasmavk_cpp_version()
{
    uint32_t version = 0;
    vkEnumerateInstanceVersion(&version);
    return version;
}
