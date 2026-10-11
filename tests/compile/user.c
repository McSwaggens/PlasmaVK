/* Any other translation unit: declarations only. */
#include "plasmavk.h"

int main(void)
{
    uint32_t version = 0;
    return vkEnumerateInstanceVersion(&version) == VK_SUCCESS && version ? 0 : 1;
}
