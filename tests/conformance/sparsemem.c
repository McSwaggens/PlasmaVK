/* Chapter "Sparse Resources": the sparse memory requirements of images
 * (none for images without sparse residency, which is all PlasmaVK has)
 * and binding memory to a sparse buffer with vkQueueBindSparse, where a
 * queue family supports sparse binding. */
#include "ct.h"

#include <stdlib.h>

#define SP_GUARD 0xA5

static int sp_untouched(const void *p, size_t n)
{
    return ct_all_bytes(p, n, SP_GUARD);
}

/* VkSparseImageMemoryRequirements2 output structures: sType and pNext set,
 * the rest the guard pattern. */
static void sp_prepare2(VkSparseImageMemoryRequirements2 *r, uint32_t n)
{
    uint32_t i;
    for (i = 0; i < n; i++) {
        memset(&r[i], SP_GUARD, sizeof(r[i]));
        r[i].sType = VK_STRUCTURE_TYPE_SPARSE_IMAGE_MEMORY_REQUIREMENTS_2;
        r[i].pNext = NULL;
    }
}

CT_TEST(sparse_requirements_non_sparse,
        "Sparse Resources > Sparse Image Memory Requirements: for an image not created with "
        "VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT, vkGetImageSparseMemoryRequirements sets the count to zero and "
        "writes nothing; and (derived, the same queries) vkGetImageSparseMemoryRequirements2 (Vulkan 1.1) and "
        "vkGetDeviceImageSparseMemoryRequirements for such create infos (Vulkan 1.3)",
        "vkGetImageSparseMemoryRequirements vkGetImageSparseMemoryRequirements2 "
        "vkGetDeviceImageSparseMemoryRequirements vkCreateImage vkDestroyImage vkGetPhysicalDeviceImageFormatProperties")
{
    struct ct_env e;
    VkImageCreateInfo infos[4];
    VkSparseImageMemoryRequirements r1[4];
    VkSparseImageMemoryRequirements2 r2[4];
    VkImageFormatProperties fp;
    uint32_t i, count;

    ct_env_init(t, &e);
    ct_env_device(t, &e, NULL);
    infos[0] = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, 4, 2,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    infos[1] = ct_image_info(VK_IMAGE_TYPE_3D, VK_FORMAT_R32_SFLOAT, 16, 16, 16, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT);
    infos[2] = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_D16_UNORM, 32, 32, 1, 1, 1,
                             VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    infos[3] = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 1, 1, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    infos[3].tiling = VK_IMAGE_TILING_LINEAR;
    for (i = 0; i < 4; i++) {
        VkImage image;
        if (vkGetPhysicalDeviceImageFormatProperties(e.pd, infos[i].format, infos[i].imageType, infos[i].tiling,
                                                     infos[i].usage, 0, &fp) != VK_SUCCESS)
            continue; /* linear R8G8B8A8_UNORM transfer sources are not required */
        CT_REQUIRE_VK(t, vkCreateImage(e.device, &infos[i], NULL, &image));
        count = 77;
        vkGetImageSparseMemoryRequirements(e.device, image, &count, NULL);
        CT_CHECKF(t, count == 0, "image %u: count %u", i, count);
        count = 4;
        memset(r1, SP_GUARD, sizeof(r1));
        vkGetImageSparseMemoryRequirements(e.device, image, &count, r1);
        CT_CHECKF(t, count == 0 && sp_untouched(r1, sizeof(r1)), "image %u: count %u with an array of 4", i, count);
        if (e.version >= VK_API_VERSION_1_1) {
            VkImageSparseMemoryRequirementsInfo2 info;
            memset(&info, 0, sizeof(info));
            info.sType = VK_STRUCTURE_TYPE_IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2;
            info.image = image;
            count = 77;
            vkGetImageSparseMemoryRequirements2(e.device, &info, &count, NULL);
            CT_CHECKF(t, count == 0, "image %u: vkGetImageSparseMemoryRequirements2 count %u", i, count);
            count = 4;
            sp_prepare2(r2, 4);
            vkGetImageSparseMemoryRequirements2(e.device, &info, &count, r2);
            CT_CHECKF(t, count == 0 && sp_untouched(&r2[0].memoryRequirements, sizeof(r2[0].memoryRequirements)) &&
                             sp_untouched(&r2[3].memoryRequirements, sizeof(r2[3].memoryRequirements)),
                      "image %u: vkGetImageSparseMemoryRequirements2 count %u with an array of 4", i, count);
        }
        if (e.version >= VK_API_VERSION_1_3) {
            VkDeviceImageMemoryRequirements dinfo;
            memset(&dinfo, 0, sizeof(dinfo));
            dinfo.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS;
            dinfo.pCreateInfo = &infos[i];
            count = 77;
            vkGetDeviceImageSparseMemoryRequirements(e.device, &dinfo, &count, NULL);
            CT_CHECKF(t, count == 0, "create info %u: vkGetDeviceImageSparseMemoryRequirements count %u", i, count);
            count = 4;
            sp_prepare2(r2, 4);
            vkGetDeviceImageSparseMemoryRequirements(e.device, &dinfo, &count, r2);
            CT_CHECKF(t, count == 0 && sp_untouched(&r2[0].memoryRequirements, sizeof(r2[0].memoryRequirements)),
                      "create info %u: vkGetDeviceImageSparseMemoryRequirements count %u with an array of 4", i,
                      count);
        }
        vkDestroyImage(e.device, image, NULL);
    }
    ct_env_fini(&e);
}

static int sp_same(const VkSparseImageMemoryRequirements *a, const VkSparseImageMemoryRequirements *b)
{
    return a->formatProperties.aspectMask == b->formatProperties.aspectMask &&
           a->formatProperties.imageGranularity.width == b->formatProperties.imageGranularity.width &&
           a->formatProperties.imageGranularity.height == b->formatProperties.imageGranularity.height &&
           a->formatProperties.imageGranularity.depth == b->formatProperties.imageGranularity.depth &&
           a->formatProperties.flags == b->formatProperties.flags && a->imageMipTailFirstLod == b->imageMipTailFirstLod &&
           a->imageMipTailSize == b->imageMipTailSize && a->imageMipTailOffset == b->imageMipTailOffset &&
           a->imageMipTailStride == b->imageMipTailStride;
}

CT_TEST(sparse_requirements_residency,
        "Sparse Resources > Sparse Image Memory Requirements: for a 2D image with sparse residency "
        "(sparseResidencyImage2D), the requirements cover its color aspect; vkGetImageSparseMemoryRequirements2 "
        "and vkGetDeviceImageSparseMemoryRequirements (for its create info) report the same requirements "
        "(derived: the same query); the count protocol writes at most pSparseMemoryRequirementCount",
        "vkGetImageSparseMemoryRequirements vkGetImageSparseMemoryRequirements2 "
        "vkGetDeviceImageSparseMemoryRequirements vkCreateImage vkDestroyImage vkGetPhysicalDeviceFeatures2 "
        "vkGetPhysicalDeviceSparseImageFormatProperties")
{
    struct ct_env e;
    struct ct_features f;
    VkPhysicalDeviceFeatures2 enable;
    VkImageCreateInfo info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 512, 512, 1, 1, 1,
                                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    VkSparseImageMemoryRequirements *r1;
    VkSparseImageMemoryRequirements2 *r2;
    VkImage image;
    uint32_t count = 0, n, i, formats = 0;
    int color = 0;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_1)
        ct_skip(t, "needs Vulkan 1.1 (VkPhysicalDeviceFeatures2)");
    ct_get_features(&e, &f, 0);
    if (!f.f2.features.sparseBinding || !f.f2.features.sparseResidencyImage2D)
        ct_skip(t, "sparseResidencyImage2D is not supported");
    vkGetPhysicalDeviceSparseImageFormatProperties(e.pd, info.format, info.imageType, VK_SAMPLE_COUNT_1_BIT, info.usage,
                                                   info.tiling, &formats, NULL);
    if (!formats)
        ct_skip(t, "no sparse residency for 2D R8G8B8A8_UNORM images");
    memset(&enable, 0, sizeof(enable));
    enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    enable.features.sparseBinding = VK_TRUE;
    enable.features.sparseResidencyImage2D = VK_TRUE;
    ct_env_device(t, &e, &enable);
    info.flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
    CT_REQUIRE_VK(t, vkCreateImage(e.device, &info, NULL, &image));
    vkGetImageSparseMemoryRequirements(e.device, image, &count, NULL);
    CT_REQUIRE(t, count >= 1);
    r1 = calloc(count + 1, sizeof(*r1));
    r2 = calloc(count + 1, sizeof(*r2));
    CT_REQUIRE(t, r1 && r2);
    memset(r1, SP_GUARD, (count + 1) * sizeof(*r1));
    n = count + 1;
    vkGetImageSparseMemoryRequirements(e.device, image, &n, r1);
    CT_CHECKF(t, n == count, "count %u, then %u written", count, n);
    CT_CHECK(t, sp_untouched(&r1[count], sizeof(r1[count])));
    for (i = 0; i < count; i++)
        color |= (r1[i].formatProperties.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
    CT_CHECKF(t, color, "no requirement for the color aspect");
    if (count > 1) {
        VkSparseImageMemoryRequirements one[2];
        memset(one, SP_GUARD, sizeof(one));
        n = 1;
        vkGetImageSparseMemoryRequirements(e.device, image, &n, one);
        CT_CHECK(t, n == 1 && sp_same(&one[0], &r1[0]) && sp_untouched(&one[1], sizeof(one[1])));
    }
    {
        VkImageSparseMemoryRequirementsInfo2 i2;
        memset(&i2, 0, sizeof(i2));
        i2.sType = VK_STRUCTURE_TYPE_IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2;
        i2.image = image;
        sp_prepare2(r2, count + 1);
        n = count + 1;
        vkGetImageSparseMemoryRequirements2(e.device, &i2, &n, r2);
        CT_CHECKF(t, n == count, "vkGetImageSparseMemoryRequirements2: count %u, expected %u", n, count);
        for (i = 0; i < count && i < n; i++)
            CT_CHECKF(t, sp_same(&r2[i].memoryRequirements, &r1[i]), "vkGetImageSparseMemoryRequirements2: requirement %u differs", i);
    }
    if (e.version >= VK_API_VERSION_1_3) {
        VkDeviceImageMemoryRequirements dinfo;
        memset(&dinfo, 0, sizeof(dinfo));
        dinfo.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS;
        dinfo.pCreateInfo = &info;
        sp_prepare2(r2, count + 1);
        n = count + 1;
        vkGetDeviceImageSparseMemoryRequirements(e.device, &dinfo, &n, r2);
        CT_CHECKF(t, n == count, "vkGetDeviceImageSparseMemoryRequirements: count %u, expected %u", n, count);
        for (i = 0; i < count && i < n; i++)
            CT_CHECKF(t, sp_same(&r2[i].memoryRequirements, &r1[i]),
                      "vkGetDeviceImageSparseMemoryRequirements: requirement %u differs", i);
    }
    free(r1);
    free(r2);
    vkDestroyImage(e.device, image, NULL);
    ct_env_fini(&e);
}

CT_TEST(queue_bind_sparse,
        "Sparse Resources > Binding Resource Memory (vkQueueBindSparse): on a queue "
        "supporting sparse binding (VUID-vkQueueBindSparse-queuetype), memory bound to a sparse buffer "
        "(sparseBinding) backs it for commands that wait on the bind's semaphore signal; a fence of a "
        "vkQueueBindSparse of no batches signals after earlier signal operations (Synchronization > Fences)",
        "vkQueueBindSparse vkCreateDevice vkDestroyDevice vkGetDeviceQueue vkGetPhysicalDeviceQueueFamilyProperties "
        "vkGetPhysicalDeviceFeatures2 vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory "
        "vkFreeMemory vkBindBufferMemory vkMapMemory vkCreateSemaphore vkDestroySemaphore vkCreateFence vkDestroyFence "
        "vkWaitForFences vkQueueSubmit vkQueueWaitIdle vkCmdCopyBuffer vkCmdPipelineBarrier vkCreateCommandPool "
        "vkDestroyCommandPool vkAllocateCommandBuffers vkFreeCommandBuffers vkBeginCommandBuffer vkEndCommandBuffer")
{
    struct ct_env e;
    struct ct_features f;
    VkQueueFamilyProperties families[64];
    uint32_t family_count = 64, family = UINT32_MAX, i, type;
    VkPhysicalDeviceFeatures enable;
    const char *names[2];
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue;
    VkDeviceCreateInfo dinfo;
    VkBufferCreateInfo binfo;
    VkMemoryRequirements req;
    VkMemoryAllocateInfo alloc;
    VkSparseMemoryBind bind;
    VkSparseBufferMemoryBindInfo buffer_bind;
    VkBindSparseInfo sparse;
    VkSemaphoreCreateInfo sinfo;
    VkFenceCreateInfo finfo;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit;
    VkBufferCopy region;
    struct ct_buffer src, dst;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkSemaphore bound;
    VkFence fence;
    VkCommandBuffer cb;
    VkDeviceSize size = 1 << 18;

    ct_env_init(t, &e);
    vkGetPhysicalDeviceQueueFamilyProperties(e.pd, &family_count, families);
    for (i = 0; i < family_count && family == UINT32_MAX; i++)
        if ((families[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) &&
            (families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)))
            family = i;
    if (family == UINT32_MAX)
        ct_skip(t, "no queue family supports sparse binding (and transfers)");
    ct_get_features(&e, &f, 0);
    CT_REQUIREF(t, f.f2.features.sparseBinding, "a queue family supports sparse binding but sparseBinding is not reported");
    memset(&enable, 0, sizeof(enable));
    enable.sparseBinding = VK_TRUE;
    memset(&queue, 0, sizeof(queue));
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueFamilyIndex = family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    memset(&dinfo, 0, sizeof(dinfo));
    dinfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dinfo.queueCreateInfoCount = 1;
    dinfo.pQueueCreateInfos = &queue;
    dinfo.enabledExtensionCount = ct_add_portability_subset(&e, names, 0);
    dinfo.ppEnabledExtensionNames = dinfo.enabledExtensionCount ? names : NULL;
    dinfo.pEnabledFeatures = &enable;
    CT_REQUIRE_VK(t, vkCreateDevice(e.pd, &dinfo, NULL, &e.device));
    e.queue_family = family;
    vkGetDeviceQueue(e.device, family, 0, &e.queue);

    memset(&binfo, 0, sizeof(binfo));
    binfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    binfo.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT;
    binfo.size = size;
    binfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    CT_REQUIRE_VK(t, vkCreateBuffer(e.device, &binfo, NULL, &buffer));
    vkGetBufferMemoryRequirements(e.device, buffer, &req);
    type = ct_find_memory_type(&e, req.memoryTypeBits, 0,
                               VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
    CT_REQUIRE(t, type != UINT32_MAX);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    CT_REQUIRE_VK(t, vkAllocateMemory(e.device, &alloc, NULL, &memory));
    ct_host_buffer(t, &e, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src);
    ct_host_buffer(t, &e, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst);
    ct_fill_pattern(src.data, (size_t)size, 11);
    memset(&sinfo, 0, sizeof(sinfo));
    sinfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreateSemaphore(e.device, &sinfo, NULL, &bound));
    memset(&finfo, 0, sizeof(finfo));
    finfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreateFence(e.device, &finfo, NULL, &fence));

    /* the whole buffer, in units of the alignment (the sparse block size) */
    memset(&bind, 0, sizeof(bind));
    bind.size = req.size;
    bind.memory = memory;
    memset(&buffer_bind, 0, sizeof(buffer_bind));
    buffer_bind.buffer = buffer;
    buffer_bind.bindCount = 1;
    buffer_bind.pBinds = &bind;
    memset(&sparse, 0, sizeof(sparse));
    sparse.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
    sparse.bufferBindCount = 1;
    sparse.pBufferBinds = &buffer_bind;
    sparse.signalSemaphoreCount = 1;
    sparse.pSignalSemaphores = &bound;
    CT_REQUIRE_VK(t, vkQueueBindSparse(e.queue, 1, &sparse, VK_NULL_HANDLE));
    CT_REQUIRE_VK(t, vkQueueBindSparse(e.queue, 0, NULL, fence));
    CT_CHECK_VK(t, vkWaitForFences(e.device, 1, &fence, VK_TRUE, 10000000000ull), VK_SUCCESS);

    cb = ct_begin_commands(t, &e);
    memset(&region, 0, sizeof(region));
    region.size = size;
    vkCmdCopyBuffer(cb, src.buffer, buffer, 1, &region);
    ct_transfer_barrier(cb);
    vkCmdCopyBuffer(cb, buffer, dst.buffer, 1, &region);
    ct_host_read_barrier(cb);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &bound;
    submit.pWaitDstStageMask = &stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    CT_REQUIRE_VK(t, vkQueueSubmit(e.queue, 1, &submit, VK_NULL_HANDLE));
    CT_REQUIRE_VK(t, vkQueueWaitIdle(e.queue));
    CT_CHECK(t, memcmp(dst.data, src.data, (size_t)size) == 0);
    vkFreeCommandBuffers(e.device, e.pool, 1, &cb);

    vkDestroyFence(e.device, fence, NULL);
    vkDestroySemaphore(e.device, bound, NULL);
    vkDestroyBuffer(e.device, buffer, NULL);
    vkFreeMemory(e.device, memory, NULL);
    ct_buffer_fini(&e, &src);
    ct_buffer_fini(&e, &dst);
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(sparse_requirements_non_sparse),
    CT_REF(sparse_requirements_residency),
    CT_REF(queue_bind_sparse),
};

CT_SUITE(sparsemem, tests)
