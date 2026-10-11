/* Chapter "Sparse Resources": the sparse memory requirements of buffers and
 * images (none for images without sparse residency), sparse image format
 * properties and their block shapes, mip tail rules, and binding memory with
 * vkQueueBindSparse where a queue family supports sparse binding: buffer
 * blocks (residency, rebinding, unbinding, aliasing), image blocks, mip
 * tails, fully resident images bound opaquely, 3D images, and the order of
 * batches. Accesses to unbound regions are checked only with
 * residencyNonResidentStrict. */
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


/* ---- Sparse binding: shared helpers ----
 *
 * A device with one queue of a family that supports sparse binding and
 * transfers (graphics, compute or transfer), with every sparse feature the
 * device reports enabled; the test skips without sparseBinding or such a
 * family. */
static void sp_device(struct ct *t, struct ct_env *e, struct ct_features *f)
{
    VkQueueFamilyProperties families[64];
    uint32_t count = 64, family = UINT32_MAX, i;
    VkPhysicalDeviceFeatures enable;
    VkDeviceQueueCreateInfo queue;
    VkDeviceCreateInfo info;
    const char *names[2];
    float priority = 1.0f;

    ct_get_features(e, f, 0);
    if (!f->f2.features.sparseBinding)
        ct_skip(t, "sparseBinding is not supported");
    vkGetPhysicalDeviceQueueFamilyProperties(e->pd, &count, families);
    for (i = 0; i < count && family == UINT32_MAX; i++)
        if ((families[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) &&
            (families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)))
            family = i;
    if (family == UINT32_MAX)
        ct_skip(t, "no queue family supports sparse binding and transfers");
    memset(&enable, 0, sizeof(enable));
    enable.sparseBinding = VK_TRUE;
    enable.sparseResidencyBuffer = f->f2.features.sparseResidencyBuffer;
    enable.sparseResidencyImage2D = f->f2.features.sparseResidencyImage2D;
    enable.sparseResidencyImage3D = f->f2.features.sparseResidencyImage3D;
    enable.sparseResidencyAliased = f->f2.features.sparseResidencyAliased;
    memset(&queue, 0, sizeof(queue));
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueFamilyIndex = family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue;
    info.enabledExtensionCount = ct_add_portability_subset(e, names, 0);
    info.ppEnabledExtensionNames = info.enabledExtensionCount ? names : NULL;
    info.pEnabledFeatures = &enable;
    CT_REQUIRE_VK(t, vkCreateDevice(e->pd, &info, NULL, &e->device));
    e->queue_family = family;
    vkGetDeviceQueue(e->device, family, 0, &e->queue);
}

/* Memory of a type the requirements allow (host-visible and coherent if
 * `host` and there is one, else any), neither protected nor lazily
 * allocated; *host_data: its mapping (NULL if not host-visible). */
static VkDeviceMemory sp_memory(struct ct *t, struct ct_env *e, uint32_t type_bits, VkDeviceSize size, int host,
                                unsigned char **host_data)
{
    const VkMemoryPropertyFlags bad = VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
    const VkMemoryPropertyFlags visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkMemoryAllocateInfo alloc;
    VkDeviceMemory memory;
    uint32_t type = host ? ct_find_memory_type(e, type_bits, visible, bad) : UINT32_MAX;
    void *p = NULL;

    if (type == UINT32_MAX)
        type = ct_find_memory_type(e, type_bits, 0, bad);
    CT_REQUIREF(t, type != UINT32_MAX, "no memory type for sparse binding in 0x%x", type_bits);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = size;
    alloc.memoryTypeIndex = type;
    CT_REQUIRE_VK(t, vkAllocateMemory(e->device, &alloc, NULL, &memory));
    if (host_data) {
        *host_data = NULL;
        if ((e->mem.memoryTypes[type].propertyFlags & visible) == visible) {
            CT_REQUIRE_VK(t, vkMapMemory(e->device, memory, 0, VK_WHOLE_SIZE, 0, &p));
            *host_data = p;
        }
    }
    return memory;
}

/* One batch of binds, then a fence of no batches waited for (it signals
 * after the binds: Synchronization > Fences). */
static void sp_bind(struct ct *t, struct ct_env *e, const VkBindSparseInfo *info)
{
    VkFenceCreateInfo finfo;
    VkFence fence;
    memset(&finfo, 0, sizeof(finfo));
    finfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreateFence(e->device, &finfo, NULL, &fence));
    CT_CHECK_VK(t, vkQueueBindSparse(e->queue, 1, info, fence), VK_SUCCESS);
    CT_CHECK_VK(t, vkWaitForFences(e->device, 1, &fence, VK_TRUE, 10000000000ull), VK_SUCCESS);
    vkDestroyFence(e->device, fence, NULL);
}

static void sp_bind_buffer(struct ct *t, struct ct_env *e, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                           VkDeviceMemory memory, VkDeviceSize memory_offset)
{
    VkSparseMemoryBind bind;
    VkSparseBufferMemoryBindInfo bi;
    VkBindSparseInfo info;
    memset(&bind, 0, sizeof(bind));
    bind.resourceOffset = offset;
    bind.size = size;
    bind.memory = memory;
    bind.memoryOffset = memory_offset;
    bi.buffer = buffer;
    bi.bindCount = 1;
    bi.pBinds = &bind;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
    info.bufferBindCount = 1;
    info.pBufferBinds = &bi;
    sp_bind(t, e, &info);
}

static void sp_bind_opaque(struct ct *t, struct ct_env *e, VkImage image, VkDeviceSize offset, VkDeviceSize size,
                           VkDeviceMemory memory, VkDeviceSize memory_offset)
{
    VkSparseMemoryBind bind;
    VkSparseImageOpaqueMemoryBindInfo bi;
    VkBindSparseInfo info;
    memset(&bind, 0, sizeof(bind));
    bind.resourceOffset = offset;
    bind.size = size;
    bind.memory = memory;
    bind.memoryOffset = memory_offset;
    bi.image = image;
    bi.bindCount = 1;
    bi.pBinds = &bind;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
    info.imageOpaqueBindCount = 1;
    info.pImageOpaqueBinds = &bi;
    sp_bind(t, e, &info);
}

/* Binds level `level`, layer `layer` texels (x, y, z) + extent (whole blocks
 * or up to the subresource's edge). */
static void sp_bind_image(struct ct *t, struct ct_env *e, VkImage image, uint32_t level, uint32_t layer,
                          VkOffset3D offset, VkExtent3D extent, VkDeviceMemory memory, VkDeviceSize memory_offset)
{
    VkSparseImageMemoryBind bind;
    VkSparseImageMemoryBindInfo bi;
    VkBindSparseInfo info;
    memset(&bind, 0, sizeof(bind));
    bind.subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bind.subresource.mipLevel = level;
    bind.subresource.arrayLayer = layer;
    bind.offset = offset;
    bind.extent = extent;
    bind.memory = memory;
    bind.memoryOffset = memory_offset;
    bi.image = image;
    bi.bindCount = 1;
    bi.pBinds = &bind;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
    info.imageBindCount = 1;
    info.pImageBinds = &bi;
    sp_bind(t, e, &info);
}

static VkBuffer sp_buffer(struct ct *t, struct ct_env *e, VkDeviceSize size, VkBufferCreateFlags flags,
                          VkMemoryRequirements *req)
{
    VkBufferCreateInfo info;
    VkBuffer buffer;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | flags;
    info.size = size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    CT_REQUIRE_VK(t, vkCreateBuffer(e->device, &info, NULL, &buffer));
    vkGetBufferMemoryRequirements(e->device, buffer, req);
    return buffer;
}

static void sp_copy(VkCommandBuffer cb, VkBuffer src, VkDeviceSize src_offset, VkBuffer dst, VkDeviceSize dst_offset,
                    VkDeviceSize size)
{
    VkBufferCopy region;
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = size;
    vkCmdCopyBuffer(cb, src, dst, 1, &region);
}

static void sp_image_copy(VkBufferImageCopy *r, VkDeviceSize offset, uint32_t level, uint32_t layer, VkOffset3D o,
                          VkExtent3D x)
{
    memset(r, 0, sizeof(*r));
    r->bufferOffset = offset;
    r->imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    r->imageSubresource.mipLevel = level;
    r->imageSubresource.baseArrayLayer = layer;
    r->imageSubresource.layerCount = 1;
    r->imageOffset = o;
    r->imageExtent = x;
}

static VkOffset3D sp_offset(int32_t x, int32_t y, int32_t z)
{
    VkOffset3D o;
    o.x = x;
    o.y = y;
    o.z = z;
    return o;
}

static VkExtent3D sp_extent(uint32_t w, uint32_t h, uint32_t d)
{
    VkExtent3D x;
    x.width = w;
    x.height = h;
    x.depth = d;
    return x;
}

/* The image's sparse memory requirement for the color aspect. */
static VkSparseImageMemoryRequirements sp_color_requirements(struct ct *t, struct ct_env *e, VkImage image)
{
    VkSparseImageMemoryRequirements r[8];
    uint32_t count = 8, i;
    vkGetImageSparseMemoryRequirements(e->device, image, &count, r);
    for (i = 0; i < count; i++)
        if (r[i].formatProperties.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT)
            return r[i];
    CT_REQUIREF(t, 0, "no sparse requirement for the color aspect (%u requirements)", count);
    return r[0];
}

/* Whether 2D (or 3D) sparse residency images of the info's parameters can be created. */
static int sp_image_supported(struct ct_env *e, const VkImageCreateInfo *info)
{
    VkImageFormatProperties fp;
    VkSparseImageFormatProperties sp[4];
    uint32_t count = 4;
    if (vkGetPhysicalDeviceImageFormatProperties(e->pd, info->format, info->imageType, info->tiling, info->usage,
                                                 info->flags, &fp) != VK_SUCCESS ||
        fp.maxMipLevels < info->mipLevels || fp.maxArrayLayers < info->arrayLayers ||
        fp.maxExtent.width < info->extent.width || fp.maxExtent.height < info->extent.height ||
        fp.maxExtent.depth < info->extent.depth)
        return 0;
    if (!(info->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT))
        return 1;
    vkGetPhysicalDeviceSparseImageFormatProperties(e->pd, info->format, info->imageType, info->samples, info->usage,
                                                   info->tiling, &count, sp);
    return count > 0;
}

static uint32_t sp_pattern(uint32_t i, uint32_t seed)
{
    return (i * 2654435761u) ^ (seed * 0x9E3779B9u) ^ 0x01010101u; /* never 0 */
}

static void sp_fill_words(unsigned char *p, uint32_t n, uint32_t seed)
{
    uint32_t i, w;
    for (i = 0; i < n; i++) {
        w = sp_pattern(i, seed);
        memcpy(p + 4 * i, &w, 4);
    }
}

static int sp_words_are(const unsigned char *p, uint32_t n, uint32_t seed)
{
    uint32_t i, w;
    for (i = 0; i < n; i++) {
        memcpy(&w, p + 4 * i, 4);
        if (w != sp_pattern(i, seed))
            return 0;
    }
    return 1;
}

/* ---- Sparse buffers ---- */

CT_TEST(sparse_buffer_requirements,
        "Sparse Resources > Sparse Buffer and Fully-Resident Image Block Size: VkMemoryRequirements::alignment of a "
        "sparse buffer is its sparse block size (a power of two, Resource Memory Association); its size covers the "
        "buffer and (derived: VUID-VkSparseMemoryBind-resourceOffset-09491 binds whole blocks) is a whole number of "
        "blocks; memoryTypeBits is nonzero and the same for buffers of the same flags and usage; and (Vulkan 1.3) "
        "vkGetDeviceBufferMemoryRequirements reports the same for the create info",
        "vkCreateDevice vkDestroyDevice vkGetDeviceQueue vkGetPhysicalDeviceQueueFamilyProperties "
        "vkGetPhysicalDeviceFeatures2 vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements "
        "vkGetDeviceBufferMemoryRequirements")
{
    static const VkDeviceSize sizes[] = { 1, 4096, 100000, 1 << 20 };
    struct ct_env e;
    struct ct_features f;
    VkBufferCreateFlags flags;
    VkMemoryRequirements req, first;
    uint32_t i, k;

    ct_env_init(t, &e);
    sp_device(t, &e, &f);
    for (k = 0; k < 2; k++) {
        flags = k ? VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT : 0;
        if (k && !f.f2.features.sparseResidencyBuffer)
            break;
        for (i = 0; i < CT_ARRAY_SIZE(sizes); i++) {
            VkBuffer buffer = sp_buffer(t, &e, sizes[i], flags, &req);
            CT_CHECKF(t, ct_is_pow2(req.alignment), "alignment %llu", (unsigned long long)req.alignment);
            CT_CHECKF(t, req.size >= sizes[i] && req.size % req.alignment == 0, "size %llu for %llu bytes",
                      (unsigned long long)req.size, (unsigned long long)sizes[i]);
            CT_CHECK(t, req.memoryTypeBits != 0);
            if (i == 0)
                first = req;
            else
                CT_CHECKF(t, req.memoryTypeBits == first.memoryTypeBits && req.alignment == first.alignment,
                          "buffer %u: memoryTypeBits 0x%x / 0x%x, alignment %llu / %llu", i, req.memoryTypeBits,
                          first.memoryTypeBits, (unsigned long long)req.alignment,
                          (unsigned long long)first.alignment);
            if (e.version >= VK_API_VERSION_1_3) {
                VkBufferCreateInfo info;
                VkDeviceBufferMemoryRequirements dinfo;
                VkMemoryRequirements2 dreq;
                memset(&info, 0, sizeof(info));
                info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                info.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | flags;
                info.size = sizes[i];
                info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                memset(&dinfo, 0, sizeof(dinfo));
                dinfo.sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS;
                dinfo.pCreateInfo = &info;
                memset(&dreq, 0, sizeof(dreq));
                dreq.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
                vkGetDeviceBufferMemoryRequirements(e.device, &dinfo, &dreq);
                CT_CHECKF(t, memcmp(&dreq.memoryRequirements, &req, sizeof(req)) == 0,
                          "vkGetDeviceBufferMemoryRequirements differs for %llu bytes", (unsigned long long)sizes[i]);
            }
            vkDestroyBuffer(e.device, buffer, NULL);
        }
    }
    ct_env_fini(&e);
}

CT_TEST(sparse_buffer_residency,
        "Sparse Resources > Sparse Partially-Resident Buffers (sparseResidencyBuffer): a bound block of a buffer "
        "accesses the bound range of its memory (VkSparseMemoryBind: resourceOffset to memoryOffset), the other "
        "blocks need not be bound; with residencyNonResidentStrict, reads of the unbound blocks return 0 and "
        "writes to them are discarded (also after a memory dependency in the same command buffer)",
        "vkCreateDevice vkDestroyDevice vkGetDeviceQueue vkGetPhysicalDeviceQueueFamilyProperties "
        "vkGetPhysicalDeviceFeatures2 vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory "
        "vkFreeMemory vkBindBufferMemory vkMapMemory vkQueueBindSparse vkCreateFence vkDestroyFence vkWaitForFences "
        "vkQueueSubmit vkResetFences vkCmdCopyBuffer vkCmdPipelineBarrier vkCreateCommandPool vkDestroyCommandPool "
        "vkAllocateCommandBuffers vkBeginCommandBuffer vkEndCommandBuffer")
{
    struct ct_env e;
    struct ct_features f;
    struct ct_buffer src, dst;
    VkMemoryRequirements req;
    VkCommandBuffer cb;
    VkDeviceMemory memory;
    VkBuffer buffer;
    VkDeviceSize block, size;
    unsigned char *mem_data, *zero;
    uint32_t b;

    ct_env_init(t, &e);
    sp_device(t, &e, &f);
    if (!f.f2.features.sparseResidencyBuffer)
        ct_skip(t, "sparseResidencyBuffer is not supported");
    buffer = sp_buffer(t, &e, 1, VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT, &req);
    block = req.alignment;
    vkDestroyBuffer(e.device, buffer, NULL);
    size = 4 * block;
    buffer = sp_buffer(t, &e, size, VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT, &req);
    memory = sp_memory(t, &e, req.memoryTypeBits, 3 * block, 1, &mem_data);
    ct_host_buffer(t, &e, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src);
    ct_host_buffer(t, &e, 2 * size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst);
    zero = calloc(1, (size_t)block);
    CT_REQUIRE(t, zero != NULL);

    /* blocks 1 and 2 at memory blocks 2 and 0 */
    sp_bind_buffer(t, &e, buffer, block, block, memory, 2 * block);
    sp_bind_buffer(t, &e, buffer, 2 * block, block, memory, 0);
    sp_fill_words(src.data, (uint32_t)(size / 4), 1);
    cb = ct_begin_commands(t, &e);
    sp_copy(cb, src.buffer, block, buffer, block, 2 * block);
    ct_transfer_barrier(cb);
    sp_copy(cb, buffer, block, dst.buffer, block, 2 * block);
    ct_host_read_barrier(cb);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    ct_submit_and_wait(t, &e, cb);
    CT_CHECK(t, memcmp(dst.data + block, src.data + block, (size_t)(2 * block)) == 0);
    if (mem_data) {
        CT_CHECKF(t, memcmp(mem_data + 2 * block, src.data + block, (size_t)block) == 0,
                  "block 1's data is not at memory offset 2 blocks");
        CT_CHECKF(t, memcmp(mem_data, src.data + 2 * block, (size_t)block) == 0,
                  "block 2's data is not at memory offset 0");
    }
    if (!e.props.sparseProperties.residencyNonResidentStrict) {
        ct_note(t, "residencyNonResidentStrict is not supported: unbound blocks not checked");
    } else {
        /* the whole buffer, in one command buffer: unbound blocks drop the writes and read 0 */
        sp_fill_words(src.data, (uint32_t)(size / 4), 2);
        cb = ct_begin_commands(t, &e);
        sp_copy(cb, src.buffer, 0, buffer, 0, size);
        ct_transfer_barrier(cb);
        sp_copy(cb, buffer, 0, dst.buffer, size, size);
        ct_host_read_barrier(cb);
        CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
        ct_submit_and_wait(t, &e, cb);
        for (b = 0; b < 4; b++) {
            const unsigned char *got = dst.data + size + b * block;
            if (b == 1 || b == 2)
                CT_CHECKF(t, memcmp(got, src.data + b * block, (size_t)block) == 0, "bound block %u", b);
            else
                CT_CHECKF(t, memcmp(got, zero, (size_t)block) == 0, "unbound block %u does not read 0", b);
        }
    }
    free(zero);
    vkDestroyBuffer(e.device, buffer, NULL);
    vkFreeMemory(e.device, memory, NULL);
    ct_buffer_fini(&e, &src);
    ct_buffer_fini(&e, &dst);
    ct_env_fini(&e);
}

CT_TEST(sparse_buffer_rebind,
        "Sparse Resources > Binding Resource Memory: binding a block to other memory makes it access that memory "
        "(whose contents binding leaves alone: the earlier memory's data is there again when bound back); with "
        "sparseResidencyBuffer a VkSparseMemoryBind with memory VK_NULL_HANDLE unbinds the range, which with "
        "residencyNonResidentStrict then reads 0",
        "vkCreateDevice vkDestroyDevice vkGetDeviceQueue vkGetPhysicalDeviceQueueFamilyProperties "
        "vkGetPhysicalDeviceFeatures2 vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory "
        "vkFreeMemory vkBindBufferMemory vkMapMemory vkQueueBindSparse vkCreateFence vkDestroyFence vkWaitForFences "
        "vkQueueSubmit vkResetFences vkCmdCopyBuffer vkCmdPipelineBarrier vkCreateCommandPool vkDestroyCommandPool "
        "vkAllocateCommandBuffers vkBeginCommandBuffer vkEndCommandBuffer")
{
    struct ct_env e;
    struct ct_features f;
    struct ct_buffer src, dst;
    VkMemoryRequirements req;
    VkDeviceMemory a, b;
    VkBuffer buffer;
    VkCommandBuffer cb;
    VkDeviceSize block;
    int residency;
    uint32_t step;
    /* bind a, write 1; bind b, write 2; bind a: 1; bind b: 2; (residency) unbind: 0 */
    static const int which[5] = { 0, 1, 0, 1, -1 };

    ct_env_init(t, &e);
    sp_device(t, &e, &f);
    residency = f.f2.features.sparseResidencyBuffer;
    buffer = sp_buffer(t, &e, 1, residency ? VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT : 0, &req);
    block = req.alignment;
    vkDestroyBuffer(e.device, buffer, NULL);
    buffer = sp_buffer(t, &e, block, residency ? VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT : 0, &req);
    a = sp_memory(t, &e, req.memoryTypeBits, block, 0, NULL);
    b = sp_memory(t, &e, req.memoryTypeBits, 2 * block, 0, NULL);
    ct_host_buffer(t, &e, 2 * block, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src);
    ct_host_buffer(t, &e, block, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst);
    sp_fill_words(src.data, (uint32_t)(block / 4), 1);
    sp_fill_words(src.data + block, (uint32_t)(block / 4), 2);
    for (step = 0; step < 5; step++) {
        if (which[step] < 0 && !(residency && e.props.sparseProperties.residencyNonResidentStrict))
            break;
        sp_bind_buffer(t, &e, buffer, 0, block, which[step] < 0 ? VK_NULL_HANDLE : which[step] ? b : a,
                       which[step] == 1 ? block : 0);
        cb = ct_begin_commands(t, &e);
        if (step < 2) { /* write the memory's data through the buffer */
            sp_copy(cb, src.buffer, step * block, buffer, 0, block);
            ct_transfer_barrier(cb);
        }
        sp_copy(cb, buffer, 0, dst.buffer, 0, block);
        ct_host_read_barrier(cb);
        CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
        ct_submit_and_wait(t, &e, cb);
        if (which[step] < 0)
            CT_CHECKF(t, ct_all_bytes(dst.data, (size_t)block, 0), "step %u: the unbound block does not read 0", step);
        else
            CT_CHECKF(t, sp_words_are(dst.data, (uint32_t)(block / 4), (uint32_t)which[step] + 1),
                      "step %u: the block does not access memory %c", step, which[step] ? 'b' : 'a');
    }
    vkDestroyBuffer(e.device, buffer, NULL);
    vkFreeMemory(e.device, a, NULL);
    vkFreeMemory(e.device, b, NULL);
    ct_buffer_fini(&e, &src);
    ct_buffer_fini(&e, &dst);
    ct_env_fini(&e);
}

CT_TEST(sparse_buffer_aliased,
        "Sparse Resources > Sparse Memory Aliasing (sparseResidencyAliased): memory bound at several places, within "
        "one buffer and in two buffers created with VK_BUFFER_CREATE_SPARSE_ALIASED_BIT, is accessed consistently: "
        "after a memory dependency (in the same command buffer), reads through each alias return what was written "
        "through another",
        "vkCreateDevice vkDestroyDevice vkGetDeviceQueue vkGetPhysicalDeviceQueueFamilyProperties "
        "vkGetPhysicalDeviceFeatures2 vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory "
        "vkFreeMemory vkBindBufferMemory vkMapMemory vkQueueBindSparse vkCreateFence vkDestroyFence vkWaitForFences "
        "vkQueueSubmit vkResetFences vkCmdCopyBuffer vkCmdPipelineBarrier vkCreateCommandPool vkDestroyCommandPool "
        "vkAllocateCommandBuffers vkBeginCommandBuffer vkEndCommandBuffer")
{
    struct ct_env e;
    struct ct_features f;
    struct ct_buffer src, dst;
    VkMemoryRequirements req, req2;
    VkDeviceMemory memory;
    VkBuffer x, y;
    VkCommandBuffer cb;
    VkDeviceSize block;
    const VkBufferCreateFlags flags = VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT | VK_BUFFER_CREATE_SPARSE_ALIASED_BIT;

    ct_env_init(t, &e);
    sp_device(t, &e, &f);
    if (!f.f2.features.sparseResidencyBuffer || !f.f2.features.sparseResidencyAliased)
        ct_skip(t, "sparseResidencyBuffer with sparseResidencyAliased is not supported");
    x = sp_buffer(t, &e, 1, flags, &req);
    block = req.alignment;
    vkDestroyBuffer(e.device, x, NULL);
    x = sp_buffer(t, &e, 2 * block, flags, &req);
    y = sp_buffer(t, &e, block, flags, &req2);
    memory = sp_memory(t, &e, req.memoryTypeBits & req2.memoryTypeBits, block, 0, NULL);
    ct_host_buffer(t, &e, 2 * block, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src);
    ct_host_buffer(t, &e, 3 * block, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst);
    sp_bind_buffer(t, &e, x, 0, block, memory, 0);
    sp_bind_buffer(t, &e, x, block, block, memory, 0);
    sp_bind_buffer(t, &e, y, 0, block, memory, 0);
    sp_fill_words(src.data, (uint32_t)(block / 4), 3);
    sp_fill_words(src.data + block, (uint32_t)(block / 4), 4);
    cb = ct_begin_commands(t, &e);
    sp_copy(cb, src.buffer, 0, x, 0, block); /* write x block 0 */
    ct_transfer_barrier(cb);
    sp_copy(cb, x, block, dst.buffer, 0, block); /* read x block 1 */
    sp_copy(cb, y, 0, dst.buffer, block, block); /* read y */
    ct_transfer_barrier(cb);
    sp_copy(cb, src.buffer, block, y, 0, block); /* write y */
    ct_transfer_barrier(cb);
    sp_copy(cb, x, 0, dst.buffer, 2 * block, block); /* read x block 0 */
    ct_host_read_barrier(cb);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    ct_submit_and_wait(t, &e, cb);
    CT_CHECKF(t, sp_words_are(dst.data, (uint32_t)(block / 4), 3), "x block 1 does not see x block 0's write");
    CT_CHECKF(t, sp_words_are(dst.data + block, (uint32_t)(block / 4), 3), "y does not see x's write");
    CT_CHECKF(t, sp_words_are(dst.data + 2 * block, (uint32_t)(block / 4), 4), "x does not see y's write");
    vkDestroyBuffer(e.device, x, NULL);
    vkDestroyBuffer(e.device, y, NULL);
    vkFreeMemory(e.device, memory, NULL);
    ct_buffer_fini(&e, &src);
    ct_buffer_fini(&e, &dst);
    ct_env_fini(&e);
}

CT_TEST(bind_sparse_batches_order,
        "Sparse Resources > Binding Resource Memory: rebinding a range in a later batch of one vkQueueBindSparse that "
        "waits on a semaphore the earlier batch signals takes effect after the earlier bind; work that waits on the "
        "later batch's semaphore sees the later binding (Synchronization > Semaphore Signaling/Waiting)",
        "vkCreateDevice vkDestroyDevice vkGetDeviceQueue vkGetPhysicalDeviceQueueFamilyProperties "
        "vkGetPhysicalDeviceFeatures2 vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory "
        "vkFreeMemory vkBindBufferMemory vkMapMemory vkQueueBindSparse vkCreateSemaphore vkDestroySemaphore "
        "vkCreateFence vkDestroyFence vkWaitForFences vkQueueSubmit vkResetFences vkCmdCopyBuffer "
        "vkCmdPipelineBarrier vkCreateCommandPool vkDestroyCommandPool vkAllocateCommandBuffers vkBeginCommandBuffer "
        "vkEndCommandBuffer")
{
    struct ct_env e;
    struct ct_features f;
    struct ct_buffer src, dst;
    VkMemoryRequirements req;
    VkDeviceMemory a, b;
    VkBuffer buffer;
    VkCommandBuffer cb;
    VkDeviceSize block;
    VkSemaphoreCreateInfo sinfo;
    VkSemaphore s1, s2;
    VkSparseMemoryBind binds[2];
    VkSparseBufferMemoryBindInfo bi[2];
    VkBindSparseInfo infos[2];
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit;
    VkFenceCreateInfo finfo;
    VkFence fence;

    ct_env_init(t, &e);
    sp_device(t, &e, &f);
    buffer = sp_buffer(t, &e, 1, 0, &req);
    block = req.alignment;
    vkDestroyBuffer(e.device, buffer, NULL);
    buffer = sp_buffer(t, &e, block, 0, &req);
    a = sp_memory(t, &e, req.memoryTypeBits, block, 0, NULL);
    b = sp_memory(t, &e, req.memoryTypeBits, block, 0, NULL);
    ct_host_buffer(t, &e, 2 * block, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src);
    ct_host_buffer(t, &e, block, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst);
    memset(&sinfo, 0, sizeof(sinfo));
    sinfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreateSemaphore(e.device, &sinfo, NULL, &s1));
    CT_REQUIRE_VK(t, vkCreateSemaphore(e.device, &sinfo, NULL, &s2));
    sp_fill_words(src.data, (uint32_t)(block / 4), 5);
    sp_fill_words(src.data + block, (uint32_t)(block / 4), 6);

    /* memory b gets pattern 6 through the buffer first */
    sp_bind_buffer(t, &e, buffer, 0, block, b, 0);
    cb = ct_begin_commands(t, &e);
    sp_copy(cb, src.buffer, block, buffer, 0, block);
    ct_host_read_barrier(cb);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    ct_submit_and_wait(t, &e, cb);
    /* and memory a pattern 5 */
    sp_bind_buffer(t, &e, buffer, 0, block, a, 0);
    cb = ct_begin_commands(t, &e);
    sp_copy(cb, src.buffer, 0, buffer, 0, block);
    ct_host_read_barrier(cb);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    ct_submit_and_wait(t, &e, cb);

    /* one call: batch 0 binds a and signals s1; batch 1 waits on s1, binds b, signals s2 */
    memset(binds, 0, sizeof(binds));
    binds[0].size = binds[1].size = block;
    binds[0].memory = a;
    binds[1].memory = b;
    bi[0].buffer = bi[1].buffer = buffer;
    bi[0].bindCount = bi[1].bindCount = 1;
    bi[0].pBinds = &binds[0];
    bi[1].pBinds = &binds[1];
    memset(infos, 0, sizeof(infos));
    infos[0].sType = infos[1].sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
    infos[0].bufferBindCount = infos[1].bufferBindCount = 1;
    infos[0].pBufferBinds = &bi[0];
    infos[0].signalSemaphoreCount = 1;
    infos[0].pSignalSemaphores = &s1;
    infos[1].pBufferBinds = &bi[1];
    infos[1].waitSemaphoreCount = 1;
    infos[1].pWaitSemaphores = &s1;
    infos[1].signalSemaphoreCount = 1;
    infos[1].pSignalSemaphores = &s2;
    CT_REQUIRE_VK(t, vkQueueBindSparse(e.queue, 2, infos, VK_NULL_HANDLE));
    cb = ct_begin_commands(t, &e);
    sp_copy(cb, buffer, 0, dst.buffer, 0, block);
    ct_host_read_barrier(cb);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &s2;
    submit.pWaitDstStageMask = &stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    memset(&finfo, 0, sizeof(finfo));
    finfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreateFence(e.device, &finfo, NULL, &fence));
    CT_REQUIRE_VK(t, vkQueueSubmit(e.queue, 1, &submit, fence));
    CT_CHECK_VK(t, vkWaitForFences(e.device, 1, &fence, VK_TRUE, 10000000000ull), VK_SUCCESS);
    CT_CHECKF(t, sp_words_are(dst.data, (uint32_t)(block / 4), 6), "the later batch's binding is not in effect");

    vkDestroyFence(e.device, fence, NULL);
    vkDestroySemaphore(e.device, s1, NULL);
    vkDestroySemaphore(e.device, s2, NULL);
    vkDestroyBuffer(e.device, buffer, NULL);
    vkFreeMemory(e.device, a, NULL);
    vkFreeMemory(e.device, b, NULL);
    ct_buffer_fini(&e, &src);
    ct_buffer_fini(&e, &dst);
    ct_env_fini(&e);
}

/* ---- Sparse images ---- */

static int sp_same_format_properties(const VkSparseImageFormatProperties *a, const VkSparseImageFormatProperties *b)
{
    return a->aspectMask == b->aspectMask && a->imageGranularity.width == b->imageGranularity.width &&
           a->imageGranularity.height == b->imageGranularity.height &&
           a->imageGranularity.depth == b->imageGranularity.depth && a->flags == b->flags;
}

CT_TEST(sparse_image_format_queries,
        "Sparse Resources > Sparse Image Format Properties (sparseResidencyImage2D): where vkGetPhysicalDeviceImageFormatProperties "
        "supports VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT for a 2D color format, vkGetPhysicalDeviceSparseImageFormatProperties "
        "reports it (derived: else the count is zero), with the color aspect and a nonzero granularity; with "
        "residencyStandard2DBlockShape the granularity is the standard block shape (in texels) and "
        "VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT is not set; vkGetPhysicalDeviceSparseImageFormatProperties2 "
        "reports the same; a count of 0 with an array writes nothing (Fundamentals > Array Results)",
        "vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceImageFormatProperties "
        "vkGetPhysicalDeviceSparseImageFormatProperties vkGetPhysicalDeviceSparseImageFormatProperties2")
{
    static const struct {
        VkFormat format;
        uint32_t w, h; /* standard 2D block shape in texels */
    } formats[] = {
        { VK_FORMAT_R8_UNORM, 256, 256 },           { VK_FORMAT_R8G8_UNORM, 256, 128 },
        { VK_FORMAT_R16_SFLOAT, 256, 128 },         { VK_FORMAT_R8G8B8A8_UNORM, 128, 128 },
        { VK_FORMAT_R32_UINT, 128, 128 },           { VK_FORMAT_R16G16B16A16_SFLOAT, 128, 64 },
        { VK_FORMAT_R32G32_SFLOAT, 128, 64 },       { VK_FORMAT_R32G32B32A32_SFLOAT, 64, 64 },
        { VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 512, 256 }, { VK_FORMAT_BC7_UNORM_BLOCK, 256, 256 },
        { VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 512, 256 }, { VK_FORMAT_ASTC_4x4_UNORM_BLOCK, 256, 256 },
        { VK_FORMAT_ASTC_8x8_UNORM_BLOCK, 512, 512 },
    };
    struct ct_env e;
    struct ct_features f;
    VkSparseImageFormatProperties p[4], guard;
    uint32_t i, k, count, supported = 0;

    ct_env_init(t, &e);
    ct_get_features(&e, &f, 0);
    if (!f.f2.features.sparseResidencyImage2D)
        ct_skip(t, "sparseResidencyImage2D is not supported");
    for (i = 0; i < CT_ARRAY_SIZE(formats); i++) {
        VkImageFormatProperties fp;
        int color = 0;
        if (vkGetPhysicalDeviceImageFormatProperties(e.pd, formats[i].format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                     VK_IMAGE_USAGE_SAMPLED_BIT,
                                                     VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
                                                     &fp) != VK_SUCCESS)
            continue;
        supported++;
        count = 4;
        memset(p, 0, sizeof(p));
        vkGetPhysicalDeviceSparseImageFormatProperties(e.pd, formats[i].format, VK_IMAGE_TYPE_2D, VK_SAMPLE_COUNT_1_BIT,
                                                       VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_TILING_OPTIMAL, &count, p);
        CT_CHECKF(t, count >= 1, "format %d: sparse residency supported, but no sparse properties", (int)formats[i].format);
        for (k = 0; k < count && k < 4; k++) {
            color |= (p[k].aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
            CT_CHECKF(t, p[k].imageGranularity.width && p[k].imageGranularity.height && p[k].imageGranularity.depth,
                      "format %d: granularity %ux%ux%u", (int)formats[i].format, p[k].imageGranularity.width,
                      p[k].imageGranularity.height, p[k].imageGranularity.depth);
            if (e.props.sparseProperties.residencyStandard2DBlockShape && (p[k].aspectMask & VK_IMAGE_ASPECT_COLOR_BIT))
                CT_CHECKF(t, p[k].imageGranularity.width == formats[i].w && p[k].imageGranularity.height == formats[i].h &&
                                 p[k].imageGranularity.depth == 1 &&
                                 !(p[k].flags & VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT),
                          "format %d: granularity %ux%ux%u flags 0x%x, standard shape %ux%u", (int)formats[i].format,
                          p[k].imageGranularity.width, p[k].imageGranularity.height, p[k].imageGranularity.depth,
                          p[k].flags, formats[i].w, formats[i].h);
        }
        CT_CHECKF(t, color || !count, "format %d: no properties for the color aspect", (int)formats[i].format);
        if (e.version >= VK_API_VERSION_1_1) {
            VkPhysicalDeviceSparseImageFormatInfo2 info;
            VkSparseImageFormatProperties2 p2[4];
            uint32_t n = 4;
            memset(&info, 0, sizeof(info));
            info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2;
            info.format = formats[i].format;
            info.type = VK_IMAGE_TYPE_2D;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            memset(p2, 0, sizeof(p2));
            for (k = 0; k < 4; k++)
                p2[k].sType = VK_STRUCTURE_TYPE_SPARSE_IMAGE_FORMAT_PROPERTIES_2;
            vkGetPhysicalDeviceSparseImageFormatProperties2(e.pd, &info, &n, p2);
            CT_CHECKF(t, n == count, "format %d: %u properties, %u from the 2 query", (int)formats[i].format, count, n);
            for (k = 0; k < n && k < count && k < 4; k++)
                CT_CHECKF(t, sp_same_format_properties(&p2[k].properties, &p[k]), "format %d: properties2 %u differ",
                          (int)formats[i].format, k);
        }
        count = 0;
        memset(&guard, SP_GUARD, sizeof(guard));
        vkGetPhysicalDeviceSparseImageFormatProperties(e.pd, formats[i].format, VK_IMAGE_TYPE_2D, VK_SAMPLE_COUNT_1_BIT,
                                                       VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_TILING_OPTIMAL, &count, &guard);
        CT_CHECK(t, count == 0 && sp_untouched(&guard, sizeof(guard)));
    }
    if (!supported)
        ct_note(t, "none of the formats supports 2D sparse residency");
    ct_env_fini(&e);
}

/* The size in bytes of a mip level (texel blocks of `bytes`). */
static VkDeviceSize sp_level_bytes(const VkImageCreateInfo *info, uint32_t level, uint32_t block_w, uint32_t block_h,
                                   uint32_t bytes)
{
    uint32_t w = info->extent.width >> level, h = info->extent.height >> level, d = info->extent.depth >> level;
    w = w ? w : 1;
    h = h ? h : 1;
    d = d ? d : 1;
    return (VkDeviceSize)((w + block_w - 1) / block_w) * ((h + block_h - 1) / block_h) * d * bytes;
}

CT_TEST(sparse_image_requirements_rules,
        "Sparse Resources > Sparse Image Memory Requirements and Mip Tail Regions (sparseResidencyImage2D): the "
        "formatProperties of an image's requirement are vkGetPhysicalDeviceSparseImageFormatProperties' (derived); "
        "imageMipTailFirstLod is at most mipLevels; the tail regions are whole blocks inside the image's memory "
        "requirements (offset + layer * stride + size, or one region with SINGLE_MIPTAIL); levels before the tail "
        "are not smaller than a block in bytes, and with ALIGNED_MIP_SIZE are whole blocks in each dimension",
        "vkCreateDevice vkDestroyDevice vkGetDeviceQueue vkGetPhysicalDeviceQueueFamilyProperties "
        "vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceImageFormatProperties "
        "vkGetPhysicalDeviceSparseImageFormatProperties vkCreateImage vkDestroyImage vkGetImageMemoryRequirements "
        "vkGetImageSparseMemoryRequirements")
{
    struct ct_env e;
    struct ct_features f;
    VkImageCreateInfo infos[4];
    uint32_t i, lv, tested = 0;

    ct_env_init(t, &e);
    sp_device(t, &e, &f);
    if (!f.f2.features.sparseResidencyImage2D)
        ct_skip(t, "sparseResidencyImage2D is not supported");
    infos[0] = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 512, 512, 1, 10, 1, VK_IMAGE_USAGE_SAMPLED_BIT);
    infos[1] = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 1000, 700, 1, 10, 1,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    infos[2] = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8_UNORM, 2048, 64, 1, 12, 3, VK_IMAGE_USAGE_SAMPLED_BIT);
    infos[3] = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, 300, 300, 1, 9, 2,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    for (i = 0; i < CT_ARRAY_SIZE(infos); i++) {
        static const uint32_t bytes[4] = { 4, 4, 1, 8 };
        VkSparseImageMemoryRequirements r;
        VkSparseImageFormatProperties fp[4];
        VkMemoryRequirements req;
        VkImage image;
        uint32_t count = 4, k;
        int same = 0;
        infos[i].flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
        if (!sp_image_supported(&e, &infos[i]))
            continue;
        tested++;
        CT_REQUIRE_VK(t, vkCreateImage(e.device, &infos[i], NULL, &image));
        vkGetImageMemoryRequirements(e.device, image, &req);
        r = sp_color_requirements(t, &e, image);
        vkGetPhysicalDeviceSparseImageFormatProperties(e.pd, infos[i].format, infos[i].imageType, infos[i].samples,
                                                       infos[i].usage, infos[i].tiling, &count, fp);
        for (k = 0; k < count && k < 4; k++)
            same |= sp_same_format_properties(&fp[k], &r.formatProperties);
        CT_CHECKF(t, same, "image %u: formatProperties are not those of the format query", i);
        CT_CHECKF(t, r.imageMipTailFirstLod <= infos[i].mipLevels, "image %u: imageMipTailFirstLod %u", i,
                  r.imageMipTailFirstLod);
        if (r.imageMipTailFirstLod < infos[i].mipLevels) {
            VkDeviceSize end = r.imageMipTailOffset + r.imageMipTailSize;
            if (!(r.formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT))
                end += (infos[i].arrayLayers - 1) * r.imageMipTailStride;
            CT_CHECKF(t, r.imageMipTailOffset % req.alignment == 0 && r.imageMipTailSize % req.alignment == 0 &&
                             r.imageMipTailSize > 0 &&
                             ((r.formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) ||
                              r.imageMipTailStride % req.alignment == 0),
                      "image %u: tail offset %llu size %llu stride %llu, block %llu", i,
                      (unsigned long long)r.imageMipTailOffset, (unsigned long long)r.imageMipTailSize,
                      (unsigned long long)r.imageMipTailStride, (unsigned long long)req.alignment);
            CT_CHECKF(t, end <= req.size, "image %u: tail regions end at %llu, the requirements' size is %llu", i,
                      (unsigned long long)end, (unsigned long long)req.size);
        }
        for (lv = 0; lv < r.imageMipTailFirstLod && lv < infos[i].mipLevels; lv++) {
            uint32_t w = infos[i].extent.width >> lv, h = infos[i].extent.height >> lv;
            CT_CHECKF(t, sp_level_bytes(&infos[i], lv, 1, 1, bytes[i]) >= req.alignment,
                      "image %u: level %u (%ux%u) is smaller than a block but not in the tail", i, lv, w, h);
            if (r.formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_ALIGNED_MIP_SIZE_BIT)
                CT_CHECKF(t, w % r.formatProperties.imageGranularity.width == 0 &&
                                 h % r.formatProperties.imageGranularity.height == 0,
                          "image %u: level %u (%ux%u) is not whole blocks of %ux%u but not in the tail "
                          "(ALIGNED_MIP_SIZE)", i, lv, w, h, r.formatProperties.imageGranularity.width,
                          r.formatProperties.imageGranularity.height);
        }
        vkDestroyImage(e.device, image, NULL);
    }
    if (!tested)
        ct_skip(t, "no 2D sparse residency for the formats tested");
    ct_env_fini(&e);
}

CT_TEST(sparse_image_blocks,
        "Sparse Resources > Sparse Partially-Resident Images (sparseResidencyImage2D): a block bound with "
        "VkSparseImageMemoryBind (offset and extent in texels, multiples of imageGranularity) holds texels written "
        "to it; with residencyNonResidentStrict, texels of unbound blocks read 0 and writes to them are discarded, "
        "also after a memory dependency in the same command buffer",
        "vkCreateDevice vkDestroyDevice vkGetDeviceQueue vkGetPhysicalDeviceQueueFamilyProperties "
        "vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceImageFormatProperties "
        "vkGetPhysicalDeviceSparseImageFormatProperties vkCreateImage vkDestroyImage vkGetImageMemoryRequirements "
        "vkGetImageSparseMemoryRequirements vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements "
        "vkAllocateMemory vkFreeMemory vkBindBufferMemory vkMapMemory vkQueueBindSparse vkCreateFence "
        "vkDestroyFence vkWaitForFences vkQueueSubmit vkResetFences vkCmdCopyBufferToImage vkCmdCopyImageToBuffer "
        "vkCmdPipelineBarrier vkCreateCommandPool vkDestroyCommandPool vkAllocateCommandBuffers vkBeginCommandBuffer "
        "vkEndCommandBuffer")
{
    struct ct_env e;
    struct ct_features f;
    struct ct_buffer src, dst;
    VkImageCreateInfo info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, 1, 1,
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    VkSparseImageMemoryRequirements r;
    VkMemoryRequirements req;
    VkBufferImageCopy copy;
    VkCommandBuffer cb;
    VkDeviceMemory memory;
    VkImage image;
    uint32_t gw, gh, w, h, x, y, bad = 0;

    ct_env_init(t, &e);
    sp_device(t, &e, &f);
    if (!f.f2.features.sparseResidencyImage2D)
        ct_skip(t, "sparseResidencyImage2D is not supported");
    info.flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
    if (!sp_image_supported(&e, &info))
        ct_skip(t, "no 2D sparse residency for R8G8B8A8_UNORM");
    /* the granularity, from a first image */
    CT_REQUIRE_VK(t, vkCreateImage(e.device, &info, NULL, &image));
    r = sp_color_requirements(t, &e, image);
    vkDestroyImage(e.device, image, NULL);
    gw = r.formatProperties.imageGranularity.width;
    gh = r.formatProperties.imageGranularity.height;
    w = 4 * gw;
    h = 2 * gh;
    info.extent.width = w;
    info.extent.height = h;
    if (!sp_image_supported(&e, &info))
        ct_skip(t, "no %ux%u sparse image", w, h);
    CT_REQUIRE_VK(t, vkCreateImage(e.device, &info, NULL, &image));
    vkGetImageMemoryRequirements(e.device, image, &req);
    r = sp_color_requirements(t, &e, image);
    if (r.imageMipTailFirstLod == 0)
        ct_skip(t, "level 0 is in the mip tail");
    memory = sp_memory(t, &e, req.memoryTypeBits, 2 * req.alignment, 0, NULL);
    ct_host_buffer(t, &e, (VkDeviceSize)w * h * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src);
    ct_host_buffer(t, &e, (VkDeviceSize)w * h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst);
    /* block (2, 1) at the memory's second block */
    sp_bind_image(t, &e, image, 0, 0, sp_offset((int32_t)(2 * gw), (int32_t)gh, 0), sp_extent(gw, gh, 1), memory,
                  req.alignment);
    sp_fill_words(src.data, w * h, 7);
    cb = ct_begin_commands(t, &e);
    ct_image_barrier(cb, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    sp_image_copy(&copy, 0, 0, 0, sp_offset((int32_t)(2 * gw), (int32_t)gh, 0), sp_extent(gw, gh, 1));
    copy.bufferRowLength = w;
    copy.bufferOffset = ((VkDeviceSize)gh * w + 2 * gw) * 4;
    vkCmdCopyBufferToImage(cb, src.buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    ct_transfer_barrier(cb);
    vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_GENERAL, dst.buffer, 1, &copy);
    ct_host_read_barrier(cb);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    ct_submit_and_wait(t, &e, cb);
    for (y = gh; y < 2 * gh && bad < 4; y++)
        for (x = 2 * gw; x < 3 * gw && bad < 4; x++)
            if (memcmp(dst.data + ((size_t)y * w + x) * 4, src.data + ((size_t)y * w + x) * 4, 4) != 0) {
                CT_CHECKF(t, 0, "bound block texel (%u, %u) differs", x, y);
                bad++;
            }
    if (!e.props.sparseProperties.residencyNonResidentStrict) {
        ct_note(t, "residencyNonResidentStrict is not supported: unbound blocks not checked");
    } else {
        /* the whole level: written, then read after a barrier */
        sp_fill_words(src.data, w * h, 8);
        memset(dst.data, 0xEE, (size_t)w * h * 4);
        cb = ct_begin_commands(t, &e);
        sp_image_copy(&copy, 0, 0, 0, sp_offset(0, 0, 0), sp_extent(w, h, 1));
        vkCmdCopyBufferToImage(cb, src.buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
        ct_transfer_barrier(cb);
        vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_GENERAL, dst.buffer, 1, &copy);
        ct_host_read_barrier(cb);
        CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
        ct_submit_and_wait(t, &e, cb);
        for (y = 0; y < h && bad < 4; y++)
            for (x = 0; x < w && bad < 4; x++) {
                const unsigned char *got = dst.data + ((size_t)y * w + x) * 4;
                int bound = x >= 2 * gw && x < 3 * gw && y >= gh;
                if (bound ? memcmp(got, src.data + ((size_t)y * w + x) * 4, 4) != 0 : !ct_all_bytes(got, 4, 0)) {
                    CT_CHECKF(t, 0, "%s texel (%u, %u): %02x%02x%02x%02x", bound ? "bound" : "unbound", x, y, got[0],
                              got[1], got[2], got[3]);
                    bad++;
                }
            }
    }
    vkDestroyImage(e.device, image, NULL);
    vkFreeMemory(e.device, memory, NULL);
    ct_buffer_fini(&e, &src);
    ct_buffer_fini(&e, &dst);
    ct_env_fini(&e);
}

CT_TEST(sparse_image_mip_tail,
        "Sparse Resources > Mip Tail Regions (sparseResidencyImage2D): the mip tail of each array layer is bound with "
        "VkSparseImageOpaqueMemoryBindInfo at imageMipTailOffset + layer * imageMipTailStride (one region with "
        "SINGLE_MIPTAIL); with the levels before it bound with image binds, every level of every layer holds what "
        "is written to it",
        "vkCreateDevice vkDestroyDevice vkGetDeviceQueue vkGetPhysicalDeviceQueueFamilyProperties "
        "vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceImageFormatProperties "
        "vkGetPhysicalDeviceSparseImageFormatProperties vkCreateImage vkDestroyImage vkGetImageMemoryRequirements "
        "vkGetImageSparseMemoryRequirements vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements "
        "vkAllocateMemory vkFreeMemory vkBindBufferMemory vkMapMemory vkQueueBindSparse vkCreateFence "
        "vkDestroyFence vkWaitForFences vkQueueSubmit vkResetFences vkCmdCopyBufferToImage vkCmdCopyImageToBuffer "
        "vkCmdPipelineBarrier vkCreateCommandPool vkDestroyCommandPool vkAllocateCommandBuffers vkBeginCommandBuffer "
        "vkEndCommandBuffer")
{
    struct ct_env e;
    struct ct_features f;
    struct ct_buffer src, dst;
    VkImageCreateInfo info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 512, 256, 1, 10, 2,
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    VkSparseImageMemoryRequirements r;
    VkMemoryRequirements req;
    VkCommandBuffer cb;
    VkDeviceMemory memory;
    VkDeviceSize used = 0, total;
    VkImage image;
    uint32_t layer, lv, tails;

    ct_env_init(t, &e);
    sp_device(t, &e, &f);
    if (!f.f2.features.sparseResidencyImage2D)
        ct_skip(t, "sparseResidencyImage2D is not supported");
    info.flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
    if (!sp_image_supported(&e, &info))
        ct_skip(t, "no 2D array sparse residency image of 10 levels");
    CT_REQUIRE_VK(t, vkCreateImage(e.device, &info, NULL, &image));
    vkGetImageMemoryRequirements(e.device, image, &req);
    r = sp_color_requirements(t, &e, image);
    if (r.imageMipTailFirstLod >= info.mipLevels)
        ct_skip(t, "no mip tail");
    total = req.size; /* enough for every block and tail */
    memory = sp_memory(t, &e, req.memoryTypeBits, total, 0, NULL);
    ct_host_buffer(t, &e, 512 * 256 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src);
    ct_host_buffer(t, &e, 512 * 256 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst);
    tails = (r.formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) ? 1 : info.arrayLayers;
    for (layer = 0; layer < tails; layer++) {
        sp_bind_opaque(t, &e, image, r.imageMipTailOffset + layer * r.imageMipTailStride, r.imageMipTailSize, memory,
                       used);
        used += r.imageMipTailSize;
    }
    for (layer = 0; layer < info.arrayLayers; layer++)
        for (lv = 0; lv < r.imageMipTailFirstLod; lv++) {
            uint32_t w = 512 >> lv, h = 256 >> lv, gw = r.formatProperties.imageGranularity.width;
            uint32_t gh = r.formatProperties.imageGranularity.height;
            VkDeviceSize blocks = (VkDeviceSize)((w + gw - 1) / gw) * ((h + gh - 1) / gh);
            CT_REQUIREF(t, used + blocks * req.alignment <= total, "memory requirements too small for every block");
            sp_bind_image(t, &e, image, lv, layer, sp_offset(0, 0, 0), sp_extent(w, h, 1), memory, used);
            used += blocks * req.alignment;
        }
    cb = ct_begin_commands(t, &e);
    ct_image_barrier(cb, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    ct_submit_and_wait(t, &e, cb);
    for (layer = 0; layer < info.arrayLayers; layer++)
        for (lv = 0; lv < info.mipLevels; lv++) {
            uint32_t w = 512 >> lv, h = 256 >> lv, seed = 40 + layer * 16 + lv;
            VkBufferImageCopy copy;
            w = w ? w : 1;
            h = h ? h : 1;
            sp_fill_words(src.data, w * h, seed);
            memset(dst.data, 0, (size_t)w * h * 4);
            cb = ct_begin_commands(t, &e);
            sp_image_copy(&copy, 0, lv, layer, sp_offset(0, 0, 0), sp_extent(w, h, 1));
            vkCmdCopyBufferToImage(cb, src.buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
            ct_transfer_barrier(cb);
            vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_GENERAL, dst.buffer, 1, &copy);
            ct_host_read_barrier(cb);
            CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
            ct_submit_and_wait(t, &e, cb);
            CT_CHECKF(t, sp_words_are(dst.data, w * h, seed), "layer %u level %u (%s) differs", layer, lv,
                      lv >= r.imageMipTailFirstLod ? "mip tail" : "bound by image binds");
        }
    vkDestroyImage(e.device, image, NULL);
    vkFreeMemory(e.device, memory, NULL);
    ct_buffer_fini(&e, &src);
    ct_buffer_fini(&e, &dst);
    ct_env_fini(&e);
}

CT_TEST(sparse_image_opaque,
        "Sparse Resources > Fully Resident Images (sparseBinding): an image created with only "
        "VK_IMAGE_CREATE_SPARSE_BINDING_BIT has no sparse image memory requirements; bound completely with "
        "opaque binds (here in two pieces of two memory objects), every subresource holds what is written to it",
        "vkCreateDevice vkDestroyDevice vkGetDeviceQueue vkGetPhysicalDeviceQueueFamilyProperties "
        "vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceImageFormatProperties vkCreateImage vkDestroyImage "
        "vkGetImageMemoryRequirements vkGetImageSparseMemoryRequirements vkCreateBuffer vkDestroyBuffer "
        "vkGetBufferMemoryRequirements vkAllocateMemory vkFreeMemory vkBindBufferMemory vkMapMemory "
        "vkQueueBindSparse vkCreateFence vkDestroyFence vkWaitForFences vkQueueSubmit vkResetFences "
        "vkCmdCopyBufferToImage vkCmdCopyImageToBuffer vkCmdPipelineBarrier vkCreateCommandPool "
        "vkDestroyCommandPool vkAllocateCommandBuffers vkBeginCommandBuffer vkEndCommandBuffer")
{
    struct ct_env e;
    struct ct_features f;
    struct ct_buffer src, dst;
    VkImageCreateInfo info = ct_image_info(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 256, 128, 1, 4, 3,
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    VkMemoryRequirements req;
    VkCommandBuffer cb;
    VkDeviceMemory a, b;
    VkDeviceSize half;
    VkImage image;
    uint32_t layer, lv, count = 7;

    ct_env_init(t, &e);
    sp_device(t, &e, &f);
    info.flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT;
    if (!sp_image_supported(&e, &info))
        ct_skip(t, "no sparse binding for this 2D array image");
    CT_REQUIRE_VK(t, vkCreateImage(e.device, &info, NULL, &image));
    vkGetImageMemoryRequirements(e.device, image, &req);
    vkGetImageSparseMemoryRequirements(e.device, image, &count, NULL);
    CT_CHECKF(t, count == 0, "%u sparse requirements without sparse residency", count);
    CT_CHECKF(t, ct_is_pow2(req.alignment) && req.size % req.alignment == 0, "size %llu, alignment %llu",
              (unsigned long long)req.size, (unsigned long long)req.alignment);
    half = req.size / req.alignment / 2 * req.alignment;
    if (half == 0)
        half = req.size;
    a = sp_memory(t, &e, req.memoryTypeBits, half, 0, NULL);
    b = half < req.size ? sp_memory(t, &e, req.memoryTypeBits, req.size - half, 1, NULL) : VK_NULL_HANDLE;
    sp_bind_opaque(t, &e, image, 0, half, a, 0);
    if (b)
        sp_bind_opaque(t, &e, image, half, req.size - half, b, 0);
    ct_host_buffer(t, &e, 256 * 128 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src);
    ct_host_buffer(t, &e, 256 * 128 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst);
    cb = ct_begin_commands(t, &e);
    ct_image_barrier(cb, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    ct_submit_and_wait(t, &e, cb);
    for (layer = 0; layer < 3; layer++)
        for (lv = 0; lv < 4; lv++) {
            uint32_t w = 256 >> lv, h = 128 >> lv, seed = 60 + layer * 4 + lv;
            VkBufferImageCopy copy;
            sp_fill_words(src.data, w * h, seed);
            memset(dst.data, 0, (size_t)w * h * 4);
            cb = ct_begin_commands(t, &e);
            sp_image_copy(&copy, 0, lv, layer, sp_offset(0, 0, 0), sp_extent(w, h, 1));
            vkCmdCopyBufferToImage(cb, src.buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
            ct_transfer_barrier(cb);
            vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_GENERAL, dst.buffer, 1, &copy);
            ct_host_read_barrier(cb);
            CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
            ct_submit_and_wait(t, &e, cb);
            CT_CHECKF(t, sp_words_are(dst.data, w * h, seed), "layer %u level %u differs", layer, lv);
        }
    vkDestroyImage(e.device, image, NULL);
    vkFreeMemory(e.device, a, NULL);
    if (b)
        vkFreeMemory(e.device, b, NULL);
    ct_buffer_fini(&e, &src);
    ct_buffer_fini(&e, &dst);
    ct_env_fini(&e);
}

CT_TEST(sparse_image_3d,
        "Sparse Resources > Sparse Partially-Resident Images (sparseResidencyImage3D): a block of a 3D image bound "
        "with VkSparseImageMemoryBind (offset and extent multiples of the 3D granularity) holds texels written to it",
        "vkCreateDevice vkDestroyDevice vkGetDeviceQueue vkGetPhysicalDeviceQueueFamilyProperties "
        "vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceImageFormatProperties "
        "vkGetPhysicalDeviceSparseImageFormatProperties vkCreateImage vkDestroyImage vkGetImageMemoryRequirements "
        "vkGetImageSparseMemoryRequirements vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements "
        "vkAllocateMemory vkFreeMemory vkBindBufferMemory vkMapMemory vkQueueBindSparse vkCreateFence "
        "vkDestroyFence vkWaitForFences vkQueueSubmit vkResetFences vkCmdCopyBufferToImage vkCmdCopyImageToBuffer "
        "vkCmdPipelineBarrier vkCreateCommandPool vkDestroyCommandPool vkAllocateCommandBuffers vkBeginCommandBuffer "
        "vkEndCommandBuffer")
{
    struct ct_env e;
    struct ct_features f;
    struct ct_buffer src, dst;
    VkImageCreateInfo info = ct_image_info(VK_IMAGE_TYPE_3D, VK_FORMAT_R32_UINT, 1, 1, 1, 1, 1,
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    VkSparseImageMemoryRequirements r;
    VkMemoryRequirements req;
    VkBufferImageCopy copy;
    VkCommandBuffer cb;
    VkDeviceMemory memory;
    VkImage image;
    uint32_t gw, gh, gd, n;

    ct_env_init(t, &e);
    sp_device(t, &e, &f);
    if (!f.f2.features.sparseResidencyImage3D)
        ct_skip(t, "sparseResidencyImage3D is not supported");
    info.flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
    if (!sp_image_supported(&e, &info))
        ct_skip(t, "no 3D sparse residency for R32_UINT");
    CT_REQUIRE_VK(t, vkCreateImage(e.device, &info, NULL, &image));
    r = sp_color_requirements(t, &e, image);
    vkDestroyImage(e.device, image, NULL);
    gw = r.formatProperties.imageGranularity.width;
    gh = r.formatProperties.imageGranularity.height;
    gd = r.formatProperties.imageGranularity.depth;
    info.extent = sp_extent(2 * gw, 2 * gh, 2 * gd);
    if (!sp_image_supported(&e, &info))
        ct_skip(t, "no %ux%ux%u sparse image", 2 * gw, 2 * gh, 2 * gd);
    CT_REQUIRE_VK(t, vkCreateImage(e.device, &info, NULL, &image));
    vkGetImageMemoryRequirements(e.device, image, &req);
    r = sp_color_requirements(t, &e, image);
    if (r.imageMipTailFirstLod == 0)
        ct_skip(t, "level 0 is in the mip tail");
    memory = sp_memory(t, &e, req.memoryTypeBits, req.alignment, 0, NULL);
    n = gw * gh * gd;
    ct_host_buffer(t, &e, (VkDeviceSize)n * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src);
    ct_host_buffer(t, &e, (VkDeviceSize)n * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst);
    /* block (1, 0, 1) */
    sp_bind_image(t, &e, image, 0, 0, sp_offset((int32_t)gw, 0, (int32_t)gd), sp_extent(gw, gh, gd), memory, 0);
    sp_fill_words(src.data, n, 9);
    cb = ct_begin_commands(t, &e);
    ct_image_barrier(cb, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    sp_image_copy(&copy, 0, 0, 0, sp_offset((int32_t)gw, 0, (int32_t)gd), sp_extent(gw, gh, gd));
    vkCmdCopyBufferToImage(cb, src.buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    ct_transfer_barrier(cb);
    vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_GENERAL, dst.buffer, 1, &copy);
    ct_host_read_barrier(cb);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cb));
    ct_submit_and_wait(t, &e, cb);
    CT_CHECK(t, sp_words_are(dst.data, n, 9));
    vkDestroyImage(e.device, image, NULL);
    vkFreeMemory(e.device, memory, NULL);
    ct_buffer_fini(&e, &src);
    ct_buffer_fini(&e, &dst);
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(sparse_requirements_non_sparse),
    CT_REF(sparse_requirements_residency),
    CT_REF(queue_bind_sparse),
    CT_REF(sparse_buffer_requirements),
    CT_REF(sparse_buffer_residency),
    CT_REF(sparse_buffer_rebind),
    CT_REF(sparse_buffer_aliased),
    CT_REF(bind_sparse_batches_order),
    CT_REF(sparse_image_format_queries),
    CT_REF(sparse_image_requirements_rules),
    CT_REF(sparse_image_blocks),
    CT_REF(sparse_image_mip_tail),
    CT_REF(sparse_image_opaque),
    CT_REF(sparse_image_3d),
};

CT_SUITE(sparsemem, tests)
