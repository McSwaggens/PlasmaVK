/* Chapter "Debugging": section "Active Tooling Information" (the tools
 * active on a physical device: none on a driver used directly; a layer such
 * as the validation layer reports itself) and section "Debug Utilities"
 * (VK_EXT_debug_utils: messengers, object names and tags, queue and command
 * buffer labels, and how its commands are obtained). */
#include "ct.h"

#include <stdio.h>
#include <stdlib.h>

#define DG_GUARD 0xA5

static int dg_same_tool(const VkPhysicalDeviceToolProperties *x, const VkPhysicalDeviceToolProperties *y)
{
    return memcmp(x->name, y->name, sizeof(x->name)) == 0 && memcmp(x->version, y->version, sizeof(x->version)) == 0 &&
           x->purposes == y->purposes && memcmp(x->description, y->description, sizeof(x->description)) == 0 &&
           memcmp(x->layer, y->layer, sizeof(x->layer)) == 0;
}

static int dg_terminated(const char *s, size_t n)
{
    return memchr(s, '\0', n) != NULL;
}

/* n structures with sType and pNext set, the rest the guard pattern. */
static void dg_prepare(VkPhysicalDeviceToolProperties *p, uint32_t n)
{
    uint32_t i;
    for (i = 0; i < n; i++) {
        memset(&p[i], DG_GUARD, sizeof(p[i]));
        p[i].sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES;
        p[i].pNext = NULL;
    }
}

static int dg_untouched(const VkPhysicalDeviceToolProperties *p)
{
    return p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES && p->pNext == NULL &&
           ct_all_bytes(p->name, sizeof(p->name), DG_GUARD) && ct_all_bytes(p->layer, sizeof(p->layer), DG_GUARD) &&
           ct_all_bytes(&p->purposes, sizeof(p->purposes), DG_GUARD);
}

CT_TEST(tool_properties,
        "Debugging > Active Tooling Information (vkGetPhysicalDeviceToolProperties, Vulkan 1.3): the count of "
        "active tools, then with an array the count overwritten with the number written, at most pToolCount "
        "(VK_INCOMPLETE with fewer, VK_SUCCESS with room to spare); name, version, description and layer are "
        "null-terminated UTF-8 strings within their arrays; the application's sType and pNext are kept",
        "vkGetPhysicalDeviceToolProperties")
{
    struct ct_env e;
    VkPhysicalDeviceToolProperties *full, *part;
    uint32_t count = 0, n, i, j, k;

    ct_env_init(t, &e);
    if (e.version < VK_API_VERSION_1_3)
        ct_skip(t, "vkGetPhysicalDeviceToolProperties is Vulkan 1.3");
    CT_REQUIRE_VK(t, vkGetPhysicalDeviceToolProperties(e.pd, &count, NULL));
    full = calloc(count + 1, sizeof(*full));
    part = calloc(count + 1, sizeof(*part));
    CT_REQUIRE(t, full && part);
    dg_prepare(full, count + 1);
    n = count + 1;
    CT_CHECK_VK(t, vkGetPhysicalDeviceToolProperties(e.pd, &n, full), VK_SUCCESS);
    CT_CHECKF(t, n == count, "%u tools counted, then %u written with room for %u", count, n, count + 1);
    CT_CHECKF(t, dg_untouched(&full[count]), "wrote past the last tool");
    for (i = 0; i < n && i < count; i++) {
        const VkPhysicalDeviceToolProperties *p = &full[i];
        CT_CHECKF(t, p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES && p->pNext == NULL,
                  "tool %u: sType or pNext changed", i);
        CT_CHECKF(t, dg_terminated(p->name, sizeof(p->name)) && dg_terminated(p->version, sizeof(p->version)) &&
                         dg_terminated(p->description, sizeof(p->description)) &&
                         dg_terminated(p->layer, sizeof(p->layer)),
                  "tool %u: a string is not null-terminated", i);
        if (dg_terminated(p->name, sizeof(p->name)) && dg_terminated(p->layer, sizeof(p->layer)))
            ct_note(t, "tool %u: \"%s\" (layer \"%s\"), purposes 0x%x", i, p->name, p->layer, p->purposes);
    }
    for (k = 0; k < count; k++) {
        dg_prepare(part, count + 1);
        n = k;
        CT_CHECK_VK(t, vkGetPhysicalDeviceToolProperties(e.pd, &n, part), VK_INCOMPLETE);
        CT_CHECKF(t, n == k, "room for %u of %u tools: count returned %u", k, count, n);
        CT_CHECKF(t, dg_untouched(&part[k]), "room for %u tools: wrote tool %u", k, k);
        for (i = 0; i < n && i < k; i++) {
            int found = 0;
            for (j = 0; j < count; j++)
                found |= dg_same_tool(&part[i], &full[j]);
            CT_CHECKF(t, found, "room for %u tools: tool %u is not one of the full list", k, i);
        }
    }
    free(full);
    free(part);
    ct_env_fini(&e);
}

/* ---- Debug Utilities (VK_EXT_debug_utils) ----
 *
 * Extension commands are resolved with vkGetInstanceProcAddr (a loader need
 * not export them). Messengers count only the tests' own messages (message
 * ID name DU_ID): a loader or layer may report its own as well. */

#define DU_EXT "VK_EXT_debug_utils"
#define DU_ID "CT-debug-utils"

#define DU_SEVERITIES                                                                                            \
    (VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |              \
     VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
#define DU_TYPES                                                                                                 \
    (VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |                \
     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)

#define DU_COMMANDS "vkCreateDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT vkSubmitDebugUtilsMessageEXT "

struct du_fns {
    PFN_vkCreateDebugUtilsMessengerEXT create;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy;
    PFN_vkSubmitDebugUtilsMessageEXT submit;
    PFN_vkSetDebugUtilsObjectNameEXT name;
    PFN_vkSetDebugUtilsObjectTagEXT tag;
    PFN_vkQueueBeginDebugUtilsLabelEXT queue_begin;
    PFN_vkQueueEndDebugUtilsLabelEXT queue_end;
    PFN_vkQueueInsertDebugUtilsLabelEXT queue_insert;
    PFN_vkCmdBeginDebugUtilsLabelEXT cmd_begin;
    PFN_vkCmdEndDebugUtilsLabelEXT cmd_end;
    PFN_vkCmdInsertDebugUtilsLabelEXT cmd_insert;
};

#define DU_GET(f, field, name)                                                                                   \
    do {                                                                                                         \
        (f)->field = (PFN_##name)vkGetInstanceProcAddr(instance, #name);                                         \
        CT_REQUIREF(t, (f)->field != NULL, "vkGetInstanceProcAddr(instance, \"%s\") is NULL with " DU_EXT        \
                    " enabled", #name);                                                                          \
    } while (0)

static void du_load(struct ct *t, VkInstance instance, struct du_fns *f)
{
    DU_GET(f, create, vkCreateDebugUtilsMessengerEXT);
    DU_GET(f, destroy, vkDestroyDebugUtilsMessengerEXT);
    DU_GET(f, submit, vkSubmitDebugUtilsMessageEXT);
    DU_GET(f, name, vkSetDebugUtilsObjectNameEXT);
    DU_GET(f, tag, vkSetDebugUtilsObjectTagEXT);
    DU_GET(f, queue_begin, vkQueueBeginDebugUtilsLabelEXT);
    DU_GET(f, queue_end, vkQueueEndDebugUtilsLabelEXT);
    DU_GET(f, queue_insert, vkQueueInsertDebugUtilsLabelEXT);
    DU_GET(f, cmd_begin, vkCmdBeginDebugUtilsLabelEXT);
    DU_GET(f, cmd_end, vkCmdEndDebugUtilsLabelEXT);
    DU_GET(f, cmd_insert, vkCmdInsertDebugUtilsLabelEXT);
}

/* What a callback was given: the last of the tests' own messages. */
struct du_message {
    VkDebugUtilsMessageSeverityFlagBitsEXT severity;
    VkDebugUtilsMessageTypeFlagsEXT types;
    int32_t number;
    char text[128];
    uint32_t object_count, queue_label_count, cb_label_count;
    VkObjectType object_types[2];
    uint64_t objects[2];
    char object_names[2][64]; /* "(null)" for none */
    char queue_labels[2][64], cb_labels[2][64];
    float queue_color[4];
};

/* The state of a messenger's callback (its pUserData). */
struct du_record {
    VkDebugUtilsMessageSeverityFlagsEXT severities; /* the messenger's filter */
    VkDebugUtilsMessageTypeFlagsEXT types;
    int ours, others;   /* the tests' messages, and any other */
    int outside_filter; /* calls with a severity or types the filter excludes */
    int malformed;      /* not one severity bit, or callback data that is not valid */
    struct du_message last;
};

static void du_copy(char *dst, size_t size, const char *src)
{
    snprintf(dst, size, "%s", src ? src : "(null)");
}

static VkBool32 VKAPI_PTR du_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                      VkDebugUtilsMessageTypeFlagsEXT types,
                                      const VkDebugUtilsMessengerCallbackDataEXT *data, void *user)
{
    struct du_record *r = user;
    struct du_message *m = &r->last;
    uint32_t i;

    if (!(severity & r->severities) || !(types & r->types))
        r->outside_filter++;
    if (!severity || (severity & (severity - 1)) || !types || !data ||
        data->sType != VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT ||
        (data->objectCount && !data->pObjects) || (data->queueLabelCount && !data->pQueueLabels) ||
        (data->cmdBufLabelCount && !data->pCmdBufLabels)) {
        r->malformed++;
        return VK_FALSE;
    }
    if (!data->pMessageIdName || strcmp(data->pMessageIdName, DU_ID) != 0) {
        r->others++;
        return VK_FALSE;
    }
    r->ours++;
    memset(m, 0, sizeof(*m));
    m->severity = severity;
    m->types = types;
    m->number = data->messageIdNumber;
    du_copy(m->text, sizeof(m->text), data->pMessage);
    m->object_count = data->objectCount;
    for (i = 0; i < data->objectCount && i < 2; i++) {
        m->object_types[i] = data->pObjects[i].objectType;
        m->objects[i] = data->pObjects[i].objectHandle;
        du_copy(m->object_names[i], sizeof(m->object_names[i]), data->pObjects[i].pObjectName);
    }
    m->queue_label_count = data->queueLabelCount;
    for (i = 0; i < data->queueLabelCount && i < 2; i++)
        du_copy(m->queue_labels[i], sizeof(m->queue_labels[i]), data->pQueueLabels[i].pLabelName);
    if (data->queueLabelCount)
        memcpy(m->queue_color, data->pQueueLabels[0].color, sizeof(m->queue_color));
    m->cb_label_count = data->cmdBufLabelCount;
    for (i = 0; i < data->cmdBufLabelCount && i < 2; i++)
        du_copy(m->cb_labels[i], sizeof(m->cb_labels[i]), data->pCmdBufLabels[i].pLabelName);
    return VK_FALSE;
}

static VkDebugUtilsMessengerCreateInfoEXT du_info(struct du_record *r, VkDebugUtilsMessageSeverityFlagsEXT severities,
                                                  VkDebugUtilsMessageTypeFlagsEXT types)
{
    VkDebugUtilsMessengerCreateInfoEXT info;
    memset(r, 0, sizeof(*r));
    r->severities = severities;
    r->types = types;
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity = severities;
    info.messageType = types;
    info.pfnUserCallback = du_callback;
    info.pUserData = r;
    return info;
}

static VkDebugUtilsMessengerEXT du_messenger(struct ct *t, const struct du_fns *f, VkInstance instance,
                                             struct du_record *r, VkDebugUtilsMessageSeverityFlagsEXT severities,
                                             VkDebugUtilsMessageTypeFlagsEXT types, const VkAllocationCallbacks *alloc)
{
    VkDebugUtilsMessengerCreateInfoEXT info = du_info(r, severities, types);
    VkDebugUtilsMessengerEXT m = VK_NULL_HANDLE;
    CT_REQUIRE_VK(t, f->create(instance, &info, alloc, &m));
    CT_REQUIRE(t, m != VK_NULL_HANDLE);
    return m;
}

static void du_submit(const struct du_fns *f, VkInstance instance, VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                      VkDebugUtilsMessageTypeFlagsEXT types, const char *text)
{
    VkDebugUtilsMessengerCallbackDataEXT data;
    memset(&data, 0, sizeof(data));
    data.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT;
    data.pMessageIdName = DU_ID;
    data.messageIdNumber = 1234;
    data.pMessage = text;
    f->submit(instance, severity, types, &data);
}

static void du_require_extension(struct ct *t)
{
    if (!ct_instance_extension_supported(DU_EXT))
        ct_skip(t, DU_EXT " is not offered");
}

/* An instance with VK_EXT_debug_utils enabled, and its commands. */
static void du_env_init(struct ct *t, struct ct_env *e, struct du_fns *f)
{
    const char *ext = DU_EXT;
    du_require_extension(t);
    ct_env_init_ex(t, e, 0, &ext, 1);
    du_load(t, e->instance, f);
}

static VkDebugUtilsLabelEXT du_label(const char *name, float r, float g, float b, float a)
{
    VkDebugUtilsLabelEXT l;
    memset(&l, 0, sizeof(l));
    l.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    l.pLabelName = name;
    l.color[0] = r;
    l.color[1] = g;
    l.color[2] = b;
    l.color[3] = a;
    return l;
}

static void du_check_record(struct ct *t, const struct du_record *r, const char *what)
{
    CT_CHECKF(t, r->outside_filter == 0, "%s: %d callbacks with a severity or types its filter excludes", what,
              r->outside_filter);
    CT_CHECKF(t, r->malformed == 0, "%s: %d callbacks with invalid arguments", what, r->malformed);
}

CT_TEST(debug_utils_messenger_filter,
        "Debugging > Debug Utilities: a messenger's callback is called for a message only if the message's severity "
        "is in its messageSeverity mask and one of the message's types in its messageType mask "
        "(vkSubmitDebugUtilsMessageEXT \"will propagate through the layers and generate callback(s) as indicated by "
        "the message's flags\"): once per such messenger, with the message's severity and types, and not otherwise",
        DU_COMMANDS)
{
    static const VkDebugUtilsMessageSeverityFlagBitsEXT severities[] = {
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT,
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
    };
    static const VkDebugUtilsMessageTypeFlagsEXT types[] = {
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        DU_TYPES,
    };
    static const struct {
        VkDebugUtilsMessageSeverityFlagsEXT severities;
        VkDebugUtilsMessageTypeFlagsEXT types;
    } filters[] = {
        { VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
          VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT },
        { DU_SEVERITIES, VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT },
        { VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT, DU_TYPES },
        { VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT },
    };
    struct du_record rec[4];
    VkDebugUtilsMessengerEXT m[4];
    struct ct_env e;
    struct du_fns f;
    size_t i, j, k;

    du_env_init(t, &e, &f);
    for (k = 0; k < 4; k++)
        m[k] = du_messenger(t, &f, e.instance, &rec[k], filters[k].severities, filters[k].types, NULL);
    for (i = 0; i < CT_ARRAY_SIZE(severities); i++) {
        for (j = 0; j < CT_ARRAY_SIZE(types); j++) {
            int before[4];
            for (k = 0; k < 4; k++)
                before[k] = rec[k].ours;
            du_submit(&f, e.instance, severities[i], types[j], "filtered");
            for (k = 0; k < 4; k++) {
                int want = (filters[k].severities & severities[i]) && (filters[k].types & types[j]);
                CT_CHECKF(t, rec[k].ours - before[k] == want,
                          "severity 0x%x, types 0x%x: messenger %zu (severities 0x%x, types 0x%x) called %d times, "
                          "expected %d", severities[i], types[j], k, filters[k].severities, filters[k].types,
                          rec[k].ours - before[k], want);
                if (want && rec[k].ours - before[k] == 1)
                    CT_CHECKF(t, rec[k].last.severity == severities[i] && rec[k].last.types == types[j],
                              "messenger %zu was given severity 0x%x and types 0x%x for 0x%x and 0x%x", k,
                              rec[k].last.severity, rec[k].last.types, severities[i], types[j]);
            }
        }
    }
    for (k = 0; k < 4; k++) {
        char what[32];
        snprintf(what, sizeof(what), "messenger %zu", k);
        du_check_record(t, &rec[k], what);
        f.destroy(e.instance, m[k], NULL);
    }
    ct_env_fini(&e);
}

CT_TEST(debug_utils_message_data,
        "Debugging > Debug Utilities: vkSubmitDebugUtilsMessageEXT passes its callback data on to the callback "
        "(\"The parameters are passed on to the callback in addition to the pUserData value that was defined at "
        "the time the messenger was registered\"): message ID name and number, message, objects with their names, "
        "queue and command buffer labels",
        DU_COMMANDS "vkEnumeratePhysicalDevices")
{
    VkDebugUtilsObjectNameInfoEXT objects[2];
    VkDebugUtilsLabelEXT queue_labels[2], cb_label;
    VkDebugUtilsMessengerCallbackDataEXT data;
    VkDebugUtilsMessengerEXT m;
    struct du_record rec;
    struct ct_env e;
    struct du_fns f;
    const struct du_message *got = &rec.last;

    du_env_init(t, &e, &f);
    m = du_messenger(t, &f, e.instance, &rec, DU_SEVERITIES, DU_TYPES, NULL);
    memset(objects, 0, sizeof(objects));
    objects[0].sType = objects[1].sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    objects[0].objectType = VK_OBJECT_TYPE_INSTANCE;
    objects[0].objectHandle = (uint64_t)(uintptr_t)e.instance;
    objects[0].pObjectName = "the instance";
    objects[1].objectType = VK_OBJECT_TYPE_PHYSICAL_DEVICE;
    objects[1].objectHandle = (uint64_t)(uintptr_t)e.pd;
    queue_labels[0] = du_label("frame", 0.25f, 0.5f, 0.75f, 1.0f);
    queue_labels[1] = du_label("pass", 0.0f, 0.0f, 0.0f, 0.0f);
    cb_label = du_label("draw", 1.0f, 0.0f, 0.0f, 1.0f);
    memset(&data, 0, sizeof(data));
    data.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT;
    data.pMessageIdName = DU_ID;
    data.messageIdNumber = -77;
    data.pMessage = "the application's message";
    data.queueLabelCount = 2;
    data.pQueueLabels = queue_labels;
    data.cmdBufLabelCount = 1;
    data.pCmdBufLabels = &cb_label;
    data.objectCount = 2;
    data.pObjects = objects;
    f.submit(e.instance, VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
             &data);
    CT_REQUIREF(t, rec.ours == 1, "the callback was called %d times for one message", rec.ours);
    CT_CHECK(t, got->severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT);
    CT_CHECK(t, got->types == VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT);
    CT_CHECKF(t, got->number == -77, "messageIdNumber %d", got->number);
    CT_CHECKF(t, strcmp(got->text, "the application's message") == 0, "pMessage \"%s\"", got->text);
    CT_REQUIREF(t, got->object_count == 2, "objectCount %u", got->object_count);
    CT_CHECK(t, got->object_types[0] == VK_OBJECT_TYPE_INSTANCE && got->objects[0] == objects[0].objectHandle);
    CT_CHECKF(t, strcmp(got->object_names[0], "the instance") == 0, "object 0 name \"%s\"", got->object_names[0]);
    CT_CHECK(t, got->object_types[1] == VK_OBJECT_TYPE_PHYSICAL_DEVICE && got->objects[1] == objects[1].objectHandle);
    CT_REQUIREF(t, got->queue_label_count == 2 && got->cb_label_count == 1, "%u queue labels, %u command buffer labels",
                got->queue_label_count, got->cb_label_count);
    CT_CHECK(t, strcmp(got->queue_labels[0], "frame") == 0 && strcmp(got->queue_labels[1], "pass") == 0);
    CT_CHECK(t, got->queue_color[0] == 0.25f && got->queue_color[1] == 0.5f && got->queue_color[2] == 0.75f &&
                    got->queue_color[3] == 1.0f);
    CT_CHECK(t, strcmp(got->cb_labels[0], "draw") == 0);
    du_check_record(t, &rec, "messenger");
    f.destroy(e.instance, m, NULL);
    ct_env_fini(&e);
}

CT_TEST(debug_utils_messenger_destroy,
        "Debugging > Debug Utilities: vkDestroyDebugUtilsMessengerEXT destroys a messenger, whose callback is not "
        "called again while the instance's other messengers still are, and does nothing with VK_NULL_HANDLE (\"If "
        "messenger is not VK_NULL_HANDLE ...\")",
        DU_COMMANDS)
{
    struct du_record r1, r2;
    VkDebugUtilsMessengerEXT m1, m2;
    struct ct_env e;
    struct du_fns f;

    du_env_init(t, &e, &f);
    m1 = du_messenger(t, &f, e.instance, &r1, DU_SEVERITIES, DU_TYPES, NULL);
    m2 = du_messenger(t, &f, e.instance, &r2, DU_SEVERITIES, DU_TYPES, NULL);
    du_submit(&f, e.instance, VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
              "both");
    CT_CHECK(t, r1.ours == 1 && r2.ours == 1);
    f.destroy(e.instance, m1, NULL);
    du_submit(&f, e.instance, VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
              "one");
    CT_CHECKF(t, r1.ours == 1, "a destroyed messenger was called (%d times in all)", r1.ours);
    CT_CHECK(t, r2.ours == 2);
    f.destroy(e.instance, m2, NULL);
    du_submit(&f, e.instance, VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
              "none");
    CT_CHECK(t, r2.ours == 2);
    f.destroy(e.instance, VK_NULL_HANDLE, NULL);
    du_check_record(t, &r1, "messenger 1");
    du_check_record(t, &r2, "messenger 2");
    ct_env_fini(&e);
}

CT_TEST(debug_utils_messenger_allocator,
        "Memory Allocation > Host Memory: the host memory of a messenger made with allocation callbacks comes from "
        "them (power-of-two alignments, valid scopes, on the calling thread, during commands) and is all freed by "
        "vkDestroyDebugUtilsMessengerEXT given the same callbacks; frees only name memory they returned",
        DU_COMMANDS)
{
    struct du_record rec;
    VkDebugUtilsMessengerEXT m;
    struct ct_allocator a;
    struct ct_env e;
    struct du_fns f;

    du_env_init(t, &e, &f);
    ct_allocator_init(&a, -1);
    CT_CALL(&a, m = du_messenger(t, &f, e.instance, &rec, DU_SEVERITIES, DU_TYPES, &a.cb));
    du_submit(&f, e.instance, VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
              "with an allocator");
    CT_CHECK(t, rec.ours == 1);
    ct_note(t, "vkCreateDebugUtilsMessengerEXT made %ld allocations", a.total);
    CT_CALL(&a, f.destroy(e.instance, m, &a.cb));
    CT_CHECKF(t, a.live == 0, "%ld allocations still live after vkDestroyDebugUtilsMessengerEXT", a.live);
    ct_allocator_check(t, &a, "messenger");
    ct_allocator_fini(&a);
    ct_env_fini(&e);
}

/* vkCreateInstance with VK_EXT_debug_utils (and portability enumeration when
 * offered) and pNext chained to the create info. */
static VkResult du_create_instance(const void *pnext, VkInstance *instance)
{
    const char *names[2] = { DU_EXT, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME };
    int portability = ct_instance_extension_supported(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    VkApplicationInfo app;
    VkInstanceCreateInfo info;
    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = CT_MIN(VK_API_VERSION_1_4, CT_VERSION(ct_instance_version()));
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pNext = pnext;
    info.flags = portability ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0;
    info.pApplicationInfo = &app;
    info.enabledExtensionCount = portability ? 2 : 1;
    info.ppEnabledExtensionNames = names;
    return vkCreateInstance(&info, NULL, instance);
}

CT_TEST(debug_utils_instance_create_info,
        "Debugging > Debug Utilities: VkDebugUtilsMessengerCreateInfoEXT structures chained to VkInstanceCreateInfo "
        "report what happens in vkCreateInstance and vkDestroyInstance, with their filters applied; such a callback "
        "\"is only valid for the duration of the vkCreateInstance and the vkDestroyInstance call\": messages in "
        "between reach the instance's messengers, not it",
        "vkCreateInstance vkDestroyInstance " DU_COMMANDS)
{
    VkDebugUtilsMessengerCreateInfoEXT chained[2];
    struct du_record filtered, all, persistent;
    VkDebugUtilsMessengerEXT m;
    VkInstance instance = VK_NULL_HANDLE;
    struct du_fns f;

    du_require_extension(t);
    chained[0] = du_info(&filtered, VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                         VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT);
    chained[1] = du_info(&all, DU_SEVERITIES, DU_TYPES);
    chained[0].pNext = &chained[1];
    CT_REQUIRE_VK(t, du_create_instance(&chained[0], &instance));
    ct_note(t, "during vkCreateInstance: %d messages to the filtered messenger, %d to the other",
            filtered.others + filtered.ours, all.others + all.ours);
    du_load(t, instance, &f);
    m = du_messenger(t, &f, instance, &persistent, DU_SEVERITIES, DU_TYPES, NULL);
    du_submit(&f, instance, VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
              "after vkCreateInstance");
    CT_CHECKF(t, persistent.ours == 1, "the instance's messenger was called %d times", persistent.ours);
    CT_CHECKF(t, filtered.ours == 0 && all.ours == 0,
              "messengers chained to VkInstanceCreateInfo were called after vkCreateInstance returned (%d, %d times)",
              filtered.ours, all.ours);
    f.destroy(instance, m, NULL);
    vkDestroyInstance(instance, NULL);
    du_check_record(t, &filtered, "chained messenger (warnings and errors)");
    du_check_record(t, &all, "chained messenger (everything)");
    du_check_record(t, &persistent, "messenger");
}

/* An instance with VK_EXT_debug_utils and a device as ct_env_render makes
 * (Vulkan 1.3, every core feature the device has), with the rectangle shaders,
 * a pipeline layout and a host buffer, as ct_scene_init. */
static void du_scene_init(struct ct *t, struct ct_scene *s, struct du_fns *f)
{
    memset(s, 0, sizeof(*s));
    du_env_init(t, &s->e, f);
    if (s->e.version < VK_API_VERSION_1_3)
        ct_skip(t, "needs Vulkan 1.3 (dynamic rendering)");
    if (!s->e.graphics)
        ct_skip(t, "no queue family supports graphics");
    ct_get_features(&s->e, &s->f, 0);
    ct_env_device(t, &s->e, &s->f.f2);
    s->vs = ct_shader_module(t, &s->e, ct_rect_vert, ct_rect_vert_size);
    s->fs = ct_shader_module(t, &s->e, ct_color_frag, ct_color_frag_size);
    s->layout = ct_pipeline_layout(t, &s->e, NULL, 0, sizeof(struct ct_rect_pc));
    ct_host_buffer(t, &s->e, 4 * CT_SIZE * CT_SIZE * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &s->out);
}

/* Names an object (checking VK_SUCCESS), renames it, removes the name with
 * NULL, names it again and removes the name with an empty string. */
static void du_name(struct ct *t, const struct ct_env *e, const struct du_fns *f, VkObjectType type, uint64_t handle,
                    const char *what)
{
    VkDebugUtilsObjectNameInfoEXT info;
    char name[96];
    CT_REQUIREF(t, handle != 0, "%s: no object", what);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = type;
    info.objectHandle = handle;
    snprintf(name, sizeof(name), "the %s", what);
    info.pObjectName = name;
    CT_CHECKF(t, f->name(e->device, &info) == VK_SUCCESS, "naming a %s failed", what);
    snprintf(name, sizeof(name), "the %s, renamed \xe2\x9c\x93", what); /* UTF-8 */
    CT_CHECKF(t, f->name(e->device, &info) == VK_SUCCESS, "renaming a %s failed", what);
    info.pObjectName = NULL;
    CT_CHECKF(t, f->name(e->device, &info) == VK_SUCCESS, "removing the name of a %s with NULL failed", what);
    info.pObjectName = name;
    CT_CHECKF(t, f->name(e->device, &info) == VK_SUCCESS, "naming a %s again failed", what);
    info.pObjectName = "";
    CT_CHECKF(t, f->name(e->device, &info) == VK_SUCCESS, "removing the name of a %s with \"\" failed", what);
    info.pObjectName = name;
    CT_CHECKF(t, f->name(e->device, &info) == VK_SUCCESS, "naming a %s a third time failed", what);
}

#define DU_H(x) ((uint64_t)(uintptr_t)(x))

CT_TEST(debug_utils_object_names,
        "Debugging > Debug Utilities > Object Debug Annotation: vkSetDebugUtilsObjectNameEXT names objects of every "
        "type (instance- and physical-device-level ones through a device of theirs), renames them, and removes a "
        "name given NULL or an empty string (\"any previously set name is removed\"); vkSetDebugUtilsObjectTagEXT "
        "attaches tags; both return VK_SUCCESS; naming does not change the objects (a named pipeline still draws)",
        DU_COMMANDS "vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectTagEXT " CT_RENDER_COMMANDS
        "vkCreateSemaphore vkDestroySemaphore vkCreateFence vkDestroyFence vkCreateEvent vkDestroyEvent "
        "vkCreateQueryPool vkDestroyQueryPool vkCreateBufferView vkDestroyBufferView vkCreatePipelineCache "
        "vkDestroyPipelineCache vkCreateRenderPass vkDestroyRenderPass vkCreateDescriptorSetLayout "
        "vkDestroyDescriptorSetLayout vkCreateSampler vkDestroySampler vkCreateDescriptorPool vkDestroyDescriptorPool "
        "vkAllocateDescriptorSets vkCreateFramebuffer vkDestroyFramebuffer vkCreateCommandPool vkDestroyCommandPool "
        "vkAllocateCommandBuffers vkBeginCommandBuffer vkEndCommandBuffer vkQueueSubmit vkWaitForFences vkResetFences "
        "vkFreeCommandBuffers vkCreateSamplerYcbcrConversion vkDestroySamplerYcbcrConversion "
        "vkCreateDescriptorUpdateTemplate vkDestroyDescriptorUpdateTemplate vkCreatePrivateDataSlot "
        "vkDestroyPrivateDataSlot vkGetPhysicalDeviceFormatProperties vkCmdBeginRendering vkCmdEndRendering "
        "vkCmdBindPipeline vkCmdDraw")
{
    struct ct_scene s;
    struct ct_target c;
    struct ct_gfx g;
    struct ct_buffer texel;
    struct du_fns f;
    struct du_record rec;
    struct ct_box want = { 0, 0, CT_SIZE, CT_SIZE, CT_RED, CT_RED };
    VkDebugUtilsMessengerEXT messenger;
    VkSemaphoreCreateInfo semi;
    VkFenceCreateInfo fi;
    VkEventCreateInfo evi;
    VkQueryPoolCreateInfo qpi;
    VkBufferViewCreateInfo bvi;
    VkPipelineCacheCreateInfo pci;
    VkAttachmentDescription att;
    VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_GENERAL };
    VkSubpassDescription sp;
    VkRenderPassCreateInfo rpi;
    VkDescriptorSetLayoutBinding binding;
    VkDescriptorSetLayoutCreateInfo dsli;
    VkSamplerCreateInfo si;
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
    VkDescriptorPoolCreateInfo dpi;
    VkDescriptorSetAllocateInfo dsai;
    VkFramebufferCreateInfo fbi;
    VkCommandPoolCreateInfo cpi;
    VkCommandBufferAllocateInfo cbai;
    VkDescriptorUpdateTemplateEntry entry;
    VkDescriptorUpdateTemplateCreateInfo dti;
    VkPrivateDataSlotCreateInfo pdi;
    VkDebugUtilsObjectTagInfoEXT tag;
    VkFormatProperties fp;
    VkSemaphore sem = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkEvent event = VK_NULL_HANDLE;
    VkQueryPool qp = VK_NULL_HANDLE;
    VkBufferView bv = VK_NULL_HANDLE;
    VkPipelineCache cache = VK_NULL_HANDLE;
    VkRenderPass rp = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorPool dp = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkFramebuffer fb = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkSamplerYcbcrConversion ycbcr = VK_NULL_HANDLE;
    VkDescriptorUpdateTemplate dut = VK_NULL_HANDLE;
    VkPrivateDataSlot pds = VK_NULL_HANDLE;
    VkPipeline pipe;
    VkCommandBuffer draw;
    const uint32_t tag_data[3] = { 1, 2, 3 };

    du_scene_init(t, &s, &f);
    messenger = du_messenger(t, &f, s.e.instance, &rec, DU_SEVERITIES, DU_TYPES, NULL);
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 256, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT, &texel);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, CT_SIZE, CT_SIZE);
    pipe = ct_gfx_pipeline(t, &s.e, &g);

    memset(&semi, 0, sizeof(semi));
    semi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreateSemaphore(s.e.device, &semi, NULL, &sem));
    memset(&fi, 0, sizeof(fi));
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreateFence(s.e.device, &fi, NULL, &fence));
    memset(&evi, 0, sizeof(evi));
    evi.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    if (!s.e.portability_subset) /* VK_KHR_portability_subset devices may lack events */
        CT_REQUIRE_VK(t, vkCreateEvent(s.e.device, &evi, NULL, &event));
    memset(&qpi, 0, sizeof(qpi));
    qpi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpi.queryType = VK_QUERY_TYPE_OCCLUSION;
    qpi.queryCount = 1;
    CT_REQUIRE_VK(t, vkCreateQueryPool(s.e.device, &qpi, NULL, &qp));
    memset(&bvi, 0, sizeof(bvi));
    bvi.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
    bvi.buffer = texel.buffer;
    bvi.format = VK_FORMAT_R32_UINT; /* uniform texel buffer support is required */
    bvi.range = VK_WHOLE_SIZE;
    CT_REQUIRE_VK(t, vkCreateBufferView(s.e.device, &bvi, NULL, &bv));
    memset(&pci, 0, sizeof(pci));
    pci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    CT_REQUIRE_VK(t, vkCreatePipelineCache(s.e.device, &pci, NULL, &cache));
    memset(&att, 0, sizeof(att));
    att.format = VK_FORMAT_R8G8B8A8_UNORM;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = att.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    memset(&sp, 0, sizeof(sp));
    sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp.colorAttachmentCount = 1;
    sp.pColorAttachments = &ref;
    memset(&rpi, 0, sizeof(rpi));
    rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpi.attachmentCount = 1;
    rpi.pAttachments = &att;
    rpi.subpassCount = 1;
    rpi.pSubpasses = &sp;
    CT_REQUIRE_VK(t, vkCreateRenderPass(s.e.device, &rpi, NULL, &rp));
    memset(&binding, 0, sizeof(binding));
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_ALL;
    memset(&dsli, 0, sizeof(dsli));
    dsli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsli.bindingCount = 1;
    dsli.pBindings = &binding;
    CT_REQUIRE_VK(t, vkCreateDescriptorSetLayout(s.e.device, &dsli, NULL, &dsl));
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 1.0f;
    CT_REQUIRE_VK(t, vkCreateSampler(s.e.device, &si, NULL, &sampler));
    memset(&dpi, 0, sizeof(dpi));
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    CT_REQUIRE_VK(t, vkCreateDescriptorPool(s.e.device, &dpi, NULL, &dp));
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = dp;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    CT_REQUIRE_VK(t, vkAllocateDescriptorSets(s.e.device, &dsai, &set));
    memset(&fbi, 0, sizeof(fbi));
    fbi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbi.renderPass = rp;
    fbi.attachmentCount = 1;
    fbi.pAttachments = &c.view;
    fbi.width = fbi.height = CT_SIZE;
    fbi.layers = 1;
    CT_REQUIRE_VK(t, vkCreateFramebuffer(s.e.device, &fbi, NULL, &fb));
    memset(&cpi, 0, sizeof(cpi));
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.queueFamilyIndex = s.e.queue_family;
    CT_REQUIRE_VK(t, vkCreateCommandPool(s.e.device, &cpi, NULL, &pool));
    memset(&cbai, 0, sizeof(cbai));
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    CT_REQUIRE_VK(t, vkAllocateCommandBuffers(s.e.device, &cbai, &cb));
    vkGetPhysicalDeviceFormatProperties(s.e.pd, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, &fp);
    if (s.f.v11.samplerYcbcrConversion &&
        (fp.optimalTilingFeatures & (VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT |
                                     VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT))) {
        VkSamplerYcbcrConversionCreateInfo yci;
        memset(&yci, 0, sizeof(yci));
        yci.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO;
        yci.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
        yci.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
        yci.ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
        yci.xChromaOffset = yci.yChromaOffset = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT)
                                                    ? VK_CHROMA_LOCATION_MIDPOINT
                                                    : VK_CHROMA_LOCATION_COSITED_EVEN;
        yci.chromaFilter = VK_FILTER_NEAREST;
        CT_REQUIRE_VK(t, vkCreateSamplerYcbcrConversion(s.e.device, &yci, NULL, &ycbcr));
    }
    memset(&entry, 0, sizeof(entry));
    entry.descriptorCount = 1;
    entry.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    entry.stride = sizeof(VkDescriptorBufferInfo);
    memset(&dti, 0, sizeof(dti));
    dti.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
    dti.descriptorUpdateEntryCount = 1;
    dti.pDescriptorUpdateEntries = &entry;
    dti.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
    dti.descriptorSetLayout = dsl;
    CT_REQUIRE_VK(t, vkCreateDescriptorUpdateTemplate(s.e.device, &dti, NULL, &dut));
    if (s.f.v13.privateData) {
        memset(&pdi, 0, sizeof(pdi));
        pdi.sType = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO;
        CT_REQUIRE_VK(t, vkCreatePrivateDataSlot(s.e.device, &pdi, NULL, &pds));
    }

    du_name(t, &s.e, &f, VK_OBJECT_TYPE_INSTANCE, DU_H(s.e.instance), "instance");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_PHYSICAL_DEVICE, DU_H(s.e.pd), "physical device");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_DEVICE, DU_H(s.e.device), "device");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_QUEUE, DU_H(s.e.queue), "queue");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_SEMAPHORE, DU_H(sem), "semaphore");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_COMMAND_BUFFER, DU_H(cb), "command buffer");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_FENCE, DU_H(fence), "fence");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_DEVICE_MEMORY, DU_H(s.out.memory), "device memory");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_BUFFER, DU_H(s.out.buffer), "buffer");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_IMAGE, DU_H(c.image.image), "image");
    if (event)
        du_name(t, &s.e, &f, VK_OBJECT_TYPE_EVENT, DU_H(event), "event");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_QUERY_POOL, DU_H(qp), "query pool");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_BUFFER_VIEW, DU_H(bv), "buffer view");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_IMAGE_VIEW, DU_H(c.view), "image view");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_SHADER_MODULE, DU_H(s.vs), "shader module");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_PIPELINE_CACHE, DU_H(cache), "pipeline cache");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_PIPELINE_LAYOUT, DU_H(s.layout), "pipeline layout");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_RENDER_PASS, DU_H(rp), "render pass");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_PIPELINE, DU_H(pipe), "pipeline");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, DU_H(dsl), "descriptor set layout");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_SAMPLER, DU_H(sampler), "sampler");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_DESCRIPTOR_POOL, DU_H(dp), "descriptor pool");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_DESCRIPTOR_SET, DU_H(set), "descriptor set");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_FRAMEBUFFER, DU_H(fb), "framebuffer");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_COMMAND_POOL, DU_H(pool), "command pool");
    if (ycbcr)
        du_name(t, &s.e, &f, VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION, DU_H(ycbcr), "sampler Y'CbCr conversion");
    du_name(t, &s.e, &f, VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE, DU_H(dut), "descriptor update template");
    if (pds)
        du_name(t, &s.e, &f, VK_OBJECT_TYPE_PRIVATE_DATA_SLOT, DU_H(pds), "private data slot");
    /* VkDebugUtilsMessengerEXT: debug_utils_object_name_messenger */

    memset(&tag, 0, sizeof(tag));
    tag.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_TAG_INFO_EXT;
    tag.objectType = VK_OBJECT_TYPE_BUFFER;
    tag.objectHandle = DU_H(s.out.buffer);
    tag.tagName = 0x1234;
    tag.tagSize = sizeof(tag_data);
    tag.pTag = tag_data;
    CT_CHECK_VK(t, f.tag(s.e.device, &tag), VK_SUCCESS);
    CT_CHECK_VK(t, f.tag(s.e.device, &tag), VK_SUCCESS); /* the same tag again */
    tag.tagName = 0x5678;
    tag.tagSize = 1;
    CT_CHECK_VK(t, f.tag(s.e.device, &tag), VK_SUCCESS);
    tag.objectType = VK_OBJECT_TYPE_PIPELINE;
    tag.objectHandle = DU_H(pipe);
    CT_CHECK_VK(t, f.tag(s.e.device, &tag), VK_SUCCESS);
    tag.objectType = VK_OBJECT_TYPE_DEVICE;
    tag.objectHandle = DU_H(s.e.device);
    CT_CHECK_VK(t, f.tag(s.e.device, &tag), VK_SUCCESS);

    /* The named objects still work. */
    draw = ct_begin_commands(t, &s.e);
    ct_begin_rendering(draw, &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(draw, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(draw, s.layout, -1.0f, -1.0f, 1.0f, 1.0f, ct_red, 0.0f);
    vkCmdEndRendering(draw);
    ct_scene_read(t, &s, draw, &c);
    ct_check_rgba8(t, s.out.data, CT_SIZE, CT_SIZE, ct_expect_box, &want, 0, "drawn with a named pipeline");
    du_check_record(t, &rec, "messenger");

    if (pds)
        vkDestroyPrivateDataSlot(s.e.device, pds, NULL);
    vkDestroyDescriptorUpdateTemplate(s.e.device, dut, NULL);
    if (ycbcr)
        vkDestroySamplerYcbcrConversion(s.e.device, ycbcr, NULL);
    vkFreeCommandBuffers(s.e.device, pool, 1, &cb);
    vkDestroyCommandPool(s.e.device, pool, NULL);
    vkDestroyFramebuffer(s.e.device, fb, NULL);
    vkDestroyDescriptorPool(s.e.device, dp, NULL);
    vkDestroySampler(s.e.device, sampler, NULL);
    vkDestroyDescriptorSetLayout(s.e.device, dsl, NULL);
    vkDestroyRenderPass(s.e.device, rp, NULL);
    vkDestroyPipelineCache(s.e.device, cache, NULL);
    vkDestroyBufferView(s.e.device, bv, NULL);
    vkDestroyQueryPool(s.e.device, qp, NULL);
    if (event)
        vkDestroyEvent(s.e.device, event, NULL);
    vkDestroyFence(s.e.device, fence, NULL);
    vkDestroySemaphore(s.e.device, sem, NULL);
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_buffer_fini(&s.e, &texel);
    ct_target_fini(&s.e, &c);
    f.destroy(s.e.instance, messenger, NULL);
    ct_scene_fini(&s);
}

CT_TEST(debug_utils_object_name_messenger,
        "Debugging > Debug Utilities > Object Debug Annotation: a VkDebugUtilsMessengerEXT (an instance-level object, "
        "VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT) is named and tagged through a device of its instance "
        "(VUID-vkSetDebugUtilsObjectNameEXT-pNameInfo-07872); the commands return VK_SUCCESS and the messenger "
        "still works",
        DU_COMMANDS "vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectTagEXT")
{
    VkDebugUtilsObjectTagInfoEXT tag;
    VkDebugUtilsMessengerEXT m;
    struct du_record rec;
    struct ct_env e;
    struct du_fns f;
    const uint8_t data = 7;

    du_env_init(t, &e, &f);
    ct_env_device(t, &e, NULL);
    m = du_messenger(t, &f, e.instance, &rec, DU_SEVERITIES, DU_TYPES, NULL);
    du_name(t, &e, &f, VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT, DU_H(m), "messenger");
    memset(&tag, 0, sizeof(tag));
    tag.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_TAG_INFO_EXT;
    tag.objectType = VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT;
    tag.objectHandle = DU_H(m);
    tag.tagName = 1;
    tag.tagSize = 1;
    tag.pTag = &data;
    CT_CHECK_VK(t, f.tag(e.device, &tag), VK_SUCCESS);
    du_submit(&f, e.instance, VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT, VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
              "to a named messenger");
    CT_CHECK(t, rec.ours == 1);
    du_check_record(t, &rec, "messenger");
    f.destroy(e.instance, m, NULL);
    ct_env_fini(&e);
}

/* Submits command buffers as one batch and waits (10 s at most). */
static void du_submit_wait(struct ct *t, struct ct_env *e, const VkCommandBuffer *cbs, uint32_t count)
{
    VkSubmitInfo submit;
    VkResult r;
    if (!e->fence) {
        VkFenceCreateInfo info;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        CT_REQUIRE_VK(t, vkCreateFence(e->device, &info, NULL, &e->fence));
    } else {
        CT_REQUIRE_VK(t, vkResetFences(e->device, 1, &e->fence));
    }
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = count;
    submit.pCommandBuffers = cbs;
    CT_REQUIRE_VK(t, vkQueueSubmit(e->queue, 1, &submit, e->fence));
    r = vkWaitForFences(e->device, 1, &e->fence, VK_TRUE, 10000000000ull);
    CT_REQUIREF(t, r == VK_SUCCESS, "vkWaitForFences: %s", ct_result_name(r));
}

/* Fills words [index * 16, index * 16 + 16) of b with value. */
static void du_fill(VkCommandBuffer cb, const struct ct_buffer *b, uint32_t index, uint32_t value)
{
    vkCmdFillBuffer(cb, b->buffer, (VkDeviceSize)index * 64, 64, value);
}

static void du_check_fills(struct ct *t, const struct ct_buffer *b, const uint32_t *values, uint32_t count)
{
    uint32_t i, k;
    for (i = 0; i < count; i++)
        for (k = 0; k < 16; k++) {
            uint32_t v;
            memcpy(&v, b->data + (size_t)i * 64 + k * 4, 4);
            if (v != values[i]) {
                ct_fail(t, __FILE__, __LINE__, "fill %u: word %u is 0x%08x, expected 0x%08x", i, k, v, values[i]);
                break;
            }
        }
}

#define DU_LABEL_COMMANDS                                                                                        \
    "vkQueueBeginDebugUtilsLabelEXT vkQueueEndDebugUtilsLabelEXT vkQueueInsertDebugUtilsLabelEXT "               \
    "vkCmdBeginDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT vkCmdInsertDebugUtilsLabelEXT "

CT_TEST(debug_utils_queue_labels,
        "Debugging > Debug Utilities > Queue Labels: vkQueueBeginDebugUtilsLabelEXT, vkQueueInsertDebugUtilsLabelEXT "
        "and vkQueueEndDebugUtilsLabelEXT open, mark and close nested label regions of a queue around submissions, "
        "which execute as without them",
        DU_LABEL_COMMANDS "vkCreateBuffer vkDestroyBuffer vkGetBufferMemoryRequirements vkAllocateMemory vkFreeMemory "
        "vkBindBufferMemory vkMapMemory vkCreateCommandPool vkDestroyCommandPool vkAllocateCommandBuffers "
        "vkFreeCommandBuffers vkBeginCommandBuffer vkEndCommandBuffer vkCmdFillBuffer vkCmdPipelineBarrier "
        "vkQueueSubmit vkWaitForFences vkCreateFence vkDestroyFence vkResetFences vkQueueWaitIdle")
{
    static const uint32_t values[3] = { 0x11111111u, 0x22222222u, 0x33333333u };
    VkDebugUtilsLabelEXT frame = du_label("frame", 1.0f, 0.0f, 0.0f, 1.0f), inner = du_label("inner", 0, 0, 0, 0);
    VkDebugUtilsLabelEXT mark = du_label("mark", 0.0f, 1.0f, 0.0f, 1.0f);
    struct ct_buffer b;
    struct ct_env e;
    struct du_fns f;
    VkCommandBuffer cb;
    uint32_t i;

    du_env_init(t, &e, &f);
    ct_env_device(t, &e, NULL);
    ct_host_buffer(t, &e, 3 * 64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &b);
    memset(b.data, 0, 3 * 64);
    f.queue_begin(e.queue, &frame);
    for (i = 0; i < 3; i++) {
        if (i == 1) {
            f.queue_insert(e.queue, &mark);
            f.queue_begin(e.queue, &inner);
        }
        cb = ct_begin_commands(t, &e);
        du_fill(cb, &b, i, values[i]);
        ct_host_read_barrier(cb);
        ct_submit_and_wait(t, &e, cb);
        if (i == 1)
            f.queue_end(e.queue);
        if (i == 2)
            f.queue_end(e.queue);
    }
    f.queue_insert(e.queue, &mark); /* with no submission after it */
    CT_CHECK_VK(t, vkQueueWaitIdle(e.queue), VK_SUCCESS);
    du_check_fills(t, &b, values, 3);
    ct_buffer_fini(&e, &b);
    ct_env_fini(&e);
}

CT_TEST(debug_utils_cmd_labels,
        "Debugging > Debug Utilities > Command Buffer Labels: vkCmdBeginDebugUtilsLabelEXT and "
        "vkCmdEndDebugUtilsLabelEXT regions nest, may span render pass instances, may begin in one command buffer and "
        "end in another of the same batch or of a later submission (\"An application may open a debug label region "
        "in one command buffer and close it in another\"), and are balanced within a secondary command buffer; "
        "vkCmdInsertDebugUtilsLabelEXT marks a point; the commands recorded execute as without them",
        DU_LABEL_COMMANDS CT_RENDER_COMMANDS "vkCmdBindPipeline vkCmdDraw vkCmdBeginRendering vkCmdEndRendering "
        "vkCmdFillBuffer vkCmdExecuteCommands vkAllocateCommandBuffers vkFreeCommandBuffers vkBeginCommandBuffer "
        "vkEndCommandBuffer vkQueueSubmit vkWaitForFences vkCreateFence vkResetFences vkDestroyFence "
        "vkCreateCommandPool vkDestroyCommandPool")
{
    static const uint32_t values[4] = { 0xa1a1a1a1u, 0xb2b2b2b2u, 0xc3c3c3c3u, 0xd4d4d4d4u };
    VkDebugUtilsLabelEXT outer = du_label("outer", 1.0f, 1.0f, 1.0f, 1.0f), transfer = du_label("transfer", 0, 0, 1, 1);
    VkDebugUtilsLabelEXT render = du_label("render", 0, 1, 0, 1), draw = du_label("draw", 0, 0, 0, 0);
    VkDebugUtilsLabelEXT spans = du_label("spans command buffers", 0, 0, 0, 0);
    VkDebugUtilsLabelEXT across = du_label("spans submissions", 0, 0, 0, 0), point = du_label("point", 1, 0, 1, 1);
    VkDebugUtilsLabelEXT secondary = du_label("secondary", 0, 0, 0, 0);
    struct ct_box want = { 0, 0, CT_SIZE / 2, CT_SIZE, CT_RED, CT_GREEN };
    VkCommandBufferInheritanceInfo inherit;
    VkCommandBufferAllocateInfo alloc;
    VkCommandBufferBeginInfo begin;
    VkCommandBuffer cbs[2], sec = VK_NULL_HANDLE, cb3, cb4;
    struct ct_buffer fills;
    struct ct_target c;
    struct ct_scene s;
    struct ct_gfx g;
    struct du_fns f;
    VkPipeline pipe;

    du_scene_init(t, &s, &f);
    ct_scene_color(t, &s, &c);
    ct_host_buffer(t, &s.e, 4 * 64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &fills);
    memset(fills.data, 0, 4 * 64);
    ct_gfx_init(&g, s.vs, s.fs, s.layout, CT_SIZE, CT_SIZE);
    pipe = ct_gfx_pipeline(t, &s.e, &g);

    /* One batch of two command buffers and a secondary. */
    cbs[0] = ct_begin_commands(t, &s.e);
    memset(&alloc, 0, sizeof(alloc));
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = s.e.pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    alloc.commandBufferCount = 1;
    CT_REQUIRE_VK(t, vkAllocateCommandBuffers(s.e.device, &alloc, &sec));
    memset(&inherit, 0, sizeof(inherit));
    inherit.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    memset(&begin, 0, sizeof(begin));
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.pInheritanceInfo = &inherit;
    CT_REQUIRE_VK(t, vkBeginCommandBuffer(sec, &begin));
    f.cmd_begin(sec, &secondary);
    f.cmd_insert(sec, &point);
    du_fill(sec, &fills, 3, values[3]);
    f.cmd_end(sec);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(sec));

    f.cmd_begin(cbs[0], &outer);
    f.cmd_begin(cbs[0], &transfer);
    du_fill(cbs[0], &fills, 0, values[0]);
    f.cmd_end(cbs[0]);
    f.cmd_insert(cbs[0], &point);
    ct_transfer_barrier(cbs[0]);
    f.cmd_begin(cbs[0], &render);
    ct_begin_rendering(cbs[0], &c, 1, ct_black, NULL, 0.0f);
    vkCmdBindPipeline(cbs[0], VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    f.cmd_begin(cbs[0], &draw);
    ct_draw_rect(cbs[0], s.layout, -1.0f, -1.0f, 0.0f, 1.0f, ct_red, 0.0f); /* the left half */
    f.cmd_insert(cbs[0], &point);
    f.cmd_end(cbs[0]);
    vkCmdEndRendering(cbs[0]);
    f.cmd_end(cbs[0]);
    f.cmd_begin(cbs[0], &spans);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cbs[0]));
    cbs[1] = ct_begin_commands(t, &s.e);
    du_fill(cbs[1], &fills, 1, values[1]);
    f.cmd_end(cbs[1]); /* spans */
    vkCmdExecuteCommands(cbs[1], 1, &sec);
    f.cmd_end(cbs[1]); /* outer */
    ct_host_read_barrier(cbs[1]);
    CT_REQUIRE_VK(t, vkEndCommandBuffer(cbs[1]));
    du_submit_wait(t, &s.e, cbs, 2);

    /* A region from one submission to the next, around a render pass
     * instance that loads what the first one rendered. */
    cb3 = ct_begin_commands(t, &s.e);
    f.cmd_begin(cb3, &across);
    du_fill(cb3, &fills, 2, values[2]);
    ct_host_read_barrier(cb3);
    ct_submit_and_wait(t, &s.e, cb3);
    cb4 = ct_begin_commands(t, &s.e);
    {
        VkMemoryBarrier mb; /* the first instance's writes before this one's loads */
        memset(&mb, 0, sizeof(mb));
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cb4, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    ct_begin_rendering(cb4, &c, 1, NULL, NULL, 0.0f);
    vkCmdBindPipeline(cb4, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    ct_draw_rect(cb4, s.layout, 0.0f, -1.0f, 1.0f, 1.0f, ct_green, 0.0f); /* the right half */
    vkCmdEndRendering(cb4);
    f.cmd_end(cb4); /* across */
    ct_scene_read(t, &s, cb4, &c);

    ct_check_rgba8(t, s.out.data, CT_SIZE, CT_SIZE, ct_expect_box, &want, 0, "left half red, right half green");
    du_check_fills(t, &fills, values, 4);
    vkFreeCommandBuffers(s.e.device, s.e.pool, 2, cbs);
    vkFreeCommandBuffers(s.e.device, s.e.pool, 1, &sec);
    vkDestroyPipeline(s.e.device, pipe, NULL);
    ct_buffer_fini(&s.e, &fills);
    ct_target_fini(&s.e, &c);
    ct_scene_fini(&s);
}

CT_TEST(debug_utils_proc_addr,
        "Initialization > Command Function Pointers: VK_EXT_debug_utils is an instance extension whose object name "
        "and label commands are device-level: with it enabled on the instance, vkGetInstanceProcAddr returns every "
        "command of it, and vkGetDeviceProcAddr the device-level ones (\"enabled extension device-level dispatchable "
        "command\") but not the instance-level ones; without it, both return NULL for all of them",
        "vkGetInstanceProcAddr vkGetDeviceProcAddr vkSetDebugUtilsObjectNameEXT vkQueueBeginDebugUtilsLabelEXT "
        "vkQueueEndDebugUtilsLabelEXT")
{
    PFN_vkSetDebugUtilsObjectNameEXT name;
    PFN_vkQueueBeginDebugUtilsLabelEXT begin;
    PFN_vkQueueEndDebugUtilsLabelEXT end;
    VkDebugUtilsObjectNameInfoEXT info;
    VkDebugUtilsLabelEXT label = du_label("through vkGetDeviceProcAddr", 0, 0, 0, 0);
    struct ct_env e;
    struct du_fns f;
    size_t i;
    int n = 0;

    du_env_init(t, &e, &f);
    ct_env_device(t, &e, NULL);
    for (i = 0; i < ct_command_count; i++) {
        const struct ct_command *c = &ct_commands[i];
        PFN_vkVoidFunction fn;
        if (strcmp(c->origin, DU_EXT) != 0)
            continue;
        n++;
        CT_CHECKF(t, vkGetInstanceProcAddr(e.instance, c->name) != NULL, "vkGetInstanceProcAddr(\"%s\") is NULL",
                  c->name);
        fn = vkGetDeviceProcAddr(e.device, c->name);
        if (c->level == CT_LEVEL_DEVICE)
            CT_CHECKF(t, fn != NULL, "vkGetDeviceProcAddr(\"%s\") is NULL with " DU_EXT " enabled on the instance",
                      c->name);
        else
            CT_CHECKF(t, fn == NULL, "vkGetDeviceProcAddr(\"%s\") (instance-level) is not NULL", c->name);
    }
    CT_CHECKF(t, n == 11, "%d VK_EXT_debug_utils commands in the command list", n);
    name = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(e.device, "vkSetDebugUtilsObjectNameEXT");
    begin = (PFN_vkQueueBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(e.device, "vkQueueBeginDebugUtilsLabelEXT");
    end = (PFN_vkQueueEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(e.device, "vkQueueEndDebugUtilsLabelEXT");
    CT_REQUIRE(t, name && begin && end);
    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = VK_OBJECT_TYPE_QUEUE;
    info.objectHandle = (uint64_t)(uintptr_t)e.queue;
    info.pObjectName = "named through vkGetDeviceProcAddr";
    CT_CHECK_VK(t, name(e.device, &info), VK_SUCCESS);
    begin(e.queue, &label);
    end(e.queue);
    ct_env_fini(&e);

    ct_env_init(t, &e); /* VK_EXT_debug_utils not enabled */
    ct_env_device(t, &e, NULL);
    for (i = 0; i < ct_command_count; i++) {
        const struct ct_command *c = &ct_commands[i];
        if (strcmp(c->origin, DU_EXT) != 0)
            continue;
        CT_CHECKF(t, vkGetInstanceProcAddr(e.instance, c->name) == NULL,
                  "vkGetInstanceProcAddr(\"%s\") is not NULL without " DU_EXT, c->name);
        CT_CHECKF(t, vkGetDeviceProcAddr(e.device, c->name) == NULL,
                  "vkGetDeviceProcAddr(\"%s\") is not NULL without " DU_EXT, c->name);
    }
    ct_env_fini(&e);
}

static const struct ct_test *const tests[] = {
    CT_REF(tool_properties),
    CT_REF(debug_utils_messenger_filter),
    CT_REF(debug_utils_message_data),
    CT_REF(debug_utils_messenger_destroy),
    CT_REF(debug_utils_messenger_allocator),
    CT_REF(debug_utils_instance_create_info),
    CT_REF(debug_utils_object_names),
    CT_REF(debug_utils_object_name_messenger),
    CT_REF(debug_utils_queue_labels),
    CT_REF(debug_utils_cmd_labels),
    CT_REF(debug_utils_proc_addr),
};

CT_SUITE(debugging, tests)
