/* A minimal XML reader for vk.xml: elements, attributes and text.
 * Comments, processing instructions and DOCTYPE are skipped. */
#ifndef PVK_XML_H
#define PVK_XML_H

#include "tool.h"

struct xml_node {
    const char *name; /* NULL for a text node */
    const char *text; /* text nodes only, entities decoded */
    const char **attrs; /* name, value, name, value, ... */
    int attr_count;
    struct xml_node **children;
    int child_count, child_cap;
};

static inline char *xml_decode(const char *s, size_t n)
{
    static const struct { const char *entity; char c; } named[] = {
        { "&lt;", '<' }, { "&gt;", '>' }, { "&amp;", '&' }, { "&quot;", '"' }, { "&apos;", '\'' },
    };
    char *out = xmalloc(n + 1), *o = out;
    const char *end = s + n;
    while (s < end) {
        size_t i;
        if (*s != '&') {
            *o++ = *s++;
            continue;
        }
        if (s[1] == '#') {
            int hex = s[2] == 'x';
            unsigned long c = strtoul(s + 2 + hex, NULL, hex ? 16 : 10);
            *o++ = (char)c; /* vk.xml only uses ASCII references */
            s = strchr(s, ';') + 1;
            continue;
        }
        for (i = 0; i < sizeof(named) / sizeof(named[0]); i++)
            if (starts_with(s, named[i].entity))
                break;
        if (i == sizeof(named) / sizeof(named[0]))
            die("xml: unknown entity near '%.16s'", s);
        *o++ = named[i].c;
        s += strlen(named[i].entity);
    }
    *o = '\0';
    return out;
}

static inline struct xml_node *xml_new(void)
{
    struct xml_node *n = xmalloc(sizeof(*n));
    memset(n, 0, sizeof(*n));
    return n;
}

static inline void xml_append(struct xml_node *parent, struct xml_node *child)
{
    if (parent->child_count == parent->child_cap) {
        parent->child_cap = parent->child_cap ? parent->child_cap * 2 : 8;
        parent->children = xrealloc(parent->children, sizeof(*parent->children) * (size_t)parent->child_cap);
    }
    parent->children[parent->child_count++] = child;
}

static inline int xml_name_char(int c)
{
    return is_ident(c) || c == ':' || c == '.' || c == '-';
}

static inline const char *xml_skip(const char *p, const char *terminator)
{
    const char *end = strstr(p, terminator);
    if (!end)
        die("xml: unterminated construct near '%.16s'", p);
    return end + strlen(terminator);
}

/* Parses a whole document and returns its root element. */
static inline struct xml_node *xml_parse(const char *p)
{
    struct xml_node *stack[256], *root = NULL;
    int depth = 0;

    while (*p) {
        if (*p != '<') {
            const char *end = strchr(p, '<');
            if (!end)
                end = p + strlen(p);
            if (depth > 0) {
                struct xml_node *text = xml_new();
                text->text = xml_decode(p, (size_t)(end - p));
                xml_append(stack[depth - 1], text);
            }
            p = end;
        } else if (starts_with(p, "<!--")) {
            p = xml_skip(p, "-->");
        } else if (starts_with(p, "<?")) {
            p = xml_skip(p, "?>");
        } else if (starts_with(p, "<![CDATA[")) {
            const char *end = strstr(p, "]]>");
            struct xml_node *text = xml_new();
            if (!end || depth == 0)
                die("xml: bad CDATA");
            text->text = xstrndup(p + 9, (size_t)(end - p - 9));
            xml_append(stack[depth - 1], text);
            p = end + 3;
        } else if (p[1] == '!') {
            p = xml_skip(p, ">");
        } else if (p[1] == '/') {
            if (depth == 0)
                die("xml: unbalanced close tag");
            depth--;
            p = xml_skip(p, ">");
        } else {
            struct xml_node *node = xml_new();
            const char *start = ++p;
            int cap = 0;
            while (xml_name_char(*p))
                p++;
            node->name = xstrndup(start, (size_t)(p - start));
            for (;;) {
                const char *name, *value;
                size_t name_len;
                char quote;
                while (isspace((unsigned char)*p))
                    p++;
                if (*p == '/' || *p == '>')
                    break;
                name = p;
                while (xml_name_char(*p))
                    p++;
                name_len = (size_t)(p - name);
                while (isspace((unsigned char)*p))
                    p++;
                if (*p++ != '=' || (*p != '"' && *p != '\''))
                    die("xml: bad attribute in <%s>", node->name);
                quote = *p++;
                value = p;
                while (*p && *p != quote)
                    p++;
                if (node->attr_count + 2 > cap) {
                    cap = cap ? cap * 2 : 8;
                    node->attrs = xrealloc(node->attrs, sizeof(*node->attrs) * (size_t)cap);
                }
                node->attrs[node->attr_count++] = xstrndup(name, name_len);
                node->attrs[node->attr_count++] = xml_decode(value, (size_t)(p - value));
                p++;
            }
            if (depth > 0)
                xml_append(stack[depth - 1], node);
            else if (!root)
                root = node;
            if (*p == '/') {
                p = xml_skip(p, ">");
            } else {
                if (depth == (int)(sizeof(stack) / sizeof(stack[0])))
                    die("xml: nesting too deep");
                stack[depth++] = node;
                p++;
            }
        }
    }
    if (!root)
        die("xml: no root element");
    return root;
}

static inline const char *xml_attr(const struct xml_node *n, const char *name)
{
    int i;
    for (i = 0; i + 1 < n->attr_count; i += 2)
        if (strcmp(n->attrs[i], name) == 0)
            return n->attrs[i + 1];
    return NULL;
}

static inline int xml_is(const struct xml_node *n, const char *name)
{
    return n->name && strcmp(n->name, name) == 0;
}

/* First child element called NAME, or NULL. */
static inline struct xml_node *xml_child(const struct xml_node *n, const char *name)
{
    int i;
    for (i = 0; i < n->child_count; i++)
        if (xml_is(n->children[i], name))
            return n->children[i];
    return NULL;
}

/* Appends all descendant text, in document order. */
static inline void xml_text(const struct xml_node *n, struct buf *out)
{
    int i;
    if (!n->name) {
        buf_puts(out, n->text);
        return;
    }
    for (i = 0; i < n->child_count; i++)
        xml_text(n->children[i], out);
}

#endif /* PVK_XML_H */
