/* Shared helpers for PlasmaVK's development tools (not part of plasmavk.h). */
#ifndef PVK_TOOL_H
#define PVK_TOOL_H

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline void die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
static inline void die(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    exit(1);
}

static inline void *xmalloc(size_t size)
{
    void *p = malloc(size ? size : 1);
    if (!p)
        die("out of memory");
    return p;
}

static inline void *xrealloc(void *p, size_t size)
{
    p = realloc(p, size ? size : 1);
    if (!p)
        die("out of memory");
    return p;
}

static inline char *xstrdup(const char *s)
{
    size_t n = strlen(s) + 1;
    return memcpy(xmalloc(n), s, n);
}

static inline char *xstrndup(const char *s, size_t n)
{
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static inline int is_ident(int c)
{
    return isalnum(c) || c == '_';
}

static inline const char *skip_space(const char *p)
{
    while (*p == ' ' || *p == '\t')
        p++;
    return p;
}

static inline int starts_with(const char *s, const char *prefix)
{
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static inline int ends_with(const char *s, const char *suffix)
{
    size_t n = strlen(s), m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

/* Writes the directory part of PATH ("." if none) into OUT. */
static inline void parent_dir(char *out, size_t size, const char *path)
{
    const char *slash = strrchr(path, '/');
    if (!slash)
        snprintf(out, size, ".");
    else if (slash == path)
        snprintf(out, size, "/");
    else
        snprintf(out, size, "%.*s", (int)(slash - path), path);
}

/* ---- Growable byte buffer ---- */

struct buf {
    char *data;
    size_t len, cap;
};

static inline void buf_write(struct buf *b, const char *s, size_t n)
{
    if (b->len + n + 1 > b->cap) {
        b->cap = (b->len + n + 1) * 2;
        b->data = xrealloc(b->data, b->cap);
    }
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

static inline void buf_puts(struct buf *b, const char *s)
{
    buf_write(b, s, strlen(s));
}

static inline void buf_putc(struct buf *b, char c)
{
    buf_write(b, &c, 1);
}

static inline void buf_printf(struct buf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static inline void buf_printf(struct buf *b, const char *fmt, ...)
{
    va_list args, copy;
    int n;
    va_start(args, fmt);
    va_copy(copy, args);
    n = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);
    if (n < 0)
        die("vsnprintf failed");
    if (b->len + (size_t)n + 1 > b->cap) {
        b->cap = (b->len + (size_t)n + 1) * 2;
        b->data = xrealloc(b->data, b->cap);
    }
    vsnprintf(b->data + b->len, (size_t)n + 1, fmt, args);
    b->len += (size_t)n;
    va_end(args);
}

/* ---- Files ---- */

static inline char *read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    char *data;
    long size;
    if (!f)
        die("cannot open %s", path);
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    data = xmalloc((size_t)size + 1);
    if (fread(data, 1, (size_t)size, f) != (size_t)size)
        die("cannot read %s", path);
    data[size] = '\0';
    fclose(f);
    return data;
}

/* Writes only when the content changed, so make sees stable timestamps. */
static inline void write_file(const char *path, const struct buf *b)
{
    FILE *f = fopen(path, "rb");
    const char *data = b->data ? b->data : "";
    if (f) {
        char *old;
        long size;
        fseek(f, 0, SEEK_END);
        size = ftell(f);
        fclose(f);
        if ((size_t)size == b->len) {
            old = read_file(path);
            if (memcmp(old, data, b->len) == 0) {
                free(old);
                return;
            }
            free(old);
        }
    }
    f = fopen(path, "wb");
    if (!f || fwrite(data, 1, b->len, f) != b->len || fclose(f) != 0)
        die("cannot write %s", path);
}

#endif /* PVK_TOOL_H */
