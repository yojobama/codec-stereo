#include "codec_stereo/cs.h"
#include "cs_backend.h"

#include <stdlib.h>
#include <string.h>

#define CS_MAX_BACKENDS 8

struct cs_context {
    void *backend_ctx;
    cs_backend_ops ops;
};

static const cs_backend_factory *cs_all_factories(size_t *count) {
    static cs_backend_factory arr[CS_MAX_BACKENDS];
    static size_t n = 0;
    static int inited = 0;

    if (!inited) {
#ifdef CS_ENABLE_REF_SAD
        arr[n++] = cs_backend_ref_sad_factory();
#endif
#ifdef CS_ENABLE_LAVC
        arr[n++] = cs_backend_lavc_sw_factory();
#endif
#ifdef CS_ENABLE_RKMPP
        arr[n++] = cs_backend_rkmpp_factory();
#endif
#ifdef CS_ENABLE_RKMPP_HWENC
        arr[n++] = cs_backend_rkmpp_hwenc_factory();
#endif
#ifdef CS_ENABLE_D3D12
        arr[n++] = cs_backend_d3d12_vme_factory();
#endif
        inited = 1;
    }

    *count = n;
    return arr;
}

/* Lower number = tried first when no explicit backend_override is given.
   CS_MODE_DIRECT (the ref_sad diagnostic backend) is deliberately lowest
   priority: it exists as a validation control, not a production path. */
static int mode_priority(cs_extract_mode mode) {
    switch (mode) {
        case CS_MODE_ME_ONLY:         return 0;
        case CS_MODE_ENCODE_READBACK: return 1;
        case CS_MODE_ENCODE_DECODE:   return 2;
        case CS_MODE_DIRECT:          return 3;
        default:                      return 4;
    }
}

const cs_backend_ops *const *cs_list_backends(void) {
    static const cs_backend_ops *arr[CS_MAX_BACKENDS + 1];
    static int inited = 0;

    if (!inited) {
        size_t count;
        const cs_backend_factory *factories = cs_all_factories(&count);
        size_t i;
        for (i = 0; i < count; i++) arr[i] = &factories[i].ops;
        arr[i] = NULL;
        inited = 1;
    }

    return arr;
}

static const cs_backend_factory *find_factory_by_name(const char *name, size_t *idx) {
    size_t count;
    const cs_backend_factory *factories = cs_all_factories(&count);
    for (size_t i = 0; i < count; i++) {
        if (factories[i].ops.name && strcmp(factories[i].ops.name, name) == 0) {
            if (idx) *idx = i;
            return &factories[i];
        }
    }
    return NULL;
}

/* Returns an array of factory indices sorted by mode_priority (stable,
   insertion-order tiebreak). Writes the count to *count. */
static void priority_order(size_t *order, size_t *count) {
    const cs_backend_factory *factories = cs_all_factories(count);
    for (size_t i = 0; i < *count; i++) order[i] = i;

    /* Simple stable insertion sort; backend counts are tiny (<= 8). */
    for (size_t i = 1; i < *count; i++) {
        size_t key = order[i];
        int key_pri = mode_priority(factories[key].mode);
        size_t j = i;
        while (j > 0) {
            int prev_pri = mode_priority(factories[order[j - 1]].mode);
            if (prev_pri <= key_pri) break;
            order[j] = order[j - 1];
            j--;
        }
        order[j] = key;
    }
}

static cs_context *try_init_factory(const cs_backend_factory *f, const cs_config *cfg) {
    void *backend_ctx = f->create ? f->create() : NULL;
    if (!backend_ctx) return NULL;

    if (f->ops.init && f->ops.init(backend_ctx, cfg) != 0) {
        if (f->ops.destroy) f->ops.destroy(backend_ctx);
        return NULL;
    }

    cs_context *ctx = (cs_context *)malloc(sizeof(cs_context));
    if (!ctx) {
        if (f->ops.destroy) f->ops.destroy(backend_ctx);
        return NULL;
    }
    ctx->backend_ctx = backend_ctx;
    ctx->ops = f->ops;
    return ctx;
}

cs_context *cs_init(const cs_config *cfg) {
    static const cs_config default_cfg = {0, 0, 0, 0, 0, 0, NULL, NULL};
    if (!cfg) cfg = &default_cfg;

    if (cfg->backend_override) {
        const cs_backend_factory *f = find_factory_by_name(cfg->backend_override, NULL);
        if (!f) return NULL;
        return try_init_factory(f, cfg);
    }

    size_t count;
    size_t order[CS_MAX_BACKENDS];
    priority_order(order, &count);

    size_t fcount;
    const cs_backend_factory *factories = cs_all_factories(&fcount);

    for (size_t i = 0; i < count; i++) {
        cs_context *ctx = try_init_factory(&factories[order[i]], cfg);
        if (ctx) return ctx;
    }

    return NULL;
}

cs_backend_caps cs_get_caps(const cs_context *ctx) {
    return ctx->ops.get_caps(ctx->backend_ctx);
}

const char *cs_get_backend_name(const cs_context *ctx) {
    return ctx->ops.name;
}

int cs_extract(cs_context *ctx, const cs_frame *left, const cs_frame *right,
               cs_mv_field *out) {
    return ctx->ops.extract(ctx->backend_ctx, left, right, out);
}

void cs_destroy(cs_context *ctx) {
    if (!ctx) return;
    if (ctx->ops.destroy) ctx->ops.destroy(ctx->backend_ctx);
    free(ctx);
}
