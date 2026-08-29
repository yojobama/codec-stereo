/*
 * cs_backend_ref_sad -- brute-force SAD block matcher, no rate-distortion
 * term, no predictor, no codec involved at all (CS_MODE_DIRECT).
 *
 * This is not a production backend. It exists as the experiment's control:
 * it runs over the exact same block grid and search window as the codec
 * backends, so a harness comparing it against e.g. cs_backend_lavc_sw
 * isolates error caused by block granularity (both suffer it identically)
 * from error caused by an encoder's rate-distortion/predictor bias (only
 * the codec backend suffers that). See Docs/codec-stereo-DESIGN.md Sec. 8.
 */

#include "codec_stereo/cs.h"
#include "cs_backend.h"

#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct ref_sad_ctx {
    int block_w, block_h;
    int search_range_x, search_range_y;
    int32_t disparity_offset;

    int cols, rows;
    int16_t *dx, *dy;
    uint16_t *cost;
    uint8_t *flags;
} ref_sad_ctx;

static cs_backend_caps ref_sad_get_caps(void *vctx) {
    ref_sad_ctx *ctx = (ref_sad_ctx *)vctx;
    cs_backend_caps caps;
    memset(&caps, 0, sizeof caps);
    caps.mode = CS_MODE_DIRECT;
    caps.native_block_w = ctx->block_w;
    caps.native_block_h = ctx->block_h;
    /* Software brute force has no hardware ceiling; this is a practical cap. */
    caps.max_search_range_x = 4096;
    caps.max_search_range_y = 256;
    caps.mv_min_x = -ctx->search_range_x;
    caps.mv_max_x = ctx->search_range_x;
    caps.mv_min_y = -ctx->search_range_y;
    caps.mv_max_y = ctx->search_range_y;
    caps.supports_subpel = 0;
    caps.cost_metric = CS_COST_SAD;
    return caps;
}

static int ref_sad_init(void *vctx, const cs_config *cfg) {
    ref_sad_ctx *ctx = (ref_sad_ctx *)vctx;
    ctx->block_w = cfg->block_w > 0 ? cfg->block_w : 16;
    ctx->block_h = cfg->block_h > 0 ? cfg->block_h : 16;
    ctx->search_range_x = cfg->search_range_x > 0 ? cfg->search_range_x : 64;
    ctx->search_range_y = cfg->search_range_y > 0 ? cfg->search_range_y : 4;
    ctx->disparity_offset = cfg->disparity_offset;
    return 0;
}

static int ensure_buffers(ref_sad_ctx *ctx, int cols, int rows) {
    if (cols == ctx->cols && rows == ctx->rows && ctx->dx) return 0;

    free(ctx->dx);
    free(ctx->dy);
    free(ctx->cost);
    free(ctx->flags);

    size_t n = (size_t)cols * (size_t)rows;
    ctx->dx = (int16_t *)malloc(n * sizeof(int16_t));
    ctx->dy = (int16_t *)malloc(n * sizeof(int16_t));
    ctx->cost = (uint16_t *)malloc(n * sizeof(uint16_t));
    ctx->flags = (uint8_t *)malloc(n * sizeof(uint8_t));
    if (!ctx->dx || !ctx->dy || !ctx->cost || !ctx->flags) {
        free(ctx->dx); ctx->dx = NULL;
        free(ctx->dy); ctx->dy = NULL;
        free(ctx->cost); ctx->cost = NULL;
        free(ctx->flags); ctx->flags = NULL;
        ctx->cols = ctx->rows = 0;
        return -1;
    }

    ctx->cols = cols;
    ctx->rows = rows;
    return 0;
}

/* Sum of absolute differences between a bw x bh block of `left` anchored at
   (lx, ly) and a same-size block of `right` anchored at (rx, ry). Both
   blocks are assumed to lie fully within their frame bounds by the caller. */
static long block_sad(const uint8_t *left, int left_stride,
                       const uint8_t *right, int right_stride,
                       int lx, int ly, int rx, int ry, int bw, int bh) {
    long sad = 0;
    for (int j = 0; j < bh; j++) {
        const uint8_t *lrow = left + (size_t)(ly + j) * left_stride + lx;
        const uint8_t *rrow = right + (size_t)(ry + j) * right_stride + rx;
        for (int i = 0; i < bw; i++) {
            int d = (int)lrow[i] - (int)rrow[i];
            sad += d < 0 ? -d : d;
        }
    }
    return sad;
}

static int ref_sad_extract(void *vctx, const cs_frame *left, const cs_frame *right,
                            cs_mv_field *out) {
    ref_sad_ctx *ctx = (ref_sad_ctx *)vctx;
    const int w = left->width, h = left->height;

    if (right->width != w || right->height != h) return -1;

    const int bw = ctx->block_w, bh = ctx->block_h;
    const int cols = (w + bw - 1) / bw;
    const int rows = (h + bh - 1) / bh;

    if (ensure_buffers(ctx, cols, rows) != 0) return -1;

    const uint8_t *lplane = left->data[0];
    const uint8_t *rplane = right->data[0];
    const int lstride = left->stride[0];
    const int rstride = right->stride[0];

    for (int by = 0; by < rows; by++) {
        int y0 = by * bh;
        int bh_eff = (y0 + bh <= h) ? bh : (h - y0);

        for (int bx = 0; bx < cols; bx++) {
            int x0 = bx * bw;
            int bw_eff = (x0 + bw <= w) ? bw : (w - x0);
            size_t idx = (size_t)by * cols + bx;

            long best_sad = LONG_MAX;
            int best_dx = 0, best_dy = 0;
            int found = 0, clamped = 0;

            for (int cdy = -ctx->search_range_y; cdy <= ctx->search_range_y; cdy++) {
                int ry = y0 + cdy;
                if (ry < 0 || ry + bh_eff > h) continue;

                for (int cdx = -ctx->search_range_x; cdx <= ctx->search_range_x; cdx++) {
                    int rx = x0 + (int)ctx->disparity_offset + cdx;
                    if (rx < 0 || rx + bw_eff > w) continue;

                    long sad = block_sad(lplane, lstride, rplane, rstride,
                                          x0, y0, rx, ry, bw_eff, bh_eff);
                    if (sad < best_sad) {
                        best_sad = sad;
                        best_dx = cdx;
                        best_dy = cdy;
                        found = 1;
                        clamped = (cdx == -ctx->search_range_x || cdx == ctx->search_range_x ||
                                   cdy == -ctx->search_range_y || cdy == ctx->search_range_y);
                    }
                }
            }

            if (!found) {
                ctx->dx[idx] = 0;
                ctx->dy[idx] = 0;
                ctx->cost[idx] = 0;
                ctx->flags[idx] = CS_BLK_NO_MATCH;
                continue;
            }

            ctx->dx[idx] = (int16_t)best_dx;
            ctx->dy[idx] = (int16_t)best_dy;
            ctx->cost[idx] = (uint16_t)(best_sad > 65535 ? 65535 : best_sad);
            ctx->flags[idx] = clamped ? (uint8_t)CS_BLK_CLAMPED : (uint8_t)CS_BLK_OK;
        }
    }

    out->dx = ctx->dx;
    out->dy = ctx->dy;
    out->cost = ctx->cost;
    out->flags = ctx->flags;
    out->block_w = bw;
    out->block_h = bh;
    out->cols = cols;
    out->rows = rows;
    out->subpel_bits = 0;
    out->disparity_offset = ctx->disparity_offset;
    return 0;
}

static void ref_sad_destroy(void *vctx) {
    ref_sad_ctx *ctx = (ref_sad_ctx *)vctx;
    if (!ctx) return;
    free(ctx->dx);
    free(ctx->dy);
    free(ctx->cost);
    free(ctx->flags);
    free(ctx);
}

static void *ref_sad_create(void) {
    return calloc(1, sizeof(ref_sad_ctx));
}

cs_backend_factory cs_backend_ref_sad_factory(void) {
    cs_backend_factory f;
    memset(&f, 0, sizeof f);
    f.create = ref_sad_create;
    f.mode = CS_MODE_DIRECT;
    f.ops.name = "ref_sad";
    f.ops.get_caps = ref_sad_get_caps;
    f.ops.init = ref_sad_init;
    f.ops.extract = ref_sad_extract;
    f.ops.destroy = ref_sad_destroy;
    return f;
}
