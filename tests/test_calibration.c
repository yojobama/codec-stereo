/*
 * Synthetic +16px shift calibration test, run against every compiled
 * backend. The right image is the left image shifted by a known amount
 * (cs_shift_gray8, see include/codec_stereo/cs_util.h); this pins down the
 * sign convention and subpel units for each backend, independent of its
 * internal representation -- it goes through the full public
 * cs_extract -> cs_mv_field_to_disparity pipeline, not raw dx/dy, so it
 * exercises exactly what a caller sees.
 */

#include "codec_stereo/cs.h"
#include "codec_stereo/cs_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define IMG_W 128
#define IMG_H 64
#define SHIFT_PX 16
#define BLOCK 16
#define SRC_W (IMG_W + SHIFT_PX)

static void fill_textured(uint8_t *buf, int n) {
    /* Deterministic pseudo-random (xorshift32) fill: enough texture that
       block matches are unambiguous, reproducible across runs/platforms. */
    uint32_t state = 0x9e3779b9u;
    for (int i = 0; i < n; i++) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        buf[i] = (uint8_t)(state & 0xFF);
    }
}

static int run_backend(const char *name) {
    /*
     * left/right are both crops of one wider textured source, offset by
     * SHIFT_PX: left[x] = source[x+SHIFT_PX], right[x] = source[x]. This
     * gives left[c] == right[c+SHIFT_PX] (disparity +SHIFT_PX) using only
     * genuine, never-clamped/replicated pixels on both sides.
     *
     * cs_shift_gray8 (edge-clamped) was tried first here and produced a
     * false failure: clamping creates a degenerate, constant-per-row strip
     * at one edge of whichever frame side a backend's internal block grid
     * anchors on. ref_sad anchors its grid on the LEFT image and searches
     * into RIGHT, so the degenerate strip only ever shows up as one losing
     * candidate in its search sweep (never chosen, since the real block
     * it's matching from is fully textured). lavc_sw's grid instead
     * follows H.264 raster order on the RIGHT image (the P-frame is the
     * "current" frame the codec partitions; the reference/I-frame -- LEFT
     * -- is only ever a search target) -- so for lavc_sw the degenerate
     * strip *is* the anchor block itself, and x264 (correctly) finds a
     * large tied set of equally-perfect matches against it, landing on
     * whichever one its search order happens to prefer. Not a bug in
     * either backend or in cs_shift_gray8 -- just the wrong tool for
     * building an unambiguous test fixture. The crop-based construction
     * below has no clamped/replicated pixels on either side at all.
     */
    uint8_t *source = (uint8_t *)malloc((size_t)SRC_W * IMG_H);
    uint8_t *left = (uint8_t *)malloc((size_t)IMG_W * IMG_H);
    uint8_t *right = (uint8_t *)malloc((size_t)IMG_W * IMG_H);
    if (!source || !left || !right) { free(source); free(left); free(right); return 1; }

    fill_textured(source, SRC_W * IMG_H);
    for (int y = 0; y < IMG_H; y++) {
        memcpy(left + (size_t)y * IMG_W, source + (size_t)y * SRC_W + SHIFT_PX, IMG_W);
        memcpy(right + (size_t)y * IMG_W, source + (size_t)y * SRC_W, IMG_W);
    }
    free(source);

    cs_frame lf = {0}, rf = {0};
    lf.data[0] = left; lf.stride[0] = IMG_W; lf.width = IMG_W; lf.height = IMG_H; lf.fmt = CS_PIX_FMT_GRAY8;
    rf.data[0] = right; rf.stride[0] = IMG_W; rf.width = IMG_W; rf.height = IMG_H; rf.fmt = CS_PIX_FMT_GRAY8;

    cs_config cfg = {0};
    cfg.block_w = BLOCK;
    cfg.block_h = BLOCK;
    cfg.search_range_x = 32;
    cfg.search_range_y = 4;
    cfg.subpel = 0;
    cfg.disparity_offset = 0;
    cfg.backend_override = name;

    int failed = 0;
    cs_context *ctx = cs_init(&cfg);
    if (!ctx) {
        printf("[%-10s] FAIL: cs_init returned NULL\n", name);
        free(left); free(right);
        return 1;
    }

    /*
     * Run cs_extract REPEATEDLY on the same cs_context, not just once.
     * A backend that reuses persistent state across calls (encoder/decoder
     * contexts kept alive instead of recreated per call, an optimization
     * worth making for real throughput -- see Docs Sec. 9's latency
     * numbers) is exactly the kind of change that can pass on the first
     * call and subtly corrupt the second: leftover GOP/reference-frame
     * state, a wrongly-reused buffer, a flush that doesn't fully reset
     * decoder state. A single-call test would never catch that class of
     * bug at all.
     */
    const int reps = 3;
    int checked_total = 0;
    int last_mode = 0, last_subpel = 0;

    for (int rep = 0; rep < reps && !failed; rep++) {
        cs_mv_field field = {0};
        if (cs_extract(ctx, &lf, &rf, &field) != 0) {
            printf("[%-10s] FAIL: cs_extract returned nonzero (rep %d)\n", name, rep);
            failed = 1;
            break;
        }

        float *disp = (float *)malloc((size_t)field.cols * field.rows * sizeof(float));
        cs_disparity_config dcfg = {0};
        dcfg.fx = 1.0f;
        dcfg.baseline = 1.0f;
        dcfg.min_disparity = 0.5f;
        dcfg.max_dy = 2;
        dcfg.max_cost = 0; /* don't gate on cost: some backends report none */
        cs_mv_field_to_disparity(&field, &dcfg, disp);

        /*
         * The outermost column on each side is genuinely unmatched-by-
         * construction, and *which* side depends on which frame a backend's
         * block grid is anchored to: a LEFT-anchored backend (ref_sad) has no
         * valid target for its last column (the shifted search target falls
         * past RIGHT's edge); a RIGHT-anchored backend, following H.264's
         * raster order on the P-frame (lavc_sw), has no valid source for its
         * first column (there's no corresponding LEFT content, since LEFT is
         * the later crop of the source image). Exclude both unconditionally
         * rather than special-case per backend -- only interior columns have
         * a well-defined answer for every anchor convention.
         */
        for (int by = 0; by < field.rows; by++) {
            for (int bx = 1; bx < field.cols - 1; bx++) {
                size_t idx = (size_t)by * field.cols + bx;
                float d = disp[idx];
                if (d == CS_DISPARITY_INVALID) {
                    printf("[%-10s] FAIL: block (%d,%d) invalid disparity (rep %d)\n", name, bx, by, rep);
                    failed = 1;
                    continue;
                }
                if (fabsf(d - (float)SHIFT_PX) > 1.0f) {
                    printf("[%-10s] FAIL: block (%d,%d) disparity=%.3f, expected ~%.1f (rep %d)\n",
                           name, bx, by, d, (float)SHIFT_PX, rep);
                    failed = 1;
                    continue;
                }
                checked_total++;
            }
        }

        last_mode = (int)cs_get_caps(ctx).mode;
        last_subpel = field.subpel_bits;
        free(disp);
    }

    if (!failed) {
        printf("[%-10s] PASS (%d blocks checked over %d reps, mode=%d, subpel_bits=%d)\n",
               name, checked_total, reps, last_mode, last_subpel);
    }
    cs_destroy(ctx);
    free(left);
    free(right);
    return failed;
}

int main(void) {
    const cs_backend_ops *const *backends = cs_list_backends();
    int any_failed = 0;
    int count = 0;

    for (int i = 0; backends[i] != NULL; i++) {
        count++;
        if (run_backend(backends[i]->name) != 0) any_failed = 1;
    }

    if (count == 0) {
        printf("No backends compiled in -- nothing to test.\n");
        return 1;
    }

    return any_failed;
}
