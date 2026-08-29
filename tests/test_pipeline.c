/*
 * Verifies cs_pipeline against the single-threaded cs_extract path: the
 * whole point of the worker pool is that it's a pure throughput
 * optimization, so N workers processing the same pair must return bitwise
 * the same disparity as one context doing it alone, in submission order,
 * every time. Concurrency bugs here (a shared buffer, a context used from
 * two threads, results returned out of order) would be invisible to the
 * per-backend correctness tests.
 */

#include "codec_stereo/cs.h"
#include "codec_stereo/cs_pipeline.h"
#include "codec_stereo/cs_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMG_W 256
#define IMG_H 128
#define SHIFT_PX 12
#define SRC_W (IMG_W + SHIFT_PX)
#define PAIRS 20

static void fill_textured(uint8_t *buf, int n) {
    uint32_t state = 0x9e3779b9u;
    for (int i = 0; i < n; i++) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        buf[i] = (uint8_t)(state & 0xFF);
    }
}

int main(int argc, char **argv) {
    const cs_backend_ops *const *backends = cs_list_backends();
    if (!backends[0]) {
        printf("No backends compiled in -- nothing to test.\n");
        return 1;
    }
    /* Defaults to the first compiled-in backend so this runs on every build
       config; pass a name to target a specific one (e.g. rkmpp_hwenc, to
       check that several concurrent MPP encoder contexts actually coexist,
       which the queue plumbing alone can't tell you). */
    const char *name = (argc > 1) ? argv[1] : backends[0]->name;

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
    cfg.block_w = 16;
    cfg.block_h = 16;
    cfg.search_range_x = 24;
    cfg.search_range_y = 4;
    cfg.backend_override = name;

    cs_disparity_config dcfg = {0};
    dcfg.fx = 1.0f;
    dcfg.baseline = 1.0f;
    dcfg.min_disparity = 0.5f;
    dcfg.max_dy = 2;

    /* Reference: single context, single thread. */
    cs_context *ctx = cs_init(&cfg);
    if (!ctx) { printf("FAIL: cs_init failed for %s\n", name); return 1; }
    cs_mv_field field = {0};
    if (cs_extract(ctx, &lf, &rf, &field) != 0) {
        printf("FAIL: reference cs_extract failed\n");
        cs_destroy(ctx);
        return 1;
    }
    int cols = field.cols, rows = field.rows;
    float *ref = (float *)malloc((size_t)cols * rows * sizeof(float));
    cs_mv_field_to_disparity(&field, &dcfg, ref);
    cs_destroy(ctx);

    int failed = 0;
    cs_pipeline *pipe = cs_pipeline_create(&cfg, &dcfg, 4);
    if (!pipe) { printf("FAIL: cs_pipeline_create failed\n"); free(ref); return 1; }

    int submitted = 0, received = 0;
    while (received < PAIRS) {
        while (submitted < PAIRS && submitted - received < 8) {
            if (cs_pipeline_submit(pipe, &lf, &rf) != 0) break;
            submitted++;
        }
        cs_pipeline_result res;
        if (cs_pipeline_get(pipe, &res) != 0) {
            printf("FAIL: cs_pipeline_get failed at %d\n", received);
            failed = 1;
            break;
        }
        if (res.seq != (uint64_t)received) {
            printf("FAIL: out-of-order result: got seq %llu, expected %d\n",
                   (unsigned long long)res.seq, received);
            failed = 1;
        }
        if (res.status != 0) {
            printf("FAIL: pair %d reported status %d\n", received, res.status);
            failed = 1;
        } else if (res.cols != cols || res.rows != rows) {
            printf("FAIL: pair %d grid %dx%d != reference %dx%d\n",
                   received, res.cols, res.rows, cols, rows);
            failed = 1;
        } else if (memcmp(res.disparity, ref, (size_t)cols * rows * sizeof(float)) != 0) {
            printf("FAIL: pair %d disparity differs from single-threaded reference\n", received);
            failed = 1;
        }
        received++;
    }

    cs_pipeline_destroy(pipe);

    if (!failed)
        printf("[%-10s] PASS (%d pairs through 4 workers, all bitwise-identical "
               "to single-threaded, all in order)\n", name, received);

    free(ref);
    free(left);
    free(right);
    return failed;
}
