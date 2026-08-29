/*
 * cs_bench -- latency benchmark: median cs_extract() time over N iterations
 * on a synthetic textured pair, plus the cs_mv_field_to_disparity conversion
 * cost. Reported as one end-to-end "extract" number for now; splitting it
 * into encode/readback sub-stages needs per-backend instrumentation hooks
 * that don't exist yet (tracked for the rkmpp/d3d12_vme bring-up phases).
 *
 * Usage: cs_bench [--backend NAME] --size WxH [--iters N]
 *                  [--block W H] [--search SX SY]
 */

#include "codec_stereo/cs.h"
#include "codec_stereo/cs_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

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
    int w = 1920, h = 1080, iters = 30;
    cs_config cfg = {0};

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            cfg.backend_override = argv[++i];
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            if (sscanf(argv[++i], "%dx%d", &w, &h) != 2) {
                fprintf(stderr, "--size expects WxH\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--block") == 0 && i + 2 < argc) {
            cfg.block_w = atoi(argv[++i]);
            cfg.block_h = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--search") == 0 && i + 2 < argc) {
            cfg.search_range_x = atoi(argv[++i]);
            cfg.search_range_y = atoi(argv[++i]);
        } else {
            fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
            return 2;
        }
    }

    if (iters < 1) iters = 1;

    uint8_t *left = (uint8_t *)malloc((size_t)w * h);
    uint8_t *right = (uint8_t *)malloc((size_t)w * h);
    if (!left || !right) { fprintf(stderr, "OOM\n"); return 1; }
    fill_textured(left, w * h);
    cs_shift_gray8(left, w, right, w, w, h, 4); /* small plausible disparity */

    cs_context *ctx = cs_init(&cfg);
    if (!ctx) {
        fprintf(stderr, "cs_init failed\n");
        free(left); free(right);
        return 1;
    }

    cs_frame lf = {0}, rf = {0};
    lf.data[0] = left; lf.stride[0] = w; lf.width = w; lf.height = h; lf.fmt = CS_PIX_FMT_GRAY8;
    rf.data[0] = right; rf.stride[0] = w; rf.width = w; rf.height = h; rf.fmt = CS_PIX_FMT_GRAY8;

    double *extract_ms = (double *)malloc((size_t)iters * sizeof(double));
    double *convert_ms = (double *)malloc((size_t)iters * sizeof(double));
    float *disp = NULL;

    for (int i = 0; i < iters; i++) {
        cs_mv_field field = {0};

        double t0 = now_ms();
        int rc = cs_extract(ctx, &lf, &rf, &field);
        double t1 = now_ms();
        if (rc != 0) {
            fprintf(stderr, "cs_extract failed on iteration %d\n", i);
            cs_destroy(ctx);
            free(left); free(right); free(extract_ms); free(convert_ms);
            return 1;
        }
        extract_ms[i] = t1 - t0;

        if (!disp) disp = (float *)malloc((size_t)field.cols * field.rows * sizeof(float));
        cs_disparity_config dcfg = {0};
        dcfg.min_disparity = 0.5f;
        double t2 = now_ms();
        cs_mv_field_to_disparity(&field, &dcfg, disp);
        double t3 = now_ms();
        convert_ms[i] = t3 - t2;
    }

    qsort(extract_ms, iters, sizeof(double), cmp_double);
    qsort(convert_ms, iters, sizeof(double), cmp_double);

    printf("backend:        %s\n", cs_get_backend_name(ctx));
    printf("frame size:     %dx%d\n", w, h);
    printf("iterations:     %d\n", iters);
    printf("extract  (ms):  median=%.3f  min=%.3f  max=%.3f\n",
           extract_ms[iters / 2], extract_ms[0], extract_ms[iters - 1]);
    printf("convert  (ms):  median=%.3f  min=%.3f  max=%.3f\n",
           convert_ms[iters / 2], convert_ms[0], convert_ms[iters - 1]);

    free(disp);
    free(extract_ms);
    free(convert_ms);
    cs_destroy(ctx);
    free(left);
    free(right);
    return 0;
}
