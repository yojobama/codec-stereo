/*
 * cs_depth -- CLI: rectified stereo pair (PGM) in, disparity map (PFM) out.
 *
 * Usage:
 *   cs_depth LEFT.pgm RIGHT.pgm OUT.pfm
 *            [--backend NAME] [--block W H] [--search SX SY]
 *            [--offset D] [--min-disparity D]
 */

#include "codec_stereo/cs.h"
#include "codec_stereo/cs_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s LEFT.pgm RIGHT.pgm OUT.pfm [--backend NAME] "
        "[--block W H] [--search SX SY] [--offset D] [--min-disparity D]\n",
        argv0);
}

int main(int argc, char **argv) {
    if (argc < 4) { usage(argv[0]); return 2; }

    const char *left_path = argv[1];
    const char *right_path = argv[2];
    const char *out_path = argv[3];

    cs_config cfg = {0};
    float min_disparity = 0.5f;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            cfg.backend_override = argv[++i];
        } else if (strcmp(argv[i], "--block") == 0 && i + 2 < argc) {
            cfg.block_w = atoi(argv[++i]);
            cfg.block_h = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--search") == 0 && i + 2 < argc) {
            cfg.search_range_x = atoi(argv[++i]);
            cfg.search_range_y = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--offset") == 0 && i + 1 < argc) {
            cfg.disparity_offset = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--min-disparity") == 0 && i + 1 < argc) {
            min_disparity = (float)atof(argv[++i]);
        } else {
            fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    uint8_t *left = NULL, *right = NULL;
    int lw, lh, rw, rh;
    if (cs_pgm_read(left_path, &left, &lw, &lh) != 0) {
        fprintf(stderr, "failed to read %s (expected binary P5 PGM)\n", left_path);
        return 1;
    }
    if (cs_pgm_read(right_path, &right, &rw, &rh) != 0) {
        fprintf(stderr, "failed to read %s (expected binary P5 PGM)\n", right_path);
        free(left);
        return 1;
    }
    if (lw != rw || lh != rh) {
        fprintf(stderr, "left/right size mismatch: %dx%d vs %dx%d\n", lw, lh, rw, rh);
        free(left); free(right);
        return 1;
    }

    cs_context *ctx = cs_init(&cfg);
    if (!ctx) {
        fprintf(stderr, "cs_init failed%s%s\n",
                cfg.backend_override ? " for backend " : "",
                cfg.backend_override ? cfg.backend_override : "");
        free(left); free(right);
        return 1;
    }
    fprintf(stderr, "backend: %s\n", cs_get_backend_name(ctx));

    cs_frame lf = {0}, rf = {0};
    lf.data[0] = left; lf.stride[0] = lw; lf.width = lw; lf.height = lh; lf.fmt = CS_PIX_FMT_GRAY8;
    rf.data[0] = right; rf.stride[0] = rw; rf.width = rw; rf.height = rh; rf.fmt = CS_PIX_FMT_GRAY8;

    cs_mv_field field = {0};
    if (cs_extract(ctx, &lf, &rf, &field) != 0) {
        fprintf(stderr, "cs_extract failed\n");
        cs_destroy(ctx);
        free(left); free(right);
        return 1;
    }

    float *disp = (float *)malloc((size_t)field.cols * field.rows * sizeof(float));
    if (!disp) {
        cs_destroy(ctx);
        free(left); free(right);
        return 1;
    }

    cs_disparity_config dcfg = {0};
    dcfg.fx = 1.0f;       /* depth conversion is a separate step; this tool emits disparity */
    dcfg.baseline = 1.0f;
    dcfg.min_disparity = min_disparity;
    dcfg.max_dy = 0;      /* 0 = no dy gate */
    dcfg.max_cost = 0;    /* 0 = no cost gate */
    cs_mv_field_to_disparity(&field, &dcfg, disp);

    if (cs_pfm_write(out_path, disp, field.cols, field.rows) != 0) {
        fprintf(stderr, "failed to write %s\n", out_path);
        free(disp);
        cs_destroy(ctx);
        free(left); free(right);
        return 1;
    }

    fprintf(stderr, "wrote %dx%d disparity (block %dx%d) to %s\n",
            field.cols, field.rows, field.block_w, field.block_h, out_path);

    free(disp);
    cs_destroy(ctx);
    free(left);
    free(right);
    return 0;
}
