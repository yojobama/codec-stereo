/*
 * cs_depth -- CLI: rectified stereo pair (PGM) in, disparity map (PFM) out.
 *
 * Usage:
 *   cs_depth LEFT.pgm RIGHT.pgm OUT.pfm
 *            [--backend NAME] [--block W H] [--search SX SY]
 *            [--offset D] [--min-disparity D]
 *            [--cross-check] [--upsample nearest|bilinear]
 *
 * --cross-check and --upsample are the opt-in extras from
 * Docs/codec-stereo-DESIGN.md Sec. 8 -- both off by default, both added
 * cost: cross-check runs a second R->L extraction (~2x latency) and
 * upsample expands the block-granular output to per-pixel resolution.
 */

#include "codec_stereo/cs.h"
#include "codec_stereo/cs_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s LEFT.pgm RIGHT.pgm OUT.pfm [--backend NAME] "
        "[--block W H] [--search SX SY] [--offset D] [--min-disparity D] "
        "[--cross-check] [--upsample nearest|bilinear]\n",
        argv0);
}

int main(int argc, char **argv) {
    if (argc < 4) { usage(argv[0]); return 2; }

    const char *left_path = argv[1];
    const char *right_path = argv[2];
    const char *out_path = argv[3];

    cs_config cfg = {0};
    float min_disparity = 0.5f;
    int do_cross_check = 0;
    int do_upsample = 0;
    cs_upsample_mode upsample_mode = CS_UPSAMPLE_NEAREST;

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
        } else if (strcmp(argv[i], "--cross-check") == 0) {
            do_cross_check = 1;
        } else if (strcmp(argv[i], "--upsample") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "nearest") == 0) upsample_mode = CS_UPSAMPLE_NEAREST;
            else if (strcmp(mode, "bilinear") == 0) upsample_mode = CS_UPSAMPLE_BILINEAR;
            else { fprintf(stderr, "--upsample expects nearest or bilinear\n"); return 2; }
            do_upsample = 1;
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

    if (do_cross_check) {
        /* R->L pass: swap which frame is "left"/"right" for this second
           extraction. disparity_offset is negated -- the physical search
           bias flips direction when the anchor/target roles swap (see
           Docs Sec. 8's cross-check note and cs_disparity_cross_check's
           docstring on matching sign conventions). */
        cs_config back_cfg = cfg;
        back_cfg.disparity_offset = -cfg.disparity_offset;
        cs_context *back_ctx = cs_init(&back_cfg);
        if (!back_ctx) {
            fprintf(stderr, "cross-check: cs_init (backward pass) failed\n");
            free(disp);
            cs_destroy(ctx);
            free(left); free(right);
            return 1;
        }

        cs_mv_field back_field = {0};
        int back_rc = cs_extract(back_ctx, &rf, &lf, &back_field); /* note: swapped */
        if (back_rc != 0 || back_field.cols != field.cols || back_field.rows != field.rows) {
            fprintf(stderr, "cross-check: backward extraction failed or grid mismatch\n");
            cs_destroy(back_ctx);
            free(disp);
            cs_destroy(ctx);
            free(left); free(right);
            return 1;
        }

        float *back_disp = (float *)malloc((size_t)field.cols * field.rows * sizeof(float));
        if (!back_disp) {
            cs_destroy(back_ctx);
            free(disp);
            cs_destroy(ctx);
            free(left); free(right);
            return 1;
        }
        /*
         * cs_mv_field_to_disparity's min_disparity gate assumes a
         * positive-valid convention (near-zero-or-negative = invalid/at-
         * infinity, per Design Sec. 7) -- but the backward pass's genuine,
         * valid matches come out on the *other* side of zero from the
         * forward pass (that's the whole point: forward+backward should
         * sum to ~0 for a consistent pair). Reusing dcfg's min_disparity
         * here would invalidate every real backward match. Disable that
         * gate for this pass; cs_disparity_cross_check's own consistency
         * check is the actual validity test for these values.
         */
        cs_disparity_config back_dcfg = dcfg;
        back_dcfg.min_disparity = -1e6f;
        cs_mv_field_to_disparity(&back_field, &back_dcfg, back_disp);

        cs_disparity_cross_check(disp, back_disp, field.cols, field.rows,
                                  field.block_w, 2.0f, field.block_w * 2, disp);

        free(back_disp);
        cs_destroy(back_ctx);
    }

    const float *to_write = disp;
    float *upsampled = NULL;
    int write_w = field.cols, write_h = field.rows;

    if (do_upsample) {
        upsampled = (float *)malloc((size_t)lw * lh * sizeof(float));
        if (!upsampled) {
            free(disp);
            cs_destroy(ctx);
            free(left); free(right);
            return 1;
        }
        cs_disparity_upsample(disp, field.cols, field.rows, field.block_w, field.block_h,
                               upsample_mode, upsampled, lw, lh);
        to_write = upsampled;
        write_w = lw;
        write_h = lh;
    }

    if (cs_pfm_write(out_path, to_write, write_w, write_h) != 0) {
        fprintf(stderr, "failed to write %s\n", out_path);
        free(upsampled);
        free(disp);
        cs_destroy(ctx);
        free(left); free(right);
        return 1;
    }

    fprintf(stderr, "wrote %dx%d disparity (block %dx%d)%s%s to %s\n",
            write_w, write_h, field.block_w, field.block_h,
            do_cross_check ? ", cross-checked" : "",
            do_upsample ? ", upsampled" : "",
            out_path);

    free(upsampled);
    free(disp);
    cs_destroy(ctx);
    free(left);
    free(right);
    return 0;
}
