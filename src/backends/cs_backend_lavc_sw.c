/*
 * cs_backend_lavc_sw -- universal software reference backend
 * (CS_MODE_ENCODE_DECODE): encode left as an I-frame and right as a single
 * P-frame referencing it with libx264, decode with
 * AV_CODEC_FLAG2_EXPORT_MVS, read back AV_FRAME_DATA_MOTION_VECTORS.
 *
 * Correctness over speed: every extract() call opens fresh encoder/decoder
 * contexts. Each stereo pair is logically an independent 2-frame sequence
 * (I referencing nothing, P referencing only that I) -- reusing encoder
 * state across calls would risk incorrect GOP/reference-frame bookkeeping
 * for no benefit, since this backend is documented as best-effort/
 * reference-only latency (Docs/codec-stereo-DESIGN.md Sec. 3), not the
 * fast path.
 *
 * Sub-partitioning is disabled (x264 "partitions=none"): every P-slice
 * macroblock is exactly one 16x16 partition, so with the default 16x16
 * block_w/h this backend's grid aligns exactly with x264's, and each
 * AVMotionVector maps to exactly one grid cell via its center position.
 * (AVMotionVector.dst_x/dst_y are documented as block *centers*, not
 * top-left corners -- see libavutil/motion_vector.h and
 * libavcodec ff_print_debug_info2's add_mb() caller.)
 */

#include "codec_stereo/cs.h"
#include "codec_stereo/cs_util.h"
#include "cs_backend.h"

#include <libavcodec/avcodec.h>
#include <libavutil/motion_vector.h>
#include <libavutil/opt.h>
#include <libavutil/frame.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct lavc_sw_ctx {
    int block_w, block_h;
    int merange;          /* x264 has one scalar search range, not separate x/y */
    int subpel_enabled;
    int qp;
    char me[8];            /* x264 "me" method: esa (exhaustive, default) or a
                               faster heuristic (umh/hex/dia) for large merange,
                               where esa's O(merange^2) cost becomes intractable */
    int32_t disparity_offset;

    int cols, rows;
    int16_t *dx, *dy;
    uint16_t *cost;        /* always zero; this backend never reports cost */
    uint8_t *flags;
} lavc_sw_ctx;

/* Parses a minimal "key=value;key2=value2" string looking for overrides
   this backend recognizes. Unknown keys are ignored. */
static void apply_backend_params(lavc_sw_ctx *ctx, const char *params) {
    if (!params) return;
    char buf[256];
    strncpy(buf, params, sizeof buf - 1);
    buf[sizeof buf - 1] = '\0';

    for (char *tok = strtok(buf, ";"); tok; tok = strtok(NULL, ";")) {
        char *eq = strchr(tok, '=');
        if (!eq) continue;
        *eq = '\0';
        const char *key = tok, *val = eq + 1;
        if (strcmp(key, "qp") == 0) ctx->qp = atoi(val);
        else if (strcmp(key, "me") == 0) {
            strncpy(ctx->me, val, sizeof ctx->me - 1);
            ctx->me[sizeof ctx->me - 1] = '\0';
        }
    }
}

static cs_backend_caps lavc_sw_get_caps(void *vctx) {
    lavc_sw_ctx *ctx = (lavc_sw_ctx *)vctx;
    cs_backend_caps caps;
    memset(&caps, 0, sizeof caps);
    caps.mode = CS_MODE_ENCODE_DECODE;
    caps.native_block_w = 16; /* H.264 MB size; partitions=none locks this */
    caps.native_block_h = 16;
    caps.max_search_range_x = 2048; /* x264's practical merange ceiling */
    caps.max_search_range_y = 2048;
    /* x264 exposes one scalar search range (merange), applied symmetrically
       to both axes -- there is no separate vertical range to under-report. */
    caps.mv_min_x = -ctx->merange;
    caps.mv_max_x = ctx->merange;
    caps.mv_min_y = -ctx->merange;
    caps.mv_max_y = ctx->merange;
    caps.supports_subpel = ctx->subpel_enabled;
    caps.cost_metric = CS_COST_NONE;
    return caps;
}

static int lavc_sw_init(void *vctx, const cs_config *cfg) {
    lavc_sw_ctx *ctx = (lavc_sw_ctx *)vctx;

    if (!avcodec_find_encoder_by_name("libx264")) return -1;
    if (!avcodec_find_decoder(AV_CODEC_ID_H264)) return -1;

    ctx->block_w = cfg->block_w > 0 ? cfg->block_w : 16;
    ctx->block_h = cfg->block_h > 0 ? cfg->block_h : 16;

    int rx = cfg->search_range_x > 0 ? cfg->search_range_x : 0;
    int ry = cfg->search_range_y > 0 ? cfg->search_range_y : 0;
    ctx->merange = rx > ry ? rx : ry;
    if (ctx->merange <= 0) ctx->merange = 16;

    ctx->subpel_enabled = cfg->subpel;
    ctx->disparity_offset = cfg->disparity_offset;
    /*
     * Chosen from a QP sweep {0,5,10,18,26} against Middlebury
     * Motorcycle-perfect (harness/cs_eval.cpp --lavc-qp): RMSE was
     * essentially flat across 5-26 (32.2-32.7) at ~50-54% density -- the
     * "lower QP -> less RD/predictor bias -> better accuracy" hypothesis
     * this project started with barely held in practice. QP=0 did get a
     * meaningfully better RMSE (31.0) but density collapsed to 12%: at
     * near-zero lambda the encoder's mode decision stopped favoring
     * inter-coding at all for most blocks, so the "improvement" is over a
     * much smaller, easier subset, not a real win. 10 sits in the flat
     * part of the curve with normal density; nothing in this sweep
     * justifies moving it.
     */
    ctx->qp = 10;
    strncpy(ctx->me, "esa", sizeof ctx->me - 1); /* exhaustive; O(merange^2),
        override to umh/hex/dia via backend_params for large merange */

    apply_backend_params(ctx, cfg->backend_params);
    return 0;
}

static int ensure_buffers(lavc_sw_ctx *ctx, int cols, int rows) {
    if (cols == ctx->cols && rows == ctx->rows && ctx->dx) return 0;

    free(ctx->dx); free(ctx->dy); free(ctx->cost); free(ctx->flags);

    size_t n = (size_t)cols * (size_t)rows;
    ctx->dx = (int16_t *)malloc(n * sizeof(int16_t));
    ctx->dy = (int16_t *)malloc(n * sizeof(int16_t));
    ctx->cost = (uint16_t *)calloc(n, sizeof(uint16_t));
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

static int log2_pow2(int v, int fallback) {
    if (v <= 0) return fallback;
    int r = 0;
    while ((1 << r) < v && r < 16) r++;
    return (1 << r) == v ? r : fallback;
}

static AVFrame *make_yuv420p_frame(const cs_frame *src, int64_t pts, int pict_type) {
    AVFrame *f = av_frame_alloc();
    if (!f) return NULL;
    f->format = AV_PIX_FMT_YUV420P;
    f->width = src->width;
    f->height = src->height;
    f->pts = pts;
    f->pict_type = (enum AVPictureType)pict_type;

    if (av_frame_get_buffer(f, 32) < 0) {
        av_frame_free(&f);
        return NULL;
    }

    for (int y = 0; y < src->height; y++) {
        memcpy(f->data[0] + (size_t)y * f->linesize[0],
               src->data[0] + (size_t)y * src->stride[0],
               (size_t)src->width);
    }

    int cw = (src->width + 1) / 2, ch = (src->height + 1) / 2;
    for (int y = 0; y < ch; y++) {
        memset(f->data[1] + (size_t)y * f->linesize[1], 128, (size_t)cw);
        memset(f->data[2] + (size_t)y * f->linesize[2], 128, (size_t)cw);
    }

    return f;
}

static int lavc_sw_extract(void *vctx, const cs_frame *left, const cs_frame *right,
                            cs_mv_field *out) {
    lavc_sw_ctx *ctx = (lavc_sw_ctx *)vctx;
    const int w = left->width, h = left->height;
    if (right->width != w || right->height != h) return -1;

    const int cols = (w + ctx->block_w - 1) / ctx->block_w;
    const int rows = (h + ctx->block_h - 1) / ctx->block_h;
    if (ensure_buffers(ctx, cols, rows) != 0) return -1;

    /* Every cell starts unresolved (no coded partition covers it yet). */
    for (int i = 0; i < cols * rows; i++) {
        ctx->dx[i] = 0;
        ctx->dy[i] = 0;
        ctx->cost[i] = 0;
        ctx->flags[i] = (uint8_t)(CS_BLK_INTRA | CS_BLK_NO_MATCH | CS_BLK_NO_COST);
    }

    int ret = -1;
    AVCodecContext *enc_ctx = NULL, *dec_ctx = NULL;
    AVFrame *fl = NULL, *fr = NULL, *dec_frame = NULL;
    AVPacket *pkt = NULL;
    uint8_t *right_shifted_buf = NULL;

    const AVCodec *enc_codec = avcodec_find_encoder_by_name("libx264");
    const AVCodec *dec_codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!enc_codec || !dec_codec) goto done;

    enc_ctx = avcodec_alloc_context3(enc_codec);
    if (!enc_ctx) goto done;
    enc_ctx->width = w;
    enc_ctx->height = h;
    enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    enc_ctx->time_base = (AVRational){1, 25};
    enc_ctx->gop_size = 2; /* exactly this one I+P pair */
    enc_ctx->max_b_frames = 0;

    {
        char params[256];
        snprintf(params, sizeof params,
                  "qp=%d:me=%s:merange=%d:subme=%d:partitions=none:ref=1:"
                  "bframes=0:rc-lookahead=0:mbtree=0:scenecut=0:weightp=0:"
                  "aq-mode=0:trellis=0:psy=0:8x8dct=0:threads=1",
                  ctx->qp, ctx->me, ctx->merange, ctx->subpel_enabled ? 7 : 1);
        av_opt_set(enc_ctx->priv_data, "preset", "ultrafast", 0);
        av_opt_set(enc_ctx->priv_data, "tune", "zerolatency", 0);
        av_opt_set(enc_ctx->priv_data, "x264-params", params, 0);
    }

    if (avcodec_open2(enc_ctx, enc_codec, NULL) < 0) goto done;

    dec_ctx = avcodec_alloc_context3(dec_codec);
    if (!dec_ctx) goto done;
    dec_ctx->flags2 |= AV_CODEC_FLAG2_EXPORT_MVS;
    if (avcodec_open2(dec_ctx, dec_codec, NULL) < 0) goto done;

    /*
     * x264 has no way for us to bias where its own motion search centers --
     * it always searches around the neighbor-derived MV predictor (zero for
     * the first MB), with radius merange. To reach a disparity range that
     * doesn't straddle zero, physically pre-shift the right image by
     * disparity_offset before encoding (so the *residual* motion x264 needs
     * to find is small / near its own predictor), then let the existing
     * dx + disparity_offset formula in cs_mv_field_to_disparity add the
     * shift back. Mirrors what ref_sad does directly in its own indexed
     * search (rx = x0 + disparity_offset + cdx) -- same disparity_offset
     * contract, different mechanism, since a codec has no equivalent knob.
     */
    const cs_frame *right_for_encode = right;
    cs_frame right_shifted;
    if (ctx->disparity_offset != 0) {
        right_shifted_buf = (uint8_t *)malloc((size_t)w * h);
        if (!right_shifted_buf) goto done;
        cs_shift_gray8(right->data[0], right->stride[0], right_shifted_buf, w,
                        w, h, -ctx->disparity_offset);
        right_shifted = *right;
        right_shifted.data[0] = right_shifted_buf;
        right_shifted.stride[0] = w;
        right_for_encode = &right_shifted;
    }

    fl = make_yuv420p_frame(left, 0, AV_PICTURE_TYPE_I);
    fr = make_yuv420p_frame(right_for_encode, 1, AV_PICTURE_TYPE_P);
    if (!fl || !fr) goto done;

    pkt = av_packet_alloc();
    dec_frame = av_frame_alloc();
    if (!pkt || !dec_frame) goto done;

    out->subpel_bits = 0; /* overwritten below iff a P-frame with side data is decoded */

    /* Encode both frames, then flush; feed every resulting packet straight
       into the decoder in the same order (encode-order == decode-order
       here since there are no B-frames / reordering configured). */
    AVFrame *enc_inputs[2] = {fl, fr};
    for (int i = 0; i < 2; i++) {
        if (avcodec_send_frame(enc_ctx, enc_inputs[i]) < 0) goto done;
        for (;;) {
            int r = avcodec_receive_packet(enc_ctx, pkt);
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
            if (r < 0) goto done;
            if (avcodec_send_packet(dec_ctx, pkt) < 0) { av_packet_unref(pkt); goto done; }
            av_packet_unref(pkt);
        }
    }
    avcodec_send_frame(enc_ctx, NULL); /* flush encoder */
    for (;;) {
        int r = avcodec_receive_packet(enc_ctx, pkt);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
        if (r < 0) goto done;
        if (avcodec_send_packet(dec_ctx, pkt) < 0) { av_packet_unref(pkt); goto done; }
        av_packet_unref(pkt);
    }
    avcodec_send_packet(dec_ctx, NULL); /* flush decoder */

    /* Drain decoded frames, keep the one with pts == 1 (the P-frame / right
       image); the I-frame (pts == 0) carries no inter motion by definition. */
    for (;;) {
        int r = avcodec_receive_frame(dec_ctx, dec_frame);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
        if (r < 0) goto done;

        if (dec_frame->pts == 1) {
            AVFrameSideData *sd = av_frame_get_side_data(dec_frame, AV_FRAME_DATA_MOTION_VECTORS);
            if (sd) {
                const AVMotionVector *mvs = (const AVMotionVector *)sd->data;
                int count = (int)(sd->size / sizeof(AVMotionVector));
                int subpel_bits = 0;
                int subpel_known = 0;

                for (int i = 0; i < count; i++) {
                    const AVMotionVector *mv = &mvs[i];
                    if (!subpel_known && mv->motion_scale > 0) {
                        subpel_bits = log2_pow2(mv->motion_scale, 2);
                        subpel_known = 1;
                    }

                    int gx = mv->dst_x / ctx->block_w;
                    int gy = mv->dst_y / ctx->block_h;
                    if (gx < 0 || gx >= cols || gy < 0 || gy >= rows) continue;

                    size_t idx = (size_t)gy * cols + gx;
                    /* Library convention: dx s.t. a LEFT-image point at x is
                       visible in RIGHT at x+dx. AVMotionVector gives
                       src_x = dst_x + motion_x/motion_scale with dst=RIGHT
                       (current/P frame) and src=LEFT (reference); so
                       dx = dst_x - src_x = -motion_x/motion_scale. */
                    ctx->dx[idx] = (int16_t)(-mv->motion_x);
                    ctx->dy[idx] = (int16_t)(-mv->motion_y);
                    ctx->cost[idx] = 0;
                    ctx->flags[idx] = (uint8_t)CS_BLK_NO_COST;
                }

                out->subpel_bits = subpel_bits;
            } else {
                out->subpel_bits = 0;
            }
        }
        av_frame_unref(dec_frame);
    }

    out->dx = ctx->dx;
    out->dy = ctx->dy;
    out->cost = ctx->cost;
    out->flags = ctx->flags;
    out->block_w = ctx->block_w;
    out->block_h = ctx->block_h;
    out->cols = cols;
    out->rows = rows;
    out->disparity_offset = ctx->disparity_offset;
    ret = 0;

done:
    if (dec_frame) av_frame_free(&dec_frame);
    if (pkt) av_packet_free(&pkt);
    if (fl) av_frame_free(&fl);
    if (fr) av_frame_free(&fr);
    if (enc_ctx) avcodec_free_context(&enc_ctx);
    if (dec_ctx) avcodec_free_context(&dec_ctx);
    free(right_shifted_buf);
    return ret;
}

static void lavc_sw_destroy(void *vctx) {
    lavc_sw_ctx *ctx = (lavc_sw_ctx *)vctx;
    if (!ctx) return;
    free(ctx->dx); free(ctx->dy); free(ctx->cost); free(ctx->flags);
    free(ctx);
}

static void *lavc_sw_create(void) {
    return calloc(1, sizeof(lavc_sw_ctx));
}

cs_backend_factory cs_backend_lavc_sw_factory(void) {
    cs_backend_factory f;
    memset(&f, 0, sizeof f);
    f.create = lavc_sw_create;
    f.mode = CS_MODE_ENCODE_DECODE;
    f.ops.name = "lavc_sw";
    f.ops.get_caps = lavc_sw_get_caps;
    f.ops.init = lavc_sw_init;
    f.ops.extract = lavc_sw_extract;
    f.ops.destroy = lavc_sw_destroy;
    return f;
}
