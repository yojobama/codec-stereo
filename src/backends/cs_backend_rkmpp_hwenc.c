/*
 * cs_backend_rkmpp_hwenc -- RK3588 VEPU580 hardware ENCODE, software
 * DECODE (CS_MODE_ENCODE_DECODE). Hybrid of cs_backend_rkmpp.c's MPP
 * encode setup and cs_backend_lavc_sw.c's decode-side MV extraction.
 *
 * Motivation: cs_backend_rkmpp reads MVs via MPP's KEY_MOTION_INFO
 * side-channel, which has real, confirmed hardware/firmware defects (see
 * that file's header) -- half a frame's rows never written, and every
 * other 16px column's MV stuck at a fixed placeholder. Those defects are
 * specific to that side-channel. The actual H.264 *bitstream* the encoder
 * produces has no reason to share them -- decoding a bitstream and reading
 * back its motion vectors is an extremely well-trodden path (every
 * software H.264 decoder does exactly this to reconstruct frames), and
 * this project has already validated that exact mechanism via
 * AV_CODEC_FLAG2_EXPORT_MVS in cs_backend_lavc_sw (which passes
 * calibration). So: get the encode speed of real VEPU580 silicon, then
 * sidestep the buggy side-channel entirely by reading MVs the same way
 * lavc_sw does, from a real (software) H.264 decode of the hardware
 * encoder's own bitstream.
 *
 * This costs a real decode pass -- CS_MODE_ENCODE_DECODE, not
 * CS_MODE_ENCODE_READBACK -- trading away part of the "no decode needed"
 * speed advantage encode+readback promised, in exchange for correctness.
 * Whether that trade is worth it in practice is exactly what running
 * tests/test_calibration and cs_bench against this backend will show;
 * that hasn't been done yet as of this file's creation.
 *
 * MPP_ENC_HEADER_MODE_EACH_IDR makes the encoder prepend inline SPS/PPS
 * NAL units to every IDR frame's own packet (matching the official MPP
 * demo's usage) -- our I-frame packet is then self-contained and needs no
 * separate MPP_ENC_GET_HDR_SYNC fetch before feeding it to a decoder.
 */

#include "codec_stereo/cs.h"
#include "codec_stereo/cs_util.h"
#include "cs_backend.h"

#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/mpp_meta.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/rk_venc_cfg.h>
#include <rockchip/rk_venc_cmd.h>

#include <libavcodec/avcodec.h>
#include <libavutil/motion_vector.h>

#include <stdlib.h>
#include <string.h>

#define RK_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

typedef struct rkmpp_hwenc_ctx {
    int block_w, block_h;
    int qp;
    int32_t disparity_offset;

    int cols, rows;
    int16_t *dx, *dy;
    uint16_t *cost; /* always zero; this backend never reports cost */
    uint8_t *flags;
} rkmpp_hwenc_ctx;

static cs_backend_caps rkmpp_hwenc_get_caps(void *vctx) {
    rkmpp_hwenc_ctx *ctx = (rkmpp_hwenc_ctx *)vctx;
    cs_backend_caps caps;
    memset(&caps, 0, sizeof caps);
    caps.mode = CS_MODE_ENCODE_DECODE;
    caps.native_block_w = ctx->block_w;
    caps.native_block_h = ctx->block_h;
    caps.max_search_range_x = 2048; /* VEPU580 search range not configurable/known -- see rkmpp's caveats */
    caps.max_search_range_y = 2048;
    caps.mv_min_x = -2048; caps.mv_max_x = 2047; /* H.264 spec-level MV range, not this HW's actual limit */
    caps.mv_min_y = -512;  caps.mv_max_y = 511;
    caps.supports_subpel = 1; /* H.264 quarter-pel, read directly from motion_scale */
    caps.cost_metric = CS_COST_NONE; /* decoded bitstream carries no cost/SAD */
    return caps;
}

static int rkmpp_hwenc_init(void *vctx, const cs_config *cfg) {
    rkmpp_hwenc_ctx *ctx = (rkmpp_hwenc_ctx *)vctx;

    if (mpp_check_support_format(MPP_CTX_ENC, MPP_VIDEO_CodingAVC) != MPP_OK)
        return -1;
    if (!avcodec_find_decoder(AV_CODEC_ID_H264)) return -1;

    ctx->block_w = cfg->block_w > 0 ? cfg->block_w : 16;
    ctx->block_h = cfg->block_h > 0 ? cfg->block_h : 16;
    ctx->qp = 12;
    ctx->disparity_offset = cfg->disparity_offset;

    if (cfg->backend_params) {
        const char *p = strstr(cfg->backend_params, "qp=");
        if (p) ctx->qp = atoi(p + 3);
    }
    return 0;
}

static int ensure_buffers(rkmpp_hwenc_ctx *ctx, int cols, int rows) {
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

static int fill_nv12(MppBuffer buf, const uint8_t *luma, int stride,
                      int w, int h, int hor_stride, int ver_stride) {
    uint8_t *p = (uint8_t *)mpp_buffer_get_ptr(buf);
    if (!p) return -1;
    size_t frm_size = (size_t)hor_stride * ver_stride * 3 / 2;
    mpp_buffer_sync_begin(buf);
    memset(p, 128, frm_size); /* flat chroma; see lavc_sw/rkmpp's identical choice */
    for (int y = 0; y < h; y++)
        memcpy(p + (size_t)y * hor_stride, luma + (size_t)y * stride, (size_t)w);
    mpp_buffer_sync_end(buf);
    return 0;
}

static int log2_pow2(int v, int fallback) {
    if (v <= 0) return fallback;
    int r = 0;
    while ((1 << r) < v && r < 16) r++;
    return (1 << r) == v ? r : fallback;
}

static int rkmpp_hwenc_extract(void *vctx, const cs_frame *left, const cs_frame *right,
                                cs_mv_field *out) {
    rkmpp_hwenc_ctx *ctx = (rkmpp_hwenc_ctx *)vctx;
    const int w = left->width, h = left->height;
    if (right->width != w || right->height != h) return -1;

    const int cols = (w + ctx->block_w - 1) / ctx->block_w;
    const int rows = (h + ctx->block_h - 1) / ctx->block_h;
    if (ensure_buffers(ctx, cols, rows) != 0) return -1;

    for (int i = 0; i < cols * rows; i++) {
        ctx->dx[i] = 0;
        ctx->dy[i] = 0;
        ctx->flags[i] = (uint8_t)(CS_BLK_INTRA | CS_BLK_NO_MATCH | CS_BLK_NO_COST);
    }

    int hor_stride = RK_ALIGN(w, 64);
    int ver_stride = RK_ALIGN(h, 16);
    size_t frm_size = (size_t)hor_stride * ver_stride * 3 / 2;
    size_t pkt_size = (size_t)hor_stride * ver_stride * 3; /* generous -- see rkmpp's
        bitstream-overflow lesson */

    int ret = -1;
    MppCtx mctx = NULL;
    MppApi *mpi = NULL;
    MppBufferGroup buf_grp = NULL;
    MppEncCfg cfg = NULL;
    MppBuffer frm_buf_l = NULL, frm_buf_r = NULL, pkt_buf_l = NULL, pkt_buf_r = NULL;
    uint8_t *right_shifted = NULL;
    uint8_t *bitstream_i = NULL, *bitstream_p = NULL;
    size_t bitstream_i_len = 0, bitstream_p_len = 0;

    AVCodecContext *dec_ctx = NULL;
    AVPacket *pkt_i = NULL, *pkt_p = NULL;
    AVFrame *dec_frame = NULL;

    if (mpp_create(&mctx, &mpi) != MPP_OK) goto done;
    if (mpp_init(mctx, MPP_CTX_ENC, MPP_VIDEO_CodingAVC) != MPP_OK) goto done;
    if (mpp_buffer_group_get_internal(&buf_grp, MPP_BUFFER_TYPE_DMA_HEAP) != MPP_OK) goto done;

    if (mpp_buffer_get(buf_grp, &frm_buf_l, frm_size) != MPP_OK) goto done;
    if (mpp_buffer_get(buf_grp, &frm_buf_r, frm_size) != MPP_OK) goto done;
    if (mpp_buffer_get(buf_grp, &pkt_buf_l, pkt_size) != MPP_OK) goto done;
    if (mpp_buffer_get(buf_grp, &pkt_buf_r, pkt_size) != MPP_OK) goto done;

    if (fill_nv12(frm_buf_l, left->data[0], left->stride[0], w, h, hor_stride, ver_stride) != 0)
        goto done;

    const uint8_t *right_luma = right->data[0];
    int right_stride = right->stride[0];
    if (ctx->disparity_offset != 0) {
        right_shifted = (uint8_t *)malloc((size_t)w * h);
        if (!right_shifted) goto done;
        cs_shift_gray8(right->data[0], right->stride[0], right_shifted, w,
                        w, h, -ctx->disparity_offset);
        right_luma = right_shifted;
        right_stride = w;
    }
    if (fill_nv12(frm_buf_r, right_luma, right_stride, w, h, hor_stride, ver_stride) != 0)
        goto done;

    mpp_enc_cfg_init(&cfg);
    if (mpi->control(mctx, MPP_ENC_GET_CFG, cfg) != MPP_OK) goto done;

    mpp_enc_cfg_set_s32(cfg, "prep:width", w);
    mpp_enc_cfg_set_s32(cfg, "prep:height", h);
    mpp_enc_cfg_set_s32(cfg, "prep:hor_stride", hor_stride);
    mpp_enc_cfg_set_s32(cfg, "prep:ver_stride", ver_stride);
    mpp_enc_cfg_set_s32(cfg, "prep:format", MPP_FMT_YUV420SP);

    mpp_enc_cfg_set_s32(cfg, "rc:mode", MPP_ENC_RC_MODE_FIXQP);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_flex", 0);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_num", 30);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_denom", 1);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_flex", 0);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_num", 30);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_denom", 1);
    mpp_enc_cfg_set_s32(cfg, "rc:gop", 2);

    mpp_enc_cfg_set_s32(cfg, "rc:qp_init", ctx->qp);
    mpp_enc_cfg_set_s32(cfg, "rc:qp_max", ctx->qp);
    mpp_enc_cfg_set_s32(cfg, "rc:qp_min", ctx->qp);
    mpp_enc_cfg_set_s32(cfg, "rc:qp_max_i", ctx->qp);
    mpp_enc_cfg_set_s32(cfg, "rc:qp_min_i", ctx->qp);
    mpp_enc_cfg_set_s32(cfg, "rc:qp_ip", 0);

    mpp_enc_cfg_set_s32(cfg, "codec:type", MPP_VIDEO_CodingAVC);
    mpp_enc_cfg_set_s32(cfg, "h264:profile", 100);
    mpp_enc_cfg_set_s32(cfg, "h264:level", 40);
    mpp_enc_cfg_set_s32(cfg, "h264:cabac_en", 1);

    if (mpi->control(mctx, MPP_ENC_SET_CFG, cfg) != MPP_OK) goto done;

    {
        RK_U32 header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
        if (mpi->control(mctx, MPP_ENC_SET_HEADER_MODE, &header_mode) != MPP_OK) goto done;
    }

    MppBuffer frm_bufs[2] = {frm_buf_l, frm_buf_r};
    MppBuffer pkt_bufs[2] = {pkt_buf_l, pkt_buf_r};
    for (int i = 0; i < 2; i++) {
        MppFrame frame = NULL;
        MppPacket packet = NULL;

        mpp_frame_init(&frame);
        mpp_frame_set_width(frame, w);
        mpp_frame_set_height(frame, h);
        mpp_frame_set_hor_stride(frame, hor_stride);
        mpp_frame_set_ver_stride(frame, ver_stride);
        mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);
        mpp_frame_set_eos(frame, i == 1 ? 1 : 0);
        mpp_frame_set_buffer(frame, frm_bufs[i]);

        mpp_packet_init_with_buffer(&packet, pkt_bufs[i]);
        mpp_packet_set_length(packet, 0);
        mpp_meta_set_packet(mpp_frame_get_meta(frame), KEY_OUTPUT_PACKET, packet);

        MPP_RET pr = mpi->encode_put_frame(mctx, frame);
        mpp_frame_deinit(&frame);
        if (pr != MPP_OK) goto done;

        MppPacket out_packet = NULL;
        if (mpi->encode_get_packet(mctx, &out_packet) != MPP_OK || !out_packet) goto done;

        void *pos = mpp_packet_get_pos(out_packet);
        size_t len = mpp_packet_get_length(out_packet);
        uint8_t *copy = (uint8_t *)malloc(len);
        if (!copy) { mpp_packet_deinit(&out_packet); goto done; }
        memcpy(copy, pos, len);
        mpp_packet_deinit(&out_packet);

        if (i == 0) { bitstream_i = copy; bitstream_i_len = len; }
        else        { bitstream_p = copy; bitstream_p_len = len; }
    }

    /* Done with MPP -- everything past here is a plain software H.264
       decode of the hardware encoder's own bitstream, identical in
       spirit to cs_backend_lavc_sw's decode side. */
    mpp_enc_cfg_deinit(cfg); cfg = NULL;
    mpp_buffer_put(frm_buf_l); frm_buf_l = NULL;
    mpp_buffer_put(frm_buf_r); frm_buf_r = NULL;
    mpp_buffer_put(pkt_buf_l); pkt_buf_l = NULL;
    mpp_buffer_put(pkt_buf_r); pkt_buf_r = NULL;
    mpp_buffer_group_put(buf_grp); buf_grp = NULL;
    mpp_destroy(mctx); mctx = NULL;

    {
        const AVCodec *dec_codec = avcodec_find_decoder(AV_CODEC_ID_H264);
        if (!dec_codec) goto done;
        dec_ctx = avcodec_alloc_context3(dec_codec);
        if (!dec_ctx) goto done;
        dec_ctx->flags2 |= AV_CODEC_FLAG2_EXPORT_MVS;
        if (avcodec_open2(dec_ctx, dec_codec, NULL) < 0) goto done;
    }

    pkt_i = av_packet_alloc();
    pkt_p = av_packet_alloc();
    dec_frame = av_frame_alloc();
    if (!pkt_i || !pkt_p || !dec_frame) goto done;
    if (av_new_packet(pkt_i, (int)bitstream_i_len) < 0) goto done;
    if (av_new_packet(pkt_p, (int)bitstream_p_len) < 0) goto done;
    memcpy(pkt_i->data, bitstream_i, bitstream_i_len);
    memcpy(pkt_p->data, bitstream_p, bitstream_p_len);
    pkt_i->pts = 0;
    pkt_p->pts = 1;

    out->subpel_bits = 0; /* overwritten below iff a P-frame with side data is decoded */

    AVPacket *dec_inputs[2] = {pkt_i, pkt_p};
    for (int i = 0; i < 2; i++) {
        if (avcodec_send_packet(dec_ctx, dec_inputs[i]) < 0) goto done;
        for (;;) {
            int r = avcodec_receive_frame(dec_ctx, dec_frame);
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
            if (r < 0) goto done;

            if (dec_frame->pts == 1) {
                AVFrameSideData *sd = av_frame_get_side_data(dec_frame, AV_FRAME_DATA_MOTION_VECTORS);
                if (sd) {
                    const AVMotionVector *mvs = (const AVMotionVector *)sd->data;
                    int count = (int)(sd->size / sizeof(AVMotionVector));
                    int subpel_bits = 0, subpel_known = 0;

                    for (int j = 0; j < count; j++) {
                        const AVMotionVector *mv = &mvs[j];
                        if (!subpel_known && mv->motion_scale > 0) {
                            subpel_bits = log2_pow2(mv->motion_scale, 2);
                            subpel_known = 1;
                        }

                        int gx = mv->dst_x / ctx->block_w;
                        int gy = mv->dst_y / ctx->block_h;
                        if (gx < 0 || gx >= cols || gy < 0 || gy >= rows) continue;

                        size_t idx = (size_t)gy * cols + gx;
                        /* Same convention/derivation as lavc_sw: dx s.t. a
                           LEFT-image point at x is visible in RIGHT at
                           x+dx; src_x = dst_x + motion_x/scale with
                           dst=RIGHT(current/P), src=LEFT(reference). */
                        ctx->dx[idx] = (int16_t)(-mv->motion_x);
                        ctx->dy[idx] = (int16_t)(-mv->motion_y);
                        ctx->flags[idx] = (uint8_t)CS_BLK_NO_COST;
                    }
                    out->subpel_bits = subpel_bits;
                }
            }
            av_frame_unref(dec_frame);
        }
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
    free(right_shifted);
    free(bitstream_i);
    free(bitstream_p);
    if (pkt_i) av_packet_free(&pkt_i);
    if (pkt_p) av_packet_free(&pkt_p);
    if (dec_frame) av_frame_free(&dec_frame);
    if (dec_ctx) avcodec_free_context(&dec_ctx);
    if (cfg) mpp_enc_cfg_deinit(cfg);
    if (frm_buf_l) mpp_buffer_put(frm_buf_l);
    if (frm_buf_r) mpp_buffer_put(frm_buf_r);
    if (pkt_buf_l) mpp_buffer_put(pkt_buf_l);
    if (pkt_buf_r) mpp_buffer_put(pkt_buf_r);
    if (buf_grp) mpp_buffer_group_put(buf_grp);
    if (mctx) mpp_destroy(mctx);
    return ret;
}

static void rkmpp_hwenc_destroy(void *vctx) {
    rkmpp_hwenc_ctx *ctx = (rkmpp_hwenc_ctx *)vctx;
    if (!ctx) return;
    free(ctx->dx); free(ctx->dy); free(ctx->cost); free(ctx->flags);
    free(ctx);
}

static void *rkmpp_hwenc_create(void) {
    return calloc(1, sizeof(rkmpp_hwenc_ctx));
}

cs_backend_factory cs_backend_rkmpp_hwenc_factory(void) {
    cs_backend_factory f;
    memset(&f, 0, sizeof f);
    f.create = rkmpp_hwenc_create;
    f.mode = CS_MODE_ENCODE_DECODE;
    f.ops.name = "rkmpp_hwenc";
    f.ops.get_caps = rkmpp_hwenc_get_caps;
    f.ops.init = rkmpp_hwenc_init;
    f.ops.extract = rkmpp_hwenc_extract;
    f.ops.destroy = rkmpp_hwenc_destroy;
    return f;
}
