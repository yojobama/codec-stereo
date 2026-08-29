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
 * Measured (CS_RKMPP_HWENC_TIMING, 1920x1080, qp=45): of ~93ms total,
 * software decode is ~57ms (61%) and the *I-frame's* decode alone is
 * ~51ms of that -- the P-frame (the one we actually need MVs from) only
 * takes ~6ms. The I-frame's CABAC entropy decode dominates, and that
 * scales with how much detail got coded (QP), not with reconstruction
 * work: CS_RKMPP_HWENC_SKIP_RECON (skip_idct/skip_loop_filter, since we
 * never look at decoded pixels) saved under 10% -- entropy decode still
 * has to fully parse every coded coefficient to know how many bits to
 * consume, whether or not the IDCT is later applied to them. Raising QP
 * is the lever that actually works: qp=12 (default, this backend's own
 * accuracy-first choice, matching lavc_sw/rkmpp's rationale) produces a
 * ~2.2MB I-frame and ~386ms total; qp=45 produces a ~327KB I-frame and
 * ~93ms total, with disparity accuracy on a synthetic known-shift pair
 * unaffected (still exact to within 16.0 +/- ~1px at the edges). A
 * genuinely separate, coarser I-frame-only QP was attempted (qp_i,
 * below) and confirmed NOT to work under MPP_ENC_RC_MODE_FIXQP.
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
#include <libavutil/mem.h>
#include <libavutil/motion_vector.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Set CS_RKMPP_HWENC_TIMING to any value to print a phase-by-phase
   breakdown to stderr: MPP setup, HW encode, MPP teardown, decoder open,
   and SW decode (per packet) -- see cs_bench's own median-of-N harness for
   whole-call timing; this is for finding out *where* the time goes inside
   one call. */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

#define RK_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

typedef struct rkmpp_hwenc_ctx {
    int block_w, block_h;
    int qp;    /* nominal QP; also the P-frame's actual QP */
    int qp_i;  /* intended as a separate, coarser I-frame QP (see below) --
                  VERIFIED NOT TO WORK under MPP_ENC_RC_MODE_FIXQP: tested
                  qp=12/qp_i=45 expecting a small I-frame + precise P-frame,
                  but the P-frame's encoded size came back identical to a
                  uniform qp=45 run, not a uniform qp=12 run -- rc:qp_init
                  (set to qp_i, since frame 0 is our I-frame) ends up
                  governing the *entire* encode, and rc:qp_max/min (the
                  intended P-frame bound) is ignored. This matches
                  Rockchip's own reference usage (MPP's mpi_enc_test.c sets
                  qp_init/qp_max/qp_min/qp_max_i/qp_min_i all to one
                  identical value for FIXQP): FIXQP appears to mean exactly
                  what it says, ONE fixed QP for the whole sequence, with no
                  per-frame-type override. Real asymmetric I/P QP would need
                  a different rc:mode (CBR/VBR with qp_ip delta), not
                  attempted here. Left in place (currently equivalent to
                  just "qp=") since it's harmless and documents a real,
                  verified dead end rather than silently re-discovering it. */
    int32_t disparity_offset;
    int cabac; /* 1=CABAC (default, denser/slower to decode), 0=CAVLC
                  (larger bitstream, no sequential-arithmetic-coding
                  dependency chain -- worth testing since I-frame CABAC
                  entropy decode is this backend's dominant cost) */

    /*
     * Persistent MPP + decoder state, created lazily on the first
     * extract() call and reused across calls at the same frame size --
     * removes ~24-32ms/call of MPP context-create + buffer-group +
     * 4-buffer-alloc, then mpp_destroy + buffer-group-put on the way out,
     * none of which has anything to do with the ASIC's actual encode
     * throughput (measured hw_encode_total is ~10-11ms regardless).
     * Recreated only if the frame size changes; freed for good in
     * rkmpp_hwenc_destroy(). Each call still gets a logically-independent
     * 2-frame I+P sequence: MPP_ENC_SET_IDR_FRAME forces frame 0 back to
     * IDR every call (rather than relying on rc:gop's own periodic-IDR
     * counting to happen to land right), frames never set MPP's eos flag
     * (that would tell MPP the whole stream just ended -- fine for a
     * fresh-per-call context, wrong for a persistent one), and
     * avcodec_flush_buffers() resets the decoder's DPB/reference state
     * before every call's 2 packets, the same way avcodec_flush_buffers
     * is normally used across a seek to a different, unrelated point in
     * a stream.
     */
    int mpp_ready;
    int cur_w, cur_h;
    MppCtx mctx;
    MppApi *mpi;
    MppBufferGroup buf_grp;
    MppBuffer frm_buf_l, frm_buf_r, pkt_buf_l, pkt_buf_r;
    AVCodecContext *dec_ctx;

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
    ctx->qp_i = 12; /* default: uniform QP, matching lavc_sw/rkmpp's own default */
    ctx->disparity_offset = cfg->disparity_offset;
    /*
     * Default flipped to CAVLC after measuring it: at qp=12 (this backend's
     * own accuracy-first default, unchanged), CABAC->CAVLC took extract()
     * from ~386ms to ~94ms -- a 4.1x speedup, bigger than at qp=45 (~99ms
     * -> ~69ms), consistent with CAVLC helping more the more coefficients
     * there are to entropy-decode. No accuracy difference observed on
     * tests/test_calibration or a synthetic known-shift pair (every block
     * came back exactly 16.0, cleaner than the qp=45 comparison's couple of
     * edge blocks). Not yet re-validated against Middlebury-scale real
     * content -- override to cabac=1 via backend_params if that turns out
     * to matter.
     */
    ctx->cabac = 0;

    if (cfg->backend_params) {
        const char *c = strstr(cfg->backend_params, "cabac=");
        if (c) ctx->cabac = atoi(c + 6);
        const char *p = strstr(cfg->backend_params, "qp=");
        if (p) { ctx->qp = atoi(p + 3); ctx->qp_i = ctx->qp; }
        const char *pi = strstr(cfg->backend_params, "qp_i=");
        if (pi) ctx->qp_i = atoi(pi + 5); /* explicit override wins over qp='s default */
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

/* Fills the whole buffer (luma padding included) with flat 128 -- see
   lavc_sw/rkmpp's identical "flat chroma" choice. Only needed once per
   buffer allocation now that frm_buf_l/r persist across calls (ensure_mpp_
   context, right after mpp_buffer_get): chroma is never touched again by
   fill_nv12 below, and stride padding beyond column w is never read by
   the encoder in a way this project's own accuracy tests have found to
   matter, so re-stamping ~1.5x the frame's worth of memory with the same
   value on every single call was pure waste. */
static int fill_flat128(MppBuffer buf, size_t frm_size) {
    uint8_t *p = (uint8_t *)mpp_buffer_get_ptr(buf);
    if (!p) return -1;
    mpp_buffer_sync_begin(buf);
    memset(p, 128, frm_size);
    mpp_buffer_sync_end(buf);
    return 0;
}

/* Per-call: overwrite just the luma rows (chroma stays flat 128 from
   fill_flat128, set once at allocation time). */
static int fill_nv12_luma(MppBuffer buf, const uint8_t *luma, int stride,
                           int w, int h, int hor_stride) {
    uint8_t *p = (uint8_t *)mpp_buffer_get_ptr(buf);
    if (!p) return -1;
    mpp_buffer_sync_begin(buf);
    for (int y = 0; y < h; y++)
        memcpy(p + (size_t)y * hor_stride, luma + (size_t)y * stride, (size_t)w);
    mpp_buffer_sync_end(buf);
    return 0;
}

static void teardown_mpp_context(rkmpp_hwenc_ctx *ctx) {
    if (ctx->dec_ctx) avcodec_free_context(&ctx->dec_ctx);
    if (ctx->frm_buf_l) { mpp_buffer_put(ctx->frm_buf_l); ctx->frm_buf_l = NULL; }
    if (ctx->frm_buf_r) { mpp_buffer_put(ctx->frm_buf_r); ctx->frm_buf_r = NULL; }
    if (ctx->pkt_buf_l) { mpp_buffer_put(ctx->pkt_buf_l); ctx->pkt_buf_l = NULL; }
    if (ctx->pkt_buf_r) { mpp_buffer_put(ctx->pkt_buf_r); ctx->pkt_buf_r = NULL; }
    if (ctx->buf_grp) { mpp_buffer_group_put(ctx->buf_grp); ctx->buf_grp = NULL; }
    if (ctx->mctx) { mpp_destroy(ctx->mctx); ctx->mctx = NULL; }
    ctx->mpi = NULL;
    ctx->mpp_ready = 0;
}

/* (Re)creates the persistent MPP encoder + decoder state for a given frame
   size, tearing down any previous instance first. No-op if already set up
   for this exact size (the common case: every call after the first). */
static int ensure_mpp_context(rkmpp_hwenc_ctx *ctx, int w, int h) {
    if (ctx->mpp_ready && ctx->cur_w == w && ctx->cur_h == h) return 0;
    teardown_mpp_context(ctx);

    int hor_stride = RK_ALIGN(w, 64);
    int ver_stride = RK_ALIGN(h, 16);
    size_t frm_size = (size_t)hor_stride * ver_stride * 3 / 2;
    size_t pkt_size = (size_t)hor_stride * ver_stride * 3; /* generous -- see
        rkmpp's bitstream-overflow lesson */

    MppEncCfg cfg = NULL;
    int ok = 0;

    if (mpp_create(&ctx->mctx, &ctx->mpi) != MPP_OK) goto done;
    if (mpp_init(ctx->mctx, MPP_CTX_ENC, MPP_VIDEO_CodingAVC) != MPP_OK) goto done;
    if (mpp_buffer_group_get_internal(&ctx->buf_grp, MPP_BUFFER_TYPE_DMA_HEAP) != MPP_OK) goto done;

    if (mpp_buffer_get(ctx->buf_grp, &ctx->frm_buf_l, frm_size) != MPP_OK) goto done;
    if (mpp_buffer_get(ctx->buf_grp, &ctx->frm_buf_r, frm_size) != MPP_OK) goto done;
    if (mpp_buffer_get(ctx->buf_grp, &ctx->pkt_buf_l, pkt_size) != MPP_OK) goto done;
    if (mpp_buffer_get(ctx->buf_grp, &ctx->pkt_buf_r, pkt_size) != MPP_OK) goto done;

    /* Once per allocation, not once per call -- see fill_flat128's comment. */
    if (fill_flat128(ctx->frm_buf_l, frm_size) != 0) goto done;
    if (fill_flat128(ctx->frm_buf_r, frm_size) != 0) goto done;

    mpp_enc_cfg_init(&cfg);
    if (ctx->mpi->control(ctx->mctx, MPP_ENC_GET_CFG, cfg) != MPP_OK) goto done;

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

    mpp_enc_cfg_set_s32(cfg, "rc:qp_init", ctx->qp_i); /* frame 0 = our I-frame */
    mpp_enc_cfg_set_s32(cfg, "rc:qp_max", ctx->qp);
    mpp_enc_cfg_set_s32(cfg, "rc:qp_min", ctx->qp);
    mpp_enc_cfg_set_s32(cfg, "rc:qp_max_i", ctx->qp_i);
    mpp_enc_cfg_set_s32(cfg, "rc:qp_min_i", ctx->qp_i);
    mpp_enc_cfg_set_s32(cfg, "rc:qp_ip", 0);

    mpp_enc_cfg_set_s32(cfg, "codec:type", MPP_VIDEO_CodingAVC);
    mpp_enc_cfg_set_s32(cfg, "h264:profile", 100);
    mpp_enc_cfg_set_s32(cfg, "h264:level", 40);
    mpp_enc_cfg_set_s32(cfg, "h264:cabac_en", ctx->cabac);

    if (ctx->mpi->control(ctx->mctx, MPP_ENC_SET_CFG, cfg) != MPP_OK) goto done;

    {
        RK_U32 header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
        if (ctx->mpi->control(ctx->mctx, MPP_ENC_SET_HEADER_MODE, &header_mode) != MPP_OK) goto done;
    }

    {
        const AVCodec *dec_codec = avcodec_find_decoder(AV_CODEC_ID_H264);
        if (!dec_codec) goto done;
        ctx->dec_ctx = avcodec_alloc_context3(dec_codec);
        if (!ctx->dec_ctx) goto done;
        ctx->dec_ctx->flags2 |= AV_CODEC_FLAG2_EXPORT_MVS;
        if (getenv("CS_RKMPP_HWENC_SKIP_RECON")) {
            ctx->dec_ctx->skip_idct = AVDISCARD_ALL;
            ctx->dec_ctx->skip_loop_filter = AVDISCARD_ALL;
        }
        if (avcodec_open2(ctx->dec_ctx, dec_codec, NULL) < 0) goto done;
    }

    ctx->cur_w = w;
    ctx->cur_h = h;
    ctx->mpp_ready = 1;
    ok = 1;

done:
    if (cfg) mpp_enc_cfg_deinit(cfg);
    if (!ok) teardown_mpp_context(ctx);
    return ok ? 0 : -1;
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

    int ret = -1;
    uint8_t *right_shifted = NULL;
    uint8_t *bitstream_i = NULL, *bitstream_p = NULL; /* av_malloc'd, padded --
        ownership transfers to pkt_i/pkt_p via av_packet_from_data below, so
        these are NOT freed directly in `done:` except on an early failure
        before that transfer happens */
    size_t bitstream_i_len = 0, bitstream_p_len = 0;

    AVPacket *pkt_i = NULL, *pkt_p = NULL;
    AVFrame *dec_frame = NULL;

    int timing = getenv("CS_RKMPP_HWENC_TIMING") != NULL;
    double t0 = timing ? now_ms() : 0;

    if (ensure_mpp_context(ctx, w, h) != 0) goto done;
    MppCtx mctx = ctx->mctx;
    MppApi *mpi = ctx->mpi;

    if (fill_nv12_luma(ctx->frm_buf_l, left->data[0], left->stride[0], w, h, hor_stride) != 0)
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
    if (fill_nv12_luma(ctx->frm_buf_r, right_luma, right_stride, w, h, hor_stride) != 0)
        goto done;

    /* Every call is logically an independent 2-frame I+P sequence even
       though the MPP context now persists -- force frame 0 back to IDR
       explicitly rather than relying on rc:gop's periodic-IDR counting to
       happen to land on a call boundary. */
    if (mpi->control(mctx, MPP_ENC_SET_IDR_FRAME, NULL) != MPP_OK) goto done;

    double t_setup_done = timing ? now_ms() : 0;

    MppBuffer frm_bufs[2] = {ctx->frm_buf_l, ctx->frm_buf_r};
    MppBuffer pkt_bufs[2] = {ctx->pkt_buf_l, ctx->pkt_buf_r};
    for (int i = 0; i < 2; i++) {
        double t_frame_start = timing ? now_ms() : 0;
        MppFrame frame = NULL;
        MppPacket packet = NULL;

        mpp_frame_init(&frame);
        mpp_frame_set_width(frame, w);
        mpp_frame_set_height(frame, h);
        mpp_frame_set_hor_stride(frame, hor_stride);
        mpp_frame_set_ver_stride(frame, ver_stride);
        mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);
        /* eos deliberately never set: that tells MPP the whole stream has
           ended, which is wrong for a context meant to keep serving future
           calls (see the struct comment above). */
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
        /* av_malloc (not plain malloc) with AV_INPUT_BUFFER_PADDING_SIZE
           slack so this buffer can be handed directly to av_packet_from_data
           below with no second copy into a separately-allocated AVPacket
           buffer (av_new_packet + memcpy, as this used to do). */
        uint8_t *copy = (uint8_t *)av_malloc(len + AV_INPUT_BUFFER_PADDING_SIZE);
        if (!copy) { mpp_packet_deinit(&out_packet); goto done; }
        memcpy(copy, pos, len);
        memset(copy + len, 0, AV_INPUT_BUFFER_PADDING_SIZE);
        mpp_packet_deinit(&out_packet);

        if (i == 0) { bitstream_i = copy; bitstream_i_len = len; }
        else        { bitstream_p = copy; bitstream_p_len = len; }

        if (timing)
            fprintf(stderr, "TIMING hw_encode[%d] %.3f ms (%zu bytes)\n",
                    i, now_ms() - t_frame_start, len);
    }

    double t_encode_done = timing ? now_ms() : 0;

    /* Done with MPP for this call -- mctx/buffers persist for the next one
       (freed only in rkmpp_hwenc_destroy). Everything past here is a plain
       software H.264 decode of the hardware encoder's own bitstream,
       identical in spirit to cs_backend_lavc_sw's decode side. */
    avcodec_flush_buffers(ctx->dec_ctx); /* reset DPB/reference state from
        any previous call, same use as across a seek to an unrelated point
        in a stream */

    double t_teardown_done = timing ? now_ms() : 0; /* kept for a stable
        TIMING line shape; this phase is now ~free (flush only) */

    double t_decoder_open_done = timing ? now_ms() : 0;

    pkt_i = av_packet_alloc();
    pkt_p = av_packet_alloc();
    dec_frame = av_frame_alloc();
    if (!pkt_i || !pkt_p || !dec_frame) goto done;
    /* Takes ownership of bitstream_i/bitstream_p (av_malloc'd with padding
       above) -- no further copy, and `done:` must not free them itself
       once this succeeds. */
    if (av_packet_from_data(pkt_i, bitstream_i, (int)bitstream_i_len) < 0) goto done;
    bitstream_i = NULL;
    if (av_packet_from_data(pkt_p, bitstream_p, (int)bitstream_p_len) < 0) goto done;
    bitstream_p = NULL;
    pkt_i->pts = 0;
    pkt_p->pts = 1;

    out->subpel_bits = 0; /* overwritten below iff a P-frame with side data is decoded */

    AVPacket *dec_inputs[2] = {pkt_i, pkt_p};
    for (int i = 0; i < 2; i++) {
        double t_pkt_start = timing ? now_ms() : 0;
        if (avcodec_send_packet(ctx->dec_ctx, dec_inputs[i]) < 0) goto done;
        for (;;) {
            int r = avcodec_receive_frame(ctx->dec_ctx, dec_frame);
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
        if (timing)
            fprintf(stderr, "TIMING sw_decode[%d] %.3f ms (%zu bytes)\n",
                    i, now_ms() - t_pkt_start, (size_t)dec_inputs[i]->size);
    }

    if (timing) {
        double t_all_done = now_ms();
        fprintf(stderr,
                "TIMING mpp_setup=%.3f hw_encode_total=%.3f mpp_teardown=%.3f "
                "decoder_open=%.3f sw_decode_total=%.3f overall=%.3f (ms)\n",
                t_setup_done - t0, t_encode_done - t_setup_done,
                t_teardown_done - t_encode_done,
                t_decoder_open_done - t_teardown_done,
                t_all_done - t_decoder_open_done,
                t_all_done - t0);
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
    /* av_malloc'd (not plain malloc) -- only still non-NULL here if
       av_packet_from_data was never reached/failed, i.e. ownership never
       transferred to pkt_i/pkt_p. */
    av_free(bitstream_i);
    av_free(bitstream_p);
    if (pkt_i) av_packet_free(&pkt_i);
    if (pkt_p) av_packet_free(&pkt_p);
    if (dec_frame) av_frame_free(&dec_frame);
    /* mctx, mpi, and all MPP buffers/the decoder context persist in ctx
       for the next call -- freed only in rkmpp_hwenc_destroy(). */
    return ret;
}

static void rkmpp_hwenc_destroy(void *vctx) {
    rkmpp_hwenc_ctx *ctx = (rkmpp_hwenc_ctx *)vctx;
    if (!ctx) return;
    teardown_mpp_context(ctx);
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
