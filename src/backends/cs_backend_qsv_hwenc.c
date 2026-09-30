/*
 * cs_backend_qsv_hwenc -- Intel Quick Sync (oneVPL) hardware H.264 ENCODE,
 * software DECODE (CS_MODE_ENCODE_DECODE). Same structure as
 * cs_backend_nvenc_hwenc.c / cs_backend_rkmpp_hwenc.c: the GPU's motion
 * search runs while encoding the pair as an I+P sequence, and the P-frame's
 * motion vectors are read back from the bitstream by the shared libavcodec
 * half (cs_h264_mvdec), with a stand-in flat-gray IDR on the decode side.
 *
 * Only hardware implementations are accepted (MFX_IMPL_TYPE_HARDWARE), so
 * init() fails -- skipping the backend -- when no Intel GPU/driver provides
 * one. Input is system memory NV12 (luma from the caller, flat 128 chroma).
 *
 * Status: not yet measured on hardware -- see Docs/codec-stereo-DESIGN.md.
 *
 * backend_params keys: qp=N (default 12), tu=1..7 (target usage, 1 = best
 * quality, 7 = fastest, default 7), cabac=0|1 (default 0), lowpower=0|1|2
 * (0 = driver default, 1 = on, 2 = off), slices=N (decoder slice threads).
 * Env: CS_QSV_DEBUG (print oneVPL errors), CS_QSV_TIMING (phase timing).
 */

#include "codec_stereo/cs.h"
#include "codec_stereo/cs_util.h"
#include "codec_stereo/cs_clock.h"
#include "cs_backend.h"
#include "cs_h264_mvdec.h"

#include <vpl/mfx.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QSV_ALIGN(x, a) (((x) + (a) - 1) / (a) * (a))

typedef struct qsv_hwenc_ctx {
    int block_w, block_h;
    int qp, tu, cabac, lowpower, slices;
    int32_t disparity_offset;
    int debug;

    mfxLoader loader;
    mfxSession session;

    int ready;
    int cur_w, cur_h;
    int enc_w, enc_h;   /* image padded to a multiple of 16 */
    uint8_t *surf_y[2]; /* system-memory NV12 surfaces: luma then interleaved chroma */
    size_t surf_size;
    mfxU8 *bs_data;
    mfxU32 bs_max;

    cs_h264_mvdec *dec;

    int cols, rows;
    int16_t *dx, *dy;
    uint16_t *cost;
    uint8_t *flags;
} qsv_hwenc_ctx;

static void qsv_log(qsv_hwenc_ctx *ctx, const char *what, mfxStatus st) {
    if (ctx->debug) fprintf(stderr, "qsv_hwenc: %s failed (mfxStatus %d)\n", what, (int)st);
}

static cs_backend_caps qsv_hwenc_get_caps(void *vctx) {
    qsv_hwenc_ctx *ctx = (qsv_hwenc_ctx *)vctx;
    cs_backend_caps caps;
    memset(&caps, 0, sizeof caps);
    caps.mode = CS_MODE_ENCODE_DECODE;
    caps.native_block_w = ctx->block_w;
    caps.native_block_h = ctx->block_h;
    caps.max_search_range_x = 2048;
    caps.max_search_range_y = 2048;
    caps.mv_min_x = -2048; caps.mv_max_x = 2047;
    caps.mv_min_y = -512;  caps.mv_max_y = 511;
    caps.supports_subpel = 1;
    caps.cost_metric = CS_COST_NONE;
    return caps;
}

static void session_teardown(qsv_hwenc_ctx *ctx) {
    cs_h264_mvdec_close(ctx->dec);
    ctx->dec = NULL;
    if (ctx->session && ctx->ready) MFXVideoENCODE_Close(ctx->session);
    for (int i = 0; i < 2; i++) { free(ctx->surf_y[i]); ctx->surf_y[i] = NULL; }
    free(ctx->bs_data);
    ctx->bs_data = NULL;
    ctx->ready = 0;
}

static void device_teardown(qsv_hwenc_ctx *ctx) {
    session_teardown(ctx);
    if (ctx->session) { MFXClose(ctx->session); ctx->session = NULL; }
    if (ctx->loader) { MFXUnload(ctx->loader); ctx->loader = NULL; }
}

static int set_filter_u32(mfxConfig cfg, const char *name, mfxU32 value) {
    mfxVariant v;
    memset(&v, 0, sizeof v);
    v.Type = MFX_VARIANT_TYPE_U32;
    v.Data.U32 = value;
    return MFXSetConfigFilterProperty(cfg, (const mfxU8 *)name, v) == MFX_ERR_NONE ? 0 : -1;
}

static int qsv_hwenc_init(void *vctx, const cs_config *cfg) {
    qsv_hwenc_ctx *ctx = (qsv_hwenc_ctx *)vctx;
    int ok = 0;
    mfxConfig c;
    mfxStatus st;

    ctx->debug = getenv("CS_QSV_DEBUG") != NULL;
    ctx->block_w = cfg->block_w > 0 ? cfg->block_w : 16;
    ctx->block_h = cfg->block_h > 0 ? cfg->block_h : 16;
    ctx->disparity_offset = cfg->disparity_offset;
    ctx->qp = 12;
    ctx->tu = 7;
    ctx->cabac = 0;
    ctx->lowpower = 0;
    if (cfg->backend_params) {
        const char *p;
        if ((p = strstr(cfg->backend_params, "qp=")))       ctx->qp = atoi(p + 3);
        if ((p = strstr(cfg->backend_params, "tu=")))       ctx->tu = atoi(p + 3);
        if ((p = strstr(cfg->backend_params, "cabac=")))    ctx->cabac = atoi(p + 6);
        if ((p = strstr(cfg->backend_params, "lowpower="))) ctx->lowpower = atoi(p + 9);
        if ((p = strstr(cfg->backend_params, "slices=")))   ctx->slices = atoi(p + 7);
    }

    {
        cs_h264_mvdec *probe = cs_h264_mvdec_open(0, 1);
        if (!probe) return -1;
        cs_h264_mvdec_close(probe);
    }

    ctx->loader = MFXLoad();
    if (!ctx->loader) goto done;

    c = MFXCreateConfig(ctx->loader);
    if (!c || set_filter_u32(c, "mfxImplDescription.Impl", MFX_IMPL_TYPE_HARDWARE) != 0) goto done;
    c = MFXCreateConfig(ctx->loader);
    if (!c || set_filter_u32(c,
            "mfxImplDescription.mfxEncoderDescription.encoder.CodecID", MFX_CODEC_AVC) != 0) goto done;

    st = MFXCreateSession(ctx->loader, 0, &ctx->session);
    if (st != MFX_ERR_NONE) { qsv_log(ctx, "MFXCreateSession", st); ctx->session = NULL; goto done; }
    ok = 1;

done:
    if (!ok) device_teardown(ctx);
    return ok ? 0 : -1;
}

static int ensure_grids(qsv_hwenc_ctx *ctx, int cols, int rows) {
    if (cols == ctx->cols && rows == ctx->rows && ctx->dx) return 0;
    free(ctx->dx); free(ctx->dy); free(ctx->cost); free(ctx->flags);
    size_t n = (size_t)cols * (size_t)rows;
    ctx->dx = (int16_t *)malloc(n * sizeof(int16_t));
    ctx->dy = (int16_t *)malloc(n * sizeof(int16_t));
    ctx->cost = (uint16_t *)calloc(n, sizeof(uint16_t));
    ctx->flags = (uint8_t *)malloc(n);
    if (!ctx->dx || !ctx->dy || !ctx->cost || !ctx->flags) {
        free(ctx->dx); free(ctx->dy); free(ctx->cost); free(ctx->flags);
        ctx->dx = ctx->dy = NULL; ctx->cost = NULL; ctx->flags = NULL;
        ctx->cols = ctx->rows = 0;
        return -1;
    }
    ctx->cols = cols;
    ctx->rows = rows;
    return 0;
}

static void fill_luma(const qsv_hwenc_ctx *ctx, uint8_t *surf, const uint8_t *luma,
                       int stride, int w, int h) {
    for (int y = 0; y < ctx->enc_h; y++) {
        const uint8_t *src = luma + (size_t)(y < h ? y : h - 1) * stride;
        uint8_t *dst = surf + (size_t)y * ctx->enc_w;
        memcpy(dst, src, (size_t)w);
        if (ctx->enc_w > w) memset(dst + w, src[w - 1], (size_t)(ctx->enc_w - w));
    }
}

/* Encodes surface `idx` (forcing an IDR if requested), returning the
   bitstream as a cs_h264_bs_alloc'd padded buffer. */
static int encode_one(qsv_hwenc_ctx *ctx, int idx, int idr, uint8_t **out_bs, size_t *out_len) {
    mfxFrameSurface1 surf;
    mfxBitstream bs;
    mfxEncodeCtrl ctrl;
    mfxSyncPoint sp = NULL;
    mfxStatus st;

    memset(&surf, 0, sizeof surf);
    surf.Info.FourCC = MFX_FOURCC_NV12;
    surf.Info.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
    surf.Info.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
    surf.Info.Width = (mfxU16)ctx->enc_w;
    surf.Info.Height = (mfxU16)ctx->enc_h;
    surf.Info.CropW = (mfxU16)ctx->enc_w;
    surf.Info.CropH = (mfxU16)ctx->enc_h;
    surf.Info.FrameRateExtN = 30;
    surf.Info.FrameRateExtD = 1;
    surf.Data.Y = ctx->surf_y[idx];
    surf.Data.UV = ctx->surf_y[idx] + (size_t)ctx->enc_w * ctx->enc_h;
    surf.Data.Pitch = (mfxU16)ctx->enc_w;

    memset(&bs, 0, sizeof bs);
    bs.Data = ctx->bs_data;
    bs.MaxLength = ctx->bs_max;

    memset(&ctrl, 0, sizeof ctrl);
    if (idr) ctrl.FrameType = MFX_FRAMETYPE_I | MFX_FRAMETYPE_IDR | MFX_FRAMETYPE_REF;

    for (;;) {
        st = MFXVideoENCODE_EncodeFrameAsync(ctx->session, &ctrl, &surf, &bs, &sp);
        if (st == MFX_WRN_DEVICE_BUSY) continue; /* busy-wait; calls are short */
        break;
    }
    if (st != MFX_ERR_NONE && st != MFX_WRN_INCOMPATIBLE_VIDEO_PARAM) {
        qsv_log(ctx, "EncodeFrameAsync", st);
        return -1;
    }
    if (!sp) return -1;
    st = MFXVideoCORE_SyncOperation(ctx->session, sp, 10000);
    if (st != MFX_ERR_NONE) { qsv_log(ctx, "SyncOperation", st); return -1; }

    uint8_t *copy = cs_h264_bs_alloc(bs.DataLength);
    if (!copy) return -1;
    memcpy(copy, bs.Data + bs.DataOffset, bs.DataLength);
    *out_bs = copy;
    *out_len = bs.DataLength;
    return 0;
}

static int ensure_session(qsv_hwenc_ctx *ctx, int w, int h) {
    if (ctx->ready && ctx->cur_w == w && ctx->cur_h == h) return 0;
    session_teardown(ctx);

    int ok = 0;
    mfxVideoParam par;
    mfxExtCodingOption co;
    mfxExtBuffer *ext[1];
    uint8_t *stub = NULL;
    size_t stub_len = 0;
    mfxStatus st;

    ctx->enc_w = QSV_ALIGN(w, 16);
    ctx->enc_h = QSV_ALIGN(h, 16);

    memset(&co, 0, sizeof co);
    co.Header.BufferId = MFX_EXTBUFF_CODING_OPTION;
    co.Header.BufferSz = sizeof co;
    co.CAVLC = ctx->cabac ? MFX_CODINGOPTION_OFF : MFX_CODINGOPTION_ON;
    ext[0] = &co.Header;

    memset(&par, 0, sizeof par);
    par.AsyncDepth = 1;
    par.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY;
    par.mfx.CodecId = MFX_CODEC_AVC;
    par.mfx.CodecProfile = MFX_PROFILE_AVC_HIGH;
    par.mfx.TargetUsage = (mfxU16)ctx->tu;
    par.mfx.GopPicSize = 0;    /* every call forces its own IDR */
    par.mfx.GopRefDist = 1;    /* I/P only, no B-frames */
    par.mfx.IdrInterval = 0;
    par.mfx.NumRefFrame = 1;
    par.mfx.RateControlMethod = MFX_RATECONTROL_CQP;
    par.mfx.QPI = par.mfx.QPP = par.mfx.QPB = (mfxU16)ctx->qp;
    par.mfx.LowPower = ctx->lowpower == 1 ? MFX_CODINGOPTION_ON
                     : ctx->lowpower == 2 ? MFX_CODINGOPTION_OFF : MFX_CODINGOPTION_UNKNOWN;
    par.mfx.FrameInfo.FourCC = MFX_FOURCC_NV12;
    par.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
    par.mfx.FrameInfo.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
    par.mfx.FrameInfo.Width = (mfxU16)ctx->enc_w;
    par.mfx.FrameInfo.Height = (mfxU16)ctx->enc_h;
    par.mfx.FrameInfo.CropW = (mfxU16)ctx->enc_w;
    par.mfx.FrameInfo.CropH = (mfxU16)ctx->enc_h;
    par.mfx.FrameInfo.FrameRateExtN = 30;
    par.mfx.FrameInfo.FrameRateExtD = 1;
    par.ExtParam = ext;
    par.NumExtParam = 1;

    st = MFXVideoENCODE_Init(ctx->session, &par);
    if (st < MFX_ERR_NONE) { qsv_log(ctx, "ENCODE_Init", st); goto done; }
    ctx->ready = 1; /* from here on, ENCODE_Close is owed */

    ctx->surf_size = (size_t)ctx->enc_w * ctx->enc_h * 3 / 2;
    for (int i = 0; i < 2; i++) {
        ctx->surf_y[i] = (uint8_t *)malloc(ctx->surf_size);
        if (!ctx->surf_y[i]) goto done;
        memset(ctx->surf_y[i], 128, ctx->surf_size); /* flat chroma, once */
    }
    ctx->bs_max = (mfxU32)ctx->surf_size * 2;
    ctx->bs_data = (mfxU8 *)malloc(ctx->bs_max);
    if (!ctx->bs_data) goto done;

    ctx->dec = cs_h264_mvdec_open(ctx->slices, getenv("CS_QSV_FULL_RECON") == NULL);
    if (!ctx->dec) goto done;

    /* Stand-in IDR from the still-flat surface 0. */
    if (encode_one(ctx, 0, 1, &stub, &stub_len) != 0) goto done;
    if (cs_h264_mvdec_set_stub_idr(ctx->dec, stub, stub_len) != 0) goto done;

    ctx->cur_w = w;
    ctx->cur_h = h;
    ok = 1;

done:
    cs_h264_bs_free(stub);
    if (!ok) session_teardown(ctx);
    return ok ? 0 : -1;
}

static int qsv_hwenc_extract(void *vctx, const cs_frame *left, const cs_frame *right,
                              cs_mv_field *out) {
    qsv_hwenc_ctx *ctx = (qsv_hwenc_ctx *)vctx;
    const int w = left->width, h = left->height;
    if (right->width != w || right->height != h) return -1;

    const int cols = (w + ctx->block_w - 1) / ctx->block_w;
    const int rows = (h + ctx->block_h - 1) / ctx->block_h;
    if (ensure_grids(ctx, cols, rows) != 0) return -1;
    for (int i = 0; i < cols * rows; i++) {
        ctx->dx[i] = 0;
        ctx->dy[i] = 0;
        ctx->flags[i] = (uint8_t)(CS_BLK_INTRA | CS_BLK_NO_MATCH | CS_BLK_NO_COST);
    }

    int ret = -1;
    uint8_t *right_shifted = NULL;
    uint8_t *bs_i = NULL, *bs_p = NULL;
    size_t len_i = 0, len_p = 0;
    int subpel_bits = 0;
    int timing = getenv("CS_QSV_TIMING") != NULL;
    double t0 = timing ? cs_now_ms() : 0;

    if (ensure_session(ctx, w, h) != 0) goto done;

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
    fill_luma(ctx, ctx->surf_y[0], left->data[0], left->stride[0], w, h);
    fill_luma(ctx, ctx->surf_y[1], right_luma, right_stride, w, h);
    double t_fill = timing ? cs_now_ms() : 0;

    if (encode_one(ctx, 0, 1, &bs_i, &len_i) != 0) goto done;
    cs_h264_bs_free(bs_i);
    bs_i = NULL;
    if (encode_one(ctx, 1, 0, &bs_p, &len_p) != 0) goto done;
    double t_enc = timing ? cs_now_ms() : 0;

    {
        int dr = cs_h264_mvdec_extract(ctx->dec, bs_p, len_p, ctx->block_w, ctx->block_h,
                                        cols, rows, ctx->dx, ctx->dy, ctx->flags,
                                        &subpel_bits, timing);
        bs_p = NULL; /* consumed even on failure */
        if (dr != 0) goto done;
    }

    if (timing) {
        double t_end = cs_now_ms();
        fprintf(stderr, "TIMING qsv fill=%.3f hw_encode=%.3f (P %zu bytes) sw_decode=%.3f overall=%.3f (ms)\n",
                t_fill - t0, t_enc - t_fill, len_p, t_end - t_enc, t_end - t0);
    }

    out->dx = ctx->dx;
    out->dy = ctx->dy;
    out->cost = ctx->cost;
    out->flags = ctx->flags;
    out->block_w = ctx->block_w;
    out->block_h = ctx->block_h;
    out->cols = cols;
    out->rows = rows;
    out->subpel_bits = subpel_bits;
    out->disparity_offset = ctx->disparity_offset;
    ret = 0;

done:
    free(right_shifted);
    cs_h264_bs_free(bs_i);
    cs_h264_bs_free(bs_p);
    return ret;
}

static void qsv_hwenc_destroy(void *vctx) {
    qsv_hwenc_ctx *ctx = (qsv_hwenc_ctx *)vctx;
    if (!ctx) return;
    device_teardown(ctx);
    free(ctx->dx); free(ctx->dy); free(ctx->cost); free(ctx->flags);
    free(ctx);
}

static void *qsv_hwenc_create(void) {
    return calloc(1, sizeof(qsv_hwenc_ctx));
}

cs_backend_factory cs_backend_qsv_hwenc_factory(void) {
    cs_backend_factory f;
    memset(&f, 0, sizeof f);
    f.create = qsv_hwenc_create;
    f.mode = CS_MODE_ENCODE_DECODE;
    f.ops.name = "qsv_hwenc";
    f.ops.get_caps = qsv_hwenc_get_caps;
    f.ops.init = qsv_hwenc_init;
    f.ops.extract = qsv_hwenc_extract;
    f.ops.destroy = qsv_hwenc_destroy;
    return f;
}
