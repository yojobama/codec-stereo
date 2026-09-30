/*
 * cs_backend_nvenc_hwenc -- NVIDIA NVENC hardware H.264 ENCODE, software
 * DECODE (CS_MODE_ENCODE_DECODE). Same shape as cs_backend_rkmpp_hwenc.c: the
 * GPU's fixed-function motion search runs while encoding the pair as an I+P
 * sequence, and the P-frame's motion vectors are read back out of the
 * bitstream by the shared libavcodec half (cs_h264_mvdec), with a stand-in
 * flat-gray IDR replacing the real I-frame on the decode side.
 *
 * NVENC and CUDA are loaded at runtime through ffnvcodec's dynlink loader
 * (nvEncodeAPI64.dll / nvcuda.dll ship with the driver), so this builds with
 * MSVC and MinGW without the CUDA toolkit or any import library, and init()
 * fails cleanly -- skipping the backend -- on machines without an NVIDIA GPU.
 *
 * Status: written against nv-codec-headers 12.2 (driver >= 551). Not yet
 * measured -- see Docs/codec-stereo-DESIGN.md for results once available.
 *
 * backend_params keys: qp=N (default 12), preset=1..7 (NVENC P1..P7,
 * default 1 = fastest), cabac=0|1 (default 0, CAVLC decodes faster),
 * slices=N (decoder slice threads, default off).
 * Env: CS_NVENC_DEBUG (print NVENC errors), CS_NVENC_TIMING (phase timing).
 */

#include "codec_stereo/cs.h"
#include "codec_stereo/cs_util.h"
#include "codec_stereo/cs_clock.h"
#include "cs_backend.h"
#include "cs_h264_mvdec.h"

#include <ffnvcodec/nvEncodeAPI.h>
#include <ffnvcodec/dynlink_loader.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NV_ALIGN(x, a) (((x) + (a) - 1) / (a) * (a))

typedef struct nvenc_hwenc_ctx {
    int block_w, block_h;
    int qp, preset, cabac, slices;
    int32_t disparity_offset;
    int debug;

    NvencFunctions *nvenc_dl;
    CudaFunctions *cuda_dl;
    CUcontext cuctx;
    NV_ENCODE_API_FUNCTION_LIST fn;
    void *enc;

    /* Encode size: the image padded up to a multiple of 16 and to the
       encoder's minimum dimensions. Invariant for the life of `enc`. */
    int ready;
    int cur_w, cur_h;   /* image size this session was built for */
    int enc_w, enc_h;
    int buf_h;          /* allocated input-buffer height (chroma offset) */
    NV_ENC_INPUT_PTR in_buf[2];
    NV_ENC_OUTPUT_PTR out_buf[2];

    cs_h264_mvdec *dec;

    int cols, rows;
    int16_t *dx, *dy;
    uint16_t *cost;
    uint8_t *flags;
} nvenc_hwenc_ctx;

static void nv_log(nvenc_hwenc_ctx *ctx, const char *what, NVENCSTATUS st) {
    if (!ctx->debug) return;
    const char *msg = (ctx->fn.nvEncGetLastErrorString && ctx->enc)
                          ? ctx->fn.nvEncGetLastErrorString(ctx->enc) : "";
    fprintf(stderr, "nvenc_hwenc: %s failed (status %d) %s\n", what, (int)st,
            msg ? msg : "");
}

#define NV_CHECK(ctx, call, what) \
    do { NVENCSTATUS st_ = (call); if (st_ != NV_ENC_SUCCESS) { nv_log((ctx), (what), st_); goto done; } } while (0)

#define INIT_STEP(cond, what) \
    do { if (cond) { if (ctx->debug) fprintf(stderr, "nvenc_hwenc: init: %s failed\n", (what)); goto done; } } while (0)

static const GUID *preset_guid(int p) {
    switch (p) {
    case 2: return &NV_ENC_PRESET_P2_GUID;
    case 3: return &NV_ENC_PRESET_P3_GUID;
    case 4: return &NV_ENC_PRESET_P4_GUID;
    case 5: return &NV_ENC_PRESET_P5_GUID;
    case 6: return &NV_ENC_PRESET_P6_GUID;
    case 7: return &NV_ENC_PRESET_P7_GUID;
    default: return &NV_ENC_PRESET_P1_GUID;
    }
}

static int guid_eq(const GUID *a, const GUID *b) { return memcmp(a, b, sizeof(GUID)) == 0; }

static cs_backend_caps nvenc_hwenc_get_caps(void *vctx) {
    nvenc_hwenc_ctx *ctx = (nvenc_hwenc_ctx *)vctx;
    cs_backend_caps caps;
    memset(&caps, 0, sizeof caps);
    caps.mode = CS_MODE_ENCODE_DECODE;
    caps.native_block_w = ctx->block_w;
    caps.native_block_h = ctx->block_h;
    caps.max_search_range_x = 2048; /* NVENC search range is not configurable */
    caps.max_search_range_y = 2048;
    caps.mv_min_x = -2048; caps.mv_max_x = 2047; /* H.264 spec-level range */
    caps.mv_min_y = -512;  caps.mv_max_y = 511;
    caps.supports_subpel = 1;
    caps.cost_metric = CS_COST_NONE;
    return caps;
}

static void session_teardown(nvenc_hwenc_ctx *ctx) {
    cs_h264_mvdec_close(ctx->dec);
    ctx->dec = NULL;
    if (ctx->enc) {
        for (int i = 0; i < 2; i++) {
            if (ctx->out_buf[i]) ctx->fn.nvEncDestroyBitstreamBuffer(ctx->enc, ctx->out_buf[i]);
            if (ctx->in_buf[i]) ctx->fn.nvEncDestroyInputBuffer(ctx->enc, ctx->in_buf[i]);
            ctx->out_buf[i] = NULL;
            ctx->in_buf[i] = NULL;
        }
        ctx->fn.nvEncDestroyEncoder(ctx->enc);
        ctx->enc = NULL;
    }
    ctx->ready = 0;
}

static void device_teardown(nvenc_hwenc_ctx *ctx) {
    session_teardown(ctx);
    if (ctx->cuctx && ctx->cuda_dl) ctx->cuda_dl->cuCtxDestroy(ctx->cuctx);
    ctx->cuctx = NULL;
    if (ctx->cuda_dl) cuda_free_functions(&ctx->cuda_dl);
    if (ctx->nvenc_dl) nvenc_free_functions(&ctx->nvenc_dl);
}

static int nvenc_hwenc_init(void *vctx, const cs_config *cfg) {
    nvenc_hwenc_ctx *ctx = (nvenc_hwenc_ctx *)vctx;
    int ok = 0;
    int cuda_current = 0;
    uint32_t max_ver = 0;
    CUdevice dev;

    ctx->debug = getenv("CS_NVENC_DEBUG") != NULL;
    ctx->block_w = cfg->block_w > 0 ? cfg->block_w : 16;
    ctx->block_h = cfg->block_h > 0 ? cfg->block_h : 16;
    ctx->disparity_offset = cfg->disparity_offset;
    ctx->qp = 12;
    ctx->preset = 1;
    ctx->cabac = 0;
    if (cfg->backend_params) {
        const char *p;
        if ((p = strstr(cfg->backend_params, "qp=")))      ctx->qp = atoi(p + 3);
        if ((p = strstr(cfg->backend_params, "preset=")))  ctx->preset = atoi(p + 7);
        if ((p = strstr(cfg->backend_params, "cabac=")))   ctx->cabac = atoi(p + 6);
        if ((p = strstr(cfg->backend_params, "slices=")))  ctx->slices = atoi(p + 7);
    }

    /* Decoder availability is part of the probe. */
    {
        cs_h264_mvdec *probe = cs_h264_mvdec_open(0, 1);
        if (!probe) return -1;
        cs_h264_mvdec_close(probe);
    }

    INIT_STEP(nvenc_load_functions(&ctx->nvenc_dl, NULL) < 0, "load nvEncodeAPI64.dll");
    INIT_STEP(cuda_load_functions(&ctx->cuda_dl, NULL) < 0, "load nvcuda.dll");

    INIT_STEP(ctx->cuda_dl->cuInit(0) != CUDA_SUCCESS, "cuInit");
    INIT_STEP(ctx->cuda_dl->cuDeviceGet(&dev, 0) != CUDA_SUCCESS, "cuDeviceGet");
    INIT_STEP(ctx->cuda_dl->cuCtxCreate(&ctx->cuctx, 0, dev) != CUDA_SUCCESS, "cuCtxCreate");
    /* cuCtxCreate makes the context current on this thread; it stays so
       through the probe below, then is popped so extract() can push/pop it
       around each call from whichever thread runs it. */
    cuda_current = 1;

    INIT_STEP(ctx->nvenc_dl->NvEncodeAPIGetMaxSupportedVersion(&max_ver) != NV_ENC_SUCCESS, "GetMaxSupportedVersion");
    if (((NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION) > max_ver) {
        if (ctx->debug)
            fprintf(stderr, "nvenc_hwenc: driver NVENC API 0x%x older than headers (%d.%d)\n",
                    max_ver, NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION);
        goto done;
    }

    memset(&ctx->fn, 0, sizeof ctx->fn);
    ctx->fn.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    INIT_STEP(ctx->nvenc_dl->NvEncodeAPICreateInstance(&ctx->fn) != NV_ENC_SUCCESS, "CreateInstance");

    /* Probe: open a throwaway session and check H.264 is supported. */
    {
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sp;
        GUID guids[16];
        uint32_t n = 0, found = 0;
        memset(&sp, 0, sizeof sp);
        sp.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        sp.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        sp.device = ctx->cuctx;
        sp.apiVersion = NVENCAPI_VERSION;
        { NVENCSTATUS os = ctx->fn.nvEncOpenEncodeSessionEx(&sp, &ctx->enc); if (os != NV_ENC_SUCCESS) { nv_log(ctx, "OpenEncodeSessionEx (probe)", os); ctx->enc = NULL; goto done; } }
        if (ctx->fn.nvEncGetEncodeGUIDs(ctx->enc, guids, 16, &n) == NV_ENC_SUCCESS)
            for (uint32_t i = 0; i < n; i++)
                if (guid_eq(&guids[i], &NV_ENC_CODEC_H264_GUID)) found = 1;
        ctx->fn.nvEncDestroyEncoder(ctx->enc);
        ctx->enc = NULL;
        INIT_STEP(!found, "H.264 encode GUID not reported");
    }
    ok = 1;

done:
    if (cuda_current) {
        CUcontext popped;
        ctx->cuda_dl->cuCtxPopCurrent(&popped);
    }
    if (!ok) device_teardown(ctx);
    return ok ? 0 : -1;
}

static int ensure_grids(nvenc_hwenc_ctx *ctx, int cols, int rows) {
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

/* Writes `luma` (w x h, edge-replicated out to enc_w x enc_h) into the NV12
   input buffer. If luma is NULL the whole buffer (luma + chroma) is set to
   flat 128. Chroma is otherwise left as written at allocation (flat 128). */
static int fill_input(nvenc_hwenc_ctx *ctx, NV_ENC_INPUT_PTR buf, const uint8_t *luma,
                       int stride, int w, int h, uint32_t *pitch_out) {
    NV_ENC_LOCK_INPUT_BUFFER lk;
    memset(&lk, 0, sizeof lk);
    lk.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
    lk.inputBuffer = buf;
    if (ctx->fn.nvEncLockInputBuffer(ctx->enc, &lk) != NV_ENC_SUCCESS) return -1;

    uint8_t *base = (uint8_t *)lk.bufferDataPtr;
    uint32_t pitch = lk.pitch;
    if (!luma) {
        memset(base, 128, (size_t)pitch * ctx->buf_h * 3 / 2);
    } else {
        for (int y = 0; y < ctx->enc_h; y++) {
            const uint8_t *src = luma + (size_t)(y < h ? y : h - 1) * stride;
            uint8_t *dst = base + (size_t)y * pitch;
            memcpy(dst, src, (size_t)w);
            if (ctx->enc_w > w) memset(dst + w, src[w - 1], (size_t)(ctx->enc_w - w));
        }
    }
    ctx->fn.nvEncUnlockInputBuffer(ctx->enc, buf);
    *pitch_out = pitch;
    return 0;
}

/* Encodes input buffer `idx`, forcing an IDR if requested, and returns the
   bitstream as a cs_h264_bs_alloc'd padded buffer. */
static int encode_one(nvenc_hwenc_ctx *ctx, int idx, uint32_t pitch, int idr,
                       uint8_t **out_bs, size_t *out_len) {
    NV_ENC_PIC_PARAMS pic;
    NV_ENC_LOCK_BITSTREAM lb;
    NVENCSTATUS st;

    memset(&pic, 0, sizeof pic);
    pic.version = NV_ENC_PIC_PARAMS_VER;
    pic.inputWidth = ctx->enc_w;
    pic.inputHeight = ctx->enc_h;
    pic.inputPitch = pitch;
    pic.inputBuffer = ctx->in_buf[idx];
    pic.outputBitstream = ctx->out_buf[idx];
    pic.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
    pic.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    pic.encodePicFlags = idr ? (NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS) : 0;

    st = ctx->fn.nvEncEncodePicture(ctx->enc, &pic);
    if (st != NV_ENC_SUCCESS) { nv_log(ctx, "nvEncEncodePicture", st); return -1; }

    memset(&lb, 0, sizeof lb);
    lb.version = NV_ENC_LOCK_BITSTREAM_VER;
    lb.outputBitstream = ctx->out_buf[idx];
    st = ctx->fn.nvEncLockBitstream(ctx->enc, &lb);
    if (st != NV_ENC_SUCCESS) { nv_log(ctx, "nvEncLockBitstream", st); return -1; }

    uint8_t *copy = cs_h264_bs_alloc(lb.bitstreamSizeInBytes);
    if (copy) memcpy(copy, lb.bitstreamBufferPtr, lb.bitstreamSizeInBytes);
    *out_len = lb.bitstreamSizeInBytes;
    ctx->fn.nvEncUnlockBitstream(ctx->enc, ctx->out_buf[idx]);
    if (!copy) return -1;
    *out_bs = copy;
    return 0;
}

/* (Re)builds the encoder session for a given image size. No-op if already
   built for it (every call after the first). */
static int ensure_session(nvenc_hwenc_ctx *ctx, int w, int h) {
    if (ctx->ready && ctx->cur_w == w && ctx->cur_h == h) return 0;
    session_teardown(ctx);

    int ok = 0;
    NV_ENC_INITIALIZE_PARAMS init;
    NV_ENC_PRESET_CONFIG preset;
    NV_ENC_CONFIG cfg;
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sp;
    NV_ENC_CAPS_PARAM cp;
    int min_w = 0, min_h = 0;
    uint32_t pitch = 0;
    uint8_t *stub = NULL;
    size_t stub_len = 0;

    memset(&sp, 0, sizeof sp);
    sp.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    sp.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    sp.device = ctx->cuctx;
    sp.apiVersion = NVENCAPI_VERSION;
    NV_CHECK(ctx, ctx->fn.nvEncOpenEncodeSessionEx(&sp, &ctx->enc), "OpenEncodeSessionEx");

    memset(&cp, 0, sizeof cp);
    cp.version = NV_ENC_CAPS_PARAM_VER;
    cp.capsToQuery = NV_ENC_CAPS_WIDTH_MIN;
    ctx->fn.nvEncGetEncodeCaps(ctx->enc, NV_ENC_CODEC_H264_GUID, &cp, &min_w);
    cp.capsToQuery = NV_ENC_CAPS_HEIGHT_MIN;
    ctx->fn.nvEncGetEncodeCaps(ctx->enc, NV_ENC_CODEC_H264_GUID, &cp, &min_h);

    ctx->enc_w = NV_ALIGN(w > min_w ? w : min_w, 16);
    ctx->enc_h = NV_ALIGN(h > min_h ? h : min_h, 16);
    ctx->buf_h = NV_ALIGN(ctx->enc_h, 32);

    memset(&preset, 0, sizeof preset);
    preset.version = NV_ENC_PRESET_CONFIG_VER;
    preset.presetCfg.version = NV_ENC_CONFIG_VER;
    NV_CHECK(ctx, ctx->fn.nvEncGetEncodePresetConfigEx(ctx->enc, NV_ENC_CODEC_H264_GUID,
                 *preset_guid(ctx->preset), NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY, &preset),
             "GetEncodePresetConfigEx");
    cfg = preset.presetCfg;
    cfg.version = NV_ENC_CONFIG_VER;
    cfg.gopLength = NVENC_INFINITE_GOPLENGTH; /* every call forces its own IDR */
    cfg.frameIntervalP = 1;                   /* I/P only, no B-frames */
    cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
    cfg.rcParams.constQP.qpInterP = ctx->qp;
    cfg.rcParams.constQP.qpInterB = ctx->qp;
    cfg.rcParams.constQP.qpIntra = ctx->qp;
    cfg.rcParams.enableLookahead = 0;
    cfg.encodeCodecConfig.h264Config.idrPeriod = NVENC_INFINITE_GOPLENGTH;
    cfg.encodeCodecConfig.h264Config.maxNumRefFrames = 1;
    cfg.encodeCodecConfig.h264Config.repeatSPSPPS = 1;
    cfg.encodeCodecConfig.h264Config.entropyCodingMode =
        ctx->cabac ? NV_ENC_H264_ENTROPY_CODING_MODE_CABAC : NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;

    memset(&init, 0, sizeof init);
    init.version = NV_ENC_INITIALIZE_PARAMS_VER;
    init.encodeGUID = NV_ENC_CODEC_H264_GUID;
    init.presetGUID = *preset_guid(ctx->preset);
    init.tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
    init.encodeWidth = ctx->enc_w;
    init.encodeHeight = ctx->enc_h;
    init.darWidth = ctx->enc_w;
    init.darHeight = ctx->enc_h;
    init.maxEncodeWidth = ctx->enc_w;
    init.maxEncodeHeight = ctx->enc_h;
    init.frameRateNum = 30;
    init.frameRateDen = 1;
    init.enablePTD = 1;
    init.encodeConfig = &cfg;
    NV_CHECK(ctx, ctx->fn.nvEncInitializeEncoder(ctx->enc, &init), "InitializeEncoder");

    for (int i = 0; i < 2; i++) {
        NV_ENC_CREATE_INPUT_BUFFER ib;
        NV_ENC_CREATE_BITSTREAM_BUFFER bb;
        memset(&ib, 0, sizeof ib);
        ib.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
        ib.width = ctx->enc_w;
        ib.height = ctx->buf_h;
        ib.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
        NV_CHECK(ctx, ctx->fn.nvEncCreateInputBuffer(ctx->enc, &ib), "CreateInputBuffer");
        ctx->in_buf[i] = ib.inputBuffer;

        memset(&bb, 0, sizeof bb);
        bb.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        NV_CHECK(ctx, ctx->fn.nvEncCreateBitstreamBuffer(ctx->enc, &bb), "CreateBitstreamBuffer");
        ctx->out_buf[i] = bb.bitstreamBuffer;

        /* Flat 128 everywhere once; per-call fills only touch luma. */
        if (fill_input(ctx, ctx->in_buf[i], NULL, 0, 0, 0, &pitch) != 0) goto done;
    }

    ctx->dec = cs_h264_mvdec_open(ctx->slices, getenv("CS_NVENC_FULL_RECON") == NULL);
    if (!ctx->dec) goto done;

    /* Stand-in IDR: encode the still-flat buffer as a forced IDR (see
       cs_h264_mvdec.h and the rkmpp_hwenc struct comment for why). */
    if (encode_one(ctx, 0, pitch, 1, &stub, &stub_len) != 0) goto done;
    if (cs_h264_mvdec_set_stub_idr(ctx->dec, stub, stub_len) != 0) goto done;

    ctx->cur_w = w;
    ctx->cur_h = h;
    ctx->ready = 1;
    ok = 1;

done:
    cs_h264_bs_free(stub);
    if (!ok) session_teardown(ctx);
    return ok ? 0 : -1;
}

static int nvenc_hwenc_extract(void *vctx, const cs_frame *left, const cs_frame *right,
                                cs_mv_field *out) {
    nvenc_hwenc_ctx *ctx = (nvenc_hwenc_ctx *)vctx;
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
    int pushed = 0;
    uint8_t *right_shifted = NULL;
    uint8_t *bs_i = NULL, *bs_p = NULL;
    size_t len_i = 0, len_p = 0;
    uint32_t pitch_l = 0, pitch_r = 0;
    int subpel_bits = 0;
    int timing = getenv("CS_NVENC_TIMING") != NULL;
    double t0 = timing ? cs_now_ms() : 0;

    if (ctx->cuda_dl->cuCtxPushCurrent(ctx->cuctx) != CUDA_SUCCESS) goto done;
    pushed = 1;
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

    if (fill_input(ctx, ctx->in_buf[0], left->data[0], left->stride[0], w, h, &pitch_l) != 0) goto done;
    if (fill_input(ctx, ctx->in_buf[1], right_luma, right_stride, w, h, &pitch_r) != 0) goto done;
    double t_fill = timing ? cs_now_ms() : 0;

    /* Frame 0: real left image as a forced IDR (the reference the P-frame's
       motion search runs against; its bitstream is discarded -- the stand-in
       IDR replaces it on the decode side). Frame 1: right image as the P. */
    if (encode_one(ctx, 0, pitch_l, 1, &bs_i, &len_i) != 0) goto done;
    cs_h264_bs_free(bs_i);
    bs_i = NULL;
    if (encode_one(ctx, 1, pitch_r, 0, &bs_p, &len_p) != 0) goto done;
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
        fprintf(stderr, "TIMING nvenc fill=%.3f hw_encode=%.3f (P %zu bytes) sw_decode=%.3f overall=%.3f (ms)\n",
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
    if (pushed) {
        CUcontext popped;
        ctx->cuda_dl->cuCtxPopCurrent(&popped);
    }
    return ret;
}

static void nvenc_hwenc_destroy(void *vctx) {
    nvenc_hwenc_ctx *ctx = (nvenc_hwenc_ctx *)vctx;
    if (!ctx) return;
    if (ctx->cuda_dl && ctx->cuctx) {
        /* NVENC resources must be released with their CUDA context current,
           but the context itself must not be current when destroyed. */
        CUcontext popped;
        ctx->cuda_dl->cuCtxPushCurrent(ctx->cuctx);
        session_teardown(ctx);
        ctx->cuda_dl->cuCtxPopCurrent(&popped);
    }
    device_teardown(ctx);
    free(ctx->dx); free(ctx->dy); free(ctx->cost); free(ctx->flags);
    free(ctx);
}

static void *nvenc_hwenc_create(void) {
    return calloc(1, sizeof(nvenc_hwenc_ctx));
}

cs_backend_factory cs_backend_nvenc_hwenc_factory(void) {
    cs_backend_factory f;
    memset(&f, 0, sizeof f);
    f.create = nvenc_hwenc_create;
    f.mode = CS_MODE_ENCODE_DECODE;
    f.ops.name = "nvenc_hwenc";
    f.ops.get_caps = nvenc_hwenc_get_caps;
    f.ops.init = nvenc_hwenc_init;
    f.ops.extract = nvenc_hwenc_extract;
    f.ops.destroy = nvenc_hwenc_destroy;
    return f;
}
