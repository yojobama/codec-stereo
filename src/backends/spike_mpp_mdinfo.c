/*
 * Go/no-go spike for the rkmpp backend (Docs/codec-stereo-DESIGN.md,
 * Phase 4 "go/no-go spike before writing the backend"). Not part of the
 * library -- a standalone program to verify, on real rk3588 hardware,
 * that:
 *   1. KEY_MOTION_INFO actually gets populated with plausible data,
 *   2. the mvx/mvy units in MppEncMDBlkInfo (integer px? quarter-pel?),
 *   3. the SAD field is in the expected ballpark for a known-good match.
 * Everything learned here (units, whether search range needs the same
 * disparity_offset pre-shift trick as lavc_sw) feeds directly into
 * src/backends/cs_backend_rkmpp.c.
 *
 * Not wired into CMakeLists.txt; build directly:
 *   gcc -O2 -o spike_mpp_mdinfo spike_mpp_mdinfo.c -lrockchip_mpp
 */
#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/mpp_meta.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/rk_venc_cfg.h>
#include <rockchip/rk_venc_cmd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 320
#define H 192
#define SHIFT_PX 16
#define QP 12

#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

/* Per-pixel white noise is unambiguous for block matching but is also the
   worst case for a real encoder's rate/buffer assumptions -- fully
   incompressible, and hit a genuine ASIC bitstream-overflow error on first
   try (see dmesg: "mpp_rkvenc2 ... found bitstream overflow"). A flat-4x4-
   block version fixed that but introduced its own problem: exact
   periodicity every 4px gives many near-tied candidate matches, confusing
   MV verification. Generate noise on a coarse (w/CELL, h/CELL) grid and
   bilinearly upsample: smooth, non-periodic, DCT-friendly (like a real
   photo), while still locally unique enough for unambiguous 16x16 block
   matching. */
#define CELL 8
static void fill_textured(uint8_t *buf, int w, int h) {
    int gw = w / CELL + 2, gh = h / CELL + 2;
    float *grid = (float *)malloc((size_t)gw * gh * sizeof(float));
    uint32_t state = 0x9e3779b9u;
    for (int i = 0; i < gw * gh; i++) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        grid[i] = (float)(state & 0xFF);
    }
    for (int y = 0; y < h; y++) {
        float gy = (float)y / CELL;
        int gy0 = (int)gy;
        float fy = gy - gy0;
        for (int x = 0; x < w; x++) {
            float gx = (float)x / CELL;
            int gx0 = (int)gx;
            float fx = gx - gx0;
            float v00 = grid[gy0 * gw + gx0];
            float v01 = grid[gy0 * gw + gx0 + 1];
            float v10 = grid[(gy0 + 1) * gw + gx0];
            float v11 = grid[(gy0 + 1) * gw + gx0 + 1];
            float v0 = v00 + (v01 - v00) * fx;
            float v1 = v10 + (v11 - v10) * fx;
            float v = v0 + (v1 - v0) * fy;
            buf[y * w + x] = (uint8_t)v;
        }
    }
    free(grid);
}

static void shift_gray8(const uint8_t *src, uint8_t *dst, int w, int h, int shift_px) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int sx = x - shift_px;
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            dst[y * w + x] = src[y * w + sx];
        }
    }
}

int main(void) {
    MPP_RET ret;
    MppCtx ctx = NULL;
    MppApi *mpi = NULL;
    MppBufferGroup buf_grp = NULL;
    MppEncCfg cfg = NULL;

    int hor_stride = MPP_ALIGN(W, 64);
    int ver_stride = MPP_ALIGN(H, 16);
    size_t frm_size = (size_t)hor_stride * ver_stride * 3 / 2; /* NV12 */
    size_t pkt_size = (size_t)hor_stride * ver_stride * 3;     /* generous safety margin */
    size_t mdinfo_size = (size_t)(MPP_ALIGN(hor_stride, 64) >> 6) *
                         (size_t)(MPP_ALIGN(ver_stride, 16) >> 4) * 16;

    printf("hor_stride=%d ver_stride=%d mdinfo_size=%zu (mb_cols=%d mb_rows=%d)\n",
           hor_stride, ver_stride, mdinfo_size,
           (MPP_ALIGN(hor_stride, 64) >> 6) * 4, MPP_ALIGN(ver_stride, 16) >> 4);

    ret = mpp_create(&ctx, &mpi);
    if (ret) { fprintf(stderr, "mpp_create failed %d\n", ret); return 1; }

    ret = mpp_init(ctx, MPP_CTX_ENC, MPP_VIDEO_CodingAVC);
    if (ret) { fprintf(stderr, "mpp_init failed %d\n", ret); return 1; }

    ret = mpp_buffer_group_get_internal(&buf_grp, MPP_BUFFER_TYPE_DMA_HEAP);
    if (ret) { fprintf(stderr, "buffer_group_get failed %d\n", ret); return 1; }

    MppBuffer frm_buf_l = NULL, frm_buf_r = NULL, pkt_buf = NULL, md_info = NULL;
    mpp_buffer_get(buf_grp, &frm_buf_l, frm_size);
    mpp_buffer_get(buf_grp, &frm_buf_r, frm_size);
    mpp_buffer_get(buf_grp, &pkt_buf, pkt_size);
    mpp_buffer_get(buf_grp, &md_info, mdinfo_size);
    if (!frm_buf_l || !frm_buf_r || !pkt_buf || !md_info) {
        fprintf(stderr, "buffer alloc failed\n");
        return 1;
    }
    /* Sentinel-fill so untouched cells are obviously distinguishable from
       real (possibly all-zero) hardware output. */
    mpp_buffer_sync_begin(md_info);
    memset(mpp_buffer_get_ptr(md_info), 0xAA, mdinfo_size);
    mpp_buffer_sync_end(md_info);

    /* NV12: fill Y plane with texture, flat 128 chroma, then build right
       as a known +SHIFT_PX horizontal shift of left. */
    {
        uint8_t *y_l = (uint8_t *)malloc((size_t)W * H);
        uint8_t *y_r = (uint8_t *)malloc((size_t)W * H);
        fill_textured(y_l, W, H);
        shift_gray8(y_l, y_r, W, H, SHIFT_PX);

        uint8_t *pl = (uint8_t *)mpp_buffer_get_ptr(frm_buf_l);
        uint8_t *pr = (uint8_t *)mpp_buffer_get_ptr(frm_buf_r);
        mpp_buffer_sync_begin(frm_buf_l);
        mpp_buffer_sync_begin(frm_buf_r);
        memset(pl, 128, frm_size);
        memset(pr, 128, frm_size);
        for (int y = 0; y < H; y++) {
            memcpy(pl + (size_t)y * hor_stride, y_l + (size_t)y * W, W);
            memcpy(pr + (size_t)y * hor_stride, y_r + (size_t)y * W, W);
        }
        mpp_buffer_sync_end(frm_buf_l);
        mpp_buffer_sync_end(frm_buf_r);
        free(y_l);
        free(y_r);
    }

    mpp_enc_cfg_init(&cfg);
    ret = mpi->control(ctx, MPP_ENC_GET_CFG, cfg);
    printf("GET_CFG ret=%d\n", ret);
    if (ret) return 1;

    mpp_enc_cfg_set_s32(cfg, "prep:width", W);
    mpp_enc_cfg_set_s32(cfg, "prep:height", H);
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

    mpp_enc_cfg_set_s32(cfg, "rc:qp_init", QP);
    mpp_enc_cfg_set_s32(cfg, "rc:qp_max", QP);
    mpp_enc_cfg_set_s32(cfg, "rc:qp_min", QP);
    mpp_enc_cfg_set_s32(cfg, "rc:qp_max_i", QP);
    mpp_enc_cfg_set_s32(cfg, "rc:qp_min_i", QP);
    mpp_enc_cfg_set_s32(cfg, "rc:qp_ip", 0);

    mpp_enc_cfg_set_s32(cfg, "codec:type", MPP_VIDEO_CodingAVC);
    mpp_enc_cfg_set_s32(cfg, "h264:profile", 100);
    mpp_enc_cfg_set_s32(cfg, "h264:level", 40);
    mpp_enc_cfg_set_s32(cfg, "h264:cabac_en", 1);
    mpp_enc_cfg_set_s32(cfg, "h264:trans8x8", 0); /* rule out 8x8-transform/partition interaction with MD layout */

    ret = mpi->control(ctx, MPP_ENC_SET_CFG, cfg);
    printf("SET_CFG ret=%d\n", ret);
    if (ret) return 1;

    int mb_cols = (MPP_ALIGN(hor_stride, 64) >> 6) * 4; /* padded raster width */
    int real_cols = (W + 15) / 16;
    int real_rows = (H + 15) / 16;

    MppBuffer frm_bufs[2] = {frm_buf_l, frm_buf_r};
    for (int i = 0; i < 2; i++) {
        MppFrame frame = NULL;
        MppPacket packet = NULL;
        MppMeta meta = NULL;

        mpp_frame_init(&frame);
        mpp_frame_set_width(frame, W);
        mpp_frame_set_height(frame, H);
        mpp_frame_set_hor_stride(frame, hor_stride);
        mpp_frame_set_ver_stride(frame, ver_stride);
        mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);
        mpp_frame_set_eos(frame, i == 1 ? 1 : 0);
        mpp_frame_set_buffer(frame, frm_bufs[i]);

        mpp_packet_init_with_buffer(&packet, pkt_buf);
        mpp_packet_set_length(packet, 0);

        meta = mpp_frame_get_meta(frame);
        mpp_meta_set_packet(meta, KEY_OUTPUT_PACKET, packet);
        mpp_meta_set_buffer(meta, KEY_MOTION_INFO, md_info);

        ret = mpi->encode_put_frame(ctx, frame);
        mpp_frame_deinit(&frame);
        if (ret) { fprintf(stderr, "encode_put_frame[%d] failed %d\n", i, ret); return 1; }

        MppPacket out_packet = NULL;
        ret = mpi->encode_get_packet(ctx, &out_packet);
        printf("frame %d: put_frame_ret=%d get_packet_ret=%d out_packet=%p len=%zu eos=%d\n",
               i, ret, ret, (void *)out_packet,
               out_packet ? mpp_packet_get_length(out_packet) : (size_t)0,
               out_packet ? mpp_packet_get_eos(out_packet) : -1);
        if (ret || !out_packet) return 1;
        mpp_packet_deinit(&out_packet);

        /* Dump raw uint32 entries -- NOT via the bitfield struct, per the
           plan's note that MppEncMDBlkInfo's mixed-width bitfields are
           implementation-defined to read that way; mask/shift explicitly. */
        mpp_buffer_sync_begin(md_info);
        {
            const uint32_t *md = (const uint32_t *)mpp_buffer_get_ptr(md_info);
            printf("--- md_info after frame %d ---\n", i);
            printf("row  col  sad    mvx  mvy\n");
            for (int r = 0; r < real_rows; r++) {
                for (int c = 0; c < real_cols; c++) {
                    uint32_t w32 = md[r * mb_cols + c];
                    uint32_t sad = w32 & 0x7FFF;
                    int32_t mvx = (int32_t)(w32 << 8) >> 23; /* bits 15..23, sign-extended */
                    int32_t mvy = (int32_t)w32 >> 24;         /* bits 24..31, sign-extended */
                    printf("%3d  %3d  %08x  %5u  %4d %4d\n", r, c, w32, sad, mvx, mvy);
                }
            }
        }
        mpp_buffer_sync_end(md_info);
    }

    mpp_enc_cfg_deinit(cfg);
    mpp_buffer_put(frm_buf_l);
    mpp_buffer_put(frm_buf_r);
    mpp_buffer_put(pkt_buf);
    mpp_buffer_put(md_info);
    mpp_buffer_group_put(buf_grp);
    mpp_destroy(ctx);
    return 0;
}
