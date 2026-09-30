#include "cs_h264_mvdec.h"

#include "codec_stereo/cs.h"
#include "codec_stereo/cs_clock.h"

#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libavutil/motion_vector.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct cs_h264_mvdec {
    AVCodecContext *ctx;
    uint8_t *stub_idr;
    size_t stub_idr_len;
};

static int log2_pow2(int v, int fallback) {
    if (v <= 0) return fallback;
    int r = 0;
    while ((1 << r) < v && r < 16) r++;
    return (1 << r) == v ? r : fallback;
}

cs_h264_mvdec *cs_h264_mvdec_open(int slices, int skip_recon) {
    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) return NULL;

    cs_h264_mvdec *d = (cs_h264_mvdec *)calloc(1, sizeof *d);
    if (!d) return NULL;
    d->ctx = avcodec_alloc_context3(codec);
    if (!d->ctx) { free(d); return NULL; }

    d->ctx->flags2 |= AV_CODEC_FLAG2_EXPORT_MVS;
    if (slices > 1) {
        /* Slice threading only: frame threading is useless here (the P
           references the stand-in IDR) and would add output latency. */
        d->ctx->thread_type = FF_THREAD_SLICE;
        d->ctx->thread_count = slices;
    }
    if (skip_recon) {
        d->ctx->skip_idct = AVDISCARD_ALL;
        d->ctx->skip_loop_filter = AVDISCARD_ALL;
    }
    if (avcodec_open2(d->ctx, codec, NULL) < 0) {
        avcodec_free_context(&d->ctx);
        free(d);
        return NULL;
    }
    return d;
}

void cs_h264_mvdec_close(cs_h264_mvdec *d) {
    if (!d) return;
    av_freep(&d->stub_idr);
    if (d->ctx) avcodec_free_context(&d->ctx);
    free(d);
}

uint8_t *cs_h264_bs_alloc(size_t len) {
    uint8_t *p = (uint8_t *)av_malloc(len + AV_INPUT_BUFFER_PADDING_SIZE);
    if (p) memset(p + len, 0, AV_INPUT_BUFFER_PADDING_SIZE);
    return p;
}

void cs_h264_bs_free(uint8_t *bs) { av_free(bs); }

int cs_h264_mvdec_set_stub_idr(cs_h264_mvdec *d, const uint8_t *bs, size_t len) {
    uint8_t *copy = cs_h264_bs_alloc(len);
    if (!copy) return -1;
    memcpy(copy, bs, len);
    av_free(d->stub_idr);
    d->stub_idr = copy;
    d->stub_idr_len = len;
    return 0;
}

int cs_h264_mvdec_extract(cs_h264_mvdec *d, uint8_t *p_bs, size_t p_len,
                           int block_w, int block_h, int cols, int rows,
                           int16_t *dx, int16_t *dy, uint8_t *flags,
                           int *subpel_bits, int timing) {
    int ret = -1;
    AVPacket *pkt_i = av_packet_alloc();
    AVPacket *pkt_p = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();

    *subpel_bits = 0;
    if (!pkt_i || !pkt_p || !frame || !d->stub_idr) goto done;

    /* Reset DPB/reference state from any previous call, the same way
       avcodec_flush_buffers is used across a seek to an unrelated point. */
    avcodec_flush_buffers(d->ctx);

    /* The stub is a couple of KB, so copying it per call is noise. */
    if (av_new_packet(pkt_i, (int)d->stub_idr_len) < 0) goto done;
    memcpy(pkt_i->data, d->stub_idr, d->stub_idr_len);

    /* Takes ownership of p_bs (padded, av_malloc'd). */
    if (av_packet_from_data(pkt_p, p_bs, (int)p_len) < 0) goto done;
    p_bs = NULL;
    pkt_i->pts = 0;
    pkt_p->pts = 1;

    AVPacket *inputs[2] = {pkt_i, pkt_p};
    for (int i = 0; i < 2; i++) {
        double t_start = timing ? cs_now_ms() : 0;
        if (avcodec_send_packet(d->ctx, inputs[i]) < 0) goto done;
        for (;;) {
            int r = avcodec_receive_frame(d->ctx, frame);
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
            if (r < 0) goto done;

            if (frame->pts == 1) {
                AVFrameSideData *sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
                if (sd) {
                    const AVMotionVector *mvs = (const AVMotionVector *)sd->data;
                    int count = (int)(sd->size / sizeof(AVMotionVector));
                    int bits = 0, bits_known = 0;

                    for (int j = 0; j < count; j++) {
                        const AVMotionVector *mv = &mvs[j];
                        if (!bits_known && mv->motion_scale > 0) {
                            bits = log2_pow2(mv->motion_scale, 2);
                            bits_known = 1;
                        }

                        int gx = mv->dst_x / block_w;
                        int gy = mv->dst_y / block_h;
                        if (gx < 0 || gx >= cols || gy < 0 || gy >= rows) continue;

                        size_t idx = (size_t)gy * cols + gx;
                        /* dst = RIGHT (current/P), src = LEFT (reference):
                           src_x = dst_x + motion_x/scale, so a LEFT point at
                           x is visible in RIGHT at x - motion_x. */
                        dx[idx] = (int16_t)(-mv->motion_x);
                        dy[idx] = (int16_t)(-mv->motion_y);
                        flags[idx] = (uint8_t)CS_BLK_NO_COST;
                    }
                    *subpel_bits = bits;
                }
            }
            av_frame_unref(frame);
        }
        if (timing)
            fprintf(stderr, "TIMING sw_decode[%d] %.3f ms (%zu bytes)\n",
                    i, cs_now_ms() - t_start, (size_t)inputs[i]->size);
    }
    ret = 0;

done:
    if (p_bs) av_free(p_bs);
    if (pkt_i) av_packet_free(&pkt_i);
    if (pkt_p) av_packet_free(&pkt_p);
    if (frame) av_frame_free(&frame);
    return ret;
}
