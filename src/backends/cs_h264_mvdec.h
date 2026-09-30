/*
 * cs_h264_mvdec -- the software-decode half shared by every
 * "hardware H.264 encode, then read motion vectors back out of the bitstream"
 * backend (rkmpp_hwenc, nvenc_hwenc, qsv_hwenc).
 *
 * The decoder never produces a pixel anyone looks at: it only parses the
 * P-frame's motion vectors (AV_CODEC_FLAG2_EXPORT_MVS). Hence the stand-in
 * IDR trick -- each call feeds the decoder a tiny pre-encoded flat-gray IDR
 * instead of the real I-frame, because a reference only has to exist, not be
 * meaningful, to parse the P-frame's syntax. See cs_backend_rkmpp_hwenc.c for
 * the measurements behind that, CAVLC, and skipping IDCT/deblocking.
 *
 * The encoder side must therefore (1) emit SPS/PPS inline with each IDR,
 * (2) produce a real IDR for `set_stub_idr` from the same encoder settings
 * (same SPS/PPS, frame_num 0), and (3) produce a P-frame referencing it.
 */
#ifndef CS_H264_MVDEC_H
#define CS_H264_MVDEC_H

#include <stddef.h>
#include <stdint.h>

typedef struct cs_h264_mvdec cs_h264_mvdec;

/* slices > 1 enables libavcodec slice threading (measured no benefit at
   1080p, see rkmpp_hwenc). skip_recon != 0 skips IDCT + loop filter.
   Returns NULL if no H.264 decoder is available. */
cs_h264_mvdec *cs_h264_mvdec_open(int slices, int skip_recon);
void cs_h264_mvdec_close(cs_h264_mvdec *d);

/* Returns a buffer of `len` bytes, zero-padded past the end as libavcodec
   requires, to be filled by the caller and handed to
   cs_h264_mvdec_extract() (which takes ownership). Free an unused one with
   cs_h264_bs_free(). */
uint8_t *cs_h264_bs_alloc(size_t len);
void cs_h264_bs_free(uint8_t *bs);

/* Stores a copy of the stand-in IDR bitstream (not consumed). */
int cs_h264_mvdec_set_stub_idr(cs_h264_mvdec *d, const uint8_t *bs, size_t len);

/*
 * Decodes stub IDR then the P-frame and writes its motion vectors into the
 * caller's cols*rows grids (dx/dy/flags are fully rewritten only where a
 * block has a vector; the caller pre-fills defaults). dx/dy use the same sign
 * convention as lavc_sw: a LEFT-image point at x is visible in RIGHT at x+dx.
 * *subpel_bits receives log2 of the MV scale (0 if no vectors were found).
 *
 * `p_bs` must come from cs_h264_bs_alloc(); ownership is transferred and it
 * is consumed even on failure. Returns 0 on success.
 */
int cs_h264_mvdec_extract(cs_h264_mvdec *d, uint8_t *p_bs, size_t p_len,
                           int block_w, int block_h, int cols, int rows,
                           int16_t *dx, int16_t *dy, uint8_t *flags,
                           int *subpel_bits, int timing);

#endif
