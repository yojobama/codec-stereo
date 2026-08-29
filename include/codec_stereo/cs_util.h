#ifndef CODEC_STEREO_CS_UTIL_H
#define CODEC_STEREO_CS_UTIL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Minimal image I/O used by the CLI tools, harness, and tests. Not part of
 * the core library's dependency-free path (cs_core/cs_disparity have no I/O),
 * but kept header-only-callable and free of any third-party dependency.
 */

/* Reads a binary (P5) PGM. On success returns 0 and *data_out is malloc'd
   (w*h bytes, caller frees). On failure returns -1 and leaves outputs unset. */
int cs_pgm_read(const char *path, uint8_t **data_out, int *w_out, int *h_out);

/* Writes w*h bytes of 8-bit grayscale as a binary PGM. Returns 0 on success. */
int cs_pgm_write(const char *path, const uint8_t *data, int w, int h);

/* Reads a single-channel (Pf) PFM into a top-to-bottom, left-to-right
   row-major float buffer (PFM stores rows bottom-to-top; this function
   flips them so index [y*w+x] matches image row y). On success returns 0
   and *data_out is malloc'd (w*h floats, caller frees). */
int cs_pfm_read(const char *path, float **data_out, int *w_out, int *h_out);

/* Writes a single-channel PFM from a top-to-bottom row-major float buffer
   (flipped internally to PFM's bottom-to-top row order on write). */
int cs_pfm_write(const char *path, const float *data, int w, int h);

/*
 * Shifts an 8-bit grayscale image horizontally by shift_px (positive moves
 * scene content to the right, i.e. sampling from src column (x - shift_px)),
 * clamping at the edges (edge replication). dst must be a distinct w*h
 * buffer of stride dst_stride; src is w*h at stride src_stride.
 *
 * Used both to synthesize known-disparity test pairs and to apply
 * disparity_offset pre-shifts to the right image before matching.
 */
void cs_shift_gray8(const uint8_t *src, int src_stride,
                     uint8_t *dst, int dst_stride,
                     int w, int h, int shift_px);

#ifdef __cplusplus
}
#endif

#endif /* CODEC_STEREO_CS_UTIL_H */
