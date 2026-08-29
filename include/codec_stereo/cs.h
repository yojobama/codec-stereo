#ifndef CODEC_STEREO_CS_H
#define CODEC_STEREO_CS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Frames                                                              */
/* ------------------------------------------------------------------ */

typedef enum cs_pix_fmt {
    CS_PIX_FMT_GRAY8 = 0,   /* single-plane 8-bit luma, stride-aligned    */
    CS_PIX_FMT_NV12          /* 8-bit luma plane + interleaved U/V plane  */
} cs_pix_fmt;

typedef struct cs_frame {
    const uint8_t *data[2]; /* [0]=luma, [1]=chroma (NV12 only, may be NULL) */
    int stride[2];
    int width, height;
    cs_pix_fmt fmt;
} cs_frame;

/* ------------------------------------------------------------------ */
/* Motion vector field                                                 */
/* ------------------------------------------------------------------ */

/* How a backend obtains its motion vectors. */
typedef enum cs_extract_mode {
    CS_MODE_ME_ONLY = 0,      /* fixed-function ME with no bitstream (d3d12_vme)   */
    CS_MODE_ENCODE_READBACK,  /* encode, read MV/cost side-channel, no decode (rkmpp) */
    CS_MODE_ENCODE_DECODE,    /* encode then decode w/ MV export (lavc_sw)         */
    CS_MODE_DIRECT            /* no codec involved at all (ref_sad)                */
} cs_extract_mode;

typedef enum cs_cost_metric {
    CS_COST_NONE = 0,
    CS_COST_SAD,
    CS_COST_SATD
} cs_cost_metric;

/* Per-block status. May be OR'd together. */
typedef enum cs_blk_flags {
    CS_BLK_OK       = 0,
    CS_BLK_INTRA    = 1u << 0,  /* block coded intra -> no inter MV exists      */
    CS_BLK_SKIP     = 1u << 1,  /* MV is a copied predictor, not a real match   */
    CS_BLK_NO_COST  = 1u << 2,  /* backend does not report a cost/SAD value     */
    CS_BLK_CLAMPED  = 1u << 3,  /* MV saturated the backend's representable range */
    CS_BLK_NO_MATCH = 1u << 4   /* grid cell not covered by any coded partition  */
} cs_blk_flags;

/*
 * Per-block motion vector field, laid out in raster order (left-to-right,
 * top-to-bottom), one entry per (block_w x block_h) cell of a
 * cols x rows grid covering the frame.
 *
 * dx/dy are in units of 1 / (1 << subpel_bits) pixels. subpel_bits == 0
 * means integer-pixel precision.
 *
 * Sign convention: dx is defined such that a point visible at column x in
 * the LEFT (reference) image is visible at column x + dx in the RIGHT
 * (matching) image. Raw codec/API motion vectors are usually defined the
 * opposite way (current-block-to-reference, with "current" being whichever
 * frame the codec treats as the P-frame) -- each backend adapter is
 * responsible for normalizing to this convention so downstream code is
 * backend-agnostic. Whether positive dx means "nearer" or "farther" in a
 * given rig depends on left/right camera ordering and is not knowable by
 * this library; verify against a known-shift pair (see tests/test_calibration)
 * and negate downstream if your convention differs.
 *
 * disparity_offset is a constant horizontal pre-shift (in whole pixels)
 * applied to the right image by the caller/backend before matching, to
 * re-center an asymmetric depth range inside a codec's symmetric search
 * window / representable MV range. Final disparity is:
 *
 *     disparity = dx / (1 << subpel_bits) + disparity_offset
 */
typedef struct cs_mv_field {
    int16_t *dx, *dy;      /* cols * rows entries each; caller-owned or backend-owned,
                              see cs_backend_ops::extract documentation */
    uint16_t *cost;        /* cols * rows entries; NULL if backend never reports cost */
    uint8_t *flags;        /* cols * rows entries; cs_blk_flags bitmask, NULL => all CS_BLK_OK */
    int block_w, block_h;
    int cols, rows;
    int subpel_bits;
    int32_t disparity_offset;
} cs_mv_field;

/* ------------------------------------------------------------------ */
/* Disparity / depth                                                   */
/* ------------------------------------------------------------------ */

#define CS_DISPARITY_INVALID (-1.0f)

typedef struct cs_disparity_config {
    float fx;                 /* focal length in pixels (post-rectification) */
    float baseline;            /* meters (or any consistent unit); depth inherits it */
    float min_disparity;       /* disparities <= this (in px) are marked invalid */
    int   max_dy;              /* |dy| in px above this marks the block invalid */
    uint16_t max_cost;         /* cost above this marks the block invalid; 0 = no cost gate */
} cs_disparity_config;

/*
 * Converts a motion vector field into a disparity map, honoring flags,
 * min_disparity and max_dy. `disparity_out` must hold cols*rows floats;
 * invalid cells are set to CS_DISPARITY_INVALID.
 */
void cs_mv_field_to_disparity(const cs_mv_field *field,
                               const cs_disparity_config *cfg,
                               float *disparity_out);

/*
 * Converts a disparity map to depth using cfg->fx/baseline.
 * disparity_in and depth_out both hold n floats. CS_DISPARITY_INVALID
 * disparities produce a depth of 0.0f (also treated as invalid).
 */
void cs_disparity_to_depth(const float *disparity_in, size_t n,
                            const cs_disparity_config *cfg,
                            float *depth_out);

/* ------------------------------------------------------------------ */
/* Opt-in extras (Design Sec. 8: off by default, added cost)           */
/* ------------------------------------------------------------------ */

typedef enum cs_upsample_mode {
    CS_UPSAMPLE_NEAREST = 0,
    CS_UPSAMPLE_BILINEAR
} cs_upsample_mode;

/*
 * Upsamples a (cols x rows) block-grid disparity map to a full
 * (out_w x out_h) per-pixel map (block_w*cols and block_h*rows need not
 * exactly equal out_w/out_h; edge blocks are clamped). CS_DISPARITY_INVALID
 * cells never contribute to a bilinear blend -- a pixel blending toward an
 * invalid neighbor falls back to its nearest valid cell.
 *
 * This is a caller-invoked convenience, not something any backend does
 * internally -- upsampling is off by default (Design Sec. 7), since it is
 * added cost the library is otherwise explicitly trying to avoid.
 */
void cs_disparity_upsample(const float *disparity_grid, int cols, int rows,
                            int block_w, int block_h,
                            cs_upsample_mode mode,
                            float *pixel_out, int out_w, int out_h);

/*
 * Cross-checks a forward (left-anchored) disparity map against a backward
 * one (computed by extracting with left/right swapped -- see Design Sec. 8's
 * "R->L, cross-check" opt-in mode). Both maps must be on the same
 * (cols x rows) grid with the same block_w and the same disparity sign
 * convention (the caller is responsible for negating the backward pass's
 * output first if its backend/argument order gives the opposite sign --
 * see cs_mv_field's sign-convention note); a consistent pair satisfies
 * forward[bx] + backward[bx shifted by forward[bx]] == 0.
 *
 * For each forward cell at column bx, this searches backward columns
 * within +/-search_px (converted to a block-column range via block_w) of
 * the forward-shifted position bx*block_w + forward[bx], and invalidates
 * the forward cell in `out` unless some candidate backward cell agrees
 * (|forward[bx] + backward[candidate]| <= max_diff). Either side already
 * invalid also invalidates. `out` may alias `forward`.
 */
void cs_disparity_cross_check(const float *forward, const float *backward,
                               int cols, int rows, int block_w,
                               float max_diff, int search_px,
                               float *out);

/* ------------------------------------------------------------------ */
/* Backend abstraction                                                 */
/* ------------------------------------------------------------------ */

typedef struct cs_backend_caps {
    cs_extract_mode mode;
    int native_block_w, native_block_h;
    int max_search_range_x, max_search_range_y;
    /* Representable MV range (not just the configured search range) --
       e.g. Rockchip MPP's mvx field is a hard 9-bit signed ceiling
       regardless of how the search window is configured. */
    int mv_min_x, mv_max_x, mv_min_y, mv_max_y;
    int supports_subpel;
    cs_cost_metric cost_metric;
} cs_backend_caps;

typedef struct cs_config {
    int block_w, block_h;          /* 0 = backend default */
    int search_range_x, search_range_y;
    int subpel;                    /* request subpel precision where available */
    int32_t disparity_offset;      /* pre-shift applied to the right image */
    const char *backend_override;  /* NULL = auto-probe; else exact backend name */
    /*
     * Optional "key=value;key2=value2" backend-specific tuning overrides
     * (e.g. "qp=18;me=umh" for lavc_sw). NULL = backend defaults. Unlike
     * the fields above, keys are backend-specific and undocumented here by
     * design -- see each backend's source for what it recognizes. This is
     * a deliberately lightweight escape hatch for the Phase 3/6 tuning
     * sweep, not a stable cross-backend API.
     */
    const char *backend_params;
} cs_config;

typedef struct cs_backend_ops {
    const char *name;
    cs_backend_caps (*get_caps)(void *ctx);
    int  (*init)(void *ctx, const cs_config *cfg);
    /*
     * Runs extraction for one stereo pair. On success, populates *out and
     * returns 0. The backend owns the buffers referenced by *out until the
     * next call to extract() or to destroy() -- callers must not free them
     * and must not retain pointers past that point.
     */
    int  (*extract)(void *ctx, const cs_frame *left, const cs_frame *right,
                     cs_mv_field *out);
    void (*destroy)(void *ctx);
} cs_backend_ops;

/* Opaque handle returned by cs_init(). */
typedef struct cs_context cs_context;

/*
 * Probes available backends in priority order (ME_ONLY > ENCODE_READBACK >
 * ENCODE_DECODE), or selects cfg->backend_override by exact name if set.
 * Returns NULL on failure (no backend available / named backend not found).
 */
cs_context *cs_init(const cs_config *cfg);

/* Returns the caps of the backend selected by cs_init(). */
cs_backend_caps cs_get_caps(const cs_context *ctx);

/* Returns the backend's name, e.g. "lavc_sw", "rkmpp", "ref_sad", "d3d12_vme". */
const char *cs_get_backend_name(const cs_context *ctx);

/*
 * Runs extraction; see cs_backend_ops::extract for ownership rules on the
 * buffers referenced by *out.
 */
int cs_extract(cs_context *ctx, const cs_frame *left, const cs_frame *right,
               cs_mv_field *out);

void cs_destroy(cs_context *ctx);

/* ------------------------------------------------------------------ */
/* Backend registry (internal use, exposed for tools/tests/benchmarks) */
/* ------------------------------------------------------------------ */

/* Returns NULL-terminated array of statically registered backends. */
const cs_backend_ops *const *cs_list_backends(void);

#ifdef __cplusplus
}
#endif

#endif /* CODEC_STEREO_CS_H */
