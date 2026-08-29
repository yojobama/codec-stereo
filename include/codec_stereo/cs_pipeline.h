#ifndef CODEC_STEREO_CS_PIPELINE_H
#define CODEC_STEREO_CS_PIPELINE_H

#include "codec_stereo/cs.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * cs_pipeline -- throughput-oriented worker pool over cs_extract().
 *
 * Why this exists: a single cs_extract() call is latency-bound by a
 * strictly sequential dependency chain -- the ASIC encodes the I-frame,
 * then the P-frame against it, then the CPU decodes the P-frame's
 * bitstream. Nothing inside one call can overlap. But *successive stereo
 * pairs are completely independent*, and the two halves of the work land
 * on different silicon: encoding on RK3588's VEPU cores (there are two:
 * fdbd0000 and fdbe0000), decoding on the CPU cores. Running several
 * pairs concurrently keeps both busy instead of leaving one idle while
 * the other works.
 *
 * Measured on an Orange Pi 5 Plus at 1920x1080 (N concurrent independent
 * contexts, rkmpp_hwenc):
 *
 *     workers=1    47 pairs/s
 *     workers=2    89 pairs/s   (1.9x)
 *     workers=3   128 pairs/s   (2.7x)
 *     workers=4   163 pairs/s   (3.5x)
 *     workers=6   194 pairs/s   (4.1x)
 *
 * It scales past the 2 encoder cores because the decode half is CPU-bound
 * and the box has 8 CPU cores, so the two bottlenecks interleave rather
 * than queueing behind each other.
 *
 * This raises THROUGHPUT, not per-pair LATENCY: any single pair still
 * takes its ~15.5ms (at 1080p) to come out the far end. Use it when
 * processing a stream of pairs; use cs_extract() directly when you have
 * one pair and want it back as soon as possible.
 *
 * Implemented in C++ (std::thread/mutex/condition_variable) behind this C
 * interface -- see src/cs_pipeline.cpp.
 */

typedef struct cs_pipeline cs_pipeline;

typedef struct cs_pipeline_result {
    /* Submission-order sequence number, starting at 0. Results are
       returned strictly in submission order regardless of which worker
       finished first. */
    uint64_t seq;
    /* 0 if this pair was processed successfully; nonzero if its
       cs_extract() failed (disparity/cols/rows are then meaningless). */
    int status;
    /* Block-grid disparity, cols*rows floats, owned by the pipeline and
       valid only until the next cs_pipeline_get() call. Invalid cells are
       CS_DISPARITY_INVALID, same as cs_mv_field_to_disparity(). */
    const float *disparity;
    int cols, rows;
    int block_w, block_h;
} cs_pipeline_result;

/*
 * Creates a pipeline of `workers` independent backend contexts (each gets
 * its own cs_init(cfg), so each gets its own encoder context and its own
 * shot at a hardware core). `workers` <= 0 selects a default. `dcfg` is
 * applied by the worker when converting each MV field to disparity.
 *
 * Returns NULL if no worker could initialize its backend.
 */
cs_pipeline *cs_pipeline_create(const cs_config *cfg,
                                 const cs_disparity_config *dcfg,
                                 int workers);

/*
 * Submits one stereo pair. Blocks while the internal queue is full (which
 * is how backpressure is applied -- submit as fast as you like, this call
 * paces you to what the workers can absorb).
 *
 * The frames' pixel data is COPIED into the pipeline before returning, so
 * the caller may reuse or free `left`/`right` immediately. That copy is
 * the price of a safe async contract; it runs on the calling thread.
 *
 * Returns 0 on success, nonzero if the pipeline is shutting down or a
 * copy could not be allocated.
 */
int cs_pipeline_submit(cs_pipeline *p, const cs_frame *left, const cs_frame *right);

/*
 * Zero-copy variant. Identical to cs_pipeline_submit() except the pixel
 * data is NOT copied: the worker reads straight out of the caller's
 * buffers.
 *
 * CONTRACT: `left` and `right` (and the pixel data they point at) must
 * stay valid and UNMODIFIED until the matching result has come back out
 * of cs_pipeline_get(). Since up to workers*2 pairs are in flight, a
 * caller streaming frames needs a rotating pool of at least that many
 * buffers -- reusing one pair of buffers for every submit will corrupt
 * in-flight work.
 *
 * Worth it at higher resolutions: the copy in cs_pipeline_submit() runs
 * on the calling thread, so it serializes against everything the workers
 * do in parallel. At 1920x1080 that copy is ~4MB/pair, which measurably
 * caps scaling (2.5x with copies at 6 workers, versus 4.7x at 640x480
 * where the same copy is ~9x smaller). Use this when you can satisfy the
 * lifetime contract; use the copying version when you'd rather not think
 * about it.
 */
int cs_pipeline_submit_borrowed(cs_pipeline *p, const cs_frame *left,
                                 const cs_frame *right);

/*
 * Retrieves the next result in submission order, blocking until it is
 * ready. `out->disparity` points into pipeline-owned memory that stays
 * valid only until the following cs_pipeline_get() call -- copy anything
 * you need to keep.
 *
 * Returns 0 on success, or nonzero when there is nothing further to
 * retrieve (every submitted pair has already been returned and the
 * pipeline is being torn down).
 */
int cs_pipeline_get(cs_pipeline *p, cs_pipeline_result *out);

/*
 * Stops the workers and frees everything. Any submitted-but-unretrieved
 * results are discarded.
 */
void cs_pipeline_destroy(cs_pipeline *p);

#ifdef __cplusplus
}
#endif

#endif /* CODEC_STEREO_CS_PIPELINE_H */
