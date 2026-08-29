# codec-stereo — Design Document

**Status:** Implemented v0.2 (as-built; revised after Phases 0-6)
**Author:** (fill in)
**Date:** 2026-08-29 (original draft), revised 2026-08-29 after implementation

This revision replaces speculative sections of the original draft with what
was actually built and verified. Where the draft's assumptions didn't survive
contact with real hardware or real validation data, that's called out
explicitly rather than quietly edited away -- see Sec. 12 for a concise
summary of what's confirmed vs. still open.

## 1. Summary

`codec-stereo` is a C library that computes depth maps from rectified stereo
image pairs by repurposing the motion-estimation (ME) block of hardware video
encoders, instead of running a dedicated stereo-matching algorithm (SGBM,
ML-based, etc.).

The left image is encoded as an I-frame; the right image is encoded as a
single P-frame referencing it. The encoder's motion vectors (MVs) — computed
by ASIC/optimized search hardware built to run millions of times a second —
are read back and reinterpreted as per-block horizontal disparity, from which
depth is derived via standard stereo geometry.

The goal is raw speed: leverage silicon (or highly tuned SIMD encoder code)
that already exists on the target device, in preference to spending compute
budget on a purpose-built matcher.

## 2. Goals / Non-Goals

### Goals
- Sub-frame-time depth estimation on embedded/edge hardware (rk3588, Jetson,
  x86+iGPU/NVENC) using existing encoder blocks.
- Pluggable backend architecture: software libavcodec fallback plus
  hardware-specific ME paths (NVENC, VAAPI/FEI, Rockchip MPP).
- Expose per-block confidence (SAD/SATD) where the backend provides it,
  at effectively zero extra cost.
- Configurable speed/quality tradeoff (block size, search range, subpel).

### Non-Goals
- Pixel-accurate, edge-aligned disparity maps (this is a block-granular
  method by construction).
- Occlusion-aware matching by default (optional opt-in cross-check mode
  only).
- Replacing dense stereo matchers where accuracy matters more than latency —
  this library is meant as a fast path / prior, with a documented option to
  fall back to a real matcher on low-confidence regions.
- Stereo rectification and calibration are **out of scope** for this library
  and are assumed to be handled upstream (OpenCV `stereoRectify` or
  equivalent). `codec-stereo` consumes already-rectified frames.

## 3. Non-Functional Requirements

| Requirement | Target |
|---|---|
| Latency (ME-only backend, 1080p, HW) | single-digit milliseconds |
| Latency (encode+decode fallback, HW) | one HW encode + one HW decode pass |
| Latency (software fallback) | best-effort, documented as reference-only |
| Memory | no persistent frame buffering beyond one stereo pair + MV field |
| Portability | Linux primary target; backend layer isolates platform code. Holds for the encode-based backends (`lavc_sw` anywhere libavcodec/libx264 exist, `rkmpp` on RK3588 Linux); the ME-only path is currently Windows-only (`d3d12_vme`) since Vulkan has no cross-vendor ME extension — see Sec. 5.1. |

## 4. Pipeline Overview

```
Left  image ──► "I-frame" (reference)
Right image ──► "P-frame" (single ref = left)
                     │
              encoder ME search
                     │
        per-block motion vectors (dx, dy)
                     │
      dx ≈ disparity  (after rectification, dy ≈ 0)
                     │
       depth = (fx * baseline) / disparity
```

Precondition: rectification must already have made epipolar lines horizontal.
Unrectified input causes vertical MV noise, and the encoder's spatial MV
predictor (biased toward smooth, neighbor-consistent motion for bitrate
reasons) will actively work against a raw disparity signal.

## 5. Backend Abstraction

Three extraction strategies exist depending on what the target backend/SDK
exposes (the original draft described two; implementation added a third,
distinct from both — see 5.1a).

### 5.1 ME-only (preferred, where available)
Some SDKs expose a mode that runs *only* the motion search and returns
vectors — no transform, quantization, or entropy coding, no bitstream.

- **D3D12 Video Motion Estimation** (`ID3D12VideoDevice1::
  CreateVideoMotionEstimator`, `EstimateMotion` + `ResolveMotionVectorHeap`,
  Windows 10 2004+) — confirmed to exist as a real, vendor-neutral API
  (`d3d12video.h`) and implemented as `cs_backend_d3d12_vme`. **Windows-only**
  by construction, and *not yet compiled or run* on this project's target GPU
  (RX 9060 XT) — see Sec. 12.
- **Vulkan** — evaluated and ruled out. `vulkaninfo` against the RX 9060 XT
  (driver Vulkan 1.4.341) shows `VK_KHR_video_encode_h264/h265/av1` (bitstream
  output only, no MV readback in the spec) and no optical-flow/motion-
  estimation extension of any kind. The only Vulkan ME extension that exists
  at all, `VK_NV_optical_flow`, is NVIDIA-only (needs the NVOFA hardware
  block). There is no cross-vendor Vulkan ME path today.
- **NVENC** / **Intel VAAPI/FEI** — not implemented; no NVIDIA or Intel GPU
  was available to test against in this project, so these remain deferred
  rather than attempted blind.

### 5.1a Encode + direct readback (no decode) — new since the original draft
Some ASIC encoders expose per-block motion/cost data as a side-channel
*during encode*, without needing a decode pass at all — a third mode
distinct from both 5.1 and 5.2:
- **Rockchip RK3588 VEPU580** via MPP's `KEY_MOTION_INFO`
  (`mpp_task_meta_set_buffer(task, KEY_MOTION_INFO, buf)` on the encode
  task): confirmed real and working (`cs_backend_rkmpp`), but with
  significant, undocumented hardware caveats found only by testing — see
  Sec. 12. The original draft's "unconfirmed, treat as fallback-path-only"
  caution was warranted; what's confirmed now is more nuanced than either
  "works" or "doesn't."

### 5.2 Encode + decode round-trip (universal fallback)
For backends without an ME-only or encode+readback mode:
1. Encode L as I-frame, R as single P-frame referencing it.
2. Decode the P-frame with `AV_CODEC_FLAG2_EXPORT_MVS` set.
3. Read `AV_FRAME_DATA_MOTION_VECTORS` side-data from the decoded `AVFrame`.

Implemented as `cs_backend_lavc_sw` (libx264 + software H.264 decode via
libavcodec). Confirmed working and calibrated; this is the only backend that
runs correctly on every machine used in this project (WSL Ubuntu, the Orange
Pi, and — untested but expected to work identically — Windows, since it has
no platform-specific dependency).

### 5.3 Backend interface (as implemented; widened from the original draft)

```c
typedef struct cs_mv_field {
    int16_t *dx, *dy;      /* per-block; units are 1/(1<<subpel_bits) px */
    uint16_t *cost;        /* WIDENED from uint8_t -- rkmpp's SAD field is 15-bit */
    uint8_t *flags;        /* NEW: per-block cs_blk_flags (INTRA/SKIP/NO_COST/
                               CLAMPED/NO_MATCH) -- see cs.h for the full list */
    int block_w, block_h;
    int cols, rows;
    int subpel_bits;       /* NEW: 0 = integer pixels */
    int32_t disparity_offset; /* NEW: see "disparity_offset" below */
} cs_mv_field;

typedef struct cs_backend_caps {
    cs_extract_mode mode;  /* CS_MODE_ME_ONLY / ENCODE_READBACK / ENCODE_DECODE / DIRECT */
    int native_block_w, native_block_h;
    int max_search_range_x, max_search_range_y;
    int mv_min_x, mv_max_x, mv_min_y, mv_max_y; /* representable range, not
                                                    just the configured search
                                                    range -- e.g. rkmpp's mvx
                                                    is a hard 9-bit ceiling */
    int supports_subpel;
    cs_cost_metric cost_metric; /* NONE / SAD / SATD */
} cs_backend_caps;

typedef struct cs_config {
    int block_w, block_h;
    int search_range_x, search_range_y;
    int subpel;
    int32_t disparity_offset;
    const char *backend_override;
    const char *backend_params; /* NEW: lightweight "key=value;..." escape
                                    hatch for backend-specific tuning (e.g.
                                    lavc_sw's "qp"/"me"), not a stable
                                    cross-backend API */
} cs_config;

typedef struct cs_backend_ops {
    const char *name;
    cs_backend_caps (*get_caps)(void *ctx);
    int  (*init)(void *ctx, const cs_config *cfg);
    int  (*extract)(void *ctx, const cs_frame *left, const cs_frame *right,
                     cs_mv_field *out);
    void (*destroy)(void *ctx);
} cs_backend_ops;
```

`disparity_offset` turned out to be load-bearing rather than a nicety, for a
reason the draft didn't anticipate: encoder/ASIC motion search is centered on
a predictor near zero, while real stereo disparity is one-sided and can be
large (Middlebury's Motorcycle scene needs ~135px). Backends that can't steer
their own search center (`lavc_sw`, `rkmpp`, `d3d12_vme` — none of the three
implemented backends exposes a "search around X" knob) physically pre-shift
the right image by `disparity_offset` pixels before matching (`cs_shift_gray8`),
so the *residual* motion the search needs to find is small; `disparity_offset`
is then added back in `cs_mv_field_to_disparity`. `ref_sad` achieves the same
effect directly in its own indexed search instead. This mechanism was
implemented for `ref_sad` and `d3d12_vme` from the start, but was initially
*missing* from `lavc_sw` (present in the struct, never actually applied) —
running the Middlebury harness caught this immediately as a real bug (disparity
values off by >100px), fixed by adding the same pre-shift there too.

### 5.4 Backend implementations (as-built)

| Backend | Mode | Status |
|---|---|---|
| `cs_backend_ref_sad` | `CS_MODE_DIRECT` | Working, calibrated. Brute-force SAD control/diagnostic backend — not a production path, exists to separate block-granularity error from encoder RD/predictor bias in validation. |
| `cs_backend_lavc_sw` | `CS_MODE_ENCODE_DECODE` | Working, calibrated, validated on Middlebury. libx264 (`partitions=none`, `me` configurable, default `qp=10` — see Sec. 7's QP sweep finding) + software H.264 decode with `AV_CODEC_FLAG2_EXPORT_MVS`. Universal fallback; works on any platform libavcodec/libx264 are available. |
| `cs_backend_rkmpp` | `CS_MODE_ENCODE_READBACK` | Working but **not yet correctness-validated** — real hardware data flows through `KEY_MOTION_INFO`, but with confirmed defects (Sec. 12): only the top ~half of a frame's rows get written, and real motion data only appears on even 16px-wide columns (odd columns' MV is stuck at a fixed placeholder while SAD is genuine — a real silicon/firmware defect, cross-confirmed by an independent bug report). Ships as an honestly-scoped starting point, not a finished backend. |
| `cs_backend_rkmpp_hwenc` | `CS_MODE_ENCODE_DECODE` | **Working, calibrated.** Real VEPU580 hardware encode, but decodes the resulting bitstream with FFmpeg's *software* H.264 decoder + `AV_CODEC_FLAG2_EXPORT_MVS` (the same mechanism `lavc_sw` already validated) instead of reading `KEY_MOTION_INFO` — entirely sidesteps `cs_backend_rkmpp`'s defects. Passes `tests/test_calibration`. Not a speed win, though: at matched QP the software decode pass dominates end-to-end time regardless of who encoded the bitstream, so it performs about the same as pure-software `lavc_sw` (Sec. 12/9) on this SoC. Its value is correctness, not latency. |
| `cs_backend_d3d12_vme` | `CS_MODE_ME_ONLY` | **Code-complete but never compiled or run** — this machine had no C++ toolchain, and installing one needs an admin UAC elevation this session couldn't obtain non-interactively. Windows-only by construction (see Sec. 5.1's Vulkan finding). |
| `cs_backend_nvenc` | — | Not implemented; no NVIDIA hardware available to test against. |
| `cs_backend_vaapi_fei` | — | Not implemented; no Intel hardware available, and FEI is dropped from the modern Intel media driver regardless. |
| `cs_backend_videotoolbox` | — | Not implemented; no Apple hardware available. |

Backend selection (`cs_init`): probes compiled-in backends in priority order
`CS_MODE_ME_ONLY` > `CS_MODE_ENCODE_READBACK` > `CS_MODE_ENCODE_DECODE` >
`CS_MODE_DIRECT` (the last is `ref_sad`, deliberately lowest priority — it's
a validation control, not a production path), or selects an exact backend by
name via `cfg.backend_override`.

## 6. Encoder Configuration for Speed

Since compressed output quality is irrelevant, tune purely for ME cost and
search accuracy in the region that matters:

- **GOP**: one I-frame, one P-frame per stereo pair, `gop=2`. No B-frames.
  Implemented this way in both `lavc_sw` and `rkmpp`; both create a fresh
  encoder context per stereo pair (correctness over reuse-overhead, since
  the fallback backends are explicitly non-latency-critical per Sec. 3).
- **Search range**: the draft's "narrow asymmetric window" assumption did not
  survive implementation. Neither libx264 (`merange` is one scalar, applied
  symmetrically to both axes) nor MPP's VEPU580 (fixed-function ASIC; no
  search-range config key was found in `rk_venc_cfg.h`'s string-keyed
  `MppEncCfg` API) exposes an asymmetric or steerable search window. Both
  backends instead use `disparity_offset`'s pre-shift mechanism (Sec. 5.3) to
  re-center a symmetric, fixed-radius search onto the working disparity
  range.
- **Partition/block size**: `lavc_sw` forces `partitions=none` (locks to
  H.264's 16×16 MB, matching this library's default `block_w/h`) rather than
  exposing finer partitions as a runtime knob — rasterizing variable-size
  `AVMotionVector` partitions onto an arbitrary requested grid was judged
  more complexity than the accuracy gain was worth for a first
  implementation. `rkmpp`'s native granularity turned out to be forced by
  hardware anyway (32×16, not 16×16 — Sec. 12).
- **MV precision**: quarter-pel is the default and only precision level
  `lavc_sw` and `d3d12_vme` were implemented against (H.264's native
  bitstream MV precision, and the only `D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_
  PRECISION_*` enumerator confirmed from Microsoft's own published sample —
  see Sec. 12 for why nothing else was assumed there).
- **Pixel format**: NV12 for `rkmpp` and `d3d12_vme` (matches Sec. 6's
  "native format" goal); `lavc_sw` uses `AV_PIX_FMT_YUV420P` with flat
  (128) chroma for simplicity, since libavcodec doesn't need chroma at all
  for the ME result and this project's frames are luma-only anyway.
- **Rate control**: fixed QP (`MPP_ENC_RC_MODE_FIXQP` / libx264 `qp=`), no
  rate control. Default QP is 10 for `lavc_sw` — see Sec. 7's QP-sweep
  finding for why.

## 7. MV → Disparity → Depth Conversion

```
disparity[u,v] = dx[block(u,v)] / (1 << subpel_bits) + disparity_offset
depth[u,v]     = (fx * baseline) / disparity[u,v]
```

Notes:
- MV sign/direction convention differs per codec/backend, exactly as the
  draft anticipated — but working through it concretely surfaced a specific,
  reusable derivation: `AVMotionVector` documents `src_x = dst_x +
  motion_x/motion_scale`, with `dst` = the current/P frame (right image) and
  `src` = the reference (left image); this library's convention is
  "`dx` such that a LEFT point at `x` is visible in RIGHT at `x+dx`", giving
  `dx = -motion_x`. The same derivation, applied to D3D12's `EstimateMotion`
  (assumed to follow the same current→reference convention as H.264 — see
  Sec. 12), gives the same negation in `d3d12_vme`.
- **This convention is the mirror of Middlebury's own** (`disparity = x_left
  - x_right`, positive). The validation harness (Sec. 9) initially got this
  backwards and searched entirely the wrong side of the image; caught
  immediately from density collapsing (real disparity ≠ 0 almost everywhere,
  yet nearly every block came back invalid). The fix wasn't a sign flip in
  the library — it's a caller-side concern, exactly as documented in
  `cs_mv_field`'s own comment ("verify against a known-shift pair and negate
  downstream if your convention differs").
- **`min_disparity`'s near-zero gate assumes a positive-valid convention**
  (near-zero-or-negative = invalid/at-infinity). This is fine for normal use,
  but it is *not* automatically compatible with a pass whose genuine, valid
  answers land on the negative side of zero — which happens on purpose for
  the R→L pass of the cross-check extra (Sec. 8) and came up again for
  Middlebury's own coordinate convention (Sec. 9). Both call sites disable
  the gate (`min_disparity = -1e6f`) for the pass that needs it and rely on
  the flags / cross-check consistency test for validity instead. This
  wasn't anticipated by the original draft and is worth remembering before
  wiring up a new caller.
- Per-block SAD/SATD cost, when available, is used directly as a confidence
  signal — see Sec. 9's sparsification-AUC result for how well that actually
  worked out on real data.
- Upsampling block-granular disparity to per-pixel is implemented
  (`cs_disparity_upsample`, nearest and bilinear) but stays off by default,
  as planned — it's a caller-invoked opt-in, not something any backend does
  internally.

## 8. Known Limitations (by design) — plus what's now implemented

- Block-granular, not pixel-accurate. Confirmed by the Middlebury results
  (Sec. 9): both `ref_sad` and `lavc_sw` trail `cv::StereoSGBM` substantially
  on bad-2.0%, exactly as a WTA (winner-take-all) per-block matcher with no
  smoothness prior would be expected to on a scene with a large disparity
  range and fine detail.
- Encoder MV predictors bias toward spatial smoothness. Confirmed directly:
  the QP sweep (Sec. 7) found `lavc_sw`'s RMSE essentially flat across QP
  5-26, with QP=0 (near-zero RD lambda) actually *worse* overall despite a
  better RMSE on the blocks it did code inter — because density collapsed
  to 12% (most blocks fell back to intra once the RD tradeoff no longer
  favored cheap skip-coding). The predictor/RD bias is real, but "just lower
  the QP" is not the fix the original draft implied.
- **Occlusion handling is now implemented as the planned opt-in**:
  `cs_disparity_cross_check` takes a forward (L-anchored) and backward
  (R-anchored, extracted with left/right swapped) disparity map on the same
  grid and invalidates any forward cell with no consistent backward match
  within a search window — ~2x extraction cost as planned, wired into
  `cs_depth --cross-check`. Verified against a synthetic known-shift pair.
- Textureless regions remain ambiguous. The cost-vs-error sparsification AUC
  on Middlebury (Sec. 9) came back only modestly better than the raw
  bad-2.0 rate (46.9 vs 50.1) — SAD cost does carry *some* real signal, but
  markedly less than Sec. 7's "no extra cost-volume pass needed" framing
  implied. Worth tempering expectations here rather than treating the SAD
  channel as a strong confidence signal.

## 9. Validation Plan — results

`harness/cs_eval.cpp` (OpenCV-based) runs `ref_sad`, `lavc_sw`, and
`cv::StereoSGBM` against Middlebury 2014 scenes, fetched directly (not the
full zip bundles) by `harness/fetch_middlebury.sh`. Because this library's
output is block-granular by construction, ground truth and SGBM's per-pixel
output are both aggregated to the block grid (averaging valid pixels per
block) for a fair cross-backend comparison; SGBM's native full-resolution
numbers are reported separately for context.

Motorcycle-perfect (2964×2000, ndisp=270), block 16×16, QP=10, `lavc_sw` with
`me=umh` (libx264's exhaustive `me=esa` is O(merange²) and intractable at
this disparity range):

| Backend | bad-1.0% | bad-2.0% | bad-4.0% | RMSE | density |
|---|---|---|---|---|---|
| `ref_sad` | 61.4% | 50.1% | 42.7% | 47.6 | 99.8% |
| `lavc_sw` | 73.8% | 55.9% | 40.6% | 32.4 | 50.5% |
| SGBM (block-avg) | 36.0% | 31.5% | 27.5% | 25.3 | 89.2% |
| SGBM (native, full-res) | 13.2% | 7.3% | 5.2% | 13.9 | 82.2% |

Reading these: `ref_sad`'s near-100% density but worse RMSE than `lavc_sw`
reflects the RD/predictor bias discussed in Sec. 8 cutting the other way too
— `lavc_sw` reports fewer answers, but a bigger fraction of the ones it does
report are cheap/predictor-friendly and happen to be close to correct.
Neither block backend is close to SGBM; this matches Sec. 2's Non-Goal
("replacing dense stereo matchers where accuracy matters more than latency")
rather than contradicting the design.

**Latency** (`cs_bench`, median of N `cs_extract()` calls, 1920×1080 unless
noted; every backend here creates a fresh encoder/decoder context per call --
Sec. 6):

| Backend | Machine | QP | CABAC/CAVLC | Median |
|---|---|---|---|---|
| `ref_sad` (±16×4 search, 640×480) | Ryzen 5600X | n/a | n/a | 468 ms |
| `ref_sad` (±16×4 search, 640×480) | Orange Pi (A76+A55) | n/a | n/a | 967 ms |
| `lavc_sw` | Ryzen 5600X | 10 | CABAC (old default)\* | 65 ms |
| `lavc_sw` | Orange Pi | 10 | CABAC (old default)\* | 138 ms |
| `lavc_sw` | Orange Pi | 45 | CABAC | 162 ms |
| `lavc_sw` | Orange Pi | 45 | **CAVLC (new default)** | 100 ms |
| `lavc_sw` | Orange Pi | 10 (accuracy default) | CABAC | 649-673 ms |
| `lavc_sw` | Orange Pi | 10 (accuracy default) | **CAVLC (new default)** | **137 ms** |
| `rkmpp` (buggy readback path) | Orange Pi | 32 | CABAC | 36 ms |
| `rkmpp_hwenc` | Orange Pi | 12 (accuracy default) | CABAC (old default) | 386 ms |
| `rkmpp_hwenc` | Orange Pi | 45 | CABAC | 99 ms |
| `rkmpp_hwenc` | Orange Pi | 45 | CAVLC | 69 ms |
| `rkmpp_hwenc` | Orange Pi | 12 (accuracy default) | CAVLC, fresh context/call | 94 ms |
| `rkmpp_hwenc` | Orange Pi | 12 (accuracy default) | **CAVLC, persistent context (current)** | **56 ms steady-state** (88 ms first call) |

\* These two rows are the very first measurements taken in this project,
before `lavc_sw` had any explicit CABAC/CAVLC control at all -- and they
match the *CAVLC* numbers (65.7ms / 137ms) far more closely than the CABAC
ones (319ms / 649-673ms), confirmed by directly re-running the exact same
qp=10 config on the Ryzen with cabac explicitly forced each way. The real
explanation: x264's `ultrafast` preset (used here from the start) already
defaults to CAVLC as part of its own speed tuning -- a well-known preset
characteristic, not something specific to this project. When explicit
`cabac`/`qp_i` support was added to this backend later in the same
investigation that produced the rest of this table, the new `ctx->cabac`
field defaulted to 1, silently *overriding* the preset's own CAVLC choice
and introducing the CABAC slowdown as a side effect of adding the control
knob -- which the fix earlier in this same investigation (defaulting
`cabac` to 0) then corrected back out. Net effect across the whole
session: no real regression, just a detour through a self-introduced,
self-corrected one. Labeled "CABAC (old default)" above is therefore
technically wrong for what was actually running at the time; kept as
originally recorded for a transparent history rather than silently edited.

Three things worth taking away from this table, since none of them match
what Sec. 3's non-functional requirements originally hoped for:

1. **`rkmpp`'s 36-49ms (Sec. 12) is nowhere near "single-digit milliseconds."**
   Measured and confirmed for `rkmpp_hwenc`: per-call MPP context/buffer-group
   setup was ~24ms and teardown ~8ms, versus `hw_encode_total` (the ASIC's
   own work) of only ~10-11ms -- more than half of every call was pure
   context churn unrelated to the encoder's actual throughput. Fixed by
   making the MPP context, its 4 buffers, and the software decoder context
   all persistent in `cs_backend_rkmpp_hwenc.c` (created lazily on first
   use, torn down only in `rkmpp_hwenc_destroy`, recreated only if the
   frame size changes): each call now forces its own IDR via
   `MPP_ENC_SET_IDR_FRAME` instead of relying on `rc:gop`'s periodic
   counting to land on a call boundary, never sets MPP's `eos` flag
   (which would tell it the whole stream had ended), and calls
   `avcodec_flush_buffers()` on the persistent decoder before each call's
   packets, the same way it's normally used across a seek to an unrelated
   point in a stream. Steady-state `mpp_setup` dropped from ~24ms to
   ~1.1ms and `mpp_teardown`/`decoder_open` to near-zero. Combined with a
   second small fix (the flat-128 chroma fill was re-stamping the *entire*
   frame buffer, including the luma region about to be overwritten anyway,
   on every single call -- now done once per buffer allocation instead),
   this took `rkmpp_hwenc` at its default settings from 94ms to **~56ms
   steady-state** (first call after startup still pays ~88ms, matching
   the one-time setup cost) -- **386ms -> 56ms overall, a 6.9x speedup**,
   verified stable and correct over 50 consecutive calls and unchanged
   accuracy on `tests/test_calibration`, which was extended to run every
   backend through `cs_extract` 3 times on the same context specifically
   to catch the class of bug this kind of change can introduce (stale
   GOP/reference state, an incompletely-reset decoder) that a single-call
   test structurally cannot.

   This same overhead almost certainly still applies to `cs_backend_rkmpp`
   (the direct `KEY_MOTION_INFO` path) and to `lavc_sw`, neither of which
   received this treatment: `rkmpp` because its correctness bugs are the
   bigger problem there, and `lavc_sw` because its own file header already
   documents fresh-context-per-call as a deliberate choice for a backend
   whose whole purpose is being a portable reference, not the fast path.
   A further, unimplemented lever for actual sustained *throughput* (not
   just per-call latency) would be pipelining -- overlapping pair N's
   software decode (CPU-bound) with pair N+1's hardware encode (ASIC-bound,
   a different resource) across an actual stream of pairs. `cs_extract`'s
   synchronous, one-shot signature doesn't allow that; it would need an
   async submit/poll-style API, which is a real design change, not
   something to bolt on incidentally.
2. **CABAC vs CAVLC is a real, large lever for *both* software-decode
   backends -- ~35-40% at matched QP, and up to 4.1x at `rkmpp_hwenc`'s
   accuracy-first QP=12 -- for two different reasons per backend.** A
   phase-by-phase breakdown (`CS_RKMPP_HWENC_TIMING` /
   `CS_LAVC_SW_TIMING`, both added specifically to answer this) shows:
   for `rkmpp_hwenc`, the saving is genuinely in *decode* (I-frame CABAC
   entropy decode: ~51ms -> ~22ms at qp=45) -- expected, since real
   VEPU580 silicon does the encoding and only decode is software here.
   For `lavc_sw`, the saving is almost entirely on the *encode* side
   (`encode_and_feed`: 155ms -> 89ms at qp=45; `decode_drain` barely
   moves, ~4.5ms either way) -- x264's own CABAC rate-cost estimation
   during mode decision is measurably more expensive than CAVLC's, a
   known x264 characteristic, not something specific to this project.
   Skipping IDCT/deblocking (`CS_RKMPP_HWENC_SKIP_RECON`, since decoded
   pixels are never inspected) saved under 10% by comparison -- entropy
   decode has to fully parse every coded coefficient regardless of
   whether an IDCT is later applied to them, so that lever doesn't help
   much. No accuracy difference from CAVLC was observed on
   `tests/test_calibration` or a synthetic known-shift pair (every block
   came back exactly the true disparity at `rkmpp_hwenc`'s new qp=12/CAVLC
   default -- cleaner than the qp=45 comparisons, which had a couple of
   edge-block artifacts either way). For `rkmpp_hwenc` (which didn't exist
   yet when Sec. 9's Middlebury run was done) this is genuinely
   unvalidated against real content. For `lavc_sw` it's a non-issue: the
   `x264-params` string never included a `cabac` key at all until this
   investigation added one, and `ultrafast` (used since Phase 1) already
   defaults to CAVLC as part of its own preset tuning -- so Sec. 9's
   Middlebury numbers were already produced under CAVLC the whole time,
   the new explicit default just makes that the same going forward
   instead of an accident of an unset option. **Both backends now default
   to CAVLC**; override to `cabac=1` via `backend_params` if real content
   ever shows a problem this hasn't caught.
3. **Once both are CAVLC-tuned, `rkmpp_hwenc` genuinely is faster than
   `lavc_sw`** (94ms vs 100ms+ at comparable settings) -- a real, if
   modest, win from real silicon, which the original CABAC-only
   comparison (both ~99-100ms) had obscured by leaving a big,
   backend-specific inefficiency (x264's CABAC mode-decision cost) baked
   into one side of the comparison and not the other.

**Where exactly the `rkmpp_hwenc` time goes** (`CS_RKMPP_HWENC_TIMING`,
1920×1080, qp=45, one call): of ~93ms total, HW encode is ~10ms, MPP
setup/teardown ~25ms, and software decode is **~57ms (61%)** -- and within
that decode time, the **I-frame alone is ~51ms of the 57ms**. The P-frame,
the one this backend actually needs motion vectors from, decodes in ~6ms.
The I-frame's CABAC entropy decode dominates, and that cost tracks how much
detail got *coded* (QP), not how much reconstruction work follows:
`CS_RKMPP_HWENC_SKIP_RECON` (skip IDCT + deblocking, since decoded pixels are
never inspected) saved under 10% -- entropy decode still has to fully parse
every coded coefficient to know how many bits to consume, whether or not an
IDCT is later applied to them. A genuinely separate, coarser I-frame-only QP
(`qp_i` in `cs_backend_rkmpp_hwenc.c`) was tried next and confirmed **not**
to work under `MPP_ENC_RC_MODE_FIXQP` -- `rc:qp_init` (necessarily the
I-frame's QP, since frame 0 is the I-frame) ends up governing the whole
encode, and `rc:qp_max`/`rc:qp_min` (the intended P-frame-only bound) are
ignored; this matches how Rockchip's own `mpi_enc_test.c` sets all of
`qp_init`/`qp_max`/`qp_min`/`qp_max_i`/`qp_min_i` to one identical value for
FIXQP, i.e. the mode appears to mean literally one fixed QP for the whole
sequence. Real per-frame-type QP would need a different `rc:mode` (CBR/VBR
with a `qp_ip` delta), not attempted here. The lever that *does* work,
verified on a synthetic known-shift pair with no accuracy loss (disparity
still exact within ~1px at the frame edges): just raise the shared QP.
qp=12 → ~2.2MB I-frame, ~386ms total; qp=45 → ~327KB I-frame, ~93ms total.

## 10. Resolved / Open Questions

Resolved by implementation:
- **rk3588 MPP does expose a direct encode+readback path** (`KEY_MOTION_INFO`,
  Sec. 5.1a) — not ME-only, and not the "encode+decode round-trip" the
  draft assumed either. See Sec. 12 for the caveats that came with it.
- **NVENC / VAAPI FEI** remain unresolved — not "confirmed unavailable," just
  never tested, since no NVIDIA or Intel GPU was available in this project.
- **Confidence output shape**: settled as raw cost (widened to `uint16_t`
  for rkmpp's 15-bit SAD) plus a `cs_blk_flags` bitmask, rather than a
  single normalized score — preserves backend-specific magnitude/units
  instead of hiding them behind a scale nothing here can actually calibrate
  yet.

Still open:
- rkmpp's three undocumented behaviors (Sec. 12) — needs either vendor
  engagement or substantially more hardware-level investigation than was
  feasible via black-box testing alone.
- `d3d12_vme` has never been compiled (Sec. 12) — its sign convention and
  block-size/precision assumptions are unverified.
- Default block size / search range presets are set from what was
  measurable (Sec. 9); a broader per-scene sweep (multiple Middlebury
  scenes, not just Motorcycle) would sharpen these but wasn't done given the
  time this investigation already took.

## 11. Next Steps

1. Get `cs_backend_d3d12_vme` actually compiling and running (needs a C++
   toolchain + Windows SDK on the Windows machine) and through
   `tests/test_calibration` — the sign-convention assumption in its file
   header comment needs a real answer.
2. Either get vendor documentation for MPP's `KEY_MOTION_INFO` layout, or
   commit to further black-box hardware investigation of the vertical
   half-frame and horizontal paired-column limitations (Sec. 12) — a
   vertical-strip-splitting workaround for the former is sketched but not
   implemented.
3. Broaden the Middlebury sweep past a single scene, and use that to revisit
   the default block size / search range / QP choices with more data than
   Sec. 9 currently has.
4. If NVIDIA or Intel hardware becomes available, prototype `cs_backend_nvenc`
   / `cs_backend_vaapi_fei` following the same pattern established here:
   go/no-go spike first, verify against a known-shift pair, *then* write the
   real backend.

## 12. Implementation Status Summary

A concise index into where the detail lives, for anyone picking this project
back up:

- **Working and calibrated**: `ref_sad`, `lavc_sw`, `rkmpp_hwenc`. All three
  pass `tests/test_calibration`; `ref_sad`/`lavc_sw` also have real
  Middlebury numbers (Sec. 9). `rkmpp_hwenc` encodes with real VEPU580
  hardware but decodes with FFmpeg's software H.264 decoder + MV export
  (the same mechanism `lavc_sw` validates) instead of `KEY_MOTION_INFO` --
  it exists specifically because it sidesteps `rkmpp`'s defects below. Its
  latency is a wash against `lavc_sw` (Sec. 9's latency table): the shared
  software decode pass dominates regardless of who encoded the bitstream, so
  its value is a correct fallback, not a speed win, on this SoC.
- **Working, not yet correctness-validated**: `rkmpp`. Real hardware data
  flows through `KEY_MOTION_INFO` (confirmed via
  `src/backends/spike_mpp_mdinfo.c`), but with two confirmed defects and one
  unresolved question, none of which appear in the public MPP repo or a web
  search at the time this was written (the HAL source only wires a buffer fd
  to a hardware register; the bit layout itself is ASIC-generated) — though
  the same symptoms turned up independently: **github.com/rockchip-linux/mpp
  issue #825** reports the exact same "only half the frame" and "mvy=32"
  behavior on RK3588, unresolved as of this writing.
  - Motion data is only written for the top ~half of a frame's MB-rows
    (confirmed at 3 heights: 96/192/384px, split almost exactly in half,
    every run).
  - Real motion data only appears at even 16-wide raw block columns; odd
    columns carry a fixed value (mvx=248, mvy=32) while SAD still varies
    genuinely per-block. Hand-verified at the bit level across dozens of
    blocks: every odd-column word's upper 16 bits were identical
    (`0x207c`) while the lower 16 bits tracked content — a real defect in
    how paired ME lanes commit their MV, not a stride/read bug (which
    would produce incoherent garbage, not a clean, consistent split).
    **This one is easy to falsely "un-find"**: an earlier pass in this
    project's own history concluded it "doesn't reproduce" after several
    clean-looking runs, purely because the check script was only looking
    for an unrelated pre-fill sentinel and could never have detected this
    specific pattern. Re-confirmed once checked for directly — reproduced
    on every run, three different test textures including per-pixel
    noise. If this comes up again, verify by checking for the actual
    (248, 32) values, not just "no sentinel."
  - mvx/mvy units are assumed quarter-pel (empirically closest, not
    vendor-confirmed) — but a per-pixel-noise exact-shift test's
    genuinely-computed SAD values came back low but non-zero, which a
    perfect match against unmodified noise shouldn't produce. This looks
    entangled with the column defect above (the search/predictor
    mechanism affected by it may not be finding the true best match even
    on the "trustworthy" column) rather than being purely a unit-scaling
    question.

  `cs_backend_rkmpp.c`'s file header has the full account. A real,
  incidental lesson from the same investigation: pure per-pixel noise as
  encoder test input triggered a genuine ASIC "bitstream overflow" hardware
  reset (visible in `dmesg`) on an under-sized output buffer — synthetic
  test textures for any future encoder-backend work need to stay
  compressible, or the output buffer needs generous headroom (this project
  used 3x the raw frame size).
- **Code-complete, never run**: `d3d12_vme`. No C++ toolchain was available
  on the Windows machine, and installing one needs admin elevation this
  session couldn't obtain. Its sign-convention and precision/block-size
  assumptions (file header comments) are reasoned from documentation and an
  H.264 analogy, not verified against real hardware.
- **Not attempted**: `nvenc`, `vaapi_fei`, `videotoolbox` — no matching
  hardware was available to test against.
