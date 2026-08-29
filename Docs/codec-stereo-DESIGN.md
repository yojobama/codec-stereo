# codec-stereo — Design Document

**Status:** Draft v0.1
**Author:** (fill in)
**Date:** 2026-08-29

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
| Portability | Linux primary target; backend layer isolates platform code |

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

Two extraction strategies exist depending on what the target backend/SDK
exposes.

### 5.1 ME-only (preferred, where available)
Some SDKs expose a mode that runs *only* the motion search and returns
vectors — no transform, quantization, or entropy coding, no bitstream.
Candidates to evaluate against current vendor docs (unverified without
supporting network calls):
- **NVENC** — documented ME-only invocation intended for exactly this class
  of computer-vision use case (optical flow, temporal filtering, stereo).
- **Intel VAAPI/FEI** — ENC-only / PAK-split pipeline exposing just the
  motion-search stage.
- Rockchip's rk3588 VEPU: unconfirmed whether an ME-only mode is exposed via
  MPP; needs SDK investigation. Treat as fallback-path-only until verified.

### 5.2 Encode + decode round-trip (universal fallback)
For backends without an ME-only mode:
1. Encode L as I-frame, R as single P-frame referencing it.
2. Decode the P-frame with `AV_CODEC_FLAG2_EXPORT_MVS` set.
3. Read `AV_FRAME_DATA_MOTION_VECTORS` side-data from the decoded `AVFrame`.

Costs one HW encode + one HW decode (dedicated ASIC blocks on rk3588
VEPU/VDPU, Nvidia NVENC/NVDEC), still far cheaper than CPU-side stereo
matching, but strictly more expensive than ME-only.

### 5.3 Backend interface

```c
typedef struct cs_mv_field {
    int16_t *dx, *dy;      // per-block, integer or subpel per config
    uint8_t *cost;         // optional SAD/SATD per block (confidence), may be NULL
    int block_w, block_h;  // e.g. 16x16 for H.264; variable CTU for HEVC
    int cols, rows;
} cs_mv_field;

typedef struct cs_backend_caps {
    int has_me_only;
    int max_search_range_x, max_search_range_y;
    int supports_subpel;
    int supports_cost_output;
} cs_backend_caps;

typedef struct cs_backend_ops {
    const char *name;
    cs_backend_caps (*get_caps)(void *ctx);
    int  (*init)(void *ctx, const cs_config *cfg);
    int  (*extract)(void *ctx, const cs_frame *left, const cs_frame *right,
                     cs_mv_field *out);
    void (*destroy)(void *ctx);
} cs_backend_ops;
```

### 5.4 Planned backend implementations

| Backend | Mode | Notes |
|---|---|---|
| `cs_backend_lavc_sw` | encode+decode | libx264/libx265 via libavcodec; always available; reference/baseline for correctness and accuracy comparisons |
| `cs_backend_nvenc` | ME-only if SDK supports, else encode+decode via NVDEC | CUDA/NVENC |
| `cs_backend_vaapi_fei` | ME-only | Intel FEI |
| `cs_backend_rkmpp` | encode+decode (ME-only TBD) | Rockchip MPP, rk3588 VEPU/VDPU |
| `cs_backend_videotoolbox` | unsupported / stub | Apple platforms likely lack MV export; needs investigation before committing |

Backend selection: probe available hardware at `cs_init()`, pick the
highest-capability backend automatically, allow explicit override in config
for benchmarking/testing.

## 6. Encoder Configuration for Speed

Since compressed output quality is irrelevant, tune purely for ME cost and
search accuracy in the region that matters:

- **GOP**: none needed — one I-frame, one P-frame per stereo pair. No
  B-frames, no lookahead, no mbtree, zero-latency equivalent presets.
- **Search range**: narrow, asymmetric window — mostly horizontal (bounded by
  working depth range), small vertical band for residual rectification
  error — instead of the codec default square search. Faster and reduces
  false matches.
- **Partition/block size**: 16×16 (H.264 MB) is coarse but fast; smaller
  partitions (down to 4×4 in H.264, variable CU in HEVC) give finer disparity
  resolution at higher cost. Exposed as a speed/quality config knob.
- **MV precision**: quarter-pel gives sub-block disparity precision "for
  free" if the backend computes it natively at similar cost — kept enabled
  by default where available.
- **Pixel format**: match the encoder's native format (typically NV12) to
  avoid a colorspace-conversion tax.
- **Rate control**: irrelevant to correctness; use whatever fixed-QP/no-RC
  setting minimizes encode latency.

## 7. MV → Disparity → Depth Conversion

```
disparity[u,v] = dx[block(u,v)] / subpel_scale
depth[u,v]     = (fx * baseline) / disparity[u,v]
```

Notes:
- MV sign/direction convention differs per codec/backend (a P-frame MV
  points from the current block to its reference); each backend adapter
  normalizes this so downstream code is codec-agnostic.
- Near-zero disparity is either "at infinity" or an ambiguous/textureless
  match — requires a floor threshold and an explicit invalid/unknown marker,
  not a raw divide.
- Per-block SAD/SATD cost, when available, is used directly as a confidence
  signal for flagging low-confidence disparity — no extra cost-volume pass
  needed.
- Upsampling block-granular disparity to per-pixel is optional and off by
  default (nearest/bilinear on the block grid is cheapest; edge-aligned
  upsampling is added cost the library is explicitly trying to avoid).

## 8. Known Limitations (by design)

- Block-granular, not pixel-accurate.
- Encoder MV predictors bias toward spatial smoothness (designed to save
  bits, not preserve depth discontinuities) — object edges/occlusion
  boundaries get smeared across a block or two.
- No occlusion handling by default. Optional opt-in mode: also encode R→L
  (right as I, left as P) and cross-check consistency at ~2x cost.
- Textureless regions remain ambiguous, as with any stereo method — the
  encoder's cost metric will pick *a* low-SAD vector, possibly wrong, and
  this is only detectable via the cost/confidence signal if the backend
  exposes one.

## 9. Validation Plan

Build an offline harness early:
1. Run the same rectified stereo pairs through `codec-stereo` and through a
   reference matcher (OpenCV SGBM or similar).
2. Diff disparity maps; characterize error by scene type (textured vs.
   textureless, near vs. far range).
3. Use results to tune default block size / search range before committing
   to per-backend APIs.

## 10. Open Questions

- Confirm current NVENC SDK ME-only parameter names/availability (version-
  dependent; verify against current vendor docs before implementation).
- Confirm whether rk3588 MPP exposes any ME-only hook, or whether
  encode+decode round-trip is the permanent path for that backend.
- Decide default block size / search range presets per backend based on
  Section 9 results.
- Decide public API shape for confidence output (raw cost vs. normalized
  score) once cost-metric availability is confirmed per backend.

## 11. Suggested Next Steps

1. Implement `cs_backend_lavc_sw` (encode+decode, `AV_CODEC_FLAG2_EXPORT_MVS`)
   as the reference backend — works everywhere, needed for validation
   harness regardless of hardware target.
2. Build the validation harness (Section 9).
3. Prototype NVENC or VAAPI/FEI ME-only path on whichever hardware is
   available first, using capability probing (Section 5.4) to fall back
   cleanly.
4. Investigate rk3588 MPP ME exposure.
