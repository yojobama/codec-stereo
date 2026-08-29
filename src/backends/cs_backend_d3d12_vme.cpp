/*
 * cs_backend_d3d12_vme -- D3D12 Video Motion Estimation (CS_MODE_ME_ONLY),
 * Windows 10 2004+. True ME-only path: EstimateMotion + ResolveMotionVectorHeap,
 * no bitstream, no decode. See Docs/codec-stereo-DESIGN.md Sec. 5.1/5.4, and
 * the plan's "why D3D12 and not Vulkan" note (no cross-vendor Vulkan ME
 * extension exists; verified against this machine's own Vulkan device).
 *
 * Fixed to 16x16 blocks / quarter-pel precision -- the only
 * D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE / *_VECTOR_PRECISION
 * enumerators referenced in Microsoft's own published sample, so the only
 * ones confirmed to exist without a machine to grep the SDK headers on
 * ahead of time. cfg->block_w/h/subpel are accordingly ignored, same
 * spirit as lavc_sw forcing partitions=none and rkmpp forcing its
 * empirically-discovered native grid.
 *
 * Sign convention (unverified until run against tests/test_calibration):
 * EstimateMotion(pInputCurrentFrame, pInputReferenceFrame) is assumed to
 * follow the near-universal encoder convention "MV points from the current
 * block to its match in the reference frame" (same as H.264 -- see
 * lavc_sw's identical derivation). Passing right=current, left=reference
 * makes the resolved MV equal -(this library's dx); negated below to
 * match. If the calibration test disagrees, delete the negation.
 *
 * disparity_offset is handled by physically pre-shifting the right image
 * before upload (cs_shift_gray8), mirroring lavc_sw/rkmpp: this API's
 * pHintMotionVectorHeap could seed the search instead, but that needs
 * populating a heap in an undocumented format with no working example
 * available while writing this, so the already-proven pre-shift is used
 * instead. Hint-vector seeding is a candidate Phase 6 refinement.
 */

#include "codec_stereo/cs.h"
#include "codec_stereo/cs_util.h"
#include "cs_backend.h"

#include <d3d12.h>
#include <d3d12video.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstring>
#include <cstdlib>

using Microsoft::WRL::ComPtr;

namespace {

struct d3d12_vme_ctx {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12VideoDevice1> video_device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12VideoEncodeCommandList> cmdlist;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = nullptr;
    UINT64 fence_value = 0;

    ComPtr<ID3D12VideoMotionEstimator> estimator;
    ComPtr<ID3D12VideoMotionVectorHeap> mv_heap;
    ComPtr<ID3D12Resource> tex_left, tex_right, resolved_tex, readback_buf;

    int32_t disparity_offset = 0;

    int w = 0, h = 0;
    int cols = 0, rows = 0;
    int16_t *dx = nullptr, *dy = nullptr;
    uint8_t *flags = nullptr;

    UINT64 readback_row_pitch = 0;
    UINT readback_rows = 0;
};

bool wait_for_gpu(d3d12_vme_ctx *ctx) {
    ctx->fence_value++;
    if (FAILED(ctx->queue->Signal(ctx->fence.Get(), ctx->fence_value))) return false;
    if (ctx->fence->GetCompletedValue() < ctx->fence_value) {
        if (FAILED(ctx->fence->SetEventOnCompletion(ctx->fence_value, ctx->fence_event)))
            return false;
        WaitForSingleObject(ctx->fence_event, INFINITE);
    }
    return true;
}

/* Uploads an 8-bit luma buffer (flat 128 chroma, matching lavc_sw/rkmpp's
   identical "flat chroma" simplification) into an NV12 default-heap
   texture via an intermediate upload buffer, respecting D3D12's per-plane
   row-pitch alignment (D3D12_TEXTURE_DATA_PITCH_ALIGNMENT). */
bool upload_nv12(d3d12_vme_ctx *ctx, ID3D12Resource *tex, const uint8_t *luma,
                  int stride, int w, int h) {
    D3D12_RESOURCE_DESC desc = tex->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layouts[2];
    UINT num_rows[2];
    UINT64 row_bytes[2];
    UINT64 total_bytes = 0;
    ctx->device->GetCopyableFootprints(&desc, 0, 2, 0, layouts, num_rows, row_bytes, &total_bytes);

    D3D12_HEAP_PROPERTIES upload_heap = {D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC buf_desc = {};
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Width = total_bytes;
    buf_desc.Height = 1;
    buf_desc.DepthOrArraySize = 1;
    buf_desc.MipLevels = 1;
    buf_desc.SampleDesc.Count = 1;
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> upload;
    if (FAILED(ctx->device->CreateCommittedResource(
            &upload_heap, D3D12_HEAP_FLAG_NONE, &buf_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload))))
        return false;

    uint8_t *mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, (void **)&mapped))) return false;

    for (int y = 0; y < h; y++)
        memcpy(mapped + layouts[0].Offset + (size_t)y * layouts[0].Footprint.RowPitch,
               luma + (size_t)y * stride, (size_t)w);
    for (int y = 0; y < h / 2; y++)
        memset(mapped + layouts[1].Offset + (size_t)y * layouts[1].Footprint.RowPitch, 128, (size_t)w);
    upload->Unmap(0, nullptr);

    for (int plane = 0; plane < 2; plane++) {
        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = tex;
        dst.Type = D3D12_TEXTURE_COPY_LOCATION_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = plane;

        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = upload.Get();
        src.Type = D3D12_TEXTURE_COPY_LOCATION_PLACED_FOOTPRINT;
        src.PlacedFootprint = layouts[plane];

        ctx->cmdlist->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }

    /* Keep the upload buffer alive until the GPU work referencing it (the
       copy just recorded above) has actually executed and completed. */
    wait_for_gpu(ctx);
    return true;
}

int ensure_resources(d3d12_vme_ctx *ctx, int w, int h) {
    if (w == ctx->w && h == ctx->h && ctx->estimator) return 0;

    ctx->estimator.Reset();
    ctx->mv_heap.Reset();
    ctx->tex_left.Reset();
    ctx->tex_right.Reset();
    ctx->resolved_tex.Reset();
    ctx->readback_buf.Reset();

    D3D12_HEAP_PROPERTIES default_heap = {D3D12_HEAP_TYPE_DEFAULT};

    D3D12_RESOURCE_DESC nv12_desc = {};
    nv12_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    nv12_desc.Width = (UINT64)w;
    nv12_desc.Height = (UINT)h;
    nv12_desc.DepthOrArraySize = 1;
    nv12_desc.MipLevels = 1;
    nv12_desc.Format = DXGI_FORMAT_NV12;
    nv12_desc.SampleDesc.Count = 1;
    nv12_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    if (FAILED(ctx->device->CreateCommittedResource(
            &default_heap, D3D12_HEAP_FLAG_NONE, &nv12_desc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&ctx->tex_left))))
        return -1;
    if (FAILED(ctx->device->CreateCommittedResource(
            &default_heap, D3D12_HEAP_FLAG_NONE, &nv12_desc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&ctx->tex_right))))
        return -1;

    D3D12_VIDEO_SIZE_RANGE size_range = {(UINT)w, (UINT)h, 16, 16};
    D3D12_VIDEO_MOTION_ESTIMATOR_DESC est_desc = {
        0, DXGI_FORMAT_NV12,
        D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_16X16,
        D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_QUARTER_PEL,
        size_range};
    if (FAILED(ctx->video_device->CreateVideoMotionEstimator(&est_desc, nullptr, IID_PPV_ARGS(&ctx->estimator))))
        return -1;

    D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC heap_desc = {
        0, DXGI_FORMAT_NV12,
        D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_16X16,
        D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_QUARTER_PEL,
        size_range};
    if (FAILED(ctx->video_device->CreateVideoMotionVectorHeap(&heap_desc, nullptr, IID_PPV_ARGS(&ctx->mv_heap))))
        return -1;

    int cols = (w + 15) / 16, rows = (h + 15) / 16;

    D3D12_RESOURCE_DESC resolved_desc = {};
    resolved_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resolved_desc.Width = (UINT64)cols;
    resolved_desc.Height = (UINT)rows;
    resolved_desc.DepthOrArraySize = 1;
    resolved_desc.MipLevels = 1;
    resolved_desc.Format = DXGI_FORMAT_R16G16_SINT;
    resolved_desc.SampleDesc.Count = 1;
    resolved_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    if (FAILED(ctx->device->CreateCommittedResource(
            &default_heap, D3D12_HEAP_FLAG_NONE, &resolved_desc,
            D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE, nullptr, IID_PPV_ARGS(&ctx->resolved_tex))))
        return -1;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
    UINT num_rows;
    UINT64 row_bytes, total_bytes;
    ctx->device->GetCopyableFootprints(&resolved_desc, 0, 1, 0, &layout, &num_rows, &row_bytes, &total_bytes);
    ctx->readback_row_pitch = layout.Footprint.RowPitch;
    ctx->readback_rows = num_rows;

    D3D12_HEAP_PROPERTIES readback_heap = {D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC rb_desc = {};
    rb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rb_desc.Width = total_bytes;
    rb_desc.Height = 1;
    rb_desc.DepthOrArraySize = 1;
    rb_desc.MipLevels = 1;
    rb_desc.SampleDesc.Count = 1;
    rb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(ctx->device->CreateCommittedResource(
            &readback_heap, D3D12_HEAP_FLAG_NONE, &rb_desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&ctx->readback_buf))))
        return -1;

    free(ctx->dx); free(ctx->dy); free(ctx->flags);
    size_t n = (size_t)cols * rows;
    ctx->dx = (int16_t *)malloc(n * sizeof(int16_t));
    ctx->dy = (int16_t *)malloc(n * sizeof(int16_t));
    ctx->flags = (uint8_t *)malloc(n * sizeof(uint8_t));
    if (!ctx->dx || !ctx->dy || !ctx->flags) return -1;

    ctx->w = w;
    ctx->h = h;
    ctx->cols = cols;
    ctx->rows = rows;
    return 0;
}

} // namespace

static cs_backend_caps d3d12_vme_get_caps(void *vctx) {
    d3d12_vme_ctx *ctx = (d3d12_vme_ctx *)vctx;
    (void)ctx;
    cs_backend_caps caps;
    memset(&caps, 0, sizeof caps);
    caps.mode = CS_MODE_ME_ONLY;
    caps.native_block_w = 16;
    caps.native_block_h = 16;
    /* Not queried via D3D12_FEATURE_VIDEO_MOTION_ESTIMATOR (its exact
       result struct fields weren't confirmable without a machine to grep
       the SDK headers on while writing this); D3D12_VIDEO_SIZE_RANGE at
       resource-creation time is the actual enforcement point instead. */
    caps.max_search_range_x = 256;
    caps.max_search_range_y = 256;
    caps.mv_min_x = -32768; caps.mv_max_x = 32767; /* DXGI_FORMAT_R16G16_SINT */
    caps.mv_min_y = -32768; caps.mv_max_y = 32767;
    caps.supports_subpel = 1; /* quarter-pel, always -- see file header */
    caps.cost_metric = CS_COST_NONE; /* this API reports no cost at all */
    return caps;
}

static int d3d12_vme_init(void *vctx, const cs_config *cfg) {
    d3d12_vme_ctx *ctx = (d3d12_vme_ctx *)vctx;
    ctx->disparity_offset = cfg->disparity_offset;

    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&ctx->device))))
        return -1;
    if (FAILED(ctx->device.As(&ctx->video_device))) return -1;

    D3D12_COMMAND_QUEUE_DESC qdesc = {};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE;
    if (FAILED(ctx->device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&ctx->queue)))) return -1;
    if (FAILED(ctx->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE,
                                                    IID_PPV_ARGS(&ctx->allocator))))
        return -1;
    if (FAILED(ctx->device->CreateCommandList1(0, D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE,
                                                D3D12_COMMAND_LIST_FLAG_NONE,
                                                IID_PPV_ARGS(&ctx->cmdlist))))
        return -1;
    if (FAILED(ctx->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx->fence))))
        return -1;
    ctx->fence_event = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
    if (!ctx->fence_event) return -1;

    return 0;
}

static int d3d12_vme_extract(void *vctx, const cs_frame *left, const cs_frame *right,
                              cs_mv_field *out) {
    d3d12_vme_ctx *ctx = (d3d12_vme_ctx *)vctx;
    const int w = left->width, h = left->height;
    if (right->width != w || right->height != h) return -1;
    if (ensure_resources(ctx, w, h) != 0) return -1;

    if (FAILED(ctx->allocator->Reset())) return -1;
    if (FAILED(ctx->cmdlist->Reset(ctx->allocator.Get()))) return -1;

    uint8_t *right_shifted = nullptr;
    const uint8_t *right_luma = right->data[0];
    int right_stride = right->stride[0];
    if (ctx->disparity_offset != 0) {
        right_shifted = (uint8_t *)malloc((size_t)w * h);
        if (!right_shifted) return -1;
        cs_shift_gray8(right->data[0], right->stride[0], right_shifted, w,
                        w, h, -ctx->disparity_offset);
        right_luma = right_shifted;
        right_stride = w;
    }

    bool ok = upload_nv12(ctx, ctx->tex_left.Get(), left->data[0], left->stride[0], w, h) &&
              upload_nv12(ctx, ctx->tex_right.Get(), right_luma, right_stride, w, h);
    free(right_shifted);
    if (!ok) return -1;

    if (FAILED(ctx->allocator->Reset())) return -1;
    if (FAILED(ctx->cmdlist->Reset(ctx->allocator.Get()))) return -1;

    D3D12_RESOURCE_BARRIER barriers[2] = {};
    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[0].Transition.pResource = ctx->tex_left.Get();
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ;
    barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barriers[1] = barriers[0];
    barriers[1].Transition.pResource = ctx->tex_right.Get();
    ctx->cmdlist->ResourceBarrier(2, barriers);

    /* current=right, reference=left -- see file header for the resulting
       sign convention this implies and how it's compensated below. */
    D3D12_VIDEO_MOTION_ESTIMATOR_OUTPUT est_out = {ctx->mv_heap.Get()};
    D3D12_VIDEO_MOTION_ESTIMATOR_INPUT est_in = {
        ctx->tex_right.Get(), 0, ctx->tex_left.Get(), 0, nullptr};
    ctx->cmdlist->EstimateMotion(ctx->estimator.Get(), &est_out, &est_in);

    D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_OUTPUT resolve_out = {ctx->resolved_tex.Get(), {}};
    D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_INPUT resolve_in = {ctx->mv_heap.Get(), (UINT)w, (UINT)h};
    ctx->cmdlist->ResolveMotionVectorHeap(&resolve_out, &resolve_in);

    D3D12_RESOURCE_BARRIER to_copy = {};
    to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_copy.Transition.pResource = ctx->resolved_tex.Get();
    to_copy.Transition.StateBefore = D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE;
    to_copy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    to_copy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ctx->cmdlist->ResourceBarrier(1, &to_copy);

    D3D12_RESOURCE_DESC resolved_desc = ctx->resolved_tex->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
    ctx->device->GetCopyableFootprints(&resolved_desc, 0, 1, 0, &layout, nullptr, nullptr, nullptr);

    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource = ctx->readback_buf.Get();
    dst.Type = D3D12_TEXTURE_COPY_LOCATION_PLACED_FOOTPRINT;
    dst.PlacedFootprint = layout;
    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource = ctx->resolved_tex.Get();
    src.Type = D3D12_TEXTURE_COPY_LOCATION_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;
    ctx->cmdlist->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    if (FAILED(ctx->cmdlist->Close())) return -1;
    ID3D12CommandList *lists[] = {ctx->cmdlist.Get()};
    ctx->queue->ExecuteCommandLists(1, lists);
    if (!wait_for_gpu(ctx)) return -1;

    void *mapped = nullptr;
    D3D12_RANGE read_range = {0, (SIZE_T)(ctx->readback_row_pitch * ctx->readback_rows)};
    if (FAILED(ctx->readback_buf->Map(0, &read_range, &mapped))) return -1;

    const int16_t *pixels = (const int16_t *)mapped;
    for (int y = 0; y < ctx->rows; y++) {
        const int16_t *row = (const int16_t *)((const uint8_t *)mapped + (size_t)y * ctx->readback_row_pitch);
        for (int x = 0; x < ctx->cols; x++) {
            size_t idx = (size_t)y * ctx->cols + x;
            int16_t r = row[x * 2 + 0];
            int16_t g = row[x * 2 + 1];
            /* Negated: see file header sign-convention note. */
            ctx->dx[idx] = (int16_t)(-r);
            ctx->dy[idx] = (int16_t)(-g);
            ctx->flags[idx] = (uint8_t)CS_BLK_NO_COST;
        }
    }
    (void)pixels;
    D3D12_RANGE no_write = {0, 0};
    ctx->readback_buf->Unmap(0, &no_write);

    out->dx = ctx->dx;
    out->dy = ctx->dy;
    out->cost = nullptr;
    out->flags = ctx->flags;
    out->block_w = 16;
    out->block_h = 16;
    out->cols = ctx->cols;
    out->rows = ctx->rows;
    out->subpel_bits = 2;
    out->disparity_offset = ctx->disparity_offset;
    return 0;
}

static void d3d12_vme_destroy(void *vctx) {
    d3d12_vme_ctx *ctx = (d3d12_vme_ctx *)vctx;
    if (!ctx) return;
    if (ctx->fence_event) CloseHandle(ctx->fence_event);
    free(ctx->dx);
    free(ctx->dy);
    free(ctx->flags);
    delete ctx;
}

static void *d3d12_vme_create(void) {
    return new (std::nothrow) d3d12_vme_ctx();
}

extern "C" cs_backend_factory cs_backend_d3d12_vme_factory(void) {
    cs_backend_factory f;
    memset(&f, 0, sizeof f);
    f.create = d3d12_vme_create;
    f.mode = CS_MODE_ME_ONLY;
    f.ops.name = "d3d12_vme";
    f.ops.get_caps = d3d12_vme_get_caps;
    f.ops.init = d3d12_vme_init;
    f.ops.extract = d3d12_vme_extract;
    f.ops.destroy = d3d12_vme_destroy;
    return f;
}
