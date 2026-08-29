#include "codec_stereo/cs.h"

#include <stddef.h>
#include <math.h>

void cs_mv_field_to_disparity(const cs_mv_field *field,
                               const cs_disparity_config *cfg,
                               float *disparity_out) {
    const int n = field->cols * field->rows;
    const float subpel_scale = (float)(1 << field->subpel_bits);

    for (int i = 0; i < n; i++) {
        uint8_t flags = field->flags ? field->flags[i] : (uint8_t)CS_BLK_OK;

        /* CS_BLK_NO_COST only means "this backend never reports cost" --
           it does not by itself invalidate the block. Every other flag
           marks the block's motion vector as unusable. */
        if (flags & (CS_BLK_INTRA | CS_BLK_NO_MATCH)) {
            disparity_out[i] = CS_DISPARITY_INVALID;
            continue;
        }

        if (cfg->max_dy > 0) {
            int16_t dy_int = (int16_t)(field->dy[i] / (int16_t)subpel_scale);
            if (dy_int > cfg->max_dy || dy_int < -cfg->max_dy) {
                disparity_out[i] = CS_DISPARITY_INVALID;
                continue;
            }
        }

        if (cfg->max_cost > 0 && field->cost && !(flags & CS_BLK_NO_COST) &&
            field->cost[i] > cfg->max_cost) {
            disparity_out[i] = CS_DISPARITY_INVALID;
            continue;
        }

        float disparity = (float)field->dx[i] / subpel_scale + (float)field->disparity_offset;

        if (disparity <= cfg->min_disparity) {
            disparity_out[i] = CS_DISPARITY_INVALID;
            continue;
        }

        disparity_out[i] = disparity;
    }
}

void cs_disparity_to_depth(const float *disparity_in, size_t n,
                            const cs_disparity_config *cfg,
                            float *depth_out) {
    const float fb = cfg->fx * cfg->baseline;

    for (size_t i = 0; i < n; i++) {
        float d = disparity_in[i];
        if (d == CS_DISPARITY_INVALID || d <= cfg->min_disparity) {
            depth_out[i] = 0.0f;
            continue;
        }
        depth_out[i] = fb / d;
    }
}

/* Nearest valid grid cell to (bx,by), by expanding-ring search; returns
   CS_DISPARITY_INVALID if the whole grid is invalid. Only used as the
   bilinear upsampler's fallback when a pixel's 4 surrounding cells include
   an invalid one -- grids are small, so this stays cheap in practice. */
static float nearest_valid(const float *grid, int cols, int rows, int bx, int by) {
    if (grid[by * cols + bx] != CS_DISPARITY_INVALID) return grid[by * cols + bx];

    for (int radius = 1; radius < cols + rows; radius++) {
        for (int dy = -radius; dy <= radius; dy++) {
            int y = by + dy;
            if (y < 0 || y >= rows) continue;
            int dx_span = radius - (dy < 0 ? -dy : dy);
            for (int dx = -dx_span; dx <= dx_span; dx += (dx_span == 0 ? 1 : 2 * dx_span)) {
                int x = bx + dx;
                if (x < 0 || x >= cols) continue;
                float v = grid[y * cols + x];
                if (v != CS_DISPARITY_INVALID) return v;
            }
        }
    }
    return CS_DISPARITY_INVALID;
}

void cs_disparity_upsample(const float *disparity_grid, int cols, int rows,
                            int block_w, int block_h,
                            cs_upsample_mode mode,
                            float *pixel_out, int out_w, int out_h) {
    for (int py = 0; py < out_h; py++) {
        for (int px = 0; px < out_w; px++) {
            size_t out_idx = (size_t)py * out_w + px;

            if (mode == CS_UPSAMPLE_NEAREST) {
                int bx = px / block_w;
                int by = py / block_h;
                if (bx >= cols) bx = cols - 1;
                if (by >= rows) by = rows - 1;
                pixel_out[out_idx] = disparity_grid[(size_t)by * cols + bx];
                continue;
            }

            /* Bilinear: sample points are block centers; fx/fy are the
               pixel's fractional position between the 4 surrounding
               centers, clamped at the grid edges (edge blocks repeat). */
            float gx = (float)px / block_w - 0.5f;
            float gy = (float)py / block_h - 0.5f;
            int bx0 = (int)floorf(gx), by0 = (int)floorf(gy);
            float fx = gx - (float)bx0, fy = gy - (float)by0;
            int bx1 = bx0 + 1, by1 = by0 + 1;
            if (bx0 < 0) bx0 = 0; if (bx0 >= cols) bx0 = cols - 1;
            if (bx1 < 0) bx1 = 0; if (bx1 >= cols) bx1 = cols - 1;
            if (by0 < 0) by0 = 0; if (by0 >= rows) by0 = rows - 1;
            if (by1 < 0) by1 = 0; if (by1 >= rows) by1 = rows - 1;

            float v00 = disparity_grid[(size_t)by0 * cols + bx0];
            float v01 = disparity_grid[(size_t)by0 * cols + bx1];
            float v10 = disparity_grid[(size_t)by1 * cols + bx0];
            float v11 = disparity_grid[(size_t)by1 * cols + bx1];

            if (v00 == CS_DISPARITY_INVALID || v01 == CS_DISPARITY_INVALID ||
                v10 == CS_DISPARITY_INVALID || v11 == CS_DISPARITY_INVALID) {
                int bx = px / block_w, by = py / block_h;
                if (bx >= cols) bx = cols - 1;
                if (by >= rows) by = rows - 1;
                pixel_out[out_idx] = nearest_valid(disparity_grid, cols, rows, bx, by);
                continue;
            }

            float v0 = v00 + (v01 - v00) * fx;
            float v1 = v10 + (v11 - v10) * fx;
            pixel_out[out_idx] = v0 + (v1 - v0) * fy;
        }
    }
}

void cs_disparity_cross_check(const float *forward, const float *backward,
                               int cols, int rows, int block_w,
                               float max_diff, int search_px,
                               float *out) {
    int search_blocks = (search_px + block_w - 1) / block_w;
    if (search_blocks < 1) search_blocks = 1;

    for (int by = 0; by < rows; by++) {
        for (int bx = 0; bx < cols; bx++) {
            size_t idx = (size_t)by * cols + bx;
            float fwd = forward[idx];
            if (fwd == CS_DISPARITY_INVALID) {
                out[idx] = CS_DISPARITY_INVALID;
                continue;
            }

            int center_bx = bx + (int)lroundf(fwd / (float)block_w);
            int found = 0;
            for (int c = center_bx - search_blocks; c <= center_bx + search_blocks && !found; c++) {
                if (c < 0 || c >= cols) continue;
                float bwd = backward[(size_t)by * cols + c];
                if (bwd == CS_DISPARITY_INVALID) continue;
                if (fabsf(fwd + bwd) <= max_diff) found = 1;
            }

            out[idx] = found ? fwd : CS_DISPARITY_INVALID;
        }
    }
}
