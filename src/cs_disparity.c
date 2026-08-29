#include "codec_stereo/cs.h"

#include <stddef.h>

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
