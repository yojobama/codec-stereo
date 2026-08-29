/* Sanity checks for cs_disparity_upsample and cs_disparity_cross_check
   (Phase 6 opt-in extras, Docs Sec. 8). Not exhaustive -- just enough to
   catch an obviously wrong sign/index/off-by-one before these are ever
   used against real data. */

#include "codec_stereo/cs.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failed = 1; } \
} while (0)

static void test_nearest_upsample(void) {
    /* 2x2 grid, block 4x4 -> 8x8 pixels. */
    float grid[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float pix[64];
    cs_disparity_upsample(grid, 2, 2, 4, 4, CS_UPSAMPLE_NEAREST, pix, 8, 8);

    CHECK(pix[0 * 8 + 0] == 10.0f, "nearest top-left block");
    CHECK(pix[0 * 8 + 7] == 20.0f, "nearest top-right block");
    CHECK(pix[7 * 8 + 0] == 30.0f, "nearest bottom-left block");
    CHECK(pix[7 * 8 + 7] == 40.0f, "nearest bottom-right block");
    CHECK(pix[3 * 8 + 3] == 10.0f, "nearest interior still block 0,0");
    CHECK(pix[4 * 8 + 4] == 40.0f, "nearest interior switches at block boundary");
}

static void test_bilinear_upsample_smooth(void) {
    /* 3x1 grid of increasing values; block centers at x=8,24,40 (block_w=16).
       At the exact center of the middle block, bilinear must reproduce the
       grid value exactly (no blending needed there). */
    float grid[3] = {0.0f, 10.0f, 20.0f};
    float pix[48 * 16];
    cs_disparity_upsample(grid, 3, 1, 16, 16, CS_UPSAMPLE_BILINEAR, pix, 48, 16);

    CHECK(fabsf(pix[8 * 48 + 24] - 10.0f) < 1e-3f, "bilinear exact at block center");
    /* Halfway between block 0 (center x=8) and block 1 (center x=24) is
       x=16, which should blend to the midpoint value 5.0. */
    CHECK(fabsf(pix[8 * 48 + 16] - 5.0f) < 1e-3f, "bilinear midpoint blend");
}

static void test_bilinear_invalid_fallback(void) {
    float grid[4] = {5.0f, CS_DISPARITY_INVALID, 5.0f, 5.0f};
    float pix[64];
    cs_disparity_upsample(grid, 2, 2, 4, 4, CS_UPSAMPLE_BILINEAR, pix, 8, 8);
    /* Every output pixel should fall back to a real (non-invalid,
       non-NaN) value -- never propagate CS_DISPARITY_INVALID silently
       into a blended float, and never divide-by-garbage into NaN. */
    for (int i = 0; i < 64; i++) {
        CHECK(pix[i] != CS_DISPARITY_INVALID, "bilinear fallback never invalid");
        CHECK(!isnan(pix[i]), "bilinear fallback never NaN");
    }
}

static void test_cross_check(void) {
    /* 4x1 grid, block_w=16, search_px=16 (=1 block, the implementation's
       own minimum). forward[bx] = +16 for all (perfect right-shift):
       block bx's content lands one column over, so a consistent backward
       pass (right-anchored) should report -16 at column bx+1. The search
       window is [bx+1-1, bx+1+1] = [bx, bx+2], so up to 3 backward
       columns are considered per forward cell -- consistent as long as
       *one* of them matches within max_diff. */
    float forward[4] = {16.0f, 16.0f, 16.0f, 16.0f};

    /* forward[0]: candidates are backward[0..2]; all invalid -> no partner
       anywhere in range, must invalidate regardless of tolerance. */
    /* forward[3]: candidates are backward[2..4], clamped to backward[2..3];
       both present and consistent -> must survive. */
    float backward[4] = {CS_DISPARITY_INVALID, CS_DISPARITY_INVALID, CS_DISPARITY_INVALID, -16.0f};
    float out[4];
    cs_disparity_cross_check(forward, backward, 4, 1, 16, 1.0f, 16, out);

    CHECK(out[0] == CS_DISPARITY_INVALID, "cross-check: no valid backward candidate in range -> invalid");
    CHECK(out[3] == 16.0f, "cross-check: consistent candidate in range -> kept");

    /* forward[1]: candidates are backward2[0..3], ALL of which disagree
       with -16 by more than max_diff -- must invalidate even though every
       candidate is individually valid data (just the wrong value). */
    float backward2[4] = {4.0f, 4.0f, 4.0f, 4.0f};
    float out2[4];
    cs_disparity_cross_check(forward, backward2, 4, 1, 16, 1.0f, 16, out2);
    CHECK(out2[1] == CS_DISPARITY_INVALID, "cross-check: every candidate inconsistent -> invalidated");
}

int main(void) {
    test_nearest_upsample();
    test_bilinear_upsample_smooth();
    test_bilinear_invalid_fallback();
    test_cross_check();

    if (!failed) printf("PASS: upsample + cross-check\n");
    return failed;
}
