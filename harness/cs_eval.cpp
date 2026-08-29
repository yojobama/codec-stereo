/*
 * cs_eval -- validation harness (Docs/codec-stereo-DESIGN.md Sec. 9).
 *
 * Runs a Middlebury 2014 stereo scene through ref_sad and lavc_sw, plus a
 * cv::StereoSGBM baseline, and reports error against ground truth. Because
 * the library's output is block-granular by construction (no upsampling by
 * default), all cross-backend comparisons are done at block-grid
 * resolution: ground truth (and SGBM's per-pixel output, for a fair
 * comparison against the block backends) is aggregated to the same grid by
 * averaging valid pixels per block. SGBM's native full-resolution numbers
 * are also reported separately, since that's the number people usually
 * cite for it.
 *
 * ref_sad's SAD cost is the only real confidence signal available (lavc_sw
 * reports none); its cost-vs-error relationship is summarized as a
 * sparsification AUC: sort blocks by ascending cost, and average the
 * bad-2.0 rate over increasing-size prefixes. A backend where cost
 * predicts error keeps this low (errors concentrate in the high-cost
 * tail); this is the direct test of Design Sec. 7's "no extra cost-volume
 * pass needed" claim for the SAD channel.
 */

#include "codec_stereo/cs.h"
#include "codec_stereo/cs_util.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>

struct CalibInfo {
    float fx = 0.0f;
    float baseline = 0.0f;
    float doffs = 0.0f;
    int ndisp = 0;
};

static bool parse_calib(const std::string &path, CalibInfo *out) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    bool got_fx = false;
    while (std::getline(f, line)) {
        if (line.rfind("cam0=", 0) == 0) {
            /* cam0=[fx 0 cx; 0 fy cy; 0 0 1] -- fx is the first number. */
            size_t b = line.find('[');
            if (b != std::string::npos) {
                out->fx = (float)atof(line.c_str() + b + 1);
                got_fx = true;
            }
        } else if (line.rfind("doffs=", 0) == 0) {
            out->doffs = (float)atof(line.c_str() + 6);
        } else if (line.rfind("baseline=", 0) == 0) {
            out->baseline = (float)atof(line.c_str() + 9);
        } else if (line.rfind("ndisp=", 0) == 0) {
            out->ndisp = atoi(line.c_str() + 6);
        }
    }
    return got_fx && out->ndisp > 0;
}

/* Aggregates a per-pixel float map to a (cols x rows) block grid by
   averaging finite values within each block; a block with no finite
   values is marked invalid (CS_DISPARITY_INVALID). */
static void aggregate_to_blocks(const float *pixels, int w, int h,
                                 int block_w, int block_h, int cols, int rows,
                                 std::vector<float> *out) {
    out->assign((size_t)cols * rows, CS_DISPARITY_INVALID);
    for (int by = 0; by < rows; by++) {
        int y0 = by * block_h, y1 = std::min(y0 + block_h, h);
        for (int bx = 0; bx < cols; bx++) {
            int x0 = bx * block_w, x1 = std::min(x0 + block_w, w);
            double sum = 0.0;
            int n = 0;
            for (int y = y0; y < y1; y++) {
                const float *row = pixels + (size_t)y * w;
                for (int x = x0; x < x1; x++) {
                    if (std::isfinite(row[x])) { sum += row[x]; n++; }
                }
            }
            if (n > 0) (*out)[(size_t)by * cols + bx] = (float)(sum / n);
        }
    }
}

struct Metrics {
    double bad1 = 0, bad2 = 0, bad4 = 0;
    double rmse = 0, mae = 0;
    double density = 0; /* valid predictions / valid ground truth */
    int n_valid_gt = 0, n_evaluated = 0;
};

static Metrics compute_metrics(const std::vector<float> &pred, const std::vector<float> &gt) {
    Metrics m;
    double se = 0, ae = 0;
    int n_valid_pred = 0;
    for (size_t i = 0; i < gt.size(); i++) {
        if (gt[i] == CS_DISPARITY_INVALID) continue;
        m.n_valid_gt++;
        if (pred[i] == CS_DISPARITY_INVALID) continue;
        n_valid_pred++;
        double err = std::fabs((double)pred[i] - (double)gt[i]);
        se += err * err;
        ae += err;
        if (err > 1.0) m.bad1++;
        if (err > 2.0) m.bad2++;
        if (err > 4.0) m.bad4++;
        m.n_evaluated++;
    }
    if (m.n_evaluated > 0) {
        m.rmse = std::sqrt(se / m.n_evaluated);
        m.mae = ae / m.n_evaluated;
        m.bad1 = 100.0 * m.bad1 / m.n_evaluated;
        m.bad2 = 100.0 * m.bad2 / m.n_evaluated;
        m.bad4 = 100.0 * m.bad4 / m.n_evaluated;
    }
    m.density = m.n_valid_gt > 0 ? (double)n_valid_pred / m.n_valid_gt : 0.0;
    return m;
}

static void print_metrics(const char *label, const Metrics &m) {
    std::printf("%-16s  bad-1.0=%6.2f%%  bad-2.0=%6.2f%%  bad-4.0=%6.2f%%  "
                "RMSE=%7.3f  MAE=%7.3f  density=%5.1f%%  (n=%d/%d)\n",
                label, m.bad1, m.bad2, m.bad4, m.rmse, m.mae,
                100.0 * m.density, m.n_evaluated, m.n_valid_gt);
}

/* Sparsification AUC: sort blocks with valid pred+gt+cost by ascending
   cost, then average the bad-2.0 rate over prefixes of increasing size
   (10%, 20%, ..., 100%). A backend whose cost predicts error keeps this
   low; a cost-blind ranking (equivalent to evaluating the full set at
   every threshold) gives back the overall bad-2.0 rate. */
static double cost_vs_error_auc(const std::vector<float> &pred, const std::vector<float> &gt,
                                 const std::vector<uint16_t> &cost, const std::vector<uint8_t> &flags) {
    struct Entry { uint16_t cost; double err; };
    std::vector<Entry> entries;
    for (size_t i = 0; i < gt.size(); i++) {
        if (gt[i] == CS_DISPARITY_INVALID || pred[i] == CS_DISPARITY_INVALID) continue;
        if (flags[i] & CS_BLK_NO_COST) continue;
        entries.push_back({cost[i], std::fabs((double)pred[i] - (double)gt[i])});
    }
    if (entries.empty()) return -1.0;
    std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) { return a.cost < b.cost; });

    double auc_sum = 0.0;
    const int steps = 10;
    for (int s = 1; s <= steps; s++) {
        size_t k = (size_t)((double)entries.size() * s / steps);
        if (k == 0) k = 1;
        int bad = 0;
        for (size_t i = 0; i < k; i++) if (entries[i].err > 2.0) bad++;
        auc_sum += 100.0 * bad / (double)k;
    }
    return auc_sum / steps;
}

static int run_backend(const char *name, const char *backend_params,
                        const cs_frame &lf, const cs_frame &rf,
                        int block_w, int block_h, int search_x, int search_y,
                        int32_t disp_offset, const std::vector<float> &gt_blocks,
                        int cols, int rows, Metrics *out_metrics = nullptr) {
    cs_config cfg = {0};
    cfg.block_w = block_w;
    cfg.block_h = block_h;
    cfg.search_range_x = search_x;
    cfg.search_range_y = search_y;
    cfg.disparity_offset = disp_offset;
    cfg.backend_override = name;
    cfg.backend_params = backend_params;

    cs_context *ctx = cs_init(&cfg);
    if (!ctx) {
        std::printf("%-16s  (backend unavailable)\n", name);
        return -1;
    }

    cs_mv_field field = {0};
    if (cs_extract(ctx, &lf, &rf, &field) != 0) {
        std::printf("%-16s  cs_extract failed\n", name);
        cs_destroy(ctx);
        return -1;
    }
    if (field.cols != cols || field.rows != rows) {
        std::printf("%-16s  unexpected grid %dx%d (wanted %dx%d)\n",
                     name, field.cols, field.rows, cols, rows);
        cs_destroy(ctx);
        return -1;
    }

    std::vector<float> pred((size_t)cols * rows);
    cs_disparity_config dcfg = {0};
    /*
     * Middlebury's disparity convention (x_left - x_right, positive) is the
     * mirror of this library's dx convention (a LEFT-anchored point at x is
     * visible in RIGHT at x+dx -- see cs.h). With LEFT/RIGHT fed as im0/im1
     * (matching disp0.pfm's own im0-anchored pixel grid, so block positions
     * line up with ground truth exactly), the physically correct disparity
     * comes out *negative* here (dx = -disparity_middlebury), which is why
     * disp_offset above is negative too. The library's min_disparity gate
     * assumes a positive-valid convention (near-zero-or-negative =
     * invalid/at-infinity) and would reject every real value in this
     * negative range, so it's disabled here; the search-window/no-match
     * flags already carry validity. The sign is flipped back for
     * Middlebury-comparable output right after this call.
     */
    dcfg.min_disparity = -1e6f;
    dcfg.max_dy = search_y > 0 ? search_y : 4;
    dcfg.max_cost = 0;
    cs_mv_field_to_disparity(&field, &dcfg, pred.data());
    for (float &d : pred) if (d != CS_DISPARITY_INVALID) d = -d;

    Metrics m = compute_metrics(pred, gt_blocks);
    print_metrics(name, m);
    if (out_metrics) *out_metrics = m;

    if (field.cost) {
        std::vector<uint16_t> cost(field.cost, field.cost + (size_t)cols * rows);
        std::vector<uint8_t> flags(field.flags, field.flags + (size_t)cols * rows);
        double auc = cost_vs_error_auc(pred, gt_blocks, cost, flags);
        if (auc >= 0.0)
            std::printf("%-16s  cost-vs-error sparsification AUC (bad-2.0%%, lower=better cost signal): %.2f\n",
                        name, auc);
    }

    cs_destroy(ctx);
    return m.n_evaluated > 0 ? 0 : -1;
}

int main(int argc, char **argv) {
    std::string data_dir = "data";
    std::string scene = "Motorcycle-perfect";
    int block_w = 16, block_h = 16;
    int search_margin = 8;
    std::string lavc_me = "umh"; /* esa is O(merange^2); intractable at ndisp~270 */
    int lavc_qp = -1; /* -1 = use lavc_sw's own default (qp=10) */

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--data" && i + 1 < argc) data_dir = argv[++i];
        else if (a == "--scene" && i + 1 < argc) scene = argv[++i];
        else if (a == "--block" && i + 2 < argc) { block_w = atoi(argv[++i]); block_h = atoi(argv[++i]); }
        else if (a == "--search-margin" && i + 1 < argc) search_margin = atoi(argv[++i]);
        else if (a == "--lavc-me" && i + 1 < argc) lavc_me = argv[++i];
        else if (a == "--lavc-qp" && i + 1 < argc) lavc_qp = atoi(argv[++i]);
        else { std::fprintf(stderr, "unrecognized argument: %s\n", a.c_str()); return 2; }
    }

    std::string scene_dir = data_dir + "/" + scene;
    CalibInfo calib;
    if (!parse_calib(scene_dir + "/calib.txt", &calib)) {
        std::fprintf(stderr, "failed to parse %s/calib.txt (run harness/fetch_middlebury.sh first)\n",
                     scene_dir.c_str());
        return 1;
    }

    cv::Mat left = cv::imread(scene_dir + "/im0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(scene_dir + "/im1.png", cv::IMREAD_GRAYSCALE);
    if (left.empty() || right.empty()) {
        std::fprintf(stderr, "failed to load im0.png/im1.png from %s\n", scene_dir.c_str());
        return 1;
    }

    float *gt_pixels = nullptr;
    int gt_w = 0, gt_h = 0;
    if (cs_pfm_read((scene_dir + "/disp0.pfm").c_str(), &gt_pixels, &gt_w, &gt_h) != 0) {
        std::fprintf(stderr, "failed to load disp0.pfm from %s\n", scene_dir.c_str());
        return 1;
    }
    if (gt_w != left.cols || gt_h != left.rows) {
        std::fprintf(stderr, "GT size %dx%d != image size %dx%d\n", gt_w, gt_h, left.cols, left.rows);
        free(gt_pixels);
        return 1;
    }

    const int w = left.cols, h = left.rows;
    const int cols = (w + block_w - 1) / block_w;
    const int rows = (h + block_h - 1) / block_h;
    const int32_t disp_offset = -(calib.ndisp / 2); /* see run_backend's dcfg comment for the sign */
    const int search_x = calib.ndisp / 2 + search_margin;
    const int search_y = 2; /* small residual-rectification-error tolerance, Design Sec. 6 */

    std::printf("scene: %s (%dx%d, fx=%.1f, baseline=%.3fmm, doffs=%.1f, ndisp=%d)\n",
                scene.c_str(), w, h, calib.fx, calib.baseline, calib.doffs, calib.ndisp);
    std::printf("block: %dx%d  search_x=%d search_y=%d disparity_offset=%d\n\n",
                block_w, block_h, search_x, search_y, disp_offset);

    std::vector<float> gt_blocks;
    aggregate_to_blocks(gt_pixels, w, h, block_w, block_h, cols, rows, &gt_blocks);

    cs_frame lf = {0}, rf = {0};
    lf.data[0] = left.data; lf.stride[0] = (int)left.step; lf.width = w; lf.height = h; lf.fmt = CS_PIX_FMT_GRAY8;
    rf.data[0] = right.data; rf.stride[0] = (int)right.step; rf.width = w; rf.height = h; rf.fmt = CS_PIX_FMT_GRAY8;

    std::printf("--- block-grid comparison (%dx%d blocks) ---\n", cols, rows);
    Metrics ref_sad_m, lavc_sw_m;
    int ref_sad_rc = run_backend("ref_sad", nullptr, lf, rf, block_w, block_h, search_x, search_y,
                                  disp_offset, gt_blocks, cols, rows, &ref_sad_m);
    std::string lavc_params = "me=" + lavc_me;
    if (lavc_qp >= 0) lavc_params += ";qp=" + std::to_string(lavc_qp);
    int lavc_sw_rc = run_backend("lavc_sw", lavc_params.c_str(), lf, rf, block_w, block_h, search_x, search_y,
                                  disp_offset, gt_blocks, cols, rows, &lavc_sw_m);

    if (ref_sad_rc == 0 && lavc_sw_rc == 0 && ref_sad_m.rmse > 0.0) {
        std::printf("\nlavc_sw RMSE / ref_sad RMSE = %.3f  "
                    "(headline cost of the encoder's RD/predictor bias vs. block granularity alone)\n",
                    lavc_sw_m.rmse / ref_sad_m.rmse);
    }

    /* SGBM baseline, aggregated to the same block grid for a fair comparison,
       and reported natively (full pixel resolution) for context. */
    int sgbm_ndisp = ((calib.ndisp + 15) / 16) * 16;
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, sgbm_ndisp, 5,
        8 * 5 * 5, 32 * 5 * 5, 1, 63, 10, 100, 32, cv::StereoSGBM::MODE_SGBM_3WAY);
    cv::Mat sgbm_disp16;
    sgbm->compute(left, right, sgbm_disp16);

    std::vector<float> sgbm_pixels((size_t)w * h);
    for (int y = 0; y < h; y++) {
        const int16_t *row = sgbm_disp16.ptr<int16_t>(y);
        for (int x = 0; x < w; x++) {
            float d = row[x] / 16.0f;
            sgbm_pixels[(size_t)y * w + x] = (d > 0.0f) ? d : CS_DISPARITY_INVALID;
        }
    }

    std::vector<float> sgbm_blocks;
    aggregate_to_blocks(sgbm_pixels.data(), w, h, block_w, block_h, cols, rows, &sgbm_blocks);
    Metrics sgbm_block_m = compute_metrics(sgbm_blocks, gt_blocks);
    print_metrics("sgbm(blockavg)", sgbm_block_m);

    std::printf("\n--- SGBM native full-resolution (context only, not block-comparable) ---\n");
    /* disp0.pfm marks unmatched/occluded pixels as +inf per the Middlebury
       spec; compute_metrics only recognizes this library's own
       CS_DISPARITY_INVALID sentinel, so translate before comparing. */
    std::vector<float> gt_full((size_t)w * h);
    for (size_t i = 0; i < gt_full.size(); i++)
        gt_full[i] = std::isfinite(gt_pixels[i]) ? gt_pixels[i] : CS_DISPARITY_INVALID;
    Metrics sgbm_full_m = compute_metrics(sgbm_pixels, gt_full);
    print_metrics("sgbm(full-res)", sgbm_full_m);

    free(gt_pixels);
    return 0;
}
