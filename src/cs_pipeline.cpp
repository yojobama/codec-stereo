/*
 * cs_pipeline -- worker-pool implementation. See
 * include/codec_stereo/cs_pipeline.h for the rationale and the measured
 * scaling numbers that motivated it.
 *
 * This is the one C++ translation unit in the library. The reason is
 * concurrency ergonomics, not raw speed: the per-pair work happens inside
 * cs_extract(), which is C and unchanged, so C++ buys nothing there. What
 * it buys here is std::thread/std::condition_variable and RAII-owned
 * buffers for the queue plumbing, which in C would be pthreads plus
 * hand-rolled ownership across ~4 allocation sites and an error path per
 * worker. The public interface stays C (extern "C" in the header) so
 * nothing else in the project has to care.
 */

#include "codec_stereo/cs_pipeline.h"

#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct Job {
    uint64_t seq = 0;
    /* Copied luma planes -- see the submit() contract in the header. Kept
       tightly packed (stride == width) so the worker hands the backend a
       simple contiguous frame. Empty when `borrowed` is set. */
    std::vector<uint8_t> left, right;
    /* Borrowed mode (cs_pipeline_submit_borrowed): read straight from the
       caller's memory, which the caller has promised to keep alive and
       unmodified until this job's result is retrieved. */
    bool borrowed = false;
    const uint8_t *bleft = nullptr, *bright = nullptr;
    int bstride_l = 0, bstride_r = 0;
    int w = 0, h = 0;

    std::vector<float> disparity;
    int cols = 0, rows = 0;
    int block_w = 0, block_h = 0;
    int status = -1;
};

using JobPtr = std::shared_ptr<Job>;

} // namespace

struct cs_pipeline {
    cs_disparity_config dcfg{};

    std::vector<std::thread> workers;
    std::vector<cs_context *> contexts;

    std::mutex mu;
    std::condition_variable cv_pending;   /* workers wait for work         */
    std::condition_variable cv_space;     /* submit waits for queue space  */
    std::condition_variable cv_done;      /* get waits for its turn        */

    std::deque<JobPtr> pending;
    std::map<uint64_t, JobPtr> completed; /* out-of-order landing zone,
                                             drained in seq order by get() */
    size_t max_pending = 0;

    uint64_t next_submit_seq = 0;
    uint64_t next_get_seq = 0;
    uint64_t submitted_total = 0;

    bool stopping = false;

    /* Keeps the buffer handed out by the previous cs_pipeline_get() alive
       until the following call, which is exactly the lifetime the header
       promises. */
    JobPtr last_returned;
};

namespace {

void worker_main(cs_pipeline *p, cs_context *ctx) {
    for (;;) {
        JobPtr job;
        {
            std::unique_lock<std::mutex> lk(p->mu);
            p->cv_pending.wait(lk, [&] { return p->stopping || !p->pending.empty(); });
            if (p->stopping && p->pending.empty()) return;
            job = p->pending.front();
            p->pending.pop_front();
        }
        /* Space freed; a blocked submitter can proceed while we work. */
        p->cv_space.notify_one();

        cs_frame lf{}, rf{};
        lf.width = job->w;
        lf.height = job->h;
        lf.fmt = CS_PIX_FMT_GRAY8;
        rf = lf;
        if (job->borrowed) {
            lf.data[0] = job->bleft;   lf.stride[0] = job->bstride_l;
            rf.data[0] = job->bright;  rf.stride[0] = job->bstride_r;
        } else {
            lf.data[0] = job->left.data();   lf.stride[0] = job->w;
            rf.data[0] = job->right.data();  rf.stride[0] = job->w;
        }

        cs_mv_field field{};
        int rc = cs_extract(ctx, &lf, &rf, &field);
        if (rc == 0) {
            job->cols = field.cols;
            job->rows = field.rows;
            job->block_w = field.block_w;
            job->block_h = field.block_h;
            job->disparity.resize(static_cast<size_t>(field.cols) * field.rows);
            /* Converted here, on the worker, and copied into the job --
               field.dx/dy point into backend-owned memory that this same
               context will overwrite on its very next cs_extract(). */
            cs_mv_field_to_disparity(&field, &p->dcfg, job->disparity.data());
        }
        job->status = rc;

        {
            std::lock_guard<std::mutex> lk(p->mu);
            p->completed.emplace(job->seq, job);
        }
        p->cv_done.notify_all();
    }
}

} // namespace

extern "C" cs_pipeline *cs_pipeline_create(const cs_config *cfg,
                                            const cs_disparity_config *dcfg,
                                            int workers) {
    if (workers <= 0) workers = 4;

    auto *p = new (std::nothrow) cs_pipeline();
    if (!p) return nullptr;
    if (dcfg) p->dcfg = *dcfg;
    /* Two jobs in flight per worker: one being processed, one queued so a
       worker never idles waiting on the submitting thread. */
    p->max_pending = static_cast<size_t>(workers) * 2;

    for (int i = 0; i < workers; i++) {
        cs_context *ctx = cs_init(cfg);
        if (!ctx) break; /* fall back to however many did initialize */
        p->contexts.push_back(ctx);
    }
    if (p->contexts.empty()) {
        delete p;
        return nullptr;
    }

    for (cs_context *ctx : p->contexts)
        p->workers.emplace_back(worker_main, p, ctx);

    return p;
}

namespace {

/* Blocks for queue space, stamps the submission-order seq, enqueues. */
int enqueue(cs_pipeline *p, const JobPtr &job) {
    {
        std::unique_lock<std::mutex> lk(p->mu);
        p->cv_space.wait(lk, [&] { return p->stopping || p->pending.size() < p->max_pending; });
        if (p->stopping) return -1;
        job->seq = p->next_submit_seq++;
        p->submitted_total++;
        p->pending.push_back(job);
    }
    p->cv_pending.notify_one();
    return 0;
}

} // namespace

extern "C" int cs_pipeline_submit(cs_pipeline *p, const cs_frame *left,
                                   const cs_frame *right) {
    if (!p || !left || !right) return -1;
    if (left->width != right->width || left->height != right->height) return -1;

    const int w = left->width, h = left->height;

    auto job = std::make_shared<Job>();
    job->w = w;
    job->h = h;
    job->left.resize(static_cast<size_t>(w) * h);
    job->right.resize(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; y++) {
        std::memcpy(job->left.data() + static_cast<size_t>(y) * w,
                    left->data[0] + static_cast<size_t>(y) * left->stride[0], w);
        std::memcpy(job->right.data() + static_cast<size_t>(y) * w,
                    right->data[0] + static_cast<size_t>(y) * right->stride[0], w);
    }

    return enqueue(p, job);
}

extern "C" int cs_pipeline_submit_borrowed(cs_pipeline *p, const cs_frame *left,
                                            const cs_frame *right) {
    if (!p || !left || !right) return -1;
    if (left->width != right->width || left->height != right->height) return -1;

    auto job = std::make_shared<Job>();
    job->w = left->width;
    job->h = left->height;
    job->borrowed = true;
    job->bleft = left->data[0];
    job->bright = right->data[0];
    job->bstride_l = left->stride[0];
    job->bstride_r = right->stride[0];

    return enqueue(p, job);
}

extern "C" int cs_pipeline_get(cs_pipeline *p, cs_pipeline_result *out) {
    if (!p || !out) return -1;

    JobPtr job;
    {
        std::unique_lock<std::mutex> lk(p->mu);
        if (p->next_get_seq >= p->submitted_total && p->stopping) return -1;
        p->cv_done.wait(lk, [&] {
            return p->stopping || p->completed.count(p->next_get_seq) > 0;
        });
        auto it = p->completed.find(p->next_get_seq);
        if (it == p->completed.end()) return -1; /* stopping, nothing pending */
        job = it->second;
        p->completed.erase(it);
        p->next_get_seq++;
        /* Hold a reference so job->disparity outlives this call, per the
           header's "valid until the next get()" contract. */
        p->last_returned = job;
    }

    out->seq = job->seq;
    out->status = job->status;
    out->disparity = job->disparity.data();
    out->cols = job->cols;
    out->rows = job->rows;
    out->block_w = job->block_w;
    out->block_h = job->block_h;
    return 0;
}

extern "C" void cs_pipeline_destroy(cs_pipeline *p) {
    if (!p) return;
    {
        std::lock_guard<std::mutex> lk(p->mu);
        p->stopping = true;
    }
    p->cv_pending.notify_all();
    p->cv_space.notify_all();
    p->cv_done.notify_all();

    for (auto &t : p->workers)
        if (t.joinable()) t.join();

    for (cs_context *ctx : p->contexts)
        cs_destroy(ctx);

    delete p;
}
