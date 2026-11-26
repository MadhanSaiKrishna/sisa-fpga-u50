// fpga_backend.h -- one GEMM backend for BOTH bitstreams (gemm_sa = SA v2.4, gemm_sisa = SISA v2.5), header-only.
//
// Everything about the board interface is copied from host/host.cpp so the existing xclbins run unchanged:
//   * kernel name: try "gemm_sisa", fall back to "gemm_sa"              (host.cpp run_main)
//   * A buffers : SISA = 8 bos, krn.group_id(0..7)  (HBM[0..7], one full packed copy of A each)
//                 SA   = 1 bo,  krn.group_id(0)     (HBM[0])
//   * B bo      : krn.group_id(nA)       (HBM[8])      nA = 8 (SISA) or 1 (SA)
//   * C bo      : krn.group_id(nA + 1)   (HBM[9])
//   * launch    : SISA  krn(A0..A7, B, C, m_tiles, n_tiles, K, f)
//                 SA    krn(A,       B, C, m_tiles, n_tiles, K)
//   * layouts   : pack_a / pack_b / unpack_c from src/common.h (the very same functions host.cpp uses)
//
// C[M x N] = A[M x K] * B[K x N]   with A = weights (row-major, exactly llama2.c's w[d][n], M = d, K = n).
//
// Weight cache: pointer-keyed on (w, M, K). The first call that sees a weight pointer packs it and writes it to
// HBM once (1 bo for SA, 8 for SISA); every later call only moves B (activations) and C (results).
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "experimental/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "../src/common.h"

namespace slm {

using clk = std::chrono::steady_clock;
static inline double usec(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

// Per-call timing breakdown (microseconds). device() == what host.cpp calls "kernel time" (enqueue + wait).
struct CallTimes {
  double pack_x = 0;    // host: write activations into the packed B staging buffer (strided for N = 1) + shadow compare
  double b_sync = 0;    // PCIe: B bo -> HBM[8]           (0 when the previous upload is reused)
  double launch = 0;    // host: set kernel args + start (krn(...) returns once the command is queued)
  double wait = 0;      // run.wait(): kernel compute + completion notification
  double c_sync = 0;    // PCIe: C bo <- HBM[9]
  double unpack = 0;    // host: packed C -> row-major result
  double upload = 0;    // one-time weight upload triggered by THIS call (not part of total())
  bool b_reused = false;
  int f = 0, steps = 0;
  double device() const { return launch + wait; }
  double total() const { return pack_x + b_sync + launch + wait + c_sync + unpack; }
};

struct WeightSet {
  int M = 0, K = 0;
  std::vector<xrt::bo> A;      // 1 (SA) or 8 (SISA) full packed copies, each on its own HBM bank
  double upload_us = 0;
  size_t bytes_per_copy = 0;
};

class Backend {
 public:
  Backend(const std::string& xclbin, bool b_reuse = true) : b_reuse_(b_reuse) {
    dev_ = xrt::device(0);                           // inside the gate sandbox only the U50 is visible
    auto uuid = dev_.load_xclbin(xclbin);
    try {
      krn_ = xrt::kernel(dev_, uuid, "gemm_sisa");
      is_sisa_ = true;
    } catch (...) {
      is_sisa_ = false;
      krn_ = xrt::kernel(dev_, uuid, "gemm_sa");
    }
    nA_ = is_sisa_ ? NUM_SLABS : 1;
  }
  bool is_sisa() const { return is_sisa_; }
  const char* name() const { return is_sisa_ ? "gemm_sisa" : "gemm_sa"; }
  void set_b_reuse(bool v) { b_reuse_ = v; }

  // Static weights -> HBM, once per (pointer, M, K).
  WeightSet& weights(const float* w, int M, int K, double* upload_us_now = nullptr) {
    const auto key = std::make_tuple(w, M, K);
    auto it = wcache_.find(key);
    if (it != wcache_.end()) return *it->second;
    const auto t0 = clk::now();
    auto ws = std::make_unique<WeightSet>();
    ws->M = M; ws->K = K;
    {
      std::vector<data_t> Afull(w, w + (size_t)M * K);
      std::vector<data_t> Ap = pack_a(Afull, M, K);
      ws->bytes_per_copy = sizeof(data_t) * Ap.size();
      for (int i = 0; i < nA_; ++i) {
        ws->A.emplace_back(dev_, ws->bytes_per_copy, krn_.group_id(i));
        std::memcpy(ws->A[i].map<data_t*>(), Ap.data(), ws->bytes_per_copy);
        ws->A[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }
    }
    ws->upload_us = usec(t0, clk::now());
    hbm_bytes_ += ws->bytes_per_copy * (size_t)nA_;
    upload_us_total_ += ws->upload_us;
    if (upload_us_now) *upload_us_now = ws->upload_us;
    return *(wcache_[key] = std::move(ws));
  }

  // C[M x N] (row-major) = A[M x K] * B[K x N] (row-major).  For N = 1 this is llama2.c's  xout[d] = w[d][n] . x[n].
  // fsel: SISA fusion factor (<= 0: choose_fusion(N)); ignored for SA (always f = NUM_SLABS).
  void gemm(const float* w, int M, int K, const float* B, int N, float* C, CallTimes& t, int fsel = -1) {
    t = CallTimes();
    const int f = is_sisa_ ? (fsel > 0 ? fsel : choose_fusion(N)) : NUM_SLABS;
    const Geo geo(M, N, f);
    t.f = f; t.steps = geo.steps;
    double up = 0;
    WeightSet& ws = weights(w, M, K, &up);
    t.upload = up;
    BBuf& bb = bbuf(K, N, f);
    CBuf& cb = cbuf(c_packed_floats(geo));

    // ---- B: pack + upload (skipped if identical to what this B buffer already holds) ----
    auto t0 = clk::now();
    const size_t nb = (size_t)K * N;
    const bool reuse = b_reuse_ && bb.valid && std::memcmp(bb.shadow.data(), B, sizeof(data_t) * nb) == 0;
    auto t1 = clk::now();
    if (reuse) {
      t.b_reused = true;
      t.pack_x = usec(t0, t1);
    } else {
      data_t* p = bb.bo.map<data_t*>();
      if (N == 1) {
        // pack_b with N = 1: P[(0*K + k)*ROWS + 0] = x[k]; every other float stays 0 (buffer zeroed at creation)
        for (int k = 0; k < K; ++k) p[(size_t)k * ROWS] = B[k];
      } else {
        const std::vector<data_t> Bp = pack_b(std::vector<data_t>(B, B + nb), K, N, f);
        std::memcpy(p, Bp.data(), sizeof(data_t) * Bp.size());
      }
      if (b_reuse_) std::memcpy(bb.shadow.data(), B, sizeof(data_t) * nb);
      auto t2 = clk::now();
      bb.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      auto t3 = clk::now();
      bb.valid = true;
      t.pack_x = usec(t0, t2);
      t.b_sync = usec(t2, t3);
    }

    // ---- launch (argument order exactly as host.cpp) ----
    auto t4 = clk::now();
    auto run = is_sisa_
        ? krn_(ws.A[0], ws.A[1], ws.A[2], ws.A[3], ws.A[4], ws.A[5], ws.A[6], ws.A[7], bb.bo, cb.bo,
               geo.m_tiles, geo.n_tiles, K, f)
        : krn_(ws.A[0], bb.bo, cb.bo, geo.m_tiles, geo.n_tiles, K);
    auto t5 = clk::now();
    run.wait();
    auto t6 = clk::now();
    t.launch = usec(t4, t5);
    t.wait = usec(t5, t6);

    // ---- C: download + unpack ----
    cb.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto t7 = clk::now();
    t.c_sync = usec(t6, t7);
    const data_t* Cp = cb.bo.map<data_t*>();
    if (N == 1) {
      // unpack_c restricted to n = 0: only slab positions with (s & (f-1)) == 0, row i = 0
      const int sh = fusion_shift(f);
      for (int gi = 0; gi < geo.groups; ++gi)
        for (int t_ = 0; t_ < T_ACC; ++t_)
          for (int s = 0; s < NUM_SLABS; ++s) {
            if (s & (f - 1)) continue;
            const int tile = gi * geo.G + (s >> sh);
            if (tile >= geo.m_tiles) continue;
            const size_t cell = ((size_t)(gi * geo.n_tiles) * T_ACC + t_) * NUM_SLABS + s;
            const data_t* src = Cp + cell * CELL_FLOATS;         // i = 0 -> first COLS floats
            const int m0 = tile * MT + t_ * COLS;
            for (int c = 0; c < COLS; ++c)
              if (m0 + c < M) C[m0 + c] = src[c];
          }
    } else {
      const std::vector<data_t> Cpv(Cp, Cp + c_packed_floats(geo));
      std::vector<data_t> Cm((size_t)M * N, -1.f);
      unpack_c(Cpv, Cm, M, N, f);
      std::memcpy(C, Cm.data(), sizeof(data_t) * Cm.size());
    }
    t.unpack = usec(t7, clk::now());
  }

  size_t n_weight_sets() const { return wcache_.size(); }
  double upload_us_total() const { return upload_us_total_; }
  size_t hbm_weight_bytes() const { return hbm_bytes_; }

 private:
  struct BBuf { xrt::bo bo; std::vector<data_t> shadow; bool valid = false; };
  struct CBuf { xrt::bo bo; };

  BBuf& bbuf(int K, int N, int f) {
    const auto key = std::make_tuple(K, N, f);
    auto it = bcache_.find(key);
    if (it != bcache_.end()) return *it->second;
    const size_t floats = (size_t)ceil_div(N, f * SLAB_H) * K * ROWS;
    auto b = std::make_unique<BBuf>();
    b->bo = xrt::bo(dev_, sizeof(data_t) * floats, krn_.group_id(nA_));
    std::memset(b->bo.map<data_t*>(), 0, sizeof(data_t) * floats);
    b->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    b->shadow.assign((size_t)K * N, 0.f);
    return *(bcache_[key] = std::move(b));
  }
  CBuf& cbuf(size_t floats) {
    auto it = ccache_.find(floats);
    if (it != ccache_.end()) return *it->second;
    auto c = std::make_unique<CBuf>();
    c->bo = xrt::bo(dev_, sizeof(data_t) * floats, krn_.group_id(nA_ + 1));
    std::memset(c->bo.map<data_t*>(), 0, sizeof(data_t) * floats);
    c->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return *(ccache_[floats] = std::move(c));
  }

  xrt::device dev_;
  xrt::kernel krn_;
  bool is_sisa_ = true, b_reuse_ = true;
  int nA_ = 8;
  std::map<std::tuple<const float*, int, int>, std::unique_ptr<WeightSet>> wcache_;
  std::map<std::tuple<int, int, int>, std::unique_ptr<BBuf>> bcache_;
  std::map<size_t, std::unique_ptr<CBuf>> ccache_;
  size_t hbm_bytes_ = 0;
  double upload_us_total_ = 0;
};

// CPU reference in llama2.c's accumulation order (sequential over k, float).
static inline void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
      C[(size_t)m * N + n] = acc;
    }
}

// Same shape classes as the llama2.c matmuls of TinyStories-15M (dim 288, hidden 768, vocab 32000), made generic:
//   M == K          -> attn        (wq, wk, wv, wo: 288x288)
//   K < M <= 4K     -> ffn_up      (w1, w3: 768x288)
//   M < K           -> ffn_down    (w2: 288x768)
//   M > 4K          -> classifier  (wcls: 32000x288)
static inline const char* shape_label(int M, int K) {
  if (M == K) return "attn";
  if (M > 4 * K) return "classifier";
  if (M > K) return "ffn_up";
  return "ffn_down";
}

}  // namespace slm
