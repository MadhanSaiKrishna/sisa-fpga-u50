// host.cpp -- XRT host for BOTH designs (gemm_sa and gemm_sisa), written to the course fpga-client contract
// (fpga-client/AGENTS.md):
//   * xclbin path = first non-option argument, else ./design.xclbin   (the gate does NOT pass it)
//   * opens the card as xrt::device(0)
//   * writes sweep.csv / metrics.txt to the CURRENT directory (that directory is returned in out/)
//   * kernel time = std::chrono around run.wait() only (buffers already synced), median over --iters
//   * golden check on the CPU folded into the exit code and metrics ("pass 0/1")
//   * exit 0 = PASS, 1 = wrong result, 2 = usage / IO / XRT error  (same codes as the kit's test_projects)
//   * one-line "RESULT: ..." on stdout (ends up in meta/stdout.txt)
//   * whole run must stay under the gate's 5-minute cap: --budget S stops the sweep early (default 200 s)
//
//   ./host [design.xclbin] [--M 768] [--K 288] [--iters 5] [--N 1,2,4,8,16,32,64,128] [--fs -1,8] [--nocheck] [--probe]
//
// --probe: after the sweep, time a few tiny / scaled shapes to separate the fixed launch+start-up cost from the per-step cost
//          (writes probe.csv and probe_* keys in metrics.txt):  K scaling (1 step), step scaling (K=288), SISA f=1 group scaling.
//
// --decode: decoder-layer replay (per-token GEMM sequence of one transformer layer, shapes from --D model dim, --F ffn dim):
//           QKV (3D x D), attn-out (D x D), gate+up (2F x D), down (D x F) at each N in --N, timed (a) as separate
//           blocking calls and (b) as --L layers x 4 GEMMs enqueued back to back with ONE wait (chained). Inputs are static
//           (no C->B repack between GEMMs: device time only). Writes decode.csv and decode_* keys in metrics.txt.
//
// --fs: SISA fusion factors to run per N. -1 = host picks choose_fusion(N) (adaptive SISA);
//       8 = all slabs fused = SISA hardware used as one monolithic array. Ignored for the baseline (= f 8).
// Inputs are generated in-process (no --input files needed): fixed-seed small-integer-multiple floats, so the
// FP32 result is exact up to accumulation order and the check tolerance is tight.
//
// Build on an RCE COMPUTE node (glibc <= 2.31 required by the gate):  make host
//   = source /opt/xilinx/xrt/setup.sh; g++ -std=c++17 -O2 host/host.cpp -o build/host -I$XILINX_XRT/include \
//       -L$XILINX_XRT/lib -lxrt_coreutil -lpthread -Wl,--allow-shlib-undefined
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "experimental/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "../src/common.h"

static std::vector<int> parse_list(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(std::atoi(tok.c_str()));
  return v;
}

static void golden(const data_t* A, const data_t* B, data_t* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      data_t acc = 0;
      for (int k = 0; k < K; ++k) acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
      C[(size_t)m * N + n] = acc;
    }
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// One timed point on freshly packed buffers (probe mode). Returns pass/fail of the golden check.
static bool probe_point(xrt::device& dev, xrt::kernel& krn, bool is_sisa, int M, int K, int N, int f_req, int iters,
                        double& med_us, double& min_us) {
  const int f = is_sisa ? (f_req > 0 ? f_req : choose_fusion(N)) : NUM_SLABS;
  const Geo geo(M, N, f);
  std::vector<data_t> Afull((size_t)M * K), B((size_t)K * N), ref((size_t)M * N);
  std::srand(77 + M + K + N);
  for (auto& v : Afull) v = (data_t)(std::rand() % 17 - 8) * 0.125f;
  for (auto& v : B) v = (data_t)(std::rand() % 13 - 6) * 0.25f;
  golden(Afull.data(), B.data(), ref.data(), M, N, K);
  const std::vector<data_t> Ap = pack_a(Afull, M, K), Bp = pack_b(B, K, N, f);
  const size_t a_bytes = sizeof(data_t) * Ap.size(), b_bytes = sizeof(data_t) * Bp.size();
  const size_t c_bytes = sizeof(data_t) * c_packed_floats(geo);
  const int nA = is_sisa ? NUM_SLABS : 1;
  std::vector<xrt::bo> boA;
  for (int i = 0; i < nA; ++i) {
    boA.emplace_back(dev, a_bytes, krn.group_id(i));
    std::memcpy(boA[i].map<data_t*>(), Ap.data(), a_bytes);
    boA[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  xrt::bo boB(dev, b_bytes, krn.group_id(nA)), boC(dev, c_bytes, krn.group_id(nA + 1));
  std::memcpy(boB.map<data_t*>(), Bp.data(), b_bytes);
  boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  std::memset(boC.map<data_t*>(), 0, c_bytes);
  boC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto launch = [&]() {
    if (is_sisa) {
      auto run = krn(boA[0], boA[1], boA[2], boA[3], boA[4], boA[5], boA[6], boA[7], boB, boC, geo.m_tiles, geo.n_tiles, K, f);
      run.wait();
    } else {
      auto run = krn(boA[0], boB, boC, geo.m_tiles, geo.n_tiles, K);
      run.wait();
    }
  };
  launch();
  std::vector<double> us;
  for (int it = 0; it < iters; ++it) {
    auto t0 = std::chrono::high_resolution_clock::now();
    launch();
    auto t1 = std::chrono::high_resolution_clock::now();
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  med_us = median(us);
  min_us = *std::min_element(us.begin(), us.end());
  boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::vector<data_t> Cp(c_packed_floats(geo)), C((size_t)M * N, -1.f);
  std::memcpy(Cp.data(), boC.map<data_t*>(), c_bytes);
  unpack_c(Cp, C, M, N, f);
  for (size_t i = 0; i < ref.size(); ++i)
    if (!(std::fabs((double)C[i] - (double)ref[i]) <= 1e-3 + 1e-4 * std::fabs((double)ref[i]))) return false;
  return true;
}


// ---- decode-layer replay -------------------------------------------------------------------------------------
struct Gemm { const char* name; int M, K; };

static bool decode_bench(xrt::device& dev, xrt::kernel& krn, bool is_sisa, const std::vector<int>& Ns, const std::vector<int>& Fs,
                         int D, int F, int L, int iters, bool do_check, double budget_s,
                         std::chrono::steady_clock::time_point wall0, std::ofstream& met, bool& truncated) {
  const char* name = is_sisa ? "gemm_sisa" : "gemm_sa";
  const std::vector<Gemm> gl = {{"qkv", 3 * D, D}, {"attn_out", D, D}, {"gate_up", 2 * F, D}, {"down", D, F}};
  std::ofstream csv("decode.csv");
  csv << "design,D,F,N,f,gemm,M,K,steps,median_us,min_us,pass\n";
  std::ofstream lcsv("decode_layer.csv");
  lcsv << "design,D,F,N,f,mode,layers,per_layer_median_us,per_layer_min_us,per_gemm_avg_us\n";
  const int nA = is_sisa ? NUM_SLABS : 1;
  bool all_ok = true;
  // A (weights) packed once per GEMM, shared by every N
  std::vector<std::vector<data_t>> Af(gl.size());
  std::vector<std::vector<xrt::bo>> boAs(gl.size());
  for (size_t gi = 0; gi < gl.size(); ++gi) {
    Af[gi].resize((size_t)gl[gi].M * gl[gi].K);
    std::srand(500 + (int)gi);
    for (auto& v : Af[gi]) v = (data_t)(std::rand() % 17 - 8) * 0.125f;
    const std::vector<data_t> Ap = pack_a(Af[gi], gl[gi].M, gl[gi].K);
    const size_t a_bytes = sizeof(data_t) * Ap.size();
    for (int i = 0; i < nA; ++i) {
      boAs[gi].emplace_back(dev, a_bytes, krn.group_id(i));
      std::memcpy(boAs[gi][i].map<data_t*>(), Ap.data(), a_bytes);
      boAs[gi][i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
  }
  for (int N : Ns) {
    for (int fsel : Fs) {
      const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
      if (elapsed > budget_s) { truncated = true; return all_ok; }
      const int f = is_sisa ? (fsel > 0 ? fsel : choose_fusion(N)) : NUM_SLABS;
      struct Item { Geo geo; xrt::bo boB, boC; std::vector<data_t> B, ref; int M, K; };
      std::vector<Item> it;
      for (size_t gi = 0; gi < gl.size(); ++gi) {
        const int M = gl[gi].M, K = gl[gi].K;
        Geo geo(M, N, f);
        std::vector<data_t> B((size_t)K * N);
        std::srand(900 + N + (int)gi);
        for (auto& v : B) v = (data_t)(std::rand() % 13 - 6) * 0.25f;
        const std::vector<data_t> Bp = pack_b(B, K, N, f);
        const size_t b_bytes = sizeof(data_t) * Bp.size(), c_bytes = sizeof(data_t) * c_packed_floats(geo);
        xrt::bo boB(dev, b_bytes, krn.group_id(nA)), boC(dev, c_bytes, krn.group_id(nA + 1));
        std::memcpy(boB.map<data_t*>(), Bp.data(), b_bytes);
        boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        std::memset(boC.map<data_t*>(), 0, c_bytes);
        boC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        std::vector<data_t> ref;
        if (do_check) { ref.resize((size_t)M * N); golden(Af[gi].data(), B.data(), ref.data(), M, N, K); }
        it.push_back({geo, boB, boC, B, ref, M, K});
      }
      auto start = [&](size_t gi) {
        auto& A = boAs[gi];
        auto& x = it[gi];
        if (is_sisa) return krn(A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], x.boB, x.boC, x.geo.m_tiles, x.geo.n_tiles, x.K, f);
        return krn(A[0], x.boB, x.boC, x.geo.m_tiles, x.geo.n_tiles, x.K);
      };
      // (a) blocking, per GEMM
      double sum_med = 0, sum_min = 0;
      std::vector<double> layer_us;
      for (int rep = 0; rep < 2; ++rep) for (size_t gi = 0; gi < gl.size(); ++gi) start(gi).wait();   // warm-up
      std::vector<std::vector<double>> per(gl.size());
      for (int i = 0; i < iters; ++i) {
        double layer = 0;
        for (size_t gi = 0; gi < gl.size(); ++gi) {
          auto t0 = std::chrono::high_resolution_clock::now();
          start(gi).wait();
          auto t1 = std::chrono::high_resolution_clock::now();
          const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
          per[gi].push_back(us);
          layer += us;
        }
        layer_us.push_back(layer);
      }
      for (size_t gi = 0; gi < gl.size(); ++gi) {
        auto& x = it[gi];
        const double med = median(per[gi]), mn = *std::min_element(per[gi].begin(), per[gi].end());
        sum_med += med; sum_min += mn;
        bool ok = true;
        if (do_check) {
          x.boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
          std::vector<data_t> Cp(c_packed_floats(x.geo)), C((size_t)x.M * N, -1.f);
          std::memcpy(Cp.data(), x.boC.map<data_t*>(), sizeof(data_t) * Cp.size());
          unpack_c(Cp, C, x.M, N, f);
          for (size_t q = 0; q < x.ref.size(); ++q)
            if (!(std::fabs((double)C[q] - (double)x.ref[q]) <= 1e-3 + 1e-4 * std::fabs((double)x.ref[q]))) { ok = false; break; }
        }
        all_ok = all_ok && ok;
        csv << name << "," << D << "," << F << "," << N << "," << f << "," << gl[gi].name << "," << x.M << "," << x.K << ","
            << x.geo.steps << "," << med << "," << mn << "," << (ok ? 1 : 0) << "\n";
        std::printf("DECODE %-8s N=%-3d f=%d M=%-5d K=%-5d steps=%-3d median=%.1f us min=%.1f us %s\n", gl[gi].name, N, f, x.M, x.K, x.geo.steps, med, mn, ok ? "ok" : "FAIL");
      }
      const double lmed = median(layer_us), lmin = *std::min_element(layer_us.begin(), layer_us.end());
      lcsv << name << "," << D << "," << F << "," << N << "," << f << ",blocking,1," << lmed << "," << lmin << "," << lmed / gl.size() << "\n";
      // (b) chained: L layers x 4 GEMMs enqueued back to back, one wait
      std::vector<double> ch;
      for (int i = 0; i < iters + 1; ++i) {
        std::vector<xrt::run> runs;
        runs.reserve((size_t)L * gl.size());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int l = 0; l < L; ++l) for (size_t gi = 0; gi < gl.size(); ++gi) runs.push_back(start(gi));
        for (auto& r : runs) r.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        if (i > 0) ch.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / L);   // first = warm-up
      }
      const double cmed = median(ch), cmin = *std::min_element(ch.begin(), ch.end());
      lcsv << name << "," << D << "," << F << "," << N << "," << f << ",chained," << L << "," << cmed << "," << cmin << "," << cmed / gl.size() << "\n";
      std::printf("DECODE layer N=%-3d f=%d blocking=%.1f us (min %.1f)  chained(%d layers)=%.1f us/layer (min %.1f)\n", N, f, lmed, lmin, L, cmed, cmin);
      met << "decode_N" << N << "_f" << f << "_blocking_layer_us " << lmed << "\n";
      met << "decode_N" << N << "_f" << f << "_chained_layer_us " << cmed << "\n";
    }
  }
  return all_ok;
}

static int run_main(int argc, char** argv) {
  const auto wall0 = std::chrono::steady_clock::now();
  std::string xclbin = "./design.xclbin";
  int M = 768, K = 288, iters = 5;
  double budget_s = 200;
  bool do_check = true, probe = false, decode = false;
  int D = 512, F = 1408, L = 12;
  std::vector<int> Ns = {1, 2, 4, 8, 16, 32, 64, 128};
  std::vector<int> Fs = {-1, 8};
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--M" && i + 1 < argc) M = std::atoi(argv[++i]);
    else if (a == "--K" && i + 1 < argc) K = std::atoi(argv[++i]);
    else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--N" && i + 1 < argc) Ns = parse_list(argv[++i]);
    else if ((a == "--fs" || a == "--f") && i + 1 < argc) Fs = parse_list(argv[++i]);
    else if (a == "--budget" && i + 1 < argc) budget_s = std::atof(argv[++i]);
    else if (a == "--nocheck") do_check = false;
    else if (a == "--probe") probe = true;
    else if (a == "--decode") decode = true;
    else if (a == "--D" && i + 1 < argc) D = std::atoi(argv[++i]);
    else if (a == "--F" && i + 1 < argc) F = std::atoi(argv[++i]);
    else if (a == "--L" && i + 1 < argc) L = std::atoi(argv[++i]);
    else if (!a.empty() && a[0] != '-') xclbin = a;
    else { std::fprintf(stderr, "usage: host [design.xclbin] [--M --K --iters --N a,b --fs a,b --budget s --nocheck --probe --decode --D --F --L]\n"); return 2; }
  }
  if (K < 1 || M < 1 || iters < 1) { std::fprintf(stderr, "bad M/K/iters\n"); return 2; }

  auto dev = xrt::device(0);                       // inside the sandbox only the U50 is visible
  auto uuid = dev.load_xclbin(xclbin);
  bool is_sisa = true;
  xrt::kernel krn;
  try {
    krn = xrt::kernel(dev, uuid, "gemm_sisa");
  } catch (...) {
    is_sisa = false;
    krn = xrt::kernel(dev, uuid, "gemm_sa");
  }
  const char* name = is_sisa ? "gemm_sisa" : "gemm_sa";
  if (!is_sisa) Fs = {NUM_SLABS};   // baseline = monolithic = f 8 layout
  std::printf("kernel=%s ROWS=%d COLS=%d T_ACC=%d SLAB_H=%d NUM_SLABS=%d M=%d K=%d iters=%d\n", name, ROWS, COLS, T_ACC, SLAB_H, NUM_SLABS, M, K, iters);

  std::ofstream csv("sweep.csv");
  csv << "design,M,N,K,f,median_us,min_us,gflops_median,max_abs_err,pass\n";
  std::ofstream met("metrics.txt");
  met << "design " << name << "\nM " << M << "\nK " << K << "\n";

  bool all_pass = true, truncated = false;
  double first_ms = -1;
  // A is packed once (weights are static); SISA gets one full copy per slab port, each in its own HBM bank
  // (see connectivity_sisa.cfg); the baseline gets one copy on one port.
  std::vector<data_t> Afull((size_t)M * K);
  std::srand(99);
  for (auto& v : Afull) v = (data_t)(std::rand() % 17 - 8) * 0.125f;
  const std::vector<data_t> Ap = pack_a(Afull, M, K);
  const size_t a_bytes = sizeof(data_t) * Ap.size();
  const int nA = is_sisa ? NUM_SLABS : 1;
  std::vector<xrt::bo> boA;
  for (int i = 0; i < nA; ++i) {
    boA.emplace_back(dev, a_bytes, krn.group_id(i));
    std::memcpy(boA[i].map<data_t*>(), Ap.data(), a_bytes);
    boA[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  for (int N : Ns) {
    std::vector<data_t> B((size_t)K * N), ref;
    std::srand(1234 + N);
    for (auto& v : B) v = (data_t)(std::rand() % 13 - 6) * 0.25f;
    if (do_check) { ref.resize((size_t)M * N); golden(Afull.data(), B.data(), ref.data(), M, N, K); }

    for (int fsel : Fs) {
      const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
      if (elapsed > budget_s) { truncated = true; break; }
      const int f = is_sisa ? (fsel > 0 ? fsel : choose_fusion(N)) : NUM_SLABS;
      const Geo geo(M, N, f);
      const std::vector<data_t> Bp = pack_b(B, K, N, f);
      const size_t b_bytes = sizeof(data_t) * Bp.size();
      const size_t c_bytes = sizeof(data_t) * c_packed_floats(geo);
      xrt::bo boB(dev, b_bytes, krn.group_id(nA));
      xrt::bo boC(dev, c_bytes, krn.group_id(nA + 1));
      std::memcpy(boB.map<data_t*>(), Bp.data(), b_bytes);
      boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      std::memset(boC.map<data_t*>(), 0, c_bytes);
      boC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      auto launch = [&]() {
        if (is_sisa) {
          auto run = krn(boA[0], boA[1], boA[2], boA[3], boA[4], boA[5], boA[6], boA[7], boB, boC, geo.m_tiles, geo.n_tiles, K, f);
          run.wait();
        } else {
          auto run = krn(boA[0], boB, boC, geo.m_tiles, geo.n_tiles, K);
          run.wait();
        }
      };
      launch();                                                    // warm-up (not timed)
      std::vector<double> us;
      for (int it = 0; it < iters; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        launch();
        auto t1 = std::chrono::high_resolution_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
      const double med = median(us), mn = *std::min_element(us.begin(), us.end());
      boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      std::vector<data_t> Cp(c_packed_floats(geo));
      std::memcpy(Cp.data(), boC.map<data_t*>(), c_bytes);
      std::vector<data_t> C((size_t)M * N, -1.f);
      unpack_c(Cp, C, M, N, f);
      double max_err = 0;
      bool pass = true;
      if (do_check) {
        for (size_t i = 0; i < ref.size(); ++i) {
          const double e = std::fabs((double)C[i] - (double)ref[i]);
          max_err = std::fmax(max_err, e);
          if (!(e <= 1e-3 + 1e-4 * std::fabs((double)ref[i]))) pass = false;
        }
      }
      if (!pass && do_check) {
        long wrong = 0, shown = 0;
        for (int m = 0; m < M; ++m)
          for (int n = 0; n < N; ++n) {
            const size_t i = (size_t)m * N + n;
            if (std::fabs((double)C[i] - (double)ref[i]) <= 1e-3 + 1e-4 * std::fabs((double)ref[i])) continue;
            ++wrong;
            if (shown < 6) { std::printf("  DIAG mismatch m=%d n=%d got=%g ref=%g\n", m, n, (double)C[i], (double)ref[i]); ++shown; }
          }
        std::printf("  DIAG N=%d f=%d wrong=%ld of %zu\n", N, f, wrong, ref.size());
      }
      const double gflops = 2.0 * M * N * K / (med * 1e3);
      if (first_ms < 0) first_ms = med / 1e3;
      std::printf("N=%-4d f=%d median=%.1f us min=%.1f us  %.2f GFLOP/s  max_err=%.3g  %s\n", N, f, med, mn, gflops, max_err, pass ? "ok" : "FAIL");
      csv << name << "," << M << "," << N << "," << K << "," << f << "," << med << "," << mn << "," << gflops << "," << max_err << "," << (pass ? 1 : 0) << "\n";
      met << "N" << N << "_f" << f << "_time_us " << med << "\n";
      met << "N" << N << "_f" << f << "_gflops " << gflops << "\n";
      all_pass = all_pass && pass;
    }
    if (truncated) break;
  }
  if (probe && !truncated) {
    std::ofstream pcsv("probe.csv");
    pcsv << "design,M,K,N,f,steps,median_us,min_us,pass\n";
    struct P { int M, K, N, f; };
    std::vector<P> pts;
    const int fl = is_sisa ? 1 : NUM_SLABS;                       // small-N point of each design
    pts.push_back({64, 1, 1, fl});                                // launch floor: ~one 16-cycle step
    for (int k : {8, 36, 72, 144, 288}) pts.push_back({64, k, 1, fl});          // K scaling, one step
    for (int m : {64, 128, 256, 512, 768}) pts.push_back({m, 288, 64, NUM_SLABS});  // step scaling (f = 8)
    if (is_sisa) for (int m : {512, 1024, 2048}) pts.push_back({m, 288, 1, 1});  // SISA f=1: 1, 2, 4 groups
    const int pit = std::max(iters, 7);
    for (const P& p : pts) {
      const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
      if (elapsed > budget_s) { truncated = true; break; }
      const Geo g(p.M, p.N, is_sisa ? p.f : NUM_SLABS);
      double med = 0, mn = 0;
      const bool ok = probe_point(dev, krn, is_sisa, p.M, p.K, p.N, p.f, pit, med, mn);
      std::printf("PROBE M=%d K=%d N=%d f=%d steps=%d median=%.1f us min=%.1f us %s\n", p.M, p.K, p.N, p.f, g.steps, med, mn, ok ? "ok" : "FAIL");
      pcsv << name << "," << p.M << "," << p.K << "," << p.N << "," << p.f << "," << g.steps << "," << med << "," << mn << "," << (ok ? 1 : 0) << "\n";
      met << "probe_M" << p.M << "_K" << p.K << "_N" << p.N << "_f" << p.f << "_us " << med << "\n";
      all_pass = all_pass && ok;
    }
  }
  if (decode && !truncated) {
    const bool ok = decode_bench(dev, krn, is_sisa, Ns, Fs, D, F, L, std::max(iters, 15), do_check, budget_s, wall0, met, truncated);
    all_pass = all_pass && ok;
  }
  met << "kernel_ms " << first_ms << "\n";            // first (N, f) point; the per-point keys above are the real data
  met << "pass " << (all_pass ? 1 : 0) << "\n";
  met << "truncated " << (truncated ? 1 : 0) << "\n";
  std::printf("RESULT: %s design=%s M=%d K=%d kernel=%.4f ms%s\n", all_pass ? "PASS" : "FAIL", name, M, K, first_ms, truncated ? " (sweep truncated by --budget)" : "");
  return all_pass ? 0 : 1;
}

int main(int argc, char** argv) {
  try {
    return run_main(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "host error: %s\n", e.what());
    return 2;
  }
}
