// sweep_main.cpp -- decode-shaped GEMM sweep on the TinyStories-15M matmul shapes, through the SAME Backend the llama2.c
// hook uses (so layout, buffers and argument order are exercised identically), with the per-call breakdown.
//
//   ./slm_sweep [design.xclbin] [--shapes 288x288,768x288,288x768,32000x288] [--N 1,2,4,8,16,32] [--iters 15]
//               [--fs -1] [--nocheck] [--budget 200]
//   shape = MxK  (M = d output features, K = n inputs; A = weights M x K, B = activations K x N)
//
// Writes slm_sweep.csv (one row per shape x N, medians/min of every component) in the current directory, plus a
// RESULT line. B reuse is OFF here (every call uploads B), so numbers are comparable with host.cpp's sweep, but this
// also includes B/C PCIe sync + pack/unpack which host.cpp does not time.
// Exit codes as host.cpp: 0 PASS, 1 wrong result, 2 usage / XRT error.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "fpga_backend.h"

using namespace slm;

static std::vector<int> parse_list(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(std::atoi(tok.c_str()));
  return v;
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double minv(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }

int main(int argc, char** argv) {
  try {
    std::string xclbin = "./design.xclbin";
    std::vector<int> Ns = {1, 2, 4, 8, 16, 32};
    std::vector<std::pair<int, int>> shapes = {{288, 288}, {768, 288}, {288, 768}, {32000, 288}};
    int iters = 15, fsel = -1;
    double budget = 200;
    bool check = true;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--N" && i + 1 < argc) Ns = parse_list(argv[++i]);
      else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
      else if ((a == "--fs" || a == "--f") && i + 1 < argc) fsel = std::atoi(argv[++i]);
      else if (a == "--budget" && i + 1 < argc) budget = std::atof(argv[++i]);
      else if (a == "--nocheck") check = false;
      else if (a == "--shapes" && i + 1 < argc) {
        shapes.clear();
        std::stringstream ss(argv[++i]);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
          int m = 0, k = 0;
          if (std::sscanf(tok.c_str(), "%dx%d", &m, &k) != 2) { std::fprintf(stderr, "bad shape %s\n", tok.c_str()); return 2; }
          shapes.push_back({m, k});
        }
      } else if (!a.empty() && a[0] != '-') xclbin = a;
      else { std::fprintf(stderr, "usage: slm_sweep [design.xclbin] [--shapes MxK,..] [--N a,b] [--iters n] [--fs f] [--nocheck] [--budget s]\n"); return 2; }
    }
    const auto wall0 = clk::now();
    Backend be(xclbin, /*b_reuse=*/false);
    std::printf("kernel=%s iters=%d fs=%d\n", be.name(), iters, fsel);
    std::ofstream csv("slm_sweep.csv");
    csv << "design,shape,M,K,N,f,steps,dev_med_us,dev_min_us,total_med_us,pack_x_us,b_sync_us,launch_us,wait_us,c_sync_us,unpack_us,max_abs_err,pass\n";
    bool all_pass = true, truncated = false;
    for (auto [M, K] : shapes) {
      std::vector<float> A((size_t)M * K);
      std::srand(500 + M + K);
      for (auto& v : A) v = (float)(std::rand() % 17 - 8) * 0.125f;       // small multiples: FP32 exact up to order
      for (int N : Ns) {
        if (usec(wall0, clk::now()) * 1e-6 > budget) { truncated = true; break; }
        std::vector<float> B((size_t)K * N), C((size_t)M * N), ref;
        std::srand(900 + N + K);
        for (auto& v : B) v = (float)(std::rand() % 13 - 6) * 0.25f;
        CallTimes t;
        be.gemm(A.data(), M, K, B.data(), N, C.data(), t, fsel);           // warm-up + one-time weight upload
        be.gemm(A.data(), M, K, B.data(), N, C.data(), t, fsel);
        std::vector<double> dev, tot, pk, bs, la, wa, cs, up;
        for (int it = 0; it < iters; ++it) {
          be.gemm(A.data(), M, K, B.data(), N, C.data(), t, fsel);
          dev.push_back(t.device()); tot.push_back(t.total()); pk.push_back(t.pack_x); bs.push_back(t.b_sync);
          la.push_back(t.launch); wa.push_back(t.wait); cs.push_back(t.c_sync); up.push_back(t.unpack);
        }
        double max_err = 0;
        bool pass = true;
        if (check) {
          ref.resize((size_t)M * N);
          cpu_gemm(A.data(), B.data(), ref.data(), M, N, K);
          for (size_t i = 0; i < ref.size(); ++i) {
            const double e = std::fabs((double)C[i] - ref[i]);
            max_err = std::fmax(max_err, e);
            if (!(e <= 1e-3 + 1e-4 * std::fabs((double)ref[i]))) pass = false;
          }
        }
        all_pass = all_pass && pass;
        const char* lab = shape_label(M, K);
        std::printf("%-10s %5dx%-4d N=%-3d f=%d steps=%-3d dev med %8.1f min %8.1f | total %8.1f = pack %6.1f + bsync %6.1f + launch %6.1f + wait %8.1f + csync %7.1f + unpack %7.1f  err=%.3g %s\n",
                    lab, M, K, N, t.f, t.steps, median(dev), minv(dev), median(tot), median(pk), median(bs), median(la), median(wa),
                    median(cs), median(up), max_err, pass ? "ok" : "FAIL");
        csv << be.name() << "," << lab << "," << M << "," << K << "," << N << "," << t.f << "," << t.steps << "," << median(dev) << ","
            << minv(dev) << "," << median(tot) << "," << median(pk) << "," << median(bs) << "," << median(la) << "," << median(wa) << ","
            << median(cs) << "," << median(up) << "," << max_err << "," << (pass ? 1 : 0) << "\n";
      }
      if (truncated) break;
    }
    std::printf("RESULT: %s design=%s weights_hbm_MB=%.1f%s\n", all_pass ? "PASS" : "FAIL", be.name(), be.hbm_weight_bytes() / 1048576.0,
                truncated ? " (sweep truncated by --budget)" : "");
    return all_pass ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "slm_sweep error: %s\n", e.what());
    return 2;
  }
}
