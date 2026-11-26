// fpga_matmul.cpp -- llama2.c hook: run.c's matmul() and forward() call into this file (see patch_run.py).
//
//   extern "C" void fpga_matmul(float* xout, float* x, float* w, int n, int d);   xout[d] = w[d][n] . x[n]
//   extern "C" void fpga_token_begin(int pos);                                    start of forward() for one token
//
// = GEMM with M = d, K = n, N = 1. Weights are cached in HBM by pointer on first sight (that happens during token 0,
// which run.c's own tok/s excludes, and which this file also excludes from its steady-state numbers).
//
// Settings (all optional). The gate rejects custom env vars, so they come from inputs/slm.conf (key=value lines, written by
// scripts/submit_slm.sh); for local runs the environment variable shown works too (env > conf > default):
//   xclbin / FPGA_XCLBIN  xclbin path                            (default ./design.xclbin)
//   FPGA_BREUSE         1: skip re-uploading an identical B     (default 1; wk/wv/w3 reuse wq's/w1's x). 0 = naive.
//   FPGA_F              SISA fusion factor, -1 = choose_fusion  (default -1; SA ignores it)
//   FPGA_VERIFY_TOKENS  check every matmul against the CPU for the first T tokens (default 0 = off)
//   FPGA_OUT_DIR        where slm_metrics.txt / slm_calls.csv / slm_tokens.csv go (default .)
//   FPGA_LABEL          free-text tag written into slm_metrics.txt
// stdout is NEVER written here (it carries the generated text, which is diffed against the CPU run); the summary
// goes to stderr and to the files above.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "fpga_backend.h"

using namespace slm;

namespace {

struct Rec {
  int tok;
  const char* label;
  int M, K;
  CallTimes t;
};

struct State {
  std::unique_ptr<Backend> be;
  std::string out_dir = ".", label;
  int fsel = -1, verify_tokens = 0;
  std::vector<Rec> calls;
  std::vector<double> tok_us, tok_mm_us;      // wall time of each token, and the part spent inside fpga_matmul
  int cur = -1;
  clk::time_point tok_t0;
  double mm_acc = 0;
  // verification
  long v_calls = 0, v_bad = 0;
  double v_max_abs = 0, v_max_rel = 0;
  bool reported = false;
};
State* S = nullptr;

// The gate only allows a few thread-count env vars, so settings also come from a key=value file shipped as an input
// (inputs/slm.conf, or $FPGA_CONF). Priority: environment variable (local runs) > conf file > default.
std::map<std::string, std::string> g_conf;
void load_conf() {
  static bool done = false;
  if (done) return;
  done = true;
  const char* path = std::getenv("FPGA_CONF");
  std::ifstream f(path ? path : "inputs/slm.conf");
  std::string line;
  while (std::getline(f, line)) {
    const size_t eq = line.find('=');
    if (eq == std::string::npos || line[0] == '#') continue;
    g_conf[line.substr(0, eq)] = line.substr(eq + 1);
  }
}
bool cfg(const char* key, const char* env, std::string& out) {
  load_conf();
  if (env && std::getenv(env)) { out = std::getenv(env); return true; }
  auto it = g_conf.find(key);
  if (it != g_conf.end()) { out = it->second; return true; }
  return false;
}

void report();

void close_token() {
  if (S->cur < 0) return;
  S->tok_us.push_back(usec(S->tok_t0, clk::now()));
  S->tok_mm_us.push_back(S->mm_acc);
  S->mm_acc = 0;
}

void init() {
  if (S) return;
  S = new State();                      // never destroyed: avoids XRT teardown order issues at exit
  std::string v;
  std::string xclbin = cfg("xclbin", "FPGA_XCLBIN", v) ? v : "./design.xclbin";
  const bool breuse = !cfg("breuse", "FPGA_BREUSE", v) || std::atoi(v.c_str()) != 0;
  if (cfg("f", "FPGA_F", v)) S->fsel = std::atoi(v.c_str());
  if (cfg("verify_tokens", "FPGA_VERIFY_TOKENS", v)) S->verify_tokens = std::atoi(v.c_str());
  if (cfg("out_dir", "FPGA_OUT_DIR", v)) S->out_dir = v;
  if (cfg("label", "FPGA_LABEL", v)) S->label = v;
  S->be = std::make_unique<Backend>(xclbin, breuse);
  std::fprintf(stderr, "[slm] kernel=%s xclbin=%s b_reuse=%d fsel=%d verify_tokens=%d\n", S->be->name(), xclbin.c_str(),
               (int)breuse, S->fsel, S->verify_tokens);
  std::atexit(report);
}

double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
double mean(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return v.empty() ? 0 : s / v.size();
}

void report() {
  if (!S || S->reported) return;
  S->reported = true;
  close_token();
  const int ntok = (int)S->tok_us.size();
  const int steady = ntok - 1;                       // token 0 = warm-up (device open, weight upload, first launches)
  std::ofstream met(S->out_dir + "/slm_metrics.txt");
  met << "design " << S->be->name() << "\nlabel " << S->label << "\n";
  met << "tokens_total " << ntok << "\nbreuse_hits_note see slm_calls.csv column b_reused\n";
  met << "weight_sets " << S->be->n_weight_sets() << "\nweights_hbm_MB " << S->be->hbm_weight_bytes() / 1048576.0
      << "\nweights_upload_ms " << S->be->upload_us_total() / 1e3 << "\n";
  if (steady < 1) { std::fprintf(stderr, "[slm] fewer than 2 tokens: no steady-state stats\n"); return; }

  double tsum = 0, mmsum = 0;
  for (int i = 1; i < ntok; ++i) { tsum += S->tok_us[i]; mmsum += S->tok_mm_us[i]; }
  const double tok_s = steady / (tsum * 1e-6);
  const double mm_ms = mmsum / steady / 1e3, cpu_ms = (tsum - mmsum) / steady / 1e3;
  met << "tokens_steady " << steady << "\ntok_s_internal " << tok_s << "\nms_per_token " << tsum / steady / 1e3
      << "\nms_per_token_matmul_total " << mm_ms << "\nms_per_token_cpu_other " << cpu_ms << "\n";
  met << "token0_ms " << S->tok_us[0] / 1e3 << "\n";

  // per-label breakdown over steady-state calls
  struct Agg { std::vector<double> dev, tot, pack, bsync, launch, wait, csync, unpack; long n = 0, reused = 0; int M = 0, K = 0, f = 0, steps = 0; };
  std::map<std::string, Agg> agg;
  for (const Rec& r : S->calls) {
    if (r.tok < 1) continue;
    Agg& a = agg[r.label];
    a.n++; a.reused += r.t.b_reused; a.M = r.M; a.K = r.K; a.f = r.t.f; a.steps = r.t.steps;
    a.dev.push_back(r.t.device()); a.tot.push_back(r.t.total()); a.pack.push_back(r.t.pack_x);
    a.bsync.push_back(r.t.b_sync); a.launch.push_back(r.t.launch); a.wait.push_back(r.t.wait);
    a.csync.push_back(r.t.c_sync); a.unpack.push_back(r.t.unpack);
  }
  std::fprintf(stderr, "\n[slm] %s  steady tokens=%d  %.2f tok/s  (%.3f ms/token: matmul %.3f ms + cpu-other %.3f ms)\n",
               S->be->name(), steady, tok_s, tsum / steady / 1e3, mm_ms, cpu_ms);
  std::fprintf(stderr, "[slm] weights: %zu sets, %.1f MB in HBM, upload %.1f ms (token 0, excluded)\n", S->be->n_weight_sets(),
               S->be->hbm_weight_bytes() / 1048576.0, S->be->upload_us_total() / 1e3);
  std::fprintf(stderr, "[slm] %-11s %6s %6s %3s %5s %5s | %9s %9s | %7s %7s %7s %7s %7s %7s | %9s %6s\n", "shape", "M", "K", "f",
               "steps", "calls", "dev_med", "dev_min", "pack_x", "b_sync", "launch", "wait", "c_sync", "unpack", "us/token", "%tok");
  for (auto& kv : agg) {
    Agg& a = kv.second;
    const double per_tok = mean(a.tot) * a.n / steady;
    std::fprintf(stderr, "[slm] %-11s %6d %6d %3d %5d %5ld | %9.1f %9.1f | %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f | %9.1f %5.1f%%\n",
                 kv.first.c_str(), a.M, a.K, a.f, a.steps, a.n / steady, median(a.dev), *std::min_element(a.dev.begin(), a.dev.end()),
                 mean(a.pack), mean(a.bsync), mean(a.launch), mean(a.wait), mean(a.csync), mean(a.unpack), per_tok,
                 100.0 * per_tok / (tsum / steady));
    const std::string k = kv.first;
    met << k << "_calls_per_token " << (double)a.n / steady << "\n"
        << k << "_dev_median_us " << median(a.dev) << "\n" << k << "_dev_min_us " << *std::min_element(a.dev.begin(), a.dev.end()) << "\n"
        << k << "_total_mean_us " << mean(a.tot) << "\n"
        << k << "_pack_x_mean_us " << mean(a.pack) << "\n" << k << "_b_sync_mean_us " << mean(a.bsync) << "\n"
        << k << "_launch_mean_us " << mean(a.launch) << "\n" << k << "_wait_mean_us " << mean(a.wait) << "\n"
        << k << "_c_sync_mean_us " << mean(a.csync) << "\n" << k << "_unpack_mean_us " << mean(a.unpack) << "\n"
        << k << "_b_reused_frac " << (double)a.reused / a.n << "\n"
        << k << "_us_per_token " << per_tok << "\n";
  }
  // host/launch overhead summary: everything in a matmul call that is not run.wait(), plus the CPU-only residue
  double launch_us = 0, pcie_us = 0, hostpack_us = 0, wait_us = 0;
  for (const Rec& r : S->calls) {
    if (r.tok < 1) continue;
    launch_us += r.t.launch; pcie_us += r.t.b_sync + r.t.c_sync; hostpack_us += r.t.pack_x + r.t.unpack; wait_us += r.t.wait;
  }
  met << "us_per_token_wait " << wait_us / steady << "\nus_per_token_launch " << launch_us / steady
      << "\nus_per_token_pcie_sync " << pcie_us / steady << "\nus_per_token_host_pack_unpack " << hostpack_us / steady
      << "\nus_per_token_cpu_other " << cpu_ms * 1e3 << "\n";
  std::fprintf(stderr, "[slm] per token: wait %.0f us | launch %.0f us | PCIe sync %.0f us | host pack/unpack %.0f us | CPU other %.0f us\n",
               wait_us / steady, launch_us / steady, pcie_us / steady, hostpack_us / steady, cpu_ms * 1e3);
  if (S->verify_tokens > 0) {
    met << "verify_calls " << S->v_calls << "\nverify_mismatches " << S->v_bad << "\nverify_max_abs_err " << S->v_max_abs
        << "\nverify_max_rel_err " << S->v_max_rel << "\n";
    std::fprintf(stderr, "[slm] verify (first %d tokens): %ld matmuls, %ld mismatching, max_abs_err %.3g max_rel_err %.3g\n",
                 S->verify_tokens, S->v_calls, S->v_bad, S->v_max_abs, S->v_max_rel);
  }
  // raw per-call / per-token CSVs
  std::ofstream cc(S->out_dir + "/slm_calls.csv");
  cc << "token,shape,M,K,f,steps,b_reused,pack_x_us,b_sync_us,launch_us,wait_us,c_sync_us,unpack_us,total_us,upload_us\n";
  for (const Rec& r : S->calls)
    cc << r.tok << "," << r.label << "," << r.M << "," << r.K << "," << r.t.f << "," << r.t.steps << "," << (int)r.t.b_reused << ","
       << r.t.pack_x << "," << r.t.b_sync << "," << r.t.launch << "," << r.t.wait << "," << r.t.c_sync << "," << r.t.unpack << ","
       << r.t.total() << "," << r.t.upload << "\n";
  std::ofstream tc(S->out_dir + "/slm_tokens.csv");
  tc << "token,token_us,matmul_us,cpu_other_us\n";
  for (int i = 0; i < ntok; ++i) tc << i << "," << S->tok_us[i] << "," << S->tok_mm_us[i] << "," << S->tok_us[i] - S->tok_mm_us[i] << "\n";
}

}  // namespace

// run.c asks for a prompt override (see patch_run.py); the conf key "prompt" avoids passing a quoted argument through the gate.
extern "C" const char* slm_prompt_override() {
  static std::string p;
  std::string v;
  if (cfg("prompt", "SLM_PROMPT", v)) { p = v; return p.c_str(); }
  return nullptr;
}

extern "C" void fpga_token_begin(int pos) {
  try {
    init();
    close_token();
    S->cur = pos;
    S->tok_t0 = clk::now();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "slm error (init): %s\n", e.what());
    std::exit(2);
  }
}

extern "C" void fpga_matmul(float* xout, float* x, float* w, int n, int d) {
  try {
    init();
    if (S->cur < 0) { S->cur = 0; S->tok_t0 = clk::now(); }       // matmul called without forward() hook
    Rec r;
    r.tok = S->cur;
    r.label = shape_label(d, n);
    r.M = d; r.K = n;
    S->be->gemm(w, d, n, x, 1, xout, r.t, S->fsel);
    S->mm_acc += r.t.total() + r.t.upload;
    if (S->cur < S->verify_tokens) {
      std::vector<float> ref(d);
      cpu_gemm(w, x, ref.data(), d, 1, n);
      for (int i = 0; i < d; ++i) {
        const double e = std::fabs((double)xout[i] - ref[i]);
        const double rel = e / (std::fabs((double)ref[i]) + 1e-12);
        S->v_max_abs = std::fmax(S->v_max_abs, e);
        if (e > 1e-6) S->v_max_rel = std::fmax(S->v_max_rel, rel);
        if (!(e <= 1e-3 + 1e-3 * std::fabs((double)ref[i]))) ++S->v_bad;
      }
      ++S->v_calls;
    }
    S->calls.push_back(r);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "slm error (matmul %dx%d): %s\n", d, n, e.what());
    std::exit(2);
  }
}
