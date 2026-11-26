// tb.cpp -- C testbench: Vitis HLS csim AND plain g++ (`make csim`, uses test/shim). Checks both kernels against a
// golden model for many (M,N,K) and EVERY fusion factor, through the same pack/unpack code as the host.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../src/kernel_common.h"

extern "C" void gemm_sa(const beat_t*, const beat_t*, beat_t*, int, int, int);
extern "C" void gemm_sisa(const beat_t*, const beat_t*, const beat_t*, const beat_t*, const beat_t*, const beat_t*,
                          const beat_t*, const beat_t*, const beat_t*, beat_t*, int, int, int, int);

static void golden(const std::vector<data_t>& A, const std::vector<data_t>& B, std::vector<data_t>& C, int M, int N, int K) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      data_t acc = 0;
      for (int k = 0; k < K; ++k) acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
      C[(size_t)m * N + n] = acc;
    }
}
static int check(const char* tag, const std::vector<data_t>& got, const std::vector<data_t>& ref, int M, int N, int K, int f) {
  int bad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double tol = 1e-3 + 1e-4 * std::fabs((double)ref[i]);
    if (!(std::fabs((double)got[i] - (double)ref[i]) <= tol)) {
      if (bad < 3) std::printf("  MISMATCH %s M=%d N=%d K=%d f=%d idx=%zu got=%g ref=%g\n", tag, M, N, K, f, i, got[i], ref[i]);
      ++bad;
    }
  }
  return bad;
}
#define BP(v) reinterpret_cast<const beat_t*>((v).data())

int main() {
  struct Shape { int M, N, K; };
  const Shape shapes[] = {{64, 1, 32},   {128, 1, 288}, {288, 1, 288},  {300, 1, 100},  {288, 3, 288},  {288, 8, 288},
                          {288, 9, 288}, {768, 16, 288}, {288, 17, 288}, {288, 32, 96},  {200, 33, 96},  {130, 64, 40},
                          {768, 65, 24}, {300, 130, 20}, {1000, 5, 64},  {1, 1, 1},      {7, 31, 5},     {64, 64, 1200}};
  int fails = 0, runs = 0;
  std::srand(1234);
  for (const Shape& s : shapes) {
    const int M = s.M, N = s.N, K = s.K;
    std::vector<data_t> A((size_t)M * K), B((size_t)K * N), ref((size_t)M * N), C((size_t)M * N);
    for (auto& v : A) v = (data_t)(std::rand() % 17 - 8) * 0.125f;
    for (auto& v : B) v = (data_t)(std::rand() % 13 - 6) * 0.25f;
    golden(A, B, ref, M, N, K);
    const std::vector<data_t> Ap = pack_a(A, M, K);

    {  // baseline == f = 8 layout
      const Geo g(M, N, NUM_SLABS);
      const std::vector<data_t> Bp = pack_b(B, K, N, NUM_SLABS);
      std::vector<data_t> Cp(c_packed_floats(g), -1.f);
#ifdef SIM_COUNT
      g_iters_sa = 0;
#endif
      gemm_sa(BP(Ap), BP(Bp), reinterpret_cast<beat_t*>(Cp.data()), g.m_tiles, g.n_tiles, K);
      std::fill(C.begin(), C.end(), -1.f);
      unpack_c(Cp, C, M, N, NUM_SLABS);
      int bad = check("sa", C, ref, M, N, K, 0);
#ifdef SIM_COUNT
      if (g_iters_sa != (unsigned long long)g.compute_iters(K)) { std::printf("  ITER sa %llu vs %lld\n", g_iters_sa, g.compute_iters(K)); ++bad; }
#endif
      std::printf("[%s] sa      M=%-4d N=%-4d K=%-4d\n", bad ? "FAIL" : "ok", M, N, K);
      fails += bad; ++runs;
    }
    for (int f = 1; f <= NUM_SLABS; f <<= 1) {
      const Geo g(M, N, f);
      const std::vector<data_t> Bp = pack_b(B, K, N, f);
      std::vector<data_t> Cp(c_packed_floats(g), -1.f);
#ifdef SIM_COUNT
      g_iters = 0;
#endif
      gemm_sisa(BP(Ap), BP(Ap), BP(Ap), BP(Ap), BP(Ap), BP(Ap), BP(Ap), BP(Ap), BP(Bp),
                reinterpret_cast<beat_t*>(Cp.data()), g.m_tiles, g.n_tiles, K, f);
      std::fill(C.begin(), C.end(), -1.f);
      unpack_c(Cp, C, M, N, f);
      int bad = check("sisa", C, ref, M, N, K, f);
#ifdef SIM_COUNT
      if (g_iters != (unsigned long long)g.compute_iters(K)) { std::printf("  ITER sisa %llu vs %lld\n", g_iters, g.compute_iters(K)); ++bad; }
#endif
      std::printf("[%s] sisa    M=%-4d N=%-4d K=%-4d f=%d%s\n", bad ? "FAIL" : "ok", M, N, K, f, f == choose_fusion(N) ? "  (auto)" : "");
      fails += bad; ++runs;
    }
  }
  std::printf("%s: %d runs, %d failures\n", fails ? "TEST FAILED" : "TEST PASSED", runs, fails);
  return fails ? 1 : 0;
}
