// tb_cosim.cpp -- tiny testbench for RTL co-simulation of gemm_sisa (make cosim-sisa). Cases cover f=1,2,4,8,
// several m-groups, a partial last group, multiple n-tiles. All buffers are padded to DEPTH beats because the
// m_axi pragmas declare depth=4096 (cosim reads/writes that much of each array).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../src/kernel_common.h"

extern "C" void gemm_sisa(const beat_t*, const beat_t*, const beat_t*, const beat_t*, const beat_t*, const beat_t*,
                          const beat_t*, const beat_t*, const beat_t*, beat_t*, int, int, int, int);

static const size_t DEPTH = 4096;   // beats; must match depth= in gemm_sisa.cpp

static int run_case(int M, int N, int K, int f) {
  std::vector<data_t> A((size_t)M * K), B((size_t)K * N), ref((size_t)M * N, 0.f), C((size_t)M * N, -1.f);
  for (auto& v : A) v = (data_t)(std::rand() % 17 - 8) * 0.125f;
  for (auto& v : B) v = (data_t)(std::rand() % 13 - 6) * 0.25f;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      data_t acc = 0;
      for (int k = 0; k < K; ++k) acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
      ref[(size_t)m * N + n] = acc;
    }
  const Geo g(M, N, f);
  const std::vector<data_t> Ap = pack_a(A, M, K), Bp = pack_b(B, K, N, f);
  const size_t FL = DEPTH * BEAT_FLOATS;   // floats per padded buffer
  if (Ap.size() > FL || Bp.size() > FL || c_packed_floats(g) > FL) { std::printf("case too big for DEPTH\n"); return 1000000; }
  std::vector<std::vector<data_t>> a(NUM_SLABS, std::vector<data_t>(FL, 0.f));
  for (auto& v : a) std::copy(Ap.begin(), Ap.end(), v.begin());
  std::vector<data_t> b(FL, 0.f), cp(FL, 0.f);
  std::copy(Bp.begin(), Bp.end(), b.begin());
#define P(i) reinterpret_cast<const beat_t*>(a[i].data())
  gemm_sisa(P(0), P(1), P(2), P(3), P(4), P(5), P(6), P(7), reinterpret_cast<const beat_t*>(b.data()),
            reinterpret_cast<beat_t*>(cp.data()), g.m_tiles, g.n_tiles, K, f);
  std::vector<data_t> cpv(cp.begin(), cp.begin() + c_packed_floats(g));
  unpack_c(cpv, C, M, N, f);
  int wrong = 0;
  for (size_t i = 0; i < ref.size(); ++i)
    if (!(std::fabs((double)C[i] - (double)ref[i]) <= 1e-3 + 1e-4 * std::fabs((double)ref[i]))) ++wrong;
  std::printf("case M=%d N=%d K=%d f=%d : %s (wrong=%d/%zu)\n", M, N, K, f, wrong ? "FAIL" : "ok", wrong, ref.size());
  return wrong;
}

int main() {
  std::srand(7);
  int bad = 0;
  bad += run_case(512, 1, 16, 1);    // f=1: one group of 8 m-tiles
  bad += run_case(576, 12, 8, 2);    // f=2: 4 groups of tiles, partial last group (9 tiles)
  bad += run_case(192, 30, 8, 4);    // f=4: 2 groups, 4 tiles per ... partial
  bad += run_case(128, 70, 8, 8);    // f=8: 2 n-tiles, 2 m-tiles
  std::printf("COSIM TB: %s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
