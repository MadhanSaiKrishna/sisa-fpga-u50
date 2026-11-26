// common.h -- v2 hardware shape + host-visible layouts, shared by BOTH kernels and the host. Plain C++ only
// (no Vitis headers here, so the XRT host can include it).
//
// NAMING:  C[M x N] = A[M x K] * B[K x N]   (A = weights, B = activations)
//   SISA paper's small/skewed dimension (tokens) is OUR N  ->  array ROWS <-> N,  array COLS <-> M.
//   PE(r,c) owns C[m][n] in a local accumulator (output-stationary); only A and B values move.
//
// v2 ARCHITECTURE (what changed vs v1, see docs/v2_plan.md):
//   * 64 x 8 array = NUM_SLABS(8) slabs of SLAB_H(8) rows -> up to 8x slab parallelism (paper: 8 slabs)
//   * dataflow: loaders -> compute -> store run concurrently, A and B STREAM from HBM (no A buffer, no K limit)
//   * 512-bit HBM ports, host pre-packs A / B and un-packs C so every HBM access is one long burst
//   * the baseline SA is the SAME engine with fusion fixed to f = NUM_SLABS and a single A port
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

typedef float data_t;

#define COLS 8        // PE columns = output features per step
#define T_ACC 8       // accumulators per PE (>= float add latency so the MAC loop runs at II=1)
#define SLAB_H 8      // rows (tokens) per slab
#define NUM_SLABS 8   // slabs (power of two, fixed: gemm_sisa has NUM_SLABS explicit A ports)
#define ROWS (SLAB_H * NUM_SLABS)   // 64
#define MT (COLS * T_ACC)           // 64 output features per m-tile
#define BEAT_FLOATS 16              // one 512-bit HBM beat
#define CELL_FLOATS (SLAB_H * COLS) // C element = one slab's results for one accumulator slot t: [i][c]

static_assert(NUM_SLABS == 8, "kernels are written for 8 slabs (A0..A7)");
static_assert(ROWS == 64 && MT == 64 && CELL_FLOATS == 64, "layouts assume 64-float rows/cells (4 beats)");
static_assert((T_ACC & (T_ACC - 1)) == 0 && T_ACC >= 8, "T_ACC: power of two >= 8");
static_assert(2 * COLS == BEAT_FLOATS, "one beat = two accumulator slots of COLS floats");

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// SISA fusion: smallest power-of-two f with f*SLAB_H >= N, capped at NUM_SLABS.
static inline int choose_fusion(int N) {
  int f = 1;
  while (f < NUM_SLABS && f * SLAB_H < N) f <<= 1;
  return f;
}
static inline int fusion_shift(int f) { int s = 0; while ((1 << s) < f) ++s; return s; }

// ---- schedule geometry for a given (M, N, f): shared by kernels' counters, tb, host --------------------
struct Geo {
  int m_tiles, G, Rg, groups, n_tiles, steps;
  Geo(int M, int N, int f) {
    m_tiles = ceil_div(M, MT);
    G = NUM_SLABS / f;                  // independent slab groups = m-tiles processed per step
    Rg = f * SLAB_H;                    // token rows per group
    groups = ceil_div(m_tiles, G);
    n_tiles = ceil_div(N, Rg);
    steps = groups * n_tiles;           // schedule: for g in groups: for nt in n_tiles
  }
  long long compute_iters(int K) const { return (long long)steps * K * T_ACC; }
};

// ---- host-side packing (weights/activations packed once; results unpacked after the run) --------------
// A_packed[((tile*K + k)*T_ACC + t)*COLS + c] = A[tile*MT + t*COLS + c][k]          (zero padded)
static inline std::vector<data_t> pack_a(const std::vector<data_t>& A, int M, int K) {
  const int tiles = ceil_div(M, MT);
  std::vector<data_t> P((size_t)tiles * K * T_ACC * COLS, 0.f);
  for (int tile = 0; tile < tiles; ++tile)
    for (int k = 0; k < K; ++k)
      for (int t = 0; t < T_ACC; ++t)
        for (int c = 0; c < COLS; ++c) {
          const int m = tile * MT + t * COLS + c;
          if (m < M) P[(((size_t)tile * K + k) * T_ACC + t) * COLS + c] = A[(size_t)m * K + k];
        }
  return P;
}
// B_packed[(nt*K + k)*ROWS + r] = B[k][nt*Rg + r]   for r < Rg and n < N (else 0); Rg depends on f
static inline std::vector<data_t> pack_b(const std::vector<data_t>& B, int K, int N, int f) {
  const int Rg = f * SLAB_H, nts = ceil_div(N, Rg);
  std::vector<data_t> P((size_t)nts * K * ROWS, 0.f);
  for (int nt = 0; nt < nts; ++nt)
    for (int k = 0; k < K; ++k)
      for (int r = 0; r < Rg; ++r) {
        const int n = nt * Rg + r;
        if (n < N) P[((size_t)nt * K + k) * ROWS + r] = B[(size_t)k * N + n];
      }
  return P;
}
// C_packed[(((step*T_ACC + t)*NUM_SLABS + s)*CELL_FLOATS) + i*COLS + c]  with step = g*n_tiles + nt,
//   tile = g*G + s/f, m = tile*MT + t*COLS + c, n = nt*Rg + (s%f)*SLAB_H + i
static inline size_t c_packed_floats(const Geo& g) { return (size_t)g.steps * T_ACC * NUM_SLABS * CELL_FLOATS; }
static inline void unpack_c(const std::vector<data_t>& Cp, std::vector<data_t>& C, int M, int N, int f) {
  const Geo g(M, N, f);
  const int sh = fusion_shift(f);
  for (int gi = 0; gi < g.groups; ++gi)
    for (int nt = 0; nt < g.n_tiles; ++nt)
      for (int t = 0; t < T_ACC; ++t)
        for (int s = 0; s < NUM_SLABS; ++s) {
          const int tile = gi * g.G + (s >> sh);
          const int ro = (s & (f - 1)) * SLAB_H;
          for (int i = 0; i < SLAB_H; ++i)
            for (int c = 0; c < COLS; ++c) {
              const int m = tile * MT + t * COLS + c, n = nt * g.Rg + ro + i;
              if (tile < g.m_tiles && m < M && n < N)
                C[(size_t)m * N + n] =
                    Cp[((((size_t)(gi * g.n_tiles + nt) * T_ACC + t) * NUM_SLABS + s) * CELL_FLOATS) + i * COLS + c];
            }
        }
}
