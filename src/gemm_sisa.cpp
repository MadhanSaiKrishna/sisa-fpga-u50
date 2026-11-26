// gemm_sisa.cpp -- v2 SISA slab engine: 8 slabs x 8 rows x 8 columns, runtime fusion factor f in {1,2,4,8}.
//
// Dataflow (all tasks run concurrently, connected by FIFOs; HBM latency/bursts hidden behind compute):
//   load_a x8  : leader slabs stream their A tile (packed, 512-bit beats) from their own HBM port
//   load_b     : streams B rows (64 floats per k) for the current n-tile
//   a_gmid..   : (v2.4) A tap chain: each slab takes its beat from its group leader's port or from slab s-1 and forwards it
//   b_mid..    : (v2.4) B tap chain: 4 beats per k travel slab to slab; slab s keeps slice (s & (f-1))
//   relay      : (v2.4) register hop on every C stream before the gather chain
//   slab_compute x8 : (v2.3) per slab: a_split -> 8 column engines (8 MACs, own enable) -> c_comb; per step: clear -> K*T_ACC MACs -> drain
//   gather chain + store_c : (v2.3) C beats serialised slab by slab, one II=1 burst copy to HBM
// Fusion: with factor f, G = 8/f groups of Rg = f*8 token rows. Slab s belongs to group s/f, uses A of its
// group leader (s & ~(f-1)) and the B rows [(s&(f-1))*8, +8). f=8 is one monolithic 64-row array.
// Schedule: for g in groups (G m-tiles at a time): for nt in n_tiles: one "step".
#include "kernel_common.h"
#define ITER_COUNTER g_iters
#include "slab_stages.h"

#ifdef SIM_COUNT
unsigned long long g_iters = 0;
#endif

static void load_a(const beat_t* A, hls::stream<beat_t>& out, int slab, int groups, int n_tiles,
                   int m_tiles, int K, int sh) {
  const int G = NUM_SLABS >> sh;
  const bool lead = (slab & ((1 << sh) - 1)) == 0;   // only group leaders feed compute
  const int nbeats = K * BEATS_PER_K;
  if (lead) {
    for (int g = 0; g < groups; ++g) {
      const int tile = g * G + (slab >> sh);
      for (int nt = 0; nt < n_tiles; ++nt) {
        if (tile < m_tiles) {
          const size_t base = (size_t)tile * nbeats;
          for (int b = 0; b < nbeats; ++b) {
#pragma HLS PIPELINE II=1
            out.write(A[base + b]);
          }
        } else {
          for (int b = 0; b < nbeats; ++b) {
#pragma HLS PIPELINE II=1
            out.write(beat_t(0));
          }
        }
      }
    }
  }
}

static void load_b(const beat_t* B, hls::stream<beat_t>& out, int groups, int n_tiles, int K) {
  const int nbeats = K * BEATS_PER_ROW;               // B row-major per n-tile: one contiguous burst per step
  for (int g = 0; g < groups; ++g)
    for (int nt = 0; nt < n_tiles; ++nt) {
      const size_t base = (size_t)nt * nbeats;
      for (int b = 0; b < nbeats; ++b) {
#pragma HLS PIPELINE II=1
        out.write(B[base + b]);
      }
    }
}

static void sisa_core(const beat_t* A0, const beat_t* A1, const beat_t* A2, const beat_t* A3,
                      const beat_t* A4, const beat_t* A5, const beat_t* A6, const beat_t* A7,
                      const beat_t* B, beat_t* C, int m_tiles, int n_tiles, int K, int sh, int groups, int steps) {
#pragma HLS DATAFLOW
  const int total = steps * T_ACC;
  hls::stream<beat_t> as0, as1, as2, as3, as4, as5, as6, as7, bs;
  hls::stream<beat_t> af[NUM_SLABS - 1], bf[NUM_SLABS - 1];
  hls::stream<beat_t> ar[NUM_SLABS];
  hls::stream<beat_t> sm[NUM_SLABS - 2];   // v2.5: a_gsel -> a_gfan for slabs 1..6
  hls::stream<bslice_t> bsl[NUM_SLABS];
  hls::stream<beat_t> cb[NUM_SLABS], cbr[NUM_SLABS];
  hls::stream<beat_t> ch[NUM_SLABS - 1], chr[NUM_SLABS - 2];
#pragma HLS STREAM variable=as0 depth=256
#pragma HLS STREAM variable=as1 depth=256
#pragma HLS STREAM variable=as2 depth=256
#pragma HLS STREAM variable=as3 depth=256
#pragma HLS STREAM variable=as4 depth=256
#pragma HLS STREAM variable=as5 depth=256
#pragma HLS STREAM variable=as6 depth=256
#pragma HLS STREAM variable=as7 depth=256
#pragma HLS STREAM variable=bs depth=256
#pragma HLS STREAM variable=af depth=4
#pragma HLS STREAM variable=sm depth=4
#pragma HLS STREAM variable=bf depth=4
#pragma HLS STREAM variable=ar depth=8
#pragma HLS STREAM variable=bsl depth=4
#pragma HLS STREAM variable=cb depth=16
#pragma HLS STREAM variable=cbr depth=4
#pragma HLS STREAM variable=ch depth=16
#pragma HLS STREAM variable=chr depth=4
  const int na = steps * K * BEATS_PER_K, nb = steps * K * BEATS_PER_ROW;
  const int nc = total * BEATS_PER_CELL;
  const int mask = (1 << sh) - 1;
  // slab s: leader iff (s & mask) == 0 ; it forwards its beat iff slab s+1 is a non-leader (same group)
  const bool lead1 = (1 & mask) == 0, lead2 = (2 & mask) == 0, lead3 = (3 & mask) == 0, lead4 = (4 & mask) == 0,
             lead5 = (5 & mask) == 0, lead6 = (6 & mask) == 0, lead7 = (7 & mask) == 0;
  const int j0 = 0, j1 = 1 & mask, j2 = 2 & mask, j3 = 3 & mask, j4 = 4 & mask, j5 = 5 & mask, j6 = 6 & mask,
            j7 = 7 & mask;

  load_a(A0, as0, 0, groups, n_tiles, m_tiles, K, sh);
  load_a(A1, as1, 1, groups, n_tiles, m_tiles, K, sh);
  load_a(A2, as2, 2, groups, n_tiles, m_tiles, K, sh);
  load_a(A3, as3, 3, groups, n_tiles, m_tiles, K, sh);
  load_a(A4, as4, 4, groups, n_tiles, m_tiles, K, sh);
  load_a(A5, as5, 5, groups, n_tiles, m_tiles, K, sh);
  load_a(A6, as6, 6, groups, n_tiles, m_tiles, K, sh);
  load_a(A7, as7, 7, groups, n_tiles, m_tiles, K, sh);
  load_b(B, bs, groups, n_tiles, K);
  a_ghead(as0, ar[0], af[0], !lead1, na);
  a_gsel(as1, af[0], sm[0], lead1, na);
  a_gfan(sm[0], ar[1], af[1], !lead2, na);
  a_gsel(as2, af[1], sm[1], lead2, na);
  a_gfan(sm[1], ar[2], af[2], !lead3, na);
  a_gsel(as3, af[2], sm[2], lead3, na);
  a_gfan(sm[2], ar[3], af[3], !lead4, na);
  a_gsel(as4, af[3], sm[3], lead4, na);
  a_gfan(sm[3], ar[4], af[4], !lead5, na);
  a_gsel(as5, af[4], sm[4], lead5, na);
  a_gfan(sm[4], ar[5], af[5], !lead6, na);
  a_gsel(as6, af[5], sm[5], lead6, na);
  a_gfan(sm[5], ar[6], af[6], !lead7, na);
  a_gend(as7, af[6], ar[7], lead7, na);
  b_head(bs, bsl[0], bf[0], j0, nb);
  b_mid(bf[0], bsl[1], bf[1], j1, nb);
  b_mid(bf[1], bsl[2], bf[2], j2, nb);
  b_mid(bf[2], bsl[3], bf[3], j3, nb);
  b_mid(bf[3], bsl[4], bf[4], j4, nb);
  b_mid(bf[4], bsl[5], bf[5], j5, nb);
  b_mid(bf[5], bsl[6], bf[6], j6, nb);
  b_end(bf[6], bsl[7], j7, nb);
  slab_compute(ar[0], bsl[0], cb[0], steps, K, 0);
  slab_compute(ar[1], bsl[1], cb[1], steps, K, 1);
  slab_compute(ar[2], bsl[2], cb[2], steps, K, 2);
  slab_compute(ar[3], bsl[3], cb[3], steps, K, 3);
  slab_compute(ar[4], bsl[4], cb[4], steps, K, 4);
  slab_compute(ar[5], bsl[5], cb[5], steps, K, 5);
  slab_compute(ar[6], bsl[6], cb[6], steps, K, 6);
  slab_compute(ar[7], bsl[7], cb[7], steps, K, 7);
  relay<beat_t>(cb[0], cbr[0], nc);
  relay<beat_t>(cb[1], cbr[1], nc);
  relay<beat_t>(cb[2], cbr[2], nc);
  relay<beat_t>(cb[3], cbr[3], nc);
  relay<beat_t>(cb[4], cbr[4], nc);
  relay<beat_t>(cb[5], cbr[5], nc);
  relay<beat_t>(cb[6], cbr[6], nc);
  relay<beat_t>(cb[7], cbr[7], nc);
  gather_n(cbr[0], cbr[1], ch[0], total, 4);
  relay<beat_t>(ch[0], chr[0], total * 8);
  gather_n(chr[0], cbr[2], ch[1], total, 8);
  relay<beat_t>(ch[1], chr[1], total * 12);
  gather_n(chr[1], cbr[3], ch[2], total, 12);
  relay<beat_t>(ch[2], chr[2], total * 16);
  gather_n(chr[2], cbr[4], ch[3], total, 16);
  relay<beat_t>(ch[3], chr[3], total * 20);
  gather_n(chr[3], cbr[5], ch[4], total, 20);
  relay<beat_t>(ch[4], chr[4], total * 24);
  gather_n(chr[4], cbr[6], ch[5], total, 24);
  relay<beat_t>(ch[5], chr[5], total * 28);
  gather_n(chr[5], cbr[7], ch[6], total, 28);
  store_c(C, ch[6], total);
}

extern "C" void gemm_sisa(const beat_t* A0, const beat_t* A1, const beat_t* A2, const beat_t* A3,
                          const beat_t* A4, const beat_t* A5, const beat_t* A6, const beat_t* A7,
                          const beat_t* B, beat_t* C, int m_tiles, int n_tiles, int K, int f) {
#pragma HLS INTERFACE m_axi port=A0 bundle=gmemA0 offset=slave depth=4096 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=A1 bundle=gmemA1 offset=slave depth=4096 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=A2 bundle=gmemA2 offset=slave depth=4096 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=A3 bundle=gmemA3 offset=slave depth=4096 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=A4 bundle=gmemA4 offset=slave depth=4096 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=A5 bundle=gmemA5 offset=slave depth=4096 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=A6 bundle=gmemA6 offset=slave depth=4096 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=A7 bundle=gmemA7 offset=slave depth=4096 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B  bundle=gmemB  offset=slave depth=4096 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C  bundle=gmemC  offset=slave depth=4096 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A0 bundle=control
#pragma HLS INTERFACE s_axilite port=A1 bundle=control
#pragma HLS INTERFACE s_axilite port=A2 bundle=control
#pragma HLS INTERFACE s_axilite port=A3 bundle=control
#pragma HLS INTERFACE s_axilite port=A4 bundle=control
#pragma HLS INTERFACE s_axilite port=A5 bundle=control
#pragma HLS INTERFACE s_axilite port=A6 bundle=control
#pragma HLS INTERFACE s_axilite port=A7 bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=m_tiles bundle=control
#pragma HLS INTERFACE s_axilite port=n_tiles bundle=control
#pragma HLS INTERFACE s_axilite port=K bundle=control
#pragma HLS INTERFACE s_axilite port=f bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
  // f must be 1,2,4 or 8 (host picks it); anything else is treated as 8 = monolithic
  const int sh = (f == 1) ? 0 : (f == 2) ? 1 : (f == 4) ? 2 : 3;
  const int G = NUM_SLABS >> sh;
  const int groups = (m_tiles + G - 1) / G;
  const int steps = groups * n_tiles;
  sisa_core(A0, A1, A2, A3, A4, A5, A6, A7, B, C, m_tiles, n_tiles, K, sh, groups, steps);
}
