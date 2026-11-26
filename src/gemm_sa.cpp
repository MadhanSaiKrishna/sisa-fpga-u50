// gemm_sa.cpp -- v2 BASELINE: conventional monolithic 64 x 8 output-stationary array (same 512 PEs, same dataflow
// loaders -> compute -> store, same 512-bit packed HBM access as gemm_sisa). Equivalent to SISA with f fixed to 8:
// ONE A port, every slab uses slab 0's A tile and its own 8-row block of B. No fusion muxes, no extra A ports.
// v2.4: operand distribution = tap chains (A: one stream, forwarded slab to slab; B: 4 beats per k, each slab keeps its
// slice); C leaves each slab through a register hop before the gather chain. A load FIFO is 256 deep like SISA's.
// v2.3: every slab is a nested dataflow region of 8 small column engines (slab_stages.h), slab inputs go through relay
// stages, and C leaves each slab as 512-bit beats through a daisy chain -> no die-spanning fan-out / wide nets.
// v2.2: every slab is its own process and C leaves through a daisy chain.
// v2.1: the A tile is replicated into NUM_SLABS independent streams (one per slab) so that each slab's operand registers
// have their own source; otherwise Vivado merges the identical per-multiplier registers into one net of fanout 128
// (route 7.8 ns across the SLR boundary, the v2 timing failure).
// Schedule: for m-tile: for nt: one step.   Output layout == gemm_sisa with f = 8 (use unpack_c(..., f=8)).
#include "kernel_common.h"
#define ITER_COUNTER g_iters_sa
#include "slab_stages.h"

#ifdef SIM_COUNT
unsigned long long g_iters_sa = 0;
#endif

static void load_a(const beat_t* A, hls::stream<beat_t>& out, int m_tiles, int n_tiles, int K) {
  const int nbeats = K * BEATS_PER_K;
  for (int tile = 0; tile < m_tiles; ++tile)
    for (int nt = 0; nt < n_tiles; ++nt) {
      const size_t base = (size_t)tile * nbeats;
      for (int b = 0; b < nbeats; ++b) {
#pragma HLS PIPELINE II=1
        out.write(A[base + b]);                       // one stream; the tap chain hands it to every slab
      }
    }
}

static void load_b(const beat_t* B, hls::stream<beat_t>& out, int m_tiles, int n_tiles, int K) {
  const int nbeats = K * BEATS_PER_ROW;               // B row-major per n-tile: one contiguous burst per step
  for (int tile = 0; tile < m_tiles; ++tile)
    for (int nt = 0; nt < n_tiles; ++nt) {
      const size_t base = (size_t)nt * nbeats;
      for (int b = 0; b < nbeats; ++b) {
#pragma HLS PIPELINE II=1
        out.write(B[base + b]);
      }
    }
}

static void sa_core(const beat_t* A, const beat_t* B, beat_t* C, int m_tiles, int n_tiles, int K, int steps) {
#pragma HLS DATAFLOW
  const int total = steps * T_ACC;
  hls::stream<beat_t> as, bs;
  hls::stream<beat_t> af[NUM_SLABS - 1], bf[NUM_SLABS - 1];
  hls::stream<beat_t> ar[NUM_SLABS];
  hls::stream<bslice_t> bsl[NUM_SLABS];
  hls::stream<beat_t> cb[NUM_SLABS], cbr[NUM_SLABS];
  hls::stream<beat_t> ch[NUM_SLABS - 1], chr[NUM_SLABS - 2];
#pragma HLS STREAM variable=as depth=256
#pragma HLS STREAM variable=bs depth=256
#pragma HLS STREAM variable=af depth=4
#pragma HLS STREAM variable=bf depth=4
#pragma HLS STREAM variable=ar depth=8
#pragma HLS STREAM variable=bsl depth=4
#pragma HLS STREAM variable=cb depth=16
#pragma HLS STREAM variable=cbr depth=4
#pragma HLS STREAM variable=ch depth=16
#pragma HLS STREAM variable=chr depth=4
  const int na = steps * K * BEATS_PER_K, nb = steps * K * BEATS_PER_ROW;
  const int nc = total * BEATS_PER_CELL;

  load_a(A, as, m_tiles, n_tiles, K);
  load_b(B, bs, m_tiles, n_tiles, K);
  a_head(as, ar[0], af[0], na);
  a_mid(af[0], ar[1], af[1], na);
  a_mid(af[1], ar[2], af[2], na);
  a_mid(af[2], ar[3], af[3], na);
  a_mid(af[3], ar[4], af[4], na);
  a_mid(af[4], ar[5], af[5], na);
  a_mid(af[5], ar[6], af[6], na);
  a_end(af[6], ar[7], na);
  b_head(bs, bsl[0], bf[0], 0, nb);
  b_mid(bf[0], bsl[1], bf[1], 1, nb);
  b_mid(bf[1], bsl[2], bf[2], 2, nb);
  b_mid(bf[2], bsl[3], bf[3], 3, nb);
  b_mid(bf[3], bsl[4], bf[4], 4, nb);
  b_mid(bf[4], bsl[5], bf[5], 5, nb);
  b_mid(bf[5], bsl[6], bf[6], 6, nb);
  b_end(bf[6], bsl[7], 7, nb);
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

extern "C" void gemm_sa(const beat_t* A, const beat_t* B, beat_t* C, int m_tiles, int n_tiles, int K) {
#pragma HLS INTERFACE m_axi port=A bundle=gmemA offset=slave depth=4096 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B bundle=gmemB offset=slave depth=4096 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C bundle=gmemC offset=slave depth=4096 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=m_tiles bundle=control
#pragma HLS INTERFACE s_axilite port=n_tiles bundle=control
#pragma HLS INTERFACE s_axilite port=K bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
  const int steps = m_tiles * n_tiles;
  sa_core(A, B, C, m_tiles, n_tiles, K, steps);
}
