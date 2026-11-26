// slab_stages.h -- v2.4 (v2.3 stages + tap-chain operand distribution + C register hops) shared dataflow stages for gemm_sa and gemm_sisa (HLS processes, one FIFO hop apart).
//
// WHY (post-route worst paths of v2 / v2.1 / v2.2; the 200 MHz request auto-scales to the 10 ns / ~100 MHz pass,
//      and 85-98 % of every worst path was routing, not logic):
//   v2   SISA : one HLS pipeline-enable register drove 4136 loads (9.3 ns net).      SA : store_c enable + 32:1 mux.
//   v2.2 SA   : slab MAC-loop pipeline enable (ap_block_pp0_stage0_subdone...) -> CE of 64 multipliers = fanout 6696,
//               8.25 ns net, crossing the SLR boundary.
//   v2.2 SISA : an fmul operand register (din0_buf1) with fanout 16, 7.9 ns net, crossing the SLR boundary.
//   Both critical paths sit INSIDE one slab (the slab the placer happened to cut across the die boundary).
//
// v2.3 CHANGE (implementation only -- same 64x8 array, same schedule, same cycle counts per step):
//   * a slab is no longer one 64-MAC loop with one enable; it is a small dataflow region of
//       a_split -> 8 column engines (8 MACs each, own pipeline/enable/operand registers) -> c_comb
//     => every pipeline-enable net now has ~1/8 of the loads, every operand register ~1/8 of the sinks, and a slab
//        consists of 8 small, individually placeable blocks instead of one 320-DSP block.
//   * relay stages (read->write, II=1) on every slab input stream so the die-boundary hop is flop -> flop.
//   * C leaves each slab as 512-bit beats (4 per cell) straight from c_comb; the old 2048-bit row FIFO and the 4:1
//     beat select in gather_n are gone (a 2048-bit hop across the die would need 4x the SLLs).
//   Principle: AutoBridge FPGA'21 / TAPA (pipeline every cut, small local control), Systolic-CNN FPGA'19 / AutoSA
//   FPGA'21 (limit fan-out, local control), and the Interim result that independent units with local control and only
//   registered data crossings keep their clock when split over SLRs (O3 vs O2e).
// Functional behaviour (data layout, schedule, MAC iteration counts) is IDENTICAL to v2.2.
#pragma once
#include "kernel_common.h"

typedef ap_uint<2 * 32> apair_t;   // {hi, lo} A values of one PE column for two consecutive accumulator slots

// ---------------------------------------------------------------------------------------------------------
// relay: one register stage (plus FIFO) on a stream that may cross the die boundary.
// ---------------------------------------------------------------------------------------------------------
template <typename T>
static void relay(hls::stream<T>& in, hls::stream<T>& out, int n) {
#pragma HLS INLINE off
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    out.write(in.read());
  }
}

// ---------------------------------------------------------------------------------------------------------
// Inside a slab. A beat (512 bit) carries A of two accumulator slots for all 8 columns: lo = floats 0..7 (even t),
// hi = floats 8..15 (odd t). a_split hands column c its (lo,hi) pair; b_dup hands every column engine the slab's
// 8 B values of the current k (separate registers per engine, so Vivado cannot merge them into one wide net).
// ---------------------------------------------------------------------------------------------------------
static void a_split(hls::stream<beat_t>& a, hls::stream<apair_t> out[COLS], int steps, int K) {
#pragma HLS INLINE off
  const int n = steps * K * BEATS_PER_K;
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    const beat_t v = a.read();
    for (int c = 0; c < COLS; ++c) {
#pragma HLS UNROLL
      apair_t p;
      p.range(31, 0) = v.range(32 * c + 31, 32 * c).to_uint();
      p.range(63, 32) = v.range(32 * (COLS + c) + 31, 32 * (COLS + c)).to_uint();
      out[c].write(p);
    }
  }
}

static void b_dup(hls::stream<bslice_t>& bs, hls::stream<bslice_t> out[COLS], int steps, int K) {
#pragma HLS INLINE off
  const int n = steps * K;
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    const bslice_t v = bs.read();
    for (int c = 0; c < COLS; ++c) {
#pragma HLS UNROLL
      out[c].write(v);
    }
  }
}

// One PE column of a slab: 8 rows x T_ACC accumulators, 8 MACs per cycle (II=1).
// Per step: clear (T_ACC cycles) -> K*T_ACC MAC cycles -> drain (T_ACC cycles, one 8-float column piece per cycle).
// ITER_COUNTER (csim only) counts slab 0 / column 0 MAC iterations (== Geo::compute_iters).
static void col_engine(hls::stream<apair_t>& a, hls::stream<bslice_t>& bs, hls::stream<bslice_t>& cout,
                       int steps, int K, int slab, int col) {
#pragma HLS INLINE off
  (void)slab; (void)col;
  data_t acc[SLAB_H][T_ACC];
#pragma HLS ARRAY_PARTITION variable=acc dim=1 complete
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram
  data_t bv[SLAB_H];
#pragma HLS ARRAY_PARTITION variable=bv complete
  apair_t acur = 0;

  for (int st = 0; st < steps; ++st) {
    for (int t = 0; t < T_ACC; ++t) {
#pragma HLS PIPELINE II=1
      for (int i = 0; i < SLAB_H; ++i) acc[i][t] = 0;
    }
    const int total = K * T_ACC;
    for (int kt = 0; kt < total; ++kt) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc type=inter direction=RAW distance=8 dependent=true
      const int t = kt & (T_ACC - 1);
      if (t == 0) {                                   // new k: this slab's 8 B values
        const bslice_t r = bs.read();
        for (int i = 0; i < SLAB_H; ++i) {
#pragma HLS UNROLL
          bv[i] = u2f(r.range(32 * i + 31, 32 * i).to_uint());
        }
      }
      if ((t & 1) == 0) acur = a.read();              // new pair = A of two accumulator slots
      const bool hi = (t & 1) != 0;
      const data_t lo_v = u2f(acur.range(31, 0).to_uint());
      const data_t hi_v = u2f(acur.range(63, 32).to_uint());
      const data_t av = hi ? hi_v : lo_v;
      for (int i = 0; i < SLAB_H; ++i) {
#pragma HLS UNROLL
        acc[i][t] = acc[i][t] + av * bv[i];
      }
#ifdef SIM_COUNT
      if (slab == 0 && col == 0) ++ITER_COUNTER;
#endif
    }
    for (int t = 0; t < T_ACC; ++t) {
#pragma HLS PIPELINE II=1
      bslice_t e;
      for (int i = 0; i < SLAB_H; ++i) {
#pragma HLS UNROLL
        e.range(32 * i + 31, 32 * i) = f2u(acc[i][t]);
      }
      cout.write(e);
    }
  }
}

// Reassembles one C cell (64 floats, layout [i][c]) from the 8 column pieces and emits it as 4 beats of 16 floats:
// beat q holds rows i = 2q and 2q+1, all 8 columns  (== the v2 packed C layout).
static void c_comb(hls::stream<bslice_t> cin[COLS], hls::stream<beat_t>& cout, int steps) {
#pragma HLS INLINE off
  bslice_t piece[COLS];
#pragma HLS ARRAY_PARTITION variable=piece complete
  const int n = steps * T_ACC * BEATS_PER_CELL;
  int q = 0;
  for (int it = 0; it < n; ++it) {
#pragma HLS PIPELINE II=1
    if (q == 0) {
      for (int c = 0; c < COLS; ++c) {
#pragma HLS UNROLL
        piece[c] = cin[c].read();
      }
    }
    beat_t bq[BEATS_PER_CELL];
#pragma HLS ARRAY_PARTITION variable=bq complete
    for (int qq = 0; qq < BEATS_PER_CELL; ++qq) {
#pragma HLS UNROLL
      for (int j = 0; j < BEAT_FLOATS; ++j) {
#pragma HLS UNROLL
        const int c = j % COLS, i = 2 * qq + j / COLS;
        bq[qq].range(32 * j + 31, 32 * j) = piece[c].range(32 * i + 31, 32 * i).to_uint();
      }
    }
    cout.write((q == 0) ? bq[0] : (q == 1) ? bq[1] : (q == 2) ? bq[2] : bq[3]);
    q = (q == BEATS_PER_CELL - 1) ? 0 : q + 1;
  }
}

// ---------------------------------------------------------------------------------------------------------
// One slab = a_split -> 8 column engines -> c_comb, as a nested dataflow region.
// ---------------------------------------------------------------------------------------------------------
static void slab_compute(hls::stream<beat_t>& a, hls::stream<bslice_t>& bs, hls::stream<beat_t>& cout,
                         int steps, int K, int slab) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  hls::stream<apair_t> ac[COLS];
  hls::stream<bslice_t> bc[COLS], cc[COLS];
#pragma HLS STREAM variable=ac depth=4
#pragma HLS STREAM variable=bc depth=4
#pragma HLS STREAM variable=cc depth=8
  a_split(a, ac, steps, K);
  b_dup(bs, bc, steps, K);
  col_engine(ac[0], bc[0], cc[0], steps, K, slab, 0);
  col_engine(ac[1], bc[1], cc[1], steps, K, slab, 1);
  col_engine(ac[2], bc[2], cc[2], steps, K, slab, 2);
  col_engine(ac[3], bc[3], cc[3], steps, K, slab, 3);
  col_engine(ac[4], bc[4], cc[4], steps, K, slab, 4);
  col_engine(ac[5], bc[5], cc[5], steps, K, slab, 5);
  col_engine(ac[6], bc[6], cc[6], steps, K, slab, 6);
  col_engine(ac[7], bc[7], cc[7], steps, K, slab, 7);
  c_comb(cc, cout, steps);
}

// ---------------------------------------------------------------------------------------------------------
// v2.4 operand distribution = TAP CHAINS (replaces the v2.3 8-way A broadcast, a_route 8:1 select and b_route 8-way
// write). Post-route worst paths of v2.3: SA = the A bus written to 8 slab FIFOs by one register (fanout 8, SLR
// crossing, 97 % routing); SISA = read enable of a 512-bit C FIFO driven through LUT2->LUT6(fo 524)->LUT4.
// A tap chain: slab s reads its beat from its group leader's A port (lead) or from slab s-1 (forwarded), writes it to
// its own slab FIFO and forwards it to slab s+1 when that slab is in the same group. Every stage touches only its
// neighbours, so every pipeline enable has fanout 2 FIFOs and every crossing is flop -> flop.
// B tap chain: the B row of one k travels as 4 beats (beat q = slices 2q, 2q+1); slab s keeps slice jsel (SA: s;
// SISA: s & (f-1)) and forwards every beat. Same data and same slab inputs as v2.3 -- only the wiring changed.
// ---------------------------------------------------------------------------------------------------------
// SA (f = 8, one A port): slab 0 = head, slabs 1..6 forward, slab 7 = end. Pure forwarding, no runtime flags.
static void a_head(hls::stream<beat_t>& in, hls::stream<beat_t>& loc, hls::stream<beat_t>& fout, int n) {
#pragma HLS INLINE off
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    const beat_t v = in.read();
    loc.write(v);
    fout.write(v);
  }
}
static void a_mid(hls::stream<beat_t>& fin, hls::stream<beat_t>& loc, hls::stream<beat_t>& fout, int n) {
#pragma HLS INLINE off
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    const beat_t v = fin.read();
    loc.write(v);
    fout.write(v);
  }
}
static void a_end(hls::stream<beat_t>& fin, hls::stream<beat_t>& loc, int n) {
#pragma HLS INLINE off
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    loc.write(fin.read());
  }
}
// SISA (runtime fusion): lead = this slab is a group leader (reads its own A port), fwd = the next slab is a
// non-leader of the same group (needs this beat). Streams of a role that is not active are neither read nor written.
static void a_ghead(hls::stream<beat_t>& own, hls::stream<beat_t>& loc, hls::stream<beat_t>& fout, bool fwd, int n) {
#pragma HLS INLINE off
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    const beat_t v = own.read();                      // slab 0 is always a leader
    loc.write(v);
    if (fwd) fout.write(v);
  }
}
static void a_gmid(hls::stream<beat_t>& own, hls::stream<beat_t>& fin, hls::stream<beat_t>& loc,
                   hls::stream<beat_t>& fout, bool lead, bool fwd, int n) {
#pragma HLS INLINE off
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    beat_t v;
    if (lead) v = own.read();
    else v = fin.read();
    loc.write(v);
    if (fwd) fout.write(v);
  }
}
// v2.5: a_gmid split into two stages so that no stage has more than 3 FIFO flags in its stall logic (v2.4 had four:
// two inputs + two outputs -> one 512-bit enable with fanout ~500-600 that the placer stretched across SLRs).
// a_gsel: pick this slab's beat (own port if leader, else upstream tap) -> one stream.  Streams of the inactive role are not touched.
static void a_gsel(hls::stream<beat_t>& own, hls::stream<beat_t>& fin, hls::stream<beat_t>& sel, bool lead, int n) {
#pragma HLS INLINE off
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    beat_t v;
    if (lead) v = own.read();
    else v = fin.read();
    sel.write(v);
  }
}
// a_gfan: deliver the selected beat to the local slab and (if the next slab is a non-leader of the group) forward it.
static void a_gfan(hls::stream<beat_t>& sel, hls::stream<beat_t>& loc, hls::stream<beat_t>& fout, bool fwd, int n) {
#pragma HLS INLINE off
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    const beat_t v = sel.read();
    loc.write(v);
    if (fwd) fout.write(v);
  }
}
static void a_gend(hls::stream<beat_t>& own, hls::stream<beat_t>& fin, hls::stream<beat_t>& loc, bool lead, int n) {
#pragma HLS INLINE off
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    beat_t v;
    if (lead) v = own.read();
    else v = fin.read();
    loc.write(v);
  }
}

// B: n = steps*K*BEATS_PER_ROW beats; the slab keeps beat q == (jsel >> 1), half (jsel & 1).
static inline void b_capture(const beat_t& v, int q, int jsel, hls::stream<bslice_t>& loc) {
  if (q == (jsel >> 1)) {
    bslice_t s;
    if (jsel & 1) s = v.range(511, 256);
    else s = v.range(255, 0);
    loc.write(s);
  }
}
static void b_head(hls::stream<beat_t>& in, hls::stream<bslice_t>& loc, hls::stream<beat_t>& fout, int jsel, int n) {
#pragma HLS INLINE off
  int q = 0;
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    const beat_t v = in.read();
    b_capture(v, q, jsel, loc);
    fout.write(v);
    q = (q == BEATS_PER_ROW - 1) ? 0 : q + 1;
  }
}
static void b_mid(hls::stream<beat_t>& fin, hls::stream<bslice_t>& loc, hls::stream<beat_t>& fout, int jsel, int n) {
#pragma HLS INLINE off
  int q = 0;
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    const beat_t v = fin.read();
    b_capture(v, q, jsel, loc);
    fout.write(v);
    q = (q == BEATS_PER_ROW - 1) ? 0 : q + 1;
  }
}
static void b_end(hls::stream<beat_t>& fin, hls::stream<bslice_t>& loc, int jsel, int n) {
#pragma HLS INLINE off
  int q = 0;
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    const beat_t v = fin.read();
    b_capture(v, q, jsel, loc);
    q = (q == BEATS_PER_ROW - 1) ? 0 : q + 1;
  }
}

// ---------------------------------------------------------------------------------------------------------
// C serialisation daisy chain. Output order per (step,t) index: slab 0 beats 0..3, slab 1 beats 0..3, ... slab 7
// (== the v2 packed C layout). Slab 0's beat FIFO is the head of the chain; gather_n(S) forwards 4*S upstream beats
// and then its own 4 beats from its slab. Every stage touches only one upstream FIFO and one slab FIFO.
// ---------------------------------------------------------------------------------------------------------
static void gather_n(hls::stream<beat_t>& up, hls::stream<beat_t>& own, hls::stream<beat_t>& cout, int total, int nup) {
#pragma HLS INLINE off
  const int per = nup + BEATS_PER_CELL;               // beats this stage emits per (step,t) index
  const int n = total * per;
  int p = 0;
  for (int it = 0; it < n; ++it) {
#pragma HLS PIPELINE II=1
    beat_t w;
    if (p < nup) w = up.read();
    else w = own.read();
    cout.write(w);
    p = (p == per - 1) ? 0 : p + 1;
  }
}

// Final stage: plain beat copy to HBM, II=1, one long burst per run.
static void store_c(beat_t* C, hls::stream<beat_t>& in, int total) {
#pragma HLS INLINE off
  const int n = total * NUM_SLABS * BEATS_PER_CELL;
  for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
    C[i] = in.read();
  }
}
