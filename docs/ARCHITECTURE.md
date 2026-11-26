# Architecture principles vs FPGA implementation

This file separates what comes from the SISA idea and the array organisation (architecture) from what was
engineering to make it close timing on an Alveo U50 (implementation). Source of the idea: Altamura et al.,
"SISA: A Scale-In Systolic Array for GEMM Acceleration", arXiv 2603.29913 (an ASIC design evaluated in a
simulator). This repository ports it to an FPGA; it does not claim the idea as new.

Naming: C[M x N] = A[M x K] * B[K x N], A = weights, B = activations. The paper's small dimension (tokens) is
our N: array rows <-> N, array columns <-> M.

## Architecture (what defines the two designs)
- One output-stationary array of ROWS x COLS = 64 x 8 = 512 FP32 PEs. PE(r, c) accumulates C[m][n = r]
  locally; only A and B values move.
- T_ACC = 8 accumulators per PE. A float add takes about 8 cycles, so cycling 8 independent accumulators lets
  the MAC loop run at II = 1 (`DEPENDENCE distance=8` in `col_engine`). One m-tile is MT = COLS * T_ACC = 64
  output features.
- The 64 rows are cut into NUM_SLABS = 8 slabs of SLAB_H = 8 rows.
- Runtime fusion factor f in {1, 2, 4, 8}: G = 8/f groups of Rg = 8f token rows. Each group works on a
  different A tile, so small N keeps more PEs busy. `choose_fusion(N)` picks the smallest power of two f with
  8f >= N.
- Schedule: steps = ceil(ceil(M/64) / G) * ceil(N / Rg). A step is: clear 8 accumulators, K * 8 MAC
  iterations, drain 8 results = 8K + 16 cycles.
- The conventional baseline (SA) is the same engine with f fixed at 8 and one A port, so it behaves as one
  monolithic array. 8 slab-parallel A ports (SISA) against 1 (SA) is part of the architecture, not a tuning
  choice: it is what lets 8 groups read 8 different tiles at once.
- Host-side layouts (`common.h`): `pack_a` (A tiles, zero padded), `pack_b` (depends on f), `unpack_c`. The
  C-cell layout is [row i][column c], 4 beats of 16 floats.

## Implementation (what is physical-design engineering)
- Dataflow of concurrent stages: load_a / load_b -> A and B tap chains -> slab_compute (a_split, b_dup, 8
  col_engine, c_comb) -> relay -> gather_n chain -> store_c. 512-bit HBM beats; a B slice is 256 bits.
- Tap chains and `relay` hops spread control fan-out and give the router registered stages across the SLR
  boundary. v2.4 added them to both designs; they helped the SA and hurt SISA until v2.5.
- v2.5: the SISA forwarding stage `a_gmid` carried four FIFO status flags in one 512-bit enable (fan-out
  522 to 603). It is split into `a_gsel` (select) and `a_gfan` (fan out), with a depth-4 stream `sm[]` between
  them. Cycle counts did not change; the clock went from 240.2 to 277.6 MHz.
- FIFO depths, stream pragmas and HBM bank mapping (`configs/connectivity_*.cfg`: SISA uses 10 HBM ports, one
  per A slab plus B and C; the SA uses 3). The host places one copy of A per SISA A port.
- None of these change what the array computes or its step count.
