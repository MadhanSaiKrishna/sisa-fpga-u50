# Results summary (Alveo U50, FP32, 512 PEs, both designs)

Post-route clocks (single seed): SA v2.4 277.7 MHz, SISA v2.5 277.6 MHz.

Post-route kernel resources (one link each; percent of user budget after the platform shell):

| Design | LUT | Registers | BRAM tiles | DSP |
|---|---|---|---|---|
| SA v2.4 | 242,966 (32.7%) | 367,652 (23.4%) | 52 (4.5%) | 2,645 (44.5%) |
| SISA v2.5 | 263,302 (35.7%) | 395,642 (25.4%) | 206 (17.7%) | 2,665 (44.8%) |

No power data exists.

Decoder layer (D=512, F=1408; QKV, attention output, gate+up, down; matmuls only, static inputs), SISA speedup
over SA per layer, two submissions:

| N | SISA f | Chained | Blocking (run 1 / run 2) |
|---|---|---|---|
| 1 to 8 | 1 | 6.20x | 4.3-4.7x / 5.1-5.3x |
| 16 | 2 | 3.59x | 3.08x / 3.39x |
| 32 | 4 | 1.92x | 1.79x / 1.92x |
| 64 | 8 | 0.99x | 0.94x / 0.98x |

Square M=768, K=768, N=768 (median, one submission each): SA 3404.7 us (266.1 GFLOPS), SISA f=8 3402.3 us
(266.3 GFLOPS); peak 284 GFLOPS. Mini-project run #203: 126.7 GFLOPS (256 lanes, 296 MHz); harnesses differ.

Limits: one model size, matmuls only, one Vivado seed per design, the baseline is the paper's monolithic array, no
power data, SISA's gain costs about 8% more LUTs, 4 times the BRAM and 8 copies of the weights in HBM.
