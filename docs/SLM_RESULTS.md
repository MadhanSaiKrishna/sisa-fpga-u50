# llama2.c (TinyStories-15M) on SA v2.4 vs SISA v2.5: initial board results (2026-11-26)

Harness: `slm/`. Bitstreams: `sa.xclbin` (v2.4, ~277.7 MHz) and `sisa.xclbin` (v2.5, ~277.6 MHz), FP32, 512 PEs each.
Greedy decode (`-t 0 -s 1`), prompt "Once upon a time", B-reuse on, tokens 1..221 timed (the story ended at the BOS
token before 256; token 0 is excluded because it includes device open and weight upload). One submission per design
in this first set; cross-submission variation is about 8% (see MEASUREMENT_TRAPS.md).

## Correctness
Both designs produced text byte-identical to the CPU run.c reference. A check of the first 2 tokens (86 matmuls each)
had 0 mismatches and max_abs_err 0. The shape sweep (M=32000 included, N=1..32) passed for every shape with err 0.

## End to end
| | CPU run.c | SA v2.4 | SISA v2.5 |
|---|---|---|---|
| tok/s (run.c achieved) | 51.8 (62.1 in an earlier run) | 73.0 | 172.5 |
| ms/token | 19.3 | 13.73 | 5.84 |

SISA/SA 2.35x; SISA 3.3x the CPU (2.8x using the 62.1 run); SA 1.41x the CPU.
The CPU baseline is noisy on shared nodes (51.8 to 62.1 tok/s in separate runs).

## Where a token goes (us per token)
| | SA v2.4 | SISA v2.5 |
|---|---|---|
| wait (kernel + completion) | 8630 | 2833 |
| launch (set args + start) | 404 | 365 |
| PCIe sync (B up + C down) | 3154 | 1647 |
| host pack/unpack | 371 | 197 |
| CPU other (attention, norms, sampling, print) | 1167 | 794 |
| total | 13726 | 5836 |

Per-call N=1 (llama, median us): SISA attn 55.7, ffn_up 64.8, ffn_down 69.9, classifier 585.6;
SA 79.8, 138.7, 149.5, 4425.

## Reading
- The SA is compute-bound at N=1. SISA needs 63 steps per classifier call against 500 for the SA, which is where
  the end-to-end gain comes from.
- For SISA, PCIe sync is 28% of a token. The packed B and C layouts inflate transfers (N=1 uses 1 of 64 floats per
  B row) and each of the 43 calls pays a fixed DMA latency of 15 to 20 us.
- Fusing qkv and gate/up calls (43 to about 25 calls per token) and a compact N=1 layout are the obvious software
  levers.

## Not yet measured
Repeats (3 per design), `--nobreuse`, `--sample` (sampled decoding), stories of exactly 256 tokens, power.
