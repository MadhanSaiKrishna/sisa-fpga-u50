# SISA-style slab systolic engine vs a conventional systolic array on the Alveo U50

Course project (FPGA Based Accelerator Design, IIIT Hyderabad). It ports the SISA idea (Altamura et al., arXiv
2603.29913, an ASIC design evaluated in a simulator) to an AMD Alveo U50 in hand-written C++ HLS (FP32) and
compares it with a conventional monolithic array built from the same engine, at equal PE count (512) and the same
HBM access for B and C. It does not claim the idea as new, and no licence has been chosen for this code yet.

Headline (details and limits in `docs/RESULTS.md`): at equal PE count and clock (277.7 / 277.6 MHz) runtime
fusion makes a decoder layer 6.2 times faster at N <= 8 (chained launches), 1.9 times at N=32 and equal at N >= 64;
at square 768^3 both designs reach about 266 GFLOPS (93.6% of peak). The gain costs about 8% more LUTs, 4 times the
BRAM and 8 copies of the weights in HBM.

## Run it without a board
```bash
make csim      # plain g++, no Vitis: both kernels against a golden model, every fusion factor (90 runs)
make model     # first-principles speedup model and its check against measured data
```
`make csim` runs the real kernel sources through a small shim of the HLS stream and integer types, uses the same
`pack_a` / `pack_b` / `unpack_c` as the host, and also checks the compute-iteration count against the schedule
formula. Expected last line: `TEST PASSED: 90 runs, 0 failures`.

## Build for the board (needs Vitis 2025.1, XRT and a U50)
```bash
source ./env.sh                  # adjust paths for your install
make hls-sa hls-sisa             # csynth -> build/gemm_*.xo   (check the loop IIs and DSP/BRAM reports)
make xclbin-sa xclbin-sisa       # about 1 hour each; run on a compute node, never a login node
make host                        # XRT host
build/host design.xclbin --N 1,2,4,8,16,32,64,128 --decode --probe
```
`--decode` replays one decoder layer (static inputs, no repacking between GEMMs) as blocking calls and as 12 chained
layers; `--probe` separates the fixed cost per call from the cost per step. `scripts/submit_board.sh` stages a design
for the course gate and is specific to that environment.

## Layout
```
src/      common.h kernel_common.h slab_stages.h gemm_sa.cpp gemm_sisa.cpp   kernels, shapes, host layouts
test/     tb.cpp (csim golden test)  tb_cosim.cpp  shim/                     runs without Vitis
host/     host.cpp                                                           XRT host (sweep, --probe, --decode)
configs/  hls_*.cfg connectivity_*.cfg                                       HLS config, HBM mapping
scripts/  fusion_model.py decode_model.py submit_board.sh submit_slm.sh
slm/      fpga_backend.h fpga_matmul.cpp sweep_main.cpp compare_slm.py       llama2.c harness (see slm/README.md)
docs/     ARCHITECTURE.md  MEASUREMENT_TRAPS.md  RESULTS.md  SLM_RESULTS.md
```

## LLM inference benchmark (llama2.c, TinyStories-15M)
`slm/` runs Karpathy's llama2.c on the two existing bitstreams without rebuilding them. Every `matmul` of the
forward pass (43 per token: 24 attention 288x288, 12 ffn_up 768x288, 6 ffn_down 288x768, 1 classifier 32000x288)
is sent to the FPGA as an N=1 GEMM; weights are uploaded once and cached by pointer, and everything else
(attention, norms, sampling) stays on the CPU. The harness records tokens/s and a per-shape split into kernel
wait, launch, PCIe sync, and host pack/unpack. Text output is checked byte for byte against the CPU run.

Greedy decode, 221 generated tokens, one board submission per design (cross-submission variation is about 8%):

| | CPU run.c | SA v2.4 | SISA v2.5 |
|---|---|---|---|
| tokens/s | 51.8 | 73.0 | 172.5 |
| ms/token | 19.3 | 13.7 | 5.8 |

SISA is 2.35x the SA end to end: it needs 63 steps per classifier call against 500 for the SA. Repeats,
sampled decoding and a no-B-reuse ablation are being added. Full tables and caveats are in
`docs/SLM_RESULTS.md`; usage is in `slm/README.md`.

```bash
# on the cluster, from the project root, with the two xclbins in build/
scripts/submit_slm.sh sisa --submit          # also: sa, --sweep, --sample, --nobreuse
python3 slm/compare_slm.py --ref slm/ref_cpu.txt --sa <dir> --sisa <dir>
```
`stories15M.bin` and `tokenizer.bin` are not in the repo: get them from the llama2.c project
(https://github.com/karpathy/llama2.c) and place them next to `slm/run_cpu`.

## Read first
- `docs/ARCHITECTURE.md`: what is architecture (output-stationary array, 8 accumulators for II=1, slab fusion,
  step count) and what is FPGA implementation (tap chains, relay hops, the a_gsel / a_gfan split, FIFO depths).
- `docs/MEASUREMENT_TRAPS.md`: why a link-time `kernel_frequency` setting hid the real clock, the 8% run-to-run
  variation between board submissions, why step-count speedups shrink once per-call cost is included, and what was
  retracted.

## Status and limits
Both designs pass the golden check on the board (exit 0) for every shape in the sweeps. Timing results are single
Vivado runs. Power was not measured. The baseline is the paper's monolithic array; a decode-oriented conventional
design would narrow the small-N gap.
