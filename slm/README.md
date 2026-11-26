# slm/ -- llama2.c (TinyStories-15M) on the existing SA v2.4 / SISA v2.5 xclbins

No bitstream is rebuilt. `fpga_backend.h` reuses the buffer/bank/argument conventions of `host/host.cpp` verbatim
(A: 1 bo on group 0 for SA, 8 bos on groups 0..7 for SISA; B on group nA; C on group nA+1; launch
`krn(A.., B, C, m_tiles, n_tiles, K[, f])`; layouts from `src/common.h`). `run.c` is patched at two call sites
(`matmul`, `forward`) by `patch_run.py`; nothing from the deleted mini1-gemm folder is used.

    matmul(xout, x, w, n, d)  ->  C[M,1] = A[M,K] * B[K,1],  M = d, K = n, A = w (pointer-keyed HBM cache)

## Files
| file | role |
|---|---|
| `fpga_backend.h` | SA/SISA GEMM backend: weight cache, B/C buffers, N=1 fast pack/unpack, general-N path, per-call timers |
| `fpga_matmul.cpp` | the llama2.c hook + instrumentation (per-shape, per-token, CSVs, summary on stderr) |
| `sweep_main.cpp` | decode-shape sweep (shapes x N) through the same backend, with golden check |
| `patch_run.py` | adds the hooks to your `run.c` (fails loudly if the pattern is not found exactly once) |
| `Makefile` | `run_cpu`, `slm_host`, `slm_sweep`, `mocktest` |
| `compare_slm.py` | side-by-side table from result folders |
| `test/` | mock XRT that calls the real `src/gemm_*.cpp`; `run_mocktest.sh` |
| `../scripts/submit_slm.sh` | stage + `fpga-submit --check` (+ `--submit`) |

## Cluster steps
```bash
# 0. copy slm/ and scripts/submit_slm.sh into your project checkout on the cluster, then:
sinteractive -c 8 -A cs2401                    # compute node; never build on the login node
cd <project> && source env.sh                  # Vitis/XRT environment (also sources /opt/xilinx/xrt/setup.sh)
cd slm

# 1. data: look for shared copies first, else fetch (or scp from your laptop)
ls /data /scratch /opt 2>/dev/null | grep -i -E 'story|llama'
wget -nc https://raw.githubusercontent.com/karpathy/llama2.c/master/run.c
wget -nc https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget -nc https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin

# 2. CPU golden reference (greedy: -t 0.0), same node
make run_cpu
./run_cpu stories15M.bin -z tokenizer.bin -t 0.0 -s 1 -n 256 -i "Once upon a time" > ref_cpu.txt 2> ref_cpu.err
tail -1 ref_cpu.err; md5sum ref_cpu.txt

# 3. no-board software check against the real kernel sources (10 tokens, a minute or so)
make mocktest

# 4. build the hosts (needs XRT from env.sh; glibc check is done by fpga-submit --check)
make slm_host slm_sweep

# 5. free pre-flight, no card slot used
cd .. && scripts/submit_slm.sh sa && scripts/submit_slm.sh sisa && scripts/submit_slm.sh sisa --sweep

# 6. real runs (one slot each; ~8% run-to-run variation between submissions -> repeat the llama run 3x if the quota allows)
scripts/submit_slm.sh sa   --submit
scripts/submit_slm.sh sisa --submit
scripts/submit_slm.sh sa   --submit --sweep
scripts/submit_slm.sh sisa --submit --sweep

# 7. correctness + comparison
python3 slm/compare_slm.py --ref slm/ref_cpu.txt --cpu-tok-s <from ref_cpu.err> --mini203 116.23 \
   --sa  results/board/slm_sa_llama/fpga-results-*   --sisa results/board/slm_sisa_llama/fpga-results-*
```

## What is measured (all microseconds, steady state = tokens 1..255; token 0 holds device open + weight upload)
Per matmul call: `pack_x` (host, write x into the packed B buffer) | `b_sync` (PCIe, B -> HBM) | `launch` (set args + start)
| `wait` (kernel + completion) | `c_sync` (PCIe, C <- HBM) | `unpack` (host). `dev = launch + wait` is what `host.cpp` calls kernel time.
Per token: `matmul total` + `cpu other` (attention, rmsnorm, RoPE, softmax, sampling, printing). Shapes are labelled
`attn` (M=K, 288x288), `ffn_up` (768x288), `ffn_down` (288x768), `classifier` (32000x288).
Two tok/s figures are printed: run.c's own (`achieved tok/s`, same window as the CPU build) and the harness's.

## Comparing with mini1-gemm (from the shared mini1-gemm folder, run #203)
`mini203_scores.json` holds the grader's #203 scores (296 MHz kernel): 116.23 tok/s = 8.604 ms/token. Its kernel-only times at N=1
(288x288 0.0568 ms, 768x288 0.0987, 288x768 0.0955, 32000x288 2.910) sum over the 43 matmuls of a token to 6.031 ms, so about
2.57 ms/token (~60 us per call) is outside the kernel: launch, PCIe, and the CPU parts. `compare_slm.py` prints the same split for
SA/SISA (`device sum` vs `everything else`) so the comparison is like for like. The grader's host code is not in the repo, so its
timing window is an assumption (run.c's own `achieved tok/s`, which skips token 0). The grader writes a SAMPLED story; use
`scripts/submit_slm.sh sa|sisa --submit --sample` (temperature 1.0, top-p 0.9, seed 1) plus a CPU run with the same flags
(`./run_cpu stories15M.bin -z tokenizer.bin -t 1.0 -p 0.9 -s 1 -n 256 -i "Once upon a time" > ref_cpu_sample.txt`) for the closest match.
Unlike the mini kernel (row-major A, B, C), the SA/SISA kernels need packed B and C, so B/C PCIe traffic and pack/unpack are part of
our per-call overhead.

## Things to know before reading the numbers
- `FPGA_BREUSE=1` (default): if x is byte-identical to what the B buffer already holds, the upload is skipped (wk, wv after wq;
  w3 after w1 = 3 of 7 matmuls per layer). `--nobreuse` gives the naive per-call upload. Report which one you used.
- At N=1 the kernel still reads a full 64-wide B row per k (256 B per k) and writes the full packed C, so B/C PCIe traffic is far
  larger than the useful data: SA classifier C is 500 steps x 16 KB = 8.2 MB per token, SISA 63 steps = 1.0 MB. The existing
  host.cpp benchmarks time `run.wait()` only, so these costs are NEW in end-to-end numbers.
- FP32 order: each output accumulates sequentially over k like run.c, so results should match the CPU to ~1e-7; if the text
  diverges, `compare_slm.py` prints the first differing character. `--verify T` logs max abs/rel error per matmul.
- Weight cache key is (pointer, M, K); weights must be immutable after first use (true for run.c).
- Unverified on hardware: M=32000 on the real bitstreams (largest M run on the board so far in your notes is 2816), the
  `--env` / quoted `-i "Once upon a time"` handling inside fpga-submit (fallback: env `SLM_PROMPT`), and the size limit of `--input`.
- Comparison with mini1-gemm #203 (116.23 tok/s): only valid if that figure used the same prompt, greedy sampling, 256 steps,
  timing window and offloaded the same matmuls (and a model with the same shapes).

## Third-party code
`slm/run.c` is the inference program from Andrej Karpathy's llama2.c
(https://github.com/karpathy/llama2.c), used under the MIT licence; see `LICENSE-llama2c`.
`patch_run.py` applies four small `#ifdef FPGA_HOOK` edits to a copy of it (`run_fpga.c`) to route
`matmul` to the FPGA backend. The model checkpoint `stories15M.bin` and `tokenizer.bin` come from the same project.
