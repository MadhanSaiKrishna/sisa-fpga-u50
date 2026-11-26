#!/usr/bin/env bash
# No-board check of the slm harness. The XRT classes are replaced by test/mock_xrt, and "the kernel" is the REAL
# src/gemm_sa.cpp / src/gemm_sisa.cpp compiled with test/shim (same as `make csim`). It verifies argument order, bank
# group ids (A_i -> 0..7, B -> nA, C -> nA+1), pack/unpack layouts, the N=1 fast path, the pointer-keyed weight cache,
# and that greedy llama2.c decoding matches the plain CPU run. It says nothing about timing.
#   needs: run.c (karpathy/llama2.c), tokenizer.bin; stories15M.bin if present, else a random-weight stand-in is generated.
#   env: RUN_C TOKENIZER CKPT TOKENS(default 10) SWEEP_ARGS
set -euo pipefail
cd "$(dirname "$0")/.."                       # slm/
ROOT=$(cd .. && pwd)
B=build_mock; mkdir -p $B
ABS=$PWD
RUN_C=${RUN_C:-run.c}; TOKENIZER=${TOKENIZER:-tokenizer.bin}; TOKENS=${TOKENS:-10}
CKPT=${CKPT:-stories15M.bin}
[ -f "$RUN_C" ] || { echo "need run.c (RUN_C=...)"; exit 2; }
[ -f "$TOKENIZER" ] || { echo "need tokenizer.bin (TOKENIZER=...)"; exit 2; }
if [ ! -f "$CKPT" ]; then echo "no $CKPT: using random-weight stand-in"; python3 test/make_fake_stories.py $B/fake.bin; CKPT=$B/fake.bin; fi

FL="-std=c++17 -O2 -ffp-contract=off -Wno-unknown-pragmas -Itest/mock_xrt -I$ROOT/test/shim -I$ROOT/src"
KSRC="$ROOT/src/gemm_sa.cpp $ROOT/src/gemm_sisa.cpp"
CKPT=$(realpath "$CKPT"); TOKENIZER=$(realpath "$TOKENIZER"); RUN_C=$(realpath "$RUN_C")
echo "== build"
python3 patch_run.py "$RUN_C" $B/run_fpga.c
gcc -O3 -DFPGA_HOOK -c $B/run_fpga.c -o $B/run_fpga.o
gcc -O3 -o $B/run_cpu "$RUN_C" -lm
g++ $FL -c fpga_matmul.cpp -o $B/fpga_matmul.o
g++ $FL $B/run_fpga.o $B/fpga_matmul.o $KSRC -o $B/slm_host_mock -lm
g++ $FL sweep_main.cpp $KSRC -o $B/slm_sweep_mock

echo "== CPU reference (greedy, $TOKENS tokens)"
$B/run_cpu "$CKPT" -z "$TOKENIZER" -t 0.0 -s 1 -n $TOKENS -i "Once upon a time" > $B/ref_cpu.txt 2> $B/ref_cpu.err
cat $B/ref_cpu.txt; tail -1 $B/ref_cpu.err

rc=0
for K in sa sisa; do
  echo "== mock $K: llama2.c decode"
  ( cd $B && MOCK_KERNEL=$K FPGA_VERIFY_TOKENS=3 FPGA_OUT_DIR=. ./slm_host_mock "$CKPT" -z "$TOKENIZER" -t 0.0 -s 1 -n $TOKENS -i "Once upon a time" > out_$K.txt 2> out_$K.err ) || rc=1
  if diff -q $B/ref_cpu.txt $B/out_$K.txt >/dev/null; then echo "[ok]   $K output identical to CPU"; else echo "[FAIL] $K output differs from CPU"; diff $B/ref_cpu.txt $B/out_$K.txt | head -5; rc=1; fi
  grep -E "verify|slm\] mock|kernel=" $B/out_$K.err | head -5
  echo "== mock $K: sweep"
  ( cd $B && MOCK_KERNEL=$K ./slm_sweep_mock ${SWEEP_ARGS:---shapes 288x288,768x288,288x768 --N 1,4,16 --iters 1} ) | tail -12 || rc=1
done
exit $rc
