#!/usr/bin/env bash
# submit_slm.sh -- stage the llama2.c (TinyStories-15M) harness for one of the existing xclbins and send it through fpga-submit.
#   scripts/submit_slm.sh sa|sisa [--sweep] [--verify T] [--nobreuse]            -> stage + `fpga-submit --check` only (free)
#   scripts/submit_slm.sh sa|sisa --submit [same options]                        -> check, confirm, REAL submit
#
#   (default) llama mode : slm/slm_host, inputs stories15M.bin + tokenizer.bin (arrive as inputs/ in the sandbox),
#                          greedy decode of "Once upon a time", 256 steps.  Results: out/slm_metrics.txt slm_calls.csv slm_tokens.csv,
#                          meta/stdout.txt = generated text (diff it against the CPU run), meta/stderr.txt = run.c's tok/s + summary.
#   --sweep              : slm/slm_sweep, decode shapes 288x288 768x288 288x768 32000x288 x N=1..32, out/slm_sweep.csv
#   --verify T           : also check every matmul of the first T tokens against the CPU (llama mode; default 2)
#   --sample             : temperature 1.0, top-p 0.9, seed 1 (what a story-sampling grader does) instead of greedy; compare against a CPU run with the same flags
#   --nobreuse           : disable the "identical B => skip re-upload" optimisation (naive per-call upload)
# Gate contract: fpga-submit [--check] design.xclbin host [--input F].. [-- HOST_ARGS]   (custom --env is REJECTED by the gate)
set -euo pipefail
DESIGN=${1:?usage: submit_slm.sh sa|sisa [--submit] [--sweep] [--verify T] [--nobreuse]}; shift
MODE=llama; DO_SUBMIT=0; VERIFY=2; BREUSE=1; SAMPLE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --submit) DO_SUBMIT=1;; --sweep) MODE=sweep;; --nobreuse) BREUSE=0;; --sample) SAMPLE=1;;
    --verify) VERIFY=${2:?}; shift;;
    *) echo "unknown option $1"; exit 2;;
  esac; shift
done
ROOT=$(cd "$(dirname "$0")/.." && pwd)
FPGA_CLIENT=${FPGA_CLIENT:-$HOME/fpga-client}
SUBMIT="$FPGA_CLIENT/fpga-submit"
STAGE="$ROOT/build/submit_slm_$DESIGN"
SUF=""; [ "$SAMPLE" = 1 ] && SUF="_sample"; [ "$BREUSE" = 0 ] && SUF="${SUF}_nobreuse"
OUT="$ROOT/results/board/slm_${DESIGN}_${MODE}${SUF}"
SLM="$ROOT/slm"
[ -x "$SUBMIT" ] || { echo "no fpga-submit at $SUBMIT (set FPGA_CLIENT)"; exit 1; }
[ -f "$ROOT/build/$DESIGN.xclbin" ] || { echo "missing build/$DESIGN.xclbin (existing bitstream; not rebuilt here)"; exit 1; }
if [ "$MODE" = llama ]; then HOST=slm_host; else HOST=slm_sweep; fi
[ -f "$SLM/$HOST" ] || { echo "missing slm/$HOST (on a compute node: source env.sh && make -C slm $HOST)"; exit 1; }
mkdir -p "$STAGE" "$OUT"
cp "$ROOT/build/$DESIGN.xclbin" "$STAGE/design.xclbin"
cp "$SLM/$HOST" "$STAGE/host"

# The gate rejects custom env vars, so settings travel as an input file (arrives as inputs/slm.conf) and the prompt too.
INPUTS=(); ENVS=()
if [ "$MODE" = llama ]; then
  for f in stories15M.bin tokenizer.bin; do [ -f "$SLM/$f" ] || { echo "missing slm/$f"; exit 1; }; INPUTS+=(--input "$SLM/$f"); done
  printf 'xclbin=design.xclbin\nbreuse=%s\nverify_tokens=%s\nlabel=slm_%s_%s\nprompt=Once upon a time\n' "$BREUSE" "$VERIFY" "$DESIGN" "$MODE" > "$STAGE/slm.conf"
  INPUTS+=(--input "$STAGE/slm.conf")
  if [ "$SAMPLE" = 1 ]; then SAMPLER=(-t 1.0 -p 0.9 -s 1); else SAMPLER=(-t 0.0 -s 1); fi   # greedy (default) | sampled like the mini-project grader
  HOST_ARGS=(inputs/stories15M.bin -z inputs/tokenizer.bin "${SAMPLER[@]}" -n 256)
else
  HOST_ARGS=(design.xclbin --iters 15)
fi

cd "$OUT"
"$SUBMIT" --check "$STAGE/design.xclbin" "$STAGE/host" ${INPUTS[@]+"${INPUTS[@]}"}
if [ "$DO_SUBMIT" = 1 ]; then
  read -r -p "REAL submit of slm/$MODE on '$DESIGN' (rate-limited card slot, 5-min cap). Continue? [y/N] " ok
  [ "$ok" = "y" ] || { echo "not submitted"; exit 0; }
  "$SUBMIT" "$STAGE/design.xclbin" "$STAGE/host" ${INPUTS[@]+"${INPUTS[@]}"} -- "${HOST_ARGS[@]}"
  echo "Now read: $OUT/fpga-results-*/meta/{status,exit_code,stdout.txt,stderr.txt} and out/{slm_metrics.txt,slm_sweep.csv,slm_calls.csv}"
fi
