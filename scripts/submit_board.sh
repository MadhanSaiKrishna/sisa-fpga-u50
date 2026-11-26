#!/usr/bin/env bash
# submit_board.sh -- stage a built design and send it through the course fpga-client (gate on node08).
#   scripts/submit_board.sh sa|sisa [host-args...]            -> stage + `fpga-submit --check` only (free, no card slot)
#   scripts/submit_board.sh sa|sisa --submit [host-args...]   -> check, confirm, then the REAL submit
#
# Contract (fpga-client README.md / AGENTS.md / fpga-submit):
#   fpga-submit [--check] design.xclbin host [--input F].. [--lib L].. [--env K=V].. [-- HOST_ARGS]
#   - the xclbin is NOT passed to the host automatically -> we pass it explicitly after `--`
#   - host must be built on an RCE compute node (glibc <= 2.31); --check verifies this with readelf
#   - one job on the card at a time, 5-minute cap, 2 queued, a few per hour -> never iterate on the board
#   - results land in ./fpga-results-<time>-<id>/{meta,out}; we run from results/board/<design>/ so they stay in this repo
# FPGA_CLIENT = folder holding fpga-submit (default ~/fpga-client). FPGA_GATE_SPOOL is honoured by fpga-submit.
set -euo pipefail
DESIGN=${1:?usage: submit_board.sh sa|sisa [--submit] [host-args...]}; shift
DO_SUBMIT=0
if [ "${1:-}" = "--submit" ]; then DO_SUBMIT=1; shift; fi
HOST_ARGS=("$@")

ROOT=$(cd "$(dirname "$0")/.." && pwd)
FPGA_CLIENT=${FPGA_CLIENT:-$HOME/fpga-client}
SUBMIT="$FPGA_CLIENT/fpga-submit"
STAGE="$ROOT/build/submit_$DESIGN"
OUT="$ROOT/results/board/$DESIGN"

[ -x "$SUBMIT" ] || { echo "no fpga-submit at $SUBMIT (set FPGA_CLIENT; run its ./setup.sh once)"; exit 1; }
[ -f "$ROOT/build/$DESIGN.xclbin" ] || { echo "missing build/$DESIGN.xclbin (make xclbin-$DESIGN)"; exit 1; }
[ -f "$ROOT/build/host" ]           || { echo "missing build/host (make host, on a compute node)"; exit 1; }
mkdir -p "$STAGE" "$OUT"
cp "$ROOT/build/$DESIGN.xclbin" "$STAGE/design.xclbin"
cp "$ROOT/build/host" "$STAGE/host"

cd "$OUT"
"$SUBMIT" --check "$STAGE/design.xclbin" "$STAGE/host"
if [ "$DO_SUBMIT" = 1 ]; then
  read -r -p "REAL submit of '$DESIGN' (uses a rate-limited card slot, 5-min cap). Continue? [y/N] " ok
  [ "$ok" = "y" ] || { echo "not submitted"; exit 0; }
  "$SUBMIT" "$STAGE/design.xclbin" "$STAGE/host" -- design.xclbin "${HOST_ARGS[@]}"
  echo "Now read: $OUT/fpga-results-*/meta/{status,exit_code,stderr.txt} and out/{metrics.txt,sweep.csv}"
fi
