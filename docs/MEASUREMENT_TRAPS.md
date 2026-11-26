# Measurement traps and negative results

Things that cost hours or produced wrong conclusions in this project. All numbers are from the runs in
`docs/RESULTS.md`; they are single-seed Vivado runs and a few board submissions unless stated.

## Timing
- **`kernel_frequency` in a link `.cfg` masks the real clock.** Builds that set it were implemented against a
  relaxed 10.000 ns constraint, so the reported 100 to 106 MHz were lower bounds, not design quality. Without the
  option the link targets 3.333 ns and Vitis scales the kernel clock to what the design achieves. The board time
  of the SA dropped 2.4 to 2.6 times, tracking the clock, with cycle counts unchanged.
- **Only post-route numbers are timing results.** HLS estimates are not. Fmax = 1000 / (period - WNS) from the
  routed kernel-clock WNS (valid for WNS <= 0). Route-dominated paths (90 to 97% routing delay), SLR crossings
  and large-fan-out enables decided the clock, not logic depth.
- **Placement directives alone are not a silver bullet.** `SSI_SpreadLogic_high` gave SA 272.9 MHz (from 277.7)
  and SISA 238.7 MHz (from 240.2): worse, single seed. The fix that worked was structural (v2.5 split of a
  512-bit enable with four flags).
- **One Vivado run per design.** Differences of a few percent in clock are within what another seed could
  change.
- **HLS resource estimates were wrong for logic**: LUT 366,741 (SA) estimated against 242,966 post-route, flip-flops
  516,166 against 367,652. DSP and BRAM matched.

## Build flow
- A Makefile that ends in `cp $(find ... -name '*.xo')` can silently copy a stale `.xo`. `rm -rf build/hls_<k>`
  first and check the `.xo` size and timestamp.
- On the course cluster, batch HLS jobs failed on one node (missing `libtinfo.so.5`); exclude it. Never run
  Vivado or Vitis on the login node.

## Board measurements
- **Blocking times are noisy between submissions.** The same bitstream gave 130.6, 141.9 and 132.5 us at N=64;
  blocking decode-layer medians moved up to 12%. Within one run the spread was under 1%. Use `min_us`, ratios from
  the same session, and report blocking speedups as a range.
- **Chained launches are reproducible** (decode-layer times within 0.3% between submissions). Use them for
  fine comparisons.
- **A fixed cost per kernel call dilutes step-count speedups.** SISA has 6 times fewer steps at M=768, N=8, but
  the measured speedup is 2.2 to 2.5 times. Probe fit: about 27 us (SA) and 37 us (SISA) per blocking call.
  Chained fit on decode layers: about 8.5 us (SA) and 13.2 us (SISA) per call inside the card, 0.31 us per step.
- **Blocking data cannot resolve a ~5 us per-call difference.** An earlier claim that the SA and SISA call costs
  agree was wrong for that reason; the chained data show SISA carries about 5 us more per call.
- **Do not compare GFLOPS across different timing harnesses.** The mini project reports grader kernel time; this
  project times a host launch plus wait.
- **Board slots are scarce.** Run the free pre-flight `--check` first; one job should carry a whole sweep.

## Retracted statements
- "SISA never loses" (it is 0 to 6% slower than the SA once fused to f=8).
- "Per-step overhead of about 3250 cycles, 72% of peak" (wrong; the cost is a fixed launch cost).
- "v2.4 SA gained 7 to 9%" (within the ~8% run-to-run variation).
- "SA and SISA per-call overheads agree within 2 us" (see above).
- "The mini project is ahead per MAC" (it used the N=128, K=288 shape; at square 768^3 ours is 0.52 GFLOPS
  per MAC against 0.50 for mini run #203).
