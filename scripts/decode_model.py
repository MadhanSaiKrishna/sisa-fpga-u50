#!/usr/bin/env python3
"""Predicted per-layer decode time for SA vs SISA from the fixed-cost + per-step model (no board needed).
   T_call = T_fixed + steps * c_step(K) / f_clk ;  c_step(K) = EFF*(8K + 16) cycles  (clear 8 + K*8 MAC + drain 8)
   Constants from the v2.5 --probe fit: T_fixed SA 31 us, SISA 37.7 us; 2346..2456 measured cycles/step vs 2320 model (EFF~1.03).
   steps: SA = m_tiles*ceil(N/64); SISA = ceil(m_tiles/G)*ceil(N/Rg) with f = choose_fusion(N).
   Chained lower bound = steps only (launch cost fully hidden); upper bound = blocking."""
import math, sys
D = int(sys.argv[1]) if len(sys.argv) > 1 else 512
F = int(sys.argv[2]) if len(sys.argv) > 2 else 1408
CLK, EFF = 277.6, 1.03
FIX = {"SA": 31.0, "SISA": 37.7}
MT, SL, NS = 64, 8, 8
def fusion(n):
    f = 1
    while f < NS and f * SL < n: f <<= 1
    return f
def steps(M, N, f):
    mt = math.ceil(M / MT); G = NS // f; Rg = f * SL
    return math.ceil(mt / G) * math.ceil(N / Rg)
gemms = [("qkv", 3 * D, D), ("attn_out", D, D), ("gate_up", 2 * F, D), ("down", D, F)]
def layer(design, N):
    f = fusion(N) if design == "SISA" else NS
    blk = chn = 0.0
    for _, M, K in gemms:
        st = steps(M, N, f)
        comp = st * EFF * (8 * K + 16) / CLK
        blk += FIX[design] + comp; chn += comp
    return blk, chn
print(f"D={D} F={F}  per-layer predicted us (blocking / chained lower bound)")
print(f"{'N':>4} {'SA blk':>8} {'SISA blk':>9} {'x blk':>6} | {'SA chn':>8} {'SISA chn':>9} {'x chn':>6}")
for N in (1, 2, 4, 8, 16, 32, 64):
    sb, sc = layer("SA", N); ib, ic = layer("SISA", N)
    print(f"{N:>4} {sb:8.1f} {ib:9.1f} {sb/ib:6.2f} | {sc:8.1f} {ic:9.1f} {sc/ic:6.2f}")
