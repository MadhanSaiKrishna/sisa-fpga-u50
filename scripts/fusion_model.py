#!/usr/bin/env python3
"""First-principles model of when runtime slab fusion wins (SISA vs the monolithic SA = same engine with f=8).

  steps_SA(M,N)   = ceil(M/64) * ceil(N/64)
  f(N)            = smallest power of two in {1,2,4,8} with 8f >= N          (choose_fusion)
  steps_SISA(M,N) = ceil(ceil(M/64) / G) * ceil(N / Rg),   G = 8/f groups, Rg = 8f token rows
  t_step(K)       = (8K + 16) cycles / f_clk                                 (clear 8 + K*8 MAC + drain 8)
  T_chained       = T_call[design] + steps * (t_step(K) + E)                 (launch cost hidden by queueing)
  T_blocking      ~ T_chained + H per call (host submit/complete, ~20 us, noisy +-5-10% between submissions)

Constants least-squares fitted to 38 chained decode-layer measurements from two board submissions
(D=512, F=1408): T_call SA 8.45 us, SISA 13.21 us (SISA ~4.8 us more per call), E = 0.306 us/step (~2%).
Max error over the 38 points 0.64%.  f_clk 277.6 MHz (SA 277.7, SISA 277.6 post-route).
"""
import math
CLK, MT, SL, NS = 277.6, 64, 8, 8
TCALL = {"SA": 8.45, "SISA": 13.21}   # us per kernel call inside the card (chained fit)
E = 0.306                               # us per step above the cycle model
H = 20.0                                # us host-side per blocking call (approximate)
def f_of(n):
    f = 1
    while f < NS and f * SL < n: f <<= 1
    return f
def steps_sa(M, N): return math.ceil(M / MT) * math.ceil(N / 64)
def steps_sisa(M, N, f=None):
    f = f or f_of(N)
    return math.ceil(math.ceil(M / MT) / (NS // f)) * math.ceil(N / (f * SL))
def tstep(K): return (8 * K + 16) / CLK
def T(design, steps, K, blocking=False):
    return TCALL[design] + (H if blocking else 0) + steps * (tstep(K) + E)
def speed(M, K, N, blocking=False):
    a, b = steps_sa(M, N), steps_sisa(M, N)
    return a / b, T("SA", a, K, blocking) / T("SISA", b, K, blocking)

if __name__ == "__main__":
    print("1) Asymptotic step speedup (large M): G = 8/f(N)")
    for N in (1, 8, 9, 16, 17, 32, 33, 64, 128):
        print(f"   N={N:<4} f={f_of(N)}  G={NS//f_of(N)}  row use SA={min(N,64)/64:5.1%} SISA={min(N, 8*f_of(N))/(8*f_of(N)):5.1%}")
    print("\n2) Small M limits the gain: steps speedup = m_tiles / ceil(m_tiles/G)   (N=1, f=1, G=8)")
    for M in (64, 128, 256, 512, 768, 1536, 4096):
        print(f"   M={M:<5} m_tiles={math.ceil(M/MT):<3} steps SA={steps_sa(M,1):<3} SISA={steps_sisa(M,1):<3} -> {steps_sa(M,1)/steps_sisa(M,1):.2f}x")
    print("\n3) Time speedup at N=1 (K=512), chained and blocking")
    for M in (64, 512, 768, 1536, 2816, 8192):
        s, c = speed(M, 512, 1); b = speed(M, 512, 1, True)[1]
        print(f"   M={M:<5} steps {s:.2f}x   chained {c:.2f}x   blocking {b:.2f}x")
    gemms = [("qkv", 1536, 512), ("attn_out", 512, 512), ("gate_up", 2816, 512), ("down", 512, 1408)]
    meas = {  # N: (SA chained, SISA chained) per layer, two submissions averaged
        1: (1509.5, 243.5), 2: (1508.5, 243.8), 4: (1510.0, 244.0), 8: (1511.7, 244.0), 16: (1511.5, 421.0),
        32: (1511.6, 788.5), 64: (1511.1, 1531.3), 128: (2983.7, 3001.3)}
    print("\n4) Validation (chained, per layer, us; constants fitted to these data, so in-sample)")
    print("   N    SA model/meas        SISA model/meas       speedup model/meas")
    for N, (sa, si) in meas.items():
        ms = sum(T("SA", steps_sa(M, N), K) for _, M, K in gemms)
        mi = sum(T("SISA", steps_sisa(M, N), K) for _, M, K in gemms)
        print(f"  {N:<4} {ms:7.1f}/{sa:7.1f} ({(ms/sa-1)*100:+.2f}%)  {mi:7.1f}/{si:7.1f} ({(mi/si-1)*100:+.2f}%)   {ms/mi:5.2f}/{sa/si:5.2f}")
