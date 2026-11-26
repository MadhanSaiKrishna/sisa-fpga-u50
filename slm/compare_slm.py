#!/usr/bin/env python3
"""compare_slm.py --sa RESULT_DIR --sisa RESULT_DIR [--ref ref_cpu.txt] [--cpu-tok-s X] [--mini203 116.23]

RESULT_DIR = an fpga-results-<time>-<id> folder (needs out/slm_metrics.txt; meta/stdout.txt and meta/stderr.txt if present).
Prints end-to-end tok/s (harness-internal and run.c's own), per-shape device/total times, per-token overhead split, and whether the
generated text equals the CPU reference. Pass several --sa / --sisa dirs (repeat the flag) to get min/median across submissions."""
import argparse, os, re, statistics as st

def load(d):
    m = {}
    for line in open(os.path.join(d, "out", "slm_metrics.txt")):
        p = line.split(None, 1)
        if len(p) == 2:
            try: m[p[0]] = float(p[1])
            except ValueError: m[p[0]] = p[1].strip()
    err = os.path.join(d, "meta", "stderr.txt")
    if os.path.exists(err):
        r = re.search(r"achieved tok/s:\s*([0-9.]+)", open(err).read())
        if r: m["tok_s_runc"] = float(r.group(1))
    return m

def agg(ms, key):
    v = [m[key] for m in ms if isinstance(m.get(key), float)]
    return (min(v), st.median(v), max(v)) if v else None

def fmt(t, nd=1):
    return "-" if t is None else (f"{t[1]:.{nd}f}" if t[0] == t[2] else f"{t[1]:.{nd}f} [{t[0]:.{nd}f}-{t[2]:.{nd}f}]")

ap = argparse.ArgumentParser()
ap.add_argument("--sa", action="append", default=[]); ap.add_argument("--sisa", action="append", default=[])
ap.add_argument("--ref"); ap.add_argument("--cpu-tok-s", type=float); ap.add_argument("--mini203", type=float, default=116.23)
ap.add_argument("--mini-scores", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mini203_scores.json"), help="grader scores json of the mini1-gemm run to compare with")
a = ap.parse_args()
D = {"SA v2.4": [load(d) for d in a.sa], "SISA v2.5": [load(d) for d in a.sisa]}
D = {k: v for k, v in D.items() if v}
names = list(D)
print(f"{'':34s}" + "".join(f"{n:>26s}" for n in names))
def row(label, key, nd=1, ms=None):
    print(f"{label:34s}" + "".join(f"{fmt(agg(D[n], key), nd):>26s}" for n in names))
row("tok/s (harness, steady state)", "tok_s_internal", 2); row("tok/s (run.c achieved)", "tok_s_runc", 2)
row("ms / token", "ms_per_token", 3); row("  matmul total (ms)", "ms_per_token_matmul_total", 3); row("  CPU other (ms)", "ms_per_token_cpu_other", 3)
print("-- per token (us): where the matmul time goes")
for k in ("wait", "launch", "pcie_sync", "host_pack_unpack", "cpu_other"):
    row(f"  {k}", f"us_per_token_{k}", 0)
print("-- per shape (us per call): device = launch+wait, total = + B/C sync + pack/unpack")
for sh in ("attn", "ffn_up", "ffn_down", "classifier"):
    for key, lab in (("dev_median_us", "dev med"), ("dev_min_us", "dev min"), ("total_mean_us", "total mean"), ("us_per_token", "us/token")):
        row(f"  {sh:10s} {lab}", f"{sh}_{key}", 1)
    row(f"  {sh:10s} calls/token", f"{sh}_calls_per_token", 0)
# device-only sum per token (what a kernel-time-only grader would call "kernel ms"): calls/token x median(launch+wait)
SHAPES = ("attn", "ffn_up", "ffn_down", "classifier")
def dev_sum(m):
    t = 0.0
    for sh in SHAPES:
        c, d = m.get(f"{sh}_calls_per_token"), m.get(f"{sh}_dev_median_us")
        if isinstance(c, float) and isinstance(d, float): t += c * d
    return t / 1e3
for n in names:
    for m in D[n]: m["dev_sum_ms"] = dev_sum(m); m["overhead_ms"] = m.get("ms_per_token", 0) - m["dev_sum_ms"]
print("-- per token (ms): device-only sum vs everything else (B/C sync, launch bookkeeping, pack/unpack, CPU work)")
row("device sum (launch+wait)", "dev_sum_ms", 3); row("everything else", "overhead_ms", 3)
print("-- context")
row("weights in HBM (MB)", "weights_hbm_MB", 1); row("weight upload (ms, token 0)", "weights_upload_ms", 1)
if a.cpu_tok_s: print(f"CPU run.c tok/s (same node, same flags): {a.cpu_tok_s}")
mini = None
if a.mini_scores and os.path.exists(a.mini_scores):
    import json
    j = json.load(open(a.mini_scores))
    n1 = {(c["M"], c["K"]): c["ms"] for c in j["cases"] if c["group"] == "n1"}
    ksum = 24 * n1[(288, 288)] + 12 * n1[(768, 288)] + 6 * n1[(288, 768)] + n1[(32000, 288)]
    mt = 1000.0 / j["tok_s"]
    print(f"-- mini1-gemm run {j['id']} (grader): {j['tok_s']:.2f} tok/s = {mt:.3f} ms/token; kernel-only sum over the 43 N=1 matmuls = {ksum:.3f} ms;")
    print(f"   so everything outside the kernel is {mt - ksum:.3f} ms/token ({(mt - ksum) / 43 * 1e3:.0f} us per call incl. CPU work). Per-shape kernel ms: "
          + ", ".join(f"{k[0]}x{k[1]} {v:.4f}" for k, v in n1.items()))
    for n in names:
        d = agg(D[n], "dev_sum_ms"); t = agg(D[n], "ms_per_token")
        if d and t: print(f"   {n}: device sum {d[1]:.3f} ms vs mini {ksum:.3f} ms (speed vs mini: {ksum / d[1]:.2f}x);  ms/token {t[1]:.3f} vs {mt:.3f} (speed vs mini: {mt / t[1]:.2f}x)")
else:
    print(f"mini1-gemm #203 (your note): {a.mini203} tok/s")
print("   comparable only if prompt, sampling mode, 256 steps, timing window and the set of offloaded matmuls match; the grader's host code is not in the repo.")
if len(D) == 2:
    s = agg(D["SISA v2.5"], "tok_s_internal"); b = agg(D["SA v2.4"], "tok_s_internal")
    if s and b: print(f"SISA/SA end-to-end tok/s: {s[1]/b[1]:.2f}x")
if a.ref:
    ref = open(a.ref).read()
    for n, dirs in (("SA", a.sa), ("SISA", a.sisa)):
        for d in dirs:
            p = os.path.join(d, "meta", "stdout.txt")
            if os.path.exists(p):
                got = open(p).read()
                if got == ref: print(f"text {n} {os.path.basename(d)}: identical to CPU reference")
                else:
                    i = next((i for i, (x, y) in enumerate(zip(got, ref)) if x != y), min(len(got), len(ref)))
                    print(f"text {n} {os.path.basename(d)}: DIFFERS from CPU at char {i}")
