#!/usr/bin/env python3
"""make_fake_stories.py out.bin -- random-weight checkpoint with the TinyStories-15M config (dim 288, hidden 768, 6 layers,
6 heads, vocab 32000, seq 256, shared classifier) in llama2.c's v0 format. For the no-board mock test only: the real
stories15M.bin is used on the cluster."""
import sys, struct
import numpy as np
out = sys.argv[1]
dim, hidden, L, H, KV, V, S = 288, 768, 6, 6, 6, 32000, 256
rng = np.random.default_rng(7)
def w(*shape, std=0.08): return (rng.standard_normal(shape) * std).astype(np.float32)
hs = dim // H
with open(out, "wb") as f:
    f.write(struct.pack("7i", dim, hidden, L, H, KV, V, S))      # vocab > 0 -> shared classifier
    for arr in [w(V, dim, std=0.5), np.ones((L, dim), np.float32), w(L, dim, dim), w(L, dim, dim), w(L, dim, dim), w(L, dim, dim),
                np.ones((L, dim), np.float32), w(L, hidden, dim), w(L, dim, hidden), w(L, hidden, dim), np.ones(dim, np.float32),
                np.zeros((S, hs // 2), np.float32), np.zeros((S, hs // 2), np.float32)]:
        f.write(arr.tobytes())
# freq_cis tables are skipped by run.c (it recomputes RoPE); zeros are fine
print("wrote", out)
