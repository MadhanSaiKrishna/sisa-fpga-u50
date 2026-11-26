#!/usr/bin/env python3
"""patch_run.py run.c run_fpga.c -- add the FPGA hooks to Karpathy's llama2.c run.c (MIT licensed, not shipped here).

Three edits, all guarded by `#ifdef FPGA_HOOK` so the same file still builds as the plain CPU program:
  1. declarations of fpga_matmul / fpga_token_begin before matmul()
  2. first statement of matmul():  fpga_matmul(xout, x, w, n, d); return;
  3. first statement of forward(): fpga_token_begin(pos);
  4. prompt override from inputs/slm.conf / $SLM_PROMPT (the gate rejects env vars and may mangle quoted arguments)
Every pattern must match exactly once, otherwise the script stops (a different run.c version must be looked at by hand).
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()

def once(s, old, new):
    n = s.count(old)
    if n != 1:
        sys.exit(f"patch_run.py: expected exactly 1 match of {old!r}, found {n}")
    return s.replace(old, new)

s = once(s, "void matmul(float* xout, float* x, float* w, int n, int d) {\n",
"""#ifdef FPGA_HOOK
extern void fpga_matmul(float* xout, float* x, float* w, int n, int d);
extern void fpga_token_begin(int pos);
#endif
void matmul(float* xout, float* x, float* w, int n, int d) {
#ifdef FPGA_HOOK
    fpga_matmul(xout, x, w, n, d);
    return;
#endif
""")
s = once(s, "float* forward(Transformer* transformer, int token, int pos) {\n",
"""float* forward(Transformer* transformer, int token, int pos) {
#ifdef FPGA_HOOK
    fpga_token_begin(pos);
#endif
""")
s = once(s, "    // parameter validation/overrides\n",
"""#ifdef FPGA_HOOK
    { extern const char* slm_prompt_override(void); const char* po = slm_prompt_override(); if (po) prompt = (char*)po; }
#endif
    // parameter validation/overrides
""")
open(dst, "w").write(s)
print(f"wrote {dst}")
