// mock kernel: dispatches to the REAL gemm_sa / gemm_sisa sources (compiled with test/shim). Checks that every buffer is
// bound to the HBM bank group the connectivity cfgs expect (A_i -> 0..7, B -> nA, C -> nA+1).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "kernel_common.h"
extern "C" void gemm_sa(const beat_t*, const beat_t*, beat_t*, int, int, int);
extern "C" void gemm_sisa(const beat_t*, const beat_t*, const beat_t*, const beat_t*, const beat_t*, const beat_t*,
                          const beat_t*, const beat_t*, const beat_t*, beat_t*, int, int, int, int);
namespace xrt {
class run { public: void wait() {} };
class kernel {
  bool sisa_ = false;
 public:
  kernel() = default;
  kernel(device&, const uuid&, const std::string& name) {
    const char* want = std::getenv("MOCK_KERNEL");           // "sa" or "sisa"
    const std::string w = want ? want : "sisa";
    if (name != "gemm_" + w) throw std::runtime_error("mock: no such kernel " + name);
    sisa_ = (w == "sisa");
  }
  int group_id(int i) const { return i; }
  static const beat_t* B_(bo& b) { return reinterpret_cast<const beat_t*>(b.map<float*>()); }
  static void chk(bo& b, int want, const char* what) {
    if (b.group() != want) { std::fprintf(stderr, "mock: %s bound to group %d, expected %d\n", what, b.group(), want); std::abort(); }
  }
  run operator()(bo& a, bo& b, bo& c, int mt, int nt, int k) {           // SA: 6 args
    chk(a, 0, "A"); chk(b, 1, "B"); chk(c, 2, "C");
    gemm_sa(B_(a), B_(b), reinterpret_cast<beat_t*>(c.map<float*>()), mt, nt, k);
    return run();
  }
  run operator()(bo& a0, bo& a1, bo& a2, bo& a3, bo& a4, bo& a5, bo& a6, bo& a7, bo& b, bo& c, int mt, int nt, int k, int f) {  // SISA: 14
    bo* as[8] = {&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7};
    for (int i = 0; i < 8; ++i) chk(*as[i], i, "A_i");
    chk(b, 8, "B"); chk(c, 9, "C");
    gemm_sisa(B_(a0), B_(a1), B_(a2), B_(a3), B_(a4), B_(a5), B_(a6), B_(a7), B_(b),
              reinterpret_cast<beat_t*>(c.map<float*>()), mt, nt, k, f);
    return run();
  }
};
}  // namespace xrt
