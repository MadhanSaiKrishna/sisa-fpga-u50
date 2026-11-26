// Minimal stand-in for Vitis' ap_int.h so `make csim` runs with plain g++ (no Vitis).
// Only what the kernels use: ap_uint<W> (W multiple of 32), .range(hi,lo) on 32-bit-aligned slices,
// assignment of slices, shifts are NOT needed. Real Vitis builds never see this file.
#pragma once
#include <cstdint>
#include <cstring>

template <int W> struct ap_uint {
  static_assert(W % 32 == 0, "shim supports widths that are multiples of 32");
  uint32_t w[W / 32];
  ap_uint() { std::memset(w, 0, sizeof w); }
  ap_uint(int v) { std::memset(w, 0, sizeof w); w[0] = (uint32_t)v; }
  struct range_ref {
    uint32_t* p; int n;
    template <int N> range_ref& operator=(const ap_uint<N>& v) { for (int i = 0; i < n; ++i) p[i] = v.w[i]; return *this; }
    range_ref& operator=(uint32_t v) { p[0] = v; return *this; }
    uint32_t to_uint() const { return p[0]; }
    template <int N> operator ap_uint<N>() const { ap_uint<N> r; for (int i = 0; i < N / 32; ++i) r.w[i] = p[i]; return r; }
  };
  struct crange_ref {
    const uint32_t* p; int n;
    uint32_t to_uint() const { return p[0]; }
    template <int N> operator ap_uint<N>() const { ap_uint<N> r; for (int i = 0; i < N / 32; ++i) r.w[i] = p[i]; return r; }
  };
  range_ref range(int hi, int lo) { return range_ref{w + lo / 32, (hi - lo + 1) / 32}; }
  crange_ref range(int hi, int lo) const { return crange_ref{w + lo / 32, (hi - lo + 1) / 32}; }
  uint32_t to_uint() const { return w[0]; }
};
