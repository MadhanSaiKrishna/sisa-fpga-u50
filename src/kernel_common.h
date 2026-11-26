// kernel_common.h -- Vitis-only types/helpers for the two kernels (not included by the host).
#pragma once
// ap_uint<2048> (one B row / C cell) exceeds the default 1024-bit limit of the C simulation headers.
#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W 2048
#endif
#include <ap_int.h>
#include <hls_stream.h>
#include "common.h"

typedef ap_uint<512> beat_t;               // one HBM beat = 16 floats
typedef ap_uint<SLAB_H * 32> bslice_t;     // 256 bits = one slab's 8 B values for one k
typedef ap_uint<ROWS * 32> row_t;          // 2048 bits = 64 floats: one B row (all token rows) / one C cell
#define BEATS_PER_ROW (ROWS / BEAT_FLOATS)            // 4
#define BEATS_PER_K   ((T_ACC * COLS) / BEAT_FLOATS)  // 4 : A beats per k
#define BEATS_PER_CELL (CELL_FLOATS / BEAT_FLOATS)    // 4

static inline float u2f(uint32_t u) { union { uint32_t i; float f; } x; x.i = u; return x.f; }
static inline uint32_t f2u(float f) { union { uint32_t i; float f; } x; x.f = f; return x.i; }

// explicit 8:1 select (comparison chain, not a dynamic array index -- see docs/v2_plan.md "slab-1 bug")
template <typename T>
static inline T mux8(int s, const T& a0, const T& a1, const T& a2, const T& a3,
                     const T& a4, const T& a5, const T& a6, const T& a7) {
  return s == 0 ? a0 : s == 1 ? a1 : s == 2 ? a2 : s == 3 ? a3 : s == 4 ? a4 : s == 5 ? a5 : s == 6 ? a6 : a7;
}

#ifdef SIM_COUNT
extern unsigned long long g_iters, g_iters_sa;   // compute iterations executed (checked vs Geo::compute_iters in csim)
#endif
