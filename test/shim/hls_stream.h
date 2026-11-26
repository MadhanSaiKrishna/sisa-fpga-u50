// Minimal stand-in for hls_stream.h (unbounded FIFO, like C simulation). Real Vitis builds never see this.
#pragma once
#include <deque>
namespace hls {
template <class T, int D = 0> class stream {
  std::deque<T> q;
 public:
  stream() {}
  explicit stream(const char*) {}
  void write(const T& v) { q.push_back(v); }
  T read() { T v = q.front(); q.pop_front(); return v; }
  bool empty() const { return q.empty(); }
  unsigned size() const { return (unsigned)q.size(); }
};
}  // namespace hls
