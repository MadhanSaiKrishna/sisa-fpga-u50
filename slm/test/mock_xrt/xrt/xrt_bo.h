#pragma once
#include <cstdlib>
#include <cstring>
#include <memory>
#include "xrt/xrt_device.h"
enum xclBOSyncDirection { XCL_BO_SYNC_BO_TO_DEVICE = 0, XCL_BO_SYNC_BO_FROM_DEVICE = 1 };
namespace xrt {
class bo {
  struct Buf { void* p = nullptr; size_t n = 0; int grp = -1; ~Buf() { std::free(p); } };
  std::shared_ptr<Buf> b_;
 public:
  bo() = default;
  bo(device&, size_t bytes, int group) : b_(std::make_shared<Buf>()) {
    b_->n = (bytes + 63) / 64 * 64;
    b_->p = std::aligned_alloc(64, b_->n);
    std::memset(b_->p, 0xCD, b_->n);        // poison: a kernel reading unwritten data would show up
    b_->grp = group;
  }
  template <typename T> T map() { return reinterpret_cast<T>(b_->p); }
  void sync(xclBOSyncDirection) {}
  int group() const { return b_->grp; }
};
}  // namespace xrt
