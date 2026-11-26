// mock_xrt: software stand-in for XRT so the slm harness can be run against the csim kernels (no board, no XRT).
#pragma once
#include <string>
namespace xrt {
struct uuid { int dummy = 0; };
class device {
 public:
  device() = default;
  explicit device(int) {}
  uuid load_xclbin(const std::string&) { return uuid(); }
};
}  // namespace xrt
