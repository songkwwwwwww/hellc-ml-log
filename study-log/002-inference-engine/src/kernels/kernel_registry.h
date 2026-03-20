#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "../core/optype.h"
#include "../core/tensor_desc.h"
#include "kernel.h"

namespace tie {

// Device type (e.g., kCuda to be added later)
enum class DeviceType { kCpu };

struct KernelKey {
  OpType op_type;
  DeviceType device;
  DataType dtype;

  bool operator==(const KernelKey& o) const {
    return op_type == o.op_type && device == o.device && dtype == o.dtype;
  }
};

struct KernelKeyHash {
  std::size_t operator()(const KernelKey& k) const {
    auto h = [](auto v) { return std::hash<int>{}(static_cast<int>(v)); };
    return h(k.op_type) ^ (h(k.device) << 8) ^ (h(k.dtype) << 16);
  }
};

// Looks up a Kernel using a combination of OpType + DeviceType + DataType.
// If lookup fails at build time, it immediately throws an exception for early detection of unsupported ops.
class KernelRegistry {
 public:
  using Factory = std::function<std::unique_ptr<Kernel>()>;

  void Register(OpType op, DeviceType device, DataType dtype, Factory factory);

  // Called at build time. Throws an exception for unsupported combinations.
  std::unique_ptr<Kernel> Lookup(OpType op, DeviceType device,
                                 DataType dtype) const;

 private:
  std::unordered_map<KernelKey, Factory, KernelKeyHash> registry_;
};

}  // namespace tie
