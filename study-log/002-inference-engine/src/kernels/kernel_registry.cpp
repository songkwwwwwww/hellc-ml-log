#include "kernel_registry.h"

namespace tie {

void KernelRegistry::Register(OpType op, DeviceType device, DataType dtype,
                               Factory factory) {
  registry_[{op, device, dtype}] = std::move(factory);
}

std::unique_ptr<Kernel> KernelRegistry::Lookup(OpType op, DeviceType device,
                                                DataType dtype) const {
  auto it = registry_.find({op, device, dtype});
  if (it == registry_.end()) {
    throw std::runtime_error("No kernel registered for the given op/device/dtype");
  }
  return it->second();
}

}  // namespace tie
