#pragma once

#include "../kernel.h"

namespace tie {

// Relu: Y = max(0, X), element-wise
class CpuReluKernel : public Kernel {
 public:
  void Compute(const std::vector<const Tensor*>& inputs,
               const std::vector<Tensor*>& outputs,
               const AttributeMap& attrs) override;
};

}  // namespace tie
