#pragma once

#include "../kernel.h"

namespace tie {

// Add: Y = X0 + X1, element-wise
class CpuAddKernel : public Kernel {
 public:
  void Compute(const std::vector<const Tensor*>& inputs,
               const std::vector<Tensor*>& outputs,
               const AttributeMap& attrs) override;
};

}  // namespace tie
