#pragma once

#include "../kernel.h"

namespace tie {

class CpuFlattenKernel : public Kernel {
 public:
  void Compute(const std::vector<const Tensor*>& inputs,
               const std::vector<Tensor*>& outputs,
               const AttributeMap& attrs) override;
};

}  // namespace tie
