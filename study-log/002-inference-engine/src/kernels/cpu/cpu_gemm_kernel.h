#pragma once

#include "../kernel.h"

namespace tie {

// Gemm: Y = alpha * A * B + beta * C
// ONNX Gemm spec: inputs[0]=A, inputs[1]=B, inputs[2]=C(optional)
// attributes: alpha(float), beta(float), transA(int), transB(int)
class CpuGemmKernel : public Kernel {
 public:
  void Compute(const std::vector<const Tensor*>& inputs,
               const std::vector<Tensor*>& outputs,
               const AttributeMap& attrs) override;
};

}  // namespace tie
