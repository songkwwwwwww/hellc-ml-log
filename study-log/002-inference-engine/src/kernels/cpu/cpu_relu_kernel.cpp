#include "cpu_relu_kernel.h"

#include <algorithm>

namespace tie {

// ONNX Relu: Y = max(0, X), element-wise
void CpuReluKernel::Compute(const std::vector<const Tensor*>& inputs,
                             const std::vector<Tensor*>& outputs,
                             const AttributeMap& /*attrs*/) {
  const float* in = inputs[0]->data_as<float>();
  float* out = outputs[0]->data_as<float>();
  const int64_t n = inputs[0]->NumElements();
  for (int64_t i = 0; i < n; ++i) {
    out[i] = std::max(0.0f, in[i]);
  }
}

}  // namespace tie
