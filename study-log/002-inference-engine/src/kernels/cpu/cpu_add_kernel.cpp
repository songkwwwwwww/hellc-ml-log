#include "cpu_add_kernel.h"

namespace tie {

// ONNX Add: Y = X0 + X1, element-wise (same shape, no broadcasting)
void CpuAddKernel::Compute(const std::vector<const Tensor*>& inputs,
                            const std::vector<Tensor*>& outputs,
                            const AttributeMap& /*attrs*/) {
  const float* a = inputs[0]->data_as<float>();
  const float* b = inputs[1]->data_as<float>();
  float* out = outputs[0]->data_as<float>();
  const int64_t n = inputs[0]->NumElements();
  for (int64_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
}

}  // namespace tie
