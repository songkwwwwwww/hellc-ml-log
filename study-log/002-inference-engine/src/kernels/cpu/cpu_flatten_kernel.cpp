#include "cpu_flatten_kernel.h"

#include <cstring>
#include <stdexcept>

namespace tie {

// Flatten: Copies the input tensor to 1D (data layout remains the same, only the shape changes).
// The output buffer is already allocated with the correct shape in the TensorStore.
void CpuFlattenKernel::Compute(const std::vector<const Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                const AttributeMap& /*attrs*/) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    throw std::runtime_error("CpuFlattenKernel: expected 1 input and 1 output");
  }
  const Tensor* in = inputs[0];
  Tensor* out = outputs[0];
  std::memcpy(out->data(), in->data(),
              static_cast<std::size_t>(in->desc().ByteSize()));
}

}  // namespace tie
