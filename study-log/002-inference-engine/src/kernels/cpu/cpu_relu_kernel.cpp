#include "cpu_relu_kernel.h"

#include <stdexcept>

namespace tie {

// TODO(M1): Implement element-wise max(0, x).
void CpuReluKernel::Compute(const std::vector<const Tensor*>& /*inputs*/,
                             const std::vector<Tensor*>& /*outputs*/,
                             const AttributeMap& /*attrs*/) {
  throw std::runtime_error("CpuReluKernel: not implemented yet (M1)");
}

}  // namespace tie
