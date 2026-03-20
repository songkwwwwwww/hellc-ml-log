#include "cpu_gemm_kernel.h"

#include <stdexcept>

namespace tie {

// TODO(M1): Implement naive triple-loop GEMM.
void CpuGemmKernel::Compute(const std::vector<const Tensor*>& /*inputs*/,
                             const std::vector<Tensor*>& /*outputs*/,
                             const AttributeMap& /*attrs*/) {
  throw std::runtime_error("CpuGemmKernel: not implemented yet (M1)");
}

}  // namespace tie
