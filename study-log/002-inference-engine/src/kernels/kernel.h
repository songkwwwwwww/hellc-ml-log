#pragma once

#include <vector>

#include "../core/node.h"
#include "../core/tensor.h"

namespace tie {

// Abstract interface responsible for the execution of a single operation.
// Providers do not own kernels directly; they look them up via KernelRegistry.
class Kernel {
 public:
  virtual ~Kernel() = default;

  // inputs:  List of input tensor pointers for this node
  // outputs: List of output tensor pointers for this node (already allocated)
  // attrs:   Node attributes (alpha, beta, transB, etc.)
  virtual void Compute(const std::vector<const Tensor*>& inputs,
                       const std::vector<Tensor*>& outputs,
                       const AttributeMap& attrs) = 0;
};

}  // namespace tie
