#pragma once

#include <string>
#include <unordered_map>

#include "../core/node.h"
#include "../core/tensor.h"
#include "../core/tensor_store.h"
#include "../memory/allocator.h"

namespace tie {

// Device strategy layer. Does not directly contain operation implementations (Kernels).
//
// Provider responsibilities:
//   - Declare ops that can be handled on this device (CanHandle)
//   - Move constant weights to device memory (PrepareWeights)
//   - Execute a single node - Fetch a kernel from KernelRegistry and write to TensorStore (Execute)
//   - Provide a memory allocator for this device (GetAllocator)
class ExecutionProvider {
 public:
  virtual ~ExecutionProvider() = default;

  virtual std::string Name() const = 0;

  // Build phase: Check if this provider can handle the given node.
  virtual bool CanHandle(const Node& node) const = 0;

  // Build phase: Prepare weight tensors in device memory.
  virtual void PrepareWeights(
      std::unordered_map<std::string, Tensor>& weights) = 0;

  // Run phase: Execute the node and store the output in TensorStore.
  virtual void Execute(const Node& node, TensorStore& store) = 0;

  virtual Allocator* GetAllocator() = 0;
};

}  // namespace tie
