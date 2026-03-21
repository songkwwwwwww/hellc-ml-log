#pragma once

#include <string>

#include "../core/tensor.h"
#include "../core/tensor_store.h"
#include "../memory/allocator.h"

namespace tie {

class Engine;

// Responsible for the run phase.
// Executes the ExecutionPlan held by the Engine in order.
class Session {
 public:
  Session(const Engine* engine, Allocator* allocator);

  // Bind a tensor (input data or pre-loaded weight) by name.
  void BindInput(const std::string& name, Tensor tensor);

  // Execute the ExecutionPlan in order.
  // Output tensors are pre-allocated from each PlanStep's output_descs.
  void Run();

  // Retrieve output tensor by name.
  const Tensor& GetOutput(const std::string& name) const;

 private:
  const Engine* engine_;
  Allocator* allocator_;
  TensorStore store_;
};

}  // namespace tie
