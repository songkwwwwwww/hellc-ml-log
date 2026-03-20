#pragma once

#include <string>
#include <unordered_map>

#include "../core/tensor.h"
#include "../core/tensor_store.h"

namespace tie {

class Engine;

// Responsible for the run phase.
// Executes the ExecutionPlan held by the Engine in order.
class Session {
 public:
  explicit Session(Engine* engine);

  // Bind input tensor by name.
  void BindInput(const std::string& name, Tensor tensor);

  // Execute the ExecutionPlan in order.
  void Run();

  // Retrieve output tensor by name.
  const Tensor& GetOutput(const std::string& name) const;

 private:
  Engine* engine_;
  TensorStore store_;
};

}  // namespace tie
