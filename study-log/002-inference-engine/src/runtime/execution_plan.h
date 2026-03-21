#pragma once

#include <memory>
#include <vector>

#include "../core/node.h"
#include "../core/tensor_desc.h"
#include "../kernels/kernel.h"

namespace tie {

class ExecutionProvider;

// A single execution unit of an ExecutionPlan.
// Generated in the build phase and strictly followed by the run phase.
struct PlanStep {
  const Node* node = nullptr;
  ExecutionProvider* provider = nullptr;
  std::unique_ptr<Kernel> kernel;
  // Pre-computed output descriptors for buffer allocation before kernel runs.
  std::vector<TensorDesc> output_descs;
};

// The result of the build phase. A list of PlanSteps sorted in topological order.
struct ExecutionPlan {
  std::vector<PlanStep> steps;
};

}  // namespace tie
