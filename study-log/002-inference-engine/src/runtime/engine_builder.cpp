#include "engine_builder.h"

#include <stdexcept>

#include "engine.h"

namespace tie {

EngineBuilder::EngineBuilder(ExecutionProvider* provider)
    : provider_(provider) {}

std::unique_ptr<Engine> EngineBuilder::Build(const Graph& graph) {
  // TODO(M1): Add shape inference and memory planning.
  ExecutionPlan plan;

  for (const Node* node : graph.TopologicallySortedNodes()) {
    if (!provider_->CanHandle(*node)) {
      throw std::runtime_error("EngineBuilder: unsupported op: " + node->name);
    }
    PlanStep step;
    step.node = node;
    step.provider = provider_;
    plan.steps.push_back(std::move(step));
  }

  return std::make_unique<Engine>(std::move(plan), provider_);
}

}  // namespace tie
