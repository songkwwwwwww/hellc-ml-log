#pragma once

#include <memory>

#include "../core/graph.h"
#include "../providers/execution_provider.h"

namespace tie {

class Engine;

// Responsible for the build phase.
// Takes a Graph and returns an Engine with kernel selection and memory planning completed.
class EngineBuilder {
 public:
  explicit EngineBuilder(ExecutionProvider* provider);

  // Graph -> Engine (Executes the entire build phase)
  std::unique_ptr<Engine> Build(const Graph& graph);

 private:
  ExecutionProvider* provider_;
};

}  // namespace tie
