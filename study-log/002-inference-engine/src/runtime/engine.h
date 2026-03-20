#pragma once

#include <memory>
#include <unordered_map>

#include "../core/tensor.h"
#include "../providers/execution_provider.h"
#include "execution_plan.h"

namespace tie {

class Session;

// The result of the build phase. Holds the ExecutionPlan and creates a Session.
class Engine {
 public:
  Engine(ExecutionPlan plan, ExecutionProvider* provider);

  std::unique_ptr<Session> CreateSession();

  const ExecutionPlan& plan() const { return plan_; }
  ExecutionProvider* provider() { return provider_; }

 private:
  ExecutionPlan plan_;
  ExecutionProvider* provider_;
};

}  // namespace tie
