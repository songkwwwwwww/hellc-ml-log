#pragma once

#include <memory>

#include "../core/tensor.h"
#include "../memory/allocator.h"
#include "../providers/execution_provider.h"
#include "execution_plan.h"

namespace tie {

class Session;

// The result of the build phase. Holds the ExecutionPlan and creates Sessions.
class Engine {
 public:
  Engine(ExecutionPlan plan, ExecutionProvider* provider);

  // Allocator is provided at session creation time so the same Engine can be
  // used with different allocators (e.g., per-request arenas in the future).
  std::unique_ptr<Session> CreateSession(Allocator* allocator);

  const ExecutionPlan& plan() const { return plan_; }
  ExecutionProvider* provider() { return provider_; }

 private:
  ExecutionPlan plan_;
  ExecutionProvider* provider_;
};

}  // namespace tie
