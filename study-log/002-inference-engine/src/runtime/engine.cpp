#include "engine.h"

#include "session.h"

namespace tie {

Engine::Engine(ExecutionPlan plan, ExecutionProvider* provider)
    : plan_(std::move(plan)), provider_(provider) {}

std::unique_ptr<Session> Engine::CreateSession() {
  return std::make_unique<Session>(this);
}

}  // namespace tie
