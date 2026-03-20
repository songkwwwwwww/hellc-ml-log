#include "session.h"

#include "engine.h"

namespace tie {

Session::Session(Engine* engine) : engine_(engine) {}

void Session::BindInput(const std::string& name, Tensor tensor) {
  store_.Set(name, std::move(tensor));
}

void Session::Run() {
  for (const PlanStep& step : engine_->plan().steps) {
    step.provider->Execute(*step.node, store_);
  }
}

const Tensor& Session::GetOutput(const std::string& name) const {
  return *store_.Get(name);
}

}  // namespace tie
