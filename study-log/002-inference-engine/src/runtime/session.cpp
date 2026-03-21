#include "session.h"

#include "engine.h"

namespace tie {

Session::Session(const Engine* engine, Allocator* allocator)
    : engine_(engine), allocator_(allocator) {}

void Session::BindInput(const std::string& name, Tensor tensor) {
  store_.Set(name, std::move(tensor));
}

void Session::Run() {
  for (const PlanStep& step : engine_->plan().steps) {
    // Pre-allocate output tensors that don't yet exist in the store.
    for (const TensorDesc& desc : step.output_descs) {
      if (!store_.Has(desc.name)) {
        store_.Set(desc.name, Tensor(desc, allocator_));
      }
    }
    step.provider->Execute(*step.node, store_);
  }
}

const Tensor& Session::GetOutput(const std::string& name) const {
  return *store_.Get(name);
}

}  // namespace tie
