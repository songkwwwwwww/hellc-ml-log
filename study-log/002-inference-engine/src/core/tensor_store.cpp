#include "tensor_store.h"

#include <stdexcept>

namespace tie {

void TensorStore::Set(const std::string& name, Tensor tensor) {
  tensors_.insert_or_assign(name, std::move(tensor));
}

const Tensor* TensorStore::Get(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::runtime_error("TensorStore: tensor not found: " + name);
  }
  return &it->second;
}

Tensor* TensorStore::GetMutable(const std::string& name) {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::runtime_error("TensorStore: tensor not found: " + name);
  }
  return &it->second;
}

bool TensorStore::Has(const std::string& name) const {
  return tensors_.count(name) > 0;
}

}  // namespace tie
