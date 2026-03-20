#pragma once

#include <string>
#include <unordered_map>

#include "tensor.h"

namespace tie {

// Manages all tensors (inputs, weights, intermediate, outputs) by name during execution.
class TensorStore {
 public:
  void Set(const std::string& name, Tensor tensor);
  const Tensor* Get(const std::string& name) const;
  Tensor* GetMutable(const std::string& name);
  bool Has(const std::string& name) const;

 private:
  std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace tie
