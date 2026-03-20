#pragma once

#include <memory>

#include "../kernels/kernel_registry.h"
#include "../memory/cpu_allocator.h"
#include "execution_provider.h"

namespace tie {

class CpuExecutionProvider : public ExecutionProvider {
 public:
  CpuExecutionProvider();

  std::string Name() const override { return "CpuExecutionProvider"; }
  bool CanHandle(const Node& node) const override;
  void PrepareWeights(std::unordered_map<std::string, Tensor>& weights) override;
  void Execute(const Node& node, TensorStore& store) override;
  Allocator* GetAllocator() override { return &allocator_; }

 private:
  CpuAllocator allocator_;
  KernelRegistry registry_;
};

}  // namespace tie
