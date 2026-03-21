#include "cpu_execution_provider.h"

#include "../kernels/cpu/cpu_flatten_kernel.h"
#include "../kernels/cpu/cpu_gemm_kernel.h"
#include "../kernels/cpu/cpu_relu_kernel.h"
#include "../core/tensor_store.h"

namespace tie {

CpuExecutionProvider::CpuExecutionProvider() {
  auto f32 = DataType::kFloat32;
  auto cpu = DeviceType::kCpu;

  registry_.Register(OpType::kFlatten, cpu, f32,
                     [] { return std::make_unique<CpuFlattenKernel>(); });
  registry_.Register(OpType::kGemm, cpu, f32,
                     [] { return std::make_unique<CpuGemmKernel>(); });
  registry_.Register(OpType::kRelu, cpu, f32,
                     [] { return std::make_unique<CpuReluKernel>(); });
}

bool CpuExecutionProvider::CanHandle(const Node& node) const {
  try {
    registry_.Lookup(node.op_type, DeviceType::kCpu, DataType::kFloat32);
    return true;
  } catch (...) {
    return false;
  }
}

// On CPU, weights are already in host memory, so no movement is necessary.
void CpuExecutionProvider::PrepareWeights(
    std::unordered_map<std::string, Tensor>& /*weights*/) {}

void CpuExecutionProvider::Execute(const Node& node, TensorStore& store) {
  auto kernel = registry_.Lookup(node.op_type, DeviceType::kCpu,
                                 DataType::kFloat32);

  std::vector<const Tensor*> inputs;
  for (const auto& name : node.inputs) {
    // Empty name means the optional input is absent (ONNX convention).
    inputs.push_back(name.empty() ? nullptr : store.Get(name));
  }

  std::vector<Tensor*> outputs;
  for (const auto& name : node.outputs) {
    outputs.push_back(store.GetMutable(name));
  }

  kernel->Compute(inputs, outputs, node.attributes);
}

}  // namespace tie
