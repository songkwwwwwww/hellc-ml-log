#include "cpu_gemm_kernel.h"

#include <stdexcept>

namespace tie {

// ONNX Gemm: Y = alpha * op(A) * op(B) + beta * C
//   op(X) = X  if transX = 0
//   op(X) = X' if transX = 1
//
// inputs[0] = A, inputs[1] = B, inputs[2] = C (bias, optional)
// outputs[0] = Y
//
// A stored as [M, K] (transA=0) or [K, M] (transA=1)
// B stored as [K, N] (transB=0) or [N, K] (transB=1)
// C shape: [N]  (broadcast across M)
// Y shape: [M, N]
void CpuGemmKernel::Compute(const std::vector<const Tensor*>& inputs,
                             const std::vector<Tensor*>& outputs,
                             const AttributeMap& attrs) {
  if (inputs.size() < 2 || outputs.empty()) {
    throw std::runtime_error("CpuGemmKernel: invalid number of inputs/outputs");
  }

  const Tensor* A = inputs[0];
  const Tensor* B = inputs[1];
  const Tensor* C =
      (inputs.size() > 2 && inputs[2] != nullptr) ? inputs[2] : nullptr;
  Tensor* Y = outputs[0];

  float alpha = 1.0f;
  float beta = 1.0f;
  int64_t trans_a = 0;
  int64_t trans_b = 0;

  if (auto it = attrs.find("alpha"); it != attrs.end()) {
    alpha = std::get<float>(it->second);
  }
  if (auto it = attrs.find("beta"); it != attrs.end()) {
    beta = std::get<float>(it->second);
  }
  if (auto it = attrs.find("transA"); it != attrs.end()) {
    trans_a = std::get<int64_t>(it->second);
  }
  if (auto it = attrs.find("transB"); it != attrs.end()) {
    trans_b = std::get<int64_t>(it->second);
  }

  const int64_t M = trans_a ? A->shape()[1] : A->shape()[0];
  const int64_t K = trans_a ? A->shape()[0] : A->shape()[1];
  const int64_t N = trans_b ? B->shape()[0] : B->shape()[1];

  const float* a = A->data_as<float>();
  const float* b = B->data_as<float>();
  float* y = Y->data_as<float>();

  // Naive triple-loop GEMM: Y = alpha * op(A) * op(B)
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        const float av = trans_a ? a[k * M + m] : a[m * K + k];
        const float bv = trans_b ? b[n * K + k] : b[k * N + n];
        sum += av * bv;
      }
      y[m * N + n] = alpha * sum;
    }
  }

  // Add bias: Y += beta * C  (C is [N], broadcast across M)
  if (C != nullptr) {
    const float* c = C->data_as<float>();
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        y[m * N + n] += beta * c[n];
      }
    }
  }
}

}  // namespace tie
