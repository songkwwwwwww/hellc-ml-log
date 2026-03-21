#include <gtest/gtest.h>

#include "../src/kernels/cpu/cpu_flatten_kernel.h"
#include "../src/kernels/cpu/cpu_gemm_kernel.h"
#include "../src/kernels/cpu/cpu_relu_kernel.h"
#include "../src/memory/cpu_allocator.h"

namespace tie {

// ── Helper ────────────────────────────────────────────────────────────────────

static Tensor MakeTensor(const std::vector<int64_t>& shape,
                          const std::vector<float>& data,
                          CpuAllocator* alloc) {
  TensorDesc desc{"", shape, DataType::kFloat32};
  Tensor t(desc, alloc);
  std::copy(data.begin(), data.end(), t.data_as<float>());
  return t;
}

// ── Relu ──────────────────────────────────────────────────────────────────────

TEST(CpuReluKernelTest, ElementWiseMax0) {
  CpuAllocator alloc;
  Tensor in = MakeTensor({4}, {-2.0f, 0.0f, 1.5f, -0.5f}, &alloc);
  TensorDesc out_desc{"", {4}, DataType::kFloat32};
  Tensor out(out_desc, &alloc);

  CpuReluKernel kernel;
  kernel.Compute({&in}, {&out}, {});

  const float* y = out.data_as<float>();
  EXPECT_FLOAT_EQ(y[0], 0.0f);
  EXPECT_FLOAT_EQ(y[1], 0.0f);
  EXPECT_FLOAT_EQ(y[2], 1.5f);
  EXPECT_FLOAT_EQ(y[3], 0.0f);
}

// ── Gemm ──────────────────────────────────────────────────────────────────────

// Y = A * B  (no bias, no transpose)
// A [2,3] * B [3,2] -> Y [2,2]
TEST(CpuGemmKernelTest, NoBiasNoTranspose) {
  CpuAllocator alloc;
  // A = [[1,2,3],[4,5,6]]
  Tensor A = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6}, &alloc);
  // B = [[7,8],[9,10],[11,12]]
  Tensor B = MakeTensor({3, 2}, {7, 8, 9, 10, 11, 12}, &alloc);
  TensorDesc out_desc{"", {2, 2}, DataType::kFloat32};
  Tensor Y(out_desc, &alloc);

  CpuGemmKernel kernel;
  kernel.Compute({&A, &B, nullptr}, {&Y}, {});

  // Y[0,0] = 1*7 + 2*9 + 3*11 = 58
  // Y[0,1] = 1*8 + 2*10 + 3*12 = 64
  // Y[1,0] = 4*7 + 5*9 + 6*11 = 139
  // Y[1,1] = 4*8 + 5*10 + 6*12 = 154
  const float* y = Y.data_as<float>();
  EXPECT_FLOAT_EQ(y[0], 58.0f);
  EXPECT_FLOAT_EQ(y[1], 64.0f);
  EXPECT_FLOAT_EQ(y[2], 139.0f);
  EXPECT_FLOAT_EQ(y[3], 154.0f);
}

// Y = A * B^T + C  (transB=1, with bias)
// Typical ONNX FFN layer: A [1,3], B stored as [2,3] (transB), bias [2]
TEST(CpuGemmKernelTest, TransBWithBias) {
  CpuAllocator alloc;
  Tensor A = MakeTensor({1, 3}, {1, 2, 3}, &alloc);
  // B stored as [2,3]; B^T = [3,2]
  Tensor B = MakeTensor({2, 3}, {1, 0, 0, 0, 1, 0}, &alloc);
  Tensor C = MakeTensor({2}, {10.0f, 20.0f}, &alloc);
  TensorDesc out_desc{"", {1, 2}, DataType::kFloat32};
  Tensor Y(out_desc, &alloc);

  AttributeMap attrs;
  attrs["transB"] = int64_t{1};

  CpuGemmKernel kernel;
  kernel.Compute({&A, &B, &C}, {&Y}, attrs);

  // A * B^T = [1,2,3] * [[1,0],[0,1],[0,0]]^T... wait let me recalc.
  // B = [[1,0,0],[0,1,0]], transB means we use B^T:
  // B^T = [[1,0],[0,1],[0,0]]
  // A * B^T = [1*1+2*0+3*0, 1*0+2*1+3*0] = [1, 2]
  // + bias [10, 20] -> [11, 22]
  const float* y = Y.data_as<float>();
  EXPECT_FLOAT_EQ(y[0], 11.0f);
  EXPECT_FLOAT_EQ(y[1], 22.0f);
}

// ── Flatten ───────────────────────────────────────────────────────────────────

TEST(CpuFlattenKernelTest, CopiesData) {
  CpuAllocator alloc;
  Tensor in = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6}, &alloc);
  TensorDesc out_desc{"", {1, 6}, DataType::kFloat32};
  Tensor out(out_desc, &alloc);

  CpuFlattenKernel kernel;
  kernel.Compute({&in}, {&out}, {});

  const float* y = out.data_as<float>();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(y[i], static_cast<float>(i + 1));
  }
}

}  // namespace tie
