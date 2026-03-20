#include <gtest/gtest.h>

#include "../src/core/tensor.h"
#include "../src/core/tensor_desc.h"
#include "../src/memory/cpu_allocator.h"

namespace tie {

TEST(TensorDescTest, NumElements) {
  TensorDesc desc;
  desc.shape = {2, 3, 4};
  EXPECT_EQ(desc.NumElements(), 24);
}

TEST(TensorDescTest, ByteSize_Float32) {
  TensorDesc desc;
  desc.shape = {4};
  desc.dtype = DataType::kFloat32;
  EXPECT_EQ(desc.ByteSize(), 16);
}

TEST(TensorTest, AllocateAndAccess) {
  CpuAllocator alloc;
  TensorDesc desc{"x", {3}, DataType::kFloat32};
  Tensor t(desc, &alloc);

  float* p = t.data_as<float>();
  p[0] = 1.0f;
  p[1] = 2.0f;
  p[2] = 3.0f;

  EXPECT_FLOAT_EQ(t.data_as<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(t.data_as<float>()[2], 3.0f);
}

TEST(TensorTest, MoveSemantics) {
  CpuAllocator alloc;
  TensorDesc desc{"x", {2}, DataType::kFloat32};
  Tensor t1(desc, &alloc);
  t1.data_as<float>()[0] = 42.0f;

  Tensor t2 = std::move(t1);
  EXPECT_FLOAT_EQ(t2.data_as<float>()[0], 42.0f);
  EXPECT_EQ(t1.data(), nullptr);
}

}  // namespace tie
