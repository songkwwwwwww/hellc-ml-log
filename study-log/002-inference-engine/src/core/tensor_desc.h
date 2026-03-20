#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tie {

enum class DataType {
  kFloat32,
  kInt64,
};

// A descriptor that only contains Tensor metadata. It does not own the actual buffer.
// Shape inference and memory planning can be performed using only TensorDesc.
struct TensorDesc {
  std::string name;
  std::vector<int64_t> shape;
  DataType dtype = DataType::kFloat32;

  int64_t NumElements() const;
  int64_t ByteSize() const;
};

}  // namespace tie
