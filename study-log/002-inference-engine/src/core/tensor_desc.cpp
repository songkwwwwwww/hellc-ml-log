#include "tensor_desc.h"

#include <numeric>
#include <stdexcept>

namespace tie {

int64_t TensorDesc::NumElements() const {
  if (shape.empty()) return 0;
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>{});
}

int64_t TensorDesc::ByteSize() const {
  switch (dtype) {
    case DataType::kFloat32:
      return NumElements() * 4;
    case DataType::kInt64:
      return NumElements() * 8;
  }
  throw std::runtime_error("Unknown DataType");
}

}  // namespace tie
