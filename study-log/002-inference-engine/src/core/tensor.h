#pragma once

#include <memory>

#include "tensor_desc.h"

namespace tie {

class Allocator;

// A class that owns both TensorDesc (metadata) and the actual buffer.
// The buffer is managed through an Allocator.
class Tensor {
 public:
  Tensor() = default;
  Tensor(TensorDesc desc, Allocator* allocator);
  ~Tensor();

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&&) noexcept;
  Tensor& operator=(Tensor&&) noexcept;

  const TensorDesc& desc() const { return desc_; }
  const std::vector<int64_t>& shape() const { return desc_.shape; }
  DataType dtype() const { return desc_.dtype; }
  int64_t NumElements() const { return desc_.NumElements(); }

  void* data() { return data_; }
  const void* data() const { return data_; }

  template <typename T>
  T* data_as() { return static_cast<T*>(data_); }

  template <typename T>
  const T* data_as() const { return static_cast<const T*>(data_); }

 private:
  TensorDesc desc_;
  void* data_ = nullptr;
  Allocator* allocator_ = nullptr;
};

}  // namespace tie
