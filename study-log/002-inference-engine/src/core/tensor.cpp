#include "tensor.h"

#include "../memory/allocator.h"

namespace tie {

Tensor::Tensor(TensorDesc desc, Allocator* allocator)
    : desc_(std::move(desc)), allocator_(allocator) {
  data_ = allocator_->Allocate(desc_.ByteSize());
}

Tensor::~Tensor() {
  if (data_ && allocator_) {
    allocator_->Deallocate(data_);
  }
}

Tensor::Tensor(Tensor&& other) noexcept
    : desc_(std::move(other.desc_)),
      data_(other.data_),
      allocator_(other.allocator_) {
  other.data_ = nullptr;
  other.allocator_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    if (data_ && allocator_) allocator_->Deallocate(data_);
    desc_ = std::move(other.desc_);
    data_ = other.data_;
    allocator_ = other.allocator_;
    other.data_ = nullptr;
    other.allocator_ = nullptr;
  }
  return *this;
}

}  // namespace tie
