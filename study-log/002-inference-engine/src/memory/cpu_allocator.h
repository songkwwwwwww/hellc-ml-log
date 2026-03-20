#pragma once

#include "allocator.h"

namespace tie {

class CpuAllocator : public Allocator {
 public:
  void* Allocate(std::size_t bytes) override;
  void Deallocate(void* ptr) override;
};

}  // namespace tie
