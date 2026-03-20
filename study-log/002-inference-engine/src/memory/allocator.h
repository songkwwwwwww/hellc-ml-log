#pragma once

#include <cstddef>

namespace tie {

class Allocator {
 public:
  virtual ~Allocator() = default;
  virtual void* Allocate(std::size_t bytes) = 0;
  virtual void Deallocate(void* ptr) = 0;
};

}  // namespace tie
