#include "cpu_allocator.h"

#include <cstdlib>
#include <new>

namespace tie {

void* CpuAllocator::Allocate(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  void* ptr = std::malloc(bytes);
  if (!ptr) throw std::bad_alloc{};
  return ptr;
}

void CpuAllocator::Deallocate(void* ptr) { std::free(ptr); }

}  // namespace tie
