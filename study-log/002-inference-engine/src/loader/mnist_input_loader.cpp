#include "mnist_input_loader.h"

#include <stdexcept>

namespace tie {

Tensor MnistInputLoader::Load(const std::string& /*path*/,
                              Allocator* /*allocator*/) {
  // TODO(M1): Implement .ubyte parsing.
  throw std::runtime_error("MnistInputLoader::Load: not implemented yet (M1)");
}

}  // namespace tie
