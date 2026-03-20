#pragma once

#include <string>
#include <vector>

#include "../core/tensor.h"

namespace tie {

// Reads image data from an MNIST .ubyte file and returns a float32 Tensor.
// Pixel values are normalized from [0, 255] -> [0.0, 1.0].
class MnistInputLoader {
 public:
  // TODO(M1): Implementation pending.
  static Tensor Load(const std::string& path, Allocator* allocator);
};

}  // namespace tie
