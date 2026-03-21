#include "mnist_input_loader.h"

#include <fstream>
#include <stdexcept>

namespace tie {

// Each .ubyte file is raw pixel bytes with no header (784 bytes for 28x28).
// Pixel values are kept in [0, 255] to match the preprocessing used when
// training the reference mnist_ffn.onnx model.
Tensor MnistInputLoader::Load(const std::string& path, Allocator* allocator) {
  constexpr int64_t kPixels = 784;  // 28 * 28

  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("MnistInputLoader: cannot open file: " + path);
  }

  unsigned char raw[kPixels];
  f.read(reinterpret_cast<char*>(raw), kPixels);
  if (f.gcount() != kPixels) {
    throw std::runtime_error("MnistInputLoader: unexpected file size: " + path);
  }

  TensorDesc desc;
  desc.name = "input";
  desc.shape = {1, kPixels};
  desc.dtype = DataType::kFloat32;

  Tensor tensor(desc, allocator);
  float* dst = tensor.data_as<float>();
  for (int64_t i = 0; i < kPixels; ++i) {
    dst[i] = static_cast<float>(raw[i]);
  }
  return tensor;
}

}  // namespace tie
