#pragma once

#include <string>
#include <unordered_map>

#include "../core/graph.h"
#include "../core/tensor.h"

namespace tie {

struct ModelData {
  Graph graph;
  std::unordered_map<std::string, Tensor> weights;  // Initializer tensors
};

// Protobuf-based ONNX loader (M1 implementation pending).
// Parses a .onnx file and returns a Graph and weight tensors.
class OnnxLoader {
 public:
  // TODO(M1): Implementation pending.
  static ModelData Load(const std::string& path, Allocator* allocator);
};

}  // namespace tie
