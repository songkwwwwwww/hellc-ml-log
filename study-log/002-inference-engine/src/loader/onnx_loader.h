#pragma once

#include <string>
#include <unordered_map>

#include "../core/graph.h"
#include "../core/tensor.h"

namespace tie {

struct ModelData {
  Graph graph;
  // Weight tensors loaded from ONNX initializers.
  // Bind all of these into the Session before calling Run().
  std::unordered_map<std::string, Tensor> weights;
};

// Protobuf-based ONNX loader (v1).
// Parses a .onnx file and returns the computation graph and weight tensors.
class OnnxLoader {
 public:
  static ModelData Load(const std::string& path, Allocator* allocator);
};

}  // namespace tie
