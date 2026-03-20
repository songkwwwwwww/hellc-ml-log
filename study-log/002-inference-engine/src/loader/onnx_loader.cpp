#include "onnx_loader.h"

#include <stdexcept>

namespace tie {

ModelData OnnxLoader::Load(const std::string& /*path*/,
                           Allocator* /*allocator*/) {
  // TODO(M1): Implement protobuf-based parsing.
  throw std::runtime_error("OnnxLoader::Load: not implemented yet (M1)");
}

}  // namespace tie
