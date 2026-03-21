#include "onnx_loader.h"

#include <fstream>
#include <stdexcept>
#include <unordered_set>

#include "study-log/002-inference-engine/src/loader/onnx-ml.pb.h"

namespace tie {

namespace {

OpType ParseOpType(const std::string& s) {
  if (s == "Flatten") return OpType::kFlatten;
  if (s == "Gemm") return OpType::kGemm;
  if (s == "Relu") return OpType::kRelu;
  if (s == "Add") return OpType::kAdd;
  if (s == "Conv") return OpType::kConv;
  if (s == "MaxPool") return OpType::kMaxPool;
  return OpType::kUnknown;
}

DataType ParseDataType(int32_t elem_type) {
  // onnx::TensorProto::FLOAT = 1, INT64 = 7
  switch (elem_type) {
    case 1: return DataType::kFloat32;
    case 7: return DataType::kInt64;
    default:
      throw std::runtime_error("OnnxLoader: unsupported elem_type " +
                               std::to_string(elem_type));
  }
}

TensorDesc ParseValueInfo(const onnx::ValueInfoProto& vi) {
  TensorDesc desc;
  desc.name = vi.name();
  desc.dtype = ParseDataType(vi.type().tensor_type().elem_type());
  for (const auto& dim : vi.type().tensor_type().shape().dim()) {
    desc.shape.push_back(dim.has_dim_value() ? dim.dim_value() : -1);
  }
  return desc;
}

}  // namespace

ModelData OnnxLoader::Load(const std::string& path, Allocator* allocator) {
  onnx::ModelProto model;
  {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      throw std::runtime_error("OnnxLoader: cannot open file: " + path);
    }
    if (!model.ParseFromIstream(&f)) {
      throw std::runtime_error("OnnxLoader: failed to parse ONNX model: " +
                               path);
    }
  }

  if (!model.has_graph()) {
    throw std::runtime_error("OnnxLoader: model has no graph");
  }

  const onnx::GraphProto& gp = model.graph();
  ModelData data;
  Graph& graph = data.graph;

  // ── 1. Collect initializer names (weights) ────────────────────────────────
  std::unordered_set<std::string> initializer_names;
  for (const auto& init : gp.initializer()) {
    initializer_names.insert(init.name());
  }

  // ── 2. Register TensorDescs for all value_info (intermediate tensors) ─────
  for (const auto& vi : gp.value_info()) {
    graph.SetTensorDesc(vi.name(), ParseValueInfo(vi));
  }
  // Graph inputs (excluding initializers = actual model inputs)
  std::vector<std::string> input_names;
  for (const auto& vi : gp.input()) {
    graph.SetTensorDesc(vi.name(), ParseValueInfo(vi));
    if (!initializer_names.count(vi.name())) {
      input_names.push_back(vi.name());
    }
  }
  // Graph outputs
  std::vector<std::string> output_names;
  for (const auto& vi : gp.output()) {
    graph.SetTensorDesc(vi.name(), ParseValueInfo(vi));
    output_names.push_back(vi.name());
  }
  graph.SetInputNames(std::move(input_names));
  graph.SetOutputNames(std::move(output_names));

  // ── 3. Build nodes ─────────────────────────────────────────────────────────
  for (int i = 0; i < gp.node_size(); ++i) {
    const onnx::NodeProto& np = gp.node(i);
    auto node = std::make_unique<Node>();
    node->name = np.name().empty() ? ("node_" + std::to_string(i)) : np.name();
    node->op_type = ParseOpType(np.op_type());

    for (const auto& inp : np.input()) {
      node->inputs.push_back(inp);  // empty string = optional absent input
    }
    for (const auto& out : np.output()) {
      node->outputs.push_back(out);
    }

    // Parse attributes
    for (const auto& attr : np.attribute()) {
      switch (attr.type()) {
        case onnx::AttributeProto::FLOAT:
          node->attributes[attr.name()] = attr.f();
          break;
        case onnx::AttributeProto::INT:
          node->attributes[attr.name()] =
              static_cast<int64_t>(attr.i());
          break;
        case onnx::AttributeProto::INTS: {
          std::vector<int64_t> vals(attr.ints().begin(), attr.ints().end());
          node->attributes[attr.name()] = std::move(vals);
          break;
        }
        default:
          break;  // Ignore unsupported attribute types for now
      }
    }

    graph.AddNode(std::move(node));
  }

  // ── 4. Load weight tensors from initializers ───────────────────────────────
  for (const auto& init : gp.initializer()) {
    TensorDesc desc;
    desc.name = init.name();
    desc.shape.assign(init.dims().begin(), init.dims().end());
    desc.dtype = ParseDataType(init.data_type());

    // Register initializer shapes into the graph so EngineBuilder can
    // use them for shape inference of downstream ops (e.g. Gemm output shape).
    graph.SetTensorDesc(init.name(), desc);

    Tensor tensor(desc, allocator);

    const int64_t n_bytes = desc.ByteSize();
    if (!init.raw_data().empty()) {
      std::memcpy(tensor.data(), init.raw_data().data(),
                  static_cast<size_t>(n_bytes));
    } else if (!init.float_data().empty()) {
      std::memcpy(tensor.data(), init.float_data().data(),
                  static_cast<size_t>(n_bytes));
    } else {
      throw std::runtime_error("OnnxLoader: initializer '" + init.name() +
                               "' has no data");
    }

    data.weights.emplace(init.name(), std::move(tensor));
  }

  return data;
}

}  // namespace tie
