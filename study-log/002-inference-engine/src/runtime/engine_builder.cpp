#include "engine_builder.h"

#include <stdexcept>
#include <unordered_map>

#include "engine.h"

namespace tie {

namespace {

// Returns the output TensorDesc for a given node given the shapes of its inputs.
// Updates `known` with the new descs so downstream nodes can use them.
std::vector<TensorDesc> InferOutputShapes(
    const Node& node,
    std::unordered_map<std::string, TensorDesc>& known) {

  auto get = [&](const std::string& name) -> const TensorDesc& {
    auto it = known.find(name);
    if (it == known.end()) {
      throw std::runtime_error(
          "ShapeInference: unknown tensor '" + name + "'");
    }
    return it->second;
  };

  std::vector<TensorDesc> out_descs;

  switch (node.op_type) {
    case OpType::kFlatten: {
      // ONNX Flatten(axis=1): output = [shape[0], prod(shape[1:])]
      const TensorDesc& in = get(node.inputs[0]);
      int64_t axis = 1;
      if (auto it = node.attributes.find("axis"); it != node.attributes.end()) {
        axis = std::get<int64_t>(it->second);
      }
      int64_t outer = 1;
      for (int64_t i = 0; i < axis; ++i) outer *= in.shape[i];
      int64_t inner = 1;
      for (int64_t i = axis; i < static_cast<int64_t>(in.shape.size()); ++i) {
        inner *= in.shape[i];
      }
      TensorDesc d{node.outputs[0], {outer, inner}, in.dtype};
      known[d.name] = d;
      out_descs.push_back(d);
      break;
    }

    case OpType::kGemm: {
      // Y = alpha * op(A) * op(B) + beta * C
      // Output shape: [M, N]
      const TensorDesc& A = get(node.inputs[0]);
      const TensorDesc& B = get(node.inputs[1]);
      int64_t trans_a = 0, trans_b = 0;
      if (auto it = node.attributes.find("transA");
          it != node.attributes.end()) {
        trans_a = std::get<int64_t>(it->second);
      }
      if (auto it = node.attributes.find("transB");
          it != node.attributes.end()) {
        trans_b = std::get<int64_t>(it->second);
      }
      const int64_t M = trans_a ? A.shape[1] : A.shape[0];
      const int64_t N = trans_b ? B.shape[0] : B.shape[1];
      TensorDesc d{node.outputs[0], {M, N}, A.dtype};
      known[d.name] = d;
      out_descs.push_back(d);
      break;
    }

    case OpType::kRelu:
    case OpType::kAdd: {
      // Same shape as first input.
      const TensorDesc& in = get(node.inputs[0]);
      TensorDesc d{node.outputs[0], in.shape, in.dtype};
      known[d.name] = d;
      out_descs.push_back(d);
      break;
    }

    default:
      throw std::runtime_error(
          "ShapeInference: unsupported op '" + node.name + "'");
  }

  return out_descs;
}

}  // namespace

EngineBuilder::EngineBuilder(ExecutionProvider* provider)
    : provider_(provider) {}

std::unique_ptr<Engine> EngineBuilder::Build(const Graph& graph) {
  // Seed the shape inference map with all TensorDescs already in the graph
  // (model inputs, initializers, value_info from the ONNX proto).
  std::unordered_map<std::string, TensorDesc> known_shapes;
  for (const auto& node_ptr : graph.Nodes()) {
    const Node* node = node_ptr.get();
    for (const auto& name : node->inputs) {
      if (name.empty()) continue;
      if (const TensorDesc* d = graph.GetTensorDesc(name); d != nullptr) {
        known_shapes[name] = *d;
      }
    }
    for (const auto& name : node->outputs) {
      if (name.empty()) continue;
      if (const TensorDesc* d = graph.GetTensorDesc(name); d != nullptr) {
        known_shapes[name] = *d;
      }
    }
  }

  ExecutionPlan plan;

  for (const Node* node : graph.TopologicallySortedNodes()) {
    if (!provider_->CanHandle(*node)) {
      throw std::runtime_error("EngineBuilder: unsupported op '" + node->name +
                               "'");
    }

    PlanStep step;
    step.node = node;
    step.provider = provider_;

    // Try to get output descs from the graph (pre-computed via ONNX value_info).
    // Fall back to op-level shape inference if not available.
    bool all_known = true;
    for (const auto& out_name : node->outputs) {
      if (out_name.empty()) continue;
      if (known_shapes.find(out_name) == known_shapes.end()) {
        all_known = false;
        break;
      }
    }

    if (all_known) {
      for (const auto& out_name : node->outputs) {
        if (out_name.empty()) continue;
        step.output_descs.push_back(known_shapes.at(out_name));
      }
    } else {
      // Run shape inference for this node.
      step.output_descs = InferOutputShapes(*node, known_shapes);
    }

    plan.steps.push_back(std::move(step));
  }

  return std::make_unique<Engine>(std::move(plan), provider_);
}

}  // namespace tie
