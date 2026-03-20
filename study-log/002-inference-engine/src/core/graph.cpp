#include "graph.h"

#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace tie {

void Graph::AddNode(std::unique_ptr<Node> node) {
  nodes_.push_back(std::move(node));
}

void Graph::SetTensorDesc(const std::string& name, TensorDesc desc) {
  tensors_[name] = std::move(desc);
}

void Graph::SetInputNames(std::vector<std::string> names) {
  input_names_ = std::move(names);
}

void Graph::SetOutputNames(std::vector<std::string> names) {
  output_names_ = std::move(names);
}

const TensorDesc* Graph::GetTensorDesc(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) return nullptr;
  return &it->second;
}

std::vector<const Node*> Graph::TopologicallySortedNodes() const {
  // Kahn's algorithm
  // Tensor name -> nodes that take this tensor as input
  std::unordered_map<std::string, std::vector<const Node*>> tensor_consumers;
  std::unordered_map<const Node*, int> in_degree;

  for (const auto& node : nodes_) {
    in_degree[node.get()] = 0;
  }

  for (const auto& node : nodes_) {
    for (const auto& output : node->outputs) {
      tensor_consumers[output];  // ensure key exists
    }
    for (const auto& input : node->inputs) {
      tensor_consumers[input].push_back(node.get());
    }
  }

  // Weights/graph-inputs are already prepared, so do not increment the
  // in_degree of nodes that consume them. Only count the node output -> consumer relationship.
  for (const auto& node : nodes_) {
    for (const auto& input : node->inputs) {
      // Check if there is any node that outputs this input
      for (const auto& producer : nodes_) {
        for (const auto& output : producer->outputs) {
          if (output == input) {
            in_degree[node.get()]++;
          }
        }
      }
    }
  }

  std::queue<const Node*> q;
  for (const auto& [node, deg] : in_degree) {
    if (deg == 0) q.push(node);
  }

  std::vector<const Node*> result;
  while (!q.empty()) {
    const Node* cur = q.front();
    q.pop();
    result.push_back(cur);

    for (const auto& output : cur->outputs) {
      for (const Node* consumer : tensor_consumers[output]) {
        if (--in_degree[consumer] == 0) {
          q.push(consumer);
        }
      }
    }
  }

  if (result.size() != nodes_.size()) {
    throw std::runtime_error("Graph has a cycle");
  }
  return result;
}

}  // namespace tie
