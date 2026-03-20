#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "node.h"
#include "tensor_desc.h"

namespace tie {

// Holds the Directed Acyclic Graph (DAG) of nodes and metadata of all tensors.
class Graph {
 public:
  Graph() = default;

  void AddNode(std::unique_ptr<Node> node);
  void SetTensorDesc(const std::string& name, TensorDesc desc);
  void SetInputNames(std::vector<std::string> names);
  void SetOutputNames(std::vector<std::string> names);

  // Returns a list of node pointers topologically sorted using Kahn's algorithm.
  std::vector<const Node*> TopologicallySortedNodes() const;

  const std::vector<std::string>& InputNames() const { return input_names_; }
  const std::vector<std::string>& OutputNames() const { return output_names_; }
  const TensorDesc* GetTensorDesc(const std::string& name) const;
  const std::vector<std::unique_ptr<Node>>& Nodes() const { return nodes_; }

 private:
  std::vector<std::unique_ptr<Node>> nodes_;
  std::unordered_map<std::string, TensorDesc> tensors_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

}  // namespace tie
