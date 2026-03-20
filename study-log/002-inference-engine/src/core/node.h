#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "optype.h"

namespace tie {

// Node attribute value: one of float, int64, vector<int64>.
using AttributeValue = std::variant<float, int64_t, std::vector<int64_t>>;
using AttributeMap = std::unordered_map<std::string, AttributeValue>;

// A single operation node in the graph.
// Inputs and outputs are connected by tensor names (string).
struct Node {
  std::string name;
  OpType op_type = OpType::kUnknown;
  std::vector<std::string> inputs;   // Tensor names
  std::vector<std::string> outputs;  // Tensor names
  AttributeMap attributes;
};

}  // namespace tie
