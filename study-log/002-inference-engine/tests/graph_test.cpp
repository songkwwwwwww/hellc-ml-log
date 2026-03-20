#include <gtest/gtest.h>

#include "../src/core/graph.h"
#include "../src/core/node.h"

namespace tie {

// Linear chain: A -> B -> C topological sort verification
TEST(GraphTest, TopologicalSort_LinearChain) {
  Graph g;

  auto n1 = std::make_unique<Node>();
  n1->name = "Flatten";
  n1->op_type = OpType::kFlatten;
  n1->inputs = {"input"};
  n1->outputs = {"flat"};

  auto n2 = std::make_unique<Node>();
  n2->name = "Gemm";
  n2->op_type = OpType::kGemm;
  n2->inputs = {"flat", "W"};
  n2->outputs = {"gemm_out"};

  auto n3 = std::make_unique<Node>();
  n3->name = "Relu";
  n3->op_type = OpType::kRelu;
  n3->inputs = {"gemm_out"};
  n3->outputs = {"output"};

  // Intentionally insert in reverse order - should output correct order after sorting.
  g.AddNode(std::move(n3));
  g.AddNode(std::move(n2));
  g.AddNode(std::move(n1));

  auto sorted = g.TopologicallySortedNodes();
  ASSERT_EQ(sorted.size(), 3u);
  EXPECT_EQ(sorted[0]->name, "Flatten");
  EXPECT_EQ(sorted[1]->name, "Gemm");
  EXPECT_EQ(sorted[2]->name, "Relu");
}

}  // namespace tie
