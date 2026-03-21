#include <gtest/gtest.h>

#include <string>

#include "../src/loader/mnist_input_loader.h"
#include "../src/loader/onnx_loader.h"
#include "../src/memory/cpu_allocator.h"
#include "../src/providers/cpu_execution_provider.h"
#include "../src/runtime/engine.h"
#include "../src/runtime/engine_builder.h"
#include "../src/runtime/session.h"

namespace tie {

// Bazel runs tests from the workspace root, so paths are relative to it.
static constexpr const char* kModelPath =
    "third_party/inference_engine/models/mnist_ffn.onnx";
static constexpr const char* kInputPath =
    "third_party/inference_engine/inputs/image_0.ubyte";

TEST(E2EMnistTest, ModelLoadsSuccessfully) {
  CpuAllocator alloc;
  ModelData data = OnnxLoader::Load(kModelPath, &alloc);

  EXPECT_FALSE(data.graph.InputNames().empty());
  EXPECT_FALSE(data.graph.OutputNames().empty());
  EXPECT_FALSE(data.weights.empty());
}

TEST(E2EMnistTest, EngineBuildsSuccessfully) {
  CpuAllocator alloc;
  ModelData data = OnnxLoader::Load(kModelPath, &alloc);

  CpuExecutionProvider provider;
  EngineBuilder builder(&provider);
  auto engine = builder.Build(data.graph);

  EXPECT_NE(engine, nullptr);
  EXPECT_FALSE(engine->plan().steps.empty());
}

TEST(E2EMnistTest, InferenceProducesValidOutput) {
  CpuAllocator alloc;
  ModelData model = OnnxLoader::Load(kModelPath, &alloc);

  CpuExecutionProvider provider;
  auto engine = EngineBuilder(&provider).Build(model.graph);
  auto session = engine->CreateSession(&alloc);

  // Bind weights.
  for (auto& [name, weight] : model.weights) {
    session->BindInput(name, std::move(weight));
  }

  // Bind input image.
  const std::string input_name = model.graph.InputNames().at(0);
  session->BindInput(input_name, MnistInputLoader::Load(kInputPath, &alloc));

  session->Run();

  const std::string output_name = model.graph.OutputNames().at(0);
  const Tensor& output = session->GetOutput(output_name);

  // Output should have 10 classes.
  EXPECT_EQ(output.NumElements(), 10);

  // All logits should be finite.
  const float* logits = output.data_as<float>();
  for (int i = 0; i < 10; ++i) {
    EXPECT_TRUE(std::isfinite(logits[i])) << "logit[" << i << "] is not finite";
  }

  // Top-1 prediction should be a valid digit (0-9).
  const int predicted = static_cast<int>(
      std::max_element(logits, logits + 10) - logits);
  EXPECT_GE(predicted, 0);
  EXPECT_LE(predicted, 9);

  std::cout << "[E2E] Prediction for image_0: " << predicted << "\n";
}

}  // namespace tie
