#include <algorithm>
#include <iostream>
#include <string>

#include "study-log/002-inference-engine/src/loader/mnist_input_loader.h"
#include "study-log/002-inference-engine/src/loader/onnx_loader.h"
#include "study-log/002-inference-engine/src/memory/cpu_allocator.h"
#include "study-log/002-inference-engine/src/providers/cpu_execution_provider.h"
#include "study-log/002-inference-engine/src/runtime/engine.h"
#include "study-log/002-inference-engine/src/runtime/engine_builder.h"
#include "study-log/002-inference-engine/src/runtime/session.h"

int main(int argc, char **argv) {
  const std::string model_path =
      argc > 1 ? argv[1]
               : "study-log/002-inference-engine/data/models/mnist_ffn.onnx";
  const std::string input_path =
      argc > 2 ? argv[2]
               : "study-log/002-inference-engine/data/inputs/image_0.ubyte";

  try {
    tie::CpuAllocator allocator;

    // ── Load model
    // ────────────────────────────────────────────────────────────
    std::cout << "Loading model: " << model_path << "\n";
    tie::ModelData model = tie::OnnxLoader::Load(model_path, &allocator);

    // ── Build engine
    // ──────────────────────────────────────────────────────────
    tie::CpuExecutionProvider provider;
    tie::EngineBuilder builder(&provider);
    auto engine = builder.Build(model.graph);

    // ── Create session and bind weights
    // ───────────────────────────────────────
    auto session = engine->CreateSession(&allocator);
    for (auto &[name, weight] : model.weights) {
      session->BindInput(name, std::move(weight));
    }

    // ── Load and bind input image
    // ─────────────────────────────────────────────
    std::cout << "Loading input: " << input_path << "\n";
    const std::string input_name = model.graph.InputNames().at(0);
    tie::Tensor input = tie::MnistInputLoader::Load(input_path, &allocator);
    session->BindInput(input_name, std::move(input));

    // ── Run inference
    // ─────────────────────────────────────────────────────────
    session->Run();

    // ── Print top-1 prediction
    // ────────────────────────────────────────────────
    const std::string output_name = model.graph.OutputNames().at(0);
    const tie::Tensor &output = session->GetOutput(output_name);
    const float *logits = output.data_as<float>();
    const int64_t num_classes = output.NumElements();

    const int predicted = static_cast<int>(
        std::max_element(logits, logits + num_classes) - logits);

    std::cout << "Prediction: " << predicted << "\n";
    std::cout << "Logits:";
    for (int64_t i = 0; i < num_classes; ++i) {
      std::cout << "  [" << i << "]=" << logits[i];
    }
    std::cout << "\n";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
