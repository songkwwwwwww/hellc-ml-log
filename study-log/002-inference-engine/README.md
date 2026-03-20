# 002: Learning Inference Engine

`study-log/002-inference-engine` is a space to analyze `third_party/inference_engine` and newly implement an inference engine for learning purposes within the same monorepo.

The three core goals are:

- Directly implement the `Session + ExecutionProvider` separation in the style of ONNX Runtime.
- Adopt a simplified version of TensorRT's `build phase` and `run phase` separation.
- Start with CPU-first, but design a structure where extending to GPU/other accelerators feels natural.

## Where to start?

The starting point should be a `design doc` rather than implementation. Especially for this topic, deciding "where to draw the line for v1 scope" is much more important than "implementing a few operations".

The recommended reading order is as follows:

1. Quickly grasp the reference engine structure with the `third_party/inference_engine` analysis documents.
2. Confirm the scope, layers, and implementation order with the new learning engine design document.
3. Then, create the Bazel targets and a minimal CPU-only execution path.

## Implementation Roadmap Summary

1. First, complete a CPU-only engine targeting `mnist_ffn.onnx`.
2. Support branched graphs and `Add` with `mnist_ffn_complex.onnx`.
3. Add `Conv` and `MaxPool` with `mnist_conv_ffn.onnx`.
4. Then, attach a memory planner, simple graph optimization, and an accelerator provider.

By following this order, we can naturally increase the learning difficulty while fully reusing the assets (models and input data) from `third_party/inference_engine`.

## Assets to Reuse

- Models:
  - `third_party/inference_engine/models/mnist_ffn.onnx`
  - `third_party/inference_engine/models/mnist_ffn_complex.onnx`
  - `third_party/inference_engine/models/mnist_conv_ffn.onnx`
- Inputs:
  - `third_party/inference_engine/inputs/image_*.ubyte`

## Immediate Tasks for the Next Implementation

- Add Bazel package.
- Write minimal skeletons for `Tensor`, `TensorDesc`, `Node`, `Graph`, `ExecutionProvider`, and `Session`.
- Implement `Flatten`, `Gemm`, and `Relu` CPU kernels.
- Verify end-to-end classification results using 1 sample input image.
