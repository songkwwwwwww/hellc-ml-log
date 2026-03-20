# Tiny Inference Engine — Design Document

## Table of Contents

1. [Overview](#1-overview)
2. [Goals and Non-Goals](#2-goals-and-non-goals)
3. [Background](#3-background)
4. [System Architecture](#4-system-architecture)
5. [Component Design](#5-component-design)
6. [Execution Flow](#6-execution-flow)
7. [ONNX Parsing Strategy](#7-onnx-parsing-strategy)
8. [Directory Structure and Bazel Targets](#8-directory-structure-and-bazel-targets)
9. [Implementation Milestones](#9-implementation-milestones)
10. [Open Questions](#10-open-questions)

---

## 1. Overview

This project is a learning-focused tiny inference engine built by directly
implementing core design patterns from [C++ Inference Engine from scratch, MichalPitr](https://github.com/MichalPitr/inference_engine), ONNX
Runtime, and TensorRT.

The goal is not to build a production-ready engine, but to develop a hands-on
understanding of the following questions:

- Why do inference engines separate the build phase from the run phase?
- What problem does the `ExecutionProvider` abstraction solve?
- Why is `KernelRegistry` needed at build time rather than run time?
- Why separate `TensorDesc` from `Tensor`?
- What does a memory planner actually optimize?

The v1 scope is end-to-end inference of `mnist_ffn.onnx` on CPU.

---

## 2. Goals and Non-Goals

### Goals

- Implement a self-contained inference engine that builds within the Bazel
  monorepo
- Adopt TensorRT-style build / run phase separation
- Adopt ONNX Runtime-style Session + ExecutionProvider separation
- Design a CPU-first architecture that naturally extends to accelerators
- Parse ONNX binary directly without protobuf (advanced milestone)
- Reuse MNIST models and input data from [C++ Inference Engine from scratch, MichalPitr](https://github.com/MichalPitr/inference_engine)

### Non-Goals

- Full ONNX operator coverage
- Dynamic shape support
- Batch size > 1 (for v1)
- Quantization, pruning, autotuning
- Production-grade error handling and scheduling
- CUDA kernel implementation (provider interface design only)

---

## 3. Background

### 3.1 Key Concepts from ONNX Runtime

ONNX Runtime is known for two core design principles.

**Session-centric API**: Model loading, initialization, and execution are all
managed through a single `InferenceSession` object, encapsulating state and
thread safety.

**ExecutionProvider abstraction**: Diverse backends (CPU, CUDA, TensorRT,
CoreML) are abstracted behind a common interface. Different nodes in the same
graph can be dispatched to different providers — this is called partial
offloading.

```
InferenceSession
  ├── Graph (ONNX IR)
  ├── ExecutionProvider[] (CPU, CUDA, ...)
  └── SessionState (optimized execution plan)
```

### 3.2 Key Concepts from TensorRT

TensorRT draws a hard line between the **build phase** and the **runtime
phase**.

**Build phase** (`IBuilder` → `ICudaEngine`):
- Parse network definition
- Layer fusion and precision calibration
- Kernel tactic selection (device-specific optimization)
- Produce a serializable `Engine` object

**Runtime phase** (`IRuntime` → `IExecutionContext`):
- Load the pre-built engine
- Bind inputs and run inference
- Reuse memory across calls

The key insight is that build cost is paid once; runtime is always fast.

### 3.3 What This Project Borrows

| Concept | Source | Applied As |
|---------|--------|------------|
| Session API | ONNX Runtime | `Session::Run()` |
| ExecutionProvider | ONNX Runtime | `ExecutionProvider` interface |
| KernelRegistry | ONNX Runtime | Kernel selection at build time |
| Build/Run separation | TensorRT | `EngineBuilder::Build()` → `Engine` |
| ExecutionPlan | TensorRT | Pre-compiled ordered step list |
| TensorDesc split | Common pattern | Separate metadata from storage |

---

## 4. System Architecture

### 4.1 Layer Diagram

```
+----------------------------------------------------------+
|                      Application                         |
|                   (mnist_runner_main)                    |
+----------------------------------------------------------+
          |
          v
+----------------------------------------------------------+
|                       Runtime                            |
|   EngineBuilder  Session  ExecutionPlan  TensorStore     |
+----------------------------------------------------------+
          |                        |
          v                        v
+--------------------+   +----------------------+
|        IR          |   |       Providers       |
| Model  Graph  Node |   | ExecutionProvider    |
| TensorDesc  OpType |   | CpuExecutionProvider |
+--------------------+   +----------------------+
                                   |
                                   v
                         +---------------------+
                         |       Kernels        |
                         | KernelRegistry       |
                         | cpu/gemm  cpu/relu   |
                         | cpu/flatten          |
                         +---------------------+
          |
          v
+----------------------------------------------------------+
|                       Memory                             |
|        Allocator   CpuAllocator   ArenaPlanner           |
+----------------------------------------------------------+
          |
          v
+----------------------------------------------------------+
|                        Loader                            |
|           OnnxLoader        MnistInputLoader             |
+----------------------------------------------------------+
```

### 4.2 Layer Responsibilities

| Layer | Responsibility |
|-------|----------------|
| IR | Static model representation (graph, nodes, tensor metadata) |
| Runtime | Orchestrate build/run phases, manage session state |
| Providers | Device strategy (memory policy, kernel dispatch) |
| Kernels | Individual operator implementations (Gemm, Relu, Flatten, ...) |
| Memory | Buffer allocation and arena-based reuse |
| Loader | ONNX binary parsing, input data loading |

---

## 5. Component Design

### 5.1 TensorDesc and Tensor

**Design principle**: separate descriptor (metadata) from storage (buffer).

Why this separation matters:
- Shape inference can run on descriptors alone, with no actual data
- Memory planning can compute buffer sizes at build time from descriptors
- The same descriptor can refer to either a CPU or GPU buffer

```
TensorDesc  (metadata only, no storage)
  ├── name:  string
  ├── shape: vector<int64_t>
  └── dtype: DataType  (kFloat32, kInt64, ...)

Tensor  (owns the buffer)
  ├── desc:      TensorDesc
  ├── data:      void*       (managed by Allocator)
  └── allocator: Allocator*
```

**Difference from the reference**: [C++ Inference Engine from scratch, MichalPitr](https://github.com/MichalPitr/inference_engine) uses a
templated `Tensor<T>` that merges descriptor and storage into one class. This
project separates them to make shape inference and memory planning concrete
learning exercises.

### 5.2 Node and Graph

**Node**: a single operation unit.

```
Node
  ├── op_type:    OpType          (Gemm, Relu, Flatten, ...)
  ├── name:       string          (for debugging)
  ├── inputs:     vector<string>  (tensor name references)
  ├── outputs:    vector<string>
  └── attributes: AttributeMap   (e.g., alpha, beta, transB for Gemm)
```

**Graph**: a DAG (Directed Acyclic Graph) of nodes.

```
Graph
  ├── nodes:    vector<unique_ptr<Node>>
  ├── inputs:   vector<string>              (model input tensor names)
  ├── outputs:  vector<string>              (model output tensor names)
  ├── tensors:  map<string, TensorDesc>     (metadata for all tensors)
  └── TopologicallySortedNodes() -> vector<Node*>
```

**Topological sort (Kahn's algorithm)**:

```
function kahn_sort(graph):
  in_degree = {node: 0 for node in graph.nodes}
  for node in graph.nodes:
    for output in node.outputs:
      for consumer in graph.consumers(output):
        in_degree[consumer] += 1

  queue = [node for node in graph.nodes if in_degree[node] == 0]
  result = []
  while queue:
    node = queue.pop()
    result.append(node)
    for output in node.outputs:
      for consumer in graph.consumers(output):
        in_degree[consumer] -= 1
        if in_degree[consumer] == 0:
          queue.append(consumer)
  return result
```

### 5.3 ExecutionProvider

The provider is the **device strategy layer**. It does not contain operator
implementations directly.

```cpp
class ExecutionProvider {
 public:
  virtual ~ExecutionProvider() = default;
  virtual std::string Name() const = 0;

  // Build phase: check whether this provider can handle a given node
  virtual bool CanHandle(const Node& node) const = 0;

  // Build phase: transfer constant weight tensors to device memory
  virtual void PrepareWeights(
      std::unordered_map<std::string, Tensor>& weights) = 0;

  // Run phase: execute a single node, writing outputs into TensorStore
  virtual void Execute(const Node& node, TensorStore& store) = 0;

  // Return this provider's memory allocator
  virtual Allocator* GetAllocator() = 0;
};
```

`CpuExecutionProvider` implements this interface and owns a `KernelRegistry`
internally.

**Difference from the reference**: [C++ Inference Engine from scratch, MichalPitr](https://github.com/MichalPitr/inference_engine)'s
`ExecutionProvider::evaluateNode()` returns a `Tensor<float>` by value. This
project has `Execute()` write directly into `TensorStore` to avoid intermediate
copies.

### 5.4 EngineBuilder and Session

**Build phase** — handled by `EngineBuilder`:

```
EngineBuilder::Build(graph) -> Engine

Steps:
  1. IR construction: Graph (Node, TensorDesc) is already available
  2. Shape inference: propagate output shapes through each node
  3. Kernel selection: KernelRegistry lookup for each node
                       (unsupported ops fail fast at build time)
  4. Memory planning: compute intermediate buffer lifetimes
  5. Produce ExecutionPlan: ordered PlanSteps with kernels assigned
```

**Run phase** — handled by `Session`:

```
Session::Run()

Steps:
  1. Input binding: register input tensors by name into TensorStore
  2. Execute each PlanStep in order
  3. Return outputs by name
```

```
EngineBuilder
  └── Build(graph) -> Engine

Engine
  ├── execution_plan: ExecutionPlan   (pre-compiled step list)
  └── CreateSession() -> Session

Session
  ├── BindInput(name, tensor)
  ├── Run()
  └── GetOutput(name) -> Tensor
```

### 5.5 ExecutionPlan

`ExecutionPlan` is the output of the build phase. The run phase follows it
directly without re-interpreting the graph.

```
ExecutionPlan
  └── steps: vector<PlanStep>

PlanStep
  ├── node:           Node*                  (the node to execute)
  ├── provider:       ExecutionProvider*     (the responsible provider)
  ├── kernel:         Kernel*                (the selected kernel)
  └── memory_events:  vector<MemEvent>       (buffers to free after this step)
```

### 5.6 TensorStore

Stores all tensors by name during execution: inputs, weights, intermediates,
and outputs.

```
TensorStore
  ├── Get(name)        -> const Tensor*
  ├── GetMutable(name) -> Tensor*
  ├── Set(name, tensor)
  └── tensors: unordered_map<string, Tensor>
```

`TensorStore` lives in the `core` layer (not `runtime`) to avoid a circular
dependency between `providers` and `runtime`.

### 5.7 KernelRegistry

Looks up kernels by `OpType + DeviceType + DataType` combination.

```
KernelRegistry
  └── Lookup(op_type, device, dtype) -> unique_ptr<Kernel>

Kernel  (abstract)
  └── Compute(inputs, outputs, attributes) = 0

CpuGemmKernel    : Kernel
CpuReluKernel    : Kernel
CpuFlattenKernel : Kernel
```

If `Lookup()` fails at build time, it throws immediately. This is the same
reason TensorRT fails during engine build rather than during inference — errors
are surfaced as early as possible.

### 5.8 Memory Allocator

```
Allocator  (abstract)
  ├── Allocate(bytes)  -> void*
  └── Deallocate(ptr)

CpuAllocator : Allocator
  └── malloc / free

ArenaAllocator : Allocator  (Milestone 4)
  └── pre-allocate a large block, distribute via bump pointer
```

---

## 6. Execution Flow

### 6.1 Build Phase

```
[User]
    |
    | OnnxLoader::Load("mnist_ffn.onnx")
    v
[Graph + weights]   <-- parsed via protobuf (v1) or direct wire format (M6)
    |
    | EngineBuilder::Build(graph)
    v
  [1] Shape Inference
      Propagate output TensorDesc shapes through each node
      e.g.: Flatten(input=[1,784])       -> output=[1,784]
            Gemm(input=[1,784], W=[784,128]) -> output=[1,128]
    |
  [2] Kernel Selection
      KernelRegistry.Lookup(Gemm, CPU, float32) -> CpuGemmKernel
      KernelRegistry.Lookup(Relu, CPU, float32) -> CpuReluKernel
      ...
      (unsupported op => throw immediately)
    |
  [3] Memory Planning   (v1: naive allocation; M4: arena-based)
      Compute sizes of all intermediate tensors
      Analyze tensor lifetimes (created/consumed at which step)
    |
  [4] Build ExecutionPlan
      Topological order -> list of PlanSteps
    |
    v
[Engine]   <-- build complete, ready for run phase
```

### 6.2 Run Phase

```
[User]
    |
    | session.BindInput("input", tensor)
    v
[TensorStore]   <-- input tensor registered
    |
    | session.Run()
    v
  for each PlanStep in ExecutionPlan:
    |
    | step.provider->Execute(node, tensor_store)
    |   └── kernel->Compute(input_tensors, output_tensors, attrs)
    |         |
    |         └── writes result into TensorStore
    |
    | (process MemEvents: free tensors no longer needed)
    |
    v
[User]
    | session.GetOutput("output")
    v
[Tensor]   <-- final output
```

---

## 7. ONNX Parsing Strategy

### 7.1 Two-Phase Approach

ONNX parsing is split into two phases across milestones.

**v1 (initial): protobuf library**

Same approach as [C++ Inference Engine from scratch, MichalPitr](https://github.com/MichalPitr/inference_engine). Use `onnx::ModelProto`
directly. Focus on learning the engine's core structure (Session, Provider,
Plan) rather than the serialization format.

```cpp
// OnnxLoader v1 — protobuf-based
onnx::ModelProto model;
std::ifstream f(path, std::ios::binary);
model.ParseFromIstream(&f);  // protobuf handles full deserialization

// Use the generated C++ objects directly
for (const auto& node : model.graph().node()) { ... }
for (const auto& init : model.graph().initializer()) { ... }
```

Bazel integration is a single line — protobuf is available in the Bazel
Central Registry (BCR):

```python
bazel_dep(name = "protobuf", version = "29.3")
```

**v2 (advanced, M6): parse protobuf wire format directly**

Remove the protobuf dependency by implementing a minimal ONNX loader that
reads the binary format by hand. Undertaken as a separate milestone after the
engine core is complete.

### 7.2 Wire Format Reference (for M6)

An ONNX file is a `ModelProto` serialized in protobuf binary format.

```
Each field = (field_number << 3 | wire_type) + payload

wire_type:
  0 = Varint        (int32, int64, bool, enum)
  2 = Length-delim  (string, bytes, embedded message, repeated)
  5 = 32-bit        (float)

Only the fields needed for inference are parsed (no full schema required):

ModelProto (field 7: graph)
  └── GraphProto
        ├── node (field 1): NodeProto[]
        │     ├── input / output (field 1/2): string[]
        │     ├── op_type (field 4): string
        │     └── attribute (field 4): AttributeProto[]
        ├── initializer (field 5): TensorProto[]   <- weights
        │     ├── dims (field 1): int64[]
        │     ├── name (field 8): string
        │     └── raw_data (field 9): bytes
        └── input / output (field 11/12): ValueInfoProto[]
```

---

## 8. Directory Structure and Bazel Targets

### 8.1 Directory Structure

```
study-log/002-inference-engine/
├── BUILD
├── README.md
├── docs/
│   └── design.md                    (this document)
├── src/
│   ├── core/
│   │   ├── tensor_desc.h / .cpp     TensorDesc (metadata only)
│   │   ├── tensor.h / .cpp          Tensor (owns buffer)
│   │   ├── tensor_store.h / .cpp    TensorStore (name-keyed tensor map)
│   │   ├── node.h                   Node, OpType, AttributeMap
│   │   ├── graph.h / .cpp           Graph, topological sort
│   │   └── optype.h                 OpType enum
│   ├── runtime/
│   │   ├── engine_builder.h / .cpp  Build phase
│   │   ├── engine.h / .cpp          Engine (holds ExecutionPlan)
│   │   ├── session.h / .cpp         Run phase
│   │   └── execution_plan.h         PlanStep, ExecutionPlan
│   ├── providers/
│   │   ├── execution_provider.h     Abstract provider interface
│   │   └── cpu_execution_provider.h / .cpp
│   ├── kernels/
│   │   ├── kernel.h                 Abstract kernel interface
│   │   ├── kernel_registry.h / .cpp
│   │   └── cpu/
│   │       ├── cpu_flatten_kernel.h / .cpp
│   │       ├── cpu_gemm_kernel.h / .cpp
│   │       └── cpu_relu_kernel.h / .cpp
│   ├── memory/
│   │   ├── allocator.h              Abstract allocator interface
│   │   └── cpu_allocator.h / .cpp
│   └── loader/
│       ├── onnx_loader.h / .cpp     Protobuf-based ONNX loader (v1)
│       └── mnist_input_loader.h / .cpp   .ubyte input file loader
├── tools/
│   └── mnist_runner_main.cpp        CLI: ONNX model + input -> prediction
└── tests/
    ├── tensor_test.cpp
    ├── graph_test.cpp
    ├── kernel_test.cpp
    └── e2e_mnist_test.cpp
```

### 8.2 Bazel Targets

```python
# study-log/002-inference-engine/BUILD

cc_library(name = "memory",       ...)   # Allocator, CpuAllocator
cc_library(name = "core",         ...)   # TensorDesc, Tensor, TensorStore,
                                         # Node, Graph, OpType
                                         # deps: [":memory"]
cc_library(name = "kernels_cpu",  ...)   # Kernel, KernelRegistry, cpu/ kernels
                                         # deps: [":core"]
cc_library(name = "providers_cpu",...)   # ExecutionProvider, CpuExecutionProvider
                                         # deps: [":core", ":kernels_cpu"]
cc_library(name = "runtime",      ...)   # EngineBuilder, Engine, Session,
                                         # ExecutionPlan
                                         # deps: [":core", ":providers_cpu"]
cc_library(name = "loader",       ...)   # OnnxLoader, MnistInputLoader
                                         # deps: [":core"]

cc_binary(name = "mnist_runner",  ...)   # deps: [":runtime", ":loader"]
cc_test(name = "inference_engine_test",...) # deps: [":core", ":runtime",
                                            #        ":loader", "@googletest//:gtest_main"]
```

**Dependency graph** (no cycles):

```
memory
  └── core  (TensorStore lives here to avoid provider <-> runtime cycle)
        ├── kernels_cpu
        │     └── providers_cpu
        │           └── runtime
        │                 └── mnist_runner
        └── loader
              └── mnist_runner
```

---

## 9. Implementation Milestones

```
Phase 1: Core engine (protobuf-based loader)
  M0 -> M1 -> M2 -> M3 -> M4 -> M5

Phase 2: Advanced
  M6  (protobuf-free ONNX loader)
```

---

### Milestone 0: Bazel skeleton ✅

Goal: establish the project skeleton; build passes with stub headers.

Work:
- Add protobuf dependency to `MODULE.bazel`
- Create `study-log/002-inference-engine/BUILD`
- Write stub headers for all layers (empty class declarations)
- Write initial unit tests for `TensorDesc`, `Tensor`, `Graph`

Done when:
- `bazel build //study-log/002-inference-engine:...` passes
- `bazel test  //study-log/002-inference-engine:inference_engine_test` passes (5/5)

---

### Milestone 1: CPU-only static runner

Goal: end-to-end inference of `mnist_ffn.onnx`. Core engine path complete.

Target model: `mnist_ffn.onnx` (Flatten → Gemm → Relu → Gemm → Relu)
Constraints: FP32, single batch, static shape, protobuf-based loader

Implementation:
- `OnnxLoader` (protobuf): `.onnx` → `Graph` + weight map
- `MnistInputLoader`: `.ubyte` → `Tensor`
- `CpuFlattenKernel`, `CpuGemmKernel`, `CpuReluKernel`
- `CpuExecutionProvider` + `KernelRegistry`
- `EngineBuilder::Build()`, `Session::Run()`

Done when:
- `image_0.ubyte` → correct MNIST digit printed
- `tensor_test`, `kernel_test`, `e2e_mnist_test` pass

---

### Milestone 2: General DAG support

Goal: execute branching graphs beyond linear chains; handle fan-in/fan-out.

Target model: `mnist_ffn_complex.onnx` (branch merged via `Add`)
New operator: `Add`

Implementation:
- `CpuAddKernel`
- Verify tensor lifetime tracking across branch/merge points

Done when:
- `mnist_ffn_complex.onnx` end-to-end inference succeeds

---

### Milestone 3: CNN support

Goal: handle spatial tensors; understand convolution memory access patterns.

Target model: `mnist_conv_ffn.onnx`
New operators: `Conv2D`, `MaxPool`

Implementation:
- `CpuConv2dKernel`, `CpuMaxPoolKernel`
- NCHW tensor layout handling

Done when:
- `mnist_conv_ffn.onnx` end-to-end inference succeeds

---

### Milestone 4: Memory Planner

Goal: analyze intermediate tensor lifetimes and reuse buffers. Deepen
understanding of the build phase.

Implementation:
- `ArenaAllocator`: pre-allocate a large block, distribute via bump pointer
- `MemoryPlanner`: lifetime-based buffer reuse plan
- Integrate `MemEvent` (alloc/free points) into `ExecutionPlan`

Done when:
- Peak memory reduction vs. naive allocation measured and documented

---

### Milestone 5: Accelerator Provider Interface

Goal: complete the provider abstraction; verify that swapping providers at
runtime works with the same graph.

Implementation:
- Refine `ExecutionProvider` interface to cover cross-device tensor movement
- Device-aware tensor management in `TensorStore`
- `MockAcceleratorProvider`: falls back to CPU but validates the swap path
- (optional) One real CUDA kernel: `CudaGemmKernel`

Done when:
- Same graph runs successfully with both `CpuProvider` and
  `MockAcceleratorProvider`

---

### Milestone 6: Protobuf-free ONNX Loader (advanced)

Goal: remove the protobuf dependency; understand the ONNX wire format by
parsing it directly.

Implementation:
- `OnnxSubsetLoader` (protobuf-free): parse varint and length-delimited fields
  by hand
- Keep the same external interface as the protobuf-based `OnnxLoader`
- Cross-check test: compare outputs of both implementations

Done when:
- Inference results match the protobuf-based loader
- `protobuf` dependency can be removed from `MODULE.bazel`

---

## 10. Open Questions

| # | Question | Milestone | Priority |
|---|----------|-----------|----------|
| 1 | Should shape inference be a separate pass, or handled inside the loader? | M1 | High |
| 2 | Should weights and intermediate tensors share the same `TensorStore` map, or be kept separate? | M1 | Medium |
| 3 | How to implement `AttributeMap` type erasure: `std::variant` vs `std::any` vs custom? | M1 | Medium |
| 4 | Who owns cross-device tensor movement: `TensorStore` or the provider? | M5 | Medium |
| 5 | What fragmentation strategy should `ArenaAllocator` use? | M4 | Low |
| 6 | `mnist_runner` CLI interface: flags for model path, input path, top-k | M1 | Low |
