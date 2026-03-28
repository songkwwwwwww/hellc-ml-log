load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

# ------------------------------------------------------------------------------
# onnx_proto: ONNX protobuf schema used by OnnxLoader (v1)
# ------------------------------------------------------------------------------
proto_library(
    name = "onnx_proto",
    srcs = ["src/loader/onnx-ml.proto"],
)

cc_proto_library(
    name = "onnx_cc_proto",
    deps = [":onnx_proto"],
)

# ------------------------------------------------------------------------------
# memory: Allocator abstract interface + CpuAllocator
# ------------------------------------------------------------------------------
cc_library(
    name = "memory",
    srcs = ["src/memory/cpu_allocator.cpp"],
    hdrs = [
        "src/memory/allocator.h",
        "src/memory/cpu_allocator.h",
    ],
)

# ------------------------------------------------------------------------------
# core: IR layer - TensorDesc, Tensor, TensorStore, Node, Graph, OpType
# ------------------------------------------------------------------------------
cc_library(
    name = "core",
    srcs = [
        "src/core/graph.cpp",
        "src/core/tensor.cpp",
        "src/core/tensor_desc.cpp",
        "src/core/tensor_store.cpp",
    ],
    hdrs = [
        "src/core/graph.h",
        "src/core/node.h",
        "src/core/optype.h",
        "src/core/tensor.h",
        "src/core/tensor_desc.h",
        "src/core/tensor_store.h",
    ],
    deps = [":memory"],
)

# ------------------------------------------------------------------------------
# kernels_cpu: Kernel interface, KernelRegistry, CPU kernel implementations
# ------------------------------------------------------------------------------
cc_library(
    name = "kernels_cpu",
    srcs = [
        "src/kernels/cpu/cpu_add_kernel.cpp",
        "src/kernels/cpu/cpu_flatten_kernel.cpp",
        "src/kernels/cpu/cpu_gemm_kernel.cpp",
        "src/kernels/cpu/cpu_relu_kernel.cpp",
        "src/kernels/kernel_registry.cpp",
    ],
    hdrs = [
        "src/kernels/cpu/cpu_add_kernel.h",
        "src/kernels/cpu/cpu_flatten_kernel.h",
        "src/kernels/cpu/cpu_gemm_kernel.h",
        "src/kernels/cpu/cpu_relu_kernel.h",
        "src/kernels/kernel.h",
        "src/kernels/kernel_registry.h",
    ],
    deps = [":core"],
)

# ------------------------------------------------------------------------------
# providers_cpu: CpuExecutionProvider
# ------------------------------------------------------------------------------
cc_library(
    name = "providers_cpu",
    srcs = ["src/providers/cpu_execution_provider.cpp"],
    hdrs = [
        "src/providers/cpu_execution_provider.h",
        "src/providers/execution_provider.h",
    ],
    deps = [
        ":core",
        ":kernels_cpu",
        ":memory",
    ],
)

# ------------------------------------------------------------------------------
# runtime: EngineBuilder, Engine, Session, ExecutionPlan
# ------------------------------------------------------------------------------
cc_library(
    name = "runtime",
    srcs = [
        "src/runtime/engine.cpp",
        "src/runtime/engine_builder.cpp",
        "src/runtime/session.cpp",
    ],
    hdrs = [
        "src/runtime/engine.h",
        "src/runtime/engine_builder.h",
        "src/runtime/execution_plan.h",
        "src/runtime/session.h",
    ],
    deps = [
        ":core",
        ":providers_cpu",
    ],
)

# ------------------------------------------------------------------------------
# loader: OnnxLoader (protobuf-based), MnistInputLoader
# ------------------------------------------------------------------------------
cc_library(
    name = "loader",
    srcs = [
        "src/loader/mnist_input_loader.cpp",
        "src/loader/onnx_loader.cpp",
    ],
    hdrs = [
        "src/loader/mnist_input_loader.h",
        "src/loader/onnx_loader.h",
    ],
    deps = [
        ":core",
        ":onnx_cc_proto",
    ],
)

# ------------------------------------------------------------------------------
# mnist_runner: End-to-end CLI binary (M1)
# ------------------------------------------------------------------------------
cc_binary(
    name = "mnist_runner",
    srcs = ["tools/mnist_runner_main.cpp"],
    data = [
        "//third_party/inference_engine/inputs:test_images",
        "//third_party/inference_engine/models:mnist_models",
    ],
    deps = [
        ":loader",
        ":runtime",
    ],
)

# ------------------------------------------------------------------------------
# inference_engine_test: Unit + integration tests
# ------------------------------------------------------------------------------
cc_test(
    name = "inference_engine_test",
    srcs = [
        "tests/e2e_mnist_test.cpp",
        "tests/graph_test.cpp",
        "tests/kernel_test.cpp",
        "tests/tensor_test.cpp",
    ],
    data = [
        "//third_party/inference_engine/inputs:test_images",
        "//third_party/inference_engine/models:mnist_models",
    ],
    deps = [
        ":core",
        ":loader",
        ":memory",
        ":runtime",
        "@googletest//:gtest_main",
    ],
)
