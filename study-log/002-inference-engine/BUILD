load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

# ------------------------------------------------------------------------------
# core: IR layer - TensorDesc, Tensor, Node, Graph, OpType
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
# kernels_cpu: Kernel interface, KernelRegistry, CPU kernel implementation
# ------------------------------------------------------------------------------
cc_library(
    name = "kernels_cpu",
    srcs = [
        "src/kernels/cpu/cpu_flatten_kernel.cpp",
        "src/kernels/cpu/cpu_gemm_kernel.cpp",
        "src/kernels/cpu/cpu_relu_kernel.cpp",
        "src/kernels/kernel_registry.cpp",
    ],
    hdrs = [
        "src/kernels/cpu/cpu_flatten_kernel.h",
        "src/kernels/cpu/cpu_gemm_kernel.h",
        "src/kernels/cpu/cpu_relu_kernel.h",
        "src/kernels/kernel.h",
        "src/kernels/kernel_registry.h",
    ],
    deps = [":core"],
)

# ------------------------------------------------------------------------------
# runtime: TensorStore, ExecutionPlan, EngineBuilder, Engine, Session
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
# loader: OnnxLoader (M1), MnistInputLoader (M1)
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
    deps = [":core"],
)

# ------------------------------------------------------------------------------
# mnist_runner: CLI binary (M1)
# ------------------------------------------------------------------------------
cc_binary(
    name = "mnist_runner",
    srcs = ["tools/mnist_runner_main.cpp"],
    deps = [
        ":loader",
        ":runtime",
    ],
)

# ------------------------------------------------------------------------------
# inference_engine_test: Unit tests
# ------------------------------------------------------------------------------
cc_test(
    name = "inference_engine_test",
    srcs = [
        "tests/graph_test.cpp",
        "tests/tensor_test.cpp",
    ],
    deps = [
        ":core",
        ":memory",
        "@googletest//:gtest_main",
    ],
)
