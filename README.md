# hellc-ml-study

A project log and experimental repository for learning and implementing Machine Learning infrastructure and core algorithms.

## Project Structure

This project is structured as a Monorepo using Bazel. Weekly study materials and implementations are organized as sub-projects within the `study-log/` directory.

- `study-log/`: Weekly study projects and experiments (e.g., `001-matmul`, `002-conv`).
- `third_party/`: External dependencies and reference projects (e.g., `inference_engine`).

## Prerequisites

### Build Tools

- [Bazel (Bazelisk recommended)](https://github.com/bazelbuild/bazelisk)
- A compiler supporting C++17 or higher (`clang`, `gcc`, etc.)
- Python 3

### System Libraries

- OpenMP runtime (`libomp`) for OpenMP-based study projects such as
  `study-log/001-matmul`
  - On Apple Silicon macOS, the current repository setup expects Homebrew's
    `libomp` at `/opt/homebrew/opt/libomp`
  - Example:
    ```bash
    brew install libomp
    ```
- BLAS library for reference implementations
  - macOS: Apple Accelerate framework
  - non-macOS: OpenBLAS

### Bazel-managed Dependencies

- `googletest` for correctness tests
- `google_benchmark` for performance benchmarks

These dependencies are declared in `MODULE.bazel` and fetched by Bazel.

## How to Run (e.g., 001-matmul)

### Correctness Tests (Google Test)
Verify that all custom implementations produce the exact same results as the reference implementation (e.g., Apple Accelerate or OpenBLAS).
```bash
bazel test //study-log/001-matmul:matmul_test
```

### Performance Benchmarks (Google Benchmark)
Measure the performance (in GFLOPS) of various implementations and optimization techniques across different matrix sizes.
```bash
bazel run -c opt //study-log/001-matmul:matmul_bench
```

The benchmark numbers currently documented for `study-log/001-matmul` were
collected on an Apple M4 Mac mini. Expect different results on other CPUs,
toolchains, or build settings.

---

## Reference

- [ml-roadmap, Logan Thorneloe, Google, ML Infra](https://github.com/loganthorneloe/ml-roadmap?tab=readme-ov-file)
