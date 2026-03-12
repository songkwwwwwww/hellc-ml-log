# hellc-ml-study

A project log and experimental repository for learning and implementing Machine Learning infrastructure and core algorithms.

## Project Structure

This project is structured as a Monorepo using Bazel. Weekly study materials and implementations are organized as sub-projects within the `study-log/` directory.

- `study-log/`: Weekly study projects and experiments (e.g., `001-matmul`, `002-conv`).
- `third_party/`: External dependencies and reference projects (e.g., `inference_engine`).

## Prerequisites

- [Bazel (Bazelisk recommended)](https://github.com/bazelbuild/bazelisk)
- A compiler supporting C++17 or higher (gcc, clang, etc.)
- Python 3

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

---

## Reference

- [ml-roadmap, Logan Thorneloe, Google, ML Infra](https://github.com/loganthorneloe/ml-roadmap?tab=readme-ov-file)