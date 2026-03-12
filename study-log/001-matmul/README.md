# 001: Matrix Multiplication (Matmul)

The goal of this project is not to write a competitive BLAS implementation, but to learn about common performance optimizations

This directory contains various implementations of the Matrix Multiplication (GEMM) algorithm, progressing from a naive approach to highly optimized hardware-aware versions. The goal is to study how different low-level software optimization techniques affect computational throughput (GFLOPS).

## Implementations

1. **`matmul_naive.cpp`**: The standard $O(N^3)$ triple-nested loop implementation without any optimizations.
2. **`matmul_loop_reorder.cpp`**: Optimizes memory access patterns (spatial locality) by reordering loops from `row-col-inner` to `row-inner-col`.
3. **`matmul_tiled.cpp`**: Introduces block-tiling to improve Cache hit rates by keeping active sub-matrices within the L1/L2 cache.
4. **`matmul_packed.cpp`**: Packs matrix tiles into continuous memory buffers to minimize TLB (Translation Lookaside Buffer) misses and cache conflicts.
5. **`matmul_simd.cpp`**: Utilizes ARM NEON Intrinsics to compute multiple data points in a single instruction cycle (Vectorization).
6. **`matmul_cache_aware.cpp`**: A sophisticated hierarchical approach applying both macro-tiling (Cache blocking) and micro-tiling (Register blocking) with SIMD.
7. **`matmul_omp.cpp`**: Introduces multi-threading using OpenMP (`#pragma omp parallel for`) and thread pinning to distribute workloads across multiple CPU cores efficiently.
8. **`matmul_reference.cpp`**: Wraps the highly-optimized system BLAS library (e.g., Apple Accelerate Framework or OpenBLAS) for ground-truth correctness and baseline performance comparisons.

## Prerequisites

### Build Tools

- `bazel` or `bazelisk`
- A C++17-compatible compiler (`clang` or `gcc`)

### System Libraries

- OpenMP runtime
  - This project links against `libomp`.
  - On Apple Silicon macOS, the current repository setup expects Homebrew's
    `libomp` at `/opt/homebrew/opt/libomp`.
  - Example:
    ```bash
    brew install libomp
    ```
- BLAS implementation for the reference path
  - macOS: Apple Accelerate framework is used automatically.
  - Non-macOS: OpenBLAS is expected via `-lopenblas`.

### Bazel-managed Dependencies

- `googletest` for correctness tests
- `google_benchmark` for performance benchmarks

These are declared in the repository's `MODULE.bazel`, so they are fetched by
Bazel automatically.

## How to Run

### Run Correctness Tests
Validates that all custom algorithms compute the correct matrix products against the reference implementation.
```bash
bazel test //study-log/001-matmul:matmul_test
```

### Run Performance Benchmarks
Measures the GFLOPS (Giga-Floating Point Operations Per Second) of each method using Google Benchmark.
```bash
bazel run -c opt //study-log/001-matmul:matmul_bench
```

To filter and benchmark a specific algorithm:
```bash
bazel run -c opt //study-log/001-matmul:matmul_bench -- --benchmark_filter="OmpThread"
```

### Benchmark Results

The benchmark results below were collected on an Apple M4 Mac mini.
Performance numbers are hardware-, compiler-, and build-option-dependent, so
results will vary on other machines.

```
Run on (10 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x10)
Load Average: 2.49, 2.51, 2.23
------------------------------------------------------------------------------------------------
Benchmark                                      Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------------------
BenchmarkMatmul/Naive/2048                 21272 ms        21214 ms            1 GFLOPS=809.848M/s
BenchmarkMatmul/NaiveRegisterAcc/2048      16651 ms        16640 ms            1 GFLOPS=1.03247G/s
BenchmarkMatmul/LoopReorder/2048            1059 ms         1058 ms            1 GFLOPS=16.2344G/s
BenchmarkMatmul/Tiled1D/2048                1036 ms         1035 ms            1 GFLOPS=16.5933G/s
BenchmarkMatmul/TiledMD/2048                1162 ms         1162 ms            1 GFLOPS=14.7893G/s
BenchmarkMatmul/SIMD/2048                   1303 ms         1303 ms            1 GFLOPS=13.1883G/s
BenchmarkMatmul/CacheAware/2048             2644 ms         2644 ms            1 GFLOPS=6.49818G/s
BenchmarkMatmul/OmpThread/2048               311 ms          299 ms            3 GFLOPS=57.479G/s
BenchmarkMatmul/Packed/2048                  965 ms          965 ms            1 GFLOPS=17.8089G/s
BenchmarkMatmul/Reference/2048              42.3 ms         42.2 ms           17 GFLOPS=406.871G/s
```

## References

- [Fast Multidimensional Matrix Multiplication on CPU from Scratch, Simon Boehm, 202208](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Optimizing matrix multiplication - Discovering optimizations one at a time, Michal Pitr, 20250216](https://michalpitr.substack.com/p/optimizing-matrix-multiplication)
- [Matrix Multiplication Deep Dive || Cache Blocking, SIMD & Parallelization - Aliaksei Sala - CppCon2025](https://www.youtube.com/watch?v=GHctcSBd6Z4)
- [MIT’s 6.172 on OCW](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/pages/syllabus/)
