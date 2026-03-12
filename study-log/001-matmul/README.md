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
7. **`matmul_omp.cpp` / `OmpThread`**: A cache-tiled OpenMP baseline that parallelizes independent `C` tiles across CPU cores.
8. **`matmul_omp.cpp` / `OmpThreadSimd`**: Adds an explicit SIMD micro-kernel inside each OpenMP tile so every worker also vectorizes its inner column updates.
9. **`matmul_omp.cpp` / `OmpThreadPacked`**: Packs per-thread `A`/`B` tiles into contiguous scratch buffers before multiplying them to reduce strided memory traffic.
10. **`matmul_omp.cpp` / `OmpThreadPackedSimd`**: Combines OpenMP tiling, per-thread packing, and SIMD in the micro-kernel to study the stacked effect of all three optimizations.
11. **`matmul_omp.cpp` / `OmpThreadPackedRegister`**: Uses a packed `4x8` register-blocked micro-kernel so each thread accumulates a small `C` tile in NEON registers before writing it back.
12. **`matmul_reference.cpp`**: Wraps the highly-optimized system BLAS library (e.g., Apple Accelerate Framework or OpenBLAS) for ground-truth correctness and baseline performance comparisons.

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

To compare the entire OpenMP variant family:
```bash
bazel run -c opt //study-log/001-matmul:matmul_bench -- --benchmark_filter="OmpThread.*"
```

### Benchmark Results

The benchmark results below were collected on an Apple M4 Mac mini.
Performance numbers are hardware-, compiler-, and build-option-dependent, so
results will vary on other machines. The sample output below predates the newer
OpenMP variants, so rerun the filtered benchmark above to compare them.

```
Run on (10 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x10)
Load Average: 3.62, 3.40, 3.01
-------------------------------------------------------------------------------------------------------
Benchmark                                             Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------------------
BenchmarkMatmul/Naive/2048                        22582 ms        22409 ms            1 GFLOPS=766.657M/s
BenchmarkMatmul/NaiveRegisterAcc/2048             17638 ms        17572 ms            1 GFLOPS=977.703M/s
BenchmarkMatmul/CacheAware/2048                    2704 ms         2702 ms            1 GFLOPS=6.35844G/s
BenchmarkMatmul/LoopReorder/2048                   1065 ms         1064 ms            1 GFLOPS=16.1496G/s
BenchmarkMatmul/Tiled1D/2048                       1047 ms         1046 ms            1 GFLOPS=16.4267G/s
BenchmarkMatmul/TiledMD/2048                       1146 ms         1145 ms            1 GFLOPS=15.0056G/s
BenchmarkMatmul/SIMD/2048                          1318 ms         1314 ms            1 GFLOPS=13.074G/s
BenchmarkMatmul/Packed/2048                         967 ms          966 ms            1 GFLOPS=17.7877G/s
BenchmarkMatmul/OmpThread/2048                      294 ms          262 ms            3 GFLOPS=65.6016G/s
BenchmarkMatmul/OmpThreadSimd/2048                  349 ms          315 ms            2 GFLOPS=54.6031G/s
BenchmarkMatmul/OmpThreadPacked/2048                208 ms          179 ms            4 GFLOPS=95.7755G/s
BenchmarkMatmul/OmpThreadPackedSimd/2048            222 ms          206 ms            3 GFLOPS=83.4217G/s
BenchmarkMatmul/OmpThreadPackedRegister/2048        107 ms         91.2 ms            8 GFLOPS=188.351G/s
BenchmarkMatmul/Reference/2048                     42.2 ms         42.1 ms           17 GFLOPS=407.641G/s
```

## References

- [Fast Multidimensional Matrix Multiplication on CPU from Scratch, Simon Boehm, 202208](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Optimizing matrix multiplication - Discovering optimizations one at a time, Michal Pitr, 20250216](https://michalpitr.substack.com/p/optimizing-matrix-multiplication)
- [Matrix Multiplication Deep Dive || Cache Blocking, SIMD & Parallelization - Aliaksei Sala - CppCon2025](https://www.youtube.com/watch?v=GHctcSBd6Z4)
- [MIT’s 6.172 on OCW](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/pages/syllabus/)
- [github.com/flame/how-to-optimize-gemm](https://github.com/flame/how-to-optimize-gemm)
