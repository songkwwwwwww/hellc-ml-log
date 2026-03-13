# 001: Matrix Multiplication (Matmul)

The goal of this project is not to write a competitive BLAS implementation, but to learn about common performance optimizations

This directory contains various implementations of the Matrix Multiplication (GEMM) algorithm, progressing from a naive approach to highly optimized hardware-aware versions. The goal is to study how different low-level software optimization techniques affect computational throughput (GFLOPS).

For readability, the study implementations now assume square inputs. The
tile-based variants also assume the matrix size is divisible by their tile
size, so the teaching code can avoid remainder-handling branches.

## Implementations

1. **`matmul_naive.cpp`**: The standard $O(N^3)$ triple-nested loop implementation without any optimizations.
2. **`matmul_loop_reorder.cpp`**: Optimizes memory access patterns (spatial locality) by reordering loops from `row-col-inner` to `row-inner-col`.
3. **`matmul_tiled.cpp`**: Introduces block-tiling to improve Cache hit rates by keeping active sub-matrices within the L1/L2 cache.
4. **`matmul_packed.cpp`**: Packs matrix tiles into continuous memory buffers to minimize TLB (Translation Lookaside Buffer) misses and cache conflicts.
5. **`matmul_simd.cpp`**: Utilizes ARM NEON Intrinsics to compute multiple data points in a single instruction cycle (Vectorization).
6. **`matmul_omp.cpp` / `OmpThread`**: A cache-tiled OpenMP baseline specialized for the fixed `2024 x 2024 x 2024` study workload, so the code can stay focused on the threading idea.
7. **`matmul_omp.cpp` / `OmpThreadSimd`**: Adds an explicit SIMD micro-kernel inside each OpenMP tile while keeping the same fixed-size study assumption.
8. **`matmul_omp.cpp` / `OmpThreadPacked`**: Packs per-thread `A`/`B` tiles into contiguous scratch buffers before multiplying them to reduce strided memory traffic.
9. **`matmul_omp.cpp` / `OmpThreadPackedSimd`**: Combines OpenMP tiling, per-thread packing, and SIMD in the micro-kernel to study the stacked effect of all three optimizations.
10. **`matmul_omp.cpp` / `OmpThreadPackedRegister`**: Uses a packed `4x8` register-blocked micro-kernel so each thread accumulates a small `C` tile in NEON registers before writing it back.
11. **`matmul_reference.cpp`**: Wraps the highly-optimized system BLAS library (e.g., Apple Accelerate Framework or OpenBLAS) for ground-truth correctness and baseline performance comparisons.

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
The OpenMP variants are intentionally checked on the fixed `2024 x 2024`
study workload because their code path now drops ragged-tile handling for
readability.

### Run Performance Benchmarks
Measures the GFLOPS (Giga-Floating Point Operations Per Second) of each method using Google Benchmark.
```bash
bazel run -c opt //study-log/001-matmul:matmul_bench
```
The benchmark target now uses one fixed `2024 x 2024` input size so the OpenMP
study variants and the simpler teaching code stay aligned.

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
results will vary on other machines. The sample output below is an older
`2048 x 2048` run, so rerun the benchmark commands above for the current
`2024 x 2024` setup.

```
Run on (10 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x10)
Load Average: 1.79, 1.64, 1.66
-------------------------------------------------------------------------------------------------------
Benchmark                                             Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------------------
BenchmarkMatmul/Naive/2024                         7188 ms         7188 ms            1 GFLOPS=2.30703G/s
BenchmarkMatmul/NaiveRegisterAcc/2024              4051 ms         4050 ms            1 GFLOPS=4.09405G/s
BenchmarkMatmul/LoopReorder/2024                    995 ms          995 ms            1 GFLOPS=16.6639G/s
BenchmarkMatmul/Tiled1D/2024                       1001 ms         1001 ms            1 GFLOPS=16.5688G/s
BenchmarkMatmul/TiledMD/2024                       1531 ms         1531 ms            1 GFLOPS=10.8294G/s
BenchmarkMatmul/SIMD/2024                          1080 ms         1080 ms            1 GFLOPS=15.3517G/s
BenchmarkMatmul/Packed/2024                         581 ms          581 ms            1 GFLOPS=28.5182G/s
BenchmarkMatmul/OmpThread/2024                      199 ms          182 ms            4 GFLOPS=91.3331G/s
BenchmarkMatmul/OmpThreadSimd/2024                  237 ms          209 ms            3 GFLOPS=79.2088G/s
BenchmarkMatmul/OmpThreadPacked/2024                283 ms          260 ms            3 GFLOPS=63.6937G/s
BenchmarkMatmul/OmpThreadPackedSimd/2024            211 ms          195 ms            4 GFLOPS=84.9529G/s
BenchmarkMatmul/OmpThreadPackedRegister/2024       88.9 ms         75.4 ms           10 GFLOPS=219.906G/s
BenchmarkMatmul/Reference/2024                     39.9 ms         39.9 ms           17 GFLOPS=415.385G/s
```

## References

- [Fast Multidimensional Matrix Multiplication on CPU from Scratch, Simon Boehm, 202208](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Optimizing matrix multiplication - Discovering optimizations one at a time, Michal Pitr, 20250216](https://michalpitr.substack.com/p/optimizing-matrix-multiplication)
- [Matrix Multiplication Deep Dive || Cache Blocking, SIMD & Parallelization - Aliaksei Sala - CppCon2025](https://www.youtube.com/watch?v=GHctcSBd6Z4)
- [MIT’s 6.172 on OCW](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/pages/syllabus/)
- [github.com/flame/how-to-optimize-gemm](https://github.com/flame/how-to-optimize-gemm)
