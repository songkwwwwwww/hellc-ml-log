# 001: Matrix Multiplication (Matmul)

This directory contains various implementations of the Matrix Multiplication (GEMM) algorithm, progressing from a naive approach to highly optimized hardware-aware versions. The goal is to study how different low-level software optimization techniques affect computational throughput (GFLOPS).

## Implementations

1. **`matmul_naive.cpp`**: The standard $O(N^3)$ triple-nested loop implementation without any optimizations.
2. **`matmul_loop_reorder.cpp`**: Optimizes memory access patterns (spatial locality) by reordering loops from $i-j-k$ to $i-k-j$.
3. **`matmul_tiled.cpp`**: Introduces block-tiling to improve Cache hit rates by keeping active sub-matrices within the L1/L2 cache.
4. **`matmul_packed.cpp`**: Packs matrix tiles into continuous memory buffers to minimize TLB (Translation Lookaside Buffer) misses and cache conflicts.
5. **`matmul_simd.cpp`**: Utilizes ARM NEON Intrinsics to compute multiple data points in a single instruction cycle (Vectorization).
6. **`matmul_cache_aware.cpp`**: A sophisticated hierarchical approach applying both macro-tiling (Cache blocking) and micro-tiling (Register blocking) with SIMD.
7. **`matmul_omp.cpp`**: Introduces multi-threading using OpenMP (`#pragma omp parallel for`) and thread pinning to distribute workloads across multiple CPU cores efficiently.
8. **`matmul_reference.cpp`**: Wraps the highly-optimized system BLAS library (e.g., Apple Accelerate Framework or OpenBLAS) for ground-truth correctness and baseline performance comparisons.

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

Run on (10 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x10)
Load Average: 2.12, 1.51, 1.44
-------------------------------------------------------------------------------------
Benchmark                           Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------
BM_Matmul/Naive/2048            16460 ms        16460 ms            1 GFLOPS=1.04375G/s
BM_Matmul/LoopReorder/2048       1024 ms         1024 ms            1 GFLOPS=16.7854G/s
BM_Matmul/Tiled/2048             1148 ms         1148 ms            1 GFLOPS=14.9613G/s
BM_Matmul/SIMD/2048              1298 ms         1298 ms            1 GFLOPS=13.2333G/s
BM_Matmul/CacheAware/2048        2623 ms         2623 ms            1 GFLOPS=6.54945G/s
BM_Matmul/OmpThread/2048          481 ms          465 ms            2 GFLOPS=36.9407G/s
BM_Matmul/Packed/2048             952 ms          952 ms            1 GFLOPS=18.0517G/s
BM_Matmul/Reference/2048         42.2 ms         42.2 ms           17 GFLOPS=407.4G/s
