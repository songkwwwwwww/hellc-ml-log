#include "matmul.h"
#include "matrix_utils.h"
#include <benchmark/benchmark.h>
#include <vector>

using namespace matmul;

/**
 * @brief Matrix Multiplication Benchmark function for Google Benchmark
 * 
 * Automatically measures the execution time and calculates the GFLOPS 
 * (Giga Floating-Point Operations Per Second) for each implementation.
 * GFLOPS is the standard metric for measuring the throughput of matrix operations.
 */
static void BM_Matmul(benchmark::State &state, MatmulFunc func,
                      const std::string &name) {
  const int M = state.range(0);
  const int N = M;
  const int K = M;

  // Allocate memory with proper alignment
  double *A = allocate_aligned(M * K);
  double *B = allocate_aligned(K * N);
  double *C = allocate_aligned(M * N);

  // Initialize with random numbers
  initialize_random(A, M * K);
  initialize_random(B, K * N);

  // Benchmark loop
  for (auto _ : state) {
    func(A, B, C, M, N, K);
    // Prevent the compiler from optimizing away the result matrix C
    benchmark::DoNotOptimize(C);
  }

  // Calculate GFLOPS: 
  // Each element in the result matrix requires K multiplications and K additions, 
  // resulting in 2 * K operations. For an M x N matrix, the total is 2 * M * N * K.
  double total_ops = static_cast<double>(state.iterations()) * 2.0 * M * N * K;
  state.counters["GFLOPS"] =
      benchmark::Counter(total_ops, benchmark::Counter::kIsRate);

  free_aligned(A);
  free_aligned(B);
  free_aligned(C);
}

// Register benchmarks using a macro for convenience
// The test range is fixed at a 2048 x 2048 matrix to represent a realistically large workload
#define REGISTER_BENCHMARK(func, name)                                         \
  BENCHMARK_CAPTURE(BM_Matmul, name, func, #name)                              \
      ->RangeMultiplier(2)                                                     \
      ->Range(2048, 2048)                                                      \
      ->Unit(benchmark::kMillisecond);

REGISTER_BENCHMARK(naive, Naive);
REGISTER_BENCHMARK(loop_reorder, LoopReorder);
REGISTER_BENCHMARK(tiled, Tiled);
REGISTER_BENCHMARK(simd, SIMD);
REGISTER_BENCHMARK(cache_aware, CacheAware);
REGISTER_BENCHMARK(omp_thread, OmpThread);
REGISTER_BENCHMARK(packed, Packed);
REGISTER_BENCHMARK(reference, Reference);

BENCHMARK_MAIN();
