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
 * GFLOPS is the standard metric for measuring the throughput of matrix
 * operations.
 */
static void BenchmarkMatmul(benchmark::State &state, MatmulFunc func,
                            const std::string &name) {
  const int rows = state.range(0);
  const int columns = rows;
  const int inners = rows;

  // Allocate memory with proper alignment
  double *A = AllocateAligned(rows * inners);
  double *B = AllocateAligned(inners * columns);
  double *C = AllocateAligned(rows * columns);

  // Initialize with random numbers
  InitializeRandom(A, rows * inners);
  InitializeRandom(B, inners * columns);

  // Benchmark loop
  for (auto _ : state) {
    func(A, B, C, rows, columns, inners);
    // Prevent the compiler from optimizing away the result matrix C
    benchmark::DoNotOptimize(C);
  }

  // Calculate GFLOPS:
  // Each element in the result matrix requires inners multiplications and
  // inners additions, resulting in 2 * inners operations. For a rows x columns
  // result matrix, the total is 2 * rows * columns * inners.
  double total_ops =
      static_cast<double>(state.iterations()) * 2.0 * rows * columns * inners;
  state.counters["GFLOPS"] =
      benchmark::Counter(total_ops, benchmark::Counter::kIsRate);

  FreeAligned(A);
  FreeAligned(B);
  FreeAligned(C);
}

// Register benchmarks using a macro for convenience
// The test range is fixed at a 2048 x 2048 matrix to represent a realistically
// large workload
#define REGISTER_BENCHMARK(func, name)                                         \
  BENCHMARK_CAPTURE(BenchmarkMatmul, name, func, #name)                        \
      ->RangeMultiplier(2)                                                     \
      ->Range(2048, 2048)                                                      \
      ->Unit(benchmark::kMillisecond);

REGISTER_BENCHMARK(Naive, Naive);
REGISTER_BENCHMARK(NaiveRegisterAcc, NaiveRegisterAcc);
REGISTER_BENCHMARK(LoopReorder, LoopReorder);
REGISTER_BENCHMARK(Tiled, Tiled);
REGISTER_BENCHMARK(Simd, SIMD);
REGISTER_BENCHMARK(CacheAware, CacheAware);
REGISTER_BENCHMARK(OmpThread, OmpThread);
REGISTER_BENCHMARK(Packed, Packed);
REGISTER_BENCHMARK(Reference, Reference);

BENCHMARK_MAIN();
