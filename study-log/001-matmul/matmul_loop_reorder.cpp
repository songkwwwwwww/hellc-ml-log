#include "matmul.h"

namespace matmul {

/**
 * @brief Loop Reordered Matrix Multiplication (i-k-j loop order)
 * 
 * By changing the loop order from i-j-k to i-k-j, we dramatically improve
 * memory access patterns. This is the simplest yet most effective software
 * optimization for matrix multiplication.
 * 
 * Why it's faster:
 * - Spatial Locality: The innermost loop is now 'j'. As 'j' increments, we access
 *   B[k * N + j] and C[i * N + j] sequentially. Since matrices are row-major,
 *   these sequential accesses perfectly align with CPU cache lines.
 * - When memory is read sequentially, the CPU prefetcher can efficiently load 
 *   upcoming data into the cache before it's needed, minimizing stall times.
 */
void loop_reorder(const double *A, const double *B, double *C, int M, int N,
                  int K) {
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      // a_val is loop-invariant for the innermost 'j' loop, so it's loaded once
      // and kept in a CPU register.
      double a_val = A[i * K + k];
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += a_val * B[k * N + j];
      }
    }
  }
}

} // namespace matmul
