#include "matmul.h"

namespace matmul {

/**
 * @brief Naive Matrix Multiplication (i-j-k loop order)
 * 
 * This is the textbook implementation of matrix multiplication. 
 * While easy to understand, it is extremely inefficient on modern hardware.
 * 
 * Performance Issues:
 * - Poor Cache Utilization: In the innermost 'k' loop, B[k * N + j] is accessed.
 *   Because matrices are stored in row-major order, accessing elements column-wise
 *   (incrementing 'k') causes large memory jumps. This ruins spatial locality,
 *   leading to frequent CPU cache misses.
 * - Time Complexity: O(M * N * K)
 */
void naive(const double *A, const double *B, double *C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

} // namespace matmul
