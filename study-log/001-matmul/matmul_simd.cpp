#include "matmul.h"
#include <algorithm>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace matmul {

namespace {

/**
 * @brief SIMD (Single Instruction, Multiple Data) Micro-Kernel
 * 
 * SIMD instructions allow the CPU to perform the same operation on multiple data
 * points simultaneously. This heavily accelerates compute-bound tasks like matmul.
 * 
 * Concept:
 * - We load multiple elements (e.g., 2 double-precision floats in a 128-bit NEON register) 
 *   into a single vector register.
 * - A single fused multiply-add (FMA) instruction multiplies them and adds them to 
 *   the accumulator vector.
 * - Vectorizing the innermost loop ('j') works best because of contiguous memory access.
 */
inline void simd_block(const double *A, const double *B, double *C, int M,
                       int N, int K, int ii, int jj, int kk, int TILE_SIZE) {
  int i_end = std::min(ii + TILE_SIZE, M);
  int k_end = std::min(kk + TILE_SIZE, K);
  int j_end = std::min(jj + TILE_SIZE, N);

  for (int i = ii; i < i_end; ++i) {
    for (int k = kk; k < k_end; ++k) {
      double a_val = A[i * K + k];
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
      // Broadcast a_val into all lanes of a vector register
      float64x2_t a_vec = vmovq_n_f64(a_val);
      int j = jj;
      
      // Process 2 elements at a time (128-bit / 64-bit = 2 lanes)
      for (; j <= j_end - 2; j += 2) {
        // Load 2 elements from B and C
        float64x2_t b_vec = vld1q_f64(&B[k * N + j]);
        float64x2_t c_vec = vld1q_f64(&C[i * N + j]);
        
        // Fused Multiply-Add: c_vec = c_vec + a_vec * b_vec
        c_vec = vmlaq_f64(c_vec, a_vec, b_vec);
        
        // Store the result back to C
        vst1q_f64(&C[i * N + j], c_vec);
      }
      
      // Handle remaining elements (edge cases when N is not a multiple of 2)
      for (; j < j_end; ++j) {
        C[i * N + j] += a_val * B[k * N + j];
      }
#else
      // Fallback for non-NEON platforms
      for (int j = jj; j < j_end; ++j) {
        C[i * N + j] += a_val * B[k * N + j];
      }
#endif
    }
  }
}

} // namespace

/**
 * @brief SIMD Matrix Multiplication
 * Combines cache tiling with SIMD vectorization.
 */
void simd(const double *A, const double *B, double *C, int M, int N, int K) {
  const int TILE_SIZE = 512;

  // Outer loops for cache tiling
  for (int ii = 0; ii < M; ii += TILE_SIZE) {
    for (int kk = 0; kk < K; kk += TILE_SIZE) {
      for (int jj = 0; jj < N; jj += TILE_SIZE) {
        simd_block(A, B, C, M, N, K, ii, jj, kk, TILE_SIZE);
      }
    }
  }
}

} // namespace matmul
