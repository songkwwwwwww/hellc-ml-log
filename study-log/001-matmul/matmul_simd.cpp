#include "matmul.h"
#include <algorithm>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace matmul {

namespace {

/**
 * @brief SIMD (Single Instruction, Multiple Data) Micro-Kernel
 *
 * SIMD instructions allow the CPU to perform the same operation on multiple
 * data points simultaneously. This heavily accelerates compute-bound tasks like
 * matmul.
 *
 * Concept:
 * - We load multiple elements (e.g., 2 double-precision floats in a 128-bit
 * NEON register) into a single vector register.
 * - A single fused multiply-add (FMA) instruction multiplies them and adds them
 * to the accumulator vector.
 * - Vectorizing the innermost loop ('col') works best because of contiguous
 * memory access.
 */
inline void SimdBlock(const double *A, const double *B, double *C, int rows,
                      int columns, int inners, int row_block, int col_block,
                      int inner_block, int tile_size) {
  int row_end = std::min(row_block + tile_size, rows);
  int inner_end = std::min(inner_block + tile_size, inners);
  int col_end = std::min(col_block + tile_size, columns);

  for (int row = row_block; row < row_end; ++row) {
    for (int inner = inner_block; inner < inner_end; ++inner) {
      double a_val = A[row * inners + inner];
#if defined(__ARM_NEON)
      // Broadcast a_val into all lanes of a vector register
      float64x2_t a_vec = vmovq_n_f64(a_val);
      int col = col_block;

      // Process 2 elements at a time (128-bit / 64-bit = 2 lanes)
      for (; col <= col_end - 2; col += 2) {
        // Load 2 elements from B and C
        float64x2_t b_vec = vld1q_f64(&B[inner * columns + col]);
        float64x2_t c_vec = vld1q_f64(&C[row * columns + col]);

        // Fused Multiply-Add: c_vec = c_vec + a_vec * b_vec
        c_vec = vmlaq_f64(c_vec, a_vec, b_vec);

        // Store the result back to C
        vst1q_f64(&C[row * columns + col], c_vec);
      }

      // Handle remaining elements (edge cases when columns is not a multiple
      // of 2)
      for (; col < col_end; ++col) {
        C[row * columns + col] += a_val * B[inner * columns + col];
      }
#else
      // Fallback for non-NEON platforms
      for (int col = col_block; col < col_end; ++col) {
        C[row * columns + col] += a_val * B[inner * columns + col];
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
void Simd(const double *A, const double *B, double *C, int rows, int columns,
          int inners) {
  const int kTileSize = 512;

  // Outer loops for cache tiling
  for (int row_block = 0; row_block < rows; row_block += kTileSize) {
    for (int inner_block = 0; inner_block < inners; inner_block += kTileSize) {
      for (int col_block = 0; col_block < columns; col_block += kTileSize) {
        SimdBlock(A, B, C, rows, columns, inners, row_block, col_block,
                  inner_block, kTileSize);
      }
    }
  }
}

} // namespace matmul
