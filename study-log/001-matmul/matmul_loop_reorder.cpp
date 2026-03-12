#include "matmul.h"

namespace matmul {

/**
 * @brief Loop Reordered Matrix Multiplication (row-inner-col loop order)
 *
 * By changing the loop order from row-col-inner to row-inner-col, we
 * dramatically improve memory access patterns. This is the simplest yet most
 * effective software optimization for matrix multiplication.
 *
 * Why it's faster:
 * - Spatial Locality: The innermost loop is now 'col'. As 'col' increments, we
 *   access B[inner * columns + col] and C[row * columns + col] sequentially.
 *   Since matrices are row-major, these sequential accesses perfectly align
 *   with CPU cache lines.
 * - When memory is read sequentially, the CPU prefetcher can efficiently load
 *   upcoming data into the cache before it's needed, minimizing stall times.
 */
void LoopReorder(const double *A, const double *B, double *C, int rows,
                 int columns, int inners) {
  for (int row = 0; row < rows; ++row) {
    for (int inner = 0; inner < inners; ++inner) {
      // a_val is loop-invariant for the innermost 'col' loop,
      // so it's loaded once and kept in a CPU register.
      double a_val = A[row * inners + inner];
      for (int col = 0; col < columns; ++col) {
        C[row * columns + col] += a_val * B[inner * columns + col];
      }
    }
  }
}

} // namespace matmul
