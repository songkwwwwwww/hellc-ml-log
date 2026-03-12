#include "matmul.h"
#include "matrix_utils.h"
#include <algorithm>
#include <cstring>

namespace matmul {

// Optimized block size for M1 Mac (considering L1 cache and register
// constraints)
constexpr int kBlockSize = 64;

/**
 * @brief Packed Block Multiplication Kernel
 *
 * Calculates a partial matrix of C using packed tile data.
 * Packing guarantees that the memory accessed by the kernel is perfectly
 * contiguous, which maximizes cache line utilization and reduces Translation
 * Lookaside Buffer (TLB) misses.
 */
void MultiplyPackedBlock(const double *__restrict__ a_packed,
                         const double *__restrict__ b_packed,
                         double *__restrict__ C, const int columns,
                         const int row_block, const int col_block,
                         const int row_limit, const int col_limit,
                         const int inner_limit) {

  // Provides an alignment hint to the compiler. This enables the compiler
  // to safely generate aligned vectorized instructions (like AVX or NEON)
  // without having to write explicit intrinsics.
  a_packed = (const double *)__builtin_assume_aligned(a_packed, 64);
  b_packed = (const double *)__builtin_assume_aligned(b_packed, 64);

  // row-inner-col loop order maximizes spatial locality within the packed
  // blocks.
  for (int row = 0; row < row_limit; ++row) {
    for (int inner = 0; inner < inner_limit; ++inner) {
      double a_val = a_packed[row * kBlockSize + inner];
      // The 'col' loop accesses contiguous memory, making it ideal for
      // auto-vectorization
      for (int col = 0; col < col_limit; ++col) {
        C[(row_block + row) * columns + (col_block + col)] +=
            a_val * b_packed[inner * kBlockSize + col];
      }
    }
  }
}

/**
 * @brief Matrix Packing Functions
 *
 * Copies a sub-tile from the original matrix (which might have large strides
 * between rows) into a small, continuous, locally allocated buffer.
 * While packing adds a small overhead, it vastly speeds up the intensive
 * multiplication kernel.
 */
void PackA(double *__restrict__ dest, const double *__restrict__ src,
           int row_start, int inner_start, int inners, int row_limit,
           int inner_limit) {
  for (int row = 0; row < row_limit; ++row) {
    std::memcpy(&dest[row * kBlockSize],
                &src[(row_start + row) * inners + inner_start],
                inner_limit * sizeof(double));
  }
}

void PackB(double *__restrict__ dest, const double *__restrict__ src,
           int inner_start, int col_start, int columns, int inner_limit,
           int col_limit) {
  for (int inner = 0; inner < inner_limit; ++inner) {
    std::memcpy(&dest[inner * kBlockSize],
                &src[(inner_start + inner) * columns + col_start],
                col_limit * sizeof(double));
  }
}

/**
 * @brief Tiled & Packed Matmul (Row-Major)
 */
void Packed(const double *A, const double *B, double *C, int rows, int columns,
            int inners) {
  // Allocate local buffers matching the tile size.
  // Allocated once and reused to minimize memory allocation overhead.
  double *a_packed = AllocateAligned(kBlockSize * kBlockSize);
  double *b_packed = AllocateAligned(kBlockSize * kBlockSize);

  // 3-level tiling loops
  for (int row_block = 0; row_block < rows; row_block += kBlockSize) {
    int row_limit = std::min(kBlockSize, rows - row_block);
    for (int col_block = 0; col_block < columns; col_block += kBlockSize) {
      int col_limit = std::min(kBlockSize, columns - col_block);
      for (int inner_block = 0; inner_block < inners;
           inner_block += kBlockSize) {
        int inner_limit = std::min(kBlockSize, inners - inner_block);

        // Pack the current tiles into contiguous local buffers
        PackA(a_packed, A, row_block, inner_block, inners, row_limit,
              inner_limit);
        PackB(b_packed, B, inner_block, col_block, columns, inner_limit,
              col_limit);

        // Execute the fast kernel on the packed data
        MultiplyPackedBlock(a_packed, b_packed, C, columns, row_block,
                            col_block, row_limit, col_limit, inner_limit);
      }
    }
  }

  FreeAligned(a_packed);
  FreeAligned(b_packed);
}

} // namespace matmul
