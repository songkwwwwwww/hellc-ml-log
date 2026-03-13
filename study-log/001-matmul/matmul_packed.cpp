#include "matmul.h"
#include "matrix_utils.h"
#include <cassert>
#include <cstring>

namespace matmul {

namespace {

constexpr int kTileSize = 88;

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
                         double *__restrict__ c_block, int size) {

  // Provides an alignment hint to the compiler. This enables the compiler
  // to safely generate aligned vectorized instructions (like AVX or NEON)
  // without having to write explicit intrinsics.
  a_packed = (const double *)__builtin_assume_aligned(a_packed, 64);
  b_packed = (const double *)__builtin_assume_aligned(b_packed, 64);

  // row-inner-col loop order maximizes spatial locality within the packed
  // blocks.
  for (int row = 0; row < kTileSize; ++row) {
    double *c_row = &c_block[row * size];
    for (int inner = 0; inner < kTileSize; ++inner) {
      double a_val = a_packed[row * kTileSize + inner];
      // The 'col' loop accesses contiguous memory, making it ideal for
      // auto-vectorization
      const double *b_row = &b_packed[inner * kTileSize];
      for (int col = 0; col < kTileSize; ++col) {
        c_row[col] += a_val * b_row[col];
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
void PackA(double *__restrict__ dest, const double *__restrict__ src, int size,
           int row_start, int inner_start) {
  for (int row = 0; row < kTileSize; ++row) {
    std::memcpy(&dest[row * kTileSize],
                &src[(row_start + row) * size + inner_start],
                kTileSize * sizeof(double));
  }
}

void PackB(double *__restrict__ dest, const double *__restrict__ src, int size,
           int inner_start, int col_start) {
  for (int inner = 0; inner < kTileSize; ++inner) {
    std::memcpy(&dest[inner * kTileSize],
                &src[(inner_start + inner) * size + col_start],
                kTileSize * sizeof(double));
  }
}

} // namespace

/**
 * @brief Tiled & Packed Matmul (Row-Major)
 */
void Packed(const double *A, const double *B, double *C, int rows, int columns,
            int inners) {
  assert(rows == columns);
  assert(columns == inners);
  const int size = rows;
  assert(size % kTileSize == 0);

  // Allocate local buffers matching the tile size.
  // Allocated once and reused to minimize memory allocation overhead.
  double *a_packed = AllocateAligned(kTileSize * kTileSize);
  double *b_packed = AllocateAligned(kTileSize * kTileSize);

  // 3-level tiling loops
  for (int row_block = 0; row_block < size; row_block += kTileSize) {
    for (int col_block = 0; col_block < size; col_block += kTileSize) {
      for (int inner_block = 0; inner_block < size; inner_block += kTileSize) {
        // Pack the current tiles into contiguous local buffers
        PackA(a_packed, A, size, row_block, inner_block);
        PackB(b_packed, B, size, inner_block, col_block);

        // Execute the fast kernel on the packed data
        MultiplyPackedBlock(a_packed, b_packed,
                            &C[row_block * size + col_block], size);
      }
    }
  }

  FreeAligned(a_packed);
  FreeAligned(b_packed);
}

} // namespace matmul
