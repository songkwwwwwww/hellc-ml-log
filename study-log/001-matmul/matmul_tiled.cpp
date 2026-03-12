#include "matmul.h"
#include <algorithm>

namespace matmul {

/**
 * @brief Tiled (Blocked) Matrix Multiplication
 *
 * Tiling or "Loop Blocking" improves Temporal Locality (data reuse in the
 * cache). Instead of computing the entire matrix at once, we compute it in
 * small blocks (tiles) that fit entirely within the CPU's fast L1 or L2 cache.
 *
 * Concept:
 * - When matrices are too large, elements fetched into the cache are evicted
 *   before they can be reused.
 * - By dividing matrices into TILE_SIZE x TILE_SIZE blocks, we ensure that
 *   the working set fits in the cache. The CPU can perform many arithmetic
 *   operations on the cached block before fetching the next one.
 *
 * Note: TILE_SIZE should ideally be tuned based on the target machine's L1/L2
 * cache size.
 */
void Tiled(const double *A, const double *B, double *C, int rows, int columns,
           int inners) {
  // A typical tile size that balances L1/L2 cache utilization on modern CPUs.
  const int kTileSize = 512;

  // Outer loops iterate over the tiles
  for (int row_block = 0; row_block < rows; row_block += kTileSize) {
    for (int inner_block = 0; inner_block < inners; inner_block += kTileSize) {
      for (int col_block = 0; col_block < columns; col_block += kTileSize) {
        // Micro-kernels: Inner loops compute the matrix multiplication for the
        // current tile
        for (int row = row_block;
             row < std::min(row_block + kTileSize, rows); ++row) {
          for (int inner = inner_block;
               inner < std::min(inner_block + kTileSize, inners); ++inner) {
            double a_val = A[row * inners + inner];
            for (int col = col_block;
                 col < std::min(col_block + kTileSize, columns); ++col) {
              C[row * columns + col] +=
                  a_val * B[inner * columns + col];
            }
          }
        }
      }
    }
  }
}

} // namespace matmul
