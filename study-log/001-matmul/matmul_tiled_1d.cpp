#include "matmul.h"
#include <cassert>

namespace matmul {

namespace {

constexpr int kTileSize = 88;

} // namespace

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
void Tiled1D(const double *A, const double *B, double *C, int rows, int columns,
             int inners) {
  assert(rows == columns);
  assert(columns == inners);
  const int size = rows;
  assert(size % kTileSize == 0);

  for (int inner_tile = 0; inner_tile < size; inner_tile += kTileSize) {
    for (int row = 0; row < size; ++row) {
      const double *a_row = &A[row * size + inner_tile];
      double *c_row = &C[row * size];
      for (int inner = 0; inner < kTileSize; ++inner) {
        const double a_value = a_row[inner];
        const double *b_row = &B[(inner_tile + inner) * size];
        for (int col = 0; col < size; ++col) {
          c_row[col] += a_value * b_row[col];
        }
      }
    }
  }
}

} // namespace matmul
