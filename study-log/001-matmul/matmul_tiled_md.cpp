#include "matmul.h"
#include <algorithm>

namespace matmul {

namespace {
// row-major access helper
static inline int idx(int row, int col, int num_cols) {
  return row * num_cols + col;
}

static inline void multiply_block(const double *A, const double *B, double *C,
                                  int columns, int inners, int row_begin,
                                  int row_end, int col_begin, int col_end,
                                  int inner_begin, int inner_end) {
  for (int row = row_begin; row < row_end; ++row) {
    for (int inner = inner_begin; inner < inner_end; ++inner) {
      const double a_value = A[idx(row, inner, inners)];

      for (int col = col_begin; col < col_end; ++col) {
        C[idx(row, col, columns)] += a_value * B[idx(inner, col, columns)];
      }
    }
  }
}

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
void TiledMD(const double *A, const double *B, double *C, int rows, int columns,
             int inners) {

  static constexpr int kTileSize = 1024;
  static constexpr int kL3Tile = kTileSize;
  static constexpr int kL2Tile = kTileSize;
  static constexpr int kL1Tile = kTileSize;

  // L3 blocking
  for (int l3_row = 0; l3_row < rows; l3_row += kL3Tile) {
    const int l3_row_end = std::min(l3_row + kL3Tile, rows);

    for (int l3_col = 0; l3_col < columns; l3_col += kL3Tile) {
      const int l3_col_end = std::min(l3_col + kL3Tile, columns);

      for (int l3_inner = 0; l3_inner < inners; l3_inner += kL3Tile) {
        const int l3_inner_end = std::min(l3_inner + kL3Tile, inners);

        // L2 blocking
        for (int l2_row = l3_row; l2_row < l3_row_end; l2_row += kL2Tile) {
          const int l2_row_end = std::min(l2_row + kL2Tile, l3_row_end);

          for (int l2_col = l3_col; l2_col < l3_col_end; l2_col += kL2Tile) {
            const int l2_col_end = std::min(l2_col + kL2Tile, l3_col_end);

            for (int l2_inner = l3_inner; l2_inner < l3_inner_end;
                 l2_inner += kL2Tile) {
              const int l2_inner_end =
                  std::min(l2_inner + kL2Tile, l3_inner_end);

              // L1 blocking
              for (int l1_row = l2_row; l1_row < l2_row_end;
                   l1_row += kL1Tile) {
                const int l1_row_end = std::min(l1_row + kL1Tile, l2_row_end);

                for (int l1_col = l2_col; l1_col < l2_col_end;
                     l1_col += kL1Tile) {
                  const int l1_col_end = std::min(l1_col + kL1Tile, l2_col_end);

                  for (int l1_inner = l2_inner; l1_inner < l2_inner_end;
                       l1_inner += kL1Tile) {
                    const int l1_inner_end =
                        std::min(l1_inner + kL1Tile, l2_inner_end);

                    multiply_block(A, B, C, columns, inners, l1_row, l1_row_end,
                                   l1_col, l1_col_end, l1_inner, l1_inner_end);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

} // namespace matmul
