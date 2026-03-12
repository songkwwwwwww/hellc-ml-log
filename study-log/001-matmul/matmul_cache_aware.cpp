#include "matmul.h"
#include <algorithm>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace matmul {
namespace {

/**
 * @brief Micro-Kernel for Register Blocking
 *
 * Register blocking is an advanced optimization where we compute a small
 * Mr x Nr tile of C entirely within the CPU's registers. This minimizes
 * memory round-trips to the L1 cache.
 *
 * Concept:
 * - We hold a Mr x Nr block of matrix C in registers.
 * - We load a Mr x 1 column from A and a 1 x Nr row from B into registers.
 * - We compute the outer product and accumulate into the C registers.
 * - This drastically improves the computation-to-memory-access ratio.
 */
template <int kColsPerRegister, int kRowsPerRegister>
inline void NeonBlock(double *c_block, const double *a_panel,
                      const double *b_panel, int columns, int inners,
                      int inner_count) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  constexpr int neon_doubles =
      128 / (sizeof(double) * 8); // 2 doubles per 128-bit vector

  for (int row = 0; row < kRowsPerRegister;
       ++row, c_block += columns, a_panel += inners) {
    const double *b_row = b_panel;
    for (int inner = 0; inner < inner_count; ++inner, b_row += columns) {
      // Broadcast one element of A to the vector register
      float64x2_t a_reg = vmovq_n_f64(a_panel[inner]);

      for (int col = 0; col < kColsPerRegister; col += neon_doubles) {
        float64x2_t b_reg = vld1q_f64(&b_row[col]);
        float64x2_t c_reg = vld1q_f64(&c_block[col]);
        // FMA: C = C + A * B
        c_reg = vmlaq_f64(c_reg, a_reg, b_reg);
        vst1q_f64(&c_block[col], c_reg);
      }
    }
  }
#else
  for (int row = 0; row < kRowsPerRegister;
       ++row, c_block += columns, a_panel += inners) {
    const double *b_row = b_panel;
    for (int inner = 0; inner < inner_count; ++inner, b_row += columns) {
      double a_val = a_panel[inner];
      for (int col = 0; col < kColsPerRegister; ++col) {
        c_block[col] += a_val * b_row[col];
      }
    }
  }
#endif
}

} // namespace

/**
 * @brief Cache-Aware Matrix Multiplication (Hierarchical Tiling)
 *
 * Implements BLIS-style hierarchical tiling to optimally utilize L1, L2, and L3
 * caches.
 *
 * Tile sizes:
 * - row/col/inner macro tiles: Designed to fit in higher-level caches
 *   (L2/L3).
 * - row/col register tiles: Designed to fit in the CPU's vector registers.
 */
void CacheAware(const double *A, const double *B, double *C, int rows,
                int columns, int inners) {
  // Optimized tile sizes for common CPU architectures
  constexpr int kRowMacroTile = 180;
  constexpr int kColMacroTile = 96;
  constexpr int kInnerMacroTile = 240;
  constexpr int kColsPerRegister = 8;
  constexpr int kRowsPerRegister = 4;

  // Macro-tiling for L2/L3 Caches
  for (int row_block = 0; row_block < rows; row_block += kRowMacroTile) {
    int row_block_end = std::min(row_block + kRowMacroTile, rows);
    for (int inner_block = 0; inner_block < inners;
         inner_block += kInnerMacroTile) {
      int inner_block_end = std::min(inner_block + kInnerMacroTile, inners);
      int inner_count = inner_block_end - inner_block;
      for (int col_block = 0; col_block < columns;
           col_block += kColMacroTile) {
        int col_block_end = std::min(col_block + kColMacroTile, columns);

        // Micro-tiling for L1 Cache and Registers
        for (int row_micro_block = row_block; row_micro_block < row_block_end;
             row_micro_block += kRowsPerRegister) {
          int row_tile_size =
              std::min(kRowsPerRegister, row_block_end - row_micro_block);
          for (int col_micro_block = col_block;
               col_micro_block < col_block_end;
               col_micro_block += kColsPerRegister) {
            int col_tile_size =
                std::min(kColsPerRegister, col_block_end - col_micro_block);

            // Fast path: fully fits the register blocking dimensions
            if (row_tile_size == kRowsPerRegister &&
                col_tile_size == kColsPerRegister) {
              const double *a_panel = &A[row_micro_block * inners + inner_block];
              const double *b_panel =
                  &B[inner_block * columns + col_micro_block];
              double *c_block =
                  &C[row_micro_block * columns + col_micro_block];
              NeonBlock<kColsPerRegister, kRowsPerRegister>(
                  c_block, a_panel, b_panel, columns, inners, inner_count);
            } else {
              // Edge case handling for boundaries that are not multiples of
              // the register tile size.
              for (int row = 0; row < row_tile_size; ++row) {
                for (int inner = 0; inner < inner_count; ++inner) {
                  double a_val =
                      A[(row_micro_block + row) * inners + inner_block + inner];
                  for (int col = 0; col < col_tile_size; ++col) {
                    C[(row_micro_block + row) * columns + col_micro_block +
                      col] += a_val * B[(inner_block + inner) * columns +
                                        col_micro_block + col];
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
