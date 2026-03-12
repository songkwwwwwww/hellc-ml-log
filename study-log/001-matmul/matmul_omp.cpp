#include "matmul.h"
#include <algorithm>
#include <cstring>
#include <omp.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace matmul {
namespace {

// This file explores the same GEMM computation through increasingly more
// "BLAS-like" execution strategies:
// 1. Split C into tiles so independent tiles can run on different threads.
// 2. Pack A/B tiles into contiguous scratch buffers to improve locality.
// 3. Keep a tiny C block in registers across the full k-loop and store once.
constexpr int kOmpRowTile = 128;
constexpr int kOmpColTile = 128;
constexpr int kOmpInnerTile = 256;
constexpr int kPackedTileSize = 64;
constexpr int kRegisterRows = 4;
constexpr int kRegisterCols = 8;
constexpr int kRegisterInnerTile = 256;

using TileKernel = void (*)(const double *A, const double *B, double *C,
                            int columns, int inners, int row_begin,
                            int row_end, int col_begin, int col_end,
                            int inner_begin, int inner_end);

using PackedKernel =
    void (*)(const double *a_packed, const double *b_packed, double *c_block,
             int columns, int row_count, int col_count, int inner_count);

inline int TileEnd(int start, int tile_size, int limit) {
  return std::min(start + tile_size, limit);
}

inline void MultiplyTileScalar(const double *A, const double *B, double *C,
                               int columns, int inners, int row_begin,
                               int row_end, int col_begin, int col_end,
                               int inner_begin, int inner_end) {
  // Within one C tile we use row-inner-col order so that B and C are touched
  // along contiguous column slices instead of striding by full rows.
  const int col_count = col_end - col_begin;
  for (int row = row_begin; row < row_end; ++row) {
    double *c_row = &C[row * columns + col_begin];
    for (int inner = inner_begin; inner < inner_end; ++inner) {
      const double a_value = A[row * inners + inner];
      const double *b_row = &B[inner * columns + col_begin];
      for (int col = 0; col < col_count; ++col) {
        c_row[col] += a_value * b_row[col];
      }
    }
  }
}

inline void MultiplyTileSimd(const double *A, const double *B, double *C,
                             int columns, int inners, int row_begin,
                             int row_end, int col_begin, int col_end,
                             int inner_begin, int inner_end) {
  // Same traversal as MultiplyTileScalar, but now each scalar A value is
  // broadcast into a vector register so one multiply-add updates multiple
  // neighboring columns of C at once.
  const int col_count = col_end - col_begin;
  for (int row = row_begin; row < row_end; ++row) {
    double *c_row = &C[row * columns + col_begin];
    for (int inner = inner_begin; inner < inner_end; ++inner) {
      const double a_value = A[row * inners + inner];
      const double *b_row = &B[inner * columns + col_begin];
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
      const float64x2_t a_vec = vmovq_n_f64(a_value);
      int col = 0;
      for (; col <= col_count - 2; col += 2) {
        float64x2_t b_vec = vld1q_f64(&b_row[col]);
        float64x2_t c_vec = vld1q_f64(&c_row[col]);
        c_vec = vmlaq_f64(c_vec, a_vec, b_vec);
        vst1q_f64(&c_row[col], c_vec);
      }
      for (; col < col_count; ++col) {
        c_row[col] += a_value * b_row[col];
      }
#else
      for (int col = 0; col < col_count; ++col) {
        c_row[col] += a_value * b_row[col];
      }
#endif
    }
  }
}

inline void PackATile(double *dest, const double *src, int row_block,
                      int inner_block, int inners, int row_count,
                      int inner_count) {
  // Packing pays a small memcpy cost to remove the original matrix stride from
  // the hot kernel. The arithmetic kernel then reads from compact scratch data.
  for (int row = 0; row < row_count; ++row) {
    std::memcpy(&dest[row * kPackedTileSize],
                &src[(row_block + row) * inners + inner_block],
                inner_count * sizeof(double));
  }
}

inline void PackBTile(double *dest, const double *src, int inner_block,
                      int col_block, int columns, int inner_count,
                      int col_count) {
  for (int inner = 0; inner < inner_count; ++inner) {
    std::memcpy(&dest[inner * kPackedTileSize],
                &src[(inner_block + inner) * columns + col_block],
                col_count * sizeof(double));
  }
}

inline void MultiplyPackedTileScalar(const double *a_packed,
                                     const double *b_packed, double *c_block,
                                     int columns, int row_count, int col_count,
                                     int inner_count) {
  for (int row = 0; row < row_count; ++row) {
    const double *a_row = &a_packed[row * kPackedTileSize];
    double *c_row = &c_block[row * columns];
    for (int inner = 0; inner < inner_count; ++inner) {
      const double a_value = a_row[inner];
      const double *b_row = &b_packed[inner * kPackedTileSize];
      for (int col = 0; col < col_count; ++col) {
        c_row[col] += a_value * b_row[col];
      }
    }
  }
}

inline void MultiplyPackedTileSimd(const double *a_packed,
                                   const double *b_packed, double *c_block,
                                   int columns, int row_count, int col_count,
                                   int inner_count) {
  a_packed =
      static_cast<const double *>(__builtin_assume_aligned(a_packed, 64));
  b_packed =
      static_cast<const double *>(__builtin_assume_aligned(b_packed, 64));

  for (int row = 0; row < row_count; ++row) {
    const double *a_row = &a_packed[row * kPackedTileSize];
    double *c_row = &c_block[row * columns];
    for (int inner = 0; inner < inner_count; ++inner) {
      const double a_value = a_row[inner];
      const double *b_row = &b_packed[inner * kPackedTileSize];
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
      const float64x2_t a_vec = vmovq_n_f64(a_value);
      int col = 0;
      for (; col <= col_count - 2; col += 2) {
        float64x2_t b_vec = vld1q_f64(&b_row[col]);
        float64x2_t c_vec = vld1q_f64(&c_row[col]);
        c_vec = vmlaq_f64(c_vec, a_vec, b_vec);
        vst1q_f64(&c_row[col], c_vec);
      }
      for (; col < col_count; ++col) {
        c_row[col] += a_value * b_row[col];
      }
#else
      for (int col = 0; col < col_count; ++col) {
        c_row[col] += a_value * b_row[col];
      }
#endif
    }
  }
}

inline void PackARegisterPanel(double *dest, const double *src, int row_block,
                               int inner_block, int inners, int row_count,
                               int inner_count) {
  for (int row = 0; row < row_count; ++row) {
    std::memcpy(&dest[row * kRegisterInnerTile],
                &src[(row_block + row) * inners + inner_block],
                inner_count * sizeof(double));
  }
}

inline void PackBRegisterPanel(double *dest, const double *src, int inner_block,
                               int col_block, int columns, int inner_count,
                               int col_count) {
  for (int inner = 0; inner < inner_count; ++inner) {
    std::memcpy(&dest[inner * kRegisterCols],
                &src[(inner_block + inner) * columns + col_block],
                col_count * sizeof(double));
  }
}

inline void MultiplyPackedRegisterRemainder(const double *a_panel,
                                            const double *b_panel,
                                            double *c_block, int columns,
                                            int row_count, int col_count,
                                            int inner_count) {
  for (int row = 0; row < row_count; ++row) {
    const double *a_row = &a_panel[row * kRegisterInnerTile];
    double *c_row = &c_block[row * columns];
    for (int inner = 0; inner < inner_count; ++inner) {
      const double a_value = a_row[inner];
      const double *b_row = &b_panel[inner * kRegisterCols];
      for (int col = 0; col < col_count; ++col) {
        c_row[col] += a_value * b_row[col];
      }
    }
  }
}

inline void MultiplyPackedRegister4x8(const double *a_panel,
                                      const double *b_panel, double *c_block,
                                      int columns, int inner_count) {
  // Register blocking is the key step toward a BLAS-style micro-kernel:
  // - 4 rows of A are broadcast from the packed panel.
  // - 8 columns of B are loaded as four 2-lane NEON vectors.
  // - A 4x8 block of C stays in registers for the whole inner loop.
  // This avoids repeatedly loading/storing C on every k iteration.
  a_panel = static_cast<const double *>(__builtin_assume_aligned(a_panel, 64));
  b_panel = static_cast<const double *>(__builtin_assume_aligned(b_panel, 64));

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  const double *a_row0 = &a_panel[0];
  const double *a_row1 = &a_panel[kRegisterInnerTile];
  const double *a_row2 = &a_panel[2 * kRegisterInnerTile];
  const double *a_row3 = &a_panel[3 * kRegisterInnerTile];

  float64x2_t c00 = vld1q_f64(&c_block[0]);
  float64x2_t c01 = vld1q_f64(&c_block[2]);
  float64x2_t c02 = vld1q_f64(&c_block[4]);
  float64x2_t c03 = vld1q_f64(&c_block[6]);
  float64x2_t c10 = vld1q_f64(&c_block[columns + 0]);
  float64x2_t c11 = vld1q_f64(&c_block[columns + 2]);
  float64x2_t c12 = vld1q_f64(&c_block[columns + 4]);
  float64x2_t c13 = vld1q_f64(&c_block[columns + 6]);
  float64x2_t c20 = vld1q_f64(&c_block[2 * columns + 0]);
  float64x2_t c21 = vld1q_f64(&c_block[2 * columns + 2]);
  float64x2_t c22 = vld1q_f64(&c_block[2 * columns + 4]);
  float64x2_t c23 = vld1q_f64(&c_block[2 * columns + 6]);
  float64x2_t c30 = vld1q_f64(&c_block[3 * columns + 0]);
  float64x2_t c31 = vld1q_f64(&c_block[3 * columns + 2]);
  float64x2_t c32 = vld1q_f64(&c_block[3 * columns + 4]);
  float64x2_t c33 = vld1q_f64(&c_block[3 * columns + 6]);

  for (int inner = 0; inner < inner_count; ++inner) {
    const double *b_row = &b_panel[inner * kRegisterCols];
    const float64x2_t b0 = vld1q_f64(&b_row[0]);
    const float64x2_t b1 = vld1q_f64(&b_row[2]);
    const float64x2_t b2 = vld1q_f64(&b_row[4]);
    const float64x2_t b3 = vld1q_f64(&b_row[6]);

    const float64x2_t a0 = vmovq_n_f64(a_row0[inner]);
    const float64x2_t a1 = vmovq_n_f64(a_row1[inner]);
    const float64x2_t a2 = vmovq_n_f64(a_row2[inner]);
    const float64x2_t a3 = vmovq_n_f64(a_row3[inner]);

    c00 = vmlaq_f64(c00, a0, b0);
    c01 = vmlaq_f64(c01, a0, b1);
    c02 = vmlaq_f64(c02, a0, b2);
    c03 = vmlaq_f64(c03, a0, b3);
    c10 = vmlaq_f64(c10, a1, b0);
    c11 = vmlaq_f64(c11, a1, b1);
    c12 = vmlaq_f64(c12, a1, b2);
    c13 = vmlaq_f64(c13, a1, b3);
    c20 = vmlaq_f64(c20, a2, b0);
    c21 = vmlaq_f64(c21, a2, b1);
    c22 = vmlaq_f64(c22, a2, b2);
    c23 = vmlaq_f64(c23, a2, b3);
    c30 = vmlaq_f64(c30, a3, b0);
    c31 = vmlaq_f64(c31, a3, b1);
    c32 = vmlaq_f64(c32, a3, b2);
    c33 = vmlaq_f64(c33, a3, b3);
  }

  vst1q_f64(&c_block[0], c00);
  vst1q_f64(&c_block[2], c01);
  vst1q_f64(&c_block[4], c02);
  vst1q_f64(&c_block[6], c03);
  vst1q_f64(&c_block[columns + 0], c10);
  vst1q_f64(&c_block[columns + 2], c11);
  vst1q_f64(&c_block[columns + 4], c12);
  vst1q_f64(&c_block[columns + 6], c13);
  vst1q_f64(&c_block[2 * columns + 0], c20);
  vst1q_f64(&c_block[2 * columns + 2], c21);
  vst1q_f64(&c_block[2 * columns + 4], c22);
  vst1q_f64(&c_block[2 * columns + 6], c23);
  vst1q_f64(&c_block[3 * columns + 0], c30);
  vst1q_f64(&c_block[3 * columns + 2], c31);
  vst1q_f64(&c_block[3 * columns + 4], c32);
  vst1q_f64(&c_block[3 * columns + 6], c33);
#else
  MultiplyPackedRegisterRemainder(a_panel, b_panel, c_block, columns,
                                  kRegisterRows, kRegisterCols, inner_count);
#endif
}

void RunOmpTiled(const double *A, const double *B, double *C, int rows,
                 int columns, int inners, TileKernel kernel) {
#pragma omp parallel for default(none)                                         \
    shared(A, B, C, rows, columns, inners, kernel) collapse(2)                \
    schedule(static)
  for (int row_block = 0; row_block < rows; row_block += kOmpRowTile) {
    for (int col_block = 0; col_block < columns; col_block += kOmpColTile) {
      const int row_end = TileEnd(row_block, kOmpRowTile, rows);
      const int col_end = TileEnd(col_block, kOmpColTile, columns);
      for (int inner_block = 0; inner_block < inners;
           inner_block += kOmpInnerTile) {
        const int inner_end = TileEnd(inner_block, kOmpInnerTile, inners);
        kernel(A, B, C, columns, inners, row_block, row_end, col_block,
               col_end, inner_block, inner_end);
      }
    }
  }
}

void RunOmpPackedRegister(const double *A, const double *B, double *C, int rows,
                          int columns, int inners) {
#pragma omp parallel default(none) shared(A, B, C, rows, columns, inners)
  {
    alignas(64) double a_panel[kRegisterRows * kRegisterInnerTile];
    alignas(64) double b_panel[kRegisterInnerTile * kRegisterCols];

#pragma omp for collapse(2) schedule(static)
    for (int row_block = 0; row_block < rows; row_block += kOmpRowTile) {
      for (int col_block = 0; col_block < columns; col_block += kOmpColTile) {
        const int row_end = TileEnd(row_block, kOmpRowTile, rows);
        const int col_end = TileEnd(col_block, kOmpColTile, columns);

        for (int inner_block = 0; inner_block < inners;
             inner_block += kRegisterInnerTile) {
          const int inner_count =
              TileEnd(inner_block, kRegisterInnerTile, inners) - inner_block;

          // Pack one B micro-panel and reuse it across several A micro-panels
          // inside the same thread-owned macro tile. This is much closer to a
          // classic GotoBLAS/BLIS macro-kernel than the simpler packed path.
          for (int col_micro = col_block; col_micro < col_end;
               col_micro += kRegisterCols) {
            const int col_count =
                TileEnd(col_micro, kRegisterCols, col_end) - col_micro;
            PackBRegisterPanel(b_panel, B, inner_block, col_micro, columns,
                               inner_count, col_count);

            for (int row_micro = row_block; row_micro < row_end;
                 row_micro += kRegisterRows) {
              const int row_count =
                  TileEnd(row_micro, kRegisterRows, row_end) - row_micro;
              PackARegisterPanel(a_panel, A, row_micro, inner_block, inners,
                                 row_count, inner_count);

              double *c_block = &C[row_micro * columns + col_micro];
              if (row_count == kRegisterRows && col_count == kRegisterCols) {
                MultiplyPackedRegister4x8(a_panel, b_panel, c_block, columns,
                                          inner_count);
              } else {
                MultiplyPackedRegisterRemainder(a_panel, b_panel, c_block,
                                                columns, row_count, col_count,
                                                inner_count);
              }
            }
          }
        }
      }
    }
  }
}

void RunOmpPacked(const double *A, const double *B, double *C, int rows,
                  int columns, int inners, PackedKernel kernel) {
#pragma omp parallel default(none) shared(A, B, C, rows, columns, inners,     \
                                          kernel)
  {
    alignas(64) double a_packed[kPackedTileSize * kPackedTileSize];
    alignas(64) double b_packed[kPackedTileSize * kPackedTileSize];

#pragma omp for collapse(2) schedule(static)
    for (int row_block = 0; row_block < rows; row_block += kPackedTileSize) {
      for (int col_block = 0; col_block < columns; col_block += kPackedTileSize) {
        const int row_count = TileEnd(row_block, kPackedTileSize, rows) - row_block;
        const int col_count =
            TileEnd(col_block, kPackedTileSize, columns) - col_block;
        double *c_block = &C[row_block * columns + col_block];
        for (int inner_block = 0; inner_block < inners;
             inner_block += kPackedTileSize) {
          const int inner_count =
              TileEnd(inner_block, kPackedTileSize, inners) - inner_block;
          PackATile(a_packed, A, row_block, inner_block, inners, row_count,
                    inner_count);
          PackBTile(b_packed, B, inner_block, col_block, columns, inner_count,
                    col_count);
          kernel(a_packed, b_packed, c_block, columns, row_count, col_count,
                 inner_count);
        }
      }
    }
  }
}

} // namespace

void OmpThread(const double *A, const double *B, double *C, int rows,
               int columns, int inners) {
  RunOmpTiled(A, B, C, rows, columns, inners, MultiplyTileScalar);
}

void OmpThreadSimd(const double *A, const double *B, double *C, int rows,
                   int columns, int inners) {
  RunOmpTiled(A, B, C, rows, columns, inners, MultiplyTileSimd);
}

void OmpThreadPacked(const double *A, const double *B, double *C, int rows,
                     int columns, int inners) {
  RunOmpPacked(A, B, C, rows, columns, inners, MultiplyPackedTileScalar);
}

void OmpThreadPackedSimd(const double *A, const double *B, double *C, int rows,
                         int columns, int inners) {
  RunOmpPacked(A, B, C, rows, columns, inners, MultiplyPackedTileSimd);
}

void OmpThreadPackedRegister(const double *A, const double *B, double *C,
                             int rows, int columns, int inners) {
  RunOmpPackedRegister(A, B, C, rows, columns, inners);
}

} // namespace matmul
