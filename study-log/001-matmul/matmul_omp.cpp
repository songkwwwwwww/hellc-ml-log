#include "matmul.h"
#include <algorithm>
#include <omp.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/thread_policy.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace matmul {
namespace {

/**
 * @brief Thread Pinning (Affinity)
 *
 * Binds the current thread to a specific CPU core. This prevents the OS
 * scheduler from migrating the thread to a different core, which would cause
 * L1/L2 cache invalidation and severely degrade performance.
 */
void PinThreadToCore(int core_id) {
#if defined(__APPLE__)
  thread_affinity_policy_data_t policy = {core_id};
  thread_policy_set(mach_thread_self(), THREAD_AFFINITY_POLICY,
                    (thread_policy_t)&policy, 1);
#else
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

template <int kColsPerRegister, int kRowsPerRegister>
inline void NeonBlock(double *c_block, const double *a_panel,
                      const double *b_panel, int columns, int inners,
                      int inner_count) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  constexpr int neon_doubles = 128 / (sizeof(double) * 8); // Equals 2

  for (int row = 0; row < kRowsPerRegister;
       ++row, c_block += columns, a_panel += inners) {
    const double *b_row = b_panel;
    for (int inner = 0; inner < inner_count; ++inner, b_row += columns) {
      float64x2_t a_reg = vmovq_n_f64(a_panel[inner]);
      for (int col = 0; col < kColsPerRegister; col += neon_doubles) {
        float64x2_t b_reg = vld1q_f64(&b_row[col]);
        float64x2_t c_reg = vld1q_f64(&c_block[col]);
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
 * @brief Multi-threaded Matrix Multiplication using OpenMP
 *
 * By dividing the matrix blocks among multiple CPU cores, we can parallelize
 * the computation. OpenMP makes it easy to distribute loop iterations.
 */
void OmpThread(const double *A, const double *B, double *C, int rows,
               int columns, int inners) {
  constexpr int kRowMacroTile = 180;
  constexpr int kColMacroTile = 96;
  constexpr int kInnerMacroTile = 240;
  constexpr int kColsPerRegister = 8;
  constexpr int kRowsPerRegister = 4;

#pragma omp parallel
  {
    // Pin each thread to a specific core to maximize cache reuse
    int thread_id = omp_get_thread_num();
    PinThreadToCore(thread_id);

    // Apply parallelization to multiple loops (top-level row and col blocks)
    // collapse(2): Fuses the 'row_block' and 'col_block' loops into a single
    // larger loop
    // before distributing it among threads. This ensures a more balanced
    // workload. schedule(dynamic): Threads dynamically request more chunks of
    // work as they finish, handling cases where some blocks take longer than
    // others.
#pragma omp for collapse(2) schedule(dynamic)
    for (int row_block = 0; row_block < rows; row_block += kRowMacroTile) {
      for (int col_block = 0; col_block < columns; col_block += kColMacroTile) {
        int row_block_end = std::min(row_block + kRowMacroTile, rows);
        int col_block_end = std::min(col_block + kColMacroTile, columns);

        for (int inner_block = 0; inner_block < inners;
             inner_block += kInnerMacroTile) {
          int inner_block_end = std::min(inner_block + kInnerMacroTile, inners);
          int inner_count = inner_block_end - inner_block;

          // Micro-kernel loops
          for (int row_micro_block = row_block; row_micro_block < row_block_end;
               row_micro_block += kRowsPerRegister) {
            int row_tile_size =
                std::min(kRowsPerRegister, row_block_end - row_micro_block);
            for (int col_micro_block = col_block;
                 col_micro_block < col_block_end;
                 col_micro_block += kColsPerRegister) {
              int col_tile_size =
                  std::min(kColsPerRegister, col_block_end - col_micro_block);

              if (row_tile_size == kRowsPerRegister &&
                  col_tile_size == kColsPerRegister) {
                const double *a_panel =
                    &A[row_micro_block * inners + inner_block];
                const double *b_panel =
                    &B[inner_block * columns + col_micro_block];
                double *c_block =
                    &C[row_micro_block * columns + col_micro_block];
                NeonBlock<kColsPerRegister, kRowsPerRegister>(
                    c_block, a_panel, b_panel, columns, inners, inner_count);
              } else {
                // Edge case handling for non-multiple boundary blocks
                for (int row = 0; row < row_tile_size; ++row) {
                  for (int inner = 0; inner < inner_count; ++inner) {
                    double a_val = A[(row_micro_block + row) * inners +
                                     inner_block + inner];
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
}

} // namespace matmul
