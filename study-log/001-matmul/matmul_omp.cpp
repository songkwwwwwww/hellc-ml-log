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

/**
 * @brief Multi-threaded Matrix Multiplication using OpenMP
 *
 * By dividing the matrix blocks among multiple CPU cores, we can parallelize
 * the computation. OpenMP makes it easy to distribute loop iterations.
 */
void OmpThread(const double *A, const double *B, double *C, int rows,
               int columns, int inners) {
  const int kTileSize = 256;
#pragma omp parallel for default(none)                                         \
    shared(A, B, C, rows, columns, inners, kTileSize) collapse(2)              \
    num_threads(8)
  for (int rowTile = 0; rowTile < rows; rowTile += 256) {
    for (int columnTile = 0; columnTile < columns; columnTile += 256) {
      for (int innerTile = 0; innerTile < inners; innerTile += kTileSize) {
        for (int row = rowTile; row < rowTile + 256; row++) {
          int innerTileEnd = std::min(inners, innerTile + kTileSize);
          for (int inner = innerTile; inner < innerTileEnd; inner++) {
            for (int col = columnTile; col < columnTile + 256; col++) {
              C[row * columns + col] +=
                  A[row * inners + inner] * B[inner * columns + col];
            }
          }
        }
      }
    }
  }
}

} // namespace matmul
