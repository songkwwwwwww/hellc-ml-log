#include "matmul.h"
#include "matrix_utils.h"
#include <algorithm>
#include <cstring>

namespace matmul {

// Optimized block size for M1 Mac (Considering L1 cache and register constraints)
const int BLOCK_SIZE = 64;

/**
 * @brief Packed Block Multiplication Kernel
 * 
 * Calculates a partial matrix of C using packed tile data.
 * Packing guarantees that the memory accessed by the kernel is perfectly
 * contiguous, which maximizes cache line utilization and reduces Translation
 * Lookaside Buffer (TLB) misses.
 */
void multiply_packed_block(const double *__restrict__ a_packed,
                           const double *__restrict__ b_packed,
                           double *__restrict__ C, const int M, const int N,
                           const int K, const int br, const int bc,
                           const int bk, const int r_limit, const int c_limit,
                           const int k_limit) {

  // Provides an alignment hint to the compiler. This enables the compiler
  // to safely generate aligned vectorized instructions (like AVX or NEON)
  // without having to write explicit intrinsics.
  a_packed = (const double *)__builtin_assume_aligned(a_packed, 64);
  b_packed = (const double *)__builtin_assume_aligned(b_packed, 64);

  // i-k-j loop order maximizes spatial locality within the packed blocks.
  for (int i = 0; i < r_limit; ++i) {
    for (int k = 0; k < k_limit; ++k) {
      double a_val = a_packed[i * BLOCK_SIZE + k];
      // The 'j' loop accesses contiguous memory, making it ideal for auto-vectorization
      for (int j = 0; j < c_limit; ++j) {
        C[(br + i) * N + (bc + j)] += a_val * b_packed[k * BLOCK_SIZE + j];
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
void pack_A(double *__restrict__ dest, const double *__restrict__ src,
            int row_start, int col_start, int M, int K, int r_limit,
            int k_limit) {
  for (int i = 0; i < r_limit; ++i) {
    std::memcpy(&dest[i * BLOCK_SIZE], &src[(row_start + i) * K + col_start],
                k_limit * sizeof(double));
  }
}

void pack_B(double *__restrict__ dest, const double *__restrict__ src,
            int row_start, int col_start, int K, int N, int k_limit,
            int c_limit) {
  for (int i = 0; i < k_limit; ++i) {
    std::memcpy(&dest[i * BLOCK_SIZE], &src[(row_start + i) * N + col_start],
                c_limit * sizeof(double));
  }
}

/**
 * @brief Tiled & Packed Matmul (Row-Major)
 */
void packed(const double *A, const double *B, double *C, int M, int N, int K) {
  // Allocate local buffers matching the tile size. 
  // Allocated once and reused to minimize memory allocation overhead.
  double *a_packed = allocate_aligned(BLOCK_SIZE * BLOCK_SIZE);
  double *b_packed = allocate_aligned(BLOCK_SIZE * BLOCK_SIZE);

  // 3-level tiling loops
  for (int br = 0; br < M; br += BLOCK_SIZE) {
    int r_limit = std::min(BLOCK_SIZE, M - br);
    for (int bc = 0; bc < N; bc += BLOCK_SIZE) {
      int c_limit = std::min(BLOCK_SIZE, N - bc);
      for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
        int k_limit = std::min(BLOCK_SIZE, K - bk);

        // Pack the current tiles into contiguous local buffers
        pack_A(a_packed, A, br, bk, M, K, r_limit, k_limit);
        pack_B(b_packed, B, bk, bc, K, N, k_limit, c_limit);

        // Execute the fast kernel on the packed data
        multiply_packed_block(a_packed, b_packed, C, M, N, K, br, bc, bk,
                              r_limit, c_limit, k_limit);
      }
    }
  }

  free_aligned(a_packed);
  free_aligned(b_packed);
}

} // namespace matmul
