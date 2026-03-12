#include "matmul.h"
#include <algorithm>

namespace matmul {

/**
 * @brief Tiled (Blocked) Matrix Multiplication
 * 
 * Tiling or "Loop Blocking" improves Temporal Locality (data reuse in the cache).
 * Instead of computing the entire matrix at once, we compute it in small blocks
 * (tiles) that fit entirely within the CPU's fast L1 or L2 cache.
 * 
 * Concept:
 * - When matrices are too large, elements fetched into the cache are evicted
 *   before they can be reused.
 * - By dividing matrices into TILE_SIZE x TILE_SIZE blocks, we ensure that
 *   the working set fits in the cache. The CPU can perform many arithmetic 
 *   operations on the cached block before fetching the next one.
 * 
 * Note: TILE_SIZE should ideally be tuned based on the target machine's L1/L2 cache size.
 */
void tiled(const double *A, const double *B, double *C, int M, int N, int K) {
  // A typical tile size that balances L1/L2 cache utilization on modern CPUs.
  const int TILE_SIZE = 512;

  // Outer loops iterate over the tiles
  for (int ii = 0; ii < M; ii += TILE_SIZE) {
    for (int kk = 0; kk < K; kk += TILE_SIZE) {
      for (int jj = 0; jj < N; jj += TILE_SIZE) {
        // Micro-kernels: Inner loops compute the matrix multiplication for the current tile
        for (int i = ii; i < std::min(ii + TILE_SIZE, M); ++i) {
          for (int k = kk; k < std::min(kk + TILE_SIZE, K); ++k) {
            double a_val = A[i * K + k];
            for (int j = jj; j < std::min(jj + TILE_SIZE, N); ++j) {
              C[i * N + j] += a_val * B[k * N + j];
            }
          }
        }
      }
    }
  }
}

} // namespace matmul
