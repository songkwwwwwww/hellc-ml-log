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
template <int Nr, int Mr>
inline void neon_block(double *c, const double *a, const double *mb, int N,
                       int K, int kc_actual) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  constexpr int neon_doubles = 128 / (sizeof(double) * 8); // 2 doubles per 128-bit vector

  for (int i = 0; i < Mr; ++i, c += N, a += K) {
    const double *b = mb;
    for (int k = 0; k < kc_actual; ++k, b += N) {
      // Broadcast one element of A to the vector register
      float64x2_t a_reg = vmovq_n_f64(a[k]);
      
      for (int j = 0; j < Nr; j += neon_doubles) {
        float64x2_t b_reg = vld1q_f64(&b[j]);
        float64x2_t c_reg = vld1q_f64(&c[j]);
        // FMA: C = C + A * B
        c_reg = vmlaq_f64(c_reg, a_reg, b_reg);
        vst1q_f64(&c[j], c_reg);
      }
    }
  }
#else
  for (int i = 0; i < Mr; ++i, c += N, a += K) {
    const double *b = mb;
    for (int k = 0; k < kc_actual; ++k, b += N) {
      double a_val = a[k];
      for (int j = 0; j < Nr; ++j) {
        c[j] += a_val * b[j];
      }
    }
  }
#endif
}

} // namespace

/**
 * @brief Cache-Aware Matrix Multiplication (Hierarchical Tiling)
 * 
 * Implements BLIS-style hierarchical tiling to optimally utilize L1, L2, and L3 caches.
 * 
 * Tile sizes:
 * - Mc, Nc, Kc: Macro-tile sizes designed to fit in higher-level caches (L2/L3).
 * - Mr, Nr: Micro-tile sizes designed to fit in the CPU's vector registers.
 */
void cache_aware(const double *A, const double *B, double *C, int M, int N,
                 int K) {
  // Optimized tile sizes for common CPU architectures
  constexpr int Mc = 180, Nc = 96, Kc = 240, Nr = 8, Mr = 4;

  // Macro-tiling for L2/L3 Caches
  for (int ib = 0; ib < M; ib += Mc) {
    int i_max = std::min(ib + Mc, M);
    for (int kb = 0; kb < K; kb += Kc) {
      int k_max = std::min(kb + Kc, K);
      int kc_actual = k_max - kb;
      for (int jb = 0; jb < N; jb += Nc) {
        int j_max = std::min(jb + Nc, N);

        // Micro-tiling for L1 Cache and Registers
        for (int i2 = ib; i2 < i_max; i2 += Mr) {
          int mr = std::min(Mr, i_max - i2);
          for (int j2 = jb; j2 < j_max; j2 += Nr) {
            int nr = std::min(Nr, j_max - j2);

            // Fast path: fully fits the register blocking dimensions
            if (mr == Mr && nr == Nr) {
              const double *a = &A[i2 * K + kb];
              const double *mb = &B[kb * N + j2];
              double *c = &C[i2 * N + j2];
              neon_block<Nr, Mr>(c, a, mb, N, K, kc_actual);
            } else {
              // Edge case handling for boundaries that are not multiples of Mr/Nr
              for (int i = 0; i < mr; ++i) {
                for (int k = 0; k < kc_actual; ++k) {
                  double a_val = A[(i2 + i) * K + kb + k];
                  for (int j = 0; j < nr; ++j) {
                    C[(i2 + i) * N + j2 + j] +=
                        a_val * B[(kb + k) * N + j2 + j];
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
