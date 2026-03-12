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
 * Binds the current thread to a specific CPU core. This prevents the OS scheduler
 * from migrating the thread to a different core, which would cause L1/L2 cache
 * invalidation and severely degrade performance.
 */
void pin_thread_to_core(int core_id) {
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

template <int Nr, int Mr>
inline void neon_block(double *c, const double *a, const double *mb, int N,
                       int K, int kc_actual) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  constexpr int neon_doubles = 128 / (sizeof(double) * 8); // Equals 2

  for (int i = 0; i < Mr; ++i, c += N, a += K) {
    const double *b = mb;
    for (int k = 0; k < kc_actual; ++k, b += N) {
      float64x2_t a_reg = vmovq_n_f64(a[k]);
      for (int j = 0; j < Nr; j += neon_doubles) {
        float64x2_t b_reg = vld1q_f64(&b[j]);
        float64x2_t c_reg = vld1q_f64(&c[j]);
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
 * @brief Multi-threaded Matrix Multiplication using OpenMP
 * 
 * By dividing the matrix blocks among multiple CPU cores, we can parallelize
 * the computation. OpenMP makes it easy to distribute loop iterations.
 */
void omp_thread(const double *A, const double *B, double *C, int M, int N,
                int K) {
  constexpr int Mc = 180, Nc = 96, Kc = 240, Nr = 8, Mr = 4;

#pragma omp parallel
  {
    // Pin each thread to a specific core to maximize cache reuse
    int thread_id = omp_get_thread_num();
    pin_thread_to_core(thread_id);

    // Apply parallelization to multiple loops (top-level ib and jb)
    // collapse(2): Fuses the 'ib' and 'jb' loops into a single larger loop 
    // before distributing it among threads. This ensures a more balanced workload.
    // schedule(dynamic): Threads dynamically request more chunks of work as they 
    // finish, handling cases where some blocks take longer than others.
#pragma omp for collapse(2) schedule(dynamic)
    for (int ib = 0; ib < M; ib += Mc) {
      for (int jb = 0; jb < N; jb += Nc) {
        int i_max = std::min(ib + Mc, M);
        int j_max = std::min(jb + Nc, N);

        for (int kb = 0; kb < K; kb += Kc) {
          int k_max = std::min(kb + Kc, K);
          int kc_actual = k_max - kb;

          // Micro-kernel loops
          for (int i2 = ib; i2 < i_max; i2 += Mr) {
            int mr = std::min(Mr, i_max - i2);
            for (int j2 = jb; j2 < j_max; j2 += Nr) {
              int nr = std::min(Nr, j_max - j2);

              if (mr == Mr && nr == Nr) {
                const double *a = &A[i2 * K + kb];
                const double *mb = &B[kb * N + j2];
                double *c = &C[i2 * N + j2];
                neon_block<Nr, Mr>(c, a, mb, N, K, kc_actual);
              } else {
                // Edge case handling for non-multiple boundary blocks
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
}

} // namespace matmul