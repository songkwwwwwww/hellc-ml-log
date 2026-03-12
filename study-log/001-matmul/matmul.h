#ifndef MATMUL_H
#define MATMUL_H

namespace matmul {

/**
 * @brief Common interface for Matrix Multiplication: C = A * B
 * 
 * This header defines the standard interface for various matrix multiplication
 * implementations. Matrix multiplication is a fundamental operation in machine
 * learning and high-performance computing.
 * 
 * Matrix Details:
 * - A: M x K matrix (stored in row-major order)
 * - B: K x N matrix (stored in row-major order)
 * - C: M x N matrix (stored in row-major order)
 * 
 * Row-major order means that elements of a row are stored in contiguous memory
 * locations. For example, A[i][j] is accessed as A[i * K + j].
 */
typedef void (*MatmulFunc)(const double *A, const double *B, double *C, int M,
                           int N, int K);

// Implementations of matrix multiplication with various optimization techniques
void naive(const double *A, const double *B, double *C, int M, int N, int K);
void loop_reorder(const double *A, const double *B, double *C, int M, int N,
                  int K);
void tiled(const double *A, const double *B, double *C, int M, int N, int K);
void simd(const double *A, const double *B, double *C, int M, int N, int K);
void cache_aware(const double *A, const double *B, double *C, int M, int N, int K);
void omp_thread(const double *A, const double *B, double *C, int M, int N, int K);
void packed(const double *A, const double *B, double *C, int M, int N, int K);
void reference(const double *A, const double *B, double *C, int M, int N,
               int K);

} // namespace matmul

#endif // MATMUL_H
