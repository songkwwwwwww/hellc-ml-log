#include "matmul.h"
#include "matrix_utils.h"
#include <gtest/gtest.h>

using namespace matmul;

/**
 * @brief Test fixture for Matrix Multiplication implementations
 * 
 * Sets up the required matrices before each test and cleans them up afterwards.
 * Uses a reference BLAS implementation to verify the correctness of our custom
 * matrix multiplication algorithms.
 */
class MatmulTest : public ::testing::Test {
protected:
  void SetUp() override {
    M = 128;
    N = 128;
    K = 128;
    
    // Allocate memory with proper alignment for SIMD testing
    A = allocate_aligned(M * K);
    B = allocate_aligned(K * N);
    C = allocate_aligned(M * N);
    refC = allocate_aligned(M * N);

    // Initialize inputs with random values
    initialize_random(A, M * K);
    initialize_random(B, K * N);

    // Generate the "golden" reference results using BLAS
    reference(A, B, refC, M, N, K);
  }

  void TearDown() override {
    free_aligned(A);
    free_aligned(B);
    free_aligned(C);
    free_aligned(refC);
  }

  int M, N, K;
  double *A, *B, *C, *refC;
};

// Test definitions for each optimization technique

TEST_F(MatmulTest, NaiveCorrectness) {
  naive(A, B, C, M, N, K);
  EXPECT_TRUE(verify_results(C, refC, M * N));
}

TEST_F(MatmulTest, LoopReorderCorrectness) {
  loop_reorder(A, B, C, M, N, K);
  EXPECT_TRUE(verify_results(C, refC, M * N));
}

TEST_F(MatmulTest, TiledCorrectness) {
  tiled(A, B, C, M, N, K);
  EXPECT_TRUE(verify_results(C, refC, M * N));
}

TEST_F(MatmulTest, SIMDCorrectness) {
  simd(A, B, C, M, N, K);
  EXPECT_TRUE(verify_results(C, refC, M * N));
}

TEST_F(MatmulTest, CacheAwareCorrectness) {
  cache_aware(A, B, C, M, N, K);
  EXPECT_TRUE(verify_results(C, refC, M * N));
}

TEST_F(MatmulTest, OmpThreadCorrectness) {
  omp_thread(A, B, C, M, N, K);
  EXPECT_TRUE(verify_results(C, refC, M * N));
}

TEST_F(MatmulTest, PackedCorrectness) {
  packed(A, B, C, M, N, K);
  EXPECT_TRUE(verify_results(C, refC, M * N));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
