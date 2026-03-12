#include "matmul.h"

namespace matmul {

void NaiveRegisterAcc(const double *A, const double *B, double *C, int rows,
                      int columns, int inners) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < columns; ++col) {
      double sum = 0.0;
      for (int inner = 0; inner < inners; ++inner) {
        sum += A[row * inners + inner] * B[inner * columns + col];
      }
      C[row * columns + col] = sum;
    }
  }
}

} // namespace matmul
