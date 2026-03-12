#include "matmul.h"
#include "matrix_utils.h"
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using namespace matmul;

void print_usage(char *prog) {
  std::cout << "Usage: " << prog << " <type> <size> <iters>\n";
  std::cout << "  type  : naive, tiled, simd, ref\n";
  std::cout << "  size  : matrix dimension (N x N)\n";
  std::cout << "  iters : number of repetitions for profiling\n";
}

int main(int argc, char **argv) {
  if (argc < 4) {
    print_usage(argv[0]);
    return 1;
  }

  std::string type = argv[1];
  int N = std::stoi(argv[2]);
  int iters = std::stoi(argv[3]);

  MatmulFunc func = nullptr;
  if (type == "naive")
    func = naive;
  else if (type == "tiled")
    func = tiled;
  else if (type == "simd")
    func = simd;
  else if (type == "ref")
    func = reference;
  else {
    std::cerr << "Unknown type: " << type << std::endl;
    return 1;
  }

  // Allocate 64-byte aligned memory for optimal vector instruction (SIMD) performance
  double *A = allocate_aligned(N * N);
  double *B = allocate_aligned(N * N);
  double *C = allocate_aligned(N * N);
  initialize_random(A, N * N);
  initialize_random(B, N * N);

  std::cout << "Profiling " << type << " (Size: " << N << "x" << N
            << ", Iters: " << iters << ")..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  // Iteration loop for profiling
  // Runs the selected matrix multiplication function 'iters' times
  for (int i = 0; i < iters; ++i) {
    func(A, B, C, N, N, N);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  std::cout << "Total time: " << diff.count() << " s" << std::endl;
  std::cout << "Avg time per op: " << diff.count() / iters << " s" << std::endl;

  free_aligned(A);
  free_aligned(B);
  free_aligned(C);

  return 0;
}
