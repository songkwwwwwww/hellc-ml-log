#include "matmul.h"
#include "matrix_utils.h"
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using namespace matmul;

void PrintUsage(char *prog) {
  std::cout << "Usage: " << prog << " <type> <size> <iters>\n";
  std::cout << "  type  : naive, tiled, simd, ref\n";
  std::cout << "  size  : matrix dimension (N x N)\n";
  std::cout << "  iters : number of repetitions for profiling\n";
}

int main(int argc, char **argv) {
  if (argc < 4) {
    PrintUsage(argv[0]);
    return 1;
  }

  std::string type = argv[1];
  int size = std::stoi(argv[2]);
  int rows = size;
  int columns = size;
  int inners = size;
  int iters = std::stoi(argv[3]);

  MatmulFunc func = nullptr;
  if (type == "naive")
    func = Naive;
  else if (type == "tiled")
    func = Tiled;
  else if (type == "simd")
    func = Simd;
  else if (type == "ref")
    func = Reference;
  else {
    std::cerr << "Unknown type: " << type << std::endl;
    return 1;
  }

  // Allocate 64-byte aligned memory for optimal vector instruction (SIMD)
  // performance
  double *A = AllocateAligned(rows * inners);
  double *B = AllocateAligned(inners * columns);
  double *C = AllocateAligned(rows * columns);
  InitializeRandom(A, rows * inners);
  InitializeRandom(B, inners * columns);

  std::cout << "Profiling " << type << " (Size: " << rows << "x" << columns
            << ", Iters: " << iters << ")..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  // Iteration loop for profiling
  // Runs the selected matrix multiplication function 'iters' times
  for (int iteration = 0; iteration < iters; ++iteration) {
    func(A, B, C, rows, columns, inners);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  std::cout << "Total time: " << diff.count() << " s" << std::endl;
  std::cout << "Avg time per op: " << diff.count() / iters << " s" << std::endl;

  FreeAligned(A);
  FreeAligned(B);
  FreeAligned(C);

  return 0;
}
