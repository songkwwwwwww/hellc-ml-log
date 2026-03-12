# hellc-ml-study

ML Infra 및 핵심 알고리즘을 학습하고 구현하는 프로젝트 로그입니다.

## Project Structure

이 프로젝트는 Bazel을 사용하는 Monorepo 구성입니다. 각 주차별 학습 내용은 `study-log/` 디렉토리에 서브 프로젝트 형태로 저장됩니다.

- `study-log/`: 주차별 학습 프로젝트 (e.g., `001-matmul`, `002-conv`)
- `third_party/`: 외부 의존성 및 참고 프로젝트 (e.g., `inference_engine`)

## Prerequisites

- [Bazel (Bazelisk 권장)](https://github.com/bazelbuild/bazelisk)
- C++17 지원 컴파일러 (gcc, clang 등)
- Python 3

## How to Run (e.g., 001-matmul)

### Correctness Test (Google Test)
모든 구현이 Reference(Accelerate)와 동일한 결과를 내는지 확인합니다.
```bash
bazel test //study-log/001-matmul:matmul_test
```

### Performance Benchmark 들(Google Benchmark)
다양한 행렬 크기에 대해 각 구현의 성능(GFLOPS)을 측정합니다.
```bash
bazel run -c opt //study-log/001-matmul:matmul_bench
```

---

## Reference

- [ml-roadmap, Logan Thorneloe, Google, ML Infra](https://github.com/loganthorneloe/ml-roadmap?tab=readme-ov-file)
