# Project Context: hellc-ml-log

This file provides the foundational context, coding standards, and architectural guidelines for the `hellc-ml-log` project.

## Core Information

* **Purpose:** This project serves as a comprehensive study log and experimental repository for exploring Machine Learning (ML) infrastructure, performance optimization techniques (such as SIMD, OpenMP, and cache-aware programming), and foundational ML algorithms.
* **Repository Structure:** This is a monorepo built and managed with **Bazel**. 
* **Study Logs:** All weekly or topical study materials, experiments, and benchmark implementations are organized under the `study-log/` subdirectory (e.g., `study-log/001-matmul/`).

## Style Guides

* **Google C++ Style Guide:** All C++ code must strictly adhere to the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). This is the authoritative standard for naming conventions, formatting, and language feature usage.
* **Other Languages:** Follow the relevant Google style guides for any other languages introduced into the project:
  * [Python Style Guide](https://google.github.io/styleguide/pyguide.html)
  * [Shell Style Guide](https://google.github.io/styleguide/shellguide.html)

## Development Workflow

* **Build System:** The project uses **Bazel** (or `bazelisk`) for building, testing, and benchmarking. 
* **Testing & Benchmarking:** Implementations and optimizations must be verified with unit tests (using `googletest`) and their performance should be measured with benchmarks (using `google_benchmark`).
* **System Capabilities:** The project actively explores hardware-level optimizations (e.g., ARM NEON SIMD, CPU Cache-awareness, and OpenMP Multi-threading). Ensure that platform-specific flags, headers, and dependencies are correctly configured in the Bazel `BUILD` files and fallback mechanisms are provided for unsupported environments.