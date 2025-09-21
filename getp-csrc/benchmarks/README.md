# GEMM Benchmark (gemm_mfma_v2 vs. rocBLAS)

This folder contains a standalone CMake project for benchmarking the custom `gemm_mfma_v2` kernel against rocBLAS.

---

## Prerequisites

- ROCm stack with `hipcc`, rocBLAS, and HIP runtime headers/libraries available (`module load rocm` on most clusters).
- CMake ≥ 3.21.
- A GPU that supports MFMA instructions (e.g. MI200 series, gfx90a).

---

## Configure & Build

The project mirrors the flags used by the main repo’s `runomp` target: it defaults to `hipcc`, adds `-O3`, and passes `--offload-arch=gfx90a`.

```bash
cmake -S getp-csrc/benchmarks -B build/bench \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_HIP_COMPILER=$(which hipcc) \
      -DHIP_HIPCC_FLAGS="--offload-arch=gfx90a -O3" \
      -DWITH_ROCBLAS=ON         # set OFF to skip rocBLAS
cmake --build build/bench
```

If CMake cannot locate rocBLAS automatically, point it to the installation:

```bash
cmake -S getp-csrc/benchmarks -B build/bench \
      -Drocblas_DIR=/opt/rocm/rocblas/lib/cmake/rocblas
```

To build without rocBLAS comparison:

```bash
cmake -S getp-csrc/benchmarks -B build/bench-nn \
      -DWITH_ROCBLAS=OFF
cmake --build build/bench-nn
```

---

## Running the Benchmark

`gemm_bench` accepts any number of matrix-size triplets `(M N K)` on the command line. Without arguments it defaults to `2048 5760 2880`.

```bash
# Single case
gemm_bench 1024 1024 1024

# Multiple cases in one launch
gemm_bench 1024 1024 1024 2048 5760 2880 4096 4096 4096
```

On a managed cluster, keep using `srun`/`sbatch` to allocate a GPU:

```bash
srun -N 1 --gres=gpu:1 ./build/bench/gemm_bench 1024 1024 1024
```

---

## Interpreting Output

For each triplet the executable prints:

- The matrix shape.
- Average runtime and GFLOP/s for `gemm_mfma_v2`.
- rocBLAS runtime/GFLOP/s plus max absolute and relative L2 differences (when `WITH_ROCBLAS=ON`).
- A separator line for readability.

All HIP and rocBLAS calls are wrapped in macros that report the failing API call, file, line, and error code before exiting. Improper CLI usage (e.g. missing multiples of three arguments) also results in a usage message and non-zero exit status.

---

## Customising the Kernel Launch

`gemm_mfma_v2_kernel.hpp` exposes `launch_gemm_mfma_v2`, which is configured with the same tile/block setup that the main application uses (`BM=64`, `BN=128`, `BK=32`, `TM=TN=32`, `BLOCK_THREADS=512`). Adjust these constants if you need to explore alternative tilings; make sure the blockDim/launch configuration stays consistent across your application and benchmark.

