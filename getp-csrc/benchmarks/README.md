# GEMM Benchmark (gemm_mfma_v2 vs. rocBLAS)

This folder contains a standalone CMake project for benchmarking the custom `gemm_mfma_v2` kernel against rocBLAS.

---

## Prerequisites

- ROCm stack with `hipcc`, rocBLAS, and HIP runtime headers/libraries available (`module load rocm` on most clusters).
- CMake ≥ 3.21.
- A GPU that supports MFMA instructions (e.g. MI200 series, gfx90a).

---

## Configure & Build


### Makefile Build (no CMake available)

```
cd getp-csrc/benchmarks
make                 # builds with rocBLAS, requires rocblas on LD_LIBRARY_PATH
make WITH_ROCBLAS=0  # builds without rocBLAS comparison
```

Set `HIPCC=/path/to/hipcc` if it is not on `PATH`. The binary is written next to the sources as `./gemm_bench`.



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
  * If your rocBLAS build lacks BF16 support, the benchmark automatically falls back to FP32 `sgemm` (it will annotate the output and note that weights were promoted). Enable the BF16 path by setting `GEMM_BENCH_TRY_BF16=1` before running.
- A separator line for readability.

All HIP and rocBLAS calls are wrapped in macros that report the failing API call, file, line, and error code before exiting. Improper CLI usage (e.g. missing multiples of three arguments) also results in a usage message and non-zero exit status.

---

## Customising the Kernel Launch

`gemm_mfma_v2_kernel.hpp` exposes `launch_gemm_mfma_v2`, which is configured with the same tile/block setup that the main application uses (`BM=64`, `BN=128`, `BK=32`, `TM=TN=32`, `BLOCK_THREADS=512`). Adjust these constants if you need to explore alternative tilings; make sure the blockDim/launch configuration stays consistent across your application and benchmark.


## Flash Attention Benchmark

`flash_attn_bench` exercises `single_query_attn_flash_batched` from `src/flash_attn_hip.cpp` with
synthetic inputs. It allocates bf16 KV caches plus scratch buffers that mirror the runtime layout
and reports average kernel latency together with a rough FLOP estimate.

### Build

```
# Using the Makefile
make flash_attn_bench

# Using CMake (same flags as gemm_bench)
cmake -S getp-csrc/benchmarks -B build/fa \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_HIP_COMPILER=$(which hipcc) \
      -DHIP_HIPCC_FLAGS="--offload-arch=gfx90a -O3"
cmake --build build/fa --target flash_attn_bench
```

### Run

```
./flash_attn_bench \
  --batch 512 --nq 64 --head-dim 64 --seq-len 1024 --kv-mul 8 \
  --sliding-window 0 --layers 1 --warmup 5 --iters 100 \
  --mode compare
```

Flags let you sweep batch size, number of query heads, sequence length, KV grouping, sliding
window, warm-up, and iteration counts. `--mode baseline` (default) times the proven kernel,
`--mode workspace` runs an alternative implementation only, and `--mode compare` executes both in
sequence, reporting speedup plus `max_abs_diff`/`l2_rel_error` against the baseline checksum. The
workspace hook lives in `flash_attn_workspace.cpp`; by default it forwards to the baseline, so you
can drop in experimental kernels there without touching the reference path.
