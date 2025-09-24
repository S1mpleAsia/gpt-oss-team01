// gemm_bench.cpp — compare custom GEMM (A=f32, B=bf16) vs rocBLAS (FP32 SGEMM)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#ifndef WITH_ROCBLAS
#define WITH_ROCBLAS 1
#endif

#if WITH_ROCBLAS
#include <rocblas/rocblas.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

// === Your kernel header (expects launch_gemm_mfma_v2) ===
#include "gemm_mfma_v2_kernel.hpp"

using bf16 = hip_bfloat16;

// ----------------- Error-check helpers -----------------
#define CHECK_HIP(cmd)                                                                             \
  do {                                                                                             \
    hipError_t _e = (cmd);                                                                         \
    if (_e != hipSuccess) {                                                                        \
      std::fprintf(stderr, "HIP error %s @ %s:%d -> %s\n", #cmd, __FILE__, __LINE__,               \
                   hipGetErrorString(_e));                                                         \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

#if WITH_ROCBLAS
#define CHECK_ROCBLAS(cmd)                                                                         \
  do {                                                                                             \
    rocblas_status _s = (cmd);                                                                     \
    if (_s != rocblas_status_success) {                                                            \
      std::fprintf(stderr, "rocBLAS error %s @ %s:%d -> status %d\n", #cmd, __FILE__, __LINE__,    \
                   (int)_s);                                                                       \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)
#endif

// ----------------- Utils -----------------
struct GemmProblem {
  int M, N, K;
};

static void fill_random_normal(std::vector<float> &dst, float mean, float stddev) {
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(mean, stddev);
  for (float &v : dst)
    v = dist(gen);
}

static inline bf16 f32_to_bf16(float v) {
  return bf16(v);
}

static float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
  float m = 0.f;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}
static float rel_l2_err(const std::vector<float> &a, const std::vector<float> &b) {
  long double num = 0.0L, den = 0.0L;
  for (size_t i = 0; i < a.size(); ++i) {
    long double d = (long double)a[i] - b[i];
    num += d * d;
    den += (long double)b[i] * b[i];
  }
  return den > 0 ? (float)std::sqrt((double)(num / den)) : 0.f;
}

// ----------------- bf16 -> f32 conversion kernel -----------------
__global__ void bf16_to_fp32_kernel(const bf16 *__restrict__ bf16_in, float *__restrict__ fp32_out,
                                    size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = gridDim.x * blockDim.x;
  for (; i < n; i += s)
    fp32_out[i] = (float)bf16_in[i];
}

// ----------------- Bench: your custom kernel -----------------
static float bench_custom(const float *dA_f32, const bf16 *dB_bf16, float *dC, int M, int N, int K,
                          hipStream_t stream, int iters) {
  CHECK_HIP(hipMemsetAsync(dC, 0, sizeof(float) * (size_t)M * N, stream));
  // warmup
  for (int i = 0; i < 3; ++i)
    launch_gemm_mfma_v2(dA_f32, dB_bf16, dC, M, N, K, stream);
  CHECK_HIP(hipStreamSynchronize(stream));
  // timed
  hipEvent_t start, stop;
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));
  CHECK_HIP(hipEventRecord(start, stream));
  for (int i = 0; i < iters; ++i)
    launch_gemm_mfma_v2(dA_f32, dB_bf16, dC, M, N, K, stream);
  CHECK_HIP(hipEventRecord(stop, stream));
  CHECK_HIP(hipEventSynchronize(stop));
  float ms = 0.f;
  CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
  CHECK_HIP(hipEventDestroy(start));
  CHECK_HIP(hipEventDestroy(stop));
  return ms / (float)iters;
}

#if WITH_ROCBLAS
// ----------------- Bench: rocBLAS FP32 SGEMM -----------------
// Row-major mapping (safe):
//   C_r[M,N] = A_r[M,K] * B_r[K,N]
// Call as column-major NN with swapped m/n and operands:
//   m=N, n=M, k=K
//   A := B_r (aliased as col-major N×K), lda=N
//   B := A_r (aliased as col-major K×M), ldb=K
//   C ldc=N (interprets C as col-major N×M)
static float bench_rocblas_sgemm_rowmajor(const float *dA_r, const bf16 *dB_r_bf16, float *dC_r,
                                          int M, int N, int K, hipStream_t stream,
                                          rocblas_handle handle, int iters) {
  CHECK_HIP(hipMemsetAsync(dC_r, 0, sizeof(float) * (size_t)M * N, stream));

  // Convert B (bf16) -> f32 once on device
  float *dB_r_f32 = nullptr;
  const size_t size_b = (size_t)K * N;
  CHECK_HIP(hipMalloc(&dB_r_f32, sizeof(float) * size_b));
  const int tpb = 256, blocks = (int)((size_b + tpb - 1) / tpb);
  hipLaunchKernelGGL(bf16_to_fp32_kernel, dim3(blocks), dim3(tpb), 0, stream, dB_r_bf16, dB_r_f32,
                     size_b);
  CHECK_HIP(hipStreamSynchronize(stream));

  const float alpha = 1.f, beta = 0.f;
  auto sgemm_nn = [&]() {
    return rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                         /*m=*/N, /*n=*/M, /*k=*/K, &alpha,
                         /*A=*/dB_r_f32, /*lda=*/N,  // B_r
                         /*B=*/dA_r, /*ldb=*/K,      // A_r
                         &beta,
                         /*C=*/dC_r, /*ldc=*/N);  // C as col-major N×M (row-major M×N)
  };

  // warmup
  for (int i = 0; i < 3; ++i)
    CHECK_ROCBLAS(sgemm_nn());
  CHECK_HIP(hipStreamSynchronize(stream));

  // timed
  hipEvent_t start, stop;
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));
  CHECK_HIP(hipEventRecord(start, stream));
  for (int i = 0; i < iters; ++i)
    CHECK_ROCBLAS(sgemm_nn());
  CHECK_HIP(hipEventRecord(stop, stream));
  CHECK_HIP(hipEventSynchronize(stop));
  float ms = 0.f;
  CHECK_HIP(hipEventElapsedTime(&ms, start, stop));

  CHECK_HIP(hipEventDestroy(start));
  CHECK_HIP(hipEventDestroy(stop));
  CHECK_HIP(hipFree(dB_r_f32));
  return ms / (float)iters;
}
#endif

// ----------------- main -----------------
int main(int argc, char **argv) {
  hipStream_t stream = nullptr;
  CHECK_HIP(hipStreamCreate(&stream));

#if WITH_ROCBLAS
  rocblas_handle handle = nullptr;
  CHECK_ROCBLAS(rocblas_create_handle(&handle));
  CHECK_ROCBLAS(rocblas_set_stream(handle, stream));
#endif

  std::vector<GemmProblem> problems;
  if ((argc - 1) % 3 != 0) {
    std::cerr << "Usage: " << argv[0] << " [M N K]...\n"
              << "  Default: 2048 5760 2880\n";
    return EXIT_FAILURE;
  }
  if (argc == 1)
    problems.push_back({2048, 5760, 2880});
  else {
    for (int i = 1; i < argc; i += 3) {
      GemmProblem p{std::atoi(argv[i]), std::atoi(argv[i + 1]), std::atoi(argv[i + 2])};
      if (p.M <= 0 || p.N <= 0 || p.K <= 0) {
        std::cerr << "Invalid dims\n";
        return EXIT_FAILURE;
      }
      problems.push_back(p);
    }
  }

  const int iters = 1000;

  for (auto &pb : problems) {
    const int M = pb.M, N = pb.N, K = pb.K;
    const size_t sizeA = (size_t)M * K, sizeB = (size_t)K * N, sizeC = (size_t)M * N;

    std::cout << "Problem M=" << M << ", N=" << N << ", K=" << K << "\n";

    // Host
    std::vector<float> hA(sizeA), hB_f32(sizeB);
    std::vector<bf16> hB_bf16(sizeB);
    fill_random_normal(hA, 0.0f, 1.0f);
    fill_random_normal(hB_f32, 0.0f, 1.0f);
    for (size_t i = 0; i < sizeB; ++i)
      hB_bf16[i] = f32_to_bf16(hB_f32[i]);

    // Device
    float *dA = nullptr, *dC_custom = nullptr, *dC_rocblas = nullptr;
    bf16 *dB = nullptr;
    CHECK_HIP(hipMalloc(&dA, sizeof(float) * sizeA));
    CHECK_HIP(hipMalloc(&dB, sizeof(bf16) * sizeB));
    CHECK_HIP(hipMalloc(&dC_custom, sizeof(float) * sizeC));
    CHECK_HIP(hipMalloc(&dC_rocblas, sizeof(float) * sizeC));
    CHECK_HIP(hipMemcpyAsync(dA, hA.data(), sizeof(float) * sizeA, hipMemcpyHostToDevice, stream));
    CHECK_HIP(
      hipMemcpyAsync(dB, hB_bf16.data(), sizeof(bf16) * sizeB, hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipStreamSynchronize(stream));

    // --- custom
    float ms_custom = bench_custom(dA, dB, dC_custom, M, N, K, stream, iters);

    // --- rocBLAS (FP32)
    float ms_rocblas = NAN;
#if WITH_ROCBLAS
    ms_rocblas = bench_rocblas_sgemm_rowmajor(dA, dB, dC_rocblas, M, N, K, stream, handle, iters);
#endif

    // Copy back & compare
    std::vector<float> hC_custom(sizeC), hC_rocblas(sizeC);
    CHECK_HIP(hipMemcpy(hC_custom.data(), dC_custom, sizeof(float) * sizeC, hipMemcpyDeviceToHost));
#if WITH_ROCBLAS
    CHECK_HIP(
      hipMemcpy(hC_rocblas.data(), dC_rocblas, sizeof(float) * sizeC, hipMemcpyDeviceToHost));
#endif

    const double gflops = (2.0 * (double)M * N * K) * 1e-6;
    std::cout << "  gemm_mfma_v2            : " << ms_custom << " ms  (" << (gflops / ms_custom)
              << " GFLOP/s)\n";
#if WITH_ROCBLAS
    std::cout << "  rocBLAS (SGEMM NN map)  : " << ms_rocblas << " ms  (" << (gflops / ms_rocblas)
              << " GFLOP/s)\n";
    std::cout << "  diff custom vs rocBLAS  : max|diff|=" << max_abs_diff(hC_custom, hC_rocblas)
              << ", rel L2=" << rel_l2_err(hC_custom, hC_rocblas) << "\n";
#else
    std::cout << "  rocBLAS                 : disabled (compile WITH_ROCBLAS=1)\n";
#endif
    std::cout << std::string(60, '-') << "\n";

    CHECK_HIP(hipFree(dA));
    CHECK_HIP(hipFree(dB));
    CHECK_HIP(hipFree(dC_custom));
    CHECK_HIP(hipFree(dC_rocblas));
  }

#if WITH_ROCBLAS
  CHECK_ROCBLAS(rocblas_destroy_handle(handle));
#endif
  CHECK_HIP(hipStreamDestroy(stream));
  return EXIT_SUCCESS;
}
