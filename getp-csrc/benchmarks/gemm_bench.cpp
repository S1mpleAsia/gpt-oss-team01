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

#include "gemm_mfma_v2_kernel.hpp"

#define CHECK_HIP(cmd)                                                                             \
  do {                                                                                             \
    hipError_t _err = (cmd);                                                                       \
    if (_err != hipSuccess) {                                                                      \
      std::fprintf(stderr, "HIP error %s at %s:%d -> %s\n", #cmd, __FILE__, __LINE__,             \
                    hipGetErrorString(_err));                                                      \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

#if WITH_ROCBLAS
#define CHECK_ROCBLAS(cmd)                                                                          \
  do {                                                                                             \
    rocblas_status _st = (cmd);                                                                    \
    if (_st != rocblas_status_success) {                                                           \
      std::fprintf(stderr, "rocBLAS error %s at %s:%d -> status %d\n", #cmd, __FILE__, __LINE__,   \
                    static_cast<int>(_st));                                                        \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)
#endif

struct GemmProblem {
  int M;
  int N;
  int K;
};

static bf16 float_to_bf16(float v) {
  return __float2bfloat16(v);
}

static void fill_random_normal(std::vector<float> &dst, float mean, float stddev) {
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(mean, stddev);
  for (float &v : dst) {
    v = dist(gen);
  }
}

static float compute_max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
  float max_diff = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
  }
  return max_diff;
}

static float compute_l2_rel_error(const std::vector<float> &a, const std::vector<float> &b) {
  double num = 0.0;
  double denom = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += diff * diff;
    denom += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return denom > 0.0 ? static_cast<float>(std::sqrt(num / denom)) : 0.0f;
}

static float benchmark_custom_kernel(const float *dA, const bf16 *dB, float *dC, int M, int N,
                                     int K, hipStream_t stream, int iters) {
  CHECK_HIP(hipMemsetAsync(dC, 0, sizeof(float) * static_cast<size_t>(M) * N, stream));

  for (int i = 0; i < 3; ++i) {
    launch_gemm_mfma_v2(dA, dB, dC, M, N, K, stream);
  }
  CHECK_HIP(hipStreamSynchronize(stream));

  hipEvent_t start, stop;
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));

  CHECK_HIP(hipEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    launch_gemm_mfma_v2(dA, dB, dC, M, N, K, stream);
  }
  CHECK_HIP(hipEventRecord(stop, stream));
  CHECK_HIP(hipEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start, stop));

  CHECK_HIP(hipEventDestroy(start));
  CHECK_HIP(hipEventDestroy(stop));

  return elapsed_ms / static_cast<float>(iters);
}

#if WITH_ROCBLAS
static float benchmark_rocblas(const float *dA, const bf16 *dB, float *dC, int M, int N, int K,
                               hipStream_t stream, rocblas_handle handle, int iters) {
  CHECK_HIP(hipMemsetAsync(dC, 0, sizeof(float) * static_cast<size_t>(M) * N, stream));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (int i = 0; i < 3; ++i) {
    CHECK_ROCBLAS(rocblas_gemm_ex(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                  N, M, K, &alpha, dB, rocblas_datatype_bf16_r, N, dA,
                                  rocblas_datatype_f32_r, K, &beta, dC, rocblas_datatype_f32_r, N,
                                  dC, rocblas_datatype_f32_r, N, rocblas_datatype_f32_r,
                                  rocblas_gemm_algo_standard, 0, 0));
  }
  CHECK_HIP(hipStreamSynchronize(stream));

  hipEvent_t start, stop;
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));

  CHECK_HIP(hipEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    CHECK_ROCBLAS(rocblas_gemm_ex(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                  N, M, K, &alpha, dB, rocblas_datatype_bf16_r, N, dA,
                                  rocblas_datatype_f32_r, K, &beta, dC, rocblas_datatype_f32_r, N,
                                  dC, rocblas_datatype_f32_r, N, rocblas_datatype_f32_r,
                                  rocblas_gemm_algo_standard, 0, 0));
  }
  CHECK_HIP(hipEventRecord(stop, stream));
  CHECK_HIP(hipEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start, stop));

  CHECK_HIP(hipEventDestroy(start));
  CHECK_HIP(hipEventDestroy(stop));

  return elapsed_ms / static_cast<float>(iters);
}
#endif

int main(int argc, char **argv) {
  hipStream_t stream = nullptr;
  CHECK_HIP(hipStreamCreate(&stream));

#if WITH_ROCBLAS
  rocblas_handle handle = nullptr;
  CHECK_ROCBLAS(rocblas_create_handle(&handle));
  CHECK_ROCBLAS(rocblas_set_stream(handle, stream));
#endif

  std::vector<GemmProblem> problems;
  problems.reserve((argc > 1 ? argc - 1 : 0) / 3 + 1);

  if ((argc - 1) % 3 != 0) {
    std::cerr << "Usage: " << argv[0] << " [M N K]..." << std::endl;
    std::cerr << "  Provide triplets of matrix sizes; defaults to 2048 5760 2880 when none given."
              << std::endl;
    return EXIT_FAILURE;
  }

  if (argc == 1) {
    problems.push_back({2048, 5760, 2880});
  } else {
    for (int i = 1; i < argc; i += 3) {
      GemmProblem problem{};
      problem.M = std::atoi(argv[i]);
      problem.N = std::atoi(argv[i + 1]);
      problem.K = std::atoi(argv[i + 2]);
      if (problem.M <= 0 || problem.N <= 0 || problem.K <= 0) {
        std::cerr << "Invalid dimensions: " << problem.M << ", " << problem.N << ", " << problem.K
                  << std::endl;
        return EXIT_FAILURE;
      }
      problems.push_back(problem);
    }
  }

  const int iters = 50;

  for (const auto &problem : problems) {
    const int M = problem.M;
    const int N = problem.N;
    const int K = problem.K;

    const size_t size_a = static_cast<size_t>(M) * K;
    const size_t size_b = static_cast<size_t>(K) * N;
    const size_t size_c = static_cast<size_t>(M) * N;

    std::vector<float> hA(size_a);
    std::vector<float> hB_float(size_b);
    std::vector<bf16> hB(size_b);
    std::vector<float> hC(size_c);
#if WITH_ROCBLAS
    std::vector<float> hC_ref(size_c);
#endif

    fill_random_normal(hA, 0.0f, 1.0f);
    fill_random_normal(hB_float, 0.0f, 1.0f);

    for (size_t i = 0; i < size_b; ++i) {
      hB[i] = float_to_bf16(hB_float[i]);
    }

    float *dA = nullptr;
    bf16 *dB = nullptr;
    float *dC_custom = nullptr;
#if WITH_ROCBLAS
    float *dC_rocblas = nullptr;
#endif

    CHECK_HIP(hipMalloc(&dA, sizeof(float) * size_a));
    CHECK_HIP(hipMalloc(&dB, sizeof(bf16) * size_b));
    CHECK_HIP(hipMalloc(&dC_custom, sizeof(float) * size_c));
#if WITH_ROCBLAS
    CHECK_HIP(hipMalloc(&dC_rocblas, sizeof(float) * size_c));
#endif

    CHECK_HIP(hipMemcpyAsync(dA, hA.data(), sizeof(float) * size_a, hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipMemcpyAsync(dB, hB.data(), sizeof(bf16) * size_b, hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipStreamSynchronize(stream));

    float ms_custom = benchmark_custom_kernel(dA, dB, dC_custom, M, N, K, stream, iters);

#if WITH_ROCBLAS
    float ms_rocblas = benchmark_rocblas(dA, dB, dC_rocblas, M, N, K, stream, handle, iters);
    CHECK_HIP(hipMemcpy(hC.data(), dC_custom, sizeof(float) * size_c, hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(hC_ref.data(), dC_rocblas, sizeof(float) * size_c, hipMemcpyDeviceToHost));

    float max_diff = compute_max_abs_diff(hC, hC_ref);
    float rel_l2 = compute_l2_rel_error(hC, hC_ref);
#endif

    const double gflops_total = (2.0 * static_cast<double>(M) * N * K) * 1e-6;
    double custom_gflops = gflops_total / static_cast<double>(ms_custom);

    std::cout << "Problem M=" << M << ", N=" << N << ", K=" << K << '\n';
    std::cout << "  gemm_mfma_v2  : " << ms_custom << " ms  (" << custom_gflops << " GFLOP/s)" << '\n';

#if WITH_ROCBLAS
    double rocblas_gflops = gflops_total / static_cast<double>(ms_rocblas);
    std::cout << "  rocBLAS       : " << ms_rocblas << " ms  (" << rocblas_gflops << " GFLOP/s)" << '\n';
    std::cout << "  max|diff|     : " << max_diff << '\n';
    std::cout << "  rel L2 error  : " << rel_l2 << '\n';
#else
    std::cout << "  rocBLAS       : disabled (configure WITH_ROCBLAS=ON to compare)" << '\n';
#endif
    std::cout << std::string(60, '-') << '\n';

    CHECK_HIP(hipFree(dA));
    CHECK_HIP(hipFree(dB));
    CHECK_HIP(hipFree(dC_custom));
#if WITH_ROCBLAS
    CHECK_HIP(hipFree(dC_rocblas));
#endif
  }

#if WITH_ROCBLAS
  CHECK_ROCBLAS(rocblas_destroy_handle(handle));
#endif
  CHECK_HIP(hipStreamDestroy(stream));

  return EXIT_SUCCESS;
}

