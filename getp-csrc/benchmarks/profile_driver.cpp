#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
#include "gemm_mfma_v2_nopf_profiled.hpp"

// Helpers
static void hipCheck(hipError_t e, const char* file, int line) {
  if (e != hipSuccess) {
    fprintf(stderr, "HIP error %s:%d: %s\n", file, line, hipGetErrorString(e));
    std::exit(1);
  }
}
#define HIP_CHECK(cmd) hipCheck((cmd), __FILE__, __LINE__)

int main() {
  // Example shape: replace with yours
  const int M = 512;
  const int N = 201088;
  const int K = 2880;

  // Allocate & init host buffers
  std::vector<float>  hA((size_t)M*K, 1.0f/3.0f);
  std::vector<bf16>   hB((size_t)K*N);
  std::vector<float>  hC((size_t)M*N, 0.0f);
  std::vector<bf16>   hBias(N);
  for (size_t i=0;i<hB.size();++i) hB[i] = (bf16)0.001f;
  for (size_t i=0;i<hBias.size();++i) hBias[i] = (bf16)0.0f;

  // Device alloc
  float *dA=nullptr, *dC=nullptr;
  bf16  *dB=nullptr, *dBias=nullptr;
  HIP_CHECK(hipMalloc(&dA, sizeof(float)*hA.size()));
  HIP_CHECK(hipMalloc(&dB, sizeof(bf16)*hB.size()));
  HIP_CHECK(hipMalloc(&dC, sizeof(float)*hC.size()));
  HIP_CHECK(hipMalloc(&dBias, sizeof(bf16)*hBias.size()));

  HIP_CHECK(hipMemcpy(dA, hA.data(), sizeof(float)*hA.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), sizeof(bf16)*hB.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dBias, hBias.data(), sizeof(bf16)*hBias.size(), hipMemcpyHostToDevice));

  // Timing accumulators on device
  unsigned long long *d_load_cycles=nullptr, *d_comp_cycles=nullptr;
  HIP_CHECK(hipMalloc(&d_load_cycles, sizeof(unsigned long long)));
  HIP_CHECK(hipMalloc(&d_comp_cycles, sizeof(unsigned long long)));
  HIP_CHECK(hipMemset(d_load_cycles, 0, sizeof(unsigned long long)));
  HIP_CHECK(hipMemset(d_comp_cycles, 0, sizeof(unsigned long long)));

  // Warmup
  launch_gemm_mfma_v2_nopf_profiled(dA, dB, dC, M, N, K, d_load_cycles, d_comp_cycles);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemset(d_load_cycles, 0, sizeof(unsigned long long)));
  HIP_CHECK(hipMemset(d_comp_cycles, 0, sizeof(unsigned long long)));

  // Events for overall kernel wall time (optional)
  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventRecord(e0));
  launch_gemm_mfma_v2_nopf_profiled(dA, dB, dC, M, N, K, d_load_cycles, d_comp_cycles);
  HIP_CHECK(hipEventRecord(e1));
  HIP_CHECK(hipEventSynchronize(e1));

  float ms_total=0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms_total, e0, e1));

  // Fetch cycle counters
  unsigned long long h_load_cycles=0, h_comp_cycles=0;
  HIP_CHECK(hipMemcpy(&h_load_cycles, d_load_cycles, sizeof(unsigned long long), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&h_comp_cycles, d_comp_cycles, sizeof(unsigned long long), hipMemcpyDeviceToHost));

  hipDeviceProp_t prop{};
  int dev=0;
  HIP_CHECK(hipGetDevice(&dev));
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  // hipDeviceProp_t.clockRate is in kHz
  const double gpu_clock_hz = (double)prop.clockRate * 1000.0;

  const double load_ms = (double)h_load_cycles / gpu_clock_hz * 1000.0;
  const double comp_ms = (double)h_comp_cycles / gpu_clock_hz * 1000.0;
  const double sum_ms  = load_ms + comp_ms;

  const double load_pct = (sum_ms > 0.0) ? (100.0 * load_ms / sum_ms) : 0.0;
  const double comp_pct = 100.0 - load_pct;

  // FLOPs & bytes (A:fp32, B:bf16, C:fp32)
  const double flops  = 2.0 * (double)M * (double)N * (double)K;
  const double bytesA = 4.0 * (double)M * (double)K;
  const double bytesB = 2.0 * (double)K * (double)N;
  const double bytesC = 4.0 * (double)M * (double)N;
  const double bytes  = bytesA + bytesB + bytesC;

  const double tflops_eff = flops / (ms_total/1000.0) / 1e12;
  const double gbps_eff   = bytes / (ms_total/1000.0) / 1e9;

  printf("Kernel wall time (events):  %.3f ms\n", ms_total);
  printf("In-kernel summed times:     load=%.3f ms, compute=%.3f ms, sum=%.3f ms\n",
         load_ms, comp_ms, sum_ms);
  printf("Phase split (sum-based):    load=%.1f%%, compute=%.1f%%\n", load_pct, comp_pct);
  printf("Effective throughput:       %.2f TFLOP/s, %.1f GB/s\n", tflops_eff, gbps_eff);
  printf("GPU clock (reported):       %.2f MHz\n", prop.clockRate/1000.0);

  // Cleanup
  hipEventDestroy(e0);
  hipEventDestroy(e1);
  hipFree(d_load_cycles);
  hipFree(d_comp_cycles);
  hipFree(dA); hipFree(dB); hipFree(dC); hipFree(dBias);
  return 0;
}
