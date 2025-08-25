#pragma once

#include "tensor.hpp"
#include <hip/hip_runtime.h>

struct GpuTimer {
  hipEvent_t start_event, stop_event;
  const char *function_name;

  // Constructor: Records the start event.
  GpuTimer(const char *name) : function_name(name) {
    CHECK_HIP(hipEventCreate(&start_event));
    CHECK_HIP(hipEventCreate(&stop_event));
    CHECK_HIP(hipEventRecord(start_event));
  }

  // Destructor: Records the stop event, synchronizes, and prints the time.
  ~GpuTimer() {
    CHECK_HIP(hipEventRecord(stop_event));
    CHECK_HIP(hipEventSynchronize(stop_event));
    float milliseconds = 0;
    CHECK_HIP(hipEventElapsedTime(&milliseconds, start_event, stop_event));
  #ifdef DEBUG
    printf("%s: %f ms\n", function_name, milliseconds);
  #endif
    CHECK_HIP(hipEventDestroy(start_event));
    CHECK_HIP(hipEventDestroy(stop_event));
  }
};

void memcpy_tensor(Tensor *to, const Tensor *from, long long to_offset,
                  long long from_offset, size_t num_elem, bool copy_host,
                  bool copy_device, hipStream_t stream = 0);
void memset_tensor(Tensor *in, int value, bool set_host,
                  bool set_device, hipStream_t stream = 0);
