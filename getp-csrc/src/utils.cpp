#include "../include/utils.hpp"

void memcpy_tensor(Tensor *to, const Tensor *from, long long to_offset, long long from_offset,
                   size_t num_elem, bool copy_host, bool copy_device, hipStream_t stream) {
  if (copy_host) {
    memcpy(to->buf + 1ll * to_offset, from->buf + 1ll * from_offset, num_elem * sizeof(float));
  }

  if (copy_device) {
    if (to->dtype == DType::BF16) {
      bf16 *to_d_buf_bf16 = (bf16 *)to->d_buf + 1ll * to_offset;
      bf16 *from_d_buf_bf16 = (bf16 *)from->d_buf + 1ll * from_offset;

      CHECK_HIP(hipMemcpyAsync(to_d_buf_bf16, from_d_buf_bf16, num_elem * sizeof(bf16),
                               hipMemcpyDeviceToDevice, stream));
    } else {
      float *to_d_buf_fp32 = (float *)to->d_buf + 1ll * to_offset;
      float *from_d_buf_fp32 = (float *)from->d_buf + 1ll * from_offset;

      CHECK_HIP(hipMemcpyAsync(to_d_buf_fp32, from_d_buf_fp32, num_elem * sizeof(float),
                               hipMemcpyDeviceToDevice, stream));
    }

    // Haven't implemented different types for Tensor device
  }
}

void memset_tensor(Tensor *in, int value, bool set_host, bool set_device, hipStream_t stream) {
  size_t num_elem = in->num_elem();
  if (set_host) {
    memset(in->buf, 0, in->num_elem() * sizeof(float));
  }

  else {
    CHECK_HIP(hipMemsetAsync(in->d_buf, 0, in->num_elem() * in->get_dtype_size(), stream));
  }
}

void printDebugFloat(bf16 *d_buf, const std::string &descr) {
  int total_used = 50;
  bf16 *tmp_test = (bf16 *)malloc(total_used * sizeof(bf16));
  CHECK_HIP(hipMemcpy(tmp_test, d_buf, total_used * sizeof(bf16), hipMemcpyDeviceToHost));

  printf("Print debug %s: ", descr.c_str());
  for (int i = 0; i < total_used; i++) {
    float value = float(tmp_test[i]);
    printf("%.6f ", value);
  }
  printf("\n");
  fflush(stdout);

  free(tmp_test);
}

void check_gpu_memory() {
  size_t free_bytes, total_bytes;
  CHECK_HIP(hipMemGetInfo(&free_bytes, &total_bytes));

  double free_gb = (double)free_bytes / (1024.0 * 1024.0 * 1024.0);
  double total_gb = (double)total_bytes / (1024.0 * 1024.0 * 1024.0);
  double used_gb = total_gb - free_gb;

  printf("GPU Memory Info:\n");
  printf(" - Total: %lf GB\n", total_gb);
  printf(" - Used: %lf GB\n", used_gb);
  printf(" - Free: %lf GB\n", free_gb);
}

double get_time_kernel() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  // Combine seconds and nanoseconds into a single double value.
  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

CPUTimer::CPUTimer(const char *name) : function_name(name) {
  startTime = get_time_kernel();
}

CPUTimer::~CPUTimer() {
  double endTime = get_time_kernel();
  double elapsedTime = endTime - startTime;
  printf("%s CPU time: %.6f seconds\n", function_name, elapsedTime);
  fflush(stdout);
}
