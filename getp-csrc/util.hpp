#include <hip/hip_runtime.h>

/* Macro for checking hip errors */
#define CHECK_HIP(cmd)                                                                             \
  do {                                                                                             \
    hipError_t error = cmd;                                                                        \
    if (error != hipSuccess) {                                                                     \
      fprintf(stderr, "HIP Error: %s (%d): %s:%d\n", hipGetErrorString(error), error, __FILE__,    \
              __LINE__);                                                                           \
      fflush(stdout);                                                                              \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

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
    printf("%s: %f ms\n", function_name, milliseconds);
    CHECK_HIP(hipEventDestroy(start_event));
    CHECK_HIP(hipEventDestroy(stop_event));
  }
};

float *alloc_mat(float *m, int R, int C) {
  CHECK_HIP(hipHostMalloc(&m, R * C * sizeof(float)));
  return m;
}

float *alloc_vec(float *m, int N) {
  return alloc_mat(m, N, 1);
}

void rand_mat(float *m, int R, int C) {
  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      m[i * C + j] = (float)rand() / (float)RAND_MAX - 0.5;
    }
  }
}

void rand_vec(float *m, int N) {
  rand_mat(m, N, 1);
}

void zero_mat(float *m, int R, int C) {
  memset(m, 0, R * C * sizeof(float));
}

void zero_vec(float *m, int N) {
  zero_mat(m, N, 1);
}