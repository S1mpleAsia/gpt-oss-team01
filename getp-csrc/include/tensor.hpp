#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using std::vector;

typedef hip_bfloat16 bf16;

#define RUN_BATCH
#define BATCH_SIZE 2
#define PRINT_LOGITS
// #define TIME_GPU

struct DType {
  enum Type { FP32, BF16 };
};

#define CHECK_HIP(call)                                                                            \
  do {                                                                                             \
    hipError_t _st = (call);                                                                       \
    if (_st != hipSuccess) {                                                                       \
      fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(_st));      \
      std::abort();                                                                                \
    }                                                                                              \
  } while (0)

struct Tensor {
  size_t ndim = 0;
  vector<size_t> shape;
  float *buf = nullptr;
  void *d_buf = nullptr;  // Unified device pointer
  DType::Type dtype;
  bool owns_host_buf;

  Tensor(const vector<size_t> &shape_, DType::Type dtype = DType::FP32);
  Tensor(const vector<size_t> &shape_, float *buf_, DType::Type dtype = DType::FP32);
  Tensor(const vector<size_t> &shape_, float *buf_, bool batch_alloc, DType::Type dtype = DType::FP32);
  ~Tensor();

  size_t num_elem() const;
  size_t get_dtype_size() const;  // Add this declaration
  void printShape(const std::string &descr) const;
  void reshape(const vector<int> &shape_);

  void to_device(hipStream_t stream = 0);
  void from_device(hipStream_t stream = 0);
};

struct TensorI32 {
  size_t ndim = 0;
  vector<size_t> shape;
  int *buf = nullptr;
  int *d_buf = nullptr;  // Device buffer
  bool owns_host_buf;

  TensorI32(const vector<size_t> &shape_);
  TensorI32(const vector<size_t> &shape_, int *buf_);
  ~TensorI32();

  size_t num_elem() const;
  void reshape(const vector<int> &shape_);

  void to_device(hipStream_t stream = 0);
  void from_device(hipStream_t stream = 0);
};
