#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using std::vector;

/* Macro for checking CUDA errors */
#define CHECK_HIP(call)                                                        \
  do {                                                                         \
    hipError_t _st = (call);                                                   \
    if (_st != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__,           \
              hipGetErrorString(_st));                                         \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// #define FP16 /* [Advanced] Uncomment this line only for FP16 */

/* [Tensor Structure] */
struct Tensor {
  size_t ndim = 0;
  vector<size_t> shape;
  //   size_t shape[5] = {1, 1, 1, 1, 1};
  float *buf = nullptr;

  Tensor(const vector<size_t> &shape_);
  Tensor(const vector<size_t> &shape_, float *buf_);
  ~Tensor();

  size_t num_elem() const;
  void reshape(const vector<int> &shape_);
};

struct TensorI32 {
  size_t ndim = 0;
  vector<size_t> shape;
  //   size_t shape[5] = {1, 1, 1, 1, 1};
  int *buf = nullptr;

  TensorI32(const vector<size_t> &shape_);
  TensorI32(const vector<size_t> &shape_, int *buf_);
  ~TensorI32();

  size_t num_elem() const;
  void reshape(const vector<int> &shape_);
};

typedef Tensor Parameter;
typedef Tensor Activation;