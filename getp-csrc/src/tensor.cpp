// tensor.cpp
#include "../include/tensor.hpp"
#include <stdexcept>

Tensor::Tensor(const vector<size_t> &shape_, hipStream_t stream, DType::Type dtype)
    : shape(shape_), dtype(dtype), owns_host_buf(true) {
  ndim = shape_.size();
  size_t N_ = num_elem();
  buf = (float *)calloc(N_, sizeof(float));

  size_t d_size = (dtype == DType::FP32) ? N_ * sizeof(float) : N_ * sizeof(bf16);
  CHECK_HIP(hipMalloc(&d_buf, d_size));
  CHECK_HIP(hipMemsetAsync(d_buf, 0, d_size, stream));
}

Tensor::Tensor(const vector<size_t> &shape_, float *buf_, hipStream_t stream, DType::Type dtype)
    : shape(shape_), buf(buf_), dtype(dtype), owns_host_buf(false) {
  ndim = shape_.size();
  size_t N_ = num_elem();

  size_t d_size = (dtype == DType::FP32) ? N_ * sizeof(float) : N_ * sizeof(bf16);
  CHECK_HIP(hipMalloc(&d_buf, d_size));

  to_device(stream);  // Copy to device with default stream
}

Tensor::~Tensor() {
  if (d_buf != nullptr) {
    CHECK_HIP(hipFree(d_buf));
  }
  if (owns_host_buf && buf != nullptr) {
    free(buf);
  }
}

size_t Tensor::num_elem() const {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  return size;
}

size_t Tensor::get_dtype_size() const {
  if (dtype == DType::FP32) {
    return sizeof(float);
  } else if (dtype == DType::BF16) {
    return sizeof(bf16);
  }
  return 0;  // Or handle as an error
}

void Tensor::to_device(hipStream_t stream) {
  size_t N_ = num_elem();
  if (dtype == DType::FP32) {
    CHECK_HIP(hipMemcpyAsync(d_buf, buf, N_ * sizeof(float), hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
  } else {
    // Convert and copy for BF16
    bf16 *temp_bf16 = (bf16 *)malloc(N_ * sizeof(bf16));
    for (size_t i = 0; i < N_; i++) {
      temp_bf16[i] = hip_bfloat16(buf[i]);
    }
    CHECK_HIP(hipMemcpyAsync(d_buf, temp_bf16, N_ * sizeof(bf16), hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    free(temp_bf16);
  }
}

void Tensor::from_device(hipStream_t stream) {
  size_t N_ = num_elem();
  if (dtype == DType::FP32) {
    CHECK_HIP(hipMemcpyAsync(buf, d_buf, N_ * sizeof(float), hipMemcpyDeviceToHost, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
  } else {
    // Convert and copy for BF16
    bf16 *temp_bf16 = (bf16 *)malloc(N_ * sizeof(bf16));
    CHECK_HIP(hipMemcpyAsync(temp_bf16, d_buf, N_ * sizeof(bf16), hipMemcpyDeviceToHost, stream));
    CHECK_HIP(hipStreamSynchronize(stream));

    for (size_t i = 0; i < N_; i++) {
      buf[i] = float(temp_bf16[i]);
    }
    free(temp_bf16);
  }
}

void Tensor::reshape(const vector<int> &shape_) {
  size_t n = 1;
  ndim = shape_.size();  // ndim<=5
  for (size_t i = 0; i < ndim; i++) {
    shape[i] = shape_[i];
    n *= shape[i];
  }
}

void Tensor::printShape(const std::string &descr) const {
  printf("Shape of %s tensor: ", descr.c_str());
  for (size_t i = 0; i < ndim; i++) {
    printf("%zu ", shape[i]);
  }
  printf("\n");
}

/* INT TENSOR */
// tensor.cpp (modifications for TensorI32)
TensorI32::TensorI32(const vector<size_t> &shape_, hipStream_t stream)
    : shape(shape_), owns_host_buf(true) {
  ndim = shape_.size();
  size_t N_ = num_elem();
  buf = (int *)calloc(N_, sizeof(int));
  CHECK_HIP(hipMalloc(&d_buf, N_ * sizeof(int)));
  CHECK_HIP(hipMemsetAsync(d_buf, 0, N_ * sizeof(int), stream));
}

TensorI32::TensorI32(const vector<size_t> &shape_, int *buf_, hipStream_t stream)
    : shape(shape_), buf(buf_), owns_host_buf(false) {
  ndim = shape_.size();
  size_t N_ = num_elem();
  CHECK_HIP(hipMalloc(&d_buf, N_ * sizeof(int)));
  to_device(stream);  // Copy to device with default stream
}

TensorI32::~TensorI32() {
  if (d_buf != nullptr) {
    CHECK_HIP(hipFree(d_buf));
  }
  if (owns_host_buf && buf != nullptr) {
    free(buf);
  }
}

void TensorI32::to_device(hipStream_t stream) {
  size_t N_ = num_elem();
  CHECK_HIP(hipMemcpyAsync(d_buf, buf, N_ * sizeof(int), hipMemcpyHostToDevice, stream));
}

void TensorI32::from_device(hipStream_t stream) {
  size_t N_ = num_elem();
  CHECK_HIP(hipMemcpyAsync(buf, d_buf, N_ * sizeof(int), hipMemcpyDeviceToHost, stream));
}

// Keep the existing num_elem() and reshape() implementations unchanged
size_t TensorI32::num_elem() const {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  return size;
}

void TensorI32::reshape(const vector<int> &shape_) {
  size_t n = 1;
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) {
    shape[i] = shape_[i];
    n *= shape[i];
  }
}
