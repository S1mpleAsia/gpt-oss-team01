#include "model.hpp"

/* NORMAL TENSOR */

Tensor::Tensor(const vector<size_t> &shape_) : shape(shape_)  {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) {
    shape[i] = shape_[i];
  }
  size_t N_ = num_elem();
  buf = (float *)calloc(N_, sizeof(float));
}

Tensor::Tensor(const vector<size_t> &shape_, float *buf_) : shape(shape_)  {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) {
    shape[i] = shape_[i];
  }
  size_t N_ = num_elem();
  buf = (float *)malloc(N_ * sizeof(float));
  memcpy(buf, buf_, N_ * sizeof(float));
}

Tensor::~Tensor() {
  if (buf != nullptr)
    free(buf);
}

size_t Tensor::num_elem() const {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  return size;
}

void Tensor::reshape(const vector<int> &shape_) {
  size_t n = 1;
  ndim = shape_.size(); // ndim<=5
  for (size_t i = 0; i < ndim; i++) {
    shape[i] = shape_[i];
    n *= shape[i];
  }
}

/* INT TENSOR */
TensorI32::TensorI32(const vector<size_t> &shape_) : shape(shape_)  {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) {
    shape[i] = shape_[i];
  }
  size_t N_ = num_elem();
  buf = (int *)calloc(N_, sizeof(int));
}

TensorI32::TensorI32(const vector<size_t> &shape_, int *buf_) : shape(shape_)  {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) {
    shape[i] = shape_[i];
  }
  size_t N_ = num_elem();
  buf = (int *)malloc(N_ * sizeof(int));
  memcpy(buf, buf_, N_ * sizeof(int));
}

TensorI32::~TensorI32() {
  if (buf != nullptr)
    free(buf);
}

size_t TensorI32::num_elem() const {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  return size;
}

void TensorI32::reshape(const vector<int> &shape_) {
  size_t n = 1;
  ndim = shape_.size(); // ndim<=5
  for (size_t i = 0; i < ndim; i++) {
    shape[i] = shape_[i];
    n *= shape[i];
  }
}