// tensor.cpp
#include "../include/tensor.hpp"
#include <stdexcept>

Tensor::Tensor(
  const vector<size_t> &shape_, int gpu_id, DType::Type dtype
) : shape(shape_), dtype(dtype), owns_host_buf(true), gpu_id(gpu_id) {
    
  CHECK_HIP(hipSetDevice(gpu_id));  // Set device before any operations
  
  ndim = shape_.size();
  size_t N_ = num_elem();
  buf = (float *)calloc(N_, sizeof(float));

  size_t d_size = (dtype == DType::FP32) ? N_ * sizeof(float) : N_ * sizeof(bf16);
  CHECK_HIP(hipMalloc(&d_buf, d_size));
  CHECK_HIP(hipMemsetAsync(d_buf, 0, d_size, 0));
}

Tensor::Tensor(
  const vector<size_t> &shape_, float *buf_, int gpu_id, DType::Type dtype
) : shape(shape_), buf(buf_), dtype(dtype), owns_host_buf(false), gpu_id(gpu_id) {
  CHECK_HIP(hipSetDevice(gpu_id));  // Set device before any operations
  
  ndim = shape_.size();
  size_t N_ = num_elem();

  size_t d_size = (dtype == DType::FP32) ? N_ * sizeof(float) : N_ * sizeof(bf16);
  CHECK_HIP(hipMalloc(&d_buf, d_size));

  to_device(0);
}

/**
 * @brief Constructs a Tensor, optionally allocating and replicating data along the batch dimension.
 *
 * If `batch_alloc` is true, this constructor allocates a new host buffer and copies the single-item
 * data from `buf_` into each slice of the batch dimension. For example, if `shape_` is (2, 512, 4, 4),
 * the data from `buf_` (assumed to be for shape (1, 512, 4, 4)) will be copied twice to fill the new buffer.
 * The Tensor will "own" this new buffer and be responsible for freeing it.
 *
 * If `batch_alloc` is false, this constructor behaves like the standard one, simply pointing to the
 * provided `buf_` without copying data or taking ownership.
 *
 * @param shape_ The desired shape of the tensor, including the batch dimension.
 * @param buf_ A pointer to the host data for a single item.
 * @param batch_alloc A flag to enable the batch replication logic.
 * @param dtype The data type of the tensor (FP32 or BF16).
 */
Tensor::Tensor(
  const vector<size_t> &shape_, float *buf_, bool batch_alloc, int gpu_id,
  DType::Type dtype
) : shape(shape_), dtype(dtype), gpu_id(gpu_id) {
  CHECK_HIP(hipSetDevice(gpu_id));  // Set device before any operations
  
  ndim = shape_.size();
  size_t N_ = num_elem();

  if (batch_alloc) {
    // We are creating a new buffer, so this tensor owns it.
    this->buf = (float *)malloc(N_ * sizeof(float));
    if (!this->buf) {
      fprintf(stderr, "Failed to allocate host memory for batched tensor.\n");
      std::abort();
    }

    // Calculate the number of elements for a single item in the batch.
    size_t single_item_elements = N_ / shape[0];
    size_t single_item_bytes = single_item_elements * sizeof(float);

    // Copy the single item's data into each batch slot.
    for (size_t i = 0; i < shape[0]; ++i) {
      float* destination_pointer = this->buf + (i * single_item_elements);
      memcpy(destination_pointer, buf_, single_item_bytes);
    }
  } else {
    this->buf = buf_;
  }

  size_t d_size = (dtype == DType::FP32) ? N_ * sizeof(float) : N_ * sizeof(bf16);
  CHECK_HIP(hipMalloc(&d_buf, d_size));
  to_device(0);
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

void Tensor::to_device(hipStream_t stream, bool set_device) {
  if (set_device) CHECK_HIP(hipSetDevice(gpu_id));  // Set device before any operations

  size_t N_ = num_elem();
  if (dtype == DType::FP32) {
    CHECK_HIP(hipMemcpyAsync(d_buf, buf, N_ * sizeof(float), hipMemcpyHostToDevice, stream));
  } else {
    // Convert and copy for BF16
    bf16 *temp_bf16 = (bf16 *)malloc(N_ * sizeof(bf16));
    for (size_t i = 0; i < N_; i++) {
      temp_bf16[i] = hip_bfloat16(buf[i]);
    }
    CHECK_HIP(hipMemcpyAsync(d_buf, temp_bf16, N_ * sizeof(bf16), hipMemcpyHostToDevice, stream));
    free(temp_bf16);
  }
}

void Tensor::from_device(hipStream_t stream, bool set_device) {
  if (set_device) CHECK_HIP(hipSetDevice(gpu_id));  // Set device before any operations

  size_t N_ = num_elem();
  if (dtype == DType::FP32) {
    CHECK_HIP(hipMemcpyAsync(buf, d_buf, N_ * sizeof(float), hipMemcpyDeviceToHost, stream));
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
  #pragma omp critical
  {
    printf("Shape of %s tensor: ", descr.c_str());
    for (size_t i = 0; i < ndim; i++) {
      printf("%zu ", shape[i]);
    }
    printf("\n");
    fflush(stdout);
  }
}

void Tensor::printDebug(
  const std::string &descr, int tp_rank, long long offset,
  bool from_device_debug
) {
  #pragma omp critical
  {
    if (from_device_debug) from_device(0);
    CHECK_HIP(hipStreamSynchronize(0));
    printf("Print debug of %s tensor, tp_rank %d: ", descr.c_str(), tp_rank);
    for (long long id_test = 0; id_test < 5; id_test++) {
      printf("%.6f ", buf[id_test + offset]);
    }
    printf("\n");
    fflush(stdout);
  }
}

/* INT TENSOR */
// tensor.cpp (modifications for TensorI32)
TensorI32::TensorI32(
  const vector<size_t> &shape_, int gpu_id
) : shape(shape_), owns_host_buf(true), gpu_id(gpu_id) {
  CHECK_HIP(hipSetDevice(gpu_id));  // Set device before any operations
  
  ndim = shape_.size();
  size_t N_ = num_elem();
  buf = (int *)calloc(N_, sizeof(int));
  CHECK_HIP(hipMalloc(&d_buf, N_ * sizeof(int)));
  CHECK_HIP(hipMemsetAsync(d_buf, 0, N_ * sizeof(int), 0));
}

TensorI32::TensorI32(
  const vector<size_t> &shape_, int *buf_, int gpu_id
) : shape(shape_), buf(buf_), owns_host_buf(false), gpu_id(gpu_id) {
  CHECK_HIP(hipSetDevice(gpu_id));  // Set device before any operations
  
  ndim = shape_.size();
  size_t N_ = num_elem();
  CHECK_HIP(hipMalloc(&d_buf, N_ * sizeof(int)));
  to_device(0);
}

TensorI32::~TensorI32() {
  if (d_buf != nullptr) {
    CHECK_HIP(hipFree(d_buf));
  }
  if (owns_host_buf && buf != nullptr) {
    free(buf);
  }
}

void TensorI32::to_device(hipStream_t stream, bool set_device) {
  if (set_device) CHECK_HIP(hipSetDevice(gpu_id));  // Set device before any operations

  size_t N_ = num_elem();
  CHECK_HIP(hipMemcpyAsync(d_buf, buf, N_ * sizeof(int), hipMemcpyHostToDevice, stream));
}

void TensorI32::from_device(hipStream_t stream, bool set_device) {
  if (set_device) CHECK_HIP(hipSetDevice(gpu_id));  // Set device before any operations

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

void TensorI32::printDebug(
  const std::string &descr, int tp_rank, long long offset,
  bool from_device_debug
) {
  #pragma omp critical
  {
    if (from_device_debug) from_device(0);
    CHECK_HIP(hipStreamSynchronize(0));
    printf("Print debug of %s tensor, tp_rank %d: ", descr.c_str(), tp_rank);
    for (long long id_test = 0; id_test < 5; id_test++) {
      printf("%d ", buf[id_test + offset]);
    }
    printf("\n");
    fflush(stdout);
  }
}
