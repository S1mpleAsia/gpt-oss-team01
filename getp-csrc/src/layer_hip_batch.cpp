#include "../include/layer_hip_batch.hpp"
#include <cmath>
#include <cfloat>
// #include "kernel.cpp"
#include "matrix_core.cpp"

#define DEFAULT_BLOCK_SIZE 256

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }

  return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float warp_sums[32];

  int lane = threadIdx.x & (warpSize - 1);
  int wid = threadIdx.x / warpSize;
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;

  v = warp_reduce_sum(v);
  if (lane == 0)
    warp_sums[wid] = v;
  __syncthreads();

  float sum = 0.0f;
  if (wid == 0) {
    sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
    sum = warp_reduce_sum(sum);
    if (lane == 0)
      warp_sums[0] = sum;
  }
  __syncthreads();
  return warp_sums[0];
}

template <int BLOCK_THREADS = 256>
__global__ void residual_rmsnorm_f32_kernel(const float *input, float *residual, const float *w,
                                            float *output, int hidden_dim, float epsilon) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  const float *in_row = input + (size_t)row * hidden_dim;
  float *res_row = residual + (size_t)row * hidden_dim;
  const float *g_row = w;
  float *out_row = output + (size_t)row * hidden_dim;

  float thread_sum = 0.f;

  const int vecN = hidden_dim >> 2;
  const float4 *in4 = reinterpret_cast<const float4 *>(in_row);
  float4 *rs4 = reinterpret_cast<float4 *>(res_row);

  for (int i = tid; i < vecN; i += BLOCK_THREADS) {
    float4 a = in4[i];
    float4 r = rs4[i];
    r.x += a.x;
    r.y += a.y;
    r.z += a.z;
    r.w += a.w;
    rs4[i] = r;
    thread_sum += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
  }

  float sum = block_reduce_sum(thread_sum);
  float s_norm;
  if (tid == 0) {
    float mean = sum / (float)hidden_dim;
    s_norm = rsqrtf(mean + epsilon);
  }
  __shared__ float s_shared;
  if (tid == 0)
    s_shared = s_norm;
  __syncthreads();
  s_norm = s_shared;

  const float4 *g4 = reinterpret_cast<const float4 *>(g_row);
  float4 *o4 = reinterpret_cast<float4 *>(out_row);

  for (int i = tid; i < vecN; i += BLOCK_THREADS) {
    float4 r = rs4[i];
    float4 g = g4[i];
    r.x = r.x * s_norm * g.x;
    r.y = r.y * s_norm * g.y;
    r.z = r.z * s_norm * g.z;
    r.w = r.w * s_norm * g.w;
    o4[i] = r;
  }
}

template <int BLOCK_THREADS = 256>
__global__ void residual_rmsnorm_bf16_kernel(const float *input, float *residual, const float *w,
                                             bf16 *output, int hidden_dim, float epsilon) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  const float *in_row = input + (size_t)row * hidden_dim;
  float *res_row = residual + (size_t)row * hidden_dim;
  const float *g_row = w;
  bf16 *out_row = output + (size_t)row * hidden_dim;

  float thread_sum = 0.f;

  const int vecN = hidden_dim >> 2;
  const float4 *in4 = reinterpret_cast<const float4 *>(in_row);
  float4 *rs4 = reinterpret_cast<float4 *>(res_row);

  for (int i = tid; i < vecN; i += BLOCK_THREADS) {
    float4 a = in4[i];
    float4 r = rs4[i];
    r.x += a.x;
    r.y += a.y;
    r.z += a.z;
    r.w += a.w;
    rs4[i] = r;
    thread_sum += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
  }

  float sum = block_reduce_sum(thread_sum);
  float s_norm;
  if (tid == 0) {
    float mean = sum / (float)hidden_dim;
    s_norm = rsqrtf(mean + epsilon);
  }
  __shared__ float s_shared;
  if (tid == 0)
    s_shared = s_norm;
  __syncthreads();
  s_norm = s_shared;

  const float4 *g4 = reinterpret_cast<const float4 *>(g_row);
  bf16x4 *o4 = reinterpret_cast<bf16x4 *>(out_row);

  for (int i = tid; i < vecN; i += BLOCK_THREADS) {
    float4 r = rs4[i];
    float4 g = g4[i];

    float4 y;
    y.x = r.x * s_norm * g.x;
    y.y = r.y * s_norm * g.y;
    y.z = r.z * s_norm * g.z;
    y.w = r.w * s_norm * g.w;

    bf16x4 tmp;
    ((__bf16 *)&tmp)[0] = (__bf16)y.x;
    ((__bf16 *)&tmp)[1] = (__bf16)y.y;
    ((__bf16 *)&tmp)[2] = (__bf16)y.z;
    ((__bf16 *)&tmp)[3] = (__bf16)y.w;

    o4[i] = tmp;
  }
}

void residual_rmsnorm_batched(Tensor *x,         // Shape: [batch_size, hidden_dim]
                              Tensor *residual,  // Shape: [batch_size, hidden_dim]
                              Tensor *w,         // Shape: [n_layers, hidden_dim]
                              Tensor *out,       // Shape: [batch_size, hidden_dim]
                              int cur_batch_size, long long layer_offset, float epsilon,
                              hipStream_t stream) {
  // GpuTimer timer("residual_rmsnorm", stream);
  const int hidden_dim = (int)residual->shape[1];

  const float *x_ptr = (const float *)x->d_buf;
  float *residual_ptr = (float *)residual->d_buf;
  float *out_ptr = (float *)out->d_buf;
  const float *w_ptr = (const float *)w->d_buf + layer_offset * hidden_dim;

  dim3 block_size(256);
  dim3 grid_size(cur_batch_size);

  residual_rmsnorm_f32_kernel<256><<<grid_size, block_size, 0, stream>>>(
    x_ptr, residual_ptr, w_ptr, out_ptr, hidden_dim, epsilon);
}

void residual_rmsnorm_batched_quantize(Tensor *x,         // Shape: [batch_size, hidden_dim]
                                       Tensor *residual,  // Shape: [batch_size, hidden_dim]
                                       Tensor *w,         // Shape: [n_layers, hidden_dim]
                                       Tensor *out,       // Shape: [batch_size, hidden_dim]
                                       int cur_batch_size, long long layer_offset, float epsilon,
                                       hipStream_t stream) {
  // GpuTimer timer("residual_rmsnorm", stream);
  const int hidden_dim = (int)residual->shape[1];

  const float *x_ptr = (const float *)x->d_buf;
  float *residual_ptr = (float *)residual->d_buf;
  bf16 *out_ptr = (bf16 *)out->d_buf;
  const float *w_ptr = (const float *)w->d_buf + layer_offset * hidden_dim;

  dim3 block_size(256);
  dim3 grid_size(cur_batch_size);

  residual_rmsnorm_bf16_kernel<256><<<grid_size, block_size, 0, stream>>>(
    x_ptr, residual_ptr, w_ptr, out_ptr, hidden_dim, epsilon);
}

__global__ void embedding_lookup_kernel(const bf16 *__restrict__ embedding_table,
                                        float *__restrict__ output, const int *__restrict__ tokens,
                                        const size_t hidden_dim, const int batch_size) {
  int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch_idx >= batch_size || hidden_idx >= hidden_dim) {
    return;
  }

  int token_idx = tokens[batch_idx];
  const bf16 *src_ptr = embedding_table + (size_t)token_idx * hidden_dim + hidden_idx;
  float *dst_ptr = output + (size_t)batch_idx * hidden_dim + hidden_idx;

  *dst_ptr = (float)(*src_ptr);
}

void embedding_lookup_batched(Tensor *embedding,      // Shape: [vocab_size, hidden_dim]
                              int *tokens,            // Shape: [batch_size]
                              TensorI32 *tokens_buf,  // Shape: [batch_size]
                              Tensor *x,              // Shape: [batch_size, hidden_dim]
                              int cur_batch_size, bool x_from_device, hipStream_t stream) {
  // GpuTimer timer("embedding_lookup", stream);

  // const int batch_size = x->shape[0];
  const size_t hidden_dim = x->shape[1];
  const bf16 *embedding_ptr = (const bf16 *)embedding->d_buf;
  float *x_ptr = (float *)x->d_buf;

  CHECK_HIP(hipMemcpyAsync(tokens_buf->d_buf, tokens, cur_batch_size * sizeof(int),
                           hipMemcpyHostToDevice, stream));

  dim3 block_size(32, 16);
  dim3 grid_size((hidden_dim + block_size.x - 1) / block_size.x,
                 (cur_batch_size + block_size.y - 1) / block_size.y);

  embedding_lookup_kernel<<<grid_size, block_size, 0, stream>>>(
    embedding_ptr, x_ptr, tokens_buf->d_buf, hidden_dim, cur_batch_size);

  if (x_from_device) {
    x->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

__global__ void embedding_lookup_kernel_shard(const bf16 *__restrict__ embedding_table,
                                              float *__restrict__ output,
                                              const int *__restrict__ tokens,
                                              const size_t shard_dim, const size_t hidden_dim,
                                              const int batch_size) {
  int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int shard_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch_idx >= batch_size || shard_idx >= shard_dim) {
    return;
  }

  int token_idx = tokens[batch_idx];
  const bf16 *src_ptr = embedding_table + (size_t)token_idx * shard_dim + shard_idx;
  float *dst_ptr = output + (size_t)batch_idx * hidden_dim + shard_idx;

  *dst_ptr = (float)(*src_ptr);
}

void embedding_lookup_shard_batched(Tensor *embedding,      // Shape: [vocab_size, hidden_dim]
                                    int *tokens,            // Shape: [batch_size]
                                    TensorI32 *tokens_buf,  // Shape: [batch_size]
                                    Tensor *x,              // Shape: [batch_size, hidden_dim]
                                    int cur_batch_size, int tp_rank, bool x_from_device,
                                    hipStream_t stream) {
  // GpuTimer timer("embedding_lookup", stream);

  // const int batch_size = x->shape[0];
  const size_t hidden_dim = x->shape[1];
  const size_t shard_dim = hidden_dim / TP;
  const bf16 *embedding_ptr = (const bf16 *)embedding->d_buf;
  float *x_ptr = (float *)x->d_buf + tp_rank * shard_dim;

  // for (int i = 0; i < BATCH_SIZE; i++) {
  //   if (tokens[i] < 0 || tokens[i] >= 201088) {
  //     printf("Index out of bound %d\n", tokens[i]);
  //     return;
  //   }
  // }

  CHECK_HIP(hipMemcpyAsync(tokens_buf->d_buf, tokens, cur_batch_size * sizeof(int),
                           hipMemcpyHostToDevice, stream));

  dim3 block_size(32, 16);
  dim3 grid_size((shard_dim + block_size.x - 1) / block_size.x,
                 (cur_batch_size + block_size.y - 1) / block_size.y);

  embedding_lookup_kernel_shard<<<grid_size, block_size, 0, stream>>>(
    embedding_ptr, x_ptr, tokens_buf->d_buf, shard_dim, hidden_dim, cur_batch_size);

  if (x_from_device) {
    x->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

__global__ void rmsnorm_kernel(const float *x, const bf16 *w, float *out, int hidden_dim,
                               float eps) {
  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  const float *x_row = x + batch_idx * hidden_dim;
  float *o_row = out + batch_idx * hidden_dim;

  double acc = 0.0;
  for (int j = tid; j < hidden_dim; j += block_size) {
    float v = x_row[j];
    acc += v * v;
  }

  extern __shared__ double s_partials[];
  s_partials[tid] = acc;
  __syncthreads();

  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_partials[tid] += s_partials[tid + s];
    }
    __syncthreads();
  }

  __shared__ double final_inv_rms;
  if (tid == 0) {
    double block_sum = s_partials[0];
    double mean = block_sum / hidden_dim;
    final_inv_rms = 1.0f / sqrtf(mean + eps);
  }
  __syncthreads();

  for (int j = tid; j < hidden_dim; j += block_size) {
    o_row[j] = (float)w[j] * (final_inv_rms * x_row[j]);
  }
}

__global__ void rmsnorm_kernel_bf16(const float *x, const bf16 *w, bf16 *out, int hidden_dim,
                                    float eps) {
  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  const float *x_row = x + batch_idx * hidden_dim;
  bf16 *o_row = out + batch_idx * hidden_dim;

  double acc = 0.0;
  for (int j = tid; j < hidden_dim; j += block_size) {
    float v = x_row[j];
    acc += v * v;
  }

  extern __shared__ double s_partials[];
  s_partials[tid] = acc;
  __syncthreads();

  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_partials[tid] += s_partials[tid + s];
    }
    __syncthreads();
  }

  __shared__ double final_inv_rms;
  if (tid == 0) {
    double block_sum = s_partials[0];
    double mean = block_sum / hidden_dim;
    final_inv_rms = 1.0f / sqrtf(mean + eps);
  }
  __syncthreads();

  for (int j = tid; j < hidden_dim; j += block_size) {
    o_row[j] = bf16((float)w[j] * (final_inv_rms * x_row[j]));
  }
}

void rmsnorm_batched(Tensor *x,    // Shape: [batch_size, hidden_dim]
                     Tensor *w,    // Shape: [n_layers, hidden_dim]
                     Tensor *out,  // Shape: [batch_size, hidden_dim]
                     int cur_batch_size, long long layer_offset, bool x_to_device,
                     bool out_from_device, float eps, hipStream_t stream) {
  // GpuTimer timer("rmsnorm", stream);
  if (x_to_device) {
    x->to_device(stream);
  }

  // const int batch_size = x->shape[0];
  const int hidden_dim = x->shape[1];

  const dim3 grid_dim(cur_batch_size);
  const dim3 block_dim(256);
  size_t shared_mem_size = block_dim.x * sizeof(double);

  const float *x_ptr = (float *)x->d_buf;
  const bf16 *w_ptr = (const bf16 *)w->d_buf + 1ll * layer_offset * hidden_dim;
  float *out_ptr = (float *)out->d_buf;

  rmsnorm_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(x_ptr, w_ptr, out_ptr,
                                                                   hidden_dim, eps);
  CHECK_HIP(hipGetLastError());

  if (out_from_device) {
    out->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

void rmsnorm_batched_quantize(Tensor *x,    // Shape: [batch_size, hidden_dim]
                              Tensor *w,    // Shape: [hidden_dim]
                              Tensor *out,  // Shape: [batch_size, hidden_dim]
                              int cur_batch_size, long long layer_offset, bool x_to_device,
                              bool out_from_device, float eps, hipStream_t stream) {
  // GpuTimer timer("rmsnorm", stream);
  if (x_to_device) {
    x->to_device(stream);
  }

  // const int batch_size = x->shape[0];
  const int hidden_dim = x->shape[1];

  const dim3 grid_dim(cur_batch_size);
  const dim3 block_dim(256);
  size_t shared_mem_size = block_dim.x * sizeof(double);

  const float *x_ptr = (float *)x->d_buf;
  const bf16 *w_ptr = (const bf16 *)w->d_buf + 1ll * layer_offset * hidden_dim;
  bf16 *out_ptr = (bf16 *)out->d_buf;

  rmsnorm_kernel_bf16<<<grid_dim, block_dim, shared_mem_size, stream>>>(x_ptr, w_ptr, out_ptr,
                                                                        hidden_dim, eps);
  CHECK_HIP(hipGetLastError());

  if (out_from_device) {
    out->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

// ---------- QKV GEMM ----------

template <int WARPS_PER_BLOCK, int TILE, int VEC = 4>
__global__ void gemm_kernel_batched(const float *__restrict__ W,     // [out_features, in_features]
                                    const float *__restrict__ x,     // [B, in_features]
                                    const float *__restrict__ bias,  // [out_features] or nullptr
                                    float *__restrict__ out,         // [B, out_features]
                                    int out_features, int in_features, int batch_size) {
  extern __shared__ float s_x[];  // TILE floats

  const int b = blockIdx.y;
  if (b >= batch_size)
    return;

  const int lane = threadIdx.x;  // 0..warpSize-1 (MI250 wave = 64)
  const int warp = threadIdx.y;  // 0..WARPS_PER_BLOCK-1
  const int row = blockIdx.x * WARPS_PER_BLOCK + warp;
  if (row >= out_features)
    return;

  const float *__restrict__ x_b = x + (size_t)b * in_features;
  float *__restrict__ out_b = out + (size_t)b * out_features;
  const float *__restrict__ W_row = W + (size_t)row * in_features;

  float acc = 0.0f;

  // Process input in TILE chunks (cooperatively load x into LDS)
  for (int k0 = 0; k0 < in_features; k0 += TILE) {
    const int tile_len = min(TILE, in_features - k0);

    // all warps load x tile
    for (int t = warp * warpSize + lane; t < tile_len; t += WARPS_PER_BLOCK * warpSize)
      s_x[t] = x_b[k0 + t];
    __syncthreads();

    // main vectorized loop (only if aligned & tile_len >= VEC)
    int vec_elems = (tile_len / VEC) * VEC;

    // assure the base is 16B aligned for float4 reads
    bool vec_ok = ((reinterpret_cast<uintptr_t>(W_row + k0) & 0xF) == 0);

    if (vec_ok) {
      // each lane handles VEC elements at a time
      int t = lane * VEC;
      for (; t < vec_elems; t += warpSize * VEC) {
        // load 4 weights
        const float4 w4 = *reinterpret_cast<const float4 *>(W_row + k0 + t);
        // read 4 x's from shared (shared bandwidth is high; scalar reads are fine)
        const float x0 = s_x[t + 0];
        const float x1 = s_x[t + 1];
        const float x2 = s_x[t + 2];
        const float x3 = s_x[t + 3];
        acc = fmaf(w4.x, x0, acc);
        acc = fmaf(w4.y, x1, acc);
        acc = fmaf(w4.z, x2, acc);
        acc = fmaf(w4.w, x3, acc);
      }
      // tail of the tile
      for (; t < tile_len; ++t)
        acc = fmaf(W_row[k0 + t], s_x[t], acc);
    } else {
      // unaligned / short tile path (perfectly safe)
      for (int t = lane; t < tile_len; t += warpSize)
        acc = fmaf(W_row[k0 + t], s_x[t], acc);
    }

    __syncthreads();
  }

// warp reduction
#pragma unroll
  for (int offs = warpSize >> 1; offs > 0; offs >>= 1)
    acc += __shfl_down(acc, offs);

  if (lane == 0) {
    float v = acc;
    if (bias)
      v += bias[row];
    out_b[row] = v;
  }
}

template <int WARPS_PER_BLOCK, int TILE, int VEC = 4>
__global__ void gemm_kernel_batched_bf16(
  const float *__restrict__ W,     // [out_features, in_features]
  const bf16 *__restrict__ x,      // [B, in_features]
  const float *__restrict__ bias,  // [out_features] or nullptr
  float *__restrict__ out,         // [B, out_features]
  int out_features, int in_features, int batch_size) {
  extern __shared__ float s_x[];  // TILE floats

  const int b = blockIdx.y;
  if (b >= batch_size)
    return;

  const int lane = threadIdx.x;  // 0..warpSize-1 (MI250 wave = 64)
  const int warp = threadIdx.y;  // 0..WARPS_PER_BLOCK-1
  const int row = blockIdx.x * WARPS_PER_BLOCK + warp;
  if (row >= out_features)
    return;

  const bf16 *__restrict__ x_b = x + (size_t)b * in_features;
  float *__restrict__ out_b = out + (size_t)b * out_features;
  const float *__restrict__ W_row = W + (size_t)row * in_features;

  float acc = 0.0f;

  // Process input in TILE chunks (cooperatively load x into LDS)
  for (int k0 = 0; k0 < in_features; k0 += TILE) {
    const int tile_len = min(TILE, in_features - k0);

    // all warps load x tile
    for (int t = warp * warpSize + lane; t < tile_len; t += WARPS_PER_BLOCK * warpSize)
      s_x[t] = (float)x_b[k0 + t];
    __syncthreads();

    // main vectorized loop (only if aligned & tile_len >= VEC)
    int vec_elems = (tile_len / VEC) * VEC;

    // assure the base is 16B aligned for float4 reads
    bool vec_ok = ((reinterpret_cast<uintptr_t>(W_row + k0) & 0xF) == 0);

    if (vec_ok) {
      // each lane handles VEC elements at a time
      int t = lane * VEC;
      for (; t < vec_elems; t += warpSize * VEC) {
        // load 4 weights
        const float4 w4 = *reinterpret_cast<const float4 *>(W_row + k0 + t);
        // read 4 x's from shared (shared bandwidth is high; scalar reads are fine)
        const float x0 = s_x[t + 0];
        const float x1 = s_x[t + 1];
        const float x2 = s_x[t + 2];
        const float x3 = s_x[t + 3];
        acc = fmaf(w4.x, x0, acc);
        acc = fmaf(w4.y, x1, acc);
        acc = fmaf(w4.z, x2, acc);
        acc = fmaf(w4.w, x3, acc);
      }
      // tail of the tile
      for (; t < tile_len; ++t)
        acc = fmaf(W_row[k0 + t], s_x[t], acc);
    } else {
      // unaligned / short tile path (perfectly safe)
      for (int t = lane; t < tile_len; t += warpSize)
        acc = fmaf(W_row[k0 + t], s_x[t], acc);
    }

    __syncthreads();
  }

// warp reduction
#pragma unroll
  for (int offs = warpSize >> 1; offs > 0; offs >>= 1)
    acc += __shfl_down(acc, offs);

  if (lane == 0) {
    float v = acc;
    if (bias)
      v += bias[row];
    out_b[row] = v;
  }
}

void qkv_gemm_batched_v2(Tensor *x,            // Shape: [batch_size, hidden_dim]
                         const Tensor *W_qkv,  // Shape: [hidden_dim, out_features]
                         const Tensor *b_qkv,  // Shape: [out_features]
                         Tensor *qkv,          // Shape: [batch_size, out_features]
                         int cur_batch_size, long long layer_offset, bool x_to_device,
                         bool qkv_from_device, hipStream_t stream) {
  // GpuTimer timer("qkv_gemm_v2", stream);
  if (x_to_device) {
    x->to_device(stream);
  }

  // const int batch_size = x->shape[0];      // M
  const int in_features = x->shape[1];     // K
  const int out_features = qkv->shape[1];  // N

  const bf16 *x_ptr = (bf16 *)x->d_buf;
  const bf16 *w_qkv_ptr =
    (const bf16 *)W_qkv->d_buf + 1ll * layer_offset * out_features * in_features;
  const bf16 *b_qkv_ptr = (const bf16 *)b_qkv->d_buf + 1ll * layer_offset * out_features;
  float *qkv_ptr = (float *)qkv->d_buf;
  {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 64;
    constexpr int TN = 32;
    constexpr int blockDim = 512;  // = 64 * (BM / TM) * (BN / TN)

    dim3 block_size(blockDim);
    dim3 grid_size((out_features + BN - 1) / BN, ((cur_batch_size + BM - 1) / BM));

    gemm_mfma_v2_bf16<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
      x_ptr, w_qkv_ptr, qkv_ptr, b_qkv_ptr, cur_batch_size, out_features, in_features);
  }

  if (qkv_from_device) {
    qkv->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

void qkv_gemm_batched_v2_quantize(Tensor *x,            // Shape: [batch_size, hidden_dim]
                                  const Tensor *W_qkv,  // Shape: [hidden_dim, out_features]
                                  const Tensor *b_qkv,  // Shape: [out_features]
                                  Tensor *qkv,          // Shape: [batch_size, out_features]
                                  int cur_batch_size, long long layer_offset, bool x_to_device,
                                  bool qkv_from_device, hipStream_t stream) {
  // GpuTimer timer("qkv_gemm_v2", stream);
  if (x_to_device) {
    x->to_device(stream);
  }

  // const int batch_size = x->shape[0];      // M
  const int in_features = x->shape[1];     // K
  const int out_features = qkv->shape[1];  // N

  const bf16 *x_ptr = (bf16 *)x->d_buf;
  const bf16 *w_qkv_ptr =
    (const bf16 *)W_qkv->d_buf + 1ll * layer_offset * out_features * in_features;
  const bf16 *b_qkv_ptr = (const bf16 *)b_qkv->d_buf + 1ll * layer_offset * out_features;
  float *qkv_ptr = (float *)qkv->d_buf;
  {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 64;
    constexpr int TN = 32;
    constexpr int blockDim = 512;  // = 64 * (BM / TM) * (BN / TN)

    dim3 block_size(blockDim);
    dim3 grid_size((out_features + BN - 1) / BN, ((cur_batch_size + BM - 1) / BM));

    gemm_mfma_v2_bf16<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
      x_ptr, w_qkv_ptr, qkv_ptr, b_qkv_ptr, cur_batch_size, out_features, in_features);
  }

  if (qkv_from_device) {
    qkv->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

// ---------- Split & RoPE ----------
template <bool KV_BF16 = false>
__global__ void qkv_split_rope_store_kernel(
  const float *__restrict__ qkv_in,    // [batch_size, (n_attn_heads + 2 * n_kv_heads) * head_dim]
  float *__restrict__ q_out,           // [batch_size, n_attn_heads * head_dim]
  void *__restrict__ K_cache,          // [batch_size, n_layers, seq_len, kv_dim]
  void *__restrict__ V_cache,          // [batch_size, n_layers, seq_len, kv_dim]
  const float *__restrict__ rope_cos,  // [head_dim / 2] row for 'pos'
  const float *__restrict__ rope_sin,  // [head_dim / 2] row for 'pos'
  int B, int head_dim, int n_q, int n_kv, int seq_len, int n_layers, int pos) {
  const int b = blockIdx.y;
  if (b >= B)
    return;

  const int lane = blockIdx.x * blockDim.x + threadIdx.x;

  const int half_head = head_dim >> 1;
  const int q_dim = n_q * head_dim;
  const int k_dim = n_kv * head_dim;
  const int v_dim = n_kv * head_dim;
  const int total = q_dim + k_dim + v_dim;

  const float *qkv_b = qkv_in + (size_t)b * total;
  float *q_b = q_out + (size_t)b * q_dim;

  // per-batch base in the cache (host already offset to layer 0)
  const size_t kv_batch_stride = (size_t)n_layers * seq_len * (size_t)(n_kv * head_dim);
  char *Kb =
    (char *)K_cache + (size_t)b * kv_batch_stride * (KV_BF16 ? sizeof(bf16) : sizeof(float));
  char *Vb =
    (char *)V_cache + (size_t)b * kv_batch_stride * (KV_BF16 ? sizeof(bf16) : sizeof(float));
  const size_t t_base = (size_t)(pos & (seq_len - 1)) * (size_t)(n_kv * head_dim);

  // -------- 1) Q: RoPE in pairs, write to q_out --------
  // treat as (n_q * h2) pairs
  for (int p = lane; p < n_q * half_head; p += gridDim.x * blockDim.x) {
    int h = p / half_head;  // q-head
    int j = p % half_head;  // pair index within head
    int i0 = h * head_dim + j;
    int i1 = i0 + half_head;

    const float c = rope_cos[j];
    const float s = rope_sin[j];

    float x0 = qkv_b[i0];
    float x1 = qkv_b[i1];

    float y0 = x0 * c - x1 * s;
    float y1 = x0 * s + x1 * c;

    q_b[i0] = y0;
    q_b[i1] = y1;
  }

  // K section starts at offset q_dim
  for (int p = lane; p < n_kv * half_head; p += gridDim.x * blockDim.x) {
    int h = p / half_head;  // kv-head
    int j = p % half_head;
    int i0 = q_dim + h * head_dim + j;  // source in qkv_b
    int i1 = i0 + half_head;

    const float c = rope_cos[j];
    const float s = rope_sin[j];

    float k0 = qkv_b[i0];
    float k1 = qkv_b[i1];

    float r0 = k0 * c - k1 * s;
    float r1 = k0 * s + k1 * c;

    size_t dst = t_base + (size_t)h * head_dim + j;  // destination index for j
                                                     // write two elements (j and j+h2)
    if constexpr (KV_BF16) {
      ((bf16 *)Kb)[dst] = (bf16)r0;
      ((bf16 *)Kb)[dst + half_head] = (bf16)r1;
    } else {
      ((float *)Kb)[dst] = r0;
      ((float *)Kb)[dst + half_head] = r1;
    }
  }

  // V section starts at q_dim + kdim ; write contiguous kv_dim elements
  for (int d = lane; d < k_dim; d += gridDim.x * blockDim.x) {
    float v = qkv_b[q_dim + k_dim + d];
    size_t dst = t_base + d;
    if constexpr (KV_BF16) {
      ((bf16 *)Vb)[dst] = (bf16)v;
    } else {
      ((float *)Vb)[dst] = v;
    }
  }
}

void qkv_split_rope_fused(Tensor *qkv_out,  // Shape: [batch_size, (n_q + 2*n_kv) * head_dim]
                          Tensor *q_out,    // Shape: [batch_size, n_q * head_dim]
                          Tensor *K_cache,  // Shape: [batch_size, n_layers, seq_len, kv_dim]
                          Tensor *V_cache,  // Shape: [batch_size, n_layers, seq_len, kv_dim]
                          const Tensor *rope_cos_pos,  // Shape: [seq_len, head_dim / 2]
                          const Tensor *rope_sin_pos,  // Shape: [seq_len, head_dim / 2]
                          int cur_batch_size, int head_dim, int n_q, int n_kv, int pos,
                          long long layer_offset, hipStream_t stream) {
  // GpuTimer timer("qkv_split_fused", stream);
  // const int batch_size = (int)qkv_out->shape[0];
  const int kv_dim = n_kv * head_dim;
  const int seq_len = K_cache->shape[2];
  const int n_layers = K_cache->shape[1];

  const size_t elem_bytes = (K_cache->dtype == DType::BF16) ? sizeof(bf16) : sizeof(float);
  void *k_ptr = (char *)K_cache->d_buf + (layer_offset >> 1) * seq_len * kv_dim * elem_bytes;
  void *v_ptr = (char *)V_cache->d_buf + (layer_offset >> 1) * seq_len * kv_dim * elem_bytes;

  const int half_head = head_dim >> 1;
  const int total_pairs = n_kv > n_q ? n_kv * half_head : n_q * half_head;
  const int vec_span = total_pairs > kv_dim ? total_pairs : kv_dim;

  dim3 block_size(256);
  dim3 grid_size(((vec_span + block_size.x - 1) / block_size.x), cur_batch_size);

  const float *cos_row = (const float *)rope_cos_pos->d_buf + (size_t)pos * half_head;
  const float *sin_row = (const float *)rope_sin_pos->d_buf + (size_t)pos * half_head;

  if (K_cache->dtype == DType::BF16) {
    qkv_split_rope_store_kernel<true><<<grid_size, block_size, 0, stream>>>(
      (const float *)qkv_out->d_buf, (float *)q_out->d_buf, k_ptr, v_ptr, cos_row, sin_row,
      cur_batch_size, head_dim, n_q, n_kv, seq_len, n_layers, pos);
  } else {
    qkv_split_rope_store_kernel<false><<<grid_size, block_size, 0, stream>>>(
      (const float *)qkv_out->d_buf, (float *)q_out->d_buf, k_ptr, v_ptr, cos_row, sin_row,
      cur_batch_size, head_dim, n_q, n_kv, seq_len, n_layers, pos);
  }
  CHECK_HIP(hipGetLastError());
}

__global__ void add_vector_kernel_batched(float *y, const float *b, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    y[i] += b[i];
  }
}

__global__ void add_vector_kernel_v2(float *y, const float *b, int len) {
  float4 *y4 = reinterpret_cast<float4 *>(y);
  const float4 *b4 = reinterpret_cast<const float4 *>(b);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int len_vec = len / 4;

  if (i < len_vec) {
    float4 y_vec = y4[i];
    const float4 b_vec = b4[i];

    y_vec.x += b_vec.x;
    y_vec.y += b_vec.y;
    y_vec.z += b_vec.z;
    y_vec.w += b_vec.w;

    y4[i] = y_vec;
  }
}

__global__ void add_vector_2d_kernel(float *__restrict__ dst, const float *__restrict__ src,
                                     int height, int width, int dst_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= height * width)
    return;

  int r = i / width;
  int c = i % width;
  dst[r * dst_stride + c] += src[r * width + c];
}

void add_vector_batched(Tensor *y,  // Shape: [batch_size, hidden_dim]
                        Tensor *b,  // Shape: [batch_size, hidden_dim]
                        int cur_batch_size, bool y_to_device, bool b_to_device, bool y_from_device,
                        hipStream_t stream) {
  // GpuTimer timer("add_vector", stream);
  if (y_to_device)
    y->to_device(stream);
  if (b_to_device)
    b->to_device(stream);

  float *y_ptr = (float *)y->d_buf;
  const float *b_ptr = (float *)b->d_buf;

  const int len = cur_batch_size * y->shape[1];

  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((len + block_dim.x - 1) / block_dim.x);
  add_vector_kernel_batched<<<grid_dim, block_dim, 0, stream>>>(y_ptr, b_ptr, len);

  if (y_from_device) {
    y->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

void attn_out_project_batched_v2_quantize(
  Tensor *tb,         // Shape: [batch_size, n_attn_heads * head_dim]
  const Tensor *W_o,  // Shape: [n_attn_heads * head_dim, hidden_dim]
  const Tensor *b_o,  // Shape: [hidden_dim]
  Tensor *y,          // Shape: [batch_size, hidden_dim]
  bool has_bias, int cur_batch_size, long long layer_offset, bool tb_to_device, bool y_from_device,
  hipStream_t stream) {
  // GpuTimer timer("attn_out_project_v2", stream);
  if (tb_to_device) {
    tb->to_device(stream);
  }

  // const int batch_size = tb->shape[0];
  const int in_features = tb->shape[1];  // n_attn_heads * head_dim
  const int out_features = y->shape[1];  // hidden_dim

  const bf16 *w_o_ptr = (const bf16 *)W_o->d_buf + 1ll * layer_offset * out_features * in_features;
  const bf16 *b_o_ptr =
    has_bias ? (const bf16 *)b_o->d_buf + 1ll * layer_offset * out_features : nullptr;

  const bf16 *tb_ptr = (bf16 *)tb->d_buf;
  float *y_ptr = (float *)y->d_buf;

  {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 64;
    constexpr int TN = 32;
    constexpr int blockDim = 512;  // = 64 * (BM / TM) * (BN / TN)

    dim3 block_size(blockDim);
    dim3 grid_size((out_features + BN - 1) / BN, ((cur_batch_size + BM - 1) / BM));

    gemm_mfma_v2_bf16<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
      tb_ptr, w_o_ptr, y_ptr, b_o_ptr, cur_batch_size, out_features, in_features);
  }

  if (y_from_device) {
    y->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

void attn_out_project_batched_v2(Tensor *tb,         // Shape: [batch_size, n_attn_heads * head_dim]
                                 const Tensor *W_o,  // Shape: [n_attn_heads * head_dim, hidden_dim]
                                 const Tensor *b_o,  // Shape: [hidden_dim]
                                 Tensor *y,          // Shape: [batch_size, hidden_dim]
                                 bool has_bias, int cur_batch_size, long long layer_offset,
                                 bool tb_to_device, bool y_from_device, hipStream_t stream) {
  // GpuTimer timer("attn_out_project_v2", stream);
  if (tb_to_device) {
    tb->to_device(stream);
  }

  // const int batch_size = tb->shape[0];
  const int in_features = tb->shape[1];  // n_attn_heads * head_dim
  const int out_features = y->shape[1];  // hidden_dim

  const bf16 *w_o_ptr = (const bf16 *)W_o->d_buf + 1ll * layer_offset * out_features * in_features;
  const bf16 *b_o_ptr =
    has_bias ? (const bf16 *)b_o->d_buf + 1ll * layer_offset * out_features : nullptr;

  const bf16 *tb_ptr = (bf16 *)tb->d_buf;
  float *y_ptr = (float *)y->d_buf;

  {
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 64;
    constexpr int TN = 32;
    constexpr int blockDim = 256;  // = 64 * (BM / TM) * (BN / TN)

    dim3 block_size(blockDim);
    dim3 grid_size((out_features + BN - 1) / BN, ((cur_batch_size + BM - 1) / BM));

    gemm_mfma_v2_bf16<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
      tb_ptr, w_o_ptr, y_ptr, b_o_ptr, cur_batch_size, out_features, in_features);
  }

  if (y_from_device) {
    y->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

// ---------- Router GEMM ----------
void router_gemm_batched(const Tensor *w_router,  // [n_experts, hidden_dim]
                         Tensor *t,               // [B, hidden_dim]
                         const Tensor *b_router,  // [n_experts]
                         Tensor *router_scores,   // [B, n_experts]
                         int cur_batch_size, long long layer_offset, bool t_to_device,
                         bool r_from_device, hipStream_t stream) {
  // GpuTimer timer("router_gemm_batched", stream);
  if (t_to_device)
    t->to_device(stream);

  // const int B = (int)t->shape[0];
  const int H = (int)t->shape[1];
  const int E = (int)router_scores->shape[1];

  const float *W = (const float *)w_router->d_buf + 1ll * layer_offset * E * H;
  const float *b = (const float *)b_router->d_buf + 1ll * layer_offset * E;
  const float *X = (const float *)t->d_buf;
  float *R = (float *)router_scores->d_buf;

  // tuned like qkv/classifier
  constexpr int WARPS_PER_BLOCK = 16;
  constexpr int TILE = 1024;

  dim3 block_dim(warpSize, WARPS_PER_BLOCK);
  dim3 grid_dim((E + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, cur_batch_size);
  size_t shmem = TILE * sizeof(float);

  gemm_kernel_batched<WARPS_PER_BLOCK, TILE>
    <<<grid_dim, block_dim, shmem, stream>>>(W, X, b, R, E, H, cur_batch_size);

  CHECK_HIP(hipGetLastError());
  if (r_from_device) {
    router_scores->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

void router_gemm_batched_quantize(const Tensor *w_router,  // [n_experts, hidden_dim]
                                  Tensor *t,               // [B, hidden_dim]
                                  const Tensor *b_router,  // [n_experts]
                                  Tensor *router_scores,   // [B, n_experts]
                                  int cur_batch_size, long long layer_offset, bool t_to_device,
                                  bool r_from_device, hipStream_t stream) {
  // GpuTimer timer("router_gemm_v2", stream);
  if (t_to_device)
    t->to_device(stream);

  // const int batch_size = x->shape[0];      // M
  const int hidden_dim = t->shape[1];             // K
  const int n_experts = router_scores->shape[1];  // N

  const bf16 *t_ptr = (bf16 *)t->d_buf;
  const bf16 *w_router_ptr =
    (const bf16 *)w_router->d_buf + 1ll * layer_offset * n_experts * hidden_dim;
  const bf16 *b_router_ptr = (const bf16 *)b_router->d_buf + 1ll * layer_offset * n_experts;
  float *router_score_ptr = (float *)router_scores->d_buf;
  {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 64;
    constexpr int TM = 16;
    constexpr int TN = 16;
    constexpr int blockDim = 256;  // = 64 * (BM / TM) * (BN / TN)

    dim3 block_size(blockDim);
    dim3 grid_size((n_experts + BN - 1) / BN, ((cur_batch_size + BM - 1) / BM));

    gemm_mfma_v2_bf16<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
      t_ptr, w_router_ptr, router_score_ptr, b_router_ptr, cur_batch_size, n_experts, hidden_dim);
  }

  if (r_from_device) {
    router_scores->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

void router_gemm_v2(const Tensor *w_router,  // Shape: [n_layers, hidden_dim, n_experts]
                    Tensor *t,               // Shape: [batch_size, hidden_dim]
                    const Tensor *b_router,  // Shape: [n_layers, n_experts]
                    Tensor *router_scores,   // Shape: [batch_size, n_experts]
                    int cur_batch_size, long long layer_offset, bool t_to_device,
                    bool r_from_device, hipStream_t stream) {
  // GpuTimer timer("router_gemm_v2", stream);
  if (t_to_device)
    t->to_device(stream);

  // const int batch_size = x->shape[0];      // M
  const int hidden_dim = t->shape[1];             // K
  const int n_experts = router_scores->shape[1];  // N

  const bf16 *t_ptr = (bf16 *)t->d_buf;
  const bf16 *w_router_ptr =
    (const bf16 *)w_router->d_buf + 1ll * layer_offset * n_experts * hidden_dim;
  const bf16 *b_router_ptr = (const bf16 *)b_router->d_buf + 1ll * layer_offset * n_experts;
  float *router_score_ptr = (float *)router_scores->d_buf;
  {
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int TM = 32;
    constexpr int TN = 32;
    constexpr int blockDim = 512;  // = 64 * (BM / TM) * (BN / TN)

    dim3 block_size(blockDim);
    dim3 grid_size((n_experts + BN - 1) / BN, ((cur_batch_size + BM - 1) / BM));

    gemm_mfma_v2_bf16<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
      t_ptr, w_router_ptr, router_score_ptr, b_router_ptr, cur_batch_size, n_experts, hidden_dim);
  }

  if (r_from_device) {
    router_scores->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

// ---------- TopK + softmax(k) ----------
__global__ void batched_topk_softmax_kernel(const float *r, int batch_size, int n_experts, int k,
                                            float *topk_vals, int *topk_idx) {
  // Each block processes one item in the batch
  int batch_idx = blockIdx.x;

  // Guard against launching more blocks than batch items
  if (batch_idx >= batch_size) {
    return;
  }

  // Use a single thread (thread 0) within the block to perform the work,
  // mirroring the logic of the original kernel.
  if (threadIdx.x != 0) {
    return;
  }

  // Externally allocated shared memory for this block's data
  extern __shared__ char s_data[];
  float *temp_vals = (float *)s_data;
  int *temp_idx = (int *)(s_data + n_experts * sizeof(float));

  // Calculate offsets for the current batch item
  const float *r_batch = r + (size_t)batch_idx * n_experts;
  float *topk_vals_batch = topk_vals + (size_t)batch_idx * k;
  int *topk_idx_batch = topk_idx + (size_t)batch_idx * k;

  // 1. Copy input data for the current batch item to shared memory
  for (int i = 0; i < n_experts; ++i) {
    temp_vals[i] = r_batch[i];
    temp_idx[i] = i;
  }

  // 2. Partial selection sort to find top-k
  for (int i = 0; i < k; ++i) {
    int max_idx = i;
    // Find maximum in the remaining portion [i, n_experts)
    for (int j = i + 1; j < n_experts; ++j) {
      if (temp_vals[j] > temp_vals[max_idx]) {
        max_idx = j;
      }
    }
    // Swap position i with the maximum found
    if (max_idx != i) {
      // Swap values
      float temp_val = temp_vals[i];
      temp_vals[i] = temp_vals[max_idx];
      temp_vals[max_idx] = temp_val;

      // Swap indices
      int temp_index = temp_idx[i];
      temp_idx[i] = temp_idx[max_idx];
      temp_idx[max_idx] = temp_index;
    }
  }

  // 3. Apply softmax to the found top-k values
  double max_val = (k > 0) ? (double)temp_vals[0] : -DBL_MAX;

  double denom = 0.0;
  for (int i = 0; i < k; ++i) {
    denom += expf(temp_vals[i] - (float)max_val);
  }

  // 4. Output results to the correct slice in global memory
  for (int i = 0; i < k; ++i) {
    topk_vals_batch[i] = expf(temp_vals[i] - (float)max_val) / (float)denom;
    topk_idx_batch[i] = temp_idx[i];
  }
}

void topk_softmax_batched(Tensor *r,            // Shape: [batch_size, n_experts]
                          Tensor *topk_vals,    // Shape: [batch_size, k]
                          TensorI32 *topk_idx,  // Shape: [batch_size, k]
                          int cur_batch_size, bool r_to_device, bool topk_vals_from_device,
                          bool topk_idx_from_device, hipStream_t stream) {
  // GpuTimer timer("topk_softmax_batched", stream);
  if (r_to_device) {
    r->to_device(stream);
  }

  // Extract dimensions from tensor shapes
  // r shape: [batch_size, n_experts]
  // topk_idx shape: [batch_size, k]
  // const int batch_size = r->shape[0];
  const int num_experts = r->shape[1];
  const int experts_per_token = topk_idx->shape[1];  // This is 'k'

  // Get raw device pointers
  const float *r_ptr = (float *)r->d_buf;
  float *topk_vals_ptr = (float *)topk_vals->d_buf;
  int *topk_idx_ptr = topk_idx->d_buf;

  // Shared memory size is calculated per block, for one batch item's data
  size_t shared_mem_size = (size_t)num_experts * (sizeof(float) + sizeof(int));

  // Configure kernel launch:
  // Grid dimension is the batch_size to process all items in parallel.
  // Block dimension is 1 since one thread does the work within the block.
  dim3 grid_dim(cur_batch_size, 1, 1);
  dim3 block_dim(1, 1, 1);

  batched_topk_softmax_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
    r_ptr, cur_batch_size, num_experts, experts_per_token, topk_vals_ptr, topk_idx_ptr);

  CHECK_HIP(hipGetLastError());

  // Data transfer and synchronization logic remains the same
  if (topk_vals_from_device) {
    topk_vals->from_device(stream);
  }
  if (topk_idx_from_device) {
    topk_idx->from_device(stream);
  }
  if (topk_vals_from_device || topk_idx_from_device) {
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

// ---------- MoE----------
static inline void moe_init_buffers_hip(Tensor *e_agg, Tensor *mlp1_out, Tensor *gate_up,
                                        Tensor *tb3, TensorI32 *sorted_pair_ids,
                                        TensorI32 *expert_offsets, Tensor *x_packed, int batch_size,
                                        int hidden_dim, hipStream_t stream) {
  // GpuTimer timer("moe_init_buffers", stream);
  moe_init_buffers(e_agg, mlp1_out, gate_up, tb3, sorted_pair_ids, expert_offsets, x_packed,
                   batch_size, hidden_dim, stream);
}

static inline void moe_build_offsets_hip(
  TensorI32 *topk_idx,         // [batch_size, experts_per_token]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  TensorI32 *expert_offsets,   // [n_experts + 1]
  int batch_size, int experts_per_token, int n_experts, hipStream_t stream) {
  // GpuTimer timer("moe_offsets", stream);

  moe_build_offsets(topk_idx, sorted_pair_ids, expert_offsets, batch_size, experts_per_token,
                    n_experts, stream);
}

static inline void moe_pack_inputs_hip(
  Tensor *x_in,                // [batch_size, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *x_packed,            // [batch_size * experts_per_token, hidden_dim]
  int batch_size, int hidden_dim, int experts_per_token, hipStream_t stream) {
  // GpuTimer timer("moe_x_packed", stream);
  moe_pack_inputs(x_in, sorted_pair_ids, x_packed, batch_size, hidden_dim, experts_per_token,
                  stream);
}

static inline void moe_pack_inputs_hip_quantize(
  Tensor *x_in,                // [batch_size, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *x_packed,            // [batch_size * experts_per_token, hidden_dim]
  int batch_size, int hidden_dim, int experts_per_token, hipStream_t stream) {
  // GpuTimer timer("moe_x_packed", stream);
  moe_pack_inputs_bf16(x_in, sorted_pair_ids, x_packed, batch_size, hidden_dim, experts_per_token,
                       stream);
}

static inline int moe_get_max_rows_per_expert_hip(TensorI32 *expert_offsets, int *d_max_rows,
                                                  int n_experts, hipStream_t stream) {
  // GpuTimer timer("moe_max_row", stream);
  return moe_get_max_rows_per_expert(expert_offsets, d_max_rows, n_experts, stream);
}

static inline void moe_mlp1_forward_hip(Tensor *x_packed,  // [total_pairs, hidden_dim]
                                        Tensor *w_mlp1, Tensor *b_mlp1,
                                        TensorI32 *expert_offsets,  // [n_experts+1]
                                        Tensor *mlp1_out,           // [total_pairs, 2*inter_dim]
                                        long long layer_offset, int n_experts, int hidden_dim,
                                        int inter_dim, int max_rows_per_expert, int total_pairs,
                                        hipStream_t stream) {
  // GpuTimer timer("moe_mlp1", stream);
  moe_mlp1_forward(x_packed, w_mlp1, b_mlp1, expert_offsets, mlp1_out, layer_offset, n_experts,
                   hidden_dim, inter_dim, max_rows_per_expert, total_pairs, stream);
}

static inline void moe_mlp1_forward_hip_quantize(Tensor *x_packed,  // [total_pairs, hidden_dim]
                                                 Tensor *w_mlp1, Tensor *b_mlp1,
                                                 TensorI32 *expert_offsets,  // [n_experts+1]
                                                 Tensor *mlp1_out,  // [total_pairs, 2*inter_dim]
                                                 long long layer_offset, int n_experts,
                                                 int hidden_dim, int inter_dim,
                                                 int max_rows_per_expert, int total_pairs,
                                                 hipStream_t stream) {
  // GpuTimer timer("moe_mlp1", stream);
  moe_mlp1_forward_bf16(x_packed, w_mlp1, b_mlp1, expert_offsets, mlp1_out, layer_offset, n_experts,
                        hidden_dim, inter_dim, max_rows_per_expert, total_pairs, stream);
}

static inline void moe_mlp1_swiglu_fused_hip(Tensor *x_packed, Tensor *w_mlp1, Tensor *b_mlp1,
                                             TensorI32 *expert_offsets, Tensor *gate_up,
                                             long long layer_offset, int n_experts, int hidden_dim,
                                             int inter_dim, float clamp_limit,
                                             int max_rows_per_expert, int total_pairs,
                                             hipStream_t stream) {
  // GpuTimer timer("moe_mlp1_swiglu_fused", stream);

  moe_mlp1_swiglu_fused(x_packed, w_mlp1, b_mlp1, expert_offsets, gate_up, layer_offset, n_experts,
                        hidden_dim, inter_dim, clamp_limit, max_rows_per_expert, total_pairs,
                        stream);
}

static inline void moe_swiglu_hip(Tensor *mlp1_out,  // [total_pairs, 2*inter_dim]
                                  Tensor *gate_up,   // [total_pairs, inter_dim]
                                  int batch_size, int experts_per_token, int inter_dim,
                                  float clamp_limit, hipStream_t stream) {
  // GpuTimer timer("moe_swiglu", stream);
  moe_swiglu(mlp1_out, gate_up, batch_size, experts_per_token, inter_dim, clamp_limit, stream);
}

static inline void moe_swiglu_hip_quantize(Tensor *mlp1_out,  // [total_pairs, 2*inter_dim]
                                           Tensor *gate_up,   // [total_pairs, inter_dim]
                                           int batch_size, int experts_per_token, int inter_dim,
                                           float clamp_limit, hipStream_t stream) {
  // GpuTimer timer("moe_swiglu", stream);
  moe_swiglu_bf16(mlp1_out, gate_up, batch_size, experts_per_token, inter_dim, clamp_limit, stream);
}

static inline void moe_mlp2_forward_hip(Tensor *gate_up,  // [total_pairs, inter_dim]
                                        Tensor *w_mlp2, Tensor *b_mlp2,
                                        TensorI32 *expert_offsets,  // [n_experts+1]
                                        Tensor *tb3,                // [total_pairs, hidden_dim]
                                        bool has_bias, long long layer_offset, int n_experts,
                                        int inter_dim, int hidden_dim, int max_rows_per_expert,
                                        int total_pairs, hipStream_t stream) {
  // GpuTimer timer("moe_mlp2", stream);
  moe_mlp2_forward(gate_up, w_mlp2, b_mlp2, expert_offsets, tb3, has_bias, layer_offset, n_experts,
                   inter_dim, hidden_dim, max_rows_per_expert, total_pairs, stream);
}

static inline void moe_mlp2_forward_hip_quantize(Tensor *gate_up,  // [total_pairs, inter_dim]
                                                 Tensor *w_mlp2, Tensor *b_mlp2,
                                                 TensorI32 *expert_offsets,  // [n_experts+1]
                                                 Tensor *tb3,  // [total_pairs, hidden_dim]
                                                 bool has_bias, long long layer_offset,
                                                 int n_experts, int inter_dim, int hidden_dim,
                                                 int max_rows_per_expert, int total_pairs,
                                                 hipStream_t stream) {
  // GpuTimer timer("moe_mlp2", stream);
  moe_mlp2_forward_bf16(gate_up, w_mlp2, b_mlp2, expert_offsets, tb3, has_bias, layer_offset,
                        n_experts, inter_dim, hidden_dim, max_rows_per_expert, total_pairs, stream);
}

static inline void moe_scatter_aggregate_hip(
  Tensor *tb3,                 // [total_pairs, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *topk_v,              // [batch_size, experts_per_token]
  Tensor *e_agg,               // [batch_size, hidden_dim]
  TensorI32 *expert_offsets,   // [n_experts+1]
  int hidden_dim, int experts_per_token, int n_experts, int max_rows_per_expert,
  hipStream_t stream) {
  // GpuTimer timer("moe_agg", stream);
  moe_scatter_aggregate(tb3, sorted_pair_ids, topk_v, e_agg, expert_offsets, hidden_dim,
                        experts_per_token, n_experts, max_rows_per_expert, stream);
}

static inline void moe_scatter_aggregate_ep_hip(
  Tensor *tb3,                 // [total_pairs, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *topk_v,              // [batch_size, experts_per_token]
  Tensor *e_agg,               // [batch_size, hidden_dim]
  TensorI32 *expert_offsets,   // [n_experts+1]
  int hidden_dim, int experts_per_token, int n_experts, int max_rows_per_expert,
  int start_expert_offset, int end_expert_offset, hipStream_t stream) {
  // GpuTimer timer("moe_agg_v2", stream);

  moe_scatter_aggregate_120b_ep(tb3, sorted_pair_ids, topk_v, e_agg, expert_offsets, hidden_dim,
                                experts_per_token, n_experts, max_rows_per_expert,
                                start_expert_offset, end_expert_offset, stream);
}

__global__ void quantize_kernel(const float *in, bf16 *out, int num_elems) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_elems) {
    return;
  }

  out[idx] = bf16(in[idx]);
}

__global__ void dequantize_kernel(const bf16 *in, float *out, int num_elems) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_elems) {
    return;
  }

  out[idx] = (float)in[idx];
}

static inline void tensor_quantize(Tensor *src, Tensor *dst_quantize, hipStream_t stream) {
  // GpuTimer timer("tensor_quantize", stream);
  const size_t num_elems = src->num_elem();

  dim3 block_size(256);
  dim3 grid_size((num_elems + block_size.x - 1) / block_size.x);

  quantize_kernel<<<grid_size, block_size, 0, stream>>>((const float *)src->d_buf,
                                                        (bf16 *)dst_quantize->d_buf, num_elems);
}

static inline void tensor_dequantize(Tensor *src, Tensor *dst_quantize, hipStream_t stream) {
  // GpuTimer timer("tensor_dequantize", stream);
  const size_t num_elems = src->num_elem();

  dim3 block_size(256);
  dim3 grid_size((num_elems + block_size.x - 1) / block_size.x);

  dequantize_kernel<<<grid_size, block_size, 0, stream>>>((const bf16 *)dst_quantize->d_buf,
                                                          (float *)src->d_buf, num_elems);
}

/* FP8 Quantization */
__device__ __forceinline__ fp8 convert_float_to_fp8(float in, __hip_fp8_interpretation_t interpret,
                                                    __hip_saturation_t sat) {
  return __hip_cvt_float_to_fp8(in, sat, interpret);
}

__device__ __forceinline__ float convert_fp8_to_float(fp8 in,
                                                      __hip_fp8_interpretation_t interpret) {
  __half hf = __hip_cvt_fp8_to_halfraw(in, interpret);
  return hf;
}

__global__ void quantize_kernel_fp8(const float *in, fp8 *out, int num_elems) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_elems) {
    return;
  }

  out[idx] = convert_float_to_fp8(in[idx], __HIP_E5M2_FNUZ, __HIP_SATFINITE);
}

__global__ void dequantize_kernel_fp8(const fp8 *in, float *out, int num_elems) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_elems) {
    return;
  }

  out[idx] = convert_fp8_to_float(in[idx], __HIP_E5M2_FNUZ);
}

static inline void tensor_quantize_fp8(Tensor *src, Tensor *dst_quantize, hipStream_t stream) {
  // GpuTimer timer("tensor_quantize_fp8", stream);
  const size_t num_elems = src->num_elem();

  dim3 block_size(256);
  dim3 grid_size((num_elems + block_size.x - 1) / block_size.x);
  quantize_kernel_fp8<<<grid_size, block_size, 0, stream>>>((const float *)src->d_buf,
                                                            (fp8 *)dst_quantize->d_buf, num_elems);
}

static inline void tensor_dequantize_fp8(Tensor *src, Tensor *dst_quantize, hipStream_t stream) {
  // GpuTimer timer("tensor_dequantize_fp8", stream);

  const size_t num_elems = src->num_elem();

  dim3 block_size(256);
  dim3 grid_size((num_elems + block_size.x - 1) / block_size.x);

  dequantize_kernel_fp8<<<grid_size, block_size, 0, stream>>>((const fp8 *)dst_quantize->d_buf,
                                                              (float *)src->d_buf, num_elems);
}
/* FP8 Quantization */

static inline void moe_scatter_aggregate_hip_120b(
  Tensor *tb3,                 // [total_pairs, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *topk_v,              // [batch_size, experts_per_token]
  Tensor *e_agg,               // [batch_size, hidden_dim]
  TensorI32 *expert_offsets,   // [n_experts+1]
  int hidden_dim, int experts_per_token, int n_experts, int max_rows_per_expert,
  hipStream_t stream) {
  // GpuTimer timer("moe_agg_120b", stream);
  moe_scatter_aggregate_120b(tb3, sorted_pair_ids, topk_v, e_agg, expert_offsets, hidden_dim,
                             experts_per_token, n_experts, max_rows_per_expert, stream);
}

void classifier_gemm_batched_v2_quantize(const Tensor *W_out,  // Shape: [hidden_dim, vocab_size]
                                         Tensor *x,            // Shape: [batch_size, hidden_dim]
                                         Tensor *logits,       // Shape: [batch_size, vocab_size]
                                         int cur_batch_size, bool x_to_device,
                                         bool logits_from_device, hipStream_t stream) {
  // GpuTimer timer("classifier_v2", stream);
  if (x_to_device) {
    x->to_device(stream);
  }

  // const int batch_size = x->shape[0];
  const int hidden_dim = x->shape[1];
  const int vocab_size = W_out->shape[1];

  const bf16 *w_out_ptr = (const bf16 *)W_out->d_buf;
  const bf16 *x_ptr = (const bf16 *)x->d_buf;
  float *logits_buf = (float *)logits->d_buf;

  {
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int TM = 64;
    constexpr int TN = 64;
    constexpr int blockDim = 512;  // = 64 * (BM / TM) * (BN / TN)

    dim3 block_size(blockDim);
    dim3 grid_size((vocab_size + BN - 1) / BN, ((cur_batch_size + BM - 1) / BM));

    gemm_mfma_v2_bf16<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
      x_ptr, w_out_ptr, logits_buf, nullptr, cur_batch_size, vocab_size, hidden_dim);
  }

  if (logits_from_device) {
    logits->from_device(stream);
  }
}

void classifier_gemm_batched_v2(const Tensor *W_out,  // Shape: [hidden_dim, vocab_size]
                                Tensor *x,            // Shape: [batch_size, hidden_dim]
                                Tensor *logits,       // Shape: [batch_size, vocab_size]
                                int cur_batch_size, bool x_to_device, bool logits_from_device,
                                hipStream_t stream) {
  // GpuTimer timer("classifier_v2", stream);
  if (x_to_device) {
    x->to_device(stream);
  }

  // const int batch_size = x->shape[0];
  const int hidden_dim = x->shape[1];
  const int vocab_size = W_out->shape[1];

  const bf16 *w_out_ptr = (const bf16 *)W_out->d_buf;
  const bf16 *x_ptr = (const bf16 *)x->d_buf;
  float *logits_buf = (float *)logits->d_buf;

  {
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int TM = 64;
    constexpr int TN = 64;
    constexpr int blockDim = 512;  // = 64 * (BM / TM) * (BN / TN)

    dim3 block_size(blockDim);
    dim3 grid_size((vocab_size + BN - 1) / BN, ((cur_batch_size + BM - 1) / BM));

    gemm_mfma_v2_bf16<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
      x_ptr, w_out_ptr, logits_buf, nullptr, cur_batch_size, vocab_size, hidden_dim);
  }
  if (logits_from_device) {
    logits->from_device(stream);
  }
}

// The GPU kernel function. This is the code that will run in parallel on the GPU.
// The __global__ specifier indicates that it's a kernel.
__global__ void max_kernel_gpu_v2(const float *__restrict__ logits, float *__restrict__ logits_max,
                                  int *__restrict__ id_max, int batch_size, int length,
                                  int tp_rank) {
  // One block = one row (batch element)
  int row = blockIdx.x;
  if (row >= batch_size)
    return;

  // Shared memory for block reduction
  extern __shared__ char new_smem[];
  float *sdata_val = (float *)new_smem;
  int *sdata_idx = (int *)(sdata_val + blockDim.x);

  int tid = threadIdx.x;
  int start = row * length;

  // Each thread scans a strided portion of this row
  float local_max = -FLT_MAX;
  int local_idx = -1;
  for (int j = tid; j < length; j += blockDim.x) {
    float val = logits[start + j];
    if (val > local_max) {
      local_max = val;
      local_idx = j;
    }
  }

  // Store partial results into shared memory
  sdata_val[tid] = local_max;
  sdata_idx[tid] = local_idx;
  __syncthreads();

  // Parallel reduction to find max and index
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata_val[tid + s] > sdata_val[tid]) {
        sdata_val[tid] = sdata_val[tid + s];
        sdata_idx[tid] = sdata_idx[tid + s];
      }
    }
    __syncthreads();
  }

  // Thread 0 writes the result for this row
  if (tid == 0) {
    int final_idx = sdata_idx[0];
    float final_max = sdata_val[0];

    if (final_idx < 0) {
      final_idx = 0;
    }

    logits_max[row] = final_max;
    id_max[row] = final_idx + tp_rank * length;
  }
}

void max_logits_batched(Tensor *logits, Tensor *logits_max, TensorI32 *logits_id,
                        int cur_batch_size, int tp_rank, bool logits_to_device,
                        bool logits_max_from_device, bool logits_id_from_device,
                        hipStream_t stream) {
  // GpuTimer timer("max_logits", stream);
  if (logits_to_device) {
    logits->to_device(stream);
  }

  const size_t num_elems = 1ll * logits->shape[1];

  const float *logits_buf = (const float *)logits->d_buf;
  float *logits_max_buf = (float *)logits_max->d_buf;
  int *logits_id_buf = (int *)logits_id->d_buf;

  int blockSize = 256;  // tune: 128/256/512 depending on GPU
  dim3 gridSize(cur_batch_size);
  size_t sharedMemSize = blockSize * (sizeof(float) + sizeof(int));

  max_kernel_gpu_v2<<<gridSize, blockSize, sharedMemSize, stream>>>(
    logits_buf, logits_max_buf, logits_id_buf, cur_batch_size, num_elems, tp_rank);

  if (logits_max_from_device) {
    logits_max->from_device(stream);
  }
  if (logits_id_from_device) {
    logits_id->from_device(stream);
  }
  if (logits_max_from_device || logits_id_from_device) {
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

__global__ void reduceLogitsKernel(const float *__restrict__ logits_max_total,
                                   const int *__restrict__ logits_id_total,
                                   float *__restrict__ logits_max_final,
                                   int *__restrict__ logits_id_final, int cur_batch_size,
                                   int batch_size_total, int tp_total) {
  // Calculate the global thread index, which corresponds to the batch index.
  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check to ensure we don't access out-of-bounds memory.
  if (batch_idx >= cur_batch_size) {
    return;
  }

  // Initialize the maximum value to a very small number and the ID.
  float max_val = -FLT_MAX;
  int max_id = -1;

  // Loop through the 'tp' dimension to find the maximum value and its ID.
  for (int tp_idx = 0; tp_idx < tp_total; ++tp_idx) {
    // Calculate the 1D index for the 2D array.
    long long idx = 1ll * tp_idx * batch_size_total + 1ll * batch_idx;

    float current_val = logits_max_total[idx];

    // If the current value is greater than the current max, update.
    if (current_val > max_val) {
      max_val = current_val;
      max_id = logits_id_total[idx];
    }
  }

  if (max_id < 0) {
    max_id = 0;
  }

  // Store the final maximum value and ID in the output arrays.
  logits_max_final[batch_idx] = max_val;
  logits_id_final[batch_idx] = max_id;
}

void reduce_logits_batched(Tensor *logits_max_total, TensorI32 *logits_id_total, Tensor *logits_max,
                           TensorI32 *logits_id, int cur_batch_size,
                           bool logits_max_total_to_device, bool logits_id_total_to_device,
                           bool logits_max_from_device, bool logits_id_from_device,
                           hipStream_t stream) {
  // GpuTimer timer("reduce_logits", stream);

  if (logits_max_total_to_device) {
    logits_max_total->to_device(stream);
  }
  if (logits_id_total_to_device) {
    logits_id_total->to_device(stream);
  }

  const size_t batch_size_total = 1ll * logits_max_total->shape[1];
  const size_t tp_total = 1ll * logits_max_total->shape[0];

  const float *logits_max_total_buf = (const float *)logits_max_total->d_buf;
  const int *logits_id_total_buf = (const int *)logits_id_total->d_buf;
  float *logits_max_buf = (float *)logits_max->d_buf;
  int *logits_id_buf = (int *)logits_id->d_buf;

  int blockSize = 1;
  dim3 threads(blockSize);
  dim3 blocks((cur_batch_size + blockSize - 1) / blockSize);

  // --- Launch the Kernel ---
  reduceLogitsKernel<<<blocks, threads, 0, stream>>>(logits_max_total_buf, logits_id_total_buf,
                                                     logits_max_buf, logits_id_buf, cur_batch_size,
                                                     batch_size_total, tp_total);

  if (logits_max_from_device) {
    logits_max->from_device(stream);
  }
  if (logits_id_from_device) {
    logits_id->from_device(stream);
  }
  if (logits_max_from_device || logits_id_from_device) {
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}
