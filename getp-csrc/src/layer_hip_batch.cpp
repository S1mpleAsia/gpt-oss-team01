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

void qkv_gemm_batched(Tensor *x,            // Shape: [batch_size, hidden_dim]
                      const Tensor *W_qkv,  // Shape: [out_features, hidden_dim]
                      const Tensor *b_qkv,  // Shape: [out_features]
                      Tensor *qkv,          // Shape: [batch_size, out_features]
                      int cur_batch_size, long long layer_offset, bool x_to_device,
                      bool qkv_from_device, hipStream_t stream) {
  // GpuTimer timer("qkv_gemm", stream);
  if (x_to_device) {
    x->to_device(stream);
  }
  // Assuming Tensor has a shape member or method, e.g., x->shape[0]
  // const int batch_size = x->shape[0];      // M
  const int in_features = x->shape[1];     // K
  const int out_features = qkv->shape[1];  // N

  const float *x_ptr = (float *)x->d_buf;
  const float *w_qkv_ptr = (float *)W_qkv->d_buf + 1ll * layer_offset * out_features * in_features;
  const float *b_qkv_ptr = (float *)b_qkv->d_buf + 1ll * layer_offset * out_features;
  float *qkv_ptr = (float *)qkv->d_buf;

  constexpr int WARPS_PER_BLOCK = 16;
  constexpr int TILE = 1024;

  dim3 block_dim(warpSize, WARPS_PER_BLOCK);

  // --- MODIFIED: Use a 2D grid to parallelize across both output features and batch size ---
  dim3 grid_dim((out_features + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, cur_batch_size);
  size_t shmem_bytes = TILE * sizeof(float);

  // Launch the modified kernel with the new batch_size parameter
  gemm_kernel_batched<WARPS_PER_BLOCK, TILE><<<grid_dim, block_dim, shmem_bytes, stream>>>(
    w_qkv_ptr, x_ptr, b_qkv_ptr, qkv_ptr, out_features, in_features, cur_batch_size);

  if (qkv_from_device) {
    qkv->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
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

  const float *x_ptr = (float *)x->d_buf;
  const bf16 *w_qkv_ptr =
    (const bf16 *)W_qkv->d_buf + 1ll * layer_offset * out_features * in_features;
  const bf16 *b_qkv_ptr = (const bf16 *)b_qkv->d_buf + 1ll * layer_offset * out_features;
  float *qkv_ptr = (float *)qkv->d_buf;
  {
#ifdef RUN_20B
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 64;
    constexpr int TN = 32;
    constexpr int blockDim = 512;  // = 64 * (BM / TM) * (BN / TN)
#else
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 32;
    constexpr int TN = 32;
    constexpr int blockDim = 512;  // = 64 * (BM / TM) * (BN / TN)
#endif

    dim3 block_size(blockDim);
    dim3 grid_size((out_features + BN - 1) / BN, ((cur_batch_size + BM - 1) / BM));

#ifdef RUN_20B
    gemm_mfma_v2<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
      x_ptr, w_qkv_ptr, qkv_ptr, b_qkv_ptr, cur_batch_size, out_features, in_features);
#else
    gemm_mfma_v2<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
      x_ptr, w_qkv_ptr, qkv_ptr, b_qkv_ptr, cur_batch_size, out_features, in_features);
#endif
  }

  // {
  //   constexpr int BM = 16;
  //   constexpr int BN = 128;
  //   constexpr int BK = 16;
  //   constexpr int TM = 1;
  //   constexpr int TN = 4;

  //   const int BLOCK_SIZE_X = BN / TN;
  //   const int BLOCK_SIZE_Y = BM / TM;
  //   dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  //   dim3 grid_size((out_features + BN - 1) / BN, (batch_size + BM - 1) / BM);

  //   matmul_kernel<BM, BN, BK, TM, TN><<<grid_size, block_size, 0, stream>>>(
  //     x_ptr, w_qkv_ptr, qkv_ptr, b_qkv_ptr, batch_size, out_features, in_features);
  //   CHECK_HIP(hipGetLastError());
  // }

  if (qkv_from_device) {
    qkv->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

// ---------- Split & RoPE ----------
__global__ void qkv_split_rope_kernel_batched(float *qkv_out, float *q_out, float *k_pos,
                                              float *v_pos, const float *rope_cos_pos,
                                              const float *rope_sin_pos, int head_dim, int n_q,
                                              int n_kv, int batch_size) {  // Added batch_size

  // --- MODIFIED: 2D indexing for batch and feature dimensions ---
  int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y;

  // --- MODIFIED: Bounds check for both dimensions ---
  int q_dims = n_q * head_dim;
  int k_dims = n_kv * head_dim;
  int v_dims = n_kv * head_dim;
  int total_dims_per_batch = q_dims + k_dims + v_dims;

  if (batch_idx >= batch_size || feature_idx >= total_dims_per_batch) {
    return;
  }

  // --- MODIFIED: Offset base pointers for the current batch item ---
  float *qkv_batch = qkv_out + (size_t)batch_idx * total_dims_per_batch;
  float *q_batch = q_out + (size_t)batch_idx * q_dims;
  float *k_batch = k_pos + (size_t)batch_idx * k_dims;
  float *v_batch = v_pos + (size_t)batch_idx * v_dims;

  int half_dim = head_dim / 2;

  // The core logic remains the same but uses the new batched pointers and feature_idx
  if (feature_idx < q_dims) {  // Processing a Q element
    int dim_idx_in_head = feature_idx % head_dim;
    int rope_idx = dim_idx_in_head % half_dim;

    float cos_val = rope_cos_pos[rope_idx];
    float sin_val = rope_sin_pos[rope_idx];

    float partner_val;
    if (dim_idx_in_head < half_dim) {
      partner_val = qkv_batch[feature_idx + half_dim];
      q_batch[feature_idx] = qkv_batch[feature_idx] * cos_val - partner_val * sin_val;
    } else {
      partner_val = qkv_batch[feature_idx - half_dim];
      q_batch[feature_idx] = qkv_batch[feature_idx] * cos_val + partner_val * sin_val;
    }

  } else if (feature_idx < q_dims + k_dims) {  // Processing a K element
    int k_local_idx = feature_idx - q_dims;
    int dim_idx_in_head = k_local_idx % head_dim;
    int rope_idx = dim_idx_in_head % half_dim;

    float cos_val = rope_cos_pos[rope_idx];
    float sin_val = rope_sin_pos[rope_idx];

    float partner_val;
    if (dim_idx_in_head < half_dim) {
      partner_val = qkv_batch[feature_idx + half_dim];
      k_batch[k_local_idx] = qkv_batch[feature_idx] * cos_val - partner_val * sin_val;
    } else {
      partner_val = qkv_batch[feature_idx - half_dim];
      k_batch[k_local_idx] = qkv_batch[feature_idx] * cos_val + partner_val * sin_val;
    }

  } else {  // Processing a V element (simple copy)
    int v_local_idx = feature_idx - q_dims - k_dims;
    v_batch[v_local_idx] = qkv_batch[feature_idx];
  }
}

void qkv_split_rope_batched(Tensor *qkv_out,             // Shape: [batch_size, (n_q + 2*n_kv)*hd]
                            Tensor *q_out,               // Shape: [batch_size, n_q*hd]
                            Tensor *k_pos,               // Shape: [batch_size, n_kv*hd]
                            Tensor *v_pos,               // Shape: [batch_size, n_kv*hd]
                            const Tensor *rope_cos_pos,  // Shape: [seq_len, hd/2]
                            const Tensor *rope_sin_pos,  // Shape: [seq_len, hd/2]
                            int cur_batch_size, int head_dim, int n_q, int n_kv, int pos,
                            bool qkv_out_to_device, bool q_out_from_device, bool k_out_from_device,
                            bool v_out_from_device, hipStream_t stream) {
  // GpuTimer timer("qkv_split_rope", stream);
  if (qkv_out_to_device) {
    qkv_out->to_device(stream);
  }

  // RoPE table offset is unchanged
  int offset = pos * (head_dim / 2);

  float *qkv_out_ptr = (float *)qkv_out->d_buf;
  float *q_out_ptr = (float *)q_out->d_buf;
  float *k_pos_ptr = (float *)k_pos->d_buf;
  float *v_pos_ptr = (float *)v_pos->d_buf;
  const float *rope_cos_pos_ptr = (float *)rope_cos_pos->d_buf + 1ll * offset;
  const float *rope_sin_pos_ptr = (float *)rope_sin_pos->d_buf + 1ll * offset;

  int total_dims_per_batch = (n_q + 2 * n_kv) * head_dim;
  dim3 block_dim(DEFAULT_BLOCK_SIZE);

  // --- MODIFIED: Launch a 2D grid ---
  // grid.x covers the feature dimension
  // grid.y covers the batch dimension
  dim3 grid_dim((total_dims_per_batch + block_dim.x - 1) / block_dim.x, cur_batch_size);

  qkv_split_rope_kernel_batched<<<grid_dim, block_dim, 0, stream>>>(
    qkv_out_ptr, q_out_ptr, k_pos_ptr, v_pos_ptr, rope_cos_pos_ptr, rope_sin_pos_ptr, head_dim, n_q,
    n_kv, cur_batch_size);  // Pass batch_size to kernel

  if (q_out_from_device)
    q_out->from_device(stream);
  if (k_out_from_device)
    k_pos->from_device(stream);
  if (v_out_from_device)
    v_pos->from_device(stream);
  if (q_out_from_device || k_out_from_device || v_out_from_device) {
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

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
  const size_t t_base = (size_t)pos * (size_t)(n_kv * head_dim);

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
  void *k_ptr = (char *)K_cache->d_buf + layer_offset * seq_len * kv_dim * elem_bytes;
  void *v_ptr = (char *)V_cache->d_buf + layer_offset * seq_len * kv_dim * elem_bytes;

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

// ---------- Attention ----------
__global__ void batched_attention_kernel_opt(
  const float *__restrict__ q,           // [B, n_q*hd]
  const float *__restrict__ K_cache,     // [B, L, S, kv_dim]  (host already offset to this layer)
  const float *__restrict__ V_cache,     // [B, L, S, kv_dim]  (host already offset to this layer)
  const float *__restrict__ mask,        // [B, S, S] or nullptr
  const float *__restrict__ attn_sinks,  // [n_q] (for this layer)
  float *__restrict__ tb,                // [B, n_q*hd]
  int head_dim, int n_q, int kv_mul, int kv_dim, int batch_size, int pos, int total_seq_len,
  int n_layers) {
  const int head_idx = blockIdx.x;   // 0..n_q-1
  const int batch_idx = blockIdx.y;  // 0..B-1
  if (head_idx >= n_q || batch_idx >= batch_size)
    return;

  const int attn_len = pos + 1;               // tokens to attend (0..pos)
  const int kv_head_idx = head_idx / kv_mul;  // GQA mapping
  const float scale = rsqrtf((float)head_dim);

  // Per-batch base pointers
  const size_t qtb_stride = (size_t)n_q * head_dim;  // stride in q/tb
  const size_t kv_batch_stride =
    (size_t)n_layers * total_seq_len * kv_dim;  // stride between batches in KV

  const float *__restrict__ q_head =
    q + (size_t)batch_idx * qtb_stride + (size_t)head_idx * head_dim;
  float *__restrict__ tb_head = tb + (size_t)batch_idx * qtb_stride + (size_t)head_idx * head_dim;

  const float *__restrict__ K_base = K_cache + (size_t)batch_idx * kv_batch_stride;
  const float *__restrict__ V_base = V_cache + (size_t)batch_idx * kv_batch_stride;

  // Mask row pointer if provided
  const float *__restrict__ mask_row = nullptr;
  if (mask) {
    const size_t mask_batch_stride = (size_t)total_seq_len * total_seq_len;
    mask_row = mask + (size_t)batch_idx * mask_batch_stride + (size_t)pos * total_seq_len;
  }

  // ---- Shared memory layout ----
  extern __shared__ unsigned char sdata[];
  float *s_q = (float *)sdata;                  // [hd]
  float *s_scores = (float *)(s_q + head_dim);  // [attn_len + 1] (last slot = sink)
  double *s_red =
    (double *)(s_scores + (size_t)attn_len + 1);  // [blockDim.x] scratch for reductions

  // 1) Stage q into shared once (all threads reuse)
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    s_q[i] = q_head[i];
  __syncthreads();

  // 2) Compute raw scores in parallel
  double local_max = -DBL_MAX;
  for (int t = threadIdx.x; t < attn_len; t += blockDim.x) {
    const float *__restrict__ k_vec = K_base + (size_t)t * kv_dim + (size_t)kv_head_idx * head_dim;

    double score = 0.0;
    int i = 0;
    for (; i + 3 < head_dim; i += 4) {
      float q0 = s_q[i + 0], q1 = s_q[i + 1], q2 = s_q[i + 2], q3 = s_q[i + 3];
      float k0 = k_vec[i + 0], k1 = k_vec[i + 1], k2 = k_vec[i + 2], k3 = k_vec[i + 3];
      score += (double)q0 * (double)k0 + (double)q1 * (double)k1 + (double)q2 * (double)k2 +
               (double)q3 * (double)k3;
    }
    for (; i < head_dim; ++i)
      score += (double)s_q[i] * (double)k_vec[i];

    score *= (double)scale;
    if (mask_row)
      score += (double)mask_row[t];

    s_scores[t] = (float)score;
    if (score > local_max)
      local_max = score;
  }

  // add sink into max via thread 0; reduce max across block
  s_red[threadIdx.x] = local_max;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      s_red[threadIdx.x] = fmax(s_red[threadIdx.x], s_red[threadIdx.x + s]);
    __syncthreads();
  }
  double max_score = s_red[0];
  if (threadIdx.x == 0) {
    const double sink = (double)attn_sinks[head_idx];
    s_scores[attn_len] = (float)sink;  // store sink at tail
    if (sink > max_score)
      max_score = sink;
    s_red[0] = max_score;  // broadcast via shared
  }
  __syncthreads();
  max_score = s_red[0];

  // 3) Exponentiate and reduce denominator in parallel
  double local_sum = 0.0;
  for (int t = threadIdx.x; t < attn_len; t += blockDim.x) {
    float e = expf(s_scores[t] - (float)max_score);
    s_scores[t] = e;  // keep e_t; we will normalize later
    local_sum += (double)e;
  }
  s_red[threadIdx.x] = local_sum;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      s_red[threadIdx.x] += s_red[threadIdx.x + s];
    __syncthreads();
  }
  double denom = s_red[0];
  if (threadIdx.x == 0) {
    denom += exp((double)s_scores[attn_len] - max_score);  // include sink prob in denom
    s_red[0] = denom;                                      // broadcast denom
  }
  __syncthreads();
  denom = s_red[0];
  const double inv_denom = 1.0 / denom;

  // 4) Normalize in-place: p_t = e_t / denom (exclude sink element)
  for (int t = threadIdx.x; t < attn_len; t += blockDim.x)
    s_scores[t] = (float)((double)s_scores[t] * inv_denom);
  __syncthreads();

  // 5) Weighted sum of V across head_dim in parallel
  const size_t kv_head_off = (size_t)kv_head_idx * head_dim;
  for (int i_out = threadIdx.x; i_out < head_dim; i_out += blockDim.x) {
    double acc = 0.0;
    for (int t = 0; t < attn_len; ++t) {
      const float p = s_scores[t];  // normalized prob
      const float *__restrict__ v_vec = V_base + (size_t)t * kv_dim + kv_head_off;
      acc += (double)p * (double)v_vec[i_out];
    }
    tb_head[i_out] = (float)acc;
  }
}

void single_query_attn_batched(Tensor *q,           // [B, n_q*hd]
                               Tensor *K_cache,     // [B, L, S, kv_dim]
                               Tensor *V_cache,     // [B, L, S, kv_dim]
                               Tensor *mask,        // [B, S, S] or nullptr
                               Tensor *attn_sinks,  // [L, n_q]
                               Tensor *tb,          // [B, n_q*hd]
                               int cur_batch_size, int head_dim, int n_q, int kv_mul, int kv_dim,
                               int seq_len, int sliding_window, int pos, long long layer_offset,
                               bool q_to_device, bool k_cache_to_device, bool v_cache_to_device,
                               bool mask_to_device, bool tb_from_device, hipStream_t stream) {
  // GpuTimer timer("single_query_attn_batched", stream);
  if (q_to_device)
    q->to_device(stream);
  if (k_cache_to_device)
    K_cache->to_device(stream);
  if (v_cache_to_device)
    V_cache->to_device(stream);
  if (mask && mask_to_device)
    mask->to_device(stream);

  // const int B = (int)q->shape[0];
  const int n_layers = (int)attn_sinks->shape[0];

  // Point at the current layer for ALL batches; per-batch stride is applied inside the kernel.
  const float *K_cache_ptr = (const float *)K_cache->d_buf + 1ll * layer_offset * seq_len * kv_dim;
  const float *V_cache_ptr = (const float *)V_cache->d_buf + 1ll * layer_offset * seq_len * kv_dim;

  const float *mask_ptr = nullptr;
  if (sliding_window > 0 && ((layer_offset & 1ll) == 0) && mask && mask->d_buf)
    mask_ptr = (const float *)mask->d_buf;

  const float *attn_sinks_ptr = (const float *)attn_sinks->d_buf + 1ll * layer_offset * n_q;

  dim3 grid_dim(n_q, cur_batch_size);
  dim3 block_dim(128);

  // shared memory: q[hd] + scores[attn_len+1] + reduction[blockDim.x doubles]
  const size_t attn_len = (size_t)pos + 1;
  size_t shmem = (size_t)head_dim * sizeof(float) + (attn_len + 1) * sizeof(float) +
                 (size_t)block_dim.x * sizeof(double);

  batched_attention_kernel_opt<<<grid_dim, block_dim, shmem, stream>>>(
    (const float *)q->d_buf, K_cache_ptr, V_cache_ptr, mask_ptr, attn_sinks_ptr, (float *)tb->d_buf,
    head_dim, n_q, kv_mul, kv_dim, cur_batch_size, pos, seq_len, n_layers);

  CHECK_HIP(hipGetLastError());
  if (tb_from_device) {
    tb->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

void attn_out_project_batched(Tensor *tb,         // Shape: [batch_size, n_q*hd]
                              const Tensor *W_o,  // Shape: [hidden_dim, n_q*hd]
                              const Tensor *b_o,  // Shape: [hidden_dim]
                              Tensor *y,          // Shape: [batch_size, hidden_dim]
                              int cur_batch_size, long long layer_offset, bool tb_to_device,
                              bool y_from_device, hipStream_t stream) {
  // GpuTimer timer("attn_out_project", stream);
  if (tb_to_device) {
    tb->to_device(stream);
  }
  // --- MODIFIED: Get dimensions from batched tensor shapes ---
  // Assumes Tensor has a `shape` member, e.g., tb->shape[0]
  // const int batch_size = tb->shape[0];
  const int n_q_hd = tb->shape[1];  // This is the 'in_features'
  const int hidden = y->shape[1];   // This is the 'out_features'

  // Pointer logic for weights and biases remains the same
  const float *w_o_ptr = (float *)W_o->d_buf + 1ll * layer_offset * n_q_hd * hidden;
  const float *b_o_ptr = (float *)b_o->d_buf + 1ll * layer_offset * hidden;

  // Base pointers for the batched input and output
  const float *tb_ptr = (float *)tb->d_buf;
  float *y_ptr = (float *)y->d_buf;

  // Kernel launch constants
  constexpr int WARPS_PER_BLOCK = 16;
  constexpr int TILE = 1024;

  dim3 block_dim(warpSize, WARPS_PER_BLOCK);

  // --- MODIFIED: Use a 2D grid for batching ---
  // grid.x handles the output features (hidden_dim)
  // grid.y handles the batch items
  dim3 grid_dim((hidden + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, cur_batch_size);

  size_t shmem_bytes = TILE * sizeof(float);

  // --- MODIFIED: Call the batched gemm_kernel ---
  gemm_kernel_batched<WARPS_PER_BLOCK, TILE><<<grid_dim, block_dim, shmem_bytes, stream>>>(
    w_o_ptr, tb_ptr, b_o_ptr, y_ptr, hidden, n_q_hd, cur_batch_size);  // Pass batch_size

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

  const float *tb_ptr = (float *)tb->d_buf;
  float *y_ptr = (float *)y->d_buf;

  {
#ifdef RUN_20B
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int TM = 32;
    constexpr int TN = 32;
    constexpr int blockDim = 512;  // = 64 * (BM / TM) * (BN / TN)
#else
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 32;
    constexpr int TN = 32;
    constexpr int blockDim = 512;  // = 64 * (BM / TM) * (BN / TN)
#endif

    dim3 block_size(blockDim);
    dim3 grid_size((out_features + BN - 1) / BN, ((cur_batch_size + BM - 1) / BM));

#ifdef RUN_20B
    gemm_mfma_v2<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
      tb_ptr, w_o_ptr, y_ptr, b_o_ptr, cur_batch_size, out_features, in_features);
#else
    gemm_mfma_v2<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
      tb_ptr, w_o_ptr, y_ptr, b_o_ptr, cur_batch_size, out_features, in_features);
#endif
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

  const float *t_ptr = (float *)t->d_buf;
  const bf16 *w_router_ptr =
    (const bf16 *)w_router->d_buf + 1ll * layer_offset * n_experts * hidden_dim;
  const bf16 *b_router_ptr = (const bf16 *)b_router->d_buf + 1ll * layer_offset * n_experts;
  float *router_score_ptr = (float *)router_scores->d_buf;
  {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int TM = 32;
    constexpr int TN = 16;
    constexpr int blockDim = 512;  // = 64 * (BM / TM) * (BN / TN)

    dim3 block_size(blockDim);
    dim3 grid_size((n_experts + BN - 1) / BN, ((cur_batch_size + BM - 1) / BM));

    gemm_mfma_v2<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
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

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// One block handles one (b,ek) pair; each warp computes one output row.
// We cache the entire input vector x in shared once, then reuse it for all rows.
template <int WARPS_PER_BLOCK>
__global__ void moe_mm_bf16w_xcached(
  const bf16 *__restrict__ W,        // [E, out_features, in_features]
  const float *__restrict__ X,       // if offset_input==0: [B, in_features]
                                     // if offset_input==1: [B,k,in_features]
  const bf16 *__restrict__ Bias,     // [E, out_features] or nullptr
  float *__restrict__ Out,           // [B,k,out_features]
  const int *__restrict__ topk_idx,  // [B,k]
  int B, int k, int out_features, int in_features, int E, bool offset_input) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warpId = threadIdx.x / WARP_SIZE;

  const int pair = blockIdx.y;  // 0..B*k-1
  if (warpId >= WARPS_PER_BLOCK)
    return;
  const int b = pair / k;
  const int ek = pair % k;

  // Resolve expert for this (b,ek)
  const int expert = topk_idx[(size_t)b * k + ek];

  // Base pointers
  const size_t W_base = (size_t)expert * (size_t)out_features * (size_t)in_features;
  const size_t BiasBase = (size_t)expert * (size_t)out_features;
  const float *x_vec =
    offset_input ? (X + (size_t)pair * (size_t)in_features) : (X + (size_t)b * (size_t)in_features);
  float *out_vec = Out + (size_t)pair * (size_t)out_features;

  extern __shared__ float x_sh[];  // size = in_features

  // Cache x once per block
  for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
    x_sh[i] = x_vec[i];
  }
  __syncthreads();

  // Each warp computes one row; grid.x tiles rows by WARPS_PER_BLOCK
  for (int row = blockIdx.x * WARPS_PER_BLOCK + warpId; row < out_features;
       row += gridDim.x * WARPS_PER_BLOCK) {
    const bf16 *__restrict__ W_row = W + W_base + (size_t)row * (size_t)in_features;

    float sum = 0.f;
    // Stride the row across lanes; memory is contiguous in 'i', so this is coalesced
    for (int i = lane; i < in_features; i += WARP_SIZE) {
      sum += (float)W_row[i] * x_sh[i];
    }
    sum = warp_reduce_sum(sum);

    if (lane == 0) {
      float y = sum;
      if (Bias)
        y += (float)Bias[BiasBase + row];
      out_vec[row] = y;
    }
  }
}

// SwiGLU over interleaved [gate, up]; NK=B*k samples
__global__ void swiglu_interleaved_batched_fast(const float *__restrict__ in2I,  // [NK, 2I]
                                                float *__restrict__ outI,        // [NK, I]
                                                int I, int NK, float clamp_limit) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t N = (size_t)NK * I;
  if (idx >= N)
    return;

  int j = idx % I;
  size_t s = idx / I;
  size_t base = s * (size_t)(2 * I);

  float g = in2I[base + 2 * j];
  float u = in2I[base + 2 * j + 1];

  if (clamp_limit > 0.f) {
    g = fminf(fmaxf(g, -clamp_limit), clamp_limit);
    u = fminf(fmaxf(u, -clamp_limit), clamp_limit);
  }
  const float alpha = 1.702f;
  float silu = g * (1.f / (1.f + expf(-alpha * g)));
  outI[idx] = silu * (u + 1.f);
}

// Deterministic, no-atomics weighted sum: out[b,:] += Σ_e w[b,e] * y[b,e,:]
__global__ void weighted_accumulate_noatom_batched(const float *__restrict__ y,  // [B*k, H]
                                                   const float *__restrict__ w,  // [B, k]
                                                   float *__restrict__ out,      // [B, H]
                                                   int B, int k, int H) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;  // 0..B*H-1
  size_t total = (size_t)B * H;
  if (idx >= total)
    return;

  int b = idx / H;
  int h = idx % H;

  float acc = 0.f;
#pragma unroll
  for (int e = 0; e < 4; ++e) {  // k ≤ 4
    if (e < k) {
      float we = w[(size_t)b * k + e];
      float ye = y[((size_t)b * k + e) * H + h];
      acc += we * ye;
    }
  }
  out[(size_t)b * H + h] += acc;
}

void moe_apply_topk_batched(Tensor *t,            // [B,H]
                            const Tensor *W1,     // [L,E,2I,H] (bf16)
                            const Tensor *b1,     // [L,E,2I]   (bf16)
                            const Tensor *W2,     // [L,E,H,I]  (bf16)
                            const Tensor *b2,     // [L,E,H]    (bf16)
                            TensorI32 *topk_idx,  // [B,k]
                            Tensor *topk_vals,    // [B,k]
                            Tensor *mlp1_out,     // [B,k,2I]
                            Tensor *gate_up,      // [B,k,I]
                            Tensor *tb3,          // [B,k,H]
                            Tensor *e_agg,        // [B,H]
                            int cur_batch_size, float clamp_limit, long long layer_offset,
                            bool t_to_device, bool topk_idx_to_device, bool topk_vals_to_device,
                            bool e_agg_from_device, hipStream_t stream) {
  // GpuTimer timer("moe_apply_topk_batched", stream);
  if (t_to_device)
    t->to_device(stream);
  if (topk_idx_to_device)
    topk_idx->to_device(stream);
  if (topk_vals_to_device)
    topk_vals->to_device(stream);

  // const int B = (int)t->shape[0];
  const int H = (int)t->shape[1];
  const int k = (int)topk_idx->shape[1];
  const int I = (int)W2->shape[3];
  const int E = (int)W2->shape[1];

  // layer offsets
  const long long L2IH = (long long)(2 * I) * H;
  const long long LH = (long long)H * I;

  const bf16 *W1p = (const bf16 *)W1->d_buf + (size_t)layer_offset * E * L2IH;
  const bf16 *b1p = (const bf16 *)b1->d_buf + (size_t)layer_offset * E * (2 * I);
  const bf16 *W2p = (const bf16 *)W2->d_buf + (size_t)layer_offset * E * LH;
  const bf16 *b2p = (const bf16 *)b2->d_buf + (size_t)layer_offset * E * H;

  const float *X1 = (const float *)t->d_buf;          // [B,H]
  const float *Wt = (const float *)topk_vals->d_buf;  // [B,k]
  const int *Ti = topk_idx->d_buf;                    // [B,k]

  float *M1 = (float *)mlp1_out->d_buf;  // [B,k,2I]
  float *GU = (float *)gate_up->d_buf;   // [B,k,I]
  float *T3 = (float *)tb3->d_buf;       // [B,k,H]
  float *EA = (float *)e_agg->d_buf;     // [B,H]

  memset_tensor(e_agg, 0, false, true, stream);

  // Tunables
  constexpr int WARPS = 8;  // 8 warps/block → 512 threads
  const dim3 blk(WARP_SIZE * WARPS);
  size_t shmem_x_B = (size_t)H * sizeof(float);  // cache x[b] once
  size_t shmem_x_I = (size_t)I * sizeof(float);  // cache gate[b,ek] once

  // 1) FFN1: [B,k,2I] = W1[ek]*x[b] + b1[ek]
  {
    const int pairs = cur_batch_size * k;
    const dim3 grd((unsigned)((2 * I + WARPS - 1) / WARPS), pairs);
    moe_mm_bf16w_xcached<WARPS><<<grd, blk, shmem_x_B, stream>>>(
      W1p, X1, b1p, M1, Ti, cur_batch_size, k, 2 * I, H, E, /*offset_input=*/false);
    CHECK_HIP(hipGetLastError());
  }

  // 2) SwiGLU
  {
    const int NK = cur_batch_size * k;
    size_t N = (size_t)NK * I;
    dim3 blk2(256), grd2((unsigned)((N + blk2.x - 1) / blk2.x));
    swiglu_interleaved_batched_fast<<<grd2, blk2, 0, stream>>>(M1, GU, I, NK, clamp_limit);
    CHECK_HIP(hipGetLastError());
  }

  // 3) FFN2: [B,k,H] = W2[ek]*gate[b,ek] + b2[ek]
  {
    const int pairs = cur_batch_size * k;
    const dim3 grd((unsigned)((H + WARPS - 1) / WARPS), pairs);
    moe_mm_bf16w_xcached<WARPS><<<grd, blk, shmem_x_I, stream>>>(
      W2p, GU, b2p, T3, Ti, cur_batch_size, k, H, I, E, /*offset_input=*/true);
    CHECK_HIP(hipGetLastError());
  }

  // 4) Weighted sum (deterministic)
  {
    size_t elems = (size_t)cur_batch_size * H;
    dim3 blk3(256), grd3((unsigned)((elems + blk3.x - 1) / blk3.x));
    weighted_accumulate_noatom_batched<<<grd3, blk3, 0, stream>>>(T3, Wt, EA, cur_batch_size, k, H);
    CHECK_HIP(hipGetLastError());
  }

  if (e_agg_from_device) {
    e_agg->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

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

static inline void moe_swiglu_hip(Tensor *mlp1_out,  // [total_pairs, 2*inter_dim]
                                  Tensor *gate_up,   // [total_pairs, inter_dim]
                                  int batch_size, int experts_per_token, int inter_dim,
                                  float clamp_limit, hipStream_t stream) {
  // GpuTimer timer("moe_swiglu", stream);
  moe_swiglu(mlp1_out, gate_up, batch_size, experts_per_token, inter_dim, clamp_limit, stream);
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
  const size_t num_elems = src->num_elem();

  dim3 block_size(256);
  dim3 grid_size((num_elems + block_size.x - 1) / block_size.x);

  quantize_kernel<<<grid_size, block_size, 0, stream>>>((const float *)src->d_buf,
                                                        (bf16 *)dst_quantize->d_buf, num_elems);
}

static inline void tensor_dequantize(Tensor *src, Tensor *dst_quantize, hipStream_t stream) {
  const size_t num_elems = src->num_elem();

  dim3 block_size(256);
  dim3 grid_size((num_elems + block_size.x - 1) / block_size.x);

  dequantize_kernel<<<grid_size, block_size, 0, stream>>>((const bf16 *)dst_quantize->d_buf,
                                                          (float *)src->d_buf, num_elems);
}

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

static inline void moe_mlp1_batched(Tensor *t, Tensor *w_mlp1, Tensor *b_mlp1, TensorI32 *topk_idx,
                                    Tensor *mlp1_out, int cur_batch_size, bool t_to_device,
                                    bool topk_idx_to_device, long long layer_offset,
                                    hipStream_t stream) {
  // GpuTimer timer("moe_mlp1", stream);
  if (t_to_device)
    t->to_device(stream);

  if (topk_idx_to_device)
    topk_idx->to_device(stream);

  // const int batch_size = (int)t->shape[0];
  const int hidden_dim = (int)t->shape[1];
  const int k = (int)topk_idx->shape[1];
  const int inter_size =
    (int)mlp1_out->shape[2];  // (batch_size, experts_per_token, 2 * intermediate_dim)

  const int inter_dim = inter_size / 2;
  const int n_experts = (int)w_mlp1->shape[1];

  const bf16 *w_mlp1_ptr =
    (const bf16 *)w_mlp1->d_buf + (size_t)layer_offset * n_experts * inter_size * hidden_dim;
  const bf16 *b_mlp1_ptr =
    (const bf16 *)b_mlp1->d_buf + (size_t)layer_offset * n_experts * inter_size;

  const float *t_ptr = (const float *)t->d_buf;
  const int *topk_idx_ptr = (const int *)topk_idx->d_buf;
  float *mlp1_out_ptr = (float *)mlp1_out->d_buf;

  constexpr int WARPS = 8;
  const int pairs = cur_batch_size * k;

  const dim3 block_size(WARP_SIZE * WARPS);
  const dim3 grid_size((inter_size + WARPS - 1) / WARPS, pairs);
  const size_t shmem_x_B = (size_t)hidden_dim * sizeof(float);

  moe_mm_bf16w_xcached<WARPS><<<grid_size, block_size, shmem_x_B, stream>>>(
    w_mlp1_ptr, t_ptr, b_mlp1_ptr, mlp1_out_ptr, topk_idx_ptr, cur_batch_size, k, inter_size,
    hidden_dim, n_experts, false);

  CHECK_HIP(hipGetLastError());
}

static inline void moe_swiglu_batched(Tensor *mlp1_out, Tensor *gate_up, int cur_batch_size, int k,
                                      float clamp_limit, hipStream_t stream) {
  // GpuTimer timer("moe_swiglu", stream);
  // const int batch_size = (int)mlp1_out->shape[0];
  const int inter_dim = (int)gate_up->shape[2];
  const size_t total_size = (size_t)cur_batch_size * k * inter_dim;

  const float *mlp1_out_ptr = (const float *)mlp1_out->d_buf;
  float *gate_up_ptr = (float *)gate_up->d_buf;

  dim3 block_size(256);
  dim3 grid_size((total_size + block_size.x - 1) / block_size.x);
  swiglu_interleaved_batched_fast<<<grid_size, block_size, 0, stream>>>(
    mlp1_out_ptr, gate_up_ptr, inter_dim, cur_batch_size * k, clamp_limit);
}

static inline void moe_mlp2_batched(Tensor *gate_up, Tensor *w_mlp2, Tensor *b_mlp2, Tensor *tb3,
                                    TensorI32 *topk_idx, int k, bool has_bias, int cur_batch_size,
                                    long long layer_offset, hipStream_t stream) {
  // GpuTimer timer("moe_mlp2", stream);
  // const int batch_size = (int)gate_up->shape[0];
  const int inter_dim = (int)gate_up->shape[2];
  const int hidden_dim = (int)tb3->shape[2];
  const int n_experts = w_mlp2->shape[1];  // phòng khi layout slice khác

  const bf16 *w_mlp2_ptr =
    (const bf16 *)w_mlp2->d_buf + (size_t)layer_offset * n_experts * hidden_dim * inter_dim;
  const bf16 *b_mlp2_ptr =
    has_bias ? (const bf16 *)b_mlp2->d_buf + (size_t)layer_offset * n_experts * hidden_dim
             : nullptr;
  const float *gate_up_ptr = (const float *)gate_up->d_buf;
  const int *topk_idx_ptr = topk_idx->d_buf;
  float *tb3_ptr = (float *)tb3->d_buf;

  constexpr int WARPS = 8;
  const int pairs = cur_batch_size * k;
  const dim3 block_size(WARP_SIZE * WARPS);
  const dim3 grid_size((unsigned)((hidden_dim + WARPS - 1) / WARPS), pairs);
  const size_t shmem_x_I = (size_t)inter_dim * sizeof(float);

  moe_mm_bf16w_xcached<WARPS><<<grid_size, block_size, shmem_x_I, stream>>>(
    w_mlp2_ptr, gate_up_ptr, b_mlp2_ptr, tb3_ptr, topk_idx_ptr, cur_batch_size, k, hidden_dim,
    inter_dim, n_experts, /*offset_input=*/true);
  CHECK_HIP(hipGetLastError());
}

static inline void moe_agg_batched(Tensor *tb3, Tensor *topk_val, Tensor *e_agg, int k,
                                   int cur_batch_size, bool e_agg_from_device, hipStream_t stream) {
  // GpuTimer timer("moe_agg", stream);
  // const int batch_size = (int)tb3->shape[0];
  const int hidden_dim = (int)tb3->shape[2];

  const float *tb3_ptr = (const float *)tb3->d_buf;
  const float *topk_val_ptr = (const float *)topk_val->d_buf;
  float *e_agg_ptr = (float *)e_agg->d_buf;

  CHECK_HIP(
    hipMemsetAsync(e_agg_ptr, 0, (size_t)cur_batch_size * hidden_dim * sizeof(float), stream));

  size_t elems = (size_t)cur_batch_size * hidden_dim;
  dim3 block_size(256);
  dim3 grid_size((unsigned)((elems + block_size.x - 1) / block_size.x));

  weighted_accumulate_noatom_batched<<<grid_size, block_size, 0, stream>>>(
    tb3_ptr, topk_val_ptr, e_agg_ptr, cur_batch_size, k, hidden_dim);
  CHECK_HIP(hipGetLastError());

  if (e_agg_from_device)
    e_agg->from_device(stream);
}

// ---------- Classifier & Residuals ----------
void classifier_gemm_batched(const Tensor *W_out,  // Shape: [vocab_size, hidden_dim]
                             Tensor *x,            // Shape: [batch_size, hidden_dim]
                             Tensor *logits,       // Shape: [batch_size, vocab_size]
                             int cur_batch_size, bool x_to_device, bool logits_from_device,
                             hipStream_t stream) {
  if (x_to_device) {
    x->to_device(stream);
  }

  // Extract dimensions from tensor shapes
  // const int batch_size = x->shape[0];
  const int hidden_dim = x->shape[1];
  const int vocab_size = W_out->shape[0];

  // Get raw device pointers
  const float *W_out_ptr = (float *)W_out->d_buf;
  const float *x_ptr = (float *)x->d_buf;
  float *logits_ptr = (float *)logits->d_buf;

  // Define kernel launch configuration constants
  constexpr int WARPS_PER_BLOCK = 16;
  constexpr int TILE = 1024;  // Assumed size for shared memory optimization

  // Each block has `warpSize` x `WARPS_PER_BLOCK` threads
  dim3 block_dim(warpSize, WARPS_PER_BLOCK);
  dim3 grid_dim((vocab_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, cur_batch_size);

  size_t shmem_bytes = TILE * sizeof(float);

  gemm_kernel_batched<WARPS_PER_BLOCK, TILE><<<grid_dim, block_dim, shmem_bytes, stream>>>(
    W_out_ptr, x_ptr, nullptr, logits_ptr, vocab_size, hidden_dim, cur_batch_size);

  if (logits_from_device) {
    logits->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
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
  const float *x_ptr = (const float *)x->d_buf;
  float *logits_buf = (float *)logits->d_buf;

  {
#ifdef RUN_20B
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int TM = 64;
    constexpr int TN = 64;
    constexpr int blockDim = 512;  // = 64 * (BM / TM) * (BN / TN)
#else
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int TM = 64;
    constexpr int TN = 64;
    constexpr int blockDim = 512;  // = 64 * (BM / TM) * (BN / TN)
#endif

    dim3 block_size(blockDim);
    dim3 grid_size((vocab_size + BN - 1) / BN, ((cur_batch_size + BM - 1) / BM));

#ifdef RUN_20B
    gemm_mfma_v2<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
      x_ptr, w_out_ptr, logits_buf, nullptr, cur_batch_size, vocab_size, hidden_dim);
#else
    gemm_mfma_v2<BM, BN, BK, TM, TN, blockDim><<<grid_size, block_size, 0, stream>>>(
      x_ptr, w_out_ptr, logits_buf, nullptr, cur_batch_size, vocab_size, hidden_dim);
#endif
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
  // GpuTimer timer("max_logits");
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
  // GpuTimer timer("reduce_logits");

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
