#include "../include/layer_hip_batch.hpp"
#include <cmath>
#include <cfloat>

#define DEFAULT_BLOCK_SIZE 256

// helper funcs:
__device__ __forceinline__ float warp_reduce_sum_batched(float v) {
  // This function is unchanged.
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}
// bf16 helpers
__device__ __forceinline__ float bf16_to_f32(bf16 v) {
    // bf16 -> f32 by left-shift then reinterpret
    uint32_t u = (uint32_t)v << 16;
    return __int_as_float((int)u);
}

__device__ __forceinline__ void bf16x2_to_f32(uint32_t packed, float &f0, float &f1) {
    // packed = [hi:bf16 | lo:bf16]
    uint32_t lo =  packed        & 0xFFFFu;
    uint32_t hi = (packed >> 16) & 0xFFFFu;
    f0 = __int_as_float((int)(lo << 16));
    f1 = __int_as_float((int)(hi << 16));
}


/*
__device__ float block_reduce_sum(float val) {
    // A block can have multiple warps.
    // Each warp reduces its values, then one thread from each warp writes its sum to shared memory.
    // Finally, the first warp reduces the values from shared memory.
    static __shared__ float shared_mem[32]; // Max 32 warps/block (1024/32)
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    val = warp_reduce_sum_batched(val); // Each warp sums its partial results

    if (lane == 0) {
        shared_mem[warp_id] = val; // The first thread of each warp writes to shared memory
    }
    __syncthreads();

    // The first warp is now responsible for summing the results from shared memory
    val = (threadIdx.x < blockDim.x / warpSize) ? shared_mem[lane] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum_batched(val);
    }
    
    return val; // The final sum is in lane 0 of the first warp
}
*/

void embedding_lookup_batched(Tensor *embedding,  // Shape: [vocab_size, hidden_dim]
                              int *tokens,        // Shape: [batch_size]
                              Tensor *x,          // Shape: [batch_size, hidden_dim]
                              bool x_from_device, hipStream_t stream) {
  GpuTimer timer("embedding_lookup");

  const int batch_size = x->shape[0];
  const size_t hidden_dim = x->shape[1];

  for (int i = 0; i < batch_size; i++) {
    if (x->dtype == DType::BF16) {
      bf16 *src = (bf16 *)embedding->d_buf + (size_t)tokens[i] * hidden_dim;
      bf16 *dst = (bf16 *)x->d_buf + 1ll * i * hidden_dim;
      CHECK_HIP(
        hipMemcpyAsync(dst, src, hidden_dim * sizeof(bf16), hipMemcpyDeviceToDevice, stream));
    } else {
      float *src = (float *)embedding->d_buf + (size_t)tokens[i] * hidden_dim;
      float *dst = (float *)x->d_buf + 1ll * i * hidden_dim;
      CHECK_HIP(hipMemcpyAsync(dst, src, hidden_dim * sizeof(float),hipMemcpyDeviceToDevice, stream));
    }
  }

  if (x_from_device) {
    x->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

__global__ void rmsnorm_kernel(const float *x, const float *w, float *out, int hidden_dim,
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
    o_row[j] = w[j] * (final_inv_rms * x_row[j]);
  }
}

void rmsnorm_batched(Tensor *x,    // Shape: [batch_size, hidden_dim]
                     Tensor *w,    // Shape: [hidden_dim]
                     Tensor *out,  // Shape: [batch_size, hidden_dim]
                     long long layer_offset, bool x_to_device, bool out_from_device,
                     float eps, hipStream_t stream) {
  GpuTimer timer("rmsnorm");
  if (x_to_device) {
    x->to_device(stream);
  }

  const int batch_size = x->shape[0];
  const int hidden_dim = x->shape[1];

  const dim3 grid_dim(batch_size);
  const dim3 block_dim(256);
  size_t shared_mem_size = block_dim.x * sizeof(double);

  const float *x_ptr = (float *)x->d_buf;
  const float *w_ptr = (float *)w->d_buf + 1ll * layer_offset * hidden_dim;
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

template<int WARPS_PER_BLOCK, int TILE, int VEC = 4>
__global__ void gemm_kernel_batched(
    const float* __restrict__ W,     // [out_features, in_features]
    const float* __restrict__ x,     // [B, in_features]
    const float* __restrict__ bias,  // [out_features] or nullptr
    float* __restrict__ out,         // [B, out_features]
    int out_features, int in_features, int batch_size)
{
    extern __shared__ float s_x[];   // TILE floats

    const int b = blockIdx.y;
    if (b >= batch_size) return;

    const int lane = threadIdx.x;        // 0..warpSize-1 (MI250 wave = 64)
    const int warp = threadIdx.y;        // 0..WARPS_PER_BLOCK-1
    const int row  = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= out_features) return;

    const float* __restrict__ x_b    = x   + (size_t)b * in_features;
    float*       __restrict__ out_b  = out + (size_t)b * out_features;
    const float* __restrict__ W_row  = W   + (size_t)row * in_features;

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
                const float4 w4 = *reinterpret_cast<const float4*>(W_row + k0 + t);
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
        if (bias) v += bias[row];
        out_b[row] = v;
    }
}

void qkv_gemm_batched(Tensor *x,            // Shape: [batch_size, hidden_dim]
                      const Tensor *W_qkv,  // Shape: [out_features, hidden_dim]
                      const Tensor *b_qkv,  // Shape: [out_features]
                      Tensor *qkv,          // Shape: [batch_size, out_features]
                      long long layer_offset, bool x_to_device, bool qkv_from_device,
                      hipStream_t stream) {
  GpuTimer timer("qkv_gemm");
  if (x_to_device) {
    x->to_device(stream);
  }

  // --- MODIFIED: Get dimensions based on batched shapes ---
  // Assuming Tensor has a shape member or method, e.g., x->shape[0]
  const int batch_size = x->shape[0];
  const int in_features = x->shape[1];   // hidden_dim
  const int out_features = qkv->shape[1];  // 3 * hidden_dim for QKV

  const float *x_ptr = (float *)x->d_buf;
  const float *w_qkv_ptr = (float *)W_qkv->d_buf + 1ll * layer_offset * out_features * in_features;
  const float *b_qkv_ptr = (float *)b_qkv->d_buf + 1ll * layer_offset * out_features;
  float *qkv_ptr = (float *)qkv->d_buf;

  constexpr int WARPS_PER_BLOCK = 16;
  constexpr int TILE = 1024;

  dim3 block_dim(warpSize, WARPS_PER_BLOCK);
  
  // --- MODIFIED: Use a 2D grid to parallelize across both output features and batch size ---
  dim3 grid_dim((out_features + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, batch_size);
  size_t shmem_bytes = TILE * sizeof(float);

  // Launch the modified kernel with the new batch_size parameter
  gemm_kernel_batched<WARPS_PER_BLOCK, TILE><<<grid_dim, block_dim, shmem_bytes, stream>>>(
    w_qkv_ptr, x_ptr, b_qkv_ptr, qkv_ptr, out_features, in_features, batch_size);

  if (qkv_from_device) {
    qkv->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  } 
}

// ---------- Split & RoPE ----------
__global__ void qkv_split_rope_kernel_batched(
    float *qkv_out, float *q_out, float *k_pos, float *v_pos,
    const float *rope_cos_pos, const float *rope_sin_pos,
    int head_dim, int n_q, int n_kv, int batch_size) { // Added batch_size

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
  if (feature_idx < q_dims) { // Processing a Q element
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

  } else if (feature_idx < q_dims + k_dims) { // Processing a K element
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

  } else { // Processing a V element (simple copy)
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
                            int head_dim, int n_q, int n_kv, int pos, bool qkv_out_to_device,
                            bool q_out_from_device, bool k_out_from_device, bool v_out_from_device,
                            hipStream_t stream
) {
    GpuTimer timer("qkv_split_rope");
  if (qkv_out_to_device) {
    qkv_out->to_device(stream);
  }

  // --- MODIFIED: Get batch_size from tensor shape ---
  const int batch_size = qkv_out->shape[0];
  
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
  dim3 grid_dim((total_dims_per_batch + block_dim.x - 1) / block_dim.x, batch_size);
  
  qkv_split_rope_kernel_batched<<<grid_dim, block_dim, 0, stream>>>(
      qkv_out_ptr, q_out_ptr, k_pos_ptr, v_pos_ptr,
      rope_cos_pos_ptr, rope_sin_pos_ptr,
      head_dim, n_q, n_kv, batch_size); // Pass batch_size to kernel

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

__global__ void add_vector_kernel_batched(float *y, const float *b, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    y[i] += b[i];
  }
}

void add_vector_batched(Tensor *y,  // Shape: [batch_size, hidden_dim]
                        Tensor *b,  // Shape: [batch_size, hidden_dim]
                        bool y_to_device, bool b_to_device, bool y_from_device,
                        hipStream_t stream) {
   GpuTimer timer("add_vector");
  if (y_to_device)
    y->to_device(stream);
  if (b_to_device)
    b->to_device(stream);

  float *y_ptr = (float *)y->d_buf;
  const float *b_ptr = (float *)b->d_buf;

  const int len = y->num_elem();

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
    const float * __restrict__ q,            // [B, n_q*hd]
    const float * __restrict__ K_cache,      // [B, L, S, kv_dim]  (host already offset to this layer)
    const float * __restrict__ V_cache,      // [B, L, S, kv_dim]  (host already offset to this layer)
    const float * __restrict__ mask,         // [B, S, S] or nullptr
    const float * __restrict__ attn_sinks,   // [n_q] (for this layer)
    float * __restrict__ tb,                 // [B, n_q*hd]
    int head_dim, int n_q, int kv_mul, int kv_dim,
    int batch_size, int pos, int total_seq_len, int n_layers
)
{
    const int head_idx  = blockIdx.x;           // 0..n_q-1
    const int batch_idx = blockIdx.y;           // 0..B-1
    if (head_idx >= n_q || batch_idx >= batch_size) return;

    const int attn_len = pos + 1;               // tokens to attend (0..pos)
    const int kv_head_idx = head_idx / kv_mul;  // GQA mapping
    const float scale = rsqrtf((float)head_dim);

    // Per-batch base pointers
    const size_t qtb_stride      = (size_t)n_q * head_dim;                 // stride in q/tb
    const size_t kv_batch_stride = (size_t)n_layers * total_seq_len * kv_dim; // stride between batches in KV

    const float* __restrict__ q_head  = q  + (size_t)batch_idx * qtb_stride + (size_t)head_idx * head_dim;
    float*       __restrict__ tb_head = tb + (size_t)batch_idx * qtb_stride + (size_t)head_idx * head_dim;

    const float* __restrict__ K_base  = K_cache + (size_t)batch_idx * kv_batch_stride;
    const float* __restrict__ V_base  = V_cache + (size_t)batch_idx * kv_batch_stride;

    // Mask row pointer if provided
    const float* __restrict__ mask_row = nullptr;
    if (mask) {
        const size_t mask_batch_stride = (size_t)total_seq_len * total_seq_len;
        mask_row = mask + (size_t)batch_idx * mask_batch_stride + (size_t)pos * total_seq_len;
    }

    // ---- Shared memory layout ----
    extern __shared__ unsigned char sdata[];
    float*  s_q      = (float*)sdata;                                   // [hd]
    float*  s_scores = (float*)(s_q + head_dim);                         // [attn_len + 1] (last slot = sink)
    double* s_red    = (double*)(s_scores + (size_t)attn_len + 1);       // [blockDim.x] scratch for reductions

    // 1) Stage q into shared once (all threads reuse)
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        s_q[i] = q_head[i];
    __syncthreads();

    // 2) Compute raw scores in parallel
    double local_max = -DBL_MAX;
    for (int t = threadIdx.x; t < attn_len; t += blockDim.x) {
        const float* __restrict__ k_vec = K_base + (size_t)t * kv_dim + (size_t)kv_head_idx * head_dim;

        double score = 0.0;
        int i = 0;
        for (; i + 3 < head_dim; i += 4) {
            float q0 = s_q[i+0], q1 = s_q[i+1], q2 = s_q[i+2], q3 = s_q[i+3];
            float k0 = k_vec[i+0], k1 = k_vec[i+1], k2 = k_vec[i+2], k3 = k_vec[i+3];
            score += (double)q0 * (double)k0
                   + (double)q1 * (double)k1
                   + (double)q2 * (double)k2
                   + (double)q3 * (double)k3;
        }
        for (; i < head_dim; ++i) score += (double)s_q[i] * (double)k_vec[i];

        score *= (double)scale;
        if (mask_row) score += (double)mask_row[t];

        s_scores[t] = (float)score;
        if (score > local_max) local_max = score;
    }

    // add sink into max via thread 0; reduce max across block
    s_red[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_red[threadIdx.x] = fmax(s_red[threadIdx.x], s_red[threadIdx.x + s]);
        __syncthreads();
    }
    double max_score = s_red[0];
    if (threadIdx.x == 0) {
        const double sink = (double)attn_sinks[head_idx];
        s_scores[attn_len] = (float)sink;                  // store sink at tail
        if (sink > max_score) max_score = sink;
        s_red[0] = max_score;                              // broadcast via shared
    }
    __syncthreads();
    max_score = s_red[0];

    // 3) Exponentiate and reduce denominator in parallel
    double local_sum = 0.0;
    for (int t = threadIdx.x; t < attn_len; t += blockDim.x) {
        float e = expf(s_scores[t] - (float)max_score);
        s_scores[t] = e;               // keep e_t; we will normalize later
        local_sum += (double)e;
    }
    s_red[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_red[threadIdx.x] += s_red[threadIdx.x + s];
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
            const float p = s_scores[t]; // normalized prob
            const float* __restrict__ v_vec = V_base + (size_t)t * kv_dim + kv_head_off;
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
                               int head_dim, int n_q, int kv_mul, int kv_dim, int seq_len,
                               int sliding_window, int pos, long long layer_offset,
                               bool q_to_device, bool k_cache_to_device, bool v_cache_to_device,
                               bool mask_to_device, bool tb_from_device, hipStream_t stream)
{
    GpuTimer timer("single_query_attn_batched");
    if (q_to_device)            q->to_device(stream);
    if (k_cache_to_device)      K_cache->to_device(stream);
    if (v_cache_to_device)      V_cache->to_device(stream);
    if (mask && mask_to_device) mask->to_device(stream);

    const int B         = (int)q->shape[0];
    const int n_layers  = (int)attn_sinks->shape[0];

    // Point at the current layer for ALL batches; per-batch stride is applied inside the kernel.
    const float *K_cache_ptr = (const float*)K_cache->d_buf + 1ll * layer_offset * seq_len * kv_dim;
    const float *V_cache_ptr = (const float*)V_cache->d_buf + 1ll * layer_offset * seq_len * kv_dim;

    const float *mask_ptr = nullptr;
    if (sliding_window > 0 && ((layer_offset & 1ll) == 0) && mask && mask->d_buf)
        mask_ptr = (const float*)mask->d_buf;

    const float *attn_sinks_ptr = (const float*)attn_sinks->d_buf + 1ll * layer_offset * n_q;

    dim3 grid_dim(n_q, B);
    dim3 block_dim(128);

    // shared memory: q[hd] + scores[attn_len+1] + reduction[blockDim.x doubles]
    const size_t attn_len = (size_t)pos + 1;
    size_t shmem = (size_t)head_dim * sizeof(float)
                 + (attn_len + 1) * sizeof(float)
                 + (size_t)block_dim.x * sizeof(double);

    batched_attention_kernel_opt<<<grid_dim, block_dim, shmem, stream>>>(
        (const float*)q->d_buf,
        K_cache_ptr, V_cache_ptr,
        mask_ptr,
        attn_sinks_ptr,
        (float*)tb->d_buf,
        head_dim, n_q, kv_mul, kv_dim,
        B, pos, seq_len, n_layers);

    CHECK_HIP(hipGetLastError());
    if (tb_from_device) { tb->from_device(stream); CHECK_HIP(hipStreamSynchronize(stream)); }
}

void attn_out_project_batched(Tensor *tb,         // Shape: [batch_size, n_q*hd]
                              const Tensor *W_o,  // Shape: [hidden_dim, n_q*hd]
                              const Tensor *b_o,  // Shape: [hidden_dim]
                              Tensor *y,          // Shape: [batch_size, hidden_dim]
                              long long layer_offset, bool tb_to_device, bool y_from_device,
                              hipStream_t stream) {
  if (tb_to_device) {
    tb->to_device(stream);
  }
  GpuTimer timer("attn_out_project");
  // --- MODIFIED: Get dimensions from batched tensor shapes ---
  // Assumes Tensor has a `shape` member, e.g., tb->shape[0]
  const int batch_size = tb->shape[0];
  const int n_q_hd = tb->shape[1];  // This is the 'in_features'
  const int hidden = y->shape[1];    // This is the 'out_features'

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
  dim3 grid_dim((hidden + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, batch_size);
  
  size_t shmem_bytes = TILE * sizeof(float);
  
  // --- MODIFIED: Call the batched gemm_kernel ---
  gemm_kernel_batched<WARPS_PER_BLOCK, TILE><<<grid_dim, block_dim, shmem_bytes, stream>>>(
      w_o_ptr, tb_ptr, b_o_ptr, y_ptr, 
      hidden, n_q_hd, batch_size); // Pass batch_size

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
                         long long layer_offset, bool t_to_device, bool r_from_device,
                         hipStream_t stream) {
  GpuTimer timer("router_gemm_batched");
  if (t_to_device) t->to_device(stream);

  const int B = (int)t->shape[0];
  const int H = (int)t->shape[1];
  const int E = (int)router_scores->shape[1];

  const float *W = (const float*)w_router->d_buf + 1ll * layer_offset * E * H;
  const float *b = (const float*)b_router->d_buf + 1ll * layer_offset * E;
  const float *X = (const float*)t->d_buf;
  float *R       = (float*)router_scores->d_buf;

  // tuned like qkv/classifier
  constexpr int WARPS_PER_BLOCK = 16;
  constexpr int TILE            = 1024;

  dim3 block_dim(warpSize, WARPS_PER_BLOCK);
  dim3 grid_dim((E + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, B);
  size_t shmem = TILE * sizeof(float);

  gemm_kernel_batched<WARPS_PER_BLOCK, TILE>
      <<<grid_dim, block_dim, shmem, stream>>>(W, X, b, R, E, H, B);

  CHECK_HIP(hipGetLastError());
  if (r_from_device) { router_scores->from_device(stream); CHECK_HIP(hipStreamSynchronize(stream)); }
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
  const float* r_batch = r + (size_t)batch_idx * n_experts;
  float* topk_vals_batch = topk_vals + (size_t)batch_idx * k;
  int* topk_idx_batch = topk_idx + (size_t)batch_idx * k;

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
                          bool r_to_device, bool topk_vals_from_device, bool topk_idx_from_device,
                          hipStream_t stream) 
{
    GpuTimer timer("topk_softmax_batched");
  if (r_to_device) {
      r->to_device(stream);
  }

  // Extract dimensions from tensor shapes
  // r shape: [batch_size, n_experts]
  // topk_idx shape: [batch_size, k]
  const int batch_size = r->shape[0];
  const int num_experts = r->shape[1];
  const int experts_per_token = topk_idx->shape[1]; // This is 'k'

  // Get raw device pointers
  const float *r_ptr = (float *)r->d_buf;
  float *topk_vals_ptr = (float *)topk_vals->d_buf;
  int *topk_idx_ptr = topk_idx->d_buf;

  // Shared memory size is calculated per block, for one batch item's data
  size_t shared_mem_size = (size_t)num_experts * (sizeof(float) + sizeof(int));
  
  // Configure kernel launch:
  // Grid dimension is the batch_size to process all items in parallel.
  // Block dimension is 1 since one thread does the work within the block.
  dim3 grid_dim(batch_size, 1, 1);
  dim3 block_dim(1, 1, 1);

  batched_topk_softmax_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
      r_ptr, batch_size, num_experts, experts_per_token, topk_vals_ptr, topk_idx_ptr);
  
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
template<int WARPS_PER_BLOCK>
__global__ void moe_mm_bf16w_xcached(
    const bf16* __restrict__ W,       // [E, out_features, in_features]
    const float* __restrict__ X,      // if offset_input==0: [B, in_features]
                                      // if offset_input==1: [B,k,in_features]
    const bf16* __restrict__ Bias,    // [E, out_features] or nullptr
    float* __restrict__ Out,          // [B,k,out_features]
    const int* __restrict__ topk_idx, // [B,k]
    int B, int k, int out_features, int in_features,
    int E, bool offset_input)
{
    const int lane   = threadIdx.x & (WARP_SIZE - 1);
    const int warpId = threadIdx.x / WARP_SIZE;

    const int pair = blockIdx.y;       // 0..B*k-1
    if (warpId >= WARPS_PER_BLOCK) return;
    const int b  = pair / k;
    const int ek = pair % k;

    // Resolve expert for this (b,ek)
    const int expert = topk_idx[(size_t)b * k + ek];

    // Base pointers
    const size_t W_base   = (size_t)expert * (size_t)out_features * (size_t)in_features;
    const size_t BiasBase = (size_t)expert * (size_t)out_features;
    const float* x_vec    = offset_input
                            ? (X + (size_t)pair * (size_t)in_features)
                            : (X + (size_t)b * (size_t)in_features);
    float* out_vec        = Out + (size_t)pair * (size_t)out_features;

    extern __shared__ float x_sh[]; // size = in_features

    // Cache x once per block
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
        x_sh[i] = x_vec[i];
    }
    __syncthreads();

    // Each warp computes one row; grid.x tiles rows by WARPS_PER_BLOCK
    for (int row = blockIdx.x * WARPS_PER_BLOCK + warpId;
         row < out_features;
         row += gridDim.x * WARPS_PER_BLOCK)
    {
        const bf16* __restrict__ W_row = W + W_base + (size_t)row * (size_t)in_features;

        float sum = 0.f;
        // Stride the row across lanes; memory is contiguous in 'i', so this is coalesced
        for (int i = lane; i < in_features; i += WARP_SIZE) {
            sum += (float)W_row[i] * x_sh[i];
        }
        sum = warp_reduce_sum_batched(sum);

        if (lane == 0) {
            float y = sum;
            if (Bias) y += (float)Bias[BiasBase + row];
            out_vec[row] = y;
        }
    }
}

// SwiGLU over interleaved [gate, up]; NK=B*k samples
__global__ void swiglu_interleaved_batched_fast(
    const float* __restrict__ in2I, // [NK, 2I]
    float* __restrict__ outI,       // [NK, I]
    int I, int NK, float clamp_limit)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t N   = (size_t)NK * I;
    if (idx >= N) return;

    int j   = idx % I;
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
__global__ void weighted_accumulate_noatom_batched(
    const float* __restrict__ y,   // [B*k, H]
    const float* __restrict__ w,   // [B, k]
    float* __restrict__ out,       // [B, H]
    int B, int k, int H)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; // 0..B*H-1
    size_t total = (size_t)B * H;
    if (idx >= total) return;

    int b = idx / H;
    int h = idx % H;

    float acc = 0.f;
    #pragma unroll
    for (int e = 0; e < 4; ++e) { // k ≤ 4
        if (e < k) {
            float we = w[(size_t)b * k + e];
            float ye = y[((size_t)b * k + e) * H + h];
            acc += we * ye;
        }
    }
    out[(size_t)b * H + h] += acc;
}

void moe_apply_topk_batched(
  Tensor *t,            // [B,H]
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
  float clamp_limit, long long layer_offset,
  bool t_to_device, bool topk_idx_to_device, bool topk_vals_to_device,
  bool e_agg_from_device, hipStream_t stream)
{
    GpuTimer timer("moe_apply_topk_batched");
    if (t_to_device)           t->to_device(stream);
    if (topk_idx_to_device)    topk_idx->to_device(stream);
    if (topk_vals_to_device)   topk_vals->to_device(stream);

    const int B = (int)t->shape[0];
    const int H = (int)t->shape[1];
    const int k = (int)topk_idx->shape[1];
    const int I = (int)W2->shape[3];
    const int E = (int)W2->shape[1];

    // layer offsets
    const long long L2IH = (long long)(2 * I) * H;
    const long long LH   = (long long)H * I;

    const bf16* W1p = (const bf16*)W1->d_buf + (size_t)layer_offset * E * L2IH;
    const bf16* b1p = (const bf16*)b1->d_buf + (size_t)layer_offset * E * (2 * I);
    const bf16* W2p = (const bf16*)W2->d_buf + (size_t)layer_offset * E * LH;
    const bf16* b2p = (const bf16*)b2->d_buf + (size_t)layer_offset * E * H;

    const float* X1 = (const float*)t->d_buf;             // [B,H]
    const float* Wt = (const float*)topk_vals->d_buf;     // [B,k]
    const int*   Ti = topk_idx->d_buf;                    // [B,k]

    float* M1 = (float*)mlp1_out->d_buf;                  // [B,k,2I]
    float* GU = (float*)gate_up->d_buf;                   // [B,k,I]
    float* T3 = (float*)tb3->d_buf;                       // [B,k,H]
    float* EA = (float*)e_agg->d_buf;                     // [B,H]

    memset_tensor(e_agg, 0, false, true, stream);

    // Tunables
    constexpr int WARPS = 8;                         // 8 warps/block → 512 threads
    const dim3 blk(WARP_SIZE * WARPS);
    size_t shmem_x_B = (size_t)H * sizeof(float);    // cache x[b] once
    size_t shmem_x_I = (size_t)I * sizeof(float);    // cache gate[b,ek] once

    // 1) FFN1: [B,k,2I] = W1[ek]*x[b] + b1[ek]
    {
        const int pairs = B * k;
        const dim3 grd((unsigned)((2 * I + WARPS - 1) / WARPS), pairs);
        moe_mm_bf16w_xcached<WARPS><<<grd, blk, shmem_x_B, stream>>>(
            W1p, X1, b1p, M1, Ti, B, k, 2 * I, H, E, /*offset_input=*/false);
        CHECK_HIP(hipGetLastError());
    }

    // 2) SwiGLU
    {
        const int NK = B * k;
        size_t N = (size_t)NK * I;
        dim3 blk2(256), grd2((unsigned)((N + blk2.x - 1) / blk2.x));
        swiglu_interleaved_batched_fast<<<grd2, blk2, 0, stream>>>(M1, GU, I, NK, clamp_limit);
        CHECK_HIP(hipGetLastError());
    }

    // 3) FFN2: [B,k,H] = W2[ek]*gate[b,ek] + b2[ek]
    {
        const int pairs = B * k;
        const dim3 grd((unsigned)((H + WARPS - 1) / WARPS), pairs);
        moe_mm_bf16w_xcached<WARPS><<<grd, blk, shmem_x_I, stream>>>(
            W2p, GU, b2p, T3, Ti, B, k, H, I, E, /*offset_input=*/true);
        CHECK_HIP(hipGetLastError());
    }

    // 4) Weighted sum (deterministic)
    {
        size_t elems = (size_t)B * H;
        dim3 blk3(256), grd3((unsigned)((elems + blk3.x - 1) / blk3.x));
        weighted_accumulate_noatom_batched<<<grd3, blk3, 0, stream>>>(T3, Wt, EA, B, k, H);
        CHECK_HIP(hipGetLastError());
    }

    if (e_agg_from_device) { e_agg->from_device(stream); CHECK_HIP(hipStreamSynchronize(stream)); }
}

// ---------- Classifier & Residuals ----------
void classifier_gemm_batched(const Tensor *W_out,  // Shape: [vocab_size, hidden_dim]
                             Tensor *x,            // Shape: [batch_size, hidden_dim]
                             Tensor *logits,       // Shape: [batch_size, vocab_size]
                             bool x_to_device, bool logits_from_device, hipStream_t stream) {
   GpuTimer timer("classifier_batched");
    if (x_to_device) {
        x->to_device(stream);
    }

    // Extract dimensions from tensor shapes
    const int batch_size = x->shape[0];
    const int hidden_dim = x->shape[1];
    const int vocab_size = W_out->shape[0];

    // Get raw device pointers
    const float *W_out_ptr = (float *)W_out->d_buf;
    const float *x_ptr = (float *)x->d_buf;
    float *logits_ptr = (float *)logits->d_buf;

    // Define kernel launch configuration constants
    constexpr int WARPS_PER_BLOCK = 16;
    constexpr int TILE = 1024; // Assumed size for shared memory optimization

    // Each block has `warpSize` x `WARPS_PER_BLOCK` threads
    dim3 block_dim(warpSize, WARPS_PER_BLOCK);
    
    // The grid is 2D:
    // - The x-dimension covers the vocabulary size.
    // - The y-dimension covers the batch size.
    // This maps each matrix-vector multiplication in the batch to a row of blocks.
    dim3 grid_dim((vocab_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, batch_size);
    
    // Shared memory is likely used by the kernel to cache the input vector `x` for faster access.
    size_t shmem_bytes = TILE * sizeof(float);

    // Launch the batched GEMM kernel
    gemm_kernel_batched<WARPS_PER_BLOCK, TILE><<<grid_dim, block_dim, shmem_bytes, stream>>>(
        W_out_ptr, 
        x_ptr, 
        nullptr,       // No bias is used in this operation
        logits_ptr, 
        vocab_size, 
        hidden_dim,
        batch_size
    );

    if (logits_from_device) {
        logits->from_device(stream);
        // Block until the kernel and data transfer are complete
        CHECK_HIP(hipStreamSynchronize(stream));
    }
}
