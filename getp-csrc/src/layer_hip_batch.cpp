#include "../include/layer_hip_batch.hpp"
#include <cmath>
#include <cfloat>

#define DEFAULT_BLOCK_SIZE 256

void embedding_lookup_batched(
  Tensor *embedding,  // Shape: [vocab_size, hidden_dim]
  int *tokens,        // Shape: [batch_size]
  Tensor *x,          // Shape: [batch_size, hidden_dim]
  bool x_from_device, int cur_batch_size, hipStream_t stream
) {
  GpuTimer("embedding_lookup");

  const size_t hidden_dim = x->shape[1];

  for (int i = 0; i < cur_batch_size; i++) {
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

__global__ void rmsnorm_kernel(
  const float *x, const float *w, float *out, int hidden_dim,
  float eps
) {
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

void rmsnorm_batched(
  Tensor *x,    // Shape: [batch_size, hidden_dim]
  Tensor *w,    // Shape: [hidden_dim]
  Tensor *out,  // Shape: [batch_size, hidden_dim]
  long long layer_offset, bool x_to_device, bool out_from_device,
  int cur_batch_size, float eps, hipStream_t stream
) {
  // GpuTimer timer("rmsnorm");
  if (x_to_device) {
    x->to_device(stream);
  }

  const int hidden_dim = x->shape[1];

  const dim3 grid_dim(cur_batch_size);
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
__device__ __forceinline__ float warp_reduce_sum_batched(float v) {
  // This function is unchanged.
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}

// Renamed and modified kernel to handle batches (GEMM)
template <int WARPS_PER_BLOCK, int TILE>
__global__ void gemm_kernel_batched(
  const float *W, const float *x, const float *bias, float *out,
  int out_features, int in_features, int batch_size
) {
  extern __shared__ float local_x[];

  // --- MODIFIED: Identify batch index from the grid's y-dimension ---
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) {
    return;
  }

  const int lane = threadIdx.x;
  const int w = threadIdx.y;
  const int row = blockIdx.x * WARPS_PER_BLOCK + w;

  if (row >= out_features) {
    return;
  }

  // --- MODIFIED: Offset input and output pointers by batch index ---
  const float *x_batch = x + 1ll * batch_idx * in_features;
  float *out_batch = out + 1ll * batch_idx * out_features;

  float acc = 0.0f;
  const int row_base = row * in_features;

  for (int k0 = 0; k0 < in_features; k0 += TILE) {
    const int tile_len = min(TILE, in_features - k0);

    // Load a tile of the current input vector into shared memory
    for (int t = w * warpSize + lane; t < tile_len; t += WARPS_PER_BLOCK * warpSize) {
      local_x[t] = x_batch[k0 + t]; // Read from batched input
    }
    __syncthreads();

    // Compute dot product for the tile
    for (int t = lane; t < tile_len; t += warpSize) {
      float wv = W[row_base + k0 + t];
      acc += wv * local_x[t];
    }
    __syncthreads();
  }

  acc = warp_reduce_sum_batched(acc);

  if (lane == 0) {
    if (bias) {
      acc += bias[row];
    }
    out_batch[row] = acc; // Write to batched output
  }
}

__global__ void batched_matmul_kernel(
  const float *W, const float *x, const float *bias, float *out,
  int out_features, int in_features, int batch_size
) {
  // --- MODIFIED: Use a 2D grid for batching ---
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y;

  // Bounds check for both output features and batch size
  if (row >= out_features || batch_idx >= batch_size) {
    return;
  }

  // --- MODIFIED: Offset input and output pointers by batch index ---
  const float *x_batch = x + (size_t)batch_idx * in_features;
  float *out_batch = out + (size_t)batch_idx * out_features;
  
  // --- Core logic remains the same, but uses the new batched pointers ---
  float sum = 0.0f;
  const float *W_row = W + (size_t)row * in_features; // W is not batched
  for (int i = 0; i < in_features; i++) {
    sum += W_row[i] * x_batch[i]; // Read from the correct batch in 'x'
  }
  if (bias != nullptr) {
    sum += bias[row]; // Bias is not batched
  }
  out_batch[row] = sum; // Write to the correct batch in 'out'
}

void qkv_gemm_batched(
  Tensor *x,            // Shape: [batch_size, hidden_dim]
  const Tensor *W_qkv,  // Shape: [out_features, hidden_dim]
  const Tensor *b_qkv,  // Shape: [out_features]
  Tensor *qkv,          // Shape: [batch_size, out_features]
  long long layer_offset, bool x_to_device, bool qkv_from_device,
  int cur_batch_size, hipStream_t stream
) {
  // GpuTimer timer("qkv_gemm");
  if (x_to_device) {
    x->to_device(stream);
  }

  // --- MODIFIED: Get dimensions based on batched shapes ---
  // Assuming Tensor has a shape member or method, e.g., x->shape[0]
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

// ---------- Split & RoPE ----------
__global__ void qkv_split_rope_kernel_batched(
  float *qkv_out, float *q_out, float *k_pos, float *v_pos,
  const float *rope_cos_pos, const float *rope_sin_pos,
  int head_dim, int n_q, int n_kv, int batch_size
) { // Added batch_size

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

void qkv_split_rope_batched(
  Tensor *qkv_out,             // Shape: [batch_size, (n_q + 2*n_kv)*hd]
  Tensor *q_out,               // Shape: [batch_size, n_q*hd]
  Tensor *k_pos,               // Shape: [batch_size, n_kv*hd]
  Tensor *v_pos,               // Shape: [batch_size, n_kv*hd]
  const Tensor *rope_cos_pos,  // Shape: [seq_len, hd/2]
  const Tensor *rope_sin_pos,  // Shape: [seq_len, hd/2]
  int head_dim, int n_q, int n_kv, int pos, bool qkv_out_to_device,
  bool q_out_from_device, bool k_out_from_device, bool v_out_from_device,
  int cur_batch_size, hipStream_t stream
) {
  if (qkv_out_to_device) {
    qkv_out->to_device(stream);
  }

  // --- MODIFIED: Get batch_size from tensor shape ---
  
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
      qkv_out_ptr, q_out_ptr, k_pos_ptr, v_pos_ptr,
      rope_cos_pos_ptr, rope_sin_pos_ptr,
      head_dim, n_q, n_kv, cur_batch_size); // Pass batch_size to kernel

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

void add_vector_batched(
  Tensor *y,  // Shape: [batch_size, hidden_dim]
  Tensor *b,  // Shape: [batch_size, hidden_dim]
  bool y_to_device, bool b_to_device, bool y_from_device, hipStream_t stream
) {
  // GpuTimer timer("add_vector");
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
__global__ void batched_attention_kernel(
  const float *q, const float *K_cache, const float *V_cache,
  const float *mask, const float *attn_sinks, float *tb,
  int head_dim, int n_q, int kv_mul, int kv_dim, int batch_size, int pos,
  int total_seq_len, int n_layers /* Full dimension of K/V cache and mask */
) {
    int head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;

    if (head_idx >= n_q || batch_idx >= batch_size) {
        return;
    }
    
    // The number of tokens to attend to is pos + 1
    int attn_len = pos + 1;

    // 1. Calculate batched pointers
    // Stride for one full batch item in q and tb
    size_t batch_q_stride = (size_t)n_q * head_dim;
    const float *q_head = q + (size_t)batch_idx * batch_q_stride + (size_t)head_idx * head_dim;
    float *tb_head = tb + (size_t)batch_idx * batch_q_stride + (size_t)head_idx * head_dim;

    // Stride for one full batch item in K/V cache
    size_t batch_kv_stride = 1ll * n_layers * total_seq_len * kv_dim;
    const float *K_cache_batch = K_cache + (size_t)batch_idx * batch_kv_stride;
    const float *V_cache_batch = V_cache + (size_t)batch_idx * batch_kv_stride;
    
    // attn_sinks are shared across the batch, indexed by head
    const float attn_sink = attn_sinks[head_idx];

    extern __shared__ char s_data[];
    // Shared memory for scores: one per token in sequence + one for the sink
    float *s_scores = (float *)s_data;
    // Shared memory for the output vector of this head
    double *s_output = (double *)(s_data + ((size_t)attn_len + 1) * sizeof(float));

    int kv_head_idx = head_idx / kv_mul;
    float scale = rsqrtf((float)head_dim);
    
    // 2. Calculate Attention Scores (Parallelized over threads)
    for (int t = threadIdx.x; t < attn_len; t += blockDim.x) {
        const float *k_vec = K_cache_batch + (size_t)t * kv_dim + (size_t)kv_head_idx * head_dim;
        
        double score = 0.0;
        for (int i = 0; i < head_dim; i++) {
            score += (double)q_head[i] * k_vec[i];
        }
        score *= scale;

        if (mask != nullptr) {
            // Stride for one full batch item in the mask tensor
            size_t batch_mask_stride = (size_t)total_seq_len * total_seq_len;
            const float* mask_row = mask + (size_t)batch_idx * batch_mask_stride + (size_t)pos * total_seq_len;
            score += mask_row[t];
        }
        s_scores[t] = (float)score;
    }
    __syncthreads();

    // 3. Softmax and weighted sum of V (Done by a single thread)
    if (threadIdx.x == 0) {
        s_scores[attn_len] = attn_sink;
        int softmax_len = attn_len + 1;

        double max_score = -DBL_MAX;
        for (int t = 0; t < softmax_len; t++) {
            if (s_scores[t] > max_score) {
                max_score = s_scores[t];
            }
        }

        double denom = 0.0;
        for (int t = 0; t < softmax_len; t++) {
            s_scores[t] = expf(s_scores[t] - (float)max_score);
            denom += s_scores[t];
        }

        for (int i = 0; i < head_dim; i++) {
            s_output[i] = 0.0;
        }

        for (int t = 0; t < attn_len; t++) {
            double prob = (double)s_scores[t] / denom;
            const float *v_vec = V_cache_batch + (size_t)t * kv_dim + (size_t)kv_head_idx * head_dim;
            for (int i = 0; i < head_dim; i++) {
                s_output[i] += prob * v_vec[i];
            }
        }
    }
    __syncthreads();

    // 4. Write result from shared to global memory (Parallelized over threads)
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        tb_head[i] = (float)s_output[i];
    }
}

void single_query_attn_batched(
  Tensor *q,           // Shape: [batch_size, n_q*hd]
  Tensor *K_cache,     // Shape: [batch_size, layer, seq_len, kv_dim]
  Tensor *V_cache,     // Shape: [batch_size, layer, seq_len, kv_dim]
  Tensor *mask,        // Shape: [batch_size, seq_len, seq_len]
  Tensor *attn_sinks,  // Shape: [n_layers, n_attn_heads]
  Tensor *tb,          // Shape: [batch_size, n_q*hd]
  int head_dim, int n_q, int kv_mul, int kv_dim, int seq_len,
  int sliding_window, int pos, long long layer_offset, bool q_to_device,
  bool k_cache_to_device, bool v_cache_to_device, bool mask_to_device,
  bool tb_from_device, int cur_batch_size, hipStream_t stream
) {
  if (q_to_device) q->to_device(stream);
  if (k_cache_to_device) K_cache->to_device(stream);
  if (v_cache_to_device) V_cache->to_device(stream);
  if (mask_to_device && mask != nullptr) mask->to_device(stream);

  const int n_layers = attn_sinks->shape[0];

  // 1. Get raw device pointers from Tensor objects
  const float *q_ptr = (float *)q->d_buf;
  float *tb_ptr = (float *)tb->d_buf;

  // 2. Calculate the offset to the current layer's KV cache for all batches.
  // The shape is [batch_size, n_layers, seq_len, kv_dim]. We want to get the pointer
  // to the start of the data for `layer_offset`.
  // The size of data for a single layer across all batches is not contiguous.
  // Assuming layout [batch, layer, seq, dim], the stride between layers is (seq_len * kv_dim).
  long long layer_stride = 1ll * n_layers * seq_len * kv_dim;
  const float *K_cache_ptr = (const float *)K_cache->d_buf + 1ll * layer_offset * seq_len * kv_dim;
  const float *V_cache_ptr = (const float *)V_cache->d_buf + 1ll * layer_offset * seq_len * kv_dim;

  // 3. Get pointer to the mask tensor if applicable
  const float *mask_ptr = nullptr;
  if (sliding_window > 0 && (layer_offset % 2 == 0)) {
    if (mask != nullptr && mask->d_buf != nullptr) {
      mask_ptr = (const float *)mask->d_buf;
    }
  }

  // 4. Calculate the pointer to the current layer's attention sinks (not batched)
  const float *attn_sinks_ptr = (const float *)attn_sinks->d_buf + layer_offset * n_q;

  // 5. Launch batched kernel
  // Grid dimensions are (number of heads, batch size)
  dim3 grid_dim(n_q, cur_batch_size);
  dim3 block_dim(256); // A common, reasonable block size

  // Shared memory size depends on the current sequence length (pos + 1)
  size_t shared_mem_size = ((size_t)pos + 2) * sizeof(float) + (size_t)head_dim * sizeof(double);

  batched_attention_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
    q_ptr, K_cache_ptr, V_cache_ptr, mask_ptr, attn_sinks_ptr, tb_ptr,
    head_dim, n_q, kv_mul, kv_dim, cur_batch_size, pos, seq_len, n_layers
  );

  if (tb_from_device) {
    tb->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

void attn_out_project_batched(
  Tensor *tb,         // Shape: [batch_size, n_q*hd]
  const Tensor *W_o,  // Shape: [hidden_dim, n_q*hd]
  const Tensor *b_o,  // Shape: [hidden_dim]
  Tensor *y,          // Shape: [batch_size, hidden_dim]
  long long layer_offset, bool tb_to_device, bool y_from_device,
  int cur_batch_size, hipStream_t stream
) {
  if (tb_to_device) {
    tb->to_device(stream);
  }

  // --- MODIFIED: Get dimensions from batched tensor shapes ---
  // Assumes Tensor has a `shape` member, e.g., tb->shape[0]
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
  dim3 grid_dim((hidden + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 
                  cur_batch_size);
  
  size_t shmem_bytes = TILE * sizeof(float);
  
  // --- MODIFIED: Call the batched gemm_kernel ---
  gemm_kernel_batched<WARPS_PER_BLOCK, TILE><<<grid_dim, block_dim, shmem_bytes, stream>>>(
    w_o_ptr, tb_ptr, b_o_ptr, y_ptr, hidden, n_q_hd, cur_batch_size
  ); // Pass batch_size

  if (y_from_device) {
    y->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

// ---------- Router GEMM ----------
void router_gemm_batched(
  const Tensor *w_router,  // Shape: [n_experts, hidden_dim]
  Tensor *t,               // Shape: [batch_size, hidden_dim]
  const Tensor *b_router,  // Shape: [n_experts]
  Tensor *router_scores,   // Shape: [batch_size, n_experts]
  long long layer_offset, bool t_to_device, bool r_from_device,
  int cur_batch_size, hipStream_t stream
) {
  // Transfer input tensor to the GPU if required
  if (t_to_device) {
    t->to_device(stream);
  }

  // Extract dimensions from the input tensors
  // t shape: [batch_size, hidden_dim]
  // w_router shape: [n_experts, hidden_dim]
  // router_score shape: [batch_size, n_experts]
  const long long hidden_dim = t->shape[1];
  const long long n_experts = router_scores->shape[1];

  // Get raw device pointers for weights, biases, input, and output
  // Apply the layer offset to select the correct weights and biases for the current layer
  const float *w_router_ptr = (float *)w_router->d_buf + 1ll * layer_offset * n_experts * hidden_dim;
  const float *b_router_ptr = (float *)b_router->d_buf + 1ll * layer_offset * n_experts;
  const float *t_ptr = (float *)t->d_buf;
  float *r_ptr = (float *)router_scores->d_buf;

  // --- Kernel Launch Configuration ---

  // Define tuning parameters for the kernel. These should match the kernel's template arguments.
  const int WARPS_PER_BLOCK = 4;  // Number of warps per thread block in the y-dimension
  const int TILE = 4;           // Size of the tile loaded into shared memory
  const int warpSize = 64;        // Number of threads in a warp (GPU architecture dependent)

  // Configure the thread block dimensions.
  // Each block has `warpSize` threads in the x-dimension and `WARPS_PER_BLOCK` in the y-dimension.
  dim3 block_dim(warpSize, WARPS_PER_BLOCK);

  // Configure the grid dimensions.
  // The grid's x-dimension is calculated to cover all `n_experts` (output features).
  // The grid's y-dimension is set to `batch_size`, so each `blockIdx.y` corresponds to a batch item.
  dim3 grid_dim((n_experts + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, cur_batch_size);

  // Calculate the required dynamic shared memory for the tile.
  size_t shared_mem_size = TILE * sizeof(float);

  // Launch the batched GEMM kernel
  gemm_kernel_batched<WARPS_PER_BLOCK, TILE><<<grid_dim, block_dim, shared_mem_size, stream>>>(
    w_router_ptr,   // W (weights)
    t_ptr,          // x (input)
    b_router_ptr,   // bias
    r_ptr,          // out (output)
    n_experts,      // out_features
    hidden_dim,     // in_features
    cur_batch_size      // batch_size
  );

  // --- End Kernel Launch ---

  // Transfer the results back from the GPU if required and synchronize the stream
  if (r_from_device) {
    router_scores->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

// ---------- TopK + softmax(k) ----------
__global__ void batched_topk_softmax_kernel(
  const float *r, int batch_size, int n_experts, int k,
  float *topk_vals, int *topk_idx
) {
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

void topk_softmax_batched(
  Tensor *r,            // Shape: [batch_size, n_experts]
  Tensor *topk_vals,    // Shape: [batch_size, k]
  TensorI32 *topk_idx,  // Shape: [batch_size, k]
  bool r_to_device, bool topk_vals_from_device, bool topk_idx_from_device,
  int cur_batch_size, hipStream_t stream
) {
  if (r_to_device) {
    r->to_device(stream);
  }

  // Extract dimensions from tensor shapes
  // r shape: [batch_size, n_experts]
  // topk_idx shape: [batch_size, k]
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
  dim3 grid_dim(cur_batch_size, 1, 1);
  dim3 block_dim(1, 1, 1);

  batched_topk_softmax_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
    r_ptr, cur_batch_size, num_experts, experts_per_token,
    topk_vals_ptr, topk_idx_ptr
  );
  
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

// ---------- MoE apply TopK  ----------
__global__ void SwiGLU_kernel_batched(
  const float *interleaved_in, float *swiglu_out, int inter_dim,
  float clamp_limit
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < inter_dim) {
    float gate_val = interleaved_in[2 * i];
    float up_val = interleaved_in[2 * i + 1];

    const float alpha = 1.702f;

    // 1. Clamping
    if (clamp_limit > 0.0f) {
      if (gate_val > clamp_limit)
        gate_val = clamp_limit;

      if (up_val > clamp_limit)
        up_val = clamp_limit;
      if (up_val < -clamp_limit)
        up_val = -clamp_limit;
    }

    float silu_val = gate_val * (1.0f / (1.0f + expf(-alpha * gate_val)));

    swiglu_out[i] = silu_val * (up_val + 1.0f);
  }
}

/**
 * @brief Performs batched matrix-vector multiplication for a set of selected experts across a batch.
 *
 * This kernel computes `out[b][j] = W_experts[j] * x[b] + b_experts[j]` for each batch item `b` and for each j in topk_idx[b].
 *
 * @param W Pointer to the weight tensor for ALL experts, shaped (num_experts, out_features, in_features).
 * @param x Pointer to the batch of input vectors, shaped (batch_size, in_features).
 * @param bias Pointer to the bias tensor for ALL experts, shaped (num_experts, out_features).
 * @param out Pointer to the output tensor, shaped (batch_size, k, out_features).
 * @param topk_idx Pointer to an array of integers containing the indices of the k selected experts for each batch item, shaped (batch_size, k).
 * @param batch_size The number of items in the batch.
 * @param k The number of experts to process per batch item.
 * @param out_features The output dimension of a single expert's matrix.
 * @param in_features The input dimension of a single expert's matrix.
 * @param offset_input If true, the input 'x' is treated as having a shape of [batch_size, k, in_features], used for the second GEMM.
 */
__global__ void moe_matmul_kernel_bf16_weights_batch(
  const bf16 *W, const float *x, const bf16 *bias, float *out,
  const int *topk_idx, int batch_size, int k, int out_features,
  int in_features, bool offset_input
) {
  // A global index for each output element across the entire batch of k experts.
  // Total work items = batch_size * k * out_features.
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < (long long)batch_size * k * out_features) {
    // 1. Decompose the global index to find batch, expert, and row.
    int work_per_batch = k * out_features;
    int batch_idx = global_idx / work_per_batch;
    int local_idx = global_idx % work_per_batch;
    int expert_k_idx = local_idx / out_features; // which of the k experts (0 to k-1)
    int row = local_idx % out_features;          // which output row for that expert

    // 2. Get the actual expert ID from the top-k index array for the current batch item.
    int expert_id = topk_idx[batch_idx * k + expert_k_idx];

    // 3. Calculate offsets for this expert's data (same as before).
    size_t expert_w_offset = (size_t)expert_id * out_features * in_features;
    size_t expert_b_offset = (size_t)expert_id * out_features;

    // Pointer to the specific row of the selected expert's weight matrix.
    const bf16 *W_row = W + expert_w_offset + (size_t)row * in_features;

    // 4. Pointer to the correct input vector based on batch_idx and offset_input flag.
    const float *x_cur;
    if (offset_input) {
      // Used for the 2nd GEMM where input is gate_up [batch_size, k, inter_dim]
      x_cur = x + (size_t)batch_idx * k * in_features + (size_t)expert_k_idx * in_features;
    } else {
      // Used for the 1st GEMM where input is t [batch_size, hidden_dim]
      x_cur = x + (size_t)batch_idx * in_features;
    }

    // Inner loop for the dot product remains the same.
    double sum = 0.0;
    for (int i = 0; i < in_features; i++) {
      float w_val = static_cast<float>(W_row[i]);
      sum += (double)w_val * x_cur[i];
    }

    if (bias != nullptr) {
      sum += static_cast<float>(bias[expert_b_offset + row]);
    }
    
    // Output is shaped [batch_size, k, out_features], so global_idx maps directly.
    out[global_idx] = (float)sum;
  }
}

/**
 * @brief Performs a batched, weighted accumulation of expert outputs for a whole batch.
 *
 * This kernel calculates `accumulator[b][j] += topk_weights[b][i] * expert_outputs[b][i][j]`
 * for each batch item `b`, expert `i`, and element `j`.
 *
 * @param expert_outputs Pointer to the outputs of all k experts for the whole batch, shaped [batch_size, k, hidden_dim].
 * @param topk_weights   Pointer to the weights for the k selected experts for the whole batch, shaped [batch_size, k].
 * @param accumulator    Pointer to the final output tensor to accumulate into, shaped [batch_size, hidden_dim].
 * @param batch_size     The number of items in the batch.
 * @param k              The number of selected experts per item.
 * @param hidden_dim     The dimension of each expert's output.
 */
__global__ void batched_weighted_accumulate_kernel_batch(
  const float *expert_outputs, const float *topk_weights, float *accumulator,
  int batch_size, int k, int hidden_dim
) {
  // Global index across all elements in the expert_outputs tensor.
  // Total work items = batch_size * k * hidden_dim.
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < (long long)batch_size * k * hidden_dim) {
    // 1. Decompose the global index.
    int work_per_batch = k * hidden_dim;
    int batch_idx = global_idx / work_per_batch;
    int local_idx = global_idx % work_per_batch;
    int expert_k_idx = local_idx / hidden_dim;  // which of the k experts (0 to k-1)
    int element_idx = local_idx % hidden_dim; // which element in the vector

    // 2. Fetch the corresponding weight for this batch item and expert.
    float weight = topk_weights[batch_idx * k + expert_k_idx];

    // 3. Fetch the output value from the expert tensor.
    float expert_out_val = expert_outputs[global_idx];

    // 4. Atomically add the weighted value to the correct accumulator for the current batch item.
    atomicAdd(&accumulator[batch_idx * hidden_dim + element_idx], weight * expert_out_val);
  }
}

void moe_apply_topk_batched(
  Tensor *t,            // Shape [batch_size, hidden_dim]
  const Tensor *W1,     // Shape [n_layers, n_experts, 2*inter_dim, hidden_dim]
  const Tensor *b1,     // Shape [n_layers, n_experts, 2*inter_dim]
  const Tensor *W2,     // Shape [n_layers, n_experts, hidden_dim, inter_dim]
  const Tensor *b2,     // Shape [n_layers, n_experts, hidden_dim]
  TensorI32 *topk_idx,  // Shape [batch_size, k]
  Tensor *topk_vals,    // Shape [batch_size, k]
  Tensor *mlp1_out,     // Shape [batch_size, k, 2*inter_dim]
  Tensor *gate_up,      // Shape [batch_size, k, inter_dim]
  Tensor *tb3,          // Shape [batch_size, k, hidden_dim]
  Tensor *e_agg,        // Shape [batch_size, hidden_dim]
  float clamp_limit, long long layer_offset, bool t_to_device,
  bool topk_idx_to_device, bool topk_vals_to_device, bool e_agg_from_device,
  int cur_batch_size, hipStream_t stream
) {
  if (t_to_device)
    t->to_device(stream);
  if (topk_idx_to_device)
    topk_idx->to_device(stream);
  if (topk_vals_to_device)
    topk_vals->to_device(stream);

  // Extract dimensions from tensor shapes
  const int hidden_dim = t->shape[1];
  const int inter_dim = W2->shape[2];
  const int k = topk_idx->shape[1];
  const int num_experts = b2->shape[1];
  const long long offset = layer_offset * num_experts;
  const long long inter_hidden = (long long)inter_dim * hidden_dim;

  // Get raw device pointers from tensors
  const float *t_ptr = (const float *)t->d_buf;
  const bf16 *W1_ptr = (const bf16 *)W1->d_buf + offset * 2 * inter_hidden;
  const bf16 *b1_ptr = (const bf16 *)b1->d_buf + offset * 2 * inter_dim;
  const bf16 *W2_ptr = (const bf16 *)W2->d_buf + offset * inter_hidden;
  const bf16 *b2_ptr = (const bf16 *)b2->d_buf + offset * hidden_dim;
  const int *topk_idx_ptr = topk_idx->d_buf;
  const float *topk_vals_ptr = (const float *)topk_vals->d_buf;
  float *mlp1_out_ptr = (float *)mlp1_out->d_buf;
  float *tb3_ptr = (float *)tb3->d_buf;
  float *gate_up_ptr = (float *)gate_up->d_buf;
  float *e_agg_ptr = (float *)e_agg->d_buf;

  // Clear the accumulator buffer for the entire batch
  memset_tensor(e_agg, 0, false, true, stream);

  // --- Step 1: First GEMM (Batched) ---
  // z = W1 * t + b1 -> write to mlp1_out
  size_t total_threads_mlp1 = (size_t)cur_batch_size * k * 2 * inter_dim;
  moe_matmul_kernel_bf16_weights_batch<<<(total_threads_mlp1 + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE,
              DEFAULT_BLOCK_SIZE, 0, stream>>>(
      W1_ptr, t_ptr, b1_ptr, mlp1_out_ptr, topk_idx_ptr, 
      cur_batch_size, k, 2 * inter_dim, hidden_dim, false);

  // --- Step 2: SwiGLU Activation (Applied over the whole batch) ---
  // swiglu = silu(gate) * up -> write to gate_up
  // The kernel is applied element-wise, so we just need to launch enough threads
  // to cover the entire batched tensor: batch_size * k * inter_dim.
  size_t swiglu_threads = (size_t)cur_batch_size * k * inter_dim;
  SwiGLU_kernel_batched<<<(swiglu_threads + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE, 0, stream>>>(mlp1_out_ptr, gate_up_ptr, swiglu_threads, clamp_limit);

  // --- Step 3: Second GEMM (Batched) ---
  // y = W2 * swiglu + b2 -> write to tb3
  size_t total_threads_mlp2 = (size_t)cur_batch_size * k * hidden_dim;
  moe_matmul_kernel_bf16_weights_batch<<<(total_threads_mlp2 + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE,
              DEFAULT_BLOCK_SIZE, 0, stream>>>(
      W2_ptr, gate_up_ptr, b2_ptr, tb3_ptr, topk_idx_ptr,
      cur_batch_size, k, hidden_dim, inter_dim, true);

  // --- Step 4: Weighted Accumulation (Batched) ---
  // e_agg_inout += weight * y
  size_t total_threads_accum = (size_t)cur_batch_size * k * hidden_dim;
  batched_weighted_accumulate_kernel_batch<<<(total_threads_accum + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE,
      DEFAULT_BLOCK_SIZE, 0, stream>>>(
      tb3_ptr, topk_vals_ptr, e_agg_ptr, cur_batch_size, k, hidden_dim);

  if (e_agg_from_device) {
      e_agg->from_device(stream);
      CHECK_HIP(hipStreamSynchronize(stream));
  }
}

// ---------- Classifier & Residuals ----------
void classifier_gemm_batched(
  const Tensor *W_out,  // Shape: [vocab_size, hidden_dim]
  Tensor *x,            // Shape: [batch_size, hidden_dim]
  Tensor *logits,       // Shape: [batch_size, vocab_size]
  bool x_to_device, bool logits_from_device,
  int cur_batch_size, hipStream_t stream
) {
  // GpuTimer timer("classifier_batched");
  if (x_to_device) {
    x->to_device(stream);
  }

  // Extract dimensions from tensor shapes
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
  dim3 grid_dim((vocab_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, cur_batch_size);
  
  // Shared memory is likely used by the kernel to cache the input vector `x` for faster access.
  size_t shmem_bytes = TILE * sizeof(float);

  // Launch the batched GEMM kernel
  gemm_kernel_batched<WARPS_PER_BLOCK, TILE><<<grid_dim, block_dim, shmem_bytes, stream>>>(
    W_out_ptr, x_ptr, nullptr, // No bias is used in this operation
    logits_ptr, vocab_size, hidden_dim, cur_batch_size
  );

  if (logits_from_device) {
    logits->from_device(stream);
    // Block until the kernel and data transfer are complete
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}
