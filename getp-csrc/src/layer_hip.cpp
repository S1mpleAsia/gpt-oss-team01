#include "../include/layer_hip.hpp"
#include <cmath>
#include <cfloat>

#define DEFAULT_BLOCK_SIZE 256

void embedding_lookup(Tensor *embedding,        // (vocab_size, hidden_dim)
                      int token_id, Tensor *x,  // (hidden_dim, )
                      bool x_from_device, hipStream_t stream) {
  // GpuTimer timer("embedding_lookup");
  // Assure that embedding->dtype == x->dtype
  const size_t hidden_dim = x->shape[1];
  if (x->dtype == DType::BF16) {
    bf16 *src = (bf16 *)embedding->d_buf + (size_t)token_id * hidden_dim;
    CHECK_HIP(hipMemcpyAsync((bf16 *)x->d_buf, src, hidden_dim * sizeof(bf16),
                             hipMemcpyDeviceToDevice, stream));
  } else {
    float *src = (float *)embedding->d_buf + (size_t)token_id * hidden_dim;
    CHECK_HIP(hipMemcpyAsync((float *)x->d_buf, src, hidden_dim * sizeof(float),
                             hipMemcpyDeviceToDevice, stream));
  }

  if (x_from_device) {
    x->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

__global__ void rmsnorm_kernel_double_precision(const float *x, const float *w, float *out,
                                                int hidden_dim, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  const float *x_row = x + row * hidden_dim;
  const float *w_row = w + row * hidden_dim;
  float *o_row = out + row * hidden_dim;

  // --- THAY ĐỔI 1: Sử dụng 'double' cho biến tích lũy ---
  double acc = 0.0;

  // Vòng lặp tính tổng bình phương của mỗi thread
  for (int j = tid; j < hidden_dim; j += block_size) {
    float v = x_row[j];
    acc += v * v;  // Ép kiểu để phép nhân và cộng thực hiện ở double
  }

  // --- THAY ĐỔI 2: Dùng shared memory cho toàn bộ quá trình reduction ---
  // (Thay thế cho warp shuffle và logic warp_sum phức tạp)
  extern __shared__ double s_partials[];  // Khai báo dynamic shared memory
  s_partials[tid] = acc;
  __syncthreads();  // Đảm bảo mọi thread đã ghi xong tổng cục bộ của mình

  // Thực hiện reduction song song trong shared memory
  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_partials[tid] += s_partials[tid + s];
    }
    __syncthreads();  // Đồng bộ sau mỗi vòng lặp reduction
  }

  // --- THAY ĐỔI 3: Chỉ thread 0 tính toán kết quả cuối cùng ---
  // Tạo một biến shared để lưu kết quả và chia sẻ cho các thread khác
  __shared__ double final_inv_rms;
  if (tid == 0) {
    double block_sum = s_partials[0];
    double mean = block_sum / hidden_dim;
    mean = mean + 1e-5f;
    final_inv_rms = 1.0f / sqrtf(mean);
  }
  __syncthreads();  // Đảm bảo mọi thread đều thấy giá trị final_inv_rms

  // --- THAY ĐỔI 4: Tất cả các thread áp dụng chuẩn hóa ---
  // Mỗi thread đọc giá trị inv_rms từ shared memory và thực hiện phép tính
  for (int j = tid; j < hidden_dim; j += block_size) {
    o_row[j] = w_row[j] * (final_inv_rms * x_row[j]);
  }
}

void rmsnorm(Tensor *x, Tensor *w, Tensor *out, long long layer_offset, bool x_to_device,
             bool out_from_device, float eps, hipStream_t stream) {
  // GpuTimer timer("rmsnorm");
  if (x_to_device)
    x->to_device(stream);

  constexpr int batch_size = 1;  // Change this when using batching
  constexpr int num_threads = 256;
  const dim3 block_dim(num_threads), grid_dim(batch_size);

  size_t shared_mem_size = num_threads * sizeof(double);

  const int hidden_dim = x->shape[1];

  const float *x_ptr = (float *)x->d_buf;
  const float *w_ptr = (float *)w->d_buf + 1ll * layer_offset * hidden_dim;
  float *out_ptr = (float *)out->d_buf;

  rmsnorm_kernel_double_precision<<<grid_dim, block_dim, shared_mem_size, stream>>>(
    x_ptr, w_ptr, out_ptr, hidden_dim, eps);
  CHECK_HIP(hipGetLastError());
  // RMSNorm_kernel<<<grid_dim, block_dim, 0, stream>>>(x, w, out, hidden_dim, eps);

  if (out_from_device) {
    out->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }

  return v;
}

template <int WARPS_PER_BLOCK, int TILE>
__global__ void gemv_kernel(const float *W, const float *x, const float *bias, float *out,
                            int out_features, int in_features) {
  extern __shared__ float local_x[];

  const int lane = threadIdx.x;
  const int w = threadIdx.y;
  const int row = blockIdx.x * WARPS_PER_BLOCK + w;

  if (row >= out_features)
    return;

  float acc = 0.0f;
  const int row_base = row * in_features;

  for (int k0 = 0; k0 < in_features; k0 += TILE) {
    const int tile_len = min(TILE, in_features - k0);

    for (int t = w * warpSize + lane; t < tile_len; t += WARPS_PER_BLOCK * warpSize) {
      local_x[t] = x[k0 + t];
    }
    __syncthreads();

    for (int t = lane; t < tile_len; t += warpSize) {
      float wv = W[row_base + k0 + t];
      acc += wv * local_x[t];
    }
    __syncthreads();
  }

  acc = warp_reduce_sum(acc);

  if (lane == 0) {
    if (bias)
      acc += bias[row];
    out[row] = acc;
  }
}

// W (out, in) * x(in, ) -> y(out, )
__global__ void matmul_kernel(const float *W, const float *x, const float *bias, float *out,
                              int out_features, int in_features) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < out_features) {
    float sum = 0.0f;
    const float *W_row = W + (size_t)row * in_features;
    for (int i = 0; i < in_features; i++) {
      sum += W_row[i] * x[i];
    }
    if (bias != nullptr) {
      sum += bias[row];
    }
    out[row] = sum;
  }
}

void qkv_gemm(Tensor *x, const Tensor *W_qkv, const Tensor *b_qkv, Tensor *qkv,
              long long layer_offset, bool x_to_device, bool qkv_from_device, hipStream_t stream) {
  // GpuTimer timer("qkv_gemm");
  if (x_to_device)
    x->to_device(stream);

  const int in_features = x->num_elem();
  const int out_features = qkv->num_elem();

  const float *x_ptr = (float *)x->d_buf;
  const float *w_qkv_ptr = (float *)W_qkv->d_buf + 1ll * layer_offset * out_features * in_features;
  const float *b_qkv_ptr = (float *)b_qkv->d_buf + 1ll * layer_offset * out_features;
  float *qkv_ptr = (float *)qkv->d_buf;

  constexpr int WARPS_PER_BLOCK = 16;
  constexpr int TILE = 1024;

  dim3 block_dim(warpSize, WARPS_PER_BLOCK);
  dim3 grid_dim((out_features + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
  size_t shmem_bytes = TILE * sizeof(float);

  gemv_kernel<WARPS_PER_BLOCK, TILE><<<grid_dim, block_dim, shmem_bytes, stream>>>(
    w_qkv_ptr, x_ptr, b_qkv_ptr, qkv_ptr, out_features, in_features);

  if (qkv_from_device) {
    qkv->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

void split_qkv(Tensor *qkv, int head_dim, int n_q, int n_kv, Tensor *q, Tensor *k, Tensor *v,
               bool qkv_to_device, bool q_from_device, bool k_from_device, bool v_from_device,
               hipStream_t stream) {
  // GpuTimer timer("split_qkv");
  // Assume they are all copy on device for now
  if (qkv_to_device)
    qkv->to_device(stream);

  size_t offset = 0;
  memcpy_tensor(q, qkv, 0, offset, n_q * head_dim, false, true, stream);
  offset += n_q * head_dim;
  memcpy_tensor(k, qkv, 0, offset, n_kv * head_dim, false, true, stream);
  offset += n_kv * head_dim;
  memcpy_tensor(v, qkv, 0, offset, n_kv * head_dim, false, true, stream);

  if (q_from_device)
    q->from_device(stream);
  if (k_from_device)
    k->from_device(stream);
  if (v_from_device)
    v->from_device(stream);
  if (q_from_device || k_from_device || v_from_device) {
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

__global__ void add_vector_kernel(float *y, const float *b, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    y[i] += b[i];
  }
}

void add_vector(Tensor *y, Tensor *b, bool y_to_device, bool b_to_device, bool y_from_device,
                hipStream_t stream) {
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
  add_vector_kernel<<<grid_dim, block_dim, 0, stream>>>(y_ptr, b_ptr, len);

  if (y_from_device) {
    y->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

__global__ void linear_bias_residual_kernel(const float *W, const float *x, const float *bias,
                                            float *x_resid_inout, int in_features,
                                            int out_features) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < out_features) {
    float sum = 0.0f;
    const float *W_row = W + (size_t)row * in_features;
    for (int i = 0; i < in_features; i++) {
      sum += W_row[i] * x[i];
    }
    if (bias != nullptr) {
      sum += bias[row];
    }
    x_resid_inout[row] += sum;
  }
}

void linear_bias_residual(const float *W, const float *x, const float *bias, float *x_resid_inout,
                          int in_features, int out_features, hipStream_t stream) {
  // GpuTimer timer("linear_bias_residual");
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((out_features + block_dim.x - 1) / block_dim.x);
  linear_bias_residual_kernel<<<grid_dim, block_dim, 0, stream>>>(W, x, bias, x_resid_inout,
                                                                  in_features, out_features);
}

//================================================================================================
// 5. RoPE (Rotary Positional Embedding)
//================================================================================================

__global__ void qkv_split_rope_kernel(float *qkv_out, float *q_out, float *k_pos, float *v_pos,
                                      const float *rope_cos_pos, const float *rope_sin_pos,
                                      int head_dim, int n_q, int n_kv) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int half_dim = head_dim / 2;

  // Total dimensions for Q, K, V
  int q_dims = n_q * head_dim;
  int k_dims = n_kv * head_dim;
  int v_dims = n_kv * head_dim;

  if (idx >= q_dims + k_dims + v_dims)
    return;

  if (idx < q_dims) {  // Processing a Q element
    int head_idx = idx / head_dim;
    int dim_idx_in_head = idx % head_dim;
    int rope_idx = dim_idx_in_head % half_dim;

    float cos_val = rope_cos_pos[rope_idx];
    float sin_val = rope_sin_pos[rope_idx];

    int partner_dim_offset;
    float partner_val;

    if (dim_idx_in_head < half_dim) {
      // This is the first half of the vector (x1)
      partner_dim_offset = half_dim;
      partner_val = qkv_out[idx + partner_dim_offset];
      q_out[idx] = qkv_out[idx] * cos_val - partner_val * sin_val;
    } else {
      // This is the second half of the vector (x2)
      partner_dim_offset = -half_dim;
      partner_val = qkv_out[idx + partner_dim_offset];
      q_out[idx] = qkv_out[idx] * cos_val + partner_val * sin_val;
    }

  } else if (idx < q_dims + k_dims) {  // Processing a K element
    int k_local_idx = idx - q_dims;
    int head_idx = k_local_idx / head_dim;
    int dim_idx_in_head = k_local_idx % head_dim;
    int rope_idx = dim_idx_in_head % half_dim;

    float cos_val = rope_cos_pos[rope_idx];
    float sin_val = rope_sin_pos[rope_idx];

    int partner_dim_offset;
    float partner_val;

    if (dim_idx_in_head < half_dim) {
      // First half
      partner_dim_offset = half_dim;
      partner_val = qkv_out[idx + partner_dim_offset];
      k_pos[k_local_idx] = qkv_out[idx] * cos_val - partner_val * sin_val;
    } else {
      // Second half
      partner_dim_offset = -half_dim;
      partner_val = qkv_out[idx + partner_dim_offset];
      k_pos[k_local_idx] = qkv_out[idx] * cos_val + partner_val * sin_val;
    }

  } else {  // Processing a V element (simple copy)
    int v_local_idx = idx - q_dims - k_dims;
    v_pos[v_local_idx] = qkv_out[idx];
  }
}

void qkv_split_rope(Tensor *qkv_out, Tensor *q_out, Tensor *k_pos, Tensor *v_pos,
                    const Tensor *rope_cos_pos, const Tensor *rope_sin_pos, int head_dim, int n_q,
                    int n_kv, int pos, bool qkv_out_to_device, bool q_out_from_device,
                    bool k_pos_from_device, bool v_pos_from_device, hipStream_t stream) {
  // GpuTimer timer("qkv_split_rope");
  if (qkv_out_to_device)
    qkv_out->to_device(stream);

  int offset = pos * (head_dim / 2);

  float *qkv_out_ptr = (float *)qkv_out->d_buf;
  float *q_out_ptr = (float *)q_out->d_buf;
  float *k_pos_ptr = (float *)k_pos->d_buf;
  float *v_pos_ptr = (float *)v_pos->d_buf;
  const float *rope_cos_pos_ptr = (float *)rope_cos_pos->d_buf + 1ll * offset;
  const float *rope_sin_pos_ptr = (float *)rope_sin_pos->d_buf + 1ll * offset;

  int total_dims = (n_q + 2 * n_kv) * head_dim;
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((total_dims + block_dim.x - 1) / block_dim.x);
  qkv_split_rope_kernel<<<grid_dim, block_dim, 0, stream>>>(qkv_out_ptr, q_out_ptr, k_pos_ptr,
                                                            v_pos_ptr, rope_cos_pos_ptr,
                                                            rope_sin_pos_ptr, head_dim, n_q, n_kv);

  if (q_out_from_device)
    q_out->from_device(stream);
  if (k_pos_from_device)
    k_pos->from_device(stream);
  if (v_pos_from_device)
    v_pos->from_device(stream);
  if (q_out_from_device || k_pos_from_device || v_pos_from_device) {
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

//================================================================================================
// 6. Attention
//================================================================================================

__global__ void attention_kernel(const float *q, const float *K_cache, const float *V_cache,
                                 const float *mask_row,
                                 const float *attn_sinks,  // Con trỏ tới mảng các sink
                                 float *tb, int head_dim, int n_q, int kv_mul, int kv_dim,
                                 int seq_len /* pos + 1 */) {
  int head_idx = blockIdx.x;
  if (head_idx >= n_q)
    return;

  const float attn_sink = attn_sinks[head_idx];
  const float *q_head = q + head_idx * head_dim;
  float *tb_head = tb + head_idx * head_dim;

  extern __shared__ char s_data[];
  float *s_scores = (float *)s_data;
  double *s_output = (double *)(s_data + ((size_t)seq_len + 1) * sizeof(float));

  int kv_head_idx = head_idx / kv_mul;
  float scale = rsqrtf((float)head_dim);

  for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
    const float *k_vec = K_cache + (size_t)t * kv_dim + (size_t)kv_head_idx * head_dim;

    double score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      score += (double)q_head[i] * k_vec[i];
    }
    score *= scale;

    if (mask_row != nullptr) {
      score += mask_row[t];
    }
    s_scores[t] = (float)score;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    s_scores[seq_len] = attn_sink;
    int softmax_len = seq_len + 1;

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

    for (int i = 0; i < head_dim; i++)
      s_output[i] = 0.0;

    for (int t = 0; t < seq_len; t++) {
      double prob = (double)s_scores[t] / denom;
      const float *v_vec = V_cache + (size_t)t * kv_dim + (size_t)kv_head_idx * head_dim;
      for (int i = 0; i < head_dim; i++) {
        s_output[i] += prob * v_vec[i];
      }
    }
  }
  __syncthreads();

  // Step 4: Ghi kết quả ra global memory
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    tb_head[i] = (float)s_output[i];
  }
}

void single_query_attn(Tensor *q, Tensor *K_cache, Tensor *V_cache, Tensor *mask,
                       Tensor *attn_sinks, Tensor *tb, int head_dim, int n_q, int kv_mul,
                       int kv_dim, int seq_len, int sliding_window, int pos, long long layer_offset,
                       bool q_to_device, bool k_cache_to_device, bool v_cache_to_device,
                       bool mask_to_device, bool tb_from_device, hipStream_t stream) {
  // GpuTimer timer("single_query_attn");
  if (q_to_device)
    q->to_device(stream);
  if (k_cache_to_device)
    K_cache->to_device(stream);
  if (v_cache_to_device)
    V_cache->to_device(stream);
  if (mask_to_device)
    mask->to_device(stream);

  // 1. Get raw device pointers from Tensor objects
  const float *q_ptr = (float *)q->d_buf;
  float *tb_ptr = (float *)tb->d_buf;

  // 2. Calculate the offset to the current layer's KV cache
  // This logic is identical to `loff = 1ll * l * loff_one` in model.cpp
  long long cache_layer_offset_elements = 1ll * layer_offset * seq_len * kv_dim;
  const float *K_cache_ptr = (const float *)K_cache->d_buf + cache_layer_offset_elements;
  const float *V_cache_ptr = (const float *)V_cache->d_buf + cache_layer_offset_elements;

  // 3. Calculate the pointer to the current row in the attention mask
  const float *mask_row_ptr = nullptr;
  // The mask is applied for specific layers with a sliding window, as in AttnScoresAllHeads
  if (sliding_window > 0 && (layer_offset % 2 == 0)) {
    if (mask != nullptr && mask->d_buf != nullptr) {
      mask_row_ptr = (const float *)mask->d_buf + (long long)pos * seq_len;
    }
  }

  // 4. Calculate the pointer to the current layer's attention sinks
  const float *attn_sinks_ptr = (const float *)attn_sinks->d_buf + layer_offset * n_q;

  /*
    SingleQueryAttentionGPU(q_ptr, K_cache_ptr, V_cache_ptr, mask_row_ptr,
                            attn_sinks_ptr, tb_ptr, head_dim, n_q, kv_mul,
                            kv_dim, pos + 1, pos, stream);
    */

  dim3 grid_dim(n_q);
  dim3 block_dim(256);

  size_t shared_mem_size = ((size_t)pos + 2) * sizeof(float) + (size_t)head_dim * sizeof(double);

  attention_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
    q_ptr, K_cache_ptr, V_cache_ptr, mask_row_ptr, attn_sinks_ptr, tb_ptr, head_dim, n_q, kv_mul,
    kv_dim, pos + 1);

  if (tb_from_device) {
    tb->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

void attn_out_project(Tensor *tb, const Tensor *W_o, const Tensor *b_o, Tensor *y,
                      long long layer_offset, bool tb_to_device, bool y_from_device,
                      hipStream_t stream) {
  // GpuTimer timer("attn_out_project");
  if (tb_to_device)
    tb->to_device(stream);

  const long long n_q_hd = tb->num_elem();
  const long long hidden = y->num_elem();
  const float *w_o_ptr = (float *)W_o->d_buf + 1ll * layer_offset * n_q_hd * hidden;
  const float *b_o_ptr = (float *)b_o->d_buf + 1ll * layer_offset * hidden;
  const float *tb_ptr = (float *)tb->d_buf;
  float *y_ptr = (float *)y->d_buf;
  // Linear(tb->buf, n_q_hd, w_o_ptr, b_o_ptr, hidden, y->buf);

  constexpr int WARPS_PER_BLOCK = 16;
  constexpr int TILE = 1024;

  dim3 block_dim(warpSize, WARPS_PER_BLOCK);
  dim3 grid_dim((hidden + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
  size_t shmem_bytes = TILE * sizeof(float);
  gemv_kernel<WARPS_PER_BLOCK, TILE>
    <<<grid_dim, block_dim, shmem_bytes, stream>>>(w_o_ptr, tb_ptr, b_o_ptr, y_ptr, hidden, n_q_hd);
  // matmul_kernel<<<grid_dim, block_dim, 0, stream>>>(w_o_ptr, tb_ptr, b_o_ptr, y_ptr, hidden,
  //                                                   n_q_hd);

  if (y_from_device) {
    y->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

//================================================================================================
// 7. MoE (Mixture of Experts)
//================================================================================================
void router_gemm(const Tensor *w_router, Tensor *t, const Tensor *b_router, Tensor *router_score,
                 long long layer_offset, bool t_to_device, bool r_from_device, hipStream_t stream) {
  // GpuTimer timer("router_gemm");
  if (t_to_device)
    t->to_device(stream);

  const long long hidden = t->num_elem();
  const long long n_experts = router_score->num_elem();
  const float *w_router_ptr = (float *)w_router->d_buf + 1ll * layer_offset * n_experts * hidden;
  const float *b_router_ptr = (float *)b_router->d_buf + 1ll * layer_offset * n_experts;
  const float *t_ptr = (float *)t->d_buf;
  float *r_ptr = (float *)router_score->d_buf;

  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((n_experts + block_dim.x - 1) / block_dim.x);
  matmul_kernel<<<grid_dim, block_dim, 0, stream>>>(w_router_ptr, t_ptr, b_router_ptr, r_ptr,
                                                    n_experts, hidden);

  if (r_from_device) {
    router_score->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

__global__ void topk_softmax_kernel(const float *r, int n_experts, int k, float *topk_vals,
                                    int *topk_idx) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  extern __shared__ char s_data[];
  float *temp_vals = (float *)s_data;
  int *temp_idx = (int *)(s_data + n_experts * sizeof(float));

  // Copy input data to shared memory
  for (int i = 0; i < n_experts; ++i) {
    temp_vals[i] = r[i];
    temp_idx[i] = i;
  }

  // Partial selection sort for top-k (MORE EFFICIENT)
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

  // Apply softmax to top-k values
  double max_val = -DBL_MAX;
  if (k > 0) {
    max_val = temp_vals[0];
  }

  double denom = 0.0;
  for (int i = 0; i < k; ++i) {
    denom += expf(temp_vals[i] - (float)max_val);
  }

  // Output results
  for (int i = 0; i < k; ++i) {
    topk_vals[i] = expf(temp_vals[i] - (float)max_val) / (float)denom;
    topk_idx[i] = temp_idx[i];
  }
}

void topk_softmax(Tensor *r, Tensor *topk_vals, TensorI32 *topk_idx, bool r_to_device,
                  bool topk_vals_from_device, bool topk_idx_from_device, hipStream_t stream) {
  // GpuTimer timer("topk_softmax");
  if (r_to_device)
    r->to_device(stream);

  const int num_experts = r->num_elem();
  const int experts_per_token = topk_idx->num_elem();

  const float *r_ptr = (float *)r->d_buf;
  float *topk_vals_ptr = (float *)topk_vals->d_buf;
  int *topk_idx_ptr = topk_idx->d_buf;

  // shared_mem_size exceeds when num_experts > 32
  size_t shared_mem_size = (size_t)num_experts * (sizeof(float) + sizeof(int));
  topk_softmax_kernel<<<1, 1, shared_mem_size, stream>>>(r_ptr, num_experts, experts_per_token,
                                                         topk_vals_ptr, topk_idx_ptr);
  CHECK_HIP(hipGetLastError());

  if (topk_vals_from_device)
    topk_vals->from_device(stream);
  if (topk_idx_from_device)
    topk_idx->from_device(stream);
  if (topk_vals_from_device || topk_idx_from_device) {
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

__global__ void SwiGLU_kernel(const float *interleaved_in, float *swiglu_out, int inter_dim,
                              float clamp_limit) {
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
 * @brief Performs batched matrix-vector multiplication for a set of selected experts in an MoE layer.
 *
 * This kernel computes `out[j] = W_experts[j] * x + b_experts[j]` for each j in topk_idx.
 *
 * @param W Pointer to the weight tensor for ALL experts, shaped (num_experts, out_features, in_features).
 * @param x Pointer to the single input vector, shaped (in_features).
 * @param bias Pointer to the bias tensor for ALL experts, shaped (num_experts, out_features).
 * @param out Pointer to the output tensor, shaped (k, out_features).
 * @param topk_idx Pointer to an array of integers containing the indices of the k selected experts.
 * @param k The number of experts to process (the 'k' in top-k).
 * @param out_features The output dimension of a single expert's matrix (number of rows).
 * @param in_features The input dimension of a single expert's matrix (number of columns).
 */
__global__ void moe_matmul_kernel_bf16_weights(const bf16 *W, const float *x, const bf16 *bias,
                                               float *out, const int *topk_idx, int k,
                                               int out_features, int in_features,
                                               bool offset_input) {
  // A global index for each output element across all k experts.
  // Total work items = k * out_features.
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < k * out_features) {
    // 1. Decompose the global index to find which expert and which row this thread handles.

    // `expert_k_idx`: which of the k experts we are working on (from 0 to k-1).
    int expert_k_idx = global_idx / out_features;
    // `row`: which output row for that specific expert (from 0 to out_features-1).
    int row = global_idx % out_features;

    // 2. Get the actual expert ID from the top-k index array.
    int expert_id = topk_idx[expert_k_idx];

    // 3. Calculate the correct offsets for this expert's data.
    size_t expert_w_offset = (size_t)expert_id * out_features * in_features;
    size_t expert_b_offset = (size_t)expert_id * out_features;

    // Pointer to the specific row of the selected expert's weight matrix.
    const bf16 *W_row = W + expert_w_offset + (size_t)row * in_features;
    const float *x_cur = offset_input ? x + 1ll * expert_k_idx * in_features : x;

    // The inner loop for the dot product remains the same.
    double sum = 0.0;
    for (int i = 0; i < in_features; i++) {
      // Convert bfloat16 weight to float for calculation.
      float w_val = static_cast<float>(W_row[i]);
      sum += (double)w_val * x_cur[i];
    }

    if (bias != nullptr) {
      // Add the bias for the corresponding expert and row.
      sum += static_cast<float>(bias[expert_b_offset + row]);
    }

    // Write the result to the correct position in the output tensor.
    // The output is structured as [expert_0_output, expert_1_output, ...].
    // `global_idx` naturally maps to the correct flat index in the output.
    out[global_idx] = (float)sum;
  }
}

/**
 * @brief Performs a batched, weighted accumulation of expert outputs.
 *
 * This kernel replaces a loop of individual kernel launches. It calculates
 * `accumulator[j] += topk_weights[i] * expert_outputs[i][j]` for all experts 'i' and all elements 'j'.
 *
 * @param expert_outputs Pointer to a contiguous block of memory containing the outputs of all k experts,
 * shaped [k, hidden_dim]. This corresponds to the 'tb3' buffer.
 * @param topk_weights   Pointer to the weights for the k selected experts on the device.
 * @param accumulator    Pointer to the final output tensor to accumulate results into, shaped [hidden_dim].
 * @param k              The number of selected experts.
 * @param hidden_dim     The dimension of each expert's output.
 */
__global__ void batched_weighted_accumulate_kernel(const float *expert_outputs,
                                                   const float *topk_weights, float *accumulator,
                                                   int k, int hidden_dim) {
  // A global index for each output element across all k experts.
  // Total work items = k * hidden_dim.
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < k * hidden_dim) {
    // 1. Decompose the global index to find which expert and which element this thread handles.

    // `expert_k_idx`: which of the k experts we are working on (from 0 to k-1).
    int expert_k_idx = global_idx / hidden_dim;

    // `element_idx`: which element within the hidden_dim vector this thread is responsible for.
    int element_idx = global_idx % hidden_dim;

    // 2. Fetch the corresponding weight for this expert.
    float weight = topk_weights[expert_k_idx];

    // 3. Fetch the output value from the expert tensor.
    float expert_out_val = expert_outputs[global_idx];

    // 4. Atomically add the weighted value to the accumulator.
    // This is crucial because multiple threads (one for each expert) will be writing
    // to the same `accumulator[element_idx]` location.
    atomicAdd(&accumulator[element_idx], weight * expert_out_val);
  }
}

void moe_apply_topk(Tensor *t, const Tensor *W1, const Tensor *b1, const Tensor *W2,
                    const Tensor *b2, TensorI32 *topk_idx, Tensor *topk_vals, Tensor *mlp1_out,
                    Tensor *gate_up, Tensor *tb3, Tensor *e_agg, float clamp_limit,
                    long long layer_offset, bool t_to_device, bool topk_idx_to_device,
                    bool topk_vals_to_device, bool e_agg_from_device, hipStream_t stream) {
  // GpuTimer("moe_apply_topk");
  if (t_to_device)
    t->to_device(stream);
  if (topk_idx_to_device)
    topk_idx->to_device(stream);
  if (topk_vals_to_device)
    topk_vals->to_device(stream);

  // Extract dimensions from tensor shapes
  const int hidden_dim = t->shape[1];
  const int inter_dim = W2->shape[2];
  const int k = topk_idx->num_elem();
  const int num_experts = b2->shape[1];
  const long long offset = layer_offset * num_experts;
  const long long inter_hidden = inter_dim * hidden_dim;

  // Get raw device pointers from tensors
  const float *t_ptr = (const float *)t->d_buf;
  const bf16 *W1_ptr = (const bf16 *)W1->d_buf + 1ll * offset * 2 * inter_hidden;
  const bf16 *b1_ptr = (const bf16 *)b1->d_buf + 1ll * offset * 2 * inter_dim;
  const bf16 *W2_ptr = (const bf16 *)W2->d_buf + 1ll * offset * inter_hidden;
  const bf16 *b2_ptr = (const bf16 *)b2->d_buf + 1ll * offset * hidden_dim;
  const int *topk_idx_ptr = topk_idx->d_buf;
  const float *topk_vals_ptr = (float *)topk_vals->d_buf;
  float *mlp1_out_ptr = (float *)mlp1_out->d_buf;
  float *tb3_ptr = (float *)tb3->d_buf;
  float *gate_up_ptr = (float *)gate_up->d_buf;
  float *e_agg_ptr = (float *)e_agg->d_buf;

  // Xóa bộ đệm tích lũy kết quả
  memset_tensor(e_agg, 0, false, true, stream);

  // --- Bước 1: GEMM thứ nhất ---
  // z = W1 * t + b1 -> ghi vào mlp1_out
  size_t total_threads_mlp1 = 1ll * k * 2 * inter_dim;

  moe_matmul_kernel_bf16_weights<<<(total_threads_mlp1 + DEFAULT_BLOCK_SIZE - 1) /
                                     DEFAULT_BLOCK_SIZE,
                                   DEFAULT_BLOCK_SIZE, 0, stream>>>(
    W1_ptr, t_ptr, b1_ptr, mlp1_out_ptr, topk_idx_ptr, k, 2 * inter_dim, hidden_dim, false);

  // --- Bước 2: Kích hoạt SwiGLU ---
  // swiglu = silu(gate) * up -> ghi vào gate_up
  size_t swiglu_threads = k * inter_dim;

  SwiGLU_kernel<<<(swiglu_threads + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE,
                  DEFAULT_BLOCK_SIZE, 0, stream>>>(mlp1_out_ptr, gate_up_ptr, k * inter_dim,
                                                   clamp_limit);

  // --- Bước 3: GEMM thứ hai ---
  // y = W2 * swiglu + b2 -> ghi vào tb3
  size_t total_threads_mlp2 = 1ll * k * hidden_dim;

  moe_matmul_kernel_bf16_weights<<<(total_threads_mlp2 + DEFAULT_BLOCK_SIZE - 1) /
                                     DEFAULT_BLOCK_SIZE,
                                   DEFAULT_BLOCK_SIZE, 0, stream>>>(
    W2_ptr, gate_up_ptr, b2_ptr, tb3_ptr, topk_idx_ptr, k, hidden_dim, inter_dim, true);

  // --- Bước 4: Cộng dồn kết quả theo trọng số ---
  // e_agg_inout += weight * y
  size_t total_threads_accum = 1ll * k * hidden_dim;

  batched_weighted_accumulate_kernel<<<(total_threads_accum + DEFAULT_BLOCK_SIZE - 1) /
                                         DEFAULT_BLOCK_SIZE,
                                       DEFAULT_BLOCK_SIZE, 0, stream>>>(tb3_ptr, topk_vals_ptr,
                                                                        e_agg_ptr, k, hidden_dim);

  if (e_agg_from_device) {
    e_agg->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

static inline void moe_mlp1(const bf16 *W1_ptr, const float *t_ptr, const bf16 *b1_ptr,
                            float *mlp1_out_ptr, const int *topk_idx_ptr, int k, int inter_dim,
                            int hidden_dim, hipStream_t stream) {
  size_t total_threads = (size_t)k * 2 * inter_dim;
  dim3 grid_dim((total_threads + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE);
  dim3 block_dim(DEFAULT_BLOCK_SIZE);

  moe_matmul_kernel_bf16_weights<<<grid_dim, block_dim, 0, stream>>>(
    W1_ptr, t_ptr, b1_ptr, mlp1_out_ptr, topk_idx_ptr, k, 2 * inter_dim, hidden_dim, false);
}

/**
 * @brief Bước 2: Áp dụng hàm kích hoạt SwiGLU.
 * Tính toán: swiglu = silu(gate) * up
 */
static inline void moe_swiglu(const float *mlp1_out_ptr, float *gate_up_ptr, int k, int inter_dim,
                              float clamp_limit, hipStream_t stream) {
  size_t total_threads = (size_t)k * inter_dim;
  dim3 grid_dim((total_threads + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE);
  dim3 block_dim(DEFAULT_BLOCK_SIZE);

  SwiGLU_kernel<<<grid_dim, block_dim, 0, stream>>>(mlp1_out_ptr, gate_up_ptr, k * inter_dim,
                                                    clamp_limit);
}

/**
 * @brief Bước 3: Thực hiện phép nhân ma trận (GEMM) thứ hai.
 * Tính toán: y = W2 * swiglu + b2
 */
static inline void moe_mlp2(const bf16 *W2_ptr, const float *gate_up_ptr, const bf16 *b2_ptr,
                            float *tb3_ptr, const int *topk_idx_ptr, int k, int hidden_dim,
                            int inter_dim, hipStream_t stream) {
  size_t total_threads = (size_t)k * hidden_dim;
  dim3 grid_dim((total_threads + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE);
  dim3 block_dim(DEFAULT_BLOCK_SIZE);

  moe_matmul_kernel_bf16_weights<<<grid_dim, block_dim, 0, stream>>>(
    W2_ptr, gate_up_ptr, b2_ptr, tb3_ptr, topk_idx_ptr, k, hidden_dim, inter_dim, true);
}

/**
 * @brief Bước 4: Tích lũy có trọng số các kết quả đầu ra của expert.
 * Tính toán: e_agg_inout += weight * y
 */
static inline void moe_agg(const float *tb3_ptr, const float *topk_vals_ptr, float *e_agg_ptr,
                           int k, int hidden_dim, hipStream_t stream) {
  size_t total_threads = (size_t)k * hidden_dim;
  dim3 grid_dim((total_threads + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE);
  dim3 block_dim(DEFAULT_BLOCK_SIZE);

  batched_weighted_accumulate_kernel<<<grid_dim, block_dim, 0, stream>>>(tb3_ptr, topk_vals_ptr,
                                                                         e_agg_ptr, k, hidden_dim);
}

void classifier_gemm(const Tensor *W_out, Tensor *x, Tensor *logits, bool x_to_device,
                     bool logits_from_device, hipStream_t stream) {
  // GpuTimer timer("classifier");
  if (x_to_device)
    x->to_device(stream);

  const int hidden_dim = x->num_elem();
  const int vocab_size = logits->num_elem();

  const float *W_out_ptr = (float *)W_out->d_buf;
  const float *x_ptr = (float *)x->d_buf;
  float *logits_ptr = (float *)logits->d_buf;

  constexpr int WARPS_PER_BLOCK = 16;
  constexpr int TILE = 1024;

  dim3 block_dim(warpSize, WARPS_PER_BLOCK);
  dim3 grid_dim((vocab_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
  size_t shmem_bytes = TILE * sizeof(float);
  gemv_kernel<WARPS_PER_BLOCK, TILE><<<grid_dim, block_dim, shmem_bytes, stream>>>(
    W_out_ptr, x_ptr, nullptr, logits_ptr, vocab_size, hidden_dim);

  if (logits_from_device) {
    logits->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}
