#include "layer.hpp"
#include <cmath>
#include <cfloat>

#define DEFAULT_BLOCK_SIZE 256

void EmbeddingLookupGPU(const float *embedding,  // (vocab_size, hidden_dim)
                        int token_id,
                        float *x,  // (hidden_dim, )
                        int hidden_dim, hipStream_t stream) {
  const float *src = embedding + (size_t)token_id * hidden_dim;
  CHECK_HIP(hipMemcpyAsync(x, src, hidden_dim * sizeof(float), hipMemcpyDeviceToDevice, stream));
}

__global__ void AddVector_kernel(float *x, const float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] += y[i];
  }
}

void AddVectorGPU(float *x, const float *y, int n, hipStream_t stream) {
  dim3 block_dim(256);
  dim3 grid_dim((n + block_dim.x - 1) / block_dim.x);
  AddVector_kernel<<<grid_dim, block_dim, 0, stream>>>(x, y, n);
}

__global__ void rmsnorm_kernel(const float *x, const float *w, float *out, int hidden_dim,
                               float eps) {
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  const float *x_row = x;
  const float *w_row = w;
  float *o_row = out;

  double acc = 0.0;
  for (int j = tid; j < hidden_dim; j += block_size) {
    double val = (double)x_row[j];
    acc += val * val;
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
    o_row[j] = (float)(w_row[j] * (final_inv_rms * x_row[j]));
  }
}

/*
  x - (B, hidden_dim)
  w - (hidden_dim)
  out - (B, hidden_dim)
*/
void RMSNormGPU(const float *x, const float *w, float *out, int hidden_dim, float eps,
                hipStream_t stream) {
  // constexpr int batch_size = 1;
  constexpr int num_threads = 256;
  const dim3 block_dim(num_threads);
  const dim3 grid_dim(1);

  size_t shared_mem_size = num_threads * sizeof(double);

  rmsnorm_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(x, w, out, hidden_dim, eps);
  CHECK_HIP(hipGetLastError());
  // RMSNorm_kernel<<<grid_dim, block_dim, 0, stream>>>(x, w, out, hidden_dim, eps);
}

// W (out, in) * x(in, ) -> y(out, )
__global__ void matmul_kernel(const float *W, const float *x, const float *bias, float *out,
                              int out_features, int in_features) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < out_features) {
    double sum = 0.0f;

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

void QKVGemmGPU(const float *w_qkv, const float *b_qkv, const float *t, float *out, int hidden_dim,
                int head_dim, int in_features, int out_features, hipStream_t stream) {
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((out_features + block_dim.x - 1) / block_dim.x);

  matmul_kernel<<<grid_dim, block_dim, 0, stream>>>(w_qkv, t, b_qkv, out, out_features,
                                                    in_features);
}

void RouterGemmGPU(const float *w_router, const float *t, const float *b_router,
                   float *router_score, int hidden_dim, int n_experts, hipStream_t stream) {
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((n_experts + block_dim.x - 1) / block_dim.x);
  matmul_kernel<<<grid_dim, block_dim, 0, stream>>>(w_router, t, b_router, router_score, n_experts,
                                                    hidden_dim);
}

void ClassifierGemmGPU(const float *W_out, const float *x, float *logits, int hidden_dim,
                       int vocab_size, hipStream_t stream) {
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((vocab_size + block_dim.x - 1) / block_dim.x);
  matmul_kernel<<<grid_dim, block_dim, 0, stream>>>(W_out, x, nullptr, logits, vocab_size,
                                                    hidden_dim);
}

//================================================================================================
// 4. Các hàm phụ trợ (AddBias, LinearBiasResidual)
//================================================================================================

__global__ void AddBias_kernel(float *y, const float *b, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    y[i] += b[i];
  }
}

void AddBiasGPU(float *y, const float *b, int len, hipStream_t stream) {
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((len + block_dim.x - 1) / block_dim.x);
  AddBias_kernel<<<grid_dim, block_dim, 0, stream>>>(y, b, len);
}

__global__ void LinearBiasResidual_kernel(const float *W, const float *x, const float *bias,
                                          float *x_resid_inout, int in_features, int out_features) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < out_features) {
    double sum = 0.0;
    const float *W_row = W + (size_t)row * in_features;
    for (int i = 0; i < in_features; i++) {
      sum += (double)W_row[i] * x[i];
    }
    if (bias != nullptr) {
      sum += bias[row];
    }
    x_resid_inout[row] += sum;
  }
}

void LinearBiasResidualGPU(const float *W, const float *x, const float *bias, float *x_resid_inout,
                           int in_features, int out_features, hipStream_t stream) {
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((out_features + block_dim.x - 1) / block_dim.x);
  LinearBiasResidual_kernel<<<grid_dim, block_dim, 0, stream>>>(W, x, bias, x_resid_inout,
                                                                in_features, out_features);
}

//================================================================================================
// 5. RoPE (Rotary Positional Embedding)
//================================================================================================

__global__ void rope_kernel(int seq_len, int head_dim, const float *d_inv_freq, float concentration,
                            float *out_cos, float *out_sin) {
  int pos = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // i là chỉ số của cặp dimension

  if (pos < seq_len && i < head_dim / 2) {
    float inv_freq_val = d_inv_freq[i];
    float val = (float)pos * inv_freq_val;
    size_t idx = (size_t)pos * (head_dim / 2) + i;

    out_cos[idx] = cosf(val) * concentration;
    out_sin[idx] = sinf(val) * concentration;
  }
}

void BuildRopeTableGPU(int seq_len, int head_dim, float rope_theta, float scaling_factor,
                       float init_ctx_len, float ntk_beta, float ntk_alpha, float *d_out_cos,
                       float *d_out_sin, hipStream_t stream) {
  float concentration;
  int d_half = head_dim / 2;

  float *h_inv_freq = (float *)malloc(d_half * sizeof(float));

  // void compute_concentration_and_inv_freq(float base, int head_dim, float scaling_factor,
  //                                         float initial_context_length, float ntk_beta,
  //                                         float ntk_alpha, float *concentration_out,
  //                                         float *inv_freq_out  // length head_dim/2
  //                                         )

  // Pre-defined in run.cpp
  compute_concentration_and_inv_freq(rope_theta, head_dim, scaling_factor, init_ctx_len, ntk_beta,
                                     ntk_alpha, &concentration, h_inv_freq);

  float *d_inv_freq;
  size_t inv_freq_size = d_half * sizeof(float);

  CHECK_HIP(hipMalloc(&d_inv_freq, inv_freq_size));
  CHECK_HIP(hipMemcpyAsync(d_inv_freq, h_inv_freq, inv_freq_size, hipMemcpyHostToDevice, stream));

  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((head_dim / 2 + block_dim.x - 1) / block_dim.x, seq_len);
  rope_kernel<<<grid_dim, block_dim, 0, stream>>>(seq_len, head_dim, d_inv_freq, concentration,
                                                  d_out_cos, d_out_sin);
  CHECK_HIP(hipGetLastError());

  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipFree(d_inv_freq));
  free(h_inv_freq);
}

__global__ void QKVEpilogueSplitRoPECache_kernel(
  const float *__restrict__ qkv_out, float *__restrict__ q_out, float *__restrict__ k_pos,
  float *__restrict__ v_pos, const float *__restrict__ rope_cos_pos,
  const float *__restrict__ rope_sin_pos, int head_dim, int n_q, int n_kv) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int q_dims = n_q * head_dim;
  int k_dims = n_kv * head_dim;
  int v_dims = n_kv * head_dim;
  int half = head_dim / 2;

  if (idx < q_dims) {
    // ---- Q slice ----
    int head_idx = idx / head_dim;
    int dim_idx = idx % head_dim;

    if (dim_idx < half) {
      size_t q_base = head_idx * head_dim;
      float x1 = qkv_out[q_base + dim_idx];
      float x2 = qkv_out[q_base + half + dim_idx];
      float c = rope_cos_pos[dim_idx];
      float s = rope_sin_pos[dim_idx];

      q_out[q_base + dim_idx] = x1 * c - x2 * s;
      q_out[q_base + half + dim_idx] = x2 * c + x1 * s;
    }
  } else if (idx < q_dims + k_dims) {
    // ---- K slice ----
    int k_idx = idx - q_dims;
    int head_idx = k_idx / head_dim;
    int dim_idx = k_idx % head_dim;

    if (dim_idx < half) {
      size_t k_base = head_idx * head_dim;
      size_t Koff = q_dims + k_base;
      float x1 = qkv_out[Koff + dim_idx];
      float x2 = qkv_out[Koff + half + dim_idx];
      float c = rope_cos_pos[dim_idx];
      float s = rope_sin_pos[dim_idx];

      k_pos[k_base + dim_idx] = x1 * c - x2 * s;
      k_pos[k_base + half + dim_idx] = x2 * c + x1 * s;
    }
  } else {
    // ---- V slice: copy ----
    int v_idx = idx - q_dims - k_dims;
    v_pos[v_idx] = qkv_out[q_dims + k_dims + v_idx];
  }
}

void QKVEpilogueSplitRoPECacheGPU(float *qkv_out, float *q_out, float *k_pos, float *v_pos,
                                  const float *rope_cos_pos, const float *rope_sin_pos,
                                  int head_dim, int n_q, int n_kv, hipStream_t stream) {
  int total_dims = (n_q + 2 * n_kv) * head_dim;
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((total_dims + block_dim.x - 1) / block_dim.x);
  QKVEpilogueSplitRoPECache_kernel<<<grid_dim, block_dim, 0, stream>>>(
    qkv_out, q_out, k_pos, v_pos, rope_cos_pos, rope_sin_pos, head_dim, n_q, n_kv);
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
      s_scores[t] = expf(s_scores[t] - max_score);
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

void SingleQueryAttentionGPU(const float *q, const float *K_cache, const float *V_cache,
                             const float *mask_row, const float *attn_sink_per_head, float *tb,
                             int head_dim, int n_q, int kv_mul, int kv_dim, int seq_len, int pos,
                             hipStream_t stream) {
  dim3 grid_dim(n_q);
  dim3 block_dim(256);

  size_t shared_mem_size =
    ((size_t)seq_len + 1) * sizeof(float) + (size_t)head_dim * sizeof(double);

  attention_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
    q, K_cache, V_cache, mask_row, attn_sink_per_head, tb, head_dim, n_q, kv_mul, kv_dim, seq_len);
}

//================================================================================================
// 7. MoE (Mixture of Experts)
//================================================================================================

__global__ void TopKSoftmax_kernel(const float *r, int n_experts, int k, float *topk_vals,
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
    denom += expf(temp_vals[i] - max_val);
  }

  // Output results
  for (int i = 0; i < k; ++i) {
    topk_vals[i] = expf(temp_vals[i] - max_val) / denom;
    topk_idx[i] = temp_idx[i];
  }
}

void TopKSoftmaxGPU(const float *r, int n_experts, int k, float *topk_vals, int *topk_idx,
                    hipStream_t stream) {
  size_t shared_mem_size = (size_t)n_experts * (sizeof(float) + sizeof(int));
  TopKSoftmax_kernel<<<1, 1, shared_mem_size, stream>>>(r, n_experts, k, topk_vals, topk_idx);
  CHECK_HIP(hipGetLastError());
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

__global__ void WeightedAccumulate_kernel(const float *expert_out, float weight, float *accumulator,
                                          int hidden_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden_dim) {
    accumulator[i] += weight * expert_out[i];
  }
}

__global__ void matmul_kernel_bf16_weights(const bf16 *W, const float *x, const bf16 *bias,
                                           float *out, int out_features, int in_features) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < out_features) {
    double sum = 0.0;  // Vẫn dùng double để tích lũy cho chính xác
    const bf16 *W_row = W + (size_t)row * in_features;

    for (int i = 0; i < in_features; i++) {
      // Chuyển đổi trọng số bfloat16 sang float để tính toán
      float w_val = static_cast<float>(W_row[i]);
      sum += (double)w_val * x[i];
    }

    if (bias != nullptr) {
      // Chuyển đổi bias bfloat16 sang float để cộng
      sum += static_cast<float>(bias[row]);
    }
    out[row] = (float)sum;
  }
}

void MoEApplyTopKGPU(const float *t, const bf16 *W1, const bf16 *b1, const bf16 *W2, const bf16 *b2,
                     const int *topk_idx, const float *topk_vals,
                     float *work_gate_up,  // scratch: [2*inter]
                     float *e_agg_inout,   // out: [hidden]
                     int hidden_dim, int inter_dim, int k, float clamp_limit, hipStream_t stream) {
  // Cảnh báo: Cài đặt naive này copy dữ liệu về host để điều khiển vòng lặp.
  // Điều này rất chậm và chỉ dùng cho mục đích minh họa/debug.
  // Một cài đặt thực tế sẽ dùng các kernel phức tạp hơn để giữ mọi thứ trên GPU.
  int h_topk_idx[k];
  float h_topk_vals[k];
  CHECK_HIP(hipMemcpy(h_topk_idx, topk_idx, k * sizeof(int), hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(h_topk_vals, topk_vals, k * sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipDeviceSynchronize());  // Đợi copy xong

  // Xóa bộ đệm tích lũy kết quả
  CHECK_HIP(hipMemsetAsync(e_agg_inout, 0, hidden_dim * sizeof(float), stream));

  // Cấp phát bộ nhớ tạm trên GPU cho kết quả của từng expert
  float *work_swiglu;    // scratch: [inter]
  float *expert_output;  // scratch: [hidden]
  CHECK_HIP(hipMalloc(&work_swiglu, inter_dim * sizeof(float)));
  CHECK_HIP(hipMalloc(&expert_output, hidden_dim * sizeof(float)));

  for (int i = 0; i < k; i++) {
    int expert_idx = h_topk_idx[i];
    float expert_weight = h_topk_vals[i];

    // Con trỏ tới trọng số của expert đang xét
    size_t w1_offset = (size_t)expert_idx * (2 * inter_dim) * hidden_dim;
    size_t b1_offset = (size_t)expert_idx * (2 * inter_dim);
    size_t w2_offset = (size_t)expert_idx * hidden_dim * inter_dim;
    size_t b2_offset = (size_t)expert_idx * hidden_dim;

    const bf16 *W1_expert = W1 + w1_offset;
    const bf16 *b1_expert = b1 + b1_offset;
    const bf16 *W2_expert = W2 + w2_offset;
    const bf16 *b2_expert = b2 + b2_offset;

    // --- Bước 1: GEMM đầu tiên ---
    // z = W1 * t + b1 -> ghi vào work_gate_up
    matmul_kernel_bf16_weights<<<(2 * inter_dim + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE,
                                 DEFAULT_BLOCK_SIZE, 0, stream>>>(
      W1_expert, t, b1_expert, work_gate_up, 2 * inter_dim, hidden_dim);

    // --- Bước 2: Kích hoạt SwiGLU ---
    // swiglu = silu(gate) * up -> ghi vào work_swiglu
    SwiGLU_kernel<<<(inter_dim + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE,
                    0, stream>>>(work_gate_up, work_swiglu, inter_dim, clamp_limit);

    // --- Bước 3: GEMM thứ hai ---
    // y = W2 * swiglu + b2 -> ghi vào expert_output
    matmul_kernel_bf16_weights<<<(hidden_dim + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE,
                                 DEFAULT_BLOCK_SIZE, 0, stream>>>(
      W2_expert, work_swiglu, b2_expert, expert_output, hidden_dim, inter_dim);

    // --- Bước 4: Cộng dồn kết quả theo trọng số ---
    // e_agg_inout += weight * y
    WeightedAccumulate_kernel<<<(hidden_dim + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE,
                                DEFAULT_BLOCK_SIZE, 0, stream>>>(expert_output, expert_weight,
                                                                 e_agg_inout, hidden_dim);
  }

  // Giải phóng bộ nhớ tạm
  CHECK_HIP(hipFree(work_swiglu));
  CHECK_HIP(hipFree(expert_output));
}
