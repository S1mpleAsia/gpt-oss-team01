#include "layer.hpp"
#include <cmath>
#include <cfloat>

#define DEFAULT_BLOCK_SIZE 256

// __global__ void BuildRopeTable_kernel(int seq_len, int head_dim, float rope_theta, float *out_cos,
//                                       float *out_sin) {
//   // Chỉ số của cặp dimension (từ 0 đến head_dim/2 - 1)
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   // Vị trí của token trong chuỗi (từ 0 đến seq_len - 1)
//   int pos = blockIdx.y;

//   int head_dim_half = head_dim / 2;

//   // Kiểm tra xem thread có nằm trong phạm vi công việc hợp lệ không
//   if (pos < seq_len && i < head_dim_half) {
//     // Tính tần số nghịch đảo (inverse frequency)
//     // Đây là phần tính toán chính của RoPE
//     float freq = 1.0f / powf(rope_theta, ((float)(i * 2)) / (float)head_dim);

//     // Tính góc quay (angle)
//     float val = (float)pos * freq;

//     // Tính chỉ số 1D tương ứng trong buffer đầu ra
//     size_t idx = (size_t)pos * head_dim_half + i;

//     // Ghi kết quả sin và cos vào bộ nhớ global
//     out_cos[idx] = cosf(val);
//     out_sin[idx] = sinf(val);
//   }
// }

// /**
//  * @brief Hàm Host để khởi chạy kernel BuildRopeTable_kernel.
//  */
// void BuildRopeTableGPU(int seq_len, int head_dim, float rope_theta, float *out_cos, float *out_sin,
//                        hipStream_t stream) {
//   // Cấu hình kích thước block và grid
//   // Phiên bản này sử dụng grid 2D để ánh xạ trực tiếp vào bài toán

//   int head_dim_half = head_dim / 2;

//   // Mỗi block có 256 thread, xử lý theo chiều `head_dim`
//   dim3 block_dim(256);

//   // Grid 2 chiều:
//   // - Chiều x: đủ số block để bao phủ `head_dim / 2`
//   // - Chiều y: mỗi hàng tương ứng với một vị trí trong `seq_len`
//   dim3 grid_dim((head_dim_half + block_dim.x - 1) / block_dim.x, seq_len);

//   // Khởi chạy kernel
//   BuildRopeTable_kernel<<<grid_dim, block_dim, 0, stream>>>(seq_len, head_dim, rope_theta, out_cos,
//                                                             out_sin);
// }

void EmbeddingLookupGPU(const float *embedding,  // [vocab, hidden]
                        int token_id,
                        float *x,  // [hidden]
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

__global__ void RMSNorm_kernel(const float *x, const float *w, float *out, int n, float eps) {
  // Kernel này giả định n đủ nhỏ để tính toán trong một block duy nhất
  __shared__ float s_variance;

  // Pass 1: Tính tổng bình phương (sum of squares)
  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    sum_sq += x[i] * x[i];
  }

  // Dùng shared memory để reduce tổng trong block
  __shared__ float s_partials[DEFAULT_BLOCK_SIZE];
  s_partials[threadIdx.x] = sum_sq;
  __syncthreads();

  // Thread 0 thực hiện phần reduce cuối cùng
  if (threadIdx.x == 0) {
    float total_sum_sq = 0.0f;
    // Giả định blockDim.x <= DEFAULT_BLOCK_SIZE
    for (int i = 0; i < blockDim.x; i++) {
      total_sum_sq += s_partials[i];
    }
    float variance = total_sum_sq / n;
    s_variance = rsqrtf(variance + eps);
  }
  __syncthreads();  // Đảm bảo mọi thread đều thấy s_variance

  // Pass 2: Áp dụng chuẩn hóa
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = x[i] * s_variance * w[i];
  }
}

void RMSNormGPU(const float *x, const float *w, float *out, int hidden_dim, float eps,
                hipStream_t stream) {
  // Phiên bản naive này chỉ dùng 1 block để thực hiện reduction
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim(1);
  RMSNorm_kernel<<<grid_dim, block_dim, 0, stream>>>(x, w, out, hidden_dim, eps);
}

//================================================================================================
// 3. GEMM (Matrix-Vector) và các hàm liên quan
//================================================================================================

// Kernel chung cho phép nhân ma trận (W[rows, cols]) với vector (x[cols])
__global__ void GemmMatrixVector_kernel(const float *W, const float *x, const float *b_or_null,
                                        float *y, int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows) {
    float sum = 0.0f;
    const float *W_row = W + (size_t)row * cols;
    for (int i = 0; i < cols; i++) {
      sum += W_row[i] * x[i];
    }
    if (b_or_null != nullptr) {
      sum += b_or_null[row];
    }
    y[row] = sum;
  }
}

void QKVGemmGPU(const float *W_qkv, const float *t, float *qkv_out, int hidden_dim, int head_dim,
                int n_q, int n_kv, hipStream_t stream) {
  int out_features = (n_q + 2 * n_kv) * head_dim;
  int in_features = hidden_dim;

  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((out_features + block_dim.x - 1) / block_dim.x);

  // Bias được xử lý trong epilogue, nên ở đây b_or_null = nullptr
  GemmMatrixVector_kernel<<<grid_dim, block_dim, 0, stream>>>(W_qkv, t, nullptr, qkv_out,
                                                              out_features, in_features);
}

void RouterGemmGPU(const float *W_router, const float *t, const float *b_router_or_null, float *r,
                   int hidden_dim, int n_experts, hipStream_t stream) {
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((n_experts + block_dim.x - 1) / block_dim.x);
  GemmMatrixVector_kernel<<<grid_dim, block_dim, 0, stream>>>(W_router, t, b_router_or_null, r,
                                                              n_experts, hidden_dim);
}

void ClassifierGemmGPU(const float *W_out, const float *x, float *logits, int hidden_dim,
                       int vocab_size, hipStream_t stream) {
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((vocab_size + block_dim.x - 1) / block_dim.x);
  GemmMatrixVector_kernel<<<grid_dim, block_dim, 0, stream>>>(W_out, x, nullptr, logits, vocab_size,
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

__global__ void LinearBiasResidual_kernel(const float *W, const float *x, const float *b_or_null,
                                          float *x_resid_inout, int in_features, int out_features) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < out_features) {
    float sum = 0.0f;
    const float *W_row = W + (size_t)row * in_features;
    for (int i = 0; i < in_features; i++) {
      sum += W_row[i] * x[i];
    }
    if (b_or_null != nullptr) {
      sum += b_or_null[row];
    }
    x_resid_inout[row] += sum;
  }
}

void LinearBiasResidualGPU(const float *W, const float *x, const float *b_or_null,
                           float *x_resid_inout, int in_features, int out_features,
                           hipStream_t stream) {
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((out_features + block_dim.x - 1) / block_dim.x);
  LinearBiasResidual_kernel<<<grid_dim, block_dim, 0, stream>>>(W, x, b_or_null, x_resid_inout,
                                                                in_features, out_features);
}

//================================================================================================
// 5. RoPE (Rotary Positional Embedding)
//================================================================================================

__global__ void BuildRopeTable_kernel(int seq_len, int head_dim, float rope_theta, float *out_cos,
                                      float *out_sin) {
  int pos = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // i là chỉ số của cặp dimension

  if (pos < seq_len && i < head_dim / 2) {
    int dim = i * 2;
    float freq = 1.0f / powf(rope_theta, (float)dim / head_dim);
    float val = (float)pos * freq;
    size_t idx = (size_t)pos * (head_dim / 2) + i;
    out_cos[idx] = cosf(val);
    out_sin[idx] = sinf(val);
  }
}

void BuildRopeTableGPU(int seq_len, int head_dim, float rope_theta, float scaling_factor,
                       float init_ctx_len, float ntk_beta, float ntk_alpha, float *out_cos,
                       float *out_sin, hipStream_t stream) {
  // Bỏ qua các scaling factor trong phiên bản naive
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((head_dim / 2 + block_dim.x - 1) / block_dim.x, seq_len);
  BuildRopeTable_kernel<<<grid_dim, block_dim, 0, stream>>>(seq_len, head_dim, rope_theta, out_cos,
                                                            out_sin);
}

__global__ void QKVEpilogueSplitRoPECache_kernel(float *qkv_out, const float *b_qkv_or_null,
                                                 float *q_out, float *k_pos, float *v_pos,
                                                 const float *rope_cos_pos,
                                                 const float *rope_sin_pos, int head_dim, int n_q,
                                                 int n_kv) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Tổng index trên tất cả các head

  // Tổng số chiều của Q, K, V
  int q_dims = n_q * head_dim;
  int k_dims = n_kv * head_dim;
  int v_dims = n_kv * head_dim;

  if (idx < q_dims + k_dims + v_dims) {
    // Cộng bias trước (nếu có)
    if (b_qkv_or_null != nullptr) {
      qkv_out[idx] += b_qkv_or_null[idx];
    }

    float val = qkv_out[idx];

    if (idx < q_dims) {  // Đây là phần tử của Q
      int head_idx = idx / head_dim;
      int dim_idx = idx % head_dim;

      // Áp dụng RoPE cho Q
      if (dim_idx % 2 == 0) {
        int pair_dim_idx = dim_idx + 1;
        int pair_idx = head_idx * head_dim + pair_dim_idx;
        float val_pair = qkv_out[pair_idx];
        if (b_qkv_or_null != nullptr)
          val_pair += b_qkv_or_null[pair_idx];

        int rope_idx = dim_idx / 2;
        float cos_val = rope_cos_pos[rope_idx];
        float sin_val = rope_sin_pos[rope_idx];

        q_out[idx] = val * cos_val - val_pair * sin_val;
        q_out[pair_idx] = val * sin_val + val_pair * cos_val;
      }

    } else if (idx < q_dims + k_dims) {  // Đây là phần tử của K
      int k_idx = idx - q_dims;
      int head_idx = k_idx / head_dim;
      int dim_idx = k_idx % head_dim;

      // Áp dụng RoPE cho K và ghi vào cache
      if (dim_idx % 2 == 0) {
        int pair_dim_idx = dim_idx + 1;
        int pair_idx = q_dims + head_idx * head_dim + pair_dim_idx;
        float val_pair = qkv_out[pair_idx];
        if (b_qkv_or_null != nullptr)
          val_pair += b_qkv_or_null[pair_idx];

        int rope_idx = dim_idx / 2;
        float cos_val = rope_cos_pos[rope_idx];
        float sin_val = rope_sin_pos[rope_idx];

        k_pos[k_idx] = val * cos_val - val_pair * sin_val;
        k_pos[k_idx + 1] = val * sin_val + val_pair * cos_val;
      }
    } else {  // Đây là phần tử của V
      int v_idx = idx - q_dims - k_dims;
      v_pos[v_idx] = val;  // Chỉ cần copy V vào cache
    }
  }
}

void QKVEpilogueSplitRoPECacheGPU(float *qkv_out, const float *b_qkv_or_null, float *q_out,
                                  float *k_pos, float *v_pos, const float *rope_cos_pos,
                                  const float *rope_sin_pos, int head_dim, int n_q, int n_kv,
                                  hipStream_t stream) {
  int total_dims = (n_q + 2 * n_kv) * head_dim;
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim((total_dims + block_dim.x - 1) / block_dim.x);
  QKVEpilogueSplitRoPECache_kernel<<<grid_dim, block_dim, 0, stream>>>(
    qkv_out, b_qkv_or_null, q_out, k_pos, v_pos, rope_cos_pos, rope_sin_pos, head_dim, n_q, n_kv);
}

//================================================================================================
// 6. Attention
//================================================================================================

__global__ void SingleQueryAttention_kernel(const float *q, const float *K_cache,
                                            const float *V_cache, float *tb, int head_dim, int n_q,
                                            int kv_dim, int seq_len) {
  // Mỗi block xử lý một query head
  int head_idx = blockIdx.x;
  if (head_idx >= n_q)
    return;

  const float *q_head = q + head_idx * head_dim;
  float *tb_head = tb + head_idx * head_dim;

  // shared memory để lưu scores và kết quả V tích lũy
  extern __shared__ float s_data[];
  float *s_scores = s_data;            // size: seq_len
  float *s_output = &s_data[seq_len];  // size: head_dim

  // Step 1: Tính scores = q_head * K_cache[t] / sqrt(head_dim)
  float scale = rsqrtf((float)head_dim);
  for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
    const float *k_vec = K_cache + t * kv_dim + (head_idx % (kv_dim / head_dim)) * head_dim;
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      score += q_head[i] * k_vec[i];
    }
    s_scores[t] = score * scale;
  }
  __syncthreads();

  // Step 2: Tìm max score để tính softmax ổn định (online softmax)
  // Naive: thread 0 làm hết
  if (threadIdx.x == 0) {
    float max_score = -FLT_MAX;
    for (int t = 0; t < seq_len; t++) {
      if (s_scores[t] > max_score) {
        max_score = s_scores[t];
      }
    }

    // Tính mẫu số (denominator) của softmax
    float denom = 0.0f;
    for (int t = 0; t < seq_len; t++) {
      s_scores[t] = expf(s_scores[t] - max_score);  // Lưu lại tử số
      denom += s_scores[t];
    }

    // Khởi tạo output
    for (int i = 0; i < head_dim; i++)
      s_output[i] = 0.0f;

    // Step 3: Tích lũy V
    for (int t = 0; t < seq_len; t++) {
      float prob = s_scores[t] / denom;
      const float *v_vec = V_cache + t * kv_dim + (head_idx % (kv_dim / head_dim)) * head_dim;
      for (int i = 0; i < head_dim; i++) {
        s_output[i] += prob * v_vec[i];
      }
    }
  }
  __syncthreads();

  // Step 4: Ghi kết quả ra global memory
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    tb_head[i] = s_output[i];
  }
}

void SingleQueryAttentionGPU(const float *q, const float *K_cache, const float *V_cache,
                             const float *mask_row, const float *attn_sink_per_head, float *tb,
                             int head_dim, int n_q, int kv_mul, int kv_dim, int seq_len, int pos,
                             hipStream_t stream) {
  // Phiên bản naive bỏ qua mask và sink
  dim3 grid_dim(n_q);
  dim3 block_dim(min((int)DEFAULT_BLOCK_SIZE, seq_len));
  size_t shared_mem_size = (seq_len + head_dim) * sizeof(float);
  SingleQueryAttention_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
    q, K_cache, V_cache, tb, head_dim, n_q, kv_dim, seq_len);
}

//================================================================================================
// 7. MoE (Mixture of Experts)
//================================================================================================

__global__ void TopKSoftmax_kernel(const float *r, int n_experts, int k, float *topk_vals,
                                   int *topk_idx) {
  // Naive: dùng một block duy nhất để tìm Top-K
  if (threadIdx.x != 0)
    return;

  // Tạm thời dùng mảng tĩnh, giả sử n_experts không quá lớn
  float temp_vals[16];
  int temp_idx[16];

  for (int i = 0; i < n_experts; ++i) {
    temp_vals[i] = r[i];
    temp_idx[i] = i;
  }

  // Simple selection sort
  for (int i = 0; i < k; ++i) {
    int max_idx = i;
    for (int j = i + 1; j < n_experts; ++j) {
      if (temp_vals[j] > temp_vals[max_idx]) {
        max_idx = j;
      }
    }
    // Swap
    float t_v = temp_vals[i];
    temp_vals[i] = temp_vals[max_idx];
    temp_vals[max_idx] = t_v;
    int t_i = temp_idx[i];
    temp_idx[i] = temp_idx[max_idx];
    temp_idx[max_idx] = t_i;
  }

  // Tính softmax cho K giá trị top
  float max_val = temp_vals[0];
  float denom = 0.0f;
  for (int i = 0; i < k; ++i) {
    denom += expf(temp_vals[i] - max_val);
  }

  for (int i = 0; i < k; ++i) {
    topk_vals[i] = expf(temp_vals[i] - max_val) / denom;
    topk_idx[i] = temp_idx[i];
  }
}

void TopKSoftmaxGPU(const float *r, int n_experts, int k, float *topk_vals, int *topk_idx,
                    hipStream_t stream) {
  TopKSoftmax_kernel<<<1, 1, 0, stream>>>(r, n_experts, k, topk_vals, topk_idx);
}

// Giả định các kernel và hàm từ câu trả lời trước đã có ở đây...

//================================================================================================
// 8. Các kernel phụ cho MoE (Mixture of Experts)
//================================================================================================

__global__ void SwiGLU_kernel(const float *gate_up_in, float *swiglu_out, int inter_dim,
                              float clamp_limit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < inter_dim) {
    const float *gate = gate_up_in;
    const float *up = gate_up_in + inter_dim;

    float gate_val = gate[i];

    // Giới hạn giá trị (clamping)
    if (clamp_limit > 0.0f) {
      gate_val = fmaxf(-clamp_limit, fminf(gate_val, clamp_limit));
    }

    // SILU (SiLU(x) = x * sigmoid(x))
    float silu_val = gate_val / (1.0f + expf(-gate_val));

    swiglu_out[i] = silu_val * up[i];
  }
}

__global__ void WeightedAccumulate_kernel(const float *expert_out, float weight, float *accumulator,
                                          int hidden_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden_dim) {
    accumulator[i] += weight * expert_out[i];
  }
}

// Cài đặt hoàn chỉnh cho MoEApplyTopKGPU
void MoEApplyTopKGPU(const float *t, const float *W1, const float *b1, const float *W2,
                     const float *b2, const int *topk_idx, const float *topk_vals,
                     float *work_gate_up,  // scratch: [2*inter]
                     float *e_agg_inout,   // out: [hidden]
                     int hidden_dim, int inter_dim, int k, float clamp_limit, hipStream_t stream) {
  // Cảnh báo: Cài đặt naive này copy dữ liệu về host để điều khiển vòng lặp.
  // Điều này rất chậm và chỉ dùng cho mục đích minh họa/debug.
  // Một cài đặt thực tế sẽ dùng các kernel phức tạp hơn để giữ mọi thứ trên GPU.
  int h_topk_idx[k];
  float h_topk_vals[k];
  hipMemcpy(h_topk_idx, topk_idx, k * sizeof(int), hipMemcpyDeviceToHost);
  hipMemcpy(h_topk_vals, topk_vals, k * sizeof(float), hipMemcpyDeviceToHost);
  hipDeviceSynchronize();  // Đợi copy xong

  // Xóa bộ đệm tích lũy kết quả
  hipMemsetAsync(e_agg_inout, 0, hidden_dim * sizeof(float), stream);

  // Cấp phát bộ nhớ tạm trên GPU cho kết quả của từng expert
  float *work_swiglu;    // scratch: [inter]
  float *expert_output;  // scratch: [hidden]
  hipMalloc(&work_swiglu, inter_dim * sizeof(float));
  hipMalloc(&expert_output, hidden_dim * sizeof(float));

  for (int i = 0; i < k; i++) {
    int expert_idx = h_topk_idx[i];
    float expert_weight = h_topk_vals[i];

    // Con trỏ tới trọng số của expert đang xét
    size_t w1_offset = (size_t)expert_idx * (2 * inter_dim) * hidden_dim;
    size_t b1_offset = (size_t)expert_idx * (2 * inter_dim);
    size_t w2_offset = (size_t)expert_idx * hidden_dim * inter_dim;
    size_t b2_offset = (size_t)expert_idx * hidden_dim;

    const float *W1_expert = W1 + w1_offset;
    const float *b1_expert = b1 + b1_offset;
    const float *W2_expert = W2 + w2_offset;
    const float *b2_expert = b2 + b2_offset;

    // --- Bước 1: GEMM đầu tiên ---
    // z = W1 * t + b1 -> ghi vào work_gate_up
    GemmMatrixVector_kernel<<<(2 * inter_dim + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE,
                              DEFAULT_BLOCK_SIZE, 0, stream>>>(
      W1_expert, t, b1_expert, work_gate_up, 2 * inter_dim, hidden_dim);

    // --- Bước 2: Kích hoạt SwiGLU ---
    // swiglu = silu(gate) * up -> ghi vào work_swiglu
    SwiGLU_kernel<<<(inter_dim + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE,
                    0, stream>>>(work_gate_up, work_swiglu, inter_dim, clamp_limit);

    // --- Bước 3: GEMM thứ hai ---
    // y = W2 * swiglu + b2 -> ghi vào expert_output
    GemmMatrixVector_kernel<<<(hidden_dim + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE,
                              DEFAULT_BLOCK_SIZE, 0, stream>>>(
      W2_expert, work_swiglu, b2_expert, expert_output, hidden_dim, inter_dim);

    // --- Bước 4: Cộng dồn kết quả theo trọng số ---
    // e_agg_inout += weight * y
    WeightedAccumulate_kernel<<<(hidden_dim + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE,
                                DEFAULT_BLOCK_SIZE, 0, stream>>>(expert_output, expert_weight,
                                                                 e_agg_inout, hidden_dim);
  }

  // Giải phóng bộ nhớ tạm
  hipFree(work_swiglu);
  hipFree(expert_output);
}

//================================================================================================
// 9. Final RMSNorm (In-place)
//================================================================================================

// Hàm này có thể tái sử dụng kernel RMSNorm đã viết ở phần trước,
// vì kernel đó nhận con trỏ input và output riêng biệt.
// Ta chỉ cần truyền cùng một con trỏ cho cả hai là được.
void RMSNormInplaceGPU(float *x, const float *scale, int hidden_dim, float eps,
                       hipStream_t stream) {
  dim3 block_dim(DEFAULT_BLOCK_SIZE);
  dim3 grid_dim(1);
  // Truyền `x` cho cả input `x` và output `out` của kernel
  RMSNorm_kernel<<<grid_dim, block_dim, 0, stream>>>(x, scale, x, hidden_dim, eps);
}