#include "../include/layer_hip.hpp"
#include <cmath>
#include <cfloat>

#define DEFAULT_BLOCK_SIZE 256

void EmbeddingLookupGPU(Tensor *embedding,  // (vocab_size, hidden_dim)
                        int token_id, Tensor *x,  // (hidden_dim, )
                        bool x_from_device, hipStream_t stream = 0) {
    // Assure that embedding->dtype == x->dtype
    const size_t hidden_dim = x->shape[0];
    if (x->dtype == DType::BF16) {
        bf16 *src = (bf16 *)embedding->d_buf + (size_t)token_id * hidden_dim;
        CHECK_HIP(hipMemcpyAsync((bf16 *)x->d_buf, src, hidden_dim * sizeof(bf16), hipMemcpyDeviceToDevice, stream));
    } else {
        float *src = (float *)embedding->d_buf + (size_t)token_id * hidden_dim;
        CHECK_HIP(hipMemcpyAsync((float *)x->d_buf, src, hidden_dim * sizeof(float), hipMemcpyDeviceToDevice, stream));
    }

    if (x_from_device) {
        x->from_device(stream);
        CHECK_HIP(hipStreamSynchronize(stream));
    }
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

__global__ void rmsnorm_kernel_double_precision(const float *x, const float *w, float *out, int hidden_dim, float eps) {
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

__global__ void rmsnorm_kernel(const float *x, const float *w, float *out, int hidden_dim, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float *x_row = x + row * hidden_dim;
    const float *w_row = w + row * hidden_dim;
    float *o_row = out + row * hidden_dim;

    float acc = 0.0f;

    for (int j = tid; j < hidden_dim; j += blockDim.x) {
        float v = x_row[j];
        acc += v * v;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }

    // 32 * 64 = 2048 threads
    __shared__ float warp_sum[32];
    int lane = tid & (warpSize - 1);
    int warp_id = (tid + warpSize - 1) / warpSize;

    if (lane == 0)
        warp_sum[warp_id] = acc;

    __syncthreads();

    __shared__ float block_sum;
    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float val = (lane < num_warps) ? warp_sum[lane] : 0.0f;

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down(val, offset);
        }

        if (lane == 0) {
            block_sum = val;
        }
    }
    __syncthreads();

    __shared__ float inv_rms;
    if (tid == 0) {
        float mean = block_sum / (float)hidden_dim;
        inv_rms = rsqrtf(mean + eps);
    }

    __syncthreads();

    for (int j = tid; j < hidden_dim; j += blockDim.x) {
        o_row[j] = w_row[j] * (x_row[j] * inv_rms);
    }
}

void RMSNormGPU(Tensor *x, Tensor *w, Tensor *out, long long layer_offset,
                bool x_to_device, bool out_from_device, float eps,
                hipStream_t stream) {
    if (x_to_device) x->to_device(stream);

    constexpr int batch_size = 1;  // Change this when using batching
    constexpr int num_threads = 256;
    const dim3 block_dim(num_threads), grid_dim(batch_size);

    size_t shared_mem_size = num_threads * sizeof(double);

    const int hidden_dim = x->shape[0];

    const float *x_ptr = (float *)x->d_buf;
    const float *w_ptr = (float *)w->d_buf + 1ll * layer_offset * hidden_dim;
    float *out_ptr = (float *)out->d_buf;

    rmsnorm_kernel_double_precision<<<grid_dim, block_dim, shared_mem_size, stream>>>(x_ptr, w_ptr, out_ptr, hidden_dim, eps);
    CHECK_HIP(hipGetLastError());
    // RMSNorm_kernel<<<grid_dim, block_dim, 0, stream>>>(x, w, out, hidden_dim, eps);

    if (out_from_device) {
        out->from_device(stream);
        CHECK_HIP(hipStreamSynchronize(stream));
    }
}

// W (out, in) * x(in, ) -> y(out, )
__global__ void matmul_kernel(const float *W, const float *x, const float *bias, float *out, int out_features, int in_features) {
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

void QKVGemmGPU_Old(const float *w_qkv, const float *b_qkv, const float *t, float *out, int hidden_dim,
                int head_dim, int in_features, int out_features, hipStream_t stream) {
    dim3 block_dim(DEFAULT_BLOCK_SIZE);
    dim3 grid_dim((out_features + block_dim.x - 1) / block_dim.x);

    matmul_kernel<<<grid_dim, block_dim, 0, stream>>>(w_qkv, t, b_qkv, out, out_features, in_features);
}

void QKVGemmGPU(Tensor *x, const Tensor *W_qkv, const Tensor *b_qkv, 
                Tensor *qkv, long long layer_offset, bool x_to_device,
                bool qkv_from_device, hipStream_t stream) {
    if (x_to_device) x->to_device(stream);

    const int in_features = x->num_elem();
    const int out_features = qkv->num_elem();

    const float *x_ptr = (float *)x->d_buf;
    const float *w_qkv_ptr = (float *)W_qkv->d_buf + 1ll * layer_offset * out_features * in_features;
    const float *b_qkv_ptr = (float *)b_qkv->d_buf + 1ll * layer_offset * out_features;
    float *qkv_ptr = (float *)qkv->d_buf;
    
    dim3 block_dim(DEFAULT_BLOCK_SIZE);
    dim3 grid_dim((out_features + block_dim.x - 1) / block_dim.x);

    matmul_kernel<<<grid_dim, block_dim, 0, stream>>>(
        w_qkv_ptr, x_ptr, b_qkv_ptr, qkv_ptr, out_features, in_features
    );

    if (qkv_from_device) {
        qkv->from_device(stream);
        CHECK_HIP(hipStreamSynchronize(stream));
    }
}

void SplitQKVGPU(Tensor *qkv, int head_dim, int n_q, int n_kv,
                Tensor *q, Tensor *k, Tensor *v, bool qkv_to_device,
                bool q_from_device, bool k_from_device,
                bool v_from_device, hipStream_t stream) {
    // Assume they are all copy on device for now
    if (qkv_to_device) qkv->to_device(stream);

    size_t offset = 0;
    MemCpy_Tensor(q, qkv, 0, offset, n_q * head_dim, false, true, stream);
    offset += n_q * head_dim;
    MemCpy_Tensor(k, qkv, 0, offset, n_kv * head_dim, false, true, stream);
    offset += n_kv * head_dim;
    MemCpy_Tensor(v, qkv, 0, offset, n_kv * head_dim, false, true, stream);

    if (q_from_device) q->from_device(stream);
    if (k_from_device) k->from_device(stream);
    if (v_from_device) v->from_device(stream);
    if (q_from_device || k_from_device || v_from_device) {
        CHECK_HIP(hipStreamSynchronize(stream));
    }
}

void RouterGemmGPU(const float *w_router, const float *t, const float *b_router,
                   float *router_score, int hidden_dim, int n_experts, hipStream_t stream) {
    dim3 block_dim(DEFAULT_BLOCK_SIZE);
    dim3 grid_dim((n_experts + block_dim.x - 1) / block_dim.x);
    matmul_kernel<<<grid_dim, block_dim, 0, stream>>>(w_router, t, b_router, router_score, n_experts, hidden_dim);
}

void ClassifierGemmGPU(const float *W_out, const float *x, float *logits, int hidden_dim,
                       int vocab_size, hipStream_t stream) {
    dim3 block_dim(DEFAULT_BLOCK_SIZE);
    dim3 grid_dim((vocab_size + block_dim.x - 1) / block_dim.x);
    matmul_kernel<<<grid_dim, block_dim, 0, stream>>>(W_out, x, nullptr, logits, vocab_size, hidden_dim);
}

//================================================================================================
// 4. Các hàm phụ trợ (AddBias, LinearBiasResidual)
//================================================================================================

__global__ void AddVector_kernel(float *y, const float *b, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        y[i] += b[i];
    }
}

void AddVectorGPU(Tensor *y, Tensor *b, bool y_to_device,
                bool b_to_device, bool y_from_device, hipStream_t stream) {
    if (y_to_device) y->to_device(stream);
    if (b_to_device) b->to_device(stream);

    float *y_ptr = (float *)y->d_buf;
    const float *b_ptr = (float *)b->d_buf;

    const int len = y->num_elem();

    dim3 block_dim(DEFAULT_BLOCK_SIZE);
    dim3 grid_dim((len + block_dim.x - 1) / block_dim.x);
    AddVector_kernel<<<grid_dim, block_dim, 0, stream>>>(y_ptr, b_ptr, len);

    if (y_from_device) {
        y->from_device(stream);
        CHECK_HIP(hipStreamSynchronize(stream));
    }
}

__global__ void LinearBiasResidual_kernel(const float *W, const float *x, const float *bias,
                                          float *x_resid_inout, int in_features, int out_features) {
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

void LinearBiasResidualGPU(const float *W, const float *x, const float *bias, float *x_resid_inout, int in_features, int out_features, hipStream_t stream) {
    dim3 block_dim(DEFAULT_BLOCK_SIZE);
    dim3 grid_dim((out_features + block_dim.x - 1) / block_dim.x);
    LinearBiasResidual_kernel<<<grid_dim, block_dim, 0, stream>>>(W, x, bias, x_resid_inout, in_features, out_features);
}

//================================================================================================
// 5. RoPE (Rotary Positional Embedding)
//================================================================================================

__global__ void QKVEpilogueSplitRoPECache_kernel(
    float *qkv_out, float *q_out, float *k_pos, float *v_pos,
    const float *rope_cos_pos, const float *rope_sin_pos,
    int head_dim, int n_q, int n_kv) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;

    // Total dimensions for Q, K, V
    int q_dims = n_q * head_dim;
    int k_dims = n_kv * head_dim;
    int v_dims = n_kv * head_dim;

    if (idx >= q_dims + k_dims + v_dims) return;

    if (idx < q_dims) { // Processing a Q element
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

    } else if (idx < q_dims + k_dims) { // Processing a K element
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

    } else { // Processing a V element (simple copy)
        int v_local_idx = idx - q_dims - k_dims;
        v_pos[v_local_idx] = qkv_out[idx];
    }
}

void QKVEpilogueSplitRoPECacheGPU(
    Tensor *qkv_out, Tensor *q_out, Tensor *k_pos, Tensor *v_pos,
    const Tensor *rope_cos_pos, const Tensor *rope_sin_pos,
    int head_dim, int n_q, int n_kv, int pos, bool qkv_out_to_device,
    bool q_out_from_device, bool k_pos_from_device,
    bool v_pos_from_device, hipStream_t stream
) {
    if (qkv_out_to_device) qkv_out->to_device(stream);

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
    QKVEpilogueSplitRoPECache_kernel<<<grid_dim, block_dim, 0, stream>>>(
        qkv_out_ptr, q_out_ptr, k_pos_ptr, v_pos_ptr, rope_cos_pos_ptr, rope_sin_pos_ptr, head_dim, n_q, n_kv
    );

    if (q_out_from_device) q_out->from_device(stream);
    if (k_pos_from_device) k_pos->from_device(stream);
    if (v_pos_from_device) v_pos->from_device(stream);
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
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

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

void TopKSoftmaxGPU(const float *r, int n_experts, int k, float *topk_vals, int *topk_idx, hipStream_t stream) {
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