#include "../include/layer_hip_batch.hpp"
#include <cmath>
#include <cfloat>

#define DEFAULT_BLOCK_SIZE 256

void embedding_lookup_batched(Tensor *embedding,  // Shape: [vocab_size, hidden_dim]
                              int *tokens,        // Shape: [batch_size]
                              Tensor *x,          // Shape: [batch_size, hidden_dim]
                              bool x_from_device, hipStream_t stream = 0) {
  GpuTimer("embedding_lookup");

  const int batch_size = x->shape[0];
  const size_t hidden_dim = x->shape[1];
  void *x_gpu = x->d_buf;

  for (int i = 0; i < batch_size; i++) {
    if (x->dtype == DType::BF16) {
      bf16 *src = (bf16 *)embedding->d_buf + (size_t)tokens[i] * hidden_dim;
      bf16 *dst = (bf16 *)x_gpu + 1ll * i * hidden_dim;
      CHECK_HIP(
        hipMemcpyAsync(dst, src, hidden_dim * sizeof(bf16), hipMemcpyDeviceToDevice, stream));
    } else {
      float *src = (float *)embedding->d_buf + (size_t)tokens[i] * hidden_dim;
      CHECK_HIP(hipMemcpyAsync((float *)x->d_buf + 1ll * i * hidden_dim, src,
                               hidden_dim * sizeof(float), hipMemcpyDeviceToDevice, stream));
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
                     float eps = 1e-5f, hipStream_t stream = 0) {
  // GpuTimer timer("rmsnorm");
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
void qkv_gemm_batched(Tensor *x,            // Shape: [batch_size, hidden_dim]
                      const Tensor *W_qkv,  // Shape: [out_features, hidden_dim]
                      const Tensor *b_qkv,  // Shape: [out_features]
                      Tensor *qkv,          // Shape: [batch_size, out_features]
                      long long layer_offset, bool x_to_device, bool qkv_from_device,
                      hipStream_t stream = 0) {}

// ---------- Split & RoPE ----------
void qkv_split_rope_batched(Tensor *qkv_out,             // Shape: [batch_size, (n_q + 2*n_kv)*hd]
                            Tensor *q_out,               // Shape: [batch_size, n_q*hd]
                            Tensor *k_pos,               // Shape: [batch_size, n_kv*hd]
                            Tensor *v_pos,               // Shape: [batch_size, n_kv*hd]
                            const Tensor *rope_cos_pos,  // Shape: [seq_len, hd/2]
                            const Tensor *rope_sin_pos,  // Shape: [seq_len, hd/2]
                            int head_dim, int n_q, int n_kv, int pos, bool qkv_out_from_device,
                            bool q_out_from_device, bool k_out_from_device, bool v_out_from_device,
                            hipStream_t stream = 0) {}

void add_vector_batched(Tensor *y,  // Shape: [batch_size, hidden_dim]
                        Tensor *b,  // Shape: [batch_size, hidden_dim]
                        bool y_to_device, bool b_to_device, bool y_from_device,
                        hipStream_t stream = 0) {}

// ---------- Attention ----------
void single_query_attn_batched(Tensor *q,           // Shape: [batch_size, n_q*hd]
                               Tensor *K_cache,     // Shape: [batch_size, layer, seq_len, kv_dim]
                               Tensor *V_cache,     // Shape: [batch_size, layer, seq_len, kv_dim]
                               Tensor *mask,        // Shape: [batch_size, seq_len, seq_len]
                               Tensor *attn_sinks,  // Shape: [n_layers, n_attn_heads]
                               Tensor *tb,          // Shape: [batch_size, n_q*hd]
                               int head_dim, int n_q, int kv_mul, int kv_dim, int seq_len,
                               int sliding_window, int pos, long long layer_offset,
                               bool q_to_device, bool k_cache_to_device, bool v_cache_to_device,
                               bool mask_to_device, bool tb_from_device, hipStream_t stream = 0) {}

void attn_out_project_batched(Tensor *tb,         // Shape: [batch_size, n_q*hd]
                              const Tensor *W_o,  // Shape: [hidden_dim, n_q*hd]
                              const Tensor *b_o,  // Shape: [hidden_dim]
                              Tensor *y,          // Shape: [batch_size, hidden_dim]
                              long long layer_offset, bool tb_to_device, bool y_from_device,
                              hipStream_t stream = 0) {}

// ---------- Router GEMM ----------
void router_gemm_batched(Tensor *t,               // Shape: [batch_size, hidden_dim]
                         const Tensor *w_router,  // Shape: [n_experts, hidden_dim]
                         const Tensor *b_router,  // Shape: [n_experts]
                         Tensor *router_scores,   // Shape: [batch_size, n_experts]
                         long long layer_offset, bool t_to_device, bool r_from_device,
                         hipStream_t stream = 0) {}

// ---------- TopK + softmax(k) ----------
void topk_softmax_batched(Tensor *r,            // Shape: [batch_size, n_experts]
                          Tensor *topk_vals,    // Shape: [batch_size, k]
                          TensorI32 *topk_idx,  // Shape: [batch_size, k]
                          bool r_to_device, bool topk_vals_from_device, bool topk_idx_from_device,
                          hipStream_t stream = 0) {}

// ---------- MoE apply TopK  ----------
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
  float clamp_limit, long long layer_offset, bool t_to_device, bool topk_idx_to_device,
  bool topk_vals_to_device, bool e_agg_from_device, hipStream_t stream = 0) {}

// ---------- Classifier & Residuals ----------
void classifier_gemm_batched(const Tensor *W_out,  // Shape: [vocab_size, hidden_dim]
                             Tensor *x,            // Shape: [batch_size, hidden_dim]
                             Tensor *logits,       // Shape: [batch_size, vocab_size]
                             bool x_to_device, bool logits_from_device, hipStream_t stream = 0) {}
