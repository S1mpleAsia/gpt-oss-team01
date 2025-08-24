#pragma once

#include "config.hpp"
#include "tensor.hpp"
#include "utils.hpp"

// ---------- Embedding ----------
void EmbeddingLookupGPU(Tensor *embedding,  // (vocab_size, hidden_dim)
                        int token_id, Tensor *x,  // (hidden_dim, )
                        bool x_from_device, hipStream_t stream);

void RMSNormGPU(Tensor *x, Tensor *w, Tensor *out, long long layer_offset,
                bool x_to_device, bool out_from_device, float eps = 1e-5f,
                hipStream_t stream = 0);

void QKVGemmGPU_Old(const float *w_qkv,  // (out, in)
                const float *b_qkv,  // (out, )
                const float *t,      // (in, )
                float *out,          // (out, )
                int hidden_dim, int head_dim, int in_features, int out_features,
                hipStream_t stream);
void QKVGemmGPU(Tensor *x, const Tensor *W_qkv, const Tensor *b_qkv, 
                Tensor *qkv, long long layer_offset, bool x_to_device,
                bool qkv_from_device, hipStream_t stream = 0);

void SplitQKVGPU(Tensor *qkv, int head_dim, int n_q, int n_kv,
                Tensor *q, Tensor *k, Tensor *v, bool qkv_to_device,
                bool q_from_device, bool k_from_device,
                bool v_from_device, hipStream_t stream = 0);

// (tuỳ chọn) cộng bias riêng nếu không có epilogue:
void AddVectorGPU(Tensor *y, Tensor *b, bool y_to_device,
                bool b_to_device, bool y_from_device, hipStream_t stream = 0);

// ---------- QKV epilogue fused: bias + split + RoPE + cache ----------
/* Đọc qkv_out (đã có/hoặc chưa có bias), cộng bias (nếu b != nullptr),
 *  tách Q; ghi K,V vào cache vị trí pos; áp RoPE cho Q và K@pos
 */
void QKVEpilogueSplitRoPECacheGPU(
    Tensor *qkv_out, Tensor *q_out, Tensor *k_pos, Tensor *v_pos,
    const Tensor *rope_cos_pos, const Tensor *rope_sin_pos,
    int head_dim, int n_q, int n_kv, int pos, bool qkv_out_to_device,
    bool q_out_from_device, bool k_pos_from_device,
    bool v_pos_from_device, hipStream_t stream = 0
);

// ---------- Single-query attention fused ----------
/* Tính logits q·k_t/√d + mask (nếu có) + 1 logit sink; softmax online;
 *  Trả về tb (concat heads, size = n_q*hd). Sink chỉ đi vào mẫu số (denom),
 *  KHÔNG cộng vào tích lũy V.
 */
void SingleQueryAttentionGPU(const float *q,                   // [n_q*hd]
                             const float *K_cache,             // [seq_len*kv_dim]
                             const float *V_cache,             // [seq_len*kv_dim]
                             const float *mask_row,            // [seq_len] or nullptr
                             const float *attn_sink_per_head,  // [n_q]
                             float *tb,                        // [n_q*hd]
                             int head_dim, int n_q,
                             int kv_mul,  // n_q / n_kv
                             int kv_dim,  // head_dim * n_kv
                             int seq_len, int pos, hipStream_t stream);

// ---------- Linear + bias + residual (epilogue nếu có Lt) ----------
/* y = W [out,in] * x [in] + b  ;  x_out = x_resid + y */
void LinearBiasResidualGPU(const float *W, const float *x, const float *bias,
                           float *x_resid_inout,  // in: residual src, out: x += y
                           int in_features, int out_features, hipStream_t stream);

// ---------- Router GEMM ----------
/* r = W_router [n_experts, hidden] * t [hidden] + b */
void RouterGemmGPU(const float *W_router, const float *t, const float *bias,
                   float *r,  // [n_experts]
                   int hidden_dim, int n_experts, hipStream_t stream);

// ---------- TopK + softmax(k) ----------
void TopKSoftmaxGPU(const float *r,  // [n_experts]
                    int n_experts, int k,
                    float *topk_vals,  // [k] (softmaxed)
                    int *topk_idx,     // [k]
                    hipStream_t stream);

// ---------- MoE apply TopK (batched) ----------
/* Với mỗi expert e trong topk_idx:
 *  z = W1[e] * t + b1[e]  (z tách thành gate/up)
 *  swiglu = silu(gate_clamped) * (up_clamped + 1)
 *  y = W2[e] * swiglu + b2[e]
 *  e_agg += topk_vals[e] * y
 * Có thể dùng gemm-strided-batched nếu layout tuần tự; nếu không, vòng for k nhỏ.
 */
void MoEApplyTopKGPU(
  const float *t,          // [hidden]
  const bf16 *W1,          // [n_layers? n_experts? 2*inter, hidden] - pass base for this layer
  const bf16 *b1,          // [n_experts, 2*inter]
  const bf16 *W2,          // [n_experts, hidden, inter]
  const bf16 *b2,          // [n_experts, hidden]
  const int *topk_idx,     // [k]
  const float *topk_vals,  // [k] (đã softmax)
  float *work_gate_up,     // [inter] scratch
  float *e_agg_inout,      // [hidden] (accumulate)
  int hidden_dim, int inter_dim,
  int k,              // experts_per_token
  float clamp_limit,  // swiglu_limit
  hipStream_t stream);

// ---------- Classifier (logits = W_out * x) ----------
void ClassifierGemmGPU(const float *W_out,  // [vocab, hidden]
                       const float *x,      // [hidden]
                       float *logits,       // [vocab]
                       int hidden_dim, int vocab_size, hipStream_t stream);