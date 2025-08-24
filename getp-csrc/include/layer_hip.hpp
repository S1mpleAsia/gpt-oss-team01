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
void SingleQueryAttentionGPU(
  Tensor *q, Tensor *K_cache, Tensor *V_cache, Tensor *mask,
  Tensor *attn_sinks, Tensor *tb, int head_dim, int n_q, int kv_mul,
  int kv_dim, int seq_len, int sliding_window, int pos,
  long long layer_offset, bool q_to_device, bool k_cache_to_device,
  bool v_cache_to_device, bool mask_to_device, bool tb_from_device,
  hipStream_t stream = 0
);

void AttnOutProjectGPU(
  Tensor *tb, const Tensor *W_o, const Tensor *b_o, Tensor *y,
  long long layer_offset, bool tb_to_device, bool y_from_device,
  hipStream_t stream = 0
);

// ---------- Linear + bias + residual (epilogue nếu có Lt) ----------
/* y = W [out,in] * x [in] + b  ;  x_out = x_resid + y */
void LinearBiasResidualGPU(const float *W, const float *x, const float *bias,
                           float *x_resid_inout,  // in: residual src, out: x += y
                           int in_features, int out_features, hipStream_t stream);

// ---------- Router GEMM ----------
/* r = W_router [n_experts, hidden] * t [hidden] + b */
void RouterGemmGPU(const Tensor *w_router, Tensor *t, const Tensor *b_router,
                   Tensor *router_score, long long layer_offset,
                   bool t_to_device, bool r_from_device,
                   hipStream_t stream = 0);

// ---------- TopK + softmax(k) ----------
void TopKSoftmaxGPU(Tensor *r, Tensor *topk_vals, TensorI32 *topk_idx,
                    bool r_to_device, bool topk_vals_from_device,
                    bool topk_idx_from_device, hipStream_t stream = 0);

// ---------- MoE apply TopK (batched) ----------
/* Với mỗi expert e trong topk_idx:
 *  z = W1[e] * t + b1[e]  (z tách thành gate/up)
 *  swiglu = silu(gate_clamped) * (up_clamped + 1)
 *  y = W2[e] * swiglu + b2[e]
 *  e_agg += topk_vals[e] * y
 * Có thể dùng gemm-strided-batched nếu layout tuần tự; nếu không, vòng for k nhỏ.
 */
void MoEApplyTopKGPU(Tensor *t, const Tensor *W1, const Tensor *b1,
                    const Tensor *W2, const Tensor *b2, TensorI32 *topk_idx,
                    Tensor *topk_vals, Tensor *mlp1_out, Tensor *gate_up, Tensor *tb3, Tensor *e_agg, float clamp_limit,
                    long long layer_offset, bool t_to_device,
                    bool topk_idx_to_device, bool topk_vals_to_device,
                    bool e_agg_from_device, hipStream_t stream = 0);

// ---------- Classifier (logits = W_out * x) ----------
void ClassifierGemmGPU(const Tensor *W_out, Tensor *x, Tensor *logits,
                        bool x_to_device, bool logits_from_device,
                        hipStream_t stream = 0);