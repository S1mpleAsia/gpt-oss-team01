#pragma once

#include "config.hpp"
#include "tensor.hpp"
#include "utils.hpp"

// ---------- Embedding ----------
void embedding_lookup(Tensor *embedding,        // (vocab_size, hidden_dim)
                      int token_id, Tensor *x,  // (hidden_dim, )
                      bool x_from_device, hipStream_t stream = 0);

void rmsnorm(Tensor *x, Tensor *w, Tensor *out, long long layer_offset, bool x_to_device,
             bool out_from_device, float eps = 1e-5f, hipStream_t stream = 0);

void qkv_gemm(Tensor *x, const Tensor *W_qkv, const Tensor *b_qkv, Tensor *qkv,
              long long layer_offset, bool x_to_device, bool qkv_from_device,
              hipStream_t stream = 0);

void add_vector(Tensor *y, Tensor *b, bool y_to_device, bool b_to_device, bool y_from_device,
                hipStream_t stream = 0);

// ---------- QKV epilogue fused: bias + split + RoPE + cache ----------
void qkv_split_rope(Tensor *qkv_out, Tensor *q_out, Tensor *k_pos, Tensor *v_pos,
                    const Tensor *rope_cos_pos, const Tensor *rope_sin_pos, int head_dim, int n_q,
                    int n_kv, int pos, bool qkv_out_to_device, bool q_out_from_device,
                    bool k_pos_from_device, bool v_pos_from_device, hipStream_t stream = 0);

// ---------- Single-query attention fused ----------
void single_query_attn(Tensor *q, Tensor *K_cache, Tensor *V_cache, Tensor *mask,
                       Tensor *attn_sinks, Tensor *tb, int head_dim, int n_q, int kv_mul,
                       int kv_dim, int seq_len, int sliding_window, int pos, long long layer_offset,
                       bool q_to_device, bool k_cache_to_device, bool v_cache_to_device,
                       bool mask_to_device, bool tb_from_device, hipStream_t stream = 0);

void attn_out_project(Tensor *tb, const Tensor *W_o, const Tensor *b_o, Tensor *y,
                      long long layer_offset, bool tb_to_device, bool y_from_device,
                      hipStream_t stream = 0);

// ---------- Router GEMM ----------
void router_gemm(const Tensor *w_router, Tensor *t, const Tensor *b_router, Tensor *router_score,
                 long long layer_offset, bool t_to_device, bool r_from_device,
                 hipStream_t stream = 0);

// ---------- TopK + softmax(k) ----------
void topk_softmax(Tensor *r, Tensor *topk_vals, TensorI32 *topk_idx, bool r_to_device,
                  bool topk_vals_from_device, bool topk_idx_from_device, hipStream_t stream = 0);

// ---------- MoE apply TopK (batched) ----------
/* Với mỗi expert e trong topk_idx:
 *  z = W1[e] * t + b1[e]  (z tách thành gate/up)
 *  swiglu = silu(gate_clamped) * (up_clamped + 1)
 *  y = W2[e] * swiglu + b2[e]
 *  e_agg += topk_vals[e] * y
 * Có thể dùng gemm-strided-batched nếu layout tuần tự; nếu không, vòng for k nhỏ.
 */
void moe_apply_topk(Tensor *t, const Tensor *W1, const Tensor *b1, const Tensor *W2,
                    const Tensor *b2, TensorI32 *topk_idx, Tensor *topk_vals, Tensor *mlp1_out,
                    Tensor *gate_up, Tensor *tb3, Tensor *e_agg, float clamp_limit,
                    long long layer_offset, bool t_to_device, bool topk_idx_to_device,
                    bool topk_vals_to_device, bool e_agg_from_device, hipStream_t stream = 0);

static inline void moe_mlp1(const bf16 *W1_ptr, const float *t_ptr, const bf16 *b1_ptr,
                            float *mlp1_out_ptr, const int *topk_idx_ptr, int k, int inter_dim,
                            int hidden_dim, hipStream_t stream);

static inline void moe_swiglu(const float *mlp1_out_ptr, float *gate_up_ptr, int k, int inter_dim,
                              float clamp_limit, hipStream_t stream);

static inline void moe_mlp2(const bf16 *W2_ptr, const float *gate_up_ptr, const bf16 *b2_ptr,
                            float *tb3_ptr, const int *topk_idx_ptr, int k, int hidden_dim,
                            int inter_dim, hipStream_t stream);

static inline void moe_agg(const float *tb3_ptr, const float *topk_vals_ptr, float *e_agg_ptr,
                           int k, int hidden_dim, hipStream_t stream);

// ---------- Classifier (logits = W_out * x) ----------
void classifier_gemm(const Tensor *W_out, Tensor *x, Tensor *logits, bool x_to_device,
                     bool logits_from_device, hipStream_t stream = 0);
