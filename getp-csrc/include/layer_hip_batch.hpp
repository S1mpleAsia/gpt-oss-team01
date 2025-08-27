#pragma once

#include "config.hpp"
#include "tensor.hpp"
#include "utils.hpp"

// ---------- Embedding ----------
void embedding_lookup_batched(
  Tensor *embedding,  // Shape: [vocab_size, hidden_dim]
  int *tokens,        // Shape: [batch_size]
  Tensor *x,          // Shape: [batch_size, hidden_dim]
  bool x_from_device, int cur_batch_size, hipStream_t stream = 0
);

// ---------- RMSNorm ----------
void rmsnorm_batched(
  Tensor *x,    // Shape: [batch_size, hidden_dim]
  Tensor *w,    // Shape: [hidden_dim]
  Tensor *out,  // Shape: [batch_size, hidden_dim]
  long long layer_offset, bool x_to_device, bool out_from_device,
  int cur_batch_size, float eps = 1e-5f, hipStream_t stream = 0
);

// ---------- QKV GEMM ----------
void qkv_gemm_batched(
  Tensor *x,            // Shape: [batch_size, hidden_dim]
  const Tensor *W_qkv,  // Shape: [out_features, hidden_dim]
  const Tensor *b_qkv,  // Shape: [out_features]
  Tensor *qkv,          // Shape: [batch_size, out_features]
  long long layer_offset, bool x_to_device, bool qkv_from_device,
  int cur_batch_size, hipStream_t stream = 0
);

// ---------- Split & RoPE ----------
void qkv_split_rope_batched(
  Tensor *qkv_out,             // Shape: [batch_size, (n_q + 2*n_kv)*hd]
  Tensor *q_out,               // Shape: [batch_size, n_q*hd]
  Tensor *k_pos,               // Shape: [batch_size, n_kv*hd]
  Tensor *v_pos,               // Shape: [batch_size, n_kv*hd]
  const Tensor *rope_cos_pos,  // Shape: [seq_len, hd/2]
  const Tensor *rope_sin_pos,  // Shape: [seq_len, hd/2]
  int head_dim, int n_q, int n_kv, int pos, bool qkv_out_to_device,
  bool q_out_from_device, bool k_out_from_device, bool v_out_from_device,
  int cur_batch_size, hipStream_t stream = 0
);

void add_vector_batched(
  Tensor *y,  // Shape: [batch_size, hidden_dim]
  Tensor *b,  // Shape: [batch_size, hidden_dim]
  bool y_to_device, bool b_to_device, bool y_from_device,
  hipStream_t stream = 0
);

// ---------- Attention ----------
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
  bool tb_from_device, int cur_batch_size, hipStream_t stream = 0
);

void attn_out_project_batched(
  Tensor *tb,         // Shape: [batch_size, n_q*hd]
  const Tensor *W_o,  // Shape: [hidden_dim, n_q*hd]
  const Tensor *b_o,  // Shape: [hidden_dim]
  Tensor *y,          // Shape: [batch_size, hidden_dim]
  long long layer_offset, bool tb_to_device, bool y_from_device,
  int cur_batch_size, hipStream_t stream = 0
);

// ---------- Router GEMM ----------
void router_gemm_batched(
  const Tensor *w_router,  // Shape: [n_experts, hidden_dim]
  Tensor *t,               // Shape: [batch_size, hidden_dim]
  const Tensor *b_router,  // Shape: [n_experts]
  Tensor *router_scores,   // Shape: [batch_size, n_experts]
  long long layer_offset, bool t_to_device, bool r_from_device,
  int cur_batch_size, hipStream_t stream = 0
);

// ---------- TopK + softmax(k) ----------
void topk_softmax_batched(
  Tensor *r,            // Shape: [batch_size, n_experts]
  Tensor *topk_vals,    // Shape: [batch_size, k]
  TensorI32 *topk_idx,  // Shape: [batch_size, k]
  bool r_to_device, bool topk_vals_from_device, bool topk_idx_from_device,
  int cur_batch_size, hipStream_t stream = 0
);

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
  float clamp_limit, long long layer_offset, bool t_to_device,
  bool topk_idx_to_device, bool topk_vals_to_device, bool e_agg_from_device,
  int cur_batch_size, hipStream_t stream = 0
);

// ---------- Classifier & Residuals ----------
void classifier_gemm_batched(
  const Tensor *W_out,  // Shape: [vocab_size, hidden_dim]
  Tensor *x,            // Shape: [batch_size, hidden_dim]
  Tensor *logits,       // Shape: [batch_size, vocab_size]
  bool x_to_device, bool logits_from_device, int cur_batch_size,
  hipStream_t stream = 0
);
