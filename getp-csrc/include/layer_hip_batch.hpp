#pragma once

#include "config.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include "kernel.hpp"

// ---------- Embedding ----------
void embedding_lookup_batched(Tensor *embedding,      // Shape: [vocab_size, hidden_dim]
                              int *tokens,            // Shape: [batch_size]
                              TensorI32 *tokens_buf,  // Shape: [batch_size]
                              Tensor *x,              // Shape: [batch_size, hidden_dim]
                              int cur_batch_size, bool x_from_device, hipStream_t stream = 0);

void embedding_lookup_shard_batched(Tensor *embedding,      // Shape: [vocab_size, hidden_dim]
                                    int *tokens,            // Shape: [batch_size]
                                    TensorI32 *tokens_buf,  // Shape: [batch_size]
                                    Tensor *x,              // Shape: [batch_size, hidden_dim]
                                    int cur_batch_size, int tp_rank, bool x_from_device,
                                    hipStream_t stream);

// ---------- RMSNorm ----------
void rmsnorm_batched(Tensor *x,    // Shape: [batch_size, hidden_dim]
                     Tensor *w,    // Shape: [hidden_dim]
                     Tensor *out,  // Shape: [batch_size, hidden_dim]
                     int cur_batch_size, long long layer_offset, bool x_to_device,
                     bool out_from_device, float eps = 1e-5f, hipStream_t stream = 0);

void residual_rmsnorm_batched(Tensor *x,         // Shape: [batch_size, hidden_dim]
                              Tensor *residual,  // Shape: [batch_size, hidden_dim]
                              Tensor *w,         // Shape: [n_layers, hidden_dim]
                              Tensor *out,       // Shape: [batch_size, hidden_dim]
                              int cur_batch_size, long long layer_offset, float epsilon,
                              hipStream_t stream = 0);

// ---------- QKV GEMM ----------
void qkv_gemm_batched(Tensor *x,            // Shape: [batch_size, hidden_dim]
                      const Tensor *W_qkv,  // Shape: [out_features, hidden_dim]
                      const Tensor *b_qkv,  // Shape: [out_features]
                      Tensor *qkv,          // Shape: [batch_size, out_features]
                      int cur_batch_size, long long layer_offset, bool x_to_device,
                      bool qkv_from_device, hipStream_t stream = 0);

void qkv_gemm_batched_v2(Tensor *x,            // Shape: [batch_size, hidden_dim]
                         const Tensor *W_qkv,  // Shape: [hidden_dim, out_features]
                         const Tensor *b_qkv,  // Shape: [out_features]
                         Tensor *qkv,          // Shape: [batch_size, out_features]
                         int cur_batch_size, long long layer_offset, bool x_to_device,
                         bool qkv_from_device, hipStream_t stream = 0);

// ---------- Split & RoPE ----------
void qkv_split_rope_batched(Tensor *qkv_out,             // Shape: [batch_size, (n_q + 2*n_kv)*hd]
                            Tensor *q_out,               // Shape: [batch_size, n_q*hd]
                            Tensor *k_pos,               // Shape: [batch_size, n_kv*hd]
                            Tensor *v_pos,               // Shape: [batch_size, n_kv*hd]
                            const Tensor *rope_cos_pos,  // Shape: [seq_len, hd/2]
                            const Tensor *rope_sin_pos,  // Shape: [seq_len, hd/2]
                            int cur_batch_size, int head_dim, int n_q, int n_kv, int pos,
                            bool qkv_out_to_device, bool q_out_from_device, bool k_out_from_device,
                            bool v_out_from_device, hipStream_t stream = 0);

void qkv_split_rope_fused(Tensor *qkv_out,  // Shape: [batch_size, (n_q + 2*n_kv)*hd]
                          Tensor *q_out,    // Shape: [batch_size, n_q*hd]
                          Tensor *K_cache,  // Shape: [batch_size, n_layers, seq_len, kv_dim]
                          Tensor *V_cache,  // Shape: [batch_size, n_layers, seq_len, kv_dim]
                          const Tensor *rope_cos_pos,  // Shape: [seq_len, hd/2]
                          const Tensor *rope_sin_pos,  // Shape: [seq_len, hd/2]
                          int cur_batch_size, int head_dim, int n_q, int n_kv, int pos,
                          long long layer_offset, hipStream_t stream = 0);

__global__ void add_vector_kernel_batched(float *y, const float *b, int len);

__global__ void add_vector_kernel_v2(float *y, const float *b, int len);

__global__ void add_vector_2d_kernel(float *__restrict__ dst, const float *__restrict__ src,
                                     int height, int width, int dst_stride);

void add_vector_batched(Tensor *y,  // Shape: [batch_size, hidden_dim]
                        Tensor *b,  // Shape: [batch_size, hidden_dim]
                        int cur_batch_size, bool y_to_device, bool b_to_device, bool y_from_device,
                        hipStream_t stream = 0);

// ---------- Attention ----------
void single_query_attn_batched(Tensor *q,           // Shape: [batch_size, n_q*hd]
                               Tensor *K_cache,     // Shape: [batch_size, layer, seq_len, kv_dim]
                               Tensor *V_cache,     // Shape: [batch_size, layer, seq_len, kv_dim]
                               Tensor *mask,        // Shape: [batch_size, seq_len, seq_len]
                               Tensor *attn_sinks,  // Shape: [n_layers, n_attn_heads]
                               Tensor *tb,          // Shape: [batch_size, n_q*hd]
                               int cur_batch_size, int head_dim, int n_q, int kv_mul, int kv_dim,
                               int seq_len, int sliding_window, int pos, long long layer_offset,
                               bool q_to_device, bool k_cache_to_device, bool v_cache_to_device,
                               bool mask_to_device, bool tb_from_device, hipStream_t stream = 0);

void attn_out_project_batched(Tensor *tb,         // Shape: [batch_size, n_q*hd]
                              const Tensor *W_o,  // Shape: [hidden_dim, n_q*hd]
                              const Tensor *b_o,  // Shape: [hidden_dim]
                              Tensor *y,          // Shape: [batch_size, hidden_dim]
                              int cur_batch_size, long long layer_offset, bool tb_to_device,
                              bool y_from_device, hipStream_t stream = 0);

void attn_out_project_batched_v2(Tensor *tb,         // Shape: [batch_size, n_attn_heads * head_dim]
                                 const Tensor *W_o,  // Shape: [n_attn_heads * head_dim, hidden_dim]
                                 const Tensor *b_o,  // Shape: [hidden_dim]
                                 Tensor *y,          // Shape: [batch_size, hidden_dim]
                                 bool has_bias, int cur_batch_size, long long layer_offset,
                                 bool tb_to_device, bool y_from_device, hipStream_t stream = 0);

// ---------- Router GEMM ----------
void router_gemm_batched(const Tensor *w_router,  // Shape: [n_experts, hidden_dim]
                         Tensor *t,               // Shape: [batch_size, hidden_dim]
                         const Tensor *b_router,  // Shape: [n_experts]
                         Tensor *router_scores,   // Shape: [batch_size, n_experts]
                         int cur_batch_size, long long layer_offset, bool t_to_device,
                         bool r_from_device, hipStream_t stream = 0);

void router_gemm_v2(const Tensor *w_router,  // Shape: [hidden_dim, n_experts]
                    Tensor *t,               // Shape: [batch_size, hidden_dim]
                    const Tensor *b_router,  // Shape: [n_experts]
                    Tensor *router_scores,   // Shape: [batch_size, n_experts]
                    int cur_batch_size, long long layer_offset, bool t_to_device,
                    bool r_from_device, hipStream_t stream = 0);

// ---------- TopK + softmax(k) ----------
void topk_softmax_batched(Tensor *r,            // Shape: [batch_size, n_experts]
                          Tensor *topk_vals,    // Shape: [batch_size, k]
                          TensorI32 *topk_idx,  // Shape: [batch_size, k]
                          int cur_batch_size, bool r_to_device, bool topk_vals_from_device,
                          bool topk_idx_from_device, hipStream_t stream = 0);

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
  int cur_batch_size, float clamp_limit, long long layer_offset, bool t_to_device,
  bool topk_idx_to_device, bool topk_vals_to_device, bool e_agg_from_device,
  hipStream_t stream = 0);

/*  --------------  Start MoE  --------------------  */
static inline void moe_init_buffers_hip(Tensor *e_agg, Tensor *mlp1_out, Tensor *gate_up,
                                        Tensor *tb3, TensorI32 *sorted_pair_ids,
                                        TensorI32 *expert_offsets, Tensor *x_packed, int batch_size,
                                        int hidden_dim, hipStream_t stream);

static inline void moe_build_offsets_hip(
  TensorI32 *topk_idx,         // [batch_size, experts_per_token]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  TensorI32 *expert_offsets,   // [n_experts + 1]
  int batch_size, int experts_per_token, int n_experts, hipStream_t stream);

static inline void moe_pack_inputs_hip(
  Tensor *x_in,                // [batch_size, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *x_packed,            // [batch_size * experts_per_token, hidden_dim]
  int batch_size, int hidden_dim, int experts_per_token, hipStream_t stream);

static inline int moe_get_max_rows_per_expert_hip(TensorI32 *expert_offsets, int *d_max_rows,
                                                  int n_experts, hipStream_t stream);

static inline void moe_mlp1_forward_hip(Tensor *x_packed,  // [total_pairs, hidden_dim]
                                        Tensor *w_mlp1, Tensor *b_mlp1,
                                        TensorI32 *expert_offsets,  // [n_experts+1]
                                        Tensor *mlp1_out,           // [total_pairs, 2*inter_dim]
                                        long long layer_offset, int n_experts, int hidden_dim,
                                        int inter_dim, int max_rows_per_expert, int total_pairs,
                                        hipStream_t stream);

static inline void moe_swiglu_hip(Tensor *mlp1_out,  // [total_pairs, 2*inter_dim]
                                  Tensor *gate_up,   // [total_pairs, inter_dim]
                                  int batch_size, int experts_per_token, int inter_dim,
                                  float clamp_limit, hipStream_t stream);

static inline void moe_mlp2_forward_hip(Tensor *gate_up,  // [total_pairs, inter_dim]
                                        Tensor *w_mlp2, Tensor *b_mlp2,
                                        TensorI32 *expert_offsets,  // [n_experts+1]
                                        Tensor *tb3,                // [total_pairs, hidden_dim]
                                        bool has_bias, long long layer_offset, int n_experts,
                                        int inter_dim, int hidden_dim, int max_rows_per_expert,
                                        int total_pairs, hipStream_t stream);

static inline void moe_scatter_aggregate_hip(
  Tensor *tb3,                 // [total_pairs, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *topk_v,              // [batch_size, experts_per_token]
  Tensor *e_agg,               // [batch_size, hidden_dim]
  TensorI32 *expert_offsets,   // [n_experts+1]
  int hidden_dim, int experts_per_token, int n_experts, int max_rows_per_expert,
  hipStream_t stream);

static inline void moe_scatter_aggregate_ep_hip(
  Tensor *tb3,                 // [total_pairs, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *topk_v,              // [batch_size, experts_per_token]
  Tensor *e_agg,               // [batch_size, hidden_dim]
  TensorI32 *expert_offsets,   // [n_experts+1]
  int hidden_dim, int experts_per_token, int n_experts, int max_rows_per_expert,
  int start_expert_offset, int end_expert_offset, hipStream_t stream);
/*  --------------  End MoE  --------------------  */

static inline void moe_mlp1_batched(Tensor *t, Tensor *w_mlp1, Tensor *b_mlp1, TensorI32 *topk_idx,
                                    Tensor *mlp1_out, int cur_batch_size, bool t_to_device,
                                    bool topk_idx_to_device, long long layer_offset,
                                    hipStream_t stream = 0);

static inline void moe_swiglu_batched(Tensor *mlp1_out, Tensor *gate_up, int cur_batch_size, int k,
                                      float clamp_limit, hipStream_t stream = 0);

static inline void moe_mlp2_batched(Tensor *gate_up, Tensor *w_mlp2, Tensor *b_mlp2, Tensor *tb3,
                                    TensorI32 *topk_idx, int k, bool has_bias, int cur_batch_size,
                                    long long layer_offset, hipStream_t stream = 0);

static inline void moe_agg_batched(Tensor *tb3, Tensor *topk_val, Tensor *e_agg, int k,
                                   int cur_batch_size, bool e_agg_from_device,
                                   hipStream_t stream = 0);

// ---------- Classifier & Residuals ----------
void classifier_gemm_batched(const Tensor *W_out,  // Shape: [vocab_size, hidden_dim]
                             Tensor *x,            // Shape: [batch_size, hidden_dim]
                             Tensor *logits,       // Shape: [batch_size, vocab_size],
                             int cur_batch_size, bool x_to_device, bool logits_from_device,
                             hipStream_t stream = 0);

void classifier_gemm_batched_v2(const Tensor *W_out,  // Shape: [vocab_size, hidden_dim]
                                Tensor *x,            // Shape: [batch_size, hidden_dim]
                                Tensor *logits,       // Shape: [batch_size, vocab_size]
                                int cur_batch_size, bool x_to_device, bool logits_from_device,
                                hipStream_t stream = 0);

void max_logits_batched(Tensor *logits, Tensor *logits_max, TensorI32 *logits_id,
                        int cur_batch_size, int tp_rank, bool logits_to_device,
                        bool logits_max_from_device, bool logits_id_from_device,
                        hipStream_t stream = 0);

void reduce_logits_batched(Tensor *logits_max_total, TensorI32 *logits_id_total, Tensor *logits_max,
                           TensorI32 *logits_id, int cur_batch_size,
                           bool logits_max_total_to_device, bool logits_id_total_to_device,
                           bool logits_max_from_device, bool logits_id_from_device,
                           hipStream_t stream = 0);
