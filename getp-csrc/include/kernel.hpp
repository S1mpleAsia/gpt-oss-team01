#pragma once

#include <hip/hip_runtime.h>
#include "tensor.hpp"

template <int BM = 16, int BN = 128, int BK = 16, int TM = 2, int TN = 8>
__global__ void matmul_kernel(const float *__restrict__ A, const float *__restrict__ B,
                              float *__restrict__ C, const float *bias, int M, int N, int K);

template <int BM = 16, int BN = 128, int BK = 16, int TM = 2, int TN = 8>
__global__ void matmul_kernel_bf16(const float *__restrict__ A, const bf16 *__restrict__ B,
                                   float *__restrict__ C, const bf16 *bias, int M, int N, int K);

static inline void moe_init_buffers(Tensor *e_agg, Tensor *mlp1_out, Tensor *gate_up, Tensor *tb3,
                                    TensorI32 *sorted_pair_ids, TensorI32 *expert_offsets,
                                    Tensor *x_packed, int batch_size, int hidden_dim,
                                    hipStream_t stream);

// 1) sort & build offsets
static inline void moe_build_offsets(
  TensorI32 *topk_idx,         // [batch_size, experts_per_token]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  TensorI32 *expert_offsets,   // [n_experts + 1]
  int batch_size, int experts_per_token, int n_experts, hipStream_t stream);

// 2) pack X theo sorted ids
static inline void moe_pack_inputs(
  Tensor *x_in,                // [batch_size, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *x_packed,            // [batch_size * experts_per_token, hidden_dim]
  int batch_size, int hidden_dim, int experts_per_token, hipStream_t stream);

static inline void moe_pack_inputs_bf16(
  Tensor *x_in,                // [batch_size, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *x_packed,            // [batch_size * experts_per_token, hidden_dim]
  int batch_size, int hidden_dim, int experts_per_token, hipStream_t stream);

// 3) lấy max số hàng trên mỗi expert từ offsets
static inline int moe_get_max_rows_per_expert(TensorI32 *expert_offsets, int *d_max_rows,
                                              int n_experts, hipStream_t stream);

// 4) MLP1: (x_packed @ W1 + b1) -> mlp1_out  (2*inter_dim)
// w_mlp1: [n_layers, n_experts, hidden_dim, 2*inter_dim]
// b_mlp1: [n_layers, n_experts, 2*inter_dim]
static inline void moe_mlp1_forward(Tensor *x_packed,  // [total_pairs, hidden_dim]
                                    Tensor *w_mlp1, Tensor *b_mlp1,
                                    TensorI32 *expert_offsets,  // [n_experts+1]
                                    Tensor *mlp1_out,           // [total_pairs, 2*inter_dim]
                                    long long layer_offset, int n_experts, int hidden_dim,
                                    int inter_dim, int max_rows_per_expert, int total_pairs,
                                    hipStream_t stream);

static inline void moe_mlp1_forward_bf16(Tensor *x_packed,  // [total_pairs, hidden_dim]
                                         Tensor *w_mlp1, Tensor *b_mlp1,
                                         TensorI32 *expert_offsets,  // [n_experts+1]
                                         Tensor *mlp1_out,           // [total_pairs, 2*inter_dim]
                                         long long layer_offset, int n_experts, int hidden_dim,
                                         int inter_dim, int max_rows_per_expert, int total_pairs,
                                         hipStream_t stream);

// 5) SwiGLU (interleaved) & clamp
static inline void moe_swiglu(Tensor *mlp1_out,  // [total_pairs, 2*inter_dim]
                              Tensor *gate_up,   // [total_pairs, inter_dim]
                              int batch_size, int experts_per_token, int inter_dim,
                              float clamp_limit, hipStream_t stream);

static inline void moe_swiglu_bf16(Tensor *mlp1_out,  // [total_pairs, 2*inter_dim]
                                   Tensor *gate_up,   // [total_pairs, inter_dim]
                                   int batch_size, int experts_per_token, int inter_dim,
                                   float clamp_limit, hipStream_t stream);

static inline void moe_mlp1_swiglu_fused(
  Tensor *x_packed,           // [total_pairs, hidden_dim]
  Tensor *w_mlp1,             // [n_layers, n_experts, hidden_dim, 2*inter_dim]
  Tensor *b_mlp1,             // [n_layers, n_experts, 2*inter_dim]
  TensorI32 *expert_offsets,  // [n_experts+1]
  Tensor *gate_up,            // [total_pairs, inter_dim]
  long long layer_offset, int n_experts, int hidden_dim, int inter_dim,
  float clamp_limit,  // << Tham số từ swiglu
  int max_rows_per_expert, int total_pairs, hipStream_t stream);

// 6) MLP2: (gate_up @ W2 + b2) -> tb3  (hidden_dim)
// w_mlp2: [n_layers, n_experts, inter_dim, hidden_dim]
// b_mlp2: [n_layers, n_experts, hidden_dim]
static inline void moe_mlp2_forward(Tensor *gate_up,  // [total_pairs, inter_dim]
                                    Tensor *w_mlp2, Tensor *b_mlp2,
                                    TensorI32 *expert_offsets,  // [n_experts+1]
                                    Tensor *tb3,                // [total_pairs, hidden_dim]
                                    bool has_bias, long long layer_offset, int n_experts,
                                    int inter_dim, int hidden_dim, int max_rows_per_expert,
                                    int total_pairs, hipStream_t stream);

static inline void moe_mlp2_forward_bf16(Tensor *gate_up,  // [total_pairs, inter_dim]
                                         Tensor *w_mlp2, Tensor *b_mlp2,
                                         TensorI32 *expert_offsets,  // [n_experts+1]
                                         Tensor *tb3,                // [total_pairs, hidden_dim]
                                         bool has_bias, long long layer_offset, int n_experts,
                                         int inter_dim, int hidden_dim, int max_rows_per_expert,
                                         int total_pairs, hipStream_t stream);

// 7) scatter-add có scale theo topk_v -> e_agg
static inline void moe_scatter_aggregate(
  Tensor *tb3,                 // [total_pairs, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *topk_v,              // [batch_size, experts_per_token]
  Tensor *e_agg,               // [batch_size, hidden_dim]
  TensorI32 *expert_offsets,   // [n_experts+1]
  int hidden_dim, int experts_per_token, int n_experts, int max_rows_per_expert,
  hipStream_t stream);

static inline void moe_scatter_aggregate_120b(
  Tensor *tb3,                 // [total_pairs, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *topk_v,              // [batch_size, experts_per_token]
  Tensor *e_agg,               // [batch_size, hidden_dim]
  TensorI32 *expert_offsets,   // [n_experts+1]
  int hidden_dim, int experts_per_token, int n_experts, int max_rows_per_expert,
  hipStream_t stream);

static inline void moe_scatter_aggregate_120b_ep(
  Tensor *tb3,                 // [total_pairs, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *topk_v,              // [batch_size, experts_per_token]
  Tensor *e_agg,               // [batch_size, hidden_dim]
  TensorI32 *expert_offsets,   // [n_experts+1]
  int hidden_dim, int experts_per_token, int n_experts, int max_rows_per_expert,
  int start_expert_offset, int end_expert_offset, hipStream_t stream);
