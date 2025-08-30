#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tensor.hpp"

#define RUN_BATCH
#define BATCH_SIZE 2
// #define PRINT_LOGITS
// #define TIME_GPU
// #define DEBUG

#define DP 2
#define PP 1
#define TP 1
#define TOTAL_GPUS_NEEDED ((DP) * (PP) * (TP))
#define TOTAL_PIPELINES ((PP) * (TP))

typedef struct {
  // token_embedding_table - embedding.weight
  Tensor *token_embedding_table;  // (vocab_size, hidden_dim) (in, out)
  // weights for rmsnorms
  Tensor *rms_attn_w;  // (n_layers, hidden_dim) [attn.norm.scale]
  Tensor *rms_ffn_w;   // (n_layers, hidden_dim) [mlp.norm.scale]
  // weights for attention [attn.qkv.weight & attn.qkv.bias]
  Tensor *w_qkv;       // (n_layers, head_dim * n_attn_heads + 2 * head_dim * n_kv_heads,
                       // hidden_dim) where w_q (head_dim * n_attn_heads, hidden_dim)
                       // (out_features, in_features) w_k (head_dim * n_kv_heads,
                       // hidden_dim)  (out_features, in_features) w_v (head_dim *
                       // n_kv_heads, hidden_dim)  (out_features, in_features)
  Tensor *w_o;         // (n_layers, hidden_dim, head_dim * n_attn_heads)
  Tensor *b_qkv;       // (n_layers, head_dim * n_attn_heads + 2 * head_dim *
                       // n_kv_heads) (head_dim * n_attn_heads) (head_dim * n_kv_heads)
                       // (head_dim * n_kv_heads)
  Tensor *b_o;         // (n_layers, hidden_dim)
  Tensor *attn_sinks;  // (n_layers, n_attn_heads)
  // weights for router [mlp.gate.weight & mlp.gate.bias]
  Tensor *w_router;  // (n_layers, hidden_dim, n_experts)
  Tensor *b_router;  // (n_layers, n_experts)
  // weights for MoE [mlp.mlp1_weight & mlp.mlp1_bias & mlp.mlp2_weight &
  // mlp.mlp2_bias] NOTE: gate_up projects from hidden_dim to intermediate_dim,
  // the shape is kinda reverted because the original code use einsum to reduce
  // over hidden_dim
  Tensor *w_mlp1;  // gate_up_proj (n_layers, n_experts, 2 * intermediate_dim,
                   // hidden_dim)
  Tensor *w_mlp2;  // down_proj (n_layers, n_experts, hidden_dim, intermediate_dim)
  Tensor *b_mlp1;  // gate_up proj (n_layers, n_experts, 2 * intermediate_dim)
  Tensor *b_mlp2;  // down_proj (n_layers, n_experts, hidden_dim)
  // final norm [norm.scale]
  Tensor *rms_out_w;  // (hidden_dim, )
  // classifier weights for the logits [unembedding.weight]
  Tensor *out;  // (vocab_size, hidden_dim) (out, in)
} OurTransformerWeights;

typedef struct {
  Tensor *cos_tensor;
  Tensor *sin_tensor;

  // current wave of activations
  Tensor *x;             // activation at current time stamp (hidden_dim, )
  Tensor *t;             // same, but inside a residual branch (hidden_dim, )
  Tensor *tb;            // (head_dim * n_attn_heads, )
  Tensor *tb2;           // (hidden_dim, )
  Tensor *tb3;           // (n_experts, hidden_dim)
  Tensor *router_score;  // router score (n_experts, )
  Tensor *topk_v;        // topk expert weights (experts_per_token, )
  TensorI32 *topk_i;     // topk expert indices (experts_per_token, )
  Tensor *mlp1_out;
  Tensor *gate;
  Tensor *up;
  Tensor *gate_up;
  Tensor *e_agg;
  Tensor *qkv;     // an additional buffer just for convenience (head_dim *
                   // (n_attn_heads + 2 * n_kv_heads), )
  Tensor *q;       // query (n_attn_heads * head_dim,)
  Tensor *k;       // key (n_kv_heads * head_dim,)
  Tensor *v;       // value (n_kv_heads * head_dim,)
  Tensor *att;     // buffer for scores/attention values (n_heads, seq_len)
  Tensor *logits;  // output logits
  // kv cache
  Tensor *key_cache;    // (layer, seq_len, kv_dim)
  Tensor *value_cache;  // (layer, seq_len, kv_dim)
  Tensor *mask;
} OurRunState;

typedef struct {
  int id;
  long long *local_token_ptr;
  int start_idx;
  int end_idx;
} ThreadArgs;

typedef struct {
  hipStream_t stream;
  hipEvent_t kernelEnd;
} StreamTotal;
