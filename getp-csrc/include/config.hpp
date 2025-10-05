#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tensor.hpp"
// #include "config_run.hpp"
#include "pipeline.hpp"

#define BATCH_SIZE 896
#define PP_SLOT 1
#define KV16
// #define PRINT_LOGITS
// #define TIME_GPU
// #define DEBUG
#define RUN_20B
#define RUN_EP

#ifdef RUN_20B
#define DP 8
#define PP 1
#define TP 1
#else
#define DP 1
#define PP 1
#define TP 8
#endif
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
  Tensor *out;         // (vocab_size, hidden_dim) (out, in)
  Tensor *out_buffer;  // (hidden_dim, shard_vocab_size)
} OurTransformerWeights;

typedef struct {
  Tensor *cos_tensor;
  Tensor *sin_tensor;

  // current wave of activations
  Tensor *x;  // activation at current time stamp (hidden_dim, )
  Tensor *x_quantize;
  Tensor *t;   // same, but inside a residual branch (hidden_dim, )
  Tensor *tb;  // (head_dim * n_attn_heads, )
  // Tensor *tb_buf;
  Tensor *tb2;  // (hidden_dim, )
  Tensor *tb2_buf;
  Tensor *tb2_recv;
  Tensor *tb2_quantize;
  Tensor *tb2_half;
  Tensor *tb2_quad;
  Tensor *tb2_oct;
  Tensor *tb3;  // (BATCH_SIZE, experts_per_token)
  Tensor *tb3_buf;
  Tensor *tb3_recv;
  Tensor *router_score;  // router score (n_experts, )
  Tensor *topk_v;        // topk expert weights (experts_per_token, )
  TensorI32 *topk_i;     // topk expert indices (experts_per_token, )
  Tensor *mlp1_out;      // [batch_size * experts_per_token, 2 * inter_dim]
  // Tensor *gate;
  // Tensor *up;
  Tensor *gate_up;  // [batch_size * experts_per_toeken, inter_dim]
  Tensor *e_agg;    // [batch_size, hidden_dim]
  Tensor *e_agg_buf;
  Tensor *e_agg_recv;
  Tensor *e_agg_quantize;
  Tensor *qkv;  // an additional buffer just for convenience (head_dim *
                // (n_attn_heads + 2 * n_kv_heads), )
  Tensor *tmp_qkv;
  Tensor *q;  // query (n_attn_heads * head_dim,)
  // Tensor *k;           // key (n_kv_heads * head_dim,)
  // Tensor *v;           // value (n_kv_heads * head_dim,)
  // Tensor *att;         // buffer for scores/attention values (n_heads, seq_len)
  Tensor *logits;      // output logits
  Tensor *tmp_logits;  // (batch_size, shard_vocab)
  // kv cache
  Tensor *key_cache;    // (layer, seq_len, kv_dim)
  Tensor *value_cache;  // (layer, seq_len, kv_dim)
  Tensor *key_cache_odd;
  Tensor *value_cache_odd;
  Tensor *key_cache_even;
  Tensor *value_cache_even;
  Tensor *mask;

  // MoE buffer
  TensorI32 *sorted_pair_ids;  // [batch_size * experts_per_token]
  TensorI32 *expert_offsets;   // [n_experts + 1]
  Tensor *x_packed;            // [batch_size * experts_per_token, hidden_dim]
  Tensor *x_packed_local;
  TensorI32 *expert_offsets_local;
  int *max_rows;

  // Multi-GPU related
  TensorI32 *tokens_buf;
  PipelineEach *pipeline_each;

  // Flash-attention
  Tensor *g_fa_pmax;  // [batch_size, n_attn_heads, c_max]
  Tensor *g_fa_psum;  // [batch_size, n_attn_heads, c_max]
  Tensor *g_fa_pnum;  // [batch_size, n_attn_heads, c_max, head_dim]

  float *logits_out;
  Tensor *logits_max;
  Tensor *logits_max_total;
  TensorI32 *logits_id;
  TensorI32 *logits_id_total;
} OurRunState;

typedef struct {
  int id;
  long long *local_token_ptr;
  int start_idx;
  int end_idx;
  pthread_barrier_t *tp_barrier;
} OnePathArgs;

typedef struct {
  int tp_rank;
  int pp_rank;
  vector<int> *current_tokens;
  int pos;
  int current_size;
  int id;
  int *tokens;
  pthread_barrier_t *tp_barrier;

  int *pos_ptr;
  bool *finished_ptr;
  pthread_barrier_t *start_barrier;
  pthread_barrier_t *end_barrier;
} OnePathInsideArgs;

typedef struct {
  hipEvent_t *tp_ready;
  hipEvent_t *tp_finish;
  hipEvent_t *pp_sync;
} hipTotalEvents_t;

OurTransformerWeights *weights;
OurRunState *rs;

Config *public_config;
Transformer *public_transformer;
Tokenizer *public_tokenizer;
Sampler *public_sampler;
Requests *public_requests;

hipStream_t *total_streams;
hipTotalEvents_t *total_events;
