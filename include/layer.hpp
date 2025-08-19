#pragma once

#include "tensor.hpp"

// =============== Configuration =====================
typedef struct {
  // Model Config
  int vocab_size;  // vocabulary size
  int hidden_dim;  // model dim
  // MLP Config
  int n_experts;          // number of experts
  int experts_per_token;  // num top-k
  int intermediate_dim;   // for ffn layers
  int n_layers;           // num hidden layers
  // Attention Config
  int head_dim;                // head dimension
  int n_attn_heads;            // number of query heads
  int n_kv_heads;              // number of key/value heads (can be < query heads because of
                               // MQA)
  int seq_len;                 // max sequence length e.g., 1024
  int initial_context_length;  // e.g., 4096
  float rope_theta;            // rope theta e.g., 150000.0
  float rope_scaling_factor;   // e.g., 32.0
  int sliding_window;          // e.g., 128
  float swiglu_limit;          // e.g., 7.0
} Config;

typedef struct {
  // token_embedding_table - embedding.weight
  float *token_embedding_table;  // (vocab_size, hidden_dim) (in, out)
  // weights for rmsnorms
  float *rms_attn_w;  // (n_layers, hidden_dim) [attn.norm.scale]
  float *rms_ffn_w;   // (n_layers, hidden_dim) [mlp.norm.scale]
  // weights for attention [attn.qkv.weight & attn.qkv.bias]
  float *w_qkv;       // (n_layers, head_dim * n_attn_heads + 2 * head_dim * n_kv_heads,
                      // hidden_dim) where w_q (head_dim * n_attn_heads, hidden_dim)
                      // (out_features, in_features) w_k (head_dim * n_kv_heads,
                      // hidden_dim)  (out_features, in_features) w_v (head_dim *
                      // n_kv_heads, hidden_dim)  (out_features, in_features)
  float *w_o;         // (n_layers, hidden_dim, head_dim * n_attn_heads)
  float *b_qkv;       // (n_layers, head_dim * n_attn_heads + 2 * head_dim *
                      // n_kv_heads) (head_dim * n_attn_heads) (head_dim * n_kv_heads)
                      // (head_dim * n_kv_heads)
  float *b_o;         // (n_layers, hidden_dim)
  float *attn_sinks;  // (n_layers, n_attn_heads)
  // weights for router [mlp.gate.weight & mlp.gate.bias]
  float *w_router;  // (n_layers, hidden_dim, n_experts)
  float *b_router;  // (n_layers, n_experts)
  // weights for MoE [mlp.mlp1_weight & mlp.mlp1_bias & mlp.mlp2_weight &
  // mlp.mlp2_bias] NOTE: gate_up projects from hidden_dim to intermediate_dim,
  // the shape is kinda reverted because the original code use einsum to reduce
  // over hidden_dim
  float *w_mlp1;  // gate_up_proj (n_layers, n_experts, 2 * intermediate_dim,
                  // hidden_dim)
  float *w_mlp2;  // down_proj (n_layers, n_experts, hidden_dim, intermediate_dim)
  float *b_mlp1;  // gate_up proj (n_layers, n_experts, 2 * intermediate_dim)
  float *b_mlp2;  // down_proj (n_layers, n_experts, hidden_dim)
  // final norm [norm.scale]
  float *rms_out_w;  // (hidden_dim, )
  // classifier weights for the logits [unembedding.weight]
  float *out;  // (vocab_size, hidden_dim) (out, in)
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x;             // activation at current time stamp (hidden_dim, )
  float *t;             // same, but inside a residual branch (hidden_dim, )
  float *tb;            // (head_dim * n_attn_heads, )
  float *tb2;           // (hidden_dim, )
  float *router_score;  // router score (n_experts, )
  float *topk_v;        // topk expert weights (experts_per_token, )
  int *topk_i;          // topk expert indices (experts_per_token, )
  float *mlp1_out;
  float *gate;
  float *up;
  float *gate_up;
  float *e_agg;
  float *qkv;     // an additional buffer just for convenience (head_dim *
                  // (n_attn_heads + 2 * n_kv_heads), )
  float *q;       // query (n_attn_heads * head_dim,)
  float *k;       // key (n_kv_heads * head_dim,)
  float *v;       // value (n_kv_heads * head_dim,)
  float *att;     // buffer for scores/attention values (n_heads, seq_len)
  float *logits;  // output logits
  // kv cache
  float *key_cache;    // (layer, seq_len, kv_dim)
  float *value_cache;  // (layer, seq_len, kv_dim)
  float *mask;
} RunState;

typedef struct {
  Config config;
  TransformerWeights weights;
  RunState state;     // buffers for the "wave" of activations in the forward pass
  int fd;             // file descriptor for memory mapping
  float *data;        // memory mapped data pointer
  ssize_t file_size;  // size of the checkpoint file in bytes
} Transformer;

// Embedding lookup: out.shape = [hidden_dim]
void EmbeddingLookup(const Tensor &embedding /*[vocab, hidden]*/,
                     int token_id,
                     Tensor &out /*[hidden]*/);

// RMSNorm: out = scale * x / rms(x)
void RMSNorm(const Tensor &x /*[hidden]*/,
             const Tensor &scale /*[hidden]*/,
             Tensor &out /*[hidden]*/,
             float eps = 1e-5f);

// y = Wx + b -- W: [out, in], x: [in], y: [out]
void Linear(const Tensor &x, const Tensor &W, Tensor *b, Tensor &y);

void Softmax(Tensor &x);

// QKV projection: qkv = W_qkv * x + b_qkv
// qkv.shape = [(n_q + 2*n_kv)*head_dim]
void QKVProject(const Tensor &x /*[hidden]*/,
                const Tensor &W_qkv /*[(n_q+2*n_kv)*hd, hidden]*/,
                const Tensor *b_qkv,
                Tensor &qkv /*[(n_q+2*n_kv)*hd]*/);

void SplitQKV(const Tensor &qkv,
              int head_dim,
              int n_q,
              int n_kv,
              Tensor &q /*[n_q*hd]*/,
              Tensor &k /*[n_kv*hd]*/,
              Tensor &v /*[n_kv*hd]*/);

// cos/sin computation for RoPE at pos
void RopeComputeCS(int pos,
                   const Config &p,
                   Tensor &cos_out /*[head_dim/2]*/,
                   Tensor &sin_out /*[head_dim/2]*/);

// RoPE (n_heads * head_dim)
void ApplyRotary(Tensor &x /*[n_heads*hd]*/,
                 const Tensor &cos /*[hd/2]*/,
                 const Tensor &sin /*[hd/2]*/,
                 int n_heads,
                 int head_dim);

// Attention scores for 1 head: att[0..pos] = q·k_t / sqrt(hd) (+ mask)
void AttnScoresOneHead(const float *q /*[hd]*/,
                       const float *k_cache_layer /*[seq_len*kv_dim]*/,
                       int kv_offset_bytes,
                       int head_dim,
                       int seq_len,
                       int pos,
                       const float *mask_row /*[seq_len] or nullptr*/,
                       float *att /*[pos+1]*/);

// Weighted sum for 1 head
void AttnWeightedSumOneHead(const float *att /*[pos+1]*/,
                            const float *v_cache_layer /*[seq_len*kv_dim]*/,
                            int kv_offset_bytes,
                            int head_dim,
                            int pos,
                            float *tb /*[hd]*/);

// Output projection for attention: y = W_o * tb + b_o
void AttnOutProject(const Tensor &tb /*[n_q*hd]*/,
                    const Tensor &W_o /*[hidden, n_q*hd]*/,
                    const Tensor *b_o,
                    Tensor &y /*[hidden]*/);

// Router: r = W_router * t + b_router
void RouterScores(const Tensor &t /*[hidden]*/,
                  const Tensor &W_router /*[n_experts, hidden]*/,
                  const Tensor *b_router,
                  Tensor &r /*[n_experts]*/) {}

void TopK(const Tensor &r /*[n_experts]*/,
          int k,
          Tensor &topk_vals /*[k]*/,
          Tensor &topk_idx /*[k]*/);

// SwiGLU (clamp + silu(gate) * (up+1))
void SwiGLU(const Tensor &gate /*[d]*/,
            const Tensor &up /*[d]*/,
            float clamp_limit,
            Tensor &out /*[d]*/);

// Expert FFN 1: z = W1 * t + b1
void ExpertFFN1(const Tensor &t, const Tensor &W1, const Tensor *b1, Tensor &z);

// Expert FFN 2: y = W2 * swiglu + b2
void ExpertFFN2(const Tensor &swiglu, const Tensor &W2, const Tensor *b2, Tensor &y);

// Cộng residual: x += y
void ResidualAdd(Tensor &x /*[hidden]*/, const Tensor &y /*[hidden]*/);

// Classifier / unembed: logits = W_out * x (no bias), W_out: [vocab, hidden]
void Classifier(const Tensor &x /*[hidden]*/,
                const Tensor &W_out /*[vocab, hidden]*/,
                Tensor &logits /*[vocab]*/);