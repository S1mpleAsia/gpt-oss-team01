#pragma once

#include "tensor.hpp"

// Embedding lookup: out.shape = [hidden_dim]
void EmbeddingLookup(const Tensor *embedding /*[vocab, hidden]*/, int token_id,
                     Tensor *out /*[hidden]*/);

// RMSNorm: out = scale * x / rms(x)
void RMSNorm(const Tensor *x /*[hidden]*/, const Tensor *scale /*[hidden]*/,
             Tensor *out /*[hidden]*/, long long layer_offset, float eps = 1e-5f);

// y = Wx + b -- W: [out, in], x: [in], y: [out]
void Linear(const float *x, size_t in_dim, const float *W, const float *b, 
            size_t out_dim, float *y);

void Softmax(float *x, size_t size);

// QKV projection: qkv = W_qkv * x + b_qkv
// qkv.shape = [(n_q + 2*n_kv)*head_dim]
void QKVProject(const Tensor *x /*[hidden]*/,
                const Tensor *W_qkv /*[(n_q+2*n_kv)*hd, hidden]*/,
                const Tensor *b_qkv, Tensor *qkv /*[(n_q+2*n_kv)*hd]*/,
                long long layer_offset);

void SplitQKV(const Tensor *qkv, int head_dim, int n_q, int n_kv,
              Tensor *q /*[n_q*hd]*/, Tensor *k /*[n_kv*hd]*/,
              Tensor *v /*[n_kv*hd]*/);

// cos/sin computation for RoPE at pos
void RopePrecomputeCS(Config *p, Tensor *cos_all_out,
                    Tensor *sin_all_out, hipStream_t stream = 0);

// RoPE (n_heads * head_dim)
void ApplyRotary(Tensor *x /*[n_heads*hd]*/,
                 const Tensor *cos_table /*[seq_len*hd/2]*/,
                 const Tensor *sin_table /*[seq_len*hd/2]*/,
                 int n_heads, int head_dim, int pos);

// Attention scores for 1 head: att[0..pos] = q·k_t / sqrt(hd) (+ mask)
void AttnScoresAllHeads(Tensor *key_cache, Tensor *q, Tensor *att,
                        Tensor *attn_sinks, Tensor *mask, long long loff_one,
                        long long layer_offset, int attn_heads, int kv_mul,
                        int head_dim, int kv_dim, int seq_len,
                        int sliding_window, int pos);

// Weighted sum for 1 head
void AttnWeightedSumAllHeads(Tensor *value_cache, Tensor *q, Tensor *att,
                            Tensor *tb, long long loff, int attn_heads,
                            int kv_mul, int head_dim, int kv_dim,
                            int seq_len, int pos);

// Output projection for attention: y = W_o * tb + b_o
void AttnOutProject(const Tensor *tb /*[n_q*hd]*/,
                    const Tensor *W_o /*[hidden, n_q*hd]*/, const Tensor *b_o,
                    Tensor *y /*[hidden]*/, long long layer_offset);

// Router: r = W_router * t + b_router
void RouterScores(const Tensor *t /*[hidden]*/,
                  const Tensor *W_router /*[n_experts, hidden]*/,
                  const Tensor *b_router, Tensor *r /*[n_experts]*/,
                  long long layer_offset);

void TopK(const Tensor *r /*[n_experts]*/, int k, Tensor *topk_vals /*[k]*/,
          TensorI32 *topk_idx /*[k]*/);

// SwiGLU (clamp + silu(gate) * (up+1))
void SwiGLU(const Tensor *gate /*[d]*/, const Tensor *up /*[d]*/,
            float clamp_limit, Tensor *out /*[d]*/);

// Expert FFN 1: z = W1 * t + b1
void ExpertFFN1_Total(const Tensor *t, const Tensor *W1, const Tensor *b1,      
                    Tensor *z, const TensorI32 *topk_i, long long layer_offset);

// Expert FFN 2: y = W2 * swiglu + b2
void ExpertFFN2_Total(const Tensor *swiglu, const Tensor *W2, const Tensor *b2,
                    Tensor *y, const TensorI32 *topk_i, long long layer_offset);

// Cộng residual: x += y
void ResidualAdd(Tensor *x /*[hidden]*/, const Tensor *y /*[hidden]*/);

// Classifier / unembed: logits = W_out * x (no bias), W_out: [vocab, hidden]
void Classifier(const Tensor *x /*[hidden]*/,
                const Tensor *W_out /*[vocab, hidden]*/,
                Tensor *logits /*[vocab]*/);