#include "../include/model.hpp"
#include <cmath>
#include <cstring>

float *forward_gpu_20b_batched(Config *p, OurTransformerWeights *weights, OurRunState *rs,
                               int *tokens, int pos, int batch_size) {
  return forward_gpu_20b(p, weights, rs, tokens[0], pos);
}

float *forward_gpu_20b(Config *p, OurTransformerWeights *weights, OurRunState *rs, int token,
                       int pos) {
  // copy the token embedding into x
  embedding_lookup(weights->token_embedding_table, token, rs->x, false);

  long long loff_one = 1ll * p->seq_len * p->head_dim * p->n_kv_heads;

  // forward all the layers
  for (int l = 0; l < p->n_layers; l++) {
    // attention rmsnorm
    rmsnorm(rs->x, weights->rms_attn_w, rs->t, 1ll * l, false, false);

    // key and value point to the kv cache
    long long loff = 1ll * l * loff_one;  // kv cache layer offset

    // QKV projection
    qkv_gemm(rs->t, weights->w_qkv, weights->b_qkv, rs->qkv, 1ll * l, false,
             false);  // This kernel diverges the most

    // Separate q, k, v + RoPE
    qkv_split_rope(rs->qkv, rs->q, rs->k, rs->v, cos_tensor, sin_tensor, p->head_dim,
                   p->n_attn_heads, p->n_kv_heads, pos, false, false, false, false);

    // Store k, v in cache
    memcpy_tensor(rs->key_cache, rs->k, loff + 1ll * pos * p->n_kv_heads * p->head_dim, 0,
                  (size_t)p->n_kv_heads * p->head_dim, false, true);
    memcpy_tensor(rs->value_cache, rs->v, loff + 1ll * pos * p->n_kv_heads * p->head_dim, 0,
                  (size_t)p->n_kv_heads * p->head_dim, false, true);

    // multihead attention
    int kv_mul = p->n_attn_heads / p->n_kv_heads;  // integer multiplier for GQA

    single_query_attn(rs->q, rs->key_cache, rs->value_cache, rs->mask, weights->attn_sinks, rs->tb,
                      p->head_dim, p->n_attn_heads, kv_mul, p->head_dim * p->n_kv_heads, p->seq_len,
                      p->sliding_window, pos, 1ll * l, false, false, false, false, false);

    // final matmul to get the output of the attention
    attn_out_project(rs->tb, weights->w_o, weights->b_o, rs->tb2, 1ll * l, false, false);

    // residual connection back into x
    add_vector(rs->x, rs->tb2, false, false, false);  // equals residual add

    // ffn rmsnorm
    rmsnorm(rs->x, weights->rms_ffn_w, rs->t, 1ll * l, false, false);

    // MoE routing
    router_gemm(weights->w_router, rs->t, weights->b_router, rs->router_score, 1ll * l, false,
                false);

    // Select top-k experts
    topk_softmax(rs->router_score, rs->topk_v, rs->topk_i, false, false, false);

    // Route the tokens to their corresponding top-k experts
    moe_apply_topk(rs->t, weights->w_mlp1, weights->b_mlp1, weights->w_mlp2, weights->b_mlp2,
                   rs->topk_i, rs->topk_v, rs->mlp1_out, rs->gate_up, rs->tb3, rs->e_agg,
                   p->swiglu_limit, 1ll * l, false, false, false, false);

    // residual connection
    add_vector(rs->x, rs->e_agg, false, false, false);  // equals residual add
  }

  // final rmsnorm
  rmsnorm(rs->x, weights->rms_out_w, rs->x, 0ll, false, false);

  // classifier into logits
  classifier_gemm(weights->out, rs->x, rs->logits, false, true);

  return rs->logits->buf;
}

float *forward_cpu_20b(Config *p, OurTransformerWeights *weights, OurRunState *rs, int token,
                       int pos) {
  // copy the token embedding into x
  EmbeddingLookup(weights->token_embedding_table, token, rs->x);

  long long loff_one = 1ll * p->seq_len * p->head_dim * p->n_kv_heads;

  // forward all the layers
  for (int l = 0; l < p->n_layers; l++) {
    // attention rmsnorm
    RMSNorm(rs->x, weights->rms_attn_w, rs->t, 1ll * l);

    // key and value point to the kv cache
    long long loff = 1ll * l * loff_one;  // kv cache layer offset

    // QKV projection
    QKVProject(rs->t, weights->w_qkv, weights->b_qkv, rs->qkv, 1ll * l);

    // Separate q, k, v + apply RoPE
    SplitQKV(rs->qkv, p->head_dim, p->n_attn_heads, p->n_kv_heads, rs->q, rs->k, rs->v);
    ApplyRotary(rs->q, cos_tensor, sin_tensor, p->n_attn_heads, p->head_dim, pos);
    ApplyRotary(rs->k, cos_tensor, sin_tensor, p->n_kv_heads, p->head_dim, pos);

    // Store k, v in cache
    memcpy(rs->key_cache->buf + loff + 1ll * pos * p->n_kv_heads * p->head_dim, rs->k->buf,
           p->n_kv_heads * p->head_dim * sizeof(float));
    memcpy(rs->value_cache->buf + loff + 1ll * pos * p->n_kv_heads * p->head_dim, rs->v->buf,
           p->n_kv_heads * p->head_dim * sizeof(float));

    // multihead attention
    int kv_mul = p->n_attn_heads / p->n_kv_heads;  // integer multiplier for GQA

    AttnScoresAllHeads(rs->key_cache, rs->q, rs->att, weights->attn_sinks, rs->mask, loff_one,
                       1ll * l, p->n_attn_heads, kv_mul, p->head_dim, p->head_dim * p->n_kv_heads,
                       p->seq_len, p->sliding_window, pos);

    AttnWeightedSumAllHeads(rs->value_cache, rs->q, rs->att, rs->tb, loff, p->n_attn_heads, kv_mul,
                            p->head_dim, p->head_dim * p->n_kv_heads, p->seq_len, pos);

    // final matmul to get the output of the attention
    AttnOutProject(rs->tb, weights->w_o, weights->b_o, rs->tb2, 1ll * l);

    // residual connection back into x
    ResidualAdd(rs->x, rs->tb2);

    // ffn rmsnorm
    RMSNorm(rs->x, weights->rms_ffn_w, rs->t, 1ll * l);

    // MoE routing
    RouterScores(rs->t, weights->w_router, weights->b_router, rs->router_score, 1ll * l);

    // Select top-k experts
    TopK(rs->router_score, p->experts_per_token, rs->topk_v, rs->topk_i);

    // Normalize selected experts using softmax
    Softmax(rs->topk_v->buf, rs->topk_v->num_elem());

    // Route the tokens to their corresponding top-k experts
    memset(rs->e_agg->buf, 0, p->hidden_dim * sizeof(float));

    ExpertFFN1_Total(rs->t, weights->w_mlp1, weights->b_mlp1, rs->mlp1_out, rs->topk_i, 1ll * l);

    for (int j = 0; j < p->experts_per_token * p->intermediate_dim; j++) {
      rs->gate->buf[j] = rs->mlp1_out->buf[2 * j];
      rs->up->buf[j] = rs->mlp1_out->buf[2 * j + 1];
    }

    SwiGLU(rs->gate, rs->up, p->swiglu_limit, rs->gate_up);

    ExpertFFN2_Total(rs->gate_up, weights->w_mlp2, weights->b_mlp2, rs->tb3, rs->topk_i, 1ll * l);

    for (int i = 0; i < p->hidden_dim; i++) {
      float sum_cur = 0.0f;
      for (int j = 0; j < p->experts_per_token; j++) {
        float expert_w = rs->topk_v->buf[j];
        sum_cur += rs->tb3->buf[j * p->hidden_dim + i] * expert_w;
      }
      rs->e_agg->buf[i] += sum_cur;
    }

    // residual connection
    ResidualAdd(rs->x, rs->e_agg);
  }

  // final rmsnorm
  RMSNorm(rs->x, weights->rms_out_w, rs->x, 0ll);

  // classifier into logits
  Classifier(rs->x, weights->out, rs->logits);

  return rs->logits->buf;
}
