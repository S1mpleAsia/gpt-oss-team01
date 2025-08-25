#include "../include/alloc.hpp"
#include <cmath>
#include <cstring>

void our_init_weights(TransformerWeights *w, Config *p, OurTransformerWeights *weights) {
  // Create Tensor wrappers for weight matrices
  weights->token_embedding_table =
    new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->token_embedding_table);

  weights->rms_attn_w = new Tensor({(size_t)p->n_layers * p->hidden_dim}, w->rms_attn_w);
  weights->rms_ffn_w = new Tensor({(size_t)p->n_layers * p->hidden_dim}, w->rms_ffn_w);

  weights->w_qkv =
    new Tensor({(size_t)p->n_layers,
                ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * (size_t)p->head_dim,
                (size_t)p->hidden_dim},
               w->w_qkv);
  weights->b_qkv = new Tensor(
    {(size_t)p->n_layers, ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim},
    w->b_qkv);

  weights->w_o = new Tensor(
    {(size_t)p->n_layers, (size_t)p->hidden_dim, (size_t)p->n_attn_heads * p->head_dim}, w->w_o);
  weights->b_o = new Tensor({(size_t)p->n_layers, (size_t)p->hidden_dim}, w->b_o);

  // Tensor *attn_sinks; // (n_layers, n_attn_heads)
  weights->attn_sinks = new Tensor({(size_t)p->n_layers, (size_t)p->n_attn_heads}, w->attn_sinks);

  weights->w_router =
    new Tensor({(size_t)p->n_layers, (size_t)p->n_experts, (size_t)p->hidden_dim}, w->w_router);
  weights->b_router = new Tensor({(size_t)p->n_layers, (size_t)p->n_experts}, w->b_router);

  weights->w_mlp1 = new Tensor({(size_t)p->n_layers, (size_t)p->n_experts,
                                2 * (size_t)p->intermediate_dim, (size_t)p->hidden_dim},
                               w->w_mlp1, DType::BF16);
  weights->b_mlp1 =
    new Tensor({(size_t)p->n_layers, (size_t)p->n_experts, 2 * (size_t)p->intermediate_dim},
               w->b_mlp1, DType::BF16);

  weights->w_mlp2 = new Tensor(
    {(size_t)p->n_layers, (size_t)p->n_experts, (size_t)p->hidden_dim, (size_t)p->intermediate_dim},
    w->w_mlp2, DType::BF16);
  weights->b_mlp2 = new Tensor({(size_t)p->n_layers, (size_t)p->n_experts, (size_t)p->hidden_dim},
                               w->b_mlp2, DType::BF16);

  weights->rms_out_w = new Tensor({(size_t)p->hidden_dim}, w->rms_out_w);
  weights->out = new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->out);
}

void our_init_run_state(RunState *s, Config *p, OurRunState *rs) {
  // Create Tensor wrappers for state buffers
  rs->x = new Tensor({BATCH_SIZE * (size_t)p->hidden_dim}, s->x);

  rs->t = new Tensor({BATCH_SIZE * (size_t)p->hidden_dim}, s->t);
  rs->tb = new Tensor({BATCH_SIZE * (size_t)p->head_dim * p->n_attn_heads}, s->tb);
  rs->tb2 = new Tensor({BATCH_SIZE * (size_t)p->hidden_dim}, s->tb2);

  rs->tb3 = new Tensor({BATCH_SIZE * (size_t)p->experts_per_token, (size_t)p->hidden_dim});

  rs->router_score = new Tensor({BATCH_SIZE * (size_t)p->n_experts}, s->router_score);
  rs->topk_v = new Tensor({BATCH_SIZE * (size_t)p->experts_per_token}, s->topk_v);
  rs->topk_i = new TensorI32({BATCH_SIZE * (size_t)p->experts_per_token}, s->topk_i);

  // rs->mlp1_out = new Tensor({2 * (size_t)p->intermediate_dim}, s->mlp1_out);
  rs->mlp1_out =
    new Tensor({BATCH_SIZE * (size_t)p->experts_per_token, 2 * (size_t)p->intermediate_dim});

  // rs->gate = new Tensor({(size_t)p->intermediate_dim}, s->gate);
  // rs->up = new Tensor({(size_t)p->intermediate_dim}, s->up);
  rs->gate = new Tensor({BATCH_SIZE * (size_t)p->experts_per_token, (size_t)p->intermediate_dim});
  rs->up = new Tensor({BATCH_SIZE * (size_t)p->experts_per_token, (size_t)p->intermediate_dim});

  // rs->gate_up = new Tensor({(size_t)p->intermediate_dim}, s->gate_up);
  rs->gate_up =
    new Tensor({BATCH_SIZE * (size_t)p->experts_per_token, (size_t)p->intermediate_dim});

  rs->e_agg = new Tensor({BATCH_SIZE * (size_t)p->hidden_dim}, s->e_agg);
  rs->qkv = new Tensor(
    {BATCH_SIZE * ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim}, s->qkv);
  rs->q = new Tensor({BATCH_SIZE * (size_t)p->n_attn_heads * p->head_dim}, s->q);
  rs->k = new Tensor({BATCH_SIZE * (size_t)p->n_kv_heads * p->head_dim});
  rs->v = new Tensor({BATCH_SIZE * (size_t)p->n_kv_heads * p->head_dim});
  rs->att = new Tensor({BATCH_SIZE * (size_t)p->n_attn_heads, (size_t)p->seq_len}, s->att);
  rs->logits = new Tensor({BATCH_SIZE * (size_t)p->vocab_size}, s->logits);

  rs->key_cache = new Tensor(
    {BATCH_SIZE * (size_t)p->n_layers, (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim},
    s->key_cache);
  rs->value_cache = new Tensor(
    {BATCH_SIZE * (size_t)p->n_layers, (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim},
    s->value_cache);

  rs->mask = new Tensor({BATCH_SIZE * (size_t)p->seq_len, (size_t)p->seq_len}, s->mask);
}

void our_init(Transformer *transformer, OurTransformerWeights *weights, OurRunState *rs) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  our_init_weights(w, p, weights);
  our_init_run_state(s, p, rs);

  cos_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2});
  sin_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2});

  RopePrecomputeCS(p, cos_tensor, sin_tensor);
}

void our_free(OurTransformerWeights *weights, OurRunState *rs) {
  // weights
  delete weights->token_embedding_table;
  delete weights->rms_attn_w;
  delete weights->rms_ffn_w;
  delete weights->w_qkv;
  delete weights->w_o;
  delete weights->b_qkv;
  delete weights->b_o;
  delete weights->attn_sinks;
  delete weights->w_router;
  delete weights->b_router;
  delete weights->w_mlp1;
  delete weights->w_mlp2;
  delete weights->b_mlp1;
  delete weights->b_mlp2;
  delete weights->rms_out_w;
  delete weights->out;

  // delete rs
  delete rs->x;
  delete rs->t;
  delete rs->tb;
  delete rs->tb2;
  delete rs->router_score;
  delete rs->topk_v;
  delete rs->topk_i;
  delete rs->mlp1_out;
  delete rs->gate;
  delete rs->up;
  delete rs->gate_up;
  delete rs->e_agg;
  delete rs->qkv;
  delete rs->q;
  delete rs->k;
  delete rs->v;
  delete rs->att;
  delete rs->logits;
  delete rs->key_cache;
  delete rs->value_cache;
  delete rs->mask;

  // Others
  delete cos_tensor;
  delete sin_tensor;
}
