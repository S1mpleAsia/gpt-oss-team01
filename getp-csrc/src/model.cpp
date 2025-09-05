#include "../include/model.hpp"
#include <cmath>
#include <cstring>

float *forward_gpu_20b_batched(Config *p, OurTransformerWeights *weights_total,
                               OurRunState *rs_total, int *tokens, int pos, int cur_batch_size,
                               int flow_id) {
  OurTransformerWeights *weights = &weights_total[flow_id];
  OurRunState *rs = &rs_total[flow_id];

  // copy the token embedding into x
  embedding_lookup_batched(weights->token_embedding_table, tokens, rs->x, false);

  long long kv_dim = 1ll * p->n_kv_heads * p->head_dim;
  long long loff_one = 1ll * p->seq_len * kv_dim;
  long long loff_one_batch = 1ll * p->n_layers * loff_one;

  // forward all the layers
  for (int l = 0; l < p->n_layers; l++) {
    // attention rmsnorm
    rmsnorm_batched(rs->x, weights->rms_attn_w, rs->t, 1ll * l, false, false);

    // key and value point to the kv cache
    long long loff = 1ll * l * loff_one;  // kv cache layer offset

    // QKV projection
    qkv_gemm_batched_v2(rs->t, weights->w_qkv, weights->b_qkv, rs->qkv, 1ll * l, false,
                        false);  // This kernel diverges the most

    // // Separate q, k, v + RoPE
    // qkv_split_rope_batched(rs->qkv, rs->q, rs->k, rs->v, rs->cos_tensor, rs->sin_tensor,
    //                        p->head_dim, p->n_attn_heads, p->n_kv_heads, pos, false, false, false,
    //                        false);

    // // Store k, v in cache
    // for (int b = 0; b < cur_batch_size; b++) {
    //   memcpy_tensor(rs->key_cache, rs->k, 1ll * b * loff_one_batch + loff + 1ll * pos * kv_dim,
    //                 1ll * b * kv_dim, kv_dim, false, true);
    //   memcpy_tensor(rs->value_cache, rs->v, 1ll * b * loff_one_batch + loff + 1ll * pos * kv_dim,
    //                 1ll * b * kv_dim, kv_dim, false, true);
    // }

    qkv_split_rope_fused(rs->qkv, rs->q, rs->key_cache, rs->value_cache, rs->cos_tensor,
                         rs->sin_tensor, p->head_dim, p->n_attn_heads, p->n_kv_heads, pos, l);

    // multihead attention
    int kv_mul = p->n_attn_heads / p->n_kv_heads;  // integer multiplier for GQA

    // FIX single_query_attn_batched later
    single_query_attn_flash_batched(rs->q, rs->key_cache, rs->value_cache, rs->mask,
                                    weights->attn_sinks, rs->tb, p->head_dim, p->n_attn_heads,
                                    kv_mul, kv_dim, p->seq_len, p->sliding_window, pos, 1ll * l,
                                    false, false, false, false, false);

    // final matmul to get the output of the attention
    attn_out_project_batched_v2(rs->tb, weights->w_o, weights->b_o, rs->tb2, 1ll * l, false, false);

    // residual connection back into x
    // add_vector_batched(rs->x, rs->tb2, false, false, false);  // equals residual add

    // ffn rmsnorm
    // rmsnorm_batched(rs->x, weights->rms_ffn_w, rs->t, 1ll * l, false, false);

    residual_rmsnorm_batched(rs->tb2, rs->x, weights->rms_ffn_w, rs->t, 1ll * l, 1e-5f);

    // MoE routing
    router_gemm_batched(weights->w_router, rs->t, weights->b_router, rs->router_score, 1ll * l,
                        false, false);

    // Select top-k experts
    topk_softmax_batched(rs->router_score, rs->topk_v, rs->topk_i, false, false, false);

    // Route the tokens to their corresponding top-k experts
    // moe_apply_topk_batched(rs->t, weights->w_mlp1, weights->b_mlp1, weights->w_mlp2,
    //                        weights->b_mlp2, rs->topk_i, rs->topk_v, rs->mlp1_out, rs->gate_up,
    //                        rs->tb3, rs->e_agg, p->swiglu_limit, 1ll * l, false, false, false,
    //                        false);

    // moe_mlp1_batched(rs->t, weights->w_mlp1, weights->b_mlp1, rs->topk_i, rs->mlp1_out, false,
    //                  false, 1ll * l);

    // moe_swiglu_batched(rs->mlp1_out, rs->gate_up, p->experts_per_token, p->swiglu_limit);

    // moe_mlp2_batched(rs->gate_up, weights->w_mlp2, weights->b_mlp2, rs->tb3, rs->topk_i,
    //                  p->experts_per_token, true, 1ll * l);

    // moe_agg_batched(rs->tb3, rs->topk_v, rs->e_agg, p->experts_per_token, false);

    int total_pairs = BATCH_SIZE * p->experts_per_token;
    moe_init_buffers_hip(rs->e_agg, rs->mlp1_out, rs->gate_up, rs->tb3, rs->sorted_pair_ids,
                         rs->expert_offsets, rs->x_packed, BATCH_SIZE, p->hidden_dim, 0);

    moe_build_offsets_hip(rs->topk_i, rs->sorted_pair_ids, rs->expert_offsets, BATCH_SIZE,
                          p->experts_per_token, p->n_experts, 0);

    moe_pack_inputs_hip(rs->t, rs->sorted_pair_ids, rs->x_packed, BATCH_SIZE, p->hidden_dim,
                        p->experts_per_token, 0);

    int max_rows = moe_get_max_rows_per_expert_hip(rs->expert_offsets, p->n_experts, 0);
    moe_mlp1_forward_hip(rs->x_packed, weights->w_mlp1, weights->b_mlp1, rs->expert_offsets,
                         rs->mlp1_out, 1ll * l, p->n_experts, p->hidden_dim, p->intermediate_dim,
                         max_rows, total_pairs, 0);

    moe_swiglu_hip(rs->mlp1_out, rs->gate_up, BATCH_SIZE, p->experts_per_token, p->intermediate_dim,
                   p->swiglu_limit, 0);

    moe_mlp2_forward_hip(rs->gate_up, weights->w_mlp2, weights->b_mlp2, rs->expert_offsets, rs->tb3,
                         true, 1ll * l, p->n_experts, p->intermediate_dim, p->hidden_dim, max_rows,
                         total_pairs, 0);

    moe_scatter_aggregate_hip(rs->tb3, rs->sorted_pair_ids, rs->topk_v, rs->e_agg,
                              rs->expert_offsets, p->hidden_dim, p->experts_per_token, p->n_experts,
                              max_rows, 0);

    // moe_block_matmul_style_hip(rs->t, rs->topk_i, rs->topk_v, weights->w_mlp1, weights->b_mlp1,
    //                            weights->w_mlp2, weights->b_mlp2, rs->e_agg, rs->mlp1_out,
    //                            rs->gate_up, rs->tb3, rs->sorted_pair_ids, rs->expert_offsets,
    //                            rs->x_packed, p->swiglu_limit, 1ll * l);

    // residual connection
    add_vector_batched(rs->x, rs->e_agg, false, false, false);  // equals residual add
  }

  // final rmsnorm
  rmsnorm_batched(rs->x, weights->rms_out_w, rs->x, 0ll, false, false);

  // classifier into logits
  classifier_gemm_batched_v2(weights->out, rs->x, rs->logits, false, true);
  return rs->logits->buf;
}

float *forward_gpu_20b(Config *p, OurTransformerWeights *weights, OurRunState *rs, int token,
                       int pos) {
  // copy the token embedding into x
  embedding_lookup(weights->token_embedding_table, token, rs->x, false);

  long long loff_one = 1ll * p->seq_len * p->head_dim * p->n_kv_heads;

  // forward all the layers
  for (int l = 0; l < p->n_layers; l++) {
    // printf("==== stage: %d - layer: %d ====\n", (l < 12) ? 0 : 1, l);

    // attention rmsnorm
    rmsnorm(rs->x, weights->rms_attn_w, rs->t, 1ll * l, false, false);

    // key and value point to the kv cache
    long long loff = 1ll * l * loff_one;  // kv cache layer offset

    // QKV projection
    qkv_gemm(rs->t, weights->w_qkv, weights->b_qkv, rs->qkv, 1ll * l, false,
             false);  // This kernel diverges the most

    // Separate q, k, v + RoPE
    qkv_split_rope(rs->qkv, rs->q, rs->k, rs->v, rs->cos_tensor, rs->sin_tensor, p->head_dim,
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

    if (l == 12) {
      int topk_vals[4];
      CHECK_HIP(hipMemcpy(topk_vals, rs->topk_i->d_buf, 4 * sizeof(int), hipMemcpyDeviceToHost));
      printf("Layer 12 topk_i: %d %d %d %d\n", topk_vals[0], topk_vals[1], topk_vals[2],
             topk_vals[3]);

      float topk_weights[4];
      CHECK_HIP(
        hipMemcpy(topk_weights, rs->topk_v->d_buf, 4 * sizeof(float), hipMemcpyDeviceToHost));
      printf("Layer 12 topk_v: %.6f %.6f %.6f %.6f\n", topk_weights[0], topk_weights[1],
             topk_weights[2], topk_weights[3]);
    }

    // Route the tokens to their corresponding top-k experts
    const int hidden_dim = rs->t->shape[1];
    const int inter_dim = weights->w_mlp2->shape[3];
    const int k = rs->topk_i->num_elem();
    const int num_experts = weights->b_mlp2->shape[1];
    const long long layer_offset = (long long)l;
    const long long offset = layer_offset * num_experts;
    const long long inter_hidden = (long long)inter_dim * hidden_dim;

    // Lấy con trỏ thô trên device từ các tensor
    const float *t_ptr = (const float *)rs->t->d_buf;
    const bf16 *W1_ptr = (const bf16 *)weights->w_mlp1->d_buf + offset * 2 * inter_hidden;
    const bf16 *b1_ptr = (const bf16 *)weights->b_mlp1->d_buf + offset * 2 * inter_dim;
    const bf16 *W2_ptr = (const bf16 *)weights->w_mlp2->d_buf + offset * inter_hidden;
    const bf16 *b2_ptr = (const bf16 *)weights->b_mlp2->d_buf + offset * hidden_dim;
    const int *topk_idx_ptr = rs->topk_i->d_buf;
    const float *topk_vals_ptr = (const float *)rs->topk_v->d_buf;
    float *mlp1_out_ptr = (float *)rs->mlp1_out->d_buf;
    float *tb3_ptr = (float *)rs->tb3->d_buf;
    float *gate_up_ptr = (float *)rs->gate_up->d_buf;
    float *e_agg_ptr = (float *)rs->e_agg->d_buf;

    memset_tensor(rs->e_agg, 0, false, true, 0);

    moe_mlp1(W1_ptr, t_ptr, b1_ptr, mlp1_out_ptr, topk_idx_ptr, k, inter_dim, hidden_dim, 0);

    moe_swiglu(mlp1_out_ptr, gate_up_ptr, k, inter_dim, p->swiglu_limit, 0);

    moe_mlp2(W2_ptr, gate_up_ptr, b2_ptr, tb3_ptr, topk_idx_ptr, k, hidden_dim, inter_dim, 0);

    moe_agg(tb3_ptr, topk_vals_ptr, e_agg_ptr, k, hidden_dim, 0);

    // moe_apply_topk(rs->t, weights->w_mlp1, weights->b_mlp1, weights->w_mlp2, weights->b_mlp2,
    //                rs->topk_i, rs->topk_v, rs->mlp1_out, rs->gate_up, rs->tb3, rs->e_agg,
    //                p->swiglu_limit, 1ll * l, false, false, false, false);

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
    ApplyRotary(rs->q, rs->cos_tensor, rs->sin_tensor, p->n_attn_heads, p->head_dim, pos);
    ApplyRotary(rs->k, rs->cos_tensor, rs->sin_tensor, p->n_kv_heads, p->head_dim, pos);

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
