#include "../include/model.hpp"
#include <cmath>
#include <cstring>

float *forward_gpu_120b_batched(
  int *tokens, int pos, int cur_batch_size, int flow_id,
  int tp_rank, int pp_rank, pthread_barrier_t *tp_barrier
) {
  Config *p = public_config;
  int cur_device = flow_id * TOTAL_PIPELINES + tp_rank;

  OurTransformerWeights *weights_now = &weights[cur_device];
  OurRunState *rs_now = &rs[cur_device];
  hipStream_t stream = total_streams[cur_device];
  hipEvent_t event = total_events[cur_device];
  // hipStream_t stream = 0;

  CHECK_HIP(hipSetDevice(cur_device));
  
  // copy the token embedding into x
  embedding_lookup_batched(weights_now->token_embedding_table, tokens, rs_now->x, false);

  long long kv_dim = 1ll * p->n_kv_heads * p->head_dim;
  long long loff_one = 1ll * p->seq_len * kv_dim;
  long long loff_one_batch = 1ll * (p->n_layers / PP) * loff_one;

  // forward all the layers
  for (int pipeline_id = 0; pipeline_id < PP; pipeline_id++) {
    if (pipeline_id) {
      CHECK_HIP(hipStreamSynchronize(0));
      
      cur_device += TP;
      CHECK_HIP(hipSetDevice(cur_device));

      OurRunState *rs_new = &rs[cur_device];

      // sync from rs_now->x to rs_now->x
      stream = total_streams[cur_device];
      CHECK_HIP(hipMemcpyPeerAsync(rs_new->x->d_buf, cur_device, rs_now->x->d_buf, cur_device - TP, rs_now->x->num_elem() * rs_now->x->get_dtype_size()));

      weights_now = &weights[cur_device];
      rs_now = &rs[cur_device];
      event = total_events[cur_device];
    }

    for (int l = 0; l < p->n_layers / PP; l++) {
      #ifdef DEBUG
        if (flow_id == 0) printf("tp_rank %d, layer %d\n", tp_rank, l);
      #endif

      // attention rmsnorm
      rmsnorm_batched(rs_now->x, weights_now->rms_attn_w, rs_now->t, 1ll * l,
                      false, false);
      
      #ifdef DEBUG
        if (flow_id == 0) rs_now->t->printDebug("rs_now->t", tp_rank);
      #endif

      // key and value point to the kv cache
      long long loff = 1ll * l * loff_one;  // kv cache layer offset

      // QKV projection
      qkv_gemm_batched(rs_now->t, weights_now->w_qkv, weights_now->b_qkv, rs_now->qkv, 1ll * l, false, false);  // This kernel diverges the most
      
      #ifdef DEBUG
        if (flow_id == 0) {
          rs_now->qkv->printDebug("rs_now->qkv", tp_rank);
          long long layer_offset_wqkv = 1ll * l * (weights_now->w_qkv->num_elem() / weights_now->w_qkv->shape[0]);
          weights_now->w_qkv->printDebug("weights_now->w_qkv", tp_rank, layer_offset_wqkv, false);
          long long layer_offset_bqkv = 1ll * l * (weights_now->b_qkv->num_elem() / weights_now->b_qkv->shape[0]);
          weights_now->b_qkv->printDebug("weights_now->b_qkv", tp_rank, layer_offset_bqkv, false);
        }
      #endif

      // Separate q, k, v + RoPE
      qkv_split_rope_batched(rs_now->qkv, rs_now->q, rs_now->k, rs_now->v, rs_now->cos_tensor,
                            rs_now->sin_tensor, p->head_dim, p->n_attn_heads,
                            p->n_kv_heads, pos, false, false, false, false);
      #ifdef DEBUG
        if (flow_id == 0) rs_now->q->printDebug("rs_now->q", tp_rank);
      #endif

      // Store k, v in cache
      for (int b = 0; b < cur_batch_size; b++) {
        memcpy_tensor(rs_now->key_cache, rs_now->k, 1ll * b * loff_one_batch + loff + 1ll * pos * kv_dim, 1ll * b * kv_dim, kv_dim, false, true);
        memcpy_tensor(rs_now->value_cache, rs_now->v, 1ll * b * loff_one_batch + loff + 1ll * pos * kv_dim, 1ll * b * kv_dim, kv_dim, false, true);
      }
      
      #ifdef DEBUG
        if (flow_id == 0) rs_now->key_cache->printDebug("rs_now->key_cache", tp_rank);
      #endif

      // multihead attention
      int kv_mul = p->n_attn_heads / p->n_kv_heads;  // integer multiplier for GQA

      // FIX single_query_attn_batched later
      single_query_attn_batched(rs_now->q, rs_now->key_cache, rs_now->value_cache, rs_now->mask, 
                                weights_now->attn_sinks, rs_now->tb, p->head_dim,
                                p->n_attn_heads, kv_mul, kv_dim, p->seq_len,
                                p->sliding_window, pos, 1ll * l, false, false,
                                false, false, false);
      #ifdef DEBUG
        if (flow_id == 0) rs_now->tb->printDebug("rs_now->tb", tp_rank);
      #endif
                                
      // final matmul to get the output of the attention
      attn_out_project_batched(rs_now->tb, weights_now->w_o, weights_now->b_o, rs_now->tb2, 1ll * l, false, false);

      #ifdef DEBUG
        if (flow_id == 0) rs_now->tb2->printDebug("rs_now->tb2", tp_rank);
      #endif

      // residual connection back into x
      add_vector_batched(rs_now->x, rs_now->tb2, false, false, false);  // equals residual add

      #ifdef DEBUG
        if (flow_id == 0) rs_now->x->printDebug("rs->x", tp_rank);
      #endif

      // ffn rmsnorm
      rmsnorm_batched(rs_now->x, weights_now->rms_ffn_w, rs_now->t, 1ll * l, false, false);
      
      #ifdef DEBUG
        if (flow_id == 0) rs_now->t->printDebug("rs_now->t", tp_rank);
      #endif

      // MoE routing
      router_gemm_batched(weights_now->w_router, rs_now->t, weights_now->b_router,
                  rs_now->router_score, 1ll * l, false, false);

      #ifdef DEBUG
        if (flow_id == 0) rs_now->router_score->printDebug("rs_now->router_score", tp_rank);
      #endif

      // Select top-k experts
      topk_softmax_batched(rs_now->router_score, rs_now->topk_v, rs_now->topk_i, false, false, false);
      
      #ifdef DEBUG
        if (flow_id == 0) rs_now->topk_v->printDebug("rs_now->topk_v", tp_rank);
      #endif

      // Route the tokens to their corresponding top-k experts
      moe_apply_topk_batched(rs_now->t, weights_now->w_mlp1, weights_now->b_mlp1, weights_now->w_mlp2, weights_now->b_mlp2,
                    rs_now->topk_i, rs_now->topk_v, rs_now->mlp1_out, rs_now->gate_up, rs_now->tb3, rs_now->e_agg,
                    p->swiglu_limit, 1ll * l, false, false, false, false, tp_rank);

      // blocking for tensor aggregation here
      CHECK_HIP(hipStreamSynchronize(0));

      pthread_barrier_wait(tp_barrier);
      
      #ifdef DEBUG
        if (flow_id == 0) rs_now->e_agg->printDebug("rs_now->e_agg", tp_rank);
      #endif

      // do tensor aggregation here, do later
      if (tp_rank == 0) {
        int device_from = cur_device + 1;
        OurRunState *rs_to_agg = &rs[cur_device + 1];
        size_t bytes_agg = rs_to_agg->e_agg->num_elem() * rs_to_agg->e_agg->get_dtype_size();
        for (int i = 1; i < TP; i++) {
          CHECK_HIP(hipMemcpyPeerAsync(rs_now->e_agg_buf->d_buf, cur_device, rs_to_agg->e_agg->d_buf, device_from, bytes_agg));

          add_vector_batched(rs_now->e_agg, rs_now->e_agg_buf, false, false, false);  // equals residual add

          rs_to_agg++;
          device_from++;
        }

        CHECK_HIP(hipStreamSynchronize(0));
      }

      pthread_barrier_wait(tp_barrier);

      if (tp_rank > 0) {
        int device_from = cur_device - tp_rank;
        OurRunState *rs_orig = &rs[device_from];
        size_t bytes_agg = rs_orig->e_agg->num_elem() * rs_orig->e_agg->get_dtype_size();

        CHECK_HIP(hipMemcpyPeerAsync(rs_now->e_agg->d_buf, device_from, rs_orig->e_agg->d_buf, cur_device, bytes_agg));

        CHECK_HIP(hipStreamSynchronize(0));
      }

      pthread_barrier_wait(tp_barrier);

      #ifdef DEBUG
        if (flow_id == 0) rs_now->e_agg->printDebug("rs_now->e_agg", tp_rank);
      #endif

      // residual connection
      add_vector_batched(rs_now->x, rs_now->e_agg, false, false, false);  // equals residual add
      
      #ifdef DEBUG
        if (flow_id == 0) rs_now->x->printDebug("rs_now->x", tp_rank);
      #endif
    }
  }

  // exit(1);

  #ifdef DEBUG
    #pragma omp critical
    {
      if (flow_id == 0) {
        printf("tp_rank %d, exiting from debug block.\n", tp_rank);
        fflush(stdout);
        exit(1);
      }
    }
  #endif

  // final rmsnorm
  rmsnorm_batched(rs_now->x, weights_now->rms_out_w, rs_now->x, 0ll, false, false);

  // classifier into logits
  classifier_gemm_batched(weights_now->out, rs_now->x, rs_now->logits, false, true);

  return rs_now->logits->buf;
}

float *forward_gpu_20b_batched(
  int *tokens, int pos, int cur_batch_size, int flow_id
) {
  Config *p = public_config;
  OurTransformerWeights *weights_now = &weights[flow_id];
  OurRunState *rs_now = &rs[flow_id];
  
  // copy the token embedding into x
  embedding_lookup_batched(weights_now->token_embedding_table, tokens, rs_now->x, false);

  long long kv_dim = 1ll * p->n_kv_heads * p->head_dim;
  long long loff_one = 1ll * p->seq_len * kv_dim;
  long long loff_one_batch = 1ll * p->n_layers * loff_one;

  // forward all the layers
  for (int l = 0; l < p->n_layers; l++) {
    // attention rmsnorm
    rmsnorm_batched(rs_now->x, weights_now->rms_attn_w, rs_now->t, 1ll * l,
                    false, false);

    // key and value point to the kv cache
    long long loff = 1ll * l * loff_one;  // kv cache layer offset

    // QKV projection
    qkv_gemm_batched(rs_now->t, weights_now->w_qkv, weights_now->b_qkv, rs_now->qkv, 1ll * l, false, false);  // This kernel diverges the most

    // Separate q, k, v + RoPE
    qkv_split_rope_batched(rs_now->qkv, rs_now->q, rs_now->k, rs_now->v, rs_now->cos_tensor,
                          rs_now->sin_tensor, p->head_dim, p->n_attn_heads,
                          p->n_kv_heads, pos, false, false, false, false);

    // Store k, v in cache
    for (int b = 0; b < cur_batch_size; b++) {
      memcpy_tensor(rs_now->key_cache, rs_now->k, 1ll * b * loff_one_batch + loff + 1ll * pos * kv_dim, 1ll * b * kv_dim, kv_dim, false, true);
      memcpy_tensor(rs_now->value_cache, rs_now->v, 1ll * b * loff_one_batch + loff + 1ll * pos * kv_dim, 1ll * b * kv_dim, kv_dim, false, true);
    }

    // multihead attention
    int kv_mul = p->n_attn_heads / p->n_kv_heads;  // integer multiplier for GQA

    // FIX single_query_attn_batched later
    single_query_attn_batched(rs_now->q, rs_now->key_cache, rs_now->value_cache, rs_now->mask, 
                              weights_now->attn_sinks, rs_now->tb, p->head_dim,
                              p->n_attn_heads, kv_mul, kv_dim, p->seq_len,
                              p->sliding_window, pos, 1ll * l, false, false,
                              false, false, false);

    // final matmul to get the output of the attention
    attn_out_project_batched(rs_now->tb, weights_now->w_o, weights_now->b_o, rs_now->tb2, 1ll * l, false, false);

    // residual connection back into x
    add_vector_batched(rs_now->x, rs_now->tb2, false, false, false);  // equals residual add

    // ffn rmsnorm
    rmsnorm_batched(rs_now->x, weights_now->rms_ffn_w, rs_now->t, 1ll * l, false, false);

    // MoE routing
    router_gemm_batched(weights_now->w_router, rs_now->t, weights_now->b_router,
                rs_now->router_score, 1ll * l, false, false);

    // Select top-k experts
    topk_softmax_batched(rs_now->router_score, rs_now->topk_v, rs_now->topk_i, false, false, false);

    // Route the tokens to their corresponding top-k experts
    moe_apply_topk_batched(rs_now->t, weights_now->w_mlp1, weights_now->b_mlp1, weights_now->w_mlp2, weights_now->b_mlp2,
                   rs_now->topk_i, rs_now->topk_v, rs_now->mlp1_out, rs_now->gate_up, rs_now->tb3, rs_now->e_agg,
                   p->swiglu_limit, 1ll * l, false, false, false, false, 0);

    // residual connection
    add_vector_batched(rs_now->x, rs_now->e_agg, false, false, false);  // equals residual add
  }

  // final rmsnorm
  rmsnorm_batched(rs_now->x, weights_now->rms_out_w, rs_now->x, 0ll, false, false);

  // classifier into logits
  classifier_gemm_batched(weights_now->out, rs_now->x, rs_now->logits, false, true);

  return rs_now->logits->buf;
}

float *forward_gpu_20b(
  Config *p, OurTransformerWeights *weights, OurRunState *rs,
  int token, int pos
) {
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
    qkv_split_rope(rs->qkv, rs->q, rs->k, rs->v, rs->cos_tensor,
                  rs->sin_tensor, p->head_dim, p->n_attn_heads, p->n_kv_heads,
                  pos, false, false, false, false);

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

float *forward_cpu_20b(
  Config *p, OurTransformerWeights *weights, OurRunState *rs,
  int token, int pos
) {
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
