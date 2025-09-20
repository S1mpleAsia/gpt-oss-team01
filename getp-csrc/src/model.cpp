#include "../include/model.hpp"
#include <cmath>
#include <cstring>

#ifdef RUN_20B

float *forward_gpu_20b_batched(int *tokens, int pos, int cur_batch_size, int flow_id) {
  Config *p = public_config;

  OurTransformerWeights *weights_now = &weights[flow_id];
  OurRunState *rs_now = &rs[flow_id];
  hipStream_t stream = total_streams[flow_id];

  // copy the token embedding into x
  embedding_lookup_batched(weights_now->token_embedding_table, tokens, rs_now->tokens_buf,
                           rs_now->x, cur_batch_size, false, stream);

  long long kv_dim = 1ll * p->n_kv_heads * p->head_dim;
  long long loff_one = 1ll * p->seq_len * kv_dim;
  long long loff_one_batch = 1ll * p->n_layers * loff_one;

  // forward all the layers
  for (int l = 0; l < p->n_layers; l++) {
#ifdef DEBUG
    printf("Layer l=%d running...\n", l);
    fflush(stdout);
#endif
    // attention rmsnorm
    rmsnorm_batched(rs_now->x, weights_now->rms_attn_w, rs_now->t, cur_batch_size, 1ll * l, false,
                    false, 1e-5f, stream);

    // key and value point to the kv cache
    long long loff = 1ll * l * loff_one;  // kv cache layer offset

    // QKV projection
    qkv_gemm_batched_v2(rs_now->t, weights_now->w_qkv, weights_now->b_qkv, rs_now->qkv,
                        cur_batch_size, 1ll * l, false, false,
                        stream);  // This kernel diverges the most

    qkv_split_rope_fused(rs_now->qkv, rs_now->q, rs_now->key_cache, rs_now->value_cache,
                         rs_now->cos_tensor, rs_now->sin_tensor, cur_batch_size, p->head_dim,
                         p->n_attn_heads, p->n_kv_heads, pos, l, stream);

    // multihead attention
    int kv_mul = p->n_attn_heads / p->n_kv_heads;  // integer multiplier for GQA

    // FIX single_query_attn_batched later
    single_query_attn_flash_batched(
      rs_now->q, rs_now->key_cache, rs_now->value_cache, rs_now->mask, weights_now->attn_sinks,
      rs_now->tb, rs_now->g_fa_pmax, rs_now->g_fa_psum, rs_now->g_fa_pnum, cur_batch_size,
      p->head_dim, p->n_attn_heads, kv_mul, kv_dim, p->seq_len, p->sliding_window, pos, 1ll * l,
      false, false, false, false, false, stream);

    // final matmul to get the output of the attention
    attn_out_project_batched_v2(rs_now->tb, weights_now->w_o, weights_now->b_o, rs_now->tb2, true,
                                cur_batch_size, 1ll * l, false, false, stream);

    // residual connection back into x
    // add_vector_batched(rs_now->x, rs_now->tb2, false, false, false);  // equals residual add

    // ffn rmsnorm
    // rmsnorm_batched(rs_now->x, weights_now->rms_ffn_w, rs_now->t, 1ll * l, false, false);

    residual_rmsnorm_batched(rs_now->tb2, rs_now->x, weights_now->rms_ffn_w, rs_now->t,
                             cur_batch_size, 1ll * l, 1e-5f, stream);

#ifdef DEBUG
    rs_now->x->printDebug("rs_now->x", 0, 0, stream);
#endif

    // MoE routing
    router_gemm_batched(weights_now->w_router, rs_now->t, weights_now->b_router,
                        rs_now->router_score, cur_batch_size, 1ll * l, false, false, stream);

#ifdef DEBUG
    rs_now->router_score->printDebug("rs_now->router_score", 0, 0, stream);
#endif

    // Select top-k experts
    topk_softmax_batched(rs_now->router_score, rs_now->topk_v, rs_now->topk_i, cur_batch_size,
                         false, false, false, stream);

#ifdef DEBUG
    rs_now->topk_v->printDebug("rs_now->topk_v", 0, 0, stream);
#endif

    // Route the tokens to their corresponding top-k experts
    // moe_apply_topk_batched(rs_now->t, weights_now->w_mlp1, weights_now->b_mlp1, weights_now->w_mlp2,
    //                        weights_now->b_mlp2, rs_now->topk_i, rs_now->topk_v, rs_now->mlp1_out, rs_now->gate_up,
    //                        rs_now->tb3, rs_now->e_agg, p->swiglu_limit, 1ll * l, false, false, false,
    //                        false);

    // moe_mlp1_batched(rs_now->t, weights_now->w_mlp1, weights_now->b_mlp1, rs_now->topk_i, rs_now->mlp1_out, false,
    //                  false, 1ll * l);

    // moe_swiglu_batched(rs_now->mlp1_out, rs_now->gate_up, p->experts_per_token, p->swiglu_limit);

    // moe_mlp2_batched(rs_now->gate_up, weights_now->w_mlp2, weights_now->b_mlp2, rs_now->tb3, rs_now->topk_i,
    //                  p->experts_per_token, true, 1ll * l);

    // moe_agg_batched(rs_now->tb3, rs_now->topk_v, rs_now->e_agg, p->experts_per_token, false);

    int total_pairs = cur_batch_size * p->experts_per_token;
    moe_init_buffers_hip(rs_now->e_agg, rs_now->mlp1_out, rs_now->gate_up, rs_now->tb3,
                         rs_now->sorted_pair_ids, rs_now->expert_offsets, rs_now->x_packed,
                         cur_batch_size, p->hidden_dim, stream);

    moe_build_offsets_hip(rs_now->topk_i, rs_now->sorted_pair_ids, rs_now->expert_offsets,
                          cur_batch_size, p->experts_per_token, p->n_experts, stream);

    moe_pack_inputs_hip(rs_now->t, rs_now->sorted_pair_ids, rs_now->x_packed, cur_batch_size,
                        p->hidden_dim, p->experts_per_token, stream);

    int max_rows = moe_get_max_rows_per_expert_hip(rs_now->expert_offsets, rs_now->max_rows,
                                                   p->n_experts, stream);
    moe_mlp1_forward_hip(rs_now->x_packed, weights_now->w_mlp1, weights_now->b_mlp1,
                         rs_now->expert_offsets, rs_now->mlp1_out, 1ll * l, p->n_experts,
                         p->hidden_dim, p->intermediate_dim, max_rows, total_pairs, stream);

    moe_swiglu_hip(rs_now->mlp1_out, rs_now->gate_up, cur_batch_size, p->experts_per_token,
                   p->intermediate_dim, p->swiglu_limit, stream);

    moe_mlp2_forward_hip(rs_now->gate_up, weights_now->w_mlp2, weights_now->b_mlp2,
                         rs_now->expert_offsets, rs_now->tb3, true, 1ll * l, p->n_experts,
                         p->intermediate_dim, p->hidden_dim, max_rows, total_pairs, stream);

    moe_scatter_aggregate_hip(rs_now->tb3, rs_now->sorted_pair_ids, rs_now->topk_v, rs_now->e_agg,
                              rs_now->expert_offsets, p->hidden_dim, p->experts_per_token,
                              p->n_experts, max_rows, stream);

#ifdef DEBUG
    rs_now->tb3->printDebug("rs_now->tb3", 0, 0, stream);
    rs_now->e_agg->printDebug("rs_now->e_agg", 0, 0, stream);
#endif

    // residual connection
    add_vector_batched(rs_now->x, rs_now->e_agg, cur_batch_size, false, false, false,
                       stream);  // equals residual add

#ifdef DEBUG
    rs_now->x->printDebug("rs_now->x_end", 0, 0, stream);
#endif
  }

  // final rmsnorm
  rmsnorm_batched(rs_now->x, weights_now->rms_out_w, rs_now->x, cur_batch_size, 0ll, false, false,
                  1e-5f, stream);

  // classifier into logits
  classifier_gemm_batched_v2(weights_now->out, rs_now->x, rs_now->logits, cur_batch_size, false,
                             true, stream);

  return rs_now->logits->buf;
}

#else

// two events are needed
float *forward_gpu_120b_batched(int *tokens, int pos, int cur_batch_size, int flow_id, int tp_rank,
                                int pp_rank, pthread_barrier_t *tp_barrier) {
  Config *p = public_config;
  int cur_device = flow_id * TOTAL_PIPELINES + pp_rank * TP + tp_rank;

  OurTransformerWeights *weights_now = &weights[cur_device];
  OurRunState *rs_now = &rs[cur_device];
  OurRunState *rs_leader = &rs[cur_device - tp_rank];
  hipStream_t stream = total_streams[cur_device];
  hipEvent_t tp_ready = total_events->tp_ready[cur_device];
  hipEvent_t tp_finish = total_events->tp_finish[cur_device];
  hipEvent_t pp_sync = total_events->pp_sync[cur_device];
  // hipStream_t stream = 0;

#ifdef DEBUG
  bool flag = (flow_id == 0);
  if (flag) {
    printf("Running on device %d..... - tp_rank = %d and pp_rank = %d\n", cur_device, tp_rank,
           pp_rank);
    fflush(stdout);
  }
#endif

  CHECK_HIP(hipSetDevice(cur_device));

  if (pp_rank == 0) {
    embedding_lookup_batched(weights_now->token_embedding_table, tokens, rs_now->tokens_buf,
                             rs_now->x_embed_buf, cur_batch_size, false, stream);
    all_gather_x(rs_now, rs_leader, tp_rank, cur_device, cur_batch_size, tp_barrier, stream, tp_ready, tp_finish);
  } else {
    rs_now->pipeline_each->dequeue(rs_now->x, stream);
  }

  long long kv_dim = 1ll * p->n_kv_heads * p->head_dim;
  long long loff_one = 1ll * p->seq_len * kv_dim;
  long long loff_one_batch = 1ll * p->n_layers * loff_one;

  for (int l = 0; l < p->n_layers / PP; l++) {
#ifdef DEBUG
    if (flag) {
      printf("Layer l=%d running...\n", l);
      fflush(stdout);
      rs_now->x->printDebug("rs_now->x", tp_rank, 0, stream);
    }
#endif

    // attention rmsnorm
    rmsnorm_batched(rs_now->x, weights_now->rms_attn_w, rs_now->t, cur_batch_size, 1ll * l, false,
                    false, 1e-5f, stream);

#ifdef DEBUG
    if (flag)
      rs_now->t->printDebug("rs_now->t", tp_rank, 0, stream);
#endif

    // key and value point to the kv cache
    long long loff = 1ll * l * loff_one;  // kv cache layer offset

    // QKV projection
    qkv_gemm_batched_v2(rs_now->t, weights_now->w_qkv, weights_now->b_qkv, rs_now->qkv,
                        cur_batch_size, 1ll * l, false, false,
                        stream);  // This kernel diverges the most

    // all_gather_qkv_v2(rs_now, rs_leader, tp_rank, cur_device, cur_batch_size, tp_barrier, stream, tp_ready, tp_finish);

#ifdef DEBUG
    if (flag)
      rs_now->qkv->printDebug("rs_now->qkv", tp_rank, 0, stream);
#endif

    // // Separate q, k, v + RoPE + Store k, v in cache
    qkv_split_rope_fused(rs_now->qkv, rs_now->q, rs_now->key_cache, rs_now->value_cache,
                         rs_now->cos_tensor, rs_now->sin_tensor, cur_batch_size, p->head_dim,
                         p->n_attn_heads / TP, p->n_kv_heads / TP, pos, l, stream);

#ifdef DEBUG
    if (flag) {
      rs_now->q->printDebug("rs_now->q", tp_rank, 0, stream);
      rs_now->key_cache->printDebug("rs_now->key_cache", tp_rank, 0, stream);
      rs_now->value_cache->printDebug("rs_now->value_cache", tp_rank, 0, stream);
    }
#endif

    // multihead attention
    int kv_mul = p->n_attn_heads / p->n_kv_heads;  // integer multiplier for GQA

    // FIX single_query_attn_batched later
    single_query_attn_flash_batched(
      rs_now->q, rs_now->key_cache, rs_now->value_cache, rs_now->mask, weights_now->attn_sinks,
      rs_now->tb, rs_now->g_fa_pmax, rs_now->g_fa_psum, rs_now->g_fa_pnum, cur_batch_size,
      p->head_dim, p->n_attn_heads / TP, kv_mul, kv_dim / TP, p->seq_len, p->sliding_window, pos,
      1ll * l, false, false, false, false, false, stream);

    // all_gather_tb(rs_now, rs_leader, tp_rank, cur_device, cur_batch_size, tp_barrier, stream,
    //               tp_ready, tp_finish);

#ifdef DEBUG
    if (flag)
      rs_now->tb->printDebug("rs_now->tb", tp_rank, 0, stream);
#endif

    // final matmul to get the output of the attention
    attn_out_project_batched_v2(rs_now->tb, weights_now->w_o, weights_now->b_o, rs_now->tb2,
                                tp_rank == 0, cur_batch_size, 1ll * l, false, false, stream);

    /*
    {
      std::string timer_label = "reduce_tb2_new_tp" + std::to_string(tp_rank);
      GpuTimer timer(timer_label.c_str(), stream);
      reduce_tb2_new(rs_now, rs_leader, tp_rank, cur_device, cur_batch_size, tp_barrier, stream, tp_ready,
               tp_finish);
    }
    */

    {
      std::string timer_label = "reduce_tb2_tp" + std::to_string(tp_rank);
      GpuTimer timer(timer_label.c_str(), stream);
      reduce_tb2(rs_now, rs_leader, tp_rank, cur_device, cur_batch_size, tp_barrier, stream, tp_ready,
               tp_finish);
    }

#ifdef DEBUG
    if (flag)
      rs_now->tb2->printDebug("rs_now->tb2", tp_rank, 0, stream);
#endif

    // residual connection back into x + ffn rmsnorm
    residual_rmsnorm_batched(rs_now->tb2, rs_now->x, weights_now->rms_ffn_w, rs_now->t,
                             cur_batch_size, 1ll * l, 1e-5f, stream);

#ifdef DEBUG
    if (flag)
      rs_now->x->printDebug("rs_now->x", tp_rank, 0, stream);
#endif

    // MoE routing
    router_gemm_batched(weights_now->w_router, rs_now->t, weights_now->b_router,
                        rs_now->router_score, cur_batch_size, 1ll * l, false, false, stream);

#ifdef DEBUG
    if (flag)
      rs_now->router_score->printDebug("rs_now->router_score", tp_rank, 0, stream);
#endif

    // Select top-k experts
    topk_softmax_batched(rs_now->router_score, rs_now->topk_v, rs_now->topk_i, cur_batch_size,
                         false, false, false, stream);

#ifdef DEBUG
    if (flag)
      rs_now->topk_v->printDebug("rs_now->topk_v", tp_rank, 0, stream);
#endif

    int total_pairs = cur_batch_size * p->experts_per_token;
    moe_init_buffers_hip(rs_now->e_agg, rs_now->mlp1_out, rs_now->gate_up, rs_now->tb3,
                         rs_now->sorted_pair_ids, rs_now->expert_offsets, rs_now->x_packed,
                         cur_batch_size, p->hidden_dim, stream);

#ifdef DEBUG
    if (flag)
      rs_now->e_agg->printDebug("rs_now->e_agg", tp_rank, 0, stream);
#endif

    if (tp_rank == 0) {
      moe_build_offsets_hip(rs_now->topk_i, rs_now->sorted_pair_ids, rs_now->expert_offsets,
                            cur_batch_size, p->experts_per_token, p->n_experts, stream);

      CHECK_HIP(hipEventRecord(tp_ready, stream));
    }

    pthread_barrier_wait(tp_barrier);

    if (tp_rank > 0) {
      hipEvent_t leader_tp_ready = total_events->tp_ready[cur_device - tp_rank];
      CHECK_HIP(hipStreamWaitEvent(stream, leader_tp_ready));

      void *dst_sorted_ids = rs_now->sorted_pair_ids->d_buf;
      const void *src_sorted_ids = rs_leader->sorted_pair_ids->d_buf;
      size_t bytes_sorted_ids = rs_now->sorted_pair_ids->num_elem() * sizeof(int);

      void *dst_offsets = rs_now->expert_offsets->d_buf;
      const void *src_offsets = rs_leader->expert_offsets->d_buf;
      size_t bytes_offsets = rs_now->expert_offsets->num_elem() * sizeof(int);

      CHECK_HIP(hipMemcpyPeerAsync(dst_sorted_ids, cur_device, src_sorted_ids, cur_device - tp_rank,
                                   bytes_sorted_ids, stream));
      CHECK_HIP(hipMemcpyPeerAsync(dst_offsets, cur_device, src_offsets, cur_device - tp_rank,
                                   bytes_offsets, stream));
    }

    pthread_barrier_wait(tp_barrier);

#ifdef DEBUG
    if (flag) {
      rs_now->sorted_pair_ids->printDebug("rs_now->sorted_pair_ids build offset", tp_rank, 0,
                                          stream);
      rs_now->expert_offsets->printDebug("rs_now->expert_offsets", tp_rank, 0, stream);
    }
#endif

    moe_pack_inputs_hip(rs_now->t, rs_now->sorted_pair_ids, rs_now->x_packed, cur_batch_size,
                        p->hidden_dim, p->experts_per_token, stream);

#ifdef RUN_EP
    rs_now->expert_offsets->from_device(stream);
    moe_build_local_ep_data(rs_now, rs_leader, tp_rank, p, tp_barrier, stream);
#endif

#ifdef DEBUG
    if (flag)
      rs_now->x_packed->printDebug("rs_now->x_packed", tp_rank, 0, stream);
#endif

#ifdef RUN_EP
    int max_rows = moe_get_max_rows_per_expert_hip(rs_now->expert_offsets_local, rs_now->max_rows,
                                                   p->n_experts / TP, stream);
    moe_mlp1_forward_hip(rs_now->x_packed_local, weights_now->w_mlp1, weights_now->b_mlp1,
                         rs_now->expert_offsets_local, rs_now->mlp1_out, 1ll * l, p->n_experts / TP,
                         p->hidden_dim, p->intermediate_dim, max_rows, total_pairs, stream);

    moe_swiglu_hip(rs_now->mlp1_out, rs_now->gate_up, cur_batch_size, p->experts_per_token,
                   p->intermediate_dim, p->swiglu_limit, stream);

#else
    int max_rows = moe_get_max_rows_per_expert_hip(rs_now->expert_offsets, rs_now->max_rows,
                                                   p->n_experts, stream);
    moe_mlp1_forward_hip(rs_now->x_packed, weights_now->w_mlp1, weights_now->b_mlp1,
                         rs_now->expert_offsets, rs_now->mlp1_out, 1ll * l, p->n_experts,
                         p->hidden_dim, p->intermediate_dim / TP, max_rows, total_pairs, stream);

    moe_swiglu_hip(rs_now->mlp1_out, rs_now->gate_up, cur_batch_size, p->experts_per_token,
                   p->intermediate_dim / TP, p->swiglu_limit, stream);
#endif

#ifdef DEBUG
    if (flag)
      rs_now->gate_up->printDebug("rs_now->gate_up", tp_rank, 0, stream);
#endif

#ifdef RUN_EP
    moe_mlp2_forward_hip(rs_now->gate_up, weights_now->w_mlp2, weights_now->b_mlp2,
                         rs_now->expert_offsets_local, rs_now->tb3, true, 1ll * l,
                         p->n_experts / TP, p->intermediate_dim, p->hidden_dim, max_rows,
                         total_pairs, stream);
#else
    moe_mlp2_forward_hip(rs_now->gate_up, weights_now->w_mlp2, weights_now->b_mlp2,
                         rs_now->expert_offsets, rs_now->tb3, tp_rank == 0, 1ll * l, p->n_experts,
                         p->intermediate_dim / TP, p->hidden_dim, max_rows, total_pairs, stream);
#endif

#ifdef DEBUG
    if (flag)
      rs_now->tb3->printDebug("rs_now->tb3", tp_rank, 0, stream);
#endif

#ifdef RUN_EP
    all_gather_tb3(rs_now, rs_leader, tp_rank, cur_device, p, tp_barrier, stream, tp_ready,
                   tp_finish);
#else
    /*
    {
      std::string timer_label = "reduce_tb3_new_tp" + std::to_string(tp_rank);
      GpuTimer timer(timer_label.c_str(), stream);
      reduce_tb3_new(rs_now, rs_leader, tp_rank, cur_device, cur_batch_size, tp_barrier, stream, tp_ready,
               tp_finish);
    }
    */

    {
      std::string timer_label = "reduce_tb3_tp" + std::to_string(tp_rank);
      GpuTimer timer(timer_label.c_str(), stream);
      reduce_tb3(rs_now, rs_leader, tp_rank, cur_device, cur_batch_size, tp_barrier, stream, tp_ready,
                tp_finish);
    }

#endif

#ifdef DEBUG
    if (flag)
      rs_now->tb3->printDebug("rs_now->tb3 after", tp_rank, 0, stream);
#endif

#ifdef RUN_EP
    max_rows = moe_get_max_rows_per_expert_hip(rs_now->expert_offsets, rs_now->max_rows,
                                               p->n_experts, stream);
#endif
    moe_scatter_aggregate_hip(rs_now->tb3, rs_now->sorted_pair_ids, rs_now->topk_v, rs_now->e_agg,
                              rs_now->expert_offsets, p->hidden_dim, p->experts_per_token,
                              p->n_experts, max_rows, stream);

#ifdef DEBUG
    if (flag)
      rs_now->e_agg->printDebug("rs_now->e_agg", tp_rank, 0, stream);
#endif

    // residual connection
    add_vector_batched(rs_now->x, rs_now->e_agg, cur_batch_size, false, false, false,
                       stream);  // equals residual add

#ifdef DEBUG
    if (flag)
      rs_now->x->printDebug("rs_now->x", tp_rank, 0, stream);
#endif
  }

  if ((pp_rank + 1) % PP == 0) {
    // final rmsnorm
    rmsnorm_batched(rs_now->x, weights_now->rms_out_w, rs_now->x, cur_batch_size, 0ll, false, false,
                    1e-5f, stream);

    // classifier into logits
    classifier_gemm_batched_v2(weights_now->out_buffer, rs_now->x, rs_now->tmp_logits,
                               cur_batch_size, false, false, stream);

    all_gather_classifier_final(rs_now, rs_leader, tp_rank, cur_device, cur_batch_size, tp_barrier,
                               stream, tp_ready, tp_finish);
                      

    return rs_leader->logits_out;
  } else {
    CHECK_HIP(hipEventRecord(pp_sync, stream));
    OurRunState *rs_new = &rs[cur_device + TP];
    hipStream_t stream_new = total_streams[cur_device + TP];
    rs_new->pipeline_each->enqueueElem(rs_now->x, cur_device, pp_sync, stream_new);

    return nullptr;
  }
}

#endif
