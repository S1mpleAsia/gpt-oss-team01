#include "../include/model.hpp"
#include <cmath>
#include <cstring>

#ifdef RUN_20B

float *forward_gpu_20b_batched(
  int *tokens, int pos, int cur_batch_size, int flow_id
) {
  Config *p = public_config;
  
  OurTransformerWeights *weights_now = &weights[flow_id];
  OurRunState *rs_now = &rs[flow_id];
  hipStream_t stream = total_streams[flow_id];

  // copy the token embedding into x
  embedding_lookup_batched(weights_now->token_embedding_table, tokens, rs_now->x, false, stream);

  long long kv_dim = 1ll * p->n_kv_heads * p->head_dim;
  long long loff_one = 1ll * p->seq_len * kv_dim;
  long long loff_one_batch = 1ll * p->n_layers * loff_one;
  
  int total_pairs = BATCH_SIZE * p->experts_per_token;

  // forward all the layers
  for (int l = 0; l < p->n_layers; l++) {
    // attention rmsnorm
    rmsnorm_batched(rs_now->x, weights_now->rms_attn_w, rs_now->t, 1ll * l, false, false, 1e-5f, stream);

    // key and value point to the kv cache
    long long loff = 1ll * l * loff_one;  // kv cache layer offset

    // QKV projection
    qkv_gemm_batched_v2(rs_now->t, weights_now->w_qkv, weights_now->b_qkv, rs_now->qkv, 1ll * l, false,
                        false, stream);  // This kernel diverges the most

    qkv_split_rope_fused(rs_now->qkv, rs_now->q, rs_now->key_cache, rs_now->value_cache, rs_now->cos_tensor,
                         rs_now->sin_tensor, p->head_dim, p->n_attn_heads, p->n_kv_heads, pos, l, stream);

    // multihead attention
    int kv_mul = p->n_attn_heads / p->n_kv_heads;  // integer multiplier for GQA

    // FIX single_query_attn_batched later
    single_query_attn_batched(rs_now->q, rs_now->key_cache, rs_now->value_cache, rs_now->mask, weights_now->attn_sinks,
                              rs_now->tb, p->head_dim, p->n_attn_heads, kv_mul, kv_dim, p->seq_len,
                              p->sliding_window, pos, 1ll * l, false, false, false, false, false, stream);

    // final matmul to get the output of the attention
    attn_out_project_batched_v2(rs_now->tb, weights_now->w_o, weights_now->b_o, rs_now->tb2, 1ll * l, false, false, stream);

    // residual connection back into x
    // add_vector_batched(rs_now->x, rs_now->tb2, false, false, false);  // equals residual add

    // ffn rmsnorm
    // rmsnorm_batched(rs_now->x, weights_now->rms_ffn_w, rs_now->t, 1ll * l, false, false);

    residual_rmsnorm_batched(rs_now->tb2, rs_now->x, weights_now->rms_ffn_w, rs_now->t, 1ll * l, 1e-5f, stream);

    // MoE routing
    router_gemm_batched(weights_now->w_router, rs_now->t, weights_now->b_router, rs_now->router_score, 1ll * l,
                        false, false, stream);

    // Select top-k experts
    topk_softmax_batched(rs_now->router_score, rs_now->topk_v, rs_now->topk_i, false, false, false, stream);

    // Route the tokens to their corresponding top-k experts
    moe_init_buffers_hip(rs_now->e_agg, rs_now->mlp1_out, rs_now->gate_up, rs_now->tb3, rs_now->sorted_pair_ids,
                         rs_now->expert_offsets, rs_now->x_packed, BATCH_SIZE, p->hidden_dim, stream);

    moe_build_offsets_hip(rs_now->topk_i, rs_now->sorted_pair_ids, rs_now->expert_offsets, BATCH_SIZE,
                          p->experts_per_token, p->n_experts, stream);

    moe_pack_inputs_hip(rs_now->t, rs_now->sorted_pair_ids, rs_now->x_packed, BATCH_SIZE, p->hidden_dim,
                        p->experts_per_token, stream);

    int max_rows = moe_get_max_rows_per_expert_hip(rs_now->expert_offsets, p->n_experts, stream);
    moe_mlp1_forward_hip(rs_now->x_packed, weights_now->w_mlp1, weights_now->b_mlp1, rs_now->expert_offsets,
                         rs_now->mlp1_out, 1ll * l, p->n_experts, p->hidden_dim, p->intermediate_dim,
                         max_rows, total_pairs, stream);

    moe_swiglu_hip(rs_now->mlp1_out, rs_now->gate_up, BATCH_SIZE, p->experts_per_token, p->intermediate_dim,
                   p->swiglu_limit, stream);

    moe_mlp2_forward_hip(rs_now->gate_up, weights_now->w_mlp2, weights_now->b_mlp2, rs_now->expert_offsets, rs_now->tb3,
                         true, 1ll * l, p->n_experts, p->intermediate_dim, p->hidden_dim, max_rows,
                         total_pairs, stream);

    moe_scatter_aggregate_hip(rs_now->tb3, rs_now->sorted_pair_ids, rs_now->topk_v, rs_now->e_agg,
                              rs_now->expert_offsets, p->hidden_dim, p->experts_per_token, p->n_experts,
                              max_rows, stream);

    // residual connection
    add_vector_batched(rs_now->x, rs_now->e_agg, false, false, false, stream);  // equals residual add
  }

  // final rmsnorm
  rmsnorm_batched(rs_now->x, weights_now->rms_out_w, rs_now->x, 0ll, false, false, 1e-5f, stream);

  // classifier into logits
  classifier_gemm_batched_v2(weights_now->out, rs_now->x, rs_now->logits, false, true, stream);

  return rs_now->logits->buf;
}

#else

void reduce_tb3(
  OurRunState *rs_now, OurRunState *rs_leader,
  int tp_rank, int cur_device, pthread_barrier_t *tp_barrier,
  hipStream_t stream, hipEvent_t tp_ready, hipEvent_t tp_finish
) {
  if (tp_rank > 0) {
    // copy all tp_rank to buffer
    size_t num_elems = rs_now->tb3->num_elem();
    size_t num_bytes = num_elems * rs_now->tb3->get_dtype_size();
    // rs of leader
    void *dst_buf = (void *)((float *)rs_leader->tb3_buf->d_buf + (tp_rank-1) * num_elems);
    const void *src_buf = rs_now->tb3->d_buf;

    CHECK_HIP(hipMemcpyPeerAsync(dst_buf, cur_device - tp_rank, src_buf, cur_device, num_bytes, stream));
    CHECK_HIP(hipEventRecord(tp_ready, stream));
  }

  #ifdef DEBUG
    bool flag = (tp_rank == 0);
    if (flag) rs_now->tb3->printDebug("rs_now->tb3 after copy to leader", tp_rank, 0, stream);
  #endif

  pthread_barrier_wait(tp_barrier);

  if (tp_rank == 0) {
    // aggregates here
    hipEvent_t tp_ready_each;
    size_t num_elements = rs_now->tb3->num_elem();
    float *d_buf = (float *)rs_now->tb3->d_buf;
    float *d_buf_each = (float *)rs_leader->tb3_buf->d_buf;

    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    for (int i = 1; i < TP; i++) {
      tp_ready_each = total_events->tp_ready[cur_device + i];
      CHECK_HIP(hipStreamWaitEvent(stream, tp_ready_each));
      // add vector kernel here, do later
      // do on stream
      
      add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(d_buf, (const float*)d_buf_each, num_elements);

      d_buf_each += num_elements;
    }
    CHECK_HIP(hipEventRecord(tp_ready, stream));
  }

  pthread_barrier_wait(tp_barrier);

  #ifdef DEBUG
    if (flag) rs_now->tb3->printDebug("rs_now->tb3 after aggregate", tp_rank, 0, stream);
  #endif

  if (tp_rank > 0) {
    // copy back
    hipEvent_t leader_tp_ready = total_events->tp_ready[cur_device - tp_rank];
    hipStream_t leader_stream = total_streams[cur_device - tp_rank];
    size_t num_bytes = rs_now->tb3->num_elem() * rs_now->tb3->get_dtype_size();
    // rs of leader TP
    void *dst_buf = rs_now->tb3->d_buf;
    const void *src_buf = rs_leader->tb3->d_buf;

    CHECK_HIP(hipStreamWaitEvent(stream, leader_tp_ready));
    CHECK_HIP(hipMemcpyPeerAsync(dst_buf, cur_device, src_buf, cur_device - tp_rank, num_bytes, stream));
    CHECK_HIP(hipEventRecord(tp_finish, stream));
  } else {
    hipEvent_t tp_finish_each; 
    for (int i = 1; i < TP; i++) {
      tp_finish_each = total_events->tp_finish[cur_device + i];
      CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_each));
    }
  }
  
  pthread_barrier_wait(tp_barrier);
}

// two events are needed
float *forward_gpu_120b_batched(
  int *tokens, int pos, int cur_batch_size, int flow_id,
  int tp_rank, int pp_rank, pthread_barrier_t *tp_barrier
) {
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
      printf("Running on device %d w/ tp_rank=%d and pp_rank=%d.....\n", cur_device, tp_rank, pp_rank);
      fflush(stdout);
    }
  #endif

  // copy the token embedding into x
  if (pp_rank == 0) {
    embedding_lookup_batched(weights_now->token_embedding_table, tokens, rs_now->x, false, stream);
  } else {
    rs_now->pipeline_each->dequeue(rs_now->x, stream);
  }

  long long kv_dim = 1ll * p->n_kv_heads * p->head_dim;
  long long loff_one = 1ll * p->seq_len * kv_dim;
  long long loff_one_batch = 1ll * p->n_layers * loff_one;

  // forward all the layers
  for (int pipeline_id = 0; pipeline_id < PP; pipeline_id++) {
    /** SYNC LOOP
      if (pipeline_id) {
        cur_device += TP;
        CHECK_HIP(hipSetDevice(cur_device));

        OurRunState *rs_new = &rs[cur_device];
        stream = total_streams[cur_device];

        // sync from rs_now->x to rs_now->x
        rs_new->pipeline_each->enqueueElem(rs_now->x, cur_device - TP, pp_sync, stream);
        rs_new->pipeline_each->dequeue(rs_new->x, stream);

        weights_now = &weights[cur_device];
        rs_now = &rs[cur_device];
        rs_leader = &rs[cur_device - tp_rank];
        tp_ready = total_events->tp_ready[cur_device];
        tp_finish = total_events->tp_finish[cur_device];
        pp_sync = total_events->pp_sync[cur_device];
      }
    */

    /** MAIN LOOP
      for (int l = 0; l < p->n_layers / PP; l++) {
        #ifdef DEBUG
          if (flag) {
            printf("Layer l=%d running...\n", l);
            fflush(stdout);
            rs_now->x->printDebug("rs_now->x", tp_rank, 0, stream);
          }
        #endif

        // attention rmsnorm
        rmsnorm_batched(rs_now->x, weights_now->rms_attn_w, rs_now->t, 1ll * l, false, false, 1e-5f, stream);

        #ifdef DEBUG
          if (flag) rs_now->t->printDebug("rs_now->t", tp_rank, 0, stream);
        #endif

        // key and value point to the kv cache
        long long loff = 1ll * l * loff_one;  // kv cache layer offset

        // QKV projection
        qkv_gemm_batched_v2(rs_now->t, weights_now->w_qkv, weights_now->b_qkv, rs_now->qkv, 1ll * l, false,
                            false, stream);  // This kernel diverges the most

        #ifdef DEBUG
          if (flag) rs_now->qkv->printDebug("rs_now->qkv", tp_rank, 0, stream);
        #endif

        // // Separate q, k, v + RoPE + Store k, v in cache
        qkv_split_rope_fused(rs_now->qkv, rs_now->q, rs_now->key_cache, rs_now->value_cache, rs_now->cos_tensor,
                            rs_now->sin_tensor, p->head_dim, p->n_attn_heads, p->n_kv_heads, pos, l, stream);

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
        single_query_attn_batched(rs_now->q, rs_now->key_cache, rs_now->value_cache, rs_now->mask, weights_now->attn_sinks,
                                  rs_now->tb, p->head_dim, p->n_attn_heads, kv_mul, kv_dim, p->seq_len,
                                  p->sliding_window, pos, 1ll * l, false, false, false, false, false, stream);

        #ifdef DEBUG
          if (flag) rs_now->tb->printDebug("rs_now->tb", tp_rank, 0, stream);
        #endif

        // final matmul to get the output of the attention
        attn_out_project_batched_v2(rs_now->tb, weights_now->w_o, weights_now->b_o, rs_now->tb2, 1ll * l, false, false, stream);

        #ifdef DEBUG
          if (flag) rs_now->tb2->printDebug("rs_now->tb2", tp_rank, 0, stream);
        #endif

        // residual connection back into x + ffn rmsnorm
        residual_rmsnorm_batched(rs_now->tb2, rs_now->x, weights_now->rms_ffn_w, rs_now->t, 1ll * l, 1e-5f, stream);

        #ifdef DEBUG
          if (flag) rs_now->x->printDebug("rs_now->x", tp_rank, 0, stream);
        #endif

        // MoE routing
        router_gemm_batched(weights_now->w_router, rs_now->t, weights_now->b_router, rs_now->router_score, 1ll * l,
                            false, false, stream);

        #ifdef DEBUG
          if (flag) rs_now->router_score->printDebug("rs_now->router_score", tp_rank, 0, stream);
        #endif

        // Select top-k experts
        topk_softmax_batched(rs_now->router_score, rs_now->topk_v, rs_now->topk_i, false, false, false, stream);

        #ifdef DEBUG
          if (flag) rs_now->topk_v->printDebug("rs_now->topk_v", tp_rank, 0, stream);
        #endif

        int total_pairs = BATCH_SIZE * p->experts_per_token;
        moe_init_buffers_hip(rs_now->e_agg, rs_now->mlp1_out, rs_now->gate_up, rs_now->tb3, rs_now->sorted_pair_ids,
                            rs_now->expert_offsets, rs_now->x_packed, BATCH_SIZE, p->hidden_dim, stream);

        #ifdef DEBUG
          if (flag) rs_now->sorted_pair_ids->printDebug("rs_now->sorted_pair_ids", tp_rank, 0, stream);
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

          CHECK_HIP(hipMemcpyPeerAsync(dst_sorted_ids, cur_device, src_sorted_ids,
                                      cur_device - tp_rank, bytes_sorted_ids, stream));
          CHECK_HIP(hipMemcpyPeerAsync(dst_offsets, cur_device, src_offsets, cur_device - tp_rank,
                                      bytes_offsets, stream));
        }

        pthread_barrier_wait(tp_barrier);

        #ifdef DEBUG
          if (flag) rs_now->sorted_pair_ids->printDebug("rs_now->sorted_pair_ids build offset", tp_rank, 0, stream);
        #endif

        moe_pack_inputs_hip(rs_now->t, rs_now->sorted_pair_ids, rs_now->x_packed, BATCH_SIZE, p->hidden_dim,
                            p->experts_per_token, stream);

        #ifdef DEBUG
          if (flag) rs_now->sorted_pair_ids->printDebug("rs_now->sorted_pair_ids pack inputs", tp_rank, 0, stream);
        #endif

        int max_rows = moe_get_max_rows_per_expert_hip(rs_now->expert_offsets, p->n_experts, stream);
        moe_mlp1_forward_hip(rs_now->x_packed, weights_now->w_mlp1, weights_now->b_mlp1, rs_now->expert_offsets,
                            rs_now->mlp1_out, 1ll * l, p->n_experts, p->hidden_dim, p->intermediate_dim / TP,
                            max_rows, total_pairs, stream);

        #ifdef DEBUG
          if (flag) rs_now->expert_offsets->printDebug("rs_now->expert_offsets", tp_rank, 0, stream);
        #endif

        moe_swiglu_hip(rs_now->mlp1_out, rs_now->gate_up, BATCH_SIZE, p->experts_per_token, p->intermediate_dim / TP,
                      p->swiglu_limit, stream);

        #ifdef DEBUG
          if (flag) rs_now->gate_up->printDebug("rs_now->gate_up", tp_rank, 0, stream);
        #endif

        moe_mlp2_forward_hip(rs_now->gate_up, weights_now->w_mlp2, weights_now->b_mlp2, rs_now->expert_offsets, rs_now->tb3,
                            tp_rank == 0, 1ll * l, p->n_experts, p->intermediate_dim / TP, p->hidden_dim, max_rows,
                            total_pairs, stream);

        #ifdef DEBUG
          if (flag) rs_now->tb3->printDebug("rs_now->tb3", tp_rank, 0, stream);
        #endif

        reduce_tb3(rs_now, rs_leader, tp_rank, cur_device, tp_barrier, 
                    stream, tp_ready, tp_finish);

        #ifdef DEBUG
          if (flag) rs_now->tb3->printDebug("rs_now->tb3 after", tp_rank, 0, stream);
        #endif

        moe_scatter_aggregate_hip(rs_now->tb3, rs_now->sorted_pair_ids, rs_now->topk_v, rs_now->e_agg,
                                  rs_now->expert_offsets, p->hidden_dim, p->experts_per_token, p->n_experts,
                                  max_rows, stream);
        
        #ifdef DEBUG
          if (flag) rs_now->e_agg->printDebug("rs_now->e_agg", tp_rank, 0, stream);
        #endif
        
        // residual connection
        add_vector_batched(rs_now->x, rs_now->e_agg, false, false, false, stream);  // equals residual add
        
        #ifdef DEBUG
          if (flag) rs_now->x->printDebug("rs_now->x", tp_rank, 0, stream);
        #endif
      }
      
      CHECK_HIP(hipEventRecord(pp_sync, stream));  
    */
  }
  
  for (int l = 0; l < p->n_layers / PP; l++) {
    #ifdef DEBUG
      if (flag) {
        printf("Layer l=%d running...\n", l);
        fflush(stdout);
        rs_now->x->printDebug("rs_now->x", tp_rank, 0, stream);
      }
    #endif

    // attention rmsnorm
    rmsnorm_batched(rs_now->x, weights_now->rms_attn_w, rs_now->t, 1ll * l, false, false, 1e-5f, stream);

    #ifdef DEBUG
      if (flag) rs_now->t->printDebug("rs_now->t", tp_rank, 0, stream);
    #endif

    // key and value point to the kv cache
    long long loff = 1ll * l * loff_one;  // kv cache layer offset

    // QKV projection
    qkv_gemm_batched_v2(rs_now->t, weights_now->w_qkv, weights_now->b_qkv, rs_now->qkv, 1ll * l, false,
                        false, stream);  // This kernel diverges the most

    #ifdef DEBUG
      if (flag) rs_now->qkv->printDebug("rs_now->qkv", tp_rank, 0, stream);
    #endif

    // // Separate q, k, v + RoPE + Store k, v in cache
    qkv_split_rope_fused(rs_now->qkv, rs_now->q, rs_now->key_cache, rs_now->value_cache, rs_now->cos_tensor,
                        rs_now->sin_tensor, p->head_dim, p->n_attn_heads, p->n_kv_heads, pos, l, stream);

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
    single_query_attn_batched(rs_now->q, rs_now->key_cache, rs_now->value_cache, rs_now->mask, weights_now->attn_sinks,
                              rs_now->tb, p->head_dim, p->n_attn_heads, kv_mul, kv_dim, p->seq_len,
                              p->sliding_window, pos, 1ll * l, false, false, false, false, false, stream);

    #ifdef DEBUG
      if (flag) rs_now->tb->printDebug("rs_now->tb", tp_rank, 0, stream);
    #endif

    // final matmul to get the output of the attention
    attn_out_project_batched_v2(rs_now->tb, weights_now->w_o, weights_now->b_o, rs_now->tb2, 1ll * l, false, false, stream);

    #ifdef DEBUG
      if (flag) rs_now->tb2->printDebug("rs_now->tb2", tp_rank, 0, stream);
    #endif

    // residual connection back into x + ffn rmsnorm
    residual_rmsnorm_batched(rs_now->tb2, rs_now->x, weights_now->rms_ffn_w, rs_now->t, 1ll * l, 1e-5f, stream);

    #ifdef DEBUG
      if (flag) rs_now->x->printDebug("rs_now->x", tp_rank, 0, stream);
    #endif

    // MoE routing
    router_gemm_batched(weights_now->w_router, rs_now->t, weights_now->b_router, rs_now->router_score, 1ll * l,
                        false, false, stream);

    #ifdef DEBUG
      if (flag) rs_now->router_score->printDebug("rs_now->router_score", tp_rank, 0, stream);
    #endif

    // Select top-k experts
    topk_softmax_batched(rs_now->router_score, rs_now->topk_v, rs_now->topk_i, false, false, false, stream);

    #ifdef DEBUG
      if (flag) rs_now->topk_v->printDebug("rs_now->topk_v", tp_rank, 0, stream);
    #endif

    int total_pairs = BATCH_SIZE * p->experts_per_token;
    moe_init_buffers_hip(rs_now->e_agg, rs_now->mlp1_out, rs_now->gate_up, rs_now->tb3, rs_now->sorted_pair_ids,
                        rs_now->expert_offsets, rs_now->x_packed, BATCH_SIZE, p->hidden_dim, stream);

    #ifdef DEBUG
      if (flag) rs_now->sorted_pair_ids->printDebug("rs_now->sorted_pair_ids", tp_rank, 0, stream);
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

      CHECK_HIP(hipMemcpyPeerAsync(dst_sorted_ids, cur_device, src_sorted_ids,
                                  cur_device - tp_rank, bytes_sorted_ids, stream));
      CHECK_HIP(hipMemcpyPeerAsync(dst_offsets, cur_device, src_offsets, cur_device - tp_rank,
                                  bytes_offsets, stream));
    }

    pthread_barrier_wait(tp_barrier);

    #ifdef DEBUG
      if (flag) rs_now->sorted_pair_ids->printDebug("rs_now->sorted_pair_ids build offset", tp_rank, 0, stream);
    #endif

    moe_pack_inputs_hip(rs_now->t, rs_now->sorted_pair_ids, rs_now->x_packed, BATCH_SIZE, p->hidden_dim,
                        p->experts_per_token, stream);

    #ifdef DEBUG
      if (flag) rs_now->sorted_pair_ids->printDebug("rs_now->sorted_pair_ids pack inputs", tp_rank, 0, stream);
    #endif

    int max_rows = moe_get_max_rows_per_expert_hip(rs_now->expert_offsets, p->n_experts, stream);
    moe_mlp1_forward_hip(rs_now->x_packed, weights_now->w_mlp1, weights_now->b_mlp1, rs_now->expert_offsets,
                        rs_now->mlp1_out, 1ll * l, p->n_experts, p->hidden_dim, p->intermediate_dim / TP,
                        max_rows, total_pairs, stream);

    #ifdef DEBUG
      if (flag) rs_now->expert_offsets->printDebug("rs_now->expert_offsets", tp_rank, 0, stream);
    #endif

    moe_swiglu_hip(rs_now->mlp1_out, rs_now->gate_up, BATCH_SIZE, p->experts_per_token, p->intermediate_dim / TP,
                  p->swiglu_limit, stream);

    #ifdef DEBUG
      if (flag) rs_now->gate_up->printDebug("rs_now->gate_up", tp_rank, 0, stream);
    #endif

    moe_mlp2_forward_hip(rs_now->gate_up, weights_now->w_mlp2, weights_now->b_mlp2, rs_now->expert_offsets, rs_now->tb3,
                        tp_rank == 0, 1ll * l, p->n_experts, p->intermediate_dim / TP, p->hidden_dim, max_rows,
                        total_pairs, stream);

    #ifdef DEBUG
      if (flag) rs_now->tb3->printDebug("rs_now->tb3", tp_rank, 0, stream);
    #endif

    reduce_tb3(rs_now, rs_leader, tp_rank, cur_device, tp_barrier, 
                stream, tp_ready, tp_finish);

    #ifdef DEBUG
      if (flag) rs_now->tb3->printDebug("rs_now->tb3 after", tp_rank, 0, stream);
    #endif

    moe_scatter_aggregate_hip(rs_now->tb3, rs_now->sorted_pair_ids, rs_now->topk_v, rs_now->e_agg,
                              rs_now->expert_offsets, p->hidden_dim, p->experts_per_token, p->n_experts,
                              max_rows, stream);
    
    #ifdef DEBUG
      if (flag) rs_now->e_agg->printDebug("rs_now->e_agg", tp_rank, 0, stream);
    #endif
    
    // residual connection
    add_vector_batched(rs_now->x, rs_now->e_agg, false, false, false, stream);  // equals residual add
    
    #ifdef DEBUG
      if (flag) rs_now->x->printDebug("rs_now->x", tp_rank, 0, stream);
    #endif
  }

  if ((pp_rank + 1) % PP == 0) {
    // final rmsnorm
    rmsnorm_batched(rs_now->x, weights_now->rms_out_w, rs_now->x, 0ll, false, false, 1e-5f, stream);

    // classifier into logits
    classifier_gemm_batched_v2(weights_now->out, rs_now->x, rs_now->logits, false, true, stream);
    
    #ifdef DEBUG
      if (flag) {
        printf("Finish DEBUG\n");
        fflush(stdout);
        exit(1);
      }
    #endif

    return rs_now->logits->buf;
  } else {
    CHECK_HIP(hipEventRecord(pp_sync, stream)); 

    OurRunState *rs_new = &rs[cur_device + TP];
    hipStream_t stream_new = total_streams[cur_device + TP];
    rs_new->pipeline_each->enqueueElem(
      rs_now->x, cur_device, pp_sync, stream_new
    );

    return nullptr;
  }
}

#endif


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
