#include "../include/parallel.hpp"
#include "../include/layer.hpp"
#include "../include/layer_hip.hpp"
#include <iostream>
#include <algorithm>

// __global__ void add_arrays_kernel(float *local, const float *remote, size_t n) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < n) {
//     local[idx] += remote[idx];
//   }
// }

// void all_reduce(Context *ctx, int local_idx, Tensor *data) {
//   CommGroups *cg = &ctx->comm_groups[local_idx];
//   if (cg->tp_group.size() <= 1)
//     return;

//   int gpu_id = ctx->gpu_ids[local_idx];
//   CHECK_HIP(hipSetDevice(gpu_id));

//   size_t count = data->num_elem();
//   float *d_ptr = (float *)data->d_buf;

//   // Simple all-reduce using peer-to-peer copies and addition
//   for (int peer_idx = 0; peer_idx < ctx->gpu_ids.size(); peer_idx++) {
//     if (ctx->comm_groups[peer_idx].pp_rank == cg->pp_rank &&
//         ctx->comm_groups[peer_idx].tp_rank != cg->tp_rank) {
//       int peer_gpu = ctx->gpu_ids[peer_idx];
//       float *peer_ptr = (float *)ctx->run_state[peer_idx]->e_agg->d_buf;

//       // Create temporary buffer for peer data
//       float *temp_buf;
//       CHECK_HIP(hipMalloc(&temp_buf, count * sizeof(float)));

//       // Copy from peer
//       CHECK_HIP(hipMemcpyPeerAsync(temp_buf, gpu_id, peer_ptr, peer_gpu, count * sizeof(float),
//                                    ctx->streams[local_idx]));
//       CHECK_HIP(hipStreamSynchronize(ctx->streams[local_idx]));

//       // Add to local buffer using the kernel
//       dim3 block(256);
//       dim3 grid((count + block.x - 1) / block.x);
//       add_arrays_kernel<<<grid, block, 0, ctx->streams[local_idx]>>>(d_ptr, temp_buf, count);

//       CHECK_HIP(hipFree(temp_buf));
//     }
//   }

//   // Broadcast the result back to all GPUs in TP group
//   for (int peer_idx = 0; peer_idx < ctx->gpu_ids.size(); peer_idx++) {
//     if (ctx->comm_groups[peer_idx].pp_rank == cg->pp_rank &&
//         ctx->comm_groups[peer_idx].tp_rank != cg->tp_rank) {
//       int peer_gpu = ctx->gpu_ids[peer_idx];
//       float *peer_ptr = (float *)ctx->run_state[peer_idx]->e_agg->d_buf;

//       CHECK_HIP(hipMemcpyPeerAsync(peer_ptr, peer_gpu, d_ptr, gpu_id, count * sizeof(float),
//                                    ctx->streams[local_idx]));
//     }
//   }
//   CHECK_HIP(hipStreamSynchronize(ctx->streams[local_idx]));
// }

void Context::init(Transformer *transformer, const std::vector<int> &assigned_gpu_ids) {
  this->transformer = transformer;
  this->gpu_ids = assigned_gpu_ids;
  Config *p = &transformer->config;

  if (this->gpu_ids.size() != REPLICA_SIZE) {
    std::cerr << "Error: ParallelContext requires exactly " << REPLICA_SIZE << " GPUs."
              << std::endl;
    exit(1);
  }

  for (int i = 0; i < REPLICA_SIZE; i++) {
    run_state[i] = new OurRunState();
    weights[i] = new OurTransformerWeights();
  }

  for (int i = 0; i < REPLICA_SIZE; i++) {
    int gpu_id = this->gpu_ids[i];
    CHECK_HIP(hipSetDevice(gpu_id));
    CHECK_HIP(hipStreamCreate(&this->streams[i]));
    CHECK_HIP(hipEventCreate(&this->events[i]));
  }

  for (int i = 0; i < REPLICA_SIZE; i++) {
    int src_gpu_id = this->gpu_ids[i];
    CHECK_HIP(hipSetDevice(src_gpu_id));
    for (int j = 0; j < REPLICA_SIZE; j++) {
      if (i != j) {
        int dst_gpu_id = this->gpu_ids[j];
        CHECK_HIP(hipDeviceEnablePeerAccess(dst_gpu_id, 0));
      }
    }
  }

  for (int i = 0; i < PP - 1; i++) {
    int local_owner_idx = (i + 1) * TP;
    int global_owner_id = this->gpu_ids[local_owner_idx];
    CHECK_HIP(hipSetDevice(global_owner_id));
    this->pipeline_buffers[i] =
      new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, this->streams[local_owner_idx]);
  }

  for (int i = 0; i < REPLICA_SIZE; i++) {
    int gpu_id = this->gpu_ids[i];
    CHECK_HIP(hipSetDevice(gpu_id));

    CommGroups *cg = &this->comm_groups[i];
    cg->local_rank = i;
    cg->pp_rank = i / TP;
    cg->tp_rank = i % TP;

    int tp_start_local_idx = cg->pp_rank * TP;
    for (int j = 0; j < TP; ++j) {
      cg->tp_group.push_back(this->gpu_ids[tp_start_local_idx + j]);  // Global rank
    }

    for (int j = 0; j < PP; ++j) {
      cg->pp_group.push_back(this->gpu_ids[j * TP + cg->tp_rank]);  // Global rank
    }

    our_init_run_state_120b(this, i);
    our_init_weights_120b(this, i);

    this->cos_tensor[i] =
      new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2}, this->streams[i]);
    this->sin_tensor[i] =
      new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2}, this->streams[i]);

    RopePrecomputeCS(p, this->cos_tensor[i], this->sin_tensor[i], this->streams[i]);
  }

  for (int i = 0; i < REPLICA_SIZE; i++) {
    int gpu_id = this->gpu_ids[i];
    CHECK_HIP(hipSetDevice(gpu_id));
    CHECK_HIP(hipStreamSynchronize(this->streams[i]));
  }

  printf("Context initialized successully\n");
  fflush(stdout);
}

void Context::destroy() {
  for (int i = 0; i < REPLICA_SIZE; i++) {
    int gpu_id = this->gpu_ids[i];
    CHECK_HIP(hipSetDevice(gpu_id));
    if (this->streams[i])
      CHECK_HIP(hipStreamSynchronize(this->streams[i]));
  }

  for (int i = 0; i < REPLICA_SIZE; i++) {
    int gpu_id = this->gpu_ids[i];
    CHECK_HIP(hipSetDevice(gpu_id));

    // free run states (chỉ khi bạn đã new trong init)
    if (this->run_state[i]) {
      auto rs = this->run_state[i];
      delete rs->x;
      delete rs->t;
      delete rs->tb;
      delete rs->tb2;
      delete rs->tb3;
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
      delete this->run_state[i];
      this->run_state[i] = nullptr;
    }

    if (this->weights[i]) {
      auto w = this->weights[i];
      delete w->token_embedding_table;
      delete w->rms_attn_w;
      delete w->rms_ffn_w;
      delete w->w_qkv;
      delete w->b_qkv;
      delete w->w_o;
      delete w->b_o;
      delete w->attn_sinks;
      delete w->w_router;
      delete w->b_router;
      delete w->w_mlp1;
      delete w->b_mlp1;
      delete w->w_mlp2;
      delete w->b_mlp2;
      delete w->out;
      delete w->rms_out_w;
      delete this->weights[i];
      this->weights[i] = nullptr;
    }

    delete this->cos_tensor[i];
    delete this->sin_tensor[i];

    CHECK_HIP(hipStreamDestroy(this->streams[i]));
    CHECK_HIP(hipEventDestroy(this->events[i]));
  }

  for (int i = 0; i < PP - 1; i++) {
    delete this->pipeline_buffers[i];
  }
}

void our_init_run_state_120b(Context *ctx, int local_gpu_id) {
  Config *p = &ctx->transformer->config;
  OurRunState *rs = ctx->run_state[local_gpu_id];
  RunState *s = &ctx->transformer->state;
  hipStream_t stream = ctx->streams[local_gpu_id];

  rs->x = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, s->x, stream);

  rs->t = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, s->t, stream);
  rs->tb = new Tensor({BATCH_SIZE, (size_t)p->head_dim * p->n_attn_heads}, s->tb, stream);
  rs->tb2 = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, s->tb2, stream);

  rs->tb3 = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->hidden_dim}, stream);

  rs->router_score = new Tensor({BATCH_SIZE, (size_t)p->n_experts}, s->router_score, stream);
  rs->topk_v = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token}, s->topk_v, stream);
  rs->topk_i = new TensorI32({BATCH_SIZE, (size_t)p->experts_per_token}, s->topk_i, stream);

  // rs->mlp1_out = new Tensor({2 * (size_t)p->intermediate_dim}, s->mlp1_out);
  rs->mlp1_out =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, 2 * (size_t)p->intermediate_dim}, stream);

  // rs->gate = new Tensor({(size_t)p->intermediate_dim}, s->gate);
  // rs->up = new Tensor({(size_t)p->intermediate_dim}, s->up);
  rs->gate =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, stream);
  rs->up =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, stream);

  // rs->gate_up = new Tensor({(size_t)p->intermediate_dim}, s->gate_up);
  rs->gate_up =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, stream);

  rs->e_agg = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, s->e_agg, stream);
  rs->qkv =
    new Tensor({BATCH_SIZE, ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim},
               s->qkv, stream);
  rs->q = new Tensor({BATCH_SIZE, (size_t)p->n_attn_heads * p->head_dim}, s->q, stream);
  rs->k = new Tensor({BATCH_SIZE, (size_t)p->n_kv_heads * p->head_dim}, stream);
  rs->v = new Tensor({BATCH_SIZE, (size_t)p->n_kv_heads * p->head_dim}, stream);
  rs->att = new Tensor({BATCH_SIZE, (size_t)p->n_attn_heads, (size_t)p->seq_len}, s->att, stream);
  rs->logits = new Tensor({BATCH_SIZE, (size_t)p->vocab_size}, s->logits, stream);

  rs->key_cache = new Tensor(
    {BATCH_SIZE, (size_t)p->n_layers, (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim},
    s->key_cache, stream);
  rs->value_cache = new Tensor(
    {BATCH_SIZE, (size_t)p->n_layers, (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim},
    s->value_cache, stream);

  rs->mask = new Tensor({BATCH_SIZE, (size_t)p->seq_len, (size_t)p->seq_len}, s->mask, stream);
}

void our_init_weights_120b(Context *ctx, int local_gpu_id) {
  Config *p = &ctx->transformer->config;
  TransformerWeights *w = &ctx->transformer->weights;
  OurTransformerWeights *weights = ctx->weights[local_gpu_id];
  hipStream_t stream = ctx->streams[local_gpu_id];

  int pp_rank = ctx->comm_groups[local_gpu_id].pp_rank;
  int tp_rank = ctx->comm_groups[local_gpu_id].tp_rank;
  int ep_rank = 0;
  int layers_per_stage = p->n_layers / PP;

  size_t qkv_layer_size =
    ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * (size_t)p->head_dim;
  size_t attn_out_layer_size = (size_t)p->n_attn_heads * p->head_dim;

  // Embedding
  weights->token_embedding_table =
    new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->token_embedding_table, stream);

  // RMS
  weights->rms_attn_w =
    new Tensor({(size_t)layers_per_stage * p->hidden_dim},
               w->rms_attn_w + 1ll * pp_rank * layers_per_stage * p->hidden_dim, stream);
  weights->rms_ffn_w =
    new Tensor({(size_t)layers_per_stage * p->hidden_dim},
               w->rms_ffn_w + 1ll * pp_rank * layers_per_stage * p->hidden_dim, stream);

  // QKV / bias
  weights->w_qkv = new Tensor(
    {(size_t)layers_per_stage, qkv_layer_size, (size_t)p->hidden_dim},
    w->w_qkv + 1ll * pp_rank * layers_per_stage * qkv_layer_size * p->hidden_dim, stream);
  weights->b_qkv = new Tensor({(size_t)layers_per_stage, qkv_layer_size},
                              w->b_qkv + 1ll * pp_rank * layers_per_stage * qkv_layer_size, stream);

  // O proj + bias
  weights->w_o = new Tensor(
    {(size_t)layers_per_stage, (size_t)p->hidden_dim, attn_out_layer_size},
    w->w_o + 1ll * pp_rank * layers_per_stage * p->hidden_dim * attn_out_layer_size, stream);

  weights->b_o = new Tensor({(size_t)layers_per_stage, (size_t)p->hidden_dim},
                            w->b_o + 1ll * pp_rank * layers_per_stage * p->hidden_dim, stream);

  // Attn sinks + Router
  weights->attn_sinks =
    new Tensor({(size_t)layers_per_stage, (size_t)p->n_attn_heads},
               w->attn_sinks + 1ll * pp_rank * layers_per_stage * p->n_attn_heads, stream);

  // w_router
  weights->w_router = new Tensor(
    {(size_t)layers_per_stage, (size_t)p->n_experts, (size_t)p->hidden_dim},
    w->w_router + 1ll * pp_rank * layers_per_stage * p->n_experts * p->hidden_dim, stream);
  weights->b_router =
    new Tensor({(size_t)layers_per_stage, (size_t)p->n_experts},
               w->b_router + 1ll * pp_rank * layers_per_stage * p->n_experts, stream);

  // MoE MLP sharding: EP in stage, TP with inter-dim
  size_t sharded_intermediate_dim = p->intermediate_dim / TP;
  int experts_per_gpu = p->n_experts;

  long long mlp1_full_layer_size = 1ll * p->n_experts * 2 * p->intermediate_dim * p->hidden_dim;
  long long mlp2_full_layer_size = 1ll * p->n_experts * p->hidden_dim * p->intermediate_dim;

  float *w_mlp1_stage_start = w->w_mlp1 + 1ll * pp_rank * layers_per_stage * mlp1_full_layer_size;
  float *w_mlp1_start_shard =
    w_mlp1_stage_start + (ep_rank * experts_per_gpu * 2 * p->intermediate_dim * p->hidden_dim) +
    (tp_rank * 2 * sharded_intermediate_dim * p->hidden_dim);

  weights->w_mlp1 = new Tensor({(size_t)layers_per_stage, (size_t)experts_per_gpu,
                                2 * sharded_intermediate_dim, (size_t)p->hidden_dim},
                               w_mlp1_start_shard, stream, DType::BF16);

  float *b_mlp1_stage_start =
    w->b_mlp1 + 1ll * pp_rank * layers_per_stage * p->n_experts * 2 * p->intermediate_dim;
  float *b_mlp1_start_shard = b_mlp1_stage_start +
                              (ep_rank * experts_per_gpu * 2 * p->intermediate_dim) +
                              (tp_rank * 2 * sharded_intermediate_dim);

  weights->b_mlp1 =
    new Tensor({(size_t)layers_per_stage, (size_t)experts_per_gpu, 2 * sharded_intermediate_dim},
               b_mlp1_start_shard, stream, DType::BF16);

  float *w_mlp2_stage_start = w->w_mlp2 + 1ll * pp_rank * layers_per_stage * mlp2_full_layer_size;
  float *w_mlp2_start_shard = w_mlp2_stage_start +
                              (ep_rank * experts_per_gpu * p->hidden_dim * p->intermediate_dim) +
                              (tp_rank * p->hidden_dim * sharded_intermediate_dim);

  weights->w_mlp2 = new Tensor({(size_t)layers_per_stage, (size_t)experts_per_gpu,
                                (size_t)p->hidden_dim, sharded_intermediate_dim},
                               w_mlp2_start_shard, stream, DType::BF16);

  float *b_mlp2_stage_start =
    w->b_mlp2 + 1ll * pp_rank * layers_per_stage * p->n_experts * p->hidden_dim;
  float *b_mlp2_start_shard = b_mlp2_stage_start + (ep_rank * experts_per_gpu * p->hidden_dim);

  weights->b_mlp2 =
    new Tensor({(size_t)layers_per_stage, (size_t)experts_per_gpu, (size_t)p->hidden_dim},
               b_mlp2_start_shard, stream, DType::BF16);

  weights->out = new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->out, stream);
  weights->rms_out_w = new Tensor({(size_t)p->hidden_dim}, w->rms_out_w, stream);
}

float *forward_gpu_120b(Context *ctx, int token, int pos) {
  Config *p = &ctx->transformer->config;
  int layer_per_stages = p->n_layers / PP;

  int stage0_size = TP;
  for (int i = 0; i < stage0_size; i++) {
    int gpu_id = ctx->gpu_ids[i];
    CHECK_HIP(hipSetDevice(gpu_id));
    embedding_lookup(ctx->weights[i]->token_embedding_table, token, ctx->run_state[i]->x, false,
                     ctx->streams[i]);
    // printf("gpu_id: %d - Done embedding lookup\n", gpu_id);
  }

  for (int l = 0; l < p->n_layers; l++) {
    int current_stage_idx = l / layer_per_stages;
    int local_layer_idx = l % layer_per_stages;
    int start_local_idx = current_stage_idx * stage0_size;
    int end_local_idx = start_local_idx + stage0_size;

    // printf("==== stage: %d - layer: %d ====\n", current_stage_idx, l);

    if (current_stage_idx > 0 && local_layer_idx == 0) {
      int leader_idx = start_local_idx;  // leader của stage hiện tại
      int leader_global_id = ctx->gpu_ids[leader_idx];

      CHECK_HIP(hipSetDevice(leader_global_id));
      CHECK_HIP(hipStreamWaitEvent(ctx->streams[leader_idx], ctx->events[leader_idx], 0));

      size_t nbytes =
        ctx->run_state[leader_idx]->x->num_elem() * ctx->run_state[leader_idx]->x->get_dtype_size();

      CHECK_HIP(hipMemcpyAsync(ctx->run_state[leader_idx]->x->d_buf,
                               ctx->pipeline_buffers[current_stage_idx - 1]->d_buf, nbytes,
                               hipMemcpyDeviceToDevice, ctx->streams[leader_idx]));

      for (int i = start_local_idx + 1; i < end_local_idx; i++) {
        int dst_global_id = ctx->gpu_ids[i];
        CHECK_HIP(hipSetDevice(dst_global_id));
        CHECK_HIP(hipMemcpyPeerAsync(ctx->run_state[i]->x->d_buf, dst_global_id,
                                     ctx->run_state[leader_idx]->x->d_buf, leader_global_id, nbytes,
                                     ctx->streams[i]));
      }
      // printf("stage %d - pipeline recv\n", current_stage_idx);
    }

    // Attention layers
    for (int i = start_local_idx; i < end_local_idx; i++) {
      CHECK_HIP(hipSetDevice(ctx->gpu_ids[i]));
      OurRunState *s = ctx->run_state[i];
      OurTransformerWeights *w = ctx->weights[i];

      rmsnorm(s->x, w->rms_attn_w, s->t, local_layer_idx, false, false, 1e-5f, ctx->streams[i]);
      // printf("stage %d - Done rmsnorm 1\n", current_stage_idx);

      qkv_gemm(s->t, w->w_qkv, w->b_qkv, s->qkv, local_layer_idx, false, false, ctx->streams[i]);
      // printf("stage %d - Done qkv_gemm\n", current_stage_idx);

      qkv_split_rope(s->qkv, s->q, s->k, s->v, ctx->cos_tensor[i], ctx->sin_tensor[i], p->head_dim,
                     p->n_attn_heads, p->n_kv_heads, pos, false, false, false, false,
                     ctx->streams[i]);
      // printf("stage %d - Done qkv_split_rope\n", current_stage_idx);

      long long kv_cache_offset =
        (long long)local_layer_idx * p->seq_len * p->n_kv_heads * p->head_dim +
        (long long)pos * p->n_kv_heads * p->head_dim;
      memcpy_tensor(s->key_cache, s->k, kv_cache_offset, 0, s->k->num_elem(), false, true,
                    ctx->streams[i]);
      memcpy_tensor(s->value_cache, s->v, kv_cache_offset, 0, s->v->num_elem(), false, true,
                    ctx->streams[i]);
      // printf("stage %d - Done memcpy_tensor\n", current_stage_idx);

      single_query_attn(s->q, s->key_cache, s->value_cache, s->mask, w->attn_sinks, s->tb,
                        p->head_dim, p->n_attn_heads, p->n_attn_heads / p->n_kv_heads,
                        p->n_kv_heads * p->head_dim, p->seq_len, p->sliding_window, pos,
                        local_layer_idx, false, false, false, false, false, ctx->streams[i]);
      // printf("stage %d - Done single_query_attn\n", current_stage_idx);

      attn_out_project(s->tb, w->w_o, w->b_o, s->tb2, local_layer_idx, false, false,
                       ctx->streams[i]);
      // printf("stage %d - Done attn_out_project\n", current_stage_idx);

      add_vector(s->x, s->tb2, false, false, false, ctx->streams[i]);
      // printf("stage %d - Done add_vector 1\n", current_stage_idx);
    }

    // MoE FFN layers
    for (int i = start_local_idx; i < end_local_idx; i++) {
      hipStream_t stream = ctx->streams[i];
      CHECK_HIP(hipSetDevice(ctx->gpu_ids[i]));
      OurRunState *s = ctx->run_state[i];
      OurTransformerWeights *w = ctx->weights[i];

      rmsnorm(s->x, w->rms_ffn_w, s->t, local_layer_idx, false, false, 1e-5f, ctx->streams[i]);
      // printf("stage %d - Done rmsnorm 2\n", current_stage_idx);

      router_gemm(w->w_router, s->t, w->b_router, s->router_score, local_layer_idx, false, false,
                  ctx->streams[i]);
      // printf("stage %d - Done router_gemm\n", current_stage_idx);

      topk_softmax(s->router_score, s->topk_v, s->topk_i, false, false, false, ctx->streams[i]);
      // printf("stage %d - Done topk_softmax\n", current_stage_idx);

      const int hidden_dim = s->t->shape[1];
      const int inter_dim = w->w_mlp2->shape[3];
      const int k = s->topk_i->num_elem();
      const int num_experts = w->b_mlp2->shape[1];
      const long long layer_offset = (long long)local_layer_idx;
      const long long offset = layer_offset * num_experts;
      const long long inter_hidden = (long long)inter_dim * hidden_dim;

      // Lấy con trỏ thô trên device từ các tensor
      const float *t_ptr = (const float *)s->t->d_buf;
      const bf16 *W1_ptr = (const bf16 *)w->w_mlp1->d_buf + offset * 2 * inter_hidden;
      const bf16 *b1_ptr = (const bf16 *)w->b_mlp1->d_buf + offset * 2 * inter_dim;
      const bf16 *W2_ptr = (const bf16 *)w->w_mlp2->d_buf + offset * inter_hidden;
      const bf16 *b2_ptr = (const bf16 *)w->b_mlp2->d_buf + offset * hidden_dim;
      const int *topk_idx_ptr = s->topk_i->d_buf;
      const float *topk_vals_ptr = (const float *)s->topk_v->d_buf;
      float *mlp1_out_ptr = (float *)s->mlp1_out->d_buf;
      float *tb3_ptr = (float *)s->tb3->d_buf;
      float *gate_up_ptr = (float *)s->gate_up->d_buf;
      float *e_agg_ptr = (float *)s->e_agg->d_buf;

      memset_tensor(s->e_agg, 0, false, true, stream);

      moe_mlp1(W1_ptr, t_ptr, b1_ptr, mlp1_out_ptr, topk_idx_ptr, k, inter_dim, hidden_dim, stream);
      moe_swiglu(mlp1_out_ptr, gate_up_ptr, k, inter_dim, p->swiglu_limit, stream);
      moe_mlp2(W2_ptr, gate_up_ptr, b2_ptr, tb3_ptr, topk_idx_ptr, k, hidden_dim, inter_dim,
               stream);
      moe_agg(tb3_ptr, topk_vals_ptr, e_agg_ptr, k, hidden_dim, stream);

      add_vector(ctx->run_state[i]->x, ctx->run_state[i]->e_agg, false, false, false,
                 ctx->streams[i]);
    }

    // for (int i = start_local_idx; i < end_local_idx; i++) {
    //   CHECK_HIP(hipSetDevice(ctx->gpu_ids[i]));
    //   all_reduce(ctx, i, ctx->run_state[i]->e_agg);
    // }

    // for (int i = start_local_idx; i < end_local_idx; i++) {
    //   CHECK_HIP(hipSetDevice(ctx->gpu_ids[i]));
    //   add_vector(ctx->run_state[i]->x, ctx->run_state[i]->e_agg, false, false, false,
    //              ctx->streams[i]);
    // }

    // --- Pipeline Send ---
    if ((l + 1) % layer_per_stages == 0 && current_stage_idx < PP - 1) {
      int leader_idx = start_local_idx;  // leader stage này
      int leader_global_id = ctx->gpu_ids[leader_idx];

      int next_leader_idx = (current_stage_idx + 1) * stage0_size;
      int next_leader_global_id = ctx->gpu_ids[next_leader_idx];

      size_t nbytes =
        ctx->run_state[leader_idx]->x->num_elem() * ctx->run_state[leader_idx]->x->get_dtype_size();

      CHECK_HIP(hipSetDevice(leader_global_id));
      CHECK_HIP(hipEventRecord(ctx->events[leader_idx], ctx->streams[leader_idx]));

      CHECK_HIP(hipSetDevice(next_leader_global_id));
      CHECK_HIP(hipStreamWaitEvent(ctx->streams[next_leader_idx], ctx->events[leader_idx], 0));

      CHECK_HIP(hipMemcpyPeerAsync(ctx->pipeline_buffers[current_stage_idx]->d_buf,
                                   next_leader_global_id, ctx->run_state[leader_idx]->x->d_buf,
                                   leader_global_id, nbytes, ctx->streams[next_leader_idx]));

      CHECK_HIP(hipEventRecord(ctx->events[next_leader_idx], ctx->streams[next_leader_idx]));
      CHECK_HIP(hipSetDevice(leader_global_id));
      CHECK_HIP(hipStreamWaitEvent(ctx->streams[leader_idx], ctx->events[next_leader_idx], 0));
      // printf("stage %d - Pipeline send\n", current_stage_idx);
    }
  }

  int last_local_idx = REPLICA_SIZE - 1;

  OurRunState *s_final = ctx->run_state[last_local_idx];
  OurTransformerWeights *w_final = ctx->weights[last_local_idx];

  CHECK_HIP(hipSetDevice(ctx->gpu_ids[last_local_idx]));
  rmsnorm(s_final->x, w_final->rms_out_w, s_final->x, 0, false, false, 1e-5f,
          ctx->streams[last_local_idx]);
  classifier_gemm(w_final->out, s_final->x, s_final->logits, false, true,
                  ctx->streams[last_local_idx]);

  CHECK_HIP(hipStreamSynchronize(ctx->streams[last_local_idx]));
  s_final->logits->from_device(ctx->streams[last_local_idx]);

  return s_final->logits->buf;
}
