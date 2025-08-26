#include "../include/parallel.hpp"
#include "../include/layer.hpp"
#include "../include/layer_hip.hpp"
#include <iostream>
#include <algorithm>

void all_reduce(Context *ctx, int gpu_id, Tensor *data);
void all_to_all(Context *ctx, int gpu_id, Tensor *send_buffer, Tensor *recv_buffer);

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
    this->pipeline_buffers[i] = new Tensor({1, (size_t)p->hidden_dim, this->streams[i]});
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
    CHECK_HIP(hipStreamDestroy(this->streams[i]));
    CHECK_HIP(hipEventDestroy(this->events[i]));

    // free run states and weights

    delete this->cos_tensor[i];
    delete this->sin_tensor[i];
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
  int ep_rank = ctx->comm_groups[local_gpu_id].local_rank;
  int tp_rank = ctx->comm_groups[local_gpu_id].tp_rank;
  int layers_per_stage = p->n_layers / PP;

  size_t qkv_layer_size =
    ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * (size_t)p->head_dim;
  size_t attn_out_layer_size = (size_t)p->n_attn_heads * p->head_dim;

  weights->token_embedding_table =
    new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->token_embedding_table, stream);

  weights->rms_attn_w =
    new Tensor({(size_t)layers_per_stage * p->hidden_dim},
               w->rms_attn_w + 1ll * pp_rank * layers_per_stage * p->hidden_dim, stream);
  weights->rms_ffn_w =
    new Tensor({(size_t)layers_per_stage * p->hidden_dim},
               w->rms_ffn_w + 1ll * pp_rank * layers_per_stage * p->hidden_dim, stream);

  weights->w_qkv = new Tensor(
    {(size_t)layers_per_stage, qkv_layer_size, (size_t)p->hidden_dim},
    w->w_qkv + 1ll * pp_rank * layers_per_stage * qkv_layer_size * p->hidden_dim, stream);
  weights->b_qkv = new Tensor({(size_t)layers_per_stage, qkv_layer_size},
                              w->b_qkv + 1ll * pp_rank * layers_per_stage * qkv_layer_size, stream);

  weights->w_o = new Tensor(
    {(size_t)layers_per_stage, (size_t)p->hidden_dim, attn_out_layer_size},
    w->w_o + 1ll * pp_rank * layers_per_stage * p->hidden_dim * attn_out_layer_size, stream);

  weights->b_o = new Tensor({(size_t)layers_per_stage, (size_t)p->hidden_dim},
                            w->b_o + 1ll * layers_per_stage * p->hidden_dim, stream);

  weights->attn_sinks =
    new Tensor({(size_t)layers_per_stage, (size_t)p->n_attn_heads},
               w->attn_sinks + 1ll * pp_rank * layers_per_stage * p->n_attn_heads, stream);
  weights->w_router = new Tensor(
    {(size_t)layers_per_stage, (size_t)p->hidden_dim, (size_t)p->n_experts},
    w->w_router + 1ll * pp_rank * layers_per_stage * p->hidden_dim * p->n_experts, stream);
  weights->b_router =
    new Tensor({(size_t)layers_per_stage, (size_t)p->n_experts},
               w->b_router + 1ll * pp_rank * layers_per_stage * p->n_experts, stream);

  size_t sharded_intermediate_dim = p->intermediate_dim / TP;
  int experts_per_gpu = p->n_experts / REPLICA_SIZE;
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
    new Tensor({(size_t)layers_per_stage, (size_t)experts_per_gpu, 2 * (size_t)p->intermediate_dim},
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
  }

  for (int l = 0; l < p->n_layers; l++) {
    int current_stage_idx = l / layer_per_stages;
    int local_layer_idx = l % layer_per_stages;
    int start_local_idx = current_stage_idx * stage0_size;
    int end_local_idx = start_local_idx + stage0_size;

    if (current_stage_idx > 0) {
      int prev_stage_idx = current_stage_idx - 1;
      int src_gpu_local_idx = prev_stage_idx * stage0_size;
      int src_gpu_global_id = ctx->gpu_ids[src_gpu_local_idx];

      CHECK_HIP(hipSetDevice(src_gpu_global_id));
      hipEventRecord(ctx->events[src_gpu_local_idx], ctx->streams[src_gpu_local_idx]);

      for (int i = start_local_idx; i < end_local_idx; ++i) {
        int dst_gpu_global_id = ctx->gpu_ids[i];
        CHECK_HIP(hipSetDevice(dst_gpu_global_id));
        hipStreamWaitEvent(ctx->streams[i], ctx->events[src_gpu_local_idx], 0);
        CHECK_HIP(hipMemcpyPeerAsync(
          ctx->run_state[i]->x->d_buf, dst_gpu_global_id,
          ctx->pipeline_buffers[prev_stage_idx]->d_buf, src_gpu_global_id,
          ctx->run_state[i]->x->num_elem() * ctx->run_state[i]->x->get_dtype_size(),
          ctx->streams[i]));
      }
    }

    for (int i = start_local_idx; i < end_local_idx; i++) {
      CHECK_HIP(hipSetDevice(ctx->gpu_ids[i]));
      OurRunState *s = ctx->run_state[i];
      OurTransformerWeights *w = ctx->weights[i];

      rmsnorm(s->x, w->rms_attn_w, s->t, local_layer_idx, false, false, 1e-5f, ctx->streams[i]);
      qkv_gemm(s->t, w->w_qkv, w->b_qkv, s->qkv, local_layer_idx, false, false, ctx->streams[i]);
      qkv_split_rope(s->qkv, s->q, s->k, s->v, ctx->cos_tensor[i], ctx->sin_tensor[i], p->head_dim,
                     p->n_attn_heads, p->n_kv_heads, pos, false, false, false, false,
                     ctx->streams[i]);

      long long kv_cache_offset =
        (long long)local_layer_idx * p->seq_len * p->n_kv_heads * p->head_dim +
        (long long)pos * p->n_kv_heads * p->head_dim;
      memcpy_tensor(s->key_cache, s->k, kv_cache_offset, 0, s->k->num_elem(), false, true,
                    ctx->streams[i]);
      memcpy_tensor(s->value_cache, s->v, kv_cache_offset, 0, s->v->num_elem(), false, true,
                    ctx->streams[i]);

      single_query_attn(s->q, s->key_cache, s->value_cache, s->mask, w->attn_sinks, s->tb,
                        p->head_dim, p->n_attn_heads, p->n_attn_heads / p->n_kv_heads,
                        p->n_kv_heads * p->head_dim, p->seq_len, p->sliding_window, pos,
                        local_layer_idx, false, false, false, false, false, ctx->streams[i]);
      attn_out_project(s->tb, w->w_o, w->b_o, s->tb2, local_layer_idx, false, false,
                       ctx->streams[i]);
      add_vector(s->x, s->tb2, false, false, false, ctx->streams[i]);
    }

    for (int i = start_local_idx; i < end_local_idx; i++) {
      CHECK_HIP(hipSetDevice(ctx->gpu_ids[i]));
      OurRunState *s = ctx->run_state[i];
      OurTransformerWeights *w = ctx->weights[i];
      rmsnorm(s->x, w->rms_ffn_w, s->t, local_layer_idx, false, false, 1e-5f, ctx->streams[i]);
      router_gemm(s->t, w->w_router, w->b_router, s->router_score, local_layer_idx, false, false,
                  ctx->streams[i]);
    }

    // --- Pipeline Send ---
    if ((l + 1) % layer_per_stages == 0 && current_stage_idx < PP - 1) {
      // Chỉ cần một GPU trong stage gửi là đủ
      int src_gpu_local_idx = start_local_idx;
      int src_gpu_global_id = ctx->gpu_ids[src_gpu_local_idx];
      CHECK_HIP(hipSetDevice(src_gpu_global_id));
      CHECK_HIP(hipMemcpyAsync(ctx->pipeline_buffers[current_stage_idx]->d_buf,
                               ctx->run_state[src_gpu_local_idx]->x->d_buf,
                               ctx->run_state[src_gpu_local_idx]->x->num_elem() *
                                 ctx->run_state[src_gpu_local_idx]->x->get_dtype_size(),
                               hipMemcpyDeviceToDevice, ctx->streams[src_gpu_local_idx]));
    }
  }

  int last_gpu_local_idx = REPLICA_SIZE - 1;
  int last_gpu_global_id = ctx->gpu_ids[last_gpu_local_idx];

  OurRunState *s_final = ctx->run_state[last_gpu_local_idx];
  OurTransformerWeights *w_final = ctx->weights[last_gpu_local_idx];

  rmsnorm(s_final->x, w_final->rms_out_w, s_final->x, 0, false, false, 1e-5f,
          ctx->streams[last_gpu_local_idx]);
  classifier_gemm(s_final->x, w_final->out, s_final->logits, false, true,
                  ctx->streams[last_gpu_local_idx]);

  CHECK_HIP(hipStreamSynchronize(ctx->streams[last_gpu_local_idx]));
  s_final->logits->from_device(ctx->streams[last_gpu_local_idx]);

  return s_final->logits->buf;
}
