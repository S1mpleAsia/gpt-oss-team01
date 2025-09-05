#include "../include/parallel.hpp"
#include "../include/layer.hpp"
#include "../include/layer_hip.hpp"
#include "../include/layer_hip_batch.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>

__global__ void add_vectors_kernel(float *dst, const float *src, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] += src[idx];
  }
}

void reduce_broadcast_tb3(Context *ctx, int start_idx, int end_idx, Tensor *tensor) {
  int leader_idx = start_idx;
  int leader_gpu_id = ctx->gpu_ids[leader_idx];
  hipStream_t leader_stream = ctx->streams[leader_idx];
  size_t num_elements = tensor->num_elem();

  CHECK_HIP(hipSetDevice(leader_gpu_id));
  float *temp_buffer = (float *)ctx->run_state[leader_idx]->reduce_temp_buffer->d_buf;

  const int block_size = 256;
  const int grid_size = (num_elements + block_size - 1) / block_size;

  CHECK_HIP(hipStreamWaitEvent(leader_stream, ctx->tp_ready_event[leader_idx], 0));
  for (int i = start_idx + 1; i < end_idx; i++) {
    int src_gpu_id = ctx->gpu_ids[i];

    CHECK_HIP(hipStreamWaitEvent(leader_stream, ctx->tp_ready_event[i], 0));
    CHECK_HIP(hipMemcpyPeerAsync(temp_buffer, leader_gpu_id, ctx->run_state[i]->tb3->d_buf,
                                 src_gpu_id, num_elements * tensor->get_dtype_size(),
                                 leader_stream));

    // CHECK_HIP(hipStreamSynchronize(leader_stream));

    add_vectors_kernel<<<grid_size, block_size, 0, leader_stream>>>((float *)tensor->d_buf,
                                                                    temp_buffer, num_elements);
  }
  // CHECK_HIP(hipStreamSynchronize(leader_stream));
  CHECK_HIP(hipEventRecord(ctx->tp_reduce_done_event[leader_idx], leader_stream));

  for (int i = start_idx + 1; i < end_idx; i++) {
    int dst_gpu_id = ctx->gpu_ids[i];

    CHECK_HIP(hipSetDevice(dst_gpu_id));
    CHECK_HIP(hipStreamWaitEvent(ctx->streams[i], ctx->tp_reduce_done_event[leader_idx], 0));

    CHECK_HIP(hipMemcpyPeerAsync(ctx->run_state[i]->tb3->d_buf, dst_gpu_id, tensor->d_buf,
                                 leader_gpu_id, num_elements * tensor->get_dtype_size(),
                                 ctx->streams[i]));
    // CHECK_HIP(hipEventRecord(ctx->events[i], ctx->streams[i]));
  }

  CHECK_HIP(hipSetDevice(leader_gpu_id));

  // for (int i = start_idx + 1; i < end_idx; i++) {
  //   CHECK_HIP(hipStreamWaitEvent(ctx->streams[leader_idx], ctx->events[i], 0));
  // }

  // CHECK_HIP(hipFree(temp_buffer));
}

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
    CHECK_HIP(hipStreamCreateWithFlags(&this->streams[i], hipStreamNonBlocking));
    CHECK_HIP(hipEventCreate(&this->events[i]));
    CHECK_HIP(hipEventCreate(&this->tp_ready_event[i]));
    CHECK_HIP(hipEventCreate(&this->tp_reduce_done_event[i]));
    CHECK_HIP(hipEventCreate(&this->pipe_recv_ready_event[i]));
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

      delete rs->sorted_pair_ids;
      delete rs->expert_offsets;
      delete rs->x_packed;
      delete rs->reduce_temp_buffer;
      delete this->run_state[i];
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
    }

    delete this->cos_tensor[i];
    delete this->sin_tensor[i];

    CHECK_HIP(hipStreamDestroy(this->streams[i]));
    CHECK_HIP(hipEventDestroy(this->events[i]));
    CHECK_HIP(hipEventDestroy(this->tp_ready_event[i]));
    CHECK_HIP(hipEventDestroy(this->tp_reduce_done_event[i]));
    CHECK_HIP(hipEventDestroy(this->pipe_recv_ready_event[i]));
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

  size_t sharded_intermediate_dim = p->intermediate_dim / TP;
  size_t layers_per_stage = p->n_layers / PP;

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
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, 2 * sharded_intermediate_dim}, stream);

  // rs->gate = new Tensor({(size_t)p->intermediate_dim}, s->gate);
  // rs->up = new Tensor({(size_t)p->intermediate_dim}, s->up);
  rs->gate =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, sharded_intermediate_dim}, stream);
  rs->up = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, sharded_intermediate_dim}, stream);

  // rs->gate_up = new Tensor({(size_t)p->intermediate_dim}, s->gate_up);
  rs->gate_up =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, sharded_intermediate_dim}, stream);

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
    {BATCH_SIZE, (size_t)layers_per_stage, (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim},
    s->key_cache, stream);
  rs->value_cache = new Tensor(
    {BATCH_SIZE, (size_t)layers_per_stage, (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim},
    s->value_cache, stream);

  rs->mask = new Tensor({BATCH_SIZE, (size_t)p->seq_len, (size_t)p->seq_len}, stream);
  size_t single_mask_elems = (size_t)p->seq_len * p->seq_len;

  for (int b = 0; b < BATCH_SIZE; b++) {
    float *dest_ptr = rs->mask->buf + b * single_mask_elems;
    memcpy(dest_ptr, s->mask, single_mask_elems * sizeof(float));
  }
  rs->mask->to_device(stream);

  // MoE buffer
  rs->sorted_pair_ids = new TensorI32({(size_t)BATCH_SIZE * p->experts_per_token}, stream);
  rs->expert_offsets = new TensorI32({(size_t)(p->n_experts + 1)}, stream);
  rs->x_packed =
    new Tensor({(size_t)BATCH_SIZE * p->experts_per_token, (size_t)p->hidden_dim}, stream);

  // rs->cos_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2});
  // rs->sin_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2});

  rs->reduce_temp_buffer =
    new Tensor({(size_t)BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->hidden_dim}, stream);
  // RoPE is computed on Context::init
}

void our_init_weights_120b(Context *ctx, int local_gpu_id) {
  Config *p = &ctx->transformer->config;
  TransformerWeights *w = &ctx->transformer->weights;
  OurTransformerWeights *weights = ctx->weights[local_gpu_id];
  hipStream_t stream = ctx->streams[local_gpu_id];

  int pp_rank = ctx->comm_groups[local_gpu_id].pp_rank;
  int tp_rank = ctx->comm_groups[local_gpu_id].tp_rank;
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
  // weights->w_qkv = new Tensor(
  //   {(size_t)layers_per_stage, qkv_layer_size, (size_t)p->hidden_dim},
  //   w->w_qkv + 1ll * pp_rank * layers_per_stage * qkv_layer_size * p->hidden_dim, stream);

  {
    size_t hidden_dim = p->hidden_dim;

    weights->w_qkv = new Tensor({(size_t)layers_per_stage, hidden_dim, qkv_layer_size}, stream);

    for (size_t l = 0; l < layers_per_stage; l++) {
      for (size_t i = 0; i < qkv_layer_size; i++) {
        for (size_t j = 0; j < hidden_dim; j++) {
          weights->w_qkv->buf[l * (hidden_dim * qkv_layer_size) + j * qkv_layer_size + i] =
            w->w_qkv[1ll * pp_rank * layers_per_stage * qkv_layer_size * hidden_dim +
                     l * qkv_layer_size * hidden_dim + i * hidden_dim + j];
        }
      }
    }

    weights->w_qkv->to_device(stream);
  }

  weights->b_qkv = new Tensor({(size_t)layers_per_stage, qkv_layer_size},
                              w->b_qkv + 1ll * pp_rank * layers_per_stage * qkv_layer_size, stream);

  // O proj + bias
  // weights->w_o = new Tensor(
  //   {(size_t)layers_per_stage, (size_t)p->hidden_dim, attn_out_layer_size},
  //   w->w_o + 1ll * pp_rank * layers_per_stage * p->hidden_dim * attn_out_layer_size, stream);

  {
    size_t hidden_dim = p->hidden_dim;
    weights->w_o = new Tensor({(size_t)layers_per_stage, attn_out_layer_size, hidden_dim}, stream);

    for (size_t l = 0; l < layers_per_stage; l++) {
      for (size_t i = 0; i < hidden_dim; i++) {
        for (size_t j = 0; j < attn_out_layer_size; j++) {
          weights->w_o->buf[l * attn_out_layer_size * hidden_dim + j * hidden_dim + i] =
            w->w_o[1ll * pp_rank * layers_per_stage * hidden_dim * attn_out_layer_size +
                   l * hidden_dim * attn_out_layer_size + i * attn_out_layer_size + j];
        }
      }
    }

    weights->w_o->to_device(stream);
  }

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

  // Sharding weights MLP1
  float *w_mlp1_stage_base = w->w_mlp1 + 1ll * pp_rank * layers_per_stage * experts_per_gpu * 2 *
                                           p->intermediate_dim * p->hidden_dim;

  {
    printf("Allocated w_mlp1\n");
    fflush(stdout);
    float *tmp_buffer = (float *)malloc(layers_per_stage * experts_per_gpu * 2 *
                                        sharded_intermediate_dim * p->hidden_dim * sizeof(float));

    for (int l = 0; l < layers_per_stage; l++) {
      for (int e = 0; e < experts_per_gpu; e++) {
        size_t offset = 1ll * (l * experts_per_gpu + e) * 2 * p->intermediate_dim * p->hidden_dim;
        size_t dst_offset =
          1ll * (l * experts_per_gpu + e) * 2 * sharded_intermediate_dim * p->hidden_dim;
        float *src =
          w_mlp1_stage_base + offset + 1ll * tp_rank * 2 * sharded_intermediate_dim * p->hidden_dim;

        for (int i = 0; i < p->hidden_dim; i++) {
          for (int j = 0; j < 2 * sharded_intermediate_dim; j++) {
            tmp_buffer[dst_offset + i * 2 * sharded_intermediate_dim + j] =
              src[j * p->hidden_dim + i];
          }
        }
      }
    }

    weights->w_mlp1 = new Tensor({(size_t)layers_per_stage, (size_t)experts_per_gpu,
                                  (size_t)p->hidden_dim, 2 * sharded_intermediate_dim},
                                 tmp_buffer, stream, DType::BF16);
    CHECK_HIP(hipStreamSynchronize(stream));
    free(tmp_buffer);
  }

  // {
  //   float *tmp_buffer = (float *)malloc(layers_per_stage * experts_per_gpu * 2 *
  //                                       sharded_intermediate_dim * p->hidden_dim * sizeof(float));

  //   for (int l = 0; l < layers_per_stage; l++) {
  //     for (int e = 0; e < experts_per_gpu; e++) {
  //       float *src = w_mlp1_stage_base +
  //                    1ll * (l * experts_per_gpu + e) * 2 * p->intermediate_dim * p->hidden_dim +
  //                    1ll * tp_rank * 2 * sharded_intermediate_dim * p->hidden_dim;

  //       float *dst = tmp_buffer +
  //                    1ll * (l * experts_per_gpu + e) * 2 * sharded_intermediate_dim * p->hidden_dim;

  //       memcpy(dst, src, 2 * sharded_intermediate_dim * p->hidden_dim * sizeof(float));
  //     }
  //   }

  //   weights->w_mlp1 = new Tensor({(size_t)layers_per_stage, (size_t)experts_per_gpu,
  //                                 2 * sharded_intermediate_dim, (size_t)p->hidden_dim},
  //                                tmp_buffer, stream, DType::BF16);
  //   free(tmp_buffer);
  // }

  // Sharding bias MLP1
  float *b_mlp1_stage_base =
    w->b_mlp1 + 1ll * pp_rank * layers_per_stage * experts_per_gpu * 2 * p->intermediate_dim;

  {
    float *tmp_buffer = (float *)malloc(layers_per_stage * experts_per_gpu * 2 *
                                        sharded_intermediate_dim * sizeof(float));
    for (int l = 0; l < layers_per_stage; l++) {
      for (int e = 0; e < experts_per_gpu; e++) {
        float *src = b_mlp1_stage_base + 1ll * (l * experts_per_gpu + e) * 2 * p->intermediate_dim +
                     1ll * tp_rank * 2 * sharded_intermediate_dim;

        float *dst = tmp_buffer + 1ll * (l * experts_per_gpu + e) * 2 * sharded_intermediate_dim;
        memcpy(dst, src, 2 * sharded_intermediate_dim * sizeof(float));
      }
    }

    weights->b_mlp1 =
      new Tensor({(size_t)layers_per_stage, (size_t)experts_per_gpu, 2 * sharded_intermediate_dim},
                 tmp_buffer, stream, DType::BF16);

    CHECK_HIP(hipStreamSynchronize(stream));
    free(tmp_buffer);
  }

  // Sharding weights MLP2
  float *w_mlp2_stage_base = w->w_mlp2 + 1ll * pp_rank * layers_per_stage * experts_per_gpu *
                                           p->hidden_dim * p->intermediate_dim;

  {
    printf("Allocate w_mlp2\n");
    fflush(stdout);
    float *tmp_transposed = (float *)malloc(layers_per_stage * experts_per_gpu * p->hidden_dim *
                                            sharded_intermediate_dim * sizeof(float));

    float *tmp_buffer = (float *)malloc(layers_per_stage * experts_per_gpu * p->hidden_dim *
                                        sharded_intermediate_dim * sizeof(float));

    for (size_t l = 0; l < layers_per_stage; l++) {
      for (size_t e = 0; e < experts_per_gpu; e++) {
        for (size_t h = 0; h < p->hidden_dim; h++) {
          float *src = w_mlp2_stage_base +
                       1ll * (l * experts_per_gpu + e) * p->hidden_dim * p->intermediate_dim +
                       1ll * h * p->intermediate_dim + 1ll * tp_rank * sharded_intermediate_dim;

          float *dst = tmp_buffer +
                       1ll * (l * experts_per_gpu + e) * p->hidden_dim * sharded_intermediate_dim +
                       1ll * h * sharded_intermediate_dim;

          memcpy(dst, src, sharded_intermediate_dim * sizeof(float));
        }
      }
    }

    for (size_t l = 0; l < layers_per_stage; l++) {
      for (size_t e = 0; e < experts_per_gpu; e++) {
        size_t offset = 1ll * (l * experts_per_gpu + e) * p->hidden_dim * sharded_intermediate_dim;
        for (size_t i = 0; i < sharded_intermediate_dim; i++) {
          for (size_t h = 0; h < p->hidden_dim; h++) {
            tmp_transposed[offset + i * p->hidden_dim + h] =
              tmp_buffer[offset + h * sharded_intermediate_dim + i];
          }
        }
      }
    }

    weights->w_mlp2 = new Tensor({(size_t)layers_per_stage, (size_t)experts_per_gpu,
                                  sharded_intermediate_dim, (size_t)p->hidden_dim},
                                 tmp_transposed, stream, DType::BF16);

    CHECK_HIP(hipStreamSynchronize(stream));
    free(tmp_buffer);
    free(tmp_transposed);
  }

  // {
  //   float *tmp_buffer = (float *)malloc(layers_per_stage * experts_per_gpu *
  //                                       sharded_intermediate_dim * p->hidden_dim * sizeof(float));

  //   for (int l = 0; l < layers_per_stage; l++) {
  //     for (int e = 0; e < experts_per_gpu; e++) {
  //       for (int h = 0; h < p->hidden_dim; h++) {
  //         float *src = w_mlp2_stage_base +
  //                      1ll * (l * experts_per_gpu + e) * p->hidden_dim * p->intermediate_dim +
  //                      1ll * h * p->intermediate_dim + 1ll * tp_rank * sharded_intermediate_dim;

  //         float *dst = tmp_buffer +
  //                      1ll * (l * experts_per_gpu + e) * p->hidden_dim * sharded_intermediate_dim +
  //                      1ll * h * sharded_intermediate_dim;

  //         memcpy(dst, src, sharded_intermediate_dim * sizeof(float));
  //       }
  //     }
  //   }

  //   weights->w_mlp2 = new Tensor({(size_t)layers_per_stage, (size_t)experts_per_gpu,
  //                                 (size_t)p->hidden_dim, sharded_intermediate_dim},
  //                                tmp_buffer, stream, DType::BF16);

  //   free(tmp_buffer);
  // }

  // Sharding bias MLP2
  float *b_mlp2_stage_base =
    w->b_mlp2 + 1ll * pp_rank * layers_per_stage * experts_per_gpu * p->hidden_dim;

  weights->b_mlp2 =
    new Tensor({(size_t)layers_per_stage, (size_t)experts_per_gpu, (size_t)p->hidden_dim},
               b_mlp2_stage_base, stream, DType::BF16);

  weights->rms_out_w = new Tensor({(size_t)p->hidden_dim}, w->rms_out_w, stream);
  // weights->out = new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->out, stream);

  {
    size_t vocab_size = p->vocab_size;
    size_t hidden_dim = p->hidden_dim;

    weights->out = new Tensor({hidden_dim, vocab_size}, stream, DType::BF16);

    for (size_t i = 0; i < vocab_size; i++) {
      for (size_t j = 0; j < hidden_dim; j++) {
        weights->out->buf[j * vocab_size + i] = w->out[i * hidden_dim + j];
      }
    }

    weights->out->to_device(stream);
  }
}

float *forward_gpu_120b(Context *ctx, int *tokens, int pos) {
  Config *p = &ctx->transformer->config;
  int layer_per_stages = p->n_layers / PP;

  // embedding lookup only for all TPs of stage 0
  for (int i = 0; i < TP; i++) {
    int gpu_id = ctx->gpu_ids[i];
    CHECK_HIP(hipSetDevice(gpu_id));
    embedding_lookup_batched(ctx->weights[i]->token_embedding_table, tokens, ctx->run_state[i]->x,
                             false, ctx->streams[i]);
    // embedding_lookup(ctx->weights[i]->token_embedding_table, token, ctx->run_state[i]->x, false,
    //                  ctx->streams[i]);
  }

  for (int l = 0; l < p->n_layers; l++) {
    int current_stage_idx = l / layer_per_stages;
    int local_layer_idx = l % layer_per_stages;
    int start_local_idx = current_stage_idx * TP;  // First TP in stage
    int end_local_idx = start_local_idx + TP;      // End TP in stage (exclusive)

    // printf("==== stage: %d - layer: %d ====\n", current_stage_idx, l);

    // Pipeline recv at stage boundary
    if (current_stage_idx > 0 && local_layer_idx == 0) {
      int leader_idx = start_local_idx;
      int leader_global_id = ctx->gpu_ids[leader_idx];

      CHECK_HIP(hipSetDevice(leader_global_id));

      // Wait buffer ready from prev stage
      CHECK_HIP(hipStreamWaitEvent(ctx->streams[leader_idx], ctx->events[leader_idx], 0));

      size_t nbytes =
        ctx->run_state[leader_idx]->x->num_elem() * ctx->run_state[leader_idx]->x->get_dtype_size();

      CHECK_HIP(hipMemcpyAsync(ctx->run_state[leader_idx]->x->d_buf,
                               ctx->pipeline_buffers[current_stage_idx - 1]->d_buf, nbytes,
                               hipMemcpyDeviceToDevice, ctx->streams[leader_idx]));

      // Signal x is ready for fan out
      CHECK_HIP(hipEventRecord(ctx->pipe_recv_ready_event[leader_idx], ctx->streams[leader_idx]));

      for (int i = start_local_idx + 1; i < end_local_idx; i++) {
        int dst_global_id = ctx->gpu_ids[i];
        CHECK_HIP(hipSetDevice(dst_global_id));

        CHECK_HIP(hipStreamWaitEvent(ctx->streams[i], ctx->pipe_recv_ready_event[leader_idx], 0));
        CHECK_HIP(hipMemcpyPeerAsync(ctx->run_state[i]->x->d_buf, dst_global_id,
                                     ctx->run_state[leader_idx]->x->d_buf, leader_global_id, nbytes,
                                     ctx->streams[i]));
      }
      // printf("stage %d - pipeline recv\n", current_stage_idx);
    }

// Attention layers
#pragma omp parallel for num_threads(TP)
    for (int i = start_local_idx; i < end_local_idx; i++) {
      CHECK_HIP(hipSetDevice(ctx->gpu_ids[i]));
      OurRunState *s = ctx->run_state[i];
      OurTransformerWeights *w = ctx->weights[i];
      hipStream_t stream = ctx->streams[i];

      rmsnorm_batched(s->x, w->rms_attn_w, s->t, local_layer_idx, false, false, 1e-5f, stream);
      // rmsnorm(s->x, w->rms_attn_w, s->t, local_layer_idx, false, false, 1e-5f, ctx->streams[i]);
      // printf("stage %d - Done rmsnorm 1\n", current_stage_idx);

      qkv_gemm_batched_v2(s->t, w->w_qkv, w->b_qkv, s->qkv, local_layer_idx, false, false, stream);
      // qkv_gemm(s->t, w->w_qkv, w->b_qkv, s->qkv, local_layer_idx, false, false, ctx->streams[i]);
      // printf("stage %d - Done qkv_gemm\n", current_stage_idx);

      qkv_split_rope_fused(s->qkv, s->q, s->key_cache, s->value_cache, ctx->cos_tensor[i],
                           ctx->sin_tensor[i], p->head_dim, p->n_attn_heads, p->n_kv_heads, pos,
                           local_layer_idx, stream);
      // qkv_split_rope(s->qkv, s->q, s->k, s->v, ctx->cos_tensor[i], ctx->sin_tensor[i], p->head_dim,
      //                p->n_attn_heads, p->n_kv_heads, pos, false, false, false, false,
      //                ctx->streams[i]);
      // // printf("stage %d - Done qkv_split_rope\n", current_stage_idx);

      // long long kv_cache_offset =
      //   (long long)local_layer_idx * p->seq_len * p->n_kv_heads * p->head_dim +
      //   (long long)pos * p->n_kv_heads * p->head_dim;
      // memcpy_tensor(s->key_cache, s->k, kv_cache_offset, 0, s->k->num_elem(), false, true,
      //               ctx->streams[i]);
      // memcpy_tensor(s->value_cache, s->v, kv_cache_offset, 0, s->v->num_elem(), false, true,
      //               ctx->streams[i]);
      // // printf("stage %d - Done memcpy_tensor\n", current_stage_idx);

      single_query_attn_batched(s->q, s->key_cache, s->value_cache, s->mask, w->attn_sinks, s->tb,
                                p->head_dim, p->n_attn_heads, p->n_attn_heads / p->n_kv_heads,
                                p->n_kv_heads * p->head_dim, p->seq_len, p->sliding_window, pos,
                                local_layer_idx, false, false, false, false, false, stream);
      // single_query_attn(s->q, s->key_cache, s->value_cache, s->mask, w->attn_sinks, s->tb,
      //                   p->head_dim, p->n_attn_heads, p->n_attn_heads / p->n_kv_heads,
      //                   p->n_kv_heads * p->head_dim, p->seq_len, p->sliding_window, pos,
      //                   local_layer_idx, false, false, false, false, false, ctx->streams[i]);
      // printf("stage %d - Done single_query_attn\n", current_stage_idx);

      attn_out_project_batched_v2(s->tb, w->w_o, w->b_o, s->tb2, local_layer_idx, false, false,
                                  stream);
      // attn_out_project(s->tb, w->w_o, w->b_o, s->tb2, local_layer_idx, false, false,
      //                  ctx->streams[i]);
      // printf("stage %d - Done attn_out_project\n", current_stage_idx);

      residual_rmsnorm_batched(s->tb2, s->x, w->rms_ffn_w, s->t, local_layer_idx, 1e-5f, stream);
      // add_vector(s->x, s->tb2, false, false, false, ctx->streams[i]);
      // printf("stage %d - Done add_vector 1\n", current_stage_idx);

      // rmsnorm(s->x, w->rms_ffn_w, s->t, local_layer_idx, false, false, 1e-5f, ctx->streams[i]);
      // printf("stage %d - Done rmsnorm 2\n", current_stage_idx);

      router_gemm_batched(w->w_router, s->t, w->b_router, s->router_score, local_layer_idx, false,
                          false, stream);
      // router_gemm(w->w_router, s->t, w->b_router, s->router_score, local_layer_idx, false, false,
      //             ctx->streams[i]);
      // printf("stage %d - Done router_gemm\n", current_stage_idx);

      topk_softmax_batched(s->router_score, s->topk_v, s->topk_i, false, false, false, stream);
      // topk_softmax(s->router_score, s->topk_v, s->topk_i, false, false, false, ctx->streams[i]);
      // printf("stage %d - Done topk_softmax\n", current_stage_idx);

      int total_pairs = BATCH_SIZE * p->experts_per_token;
      moe_init_buffers_hip(s->e_agg, s->mlp1_out, s->gate_up, s->tb3, s->sorted_pair_ids,
                           s->expert_offsets, s->x_packed, BATCH_SIZE, p->hidden_dim, stream);

      moe_build_offsets_hip(s->topk_i, s->sorted_pair_ids, s->expert_offsets, BATCH_SIZE,
                            p->experts_per_token, p->n_experts, stream);

      moe_pack_inputs_hip(s->t, s->sorted_pair_ids, s->x_packed, BATCH_SIZE, p->hidden_dim,
                          p->experts_per_token, stream);

      s->max_rows = moe_get_max_rows_per_expert_hip(s->expert_offsets, p->n_experts, stream);
      moe_mlp1_forward_hip(s->x_packed, w->w_mlp1, w->b_mlp1, s->expert_offsets, s->mlp1_out,
                           local_layer_idx, p->n_experts, p->hidden_dim, p->intermediate_dim / TP,
                           s->max_rows, total_pairs, stream);

      moe_swiglu_hip(s->mlp1_out, s->gate_up, BATCH_SIZE, p->experts_per_token,
                     p->intermediate_dim / TP, p->swiglu_limit, stream);

      moe_mlp2_forward_hip(s->gate_up, w->w_mlp2, w->b_mlp2, s->expert_offsets, s->tb3,
                           (i == start_local_idx), local_layer_idx, p->n_experts,
                           p->intermediate_dim / TP, p->hidden_dim, s->max_rows, total_pairs,
                           stream);

      // moe_mlp1_batched(s->t, w->w_mlp1, w->b_mlp1, s->topk_i, s->mlp1_out, false, false,
      //                  local_layer_idx, stream);

      // moe_swiglu_batched(s->mlp1_out, s->gate_up, p->experts_per_token, p->swiglu_limit, stream);
      // moe_mlp2_batched(s->gate_up, w->w_mlp2, w->b_mlp2, s->tb3, s->topk_i, p->experts_per_token,
      //                  (i == start_local_idx), local_layer_idx, stream);

      CHECK_HIP(hipEventRecord(ctx->tp_ready_event[i], ctx->streams[i]));
    }

    if (TP > 1) {
      reduce_broadcast_tb3(ctx, start_local_idx, end_local_idx,
                           ctx->run_state[start_local_idx]->tb3);
    }

    // if (l == 0) {
    //   ctx->run_state[start_local_idx]->tb3->from_device(ctx->streams[start_local_idx]);
    //   for (int k = 0; k < p->experts_per_token; k++) {
    //     printf("======= Expert %d =======\n", k);
    //     for (int i = 0; i < 20; i++) {
    //       printf("tb3[%d] = %f\n", i,
    //              ctx->run_state[start_local_idx]->tb3->buf[k * p->hidden_dim + i]);
    //     }
    //   }
    // }

#pragma omp parallel for num_threads(TP)
    for (int i = start_local_idx; i < end_local_idx; i++) {
      CHECK_HIP(hipSetDevice(ctx->gpu_ids[i]));
      OurRunState *s = ctx->run_state[i];
      hipStream_t stream = ctx->streams[i];

      moe_scatter_aggregate_hip(s->tb3, s->sorted_pair_ids, s->topk_v, s->e_agg, s->expert_offsets,
                                p->hidden_dim, p->experts_per_token, p->n_experts, s->max_rows,
                                stream);
      // moe_agg_batched(s->tb3, s->topk_v, s->e_agg, p->experts_per_token, false, stream);
      // moe_agg(tb3_ptr, topk_v_ptr, e_agg_ptr, k, hidden_dim, ctx->streams[i]);

      add_vector_batched(s->x, s->e_agg, false, false, false, stream);
      // add_vector(ctx->run_state[i]->x, ctx->run_state[i]->e_agg, false, false, false,
      //            ctx->streams[i]);
    }

    // --- Pipeline Send ---
    if ((l + 1) % layer_per_stages == 0 && current_stage_idx < PP - 1) {
      int leader_idx = start_local_idx;  // leader stage này
      int leader_global_id = ctx->gpu_ids[leader_idx];

      int next_leader_idx = (current_stage_idx + 1) * TP;
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
  hipStream_t stream_final = ctx->streams[last_local_idx];

  CHECK_HIP(hipSetDevice(ctx->gpu_ids[last_local_idx]));

  rmsnorm_batched(s_final->x, w_final->rms_out_w, s_final->x, 0, false, false, 1e-5f, stream_final);
  classifier_gemm_batched_v2(w_final->out, s_final->x, s_final->logits, false, true, stream_final);
  // rmsnorm(s_final->x, w_final->rms_out_w, s_final->x, 0, false, false, 1e-5f,
  //         ctx->streams[last_local_idx]);
  // classifier_gemm(w_final->out, s_final->x, s_final->logits, false, true,
  //                 ctx->streams[last_local_idx]);

  s_final->logits->from_device(ctx->streams[last_local_idx]);

  return s_final->logits->buf;
}
