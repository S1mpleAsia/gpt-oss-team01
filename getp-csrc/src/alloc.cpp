#include "../include/alloc.hpp"
#include <cmath>
#include <cstring>

#ifdef RUN_20B
void our_init_weights(TransformerWeights *w, Config *p, OurTransformerWeights *weights,
                      int device_id, hipStream_t stream) {
  CHECK_HIP(hipSetDevice(device_id));

  weights->token_embedding_table = new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim},
                                              w->token_embedding_table, stream, DType::BF16);

  weights->rms_attn_w =
    new Tensor({(size_t)p->n_layers * p->hidden_dim}, w->rms_attn_w, stream, DType::BF16);
  weights->rms_ffn_w = new Tensor({(size_t)p->n_layers * p->hidden_dim}, w->rms_ffn_w, stream);

  {
    size_t n_layers = p->n_layers;
    size_t n_heads = p->n_attn_heads + 2 * p->n_kv_heads;
    size_t head_dim = p->head_dim;
    size_t hidden_dim = p->hidden_dim;

    weights->w_qkv = new Tensor({n_layers, hidden_dim, n_heads * head_dim}, stream, DType::BF16);

    for (size_t l = 0; l < n_layers; l++) {
      for (size_t i = 0; i < n_heads * head_dim; i++) {
        for (size_t j = 0; j < hidden_dim; j++) {
          // hoán vị 2 chiều cuối
          weights->w_qkv
            ->buf[l * (hidden_dim * n_heads * head_dim) + j * (n_heads * head_dim) + i] =
            w->w_qkv[l * ((n_heads * head_dim) * hidden_dim) + i * hidden_dim + j];
        }
      }
    }

    weights->w_qkv->to_device(stream);
  }

  weights->b_qkv = new Tensor(
    {(size_t)p->n_layers, ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim},
    w->b_qkv, stream, DType::BF16);

  {
    size_t n_layers = p->n_layers;
    size_t hidden_dim = p->hidden_dim;
    size_t n_heads = p->n_attn_heads;
    size_t head_dim = p->head_dim;

    // Tạo tensor với shape transposed
    weights->w_o = new Tensor({n_layers, n_heads * head_dim, hidden_dim}, stream, DType::BF16);

    for (size_t l = 0; l < n_layers; l++) {
      for (size_t i = 0; i < hidden_dim; i++) {
        for (size_t j = 0; j < n_heads * head_dim; j++) {
          weights->w_o->buf[l * (n_heads * head_dim * hidden_dim) + j * hidden_dim + i] =
            w->w_o[l * (hidden_dim * n_heads * head_dim) + i * (n_heads * head_dim) + j];
        }
      }
    }

    weights->w_o->to_device(stream);
  }

  weights->b_o =
    new Tensor({(size_t)p->n_layers, (size_t)p->hidden_dim}, w->b_o, stream, DType::BF16);

  weights->attn_sinks =
    new Tensor({(size_t)p->n_layers, (size_t)p->n_attn_heads}, w->attn_sinks, stream);

  weights->w_router = new Tensor({(size_t)p->n_layers, (size_t)p->n_experts, (size_t)p->hidden_dim},
                                 w->w_router, stream);
  weights->b_router = new Tensor({(size_t)p->n_layers, (size_t)p->n_experts}, w->b_router, stream);

  {
    printf("Starting alloc mlp1\n");
    fflush(stdout);
    size_t n_layers = p->n_layers;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;

    weights->w_mlp1 =
      new Tensor({n_layers, n_experts, hidden_dim, 2 * inter_dim}, stream, DType::BF16);

    size_t tmp_elems = hidden_dim * 2 * inter_dim;
    bf16 *tmp = (bf16 *)malloc(tmp_elems * sizeof(bf16));

    bf16 *d_buf = (bf16 *)(weights->w_mlp1->d_buf);

    for (size_t l = 0; l < n_layers; l++) {
      for (size_t e = 0; e < n_experts; e++) {
        size_t base = l * n_experts * hidden_dim * 2 * inter_dim + e * hidden_dim * 2 * inter_dim;
        for (size_t h = 0; h < hidden_dim; h++) {
          for (size_t i = 0; i < 2 * inter_dim; i++) {
            float value = w->w_mlp1[base + i * hidden_dim + h];
            tmp[h * 2 * inter_dim + i] = bf16(value);
          }
        }

        size_t d_offset =
          l * n_experts * hidden_dim * 2 * inter_dim + e * hidden_dim * 2 * inter_dim;
        CHECK_HIP(hipMemcpyAsync(d_buf + d_offset, tmp, tmp_elems * sizeof(bf16),
                                 hipMemcpyHostToDevice, stream));
      }
    }

    CHECK_HIP(hipStreamSynchronize(stream));
    free(tmp);

    printf("End alloc mlp1\n");
    fflush(stdout);
  }

  weights->b_mlp1 =
    new Tensor({(size_t)p->n_layers, (size_t)p->n_experts, 2 * (size_t)p->intermediate_dim},
               w->b_mlp1, stream, DType::BF16);

  {
    printf("Starting alloc mlp2\n");
    fflush(stdout);
    size_t n_layers = p->n_layers;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;

    weights->w_mlp2 = new Tensor({n_layers, n_experts, inter_dim, hidden_dim}, stream, DType::BF16);

    size_t tmp_elems = inter_dim * hidden_dim;
    bf16 *tmp = (bf16 *)malloc(tmp_elems * sizeof(bf16));

    bf16 *d_buf = (bf16 *)(weights->w_mlp2->d_buf);

    for (size_t l = 0; l < n_layers; l++) {
      for (size_t e = 0; e < n_experts; e++) {
        size_t base = l * n_experts * hidden_dim * inter_dim + e * hidden_dim * inter_dim;

        for (size_t i = 0; i < inter_dim; i++) {
          for (size_t h = 0; h < hidden_dim; h++) {
            float value = w->w_mlp2[base + h * inter_dim + i];
            tmp[i * hidden_dim + h] = bf16(value);
          }
        }

        size_t d_offset = l * n_experts * inter_dim * hidden_dim + e * inter_dim * hidden_dim;
        CHECK_HIP(hipMemcpyAsync(d_buf + d_offset, tmp, tmp_elems * sizeof(bf16),
                                 hipMemcpyHostToDevice, stream));
      }
    }

    CHECK_HIP(hipStreamSynchronize(stream));
    free(tmp);

    // weights->w_mlp2->to_device(0);
    printf("End alloc mlp2\n");
    fflush(stdout);
  }

  weights->b_mlp2 = new Tensor({(size_t)p->n_layers, (size_t)p->n_experts, (size_t)p->hidden_dim},
                               w->b_mlp2, stream, DType::BF16);

  weights->rms_out_w = new Tensor({(size_t)p->hidden_dim}, w->rms_out_w, stream, DType::BF16);

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

void our_init_run_state(RunState *s, Config *p, OurRunState *rs, int device_id,
                        hipStream_t stream) {
  CHECK_HIP(hipSetDevice(device_id));

  rs->x = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, stream);

  rs->t = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, stream);
  rs->tb = new Tensor({BATCH_SIZE, (size_t)p->head_dim * p->n_attn_heads}, stream);
  rs->tb2 = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, stream);

  rs->tb3 = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->hidden_dim}, stream);

  rs->router_score = new Tensor({BATCH_SIZE, (size_t)p->n_experts}, stream);
  rs->topk_v = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token}, stream);
  rs->topk_i = new TensorI32({BATCH_SIZE, (size_t)p->experts_per_token}, stream);

  rs->mlp1_out =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, 2 * (size_t)p->intermediate_dim}, stream);
  rs->gate_up =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, stream);

  rs->e_agg = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, stream);
  rs->e_agg_buf = nullptr;

  rs->qkv = new Tensor(
    {BATCH_SIZE, ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim}, stream);
  rs->q = new Tensor({BATCH_SIZE, (size_t)p->n_attn_heads * p->head_dim}, stream);
  rs->logits = new Tensor({BATCH_SIZE, (size_t)p->vocab_size}, stream);

#ifdef KV16
  printf("using BF16 KV cache\n");
  rs->key_cache = new Tensor(
    {BATCH_SIZE, (size_t)p->n_layers, (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim},
    stream, DType::BF16);
  rs->value_cache = new Tensor(
    {BATCH_SIZE, (size_t)p->n_layers, (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim},
    stream, DType::BF16);
#else 
  print("using FP32 KV cache\n");
  rs->key_cache = new Tensor(
    {BATCH_SIZE, (size_t)p->n_layers, (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim},
    stream);
  rs->value_cache = new Tensor(
    {BATCH_SIZE, (size_t)p->n_layers, (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim},
    stream);
#endif

  rs->mask = new Tensor({BATCH_SIZE, (size_t)1, (size_t)1}, stream);
  rs->mask->to_device(stream);

  // MoE buffer
  rs->sorted_pair_ids = new TensorI32({(size_t)BATCH_SIZE * p->experts_per_token}, stream);
  rs->expert_offsets = new TensorI32({(size_t)(p->n_experts + 1)}, stream);
  rs->x_packed =
    new Tensor({(size_t)BATCH_SIZE * p->experts_per_token, (size_t)p->hidden_dim}, stream);

  rs->tokens_buf = new TensorI32({(size_t)BATCH_SIZE}, stream);

  rs->cos_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2}, stream);
  rs->sin_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2}, stream);
  RopePrecomputeCS(p, rs->cos_tensor, rs->sin_tensor);
}

#else
void our_init_weights(TransformerWeights *w, Config *p, OurTransformerWeights *weights,
                      int device_id, hipStream_t stream) {
  CHECK_HIP(hipSetDevice(device_id));
  size_t layers_per_stage = p->n_layers / PP;
  size_t experts_per_gpu = p->n_experts / 1;
  size_t qkv_layer_size =
    ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * (size_t)p->head_dim;

  int tp_rank = device_id % TP;
  int pp_rank = (device_id % TOTAL_PIPELINES) / TP;

  long long pp_offset = 1ll * pp_rank * layers_per_stage;
  long long tp_offset = 1ll * tp_rank * experts_per_gpu;

  int shard_inter_dim = p->intermediate_dim / TP;

  // Create Tensor wrappers for weight matrices
  if (pp_rank == 0) {
    weights->token_embedding_table =
      new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->token_embedding_table, stream, DType::BF16);
  } else {
    weights->token_embedding_table = nullptr;
  }

  weights->rms_attn_w = new Tensor({layers_per_stage * p->hidden_dim},
                                   w->rms_attn_w + 1ll * pp_offset * p->hidden_dim, stream, DType::BF16);
  weights->rms_ffn_w = new Tensor({layers_per_stage * p->hidden_dim},
                                  w->rms_ffn_w + 1ll * pp_offset * p->hidden_dim, stream);
  {
    size_t hidden_dim = p->hidden_dim;
    size_t shard_qkv_size = qkv_layer_size / TP;
    size_t q_size = p->n_attn_heads * p->head_dim;
    size_t kv_size = p->n_kv_heads * p->head_dim;

    size_t q_shard_size = q_size / TP;
    size_t kv_shard_size = kv_size / TP;

    weights->w_qkv =
      new Tensor({layers_per_stage, hidden_dim, shard_qkv_size}, stream, DType::BF16);

    float *w_qkv_ptr = w->w_qkv + 1ll * pp_offset * qkv_layer_size * hidden_dim;

    for (size_t l = 0; l < layers_per_stage; l++) {
      float *src_ptr = w_qkv_ptr + l * hidden_dim * qkv_layer_size;
      float *dst_ptr = weights->w_qkv->buf + l * hidden_dim * shard_qkv_size;

      for (size_t i = 0; i < q_shard_size; i++) {
        size_t src_q_col = tp_rank * q_shard_size + i;
        for (size_t j = 0; j < hidden_dim; j++) {
          float value = src_ptr[src_q_col * hidden_dim + j];
          dst_ptr[j * shard_qkv_size + i] = value;
        }
      }

      for (size_t i = 0; i < kv_shard_size; i++) {
        size_t src_k_col = q_size + tp_rank * kv_shard_size + i;
        size_t dst_k_col = q_shard_size + i;
        for (size_t j = 0; j < hidden_dim; j++) {
          float value = src_ptr[src_k_col * hidden_dim + j];
          dst_ptr[j * shard_qkv_size + dst_k_col] = value;
        }
      }

      for (size_t i = 0; i < kv_shard_size; i++) {
        size_t src_v_col = q_size + kv_size + tp_rank * kv_shard_size + i;
        size_t dst_v_col = q_shard_size + kv_shard_size + i;
        for (size_t j = 0; j < hidden_dim; j++) {
          float value = src_ptr[src_v_col * hidden_dim + j];
          dst_ptr[j * shard_qkv_size + dst_v_col] = value;
        }
      }
    }

    weights->w_qkv->to_device(stream);
  }

  {
    size_t shard_qkv_size = qkv_layer_size / TP;
    size_t q_size = p->n_attn_heads * p->head_dim;
    size_t kv_size = p->n_kv_heads * p->head_dim;

    size_t q_shard_size = q_size / TP;
    size_t kv_shard_size = kv_size / TP;

    weights->b_qkv = new Tensor({layers_per_stage, shard_qkv_size}, stream, DType::BF16);

    float *b_qkv_ptr = w->b_qkv + 1ll * pp_offset * qkv_layer_size;

    for (size_t l = 0; l < layers_per_stage; l++) {
      float *src_ptr = b_qkv_ptr + l * qkv_layer_size;
      float *dst_ptr = weights->b_qkv->buf + l * shard_qkv_size;

      float *src_q = src_ptr + tp_rank * q_shard_size;
      float *dst_q = dst_ptr;
      memcpy(dst_q, src_q, q_shard_size * sizeof(float));

      float *src_k = src_ptr + q_size + tp_rank * kv_shard_size;
      float *dst_k = dst_ptr + q_shard_size;
      memcpy(dst_k, src_k, kv_shard_size * sizeof(float));

      float *src_v = src_ptr + q_size + kv_size + tp_rank * kv_shard_size;
      float *dst_v = dst_ptr + q_shard_size + kv_shard_size;
      memcpy(dst_v, src_v, kv_shard_size * sizeof(float));
    }

    weights->b_qkv->to_device(stream);
  }

  {
    size_t hidden_dim = p->hidden_dim;
    size_t n_attn_heads = p->n_attn_heads;
    size_t head_dim = p->head_dim;
    size_t head_size = n_attn_heads * head_dim;
    size_t shard_head_size = head_size / TP;

    weights->w_o = new Tensor({layers_per_stage, shard_head_size, hidden_dim}, stream, DType::BF16);
    float *w_o_ptr = w->w_o + 1ll * pp_offset * hidden_dim * head_size;

    for (size_t l = 0; l < layers_per_stage; l++) {
      float *src_ptr = w_o_ptr + 1ll * l * hidden_dim * head_size + tp_rank * shard_head_size;
      float *dst_ptr = weights->w_o->buf + l * shard_head_size * hidden_dim;

      for (size_t i = 0; i < shard_head_size; i++) {
        for (size_t h = 0; h < hidden_dim; h++) {
          dst_ptr[i * hidden_dim + h] = src_ptr[h * head_size + i];
        }
      }
    }

    weights->w_o->to_device(stream);
  }

  weights->b_o = new Tensor({layers_per_stage, (size_t)p->hidden_dim},
                            w->b_o + 1ll * pp_offset * (size_t)p->hidden_dim, stream, DType::BF16);

  {
    size_t n_attn_heads = p->n_attn_heads;
    size_t shard_attn_heads = n_attn_heads / TP;

    weights->attn_sinks = new Tensor({layers_per_stage, (size_t)shard_attn_heads}, stream);
    float *attn_sinks_ptr = w->attn_sinks + 1ll * pp_offset * n_attn_heads;

    for (size_t l = 0; l < layers_per_stage; l++) {
      float *src_ptr = attn_sinks_ptr + l * n_attn_heads + tp_rank * shard_attn_heads;
      float *dst_ptr = weights->attn_sinks->buf + l * shard_attn_heads;

      memcpy(dst_ptr, src_ptr, shard_attn_heads * sizeof(float));
    }

    weights->attn_sinks->to_device(stream);
  }

  weights->w_router = new Tensor(
    {layers_per_stage, (size_t)p->n_experts, (size_t)p->hidden_dim},
    w->w_router + 1ll * pp_offset * (size_t)p->n_experts * (size_t)p->hidden_dim, stream);
  weights->b_router = new Tensor({layers_per_stage, (size_t)p->n_experts},
                                 w->b_router + 1ll * pp_offset * (size_t)p->n_experts, stream);

  float *w_mlp1_ptr = w->w_mlp1 + 1ll * pp_offset * (size_t)p->n_experts * 2 *
                                    (size_t)p->intermediate_dim * (size_t)p->hidden_dim;
  {
    printf("Starting alloc mlp1\n");
    fflush(stdout);
    size_t n_layers = layers_per_stage;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;
    size_t shard_dim = shard_inter_dim;

    weights->w_mlp1 =
      new Tensor({n_layers, n_experts, hidden_dim, 2 * shard_dim}, stream, DType::BF16);

    size_t tmp_elems = hidden_dim * 2 * shard_dim;
    bf16 *tmp = (bf16 *)malloc(tmp_elems * sizeof(bf16));

    bf16 *d_buf = (bf16 *)(weights->w_mlp1->d_buf);

    w_mlp1_ptr += tp_rank * (2 * shard_dim) * hidden_dim;  // offset shard_dim;

    size_t l_offset = 1ll * n_experts * (2 * inter_dim) * hidden_dim;
    size_t e_offset = 1ll * (2 * inter_dim) * hidden_dim;

    size_t l_offset_d = 1ll * n_experts * hidden_dim * (2 * shard_dim);
    size_t e_offset_d = 1ll * hidden_dim * (2 * shard_dim);

    for (size_t l = 0; l < n_layers; l++) {
      for (size_t e = 0; e < n_experts; e++) {
        size_t base = 1ll * l * l_offset + 1ll * e * e_offset;

        for (size_t h = 0; h < hidden_dim; h++) {
          for (size_t i = 0; i < 2 * shard_dim; i++) {
            float value = w_mlp1_ptr[base + i * hidden_dim + h];
            tmp[h * 2 * shard_dim + i] = bf16(value);
          }
        }

        size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d;
        CHECK_HIP(
          hipMemcpy(d_buf + d_offset, tmp, tmp_elems * sizeof(bf16), hipMemcpyHostToDevice));
      }
    }

    CHECK_HIP(hipStreamSynchronize(stream));
    free(tmp);

    printf("End alloc mlp1\n");
    fflush(stdout);
  }
  /** w_mlp1 EXPERT PARALLELISM
    w_mlp1_ptr += 1ll * tp_offset * 2 * (size_t)p->intermediate_dim * (size_t)p->hidden_dim;
    bf16 *w_mlp1_d_buf = (bf16 *)weights->w_mlp1->d_buf;

    for (int l = 0; l < layers_per_stage; l++) {
      // Convert and copy for BF16
      size_t N_ = experts_per_gpu * 2 * (size_t)p->intermediate_dim *(size_t)p->hidden_dim;
      bf16 *temp_bf16 = (bf16 *)malloc(N_ * sizeof(bf16));
      for (size_t i = 0; i < N_; i++) {
        temp_bf16[i] = hip_bfloat16(w_mlp1_ptr[i]);
      }
      CHECK_HIP(hipMemcpy(w_mlp1_d_buf, temp_bf16, N_ * sizeof(bf16), hipMemcpyHostToDevice));
      free(temp_bf16);

      w_mlp1_ptr += 1ll * (size_t)p->n_experts *
                                  2 * (size_t)p->intermediate_dim * (size_t)p->hidden_dim;
      w_mlp1_d_buf += 1ll * N_;
    }
  */

  // Initialize pointer with pipeline parallelism (pp) offset
  float *b_mlp1_ptr =
    w->b_mlp1 + 1ll * pp_offset * (size_t)p->n_experts * 2 * (size_t)p->intermediate_dim;
  /** w->b_mlp1
    weights->b_mlp1 = new Tensor(
      {layers_per_stage, experts_per_gpu, 2 * (size_t)p->intermediate_dim},
      b_mlp1_ptr, device_id, DType::BF16
    );
  */
  {
    printf("Starting alloc b_mlp1\n");
    fflush(stdout);
    size_t inter_dim = p->intermediate_dim;
    size_t shard_dim = shard_inter_dim;

    weights->b_mlp1 =
      new Tensor({layers_per_stage, experts_per_gpu, 2 * shard_dim}, stream, DType::BF16);

    size_t tmp_elems = 2 * shard_dim;
    bf16 *tmp = (bf16 *)malloc(tmp_elems * sizeof(bf16));

    bf16 *d_buf = (bf16 *)(weights->b_mlp1->d_buf);

    b_mlp1_ptr += 1ll * tp_rank * (2 * shard_dim);  // offset shard_dim;

    size_t l_offset = 1ll * experts_per_gpu * 2 * inter_dim;
    size_t e_offset = 1ll * 2 * inter_dim;

    size_t l_offset_d = 1ll * experts_per_gpu * 2 * shard_dim;
    size_t e_offset_d = 1ll * 2 * shard_dim;

    for (size_t l = 0; l < layers_per_stage; l++) {
      for (size_t e = 0; e < experts_per_gpu; e++) {
        size_t base = 1ll * l * l_offset + 1ll * e * e_offset;

        for (size_t i = 0; i < 2 * shard_dim; i++) {
          float value = b_mlp1_ptr[base + i];
          tmp[i] = bf16(value);
        }

        size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d;
        CHECK_HIP(
          hipMemcpy(d_buf + d_offset, tmp, tmp_elems * sizeof(bf16), hipMemcpyHostToDevice));
      }
    }

    CHECK_HIP(hipStreamSynchronize(stream));
    free(tmp);

    printf("End alloc b_mlp1\n");
    fflush(stdout);
  }

  /** b_mlp1 EXPERT PARALLELISM
    // Apply tensor parallelism (tp) offset for the expert dimension
    b_mlp1_ptr += 1ll * tp_offset * 2 * (size_t)p->intermediate_dim;
    bf16 *b_mlp1_d_buf = (bf16 *)weights->b_mlp1->d_buf;

    for (int l = 0; l < layers_per_stage; l++) {
      // Calculate the total number of elements for one layer on this device
      size_t N_ = experts_per_gpu * 2 * (size_t)p->intermediate_dim;
      bf16 *temp_bf16 = (bf16 *)malloc(N_ * sizeof(bf16));

      // Convert from FP32 to BF16 on the host
      for (size_t i = 0; i < N_; i++) {
        temp_bf16[i] = hip_bfloat16(b_mlp1_ptr[i]);
      }

      // Copy the converted data to the device
      CHECK_HIP(hipMemcpy(b_mlp1_d_buf, temp_bf16, N_ * sizeof(bf16), hipMemcpyHostToDevice));
      free(temp_bf16);

      // Advance host pointer to the start of the next layer's full set of experts
      b_mlp1_ptr += 1ll * (size_t)p->n_experts * 2 * (size_t)p->intermediate_dim;
      // Advance device pointer to the next layer's destination block
      b_mlp1_d_buf += 1ll * N_;
    }
  */

  // Initialize pointer with pipeline parallelism (pp) offset
  float *w_mlp2_ptr = w->w_mlp2 + 1ll * pp_offset * (size_t)p->n_experts * (size_t)p->hidden_dim *
                                    (size_t)p->intermediate_dim;
  /** w_mlp2
    weights->w_mlp2 = new Tensor(
      {layers_per_stage, experts_per_gpu, (size_t)p->hidden_dim, (size_t)p->intermediate_dim}, w_mlp2_ptr, device_id, DType::BF16
    );
  */
  {
    printf("Starting alloc mlp2\n");
    fflush(stdout);
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;
    size_t shard_dim = shard_inter_dim;

    weights->w_mlp2 =
      new Tensor({layers_per_stage, experts_per_gpu, shard_dim, hidden_dim}, stream, DType::BF16);

    size_t tmp_elems = shard_dim * hidden_dim;
    bf16 *tmp = (bf16 *)malloc(tmp_elems * sizeof(bf16));

    bf16 *d_buf = (bf16 *)(weights->w_mlp2->d_buf);

    w_mlp2_ptr += 1ll * tp_rank * shard_dim;  // offset shard_dim;

    size_t l_offset = 1ll * experts_per_gpu * hidden_dim * inter_dim;
    size_t e_offset = 1ll * hidden_dim * inter_dim;

    size_t l_offset_d = 1ll * experts_per_gpu * shard_dim * hidden_dim;
    size_t e_offset_d = 1ll * shard_dim * hidden_dim;

    for (size_t l = 0; l < layers_per_stage; l++) {
      for (size_t e = 0; e < experts_per_gpu; e++) {
        size_t base = 1ll * l * l_offset + 1ll * e * e_offset;

        for (size_t i = 0; i < shard_dim; i++) {
          for (size_t h = 0; h < hidden_dim; h++) {
            float value = w_mlp2_ptr[base + h * inter_dim + i];
            tmp[i * hidden_dim + h] = bf16(value);
          }
        }

        size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d;
        CHECK_HIP(
          hipMemcpy(d_buf + d_offset, tmp, tmp_elems * sizeof(bf16), hipMemcpyHostToDevice));
      }
    }

    CHECK_HIP(hipStreamSynchronize(stream));
    free(tmp);

    // weights->w_mlp2->to_device(0);
    printf("End alloc mlp2\n");
    fflush(stdout);
  }

  /** w_mlp2 EXPERT PARALLELISM
    // Apply tensor parallelism (tp) offset for the expert dimension
    w_mlp2_ptr += 1ll * tp_offset * (size_t)p->hidden_dim * (size_t)p->intermediate_dim;
    bf16 *w_mlp2_d_buf = (bf16 *)weights->w_mlp2->d_buf;

    for (int l = 0; l < layers_per_stage; l++) {
      // Calculate the total number of elements for one layer on this device
      size_t N_ = experts_per_gpu * (size_t)p->hidden_dim * (size_t)p->intermediate_dim;
      bf16 *temp_bf16 = (bf16 *)malloc(N_ * sizeof(bf16));

      // Convert from FP32 to BF16 on the host
      for (size_t i = 0; i < N_; i++) {
        temp_bf16[i] = hip_bfloat16(w_mlp2_ptr[i]);
      }

      // Copy the converted data to the device
      CHECK_HIP(hipMemcpy(w_mlp2_d_buf, temp_bf16, N_ * sizeof(bf16), hipMemcpyHostToDevice));
      free(temp_bf16);

      // Advance host pointer to the start of the next layer's full set of experts
      w_mlp2_ptr += 1ll * (size_t)p->n_experts * (size_t)p->hidden_dim * (size_t)p->intermediate_dim;
      // Advance device pointer to the next layer's destination block
      w_mlp2_d_buf += 1ll * N_;
    }
  */

  // Initialize pointer with pipeline parallelism (pp) offset
  float *b_mlp2_ptr = w->b_mlp2 + 1ll * pp_offset * (size_t)p->n_experts * (size_t)p->hidden_dim;
  weights->b_mlp2 = new Tensor({layers_per_stage, experts_per_gpu, (size_t)p->hidden_dim},
                               b_mlp2_ptr, stream, DType::BF16);

  /** b_mlp2 EXPERT PARALLELISM
    // Apply tensor parallelism (tp) offset for the expert dimension
    b_mlp2_ptr += 1ll * tp_offset * (size_t)p->hidden_dim;
    bf16 *b_mlp2_d_buf = (bf16 *)weights->b_mlp2->d_buf;

    for (int l = 0; l < layers_per_stage; l++) {
      // Calculate the total number of elements for one layer on this device
      size_t N_ = experts_per_gpu * (size_t)p->hidden_dim;
      bf16 *temp_bf16 = (bf16 *)malloc(N_ * sizeof(bf16));

      // Convert from FP32 to BF16 on the host
      for (size_t i = 0; i < N_; i++) {
        temp_bf16[i] = hip_bfloat16(b_mlp2_ptr[i]);
      }

      // Copy the converted data to the device
      CHECK_HIP(hipMemcpy(b_mlp2_d_buf, temp_bf16, N_ * sizeof(bf16), hipMemcpyHostToDevice));
      free(temp_bf16);

      // Advance host pointer to the start of the next layer's full set of experts
      b_mlp2_ptr += 1ll * (size_t)p->n_experts * (size_t)p->hidden_dim;
      // Advance device pointer to the next layer's destination block
      b_mlp2_d_buf += 1ll * N_;
    }
  */

  if (pp_rank + 1 == PP) {
    weights->rms_out_w = new Tensor({(size_t)p->hidden_dim}, w->rms_out_w, stream, DType::BF16);
    
    {
      size_t vocab_size = p->vocab_size;
      size_t hidden_dim = p->hidden_dim;
      size_t shard_vocab_size = vocab_size / TP;

      weights->out_buffer = new Tensor({hidden_dim, shard_vocab_size}, stream, DType::BF16);
      float *w_out_ptr = w->out + 1ll * tp_rank * shard_vocab_size * hidden_dim;

      for (size_t i = 0; i < shard_vocab_size; i++) {
        for (size_t j = 0; j < hidden_dim; j++) {
          float value = w_out_ptr[i * hidden_dim + j];
          weights->out_buffer->buf[j * shard_vocab_size + i] = bf16(value);
        }
      }

      weights->out_buffer->to_device(stream);
    }
  } else {
    weights->rms_out_w = nullptr;
    
    weights->out_buffer = nullptr;
  }
}

void our_init_run_state(RunState *s, Config *p, OurRunState *rs, int device_id,
                        hipStream_t stream) {
  CHECK_HIP(hipSetDevice(device_id));
  
  rs->x = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, stream);

  rs->t = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, stream);
  rs->tb = new Tensor({BATCH_SIZE, (size_t)p->head_dim * p->n_attn_heads / TP}, stream);
  rs->tb_buf = new Tensor({BATCH_SIZE, (size_t)p->head_dim * p->n_attn_heads / TP}, stream);

  rs->tb2 = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, stream);
  if (device_id % TP == 0) {
    rs->tb2_buf = new Tensor({(TP - 1), BATCH_SIZE, (size_t)p->hidden_dim}, stream);
  } else {
    rs->tb2_buf = nullptr;
  }

  rs->tb3 = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->hidden_dim}, stream);
  if (device_id % TP == 0) {
    rs->tb3_buf = new Tensor(
      {(TP - 1), BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->hidden_dim}, stream);
  } else {
    rs->tb3_buf = nullptr;
  }

  rs->router_score = new Tensor({BATCH_SIZE, (size_t)p->n_experts}, stream);
  rs->topk_v = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token}, stream);
  rs->topk_i = new TensorI32({BATCH_SIZE, (size_t)p->experts_per_token}, stream);

  rs->mlp1_out =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, 2 * (size_t)p->intermediate_dim}, stream);

  rs->gate =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, stream);
  rs->up =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, stream);

  rs->gate_up =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, stream);

  rs->e_agg = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, stream);
  if (device_id % TP == 0) {
    rs->e_agg_buf = nullptr;
  } else {
    rs->e_agg_buf = nullptr;
  }

  rs->qkv = new Tensor(
    {BATCH_SIZE, ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim / TP}, stream);

  rs->q = new Tensor({BATCH_SIZE, (size_t)p->n_attn_heads * p->head_dim / TP}, stream);
  rs->k = new Tensor({BATCH_SIZE, (size_t)p->n_kv_heads * p->head_dim / TP}, stream);
  rs->v = new Tensor({BATCH_SIZE, (size_t)p->n_kv_heads * p->head_dim / TP}, stream);

  rs->att = new Tensor({BATCH_SIZE, (size_t)p->n_attn_heads, (size_t)p->seq_len}, stream);
  rs->logits = new Tensor({BATCH_SIZE, (size_t)p->vocab_size}, stream);
  rs->tmp_logits = new Tensor({BATCH_SIZE, (size_t)p->vocab_size / TP}, stream);

#ifdef KV16
  printf("using BF16 KV cache\n");
  rs->key_cache = new Tensor({BATCH_SIZE, ((size_t)p->n_layers / PP), (size_t)p->seq_len,
                              (size_t)p->n_kv_heads * p->head_dim / TP},
                             stream, DType::BF16);
  rs->value_cache = new Tensor({BATCH_SIZE, ((size_t)p->n_layers / PP), (size_t)p->seq_len,
                                (size_t)p->n_kv_heads * p->head_dim / TP},
                               stream, DType::BF16);
#else
  printf("using FP32 KV cache\n");
  rs->key_cache = new Tensor({BATCH_SIZE, ((size_t)p->n_layers / PP), (size_t)p->seq_len,
                              (size_t)p->n_kv_heads * p->head_dim / TP},
                             stream);
  rs->value_cache = new Tensor({BATCH_SIZE, ((size_t)p->n_layers / PP), (size_t)p->seq_len,
                                (size_t)p->n_kv_heads * p->head_dim / TP},
                               stream);
#endif
                

  // mask needs to be batch because they are not zero_allocated
  rs->mask = new Tensor({BATCH_SIZE, (size_t)p->seq_len, (size_t)p->seq_len}, stream);
  size_t single_mask_elems = (size_t)p->seq_len * p->seq_len;
  for (int b = 0; b < BATCH_SIZE; b++) {
    float *dest_ptr = rs->mask->buf + b * single_mask_elems;
    memcpy(dest_ptr, s->mask, single_mask_elems * sizeof(float));
  }
  rs->mask->to_device(stream);

  // MoE buffers
  rs->sorted_pair_ids = new TensorI32({(size_t)BATCH_SIZE * p->experts_per_token}, stream);
  rs->expert_offsets = new TensorI32({(size_t)(p->n_experts + 1)}, stream);
  rs->x_packed =
    new Tensor({(size_t)BATCH_SIZE * p->experts_per_token, (size_t)p->hidden_dim}, stream);

  rs->cos_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2}, stream);
  rs->sin_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2}, stream);
  RopePrecomputeCS(p, rs->cos_tensor, rs->sin_tensor);
}

#endif

void our_init(Transformer *transformer, OurTransformerWeights *weights, OurRunState *rs,
              hipStream_t *total_streams, hipTotalEvents_t *total_events) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  total_events->tp_ready = new hipEvent_t[TOTAL_GPUS_NEEDED];
  total_events->tp_finish = new hipEvent_t[TOTAL_GPUS_NEEDED];
  total_events->pp_sync = new hipEvent_t[TOTAL_GPUS_NEEDED];

#pragma omp parallel for num_threads(TOTAL_GPUS_NEEDED)
  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    CHECK_HIP(hipSetDevice(i));

#ifdef RUN_20B
    CHECK_HIP(hipStreamCreate(&total_streams[i]));
    CHECK_HIP(hipEventCreate(&(total_events->tp_ready[i])));
    CHECK_HIP(hipEventCreate(&(total_events->tp_finish[i])));
    CHECK_HIP(hipEventCreate(&(total_events->pp_sync[i])));
    our_init_weights(w, p, &weights[i], i, total_streams[i]);
    our_init_run_state(s, p, &rs[i], i, total_streams[i]);
#else
    CHECK_HIP(hipStreamCreate(&total_streams[i]));
    CHECK_HIP(hipEventCreate(&(total_events->tp_ready[i])));
    CHECK_HIP(hipEventCreate(&(total_events->tp_finish[i])));
    CHECK_HIP(hipEventCreate(&(total_events->pp_sync[i])));
    our_init_weights(w, p, &weights[i], i, total_streams[i]);
    our_init_run_state(s, p, &rs[i], i, total_streams[i]);
#endif
  }
}

void our_free_each(OurTransformerWeights *weights, OurRunState *rs) {
  // weights
  if (weights->token_embedding_table)
    delete weights->token_embedding_table;
  if (weights->rms_attn_w)
    delete weights->rms_attn_w;
  if (weights->rms_ffn_w)
    delete weights->rms_ffn_w;
  if (weights->w_qkv)
    delete weights->w_qkv;
  if (weights->w_o)
    delete weights->w_o;
  if (weights->b_qkv)
    delete weights->b_qkv;
  if (weights->b_o)
    delete weights->b_o;
  if (weights->attn_sinks)
    delete weights->attn_sinks;
  if (weights->w_router)
    delete weights->w_router;
  if (weights->b_router)
    delete weights->b_router;
  if (weights->w_mlp1)
    delete weights->w_mlp1;
  if (weights->w_mlp2)
    delete weights->w_mlp2;
  if (weights->b_mlp1)
    delete weights->b_mlp1;
  if (weights->b_mlp2)
    delete weights->b_mlp2;
  if (weights->rms_out_w)
    delete weights->rms_out_w;
  if (weights->out)
    delete weights->out;

  // delete rs
  if (rs->x)
    delete rs->x;
  if (rs->t)
    delete rs->t;
  if (rs->tb)
    delete rs->tb;
  if (rs->tb2)
    delete rs->tb2;
  if (rs->router_score)
    delete rs->router_score;
  if (rs->topk_v)
    delete rs->topk_v;
  if (rs->topk_i)
    delete rs->topk_i;
  if (rs->mlp1_out)
    delete rs->mlp1_out;
  if (rs->gate)
    delete rs->gate;
  if (rs->up)
    delete rs->up;
  if (rs->gate_up)
    delete rs->gate_up;
  if (rs->e_agg)
    delete rs->e_agg;
  if (rs->e_agg_buf)
    delete rs->e_agg_buf;
  if (rs->qkv)
    delete rs->qkv;
  if (rs->q)
    delete rs->q;
  if (rs->k)
    delete rs->k;
  if (rs->v)
    delete rs->v;
  if (rs->att)
    delete rs->att;
  if (rs->logits)
    delete rs->logits;
  if (rs->key_cache)
    delete rs->key_cache;
  if (rs->value_cache)
    delete rs->value_cache;
  if (rs->mask)
    delete rs->mask;

  // MoE buffer
  if (rs->sorted_pair_ids)
    delete rs->sorted_pair_ids;
  if (rs->expert_offsets)
    delete rs->expert_offsets;
  if (rs->x_packed)
    delete rs->x_packed;

  if (rs->tokens_buf)
    delete rs->tokens_buf;

  // Others
  if (rs->cos_tensor)
    delete rs->cos_tensor;
  if (rs->sin_tensor)
    delete rs->sin_tensor;
}

void our_free(OurTransformerWeights *weights, OurRunState *rs, hipStream_t *total_streams,
              hipTotalEvents_t *total_events) {
  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    fprintf(stderr, "Freeing weights and run state of id %d\n", i);

    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipStreamDestroy(total_streams[i]));
    CHECK_HIP(hipEventDestroy(total_events->tp_ready[i]));
    CHECK_HIP(hipEventDestroy(total_events->tp_finish[i]));
    CHECK_HIP(hipEventDestroy(total_events->pp_sync[i]));

    our_free_each(&weights[i], &rs[i]);

    fprintf(stderr, "Finish freeing weights and run state of id %d, moving on to actual pointer\n",
            i);
    fprintf(stderr, "Finish everything id %d\n", i);
  }

  delete total_events->tp_ready;
  delete total_events->tp_finish;
  delete total_events->pp_sync;
}
