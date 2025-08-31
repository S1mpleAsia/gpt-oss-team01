#include "../include/alloc.hpp"
#include <cmath>
#include <cstring>

void our_init_weights(TransformerWeights *w, Config *p, OurTransformerWeights *weights,
                      int device_id) {
  // Create Tensor wrappers for weight matrices
  weights->token_embedding_table =
    new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->token_embedding_table, device_id);

  weights->rms_attn_w = new Tensor({(size_t)p->n_layers * p->hidden_dim}, w->rms_attn_w, device_id);
  weights->rms_ffn_w = new Tensor({(size_t)p->n_layers * p->hidden_dim}, w->rms_ffn_w, device_id);

  // weights->w_qkv =
  //   new Tensor({(size_t)p->n_layers,
  //               ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * (size_t)p->head_dim,
  //               (size_t)p->hidden_dim},
  //              w->w_qkv, device_id);

  {
    size_t n_layers = p->n_layers;
    size_t n_heads = p->n_attn_heads + 2 * p->n_kv_heads;
    size_t head_dim = p->head_dim;
    size_t hidden_dim = p->hidden_dim;

    weights->w_qkv = new Tensor({n_layers, hidden_dim, n_heads * head_dim}, device_id);

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

    weights->w_qkv->to_device(0);
  }

  weights->b_qkv = new Tensor(
    {(size_t)p->n_layers, ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim},
    w->b_qkv, device_id);

  // weights->w_o =
  //   new Tensor({(size_t)p->n_layers, (size_t)p->hidden_dim, (size_t)p->n_attn_heads * p->head_dim},
  //              w->w_o, device_id);
  {
    size_t n_layers = p->n_layers;
    size_t hidden_dim = p->hidden_dim;
    size_t n_heads = p->n_attn_heads;
    size_t head_dim = p->head_dim;

    // Tạo tensor với shape transposed
    weights->w_o = new Tensor({n_layers, n_heads * head_dim, hidden_dim}, device_id);

    for (size_t l = 0; l < n_layers; l++) {
      for (size_t i = 0; i < hidden_dim; i++) {
        for (size_t j = 0; j < n_heads * head_dim; j++) {
          weights->w_o->buf[l * (n_heads * head_dim * hidden_dim) + j * hidden_dim + i] =
            w->w_o[l * (hidden_dim * n_heads * head_dim) + i * (n_heads * head_dim) + j];
        }
      }
    }

    weights->w_o->to_device(0);
  }

  weights->b_o = new Tensor({(size_t)p->n_layers, (size_t)p->hidden_dim}, w->b_o, device_id);

  // Tensor *attn_sinks; // (n_layers, n_attn_heads)
  weights->attn_sinks =
    new Tensor({(size_t)p->n_layers, (size_t)p->n_attn_heads}, w->attn_sinks, device_id);

  weights->w_router = new Tensor({(size_t)p->n_layers, (size_t)p->n_experts, (size_t)p->hidden_dim},
                                 w->w_router, device_id);
  weights->b_router =
    new Tensor({(size_t)p->n_layers, (size_t)p->n_experts}, w->b_router, device_id);

  // weights->w_mlp1 = new Tensor({(size_t)p->n_layers, (size_t)p->n_experts,
  //                               2 * (size_t)p->intermediate_dim, (size_t)p->hidden_dim},
  //                              w->w_mlp1, device_id, DType::BF16);

  {
    printf("Starting alloc mlp1\n");
    fflush(stdout);
    size_t n_layers = p->n_layers;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;

    weights->w_mlp1 =
      new Tensor({n_layers, n_experts, hidden_dim, 2 * inter_dim}, device_id, DType::BF16);

    bf16 *tmp = nullptr;
    size_t tmp_elems = hidden_dim * 2 * inter_dim;
    CHECK_HIP(hipHostMalloc((void **)&tmp, tmp_elems * sizeof(bf16)));

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
                                 hipMemcpyHostToDevice, 0));
      }
    }

    CHECK_HIP(hipStreamSynchronize(0));
    CHECK_HIP(hipHostFree(tmp));

    printf("End alloc mlp1\n");
    fflush(stdout);
  }

  weights->b_mlp1 =
    new Tensor({(size_t)p->n_layers, (size_t)p->n_experts, 2 * (size_t)p->intermediate_dim},
               w->b_mlp1, device_id, DType::BF16);

  // weights->w_mlp2 = new Tensor(
  //   {(size_t)p->n_layers, (size_t)p->n_experts, (size_t)p->hidden_dim, (size_t)p->intermediate_dim},
  //   w->w_mlp2, device_id, DType::BF16);

  {
    printf("Starting alloc mlp2\n");
    fflush(stdout);
    size_t n_layers = p->n_layers;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;

    weights->w_mlp2 =
      new Tensor({n_layers, n_experts, inter_dim, hidden_dim}, device_id, DType::BF16);

    bf16 *tmp = nullptr;
    size_t tmp_elems = inter_dim * hidden_dim;
    CHECK_HIP(hipHostMalloc((void **)&tmp, tmp_elems * sizeof(bf16)));

    bf16 *d_buf = (bf16 *)(weights->w_mlp2->d_buf);

    for (size_t l = 0; l < n_layers; l++) {
      for (size_t e = 0; e < n_experts; e++) {
        size_t base = l * n_experts * hidden_dim * inter_dim + e * hidden_dim * inter_dim;

        for (size_t i = 0; i < inter_dim; i++) {
          for (size_t h = 0; h < hidden_dim; h++) {
            float value = w->w_mlp2[base + h * inter_dim + i];
            tmp[i * hidden_dim + h] = bf16(value);
            // weights->w_mlp2->buf[l * n_experts * inter_dim * hidden_dim +
            //                      e * inter_dim * hidden_dim + i * hidden_dim + h] =
            //   w->w_mlp2[l * n_experts * inter_dim * hidden_dim + e * inter_dim * hidden_dim +
            //             h * inter_dim + i];
          }
        }

        size_t d_offset = l * n_experts * inter_dim * hidden_dim + e * inter_dim * hidden_dim;
        CHECK_HIP(hipMemcpyAsync(d_buf + d_offset, tmp, tmp_elems * sizeof(bf16),
                                 hipMemcpyHostToDevice, 0));
      }
    }

    CHECK_HIP(hipStreamSynchronize(0));
    CHECK_HIP(hipHostFree(tmp));

    // weights->w_mlp2->to_device(0);
    printf("End alloc mlp2\n");
    fflush(stdout);
  }

  weights->b_mlp2 = new Tensor({(size_t)p->n_layers, (size_t)p->n_experts, (size_t)p->hidden_dim},
                               w->b_mlp2, device_id, DType::BF16);

  weights->rms_out_w = new Tensor({(size_t)p->hidden_dim}, w->rms_out_w, device_id);
  // weights->out = new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->out, device_id);

  {
    size_t vocab_size = p->vocab_size;
    size_t hidden_dim = p->hidden_dim;

    weights->out = new Tensor({hidden_dim, vocab_size}, device_id, DType::BF16);

    for (size_t i = 0; i < vocab_size; i++) {
      for (size_t j = 0; j < hidden_dim; j++) {
        weights->out->buf[j * vocab_size + i] = w->out[i * hidden_dim + j];
      }
    }

    weights->out->to_device(0);
  }
}

void our_init_run_state(RunState *s, Config *p, OurRunState *rs, int device_id) {
  // Create Tensor wrappers for state buffers
  rs->x = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, device_id);

  rs->t = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, device_id);
  rs->tb = new Tensor({BATCH_SIZE, (size_t)p->head_dim * p->n_attn_heads}, device_id);
  rs->tb2 = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, device_id);

  rs->tb3 =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->hidden_dim}, device_id);

  rs->router_score = new Tensor({BATCH_SIZE, (size_t)p->n_experts}, device_id);
  rs->topk_v = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token}, device_id);
  rs->topk_i = new TensorI32({BATCH_SIZE, (size_t)p->experts_per_token}, device_id);

  rs->mlp1_out = new Tensor(
    {BATCH_SIZE, (size_t)p->experts_per_token, 2 * (size_t)p->intermediate_dim}, device_id);

  rs->gate =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, device_id);
  rs->up =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, device_id);

  // rs->gate_up = new Tensor({(size_t)p->intermediate_dim}, s->gate_up);
  rs->gate_up =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, device_id);

  rs->e_agg = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, device_id);
  rs->qkv = new Tensor(
    {BATCH_SIZE, ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim}, device_id);
  rs->q = new Tensor({BATCH_SIZE, (size_t)p->n_attn_heads * p->head_dim}, device_id);
  rs->k = new Tensor({BATCH_SIZE, (size_t)p->n_kv_heads * p->head_dim}, device_id);
  rs->v = new Tensor({BATCH_SIZE, (size_t)p->n_kv_heads * p->head_dim}, device_id);
  rs->att = new Tensor({BATCH_SIZE, (size_t)p->n_attn_heads, (size_t)p->seq_len}, device_id);
  rs->logits = new Tensor({BATCH_SIZE, (size_t)p->vocab_size}, device_id);

  rs->key_cache = new Tensor(
    {BATCH_SIZE, (size_t)p->n_layers, (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim},
    device_id);
  rs->value_cache = new Tensor(
    {BATCH_SIZE, (size_t)p->n_layers, (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim},
    device_id);

  // mask needs to be batch because they are not zero_allocated
  rs->mask =
    new Tensor({BATCH_SIZE, (size_t)p->seq_len, (size_t)p->seq_len}, s->mask, true, device_id);

  rs->cos_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2}, device_id);
  rs->sin_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2}, device_id);
  RopePrecomputeCS(p, rs->cos_tensor, rs->sin_tensor);
}

void our_init(Transformer *transformer, OurTransformerWeights *weights, OurRunState *rs) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    our_init_weights(w, p, &weights[i], i);
    our_init_run_state(s, p, &rs[i], i);
  }
}

void our_free_each(OurTransformerWeights *weights, OurRunState *rs) {
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
  delete rs->cos_tensor;
  delete rs->sin_tensor;
}

void our_free(OurTransformerWeights *weights, OurRunState *rs) {
  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    fprintf(stderr, "Freeing weights and run state of id %d\n", i);
    CHECK_HIP(hipSetDevice(i));
    our_free_each(&weights[i], &rs[i]);
    fprintf(stderr, "Finish freeing weights and run state of id %d, moving on to actual pointer\n",
            i);
    fprintf(stderr, "Finish everything id %d\n", i);
  }
}
