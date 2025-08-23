#include "model.hpp"
#include <vector>

void convert_and_copy_fp32_to_bf16(bf16 **d_ptr, const float *h_ptr, size_t num_elements) {
  size_t bf16_size = num_elements * sizeof(bf16);

  bf16 *h_bf16_buffer = (bf16 *)malloc(bf16_size);

  for (size_t i = 0; i < num_elements; i++) {
    h_bf16_buffer[i] = hip_bfloat16(h_ptr[i]);
  }

  CHECK_HIP(hipMalloc(d_ptr, bf16_size));
  CHECK_HIP(hipMemcpy(*d_ptr, h_bf16_buffer, bf16_size, hipMemcpyHostToDevice));

  free(h_bf16_buffer);
}

void device_alloc_run_state(DeviceRunState *rs_d, Config *config) {
  int kv_dim = config->head_dim * config->n_kv_heads;

  size_t hidden_dim_size = (size_t)config->hidden_dim * sizeof(float);
  size_t attn_heads_size = (size_t)config->head_dim * config->n_attn_heads * sizeof(float);
  size_t n_experts_size = (size_t)config->n_experts * sizeof(float);
  size_t experts_per_token_float_size = (size_t)config->experts_per_token * sizeof(float);
  size_t experts_per_token_int_size = (size_t)config->experts_per_token * sizeof(int);
  size_t intermediate_dim_size = (size_t)config->intermediate_dim * sizeof(float);
  size_t qkv_size =
    (size_t)config->head_dim * (config->n_attn_heads + 2 * config->n_kv_heads) * sizeof(float);
  size_t q_size = (size_t)config->n_attn_heads * config->head_dim * sizeof(float);
  size_t cache_size = (size_t)config->n_layers * config->seq_len * kv_dim *
                      sizeof(float);  // Allocated for only 1 prompt
  size_t att_size =
    (size_t)config->n_attn_heads * config->seq_len * sizeof(float);  // buffer for attention score
  size_t logit_size = (size_t)config->vocab_size * sizeof(float);

  CHECK_HIP(hipMalloc(&rs_d->x, hidden_dim_size));
  CHECK_HIP(hipMemset(rs_d->x, 0, hidden_dim_size));

  CHECK_HIP(hipMalloc(&rs_d->t, hidden_dim_size));
  CHECK_HIP(hipMemset(rs_d->t, 0, hidden_dim_size));

  CHECK_HIP(hipMalloc(&rs_d->tb, attn_heads_size));
  CHECK_HIP(hipMemset(rs_d->tb, 0, attn_heads_size));

  CHECK_HIP(hipMalloc(&rs_d->tb2, hidden_dim_size));
  CHECK_HIP(hipMemset(rs_d->tb2, 0, hidden_dim_size));

  CHECK_HIP(hipMalloc(&rs_d->router_score, n_experts_size));
  CHECK_HIP(hipMemset(rs_d->router_score, 0, n_experts_size));

  CHECK_HIP(hipMalloc(&rs_d->topk_v, experts_per_token_float_size));
  CHECK_HIP(hipMemset(rs_d->topk_v, 0, experts_per_token_float_size));

  CHECK_HIP(hipMalloc(&rs_d->topk_i, experts_per_token_int_size));
  CHECK_HIP(hipMemset(rs_d->topk_i, 0, experts_per_token_int_size));

  CHECK_HIP(hipMalloc(&rs_d->mlp1_out, 2 * intermediate_dim_size));
  CHECK_HIP(hipMemset(rs_d->mlp1_out, 0, 2 * intermediate_dim_size));

  CHECK_HIP(hipMalloc(&rs_d->gate, intermediate_dim_size));
  CHECK_HIP(hipMemset(rs_d->gate, 0, intermediate_dim_size));

  CHECK_HIP(hipMalloc(&rs_d->up, intermediate_dim_size));
  CHECK_HIP(hipMemset(rs_d->up, 0, intermediate_dim_size));

  CHECK_HIP(hipMalloc(&rs_d->gate_up, intermediate_dim_size));
  CHECK_HIP(hipMemset(rs_d->gate_up, 0, intermediate_dim_size));

  CHECK_HIP(hipMalloc(&rs_d->e_agg, hidden_dim_size));
  CHECK_HIP(hipMemset(rs_d->e_agg, 0, hidden_dim_size));

  //   CHECK_HIP(hipMalloc(&rs_d->gate_up, intermediate_dim_size));
  //   CHECK_HIP(hipMemset(rs_d->gate_up, 0, intermediate_dim_size));

  CHECK_HIP(hipMalloc(&rs_d->qkv, qkv_size));
  CHECK_HIP(hipMemset(rs_d->qkv, 0, qkv_size));

  CHECK_HIP(hipMalloc(&rs_d->q, q_size));
  CHECK_HIP(hipMemset(rs_d->q, 0, q_size));

  CHECK_HIP(hipMalloc(&rs_d->key_cache, cache_size));
  CHECK_HIP(hipMemset(rs_d->key_cache, 0, cache_size));

  CHECK_HIP(hipMalloc(&rs_d->value_cache, cache_size));
  CHECK_HIP(hipMemset(rs_d->value_cache, 0, cache_size));

  CHECK_HIP(hipMalloc(&rs_d->att, att_size));
  CHECK_HIP(hipMemset(rs_d->att, 0, att_size));

  CHECK_HIP(hipMalloc(&rs_d->logits, logit_size));
  CHECK_HIP(hipMemset(rs_d->logits, 0, logit_size));

  if (config->sliding_window > 0) {
    size_t mask_size = (size_t)config->seq_len * config->seq_len * sizeof(float);

    float *h_mask = (float *)malloc(mask_size);
    if (!h_mask) {
      fprintf(stderr, "malloc failed for host mask.\n");
      exit(EXIT_FAILURE);
    }

    memset(h_mask, 0, mask_size);

    for (int i = 0; i < config->seq_len; i++) {
      for (int j = 0; j < config->seq_len; j++) {
        if (i - j >= config->sliding_window) {
          h_mask[i * config->seq_len + j] = -INFINITY;
        }
      }
    }

    CHECK_HIP(hipMalloc(&rs_d->mask, mask_size));
    CHECK_HIP(hipMemcpy(rs_d->mask, h_mask, mask_size, hipMemcpyHostToDevice));

    free(h_mask);
  } else {
    rs_d->mask = nullptr;
  }
}

void device_copy_model_weight(DeviceTransformerWeights *d_w, TransformerWeights *t,
                              Config *config) {
  // 1. Token embedding table
  size_t token_embedding_size = (size_t)config->vocab_size * config->hidden_dim * sizeof(float);
  CHECK_HIP(hipMalloc(&d_w->token_embedding_table, token_embedding_size));
  CHECK_HIP(hipMemcpy(d_w->token_embedding_table, t->token_embedding_table, token_embedding_size,
                      hipMemcpyHostToDevice));

  // 2. Output (unembedding)
  size_t out_size = (size_t)config->vocab_size * config->hidden_dim * sizeof(float);
  CHECK_HIP(hipMalloc(&d_w->out, out_size));
  CHECK_HIP(hipMemcpy(d_w->out, t->out, out_size, hipMemcpyHostToDevice));

  // 3. RMSNorm weights
  size_t rms_attn_size = (size_t)config->n_layers * config->hidden_dim * sizeof(float);
  CHECK_HIP(hipMalloc(&d_w->rms_attn_w, rms_attn_size));
  CHECK_HIP(hipMemcpy(d_w->rms_attn_w, t->rms_attn_w, rms_attn_size, hipMemcpyHostToDevice));

  size_t rms_ffn_size = (size_t)config->n_layers * config->hidden_dim * sizeof(float);
  CHECK_HIP(hipMalloc(&d_w->rms_ffn_w, rms_ffn_size));
  CHECK_HIP(hipMemcpy(d_w->rms_ffn_w, t->rms_ffn_w, rms_ffn_size, hipMemcpyHostToDevice));

  size_t rms_out_size = (size_t)config->hidden_dim * sizeof(float);
  CHECK_HIP(hipMalloc(&d_w->rms_out_w, rms_out_size));
  CHECK_HIP(hipMemcpy(d_w->rms_out_w, t->rms_out_w, rms_out_size, hipMemcpyHostToDevice));

  // 4. Attention weights
  size_t w_qkv_size =
    (size_t)config->n_layers * config->hidden_dim *
    (config->head_dim * config->n_attn_heads + 2 * config->head_dim * config->n_kv_heads) *
    sizeof(float);
  CHECK_HIP(hipMalloc(&d_w->w_qkv, w_qkv_size));
  CHECK_HIP(hipMemcpy(d_w->w_qkv, t->w_qkv, w_qkv_size, hipMemcpyHostToDevice));

  size_t b_qkv_size =
    (size_t)config->n_layers *
    (config->head_dim * config->n_attn_heads + 2 * config->head_dim * config->n_kv_heads) *
    sizeof(float);
  CHECK_HIP(hipMalloc(&d_w->b_qkv, b_qkv_size));
  CHECK_HIP(hipMemcpy(d_w->b_qkv, t->b_qkv, b_qkv_size, hipMemcpyHostToDevice));

  size_t w_o_size = (size_t)config->n_layers * (config->head_dim * config->n_attn_heads) *
                    config->hidden_dim * sizeof(float);
  CHECK_HIP(hipMalloc(&d_w->w_o, w_o_size));
  CHECK_HIP(hipMemcpy(d_w->w_o, t->w_o, w_o_size, hipMemcpyHostToDevice));

  size_t b_o_size = (size_t)config->n_layers * config->hidden_dim * sizeof(float);
  CHECK_HIP(hipMalloc(&d_w->b_o, b_o_size));
  CHECK_HIP(hipMemcpy(d_w->b_o, t->b_o, b_o_size, hipMemcpyHostToDevice));

  size_t attn_sinks_size = (size_t)config->n_layers * config->n_attn_heads * sizeof(float);
  CHECK_HIP(hipMalloc(&d_w->attn_sinks, attn_sinks_size));
  CHECK_HIP(hipMemcpy(d_w->attn_sinks, t->attn_sinks, attn_sinks_size, hipMemcpyHostToDevice));

  // 5. Router weights
  size_t w_router_size =
    (size_t)config->n_layers * config->hidden_dim * config->n_experts * sizeof(float);
  CHECK_HIP(hipMalloc(&d_w->w_router, w_router_size));
  CHECK_HIP(hipMemcpy(d_w->w_router, t->w_router, w_router_size, hipMemcpyHostToDevice));

  size_t b_router_size = (size_t)config->n_layers * config->n_experts * sizeof(float);
  CHECK_HIP(hipMalloc(&d_w->b_router, b_router_size));
  CHECK_HIP(hipMemcpy(d_w->b_router, t->b_router, b_router_size, hipMemcpyHostToDevice));

  // 6. MoE weights
  size_t w_mlp1_size = (size_t)config->n_layers * config->n_experts * 2 * config->intermediate_dim *
                       config->hidden_dim;
  convert_and_copy_fp32_to_bf16(&d_w->w_mlp1, t->w_mlp1, w_mlp1_size);
  // CHECK_HIP(hipMalloc(&d_w->w_mlp1, w_mlp1_size));
  // CHECK_HIP(hipMemcpy(d_w->w_mlp1, t->w_mlp1, w_mlp1_size, hipMemcpyHostToDevice));

  size_t b_mlp1_size = (size_t)config->n_layers * config->n_experts * 2 * config->intermediate_dim;
  convert_and_copy_fp32_to_bf16(&d_w->b_mlp1, t->b_mlp1, b_mlp1_size);
  // CHECK_HIP(hipMalloc(&d_w->b_mlp1, b_mlp1_size));
  // CHECK_HIP(hipMemcpy(d_w->b_mlp1, t->b_mlp1, b_mlp1_size, hipMemcpyHostToDevice));

  size_t w_mlp2_size =
    (size_t)config->n_layers * config->n_experts * config->hidden_dim * config->intermediate_dim;
  convert_and_copy_fp32_to_bf16(&d_w->w_mlp2, t->w_mlp2, w_mlp2_size);
  // CHECK_HIP(hipMalloc(&d_w->w_mlp2, w_mlp2_size));
  // CHECK_HIP(hipMemcpy(d_w->w_mlp2, t->w_mlp2, w_mlp2_size, hipMemcpyHostToDevice));

  size_t b_mlp2_size = (size_t)config->n_layers * config->n_experts * config->hidden_dim;
  convert_and_copy_fp32_to_bf16(&d_w->b_mlp2, t->b_mlp2, b_mlp2_size);
  // CHECK_HIP(hipMalloc(&d_w->b_mlp2, b_mlp2_size));
  // CHECK_HIP(hipMemcpy(d_w->b_mlp2, t->b_mlp2, b_mlp2_size, hipMemcpyHostToDevice));
}

void free_device_run_state(DeviceRunState *rs) {
  CHECK_HIP(hipFree(rs->x));
  CHECK_HIP(hipFree(rs->t));
  CHECK_HIP(hipFree(rs->tb));
  CHECK_HIP(hipFree(rs->tb2));
  CHECK_HIP(hipFree(rs->router_score));
  CHECK_HIP(hipFree(rs->topk_v));
  CHECK_HIP(hipFree(rs->topk_i));
  CHECK_HIP(hipFree(rs->mlp1_out));
  CHECK_HIP(hipFree(rs->gate));
  CHECK_HIP(hipFree(rs->up));
  CHECK_HIP(hipFree(rs->gate_up));
  CHECK_HIP(hipFree(rs->e_agg));
  CHECK_HIP(hipFree(rs->qkv));
  CHECK_HIP(hipFree(rs->q));
  CHECK_HIP(hipFree(rs->key_cache));
  CHECK_HIP(hipFree(rs->value_cache));
  CHECK_HIP(hipFree(rs->att));
  CHECK_HIP(hipFree(rs->logits));

  // Con trỏ mask chỉ được cấp phát có điều kiện, nên cần kiểm tra trước khi giải phóng
  if (rs->mask != nullptr) {
    CHECK_HIP(hipFree(rs->mask));
  }
}

void free_model_weight(DeviceTransformerWeights *d_weights) {
  CHECK_HIP(hipFree(d_weights->token_embedding_table));
  CHECK_HIP(hipFree(d_weights->out));
  CHECK_HIP(hipFree(d_weights->rms_attn_w));
  CHECK_HIP(hipFree(d_weights->rms_ffn_w));
  CHECK_HIP(hipFree(d_weights->rms_out_w));
  CHECK_HIP(hipFree(d_weights->w_qkv));
  CHECK_HIP(hipFree(d_weights->b_qkv));
  CHECK_HIP(hipFree(d_weights->w_o));
  CHECK_HIP(hipFree(d_weights->b_o));
  CHECK_HIP(hipFree(d_weights->attn_sinks));
  CHECK_HIP(hipFree(d_weights->w_router));
  CHECK_HIP(hipFree(d_weights->b_router));
  CHECK_HIP(hipFree(d_weights->w_mlp1));
  CHECK_HIP(hipFree(d_weights->b_mlp1));
  CHECK_HIP(hipFree(d_weights->w_mlp2));
  CHECK_HIP(hipFree(d_weights->b_mlp2));
}

float *hip_forward(DeviceTransformerWeights *w, DeviceRunState *rs, Config *config, int token,
                   int pos, hipStream_t stream, const float *d_rope_cos, const float *d_rope_sin) {
  int head_dim = config->head_dim;
  int hidden_dim = config->hidden_dim;
  int kv_dim = head_dim * config->n_kv_heads;
  int n_experts = config->n_experts;
  int experts_per_token = config->experts_per_token;
  int intermediate_dim = config->intermediate_dim;
  int n_q_heads = config->n_attn_heads;
  int n_kv_heads = config->n_kv_heads;

  // 1. Embedding lookup
  EmbeddingLookupGPU(w->token_embedding_table, token, rs->x, hidden_dim, stream);

  for (int l = 0; l < config->n_layers; l++) {
    const float *w_rms_attn = w->rms_attn_w + 1ll * l * hidden_dim;
    const float *w_qkv =
      w->w_qkv + 1ll * l * hidden_dim * (head_dim * n_q_heads + 2 * head_dim * n_kv_heads);
    const float *b_qkv = w->b_qkv + 1ll * l * (head_dim * n_q_heads + 2 * head_dim * n_kv_heads);
    const float *w_o = w->w_o + 1ll * l * (head_dim * n_q_heads) * hidden_dim;
    const float *b_o = w->b_o + 1ll * l * hidden_dim;
    const float *attn_sinks = w->attn_sinks + 1ll * l * n_q_heads;

    // 2. Pre-attention RMSNorm
    RMSNormGPU(rs->x, w_rms_attn, rs->t, hidden_dim, 1e-5f, stream);

    CHECK_HIP(hipStreamSynchronize(stream));
    float *h_buffer = (float *)malloc(hidden_dim * sizeof(float));
    CHECK_HIP(hipMemcpy(h_buffer, rs->t, hidden_dim * sizeof(float), hipMemcpyDeviceToHost));

    for (int i = 0; i < 10; i++) {
      printf("t[%d] = %f\n", i, h_buffer[i]);
    }

    // 3. Compute Q, K, V
    // w_qkv (head_dim * (n_attn_head + 2 * n_kv_head), hidden_dim) @ rs->t (hidden_dim, ) + b_qkv => rs->qkv
    QKVGemmGPU(w_qkv, b_qkv, rs->t, rs->qkv, hidden_dim, head_dim, hidden_dim,
               head_dim * (n_q_heads + 2 * n_kv_heads), stream);

    // int qkv_dim = head_dim * (n_q_heads + 2 * n_kv_heads);
    // CHECK_HIP(hipStreamSynchronize(stream));
    // float *h_buffer = (float *)malloc(qkv_dim * sizeof(float));
    // CHECK_HIP(hipMemcpy(h_buffer, rs->qkv, qkv_dim * sizeof(float), hipMemcpyDeviceToHost));

    // for (int i = 0; i < 10; i++) {
    //   printf("qkv[%d] = %f\n", i, h_buffer[i]);
    // }

    // 4. Split QKV and RoPE
    const int loff = l * config->seq_len * kv_dim;  // layer offset in KV cache
    float *k_pos = rs->key_cache + loff + pos * kv_dim;
    float *v_pos = rs->value_cache + loff + pos * kv_dim;

    const float *rope_cos_pos = d_rope_cos + pos * (head_dim / 2);
    const float *rope_sin_pos = d_rope_sin + pos * (head_dim / 2);

    QKVEpilogueSplitRoPECacheGPU(rs->qkv, rs->q, k_pos, v_pos, rope_cos_pos, rope_sin_pos, head_dim,
                                 n_q_heads, n_kv_heads, stream);

    // int q_dim = n_q_heads * head_dim;
    // CHECK_HIP(hipStreamSynchronize(stream));
    // float *q_buffer = (float *)malloc(q_dim * sizeof(float));
    // float *k_buffer = (float *)malloc(kv_dim * sizeof(float));
    // float *v_buffer = (float *)malloc(kv_dim * sizeof(float));

    // CHECK_HIP(hipMemcpy(q_buffer, rs->q, q_dim * sizeof(float), hipMemcpyDeviceToHost));
    // CHECK_HIP(hipMemcpy(k_buffer, k_pos, kv_dim * sizeof(float), hipMemcpyDeviceToHost));
    // CHECK_HIP(hipMemcpy(v_buffer, v_pos, kv_dim * sizeof(float), hipMemcpyDeviceToHost));

    // printf("Q vector (10 elements):\n");
    // for (int i = 0; i < 10; i++) {
    //   printf("\tq[%d] = %f\n", i, q_buffer[i]);
    // }

    // printf("K vector (10 elements):\n");
    // for (int i = 0; i < 10; i++) {
    //   printf("\tk[%d] = %f\n", i, k_buffer[i]);
    // }

    // printf("V vector (10 elements):\n");
    // for (int i = 0; i < 10; i++) {
    //   printf("\tv[%d] = %f\n", i, v_buffer[i]);
    // }

    // 5. Multi-Head Attention
    const float *k_cache = rs->key_cache + loff;
    const float *v_cache = rs->value_cache + loff;
    const float *mask_row =
      (config->sliding_window > 0) ? (rs->mask + (size_t)pos * config->seq_len) : nullptr;

    SingleQueryAttentionGPU(rs->q, k_cache, v_cache, mask_row, attn_sinks, rs->tb, head_dim,
                            n_q_heads, n_q_heads / n_kv_heads, kv_dim, pos + 1, pos, stream);

    // int tb_dim = head_dim * n_q_heads;
    // CHECK_HIP(hipStreamSynchronize(stream));
    // float *h_buffer = (float *)malloc(tb_dim * sizeof(float));
    // CHECK_HIP(hipMemcpy(h_buffer, rs->tb, tb_dim * sizeof(float), hipMemcpyDeviceToHost));

    // for (int i = 0; i < 10; i++) {
    //   printf("tb[%d] = %f\n", i, h_buffer[i]);
    // }

    // 6. Post-attention Linear and residual
    // rs->x += w_o * rs->tb + b_o
    LinearBiasResidualGPU(w_o, rs->tb, b_o, rs->x, head_dim * n_q_heads, hidden_dim, stream);

    // CHECK_HIP(hipStreamSynchronize(stream));
    // float *h_buffer = (float *)malloc(hidden_dim * sizeof(float));
    // CHECK_HIP(hipMemcpy(h_buffer, rs->x, hidden_dim * sizeof(float), hipMemcpyDeviceToHost));

    // for (int i = 0; i < 10; i++) {
    //   printf("x[%d] = %f\n", i, h_buffer[i]);
    // }

    // --- MOE FFN BLOCK ---

    // Tính toán con trỏ offset cho FFN/MoE của layer hiện tại (l)
    const float *w_rms_ffn = w->rms_ffn_w + 1ll * l * hidden_dim;
    const float *w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
    const float *b_router = w->b_router + 1ll * l * n_experts;
    const bf16 *w_mlp1 = w->w_mlp1;
    const bf16 *b_mlp1 = w->b_mlp1;
    const bf16 *w_mlp2 = w->w_mlp2;
    const bf16 *b_mlp2 = w->b_mlp2;

    // 7. Pre-FFN RMSNorm
    RMSNormGPU(rs->x, w_rms_ffn, rs->t, hidden_dim, 1e-5f, stream);
    // CHECK_HIP(hipStreamSynchronize(stream));

    // float *h_buffer = (float *)malloc(hidden_dim * sizeof(float));
    // CHECK_HIP(hipMemcpy(h_buffer, rs->t, hidden_dim * sizeof(float), hipMemcpyDeviceToHost));

    // for (int i = 0; i < 10; i++) {
    //   printf("t[%d] = %f\n", i, h_buffer[i]);
    // }

    // 8. Router GEMM: Tính điểm cho các expert
    RouterGemmGPU(w_router, rs->t, b_router, rs->router_score, hidden_dim, n_experts, stream);
    // float *h_buffer = (float *)malloc(n_experts * sizeof(float));

    // CHECK_HIP(hipStreamSynchronize(stream));
    // CHECK_HIP(
    //   hipMemcpy(h_buffer, rs->router_score, n_experts * sizeof(float), hipMemcpyDeviceToHost));

    // for (int i = 0; i < n_experts; i++) {
    //   printf("score[%d] = %f\n", i, h_buffer[i]);
    // }

    // 9. Top-K + Softmax
    TopKSoftmaxGPU(rs->router_score, n_experts, experts_per_token, rs->topk_v, rs->topk_i, stream);

    // float *topk_v_buffer = (float *)malloc(experts_per_token * sizeof(float));
    // int *topk_i_buffer = (int *)malloc(experts_per_token * sizeof(int));

    // CHECK_HIP(hipStreamSynchronize(stream));
    // CHECK_HIP(
    //   hipMemcpy(topk_i_buffer, rs->topk_i, experts_per_token * sizeof(int), hipMemcpyDeviceToHost));
    // CHECK_HIP(hipMemcpy(topk_v_buffer, rs->topk_v, experts_per_token * sizeof(float),
    //                     hipMemcpyDeviceToHost));

    // for (int i = 0; i < experts_per_token; i++) {
    //   printf("topk_i[%d] = %d\n", i, topk_i_buffer[i]);
    //   printf("topk_v[%d] = %f\n", i, topk_v_buffer[i]);
    // }

    size_t layer_w1_offset = 1ll * l * n_experts * (2 * intermediate_dim) * hidden_dim;
    size_t layer_b1_offset = 1ll * l * n_experts * (2 * intermediate_dim);
    size_t layer_w2_offset = 1ll * l * n_experts * hidden_dim * intermediate_dim;
    size_t layer_b2_offset = 1ll * l * n_experts * hidden_dim;

    // Lấy con trỏ trên GPU cho layer hiện tại (w_mlp1 là con trỏ gốc của DeviceTransformerWeights)
    const bf16 *d_w_mlp1_layer = w_mlp1 + layer_w1_offset;
    const bf16 *d_b_mlp1_layer = b_mlp1 + layer_b1_offset;
    const bf16 *d_w_mlp2_layer = w_mlp2 + layer_w2_offset;
    const bf16 *d_b_mlp2_layer = b_mlp2 + layer_b2_offset;

    MoEApplyTopKGPU(rs->t, d_w_mlp1_layer, d_b_mlp1_layer,  // Truyền con trỏ của layer 'l'
                    d_w_mlp2_layer, d_b_mlp2_layer,         // Truyền con trỏ của layer 'l'
                    rs->topk_i, rs->topk_v, rs->mlp1_out, rs->e_agg, hidden_dim, intermediate_dim,
                    experts_per_token, config->swiglu_limit, stream);

    // float *h_buffer = (float *)malloc(hidden_dim * sizeof(float));
    // CHECK_HIP(hipStreamSynchronize(stream));
    // CHECK_HIP(hipMemcpy(h_buffer, rs->e_agg, hidden_dim * sizeof(int), hipMemcpyDeviceToHost));

    // for (int i = 0; i < 10; i++) {
    //   printf("e_agg[%d] = %f\n", i, rs->e_agg[i]);
    // }

    // 11. Kết nối residual cuối cùng của layer
    // rs->x += rs->e_agg
    AddVectorGPU(rs->x, rs->e_agg, hidden_dim, stream);
  }
  // --- FINAL CLASSIFIER ---

  // 12. Final RMSNorm
  RMSNormGPU(rs->x, w->rms_out_w, rs->x, hidden_dim, 1e-5f, stream);
  CHECK_HIP(hipStreamSynchronize(stream));
  float *h_buffer = (float *)malloc(hidden_dim * sizeof(float));
  CHECK_HIP(hipMemcpy(h_buffer, rs->x, hidden_dim * sizeof(float), hipMemcpyDeviceToHost));

  // for (int i = 0; i < 10; i++) {
  //   printf("x[%d] = %f\n", i, h_buffer[i]);
  // }

  // 13. Classifier GEMM: Tính toán logits cuối cùng
  ClassifierGemmGPU(w->out, rs->x, rs->logits, hidden_dim, config->vocab_size, stream);

  // float *h_buffer = (float *)malloc(config->vocab_size * sizeof(float));
  // CHECK_HIP(hipStreamSynchronize(stream));
  // CHECK_HIP(
  //   hipMemcpy(h_buffer, rs->logits, config->vocab_size * sizeof(float), hipMemcpyDeviceToHost));
  // for (int i = 0; i < 15; i++) {
  //   printf("logits[%d] = %f\n", i, h_buffer[i]);
  // }

  // Trả về con trỏ device tới logits
  return rs->logits;
}
