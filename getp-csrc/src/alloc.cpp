#include "../include/alloc.hpp"
#include <cmath>
#include <cstring>

#ifdef RUN_20B

void our_init_weights(TransformerWeights *w, Config *p, OurTransformerWeights *weights, int device_id) {
  // Create Tensor wrappers for weight matrices
  weights->token_embedding_table =
    new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->token_embedding_table, device_id);

  weights->rms_attn_w = new Tensor({(size_t)p->n_layers * p->hidden_dim}, w->rms_attn_w, device_id);
  weights->rms_ffn_w = new Tensor({(size_t)p->n_layers * p->hidden_dim}, w->rms_ffn_w, device_id);

  weights->w_qkv =
    new Tensor({(size_t)p->n_layers,
                ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * (size_t)p->head_dim,
                (size_t)p->hidden_dim},
               w->w_qkv, device_id);
  weights->b_qkv = new Tensor(
    {(size_t)p->n_layers, ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim},
    w->b_qkv, device_id);

  weights->w_o = new Tensor(
    {(size_t)p->n_layers, (size_t)p->hidden_dim, (size_t)p->n_attn_heads * p->head_dim}, w->w_o, device_id);
  weights->b_o = new Tensor({(size_t)p->n_layers, (size_t)p->hidden_dim}, w->b_o, device_id);

  // Tensor *attn_sinks; // (n_layers, n_attn_heads)
  weights->attn_sinks = new Tensor({(size_t)p->n_layers, (size_t)p->n_attn_heads}, w->attn_sinks, device_id);

  weights->w_router =
    new Tensor({(size_t)p->n_layers, (size_t)p->n_experts, (size_t)p->hidden_dim}, w->w_router, device_id);
  weights->b_router = new Tensor({(size_t)p->n_layers, (size_t)p->n_experts}, w->b_router, device_id);

  weights->w_mlp1 = new Tensor({(size_t)p->n_layers, (size_t)p->n_experts,
                                2 * (size_t)p->intermediate_dim, (size_t)p->hidden_dim},
                               w->w_mlp1, device_id, DType::BF16);
  weights->b_mlp1 =
    new Tensor({(size_t)p->n_layers, (size_t)p->n_experts, 2 * (size_t)p->intermediate_dim},
               w->b_mlp1, device_id, DType::BF16);

  weights->w_mlp2 = new Tensor(
    {(size_t)p->n_layers, (size_t)p->n_experts, (size_t)p->hidden_dim, (size_t)p->intermediate_dim},
    w->w_mlp2, device_id, DType::BF16);
  weights->b_mlp2 = new Tensor({(size_t)p->n_layers, (size_t)p->n_experts, (size_t)p->hidden_dim},
                               w->b_mlp2, device_id, DType::BF16);

  weights->rms_out_w = new Tensor({(size_t)p->hidden_dim}, w->rms_out_w, device_id);
  weights->out = new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->out, device_id);
}

void our_init_run_state(RunState *s, Config *p, OurRunState *rs, int device_id) {
  // Create Tensor wrappers for state buffers
  rs->x = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, device_id);

  rs->t = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, device_id);
  rs->tb = new Tensor({BATCH_SIZE, (size_t)p->head_dim * p->n_attn_heads}, device_id);
  rs->tb2 = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, device_id);

  rs->tb3 = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->hidden_dim}, device_id);

  rs->router_score = new Tensor({BATCH_SIZE, (size_t)p->n_experts}, device_id);
  rs->topk_v = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token}, device_id);
  rs->topk_i = new TensorI32({BATCH_SIZE, (size_t)p->experts_per_token}, device_id);

  rs->mlp1_out =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, 2 * (size_t)p->intermediate_dim}, device_id);

  rs->gate = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, device_id);
  rs->up = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, device_id);

  // rs->gate_up = new Tensor({(size_t)p->intermediate_dim}, s->gate_up);
  rs->gate_up = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, device_id);

  rs->e_agg = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, device_id);
  rs->e_agg_buf = nullptr;
  rs->qkv = new Tensor(
    {BATCH_SIZE, ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim}, device_id);
  rs->q = new Tensor({BATCH_SIZE, (size_t)p->n_attn_heads * p->head_dim}, device_id);
  rs->k = new Tensor({BATCH_SIZE, (size_t)p->n_kv_heads * p->head_dim}, device_id);
  rs->v = new Tensor({BATCH_SIZE, (size_t)p->n_kv_heads * p->head_dim}, device_id);
  rs->att = new Tensor({BATCH_SIZE, (size_t)p->n_attn_heads, (size_t)p->seq_len}, device_id);
  rs->logits = new Tensor({BATCH_SIZE, (size_t)p->vocab_size}, device_id);

  rs->key_cache = new Tensor(
    {BATCH_SIZE, (size_t)p->n_layers, (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim}, device_id);
  rs->value_cache = new Tensor(
    {BATCH_SIZE, (size_t)p->n_layers, (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim}, device_id);

  // mask needs to be batch because they are not zero_allocated
  rs->mask = new Tensor({BATCH_SIZE, (size_t)p->seq_len, (size_t)p->seq_len}, s->mask, true, device_id);

  rs->cos_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2}, device_id);
  rs->sin_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2}, device_id);
  RopePrecomputeCS(p, rs->cos_tensor, rs->sin_tensor);
}

#else

void our_init_weights(TransformerWeights *w, Config *p, OurTransformerWeights *weights, int device_id) {
  size_t layers_each = p->n_layers / PP;
  size_t experts_each = p->n_experts / TP;

  int tp_rank = device_id % TP;
  int pp_rank = (device_id % TOTAL_PIPELINES) / TP;

  long long pp_offset = 1ll * pp_rank * layers_each;
  long long tp_offset = 1ll * tp_rank * experts_each;

  // Create Tensor wrappers for weight matrices
  if (pp_rank == 0) {
    weights->token_embedding_table =
      new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->token_embedding_table, device_id);
  } else {
    weights->token_embedding_table = nullptr;
  }

  weights->rms_attn_w = new Tensor({layers_each * p->hidden_dim}, w->rms_attn_w + 1ll * pp_offset * p->hidden_dim, device_id);
  weights->rms_ffn_w = new Tensor({layers_each * p->hidden_dim}, w->rms_ffn_w + 1ll * pp_offset * p->hidden_dim, device_id);

  weights->w_qkv =
    new Tensor({layers_each,
                ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * (size_t)p->head_dim,
                (size_t)p->hidden_dim},
               w->w_qkv + 1ll * pp_offset * ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * (size_t)p->head_dim * (size_t)p->hidden_dim, device_id);
  weights->b_qkv = new Tensor(
    {layers_each, ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim},
    w->b_qkv + 1ll * pp_offset * ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim, device_id);

  weights->w_o = new Tensor(
    {layers_each, (size_t)p->hidden_dim, (size_t)p->n_attn_heads * p->head_dim}, w->w_o + 1ll * pp_offset * (size_t)p->hidden_dim * (size_t)p->n_attn_heads * p->head_dim, device_id);
  weights->b_o = new Tensor({layers_each, (size_t)p->hidden_dim}, w->b_o + 1ll * pp_offset * (size_t)p->hidden_dim, device_id);

  // Tensor *attn_sinks; // (n_layers, n_attn_heads)
  weights->attn_sinks = new Tensor({layers_each, (size_t)p->n_attn_heads}, w->attn_sinks + 1ll * pp_offset * (size_t)p->n_attn_heads, device_id);

  weights->w_router =
    new Tensor({layers_each, (size_t)p->n_experts, (size_t)p->hidden_dim}, w->w_router + 1ll * pp_offset * (size_t)p->n_experts * (size_t)p->hidden_dim, device_id);
  weights->b_router = new Tensor({layers_each, (size_t)p->n_experts}, w->b_router + 1ll * pp_offset * (size_t)p->n_experts, device_id);

  float *w_mlp1_ptr = w->w_mlp1 + 1ll * pp_offset * (size_t)p->n_experts *
                                2 * (size_t)p->intermediate_dim * (size_t)p->hidden_dim;
  weights->w_mlp1 = new Tensor(
    {layers_each, experts_each, 2 * (size_t)p->intermediate_dim, (size_t)p->hidden_dim}, w_mlp1_ptr, device_id, DType::BF16
  );

  w_mlp1_ptr += 1ll * tp_offset * 2 * (size_t)p->intermediate_dim * (size_t)p->hidden_dim;
  bf16 *w_mlp1_d_buf = (bf16 *)weights->w_mlp1->d_buf;

  for (int l = 0; l < layers_each; l++) {
    // Convert and copy for BF16
    size_t N_ = experts_each * 2 * (size_t)p->intermediate_dim *(size_t)p->hidden_dim;
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
  
  // Initialize pointer with pipeline parallelism (pp) offset
  float *b_mlp1_ptr = w->b_mlp1 + 1ll * pp_offset * (size_t)p->n_experts * 2 * (size_t)p->intermediate_dim;
  weights->b_mlp1 = new Tensor(
    {layers_each, experts_each, 2 * (size_t)p->intermediate_dim},
    b_mlp1_ptr, device_id, DType::BF16
  );

  // Apply tensor parallelism (tp) offset for the expert dimension
  b_mlp1_ptr += 1ll * tp_offset * 2 * (size_t)p->intermediate_dim;
  bf16 *b_mlp1_d_buf = (bf16 *)weights->b_mlp1->d_buf;

  for (int l = 0; l < layers_each; l++) {
    // Calculate the total number of elements for one layer on this device
    size_t N_ = experts_each * 2 * (size_t)p->intermediate_dim;
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

  // Initialize pointer with pipeline parallelism (pp) offset
  float *w_mlp2_ptr = w->w_mlp2 + 1ll * pp_offset * (size_t)p->n_experts * (size_t)p->hidden_dim * (size_t)p->intermediate_dim;
  weights->w_mlp2 = new Tensor(
    {layers_each, experts_each, (size_t)p->hidden_dim, (size_t)p->intermediate_dim}, w_mlp2_ptr, device_id, DType::BF16
  );

  // Apply tensor parallelism (tp) offset for the expert dimension
  w_mlp2_ptr += 1ll * tp_offset * (size_t)p->hidden_dim * (size_t)p->intermediate_dim;
  bf16 *w_mlp2_d_buf = (bf16 *)weights->w_mlp2->d_buf;

  for (int l = 0; l < layers_each; l++) {
    // Calculate the total number of elements for one layer on this device
    size_t N_ = experts_each * (size_t)p->hidden_dim * (size_t)p->intermediate_dim;
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

  // Initialize pointer with pipeline parallelism (pp) offset
  float *b_mlp2_ptr = w->b_mlp2 + 1ll * pp_offset * (size_t)p->n_experts * (size_t)p->hidden_dim;
  weights->b_mlp2 = new Tensor(
    {layers_each, experts_each, (size_t)p->hidden_dim},
    b_mlp2_ptr, device_id, DType::BF16
  );

  // Apply tensor parallelism (tp) offset for the expert dimension
  b_mlp2_ptr += 1ll * tp_offset * (size_t)p->hidden_dim;
  bf16 *b_mlp2_d_buf = (bf16 *)weights->b_mlp2->d_buf;

  for (int l = 0; l < layers_each; l++) {
    // Calculate the total number of elements for one layer on this device
    size_t N_ = experts_each * (size_t)p->hidden_dim;
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

  if (pp_rank + 1 == PP) {
    weights->rms_out_w = new Tensor({(size_t)p->hidden_dim}, w->rms_out_w, device_id);
    weights->out = new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->out, device_id);
  } else {
    weights->rms_out_w = nullptr;
    weights->out = nullptr;
  }
}

void our_init_run_state(RunState *s, Config *p, OurRunState *rs, int device_id) {
  // Create Tensor wrappers for state buffers
  rs->x = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, device_id);

  rs->t = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, device_id);
  rs->tb = new Tensor({BATCH_SIZE, (size_t)p->head_dim * p->n_attn_heads}, device_id);
  rs->tb2 = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, device_id);

  rs->tb3 = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->hidden_dim}, device_id);

  rs->router_score = new Tensor({BATCH_SIZE, (size_t)p->n_experts}, device_id);
  rs->topk_v = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token}, device_id);
  rs->topk_i = new TensorI32({BATCH_SIZE, (size_t)p->experts_per_token}, device_id);

  rs->mlp1_out =
    new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, 2 * (size_t)p->intermediate_dim}, device_id);

  rs->gate = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, device_id);
  rs->up = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, device_id);

  // rs->gate_up = new Tensor({(size_t)p->intermediate_dim}, s->gate_up);
  rs->gate_up = new Tensor({BATCH_SIZE, (size_t)p->experts_per_token, (size_t)p->intermediate_dim}, device_id);

  rs->e_agg = new Tensor({BATCH_SIZE, (size_t)p->hidden_dim}, device_id);
  if (device_id % TP == 0) {
    rs->e_agg_buf = new Tensor({TP, BATCH_SIZE, (size_t)p->hidden_dim}, device_id);
  } else {
    rs->e_agg_buf = nullptr;
  }
  rs->qkv = new Tensor(
    {BATCH_SIZE, ((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim}, device_id);
  rs->q = new Tensor({BATCH_SIZE, (size_t)p->n_attn_heads * p->head_dim}, device_id);
  rs->k = new Tensor({BATCH_SIZE, (size_t)p->n_kv_heads * p->head_dim}, device_id);
  rs->v = new Tensor({BATCH_SIZE, (size_t)p->n_kv_heads * p->head_dim}, device_id);
  rs->att = new Tensor({BATCH_SIZE, (size_t)p->n_attn_heads, (size_t)p->seq_len}, device_id);
  rs->logits = new Tensor({BATCH_SIZE, (size_t)p->vocab_size}, device_id);

  rs->key_cache = new Tensor(
    {BATCH_SIZE, ((size_t)p->n_layers / PP), (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim}, device_id);
  rs->value_cache = new Tensor(
    {BATCH_SIZE, ((size_t)p->n_layers / PP), (size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim}, device_id);

  // mask needs to be batch because they are not zero_allocated
  rs->mask = new Tensor({BATCH_SIZE, (size_t)p->seq_len, (size_t)p->seq_len}, s->mask, true, device_id);

  rs->cos_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2}, device_id);
  rs->sin_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->head_dim / 2}, device_id);
  RopePrecomputeCS(p, rs->cos_tensor, rs->sin_tensor);
}

#endif

void our_init(Transformer *transformer, OurTransformerWeights *weights, OurRunState *rs, hipStream_t *total_streams, hipEvent_t *total_events) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    CHECK_HIP(hipSetDevice(i));
    
    CHECK_HIP(hipStreamCreate(&total_streams[i]));
    CHECK_HIP(hipEventCreate(&total_events[i]));

    our_init_weights(w, p, &weights[i], i);
    our_init_run_state(s, p, &rs[i], i);
  }
}

void our_free_each(OurTransformerWeights *weights, OurRunState *rs) {
  // weights
  if (weights->token_embedding_table) delete weights->token_embedding_table;
  if (weights->rms_attn_w) delete weights->rms_attn_w;
  if (weights->rms_ffn_w) delete weights->rms_ffn_w;
  if (weights->w_qkv) delete weights->w_qkv;
  if (weights->w_o) delete weights->w_o;
  if (weights->b_qkv) delete weights->b_qkv;
  if (weights->b_o) delete weights->b_o;
  if (weights->attn_sinks) delete weights->attn_sinks;
  if (weights->w_router) delete weights->w_router;
  if (weights->b_router) delete weights->b_router;
  if (weights->w_mlp1) delete weights->w_mlp1;
  if (weights->w_mlp2) delete weights->w_mlp2;
  if (weights->b_mlp1) delete weights->b_mlp1;
  if (weights->b_mlp2) delete weights->b_mlp2;
  if (weights->rms_out_w) delete weights->rms_out_w;
  if (weights->out) delete weights->out;

  // delete rs
  if (rs->x) delete rs->x;
  if (rs->t) delete rs->t;
  if (rs->tb) delete rs->tb;
  if (rs->tb2) delete rs->tb2;
  if (rs->router_score) delete rs->router_score;
  if (rs->topk_v) delete rs->topk_v;
  if (rs->topk_i) delete rs->topk_i;
  if (rs->mlp1_out) delete rs->mlp1_out;
  if (rs->gate) delete rs->gate;
  if (rs->up) delete rs->up;
  if (rs->gate_up) delete rs->gate_up;
  if (rs->e_agg) delete rs->e_agg;
  if (rs->e_agg_buf) delete rs->e_agg_buf;
  if (rs->qkv) delete rs->qkv;
  if (rs->q) delete rs->q;
  if (rs->k) delete rs->k;
  if (rs->v) delete rs->v;
  if (rs->att) delete rs->att;
  if (rs->logits) delete rs->logits;
  if (rs->key_cache) delete rs->key_cache;
  if (rs->value_cache) delete rs->value_cache;
  if (rs->mask) delete rs->mask;

  // Others
  if (rs->cos_tensor) delete rs->cos_tensor;
  if (rs->sin_tensor) delete rs->sin_tensor;
}

void our_free(OurTransformerWeights *weights, OurRunState *rs, hipStream_t *total_streams, hipEvent_t *total_events) {
  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    fprintf(stderr, "Freeing weights and run state of id %d\n", i);

    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipStreamDestroy(total_streams[i]));
    CHECK_HIP(hipEventDestroy(total_events[i]));

    our_free_each(&weights[i], &rs[i]);

    fprintf(stderr, "Finish freeing weights and run state of id %d, moving on to actual pointer\n", i);
    fprintf(stderr, "Finish everything id %d\n", i);
  }
}
