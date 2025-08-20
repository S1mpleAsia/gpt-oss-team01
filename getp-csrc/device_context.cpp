#include "device_context.hpp"

void device_init_context(DeviceContext *ctx, TransformerWeights *cpu_weights, Config *config,
                         int device_id) {
  ctx->device_id = device_id;

  CHECK_HIP(hipSetDevice(device_id));
  CHECK_HIP(hipStreamCreate(&ctx->stream));

  device_alloc_run_state(&ctx->d_state, config);
  device_copy_model_weight(&ctx->d_weights, cpu_weights, config);

  // RoPE pre-compute
  size_t rope_table_size = (size_t)config->seq_len * (config->head_dim / 2) * sizeof(float);
  CHECK_HIP(hipMalloc(&ctx->d_rope_cos, rope_table_size));
  CHECK_HIP(hipMalloc(&ctx->d_rope_sin, rope_table_size));

  BuildRopeTableGPU(config->seq_len, config->head_dim, config->rope_theta,
                    config->rope_scaling_factor, config->initial_context_length, 32.0f, 1.0f,
                    ctx->d_rope_cos, ctx->d_rope_sin, ctx->stream);

  CHECK_HIP(hipStreamSynchronize(ctx->stream));
}

void device_free_context(DeviceContext *ctx) {
  CHECK_HIP(hipSetDevice(ctx->device_id));

  free_run_state(&ctx->d_state);
  free_model_weight(&ctx->d_weights);

  CHECK_HIP(hipFree(ctx->d_rope_cos));
  CHECK_HIP(hipFree(ctx->d_rope_sin));
  CHECK_HIP(hipStreamDestroy(ctx->stream));
}