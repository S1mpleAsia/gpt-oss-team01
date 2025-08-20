#pragma once
#include "model.hpp"

struct DeviceContext {
  int device_id;
  DeviceTransformerWeights d_weights;
  DeviceRunState d_state;
  float *d_rope_cos;
  float *d_rope_sin;
  hipStream_t stream;
};

void device_init_context(DeviceContext *ctx, TransformerWeights *cpu_weights, Config *config,
                         int device_id);

void device_free_context(DeviceContext *ctx);