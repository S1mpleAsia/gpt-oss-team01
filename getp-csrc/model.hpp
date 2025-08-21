#pragma once

#include "layer.hpp"

void device_alloc_run_state(DeviceRunState *rs, Config *config);

void device_copy_model_weight(DeviceTransformerWeights *d_w, TransformerWeights *t, Config *config);

void free_device_run_state(DeviceRunState *rs);

void free_model_weight(DeviceTransformerWeights *d_weights);

float *hip_forward(DeviceTransformerWeights *w, DeviceRunState *rs, Config *config, int token,
                   int pos, hipStream_t stream, const float *d_rope_cos, const float *d_rope_sin);
