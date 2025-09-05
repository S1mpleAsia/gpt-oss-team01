#pragma once

#include "layer.hpp"
#include "layer_hip.hpp"
#include "layer_hip_batch.hpp"
#include "utils.hpp"
#include "config.hpp"
#include "flash_attn_hip.hpp"

float *forward_gpu_20b_batched(Config *p, OurTransformerWeights *weights, OurRunState *rs,
                               int *tokens, int pos, int batch_size);
float *forward_gpu_20b(Config *p, OurTransformerWeights *weights, OurRunState *rs, int token,
                       int pos);
float *forward_cpu_20b(Config *p, OurTransformerWeights *weights, OurRunState *rs, int token,
                       int pos);
