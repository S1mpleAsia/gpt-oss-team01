#pragma once

#include "layer.hpp"
#include "layer_hip.hpp"
#include "layer_hip_batch.hpp"
#include "utils.hpp"
#include "config.hpp"
#include "flash_attn_hip.hpp"
#include <hip/hip_runtime.h>

float *forward_gpu_120b_batched(int *tokens, int pos, int batch_size, int flow_id, int tp_rank,
                                int pp_rank, pthread_barrier_t *tp_barrier);

float *forward_gpu_20b_batched(int *tokens, int pos, int batch_size, int flow_id);

float *forward_gpu_20b(Config *p, OurTransformerWeights *weights, OurRunState *rs, int token,
                       int pos);

float *forward_cpu_20b(Config *p, OurTransformerWeights *weights, OurRunState *rs, int token,
                       int pos);
