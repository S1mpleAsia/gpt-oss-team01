#pragma once

#include "layer.hpp"
#include "layer_hip.hpp"
#include "utils.hpp"
#include "config.hpp"

float *forward_gpu_20b(Config *p, OurTransformerWeights *weights, OurRunState *rs, int token, int pos);
float *forward_cpu_20b(Config *p, OurTransformerWeights *weights, OurRunState *rs, int token, int pos);

