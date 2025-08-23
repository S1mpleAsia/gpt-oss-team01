#pragma once

#include "layer.hpp"
#include "utils.hpp"
#include "config.hpp"

float *our_forward(Config *p, OurTransformerWeights *weights, OurRunState *rs, int token, int pos);
