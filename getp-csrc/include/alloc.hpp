#pragma once

#include "layer.hpp"
#include "config.hpp"

void our_init(Transformer *transformer, OurTransformerWeights *weights, OurRunState *rs);
void our_free(OurTransformerWeights *weights, OurRunState *rs);
