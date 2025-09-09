#pragma once

#include "layer.hpp"
#include "config.hpp"
#include "alloc_utils.hpp"

void our_init(Transformer *transformer, OurTransformerWeights *weights, OurRunState *rs, hipStream_t *total_streams, hipTotalEvents_t *total_events);
void our_free(OurTransformerWeights *weights, OurRunState *rs, hipStream_t *total_streams, hipTotalEvents_t *total_events);
