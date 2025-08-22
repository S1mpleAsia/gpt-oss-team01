#pragma once

#include "layer.hpp"

void our_init(Transformer *transformer);
void our_free();
float *our_forward(Transformer *transformer, int token, int pos);
