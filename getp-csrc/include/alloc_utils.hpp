#pragma once

#include "config.hpp"
#include "utils.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h> // Header for bfloat16 types and conversion functions

#define BUFFER_MLP1 2
#define BATCH_MLP1 2
#define BUFFER_MLP2 2
#define BATCH_MLP2 2

#define TOTAL_BASE_VALUES_MLP1 ((TOTAL_PIPELINES) * (BATCH_MLP1))
#define TOTAL_BASE_VALUES_MLP2 ((TOTAL_PIPELINES) * (BATCH_MLP2))

#define OFFSET_LAYER 3
#define OFFSET_MOE 32

void alloc_w_mlp1_final(
    OurTransformerWeights *weights_total,
    float* __restrict__ w_mlp1_ptr, Config *p
);
void alloc_w_mlp1(
    Tensor* &w_mlp1, float *w_mlp1_ptr, Config *p, int device_id
);
void alloc_w_mlp2_final(
    OurTransformerWeights *weights_total,
    float* __restrict__ w_mlp2_ptr, Config *p
);
void alloc_w_mlp2(
    Tensor* &w_mlp2, float *w_mlp2_ptr, Config *p, int device_id
);
void alloc_out(
    Tensor* &out, float *w_out, Config *p, int device_id
);
void alloc_out_new(
    Tensor* &out, float *w_out, Config *p, int device_id
);