#pragma once

#include "config.hpp"
#include "utils.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h> // Header for bfloat16 types and conversion functions

#define BUFFER_MLP1 8
#define BATCH_MLP1 320
#define BUFFER_MLP2 16
#define BATCH_MLP2 16

#define OFFSET_LAYER 3
#define OFFSET_MOE 32

void alloc_w_mlp1_transpose(
    Tensor* &w_mlp1, float *w_mlp1_ptr, Config *p, int device_id
);
void alloc_w_mlp1_final(
    OurTransformerWeights *weights_total, float *w_mlp1_ptr, Config *p,
    int start_layer, int start_moe
);
void alloc_w_mlp1_v2(
    Tensor* &w_mlp1, float *w_mlp1_ptr, Config *p, int device_id,
    int start_layer, int start_moe
);
void alloc_w_mlp1(
    Tensor* &w_mlp1, float *w_mlp1_ptr, Config *p, int device_id
);
void alloc_w_mlp2_transpose(
    Tensor* &w_mlp2, float *w_mlp2_ptr, Config *p, int device_id
);
void alloc_w_mlp2_new(
    Tensor* &w_mlp2, float *w_mlp2_ptr, Config *p, int device_id
);
void alloc_w_mlp2_v2(
    Tensor* &w_mlp2, float *w_mlp2_ptr, Config *p, int device_id
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