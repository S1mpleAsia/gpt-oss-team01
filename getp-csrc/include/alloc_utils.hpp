#pragma once

#include "config.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h> // Header for bfloat16 types and conversion functions

#define BUFFER_MLP1 8
#define BATCH_MLP1 2
#define BUFFER_MLP2 16
#define BATCH_MLP2 4

void alloc_w_mlp1_new(
    Tensor* &w_mlp1, float *w_mlp1_ptr, Config *p, int device_id
);
void alloc_w_mlp1(
    Tensor* &w_mlp1, float *w_mlp1_ptr, Config *p, int device_id
);
void alloc_w_mlp2_new(
    Tensor* &w_mlp2, float *w_mlp2_ptr, Config *p, int device_id
);
void alloc_w_mlp2(
    Tensor* &w_mlp2, float *w_mlp2_ptr, Config *p, int device_id
);