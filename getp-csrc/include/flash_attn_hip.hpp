#pragma once

#include <hip/hip_runtime.h>
#include "config.hpp"
#include "tensor.hpp"
#include "utils.hpp"

// Flash-style single-query attention (batched), drop-in replacement.
// Same semantics as single_query_attn_batched().
void single_query_attn_flash_batched(Tensor *q,        // [B, n_q*hd] (float32)
                                     Tensor *K_cache,  // [B, L, S, kv_dim] (float32 or bf16)
                                     Tensor *V_cache,  // [B, L, S, kv_dim] (float32 or bf16)
                                     Tensor *mask,  // [B, S, S] or nullptr (float32; additive mask)
                                     Tensor *attn_sinks,  // [L, n_q]  (float32)
                                     Tensor *tb,          // [B, n_q*hd] (float32)
                                     int cur_batch_size, int head_dim, int num_query_heads,
                                     int kv_mul, int kv_dim, int seq_len,
                                     int sliding_window,  // 0 means disabled
                                     int pos, long long layer_offset, bool q_to_device,
                                     bool k_cache_to_device, bool v_cache_to_device,
                                     bool mask_to_device, bool tb_from_device,
                                     hipStream_t stream = 0);
