#pragma once
#include <hip/hip_runtime.h>
#include "../include/tensor.hpp"

// Baseline kernel (short signature)
void single_query_attn_flash_batched(Tensor *q, Tensor *K_cache, Tensor *V_cache, Tensor *mask,
                                     Tensor *attn_sinks, Tensor *tb, Tensor *g_fa_pmax,
                                     Tensor *g_fa_psum, Tensor *g_fa_pnum, int cur_batch_size,
                                     int head_dim, int n_q, int kv_mul, int kv_dim, int seq_len,
                                     int sliding_window, int pos, long long layer_offset,
                                     bool q_to_device, bool k_cache_to_device,
                                     bool v_cache_to_device, bool mask_to_device,
                                     bool tb_from_device, hipStream_t stream);

// Optional optimized kernel (extended signature). Mark weak so it can be absent.
extern "C" void single_query_attn_flash_batched_opt(
  Tensor *q, Tensor *K_cache, Tensor *V_cache, Tensor *mask, Tensor *attn_sinks, Tensor *tb,
  Tensor *g_fa_pmax, Tensor *g_fa_psum, Tensor *g_fa_pnum, int cur_batch_size, int head_dim,
  int num_query_heads, int kv_mul, int kv_dim, int seq_len, int sliding_window, int pos,
  long long layer_offset, bool q_to_device, bool k_cache_to_device, bool v_cache_to_device,
  bool mask_to_device, bool tb_from_device, const size_t *layer_offsets_tokens,
  const int *layer_tokens, const uint8_t *layer_is_window, size_t kv_batch_stride_tokens,
  hipStream_t stream) __attribute__((weak));
