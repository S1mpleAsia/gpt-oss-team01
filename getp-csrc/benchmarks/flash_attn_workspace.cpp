#include "config_types_fwd.hpp"
#include "flash_attn_iface.hpp"
#include "../include/tensor.hpp"

// Workspace hook: replace the body with your optimized implementation when ready.
void single_query_attn_flash_batched_opt(Tensor *q, Tensor *K_cache, Tensor *V_cache, Tensor *mask,
                                         Tensor *attn_sinks, Tensor *tb, Tensor *g_fa_pmax,
                                         Tensor *g_fa_psum, Tensor *g_fa_pnum, int cur_batch_size,
                                         int head_dim, int num_query_heads, int kv_mul, int kv_dim,
                                         int seq_len, int sliding_window, int pos,
                                         long long layer_offset, bool q_to_device,
                                         bool k_cache_to_device, bool v_cache_to_device,
                                         bool mask_to_device, bool tb_from_device,
                                         const size_t * /*layer_offsets_tokens*/,
                                         const int * /*layer_tokens*/,
                                         const uint8_t * /*layer_is_window*/,
                                         size_t /*kv_batch_stride_tokens*/, hipStream_t stream) {
  // Forward to the baseline kernel. Extra workspace parameters are ignored here.
  single_query_attn_flash_batched(q, K_cache, V_cache, mask, attn_sinks, tb, g_fa_pmax, g_fa_psum,
                                  g_fa_pnum, cur_batch_size, head_dim, num_query_heads, kv_mul,
                                  kv_dim, seq_len, sliding_window, pos, layer_offset, q_to_device,
                                  k_cache_to_device, v_cache_to_device, mask_to_device,
                                  tb_from_device, stream);
}
