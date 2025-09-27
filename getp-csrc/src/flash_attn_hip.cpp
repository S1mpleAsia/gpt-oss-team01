#include "../include/flash_attn_hip.hpp"

#include <hip/hip_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>

#ifdef KV16

#ifndef CHECK_HIP
#define CHECK_HIP(x) do { auto _e = (x); if (_e != hipSuccess) { \
  fprintf(stderr,"HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); abort(); } } while(0)
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef FA_THREADS
#define FA_THREADS 128
#endif
#ifndef FA_TILE
#define FA_TILE 256 
#endif
#ifndef FLASH_ATTN_TILE
#define FLASH_ATTN_TILE 256
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 8
#endif
#ifndef ELEMS_PER_LOAD
#define ELEMS_PER_LOAD 8
#endif
#ifndef FA_MIN_BLOCKS_PER_CU
#define FA_MIN_BLOCKS_PER_CU 2
#endif
#ifndef FA_VGPR_RELAX
#define FA_VGPR_RELAX 0
#endif

#if FA_VGPR_RELAX
  #define FA_WAVES_EU __attribute__((amdgpu_waves_per_eu(1,1)))
#else
  #define FA_WAVES_EU
#endif

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t u) {
  bf16 h; *reinterpret_cast<uint16_t*>(&h) = u; return float(h);
}
__device__ __forceinline__ uint32_t ld_u32(const uint32_t* p) { return *p; }

__device__ __forceinline__ float warp_reduce_max64(float v) {
  #pragma unroll
  for (int off = 32; off > 0; off >>= 1) v = fmaxf(v, __shfl_down(v, off));
  return v;
}
__device__ __forceinline__ float warp_reduce_sum64(float v) {
  #pragma unroll
  for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off);
  return v;
}
__device__ __forceinline__ float shfl_xor_width(float v, int mask, int width){
  return __shfl_xor(v, mask, width);
}

template <int D, int G, int LOAD_ELEMS>
__global__ FA_WAVES_EU __launch_bounds__(FA_THREADS, FA_MIN_BLOCKS_PER_CU)
void flash_fused_bf16_kernel_tiled(
  const float *__restrict__ q,
  const bf16  *__restrict__ K_cache,
  const bf16  *__restrict__ V_cache,
  const float *__restrict__ attn_sinks,
  float *__restrict__ tb,
  int n_q, int kv_mul, int kv_dim, int n_layers, int seq_len,
  int t_begin, int t_end, int batch_size,
  int head_stride_out
){
  static_assert(D == 64, "");
  static_assert(D % G == 0, "");
  static_assert((D / G) % LOAD_ELEMS == 0, "");
  static_assert(LOAD_ELEMS == 8, "");

  const int h = blockIdx.x;
  const int b = blockIdx.y;
  if (b >= batch_size || h >= n_q) return;

  const int kv_h = h / kv_mul;
  const int T_full = t_end - t_begin;
  if (T_full <= 0) {
    for (int i = threadIdx.x; i < D; i += blockDim.x)
      tb[b * head_stride_out + h * D + i] = 0.f;
    return;
  }

  const size_t qtb_stride      = (size_t)n_q * D;
  const size_t kv_batch_stride = (size_t)n_layers * seq_len * kv_dim;
  const size_t kv_head_off     = (size_t)kv_h * D;

  const float *__restrict__ q_head = q + (size_t)b * qtb_stride + (size_t)h * D;
  const bf16  *__restrict__ Kb     = K_cache + (size_t)b * kv_batch_stride;
  const bf16  *__restrict__ Vb     = V_cache + (size_t)b * kv_batch_stride;
  float *__restrict__ out_head = tb + (size_t)b * (size_t)head_stride_out + (size_t)h * (size_t)D;
  const float sink_val = attn_sinks ? attn_sinks[h] : -INFINITY;

  extern __shared__ float shared_mem[];
  float *S_smem = shared_mem;
  
  __shared__ float warp_buf_max[FA_THREADS / WARP_SIZE];
  __shared__ float warp_buf_sum[FA_THREADS / WARP_SIZE];

  const int lane   = threadIdx.x & (WARP_SIZE - 1);
  const int wid    = threadIdx.x >> 6;
  const int glane  = lane % G;
  const int gid    = lane / G;
  const int groups_per_wave  = WARP_SIZE / G;
  const int waves_per_block  = blockDim.x / WARP_SIZE;
  const int groups_per_block = waves_per_block * groups_per_wave;

  const int ELEMS_PER_GROUP = D / G;
  const int groupIters      = ELEMS_PER_GROUP / LOAD_ELEMS;
  float RQ[ELEMS_PER_GROUP];
  const float inv_sqrt_D = rsqrtf((float)D);
  #pragma unroll
  for (int it = 0; it < groupIters; ++it) {
    const int d0 = (it * G + glane) * LOAD_ELEMS;
    #pragma unroll
    for (int j = 0; j < LOAD_ELEMS; ++j)
      RQ[it*LOAD_ELEMS + j] = q_head[d0 + j] * inv_sqrt_D;
  }
  
  float m_i = -INFINITY;
  float l_i = 0.f;
  float acc[ELEMS_PER_GROUP];
  #pragma unroll
  for (int i=0; i<ELEMS_PER_GROUP; ++i) acc[i] = 0.f;

  for (int t0 = 0; t0 < T_full; t0 += FA_TILE) {
      const int T_tile = min(FA_TILE, T_full - t0);
      float local_max = -INFINITY;

      for (int base = wid * groups_per_wave; base < T_tile; base += groups_per_block) {
        const int t = base + gid;
        float tmp = 0.f;

        if (t < T_tile) {
          const int tok = t_begin + t0 + t;
          #pragma unroll
          for (int it = 0; it < groupIters; ++it) {
            const int d0 = (it * G + glane) * LOAD_ELEMS;
            const size_t base_k = (size_t)tok * (size_t)kv_dim + kv_head_off + d0;
            const uint32_t w0 = ld_u32(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint16_t*>(Kb) + base_k + 0));
            const uint32_t w1 = ld_u32(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint16_t*>(Kb) + base_k + 2));
            const uint32_t w2 = ld_u32(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint16_t*>(Kb) + base_k + 4));
            const uint32_t w3 = ld_u32(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint16_t*>(Kb) + base_k + 6));
            tmp = fmaf(RQ[it*LOAD_ELEMS + 0], bf16_bits_to_f32((uint16_t)(w0 & 0xFFFF)), tmp);
            tmp = fmaf(RQ[it*LOAD_ELEMS + 1], bf16_bits_to_f32((uint16_t)(w0 >> 16)),  tmp);
            tmp = fmaf(RQ[it*LOAD_ELEMS + 2], bf16_bits_to_f32((uint16_t)(w1 & 0xFFFF)), tmp);
            tmp = fmaf(RQ[it*LOAD_ELEMS + 3], bf16_bits_to_f32((uint16_t)(w1 >> 16)),  tmp);
            tmp = fmaf(RQ[it*LOAD_ELEMS + 4], bf16_bits_to_f32((uint16_t)(w2 & 0xFFFF)), tmp);
            tmp = fmaf(RQ[it*LOAD_ELEMS + 5], bf16_bits_to_f32((uint16_t)(w2 >> 16)),  tmp);
            tmp = fmaf(RQ[it*LOAD_ELEMS + 6], bf16_bits_to_f32((uint16_t)(w3 & 0xFFFF)), tmp);
            tmp = fmaf(RQ[it*LOAD_ELEMS + 7], bf16_bits_to_f32((uint16_t)(w3 >> 16)),  tmp);
          }
        }
        #pragma unroll
        for (int offs = G >> 1; offs > 0; offs >>= 1) tmp += shfl_xor_width(tmp, offs, G);

        if (glane == 0 && t < T_tile) {
          S_smem[t] = tmp;
          local_max = fmaxf(local_max, tmp);
        }
      }
      __syncthreads();

      float v = warp_reduce_max64(local_max);
      if (lane == 0) warp_buf_max[wid] = v;
      __syncthreads();
      float m_tile = -INFINITY;
      if (wid == 0) {
        float r = (lane < waves_per_block) ? warp_buf_max[lane] : -INFINITY;
        m_tile = warp_reduce_max64(r);
        if (lane == 0) warp_buf_max[0] = m_tile;
      }
      __syncthreads();
      m_tile = warp_buf_max[0];
      
      const float m_new = fmaxf(m_i, m_tile);
      const float alpha = __expf(m_i - m_new);
      m_i = m_new;
      
      float local_sum = 0.f;
      for (int t = threadIdx.x; t < T_tile; t += blockDim.x) {
        const float p = __expf(S_smem[t] - m_i);
        S_smem[t] = p;
        local_sum += p;
      }

      v = warp_reduce_sum64(local_sum);
      if (lane == 0) warp_buf_sum[wid] = v;
      __syncthreads();

      float l_tile = 0.f;
      if (wid == 0) {
        float r = (lane < waves_per_block) ? warp_buf_sum[lane] : 0.f;
        l_tile = warp_reduce_sum64(r);
        if (lane == 0) warp_buf_sum[0] = l_tile;
      }
      __syncthreads();
      l_tile = warp_buf_sum[0];

      l_i = l_i * alpha + l_tile;
      
      #pragma unroll
      for(int i=0; i<ELEMS_PER_GROUP; ++i) acc[i] *= alpha;
      __syncthreads();

      for (int base = wid * groups_per_wave; base < T_tile; base += groups_per_block) {
          const int t = base + gid;
          if (t >= T_tile) continue;
          const int tok = t_begin + t0 + t;
          const float wP = S_smem[t];
          #pragma unroll
          for (int it = 0; it < groupIters; ++it) {
              const int d0  = (it * G + glane) * LOAD_ELEMS;
              const size_t off = (size_t)tok * (size_t)kv_dim + kv_head_off + d0;
              const uint32_t v0 = ld_u32(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint16_t*>(Vb) + off + 0));
              const uint32_t v1 = ld_u32(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint16_t*>(Vb) + off + 2));
              const uint32_t v2 = ld_u32(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint16_t*>(Vb) + off + 4));
              const uint32_t v3 = ld_u32(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint16_t*>(Vb) + off + 6));
              acc[it*LOAD_ELEMS + 0] = fmaf(wP, bf16_bits_to_f32((uint16_t)(v0 & 0xFFFF)), acc[it*LOAD_ELEMS + 0]);
              acc[it*LOAD_ELEMS + 1] = fmaf(wP, bf16_bits_to_f32((uint16_t)(v0 >> 16)),  acc[it*LOAD_ELEMS + 1]);
              acc[it*LOAD_ELEMS + 2] = fmaf(wP, bf16_bits_to_f32((uint16_t)(v1 & 0xFFFF)), acc[it*LOAD_ELEMS + 2]);
              acc[it*LOAD_ELEMS + 3] = fmaf(wP, bf16_bits_to_f32((uint16_t)(v1 >> 16)),  acc[it*LOAD_ELEMS + 3]);
              acc[it*LOAD_ELEMS + 4] = fmaf(wP, bf16_bits_to_f32((uint16_t)(v2 & 0xFFFF)), acc[it*LOAD_ELEMS + 4]);
              acc[it*LOAD_ELEMS + 5] = fmaf(wP, bf16_bits_to_f32((uint16_t)(v2 >> 16)),  acc[it*LOAD_ELEMS + 5]);
              acc[it*LOAD_ELEMS + 6] = fmaf(wP, bf16_bits_to_f32((uint16_t)(v3 & 0xFFFF)), acc[it*LOAD_ELEMS + 6]);
              acc[it*LOAD_ELEMS + 7] = fmaf(wP, bf16_bits_to_f32((uint16_t)(v3 >> 16)),  acc[it*LOAD_ELEMS + 7]);
          }
      }
      __syncthreads();
  }

  const float m_final = fmaxf(m_i, sink_val);
  const float alpha_f = __expf(m_i - m_final);
  const float sink_exp = isfinite(sink_val) ? __expf(sink_val - m_final) : 0.f;
  const float l_final = l_i * alpha_f + sink_exp;
  const float inv_l_final = 1.f / l_final;
  
  #pragma unroll
  for(int i=0; i<ELEMS_PER_GROUP; ++i) acc[i] *= alpha_f * inv_l_final;
  
  float *s_out  = shared_mem;
  for (int i = threadIdx.x; i < D; i += blockDim.x) s_out[i] = 0.f;
  __syncthreads();
  
  #pragma unroll
  for (int it = 0; it < groupIters; ++it) {
    #pragma unroll
    for (int j = 0; j < LOAD_ELEMS; ++j) {
      const int d_idx = (it * G + glane) * LOAD_ELEMS + j;
      atomicAdd(&s_out[d_idx], acc[it*LOAD_ELEMS + j]);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    out_head[i] = s_out[i];
  }
}

template <int TILE_TOKENS, int THREADS, int D, int G, int LOAD_ELEMS>
__global__ FA_WAVES_EU __launch_bounds__(THREADS)
void fd_partial_kernel_bf16_grouped(
  const float *__restrict__ q,
  const bf16  *__restrict__ K_cache,
  const bf16  *__restrict__ V_cache,
  float *__restrict__ partial_max,
  float *__restrict__ partial_sum,
  float *__restrict__ partial_num,
  int n_q, int kv_mul, int kv_dim, int n_layers, int seq_len,
  int t_begin, int t_end, int batch_size, int chunk)
{
  static_assert(D == 64, "");
  static_assert(D % G == 0, "");
  static_assert((D / G) % LOAD_ELEMS == 0, "");
  static_assert(LOAD_ELEMS == 8, "");

  const int bx = blockIdx.x, b = blockIdx.y;
  if (b >= batch_size) return;

  const int cidx = bx / n_q, h = bx % n_q;
  if (cidx >= chunk || h >= n_q) return;

  const int kv_h = h / kv_mul;
  const int start = t_begin + cidx * TILE_TOKENS;
  const int stop  = min(t_end, start + TILE_TOKENS);
  const int T     = max(0, stop - start);

  const size_t qtb_stride      = (size_t)n_q * D;
  const size_t kv_batch_stride = (size_t)n_layers * seq_len * kv_dim;
  const size_t kv_head_off     = (size_t)kv_h * D;

  const float *__restrict__ q_head = q + (size_t)b * qtb_stride + (size_t)h * D;
  const bf16  *__restrict__ Kb     = K_cache + (size_t)b * kv_batch_stride;
  const bf16  *__restrict__ Vb     = V_cache + (size_t)b * kv_batch_stride;

  const size_t bhc         = ((size_t)b * n_q + h) * (size_t)chunk;
  float *__restrict__ pM = partial_max + bhc;
  float *__restrict__ pS = partial_sum + bhc;
  float *__restrict__ pN = partial_num + bhc * (size_t)D;

  extern __shared__ float s_smem[];
  float *s_q  = s_smem;
  float *s_sc = s_q + D;

  __shared__ float warp_buf_max[THREADS / WARP_SIZE];
  __shared__ float warp_buf_sum[THREADS / WARP_SIZE];

  const int lane   = threadIdx.x & (WARP_SIZE - 1);
  const int wid    = threadIdx.x >> 6;
  const int glane  = lane % G;
  const int gid    = lane / G;
  const int groups_per_wave  = WARP_SIZE / G;
  const int waves_per_block  = blockDim.x / WARP_SIZE;
  const int groups_per_block = waves_per_block * groups_per_wave;

  const float inv_sqrt_D = rsqrtf((float)D);
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    s_q[i] = q_head[i] * inv_sqrt_D;
  __syncthreads();

  const int numElemPerGroup = D / G;
  const int groupIters      = numElemPerGroup / LOAD_ELEMS;
  float local_max = -INFINITY;

  for (int base = wid * groups_per_wave; base < T; base += groups_per_block) {
    const int t = base + gid;
    float tmp = 0.f;
    if (t < T) {
      #pragma unroll
      for (int it = 0; it < groupIters; ++it) {
        const int d0 = (it * G + glane) * LOAD_ELEMS;
        const size_t base_k = (size_t)(start + t) * (size_t)kv_dim + kv_head_off + d0;

        const uint32_t w0 = ld_u32(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint16_t*>(Kb) + base_k +  0));
        const uint32_t w1 = ld_u32(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint16_t*>(Kb) + base_k +  2));
        const uint32_t w2 = ld_u32(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint16_t*>(Kb) + base_k +  4));
        const uint32_t w3 = ld_u32(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint16_t*>(Kb) + base_k +  6));

        tmp = fmaf(s_q[d0 + 0], bf16_bits_to_f32((uint16_t)(w0 & 0xFFFF)), tmp);
        tmp = fmaf(s_q[d0 + 1], bf16_bits_to_f32((uint16_t)(w0 >> 16)),  tmp);
        tmp = fmaf(s_q[d0 + 2], bf16_bits_to_f32((uint16_t)(w1 & 0xFFFF)), tmp);
        tmp = fmaf(s_q[d0 + 3], bf16_bits_to_f32((uint16_t)(w1 >> 16)),  tmp);
        tmp = fmaf(s_q[d0 + 4], bf16_bits_to_f32((uint16_t)(w2 & 0xFFFF)), tmp);
        tmp = fmaf(s_q[d0 + 5], bf16_bits_to_f32((uint16_t)(w2 >> 16)),  tmp);
        tmp = fmaf(s_q[d0 + 6], bf16_bits_to_f32((uint16_t)(w3 & 0xFFFF)), tmp);
        tmp = fmaf(s_q[d0 + 7], bf16_bits_to_f32((uint16_t)(w3 >> 16)),  tmp);
      }
    }
    #pragma unroll
    for (int offs = G >> 1; offs > 0; offs >>= 1)
      tmp += shfl_xor_width(tmp, offs, G);

    if (glane == 0 && t < T) {
      s_sc[t] = tmp;
      local_max = fmaxf(local_max, tmp);
    }
  }
  __syncthreads();

  float v = warp_reduce_max64(local_max);
  if ((lane & (WARP_SIZE - 1)) == 0) warp_buf_max[wid] = v;
  __syncthreads();
  float m_c = -INFINITY;
  if (wid == 0) {
    float r = (lane < waves_per_block) ? warp_buf_max[lane] : -INFINITY;
    r = warp_reduce_max64(r);
    if (lane == 0) warp_buf_max[0] = r;
  }
  __syncthreads();
  m_c = warp_buf_max[0];

  float local_sum = 0.f;
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    const float w = __expf(s_sc[t] - m_c);
    s_sc[t] = w;
    local_sum += w;
  }

  v = warp_reduce_sum64(local_sum);
  if ((lane & (WARP_SIZE - 1)) == 0) warp_buf_sum[wid] = v;
  __syncthreads();
  float l_c = 0.f;
  if (wid == 0) {
    float r = (lane < waves_per_block) ? warp_buf_sum[lane] : 0.f;
    r = warp_reduce_sum64(r);
    if (lane == 0) warp_buf_sum[0] = r;
  }
  __syncthreads();
  l_c = warp_buf_sum[0];

  for (int i_out = (threadIdx.x << 1); i_out < D; i_out += (blockDim.x << 1)) {
    float acc0 = 0.f, acc1 = 0.f;
    #pragma unroll 8
    for (int t = 0; t < T; ++t) {
      const float w = s_sc[t];
      const size_t base = (size_t)(start + t) * (size_t)kv_dim + kv_head_off + (size_t)i_out;
      const uint32_t vp = ld_u32(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint16_t*>(Vb) + base));
      const uint16_t v0 = (uint16_t)(vp & 0xFFFF);
      const uint16_t v1 = (uint16_t)(vp >> 16);
      acc0 = fmaf(w, bf16_bits_to_f32(v0), acc0);
      if (i_out + 1 < D) acc1 = fmaf(w, bf16_bits_to_f32(v1), acc1);
    }
    float *dst = pN + (size_t)cidx * (size_t)D + (size_t)i_out;
    dst[0] = acc0;
    if (i_out + 1 < D) dst[1] = acc1;
  }

  if (threadIdx.x == 0) { pM[cidx] = m_c; pS[cidx] = l_c; }
}

template <int D>
__global__ void fd_reduce_kernel_stable_v2(
  const float *__restrict__ partial_max,
  const float *__restrict__ partial_sum,
  const float *__restrict__ partial_num,
  const float *__restrict__ attn_sinks,
  float *__restrict__ tb,
  int n_q, int chunk, int batch_size)
{
  const int h = blockIdx.x, b = blockIdx.y;
  if (b >= batch_size || h >= n_q) return;

  const size_t qtb_stride = (size_t)n_q * (size_t)D;
  float *__restrict__ out = tb + (size_t)b * qtb_stride + (size_t)h * D;

  const size_t baseBH = ((size_t)b * n_q + h);
  const float *mBH = partial_max + baseBH * (size_t)chunk;
  const float *sBH = partial_sum + baseBH * (size_t)chunk;
  const float *nBH = partial_num + baseBH * (size_t)chunk * (size_t)D;

  double m_star = -DBL_MAX;
  for (int c = 0; c < chunk; ++c) m_star = fmax(m_star, (double)mBH[c]);
  m_star = fmax(m_star, (double)attn_sinks[h]);

  const int i = threadIdx.x;
  double denom_chunks = 0.0, num_i = 0.0;
  for (int c = 0; c < chunk; ++c) {
    const double mc = (double)mBH[c];
    const double sc = isfinite(mc) ? exp(mc - m_star) : 0.0;
    denom_chunks += (double)sBH[c] * sc;
    if (i < D) num_i += (double)nBH[(size_t)c * D + i] * sc;
  }
  const double denom = denom_chunks + exp((double)attn_sinks[h] - m_star);
  if (i < D) out[i] = (float)(num_i / denom);
}


void single_query_attn_flash_batched(Tensor *q, Tensor *K_cache, Tensor *V_cache, Tensor *mask,
                                     Tensor *attn_sinks, Tensor *tb, Tensor *g_fa_pmax,
                                     Tensor *g_fa_psum, Tensor *g_fa_pnum,
                                     int cur_batch_size, int head_dim, int n_q,
                                     int kv_mul, int kv_dim, int seq_len, int sliding_window,
                                     int pos, long long layer_offset, bool q_to_device,
                                     bool k_cache_to_device, bool v_cache_to_device,
                                     bool mask_to_device, bool tb_from_device, hipStream_t stream)
{
  if (head_dim != 64) {
      fprintf(stderr, "This kernel only supports head_dim 64, got %d.\n", head_dim);
      abort();
  }

  if (q_to_device)       q->to_device(stream);
  if (k_cache_to_device) K_cache->to_device(stream);
  if (v_cache_to_device) V_cache->to_device(stream);

  const size_t n_layers   = (size_t)K_cache->shape[1];
  const size_t layer_span = (size_t)seq_len * kv_dim;
  const bf16 *K_ptr = ((const bf16 *)K_cache->d_buf) + layer_offset * layer_span;
  const bf16 *V_ptr = ((const bf16 *)V_cache->d_buf) + layer_offset * layer_span;
  const float *S_ptr = (const float *)attn_sinks->d_buf + layer_offset * n_q;

  const int attn_len = pos + 1;
  const bool use_window = (sliding_window > 0) && ((layer_offset & 1ll) == 0);
  const int L_eff = use_window ? min(attn_len, sliding_window) : attn_len;
  const int t_end = attn_len;
  const int t_begin = t_end - L_eff;

  if (L_eff > FA_TILE) {
    dim3 grid(n_q, cur_batch_size);
    dim3 block(FA_THREADS);
    const size_t shmem_bytes = (size_t)FA_TILE * sizeof(float);
    const int head_stride_out = n_q * head_dim;

    hipLaunchKernelGGL((flash_fused_bf16_kernel_tiled<64, GROUP_SIZE, ELEMS_PER_LOAD>),
        grid, block, shmem_bytes, stream,
        (const float*)q->d_buf, K_ptr, V_ptr, S_ptr,
        (float*)tb->d_buf, n_q, kv_mul, kv_dim, (int)n_layers, seq_len,
        t_begin, t_end, cur_batch_size, head_stride_out);
  } else {
    const int chunk = (L_eff + FA_TILE - 1) / FA_TILE;
    const int g_fa_C_max = (seq_len + FLASH_ATTN_TILE - 1) / FLASH_ATTN_TILE;
    if (g_fa_C_max < chunk) {
        fprintf(stderr, "C %d > g_fa_C_max %d\n", chunk, g_fa_C_max);
        abort();
    }

    float *d_pmax = (float *)g_fa_pmax->d_buf;
    float *d_psum = (float *)g_fa_psum->d_buf;
    float *d_pnum = (float *)g_fa_pnum->d_buf;

    dim3 grid_p(n_q * chunk, cur_batch_size);
    dim3 block_p(FA_THREADS);
    const size_t shmem_p = (size_t)head_dim * sizeof(float) + (size_t)FA_TILE * sizeof(float);

    fd_partial_kernel_bf16_grouped<FA_TILE, FA_THREADS, 64, GROUP_SIZE, ELEMS_PER_LOAD>
    <<<grid_p, block_p, shmem_p, stream>>>(
        (const float*)q->d_buf, K_ptr, V_ptr,
        d_pmax, d_psum, d_pnum,
        n_q, kv_mul, kv_dim, (int)n_layers, seq_len,
        t_begin, t_end, cur_batch_size, chunk);
    
    dim3 grid_r(n_q, cur_batch_size);
    dim3 block_r(head_dim);
    fd_reduce_kernel_stable_v2<64>
    <<<grid_r, block_r, 0, stream>>>(
        d_pmax, d_psum, d_pnum, S_ptr, (float*)tb->d_buf,
        n_q, chunk, cur_batch_size);
  }
  
  CHECK_HIP(hipGetLastError());

  if (tb_from_device) {
    tb->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

#else

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

__device__ __forceinline__ double block_reduce_max(double v, double *s_red) {
  const int tid = threadIdx.x;
  s_red[tid] = v;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s)
      s_red[tid] = fmax(s_red[tid], s_red[tid + s]);
    __syncthreads();
  }
  return s_red[0];
}

__device__ __forceinline__ double block_reduce_sum(double v, double *s_red) {
  const int tid = threadIdx.x;
  s_red[tid] = v;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s)
      s_red[tid] += s_red[tid + s];
    __syncthreads();
  }
  return s_red[0];
}

template <int TILE_TOKENS = 128, int THREADS = 128>
__global__ void flash_attn_decode_kernel(
  const float *__restrict__ q,
  const float *__restrict__ K_cache,
  const float *__restrict__ V_cache,
  const float *__restrict__ attn_sinks,
  float *__restrict__ tb,
  int head_dim, int n_q, int kv_mul, int kv_dim, int total_seq_len, int pos, int sliding_window,
  int batch_size, int total_layers, int use_window) {
  const int head_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  if (head_idx >= n_q || batch_idx >= batch_size)
    return;
  const int kv_head_idx = head_idx / kv_mul;
  const size_t qtb_stride = (size_t)n_q * head_dim;
  const size_t kv_batch_stride = (size_t)total_layers * total_seq_len * kv_dim;
  const float *__restrict__ q_head = q + (size_t)batch_idx * qtb_stride + (size_t)head_idx * head_dim;
  float *__restrict__ tb_head = tb + (size_t)batch_idx * qtb_stride + (size_t)head_idx * head_dim;
  const float *__restrict__ K_base = K_cache + (size_t)batch_idx * kv_batch_stride;
  const float *__restrict__ V_base = V_cache + (size_t)batch_idx * kv_batch_stride;
  const float *__restrict__ attn_sinks_head = attn_sinks + (size_t)head_idx;
  const float scale = rsqrtf((float)head_dim);
  const int attn_len = pos + 1;
  int t_begin = 0;
  if (use_window)
    t_begin = max(0, attn_len - sliding_window);
  const int t_end = attn_len;
  extern __shared__ unsigned char sdata[];
  float *s_q = (float *)sdata;
  float *s_scores = (float *)(s_q + head_dim);
  double *s_red = (double *)(s_scores + TILE_TOKENS);
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    s_q[i] = q_head[i];
  __syncthreads();
  double m = -DBL_MAX;
  double l = 0.0;
  double o_i = 0.0;
  const size_t kv_head_off = (size_t)kv_head_idx * head_dim;
  for (int t0 = t_begin; t0 < t_end; t0 += TILE_TOKENS) {
    const int tile_len = min(TILE_TOKENS, t_end - t0);
    double local_max = -DBL_MAX;
    for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
      const float *__restrict__ k_vec = K_base + (size_t)(t0 + t) * kv_dim + kv_head_off;
      float s = 0.f;
#pragma unroll 64
      for (int d = 0; d < 64; ++d) {
        s += s_q[d] * k_vec[d];
      }
      const float score_f = s * scale;
      s_scores[t] = score_f;
      local_max = fmax(local_max, (double)score_f);
    }
    if (tile_len == 0)
      local_max = -DBL_MAX;
    double tile_max = block_reduce_max(local_max, s_red);
    const double m_new = fmax(m, tile_max);
    const double alpha = exp(m - m_new);
    l *= alpha;
    if (threadIdx.x < (unsigned)head_dim)
      o_i *= alpha;
    __syncthreads();
    m = m_new;
    double local_sum_w = 0.0;
    for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
      const double w = exp((double)s_scores[t] - m);
      s_scores[t] = (float)w;
      local_sum_w += w;
    }
    const double sum_w = block_reduce_sum(local_sum_w, s_red);
    if (threadIdx.x < (unsigned)head_dim) {
      const int i_out = threadIdx.x;
      double acc = o_i;
      for (int t = 0; t < tile_len; ++t) {
        const double w = (double)s_scores[t];
        const float *__restrict__ v_vec = V_base + (size_t)(t0 + t) * kv_dim + kv_head_off;
        acc += w * (double)v_vec[i_out];
      }
      o_i = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0)
      s_red[0] = l + sum_w;
    __syncthreads();
    l = s_red[0];
  }
  const double sink = (double)(*attn_sinks_head);
  const double m_final = fmax(m, sink);
  const double alpha_f = exp(m - m_final);
  const double l_final = l * alpha_f + exp(sink - m_final);
  if (threadIdx.x < (unsigned)head_dim) {
    const double o_scaled = o_i * alpha_f;
    tb_head[threadIdx.x] = (float)(o_scaled / l_final);
  }
}

void single_query_attn_flash_batched(Tensor *q, Tensor *K_cache, Tensor *V_cache, Tensor *mask,
                                     Tensor *attn_sinks, Tensor *tb, Tensor *g_fa_pmax,
                                     Tensor *g_fa_psum, Tensor *g_fa_pnum, int cur_batch_size,
                                     int head_dim, int n_q, int kv_mul, int kv_dim, int seq_len,
                                     int sliding_window, int pos, long long layer_offset,
                                     bool q_to_device, bool k_cache_to_device,
                                     bool v_cache_to_device, bool mask_to_device,
                                     bool tb_from_device, hipStream_t stream) {
  if (q_to_device)
    q->to_device(stream);
  if (k_cache_to_device)
    K_cache->to_device(stream);
  if (v_cache_to_device)
    V_cache->to_device(stream);

  const int n_layers = (int)attn_sinks->shape[0];
  const float *K_ptr = (const float *)K_cache->d_buf + 1ll * layer_offset * seq_len * kv_dim;
  const float *V_ptr = (const float *)V_cache->d_buf + 1ll * layer_offset * seq_len * kv_dim;
  const float *S_ptr = (const float *)attn_sinks->d_buf + 1ll * layer_offset * n_q;
  const int use_window = (sliding_window > 0 && ((layer_offset & 1ll) == 0) && mask && mask->d_buf) ? 1 : 0;
  constexpr int TILE_TOKENS = 128;
  constexpr int THREADS = 128;
  dim3 grid(n_q, cur_batch_size), block(THREADS);
  size_t shmem = (size_t)head_dim * sizeof(float) + (size_t)TILE_TOKENS * sizeof(float) +
                 (size_t)THREADS * sizeof(double);
  flash_attn_decode_kernel<TILE_TOKENS, THREADS><<<grid, block, shmem, stream>>>(
    (const float *)q->d_buf, K_ptr, V_ptr, S_ptr, (float *)tb->d_buf, head_dim, n_q, kv_mul, kv_dim,
    seq_len, pos, sliding_window, cur_batch_size, n_layers, use_window);

  CHECK_HIP(hipGetLastError());
  if (tb_from_device) {
    tb->from_device(stream);
    CHECK_HIP(hipStreamSynchronize(stream));
  }
}

#endif

