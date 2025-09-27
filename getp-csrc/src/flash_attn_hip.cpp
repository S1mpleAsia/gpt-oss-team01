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
#define GROUP_SIZE 8      // sub-wave size (8 or 16 are good choices)
#endif
#ifndef ELEM_BYTES
#define ELEM_BYTES 2      // bf16/half = 2 bytes
#endif
#ifndef ELEMS_PER_LOAD
#define ELEMS_PER_LOAD 8 // 128-bit / 2B = 8 elems per vector load
#endif
#ifndef FA_MIN_BLOCKS_PER_CU
#define FA_MIN_BLOCKS_PER_CU 2
#endif
#ifndef FA_VGPR_RELAX
#define FA_VGPR_RELAX 0
#endif
#ifndef HEAD_DIM_128_GROUP
#define HEAD_DIM_128_GROUP 16
#endif

#if FA_VGPR_RELAX
  #define FA_WAVES_PER_EU __attribute__((amdgpu_waves_per_eu(1,1)))
#else
  #define FA_WAVES_PER_EU
#endif

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t u) {
  bf16 h; *reinterpret_cast<uint16_t*>(&h) = u; return float(h);
}
__device__ __forceinline__ uint32_t ld_u32(const uint32_t *p) { return *p; }

// wave64 reductions (optionally width-limited)
__device__ __forceinline__ float shfl_xor_width(float v, int mask, int width){
  // HIP shfl_xor supports a width parameter
  return __shfl_xor(v, mask, width);
}
__device__ __forceinline__ float warp_reduce_max64_float(float v) {
  #pragma unroll
  for (int off = 32; off > 0; off >>= 1) v = fmaxf(v, __shfl_down(v, off));
  return v;
}
__device__ __forceinline__ float warp_reduce_sum64_float(float v) {
  #pragma unroll
  for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off);
  return v;
}

// Each wave (64 lanes) is split into GROUP_SIZE sub-waves; each sub-wave computes one token dot.
// Each thread loads ELEMS_PER_LOAD contiguous bf16 from K via 128-bit, reduces within the group.
template <int TILE_TOKENS, int THREADS, int D, int G, int LOAD_ELEMS>
__global__ FA_WAVES_PER_EU __launch_bounds__(THREADS, FA_MIN_BLOCKS_PER_CU)
void fd_partial_kernel_bf16_grouped(
  const float *__restrict__ q,       // [B, n_q*D]
  const bf16  *__restrict__ K_cache, // [B, L, S, kv_dim]
  const bf16  *__restrict__ V_cache, // [B, L, S, kv_dim]
  float *__restrict__ partial_max,   // [B, n_q, C]
  float *__restrict__ partial_sum,   // [B, n_q, C]
  float *__restrict__ partial_num,   // [B, n_q, C, D]
  int n_q, int kv_mul, int kv_dim, int n_layers, int seq_len,
  int t_begin, int t_end, int batch_size, int chunk)
{
  static_assert(D % G == 0, "D must be divisible by GROUP_SIZE");
  static_assert((D / G) % LOAD_ELEMS == 0, "numElemPerGroup must be multiple of ELEMS_PER_LOAD");
  static_assert(LOAD_ELEMS == 8, "This kernel assumes 128-bit load => 8 bf16 elements");

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

  // Shared: Q(D) + scores/weights(TILE_TOKENS)
  extern __shared__ float s_smem[];
  float *s_q  = s_smem;
  float *s_sc = s_q + D;

  __shared__ float warp_buf_max[THREADS / WARP_SIZE];
  __shared__ float warp_buf_sum[THREADS / WARP_SIZE];

  // lane decomposition
  const int lane   = threadIdx.x & (WARP_SIZE - 1);   // 0..63
  const int wid    = threadIdx.x >> 6;                // warp id in block
  const int glane  = lane % G;                        // lane inside group
  const int gid    = lane / G;                        // group id inside wave
  const int groups_per_wave  = WARP_SIZE / G;
  const int waves_per_block  = blockDim.x / WARP_SIZE;
  const int groups_per_block = waves_per_block * groups_per_wave;

  // pre-scale Q
  const float inv_sqrt_D = rsqrtf((float)D);
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    s_q[i] = q_head[i] * inv_sqrt_D;
  __syncthreads();

  // --- Pass 1: compute s_sc[t] with grouped 128-bit loads of K ---
  const int numElemPerGroup = D / G;           // elements of D handled per group
  const int groupIters      = numElemPerGroup / LOAD_ELEMS; // typically 1

  float local_max = -INFINITY;

  // iterate tokens by "group scheduling": each group handles a distinct token index
  for (int base = wid * groups_per_wave; base < T; base += groups_per_block) {
    const int t = base + gid;
    float tmp = 0.f;

    if (t < T) {
      // Each iteration loads 128 bits (8 bf16) at offset aligned for this group's slice
      #pragma unroll
      for (int it = 0; it < groupIters; ++it) {
        const int d0 = (it * G + glane) * LOAD_ELEMS;  // start index into D for this thread
        const size_t base_k = (size_t)(start + t) * (size_t)kv_dim + kv_head_off + d0;

        // Read 128 bits = 4 x u32; each u32 packs 2 bf16
        const uint32_t w0 = ld_u32(reinterpret_cast<const uint32_t*>(
                                   reinterpret_cast<const uint16_t*>(Kb) + base_k +  0));
        const uint32_t w1 = ld_u32(reinterpret_cast<const uint32_t*>(
                                   reinterpret_cast<const uint16_t*>(Kb) + base_k +  2));
        const uint32_t w2 = ld_u32(reinterpret_cast<const uint32_t*>(
                                   reinterpret_cast<const uint16_t*>(Kb) + base_k +  4));
        const uint32_t w3 = ld_u32(reinterpret_cast<const uint32_t*>(
                                   reinterpret_cast<const uint16_t*>(Kb) + base_k +  6));

        // Unpack and FMA with pre-scaled q
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

    // reduce inside the GROUP_SIZE sub-wave
    #pragma unroll
    for (int offs = G >> 1; offs > 0; offs >>= 1)
      tmp += shfl_xor_width(tmp, offs, G);

    if (glane == 0 && t < T) {
      s_sc[t] = tmp;
      local_max = fmaxf(local_max, tmp);
    }
  }
  __syncthreads();

  // block-wide max via warps
  float v = warp_reduce_max64_float(local_max);
  if ((lane & (WARP_SIZE - 1)) == 0) warp_buf_max[wid] = v;
  __syncthreads();
  float m_c = -INFINITY;
  if (wid == 0) {
    float r = (lane < waves_per_block) ? warp_buf_max[lane] : -INFINITY;
    r = warp_reduce_max64_float(r);
    if (lane == 0) warp_buf_max[0] = r;
  }
  __syncthreads();
  m_c = warp_buf_max[0];

  // --- Pass 2: weights + denom ---
  float local_sum = 0.f;
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    const float w = __expf(s_sc[t] - m_c);
    s_sc[t] = w;
    local_sum += w;
  }

  v = warp_reduce_sum64_float(local_sum);
  if ((lane & (WARP_SIZE - 1)) == 0) warp_buf_sum[wid] = v;
  __syncthreads();
  float l_c = 0.f;
  if (wid == 0) {
    float r = (lane < waves_per_block) ? warp_buf_sum[lane] : 0.f;
    r = warp_reduce_sum64_float(r);
    if (lane == 0) warp_buf_sum[0] = r;
  }
  __syncthreads();
  l_c = warp_buf_sum[0];

  // --- Pass 3: numerator (stream V) ---
  for (int i_out = (threadIdx.x << 1); i_out < D; i_out += (blockDim.x << 1)) {
    float acc0 = 0.f, acc1 = 0.f;
    #pragma unroll 8
    for (int t = 0; t < T; ++t) {
      const float w = s_sc[t];
      const size_t base = (size_t)(start + t) * (size_t)kv_dim + kv_head_off + (size_t)i_out;
      const uint32_t vp =
        ld_u32(reinterpret_cast<const uint32_t*>(
          reinterpret_cast<const uint16_t*>(Vb) + base));
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
  const float *__restrict__ partial_max,  // [B, n_q, chunk]
  const float *__restrict__ partial_sum,  // [B, n_q, chunk]
  const float *__restrict__ partial_num,  // [B, n_q, chunk, D]
  const float *__restrict__ attn_sinks,   // [n_q]
  float *__restrict__ tb,                 // [B, n_q * D]
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
                                     Tensor *g_fa_psum, Tensor *g_fa_pnum, int cur_batch_size,
                                     int head_dim, int n_q, int kv_mul, int kv_dim, int seq_len,
                                     int sliding_window, int pos, long long layer_offset,
                                     bool q_to_device, bool k_cache_to_device,
                                     bool v_cache_to_device, bool mask_to_device,
                                     bool tb_from_device, hipStream_t stream) {
    GpuTimer timer("flash_attn");
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

    const int chunk = (L_eff + FA_TILE - 1) / FA_TILE;
    const int g_fa_C_max = (seq_len + FLASH_ATTN_TILE - 1) / FLASH_ATTN_TILE;
    if (g_fa_C_max < chunk) {
        fprintf(stderr, "[FlashDec opt] C=%d > g_fa_C_max=%d\n", chunk, g_fa_C_max);
        abort();
    }

    float *d_pmax = (float *)g_fa_pmax->d_buf;
    float *d_psum = (float *)g_fa_psum->d_buf;
    float *d_pnum = (float *)g_fa_pnum->d_buf;

    dim3 grid_p(n_q * chunk, cur_batch_size);
    dim3 block_p(FA_THREADS);
    // shared: s_q (D) + s_sc (FA_TILE)
    const size_t shmem_p = (size_t)head_dim * sizeof(float) + (size_t)FA_TILE * sizeof(float);

    fd_partial_kernel_bf16_grouped<FA_TILE, FA_THREADS, 64, GROUP_SIZE, ELEMS_PER_LOAD>
    <<<grid_p, block_p, shmem_p, stream>>>(
        (const float*)q->d_buf, K_ptr, V_ptr,
        d_pmax, d_psum, d_pnum,
        n_q, kv_mul, kv_dim, (int)n_layers, seq_len,
        t_begin, t_end, cur_batch_size, chunk);
    CHECK_HIP(hipGetLastError());

    dim3 grid_r(n_q, cur_batch_size);
    dim3 block_r(head_dim);
    fd_reduce_kernel_stable_v2<64>
    <<<grid_r, block_r, 0, stream>>>(
        d_pmax, d_psum, d_pnum, S_ptr, (float*)tb->d_buf,
        n_q, chunk, cur_batch_size);
    CHECK_HIP(hipGetLastError());

    if (tb_from_device) {
        tb->from_device(stream);
        CHECK_HIP(hipStreamSynchronize(stream));
    }
}

__device__ __forceinline__ double warp_reduce_sum64(double v) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}

__device__ __forceinline__ double warp_reduce_max64(double v) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    v = fmax(v, __shfl_down(v, offset));
  }
  return v;
}

template <int tile_num_queries, int tile_num_kv, int head_dim = 64>
__global__ void flash_attn_2d_kernel(const float *__restrict__ q,
                                     const bf16 *__restrict__ key_cache,
                                     const bf16 *__restrict__ value_cache,
                                     const float *__restrict__ attn_sinks, float *__restrict__ tb,
                                     int batch_size, int n_attn_heads, int kv_mul, int kv_dim,
                                     int seq_len, int n_layers, int pos_begin, int pos_end) {
  const int b = blockIdx.y;
  if (b >= batch_size)
    return;

  const int head_group = blockIdx.x;
  const int base_head = head_group * tile_num_queries;

  const int lane = threadIdx.x % 64;
  const int row = threadIdx.x >> 6;  // For each query head tile
  if (row >= tile_num_queries)
    return;

  const int h = base_head + row;
  if (h >= n_attn_heads)
    return;

  const int kv_head = h / kv_mul;
  const size_t kv_head_offset = kv_head * head_dim;
  const float *q_ptr = q + b * n_attn_heads * head_dim;
  float *tb_ptr = tb + b * n_attn_heads * head_dim;
  const float *q_head_ptr = q_ptr + h * head_dim;
  float *out_head_ptr = tb_ptr + h * head_dim;

  const bf16 *k_cache_ptr = key_cache + (size_t)b * n_layers * seq_len * kv_dim;
  const bf16 *v_cache_ptr = value_cache + (size_t)b * n_layers * seq_len * kv_dim;

  const int active_len = (pos_end - pos_begin > 0) ? (pos_end - pos_begin) : 0;
  if (active_len == 0) {
    for (int i = lane; i < head_dim; i += 64)
      out_head_ptr[i] = 0.f;
    return;
  }

  extern __shared__ unsigned char shmem[];

  // Initalize for local compute
  float *local_q = reinterpret_cast<float *>(shmem);
  bf16 *local_k = reinterpret_cast<bf16 *>(local_q + tile_num_queries * head_dim);
  bf16 *local_v = local_k + 2 * tile_num_kv * head_dim;
  float *local_out = reinterpret_cast<float *>(local_v + 2 * tile_num_kv * head_dim);

  for (int i = lane; i < head_dim; i += WARP_SIZE) {
    local_q[row * head_dim + i] = q_head_ptr[i];
  }

  double m_i = -DBL_MAX;
  double l_i = 0.0;
  for (int i = lane; i < head_dim; i += WARP_SIZE) {
    local_out[row * head_dim + i] = 0.f;
  }
  __syncthreads();

  const int num_kv_tiles = (active_len + tile_num_kv - 1) / tile_num_kv;
  int cur = 0, next = 1;

  // Prefetch tile 0
  {
    const int t0 = 0;
    const int t1 = (tile_num_kv > active_len) ? active_len : tile_num_kv;

    for (int pos = threadIdx.x; pos < (t1 - t0) * head_dim; pos += blockDim.x) {
      size_t local_pos = pos / head_dim;
      size_t d = pos % head_dim;
      size_t global_pos = pos_begin + t0 + local_pos;
      size_t base = global_pos * kv_dim + kv_head_offset + d;

      local_k[cur * tile_num_kv * head_dim + local_pos * head_dim + d] = k_cache_ptr[base];
      local_v[cur * tile_num_kv * head_dim + local_pos * head_dim + d] = v_cache_ptr[base];
    }
  }
  __syncthreads();

  const float inv_sqrt_d = rsqrtf((float)head_dim);

  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int t0 = kv_tile * tile_num_kv;
    const int t1 = (t0 + tile_num_kv > active_len) ? active_len : (t0 + tile_num_kv);
    const int tile_len = t1 - t0;

    if (kv_tile + 1 < num_kv_tiles) {
      const int u0 = (kv_tile + 1) * tile_num_kv;
      const int u1 = (u0 + tile_num_kv > active_len) ? active_len : (u0 + tile_num_kv);

      for (int pos = threadIdx.x; pos < (u1 - u0) * head_dim; pos += blockDim.x) {
        size_t local_pos = pos / head_dim;
        size_t d = pos % head_dim;
        size_t global_pos = pos_begin + u0 + local_pos;
        size_t base = global_pos * kv_dim + kv_head_offset + d;
        local_k[next * tile_num_kv * head_dim + local_pos * head_dim + d] = k_cache_ptr[base];
        local_v[next * tile_num_kv * head_dim + local_pos * head_dim + d] = v_cache_ptr[base];
      }
    }

    __syncthreads();

    double local_max = -DBL_MAX;
    for (int tt = 0; tt < tile_len; tt++) {
      const bf16 *k_vec = &local_k[cur * tile_num_kv * head_dim + tt * head_dim];
      double part = 0.0;

      // Compute element wise of Q.K
      for (int d = lane * 2; d < head_dim; d += WARP_SIZE * 2) {
        const uint32_t pk = ld_u32(reinterpret_cast<const uint32_t *>(k_vec + d));
        const uint16_t u0 = (uint16_t)(pk & 0xFFFF);
        const uint16_t u1 = (uint16_t)(pk >> 16);

        float q0 = local_q[row * head_dim + d];
        float q1 = (d + 1 < head_dim) ? local_q[row * head_dim + d + 1] : 0.0f;

        part = fmaf(q0, bf16_bits_to_f32(u0), part);
        part = fmaf(q1, bf16_bits_to_f32(u1), part);
      }

      double score = warp_reduce_sum64(part);
      if (lane == 0) {
        score *= (double)inv_sqrt_d;
      }

      score = __shfl(score, 0);
      local_max = fmax(local_max, score);
    }

    // Max score of tile
    double m_tile_row = warp_reduce_max64(local_max);

    const double m_new = fmax(m_i, m_tile_row);
    const double alpha = isfinite(m_i) ? exp(m_i - m_new) : 0.0;
    const double beta = exp(m_tile_row - m_new);

    if (lane == 0) {
      l_i *= alpha;
    }

    for (int i = lane; i < head_dim; i += WARP_SIZE) {
      local_out[row * head_dim + i] = (float)((double)local_out[row * head_dim + i] * alpha);
    }

    double l_tile_sum = 0.0;
    for (int out_col = lane * 2; out_col < head_dim; out_col += WARP_SIZE * 2) {
      double acc0 = 0.0, acc1 = 0.0;

      for (int tt = 0; tt < tile_len; tt++) {
        const bf16 *k_vec = &local_k[cur * tile_num_kv * head_dim + tt * head_dim];

        double part = 0.0;
        for (int d = 0; d < head_dim; d += 2) {
          const uint32_t pk = ld_u32(reinterpret_cast<const uint32_t *>(k_vec + d));
          const uint16_t u0 = (uint16_t)(pk & 0xFFFF);
          const uint16_t u1 = (uint16_t)(pk >> 16);

          float q0 = local_q[row * head_dim + d];
          float q1 = local_q[row * head_dim + d + 1];
          part = fmaf(q0, bf16_bits_to_f32(u0), part);
          part = fmaf(q1, bf16_bits_to_f32(u1), part);
        }

        double score = part * (double)inv_sqrt_d;
        const double w = exp(score - m_tile_row);

        if (out_col == 0) {
          l_tile_sum += w;
        }

        const bf16 *v_vec = &local_v[cur * tile_num_kv * head_dim + tt * head_dim];
        const uint32_t pv = ld_u32(reinterpret_cast<const uint32_t *>(v_vec + out_col));
        const uint16_t v0 = (uint16_t)(pv & 0xFFFF);
        const uint16_t v1 = (uint16_t)(pv >> 16);

        acc0 += w * (double)bf16_bits_to_f32(v0);
        if (out_col + 1 < head_dim) {
          acc1 += w * (double)bf16_bits_to_f32(v1);
        }
      }

      float *out_row = &local_out[row * head_dim];
      out_row[out_col] = (float)((double)out_row[out_col] + beta * acc0);
      if (out_col + 1 < head_dim) {
        out_row[out_col + 1] = (float)((double)out_row[out_col + 1] + beta * acc1);
      }
    }

    l_tile_sum = warp_reduce_sum64(l_tile_sum);
    if (lane == 0) {
      l_i += beta * l_tile_sum;
      m_i = m_new;
    }

    __syncthreads();

    int tmp = cur;
    cur = next;
    next = tmp;
  }

  const double sink_logit = (double)attn_sinks[h];
  const double m_final = fmax(m_i, sink_logit);
  const double alpha_f = exp(m_i - m_final);

  // Mẫu số cuối cùng
  double denom_final = l_i * alpha_f + exp(sink_logit - m_final);

  for (int i = lane; i < head_dim; i += WARP_SIZE) {
    double num_i = (double)local_out[row * head_dim + i];
    double num_final = num_i * alpha_f;
    out_head_ptr[i] = (float)(num_final / denom_final);
  }
}

void flash_attn_batched_v2(Tensor *q,       /* (batch_size, n_attn_heads * head_dim) */
                             Tensor *key_cache,   /* (batch_size, n_layers, seq_len, kv_dim)*/
                             Tensor *value_cache, /* (batch_size, n_layers, seq_len, kv_dim)*/
                             Tensor *mask,        /* (batch_size, seq_len, seq_len)*/
                             Tensor *attn_sinks,  /* (n_layers, n_attn_heads) */
                             Tensor *tb,          /* (batch_size, n_attn_heads * head_dim)*/
                             int cur_batch_size, int head_dim, int n_attn_heads, int kv_mul,
                             int kv_dim, int seq_len, int sliding_window, int pos,
                             long long layer_offset, hipStream_t stream) {
  const size_t layer_span = (size_t)seq_len * kv_dim;
  const size_t n_layers = key_cache->shape[1];
  const bf16 *k_ptr = (const bf16 *)key_cache->d_buf + layer_offset * layer_span;
  const bf16 *v_ptr = (const bf16 *)value_cache->d_buf + layer_offset * layer_span;
  const float *s_ptr = (const float *)attn_sinks->d_buf + layer_offset * n_attn_heads;

  const int attn_len = pos + 1;
  const bool use_window = (sliding_window > 0) && ((layer_offset & 1ll) == 0);
  const int active_len = use_window ? std::min(attn_len, sliding_window) : attn_len;
  const int pos_end = attn_len;
  const int pos_begin = pos_end - active_len;

  constexpr int tile_num_queries = 8;  // Br
  constexpr int tile_num_kv = 128;     // Bc - pos * kv_dim

  const int head_groups = (n_attn_heads + tile_num_queries - 1) / tile_num_queries;

  dim3 block_size(tile_num_queries * WARP_SIZE);
  dim3 grid_size(head_groups, cur_batch_size);

  size_t shmem_bytes = 1ll * tile_num_queries * head_dim * sizeof(float) +
                       1ll * 2 * tile_num_kv * head_dim * sizeof(bf16) +
                       1ll * 2 * tile_num_kv * head_dim * sizeof(bf16) +
                       1ll * tile_num_queries * head_dim * sizeof(float);

  flash_attn_2d_kernel<tile_num_queries, tile_num_kv, 64>
    <<<grid_size, block_size, shmem_bytes, stream>>>(
      (const float *)q->d_buf, k_ptr, v_ptr, s_ptr, (float *)tb->d_buf, cur_batch_size,
      n_attn_heads, kv_mul, kv_dim, seq_len, n_layers, pos_begin, pos_end);
}

#else // ifdef KV16 -> The non-KV16 path starts here

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
  const float *__restrict__ q,           // [B, n_q*hd]
  const float *__restrict__ K_cache,     // [B, L, S, kv_dim]  (base already at layer)
  const float *__restrict__ V_cache,     // [B, L, S, kv_dim]  (base already at layer)
  const float *__restrict__ attn_sinks,  // [n_q]              (for this layer)
  float *__restrict__ tb,                // [B, n_q*hd]
  int head_dim, int n_q, int kv_mul, int kv_dim, int total_seq_len, int pos, int sliding_window,
  int batch_size, int total_layers, int use_window) {
  const int head_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  if (head_idx >= n_q || batch_idx >= batch_size)
    return;

  const int kv_head_idx = head_idx / kv_mul;

  const size_t qtb_stride = (size_t)n_q * head_dim;
  const size_t kv_batch_stride = (size_t)total_layers * total_seq_len * kv_dim;

  const float *__restrict__ q_head =
    q + (size_t)batch_idx * qtb_stride + (size_t)head_idx * head_dim;
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
  float *s_q = (float *)sdata;                      // [head_dim]
  float *s_scores = (float *)(s_q + head_dim);      // [TILE_TOKENS]
  double *s_red = (double *)(s_scores + TILE_TOKENS); // [THREADS]

  // Stage q
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    s_q[i] = q_head[i];
  __syncthreads();

  // Online softmax state
  double m = -DBL_MAX;  // running max
  double l = 0.0;       // running sum exp(score - m)
  double o_i = 0.0;     // this thread's output component if i < head_dim

  const size_t kv_head_off = (size_t)kv_head_idx * head_dim;

  for (int t0 = t_begin; t0 < t_end; t0 += TILE_TOKENS) {
    const int tile_len = min(TILE_TOKENS, t_end - t0);

    // Pass 1: scores + tile max
    double local_max = -DBL_MAX;
    for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
      const float *__restrict__ k_vec = K_base + (size_t)(t0 + t) * kv_dim + kv_head_off;
      float s = 0.f;
#pragma unroll 64
      for (int d = 0; d < 64; ++d) {
        // for (int d = 0; d < head_dim; ++d)
        s += s_q[d] * k_vec[d];
      }
      const float score_f = s * scale;
      s_scores[t] = score_f;
      local_max = fmax(local_max, (double)score_f);
    }
    if (tile_len == 0)
      local_max = -DBL_MAX;
    double tile_max = block_reduce_max(local_max, s_red);

    // Rescale prev accumulators if max increases
    const double m_new = fmax(m, tile_max);
    const double alpha = exp(m - m_new);
    l *= alpha;
    if (threadIdx.x < (unsigned)head_dim)
      o_i *= alpha;
    __syncthreads();
    m = m_new;

    // Pass 2: denominator and numerator
    double local_sum_w = 0.0;
    for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
      const double w = exp((double)s_scores[t] - m);
      s_scores[t] = (float)w;  // cache weights for V accumulation
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
  // GpuTimer timer("flash_attention", stream);
  if (q_to_device)
    q->to_device(stream);
  if (k_cache_to_device)
    K_cache->to_device(stream);
  if (v_cache_to_device)
    V_cache->to_device(stream);

  // const int B = (int)q->shape[0];
  const int n_layers = (int)attn_sinks->shape[0];

  const float *K_ptr = (const float *)K_cache->d_buf + 1ll * layer_offset * seq_len * kv_dim;
  const float *V_ptr = (const float *)V_cache->d_buf + 1ll * layer_offset * seq_len * kv_dim;
  const float *S_ptr = (const float *)attn_sinks->d_buf + 1ll * layer_offset * n_q;

  const int use_window =
    (sliding_window > 0 && ((layer_offset & 1ll) == 0) && mask && mask->d_buf) ? 1 : 0;

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
