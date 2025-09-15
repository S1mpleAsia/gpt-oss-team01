// flash_attn_hip.cpp — Flash-Decoding (BF16 KV), fixed batch stride + fast path
#include "../include/flash_attn_hip.hpp"
#include "../include/alloc.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cfloat>
#include <cmath>

using bf16 = hip_bfloat16;

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

#ifndef FA_USE_NONTEMPORAL
#define FA_USE_NONTEMPORAL 1 
#endif

#if FA_USE_NONTEMPORAL && defined(__HIP_PLATFORM_AMD__)
  #define NT_STORE_SCALAR(ptr, val) __builtin_nontemporal_store((val),(ptr))
#else
  #define NT_STORE_SCALAR(ptr, val) (*(ptr) = (val))
#endif

#ifndef FA_SCHED_KV_GROUPED
#define FA_SCHED_KV_GROUPED 1
#endif

__device__ __forceinline__ int map_head_by_kv_group(int bx, int n_q, int kv_mul) {
  if (kv_mul <= 1) return bx;
  const int n_kv = n_q / kv_mul;
  if (n_kv <= 0) return bx;
  const int u  = bx / n_kv;
  const int kv = bx % n_kv;
  return kv * kv_mul + u;
}

__device__ __forceinline__ float bf16_to_f32_bits(uint16_t u) {
  bf16 h; *reinterpret_cast<uint16_t*>(&h) = u; return static_cast<float>(h);
}

__device__ __forceinline__ uint32_t ld_u32(const uint32_t* p) { return *p; }

__device__ __forceinline__ double block_reduce_max(double v, double *s_red) {
  const int tid = threadIdx.x; s_red[tid] = v; __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) s_red[tid] = fmax(s_red[tid], s_red[tid + s]);
    __syncthreads();
  }
  return s_red[0];
}
__device__ __forceinline__ double block_reduce_sum(double v, double *s_red) {
  const int tid = threadIdx.x; s_red[tid] = v; __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) s_red[tid] += s_red[tid + s];
    __syncthreads();
  }
  return s_red[0];
}

// Fast single-chunk kernel (C==1), BF16 KV
template<int TILE_TOKENS /*=128*/, int THREADS /*=128*/, int D /*=64*/>
__global__ void fd_single_kernel_bf16(
  const float *__restrict__ q,          // [B, n_q*D]
  const bf16  *__restrict__ K_cache,    // [B, L, S, kv_dim] already layer-offset in wrapper
  const bf16  *__restrict__ V_cache,    // [B, L, S, kv_dim] "
  const float *__restrict__ attn_sinks, // [n_q]
  float       *__restrict__ tb,         // [B, n_q*D]
  int n_q, int kv_mul, int kv_dim,
  int L, int S,                        
  int t_begin, int t_end,
  int B)
{
  const int b = blockIdx.y;
#if FA_SCHED_KV_GROUPED
  const int h = map_head_by_kv_group(blockIdx.x, n_q, kv_mul);
#else
  const int h = blockIdx.x;
#endif
  if (b >= B || h >= n_q) return;

  const int kv_h = h / kv_mul;
  const int T    = max(0, t_end - t_begin);
  const float scale = rsqrtf((float)D);

  const size_t qtb_stride      = (size_t)n_q * (size_t)D;
  const size_t kv_batch_stride = (size_t)L * (size_t)S * (size_t)kv_dim;
  const size_t kv_head_off     = (size_t)kv_h * (size_t)D;

  const float * __restrict__ q_head = q  + (size_t)b * qtb_stride + (size_t)h * D;
  float       * __restrict__ out    = tb + (size_t)b * qtb_stride + (size_t)h * D;
  const bf16  * __restrict__ Kb     = K_cache + (size_t)b * kv_batch_stride;
  const bf16  * __restrict__ Vb     = V_cache + (size_t)b * kv_batch_stride;

  extern __shared__ int __fd_smem_i32_single[];
  unsigned char *sbase = (unsigned char*)__fd_smem_i32_single;
  float  *s_q   = (float*) sbase;
  float  *s_sc  = (float*)(s_q + D);
  double *s_red = (double*)(s_sc + TILE_TOKENS);

  // Stage Q
  for (int i = threadIdx.x; i < D; i += blockDim.x) s_q[i] = q_head[i];
  __syncthreads();

  // Pass 1
  double local_max = -DBL_MAX;
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    const size_t base = (size_t)(t_begin + t) * (size_t)kv_dim + kv_head_off;

    float sf = 0.f;
#pragma unroll
    for (int d = 0; d < D; d += 2) {
      const uint32_t p = ld_u32(reinterpret_cast<const uint32_t*>(
                        reinterpret_cast<const uint16_t*>(Kb) + base + d));
      const uint16_t u0 = (uint16_t)(p & 0xFFFF);
      const uint16_t u1 = (uint16_t)(p >> 16);
      sf = fmaf(s_q[d+0], bf16_to_f32_bits(u0), sf);
      sf = fmaf(s_q[d+1], bf16_to_f32_bits(u1), sf);
    }
    sf *= scale;
    s_sc[t] = sf;
    if ((double)sf > local_max) local_max = (double)sf;
  }
  double m = (T>0) ? block_reduce_max(local_max, s_red) : -DBL_MAX;
  if (threadIdx.x == 0) m = fmax(m, (double)attn_sinks[h]);
  __syncthreads();
  if (threadIdx.x == 0) s_red[0] = m;
  __syncthreads(); m = s_red[0];

  // Pass 2: exp + denom in double
  double local_sum = 0.0;
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    const double w = exp((double)s_sc[t] - m);
    s_sc[t] = (float)w;
    local_sum += w;
  }
  double l = (T>0) ? block_reduce_sum(local_sum, s_red) : 0.0;
  if (threadIdx.x == 0) s_red[0] = l + exp((double)attn_sinks[h] - m);
  __syncthreads();
  const double inv_denom = 1.0 / s_red[0];

  // Normalize
  for (int t = threadIdx.x; t < T; t += blockDim.x) s_sc[t] = (float)((double)s_sc[t] * inv_denom);
  __syncthreads();

  // Pass 3: V·p 
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    double acc = 0.0;
#pragma unroll 8
    for (int t = 0; t < T; ++t) {
      const size_t off = (size_t)(t_begin + t) * (size_t)kv_dim + kv_head_off + (size_t)i;
      const uint16_t u = reinterpret_cast<const uint16_t*>(Vb)[off];
      acc = fma((double)s_sc[t], (double)bf16_to_f32_bits(u), acc);
    }
    NT_STORE_SCALAR(out + i, (float)acc);   // streaming store for tb
  }
}

// Split-K partials (C>1)
template<int TILE_TOKENS /*=128*/, int THREADS /*=128*/, int D /*=64*/>
__global__ void fd_partial_kernel_bf16(
  const float *__restrict__ q,
  const bf16  *__restrict__ K_cache,   // layer-offset; still need batch stride = L*S*kv_dim
  const bf16  *__restrict__ V_cache,
  float *__restrict__ partial_max,     // [B, n_q, C]   (write-once -> NT store)
  float *__restrict__ partial_sum,     // [B, n_q, C]   (write-once -> NT store)
  float *__restrict__ partial_num,     // [B, n_q, C, D] (write-once -> NT store)
  int n_q, int kv_mul, int kv_dim,
  int L, int S,                         
  int t_begin, int t_end, int B, int C)
{
  const int bx = blockIdx.x, b = blockIdx.y;
  if (b >= B) return;

  const int cidx  = bx / n_q;
  const int h_lin = bx %  n_q;
#if FA_SCHED_KV_GROUPED
  const int h = map_head_by_kv_group(h_lin, n_q, kv_mul);
#else
  const int h = h_lin;
#endif
  if (cidx >= C || h >= n_q) return;

  const int kv_h = h / kv_mul;
  const int start = t_begin + cidx * TILE_TOKENS;
  const int stop  = min(t_end, start + TILE_TOKENS);
  const int T     = max(0, stop - start);

  const size_t qtb_stride      = (size_t)n_q * (size_t)D;
  const size_t kv_batch_stride = (size_t)L * (size_t)S * (size_t)kv_dim;
  const size_t kv_head_off     = (size_t)kv_h * (size_t)D;

  const float * __restrict__ q_head = q + (size_t)b * qtb_stride + (size_t)h * D;
  const bf16  * __restrict__ Kb     = K_cache + (size_t)b * kv_batch_stride;
  const bf16  * __restrict__ Vb     = V_cache + (size_t)b * kv_batch_stride;

  const size_t bhc = ((size_t)b * n_q + h) * (size_t)C;
  float * __restrict__ pmax_bhc = partial_max + bhc;
  float * __restrict__ psum_bhc = partial_sum + bhc;
  float * __restrict__ pnum_bhc = partial_num + bhc * (size_t)D;

  const float scale = rsqrtf((float)D);

  extern __shared__ int __fd_smem_i32[];
  unsigned char *sbase = (unsigned char*)__fd_smem_i32;
  float  *s_q   = (float*) sbase;
  float  *s_sc  = (float*)(s_q + D);
  double *s_red = (double*)(s_sc + TILE_TOKENS);

  for (int i = threadIdx.x; i < D; i += blockDim.x) s_q[i] = q_head[i];
  __syncthreads();

  // QK in float; track max in double
  double local_max = -DBL_MAX;
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    const size_t base = (size_t)(start + t) * (size_t)kv_dim + kv_head_off;

    float sf = 0.f;
#pragma unroll
    for (int d = 0; d < D; d += 2) {
      const uint32_t p = ld_u32(reinterpret_cast<const uint32_t*>(
                        reinterpret_cast<const uint16_t*>(Kb) + base + d));
      const uint16_t u0 = (uint16_t)(p & 0xFFFF);
      const uint16_t u1 = (uint16_t)(p >> 16);
      sf = fmaf(s_q[d+0], bf16_to_f32_bits(u0), sf);
      sf = fmaf(s_q[d+1], bf16_to_f32_bits(u1), sf);
    }
    sf *= scale;
    s_sc[t] = sf;
    if ((double)sf > local_max) local_max = (double)sf;
  }
  const double m_c = (T>0) ? block_reduce_max(local_max, s_red) : -DBL_MAX;

  // exp wrt m_c + denom
  double local_sum = 0.0;
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    const double w = exp((double)s_sc[t] - m_c);
    s_sc[t] = (float)w;
    local_sum += w;
  }
  const double l_c = (T>0) ? block_reduce_sum(local_sum, s_red) : 0.0;

  // numerator (2 outputs/thread) — write-once -> NT stores already used below
  for (int i_out = (threadIdx.x << 1); i_out < D; i_out += (blockDim.x << 1)) {
    double acc0 = 0.0, acc1 = 0.0;
#pragma unroll 8
    for (int t = 0; t < T; ++t) {
      const double w = (double)s_sc[t];
      const size_t base = (size_t)(start + t) * (size_t)kv_dim + kv_head_off + (size_t)i_out;

      const uint32_t p = ld_u32(reinterpret_cast<const uint32_t*>(
                        reinterpret_cast<const uint16_t*>(Vb) + base));
      const uint16_t v0 = (uint16_t)(p & 0xFFFF);
      const uint16_t v1 = (uint16_t)(p >> 16);
      acc0 += w * (double)bf16_to_f32_bits(v0);
      if (i_out + 1 < D) acc1 += w * (double)bf16_to_f32_bits(v1);
    }
    float *dst = pnum_bhc + (size_t)cidx * (size_t)D + (size_t)i_out;
    NT_STORE_SCALAR(dst + 0, (float)acc0);       // streaming store
    if (i_out + 1 < D) NT_STORE_SCALAR(dst + 1, (float)acc1); // streaming store
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    NT_STORE_SCALAR(pmax_bhc + cidx, (float)m_c); // streaming store
    NT_STORE_SCALAR(psum_bhc + cidx, (float)l_c); // streaming store
  }
}

// Final stable reducer 
template<int D /*=64*/>
__global__ void fd_reduce_kernel_stable_v2(
  const float *__restrict__ partial_max,
  const float *__restrict__ partial_sum,
  const float *__restrict__ partial_num,
  const float *__restrict__ attn_sinks,
  float       *__restrict__ tb,
  int n_q, int C, int B)
{
  const int h = blockIdx.x, b = blockIdx.y;
  if (b >= B || h >= n_q) return;

  const size_t qtb_stride = (size_t)n_q * (size_t)D;
  float * __restrict__ out = tb + (size_t)b * qtb_stride + (size_t)h * D;

  const size_t baseBH = ((size_t)b * n_q + h);
  const float *mBH = partial_max + baseBH * (size_t)C;
  const float *sBH = partial_sum + baseBH * (size_t)C;
  const float *nBH = partial_num + baseBH * (size_t)C * (size_t)D;

  double m_star = -DBL_MAX;
  for (int c = 0; c < C; ++c) m_star = fmax(m_star, (double)mBH[c]);
  m_star = fmax(m_star, (double)attn_sinks[h]);

  const int i = threadIdx.x;
  double denom_chunks = 0.0, num_i = 0.0;

  for (int c = 0; c < C; ++c) {
    const double mc = (double)mBH[c];
    const double sc = isfinite(mc) ? exp(mc - m_star) : 0.0;
    denom_chunks += (double)sBH[c] * sc;
    if (i < D) num_i += (double)nBH[(size_t)c*D + i] * sc;
  }
  const double denom = denom_chunks + exp((double)attn_sinks[h] - m_star);
  if (i < D) NT_STORE_SCALAR(out + i, (float)(num_i / denom));  // streaming store for tb
}

void single_query_attn_flash_batched(
  Tensor *q, Tensor *K_cache, Tensor *V_cache, Tensor * /*mask*/,
  Tensor *attn_sinks, Tensor *tb, int cur_batch_size,
  int head_dim, int n_q, int kv_mul, int kv_dim, int seq_len,
  int sliding_window, int pos, long long layer_offset,
  bool q_to_device, bool k_cache_to_device,
  bool v_cache_to_device, bool /*mask_to_device*/,
  bool tb_from_device, hipStream_t stream)
{
  //GpuTimer timer("flash_attention");
  if (q_to_device)       q->to_device(stream);
  if (k_cache_to_device) K_cache->to_device(stream);
  if (v_cache_to_device) V_cache->to_device(stream);

#ifndef KV16
  fprintf(stderr, "[FlashDec] ERROR: expects BF16 KV (KV16).\n"); abort();
#endif

  const int B = cur_batch_size;
  const int D = head_dim;
  const int L = (int)K_cache->shape[1];
  const int S = seq_len;
  const size_t layer_span = (size_t)S * (size_t)kv_dim; // per-layer span
  const bf16 *K_ptr = ((const bf16*)K_cache->d_buf) + layer_offset * layer_span;
  const bf16 *V_ptr = ((const bf16*)V_cache->d_buf) + layer_offset * layer_span;
  const float *S_ptr = (const float*)attn_sinks->d_buf + (size_t)layer_offset * (size_t)n_q;

  // even-layer windowing
  const int attn_len = pos + 1;
  const bool use_window = (sliding_window > 0) && ((layer_offset & 1ll) == 0);
  const int L_eff     = use_window ? min(attn_len, sliding_window) : attn_len;
  const int t_end     = attn_len;
  const int t_begin   = t_end - L_eff;

  int dev_id = 0; CHECK_HIP(hipGetDevice(&dev_id));
  const int TILE = g_fa_tile[dev_id];          // e.g., 128
  const int C    = (L_eff <= TILE) ? 1 : ((L_eff + TILE - 1) / TILE);

  //printf("n_q: %d & pos: %d\n", n_q, pos);

  if (C == 1) {
    dim3 grid(n_q, B), block(128);
    const size_t shmem = (size_t)D*sizeof(float) + (size_t)TILE*sizeof(float) + (size_t)block.x*sizeof(double);
    if      (TILE==256) fd_single_kernel_bf16<256,128,64><<<grid,block,shmem,stream>>>((const float*)q->d_buf, K_ptr, V_ptr, S_ptr, (float*)tb->d_buf, n_q, kv_mul, kv_dim, L, S, t_begin, t_end, B);
    else if (TILE==192) fd_single_kernel_bf16<192,128,64><<<grid,block,shmem,stream>>>((const float*)q->d_buf, K_ptr, V_ptr, S_ptr, (float*)tb->d_buf, n_q, kv_mul, kv_dim, L, S, t_begin, t_end, B);
    else if (TILE==160) fd_single_kernel_bf16<160,128,64><<<grid,block,shmem,stream>>>((const float*)q->d_buf, K_ptr, V_ptr, S_ptr, (float*)tb->d_buf, n_q, kv_mul, kv_dim, L, S, t_begin, t_end, B);
    else if (TILE==128) fd_single_kernel_bf16<128,128,64><<<grid,block,shmem,stream>>>((const float*)q->d_buf, K_ptr, V_ptr, S_ptr, (float*)tb->d_buf, n_q, kv_mul, kv_dim, L, S, t_begin, t_end, B);
    else if (TILE==96 ) fd_single_kernel_bf16< 96,128,64><<<grid,block,shmem,stream>>>((const float*)q->d_buf, K_ptr, V_ptr, S_ptr, (float*)tb->d_buf, n_q, kv_mul, kv_dim, L, S, t_begin, t_end, B);
    else                fd_single_kernel_bf16<128,128,64><<<grid,block,shmem,stream>>>((const float*)q->d_buf, K_ptr, V_ptr, S_ptr, (float*)tb->d_buf, n_q, kv_mul, kv_dim, L, S, t_begin, t_end, B);
    CHECK_HIP(hipGetLastError());
  } else {
    if (g_fa_C_max[dev_id] < C) { fprintf(stderr,"[FlashDec] C=%d > g_fa_C_max=%d\n", C, g_fa_C_max[dev_id]); abort(); }
    float *d_pmax = g_fa_pmax[dev_id], *d_psum = g_fa_psum[dev_id], *d_pnum = g_fa_pnum[dev_id];

    dim3 grid_p(n_q*C, B), block_p(128);
    const size_t shmem_p = (size_t)D*sizeof(float) + (size_t)TILE*sizeof(float) + (size_t)block_p.x*sizeof(double);
    if      (TILE==256) fd_partial_kernel_bf16<256,128,64><<<grid_p,block_p,shmem_p,stream>>>((const float*)q->d_buf, K_ptr, V_ptr, d_pmax, d_psum, d_pnum, n_q, kv_mul, kv_dim, L, S, t_begin, t_end, B, C);
    else if (TILE==192) fd_partial_kernel_bf16<192,128,64><<<grid_p,block_p,shmem_p,stream>>>((const float*)q->d_buf, K_ptr, V_ptr, d_pmax, d_psum, d_pnum, n_q, kv_mul, kv_dim, L, S, t_begin, t_end, B, C);
    else if (TILE==160) fd_partial_kernel_bf16<160,128,64><<<grid_p,block_p,shmem_p,stream>>>((const float*)q->d_buf, K_ptr, V_ptr, d_pmax, d_psum, d_pnum, n_q, kv_mul, kv_dim, L, S, t_begin, t_end, B, C);
    else if (TILE==128) fd_partial_kernel_bf16<128,128,64><<<grid_p,block_p,shmem_p,stream>>>((const float*)q->d_buf, K_ptr, V_ptr, d_pmax, d_psum, d_pnum, n_q, kv_mul, kv_dim, L, S, t_begin, t_end, B, C);
    else if (TILE==96 ) fd_partial_kernel_bf16< 96,128,64><<<grid_p,block_p,shmem_p,stream>>>((const float*)q->d_buf, K_ptr, V_ptr, d_pmax, d_psum, d_pnum, n_q, kv_mul, kv_dim, L, S, t_begin, t_end, B, C);
    else                fd_partial_kernel_bf16<128,128,64><<<grid_p,block_p,shmem_p,stream>>>((const float*)q->d_buf, K_ptr, V_ptr, d_pmax, d_psum, d_pnum, n_q, kv_mul, kv_dim, L, S, t_begin, t_end, B, C);
    CHECK_HIP(hipGetLastError());

    dim3 grid_r(n_q, B), block_r(D);
    fd_reduce_kernel_stable_v2<64><<<grid_r, block_r, 0, stream>>>(d_pmax, d_psum, d_pnum, S_ptr, (float*)tb->d_buf, n_q, C, B);
    CHECK_HIP(hipGetLastError());
  }

  if (tb_from_device) { tb->from_device(stream); CHECK_HIP(hipStreamSynchronize(stream)); }
}
