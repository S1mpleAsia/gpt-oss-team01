#include "../include/flash_attn_hip.hpp"
#include <cfloat>
#include <cmath>
#include <hip/hip_runtime.h>

// Wave size on AMD
#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

__device__ __forceinline__
double block_reduce_max(double v, double* s_red) {
 const int tid = threadIdx.x;
 s_red[tid] = v;
 __syncthreads();
 for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
   if (tid < s) s_red[tid] = fmax(s_red[tid], s_red[tid + s]);
   __syncthreads();
 }
 return s_red[0];
}

__device__ __forceinline__
double block_reduce_sum(double v, double* s_red) {
 const int tid = threadIdx.x;
 s_red[tid] = v;
 __syncthreads();
 for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
   if (tid < s) s_red[tid] += s_red[tid + s];
   __syncthreads();
 }
 return s_red[0];
}

template<int TILE_TOKENS = 128, int THREADS = 128>
__global__ void flash_attn_decode_kernel(
 const float* __restrict__ q,           // [B, n_q*hd]
 const float* __restrict__ K_cache,     // [B, L, S, kv_dim]   (base already at layer)
 const float* __restrict__ V_cache,     // [B, L, S, kv_dim]   (base already at layer)
 const float* __restrict__ attn_sinks,  // [n_q]               (for this layer)
 float* __restrict__ tb,                // [B, n_q*hd]
 int head_dim, int n_q, int kv_mul, int kv_dim,
 int total_seq_len, int pos, int sliding_window,
 int batch_size, int total_layers, int use_window)
{
 const int head_idx  = blockIdx.x;
 const int batch_idx = blockIdx.y;
 if (head_idx >= n_q || batch_idx >= batch_size) return;

 const int kv_head_idx = head_idx / kv_mul;

 const size_t qtb_stride      = (size_t)n_q * head_dim;
 const size_t kv_batch_stride = (size_t)total_layers * total_seq_len * kv_dim;

 const float* __restrict__ q_head =
     q  + (size_t)batch_idx * qtb_stride + (size_t)head_idx * head_dim;
 float* __restrict__ tb_head =
     tb + (size_t)batch_idx * qtb_stride + (size_t)head_idx * head_dim;

 const float* __restrict__ K_base = K_cache + (size_t)batch_idx * kv_batch_stride;
 const float* __restrict__ V_base = V_cache + (size_t)batch_idx * kv_batch_stride;
 const float* __restrict__ attn_sinks_head = attn_sinks + (size_t)head_idx;

 const float scale = rsqrtf((float)head_dim);

 const int attn_len = pos + 1;
 int t_begin = 0;
 if (use_window) t_begin = max(0, attn_len - sliding_window);
 const int t_end = attn_len;

 extern __shared__ unsigned char sdata[];
 float*  s_q      = (float*)  sdata;                          // [head_dim]
 float*  s_scores = (float*) (s_q + head_dim);                // [TILE_TOKENS]
 double* s_red    = (double*)(s_scores + TILE_TOKENS);        // [THREADS]

 // Stage q
 for (int i = threadIdx.x; i < head_dim; i += blockDim.x) s_q[i] = q_head[i];
 __syncthreads();

 // Online softmax state
 double m = -DBL_MAX;   // running max
 double l = 0.0;        // running sum exp(score - m)
 double o_i = 0.0;      // this thread's output component if i < head_dim

 const size_t kv_head_off = (size_t)kv_head_idx * head_dim;

 for (int t0 = t_begin; t0 < t_end; t0 += TILE_TOKENS) {
   const int tile_len = min(TILE_TOKENS, t_end - t0);

   // Pass 1: scores + tile max
   double local_max = -DBL_MAX;
   for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
     const float* __restrict__ k_vec = K_base + (size_t)(t0 + t) * kv_dim + kv_head_off;
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
   if (tile_len == 0) local_max = -DBL_MAX;
   double tile_max = block_reduce_max(local_max, s_red);

   // Rescale prev accumulators if max increases
   const double m_new = fmax(m, tile_max);
   const double alpha = exp(m - m_new);
   l *= alpha;
   if (threadIdx.x < (unsigned)head_dim) o_i *= alpha;
   __syncthreads();
   m = m_new;

   // Pass 2: denominator and numerator
   double local_sum_w = 0.0;
   for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
     const double w = exp((double)s_scores[t] - m);
     s_scores[t] = (float)w;    // cache weights for V accumulation
     local_sum_w += w;
   }
   const double sum_w = block_reduce_sum(local_sum_w, s_red);

   if (threadIdx.x < (unsigned)head_dim) {
     const int i_out = threadIdx.x;
     double acc = o_i;
     for (int t = 0; t < tile_len; ++t) {
       const double w = (double)s_scores[t];
       const float* __restrict__ v_vec = V_base + (size_t)(t0 + t) * kv_dim + kv_head_off;
       acc += w * (double)v_vec[i_out];
     }
     o_i = acc;
   }
   __syncthreads();

   if (threadIdx.x == 0) s_red[0] = l + sum_w;
   __syncthreads();
   l = s_red[0];
 }

 const double sink    = (double)(*attn_sinks_head);
 const double m_final = fmax(m, sink);
 const double alpha_f = exp(m - m_final);

 const double l_final = l * alpha_f + exp(sink - m_final);

 if (threadIdx.x < (unsigned)head_dim) {
   const double o_scaled = o_i * alpha_f;        
   tb_head[threadIdx.x] = (float)(o_scaled / l_final);
 }
}

void single_query_attn_flash_batched(
 Tensor *q, Tensor *K_cache, Tensor *V_cache, Tensor *mask,
 Tensor *attn_sinks, Tensor *tb,
 int head_dim, int n_q, int kv_mul, int kv_dim, int seq_len,
 int sliding_window, int pos, long long layer_offset,
 bool q_to_device, bool k_cache_to_device, bool v_cache_to_device,
 bool mask_to_device, bool tb_from_device, hipStream_t stream)
{
   GpuTimer timer("single_query_attn_flash_batched");
 if (q_to_device)       q->to_device(stream);
 if (k_cache_to_device) K_cache->to_device(stream);
 if (v_cache_to_device) V_cache->to_device(stream);

 const int B          = (int)q->shape[0];
 const int n_layers   = (int)attn_sinks->shape[0];

 const float *K_ptr = (const float*)K_cache->d_buf + 1ll * layer_offset * seq_len * kv_dim;
 const float *V_ptr = (const float*)V_cache->d_buf + 1ll * layer_offset * seq_len * kv_dim;
 const float *S_ptr = (const float*)attn_sinks->d_buf + 1ll * layer_offset * n_q;

 const int use_window = (sliding_window > 0 && ((layer_offset & 1ll) == 0) &&
                         mask && mask->d_buf) ? 1 : 0;

 constexpr int TILE_TOKENS = 128;
 constexpr int THREADS     = 128;
 dim3 grid(n_q, B), block(THREADS);
 size_t shmem = (size_t)head_dim * sizeof(float)
              + (size_t)TILE_TOKENS * sizeof(float)
              + (size_t)THREADS * sizeof(double);

 flash_attn_decode_kernel<TILE_TOKENS, THREADS>
   <<<grid, block, shmem, stream>>>(
     (const float*)q->d_buf, K_ptr, V_ptr, S_ptr, (float*)tb->d_buf,
     head_dim, n_q, kv_mul, kv_dim, seq_len, pos, sliding_window,
     B, n_layers, use_window);

 CHECK_HIP(hipGetLastError());
 if (tb_from_device) { tb->from_device(stream); CHECK_HIP(hipStreamSynchronize(stream)); }
}


