#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

using bf16 = hip_bfloat16;
using f32x4 = float __attribute__((ext_vector_type(4)));
using bf16_isa = __bf16;
using bf16x4 = bf16_isa __attribute__((__vector_size__(4 * sizeof(bf16_isa))));

#if defined(__HIP_DEVICE_COMPILE__)
# if defined(__gfx90a__)
#  if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16)
#   define BF16_MFMA_KSTEP 16
#   define MFMA_BF16_16x16(a,b,c) __builtin_amdgcn_mfma_f32_16x16x16bf16((a),(b),(c),0,0,0)
#  elif __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
#   define BF16_MFMA_KSTEP 16
#   define MFMA_BF16_16x16(a,b,c) __builtin_amdgcn_mfma_f32_16x16x16bf16_1k((a),(b),(c),0,0,0)
#  elif __has_builtin(__builtin_amdgcn_mfma_f32_16x16x8bf16)
#   define BF16_MFMA_KSTEP 8
#   define MFMA_BF16_16x16(a,b,c) __builtin_amdgcn_mfma_f32_16x16x8bf16((a),(b),(c),0,0,0)
#  elif __has_builtin(__builtin_amdgcn_mfma_f32_16x16x8bf16_1k)
#   define BF16_MFMA_KSTEP 8
#   define MFMA_BF16_16x16(a,b,c) __builtin_amdgcn_mfma_f32_16x16x8bf16_1k((a),(b),(c),0,0,0)
#  else
#   error "gfx90a device compile without BF16 MFMA builtins"
#  endif
# else
#  error "This kernel targets gfx90a (MI250)."
# endif
#else
# define BF16_MFMA_KSTEP 16
# define MFMA_BF16_16x16(a,b,c) (c)
#endif

__device__ __forceinline__ bf16x4 pack_f4_to_bf16x4(const float4 &v) {
  bf16x4 r;
  r[0] = (bf16_isa)v.x; r[1] = (bf16_isa)v.y; r[2] = (bf16_isa)v.z; r[3] = (bf16_isa)v.w;
  return r;
}

/**
 * No-prefetch + strict timing:
 *  For each K tile:
 *    - time LOAD:   global A/B -> LDS (ends after sync)
 *    - time COMPUTE: MFMA loop only (start after sync, stop before sync)
 */
template <int BM, int BN, int BK, int TM, int TN, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS)
void gemm_mfma_v2_nopf_profiled(const float *__restrict__ A,   // [M,K] fp32
                                       const bf16  *__restrict__ B,   // [K,N] bf16
                                       float       *__restrict__ C,   // [M,N] fp32
                                       const bf16  *__restrict__ bias,// [N] or null
                                       int M, int N, int K,
                                       unsigned long long *g_load_cycles,
                                       unsigned long long *g_comp_cycles) {
  static_assert(BK % BF16_MFMA_KSTEP == 0, "BK must be multiple of MFMA K-step");
  constexpr int WM = 16, WN = 16, WK = BF16_MFMA_KSTEP;
  constexpr int VEC_B_SIZE = 8;   // 8*bf16 = 16B -> uint4
  constexpr int VEC_A_SIZE = 4;   // float4

  const int tid     = threadIdx.x;
  const int wave_id = tid >> 6;
  const int lane_id = tid & 63;

  const int block_row_start = blockIdx.y * BM;
  const int block_col_start = blockIdx.x * BN;

  const int waves_per_block_m = BM / TM;
  const int waves_per_block_n = BN / TN;
  const int wave_row = wave_id /  waves_per_block_n;
  const int wave_col = wave_id %  waves_per_block_n;

  const int wave_row_start = block_row_start + wave_row * TM;
  const int wave_col_start = block_col_start + wave_col * TN;

  // +2 padding to ease LDS bank conflicts on strides
  __shared__ bf16_isa As[BM][BK + 2];
  __shared__ bf16_isa Bs[BK][BN + 2];

  constexpr int M_TILES = TM / WM;
  constexpr int N_TILES = TN / WN;
  f32x4 acc[M_TILES][N_TILES];
#pragma unroll
  for (int i = 0; i < M_TILES; ++i)
#pragma unroll
    for (int j = 0; j < N_TILES; ++j)
      acc[i][j] = {0.f, 0.f, 0.f, 0.f};

  constexpr int A_ELEMS_TILE = BM * BK;
  constexpr int B_ELEMS_TILE = BK * BN;
  constexpr int A_VEC_PER_THR = A_ELEMS_TILE / VEC_A_SIZE / BLOCK_THREADS;
  constexpr int B_VEC_ELEMS   = B_ELEMS_TILE / VEC_B_SIZE;
  constexpr int B_VEC_PER_THR = B_VEC_ELEMS / BLOCK_THREADS;

  const int lx = lane_id & 15;
  const int ly = (lane_id >> 4) & (WK == 16 ? 3 : 1);

  unsigned long long load_cycles_sum = 0;
  unsigned long long comp_cycles_sum = 0;

  for (int k_base = 0; k_base < K; k_base += BK) {
    // ---- LOAD phase: global -> LDS (time this) ----
    unsigned long long t_ld0=0, t_ld1=0;
    if (threadIdx.x == 0) t_ld0 = clock64();

#pragma unroll
    for (int i = 0; i < A_VEC_PER_THR; i++) {
      const int vec_idx  = tid + i * BLOCK_THREADS;
      const int elem_idx = vec_idx * VEC_A_SIZE;
      const int r = elem_idx / BK;
      const int c = elem_idx % BK;
      const int g_row = block_row_start + r;
      const int g_col = k_base + c;

      if (g_row < M && (g_col + VEC_A_SIZE - 1) < K) {
        const float4 v = *reinterpret_cast<const float4 *>(&A[(size_t)g_row * K + g_col]);
        const bf16x4 packed = pack_f4_to_bf16x4(v);
        As[r][c + 0] = packed[0]; As[r][c + 1] = packed[1];
        As[r][c + 2] = packed[2]; As[r][c + 3] = packed[3];
      } else {
#pragma unroll
        for (int j = 0; j < VEC_A_SIZE; ++j) {
          const float val = (g_row < M && (g_col + j) < K)
                              ? A[(size_t)g_row * K + (g_col + j)] : 0.0f;
          As[r][c + j] = (bf16_isa)val;
        }
      }
    }

#pragma unroll
    for (int i = 0; i < B_VEC_PER_THR; ++i) {
      const int vec_idx  = tid + i * BLOCK_THREADS;
      const int elem_idx = vec_idx * VEC_B_SIZE;
      const int r = elem_idx / BN;
      const int c = elem_idx % BN;
      const int g_row = k_base + r;
      const int g_col = block_col_start + c;

      if (g_row < K && (g_col + VEC_B_SIZE - 1) < N) {
        *reinterpret_cast<uint4 *>(&Bs[r][c]) =
          *reinterpret_cast<const uint4 *>(&B[(size_t)g_row * N + g_col]);
      } else {
        uint4 zero = {0,0,0,0};
        *reinterpret_cast<uint4 *>(&Bs[r][c]) = zero;
      }
    }

    __syncthreads(); // LOAD ends after this sync
    if (threadIdx.x == 0) { t_ld1 = clock64(); load_cycles_sum += (t_ld1 - t_ld0); }

    // ---- COMPUTE phase: MFMA only (time this) ----
    unsigned long long t_c0=0, t_c1=0;
    if (threadIdx.x == 0) t_c0 = clock64();

#pragma unroll
    for (int kk = 0; kk < BK; kk += WK) {
#pragma unroll
      for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
          const int a_row = wave_row * TM + mt * WM + lx;
          const int b_col = wave_col * TN + nt * WN + lx;

          bf16x4 a_vec, b_vec;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int kcol = i + ly * 4;
            a_vec[i] = As[a_row][kk + kcol];
            b_vec[i] = Bs[kk + kcol][b_col];
          }
          acc[mt][nt] = MFMA_BF16_16x16(a_vec, b_vec, acc[mt][nt]);
        }
      }
    }

    if (threadIdx.x == 0) { t_c1 = clock64(); comp_cycles_sum += (t_c1 - t_c0); }
    __syncthreads(); // not counted in compute; prepares for next tile load
  }

  // ---- Write back ----
  const int lxx = lane_id & 15;
#pragma unroll
  for (int mt = 0; mt < M_TILES; ++mt)
#pragma unroll
    for (int nt = 0; nt < N_TILES; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = wave_row_start + mt * WM + (i + 4 * (lane_id >> 4));
        const int col = wave_col_start + nt * WN + lxx;
        if (row < M && col < N) {
          float out = acc[mt][nt][i];
          if (bias) out += static_cast<float>(bias[col]);
          C[(size_t)row * N + col] = out;
        }
      }

  if (threadIdx.x == 0) {
    atomicAdd(g_load_cycles, load_cycles_sum);
    atomicAdd(g_comp_cycles, comp_cycles_sum);
  }
}

// Launcher (same tiling as your other kernels)
inline void launch_gemm_mfma_v2_nopf_profiled(
    const float *A, const bf16 *B, float *C, int M, int N, int K,
    unsigned long long *d_load_cycles, unsigned long long *d_comp_cycles,
    hipStream_t stream = nullptr, const bf16 *bias = nullptr) {

  constexpr int BM = 64, BN = 128, BK = 32, TM = 32, TN = 32, BLOCK_THREADS = 512;
  dim3 block_dim(BLOCK_THREADS);
  dim3 grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM);

  hipLaunchKernelGGL(
      (gemm_mfma_v2_nopf_profiled<BM, BN, BK, TM, TN, BLOCK_THREADS>),
      grid_dim, block_dim, 0, stream,
      A, B, C, bias, M, N, K, d_load_cycles, d_comp_cycles);
}
