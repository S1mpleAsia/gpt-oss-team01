#include <hip/hip_runtime.h>
#include "../include/tensor.hpp"

typedef float f32x4 __attribute__((ext_vector_type(4)));

__global__ void gemm_mfma_fp32_naive(const float *__restrict__ A,
                                     const hip_bfloat16 *__restrict__ B, float *__restrict__ D,
                                     int M, int N, int K) {
  const int lx = threadIdx.x;
  const int ly = threadIdx.y;
  const int m0 = blockIdx.y * 16;
  const int n0 = blockIdx.x * 16;

  const int LDA = K, LDB = N, LDD = N;

  f32x4 acc = {0.f, 0.f, 0.f, 0.f};

  for (int k0 = 0; k0 < K; k0 += 4) {
    float av = 0.f, bv = 0.f;

    int a_r = m0 + lx;
    int a_c = k0 + ly;
    if (a_r < M && a_c < K)
      av = A[(size_t)a_r * LDA + a_c];

    int b_r = k0 + ly;
    int b_c = n0 + lx;
    if (b_r < K && b_c < N)
      bv = (float)B[(size_t)b_r * LDB + b_c];

    acc = __builtin_amdgcn_mfma_f32_16x16x4f32(av, bv, acc, /*cbsz*/ 0, /*abid*/ 0, /*blgp*/ 0);
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int r = m0 + (i + 4 * ly);
    int c = n0 + lx;
    if (r < M && c < N)
      D[(size_t)r * LDD + c] = acc[i];
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN, const int blockDim>
__global__ void gemm_mfma(const float *__restrict__ A, const bf16 *__restrict__ B,
                          float *__restrict__ C, const bf16 *__restrict__ bias, int M, int N,
                          int K) {
  // constexpr int VEC_A_SIZE = 4;
  constexpr int VEC_B_SIZE = 8;
  constexpr int WM = 16;
  constexpr int WN = 16;
  constexpr int WK = 4;

  const int tid = threadIdx.x;
  const int wave_id = tid >> 6;  // wave_id = tid / 64
  const int lane_id = tid % 64;

  const int waves_per_block_m = BM / TM;
  const int waves_per_block_n = BN / TN;
  const int wave_row = wave_id / waves_per_block_n;
  const int wave_col = wave_id % waves_per_block_n;

  const int block_row_start = blockIdx.y * BM;
  const int block_col_start = blockIdx.x * BN;

  const int wave_row_start = block_row_start + wave_row * TM;
  const int wave_col_start = block_col_start + wave_col * TN;

  __shared__ float As[BM][BK + 2];
  __shared__ bf16 Bs[BK][BN + 2];

  constexpr int M_TILES = TM / WM;
  constexpr int N_TILES = TN / WN;
  f32x4 acc[M_TILES][N_TILES];

#pragma unroll
  for (int i = 0; i < M_TILES; i++) {
#pragma unroll
    for (int j = 0; j < N_TILES; j++) {
      acc[i][j] = {0.f, 0.f, 0.f, 0.f};
    }
  }

  // constexpr int A_ELEMS = BM * BK;
  constexpr int A_TILES_PER_THREAD = (BM * BK) / blockDim;
  constexpr int B_TILES_PER_THREAD = (BK * BN) / blockDim;

  // constexpr int A_VEC_ELEMS = (BM * BK) / VEC_A_SIZE;
  // constexpr int A_VEC_PER_THREAD = A_VEC_ELEMS / blockDim;

  constexpr int B_VEC_ELEMS = (BK * BN) / VEC_B_SIZE;
  constexpr int B_VEC_PER_THREAD = B_VEC_ELEMS / blockDim;

  for (int k_base = 0; k_base < K; k_base += BK) {
#pragma unroll
    for (int i = 0; i < A_TILES_PER_THREAD; i++) {
      int idx = tid + i * blockDim;
      int r = idx / BK;
      int c = idx % BK;
      int g_row = block_row_start + r;
      int g_col = k_base + c;

      if (g_row < M && g_col < K) {
        As[r][c] = A[(size_t)g_row * K + g_col];
      } else {
        As[r][c] = 0.0f;
      }
    }

    // for (int idx = tid; idx < A_ELEMS; idx += blockDim) {
    //   int r = idx / BK;
    //   int c = idx % BK;
    //   int g_row = block_row_start + r;
    //   int g_col = k_base + c;
    //   if (g_row < M && g_col < K) {
    //     As[r][c] = A[(size_t)g_row * K + g_col];
    //   } else {
    //     As[r][c] = 0.0f;
    //   }
    // }

    // #pragma unroll
    //     for (int i = 0; i < B_TILES_PER_THREAD; i++) {
    //       int idx = tid + i * blockDim;
    //       int r = idx / BN;
    //       int c = idx % BN;
    //       int g_row = k_base + r;
    //       int g_col = block_col_start + c;
    //       if (g_row < K && g_col < N) {
    //         Bs[r][c] = B[(size_t)g_row * N + g_col];
    //       } else {
    //         Bs[r][c] = 0.0f;
    //       }
    //     }

#pragma unroll
    for (int i = 0; i < B_VEC_PER_THREAD; i++) {
      int vec_idx = tid + i * blockDim;
      int elem_idx = vec_idx * VEC_B_SIZE;
      int r = elem_idx / BN;
      int c = elem_idx % BN;
      int g_row = k_base + r;
      int g_col = block_col_start + c;

      if (g_row < K && (g_col + VEC_B_SIZE - 1) < N) {
        *reinterpret_cast<uint4 *>(&Bs[r][c]) =
          *reinterpret_cast<const uint4 *>(&B[(size_t)g_row * N + g_col]);
      } else {
        for (int j = 0; j < VEC_B_SIZE; j++) {
          if (g_row < K && (g_col + j) < N) {
            Bs[r][c + j] = B[(size_t)g_row * N + (g_col + j)];
          } else {
            Bs[r][c + j] = 0.0f;
          }
        }
      }
    }

    __syncthreads();

    const int lx = lane_id % 16;
    const int ly = (lane_id / 16) % 4;

#pragma unroll
    for (int k = 0; k < BK; k += WK) {
#pragma unroll
      for (int m_tile = 0; m_tile < M_TILES; m_tile++) {
#pragma unroll
        for (int n_tile = 0; n_tile < N_TILES; n_tile++) {
          int a_row_offset = wave_row * TM + m_tile * WM;
          int b_col_offset = wave_col * TN + n_tile * WN;

          float av = As[a_row_offset + lx][k + ly];
          float bv = (float)Bs[k + ly][b_col_offset + lx];

          acc[m_tile][n_tile] =
            __builtin_amdgcn_mfma_f32_16x16x4f32(av, bv, acc[m_tile][n_tile], 0, 0, 0);
        }
      }
    }

    __syncthreads();
  }

  const int lx = lane_id % 16;
#pragma unroll
  for (int m_tile = 0; m_tile < M_TILES; m_tile++) {
#pragma unroll
    for (int n_tile = 0; n_tile < N_TILES; n_tile++) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int row = wave_row_start + m_tile * WM + (i + 4 * (lane_id / 16));
        int col = wave_col_start + n_tile * WN + lx;
        if (row < M && col < N) {
          float res = acc[m_tile][n_tile][i];
          if (bias) {
            res += (float)bias[col];
          }
          C[(size_t)row * N + col] = res;
        }
      }
    }
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN, const int blockDim>
__global__ void gemm_mfma_moe(const float *__restrict__ A, const bf16 *__restrict__ B,
                              float *__restrict__ C, const bf16 *__restrict__ bias,
                              const int *__restrict__ expert_offsets, int M_total, int N, int K) {
  constexpr int VEC_B_SIZE = 8;
  constexpr int WM = 16, WN = 16, WK = 4;

  const int expert_id = blockIdx.z;

  const int row_begin = expert_offsets[expert_id];
  const int row_end = expert_offsets[expert_id + 1];
  const int M = (row_end - row_begin > 0) ? (row_end - row_begin) : 0;

  if (M <= 0) {
    return;
  }

  const bf16 *B_ptr = B + (size_t)expert_id * N * K;
  const bf16 *bias_ptr = bias ? (bias + (size_t)expert_id * N) : nullptr;

  const int tid = threadIdx.x;
  const int wave_id = tid >> 6;
  const int lane_id = tid % 64;

  const int waves_per_block_m = BM / TM;
  const int waves_per_block_n = BN / TN;
  const int wave_row = wave_id / waves_per_block_n;
  const int wave_col = wave_id % waves_per_block_n;

  const int block_row_local_start = blockIdx.y * BM;
  const int block_col_start = blockIdx.x * BN;

  if (block_row_local_start >= M) {
    return;
  }

  const int wave_row_start = block_row_local_start + wave_row * TM;
  const int wave_col_start = block_col_start + wave_col * TN;

  __shared__ float As[BM][BK + 4];
  __shared__ bf16 Bs[BK][BN + 4];

  constexpr int M_TILES = TM / WM;
  constexpr int N_TILES = TN / WN;
  f32x4 acc[M_TILES][N_TILES] = {};

#pragma unroll
  for (int i = 0; i < M_TILES; i++) {
#pragma unroll
    for (int j = 0; j < N_TILES; j++) {
      acc[i][j] = {0.f, 0.f, 0.f, 0.f};
    }
  }

  constexpr int A_TILES_PER_THREAD = (BM * BK) / blockDim;
  constexpr int B_TILES_PER_THREAD = (BK * BN) / blockDim;

  constexpr int B_VEC_ELEMS = (BK * BN) / VEC_B_SIZE;
  constexpr int B_VEC_PER_THREAD = B_VEC_ELEMS / blockDim;

  for (int k_base = 0; k_base < K; k_base += BK) {
#pragma unroll
    for (int i = 0; i < A_TILES_PER_THREAD; i++) {
      int idx = tid + i * blockDim;
      int r_local = idx / BK;
      int c = idx % BK;
      int g_row_local = block_row_local_start + r_local;
      int g_row_global = row_begin + g_row_local;
      int g_col = k_base + c;

      As[r_local][c] = (g_row_local < M && g_col < K) ? A[(size_t)g_row_global * K + g_col] : 0.0f;
    }

#pragma unroll
    for (int i = 0; i < B_VEC_PER_THREAD; i++) {
      int vec_idx = tid + i * blockDim;
      int elem_idx = vec_idx * VEC_B_SIZE;
      int r = elem_idx / BN;
      int c = elem_idx % BN;
      int g_row = k_base + r;
      int g_col = block_col_start + c;

      if (g_row < K && (g_col + VEC_B_SIZE - 1) < N) {
        *reinterpret_cast<uint4 *>(&Bs[r][c]) =
          *reinterpret_cast<const uint4 *>(&B_ptr[(size_t)g_row * N + g_col]);
      } else {
        for (int j = 0; j < VEC_B_SIZE; j++) {
          Bs[r][c + j] =
            (g_row < K && (g_col + j) < N) ? B_ptr[(size_t)g_row * N + (g_col + j)] : 0.0f;
        }
      }
    }

    __syncthreads();

    const int lx = lane_id % 16;
    const int ly = (lane_id / 16) % 4;

#pragma unroll
    for (int k = 0; k < BK; k += WK) {
#pragma unroll
      for (int m_tile = 0; m_tile < M_TILES; m_tile++) {
#pragma unroll
        for (int n_tile = 0; n_tile < N_TILES; n_tile++) {
          int a_row_offset = wave_row * TM + m_tile * WM;
          int b_col_offset = wave_col * TN + n_tile * WN;
          float av = As[a_row_offset + lx][k + ly];
          float bv = (float)Bs[k + ly][b_col_offset + lx];
          acc[m_tile][n_tile] =
            __builtin_amdgcn_mfma_f32_16x16x4f32(av, bv, acc[m_tile][n_tile], 0, 0, 0);
        }
      }
    }

    __syncthreads();
  }

  const int lx = lane_id % 16;

#pragma unroll
  for (int m_tile = 0; m_tile < M_TILES; m_tile++) {
#pragma unroll
    for (int n_tile = 0; n_tile < N_TILES; n_tile++) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int row_local = wave_row_start + m_tile * WM + (i + 4 * (lane_id / 16));
        int row_global = row_begin + row_local;
        int col = wave_col_start + n_tile * WN + lx;

        if (row_local < M && col < N) {
          float res = acc[m_tile][n_tile][i];
          if (bias_ptr) {
            res += (float)bias_ptr[col];
          }

          C[(size_t)row_global * N + col] = res;
        }
      }
    }
  }
}
