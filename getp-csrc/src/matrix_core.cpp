#include "../include/tensor.hpp"
#include <hip/hip_runtime.h>
#include <cassert>

using bf16 = hip_bfloat16;
using f32x4 = float __attribute__((ext_vector_type(4)));
using bf16_isa = __bf16;
using bf16x4 = bf16_isa __attribute__((__vector_size__(4 * sizeof(bf16_isa))));
using bf16x8 = bf16_isa __attribute__((__vector_size__(8 * sizeof(bf16_isa))));
using i32x4 = int32_t __attribute__((ext_vector_type(4)));

#define BF16_MFMA_KSTEP 16
#define MFMA_BF16_16x16(a, b, c) __builtin_amdgcn_mfma_f32_16x16x16bf16_1k((a), (b), (c), 0, 0, 0)

// Buffer load intrinsics
__device__ uint8_t llvm_amdgcn_raw_buffer_load_b8(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i8");

__device__ uint16_t llvm_amdgcn_raw_buffer_load_b16(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i16");

__device__ uint32_t llvm_amdgcn_raw_buffer_load_b32(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ uint64_t llvm_amdgcn_raw_buffer_load_b64(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i64");

__device__ __uint128_t llvm_amdgcn_raw_buffer_load_b128(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i128");

enum class coherency {
    cache_all = 0,
    cache_global = 1,
    cache_stream = 2,
    non_temporal = 3
};

template <typename T>
struct buffer {
    struct buffer_resource {
        const void* ptr;
        uint32_t range;
        uint32_t config;
    };

    using elem_type = std::remove_cv_t<T>;
    i32x4 srsrc;

    __device__ __forceinline__
    buffer(const T* pointer, size_t size) {
        buffer_resource res{
            reinterpret_cast<const void*>(pointer),
            static_cast<uint32_t>(size * sizeof(elem_type)),
            0x00020000U  // DATA_FORMAT = 32 bit
        };
        i32x4 raw = __builtin_bit_cast(i32x4, res);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            raw[i] = __builtin_amdgcn_readfirstlane(raw[i]);
        srsrc = raw;
    }

    template<size_t N, coherency c = coherency::cache_all>
    __device__ __forceinline__
    std::array<elem_type, N> load(
        uint32_t wave_offset,
        uint32_t thread_offset,
        bool in_bounds = true
    ) const {
        using result_type = std::array<elem_type, N>;
        constexpr const int load_size = sizeof(elem_type) * N;

        wave_offset *= sizeof(elem_type);
        thread_offset *= sizeof(elem_type);
        thread_offset = in_bounds ? thread_offset : 0xFFFF'FFFF;
        const int cc = static_cast<int>(c);

        i32x4 resource = srsrc;

        if constexpr (load_size == 1) {
            return __builtin_bit_cast(result_type, llvm_amdgcn_raw_buffer_load_b8(
                resource,
                thread_offset,
                wave_offset,
                cc
            ));
        } else if constexpr (load_size == 2) {
            return __builtin_bit_cast(result_type, llvm_amdgcn_raw_buffer_load_b16(
                resource,
                thread_offset,
                wave_offset,
                cc
            ));
        } else if constexpr (load_size == 4) {
            return __builtin_bit_cast(result_type, llvm_amdgcn_raw_buffer_load_b32(
                resource,
                thread_offset,
                wave_offset,
                cc
            ));
        } else if constexpr (load_size == 8) {
            return __builtin_bit_cast(result_type, llvm_amdgcn_raw_buffer_load_b64(
                resource,
                thread_offset,
                wave_offset,
                cc
            ));
        } else if constexpr (load_size == 16) {
            return __builtin_bit_cast(result_type, llvm_amdgcn_raw_buffer_load_b128(
                resource,
                thread_offset,
                wave_offset,
                cc
            ));
        } else {
            static_assert(load_size == -1, "load size not implemented");
        }
    }
};

__device__ __forceinline__ bf16x4 pack_f4_to_bf16x4(const float4 &v) {
  bf16x4 r;
  r[0] = (bf16_isa)v.x;
  r[1] = (bf16_isa)v.y;
  r[2] = (bf16_isa)v.z;
  r[3] = (bf16_isa)v.w;
  return r;
}

template <int BM, int BN, int BK, int TM, int TN, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS) void gemm_mfma_v2(const float *__restrict__ A,
                                                              const bf16 *__restrict__ B,
                                                              float *__restrict__ C,
                                                              const bf16 *__restrict__ bias, int M,
                                                              int N, int K) {
  constexpr int WM = 16, WN = 16, WK = 16;
  constexpr int VEC_B_SIZE = 8;
  constexpr int VEC_A_SIZE = 4;

  const int tid = threadIdx.x;
  const int wave_id = tid >> 6;
  const int lane_id = tid & 63;

  const int block_row_start = blockIdx.y * BM;
  const int block_col_start = blockIdx.x * BN;

  const int waves_per_block_m = BM / TM;
  const int waves_per_block_n = BN / TN;
  const int wave_row = wave_id / waves_per_block_n;
  const int wave_col = wave_id % waves_per_block_n;

  const int wave_row_start = block_row_start + wave_row * TM;
  const int wave_col_start = block_col_start + wave_col * TN;

  __shared__ __bf16 As[BM][BK + 2];
  __shared__ __bf16 Bs[BK][BN + 2];

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
  constexpr int B_VEC_ELEMS = B_ELEMS_TILE / VEC_B_SIZE;
  constexpr int B_VEC_PER_THR = B_VEC_ELEMS / BLOCK_THREADS;

  const int lx = lane_id & 15;
  const int ly = lane_id >> 4;

#pragma unroll
  for (int i = 0; i < A_VEC_PER_THR; i++) {
    const int vec_idx = tid + i * BLOCK_THREADS;
    const int elem_idx = vec_idx * VEC_A_SIZE;
    const int r = elem_idx / BK;
    const int c = elem_idx % BK;
    const int g_row = block_row_start + r;
    const int g_col = 0 + c;

   if (g_row < M && (g_col + VEC_A_SIZE - 1) < K) {
      const float4 v = *reinterpret_cast<const float4 *>(&A[(size_t)g_row * K + g_col]);
      const bf16x4 p = pack_f4_to_bf16x4(v);
      As[r][c + 0] = p[0];
      As[r][c + 1] = p[1];
      As[r][c + 2] = p[2];
      As[r][c + 3] = p[3];
    } 
     else {
 #pragma unroll
       for (int j = 0; j < VEC_A_SIZE; ++j) {
         const float v = (g_row < M && (g_col + j) < K) ? A[(size_t)g_row * K + (g_col + j)] : 0.0f;
         As[r][c + j] = (__bf16)v;
       }
     }
  }

#pragma unroll
  for (int i = 0; i < B_VEC_PER_THR; ++i) {
    const int vec_idx = tid + i * BLOCK_THREADS;
    const int elem_idx = vec_idx * VEC_B_SIZE;
    const int r = elem_idx / BN;
    const int c = elem_idx % BN;
    const int g_row = 0 + r;
    const int g_col = block_col_start + c;

   if (g_row < K && (g_col + VEC_B_SIZE - 1) < N) {
      *reinterpret_cast<uint4 *>(&Bs[r][c]) =
        *reinterpret_cast<const uint4 *>(&B[(size_t)g_row * N + g_col]);
    } 
     else {
       uint4 z = {0, 0, 0, 0};
       *reinterpret_cast<uint4 *>(&Bs[r][c]) = z;
     }
  }
  __syncthreads();

  for (int k_base = 0; k_base < K; k_base += BK) {
    float4 regA[A_VEC_PER_THR];
    uint4 regB[B_VEC_PER_THR];
    const bool has_next = (k_base + BK) < K;

    if (has_next) {
#pragma unroll
      for (int i = 0; i < A_VEC_PER_THR; i++) {
        const int vec_idx = tid + i * BLOCK_THREADS;
        const int elem_idx = vec_idx * VEC_A_SIZE;
        const int r = elem_idx / BK;
        const int c = elem_idx % BK;
        const int g_row = block_row_start + r;
        const int g_col = (k_base + BK) + c;

       if (g_row < M && (g_col + VEC_A_SIZE - 1) < K) {
          regA[i] = *reinterpret_cast<const float4 *>(&A[(size_t)g_row * K + g_col]);
       } else {
           float tmp[4] = {0, 0, 0, 0};
 #pragma unroll
           for (int j = 0; j < 4; ++j)
             if (g_row < M && (g_col + j) < K)
               tmp[j] = A[(size_t)g_row * K + (g_col + j)];
           regA[i] = *reinterpret_cast<float4 *>(tmp);
         }
      }

#pragma unroll
      for (int i = 0; i < B_VEC_PER_THR; ++i) {
        const int vec_idx = tid + i * BLOCK_THREADS;
        const int elem_idx = vec_idx * VEC_B_SIZE;
        const int r = elem_idx / BN;
        const int c = elem_idx % BN;
        const int g_row = (k_base + BK) + r;
        const int g_col = block_col_start + c;

        if (g_row < K && (g_col + VEC_B_SIZE - 1) < N) {
          regB[i] = *reinterpret_cast<const uint4 *>(&B[(size_t)g_row * N + g_col]);
        } 
         else {
           regB[i] = uint4{0, 0, 0, 0};
         }
      }
    }

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

    __syncthreads();

    if (has_next) {
#pragma unroll
      for (int i = 0; i < A_VEC_PER_THR; i++) {
        const int elem_idx = (tid + i * BLOCK_THREADS) * VEC_A_SIZE;
        const int r = elem_idx / BK;
        const int c = elem_idx % BK;
        const bf16x4 p = pack_f4_to_bf16x4(regA[i]);
        As[r][c + 0] = p[0];
        As[r][c + 1] = p[1];
        As[r][c + 2] = p[2];
        As[r][c + 3] = p[3];
      }

#pragma unroll
      for (int i = 0; i < B_VEC_PER_THR; ++i) {
        const int elem_idx = (tid + i * BLOCK_THREADS) * VEC_B_SIZE;
        const int r = elem_idx / BN;
        const int c = elem_idx % BN;
        *reinterpret_cast<uint4 *>(&Bs[r][c]) = regB[i];
      }

      __syncthreads();
    }
  }

#pragma unroll
  for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
    for (int nt = 0; nt < N_TILES; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = wave_row_start + mt * WM + (i + 4 * ly);
        const int col = wave_col_start + nt * WN + lx;
        if (row < M && col < N) {
          float out = acc[mt][nt][i];
          if (bias)
            out += (float)bias[col];
          C[(size_t)row * N + col] = out;
        }
      }
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS) void gemm_mfma_moe_v2(
  const float *__restrict__ A, const bf16 *__restrict__ B, float *__restrict__ C,
  const bf16 *__restrict__ bias, const int *__restrict__ expert_offsets, int M_total, int N,
  int K) {
  constexpr int WM = 16, WN = 16, WK = 16;
  constexpr int VEC_B_SIZE = 8;
  constexpr int VEC_A_SIZE = 4;

  const int expert_id = blockIdx.z;
  const int row_begin = expert_offsets[expert_id];
  const int row_end = expert_offsets[expert_id + 1];
  const int M = (row_end - row_begin > 0) ? (row_end - row_begin) : 0;
  if (M <= 0)
    return;

  const bf16 *__restrict__ B_ptr = B + (size_t)expert_id * N * K;
  const bf16 *__restrict__ bias_ptr = bias ? (bias + (size_t)expert_id * N) : nullptr;

  const int tid = threadIdx.x;
  const int wave_id = tid >> 6;
  const int lane_id = tid & 63;

  const int block_row_local_start = blockIdx.y * BM;
  const int block_col_start = blockIdx.x * BN;
  if (block_row_local_start >= M)
    return;

  const int waves_per_block_m = BM / TM;
  const int waves_per_block_n = BN / TN;
  const int wave_row = wave_id / waves_per_block_n;
  const int wave_col = wave_id % waves_per_block_n;

  const int wave_row_start = block_row_local_start + wave_row * TM;
  const int wave_col_start = block_col_start + wave_col * TN;

  const int block_row_global_start = row_begin + block_row_local_start;

  __shared__ __bf16 As[BM][BK + 2];
  __shared__ __bf16 Bs[BK][BN + 2];

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
  constexpr int B_VEC_ELEMS = B_ELEMS_TILE / VEC_B_SIZE;
  constexpr int B_VEC_PER_THR = B_VEC_ELEMS / BLOCK_THREADS;

  const int lx = lane_id & 15;
  const int ly = lane_id >> 4;

#pragma unroll
  for (int i = 0; i < A_VEC_PER_THR; i++) {
    const int vec_idx = tid + i * BLOCK_THREADS;
    const int elem_idx = vec_idx * VEC_A_SIZE;
    const int r_local = elem_idx / BK;
    const int c = elem_idx % BK;
    const int g_row_local = r_local;
    const int g_row_global = block_row_global_start + g_row_local;
    const int g_col = 0 + c;

    if ((block_row_local_start + g_row_local) < M && (g_col + VEC_A_SIZE - 1) < K) {
      const float4 v = *reinterpret_cast<const float4 *>(&A[(size_t)g_row_global * K + g_col]);
      const bf16x4 p = pack_f4_to_bf16x4(v);
      As[r_local][c + 0] = p[0];
      As[r_local][c + 1] = p[1];
      As[r_local][c + 2] = p[2];
      As[r_local][c + 3] = p[3];
    } 
     else {
 #pragma unroll
       for (int j = 0; j < VEC_A_SIZE; j++) {
         const float v = ((block_row_local_start + g_row_local) < M && (g_col + j) < K)
                           ? A[(size_t)g_row_global * K + (g_col + j)]
                           : 0.0f;
         As[r_local][c + j] = (__bf16)v;
       }
    }
  }

#pragma unroll
  for (int i = 0; i < B_VEC_PER_THR; ++i) {
    const int vec_idx = tid + i * BLOCK_THREADS;
    const int elem_idx = vec_idx * VEC_B_SIZE;
    const int r = elem_idx / BN;
    const int c = elem_idx % BN;
    const int g_row = 0 + r;
    const int g_col = block_col_start + c;

    if (g_row < K && (g_col + VEC_B_SIZE - 1) < N) {
      *reinterpret_cast<uint4 *>(&Bs[r][c]) =
        *reinterpret_cast<const uint4 *>(&B_ptr[(size_t)g_row * N + g_col]);
    } 
     else {
       uint4 z = {0, 0, 0, 0};
       *reinterpret_cast<uint4 *>(&Bs[r][c]) = z;
     }
  }
  __syncthreads();

  for (int k_base = 0; k_base < K; k_base += BK) {
    float4 regA[A_VEC_PER_THR];
    uint4 regB[B_VEC_PER_THR];
    const bool has_next = (k_base + BK) < K;

    if (has_next) {
#pragma unroll
      for (int i = 0; i < A_VEC_PER_THR; i++) {
        const int vec_idx = tid + i * BLOCK_THREADS;
        const int elem_idx = vec_idx * VEC_A_SIZE;
        const int r_local = elem_idx / BK;
        const int c = elem_idx % BK;
        const int g_row_local = r_local;
        const int g_row_global = block_row_global_start + g_row_local;
        const int g_col = (k_base + BK) + c;

        if ((block_row_local_start + g_row_local) < M && (g_col + VEC_A_SIZE - 1) < K) {
          regA[i] = *reinterpret_cast<const float4 *>(&A[(size_t)g_row_global * K + g_col]);
        } 
         else {
           float tmp[4] = {0, 0, 0, 0};
 #pragma unroll
           for (int j = 0; j < 4; ++j)
             if ((block_row_local_start + g_row_local) < M && (g_col + j) < K)
               tmp[j] = A[(size_t)g_row_global * K + (g_col + j)];
           regA[i] = *reinterpret_cast<float4 *>(tmp);
         }
      }

#pragma unroll
      for (int i = 0; i < B_VEC_PER_THR; ++i) {
        const int vec_idx = tid + i * BLOCK_THREADS;
        const int elem_idx = vec_idx * VEC_B_SIZE;
        const int r = elem_idx / BN;
        const int c = elem_idx % BN;
        const int g_row = (k_base + BK) + r;
        const int g_col = block_col_start + c;

       if (g_row < K && (g_col + VEC_B_SIZE - 1) < N) {
          regB[i] = *reinterpret_cast<const uint4 *>(&B_ptr[(size_t)g_row * N + g_col]);
        } 
         else {
           regB[i] = uint4{0, 0, 0, 0};
         }
      }
    }

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

    __syncthreads();

    if (has_next) {
#pragma unroll
      for (int i = 0; i < A_VEC_PER_THR; i++) {
        const int elem_idx = (tid + i * BLOCK_THREADS) * VEC_A_SIZE;
        const int r_local = elem_idx / BK;
        const int c = elem_idx % BK;
        const bf16x4 p = pack_f4_to_bf16x4(regA[i]);
        As[r_local][c + 0] = p[0];
        As[r_local][c + 1] = p[1];
        As[r_local][c + 2] = p[2];
        As[r_local][c + 3] = p[3];
      }

#pragma unroll
      for (int i = 0; i < B_VEC_PER_THR; ++i) {
        const int elem_idx = (tid + i * BLOCK_THREADS) * VEC_B_SIZE;
        const int r = elem_idx / BN;
        const int c = elem_idx % BN;
        *reinterpret_cast<uint4 *>(&Bs[r][c]) = regB[i];
      }

      __syncthreads();
    }
  }

#pragma unroll
  for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
    for (int nt = 0; nt < N_TILES; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row_local = wave_row_start + mt * WM + (i + 4 * ly);
        const int row_global = row_begin + row_local;
        const int col = wave_col_start + nt * WN + lx;
        if (row_local < M && col < N) {
          float out = acc[mt][nt][i];
          if (bias_ptr)
            out += (float)bias_ptr[col];
          C[(size_t)row_global * N + col] = out;
        }
      }
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS) void gemm_mfma_moe_fused(
  const float *__restrict__ A, const bf16 *__restrict__ B, float *__restrict__ C,
  const bf16 *__restrict__ bias, const int *__restrict__ expert_offsets, int M_total, int N, int K,
  float clamp_limit) {
  constexpr int WM = 16, WN = 16, WK = 16;
  constexpr int VEC_B_SIZE = 8;
  constexpr int VEC_A_SIZE = 4;

  const int expert_id = blockIdx.z;
  const int row_begin = expert_offsets[expert_id];
  const int row_end = expert_offsets[expert_id + 1];
  const int M = (row_end - row_begin > 0) ? (row_end - row_begin) : 0;
  if (M <= 0)
    return;

  const bf16 *__restrict__ B_ptr = B + (size_t)expert_id * N * K;
  const bf16 *__restrict__ bias_ptr = bias ? (bias + (size_t)expert_id * N) : nullptr;

  const int tid = threadIdx.x;
  const int wave_id = tid >> 6;
  const int lane_id = tid & 63;

  const int block_row_local_start = blockIdx.y * BM;
  const int block_col_start = blockIdx.x * BN;
  if (block_row_local_start >= M)
    return;

  const int waves_per_block_m = BM / TM;
  const int waves_per_block_n = BN / TN;
  const int wave_row = wave_id / waves_per_block_n;
  const int wave_col = wave_id % waves_per_block_n;

  const int wave_row_start = block_row_local_start + wave_row * TM;
  const int wave_col_start = block_col_start + wave_col * TN;

  const int block_row_global_start = row_begin + block_row_local_start;

  __shared__ __bf16 As[BM][BK + 2];
  __shared__ __bf16 Bs[BK][BN + 2];

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
  constexpr int B_VEC_ELEMS = B_ELEMS_TILE / VEC_B_SIZE;
  constexpr int B_VEC_PER_THR = B_VEC_ELEMS / BLOCK_THREADS;

  const int lx = lane_id & 15;
  const int ly = lane_id >> 4;

#pragma unroll
  for (int i = 0; i < A_VEC_PER_THR; i++) {
    const int vec_idx = tid + i * BLOCK_THREADS;
    const int elem_idx = vec_idx * VEC_A_SIZE;
    const int r_local = elem_idx / BK;
    const int c = elem_idx % BK;
    const int g_row_local = r_local;
    const int g_row_global = block_row_global_start + g_row_local;
    const int g_col = 0 + c;

    if ((block_row_local_start + g_row_local) < M && (g_col + VEC_A_SIZE - 1) < K) {
      const float4 v = *reinterpret_cast<const float4 *>(&A[(size_t)g_row_global * K + g_col]);
      const bf16x4 p = pack_f4_to_bf16x4(v);
      As[r_local][c + 0] = p[0];
      As[r_local][c + 1] = p[1];
      As[r_local][c + 2] = p[2];
      As[r_local][c + 3] = p[3];
    } else {
#pragma unroll
      for (int j = 0; j < VEC_A_SIZE; j++) {
        const float v = ((block_row_local_start + g_row_local) < M && (g_col + j) < K)
                          ? A[(size_t)g_row_global * K + (g_col + j)]
                          : 0.0f;
        As[r_local][c + j] = (__bf16)v;
      }
    }
  }

#pragma unroll
  for (int i = 0; i < B_VEC_PER_THR; ++i) {
    const int vec_idx = tid + i * BLOCK_THREADS;
    const int elem_idx = vec_idx * VEC_B_SIZE;
    const int r = elem_idx / BN;
    const int c = elem_idx % BN;
    const int g_row = 0 + r;
    const int g_col = block_col_start + c;

    if (g_row < K && (g_col + VEC_B_SIZE - 1) < N) {
      *reinterpret_cast<uint4 *>(&Bs[r][c]) =
        *reinterpret_cast<const uint4 *>(&B_ptr[(size_t)g_row * N + g_col]);
    } else {
      uint4 z = {0, 0, 0, 0};
      *reinterpret_cast<uint4 *>(&Bs[r][c]) = z;
    }
  }
  __syncthreads();

  for (int k_base = 0; k_base < K; k_base += BK) {
    float4 regA[A_VEC_PER_THR];
    uint4 regB[B_VEC_PER_THR];
    const bool has_next = (k_base + BK) < K;

    if (has_next) {
#pragma unroll
      for (int i = 0; i < A_VEC_PER_THR; i++) {
        const int vec_idx = tid + i * BLOCK_THREADS;
        const int elem_idx = vec_idx * VEC_A_SIZE;
        const int r_local = elem_idx / BK;
        const int c = elem_idx % BK;
        const int g_row_local = r_local;
        const int g_row_global = block_row_global_start + g_row_local;
        const int g_col = (k_base + BK) + c;

        if ((block_row_local_start + g_row_local) < M && (g_col + VEC_A_SIZE - 1) < K) {
          regA[i] = *reinterpret_cast<const float4 *>(&A[(size_t)g_row_global * K + g_col]);
        } else {
          float tmp[4] = {0, 0, 0, 0};
#pragma unroll
          for (int j = 0; j < 4; ++j)
            if ((block_row_local_start + g_row_local) < M && (g_col + j) < K)
              tmp[j] = A[(size_t)g_row_global * K + (g_col + j)];
          regA[i] = *reinterpret_cast<float4 *>(tmp);
        }
      }

#pragma unroll
      for (int i = 0; i < B_VEC_PER_THR; ++i) {
        const int vec_idx = tid + i * BLOCK_THREADS;
        const int elem_idx = vec_idx * VEC_B_SIZE;
        const int r = elem_idx / BN;
        const int c = elem_idx % BN;
        const int g_row = (k_base + BK) + r;
        const int g_col = block_col_start + c;

        if (g_row < K && (g_col + VEC_B_SIZE - 1) < N) {
          regB[i] = *reinterpret_cast<const uint4 *>(&B_ptr[(size_t)g_row * N + g_col]);
        } else {
          regB[i] = uint4{0, 0, 0, 0};
        }
      }
    }

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

    __syncthreads();

    if (has_next) {
#pragma unroll
      for (int i = 0; i < A_VEC_PER_THR; i++) {
        const int elem_idx = (tid + i * BLOCK_THREADS) * VEC_A_SIZE;
        const int r_local = elem_idx / BK;
        const int c = elem_idx % BK;
        const bf16x4 p = pack_f4_to_bf16x4(regA[i]);
        As[r_local][c + 0] = p[0];
        As[r_local][c + 1] = p[1];
        As[r_local][c + 2] = p[2];
        As[r_local][c + 3] = p[3];
      }

#pragma unroll
      for (int i = 0; i < B_VEC_PER_THR; ++i) {
        const int elem_idx = (tid + i * BLOCK_THREADS) * VEC_B_SIZE;
        const int r = elem_idx / BN;
        const int c = elem_idx % BN;
        *reinterpret_cast<uint4 *>(&Bs[r][c]) = regB[i];
      }

      __syncthreads();
    }
  }

  __shared__ float C_tile[BM][BN];

#pragma unroll
  for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
    for (int nt = 0; nt < N_TILES; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row_local_sm = wave_row * TM + mt * WM + (i + 4 * ly);
        const int col_sm = wave_col * TN + nt * WN + lx;
        const int col_global = block_col_start + col_sm;
        const int row_global = block_row_global_start + row_local_sm;
        if (row_local_sm < BM && col_sm < BN && (block_row_local_start + row_local_sm) < M &&
            col_global < N) {
          float out = acc[mt][nt][i];
          if (bias_ptr)
            out += (float)bias_ptr[col_global];
          C_tile[row_local_sm][col_sm] = out;
        }
      }
    }
  }
  __syncthreads();

  const int H = N >> 1;
  const int m_valid = min(BM, M - block_row_local_start);
  const int n_valid_pairs = min(BN, N - block_col_start) >> 1;

  for (int linear = threadIdx.x; linear < m_valid * n_valid_pairs; linear += BLOCK_THREADS) {
    const int row_local = linear / n_valid_pairs;
    const int pair_idx = linear % n_valid_pairs;

    const int col_gate_local = (pair_idx << 1);
    const int col_up_local = col_gate_local + 1;

    const int row_global = row_begin + block_row_local_start + row_local;
    const int out_col_global = (block_col_start >> 1) + pair_idx;

    float g = C_tile[row_local][col_gate_local];
    float u = C_tile[row_local][col_up_local];

    g = fminf(g, clamp_limit);
    u = fminf(fmaxf(u, -clamp_limit), clamp_limit);

    constexpr float alpha = 1.702f;
    const float silu = g * (1.f / (1.f + expf(-alpha * g)));
    const float outv = silu * (u + 1.f);

    C[(size_t)row_global * H + out_col_global] = outv;
  }
}

template <int BM, int BN, int BK, int TM, int TN, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS) void dangerous_moe(
  const float *__restrict__ A, const bf16 *__restrict__ B, float *__restrict__ C,
  const bf16 *__restrict__ bias, const int *__restrict__ expert_offsets, int M_total, int N,
  int K) {
  constexpr int WM = 16, WN = 16, WK = 16;
  constexpr int VEC_B_SIZE = 8;
  constexpr int VEC_A_SIZE = 4;

  const int expert_id = blockIdx.z;
  const int row_begin = expert_offsets[expert_id];
  const int row_end = expert_offsets[expert_id + 1];
  const int M = (row_end - row_begin > 0) ? (row_end - row_begin) : 0;
  if (M <= 0)
    return;

  const bf16 *__restrict__ B_ptr = B + (size_t)expert_id * N * K;
  const bf16 *__restrict__ bias_ptr = bias ? (bias + (size_t)expert_id * N) : nullptr;

  // Create buffer objects
  buffer<const float> A_buf(A, static_cast<size_t>(M_total) * K);
  buffer<const bf16> B_buf(B_ptr, static_cast<size_t>(K) * N);

  const int tid = threadIdx.x;
  const int wave_id = tid >> 6;
  const int lane_id = tid & 63;

  const int block_row_local_start = blockIdx.y * BM;
  const int block_col_start = blockIdx.x * BN;
  if (block_row_local_start >= M)
    return;

  const int waves_per_block_m = BM / TM;
  const int waves_per_block_n = BN / TN;
  const int wave_row = wave_id / waves_per_block_n;
  const int wave_col = wave_id % waves_per_block_n;

  const int wave_row_start = block_row_local_start + wave_row * TM;
  const int wave_col_start = block_col_start + wave_col * TN;

  const int block_row_global_start = row_begin + block_row_local_start;

  __shared__ __bf16 As[BM][BK + 2];
  __shared__ __bf16 Bs[BK][BN + 2];

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
  constexpr int B_VEC_ELEMS = B_ELEMS_TILE / VEC_B_SIZE;
  constexpr int B_VEC_PER_THR = B_VEC_ELEMS / BLOCK_THREADS;

  const int lx = lane_id & 15;
  const int ly = lane_id >> 4;

  // Initial load of A tile
#pragma unroll
  for (int i = 0; i < A_VEC_PER_THR; i++) {
    const int vec_idx = tid + i * BLOCK_THREADS;
    const int elem_idx = vec_idx * VEC_A_SIZE;
    const int r_local = elem_idx / BK;
    const int c = elem_idx % BK;
    const int g_row_local = r_local;
    const int g_row_global = block_row_global_start + g_row_local;
    const int g_col = 0 + c;

    const bool in_bounds = (block_row_local_start + g_row_local) < M && (g_col + VEC_A_SIZE - 1) < K;
    auto vals = A_buf.template load<VEC_A_SIZE>(0, static_cast<uint32_t>((size_t)g_row_global * K + g_col), in_bounds);
    const float4 v = __builtin_bit_cast(float4, vals);
    const bf16x4 p = pack_f4_to_bf16x4(v);
    *reinterpret_cast<bf16x4 *>(&As[r_local][c]) = p;
  }

  // Initial load of B tile
#pragma unroll
  for (int i = 0; i < B_VEC_PER_THR; ++i) {
    const int vec_idx = tid + i * BLOCK_THREADS;
    const int elem_idx = vec_idx * VEC_B_SIZE;
    const int r = elem_idx / BN;
    const int c = elem_idx % BN;
    const int g_row = 0 + r;
    const int g_col = block_col_start + c;

    const bool in_bounds = g_row < K && (g_col + VEC_B_SIZE - 1) < N;
    auto vals = B_buf.template load<VEC_B_SIZE>(0, static_cast<uint32_t>((size_t)g_row * N + g_col), in_bounds);
    const bf16x8 packed = __builtin_bit_cast(bf16x8, vals);
    *reinterpret_cast<bf16x8 *>(&Bs[r][c]) = packed;
  }
  __syncthreads();

  // Main loop
  for (int k_base = 0; k_base < K; k_base += BK) {
    float4 regA[A_VEC_PER_THR];
    bf16x8 regB[B_VEC_PER_THR];
    const bool has_next = (k_base + BK) < K;

    // Prefetch next tiles
    if (has_next) {
#pragma unroll
      for (int i = 0; i < A_VEC_PER_THR; i++) {
        const int vec_idx = tid + i * BLOCK_THREADS;
        const int elem_idx = vec_idx * VEC_A_SIZE;
        const int r_local = elem_idx / BK;
        const int c = elem_idx % BK;
        const int g_row_local = r_local;
        const int g_row_global = block_row_global_start + g_row_local;
        const int g_col = (k_base + BK) + c;

        const bool in_bounds = (block_row_local_start + g_row_local) < M && (g_col + VEC_A_SIZE - 1) < K;
        auto vals = A_buf.template load<VEC_A_SIZE>(0, static_cast<uint32_t>((size_t)g_row_global * K + g_col), in_bounds);
        regA[i] = __builtin_bit_cast(float4, vals);
      }

#pragma unroll
      for (int i = 0; i < B_VEC_PER_THR; ++i) {
        const int vec_idx = tid + i * BLOCK_THREADS;
        const int elem_idx = vec_idx * VEC_B_SIZE;
        const int r = elem_idx / BN;
        const int c = elem_idx % BN;
        const int g_row = (k_base + BK) + r;
        const int g_col = block_col_start + c;

        const bool in_bounds = g_row < K && (g_col + VEC_B_SIZE - 1) < N;
        auto vals = B_buf.template load<VEC_B_SIZE>(0, static_cast<uint32_t>((size_t)g_row * N + g_col), in_bounds);
        regB[i] = __builtin_bit_cast(bf16x8, vals);
      }
    }

    // Compute with current tiles
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

    __syncthreads();

    // Write prefetched tiles to shared memory
    if (has_next) {
#pragma unroll
      for (int i = 0; i < A_VEC_PER_THR; i++) {
        const int elem_idx = (tid + i * BLOCK_THREADS) * VEC_A_SIZE;
        const int r_local = elem_idx / BK;
        const int c = elem_idx % BK;
        const bf16x4 p = pack_f4_to_bf16x4(regA[i]);
        *reinterpret_cast<bf16x4 *>(&As[r_local][c]) = p;
      }

#pragma unroll
      for (int i = 0; i < B_VEC_PER_THR; ++i) {
        const int elem_idx = (tid + i * BLOCK_THREADS) * VEC_B_SIZE;
        const int r = elem_idx / BN;
        const int c = elem_idx % BN;
        *reinterpret_cast<bf16x8 *>(&Bs[r][c]) = regB[i];
      }

      __syncthreads();
    }
  }

  // Write results
#pragma unroll
  for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
    for (int nt = 0; nt < N_TILES; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row_local = wave_row_start + mt * WM + (i + 4 * ly);
        const int row_global = row_begin + row_local;
        const int col = wave_col_start + nt * WN + lx;
        if (row_local < M && col < N) {
          float out = acc[mt][nt][i];
          if (bias_ptr)
            out += (float)bias_ptr[col];
          C[(size_t)row_global * N + col] = out;
        }
      }
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS) void dangerous_gemm(const float *__restrict__ A,
                                                              const bf16 *__restrict__ B,
                                                              float *__restrict__ C,
                                                              const bf16 *__restrict__ bias, int M,
                                                              int N, int K) {

  constexpr int WM = 16, WN = 16, WK = 16;
  constexpr int VEC_B_SIZE = 8;
  constexpr int VEC_A_SIZE = 4;

  buffer<const float> A_buf(A, static_cast<size_t>(M) * K);
  buffer<const bf16> B_buf(B, static_cast<size_t>(K) * N);

  const int tid = threadIdx.x;
  const int wave_id = tid >> 6;
  const int lane_id = tid & 63;

  const int block_row_start = blockIdx.y * BM;
  const int block_col_start = blockIdx.x * BN;

  const int waves_per_block_m = BM / TM;
  const int waves_per_block_n = BN / TN;
  const int wave_row = wave_id / waves_per_block_n;
  const int wave_col = wave_id % waves_per_block_n;

  const int wave_row_start = block_row_start + wave_row * TM;
  const int wave_col_start = block_col_start + wave_col * TN;

  __shared__ __bf16 As[BM][BK + 2];
  __shared__ __bf16 Bs[BK][BN + 2];

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
  constexpr int B_VEC_ELEMS = B_ELEMS_TILE / VEC_B_SIZE;
  constexpr int B_VEC_PER_THR = B_VEC_ELEMS / BLOCK_THREADS;

  const int lx = lane_id & 15;
  const int ly = lane_id >> 4;

#pragma unroll
  for (int i = 0; i < A_VEC_PER_THR; i++) {
    const int vec_idx = tid + i * BLOCK_THREADS;
    const int elem_idx = vec_idx * VEC_A_SIZE;
    const int r = elem_idx / BK;
    const int c = elem_idx % BK;
    const int g_row = block_row_start + r;
    const int g_col = 0 + c;

      auto vals = A_buf.template load<VEC_A_SIZE>(0, static_cast<uint32_t>((size_t)g_row * K + g_col));
      const float4 v = __builtin_bit_cast(float4, vals);
      const bf16x4 p = pack_f4_to_bf16x4(v);
      *reinterpret_cast<bf16x4 *>(&As[r][c]) = p;
  }

#pragma unroll
  for (int i = 0; i < B_VEC_PER_THR; ++i) {
    const int vec_idx = tid + i * BLOCK_THREADS;
    const int elem_idx = vec_idx * VEC_B_SIZE;
    const int r = elem_idx / BN;
    const int c = elem_idx % BN;
    const int g_row = 0 + r;
    const int g_col = block_col_start + c;

      auto vals = B_buf.template load<VEC_B_SIZE>(0, static_cast<uint32_t>((size_t)g_row * N + g_col));
      const bf16x8 packed = __builtin_bit_cast(bf16x8, vals);
      *reinterpret_cast<bf16x8 *>(&Bs[r][c]) = packed;
  }
  __syncthreads();

  for (int k_base = 0; k_base < K; k_base += BK) {
    float4 regA[A_VEC_PER_THR];
    bf16x8 regB[B_VEC_PER_THR];
    const bool has_next = (k_base + BK) < K;

    if (has_next) {
#pragma unroll
      for (int i = 0; i < A_VEC_PER_THR; i++) {
        const int vec_idx = tid + i * BLOCK_THREADS;
        const int elem_idx = vec_idx * VEC_A_SIZE;
        const int r = elem_idx / BK;
        const int c = elem_idx % BK;
        const int g_row = block_row_start + r;
        const int g_col = (k_base + BK) + c;

          auto vals = A_buf.template load<VEC_A_SIZE>(0, static_cast<uint32_t>((size_t)g_row * K + g_col));
          regA[i] = __builtin_bit_cast(float4, vals);
      }

#pragma unroll
      for (int i = 0; i < B_VEC_PER_THR; ++i) {
        const int vec_idx = tid + i * BLOCK_THREADS;
        const int elem_idx = vec_idx * VEC_B_SIZE;
        const int r = elem_idx / BN;
        const int c = elem_idx % BN;
        const int g_row = (k_base + BK) + r;
        const int g_col = block_col_start + c;

          auto vals = B_buf.template load<VEC_B_SIZE>(0, static_cast<uint32_t>((size_t)g_row * N + g_col));
          regB[i] = __builtin_bit_cast(bf16x8, vals);
      }
    }

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

    __syncthreads();

    if (has_next) {
#pragma unroll
      for (int i = 0; i < A_VEC_PER_THR; i++) {
        const int elem_idx = (tid + i * BLOCK_THREADS) * VEC_A_SIZE;
        const int r = elem_idx / BK;
        const int c = elem_idx % BK;
        const bf16x4 p = pack_f4_to_bf16x4(regA[i]);
        As[r][c + 0] = p[0];
        As[r][c + 1] = p[1];
        As[r][c + 2] = p[2];
        As[r][c + 3] = p[3];
      }

#pragma unroll
      for (int i = 0; i < B_VEC_PER_THR; ++i) {
        const int elem_idx = (tid + i * BLOCK_THREADS) * VEC_B_SIZE;
        const int r = elem_idx / BN;
        const int c = elem_idx % BN;
        *reinterpret_cast<bf16x8 *>(&Bs[r][c]) = regB[i];
      }

      __syncthreads();
    }
  }

#pragma unroll
  for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
    for (int nt = 0; nt < N_TILES; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = wave_row_start + mt * WM + (i + 4 * ly);
        const int col = wave_col_start + nt * WN + lx;
        if (row < M && col < N) {
          float out = acc[mt][nt][i];
          if (bias)
            out += (float)bias[col];
          C[(size_t)row * N + col] = out;
        }
      }
    }
  }
}