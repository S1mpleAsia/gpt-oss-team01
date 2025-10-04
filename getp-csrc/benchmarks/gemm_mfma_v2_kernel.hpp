#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

using bf16 = hip_bfloat16;
using f32x4 = float __attribute__((ext_vector_type(4)));
using bf16_isa = __bf16;
using bf16x4 = bf16_isa __attribute__((__vector_size__(4 * sizeof(bf16_isa))));
static constexpr unsigned kWarpSize = 64;

// ===== MFMA (gfx90a) =====
#define BF16_MFMA_KSTEP 16
#define MFMA_BF16_16x16(a, b, c) __builtin_amdgcn_mfma_f32_16x16x16bf16_1k((a), (b), (c), 0, 0, 0)

// ===== RAW buffer LLVM intrinsics (petit-style) =====
typedef int v4i __attribute__((ext_vector_type(4)));

#if defined(__HIP_DEVICE_COMPILE__)
static __device__ __forceinline__ v4i llvm_amdgcn_raw_buffer_load_v4i32(
  v4i rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

static __device__ __forceinline__ void llvm_amdgcn_raw_buffer_store_v4i32(
  v4i data, v4i rsrc, int voffset, int soffset,
  int aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

static __device__ __forceinline__ void llvm_amdgcn_raw_buffer_store_f32(
  float data, v4i rsrc, int voffset, int soffset,
  int aux) __asm("llvm.amdgcn.raw.buffer.store.f32");
#endif

// Minimal buffer resource (SRD)
union BufferResource {
  static constexpr unsigned kDataFormatU32Config = 4u << 15;
  enum { kNone = 0, kGLCBit = 1 << 0, kSLCBit = 1 << 1 };

  v4i content;
  struct {
    uintptr_t ptr;   // base (64-bit)
    unsigned range;  // bytes (bounds)
    unsigned config;
  } v;

  __device__ __forceinline__ v4i LoadV4(int byte_offset, int aux = kNone) const {
#if defined(__HIP_DEVICE_COMPILE__)
    return llvm_amdgcn_raw_buffer_load_v4i32(content, byte_offset, 0, aux);
#else
    return (v4i){0, 0, 0, 0};
#endif
  }
  __device__ __forceinline__ void StoreV4(int byte_offset, v4i data, int aux = kNone) const {
#if defined(__HIP_DEVICE_COMPILE__)
    llvm_amdgcn_raw_buffer_store_v4i32(data, content, byte_offset, 0, aux);
#endif
  }
  __device__ __forceinline__ void StoreF32(int byte_offset, float x, int aux = kNone) const {
#if defined(__HIP_DEVICE_COMPILE__)
    llvm_amdgcn_raw_buffer_store_f32(x, content, byte_offset, 0, aux);
#endif
  }
};

// ===== Helpers =====
__device__ __forceinline__ bf16x4 pack_f4_to_bf16x4(const float4 &v) {
  bf16x4 r;
  r[0] = (bf16_isa)v.x;
  r[1] = (bf16_isa)v.y;
  r[2] = (bf16_isa)v.z;
  r[3] = (bf16_isa)v.w;
  return r;
}

// === LDS bank-conflict swizzle (set to 0 to disable); try 0/8/16 ===
#ifndef LDS_SWZ_MASK_A
#define LDS_SWZ_MASK_A 16
#endif
#ifndef LDS_SWZ_MASK_B
#define LDS_SWZ_MASK_B 16
#endif
__device__ __forceinline__ int lds_swz_a(int row, int col) {
  return col ^ ((row & 0x1) ? LDS_SWZ_MASK_A : 0);
}
__device__ __forceinline__ int lds_swz_b(int row, int col) {
  return col ^ ((row & 0x1) ? LDS_SWZ_MASK_B : 0);
}

// Toggle caching behavior (SLC) per operand
#ifndef AUX_A
#define AUX_A BufferResource::kNone
#endif
#ifndef AUX_B
#define AUX_B BufferResource::kSLCBit  // weights often benefit from L2
#endif
#ifndef AUX_C
#define AUX_C BufferResource::kNone
#endif

// ===== GEMM kernel =====
template <int BM, int BN, int BK, int TM, int TN, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS) void gemm_mfma_v2(
  const float *__restrict__ A,    // [M,K] fp32
  const bf16 *__restrict__ B,     // [K,N] bf16
  float *__restrict__ C,          // [M,N] fp32
  const bf16 *__restrict__ bias,  // [N] or nullptr
  int M, int N, int K) {
  static_assert(BK % BF16_MFMA_KSTEP == 0, "BK must be multiple of MFMA K-step");
  constexpr int WM = 16, WN = 16, WK = BF16_MFMA_KSTEP;
  constexpr int VEC_A_SIZE = 4;  // float4 (16B)
  constexpr int VEC_B_SIZE = 8;  // 8*bf16  (16B)

  const int tid = threadIdx.x;
  const int wave_id = tid / kWarpSize;
  const int lane_id = tid & (kWarpSize - 1);

  const int block_row_start = blockIdx.y * BM;
  const int block_col_start = blockIdx.x * BN;

  const int waves_per_block_m = BM / TM;
  const int waves_per_block_n = BN / TN;
  const int wave_row = wave_id / waves_per_block_n;
  const int wave_col = wave_id % waves_per_block_n;

  const int wave_row_start = block_row_start + wave_row * TM;
  const int wave_col_start = block_col_start + wave_col * TN;

  // double-buffered LDS (with +2 padding like your original)
  __shared__ bf16_isa As[2][BM][BK + 2];
  __shared__ bf16_isa Bs[2][BK][BN + 2];

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
  const int ly = (lane_id >> 4) & (WK == 16 ? 3 : 1);

  // Buffer resources (ranged — branch-free tails)
  BufferResource rA = {.v = {
                         .ptr = reinterpret_cast<uintptr_t>(A),
                         .range = (M > 0 && K > 0) ? (unsigned)((size_t)M * K * sizeof(float)) : 0u,
                         .config = BufferResource::kDataFormatU32Config,
                       }};
  BufferResource rB = {.v = {
                         .ptr = reinterpret_cast<uintptr_t>(B),
                         .range = (K > 0 && N > 0) ? (unsigned)((size_t)K * N * sizeof(bf16)) : 0u,
                         .config = BufferResource::kDataFormatU32Config,
                       }};
  BufferResource rC = {.v = {
                         .ptr = reinterpret_cast<uintptr_t>(C),
                         .range = (M > 0 && N > 0) ? (unsigned)((size_t)M * N * sizeof(float)) : 0u,
                         .config = BufferResource::kDataFormatU32Config,
                       }};

  // ===== Stage 0 preload (k_base = 0) =====
  int stage = 0;

  // A: 16B loads -> pack -> LDS[0] (swizzled)
#pragma unroll
  for (int i = 0; i < A_VEC_PER_THR; i++) {
    int vec_idx = tid + i * BLOCK_THREADS;
    int elem_idx = vec_idx * VEC_A_SIZE;
    int r = elem_idx / BK;
    int c = elem_idx % BK;

    int g_row = block_row_start + r;
    int g_col = 0 + c;

    int byte_off = ((int)((size_t)g_row * K + g_col) * (int)sizeof(float));
    v4i raw = rA.LoadV4(byte_off, AUX_A);
    float4 v = *reinterpret_cast<const float4 *>(&raw);

    const bf16x4 p = pack_f4_to_bf16x4(v);
    As[stage][r][lds_swz_a(r, c + 0)] = p[0];
    As[stage][r][lds_swz_a(r, c + 1)] = p[1];
    As[stage][r][lds_swz_a(r, c + 2)] = p[2];
    As[stage][r][lds_swz_a(r, c + 3)] = p[3];
  }

  // B: 16B loads -> LDS[0] (swizzled, SLC on)
#pragma unroll
  for (int i = 0; i < B_VEC_PER_THR; ++i) {
    int vec_idx = tid + i * BLOCK_THREADS;
    int elem_idx = vec_idx * VEC_B_SIZE;
    int r = elem_idx / BN;
    int c = elem_idx % BN;

    int g_row = 0 + r;
    int g_col = block_col_start + c;

    int byte_off = ((int)((size_t)g_row * N + g_col) * (int)sizeof(bf16));
    v4i raw = rB.LoadV4(byte_off, AUX_B);
    *reinterpret_cast<v4i *>(&Bs[stage][r][lds_swz_b(r, c)]) = raw;
  }
  __syncthreads();

  // ===== Main K loop =====
  for (int k_base = 0; k_base < K; k_base += BK) {
    const bool has_next = (k_base + BK) < K;
    const int next_stage = stage ^ 1;

    // Prefetch NEXT slab into regs
    v4i regA[A_VEC_PER_THR];
    v4i regB[B_VEC_PER_THR];

    if (has_next) {
#pragma unroll
      for (int i = 0; i < A_VEC_PER_THR; i++) {
        int vec_idx = tid + i * BLOCK_THREADS;
        int elem_idx = vec_idx * VEC_A_SIZE;
        int r = elem_idx / BK;
        int c = elem_idx % BK;

        int g_row = block_row_start + r;
        int g_col = (k_base + BK) + c;

        int byte_off = ((int)((size_t)g_row * K + g_col) * (int)sizeof(float));
        regA[i] = rA.LoadV4(byte_off, AUX_A);
      }
#pragma unroll
      for (int i = 0; i < B_VEC_PER_THR; ++i) {
        int vec_idx = tid + i * BLOCK_THREADS;
        int elem_idx = vec_idx * VEC_B_SIZE;
        int r = elem_idx / BN;
        int c = elem_idx % BN;

        int g_row = (k_base + BK) + r;
        int g_col = block_col_start + c;

        int byte_off = ((int)((size_t)g_row * N + g_col) * (int)sizeof(bf16));
        regB[i] = rB.LoadV4(byte_off, AUX_B);
      }
    }

    // ---- Compute current slab ----
#pragma unroll
    for (int kk = 0; kk < BK; kk += WK) {
      asm volatile("s_nop 1");  // small spacing; try 0/1/2

#pragma unroll
      for (int mt = 0; mt < (TM / WM); ++mt) {
#pragma unroll
        for (int nt = 0; nt < (TN / WN); ++nt) {
          const int a_row = wave_row * TM + mt * WM + lx;
          const int b_col = wave_col * TN + nt * WN + lx;

          bf16x4 a_vec, b_vec;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int kcol = i + ly * 4;  // 0..15 (WK=16)
            a_vec[i] = As[stage][a_row][lds_swz_a(a_row, kk + kcol)];
            b_vec[i] = Bs[stage][kk + kcol][lds_swz_b(kk + kcol, b_col)];
          }
          acc[mt][nt] = MFMA_BF16_16x16(a_vec, b_vec, acc[mt][nt]);
        }
      }
    }

    __syncthreads();

    // ---- Commit NEXT slab to LDS ----
    if (has_next) {
#pragma unroll
      for (int i = 0; i < A_VEC_PER_THR; i++) {
        int elem_idx = (tid + i * BLOCK_THREADS) * VEC_A_SIZE;
        int r = elem_idx / BK;
        int c = elem_idx % BK;
        float4 v = *reinterpret_cast<float4 *>(&regA[i]);
        const bf16x4 p = pack_f4_to_bf16x4(v);
        As[next_stage][r][lds_swz_a(r, c + 0)] = p[0];
        As[next_stage][r][lds_swz_a(r, c + 1)] = p[1];
        As[next_stage][r][lds_swz_a(r, c + 2)] = p[2];
        As[next_stage][r][lds_swz_a(r, c + 3)] = p[3];
      }
#pragma unroll
      for (int i = 0; i < B_VEC_PER_THR; ++i) {
        int elem_idx = (tid + i * BLOCK_THREADS) * VEC_B_SIZE;
        int r = elem_idx / BN;
        int c = elem_idx % BN;
        *reinterpret_cast<v4i *>(&Bs[next_stage][r][lds_swz_b(r, c)]) = regB[i];
      }
      __syncthreads();
    }

    stage = next_stage;
  }

  // ===== Branch-free C stores (+bias) =====
  const int lxx = lane_id & 15;

#pragma unroll
  for (int mt = 0; mt < (TM / WM); ++mt) {
#pragma unroll
    for (int nt = 0; nt < (TN / WN); ++nt) {
      const int base_row = wave_row_start + mt * WM + (4 * (lane_id >> 4));
      const int base_col = wave_col_start + nt * WN + lxx;

      float bias_val = bias ? (float)bias[base_col] : 0.f;

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = base_row + i;
        const int col = base_col;

        float out = acc[mt][nt][i] + bias_val;
        const int byte_off = ((int)((size_t)row * N + col) * (int)sizeof(float));
        rC.StoreF32(byte_off, out, AUX_C);  // dropped if OOB by SRD range
      }
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS) void gemm_mfma_v2_origin(const float *__restrict__ A,
                                                                     const bf16 *__restrict__ B,
                                                                     float *__restrict__ C,
                                                                     const bf16 *__restrict__ bias,
                                                                     int M, int N, int K) {
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
    } else {
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
__global__ __launch_bounds__(BLOCK_THREADS) void gemm_mfma_v2_bf16(const bf16 *__restrict__ A,
                                                                   const bf16 *__restrict__ B,
                                                                   float *__restrict__ C,
                                                                   const bf16 *__restrict__ bias,
                                                                   int M, int N, int K) {
  constexpr int WM = 16, WN = 16, WK = 16;
  constexpr int VEC_B_SIZE = 8;
  constexpr int VEC_A_SIZE = 8;

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

  // Khai báo shared memory
  __shared__ __bf16 As[BM][BK + 2];
  __shared__ __bf16 Bs[BK][BN + 2];

  // Khai báo thanh ghi tích lũy (accumulator)
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
  constexpr int A_VEC_PER_THR = (A_ELEMS_TILE / VEC_A_SIZE) / BLOCK_THREADS;
  constexpr int B_VEC_PER_THR = (B_ELEMS_TILE / VEC_B_SIZE) / BLOCK_THREADS;

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

    // THAY ĐỔI: Tải 8 phần tử bf16 (16 bytes) bằng uint4
    if (g_row < M && (g_col + VEC_A_SIZE - 1) < K) {
      *reinterpret_cast<uint4 *>(&As[r][c]) =
        *reinterpret_cast<const uint4 *>(&A[(size_t)g_row * K + g_col]);
    } else {
      // Xử lý vùng biên, điền 0
      uint4 z = {0, 0, 0, 0};
      *reinterpret_cast<uint4 *>(&As[r][c]) = z;
    }
  }

  // Tải tile đầu tiên của B vào shared memory
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
    } else {
      uint4 z = {0, 0, 0, 0};
      *reinterpret_cast<uint4 *>(&Bs[r][c]) = z;
    }
  }
  __syncthreads();

  // Vòng lặp chính - Software Pipelining
  for (int k_base = 0; k_base < K; k_base += BK) {
    // THAY ĐỔI: Thanh ghi cho A bây giờ là uint4 để chứa 8 phần tử bf16
    uint4 regA[A_VEC_PER_THR];
    uint4 regB[B_VEC_PER_THR];
    const bool has_next = (k_base + BK) < K;

    if (has_next) {
      // Tải trước (prefetch) tile tiếp theo của A vào thanh ghi
#pragma unroll
      for (int i = 0; i < A_VEC_PER_THR; i++) {
        const int vec_idx = tid + i * BLOCK_THREADS;
        const int elem_idx = vec_idx * VEC_A_SIZE;
        const int r = elem_idx / BK;
        const int c = elem_idx % BK;
        const int g_row = block_row_start + r;
        const int g_col = (k_base + BK) + c;

        // THAY ĐỔI: Tải trực tiếp bf16 vào regA, không cần chuyển đổi
        if (g_row < M && (g_col + VEC_A_SIZE - 1) < K) {
          regA[i] = *reinterpret_cast<const uint4 *>(&A[(size_t)g_row * K + g_col]);
        } else {
          regA[i] = uint4{0, 0, 0, 0};
        }
      }

      // Tải trước (prefetch) tile tiếp theo của B vào thanh ghi
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
        } else {
          regB[i] = uint4{0, 0, 0, 0};
        }
      }
    }

    // Vòng lặp tính toán MFMA
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

    // Ghi dữ liệu đã tải trước từ thanh ghi vào shared memory cho lần lặp tiếp theo
    if (has_next) {
#pragma unroll
      for (int i = 0; i < A_VEC_PER_THR; i++) {
        const int elem_idx = (tid + i * BLOCK_THREADS) * VEC_A_SIZE;
        const int r = elem_idx / BK;
        const int c = elem_idx % BK;
        // THAY ĐỔI: Ghi trực tiếp từ regA (uint4) vào As
        *reinterpret_cast<uint4 *>(&As[r][c]) = regA[i];
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

  // Ghi kết quả cuối cùng từ thanh ghi tích lũy ra global memory
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

// ===== Launcher =====
inline void launch_gemm_mfma_v2_bf16(const bf16 *A, const bf16 *B, float *C, int M, int N, int K,
                                     hipStream_t stream = nullptr, const bf16 *bias = nullptr) {
  constexpr int BM = 128;
  constexpr int BN = 256;
  constexpr int BK = 32;  // multiple of 16; try 64 for big-K
  constexpr int TM = 64;
  constexpr int TN = 64;
  constexpr int BLOCK_THREADS = 512;

  dim3 block_dim(BLOCK_THREADS);
  dim3 grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM);

  hipLaunchKernelGGL((gemm_mfma_v2_bf16<BM, BN, BK, TM, TN, BLOCK_THREADS>), grid_dim, block_dim, 0,
                     stream, A, B, C, bias, M, N, K);
}

// ===== Launcher =====
inline void launch_gemm_mfma_v2(const float *A, const bf16 *B, float *C, int M, int N, int K,
                                hipStream_t stream = nullptr, const bf16 *bias = nullptr) {
  constexpr int BM = 128;
  constexpr int BN = 256;
  constexpr int BK = 32;  // multiple of 16; try 64 for big-K
  constexpr int TM = 64;
  constexpr int TN = 64;
  constexpr int BLOCK_THREADS = 512;

  dim3 block_dim(BLOCK_THREADS);
  dim3 grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM);

  hipLaunchKernelGGL((gemm_mfma_v2_origin<BM, BN, BK, TM, TN, BLOCK_THREADS>), grid_dim, block_dim,
                     0, stream, A, B, C, bias, M, N, K);
}
