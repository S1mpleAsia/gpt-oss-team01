#include <hip/hip_runtime.h>
#include "../include/tensor.hpp"

template <int BM = 16, int BN = 128, int BK = 16, int TM = 2, int TN = 8>
__global__ void matmul_kernel(const float *__restrict__ A, const float *__restrict__ B,
                              float *__restrict__ C, const float *bias, int M, int N, int K) {
  const int VEC_SIZE = 4;

  const int BLOCK_SIZE_X = BN / TN;
  const int BLOCK_SIZE_Y = BM / TM;

  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  const int block_col = blockIdx.x;
  const int block_row = blockIdx.y;

  // Padding có thể giúp hiệu năng, giữ lại để linh hoạt
  __shared__ float As[BM][BK + 8];
  __shared__ float Bs[BK][BN + 8];

  float acc[TM][TN] = {0.0f};

  for (int k_base = 0; k_base < K; k_base += BK) {
    constexpr int ITERS_A = (BK + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
#pragma unroll
    for (int it = 0; it < ITERS_A; ++it) {
      int kk = it * BLOCK_SIZE_X + tid_x;  // kk phụ thuộc tid_x, nhưng ITERS_A là hằng
      if (kk < BK) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          int row = block_row * BM + tid_y * TM + i;
          int col = k_base + kk;
          As[tid_y * TM + i][kk] = (row < M && col < K) ? A[row * K + col] : 0.0f;
        }
      }
    }

    constexpr int ITERS_B = (BK + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
#pragma unroll
    for (int it = 0; it < ITERS_B; ++it) {
      int tile_row = it * BLOCK_SIZE_Y + tid_y;
      if (tile_row < BK) {
        int global_row = k_base + tile_row;

        constexpr int V_ITERS = TN / 4;  // VEC_SIZE=4
#pragma unroll
        for (int i = 0; i < V_ITERS; ++i) {
          int global_col = block_col * BN + tid_x * TN + i * 4;

          float4 *dst = reinterpret_cast<float4 *>(&Bs[tile_row][tid_x * TN + i * 4]);
          if (global_row < K && (global_col + 3) < N) {
            const float4 *src = reinterpret_cast<const float4 *>(&B[global_row * N + global_col]);
            *dst = *src;
          } else {
            // fallback an toàn khi chạm mép
            for (int t = 0; t < 4; ++t) {
              int gc = global_col + t;
              Bs[tile_row][tid_x * TN + i * 4 + t] =
                (global_row < K && gc < N) ? B[global_row * N + gc] : 0.0f;
            }
          }
        }
      }
    }

    __syncthreads();

// Phần tính toán (accumulate) đã đúng, giữ nguyên
#pragma unroll
    for (int k = 0; k < BK; k++) {
      float a_reg[TM];
#pragma unroll
      for (int i = 0; i < TM; i++) {
        a_reg[i] = As[tid_y * TM + i][k];
      }

#pragma unroll
      for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < TN; j++) {
          acc[i][j] = fmaf(a_reg[i], Bs[k][tid_x * TN + j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

// Phần ghi kết quả (Write back) đã đúng, giữ nguyên
// --- WRITE BACK (+ optional bias) ---
#pragma unroll
  for (int i = 0; i < TM; i++) {
#pragma unroll
    for (int j = 0; j < TN; j += VEC_SIZE) {
      int row = block_row * BM + tid_y * TM + i;
      int col = block_col * BN + tid_x * TN + j;
      if (row < M && col < N) {
        if ((col + (VEC_SIZE - 1)) < N) {  // nguyên khối 4 phần tử
          float4 v = *reinterpret_cast<float4 *>(&acc[i][j]);
          if (bias) {
            const float4 b = *reinterpret_cast<const float4 *>(&bias[col]);
            v.x += b.x;
            v.y += b.y;
            v.z += b.z;
            v.w += b.w;
          }
          *reinterpret_cast<float4 *>(&C[row * N + col]) = v;
        } else {  // đuôi lẻ
          for (int t = 0; t < VEC_SIZE && (col + t) < N; ++t) {
            float v = acc[i][j + t];
            if (bias)
              v += bias[col + t];
            C[row * N + col + t] = v;
          }
        }
      }
    }
  }
}

template <int BM = 16, int BN = 128, int BK = 16, int TM = 2, int TN = 8>
__global__ void matmul_kernel_bf16(const float *__restrict__ A, const bf16 *__restrict__ B,
                                   float *__restrict__ C, const float *bias, int M, int N, int K) {
  const int VEC_SIZE = 4;
  const int VEC_SIZE_B = 8;

  const int BLOCK_SIZE_X = BN / TN;
  const int BLOCK_SIZE_Y = BM / TM;

  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  const int block_col = blockIdx.x;
  const int block_row = blockIdx.y;

  __shared__ float As[BM][BK + 8];
  __shared__ bf16 Bs[BK][BN + 8];

  float acc[TM][TN] = {0.0f};

  for (int k_base = 0; k_base < K; k_base += BK) {
    constexpr int ITERS_A = (BK + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
#pragma unroll
    for (int it = 0; it < ITERS_A; ++it) {
      int kk = it * BLOCK_SIZE_X + tid_x;  // kk phụ thuộc tid_x, nhưng ITERS_A là hằng
      if (kk < BK) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          int row = block_row * BM + tid_y * TM + i;
          int col = k_base + kk;
          As[tid_y * TM + i][kk] = (row < M && col < K) ? A[row * K + col] : 0.0f;
        }
      }
    }

    constexpr int ITERS_B = (BK + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
#pragma unroll
    for (int it = 0; it < ITERS_B; ++it) {
      int tile_row = it * BLOCK_SIZE_Y + tid_y;
      if (tile_row < BK) {
        int global_row = k_base + tile_row;
        constexpr int V_ITERS = TN / VEC_SIZE_B;
#pragma unroll
        for (int i = 0; i < V_ITERS; ++i) {
          int global_col = block_col * BN + tid_x * TN + i * VEC_SIZE_B;

          if (global_row < K && (global_col + VEC_SIZE_B - 1) < N) {
            const uint4 *src = reinterpret_cast<const uint4 *>(&B[global_row * N + global_col]);
            uint4 *dst = reinterpret_cast<uint4 *>(&Bs[tile_row][tid_x * TN + i * VEC_SIZE_B]);
            *dst = *src;
          } else {
            // fallback an toàn khi chạm mép
            for (int t = 0; t < VEC_SIZE_B; ++t) {
              int gc = global_col + t;
              Bs[tile_row][tid_x * TN + i * VEC_SIZE_B + t] =
                (global_row < K && gc < N) ? B[global_row * N + gc] : 0.0f;
            }
          }
        }
      }
    }

    __syncthreads();

// Phần tính toán (accumulate) đã đúng, giữ nguyên
#pragma unroll
    for (int k = 0; k < BK; k++) {
      float a_reg[TM];
#pragma unroll
      for (int i = 0; i < TM; i++) {
        a_reg[i] = As[tid_y * TM + i][k];
      }

#pragma unroll
      for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < TN; j++) {
          // 1. Đọc bfloat16 từ shared memory
          bf16 b_val_bf16 = Bs[k][tid_x * TN + j];
          // 2. Chuyển đổi sang float
          float b_val_f32 = float(b_val_bf16);
          // 3. Thực hiện fmaf
          acc[i][j] = fmaf(a_reg[i], b_val_f32, acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

// Phần ghi kết quả (Write back) đã đúng, giữ nguyên
// --- WRITE BACK (+ optional bias) ---
#pragma unroll
  for (int i = 0; i < TM; i++) {
#pragma unroll
    for (int j = 0; j < TN; j += VEC_SIZE) {
      int row = block_row * BM + tid_y * TM + i;
      int col = block_col * BN + tid_x * TN + j;
      if (row < M && col < N) {
        if ((col + (VEC_SIZE - 1)) < N) {  // nguyên khối 4 phần tử
          float4 v = *reinterpret_cast<float4 *>(&acc[i][j]);
          if (bias) {
            const float4 b = *reinterpret_cast<const float4 *>(&bias[col]);
            v.x += b.x;
            v.y += b.y;
            v.z += b.z;
            v.w += b.w;
          }
          *reinterpret_cast<float4 *>(&C[row * N + col]) = v;
        } else {  // đuôi lẻ
          for (int t = 0; t < VEC_SIZE && (col + t) < N; ++t) {
            float v = acc[i][j + t];
            if (bias)
              v += bias[col + t];
            C[row * N + col + t] = v;
          }
        }
      }
    }
  }
}

template <int BM = 16, int BN = 128, int BK = 16, int TM = 2, int TN = 8>
__global__ void matmul_kernel_residual(const float *__restrict__ A, const float *__restrict__ B,
                                       float *__restrict__ C, const float *bias, int M, int N,
                                       int K) {
  const int VEC_SIZE = 4;

  const int BLOCK_SIZE_X = BN / TN;
  const int BLOCK_SIZE_Y = BM / TM;

  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  const int block_col = blockIdx.x;
  const int block_row = blockIdx.y;

  // Padding có thể giúp hiệu năng, giữ lại để linh hoạt
  __shared__ float As[BM][BK + 8];
  __shared__ float Bs[BK][BN + 8];

  float acc[TM][TN] = {0.0f};

  for (int k_base = 0; k_base < K; k_base += BK) {
    constexpr int ITERS_A = (BK + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
#pragma unroll
    for (int it = 0; it < ITERS_A; ++it) {
      int kk = it * BLOCK_SIZE_X + tid_x;  // kk phụ thuộc tid_x, nhưng ITERS_A là hằng
      if (kk < BK) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          int row = block_row * BM + tid_y * TM + i;
          int col = k_base + kk;
          As[tid_y * TM + i][kk] = (row < M && col < K) ? A[row * K + col] : 0.0f;
        }
      }
    }

    constexpr int ITERS_B = (BK + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
#pragma unroll
    for (int it = 0; it < ITERS_B; ++it) {
      int tile_row = it * BLOCK_SIZE_Y + tid_y;
      if (tile_row < BK) {
        int global_row = k_base + tile_row;

        constexpr int V_ITERS = TN / 4;  // VEC_SIZE=4
#pragma unroll
        for (int i = 0; i < V_ITERS; ++i) {
          int global_col = block_col * BN + tid_x * TN + i * 4;

          float4 *dst = reinterpret_cast<float4 *>(&Bs[tile_row][tid_x * TN + i * 4]);
          if (global_row < K && (global_col + 3) < N) {
            const float4 *src = reinterpret_cast<const float4 *>(&B[global_row * N + global_col]);
            *dst = *src;
          } else {
            // fallback an toàn khi chạm mép
            for (int t = 0; t < 4; ++t) {
              int gc = global_col + t;
              Bs[tile_row][tid_x * TN + i * 4 + t] =
                (global_row < K && gc < N) ? B[global_row * N + gc] : 0.0f;
            }
          }
        }
      }
    }

    __syncthreads();

// Phần tính toán (accumulate) đã đúng, giữ nguyên
#pragma unroll
    for (int k = 0; k < BK; k++) {
      float a_reg[TM];
#pragma unroll
      for (int i = 0; i < TM; i++) {
        a_reg[i] = As[tid_y * TM + i][k];
      }

#pragma unroll
      for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < TN; j++) {
          acc[i][j] = fmaf(a_reg[i], Bs[k][tid_x * TN + j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

// Phần ghi kết quả (Write back) đã đúng, giữ nguyên
// --- WRITE BACK (+ optional bias) ---
#pragma unroll
  for (int i = 0; i < TM; i++) {
#pragma unroll
    for (int j = 0; j < TN; j += VEC_SIZE) {
      int row = block_row * BM + tid_y * TM + i;
      int col = block_col * BN + tid_x * TN + j;
      if (row < M && col < N) {
        if ((col + (VEC_SIZE - 1)) < N) {  // nguyên khối 4 phần tử
          float4 v = *reinterpret_cast<float4 *>(&acc[i][j]);
          if (bias) {
            const float4 b = *reinterpret_cast<const float4 *>(&bias[col]);
            v.x += b.x;
            v.y += b.y;
            v.z += b.z;
            v.w += b.w;
          }

          float4 c_old = *reinterpret_cast<float4 *>(&C[row * N + col]);
          v.x += c_old.x;
          v.y += c_old.y;
          v.z += c_old.z;
          v.w += c_old.w;
          *reinterpret_cast<float4 *>(&C[row * N + col]) = v;
        } else {  // đuôi lẻ
          for (int t = 0; t < VEC_SIZE && (col + t) < N; ++t) {
            float v = acc[i][j + t];
            if (bias)
              v += bias[col + t];

            v += C[row * N + col + t];
            C[row * N + col + t] = v;
          }
        }
      }
    }
  }
}
