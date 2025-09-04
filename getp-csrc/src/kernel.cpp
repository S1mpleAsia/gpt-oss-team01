#include "../include/kernel.hpp"

template <int BM, int BN, int BK, int TM, int TN>
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

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_kernel_bf16(const float *__restrict__ A, const bf16 *__restrict__ B,
                                   float *__restrict__ C, const bf16 *bias, int M, int N, int K) {
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
            float b0 = float(bias[col + 0]);
            float b1 = float(bias[col + 1]);
            float b2 = float(bias[col + 2]);
            float b3 = float(bias[col + 3]);
            // const float4 b = *reinterpret_cast<const float4 *>(&bias[col]);
            v.x += b0;
            v.y += b1;
            v.z += b2;
            v.w += b3;
          }
          *reinterpret_cast<float4 *>(&C[row * N + col]) = v;
        } else {  // đuôi lẻ
          for (int t = 0; t < VEC_SIZE && (col + t) < N; ++t) {
            float v = acc[i][j + t];
            if (bias)
              v += float(bias[col + t]);
            C[row * N + col + t] = v;
          }
        }
      }
    }
  }
}

template <int BM = 16, int BN = 128, int BK = 16, int TM = 2, int TN = 8>
__global__ void matmul_kernel_bf16_moe(const float *__restrict__ A, const bf16 *__restrict__ B,
                                       float *__restrict__ C, const bf16 *bias,
                                       const int *expert_offsets, int M, int N, int K) {
  const int VEC_SIZE = 4;
  const int VEC_SIZE_B = 8;

  const int BLOCK_SIZE_X = BN / TN;
  const int BLOCK_SIZE_Y = BM / TM;

  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  const int block_col = blockIdx.x;
  const int block_row = blockIdx.y;
  const int expert_id = blockIdx.z;

  const int row_begin = expert_offsets[expert_id];
  const int row_end = expert_offsets[expert_id + 1];
  const int rows_M = (row_end - row_begin > 0) ? (row_end - row_begin) : 0;

  if (rows_M <= 0)
    return;

  const bf16 *B_ptr = B + (size_t)expert_id * N * K;
  const bf16 *bias_ptr = bias ? (bias + (size_t)expert_id * N) : nullptr;

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
          int row_local = block_row * BM + tid_y * TM + i;
          int row_global = row_begin + row_local;
          int col = k_base + kk;
          As[tid_y * TM + i][kk] = (row_local < rows_M && col < K) ? A[row_global * K + col] : 0.0f;
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
            const uint4 *src = reinterpret_cast<const uint4 *>(&B_ptr[global_row * N + global_col]);
            uint4 *dst = reinterpret_cast<uint4 *>(&Bs[tile_row][tid_x * TN + i * VEC_SIZE_B]);
            *dst = *src;
          } else {
#pragma unroll
            for (int t = 0; t < VEC_SIZE_B; ++t) {
              int gc = global_col + t;
              Bs[tile_row][tid_x * TN + i * VEC_SIZE_B + t] =
                (global_row < K && gc < N) ? B_ptr[global_row * N + gc] : 0.0f;
            }
          }
        }
      }
    }

    __syncthreads();

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
          float b_val = float(Bs[k][tid_x * TN + j]);
          acc[i][j] = fmaf(a_reg[i], b_val, acc[i][j]);
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
      int row_local = block_row * BM + tid_y * TM + i;
      int row_global = row_begin + row_local;
      int col = block_col * BN + tid_x * TN + j;
      if (row_local < rows_M && col < N) {
        if ((col + (VEC_SIZE - 1)) < N) {  // nguyên khối 4 phần tử
          float4 v = *reinterpret_cast<float4 *>(&acc[i][j]);
          if (bias_ptr) {
            float4 b;
            b.x = float(bias_ptr[col + 0]);
            b.y = float(bias_ptr[col + 1]);
            b.z = float(bias_ptr[col + 2]);
            b.w = float(bias_ptr[col + 3]);

            v.x += b.x;
            v.y += b.y;
            v.z += b.z;
            v.w += b.w;
          }
          *reinterpret_cast<float4 *>(&C[row_global * N + col]) = v;
        } else {  // đuôi lẻ
          for (int t = 0; t < VEC_SIZE && (col + t) < N; ++t) {
            float v = acc[i][j + t];
            if (bias_ptr)
              v += float(bias_ptr[col + t]);
            C[row_global * N + col + t] = v;
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

__global__ void build_expert_offsets_kernel(
  const int *__restrict__ topk_idx,   // [batch_size * experts_per_token]
  int *__restrict__ sorted_pair_ids,  // [batch_size * experts_per_token]
  int *__restrict__ expert_offsets,   // [n_experts + 1]
  int batch_size, int experts_per_token, int n_experts) {
  extern __shared__ int smem[];
  int *expert_counts = smem;

  for (int i = threadIdx.x; i < n_experts; i += blockDim.x)
    expert_counts[i] = 0;
  __syncthreads();

  const int total_pairs = batch_size * experts_per_token;
  for (int pair = threadIdx.x; pair < total_pairs; pair += blockDim.x) {
    int e = topk_idx[pair];
    if (0 <= e && e < n_experts)
      atomicAdd(&expert_counts[e], 1);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    expert_offsets[0] = 0;
    for (int e = 0; e < n_experts; ++e)
      expert_offsets[e + 1] = expert_offsets[e] + expert_counts[e];
  }
  __syncthreads();

  for (int i = threadIdx.x; i < n_experts; i += blockDim.x)
    expert_counts[i] = 0;
  __syncthreads();

  for (int pair = threadIdx.x; pair < total_pairs; pair += blockDim.x) {
    int e = topk_idx[pair];
    if (0 <= e && e < n_experts) {
      int pos = expert_offsets[e] + atomicAdd(&expert_counts[e], 1);
      sorted_pair_ids[pos] = pair;
    }
  }
}

// 2) Pack X by sorted_ids
__global__ void gather_inputs_by_sorted_kernel(
  const float *x_in,  // [batch_size, hidden_dim]
  const int *sorted_pair_ids,
  float *x_packed,  // [batch_size * experts_per_token, hidden_dim]
  int batch_size, int hidden_dim, int experts_per_token) {
  int packed_row = blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (packed_row >= batch_size * experts_per_token || col >= hidden_dim)
    return;

  int pair_id = sorted_pair_ids[packed_row];  // 0..B*k-1
  int token_id = pair_id / experts_per_token;
  x_packed[packed_row * hidden_dim + col] = x_in[token_id * hidden_dim + col];
}

// Just copied from the origin
__global__ void swiglu_interleaved_batched_fast_v2(const float *__restrict__ in2I,  // [NK, 2I]
                                                   float *__restrict__ outI,        // [NK, I]
                                                   int I, int NK, float clamp_limit) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t N = (size_t)NK * I;
  if (idx >= N)
    return;

  int j = idx % I;
  size_t s = idx / I;
  size_t base = s * (size_t)(2 * I);

  float g = in2I[base + 2 * j];
  float u = in2I[base + 2 * j + 1];

  if (clamp_limit > 0.f) {
    g = fminf(fmaxf(g, -clamp_limit), clamp_limit);
    u = fminf(fmaxf(u, -clamp_limit), clamp_limit);
  }
  const float alpha = 1.702f;
  float silu = g * (1.f / (1.f + expf(-alpha * g)));
  outI[idx] = silu * (u + 1.f);
}

__global__ void scale_scatter_add_kernel_sorted(
  const float *__restrict__ z_sorted,  // subrange: expert's rows
  const int *__restrict__ sorted_ids,  // global [B*k]
  const float *__restrict__ topk_v,    // [B, k]
  float *__restrict__ e_agg,           // [B, H]
  int start_pos, int rows_per_experts, int hidden_dim, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;  // 0..rows_per_experts-1
  int h = blockIdx.x * blockDim.x + threadIdx.x;    // 0..H-1
  if (row >= rows_per_experts || h >= hidden_dim)
    return;

  int pos = start_pos + row;
  int pair = sorted_ids[pos];
  int b = pair / k;
  int e = pair % k;

  float w = topk_v[(size_t)b * k + e];
  float val = z_sorted[(size_t)row * hidden_dim + h] * w;

  atomicAdd(&e_agg[(size_t)b * hidden_dim + h], val);
}

__global__ void compute_max_rows_from_offsets_kernel(const int *__restrict__ expert_offsets,
                                                     int n_experts, int *__restrict__ max_rows) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  int local_max = 0;
  if (e < n_experts) {
    int rows = expert_offsets[e + 1] - expert_offsets[e];
    local_max = rows > 0 ? rows : 0;
  }
  if (e < n_experts)
    atomicMax(max_rows, local_max);
}

__global__ void scale_scatter_add_kernel_sorted_all(
  const float *__restrict__ z_sorted,      // [total_pairs, hidden_dim]
  const int *__restrict__ sorted_ids,      // [total_pairs]
  const float *__restrict__ topk_v,        // [batch_size, experts_per_token]
  float *__restrict__ e_agg,               // [batch_size, hidden_dim]
  const int *__restrict__ expert_offsets,  // [n_experts+1]
  int hidden_dim, int experts_per_token) {
  int expert_id = blockIdx.z;
  int row_local = blockIdx.y * blockDim.y + threadIdx.y;
  int h = blockIdx.x * blockDim.x + threadIdx.x;

  int row_begin = expert_offsets[expert_id];
  int row_end = expert_offsets[expert_id + 1];
  int rows_M = max(0, row_end - row_begin);
  if (row_local >= rows_M || h >= hidden_dim)
    return;

  int pos = row_begin + row_local;
  int pair = sorted_ids[pos];
  int b = pair / experts_per_token;
  int e = pair % experts_per_token;

  float w = topk_v[(size_t)b * experts_per_token + e];
  float val = z_sorted[(size_t)pos * hidden_dim + h] * w;

  atomicAdd(&e_agg[(size_t)b * hidden_dim + h], val);
}

static inline void moe_init_buffers(Tensor *e_agg, Tensor *mlp1_out, Tensor *gate_up, Tensor *tb3,
                                    TensorI32 *sorted_pair_ids, TensorI32 *expert_offsets,
                                    Tensor *x_packed, int batch_size, int hidden_dim,
                                    hipStream_t stream) {
  float *e_agg_ptr = (float *)e_agg->d_buf;
  CHECK_HIP(hipMemsetAsync(e_agg_ptr, 0, (size_t)batch_size * hidden_dim * sizeof(float), stream));
  CHECK_HIP(
    hipMemsetAsync(mlp1_out->d_buf, 0, mlp1_out->num_elem() * mlp1_out->get_dtype_size(), stream));
  CHECK_HIP(
    hipMemsetAsync(gate_up->d_buf, 0, gate_up->num_elem() * gate_up->get_dtype_size(), stream));
  CHECK_HIP(hipMemsetAsync(tb3->d_buf, 0, tb3->num_elem() * tb3->get_dtype_size(), stream));
  CHECK_HIP(
    hipMemsetAsync(sorted_pair_ids->d_buf, 0, sorted_pair_ids->num_elem() * sizeof(int), stream));
  CHECK_HIP(
    hipMemsetAsync(expert_offsets->d_buf, 0, expert_offsets->num_elem() * sizeof(int), stream));
  CHECK_HIP(
    hipMemsetAsync(x_packed->d_buf, 0, x_packed->num_elem() * x_packed->get_dtype_size(), stream));
}

// 1) sort & build offsets
static inline void moe_build_offsets(
  TensorI32 *topk_idx,         // [batch_size, experts_per_token]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  TensorI32 *expert_offsets,   // [n_experts + 1]
  int batch_size, int experts_per_token, int n_experts, hipStream_t stream) {
  const int *topk_idx_ptr = (const int *)topk_idx->d_buf;

  const size_t shmem_bytes = (size_t)n_experts * sizeof(int);
  dim3 block_size(256);
  dim3 grid_size(1);

  build_expert_offsets_kernel<<<grid_size, block_size, shmem_bytes, stream>>>(
    topk_idx_ptr, sorted_pair_ids->d_buf, expert_offsets->d_buf, batch_size, experts_per_token,
    n_experts);
}

// 2) pack X theo sorted ids
static inline void moe_pack_inputs(
  Tensor *x_in,                // [batch_size, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *x_packed,            // [batch_size * experts_per_token, hidden_dim]
  int batch_size, int hidden_dim, int experts_per_token, hipStream_t stream) {
  const float *x_ptr = (const float *)x_in->d_buf;

  const int total_pairs = batch_size * experts_per_token;
  dim3 block_size(256, 1);
  dim3 grid_size((hidden_dim + block_size.x - 1) / block_size.x, total_pairs);

  gather_inputs_by_sorted_kernel<<<grid_size, block_size, 0, stream>>>(
    x_ptr, sorted_pair_ids->d_buf, (float *)x_packed->d_buf, batch_size, hidden_dim,
    experts_per_token);
}

// 3) lấy max số hàng trên mỗi expert từ offsets
static inline int moe_get_max_rows_per_expert(TensorI32 *expert_offsets, int n_experts,
                                              hipStream_t stream) {
  int max_rows_per_expert = 0;
  int *d_max_rows = nullptr;
  CHECK_HIP(hipMalloc(&d_max_rows, sizeof(int)));
  CHECK_HIP(hipMemsetAsync(d_max_rows, 0, sizeof(int), stream));
  compute_max_rows_from_offsets_kernel<<<(n_experts + 255) / 256, 256, 0, stream>>>(
    expert_offsets->d_buf, n_experts, d_max_rows);
  CHECK_HIP(
    hipMemcpyAsync(&max_rows_per_expert, d_max_rows, sizeof(int), hipMemcpyDeviceToHost, stream));
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipFree(d_max_rows));
  return max_rows_per_expert;
}

// 4) MLP1: (x_packed @ W1 + b1) -> mlp1_out  (2*inter_dim)
// w_mlp1: [n_layers, n_experts, hidden_dim, 2*inter_dim]
// b_mlp1: [n_layers, n_experts, 2*inter_dim]
static inline void moe_mlp1_forward(Tensor *x_packed,  // [total_pairs, hidden_dim]
                                    Tensor *w_mlp1, Tensor *b_mlp1,
                                    TensorI32 *expert_offsets,  // [n_experts+1]
                                    Tensor *mlp1_out,           // [total_pairs, 2*inter_dim]
                                    long long layer_offset, int n_experts, int hidden_dim,
                                    int inter_dim, int max_rows_per_expert, int total_pairs,
                                    hipStream_t stream) {
  const bf16 *w1_ptr =
    (const bf16 *)w_mlp1->d_buf + (size_t)layer_offset * n_experts * hidden_dim * 2 * inter_dim;
  const bf16 *b1_ptr =
    (const bf16 *)b_mlp1->d_buf + (size_t)layer_offset * n_experts * 2 * inter_dim;

  constexpr int BM = 16, BN = 128, BK = 16, TM = 2, TN = 8;
  dim3 block_size(BN / TN, BM / TM);
  dim3 grid_size((2 * inter_dim + BN - 1) / BN, (max_rows_per_expert + BM - 1) / BM, n_experts);

  matmul_kernel_bf16_moe<BM, BN, BK, TM, TN><<<grid_size, block_size, 0, stream>>>(
    (const float *)x_packed->d_buf, w1_ptr, (float *)mlp1_out->d_buf, b1_ptr, expert_offsets->d_buf,
    total_pairs, 2 * inter_dim, hidden_dim);
}

// 5) SwiGLU (interleaved) & clamp
static inline void moe_swiglu(Tensor *mlp1_out,  // [total_pairs, 2*inter_dim]
                              Tensor *gate_up,   // [total_pairs, inter_dim]
                              int batch_size, int experts_per_token, int inter_dim,
                              float clamp_limit, hipStream_t stream) {
  size_t total = (size_t)batch_size * experts_per_token * inter_dim;
  dim3 block_size(256);
  dim3 grid_size((total + 255) / 256);

  swiglu_interleaved_batched_fast_v2<<<grid_size, block_size, 0, stream>>>(
    (const float *)mlp1_out->d_buf, (float *)gate_up->d_buf, inter_dim,
    batch_size * experts_per_token, clamp_limit);
}

// 6) MLP2: (gate_up @ W2 + b2) -> tb3  (hidden_dim)
// w_mlp2: [n_layers, n_experts, inter_dim, hidden_dim]
// b_mlp2: [n_layers, n_experts, hidden_dim]
static inline void moe_mlp2_forward(Tensor *gate_up,  // [total_pairs, inter_dim]
                                    Tensor *w_mlp2, Tensor *b_mlp2,
                                    TensorI32 *expert_offsets,  // [n_experts+1]
                                    Tensor *tb3,                // [total_pairs, hidden_dim]
                                    bool has_bias, long long layer_offset, int n_experts,
                                    int inter_dim, int hidden_dim, int max_rows_per_expert,
                                    int total_pairs, hipStream_t stream) {
  const bf16 *w2_ptr =
    (const bf16 *)w_mlp2->d_buf + (size_t)layer_offset * n_experts * inter_dim * hidden_dim;
  const bf16 *b2_ptr =
    has_bias ? (const bf16 *)b_mlp2->d_buf + (size_t)layer_offset * n_experts * hidden_dim
             : nullptr;

  constexpr int BM = 16, BN = 128, BK = 16, TM = 2, TN = 8;
  dim3 block_size(BN / TN, BM / TM);
  dim3 grid_size((hidden_dim + BN - 1) / BN, (max_rows_per_expert + BM - 1) / BM, n_experts);

  matmul_kernel_bf16_moe<BM, BN, BK, TM, TN><<<grid_size, block_size, 0, stream>>>(
    (const float *)gate_up->d_buf, w2_ptr, (float *)tb3->d_buf, b2_ptr, expert_offsets->d_buf,
    total_pairs, hidden_dim, inter_dim);
}

// 7) scatter-add có scale theo topk_v -> e_agg
static inline void moe_scatter_aggregate(
  Tensor *tb3,                 // [total_pairs, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  Tensor *topk_v,              // [batch_size, experts_per_token]
  Tensor *e_agg,               // [batch_size, hidden_dim]
  TensorI32 *expert_offsets,   // [n_experts+1]
  int hidden_dim, int experts_per_token, int n_experts, int max_rows_per_expert,
  hipStream_t stream) {
  const float *topk_v_ptr = (const float *)topk_v->d_buf;
  float *e_agg_ptr = (float *)e_agg->d_buf;

  dim3 block_size(32, 8, 1);
  dim3 grid_size((hidden_dim + block_size.x - 1) / block_size.x,
                 (max_rows_per_expert + block_size.y - 1) / block_size.y, n_experts);

  scale_scatter_add_kernel_sorted_all<<<grid_size, block_size, 0, stream>>>(
    (const float *)tb3->d_buf, (const int *)sorted_pair_ids->d_buf, topk_v_ptr, e_agg_ptr,
    expert_offsets->d_buf, hidden_dim, experts_per_token);
}

void moe_block_matmul_style(
  Tensor *x_in,                // [batch_size, hidden_dim]
  TensorI32 *topk_idx,         // [batch_size, experts_per_token]
  Tensor *topk_v,              // [batch_size, experts_per_token]
  Tensor *w_mlp1,              // [n_layers, n_experts, hidden_dim, 2*intermediate_dim]
  Tensor *b_mlp1,              // [n_layers, n_experts, 2*intermediate_dim]
  Tensor *w_mlp2,              // [n_layers, n_experts, intermediate_dim, hidden_dim]
  Tensor *b_mlp2,              // [n_layers, n_experts, hidden_dim]
  Tensor *e_agg,               // [batch_size, hidden_dim]
  Tensor *mlp1_out,            // [batch_size * experts_per_token, 2 * inter_dim]
  Tensor *gate_up,             // [batch_size * experts_per_token, inter_dim]
  Tensor *tb3,                 // [batch_size * experts_per_token, hidden_dim]
  TensorI32 *sorted_pair_ids,  // [batch_size * experts_per_token]
  TensorI32 *expert_offsets,   // [n_experts + 1]
  Tensor *x_packed,            // [batch_size * experts_per_token, hidden_dim]
  float clamp_limit, long long layer_offset, hipStream_t stream) {
  // ====== Lấy kích thước ======
  const int batch_size = (int)x_in->shape[0];
  const int hidden_dim = (int)x_in->shape[1];
  const int experts_per_token = (int)topk_idx->shape[1];
  const int n_experts = (int)w_mlp1->shape[1];
  const int inter_dim = (int)w_mlp1->shape[3] / 2;
  const int total_pairs = batch_size * experts_per_token;

  // ====== Con trỏ ======
  const float *x_ptr = (const float *)x_in->d_buf;
  const int *topk_idx_ptr = (const int *)topk_idx->d_buf;
  const float *topk_v_ptr = (const float *)topk_v->d_buf;

  const bf16 *w1_ptr =
    (const bf16 *)w_mlp1->d_buf + (size_t)layer_offset * n_experts * hidden_dim * 2 * inter_dim;
  const bf16 *b1_ptr =
    (const bf16 *)b_mlp1->d_buf + (size_t)layer_offset * n_experts * 2 * inter_dim;
  const bf16 *w2_ptr =
    (const bf16 *)w_mlp2->d_buf + (size_t)layer_offset * n_experts * inter_dim * hidden_dim;
  const bf16 *b2_ptr = (const bf16 *)b_mlp2->d_buf + (size_t)layer_offset * n_experts * hidden_dim;

  float *e_agg_ptr = (float *)e_agg->d_buf;

  CHECK_HIP(hipMemsetAsync(e_agg_ptr, 0, (size_t)batch_size * hidden_dim * sizeof(float), stream));
  CHECK_HIP(
    hipMemsetAsync(mlp1_out->d_buf, 0, mlp1_out->num_elem() * mlp1_out->get_dtype_size(), stream));
  CHECK_HIP(
    hipMemsetAsync(gate_up->d_buf, 0, gate_up->num_elem() * gate_up->get_dtype_size(), stream));
  CHECK_HIP(hipMemsetAsync(tb3->d_buf, 0, tb3->num_elem() * tb3->get_dtype_size(), stream));

  CHECK_HIP(
    hipMemsetAsync(sorted_pair_ids->d_buf, 0, sorted_pair_ids->num_elem() * sizeof(int), stream));
  CHECK_HIP(
    hipMemsetAsync(expert_offsets->d_buf, 0, expert_offsets->num_elem() * sizeof(int), stream));
  CHECK_HIP(
    hipMemsetAsync(x_packed->d_buf, 0, x_packed->num_elem() * x_packed->get_dtype_size(), stream));

  // ====== 1) sort & offsets ======
  {
    const size_t shmem_bytes = (size_t)n_experts * sizeof(int);
    dim3 block_size(256);
    dim3 grid_size(1);

    build_expert_offsets_kernel<<<grid_size, block_size, shmem_bytes, stream>>>(
      topk_idx_ptr, sorted_pair_ids->d_buf, expert_offsets->d_buf, batch_size, experts_per_token,
      n_experts);
  }

  // ====== 2) pack X theo sorted_ids ======
  {
    dim3 block_size(256, 1);
    dim3 grid_size((hidden_dim + block_size.x - 1) / block_size.x, total_pairs);
    gather_inputs_by_sorted_kernel<<<grid_size, block_size, 0, stream>>>(
      x_ptr, sorted_pair_ids->d_buf, (float *)x_packed->d_buf, batch_size, hidden_dim,
      experts_per_token);
  }

  int max_rows_per_expert = 0;
  {
    int *d_max_rows = nullptr;
    CHECK_HIP(hipMalloc(&d_max_rows, sizeof(int)));
    CHECK_HIP(hipMemsetAsync(d_max_rows, 0, sizeof(int), stream));
    compute_max_rows_from_offsets_kernel<<<(n_experts + 255) / 256, 256, 0, stream>>>(
      expert_offsets->d_buf, n_experts, d_max_rows);
    CHECK_HIP(
      hipMemcpyAsync(&max_rows_per_expert, d_max_rows, sizeof(int), hipMemcpyDeviceToHost, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    CHECK_HIP(hipFree(d_max_rows));
  }

  {
    constexpr int BM = 16, BN = 128, BK = 16, TM = 2, TN = 8;
    dim3 block_size(BN / TN, BM / TM);
    dim3 grid_size((2 * inter_dim + BN - 1) / BN, (max_rows_per_expert + BM - 1) / BM, n_experts);
    matmul_kernel_bf16_moe<BM, BN, BK, TM, TN><<<grid_size, block_size, 0, stream>>>(
      (const float *)x_packed->d_buf, w1_ptr, (float *)mlp1_out->d_buf, b1_ptr,
      expert_offsets->d_buf, total_pairs, 2 * inter_dim, hidden_dim);
  }

  {
    size_t total = (size_t)batch_size * experts_per_token * inter_dim;
    dim3 block_size(256);
    dim3 grid_size((total + 255) / 256);
    swiglu_interleaved_batched_fast_v2<<<grid_size, block_size, 0, stream>>>(
      (const float *)mlp1_out->d_buf, (float *)gate_up->d_buf, inter_dim,
      batch_size * experts_per_token, clamp_limit);
  }

  {
    constexpr int BM = 16, BN = 128, BK = 16, TM = 2, TN = 8;
    dim3 block_size(BN / TN, BM / TM);
    dim3 grid_size((hidden_dim + BN - 1) / BN, (max_rows_per_expert + BM - 1) / BM, n_experts);
    matmul_kernel_bf16_moe<BM, BN, BK, TM, TN><<<grid_size, block_size, 0, stream>>>(
      (const float *)gate_up->d_buf, w2_ptr, (float *)tb3->d_buf, b2_ptr, expert_offsets->d_buf,
      total_pairs, hidden_dim, inter_dim);
  }

  {
    dim3 block_size(32, 8, 1);
    dim3 grid_size((hidden_dim + block_size.x - 1) / block_size.x,
                   (max_rows_per_expert + block_size.y - 1) / block_size.y, n_experts);
    scale_scatter_add_kernel_sorted_all<<<grid_size, block_size, 0, stream>>>(
      (const float *)tb3->d_buf, (const int *)sorted_pair_ids->d_buf, topk_v_ptr, e_agg_ptr,
      expert_offsets->d_buf, hidden_dim, experts_per_token);
  }

  CHECK_HIP(hipGetLastError());
}