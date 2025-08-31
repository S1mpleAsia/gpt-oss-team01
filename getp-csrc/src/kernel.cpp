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

__global__ void build_expert_offsets_kernel(const int *__restrict__ topk_idx,   // [B*k]
                                            int *__restrict__ sorted_pair_ids,  // [B*k]
                                            int *__restrict__ expert_offsets,   // [E+1]
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

  // reset counts vai trò “cursor”
  for (int i = threadIdx.x; i < n_experts; i += blockDim.x)
    expert_counts[i] = 0;
  __syncthreads();

  for (int pair = threadIdx.x; pair < total_pairs; pair += blockDim.x) {
    int e = topk_idx[pair];
    if (0 <= e && e < n_experts) {
      int pos = expert_offsets[e] + atomicAdd(&expert_counts[e], 1);
      sorted_pair_ids[pos] = pair;  // pair = token_id*k + which_k
    }
  }
}

// 2) Pack X theo sorted_ids: X_pack[batch_size * experts_per_token, hidden_dim ]
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

// 3) Cộng bias bf16 theo cột vào C float [M,N]
__global__ void add_bias_bf16_kernel(float *__restrict__ C, const bf16 *__restrict__ bias,
                                     int rows_M, int cols_N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows_M && col < cols_N) {
    C[row * cols_N + col] += static_cast<float>(bias[col]);
  }
}

// SwiGLU over interleaved [gate, up]; NK=B*k samples
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

  int pos = start_pos + row;   // vị trí trong mảng đã sort (global)
  int pair = sorted_ids[pos];  // pair = b*k + e
  int b = pair / k;
  int e = pair % k;

  float w = topk_v[(size_t)b * k + e];
  float val = z_sorted[(size_t)row * hidden_dim + h] * w;

  // nondeterministic do thứ tự cộng, nhưng đơn giản và nhanh
  atomicAdd(&e_agg[(size_t)b * hidden_dim + h], val);
}

void moe_block_matmul_style(
  Tensor *x_in,         // [batch_size, hidden_dim] (float32)      -- rs->t
  TensorI32 *topk_idx,  // [batch_size, experts_per_token] (int32) -- rs->topk_i
  Tensor *topk_v,       // [batch_size, experts_per_token] (float) -- rs->topk_v
  Tensor *w_mlp1,       // [n_layers, n_experts, hidden_dim, 2*intermediate_dim] (bf16)
  Tensor *b_mlp1,       // [n_layers, n_experts, 2*intermediate_dim]          (bf16)
  Tensor *w_mlp2,       // [n_layers, n_experts, intermediate_dim, hidden_dim] (bf16)
  Tensor *b_mlp2,       // [n_layers, n_experts, hidden_dim]                    (bf16)
  Tensor *e_agg,  // [batch_size, hidden_dim] (float32)       -- rs->e_agg (đã zero trước)
  float clamp_limit, long long layer_offset, hipStream_t stream) {
  // ====== Lấy kích thước ======
  const int batch_size = (int)x_in->shape[0];
  const int hidden_dim = (int)x_in->shape[1];
  const int experts_per_token = (int)topk_idx->shape[1];
  const int n_experts = (int)w_mlp1->shape[1];
  const int inter_dim = (int)w_mlp1->shape[3] / 2;  // vì w_mlp1: [E, K, 2I]
  const int total_pairs = batch_size * experts_per_token;

  // ====== Con trỏ ======
  const float *x_ptr = (const float *)x_in->d_buf;
  const int *topk_idx_ptr = (const int *)topk_idx->d_buf;
  const float *topk_v_ptr = (const float *)topk_v->d_buf;

  const bf16 *w1_ptr = (const bf16 *)w_mlp1->d_buf +
                       (size_t)layer_offset * n_experts * hidden_dim * 2 * inter_dim;  // [E, K, 2I]
  const bf16 *b1_ptr =
    (const bf16 *)b_mlp1->d_buf + (size_t)layer_offset * n_experts * 2 * inter_dim;  // [E, 2I]
  const bf16 *w2_ptr = (const bf16 *)w_mlp2->d_buf +
                       (size_t)layer_offset * n_experts * inter_dim * hidden_dim;  // [E, I, H]
  const bf16 *b2_ptr =
    (const bf16 *)b_mlp2->d_buf + (size_t)layer_offset * n_experts * hidden_dim;  // [E, H]

  float *e_agg_ptr = (float *)e_agg->d_buf;

  // ====== Buffers tạm (device) ======
  TensorI32 sorted_pair_ids({(size_t)total_pairs},
                            x_in->gpu_id);  // [batch_size * experts_per_token]
  TensorI32 expert_offsets({(size_t)(n_experts + 1)}, x_in->gpu_id);  // [n_experts + 1]
  Tensor x_packed({(size_t)total_pairs, (size_t)hidden_dim},
                  x_in->gpu_id);  // [batch_size * experts_per_token, hidden_dim]
  Tensor mlp1_out({(size_t)total_pairs, (size_t)(2 * inter_dim)},
                  x_in->gpu_id);  // [batch_size * experts_per_token, 2 * inter_dim]
  Tensor gate_up({(size_t)total_pairs, (size_t)inter_dim},
                 x_in->gpu_id);  // [batch_size * experts_per_token, inter_dim]
  Tensor tb3({(size_t)total_pairs, (size_t)hidden_dim},
             x_in->gpu_id);  // [B*k, H] (tái sử dụng nếu muốn)

  // Clear output trước khi atomicAdd
  CHECK_HIP(hipMemsetAsync(e_agg_ptr, 0, (size_t)batch_size * hidden_dim * sizeof(float), stream));

  // ====== 1) sort & offsets ======
  {
    GpuTimer timer("moe_offsets");
    const size_t shmem_bytes = (size_t)n_experts * sizeof(int);
    dim3 block_size(256);
    dim3 grid_size(1);

    build_expert_offsets_kernel<<<grid_size, block_size, shmem_bytes, stream>>>(
      topk_idx_ptr, sorted_pair_ids.d_buf, expert_offsets.d_buf, batch_size, experts_per_token,
      n_experts);
  }

  // ====== 2) pack X theo sorted_ids ======
  {
    GpuTimer timer("moe_x_packed");
    dim3 block_size(256, 1);
    dim3 grid_size((hidden_dim + block_size.x - 1) / block_size.x, total_pairs);
    gather_inputs_by_sorted_kernel<<<grid_size, block_size, 0, stream>>>(
      x_ptr, sorted_pair_ids.d_buf, (float *)x_packed.d_buf, batch_size, hidden_dim,
      experts_per_token);
  }

  // Lấy offsets về host để lặp theo expert
  expert_offsets.from_device(stream);
  CHECK_HIP(hipStreamSynchronize(stream));

  // ====== 3) Lặp theo expert: GEMM1 → +bias → SwiGLU → GEMM2 → +bias → Agg ======
  // cấu hình GEMM (giữ cái bạn đã test ổn)
  constexpr int BM = 16, BN = 128, BK = 64, TM = 2, TN = 8;
  const dim3 block_size(BN / TN, BM / TM);
  const int smem_mm = 0;  // kernel của bạn không dùng dynamic shared mem

  for (int expert_id = 0; expert_id < n_experts; ++expert_id) {
    const int start_pair_index = expert_offsets.buf[expert_id];
    const int end_pair_index = expert_offsets.buf[expert_id + 1];
    const int rows_per_expert = end_pair_index - start_pair_index;
    if (rows_per_expert <= 0)
      continue;

    // vùng con theo expert
    const float *x_ptr = (const float *)x_packed.d_buf + (size_t)start_pair_index * hidden_dim;
    const bf16 *w_mlp1_ptr = w1_ptr + (size_t)expert_id * hidden_dim * 2 * inter_dim;
    float *mlp1_out_ptr = (float *)mlp1_out.d_buf + (size_t)start_pair_index * 2 * inter_dim;
    const bf16 *b_mlp1_ptr = b1_ptr + (size_t)expert_id * 2 * inter_dim;

    // --- GEMM1: [rows_per_expert, hidden_dim] × [hidden_dim, 2 * inter_dim] => [rows_per_expert, 2 * inter_dim]
    {
      GpuTimer timer("moe_mlp1");
      dim3 grid((2 * inter_dim + BN - 1) / BN, (rows_per_expert + BM - 1) / BM);
      matmul_kernel_bf16<BM, BN, BK, TM, TN><<<grid, block_size, smem_mm, stream>>>(
        x_ptr, w_mlp1_ptr, mlp1_out_ptr, b_mlp1_ptr, rows_per_expert, 2 * inter_dim, hidden_dim);
    }

    // SwiGLU: [rows_per_expert, 2 * inter_dim] → [rows_per_expert, inter_dim]
    {
      GpuTimer timer("moe_swiglu");
      const size_t total = (size_t)rows_per_expert * inter_dim;
      float *gate_up_ptr = (float *)gate_up.d_buf + (size_t)start_pair_index * (size_t)inter_dim;

      dim3 block_size(256);
      dim3 grid_size((total + block_size.x - 1) / block_size.x);
      swiglu_interleaved_batched_fast_v2<<<grid_size, block_size, 0, stream>>>(
        mlp1_out_ptr, gate_up_ptr, inter_dim, rows_per_expert, clamp_limit);
    }

    // --- GEMM2: [rows_per_expert, inter_dim] × [inter_dim, hidden_dim] → [rows_per_expert, hidden_dim]
    {
      GpuTimer timer("moe_mlp2");
      const float *gate_up_ptr =
        (const float *)gate_up.d_buf + (size_t)start_pair_index * (size_t)inter_dim;
      const bf16 *w_mlp2_ptr = w2_ptr + (size_t)expert_id * (size_t)inter_dim * (size_t)hidden_dim;
      float *tb3_ptr = (float *)tb3.d_buf + (size_t)start_pair_index * (size_t)hidden_dim;
      const bf16 *b_mlp2_ptr = b2_ptr + (size_t)expert_id * (size_t)hidden_dim;

      dim3 grid2((hidden_dim + BN - 1) / BN, (rows_per_expert + BM - 1) / BM);
      matmul_kernel_bf16<BM, BN, BK, TM, TN><<<grid2, block_size, smem_mm, stream>>>(
        gate_up_ptr, w_mlp2_ptr, tb3_ptr, b_mlp2_ptr, rows_per_expert, hidden_dim, inter_dim);
    }

    // --- Agg: scale theo router & scatter-add vào e_agg[B,H] ---
    {
      GpuTimer timer("moe_agg");
      const float *tb3_ptr = (const float *)tb3.d_buf + (size_t)start_pair_index * hidden_dim;

      dim3 block_size(32, 8);
      dim3 grid_size((hidden_dim + block_size.x - 1) / block_size.x,
                     (rows_per_expert + block_size.y - 1) / block_size.y);

      scale_scatter_add_kernel_sorted<<<grid_size, block_size, 0, stream>>>(
        tb3_ptr, (const int *)sorted_pair_ids.d_buf, topk_v_ptr, e_agg_ptr,
        /*start_pos=*/start_pair_index,
        /*rows=*/rows_per_expert,
        /*H=*/hidden_dim,
        /*k=*/experts_per_token);
    }
  }

  CHECK_HIP(hipGetLastError());
}
