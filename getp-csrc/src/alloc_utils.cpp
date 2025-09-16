#include "../include/alloc_utils.hpp"
#include <immintrin.h>
#include <cstdint>

__global__ void transpose_kernel_bf16(const bf16 *__restrict__ d_orig,
                                      bf16 *__restrict__ d_transpose, int h, int w) {
  // Get the global thread indices for the 2D matrix
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // Check if the thread is within the bounds of the original matrix and batch
  if (row < h && col < w) {
    // Calculate the base offset for the current matrix in the batch
    // Calculate the 1D index for the original matrix, including the batch offset
    int orig_idx = row * w + col;
    int transpose_idx = col * h + row;
    bf16 val = d_orig[orig_idx];
    d_transpose[transpose_idx] = val;
  }
}

__global__ void batch_transpose_kernel_bf16(const bf16 *__restrict__ input,
                                            bf16 *__restrict__ output, int batch, int h, int w) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  int batch_idx = blockIdx.z;

  if (batch_idx < batch && row < h && col < w) {
    long long orig_idx = 1ll * batch_idx * h * w + 1ll * row * w + col;

    long long transpose_idx = 1ll * batch_idx * w * h + 1ll * col * h + row;

    bf16 val = input[orig_idx];

    output[transpose_idx] = val;
  }
}

void alloc_w_mlp1_final(OurTransformerWeights *weights_total, float *__restrict__ w_mlp1_ptr,
                        Config *p, hipStream_t *total_streams) {
  CPUTimer timer("alloc_w_mlp1_final");
  printf("Starting alloc mlp1 final\n");
  fflush(stdout);

  size_t layers_per_stage = p->n_layers / PP;
  size_t n_experts = p->n_experts;
  size_t hidden_dim = p->hidden_dim;
  size_t inter_dim = p->intermediate_dim;
  size_t shard_dim = p->intermediate_dim / TP;

  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    CHECK_HIP(hipSetDevice(i));
    (&(weights_total[i]))->w_mlp1 =
      new Tensor({layers_per_stage, n_experts, hidden_dim, 2 * shard_dim}, w_mlp1_ptr,
                 total_streams[i], DType::BF16, false);
  }

  size_t tmp_elems = hidden_dim * 2 * shard_dim;
  size_t batch_tmp_elems = BATCH_MLP1 * tmp_elems;
  size_t size_tmp = 1ll * TOTAL_PIPELINES * BUFFER_MLP1 * batch_tmp_elems;
  bf16 *tmp = (bf16 *)malloc(size_tmp * sizeof(bf16));
  bf16 **d_tmp_arr = new bf16 *[TOTAL_GPUS_NEEDED];
#pragma omp parallel for
  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipMalloc(&d_tmp_arr[i], 1ll * BUFFER_MLP1 * batch_tmp_elems * sizeof(bf16)));
  }

  bf16 **d_buf_array = new bf16 *[TOTAL_GPUS_NEEDED];
  hipStream_t *copy_streams = new hipStream_t[TOTAL_GPUS_NEEDED * BUFFER_MLP1];
  hipEvent_t *copy_dones = new hipEvent_t[TOTAL_GPUS_NEEDED * BUFFER_MLP1];

  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    d_buf_array[i] = (bf16 *)((&(weights_total[i]))->w_mlp1->d_buf);

    CHECK_HIP(hipSetDevice(i));
    for (int j = 0; j < BUFFER_MLP1; j++) {
      CHECK_HIP(hipStreamCreate(&copy_streams[i + j * TOTAL_GPUS_NEEDED]));
      CHECK_HIP(hipEventCreate(&copy_dones[i + j * TOTAL_GPUS_NEEDED]));
    }
  }

  size_t l_offset = 1ll * n_experts * (2 * inter_dim) * hidden_dim;
  size_t e_offset = 1ll * (2 * inter_dim) * hidden_dim;

  size_t l_offset_d = 1ll * n_experts * hidden_dim * (2 * shard_dim);
  size_t e_offset_d = 1ll * hidden_dim * (2 * shard_dim);

  size_t tp_offset = 1ll * e_offset_d;
  size_t pp_offset = 1ll * layers_per_stage * n_experts * hidden_dim * 2 * inter_dim;
  size_t le_slot_offset = 1ll * TOTAL_PIPELINES * batch_tmp_elems;

  size_t base_values[TOTAL_BASE_VALUES_MLP1];
  size_t base_tmp_values[TOTAL_BASE_VALUES_MLP1];

// Loop to populate the new 3D array
#pragma omp parallel for collapse(3)
  for (int pp_rank = 0; pp_rank < PP; pp_rank++) {
    for (int e_sm = 0; e_sm < BATCH_MLP1; e_sm++) {
      for (int tp_rank = 0; tp_rank < TP; tp_rank++) {
        // The new value is the sum of the corresponding values from the original arrays
        int id = (pp_rank * BATCH_MLP1 + e_sm) * TP + tp_rank;
        base_values[id] =
          1ll * pp_rank * pp_offset + 1ll * tp_rank * tp_offset + 1ll * e_sm * e_offset;
        base_tmp_values[id] =
          1ll * (pp_rank * TP + tp_rank) * batch_tmp_elems + 1ll * e_sm * tmp_elems;
      }
    }
  }

  int w = hidden_dim;
  int h = 2 * shard_dim;

  // --- Kernel launch configuration ---
  const int BLOCK_SIZE_X = 16;
  const int BLOCK_SIZE_Y = 16;
  dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  dim3 grid_dim((w + block_dim.x - 1) / block_dim.x, (h + block_dim.y - 1) / block_dim.y,
                BATCH_MLP1);

  for (size_t l = 0; l < layers_per_stage; l++) {
    CPUTimer each_exp_timer("each_exp_timer");
    for (size_t e = 0; e < n_experts; e += BATCH_MLP1) {
      size_t base_out = 1ll * l * l_offset + 1ll * e * e_offset;
      int le_id = (l * n_experts + e) / (BUFFER_MLP1 * BATCH_MLP1);
      int le_slot_id = ((l * n_experts + e) / BATCH_MLP1) % BUFFER_MLP1;
      int le_slot = le_slot_id * TOTAL_GPUS_NEEDED;
      size_t base_tmp_out = 1ll * le_slot_id * le_slot_offset;
      size_t base_tmp_out_each = 1ll * le_slot_id * batch_tmp_elems;

      if (le_id) {
        // CPUTimer sync("event sync");
#pragma omp parallel for
        for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
          CHECK_HIP(hipEventSynchronize(copy_dones[i + le_slot]));
        }
      }

// CPUTimer *convert_timer = new CPUTimer("convert_timer");
#pragma omp parallel for collapse(2)
      for (int idx = 0; idx < TOTAL_BASE_VALUES_MLP1; idx++) {
        for (size_t i = 0; i < 2 * shard_dim; i++) {
#pragma omp simd
          for (size_t h = 0; h < hidden_dim; h++) {
            float value = w_mlp1_ptr[base_out + base_values[idx] + i * hidden_dim + h];
            tmp[base_tmp_out + base_tmp_values[idx] + i * hidden_dim + h] = bf16(value);
          }
        }
      }
      // delete convert_timer;

      size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d;
#pragma omp parallel for
      for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        int tp_rank = i % TP;
        int pp_rank = (i / TP) % PP;
        int pipeline_stage = pp_rank * TP + tp_rank;
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipMemcpyAsync(d_tmp_arr[i] + base_tmp_out_each,
                                 tmp + base_tmp_out + 1ll * pipeline_stage * batch_tmp_elems,
                                 batch_tmp_elems * sizeof(bf16), hipMemcpyHostToDevice,
                                 copy_streams[i + le_slot]));

        // Launch the kernel
        batch_transpose_kernel_bf16<<<grid_dim, block_dim, 0, copy_streams[i + le_slot]>>>(
          d_tmp_arr[i] + base_tmp_out_each, d_buf_array[i] + d_offset, BATCH_MLP1, h, w);

        CHECK_HIP(hipEventRecord(copy_dones[i + le_slot], copy_streams[i + le_slot]));
      }
    }
  }

// final sync
#pragma omp parallel for
  for (int i = 0; i < TOTAL_GPUS_NEEDED * BUFFER_MLP1; i++) {
    CHECK_HIP(hipEventSynchronize(copy_dones[i]));
  }

  free(tmp);

#pragma omp parallel for
  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    CHECK_HIP(hipFree(d_tmp_arr[i]));
  }

#pragma omp parallel for
  for (int i = 0; i < TOTAL_GPUS_NEEDED * BUFFER_MLP1; i++) {
    CHECK_HIP(hipStreamDestroy(copy_streams[i]));
    CHECK_HIP(hipEventDestroy(copy_dones[i]));
  }

  delete[] d_tmp_arr;
  delete[] d_buf_array;
  delete[] copy_streams;
  delete[] copy_dones;

  printf("End alloc mlp1 final\n");
  fflush(stdout);
}

void alloc_w_mlp1_ep(OurTransformerWeights *weights_total, float *__restrict__ w_mlp1_ptr,
                     Config *p, hipStream_t *total_streams) {
  CPUTimer timer("alloc_w_mlp1_ep");
  printf("Starting alloc mlp1 EP\n");
  fflush(stdout);

  size_t layers_per_stage = p->n_layers / PP;
  size_t total_experts = p->n_experts;
  size_t hidden_dim = p->hidden_dim;
  size_t inter_dim = p->intermediate_dim;
  size_t experts_per_gpu = total_experts / TP;

  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    CHECK_HIP(hipSetDevice(i));
    (&(weights_total[i]))->w_mlp1 =
      new Tensor({layers_per_stage, experts_per_gpu, hidden_dim, 2 * inter_dim}, w_mlp1_ptr,
                 total_streams[i], DType::BF16, false);
  }

  size_t expert_elems = 2 * inter_dim * hidden_dim;
  size_t expert_bytes = expert_elems * sizeof(bf16);

  size_t batch_tmp_elems = BATCH_MLP1 * expert_elems;
  size_t batch_tmp_bytes = BATCH_MLP1 * expert_bytes;

  size_t size_tmp = 1ll * TOTAL_PIPELINES * BUFFER_MLP1 * batch_tmp_elems;
  bf16 *h_circular_buffer = (bf16 *)malloc(size_tmp * sizeof(bf16));

  bf16 **d_circular_buffers = new bf16 *[TOTAL_GPUS_NEEDED];
#pragma omp parallel for
  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipMalloc(&d_circular_buffers[i], 1ll * BUFFER_MLP1 * batch_tmp_bytes));
  }

  bf16 **d_dest_ptr = new bf16 *[TOTAL_GPUS_NEEDED];
  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    d_dest_ptr[i] = (bf16 *)((&weights_total[i])->w_mlp1->d_buf);
  }

  hipStream_t *copy_streams = new hipStream_t[TOTAL_GPUS_NEEDED * BUFFER_MLP1];
  hipEvent_t *copy_dones = new hipEvent_t[TOTAL_GPUS_NEEDED * BUFFER_MLP1];

  for (int i = 0; i < TOTAL_GPUS_NEEDED * BUFFER_MLP1; i++) {
    int gpu_id = i % TOTAL_GPUS_NEEDED;
    CHECK_HIP(hipSetDevice(gpu_id));
    CHECK_HIP(hipStreamCreate(&copy_streams[i]));
    CHECK_HIP(hipEventCreate(&copy_dones[i]));
  }

  for (size_t l = 0; l < layers_per_stage; l++) {
    CPUTimer each_exp_timer("each_exp_timer");
    for (size_t e = 0; e < experts_per_gpu; e += BATCH_MLP1) {
      int le_id = (l * experts_per_gpu + e) / (BUFFER_MLP1 * BATCH_MLP1);
      int le_slot_id = ((l * experts_per_gpu + e) / BATCH_MLP1) % BUFFER_MLP1;
      int le_slot_offset = le_slot_id * TOTAL_GPUS_NEEDED;

      size_t h_circular_buf_offset = 1ll * le_slot_id * TOTAL_PIPELINES * batch_tmp_elems;
      size_t d_circular_buf_offset = 1ll * le_slot_id * batch_tmp_elems;

      if (le_id > 0) {
#pragma omp parallel for
        for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
          CHECK_HIP(hipEventSynchronize(copy_dones[i + le_slot_offset]));
        }
      }

#pragma omp parallel for collapse(2)
      for (int gpu_id = 0; gpu_id < TOTAL_GPUS_NEEDED; gpu_id++) {
        for (int e_in_batch = 0; e_in_batch < BATCH_MLP1; e_in_batch++) {
          int pp_rank = gpu_id / TP;
          int tp_rank = gpu_id % TP;  // tp_rank bây giờ là expert_parallel_rank

          // Con trỏ tới expert cụ thể trong file trọng số gốc
          size_t expert_id_in_layer = tp_rank * experts_per_gpu + e + e_in_batch;
          float *src_ptr =
            w_mlp1_ptr +
            1ll * pp_rank * layers_per_stage * total_experts * expert_elems +  // Offset PP
            1ll * l * total_experts * expert_elems +                           // Offset Layer
            1ll * expert_id_in_layer * expert_elems;                           // Offset Expert

          // Con trỏ đích trong circular buffer trên host
          bf16 *dst_ptr_host = h_circular_buffer + h_circular_buf_offset +
                               1ll * gpu_id * batch_tmp_elems + 1ll * e_in_batch * expert_elems;

          // Thực hiện chuyển đổi
          for (size_t i = 0; i < expert_elems; ++i) {
            dst_ptr_host[i] = bf16(src_ptr[i]);
          }
        }
      }

#pragma omp parallel for
      for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        CHECK_HIP(hipSetDevice(i));
        hipStream_t current_stream = copy_streams[i + le_slot_offset];

        // Copy batch expert (chưa chuyển vị) lên buffer tạm của GPU
        CHECK_HIP(
          hipMemcpyAsync(d_circular_buffers[i] + d_circular_buf_offset,
                         h_circular_buffer + h_circular_buf_offset + 1ll * i * batch_tmp_elems,
                         batch_tmp_bytes, hipMemcpyHostToDevice, current_stream));

        // Vị trí đích cuối cùng cho batch expert này
        size_t d_final_offset = 1ll * l * experts_per_gpu * expert_elems + 1ll * e * expert_elems;

        // Kích thước ma trận nguồn
        int src_h = 2 * inter_dim;
        int src_w = hidden_dim;

        const int TILE_DIM = 16;
        dim3 block_dim(TILE_DIM, TILE_DIM, 1);
        dim3 grid_dim((src_w + TILE_DIM - 1) / TILE_DIM, (src_h + TILE_DIM - 1) / TILE_DIM,
                      BATCH_MLP1);

        batch_transpose_kernel_bf16<<<grid_dim, block_dim, 0, current_stream>>>(
          d_circular_buffers[i] + d_circular_buf_offset, d_dest_ptr[i] + d_final_offset, BATCH_MLP1,
          src_h, src_w);

        CHECK_HIP(hipEventRecord(copy_dones[i + le_slot_offset], current_stream));
      }
    }
  }

  for (int i = 0; i < TOTAL_GPUS_NEEDED * BUFFER_MLP1; i++) {
    CHECK_HIP(hipEventSynchronize(copy_dones[i]));
  }

  free(h_circular_buffer);
  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipFree(d_circular_buffers[i]));
  }
  for (int i = 0; i < TOTAL_GPUS_NEEDED * BUFFER_MLP1; i++) {
    int gpu_id = i % TOTAL_GPUS_NEEDED;
    CHECK_HIP(hipSetDevice(gpu_id));
    CHECK_HIP(hipStreamDestroy(copy_streams[i]));
    CHECK_HIP(hipEventDestroy(copy_dones[i]));
  }

  delete[] d_circular_buffers;
  delete[] d_dest_ptr;
  delete[] copy_streams;
  delete[] copy_dones;

  printf("End alloc mlp1 for EP\n");
  fflush(stdout);
}

void alloc_w_mlp2_ep(OurTransformerWeights *weights_total, float *__restrict__ w_mlp2_ptr,
                     Config *p, hipStream_t *total_streams) {
  CPUTimer timer("alloc_w_mlp2_ep");
  printf("Starting alloc mlp2 EP\n");
  fflush(stdout);

  size_t layers_per_stage = p->n_layers / PP;
  size_t total_experts = p->n_experts;
  size_t hidden_dim = p->hidden_dim;
  size_t inter_dim = p->intermediate_dim;
  size_t experts_per_gpu = total_experts / TP;

  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    CHECK_HIP(hipSetDevice(i));
    (&(weights_total[i]))->w_mlp2 =
      new Tensor({layers_per_stage, experts_per_gpu, inter_dim, hidden_dim}, w_mlp2_ptr,
                 total_streams[i], DType::BF16, false);
  }

  size_t single_expert_elems = inter_dim * hidden_dim;
  size_t single_expert_bytes = single_expert_elems * sizeof(bf16);

  size_t batch_tmp_elems = BATCH_MLP2 * single_expert_elems;
  size_t batch_tmp_bytes = BATCH_MLP2 * single_expert_bytes;

  size_t size_tmp = 1ll * TOTAL_PIPELINES * BUFFER_MLP2 * batch_tmp_elems;
  bf16 *h_circular_buffer = (bf16 *)malloc(size_tmp * sizeof(bf16));

  bf16 **d_circular_buffers = new bf16 *[TOTAL_GPUS_NEEDED];
#pragma omp parallel for
  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipMalloc(&d_circular_buffers[i], 1ll * BUFFER_MLP2 * batch_tmp_bytes));
  }

  bf16 **d_final_dest_array = new bf16 *[TOTAL_GPUS_NEEDED];
  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    d_final_dest_array[i] = (bf16 *)((&(weights_total[i]))->w_mlp2->d_buf);
  }

  hipStream_t *copy_streams = new hipStream_t[TOTAL_GPUS_NEEDED * BUFFER_MLP2];
  hipEvent_t *copy_dones = new hipEvent_t[TOTAL_GPUS_NEEDED * BUFFER_MLP2];
  for (int i = 0; i < TOTAL_GPUS_NEEDED * BUFFER_MLP2; i++) {
    int gpu_id = i % TOTAL_GPUS_NEEDED;
    CHECK_HIP(hipSetDevice(gpu_id));
    CHECK_HIP(hipStreamCreate(&copy_streams[i]));
    CHECK_HIP(hipEventCreate(&copy_dones[i]));
  }

  for (size_t l = 0; l < layers_per_stage; l++) {
    CPUTimer each_exp_timer("each_exp_timer");
    for (size_t e_group_start = 0; e_group_start < experts_per_gpu; e_group_start += BATCH_MLP2) {
      int le_id = (l * experts_per_gpu + e_group_start) / (BUFFER_MLP2 * BATCH_MLP2);
      int le_slot_id = ((l * experts_per_gpu + e_group_start) / BATCH_MLP2) % BUFFER_MLP2;
      int le_slot_offset = le_slot_id * TOTAL_GPUS_NEEDED;

      size_t h_circular_buf_offset = 1ll * le_slot_id * TOTAL_PIPELINES * batch_tmp_elems;
      size_t d_circular_buf_offset = 1ll * le_slot_id * batch_tmp_elems;

      if (le_id > 0) {
#pragma omp parallel for
        for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
          CHECK_HIP(hipEventSynchronize(copy_dones[i + le_slot_offset]));
        }
      }

#pragma omp parallel for collapse(2)
      for (int gpu_id = 0; gpu_id < TOTAL_GPUS_NEEDED; gpu_id++) {
        for (int e_in_batch = 0; e_in_batch < BATCH_MLP2; e_in_batch++) {
          int pp_rank = gpu_id / TP;
          int tp_rank = gpu_id % TP;

          size_t expert_id_in_layer = tp_rank * experts_per_gpu + e_group_start + e_in_batch;
          float *src_ptr = w_mlp2_ptr +
                           1ll * pp_rank * layers_per_stage * total_experts * single_expert_elems +
                           1ll * l * total_experts * single_expert_elems +
                           1ll * expert_id_in_layer * single_expert_elems;

          bf16 *dst_ptr_host = h_circular_buffer + h_circular_buf_offset +
                               1ll * gpu_id * batch_tmp_elems +
                               1ll * e_in_batch * single_expert_elems;

          for (size_t i = 0; i < single_expert_elems; ++i) {
            dst_ptr_host[i] = bf16(src_ptr[i]);
          }
        }
      }

#pragma omp parallel for
      for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        CHECK_HIP(hipSetDevice(i));
        hipStream_t current_stream = copy_streams[i + le_slot_offset];

        CHECK_HIP(
          hipMemcpyAsync(d_circular_buffers[i] + d_circular_buf_offset,
                         h_circular_buffer + h_circular_buf_offset + 1ll * i * batch_tmp_elems,
                         batch_tmp_bytes, hipMemcpyHostToDevice, current_stream));

        size_t d_final_offset = 1ll * l * experts_per_gpu * single_expert_elems +
                                1ll * e_group_start * single_expert_elems;

        // Kích thước ma trận nguồn w_mlp2
        int src_h = hidden_dim;
        int src_w = inter_dim;

        const int TILE_DIM = 16;
        dim3 block_dim(TILE_DIM, TILE_DIM, 1);
        dim3 grid_dim((src_w + TILE_DIM - 1) / TILE_DIM, (src_h + TILE_DIM - 1) / TILE_DIM,
                      BATCH_MLP2);

        batch_transpose_kernel_bf16<<<grid_dim, block_dim, 0, current_stream>>>(
          d_circular_buffers[i] + d_circular_buf_offset, d_final_dest_array[i] + d_final_offset,
          BATCH_MLP2, src_h, src_w);

        CHECK_HIP(hipEventRecord(copy_dones[i + le_slot_offset], current_stream));
      }
    }
  }

  for (int i = 0; i < TOTAL_GPUS_NEEDED * BUFFER_MLP2; i++) {
    CHECK_HIP(hipEventSynchronize(copy_dones[i]));
  }
  free(h_circular_buffer);
  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipFree(d_circular_buffers[i]));
  }
  for (int i = 0; i < TOTAL_GPUS_NEEDED * BUFFER_MLP2; i++) {
    int gpu_id = i % TOTAL_GPUS_NEEDED;
    CHECK_HIP(hipSetDevice(gpu_id));
    CHECK_HIP(hipStreamDestroy(copy_streams[i]));
    CHECK_HIP(hipEventDestroy(copy_dones[i]));
  }
  delete[] d_circular_buffers;
  delete[] d_final_dest_array;
  delete[] copy_streams;
  delete[] copy_dones;

  printf("End alloc mlp2 EP\n");
  fflush(stdout);
}

void alloc_w_mlp2_final(OurTransformerWeights *weights_total, float *__restrict__ w_mlp2_ptr,
                        Config *p, hipStream_t *total_streams) {
  CPUTimer timer("alloc_w_mlp2 final");
  printf("Starting alloc mlp2\n");
  fflush(stdout);

  size_t n_layers = p->n_layers / PP;
  size_t n_experts = p->n_experts;
  size_t hidden_dim = p->hidden_dim;
  size_t inter_dim = p->intermediate_dim;
  size_t shard_dim = p->intermediate_dim / TP;

  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    CHECK_HIP(hipSetDevice(i));
    (&(weights_total[i]))->w_mlp2 = new Tensor({n_layers, n_experts, shard_dim, hidden_dim},
                                               w_mlp2_ptr, total_streams[i], DType::BF16, false);
  }

  size_t tmp_elems = shard_dim * hidden_dim;
  size_t batch_tmp_elems = BATCH_MLP2 * tmp_elems;
  size_t size_tmp = 1ll * TOTAL_PIPELINES * BUFFER_MLP2 * batch_tmp_elems;
  bf16 *tmp = (bf16 *)malloc(size_tmp * sizeof(bf16));

  bf16 **d_tmp_arr = new bf16 *[TOTAL_GPUS_NEEDED];
#pragma omp parallel for
  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipMalloc(&d_tmp_arr[i], 1ll * BUFFER_MLP2 * batch_tmp_elems * sizeof(bf16)));
  }

  bf16 **d_buf_array = new bf16 *[TOTAL_GPUS_NEEDED];
  hipStream_t *copy_streams = new hipStream_t[TOTAL_GPUS_NEEDED * BUFFER_MLP2];
  hipEvent_t *copy_dones = new hipEvent_t[TOTAL_GPUS_NEEDED * BUFFER_MLP2];

  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    d_buf_array[i] = (bf16 *)((&(weights_total[i]))->w_mlp2->d_buf);

    CHECK_HIP(hipSetDevice(i));
    for (int j = 0; j < BUFFER_MLP2; j++) {
      CHECK_HIP(hipStreamCreate(&copy_streams[i + j * TOTAL_GPUS_NEEDED]));
      CHECK_HIP(hipEventCreate(&copy_dones[i + j * TOTAL_GPUS_NEEDED]));
    }
  }

  size_t l_offset = 1ll * n_experts * hidden_dim * inter_dim;
  size_t e_offset = 1ll * hidden_dim * inter_dim;

  size_t l_offset_d = 1ll * n_experts * shard_dim * hidden_dim;
  size_t e_offset_d = 1ll * shard_dim * hidden_dim;

  size_t tp_offset = 1ll * shard_dim;
  size_t pp_offset = 1ll * n_layers * n_experts * hidden_dim * inter_dim;
  size_t le_slot_offset = 1ll * TOTAL_PIPELINES * batch_tmp_elems;

  int w = shard_dim;
  int h = hidden_dim;

  // --- Kernel launch configuration ---
  const int BLOCK_SIZE_X = 16;
  const int BLOCK_SIZE_Y = 16;
  dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  dim3 grid_dim((w + block_dim.x - 1) / block_dim.x, (h + block_dim.y - 1) / block_dim.y,
                BATCH_MLP2);

  size_t base_values[TOTAL_BASE_VALUES_MLP2];
  size_t base_tmp_values[TOTAL_BASE_VALUES_MLP2];

// Loop to populate the new 3D array
#pragma omp parallel for collapse(3)
  for (int pp_rank = 0; pp_rank < PP; pp_rank++) {
    for (int e_sm = 0; e_sm < BATCH_MLP2; e_sm++) {
      for (int tp_rank = 0; tp_rank < TP; tp_rank++) {
        // The new value is the sum of the corresponding values from the original arrays
        int id = (pp_rank * BATCH_MLP2 + e_sm) * TP + tp_rank;
        base_values[id] =
          1ll * pp_rank * pp_offset + 1ll * tp_rank * tp_offset + 1ll * e_sm * e_offset;
        base_tmp_values[id] =
          1ll * (pp_rank * TP + tp_rank) * batch_tmp_elems + 1ll * e_sm * tmp_elems;
      }
    }
  }

  for (size_t l = 0; l < n_layers; l++) {
    CPUTimer each_exp_timer("each_exp_timer");
    for (size_t e = 0; e < n_experts; e += BATCH_MLP2) {
      size_t base_out = 1ll * l * l_offset + 1ll * e * e_offset;
      int le_id = (l * n_experts + e) / (BUFFER_MLP2 * BATCH_MLP2);
      int le_slot_id = ((l * n_experts + e) / BATCH_MLP2) % BUFFER_MLP2;
      int le_slot = le_slot_id * TOTAL_GPUS_NEEDED;
      size_t base_tmp_out = 1ll * le_slot_id * le_slot_offset;
      size_t base_tmp_out_each = 1ll * le_slot_id * batch_tmp_elems;

      if (le_id) {
// CPUTimer sync("event sync");
#pragma omp parallel for
        for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
          CHECK_HIP(hipEventSynchronize(copy_dones[i + le_slot]));
        }
      }

// CPUTimer *convert_timer = new CPUTimer("convert_timer");
#pragma omp parallel for collapse(2)
      for (int idx = 0; idx < TOTAL_BASE_VALUES_MLP2; idx++) {
        for (size_t h = 0; h < hidden_dim; h++) {
#pragma omp simd
          for (size_t i = 0; i < shard_dim; i++) {
            float value = w_mlp2_ptr[base_out + base_values[idx] + h * inter_dim + i];
            tmp[base_tmp_out + base_tmp_values[idx] + h * shard_dim + i] = bf16(value);
          }
        }
      }
      // delete convert_timer;

      size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d;
#pragma omp parallel for
      for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        int tp_rank = i % TP;
        int pp_rank = (i / TP) % PP;
        int pipeline_stage = pp_rank * TP + tp_rank;
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipMemcpyAsync(d_tmp_arr[i] + base_tmp_out_each,
                                 tmp + base_tmp_out + 1ll * pipeline_stage * batch_tmp_elems,
                                 batch_tmp_elems * sizeof(bf16), hipMemcpyHostToDevice,
                                 copy_streams[i + le_slot]));

        // Launch the kernel
        batch_transpose_kernel_bf16<<<grid_dim, block_dim, 0, copy_streams[i + le_slot]>>>(
          d_tmp_arr[i] + base_tmp_out_each, d_buf_array[i] + d_offset, BATCH_MLP2, h, w);

        CHECK_HIP(hipEventRecord(copy_dones[i + le_slot], copy_streams[i + le_slot]));
      }
    }
  }

// final sync
#pragma omp parallel for
  for (int i = 0; i < TOTAL_GPUS_NEEDED * BUFFER_MLP2; i++) {
    CHECK_HIP(hipEventSynchronize(copy_dones[i]));
  }

  free(tmp);

#pragma omp parallel for
  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    CHECK_HIP(hipFree(d_tmp_arr[i]));
  }

#pragma omp parallel for
  for (int i = 0; i < TOTAL_GPUS_NEEDED * BUFFER_MLP2; i++) {
    CHECK_HIP(hipStreamDestroy(copy_streams[i]));
    CHECK_HIP(hipEventDestroy(copy_dones[i]));
  }

  delete[] d_tmp_arr;
  delete[] d_buf_array;
  delete[] copy_streams;
  delete[] copy_dones;

  printf("End alloc mlp2 final\n");
  fflush(stdout);
}

void alloc_w_mlp2(Tensor *&w_mlp2, float *w_mlp2_ptr, Config *p, int device_id,
                  hipStream_t stream) {
  CPUTimer timer("alloc_w_mlp2");
  printf("Starting alloc mlp2\n");
  fflush(stdout);

  int tp_rank = device_id % TP;
  size_t n_layers = p->n_layers / PP;
  size_t n_experts = p->n_experts;
  size_t hidden_dim = p->hidden_dim;
  size_t inter_dim = p->intermediate_dim;
  size_t shard_dim = p->intermediate_dim / TP;

  w_mlp2_ptr += 1ll * tp_rank * shard_dim;  // offset shard_dim;

  w_mlp2 = new Tensor({n_layers, n_experts, shard_dim, hidden_dim}, w_mlp2_ptr, stream, DType::BF16,
                      false);

  size_t tmp_elems = shard_dim * hidden_dim;
  bf16 *tmp = (bf16 *)malloc(tmp_elems * sizeof(bf16));

  bf16 *d_buf = (bf16 *)(w_mlp2->d_buf);

  size_t l_offset = 1ll * n_experts * hidden_dim * inter_dim;
  size_t e_offset = 1ll * hidden_dim * inter_dim;

  size_t l_offset_d = 1ll * n_experts * shard_dim * hidden_dim;
  size_t e_offset_d = 1ll * shard_dim * hidden_dim;

  for (size_t l = 0; l < n_layers; l++) {
    for (size_t e = 0; e < n_experts; e++) {
      size_t base = 1ll * l * l_offset + 1ll * e * e_offset;

      for (size_t i = 0; i < shard_dim; i++) {
        for (size_t h = 0; h < hidden_dim; h++) {
          float value = w_mlp2_ptr[base + h * inter_dim + i];
          tmp[i * hidden_dim + h] = bf16(value);
        }
      }

      size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d;
      CHECK_HIP(
        hipMemcpyAsync(d_buf + d_offset, tmp, tmp_elems * sizeof(bf16), hipMemcpyHostToDevice, 0));
    }
  }

  CHECK_HIP(hipStreamSynchronize(stream));
  free(tmp);

  // w_mlp2->to_device(0);
  printf("End alloc mlp2\n");
  fflush(stdout);
}

void alloc_out_new(Tensor *&out, float *w_out, Config *p, int device_id, hipStream_t stream) {
  CPUTimer timer("alloc_out_new");
  printf("Starting alloc out...\n");
  fflush(stdout);
  size_t vocab_size = p->vocab_size;
  size_t hidden_dim = p->hidden_dim;

  size_t tmp_elems = 1ll * hidden_dim * vocab_size;

  bf16 *tmp = (bf16 *)malloc(tmp_elems * sizeof(bf16));

  out = new Tensor({hidden_dim, vocab_size}, w_out, stream, DType::BF16, false);

  bf16 *d_buf = (bf16 *)out->d_buf;

  for (size_t i = 0; i < vocab_size; i++) {
    for (size_t j = 0; j < hidden_dim; j++) {
      tmp[j * vocab_size + i] = bf16(w_out[i * hidden_dim + j]);
    }
  }

  CHECK_HIP(hipMemcpy(d_buf, tmp, tmp_elems * sizeof(bf16), hipMemcpyHostToDevice));
  CHECK_HIP(hipStreamSynchronize(stream));
  free(tmp);

  printf("End alloc out\n");
  fflush(stdout);
}

void alloc_out(Tensor *&out, float *w_out, Config *p, int device_id, hipStream_t stream) {
  CPUTimer timer("alloc_out");
  printf("Starting alloc out...\n");
  fflush(stdout);
  size_t vocab_size = p->vocab_size;
  size_t hidden_dim = p->hidden_dim;

  size_t tmp_elems = 1ll * hidden_dim * vocab_size;

  bf16 *tmp = (bf16 *)malloc(tmp_elems * sizeof(bf16));

  out = new Tensor({hidden_dim, vocab_size}, w_out, stream, DType::BF16, false);

  bf16 *d_buf = (bf16 *)out->d_buf;

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < vocab_size; i++) {
    for (size_t j = 0; j < hidden_dim; j++) {
      tmp[j * vocab_size + i] = bf16(w_out[i * hidden_dim + j]);
    }
  }

  CHECK_HIP(hipMemcpy(d_buf, tmp, tmp_elems * sizeof(bf16), hipMemcpyHostToDevice));
  CHECK_HIP(hipStreamSynchronize(stream));
  free(tmp);

  printf("End alloc out\n");
  fflush(stdout);
}
