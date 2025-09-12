#include "../include/alloc_utils.hpp"
#include <immintrin.h>
#include <cstdint>

__global__ void transpose_kernel_bf16(
    const bf16* __restrict__ d_orig,
    bf16* __restrict__ d_transpose, int h, int w
) {
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

__global__ void batch_transpose_kernel_bf16(
    const bf16* __restrict__ d_orig,
    bf16* __restrict__ d_transpose,
    int batch, int h, int w
) {
    // Get the global thread indices for the 2D matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Get the global thread index for the batch
    int batch_idx = blockIdx.z;

    // Check if the thread is within the bounds of the original matrix and batch
    if (batch_idx < batch && row < h && col < w) {
        // Calculate the base offset for the current matrix in the batch
        // Calculate the 1D index for the original matrix, including the batch offset
        long long orig_idx = 1ll * batch_idx * h * w + 1ll * row * w + col;

        // Calculate the 1D index for the transposed matrix, including the batch offset.
        // The transposed matrix has dimensions (w, h), and the element at (row, col)
        // in the original matrix moves to (col, row) in the transposed matrix.
        long long transpose_idx = 1ll * batch_idx * w * h + 1ll * col * h + row;

        // Read the bf16 value from the original matrix
        bf16 val = d_orig[orig_idx];

        // Store the bf16 value in the transposed matrix
        d_transpose[transpose_idx] = val;
    }
}

void alloc_w_mlp1_final(
    OurTransformerWeights *weights_total, float *w_mlp1_ptr, Config *p,
    int start_layer, int start_moe
) {
    CPUTimer timer("alloc_w_mlp1_final");
    printf("Starting alloc mlp1 final\n");
    fflush(stdout);

    size_t n_layers = p->n_layers / PP;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;
    size_t shard_dim = p->intermediate_dim / TP;

    if (start_layer == 0 && start_moe == 0) {
        for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
            (&(weights_total[i]))->w_mlp1 = new Tensor({n_layers, n_experts, hidden_dim, 2 * shard_dim}, w_mlp1_ptr, i, false, DType::BF16);
        }
    }

    size_t tmp_elems = hidden_dim * 2 * shard_dim;
    size_t size_tmp = 1ll * TOTAL_PIPELINES * BUFFER_MLP1 * BATCH_MLP1 * tmp_elems;
    bf16 *tmp = (bf16 *)malloc(size_tmp * sizeof(bf16));
    bf16 **d_tmp_arr = new bf16*[TOTAL_GPUS_NEEDED];
    #pragma omp parallel for
    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipMalloc(&d_tmp_arr[i], 1ll * BUFFER_MLP1 * BATCH_MLP1 * tmp_elems * sizeof(bf16)));
    }

    bf16** d_buf_array = new bf16*[TOTAL_GPUS_NEEDED];
    hipStream_t *copy_streams = new hipStream_t[TOTAL_GPUS_NEEDED * BUFFER_MLP1];
    hipStream_t *transpose_streams = new hipStream_t[TOTAL_GPUS_NEEDED * BUFFER_MLP1];
    hipEvent_t *copy_dones = new hipEvent_t[TOTAL_GPUS_NEEDED * BUFFER_MLP1];

    int end_layer = start_layer + OFFSET_LAYER;
    int end_moe = start_moe + OFFSET_MOE;

    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        d_buf_array[i] = (bf16 *)((&(weights_total[i]))->w_mlp1->d_buf);

        CHECK_HIP(hipSetDevice(i));
        for (int j = 0; j < BUFFER_MLP1; j++) {
            CHECK_HIP(hipStreamCreate(&copy_streams[i + j * TOTAL_GPUS_NEEDED]));
            CHECK_HIP(hipStreamCreate(&transpose_streams[i + j * TOTAL_GPUS_NEEDED]));
            CHECK_HIP(hipEventCreate(&copy_dones[i + j * TOTAL_GPUS_NEEDED]));
        }
    }

    size_t l_offset = 1ll * n_experts * (2 * inter_dim) * hidden_dim;
    size_t e_offset = 1ll * (2 * inter_dim) * hidden_dim;

    size_t l_offset_d = 1ll * n_experts * hidden_dim * (2 * shard_dim);
    size_t e_offset_d = 1ll * hidden_dim * (2 * shard_dim);

    size_t tp_offset = 1ll * e_offset_d;
    size_t pp_offset = 1ll * n_layers * n_experts * hidden_dim * 2 * inter_dim;
    size_t le_slot_offset = 1ll * TOTAL_PIPELINES * BATCH_MLP1 * tmp_elems;

    int w = hidden_dim;
    int h = 2 * shard_dim;
    // CHECK_HIP(hipStreamWaitEvent(transpose_streams[i + le_slot], copy_dones[i + le_slot]));
    
    // --- Kernel launch configuration ---
    const int BLOCK_SIZE_X = 16;
    const int BLOCK_SIZE_Y = 16;
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 grid_dim(
        (w + block_dim.x - 1) / block_dim.x,
        (h + block_dim.y - 1) / block_dim.y,
        BATCH_MLP1
    );

    for (size_t l = start_layer; l < end_layer; l++) {
        for (size_t e = start_moe; e < end_moe; e += BATCH_MLP1) {
            size_t base_out = 1ll * l * l_offset + 1ll * e * e_offset;
            int le_id = (l * n_experts + e) / (BUFFER_MLP1 * BATCH_MLP1);
            int le_slot_id = ((l * n_experts + e) / BATCH_MLP1) % BUFFER_MLP1;
            int le_slot = le_slot_id * TOTAL_GPUS_NEEDED;
            size_t base_tmp_out = 1ll * le_slot_id * le_slot_offset;
            size_t base_tmp_out_each = 1ll * le_slot_id * BATCH_MLP1 * tmp_elems;

            if (le_id) {
                // CPUTimer sync("event sync");
                #pragma omp parallel for
                for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
                    CHECK_HIP(hipEventSynchronize(copy_dones[i + le_slot]));
                }
            }

            // CPUTimer *convert_timer = new CPUTimer("convert_timer");
            #pragma omp parallel for collapse(3)
            for (int tp_rank = 0; tp_rank < TP; tp_rank++) {
                for (int pp_rank = 0; pp_rank < PP; pp_rank++) {
                    for (int e_sm = 0; e_sm < BATCH_MLP1; e_sm++) {
                        size_t base = base_out + 1ll * pp_rank * pp_offset + 1ll * tp_rank * tp_offset + 1ll * e_sm * e_offset;
                        size_t base_tmp = base_tmp_out + 1ll * (pp_rank * TP + tp_rank) * BATCH_MLP1 * tmp_elems + 1ll * e_sm * tmp_elems;
                        #pragma omp simd
                        for (size_t i = 0; i < 2ll * shard_dim * hidden_dim; i++) {
                            float value = w_mlp1_ptr[base + i];
                            tmp[base_tmp + i] = bf16(value);
                        }
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
                CHECK_HIP(hipMemcpyAsync(
                    d_tmp_arr[i] + base_tmp_out_each,
                    tmp + base_tmp_out + 1ll * pipeline_stage * BATCH_MLP1 * tmp_elems, 
                    BATCH_MLP1 * tmp_elems * sizeof(bf16),
                    hipMemcpyHostToDevice, copy_streams[i + le_slot]
                ));

                // Launch the kernel
                batch_transpose_kernel_bf16<<<grid_dim, block_dim, 0, copy_streams[i + le_slot]>>>(
                    d_tmp_arr[i] + base_tmp_out_each,
                    d_buf_array[i] + d_offset, BATCH_MLP1, h, w
                );
                    
                CHECK_HIP(hipEventRecord(copy_dones[i + le_slot], copy_streams[i + le_slot]));
            }
        }
    }

    // final sync
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        for (int j = 0; j < BUFFER_MLP1; j++) {
            CHECK_HIP(hipEventSynchronize(copy_dones[i + j * TOTAL_GPUS_NEEDED]));
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        for (int j = 0; j < BUFFER_MLP1; j++) {
            // CHECK_HIP(hipStreamSynchronize(transpose_streams[i + j * TOTAL_GPUS_NEEDED]));
        }
    }
    
    free(tmp);

    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        CHECK_HIP(hipFree(d_tmp_arr[i]));
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        for (int j = 0; j < BUFFER_MLP1; j++) {
            CHECK_HIP(hipStreamDestroy(copy_streams[i + j * TOTAL_GPUS_NEEDED]));
            CHECK_HIP(hipStreamDestroy(transpose_streams[i + j * TOTAL_GPUS_NEEDED]));
            CHECK_HIP(hipEventDestroy(copy_dones[i + j * TOTAL_GPUS_NEEDED]));
        }
    }
    
    delete[] d_tmp_arr;
    delete[] d_buf_array;
    delete[] copy_streams;
    delete[] transpose_streams;
    delete[] copy_dones;

    printf("End alloc mlp1 final\n");
    fflush(stdout);
}

void alloc_w_mlp1(
    Tensor* &w_mlp1, float *w_mlp1_ptr, Config *p, int device_id
) {
    CPUTimer timer("alloc_w_mlp1");
    printf("Starting alloc mlp1\n");
    fflush(stdout);

    int tp_rank = device_id % TP;
    size_t n_layers = p->n_layers / PP;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;
    size_t shard_dim = p->intermediate_dim / TP;

    w_mlp1_ptr += tp_rank * (2 * shard_dim) * hidden_dim; // offset shard_dim;

    w_mlp1 = new Tensor({n_layers, n_experts, hidden_dim, 2 * shard_dim}, w_mlp1_ptr, device_id, false, DType::BF16);

    size_t tmp_elems = hidden_dim * 2 * shard_dim;
    bf16 *tmp = (bf16 *)malloc(tmp_elems * sizeof(bf16));

    bf16 *d_buf = (bf16 *)(w_mlp1->d_buf);

    size_t l_offset = 1ll * n_experts * (2 * inter_dim) * hidden_dim;
    size_t e_offset = 1ll * (2 * inter_dim) * hidden_dim;

    size_t l_offset_d = 1ll * n_experts * hidden_dim * (2 * shard_dim);
    size_t e_offset_d = 1ll * hidden_dim * (2 * shard_dim);

    for (size_t l = 0; l < n_layers; l++) {
        for (size_t e = 0; e < n_experts; e++) {
            size_t base = 1ll * l * l_offset + 1ll * e * e_offset;

            for (size_t h = 0; h < hidden_dim; h++) {
                for (size_t i = 0; i < 2 * shard_dim; i++) {
                    float value = w_mlp1_ptr[base + i * hidden_dim + h];
                    tmp[h * 2 * shard_dim + i] = bf16(value);
                }
            }

            size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d;
            CHECK_HIP(hipMemcpyAsync(d_buf + d_offset, tmp, tmp_elems * sizeof(bf16),
                                    hipMemcpyHostToDevice, 0));
        }
    }

    CHECK_HIP(hipStreamSynchronize(0));
    free(tmp);

    printf("End alloc mlp1\n");
    fflush(stdout);
}

void alloc_w_mlp2_final(
    OurTransformerWeights *weights_total, float *w_mlp2_ptr, Config *p,
    int start_layer, int start_moe
) {
    CPUTimer timer("alloc_w_mlp2 final");
    printf("Starting alloc mlp2\n");
    fflush(stdout);

    size_t n_layers = p->n_layers / PP;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;
    size_t shard_dim = p->intermediate_dim / TP;

    if (start_layer == 0 && start_moe == 0) {
        for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
            (&(weights_total[i]))->w_mlp2 = new Tensor({n_layers, n_experts, shard_dim, hidden_dim}, w_mlp2_ptr, i, false, DType::BF16);
        }
    }

    size_t tmp_elems = shard_dim * hidden_dim;
    size_t size_tmp = 1ll * TOTAL_PIPELINES * BUFFER_MLP2 * BATCH_MLP2 * tmp_elems;
    bf16 *tmp = (bf16 *)malloc(size_tmp * sizeof(bf16));
    
    bf16 **d_tmp_arr = new bf16*[TOTAL_GPUS_NEEDED];
    #pragma omp parallel for
    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipMalloc(&d_tmp_arr[i], 1ll * BUFFER_MLP2 * BATCH_MLP2 * tmp_elems * sizeof(bf16)));
    }

    bf16** d_buf_array = new bf16*[TOTAL_GPUS_NEEDED];
    hipStream_t *copy_streams = new hipStream_t[TOTAL_GPUS_NEEDED * BUFFER_MLP2];
    hipStream_t *transpose_streams = new hipStream_t[TOTAL_GPUS_NEEDED * BUFFER_MLP2];
    hipEvent_t *copy_dones = new hipEvent_t[TOTAL_GPUS_NEEDED * BUFFER_MLP2];

    int end_layer = start_layer + OFFSET_LAYER;
    int end_moe = start_moe + OFFSET_MOE;

    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        d_buf_array[i] = (bf16 *)((&(weights_total[i]))->w_mlp2->d_buf);

        CHECK_HIP(hipSetDevice(i));
        for (int j = 0; j < BUFFER_MLP2; j++) {
            CHECK_HIP(hipStreamCreate(&copy_streams[i + j * TOTAL_GPUS_NEEDED]));
            CHECK_HIP(hipStreamCreate(&transpose_streams[i + j * TOTAL_GPUS_NEEDED]));
            CHECK_HIP(hipEventCreate(&copy_dones[i + j * TOTAL_GPUS_NEEDED]));
        }
    }

    size_t l_offset = 1ll * n_experts * hidden_dim * inter_dim;
    size_t e_offset = 1ll * hidden_dim * inter_dim;

    size_t l_offset_d = 1ll * n_experts * shard_dim * hidden_dim;
    size_t e_offset_d = 1ll * shard_dim * hidden_dim;

    size_t tp_offset = 1ll * shard_dim;
    size_t pp_offset = 1ll * n_layers * n_experts * hidden_dim * inter_dim;
    size_t le_slot_offset = 1ll * TOTAL_PIPELINES * BATCH_MLP2 * tmp_elems;

    int w = shard_dim;
    int h = hidden_dim;
    // CHECK_HIP(hipStreamWaitEvent(transpose_streams[i + le_slot], copy_dones[i + le_slot]));
    
    // --- Kernel launch configuration ---
    const int BLOCK_SIZE_X = 16;
    const int BLOCK_SIZE_Y = 16;
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 grid_dim(
        (w + block_dim.x - 1) / block_dim.x,
        (h + block_dim.y - 1) / block_dim.y,
        BATCH_MLP2
    );

    for (size_t l = start_layer; l < end_layer; l++) {
        for (size_t e = start_moe; e < end_moe; e += BATCH_MLP2) {
            size_t base_out = 1ll * l * l_offset + 1ll * e * e_offset;
            int le_id = (l * n_experts + e) / (BUFFER_MLP2 * BATCH_MLP2);
            int le_slot_id = ((l * n_experts + e) / BATCH_MLP2) % BUFFER_MLP2;
            int le_slot = le_slot_id * TOTAL_GPUS_NEEDED;
            size_t base_tmp_out = 1ll * le_slot_id * le_slot_offset;
            size_t base_tmp_out_each = 1ll * le_slot_id * BATCH_MLP2 * tmp_elems;

            if (le_id) {
                // CPUTimer sync("event sync");
                #pragma omp parallel for
                for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
                    CHECK_HIP(hipEventSynchronize(copy_dones[i + le_slot]));
                }
            }

            // CPUTimer *convert_timer = new CPUTimer("convert_timer");
            // #pragma omp parallel for collapse(3)
            for (int tp_rank = 0; tp_rank < TP; tp_rank++) {
                for (int pp_rank = 0; pp_rank < PP; pp_rank++) {
                    for (int e_sm = 0; e_sm < BATCH_MLP2; e_sm++) {
                        size_t base = base_out + 1ll * pp_rank * pp_offset + 1ll * tp_rank * tp_offset + 1ll * e_sm * e_offset;
                        size_t base_tmp = base_tmp_out + 1ll * (pp_rank * TP + tp_rank) * BATCH_MLP2 * tmp_elems + 1ll * e_sm * tmp_elems;
                        #pragma omp parallel for
                        for (size_t h = 0; h < hidden_dim; h++) {
                            #pragma omp simd
                            for (size_t i = 0; i < shard_dim; i++) {
                                float value = w_mlp2_ptr[base_out + h * inter_dim + i];
                                tmp[base_tmp + h * shard_dim + i] = bf16(value);
                            }
                        }
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
                CHECK_HIP(hipMemcpyAsync(
                    d_tmp_arr[i] + base_tmp_out_each,
                    tmp + base_tmp_out + 1ll * pipeline_stage * BATCH_MLP2 * tmp_elems, 
                    BATCH_MLP2 * tmp_elems * sizeof(bf16),
                    hipMemcpyHostToDevice, copy_streams[i + le_slot]
                ));

                // Launch the kernel
                batch_transpose_kernel_bf16<<<grid_dim, block_dim, 0, copy_streams[i + le_slot]>>>(
                    d_tmp_arr[i] + base_tmp_out_each,
                    d_buf_array[i] + d_offset, BATCH_MLP2, h, w
                );
                    
                CHECK_HIP(hipEventRecord(copy_dones[i + le_slot], copy_streams[i + le_slot]));
            }
        }
    }

    // final sync
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        for (int j = 0; j < BUFFER_MLP2; j++) {
            CHECK_HIP(hipEventSynchronize(copy_dones[i + j * TOTAL_GPUS_NEEDED]));
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        for (int j = 0; j < BUFFER_MLP2; j++) {
            // CHECK_HIP(hipStreamSynchronize(transpose_streams[i + j * TOTAL_GPUS_NEEDED]));
        }
    }
    
    free(tmp);

    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        CHECK_HIP(hipFree(d_tmp_arr[i]));
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        for (int j = 0; j < BUFFER_MLP2; j++) {
            CHECK_HIP(hipStreamDestroy(copy_streams[i + j * TOTAL_GPUS_NEEDED]));
            CHECK_HIP(hipStreamDestroy(transpose_streams[i + j * TOTAL_GPUS_NEEDED]));
            CHECK_HIP(hipEventDestroy(copy_dones[i + j * TOTAL_GPUS_NEEDED]));
        }
    }
    
    delete[] d_tmp_arr;
    delete[] d_buf_array;
    delete[] copy_streams;
    delete[] transpose_streams;
    delete[] copy_dones;

    printf("End alloc mlp1 final\n");
    fflush(stdout);
}

void alloc_w_mlp2(
    Tensor* &w_mlp2, float *w_mlp2_ptr, Config *p, int device_id
) {
    CPUTimer timer("alloc_w_mlp2");
    printf("Starting alloc mlp2\n");
    fflush(stdout);

    int tp_rank = device_id % TP;
    size_t n_layers = p->n_layers / PP;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;
    size_t shard_dim = p->intermediate_dim / TP;

    w_mlp2_ptr += 1ll * tp_rank * shard_dim; // offset shard_dim;

    w_mlp2 = new Tensor({n_layers, n_experts, shard_dim, hidden_dim}, w_mlp2_ptr, device_id, false, DType::BF16);

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
            CHECK_HIP(hipMemcpyAsync(d_buf + d_offset, tmp, tmp_elems * sizeof(bf16),
                                    hipMemcpyHostToDevice, 0));
        }
    }

    CHECK_HIP(hipStreamSynchronize(0));
    free(tmp);

    // w_mlp2->to_device(0);
    printf("End alloc mlp2\n");
    fflush(stdout);
}

void alloc_out_new(
    Tensor* &out, float *w_out, Config *p, int device_id
) {
    CPUTimer timer("alloc_out_new");
    printf("Starting alloc out...\n");
    fflush(stdout);
    size_t vocab_size = p->vocab_size;
    size_t hidden_dim = p->hidden_dim;

    size_t tmp_elems = 1ll * hidden_dim * vocab_size;

    bf16 *tmp = (bf16 *)malloc(tmp_elems * sizeof(bf16));

    out = new Tensor({hidden_dim, vocab_size}, w_out, device_id, false, DType::BF16);

    bf16 *d_buf = (bf16 *)out->d_buf;

    for (size_t i = 0; i < vocab_size; i++) {
        for (size_t j = 0; j < hidden_dim; j++) {
            tmp[j * vocab_size + i] = bf16(w_out[i * hidden_dim + j]);
        }
    }

    CHECK_HIP(hipMemcpy(d_buf, tmp, tmp_elems * sizeof(bf16),
                            hipMemcpyHostToDevice));
    CHECK_HIP(hipStreamSynchronize(0));
    free(tmp);
    
    printf("End alloc out\n");
    fflush(stdout);
}

void alloc_out(
    Tensor* &out, float *w_out, Config *p, int device_id
) {
    CPUTimer timer("alloc_out");
    printf("Starting alloc out...\n");
    fflush(stdout);
    size_t vocab_size = p->vocab_size;
    size_t hidden_dim = p->hidden_dim;

    size_t tmp_elems = 1ll * hidden_dim * vocab_size;

    bf16 *tmp = (bf16 *)malloc(tmp_elems * sizeof(bf16));

    out = new Tensor({hidden_dim, vocab_size}, w_out, device_id, false, DType::BF16);

    bf16 *d_buf = (bf16 *)out->d_buf;

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < vocab_size; i++) {
        for (size_t j = 0; j < hidden_dim; j++) {
            tmp[j * vocab_size + i] = bf16(w_out[i * hidden_dim + j]);
        }
    }

    CHECK_HIP(hipMemcpy(d_buf, tmp, tmp_elems * sizeof(bf16),
                            hipMemcpyHostToDevice));
    CHECK_HIP(hipStreamSynchronize(0));
    free(tmp);
    
    printf("End alloc out\n");
    fflush(stdout);
}
