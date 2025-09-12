#include "../include/alloc_utils.hpp"
#include <immintrin.h>
#include <cstdint>

__global__ void transpose_kernel(
    const float* __restrict__ d_orig,
    bf16* __restrict__ d_transpose, int h, int w
) {
    // Get the global thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the bounds of the original matrix
    if (row < h && col < w) {
        // Calculate the 1D index for the original matrix
        int orig_idx = row * w + col;
        
        // Calculate the 1D index for the transposed matrix.
        // The transposed matrix dimensions are (w, h), and the element at (row, col)
        // in the original matrix moves to (col, row) in the transposed matrix.
        int transpose_idx = col * h + row;

        // Read the float value from the original matrix
        float val = d_orig[orig_idx];

        // Convert the float value to bfloat16.
        // The __hip_convert_f32_to_bf16() intrinsic is used for this.
        bf16 bf16_val = bf16(val);

        // Store the bfloat16 value in the transposed matrix
        d_transpose[transpose_idx] = bf16_val;
    }
}

__global__ void partial_transpose_kernel(
    const float* __restrict__ d_orig,
    bf16* __restrict__ d_transpose,
    int h, int w, int start_h, int total_h
) {
    // Calculate the global row and column indices for the portion of the matrix we are processing.
    // The thread index `y` is relative to the `total_h` block.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the bounds of the specified portion.
    if (x < w && y < total_h) {
        // Calculate the index in the original matrix.
        // We add `start_h` to `y` to get the correct row index in the full matrix.
        int orig_idx = (y + start_h) * w + x;

        // Calculate the corresponding index in the transposed matrix.
        // The transpose operation swaps rows and columns.
        int trans_idx = x * h + (y + start_h);

        // Perform the transpose and convert the float value to bfloat16.
        d_transpose[trans_idx] = __float2bfloat16(d_orig[orig_idx]);
    }
}

__global__ void batch_transpose_kernel(
    const float* __restrict__ d_orig,
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
        long long orig_idx = 1ll * batch_idx * w * h + 1ll * row * w + col;

        // Calculate the 1D index for the transposed matrix, including the batch offset.
        // The transposed matrix has dimensions (w, h), and the element at (row, col)
        // in the original matrix moves to (col, row) in the transposed matrix.
        long long transpose_idx = 1ll * batch_idx * w * h + 1ll * col * h + 1ll * row;

        // Read the float value from the original matrix
        float val = d_orig[orig_idx];

        // Store the bf16 value in the transposed matrix
        d_transpose[transpose_idx] = bf16(val);
    }
}

void alloc_w_mlp1_transpose(
    Tensor* &w_mlp1, float *w_mlp1_ptr, Config *p, int device_id
) {
    CPUTimer timer("alloc_w_mlp1_transpose");
    printf("Starting alloc mlp1 new\n");
    fflush(stdout);

    int tp_rank = device_id % TP;
    size_t n_layers = p->n_layers / PP;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;
    size_t shard_dim = p->intermediate_dim / TP;

    w_mlp1_ptr += tp_rank * (2 * shard_dim) * hidden_dim; // offset shard_dim;

    w_mlp1 = new Tensor({n_layers, n_experts, hidden_dim, 2 * shard_dim}, w_mlp1_ptr, device_id, false, DType::BF16);

    hipStream_t *copy_streams = new hipStream_t[BUFFER_MLP1];
    for (int i = 0; i < BUFFER_MLP1; i++) {   
        CHECK_HIP(hipStreamCreate(&copy_streams[i]));
    }

    float *d_tmp = nullptr;
    size_t tmp_elems = (2 * shard_dim) * hidden_dim;
    CHECK_HIP(hipMalloc(&d_tmp, BUFFER_MLP1 * BATCH_MLP1 * tmp_elems * sizeof(float)));

    bf16 *d_buf = (bf16 *)(w_mlp1->d_buf);

    size_t l_offset = 1ll * n_experts * (2 * inter_dim) * hidden_dim;
    size_t e_offset = 1ll * (2 * inter_dim) * hidden_dim;

    size_t l_offset_d = 1ll * n_experts * hidden_dim * (2 * shard_dim);
    size_t e_offset_d = 1ll * hidden_dim * (2 * shard_dim);

    for (size_t l = 0; l < n_layers; l++) {
        for (size_t e = 0; e < n_experts; e += BATCH_MLP1) {
            size_t base = 1ll * l * l_offset + 1ll * e * e_offset;
            size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d;
            int stream_id = ((l * n_experts + e) / BATCH_MLP1) % BUFFER_MLP1;

            for (size_t e_sm = 0; e_sm < BATCH_MLP1; e_sm ++) {
                int slot_id = stream_id * BATCH_MLP1 + e_sm;
                size_t base_now = base + 1ll * e_sm * e_offset;
                // copy l, e shard to gpu using stream id, offset l_offset and e_offset
                CHECK_HIP(hipMemcpyAsync(
                    d_tmp + slot_id * tmp_elems, // destination on device
                    w_mlp1_ptr + base_now, // source on host
                    tmp_elems * sizeof(float), // size of data to copy
                    hipMemcpyHostToDevice, // direction
                    copy_streams[stream_id] // stream
                ));
            }

            /*
            printf("Layer l=%zu, e=%zu\n", l, e);
            fflush(stdout);
            */

            dim3 blockDim(16, 16);
            // Grid dimensions for the 3D tensor
            // x and y dimensions cover the matrix, z dimension covers the batch
            dim3 gridDim(
                (hidden_dim + blockDim.x - 1) / blockDim.x,
                ((2 * shard_dim) + blockDim.y - 1) / blockDim.y,
                BATCH_MLP1
            );
            batch_transpose_kernel<<<gridDim, blockDim, 0, copy_streams[stream_id]>>>(
                d_tmp + stream_id * BATCH_MLP1 * tmp_elems, // source
                d_buf + d_offset, // destination
                BATCH_MLP1, (2 * shard_dim), hidden_dim
            );
        }
    }

    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipFree(d_tmp));
    for (int i = 0; i < BUFFER_MLP1; i++) {
        CHECK_HIP(hipStreamDestroy(copy_streams[i]));
    }
    free(copy_streams);
    
    printf("End alloc mlp1 new\n");
    fflush(stdout);
}

void alloc_w_mlp1_final_old(
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

    bf16** d_buf_array = new bf16*[TOTAL_GPUS_NEEDED];
    hipStream_t *copy_streams = new hipStream_t[TOTAL_GPUS_NEEDED * BUFFER_MLP1];
    hipEvent_t *copy_dones = new hipEvent_t[TOTAL_GPUS_NEEDED * BUFFER_MLP1];

    int end_layer = start_layer + OFFSET_LAYER;
    int end_moe = start_moe + OFFSET_MOE;

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
    size_t pp_offset = 1ll * n_layers * n_experts * hidden_dim * 2 * inter_dim;
    size_t le_slot_offset = 1ll * TOTAL_PIPELINES * BATCH_MLP1 * tmp_elems;

    for (size_t l = start_layer; l < end_layer; l++) {
        for (size_t e = start_moe; e < end_moe; e += BATCH_MLP1) {
            size_t base_out = 1ll * l * l_offset + 1ll * e * e_offset;
            int le_id = (l * n_experts + e) / (BUFFER_MLP1 * BATCH_MLP1);
            int le_slot_id = ((l * n_experts + e) / BATCH_MLP1) % BUFFER_MLP1;
            int le_slot = le_slot_id  * TOTAL_GPUS_NEEDED;
            size_t base_tmp_out = 1ll * le_slot_id * le_slot_offset;

            if (le_id) {
                CPUTimer sync("event sync");
                #pragma omp parallel for
                for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
                    CHECK_HIP(hipEventSynchronize(copy_dones[i + le_slot]));
                }
            }

            CPUTimer *transpose_timer = new CPUTimer("transpose");
            #pragma omp parallel for collapse(5)
            for (int tp_rank = 0; tp_rank < TP; tp_rank++) {
                for (int pp_rank = 0; pp_rank < PP; pp_rank++) {
                    for (int e_sm = 0; e_sm < BATCH_MLP1; e_sm++) {
                        for (size_t h = 0; h < hidden_dim; h++) {
                            for (size_t i = 0; i < 2 * shard_dim; i++) {
                                float value = w_mlp1_ptr[base_out + 1ll * pp_rank * pp_offset + 1ll * tp_rank * tp_offset + 1ll * e_sm * e_offset + i * hidden_dim + h];
                                tmp[base_tmp_out + 1ll * (pp_rank * TP + tp_rank) * BATCH_MLP1 * tmp_elems + 1ll * e_sm * tmp_elems + h * 2 * shard_dim + i] = bf16(value);
                            }
                        }
                    }
                }
            }
            delete transpose_timer;

            size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d;
            #pragma omp parallel for
            for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
                int tp_rank = i % TP;
                int pp_rank = (i / TP) % PP;
                int pipeline_stage = pp_rank * TP + tp_rank;
                CHECK_HIP(hipSetDevice(i));
                CHECK_HIP(hipMemcpyAsync(
                    d_buf_array[i] + d_offset,
                    tmp + base_tmp_out + 1ll * pipeline_stage * BATCH_MLP1 * tmp_elems, 
                    BATCH_MLP1 * tmp_elems * sizeof(bf16),
                    hipMemcpyHostToDevice, copy_streams[i + le_slot]
                ));
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

    free(tmp);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        for (int j = 0; j < BUFFER_MLP1; j++) {
            CHECK_HIP(hipStreamDestroy(copy_streams[i + j * TOTAL_GPUS_NEEDED]));
            CHECK_HIP(hipEventDestroy(copy_dones[i + j * TOTAL_GPUS_NEEDED]));
        }
    }
    
    delete[] d_buf_array;
    delete[] copy_streams;
    delete[] copy_dones;

    printf("End alloc mlp1 final\n");
    fflush(stdout);
}

void alloc_w_mlp1_final_old_v2(
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

    size_t tmp_elems = 2 * shard_dim;
    size_t size_tmp = 1ll * TOTAL_PIPELINES * BUFFER_MLP1 * BATCH_MLP1 * tmp_elems;
    bf16 *tmp = (bf16 *)malloc(size_tmp * sizeof(bf16));

    bf16** d_buf_array = new bf16*[TOTAL_GPUS_NEEDED];
    hipStream_t *copy_streams = new hipStream_t[TOTAL_GPUS_NEEDED * BUFFER_MLP1];
    hipEvent_t *copy_dones = new hipEvent_t[TOTAL_GPUS_NEEDED * BUFFER_MLP1];

    int end_layer = start_layer + OFFSET_LAYER;
    int end_moe = start_moe + OFFSET_MOE;

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
    size_t pp_offset = 1ll * n_layers * n_experts * hidden_dim * 2 * inter_dim;

    size_t le_slot_offset = 1ll * TOTAL_PIPELINES * BATCH_MLP1 * tmp_elems;

    for (size_t l = start_layer; l < end_layer; l++) {
        for (size_t e = start_moe; e < end_moe; e++) {
            for (size_t h = 0; h < hidden_dim; h += BATCH_MLP1) {
                size_t base_out = 1ll * l * l_offset + 1ll * e * e_offset + 1ll * h;
                int le_id = (1ll * (l * n_experts + e) * hidden_dim + h) / (BUFFER_MLP1 * BATCH_MLP1);
                int le_slot_id = ((1ll * (l * n_experts + e) * hidden_dim + h) / BATCH_MLP1) % BUFFER_MLP1;
                int le_slot = le_slot_id  * TOTAL_GPUS_NEEDED;
                size_t base_tmp_out = 1ll * le_slot_id * le_slot_offset;

                if (le_id) {
                    CPUTimer sync("event sync");
                    #pragma omp parallel for
                    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
                        CHECK_HIP(hipEventSynchronize(copy_dones[i + le_slot]));
                    }
                }
                
                // transpose
                CPUTimer *transpose_timer = new CPUTimer("transpose");
                #pragma omp parallel for collapse(4)
                for (int tp_rank = 0; tp_rank < TP; tp_rank++) {
                    for (int pp_rank = 0; pp_rank < PP; pp_rank++) {
                        for (int h_sm = 0; h_sm < BATCH_MLP1; h_sm++) {
                            for (size_t i = 0; i < 2 * shard_dim; i++) {
                                float value = w_mlp1_ptr[base_out + 1ll * pp_rank * pp_offset + 1ll * tp_rank * tp_offset + 1ll * h_sm + i * hidden_dim];
                                tmp[base_tmp_out + 1ll * (pp_rank * TP + tp_rank) * BATCH_MLP1 * tmp_elems + 1ll * h_sm * tmp_elems + i] = bf16(value);
                            }
                        }
                    }
                }
                delete transpose_timer;
                
                size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d + 1ll * h * tmp_elems;
                #pragma omp parallel for
                for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
                    int tp_rank = i % TP;
                    int pp_rank = (i / TP) % PP;
                    int pipeline_stage = pp_rank * TP + tp_rank;
                    CHECK_HIP(hipSetDevice(i));
                    CHECK_HIP(hipMemcpyAsync(
                        d_buf_array[i] + d_offset,
                        tmp + base_tmp_out + 1ll * pipeline_stage * BATCH_MLP1 * tmp_elems, 
                        BATCH_MLP1 * tmp_elems * sizeof(bf16),
                        hipMemcpyHostToDevice, copy_streams[i + le_slot]
                    ));
                    CHECK_HIP(hipEventRecord(copy_dones[i + le_slot], copy_streams[i + le_slot]));
                }
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

    free(tmp);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
        for (int j = 0; j < BUFFER_MLP1; j++) {
            CHECK_HIP(hipStreamDestroy(copy_streams[i + j * TOTAL_GPUS_NEEDED]));
            CHECK_HIP(hipEventDestroy(copy_dones[i + j * TOTAL_GPUS_NEEDED]));
        }
    }
    
    delete[] d_buf_array;
    delete[] copy_streams;
    delete[] copy_dones;

    printf("End alloc mlp1 final\n");
    fflush(stdout);
}

void alloc_w_mlp1_v2(
    Tensor* &w_mlp1, float *w_mlp1_ptr, Config *p, int device_id,
    int start_layer, int start_moe
) {
    CPUTimer timer("alloc_w_mlp1_v2");
    printf("Starting alloc mlp1 v2\n");
    fflush(stdout);

    int tp_rank = device_id % TP;
    size_t n_layers = p->n_layers / PP;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;
    size_t shard_dim = p->intermediate_dim / TP;

    int end_layer = start_layer + OFFSET_LAYER;
    int end_moe = start_moe + OFFSET_MOE;

    w_mlp1_ptr += tp_rank * (2 * shard_dim) * hidden_dim; // offset shard_dim;

    if (start_layer == 0 and start_moe == 0) {
        w_mlp1 = new Tensor({n_layers, n_experts, hidden_dim, 2 * shard_dim}, w_mlp1_ptr, device_id, false, DType::BF16);
    }

    size_t tmp_elems = hidden_dim * 2 * shard_dim;
    bf16 *tmp = (bf16 *)malloc(BATCH_MLP1 * tmp_elems * sizeof(bf16));

    bf16 *d_buf = (bf16 *)(w_mlp1->d_buf);

    size_t l_offset = 1ll * n_experts * (2 * inter_dim) * hidden_dim;
    size_t e_offset = 1ll * (2 * inter_dim) * hidden_dim;

    size_t l_offset_d = 1ll * n_experts * hidden_dim * (2 * shard_dim);
    size_t e_offset_d = 1ll * hidden_dim * (2 * shard_dim);

    for (size_t l = start_layer; l < end_layer; l++) {
        for (size_t e = start_moe; e < end_moe; e += BATCH_MLP1) {
            size_t base_out = 1ll * l * l_offset + 1ll * e * e_offset;

            #pragma omp parallel for collapse(3)
            for (size_t e_sm = 0; e_sm < BATCH_MLP1; e_sm++) {
                size_t base = base_out + 1ll * e_sm * e_offset;
                size_t base_tmp = 1ll * e_sm * tmp_elems;
                for (size_t h = 0; h < hidden_dim; h++) {
                    for (size_t i = 0; i < 2 * shard_dim; i++) {
                        float value = w_mlp1_ptr[base + i * hidden_dim + h];
                        tmp[base_tmp + h * 2 * shard_dim + i] = bf16(value);
                    }
                }
            }

            size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d;
            CHECK_HIP(hipMemcpyAsync(d_buf + d_offset, tmp, BATCH_MLP1 * tmp_elems * sizeof(bf16),
                                    hipMemcpyHostToDevice, 0));
        }
    }

    CHECK_HIP(hipStreamSynchronize(0));
    free(tmp);

    printf("End alloc mlp1 v2\n");
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

void alloc_w_mlp2_transpose(
    Tensor* &w_mlp2, float *w_mlp2_ptr, Config *p, int device_id
) {
    CPUTimer timer("alloc_w_mlp2_transpose");
    printf("Starting alloc mlp2 new\n");
    fflush(stdout);

    int tp_rank = device_id % TP;
    size_t n_layers = p->n_layers / PP;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;
    size_t shard_dim = p->intermediate_dim / TP;

    w_mlp2_ptr += 1ll * tp_rank * shard_dim; // offset shard_dim;

    w_mlp2 = new Tensor({n_layers, n_experts, shard_dim, hidden_dim}, w_mlp2_ptr, device_id, false, DType::BF16);

    hipStream_t *copy_streams = new hipStream_t[BUFFER_MLP2];
    for (int i = 0; i < BUFFER_MLP2; i++) {   
        CHECK_HIP(hipStreamCreate(&copy_streams[i]));
    }

    float *d_tmp = nullptr;
    size_t tmp_elems = hidden_dim * shard_dim;
    CHECK_HIP(hipMalloc(&d_tmp, BUFFER_MLP2 * tmp_elems * sizeof(float)));

    bf16 *d_buf = (bf16 *)(w_mlp2->d_buf);

    size_t l_offset = 1ll * n_experts * hidden_dim * inter_dim;
    size_t e_offset = 1ll * hidden_dim * inter_dim;

    size_t l_offset_d = 1ll * n_experts * shard_dim * hidden_dim;
    size_t e_offset_d = 1ll * shard_dim * hidden_dim;
    
    for (size_t l = 0; l < n_layers; l++) {
        for (size_t e = 0; e < n_experts; e++) {
            size_t base = 1ll * l * l_offset + 1ll * e * e_offset;
            size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d;
            int stream_id = (l * n_experts + e) % BUFFER_MLP2;
            float *d_tmp_cur = (float *)d_tmp + stream_id * tmp_elems;
            float *w_mlp2_cur = (float *)w_mlp2_ptr + base;

            // copy l, e shard to gpu using stream id, offset l_offset and e_offset
            for (int h = 0; h < hidden_dim; h++) {
                // int stream_id = (h / BATCH_MLP2) % BUFFER_MLP2;
                CHECK_HIP(hipMemcpyAsync(
                    d_tmp_cur + h * shard_dim, // destination on device
                    w_mlp2_cur + h * inter_dim, // source on host
                    shard_dim * sizeof(float), // size of data to copy
                    hipMemcpyHostToDevice, // direction
                    copy_streams[stream_id] // stream
                ));

                if ((h+1) % BATCH_MLP2 == 0 && false) {
                    dim3 blockDim(16, 16);
                    dim3 gridDim(
                        (shard_dim + blockDim.x - 1) / blockDim.x,
                        (BATCH_MLP2 + blockDim.y - 1) / blockDim.y
                    );
                    partial_transpose_kernel<<<gridDim, blockDim, 0, copy_streams[stream_id]>>>(
                        d_tmp_cur, d_buf + d_offset,
                        hidden_dim, shard_dim, h+1-BATCH_MLP2, BATCH_MLP2
                    );
                }
            }

            // --- Kernel launch configuration ---
            const int BLOCK_SIZE_X = 32;
            const int BLOCK_SIZE_Y = 32;
            dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 grid_dim(
                (shard_dim + block_dim.x - 1) / block_dim.x,
                (hidden_dim + block_dim.y - 1) / block_dim.y
            );

            // Launch the kernel
            transpose_kernel<<<grid_dim, block_dim, 0, copy_streams[stream_id]>>>(
                d_tmp_cur, d_buf + d_offset, hidden_dim, shard_dim
            );
        }
    }

    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipFree(d_tmp));
    for (int i = 0; i < BUFFER_MLP2; i++) {
        CHECK_HIP(hipStreamDestroy(copy_streams[i]));
    }
    free(copy_streams);

    // w_mlp2->to_device(0);
    printf("End alloc mlp2 new\n");
    fflush(stdout);
}

void alloc_w_mlp2_new(
    Tensor* &w_mlp2, float *w_mlp2_ptr, Config *p, int device_id
) {
    CPUTimer timer("alloc_w_mlp2_new");
    printf("Starting alloc mlp2 new\n");
    fflush(stdout);

    int tp_rank = device_id % TP;
    size_t n_layers = p->n_layers / PP;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;
    size_t shard_dim = p->intermediate_dim / TP;

    w_mlp2_ptr += 1ll * tp_rank * shard_dim; // offset shard_dim;

    w_mlp2 = new Tensor({n_layers, n_experts, shard_dim, hidden_dim}, w_mlp2_ptr, device_id, false, DType::BF16);

    size_t buffer_streams = (1ll * BUFFER_MLP2 * hidden_dim) / BATCH_MLP2;
    hipStream_t *copy_streams = new hipStream_t[buffer_streams];
    for (int i = 0; i < buffer_streams; i++) {   
        CHECK_HIP(hipStreamCreate(&copy_streams[i]));
    }

    float *d_tmp = nullptr;
    size_t tmp_elems = hidden_dim * shard_dim;
    CHECK_HIP(hipMalloc(&d_tmp, BUFFER_MLP2 * tmp_elems * sizeof(float)));

    bf16 *d_buf = (bf16 *)(w_mlp2->d_buf);

    size_t l_offset = 1ll * n_experts * hidden_dim * inter_dim;
    size_t e_offset = 1ll * hidden_dim * inter_dim;

    size_t l_offset_d = 1ll * n_experts * shard_dim * hidden_dim;
    size_t e_offset_d = 1ll * shard_dim * hidden_dim;
    
    for (size_t l = 0; l < n_layers; l++) {
        for (size_t e = 0; e < n_experts; e++) {
            size_t base = 1ll * l * l_offset + 1ll * e * e_offset;
            size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d;
            float *w_mlp2_cur = (float *)w_mlp2_ptr + base;
            int buf_offset = (e % BUFFER_MLP2);
            int stream_offset = buf_offset * (buffer_streams / BUFFER_MLP2);

            // copy l, e shard to gpu using stream id, offset l_offset and e_offset
            for (int h = 0; h < hidden_dim; h++) {
                int stream_id = (stream_offset + h / BATCH_MLP2) % buffer_streams;
                int slot_id = stream_id * BATCH_MLP2 + (h % BATCH_MLP2);

                /*
                    0 1 2 3
                    4 5 6 7
                */
                
                // int stream_id = (h / BATCH_MLP2) % BUFFER_MLP2;
                CHECK_HIP(hipMemcpyAsync(
                    d_tmp + slot_id * shard_dim, // destination on device
                    w_mlp2_cur + h * inter_dim, // source on host
                    shard_dim * sizeof(float), // size of data to copy
                    hipMemcpyHostToDevice, // direction
                    copy_streams[stream_id] // stream
                ));

                if ((h+1) % BATCH_MLP2 == 0) {
                    dim3 blockDim(16, 16);
                    dim3 gridDim(
                        (shard_dim + blockDim.x - 1) / blockDim.x,
                        (BATCH_MLP2 + blockDim.y - 1) / blockDim.y
                    );
                    partial_transpose_kernel<<<gridDim, blockDim, 0, copy_streams[stream_id]>>>(
                        d_tmp + buf_offset * tmp_elems,
                        d_buf + d_offset,
                        hidden_dim, shard_dim, h+1-BATCH_MLP2, BATCH_MLP2
                    );
                }
            }
        }
    }

    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipFree(d_tmp));
    for (int i = 0; i < buffer_streams; i++) {
        CHECK_HIP(hipStreamDestroy(copy_streams[i]));
    }
    free(copy_streams);

    // w_mlp2->to_device(0);
    printf("End alloc mlp2 new\n");
    fflush(stdout);
}

void alloc_w_mlp2_v2(
    Tensor* &w_mlp2, float *w_mlp2_ptr, Config *p, int device_id
) {
    CPUTimer timer("alloc_w_mlp2_v2");
    printf("Starting alloc mlp2 v2\n");
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
    bf16 *tmp = (bf16 *)malloc(BATCH_MLP2 * tmp_elems * sizeof(bf16));

    bf16 *d_buf = (bf16 *)(w_mlp2->d_buf);

    size_t l_offset = 1ll * n_experts * hidden_dim * inter_dim;
    size_t e_offset = 1ll * hidden_dim * inter_dim;

    size_t l_offset_d = 1ll * n_experts * shard_dim * hidden_dim;
    size_t e_offset_d = 1ll * shard_dim * hidden_dim;
    
    for (size_t l = 0; l < n_layers; l++) {
        for (size_t e = 0; e < n_experts; e += BATCH_MLP2) {
            size_t base_out = 1ll * l * l_offset + 1ll * e * e_offset;

            #pragma omp parallel for collapse(3)
            for (size_t e_sm = 0; e_sm < BATCH_MLP2; e_sm++) {
                size_t base = base_out + 1ll * e_sm * e_offset;
                size_t base_tmp = 1ll * e_sm * tmp_elems;
                for (size_t i = 0; i < shard_dim; i++) {
                    for (size_t h = 0; h < hidden_dim; h++) {
                        float value = w_mlp2_ptr[base + h * inter_dim + i];
                        tmp[base_tmp + i * hidden_dim + h] = bf16(value);
                    }
                }
            }

            size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d;
            CHECK_HIP(hipMemcpyAsync(d_buf + d_offset, tmp, BATCH_MLP2 * tmp_elems * sizeof(bf16),
                                    hipMemcpyHostToDevice, 0));
        }
    }

    CHECK_HIP(hipStreamSynchronize(0));
    free(tmp);

    // w_mlp2->to_device(0);
    printf("End alloc mlp2 v2\n");
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
