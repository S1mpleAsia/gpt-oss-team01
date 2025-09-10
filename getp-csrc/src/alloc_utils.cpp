#include "../include/alloc_utils.hpp"

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
        hip_bfloat16 bf16_val = bf16(val);

        // Store the bfloat16 value in the transposed matrix
        d_transpose[transpose_idx] = bf16_val;
    }
}

// HIP kernel to transpose a specific shard of a 3D tensor
// and convert from float to bfloat16.
__global__ void hipTransposeShardHeight(
    const float *__restrict__ d_orig,
    bf16 *__restrict__ d_transpose, int batch, int w,
    int shard_dim, int h_total, int shard_id
) {
    // Calculate global thread indices for the output tensor (d_transpose)
    int out_h_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_w_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_batch_idx = blockIdx.z * blockDim.z + threadIdx.z;

    // Perform bounds checking to ensure thread is within the output tensor's dimensions
    if (out_batch_idx < batch && out_w_idx < w && out_h_idx < shard_dim) {

        // Calculate the corresponding row index in the original tensor (d_orig).
        // This is based on the shard_id and the current thread's row index within the shard.
        long long orig_h_idx = 1ll * shard_id * shard_dim + out_h_idx;

        // Calculate linear indices for both tensors.
        // d_orig is (batch, h_total, w)
        long long orig_idx = 1ll * out_batch_idx * (h_total * w) + 1ll * orig_h_idx * w + 1ll * out_w_idx;

        // d_transpose is (batch, w, shard_dim)
        long long transpose_idx = 1ll * out_batch_idx * (w * shard_dim) + 1ll * out_w_idx * shard_dim + 1ll * out_h_idx;

        // Load the float value from the original tensor
        float value = d_orig[orig_idx];

        // Convert the float value to bfloat16 and store it in the transposed tensor
        d_transpose[transpose_idx] = bf16(value);
    }
}

__global__ void hipTransposeShardWidth(
    const float *__restrict__ d_orig,
    bf16 *__restrict__ d_transpose, int batch, int h,
    int w_shard, int w_total, int shard_id
) {

    // Calculate global thread indices for the output tensor (d_transpose)
    // The dimensions of the output are (batch, w_shard, h)
    int out_h_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_w_shard_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_batch_idx = blockIdx.z * blockDim.z + threadIdx.z;

    // Perform bounds checking to ensure thread is within the output tensor's dimensions
    if (out_batch_idx < batch && out_w_shard_idx < w_shard && out_h_idx < h) {

        // Calculate the corresponding column index in the original tensor (d_orig).
        // This is based on the shard_id and the current thread's column index within the shard.
        long long orig_w_idx = 1ll * shard_id * w_shard + out_w_shard_idx;

        // Calculate linear indices for both tensors.
        // d_orig is (batch, h, w_total)
        long long orig_idx = 1ll * out_batch_idx * (h * w_total) + 1ll * out_h_idx * w_total + 1ll * orig_w_idx;

        // d_transpose is (batch, w_shard, h)
        long long transpose_idx = 1ll * out_batch_idx * (w_shard * h) + 1ll * out_w_shard_idx * h + 1ll * out_h_idx;

        // Load the float value from the original tensor
        float value = d_orig[orig_idx];

        // Convert the float value to bfloat16 and store it in the transposed tensor
        d_transpose[transpose_idx] = bf16(value);
    }
}

void alloc_w_mlp1_new(
    Tensor* &w_mlp1, float *w_mlp1_ptr, Config *p, int device_id
) {
    printf("Starting alloc mlp1 new\n");
    fflush(stdout);

    int tp_rank = device_id % TP;
    size_t n_layers = p->n_layers / PP;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;
    size_t shard_dim = p->intermediate_dim / TP;

    // w_mlp1_ptr += tp_rank * (2 * shard_dim) * hidden_dim; // offset shard_dim;

    w_mlp1 = new Tensor({n_layers, n_experts, hidden_dim, 2 * shard_dim}, w_mlp1_ptr, device_id, false, DType::BF16);

    hipStream_t *copy_streams = new hipStream_t[BUFFER_MLP1];
    for (int i = 0; i < BUFFER_MLP1; i++) {   
        CHECK_HIP(hipStreamCreate(&copy_streams[i]));
    }

    float *d_tmp = nullptr;
    size_t tmp_elems = BATCH_MLP1 * (2 * inter_dim) * hidden_dim;
    CHECK_HIP(hipMalloc(&d_tmp, BUFFER_MLP1 * tmp_elems * sizeof(float)));

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

            // copy l, e shard to gpu using stream id, offset l_offset and e_offset
            CHECK_HIP(hipMemcpyAsync(
                d_tmp + stream_id * tmp_elems, // destination on device
                w_mlp1_ptr + base, // source on host
                tmp_elems * sizeof(float), // size of data to copy
                hipMemcpyHostToDevice, // direction
                copy_streams[stream_id] // stream
            ));

            // --- Kernel launch configuration ---
            dim3 block_dim(16, 16, 4); // Threads per block
            dim3 grid_dim(
                (2 * shard_dim + block_dim.x - 1) / block_dim.x,
                (hidden_dim + block_dim.y - 1) / block_dim.y,
                (BATCH_MLP1 + block_dim.z - 1) / block_dim.z
            ); // Blocks in grid

            // Launch the kernel
            hipTransposeShardHeight<<<grid_dim, block_dim, 0, copy_streams[stream_id]>>>(
                d_tmp + stream_id * tmp_elems,
                d_buf + d_offset,
                BATCH_MLP1, hidden_dim, 2 * shard_dim, 2 * inter_dim, tp_rank
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

void alloc_w_mlp1(
    Tensor* &w_mlp1, float *w_mlp1_ptr, Config *p, int device_id
) {
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

void alloc_w_mlp2_new(
    Tensor* &w_mlp2, float *w_mlp2_ptr, Config *p, int device_id
) {
    printf("Starting alloc mlp2 new\n");
    fflush(stdout);

    int tp_rank = device_id % TP;
    size_t n_layers = p->n_layers / PP;
    size_t n_experts = p->n_experts;
    size_t hidden_dim = p->hidden_dim;
    size_t inter_dim = p->intermediate_dim;
    size_t shard_dim = p->intermediate_dim / TP;

    // w_mlp2_ptr += 1ll * tp_rank * shard_dim; // offset shard_dim;

    w_mlp2 = new Tensor({n_layers, n_experts, shard_dim, hidden_dim}, w_mlp2_ptr, device_id, false, DType::BF16);

    hipStream_t *copy_streams = new hipStream_t[BUFFER_MLP2];
    for (int i = 0; i < BUFFER_MLP2; i++) {   
        CHECK_HIP(hipStreamCreate(&copy_streams[i]));
    }

    float *d_tmp = nullptr;
    size_t tmp_elems = BATCH_MLP2 * hidden_dim * inter_dim;
    CHECK_HIP(hipMalloc(&d_tmp, BUFFER_MLP2 * tmp_elems * sizeof(float)));

    bf16 *d_buf = (bf16 *)(w_mlp2->d_buf);

    size_t l_offset = 1ll * n_experts * hidden_dim * inter_dim;
    size_t e_offset = 1ll * hidden_dim * inter_dim;

    size_t l_offset_d = 1ll * n_experts * shard_dim * hidden_dim;
    size_t e_offset_d = 1ll * shard_dim * hidden_dim;
    
    for (size_t l = 0; l < n_layers; l++) {
        for (size_t e = 0; e < n_experts; e += BATCH_MLP2) {
            size_t base = 1ll * l * l_offset + 1ll * e * e_offset;
            size_t d_offset = 1ll * l * l_offset_d + 1ll * e * e_offset_d;
            int stream_id = ((l * n_experts + e) / BATCH_MLP2) % BUFFER_MLP2;

            // copy l, e shard to gpu using stream id, offset l_offset and e_offset
            CHECK_HIP(hipMemcpyAsync(
                d_tmp + stream_id * tmp_elems, // destination on device
                w_mlp2_ptr + base, // source on host
                tmp_elems * sizeof(float), // size of data to copy
                hipMemcpyHostToDevice, // direction
                copy_streams[stream_id] // stream
            ));

            // --- Kernel launch configuration ---
            dim3 block_dim(16, 16, 4); // Threads per block
            dim3 grid_dim(
                (hidden_dim + block_dim.x - 1) / block_dim.x,
                (shard_dim + block_dim.y - 1) / block_dim.y,
                (BATCH_MLP2 + block_dim.z - 1) / block_dim.z
            ); // Blocks in grid

            // Launch the kernel
            hipTransposeShardWidth<<<grid_dim, block_dim, 0, copy_streams[stream_id]>>>(
                d_tmp + stream_id * tmp_elems,
                d_buf + d_offset,
                BATCH_MLP2, hidden_dim, shard_dim, inter_dim, tp_rank
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

void alloc_w_mlp2(
    Tensor* &w_mlp2, float *w_mlp2_ptr, Config *p, int device_id
) {
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
