__device__ __forceinline__ uint32_t ld_u32(const uint32_t *p) {
    return *p;
}
__device__ __forceinline__ double block_reduce_max(double v, double *s_red) {
    const int tid = threadIdx.x;
    s_red[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) 
            s_red[tid] = fmax(s_red[tid], s_red[tid + s]);
        __syncthreads();
    }
    return s_red[0];
}

__device__ __forceinline__ double block_reduce_sum(double v, double *s_red) {
    const int tid = threadIdx.x;
    s_red[tid] = v;
    __syncthreads();
    for (int s = blockDim >> 1; s > 0; s >>= 1) {
        if (tid < s) 
            s_red[tid] += s_red[tid + s];
        __syncthreads();
    }
    return s_red[0];
}

template <int TILE_TOKENS, int THREADS, int D> 
__global__ void fd_partial_kernel_bf16(
    const float *__restrict__ q, 
    const bf16 *__restrict__ K_cache, 
    const bf16 *__restrict__ V_cache, 
    float *__restrict__ partial_max, 
    float *__restrict__ partial_sum, 
    float *__restrict__ partial_num, 
    int n_q, int kv_mul, int kv_dim, int n_layers, int seq_len, int t_begin, int t_end, 
    int batch_size, int chunk) 
{
    const int bx = blockIdx.x;
    const int b = blockIdx.y;
    if (b >= batch_size) return;

    const int cidx = bx / n_q;
    const int h = bx % n_q;
    if (cidx >= chunk || h >= n_q) return;
    
    const int kv_h = h / kv_mul;
    const int start = t_begin + cidx * TILE_TOKENS;
    const int stop = min(t_end, start + TILE_TOKENS);
    const int T = max(0, stop - start);
    
    const size_t qtb_stride = (size_t)n_q * D;
    const size_t kv_batch_stride = (size_t)n_layers * seq_len * kv_dim;
    const size_t kv_head_off = (size_t)kv_h * D;

    const float *__restrict__ q_head = q + (size_t)b * qtb_stride + (size_t)h * D;
    const bf16 *__restrict__ Kb = K_cache + (size_t)b * kv_batch_stride;
    const bf16 *__restrict__ Vb = V_cache + (size_t)b * kv_batch_stride;

    const size_t bhc = ((size_t)b * n_q + h) * (size_t)chunk;
    float *__restrict__ pmax_bhc = partial_max + bhc;
    float *__restrict__ psum_bhc = partial_sum + bhc;
    float *__restrict__ pnum_bhc = partial_num + bhc * (size_t)D;

    const float scale = rsqrtf((float)D);
    
    extern __shared__ int __fd_smem_i32[];
    unsigned char *sbase = (unsigned char *)__fd_smem_i32;  
    float *s_q = (float *)sbase;
    float *s_sc = (float *)(s_q + D);
    double *s_red = (double *)(s_sc + TILE_TOKENS);

    // stage Q
    for (int i = threadIdx.x; i < D; i += blockDim.x) 
        s_q[i] = q_head[i];
    __syncthreads();

    // Pass 1: QK 
    double local_max = -DBL_MAX;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        const size_t base = (size_t)(start + t) * (size_t)kv_dim + kv_head_off;
        float sf = 0.f;
        #pragma unroll 
        for (int d = 0; d < D; d += 2) {
            const uint32_t p = ld_u32(
                reinterpret_cast<const uint32_t *>(reinterpret_cast<const uint16_t *>(Kb) + base + d));
            const uint16_t u0 = (uint16_t)(p & 0xFFFF);
            const uint16_t u1 = (uint16_t)(p >> 16);
            sf = fmaf(s_q[d + 0], bf16_to_f32_bits(u0), sf);
            sf = fmaf(s_q[d + 1], bf16_to_f32_bits(u1), sf);
        }
        sf *= scale;
        s_sc[t] = sf;
        if ((double)sf > local_max) 
            local_max = (double)sf;
    }
    const double m_c = (T > 0) ? block_reduce_max(local_max, s_red) : -DBL_MAX;
    
    // Pass 2: exp wrt m_c + denom (double) 
    double local_sum = 0.0;
    for (int t= threadIdx.x; t < T; t += blockDim.x) {
        const double w = exp((double)s_sc[t] - m_c);
        s_sc[t] = (float)w;
        local_sum += w;
    } 
    const double l_c = (T > 0) ? block_reduce_sum(local_sum, s_red) : 0.0;

    // Pass 3: numerator with 2 outputs/thread
    for (int i_out = (threadIdx.x << 1); i_out < D; i_out += (blockDim.x << 1)) {
        double acc = 0.0, acc1 = 0.0;
        #pragma unroll 8 
        for (int t = 0; t < T; ++t) {
            const double w = (double)s_sc[t];
            const size_t base = (size_t)(start + t) * (size_t)kv_dim + kv_head_off + (size_t)i_out;
            const uint32_t vp = 
                ld_u32(reinterpret_cast<const uint32_t *>(reinterpret_cast<const uint16_t *>(Vb) + base));
            
        }
    }
}

void flash_attention(
    Tensor *q, 
    Tensor *K_cache, 
    Tensor *V_cache, 
    Tensor *mask, 
    Tensor *attn_sinks, 
    Tensor *tb, 
    Tensor *g_fa_pmax, 
    Tensor *g_fa_psum, 
    Tensor *g_fa_pnum, 
    int cur_batch_size, 
    int head_dim,
    int n_q, 
    int kv_mul,
    int kv_dim, 
    int seq_len, 
    int sliding_window, 
    int pos, 
    long long layer_offset, 
    bool q_to_device, bool k_cache_to_device, 
    bool v_cache_to..
) {
    const size_t n_layers = (size_t)K_cache->shape[1];
    const size_t layer_span = (size_t)seq_len * kv_dim;
    const bf16 *K_ptr = ((const bf16 *)K_cache->d_buf) + layer_offset * layer_span;
    const bf16 *V_ptr = ((const bf16 *)V_cache->d_buf) + layer_offset * layer_span;
    const float *S_ptr = (const float *)attn_sinks->d_buf + layer_offset * n_q;
    
    const int attn_len = pos + 1;
    const bool use_window = (sliding_window > 0) && ((layer_offset & 1ll) == 0);
    const int L_eff = use_window ? min(attn_len, sliding_window) : attn_len;
    const int t_end = attn_len;
    const int t_begin = t_end - L_eff;

    const int chunk = (L_eff + FA_TILE - 1) / FA_TILE;
    const int g_fa_C_max = (seq_len + FLASH_ATTN_TILE - 1) / FLASH_ATTN_TILE;
    assert(g_fa_C_max >= chunk);
    float *d_pmax = (float *)g_fa_pmax->d_buf;
    float *d_psum = (float *)g_fa_psum->d_buf;
    float *d_pnum = (float *)g_fa_pnum->d_buf;
    dim3 grid_p(n_q * chunk, cur_batch_size), block_p(128);

    fd_partial_kernel_bf16<FA_TILE, 128, 64><<<grid_p, block_p, shmem_p, stream>>>(
        (const float*)q->d_buf, K_ptr, V_ptr, d_pmax, d_psum, d_pnum, n_q, kv_mul, /
        kv_dim, n_layers, /
        seq_len, t_begin, t_end, cur_batch_size, chunk);
    dim3 grid_r(n_q, cur_batch_size), block_r(head_dim);
    fd_reduce_kernel_stable_v2<64><<<grid_r, block_r, 0, stream>>>(
        d_pmax, d_psum, d_pnum, S_ptr, (float *)tb->d_buf, n_q, chunk, cur_batch_size);
    

}

template <int D> 
__global__ void fd_reduce_kernel_stable_v2(
    const float *__restrict__ partial_max,
    const float *__restrict__ partial_sum, 
    const float *__restrict__ partial_num, 
    const float *__restrict__ attn_sinks, 
    float *__restrict__ tb, 
    int n_q, int chunk, int batch_size) 
{
    const int h = blockIdx.x, b = blockIdx.y;
    if (b >= batch_size || h >= n_q) return;

    const size_t qtb_stride = (size_t)n_q * (size_t)D;
    float *__restrict__ out = tb + (size_t)b * qtb_stride + (size_t)h * D;

    const size_t baseBH = ((size_t)b * nq + h);
    const float *mBH = partial_max + baseBH * (size_t)chunk;
    const float *sBH = partial_sum + baseBH * (size_t)chunk;
    const float *nBH = partial_num + baseBH * (size_t)chunk * (size_t)D;

    double m_star = -DBL_MAX;
    for (int c = 0; c < chunk; ++c) 
        m_star = fmax(m_star, (double)mBH[c]);
    m_star = fmax(m_star, (double)attn_sinks[h]);
    const int i = threadIdx.x;
    double denom_chunks = 0.0, num_i = 0.0;
    for (int c = 0; c < chunk; ++c) {
        const double mc = (double)mBH[c];
        const double sc = isfinite(mc) ? exp(mc - m_star) : 0.0;
        denom_chunks += (double)sBH[c] * sc;
        if (i < D) 
            num_i += (double)nBH[(size_t)c * D + i] * sc;
    }
    const double denom = denom_chunks + exp((double)attn_sinks[h] - m_star);
    
}