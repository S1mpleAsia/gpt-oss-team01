#include "layer.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>

void EmbeddingLookup(const Tensor *embedding /*[vocab, hidden]*/, int token_id,
                     Tensor *out /*[hidden]*/) {
    size_t hidden_dim = embedding->shape[1];
    memcpy(out->buf, embedding->buf + token_id * hidden_dim, hidden_dim * sizeof(float));
}

// RMSNorm: out = scale * x / rms(x)
void RMSNorm(const Tensor *x /*[hidden]*/, const Tensor *scale /*[hidden]*/,
             Tensor *out /*[hidden]*/, long long layer_offset, float eps) {
    size_t size = x->num_elem();
    float *scale_buf = scale->buf + 1ll * layer_offset * size;
    // calculate sum of squares
    double ss = 0.0;
    for (size_t j = 0; j < size; j++) {
        ss += x->buf[j] * x->buf[j];
    }
    ss /= size;
    ss += eps;
    ss = 1.0 / sqrt(ss);
    // normalize and scale
    for (size_t j = 0; j < size; j++) {
        out->buf[j] = scale_buf[j] * (ss * x->buf[j]);
    }
}

// y = Wx + b -- W: [out, in], x: [in], y: [out]
void Linear(const float *x, size_t in_dim, const float *W, const float *b, 
            size_t out_dim, float *y) {
    for (size_t i = 0; i < out_dim; i++) {
        double val = 0.0;
        for (size_t j = 0; j < in_dim; j++) {
            val += W[i * in_dim + j] * x[j];
        }
        if (b) val += b[i];
        y[i] = val;
    }
}

void Softmax(float *x, size_t size) {
    // find max value (for numerical stability)
    double max_val = x[0];
    for (size_t i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    double sum = 0.0;
    for (size_t i = 0; i < size; i++) {
        // printf("x->buf[%d] before: %.6f\n", i, x->buf[i]);
        x[i] = expf(x[i] - max_val);
        // printf("x->buf[%d] after: %.6f\n", i, x->buf[i]);
        sum += x[i];
    }
    // printf("sum: %.6f\n", sum);
    // printf("max_val: %.6f\n", max_val);
    // normalize
    for (size_t i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// QKV projection: qkv = W_qkv * x + b_qkv
// qkv.shape = [(n_q + 2*n_kv)*head_dim]
void QKVProject(const Tensor *x /*[hidden]*/,
                const Tensor *W_qkv /*[(n_q+2*n_kv)*hd, hidden]*/,
                const Tensor *b_qkv, Tensor *qkv /*[(n_q+2*n_kv)*hd]*/,
                long long layer_offset) {
    const long long hidden = x->num_elem();
    const long long nq_kv_hd = qkv->num_elem();
    
    const float *w_qkv_ptr = W_qkv->buf + 1ll * layer_offset * nq_kv_hd * hidden;
    const float *b_qkv_ptr = b_qkv->buf + 1ll * layer_offset * nq_kv_hd;

    Linear(x->buf, hidden, w_qkv_ptr, b_qkv_ptr, nq_kv_hd, qkv->buf);
}

void SplitQKV(const Tensor *qkv, int head_dim, int n_q, int n_kv,
              Tensor *q /*[n_q*hd]*/, Tensor *k /*[n_kv*hd]*/,
              Tensor *v /*[n_kv*hd]*/) {
    size_t offset = 0;
    memcpy(q->buf, qkv->buf + offset, n_q * head_dim * sizeof(float));
    offset += n_q * head_dim;
    memcpy(k->buf, qkv->buf + offset, n_kv * head_dim * sizeof(float));
    offset += n_kv * head_dim;
    memcpy(v->buf, qkv->buf + offset, n_kv * head_dim * sizeof(float));
}

void our_compute_concentration_and_inv_freq(float base, int head_dim,
                                        float scaling_factor,
                                        float initial_context_length,
                                        float ntk_beta, float ntk_alpha,
                                        float *concentration_out,
                                        float *inv_freq_out // length head_dim/2
) {
    int d_half = head_dim / 2;

    // freq[i] = base ** (i / head_dim)
    float *freq = (float *)malloc(d_half * sizeof(float));
    for (int i = 0; i < d_half; i++) {
        freq[i] = powf(base, ((float)(2 * i)) / (float)head_dim);
    }

    float concentration;
    if (scaling_factor > 1.0f) {
        // YaRN concentration
        concentration = 0.1f * logf(scaling_factor) + 1.0f;

        // NTK by parts
        float low = d_half *
                    logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) /
                    logf(base);
        float high = d_half *
                    logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) /
                    logf(base);

        assert(0 < low && low < high && high < d_half - 1);

        // interpolation = 1 / (scaling_factor * freq)
        // extrapolation = 1 / freq
        for (int i = 0; i < d_half; i++) {
            float interpolation = 1.0f / (scaling_factor * freq[i]);
            float extrapolation = 1.0f / freq[i];

            float ramp = ((float)i - low) / (high - low);
            if (ramp < 0) ramp = 0;
            if (ramp > 1) ramp = 1;

            float mask = 1.0f - ramp;
            inv_freq_out[i] = interpolation * (1.0f - mask) + extrapolation * mask;
        }
    } else {
        concentration = 1.0f;
        for (int i = 0; i < d_half; i++) {
            inv_freq_out[i] = 1.0f / freq[i];
        }
    }

    *concentration_out = concentration;
    free(freq);
}

// cos/sin computation for RoPE at pos
void RopeComputeCS(int pos, const Config &p, Tensor *cos_out /*[head_dim/2]*/,
                   Tensor *sin_out /*[head_dim/2]*/) {
    int d_half = cos_out->num_elem();
    float base = p.rope_theta;
    int head_dim = d_half * 2;
    float scaling_factor = p.rope_scaling_factor;
    float initial_context_length = p.initial_context_length;
    float ntk_beta = 32.0f;
    float ntk_alpha = 1.0f;

    // Get concentration + inv_freq
    float concentration;
    float *inv_freq = (float *)malloc(d_half * sizeof(float));

    our_compute_concentration_and_inv_freq(base, head_dim, scaling_factor,
                                        initial_context_length, ntk_beta,
                                        ntk_alpha, &concentration, inv_freq);

    // Compute cos and sin for this position
    for (int j = 0; j < d_half; j++) {
        float val = (float)pos * inv_freq[j];
        cos_out->buf[j] = cosf(val) * concentration;
        sin_out->buf[j] = sinf(val) * concentration;
    }

    free(inv_freq);                    
}

// RoPE (n_heads * head_dim)
void ApplyRotary(Tensor *x /*[n_heads*hd]*/, const Tensor *cos /*[hd/2]*/,
                 const Tensor *sin /*[hd/2]*/, int n_heads, int head_dim) {
    int half = head_dim / 2;

    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < half; i++) {
            // Indexing: head h, dim i
            float x1 = x->buf[h * head_dim + i];        // first half
            float x2 = x->buf[h * head_dim + half + i]; // second half

            float c = cos->buf[i];
            float s = sin->buf[i];

            float o1 = x1 * c - x2 * s;
            float o2 = x2 * c + x1 * s;

            x->buf[h * head_dim + i] = o1;
            x->buf[h * head_dim + half + i] = o2;
        }
    }
}

// Attention scores for 1 head: att[0..pos] = q·k_t / sqrt(hd) (+ mask)
void AttnScoresOneHead(const float *q /*[hd]*/,
                       const float *k_cache_layer /*[seq_len*kv_dim]*/,
                       int kv_head_idx, int head_dim, int kv_dim, int seq_len, int pos,
                       const float *mask_row /*[seq_len] or nullptr*/,
                       float *att /*[pos+1]*/) {
    for (int t = 0; t <= pos; t++) {
        // Calculate the correct key position
        const float *k = k_cache_layer + t * kv_dim + kv_head_idx * head_dim;
        double score = 0.0;
        for (int i = 0; i < head_dim; i++) {
            score += q[i] * k[i];
        }
        score /= sqrtf(head_dim);
        if (mask_row) {
            score += mask_row[t];
        }
        att[t] = score;
    }
}

// Weighted sum for 1 head
/*
void AttnWeightedSumOneHead(const float *att,
                            const float *v_cache_layer,
                            int kv_offset_bytes, int head_dim, int pos,
                            float *tb) {
    memset(tb, 0, head_dim * sizeof(float));
    for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        const float *v = v_cache_layer + t * head_dim;
        // get the attention weight for this timestep
        float a = att[t];
        printf("v_cache current: ");
        for (int i = 0; i < min(pos+2, 5); i++) {
          printf("%.6f ", v[i]);
        }
        printf("\n");
        // accumulate the weighted value into tb
        for (int i = 0; i < head_dim; i++) {
            tb[i] += a * v[i];
        }
    }
}
*/

void AttnWeightedSumOneHead(const float *att /*[pos+1]*/,
                            const float *v_cache_layer /*[seq_len*kv_dim]*/,
                            int kv_head_idx, int head_dim, int kv_dim, int pos,
                            float *tb /*[hd]*/) {
    memset(tb, 0, head_dim * sizeof(float));
    for (int t = 0; t <= pos; t++) {
        const float *v = v_cache_layer + t * kv_dim + kv_head_idx * head_dim;
        float a = att[t];
        for (int i = 0; i < head_dim; i++) {
            tb[i] += a * v[i];
        }
    }
}

// Output projection for attention: y = W_o * tb + b_o
void AttnOutProject(const Tensor *tb /*[n_q*hd]*/,
                    const Tensor *W_o /*[hidden, n_q*hd]*/, const Tensor *b_o,
                    Tensor *y /*[hidden]*/, long long layer_offset) {
    const long long n_q_hd = tb->num_elem();
    const long long hidden = y->num_elem();
    const float *w_o_ptr = W_o->buf + 1ll * layer_offset * n_q_hd * hidden;
    const float *b_o_ptr = b_o->buf + 1ll * layer_offset * hidden;
    Linear(tb->buf, n_q_hd, w_o_ptr, b_o_ptr, hidden, y->buf);
}

// Router: r = W_router * t + b_router
void RouterScores(const Tensor *t /*[hidden]*/,
                  const Tensor *W_router /*[n_experts, hidden]*/,
                  const Tensor *b_router, Tensor *r /*[n_experts]*/,
                  long long layer_offset) {
    const long long hidden = t->num_elem();
    const long long n_experts = r->num_elem();
    const float *w_router_ptr = W_router->buf + 1ll * layer_offset * n_experts * hidden;
    const float *b_router_ptr = b_router->buf + 1ll * layer_offset * n_experts;
    // memcpy(w_router_tensor->buf, w->w_router + l * p->n_experts * p->hidden_dim, (size_t)p->n_experts * (size_t)p->hidden_dim * sizeof(float));
    // memcpy(b_router_tensor->buf, w->b_router + l * p->n_experts, (size_t)p->n_experts * sizeof(float));
    Linear(t->buf, hidden, w_router_ptr, b_router_ptr, n_experts, r->buf);
}

void TopK(const Tensor *r /*[n_experts]*/, int k, Tensor *topk_vals /*[k]*/,
          TensorI32 *topk_idx /*[k]*/) {
    int num_experts = r->num_elem();
    int experts_per_token = k;

    // Allocate temp array to store (value, index) pairs
    Pair *pairs = reinterpret_cast<Pair *>(malloc(num_experts * sizeof(Pair)));
    if (!pairs) {
        fprintf(stderr, "Memory allocation failed for pairs\n");
        return;
    }
    for (int i = 0; i < num_experts; ++i) {
        pairs[i].value = r->buf[i];
        pairs[i].index = i;
    }

    // Sort in descending order of value
    qsort(pairs, num_experts, sizeof(Pair), compare_desc);

    // Fill output arrays
    for (int i = 0; i < experts_per_token; ++i) {
        topk_vals->buf[i] = pairs[i].value;
        topk_idx->buf[i] = pairs[i].index;
    }
    free(pairs);
}

// Corrected SwiGLU implementation
void SwiGLU(const Tensor *gate /*[d]*/, const Tensor *up /*[d]*/,
            float clamp_limit, Tensor *out /*[d]*/) {
    // This constant is used in the sigmoid calculation
    const float alpha = 1.702f;
    size_t size = gate->num_elem();

    for (size_t i = 0; i < size; i++) {
        float gate_val = gate->buf[i];
        float up_val = up->buf[i];

        // --- FIX 1: Corrected Clamping ---
        // The 'gate' tensor is only clamped on its upper bound.
        if (gate_val > clamp_limit) gate_val = clamp_limit;

        // The 'up' tensor is clamped on both bounds.
        if (up_val > clamp_limit) up_val = clamp_limit;
        if (up_val < -clamp_limit) up_val = -clamp_limit;

        // --- FIX 2: Corrected SiLU Calculation ---
        // silu(x) = x * σ(αx), where σ(x) is the logistic sigmoid.
        // The 'alpha' constant was missing in your original code.
        gate_val *= (1.0f / (1.0f + expf(-alpha * gate_val)));
        
        // Element-wise multiply with the modified 'up' tensor
        out->buf[i] = gate_val * (up_val + 1.0f);
    }
}

// Expert FFN 1: z = W1 * t + b1
void ExpertFFN1(const Tensor *t, const Tensor *W1, const Tensor *b1,
                Tensor *z, long long layer_offset, long long expert_offset) {
    const long long offset = 1ll * (layer_offset * b1->shape[1] + expert_offset);
    const long long intermediate = z->num_elem();
    const long long hidden = t->num_elem();
    const float *w1_buf = W1->buf + offset * intermediate * hidden;
    const float *b1_buf = b1->buf + offset * intermediate;
    
    Linear(t->buf, hidden, w1_buf, b1_buf, intermediate, z->buf);
}

// Expert FFN 2: y = W2 * swiglu + b2
void ExpertFFN2(const Tensor *swiglu, const Tensor *W2, const Tensor *b2,
                Tensor *y, long long layer_offset, long long expert_offset) {
    const long long offset = 1ll * (layer_offset * b2->shape[1] + expert_offset);
    const long long intermediate = swiglu->num_elem();
    const long long hidden = y->num_elem();
    const float *w2_buf = W2->buf + offset * intermediate * hidden;
    const float *b2_buf = b2->buf + offset * hidden;

    Linear(swiglu->buf, intermediate, w2_buf, b2_buf, hidden, y->buf);
}

// Add residual: x += y
void ResidualAdd(Tensor *x /*[hidden]*/, const Tensor *y /*[hidden]*/) {
    size_t size = x->num_elem();
    for (size_t i = 0; i < size; i++) {
        x->buf[i] += y->buf[i];
    }
}

// Classifier / unembed: logits = W_out * x (no bias), W_out: [vocab, hidden]
void Classifier(const Tensor *x /*[hidden]*/,
                const Tensor *W_out /*[vocab, hidden]*/,
                Tensor *logits /*[vocab]*/) {
    Linear(x->buf, x->num_elem(), W_out->buf, 
           nullptr, logits->num_elem(), logits->buf);
}