#include "model.hpp"
#include <cmath>
#include <cstring>

float *our_forward(Transformer *transformer, int token, int pos) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;

    // Create Tensor wrappers for state buffers
    Tensor *x_tensor = new Tensor({(size_t)p->hidden_dim}, s->x);
    Tensor *t_tensor = new Tensor({(size_t)p->hidden_dim}, s->t);
    Tensor *tb_tensor = new Tensor({(size_t)p->head_dim * p->n_attn_heads}, s->tb);
    Tensor *tb2_tensor = new Tensor({(size_t)p->hidden_dim}, s->tb2);

    Tensor *router_score_tensor = new Tensor({(size_t)p->n_experts}, s->router_score);
    Tensor *topk_v_tensor = new Tensor({(size_t)p->experts_per_token}, s->topk_v);
    TensorI32 *topk_i_tensor = new TensorI32({(size_t)p->experts_per_token}, s->topk_i);

    Tensor *mlp1_out_tensor = new Tensor({2 * (size_t)p->intermediate_dim}, s->mlp1_out);
    Tensor *gate_tensor = new Tensor({(size_t)p->intermediate_dim}, s->gate);
    Tensor *up_tensor = new Tensor({(size_t)p->intermediate_dim}, s->up);
    Tensor *gate_up_tensor = new Tensor({(size_t)p->intermediate_dim}, s->gate_up);
    Tensor *e_agg_tensor = new Tensor({(size_t)p->hidden_dim}, s->e_agg);
    Tensor *qkv_tensor = new Tensor({((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim}, s->qkv);
    Tensor *q_tensor = new Tensor({(size_t)p->n_attn_heads * p->head_dim}, s->q);
    Tensor *k_tensor = new Tensor({(size_t)p->n_kv_heads * p->head_dim});
    Tensor *v_tensor = new Tensor({(size_t)p->n_kv_heads * p->head_dim});
    Tensor *att_tensor = new Tensor({(size_t)p->n_attn_heads, (size_t)p->seq_len}, s->att);
    Tensor *logits_tensor = new Tensor({(size_t)p->vocab_size}, s->logits);

    // Create Tensor wrappers for weight matrices
    Tensor *token_embedding_tensor = new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->token_embedding_table);
    
    // copy the token embedding into x
    EmbeddingLookup(token_embedding_tensor, token, x_tensor);

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {
        // printf("Layer %d\n", l);
        // attention rmsnorm
        Tensor *rms_attn_w_tensor = new Tensor({(size_t)p->hidden_dim}, w->rms_attn_w + l * p->hidden_dim);
        RMSNorm(x_tensor, rms_attn_w_tensor, t_tensor);
        delete rms_attn_w_tensor;

        // key and value point to the kv cache
        long long loff = 1ll * l * p->seq_len * p->head_dim * p->n_kv_heads; // kv cache layer offset
        Tensor *k_cache_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim}, s->key_cache + loff);
        Tensor *v_cache_tensor = new Tensor({(size_t)p->seq_len, (size_t)p->n_kv_heads * p->head_dim}, s->value_cache + loff);

        // QKV projection
        Tensor *w_qkv_tensor = new Tensor({((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * (size_t)p->head_dim, (size_t)p->hidden_dim}, 
                           w->w_qkv + 1ll * l * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim * p->hidden_dim);
        Tensor *b_qkv_tensor = new Tensor({((size_t)p->n_attn_heads + 2 * (size_t)p->n_kv_heads) * p->head_dim}, 
                           w->b_qkv + 1ll * l * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim);
        QKVProject(t_tensor, w_qkv_tensor, b_qkv_tensor, qkv_tensor);
        delete w_qkv_tensor;
        delete b_qkv_tensor;

        // Separate q, k, v
        SplitQKV(qkv_tensor, p->head_dim, p->n_attn_heads, p->n_kv_heads, q_tensor, k_tensor, v_tensor);

        // Store k, v in cache
        /*
        memcpy(s->key_cache + loff + pos * p->n_kv_heads * p->head_dim, 
               k_tensor->buf, p->n_kv_heads * p->head_dim * sizeof(float));
        memcpy(s->value_cache + loff + pos * p->n_kv_heads * p->head_dim, 
               v_tensor->buf, p->n_kv_heads * p->head_dim * sizeof(float));
        */

        /*
        printf("CUSTOM\n");
        printf("q_tensor: ");
        for (int i = 0; i < 5; i++) {
            printf("%.6f ", q_tensor->buf[i]);
        }
        printf("\n");
        printf("k_tensor: ");
        for (int i = 0; i < 5; i++) {
            printf("%.6f ", k_tensor->buf[i]);
        }
        printf("\n");
        printf("v_tensor: ");
        for (int i = 0; i < 5; i++) {
            printf("%.6f ", v_tensor->buf[i]);
        }
        printf("\n");
        */

        // RoPE relative positional encoding
        Tensor *cos_tensor = new Tensor({(size_t)p->head_dim / 2});
        Tensor *sin_tensor = new Tensor({(size_t)p->head_dim / 2});
        RopeComputeCS(pos, *p, cos_tensor, sin_tensor);
        ApplyRotary(q_tensor, cos_tensor, sin_tensor, p->n_attn_heads, p->head_dim);
        ApplyRotary(k_tensor, cos_tensor, sin_tensor, p->n_kv_heads, p->head_dim);
        delete cos_tensor;
        delete sin_tensor;

        // printf("p->n_kv_heads: %d, p->head_dim: %d\n", p->n_kv_heads, p->head_dim);

        memcpy(k_cache_tensor->buf + 1ll * pos * p->n_kv_heads * p->head_dim, 
               k_tensor->buf, p->n_kv_heads * p->head_dim * sizeof(float));
        memcpy(v_cache_tensor->buf + 1ll * pos * p->n_kv_heads * p->head_dim, 
               v_tensor->buf, p->n_kv_heads * p->head_dim * sizeof(float));

        // multihead attention
        int kv_mul = p->n_attn_heads / p->n_kv_heads; // integer multiplier for GQA
        
        // For each head, compute attention scores and weighted sum
        for (int h = 0; h < p->n_attn_heads; h++) {
            // get the query vector for this head
            float *q_head = q_tensor->buf + h * p->head_dim;
            
            // attention scores for this head
            float *att_head = att_tensor->buf + h * p->seq_len;
            
            // Get the key cache for this head group (GQA)
            int kv_head_idx = h / kv_mul;
            // float *k_cache_head = s->key_cache + loff + kv_head_idx * p->head_dim;
            
            // Create mask if needed
            float *mask_row = nullptr;
            if (p->sliding_window > 0 && (l % 2 == 0)) {
                mask_row = s->mask + pos * p->seq_len;
            }

            /*
            if (h < 3) {
                printf("CUSTOM head %d\n", h);
                printf("q: ");
                for (int i = 0; i < 5; i++) {
                    printf("%.6f ", q_head[i]);
                }
                printf("\n");
                printf("att: ");
                for (int i = 0; i < 5; i++) {
                    printf("%.6f ", att_head[i]);
                }
                printf("\n");
                float *skeycache = s->key_cache + loff;
                printf("s->key_cache: ");
                for (int i = 0; i < 5; i++) {
                    printf("%.6f ", skeycache[i]);
                }
                printf("\n");
            }
            */
            
            // Compute attention scores
            AttnScoresOneHead(q_head, k_cache_tensor->buf, kv_head_idx, p->head_dim, p->head_dim * p->n_kv_heads, p->seq_len, pos, mask_row, att_head);
            
            // Add attention sink score
            att_head[pos + 1] = w->attn_sinks[l * p->n_attn_heads + h];

            /*
            if (h < 3) {
                printf("CUSTOM head %d\n", h);
                printf("att: ");
                for (int i = 0; i < min(pos+2, 5); i++) {
                    printf("%.6f ", att_head[i]);
                }
                printf("\n");
            }
            */
            
            // softmax the scores to get attention weights
            Tensor *att_head_tensor = new Tensor({(size_t)pos + 2}, att_head);
            Softmax(att_head_tensor);
            memcpy(att_head, att_head_tensor->buf, (pos+2) * sizeof(float));
            delete att_head_tensor;

            /*
            if (h < 3) {
                printf("CUSTOM head %d\n", h);
                printf("att: ");
                for (int i = 0; i < min(pos+2, 5); i++) {
                    printf("%.6f ", att_head[i]);
                }
                printf("\n");
            }
            */    
            
            // Get the value cache for this head group (GQA)
            float *v_cache_head = v_cache_tensor->buf + 1ll * kv_head_idx * p->head_dim;
            
            // weighted sum of the values
            float *tb_head = tb_tensor->buf + h * p->head_dim;
            // AttnWeightedSumOneHead(att_head, v_cache_head, 0, p->head_dim, pos, tb_head);
            AttnWeightedSumOneHead(att_head, v_cache_tensor->buf, kv_head_idx, p->head_dim, p->head_dim * p->n_kv_heads, pos, tb_head); 

            // Debug output for first few heads
            /*
            if (h < 3) {
                printf("CUSTOM head %d\n", h);
                printf("att: ");
                for (int i = 0; i < min(pos+2, 5); i++) {
                    printf("%.6f ", att_head[i]);
                }
                printf("\n");
                printf("v_cache_head: ");
                for (int i = 0; i < min(pos+2, 5); i++) {
                    printf("%.6f ", v_cache_head[i]);
                }
                printf("\n");
                printf("tb: ");
                for (int i = 0; i < min(p->head_dim, 5); i++) {
                    printf("%.6f ", tb_head[i]);
                }
                printf("\n");
            }
            */
        }

        // final matmul to get the output of the attention
        Tensor *w_o_tensor = new Tensor({(size_t)p->hidden_dim, (size_t)p->n_attn_heads * p->head_dim}, 
                         w->w_o + 1ll * l * p->hidden_dim * p->n_attn_heads * p->head_dim);
        Tensor *b_o_tensor = new Tensor({(size_t)p->hidden_dim}, w->b_o + 1ll * l * p->hidden_dim);
        AttnOutProject(tb_tensor, w_o_tensor, b_o_tensor, tb2_tensor);
        delete w_o_tensor;
        delete b_o_tensor;

        // residual connection back into x
        ResidualAdd(x_tensor, tb2_tensor);

        // ffn rmsnorm
        Tensor *rms_ffn_w_tensor = new Tensor({(size_t)p->hidden_dim}, w->rms_ffn_w + 1ll * l * p->hidden_dim);
        RMSNorm(x_tensor, rms_ffn_w_tensor, t_tensor);
        delete rms_ffn_w_tensor;

        /*
        if (l < 3) {
            printf("CUSTOM:\n");
            printf("x_tensor: ");
            for (int i=0; i<5; i++) {
                printf("%.6f ", x_tensor->buf[i]);
            }
            printf("\n");
            printf("t_tensor: ");
            for (int i=0; i<5; i++) {
                printf("%.6f ", t_tensor->buf[i]);
            }
            printf("\n");
        }
        */

        // MoE routing
        Tensor *w_router_tensor = new Tensor({(size_t)p->n_experts, (size_t)p->hidden_dim}, 
                              w->w_router + l * p->n_experts * p->hidden_dim);
        Tensor *b_router_tensor = new Tensor({(size_t)p->n_experts}, w->b_router + l * p->n_experts);
        RouterScores(t_tensor, w_router_tensor, b_router_tensor, router_score_tensor);
        delete w_router_tensor;
        delete b_router_tensor;
        
        // Select top-k experts
        TopK(router_score_tensor, p->experts_per_token, topk_v_tensor, topk_i_tensor);
        
        // Normalize selected experts using softmax
        Softmax(topk_v_tensor);

        // Route the tokens to their corresponding top-k experts
        memset(e_agg_tensor->buf, 0, p->hidden_dim * sizeof(float));
        
        for (int idx = 0; idx < p->experts_per_token; idx++) {
            int e = topk_i_tensor->buf[idx];
            float expert_w = topk_v_tensor->buf[idx];

            // printf("e: %d\n", e);

            float *wmlp1_check = w->w_mlp1 + 1ll * (l * p->n_experts + e) * 2 * p->intermediate_dim * p->hidden_dim;
            
            // Expert FFN 1: gate_up = W1 * t + b1
            Tensor *w_mlp1_tensor = new Tensor({2 * (size_t)p->intermediate_dim, (size_t)p->hidden_dim}, 
                                w->w_mlp1 + 1ll * (l * p->n_experts + e) * 2 * p->intermediate_dim * p->hidden_dim);
            Tensor *b_mlp1_tensor = new Tensor({2 * (size_t)p->intermediate_dim}, 
                                w->b_mlp1 + 1ll *  (l * p->n_experts + e) * 2 * p->intermediate_dim);
    
            ExpertFFN1(t_tensor, w_mlp1_tensor, b_mlp1_tensor, mlp1_out_tensor);

            /*
            printf("t_tensor: ");
            for (int i=0; i<5; i++) {
                printf("%.6f ", t_tensor->buf[i]);
            }
            printf("\n");
            printf("w_mlp1_tensor: ");
            for (int i=0; i<5; i++) {
                printf("%.6f ", w_mlp1_tensor->buf[i]);
            }
            printf("\n");
            printf("b_mlp1_tensor: ");
            for (int i=0; i<5; i++) {
                printf("%.6f ", b_mlp1_tensor->buf[i]);
            }
            printf("\n");
            */

            delete w_mlp1_tensor;
            delete b_mlp1_tensor;
            
            // Split into gate and up
            for (int j = 0; j < p->intermediate_dim; j++) {
                gate_tensor->buf[j] = mlp1_out_tensor->buf[2 * j];
                up_tensor->buf[j] = mlp1_out_tensor->buf[2 * j + 1];
            }

            /*
            printf("gate: ");
            for (int i=0; i<5; i++) {
                printf("%.6f ", gate_tensor->buf[i]);
            }
            printf("\n");
            printf("up: ");
            for (int i=0; i<5; i++) {
                printf("%.6f ", up_tensor->buf[i]);
            }
            printf("\n");
            */
                
            // SwiGLU non-linearity
            SwiGLU(gate_tensor, up_tensor, p->swiglu_limit, gate_up_tensor);
        
            /*
            printf("gate_up: ");
            for (int i=0; i<5; i++) {
                printf("%.6f ", gate_up_tensor->buf[i]);
            }
            printf("\n");
            */
            
            // Expert FFN 2: y = W2 * swiglu + b2
            Tensor *w_mlp2_tensor = new Tensor({(size_t)p->hidden_dim, (size_t)p->intermediate_dim}, 
                                w->w_mlp2 + 1ll * (l * p->n_experts + e) * p->hidden_dim * p->intermediate_dim);
            Tensor *b_mlp2_tensor = new Tensor({(size_t)p->hidden_dim}, 
                                w->b_mlp2 + 1ll * (l * p->n_experts + e) * p->hidden_dim);
            ExpertFFN2(gate_up_tensor, w_mlp2_tensor, b_mlp2_tensor, tb2_tensor);
            delete w_mlp2_tensor;
            delete b_mlp2_tensor;
        
            /*
            printf("tb2_tensor: ");
            for (int i=0; i<5; i++) {
                printf("%.6f ", tb2_tensor->buf[i]);
            }
            printf("\n");
            */
            
            // aggregate topk experts using weighted sum
            for (int i = 0; i < p->hidden_dim; i++) {
                e_agg_tensor->buf[i] += tb2_tensor->buf[i] * expert_w;
            }

            /*
            printf("e_agg_tensor: ");
            for (int i=0; i<5; i++) {
                printf("%.6f ", e_agg_tensor->buf[i]);
            }
            printf("\n");
            */
        }

        /*
        printf("e_agg_tensor: ");
        for (int i=0; i<5; i++) {
            printf("%.6f ", e_agg_tensor->buf[i]);
        }
        printf("\n");
        */

        // residual connection
        ResidualAdd(x_tensor, e_agg_tensor);
    
        /*
        printf("x after expert: ");
        for (int i=0; i<5; i++) {
            printf("%.6f ", x_tensor->buf[i]);
        }
        printf("\n");
        */

        // if (l > 10) exit(1);

        memcpy(s->key_cache + loff, k_cache_tensor->buf, (size_t)p->seq_len * (size_t)p->n_kv_heads * p->head_dim * sizeof(float)); // k_cache
        memcpy(s->value_cache + loff, v_cache_tensor->buf, (size_t)p->seq_len * (size_t)p->n_kv_heads * p->head_dim * sizeof(float)); // v_cache

        delete k_cache_tensor;
        delete v_cache_tensor;
    }
    
    // final rmsnorm
    Tensor *rms_out_w_tensor = new Tensor({(size_t)p->hidden_dim}, w->rms_out_w);
    RMSNorm(x_tensor, rms_out_w_tensor, x_tensor);
    delete rms_out_w_tensor;

    // classifier into logits
    Tensor *w_out_tensor = new Tensor({(size_t)p->vocab_size, (size_t)p->hidden_dim}, w->out);
    Classifier(x_tensor, w_out_tensor, logits_tensor);
    delete w_out_tensor;

    memcpy(s->att, att_tensor->buf, (size_t)p->n_attn_heads * p->seq_len * sizeof(float)); // att_head
    memcpy(s->logits, logits_tensor->buf, (size_t)p->vocab_size * sizeof(float)); // s_logits
    memcpy(s->tb, tb_tensor->buf, (size_t)p->head_dim * p->n_attn_heads * sizeof(float)); // s->tb

    // Clean up all allocated tensors
    delete x_tensor;
    delete t_tensor;
    delete tb_tensor;
    delete tb2_tensor;
    delete router_score_tensor;
    delete topk_v_tensor;
    delete topk_i_tensor;
    delete mlp1_out_tensor;
    delete gate_tensor;
    delete up_tensor;
    delete gate_up_tensor;
    delete e_agg_tensor;
    delete qkv_tensor;
    delete q_tensor;
    delete k_tensor;
    delete v_tensor;
    delete att_tensor;
    delete logits_tensor;
    delete token_embedding_tensor;
    
    return s->logits;
}