#include "../include/model.hpp"
#include <cmath>
#include <cstring>

float *our_forward(Config *p, OurTransformerWeights *weights, OurRunState *rs, int token, int pos) {

    // copy the token embedding into x
    // EmbeddingLookup(weights->token_embedding_table, token, rs->x);
    EmbeddingLookupGPU(weights->token_embedding_table, token, rs->x, false);

    long long loff_one = 1ll * p->seq_len * p->head_dim * p->n_kv_heads;

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {
        // printf("Layer %d\n", l);
        // attention rmsnorm
        // RMSNorm(rs->x, weights->rms_attn_w, rs->t, 1ll * l);
        RMSNormGPU(rs->x, weights->rms_attn_w, rs->t, 1ll * l, false, false);

        // key and value point to the kv cache
        long long loff = 1ll * l * loff_one; // kv cache layer offset

        // QKV projection 
        // QKVProject(rs->t, weights->w_qkv, weights->b_qkv, rs->qkv, 1ll * l);
        QKVGemmGPU(rs->t, weights->w_qkv, weights->b_qkv, rs->qkv,
                    1ll * l, false, false); // This kernel diverges the most

        // Separate q, k, v
        // SplitQKV(rs->qkv, p->head_dim, p->n_attn_heads, p->n_kv_heads, rs->q, rs->k, rs->v);
        // SplitQKVGPU(rs->qkv, p->head_dim, p->n_attn_heads, p->n_kv_heads, rs->q, rs->k, rs->v, false, true, true, true);
        QKVEpilogueSplitRoPECacheGPU(
            rs->qkv, rs->q, rs->k, rs->v, cos_tensor, sin_tensor,
            p->head_dim, p->n_attn_heads, p->n_kv_heads, pos,
            false, false, false, false
        );

        // Store k, v in cache
        // memcpy(rs->key_cache->buf + loff + 1ll * pos * p->n_kv_heads * p->head_dim, rs->k->buf, p->n_kv_heads * p->head_dim * sizeof(float));
        // memcpy(rs->value_cache->buf + loff + 1ll * pos * p->n_kv_heads * p->head_dim, rs->v->buf, p->n_kv_heads * p->head_dim * sizeof(float));
        MemCpy_Tensor(rs->key_cache, rs->k, loff + 1ll * pos * p->n_kv_heads * p->head_dim, 0, (size_t)p->n_kv_heads * p->head_dim, false, true);
        MemCpy_Tensor(rs->value_cache, rs->v, loff + 1ll * pos * p->n_kv_heads * p->head_dim, 0, (size_t)p->n_kv_heads * p->head_dim, false, true);

        // multihead attention
        int kv_mul = p->n_attn_heads / p->n_kv_heads; // integer multiplier for GQA

        /*
        AttnScoresAllHeads(rs->key_cache, rs->q, rs->att, weights->attn_sinks,
                            rs->mask, loff_one, 1ll * l, p->n_attn_heads,
                            kv_mul, p->head_dim, p->head_dim * p->n_kv_heads,
                            p->seq_len, p->sliding_window, pos);

        AttnWeightedSumAllHeads(rs->value_cache, rs->q, rs->att, rs->tb, 
                                loff, p->n_attn_heads, kv_mul, p->head_dim,
                                p->head_dim * p->n_kv_heads, p->seq_len, pos);
        */
        SingleQueryAttentionGPU(rs->q, rs->key_cache, rs->value_cache, rs->mask,
                                weights->attn_sinks, rs->tb, p->head_dim,
                                p->n_attn_heads, kv_mul,
                                p->head_dim * p->n_kv_heads, p->seq_len, p->sliding_window, pos, 1ll * l, false, false,
                                false, false, false);


        // final matmul to get the output of the attention
        // AttnOutProject(rs->tb, weights->w_o, weights->b_o, rs->tb2, 1ll * l);
        AttnOutProjectGPU(rs->tb, weights->w_o, weights->b_o, rs->tb2,
                            1ll * l, false, false);

        // residual connection back into x
        // ResidualAdd(rs->x, rs->tb2);
        AddVectorGPU(rs->x, rs->tb2, false, false, false); // equals residual add

        // ffn rmsnorm
        // RMSNorm(rs->x, weights->rms_ffn_w, rs->t, 1ll * l);
        RMSNormGPU(rs->x, weights->rms_ffn_w, rs->t, 1ll * l, false, true);
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
        // RouterScores(rs->t, weights->w_router, weights->b_router, rs->router_score, 1ll * l);
        RouterGemmGPU(weights->w_router, rs->t, weights->b_router,
                        rs->router_score, 1ll * l, false, false);
        
        // Select top-k experts
        // TopK(rs->router_score, p->experts_per_token, rs->topk_v, rs->topk_i);
        
        // Normalize selected experts using softmax
        // Softmax(rs->topk_v->buf, rs->topk_v->num_elem());

        TopKSoftmaxGPU(rs->router_score, rs->topk_v, rs->topk_i,
                        false, true, true);

        // Route the tokens to their corresponding top-k experts
        // memset(rs->e_agg->buf, 0, p->hidden_dim * sizeof(float));
        MemSet_Tensor(rs->e_agg, 0, true, true);

        // printf("Layer: %d\n", l);
        ExpertFFN1_Total(rs->t, weights->w_mlp1, weights->b_mlp1, 
                        rs->mlp1_out, rs->topk_i, 1ll * l);

        // printf("Finish ExpertFFN1...\n");

        for (int j = 0; j < p->experts_per_token * p->intermediate_dim; j++) {
            rs->gate->buf[j] = rs->mlp1_out->buf[2 * j];
            rs->up->buf[j] = rs->mlp1_out->buf[2 * j + 1];
        }

        // printf("Finish to gate&up...\n");
        
        SwiGLU(rs->gate, rs->up, p->swiglu_limit, rs->gate_up);
        
        // printf("Finish SwiGLU...\n");

        ExpertFFN2_Total(rs->gate_up, weights->w_mlp2, weights->b_mlp2, 
                        rs->tb3, rs->topk_i, 1ll * l);
        
        for (int i = 0; i < p->hidden_dim; i++) {
            float sum_cur = 0.0f;
            for (int j = 0; j < p->experts_per_token; j++) {
                float expert_w = rs->topk_v->buf[j];
                sum_cur += rs->tb3->buf[j * p->hidden_dim + i] * expert_w;
            }
            rs->e_agg->buf[i] += sum_cur;
        }
        
        // printf("Finish combined...\n");
        
        /*
        printf("e_agg_tensor: ");
        for (int i=0; i<5; i++) {
            printf("%.6f ", rs->e_agg->buf[i]);
        }
        printf("\n");
        */

        // residual connection
        // ResidualAdd(rs->x, rs->e_agg);
        AddVectorGPU(rs->x, rs->e_agg, false, true, false); // equals residual add
    
        /*
        printf("x after expert: ");
        for (int i=0; i<5; i++) {
            printf("%.6f ", x_tensor->buf[i]);
        }
        printf("\n");
        */

        // if (l > 10) exit(1);
    }
    
    // final rmsnorm
    // RMSNorm(rs->x, weights->rms_out_w, rs->x, 0ll);
    RMSNormGPU(rs->x, weights->rms_out_w, rs->x, 0ll, false, false);

    // classifier into logits
    // Classifier(rs->x, weights->out, rs->logits);
    ClassifierGemmGPU(weights->out, rs->x, rs->logits, false, true);
    
    return rs->logits->buf;
}

