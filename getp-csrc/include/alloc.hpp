#pragma once

#include "config.hpp"
#include "alloc_utils.hpp"

void our_init(Transformer *transformer, OurTransformerWeights *weights, OurRunState *rs,
              hipStream_t *total_streams, hipTotalEvents_t *total_events);
void our_free(OurTransformerWeights *weights, OurRunState *rs, hipStream_t *total_streams,
              hipTotalEvents_t *total_events);

void our_compute_concentration_and_inv_freq(float base, int head_dim, float scaling_factor,
                                            float initial_context_length, float ntk_beta,
                                            float ntk_alpha, float *concentration_out,
                                            float *inv_freq_out  // length head_dim/2
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
    float low = d_half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(base);
    float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);

    assert(0 < low && low < high && high < d_half - 1);

    // interpolation = 1 / (scaling_factor * freq)
    // extrapolation = 1 / freq
    for (int i = 0; i < d_half; i++) {
      float interpolation = 1.0f / (scaling_factor * freq[i]);
      float extrapolation = 1.0f / freq[i];

      float ramp = ((float)i - low) / (high - low);
      if (ramp < 0)
        ramp = 0;
      if (ramp > 1)
        ramp = 1;

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

void rope_precompute_cs(Config *p, Tensor *cos_all_out, Tensor *sin_all_out, hipStream_t stream) {
  int seq_len = p->seq_len;
  int head_dim = p->head_dim;
  int d_half = head_dim / 2;

  float base = p->rope_theta;
  float scaling_factor = p->rope_scaling_factor;
  float initial_context_length = p->initial_context_length;
  float ntk_beta = 32.0f;
  float ntk_alpha = 1.0f;

  float concentration;
  float *inv_freq = (float *)malloc(d_half * sizeof(float));
  our_compute_concentration_and_inv_freq(base, head_dim, scaling_factor, initial_context_length,
                                         ntk_beta, ntk_alpha, &concentration, inv_freq);

  for (int pos = 0; pos < seq_len; pos++) {
    for (int j = 0; j < d_half; j++) {
      float val = (float)pos * inv_freq[j];

      int index = pos * d_half + j;

      cos_all_out->buf[index] = cosf(val) * concentration;
      sin_all_out->buf[index] = sinf(val) * concentration;
    }
  }

  free(inv_freq);

  cos_all_out->to_device(stream);
  sin_all_out->to_device(stream);
  CHECK_HIP(hipStreamSynchronize(stream));
}
