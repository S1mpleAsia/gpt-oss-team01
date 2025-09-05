// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"

#include "src/tensor.cpp"
#include "src/layer.cpp"
#include "src/layer_hip.cpp"
#include "src/layer_hip_batch.cpp"
#include "src/flash_attn_hip.cpp"
#include "src/model.cpp"
#include "src/alloc.cpp"
#include "src/utils.cpp"
#include "include/utils.hpp"
#include "include/config.hpp"
#include "include/parallel.hpp"
#include "src/parallel.cpp"
#include "vector"

#ifndef GETP_RUN
#define GETP_RUN

Context context[DP_120B];

bool multi_gpu = false;
long long getp_generate_120b(Context *context, Tokenizer *tokenizer, Sampler *sampler,
                             const char *input_seq, int *output_tokens, int steps);

#define RUN_BATCH

OurTransformerWeights *weights;
OurRunState *rs;

Config *public_config;
Transformer *public_transformer;
Tokenizer *public_tokenizer;
Sampler *public_sampler;
Requests *public_requests;

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the warm-up process
  // TODO:
  // - Memory allocation
  // - Load model
  // - ...
  public_config = &transformer->config;
  public_transformer = transformer;
  public_tokenizer = tokenizer;

  if (!multi_gpu) {
    int actualGPUs = 0;
    CHECK_HIP(hipGetDeviceCount(&actualGPUs));

    if (DP_20B != actualGPUs) {
      fprintf(stderr, "Error: Required %d GPUs, but only %d available.\n", DP_20B, actualGPUs);
      exit(1);
    }

    weights = new OurTransformerWeights[DP_20B];
    rs = new OurRunState[DP_20B];
    our_init(transformer, weights, rs);

    // weights = new OurTransformerWeights;
    // rs = new OurRunState;
    // p = &transformer->config;
    // our_init(transformer, weights, rs);
  } else {
    printf("Multi GPU here...\n");
    std::vector<int> gpu_0 = {0, 1, 2, 3};
    // std::vector<int> gpu_1 = {2, 3};

    context[0].init(transformer, gpu_0);
    // context[1].init(transformer, gpu_1);
  }
}

void setup(Sampler *sampler, Requests *requests) {
  public_sampler = sampler;
  public_requests = requests;
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the finish process
  // TODO:
  // - Memory deallocation
  // - Unload model
  // - ...

  if (!multi_gpu) {
    our_free(weights, rs);
    delete weights;
    delete rs;
  } else {
    printf("Destroy multi GPU\n");
    context[0].destroy();
    // context[1].destroy();
  }
}

#ifndef RUN_BATCH
long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                               const char *input_seq, int *output_tokens, int steps) {
  // <|start|>: 200006
  // <|end|>: 200007
  // <|return|>: 200002
  // <|message|>: 200008
  // <|channel|>: 200005
  // <|constrain|>: 200003
  // <|endoftext|>: 199999

  // Inference here
  const char *empty_prompt = "";
  if (input_seq == NULL) {
    input_seq = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens =
    (int *)malloc((strlen(input_seq) + 3) * sizeof(int));  // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, input_seq, -1, -1, prompt_tokens, &num_prompt_tokens,
         transformer->config.initial_context_length);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  int next;                      // will store the next token in the sequence
  int token = prompt_tokens[0];  // kick off with the first token in the prompt
  int pos = 0;                   // position in the sequence

  // print the very first token
  // should be removed
  const char *first_piece = decode_piece(tokenizer, 200006, token);
  safe_printf(first_piece);
  fflush(stdout);

  while (pos < steps) {
    // forward the transformer to get logits for the next token
    float *logits = forward_gpu_20b(p, weights, rs, token, pos);
    // float *logits = forward(transformer, token, pos); <---- real code from run.cpp

    // advance the state machine
    {
      // GpuTimer timer("sample");
      pos++;
      if (pos < num_prompt_tokens) {
        // if we are still processing the input prompt, force the next prompt
        // token
        next = prompt_tokens[pos];
      } else {
        // otherwise sample the next token from the logits
        next = sample(sampler, logits);
        // save the output token, it will be printed to file
        output_tokens[pos - num_prompt_tokens] = next;
      }

// --- MODIFICATION START ---
// This single block replaces the two original PRINT_LOGITS blocks.
#ifdef PRINT_LOGITS
      // Decode the next token to get its string representation
      const char *piece = decode_piece(tokenizer, token, next);

      // Print in the requested format
      printf("batch id 0 --> ");
      safe_printf(piece);
      printf("logits: ");
      for (int j = 0; j < 5; j++) {
        printf("%.6f ", logits[j]);
      }
      printf("\n");
      fflush(stdout);
#endif
      // --- MODIFICATION END ---

      // data-dependent terminating condition: the EOS (=199999 or =200002) token
      // delimits sequences
      if (next == 199999 || next == 200002) {
        break;
      }

      token = next;
    }
  }

// should be removed
#ifdef PRINT_LOGITS
  printf("\n");
#endif

  // Marker for end of sequence
  output_tokens[pos - num_prompt_tokens + 1] = -1;

  free(prompt_tokens);

  return pos - num_prompt_tokens + 1;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                    Requests *requests) {
  if (!multi_gpu) {
    long long num_token_out = 0;
    for (int idx = 0; idx < requests->num_reqs; ++idx) {
      const char *input_seq = get_str_req_ptr(requests, idx);
      int *output_tokens = get_tok_gen_ptr(requests, idx);
      num_token_out += simple_getp_generate(transformer, tokenizer, sampler, input_seq,
                                            output_tokens, requests->max_seq_len);
    }
    return num_token_out;
  } else {
    long long num_token_out = 0;
    for (int idx = 0; idx < requests->num_reqs; ++idx) {
      const char *input_seq = get_str_req_ptr(requests, idx);
      int *output_tokens = get_tok_gen_ptr(requests, idx);
      num_token_out += getp_generate_120b(&context[0], tokenizer, sampler, input_seq, output_tokens,
                                          requests->max_seq_len);
    }
    return num_token_out;
  }

  // long long num_token_out = 0;
  // pthread_t threads[NUM_REPLICAS];
  // ThreadArgs args[NUM_REPLICAS];
  // pthread_mutex_t mutex;
  // pthread_mutex_init(&mutex, NULL);

  // for (int i = 0; i < NUM_REPLICAS; i++) {
  //   args[i].id = i;
  //   args[i].tokenizer = tokenizer;
  //   args[i].sampler = sampler;
  //   args[i].reqs = requests;
  //   args[i].total_token_out = &num_token_out;
  //   args[i].mutex = &mutex;

  //   pthread_create(&threads[i], NULL, thread_handler, (void *)&args[i]);
  // }

  // for (int i = 0; i < NUM_REPLICAS; i++) {
  //   pthread_join(threads[i], NULL);
  // }

  // pthread_mutex_destroy(&mutex);
  // return num_token_out;
}
#else

// long long getp_generate_120b(Context *context, Tokenizer *tokenizer, Sampler *sampler,
//                              const char *input_seq, int *output_tokens, int steps) {
//   // <|start|>: 200006
//   // <|end|>: 200007
//   // <|return|>: 200002
//   // <|message|>: 200008
//   // <|channel|>: 200005
//   // <|constrain|>: 200003
//   // <|endoftext|>: 199999

//   // Inference here
//   const char *empty_prompt = "";
//   if (input_seq == NULL) {
//     input_seq = empty_prompt;
//   }

//   // encode the (string) prompt into tokens sequence
//   int num_prompt_tokens = 0;
//   int *prompt_tokens =
//     (int *)malloc((strlen(input_seq) + 3) * sizeof(int));  // +3 for '\0', ?BOS, ?EOS
//   encode(tokenizer, input_seq, -1, -1, prompt_tokens, &num_prompt_tokens,
//          context->transformer->config.initial_context_length);
//   if (num_prompt_tokens < 1) {
//     fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
//     exit(EXIT_FAILURE);
//   }

//   // start the main loop
//   int next;                      // will store the next token in the sequence
//   int token = prompt_tokens[0];  // kick off with the first token in the prompt
//   int pos = 0;                   // position in the sequence

//   // print the very first token
//   // should be removed
//   const char *first_piece = decode_piece(tokenizer, 200006, token);
//   safe_printf(first_piece);
//   fflush(stdout);

//   while (pos < steps) {
//     // forward the transformer to get logits for the next token
//     // printf("Generate token %d\n", pos);
//     // fflush(stdout);
//     float *logits = forward_gpu_120b(context, token, pos);
//     // float *logits = forward(transformer, token, pos); <---- real code from run.cpp

//     // printf("logits: ");
//     // for (int i = 0; i < 5; i++) {
//     //   printf("%.6f ", logits[i]);
//     // }
//     // printf("\n");
//     // exit(1);

//     // advance the state machine
//     {
//       // GpuTimer timer("sample");
//       pos++;
//       if (pos < num_prompt_tokens) {
//         // if we are still processing the input prompt, force the next prompt
//         // token
//         next = prompt_tokens[pos];
//       } else {
//         // otherwise sample the next token from the logits
//         next = sample(sampler, logits);
//         // save the output token, it will be printed to file
//         output_tokens[pos - num_prompt_tokens] = next;
//       }

//       // data-dependent terminating condition: the EOS (=199999 or =200002) token
//       // delimits sequences
//       if (next == 199999 || next == 200002) {
//         break;
//       }

//       // print the token as string, decode it with the Tokenizer object
//       // should be removed
//       const char *piece = decode_piece(tokenizer, token, next);
//       safe_printf(piece);  // same as printf("%s", piece), but skips "unsafe" bytes
//       fflush(stdout);

//       token = next;
//     }
//   }

//   // should be removed
//   printf("\n");

//   // Marker for end of sequence
//   output_tokens[pos - num_prompt_tokens + 1] = -1;

//   free(prompt_tokens);

//   return pos - num_prompt_tokens + 1;
// }

long long batched_getp_generate(const std::vector<const char *> &input_batch,
                                const std::vector<int *> &output_batch, int steps, int flow_id) {
  const int batch_size = input_batch.size();
  if (batch_size == 0)
    return 0;

  vector<vector<int>> batch_prompt_tokens(batch_size);
  vector<int> num_prompt_tokens(batch_size);

  for (int i = 0; i < batch_size; i++) {
    const char *input_seq = input_batch[i] ? input_batch[i] : "";
    int *prompt_tokens_buffer = (int *)malloc(strlen((input_seq) + 3) * sizeof(int));
    int count = 0;
    encode(public_tokenizer, input_seq, -1, -1, prompt_tokens_buffer, &count,
           public_config->initial_context_length);

    if (count < 1) {
      fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
      exit(EXIT_FAILURE);
    } else {
      batch_prompt_tokens[i] = vector<int>(prompt_tokens_buffer, prompt_tokens_buffer + count);
      num_prompt_tokens[i] = count;
    }

    free(prompt_tokens_buffer);
  }

  vector<int> current_tokens(batch_size);
  vector<int> current_pos(batch_size, 0);
  vector<bool> active(batch_size, true);
  int active_count = batch_size;
  long long total_generate_tokens = 0;

  for (int i = 0; i < batch_size; i++) {
    current_tokens[i] = batch_prompt_tokens[i][0];
  }

  for (int pos = 0; pos < steps && active_count > 0; ++pos) {
    float *batch_logits = nullptr;
    if (!multi_gpu) {
      batch_logits = forward_gpu_20b_batched(public_config, weights, rs, current_tokens.data(), pos,
                                             batch_size, flow_id);
    } else {
      batch_logits = forward_gpu_120b(&context[flow_id], current_tokens.data(), pos);
    }

    for (int i = 0; i < batch_size; i++) {
      if (!active[i])
        continue;

      float *logits = batch_logits + 1ll * i * public_config->vocab_size;

      int next_token;
      if (current_pos[i] < num_prompt_tokens[i] - 1) {
        next_token = batch_prompt_tokens[i][current_pos[i] + 1];
      } else {
        next_token = sample(public_sampler, logits);
        output_batch[i][current_pos[i] - (num_prompt_tokens[i] - 1)] = next_token;
      }

// Print the logits in the desired format if the flag is enabled
#ifdef PRINT_LOGITS
      // Decode the next token to get its string representation
      const char *piece = decode_piece(public_tokenizer, current_tokens[i], next_token);

      // Use a critical section for printing to prevent interleaved output
      // #pragma omp critical
      // {
      // printf("batch id %d --> ", i);
      safe_printf(piece);
      // printf("logits: ");
      // for (int j = 0; j < 5; j++) {
      // printf("%.6f ", logits[j]);
      // }
      // printf("\n");
      fflush(stdout);
//       }
#endif

      if (next_token == 199999 || next_token == 200002) {
        // #pragma omp critical
        // {
        if (active[i]) {
          active[i] = false;
          active_count--;
        }
        // }
      }

      current_tokens[i] = next_token;
      current_pos[i]++;
    }
  }

// should be removed
#ifdef PRINT_LOGITS
  printf("\n");
#endif

  for (int i = 0; i < batch_size; i++) {
    int generated_len = current_pos[i] - num_prompt_tokens[i];
    if (generated_len < 0)
      generated_len = 0;

    output_batch[i][generated_len + 1] = -1;
    total_generate_tokens += generated_len;
  }

  return total_generate_tokens;
}

void *thread_handler(void *arg) {
  ThreadArgs *args = (ThreadArgs *)arg;

  long long local_token_count = 0ll;
  int id = args->id;
  int max_seq_len = public_requests->max_seq_len;
  int end_idx = args->end_idx;

  CHECK_HIP(hipSetDevice(id));

  for (int i = args->start_idx; i < end_idx; i += BATCH_SIZE) {
    int current_size = min(BATCH_SIZE, end_idx - i);

    vector<const char *> input_batch;
    vector<int *> output_batch;

    for (int j = 0; j < current_size; j++) {
      input_batch.push_back(get_str_req_ptr(public_requests, i + j));
      output_batch.push_back(get_tok_gen_ptr(public_requests, i + j));
    }

    local_token_count += batched_getp_generate(input_batch, output_batch, max_seq_len, id);
  }

  *(args->local_token_ptr) = local_token_count;

  return nullptr;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                    Requests *requests) {
  setup(sampler, requests);

  if (!multi_gpu) {
    long long num_token_out = 0;
    pthread_t threads[DP_20B];
    ThreadArgs args[DP_20B];
    long long tokens_out[DP_20B];

    int total_reqs = public_requests->num_reqs;
    int chunk_size = (total_reqs + DP_20B - 1) / DP_20B;

    for (int i = 0; i < DP_20B; i++) {
      args[i].id = i;
      args[i].local_token_ptr = &tokens_out[i];
      args[i].start_idx = i * chunk_size;
      args[i].end_idx = min((i + 1) * chunk_size, total_reqs);

      pthread_create(&threads[i], NULL, thread_handler, (void *)&args[i]);
    }

    for (int i = 0; i < DP_20B; i++) {
      pthread_join(threads[i], NULL);
      num_token_out += tokens_out[i];
    }

    printf("Total tokens generated: %lld\n", num_token_out);
    fflush(stdout);

    return num_token_out;
  } else {
    long long num_token_out = 0;
    pthread_t threads[DP_120B];
    ThreadArgs args[DP_120B];
    long long tokens_out[DP_120B];

    int total_reqs = public_requests->num_reqs;
    int chunk_size = (total_reqs + DP_120B - 1) / DP_120B;

    for (int i = 0; i < DP_120B; i++) {
      args[i].id = i;
      args[i].local_token_ptr = &tokens_out[i];
      args[i].start_idx = i * chunk_size;
      args[i].end_idx = min((i + 1) * chunk_size, total_reqs);

      pthread_create(&threads[i], NULL, thread_handler, (void *)&args[i]);
    }

    for (int i = 0; i < DP_120B; i++) {
      pthread_join(threads[i], NULL);
      num_token_out += tokens_out[i];
    }

    printf("Total tokens generated: %lld\n", num_token_out);
    fflush(stdout);

    return num_token_out;
  }
}
#endif

#endif  // GETP_RUN
