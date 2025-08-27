// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"

#include "src/tensor.cpp"
#include "include/tensor.hpp"
#include "src/layer.cpp"
#include "include/layer.hpp"
#include "src/layer_hip.cpp"
#include "include/layer_hip.hpp"
#include "src/model.cpp"
#include "include/model.hpp"
#include "src/alloc.cpp"
#include "include/alloc.hpp"
#include "src/utils.cpp"
#include "include/utils.hpp"
#include "include/config.hpp"
#include "include/parallel.hpp"
#include "src/parallel.cpp"
#include "vector"

#ifndef GETP_RUN
#define GETP_RUN

#define NUM_REPLICAS 1

Context context[NUM_REPLICAS];

bool multi_gpu = true;
long long getp_generate_120b(Context *context, Tokenizer *tokenizer, Sampler *sampler,
                             const char *input_seq, int *output_tokens, int steps);

// #define RUN_BATCH

struct ThreadArgs {
  int id;
  Tokenizer *tokenizer;
  Sampler *sampler;
  Requests *reqs;
  long long *total_token_out;
  pthread_mutex_t *mutex;
};

void *thread_handler(void *arg) {
  ThreadArgs *args = (ThreadArgs *)arg;
  int id = args->id;
  Context *ctx = &context[id];

  int start_idx = (id == 0) ? 0 : args->reqs->num_reqs / 2;
  int end_idx = (id == 0) ? args->reqs->num_reqs / 2 : args->reqs->num_reqs;
  long long local_token_count = 0;

  for (int i = start_idx; i < end_idx; i++) {
    const char *input_seq = get_str_req_ptr(args->reqs, i);
    int *output_tokens = get_tok_gen_ptr(args->reqs, i);
    local_token_count += getp_generate_120b(ctx, args->tokenizer, args->sampler, input_seq,
                                            output_tokens, args->reqs->max_seq_len);
  }

  pthread_mutex_lock(args->mutex);
  *(args->total_token_out) += local_token_count;
  pthread_mutex_lock(args->mutex);

  return nullptr;
}

OurTransformerWeights *weights;
OurRunState *rs;
Config *p;

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the warm-up process
  // TODO:
  // - Memory allocation
  // - Load model
  // - ...

  if (!multi_gpu) {
    weights = new OurTransformerWeights;
    rs = new OurRunState;
    p = &transformer->config;
    our_init(transformer, weights, rs);
  } else {
    printf("Multi GPU here...\n");
    std::vector<int> gpu_0 = {0, 1};
    // std::vector<int> gpu_1 = {2, 3};

    context[0].init(transformer, gpu_0);
    // context[1].init(transformer, gpu_1);
  }
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
long long getp_generate_120b(Context *context, Tokenizer *tokenizer, Sampler *sampler,
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
         context->transformer->config.initial_context_length);
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
    // printf("Generate token %d\n", pos);
    fflush(stdout);
    float *logits = forward_gpu_120b(context, token, pos);
    // float *logits = forward(transformer, token, pos); <---- real code from run.cpp

    // printf("logits: ");
    // for (int i = 0; i < 5; i++) {
    //   printf("%.6f ", logits[i]);
    // }
    // printf("\n");
    // exit(1);

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

      // data-dependent terminating condition: the EOS (=199999 or =200002) token
      // delimits sequences
      if (next == 199999 || next == 200002) {
        break;
      }

      // print the token as string, decode it with the Tokenizer object
      // should be removed
      const char *piece = decode_piece(tokenizer, token, next);
      safe_printf(piece);  // same as printf("%s", piece), but skips "unsafe" bytes
      fflush(stdout);

      token = next;
    }
  }

  // should be removed
  printf("\n");

  // Marker for end of sequence
  output_tokens[pos - num_prompt_tokens + 1] = -1;

  free(prompt_tokens);

  return pos - num_prompt_tokens + 1;
}

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

    // printf("logits: ");
    // for (int i = 0; i < 5; i++) {
    //   printf("%.6f ", logits[i]);
    // }
    // printf("\n");
    // exit(1);

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

      // data-dependent terminating condition: the EOS (=199999 or =200002) token
      // delimits sequences
      if (next == 199999 || next == 200002) {
        break;
      }

      // print the token as string, decode it with the Tokenizer object
      // should be removed
      const char *piece = decode_piece(tokenizer, token, next);
      safe_printf(piece);  // same as printf("%s", piece), but skips "unsafe" bytes
      fflush(stdout);

      token = next;
    }
  }

  // should be removed
  printf("\n");

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

long long batched_getp_generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                                const std::vector<const char *> &input_batch,
                                const std::vector<int *> &output_batch, int steps) {
  Config *p = &transformer->config;

  const int batch_size = input_batch.size();
  if (batch_size == 0)
    return 0;

  vector<vector<int>> batch_prompt_tokens(batch_size);
  vector<int> num_prompt_tokens(batch_size);

  for (int i = 0; i < batch_size; i++) {
    const char *input_seq = input_batch[i] ? input_batch[i] : "";
    int *prompt_tokens_buffer = (int *)malloc(strlen((input_seq) + 3) * sizeof(int));
    int count = 0;
    encode(tokenizer, input_seq, -1, -1, prompt_tokens_buffer, &count, p->initial_context_length);

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
    float *batch_logits =
      forward_gpu_20b_batched(p, weights, rs, current_tokens.data(), pos, batch_size);

#pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
      if (!active[i])
        continue;

      int next_token;
      if (current_pos[i] < num_prompt_tokens[i] - 1) {
        next_token = batch_prompt_tokens[i][current_pos[i] + 1];
      } else {
        float *logits = batch_logits + 1ll * i * p->vocab_size;
        next_token = sample(sampler, logits);
        output_batch[i][current_pos[i] - (num_prompt_tokens[i] - 1)] = next_token;
      }

      if (next_token == 199999 || next_token == 200002) {
#pragma omp critical
        {
          if (active[i]) {
            active[i] = false;
            active_count--;
          }
        }
      }

      current_tokens[i] = next_token;
      current_pos[i]++;
    }
  }

  for (int i = 0; i < batch_size; i++) {
    int generated_len = current_pos[i] - num_prompt_tokens[i];
    if (generated_len < 0)
      generated_len = 0;

    output_batch[i][generated_len] = -1;
    total_generate_tokens += generated_len;
  }

  return total_generate_tokens;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                    Requests *requests) {
  long long num_token_out = 0;

  for (int start_idx = 0; start_idx < requests->num_reqs; start_idx += BATCH_SIZE) {
    int current_size = BATCH_SIZE;
    if (start_idx + BATCH_SIZE > requests->num_reqs) {
      current_size = requests->num_reqs - BATCH_SIZE;
    }

    vector<const char *> input_batch;
    vector<int *> output_batch;

    for (int i = 0; i < current_size; i++) {
      input_batch.push_back(get_str_req_ptr(requests, start_idx + i));
      output_batch.push_back(get_tok_gen_ptr(requests, start_idx + i));
    }

    // const char *input_seq = get_str_req_ptr(requests, idx);
    // int *output_tokens = get_tok_gen_ptr(requests, idx);
    num_token_out += batched_getp_generate(transformer, tokenizer, sampler, input_batch,
                                           output_batch, requests->max_seq_len);
  }
  return num_token_out;
}
#endif

#endif  // GETP_RUN
