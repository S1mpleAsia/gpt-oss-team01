// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"

#include "src/tensor.cpp"
#include "src/layer.cpp"
#include "src/layer_hip.cpp"
#include "src/layer_hip_batch.cpp"
#include "src/model.cpp"
#include "src/alloc.cpp"
#include "src/utils.cpp"

#ifndef GETP_RUN
#define GETP_RUN

OurTransformerWeights *weights;
OurRunState *rs;

Config *public_config;
Transformer *public_transformer;
Tokenizer *public_tokenizer;
Sampler *public_sampler;
Requests *public_requests;
StreamTotal *total_streams;

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the warm-up process
  // TODO:
  // - Memory allocation
  // - Load model
  // - ...
  int actualGPUs = 0;
  CHECK_HIP(hipGetDeviceCount(&actualGPUs));

  if (TOTAL_GPUS_NEEDED != actualGPUs) {
    fprintf(stderr, "Error: Required %d GPUs, but only %d available.\n", TOTAL_GPUS_NEEDED, actualGPUs);
    exit(1);
  }

  weights = new OurTransformerWeights[TOTAL_GPUS_NEEDED];
  rs = new OurRunState[TOTAL_GPUS_NEEDED];
  total_streams = new StreamTotal[TOTAL_GPUS_NEEDED];
  our_init(transformer, weights, rs);

  public_config = &transformer->config;
  public_transformer = transformer;
  public_tokenizer = tokenizer;
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
  our_free(weights, rs);
  delete weights;
  delete rs;
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
  long long num_token_out = 0;
  for (int idx = 0; idx < requests->num_reqs; ++idx) {
    const char *input_seq = get_str_req_ptr(requests, idx);
    int *output_tokens = get_tok_gen_ptr(requests, idx);
    num_token_out += simple_getp_generate(transformer, tokenizer, sampler, input_seq, output_tokens,
                                          requests->max_seq_len);
  }
  return num_token_out;
}
#else

long long batched_getp_generate(
  const std::vector<const char *> &input_batch,
  const std::vector<int *> &output_batch, int steps, int flow_id
) {
  const int batch_size = input_batch.size();
  if (batch_size == 0)
    return 0;

  vector<vector<int>> batch_prompt_tokens(batch_size);
  vector<int> num_prompt_tokens(batch_size);

  for (int i = 0; i < batch_size; i++) {
    const char *input_seq = input_batch[i] ? input_batch[i] : "";
    int *prompt_tokens_buffer = (int *)malloc(strlen((input_seq) + 3) * sizeof(int));
    int count = 0;
    encode(public_tokenizer, input_seq, -1, -1, prompt_tokens_buffer, &count, public_config->initial_context_length);

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
      forward_gpu_120b_batched(public_config, weights, rs, current_tokens.data(), pos, batch_size, flow_id);

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
        const char *piece = decode_piece(tokenizer, current_tokens[i], next_token);
        
        // Use a critical section for printing to prevent interleaved output
        #pragma omp critical
        {
            printf("batch id %d --> ", i);
            safe_printf(piece);
            printf("logits: ");
            for (int j = 0; j < 5; j++) {
                printf("%.6f ", logits[j]);
            }
            printf("\n");
            fflush(stdout);
        }
      #endif

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

long long inference_old(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                    Requests *requests) {
  setup(sampler, requests);

  long long num_token_out = 0;

  for (int start_idx = 0; start_idx < requests->num_reqs; start_idx += BATCH_SIZE) {
    int current_size = BATCH_SIZE;
    if (start_idx + BATCH_SIZE > requests->num_reqs) {
      current_size = requests->num_reqs - start_idx;
    }

    vector<const char *> input_batch;
    vector<int *> output_batch;

    for (int i = 0; i < current_size; i++) {
      input_batch.push_back(get_str_req_ptr(requests, i + start_idx));
      output_batch.push_back(get_tok_gen_ptr(requests, i + start_idx));
    }

    // const char *input_seq = get_str_req_ptr(requests, idx);
    // int *output_tokens = get_tok_gen_ptr(requests, idx);
    num_token_out += batched_getp_generate(
      input_batch, output_batch, requests->max_seq_len, 0
    );
  }
  return num_token_out;
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

    local_token_count += batched_getp_generate(
      input_batch, output_batch, max_seq_len, id
    );
  }

  *(args->local_token_ptr) = local_token_count;

  return nullptr;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                    Requests *requests) {
  setup(sampler, requests);

  long long num_token_out = 0;
  pthread_t threads[DP];
  ThreadArgs args[DP];
  long long tokens_out[DP];

  int total_reqs = public_requests->num_reqs;
  int chunk_size = (total_reqs + DP - 1) / DP;
    
  for (int i = 0; i < DP; i++) {
    args[i].id = i;
    args[i].local_token_ptr = &tokens_out[i];
    args[i].start_idx = i * chunk_size;
    args[i].end_idx = min((i+1) * chunk_size, total_reqs);

    pthread_create(&threads[i], NULL, thread_handler, (void *)&args[i]);
  }

  for (int i = 0; i < DP; i++) {
    pthread_join(threads[i], NULL);
    num_token_out += tokens_out[i];
  }

  printf("Total tokens generated: %lld\n", num_token_out);
  fflush(stdout);

  return num_token_out;
}


#endif

#endif  // GETP_RUN
