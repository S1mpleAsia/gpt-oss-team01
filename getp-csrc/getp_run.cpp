// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"

#include "src/tensor.cpp"
#include "src/layer_hip_batch.cpp"
#include "src/model.cpp"
#include "src/alloc.cpp"
#include "src/alloc_utils.cpp"
#include "src/utils.cpp"
#include "src/pipeline.cpp"
#include "src/parallelism.cpp"
#include "src/kernel.cpp"
#include "src/flash_attn_hip.cpp"

#ifndef GETP_RUN
#define GETP_RUN

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
    fprintf(stderr, "Error: Required %d GPUs, but only %d available.\n", TOTAL_GPUS_NEEDED,
            actualGPUs);
    exit(1);
  }

#ifdef RUN_20B
  if (TP != 1 || PP != 1) {
    fprintf(stderr, "Error: TP and PP must be equal to 1 in 20B mode\n");
    exit(1);
  }
#endif

  for (int i = 0; i < TOTAL_GPUS_NEEDED; i++) {
    for (int j = 0; j < TOTAL_GPUS_NEEDED; j++) {
      CHECK_HIP(hipSetDevice(i));
      if (i != j) {
        CHECK_HIP(hipDeviceEnablePeerAccess(j, 0));
      }
    }
  }

  weights = new OurTransformerWeights[TOTAL_GPUS_NEEDED];
  rs = new OurRunState[TOTAL_GPUS_NEEDED];
  total_streams = new hipStream_t[TOTAL_GPUS_NEEDED];
  total_events = new hipTotalEvents_t;

  our_init(transformer, weights, rs, total_streams, total_events);

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
  our_free(weights, rs, total_streams, total_events);
  delete weights;
  delete rs;
  delete total_streams;
  delete total_events;
}

#ifndef RUN_20B
void *inside_thread_handler(void *arg) {
  OnePathInsideArgs *args = (OnePathInsideArgs *)arg;
  int cur_device = args->id * TOTAL_PIPELINES + args->pp_rank * TP + args->tp_rank;
  CHECK_HIP(hipSetDevice(cur_device));

  pthread_barrier_wait(args->tp_barrier);

  while (true) {
    pthread_barrier_wait(args->start_barrier);
    if (*(args->finished_ptr))
      break;

    args->tokens =
      forward_gpu_120b_batched(args->current_tokens->data(), *(args->pos_ptr), args->current_size,
                               args->id, args->tp_rank, args->pp_rank, args->tp_barrier);

    pthread_barrier_wait(args->end_barrier);
  }

  return nullptr;
}
#endif

void *thread_handler(void *arg) {
  OnePathArgs *args = (OnePathArgs *)arg;

  long long local_token_count = 0ll;
  int id = args->id;
  int max_seq_len = public_requests->max_seq_len;
  int end_idx = args->end_idx;
  args->tp_barrier = new pthread_barrier_t[PP * 13];

#ifdef RUN_20B
  CHECK_HIP(hipSetDevice(id));
#else
  pthread_t inside_threads[TOTAL_PIPELINES];
  OnePathInsideArgs inside_args[TOTAL_PIPELINES];

  pthread_barrier_t start_barrier, end_barrier;
  pthread_barrier_init(&start_barrier, NULL, TOTAL_PIPELINES + 1);
  pthread_barrier_init(&end_barrier, NULL, TOTAL_PIPELINES + 1);

  // barrier for different pipeline stages
  for (int i = 0; i < PP; i++) {
    int base_idx = i * 13;
    pthread_barrier_init(&(args->tp_barrier[base_idx]), NULL, TP);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 1]), NULL, 2);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 2]), NULL, 2);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 3]), NULL, 2);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 4]), NULL, 2);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 5]), NULL, 2);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 6]), NULL, 2);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 7]), NULL, 2);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 8]), NULL, 2);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 9]), NULL, 2);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 10]), NULL, 2);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 11]), NULL, 2);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 12]), NULL, 2);
  }

  int shared_pos = 0;
  bool finished = false;

  for (int i = 0; i < TOTAL_PIPELINES; i++) {
    inside_args[i].tp_rank = (i % TP);
    inside_args[i].pp_rank = (i % TOTAL_PIPELINES) / TP;
    inside_args[i].tp_barrier = &(args->tp_barrier[inside_args[i].pp_rank * 13]);
    inside_args[i].id = id;
    inside_args[i].pos_ptr = &shared_pos;
    inside_args[i].finished_ptr = &finished;
    inside_args[i].start_barrier = &start_barrier;
    inside_args[i].end_barrier = &end_barrier;
  }

  for (int i = 0; i < TOTAL_PIPELINES; i++) {
    pthread_create(&inside_threads[i], NULL, inside_thread_handler, (void *)&inside_args[i]);
  }
#endif

  vector<const char *> input_batch(BATCH_SIZE);
  vector<int *> output_batch(BATCH_SIZE);

  vector<vector<int>> batch_prompt_tokens(BATCH_SIZE);
  vector<int> num_prompt_tokens(BATCH_SIZE);

  std::vector<int> current_tokens(BATCH_SIZE);
  std::vector<int> current_pos(BATCH_SIZE, 0);
  std::vector<bool> active(BATCH_SIZE, true);

  static thread_local std::vector<int> encode_buf;

  for (int cur_idx = args->start_idx; cur_idx < end_idx; cur_idx += BATCH_SIZE) {
    int current_size = min(BATCH_SIZE, end_idx - cur_idx);

    if (current_size == 0)
      continue;

    input_batch.clear();
    output_batch.clear();

    for (int j = 0; j < current_size; j++) {
      input_batch.push_back(get_str_req_ptr(public_requests, cur_idx + j));
      output_batch.push_back(get_tok_gen_ptr(public_requests, cur_idx + j));
    }

    for (int i = 0; i < current_size; i++) {
      const char *input_seq = input_batch[i] ? input_batch[i] : "";
      size_t need = strlen(input_seq) + 3;  // biên an toàn cho encode()
      if (encode_buf.size() < need)
        encode_buf.resize(need);

      int count = 0;
      encode(public_tokenizer, input_seq, -1, -1, encode_buf.data(), &count,
             public_config->initial_context_length);

      if (count < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
      }
      // Ghi lại vào slot i (không tạo vector mới)
      auto &dst = batch_prompt_tokens[i];
      dst.assign(encode_buf.data(), encode_buf.data() + count);
      num_prompt_tokens[i] = count;
    }

    for (int i = 0; i < current_size; ++i) {
      current_tokens[i] = batch_prompt_tokens[i][0];
      current_pos[i] = 0;
      active[i] = true;
    }
    int active_count = current_size;
    long long total_generate_tokens = 0;

#ifndef RUN_20B
    for (int i = 0; i < TOTAL_PIPELINES; i++) {
      inside_args[i].current_tokens = &current_tokens;
      inside_args[i].current_size = current_size;
    }
#endif

    for (int pos = 0; pos < max_seq_len - 1 && active_count > 0; ++pos) {
#ifdef RUN_20B
      int *batch_tokens = forward_gpu_20b_batched(current_tokens.data(), pos, current_size, id);
#else
      shared_pos = pos;
      pthread_barrier_wait(&start_barrier);

      pthread_barrier_wait(&end_barrier);

      int *batch_tokens = inside_args[(PP - 1) * TP].tokens;
#endif

      for (int i = 0; i < current_size; i++) {
        if (!active[i])
          continue;

        int next_token = batch_tokens[i];
        if (current_pos[i] < num_prompt_tokens[i] - 1) {
          next_token = batch_prompt_tokens[i][current_pos[i] + 1];
        } else {
          output_batch[i][current_pos[i] - (num_prompt_tokens[i] - 1)] = next_token;
        }

// Print the logits in the desired format if the flag is enabled
#ifdef PRINT_LOGITS
        // Decode the next token to get its string representation
        const char *piece = decode_piece(public_tokenizer, current_tokens[i], next_token);

        // Use a critical section for printing to prevent interleaved output
        // #pragma omp critical
        {
          printf("batch id %d --> next_token %d: ", i, next_token);
          safe_printf(piece);
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

    for (int i = 0; i < current_size; i++) {
      int generated_len = current_pos[i] - num_prompt_tokens[i];
      if (generated_len < 0)
        generated_len = 0;

      output_batch[i][generated_len + 1] = -1;
      local_token_count += (generated_len + 1);
    }
  }

#ifndef RUN_20B

  finished = true;
  pthread_barrier_wait(&start_barrier);

  for (int i = 0; i < TOTAL_PIPELINES; i++) {
    pthread_join(inside_threads[i], NULL);
  }

  pthread_barrier_destroy(&start_barrier);
  pthread_barrier_destroy(&end_barrier);

  for (int i = 0; i < PP * 13; i++) {
    pthread_barrier_destroy(&(args->tp_barrier[i]));
  }
#endif

  delete[] args->tp_barrier;

  *(args->local_token_ptr) = local_token_count;

  return nullptr;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                    Requests *requests) {
  setup(sampler, requests);

  long long num_token_out = 0;
  pthread_t threads[DP];
  OnePathArgs args[DP];
  long long tokens_out[DP];

  int total_reqs = public_requests->num_reqs;
  int chunk_size = (total_reqs + DP - 1) / DP;

  for (int i = 0; i < DP; i++) {
    args[i].id = i;
    args[i].local_token_ptr = &tokens_out[i];
    args[i].start_idx = i * chunk_size;
    args[i].end_idx = min((i + 1) * chunk_size, total_reqs);

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

#endif  // GETP_RUN
