#include "../include/config.hpp"
#include "../include/layer_hip_batch.hpp"

void *thread_handler_20b(void *arg) {
  OnePathArgs *args = (OnePathArgs *)arg;

  long long local_token_count = 0ll;
  int id = args->id;
  int max_seq_len = public_requests->max_seq_len;
  int end_idx = args->end_idx;
  args->tp_barrier = new pthread_barrier_t[PP_20B * 5];

  CHECK_HIP(hipSetDevice(id));

  for (int cur_idx = args->start_idx; cur_idx < end_idx; cur_idx += BATCH_SIZE) {
    int current_size = min(BATCH_SIZE, end_idx - cur_idx);

    if (current_size == 0)
      continue;

    vector<const char *> input_batch;
    vector<int *> output_batch;

    for (int j = 0; j < current_size; j++) {
      input_batch.push_back(get_str_req_ptr(public_requests, cur_idx + j));
      output_batch.push_back(get_tok_gen_ptr(public_requests, cur_idx + j));
    }

    vector<vector<int>> batch_prompt_tokens(current_size);
    vector<int> num_prompt_tokens(current_size);

    for (int i = 0; i < current_size; i++) {
      const char *input_seq = input_batch[i] ? input_batch[i] : "";
      int *prompt_tokens_buffer = (int *)malloc((strlen(input_seq) + 3) * sizeof(int));
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

    vector<int> current_tokens(current_size);
    vector<int> current_pos(current_size, 0);
    vector<bool> active(current_size, true);
    int active_count = current_size;
    long long total_generate_tokens = 0;

    for (int i = 0; i < current_size; i++) {
      current_tokens[i] = batch_prompt_tokens[i][0];
    }

    for (int pos = 0; pos < max_seq_len - 1 && active_count > 0; ++pos) {
      int *batch_tokens = forward_gpu_20b_batched(current_tokens.data(), pos, current_size, id);

      for (int i = 0; i < current_size; i++) {
        if (!active[i])
          continue;

        int next_token = batch_tokens[i];
        if (current_pos[i] < num_prompt_tokens[i] - 1) {
          next_token = batch_prompt_tokens[i][current_pos[i] + 1];
        } else {
          output_batch[i][current_pos[i] - (num_prompt_tokens[i] - 1)] = next_token;
        }

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

  delete[] args->tp_barrier;

  *(args->local_token_ptr) = local_token_count;

  return nullptr;
}

void *inside_thread_handler_120b(void *arg) {
  OnePathInsideArgs *args = (OnePathInsideArgs *)arg;
  int cur_device = args->id * TOTAL_PIPELINES_120B + args->pp_rank * TP_120B + args->tp_rank;
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

void *thread_handler_120b(void *arg) {
  OnePathArgs *args = (OnePathArgs *)arg;

  long long local_token_count = 0ll;
  int id = args->id;
  int max_seq_len = public_requests->max_seq_len;
  int end_idx = args->end_idx;
  args->tp_barrier = new pthread_barrier_t[PP_120B * 5];

  pthread_t inside_threads[TOTAL_PIPELINES_120B];
  OnePathInsideArgs inside_args[TOTAL_PIPELINES_120B];

  pthread_barrier_t start_barrier, end_barrier;
  pthread_barrier_init(&start_barrier, NULL, TOTAL_PIPELINES_120B + 1);
  pthread_barrier_init(&end_barrier, NULL, TOTAL_PIPELINES_120B + 1);

  // barrier for different pipeline stages
  for (int i = 0; i < PP; i++) {
    int base_idx = i * 5;
    pthread_barrier_init(&(args->tp_barrier[base_idx]), NULL, TP_120B);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 1]), NULL, TP_120B / 2);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 2]), NULL, TP_120B / 2);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 3]), NULL, TP_120B / 2);
    pthread_barrier_init(&(args->tp_barrier[base_idx + 4]), NULL, TP_120B / 2);
  }

  int shared_pos = 0;
  bool finished = false;

  for (int i = 0; i < TOTAL_PIPELINES_120B; i++) {
    inside_args[i].tp_rank = (i % TP_120B);
    inside_args[i].pp_rank = (i % TOTAL_PIPELINES_120B) / TP_120B;
    inside_args[i].tp_barrier = &(args->tp_barrier[inside_args[i].pp_rank * 5]);
    inside_args[i].id = id;
    inside_args[i].pos_ptr = &shared_pos;
    inside_args[i].finished_ptr = &finished;
    inside_args[i].start_barrier = &start_barrier;
    inside_args[i].end_barrier = &end_barrier;
  }

  for (int i = 0; i < TOTAL_PIPELINES_120B; i++) {
    pthread_create(&inside_threads[i], NULL, inside_thread_handler_120b, (void *)&inside_args[i]);
  }

  for (int cur_idx = args->start_idx; cur_idx < end_idx; cur_idx += BATCH_SIZE) {
    int current_size = min(BATCH_SIZE, end_idx - cur_idx);

    if (current_size == 0)
      continue;

    vector<const char *> input_batch;
    vector<int *> output_batch;

    for (int j = 0; j < current_size; j++) {
      input_batch.push_back(get_str_req_ptr(public_requests, cur_idx + j));
      output_batch.push_back(get_tok_gen_ptr(public_requests, cur_idx + j));
    }

    vector<vector<int>> batch_prompt_tokens(current_size);
    vector<int> num_prompt_tokens(current_size);

    for (int i = 0; i < current_size; i++) {
      const char *input_seq = input_batch[i] ? input_batch[i] : "";
      int *prompt_tokens_buffer = (int *)malloc((strlen(input_seq) + 3) * sizeof(int));
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

    vector<int> current_tokens(current_size);
    vector<int> current_pos(current_size, 0);
    vector<bool> active(current_size, true);
    int active_count = current_size;
    long long total_generate_tokens = 0;

    for (int i = 0; i < current_size; i++) {
      current_tokens[i] = batch_prompt_tokens[i][0];
    }

    for (int i = 0; i < TOTAL_PIPELINES_120B; i++) {
      inside_args[i].current_tokens = &current_tokens;
      inside_args[i].current_size = current_size;
    }

    for (int pos = 0; pos < max_seq_len - 1 && active_count > 0; ++pos) {
      shared_pos = pos;
      pthread_barrier_wait(&start_barrier);

      pthread_barrier_wait(&end_barrier);

      int *batch_tokens = inside_args[(PP - 1) * TP].tokens;

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

  finished = true;
  pthread_barrier_wait(&start_barrier);

  for (int i = 0; i < TOTAL_PIPELINES; i++) {
    pthread_join(inside_threads[i], NULL);
  }

  pthread_barrier_destroy(&start_barrier);
  pthread_barrier_destroy(&end_barrier);

  for (int i = 0; i < PP * 5; i++) {
    pthread_barrier_destroy(&(args->tp_barrier[i]));
  }

  delete[] args->tp_barrier;

  *(args->local_token_ptr) = local_token_count;

  return nullptr;
}
