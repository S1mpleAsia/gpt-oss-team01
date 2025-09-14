#include "../include/pipeline.hpp"

PipelineEach::PipelineEach(
    const vector<size_t> &shape_, int gpu_id, int size_buf_, DType::Type dtype
) : size_buf(size_buf_), head(0), tail(0), gpu_id(gpu_id) {
    
    CHECK_HIP(hipSetDevice(gpu_id));
    
    // Create shape for the buffer: [size_buf] + shape_
    vector<size_t> buffer_shape;
    buffer_shape.push_back(size_buf);
    buffer_shape.insert(buffer_shape.end(), shape_.begin(), shape_.end());
    
    x_buf = new Tensor(buffer_shape, gpu_id, dtype);
    
    // Initialize events for each slot
    slot_events.resize(size_buf);
    for (int i = 0; i < size_buf; i++) {
        CHECK_HIP(hipEventCreate(&slot_events[i]));
    }
}

PipelineEach::~PipelineEach() {
    delete x_buf;
    for (auto &event : slot_events) {
        CHECK_HIP(hipEventDestroy(event));
    }
}

void PipelineEach::enqueueElem(
    Tensor *x, int sent_gpu_id, hipEvent_t x_ready, hipStream_t recv_stream
) {
    std::unique_lock<std::mutex> lock(mutex);
    
    // Wait if buffer is full
    not_full_cond.wait(lock, [this]() { return (tail + 1) % size_buf != head; });
    
    // Wait for the slot to be ready
    CHECK_HIP(hipSetDevice(gpu_id));
    CHECK_HIP(hipStreamWaitEvent(recv_stream, x_ready));
    CHECK_HIP(hipStreamWaitEvent(recv_stream, slot_events[tail]));
    
    // Copy data using hipMemcpyPeerAsync
    size_t slot_size = x_buf->num_elem() / size_buf;
    size_t offset = tail * slot_size * x_buf->get_dtype_size();
    
    CHECK_HIP(hipMemcpyPeerAsync(
        (char*)x_buf->d_buf + offset, gpu_id,
        x->d_buf, sent_gpu_id,
        slot_size * x_buf->get_dtype_size(),
        recv_stream
    ));
    
    // Record event for this slot
    CHECK_HIP(hipEventRecord(slot_events[tail], recv_stream));
    
    tail = (tail + 1) % size_buf;
    not_empty_cond.notify_one();
}

void PipelineEach::dequeue(Tensor *x, hipStream_t stream) {
    std::unique_lock<std::mutex> lock(mutex);
    
    // Wait if buffer is empty
    not_empty_cond.wait(lock, [this]() { return head != tail; });
    
    // Wait for the slot to be ready
    CHECK_HIP(hipStreamWaitEvent(stream, slot_events[head]));
    
    // Copy data to output tensor
    size_t slot_size = x_buf->num_elem() / size_buf;
    size_t offset = head * slot_size * x_buf->get_dtype_size();
    
    CHECK_HIP(hipMemcpyAsync(
        x->d_buf,
        (char*)x_buf->d_buf + offset,
        slot_size * x_buf->get_dtype_size(),
        hipMemcpyDeviceToDevice,
        stream
    ));
    
    // Record event for this slot
    CHECK_HIP(hipEventRecord(slot_events[head], stream));
    
    head = (head + 1) % size_buf;
    not_full_cond.notify_one();
}

// WORK ITEM

WorkItem::WorkItem(int current_size) : current_size(current_size), active_count(current_size), total_generate_tokens(0) {
    // Allocate memory for the arrays
    input_batch = new const char *[current_size];
    output_batch = new int *[current_size];
    num_prompt_tokens = new int[current_size];
    current_tokens = new int[current_size];
    current_pos = new int[current_size](); // Initializes all elements to 0
    active = new bool[current_size];
    batch_prompt_tokens = new int*[current_size];
}

// Don't forget to add a destructor to free the allocated memory!
WorkItem::~WorkItem() {
    delete[] input_batch;
    delete[] output_batch;
    delete[] num_prompt_tokens;
    delete[] current_tokens;
    delete[] current_pos;
    delete[] active;
    
    for (int i = 0; i < current_size; i++) {
        delete[] batch_prompt_tokens[i];
    }
    delete[] batch_prompt_tokens;
}

void WorkItem::initItem(Requests *requests, Tokenizer *tokenizer, Config *p, int cur_idx) {
    // Initialize input_batch and output_batch
    for (int j = 0; j < current_size; j++) {
        input_batch[j] = get_str_req_ptr(requests, cur_idx + j);
        output_batch[j] = get_tok_gen_ptr(requests, cur_idx + j);
    }

    // Process prompts and populate batch_prompt_tokens
    for (int i = 0; i < current_size; i++) {
        const char *input_seq = input_batch[i] ? input_batch[i] : "";
        
        // This is a rough estimation for buffer size. A more robust solution is recommended.
        int prompt_tokens_buffer_size = strlen(input_seq) + 3;
        int *prompt_tokens_buffer = (int *)malloc(prompt_tokens_buffer_size * sizeof(int));
        int count = 0;
        
        encode(tokenizer, input_seq, -1, -1, prompt_tokens_buffer, &count, p->initial_context_length);

        if (count < 1) {
            fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
            exit(EXIT_FAILURE);
        } else {
            // Allocate memory for this specific batch item's tokens
            batch_prompt_tokens[i] = new int[count];
            // Copy the tokens from the buffer to the new array
            memcpy(batch_prompt_tokens[i], prompt_tokens_buffer, count * sizeof(int));
            num_prompt_tokens[i] = count;
        }

        free(prompt_tokens_buffer);
    }

    // Initialize current_tokens
    for (int i = 0; i < current_size; i++) {
        current_tokens[i] = batch_prompt_tokens[i][0];
    }
    
    // Set all items to active
    for (int i = 0; i < current_size; i++) {
        active[i] = true;
    }
}

void WorkItem::nextStep(Config *p, float *batch_logits) {
    for (int i = 0; i < current_size; i++) {
        if (!active[i]) continue;
        
        float *logits = batch_logits + 1ll * i * p->vocab_size;

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

void WorkItem::retrieveResults(long long *result_out) {
    for (int i = 0; i < current_size; i++) {
        int generated_len = current_pos[i] - num_prompt_tokens[i];
        if (generated_len < 0) generated_len = 0;

        output_batch[i][generated_len + 1] = -1;
        total_generate_tokens += generated_len;
    }

    *(result_out) += total_generate_tokens; 
}
