#pragma once

#include "tensor.hpp"
#include <vector>
#include <hip/hip_runtime.h>
#include <condition_variable>
#include <mutex>

#define SIZE_OF_BUF 2

struct PipelineEach {
    Tensor *x_buf;
    int size_buf;
    int head;
    int tail;
    int gpu_id;
    
    std::mutex mutex;
    std::condition_variable not_empty_cond;
    std::condition_variable not_full_cond;
    std::vector<hipEvent_t> slot_events;

    PipelineEach(const vector<size_t> &shape_, int gpu_id, int size_buf_, DType::Type dtype = DType::FP32);
    ~PipelineEach();

    void enqueueElem(
        Tensor *x, int sent_gpu_id, hipEvent_t x_ready, hipStream_t recv_stream
    );
    void dequeue(Tensor *x, hipStream_t stream);
};

struct WorkItem {
    int current_size;

    const char **input_batch;
    int **output_batch;
    int **batch_prompt_tokens;
    int *num_prompt_tokens;
    int *current_tokens;
    int *current_pos;
    bool *active;
    int active_count;
    long long total_generate_tokens;

    WorkItem(int current_size);
    ~WorkItem();

    void initItem(
        Requests *requests, Tokenizer *tokenizer, Config *p, int cur_idx
    );
    void nextStep(Config *p, float *batch_logits);
    void retrieveResults(long long *result_out);
};
