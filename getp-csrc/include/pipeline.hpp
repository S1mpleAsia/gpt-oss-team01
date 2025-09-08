#pragma once

#include "tensor.hpp"
#include "config.hpp"

struct PipelineEach {
    Tensor *x_buf;
    int size_buf;
    int head;
    int tail;
    int gpu_id;
    // a mutex here to add/delete if necessary
    // a semaphore to control the amount of elem in the queue
    // a pipeline ready event for size_buf slots. 

    /** Initialize:
        shape of x_buf: size_buf * shape (shape of one slot is in const vector<size_t> &shape_)
        use tensor wrapper to help
    */
    PipelineEach(const vector<size_t> &shape_, int gpu_id, int size_buf_, DType::Type dtype = DType::FP32); 
    ~PipelineEach();

    /** enqueueElem
        if have space add, if not WAIT
        if not ready for the slot, WAIT
        Tensor *x: input tensor x w/ shape of each slot
        int sent_gpu_id: gpu_id that is sending tensor x from
        use hipMemcpyPeerAsync to copy Tensor *x (different gpu) to the new slot.
    */
    void enqueueElem(Tensor *x, int sent_gpu_id);
    /** 
        if have elem dequeue, if not WAIT
        if not ready for the slot, WAIT
        Tensor *x: tensor to copy from w/ shape of each slot (assure same gpu_id --> hipMemcpy
    */
    void dequeue(Tensor *x);
};