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
