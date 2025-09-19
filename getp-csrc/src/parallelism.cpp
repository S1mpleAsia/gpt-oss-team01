#include "../include/parallelism.hpp"

void all_gather_x(
  OurRunState *rs_now, OurRunState *rs_leader, int tp_rank,
  int cur_device, int cur_batch_size, pthread_barrier_t *tp_barrier,
  hipStream_t stream, hipEvent_t tp_ready, hipEvent_t tp_finish
) {
  float *d_embed_buf[TP];
  float *d_buf_ptr = (float *)rs_now->x->d_buf;
  int leader_device = cur_device - tp_rank;

  size_t offset_buf = 1ll * rs_now->x->shape[1];
  size_t offset_d_buf = 1ll * rs_now->x_embed_buf->shape[1];
  size_t elems = 1ll * offset_d_buf * sizeof(float);

  for (int i = 0; i < TP; i++) {
    d_embed_buf[i] = (float *)(rs_leader + i)->x_embed_buf->d_buf;
  }

  CHECK_HIP(hipEventRecord(tp_ready));

  pthread_barrier_wait(tp_barrier);

  for (int i = 0; i < TP; i++) {
    hipEvent_t tp_ready_each = total_events->tp_ready[leader_device + i];
    CHECK_HIP(hipStreamWaitEvent(stream, tp_ready_each));
  }

  for (int b = 0; b < cur_batch_size; b++) {
    for (int i = 0; i < TP; i++) {
      CHECK_HIP(hipMemcpyPeerAsync(
        d_buf_ptr, cur_device, d_embed_buf[i], leader_device + i, elems, stream
      ));
      d_embed_buf[i] += offset_d_buf;
      d_buf_ptr += offset_d_buf;
    }
  }
}

void reduce_tb3(
    OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
    int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
    hipEvent_t tp_ready, hipEvent_t tp_finish
) {
  size_t active_elems = (size_t)cur_batch_size * rs_now->tb3->shape[1] * rs_now->tb3->shape[2];
  size_t active_num_bytes = active_elems * rs_now->tb3->get_dtype_size();

  size_t num_elems = rs_now->tb3->num_elem();
  size_t num_bytes = num_elems * rs_now->tb3->get_dtype_size();

  if (tp_rank > 0) {
    // copy all tp_rank to buffer
    // rs of leader
    void *dst_buf = (void *)((float *)rs_leader->tb3_buf->d_buf + (tp_rank - 1) * num_elems);
    const void *src_buf = rs_now->tb3->d_buf;

    CHECK_HIP(hipMemcpyPeerAsync(dst_buf, cur_device - tp_rank, src_buf, cur_device,
                                 active_num_bytes, stream));
    CHECK_HIP(hipEventRecord(tp_ready, stream));
  }

#ifdef DEBUG
  bool flag = true;
  if (flag)
    rs_now->tb3->printDebug("rs_now->tb3 after copy to leader", tp_rank, 0, stream);
#endif

  pthread_barrier_wait(tp_barrier);

  if (tp_rank == 0) {
    // aggregates here
    hipEvent_t tp_ready_each;
    float *d_buf = (float *)rs_now->tb3->d_buf;
    float *d_buf_each = (float *)rs_leader->tb3_buf->d_buf;

    const int block_size = 256;
    const int grid_size = (active_elems + block_size - 1) / block_size;

    for (int i = 1; i < TP; i++) {
      tp_ready_each = total_events->tp_ready[cur_device + i];
      CHECK_HIP(hipStreamWaitEvent(stream, tp_ready_each));
      // add vector kernel here, do later
      // do on stream

      add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(
        d_buf, (const float *)d_buf_each, active_elems);

      d_buf_each += num_elems;
    }
    CHECK_HIP(hipEventRecord(tp_ready, stream));
  }

  pthread_barrier_wait(tp_barrier);

#ifdef DEBUG
  if (flag)
    rs_now->tb3->printDebug("rs_now->tb3 after aggregate", tp_rank, 0, stream);
#endif

  if (tp_rank > 0) {
    // copy back
    hipEvent_t leader_tp_ready = total_events->tp_ready[cur_device - tp_rank];

    // rs of leader TP
    void *dst_buf = rs_now->tb3->d_buf;
    const void *src_buf = rs_leader->tb3->d_buf;

    CHECK_HIP(hipStreamWaitEvent(stream, leader_tp_ready));
    CHECK_HIP(hipMemcpyPeerAsync(dst_buf, cur_device, src_buf, cur_device - tp_rank,
                                 active_num_bytes, stream));
    CHECK_HIP(hipEventRecord(tp_finish, stream));
  } else {
    hipEvent_t tp_finish_each;
    for (int i = 1; i < TP; i++) {
      tp_finish_each = total_events->tp_finish[cur_device + i];
      CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_each));
    }
  }

  pthread_barrier_wait(tp_barrier);
}

void reduce_tb2(
    OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
    int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
    hipEvent_t tp_ready, hipEvent_t tp_finish
) {
  size_t active_elems = (size_t)cur_batch_size * rs_now->tb2->shape[1];
  size_t active_num_bytes = active_elems * rs_now->tb2->get_dtype_size();
  size_t num_elems = rs_now->tb2->num_elem();
  size_t num_bytes = num_elems * rs_now->tb2->get_dtype_size();

  if (tp_rank > 0) {
    void *dst_buf = (void *)((float *)rs_leader->tb2_buf->d_buf + (tp_rank - 1) * num_elems);
    const void *src_buf = rs_now->tb2->d_buf;

    CHECK_HIP(hipMemcpyPeerAsync(dst_buf, cur_device - tp_rank, src_buf, cur_device,
                                 active_num_bytes, stream));
    CHECK_HIP(hipEventRecord(tp_ready, stream));
  }

  pthread_barrier_wait(tp_barrier);

  if (tp_rank == 0) {
    hipEvent_t tp_ready_each;
    float *d_buf = (float *)rs_now->tb2->d_buf;
    float *d_buf_each = (float *)rs_leader->tb2_buf->d_buf;

    const int block_size = 256;
    const int grid_size = (active_elems + block_size - 1) / block_size;

    for (int i = 1; i < TP; i++) {
      tp_ready_each = total_events->tp_ready[cur_device + i];
      CHECK_HIP(hipStreamWaitEvent(stream, tp_ready_each));

      add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(
        d_buf, (const float *)d_buf_each, active_elems);

      d_buf_each += num_elems;
    }
    CHECK_HIP(hipEventRecord(tp_ready, stream));
  }

  pthread_barrier_wait(tp_barrier);

  if (tp_rank > 0) {
    hipEvent_t leader_tp_ready = total_events->tp_ready[cur_device - tp_rank];
    void *dst_buf = rs_now->tb2->d_buf;

    const void *src_buf = rs_leader->tb2->d_buf;

    CHECK_HIP(hipStreamWaitEvent(stream, leader_tp_ready));
    CHECK_HIP(hipMemcpyPeerAsync(dst_buf, cur_device, src_buf, cur_device - tp_rank,
                                 active_num_bytes, stream));
    CHECK_HIP(hipEventRecord(tp_finish, stream));
  } else {
    hipEvent_t tp_finish_each;
    for (int i = 1; i < TP; i++) {
      tp_finish_each = total_events->tp_finish[cur_device + i];
      CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_each));
    }
  }

  pthread_barrier_wait(tp_barrier);
}

void all_gather_classifier_v2(
    OurRunState *rs_now, OurRunState *rs_leader, int tp_rank,
    int cur_device, int cur_batch_size, pthread_barrier_t *tp_barrier,
    hipStream_t stream, hipEvent_t tp_ready, hipEvent_t tp_finish
) {
  size_t shard_vocab_size = rs_now->tmp_logits->shape[1];
  size_t vocab_size = shard_vocab_size * TP;

  size_t width_bytes = shard_vocab_size * sizeof(float);
  size_t height = cur_batch_size;

  const void *src_buf = rs_now->tmp_logits->d_buf;
  size_t src_pitch = width_bytes;

  float *dst_base_buf = (float *)rs_leader->logits->d_buf;
  size_t dst_pitch = vocab_size * sizeof(float);

  void *dst_buf = (void *)(dst_base_buf + tp_rank * shard_vocab_size);

  CHECK_HIP(hipMemcpy2DAsync(dst_buf, dst_pitch, src_buf, src_pitch, width_bytes, height,
                             hipMemcpyDeviceToDevice, stream));

  if (tp_rank > 0) {
    CHECK_HIP(hipEventRecord(tp_ready, stream));
  }

  pthread_barrier_wait(tp_barrier);

  if (tp_rank == 0) {
    for (int i = 1; i < TP; i++) {
      hipEvent_t worker_event = total_events->tp_ready[cur_device + i];
      CHECK_HIP(hipStreamWaitEvent(stream, worker_event, 0));
    }
  }

  pthread_barrier_wait(tp_barrier);
}

void all_gather_classifier_final(
    OurRunState *rs_now, OurRunState *rs_leader, int tp_rank,
    int cur_device, int cur_batch_size, pthread_barrier_t *tp_barrier,
    hipStream_t stream, hipEvent_t tp_ready, hipEvent_t tp_finish
) {
  size_t shard_vocab_size = rs_now->tmp_logits->shape[1];
  size_t vocab_size = shard_vocab_size * TP;

  size_t width_bytes = shard_vocab_size * sizeof(float);
  size_t height = cur_batch_size;

  const void *src_buf = rs_now->tmp_logits->d_buf;
  size_t src_pitch = width_bytes;

  float *dst_base_buf = (float *)rs_leader->logits_out;
  size_t dst_pitch = vocab_size * sizeof(float);

  void *dst_buf = (void *)(dst_base_buf + tp_rank * shard_vocab_size);

  CHECK_HIP(hipMemcpy2DAsync(dst_buf, dst_pitch, src_buf, src_pitch, width_bytes, height,
                             hipMemcpyDeviceToHost, stream));
  
  CHECK_HIP(hipStreamSynchronize(stream));
                             
  pthread_barrier_wait(tp_barrier);
}

void all_gather_qkv_v2(
    OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
    int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
    hipEvent_t tp_ready, hipEvent_t tp_finish
) {
  size_t shard_qkv_size = rs_now->tmp_qkv->shape[1];
  size_t full_qkv_size = shard_qkv_size * TP;

  size_t width_bytes = shard_qkv_size * sizeof(float);
  size_t height = cur_batch_size;

  const void *src_buf = rs_now->tmp_qkv->d_buf;
  size_t src_pitch = width_bytes;

  float *dst_base_buf = (float *)rs_leader->qkv->d_buf;
  size_t dst_pitch = full_qkv_size * sizeof(float);

  void *dst_buf = (void *)(dst_base_buf + tp_rank * shard_qkv_size);

  CHECK_HIP(hipMemcpy2DAsync(dst_buf, dst_pitch, src_buf, src_pitch, width_bytes, height,
                             hipMemcpyDeviceToDevice, stream));

  if (tp_rank > 0) {
    CHECK_HIP(hipEventRecord(tp_ready, stream));
  }

  pthread_barrier_wait(tp_barrier);

  if (tp_rank == 0) {
    for (int i = 1; i < TP; i++) {
      hipEvent_t worker_event = total_events->tp_ready[cur_device + i];
      CHECK_HIP(hipStreamWaitEvent(stream, worker_event, 0));
    }

    CHECK_HIP(hipEventRecord(tp_ready, stream));
  }

  pthread_barrier_wait(tp_barrier);

  if (tp_rank > 0) {
    hipEvent_t leader_tp_ready = total_events->tp_ready[cur_device - tp_rank];
    CHECK_HIP(hipStreamWaitEvent(stream, leader_tp_ready));

    size_t qkv_bytes = rs_now->qkv->num_elem() * rs_now->qkv->get_dtype_size();
    void *dst_buf = rs_now->qkv->d_buf;
    const void *src_buf = rs_leader->qkv->d_buf;

    CHECK_HIP(
      hipMemcpyPeerAsync(dst_buf, cur_device, src_buf, cur_device - tp_rank, qkv_bytes, stream));
    CHECK_HIP(hipEventRecord(tp_finish, stream));
  } else {
    // Leader waits for all workers to finish broadcast
    for (int i = 1; i < TP; i++) {
      hipEvent_t worker_finish = total_events->tp_finish[cur_device + i];
      CHECK_HIP(hipStreamWaitEvent(stream, worker_finish, 0));
    }
  }

  pthread_barrier_wait(tp_barrier);
}

void all_gather_tb3(
    OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
    Config *p, pthread_barrier_t *tp_barrier, hipStream_t stream,
    hipEvent_t tp_ready, hipEvent_t tp_finish
) {
  int experts_per_gpu = p->n_experts / TP;
  int hidden_dim = p->hidden_dim;

  size_t start_offset = rs_now->expert_offsets->buf[tp_rank * experts_per_gpu];
  size_t end_offset = rs_now->expert_offsets->buf[(tp_rank + 1) * experts_per_gpu];
  size_t local_num_tokens = end_offset - start_offset;
  size_t local_num_bytes = local_num_tokens * hidden_dim * sizeof(float);

  void *leader_buf = rs_leader->tb3->d_buf;
  void *dst_ptr = (char *)leader_buf + start_offset * hidden_dim * sizeof(float);

  if (local_num_tokens > 0) {
    if (tp_rank == 0) {
      CHECK_HIP(hipMemcpyAsync(dst_ptr, rs_now->tb3->d_buf, local_num_bytes,
                               hipMemcpyDeviceToDevice, stream));
    } else {
      CHECK_HIP(hipMemcpyPeerAsync(dst_ptr, cur_device - tp_rank, rs_now->tb3->d_buf, cur_device,
                                   local_num_bytes, stream));

      CHECK_HIP(hipEventRecord(tp_ready, stream));
    }
  }

  pthread_barrier_wait(tp_barrier);

  if (tp_rank == 0) {
    for (int i = 1; i < TP; i++) {
      size_t worker_start_offset = rs_leader[i].expert_offsets->buf[i * experts_per_gpu];
      size_t worker_end_offset = rs_leader[i].expert_offsets->buf[(i + 1) * experts_per_gpu];

      if (worker_end_offset > worker_start_offset) {
        hipEvent_t worker_event = total_events->tp_ready[cur_device + i];
        CHECK_HIP(hipStreamWaitEvent(stream, worker_event));
      }
    }

    CHECK_HIP(hipEventRecord(tp_ready, stream));
  }

  pthread_barrier_wait(tp_barrier);

  size_t total_tokens = rs_now->expert_offsets->buf[p->n_experts];
  size_t total_bytes = total_tokens * hidden_dim * sizeof(float);

  if (tp_rank > 0) {
    hipEvent_t leader_tp_ready = total_events->tp_ready[cur_device - tp_rank];
    CHECK_HIP(hipStreamWaitEvent(stream, leader_tp_ready));

    const void *src_ptr = rs_leader->tb3->d_buf;
    void *dst_ptr = rs_now->tb3->d_buf;

    CHECK_HIP(
      hipMemcpyPeerAsync(dst_ptr, cur_device, src_ptr, cur_device - tp_rank, total_bytes, stream));

    CHECK_HIP(hipEventRecord(tp_finish, stream));
  }

  pthread_barrier_wait(tp_barrier);

  if (tp_rank == 0) {
    for (int i = 1; i < TP; i++) {
      hipEvent_t worker_tp_finish = total_events->tp_finish[cur_device + i];
      CHECK_HIP(hipStreamWaitEvent(stream, worker_tp_finish));
    }
  }

  pthread_barrier_wait(tp_barrier);
}

__global__ void extract_local_tokens_kernel(const float *x_packed, float *x_packed_local,
                                            size_t local_num_tokens, size_t start_offset,
                                            size_t hidden_dim) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elems = local_num_tokens * hidden_dim;

  if (idx < total_elems) {
    size_t global_idx = start_offset * hidden_dim + idx;
    x_packed_local[idx] = x_packed[global_idx];
  }
}

__global__ void build_local_expert_offsets_kernel(const int *expert_offsets, int *local_offsets,
                                                  int tp_rank, int experts_per_gpu) {
  int i = threadIdx.x;
  if (i > experts_per_gpu)
    return;

  int global_expert_id = tp_rank * experts_per_gpu;
  int base_offset = expert_offsets[global_expert_id];

  local_offsets[i] = expert_offsets[global_expert_id + i] - base_offset;
}

void moe_build_local_ep_data(
    OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, Config *p,
    pthread_barrier_t *tp_barrier, hipStream_t stream
) {
  size_t experts_per_gpu = p->n_experts / TP;
  size_t start_offset = rs_now->expert_offsets->buf[tp_rank * experts_per_gpu];
  size_t end_offset = rs_now->expert_offsets->buf[(tp_rank + 1) * experts_per_gpu];
  size_t local_num_tokens = end_offset - start_offset;

  if (local_num_tokens > 0) {
    dim3 block_size(256);
    dim3 grid_size((local_num_tokens * p->hidden_dim + 255) / 256);
    extract_local_tokens_kernel<<<grid_size, block_size, 0, stream>>>(
      (const float *)rs_now->x_packed->d_buf, (float *)rs_now->x_packed_local->d_buf,
      local_num_tokens, start_offset, p->hidden_dim);
  }

  {
    dim3 block_size(experts_per_gpu + 1);
    dim3 grid_size(1);

    build_local_expert_offsets_kernel<<<grid_size, block_size, 0, stream>>>(
      (const int *)rs_now->expert_offsets->d_buf, (int *)rs_now->expert_offsets_local->d_buf,
      tp_rank, experts_per_gpu);
  }
}
