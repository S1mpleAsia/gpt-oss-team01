#include "../include/parallelism.hpp"

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

__global__ void add_vector_kernel_direct(float *c, const float *a, const float *b, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    c[i] = a[i] + b[i];
  }
}

void all_gather_x(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                  int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                  hipEvent_t tp_ready, hipEvent_t tp_finish) {
  CHECK_HIP(hipEventRecord(tp_ready));

  float *d_embed_buf[TP];
  float *d_buf_ptr = (float *)rs_now->x->d_buf;
  int leader_device = cur_device - tp_rank;

  size_t offset_buf = 1ll * rs_now->x->shape[1];
  size_t offset_d_buf = 1ll * (offset_buf / TP);
  size_t elems = 1ll * offset_d_buf * sizeof(float);

  for (int i = 0; i < TP; i++) {
    d_embed_buf[i] = (float *)(rs_leader + i)->x->d_buf;
  }

  size_t src_pitch_in_bytes = 1ll * elems;
  size_t width_in_bytes = 1ll * elems;
  size_t dest_pitch_in_bytes = 1ll * elems * TP;
  size_t height_in_elems = cur_batch_size;

  pthread_barrier_wait(tp_barrier);

  for (int i = 0; i < TP; i++) {
    hipEvent_t tp_ready_each = total_events->tp_ready[leader_device + i];
    CHECK_HIP(hipStreamWaitEvent(stream, tp_ready_each));
  }

  for (int i = 0; i < TP; i++) {
    if (i == tp_rank)
      continue;
    size_t dest_offset = 1ll * i * offset_d_buf;
    float *dest_ptr = (float *)d_buf_ptr + dest_offset;
    float *src_ptr = (float *)d_embed_buf[i] + dest_offset;

    // Perform the 2D asynchronous copy.
    CHECK_HIP(hipMemcpy2DAsync(
      dest_ptr, dest_pitch_in_bytes, src_ptr, dest_pitch_in_bytes, width_in_bytes, height_in_elems,
      hipMemcpyDeviceToDevice,  // Note: This copy is within the same device's perspective, but `hipMemcpyPeerAsync` is preferred for cross-device copies.
      stream));
  }
}

void all_gather_x_new(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                      int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                      hipEvent_t tp_ready, hipEvent_t tp_finish) {
  // GpuTimer timer("all_gather_x", stream);
  int reduce_rank = tp_rank / 2;
  int leader_rank = tp_rank % 2;
  int partner_offset = (leader_rank == 0) ? 1 : -1;
  int leader_offset = (reduce_rank == 0) ? 2 : -2;

  float *d_buf_ptr = (float *)rs_now->x->d_buf;
  float *d_tmp_buf_partner = (float *)(rs_now + partner_offset)->x->d_buf;
  float *d_buf_leader = (float *)(rs_now + leader_offset)->x->d_buf;
  int leader_device = cur_device - tp_rank;

  size_t offset_buf = 1ll * rs_now->x->shape[1];
  size_t offset_d_buf = 1ll * (offset_buf / TP);
  size_t elems = 1ll * offset_d_buf * sizeof(float);

  size_t src_pitch_in_bytes = 1ll * elems;
  size_t width_in_bytes = 1ll * elems;
  size_t dest_pitch_in_bytes = 1ll * elems * TP;
  size_t height_in_elems = cur_batch_size;

  CHECK_HIP(hipEventRecord(tp_ready, stream));

  pthread_barrier_wait(tp_barrier + reduce_rank + 1);

  hipEvent_t tp_ready_partner = total_events->tp_ready[cur_device + partner_offset];
  CHECK_HIP(hipStreamWaitEvent(stream, tp_ready_partner));

  size_t partner_pos_offset = 1ll * (tp_rank + partner_offset) * offset_d_buf;

  // Perform the 2D asynchronous copy.
  CHECK_HIP(hipMemcpy2DAsync(d_buf_ptr + partner_pos_offset, dest_pitch_in_bytes,
                             d_tmp_buf_partner + partner_pos_offset, dest_pitch_in_bytes,
                             width_in_bytes, height_in_elems, hipMemcpyDeviceToDevice, stream));

  CHECK_HIP(hipEventRecord(tp_finish, stream));

  pthread_barrier_wait(tp_barrier + leader_rank + 3);

  hipEvent_t tp_finish_leader = total_events->tp_finish[cur_device + leader_offset];
  CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_leader));

  size_t pos_offset = 1ll * (tp_rank - leader_rank + leader_offset) * offset_d_buf;

  // Perform the 2D asynchronous copy.
  CHECK_HIP(hipMemcpy2DAsync(
    d_buf_ptr + pos_offset, dest_pitch_in_bytes, d_buf_leader + pos_offset, dest_pitch_in_bytes,
    width_in_bytes * 2, height_in_elems,
    hipMemcpyDeviceToDevice,  // Note: This copy is within the same device's perspective, but `hipMemcpyPeerAsync` is preferred for cross-device copies.
    stream));

  // pthread_barrier_wait(tp_barrier + leader_rank + 3);
}

void all_gather_x_full_tp(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                          int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                          hipEvent_t tp_ready, hipEvent_t tp_finish) {
  // GpuTimer timer("all_gather_x_full");

  // Phase 1: 0-1 | 2-3 | 4-5 | 6-7
  int reduce_group = tp_rank / 2;
  int local_rank = tp_rank % 2;
  int partner_offset = (local_rank == 0) ? 1 : -1;

  float *d_buf_ptr = (float *)rs_now->x->d_buf;
  float *d_buf_partner_ptr = (float *)(rs_now + partner_offset)->x->d_buf;

  size_t hidden_dim = rs_now->x->shape[1];
  size_t shard_dim = hidden_dim / TP;
  size_t shard_size = shard_dim * sizeof(float);

  size_t width = shard_size;
  size_t pitch_size = hidden_dim * sizeof(float);
  size_t height = cur_batch_size;

  CHECK_HIP(hipEventRecord(tp_ready, stream));
  // pthread_barrier_wait(tp_barrier);
  pthread_barrier_wait(tp_barrier + reduce_group + 1);

  hipEvent_t tp_ready_partner = total_events->tp_ready[cur_device + partner_offset];
  CHECK_HIP(hipStreamWaitEvent(stream, tp_ready_partner));

  size_t partner_pos_offset = (tp_rank + partner_offset) * shard_dim;

  CHECK_HIP(hipMemcpy2DAsync(d_buf_ptr + partner_pos_offset, pitch_size,
                             d_buf_partner_ptr + partner_pos_offset, pitch_size, width, height,
                             hipMemcpyDeviceToDevice, stream));

  CHECK_HIP(hipEventRecord(tp_finish, stream));

  // Phase 2: 0-2 | 1-3 | 4-6 | 5-7
  // pthread_barrier_wait(tp_barrier);
  if (tp_rank == 0 || tp_rank == 2)
    pthread_barrier_wait(tp_barrier + 5);
  else if (tp_rank == 1 || tp_rank == 3)
    pthread_barrier_wait(tp_barrier + 6);
  else if (tp_rank == 4 || tp_rank == 6)
    pthread_barrier_wait(tp_barrier + 7);
  else if (tp_rank == 5 || tp_rank == 7)
    pthread_barrier_wait(tp_barrier + 8);

  int p = 1;
  int mask = 1 << p;
  int partner = tp_rank ^ mask;
  int chunk_size = 1 << p;
  partner_offset = partner - tp_rank;

  hipEvent_t tp_finish_partner_2 = total_events->tp_finish[cur_device + partner_offset];
  CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_partner_2));

  float *d_buf_ptr_2 = (float *)rs_now->x->d_buf;
  float *d_buf_partner_ptr_2 = (float *)(rs_now + partner_offset)->x->d_buf;
  size_t width_2 = chunk_size * shard_size;

  const int partner_base = (tp_rank & ~(chunk_size - 1)) ^ mask;
  const size_t col_offset = (size_t)partner_base * shard_dim;

  CHECK_HIP(hipMemcpy2DAsync(d_buf_ptr_2 + col_offset, pitch_size, d_buf_partner_ptr_2 + col_offset,
                             pitch_size, width_2, height, hipMemcpyDeviceToDevice, stream));
  CHECK_HIP(hipEventRecord(tp_finish, stream));

  // Phase 3: 0-4 | 2-6 | 1-5 | 3-7
  // pthread_barrier_wait(tp_barrier);
  if (tp_rank == 0 || tp_rank == 4)
    pthread_barrier_wait(tp_barrier + 9);
  if (tp_rank == 2 || tp_rank == 6)
    pthread_barrier_wait(tp_barrier + 10);
  if (tp_rank == 1 || tp_rank == 5)
    pthread_barrier_wait(tp_barrier + 11);
  if (tp_rank == 3 || tp_rank == 7)
    pthread_barrier_wait(tp_barrier + 12);

  p = 2;
  mask = 1 << p;
  partner = tp_rank ^ mask;
  chunk_size = 1 << p;
  partner_offset = partner - tp_rank;

  hipEvent_t tp_finish_partner_3 = total_events->tp_finish[cur_device + partner_offset];
  CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_partner_3));

  float *d_buf_ptr_3 = (float *)rs_now->x->d_buf;
  float *d_buf_partner_ptr_3 = (float *)(rs_now + partner_offset)->x->d_buf;
  size_t width_3 = chunk_size * shard_size;

  const int partner_base_last = (tp_rank & ~(chunk_size - 1)) ^ mask;
  const size_t col_offset_last = (size_t)partner_base_last * shard_dim;

  CHECK_HIP(hipMemcpy2DAsync(d_buf_ptr_3 + col_offset_last, pitch_size,
                             d_buf_partner_ptr_3 + col_offset_last, pitch_size, width_3, height,
                             hipMemcpyDeviceToDevice, stream));
}

void reduce_tb3(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                hipEvent_t tp_ready, hipEvent_t tp_finish) {
  // GpuTimer timer("reduce_tb3", stream);

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

void reduce_tb3_new(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                    int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                    hipEvent_t tp_ready, hipEvent_t tp_finish) {
  // GpuTimer timer("reduce_tb3_new", stream);

  // only works for TP == 4
  size_t active_elems = (size_t)cur_batch_size * rs_now->tb3->shape[1] * rs_now->tb3->shape[2];
  size_t active_num_bytes = active_elems * rs_now->tb3->get_dtype_size();

  // Ranks 1 and 2 are leaders. Ranks 0 and 3 are partners.
  bool is_leader = (tp_rank == 0 || tp_rank == 2);
  bool is_partner = !is_leader;
  int reduce_rank = tp_rank / 2;

  // --- Phase 1: Partners (0 and 3) send their data to leaders (1 and 2)
  if (is_partner) {
    // Rank 0 sends to leader 1 (device +1). Rank 3 sends to leader 2 (device -1).
    OurRunState *rs_leader_peer = rs_now - 1;
    void *dst_buf = rs_leader_peer->tb3_buf->d_buf;
    const void *src_buf = rs_now->tb3->d_buf;

    CHECK_HIP(
      hipMemcpyPeerAsync(dst_buf, cur_device - 1, src_buf, cur_device, active_num_bytes, stream));
    CHECK_HIP(hipEventRecord(tp_ready, stream));
  }

  pthread_barrier_wait(tp_barrier + reduce_rank + 1);

  // --- Phase 2: Per-pair leaders (1 and 2) add their partner's data
  if (is_leader) {
    // Leader 1 waits for partner 0 (device -1). Leader 2 waits for partner 3 (device +1).
    hipEvent_t partner_ready = total_events->tp_ready[cur_device + 1];
    CHECK_HIP(hipStreamWaitEvent(stream, partner_ready));
    int lead_offset = (tp_rank == 0) ? 2 : -2;

    const float *d_buf = (const float *)rs_now->tb3->d_buf;
    float *d_buf_each = (float *)rs_now->tb3_buf->d_buf;

    const int block_size = 256;
    const int grid_size = (active_elems + block_size - 1) / block_size;

    add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(d_buf_each, d_buf,
                                                                    active_elems);

    // Copy rank 2's tb3 into rank 1's per-rank buf.
    const void *src_buf = rs_now->tb3_buf->d_buf;
    void *dst_buf = (void *)((float *)(rs_now + lead_offset)->tb3_buf->d_buf + active_elems);

    CHECK_HIP(hipMemcpyPeerAsync(dst_buf, cur_device + lead_offset, src_buf, cur_device,
                                 active_num_bytes, stream));

    CHECK_HIP(hipEventRecord(tp_ready, stream));

    pthread_barrier_wait(tp_barrier + 3);

    // Rank 1 is the final aggregator. It waits for rank 2's data.
    hipEvent_t other_leader_ready = total_events->tp_ready[cur_device + lead_offset];
    CHECK_HIP(hipStreamWaitEvent(stream, other_leader_ready));

    // Add rank 2's data into rank 1's tb3.
    float *d_buf_now = (float *)rs_now->tb3->d_buf;
    const float *d_buf_1 = (const float *)rs_now->tb3_buf->d_buf;
    const float *d_buf_2 = (const float *)(rs_now->tb3_buf->d_buf) + active_elems;

    add_vector_kernel_direct<<<grid_size, block_size, 0, stream>>>(d_buf_now, d_buf_1, d_buf_2,
                                                                   active_elems);

    CHECK_HIP(hipEventRecord(tp_finish, stream));
  }

  pthread_barrier_wait(tp_barrier + reduce_rank + 1);

  // --- Phase 4: Final broadcast from Rank 1 to all other ranks
  if (is_partner) {
    // Rank 1 broadcasts the final aggregated data to all other ranks.
    hipEvent_t tp_finish_leader = total_events->tp_finish[cur_device - 1];

    CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_leader));

    void *dst_buf = rs_now->tb3->d_buf;
    const void *src_buf = (rs_now - 1)->tb3->d_buf;  // rs_leader is now the rank 1 runstate

    CHECK_HIP(
      hipMemcpyPeerAsync(dst_buf, cur_device, src_buf, cur_device - 1, active_num_bytes, stream));
    CHECK_HIP(hipEventRecord(tp_finish, stream));
  } else {
    // All other ranks (0, 2, 3) wait for Rank 1's final result and copy it.
    hipEvent_t tp_finish_each = total_events->tp_finish[cur_device + 1];
    CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_each));
  }

  // pthread_barrier_wait(tp_barrier + reduce_rank + 1);
}

void ring_all_reduce_tb3(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                         int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream) {
  // GpuTimer timer("ring_all_reduce_tb3", stream);

  float *d_buf_now = (float *)rs_now->tb3->d_buf;
  size_t total_elems = cur_batch_size * rs_now->tb3->shape[1] * rs_now->tb3->shape[2];
  size_t chunk_elems = total_elems / TP;
  size_t chunk_bytes = chunk_elems * rs_now->tb3->get_dtype_size();

  int left_peer_rank = (tp_rank - 1 + TP) % TP;
  OurRunState *rs_left = &rs_leader[left_peer_rank];
  int left_device = cur_device - tp_rank + left_peer_rank;

  void *tmp_ptr = rs_now->tb3_recv->d_buf;

  const int block_size = 256;
  const int grid_size = (chunk_elems + block_size - 1) / block_size;

  CHECK_HIP(hipStreamSynchronize(stream));
  pthread_barrier_wait(tp_barrier);

  for (int i = 0; i < TP - 1; ++i) {
    int chunk_idx = (tp_rank - i - 1 + TP) % TP;
    size_t chunk_offset = chunk_idx * chunk_elems;

    const float *left_src_ptr = (const float *)rs_left->tb3->d_buf + chunk_offset;
    float *dst_local_ptr = d_buf_now + chunk_offset;

    CHECK_HIP(
      hipMemcpyPeerAsync(tmp_ptr, cur_device, left_src_ptr, left_device, chunk_bytes, stream));

    add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(
      dst_local_ptr, (const float *)tmp_ptr, chunk_elems);

    CHECK_HIP(hipStreamSynchronize(stream));
    pthread_barrier_wait(tp_barrier);
  }

  for (int i = 0; i < TP - 1; i++) {
    int send_chunk_idx = (tp_rank - i + TP) % TP;
    size_t chunk_offset = send_chunk_idx * chunk_elems;

    const float *left_src_ptr = (const float *)rs_left->tb3->d_buf + chunk_offset;
    float *dst_local_ptr = d_buf_now + chunk_offset;

    CHECK_HIP(hipMemcpyPeerAsync(dst_local_ptr, cur_device, left_src_ptr, left_device, chunk_bytes,
                                 stream));

    CHECK_HIP(hipStreamSynchronize(stream));
    pthread_barrier_wait(tp_barrier);
  }
}

void reduce_tb2(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                hipEvent_t tp_ready, hipEvent_t tp_finish) {
  // GpuTimer timer("reduce_tb2", stream);
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

void reduce_tb2_full_tp(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                        int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                        hipEvent_t tp_ready, hipEvent_t tp_finish) {
  // GpuTimer timer("reduce_tb2_full");

  // Phase 1: 0-1 | 2-3 | 4-5 | 6-7
  int reduce_group = tp_rank / 2;
  int local_rank = tp_rank % 2;
  int partner_offset = (local_rank == 0) ? 1 : -1;

  float *tb2_ptr = (float *)rs_now->tb2->d_buf;
  float *local_ptr = (float *)rs_now->tb2_buf->d_buf;
  float *tb2_partner_ptr = (float *)(rs_now + partner_offset)->tb2->d_buf;
  size_t active_elems = (size_t)cur_batch_size * rs_now->tb2->shape[1];
  size_t active_num_bytes = active_elems * rs_now->tb2->get_dtype_size();

  const int block_size = 256;
  const int grid_size = (active_elems + block_size - 1) / block_size;

  CHECK_HIP(hipEventRecord(tp_ready, stream));
  // pthread_barrier_wait(tp_barrier);
  pthread_barrier_wait(tp_barrier + reduce_group + 1);

  hipEvent_t tp_ready_partner = total_events->tp_ready[cur_device + partner_offset];
  CHECK_HIP(hipStreamWaitEvent(stream, tp_ready_partner));

  CHECK_HIP(hipMemcpyPeerAsync(local_ptr, cur_device, tb2_partner_ptr, cur_device + partner_offset,
                               active_num_bytes, stream));
  add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(tb2_ptr, (const float *)local_ptr,
                                                                  active_elems);

  CHECK_HIP(hipEventRecord(tp_finish, stream));

  // Phase 2: 0-2 | 1-3 | 4-6 | 5-7
  // pthread_barrier_wait(tp_barrier);
  if (tp_rank == 0 || tp_rank == 2)
    pthread_barrier_wait(tp_barrier + 5);
  else if (tp_rank == 1 || tp_rank == 3)
    pthread_barrier_wait(tp_barrier + 6);
  else if (tp_rank == 4 || tp_rank == 6)
    pthread_barrier_wait(tp_barrier + 7);
  else if (tp_rank == 5 || tp_rank == 7)
    pthread_barrier_wait(tp_barrier + 8);

  int p = 1;
  int mask = 1 << p;
  int partner = tp_rank ^ mask;
  int chunk_size = 1 << p;
  partner_offset = partner - tp_rank;

  hipEvent_t tp_finish_partner_2 = total_events->tp_finish[cur_device + partner_offset];
  CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_partner_2));

  tb2_ptr = (float *)rs_now->tb2->d_buf;
  local_ptr = (float *)rs_now->tb2_buf->d_buf;
  tb2_partner_ptr = (float *)(rs_now + partner_offset)->tb2->d_buf;

  CHECK_HIP(hipMemcpyPeerAsync(local_ptr, cur_device, tb2_partner_ptr, cur_device + partner_offset,
                               active_num_bytes, stream));
  add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(tb2_ptr, (const float *)local_ptr,
                                                                  active_elems);

  CHECK_HIP(hipEventRecord(tp_finish, stream));

  // Phase 3: 0-4 | 2-6 | 1-5 | 3-7
  // pthread_barrier_wait(tp_barrier);
  if (tp_rank == 0 || tp_rank == 4)
    pthread_barrier_wait(tp_barrier + 9);
  if (tp_rank == 2 || tp_rank == 6)
    pthread_barrier_wait(tp_barrier + 10);
  if (tp_rank == 1 || tp_rank == 5)
    pthread_barrier_wait(tp_barrier + 11);
  if (tp_rank == 3 || tp_rank == 7)
    pthread_barrier_wait(tp_barrier + 12);

  p = 2;
  mask = 1 << p;
  partner = tp_rank ^ mask;
  chunk_size = 1 << p;
  partner_offset = partner - tp_rank;

  hipEvent_t tp_finish_partner_3 = total_events->tp_finish[cur_device + partner_offset];
  CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_partner_3));

  tb2_ptr = (float *)rs_now->tb2->d_buf;
  local_ptr = (float *)rs_now->tb2_buf->d_buf;
  tb2_partner_ptr = (float *)(rs_now + partner_offset)->tb2->d_buf;

  CHECK_HIP(hipMemcpyPeerAsync(local_ptr, cur_device, tb2_partner_ptr, cur_device + partner_offset,
                               active_num_bytes, stream));
  add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(tb2_ptr, (const float *)local_ptr,
                                                                  active_elems);
}

void reduce_tb2_new(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                    int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                    hipEvent_t tp_ready, hipEvent_t tp_finish) {
  // GpuTimer timer("reduce_tb2_new", stream);
  // only works for TP == 4
  size_t active_elems = (size_t)cur_batch_size * rs_now->tb2->shape[1];
  size_t active_num_bytes = active_elems * rs_now->tb2->get_dtype_size();

  // Ranks 1 and 2 are leaders. Ranks 0 and 3 are partners.
  bool is_leader = (tp_rank == 0 || tp_rank == 2);
  bool is_partner = !is_leader;
  int reduce_rank = tp_rank / 2;

  // --- Phase 1: Partners (0 and 3) send their data to leaders (1 and 2)
  if (is_partner) {
    // Rank 0 sends to leader 1 (device +1). Rank 3 sends to leader 2 (device -1).
    OurRunState *rs_leader_peer = rs_now - 1;
    void *dst_buf = rs_leader_peer->tb2_buf->d_buf;
    const void *src_buf = rs_now->tb2->d_buf;

    CHECK_HIP(
      hipMemcpyPeerAsync(dst_buf, cur_device - 1, src_buf, cur_device, active_num_bytes, stream));
    CHECK_HIP(hipEventRecord(tp_ready, stream));
  }

  pthread_barrier_wait(tp_barrier + reduce_rank + 1);

  // --- Phase 2: Per-pair leaders (1 and 2) add their partner's data
  if (is_leader) {
    // Leader 1 waits for partner 0 (device -1). Leader 2 waits for partner 3 (device +1).
    hipEvent_t partner_ready = total_events->tp_ready[cur_device + 1];
    CHECK_HIP(hipStreamWaitEvent(stream, partner_ready));
    int lead_offset = (tp_rank == 0) ? 2 : -2;

    const float *d_buf = (const float *)rs_now->tb2->d_buf;
    float *d_buf_each = (float *)rs_now->tb2_buf->d_buf;

    const int block_size = 256;
    const int grid_size = (active_elems + block_size - 1) / block_size;

    add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(d_buf_each, d_buf,
                                                                    active_elems);

    // Copy rank 2's tb2 into rank 1's per-rank buf.
    const void *src_buf = rs_now->tb2_buf->d_buf;
    void *dst_buf = (void *)((float *)(rs_now + lead_offset)->tb2_buf->d_buf + active_elems);

    CHECK_HIP(hipMemcpyPeerAsync(dst_buf, cur_device + lead_offset, src_buf, cur_device,
                                 active_num_bytes, stream));

    CHECK_HIP(hipEventRecord(tp_ready, stream));

    pthread_barrier_wait(tp_barrier + 3);

    // Rank 1 is the final aggregator. It waits for rank 2's data.
    hipEvent_t other_leader_ready = total_events->tp_ready[cur_device + lead_offset];
    CHECK_HIP(hipStreamWaitEvent(stream, other_leader_ready));

    // Add rank 2's data into rank 1's tb2.
    float *d_buf_now = (float *)rs_now->tb2->d_buf;
    const float *d_buf_1 = (const float *)rs_now->tb2_buf->d_buf;
    const float *d_buf_2 = (const float *)(rs_now->tb2_buf->d_buf) + active_elems;

    add_vector_kernel_direct<<<grid_size, block_size, 0, stream>>>(d_buf_now, d_buf_1, d_buf_2,
                                                                   active_elems);

    CHECK_HIP(hipEventRecord(tp_finish, stream));
  }

  pthread_barrier_wait(tp_barrier + reduce_rank + 1);

  // --- Phase 4: Final broadcast from Rank 1 to all other ranks
  if (is_partner) {
    // Rank 1 broadcasts the final aggregated data to all other ranks.
    hipEvent_t tp_finish_leader = total_events->tp_finish[cur_device - 1];

    CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_leader));

    void *dst_buf = rs_now->tb2->d_buf;
    const void *src_buf = (rs_now - 1)->tb2->d_buf;  // rs_leader is now the rank 1 runstate

    CHECK_HIP(
      hipMemcpyPeerAsync(dst_buf, cur_device, src_buf, cur_device - 1, active_num_bytes, stream));
    CHECK_HIP(hipEventRecord(tp_finish, stream));
  } else {
    // All other ranks (0, 2, 3) wait for Rank 1's final result and copy it.
    hipEvent_t tp_finish_each = total_events->tp_finish[cur_device + 1];
    CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_each));
  }

  // pthread_barrier_wait(tp_barrier + reduce_rank + 1);
}

void ring_all_reduce_tb2(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                         int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream) {
  // GpuTimer timer("ring_all_reduce_tb2", stream);

  float *d_buf_now = (float *)rs_now->tb2->d_buf;
  size_t hidden_dim = rs_now->tb2->shape[1];
  size_t total_elems = cur_batch_size * hidden_dim;
  size_t chunk_elems = total_elems / TP;
  size_t chunk_bytes = chunk_elems * rs_now->tb2->get_dtype_size();

  int left_peer_rank = (tp_rank - 1 + TP) % TP;
  OurRunState *rs_left = &rs_leader[left_peer_rank];
  int left_device = cur_device - tp_rank + left_peer_rank;

  void *tmp_ptr = rs_now->tb2_recv->d_buf;

  const int block_size = 256;
  const int grid_size = (chunk_elems + block_size - 1) / block_size;

  CHECK_HIP(hipStreamSynchronize(stream));
  pthread_barrier_wait(tp_barrier);

  for (int i = 0; i < TP - 1; ++i) {
    int chunk_idx = (tp_rank - i - 1 + TP) % TP;
    size_t chunk_offset = chunk_idx * chunk_elems;

    const float *left_src_ptr = (const float *)rs_left->tb2->d_buf + chunk_offset;
    float *dst_local_ptr = d_buf_now + chunk_offset;

    CHECK_HIP(
      hipMemcpyPeerAsync(tmp_ptr, cur_device, left_src_ptr, left_device, chunk_bytes, stream));

    add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(
      dst_local_ptr, (const float *)tmp_ptr, chunk_elems);

    CHECK_HIP(hipStreamSynchronize(stream));
    pthread_barrier_wait(tp_barrier);
  }

  for (int i = 0; i < TP - 1; i++) {
    int send_chunk_idx = (tp_rank - i + TP) % TP;
    size_t chunk_offset = send_chunk_idx * chunk_elems;

    const float *left_src_ptr = (const float *)rs_left->tb2->d_buf + chunk_offset;
    float *dst_local_ptr = d_buf_now + chunk_offset;

    CHECK_HIP(hipMemcpyPeerAsync(dst_local_ptr, cur_device, left_src_ptr, left_device, chunk_bytes,
                                 stream));

    CHECK_HIP(hipStreamSynchronize(stream));
    pthread_barrier_wait(tp_barrier);
  }
}

void all_gather_classifier_final(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank,
                                 int cur_device, int cur_batch_size, pthread_barrier_t *tp_barrier,
                                 hipStream_t stream, hipEvent_t tp_ready, hipEvent_t tp_finish) {
  // GpuTimer timer("all_gather_classifier", stream);

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

void all_gather_classifier_v2(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank,
                              int cur_device, int cur_batch_size, pthread_barrier_t *tp_barrier,
                              hipStream_t stream, hipEvent_t tp_ready, hipEvent_t tp_finish) {
  // GpuTimer timer("all_gather_classifier", stream);

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

void all_gather_logits_id(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                          int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                          hipEvent_t tp_ready, hipEvent_t tp_finish) {
  const float *src_max_buf = (const float *)rs_now->logits_max->d_buf;
  size_t dst_max_offset = 1ll * tp_rank * rs_leader->logits_max_total->shape[1];
  float *dst_max_buf = (float *)rs_leader->logits_max_total->d_buf + dst_max_offset;
  size_t bytes_max = 1ll * cur_batch_size * sizeof(float);

  CHECK_HIP(hipMemcpyPeerAsync(dst_max_buf, cur_device - tp_rank, src_max_buf, cur_device,
                               bytes_max, stream));

  const int *src_id_buf = (const int *)rs_now->logits_id->d_buf;
  size_t dst_id_offset = 1ll * tp_rank * rs_leader->logits_id_total->shape[1];
  int *dst_id_buf = (int *)rs_leader->logits_id_total->d_buf + dst_id_offset;
  size_t bytes_id = 1ll * cur_batch_size * sizeof(int);

  CHECK_HIP(
    hipMemcpyPeerAsync(dst_id_buf, cur_device - tp_rank, src_id_buf, cur_device, bytes_id, stream));

  CHECK_HIP(hipEventRecord(tp_ready, stream));

  pthread_barrier_wait(tp_barrier);
}

void all_gather_qkv_v2(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                       int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                       hipEvent_t tp_ready, hipEvent_t tp_finish) {
  // GpuTimer timer("all_gather_qkv", stream);

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

void all_gather_tb3(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                    Config *p, pthread_barrier_t *tp_barrier, hipStream_t stream,
                    hipEvent_t tp_ready, hipEvent_t tp_finish) {
  // GpuTimer timer("all_gather_tb3", stream);

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

void ring_reduce_agg(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                     int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream) {
  // GpuTimer timer("ring_reduce_agg", stream);

  float *d_buf_now = (float *)rs_now->e_agg->d_buf;
  size_t hidden_dim = rs_now->e_agg->shape[1];
  size_t total_elems = cur_batch_size * hidden_dim;
  size_t chunk_elems = total_elems / TP;
  size_t chunk_bytes = chunk_elems * rs_now->e_agg->get_dtype_size();

  int left_peer_rank = (tp_rank - 1 + TP) % TP;
  OurRunState *rs_left = &rs_leader[left_peer_rank];
  int left_device = cur_device - tp_rank + left_peer_rank;

  void *tmp_ptr = rs_now->e_agg_recv->d_buf;

  const int block_size = 256;
  const int grid_size = (chunk_elems + block_size - 1) / block_size;

  CHECK_HIP(hipStreamSynchronize(stream));
  pthread_barrier_wait(tp_barrier);

  for (int i = 0; i < TP - 1; ++i) {
    int chunk_idx = (tp_rank - i - 1 + TP) % TP;
    size_t chunk_offset = chunk_idx * chunk_elems;

    const float *left_src_ptr = (const float *)rs_left->e_agg->d_buf + chunk_offset;
    float *dst_local_ptr = d_buf_now + chunk_offset;

    CHECK_HIP(
      hipMemcpyPeerAsync(tmp_ptr, cur_device, left_src_ptr, left_device, chunk_bytes, stream));

    add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(
      dst_local_ptr, (const float *)tmp_ptr, chunk_elems);

    CHECK_HIP(hipStreamSynchronize(stream));
    pthread_barrier_wait(tp_barrier);
  }

  for (int i = 0; i < TP - 1; i++) {
    int send_chunk_idx = (tp_rank - i + TP) % TP;
    size_t chunk_offset = send_chunk_idx * chunk_elems;

    const float *left_src_ptr = (const float *)rs_left->e_agg->d_buf + chunk_offset;
    float *dst_local_ptr = d_buf_now + chunk_offset;

    CHECK_HIP(hipMemcpyPeerAsync(dst_local_ptr, cur_device, left_src_ptr, left_device, chunk_bytes,
                                 stream));

    CHECK_HIP(hipStreamSynchronize(stream));
    pthread_barrier_wait(tp_barrier);
  }
}

void reduce_agg_full_tp(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                        int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                        hipEvent_t tp_ready, hipEvent_t tp_finish) {
  // GpuTimer timer("reduce_agg_full");

  // Phase 1: 0-1 | 2-3 | 4-5 | 6-7
  int reduce_group = tp_rank / 2;
  int local_rank = tp_rank % 2;
  int partner_offset = (local_rank == 0) ? 1 : -1;

  float *e_agg_ptr = (float *)rs_now->e_agg->d_buf;
  float *local_ptr = (float *)rs_now->e_agg_buf->d_buf;
  float *e_agg_partner_ptr = (float *)(rs_now + partner_offset)->e_agg->d_buf;
  size_t active_elems = (size_t)cur_batch_size * rs_now->e_agg->shape[1];
  size_t active_num_bytes = active_elems * rs_now->e_agg->get_dtype_size();

  const int block_size = 256;
  const int grid_size = (active_elems + block_size - 1) / block_size;

  CHECK_HIP(hipEventRecord(tp_ready, stream));
  // pthread_barrier_wait(tp_barrier);
  pthread_barrier_wait(tp_barrier + reduce_group + 1);

  hipEvent_t tp_ready_partner = total_events->tp_ready[cur_device + partner_offset];
  CHECK_HIP(hipStreamWaitEvent(stream, tp_ready_partner));

  CHECK_HIP(hipMemcpyPeerAsync(local_ptr, cur_device, e_agg_partner_ptr,
                               cur_device + partner_offset, active_num_bytes, stream));
  add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(
    e_agg_ptr, (const float *)local_ptr, active_elems);

  CHECK_HIP(hipEventRecord(tp_finish, stream));

  // Phase 2: 0-2 | 1-3 | 4-6 | 5-7
  // pthread_barrier_wait(tp_barrier);
  if (tp_rank == 0 || tp_rank == 2)
    pthread_barrier_wait(tp_barrier + 5);
  else if (tp_rank == 1 || tp_rank == 3)
    pthread_barrier_wait(tp_barrier + 6);
  else if (tp_rank == 4 || tp_rank == 6)
    pthread_barrier_wait(tp_barrier + 7);
  else if (tp_rank == 5 || tp_rank == 7)
    pthread_barrier_wait(tp_barrier + 8);

  int p = 1;
  int mask = 1 << p;
  int partner = tp_rank ^ mask;
  int chunk_size = 1 << p;
  partner_offset = partner - tp_rank;

  hipEvent_t tp_finish_partner_2 = total_events->tp_finish[cur_device + partner_offset];
  CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_partner_2));

  e_agg_ptr = (float *)rs_now->e_agg->d_buf;
  local_ptr = (float *)rs_now->e_agg_buf->d_buf;
  e_agg_partner_ptr = (float *)(rs_now + partner_offset)->e_agg->d_buf;

  CHECK_HIP(hipMemcpyPeerAsync(local_ptr, cur_device, e_agg_partner_ptr,
                               cur_device + partner_offset, active_num_bytes, stream));
  add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(
    e_agg_ptr, (const float *)local_ptr, active_elems);

  CHECK_HIP(hipEventRecord(tp_finish, stream));

  // Phase 3: 0-4 | 2-6 | 1-5 | 3-7
  // pthread_barrier_wait(tp_barrier);
  if (tp_rank == 0 || tp_rank == 4)
    pthread_barrier_wait(tp_barrier + 9);
  if (tp_rank == 2 || tp_rank == 6)
    pthread_barrier_wait(tp_barrier + 10);
  if (tp_rank == 1 || tp_rank == 5)
    pthread_barrier_wait(tp_barrier + 11);
  if (tp_rank == 3 || tp_rank == 7)
    pthread_barrier_wait(tp_barrier + 12);

  p = 2;
  mask = 1 << p;
  partner = tp_rank ^ mask;
  chunk_size = 1 << p;
  partner_offset = partner - tp_rank;

  hipEvent_t tp_finish_partner_3 = total_events->tp_finish[cur_device + partner_offset];
  CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_partner_3));

  e_agg_ptr = (float *)rs_now->e_agg->d_buf;
  local_ptr = (float *)rs_now->e_agg_buf->d_buf;
  e_agg_partner_ptr = (float *)(rs_now + partner_offset)->e_agg->d_buf;

  CHECK_HIP(hipMemcpyPeerAsync(local_ptr, cur_device, e_agg_partner_ptr,
                               cur_device + partner_offset, active_num_bytes, stream));
  add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(
    e_agg_ptr, (const float *)local_ptr, active_elems);
}

void reduce_agg(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                hipEvent_t tp_ready, hipEvent_t tp_finish) {
  // GpuTimer timer("reduce_agg", stream);

  size_t active_elems = (size_t)cur_batch_size * rs_now->e_agg->shape[1];
  size_t active_num_bytes = active_elems * rs_now->e_agg->get_dtype_size();

  bool is_leader = (tp_rank == 0 || tp_rank == 2);
  bool is_partner = !is_leader;
  int reduce_rank = tp_rank / 2;

  if (is_partner) {
    OurRunState *rs_leader_peer = rs_now - 1;
    void *dst_buf = rs_leader_peer->tb2_buf->d_buf;
    const void *src_buf = rs_now->e_agg->d_buf;

    CHECK_HIP(
      hipMemcpyPeerAsync(dst_buf, cur_device - 1, src_buf, cur_device, active_num_bytes, stream));
    CHECK_HIP(hipEventRecord(tp_ready, stream));
  }

  pthread_barrier_wait(tp_barrier + reduce_rank + 1);

  if (is_leader) {
    hipEvent_t partner_ready = total_events->tp_ready[cur_device + 1];
    CHECK_HIP(hipStreamWaitEvent(stream, partner_ready));
    int lead_offset = (tp_rank == 0) ? 2 : -2;

    const float *d_buf = (const float *)rs_now->e_agg->d_buf;
    float *d_buf_each = (float *)rs_now->tb2_buf->d_buf;

    const int block_size = 256;
    const int grid_size = (active_elems + block_size - 1) / block_size;

    add_vector_kernel_batched<<<grid_size, block_size, 0, stream>>>(d_buf_each, d_buf,
                                                                    active_elems);

    const void *src_buf = rs_now->tb2_buf->d_buf;
    void *dst_buf = (void *)((float *)(rs_now + lead_offset)->tb2_buf->d_buf + active_elems);

    CHECK_HIP(hipMemcpyPeerAsync(dst_buf, cur_device + lead_offset, src_buf, cur_device,
                                 active_num_bytes, stream));

    CHECK_HIP(hipEventRecord(tp_ready, stream));

    pthread_barrier_wait(tp_barrier + 3);

    hipEvent_t other_leader_ready = total_events->tp_ready[cur_device + lead_offset];
    CHECK_HIP(hipStreamWaitEvent(stream, other_leader_ready));

    float *d_buf_now = (float *)rs_now->e_agg->d_buf;
    const float *d_buf_1 = (const float *)rs_now->tb2_buf->d_buf;
    const float *d_buf_2 = (const float *)(rs_now->tb2_buf->d_buf) + active_elems;

    add_vector_kernel_direct<<<grid_size, block_size, 0, stream>>>(d_buf_now, d_buf_1, d_buf_2,
                                                                   active_elems);

    CHECK_HIP(hipEventRecord(tp_finish, stream));
  }

  pthread_barrier_wait(tp_barrier + reduce_rank + 1);

  if (is_partner) {
    hipEvent_t tp_finish_leader = total_events->tp_finish[cur_device - 1];

    CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_leader));

    void *dst_buf = rs_now->e_agg->d_buf;
    const void *src_buf = (rs_now - 1)->e_agg->d_buf;

    CHECK_HIP(
      hipMemcpyPeerAsync(dst_buf, cur_device, src_buf, cur_device - 1, active_num_bytes, stream));
    CHECK_HIP(hipEventRecord(tp_finish, stream));
  } else {
    hipEvent_t tp_finish_each = total_events->tp_finish[cur_device + 1];
    CHECK_HIP(hipStreamWaitEvent(stream, tp_finish_each));
  }

  // pthread_barrier_wait(tp_barrier + reduce_rank + 1);
}

void moe_build_local_ep_data(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, Config *p,
                             pthread_barrier_t *tp_barrier, hipStream_t stream) {
  // GpuTimer timer("moe_local_ep", stream);

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
