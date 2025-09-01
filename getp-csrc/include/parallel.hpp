#pragma once

#include "tensor.hpp"
#include "config.hpp"
#include <cstring>
#include <hip/hip_runtime.h>
#include <vector>

struct CommGroups {
  int local_rank;
  int pp_rank;
  int tp_rank;
  vector<int> pp_group;
  vector<int> tp_group;
};

struct Context {
  vector<int> gpu_ids;
  OurTransformerWeights *weights[REPLICA_SIZE];
  OurRunState *run_state[REPLICA_SIZE];
  CommGroups comm_groups[REPLICA_SIZE];
  hipStream_t streams[REPLICA_SIZE];

  Tensor *pipeline_buffers[PP - 1];
  Tensor *cos_tensor[REPLICA_SIZE];
  Tensor *sin_tensor[REPLICA_SIZE];

  hipEvent_t events[REPLICA_SIZE];  // Used for pipeline only
  hipEvent_t tp_ready_event[REPLICA_SIZE];
  hipEvent_t tp_reduce_done_event[REPLICA_SIZE];
  hipEvent_t pipe_recv_ready_event[REPLICA_SIZE];

  Transformer *transformer;

  void init(Transformer *transformer, const vector<int> &assigned_gpu_ids);
  void destroy();
};

void our_init_weights_120b(Context *ctx, int gpu_id);
void our_init_run_state_120b(Context *ctx, int gpu_id);

float *forward_gpu_120b(Context *ctx, int *tokens, int pos);
