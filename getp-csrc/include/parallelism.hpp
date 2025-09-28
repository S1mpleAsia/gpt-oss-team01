#pragma once

#include "config.hpp"
#include "layer_hip_batch.hpp"
#include <hip/hip_runtime.h>

void all_gather_x(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                  int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                  hipEvent_t tp_ready, hipEvent_t tp_finish);

void all_gather_x_new(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                      int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                      hipEvent_t tp_ready, hipEvent_t tp_finish);

void all_gather_x_full_tp(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                          int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                          hipEvent_t tp_ready, hipEvent_t tp_finish);

void reduce_tb3(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                hipEvent_t tp_ready, hipEvent_t tp_finish);

void reduce_tb3_new(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                    int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                    hipEvent_t tp_ready, hipEvent_t tp_finish);

void ring_all_reduce_tb3(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                         int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream);

void reduce_tb2(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                hipEvent_t tp_ready, hipEvent_t tp_finish);

void reduce_tb2_full_tp(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                        int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                        hipEvent_t tp_ready, hipEvent_t tp_finish);

void reduce_tb2_new(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                    int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                    hipEvent_t tp_ready, hipEvent_t tp_finish);

void ring_all_reduce_tb2(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                         int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream);

void all_gather_classifier_v2(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank,
                              int cur_device, int cur_batch_size, pthread_barrier_t *tp_barrier,
                              hipStream_t stream, hipEvent_t tp_ready, hipEvent_t tp_finish);

void all_gather_classifier_final(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank,
                                 int cur_device, int cur_batch_size, pthread_barrier_t *tp_barrier,
                                 hipStream_t stream, hipEvent_t tp_ready, hipEvent_t tp_finish);

void all_gather_logits_id(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                          int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                          hipEvent_t tp_ready, hipEvent_t tp_finish);

void all_gather_qkv_v2(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                       int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                       hipEvent_t tp_ready, hipEvent_t tp_finish);

void all_gather_tb3(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                    Config *p, pthread_barrier_t *tp_barrier, hipStream_t stream,
                    hipEvent_t tp_ready, hipEvent_t tp_finish);

void ring_reduce_agg(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                     int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream);

void reduce_agg_full_tp(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                        int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                        hipEvent_t tp_ready, hipEvent_t tp_finish);

void reduce_agg(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, int cur_device,
                int cur_batch_size, pthread_barrier_t *tp_barrier, hipStream_t stream,
                hipEvent_t tp_ready, hipEvent_t tp_finish);

void moe_build_local_ep_data(OurRunState *rs_now, OurRunState *rs_leader, int tp_rank, Config *p,
                             pthread_barrier_t *tp_barrier, hipStream_t stream);
