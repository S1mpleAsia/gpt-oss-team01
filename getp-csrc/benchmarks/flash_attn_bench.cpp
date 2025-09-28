#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "config_types_fwd.hpp"
#include "flash_attn_iface.hpp"
#include "../include/tensor.hpp"

using FlashKernelFn = void (*)(Tensor *, Tensor *, Tensor *, Tensor *, Tensor *, Tensor *, Tensor *,
                               Tensor *, Tensor *, int, int, int, int, int, int, int, int,
                               long long, bool, bool, bool, bool, bool, const size_t *,
                               const int *, const uint8_t *, size_t, hipStream_t);

extern "C" void single_query_attn_flash_batched_opt(
  Tensor *q, Tensor *K_cache, Tensor *V_cache, Tensor *mask,
  Tensor *attn_sinks, Tensor *tb, Tensor *g_fa_pmax, Tensor *g_fa_psum, Tensor *g_fa_pnum,
  int cur_batch_size, int head_dim, int num_query_heads, int kv_mul, int kv_dim,
  int seq_len, int sliding_window, int pos, long long layer_offset,
  bool q_to_device, bool k_cache_to_device, bool v_cache_to_device,
  bool mask_to_device, bool tb_from_device,
  const size_t *layer_offsets_tokens, const int *layer_tokens, const uint8_t *layer_is_window,
  size_t kv_batch_stride_tokens, hipStream_t stream) __attribute__((weak));

static void single_query_attn_flash_batched_adapter(
  Tensor *q, Tensor *K_cache, Tensor *V_cache, Tensor *mask,
  Tensor *attn_sinks, Tensor *tb, Tensor *g_fa_pmax, Tensor *g_fa_psum, Tensor *g_fa_pnum,
  int cur_batch_size, int head_dim, int num_query_heads, int kv_mul, int kv_dim,
  int seq_len, int sliding_window, int pos, long long layer_offset,
  bool q_to_device, bool k_cache_to_device, bool v_cache_to_device,
  bool mask_to_device, bool tb_from_device,
  const size_t * /*layer_offsets_tokens*/, const int * /*layer_tokens*/, const uint8_t * /*layer_is_window*/,
  size_t /*kv_batch_stride_tokens*/, hipStream_t stream) {
  (void)g_fa_pmax; (void)g_fa_psum; (void)g_fa_pnum; // unused by baseline
  single_query_attn_flash_batched(q, K_cache, V_cache, mask, attn_sinks, tb,
                                  g_fa_pmax, g_fa_psum, g_fa_pnum,
                                  cur_batch_size, head_dim, num_query_heads, kv_mul, kv_dim,
                                  seq_len, sliding_window, pos, layer_offset,
                                  q_to_device, k_cache_to_device, v_cache_to_device,
                                  mask_to_device, tb_from_device, stream);
}

namespace {
struct BenchConfig {
  int batch = 512;
  int n_q = 64;
  int head_dim = 64;
  int seq_len = 1024;
  int kv_mul = 8;
  int sliding_window = 0;
  int layers = 1;
  int warmup = 5;
  int iters = 100;
  std::string mode = "baseline";  // baseline | workspace | compare
};

constexpr int kFlashTile = 128;

enum class RunMode { Baseline, Workspace, Compare };

void fill_tensor_uniform(Tensor &t, float low, float high) {
  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dist(low, high);
  float *buf = t.buf;
  size_t n = t.num_elem();
  for (size_t i = 0; i < n; ++i) {
    buf[i] = dist(gen);
  }
}

void print_usage(const char *prog) {
  std::cerr << "Usage: " << prog
            << " [--batch N] [--nq N] [--head-dim N] [--seq-len N] [--kv-mul N]"
               " [--sliding-window N] [--layers N] [--warmup N] [--iters N]"
               " [--mode baseline|workspace|compare]\n";
}

BenchConfig parse_args(int argc, char **argv) {
  BenchConfig cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](int &dst) {
      if (i + 1 >= argc)
        throw std::invalid_argument("missing value for " + arg);
      dst = std::stoi(argv[++i]);
    };
    if (arg == "--batch") {
      need_value(cfg.batch);
    } else if (arg == "--nq") {
      need_value(cfg.n_q);
    } else if (arg == "--head-dim") {
      need_value(cfg.head_dim);
    } else if (arg == "--seq-len") {
      need_value(cfg.seq_len);
    } else if (arg == "--kv-mul") {
      need_value(cfg.kv_mul);
    } else if (arg == "--sliding-window") {
      need_value(cfg.sliding_window);
    } else if (arg == "--layers") {
      need_value(cfg.layers);
    } else if (arg == "--warmup") {
      need_value(cfg.warmup);
    } else if (arg == "--iters") {
      need_value(cfg.iters);
    } else if (arg == "--mode") {
      if (i + 1 >= argc)
        throw std::invalid_argument("missing value for --mode");
      cfg.mode = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  return cfg;
}

RunMode resolve_mode(const std::string &mode_str) {
  if (mode_str == "baseline")
    return RunMode::Baseline;
  if (mode_str == "workspace")
    return RunMode::Workspace;
  if (mode_str == "compare")
    return RunMode::Compare;
  throw std::invalid_argument("invalid mode: " + mode_str);
}

double compute_gflops(float avg_ms, int effective_tokens, const BenchConfig &cfg) {
  if (avg_ms <= 0.0f)
    return 0.0;
  const double flops_per_head = 4.0 * static_cast<double>(effective_tokens) * cfg.head_dim;
  const double total_flops = flops_per_head * cfg.n_q * cfg.batch;
  return total_flops / (avg_ms * 1.0e6);
}

double compute_max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
  double max_diff = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff,
                        std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return max_diff;
}

double compute_l2_rel_error(const std::vector<float> &ref, const std::vector<float> &test) {
  long double num = 0.0;
  long double denom = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const long double diff = static_cast<long double>(test[i]) - static_cast<long double>(ref[i]);
    num += diff * diff;
    denom += static_cast<long double>(ref[i]) * static_cast<long double>(ref[i]);
  }
  if (denom == 0.0L)
    return num == 0.0L ? 0.0 : std::numeric_limits<double>::infinity();
  return static_cast<double>(std::sqrt(num / denom));
}

struct RunResult {
  float avg_ms = 0.0f;
  double checksum = 0.0;
  std::vector<float> host_output;
};

RunResult run_kernel(FlashKernelFn kernel, Tensor &q, Tensor &k_cache, Tensor &v_cache,
                     Tensor &attn_sinks, Tensor &tb, Tensor &g_pmax, Tensor &g_psum,
                     Tensor &g_pnum, const BenchConfig &cfg, hipStream_t stream, int kv_dim,
                     int pos) {
  if (!kernel)
    throw std::runtime_error("flash attention kernel pointer is null");

  for (int i = 0; i < cfg.warmup; ++i) {
    kernel(&q, &k_cache, &v_cache, nullptr, &attn_sinks, &tb, &g_pmax, &g_psum, &g_pnum,
           cfg.batch, cfg.head_dim, cfg.n_q, cfg.kv_mul, kv_dim, cfg.seq_len, cfg.sliding_window,
           pos, 0, false, false, false, false, false, nullptr, nullptr, nullptr, 0, stream);
  }
  CHECK_HIP(hipStreamSynchronize(stream));

  hipEvent_t start, stop;
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));

  CHECK_HIP(hipEventRecord(start, stream));
  for (int i = 0; i < cfg.iters; ++i) {
    kernel(&q, &k_cache, &v_cache, nullptr, &attn_sinks, &tb, &g_pmax, &g_psum, &g_pnum,
           cfg.batch, cfg.head_dim, cfg.n_q, cfg.kv_mul, kv_dim, cfg.seq_len, cfg.sliding_window,
           pos, 0, false, false, false, false, false, nullptr, nullptr, nullptr, 0, stream);
  }
  CHECK_HIP(hipEventRecord(stop, stream));
  CHECK_HIP(hipEventSynchronize(stop));

  float total_ms = 0.0f;
  CHECK_HIP(hipEventElapsedTime(&total_ms, start, stop));
  CHECK_HIP(hipEventDestroy(start));
  CHECK_HIP(hipEventDestroy(stop));

  tb.from_device(stream);
  CHECK_HIP(hipStreamSynchronize(stream));

  RunResult result;
  result.avg_ms = total_ms / static_cast<float>(cfg.iters);
  result.host_output.assign(tb.buf, tb.buf + tb.num_elem());
  for (float v : result.host_output)
    result.checksum += static_cast<double>(v);
  return result;
}
}  // namespace

int main(int argc, char **argv) {
  BenchConfig cfg;
  try {
    cfg = parse_args(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  if (cfg.batch <= 0 || cfg.n_q <= 0 || cfg.head_dim <= 0 || cfg.seq_len <= 0 || cfg.kv_mul <= 0 ||
      cfg.layers <= 0 || cfg.warmup < 0 || cfg.iters <= 0) {
    std::cerr << "All sizes must be positive (warmup can be zero).\n";
    return EXIT_FAILURE;
  }
  if (cfg.n_q % cfg.kv_mul != 0) {
    std::cerr << "n_q must be divisible by kv_mul.\n";
    return EXIT_FAILURE;
  }

  RunMode mode;
  try {
    mode = resolve_mode(cfg.mode);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }

  const int kv_heads = cfg.n_q / cfg.kv_mul;
  const int kv_dim = kv_heads * cfg.head_dim;
  const int pos = cfg.seq_len - 1;
  const int attn_len = pos + 1;
  const int effective_tokens = (cfg.sliding_window > 0) ? std::min(attn_len, cfg.sliding_window)
                                                        : attn_len;

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  Tensor q({(size_t)cfg.batch, (size_t)cfg.n_q * (size_t)cfg.head_dim}, stream);
  Tensor k_cache({(size_t)cfg.batch, (size_t)cfg.layers, (size_t)cfg.seq_len, (size_t)kv_dim},
                 stream, DType::BF16);
  Tensor v_cache({(size_t)cfg.batch, (size_t)cfg.layers, (size_t)cfg.seq_len, (size_t)kv_dim},
                 stream, DType::BF16);
  Tensor attn_sinks({(size_t)cfg.layers, (size_t)cfg.n_q}, stream);
  Tensor tb({(size_t)cfg.batch, (size_t)cfg.n_q * (size_t)cfg.head_dim}, stream);

  const size_t c_max = (cfg.seq_len + kFlashTile - 1) / kFlashTile;
  Tensor g_fa_pmax({(size_t)cfg.batch, (size_t)cfg.n_q, c_max}, stream);
  Tensor g_fa_psum({(size_t)cfg.batch, (size_t)cfg.n_q, c_max}, stream);
  Tensor g_fa_pnum({(size_t)cfg.batch, (size_t)cfg.n_q, c_max, (size_t)cfg.head_dim}, stream);

  fill_tensor_uniform(q, -1.0f, 1.0f);
  fill_tensor_uniform(k_cache, -1.0f, 1.0f);
  fill_tensor_uniform(v_cache, -1.0f, 1.0f);
  fill_tensor_uniform(attn_sinks, -1.0f, 1.0f);

  q.to_device(stream);
  k_cache.to_device(stream);
  v_cache.to_device(stream);
  attn_sinks.to_device(stream);
  CHECK_HIP(hipStreamSynchronize(stream));

  FlashKernelFn baseline_kernel = &single_query_attn_flash_batched_adapter;
  FlashKernelFn workspace_kernel = nullptr;
  if (&single_query_attn_flash_batched_opt)
    workspace_kernel = &single_query_attn_flash_batched_opt;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Flash attention decode benchmark (mode=" << cfg.mode << ")\n";
  std::cout << "  batch=" << cfg.batch << ", layers=" << cfg.layers << ", n_q=" << cfg.n_q
            << ", head_dim=" << cfg.head_dim << ", seq_len=" << cfg.seq_len
            << ", kv_mul=" << cfg.kv_mul << ", sliding_window=" << cfg.sliding_window << "\n";

  auto cleanup_stream = [&]() { CHECK_HIP(hipStreamDestroy(stream)); };

  try {
    if (mode == RunMode::Baseline) {
      RunResult baseline = run_kernel(baseline_kernel, q, k_cache, v_cache, attn_sinks, tb,
                                      g_fa_pmax, g_fa_psum, g_fa_pnum, cfg, stream, kv_dim, pos);
      const double gflops = compute_gflops(baseline.avg_ms, effective_tokens, cfg);
      std::cout << "  baseline_avg_ms=" << baseline.avg_ms << ", baseline_gflops=" << gflops
                << ", checksum=" << baseline.checksum << "\n";
    } else if (mode == RunMode::Workspace) {
      if (!workspace_kernel)
        throw std::runtime_error(
          "workspace mode requested but single_query_attn_flash_batched_opt is not linked");
      RunResult workspace = run_kernel(workspace_kernel, q, k_cache, v_cache, attn_sinks, tb,
                                       g_fa_pmax, g_fa_psum, g_fa_pnum, cfg, stream, kv_dim, pos);
      const double gflops = compute_gflops(workspace.avg_ms, effective_tokens, cfg);
      std::cout << "  workspace_avg_ms=" << workspace.avg_ms << ", workspace_gflops=" << gflops
                << ", checksum=" << workspace.checksum << "\n";
    } else {
      if (!workspace_kernel)
        throw std::runtime_error(
          "compare mode requested but single_query_attn_flash_batched_opt is not linked");
      RunResult baseline = run_kernel(baseline_kernel, q, k_cache, v_cache, attn_sinks, tb,
                                      g_fa_pmax, g_fa_psum, g_fa_pnum, cfg, stream, kv_dim, pos);
      RunResult workspace = run_kernel(workspace_kernel, q, k_cache, v_cache, attn_sinks, tb,
                                       g_fa_pmax, g_fa_psum, g_fa_pnum, cfg, stream, kv_dim, pos);

      const double baseline_gflops = compute_gflops(baseline.avg_ms, effective_tokens, cfg);
      const double workspace_gflops = compute_gflops(workspace.avg_ms, effective_tokens, cfg);
      const double speedup = (workspace.avg_ms > 0.0f)
                               ? static_cast<double>(baseline.avg_ms) / workspace.avg_ms
                               : std::numeric_limits<double>::quiet_NaN();
      const double max_diff =
        compute_max_abs_diff(baseline.host_output, workspace.host_output);
      const double rel_l2 = compute_l2_rel_error(baseline.host_output, workspace.host_output);

      std::cout << "  baseline_avg_ms=" << baseline.avg_ms
                << ", baseline_gflops=" << baseline_gflops
                << ", baseline_checksum=" << baseline.checksum << "\n";
      std::cout << "  workspace_avg_ms=" << workspace.avg_ms
                << ", workspace_gflops=" << workspace_gflops
                << ", workspace_checksum=" << workspace.checksum << "\n";
      std::cout << "  speedup=" << speedup << ", max_abs_diff=" << max_diff
                << ", l2_rel_error=" << rel_l2 << "\n";
    }
  } catch (const std::exception &e) {
    cleanup_stream();
    std::cerr << "Benchmark error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  cleanup_stream();
  return EXIT_SUCCESS;
}
