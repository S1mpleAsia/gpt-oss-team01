<div align="center">

# GPT-OSS from Scratch on AMD GPUs

</div>

## Overview

This a pure C++ implementation of OpenAI’s GPT-OSS models designed to **maximize inference throughput on AMD GPUs without relying on external libraries**. Our goal is to explore end-to-end LLM optimization, from kernel-level improvements to system-level design, providing insights for researchers and developers interested in high-performance computing and model-level optimization.

Inspired by [llama2.c](https://github.com/karpathy/llama2.c), our implementation uses HIP (an AMD programming model equivalent to CUDA) and avoids dependencies such as rocBLAS, hipBLAS, RCCL, and MPI. We utilize multiple optimization strategies for the 20B and 120B models, including efficient model loading, batching, multi-streaming, multi-GPU communication, optimized CPU–GPU–SRAM memory access, FlashAttention, matrix-core–based GEMM, and load balancing for MoE routing.

Experiments on a single node with 8× AMD MI250 GPUs show that our implementation achieves about 40k TPS on the 20B model and around 13.6k TPS on the 120B model in custom benchmarks, demonstrating the effectiveness of our optimizations and the strong potential of AMD GPUs for large-scale LLM inference.

---

## Experiments

| Model          | Throughput (TPS) | METEOR | BERTScore |
| -------------- | ---------------- | ------ | --------- |
| `gpt-oss-20b`  | 39120            | 0.48   | 0.97      |
| `gpt-oss-120b` | 13663            | 0.34   | 0.97      |

---

## Acknowledgments

This project was part of the GPU Engineer Training Program, a collaboration between [Moreh](https://www.linkedin.com/company/moreh-inc) and [THUNDER Research Group](http://snuvm.snu.ac.kr/) (Seoul National University).