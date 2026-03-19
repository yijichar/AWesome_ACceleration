# Lightweight LLM Inference Engine for Qwen3

A lightweight large language model inference engine built with PyTorch, designed for Qwen3-style decoder-only models.  
This project focuses on efficient **KV cache management**, **continuous batching**, **FlashAttention integration**, and **multi-GPU inference parallelism**, with unified benchmarking against **HuggingFace Transformers** and **vLLM-style serving systems**.

## Highlights

- **Asynchronous inference pipeline** with request queue and background scheduling
- **Slot-based KV cache management** for efficient sequence lifecycle tracking
- **Continuous batching** for mixed prefill/decode scheduling
- **FlashAttention integration** in both prefill and decode paths
- **Fused QKV projection** to reduce kernel launch and memory access overhead
- **Paged KV cache** for dynamic sequence growth without repeated reallocation
- **Tensor Parallel (TP)** inference for multi-GPU model execution
- **Data Parallel (DP)** and **hybrid DP×TP** support for higher-throughput benchmarking
- **Two-level parallel design** with **per-replica scheduling + intra-replica tensor parallelism**
- **Profiling support** for communication overhead such as broadcast, all-reduce, and all-gather
- **Packed decode-stage broadcast optimization** to reduce control-path communication overhead

## Performance

On **4× RTX 4090 GPUs**, we benchmarked **Qwen3-0.6B** with **256 sequences**, **1024 input tokens**, and **1024 output tokens**.
- **Single GPU:** **6846 tok/s**
- **Tensor Parallel (2 GPUs):** **11321 tok/s**
- **Data Parallel (2 GPUs):** **12209 tok/s**
- **DP×TP (4 GPUs, 2-way DP × 2-way TP):** **21687 tok/s**


---

## Project Structure

```text
├── benchmark_tp.py          # Benchmark entry for single-GPU / TP / DP×TP evaluation
├── llm.py                   # Single-GPU inference engine
├── llm_tp.py                # TP / DP×TP inference engine
├── model/
│   ├── distributed.py       # Distributed runtime, TP/DP groups, communication wrappers
│   ├── profiler_tp.py       # Lightweight profiler for communication / stage timing
│   ├── parallel_layers.py   # ColumnParallel / RowParallel / QKV parallel layers
│   ├── model.py             # Single-GPU Qwen-style model
│   └── model_tp.py          # Tensor-parallel Qwen-style model
└── README.md
```

## TODO

- [ ] Extend the current PyTorch inference engine to support **Qwen3-30B-A3B MoE** models.
