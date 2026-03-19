#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark: Custom Qwen3 inference engine vs HuggingFace Transformers (fair-ish offline fixed-length)

Goals
-----
- Same model path / tokenizer / chat template
- Same prompt set
- Fixed output length (ignore EOS as much as possible)
- Report:
  * end-to-end batch time
  * total tok/s = (sum(prompt_tokens) + sum(gen_tokens)) / elapsed
  * decode tok/s = sum(gen_tokens) / elapsed
  * TTFT (optional; not available for custom engine without instrumentation)

Usage
-----
python benchmark_compare.py \
  --model-path /mnt/data0/Qwen30.6B \
  --num-prompts 64 \
  --input-tokens 512 \
  --output-tokens 128 \
  --dtype bf16 \
  --device cuda

Notes
-----
1) Your custom engine currently applies chat template inside generate(). This script generates plain user strings
   and lets BOTH sides apply the same chat template.
2) "Fixed output length" on HF is approximated with min_new_tokens=max_new_tokens and eos_token_id=None.
   Some models/configs may still stop early if internal stopping kicks in; script detects actual lengths.
3) TTFT for custom engine is shown as N/A unless you instrument per-request first-token callbacks.
"""

import os
import time
import json
import math
import random
import argparse
import statistics
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Import your custom engine ===
# Adjust path if needed
# e.g., sys.path.insert(0, "/home/zyf/workspace/Qwen-vllm copy")
from llm import LLM as CustomLLM  # your custom engine class


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dtype(s: str):
    s = s.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class BenchResult:
    name: str
    num_prompts: int
    sum_prompt_tokens: int
    sum_generated_tokens: int
    elapsed_s: float
    ttft_s_mean: Optional[float] = None
    ttft_s_p50: Optional[float] = None
    ttft_s_p95: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None

    @property
    def total_tokens(self):
        return self.sum_prompt_tokens + self.sum_generated_tokens

    @property
    def total_tok_per_s(self):
        return self.total_tokens / self.elapsed_s if self.elapsed_s > 0 else float("nan")

    @property
    def decode_tok_per_s(self):
        return self.sum_generated_tokens / self.elapsed_s if self.elapsed_s > 0 else float("nan")


def print_result(r: BenchResult):
    print(f"\n=== {r.name} ===")
    print(f"prompts           : {r.num_prompts}")
    print(f"prompt tokens sum : {r.sum_prompt_tokens}")
    print(f"gen tokens sum    : {r.sum_generated_tokens}")
    print(f"E2E batch time    : {r.elapsed_s:.4f} s")
    print(f"total tok/s       : {r.total_tok_per_s:.2f}")
    print(f"decode tok/s      : {r.decode_tok_per_s:.2f}")
    if r.ttft_s_mean is None:
        print(f"TTFT              : N/A (not instrumented)")
    else:
        print(f"TTFT mean / p50 / p95 : {r.ttft_s_mean:.4f} / {r.ttft_s_p50:.4f} / {r.ttft_s_p95:.4f} s")
    if r.extra:
        for k, v in r.extra.items():
            print(f"{k:18}: {v}")


# -----------------------------
# Prompt generation (target token length under chat template)
# -----------------------------

def chat_token_len(tokenizer, user_text: str) -> int:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids)


def build_prompt_near_target_tokens(tokenizer, target_tokens: int, idx: int) -> str:
    """
    Build a deterministic prompt whose tokenized length (after chat template) is near target_tokens.
    Uses repeated simple segments to avoid weird Unicode/tokenization artifacts.
    """
    # Vary content slightly across prompts to avoid exact duplicates.
    prefix = f"Request #{idx}. Please answer concisely. "
    # Quick path if prefix already exceeds target (rare for small target)
    if chat_token_len(tokenizer, prefix) >= target_tokens:
        return prefix

    unit_pool = [
        "alpha beta gamma delta ",
        "one two three four five ",
        "red blue green yellow ",
        "data model token cache ",
        "hello world benchmark test ",
    ]
    # Grow content until reaching/exceeding target, then trim by binary search on repetitions.
    unit = unit_pool[idx % len(unit_pool)]

    # Exponential search
    lo, hi = 0, 1
    while True:
        s = prefix + unit * hi
        n = chat_token_len(tokenizer, s)
        if n >= target_tokens:
            break
        lo = hi
        hi *= 2
        if hi > 1_000_000:
            break

    # Binary search best repetitions
    best = prefix + unit * lo
    best_gap = abs(chat_token_len(tokenizer, best) - target_tokens)

    l, r = lo, hi
    while l <= r:
        m = (l + r) // 2
        s = prefix + unit * m
        n = chat_token_len(tokenizer, s)
        gap = abs(n - target_tokens)
        if gap < best_gap:
            best, best_gap = s, gap
        if n < target_tokens:
            l = m + 1
        elif n > target_tokens:
            r = m - 1
        else:
            return s

    return best


def build_prompt_batch(tokenizer, num_prompts: int, target_input_tokens: int) -> List[str]:
    prompts = []
    lengths = []
    for i in range(num_prompts):
        p = build_prompt_near_target_tokens(tokenizer, target_input_tokens, i)
        prompts.append(p)
        lengths.append(chat_token_len(tokenizer, p))
    print(f"[Prompt builder] target={target_input_tokens}, actual min/mean/max="
          f"{min(lengths)}/{sum(lengths)/len(lengths):.1f}/{max(lengths)}")
    prompts = [
    # 1-16: 超短回答 / 单句任务（更容易提前结束）
    "What is 1+1? Output only the number.",
    "Name one color.",
    "Say hello in Chinese.",
    "What is the capital of France? One short answer.",
    "Is Python a programming language? Answer yes or no only.",
    "Translate 'good morning' to Chinese.",
    "What day comes after Monday?",
    "Which is bigger: 9 or 12? Output only the number.",
    "Give one synonym of 'happy'.",
    "What is the opposite of 'hot'?",
    "How many days are in a week? Output only the number.",
    "Translate '谢谢' into English.",
    "Name one fruit.",
    "What is 10 minus 3? Output only the number.",
    "Is water wet? One short sentence.",
    "Write the Chinese word for 'cat'.",

    # 17-32: 短列表 / 简单解释（中等长度）
    "List three primary colors.",
    "Give three tips for staying focused while studying.",
    "Explain what machine learning is in two simple sentences.",
    "Translate this to Chinese: 'I am learning artificial intelligence.'",
    "Write three bullet points about the benefits of exercise.",
    "What is overfitting in machine learning? Explain simply.",
    "Summarize the purpose of a GPU in one paragraph.",
    "List 5 common Python data types.",
    "What is the difference between RAM and storage? Keep it short.",
    "Explain what an API is for a beginner.",
    "Write a short reply to: 'Thanks for your help!'",
    "Give 4 interview tips for a software engineering intern.",
    "Explain the meaning of 'continuous batching' in simple words.",
    "What is a transformer model? Answer for a college student.",
    "List three use cases of diffusion models.",
    "Translate to English: '我正在做一个推理加速项目。'",

    # 33-48: 中等复杂推理 / 结构化输出（长度差异会更明显）
    "Compare CNN and Transformer in 4 bullet points.",
    "Explain the difference between throughput and latency with examples.",
    "Write a concise definition of KV cache and why it helps inference.",
    "If a model generates 128 tokens in 0.5 seconds, what is decode throughput? Show calculation.",
    "Describe the advantages of mixed precision inference in practical deployment.",
    "Give a step-by-step plan to benchmark two inference engines fairly.",
    "Write a short email asking a professor for a meeting next week.",
    "Summarize the key idea of greedy decoding vs sampling.",
    "Explain what causes CUDA out-of-memory errors and how to debug them.",
    "List 5 factors that affect LLM inference speed on a single GPU.",
    "Describe how padding side (left vs right) can affect decoder-only generation.",
    "Write a polite response declining an invitation because of a deadline.",
    "Explain why fixed-length output may hide the benefits of continuous batching.",
    "Give a short checklist for reproducing benchmark results.",
    "Explain the role of RoPE in transformer attention.",
    "What is QK norm and why might Qwen3 use it?",

    # 49-64: 开放式写作 / 更长输出倾向（不一定早停）
    "Write a short introduction paragraph for a report comparing custom inference engines and HuggingFace Transformers.",
    "Explain continuous batching as if teaching a first-year graduate student in AI systems.",
    "Write a mini tutorial on how to profile a PyTorch inference pipeline step by step.",
    "Describe a fair benchmark protocol for comparing offline throughput between two LLM inference frameworks.",
    "Write a short Chinese explanation of why 'correctness first, optimization second' is important in systems engineering.",
    "Give a practical debugging strategy for when a custom LLM engine outputs repeated nonsense tokens.",
    "Write a concise note on why matching prefill and decode paths is essential in custom KV-cache implementations.",
    "Explain how FlashAttention helps attention efficiency without going too deep into math.",
    "Write a short comparison between static batching and continuous batching in online serving.",
    "Describe common sources of unfairness when comparing vLLM, HF generate, and a custom engine.",
    "Write a brief project summary for a resume about building a slot-based KV cache inference engine.",
    "Explain how EOS behavior can affect throughput benchmarks and result interpretation.",
    "Write a short recommendation section for improving a custom inference engine after correctness is achieved.",
    "Describe what metrics (TTFT, throughput, latency) should be tracked in LLM serving evaluation.",
    "Write a short Chinese summary of the difference between total tok/s and decode tok/s.",
    "Explain how heterogeneous prompt lengths can reveal the benefits of continuous scheduling.","What is 1+1? Output only the number.",
    "Name one color.",
    "Say hello in Chinese.",
    "What is the capital of France? One short answer.",
    "Is Python a programming language? Answer yes or no only.",
    "Translate 'good morning' to Chinese.",
    "What day comes after Monday?",
    "Which is bigger: 9 or 12? Output only the number.",
    "Give one synonym of 'happy'.",
    "What is the opposite of 'hot'?",
    "How many days are in a week? Output only the number.",
    "Translate '谢谢' into English.",
    "Name one fruit.",
    "What is 10 minus 3? Output only the number.",
    "Is water wet? One short sentence.",
    "Write the Chinese word for 'cat'.",

    # 17-32: 短列表 / 简单解释（中等长度）
    "List three primary colors.",
    "Give three tips for staying focused while studying.",
    "Explain what machine learning is in two simple sentences.",
    "Translate this to Chinese: 'I am learning artificial intelligence.'",
    "Write three bullet points about the benefits of exercise.",
    "What is overfitting in machine learning? Explain simply.",
    "Summarize the purpose of a GPU in one paragraph.",
    "List 5 common Python data types.",
    "What is the difference between RAM and storage? Keep it short.",
    "Explain what an API is for a beginner.",
    "Write a short reply to: 'Thanks for your help!'",
    "Give 4 interview tips for a software engineering intern.",
    "Explain the meaning of 'continuous batching' in simple words.",
    "What is a transformer model? Answer for a college student.",
    "List three use cases of diffusion models.",
    "Translate to English: '我正在做一个推理加速项目。'",

    # 33-48: 中等复杂推理 / 结构化输出（长度差异会更明显）
    "Compare CNN and Transformer in 4 bullet points.",
    "Explain the difference between throughput and latency with examples.",
    "Write a concise definition of KV cache and why it helps inference.",
    "If a model generates 128 tokens in 0.5 seconds, what is decode throughput? Show calculation.",
    "Describe the advantages of mixed precision inference in practical deployment.",
    "Give a step-by-step plan to benchmark two inference engines fairly.",
    "Write a short email asking a professor for a meeting next week.",
    "Summarize the key idea of greedy decoding vs sampling.",
    "Explain what causes CUDA out-of-memory errors and how to debug them.",
    "List 5 factors that affect LLM inference speed on a single GPU.",
    "Describe how padding side (left vs right) can affect decoder-only generation.",
    "Write a polite response declining an invitation because of a deadline.",
    "Explain why fixed-length output may hide the benefits of continuous batching.",
    "Give a short checklist for reproducing benchmark results.",
    "Explain the role of RoPE in transformer attention.",
    "What is QK norm and why might Qwen3 use it?",

    # 49-64: 开放式写作 / 更长输出倾向（不一定早停）
    "Write a short introduction paragraph for a report comparing custom inference engines and HuggingFace Transformers.",
    "Explain continuous batching as if teaching a first-year graduate student in AI systems.",
    "Write a mini tutorial on how to profile a PyTorch inference pipeline step by step.",
    "Describe a fair benchmark protocol for comparing offline throughput between two LLM inference frameworks.",
    "Write a short Chinese explanation of why 'correctness first, optimization second' is important in systems engineering.",
    "Give a practical debugging strategy for when a custom LLM engine outputs repeated nonsense tokens.",
    "Write a concise note on why matching prefill and decode paths is essential in custom KV-cache implementations.",
    "Explain how FlashAttention helps attention efficiency without going too deep into math.",
    "Write a short comparison between static batching and continuous batching in online serving.",
    "Describe common sources of unfairness when comparing vLLM, HF generate, and a custom engine.",
    "Write a brief project summary for a resume about building a slot-based KV cache inference engine.",
    "Explain how EOS behavior can affect throughput benchmarks and result interpretation.",
    "Write a short recommendation section for improving a custom inference engine after correctness is achieved.",
    "Describe what metrics (TTFT, throughput, latency) should be tracked in LLM serving evaluation.",
    "Write a short Chinese summary of the difference between total tok/s and decode tok/s.",
    "Explain how heterogeneous prompt lengths can reveal the benefits of continuous scheduling.","What is 1+1? Output only the number.",
    "Name one color.",
    "Say hello in Chinese.",
    "What is the capital of France? One short answer.",
    "Is Python a programming language? Answer yes or no only.",
    "Translate 'good morning' to Chinese.",
    "What day comes after Monday?",
    "Which is bigger: 9 or 12? Output only the number.",
    "Give one synonym of 'happy'.",
    "What is the opposite of 'hot'?",
    "How many days are in a week? Output only the number.",
    "Translate '谢谢' into English.",
    "Name one fruit.",
    "What is 10 minus 3? Output only the number.",
    "Is water wet? One short sentence.",
    "Write the Chinese word for 'cat'.",

    # 17-32: 短列表 / 简单解释（中等长度）
    "List three primary colors.",
    "Give three tips for staying focused while studying.",
    "Explain what machine learning is in two simple sentences.",
    "Translate this to Chinese: 'I am learning artificial intelligence.'",
    "Write three bullet points about the benefits of exercise.",
    "What is overfitting in machine learning? Explain simply.",
    "Summarize the purpose of a GPU in one paragraph.",
    "List 5 common Python data types.",
    "What is the difference between RAM and storage? Keep it short.",
    "Explain what an API is for a beginner.",
    "Write a short reply to: 'Thanks for your help!'",
    "Give 4 interview tips for a software engineering intern.",
    "Explain the meaning of 'continuous batching' in simple words.",
    "What is a transformer model? Answer for a college student.",
    "List three use cases of diffusion models.",
    "Translate to English: '我正在做一个推理加速项目。'",

    # 33-48: 中等复杂推理 / 结构化输出（长度差异会更明显）
    "Compare CNN and Transformer in 4 bullet points.",
    "Explain the difference between throughput and latency with examples.",
    "Write a concise definition of KV cache and why it helps inference.",
    "If a model generates 128 tokens in 0.5 seconds, what is decode throughput? Show calculation.",
    "Describe the advantages of mixed precision inference in practical deployment.",
    "Give a step-by-step plan to benchmark two inference engines fairly.",
    "Write a short email asking a professor for a meeting next week.",
    "Summarize the key idea of greedy decoding vs sampling.",
    "Explain what causes CUDA out-of-memory errors and how to debug them.",
    "List 5 factors that affect LLM inference speed on a single GPU.",
    "Describe how padding side (left vs right) can affect decoder-only generation.",
    "Write a polite response declining an invitation because of a deadline.",
    "Explain why fixed-length output may hide the benefits of continuous batching.",
    "Give a short checklist for reproducing benchmark results.",
    "Explain the role of RoPE in transformer attention.",
    "What is QK norm and why might Qwen3 use it?",

    # 49-64: 开放式写作 / 更长输出倾向（不一定早停）
    "Write a short introduction paragraph for a report comparing custom inference engines and HuggingFace Transformers.",
    "Explain continuous batching as if teaching a first-year graduate student in AI systems.",
    "Write a mini tutorial on how to profile a PyTorch inference pipeline step by step.",
    "Describe a fair benchmark protocol for comparing offline throughput between two LLM inference frameworks.",
    "Write a short Chinese explanation of why 'correctness first, optimization second' is important in systems engineering.",
    "Give a practical debugging strategy for when a custom LLM engine outputs repeated nonsense tokens.",
    "Write a concise note on why matching prefill and decode paths is essential in custom KV-cache implementations.",
    "Explain how FlashAttention helps attention efficiency without going too deep into math.",
    "Write a short comparison between static batching and continuous batching in online serving.",
    "Describe common sources of unfairness when comparing vLLM, HF generate, and a custom engine.",
    "Write a brief project summary for a resume about building a slot-based KV cache inference engine.",
    "Explain how EOS behavior can affect throughput benchmarks and result interpretation.",
    "Write a short recommendation section for improving a custom inference engine after correctness is achieved.",
    "Describe what metrics (TTFT, throughput, latency) should be tracked in LLM serving evaluation.",
    "Write a short Chinese summary of the difference between total tok/s and decode tok/s.",
    "Explain how heterogeneous prompt lengths can reveal the benefits of continuous scheduling.","What is 1+1? Output only the number.",
    "Name one color.",
    "Say hello in Chinese.",
    "What is the capital of France? One short answer.",
    "Is Python a programming language? Answer yes or no only.",
    "Translate 'good morning' to Chinese.",
    "What day comes after Monday?",
    "Which is bigger: 9 or 12? Output only the number.",
    "Give one synonym of 'happy'.",
    "What is the opposite of 'hot'?",
    "How many days are in a week? Output only the number.",
    "Translate '谢谢' into English.",
    "Name one fruit.",
    "What is 10 minus 3? Output only the number.",
    "Is water wet? One short sentence.",
    "Write the Chinese word for 'cat'.",

    # 17-32: 短列表 / 简单解释（中等长度）
    "List three primary colors.",
    "Give three tips for staying focused while studying.",
    "Explain what machine learning is in two simple sentences.",
    "Translate this to Chinese: 'I am learning artificial intelligence.'",
    "Write three bullet points about the benefits of exercise.",
    "What is overfitting in machine learning? Explain simply.",
    "Summarize the purpose of a GPU in one paragraph.",
    "List 5 common Python data types.",
    "What is the difference between RAM and storage? Keep it short.",
    "Explain what an API is for a beginner.",
    "Write a short reply to: 'Thanks for your help!'",
    "Give 4 interview tips for a software engineering intern.",
    "Explain the meaning of 'continuous batching' in simple words.",
    "What is a transformer model? Answer for a college student.",
    "List three use cases of diffusion models.",
    "Translate to English: '我正在做一个推理加速项目。'",

    # 33-48: 中等复杂推理 / 结构化输出（长度差异会更明显）
    "Compare CNN and Transformer in 4 bullet points.",
    "Explain the difference between throughput and latency with examples.",
    "Write a concise definition of KV cache and why it helps inference.",
    "If a model generates 128 tokens in 0.5 seconds, what is decode throughput? Show calculation.",
    "Describe the advantages of mixed precision inference in practical deployment.",
    "Give a step-by-step plan to benchmark two inference engines fairly.",
    "Write a short email asking a professor for a meeting next week.",
    "Summarize the key idea of greedy decoding vs sampling.",
    "Explain what causes CUDA out-of-memory errors and how to debug them.",
    "List 5 factors that affect LLM inference speed on a single GPU.",
    "Describe how padding side (left vs right) can affect decoder-only generation.",
    "Write a polite response declining an invitation because of a deadline.",
    "Explain why fixed-length output may hide the benefits of continuous batching.",
    "Give a short checklist for reproducing benchmark results.",
    "Explain the role of RoPE in transformer attention.",
    "What is QK norm and why might Qwen3 use it?",

    # 49-64: 开放式写作 / 更长输出倾向（不一定早停）
    "Write a short introduction paragraph for a report comparing custom inference engines and HuggingFace Transformers.",
    "Explain continuous batching as if teaching a first-year graduate student in AI systems.",
    "Write a mini tutorial on how to profile a PyTorch inference pipeline step by step.",
    "Describe a fair benchmark protocol for comparing offline throughput between two LLM inference frameworks.",
    "Write a short Chinese explanation of why 'correctness first, optimization second' is important in systems engineering.",
    "Give a practical debugging strategy for when a custom LLM engine outputs repeated nonsense tokens.",
    "Write a concise note on why matching prefill and decode paths is essential in custom KV-cache implementations.",
    "Explain how FlashAttention helps attention efficiency without going too deep into math.",
    "Write a short comparison between static batching and continuous batching in online serving.",
    "Describe common sources of unfairness when comparing vLLM, HF generate, and a custom engine.",
    "Write a brief project summary for a resume about building a slot-based KV cache inference engine.",
    "Explain how EOS behavior can affect throughput benchmarks and result interpretation.",
    "Write a short recommendation section for improving a custom inference engine after correctness is achieved.",
    "Describe what metrics (TTFT, throughput, latency) should be tracked in LLM serving evaluation.",
    "Write a short Chinese summary of the difference between total tok/s and decode tok/s.",
    "Explain how heterogeneous prompt lengths can reveal the benefits of continuous scheduling.",
    "What is 1+1? Output only the number.",
    "Name one color.",
    "Say hello in Chinese.",
    "What is the capital of France? One short answer.",
    "Is Python a programming language? Answer yes or no only.",
    "Translate 'good morning' to Chinese.",
    "What day comes after Monday?",
    "Which is bigger: 9 or 12? Output only the number.",
    "Give one synonym of 'happy'.",
    "What is the opposite of 'hot'?",
    "How many days are in a week? Output only the number.",
    "Translate '谢谢' into English.",
    "Name one fruit.",
    "What is 10 minus 3? Output only the number.",
    "Is water wet? One short sentence.",
    "Write the Chinese word for 'cat'.",

    # 17-32: 短列表 / 简单解释（中等长度）
    "List three primary colors.",
    "Give three tips for staying focused while studying.",
    "Explain what machine learning is in two simple sentences.",
    "Translate this to Chinese: 'I am learning artificial intelligence.'",
    "Write three bullet points about the benefits of exercise.",
    "What is overfitting in machine learning? Explain simply.",
    "Summarize the purpose of a GPU in one paragraph.",
    "List 5 common Python data types.",
    "What is the difference between RAM and storage? Keep it short.",
    "Explain what an API is for a beginner.",
    "Write a short reply to: 'Thanks for your help!'",
    "Give 4 interview tips for a software engineering intern.",
    "Explain the meaning of 'continuous batching' in simple words.",
    "What is a transformer model? Answer for a college student.",
    "List three use cases of diffusion models.",
    "Translate to English: '我正在做一个推理加速项目。'",

    # 33-48: 中等复杂推理 / 结构化输出（长度差异会更明显）
    "Compare CNN and Transformer in 4 bullet points.",
    "Explain the difference between throughput and latency with examples.",
    "Write a concise definition of KV cache and why it helps inference.",
    "If a model generates 128 tokens in 0.5 seconds, what is decode throughput? Show calculation.",
    "Describe the advantages of mixed precision inference in practical deployment.",
    "Give a step-by-step plan to benchmark two inference engines fairly.",
    "Write a short email asking a professor for a meeting next week.",
    "Summarize the key idea of greedy decoding vs sampling.",
    "Explain what causes CUDA out-of-memory errors and how to debug them.",
    "List 5 factors that affect LLM inference speed on a single GPU.",
    "Describe how padding side (left vs right) can affect decoder-only generation.",
    "Write a polite response declining an invitation because of a deadline.",
    "Explain why fixed-length output may hide the benefits of continuous batching.",
    "Give a short checklist for reproducing benchmark results.",
    "Explain the role of RoPE in transformer attention.",
    "What is QK norm and why might Qwen3 use it?",

    # 49-64: 开放式写作 / 更长输出倾向（不一定早停）
    "Write a short introduction paragraph for a report comparing custom inference engines and HuggingFace Transformers.",
    "Explain continuous batching as if teaching a first-year graduate student in AI systems.",
    "Write a mini tutorial on how to profile a PyTorch inference pipeline step by step.",
    "Describe a fair benchmark protocol for comparing offline throughput between two LLM inference frameworks.",
    "Write a short Chinese explanation of why 'correctness first, optimization second' is important in systems engineering.",
    "Give a practical debugging strategy for when a custom LLM engine outputs repeated nonsense tokens.",
    "Write a concise note on why matching prefill and decode paths is essential in custom KV-cache implementations.",
    "Explain how FlashAttention helps attention efficiency without going too deep into math.",
    "Write a short comparison between static batching and continuous batching in online serving.",
    "Describe common sources of unfairness when comparing vLLM, HF generate, and a custom engine.",
    "Write a brief project summary for a resume about building a slot-based KV cache inference engine.",
    "Explain how EOS behavior can affect throughput benchmarks and result interpretation.",
    "Write a short recommendation section for improving a custom inference engine after correctness is achieved.",
    "Describe what metrics (TTFT, throughput, latency) should be tracked in LLM serving evaluation.",
    "Write a short Chinese summary of the difference between total tok/s and decode tok/s.",
    "Explain how heterogeneous prompt lengths can reveal the benefits of continuous scheduling.","What is 1+1? Output only the number.",
    "Name one color.",
    "Say hello in Chinese.",
    "What is the capital of France? One short answer.",
    "Is Python a programming language? Answer yes or no only.",
    "Translate 'good morning' to Chinese.",
    "What day comes after Monday?",
    "Which is bigger: 9 or 12? Output only the number.",
    "Give one synonym of 'happy'.",
    "What is the opposite of 'hot'?",
    "How many days are in a week? Output only the number.",
    "Translate '谢谢' into English.",
    "Name one fruit.",
    "What is 10 minus 3? Output only the number.",
    "Is water wet? One short sentence.",
    "Write the Chinese word for 'cat'.",

    # 17-32: 短列表 / 简单解释（中等长度）
    "List three primary colors.",
    "Give three tips for staying focused while studying.",
    "Explain what machine learning is in two simple sentences.",
    "Translate this to Chinese: 'I am learning artificial intelligence.'",
    "Write three bullet points about the benefits of exercise.",
    "What is overfitting in machine learning? Explain simply.",
    "Summarize the purpose of a GPU in one paragraph.",
    "List 5 common Python data types.",
    "What is the difference between RAM and storage? Keep it short.",
    "Explain what an API is for a beginner.",
    "Write a short reply to: 'Thanks for your help!'",
    "Give 4 interview tips for a software engineering intern.",
    "Explain the meaning of 'continuous batching' in simple words.",
    "What is a transformer model? Answer for a college student.",
    "List three use cases of diffusion models.",
    "Translate to English: '我正在做一个推理加速项目。'",

    # 33-48: 中等复杂推理 / 结构化输出（长度差异会更明显）
    "Compare CNN and Transformer in 4 bullet points.",
    "Explain the difference between throughput and latency with examples.",
    "Write a concise definition of KV cache and why it helps inference.",
    "If a model generates 128 tokens in 0.5 seconds, what is decode throughput? Show calculation.",
    "Describe the advantages of mixed precision inference in practical deployment.",
    "Give a step-by-step plan to benchmark two inference engines fairly.",
    "Write a short email asking a professor for a meeting next week.",
    "Summarize the key idea of greedy decoding vs sampling.",
    "Explain what causes CUDA out-of-memory errors and how to debug them.",
    "List 5 factors that affect LLM inference speed on a single GPU.",
    "Describe how padding side (left vs right) can affect decoder-only generation.",
    "Write a polite response declining an invitation because of a deadline.",
    "Explain why fixed-length output may hide the benefits of continuous batching.",
    "Give a short checklist for reproducing benchmark results.",
    "Explain the role of RoPE in transformer attention.",
    "What is QK norm and why might Qwen3 use it?",

    # 49-64: 开放式写作 / 更长输出倾向（不一定早停）
    "Write a short introduction paragraph for a report comparing custom inference engines and HuggingFace Transformers.",
    "Explain continuous batching as if teaching a first-year graduate student in AI systems.",
    "Write a mini tutorial on how to profile a PyTorch inference pipeline step by step.",
    "Describe a fair benchmark protocol for comparing offline throughput between two LLM inference frameworks.",
    "Write a short Chinese explanation of why 'correctness first, optimization second' is important in systems engineering.",
    "Give a practical debugging strategy for when a custom LLM engine outputs repeated nonsense tokens.",
    "Write a concise note on why matching prefill and decode paths is essential in custom KV-cache implementations.",
    "Explain how FlashAttention helps attention efficiency without going too deep into math.",
    "Write a short comparison between static batching and continuous batching in online serving.",
    "Describe common sources of unfairness when comparing vLLM, HF generate, and a custom engine.",
    "Write a brief project summary for a resume about building a slot-based KV cache inference engine.",
    "Explain how EOS behavior can affect throughput benchmarks and result interpretation.",
    "Write a short recommendation section for improving a custom inference engine after correctness is achieved.",
    "Describe what metrics (TTFT, throughput, latency) should be tracked in LLM serving evaluation.",
    "Write a short Chinese summary of the difference between total tok/s and decode tok/s.",
    "Explain how heterogeneous prompt lengths can reveal the benefits of continuous scheduling.","What is 1+1? Output only the number.",
    "Name one color.",
    "Say hello in Chinese.",
    "What is the capital of France? One short answer.",
    "Is Python a programming language? Answer yes or no only.",
    "Translate 'good morning' to Chinese.",
    "What day comes after Monday?",
    "Which is bigger: 9 or 12? Output only the number.",
    "Give one synonym of 'happy'.",
    "What is the opposite of 'hot'?",
    "How many days are in a week? Output only the number.",
    "Translate '谢谢' into English.",
    "Name one fruit.",
    "What is 10 minus 3? Output only the number.",
    "Is water wet? One short sentence.",
    "Write the Chinese word for 'cat'.",

    # 17-32: 短列表 / 简单解释（中等长度）
    "List three primary colors.",
    "Give three tips for staying focused while studying.",
    "Explain what machine learning is in two simple sentences.",
    "Translate this to Chinese: 'I am learning artificial intelligence.'",
    "Write three bullet points about the benefits of exercise.",
    "What is overfitting in machine learning? Explain simply.",
    "Summarize the purpose of a GPU in one paragraph.",
    "List 5 common Python data types.",
    "What is the difference between RAM and storage? Keep it short.",
    "Explain what an API is for a beginner.",
    "Write a short reply to: 'Thanks for your help!'",
    "Give 4 interview tips for a software engineering intern.",
    "Explain the meaning of 'continuous batching' in simple words.",
    "What is a transformer model? Answer for a college student.",
    "List three use cases of diffusion models.",
    "Translate to English: '我正在做一个推理加速项目。'",

    # 33-48: 中等复杂推理 / 结构化输出（长度差异会更明显）
    "Compare CNN and Transformer in 4 bullet points.",
    "Explain the difference between throughput and latency with examples.",
    "Write a concise definition of KV cache and why it helps inference.",
    "If a model generates 128 tokens in 0.5 seconds, what is decode throughput? Show calculation.",
    "Describe the advantages of mixed precision inference in practical deployment.",
    "Give a step-by-step plan to benchmark two inference engines fairly.",
    "Write a short email asking a professor for a meeting next week.",
    "Summarize the key idea of greedy decoding vs sampling.",
    "Explain what causes CUDA out-of-memory errors and how to debug them.",
    "List 5 factors that affect LLM inference speed on a single GPU.",
    "Describe how padding side (left vs right) can affect decoder-only generation.",
    "Write a polite response declining an invitation because of a deadline.",
    "Explain why fixed-length output may hide the benefits of continuous batching.",
    "Give a short checklist for reproducing benchmark results.",
    "Explain the role of RoPE in transformer attention.",
    "What is QK norm and why might Qwen3 use it?",

    # 49-64: 开放式写作 / 更长输出倾向（不一定早停）
    "Write a short introduction paragraph for a report comparing custom inference engines and HuggingFace Transformers.",
    "Explain continuous batching as if teaching a first-year graduate student in AI systems.",
    "Write a mini tutorial on how to profile a PyTorch inference pipeline step by step.",
    "Describe a fair benchmark protocol for comparing offline throughput between two LLM inference frameworks.",
    "Write a short Chinese explanation of why 'correctness first, optimization second' is important in systems engineering.",
    "Give a practical debugging strategy for when a custom LLM engine outputs repeated nonsense tokens.",
    "Write a concise note on why matching prefill and decode paths is essential in custom KV-cache implementations.",
    "Explain how FlashAttention helps attention efficiency without going too deep into math.",
    "Write a short comparison between static batching and continuous batching in online serving.",
    "Describe common sources of unfairness when comparing vLLM, HF generate, and a custom engine.",
    "Write a brief project summary for a resume about building a slot-based KV cache inference engine.",
    "Explain how EOS behavior can affect throughput benchmarks and result interpretation.",
    "Write a short recommendation section for improving a custom inference engine after correctness is achieved.",
    "Describe what metrics (TTFT, throughput, latency) should be tracked in LLM serving evaluation.",
    "Write a short Chinese summary of the difference between total tok/s and decode tok/s.",
    "Explain how heterogeneous prompt lengths can reveal the benefits of continuous scheduling.","What is 1+1? Output only the number.",
    "Name one color.",
    "Say hello in Chinese.",
    "What is the capital of France? One short answer.",
    "Is Python a programming language? Answer yes or no only.",
    "Translate 'good morning' to Chinese.",
    "What day comes after Monday?",
    "Which is bigger: 9 or 12? Output only the number.",
    "Give one synonym of 'happy'.",
    "What is the opposite of 'hot'?",
    "How many days are in a week? Output only the number.",
    "Translate '谢谢' into English.",
    "Name one fruit.",
    "What is 10 minus 3? Output only the number.",
    "Is water wet? One short sentence.",
    "Write the Chinese word for 'cat'.",

    # 17-32: 短列表 / 简单解释（中等长度）
    "List three primary colors.",
    "Give three tips for staying focused while studying.",
    "Explain what machine learning is in two simple sentences.",
    "Translate this to Chinese: 'I am learning artificial intelligence.'",
    "Write three bullet points about the benefits of exercise.",
    "What is overfitting in machine learning? Explain simply.",
    "Summarize the purpose of a GPU in one paragraph.",
    "List 5 common Python data types.",
    "What is the difference between RAM and storage? Keep it short.",
    "Explain what an API is for a beginner.",
    "Write a short reply to: 'Thanks for your help!'",
    "Give 4 interview tips for a software engineering intern.",
    "Explain the meaning of 'continuous batching' in simple words.",
    "What is a transformer model? Answer for a college student.",
    "List three use cases of diffusion models.",
    "Translate to English: '我正在做一个推理加速项目。'",

    # 33-48: 中等复杂推理 / 结构化输出（长度差异会更明显）
    "Compare CNN and Transformer in 4 bullet points.",
    "Explain the difference between throughput and latency with examples.",
    "Write a concise definition of KV cache and why it helps inference.",
    "If a model generates 128 tokens in 0.5 seconds, what is decode throughput? Show calculation.",
    "Describe the advantages of mixed precision inference in practical deployment.",
    "Give a step-by-step plan to benchmark two inference engines fairly.",
    "Write a short email asking a professor for a meeting next week.",
    "Summarize the key idea of greedy decoding vs sampling.",
    "Explain what causes CUDA out-of-memory errors and how to debug them.",
    "List 5 factors that affect LLM inference speed on a single GPU.",
    "Describe how padding side (left vs right) can affect decoder-only generation.",
    "Write a polite response declining an invitation because of a deadline.",
    "Explain why fixed-length output may hide the benefits of continuous batching.",
    "Give a short checklist for reproducing benchmark results.",
    "Explain the role of RoPE in transformer attention.",
    "What is QK norm and why might Qwen3 use it?",

    # 49-64: 开放式写作 / 更长输出倾向（不一定早停）
    "Write a short introduction paragraph for a report comparing custom inference engines and HuggingFace Transformers.",
    "Explain continuous batching as if teaching a first-year graduate student in AI systems.",
    "Write a mini tutorial on how to profile a PyTorch inference pipeline step by step.",
    "Describe a fair benchmark protocol for comparing offline throughput between two LLM inference frameworks.",
    "Write a short Chinese explanation of why 'correctness first, optimization second' is important in systems engineering.",
    "Give a practical debugging strategy for when a custom LLM engine outputs repeated nonsense tokens.",
    "Write a concise note on why matching prefill and decode paths is essential in custom KV-cache implementations.",
    "Explain how FlashAttention helps attention efficiency without going too deep into math.",
    "Write a short comparison between static batching and continuous batching in online serving.",
    "Describe common sources of unfairness when comparing vLLM, HF generate, and a custom engine.",
    "Write a brief project summary for a resume about building a slot-based KV cache inference engine.",
    "Explain how EOS behavior can affect throughput benchmarks and result interpretation.",
    "Write a short recommendation section for improving a custom inference engine after correctness is achieved.",
    "Describe what metrics (TTFT, throughput, latency) should be tracked in LLM serving evaluation.",
    "Write a short Chinese summary of the difference between total tok/s and decode tok/s.",
    "Explain how heterogeneous prompt lengths can reveal the benefits of continuous scheduling.",
]
    prompts = prompts[:num_prompts]

    return prompts


# -----------------------------
# HF benchmark
# -----------------------------

def benchmark_hf(
    model_path: str,
    prompts: List[str],
    output_tokens: int,
    dtype: torch.dtype,
    device: str = "cuda",
    warmup: bool = True,
) -> BenchResult:
    print("\n[HF] Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Build templated texts (same logic as custom generate after your update)
    templated_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True
        )
        for p in prompts
    ]

    if warmup:
        print("[HF] Warmup...")
        warm_text = templated_texts[:1]
        with torch.inference_mode():
            inputs = tokenizer(warm_text, return_tensors="pt", padding=True).to(device)
            _ = model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=8,
                # min_new_tokens=8,
                pad_token_id=tokenizer.pad_token_id,
                # eos_token_id=None,         # try to ignore EOS
            )
        sync_cuda()

    print("[HF] Running benchmark...")
    t0 = time.perf_counter()

    # Include tokenization + generate + decode in E2E (closer to your custom engine future API timing)
    inputs = tokenizer(templated_texts, return_tensors="pt", padding=True).to(device)
    input_lens = inputs["attention_mask"].sum(dim=1).tolist()

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            do_sample=False,              # greedy for fairness
            use_cache=True,
            max_new_tokens=output_tokens,
            # min_new_tokens=output_tokens, # fixed-length attempt
            pad_token_id=tokenizer.pad_token_id,
            # eos_token_id=None,            # try to disable EOS stopping
        )
    sync_cuda()

    # Decode (count decode in E2E to match custom behavior)
    _decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

    elapsed = time.perf_counter() - t0

    # Count generated lengths exactly from output_ids - input_lengths
    padded_input_len = inputs["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id

    gen_lens = []
    for i in range(len(prompts)):
        gen_tokens = out_ids[i, padded_input_len:].tolist()

        # Find first EOS in generated segment (inclusive)
        eff_len = len(gen_tokens)
        if eos_id is not None:
            try:
                first_eos = gen_tokens.index(eos_id)
                eff_len = first_eos + 1
            except ValueError:
                pass

        # Safety cap (should not exceed max_new_tokens in a correct setup)
        eff_len = min(eff_len, output_tokens)
        gen_lens.append(max(eff_len, 0))

    res = BenchResult(
        name="HF Transformers.generate",
        num_prompts=len(prompts),
        sum_prompt_tokens=int(sum(input_lens)),
        sum_generated_tokens=int(sum(gen_lens)),
        elapsed_s=elapsed,
        ttft_s_mean=None,  # Optional with streamer/thread instrumentation
        extra={
            "actual_gen_min/mean/max": f"{min(gen_lens)}/{sum(gen_lens)/len(gen_lens):.1f}/{max(gen_lens)}",
            "dtype": str(dtype).replace("torch.", ""),
            "device": device,
        }
    )
    return res


# -----------------------------
# Custom engine benchmark
# -----------------------------

def benchmark_custom(
    model_path: str,
    prompts: List[str],
    output_tokens: int,
    dtype: torch.dtype,
    device: str = "cuda",
    max_num_seqs: int = 32,
    max_seq_len: int = 4096,
    warmup: bool = True,
) -> BenchResult:
    print("\n[Custom] Initializing engine...")
    llm = CustomLLM(
        model_path=model_path,
        max_num_seqs=max_num_seqs,
        max_seq_len=max_seq_len,
        dtype=dtype,
    )

    # Warmup via one short call (engine already warms up internally, but keep symmetric)
    if warmup:
        print("[Custom] Warmup request...")
        fut = llm.generate(["warmup"], max_tokens=8, temperature=0.0, ignore_eos=False)
        _ = fut.result()

    print("[Custom] Running benchmark...")
    t0 = time.perf_counter()
    fut = llm.generate(
        prompts,
        max_tokens=output_tokens,
        temperature=0.0,   # greedy
        ignore_eos=False,   # fixed-length fairness
    )
    results = fut.result()
    sync_cuda()
    elapsed = time.perf_counter() - t0

    # Your custom GenerationOutput already tracks prompt/generated tokens in latest code
    sum_prompt = 0
    sum_gen = 0
    gen_lens = []
    for r in results:
        p = int(getattr(r, "prompt_tokens", 0))
        g = int(getattr(r, "generated_tokens", len(r.token_ids)))
        sum_prompt += p
        sum_gen += g
        gen_lens.append(g)

    res = BenchResult(
        name="Custom Engine",
        num_prompts=len(prompts),
        sum_prompt_tokens=sum_prompt,
        sum_generated_tokens=sum_gen,
        elapsed_s=elapsed,
        ttft_s_mean=None,  # Not available unless engine exposes first-token callback/timestamps
        extra={
            "actual_gen_min/mean/max": f"{min(gen_lens)}/{sum(gen_lens)/len(gen_lens):.1f}/{max(gen_lens)}",
            "dtype": str(dtype).replace("torch.", ""),
            "device": device,
            "max_num_seqs": max_num_seqs,
            "max_seq_len": max_seq_len,
        }
    )

    llm.stop()
    return res


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--input-tokens", type=int, default=512,
                        help="Target total input tokens AFTER chat template (approximate)")
    parser.add_argument("--output-tokens", type=int, default=128,
                        help="Fixed output length target")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-custom", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    dtype = parse_dtype(args.dtype)

    print("=" * 80)
    print("Qwen3 Offline Throughput Benchmark (Custom vs HF)")
    print("=" * 80)
    print(json.dumps({
        "model_path": args.model_path,
        "num_prompts": args.num_prompts,
        "target_input_tokens": args.input_tokens,
        "target_output_tokens": args.output_tokens,
        "dtype": args.dtype,
        "device": args.device,
        "seed": args.seed,
        "max_num_seqs(custom)": args.max_num_seqs,
        "max_seq_len(custom)": args.max_seq_len,
    }, indent=2))

    # Build prompts once using the same tokenizer family
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = build_prompt_batch(tok, args.num_prompts, args.input_tokens)

    custom_res = None
    hf_res = None

    if not args.skip_custom:
        custom_res = benchmark_custom(
            model_path=args.model_path,
            prompts=prompts,
            output_tokens=args.output_tokens,
            dtype=dtype,
            device=args.device,
            max_num_seqs=args.max_num_seqs,
            max_seq_len=max(args.max_seq_len, args.input_tokens + args.output_tokens + 16),
            warmup=True,
        )
        print_result(custom_res)

    if not args.skip_hf:
        hf_res = benchmark_hf(
            model_path=args.model_path,
            prompts=prompts,
            output_tokens=args.output_tokens,
            dtype=dtype,
            device=args.device,
            warmup=True,
        )
        print_result(hf_res)

    # Summary comparison
    if custom_res and hf_res:
        print("\n" + "=" * 80)
        print("Summary (HF baseline = 1.00x)")
        print("=" * 80)

        def ratio(a, b):
            return a / b if b > 0 else float("nan")

        print(f"Custom total tok/s : {custom_res.total_tok_per_s:.2f}")
        print(f"HF total tok/s     : {hf_res.total_tok_per_s:.2f}")
        print(f"Speedup total tok/s: {ratio(custom_res.total_tok_per_s, hf_res.total_tok_per_s):.3f}x")
        print("-" * 80)
        print(f"Custom decode tok/s : {custom_res.decode_tok_per_s:.2f}")
        print(f"HF decode tok/s     : {hf_res.decode_tok_per_s:.2f}")
        print(f"Speedup decode tok/s: {ratio(custom_res.decode_tok_per_s, hf_res.decode_tok_per_s):.3f}x")
        print("-" * 80)
        print(f"Custom E2E batch time : {custom_res.elapsed_s:.4f} s")
        print(f"HF E2E batch time     : {hf_res.elapsed_s:.4f} s")
        print(f"E2E time ratio (HF/custom): {ratio(hf_res.elapsed_s, custom_res.elapsed_s):.3f}x slower")
        print("=" * 80)

    print("\nDone.")


if __name__ == "__main__":
    main()

    #CUDA_VISIBLE_DEVICES=3 python benchmark.py --model-path /mnt/data0/Qwen30.6B --num-prompts 256 --input-tokens 512 --output-tokens 1024 --dtype bf16 