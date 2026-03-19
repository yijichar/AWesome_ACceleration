#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DP 2 + TP 2
TP_PROFILE=1 TP_PROFILE_SYNC=0  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 benchmark_tp.py   --engine custom --tp --tp-size 2   --model-path /mnt/data0/Qwen30.6B   --num-prompts 1024 --input-tokens 1024 --output-tokens 1024   --dtype bf16 --max-num-seqs 256

TP 2
TP_PROFILE=1 TP_PROFILE_SYNC=0 CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 benchmark_tp.py   --engine custom --tp --tp-size 2   --model-path /mnt/data0/Qwen30.6B   --num-prompts 512 --input-tokens 1024 --output-tokens 1024   --dtype bf16 --max-num-seqs 256
"""
import os
import sys
import time
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Dist detection (torchrun)
# -----------------------------
IS_DISTRIBUTED_ENV = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)


def env_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_rank0() -> bool:
    return env_rank() == 0


# -----------------------------
# Import engines lazily
# -----------------------------
def get_custom_engine_class(use_tp: bool):
    if use_tp:
        from llm_tp import LLMTP as CustomLLM
    else:
        from llm import LLM as CustomLLM
    return CustomLLM


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
        print("TTFT              : N/A (not instrumented)")
    else:
        print(f"TTFT mean / p50 / p95 : {r.ttft_s_mean:.4f} / {r.ttft_s_p50:.4f} / {r.ttft_s_p95:.4f} s")
    if r.extra:
        for k, v in r.extra.items():
            print(f"{k:18}: {v}")


# -----------------------------
# Prompt generation
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
    prefix = f"Request #{idx}. Please answer concisely. "
    if chat_token_len(tokenizer, prefix) >= target_tokens:
        return prefix

    unit_pool = [
        "alpha beta gamma delta ",
        "one two three four five ",
        "red blue green yellow ",
        "data model token cache ",
        "hello world benchmark test ",
    ]
    unit = unit_pool[idx % len(unit_pool)]

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
    return prompts

# -----------------------------
# HF benchmark (single process only)
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
        dtype=dtype,   # newer transformers prefers dtype
        trust_remote_code=True,
    ).to(device)
    model.eval()

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
                pad_token_id=tokenizer.pad_token_id,
            )
        sync_cuda()

    print("[HF] Running benchmark...")
    t0 = time.perf_counter()

    inputs = tokenizer(templated_texts, return_tensors="pt", padding=True).to(device)
    input_lens = inputs["attention_mask"].sum(dim=1).tolist()

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=output_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    sync_cuda()

    _ = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
    elapsed = time.perf_counter() - t0

    padded_input_len = inputs["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id

    gen_lens = []
    for i in range(len(prompts)):
        gen_tokens = out_ids[i, padded_input_len:].tolist()

        eff_len = len(gen_tokens)
        if eos_id is not None:
            try:
                first_eos = gen_tokens.index(eos_id)
                eff_len = first_eos + 1
            except ValueError:
                pass

        eff_len = min(eff_len, output_tokens)
        gen_lens.append(max(eff_len, 0))

    return BenchResult(
        name="HF Transformers.generate",
        num_prompts=len(prompts),
        sum_prompt_tokens=int(sum(input_lens)),
        sum_generated_tokens=int(sum(gen_lens)),
        elapsed_s=elapsed,
        ttft_s_mean=None,
        extra={
            "actual_gen_min/mean/max": f"{min(gen_lens)}/{sum(gen_lens)/len(gen_lens):.1f}/{max(gen_lens)}",
            "dtype": str(dtype).replace("torch.", ""),
            "device": device,
        }
    )

def shard_list_round_robin(xs: List[str], n: int) -> List[List[str]]:
    shards = [[] for _ in range(n)]
    for i, x in enumerate(xs):
        shards[i % n].append(x)
    return shards

def flatten_list_of_lists(xss: List[List[int]]) -> List[int]:
    out = []
    for xs in xss:
        out.extend(xs)
    return out

# -----------------------------
# Custom benchmark (single GPU or TP)
# -----------------------------
def benchmark_custom(
    model_path: str,
    prompts: Optional[List[str]],
    output_tokens: int,
    dtype: torch.dtype,
    max_num_seqs: int = 32,
    max_seq_len: int = 1024,
    warmup: bool = True,
    use_tp: bool = False,
    tp_size: int = 1,
) -> Optional[BenchResult]:
    import torch.distributed as dist
    CustomLLM = get_custom_engine_class(use_tp=use_tp)

    print_prefix = "[Custom-TP]" if use_tp else "[Custom]"
    if is_rank0():
        print(f"\n{print_prefix} Initializing engine...")

    # NOTE: In TP mode, all ranks must instantiate the engine.
    if use_tp:
        llm = CustomLLM(
            model_path=model_path,
            max_num_seqs=max_num_seqs,
            max_seq_len=max_seq_len,
            dtype=dtype,
            enable_tp=True,
            tp_size=tp_size,
        )
    else:
        llm = CustomLLM(
            model_path=model_path,
            max_num_seqs=max_num_seqs,
            max_seq_len=max_seq_len,
            dtype=dtype,
        )

    # TP worker ranks: hand over to worker loop and exit benchmark path
    if use_tp and not llm.is_tp_leader:
        if hasattr(llm, "serve_worker"):
            llm.serve_worker()
        elif hasattr(llm, "_worker_loop"):
            llm._worker_loop()
        else:
            raise RuntimeError("TP worker rank has no serve_worker() / _worker_loop().")
        return None

    # Rank0 only below
    # ------------------------------------------------------------------
    # 单卡 custom：保留原逻辑
    # ------------------------------------------------------------------
    if not use_tp:
        assert prompts is not None and len(prompts) > 0, "Single-GPU custom requires prompts"
        if warmup:
            print(f"{print_prefix} Warmup request...")
            fut = llm.generate(["warmup"], max_tokens=8, temperature=0.0, ignore_eos=False)
            _ = fut.result()

        print(f"{print_prefix} Running benchmark...")
        t0 = time.perf_counter()

        fut = llm.generate(
            prompts,
            max_tokens=output_tokens,
            temperature=0.0,
            ignore_eos=False,
        )
        results = fut.result()
        sync_cuda()
        elapsed = time.perf_counter() - t0

        sum_prompt = 0
        sum_gen = 0
        gen_lens = []
        for r in results:
            p = int(getattr(r, "prompt_tokens", 0))
            g = int(getattr(r, "generated_tokens", len(getattr(r, "token_ids", []))))
            sum_prompt += p
            sum_gen += g
            gen_lens.append(g)

        llm.stop()
        return BenchResult(
            name="Custom Engine",
            num_prompts=len(prompts),
            sum_prompt_tokens=sum_prompt,
            sum_generated_tokens=sum_gen,
            elapsed_s=elapsed,
            ttft_s_mean=None,
            extra={
                "actual_gen_min/mean/max": f"{min(gen_lens)}/{sum(gen_lens)/len(gen_lens):.1f}/{max(gen_lens)}",
                "dtype": str(dtype).replace("torch.", ""),
                "device": "cuda",
                "max_num_seqs": max_num_seqs,
                "max_seq_len": max_seq_len,
                "tp": False,
                "rank": env_rank(),
                "world_size": env_world_size(),
            }
        )
    # ------------------------------------------------------------------
    # TP / DP×TP 模式
    # ------------------------------------------------------------------
    assert hasattr(llm, "is_tp_leader")
    assert hasattr(llm, "tp_size")
    assert hasattr(llm, "dp_size")
    assert hasattr(llm, "global_rank")
    assert hasattr(llm, "is_global_rank_0")

    # worker rank 只进入 worker loop
    if not llm.is_tp_leader:
        if hasattr(llm, "serve_worker"):
            llm.serve_worker()
        elif hasattr(llm, "_worker_loop"):
            llm._worker_loop()
        else:
            raise RuntimeError("TP worker rank has no serve_worker() / _worker_loop().")
        return None

    leader_ranks = [i * llm.tp_size for i in range(llm.dp_size)]

    # global rank0 切分 prompts 发给各 replica leader
    leader_ranks = [i * llm.tp_size for i in range(llm.dp_size)]

    # 每个 TP leader 都本地重建同一份 prompts，然后各自拿自己那份 shard
    assert prompts is not None and len(prompts) > 0, "TP leader must have prompts"
    all_shards = shard_list_round_robin(prompts, llm.dp_size)
    local_prompts = all_shards[llm.dp_rank]

    if llm.is_global_rank_0:
        print(f"[Global rank0] Prompt shards: {[len(x) for x in all_shards]}")
    print(f"[Leader rank={llm.global_rank}] local shard size={len(local_prompts)}")

    assert local_prompts is not None
    assert isinstance(local_prompts, list)

    # 每个 leader 本地 warmup
    if warmup:
        print(f"[Replica leader rank={llm.global_rank}] Warmup request...")
        fut = llm.generate(["warmup"], max_tokens=8, temperature=0.0, ignore_eos=False)
        _ = fut.result()

    print(f"[Replica leader rank={llm.global_rank}] Running local benchmark on {len(local_prompts)} prompts...")
    t0 = time.perf_counter()

    fut = llm.generate(
        local_prompts,
        max_tokens=output_tokens,
        temperature=0.0,
        ignore_eos=False,
    )
    results = fut.result()
    sync_cuda()
    elapsed = time.perf_counter() - t0

    sum_prompt = 0
    sum_gen = 0
    gen_lens = []
    for r in results:
        p = int(getattr(r, "prompt_tokens", 0))
        g = int(getattr(r, "generated_tokens", len(getattr(r, "token_ids", []))))
        sum_prompt += p
        sum_gen += g
        gen_lens.append(g)

    local_stat = {
        "name": f"Custom Engine Replica(rank={llm.global_rank})",
        "num_prompts": len(local_prompts),
        "sum_prompt_tokens": sum_prompt,
        "sum_generated_tokens": sum_gen,
        "elapsed_s": elapsed,
        "gen_lens": gen_lens,
        "rank": llm.global_rank,
    }

    # 非 global rank0 leader：发送统计回去
    if not llm.is_global_rank_0:
        dist.send_object_list([local_stat], dst=0)
        llm.stop()
        return None

    # global rank0：收集所有副本统计
    all_stats = [local_stat]
    for leader_rank in leader_ranks[1:]:
        obj_list = [None]
        dist.recv_object_list(obj_list, src=leader_rank)
        all_stats.append(obj_list[0])

    all_gen_lens = flatten_list_of_lists([s["gen_lens"] for s in all_stats])
    
    # -----------------------------
    # Profiler summary (global rank0 replica only)
    # -----------------------------
    if hasattr(llm, "prof") and llm.prof is not None and llm.prof.enabled:
        print("\n" + "=" * 80)
        print("Profiler Summary (replica on global rank0 only)")
        print("=" * 80)
        print(llm.prof.summary_filtered("rank0.prefill_flat", prefix="[TPROF] "))
        print()
        print(llm.prof.summary_filtered("rank0.decode", prefix="[TPROF] "))
        print()
        print(llm.prof.summary_filtered("comm.", prefix="[TPROF] "))
        print()
    res = BenchResult(
        name="Custom Engine (TP/DP static-sharded)",
        num_prompts=sum(s["num_prompts"] for s in all_stats),
        sum_prompt_tokens=sum(s["sum_prompt_tokens"] for s in all_stats),
        sum_generated_tokens=sum(s["sum_generated_tokens"] for s in all_stats),
        elapsed_s=max(s["elapsed_s"] for s in all_stats),
        ttft_s_mean=None,
        extra={
            "actual_gen_min/mean/max": f"{min(all_gen_lens)}/{sum(all_gen_lens)/len(all_gen_lens):.1f}/{max(all_gen_lens)}",
            "dtype": str(dtype).replace("torch.", ""),
            "device": "cuda",
            "max_num_seqs": max_num_seqs,
            "max_seq_len": max_seq_len,
            "tp": True,
            "tp_size": llm.tp_size,
            "dp_size": llm.dp_size,
            "world_size": env_world_size(),
            "leader_ranks": leader_ranks,
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
    parser.add_argument("--engine", type=str, default="custom", choices=["custom", "hf", "both"])
    parser.add_argument("--tp", action="store_true", help="Use Tensor Parallel custom engine (torchrun)")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--input-tokens", type=int, default=512,
                        help="Target input tokens AFTER chat template (approximate; only for synthetic builder)")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--tp-size", type=int, default=1, help="TP size per replica")
    args = parser.parse_args()

    # Safety checks
    if args.tp and IS_DISTRIBUTED_ENV:
        if env_world_size() % args.tp_size != 0:
            if is_rank0():
                raise RuntimeError(
                    f"WORLD_SIZE ({env_world_size()}) must be divisible by tp_size ({args.tp_size})."
                )
                
    if args.tp and not IS_DISTRIBUTED_ENV:
        if is_rank0():
            raise RuntimeError("--tp requires torchrun (distributed env vars RANK/WORLD_SIZE).")

    if IS_DISTRIBUTED_ENV and args.engine in ("hf", "both"):
        if is_rank0():
            raise RuntimeError("Under torchrun, only --engine custom is supported. Run HF baseline separately (single process).")

    set_seed(args.seed)
    dtype = parse_dtype(args.dtype)

    if is_rank0():
        print("=" * 80)
        print("Qwen3 Benchmark (Custom single/TP vs HF)")
        print("=" * 80)
        print(json.dumps({
            "distributed": IS_DISTRIBUTED_ENV,
            "rank": env_rank(),
            "world_size": env_world_size(),
            "model_path": args.model_path,
            "engine": args.engine,
            "tp": args.tp,
            "tp_size": args.tp_size,
            "dp_size(expected)": (env_world_size() // args.tp_size) if (args.tp and IS_DISTRIBUTED_ENV) else 1,
            "num_prompts": args.num_prompts,
            "target_input_tokens": args.input_tokens,
            "target_output_tokens": args.output_tokens,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "max_num_seqs(custom)": args.max_num_seqs,
            "max_seq_len(custom)": args.max_seq_len,
        }, indent=2))

    # IMPORTANT: Build prompts on rank0 only. Workers do not need prompts.
    prompts = None
    if args.tp:
        # TP / DP×TP benchmark: 所有 rank 都构造同一份 prompts，避免 leader/worker 在 llm init 前时序失衡
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        prompts = build_prompt_batch(tok, args.num_prompts, args.input_tokens)
    else:
        # 单卡 / HF：只需要 rank0
        if is_rank0():
            tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            prompts = build_prompt_batch(tok, args.num_prompts, args.input_tokens)
    custom_res = None
    hf_res = None

    # Custom (single or TP)
    if args.engine in ("custom", "both"):
        custom_res = benchmark_custom(
            model_path=args.model_path,
            prompts=prompts,  # rank0 gets list; worker ranks get None
            output_tokens=args.output_tokens,
            dtype=dtype,
            max_num_seqs=args.max_num_seqs,
            max_seq_len=max(args.max_seq_len, args.input_tokens + args.output_tokens + 16),
            warmup=True,
            use_tp=args.tp,
            tp_size=args.tp_size,
        )
        if is_rank0() and custom_res is not None:
            print_result(custom_res)

    # HF (single process only)
    if args.engine in ("hf", "both"):
        assert not IS_DISTRIBUTED_ENV, "HF benchmark must be run in single process"
        hf_res = benchmark_hf(
            model_path=args.model_path,
            prompts=prompts,
            output_tokens=args.output_tokens,
            dtype=dtype,
            device=args.device,
            warmup=True,
        )
        print_result(hf_res)

    # Summary (rank0 only)
    if is_rank0() and custom_res and hf_res:
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

    if IS_DISTRIBUTED_ENV:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    if is_rank0():
        print("\nDone.")


if __name__ == "__main__":
    main()