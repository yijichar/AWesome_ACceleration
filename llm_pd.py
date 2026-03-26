#!/usr/bin/env python3
# /home/zyf/workspace/Qwen-vllm-tp/llm_pd.py
"""
L3-PD: Prefill/Decode (PD) 分离推理引擎（MVP）

目标：
- 在 4×4090 上跑通 PD 分离：2 张卡做 prefill，2 张卡做 decode
- MVP：先保证“能跑 + 结构清晰 + 可 benchmark”，暂不实现 Continuous Batching / Prefix Caching

默认拓扑（建议）：
- tp_size = 1
- prefill_replicas = 2  -> ranks: 0,1
- decode_replicas  = 2  -> ranks: 2,3
  映射：prefill0 -> decode0，prefill1 -> decode1

通信：
- 控制流（Python 对象）走 world CPU group (gloo)：send_object/recv_object
- KV 走 NCCL GPU tensor 点对点：send_tensor/recv_tensor

Demo:
    torchrun --nproc_per_node=4 llm_pd.py --model-path /mnt/data0/Qwen30.6B \
        --max-num-seqs 64 --max-seq-len 4096 --max-tokens 128 --temperature 0.0 \
        --prompt "Hello, who are you?" --prompt "Explain KV cache in one paragraph."
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from model.distributed import (
    init_distributed,
    get_world_size,
    get_global_rank,
    get_local_rank,
    get_role,
    get_num_prefill_replicas,
    get_num_decode_replicas,
    is_prefill_worker,
    is_decode_worker,
    is_pd_mode,
    send_object,
    recv_object,
    send_tensor,
    recv_tensor,
)


# ------------------------------
# Sampling helper
# ------------------------------
def _sample_from_logits(logits: torch.Tensor, temperature: float) -> int:
    """
    logits: [vocab]
    """
    if temperature is None or temperature <= 0.0:
        return int(torch.argmax(logits).item())
    probs = F.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@dataclass
class PDTask:
    req_id: int
    token_ids: List[int]  # prompt tokens
    max_tokens: int       # generated tokens count (NOT including prompt)
    temperature: float
    ignore_eos: bool


class LLMPD:
    """
    - rank0：提供 generate API（tokenize / detokenize / 调度）
    - prefill ranks：prefill + export KV + send to decode
    - decode ranks：import KV + decode to completion + send results to rank0
    """
    def __init__(
        self,
        model_path: str,
        max_num_seqs: int = 64,
        max_seq_len: int = 4096,
        dtype: torch.dtype = torch.bfloat16,
        tp_size: int = 1,
        num_prefill_replicas: int = 2,
        num_decode_replicas: int = 2,
    ):
        init_distributed(
            tp_size=tp_size,
            pd_mode=True,
            num_prefill_replicas=num_prefill_replicas,
            num_decode_replicas=num_decode_replicas,
        )
        assert is_pd_mode(), "LLMPD requires pd_mode=True"

        self.world_size = get_world_size()
        self.rank = get_global_rank()
        self.local_rank = get_local_rank()
        self.role = get_role()
        self.P = get_num_prefill_replicas()
        self.D = get_num_decode_replicas()
        self.tp_size = tp_size

        self.device = torch.device(f"cuda:{self.local_rank}")
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.max_num_seqs = max_num_seqs

        # 只在 rank0 初始化 tokenizer
        if self.rank == 0:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.eos_token_id = int(self.tokenizer.eos_token_id)
        else:
            self.tokenizer = None
            self.eos_token_id = -1

        # 模型（prefill/decode 都需要）
        from model.model_tp import Qwen3Config, Qwen3ForCausalLMTP as ModelClass

        config_path = os.path.join(model_path, "config.json")
        self.config = Qwen3Config.from_json(config_path)

        if self.rank == 0:
            print(f"[LLMPD] world_size={self.world_size}, P={self.P}, D={self.D}, tp_size={self.tp_size}")
            print(f"[LLMPD] Loading model on rank0 (role={self.role}) ...")
        self.model = ModelClass.from_pretrained(model_path, self.config, self.device, dtype)
        self.model.eval()

        if self.rank == 0:
            print(f"[LLMPD] Allocating KV cache: slots={self.max_num_seqs}, max_seq_len={self.max_seq_len}")
        self.model.init_kv_cache(self.max_num_seqs, self.max_seq_len, self.device, dtype)

        # slot 池
        self._free_slots: List[int] = list(range(self.max_num_seqs))

        # 下发 eos id（只做一次：rank0 -> all decode ranks）
        if self.rank == 0:
            for dr in self._decode_global_ranks():
                send_object({"op": "eos_id", "eos_token_id": int(self.eos_token_id)}, dst=dr)

    # ------------------------------
    # Role mapping (MVP)
    # ------------------------------
    def _prefill_global_ranks(self) -> List[int]:
        return list(range(0, self.P * self.tp_size, self.tp_size))

    def _decode_global_ranks(self) -> List[int]:
        base = self.P * self.tp_size
        return list(range(base, base + self.D * self.tp_size, self.tp_size))

    def _prefill_replica_id_of_rank(self, rank: int) -> int:
        return rank // self.tp_size

    def _decode_replica_id_of_rank(self, rank: int) -> int:
        return (rank // self.tp_size) - self.P

    def _map_prefill_replica_to_decode_rank(self, prefill_replica_id: int) -> int:
        decode_replica_id = prefill_replica_id % self.D
        return self.P * self.tp_size + decode_replica_id * self.tp_size  # leader rank

    def _alloc_slot(self) -> int:
        if not self._free_slots:
            raise RuntimeError("No free slots. Increase max_num_seqs.")
        return self._free_slots.pop()

    def _free_slot(self, slot: int):
        self.model.clear_slot(slot)
        self._free_slots.append(slot)

    # ------------------------------
    # Prefill (local) - per request (no padding, correctness-first)
    # ------------------------------
    @torch.no_grad()
    def _prefill_one(self, task: PDTask) -> Tuple[int, torch.Tensor, torch.Tensor, int]:
        slot = self._alloc_slot()
        tokens = task.token_ids
        L = len(tokens)
        if L <= 0 or L > self.max_seq_len:
            self._free_slot(slot)
            raise ValueError(f"Invalid prompt length: {L} (max_seq_len={self.max_seq_len})")

        input_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)   # [1, L]
        position_ids = torch.arange(L, device=self.device, dtype=torch.long).unsqueeze(0)     # [1, L]
        slot_indices = torch.tensor([slot], dtype=torch.long, device=self.device)            # [1]

        logits = self.model(input_ids, position_ids, slot_indices=slot_indices)              # [1, L, vocab]
        last_logits = logits[0, L - 1].float()
        first_token = _sample_from_logits(last_logits, task.temperature)

        k_all, v_all, seqlen = self.model.export_request_kv(slot, L)

        # 清理本地 slot（decode 侧会 import）
        self._free_slot(slot)
        return int(first_token), k_all, v_all, int(seqlen)

    # ------------------------------
    # Decode (local) - per request
    # ------------------------------
    @torch.no_grad()
    def _decode_one_local(
        self,
        first_token: int,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        prompt_len: int,
        max_tokens: int,
        temperature: float,
        ignore_eos: bool,
        eos_token_id: int,
    ) -> List[int]:
        """
        生成 tokens（不含 prompt tokens）。
        注意：prefill 已经生成了 first_token，所以剩余步数为 max_tokens-1
        """
        slot = self._alloc_slot()
        self.model.import_request_kv(slot, k_all, v_all, prompt_len)

        generated: List[int] = []
        token = int(first_token)
        generated.append(token)

        cur_pos = int(prompt_len)

        remaining = max(0, int(max_tokens) - 1)
        for _ in range(remaining):
            input_ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
            position_ids = torch.tensor([[cur_pos]], dtype=torch.long, device=self.device)
            slot_indices = torch.tensor([slot], dtype=torch.long, device=self.device)

            logits = self.model(input_ids, position_ids, slot_indices=slot_indices)  # [1,1,vocab]
            next_logits = logits[0, 0].float()
            nxt = _sample_from_logits(next_logits, temperature)
            if (not ignore_eos) and (int(nxt) == int(eos_token_id)):
                break
            generated.append(int(nxt))
            token = int(nxt)
            cur_pos += 1

        self._free_slot(slot)
        return generated

    # ------------------------------
    # Worker loops
    # ------------------------------
    def run_forever(self):
        """
        非 rank0 入口：阻塞运行 worker loop。
        """
        if self.rank == 0:
            return
        if is_prefill_worker():
            self._prefill_worker_loop()
            return
        if is_decode_worker():
            self._decode_worker_loop()
            return
        raise RuntimeError(f"Unexpected role={self.role} on rank={self.rank}")

    def _prefill_worker_loop(self):
        """
        仅接收来自 rank0 的 prefill_batch。
        对每个 task：prefill -> send to mapped decode rank。
        shutdown：同时转发给对应 decode rank。
        """
        src = 0
        my_prefill_replica_id = self._prefill_replica_id_of_rank(self.rank)
        my_decode_rank = self._map_prefill_replica_to_decode_rank(my_prefill_replica_id)

        while True:
            msg = recv_object(src)
            if not isinstance(msg, dict):
                raise RuntimeError(f"[prefill rank{self.rank}] invalid msg type: {type(msg)}")
            op = msg.get("op")
            if op == "shutdown":
                # 转发给对应 decode
                send_object({"op": "shutdown"}, dst=my_decode_rank)
                break
            if op != "prefill_batch":
                raise RuntimeError(f"[prefill rank{self.rank}] unknown op: {op}")

            tasks = msg["tasks"]
            assert isinstance(tasks, list)
            for t in tasks:
                task = PDTask(**t)
                first_token, k_all, v_all, prompt_len = self._prefill_one(task)

                meta = {
                    "op": "decode_task",
                    "req_id": task.req_id,
                    "first_token": int(first_token),
                    "prompt_len": int(prompt_len),
                    "max_tokens": int(task.max_tokens),
                    "temperature": float(task.temperature),
                    "ignore_eos": bool(task.ignore_eos),
                }
                send_object(meta, dst=my_decode_rank)
                send_tensor(k_all, dst=my_decode_rank)
                send_tensor(v_all, dst=my_decode_rank)

    def _decode_worker_loop(self):
        """
        MVP：decode replica i 只接收 prefill replica i 的消息（避免 ANY_SOURCE）。
        启动时先从 rank0 收到 eos_id。
        """
        # receive eos_id once
        eos_msg = recv_object(0)
        assert eos_msg.get("op") == "eos_id"
        eos_token_id = int(eos_msg["eos_token_id"])

        my_decode_replica_id = self._decode_replica_id_of_rank(self.rank)
        src_prefill_rank = my_decode_replica_id * self.tp_size  # leader rank of matched prefill replica

        while True:
            meta = recv_object(src_prefill_rank)
            op = meta.get("op")
            if op == "shutdown":
                break
            if op != "decode_task":
                raise RuntimeError(f"[decode rank{self.rank}] unknown op: {op}")

            req_id = int(meta["req_id"])
            first_token = int(meta["first_token"])
            prompt_len = int(meta["prompt_len"])
            max_tokens = int(meta["max_tokens"])
            temperature = float(meta["temperature"])
            ignore_eos = bool(meta["ignore_eos"])

            # recv KV tensors
            num_layers = int(self.config.num_hidden_layers)
            kv_heads = int(self.model.model.layers[0].self_attn.num_kv_heads_per_partition)
            head_dim = int(self.model.model.layers[0].self_attn.head_dim)
            k_all = torch.empty((num_layers, prompt_len, kv_heads, head_dim), device=self.device, dtype=self.dtype)
            v_all = torch.empty_like(k_all)
            recv_tensor(k_all, src=src_prefill_rank)
            recv_tensor(v_all, src=src_prefill_rank)

            gen = self._decode_one_local(
                first_token=first_token,
                k_all=k_all,
                v_all=v_all,
                prompt_len=prompt_len,
                max_tokens=max_tokens,
                temperature=temperature,
                ignore_eos=ignore_eos,
                eos_token_id=eos_token_id,
            )

            send_object({"op": "result", "req_id": req_id, "generated": gen}, dst=0)

    # ------------------------------
    # Public API (rank0 only)
    # ------------------------------
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        temperature: float = 0.0,
        ignore_eos: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        rank0 only.
        返回每条 prompt 的生成结果（generated tokens only + text）。
        """
        assert self.rank == 0, "generate() must be called on global rank0"
        assert self.tokenizer is not None

        prefill_ranks = self._prefill_global_ranks()
        tasks_by_prefill: Dict[int, List[PDTask]] = {r: [] for r in prefill_ranks}

        # build tasks
        for rid, p in enumerate(prompts):
            token_ids = self.tokenizer.encode(p, add_special_tokens=False)
            task = PDTask(
                req_id=rid,
                token_ids=token_ids,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                ignore_eos=bool(ignore_eos),
            )
            pr = prefill_ranks[rid % len(prefill_ranks)]
            tasks_by_prefill[pr].append(task)

        # send tasks to remote prefill ranks
        for pr, tlist in tasks_by_prefill.items():
            if pr == 0:
                continue
            send_object({"op": "prefill_batch", "tasks": [t.__dict__ for t in tlist]}, dst=pr)

        # local prefill (rank0) -> send to decode0
        local_prefill_replica_id = 0
        local_decode_rank = self._map_prefill_replica_to_decode_rank(local_prefill_replica_id)
        for task in tasks_by_prefill.get(0, []):
            first_token, k_all, v_all, prompt_len = self._prefill_one(task)
            meta = {
                "op": "decode_task",
                "req_id": task.req_id,
                "first_token": int(first_token),
                "prompt_len": int(prompt_len),
                "max_tokens": int(task.max_tokens),
                "temperature": float(task.temperature),
                "ignore_eos": bool(task.ignore_eos),
            }
            send_object(meta, dst=local_decode_rank)
            send_tensor(k_all, dst=local_decode_rank)
            send_tensor(v_all, dst=local_decode_rank)

        # collect results deterministically (by decode rank)
        results: Dict[int, List[int]] = {}
        for prefill_replica_id, pr in enumerate(prefill_ranks):
            dr = self._map_prefill_replica_to_decode_rank(prefill_replica_id)
            cnt = len(tasks_by_prefill.get(pr, []))
            for _ in range(cnt):
                msg = recv_object(src=dr)
                assert msg.get("op") == "result"
                results[int(msg["req_id"])] = list(msg["generated"])

        # detokenize
        out: List[Dict[str, Any]] = []
        for rid, p in enumerate(prompts):
            gen_ids = results.get(rid, [])
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            out.append({
                "text": text,
                "token_ids": gen_ids,
                "prompt_tokens": len(self.tokenizer.encode(p, add_special_tokens=False)),
                "generated_tokens": len(gen_ids),
            })
        return out

    def shutdown(self):
        """
        rank0 only：关闭所有 worker。
        注意：decode rank3 的 shutdown 需要通过 prefill rank1 转发（因为 decode 只收对应 prefill 的消息）。
        """
        if self.rank != 0:
            return
        prefill_ranks = self._prefill_global_ranks()
        decode_ranks = self._decode_global_ranks()

        # 1) 关闭 remote prefill（prefill 会转发给对应 decode）
        for pr in prefill_ranks:
            if pr == 0:
                continue
            send_object({"op": "shutdown"}, dst=pr)

        # 2) 关闭 rank0 对应的 decode（decode0）
        if len(decode_ranks) > 0:
            send_object({"op": "shutdown"}, dst=decode_ranks[0])


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--prefill-replicas", type=int, default=2)
    parser.add_argument("--decode-replicas", type=int, default=2)

    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--prompt", type=str, action="append", default=[])

    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    engine = LLMPD(
        model_path=args.model_path,
        max_num_seqs=args.max_num_seqs,
        max_seq_len=args.max_seq_len,
        dtype=dtype,
        tp_size=args.tp_size,
        num_prefill_replicas=args.prefill_replicas,
        num_decode_replicas=args.decode_replicas,
    )

    rank = get_global_rank()
    if rank != 0:
        engine.run_forever()
        return

    prompts = args.prompt if args.prompt else ["Hello!"]
    outputs = engine.generate(
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        ignore_eos=args.ignore_eos,
    )

    for i, out in enumerate(outputs):
        print("=" * 60)
        print(f"[Prompt {i}] {prompts[i]}")
        print(f"[Generated] {out['text']}")
        print(f"[gen_tokens] {out['generated_tokens']}")

    engine.shutdown()


if __name__ == "__main__":
    main()