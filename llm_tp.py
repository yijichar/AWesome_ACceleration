#!/usr/bin/env python3
"""
L3: 引擎层 - 支持 Tensor Parallel 的推理引擎（MVP可运行版）
基于原始 llm.py，添加多卡调度逻辑

设计目标（当前版本）：
1) Rank 0 负责调度、采样、返回结果
2) 所有 rank 必须参与 prefill/decode，避免 TP collectives 死锁
3) 用 broadcast_object + broadcast_tensor 实现控制流
4) 先保证正确性，后续再优化 prefill 为 packed varlen
"""
import os
import time
import threading
import queue
import concurrent.futures
import bisect
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from flash_attn import flash_attn_varlen_func
from model.distributed import (
    get_world_size,
    get_global_rank,
    get_dp_world_size,
    get_dp_rank,
    is_tp_leader,
    is_global_rank_0,
)
import torch.distributed as dist
from model.distributed import (
    init_distributed,
    is_distributed,
    get_tp_rank,
    get_tp_group,
    get_tp_world_size,
    broadcast_tensor,
    broadcast_object,
)

from model.profiler_tp import TPProfiler
from model.distributed import set_tp_profiler

# ============================================================================
# Request / Output
# ============================================================================

@dataclass
class GenerationOutput:
    """Result of a single prompt generation"""
    text: str
    token_ids: list[int]
    prompt_tokens: int = 0
    generated_tokens: int = 0


@dataclass
class _Request:
    """Internal request tracking for continuous batching"""
    token_ids: list[list[int]]
    max_tokens: int
    temperature: float
    ignore_eos: bool
    future: concurrent.futures.Future
    results: list = field(default_factory=list)
    pending_indices: list = field(default_factory=list)


# ============================================================================
# LLMTP
# ============================================================================
def _tp_barrier():
    if torch.distributed.is_initialized():
        group = get_tp_group()
        if group is not None:
            torch.distributed.barrier(group=group)
class LLMTP:
    """
    High-throughput LLM inference engine with Tensor Parallel support (MVP)

    TP 策略（当前实现）：
    - Rank 0：调度（队列、slot分配、采样、detokenize）
    - Worker ranks：等待 rank0 指令并参与 forward
    - 通过广播 control + tensors 保证所有 rank 同步进入 TP collectives

    启动方式：
        torchrun --nproc_per_node=2 your_script.py

    注意：
    - 当前 prefill 为 padded batch 版本（稳定优先）
    - 后续可升级成 packed varlen + 更高效广播
    """

    def __init__(
        self,
        model_path: str,
        max_num_seqs: int = 32,
        max_seq_len: int = 1024,
        dtype=torch.bfloat16,
        enable_tp: bool = True,
        tp_size: int = 1,
    ):
        # ---------------------------------------------------------------------
        # Init distributed
        # ---------------------------------------------------------------------
        if enable_tp:
            init_distributed(tp_size=tp_size)

        self.tp_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()
        self.dp_size = get_dp_world_size()
        self.dp_rank = get_dp_rank()
        self.global_rank = get_global_rank()

        self.is_tp_leader = is_tp_leader()
        self.is_global_rank_0 = is_global_rank_0()

        # 兼容旧逻辑：现在表示“当前 TP 组 leader”
        self.is_rank_0 = self.is_tp_leader

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{local_rank}")
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        # ---------------------------------------------------------------------
        # Choose model implementation
        # ---------------------------------------------------------------------
        if is_distributed():
            from model.model_tp import Qwen3Config, Qwen3ForCausalLMTP as ModelClass
            if self.is_tp_leader:
                print(f"\n{'='*60}")
                print("Initializing Qwen 3 with Tensor Parallel / DP×TP")
                print(f"  Global rank : {self.global_rank}")
                print(f"  DP rank     : {self.dp_rank}")
                print(f"  TP rank     : {self.tp_rank}")
                print(f"  TP size     : {self.tp_size}")
                print(f"  DP size     : {self.dp_size}")
                print(f"  Device      : {self.device}")
                print(f"{'='*60}\n")
        else:
            from model.model import Qwen3Config, Qwen3ForCausalLM as ModelClass
            if self.is_tp_leader:
                print("\nInitializing Qwen 3 (single GPU)")

        self._ModelClass = ModelClass
        self._Qwen3Config = Qwen3Config

        # ---------------------------------------------------------------------
        # Load config
        # ---------------------------------------------------------------------
        config_path = os.path.join(model_path, "config.json")
        self.config = Qwen3Config.from_json(config_path)

        # ---------------------------------------------------------------------
        # Estimate max batch size by KV cache memory
        # ---------------------------------------------------------------------
        gpu_memory_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        kv_bytes_per_token = (
            2 *  # K and V
            self.config.num_key_value_heads *
            self.config.head_dim *
            2 *  # bf16/fp16 = 2 bytes
            self.config.num_hidden_layers
        )

        # TP下每张卡只存本rank的KV heads
        if is_distributed() and self.tp_size > 1:
            kv_bytes_per_token = kv_bytes_per_token // self.tp_size

        available_memory = (gpu_memory_gb - 15.0) * (1024**3)
        calculated_max_seqs = max(1, int(available_memory / kv_bytes_per_token) // max_seq_len)
        self.max_num_seqs = min(max_num_seqs, calculated_max_seqs)

        if self.is_rank_0:
            print(f"  GPU memory: {gpu_memory_gb:.1f} GB")
            print(f"  Max batch size: {self.max_num_seqs}")
            print(f"  Max sequence length: {self.max_seq_len}")

        # ---------------------------------------------------------------------
        # Tokenizer (rank0 only)
        # ---------------------------------------------------------------------
        if self.is_rank_0:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            # 避免 pad_token_id 缺失
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = None

        # ---------------------------------------------------------------------
        # Load model
        # ---------------------------------------------------------------------
        if self.is_rank_0:
            print("\nLoading model...")
        self.model = ModelClass.from_pretrained(model_path, self.config, self.device, dtype)
        self.model.eval()
        if is_distributed():
            _tp_barrier()

        # ---------------------------------------------------------------------
        # Init KV cache
        # ---------------------------------------------------------------------
        if self.is_rank_0:
            print(f"Allocating KV cache ({self.max_num_seqs} slots × {self.max_seq_len} tokens)...")
        self.model.init_kv_cache(self.max_num_seqs, self.max_seq_len, self.device, dtype)
        if is_distributed():
            _tp_barrier()
        # ---------------------------------------------------------------------
        # Preallocated decode buffers (all ranks)
        # ---------------------------------------------------------------------
        self._decode_input_ids = torch.zeros(self.max_num_seqs, 1, dtype=torch.long, device=self.device)
        self._decode_positions = torch.zeros(self.max_num_seqs, 1, dtype=torch.long, device=self.device)

        # ---------------------------------------------------------------------
        # Async queue (rank0 only)
        # ---------------------------------------------------------------------
        self._request_queue: Optional[queue.Queue] = None
        self._loop_running = False
        self._loop_thread = None
        if self.is_rank_0:
            self._request_queue = queue.Queue()

        # ---------------------------------------------------------------------
        # Warmup (all ranks must run)
        # ---------------------------------------------------------------------
        if self.is_rank_0:
            print("Warming up...")
        with torch.no_grad():
            dummy_ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            dummy_pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            dummy_slots = torch.zeros(1, dtype=torch.long, device=self.device)
            _ = self.model(dummy_ids, dummy_pos, dummy_slots)
        self.model.clear_all_slots()
        if is_distributed():
            _tp_barrier()
        if self.is_rank_0:
            print("✓ Engine ready\n")

        # 可通过环境变量控制，避免一直开
        enable_prof = os.environ.get("TP_PROFILE", "0") == "1"
        cuda_sync_prof = os.environ.get("TP_PROFILE_SYNC", "0") == "1"

        self.prof = TPProfiler(enabled=enable_prof, cuda_sync=cuda_sync_prof, keep_last=64)
        set_tp_profiler(self.prof)
    # =========================================================================
    # Worker Serving
    # =========================================================================

    def serve_worker(self):
        """
        供当前 TP 组内的 worker 进程调用：阻塞等待本组 leader 的指令并参与推理。
        """
        assert not self.is_tp_leader, "serve_worker() should only be called on TP worker ranks."
        self._loop_running = True
        self._inference_loop()

    # =========================================================================
    # TP Tensor Broadcast Helpers
    # =========================================================================

    def _tp_broadcast_decode_tensors(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_indices: torch.Tensor,
    ):
        """
        Rank 0 广播 decode 需要的张量给所有 worker。
        """
        broadcast_tensor(input_ids, src=0)
        broadcast_tensor(positions, src=0)
        broadcast_tensor(slot_indices, src=0)

    def _tp_recv_decode_tensors(self, batch_size: int):
        """
        Worker 接收 decode 张量（使用预分配 buffer）。
        """
        input_ids = self._decode_input_ids[:batch_size]
        positions = self._decode_positions[:batch_size]
        slot_indices = torch.empty(batch_size, dtype=torch.long, device=self.device)

        broadcast_tensor(input_ids, src=0)
        broadcast_tensor(positions, src=0)
        broadcast_tensor(slot_indices, src=0)
        return input_ids, positions, slot_indices

    def _tp_broadcast_prefill_tensors(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_indices: torch.Tensor,
    ):
        """
        Rank 0 广播 prefill 的 padded batch tensors。
        """
        broadcast_tensor(input_ids, src=0)
        broadcast_tensor(position_ids, src=0)
        broadcast_tensor(attention_mask, src=0)
        broadcast_tensor(seq_lens, src=0)
        broadcast_tensor(slot_indices, src=0)

    def _tp_recv_prefill_tensors(self, batch_size: int, max_len: int):
        """
        Worker 接收 prefill 的 padded batch tensors。
        """
        input_ids = torch.empty(batch_size, max_len, dtype=torch.long, device=self.device)
        position_ids = torch.empty(batch_size, max_len, dtype=torch.long, device=self.device)
        attention_mask = torch.empty(batch_size, max_len, dtype=torch.bool, device=self.device)
        seq_lens = torch.empty(batch_size, dtype=torch.int32, device=self.device)
        slot_indices = torch.empty(batch_size, dtype=torch.long, device=self.device)

        broadcast_tensor(input_ids, src=0)
        broadcast_tensor(position_ids, src=0)
        broadcast_tensor(attention_mask, src=0)
        broadcast_tensor(seq_lens, src=0)
        broadcast_tensor(slot_indices, src=0)
        return input_ids, position_ids, attention_mask, seq_lens, slot_indices

    # =========================================================================
    # Core Inference
    # =========================================================================
    def _prefill_flat_local(self, sequences: list[list[int]], slot_indices: list[int]) -> torch.Tensor:
        """
        TP flat+varlen prefill (all ranks run this).
        - No padding
        - Writes KV cache
        - Uses flash_attn_varlen_func
        - Returns logits only meaningful on rank0
        """
        with self.prof.section(f"rank{self.tp_rank}.prefill_flat.total"):
            return self._prefill_flat_local_impl(sequences, slot_indices)

    def _prefill_flat_local_impl(self, sequences: list[list[int]], slot_indices: list[int]) -> torch.Tensor:
        if self.is_tp_leader:
            print

        num_seqs = len(sequences)

        with self.prof.section(f"rank{self.tp_rank}.prefill_flat.build_meta"):
            seq_lens = [len(s) for s in sequences]
            total_tokens = sum(seq_lens)
            max_seqlen_in_batch = max(seq_lens)

        with self.prof.section(f"rank{self.tp_rank}.prefill_flat.build_flat_tokens"):
            flat_tokens = torch.tensor(
                [t for s in sequences for t in s],
                dtype=torch.long,
                device=self.device
            )
            seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)

            cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=self.device)
            cu_seqlens[1:] = seq_lens_t.cumsum(0)

            positions = torch.cat([torch.arange(l, device=self.device) for l in seq_lens])

            slot_indices_t = torch.tensor(slot_indices, dtype=torch.long, device=self.device)
            scatter_indices = torch.repeat_interleave(
                slot_indices_t * self.max_seq_len,
                seq_lens_t.long()
            ) + positions

        with self.prof.section(f"rank{self.tp_rank}.prefill_flat.embed"):
            hidden = self.model.model.embed_tokens(flat_tokens)

        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn

            # ---- attention block ----
            with self.prof.section(f"rank{self.tp_rank}.layer{layer_idx}.attn.input_ln"):
                residual = hidden
                hidden = layer.input_layernorm(hidden)

            with self.prof.section(f"rank{self.tp_rank}.layer{layer_idx}.attn.qkv_proj"):
                q, k, v = attn._qkv_proj(hidden)

            with self.prof.section(f"rank{self.tp_rank}.layer{layer_idx}.attn.qk_norm"):
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            with self.prof.section(f"rank{self.tp_rank}.layer{layer_idx}.attn.rope"):
                q, k = attn.rotary_emb(
                    q.unsqueeze(0),
                    k.unsqueeze(0),
                    positions.unsqueeze(0)
                )
                q = q.squeeze(0)
                k = k.squeeze(0)

            with self.prof.section(f"rank{self.tp_rank}.layer{layer_idx}.attn.kv_cache_write"):
                kv_flat = attn._kv_cache.view(-1, attn.num_kv_heads_per_partition, attn.head_dim)
                v_flat = attn._v_cache.view(-1, attn.num_kv_heads_per_partition, attn.head_dim)

                if k.dtype != kv_flat.dtype:
                    k = k.to(kv_flat.dtype)
                if v.dtype != v_flat.dtype:
                    v = v.to(v_flat.dtype)
                if q.dtype != kv_flat.dtype:
                    q = q.to(kv_flat.dtype)

                kv_flat.index_copy_(0, scatter_indices, k)
                v_flat.index_copy_(0, scatter_indices, v)

                attn._cache_seqlens.index_copy_(0, slot_indices_t, seq_lens_t)

            with self.prof.section(f"rank{self.tp_rank}.layer{layer_idx}.attn.flash_varlen"):
                attn_out = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens, cu_seqlens,
                    max_seqlen_in_batch, max_seqlen_in_batch,
                    causal=True,
                    softmax_scale=attn.scaling,
                )

            with self.prof.section(f"rank{self.tp_rank}.layer{layer_idx}.attn.o_proj"):
                hidden = attn.o_proj(attn_out.reshape(total_tokens, -1))

            with self.prof.section(f"rank{self.tp_rank}.layer{layer_idx}.attn.residual_add"):
                hidden = residual + hidden

            # ---- mlp block ----
            with self.prof.section(f"rank{self.tp_rank}.layer{layer_idx}.mlp.pre_ln"):
                residual = hidden
                hidden = layer.post_attention_layernorm(hidden)

            with self.prof.section(f"rank{self.tp_rank}.layer{layer_idx}.mlp.forward"):
                hidden = layer.mlp(hidden)

            with self.prof.section(f"rank{self.tp_rank}.layer{layer_idx}.mlp.residual_add"):
                hidden = residual + hidden

        with self.prof.section(f"rank{self.tp_rank}.prefill_flat.final_norm"):
            hidden = self.model.model.norm(hidden)

        with self.prof.section(f"rank{self.tp_rank}.prefill_flat.select_last_hidden"):
            last_token_indices = cu_seqlens[1:] - 1
            last_hidden = hidden[last_token_indices]

        if self.is_rank_0:
            with self.prof.section("rank0.prefill_flat.lm_head"):
                if self.model.lm_head is None:
                    logits = F.linear(last_hidden, self.model.model.embed_tokens.weight)
                else:
                    logits = self.model.lm_head(last_hidden)
        else:
            with self.prof.section(f"rank{self.tp_rank}.prefill_flat.dummy_logits"):
                logits = torch.empty(
                    num_seqs, self.config.vocab_size,
                    device=self.device,
                    dtype=hidden.dtype
                )

        return logits
        
    def _prefill_padded_local(
        self,
        input_ids: torch.Tensor,      # [B, T]
        position_ids: torch.Tensor,   # [B, T]
        attention_mask: torch.Tensor, # [B, T] bool, True=valid
        seq_lens: torch.Tensor,       # [B] int32
        slot_indices: torch.Tensor,   # [B] long
    ) -> torch.Tensor:
        """
        Padded-batch prefill (all ranks run this).
        这是 TP MVP 的稳定版本，避免 packed varlen + QKVParallelLinear(2D/3D) shape 坑。

        返回：
            rank0: [B, vocab]
            worker: dummy tensor [B, vocab]（占位）
        """
        B, T = input_ids.shape
        device = self.device

        # Forward through full model (prefill path)
        # 你的 model_tp.forward 会在 slot_indices 不为 None 且 seq_len>1 时走 prefill+cache 写入逻辑
        logits = self.model(input_ids, position_ids, slot_indices)  # [B, T, vocab] on rank0 / dummy on workers

        # 取每个样本最后一个有效 token 的 logits
        # seq_lens 是长度，最后有效位置 = seq_lens - 1
        last_pos = (seq_lens.to(torch.long) - 1).clamp(min=0)
        batch_idx = torch.arange(B, device=device)

        # rank0有效，worker是dummy但形状匹配
        last_logits = logits[batch_idx, last_pos, :]
        return last_logits

    def _prefill(self, sequences: list[list[int]], slot_indices: list[int]) -> torch.Tensor:
        """
        TP-aware prefill wrapper（所有 rank都会调用）
        当前实现：构造成 padded batch 后调用 _prefill_padded_local。
        """
        batch_size = len(sequences)
        max_len = max(len(s) for s in sequences)
        pad_id = self.config.eos_token_id  # 仅作为padding占位；attention_mask会屏蔽

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=self.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        seq_lens = torch.tensor([len(s) for s in sequences], dtype=torch.int32, device=self.device)
        slot_indices_t = torch.tensor(slot_indices, dtype=torch.long, device=self.device)

        for i, seq in enumerate(sequences):
            L = len(seq)
            input_ids[i, :L] = torch.tensor(seq, dtype=torch.long, device=self.device)
            position_ids[i, :L] = torch.arange(L, dtype=torch.long, device=self.device)
            attention_mask[i, :L] = True

        return self._prefill_padded_local(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            seq_lens=seq_lens,
            slot_indices=slot_indices_t,
        )

    def _decode_step(
        self,
        input_ids: torch.Tensor,      # [B,1]
        positions: torch.Tensor,      # [B,1]
        slot_indices: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        """
        Decode step with TP support.
        所有 rank 执行，但只有 rank0 logits 有效（取决于你的 model_tp 实现）。
        """
        logits = self.model(input_ids, positions, slot_indices)
        return logits[:, 0, :]  # [B, vocab]

    def _sample_tokens(self, logits: torch.Tensor, temperature: float) -> list[int]:
        """
        只在 rank0 调用。
        """
        if temperature == 0:
            return logits.argmax(dim=-1).tolist()
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1).tolist()

    # =========================================================================
    # Public API (rank0 only)
    # =========================================================================

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 100,
        temperature: float = 0.0,
        ignore_eos: bool = False,
    ) -> concurrent.futures.Future:
        assert self.is_tp_leader, "generate() should only be called on the TP-group leader."

        if not self._loop_running:
            self._loop_running = True
            self._loop_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._loop_thread.start()

        future = concurrent.futures.Future()

        token_ids = []
        for p in prompts:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True
            )
            token_ids.append(self.tokenizer.encode(text, add_special_tokens=False))

        request = _Request(
            token_ids=token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=ignore_eos,
            future=future,
            results=[None] * len(prompts),
            pending_indices=list(range(len(prompts))),
        )
        self._request_queue.put(request)
        return future

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 100,
        temperature: float = 0.0,
        ignore_eos: bool = False,
    ) -> concurrent.futures.Future:
        assert self.is_tp_leader, "chat() should only be called on the TP-group leader."

        if not self._loop_running:
            self._loop_running = True
            self._loop_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._loop_thread.start()

        future = concurrent.futures.Future()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = [self.tokenizer.encode(prompt, add_special_tokens=False)]

        request = _Request(
            token_ids=token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=ignore_eos,
            future=future,
            results=[None],
            pending_indices=[0],
        )
        self._request_queue.put(request)
        return future

    def stop(self, timeout: float = 30.0):
        """
        停止当前副本 leader 的调度循环；循环内部会向本 TP 组 worker 广播 exit。
        不在这里 destroy process group，统一由 benchmark/main 最外层收尾。
        """
        if self.is_tp_leader:
            if self._loop_running:
                self._loop_running = False
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=timeout)
                self._loop_thread = None

    # =========================================================================
    # TP Worker Loop
    # =========================================================================

    @torch.inference_mode()
    def _worker_loop(self):
        """
        非 rank0 进程的执行循环：
        等 rank0 的 control，然后接收张量并参与 TP forward。
        """
        print(
            f"[Global rank {self.global_rank} | DP {self.dp_rank} | TP {self.tp_rank}] "
            f"Worker ready, waiting for TP-group leader commands..."
        )
        self.model.clear_all_slots()

        while True:
            control = broadcast_object(None, src=0)
            action = control["action"]

            if action == "exit":
                break

            elif action == "clear_slot":
                slot = int(control["slot"])
                self.model.clear_slot(slot)

            elif action == "clear_all_slots":
                self.model.clear_all_slots()

            elif action == "prefill":
                batch_size = int(control["batch_size"])
                max_len = int(control["max_len"])
                # 接收padded prefill tensors
                input_ids, position_ids, attention_mask, seq_lens, slot_indices = self._tp_recv_prefill_tensors(
                    batch_size, max_len
                )
                _ = self._prefill_padded_local(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    seq_lens=seq_lens,
                    slot_indices=slot_indices,
                )

            elif action == "prefill_flat":
                with self.prof.section(f"rank{self.tp_rank}.prefill_flat.recv_control"):
                    sequences = control["sequences"]
                    slots = control["slots"]

                with self.prof.section(f"rank{self.tp_rank}.prefill_flat.model_forward"):
                    _ = self._prefill_flat_local(sequences, slots)
            
            elif action == "decode":
                batch_size = int(control["batch_size"])
                input_ids, positions, slot_indices = self._tp_recv_decode_tensors(batch_size)
                _ = self._decode_step(input_ids, positions, slot_indices)

            else:
                raise RuntimeError(f"[Rank {self.tp_rank}] Unknown action: {action}")

    # =========================================================================
    # Main Inference Loop (rank0 drives; workers follow commands)
    # =========================================================================

    @torch.inference_mode()
    def _inference_loop(self):
        """
        TP-aware inference loop.

        Rank 0:
          - scheduling, sampling, state updates
          - broadcasts control + input tensors
        Worker ranks:
          - wait for commands and participate in forward
        """
        # Worker ranks do not schedule; they just follow commands.
        if not self.is_rank_0:
            self._worker_loop()
            return

        free_slots = list(range(self.max_num_seqs))
        active_generations = {}  # slot -> (req, prompt_idx, tokens, pos)
        pending_requests = []

        total_prompts = 0
        decode_times = deque(maxlen=16)
        pbar = None

        self.model.clear_all_slots()
        if is_distributed():
            broadcast_object({"action": "clear_all_slots"}, src=0)

        try:
            while self._loop_running or pending_requests or active_generations:
                # ------------------------------------------------------------------
                # Phase 1: Drain queue
                # ------------------------------------------------------------------
                while True:
                    try:
                        req = self._request_queue.get_nowait()
                    except queue.Empty:
                        break

                    pending_requests.append(req)
                    total_prompts += len(req.token_ids)
                    if pbar is None:
                        pbar = tqdm(total=total_prompts, desc="Generating", unit="req", ncols=100)
                    else:
                        pbar.total = total_prompts
                        pbar.refresh()

                # ------------------------------------------------------------------
                # Idle state
                # ------------------------------------------------------------------
                if not pending_requests and not active_generations:
                    if not self._loop_running:
                        break
                    time.sleep(0.001)
                    continue

                # ------------------------------------------------------------------
                # Phase 2: Assign slots
                # ------------------------------------------------------------------
                new_work = []
                for req in list(pending_requests):
                    while req.pending_indices and free_slots:
                        new_work.append((req, req.pending_indices.pop(0), free_slots.pop(0)))
                    if not req.pending_indices:
                        pending_requests.remove(req)

                # ------------------------------------------------------------------
                # Phase 3: Prefill (flat)
                # ------------------------------------------------------------------
                if new_work:
                    with self.prof.section("rank0.prefill_flat.prepare_batch"):
                        seqs = [r.token_ids[i] for r, i, _ in new_work]
                        slots = [s for _, _, s in new_work]

                        # 可选：记一些规模信息，后面打印吞吐更方便
                        prefill_batch_size = len(seqs)
                        prefill_prompt_tokens = sum(len(s) for s in seqs)
                        self.prof.add("metric.prefill_flat.batch_size", float(prefill_batch_size))
                        self.prof.add("metric.prefill_flat.prompt_tokens", float(prefill_prompt_tokens))

                    # 通知 worker 参与 flat prefill
                    if is_distributed():
                        with self.prof.section("rank0.prefill_flat.broadcast_control"):
                            broadcast_object(
                                {"action": "prefill_flat", "sequences": seqs, "slots": slots},
                                src=0
                            )

                    # Rank0 本地也执行同样 prefill
                    with self.prof.section("rank0.prefill_flat.model_forward"):
                        logits = self._prefill_flat_local(seqs, slots)

                    with self.prof.section("rank0.prefill_flat.sample"):
                        sampled = self._sample_tokens(logits, new_work[0][0].temperature)

                    with self.prof.section("rank0.prefill_flat.postprocess"):
                        for idx, (req, prompt_idx, slot) in enumerate(new_work):
                            tok = sampled[idx]
                            prompt_len = len(req.token_ids[prompt_idx])

                            if 1 >= req.max_tokens or (not req.ignore_eos and tok == self.config.eos_token_id):
                                text = self.tokenizer.decode([tok], skip_special_tokens=True)
                                req.results[prompt_idx] = GenerationOutput(
                                    text=text,
                                    token_ids=[tok],
                                    prompt_tokens=prompt_len,
                                    generated_tokens=1,
                                )

                                with self.prof.section("rank0.prefill_flat.clear_slot_local"):
                                    self.model.clear_slot(slot)

                                if is_distributed():
                                    with self.prof.section("rank0.prefill_flat.clear_slot_broadcast"):
                                        broadcast_object({"action": "clear_slot", "slot": int(slot)}, src=0)

                                bisect.insort(free_slots, slot)
                                if pbar:
                                    pbar.update(1)
                                if all(r is not None for r in req.results):
                                    req.future.set_result(req.results)
                            else:
                                # 下一步 decode 的 position = prompt_len
                                active_generations[slot] = (req, prompt_idx, [tok], prompt_len)
                if not active_generations:
                    continue

                # ------------------------------------------------------------------
                # Phase 4: Decode step
                # ------------------------------------------------------------------
                slots = list(active_generations.keys())
                batch_size = len(slots)

                input_ids = self._decode_input_ids[:batch_size]
                positions = self._decode_positions[:batch_size]

                with self.prof.section("rank0.decode.prepare_inputs"):
                    for i, slot in enumerate(slots):
                        req, prompt_idx, tokens, pos = active_generations[slot]
                        input_ids[i, 0] = tokens[-1]
                        positions[i, 0] = pos

                slot_indices = torch.tensor(slots, dtype=torch.long, device=self.device)
                temperature = active_generations[slots[0]][0].temperature

                if is_distributed():
                    with self.prof.section("rank0.decode.broadcast_control"):
                        broadcast_object({"action": "decode", "batch_size": batch_size}, src=0)
                    with self.prof.section("rank0.decode.broadcast_tensors"):
                        self._tp_broadcast_decode_tensors(input_ids, positions, slot_indices)

                t0 = time.perf_counter()

                with self.prof.section("rank0.decode.model_forward"):
                    logits = self._decode_step(input_ids, positions, slot_indices)

                with self.prof.section("rank0.decode.sample"):
                    sampled = self._sample_tokens(logits, temperature)

                decode_times.append((batch_size, time.perf_counter() - t0))

                if pbar and decode_times:
                    tok_per_sec = sum(n for n, _ in decode_times) / max(sum(t for _, t in decode_times), 1e-9)
                    pbar.set_postfix_str(f"batch={batch_size} tok/s={tok_per_sec:.0f}")

                # ------------------------------------------------------------------
                # Phase 5: Process generated tokens
                # ------------------------------------------------------------------
                for i, slot in enumerate(slots):
                    tok = sampled[i]
                    req, prompt_idx, tokens, pos = active_generations[slot]
                    tokens.append(tok)
                    pos += 1

                    should_stop = (
                        len(tokens) >= req.max_tokens or
                        pos >= self.max_seq_len or
                        (not req.ignore_eos and tok == self.config.eos_token_id)
                    )

                    if should_stop:
                        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                        req.results[prompt_idx] = GenerationOutput(
                            text=text,
                            token_ids=tokens,
                            prompt_tokens=len(req.token_ids[prompt_idx]),
                            generated_tokens=len(tokens),
                        )

                        del active_generations[slot]

                        self.model.clear_slot(slot)
                        if is_distributed():
                            broadcast_object({"action": "clear_slot", "slot": int(slot)}, src=0)

                        bisect.insort(free_slots, slot)
                        if pbar:
                            pbar.update(1)

                        if all(r is not None for r in req.results):
                            req.future.set_result(req.results)
                    else:
                        active_generations[slot] = (req, prompt_idx, tokens, pos)

            # normal shutdown: tell workers to exit
            if is_distributed():
                broadcast_object({"action": "exit"}, src=0)

        except Exception:
            # prevent workers from hanging forever on broadcast
            if is_distributed():
                try:
                    broadcast_object({"action": "exit"}, src=0)
                except Exception:
                    pass
            raise
        finally:
            if pbar:
                pbar.close()


# ============================================================================
# Example usage (torchrun)
# ============================================================================

if __name__ == "__main__":
    """
    运行示例：
      torchrun --nproc_per_node=2 llm_tp.py
    """
    MODEL_PATH = "/mnt/data0/Qwen30.6B"

    llm = LLMTP(MODEL_PATH, max_num_seqs=8, max_seq_len=1024, enable_tp=True)

    if llm.is_tp_leader:
        prompts = [
            f"[replica {llm.dp_rank}] Hello, how are you?",
            f"[replica {llm.dp_rank}] What is 2+2?",
            f"[replica {llm.dp_rank}] Write a haiku about AI.",
        ]
        fut = llm.generate(prompts, max_tokens=32, temperature=0.0)
        outs = fut.result()
        for i, o in enumerate(outs):
            print(f"\n[global_rank={llm.global_rank}][{i}] {o.text}")
        llm.stop()
    else:
        llm.serve_worker()
            