# pd/prefill_server.py
import os
import time
import queue
import threading
import socket
from concurrent.futures import Future
from typing import Optional

import torch
from transformers import AutoTokenizer
from flash_attn import flash_attn_varlen_func

from pd.ipc_protocol import send_obj, recv_obj
from pd.kv_connector_shm import SharedMemoryKVConnector

from model.model import Qwen3Config, Qwen3ForCausalLM


class PrefillServer:
    def __init__(
        self,
        server_id: str,
        host: str,
        port: int,
        model_path: str,
        device: str,
        kv_base_dir: str | None = None,   # 兼容旧 launch_pd.py，当前 shm 版不使用
        max_num_seqs: int = 32,
        max_seq_len: int = 2048,
        dtype=torch.bfloat16,
        batch_window_ms: float = 5.0,
    ):
        self.server_id = server_id
        self.host = host
        self.port = port
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_num_seqs = max_num_seqs
        self.max_seq_len = max_seq_len
        self.batch_window_ms = batch_window_ms

        # 共享内存版 connector，不依赖 kv_base_dir
        self.connector = SharedMemoryKVConnector(prefix="pd_kv")

        config_path = os.path.join(model_path, "config.json")
        self.config = Qwen3Config.from_json(config_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = Qwen3ForCausalLM.from_pretrained(model_path, self.config, self.device, dtype)
        self.model.eval()
        print("Fusing QKV projections...")
        self.model.fuse_qkv()
        self.model.init_kv_cache(self.max_num_seqs, self.max_seq_len, self.device, dtype)

        self._req_queue: "queue.Queue[tuple[dict, Future]]" = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        self._free_slots = list(range(self.max_num_seqs))

    def _alloc_slot(self) -> int:
        if not self._free_slots:
            raise RuntimeError("PrefillServer has no free slots")
        return self._free_slots.pop(0)

    def _free_slot(self, slot: int):
        self.model.clear_slot(slot)
        self._free_slots.append(slot)
        self._free_slots.sort()

    @torch.inference_mode()
    def _prefill_flat_batch(self, reqs: list[dict]) -> list[dict]:
        """
        flat packed prefill:
        - 无 padding
        - 用 flash_attn_varlen_func
        - 只负责写 KV，不做 first-token sampling
        """
        slots = []
        try:
            for _ in reqs:
                slots.append(self._alloc_slot())

            sequences = [r["prompt_token_ids"] for r in reqs]
            seq_lens = [len(s) for s in sequences]

            if any(L > self.max_seq_len for L in seq_lens):
                raise ValueError(
                    f"Some prompt length exceeds max_seq_len={self.max_seq_len}, got max={max(seq_lens)}"
                )

            num_seqs = len(sequences)
            total_tokens = sum(seq_lens)
            max_seqlen_in_batch = max(seq_lens)

            # ------------------------------------------------------------------
            # build packed metadata
            # ------------------------------------------------------------------
            flat_tokens = torch.tensor(
                [t for seq in sequences for t in seq],
                dtype=torch.long,
                device=self.device,
            )  # [N]

            seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)  # [B]

            cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=self.device)
            cu_seqlens[1:] = seq_lens_t.cumsum(0)

            positions = torch.cat(
                [torch.arange(L, dtype=torch.long, device=self.device) for L in seq_lens],
                dim=0,
            )  # [N]

            slot_indices_t = torch.tensor(slots, dtype=torch.long, device=self.device)  # [B]

            # 每个 token 写到对应 slot 的对应位置
            scatter_indices = torch.repeat_interleave(
                slot_indices_t * self.max_seq_len,
                seq_lens_t.long(),
            ) + positions  # [N]

            # ------------------------------------------------------------------
            # embedding
            # ------------------------------------------------------------------
            hidden = self.model.model.embed_tokens(flat_tokens)  # [N, hidden]

            # ------------------------------------------------------------------
            # layer-by-layer packed prefill
            # ------------------------------------------------------------------
            for layer in self.model.model.layers:
                attn = layer.self_attn

                # ---- attention block ----
                residual = hidden
                hidden = layer.input_layernorm(hidden)

                qkv = attn._qkv_proj(hidden)
                q = qkv[..., :attn._q_size].view(total_tokens, attn.num_heads, attn.head_dim)
                k = qkv[..., attn._q_size:attn._q_size + attn._kv_size].view(total_tokens, attn.num_kv_heads, attn.head_dim)
                v = qkv[..., attn._q_size + attn._kv_size:].view(total_tokens, attn.num_kv_heads, attn.head_dim)
                # 这里默认 fused qkv 输出：
                # q: [N, num_heads, head_dim]
                # k,v: [N, num_kv_heads, head_dim]
                # 如果你实际 shape 不同，在这里加 reshape

                q = attn.q_norm(q)
                k = attn.k_norm(k)

                q, k = attn.rotary_emb(
                    q.unsqueeze(0),
                    k.unsqueeze(0),
                    positions.unsqueeze(0),
                )
                q = q.squeeze(0)
                k = k.squeeze(0)

                kv_flat = attn._kv_cache.view(-1, attn.num_kv_heads, attn.head_dim)
                v_flat = attn._v_cache.view(-1, attn.num_kv_heads, attn.head_dim)

                if k.dtype != kv_flat.dtype:
                    k = k.to(kv_flat.dtype)
                if v.dtype != v_flat.dtype:
                    v = v.to(v_flat.dtype)
                if q.dtype != kv_flat.dtype:
                    q = q.to(kv_flat.dtype)

                kv_flat.index_copy_(0, scatter_indices, k)
                v_flat.index_copy_(0, scatter_indices, v)
                attn._cache_seqlens.index_copy_(0, slot_indices_t, seq_lens_t)

                attn_out = flash_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen_in_batch,
                    max_seqlen_k=max_seqlen_in_batch,
                    causal=True,
                    softmax_scale=attn.scaling,
                )

                hidden = attn.o_proj(attn_out.reshape(total_tokens, -1))
                hidden = residual + hidden

                # ---- mlp block ----
                residual = hidden
                hidden = layer.post_attention_layernorm(hidden)
                hidden = layer.mlp(hidden)
                hidden = residual + hidden

            # ------------------------------------------------------------------
            # export per-request KV
            # ------------------------------------------------------------------
            results = []
            for req, slot, prompt_len in zip(reqs, slots, seq_lens):
                payload = self.model.export_request_kv(slot_idx=slot, length=prompt_len)

                handle = self.connector.save_kv(
                    producer_id=self.server_id,
                    request_id=req["request_id"],
                    payload=payload,
                    num_layers=self.config.num_hidden_layers,
                    num_tokens=prompt_len,
                    num_kv_heads=self.config.num_key_value_heads,
                    head_dim=self.config.head_dim,
                    dtype=str(self.dtype).replace("torch.", ""),
                )

                results.append({
                    "request_id": req["request_id"],
                    "prompt_token_ids": req["prompt_token_ids"],
                    "prompt_len": prompt_len,
                    "sampling_params": req["sampling_params"],
                    "kv_handle": {
                        "request_id": handle.request_id,
                        "producer_id": handle.producer_id,
                        "num_layers": handle.num_layers,
                        "num_tokens": handle.num_tokens,
                        "num_kv_heads": handle.num_kv_heads,
                        "head_dim": handle.head_dim,
                        "dtype": handle.dtype,
                        "layer_metas": handle.layer_metas,
                        "created_at": handle.created_at,
                    },
                })

            return results
        except Exception:
            import traceback
            print(f"[PrefillServer:{self.server_id}] EXCEPTION in _prefill_batch", flush=True)
            traceback.print_exc()
            raise
        finally:
            # producer 本地 slot 立即释放；共享内存由 consumer 侧 cleanup
            for slot in slots:
                self._free_slot(slot)

    def _batch_loop(self):
        while self._running:
            try:
                item = self._req_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch_items = [item]
            t0 = time.perf_counter()

            while len(batch_items) < self.max_num_seqs:
                remain = self.batch_window_ms / 1000.0 - (time.perf_counter() - t0)
                if remain <= 0:
                    break
                try:
                    nxt = self._req_queue.get(timeout=remain)
                    batch_items.append(nxt)
                except queue.Empty:
                    break

            reqs = [x[0] for x in batch_items]
            futs = [x[1] for x in batch_items]

            try:
                outs = self._prefill_flat_batch(reqs)
                for fut, out in zip(futs, outs):
                    fut.set_result(out)
            except Exception as e:
                import traceback
                print(f"[PrefillServer:{self.server_id}] EXCEPTION in _batch_loop", flush=True)
                traceback.print_exc()
                for fut in futs:
                    fut.set_exception(e)

    def _handle_client(self, conn: socket.socket):
        try:
            msg = recv_obj(conn)
            op = msg.get("op")

            if op == "health":
                send_obj(conn, {"ok": True, "server_id": self.server_id, "role": "prefill"})
                return

            if op != "prefill":
                send_obj(conn, {"ok": False, "error": f"unknown op={op}"})
                return

            fut = Future()
            self._req_queue.put((msg, fut))
            out = fut.result()
            send_obj(conn, {"ok": True, "data": out})

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[PrefillServer:{self.server_id}] EXCEPTION in _handle_client", flush=True)
            print(tb, flush=True)
            send_obj(conn, {
                "ok": False,
                "error": repr(e),
                "traceback": tb,
            })
        finally:
            conn.close()

    def serve_forever(self):
        self._running = True
        self._worker_thread = threading.Thread(target=self._batch_loop, daemon=True)
        self._worker_thread.start()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(128)

        print(f"[PrefillServer:{self.server_id}] listening on {self.host}:{self.port}, device={self.device}")

        try:
            while True:
                conn, _ = server.accept()
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
        finally:
            self._running = False
            server.close()