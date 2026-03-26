# pd/decode_server.py
import os
import time
import queue
import threading
import socket
from dataclasses import dataclass
from typing import Optional
import traceback

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from pd.common import SamplingParams, KVHandle
from pd.ipc_protocol import send_obj, recv_obj
from pd.kv_connector_shm import SharedMemoryKVConnector

from model.model import Qwen3Config, Qwen3ForCausalLM


@dataclass
class DecodeState:
    request_id: str
    local_slot: int
    prompt_token_ids: list[int]
    prompt_len: int
    sampling_params: SamplingParams
    generated_token_ids: list[int]
    current_position: int
    finished: bool = False
    finish_reason: Optional[str] = None


class DecodeServer:
    def __init__(
        self,
        server_id: str,
        host: str,
        port: int,
        model_path: str,
        device: str,
        kv_base_dir: str | None = None,   # 兼容旧 launch_pd.py，当前 shm 版不使用
        max_num_seqs: int = 64,
        max_seq_len: int = 2048,
        dtype=torch.bfloat16,
        idle_sleep_ms: float = 1.0,
    ):
        self.server_id = server_id
        self.host = host
        self.port = port
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_num_seqs = max_num_seqs
        self.max_seq_len = max_seq_len
        self.idle_sleep_ms = idle_sleep_ms

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

        self._free_slots = list(range(self.max_num_seqs))
        self._pending_attach: "queue.Queue[dict]" = queue.Queue()
        self._active: dict[str, DecodeState] = {}
        self._finished: dict[str, dict] = {}

        self._running = False
        self._decode_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _alloc_slot(self) -> int:
        if not self._free_slots:
            raise RuntimeError("DecodeServer has no free slots")
        return self._free_slots.pop(0)

    def _free_slot(self, slot: int):
        self.model.clear_slot(slot)
        self._free_slots.append(slot)
        self._free_slots.sort()

    def _sample_one(self, logits_1row: torch.Tensor, temperature: float) -> int:
        if temperature == 0.0:
            return int(logits_1row.argmax(dim=-1).item())
        probs = F.softmax(logits_1row / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _try_attach_sessions(self):
        while self._free_slots:
            try:
                msg = self._pending_attach.get_nowait()
            except queue.Empty:
                break

            request_id = msg["request_id"]
            prompt_token_ids = msg["prompt_token_ids"]
            prompt_len = int(msg["prompt_len"])
            sampling_params = SamplingParams(**msg["sampling_params"])
            kv_handle_dict = msg["kv_handle"]

            slot = self._alloc_slot()
            try:
                handle = KVHandle(**kv_handle_dict)

                # load_kv 会 copy 出 CPU tensor，所以后面可以立刻 cleanup shm
                payload = self.connector.load_kv(handle)
                self.model.import_request_kv(slot_idx=slot, payload=payload, length=prompt_len)

                # 无 cleanup RPC 时，consumer 导入成功后立即释放共享内存
                self.connector.cleanup(handle)

                self._active[request_id] = DecodeState(
                    request_id=request_id,
                    local_slot=slot,
                    prompt_token_ids=prompt_token_ids,
                    prompt_len=prompt_len,
                    sampling_params=sampling_params,
                    generated_token_ids=[],
                    current_position=prompt_len,
                )
            except Exception:
                self._log("Exception inside _try_attach_sessions:")
                self._log(traceback.format_exc())
                self._free_slot(slot)
                raise

    @torch.inference_mode()
    def _bootstrap_or_step_batch(self):
        if not self._active:
            return

        req_ids = list(self._active.keys())
        batch_size = len(req_ids)

        input_ids = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
        positions = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
        slot_indices = torch.empty((batch_size,), dtype=torch.long, device=self.device)

        for i, rid in enumerate(req_ids):
            st = self._active[rid]
            if len(st.generated_token_ids) == 0:
                input_tok = st.prompt_token_ids[-1]
                pos = st.prompt_len - 1
            else:
                input_tok = st.generated_token_ids[-1]
                pos = st.current_position

            input_ids[i, 0] = input_tok
            positions[i, 0] = pos
            slot_indices[i] = st.local_slot

        logits = self.model(input_ids, positions, slot_indices)[:, 0, :]

        done_req_ids = []
        for i, rid in enumerate(req_ids):
            st = self._active[rid]
            tok = self._sample_one(logits[i:i+1], st.sampling_params.temperature)
            st.generated_token_ids.append(tok)

            if len(st.generated_token_ids) == 1:
                st.current_position = st.prompt_len
            else:
                st.current_position += 1

            should_stop = (
                len(st.generated_token_ids) >= st.sampling_params.max_tokens
                or st.current_position >= self.max_seq_len
                or (
                    not st.sampling_params.ignore_eos
                    and tok == self.config.eos_token_id
                )
            )

            if should_stop:
                st.finished = True
                if tok == self.config.eos_token_id:
                    st.finish_reason = "eos"
                elif st.current_position >= self.max_seq_len:
                    st.finish_reason = "length_cap"
                else:
                    st.finish_reason = "max_tokens"

                text = self.tokenizer.decode(st.generated_token_ids, skip_special_tokens=True)
                self._finished[rid] = {
                    "request_id": rid,
                    "text": text,
                    "token_ids": st.generated_token_ids,
                    "prompt_tokens": st.prompt_len,
                    "generated_tokens": len(st.generated_token_ids),
                    "finish_reason": st.finish_reason,
                }
                done_req_ids.append(rid)

        for rid in done_req_ids:
            st = self._active.pop(rid)
            self._free_slot(st.local_slot)

    def _decode_loop(self):
        while self._running:
            try:
                with self._lock:
                    self._try_attach_sessions()
                    self._bootstrap_or_step_batch()
            except Exception as e:
                print(f"[DecodeServer:{self.server_id}] decode loop error: {e}")
                time.sleep(0.05)

            if not self._active and self._pending_attach.empty():
                time.sleep(self.idle_sleep_ms / 1000.0)

    def _handle_client(self, conn: socket.socket):
        try:
            msg = recv_obj(conn)
            op = msg.get("op")

            if op == "health":
                send_obj(conn, {
                    "ok": True,
                    "server_id": self.server_id,
                    "role": "decode",
                    "active": len(self._active),
                    "free_slots": len(self._free_slots),
                })
                return

            if op == "attach":
                with self._lock:
                    if len(self._active) + self._pending_attach.qsize() >= self.max_num_seqs:
                        send_obj(conn, {"ok": False, "error": "decode server is full"})
                        return
                    self._pending_attach.put(msg["data"])
                send_obj(conn, {"ok": True, "queued": True})
                return

            if op == "poll":
                request_id = msg["request_id"]
                with self._lock:
                    if request_id in self._finished:
                        out = self._finished.pop(request_id)
                        send_obj(conn, {"ok": True, "ready": True, "data": out})
                        return

                    if request_id in self._active:
                        send_obj(conn, {"ok": True, "ready": False, "state": "active"})
                        return

                    send_obj(conn, {"ok": True, "ready": False, "state": "pending"})
                return

            send_obj(conn, {"ok": False, "error": f"unknown op={op}"})
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Exception inside _handle_client:")
            self._log(tb)
            send_obj(conn, {
                "ok": False,
                "error": repr(e),
                "traceback": tb,
            })
        finally:
            conn.close()

    def serve_forever(self):
        self._running = True
        self._decode_thread = threading.Thread(target=self._decode_loop, daemon=True)
        self._decode_thread.start()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(128)

        print(f"[DecodeServer:{self.server_id}] listening on {self.host}:{self.port}, device={self.device}")

        try:
            while True:
                conn, _ = server.accept()
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
        finally:
            self._running = False
            server.close()