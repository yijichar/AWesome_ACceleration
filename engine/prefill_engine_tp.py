# engine/prefill_engine_tp.py
from dataclasses import dataclass
import os
import torch
import torch.nn.functional as F

from engine.engine_base import EngineBase
from kv_transfer.types import KVHandle

# 这里直接复用你现有模型实现
from model.model_tp import Qwen3Config, Qwen3ForCausalLMTP


@dataclass
class PrefillOutput:
    request_id: str
    kv_handle: KVHandle
    prompt_len: int


class PrefillEngineTP(EngineBase):
    def __init__(
        self,
        model_path: str,
        connector,
        instance_id: str = "prefill-0",
        max_num_seqs: int = 32,
        max_seq_len: int = 1024,
        dtype=torch.bfloat16,
        enable_tp: bool = False,
        tp_size: int = 1,
    ):
        super().__init__(
            model_path=model_path,
            connector=connector,
            instance_id=instance_id,
            max_num_seqs=max_num_seqs,
            max_seq_len=max_seq_len,
            dtype=dtype,
            enable_tp=enable_tp,
            tp_size=tp_size,
            role="prefill",
        )

        config_path = os.path.join(model_path, "config.json")
        self.config = Qwen3Config.from_json(config_path)

        self.model = Qwen3ForCausalLMTP.from_pretrained(
            model_path=model_path,
            config=self.config,
            device=self.device,
            dtype=dtype,
        )
        self.model.eval()
        self.model.init_kv_cache(self.max_num_seqs, self.max_seq_len, self.device, dtype)

    @torch.inference_mode()
    def prefill(self, request_id: str, prompt_token_ids: list[int]) -> PrefillOutput:
        slot = self.cache_manager.allocate_slot()
        try:
            input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            position_ids = torch.arange(len(prompt_token_ids), dtype=torch.long, device=self.device).unsqueeze(0)
            slot_indices = torch.tensor([slot], dtype=torch.long, device=self.device)

            _ = self.model(input_ids, position_ids, slot_indices)

            prompt_len = len(prompt_token_ids)

            payload = self.model.export_request_kv(
                slot_idx=slot,
                length=prompt_len,
            )

            kv_handle = self.connector.build_kv_handle(
                request_id=request_id,
                producer_instance_id=self.instance_id,
                slot_index=slot,
                num_tokens=prompt_len,
                max_seq_len=self.max_seq_len,
                num_layers=self.config.num_hidden_layers,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                dtype=str(self.dtype).replace("torch.", ""),
                connector_meta={},
            )

            self.connector.save_kv(kv_handle, payload)

            # 注意：producer slot 当前先不立刻 clear
            # 因为真正的远端传输实现里，可能还需要 sender 生命周期
            return PrefillOutput(
                request_id=request_id,
                kv_handle=kv_handle,
                prompt_len=prompt_len,
            )
        except Exception:
            self.model.clear_slot(slot)
            self.cache_manager.free_slot(slot)
            raise

    def release_prefill_kv(self, kv_handle: KVHandle):
        slot = kv_handle.slot_index
        self.model.clear_slot(slot)
        self.cache_manager.free_slot(slot)
        self.connector.cleanup(kv_handle)