# engine/engine_base.py
import os
import uuid
import torch
from transformers import AutoTokenizer

from engine.cache_manager import SlotCacheManager
from kv_transfer.base import KVConnectorBase


class EngineBase:
    def __init__(
        self,
        model_path: str,
        connector: KVConnectorBase,
        instance_id: str,
        max_num_seqs: int = 32,
        max_seq_len: int = 1024,
        dtype=torch.bfloat16,
        enable_tp: bool = False,
        tp_size: int = 1,
        role: str = "base",
    ):
        self.model_path = model_path
        self.connector = connector
        self.instance_id = instance_id
        self.max_num_seqs = max_num_seqs
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.enable_tp = enable_tp
        self.tp_size = tp_size
        self.role = role

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        self.cache_manager = SlotCacheManager(max_num_seqs=max_num_seqs)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def new_request_id() -> str:
        return str(uuid.uuid4())

    def encode_prompt(self, prompt: str) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)