# kv_transfer/types.py
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class KVHandle:
    """
    Prefill 阶段产出的可迁移 KV 引用。
    当前 MVP 里 connector_meta 里只放本地 registry key。
    后续可扩展为：
      - shm path
      - cuda ipc handle
      - nccl peer meta
      - page table metadata
    """
    request_id: str
    producer_instance_id: str
    slot_index: int
    num_tokens: int
    max_seq_len: int

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str

    connector_name: str
    connector_meta: Optional[Dict[str, Any]] = None