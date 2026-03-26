# pd/common.py
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, Any


def gen_request_id() -> str:
    return str(uuid.uuid4())


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0
    ignore_eos: bool = False


@dataclass
class RequestMessage:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams


@dataclass
class KVHandle:
    """
    共享内存版 KV 句柄。
    layer_metas 里描述每层 K/V 的 shm 名称、shape、dtype、seqlen。
    """
    request_id: str
    producer_id: str
    num_layers: int
    num_tokens: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    layer_metas: list[dict]
    created_at: float = field(default_factory=time.time)


@dataclass
class PrefillDoneMessage:
    request_id: str
    prompt_token_ids: list[int]
    prompt_len: int
    sampling_params: SamplingParams
    kv_handle: KVHandle


@dataclass
class TokenPiece:
    request_id: str
    token_id: int
    text_piece: str
    finished: bool
    finish_reason: Optional[str] = None


@dataclass
class FinalResult:
    request_id: str
    text: str
    token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    finish_reason: Optional[str]


def dataclass_to_dict(obj: Any):
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj