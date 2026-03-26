# scheduler/session.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from kv_transfer.types import KVHandle


@dataclass
class SamplingParams:
    max_tokens: int = 100
    temperature: float = 0.0
    ignore_eos: bool = False


@dataclass
class RequestSession:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams

    prefill_instance_id: Optional[str] = None
    decode_instance_id: Optional[str] = None

    prompt_len: int = 0
    kv_handle: Optional[KVHandle] = None
    prefill_done: bool = False

    generated_token_ids: list[int] = field(default_factory=list)
    next_position: int = 0
    finished: bool = False
    finish_reason: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)