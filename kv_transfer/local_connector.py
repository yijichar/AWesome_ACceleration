# kv_transfer/local_connector.py
import copy
import threading
from kv_transfer.base import KVConnectorBase
from kv_transfer.types import KVHandle


_LOCAL_KV_REGISTRY = {}
_LOCAL_KV_LOCK = threading.Lock()


class LocalRegistryConnector(KVConnectorBase):
    """
    单进程 demo 用的本地 KV connector。
    save_kv 时把 payload 放到全局 registry，
    load_kv 时再取回来。
    """

    def __init__(self):
        self._name = "local_registry"

    def name(self) -> str:
        return self._name

    def build_kv_handle(
        self,
        request_id: str,
        producer_instance_id: str,
        slot_index: int,
        num_tokens: int,
        max_seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: str,
        connector_meta: dict | None = None,
    ) -> KVHandle:
        return KVHandle(
            request_id=request_id,
            producer_instance_id=producer_instance_id,
            slot_index=slot_index,
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            connector_name=self._name,
            connector_meta=connector_meta or {},
        )

    def save_kv(self, kv_handle: KVHandle, payload: dict) -> None:
        key = kv_handle.request_id
        with _LOCAL_KV_LOCK:
            # 深拷贝引用结构；tensor 本身这里不 clone，避免太重
            _LOCAL_KV_REGISTRY[key] = payload

    def load_kv(self, kv_handle: KVHandle) -> dict:
        key = kv_handle.request_id
        with _LOCAL_KV_LOCK:
            if key not in _LOCAL_KV_REGISTRY:
                raise KeyError(f"KV payload for request_id={key} not found in local registry")
            return _LOCAL_KV_REGISTRY[key]

    def cleanup(self, kv_handle: KVHandle) -> None:
        key = kv_handle.request_id
        with _LOCAL_KV_LOCK:
            _LOCAL_KV_REGISTRY.pop(key, None)