# kv_transfer/base.py
from abc import ABC, abstractmethod
from kv_transfer.types import KVHandle


class KVConnectorBase(ABC):
    """
    KV 传输抽象层。
    当前只定义 producer / consumer 所需的最小接口。
    """

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def save_kv(
        self,
        kv_handle: KVHandle,
        payload: dict,
    ) -> None:
        """
        producer 调用：保存/注册本次请求的 KV 数据
        payload 格式由 engine/model 层约定
        """
        raise NotImplementedError

    @abstractmethod
    def load_kv(
        self,
        kv_handle: KVHandle,
    ) -> dict:
        """
        consumer 调用：根据 KVHandle 拉取 KV payload
        """
        raise NotImplementedError

    @abstractmethod
    def cleanup(self, kv_handle: KVHandle) -> None:
        """
        producer/consumer 结束后做清理
        """
        raise NotImplementedError