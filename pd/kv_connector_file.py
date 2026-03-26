# pd/kv_connector_file.py
import os
import json
import shutil
from pathlib import Path
import torch

from pd.common import KVHandle, ensure_dir


class FileKVConnector:
    """
    单机多进程可用的 KV connector。
    Producer 将每层 K/V 存成 .pt 文件；
    Consumer 再读回来。

    优点：
    - 跨进程稳定
    - 架构清晰
    - 方便调试

    缺点：
    - 不是高性能版本
    - 后续你可替换为 CUDA IPC / P2P
    """

    def __init__(self, base_dir: str = "/dev/shm/pd_kv_cache"):
        self.base_dir = base_dir
        ensure_dir(base_dir)

    def _request_dir(self, producer_id: str, request_id: str) -> str:
        d = os.path.join(self.base_dir, producer_id, request_id)
        ensure_dir(d)
        return d

    def save_kv(
        self,
        producer_id: str,
        request_id: str,
        payload: dict,
        num_layers: int,
        num_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: str,
    ) -> KVHandle:
        req_dir = self._request_dir(producer_id, request_id)

        meta = {
            "num_layers": num_layers,
            "num_tokens": num_tokens,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype": dtype,
        }
        with open(os.path.join(req_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        layers = payload["layers"]
        for i, layer_payload in enumerate(layers):
            torch.save(layer_payload["k"], os.path.join(req_dir, f"layer_{i}_k.pt"))
            torch.save(layer_payload["v"], os.path.join(req_dir, f"layer_{i}_v.pt"))
            with open(os.path.join(req_dir, f"layer_{i}_seqlen.txt"), "w", encoding="utf-8") as f:
                f.write(str(layer_payload["seqlen"]))

        return KVHandle(
            request_id=request_id,
            producer_id=producer_id,
            kv_dir=req_dir,
            num_layers=num_layers,
            num_tokens=num_tokens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

    def load_kv(self, handle: KVHandle) -> dict:
        req_dir = handle.kv_dir
        layers = []
        for i in range(handle.num_layers):
            k = torch.load(os.path.join(req_dir, f"layer_{i}_k.pt"), map_location="cpu")
            v = torch.load(os.path.join(req_dir, f"layer_{i}_v.pt"), map_location="cpu")
            with open(os.path.join(req_dir, f"layer_{i}_seqlen.txt"), "r", encoding="utf-8") as f:
                seqlen = int(f.read().strip())
            layers.append({"k": k, "v": v, "seqlen": seqlen})
        return {"layers": layers}

    def cleanup(self, handle: KVHandle):
        req_dir = handle.kv_dir
        if os.path.isdir(req_dir):
            shutil.rmtree(req_dir, ignore_errors=True)