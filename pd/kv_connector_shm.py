# pd/kv_connector_shm.py
import uuid
import numpy as np
import torch
from multiprocessing import shared_memory

from pd.common import KVHandle


_TORCH_TO_NUMPY = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.bfloat16: None,   # numpy 不原生支持，下面特殊处理
    torch.int32: np.int32,
    torch.int64: np.int64,
}

_NUMPY_STR_TO_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "uint16": np.uint16,   # 用于承载 bf16 的原始 16bit 数据
    "int32": np.int32,
    "int64": np.int64,
}


def _torch_tensor_to_numpy_exportable(t: torch.Tensor):
    """
    把 CPU tensor 转成可放进 shared memory 的 numpy array。
    对 bf16，转成 uint16 视图保存原始 bit pattern。
    """
    assert t.device.type == "cpu", "save_kv expects CPU tensors"

    if t.dtype == torch.bfloat16:
        # 通过 int16/uint16 承载原始 bf16 bits
        # 先转成 contiguous，再 view 成 uint16
        arr = t.contiguous().view(torch.uint16).numpy()
        dtype_str = "uint16"
        original_torch_dtype = "bfloat16"
        return arr, dtype_str, original_torch_dtype

    np_dtype = _TORCH_TO_NUMPY.get(t.dtype, None)
    if np_dtype is None:
        raise TypeError(f"Unsupported torch dtype for shm export: {t.dtype}")

    arr = t.contiguous().numpy()
    dtype_str = str(arr.dtype)
    original_torch_dtype = str(t.dtype).replace("torch.", "")
    return arr, dtype_str, original_torch_dtype


def _numpy_importable_to_torch(arr: np.ndarray, original_torch_dtype: str) -> torch.Tensor:
    """
    从 numpy array 恢复成 torch tensor。
    对 bf16，从 uint16 bit pattern 恢复。
    """
    if original_torch_dtype == "bfloat16":
        # arr dtype == uint16
        t = torch.from_numpy(arr.view(np.uint16))
        return t.view(torch.bfloat16)

    t = torch.from_numpy(arr)
    if original_torch_dtype == "float16":
        return t.to(torch.float16)
    if original_torch_dtype == "float32":
        return t.to(torch.float32)
    if original_torch_dtype == "int32":
        return t.to(torch.int32)
    if original_torch_dtype == "int64":
        return t.to(torch.int64)

    raise TypeError(f"Unsupported original_torch_dtype={original_torch_dtype}")


class SharedMemoryKVConnector:
    """
    单机多进程共享内存版 KV connector。

    save_kv:
      payload["layers"][i]["k"], ["v"] 必须是 CPU tensor
      每层 K/V 各创建一个 SharedMemory block

    load_kv:
      attach 共享内存，拷回 torch CPU tensor，供 decode 导入 GPU cache

    cleanup:
      unlink 所有 shm block
    """

    def __init__(self, prefix: str = "pd_kv"):
        self.prefix = prefix

    def _new_shm_name(self, producer_id: str, request_id: str, layer_idx: int, kind: str) -> str:
        suffix = uuid.uuid4().hex[:8]
        return f"{self.prefix}_{producer_id}_{request_id}_L{layer_idx}_{kind}_{suffix}"

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
        layers = payload["layers"]
        if len(layers) != num_layers:
            raise ValueError(f"layers mismatch: payload={len(layers)}, num_layers={num_layers}")

        layer_metas = []

        for i, lp in enumerate(layers):
            k_cpu: torch.Tensor = lp["k"].contiguous().cpu()
            v_cpu: torch.Tensor = lp["v"].contiguous().cpu()
            seqlen = int(lp["seqlen"])

            k_arr, k_dtype_str, k_original_torch_dtype = _torch_tensor_to_numpy_exportable(k_cpu)
            v_arr, v_dtype_str, v_original_torch_dtype = _torch_tensor_to_numpy_exportable(v_cpu)

            k_name = self._new_shm_name(producer_id, request_id, i, "k")
            v_name = self._new_shm_name(producer_id, request_id, i, "v")

            k_shm = shared_memory.SharedMemory(name=k_name, create=True, size=k_arr.nbytes)
            v_shm = shared_memory.SharedMemory(name=v_name, create=True, size=v_arr.nbytes)

            try:
                k_view = np.ndarray(k_arr.shape, dtype=k_arr.dtype, buffer=k_shm.buf)
                v_view = np.ndarray(v_arr.shape, dtype=v_arr.dtype, buffer=v_shm.buf)
                k_view[...] = k_arr
                v_view[...] = v_arr
            finally:
                k_shm.close()
                v_shm.close()

            layer_metas.append({
                "layer_idx": i,

                "k_name": k_name,
                "k_shape": list(k_arr.shape),
                "k_dtype": k_dtype_str,
                "k_original_torch_dtype": k_original_torch_dtype,

                "v_name": v_name,
                "v_shape": list(v_arr.shape),
                "v_dtype": v_dtype_str,
                "v_original_torch_dtype": v_original_torch_dtype,

                "seqlen": seqlen,
            })

        return KVHandle(
            request_id=request_id,
            producer_id=producer_id,
            num_layers=num_layers,
            num_tokens=num_tokens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            layer_metas=layer_metas,
        )

    def load_kv(self, handle: KVHandle) -> dict:
        layers = []

        for meta in handle.layer_metas:
            k_shm = shared_memory.SharedMemory(name=meta["k_name"], create=False)
            v_shm = shared_memory.SharedMemory(name=meta["v_name"], create=False)

            try:
                k_np = np.ndarray(
                    shape=tuple(meta["k_shape"]),
                    dtype=_NUMPY_STR_TO_DTYPE[meta["k_dtype"]],
                    buffer=k_shm.buf,
                )
                v_np = np.ndarray(
                    shape=tuple(meta["v_shape"]),
                    dtype=_NUMPY_STR_TO_DTYPE[meta["v_dtype"]],
                    buffer=v_shm.buf,
                )

                # 这里 copy 一份，避免后面 close shm 后 tensor 悬空
                k_cpu = _numpy_importable_to_torch(k_np.copy(), meta["k_original_torch_dtype"]).contiguous()
                v_cpu = _numpy_importable_to_torch(v_np.copy(), meta["v_original_torch_dtype"]).contiguous()

                layers.append({
                    "k": k_cpu,
                    "v": v_cpu,
                    "seqlen": int(meta["seqlen"]),
                })
            finally:
                k_shm.close()
                v_shm.close()

        return {"layers": layers}

    def cleanup(self, handle: KVHandle):
        """
        请求生命周期结束后调用，真正 unlink 共享内存。
        """
        for meta in handle.layer_metas:
            for shm_name_key in ("k_name", "v_name"):
                name = meta[shm_name_key]
                try:
                    shm = shared_memory.SharedMemory(name=name, create=False)
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass