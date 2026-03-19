"""
L0: 分布式运行时层 - Tensor Parallel / DP×TP 通信封装
"""
import os
import torch
import torch.distributed as dist
from typing import Optional

_GLOBAL_TP_PROFILER = None  # type: ignore

def set_tp_profiler(profiler):
    global _GLOBAL_TP_PROFILER
    _GLOBAL_TP_PROFILER = profiler


# ============================================================================
# Global State
# ============================================================================
_TP_GROUP: Optional[dist.ProcessGroup] = None
_DP_GROUP: Optional[dist.ProcessGroup] = None
_TP_CPU_GROUP: Optional[dist.ProcessGroup] = None
_WORLD_SIZE: int = 1
_GLOBAL_RANK: int = 0
_LOCAL_RANK: int = 0

_TP_WORLD_SIZE: int = 1
_TP_RANK: int = 0

_DP_WORLD_SIZE: int = 1
_DP_RANK: int = 0

_TP_LEADER_GLOBAL_RANK: int = 0


# ============================================================================
# Initialization
# ============================================================================
def init_distributed(tp_size: int, backend: str = "nccl"):
    global _TP_GROUP, _DP_GROUP, _TP_CPU_GROUP
    global _WORLD_SIZE, _GLOBAL_RANK, _LOCAL_RANK
    global _TP_WORLD_SIZE, _TP_RANK
    global _DP_WORLD_SIZE, _DP_RANK
    global _TP_LEADER_GLOBAL_RANK

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    _WORLD_SIZE = dist.get_world_size()
    _GLOBAL_RANK = dist.get_rank()
    _LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

    assert _WORLD_SIZE % tp_size == 0, \
        f"WORLD_SIZE ({_WORLD_SIZE}) must be divisible by tp_size ({tp_size})"

    _TP_WORLD_SIZE = tp_size
    _DP_WORLD_SIZE = _WORLD_SIZE // tp_size

    replica_id = _GLOBAL_RANK // tp_size
    _TP_RANK = _GLOBAL_RANK % tp_size
    _DP_RANK = replica_id

    torch.cuda.set_device(_LOCAL_RANK)

    # 1) all TP NCCL groups
    tp_group_ranks_list = []
    for rid in range(_DP_WORLD_SIZE):
        ranks = list(range(rid * tp_size, (rid + 1) * tp_size))
        tp_group_ranks_list.append(ranks)

    for ranks in tp_group_ranks_list:
        g = dist.new_group(ranks=ranks, backend="nccl")
        if _GLOBAL_RANK in ranks:
            _TP_GROUP = g
            _TP_LEADER_GLOBAL_RANK = ranks[0]

    # 2) all TP Gloo groups (for object control)
    for ranks in tp_group_ranks_list:
        g = dist.new_group(ranks=ranks, backend="gloo")
        if _GLOBAL_RANK in ranks:
            _TP_CPU_GROUP = g

    # 3) all DP groups
    dp_group_ranks_list = []
    for tp_local_rank in range(tp_size):
        ranks = [rid * tp_size + tp_local_rank for rid in range(_DP_WORLD_SIZE)]
        dp_group_ranks_list.append(ranks)

    for ranks in dp_group_ranks_list:
        g = dist.new_group(ranks=ranks, backend="nccl")
        if _GLOBAL_RANK in ranks:
            _DP_GROUP = g

    if _GLOBAL_RANK == 0:
        print(f"[Dist] world_size={_WORLD_SIZE}, tp_size={_TP_WORLD_SIZE}, dp_size={_DP_WORLD_SIZE}")

# ============================================================================
# Query Functions
# ============================================================================
def get_world_size() -> int:
    return _WORLD_SIZE

def get_global_rank() -> int:
    return _GLOBAL_RANK

def get_dp_world_size() -> int:
    return _DP_WORLD_SIZE

def get_dp_rank() -> int:
    return _DP_RANK

def get_dp_group():
    return _DP_GROUP

def is_tp_leader() -> bool:
    return _TP_RANK == 0

def get_tp_leader_global_rank() -> int:
    return _TP_LEADER_GLOBAL_RANK

def is_global_rank_0() -> bool:
    return _GLOBAL_RANK == 0

def is_distributed() -> bool:
    # 这里沿用“是否需要 TP collectives”的语义
    return _TP_WORLD_SIZE > 1

def get_tp_world_size() -> int:
    return _TP_WORLD_SIZE

def get_tp_rank() -> int:
    return _TP_RANK

def get_local_rank() -> int:
    return _LOCAL_RANK

def is_tp_rank_0() -> bool:
    return _TP_RANK == 0

def get_tp_group() -> Optional[dist.ProcessGroup]:
    return _TP_GROUP


# ============================================================================
# Communication Primitives
# ============================================================================
def _tp_local_rank_to_global_rank(src_local_rank: int) -> int:
    return _TP_LEADER_GLOBAL_RANK + src_local_rank


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return input_

    prof = _GLOBAL_TP_PROFILER
    if prof is None or not getattr(prof, "enabled", False):
        dist.all_reduce(input_, group=_TP_GROUP)
        return input_

    with prof.section("comm.all_reduce"):
        dist.all_reduce(input_, group=_TP_GROUP)
    return input_


def tensor_model_parallel_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if not is_distributed():
        return input_

    world_size = get_tp_world_size()
    gathered_tensors = [torch.empty_like(input_) for _ in range(world_size)]

    prof = _GLOBAL_TP_PROFILER
    if prof is None or not getattr(prof, "enabled", False):
        dist.all_gather(gathered_tensors, input_, group=_TP_GROUP)
        return torch.cat(gathered_tensors, dim=dim)

    with prof.section("comm.all_gather"):
        dist.all_gather(gathered_tensors, input_, group=_TP_GROUP)

    return torch.cat(gathered_tensors, dim=dim)


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    if not is_distributed():
        return tensor

    global_src = _tp_local_rank_to_global_rank(src)
    prof = _GLOBAL_TP_PROFILER

    if prof is None or not getattr(prof, "enabled", False):
        dist.broadcast(tensor, src=global_src, group=_TP_GROUP)
        return tensor

    with prof.section("comm.broadcast_tensor"):
        dist.broadcast(tensor, src=global_src, group=_TP_GROUP)
    return tensor


def broadcast_object(obj, src: int = 0):
    if not is_distributed():
        return obj

    obj_list = [obj]
    global_src = _tp_local_rank_to_global_rank(src)
    prof = _GLOBAL_TP_PROFILER

    # object control should use Gloo group, not NCCL TP group
    group = _TP_CPU_GROUP
    assert group is not None, "TP CPU control group is not initialized"

    if prof is None or not getattr(prof, "enabled", False):
        dist.broadcast_object_list(obj_list, src=global_src, group=group)
        return obj_list[0]

    with prof.section("comm.broadcast_object"):
        dist.broadcast_object_list(obj_list, src=global_src, group=group)
    return obj_list[0]

# ============================================================================
# Utilities
# ============================================================================
def split_tensor_along_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    dim: int,
    contiguous: bool = True
) -> torch.Tensor:
    assert tensor.shape[dim] % num_partitions == 0, \
        f"Dimension {dim} ({tensor.shape[dim]}) not divisible by {num_partitions}"

    chunks = torch.chunk(tensor, num_partitions, dim=dim)
    output = chunks[get_tp_rank()]
    if contiguous:
        output = output.contiguous()
    return output


def gather_from_tensor_model_parallel_region(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return tensor_model_parallel_all_gather(input_, dim=dim)


# ============================================================================
# Context Manager
# ============================================================================
class SyncMode:
    def __enter__(self):
        if is_distributed():
            dist.barrier(group=_TP_GROUP)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if is_distributed():
            dist.barrier(group=_TP_GROUP)
        return False