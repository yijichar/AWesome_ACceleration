"""
L1: 并行层 - Tensor Parallel 线性层实现
参考 vLLM 的 layers/linear.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .distributed import (
    get_tp_world_size,
    get_tp_rank,
    is_distributed,
    tensor_model_parallel_all_reduce,
    split_tensor_along_dim,
)


# ============================================================================
# Column Parallel Linear
# ============================================================================

class ColumnParallelLinear(nn.Module):
    """
    列并行线性层：输出维度切分
    
    Y = XA^T, A 按列切分
    
    切分方式：
        A: [in_features, out_features] -> [in_features, out_features // tp_size]
        每个 rank 计算部分输出
    
    用于：
        - MLP gate_proj, up_proj (输出 intermediate_size)
        - Attention QKV proj (输出 num_heads * head_dim)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = False,
        skip_bias_add: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        
        # 计算当前 rank 的输出维度
        self.tp_size = get_tp_world_size()
        assert out_features % self.tp_size == 0, \
            f"out_features ({out_features}) must be divisible by tp_size ({self.tp_size})"
        
        self.out_features_per_partition = out_features // self.tp_size
        
        # 创建分片权重
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_partition)
            )
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_: [batch, seq_len, in_features]
        
        Returns:
            [batch, seq_len, out_features] if gather_output
            [batch, seq_len, out_features // tp_size] otherwise
        """
        # 线性变换
        output = F.linear(input_, self.weight, self.bias)
        
        # 是否收集所有分片
        if self.gather_output and is_distributed():
            from .distributed import tensor_model_parallel_all_gather
            output = tensor_model_parallel_all_gather(output, dim=-1)
        
        return output
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, gather_output: bool = False):
        """
        从标准 Linear 层创建 ColumnParallel 层（用于权重加载）
        
        Args:
            linear: 原始 nn.Linear
            gather_output: 是否收集输出
        
        Returns:
            ColumnParallelLinear 实例
        """
        col_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            gather_output=gather_output,
        )
        
        # 切分权重
        col_linear.weight.data = split_tensor_along_dim(
            linear.weight.data, col_linear.tp_size, dim=0
        )
        
        if linear.bias is not None:
            col_linear.bias.data = split_tensor_along_dim(
                linear.bias.data, col_linear.tp_size, dim=0
            )
        
        return col_linear


# ============================================================================
# Row Parallel Linear
# ============================================================================

class RowParallelLinear(nn.Module):
    """
    行并行线性层：输入维度切分 + all_reduce
    
    Y = XA^T, A 按行切分
    
    切分方式：
        A: [in_features, out_features] -> [in_features // tp_size, out_features]
        每个 rank 计算部分结果，最后 all_reduce 求和
    
    用于：
        - MLP down_proj (输入 intermediate_size)
        - Attention o_proj (输入 num_heads * head_dim)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        
        # 计算当前 rank 的输入维度
        self.tp_size = get_tp_world_size()
        assert in_features % self.tp_size == 0, \
            f"in_features ({in_features}) must be divisible by tp_size ({self.tp_size})"
        
        self.in_features_per_partition = in_features // self.tp_size
        
        # 创建分片权重
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        
        if bias:
            # bias 只在 rank 0 保留（all_reduce 后只加一次）
            if get_tp_rank() == 0:
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_: [batch, seq_len, in_features // tp_size] if input_is_parallel
                    [batch, seq_len, in_features] otherwise
        
        Returns:
            [batch, seq_len, out_features]
        """
        # 如果输入不是并行的，先切分
        if not self.input_is_parallel and is_distributed():
            input_ = split_tensor_along_dim(input_, self.tp_size, dim=-1)
        
        # 线性变换
        output = F.linear(input_, self.weight)
        
        # All-reduce 聚合结果
        if is_distributed():
            output = tensor_model_parallel_all_reduce(output)
        
        # 添加 bias（只在 rank 0）
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, input_is_parallel: bool = True):
        """
        从标准 Linear 层创建 RowParallel 层
        
        Args:
            linear: 原始 nn.Linear
            input_is_parallel: 输入是否已经是并行切分的
        
        Returns:
            RowParallelLinear 实例
        """
        row_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            input_is_parallel=input_is_parallel,
        )
        
        # 切分权重（按行切分 = 按列切分转置矩阵）
        row_linear.weight.data = split_tensor_along_dim(
            linear.weight.data, row_linear.tp_size, dim=1
        )
        
        # bias 只在 rank 0
        if linear.bias is not None and get_tp_rank() == 0:
            row_linear.bias.data = linear.bias.data.clone()
        
        return row_linear


# ============================================================================
# QKV Parallel Linear (专用于 Attention)
# ============================================================================

class QKVParallelLinear(nn.Module):
    """
    QKV 并行线性层：按 attention head 切分
    
    特点：
        - Q, K, V 融合在一个矩阵中 [hidden, (num_q_heads + 2 * num_kv_heads) * head_dim]
        - 按 head 维度切分，保证每个 rank 有完整的 head
        - 支持 GQA (num_q_heads != num_kv_heads)
    
    切分方式：
        num_q_heads_per_rank = num_q_heads // tp_size
        num_kv_heads_per_rank = num_kv_heads // tp_size
    """
    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        self.tp_size = get_tp_world_size()
        
        # 检查可切分性
        assert num_q_heads % self.tp_size == 0, \
            f"num_q_heads ({num_q_heads}) must be divisible by tp_size ({self.tp_size})"
        assert num_kv_heads % self.tp_size == 0, \
            f"num_kv_heads ({num_kv_heads}) must be divisible by tp_size ({self.tp_size})"
        
        self.num_q_heads_per_partition = num_q_heads // self.tp_size
        self.num_kv_heads_per_partition = num_kv_heads // self.tp_size
        
        # 总输出维度
        self.q_size = num_q_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.qkv_size = self.q_size + 2 * self.kv_size
        
        # 分片输出维度
        self.q_size_per_partition = self.num_q_heads_per_partition * head_dim
        self.kv_size_per_partition = self.num_kv_heads_per_partition * head_dim
        self.qkv_size_per_partition = self.q_size_per_partition + 2 * self.kv_size_per_partition
        
        # 权重
        self.weight = nn.Parameter(
            torch.empty(self.qkv_size_per_partition, hidden_size)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.qkv_size_per_partition)
            )
        else:
            self.register_parameter('bias', None)
    
    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states:
                - [batch, seq_len, hidden_size]  (standard path)
                - [total_tokens, hidden_size]    (packed prefill path)
        Returns:
            If 3D input:
                q: [batch, seq_len, num_q_heads_per_partition, head_dim]
                k: [batch, seq_len, num_kv_heads_per_partition, head_dim]
                v: [batch, seq_len, num_kv_heads_per_partition, head_dim]
            If 2D input:
                q: [total_tokens, num_q_heads_per_partition, head_dim]
                k: [total_tokens, num_kv_heads_per_partition, head_dim]
                v: [total_tokens, num_kv_heads_per_partition, head_dim]
        """
        if hidden_states.dim() == 3:
            batch_size, seq_len, _ = hidden_states.shape
            qkv = F.linear(hidden_states, self.weight, self.bias)

            q = qkv[..., :self.q_size_per_partition]
            k = qkv[..., self.q_size_per_partition:self.q_size_per_partition + self.kv_size_per_partition]
            v = qkv[..., self.q_size_per_partition + self.kv_size_per_partition:]

            q = q.view(batch_size, seq_len, self.num_q_heads_per_partition, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_heads_per_partition, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_kv_heads_per_partition, self.head_dim)
            return q, k, v

        elif hidden_states.dim() == 2:
            total_tokens, _ = hidden_states.shape
            qkv = F.linear(hidden_states, self.weight, self.bias)

            q = qkv[:, :self.q_size_per_partition]
            k = qkv[:, self.q_size_per_partition:self.q_size_per_partition + self.kv_size_per_partition]
            v = qkv[:, self.q_size_per_partition + self.kv_size_per_partition:]

            q = q.view(total_tokens, self.num_q_heads_per_partition, self.head_dim)
            k = k.view(total_tokens, self.num_kv_heads_per_partition, self.head_dim)
            v = v.view(total_tokens, self.num_kv_heads_per_partition, self.head_dim)
            return q, k, v

        else:
            raise ValueError(f"hidden_states must be 2D or 3D, got shape={tuple(hidden_states.shape)}")
    
    @classmethod
    def from_fused_qkv(
        cls,
        qkv_weight: torch.Tensor,
        qkv_bias: Optional[torch.Tensor],
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        """
        从融合的 QKV 权重创建并行层
        
        Args:
            qkv_weight: [(num_q_heads + 2 * num_kv_heads) * head_dim, hidden_size]
            qkv_bias: [(num_q_heads + 2 * num_kv_heads) * head_dim] or None
        """
        qkv_parallel = cls(
            hidden_size, num_q_heads, num_kv_heads, head_dim,
            bias=qkv_bias is not None
        )
        
        tp_rank = get_tp_rank()
        tp_size = get_tp_world_size()
        
        # 计算切分索引
        q_size = num_q_heads * head_dim
        kv_size = num_kv_heads * head_dim
        
        q_per_rank = q_size // tp_size
        kv_per_rank = kv_size // tp_size
        
        # 切分 Q
        q_start = tp_rank * q_per_rank
        q_end = q_start + q_per_rank
        q_weight = qkv_weight[q_start:q_end]
        
        # 切分 K
        k_start = q_size + tp_rank * kv_per_rank
        k_end = k_start + kv_per_rank
        k_weight = qkv_weight[k_start:k_end]
        
        # 切分 V
        v_start = q_size + kv_size + tp_rank * kv_per_rank
        v_end = v_start + kv_per_rank
        v_weight = qkv_weight[v_start:v_end]
        
        # 拼接 [Q | K | V]
        qkv_parallel.weight.data = torch.cat([q_weight, k_weight, v_weight], dim=0)
        
        if qkv_bias is not None:
            q_bias = qkv_bias[q_start:q_end]
            k_bias = qkv_bias[k_start:k_end]
            v_bias = qkv_bias[v_start:v_end]
            qkv_parallel.bias.data = torch.cat([q_bias, k_bias, v_bias], dim=0)
        
        return qkv_parallel


# ============================================================================
# Vocab Parallel Embedding
# ============================================================================

class VocabParallelEmbedding(nn.Module):
    """
    词表并行 Embedding：按 vocab 维度切分
    
    切分方式：
        vocab_size -> vocab_size // tp_size
        每个 rank 负责部分词表
    
    注意：
        - 需要处理 padding_idx
        - 输入 token_id 需要映射到本地词表范围
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        self.tp_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()
        
        # 计算分片大小
        assert num_embeddings % self.tp_size == 0, \
            f"num_embeddings ({num_embeddings}) must be divisible by tp_size ({self.tp_size})"
        
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_index = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition
        
        # 创建分片 embedding
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
        
        Returns:
            [batch, seq_len, embedding_dim]
        """
        # 创建 mask：哪些 token 属于当前 rank
        mask = (input_ids >= self.vocab_start_index) & (input_ids < self.vocab_end_index)
        
        # 映射到本地索引
        local_ids = input_ids - self.vocab_start_index
        local_ids = torch.clamp(local_ids, 0, self.num_embeddings_per_partition - 1)
        
        # Embedding lookup
        output = F.embedding(local_ids, self.weight)
        
        # 只保留属于当前 rank 的结果
        output = output * mask.unsqueeze(-1)
        
        # All-reduce 聚合所有 rank 的结果
        if is_distributed():
            output = tensor_model_parallel_all_reduce(output)
        
        return output
    
    @classmethod
    def from_embedding(cls, embedding: nn.Embedding):
        """从标准 Embedding 创建并行版本"""
        vocab_parallel = cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            padding_idx=embedding.padding_idx,
        )
        
        # 切分权重
        vocab_parallel.weight.data = split_tensor_along_dim(
            embedding.weight.data, vocab_parallel.tp_size, dim=0
        )
        
        return vocab_parallel
