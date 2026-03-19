"""
L2: 模型层 - 支持 Tensor Parallel 的 Qwen3 模型
基于原始 model.py，替换为并行层
"""
import json, math
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from flash_attn import flash_attn_with_kvcache
from safetensors import safe_open

from .distributed import (
    get_tp_world_size,
    get_tp_rank,
    is_distributed,
    is_tp_rank_0,
)
from .parallel_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    VocabParallelEmbedding,
)


@dataclass
class Qwen3Config:
    """Qwen 3 model configuration"""
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40960
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    bos_token_id: int = 151643
    eos_token_id: int = 151645

    @classmethod
    def from_json(cls, path: str):
        """Load config from JSON file"""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _update_cos_sin_cache(self, seq_len, device, dtype):
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(self, q, k, position_ids):
        seq_len = position_ids.max().item() + 1
        self._update_cos_sin_cache(seq_len, q.device, q.dtype)
        
        cos = self._cos_cached[position_ids].unsqueeze(2)
        sin = self._sin_cached[position_ids].unsqueeze(2)
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class AttentionTP(nn.Module):
    """
    Multi-head attention with Tensor Parallel support
    
    TP 策略：
        - QKV 按 head 切分（每个 rank 有 num_heads // tp_size 个完整 head）
        - O projection 使用 RowParallel（输入已切分，输出 all_reduce）
        - KV cache 也按 head 切分
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        
        self.tp_size = get_tp_world_size()
        
        # 检查可切分性
        assert self.num_heads % self.tp_size == 0, \
            f"num_heads ({self.num_heads}) must be divisible by tp_size ({self.tp_size})"
        assert self.num_kv_heads % self.tp_size == 0, \
            f"num_kv_heads ({self.num_kv_heads}) must be divisible by tp_size ({self.tp_size})"
        
        # 每个 rank 的 head 数量
        self.num_heads_per_partition = self.num_heads // self.tp_size
        self.num_kv_heads_per_partition = self.num_kv_heads // self.tp_size
        self.num_kv_groups = self.num_heads_per_partition // self.num_kv_heads_per_partition
        
        self._q_size = self.num_heads * self.head_dim
        self._kv_size = self.num_kv_heads * self.head_dim
        
        # QK norm
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # QKV 并行投影
        self._qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            bias=False,
        )
        
        # O 投影（RowParallel：输入已切分，输出需要 all_reduce）
        self.o_proj = RowParallelLinear(
            self._q_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
        )
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, 
            config.max_position_embeddings, 
            config.rope_theta
        )
        self.scaling = self.head_dim ** -0.5
        
        # KV cache（每个 rank 只存储自己的 heads）
        self._kv_cache = None
        self._v_cache = None
        self._cache_seqlens = None

    def forward(self, hidden_states, position_ids, slot_indices=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: [batch, seq_len]
            slot_indices: [batch] - KV cache slots
        
        Returns:
            [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV 投影（已经是切分后的）
        q, k, v = self._qkv_proj(hidden_states)
        # q: [batch, seq_len, num_heads_per_partition, head_dim]
        # k, v: [batch, seq_len, num_kv_heads_per_partition, head_dim]
        
        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # RoPE
        q, k = self.rotary_emb(q, k, position_ids)
        
        # Flash attention with KV cache
        if slot_indices is not None and seq_len == 1:
            # Decode: use cached KV
            attn_output = flash_attn_with_kvcache(
                q, self._kv_cache, self._v_cache,
                k=k, v=v,
                cache_seqlens=self._cache_seqlens[slot_indices].int(),
                cache_batch_idx=slot_indices.int(),
                softmax_scale=self.scaling,
                causal=False
            )
            self._cache_seqlens[slot_indices] += 1
        else:
            # Prefill: populate KV cache
            if slot_indices is not None:
                for i, slot_idx in enumerate(slot_indices):
                    seq_len_i = position_ids[i].max().item() + 1
                    self._kv_cache[slot_idx, :seq_len_i].copy_(k[i, :seq_len_i])
                    self._v_cache[slot_idx, :seq_len_i].copy_(v[i, :seq_len_i])
                    self._cache_seqlens[slot_idx] = seq_len_i
            
            from flash_attn import flash_attn_func
            attn_output = flash_attn_func(q, k, v, softmax_scale=self.scaling, causal=True)
        
        # Output projection（RowParallel 会自动 all_reduce）
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        
        return output


class MLPTP(nn.Module):
    """
    Feed-forward network with Tensor Parallel
    
    TP 策略：
        - gate_proj, up_proj: ColumnParallel（输出 intermediate_size 切分）
        - down_proj: RowParallel（输入已切分，输出 all_reduce）
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
        )
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # gate 和 up 输出都是切分的 [batch, seq, intermediate // tp_size]
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        # down 输入切分，输出 all_reduce 后是完整的 [batch, seq, hidden]
        return self.down_proj(gate * up)


class TransformerBlockTP(nn.Module):
    """Transformer layer with Tensor Parallel"""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = AttentionTP(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLPTP(config)

    def forward(self, hidden_states, position_ids, slot_indices=None):
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, slot_indices)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class Qwen3ModelTP(nn.Module):
    """Qwen 3 transformer backbone with Tensor Parallel"""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        
        # Embedding（可选：使用 VocabParallel 或保持复制）
        # 这里为了简化，保持 embedding 在所有 rank 复制
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = nn.ModuleList([
            TransformerBlockTP(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, position_ids, slot_indices=None):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, slot_indices)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3ForCausalLMTP(nn.Module):
    """Qwen 3 model with language modeling head and Tensor Parallel"""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3ModelTP(config)
        
        # LM head（保持复制，只在 rank 0 计算 logits）
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids, slot_indices=None):
        hidden_states = self.model(input_ids, position_ids, slot_indices)
        
        # 只在 rank 0 计算 logits（节省通信）
        if is_tp_rank_0():
            if self.lm_head is None:
                logits = F.linear(hidden_states, self.model.embed_tokens.weight)
            else:
                logits = self.lm_head(hidden_states)
        else:
            # 其他 rank 返回 dummy tensor
            logits = torch.empty(
                hidden_states.shape[0], hidden_states.shape[1], self.config.vocab_size,
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        
        return logits

    def init_kv_cache(self, num_slots: int, max_seq_len: int, device, dtype):
        """Allocate slot-based KV cache（每个 rank 只存储自己的 heads）"""
        for layer in self.model.layers:
            attn = layer.self_attn
            attn._kv_cache = torch.zeros(
                num_slots, max_seq_len, 
                attn.num_kv_heads_per_partition,  # 注意：是切分后的 head 数
                attn.head_dim,
                device=device, dtype=dtype
            )
            attn._v_cache = torch.zeros(
                num_slots, max_seq_len, 
                attn.num_kv_heads_per_partition,
                attn.head_dim,
                device=device, dtype=dtype
            )
            attn._cache_seqlens = torch.zeros(num_slots, dtype=torch.int32, device=device)

    def clear_slot(self, idx: int):
        for layer in self.model.layers:
            layer.self_attn._cache_seqlens[idx] = 0

    def clear_all_slots(self):
        for layer in self.model.layers:
            layer.self_attn._cache_seqlens.zero_()

    @classmethod
    def from_pretrained(cls, model_path: str, config: Qwen3Config, device, dtype):
        """
        Load model from safetensors and convert to TP
        
        策略：
            1. Rank 0 加载完整权重
            2. 转换为并行层（自动切分）
            3. 广播到其他 rank（或各 rank 独立加载后切分）
        """
        model_path = Path(model_path)
        
        # 创建模型
        model = cls(config).to(dtype)
        
        # 加载权重
        safetensors_file = model_path / "model.safetensors"
        if not safetensors_file.exists():
            raise FileNotFoundError(f"No model.safetensors found in {model_path}")
        
        if is_tp_rank_0():
            print(f"[Rank 0] Loading weights from {safetensors_file}")
        
        # 所有 rank 都加载完整权重，然后各自切分
        state_dict = {}
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in tqdm(f.keys(), desc=f"[Rank {get_tp_rank()}] Loading", ncols=80, disable=not is_tp_rank_0()):
                tensor = f.get_tensor(key)
                if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16):
                    tensor = tensor.to(dtype)
                state_dict[key] = tensor
        
        # 转换权重到 TP 格式
        model = cls._convert_to_tp(model, state_dict, config)
        model = model.to(device)
        
        if is_tp_rank_0():
            print(f"✓ Model loaded with TP (world_size={get_tp_world_size()})")
        
        return model
    
    @classmethod
    def _convert_to_tp(cls, model, state_dict, config):
        """
        将标准权重转换为 TP 格式
        
        处理：
            - QKV: 按 head 切分
            - O proj: 按输入维度切分
            - MLP gate/up: 按输出维度切分
            - MLP down: 按输入维度切分
        """
        from .distributed import split_tensor_along_dim
        
        tp_rank = get_tp_rank()
        tp_size = get_tp_world_size()
        
        # Embedding（复制到所有 rank）
        model.model.embed_tokens.weight.data = state_dict["model.embed_tokens.weight"]
        
        # LM head
        if not config.tie_word_embeddings and "lm_head.weight" in state_dict:
            model.lm_head.weight.data = state_dict["lm_head.weight"]
        
        # Transformer layers
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            layer = model.model.layers[layer_idx]
            
            # LayerNorm（复制）
            layer.input_layernorm.weight.data = state_dict[f"{prefix}.input_layernorm.weight"]
            layer.post_attention_layernorm.weight.data = state_dict[f"{prefix}.post_attention_layernorm.weight"]
            
            # Attention QKV（需要特殊处理：融合 + 按 head 切分）
            q_weight = state_dict[f"{prefix}.self_attn.q_proj.weight"]
            k_weight = state_dict[f"{prefix}.self_attn.k_proj.weight"]
            v_weight = state_dict[f"{prefix}.self_attn.v_proj.weight"]
            
            # 拼接 QKV
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            
            # 使用 QKVParallelLinear 的切分逻辑
            q_size = config.num_attention_heads * config.head_dim
            kv_size = config.num_key_value_heads * config.head_dim
            
            q_per_rank = q_size // tp_size
            kv_per_rank = kv_size // tp_size
            
            # 切分 Q
            q_start = tp_rank * q_per_rank
            q_end = q_start + q_per_rank
            q_weight_split = qkv_weight[q_start:q_end]
            
            # 切分 K
            k_start = q_size + tp_rank * kv_per_rank
            k_end = k_start + kv_per_rank
            k_weight_split = qkv_weight[k_start:k_end]
            
            # 切分 V
            v_start = q_size + kv_size + tp_rank * kv_per_rank
            v_end = v_start + kv_per_rank
            v_weight_split = qkv_weight[v_start:v_end]
            
            # 拼接当前 rank 的 QKV
            layer.self_attn._qkv_proj.weight.data = torch.cat([
                q_weight_split, k_weight_split, v_weight_split
            ], dim=0)
            
            # QK norm（复制）
            layer.self_attn.q_norm.weight.data = state_dict[f"{prefix}.self_attn.q_norm.weight"]
            layer.self_attn.k_norm.weight.data = state_dict[f"{prefix}.self_attn.k_norm.weight"]
            
            # Attention O proj（RowParallel：按列切分）
            o_weight = state_dict[f"{prefix}.self_attn.o_proj.weight"]
            layer.self_attn.o_proj.weight.data = split_tensor_along_dim(o_weight, tp_size, dim=1)
            
            # MLP gate_proj（ColumnParallel：按行切分）
            gate_weight = state_dict[f"{prefix}.mlp.gate_proj.weight"]
            layer.mlp.gate_proj.weight.data = split_tensor_along_dim(gate_weight, tp_size, dim=0)
            
            # MLP up_proj（ColumnParallel：按行切分）
            up_weight = state_dict[f"{prefix}.mlp.up_proj.weight"]
            layer.mlp.up_proj.weight.data = split_tensor_along_dim(up_weight, tp_size, dim=0)
            
            # MLP down_proj（RowParallel：按列切分）
            down_weight = state_dict[f"{prefix}.mlp.down_proj.weight"]
            layer.mlp.down_proj.weight.data = split_tensor_along_dim(down_weight, tp_size, dim=1)
        
        # Final norm（复制）
        model.model.norm.weight.data = state_dict["model.norm.weight"]
        
        return model
