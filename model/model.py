"""Qwen 3 0.6B Model optimized with Flash Attention 2, continuous batching, and slot-based KV cache"""
import json, math
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from flash_attn import flash_attn_with_kvcache
from safetensors import safe_open


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
        # Extract only fields that exist in our dataclass
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
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _update_cos_sin_cache(self, seq_len, device, dtype):
        """Update cached cos/sin values if sequence length changed"""
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(self, q, k, position_ids):
        """Apply rotary embeddings to query and key tensors"""
        seq_len = position_ids.max().item() + 1
        self._update_cos_sin_cache(seq_len, q.device, q.dtype)
        
        # Select cos/sin for the given positions
        cos = self._cos_cached[position_ids].unsqueeze(2)  # [batch, seq_len, 1, dim]
        sin = self._sin_cached[position_ids].unsqueeze(2)  # [batch, seq_len, 1, dim]
        
        # Apply rotation
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x):
        """Rotate half the hidden dims of the input"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    """Multi-head attention with GQA and slot-based KV caching"""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self._q_size = self.num_heads * self.head_dim
        self._kv_size = self.num_kv_heads * self.head_dim
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Separate Q/K/V projections - will be fused after loading
        self._qkv_proj = None
        self.q_proj = nn.Linear(self.hidden_size, self._q_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self._kv_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self._kv_size, bias=False)
        self.o_proj = nn.Linear(self._q_size, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)
        self.scaling = self.head_dim ** -0.5
        
        # Slot-based KV cache [num_slots, max_seq_len, num_kv_heads, head_dim]
        self._kv_cache = None
        self._v_cache = None
        self._cache_seqlens = None

    def fuse_qkv(self):
        """Combine Q, K, V into single fused projection for efficiency"""
        if self._qkv_proj is not None:
            return
        
        device, dtype = self.q_proj.weight.device, self.q_proj.weight.dtype
        self._qkv_proj = nn.Linear(
            self.hidden_size, 
            self._q_size + 2 * self._kv_size, 
            bias=False, 
            device=device, 
            dtype=dtype
        )
        
        # Concatenate weights [Q | K | V]
        with torch.no_grad():
            self._qkv_proj.weight[:self._q_size] = self.q_proj.weight
            self._qkv_proj.weight[self._q_size:self._q_size + self._kv_size] = self.k_proj.weight
            self._qkv_proj.weight[self._q_size + self._kv_size:] = self.v_proj.weight
        
        # Delete original projections to free memory
        del self.q_proj, self.k_proj, self.v_proj
        self.q_proj = self.k_proj = self.v_proj = None


class MLP(nn.Module):
    """Feed-forward network with SiLU activation"""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer layer with pre-norm architecture"""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, hidden_states, position_ids, slot_indices=None):
        """
        Forward pass for a single layer
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: [batch, seq_len]
            slot_indices: [batch] - which KV cache slots to use (None for prefill)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        batch_size, seq_len, _ = hidden_states.shape
        attn = self.self_attn
        
        # Fused QKV projection
        qkv = attn._qkv_proj(hidden_states)
        q = qkv[..., :attn._q_size].view(batch_size, seq_len, attn.num_heads, attn.head_dim)
        k = qkv[..., attn._q_size:attn._q_size + attn._kv_size].view(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
        v = qkv[..., attn._q_size + attn._kv_size:].view(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
        
        # Qwen3-specific QK norm
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        # Apply rotary embeddings
        q, k = attn.rotary_emb(q, k, position_ids)
        
        # Flash attention with KV cache
        if slot_indices is not None and seq_len == 1:
            # Decode: use cached KV
            attn_output = flash_attn_with_kvcache(
                q, attn._kv_cache, attn._v_cache,
                k=k, v=v,
                cache_seqlens=attn._cache_seqlens[slot_indices].int(),
                cache_batch_idx=slot_indices.int(),
                softmax_scale=attn.scaling,
                causal=False
            )
            attn._cache_seqlens[slot_indices] += 1
        else:
            # Prefill: populate KV cache
            if slot_indices is not None:
                # Store K, V in cache
                for i, slot_idx in enumerate(slot_indices):
                    seq_len_i = position_ids[i].max().item() + 1
                    attn._kv_cache[slot_idx, :seq_len_i] = k[i]
                    attn._v_cache[slot_idx, :seq_len_i] = v[i]
                    attn._cache_seqlens[slot_idx] = seq_len_i
            
            # Standard flash attention for prefill
            from flash_attn import flash_attn_func
            attn_output = flash_attn_func(q, k, v, softmax_scale=attn.scaling, causal=True)
        
        # Output projection
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        hidden_states = attn.o_proj(attn_output)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class Qwen3Model(nn.Module):
    """Qwen 3 transformer backbone"""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, position_ids, slot_indices=None):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, slot_indices)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen 3 model with language modeling head"""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        
        # LM head - may share weights with embeddings
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids, slot_indices=None):
        """
        Forward pass
        
        Args:
            input_ids: [batch, seq_len]
            position_ids: [batch, seq_len]
            slot_indices: [batch] - KV cache slots (None for prefill)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        hidden_states = self.model(input_ids, position_ids, slot_indices)
        
        if self.lm_head is None:
            # Tied embeddings
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        return logits

    def init_kv_cache(self, num_slots: int, max_seq_len: int, device, dtype):
        """Allocate slot-based KV cache"""
        for layer in self.model.layers:
            attn = layer.self_attn
            attn._kv_cache = torch.zeros(
                num_slots, max_seq_len, attn.num_kv_heads, attn.head_dim,
                device=device, dtype=dtype
            )
            attn._v_cache = torch.zeros(
                num_slots, max_seq_len, attn.num_kv_heads, attn.head_dim,
                device=device, dtype=dtype
            )
            attn._cache_seqlens = torch.zeros(num_slots, dtype=torch.int32, device=device)

    def clear_slot(self, idx: int):
        """Reset cache length for a slot"""
        for layer in self.model.layers:
            layer.self_attn._cache_seqlens[idx] = 0

    def clear_all_slots(self):
        """Reset all cache lengths"""
        for layer in self.model.layers:
            layer.self_attn._cache_seqlens.zero_()

    def fuse_qkv(self):
        """Apply QKV fusion to all layers"""
        for layer in self.model.layers:
            layer.self_attn.fuse_qkv()

    @classmethod
    def from_pretrained(cls, model_path: str, config: Qwen3Config, device, dtype):
        """Load model from safetensors"""
        model_path = Path(model_path)
        
        # Create model
        model = cls(config).to(dtype)
        
        # Load weights from safetensors
        safetensors_file = model_path / "model.safetensors"
        if not safetensors_file.exists():
            raise FileNotFoundError(f"No model.safetensors found in {model_path}")
        
        print(f"Loading weights from {safetensors_file}")
        state_dict = {}
        with safe_open(safetensors_file, framework="pt", device=str(device)) as f:
            for key in tqdm(f.keys(), desc="Loading weights", ncols=80):
                tensor = f.get_tensor(key)
                # Convert to target dtype if needed
                if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16):
                    tensor = tensor.to(dtype)
                state_dict[key] = tensor
        
        # Load state dict
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        
        model = model.to(device)
        return model
