#!/usr/bin/env python3
"""Qwen 3 0.6B inference engine with continuous batching and async request queue"""
import os, time, threading, queue, concurrent.futures, bisect
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import torch
import torch.nn.functional as F
from tqdm import tqdm
from flash_attn import flash_attn_varlen_func
from transformers import AutoTokenizer


@dataclass
class GenerationOutput:
    """Result of a single prompt generation"""
    text: str                              # Generated text
    token_ids: list[int]                   # Generated token IDs
    prompt_tokens: int = 0                 # Number of prompt tokens
    generated_tokens: int = 0              # Number of generated tokens


@dataclass
class _Request:
    """Internal request tracking for continuous batching"""
    token_ids: list[list[int]]             # Tokenized prompts
    max_tokens: int                        # Generation limit per prompt
    temperature: float                     # Sampling temperature
    ignore_eos: bool                       # Continue after EOS
    future: concurrent.futures.Future      # Result future
    results: list = field(default_factory=list)
    pending_indices: list = field(default_factory=list)


class LLM:
    """
    High-throughput LLM inference engine with continuous batching.
    
    Features:
    - Async by default: background thread processes requests
    - Continuous batching: new requests join mid-generation
    - Slot-based KV cache: zero-copy sequence management
    - Fused QKV projections: 3 matmuls → 1
    - Flash Attention 2: efficient attention
    """
    def __init__(self, model_path: str, max_num_seqs: int = 32, max_seq_len: int = 4096, dtype=torch.bfloat16):
        self.device = torch.device("cuda")
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        
        # Import model after setting path
        from model.model import Qwen3Config, Qwen3ForCausalLM
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        self.config = Qwen3Config.from_json(config_path)
        
        # Calculate max batch size based on available GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        kv_bytes_per_token = (
            2 *  # K and V
            self.config.num_key_value_heads * 
            self.config.head_dim * 
            2 *  # bfloat16 = 2 bytes
            self.config.num_hidden_layers
        )
        # Reserve memory for model weights (estimate ~2GB for 0.6B model)
        available_memory = (gpu_memory_gb - 3.0) * (1024**3)
        calculated_max_seqs = max(1, int(available_memory / kv_bytes_per_token) // max_seq_len)
        self.max_num_seqs = min(max_num_seqs, calculated_max_seqs)
        
        print(f"Initializing Qwen 3 0.6B engine...")
        print(f"  GPU memory: {gpu_memory_gb:.1f} GB")
        print(f"  Max batch size: {self.max_num_seqs}")
        print(f"  Max sequence length: {max_seq_len}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load model
        self.model = Qwen3ForCausalLM.from_pretrained(model_path, self.config, self.device, dtype)
        self.model.eval()
        
        # Fuse QKV projections for efficiency
        print("Fusing QKV projections...")
        self.model.fuse_qkv()
        
        # Initialize KV cache
        print(f"Allocating KV cache ({self.max_num_seqs} slots × {max_seq_len} tokens)...")
        self.model.init_kv_cache(self.max_num_seqs, max_seq_len, self.device, dtype)
        
        # Pre-allocated decode tensors
        self._decode_input_ids = torch.zeros(self.max_num_seqs, 1, dtype=torch.long, device=self.device)
        self._decode_positions = torch.zeros(self.max_num_seqs, 1, dtype=torch.long, device=self.device)
        
        # Async request queue
        self._request_queue: queue.Queue[_Request] = queue.Queue()
        self._loop_running = False
        self._loop_thread = None
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            dummy_ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            dummy_pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            self.model(dummy_ids, dummy_pos, torch.zeros(1, dtype=torch.long, device=self.device))
        self.model.clear_all_slots()
        
        print("✓ Engine ready")

    # ═══════════════════════════════════════════════════════════════════════════════
    # CORE INFERENCE
    # ═══════════════════════════════════════════════════════════════════════════════

    def _prefill(self, sequences: list[list[int]], slot_indices: list[int]) -> torch.Tensor:
        """
        Process prompt tokens in parallel, populating KV cache.
        
        Uses flash_attn_varlen_func for variable-length sequences.
        """
        num_seqs = len(sequences)
        seq_lens = [len(s) for s in sequences]
        total_tokens = sum(seq_lens)
        max_seq_len = max(seq_lens)
        
        # Pack all tokens into single tensor
        flat_tokens = torch.tensor([t for s in sequences for t in s], dtype=torch.long, device=self.device)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        
        # Cumulative sequence lengths for flash attention
        cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=self.device)
        cu_seqlens[1:] = seq_lens_t.cumsum(0)
        
        # Position IDs for each token
        positions = torch.cat([torch.arange(l, device=self.device) for l in seq_lens])
        
        # Scatter indices for KV cache
        slot_indices_t = torch.tensor(slot_indices, dtype=torch.long, device=self.device)
        scatter_indices = torch.repeat_interleave(slot_indices_t * self.max_seq_len, seq_lens_t.long()) + positions
        
        # Forward through model
        hidden = self.model.model.embed_tokens(flat_tokens)
        
        for layer in self.model.model.layers:
            attn = layer.self_attn
            
            # Layer norm
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            
            # Fused QKV projection
            qkv = attn._qkv_proj(hidden)
            q = qkv[..., :attn._q_size].view(total_tokens, attn.num_heads, attn.head_dim)
            k = qkv[..., attn._q_size:attn._q_size + attn._kv_size].view(total_tokens, attn.num_kv_heads, attn.head_dim)
            v = qkv[..., attn._q_size + attn._kv_size:].view(total_tokens, attn.num_kv_heads, attn.head_dim)
            
            # Qwen3-specific QK norm (must match model.forward/decode path)
            q = attn.q_norm(q)
            k = attn.k_norm(k)

            # Apply RoPE
            q, k = attn.rotary_emb(
                q.unsqueeze(0), 
                k.unsqueeze(0), 
                positions.unsqueeze(0)
            )
            q, k = q.squeeze(0), k.squeeze(0)
            
            # Populate KV cache
            kv_flat = attn._kv_cache.view(-1, attn.num_kv_heads, attn.head_dim)
            v_flat = attn._v_cache.view(-1, attn.num_kv_heads, attn.head_dim)
            kv_flat.index_copy_(0, scatter_indices, k)
            v_flat.index_copy_(0, scatter_indices, v)
            attn._cache_seqlens.index_copy_(0, slot_indices_t, seq_lens_t)
            
            # Flash attention
            attn_out = flash_attn_varlen_func(
                q, k, v, 
                cu_seqlens, cu_seqlens, 
                max_seq_len, max_seq_len,
                causal=True, 
                softmax_scale=attn.scaling
            )
            
            # Output projection
            hidden = attn.o_proj(attn_out.reshape(total_tokens, -1))
            hidden = residual + hidden
            
            # MLP
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
        
        # Final norm
        hidden = self.model.model.norm(hidden)
        
        # Get logits for last token of each sequence
        last_token_indices = cu_seqlens[1:] - 1
        last_hidden = hidden[last_token_indices]
        
        if self.model.lm_head is None:
            logits = F.linear(last_hidden, self.model.model.embed_tokens.weight)
        else:
            logits = self.model.lm_head(last_hidden)
        
        return logits

    def _decode_step(self, input_ids: torch.Tensor, positions: torch.Tensor, slot_indices: torch.Tensor) -> torch.Tensor:
        """Generate logits for one token per sequence using cached KV"""
        logits = self.model(input_ids, positions, slot_indices)
        return logits[:, 0, :]  # [batch, vocab_size]

    def _sample_tokens(self, logits: torch.Tensor, temperature: float) -> list[int]:
        """Sample next tokens from logits"""
        if temperature == 0:
            return logits.argmax(dim=-1).tolist()
        return torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).squeeze(-1).tolist()

    # ═══════════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════════════

    def generate(self, prompts: list[str], max_tokens: int = 100, temperature: float = 0.0,
                 ignore_eos: bool = False) -> concurrent.futures.Future:
        """
        Submit prompts for generation.
        
        Args:
            prompts: List of text prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature (0 = greedy)
            ignore_eos: Continue generating after EOS
            
        Returns:
            Future that resolves to list[GenerationOutput]
        """
        if not self._loop_running:
            self._loop_running = True
            self._loop_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._loop_thread.start()
        
        future = concurrent.futures.Future()
        # token_ids = [self.tokenizer.encode(p, add_special_tokens=True) for p in prompts]
        token_ids = []
        for p in prompts:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True
            )
            token_ids.append(self.tokenizer.encode(text, add_special_tokens=False))
        request = _Request(
            token_ids=token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=ignore_eos,
            future=future,
            results=[None] * len(prompts),
            pending_indices=list(range(len(prompts)))
        )
        self._request_queue.put(request)
        return future

    def chat(self, messages: list[dict], max_tokens: int = 100, temperature: float = 0.0,
             ignore_eos: bool = False) -> concurrent.futures.Future:
        """
        Multi-turn chat generation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            ignore_eos: Continue after EOS
            
        Returns:
            Future that resolves to list[GenerationOutput] with single result
        """
        if not self._loop_running:
            self._loop_running = True
            self._loop_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._loop_thread.start()
        
        future = concurrent.futures.Future()
        # Use tokenizer's chat template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = [self.tokenizer.encode(prompt, add_special_tokens=False)]
        
        request = _Request(
            token_ids=token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=ignore_eos,
            future=future,
            results=[None],
            pending_indices=[0]
        )
        self._request_queue.put(request)
        return future

    def stop(self, timeout: float = 30.0):
        """Stop the background inference loop"""
        if not self._loop_running:
            return
        self._loop_running = False
        if self._loop_thread:
            self._loop_thread.join(timeout=timeout)
            self._loop_thread = None

    # ═══════════════════════════════════════════════════════════════════════════════
    # BACKGROUND INFERENCE LOOP
    # ═══════════════════════════════════════════════════════════════════════════════

    @torch.inference_mode()
    def _inference_loop(self):
        """Main inference loop with continuous batching"""
        free_slots = list(range(self.max_num_seqs))
        active_generations = {}  # slot -> (request, prompt_idx, tokens, position)
        pending_requests = []
        
        total_prompts = 0
        decode_times = deque(maxlen=16)
        pbar = None
        self.model.clear_all_slots()
        
        while self._loop_running or pending_requests or active_generations:
            # Phase 1: Drain request queue
            while True:
                try:
                    req = self._request_queue.get_nowait()
                except queue.Empty:
                    break
                
                pending_requests.append(req)
                total_prompts += len(req.token_ids)
                if pbar is None:
                    pbar = tqdm(total=total_prompts, desc="Generating", unit="req", ncols=100)
                else:
                    pbar.total = total_prompts
                    pbar.refresh()
            
            # Idle state
            if not pending_requests and not active_generations:
                if not self._loop_running:
                    break
                time.sleep(0.001)
                continue
            
            # Phase 2: Assign slots to pending requests
            new_work = []
            for req in list(pending_requests):
                while req.pending_indices and free_slots:
                    new_work.append((req, req.pending_indices.pop(0), free_slots.pop(0)))
                if not req.pending_indices:
                    pending_requests.remove(req)
            
            # Phase 3: Prefill new sequences
            if new_work:
                seqs = [r.token_ids[i] for r, i, _ in new_work]
                slots = [s for _, _, s in new_work]
                
                logits = self._prefill(seqs, slots)
                sampled = self._sample_tokens(logits, new_work[0][0].temperature)
                
                for idx, (req, prompt_idx, slot) in enumerate(new_work):
                    tok = sampled[idx]
                    prompt_len = len(req.token_ids[prompt_idx])
                    
                    # Check immediate completion
                    if 1 >= req.max_tokens or (not req.ignore_eos and tok == self.config.eos_token_id):
                        text = self.tokenizer.decode([tok], skip_special_tokens=True)
                        req.results[prompt_idx] = GenerationOutput(
                            text, [tok], prompt_len, 1
                        )
                        self.model.clear_slot(slot)
                        bisect.insort(free_slots, slot)
                        if pbar:
                            pbar.update(1)
                        if all(r is not None for r in req.results):
                            req.future.set_result(req.results)
                    else:
                        active_generations[slot] = (req, prompt_idx, [tok], prompt_len)
            
            if not active_generations:
                continue
            
            # Phase 4: Decode step for active sequences
            slots = list(active_generations.keys())
            batch_size = len(slots)
            
            input_ids = self._decode_input_ids[:batch_size]
            positions = self._decode_positions[:batch_size]
            
            for i, slot in enumerate(slots):
                req, prompt_idx, tokens, pos = active_generations[slot]
                input_ids[i, 0] = tokens[-1]
                positions[i, 0] = pos
            
            slot_indices = torch.tensor(slots, dtype=torch.long, device=self.device)
            temperature = active_generations[slots[0]][0].temperature
            
            t0 = time.perf_counter()
            logits = self._decode_step(input_ids, positions, slot_indices)
            sampled = self._sample_tokens(logits, temperature)
            decode_times.append((batch_size, time.perf_counter() - t0))
            
            # Update progress bar
            if pbar and decode_times:
                tok_per_sec = sum(n for n, _ in decode_times) / max(sum(t for _, t in decode_times), 1e-9)
                pbar.set_postfix_str(f"batch={batch_size} tok/s={tok_per_sec:.0f}")
            
            # Phase 5: Process generated tokens
            for i, slot in enumerate(slots):
                tok = sampled[i]
                req, prompt_idx, tokens, pos = active_generations[slot]
                tokens.append(tok)
                pos += 1
                
                # Check completion
                should_stop = (
                    len(tokens) >= req.max_tokens or
                    pos >= self.max_seq_len or
                    (not req.ignore_eos and tok == self.config.eos_token_id)
                )
                
                if should_stop:
                    text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    req.results[prompt_idx] = GenerationOutput(
                        text, tokens, 
                        len(req.token_ids[prompt_idx]), 
                        len(tokens)
                    )
                    del active_generations[slot]
                    self.model.clear_slot(slot)
                    bisect.insort(free_slots, slot)
                    if pbar:
                        pbar.update(1)
                    
                    if all(r is not None for r in req.results):
                        req.future.set_result(req.results)
                else:
                    active_generations[slot] = (req, prompt_idx, tokens, pos)
        
        if pbar:
            pbar.close()


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    MODEL_PATH = "/mnt/data0/Qwen30.6B"
    
    # Initialize engine
    llm = LLM(MODEL_PATH, max_num_seqs=32, max_seq_len=2048)
    
    # Simple generation
    print("\n=== Simple Generation ===")
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming."
    ]
    
    future = llm.generate(prompts, max_tokens=50, temperature=0.7)
    results = future.result()
    
    for i, result in enumerate(results):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {result.text}")
        print(f"Tokens: {result.prompt_tokens} prompt + {result.generated_tokens} generated")
    
    # Chat example
    print("\n=== Chat Example ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    
    future = llm.chat(messages, max_tokens=100, temperature=0.7)
    result = future.result()[0]
    print(f"Response: {result.text}")
    
    # Stop engine
    llm.stop()
