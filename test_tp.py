#!/usr/bin/env python3
"""
测试 Tensor Parallel 功能

启动方式：
    # 单卡（baseline）
    python test_tp.py
    
    # 2 卡 TP
    torchrun --nproc_per_node=2 test_tp.py
    
    # 4 卡 TP
    torchrun --nproc_per_node=4 test_tp.py
"""
import os
import sys
import torch

# 检测是否在分布式环境
IS_DISTRIBUTED = "RANK" in os.environ

if IS_DISTRIBUTED:
    from llm_tp import LLMTP as LLM
    from model.distributed import is_tp_rank_0, get_tp_rank
else:
    from llm import LLM
    def is_tp_rank_0():
        return True
    def get_tp_rank():
        return 0

MODEL_PATH = "/mnt/data0/Qwen30.6B"


def test_basic_generation():
    """Test 1: Basic text generation"""
    if is_tp_rank_0():
        print("\n" + "="*60)
        print("Test 1: Basic Generation")
        print("="*60)
    
    llm = LLM(MODEL_PATH, max_num_seqs=8, max_seq_len=1024, enable_tp=IS_DISTRIBUTED)
    
    if is_tp_rank_0():
        prompts = [
            "Hello, how are you?",
            "What is 2+2?",
            "Write a haiku about AI."
        ]
        
        future = llm.generate(prompts, max_tokens=50, temperature=0.7)
        results = future.result()
        
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Response: {result.text}")
            print(f"Stats: {result.prompt_tokens} prompt + {result.generated_tokens} generated")
        
        llm.stop()
        print("\n✓ Test 1 passed")
    else:
        # Worker ranks just wait
        import signal
        signal.pause()


def test_chat():
    """Test 2: Chat interface"""
    if is_tp_rank_0():
        print("\n" + "="*60)
        print("Test 2: Chat Interface")
        print("="*60)
    
    llm = LLM(MODEL_PATH, max_num_seqs=8, max_seq_len=1024, enable_tp=IS_DISTRIBUTED)
    
    if is_tp_rank_0():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"}
        ]
        
        future = llm.chat(messages, max_tokens=100, temperature=0.7)
        result = future.result()[0]
        
        print(f"\nUser: {messages[1]['content']}")
        print(f"Assistant: {result.text}")
        print(f"Stats: {result.prompt_tokens} prompt + {result.generated_tokens} generated")
        
        llm.stop()
        print("\n✓ Test 2 passed")
    else:
        import signal
        signal.pause()


def test_throughput():
    """Test 3: Throughput comparison"""
    if is_tp_rank_0():
        print("\n" + "="*60)
        print("Test 3: Throughput Test")
        print("="*60)
    
    llm = LLM(MODEL_PATH, max_num_seqs=16, max_seq_len=1024, enable_tp=IS_DISTRIBUTED)
    
    if is_tp_rank_0():
        import time
        
        # Generate 10 prompts
        prompts = [f"Tell me about topic {i}" for i in range(10)]
        
        start = time.time()
        future = llm.generate(prompts, max_tokens=50, temperature=0.0)
        results = future.result()
        elapsed = time.time() - start
        
        total_tokens = sum(r.generated_tokens for r in results)
        throughput = total_tokens / elapsed
        
        print(f"\nGenerated {total_tokens} tokens in {elapsed:.2f}s")
        print(f"Throughput: {throughput:.1f} tokens/s")
        
        llm.stop()
        print("\n✓ Test 3 passed")
    else:
        import signal
        signal.pause()


if __name__ == "__main__":
    try:
        if is_tp_rank_0():
            print("\n" + "="*60)
            if IS_DISTRIBUTED:
                print("Running with Tensor Parallel")
            else:
                print("Running on single GPU")
            print("="*60)
        
        # Run tests
        test_basic_generation()
        
        if is_tp_rank_0():
            print("\n" + "="*60)
            print("All tests passed! ✓")
            print("="*60 + "\n")
        
    except Exception as e:
        if is_tp_rank_0():
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)
