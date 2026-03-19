#!/usr/bin/env python3
"""Simple test script for Qwen 3 0.6B inference engine"""
from llm import LLM

MODEL_PATH = "/mnt/data0/Qwen30.6B"

def test_basic_generation():
    """Test basic text generation"""
    print("\n=== Test 1: Basic Generation ===")
    llm = LLM(MODEL_PATH, max_num_seqs=8, max_seq_len=1024)
    
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
        print(f"Stats: {result.prompt_tokens} prompt + {result.generated_tokens} generated tokens")
    
    llm.stop()
    print("\n✓ Test 1 passed")


def test_chat():
    """Test chat interface"""
    print("\n=== Test 2: Chat Interface ===")
    llm = LLM(MODEL_PATH, max_num_seqs=8, max_seq_len=1024)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ]
    
    future = llm.chat(messages, max_tokens=100, temperature=0.7)
    result = future.result()[0]
    
    print(f"\nUser: {messages[1]['content']}")
    print(f"Assistant: {result.text}")
    print(f"Stats: {result.prompt_tokens} prompt + {result.generated_tokens} generated tokens")
    
    llm.stop()
    print("\n✓ Test 2 passed")


def test_continuous_batching():
    """Test continuous batching with staggered requests"""
    print("\n=== Test 3: Continuous Batching ===")
    llm = LLM(MODEL_PATH, max_num_seqs=8, max_seq_len=1024)
    
    import time
    import threading
    
    results = []
    
    def submit_request(prompt, delay):
        time.sleep(delay)
        future = llm.generate([prompt], max_tokens=30, temperature=0.0)
        result = future.result()[0]
        results.append((prompt, result))
        print(f"  Completed: {prompt[:30]}...")
    
    # Submit requests with staggered timing
    threads = []
    prompts = [
        ("First request", 0.0),
        ("Second request", 0.5),
        ("Third request", 1.0),
        ("Fourth request", 1.5),
    ]
    
    print("Submitting staggered requests...")
    for prompt, delay in prompts:
        t = threading.Thread(target=submit_request, args=(prompt, delay))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    print(f"\n✓ Test 3 passed - {len(results)} requests completed")
    llm.stop()


def test_temperature():
    """Test different temperature settings"""
    print("\n=== Test 4: Temperature Settings ===")
    llm = LLM(MODEL_PATH, max_num_seqs=8, max_seq_len=1024)
    
    prompt = "Once upon a time"
    temperatures = [0.0, 0.5, 1.0]
    
    for temp in temperatures:
        future = llm.generate([prompt], max_tokens=30, temperature=temp)
        result = future.result()[0]
        print(f"\nTemperature {temp}: {result.text}")
    
    llm.stop()
    print("\n✓ Test 4 passed")


if __name__ == "__main__":
    print("Starting Qwen 3 0.6B tests...")
    
    try:
        test_basic_generation()
        test_chat()
        test_continuous_batching()
        test_temperature()
        
        print("\n" + "="*50)
        print("All tests passed! ✓")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
