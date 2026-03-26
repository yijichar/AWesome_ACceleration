import os
from transformers import AutoTokenizer
from pd.router_client import RouterClient

MODEL_PATH = "/mnt/data0/Qwen30.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

router = RouterClient(
    prefill_servers=[("127.0.0.1", 9001)],
    decode_servers=[
        ("127.0.0.1", 9101),
        ("127.0.0.1", 9102),
    ],
)

prompts = [
    "你好，解释一下什么是 PD 分离。",
    "请简要说明 KV cache 的作用。",
    "写一首关于 GPU 的短诗。",
]

for p in prompts:
    out = router.generate(
        prompt=p,
        model_tokenizer=tokenizer,
        max_tokens=64,
        temperature=0.0,
        ignore_eos=False,
    )
    print("=" * 80)
    print(out.request_id)
    print(out.text)