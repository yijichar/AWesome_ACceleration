# launch/launch_pd_demo.py
import torch

from kv_transfer.local_connector import LocalRegistryConnector
from engine.prefill_engine_tp import PrefillEngineTP
from engine.decode_engine_tp import DecodeEngineTP
from scheduler.pd_router import PDRouter


def main():
    model_path = "/mnt/data0/Qwen30.6B"

    connector = LocalRegistryConnector()

    prefill_engine = PrefillEngineTP(
        model_path=model_path,
        connector=connector,
        instance_id="prefill-0",
        max_num_seqs=8,
        max_seq_len=1024,
        dtype=torch.bfloat16,
        enable_tp=False,   # 先单卡骨架跑通
        tp_size=1,
    )

    decode_engine = DecodeEngineTP(
        model_path=model_path,
        connector=connector,
        instance_id="decode-0",
        max_num_seqs=8,
        max_seq_len=1024,
        dtype=torch.bfloat16,
        enable_tp=False,
        tp_size=1,
    )

    router = PDRouter(
        prefill_engine=prefill_engine,
        decode_engine=decode_engine,
    )

    prompts = [
        "你好，介绍一下什么是 PD 分离。",
        "请用通俗语言解释 KV Cache 为什么能加速大模型推理。",
        "写一首关于 GPU 的短诗。",
    ]

    for i, p in enumerate(prompts):
        out = router.generate(
            prompt=p,
            max_tokens=64,
            temperature=0.0,
            ignore_eos=False,
        )
        print("=" * 80)
        print(f"[{i}] request_id={out.request_id}")
        print(f"prompt_tokens={out.prompt_tokens}, generated_tokens={out.generated_tokens}, finish={out.finish_reason}")
        print(out.text)


if __name__ == "__main__":
    main()