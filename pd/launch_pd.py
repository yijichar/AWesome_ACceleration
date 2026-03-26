# pd/launch_pd.py
import argparse
import torch

from pd.prefill_server import PrefillServer
from pd.decode_server import DecodeServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=str, required=True, choices=["prefill", "decode"])
    parser.add_argument("--server-id", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--kv-base-dir", type=str, default="/dev/shm/pd_kv_cache")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    if args.role == "prefill":
        srv = PrefillServer(
            server_id=args.server_id,
            host=args.host,
            port=args.port,
            model_path=args.model_path,
            device=args.device,
            kv_base_dir=args.kv_base_dir,
            max_num_seqs=args.max_num_seqs,
            max_seq_len=args.max_seq_len,
            dtype=dtype,
        )
    else:
        srv = DecodeServer(
            server_id=args.server_id,
            host=args.host,
            port=args.port,
            model_path=args.model_path,
            device=args.device,
            kv_base_dir=args.kv_base_dir,
            max_num_seqs=args.max_num_seqs,
            max_seq_len=args.max_seq_len,
            dtype=dtype,
        )

    srv.serve_forever()


if __name__ == "__main__":
    main()