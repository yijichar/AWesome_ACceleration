# pd/router_client.py
import time
from dataclasses import asdict

from pd.common import (
    gen_request_id,
    SamplingParams,
    FinalResult,
)
from pd.ipc_protocol import request_reply


class RouterClient:
    def __init__(
        self,
        prefill_servers: list[tuple[str, int]],
        decode_servers: list[tuple[str, int]],
        poll_interval_ms: float = 10.0,
    ):
        self.prefill_servers = prefill_servers
        self.decode_servers = decode_servers
        self.poll_interval_ms = poll_interval_ms
        self._prefill_rr = 0

    def _pick_prefill_server(self) -> tuple[str, int]:
        idx = self._prefill_rr % len(self.prefill_servers)
        self._prefill_rr += 1
        return self.prefill_servers[idx]

    def _pick_decode_server(self) -> tuple[str, int]:
        best = None
        best_free = -1
        for host, port in self.decode_servers:
            resp = request_reply(host, port, {"op": "health"})
            if resp["ok"]:
                free_slots = int(resp.get("free_slots", 0))
                if free_slots > best_free:
                    best_free = free_slots
                    best = (host, port)
        if best is None:
            raise RuntimeError("no decode server available")
        return best

    def generate(
        self,
        prompt: str,
        model_tokenizer,
        max_tokens: int = 128,
        temperature: float = 0.0,
        ignore_eos: bool = False,
    ) -> FinalResult:
        request_id = gen_request_id()

        text = model_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_token_ids = model_tokenizer.encode(text, add_special_tokens=False)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=ignore_eos,
        )

        phost, pport = self._pick_prefill_server()
        prefill_resp = request_reply(
            phost, pport,
            {
                "op": "prefill",
                "request_id": request_id,
                "prompt": prompt,
                "prompt_token_ids": prompt_token_ids,
                "sampling_params": asdict(sampling_params),
            }
        )
        if not prefill_resp["ok"]:
            tb = prefill_resp.get("traceback", "")
            raise RuntimeError(
                f"prefill failed: {prefill_resp.get('error')}\n\n"
                f"=== server traceback ===\n{tb}"
            )

        prefill_done = prefill_resp["data"]

        dhost, dport = self._pick_decode_server()
        attach_resp = request_reply(
            dhost, dport,
            {
                "op": "attach",
                "data": prefill_done,
            }
        )
        if not attach_resp["ok"]:
            tb = attach_resp.get("traceback", "")
            raise RuntimeError(
                f"decode attach failed: {attach_resp.get('error')}\n\n"
                f"=== server traceback ===\n{tb}"
            )

        while True:
            poll_resp = request_reply(
                dhost, dport,
                {
                    "op": "poll",
                    "request_id": request_id,
                }
            )
            if not poll_resp["ok"]:
                tb = poll_resp.get("traceback", "")
                raise RuntimeError(
                    f"decode poll failed: {poll_resp.get('error')}\n\n"
                    f"=== server traceback ===\n{tb}"
                )
            if poll_resp.get("ready", False):
                data = poll_resp["data"]
                return FinalResult(
                    request_id=data["request_id"],
                    text=data["text"],
                    token_ids=data["token_ids"],
                    prompt_tokens=data["prompt_tokens"],
                    generated_tokens=data["generated_tokens"],
                    finish_reason=data["finish_reason"],
                )
            time.sleep(self.poll_interval_ms / 1000.0)