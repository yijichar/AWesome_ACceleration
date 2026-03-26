# engine/decode_engine_tp.py
from dataclasses import dataclass
import os
import torch
import torch.nn.functional as F

from engine.engine_base import EngineBase
from kv_transfer.types import KVHandle
from scheduler.session import SamplingParams
from model.model_tp import Qwen3Config, Qwen3ForCausalLMTP


@dataclass
class DecodeState:
    request_id: str
    local_slot: int
    prompt_len: int
    current_position: int
    sampling_params: SamplingParams
    generated_token_ids: list[int]
    finished: bool = False
    finish_reason: str | None = None


class DecodeEngineTP(EngineBase):
    def __init__(
        self,
        model_path: str,
        connector,
        instance_id: str = "decode-0",
        max_num_seqs: int = 32,
        max_seq_len: int = 1024,
        dtype=torch.bfloat16,
        enable_tp: bool = False,
        tp_size: int = 1,
    ):
        super().__init__(
            model_path=model_path,
            connector=connector,
            instance_id=instance_id,
            max_num_seqs=max_num_seqs,
            max_seq_len=max_seq_len,
            dtype=dtype,
            enable_tp=enable_tp,
            tp_size=tp_size,
            role="decode",
        )

        config_path = os.path.join(model_path, "config.json")
        self.config = Qwen3Config.from_json(config_path)

        self.model = Qwen3ForCausalLMTP.from_pretrained(
            model_path=model_path,
            config=self.config,
            device=self.device,
            dtype=dtype,
        )
        self.model.eval()
        self.model.init_kv_cache(self.max_num_seqs, self.max_seq_len, self.device, dtype)

        self.active: dict[str, DecodeState] = {}

    def _sample(self, logits: torch.Tensor, temperature: float) -> int:
        if temperature == 0.0:
            return int(logits.argmax(dim=-1).item())
        probs = F.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    @torch.inference_mode()
    def attach_session(
        self,
        request_id: str,
        kv_handle: KVHandle,
        prompt_len: int,
        sampling_params: SamplingParams,
    ) -> None:
        local_slot = self.cache_manager.allocate_slot()
        try:
            payload = self.connector.load_kv(kv_handle)

            self.model.import_request_kv(
                slot_idx=local_slot,
                payload=payload,
                length=prompt_len,
            )

            self.active[request_id] = DecodeState(
                request_id=request_id,
                local_slot=local_slot,
                prompt_len=prompt_len,
                current_position=prompt_len,
                sampling_params=sampling_params,
                generated_token_ids=[],
                finished=False,
                finish_reason=None,
            )
        except Exception:
            self.model.clear_slot(local_slot)
            self.cache_manager.free_slot(local_slot)
            raise

    @torch.inference_mode()
    def step_one_token(self, request_id: str) -> int:
        state = self.active[request_id]
        if state.finished:
            raise RuntimeError(f"request_id={request_id} already finished")

        if len(state.generated_token_ids) == 0:
            # 第一步 decode 的输入 token = prompt 最后一个 token
            # 为了不额外保存 prompt，这里简单约定：
            # attach 前由 router 传入首个 decode input 更合理。
            # 当前 MVP 里，我们直接用 eos 作为占位不可行，
            # 所以这里改成要求外部先通过 warm-start 给 last_token。
            raise RuntimeError(
                "DecodeState has no bootstrap token. "
                "Use bootstrap_first_token() before step_one_token() in this MVP."
            )

        input_token = state.generated_token_ids[-1]
        input_ids = torch.tensor([[input_token]], dtype=torch.long, device=self.device)
        positions = torch.tensor([[state.current_position]], dtype=torch.long, device=self.device)
        slot_indices = torch.tensor([state.local_slot], dtype=torch.long, device=self.device)

        logits = self.model(input_ids, positions, slot_indices)[:, 0, :]
        next_token = self._sample(logits, state.sampling_params.temperature)

        state.generated_token_ids.append(next_token)
        state.current_position += 1

        if (
            len(state.generated_token_ids) >= state.sampling_params.max_tokens
            or state.current_position >= self.max_seq_len
            or (
                not state.sampling_params.ignore_eos
                and next_token == self.config.eos_token_id
            )
        ):
            state.finished = True
            if next_token == self.config.eos_token_id:
                state.finish_reason = "eos"
            elif state.current_position >= self.max_seq_len:
                state.finish_reason = "length_cap"
            else:
                state.finish_reason = "max_tokens"

        return next_token

    @torch.inference_mode()
    def bootstrap_first_token(
        self,
        request_id: str,
        last_prompt_token_id: int,
    ) -> int:
        """
        让 decode 侧自己产生第一个生成 token。
        输入是 prompt 最后一个 token。
        """
        state = self.active[request_id]
        input_ids = torch.tensor([[last_prompt_token_id]], dtype=torch.long, device=self.device)
        positions = torch.tensor([[state.prompt_len - 1]], dtype=torch.long, device=self.device)
        slot_indices = torch.tensor([state.local_slot], dtype=torch.long, device=self.device)

        logits = self.model(input_ids, positions, slot_indices)[:, 0, :]
        first_token = self._sample(logits, state.sampling_params.temperature)
        state.generated_token_ids.append(first_token)
        state.current_position = state.prompt_len

        if (
            len(state.generated_token_ids) >= state.sampling_params.max_tokens
            or state.current_position >= self.max_seq_len
            or (
                not state.sampling_params.ignore_eos
                and first_token == self.config.eos_token_id
            )
        ):
            state.finished = True
            if first_token == self.config.eos_token_id:
                state.finish_reason = "eos"
            elif state.current_position >= self.max_seq_len:
                state.finish_reason = "length_cap"
            else:
                state.finish_reason = "max_tokens"

        return first_token

    def collect_result(self, request_id: str) -> tuple[str, list[int], str | None]:
        state = self.active[request_id]
        text = self.decode_tokens(state.generated_token_ids)
        return text, state.generated_token_ids, state.finish_reason

    def release_session(self, request_id: str) -> None:
        state = self.active.pop(request_id)
        self.model.clear_slot(state.local_slot)
        self.cache_manager.free_slot(state.local_slot)