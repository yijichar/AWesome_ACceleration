# scheduler/pd_router.py
from dataclasses import dataclass

from scheduler.session import RequestSession, SamplingParams
from engine.prefill_engine_tp import PrefillEngineTP
from engine.decode_engine_tp import DecodeEngineTP


@dataclass
class PDResult:
    request_id: str
    text: str
    token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str | None


class PDRouter:
    def __init__(
        self,
        prefill_engine: PrefillEngineTP,
        decode_engine: DecodeEngineTP,
    ):
        self.prefill_engine = prefill_engine
        self.decode_engine = decode_engine

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        ignore_eos: bool = False,
    ) -> PDResult:
        request_id = self.prefill_engine.new_request_id()
        prompt_token_ids = self.prefill_engine.encode_prompt(prompt)

        session = RequestSession(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                ignore_eos=ignore_eos,
            ),
        )

        # 1) prefill
        prefill_out = self.prefill_engine.prefill(
            request_id=session.request_id,
            prompt_token_ids=session.prompt_token_ids,
        )
        session.prefill_instance_id = self.prefill_engine.instance_id
        session.kv_handle = prefill_out.kv_handle
        session.prompt_len = prefill_out.prompt_len
        session.prefill_done = True

        # 2) attach 到 decode
        self.decode_engine.attach_session(
            request_id=session.request_id,
            kv_handle=session.kv_handle,
            prompt_len=session.prompt_len,
            sampling_params=session.sampling_params,
        )
        session.decode_instance_id = self.decode_engine.instance_id

        # 3) decode bootstrap：用 prompt 最后一个 token 触发第一步生成
        last_prompt_token = session.prompt_token_ids[-1]
        first_token = self.decode_engine.bootstrap_first_token(
            request_id=session.request_id,
            last_prompt_token_id=last_prompt_token,
        )
        session.generated_token_ids.append(first_token)

        # 4) decode until finish
        while True:
            state = self.decode_engine.active[session.request_id]
            if state.finished:
                break
            tok = self.decode_engine.step_one_token(session.request_id)
            session.generated_token_ids.append(tok)

        text, token_ids, finish_reason = self.decode_engine.collect_result(session.request_id)

        # 5) cleanup
        self.decode_engine.release_session(session.request_id)
        self.prefill_engine.release_prefill_kv(session.kv_handle)

        return PDResult(
            request_id=session.request_id,
            text=text,
            token_ids=token_ids,
            prompt_tokens=session.prompt_len,
            generated_tokens=len(token_ids),
            finish_reason=finish_reason,
        )