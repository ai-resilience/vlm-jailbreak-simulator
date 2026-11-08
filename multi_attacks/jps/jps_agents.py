"""JPS three-agent textual steering (feedback -> summarize -> revision).

The official implementation uses a local Qwen2.5-14B-Instruct model with
separate system/user roles and greedy decoding. ``huggingface`` reproduces
that path; The simulator integration exposes this local backend only.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

from . import jps_prompts as P


class _ChatBackend(Protocol):
    def chat(self, system: str, user: str, *, max_tokens: int) -> str | None: ...

    def chat_batch(self, system: str, users: list[str], *,
                   max_tokens: int) -> list[str | None]: ...


class _HuggingFaceChat:
    def __init__(self, model: str, revision: str | None = None):
        from multimodalmodels.loaders import load_causal_lm_agent

        self.model, self.tokenizer = load_causal_lm_agent(model, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def chat(self, system: str, user: str, *, max_tokens: int) -> str:
        return self.chat_batch(system, [user], max_tokens=max_tokens)[0]

    def chat_batch(self, system: str, users: list[str], *,
                   max_tokens: int) -> list[str]:
        import torch

        conversations = [
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            for user in users
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in conversations
        ]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[:, inputs["input_ids"].shape[1]:]
        return [text.strip() for text in self.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )]


def _extract_steer(text: str) -> str | None:
    """Pull the final steer prompt from the official ``@@...@@`` format."""
    matches = re.findall(r"@@(.*?)@@", text or "", re.DOTALL)
    return matches[-1].strip() if matches else None


class JPSAgents:
    """Run JPS's three agents for both official revision angles."""

    def __init__(self, model: str = "Qwen/Qwen2.5-14B-Instruct",
                 backend: str = "huggingface", strict: bool = True,
                 revision: str | None = None,
                 revision_retries: int = 4,
                 trace_path: str | None = None,
                 chat_backend: _ChatBackend | None = None):
        if revision_retries < 0:
            raise ValueError("JPS revision_retries must be non-negative")
        self.model = model
        self.backend = backend
        self.strict = strict
        self.revision_retries = revision_retries
        self.trace_path = trace_path
        if chat_backend is not None:
            self._backend = chat_backend
        elif backend == "huggingface":
            self._backend = _HuggingFaceChat(model, revision=revision)
        else:
            raise ValueError(
                f"Unknown JPS agent_backend {backend!r}; use 'huggingface'"
            )

    def _record(self, *, stage: str, angle: str, attempt: int,
                response: str, valid: bool, round_index: int | None,
                item_index: int | None = None) -> None:
        """Append one raw agent response and its parser verdict to JSONL."""
        if not self.trace_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.trace_path)), exist_ok=True)
        record = {
            "round": round_index,
            "stage": stage,
            "angle": angle,
            "attempt": attempt,
            "valid": valid,
            "response": response,
            "model": self.model,
            "backend": self.backend,
        }
        if item_index is not None:
            record["item_index"] = item_index
        with open(self.trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _chat(self, system: str, user: str, max_tokens: int) -> str:
        result = self._backend.chat(system, user, max_tokens=max_tokens)
        if result:
            return result
        if self.strict:
            raise RuntimeError("JPS agent returned no response")
        return ""

    def _chat_batch(self, system: str, users: list[str], max_tokens: int) -> list[str]:
        if hasattr(self._backend, "chat_batch"):
            results = self._backend.chat_batch(
                system, users, max_tokens=max_tokens
            )
        else:  # small injected test doubles / third-party backends
            results = [
                self._backend.chat(system, user, max_tokens=max_tokens)
                for user in users
            ]
        if len(results) != len(users):
            raise RuntimeError(
                f"JPS agent returned {len(results)} responses for {len(users)} prompts"
            )
        if self.strict and any(not result for result in results):
            raise RuntimeError("JPS agent returned an empty batched response")
        return [result or "" for result in results]

    def revise(self, qa_list: list[tuple[str, str]], prompt_list: list[str],
               *, round_index: int | None = None) -> list[str]:
        if len(prompt_list) != len(P.ANGLES):
            raise ValueError(f"JPS needs {len(P.ANGLES)} steer prompts, got {len(prompt_list)}")

        new_prompts = list(prompt_list)
        for index, angle in enumerate(P.ANGLES):
            qa_feedback = []
            feedback_users = [
                P.FEEDBACK_PROMPT[angle].format(question=question, response=response)
                for question, response in qa_list
            ]
            feedback_results = self._chat_batch(
                P.FEEDBACK_SYSTEM, feedback_users, 100
            )
            for item_index, ((question, response), feedback) in enumerate(
                zip(qa_list, feedback_results)
            ):
                valid_feedback = bool(re.match(
                    r"Rating:\s*\[\[(\d+)\]\]\s*Suggestion:\s*(.*)",
                    feedback,
                    re.DOTALL,
                ))
                self._record(
                    stage="feedback", angle=angle, attempt=0,
                    response=feedback, valid=valid_feedback,
                    round_index=round_index, item_index=item_index,
                )
                if valid_feedback:
                    qa_feedback.append((question, response, feedback))

            if not qa_feedback:
                if self.strict:
                    raise RuntimeError(f"JPS {angle} feedback agent produced no valid ratings")
                continue

            previous = prompt_list[index]
            history = ""
            if previous:
                history = (
                    "The following prompt is the steer prompt that used to send "
                    f'to the language model: "{previous}"\n'
                    "When providing suggestions, try to think more about the "
                    "shortcomings of this prompt. The current responses are all "
                    "generated by combining the question with this prompt, then "
                    "using it with the target LLM."
                )
            qa_feedback_text = ""
            for number, (question, answer, evaluation) in enumerate(qa_feedback, 1):
                qa_feedback_text += (
                    f'"\nQuestion {number}: "{question}"\n'
                    f'Answer {number}: "{answer}"\n'
                    f'Evaluation {number}: "{evaluation}"\n"\n'
                )
            summary = self._chat(
                P.FEEDBACK_SYSTEM,
                P.SUMMARIZE_PROMPT[angle].format(
                    qa_feedback_prompt=qa_feedback_text, history_prompt=history
                ),
                8192,
            )
            self._record(
                stage="summary", angle=angle, attempt=0,
                response=summary, valid=bool(summary.strip()),
                round_index=round_index,
            )

            revision_history = ""
            if previous:
                revision_history = (
                    "The following prompt is the steer prompt you previously "
                    f'generated and used to send to the language model: "{previous}"\n'
                    "Refer to the previous prompt you generated and improve it "
                    "to make it more effective, refined, and enhanced for better results."
                )
            revision_prompt = P.REVISION_PROMPT[angle].format(
                qa_summarize=summary, history_prompt=revision_history
            )
            steer = None
            previous_output = ""
            for attempt in range(self.revision_retries + 1):
                if attempt == 0:
                    user_prompt = revision_prompt
                else:
                    user_prompt = (
                        revision_prompt
                        + "\n\nFORMAT CORRECTION REQUIRED:\n"
                        + "Your previous response omitted or malformed the required "
                          "delimiters. Preserve the intended revised steer prompt, but "
                          "return it with exactly one non-empty prompt enclosed as "
                          "@@YOUR COMPLETE REVISED STEER PROMPT@@. Do not omit the "
                          "opening or closing @@ delimiters.\n\n"
                        + "Previous response:\n"
                        + previous_output
                    )
                output = self._chat(P.FEEDBACK_SYSTEM, user_prompt, 8192)
                steer = _extract_steer(output)
                self._record(
                    stage="revision", angle=angle, attempt=attempt,
                    response=output, valid=bool(steer),
                    round_index=round_index,
                )
                if steer:
                    break
                previous_output = output
            if not steer:
                if self.strict:
                    trace_hint = (
                        f"; raw responses: {self.trace_path}"
                        if self.trace_path else ""
                    )
                    raise RuntimeError(
                        f"JPS {angle} revision did not contain an @@...@@ steer prompt "
                        f"after {self.revision_retries + 1} attempts{trace_hint}"
                    )
                continue
            new_prompts[index] = steer
        return new_prompts
