"""OpenAI vision target used by released CS-DJ, JOOD, and MML experiments."""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from collections.abc import Sequence

from data_paths import resolve_path


class OpenAIVisionTarget:
    """Chat Completions wrapper matching the three upstream repositories."""

    def __init__(
        self,
        model: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        retry_limit: int = 3,
        retry_wait_seconds: float | None = None,
        bad_request_fallback: str | None = None,
    ):
        self.name = model
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.retry_limit = retry_limit
        self.retry_wait_seconds = retry_wait_seconds
        self.bad_request_fallback = bad_request_fallback
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is required for the official OpenAI target "
                    f"{self.model!r}"
                )
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    @staticmethod
    def _data_url(path: str) -> str:
        resolved = resolve_path(path)
        mime_type = mimetypes.guess_type(resolved)[0] or "application/octet-stream"
        with open(resolved, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def generate(self, text: str, image_paths: Sequence[str]) -> str:
        content: list[dict] = [{"type": "text", "text": text}]
        content.extend({
            "type": "image_url",
            "image_url": {"url": self._data_url(path)},
        } for path in image_paths)

        last_error: Exception | None = None
        for attempt in range(self.retry_limit):
            try:
                response = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )
                message = response.choices[0].message
                return getattr(message, "refusal", None) or message.content or ""
            except Exception as exc:  # SDK exception types vary by version
                last_error = exc
                if self.bad_request_fallback is not None and "Error code: 400" in str(exc):
                    return self.bad_request_fallback
                if attempt + 1 < self.retry_limit:
                    delay = (
                        self.retry_wait_seconds
                        if self.retry_wait_seconds is not None
                        else min(2 ** attempt, 5)
                    )
                    time.sleep(delay)
        assert last_error is not None
        raise last_error
