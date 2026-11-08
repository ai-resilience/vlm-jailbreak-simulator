"""LLaVA-1.5 target model adapted to the shared target interface.

Reuses the existing multimodalmodels.LLaVA_v1_5 (weights, conversation template,
greedy-decoding settings) but exposes ``generate(text, image_paths)`` with
repo-root-relative paths and a true batched path.

LLaVA-1.5 is single-image; this wrapper accepts either one image or no image.
The latter exposes the language-only path used by ECSO's "eyes closed" stage.
Multi-image attacks (DMN, CS-DJ) should target a multi-image VLM.
"""

from __future__ import annotations

from typing import Sequence

from .paths import resolve_path
from .base import TargetModel


class LlavaTarget(TargetModel):
    name = "llava_v1_5"
    max_images = 1

    def __init__(self, max_new_tokens: int = 256):
        from multimodalmodels import LLaVA_v1_5

        self._model = LLaVA_v1_5()
        self.max_new_tokens = max_new_tokens
        self.do_sample = False
        self.temperature = 1.0
        self.top_p = 1.0

    def _resolve(self, image_paths: Sequence[str]) -> str | None:
        if not image_paths:
            return None
        if len(image_paths) != 1:
            raise ValueError(
                f"LLaVA-1.5 accepts zero or one image; got {len(image_paths)}. "
                "Use a multi-image target for this attack."
            )
        # data/... follows VLM_JAILBREAK_DATA_ROOT; other relative paths are repo-relative
        return resolve_path(image_paths[0])

    def generate(self, text: str, image_paths: Sequence[str]) -> str:
        return self.generate_batch([(text, image_paths)])[0]

    def generate_batch(self, prompts: Sequence[tuple[str, Sequence[str]]]) -> list[str]:
        # The underlying generate() reads images as (image_path, prompt) pairs
        # relative to image_dir; pass absolute paths with image_dir="".
        cases = [(self._resolve(imgs), text) for text, imgs in prompts]
        generation = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "num_beams": 1,
            "progress_bar": False,
            "use_cache": True,
        }
        if self.do_sample:
            generation.update(temperature=self.temperature)
            if self.top_p is not None:
                generation.update(top_p=self.top_p)
        return self._model.generate(
            cases,
            image_dir="",
            **generation,
        )
