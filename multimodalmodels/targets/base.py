"""Shared target interface, independent of datasets and output schemas."""

from typing import Sequence


class TargetModel:
    name = ""
    min_images = 0
    max_images = None

    def generate(self, text: str, image_paths: Sequence[str]) -> str:
        raise NotImplementedError

    def generate_batch(self, prompts: Sequence[tuple[str, Sequence[str]]]) -> list[str]:
        return [self.generate(text, images) for text, images in prompts]
