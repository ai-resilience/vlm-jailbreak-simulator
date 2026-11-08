"""Qwen2-VL-7B target wrapper for JPS-aligned evaluation."""

from __future__ import annotations

from typing import Sequence

from .base import TargetModel
from .paths import resolve_path


class Qwen2VLTarget(TargetModel):
    """Greedy single-image inference through the centralized pinned loader."""

    name = "qwen2_vl_7b"
    min_images = 1
    max_images = 1
    model_id = None
    revision = None

    def __init__(self, max_new_tokens: int = 256):
        from multimodalmodels.loaders import load_qwen2vl
        from multimodalmodels.registry import QWEN2_VL_HF_ID, QWEN2_VL_REVISION

        model_id = self.model_id or QWEN2_VL_HF_ID
        revision = self.revision if self.model_id is not None else QWEN2_VL_REVISION

        self.model, self.processor = load_qwen2vl(
            model_id,
            revision=revision,
            processor_kwargs={"use_fast": False},
        )
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def _resolve(image_paths: Sequence[str]) -> str:
        if len(image_paths) != 1:
            raise ValueError(
                f"Qwen2-VL target expects exactly one image; got {len(image_paths)}"
            )
        return resolve_path(image_paths[0])

    def generate(self, text: str, image_paths: Sequence[str]) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        image_path = self._resolve(image_paths)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text},
            ],
        }]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[rendered],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
            )
        generated = output[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def generate_batch(
        self, prompts: Sequence[tuple[str, Sequence[str]]]
    ) -> list[str]:
        # Sequential processing is intentional: it supports different image
        # aspect ratios without relying on qwen-vl-utils batch flattening.
        return [self.generate(text, images) for text, images in prompts]


class Qwen25VLTarget(Qwen2VLTarget):
    """Qwen2.5-VL target using the same official Qwen vision utility path."""

    from multimodalmodels.registry import QWEN25_VL_HF_ID, QWEN25_VL_REVISION

    name = "qwen2_5_vl_7b"
    model_id = QWEN25_VL_HF_ID
    revision = QWEN25_VL_REVISION
