"""Targets for native Hugging Face multimodal chat checkpoints.

The three families here expose the same :class:`multimodalmodels.targets.base.TargetModel`
surface while retaining their official processor chat templates.  Model
construction is centralized in :mod:`multimodalmodels.loaders` and therefore
uses the pinned repository revisions from ``multimodalmodels/registry.py``.
"""

from __future__ import annotations

from typing import Sequence

from .base import TargetModel
from .paths import resolve_path


LLAVA_QWEN_15_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n'}}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\\n' }}"
    "{% endfor %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] }}"
    "{% endfor %}"
    "{{'<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


class NativeHFVLMTarget(TargetModel):
    """Shared deterministic generation path for modern native-HF VLMs."""

    model_id: str = ""
    revision: str | None = None
    max_images: int | None = None
    torch_dtype: str = "auto"
    processor_kwargs: dict | None = None
    chat_template: str | None = None

    def __init__(self, max_new_tokens: int = 256):
        from multimodalmodels.loaders import load_native_vlm

        self.model, self.processor = load_native_vlm(
            self.model_id,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            eval_mode=True,
            processor_kwargs=self.processor_kwargs,
        )
        self.max_new_tokens = max_new_tokens
        self.do_sample = False
        self.temperature = 1.0
        self.top_p = 1.0

    def _images(self, image_paths: Sequence[str]):
        from PIL import Image

        if self.max_images is not None and len(image_paths) > self.max_images:
            raise ValueError(
                f"{self.name} accepts at most {self.max_images} image(s); "
                f"got {len(image_paths)}"
            )
        # Copy pixels before closing the files.  Processor chat templates can
        # then consume ordinary PIL objects without holding file descriptors.
        images = []
        for path in image_paths:
            with Image.open(resolve_path(path)) as image:
                images.append(image.convert("RGB").copy())
        return images

    @staticmethod
    def _messages(text: str, images) -> list[dict]:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def generate(self, text: str, image_paths: Sequence[str]) -> str:
        import torch

        messages = self._messages(text, self._images(image_paths))
        template_kwargs = {}
        if self.chat_template is not None:
            template_kwargs["chat_template"] = self.chat_template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        ).to(self.model.device)

        generation = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generation["temperature"] = self.temperature
            if self.top_p is not None:
                generation["top_p"] = self.top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def generate_batch(
        self, prompts: Sequence[tuple[str, Sequence[str]]]
    ) -> list[str]:
        # Sequential execution keeps heterogeneous aspect ratios and image
        # counts valid across all three processor families.
        return [self.generate(text, images) for text, images in prompts]


class LlavaNextTarget(NativeHFVLMTarget):
    from multimodalmodels.registry import LLAVA_16_HF_ID, LLAVA_16_REVISION

    name = "llava_v1_6"
    model_id = LLAVA_16_HF_ID
    revision = LLAVA_16_REVISION
    max_images = 1


class LlavaOneVision7BChatTarget(NativeHFVLMTarget):
    from multimodalmodels.registry import (LLAVA_ONEVISION_7B_CHAT_HF_ID,
                                           LLAVA_ONEVISION_7B_CHAT_REVISION)

    name = "llava_onevision_7b_chat"
    model_id = LLAVA_ONEVISION_7B_CHAT_HF_ID
    revision = LLAVA_ONEVISION_7B_CHAT_REVISION
    max_images = 1
    # The converted checkpoint declares float16 at the composite-model level
    # but bfloat16 inside text_config. ``auto`` consequently loads mixed
    # weights and fails at lm_head (BF16 activations x FP16 weights).  The HF
    # model card's generation example explicitly loads the whole model in FP16.
    torch_dtype = "float16"
    # The fast processor is pathologically slow for CS-DJ's 1500x2120 collage
    # under Transformers 4.57.1.  The slow processor is pixel-equivalent here
    # and matches the conversion validation path.
    processor_kwargs = {"use_fast": False}
    chat_template = LLAVA_QWEN_15_CHAT_TEMPLATE

    @staticmethod
    def _messages(text: str, images) -> list[dict]:
        """Match the original LLaVA Qwen conversation's default system turn."""
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": text})
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ],
            },
            {"role": "user", "content": content},
        ]


class Qwen3VL2BTarget(NativeHFVLMTarget):
    from multimodalmodels.registry import QWEN3_VL_2B_HF_ID, QWEN3_VL_2B_REVISION

    name = "qwen3_vl_2b"
    model_id = QWEN3_VL_2B_HF_ID
    revision = QWEN3_VL_2B_REVISION


class Qwen3VL8BTarget(NativeHFVLMTarget):
    from multimodalmodels.registry import QWEN3_VL_8B_HF_ID, QWEN3_VL_8B_REVISION

    name = "qwen3_vl_8b"
    model_id = QWEN3_VL_8B_HF_ID
    revision = QWEN3_VL_8B_REVISION


class InternVL3_2BTarget(NativeHFVLMTarget):
    from multimodalmodels.registry import INTERNVL3_2B_HF_ID, INTERNVL3_2B_REVISION

    name = "internvl3_2b"
    model_id = INTERNVL3_2B_HF_ID
    revision = INTERNVL3_2B_REVISION


class InternVL3_8BTarget(NativeHFVLMTarget):
    from multimodalmodels.registry import INTERNVL3_8B_HF_ID, INTERNVL3_8B_REVISION

    name = "internvl3_8b"
    model_id = INTERNVL3_8B_HF_ID
    revision = INTERNVL3_8B_REVISION


__all__ = [
    "NativeHFVLMTarget",
    "LlavaNextTarget",
    "LlavaOneVision7BChatTarget",
    "Qwen3VL2BTarget",
    "Qwen3VL8BTarget",
    "InternVL3_2BTarget",
    "InternVL3_8BTarget",
]
