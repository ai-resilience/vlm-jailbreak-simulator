"""Target adapters ported from core/targets into the existing model package."""

from functools import partial

from ..registry import (
    MODEL_ALIASES, LLAVA_15_HF_ID, LLAVA_16_HF_ID, LLAVA_ONEVISION_7B_CHAT_HF_ID,
    QWEN2_VL_HF_ID, QWEN25_VL_HF_ID, QWEN3_VL_2B_HF_ID, QWEN3_VL_8B_HF_ID,
)
from .llava import LlavaTarget
from .native_hf import (
    LlavaNextTarget, LlavaOneVision7BChatTarget, Qwen3VL2BTarget, Qwen3VL8BTarget,
)
from .qwen2vl import Qwen2VLTarget, Qwen25VLTarget


_CHECKPOINT_TARGETS = {
    LLAVA_15_HF_ID: LlavaTarget,
    LLAVA_16_HF_ID: LlavaNextTarget,
    LLAVA_ONEVISION_7B_CHAT_HF_ID: LlavaOneVision7BChatTarget,
    QWEN2_VL_HF_ID: Qwen2VLTarget,
    QWEN25_VL_HF_ID: Qwen25VLTarget,
    QWEN3_VL_2B_HF_ID: Qwen3VL2BTarget,
    QWEN3_VL_8B_HF_ID: Qwen3VL8BTarget,
}
TARGET_REGISTRY = dict(_CHECKPOINT_TARGETS)
TARGET_REGISTRY.update({alias: _CHECKPOINT_TARGETS[model_id]
                        for alias, model_id in MODEL_ALIASES.items()
                        if model_id in _CHECKPOINT_TARGETS})
TARGET_REGISTRY.update({
    "LLaVA-OneVision-Chat-7B": LlavaOneVision7BChatTarget,
    "Qwen2-VL-7B-Instruct": Qwen2VLTarget,
    "Qwen2.5-VL-7B-Instruct": Qwen25VLTarget,
    "Qwen3-VL-2B-Instruct": Qwen3VL2BTarget,
    "Qwen3-VL-8B-Instruct": Qwen3VL8BTarget,
})


def get_target_factory(name):
    try:
        return TARGET_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown target {name!r}. Available: {sorted(TARGET_REGISTRY)}") from None


def validate_image_count(name, count):
    factory = get_target_factory(name)
    cls = factory.func if isinstance(factory, partial) else factory
    minimum, maximum = cls.min_images, cls.max_images
    if count < minimum or maximum is not None and count > maximum:
        allowed = f"{minimum}..{maximum}" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"{name} accepts {allowed} images; got {count}")


def load_target(name, max_new_tokens=256):
    return get_target_factory(name)(max_new_tokens=max_new_tokens)


__all__ = [
    "TARGET_REGISTRY", "get_target_factory", "validate_image_count", "load_target",
    "LlavaTarget", "LlavaNextTarget", "LlavaOneVision7BChatTarget",
    "Qwen2VLTarget", "Qwen25VLTarget", "Qwen3VL2BTarget", "Qwen3VL8BTarget",
]
