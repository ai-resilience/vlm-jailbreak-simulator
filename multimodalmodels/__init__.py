"""Resolve model wrappers only when the selected model is requested."""

import importlib

from . import registry as _registry
from .registry import (
    MODEL_ALIASES, MODEL_SPECS, ModelSpec, default_revision, get_spec, resolve_model_id,
)

for _name, _value in vars(_registry).items():
    if _name.endswith(("_HF_ID", "_REVISION")):
        globals()[_name] = _value


_WRAPPERS = {
    "LLaVA_v1_5": (".llava", "LLaVA_v1_5"),
    "LLaVA_v1_6": (".llava", "LLaVA_v1_6"),
    "InstructBLIP": (".instructblip", "InstructBLIP"),
    "Qwen_VL_Chat": (".qwen", "Qwen_VL_Chat"),
    "MiniGPT4": (".minigpt4.minigpt4_model", "MiniGPT4"),
    "MultiModalModel": (".multimodalmodel", "MultiModalModel"),
    "HarmBenchMultiModalModel": (".multimodalmodel", "HarmBenchMultiModalModel"),
}
for _name in (
    "load_llava", "load_qwen2vl", "load_native_vlm", "load_llava_next",
    "load_llava_onevision", "load_qwen3vl", "load_internvl", "load_causal_lm_agent",
    "load_sentence_transformer", "load_clip_image_encoder", "load_clip_model",
    "load_clip_tokenizer", "load_causal_lm_classifier", "load_sequence_classifier",
    "load_pretraining_encoder", "load_image_classifier", "load_torch_checkpoint", "load_spacy",
):
    _WRAPPERS[_name] = (".loaders", _name)
__all__ = list(_WRAPPERS) + [
    "MODEL_ALIASES", "MODEL_SPECS", "ModelSpec", "default_revision", "get_spec", "resolve_model_id",
] + [name for name in globals() if name.endswith(("_HF_ID", "_REVISION"))]


def __getattr__(name):
    if name not in _WRAPPERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _WRAPPERS[name]
    value = getattr(importlib.import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_WRAPPERS))

