"""Shared loading helpers for VLM targets, judges and model integrations.

IDs and default revisions come from registry.py. Explicit revisions take
precedence. Heavy libraries are imported only when a loader is called.
Attack-specific losses and optimization stay outside this module.
"""

from __future__ import annotations

from typing import Any, Optional

from .registry import (BEAVER_DAM_HF_ID, CLIP_VIT_L14_336_HF_ID, CLIP_VIT_L14_HF_ID,
                       CLIP_VIT_L14_ST_HF_ID,
                       HARMBENCH_CLS_HF_ID, INTERNVL3_8B_HF_ID,
                       LLAVA_15_HF_ID, LLAVA_16_HF_ID,
                       LLAVA_ONEVISION_7B_CHAT_HF_ID,
                       QWEN25_VL_HF_ID, QWEN2_VL_HF_ID, QWEN3_VL_8B_HF_ID,
                       default_revision, get_spec, resolve_model_id)


def _revision(model_id: str, revision: Optional[str]) -> Optional[str]:
    """An explicit revision wins; otherwise use the registry pin (if any)."""
    return revision if revision is not None else default_revision(model_id)


def _dtype(torch_dtype: Any):
    """Accept "float16" / "bfloat16" / "float32" / "auto" or a torch.dtype."""
    if not isinstance(torch_dtype, str) or torch_dtype == "auto":
        return torch_dtype
    import torch
    return getattr(torch, torch_dtype)


def load_llava(
    model_id: str = LLAVA_15_HF_ID,
    revision: Optional[str] = None,
    torch_dtype: Any = "float16",
    device_map: Optional[str] = "auto",
    to_device=None,
    eval_mode: bool = False,
    processor_kwargs: Optional[dict] = None,
):
    """LLaVA-1.5 model + processor.

    ``device_map="auto"`` shards across visible GPUs (inference: the eval target
    and the CE surrogate). Pass ``device_map=None`` with ``to_device`` for a
    single-device white-box attack that installs forward hooks (FORCE).
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    revision = _revision(model_id, revision)
    kwargs = {"revision": revision, "torch_dtype": _dtype(torch_dtype)}
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if to_device is not None:
        model = model.to(to_device)
    if eval_mode:
        model = model.eval()
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, **(processor_kwargs or {}))
    return model, processor


def load_qwen2vl(
    model_id: str = QWEN2_VL_HF_ID,
    revision: Optional[str] = None,
    torch_dtype: Any = "auto",
    device_map: Optional[str] = "auto",
    processor_kwargs: Optional[dict] = None,
):
    """Qwen2-VL (or Qwen2.5-VL) model + processor."""
    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              Qwen2VLForConditionalGeneration)

    revision = _revision(model_id, revision)
    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        **(processor_kwargs or {}),
    )
    cls = (Qwen2_5_VLForConditionalGeneration if model_id == QWEN25_VL_HF_ID
           else Qwen2VLForConditionalGeneration)
    model = cls.from_pretrained(model_id, revision=revision,
                                torch_dtype=_dtype(torch_dtype), device_map=device_map)
    return model, processor


def load_native_vlm(
    model_id: str,
    revision: Optional[str] = None,
    torch_dtype: Any = "auto",
    device_map: Optional[str] = "auto",
    to_device=None,
    eval_mode: bool = True,
    processor_kwargs: Optional[dict] = None,
    model_kwargs: Optional[dict] = None,
):
    """Load a native Transformers image-text model and its processor.

    This loader covers the modern target/surrogate families added after the
    original benchmark: LLaVA-NeXT (LLaVA-1.6), Qwen3-VL, and the native-HF
    InternVL3 checkpoints.  They deliberately share a single processor chat
    contract, while model-class selection remains explicit so an unsupported
    Transformers version fails loudly instead of silently loading a different
    architecture.

    Qwen3-VL requires Transformers >= 4.57.0.  The native ``-hf`` InternVL
    checkpoints are used so no unpinned ``trust_remote_code`` is executed.
    ``device_map=None`` plus ``to_device`` is intended for a one-device
    differentiable surrogate; target inference normally keeps ``auto``.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = resolve_model_id(model_id)
    spec = get_spec(model_id)
    kind = spec.kind if spec is not None else None
    if kind not in {"llava-next", "llava-onevision", "qwen3-vl", "internvl"}:
        raise ValueError(
            f"{model_id!r} is not a registered native VLM; expected a "
            "LLaVA-NeXT/OneVision, Qwen3-VL, or native-HF InternVL checkpoint"
        )

    if kind == "llava-next":
        from transformers import LlavaNextForConditionalGeneration
        model_cls = LlavaNextForConditionalGeneration
    elif kind == "llava-onevision":
        from transformers import LlavaOnevisionForConditionalGeneration
        model_cls = LlavaOnevisionForConditionalGeneration
    elif kind == "qwen3-vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL requires transformers>=4.57.0. Install the pinned "
                "project requirements before loading this model."
            ) from exc
        model_cls = Qwen3VLForConditionalGeneration
    else:
        model_cls = AutoModelForImageTextToText

    revision = _revision(model_id, revision)
    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "torch_dtype": _dtype(torch_dtype),
        **(model_kwargs or {}),
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    model = model_cls.from_pretrained(model_id, **load_kwargs)
    if to_device is not None:
        model = model.to(to_device)
    if eval_mode:
        model = model.eval()

    proc_kwargs = {"use_fast": True, **(processor_kwargs or {})}
    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        **proc_kwargs,
    )
    return model, processor


def load_llava_next(**kwargs):
    """Convenience loader for the pinned Apache-2.0 LLaVA-1.6 checkpoint."""
    return load_native_vlm(LLAVA_16_HF_ID, **kwargs)


def load_llava_onevision(**kwargs):
    """Load the pinned HF conversion of LLaVA-OneVision Chat 7B."""
    return load_native_vlm(LLAVA_ONEVISION_7B_CHAT_HF_ID, **kwargs)


def load_qwen3vl(model_id: str = QWEN3_VL_8B_HF_ID, **kwargs):
    """Convenience loader for a registered Qwen3-VL checkpoint."""
    return load_native_vlm(model_id, **kwargs)


def load_internvl(model_id: str = INTERNVL3_8B_HF_ID, **kwargs):
    """Convenience loader for a native-HF InternVL3 checkpoint."""
    return load_native_vlm(model_id, **kwargs)


def load_causal_lm_agent(
    model_id: str,
    revision: Optional[str] = None,
    torch_dtype: Any = "float16",
    device_map: Optional[str] = "auto",
):
    """Local/HuggingFace causal LM used by JPS's three text agents."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = _revision(model_id, revision)
    kwargs = {
        "revision": revision,
        "torch_dtype": _dtype(torch_dtype),
        "low_cpu_mem_usage": True,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, padding_side="left"
    )
    return model, tokenizer


def load_sentence_transformer(
    model_id: str = CLIP_VIT_L14_ST_HF_ID,
    revision: Optional[str] = None,
    device: Optional[str] = None,
):
    """SentenceTransformer encoder used for CS-DJ image/text retrieval."""
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"revision": _revision(model_id, revision)}
    if device is not None:
        kwargs["device"] = device
    return SentenceTransformer(model_id, **kwargs)


def load_clip_image_encoder(
    model_id: str = CLIP_VIT_L14_336_HF_ID,
    revision: Optional[str] = None,
    device=None,
    do_normalize: bool = False,
    local_files_only: Optional[bool] = None,
):
    """CLIP model + image processor used by Jailbreak In Pieces.

    The released attack explicitly disables processor normalization before
    optimizing pixels in ``[0, 1]``. Keeping that switch here makes model
    construction auditable without moving attack-specific loss code into the
    loading layer.
    """
    from transformers import CLIPImageProcessor, CLIPModel

    revision = _revision(model_id, revision)
    load_kwargs: dict[str, Any] = {"revision": revision}
    if local_files_only is not None:
        load_kwargs["local_files_only"] = local_files_only
    model = CLIPModel.from_pretrained(model_id, **load_kwargs)
    if device is not None:
        model = model.to(device)
    model = model.eval()
    processor = CLIPImageProcessor.from_pretrained(model_id, **load_kwargs)
    processor.do_normalize = do_normalize
    return model, processor


def load_clip_model(
    model_id: str = CLIP_VIT_L14_HF_ID,
    revision: Optional[str] = None,
    device=None,
    local_files_only: Optional[bool] = None,
):
    """CLIP model + full processor (image processor and tokenizer) used by the
    VLMShield detector, loaded in fp32 exactly like the upstream script."""
    from transformers import CLIPModel, CLIPProcessor

    revision = _revision(model_id, revision)
    load_kwargs: dict[str, Any] = {"revision": revision}
    if local_files_only is not None:
        load_kwargs["local_files_only"] = local_files_only
    model = CLIPModel.from_pretrained(model_id, **load_kwargs)
    if device is not None:
        model = model.to(device)
    model = model.eval()
    processor = CLIPProcessor.from_pretrained(model_id, **load_kwargs)
    return model, processor


def load_openai_clip(checkpoint_path, device='cpu'):
    """Load AdaShield's original OpenAI CLIP checkpoint without downloading."""
    from pathlib import Path
    import clip
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f'OpenAI CLIP checkpoint not found: {checkpoint_path}')
    model, preprocess = clip.load(str(checkpoint_path), device=device, jit=False)
    return model.eval(), preprocess


def load_clip_tokenizer(
    model_id: str = CLIP_VIT_L14_336_HF_ID,
    revision: Optional[str] = None,
    local_files_only: bool = True,
):
    """Pinned CLIP text input for the paper's textual embedding target."""
    from transformers import CLIPTokenizerFast

    return CLIPTokenizerFast.from_pretrained(
        model_id, revision=_revision(model_id, revision), local_files_only=local_files_only,
    )


def load_causal_lm_classifier(
    model_id: str = HARMBENCH_CLS_HF_ID,
    revision: Optional[str] = None,
    device_map: Optional[str] = "auto",
    load_in_8bit: bool = True,
    tokenizer_kwargs: Optional[dict] = None,
    torch_dtype: Any = None,
):
    """A causal-LM classifier (the HarmBench judge) + its tokenizer.

    8-bit loading uses ``BitsAndBytesConfig``, which is what transformers
    builds internally from the older ``load_in_8bit=True`` kwarg.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = _revision(model_id, revision)
    kwargs: dict[str, Any] = {"revision": revision}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = _dtype(torch_dtype)
    if device_map is not None:
        kwargs["device_map"] = device_map
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, **(tokenizer_kwargs or {}))
    return model, tokenizer


def load_sequence_classifier(
    model_id: str = BEAVER_DAM_HF_ID,
    revision: Optional[str] = None,
    device_map: Optional[str] = "auto",
    model_max_length: int = 1000,
):
    """Sequence classifier + tokenizer used by CS-DJ's Beaver-Dam judge."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    revision = _revision(model_id, revision)
    kwargs: dict[str, Any] = {"revision": revision}
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=(model.config.model_type != "llama"),
    )
    special_tokens = {}
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = "<pad>"
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = "</s>"
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = "<s>"
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = "<unk>"
    added = tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if added:
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is not None:
            weights = input_embeddings.weight.data
            weights[-added:] = weights[:-added].mean(dim=0, keepdim=True)
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            weights = output_embeddings.weight.data
            weights[-added:] = weights[:-added].mean(dim=0, keepdim=True)
    return model, tokenizer


def load_pretraining_encoder(
    model_path: str,
    revision: Optional[str] = None,
    torch_dtype: Any = "float16",
    low_cpu_mem_usage: bool = True,
    to_device=None,
):
    """A vision-language model used as an embedding encoder (CIDER) + its
    tokenizer and image processor."""
    from transformers import AutoImageProcessor, AutoModelForPreTraining, AutoTokenizer

    revision = _revision(model_path, revision)
    model = AutoModelForPreTraining.from_pretrained(
        model_path, revision=revision, torch_dtype=_dtype(torch_dtype),
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    if to_device is not None:
        model = model.to(to_device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    image_processor = AutoImageProcessor.from_pretrained(model_path, revision=revision)
    return model, tokenizer, image_processor


def load_image_classifier(model_path: str, revision: Optional[str] = None):
    """A ViT image classifier (CIDER's denoiser classifier)."""
    from transformers import ViTForImageClassification

    return ViTForImageClassification.from_pretrained(model_path, revision=_revision(model_path, revision))


def load_torch_checkpoint(path_or_file, map_location=None, **kwargs):
    """Weights / state dict from a ``.pt`` checkpoint (path or file-like), e.g.
    CIDER's diffusion denoiser.

    Extra kwargs go straight to ``torch.load`` so each caller keeps its own
    semantics: CIDER passes ``weights_only=False`` (upstream behaviour -- only
    load trusted checkpoints), while callers that pass nothing get torch's
    default (``weights_only=True`` since torch 2.6).
    """
    import torch

    return torch.load(path_or_file, map_location=map_location, **kwargs)


def load_spacy(name: str):
    """A spaCy pipeline (JailGuard's similarity metric). Its version is fixed by
    the installed spaCy model package rather than an HF revision."""
    import spacy

    return spacy.load(name)
