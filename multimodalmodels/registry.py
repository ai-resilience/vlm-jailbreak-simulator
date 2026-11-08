"""Single source of truth for every model this repo loads.

Each entry pins a HuggingFace repo id to an exact commit (``revision``) so the
eval target, the judge, attack surrogates and defense encoders are reproducible
and auditable in one table. The pinned revisions were taken from, and are
checked against, the local HF cache snapshots (see
``core/tests/test_model_loading.py``). An entry whose ``revision`` is ``None``
is not pinned yet (it was not in the local cache) and resolves to the hub's
default branch; pin it before relying on it for a reproducible run.

This module is pure data (no torch / transformers import), so attack modules
can read pinned revisions without pulling in model code. The loading itself
lives in ``multimodalmodels/loaders.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    revision: Optional[str]   # pinned commit; None = not pinned yet
    kind: str                 # loader family in multimodalmodels/loaders.py
    used_by: str              # where the repo uses it (for auditing)


LLAVA_15_HF_ID = "llava-hf/llava-1.5-7b-hf"
LLAVA_16_HF_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
LLAVA_ONEVISION_7B_CHAT_HF_ID = "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"
QWEN2_VL_HF_ID = "Qwen/Qwen2-VL-7B-Instruct"
QWEN25_VL_HF_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN3_VL_2B_HF_ID = "Qwen/Qwen3-VL-2B-Instruct"
QWEN3_VL_8B_HF_ID = "Qwen/Qwen3-VL-8B-Instruct"
INTERNVL3_2B_HF_ID = "OpenGVLab/InternVL3-2B-hf"
INTERNVL3_8B_HF_ID = "OpenGVLab/InternVL3-8B-hf"
QWEN25_14B_HF_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN25_3B_HF_ID = "Qwen/Qwen2.5-3B-Instruct"
CLIP_VIT_L14_ST_HF_ID = "sentence-transformers/clip-ViT-L-14"
CLIP_VIT_L14_336_HF_ID = "openai/clip-vit-large-patch14-336"
CLIP_VIT_L14_HF_ID = "openai/clip-vit-large-patch14"
HARMBENCH_CLS_HF_ID = "cais/HarmBench-Llama-2-13b-cls"
BEAVER_DAM_HF_ID = "PKU-Alignment/beaver-dam-7b"
LLAMA_GUARD2_HF_ID = "meta-llama/Meta-Llama-Guard-2-8B"
LLAMA_GUARD3_HF_ID = "meta-llama/Llama-Guard-3-8B"
VIT_BASE_HF_ID = "google/vit-base-patch16-224"

LLAVA_15_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
LLAVA_16_REVISION = "2424fdd47412fccc66d91719126b420e9fbd7065"
LLAVA_ONEVISION_7B_CHAT_REVISION = "fa83f9361d2d053d9cfa12f2ceea23425762a8d3"
QWEN2_VL_REVISION = "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
QWEN25_VL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
QWEN3_VL_2B_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
QWEN3_VL_8B_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
INTERNVL3_2B_REVISION = "cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
INTERNVL3_8B_REVISION = "259a3b64a14623c0ec91a045cb43f7c5af5fa6af"
QWEN25_14B_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
QWEN25_3B_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
CLIP_VIT_L14_ST_REVISION = "1b4b2e899178706d1b9905460ac21de1e0ba86a5"
CLIP_VIT_L14_336_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
CLIP_VIT_L14_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
HARMBENCH_CLS_REVISION = "bda705349d1144fa618770bea64d99ce54e3835b"

MODEL_SPECS: dict[str, ModelSpec] = {
    spec.hf_id: spec
    for spec in (
        ModelSpec(LLAVA_15_HF_ID, LLAVA_15_REVISION, "llava",
                  "eval target; white-box surrogate (FORCE, VisualAdv, DeltaJP); "
                  "CIDER encoder; ECSO target and text-only generator"),
        ModelSpec(LLAVA_16_HF_ID, LLAVA_16_REVISION, "llava-next",
                  "eval target; optional CE surrogate (VisualAdv, DeltaJP, HADES)"),
        ModelSpec(LLAVA_ONEVISION_7B_CHAT_HF_ID,
                  LLAVA_ONEVISION_7B_CHAT_REVISION,
                  "llava-onevision",
                  "CS-DJ Table 6 approximate-reproduction eval target"),
        ModelSpec(QWEN2_VL_HF_ID, QWEN2_VL_REVISION, "qwen2-vl",
                  "transfer surrogate (UltraBreak, JPS, VisualAdv/DeltaJP qwen2)"),
        ModelSpec(QWEN25_VL_HF_ID, QWEN25_VL_REVISION, "qwen2-vl",
                  "optional Qwen2.5-VL eval target / surrogate"),
        ModelSpec(QWEN3_VL_2B_HF_ID, QWEN3_VL_2B_REVISION, "qwen3-vl",
                  "lightweight CE surrogate; optional eval target"),
        ModelSpec(QWEN3_VL_8B_HF_ID, QWEN3_VL_8B_REVISION, "qwen3-vl",
                  "eval target; optional CE surrogate"),
        ModelSpec(INTERNVL3_2B_HF_ID, INTERNVL3_2B_REVISION, "internvl",
                  "lightweight CE surrogate; optional eval target"),
        ModelSpec(INTERNVL3_8B_HF_ID, INTERNVL3_8B_REVISION, "internvl",
                  "eval target; optional CE surrogate"),
        ModelSpec(QWEN25_14B_HF_ID, QWEN25_14B_REVISION, "causal-lm-agent",
                  "JPS official multi-agent feedback/summarization/revision model"),
        ModelSpec(QWEN25_3B_HF_ID, QWEN25_3B_REVISION, "causal-lm-agent",
                  "CS-DJ official harmful-query decomposition model"),
        ModelSpec(CLIP_VIT_L14_ST_HF_ID, CLIP_VIT_L14_ST_REVISION,
                  "sentence-transformer", "CS-DJ contrastive image retrieval"),
        ModelSpec(CLIP_VIT_L14_336_HF_ID, CLIP_VIT_L14_336_REVISION,
                  "clip-image-encoder",
                  "JailbreakInPieces target-image embedding optimization; "
                  "optional benchmark CLIP image-similarity metric"),
        ModelSpec(CLIP_VIT_L14_HF_ID, CLIP_VIT_L14_REVISION, "clip-model",
                  "VLMShield detector image/text encoder (official checkpoint)"),
        ModelSpec(HARMBENCH_CLS_HF_ID, HARMBENCH_CLS_REVISION, "causal-lm-classifier",
                  "HarmBench judge"),
        ModelSpec(BEAVER_DAM_HF_ID,
                  "3a5b70888869f04336d0d8344185ed74268ef896",
                  "sequence-classifier",
                  "CS-DJ official Beaver-Dam judge; upstream did not pin a revision"),
        ModelSpec(LLAMA_GUARD2_HF_ID, None, "causal-lm-classifier",
                  "JOOD official ASR judge (not pinned upstream)"),
        ModelSpec(LLAMA_GUARD3_HF_ID, None, "causal-lm-classifier",
                  "JPS official ASR judge (not pinned upstream)"),
        ModelSpec(VIT_BASE_HF_ID, None, "image-classifier",
                  "CIDER denoiser classifier fallback (not pinned: not in local cache)"),
    )
}


# User-facing names accepted by target/surrogate CLIs.  Keep resolution here so
# provenance always records the exact Hugging Face repository rather than an
# ambiguous display name.
MODEL_ALIASES: dict[str, str] = {
    "LLaVA-OneVision-Chat-7B": LLAVA_ONEVISION_7B_CHAT_HF_ID,
    "Qwen2-VL-7B-Instruct": QWEN2_VL_HF_ID,
    "Qwen2.5-VL-7B-Instruct": QWEN25_VL_HF_ID,
    "Qwen3-VL-2B-Instruct": QWEN3_VL_2B_HF_ID,
    "Qwen3-VL-8B-Instruct": QWEN3_VL_8B_HF_ID,
    "LLaVA_v1_5": LLAVA_15_HF_ID,
    "llava_v1_5": LLAVA_15_HF_ID,
    "LLaVA_v1_6": LLAVA_16_HF_ID,
    "llava_v1_6": LLAVA_16_HF_ID,
    "llava_next_7b": LLAVA_16_HF_ID,
    "llava_onevision_7b_chat": LLAVA_ONEVISION_7B_CHAT_HF_ID,
    "qwen2_vl_7b": QWEN2_VL_HF_ID,
    "qwen2_5_vl_7b": QWEN25_VL_HF_ID,
    "qwen3_vl_2b": QWEN3_VL_2B_HF_ID,
    "qwen3_vl_8b": QWEN3_VL_8B_HF_ID,
    "internvl3_2b": INTERNVL3_2B_HF_ID,
    "internvl3_8b": INTERNVL3_8B_HF_ID,
}


def resolve_model_id(model_id_or_alias: str) -> str:
    """Resolve a stable CLI alias to its exact Hugging Face repository id."""
    return MODEL_ALIASES.get(model_id_or_alias, model_id_or_alias)


def get_spec(hf_id: str) -> Optional[ModelSpec]:
    """The registered spec for ``hf_id`` (None for local paths / unknown ids)."""
    return MODEL_SPECS.get(resolve_model_id(hf_id))


def default_revision(hf_id: str) -> Optional[str]:
    """The pinned revision for ``hf_id``, or None if unknown or not pinned."""
    spec = MODEL_SPECS.get(resolve_model_id(hf_id))
    return spec.revision if spec else None
