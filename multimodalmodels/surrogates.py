"""Model profiles and factories for differentiable attack integrations.

These profiles describe *implemented* differentiable interfaces, not every VLM
that can merely be loaded.  The modern profiles currently support the shared
teacher-forced CE contract used by VisualAdv, DeltaJP, and HADES white-box.
Algorithm-specific adapters (JPS, UltraBreak, FORCE) intentionally
remain separate and must not silently consume these profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

from .registry import (
    INTERNVL3_2B_HF_ID,
    INTERNVL3_2B_REVISION,
    INTERNVL3_8B_HF_ID,
    INTERNVL3_8B_REVISION,
    LLAVA_15_HF_ID,
    LLAVA_15_REVISION,
    LLAVA_16_HF_ID,
    LLAVA_16_REVISION,
    QWEN2_VL_HF_ID,
    QWEN2_VL_REVISION,
    QWEN3_VL_2B_HF_ID,
    QWEN3_VL_2B_REVISION,
    QWEN3_VL_8B_HF_ID,
    QWEN3_VL_8B_REVISION,
)


CE_ATTACKS = ("VisualAdv", "DeltaJP", "HADES")


@dataclass(frozen=True)
class SurrogateSpec:
    backend: str
    model_id: str
    revision: str | None
    supported_attacks: tuple[str, ...]
    recommended_role: str


SURROGATE_SPECS: dict[str, SurrogateSpec] = {
    "llava_v1_5": SurrogateSpec(
        "llava", LLAVA_15_HF_ID, LLAVA_15_REVISION, CE_ATTACKS,
        "legacy white-box CE",
    ),
    "qwen2_vl_7b": SurrogateSpec(
        "qwen2", QWEN2_VL_HF_ID, QWEN2_VL_REVISION, CE_ATTACKS,
        "legacy transfer CE; JPS/UltraBreak use their own Qwen2 adapter",
    ),
    "llava_v1_6": SurrogateSpec(
        "llava-next", LLAVA_16_HF_ID, LLAVA_16_REVISION, CE_ATTACKS,
        "modern target/CE surrogate",
    ),
    "qwen3_vl_2b": SurrogateSpec(
        "qwen3-vl", QWEN3_VL_2B_HF_ID, QWEN3_VL_2B_REVISION, CE_ATTACKS,
        "recommended lightweight CE surrogate",
    ),
}

DEFAULT_PROFILE_BY_BACKEND = {
    "llava": "llava_v1_5",
    "qwen2": "qwen2_vl_7b",
    "qwen2-vl": "qwen2_vl_7b",
    "llava-next": "llava_v1_6",
    "qwen3": "qwen3_vl_2b",
    "qwen3-vl": "qwen3_vl_2b",
}


def default_surrogate_for_backend(backend: str) -> SurrogateSpec:
    key = DEFAULT_PROFILE_BY_BACKEND.get((backend or "llava").lower())
    if key is None:
        raise ValueError(f"Unknown surrogate backend {backend!r}")
    return SURROGATE_SPECS[key]


def list_surrogate_profiles() -> list[str]:
    return sorted(SURROGATE_SPECS)


def get_surrogate_profile(name: str) -> SurrogateSpec:
    try:
        return SURROGATE_SPECS[name]
    except KeyError:
        raise KeyError(
            f"Unknown surrogate profile {name!r}. Available: "
            f"{list_surrogate_profiles()}"
        ) from None


def load_surrogate(name: str, *, revision=None):
    """Load a profile's shared [0,1]-RGB/teacher-forced-CE interface.

    Existing attack algorithms retain their own update loops. Integrations
    consume to_unit01/compute_loss_batch/zero_grad from this model interface.
    """
    from .ce_surrogate import _LlavaCE, _NativeHFCE

    spec = get_surrogate_profile(name)
    if spec.backend == "llava":
        return _LlavaCE("LLaVA_v1_5", revision=revision or spec.revision)
    return _NativeHFCE(spec.backend, spec.model_id, revision=revision or spec.revision)


__all__ = [
    "CE_ATTACKS", "SurrogateSpec", "SURROGATE_SPECS",
    "DEFAULT_PROFILE_BY_BACKEND", "default_surrogate_for_backend",
    "get_surrogate_profile", "list_surrogate_profiles",
    "load_surrogate",
]
