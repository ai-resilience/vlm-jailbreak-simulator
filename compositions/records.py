"""Shared data classes for the unified jailbreak pipeline.

These three dataclasses are the contract between datasets, attacks, target
models, and judges. They are deliberately plain (JSON-serializable) so that
every stage can read and write them to disk without importing torch or any
model code.

Design notes
------------
* A ``Behavior`` is the *scored unit*. ASR is averaged per behavior, so the
  original harmful ``instruction`` is always preserved verbatim here and is what
  the judge conditions on -- never the attack-mutated prompt. This avoids the
  class of bug where a masked / rephrased prompt is used for scoring.

* Optional fields (``keyword``, ``statement``, ``images``, ``target``) cover the
  union of what the registered attacks need. A dataset fills in whatever it natively
  provides; a field that an attack needs but the dataset lacks is produced in a
  separate, cached enrichment step (not here).

* A ``TestCase`` holds a *list* of image paths and an ``attempt`` index, so it
  can represent single-image attacks (FigStep), multi-image attacks (DMN,
  CS-DJ), text-only baselines, and multi-attempt attacks (JOOD, MML Pass@N).
  ``parent_id`` lets a plug-in attack (ASO) reference the test case it enhances.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class EvalType(str, Enum):
    """How a behavior should be judged.

    Not every benchmark category is a "did the model help with a harmful act"
    question. Advice/opinion categories (legal, financial, medical, political,
    religious) must be scored differently or excluded from the main ASR, so the
    judge and the aggregator both branch on this.
    """

    HARMFUL = "harmful"   # standard harmful-instruction following
    ADVICE = "advice"     # professional advice categories (legal/financial/medical)
    OPINION = "opinion"   # subjective/opinion categories (political/religious)
    COPYRIGHT = "copyright"  # verbatim-reproduction behaviors (hash-checked)


@dataclass
class Behavior:
    """One harmful request from a benchmark; the unit over which ASR is averaged."""

    behavior_id: str                 # globally unique, stable across runs
    dataset: str                     # source dataset name, e.g. "HADES"
    instruction: str                 # ORIGINAL harmful request; judged against this
    eval_type: EvalType = EvalType.HARMFUL
    category: Optional[str] = None   # normalized category (cross-dataset)
    category_orig: Optional[str] = None  # the dataset's own category label

    # Optional enrichment used by particular attacks (None when not available):
    keyword: Optional[str] = None        # harmful keyword to mask/typeset (HADES, CS-DJ, JOOD)
    keyword_type: Optional[str] = None   # e.g. HADES "behavior"/"object"/"concept"
    statement: Optional[str] = None      # declarative rephrase, "Steps to ..." (FigStep)
    target: Optional[str] = None         # desired-prefix string, "Sure, here is ..." (FORCE, optimization attacks)

    # Images supplied by the external dataset. Paths are repo-relative when the
    # data root is local and absolute when it is mounted outside the repository.
    images: list[str] = field(default_factory=list)

    # Anything dataset-specific an attack might want but the schema doesn't name.
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["eval_type"] = self.eval_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Behavior":
        d = dict(d)
        if isinstance(d.get("eval_type"), str):
            d["eval_type"] = EvalType(d["eval_type"])
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = {k: d.pop(k) for k in list(d) if k not in known}
        obj = cls(**d)
        obj.meta.update(extra)
        return obj


@dataclass
class TestCase:
    """A concrete model input produced by an attack from one Behavior."""

    test_case_id: str                # unique: typically f"{attack}:{behavior_id}:{attempt}"
    behavior_id: str                 # which Behavior this scores toward
    attack: str                      # attack name that produced it
    text: str                        # the text prompt sent to the target model
    images: list[str] = field(default_factory=list)  # repo-relative or absolute paths (0..N)
    attempt: int = 0                 # 0-based attempt index (Pass@k attacks)
    parent_id: Optional[str] = None  # test case this one enhances (ASO plug-in)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TestCase":
        d = dict(d)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = {k: d.pop(k) for k in list(d) if k not in known}
        obj = cls(**d)
        obj.meta.update(extra)
        return obj

