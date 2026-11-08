"""Internal, model-free records used by the reference metric implementations."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Behavior:
    behavior_id: str
    dataset: str
    instruction: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Completion:
    test_case_id: str
    behavior_id: str
    attack: str
    model: str
    response: str
    attempt: int = 0
    blocked: bool = False
    label: int | None = None
    judge: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
