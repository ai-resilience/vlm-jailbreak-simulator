"""Small record bridge for reference MML logic; no unified-core dependency."""
from dataclasses import dataclass, field


@dataclass
class Behavior:
    behavior_id: str
    dataset: str
    instruction: str
    statement: str | None = None
    category: str | None = None
    category_orig: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class TestCase:
    test_case_id: str
    behavior_id: str
    attack: str
    text: str
    images: list
    attempt: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class Completion:
    test_case_id: str
    behavior_id: str
    attack: str
    model: str
    response: str
    attempt: int = 0
    label: int | None = None
    judge: str | None = None
    meta: dict = field(default_factory=dict)


class Attack:
    name = 'MML'

    def __init__(self, config=None):
        self.config = dict(config or {})

    @staticmethod
    def test_case_id(name, behavior_id, attempt):
        return f'{name}:{behavior_id}:{attempt}'

    def missing_requirements(self, behavior):
        return []

    def provenance(self):
        return {'official_repo': self.official_repo, 'official_commit': self.official_commit,
                'algorithm_fidelity': self.algorithm_fidelity, 'paper_reproduction': False}
