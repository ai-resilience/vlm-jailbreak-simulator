"""Shared contracts for composing multiple jailbreak generation stages.

Compositions are intentionally represented separately from a single
``Attack``.  A paper implementation normally owns one end-to-end input
transformation, whereas a composition extracts and connects stages from
several methods.  The distinction is important for provenance: every final
sample produced here is an ``adapted_composition`` and is never advertised as
an exact reproduction of any one paper.

The module is model-free.  It provides deterministic parent selection,
content hashing, stage lineage, and strict configuration validation.  Concrete
components live in :mod:`compositions.components` and load model-backed
attacks lazily.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from .paths import file_sha256
from .paths import REPO_ROOT, resolve_path
from .records import Behavior


COMPOSITION_SCHEMA_VERSION = 1
FINAL_ALGORITHM_FIDELITY = "adapted_composition"
FORBIDDEN_COMPONENT_ATTACKS = {"bestofn", "deltajp", "visualadv", "force"}
KNOWN_COMPONENTS = {
    "cs_dj_split",
    "cs_dj_full",
    "ultrabreak_patch",
    "figstep_render",
    "mml_layout",
    "jood_aug",
    "hades_patch",
    "delta_jp",
    "jailbreak_in_pieces_embedding",
    "force_refine",
    "jps_joint",
}
COMPONENT_ATTACKS = {
    "cs_dj_split": "CS-DJ", "cs_dj_full": "CS-DJ",
    "ultrabreak_patch": "UltraBreak", "figstep_render": "FigStep",
    "mml_layout": "MML", "jood_aug": "JOOD",
    "hades_patch": "HADES", "delta_jp": "DeltaJP",
    "jailbreak_in_pieces_embedding": "JailbreakInPieces",
    "force_refine": "FORCE", "jps_joint": "JPS",
}


def _attack_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def normalize_composed_text(text: str) -> str:
    """Canonical text used only for duplicate/content identity checks."""

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def strict_image_hashes(images: Sequence[str]) -> list[str]:
    """Return ordered image SHA-256 values and fail on a missing artifact."""

    hashes: list[str] = []
    for path in images:
        absolute = resolve_path(path)
        digest = file_sha256(absolute)
        if digest is None:
            raise FileNotFoundError(f"composition image artifact missing: {absolute}")
        hashes.append(digest)
    return hashes


def composition_content_hash(text: str, images: Sequence[str]) -> str:
    """Identity of one final sample: normalized text + ordered image bytes."""

    digest = hashlib.sha256()
    digest.update(normalize_composed_text(text).encode("utf-8"))
    digest.update(b"\0")
    for image_digest in strict_image_hashes(images):
        digest.update(image_digest.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_composed_text(text).encode("utf-8")).hexdigest()


def config_sha256(path: str | None) -> str | None:
    if not path:
        return None
    return file_sha256(path)


def stable_fraction(value: str, *, seed: int = 0) -> float:
    payload = f"{seed}\0{value}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(1 << 64)


def assign_source_split(
    source_behavior_id: str,
    *,
    seed: int = 0,
    train: float = 0.70,
    validation: float = 0.10,
) -> str:
    """Assign all derivatives of one parent to the same 70/10/20 split."""

    if train <= 0 or validation < 0 or train + validation >= 1:
        raise ValueError("split ratios must satisfy train>0, validation>=0, train+validation<1")
    value = stable_fraction(source_behavior_id, seed=seed)
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "test"


def deterministic_stratified_select(
    behaviors: Iterable[Behavior], count: int, *, seed: int = 0
) -> list[Behavior]:
    """Round-robin categories after deterministic hash ordering within groups."""

    if count < 0:
        raise ValueError("selection count must be non-negative")
    unique: dict[str, Behavior] = {}
    for behavior in behaviors:
        unique.setdefault(behavior.behavior_id, behavior)
    groups: dict[str, list[Behavior]] = {}
    for behavior in unique.values():
        groups.setdefault(behavior.category or "__uncategorized__", []).append(behavior)
    for category, values in groups.items():
        values.sort(
            key=lambda item: hashlib.sha256(
                f"{seed}\0{category}\0{item.behavior_id}".encode("utf-8")
            ).hexdigest()
        )
    selected: list[Behavior] = []
    categories = sorted(groups)
    while len(selected) < min(count, len(unique)):
        progressed = False
        for category in categories:
            if groups[category]:
                selected.append(groups[category].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


@dataclass(frozen=True)
class StageRecord:
    """Tamper-evident input/output lineage for one composition stage."""

    stage_index: int
    component: str
    source_attack: str
    algorithm_fidelity: str
    paper_reproduction: bool
    config_path: str | None
    config_sha256: str | None
    effective_config: dict[str, Any]
    input_content_hash: str
    output_content_hash: str
    input_text_sha256: str
    output_text_sha256: str
    input_images: list[str]
    output_images: list[str]
    input_image_sha256s: list[str]
    output_image_sha256s: list[str]
    duration_seconds: float
    full_iterations_completed: bool
    iteration_plan: dict[str, Any] = field(default_factory=dict)
    iteration_result: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComposedArtifact:
    """Current text/images for one parent as it moves through a recipe."""

    source_behavior_id: str
    text: str
    images: list[str] = field(default_factory=list)
    stage_records: list[StageRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return composition_content_hash(self.text, self.images)

    def with_payload(
        self,
        *,
        text: str | None = None,
        images: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ComposedArtifact":
        merged = copy.deepcopy(self.metadata)
        if metadata:
            merged.update(copy.deepcopy(dict(metadata)))
        return replace(
            self,
            text=self.text if text is None else str(text),
            images=list(self.images if images is None else images),
            metadata=merged,
        )


@dataclass(frozen=True)
class ComponentResult:
    artifact: ComposedArtifact
    full_iterations_completed: bool = True
    iteration_plan: dict[str, Any] = field(default_factory=dict)
    iteration_result: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class CompositionComponent:
    """Base API implemented by every decomposed/full composition stage."""

    component_name = ""
    source_attack = ""
    algorithm_fidelity = "adapted_decomposition"
    paper_reproduction = False

    def __init__(self, stage_spec: Mapping[str, Any], *, repo_root: str = REPO_ROOT):
        self.stage_spec = copy.deepcopy(dict(stage_spec))
        raw_path = self.stage_spec.get("config")
        self.config_path = None
        self.config: dict[str, Any] = {}
        if raw_path:
            self.config_path = (
                os.path.abspath(os.path.expanduser(str(raw_path)))
                if os.path.isabs(str(raw_path))
                else os.path.join(repo_root, str(raw_path))
            )
            if not os.path.isfile(self.config_path):
                raise FileNotFoundError(f"composition stage config missing: {self.config_path}")
            import yaml

            with open(self.config_path, encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"stage config must be a mapping: {self.config_path}")
            self.config = loaded
        self.config = deep_merge(self.config, dict(self.stage_spec.get("overrides") or {}))
        self._prepared = False
        self.prepare_seconds = 0.0
        self.prepare_out_dir: str | None = None

    @property
    def effective_config(self) -> dict[str, Any]:
        return copy.deepcopy(self.config)

    def prepare(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.prepare_out_dir = os.path.abspath(out_dir)
        self._prepared = True

    def evaluation_exclusion_reason(self, behavior: Behavior) -> str | None:
        return None

    def preflight(self) -> None:
        """Check local prerequisites without loading model weights."""

    def transform(
        self, artifact: ComposedArtifact, behavior: Behavior, out_dir: str
    ) -> ComponentResult:
        raise NotImplementedError

    def provenance(self) -> dict[str, Any]:
        return {
            "component": self.component_name,
            "source_attack": self.source_attack,
            "algorithm_fidelity": self.algorithm_fidelity,
            "paper_reproduction": False,
            "config_path": self.config_path,
            "config_sha256": config_sha256(self.config_path),
            "effective_config": self.effective_config,
            "prepare_seconds": self.prepare_seconds,
            "prepare_out_dir": self.prepare_out_dir,
        }


def execute_component(
    component: CompositionComponent,
    artifact: ComposedArtifact,
    behavior: Behavior,
    out_dir: str,
    *,
    stage_index: int,
) -> ComposedArtifact:
    """Run one stage and append a verified continuity record."""

    if artifact.source_behavior_id != behavior.behavior_id:
        raise ValueError(
            "composition artifact/behavior mismatch: "
            f"{artifact.source_behavior_id!r} != {behavior.behavior_id!r}"
        )
    input_images = strict_image_hashes(artifact.images)
    input_hash = composition_content_hash(artifact.text, artifact.images)
    started = time.perf_counter()
    result = component.transform(artifact, behavior, out_dir)
    duration = time.perf_counter() - started
    output = result.artifact
    if output.source_behavior_id != artifact.source_behavior_id:
        raise ValueError("a composition component changed source_behavior_id")
    if not output.text.strip():
        raise ValueError(f"{component.component_name} produced empty text")
    output_images = strict_image_hashes(output.images)
    output_hash = composition_content_hash(output.text, output.images)
    if not result.full_iterations_completed:
        raise RuntimeError(
            f"{component.component_name} did not complete its configured iterations"
        )
    record = StageRecord(
        stage_index=stage_index,
        component=component.component_name,
        source_attack=component.source_attack,
        algorithm_fidelity=component.algorithm_fidelity,
        paper_reproduction=False,
        config_path=component.config_path,
        config_sha256=config_sha256(component.config_path),
        effective_config=component.effective_config,
        input_content_hash=input_hash,
        output_content_hash=output_hash,
        input_text_sha256=text_sha256(artifact.text),
        output_text_sha256=text_sha256(output.text),
        input_images=list(artifact.images),
        output_images=list(output.images),
        input_image_sha256s=input_images,
        output_image_sha256s=output_images,
        duration_seconds=duration,
        full_iterations_completed=True,
        iteration_plan=copy.deepcopy(result.iteration_plan),
        iteration_result=copy.deepcopy(result.iteration_result),
        metadata=copy.deepcopy(result.metadata),
    )
    return replace(output, stage_records=[*artifact.stage_records, record])


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; lists/scalars are deliberately replaced."""

    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_composition_config(path: str) -> dict[str, Any]:
    import yaml

    absolute = os.path.abspath(os.path.expanduser(path))
    with open(absolute, encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("composition config must be a mapping")
    validate_composition_config(config)
    config["_config_path"] = absolute
    config["_config_sha256"] = file_sha256(absolute)
    return config


def validate_composition_config(config: Mapping[str, Any]) -> None:
    """Fail closed on count drift, unsupported attacks, and malformed recipes."""

    if int(config.get("schema_version", -1)) != COMPOSITION_SCHEMA_VERSION:
        raise ValueError(
            f"composition schema_version must be {COMPOSITION_SCHEMA_VERSION}"
        )
    multiplier = float(config.get("candidate_multiplier", 1.2))
    if multiplier < 1.0:
        raise ValueError("candidate_multiplier must be >= 1")
    checkpoint_every = config.get("checkpoint_every", 10)
    if not isinstance(checkpoint_every, int) or isinstance(checkpoint_every, bool) or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    if config.get("algorithm_fidelity") != FINAL_ALGORITHM_FIDELITY:
        raise ValueError(f"algorithm_fidelity must be {FINAL_ALGORITHM_FIDELITY!r}")
    if config.get("paper_reproduction") is not False:
        raise ValueError("composed datasets must set paper_reproduction: false")
    if "no_asr" not in str(config.get("selection_policy", "")):
        raise ValueError("composition selection_policy must explicitly exclude ASR")
    split = config.get("split") or {}
    train_ratio = float(split.get("train", 0.70))
    validation_ratio = float(split.get("validation", 0.10))
    test_ratio = float(split.get("test", 0.20))
    if any(value < 0 for value in (train_ratio, validation_ratio, test_ratio)):
        raise ValueError("composition split ratios cannot be negative")
    if not math.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("composition split ratios must sum to 1")
    recipes = config.get("recipes")
    if not isinstance(recipes, list) or not recipes:
        raise ValueError("composition config requires a non-empty recipes list")
    identifiers: set[str] = set()
    final_total = raw_total = 0
    for recipe in recipes:
        if not isinstance(recipe, Mapping):
            raise ValueError("each recipe must be a mapping")
        recipe_id = str(recipe.get("id", "")).strip()
        if not recipe_id or recipe_id in identifiers:
            raise ValueError(f"missing or duplicate recipe id: {recipe_id!r}")
        identifiers.add(recipe_id)
        stages = recipe.get("stages")
        if not isinstance(stages, list) or len(stages) < 2:
            raise ValueError(f"recipe {recipe_id} must contain at least two stages")
        arity = int(recipe.get("arity", len(stages)))
        if arity != len(stages) or arity not in {2, 3, 4}:
            raise ValueError(f"recipe {recipe_id} arity/stage count must be 2, 3, or 4")
        datasets = recipe.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError(f"recipe {recipe_id} requires at least one dataset")
        final_count = int(recipe.get("final_count", 0))
        raw_count = int(recipe.get("raw_count", math.ceil(final_count * multiplier)))
        if final_count <= 0 or raw_count < final_count:
            raise ValueError(f"recipe {recipe_id} has invalid final/raw counts")
        expected_raw = math.ceil(final_count * multiplier)
        if raw_count != expected_raw:
            raise ValueError(
                f"recipe {recipe_id} raw_count {raw_count} != ceil("
                f"{final_count} * {multiplier})={expected_raw}"
            )
        final_total += final_count
        raw_total += raw_count
        recipe_attacks: set[str] = set()
        for stage in stages:
            if not isinstance(stage, Mapping) or not stage.get("component"):
                raise ValueError(f"recipe {recipe_id} contains a malformed stage")
            if str(stage["component"]) not in KNOWN_COMPONENTS:
                raise ValueError(
                    f"recipe {recipe_id} uses unknown component {stage['component']!r}"
                )
            source_attack = COMPONENT_ATTACKS[str(stage["component"])]
            attack = _attack_key(stage.get("attack", source_attack))
            if attack in FORBIDDEN_COMPONENT_ATTACKS or _attack_key(source_attack) in FORBIDDEN_COMPONENT_ATTACKS:
                raise ValueError(
                    f"recipe {recipe_id} uses excluded attack {stage.get('attack')!r}"
                )
            if attack != _attack_key(source_attack):
                raise ValueError(f"recipe {recipe_id} component/attack mismatch: {stage['component']}")
            if attack in recipe_attacks:
                raise ValueError(f"recipe {recipe_id} repeats the same methodology: {source_attack}")
            recipe_attacks.add(attack)
        # Renderers and JPS construct a fresh image; placing them after an image
        # attack would discard the earlier image rather than compose it.
        image_seen = False
        for stage in stages:
            name = stage["component"]
            if image_seen and name in {"figstep_render", "mml_layout", "jps_joint", "cs_dj_full"}:
                raise ValueError(f"recipe {recipe_id} would replace a preceding image at {name}")
            image_seen |= name != "cs_dj_split"
    expected = config.get("expected_totals") or {}
    if expected:
        if int(expected.get("final", final_total)) != final_total:
            raise ValueError(
                f"recipe final total {final_total} != expected {expected.get('final')}"
            )
        if int(expected.get("raw", raw_total)) != raw_total:
            raise ValueError(
                f"recipe raw total {raw_total} != expected {expected.get('raw')}"
            )
    controls = config.get("controls") or {}
    if controls:
        component_total = sum(int(controls.get(key, 0)) for key in (
            "benign_fpr_external", "unmodified_harmful", "single_or_leave_one_out"
        ))
        if int(controls.get("total", -1)) != component_total:
            raise ValueError("control component counts do not sum to controls.total")
        if controls.get("included_in_jailbreak_set") is not False:
            raise ValueError("controls must remain outside the jailbreak dataset")
    groups = config.get("execution_groups")
    if groups is not None:
        if not isinstance(groups, list) or not groups:
            raise ValueError("execution_groups must be a non-empty list")
        grouped: list[str] = []
        names: set[str] = set()
        for group in groups:
            name = str(group.get("name", ""))
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in names:
                raise ValueError(f"invalid or duplicate execution group name: {name!r}")
            names.add(name)
            members = group.get("recipes")
            if not isinstance(members, list) or not members:
                raise ValueError(f"execution group {name} requires recipe IDs")
            grouped.extend(str(value) for value in members)
        if len(grouped) != len(set(grouped)) or set(grouped) != identifiers:
            raise ValueError("execution_groups must cover every recipe exactly once")


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy without private loader annotations, suitable for a manifest."""

    return {key: copy.deepcopy(value) for key, value in config.items() if not key.startswith("_")}


def require_matching_run_config(config_path: str, run_dir: str, recipe_ids: str = "") -> None:
    """Reject reuse of a valid but different plan (including legacy v1 groups)."""

    config = load_composition_config(config_path)
    with open(os.path.join(run_dir, "composition_manifest.json"), encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("config_sha256") != config["_config_sha256"]:
        raise ValueError(f"composition config mismatch in {run_dir}; choose a new output root")
    expected = set(recipe_ids.split(",")) if recipe_ids else {
        str(recipe["id"]) for recipe in config["recipes"]
    }
    actual = {str(recipe["recipe_id"]) for recipe in manifest.get("recipes", [])}
    if actual != expected:
        raise ValueError(f"recipe group mismatch in {run_dir}")


def write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
