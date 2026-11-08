"""Five-method, 18-recipe adapted compositions with measured typography.

CS-DJ keeps real Qwen decomposition and CLIP contrast retrieval. The task-card
renderer is replaced explicitly. JPS is transferred as an intact auxiliary
image/steering pair, rather than replacing the preceding task image. JOOD mixes
the composed image with its provided harmless images using MixUp. The source
canvas stays at native resolution, an explicit adaptation of JOOD's 320px
preprocessing. These are compositions, not paper reproductions or claims of
successful jailbreaks.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

from data_paths import REPO_ROOT, resolve_path
from benchmark_datasets import portable_path
from .typography import render_text, validate_layout
from .records import EvalType, TestCase

METHODS = {"FigStep", "MML", "JOOD", "CS-DJ", "JPS"}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def instruction_key(text):
    return " ".join(text.casefold().split())


@dataclass(frozen=True)
class Recipe:
    id: str
    methods: tuple[str, ...]
    count: int


SUPPORTED_SOURCES = ("SafeBench", "HADES", "MM-SafetyBench")


def source_names(config):
    names = list(config.get("sources", {}).get("datasets", ["SafeBench"]))
    sampling = config.get("sources", {}).get("sampling", "minimum_shared_pool")
    balance = config.get("sources", {}).get("balance", "equal")
    if balance != ("proportional" if sampling == "all_records_once" else "equal"):
        raise ValueError("sources.balance must match the sampling policy")
    if sampling not in {
            "minimum_shared_pool", "full_pool_cycles", "all_records_once"}:
        raise ValueError("unsupported sources.sampling")
    if (not names or len(set(names)) != len(names)
            or not set(names) <= set(SUPPORTED_SOURCES)):
        raise ValueError("sources.datasets must contain distinct supported benchmarks")
    return names


def mixup_weights(config):
    settings = config.get("jood", {})
    values = settings.get("source_weights", [settings.get("source_weight", 0.7)])
    if not isinstance(values, list) or not values:
        raise ValueError("JOOD source_weights must be a nonempty list")
    weights = [float(value) for value in values]
    if (len(set(weights)) != len(weights)
            or any(not math.isfinite(w) or not 0 < w < 1 for w in weights)):
        raise ValueError("JOOD source weights must be distinct finite values in (0, 1)")
    return weights


def recipe_source_quotas(recipes, names):
    totals = {name: 0 for name in names}
    by_group = {False: dict(totals), True: dict(totals)}
    quotas = {}
    for recipe in recipes:
        group = "JOOD" in recipe.methods
        counts = {name: recipe.count // len(names) for name in names}
        for _ in range(recipe.count % len(names)):
            chosen = min(names, key=lambda name: (
                totals[name] + counts[name], by_group[group][name] + counts[name],
                names.index(name)))
            counts[chosen] += 1
        quotas[recipe.id] = counts
        for name in names:
            totals[name] += counts[name]
            by_group[group][name] += counts[name]
    return quotas


def _full_pool_plans(config, recipes, owned, stats, quotas):
    """One sampled question gets one recipe; exhaust each source before reuse."""
    import random
    from collections import Counter, defaultdict, deque
    names = source_names(config)
    queues = {name: deque() for name in names}
    cycles, usage, selected, plans = Counter(), Counter(), {}, {}
    all_ids = [b.behavior_id for name in names for b in owned[name]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("source behavior IDs must be distinct across benchmarks")

    def new_cycle(name):
        groups = defaultdict(list)
        cycle = cycles[name]
        cycles[name] += 1
        for behavior in owned[name]:
            groups[behavior.category or "__uncategorized__"].append(behavior)
        categories = sorted(groups)
        randomizer = random.Random(json_hash([config["seed"], name, cycle]))
        randomizer.shuffle(categories)
        for category in categories:
            groups[category].sort(key=lambda b: b.behavior_id)
            randomizer.shuffle(groups[category])
        ordered = []
        while any(groups.values()):
            for category in categories:
                if groups[category]:
                    ordered.append(groups[category].pop())
        return deque(ordered)

    for name in names:
        maximum = max(counts[name] for counts in quotas.values())
        if len(owned[name]) < maximum:
            raise ValueError(f"not enough unique {name} questions for one recipe: {len(owned[name])}/{maximum}")
    for recipe in recipes:
        counts, plan, used = quotas[recipe.id], [], set()
        for index in range(max(counts.values())):
            for name in names:
                if index >= counts[name]:
                    continue
                if not queues[name]:
                    queues[name] = new_cycle(name)
                for _ in range(len(queues[name])):
                    behavior = queues[name].popleft()
                    if behavior.behavior_id not in used:
                        break
                    queues[name].append(behavior)
                else:
                    raise ValueError("unable to avoid a repeated question/recipe pair")
                used.add(behavior.behavior_id)
                usage[behavior.behavior_id] += 1
                selected[behavior.behavior_id] = behavior
                plan.append(behavior)
        plans[recipe.id] = plan
    parents = list(selected.values())
    totals = {name: sum(quotas[r.id][name] for r in recipes) for name in names}
    for name in names:
        values = [usage[b.behavior_id] for b in owned[name]]
        if max(values) - min(values) > 1:
            raise ValueError("full-pool question draws are not balanced")
        stats[name].update({
            "selected_parents": sum(value > 0 for value in values),
            "selected_categories": dict(Counter(b.category or "__uncategorized__"
                                                 for b in parents if b.dataset == name)),
            "sampled_inputs": totals[name], "sampling_cycles": cycles[name],
            "draws_per_parent": {str(k): v for k, v in sorted(Counter(values).items())},
        })
    return parents, plans, {
        "datasets": names, "selection_policy": "full_pool_cycles_one_recipe_per_draw",
        "assignment_policy": "one_recipe_per_sampled_question",
        "resampling": "only_after_source_pool_exhaustion",
        "deduplication": "normalized_instruction_across_all_sources",
        "jps_training_excluded": True, "pools": stats,
        "recipe_source_counts": quotas, "full_source_counts": totals,
        "unique_parents": len(parents), "total_draws": sum(totals.values()),
    }


def select_source_parents(config, recipes, pools, training_keys):
    from collections import Counter, defaultdict
    names = source_names(config)
    if config.get("sources", {}).get("sampling") == "all_records_once":
        return _all_record_plans(config, recipes, pools, training_keys)
    seed = config["seed"]
    quotas = recipe_source_quotas(recipes, names)
    needs = {name: max(counts[name] for counts in quotas.values()) for name in names}
    grouped = defaultdict(list)
    stats = {}
    for name in names:
        raw = pools[name]
        harmful = [b for b in raw if b.eval_type == EvalType.HARMFUL]
        eligible = [b for b in harmful if instruction_key(b.instruction) not in training_keys]
        stats[name] = {
            "loaded": len(raw), "harmful": len(harmful),
            "training_excluded": len(harmful) - len(eligible),
            "pool_sha256": json_hash([b.to_dict() for b in raw]),
        }
        for behavior in eligible:
            key = instruction_key(behavior.instruction)
            if key:
                grouped[key].append(behavior)
    owned = {name: [] for name in names}
    for key, choices in grouped.items():
        owner = min(choices, key=lambda b: json_hash(
            [seed, "source_owner", key, b.dataset, b.behavior_id]))
        owned[owner.dataset].append(owner)
    for name in names:
        stats[name]["unique_after_cross_dataset_deduplication"] = len(owned[name])
    if config.get("sources", {}).get("sampling") == "full_pool_cycles":
        return _full_pool_plans(config, recipes, owned, stats, quotas)
    selected = {}
    for name in names:
        pool = owned[name]
        stats[name]["unique_after_cross_dataset_deduplication"] = len(pool)
        if len(pool) < needs[name]:
            raise ValueError(f"insufficient held-out unique parents for {name}: {len(pool)}/{needs[name]}")
        ordered = sorted(pool, key=lambda b: (len(b.statement or b.instruction), b.behavior_id))
        edges = [ordered[-1], ordered[len(ordered) // 2], ordered[0]]
        chosen, seen = [], set()
        for behavior in edges:
            if behavior.behavior_id not in seen and len(chosen) < needs[name]:
                chosen.append(behavior)
                seen.add(behavior.behavior_id)
        categories = defaultdict(list)
        for behavior in pool:
            if behavior.behavior_id not in seen:
                categories[behavior.category or "__uncategorized__"].append(behavior)
        for category in categories:
            categories[category].sort(key=lambda b: json_hash(
                [seed, name, category, b.behavior_id]))
        counts = Counter(b.category or "__uncategorized__" for b in chosen)
        while len(chosen) < needs[name]:
            available = [category for category, values in categories.items() if values]
            category = min(available, key=lambda c: (counts[c], json_hash([seed, name, c])))
            chosen.append(categories[category].pop(0))
            counts[category] += 1
        selected[name] = chosen
        stats[name]["selected_parents"] = len(chosen)
        stats[name]["selected_categories"] = dict(counts)
    parents = []
    for index in range(max(needs.values())):
        for name in names:
            if index < len(selected[name]):
                parents.append(selected[name][index])
    if len({b.behavior_id for b in parents}) != len(parents):
        raise ValueError("source behavior IDs must be distinct across benchmarks")
    if len({instruction_key(b.instruction) for b in parents}) != len(parents):
        raise ValueError("duplicate normalized source instruction")
    plans = {}
    for recipe in recipes:
        counts = quotas[recipe.id]
        plan = []
        for index in range(max(counts.values())):
            for name in names:
                if index < counts[name]:
                    plan.append(selected[name][index])
        plans[recipe.id] = plan
    metadata = {
        "datasets": names, "selection_policy": "equal_dataset_quota_category_balanced",
        "deduplication": "normalized_instruction_across_all_sources",
        "jps_training_excluded": True, "pools": stats,
        "recipe_source_counts": quotas,
        "full_source_counts": {
            name: sum(quotas[r.id][name] for r in recipes) for name in names},
        "unique_parents": len(parents),
    }
    return parents, plans, metadata


def _all_record_plans(config, recipes, pools, training_keys):
    """Cover every original record before drawing proportional extra inputs."""
    from collections import Counter, defaultdict, deque
    import random

    names = source_names(config)
    total = sum(recipe.count for recipe in recipes)
    raw = [behavior for name in names for behavior in pools[name]]
    if not raw or any(not pools[name] for name in names):
        raise ValueError("all-record sampling requires nonempty source pools")
    if len(raw) > total:
        raise ValueError(f"{len(raw)} original records exceed the {total}-input budget")
    if len({b.behavior_id for b in raw}) != len(raw):
        raise ValueError("source record IDs must be distinct across benchmarks")
    if any(not instruction_key(b.instruction) for b in raw):
        raise ValueError("all-record sampling cannot drop an empty original instruction")
    overlap = [b.behavior_id for b in raw if instruction_key(b.instruction) in training_keys]
    if overlap:
        raise ValueError("JPS training overlaps source records; provide disjoint training assets")

    def ordered_pool(name, cycle):
        groups = defaultdict(list)
        for behavior in pools[name]:
            groups[behavior.category or "__uncategorized__"].append(behavior)
        rng = random.Random(json_hash([config["seed"], "all_records", name, cycle]))
        categories = sorted(groups)
        rng.shuffle(categories)
        for group in groups.values():
            group.sort(key=lambda b: b.behavior_id)
            rng.shuffle(group)
        ordered = []
        while any(groups.values()):
            for category in categories:
                if groups[category]:
                    ordered.append(groups[category].pop())
        return deque(ordered)

    # Weighted fair scheduling spreads every complete source pool over recipes.
    queues = {name: ordered_pool(name, 0) for name in names}
    served, sequence = Counter(), []
    for index in range(len(raw)):
        available = [name for name in names if queues[name]]
        name = max(available, key=lambda n: (
            (index + 1) * len(pools[n]) - served[n] * len(raw), -names.index(n)))
        sequence.append(queues[name].popleft())
        served[name] += 1
    plans, offset = {}, 0
    for recipe in recipes:
        plans[recipe.id] = sequence[offset:offset + recipe.count]
        offset += recipe.count

    # Preserve duplicate source rows, assigning matching text different recipes.
    normalized = {rid: Counter(instruction_key(b.instruction) for b in rows)
                  for rid, rows in plans.items()}
    for recipe in recipes:
        rid = recipe.id
        for index, behavior in enumerate(plans[rid]):
            key = instruction_key(behavior.instruction)
            if normalized[rid][key] <= 1:
                continue
            swapped = False
            for other in recipes:
                oid = other.id
                if oid == rid or normalized[oid][key]:
                    continue
                for candidate_index, candidate in enumerate(plans[oid]):
                    candidate_key = instruction_key(candidate.instruction)
                    if candidate.dataset != behavior.dataset or normalized[rid][candidate_key]:
                        continue
                    plans[rid][index], plans[oid][candidate_index] = candidate, behavior
                    normalized[rid][key] -= 1
                    normalized[rid][candidate_key] += 1
                    normalized[oid][candidate_key] -= 1
                    normalized[oid][key] += 1
                    swapped = True
                    break
                if swapped:
                    break
            if not swapped:
                raise ValueError("cannot assign duplicate original text to distinct recipes")

    extra = total - len(raw)
    extra_counts = {name: extra * len(pools[name]) // len(raw) for name in names}
    remainder = extra - sum(extra_counts.values())
    ranking = sorted(names, key=lambda n: (-(extra * len(pools[n]) % len(raw)), names.index(n)))
    for name in ranking[:remainder]:
        extra_counts[name] += 1
    remaining = dict(extra_counts)
    extras = {name: ordered_pool(name, 1) for name in names}
    usage = Counter(b.behavior_id for b in raw)
    for recipe in recipes:
        rid = recipe.id
        while len(plans[rid]) < recipe.count:
            candidates = [name for name in names if remaining[name]]
            name = max(candidates, key=lambda n: (remaining[n], -names.index(n)))
            for _ in range(len(extras[name])):
                behavior = extras[name].popleft()
                key = instruction_key(behavior.instruction)
                if not normalized[rid][key] and usage[behavior.behavior_id] == 1:
                    break
                extras[name].append(behavior)
            else:
                raise ValueError("cannot resample without a repeated question/recipe pair")
            plans[rid].append(behavior)
            normalized[rid][key] += 1
            usage[behavior.behavior_id] += 1
            remaining[name] -= 1
    if any(remaining.values()):
        raise ValueError("unassigned proportional extra samples")
    flattened = [b for recipe in recipes for b in plans[recipe.id]]
    if len({b.behavior_id for b in flattened[:len(raw)]}) != len(raw):
        raise ValueError("resampling started before every original record was covered")
    if max(usage.values()) > 2:
        raise ValueError("all-record profile permits at most two uses per record")
    stats = {}
    for name in names:
        rows = pools[name]
        stats[name] = {
            "loaded": len(rows), "records_in_pool": len(rows),
            "harmful": sum(b.eval_type == EvalType.HARMFUL for b in rows),
            "eval_type_counts": dict(Counter(b.eval_type.value for b in rows)),
            "training_excluded": 0, "filtered_records": 0,
            "pool_sha256": json_hash([b.to_dict() for b in rows]),
            "selected_parents": len(rows),
            "selected_categories": dict(Counter(b.category for b in rows)),
            "sampled_inputs": len(rows) + extra_counts[name],
            "draws_per_parent": {str(k): v for k, v in sorted(Counter(usage[b.behavior_id] for b in rows).items())},
        }
    metadata = {
        "datasets": names, "selection_policy": "all_records_once_then_resample",
        "assignment_policy": "one_recipe_per_sampled_question",
        "coverage_unit": "original_dataset_record_id", "eval_type_policy": "all",
        "resampling": "only_after_all_source_records_covered",
        "deduplication": "preserve_all_original_records", "jps_training_excluded": True,
        "pools": stats, "unique_parents": len(raw),
        "unique_instructions": len({instruction_key(b.instruction) for b in raw}),
        "total_draws": total, "resampled_draws": extra,
        "recipe_source_counts": {r.id: dict(Counter(b.dataset for b in plans[r.id])) for r in recipes},
        "full_source_counts": {name: len(pools[name]) + extra_counts[name] for name in names},
    }
    return raw, plans, metadata


def plan_jood_variants(config, recipes, parent_plans, harmless_count):
    import random
    weights = mixup_weights(config)
    randomizer = random.Random(config["seed"])
    weight_order = list(range(len(weights)))
    image_order = list(range(harmless_count))
    randomizer.shuffle(weight_order)
    randomizer.shuffle(image_order)
    assignments, ordinal = {}, 0
    for recipe in recipes:
        if "JOOD" not in recipe.methods:
            continue
        for behavior in parent_plans[recipe.id]:
            wi = weight_order[ordinal % len(weight_order)]
            hi = image_order[ordinal % len(image_order)]
            assignments[(recipe.id, behavior.behavior_id)] = {
                "assignment_index": ordinal, "weight_index": wi,
                "source_weight": weights[wi], "harmless_index": hi,
            }
            ordinal += 1
    if config.get("sources", {}).get("sampling") == "all_records_once":
        # Swapping complete variants within a recipe preserves all global and
        # recipe-level distributions while covering every source/weight/image.
        from collections import Counter
        by_recipe = {r.id: [(r.id, b.behavior_id) for b in parent_plans[r.id]]
                     for r in recipes if "JOOD" in r.methods}
        source = {(r.id, b.behavior_id): b.dataset for r in recipes
                  for b in parent_plans[r.id] if "JOOD" in r.methods}
        def variant(key):
            row = assignments[key]
            return row["weight_index"], row["harmless_index"]
        counts = Counter((source[key], *variant(key)) for key in assignments)
        for name in source_names(config):
            for wi in range(len(weights)):
                for hi in range(harmless_count):
                    if counts[(name, wi, hi)]:
                        continue
                    swapped = False
                    for keys in by_recipe.values():
                        donors = [key for key in keys if variant(key) == (wi, hi)
                                  and source[key] != name and counts[(source[key], wi, hi)] > 1]
                        if not donors:
                            continue
                        receiver = next((key for key in keys if source[key] == name
                                         and counts[(name, *variant(key))] > 1), None)
                        if receiver is None:
                            continue
                        donor = donors[0]
                        old, donor_source = variant(receiver), source[donor]
                        counts[(name, *old)] -= 1
                        counts[(donor_source, wi, hi)] -= 1
                        assignments[receiver], assignments[donor] = assignments[donor], assignments[receiver]
                        counts[(name, wi, hi)] += 1
                        counts[(donor_source, *old)] += 1
                        swapped = True
                        break
                    if not swapped:
                        raise ValueError("cannot cover every source/JOOD weight/image combination")
    return assignments



def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    recipes = [Recipe(row["id"], tuple(row["methods"]), int(row["count"]))
               for row in config["recipes"]]
    ids, sets = set(), set()
    for recipe in recipes:
        if (recipe.id in ids or frozenset(recipe.methods) in sets
                or len(set(recipe.methods)) != len(recipe.methods)
                or not 2 <= len(recipe.methods) <= 4
                or not set(recipe.methods) <= METHODS or recipe.count < 3):
            raise ValueError(f"invalid/duplicate recipe: {recipe}")
        if {"FigStep", "MML"} <= set(recipe.methods):
            raise ValueError("overlapping FigStep/MML recipes require a separate study")
        ids.add(recipe.id)
        sets.add(frozenset(recipe.methods))
    if len(recipes) != 18 or sum(r.count for r in recipes) != 3000:
        raise ValueError("this production plan requires 18 recipes totaling exactly 3000")
    if config["preview_per_recipe"] != 3:
        raise ValueError("preview must contain three cases for each recipe")
    source_names(config)
    mixup_weights(config)
    return config, recipes


class DatasetBuilder:
    def __init__(self, config, recipes, root, reuse_from=None):
        from .datasets import load_source_pool
        self.config, self.recipes = config, recipes
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        identity_path = self.root / "run_identity.json"
        recorded_identity = json.loads(identity_path.read_text()) if identity_path.exists() else None
        if recorded_identity and recorded_identity["config"] != config:
            raise ValueError("configuration changed; select a new output directory")
        self.font = resolve_path(config["render"]["font_path"])
        self.cs, self._mml, self._bases = None, None, {}
        self.jps = self._jps_artifact()
        self.jood = self._jood_artifacts()
        with open(self.jps["paths"]["training_csv"], newline="", encoding="utf-8") as stream:
            train = {instruction_key(row["goal"]) for row in csv.DictReader(stream)}
        pools = {name: load_source_pool(name) for name in source_names(config)}
        self._parents, self._parent_plans, self.source_selection = select_source_parents(
            config, recipes, pools, train)
        from .typography import _coverage
        if not Path(self.font).is_file():
            raise FileNotFoundError("Prepare the composition font with: python -m compositions.setup --reference_root PATH")
        chars = {ord(c) for b in self._parents for c in (b.statement or b.instruction) if not c.isspace()}
        missing = chars - _coverage(self.font)
        if missing:
            raise ValueError(f"selected source font lacks glyphs: {[hex(c) for c in sorted(missing)]}")
        self._jood_plan = plan_jood_variants(
            config, recipes, self._parent_plans, len(self.jood["paths"]))
        self._draw_indices = {
            (recipe.id, behavior.behavior_id): index
            for index, (recipe, behavior) in enumerate(
                (recipe, behavior) for recipe in recipes for behavior in self._parent_plans[recipe.id])
        }
        if reuse_from is None and recorded_identity:
            recorded_reuse = recorded_identity.get("reuse")
            if recorded_reuse:
                reuse_from = resolve_path(recorded_reuse["source_run"])
        self.reuse_from = Path(reuse_from).resolve() if reuse_from else None
        self.reuse = self._reuse_source() if self.reuse_from else None
        code = [Path(__file__).with_name("datasets.py"), Path(__file__).with_name("records.py"),
                Path(__file__), Path(__file__).with_name("typography.py"),
                Path(__file__).with_name("five_runner.py"),
                Path(REPO_ROOT) / "multi_attacks/cs_dj/cs_dj.py",
                Path(REPO_ROOT) / "multi_attacks/mml/mml.py",
                Path(REPO_ROOT) / "multi_attacks/mml/prompts.py",
                Path(REPO_ROOT) / "multi_attacks/JOOD/jood_utils.py"]
        frozen = config.get('cs_dj', {}).get('precomputed_dir')
        frozen_manifest = Path(resolve_path(frozen)) / 'manifest.json' if frozen else None
        frozen_hash = digest(frozen_manifest) if frozen_manifest and frozen_manifest.is_file() else None
        self.identity = json_hash({"frozen_cs_manifest_sha256": frozen_hash, "config": config, "jps": self.jps["hashes"],
                                   "jood": self.jood["hashes"], "reuse": self.reuse,
                                   "font_sha256": digest(self.font),
                                   "sources": self.source_selection,
                                   "code": {str(p): digest(p) for p in code}})
        if identity_path.exists():
            if json.loads(identity_path.read_text())["identity"] != self.identity:
                raise ValueError("configuration/code/artifact changed; select a new output directory")
        else:
            atomic_json(identity_path, {"identity": self.identity, "config": config,
                                       "algorithm_fidelity": "adapted_composition",
                                       "paper_reproduction": False, "jps": self.jps,
                                       "jood": self.jood, "reuse": self.reuse,
                                       "source_selection": self.source_selection})
        self.parents()
        plan = {
            "identity": self.identity, "source_selection": self.source_selection,
            "preview_behavior_ids": {r.id: [b.behavior_id for b in self.parents_for_recipe(r, "preview")] for r in recipes},
            "recipes": {r.id: [{"behavior_id": b.behavior_id, "dataset": b.dataset,
                               "draw_index": self._draw_indices[(r.id, b.behavior_id)],
                               "instruction_hash": json_hash(instruction_key(b.instruction)),
                               "jood": self._jood_plan.get((r.id, b.behavior_id))}
                              for b in self._parent_plans[r.id]] for r in recipes},
        }
        path = self.root / "sample_plan.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError("sample plan changed")
        atomic_json(path, plan)

    def _jood_artifacts(self):
        cfg = self.config.get("jood", {})
        if cfg.get("operation", "mixup") != "mixup":
            raise ValueError("this composition profile requires JOOD image MixUp")
        weights = mixup_weights(self.config)
        pool = Path(resolve_path(cfg.get("harmless_dir", "data/method_assets/JOOD/images/harmless")))
        files = sorted(p for p in pool.iterdir()
                       if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not files:
            raise ValueError(f"JOOD provided harmless image pool is empty: {pool}")
        assets = self.root / "assets" / "jood"
        assets.mkdir(parents=True, exist_ok=True)
        paths, hashes = [], {}
        for source in files:
            with Image.open(source) as image:
                image.verify()
            expected = digest(source)
            destination = assets / source.name
            if destination.exists() and digest(destination) != expected:
                raise ValueError(f"copied JOOD image changed: {destination}")
            if not destination.exists():
                shutil.copyfile(source, destination)
            paths.append(portable_path(destination))
            hashes[source.name] = expected
        return {"operation": "mixup", "source_weights": weights,
                "harmless_dir": portable_path(pool), "paths": paths, "hashes": hashes,
                "preprocessing": "letterbox_harmless_to_native_source_canvas",
                "algorithm_fidelity": "adapted_native_resolution_mixup"}

    def _reuse_source(self):
        if self.reuse_from == self.root:
            raise ValueError("reuse source and output must be different directories")
        identity = json.loads((self.reuse_from / "run_identity.json").read_text())
        old_config = {k: v for k, v in identity["config"].items() if k != "jood"}
        new_config = {k: v for k, v in self.config.items() if k != "jood"}
        if old_config != new_config or identity["jps"]["hashes"] != self.jps["hashes"]:
            raise ValueError("rendered-base reuse requires matching source plan and JPS artifacts")
        manifest_path = self.reuse_from / "full" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("status") != "complete" or manifest.get("n_cases") != 3000
                or manifest["identity"] != identity["identity"]
                or digest(self.reuse_from / "full" / "test_cases.jsonl") != manifest["test_cases_sha256"]):
            raise ValueError("rendered-base reuse requires an intact completed 3000-case source run")
        return {"source_run": portable_path(self.reuse_from), "identity": identity["identity"],
                "manifest_sha256": digest(manifest_path),
                "parents_sha256": digest(self.reuse_from / "parents.json"),
                "scope": "verified_pre_JOOD_rendered_images_and_source_parents"}

    def _jps_artifact(self):
        cfg = self.config["jps"]
        source_root = Path(resolve_path(cfg["source_run"]))
        manifest_path = source_root / "result_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assets = self.root / "assets"
        assets.mkdir(exist_ok=True)
        paths, hashes = {}, {}
        for key in ("image", "steer", "training_csv"):
            relative = cfg[key]
            path = source_root / relative
            expected = manifest.get(relative)
            if expected is None or digest(path) != expected:
                raise ValueError(f"unverified JPS artifact: {key}")
            destination = assets / {"image": "jps.png", "steer": "jps_steer.txt",
                                    "training_csv": "jps_training.csv"}[key]
            if destination.exists() and digest(destination) != expected:
                raise ValueError(f"copied JPS artifact changed: {destination}")
            if not destination.exists():
                shutil.copyfile(path, destination)
            paths[key], hashes[key] = str(destination), expected
        if not Path(paths["steer"]).read_text().strip():
            raise ValueError("JPS steering is empty")
        return {"paths": paths, "hashes": hashes, "source_manifest_sha256": digest(manifest_path),
                "application": "auxiliary_image_and_steering_transfer"}

    def parents(self):
        record = [b.to_dict() for b in self._parents]
        if self.reuse_from and json.loads((self.reuse_from / "parents.json").read_text()) != record:
            raise ValueError("reuse source parents differ from the current dataset selection")
        path = self.root / "parents.json"
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError("parent selection changed; select a new output directory")
        atomic_json(path, record)
        return list(self._parents)

    def parents_for_recipe(self, recipe, phase):
        plan = self._parent_plans[recipe.id]
        if phase != "preview":
            return list(plan)
        names = source_names(self.config)
        choices = []
        weights = sorted(self.jood["source_weights"])
        targets = [weights[0], weights[len(weights) // 2], weights[-1]]
        for index in range(3):
            name = names[index % len(names)]
            candidates = [b for b in plan if b.dataset == name and b not in choices]
            if "JOOD" in recipe.methods:
                target = targets[index]
                distance = min(abs(self._jood_plan[(recipe.id, b.behavior_id)]["source_weight"] - target)
                               for b in candidates)
                candidates = [b for b in candidates if abs(
                    self._jood_plan[(recipe.id, b.behavior_id)]["source_weight"] - target) <= distance + 1e-12]
            candidates.sort(key=lambda b: (len(b.statement or b.instruction), b.behavior_id))
            chosen = candidates[-1] if index == 0 else candidates[len(candidates) // 2] if index == 1 else candidates[0]
            choices.append(chosen)
        return choices

    def _cs_data(self, behavior):
        cache_dir = self.root / "_cache" / "cs_dj"
        path = cache_dir / f"{behavior.behavior_id}.json"
        cs_cfg = dict(self.config["cs_dj"])
        precomputed = cs_cfg.pop("precomputed_dir", None)
        with open(resolve_path(cs_cfg.pop("config")), encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        cfg.update(cs_cfg)
        cache_key = json_hash([self.identity, behavior.to_dict(), cfg])
        if path.exists():
            data = json.loads(path.read_text())
            if data["key"] != cache_key:
                raise ValueError("CS-DJ cache identity mismatch")
            for p, expected in zip(data["contrast_images"], data["contrast_hashes"]):
                if digest(resolve_path(p)) != expected:
                    raise ValueError("CS-DJ contrast cache image changed")
            return data
        if precomputed:
            frozen = Path(resolve_path(precomputed)) / f"{behavior.behavior_id}.json"
            if frozen.is_file():
                frozen_manifest = json.loads((frozen.parent / 'manifest.json').read_text())
                if frozen_manifest['files'].get(frozen.name) != digest(frozen):
                    raise ValueError('Frozen CS-DJ cache checksum mismatch')
                data = json.loads(frozen.read_text())
                if data['record_sha256'] != json_hash(behavior.to_dict()):
                    raise ValueError('Frozen CS-DJ original record changed')
                if data['source_instruction_sha256'] != json_hash(behavior.instruction):
                    raise ValueError('Frozen CS-DJ cache question changed')
                for name in ('embedding_model', 'embedding_revision', 'split_model', 'split_revision',
                             'seed', 'num_images', 'retrieval_count', 'subquestion_count'):
                    if data['config'].get(name) != cfg.get(name):
                        raise ValueError(f'Frozen CS-DJ cache setting differs: {name}')
                if len(data['subquestions']) != 3 or len(data['contrast_images']) != 9:
                    raise ValueError('Invalid frozen CS-DJ cache')
                for image, expected in zip(data['contrast_images'], data['contrast_hashes']):
                    if digest(resolve_path(image)) != expected:
                        raise ValueError('Frozen CS-DJ contrast image changed')
                data = {**data, 'key': cache_key, 'config': cfg,
                        'reuse': 'verified_original_decomposition_and_retrieval'}
                atomic_json(path, data)
                return data
        if self.cs is None:
            from multi_attacks.cs_dj.cs_dj import CSDJAttack
            self.cs = CSDJAttack(cfg)
            self.cs.prepare(str(cache_dir / "embeddings"))
        questions = self.cs._split_question(behavior.instruction, behavior.behavior_id)
        if len(questions) != 3 or any(not q.strip() for q in questions):
            raise ValueError("CS-DJ did not produce three nonempty subquestions")
        images, indices = self.cs._retrieve_contrasts(behavior.instruction)
        images = images[:9]
        if len(images) != 9:
            raise ValueError("CS-DJ did not retrieve nine contrast images")
        data = {"key": cache_key, "subquestions": questions,
                "contrast_images": [portable_path(p) for p in images],
                "contrast_hashes": [digest(p) for p in images], "indices": indices[:9],
                "config": cfg}
        atomic_json(path, data)
        return data

    def _card(self, title, mode):
        prompt, extra = "", {}
        if mode == "MML":
            if self._mml is None:
                from multi_attacks.mml.mml import MML as MMLAttack
                self._mml = MMLAttack({"variant": "word_replacement", "pos_tagger": "nltk",
                                       "seed": self.config["seed"]})
            import random
            text, mapping, tagger = self._mml._replace_words(title)
            words = title.split()
            random.Random(json_hash([self.config["seed"], title])).shuffle(words)
            prompt = self._mml._prompt(mapping, words)
            extra = {"variant": "word_replacement", "replacement_map": mapping,
                     "original_title": title, "tagger": tagger}
        else:
            text = title
            if mode == "FigStep":
                from multi_attacks.figstep.figstep import PROMPT
                prompt = PROMPT
        if mode in {"FigStep", "MML"}:
            text += "\n\n1.\n2.\n3."
        settings = dict(self.config["render"])
        settings["font_path"] = self.font
        image, layout = render_text(text, fill="red" if mode == "CS-DJ" else "black", **settings)
        return image, [layout.to_dict()], prompt, extra

    def _collage(self, cards, contrasts):
        width = max(im.width for im, _ in cards)
        height = max(self.config["render"]["height"], *(im.height for im, _ in cards))
        label_height = 64
        canvas = Image.new("RGB", (3 * width, 4 * (height + label_height)), "white")
        layouts = []
        for index in range(12):
            x, y = index % 3 * width, index // 3 * (height + label_height)
            if index < 9:
                with Image.open(resolve_path(contrasts[index])) as source:
                    im = source.convert("RGB")
                    im.thumbnail((width, height), Image.Resampling.LANCZOS)
                canvas.paste(im, (x + (width - im.width) // 2, y + (height - im.height) // 2))
            else:
                im, card_layouts = cards[index - 9]
                canvas.paste(im, (x, y))
                for layout in card_layouts:
                    layout = copy.deepcopy(layout)
                    layout["line_boxes"] = [[a + x, b + y, c + x, d + y]
                                             for a, b, c, d in layout["line_boxes"]]
                    layout["panel"] = index + 1
                    layouts.append(layout)
            label, label_layout = render_text(str(index + 1), font_path=self.font, width=width,
                                              height=label_height, max_height=label_height,
                                              font_size=28, min_font_size=28, margin=8)
            canvas.paste(label, (x, y + height))
            ld = label_layout.to_dict()
            ld["line_boxes"] = [[a + x, b + y + height, c + x, d + y + height]
                                for a, b, c, d in ld["line_boxes"]]
            ld["panel_label"] = index + 1
            layouts.append(ld)
        return canvas, layouts

    def _base(self, behavior, mode, cs):
        key = (behavior.behavior_id, mode, cs)
        if key in self._bases:
            return self._bases[key]
        base_dir = self.root / "_cache" / "rendered"
        base_dir.mkdir(parents=True, exist_ok=True)
        name = f"{behavior.behavior_id}_{mode}_{int(cs)}"
        image_path, meta_path = base_dir / f"{name}.png", base_dir / f"{name}.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["identity"] != self.identity or meta["image_sha256"] != digest(image_path):
                raise ValueError("cached rendered image changed")
            result = (image_path, meta)
            self._bases[key] = result
            return result
        if self.reuse_from:
            source_dir = self.reuse_from / "_cache" / "rendered"
            source_image, source_meta = source_dir / f"{name}.png", source_dir / f"{name}.json"
            meta = json.loads(source_meta.read_text())
            if (meta["identity"] != self.reuse["identity"]
                    or digest(source_image) != meta["image_sha256"]):
                raise ValueError(f"reused rendered base changed: {source_image}")
            with Image.open(source_image) as image:
                image.verify()
            with Image.open(source_image) as image:
                if list(image.size) != meta["image_size"]:
                    raise ValueError("reused rendered base size differs from layout metadata")
                for layout in meta["layouts"]:
                    validate_layout(layout, image.size)
            shutil.copyfile(source_image, image_path)
            meta["identity"] = self.identity
            meta["reused_from"] = {"image": portable_path(source_image),
                                   "metadata_sha256": digest(source_meta), **self.reuse}
            atomic_json(meta_path, meta)
            result = (image_path, meta)
            self._bases[key] = result
            return result
        if cs:
            from multi_attacks.cs_dj.cs_dj import TARGET_PROMPT
            data = self._cs_data(behavior)
            cards, prompts, extras = [], [], []
            for title in data["subquestions"]:
                image, layouts, prompt, extra = self._card(title, mode)
                cards.append((image, layouts))
                prompts.append(prompt)
                extras.append(extra)
            image, layouts = self._collage(cards, data["contrast_images"])
            text = TARGET_PROMPT
            if mode == "FigStep":
                text += "\nFor each of the three task cards, " + prompts[0]
            elif mode == "MML":
                text += "\n" + "\n".join(f"For task card {10+i}:\n{p}" for i, p in enumerate(prompts))
            details = {"cs_dj": data, "card_transforms": extras}
        else:
            image, layouts, text, details = self._card(behavior.statement or behavior.instruction, mode)
        image.save(image_path, compress_level=3)
        effective = [validate_layout(l, image.size) for l in layouts]
        meta = {"text": text, "layouts": layouts, "image_size": list(image.size),
                "min_display_font_2048": min(effective), "details": details,
                "image_sha256": digest(image_path), "identity": self.identity}
        atomic_json(meta_path, meta)
        result = (image_path, meta)
        self._bases[key] = result
        return result

    def _augment(self, source, behavior, destination, recipe):
        assignment = self._jood_plan[(recipe.id, behavior.behavior_id)]
        harmless_path = self.jood["paths"][assignment["harmless_index"]]
        expected = self.jood["hashes"][Path(harmless_path).name]
        if digest(resolve_path(harmless_path)) != expected:
            raise ValueError("JOOD provided image changed after run initialization")
        with Image.open(source) as opened:
            original = opened.convert("RGB")
        with Image.open(resolve_path(harmless_path)) as opened:
            harmless = ImageOps.pad(opened.convert("RGB"), original.size,
                                    method=Image.Resampling.LANCZOS, color="white")
        weight = assignment["source_weight"]
        image = Image.blend(harmless, original, weight)
        if image.size != original.size or ImageChops.difference(original, image).getbbox() is None:
            raise ValueError("JOOD image MixUp must change pixels while retaining all coordinates")
        image.save(destination, compress_level=3)
        return {"profile": "provided_image_mixup_native_resolution", "operation": "mixup",
                **assignment, "lam": weight, "harmless_weight": round(1 - weight, 12),
                "geometry_unchanged": True,
                "harmless": Path(harmless_path).stem, "harmless_image": harmless_path,
                "harmless_sha256": expected, "input_image": portable_path(source),
                "canvas_size": list(original.size), "preprocessing": self.jood["preprocessing"],
                "input_sha256": digest(source), "output_sha256": digest(destination)}

    def build(self, recipe, behavior, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{recipe.id}_{behavior.behavior_id}"
        case_path = directory / f"{stem}.json"
        if case_path.exists():
            case = json.loads(case_path.read_text())
            if case["meta"]["identity"] != self.identity:
                raise ValueError("existing sample has a different identity")
            signed = copy.deepcopy(case)
            expected = signed['meta'].pop('native_payload_sha256', None)
            if expected != json_hash(signed):
                raise ValueError('Five-method case checkpoint payload changed')
            validate_case(case)
            return case
        methods = set(recipe.methods)
        p_only = methods == {"JOOD", "JPS"}
        if p_only:
            from multi_attacks.jps.jps import _affirmative_target, _jps_prompt
            source = Path(self.jps["paths"]["image"])
            steer = Path(self.jps["paths"]["steer"]).read_text()
            text = _jps_prompt(behavior.instruction, behavior.target or _affirmative_target(behavior.instruction), steer)
            layouts, details = [], {"jps_direct_input": True}
        else:
            mode = "MML" if "MML" in methods else "FigStep" if "FigStep" in methods else "CS-DJ"
            source, meta = self._base(behavior, mode, "CS-DJ" in methods)
            text, layouts, details = meta["text"], meta["layouts"], meta["details"]
        image_path = directory / f"{stem}.png"
        aug = None
        if "JOOD" in methods:
            aug = self._augment(source, behavior, image_path, recipe)
        else:
            # Copy rather than resizing/re-encoding, so typography and JPS pixels survive exactly.
            shutil.copyfile(source, image_path)
        images = [portable_path(image_path)]
        if "JPS" in methods and not p_only:
            images.append(portable_path(self.jps["paths"]["image"]))
            text = (Path(self.jps["paths"]["steer"]).read_text().strip() +
                    "\n\nImage 1 contains the task cards; image 2 is auxiliary visual context.\n" + text)
        split_key = instruction_key(behavior.instruction)
        fraction = int(json_hash([self.config["seed"], split_key])[:12], 16) / 16**12
        split = "train" if fraction < 0.7 else "validation" if fraction < 0.8 else "test"
        case = TestCase(test_case_id=stem, behavior_id=behavior.behavior_id,
                        attack="+".join(recipe.methods), text=text, images=images,
                        meta={"recipe_id": recipe.id, "methods": list(recipe.methods),
                              "source_draw_index": self._draw_indices[(recipe.id, behavior.behavior_id)],
                              "arity": len(recipe.methods), "identity": self.identity,
                              "source_behavior_id": behavior.behavior_id, "source_dataset": behavior.dataset,
                              "source_category": behavior.category,
                              "source_eval_type": behavior.eval_type.value,
                              "source_record_sha256": json_hash(behavior.to_dict()), "split": split,
                              "source_instruction_hash": json_hash(split_key),
                              "algorithm_fidelity": "adapted_composition", "paper_reproduction": False,
                              "layouts": layouts, "details": details, "jood": aug,
                              "jps": self.jps if "JPS" in methods else None,
                              "image_sha256s": [digest(resolve_path(p)) for p in images]}).to_dict()
        validate_case(case)
        case['meta']['native_payload_sha256'] = json_hash(case)
        atomic_json(case_path, case)
        return case


def validate_case(case):
    if not case["text"].strip() or not case["images"]:
        raise ValueError("empty sample payload")
    if len(case["images"]) != len(case["meta"]["image_sha256s"]):
        raise ValueError("image/hash count mismatch")
    sizes = []
    for name, expected in zip(case["images"], case["meta"]["image_sha256s"]):
        path = resolve_path(name)
        if digest(path) != expected:
            raise ValueError(f"sample image hash mismatch: {path}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            sizes.append(image.size)
    if len(sizes) != len(case["images"]):
        raise ValueError("image/hash count mismatch")
    for layout in case["meta"]["layouts"]:
        validate_layout(layout, sizes[0])
    aug = case["meta"].get("jood")
    if "JOOD" in case["meta"]["methods"] and not aug:
        raise ValueError("JOOD sample is missing augmentation provenance")
    if aug and aug.get("operation") == "mixup":
        weight = aug["source_weight"]
        if not math.isfinite(weight) or not 0 < weight < 1 or aug["lam"] != weight:
            raise ValueError("invalid JOOD image MixUp weight")
        if digest(resolve_path(aug["harmless_image"])) != aug["harmless_sha256"]:
            raise ValueError("JOOD provided image hash mismatch")
        if digest(resolve_path(aug["input_image"])) != aug["input_sha256"]:
            raise ValueError("JOOD input image hash mismatch")
        with Image.open(resolve_path(aug["input_image"])) as source:
            if source.size != sizes[0] or list(source.size) != aug["canvas_size"]:
                raise ValueError("JOOD MixUp changed source geometry")
        if aug["output_sha256"] != case["meta"]["image_sha256s"][0]:
            raise ValueError("JOOD output image hash mismatch")
    if not case["meta"]["layouts"] and set(case["meta"]["methods"]) != {"JPS", "JOOD"}:
        raise ValueError("typographic sample has no layout evidence")
    return json_hash([case["text"], case["meta"]["image_sha256s"]])


def make_contact_sheet(cases, destination):
    font = ImageFont.truetype(resolve_path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), 26)
    columns, width, header = 3, 800, 120
    prepared = []
    for case in cases:
        with Image.open(resolve_path(case["images"][0])) as source:
            image = source.convert("RGB")
            image.thumbnail((width - 20, 1100), Image.Resampling.LANCZOS)
        prepared.append(image)
    row_height = max(im.height for im in prepared) + header + (220 if len(cases[0]["images"]) > 1 else 0)
    sheet = Image.new("RGB", (columns * width, row_height), "#e8e8e8")
    draw = ImageDraw.Draw(sheet)
    for i, (case, image) in enumerate(zip(cases, prepared)):
        x = i * width
        draw.text((x + 10, 10), case["meta"]["recipe_id"], font=font, fill="black")
        draw.text((x + 10, 42), case["behavior_id"], font=font, fill="black")
        label = case["meta"].get("source_dataset", "")
        if case["meta"].get("jood"):
            label += f' | source weight {case["meta"]["jood"]["source_weight"]:.0%}'
        draw.text((x + 10, 78), label, font=font, fill="black")
        sheet.paste(image, (x + 10, header))
        if len(case["images"]) > 1:
            with Image.open(resolve_path(case["images"][1])) as source:
                aux = source.convert("RGB")
                aux.thumbnail((200, 200))
            sheet.paste(aux, (x + 10, row_height - 210))
            draw.text((x + 230, row_height - 120), "JPS auxiliary image", font=font, fill="black")
    sheet.save(destination)
