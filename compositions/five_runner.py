#!/usr/bin/env python
"""Generate previews, record a visual review, then build all 3,000 inputs.

See --help. The production phase requires an unchanged, validated 54-case
preview and an explicit visual-review record. No target-model queries occur.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .five_methods import (
    DatasetBuilder,
    atomic_json,
    digest,
    load_config,
    make_contact_sheet,
    validate_case,
    json_hash,
)


def validate_phase(root, recipes, phase, identity):
    directory = root / phase
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["identity"] != identity or manifest["status"] != "complete":
        raise ValueError("incomplete or mismatched phase manifest")
    expected = {r.id: 3 if phase == "preview" else r.count for r in recipes}
    cases = [json.loads(line) for line in (directory / "test_cases.jsonl").read_text().splitlines() if line]
    plan = None
    if manifest.get("source_selection"):
        if (digest(root / "sample_plan.json") != manifest["sample_plan_sha256"]
                or digest(root / "parents.json") != manifest["parents_sha256"]):
            raise ValueError("source selection/assignment plan changed")
        plan = json.loads((root / "sample_plan.json").read_text())
        if plan["identity"] != identity or plan["source_selection"] != manifest["source_selection"]:
            raise ValueError("source plan identity mismatch")
        planned = {(rid, row["behavior_id"]): row for rid, rows in plan["recipes"].items() for row in rows}
        parents = {b["behavior_id"]: b for b in json.loads((root / "parents.json").read_text())}
        if digest(directory / "behaviors.jsonl") != manifest["behaviors_sha256"]:
            raise ValueError("original scoring behaviors changed")
        behavior_rows = [json.loads(l) for l in (directory / "behaviors.jsonl").read_text().splitlines() if l]
        if (len({b["behavior_id"] for b in behavior_rows}) != len(behavior_rows)
                or any(parents.get(b["behavior_id"]) != b for b in behavior_rows)
                or {b["behavior_id"] for b in behavior_rows} != {c["behavior_id"] for c in cases}):
            raise ValueError("scoring behaviors differ from the selected original parents")
    counts, contents, ids, source_splits = Counter(), set(), set(), {}
    source_counts, source_categories, recipe_sources = Counter(), {}, {}
    jood_counts, harmless_counts, weight_counts = Counter(), Counter(), Counter()
    recipe_weights, source_weights, joint_counts = {}, {}, Counter()
    for case in cases:
        meta = case["meta"]
        if meta["identity"] != identity:
            raise ValueError("sample identity mismatch")
        if case["test_case_id"] in ids:
            raise ValueError("duplicate test case id")
        ids.add(case["test_case_id"])
        rid = meta["recipe_id"]
        content = validate_case(case)
        row = None
        if plan:
            row = planned.get((rid, case["behavior_id"]))
            if (not row or meta["source_dataset"] != row["dataset"]
                    or meta["source_instruction_hash"] != row["instruction_hash"]
                    or meta["source_record_sha256"] != json_hash(parents[case["behavior_id"]])):
                raise ValueError("sample differs from its planned original benchmark record")
            if row.get("draw_index") is not None and meta.get("source_draw_index") != row["draw_index"]:
                raise ValueError("sample differs from its question draw assignment")
            if (manifest["source_selection"].get("eval_type_policy") == "all"
                    and meta.get("source_eval_type") != parents[case["behavior_id"]]["eval_type"]):
                raise ValueError("original benchmark evaluation type changed")
            if phase == "preview" and case["behavior_id"] not in plan["preview_behavior_ids"][rid]:
                raise ValueError("preview parent differs from the declared representative subset")
            source = meta["source_dataset"]
            source_counts[source] += 1
            source_categories.setdefault(source, Counter())[meta["source_category"]] += 1
            recipe_sources.setdefault(rid, Counter())[source] += 1
        if "JOOD" in meta["methods"] and manifest.get("jood"):
            aug = meta["jood"]
            profile = manifest["jood"]
            weights = profile.get("source_weights", [profile.get("source_weight", 0.7)])
            if (aug.get("operation") != "mixup" or aug["source_weight"] not in weights
                    or profile["hashes"].get(Path(aug["harmless_image"]).name) != aug["harmless_sha256"]):
                raise ValueError("JOOD sample differs from the declared image/weight profile")
            if row:
                if any(aug[k] != row["jood"][k] for k in row["jood"]):
                    raise ValueError("JOOD variant differs from its deterministic assignment")
                if aug["harmless_image"] != profile["paths"][aug["harmless_index"]]:
                    raise ValueError("JOOD harmless-image assignment mismatch")
            weight = f'{aug["source_weight"]:.2f}'
            jood_counts[rid] += 1
            harmless_counts[aug["harmless"]] += 1
            weight_counts[weight] += 1
            recipe_weights.setdefault(rid, Counter())[weight] += 1
            if plan:
                source_weights.setdefault(meta["source_dataset"], Counter())[weight] += 1
            joint_counts[weight + "|" + aug["harmless"]] += 1
        if content in contents:
            raise ValueError("duplicate final text/image input")
        contents.add(content)
        counts[rid] += 1
        source = meta["source_instruction_hash"]
        split = meta["split"]
        if source in source_splits and source_splits[source] != split:
            raise ValueError("same source instruction crosses dataset splits")
        source_splits[source] = split
    if dict(counts) != expected:
        raise ValueError(f"recipe quota mismatch: {dict(counts)} != {expected}")
    if plan:
        declared = manifest["source_selection"]
        if phase == "full":
            actual_pairs = {(c["meta"]["recipe_id"], c["behavior_id"]) for c in cases}
            if actual_pairs != set(planned) or len(actual_pairs) != len(cases):
                raise ValueError("full output does not contain exactly one case per planned question/recipe pair")
            if declared.get("selection_policy") == "all_records_once_then_resample":
                expected_ids = set(parents)
                covered_ids = {case["behavior_id"] for case in cases}
                ordered = sorted(cases, key=lambda c: c["meta"]["source_draw_index"])
                if (covered_ids != expected_ids
                        or len(expected_ids) != sum(p["loaded"] for p in declared["pools"].values())
                        or {c["behavior_id"] for c in ordered[:len(expected_ids)]} != expected_ids
                        or {c["meta"]["source_draw_index"] for c in cases} != set(range(len(cases)))):
                    raise ValueError("not every original record was used before resampling")
                normalized_pairs = {(c["meta"]["recipe_id"], c["meta"]["source_instruction_hash"]) for c in cases}
                if len(normalized_pairs) != len(cases):
                    raise ValueError("duplicate normalized question/recipe pair")
                for name, stats in declared["pools"].items():
                    uses = Counter(c["behavior_id"] for c in cases if c["meta"]["source_dataset"] == name)
                    types = Counter(b["eval_type"] for b in behavior_rows if b["dataset"] == name)
                    if (len(uses) != stats["loaded"] or max(uses.values()) > 2
                            or dict(types) != stats["eval_type_counts"]
                            or {str(k): v for k, v in Counter(uses.values()).items()} != stats["draws_per_parent"]):
                        raise ValueError(f"incomplete all-record coverage: {name}")
            if declared.get("selection_policy") == "full_pool_cycles_one_recipe_per_draw":
                for name, stats in declared["pools"].items():
                    draws = Counter(c["behavior_id"] for c in cases if c["meta"]["source_dataset"] == name)
                    pool_size = stats["unique_after_cross_dataset_deduplication"]
                    expected_unique = min(pool_size, declared["full_source_counts"][name])
                    histogram = dict(Counter(draws.values()))
                    if pool_size > len(draws):
                        histogram[0] = pool_size - len(draws)
                    if (len(draws) != expected_unique or len(draws) != stats["selected_parents"]
                            or max(histogram) - min(histogram) > 1
                            or {str(k): v for k, v in histogram.items()} != stats["draws_per_parent"]):
                        raise ValueError(f"full source-pool coverage/reuse mismatch: {name}")
        for recipe in recipes:
            target = (dict(Counter(planned[(recipe.id, bid)]["dataset"] for bid in plan["preview_behavior_ids"][recipe.id])) if phase == "preview"
                      else declared["recipe_source_counts"][recipe.id])
            if dict(recipe_sources[recipe.id]) != target:
                raise ValueError(f"benchmark balance mismatch: {recipe.id}")
        target = (dict(Counter(planned[(rid, bid)]["dataset"] for rid, bids in plan["preview_behavior_ids"].items() for bid in bids)) if phase == "preview"
                  else declared["full_source_counts"])
        if dict(source_counts) != target:
            raise ValueError("global benchmark balance mismatch")
    expected_jood = {r.id: expected[r.id] for r in recipes if "JOOD" in r.methods}
    if manifest.get("jood") and dict(jood_counts) != expected_jood:
        raise ValueError("JOOD provided-image mixing quota mismatch")
    if phase == "full" and manifest.get("jood", {}).get("source_weights"):
        keys = [f"{w:.2f}" for w in manifest["jood"]["source_weights"]]
        for distribution in [weight_counts, *recipe_weights.values()]:
            values = [distribution[k] for k in keys]
            if max(values) - min(values) > 1:
                raise ValueError("JOOD ratio distribution is not balanced")
        if min(weight_counts[k] for k in keys) < 1:
            raise ValueError("not all declared JOOD ratios were generated")
    if digest(directory / "test_cases.jsonl") != manifest["test_cases_sha256"]:
        raise ValueError("test_cases.jsonl changed after generation")
    if phase == "preview":
        for recipe in recipes:
            sheet = directory / "contact_sheets" / f"{recipe.id}.png"
            if digest(sheet) != manifest["contact_sheet_sha256s"][recipe.id]:
                raise ValueError("preview contact sheet changed")
    report = {"status": "passed", "n_cases": len(cases), "counts": dict(counts),
              "unique_inputs": len(contents), "n_source_instructions": len(source_splits),
              "n_source_records": len({case["behavior_id"] for case in cases}),
              "clipped_layouts": 0, "identity": identity,
              "test_cases_sha256": digest(directory / "test_cases.jsonl"),
              "manifest_sha256": digest(manifest_path)}
    if plan:
        report.update({"source_counts": dict(source_counts),
                       "selection_policy": manifest["source_selection"]["selection_policy"],
                       "unique_parents_per_source": {
                           name: len({c["behavior_id"] for c in cases if c["meta"]["source_dataset"] == name})
                           for name in source_counts},
                       "recipe_source_counts": {k: dict(v) for k, v in recipe_sources.items()},
                       "source_categories": {k: dict(v) for k, v in source_categories.items()}})
        if manifest["source_selection"].get("eval_type_policy") == "all":
            report.update({"eval_type_policy": "all",
                           "source_eval_type_counts": dict(Counter(b["eval_type"] for b in behavior_rows)),
                           "resampled_draws": len(cases) - len(behavior_rows),
                           "all_original_records_covered": phase == "full" and len(behavior_rows) == len(parents)})
    if manifest.get("jood"):
        report.update({"jood_image_mixed_cases": sum(jood_counts.values()),
                       "jood_counts": dict(jood_counts), "jood_harmless_images": dict(harmless_counts),
                       "jood_source_weights": dict(sorted(weight_counts.items())),
                       "jood_recipe_weights": {k: dict(sorted(v.items())) for k, v in recipe_weights.items()},
                       "jood_dataset_weights": {k: dict(sorted(v.items())) for k, v in source_weights.items()},
                       "jood_weight_image_pairs": dict(sorted(joint_counts.items()))})
    atomic_json(directory / "qc.json", report)
    return report


def generate(builder, recipes, phase):
    root = builder.root
    directory = root / phase
    directory.mkdir(exist_ok=True)
    parents = builder.parents()
    cases, counts = [], Counter()
    atomic_json(directory / "status.json", {"status": "running", "phase": phase, "identity": builder.identity})
    for recipe in recipes:
        count = 3 if phase == "preview" else recipe.count
        print(f"[{phase}] {recipe.id} {' + '.join(recipe.methods)}: {count}", flush=True)
        selected = []
        for i, parent in enumerate(builder.parents_for_recipe(recipe, phase)):
            case = builder.build(recipe, parent, directory / "cases" / recipe.id)
            cases.append(case)
            selected.append(case)
            counts[recipe.id] += 1
            if (i + 1) % 20 == 0:
                print(f"  {recipe.id}: {i + 1}/{count}", flush=True)
                atomic_json(directory / "status.json", {"status": "running", "phase": phase,
                             "n_generated": len(cases), "counts": dict(counts), "identity": builder.identity})
        if phase == "preview":
            sheets = directory / "contact_sheets"
            sheets.mkdir(exist_ok=True)
            make_contact_sheet(selected, sheets / f"{recipe.id}.png")
    temporary = directory / "test_cases.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    temporary.replace(directory / "test_cases.jsonl")
    selected_ids = {case["behavior_id"] for case in cases}
    behavior_path = directory / "behaviors.jsonl"
    with behavior_path.open("w", encoding="utf-8") as stream:
        for behavior in parents:
            if behavior.behavior_id in selected_ids:
                stream.write(json.dumps(behavior.to_dict(), ensure_ascii=False) + "\n")
    manifest = {"status": "complete", "phase": phase, "identity": builder.identity,
                "n_cases": len(cases), "counts": dict(counts),
                "algorithm_fidelity": "adapted_composition", "paper_reproduction": False,
                "target_model_queried": False,
                "jood": builder.jood, "reused_rendered_bases": builder.reuse,
                "source_selection": builder.source_selection,
                "sample_plan_sha256": digest(root / "sample_plan.json"),
                "parents_sha256": digest(root / "parents.json"),
                "behaviors_sha256": digest(behavior_path),
                "test_cases_sha256": digest(directory / "test_cases.jsonl")}
    if phase == "preview":
        manifest["contact_sheet_sha256s"] = {
            r.id: digest(directory / "contact_sheets" / f"{r.id}.png") for r in recipes}
    atomic_json(directory / "manifest.json", manifest)
    report = validate_phase(root, recipes, phase, builder.identity)
    atomic_json(directory / "status.json", {"status": "complete", **report})
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["preview", "review-preview", "full", "validate"])
    parser.add_argument("--config", default="configs/composition_configs/five_methods_3000.yaml")
    parser.add_argument("--output", default="runs/five_methods_3000_all_records_20261104")
    parser.add_argument("--reuse-from", help="Completed source run whose verified pre-JOOD renders are reused")
    parser.add_argument("--validate-phase", choices=["preview", "full"], default="full")
    parser.add_argument("--review-note", help="Visual observations after inspecting all 18 contact sheets")
    args = parser.parse_args(argv)
    config, recipes = load_config(args.config)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder = DatasetBuilder(config, recipes, root, reuse_from=args.reuse_from)
        try:
            if args.phase == "validate":
                report = validate_phase(root, recipes, args.validate_phase, builder.identity)
                print(json.dumps(report, ensure_ascii=False), flush=True)
            elif args.phase == "review-preview":
                if not args.review_note or not args.review_note.strip():
                    parser.error("--review-note is required after visual inspection")
                qc = validate_phase(root, recipes, "preview", builder.identity)
                atomic_json(root / "preview" / "visual_review.json", {
                    "accepted": True, "identity": builder.identity,
                    "manifest_sha256": qc["manifest_sha256"],
                    "test_cases_sha256": qc["test_cases_sha256"],
                    "n_cases": 54, "n_recipe_sheets": 18, "note": args.review_note,
                    "reviewed_at": datetime.now(timezone.utc).isoformat()})
                print("Visual review recorded for the unchanged 54-case preview.", flush=True)
            elif args.phase == "full":
                qc = validate_phase(root, recipes, "preview", builder.identity)
                review = json.loads((root / "preview" / "visual_review.json").read_text())
                if (review.get("accepted") is not True or review["identity"] != builder.identity
                        or review["manifest_sha256"] != qc["manifest_sha256"]
                        or review["test_cases_sha256"] != qc["test_cases_sha256"]):
                    raise ValueError("production requires a valid, unchanged visually reviewed preview")
                generate(builder, recipes, "full")
            else:
                generate(builder, recipes, "preview")
        except Exception as exc:
            atomic_json(root / "last_error.json", {"phase": args.phase, "error": str(exc),
                                                    "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
