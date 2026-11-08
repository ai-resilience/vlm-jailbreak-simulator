"""Generate completions through shared targets while retaining JSON CLI inputs."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import yaml

from multimodalmodels.inference import case_groups, load_generation_function, model_metadata
from benchmark_datasets import digest, read_behavior_sidecar, sidecar_path, write_behavior_sidecar


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate target-model responses for JSON test cases.")
    parser.add_argument("--model_name", required=True, help="Target model alias or configured model name")
    parser.add_argument("--models_config_file", default="./configs/model_configs/models.yaml")
    parser.add_argument("--behaviors_path", default="./data/original/HarmBench/behaviors/harmbench_behaviors_text_all.csv",
                        help="Accepted for compatibility; this stage reads prompts from test_cases_path")
    parser.add_argument("--test_cases_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--protocol", choices=("standard", "mml_official"), default="standard")
    parser.add_argument("--defense_name", help="Optional defense from configs/defense_configs")
    parser.add_argument("--defense_config", help="Optional YAML override for the selected defense")
    parser.add_argument("--attempts", type=int, help="MML maximum query budget, 1..5")
    parser.add_argument("--temperature", type=float, help="MML target sampling temperature")
    parser.add_argument("--retry_limit", type=int, help="MML target request retries")
    parser.add_argument("--judge_model", help="MML CLAS chat judge")
    parser.add_argument("--allow_subset", action="store_true", help="Explicitly permit an MML non-paper subset")
    parser.add_argument("--summary_path", help="MML summary JSON")
    parser.add_argument("--upstream_const_path", default="data/method_assets/MML/upstream/const.py")
    parser.add_argument("--upstream_const_sha256", default="89d7480ff98a4872b50f16c8bf4271f37074127fa0c1cfdd46a1f50b2cdce9a7")
    parser.add_argument("--generate_with_vllm", action="store_true")
    parser.add_argument("--incremental_update", action="store_true")
    args = parser.parse_args(argv)
    if args.protocol == "mml_official":
        if args.defense_name or args.defense_config:
            parser.error("mml_official does not compose with defenses; use the standard protocol")
        import math
        args.max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 2000
        args.attempts = args.attempts if args.attempts is not None else 5
        args.temperature = args.temperature if args.temperature is not None else 0.7
        args.retry_limit = args.retry_limit if args.retry_limit is not None else 3
        args.judge_model = args.judge_model or "gpt-4o-mini-2024-07-18"
        if not 1 <= args.attempts <= 5 or args.retry_limit < 1:
            parser.error("MML requires 1..5 attempts and positive retry_limit")
        if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
            parser.error("MML temperature must be in [0,2]")
        if args.incremental_update:
            parser.error("mml_official does not support incremental_update; use a new output path")
    else:
        if args.defense_config and not args.defense_name:
            parser.error("--defense_config requires --defense_name")
        if any(value is not None for value in (args.attempts, args.temperature, args.retry_limit, args.judge_model, args.summary_path)) or args.allow_subset:
            parser.error("MML protocol options require --protocol mml_official")
        args.max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 256
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be positive")
    if args.generate_with_vllm:
        parser.error("The registered target adapters use Transformers/API inference; remove --generate_with_vllm")
    return args


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as output:
            temporary = Path(output.name)
            json.dump(data, output, indent=4, ensure_ascii=False)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    args = parse_args(argv)
    if args.protocol == "mml_official":
        from multi_attacks.mml.runner import execute
        return execute(args)
    with open(args.models_config_file, encoding="utf-8") as source:
        configs = yaml.safe_load(source) or {}
    entry = configs.get(args.model_name)
    config = dict(entry["model"]) if entry is not None else {"model_name_or_path": args.model_name}
    metadata = {"model_name": args.model_name, **model_metadata(config, args.max_new_tokens)}
    with open(args.test_cases_path, encoding="utf-8") as source:
        payload = json.load(source)
    groups = case_groups(payload)
    snapshot = read_behavior_sidecar(args.test_cases_path, payload, verify_artifacts=True)
    if snapshot is not None and snapshot.get('composition'):
        from compositions.validation import validate_input
        metadata['composition'] = validate_input(payload, snapshot)
    defense = None
    if args.defense_name:
        from defenses import load_defense
        from defenses.inference import preflight_cases
        defense = load_defense(args.defense_name, config_path=args.defense_config)
        preflight_cases(config, [case for group in groups.values() for case in group], args.test_cases_path, defense)
        defense.prepare()
        metadata['defense'] = defense.metadata()
    returned = {}
    metadata_path = Path(str(args.save_path) + ".metadata.json")
    if Path(args.save_path).is_file():
        if metadata_path.is_file():
            previous_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if previous_metadata.get('defense') != metadata.get('defense'):
                raise ValueError('Existing completions use another defense/configuration; choose another --save_path')
        elif defense is not None:
            raise ValueError('Existing completions have no defense provenance; choose another --save_path')
    if args.incremental_update and Path(args.save_path).is_file():
        if metadata_path.is_file():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if previous != metadata:
                raise ValueError("Existing completions use different model settings; choose another --save_path")
        with open(args.save_path, encoding="utf-8") as source:
            returned = json.load(source)
        if not isinstance(returned, dict):
            raise ValueError("Existing completions must be keyed by BehaviorID")
        previous_snapshot = read_behavior_sidecar(args.save_path, returned)
        if digest(previous_snapshot) != digest(snapshot):
            raise ValueError("Existing completions use different saved questions or generation settings; choose another --save_path")
        if set(returned) - set(groups):
            raise ValueError("Existing completions contain other behaviors; choose another --save_path")
    pending = []
    for behavior_id, cases in groups.items():
        existing = returned.get(behavior_id, [])
        if not isinstance(existing, list):
            raise ValueError(f"Existing completions for {behavior_id!r} must be a list")
        records = []
        for index, case in enumerate(cases):
            cached = existing[index] if index < len(existing) else None
            if cached is not None and defense is not None:
                if not isinstance(cached, dict) or type(cached.get('blocked')) is not bool or cached.get('meta', {}).get('defense') != defense.name:
                    raise ValueError('Cached completion is missing the selected defense decision')
            if isinstance(cached, dict) and defense is None and cached.get('meta', {}).get('defense'):
                raise ValueError('Cached defended completion cannot be reused as an undefended baseline')
            if (isinstance(cached, dict) and cached.get("test_case") == case
                    and isinstance(cached.get("generation"), str)):
                records.append(cached)
            else:
                records.append(None)
                pending.append((behavior_id, index, case))
        returned[behavior_id] = records
    if not pending:
        print("No test cases to generate completions for")
        return
    if defense is None:
        generate = load_generation_function(config, args.max_new_tokens, args.test_cases_path)
    else:
        from defenses.inference import load_defended_generation_function
        generate = load_defended_generation_function(config, args.max_new_tokens, args.test_cases_path, defense)
    print(f"Generating {len(pending)} completions with {args.model_name}")
    responses = generate(test_cases=[case for _, _, case in pending], progress_bar=True)
    for (behavior_id, index, case), response in zip(pending, responses):
        record = {"test_case": case}
        if defense is None:
            record['generation'] = response
        else:
            record.update(response)
        if isinstance(case, dict):
            for key in ('test_case_id', 'attempt', 'attack'):
                if key in case:
                    record[key] = case[key]
            if case.get('meta', {}).get('composition_identity'):
                record.setdefault('meta', {})['composition'] = dict(case['meta'])
        returned[behavior_id][index] = record
    write_json(args.save_path, returned)
    write_json(metadata_path, metadata)
    if snapshot is not None:
        write_behavior_sidecar(args.save_path, snapshot)
    elif sidecar_path(args.save_path).is_file():
        # A fresh run with legacy inputs must not retain another run's questions.
        sidecar_path(args.save_path).unlink()
    print(f"Saved completions to {args.save_path}")


if __name__ == "__main__":
    main()
