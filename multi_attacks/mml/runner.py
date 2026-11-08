"""MML online CLAS protocol for the existing JSON completion CLI."""
import json
import os
from pathlib import Path

from benchmark_datasets import read_behavior_sidecar, write_behavior_sidecar
from data_paths import resolve_path
from image_attacks.jp_common import write_json
from multimodalmodels.inference import case_groups, prepare_case
from .clas import MMLCLASJudge
from .mml import MML
from .official_source import load_upstream_judge_prompt
from .protocol import run_official_loop
from .remote_target import OpenAIVisionTarget
from .schema import TestCase

UPSTREAM_CONST_SHA256 = '89d7480ff98a4872b50f16c8bf4271f37074127fa0c1cfdd46a1f50b2cdce9a7'
OFFICIAL_DATASET_COUNTS = {'SafeBench': 500, 'MM-SafetyBench': 1180, 'HADES': 750}


def execute(args):
    path = Path(args.test_cases_path).resolve()
    payload = json.loads(path.read_text())
    snapshot = read_behavior_sidecar(path, payload, verify_artifacts=True)
    if snapshot is None or snapshot['method_name'] != 'MML':
        raise ValueError('mml_official requires MML test cases with original-question sidecar')
    rows = snapshot['behaviors']
    groups = case_groups(payload)
    if not rows or not groups:
        raise ValueError('MML protocol requires nonempty questions and test cases')
    cases = []
    for behavior_id, group in groups.items():
        if len(group) < args.attempts:
            raise ValueError('Every MML question must have the complete requested attempt budget')
        seen = set()
        for case in group:
            if not isinstance(case, dict) or case.get('attack') != 'MML':
                raise ValueError('mml_official accepts only MML record cases')
            attempt = case.get('attempt')
            if type(attempt) is not int or attempt < 0 or attempt in seen:
                raise ValueError('MML attempt indices must be unique nonnegative integers')
            seen.add(attempt)
            text, images = prepare_case(case, path.parent / 'images')
            if not isinstance(case.get('meta', {}).get('original_title'), str):
                raise ValueError('MML official protocol requires original_title metadata')
            cases.append(TestCase(case['test_case_id'], behavior_id, 'MML', text,
                                  images, attempt, case['meta']))
        if set(range(args.attempts)) - seen:
            raise ValueError('MML requested attempts must start at zero without gaps')
    dataset = snapshot['source']['dataset_name']
    row_datasets = {row.get('dataset') for row in rows}
    if dataset == 'custom' and len(row_datasets) == 1 and None not in row_datasets:
        dataset = next(iter(row_datasets))
    expected = OFFICIAL_DATASET_COUNTS.get(dataset)
    if not args.allow_subset and expected is not None and len(rows) != expected:
        raise ValueError(f'Official MML {dataset} requires {expected} questions; use --allow_subset for a non-paper smoke run')
    save = Path(args.save_path).resolve()
    summary_path = Path(args.summary_path or str(save) + '.mml-summary.json').resolve()
    outputs = [save, summary_path, Path(str(save) + '.metadata.json'), Path(str(save) + '.behaviors.json')]
    inputs = {path, Path(str(path) + '.behaviors.json'), Path(resolve_path(args.upstream_const_path)).resolve()}
    inputs.update(Path(image) for case in cases for image in case.images)
    if len(outputs) != len(set(outputs)) or inputs.intersection(outputs):
        raise ValueError('MML output paths must be distinct and must not overwrite inputs')
    judge_prompt = load_upstream_judge_prompt(resolve_path(args.upstream_const_path),
                                              expected_sha256=args.upstream_const_sha256)
    # Credentials are checked before any target creation or paid request.
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY is required for MML CLAS; no target was loaded or queried')
    if args.model_name.startswith('gpt-'):
        target = OpenAIVisionTarget(args.model_name, max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature, retry_limit=args.retry_limit,
                                    retry_wait_seconds=5, bad_request_fallback="I'm sorry.")
        revision = None
    else:
        from multimodalmodels.targets import load_target
        from multimodalmodels.registry import default_revision
        target = load_target(args.model_name, max_new_tokens=args.max_new_tokens)
        target.name = args.model_name
        target.temperature = args.temperature
        if hasattr(target, 'do_sample'):
            target.do_sample = args.temperature > 0
        revision = default_revision(args.model_name)
    judge = MMLCLASJudge(args.judge_model, retry_limit=5, prompt_template=judge_prompt)
    queried, summary = run_official_loop(cases, [MML.behavior(row) for row in rows],
                                         target, judge, attempts=args.attempts)
    originals = {case['test_case_id']: case for group in groups.values() for case in group}
    returned = {behavior_id: [] for behavior_id in groups}
    for record in queried:
        returned[record.behavior_id].append({
            'test_case': originals[record.test_case_id], 'generation': record.response,
            'test_case_id': record.test_case_id, 'attempt': record.attempt, 'attack': 'MML',
            'label': record.label, 'judge': record.judge, 'meta': record.meta})
    metadata = {'model_name': args.model_name, 'model_id': args.model_name, 'revision': revision,
                'max_new_tokens': args.max_new_tokens, 'generation': {'temperature': args.temperature},
                'protocol': 'mml_official', 'judge_model': args.judge_model, 'attempts': args.attempts,
                'allow_subset': args.allow_subset, 'paper_reproduction': False,
                'target_model_substituted': args.model_name != 'gpt-4o-2024-08-06'}
    write_json(save, returned)
    write_json(str(save) + '.metadata.json', metadata)
    write_behavior_sidecar(save, snapshot)
    write_json(summary_path, summary)
    print(f'MML CLAS: {len(queried)} queries; summary saved to {summary_path}')
    return returned, summary
