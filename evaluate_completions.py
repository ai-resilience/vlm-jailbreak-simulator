"""Evaluate existing completion JSON and report six attack/six defense metrics."""

import argparse
import json
import os
from pathlib import Path
import tempfile

from benchmark_datasets import (
    DATASET_NAMES, DEFAULT_BEHAVIORS_PATH, load_dataset, merge_original_behaviors,
    read_behavior_sidecar, read_csv, sidecar_path, write_behavior_sidecar,
)

from eval_utils import (
    ResponseGrader,
    compute_evaluation_metrics,
    compute_results_advbench,
    compute_results_classifier,
    compute_results_hashing,
    grade_evaluation_results,
    normalize_evaluation_behaviors,
)


def save_asr_to_file(model_name, asr_value, file_path):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as output:
        value = f'{asr_value:.4f}' if asr_value is not None else 'unavailable'
        output.write(f'Model: {model_name}  ASR: {value}\n')
    print(f'ASR result saved to {file_path}: {value}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method_name', help='Attack method name')
    parser.add_argument('--model_name', help='Target model; otherwise read from the completion metadata')
    parser.add_argument('--cls_path', default='cais/HarmBench-Llama-2-13b-cls')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--dataset_name', choices=DATASET_NAMES)
    source.add_argument('--behaviors_path', help='Custom CSV; otherwise use saved questions, then archived HarmBench')
    parser.add_argument('--completions_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--include_advbench_metric', action='store_true')
    parser.add_argument('--num_tokens', type=int, default=512,
                        help='Compatibility option; evaluation uses the full response')
    parser.add_argument('--asr_results_file', default='asr_results.txt')
    parser.add_argument('--metrics_path', help='Summary JSON; default: SAVE_PATH.metrics.json')
    parser.add_argument('--metrics_only', action='store_true', help='Use existing labels without loading HarmBench')
    parser.add_argument('--grade_responses', action='store_true', help='Add response-quality grades with a chat judge')
    parser.add_argument('--judge_model', default='gpt-4o-mini', help='Chat judge for response-quality rubrics')
    parser.add_argument('--baseline_results_path', help='Same attack inputs without defense')
    parser.add_argument('--benign_results_path', help='Defended benign results with detector decisions')
    parser.add_argument('--benign_baseline_results_path', help='Same benign inputs without defense')
    parser.add_argument('--benign_behaviors_path', help='Original benign task CSV, including references when available')
    parser.add_argument('--benign_label_means_utility', action='store_true',
                        help='Explicitly interpret benign labels as task correctness')
    args = parser.parse_args(argv)
    if args.grade_responses and (args.benign_results_path or args.benign_baseline_results_path) and not args.benign_behaviors_path:
        parser.error('--grade_responses with benign results requires --benign_behaviors_path')
    return args


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', delete=False) as output:
            temporary = Path(output.name)
            json.dump(data, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_results(path, selected_model):
    if not path:
        return {}, {}
    with open(path, encoding='utf-8') as source:
        results = json.load(source)
    if not isinstance(results, dict):
        raise ValueError('Completions must be a JSON object keyed by BehaviorID')
    metadata_path = Path(str(path) + '.metadata.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
    from multimodalmodels.registry import resolve_model_id
    recorded_model = metadata.get('model_id') or metadata.get('model_name')
    if selected_model and recorded_model and resolve_model_id(selected_model) != resolve_model_id(recorded_model):
        raise ValueError(f'Target model differs from the metadata for {path}')
    model = recorded_model or selected_model
    for records in results.values():
        if not isinstance(records, list):
            raise ValueError('Each BehaviorID must map to a list of completion records')
        for record in records:
            if not isinstance(record, dict):
                raise ValueError('Each completion record must be an object')
            if model:
                if record.get('model') and resolve_model_id(record['model']) != resolve_model_id(model):
                    raise ValueError(f'Mixed target models in {path}')
                record['model'] = resolve_model_id(model)
            if metadata.get('revision'):
                meta = record.setdefault('meta', {})
                if not isinstance(meta, dict):
                    raise ValueError('Completion meta must be a JSON object')
                if meta.get('target_revision', metadata['revision']) != metadata['revision']:
                    raise ValueError(f'Target revision differs from the metadata for {path}')
                meta['target_revision'] = metadata['revision']
    return results, metadata


def _behavior_tables(records):
    originals = normalize_evaluation_behaviors(records)
    classifier_rows = {}
    for record, behavior in zip(records, originals.values()):
        classifier_rows[behavior.behavior_id] = {
            **record, 'BehaviorID': behavior.behavior_id, 'Behavior': behavior.instruction,
            'Tags': record.get('Tags') or '',
        }
    return originals, classifier_rows


def _load_behaviors(path):
    return _behavior_tables(read_csv(path))


def main(argv=None):
    args = parse_args(argv)
    paths = {'results': args.completions_path, 'baseline': args.baseline_results_path,
             'benign': args.benign_results_path, 'benign_baseline': args.benign_baseline_results_path}
    outputs = {'results': Path(args.save_path)}
    outputs.update({role: Path(f'{args.save_path}.{role}.json') for role in paths if role != 'results' and paths[role]})
    metrics_path = Path(args.metrics_path or f'{args.save_path}.metrics.json')
    all_outputs = [*outputs.values(), metrics_path, Path(f'{args.save_path}.metadata.json')]
    all_outputs += [Path(str(path) + '.metadata.json') for role, path in outputs.items() if role != 'results']
    all_outputs += [sidecar_path(path) for path in outputs.values()]
    if args.method_name:
        all_outputs.append(Path(args.asr_results_file))
    resolved_outputs = [path.resolve() for path in all_outputs]
    from data_paths import resolve_path
    inputs = {Path(path).resolve() for path in paths.values() if path}
    inputs.update(sidecar_path(path).resolve() for path in paths.values() if path)
    inputs.update(Path(resolve_path(path)).resolve()
                  for path in (args.behaviors_path, args.benign_behaviors_path) if path)
    if not args.dataset_name and not args.behaviors_path and not sidecar_path(args.completions_path).is_file():
        inputs.add(Path(resolve_path(DEFAULT_BEHAVIORS_PATH)).resolve())
    if len(resolved_outputs) != len(set(resolved_outputs)) or inputs.intersection(resolved_outputs):
        raise ValueError('Output paths must be distinct and must not overwrite input files')
    cohorts, metadata, snapshots = {}, {}, {}
    for role, path in paths.items():
        cohorts[role], metadata[role] = _load_results(path, args.model_name)
        # Verify against the unmodified JSON before adding target metadata.
        snapshots[role] = read_behavior_sidecar(path) if path else None
    if args.dataset_name:
        records = load_dataset(args.dataset_name)
        behavior_source = {'dataset_name': args.dataset_name}
    elif args.behaviors_path:
        records = read_csv(args.behaviors_path)
        behavior_source = {'dataset_name': 'custom', 'behaviors_path': args.behaviors_path}
    elif snapshots['results'] is not None:
        records = snapshots['results']['behaviors']
        behavior_source = snapshots['results']['source']
        if snapshots['baseline'] is not None:
            records = merge_original_behaviors(records, snapshots['baseline']['behaviors'])
    else:
        records = read_csv(DEFAULT_BEHAVIORS_PATH)
        behavior_source = {'dataset_name': 'HarmBench', 'behaviors_path': DEFAULT_BEHAVIORS_PATH}
    # An explicit selector must agree with the questions saved during generation.
    for role in ('results', 'baseline'):
        if snapshots[role] is not None:
            lookup = _behavior_tables(records)[0]
            saved = snapshots[role]['behaviors']
            if any(row['BehaviorID'] not in lookup for row in saved):
                raise ValueError('Selected dataset does not contain the saved behavior IDs')
            merge_original_behaviors(records, saved)
    originals, classifier_rows = _behavior_tables(records)
    benign_records = read_csv(args.benign_behaviors_path) if args.benign_behaviors_path else None
    benign_originals = _behavior_tables(benign_records)[0] if benign_records is not None else None
    options = dict(behaviors=originals, benign_behaviors=benign_originals,
                   model_name=args.model_name, method_name=args.method_name,
                   benign_label_means_utility=args.benign_label_means_utility)
    composition_context = None
    if snapshots['results'] is not None and snapshots['results'].get('composition'):
        from compositions.evaluation import evaluation_context
        composition_context = evaluation_context(cohorts['results'], snapshots['results'])
        options['behavior_ids'] = composition_context['allowed_ids']
    # Validate identities, original tasks and comparisons before loading a judge.
    compute_evaluation_metrics(**cohorts, **options)
    if not args.metrics_only:
        classifier = None
        for role in ('results', 'baseline'):
            for behavior_id, records in cohorts[role].items():
                behavior = classifier_rows[behavior_id]
                if composition_context is not None and behavior_id not in composition_context['allowed_ids']:
                    for record in records:
                        record.update(label=None, judge=None)
                    continue
                blocked, pending = [], []
                for record in records:
                    (blocked if record.get('blocked', record.get('meta', {}).get('blocked', False)) else pending).append(record)
                for record in blocked:
                    record.update(label=0, judge=args.cls_path)
                if pending:
                    if 'hash_check' in behavior['Tags'].split(', '):
                        current = compute_results_hashing(behavior, pending)
                        judge_name = 'copyright_hash'
                    else:
                        if classifier is None:
                            from multimodalmodels import HarmBenchMultiModalModel
                            classifier = HarmBenchMultiModalModel(args.cls_path)
                        current = compute_results_classifier(behavior, pending, classifier)
                        judge_name = args.cls_path
                    for record in current:
                        record['judge'] = judge_name
    if args.include_advbench_metric:
        for role in ('results', 'baseline'):
            for records in cohorts[role].values():
                compute_results_advbench(records, records)
    if args.grade_responses:
        grader, cache = ResponseGrader(args.judge_model), {}
        for role in paths:
            if paths[role]:
                kind = 'benign' if role.startswith('benign') else 'jailbreak'
                cohorts[role] = grade_evaluation_results(
                    cohorts[role], benign_originals if kind == 'benign' else originals,
                    kind=kind, grader=grader, model_name=args.model_name,
                    method_name=args.method_name, annotation_cache=cache,
                )
    report = compute_evaluation_metrics(**cohorts, **options)
    if composition_context is not None:
        from compositions.evaluation import add_recipe_reports
        report = add_recipe_reports(report, cohorts, options, composition_context)
    for role, path in outputs.items():
        _write_json(path, cohorts[role])
        role_meta = dict(metadata[role])
        models = {record.get('model') for records in cohorts[role].values() for record in records} - {None}
        if len(models) == 1:
            role_meta['model_id'] = models.pop()
        role_meta['evaluation'] = {
            'binary_judge': 'existing_labels' if args.metrics_only else args.cls_path,
            'quality_judge': args.judge_model if args.grade_responses else None,
            'source': str(paths[role]), 'metrics_path': str(metrics_path),
            'behavior_source': behavior_source if not role.startswith('benign') else args.benign_behaviors_path,
        }
        _write_json(str(path) + '.metadata.json', role_meta)
        snapshot = snapshots[role]
        if snapshot is None:
            rows = benign_records if role.startswith('benign') else records
            if rows is not None:
                # Normalize custom CSV identifiers through the same evaluator.
                normalized, classified = _behavior_tables(rows)
                saved_rows = [classified[key] for key in cohorts[role] if key in normalized]
                snapshot = {'source': behavior_source if not role.startswith('benign') else
                            {'dataset_name': 'custom', 'behaviors_path': args.benign_behaviors_path},
                            'method_name': args.method_name, 'behaviors': saved_rows}
        if snapshot is not None:
            write_behavior_sidecar(path, snapshot)
    _write_json(metrics_path, report)
    asr = report['jailbreak']['asr']['value']
    print(f'Average ASR: {asr if asr is not None else "unavailable"}')
    print(f'Saved 12-metric report to {metrics_path}')
    if args.method_name:
        save_asr_to_file(args.method_name, asr, args.asr_results_file)
    return report


if __name__ == '__main__':
    main()
