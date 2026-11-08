"""Generate attack inputs from a named benchmark or a custom behavior CSV."""

import argparse
import copy
from pathlib import Path

import yaml

from attacks import get_method_class, init_method
from benchmark_datasets import (
    DATASET_NAMES, DEFAULT_BEHAVIORS_PATH, digest, load_dataset, portable_path,
    read_csv, select_behaviors, write_behavior_sidecar,
)
from data_paths import resolve_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--method_name')
    mode.add_argument('--composition_config', help='Composition YAML instead of a single attack')
    parser.add_argument('--composition_phase', choices=['dry-run', 'preflight', 'preview', 'review-preview', 'full', 'validate'], default='preview')
    parser.add_argument('--composition_recipes', help='Comma-separated stage-recipe IDs')
    parser.add_argument('--composition_per_recipe', type=int, help='Stage-recipe sample-count override')
    parser.add_argument('--composition_validate_phase', choices=['preview', 'full'], default='preview')
    parser.add_argument('--composition_review_note', help='Observations from visual preview inspection')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--dataset_name', choices=DATASET_NAMES, help='Benchmark name or the JOOD method-specific question set')
    source.add_argument('--behaviors_path', help='Custom behavior CSV instead of a named question set')
    parser.add_argument('--save_dir', default='./test_cases')
    parser.add_argument('--config', help='Method YAML; default: configs/method_configs/METHOD.yaml')
    parser.add_argument('--behavior_start_idx', type=int, default=0, help='First question index, inclusive')
    parser.add_argument('--behavior_end_idx', type=int, help='Last question index, exclusive; omit for the end')
    parser.add_argument('--behavior_ids_subset', default='',
                        help='Comma-separated IDs or newline-separated ID file; takes precedence over the slice')
    parser.add_argument('--run_id', help='Accepted for compatibility')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    if args.composition_config:
        if (args.config or args.dataset_name or args.behaviors_path or args.behavior_start_idx
                or args.behavior_end_idx is not None or args.behavior_ids_subset or args.overwrite or args.run_id):
            parser.error('Composition YAML controls sources/counts; use a separate output directory and composition options')
    else:
        if args.composition_recipes or args.composition_per_recipe is not None or args.composition_review_note or args.composition_phase != 'preview':
            parser.error('Composition options require --composition_config')
        args.method_name = args.method_name or 'FigStep'
    if args.behavior_start_idx < 0 or args.behavior_end_idx is not None and args.behavior_end_idx < args.behavior_start_idx:
        parser.error('Require 0 <= behavior_start_idx <= behavior_end_idx')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.composition_config:
        from compositions import generate
        return generate(args)
    config_file = args.config or f'configs/method_configs/{args.method_name}.yaml'
    with open(config_file, encoding='utf-8') as stream:
        method_config = yaml.safe_load(stream) or {}
    method_class = get_method_class(args.method_name)
    dataset_name = args.dataset_name
    if not dataset_name and not args.behaviors_path:
        dataset_name = getattr(method_class, 'default_dataset', None)
    if dataset_name:
        behaviors = load_dataset(dataset_name)
        source = {'dataset_name': dataset_name}
    else:
        path = args.behaviors_path or DEFAULT_BEHAVIORS_PATH
        behaviors = read_csv(path)
        source = {'dataset_name': 'custom' if args.behaviors_path else 'HarmBench',
                  'behaviors_path': portable_path(resolve_path(path))}
    normalize = getattr(method_class, 'normalize_behaviors', None)
    if normalize is not None:
        behaviors = normalize(behaviors)
    selected, selection = select_behaviors(
        behaviors, args.behavior_start_idx, args.behavior_end_idx, args.behavior_ids_subset)
    method = init_method(method_class, method_config)
    exclusions = []
    filter_evaluation = getattr(method, 'filter_evaluation_behaviors', None)
    if filter_evaluation is not None:
        requested_count = len(selected)
        selected, exclusions = filter_evaluation(selected)
        selection = {**selection, 'requested_count': requested_count,
                     'selected_count': len(selected),
                     'behavior_ids': [row['BehaviorID'] for row in selected]}
    snapshot = {'source': source, 'selection': selection, 'method_name': args.method_name,
                'method_config_sha256': digest(method_config), 'behaviors': copy.deepcopy(selected)}
    if filter_evaluation is not None:
        snapshot['evaluation_exclusions'] = exclusions
    save_dir = Path(args.save_dir).resolve()
    print(f'Generating {len(selected)} questions from {source["dataset_name"]}')
    method.generate_test_cases(behaviors=selected, save_dir=str(save_dir), verbose=args.verbose)
    fingerprints = getattr(method, 'artifact_fingerprints', None)
    if fingerprints is not None:
        snapshot['generated_images'] = fingerprints()
    filenames = {'Visual_Adv': 'Visual_Adv.json', 'Hades': 'hades_test_cases.json'}
    cases_path = save_dir / filenames.get(args.method_name, f'{args.method_name}_test_cases.json')
    write_behavior_sidecar(cases_path, snapshot)
    print(f'Saved test cases and original questions to {save_dir}')
    return snapshot


if __name__ == '__main__':
    main()
