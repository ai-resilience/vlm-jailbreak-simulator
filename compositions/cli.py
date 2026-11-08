"""Composition dispatch for generate_test_cases.py."""
import json
from pathlib import Path

import yaml

from data_paths import resolve_path
from .export import export_cases
from .records import Behavior


def generate(args):
    path = Path(resolve_path(args.composition_config))
    config = yaml.safe_load(path.read_text())
    mode = config.get('mode', 'stages')
    if mode == 'stages':
        from .generator import generate as stages
        return stages(args, config, path)
    if mode != 'five_methods':
        raise ValueError(f'Unknown composition mode: {mode}')
    if args.composition_recipes or args.composition_per_recipe is not None:
        raise ValueError('The five-method production plan preserves all 18 recipes; use a stages config for subsets')
    from .five_methods import DatasetBuilder, load_config
    from .five_runner import main, validate_phase
    config, recipes = load_config(path)
    root = Path(args.save_dir).resolve()
    phase = args.composition_phase
    if phase == 'dry-run':
        result = {'mode': mode, 'n_recipes': len(recipes), 'n_cases': sum(r.count for r in recipes),
                  'preview_cases': 3 * len(recipes), 'recipes': [r.__dict__ for r in recipes]}
        print(json.dumps(result, indent=2))
        return result
    if phase == 'preflight':
        builder = DatasetBuilder(config, recipes, root)
        result = {'status': 'passed', 'identity': builder.identity, 'source_selection': builder.source_selection}
        print(json.dumps(result, indent=2))
        return result
    command = [phase, '--config', str(path), '--output', str(root), '--validate-phase', args.composition_validate_phase]
    if args.composition_review_note:
        command += ['--review-note', args.composition_review_note]
    main(command)
    completed_phase = args.composition_validate_phase if phase == 'validate' else 'preview' if phase == 'review-preview' else phase
    directory = root / completed_phase
    cases = [json.loads(line) for line in (directory / 'test_cases.jsonl').read_text().splitlines() if line]
    parents = [Behavior.from_dict(b) for b in json.loads((root / 'parents.json').read_text())]
    identity = json.loads((root / 'run_identity.json').read_text())['identity']
    snapshot = export_cases(cases, parents, directory / 'test_cases.json', identity=identity, config=config, mode=mode)
    print(f'Saved {len(cases)} composition inputs to {directory / "test_cases.json"}')
    return snapshot
