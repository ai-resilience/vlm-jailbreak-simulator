"""Resumable stage recipes using structural selection without response judging."""
from collections import Counter
from dataclasses import asdict
import fcntl
import json
import math
from pathlib import Path

from data_paths import REPO_ROOT
from .components import make_component
from .datasets import load_source_pool
from .engine import (ComposedArtifact, assign_source_split, composition_content_hash,
                     deterministic_stratified_select, execute_component, strict_image_hashes,
                     validate_composition_config)
from .export import export_cases
from .five_methods import atomic_json, digest, instruction_key, json_hash


def validate_case(case, behavior=None):
    meta = case['meta']
    if strict_image_hashes(case['images']) != meta['image_sha256s']:
        raise ValueError('Stage composition final image changed')
    if not case['text'].strip() or not case['images']:
        raise ValueError('Empty stage composition output')
    previous = None
    for index, record in enumerate(meta['stage_records']):
        if index != record['stage_index'] or not record['full_iterations_completed']:
            raise ValueError('Stage composition lineage is incomplete')
        if strict_image_hashes(record['input_images']) != record['input_image_sha256s']:
            raise ValueError('Stage composition intermediate input image changed')
        if strict_image_hashes(record['output_images']) != record['output_image_sha256s']:
            raise ValueError('Stage composition intermediate output image changed')
        if previous is not None and previous != record['input_content_hash']:
            raise ValueError('Stage composition lineage is discontinuous')
        previous = record['output_content_hash']
    content = composition_content_hash(case['text'], case['images'])
    if previous != content:
        raise ValueError('Stage composition final output differs from its lineage')
    if behavior and meta['source_record_sha256'] != json_hash(behavior.to_dict()):
        raise ValueError('Stage composition original record changed')
    return content


def generate(args, config, config_path):
    validate_composition_config(config)
    selected_ids = set(args.composition_recipes.split(',')) if args.composition_recipes else None
    available = {r['id'] for r in config['recipes']}
    if selected_ids and selected_ids - available:
        raise ValueError(f'Unknown composition recipes: {sorted(selected_ids - available)}')
    recipes = [r for r in config['recipes'] if selected_ids is None or r['id'] in selected_ids]
    if args.composition_per_recipe is not None and args.composition_per_recipe < 1:
        raise ValueError('composition_per_recipe must be positive')
    if args.composition_phase == 'review-preview':
        raise ValueError('review-preview is part of the five_methods workflow')
    # Resolve all requested adapters before loading a model or creating output artifacts.
    adapters = {}
    recipe_components = {}
    for recipe in recipes:
        parts = []
        for spec in recipe['stages']:
            spec = dict(spec)
            if spec.get('config') and not Path(spec['config']).is_absolute():
                candidate = config_path.parent / spec['config']
                spec['config'] = str((candidate if candidate.is_file() else REPO_ROOT / spec['config']).resolve())
            key = json_hash(spec)
            if key not in adapters:
                adapters[key] = make_component(spec)
            parts.append((key, adapters[key]))
        recipe_components[recipe['id']] = parts
    planning_phase = args.composition_validate_phase if args.composition_phase == 'validate' else args.composition_phase
    plan = {r['id']: {'final': args.composition_per_recipe or (min(3, r['final_count']) if planning_phase == 'preview' else r['final_count']),
                     'stages': [s['component'] for s in r['stages']]} for r in recipes}
    if args.composition_phase == 'dry-run':
        print(json.dumps(plan, indent=2)); return plan
    for component in adapters.values():
        component.preflight()
    phase = args.composition_validate_phase if args.composition_phase == 'validate' else args.composition_phase
    if phase == 'preflight':
        print(json.dumps({'status': 'passed', 'recipes': plan}, indent=2)); return plan
    root = Path(args.save_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {name: load_source_pool(name) for r in recipes for name in r['datasets']}
        generation_modules = ['generator.py', 'components.py', 'engine.py', 'records.py', 'datasets.py',
                              'export.py', 'paths.py', 'five_methods.py']
        code_paths = [Path(__file__).with_name(name) for name in generation_modules]
        code_paths += [REPO_ROOT / 'generate_test_cases.py', REPO_ROOT / 'benchmark_datasets.py',
                       REPO_ROOT / 'data_paths.py']
        for pattern in ['multi_attacks/figstep/*.py', 'multi_attacks/cs_dj/*.py', 'multi_attacks/JOOD/*.py',
                        'multi_attacks/mml/*.py', 'multi_attacks/jps/*.py', 'multimodalmodels/*.py']:
            code_paths.extend(REPO_ROOT.glob(pattern))
        code = {str(p.relative_to(REPO_ROOT)): digest(p) for p in code_paths}
        identity = json_hash({'config': config, 'plan': plan, 'phase': phase, 'code': code,
            'sources': {name: [b.to_dict() for b in values] for name, values in sources.items()},
            'components': {key: c.effective_config for key, c in adapters.items()}})
        directory = root / phase
        directory.mkdir(exist_ok=True)
        identity_path = directory / 'run_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text())['identity'] != identity:
            raise ValueError('Composition configuration/code/source changed; choose a new output directory')
        atomic_json(identity_path, {'identity': identity, 'config': config, 'plan': plan,
                                   'implementation_sha256': code})
        if args.composition_phase == 'validate':
            return validate_output(directory)
        all_cases, parents, all_contents, raw_counts, failures = [], {}, set(), {}, []
        prepared = set()
        for recipe in recipes:
            rid = recipe['id']
            count = plan[rid]['final']
            raw_count = math.ceil(count * float(config.get('candidate_multiplier', 1.2)))
            pool = [b for name in recipe['datasets'] for b in sources[name]
                    if b.eval_type.value in ('harmful', 'copyright')]
            components = recipe_components[rid]
            pool = [b for b in pool if not any(c.evaluation_exclusion_reason(b) for _, c in components)]
            candidates = deterministic_stratified_select(pool, raw_count, seed=int(config.get('seed', 0)))
            if len(candidates) < count:
                raise ValueError(f'Insufficient eligible original questions for {rid}: need {count}, have {len(candidates)}')
            chosen, generated = [], 0
            for behavior in candidates:
                candidate_root = directory / 'cases' / rid / behavior.behavior_id
                cached = candidate_root / 'case.json'
                if cached.exists():
                    saved = json.loads(cached.read_text()); case = saved['case']
                    if saved['sha256'] != json_hash(case) or case['meta']['identity'] != identity:
                        raise ValueError('Composition checkpoint was changed')
                    content = validate_case(case, behavior)
                else:
                    artifact = ComposedArtifact(behavior.behavior_id, behavior.instruction, list(behavior.images))
                    try:
                        for index, (key, component) in enumerate(components):
                            if key not in prepared:
                                component.prepare(str(root / '_prepared' / key))
                                prepared.add(key)
                            artifact = execute_component(component, artifact, behavior,
                                str(candidate_root / f'stage_{index:02d}'), stage_index=index)
                    except Exception as error:
                        # Keep a failed candidate visible; errors never count as successful generation.
                        failures.append({'recipe_id': rid, 'behavior_id': behavior.behavior_id,
                                         'error': f'{type(error).__name__}: {error}'})
                        atomic_json(directory / 'failures.json', failures)
                        continue
                    case = {'test_case_id': f'{rid}:{behavior.behavior_id}', 'behavior_id': behavior.behavior_id,
                        'attack': '+'.join(c.source_attack for _, c in components), 'attempt': 0,
                        'text': artifact.text, 'images': list(artifact.images),
                        'meta': {'identity': identity, 'recipe_id': rid,
                                 'methods': [c.source_attack for _, c in components],
                                 'source_behavior_id': behavior.behavior_id, 'source_dataset': behavior.dataset,
                                 'source_eval_type': behavior.eval_type.value,
                                 'source_record_sha256': json_hash(behavior.to_dict()),
                                 'source_instruction_hash': json_hash(instruction_key(behavior.instruction)),
                                 'split': assign_source_split(instruction_key(behavior.instruction), seed=int(config.get('seed', 0)),
                                     train=config.get('split', {}).get('train', .7), validation=config.get('split', {}).get('validation', .1)),
                                 'stage_records': [asdict(r) for r in artifact.stage_records],
                                 'image_sha256s': strict_image_hashes(artifact.images),
                                 'algorithm_fidelity': 'adapted_composition', 'paper_reproduction': False}}
                    content = validate_case(case, behavior)
                    atomic_json(cached, {'case': case, 'sha256': json_hash(case)})
                generated += 1
                if len(chosen) < count and content not in all_contents:
                    chosen.append(case); all_contents.add(content)
                    parents[behavior.behavior_id] = behavior
            if len(chosen) != count:
                raise ValueError(f'{rid} produced {len(chosen)}/{count} distinct valid cases; see failures.json')
            all_cases.extend(chosen); raw_counts[rid] = generated
        atomic_json(directory / 'native_cases.json', all_cases)
        snapshot = export_cases(all_cases, list(parents.values()), directory / 'test_cases.json',
                                identity=identity, config=config, mode='stages')
        atomic_json(directory / 'manifest.json', {'identity': identity, 'status': 'complete',
            'n_cases': len(all_cases), 'counts': dict(Counter(c['meta']['recipe_id'] for c in all_cases)),
            'raw_valid_counts': raw_counts, 'failed_candidates': failures,
            'test_cases_sha256': digest(directory / 'test_cases.json'),
            'native_cases_sha256': digest(directory / 'native_cases.json'),
            'algorithm_fidelity': 'adapted_composition', 'paper_reproduction': False,
            'target_model_queried': False})
        validate_output(directory)
        return snapshot


def validate_output(directory):
    from benchmark_datasets import read_behavior_sidecar
    from .validation import validate_input
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name in ('native_cases', 'test_cases'):
        if digest(directory / f'{name}.json') != manifest[f'{name}_sha256']:
            raise ValueError('Composition output checksum mismatch')
    cases = json.loads((directory / 'native_cases.json').read_text())
    contents = [validate_case(c) for c in cases]
    if len(contents) != len(set(contents)) or len(cases) != manifest['n_cases']:
        raise ValueError('Duplicate or missing composed inputs')
    path = directory / 'test_cases.json'
    payload = json.loads(path.read_text())
    result = validate_input(payload, read_behavior_sidecar(path, verify_artifacts=True))
    if result['counts'] != manifest['counts'] or result['identity'] != manifest['identity']:
        raise ValueError('Composition output differs from its manifest')
    result.update(status='passed')
    atomic_json(directory / 'qc.json', result)
    return result
