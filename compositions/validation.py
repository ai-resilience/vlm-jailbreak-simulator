"""Validate the shared input contract before target or judge model loading."""
from collections import Counter
from pathlib import Path

from benchmark_datasets import digest
from multimodalmodels.inference import case_groups
from data_paths import resolve_path


def validate_input(payload, snapshot):
    if not snapshot or 'composition' not in snapshot:
        return None
    declaration = snapshot['composition']
    if declaration.get('schema_version') != 1:
        raise ValueError('Unsupported composition schema')
    if (declaration.get('algorithm_fidelity') != 'adapted_composition'
            or declaration.get('paper_reproduction') is not False):
        raise ValueError('Composition provenance is invalid')
    ids, counts, splits = set(), Counter(), {}
    parents = {row['BehaviorID']: row for row in snapshot['behaviors']}
    for behavior_id, cases in case_groups(payload).items():
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError('Composition cases must preserve structured metadata')
            meta = case.get('meta', {})
            case_id = case.get('test_case_id')
            if not isinstance(case_id, str) or not case_id or case_id in ids:
                raise ValueError('Duplicate or missing composition test_case_id')
            ids.add(case_id)
            if (meta.get('composition_identity') != declaration['identity']
                    or meta.get('source_behavior_id') != behavior_id
                    or meta.get('source_eval_type') != parents[behavior_id].get('eval_type', 'harmful')):
                raise ValueError('Composition parent or run identity differs')
            if len(case.get('images', [])) != len(meta.get('image_sha256s', [])):
                raise ValueError('Composition image/hash count mismatch')
            for reference, expected in zip(case['images'], meta['image_sha256s']):
                absolute = str(Path(resolve_path(reference)).resolve())
                if snapshot.get('generated_images', {}).get(absolute) != expected:
                    raise ValueError('Composition image hash differs from the original snapshot')
            if meta.get('algorithm_fidelity') != 'adapted_composition' or meta.get('paper_reproduction') is not False:
                raise ValueError('Composition case has invalid fidelity')
            if not isinstance(meta.get('recipe_id'), str) or not meta['recipe_id']:
                raise ValueError('Missing composition recipe ID')
            key, split = meta['source_instruction_hash'], meta['split']
            if key in splits and splits[key] != split:
                raise ValueError('Same source instruction crosses composition splits')
            splits[key] = split
            counts[meta['recipe_id']] += 1
    return {'identity': declaration['identity'], 'mode': declaration['mode'],
            'schema_version': 1, 'counts': dict(counts), 'n_cases': len(ids),
            'snapshot_sha256': digest(snapshot)}
