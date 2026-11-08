"""Export native composition cases into the shared BehaviorID JSON contract."""
import copy
from pathlib import Path

from benchmark_datasets import digest, write_behavior_sidecar
from data_paths import resolve_path
from .datasets import scoring_row
from .five_methods import atomic_json, digest as file_digest


def export_cases(cases, behaviors, path, *, identity, config, mode):
    grouped, images, ids = {}, {}, set()
    for original in cases:
        case = copy.deepcopy(original)
        if case['test_case_id'] in ids:
            raise ValueError('Duplicate composition test_case_id')
        ids.add(case['test_case_id'])
        behavior_id = case.pop('behavior_id')
        group = grouped.setdefault(behavior_id, [])
        case['attempt'] = len(group)
        for reference, expected in zip(case['images'], case['meta']['image_sha256s']):
            absolute = str(Path(resolve_path(reference)).resolve())
            if file_digest(absolute) != expected:
                raise ValueError(f'Composition image changed before export: {reference}')
            images[absolute] = expected
        case['meta']['composition_mode'] = mode
        case['meta']['composition_identity'] = identity
        group.append(case)
    lookup = {b.behavior_id: b for b in behaviors}
    snapshot = {'source': {'dataset_name': 'composed', 'datasets': sorted({lookup[b].dataset for b in grouped})},
        'selection': {'selected_count': len(grouped), 'n_cases': len(cases)},
        'method_name': 'Composed', 'method_config_sha256': digest(config),
        'behaviors': [scoring_row(lookup[b]) for b in grouped], 'generated_images': images,
        'composition': {'schema_version': 1, 'mode': mode, 'identity': identity,
                        'algorithm_fidelity': 'adapted_composition', 'paper_reproduction': False,
                        'config': config}}
    atomic_json(path, grouped)
    write_behavior_sidecar(path, snapshot)
    return snapshot
