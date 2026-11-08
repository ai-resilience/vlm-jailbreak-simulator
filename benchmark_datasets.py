"""Benchmark loaders, stable question selection and saved evaluation inputs.

The loaders retain the reference simulator's IDs, question order, original
instructions and image associations. They do not import models or modify data.
"""

import copy
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile

from data_paths import REPO_ROOT, resolve_path


DATASET_NAMES = ('AdvBench', 'AdvBenchM', 'HADES', 'MM-SafetyBench', 'SafeBench', 'JOOD')
DEFAULT_BEHAVIORS_PATH = 'data/original/HarmBench/behaviors/harmbench_behaviors_text_all.csv'


def portable_path(path):
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def read_csv(path):
    with open(resolve_path(path), newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def _jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _row(behavior_id, instruction, dataset, *, images=(), meta=None, **fields):
    return {'BehaviorID': behavior_id, 'Behavior': instruction, 'dataset': dataset,
            'Tags': '', 'images': list(images), 'meta': meta or {},
            'eval_type': 'harmful', **fields}


def _advbench(root):
    return [_row(f'advbench_{i:03d}', row['prompt'], 'AdvBench', target=row['target'], meta={'index': i})
            for i, row in enumerate(read_csv(root / 'harmful_behaviors.csv'))]


def _advbenchm(root):
    pools = {}
    for row in _jsonl(root / 'images.jsonl'):
        key = (row['category'], int(row['class_id']))
        pools.setdefault(key, []).append(portable_path(root / row['path']))
    rows = []
    for row in _jsonl(root / 'instructions.jsonl'):
        key = (row['category'], int(row['class_id']))
        if key not in pools or row['image_association'] != 'category_pool_not_one_to_one':
            raise ValueError(f'Invalid AdvBenchM image association: {key}')
        rows.append(_row(
            row['id'].replace('/', '_'), row['instruction'], 'AdvBenchM',
            target=row.get('target'), category=row['category'], images=pools[key],
            meta={'upstream_id': row['id'], 'class_id': key[1], 'source_csv': row['source_csv'],
                  'source_record_index': row['source_record_index'],
                  'image_pool': portable_path(root / row['image_pool']),
                  'image_association': row['image_association']},
        ))
    return rows


def _hades(root):
    rows = []
    for row in read_csv(root / 'hades.csv'):
        images = []
        for step in range(6):
            image = row[f'image_step{step}']
            for prefix in ('data/benchmarks/HADES/', 'data/HADES/'):
                if image.startswith(prefix):
                    image = portable_path(root / image[len(prefix):])
                    break
            images.append(image)
        rows.append(_row(
            row['hades_id'], row['instruction'], 'HADES', category=row['scenario'],
            keyword=row['keywords'], keyword_type=row['category'], images=images,
            meta={'official_prompt': row['official_prompt']},
        ))
    return rows


def _mm_safetybench(root):
    eval_types = {'08-Political_Lobbying': 'opinion', '13-Gov_Decision': 'opinion',
                  '10-Legal_Opinion': 'advice', '11-Financial_Advice': 'advice',
                  '12-Health_Consultation': 'advice'}
    rows = []
    for directory in sorted(root.iterdir()):
        path = directory / f'{directory.name}.json'
        if not directory.is_dir() or not path.is_file():
            continue
        questions = json.loads(path.read_text(encoding='utf-8'))
        number, _, category = directory.name.partition('-')
        for key in sorted(questions, key=int):
            question = questions[key]
            by_type = {name: portable_path(directory / name / f'{key}.jpg')
                       for name in ('SD', 'SD_TYPO', 'TYPO')}
            rows.append(_row(
                f'mmsb_{number}_{int(key):03d}', question['Question'], 'MM-SafetyBench',
                category=category, eval_type=eval_types.get(directory.name, 'harmful'),
                keyword=question['Key Phrase'], keyword_type=question['Phrase Type'],
                images=list(by_type.values()), meta={
                    'scenario_id': int(number), 'index': int(key), 'images_by_type': by_type,
                    'rephrased_question': question['Rephrased Question'],
                    'rephrased_question_sd': question['Rephrased Question(SD)'],
                    'changed_question': question['Changed Question'],
                },
            ))
    return rows


def _safebench(root):
    rows = []
    for row in read_csv(root / 'safebench.csv'):
        category_id, task_id = int(row['category_id']), int(row['task_id'])
        rows.append(_row(
            f'safebench_{category_id:02d}_{task_id:02d}', row['question'], 'SafeBench',
            statement=row['instruction'], category=row['category_name'],
            eval_type='advice' if category_id in (8, 9, 10) else 'harmful',
            meta={'category_id': category_id, 'task_id': task_id, 'source': row['dataset']},
        ))
    return rows


def _jood(root):
    rows = []
    for path in sorted((root / 'prompts/eval_all_instructions').glob('*.json')):
        scenario = path.stem
        variants = [json.loads((root / 'prompts' / kind / path.name).read_text(encoding='utf-8'))['instructions']
                    for kind in ('eval_all_instructions', 'all_instructions', 'all_instructions_harmful_annotated')]
        if not all(isinstance(items, list) for items in variants) or len({len(items) for items in variants}) != 1:
            raise ValueError(f'Misaligned JOOD prompt variants: {scenario}')
        images = sorted(p for p in (root / 'images/harmful' / scenario).iterdir()
                        if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        if not images:
            raise FileNotFoundError(f'No JOOD representative image for {scenario}')
        for index, (original, multimodal, annotated) in enumerate(zip(*variants)):
            if any(not isinstance(value, str) or not value.strip() for value in (original, multimodal, annotated)):
                raise ValueError(f'Empty JOOD prompt: {scenario}:{index}')
            match = re.search(r'<HARMFUL>(.*?)</HARMFUL>', annotated, re.DOTALL)
            rows.append(_row(
                f'advbenchm_{scenario}_{index:03d}', original, 'JOOD', category=scenario,
                category_orig=scenario, keyword=match.group(1) if match else None,
                images=[portable_path(images[0])],
                meta={'scenario': scenario, 'index': index, 'multimodal_prompt': multimodal,
                      'annotated_prompt': annotated},
            ))
    return rows


_LOADERS = {'AdvBench': ('benchmarks/advbench', _advbench), 'AdvBenchM': ('benchmarks/AdvBenchM', _advbenchm),
            'HADES': ('benchmarks/HADES', _hades), 'MM-SafetyBench': ('benchmarks/MM-SafetyBench', _mm_safetybench),
            'SafeBench': ('benchmarks/SafeBench', _safebench), 'JOOD': ('method_assets/JOOD', _jood)}


def validate_behaviors(rows):
    ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each saved behavior must be an object')
        behavior_id, instruction = row.get('BehaviorID'), row.get('Behavior')
        if (not isinstance(behavior_id, str) or not behavior_id or behavior_id in ('.', '..')
                or any(c in behavior_id for c in ('/', '\\', '\x00'))):
            raise ValueError(f'Invalid BehaviorID: {behavior_id!r}')
        if behavior_id in ids:
            raise ValueError(f'Duplicate BehaviorID: {behavior_id}')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f'Original instruction missing: {behavior_id}')
        ids.add(behavior_id)
    return ids


def load_dataset(name):
    if name not in _LOADERS:
        raise ValueError(f'Unknown dataset: {name}. Choose one of {DATASET_NAMES}')
    directory, loader = _LOADERS[name]
    rows = loader(Path(resolve_path(f'data/{directory}')))
    if not rows:
        raise ValueError(f'Dataset contains no questions: {name}')
    validate_behaviors(rows)
    return rows


def select_behaviors(rows, start=0, end=None, ids_subset=''):
    """Apply a question slice, or an ID subset, without changing question IDs."""
    validate_behaviors(rows)
    start = 0 if start is None else start
    if start < 0 or end is not None and (end < 0 or end < start):
        raise ValueError('Require 0 <= behavior_start_idx <= behavior_end_idx')
    if ids_subset:
        path = Path(ids_subset).expanduser()
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False  # A long comma-separated ID list is not a filename.
        requested = path.read_text(encoding='utf-8').splitlines() if is_file else ids_subset.split(',')
        requested = {item.strip() for item in requested if item.strip()}
        missing = requested - {row['BehaviorID'] for row in rows}
        if missing:
            raise ValueError(f'Unknown behavior IDs: {sorted(missing)}')
        selected = [row for row in rows if row['BehaviorID'] in requested]
    else:
        selected = rows[start:end]
    if not selected:
        raise ValueError('No behaviors selected; check the indices or IDs')
    selection = {'mode': 'ids' if ids_subset else 'slice', 'start': start, 'end': end,
                 'source_count': len(rows), 'selected_count': len(selected),
                 'behavior_ids': [row['BehaviorID'] for row in selected]}
    return selected, selection


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def sidecar_path(path):
    return Path(str(path) + '.behaviors.json')


def _validate_snapshot(snapshot, payload):
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('behaviors'), list):
        raise ValueError('Invalid behavior snapshot')
    ids = validate_behaviors(snapshot['behaviors'])
    if not isinstance(payload, dict) or ids != set(payload):
        raise ValueError('Saved behaviors do not match the artifact BehaviorIDs')


def read_behavior_sidecar(path, payload=None, *, verify_artifacts=False):
    """Read original questions bound to the exact test-case/completion JSON.

    For test-case inputs, verify_artifacts also checks generated image bytes.
    """
    companion = sidecar_path(path)
    if not companion.is_file():
        return None
    data = json.loads(companion.read_text(encoding='utf-8'))
    if data.get('schema_version') != 1:
        raise ValueError(f'Unsupported behavior snapshot schema: {companion}')
    snapshot = data.get('snapshot')
    if data.get('snapshot_sha256') != digest(snapshot):
        raise ValueError(f'Behavior snapshot digest differs: {companion}')
    if payload is None:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('payload_sha256') != digest(payload):
        raise ValueError(f'Stale behavior snapshot for changed artifact: {path}')
    _validate_snapshot(snapshot, payload)
    if verify_artifacts:
        for reference, expected in snapshot.get('generated_images', {}).items():
            image = Path(reference)
            if not image.is_absolute():
                image = Path(path).resolve().parent / image
            if not image.is_file():
                raise ValueError(f'Generated image is missing: {reference}')
            actual = hashlib.sha256()
            with image.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    actual.update(block)
            if actual.hexdigest() != expected:
                raise ValueError(f'Generated image changed: {reference}')
    return snapshot


def write_behavior_sidecar(path, snapshot):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    _validate_snapshot(snapshot, payload)
    data = {'schema_version': 1, 'snapshot': snapshot,
            'snapshot_sha256': digest(snapshot), 'payload_sha256': digest(payload)}
    companion = sidecar_path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=companion.parent,
                                         prefix=f'.{companion.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, companion)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def merge_original_behaviors(left, right):
    """Merge lookup records, rejecting a reused ID with a changed question/context."""
    from evaluation.compat import normalize_evaluation_behaviors
    from evaluation.response_grading import reference_context

    def canonical(rows):
        normalized = normalize_evaluation_behaviors(rows)
        return [{**row, 'BehaviorID': behavior.behavior_id, 'Behavior': behavior.instruction}
                for row, behavior in zip(rows, normalized.values())]

    left, right = canonical(left), canonical(right)
    result = {row['BehaviorID']: copy.deepcopy(row) for row in left}
    for row in right:
        previous = result.get(row['BehaviorID'])
        if previous is not None:
            if previous['Behavior'] != row['Behavior']:
                raise ValueError(f'Original question differs for {row["BehaviorID"]}')
            a, b = (next(iter(normalize_evaluation_behaviors([item]).values())) for item in (previous, row))
            if reference_context(a) != reference_context(b):
                raise ValueError(f'Original reference/context differs for {row["BehaviorID"]}')
        else:
            result[row['BehaviorID']] = copy.deepcopy(row)
    return list(result.values())
