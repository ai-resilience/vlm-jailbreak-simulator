"""Adapt the existing BehaviorID-keyed JSON format to the twelve metrics."""

import copy
from collections import Counter
from collections.abc import Mapping

from .report import build_evaluation_report
from .response_grading import ResponseGrader, get_annotation, json_digest, reference_context, text_digest
from .schema import Behavior, Completion


def normalize_evaluation_behaviors(behaviors):
    """Accept HarmBench CSV rows, normalized rows, or the native SafeBench CSV."""
    normalized = {}
    entries = behaviors.items() if isinstance(behaviors, Mapping) else ((None, b) for b in behaviors)
    for key, value in entries:
        if isinstance(value, Behavior):
            behavior = copy.deepcopy(value)
        else:
            value = dict(value)
            behavior_id = value.get('BehaviorID') or value.get('behavior_id') or key
            if behavior_id is None and all(k in value for k in ('category_id', 'task_id')):
                behavior_id = f"safebench_{int(value['category_id']):02d}_{int(value['task_id']):02d}"
            instruction = value.get('Behavior') or value.get('question') or value.get('instruction')
            meta = dict(value.get('meta') or {})
            for field in ('reference_answer', 'reference_answers', 'expected_answer', 'context'):
                if value.get(field) not in (None, ''):
                    meta[field] = value[field]
            for field in ('ContextString', 'RedactedImageDescription'):
                if value.get(field):
                    meta['context'] = value[field]
            behavior = Behavior(str(behavior_id or ''), value.get('dataset', 'provided'), instruction, meta)
        if not behavior.behavior_id or not isinstance(behavior.instruction, str) or not behavior.instruction.strip():
            raise ValueError('Each behavior needs an ID and an original instruction')
        if key is not None and str(key) != behavior.behavior_id:
            raise ValueError(f'Behavior dictionary key differs from its ID: {key}')
        if behavior.behavior_id in normalized:
            raise ValueError(f'Duplicate behavior ID: {behavior.behavior_id}')
        normalized[behavior.behavior_id] = behavior
    return normalized


def _model_id(name):
    if not isinstance(name, str) or not name:
        raise ValueError('Target model must be a nonempty string')
    from multimodalmodels.registry import resolve_model_id
    return resolve_model_id(name)


def _records(results, *, model_name=None, method_name=None, judge_name=None):
    if not isinstance(results, Mapping):
        raise ValueError('Results must be a JSON object keyed by BehaviorID')
    rows, locations = [], []
    for behavior_id, attempts in results.items():
        if not isinstance(behavior_id, str) or not behavior_id or not isinstance(attempts, list):
            raise ValueError('Each BehaviorID must map to a list of completion records')
        occurrences = Counter()
        for index, record in enumerate(attempts):
            if not isinstance(record, dict) or not isinstance(record.get('generation'), str):
                raise ValueError(f'Completion needs a generation string: {behavior_id}:{index}')
            meta = copy.deepcopy(record.get('meta', {}))
            if not isinstance(meta, dict):
                raise ValueError('Completion meta must be a JSON object')
            for key in ('evaluation', 'detected_adversarial', 'unsafe_detected', 'defense',
                        'refused', 'is_refusal', 'utility_correct', 'task_correct', 'target_revision'):
                if key in record:
                    if key in meta and meta[key] != record[key]:
                        raise ValueError(f'Conflicting completion metadata: {key}')
                    meta[key] = copy.deepcopy(record[key])
            case_digest = json_digest(record['test_case']) if 'test_case' in record else None
            if case_digest is not None:
                if meta.get('test_case_sha256', case_digest) != case_digest:
                    raise ValueError('Stored test case fingerprint differs from its input')
                meta['test_case_sha256'] = case_digest
            case_id = record.get('test_case_id')
            if case_id is None:
                if case_digest is None:
                    raise ValueError('Completion needs test_case or an explicit test_case_id')
                ordinal = occurrences[case_digest]
                occurrences[case_digest] += 1
                case_id = 'json:' + json_digest([behavior_id, case_digest, ordinal])
            if not isinstance(case_id, str) or not case_id:
                raise ValueError('test_case_id must be a nonempty string')
            attempt = record.get('attempt', index)
            if type(attempt) is not int or attempt < 0:
                raise ValueError('attempt must be a nonnegative integer')
            blocked = record.get('blocked', meta.get('blocked', False))
            if type(blocked) is not bool:
                raise ValueError('blocked must be boolean')
            label = record.get('label')
            if label is not None and (type(label) is not int or label not in (-1, 0, 1)):
                raise ValueError('label must be -1, 0, 1 or null')
            label = None if label == -1 else label
            if blocked and label is None:
                label = 0
            if blocked and label == 1:
                raise ValueError('A blocked completion cannot have a successful attack label')
            model = _model_id(record.get('model') or model_name or 'unspecified')
            if model_name and model != _model_id(model_name):
                raise ValueError('Completion target model differs from the selected model')
            judge = record.get('judge') or (judge_name or 'provided_label' if label is not None else None)
            if judge is not None and not isinstance(judge, str):
                raise ValueError('judge must be a string or null')
            rows.append(Completion(case_id, behavior_id, record.get('attack') or method_name or 'provided',
                                   model, record['generation'], attempt, blocked, label, judge, meta))
            locations.append((behavior_id, index))
    return rows, locations


def _check_pair_inputs(left, right):
    lookup = {row.test_case_id: row for row in right}
    for row in left:
        other = lookup.get(row.test_case_id)
        if other is None:
            continue
        for key in ('test_case_sha256', 'target_revision'):
            a, b = row.meta.get(key), other.meta.get(key)
            if a is not None and b is not None and a != b:
                raise ValueError(f'Paired {key} differs: {row.test_case_id}')


def compute_evaluation_metrics(results, *, baseline=None, benign=None, benign_baseline=None,
                               behaviors=None, benign_behaviors=None, model_name=None,
                               method_name=None, judge_name=None, behavior_ids=None,
                               benign_label_means_utility=False):
    """Return six jailbreak and six defense metrics for existing result JSON.

    Missing grades/paired data produce null values with coverage and a reason.
    ASR uses one vote per BehaviorID, with success if any scored attempt succeeds.
    Detection decisions must be recorded explicitly; a failed attack is not a detection.
    """
    options = dict(model_name=model_name, method_name=method_name, judge_name=judge_name)
    cohorts = [_records(data if data is not None else {}, **options)[0]
               for data in (results, baseline, benign, benign_baseline)]
    attack, base, normal, normal_base = cohorts
    if any(cohorts[1:]) and any(row.model == 'unspecified' for rows in cohorts for row in rows):
        raise ValueError('Comparisons require the target model in metadata or model_name')
    _check_pair_inputs(base, attack)
    _check_pair_inputs(normal_base, normal)
    originals = normalize_evaluation_behaviors(behaviors) if behaviors is not None else None
    if originals is not None:
        missing = {row.behavior_id for row in [*attack, *base]} - originals.keys()
        if missing:
            raise ValueError(f'Original behaviors missing: {sorted(missing)}')
        for row in [*attack, *base]:
            annotation = get_annotation(row, 'jailbreak')
            if annotation and annotation.get('reference_sha256') != json_digest(reference_context(originals[row.behavior_id])):
                raise ValueError(f'Annotation differs from the original task context: {row.behavior_id}')
    if benign_behaviors is not None:
        normal_originals = normalize_evaluation_behaviors(benign_behaviors)
        for row in [*normal, *normal_base]:
            if row.behavior_id not in normal_originals:
                raise ValueError(f'Original benign behavior missing: {row.behavior_id}')
            annotation = get_annotation(row, 'benign')
            behavior = normal_originals[row.behavior_id]
            if annotation and (annotation.get('instruction_sha256') != text_digest(behavior.instruction)
                               or annotation.get('reference_sha256') != json_digest(reference_context(behavior))):
                raise ValueError(f'Annotation differs from the original benign task: {row.behavior_id}')
    return build_evaluation_report(
        attack, baseline=base, benign=normal, benign_baseline=normal_base,
        behavior_ids=behavior_ids, benign_label_means_utility=benign_label_means_utility,
        original_instructions={key: row.instruction for key, row in originals.items()} if originals is not None else None,
    )


def grade_evaluation_results(results, behaviors, *, kind='jailbreak', grader=None,
                             model_name=None, method_name=None, annotation_cache=None):
    """Attach rubric grades while preserving the existing completion JSON fields.

    An injected ResponseGrader/client supports OpenAI-compatible judge services.
    Cached annotations are reused only when request, response, model and rubric match.
    """
    if kind not in ('jailbreak', 'benign'):
        raise ValueError('kind must be jailbreak or benign')
    originals = normalize_evaluation_behaviors(behaviors)
    rows, locations = _records(results, model_name=model_name, method_name=method_name)
    missing = {row.behavior_id for row in rows} - originals.keys()
    if missing:
        raise ValueError(f'Original behaviors missing: {sorted(missing)}')
    if len({row.test_case_id for row in rows}) != len(rows):
        raise ValueError('Duplicate test_case_id')
    grader = grader or ResponseGrader()
    cache = annotation_cache if annotation_cache is not None else {}
    # Validate existing annotations before spending any judge requests.
    for row in rows:
        get_annotation(row, kind)
    output = copy.deepcopy(results)
    for row, (behavior_id, index) in zip(rows, locations):
        behavior = originals[behavior_id]
        fingerprint = grader.fingerprint(row, behavior, kind)
        existing = row.meta.get('evaluation', {})
        annotation = existing if existing.get('input_sha256') == fingerprint else cache.get(fingerprint)
        if annotation is None:
            annotation = grader.grade(row, behavior, kind)
        graded = grader.attach(row, behavior, kind, annotation)
        cache[fingerprint] = annotation
        record = output[behavior_id][index]
        record.update(test_case_id=graded.test_case_id, attempt=graded.attempt,
                      attack=graded.attack, model=graded.model, blocked=graded.blocked,
                      label=graded.label, judge=graded.judge, meta=graded.meta)
        # Keep a pre-existing top-level annotation in sync with the canonical meta field.
        if 'evaluation' in record:
            record['evaluation'] = copy.deepcopy(annotation)
    return output
