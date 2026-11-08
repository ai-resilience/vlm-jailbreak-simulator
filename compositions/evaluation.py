"""Recipe-aware reporting while retaining the original question as the scored unit."""
from collections import Counter


def case_meta(record):
    case = record.get('test_case')
    return case.get('meta', {}) if isinstance(case, dict) else {}


def evaluation_context(results, snapshot):
    if not snapshot or not snapshot.get('composition'):
        return None
    originals = {row['BehaviorID']: row for row in snapshot['behaviors']}
    allowed = {key for key, row in originals.items() if row.get('eval_type', 'harmful') in ('harmful', 'copyright')}
    counts = Counter()
    for behavior_id, records in results.items():
        for record in records:
            meta = case_meta(record)
            if (not meta.get('recipe_id') or meta.get('composition_identity') != snapshot['composition']['identity']
                    or meta.get('source_behavior_id') != behavior_id
                    or meta.get('source_eval_type') != originals[behavior_id].get('eval_type', 'harmful')):
                raise ValueError('Completion composition metadata differs from the original question/run')
            counts[meta['recipe_id']] += 1
    return {'allowed_ids': allowed, 'counts': dict(counts),
            'source_eval_type_counts': dict(Counter(row.get('eval_type', 'harmful') for row in originals.values()))}


def add_recipe_reports(report, cohorts, options, context):
    if context is None:
        return report
    from evaluation.compat import compute_evaluation_metrics
    per_recipe = {}
    for recipe_id in sorted(context['counts']):
        selected = {}
        for role, rows in cohorts.items():
            if role.startswith('benign'):
                # Benign controls are shared unless they explicitly carry a recipe ID.
                selected[role] = {key: [row for row in records if case_meta(row).get('recipe_id', recipe_id) == recipe_id]
                                  for key, records in rows.items()}
            else:
                selected[role] = {key: [row for row in records if case_meta(row).get('recipe_id') == recipe_id]
                                  for key, records in rows.items()}
            selected[role] = {key: value for key, value in selected[role].items() if value}
        recipe_options = {**options, 'behavior_ids': sorted(context['allowed_ids'] & set(selected['results']))}
        per_recipe[recipe_id] = compute_evaluation_metrics(**selected, **recipe_options)
    report['by_recipe'] = per_recipe
    report['composition'] = {'counts': context['counts'], 'source_eval_type_counts': context['source_eval_type_counts'],
        'headline_eval_types': ['harmful', 'copyright'],
        'aggregation': 'pass@k per original question; recipe reports use only that recipe attempts',
        'non_harmful_inputs_retained': True}
    return report
