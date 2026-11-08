"""Adapt shared benchmark rows to the original composition record contract."""
import copy

from benchmark_datasets import load_dataset, validate_behaviors
from .records import Behavior, EvalType


def from_row(row):
    row = copy.deepcopy(row)
    known = ('category', 'keyword', 'keyword_type', 'statement', 'target', 'images', 'meta')
    values = {key: row[key] for key in known if key in row}
    category_orig = row.get('category_orig', row.get('category'))
    if row.get('dataset') == 'MM-SafetyBench' and 'category_orig' not in row:
        category_orig = f"{row['meta']['scenario_id']:02d}-{row['category']}"
    return Behavior(row['BehaviorID'], row.get('dataset', 'custom'), row['Behavior'],
                    EvalType(row.get('eval_type', 'harmful')), category_orig=category_orig, **values)


def load_source_pool(name):
    return [from_row(row) for row in load_dataset(name)]


def scoring_row(behavior):
    return {'BehaviorID': behavior.behavior_id, 'Behavior': behavior.instruction,
            'dataset': behavior.dataset, 'eval_type': behavior.eval_type.value,
            'category': behavior.category, 'category_orig': behavior.category_orig,
            'statement': behavior.statement, 'keyword': behavior.keyword,
            'keyword_type': behavior.keyword_type, 'target': behavior.target,
            'images': list(behavior.images), 'meta': copy.deepcopy(behavior.meta), 'Tags': ''}
