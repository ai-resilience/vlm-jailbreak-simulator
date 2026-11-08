"""Reviewed exact-ID/title-hash aliases for 21 released SafeBench typos.

No runtime fuzzy matching is performed. All other rows use the reference's
normal title lookup; unknown/mutated titles still fail.
"""
import hashlib
import json
from pathlib import Path

from benchmark_datasets import portable_path
from data_paths import resolve_path
from image_attacks.jp_common import sha256

DEFAULT_ALIGNMENT = 'data/method_assets/MML/safebench_alignment.json'
DEFAULT_ALIGNMENT_SHA256 = '66202a7efca64b097c2ceeba37554a5067c4f69159384d54fce5f3bec0a9c5ae'


def reviewed_record(method, behavior):
    if method._dataset_key(behavior.dataset) != 'safebench':
        return None
    path = Path(resolve_path(method.config.get('replay_alignment_path', DEFAULT_ALIGNMENT)))
    expected = method.config.get('replay_alignment_sha256')
    if expected is None:
        if method.config.get('replay_alignment_path'):
            raise ValueError('Custom MML replay_alignment_path requires replay_alignment_sha256')
        expected = DEFAULT_ALIGNMENT_SHA256
    if sha256(path) != expected:
        raise ValueError('MML reviewed alignment manifest fingerprint differs')
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('dataset') != 'SafeBench':
        raise ValueError('Invalid MML alignment manifest')
    entry = manifest.get('alignments', {}).get(behavior.behavior_id)
    if entry is None:
        return None
    candidate_hashes = {hashlib.sha256(title.encode()).hexdigest() for title in method._candidate_titles(behavior)}
    if entry['benchmark_title_sha256'] not in candidate_hashes:
        raise ValueError('MML reviewed alignment question fingerprint differs')
    index = method._precomputed_indices['safebench']
    matches = [r for r in index.values() if r['subclass'] == entry['official_subclass']
               and r['row']['id'] == entry['official_row_id']]
    if len(matches) != 1:
        raise ValueError('MML reviewed alignment has no unique released row')
    record = matches[0]
    title_hash = hashlib.sha256(record['row']['original_prompt'].encode()).hexdigest()
    if title_hash != entry['upstream_title_sha256'] or sha256(record['metadata_path']) != entry['metadata_sha256']:
        raise ValueError('MML reviewed alignment released-data fingerprint differs')
    method._applied_alignments[behavior.behavior_id] = {
        'matching_strategy': 'reviewed_id_and_title_hash_alignment',
        'alignment_manifest': portable_path(path), 'alignment_manifest_sha256': expected,
        'benchmark_title_sha256': entry['benchmark_title_sha256'], 'upstream_title_sha256': title_hash}
    return record
