"""MML's replay/renderer adapted to the existing BehaviorID JSON pipeline."""
import copy
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import shutil
import tempfile

from PIL import Image

from benchmark_datasets import portable_path, validate_behaviors
from data_paths import REPO_ROOT, resolve_path
from image_attacks.jp_common import sha256, write_json
from .engine import MMLAttack as ReferenceMML
from .official_source import load_upstream_prompts
from .schema import Behavior


class MML(ReferenceMML):
    default_dataset = 'SafeBench'

    def __init__(self, config=None, **kwargs):
        c = {**(config or {}), **kwargs}
        for key in ('attempts', 'canvas_width', 'canvas_height', 'font_size', 'wrap_width', 'steps'):
            if key in c and (type(c[key]) is not int or c[key] < 1):
                raise ValueError(f'MML {key} must be a positive integer')
        if c.get('seed') is not None and type(c['seed']) is not int:
            raise ValueError('MML seed must be an integer or null')
        if 'strict_font' in c and type(c['strict_font']) is not bool:
            raise ValueError('MML strict_font must be boolean')
        super().__init__(c)
        self._generated_images = {}
        self._last_exclusions = []
        self._applied_alignments = {}

    def _find_precomputed_record(self, behavior):
        record = super()._find_precomputed_record(behavior)
        if record is None:
            from .replay_alignment import reviewed_record
            return reviewed_record(self, behavior)
        return record

    def _tag_tokens(self, title):
        import nltk
        resource = resolve_path(self.config.get('nltk_data_path', 'data/method_assets/MML/nltk_data'))
        if Path(resource).is_dir() and resource not in nltk.data.path:
            nltk.data.path.insert(0, resource)
        return super()._tag_tokens(title)

    @staticmethod
    def normalize_behaviors(behaviors):
        rows = []
        for original in behaviors:
            row = copy.deepcopy(original if isinstance(original, dict) else original.to_dict())
            row['BehaviorID'] = row.get('BehaviorID') or row.get('behavior_id')
            row['Behavior'] = row.get('Behavior') or row.get('instruction') or row.get('question')
            if isinstance(row.get('meta'), str):
                row['meta'] = json.loads(row['meta'])
            if not isinstance(row.get('meta') or {}, dict):
                raise ValueError('MML behavior meta must be an object')
            rows.append(row)
        validate_behaviors(rows)
        return rows

    @staticmethod
    def behavior(row):
        return Behavior(row['BehaviorID'], row.get('dataset', 'custom'), row['Behavior'],
                        row.get('statement'), row.get('category'), row.get('category_orig'), row.get('meta') or {})

    def filter_evaluation_behaviors(self, rows):
        if self.prompt_source == 'upstream':
            self._upstream_prompts = load_upstream_prompts(self.upstream_const_path,
                expected_sha256=self.upstream_const_sha256)
        selected, excluded = [], []
        for row in rows:
            missing = self.missing_requirements(self.behavior(row))
            if missing == ['official_mml_filtered_row']:
                excluded.append({'BehaviorID': row['BehaviorID'], 'reason': 'official_mml_filtered_row'})
            elif missing:
                raise FileNotFoundError('MML released dataset is unavailable for this dataset; supply precomputed_root or select generated mode')
            else:
                selected.append(row)
        if not selected:
            raise ValueError('No MML questions remain after official data filtering')
        self._last_exclusions = excluded
        return selected, excluded

    def artifact_fingerprints(self):
        return dict(self._generated_images)

    def effective_config(self):
        result = super().effective_config()
        if self.input_mode == 'precomputed':
            from .replay_alignment import DEFAULT_ALIGNMENT, DEFAULT_ALIGNMENT_SHA256
            result.update(replay_alignment_path=portable_path(resolve_path(self.config.get('replay_alignment_path', DEFAULT_ALIGNMENT))),
                          replay_alignment_sha256=self.config.get('replay_alignment_sha256', DEFAULT_ALIGNMENT_SHA256))
        return result

    def generate_test_cases(self, behaviors, save_dir, verbose=False):
        previous = list(self._last_exclusions)
        rows, excluded = self.filter_evaluation_behaviors(self.normalize_behaviors(behaviors))
        output = Path(save_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        with (output / '.MML.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Stage the entire selected set: failure cannot publish completed cases.
            with tempfile.TemporaryDirectory(dir=output, prefix='.mml-') as temporary:
                staging = Path(temporary)
                cases, source_records = {}, {}
                self._generated_images = {}
                for row in rows:
                    built = self.build(self.behavior(row), str(staging))
                    path = Path(resolve_path(built[0].images[0]))
                    with Image.open(path) as image:
                        image.verify()
                    if self.input_mode == 'precomputed':
                        from .engine import _safe_stem
                        filename = _safe_stem(row['BehaviorID']) + path.suffix.lower()
                        shutil.copyfile(path, staging / filename)
                    else:
                        filename = path.name
                    records = []
                    for case in built:
                        record = asdict(case)
                        record['images'] = [filename]
                        record['meta'].update(self._applied_alignments.get(row['BehaviorID'], {}))
                        records.append(record)
                    cases[row['BehaviorID']] = records
                    source_record = {'image_sha256': sha256(staging / filename)}
                    if self.input_mode == 'precomputed':
                        metadata = Path(resolve_path(built[0].meta['official_metadata']))
                        source_record.update(image=portable_path(path), metadata=portable_path(metadata),
                                             metadata_sha256=sha256(metadata))
                    source_records[row['BehaviorID']] = source_record
                    self._generated_images['images/' + filename] = source_record['image_sha256']
                inputs = {'behaviors': rows, 'sources': source_records}
                if self.prompt_source == 'upstream':
                    inputs['upstream_const_sha256'] = sha256(self.upstream_const_path)
                if self._font_source and Path(self._font_source).is_file():
                    inputs['font_sha256'] = sha256(self._font_source)
                if any(case['meta'].get('pos_tagger_used') == 'nltk'
                       for records in cases.values() for case in records):
                    import nltk
                    resource = nltk.data.find('taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle')
                    inputs['nltk_tagger_sha256'] = sha256(str(resource))
                names = ['multi_attacks/mml/' + name for name in ('engine.py', 'mml.py', 'schema.py', 'prompts.py', 'official_source.py', 'replay_alignment.py')]
                signature = {'effective_config': self.effective_config(), 'inputs': inputs,
                             'implementation_sha256': {name: sha256(REPO_ROOT / name) for name in names}}
                metadata_path = output / 'MML_metadata.json'
                if metadata_path.exists():
                    old = json.loads(metadata_path.read_text())
                    if old.get('generation_signature') != signature:
                        raise ValueError('MML config/input/source/code fingerprint differs; choose a new output directory')
                image_dir = output / 'images'
                image_dir.mkdir(exist_ok=True)
                for relative in self._generated_images:
                    filename = Path(relative).name
                    shutil.copyfile(staging / filename, image_dir / filename)
                metadata = {'schema_version': 1, 'attack': 'MML', **self.provenance(),
                            'generation_signature': signature, 'effective_config': self.effective_config(),
                            'n_behaviors': len(rows), 'n_cases': len(rows) * self.attempts,
                            'reviewed_replay_alignments': self._applied_alignments,
                            'evaluation_exclusions': excluded or previous, 'generated_images': self._generated_images,
                            'protocol_note': 'Standard completion queries every attempt; mml_official interleaves CLAS and stops on score 5'}
                write_json(output / 'MML_test_cases.json', cases)
                write_json(metadata_path, metadata)
        if verbose:
            print(f'MML {self.variant}/{self.input_mode}: {len(rows)} questions, {self.attempts} attempts each')
        return cases


MMLAttack = MML
