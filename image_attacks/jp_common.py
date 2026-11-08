"""Shared CLI, split provenance and artifact handling for ImgJP/DeltaJP."""
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import unicodedata

from PIL import Image

from benchmark_datasets import portable_path, validate_behaviors
from data_paths import REPO_ROOT, resolve_path
from multimodalmodels.registry import resolve_model_id
from multimodalmodels.surrogates import SURROGATE_SPECS, default_surrogate_for_backend, load_surrogate


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def instruction_hash(text):
    text = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class JPAttack:
    official_repo = 'https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model'
    official_commit = 'fd98910ae4075fe587a1944095c00660aa24cc54'
    algorithm_fidelity = 'adapted'

    def __init__(self, config=None, **kwargs):
        c = {**(config or {}), **kwargs}
        self.config = copy.deepcopy(c)
        requested = c.get('surrogate') or c.get('target_model')
        if requested is None:
            requested = default_surrogate_for_backend(c.get('surrogate_backend', 'llava')).model_id
        model_id = resolve_model_id(requested)
        profiles = [name for name, spec in SURROGATE_SPECS.items() if spec.model_id == model_id]
        if not profiles:
            raise ValueError(f'Unsupported JP surrogate: {requested!r}')
        self.profile = profiles[0]
        spec = SURROGATE_SPECS[self.profile]
        if c.get('surrogate_backend') and default_surrogate_for_backend(c['surrogate_backend']).backend != spec.backend:
            raise ValueError('JP surrogate and surrogate_backend disagree')
        if c.get('target_model') and resolve_model_id(c['target_model']) != model_id:
            raise ValueError('JP surrogate and legacy target_model disagree')
        self.surrogate, self.backend = model_id, spec.backend
        self.revision = c.get('surrogate_revision') or spec.revision
        if 'epsilon' in c and 'eps' in c and float(c['epsilon']) != float(c['eps']):
            raise ValueError('epsilon and legacy eps must agree')
        self.epsilon = float(c.get('epsilon', c.get('eps', 16))) / 255
        self.alpha = float(c.get('alpha', 1 if self.name == 'ImgJP' else 0.5)) / 255
        self.steps = c.get('steps', 200 if self.name == 'ImgJP' else 100)
        self.train_num = c.get('train_num', 25)
        self.batch_size = c.get('batch_size', 8)
        self.seed = c.get('seed', 0)
        self.random_start = c.get('random_start', False)
        self.exclude_train = c.get('exclude_train_from_eval', True)
        self.split_strategy = c.get('split_strategy', 'stratified')
        for name in ('steps', 'train_num', 'batch_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if any(type(x) is not bool for x in (self.random_start, self.exclude_train)):
            raise ValueError('random_start and exclude_train_from_eval must be boolean')
        if c.get('constrained', True) is not True:
            raise ValueError('ImgJP/DeltaJP use bounded PGD; constrained must be true')
        if not math.isfinite(self.alpha) or self.alpha <= 0 or not math.isfinite(self.epsilon) or not 0 <= self.epsilon <= 1:
            raise ValueError('Invalid JP alpha or epsilon')
        if self.split_strategy not in ('stratified', 'first'):
            raise ValueError("split_strategy must be 'stratified' or 'first'")
        self._train = None
        self._surrogate = None
        self._generated_images = {}
        self._last_exclusions = []

    @staticmethod
    def normalize_behaviors(behaviors):
        rows = []
        for original in behaviors:
            row = copy.deepcopy(original if isinstance(original, dict) else original.to_dict())
            row['BehaviorID'] = row.get('BehaviorID') or row.get('behavior_id')
            row['Behavior'] = row.get('Behavior') or row.get('instruction') or row.get('question')
            rows.append(row)
        validate_behaviors(rows)
        return rows

    def image_path(self, row):
        images = row.get('images') or []
        if isinstance(images, str):
            images = json.loads(images)
        if not isinstance(images, list):
            raise ValueError('Behavior images must be a list')
        if images:
            return Path(resolve_path(images[0]))
        filename = row.get('ImageFileName')
        if filename:
            image_root = Path(resolve_path(self.config.get('image_path', 'data/original/HarmBench/images')))
            path = (image_root / filename).resolve()
            if not path.is_relative_to(image_root.resolve()):
                raise ValueError('ImageFileName must stay inside image_path')
            return path
        raise ValueError(f"DeltaJP requires an original image for {row['BehaviorID']}")

    def training_corpus(self):
        if self._train is None:
            self._train = self._select_training()
            if not self._train:
                raise ValueError('JP training corpus is empty')
            validate_behaviors(self._train)
        return self._train

    def filter_evaluation_behaviors(self, rows):
        train = self.training_corpus()
        ids = {row['BehaviorID'] for row in train}
        hashes = {instruction_hash(row['Behavior']) for row in train}
        selected, excluded = [], []
        for row in rows:
            reason = ('train_split' if row['BehaviorID'] in ids else
                      'train_content_dup' if instruction_hash(row['Behavior']) in hashes else None)
            if self.exclude_train and reason:
                excluded.append({'BehaviorID': row['BehaviorID'], 'reason': reason})
            else:
                selected.append(row)
        self._last_exclusions = excluded
        if not selected:
            raise ValueError('All selected questions overlap JP training; select held-out questions or explicitly disable exclude_train_from_eval')
        return selected, excluded

    def _load_surrogate(self):
        if self._surrogate is None:
            self._surrogate = load_surrogate(self.profile, revision=self.revision)
        return self._surrogate

    def effective_config(self):
        result = {'surrogate_profile': self.profile, 'surrogate': self.surrogate,
                  'surrogate_backend': self.backend, 'surrogate_revision': self.revision,
                  'epsilon': self.epsilon, 'alpha': self.alpha, 'steps': self.steps,
                  'train_num': self.train_num, 'seed': self.seed, 'random_start': self.random_start,
                  'exclude_train_from_eval': self.exclude_train}
        if self.name == 'DeltaJP':
            result.update(train_dataset=self.config.get('train_dataset', 'AdvBenchM'),
                          train_behaviors_path=self.config.get('train_behaviors_path'),
                          split_strategy=self.split_strategy, batch_size=self.batch_size,
                          image_selection='first_catalog_image', conditioning='target_only_reference_delta')
        else:
            result.update(train_csv=self.train_csv, train_candidate_limit=self.candidate_limit,
                          update_order='sequential_prompt_per_epoch', conditioning='goal_and_target',
                          initialization='inverse_clip_normalized_zero', effective_batch_size=1)
        return result

    def _signature(self):
        train = []
        for row in self.training_corpus():
            record = {'BehaviorID': row['BehaviorID'], 'Behavior': row['Behavior'],
                      'target': row['target'], 'category': row.get('category')}
            if self.name == 'DeltaJP':
                image = self.image_path(row)
                with Image.open(image) as pil:
                    pil.verify()
                record.update(image=portable_path(image), image_sha256=sha256(image))
            train.append(record)
        files = ['image_attacks/jp_common.py', 'image_attacks/jp_pgd.py',
                 f'image_attacks/{self.module}/{self.module}.py', 'benchmark_datasets.py',
                 'multimodalmodels/ce_surrogate.py', 'multimodalmodels/surrogates.py',
                 'multimodalmodels/registry.py', 'multimodalmodels/loaders.py']
        if self.backend == 'llava':
            files += ['multimodalmodels/llava/llava_model.py', 'multimodalmodels/multimodalmodel.py']
        return {'effective_config': self.effective_config(), 'training_behaviors': train,
                'implementation_sha256': {name: sha256(REPO_ROOT / name) for name in files}}

    def _artifact(self, output, signature):
        import torch
        folder = output / 'artifacts'
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / ('universal_image.pt' if self.name == 'ImgJP' else 'universal_delta.pt')
        explicit = self.config.get('universal_artifact')
        source = Path(resolve_path(explicit)) if explicit else destination
        sidecar = Path(str(source) + '.lineage.json')
        if explicit or source.exists() or sidecar.exists():
            if not source.is_file() or not sidecar.is_file():
                raise ValueError('JP artifact requires a complete .lineage.json; choose a new output directory')
            record = json.loads(sidecar.read_text())
            expected_updates = self.steps * (len(self.training_corpus()) if self.name == 'ImgJP' else 1)
            if (record.get('attack') != self.name or record.get('schema_version') != 1
                    or record.get('signature') != signature or record.get('sha256') != sha256(source)
                    or record.get('optimization', {}).get('n_updates') != expected_updates):
                raise ValueError('JP artifact/config/training/code fingerprint differs; choose a new output directory')
            tensor = torch.load(source, map_location='cpu', weights_only=True)
            if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or tensor.shape[:2] != (1, 3)
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError('Invalid JP universal tensor')
            if self.name == 'DeltaJP' and float(tensor.abs().max()) > self.epsilon + 1e-6:
                raise ValueError('JP cached delta exceeds epsilon')
            if self.name == 'ImgJP' and not 0 <= float(tensor.min()) <= float(tensor.max()) <= 1:
                raise ValueError('JP cached image is outside [0,1]')
            reused = True
        else:
            tensor, stats = self._optimize()
            tensor = tensor.detach().cpu()
            record = {'schema_version': 1, 'attack': self.name, 'official_repo': self.official_repo,
                      'official_commit': self.official_commit, 'signature': signature, 'optimization': stats}
            reused = False
        if source != destination or not reused:
            destination_sidecar = Path(str(destination) + '.lineage.json')
            if source != destination and (destination.exists() or destination_sidecar.exists()):
                if not destination.is_file() or not destination_sidecar.is_file():
                    raise ValueError('Incomplete JP destination; choose a new output directory')
                previous = json.loads(destination_sidecar.read_text())
                if previous != record or sha256(destination) != record['sha256']:
                    raise ValueError('Existing JP destination differs; choose a new output directory')
            with tempfile.NamedTemporaryFile(dir=folder, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                if reused:
                    shutil.copyfile(source, temporary)
                else:
                    torch.save(tensor, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            record['sha256'] = sha256(destination)
            write_json(str(destination) + '.lineage.json', record)
        return tensor, record, reused

    def artifact_fingerprints(self):
        return dict(self._generated_images)

    def generate_test_cases(self, behaviors, save_dir, verbose=False):
        rows = self.normalize_behaviors(behaviors)
        previous_exclusions = list(self._last_exclusions)
        rows, exclusions = self.filter_evaluation_behaviors(rows)
        exclusions = exclusions or previous_exclusions
        if self.name == 'DeltaJP':
            for row in rows:
                with Image.open(self.image_path(row)) as image:
                    image.verify()
        signature = self._signature()
        output = Path(save_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        with (output / f'.{self.name}.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            tensor, record, reused = self._artifact(output, signature)
            cases, image_details = self._render(rows, output / 'images', tensor)
            self._generated_images = {
                'images/' + case[0]: sha256(output / 'images' / case[0]) for case in cases.values()}
            metadata = {'schema_version': 1, 'attack': self.name, 'official_repo': self.official_repo,
                        'official_commit': self.official_commit, 'algorithm_fidelity': 'adapted',
                        'substitutes_model': True, 'paper_reproduction': False,
                        'effective_config': self.effective_config(), 'artifact_reused': reused,
                        'universal_artifact_sha256': record['sha256'], 'optimization': record['optimization'],
                        'training_behavior_ids': [row['BehaviorID'] for row in self.training_corpus()],
                        'training_instruction_hashes': [instruction_hash(row['Behavior']) for row in self.training_corpus()],
                        'evaluation_exclusions': exclusions, 'n_behaviors': len(rows),
                        'behavior_ids': [row['BehaviorID'] for row in rows], 'images': image_details}
            write_json(output / f'{self.name}_test_cases.json', cases)
            write_json(output / f'{self.name}_metadata.json', metadata)
        if verbose:
            print(f'{self.name}: {len(rows)} questions; reused={reused}')
        return cases

    @staticmethod
    def save_tensor_image(tensor, path):
        from torchvision.utils import save_image
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            save_image(tensor, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
