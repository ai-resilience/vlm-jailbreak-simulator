"""VisualAdv PGD port on the existing BehaviorID-keyed CSV/JSON interface.

One image is optimized over the independent target corpus and reused for all
evaluation questions. pgd.py is copied unchanged from our reference simulator.
Shared surrogates are declared substitutions for the original paper models.
"""
from __future__ import annotations

import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from PIL import Image
from benchmark_datasets import portable_path, validate_behaviors
from data_paths import REPO_ROOT, resolve_path
from multimodalmodels.registry import resolve_model_id
from multimodalmodels.surrogates import (
    SURROGATE_SPECS, default_surrogate_for_backend, load_surrogate,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Visual_Adv:
    """Legacy CLI name with reference PGD and shared CE surrogate support."""

    official_repo = 'https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models'
    official_commit = '5fc9883cfa9b4183cf05c6eceef49e449e45e9c9'
    algorithm_fidelity = 'faithful_port'
    substitutes_model = True

    def __init__(self, config=None, **kwargs):
        c = {**(config or {}), **kwargs}
        self.config = copy.deepcopy(c)
        requested = c.get('surrogate') or c.get('target_model')
        if requested is None:
            requested = default_surrogate_for_backend(c.get('surrogate_backend', 'llava')).model_id
        model_id = resolve_model_id(requested)
        profiles = [name for name, spec in SURROGATE_SPECS.items() if spec.model_id == model_id]
        if not profiles:
            raise ValueError(f'Unsupported VisualAdv surrogate {requested!r}; choose {sorted(SURROGATE_SPECS)}')
        self.surrogate_profile = profiles[0]
        spec = SURROGATE_SPECS[self.surrogate_profile]
        if c.get('surrogate_backend'):
            expected = default_surrogate_for_backend(c['surrogate_backend']).backend
            if expected != spec.backend:
                raise ValueError('VisualAdv surrogate and surrogate_backend disagree')
        if c.get('target_model') and resolve_model_id(c['target_model']) != model_id:
            raise ValueError('Legacy target_model and surrogate disagree')
        self.surrogate = model_id
        self.surrogate_backend = spec.backend
        self.surrogate_revision = c.get('surrogate_revision') or spec.revision
        self.image_path = resolve_path(c.get('image_path', 'data/original/clean.jpeg'))
        self.target_path = resolve_path(c.get('target_path', 'data/method_assets/VisualAdv/derogatory_corpus.csv'))
        self.epsilon = float(c.get('epsilon', 16)) / 255.0
        self.alpha = float(c.get('alpha', 1)) / 255.0
        self.n_iters = c.get('n_iters', 5000)
        self.batch_size = c.get('batch_size', 8)
        self.constrained = c.get('constrained', True)
        self.seed = c.get('seed')
        self.iteration_convention = c.get('iteration_convention', 'released')
        if self.iteration_convention not in {'paper', 'released'}:
            raise ValueError("iteration_convention must be 'paper' or 'released'")
        if type(self.n_iters) is not int or self.n_iters < 0:
            raise ValueError('n_iters must be a nonnegative integer')
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if type(self.constrained) is not bool:
            raise ValueError('constrained must be boolean')
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError('seed must be an integer or null')
        self.actual_updates = self.n_iters + int(self.iteration_convention == 'released')
        if (self.actual_updates < 1 or not math.isfinite(self.alpha) or self.alpha <= 0
                or not math.isfinite(self.epsilon) or not 0 <= self.epsilon <= 1):
            raise ValueError('Invalid VisualAdv update count, alpha or epsilon')
        self._surrogate = None
        self._universal_path = None
        self._generated_images = {}
        self._optimization = None
        self._artifact_reused = False

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

    def effective_config(self):
        return {
            'surrogate_profile': self.surrogate_profile,
            'surrogate_backend': self.surrogate_backend, 'surrogate': self.surrogate,
            'surrogate_revision': self.surrogate_revision,
            'image_path': self.image_path, 'target_path': self.target_path,
            'epsilon': self.epsilon, 'alpha': self.alpha, 'n_iters': self.n_iters,
            'actual_updates': self.actual_updates, 'batch_size': self.batch_size,
            'constrained': self.constrained, 'seed': self.seed,
            'iteration_convention': self.iteration_convention,
            'initialization': 'benign_plus_uniform_delta' if self.constrained else 'uniform_image',
        }

    def _load_targets(self):
        with open(self.target_path, newline='', encoding='utf-8-sig') as source:
            targets = [row[0] for row in csv.reader(source) if row and row[0].strip()]
        if len(targets) < self.batch_size:
            raise ValueError('VisualAdv requires at least batch_size nonempty targets')
        return targets

    def _load_surrogate(self):
        if self._surrogate is None:
            self._surrogate = load_surrogate(self.surrogate_profile, revision=self.surrogate_revision)
        return self._surrogate

    def _signature(self):
        files = ['image_attacks/visual_adv/visual_adv.py', 'image_attacks/visual_adv/pgd.py',
                 'multimodalmodels/ce_surrogate.py', 'multimodalmodels/surrogates.py',
                 'multimodalmodels/loaders.py', 'multimodalmodels/registry.py']
        if self.surrogate_backend == 'llava':
            files += ['multimodalmodels/llava/llava_model.py', 'multimodalmodels/multimodalmodel.py']
        return {
            'effective_config': self.effective_config(),
            'inputs': {'image_sha256': _sha256(self.image_path), 'targets_sha256': _sha256(self.target_path)},
            'implementation_sha256': {name: _sha256(REPO_ROOT / name) for name in files},
        }

    def _verify_artifact(self, path, signature):
        sidecar = Path(str(path) + '.lineage.json')
        if not path.is_file() or not sidecar.is_file():
            raise ValueError('VisualAdv image requires a complete .lineage.json sidecar; use a new output directory')
        record = json.loads(sidecar.read_text(encoding='utf-8'))
        if (record.get('schema_version') != 1 or record.get('attack') != 'VisualAdv'
                or record.get('signature') != signature or record.get('image_sha256') != _sha256(path)
                or record.get('optimization', {}).get('n_updates') != self.actual_updates):
            raise ValueError('VisualAdv artifact/config/input/code fingerprint differs; use a new output directory')
        with Image.open(path) as image:
            image.verify()
        return record

    def _optimize(self, base_pil, targets):
        from tqdm import tqdm
        from .pgd import optimize_pixels

        surrogate = self._load_surrogate()
        base = surrogate.to_unit01(base_pil).clamp(0, 1).clone()
        losses, gradients = [], []
        gradient_hook = None

        def loss_fn(image, batch):
            nonlocal gradient_hook
            def observe(gradient):
                gradients.append({'finite': bool(gradient.isfinite().all()),
                                  'nonzero': bool(gradient.count_nonzero()),
                                  'mean_abs': float(gradient.detach().abs().mean())})
            gradient_hook = image.register_hook(observe)
            return surrogate.compute_loss_batch(image, batch, base_pil)

        with tqdm(total=self.actual_updates, desc='VisualAdv') as progress_bar:
            def progress(step, image, loss):
                # Unconstrained PGD reuses the same leaf tensor across updates.
                # Remove this update's observer before registering the next one.
                gradient_hook.remove()
                surrogate.zero_grad()
                losses.append(loss)
                progress_bar.update()
                if step % 50 == 0:
                    progress_bar.set_description(f'VisualAdv loss={loss:.4f}')
            final = optimize_pixels(
                base, targets, loss_fn, constrained=self.constrained, epsilon=self.epsilon,
                alpha=self.alpha, steps=self.actual_updates, batch_size=self.batch_size,
                seed=self.seed, callback=progress,
            )
        stats = {'n_updates': len(losses), 'target_count': len(targets), 'losses': losses,
                 'pixel_gradients': gradients, 'pixel_min': float(final.min()),
                 'pixel_max': float(final.max()), 'linf_from_base': float((final - base).abs().max()),
                 'image_shape': list(final.shape)}
        return final.squeeze(0).cpu(), stats

    def prepare(self, out_dir):
        """Optimize once or reuse only an authenticated, completed artifact."""
        from torchvision.utils import save_image

        targets = self._load_targets()
        with Image.open(self.image_path) as image:
            base = image.convert('RGB')
        signature = self._signature()
        output = Path(out_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        destination = output / 'visual_adv.png'
        explicit = self.config.get('adv_image')
        source = Path(resolve_path(explicit)) if explicit else self._universal_path or destination
        if explicit or source.exists() or Path(str(source) + '.lineage.json').exists():
            record = self._verify_artifact(source, signature)
            self._artifact_reused = True
            if source != destination:
                if destination.exists() or Path(str(destination) + '.lineage.json').exists():
                    self._verify_artifact(destination, signature)
                    if _sha256(destination) != record['image_sha256']:
                        raise ValueError('Existing VisualAdv destination contains a different artifact')
                else:
                    shutil.copyfile(source, destination)
                    _write_json(str(destination) + '.lineage.json', record)
        else:
            final, stats = self._optimize(base, targets)
            handle = tempfile.NamedTemporaryFile(dir=output, prefix='.visual_adv.', suffix='.png', delete=False)
            temporary = Path(handle.name)
            handle.close()
            try:
                save_image(final, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            record = {'schema_version': 1, 'attack': 'VisualAdv',
                      'official_repo': self.official_repo, 'official_commit': self.official_commit,
                      'signature': signature, 'image_sha256': _sha256(destination), 'optimization': stats}
            _write_json(str(destination) + '.lineage.json', record)
            self._artifact_reused = False
        self._universal_path = destination
        self._optimization = record['optimization']

    def artifact_fingerprints(self):
        return dict(self._generated_images)

    def generate_test_cases(self, behaviors, save_dir, verbose=False):
        rows = self.normalize_behaviors(behaviors)
        if not rows:
            raise ValueError('VisualAdv needs at least one evaluation question')
        self._load_targets()
        with Image.open(self.image_path) as image:
            image.verify()
        output = Path(save_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        with (output / '.Visual_Adv.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.prepare(output / 'images')
            cases = {row['BehaviorID']: ['visual_adv.png', row['Behavior']] for row in rows}
            self._generated_images = {'images/visual_adv.png': _sha256(self._universal_path)}
            metadata = {
                'schema_version': 1, 'attack': 'VisualAdv', 'official_repo': self.official_repo,
                'official_commit': self.official_commit, 'algorithm_fidelity': self.algorithm_fidelity,
                'substitutes_model': True, 'paper_reproduction': False,
                'effective_config': self.effective_config(), 'artifact_reused': self._artifact_reused,
                'universal_image': 'images/visual_adv.png', 'image_sha256': _sha256(self._universal_path),
                'optimization': self._optimization, 'n_behaviors': len(rows),
                'behavior_ids': [row['BehaviorID'] for row in rows],
            }
            _write_json(output / 'Visual_Adv.json', cases)
            _write_json(output / 'Visual_Adv_metadata.json', metadata)
        if verbose:
            print(f'VisualAdv: {len(rows)} questions share {portable_path(self._universal_path)}')
        return cases


VisualAdvAttack = Visual_Adv
