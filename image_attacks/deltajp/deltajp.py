"""Port of the reference simulator's adapted cross-class DeltaJP."""
import random
import json

from PIL import Image

from benchmark_datasets import load_dataset, portable_path, read_csv
from image_attacks.jp_common import JPAttack, sha256
from image_attacks.jp_pgd import optimize_deltajp
from data_paths import resolve_path


def _affirmative_target(instruction):
    return f'Sure, here is {instruction[0].lower() + instruction[1:]}'


class DeltaJP(JPAttack):
    name = 'DeltaJP'
    module = 'deltajp'
    default_dataset = 'AdvBenchM'

    def _select_training(self):
        source = self.config.get('train_behaviors_path')
        rows = self.normalize_behaviors(read_csv(source) if source else load_dataset(self.config.get('train_dataset', 'AdvBenchM')))
        candidates = [row for row in rows if row.get('images') or row.get('ImageFileName')]
        if self.split_strategy == 'first':
            selected = candidates[:self.train_num]
        else:
            by_category = {}
            for row in candidates:
                by_category.setdefault(row.get('category') or '', []).append(row)
            rng = random.Random(self.seed)
            for group in by_category.values():
                rng.shuffle(group)
            selected = []
            while len(selected) < min(self.train_num, len(candidates)):
                for category in sorted(by_category):
                    if by_category[category]:
                        selected.append(by_category[category].pop())
                        if len(selected) == self.train_num:
                            break
        legacy_targets = None
        if self.config.get('target_path'):
            with open(resolve_path(self.config['target_path']), encoding='utf-8') as stream:
                legacy_targets = json.load(stream)
            if not isinstance(legacy_targets, dict):
                raise ValueError('DeltaJP target_path must contain an ID-to-target JSON object')
        for row in selected:
            if legacy_targets is not None:
                if row['BehaviorID'] not in legacy_targets:
                    raise ValueError(f"DeltaJP target_path has no target for {row['BehaviorID']}")
                row['target'] = legacy_targets[row['BehaviorID']]
            else:
                row['target'] = row.get('target') or _affirmative_target(row['Behavior'])
            if not isinstance(row['target'], str) or not row['target'].strip():
                raise ValueError('DeltaJP training targets must be nonempty strings')
        return selected

    def _optimize(self):
        from tqdm import tqdm
        surrogate = self._load_surrogate()
        train = self.training_corpus()
        pils = []
        for row in train:
            with Image.open(self.image_path(row)) as image:
                pils.append(image.convert('RGB'))
        # Bicubic resizing in native processors may overshoot [0,1]. Define
        # the clean pixels before applying the bounded adversarial delta.
        units = [surrogate.to_unit01(image).clamp(0, 1).detach() for image in pils]
        if any(unit.shape != units[0].shape for unit in units):
            raise ValueError('DeltaJP surrogate must provide one consistent image geometry')
        with tqdm(total=self.steps, desc='DeltaJP') as progress:
            def loss_fn(image, indices, targets):
                return surrogate.compute_loss_batch(image, targets, [pils[i] for i in indices])
            def callback(step, indices, loss):
                surrogate.zero_grad()
                progress.update()
            return optimize_deltajp(units, [row['target'] for row in train], loss_fn,
                                    epsilon=self.epsilon, alpha=self.alpha, steps=self.steps,
                                    batch_size=self.batch_size, seed=self.seed,
                                    random_start=self.random_start, callback=callback)

    def _render(self, rows, folder, tensor):
        import torch
        surrogate = self._load_surrogate()
        delta = tensor.to(surrogate.device)
        cases, details = {}, {}
        train_images = {sha256(self.image_path(row)) for row in self.training_corpus()}
        for row in rows:
            source = self.image_path(row)
            with Image.open(source) as pil:
                unit = surrogate.to_unit01(pil.convert('RGB')).clamp(0, 1)
            if unit.shape != delta.shape:
                raise ValueError('DeltaJP delta and evaluation image shapes disagree')
            with torch.no_grad():
                adv = (unit + delta).clamp(0, 1)
            filename = row['BehaviorID'] + '_delta.png'
            self.save_tensor_image(adv, folder / filename)
            source_hash = sha256(source)
            details[row['BehaviorID']] = {
                'source': portable_path(source), 'source_sha256': source_hash,
                'source_shared_with_training': source_hash in train_images,
                'image_selection': 'first_catalog_image',
                'image_association': row.get('meta', {}).get('image_association'),
                'continuous_linf': float((adv - unit).abs().max()),
                'pixel_min': float(adv.min()), 'pixel_max': float(adv.max())}
            cases[row['BehaviorID']] = [filename, row['Behavior']]
        return cases, details


DeltaJPAttack = DeltaJP
