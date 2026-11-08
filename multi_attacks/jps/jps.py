"""JPS integration on the BehaviorID CSV/JSON CLI; dedicated Qwen2-VL path."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from PIL import Image

from benchmark_datasets import load_dataset, read_csv, validate_behaviors
from data_paths import REPO_ROOT, resolve_path
from image_attacks.jp_common import instruction_hash, sha256, write_json
from multimodalmodels.registry import (QWEN2_VL_HF_ID, QWEN2_VL_REVISION,
                                       QWEN25_14B_HF_ID, QWEN25_14B_REVISION,
                                       default_revision, resolve_model_id)
from .jps_optimization import (JPSOptimization, _affirmative_target, _jps_prompt,
                               _official_target, _parse_ratio, _project_linf)
from .jps_prompts import ANGLES, JB_PROMPT_TEMPLATE


@dataclass
class TrainingBehavior:
    behavior_id: str
    instruction: str
    target: str | None


class JPS(JPSOptimization):
    official_repo = 'https://github.com/thu-coai/JPS'
    official_commit = 'beacbc58a359005e00a7e7f7b0b485eae64bb163'
    algorithm_fidelity = 'adapted'
    default_dataset = 'AdvBench'

    def __init__(self, config=None, **kwargs):
        self.config = c = {**(config or {}), **kwargs}
        self.surrogate = resolve_model_id(c.get('surrogate', QWEN2_VL_HF_ID))
        if self.surrogate != QWEN2_VL_HF_ID:
            raise ValueError('JPS needs its dedicated Qwen2-VL-7B adapter; common CE profiles are not interchangeable')
        self.surrogate_revision = c.get('surrogate_revision') or QWEN2_VL_REVISION
        self.agent_model = resolve_model_id(c.get('agent_model', QWEN25_14B_HF_ID))
        self.agent_revision = c.get('agent_revision') or default_revision(self.agent_model)
        self.agent_backend = c.get('agent_backend', 'huggingface')
        if self.agent_backend != 'huggingface':
            raise ValueError('This JPS integration supports the local huggingface agent backend')
        self.base_image = resolve_path(c.get('base_image', 'data/original/clean.jpeg'))
        self.train_csv = resolve_path(c.get('train_csv', 'data/method_assets/JPS/advbench_subset.csv')) if c.get('train_csv', True) else None
        self.train_dataset = c.get('train_dataset', 'AdvBench')
        self.image_size = c.get('image_size', 448)
        self.epsilon = _parse_ratio(c.get('epsilon', '32/255'))
        self.alpha = _parse_ratio(c.get('alpha', '1/255'))
        self.attack_steps = c.get('attack_steps', 300)
        self.batch_size = c.get('batch_size', 4)
        self.rounds = c.get('rounds', 5)
        self.attack_data_num = c.get('attack_data_num', 50)
        self.response_max_new_tokens = c.get('response_max_new_tokens', 256)
        self.momentum_decay = float(c.get('momentum_decay', 0.9))
        self.min_loss_limit = float(c.get('min_loss_limit', 0.01))
        self.strict_agents = c.get('strict_agents', True)
        self.revision_retries = c.get('revision_retries', 2)
        self.exclude_train_from_eval = c.get('exclude_train_from_eval', True)
        for name in ('image_size', 'attack_steps', 'batch_size', 'attack_data_num', 'response_max_new_tokens'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'JPS {name} must be a positive integer')
        for name in ('rounds', 'revision_retries'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'JPS {name} must be a nonnegative integer')
        if self.image_size % 28:
            raise ValueError('JPS image_size must be divisible by Qwen2 patch*merge size 28')
        if (not math.isfinite(self.epsilon) or not 0 <= self.epsilon <= 1
                or not math.isfinite(self.alpha) or self.alpha <= 0
                or not math.isfinite(self.min_loss_limit) or self.min_loss_limit < 0
                or not math.isfinite(self.momentum_decay) or not 0 <= self.momentum_decay <= 1):
            raise ValueError('Invalid JPS epsilon, alpha, loss limit or momentum')
        if type(self.strict_agents) is not bool or type(self.exclude_train_from_eval) is not bool:
            raise ValueError('JPS strict_agents and exclude_train_from_eval must be boolean')
        self._adapter = self._agents = None
        self._train_behaviors = None
        self._train_target_overrides = {}
        self._agent_trace_path = None
        self._steer = ''
        self._pgd_passes = []
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

    @staticmethod
    def _rel(path):
        return resolve_path(path)

    def _load(self):
        if self._adapter is None:
            from .jps_adapter import Qwen2Adapter
            self._adapter = Qwen2Adapter(self.surrogate, revision=self.surrogate_revision,
                                         image_size=(self.image_size, self.image_size))
        return self._adapter

    def _load_agents(self):
        if self._agents is None:
            from .jps_agents import JPSAgents
            self._agents = JPSAgents(model=self.agent_model, backend=self.agent_backend,
                                     revision=self.agent_revision, strict=self.strict_agents,
                                     revision_retries=self.revision_retries, trace_path=self._agent_trace_path)
        return self._agents

    def _target_for(self, behavior):
        if isinstance(behavior, dict):
            behavior = TrainingBehavior(behavior['BehaviorID'], behavior['Behavior'], behavior.get('target'))
        raw = self._train_target_overrides.get(behavior.behavior_id,
                  behavior.target or _affirmative_target(behavior.instruction))
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError('JPS targets must be nonempty strings')
        return _official_target(raw)

    def _training_corpus(self):
        if self._train_behaviors is not None:
            return self._train_behaviors
        candidates = load_dataset(self.train_dataset)
        if self.train_csv:
            rows = read_csv(self.train_csv)
            if len(rows) < self.attack_data_num:
                raise ValueError('JPS train_csv has fewer rows than attack_data_num')
            by_instruction = {row['Behavior'].strip(): row for row in candidates}
            selected = []
            for item in rows[:self.attack_data_num]:
                goal = (item.get('goal') or item.get('prompt') or item.get('origin_question') or '').strip()
                if goal not in by_instruction:
                    raise ValueError('JPS train_csv contains a question absent from train_dataset')
                row = by_instruction[goal]
                selected.append(row)
                target = (item.get('target') or '').strip()
                if target:
                    self._train_target_overrides[row['BehaviorID']] = target
        else:
            if len(candidates) < self.attack_data_num:
                raise ValueError('JPS train_dataset has fewer rows than attack_data_num')
            selected = candidates[:self.attack_data_num]
        validate_behaviors(selected)
        self._train_behaviors = [TrainingBehavior(row['BehaviorID'], row['Behavior'], row.get('target')) for row in selected]
        return self._train_behaviors

    def training_behavior_ids(self):
        return [row.behavior_id for row in self._training_corpus()]

    def filter_evaluation_behaviors(self, rows):
        train = self._training_corpus()
        ids = {row.behavior_id for row in train}
        hashes = {instruction_hash(row.instruction) for row in train}
        selected, excluded = [], []
        for row in rows:
            reason = ('train_split' if row['BehaviorID'] in ids else
                      'train_content_dup' if instruction_hash(row['Behavior']) in hashes else None)
            if self.exclude_train_from_eval and reason:
                excluded.append({'BehaviorID': row['BehaviorID'], 'reason': reason})
            else:
                selected.append(row)
        if not selected:
            raise ValueError('All selected questions overlap JPS training; select held-out questions')
        self._last_exclusions = excluded
        return selected, excluded

    def effective_config(self):
        keys = ('surrogate', 'surrogate_revision', 'agent_model', 'agent_backend', 'agent_revision',
                'base_image', 'train_csv', 'train_dataset', 'image_size', 'epsilon', 'alpha',
                'attack_steps', 'batch_size', 'min_loss_limit', 'rounds', 'attack_data_num',
                'momentum_decay', 'response_max_new_tokens', 'strict_agents', 'revision_retries',
                'exclude_train_from_eval')
        return {**{key: getattr(self, key) for key in keys}, 'processor_use_fast': False,
                'jb_prompt_template': JB_PROMPT_TEMPLATE, 'revision_angles': list(ANGLES),
                'target_rewrite': 'Sure, -> I am happy to help you with that.; here -> Here'}

    def _signature(self):
        with Image.open(self.base_image) as image:
            image.verify()
        inputs = {'base_image': sha256(self.base_image)}
        if self.train_csv:
            inputs['train_csv'] = sha256(self.train_csv)
        train = [{'id': row.behavior_id, 'instruction': row.instruction, 'target': self._target_for(row)}
                 for row in self._training_corpus()]
        files = ['multi_attacks/jps/' + name for name in
                 ('jps.py', 'jps_optimization.py', 'jps_adapter.py', 'jps_agents.py', 'jps_prompts.py')]
        files += ['image_attacks/jp_common.py', 'benchmark_datasets.py',
                  'multimodalmodels/loaders.py', 'multimodalmodels/registry.py']
        return {'effective_config': self.effective_config(), 'inputs': inputs, 'training_behaviors': train,
                'implementation_sha256': {name: sha256(REPO_ROOT / name) for name in files}}

    def _pgd(self, *args, **kwargs):
        result = super()._pgd(*args, **kwargs)
        self._pgd_passes.append(copy.deepcopy(self._last_pgd_stats))
        return result

    def _verify_artifact(self, source, signature):
        sidecar = Path(str(source) + '.lineage.json')
        if not source.is_file() or not sidecar.is_file():
            raise ValueError('JPS image requires a complete .lineage.json; choose a new output directory')
        record = json.loads(sidecar.read_text())
        if (record.get('schema_version') != 1 or record.get('attack') != 'JPS'
                or record.get('signature') != signature or record.get('image_sha256') != sha256(source)):
            raise ValueError('JPS artifact/config/input/code fingerprint differs; choose a new output directory')
        for name, wanted in record.get('auxiliary_sha256', {}).items():
            if Path(name).name != name or not (source.parent / name).is_file() or sha256(source.parent / name) != wanted:
                raise ValueError('JPS steer/trace fingerprint differs; choose a new output directory')
        if set(record.get('auxiliary_sha256', {})) != {'jps_adv_steer.txt', 'jps_agent_trace.jsonl'}:
            raise ValueError('JPS artifact requires both the steer prompt and agent trace')
        if len(record.get('optimization', [])) != self.rounds + 1:
            raise ValueError('JPS artifact has an incomplete optimization history')
        with Image.open(source) as image:
            image.verify()
        return record

    def prepare(self, output):
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        signature = self._signature()
        destination = output / 'jps_adv.png'
        source = Path(resolve_path(self.config['adv_image'])) if self.config.get('adv_image') else destination
        if self.config.get('adv_image') or source.exists() or Path(str(source) + '.lineage.json').exists():
            record = self._verify_artifact(source, signature)
            if self.config.get('steer_path') and sha256(resolve_path(self.config['steer_path'])) != record['auxiliary_sha256']['jps_adv_steer.txt']:
                raise ValueError('JPS explicit steer_path differs from the authenticated prompt')
            if source != destination:
                if destination.exists() or Path(str(destination) + '.lineage.json').exists():
                    if self._verify_artifact(destination, signature) != record:
                        raise ValueError('JPS destination differs; choose a new output directory')
                else:
                    for name in ('jps_adv.png', *record['auxiliary_sha256']):
                        shutil.copyfile(source if name == 'jps_adv.png' else source.parent / name, output / name)
                    write_json(str(destination) + '.lineage.json', record)
            reused = True
        else:
            with tempfile.TemporaryDirectory(dir=output, prefix='.jps-') as temporary:
                staging = Path(temporary)
                self._pgd_passes = []
                self._optimize(self._training_corpus(), str(staging))
                names = ('jps_adv.png', 'jps_adv_steer.txt', 'jps_agent_trace.jsonl')
                record = {'schema_version': 1, 'attack': 'JPS', 'signature': signature,
                          'official_commit': self.official_commit, 'optimization': self._pgd_passes,
                          'image_sha256': sha256(staging / 'jps_adv.png'),
                          'auxiliary_sha256': {name: sha256(staging / name) for name in names[1:]}}
                for name in names:
                    os.replace(staging / name, output / name)
                write_json(str(destination) + '.lineage.json', record)
            reused = False
        self._steer = (output / 'jps_adv_steer.txt').read_text(encoding='utf-8')
        return record, reused

    def artifact_fingerprints(self):
        return dict(self._generated_images)

    def generate_test_cases(self, behaviors, save_dir, verbose=False):
        previous = list(self._last_exclusions)
        rows, excluded = self.filter_evaluation_behaviors(self.normalize_behaviors(behaviors))
        output = Path(save_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        with (output / '.JPS.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            record, reused = self.prepare(output / 'images')
            cases = {row['BehaviorID']: ['jps_adv.png', _jps_prompt(row['Behavior'], self._target_for(row), self._steer)] for row in rows}
            self._generated_images = {'images/jps_adv.png': record['image_sha256']}
            metadata = {'schema_version': 1, 'attack': 'JPS', 'official_repo': self.official_repo,
                        'official_commit': self.official_commit, 'algorithm_fidelity': 'adapted',
                        'paper_reproduction': False, 'full_joint_loop': self.rounds > 0,
                        'base_image_provenance': self.config.get('base_image_provenance', 'user_supplied_or_local_substitute; upstream original unavailable'),
                        'agent_model_substituted': self.agent_model != QWEN25_14B_HF_ID,
                        'effective_config': self.effective_config(), 'artifact_reused': reused,
                        'optimization': record['optimization'], 'image_sha256': record['image_sha256'],
                        'auxiliary_sha256': record['auxiliary_sha256'], 'has_steer': bool(self._steer),
                        'training_behavior_ids': self.training_behavior_ids(),
                        'training_instruction_hashes': [instruction_hash(row.instruction) for row in self._training_corpus()],
                        'evaluation_exclusions': excluded or previous, 'n_behaviors': len(rows)}
            write_json(output / 'JPS_test_cases.json', cases)
            write_json(output / 'JPS_metadata.json', metadata)
        if verbose:
            print(f'JPS: {len(rows)} questions; reused={reused}; steer={bool(self._steer)}')
        return cases


JPSAttack = JPS
