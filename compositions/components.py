"""Transform adapters reusing the attack implementations already installed here."""
import copy
import hashlib
from pathlib import Path

from PIL import Image

from benchmark_datasets import portable_path
from data_paths import resolve_path
from .datasets import scoring_row
from .engine import CompositionComponent, ComponentResult


class AttackComponent(CompositionComponent):
    def __init__(self, stage_spec, **kwargs):
        super().__init__(stage_spec, **kwargs)
        name = stage_spec['component']
        from multi_attacks.figstep.figstep import FigStep
        from multi_attacks.cs_dj.cs_dj import CSDJAttack
        from multi_attacks.mml.mml import MML
        from multi_attacks.jps.jps import JPS
        self.component_name = name
        self.source_attack, cls = {
            'figstep_render': ('FigStep', FigStep), 'mml_layout': ('MML', MML),
            'cs_dj_split': ('CS-DJ', CSDJAttack), 'cs_dj_full': ('CS-DJ', CSDJAttack),
            'jps_joint': ('JPS', JPS),
        }[name]
        if self.source_attack == 'MML' and self.config.get('input_mode', 'generated') != 'generated':
            raise ValueError('MML composition needs generated mode to consume the preceding text')
        self.attack = cls(**self.config)
        self._training_hashes = None

    def preflight(self):
        if self.component_name == 'cs_dj_full':
            self.attack._sample_library()
            self.attack._font(self.attack.text_font_size)
        if self.component_name == 'figstep_render':
            # Resolve and check the actual font using the existing renderer.
            self.attack._render('Preflight')
        if self.component_name == 'mml_layout':
            self.attack._font()

    def prepare(self, out_dir):
        super().prepare(out_dir)
        if self.component_name == 'cs_dj_full':
            self.attack.prepare(out_dir)

    def evaluation_exclusion_reason(self, behavior):
        if self.component_name == 'jps_joint':
            from image_attacks.jp_common import instruction_hash
            if self._training_hashes is None:
                self._training_hashes = {instruction_hash(b.instruction) for b in self.attack._training_corpus()}
            if self.attack.exclude_train_from_eval and instruction_hash(behavior.instruction) in self._training_hashes:
                return 'JPS training question'
        return None

    def transform(self, artifact, behavior, out_dir):
        if self.component_name == 'jps_joint':
            from multi_attacks.jps.jps_optimization import _affirmative_target, _jps_prompt
            directory = Path(self.prepare_out_dir) / 'images'
            self.attack.prepare(directory)
            text = _jps_prompt(artifact.text, behavior.target or _affirmative_target(artifact.text), self.attack._steer)
            return ComponentResult(artifact.with_payload(text=text, images=[portable_path(directory / 'jps_adv.png')]),
                                   iteration_result={'optimization_scope': 'once_per_component_config'})
        if self.component_name == 'cs_dj_split':
            questions = self.attack._split_question(artifact.text, behavior.behavior_id)
            if len(questions) != 3 or any(not q.strip() for q in questions):
                raise ValueError('CS-DJ split needs three nonempty subquestions')
            return ComponentResult(artifact.with_payload(text='\n'.join(
                f'{i}. {q}' for i, q in enumerate(questions, 1))),
                iteration_result={'subquestions_generated': 3})
        row = scoring_row(behavior)
        row.update(Behavior=artifact.text, statement=artifact.text, images=list(artifact.images or behavior.images))
        row['meta'].pop('images_by_type', None)
        directory = Path(out_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if self.component_name in ('cs_dj_full', 'mml_layout'):
            # Both ports retain build(); use their own record converters where available.
            current = self.attack._behavior_record(row) if self.component_name == 'cs_dj_full' else self.attack.behavior(row)
            cases = self.attack.build(current, str(directory))
            index = int(self.stage_spec.get('output_index', 0))
            if not 0 <= index < len(cases):
                raise ValueError('Composition output_index is outside the attack outputs')
            case = cases[index]
            return ComponentResult(artifact.with_payload(text=case.text, images=case.images),
                                   metadata={'attack_case_meta': copy.deepcopy(case.meta)},
                                   iteration_result={'outputs_built': len(cases), 'selected_output_index': index})
        outputs = self.attack.generate_test_cases([row], str(directory))
        pair = outputs[behavior.behavior_id]
        image, text = pair
        reference = Path(image)
        if not reference.is_absolute():
            reference = directory / 'images' / reference
        return ComponentResult(artifact.with_payload(text=text, images=[portable_path(reference)]),
                               iteration_result={'outputs_built': 1})


class JOODAugComponent(CompositionComponent):
    component_name = 'jood_aug'
    source_attack = 'JOOD'

    def __init__(self, stage_spec, **kwargs):
        super().__init__(stage_spec, **kwargs)
        self.aug = str(self.config.get('aug', 'randaug1'))
        if not self.aug.startswith('randaug'):
            raise ValueError('jood_aug requires randaugN; use five_methods for native-resolution MixUp')
        self.n = int(self.aug.removeprefix('randaug'))
        self.strengths = [float(x) for x in self.config.get('lams', [.5])]
        if not 1 <= self.n <= 14 or not self.strengths or any(not 0 <= x <= 1 for x in self.strengths):
            raise ValueError('Invalid JOOD composition strengths or operation count')
        self.seed = int(self.config.get('seed') or 0)

    def transform(self, artifact, behavior, out_dir):
        from multi_attacks.JOOD.jood_utils import RandAug
        images = artifact.images or behavior.images
        if not images:
            raise ValueError('JOOD stage needs the preceding image')
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        candidates, seeds = [], []
        for i, strength in enumerate(self.strengths):
            tag = f'{behavior.behavior_id}\0{self.aug}\0{strength}\0{self.seed}'
            seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], 'big')
            path = directory / f'{behavior.behavior_id}_{i:02d}.png'
            RandAug(self.n, strength).apply(resolve_path(images[0]), seed).convert('RGB').save(path)
            candidates.append(portable_path(path)); seeds.append(seed)
        tag = f'{behavior.behavior_id}\0{self.component_name}\0{self.seed}'
        selected = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], 'big') % len(candidates)
        return ComponentResult(artifact.with_payload(images=[candidates[selected]]),
            iteration_plan={'augmentations': len(self.strengths), 'randaug_n': self.n},
            iteration_result={'augmentations_completed': len(candidates), 'selected_index': selected},
            metadata={'candidate_images': candidates, 'candidate_seeds': seeds,
                      'selection_policy': 'deterministic_hash_not_asr'})


SUPPORTED_COMPONENTS = {'figstep_render', 'mml_layout', 'cs_dj_split', 'cs_dj_full', 'jood_aug', 'jps_joint'}


def make_component(spec):
    name = spec.get('component')
    if name not in SUPPORTED_COMPONENTS:
        raise ValueError(f'Composition component {name!r} needs an attack/stage that has not been ported to ai-resilience; available: {sorted(SUPPORTED_COMPONENTS)}')
    return JOODAugComponent(spec) if name == 'jood_aug' else AttackComponent(spec)
