"""Apply a selected defense around a single shared target instance."""
import time
from pathlib import Path

from multimodalmodels.inference import prepare_case
from multimodalmodels.targets import get_target_factory, load_target, validate_image_count
from multimodalmodels.targets.base import TargetModel


class TrackedTarget(TargetModel):
    """Transparent target wrapper retaining mutation-query/token accounting."""
    def __init__(self, target):
        self.target = target
        self.calls = []
        self.min_images = target.min_images
        self.max_images = target.max_images

    @property
    def name(self):
        return self.target.name

    @property
    def max_new_tokens(self):
        return self.target.max_new_tokens

    @max_new_tokens.setter
    def max_new_tokens(self, value):
        self.target.max_new_tokens = value

    def __getattr__(self, name):
        return getattr(self.target, name)

    def generate(self, text, images):
        start = time.perf_counter()
        budget = getattr(self.target, 'max_new_tokens', None)
        response = self.target.generate(text, images)
        if not isinstance(response, str):
            raise RuntimeError('Defense target must return a string response')
        self.calls.append({'max_new_tokens': budget, 'seconds': time.perf_counter() - start})
        return response


def preflight_cases(model_config, cases, test_cases_path, defense):
    name = model_config['model_name_or_path']
    get_target_factory(name)
    defense.validate_target(model_config)
    image_dir = Path(test_cases_path).resolve().parent / 'images'
    prepared = [prepare_case(case, image_dir) for case in cases]
    for text, images in prepared:
        validate_image_count(name, len(images))
        defense.preflight(text, images)
    return prepared


def run_one(defense, target, text, images):
    tracked = TrackedTarget(target)
    start = time.perf_counter()
    decision = defense.apply_input(text, images, tracked)
    defense_seconds = time.perf_counter() - start
    if not isinstance(decision, dict) or type(decision.get('blocked')) is not bool:
        raise RuntimeError('Defense must return a boolean blocked decision')
    variant_calls = len(tracked.calls)
    blocked = decision['blocked']
    if blocked:
        response = decision.get('default_response')
        if response is None:
            response = decision.get('response')
    elif decision.get('response') is not None:
        response = decision['response']
    else:
        response = tracked.generate(decision.get('text', text), decision.get('images', images))
    if not isinstance(response, str):
        raise RuntimeError('Defense must supply a string response when it blocks')
    meta = dict(decision.get('meta', {}))
    meta.update(defense=defense.name, blocked=blocked,
                target_calls=len(tracked.calls), defense_target_calls=variant_calls,
                final_target_calls=len(tracked.calls) - variant_calls,
                target_call_max_new_tokens=[call['max_new_tokens'] for call in tracked.calls],
                defense_seconds=defense_seconds, total_seconds=time.perf_counter() - start,
                paper_reproduction=False)
    return {'generation': response, 'blocked': blocked, 'meta': meta}


def load_defended_generation_function(model_config, max_new_tokens, test_cases_path, defense):
    target = None

    def generate(test_cases, progress_bar=False):
        nonlocal target
        prepared = preflight_cases(model_config, test_cases, test_cases_path, defense)
        if not prepared:
            return []
        defense.prepare()
        if target is None:
            target = load_target(model_config['model_name_or_path'], max_new_tokens=max_new_tokens)
            for key, value in model_config.get('generation', {}).items():
                if key not in {'do_sample', 'temperature', 'top_p', 'retry_limit'}:
                    raise ValueError(f'Unsupported generation option: {key!r}')
                if not hasattr(target, key):
                    raise ValueError(f'Target does not support generation option: {key!r}')
                setattr(target, key, value)
        if progress_bar:
            from tqdm import tqdm
            prepared = tqdm(prepared, desc=defense.name)
        return [run_one(defense, target, text, images) for text, images in prepared]

    return generate
