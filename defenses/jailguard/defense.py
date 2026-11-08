"""JailGuard image detector adapted to the simulator's unified pipeline.

The detector follows the official image runner: create eight image variants
with policy mutator ``PL``, keep the question unchanged, query the target for
every variant, and detect unstable answers using spaCy similarity and pairwise
KL divergence.
"""

import hashlib
import math
import os
import shutil
import tempfile

from PIL import Image

from ..base import BaseDefense
from .image_mutations import get_image_mutator
from .utils import detect_attack, update_divergence


OFFICIAL_REPO = "https://github.com/shiningrain/JailGuard"
OFFICIAL_COMMIT = "419e1d3d08f4d02998a11a687f925518d2d5dd0c"
OFFICIAL_IMAGE_MUTATOR = "PL"
OFFICIAL_IMAGE_VARIANTS = 8
OFFICIAL_IMAGE_THRESHOLD = 0.025
OFFICIAL_TOP_CHARS = 100
OFFICIAL_VARIANT_MAX_NEW_TOKENS = 300
DEFAULT_REFUSAL = "I'm sorry, but I cannot assist with that request."


class JailGuardDefense(BaseDefense):
    name = 'JailGuard'

    def __init__(self, config=None):
        defaults = {'modality': 'image', 'default_mutator': OFFICIAL_IMAGE_MUTATOR,
                    'default_number': OFFICIAL_IMAGE_VARIANTS, 'default_threshold': OFFICIAL_IMAGE_THRESHOLD,
                    'spacy_model': 'en_core_web_md', 'top_string': OFFICIAL_TOP_CHARS,
                    'variant_max_new_tokens': OFFICIAL_VARIANT_MAX_NEW_TOKENS,
                    'store_variant_responses': False, 'refusal_response': DEFAULT_REFUSAL}
        super().__init__({**defaults, **(config or {})})
        if self.config['modality'] != 'image':
            raise ValueError('JailGuard implements the image modality only')
        for key, minimum in (('default_number', 2), ('top_string', 1), ('variant_max_new_tokens', 1)):
            if type(self.config[key]) is not int or self.config[key] < minimum:
                raise ValueError(f'JailGuard {key} must be an integer >= {minimum}')
        if self.config['variant_max_new_tokens'] != OFFICIAL_VARIANT_MAX_NEW_TOKENS:
            raise ValueError('JailGuard variant_max_new_tokens must be 300')
        if not math.isfinite(float(self.config['default_threshold'])):
            raise ValueError('JailGuard threshold must be finite')
        if type(self.config['store_variant_responses']) is not bool:
            raise ValueError('JailGuard store_variant_responses must be boolean')
        if not isinstance(self.config['refusal_response'], str) or not self.config['refusal_response']:
            raise ValueError('JailGuard refusal_response must be a nonempty string')
        self.config['default_mutator'] = str(self.config['default_mutator']).upper()
        get_image_mutator(self.config['default_mutator'])
        self._metric = None
        self._metadata = None

    def preflight(self, text, images):
        super().preflight(text, images)
        if len(images) != 1:
            raise ValueError(f'JailGuard requires exactly one image; got {len(images)}')
        with Image.open(images[0]) as image:
            image.verify()
        if self.config.get('temp_dir') and not os.path.isdir(self.config['temp_dir']):
            raise FileNotFoundError('JailGuard temp_dir must be an existing directory')

    def prepare(self):
        self._get_metric()

    def metadata(self):
        if self._metadata is None:
            import importlib.metadata
            from pathlib import Path
            from data_paths import REPO_ROOT
            from image_attacks.jp_common import sha256
            metric = self._get_metric()
            resources = {}
            if getattr(metric, 'path', None):
                root = Path(metric.path)
                resources = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
                             if p.is_file() and '__pycache__' not in p.parts}
            files = ['defenses/__init__.py', 'defenses/base.py', 'defenses/inference.py',
                     'defenses/jailguard/__init__.py', 'defenses/jailguard/defense.py',
                     'defenses/jailguard/image_mutations.py', 'defenses/jailguard/utils.py',
                     'defenses/jailguard/similarity.py',
                     'generate_completions.py']
            self._metadata = {'name': 'JailGuard', 'effective_config': self.config,
                'official_repo': OFFICIAL_REPO, 'official_commit': OFFICIAL_COMMIT,
                'algorithm_fidelity': 'faithful_port', 'paper_reproduction': False,
                'implementation_sha256': {name: sha256(REPO_ROOT / name) for name in files},
                'metric': {'spacy_version': importlib.metadata.version('spacy'),
                           'model_meta': getattr(metric, 'meta', {}),
                           'model_resource_sha256': resources,
                           'official_spacy_version': '3.7.2', 'official_model_version': '3.7.1'}}
        return self._metadata

    def apply_input(self, text, images, target):
        self.preflight(text, images)
        return self._apply_path(images[0], text, target)

    def _get_metric(self):
        if self._metric is None:
            model_name = self.config.get("spacy_model", "en_core_web_md")
            try:
                from multimodalmodels.loaders import load_spacy

                self._metric = load_spacy(model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"JailGuard requires the spaCy model {model_name!r}. "
                    "Install the configured spaCy model; see defenses/jailguard/README.md"
                ) from exc
        return self._metric

    def _generate_variant_responses(self, prompt, image_paths, target):
        """Query variants serially, as in the official ``main_img.py``."""
        from multimodalmodels.targets.base import TargetModel

        if isinstance(target, TargetModel):
            old_max_tokens = getattr(target, "max_new_tokens", None)
            changed_max_tokens = hasattr(target, "max_new_tokens")
            if changed_max_tokens:
                target.max_new_tokens = OFFICIAL_VARIANT_MAX_NEW_TOKENS
            try:
                return [str(target.generate(prompt, [path])) for path in image_paths]
            finally:
                if changed_max_tokens:
                    target.max_new_tokens = old_max_tokens

        # Compatibility with the repository's pre-unified model wrappers.
        test_cases = [[os.path.basename(path), prompt] for path in image_paths]
        outputs = target.generate(
            test_cases=test_cases,
            image_dir=os.path.dirname(image_paths[0]),
            num_beams=1,
            temperature=1.0,
            max_new_tokens=OFFICIAL_VARIANT_MAX_NEW_TOKENS,
            progress_bar=False,
        )
        return [str(output) for output in outputs]

    def _apply_path(self, image_path: str, prompt: str, target) -> dict:
        if target is None:
            raise ValueError("JailGuard requires a loaded target model.")
        if self.config.get("modality", "image") != "image":
            raise ValueError("This VLM adapter implements official image JailGuard only.")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"JailGuard input image not found: {image_path}")

        mutator_name = str(
            self.config.get("default_mutator", OFFICIAL_IMAGE_MUTATOR)
        ).upper()
        number = int(self.config.get("default_number", OFFICIAL_IMAGE_VARIANTS))
        threshold = float(
            self.config.get("default_threshold", OFFICIAL_IMAGE_THRESHOLD)
        )
        top_string = int(self.config.get("top_string", OFFICIAL_TOP_CHARS))
        variant_max_tokens = int(
            self.config.get(
                "variant_max_new_tokens", OFFICIAL_VARIANT_MAX_NEW_TOKENS
            )
        )
        if number < 2:
            raise ValueError("JailGuard needs at least two mutated variants.")
        if top_string < 1:
            raise ValueError("JailGuard top_string must be positive.")
        if variant_max_tokens != OFFICIAL_VARIANT_MAX_NEW_TOKENS:
            raise ValueError(
                "Official JailGuard image inference uses max_new_tokens=300."
            )

        image_mutator = get_image_mutator(mutator_name)
        tmp_dir = tempfile.mkdtemp(
            prefix="jailguard_", dir=self.config.get("temp_dir")
        )
        variant_paths = []
        try:
            with Image.open(image_path) as opened:
                original = opened.convert("RGB")
                for index in range(number):
                    # Some official mutations operate in-place, so each one
                    # receives a fresh copy of the unmodified source image.
                    variant = image_mutator(original.copy())
                    variant_path = os.path.join(
                        tmp_dir, f"{index}-{mutator_name}.jpg"
                    )
                    variant.convert("RGB").save(variant_path, format="JPEG")
                    variant_paths.append(variant_path)

            responses = self._generate_variant_responses(
                prompt, variant_paths, target
            )
            if len(responses) != number:
                raise RuntimeError(
                    f"JailGuard expected {number} variant responses, "
                    f"got {len(responses)}."
                )
            max_div, all_block, details = update_divergence(
                responses,
                metric=self._get_metric(),
                top_string=top_string,
                return_details=True,
            )
            if not math.isfinite(float(max_div)) or any(
                not math.isfinite(float(value))
                for matrix in details.values() for row in matrix for value in row
            ):
                raise RuntimeError('JailGuard similarity/divergence must be finite')
            detected = bool(detect_attack(max_div, all_block, threshold))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        reasons = []
        if max_div > threshold:
            reasons.append("response_divergence")
        if all_block[0]:
            reasons.append("all_variants_refused")

        target_name = str(
            getattr(target, "name", None) or target.__class__.__name__
        )
        normalized_target = target_name.lower().replace("-", "").replace("_", "")
        meta = {
            "defense": "JailGuard",
            "modality": "image",
            "mutator": mutator_name,
            "number": number,
            "query_budget": number,
            "threshold": threshold,
            "top_string": top_string,
            "variant_max_new_tokens": variant_max_tokens,
            "max_divergence": float(max_div),
            "all_variants_refused": bool(all_block[0]),
            "minimum_refusal_keywords": int(all_block[1]),
            "detected_adversarial": detected,
            "detection_reasons": reasons,
            "similarity_matrix": details["similarity_matrix"],
            "divergence_matrix": details["divergence_matrix"],
            "variant_response_sha256": [
                hashlib.sha256(response.encode("utf-8")).hexdigest()
                for response in responses
            ],
            "target_model": target_name,
            "official_target_model": "MiniGPT-4",
            "substitutes_target_model": "minigpt4" not in normalized_target,
            "official_repo": OFFICIAL_REPO,
            "official_commit": OFFICIAL_COMMIT,
            "algorithm_fidelity": "faithful_port",
        }
        if self.config.get("store_variant_responses", False):
            meta["variant_responses"] = responses

        refusal = self.config.get("refusal_response", DEFAULT_REFUSAL)
        return {
            "blocked": detected,
            "text": prompt,
            "images": [image_path],
            "default_response": refusal if detected else None,
            "meta": meta,
        }

    def apply(self, image_filename, prompt, model=None, image_dir=None, **kwargs):
        if model is None or image_dir is None:
            raise ValueError(
                "JailGuardDefense.apply() requires `model` and `image_dir`."
            )
        image_path = (
            image_filename
            if os.path.isabs(image_filename)
            else os.path.join(image_dir, image_filename)
        )
        result = self._apply_path(image_path, prompt, model)
        return {
            "blocked": result["blocked"],
            "image_filename": image_filename,
            "prompt": result["text"],
            "default_response": result["default_response"],
            "meta": result["meta"],
        }

    def apply_core(self, test_case, target=None) -> dict:
        if target is None:
            raise ValueError("JailGuard.apply_core() requires a loaded target model.")
        if len(test_case.images) != 1:
            raise ValueError(
                "JailGuard's image adapter requires exactly one image; "
                f"got {len(test_case.images)}."
            )
        from data_paths import resolve_path

        result = self._apply_path(
            resolve_path(test_case.images[0]), test_case.text, target
        )
        # A passing pair reaches the downstream target unchanged.
        result["images"] = list(test_case.images)
        return result
