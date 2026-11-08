"""VLMShield input-filter adapter for the unified pipeline.

VLMShield (Qi et al., arXiv:2604.06502; pgqihere/VLMShield@344ca14) classifies
a (text, image) pair as safe / unsafe from MAFE features: CLIP ViT-L/14 image
[CLS] embedding (768) + chunked, similarity-weighted text [EOS] embedding (768)
-> released 3-layer MLP.  It never queries the target model, so a pair that is
predicted unsafe is blocked with a fixed refusal and a passing pair reaches the
target unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
from typing import Any

from ..base import BaseDefense
from data_paths import REPO_ROOT, resolve_path
from .features import (
    CLIP_EMBED_DIM,
    EMBEDDING_DIM,
    ThreeLayerClassifier,
    classify,
    multimodal_embedding,
)


OFFICIAL_REPO = "https://github.com/pgqihere/VLMShield"
OFFICIAL_COMMIT = "344ca14778383aef57315229118f7669454fb4bb"
OFFICIAL_CLIP_MODEL = "openai/clip-vit-large-patch14"
OFFICIAL_CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
OFFICIAL_CLASSIFIER_NAME = "vlmshield_classifier.pt"
OFFICIAL_CLASSIFIER_SHA256 = (
    "b36702877791f347f0b846d93b9b9162c86a8895efe962b6052d322ed0c53c34"
)
OFFICIAL_CLASSIFIER_SIZE = 8401604
DEFAULT_CLASSIFIER_PATH = Path('data/method_assets/VLMShield/checkpoints') / OFFICIAL_CLASSIFIER_NAME
# Upstream only computes detection rates; the refusal text that a blocked pair
# receives in the simulator is the same fixed sentence the JailGuard adapter uses.
DEFAULT_REFUSAL = "I'm sorry, but I cannot assist with that request."
LABEL_MEANING = {0: "unsafe", 1: "safe"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VLMShieldDefense(BaseDefense):
    """Official-profile VLMShield detector (CLIP ViT-L/14 + released MLP)."""

    name = 'VLMShield'

    def __init__(self, config: dict[str, Any] | None = None):
        defaults = {'experiment_profile': 'official', 'clip_model': OFFICIAL_CLIP_MODEL,
                    'clip_revision': OFFICIAL_CLIP_REVISION,
                    'classifier_path': str(DEFAULT_CLASSIFIER_PATH),
                    'expected_classifier_sha256': OFFICIAL_CLASSIFIER_SHA256,
                    'refusal_response': DEFAULT_REFUSAL, 'store_embedding': False}
        super().__init__({**defaults, **(config or {})})
        if self.config['clip_model'] != OFFICIAL_CLIP_MODEL:
            raise ValueError(f'Official VLMShield requires {OFFICIAL_CLIP_MODEL}')
        if not isinstance(self.config['refusal_response'], str):
            raise ValueError('VLMShield refusal_response must be a string')
        if type(self.config['store_embedding']) is not bool:
            raise ValueError('VLMShield store_embedding must be boolean')
        self._clip = None
        self._processor = None
        self._classifier = None
        self._classifier_digest: str | None = None
        self._device = None
        self._classifier_asset = None
        self._classifier_signature = None
        self._clip_asset = None

    def preflight(self, text, images):
        super().preflight(text, images)
        if len(images) > 1:
            raise ValueError(f'VLMShield accepts zero or one image; got {len(images)}')
        from PIL import Image
        for path in images:
            with Image.open(path) as image:
                image.verify()
        self._check_classifier()

    def prepare(self):
        # Validate the small classifier before allocating the CLIP encoder.
        self._get_classifier()
        self._get_clip()

    def metadata(self):
        self.prepare()
        import torch
        files = ['defenses/__init__.py', 'defenses/base.py', 'defenses/inference.py',
                 'generate_completions.py', 'multimodalmodels/loaders.py', 'multimodalmodels/registry.py']
        files += [str(p.relative_to(REPO_ROOT)) for p in sorted(Path(__file__).parent.glob('*.py'))]
        return {'name': self.name, 'effective_config': self.config,
                'detector': True, 'official_repo': OFFICIAL_REPO, 'official_commit': OFFICIAL_COMMIT,
                'algorithm_fidelity': 'faithful_port', 'paper_reproduction': False,
                'classifier': self._check_classifier(), 'clip': self._clip_provenance(),
                'embedding_dim': EMBEDDING_DIM,
                'label_meaning': {str(key): value for key, value in LABEL_MEANING.items()},
                'decision_rule': 'softmax_argmax; 0=unsafe, 1=safe; ties choose 0',
                'device': str(self._get_device()), 'dtype': 'float32',
                'versions': {name: importlib.metadata.version(name)
                             for name in ('torch', 'transformers', 'Pillow', 'huggingface-hub')},
                'float32_matmul_precision': torch.get_float32_matmul_precision(),
                'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'implementation_sha256': {name: _sha256_file(REPO_ROOT / name) for name in files}}

    def apply_input(self, text, images, target):
        self.preflight(text, images)
        result = self._apply_pair(text, images[0] if images else None)
        result['images'] = list(images)
        return result

    def _check_classifier(self):
        path = self._resolve_path(self.config['classifier_path'])
        if not path.is_file():
            raise FileNotFoundError(f'VLMShield classifier not found: {path}. Run: python -m defenses.vlmshield.setup')
        stat = path.stat()
        signature = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._classifier_asset is not None and signature == self._classifier_signature:
            return dict(self._classifier_asset)
        if self._classifier is not None:
            raise RuntimeError('VLMShield classifier changed after loading; create a new defense')
        digest = _sha256_file(path)
        expected = self.config['expected_classifier_sha256']
        if expected and digest != expected:
            raise ValueError(f'VLMShield classifier SHA-256 mismatch: expected {expected}, got {digest} ({path})')
        self._classifier_asset = {'path': str(path), 'sha256': digest, 'size_bytes': stat.st_size,
                                  'matches_official': digest == OFFICIAL_CLASSIFIER_SHA256}
        self._classifier_signature = signature
        return dict(self._classifier_asset)

    def _clip_provenance(self):
        if self._clip_asset is None:
            from huggingface_hub import snapshot_download
            revision = self.config.get('clip_revision') or OFFICIAL_CLIP_REVISION
            directory = Path(snapshot_download(OFFICIAL_CLIP_MODEL, revision=revision, local_files_only=True))
            names = ['config.json', 'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json',
                     'special_tokens_map.json', 'merges.txt', 'vocab.json']
            # Transformers prefers safetensors; record the actual available weight format.
            weights = 'model.safetensors' if (directory / 'model.safetensors').is_file() else 'pytorch_model.bin'
            names.append(weights)
            self._clip_asset = {'model': OFFICIAL_CLIP_MODEL, 'revision': directory.name,
                'matches_official_revision': directory.name == OFFICIAL_CLIP_REVISION,
                'files': {name: {'sha256': _sha256_file(directory / name),
                                 'size_bytes': (directory / name).stat().st_size} for name in names}}
        return self._clip_asset

    # ------------------------------------------------------------ resources
    def _resolve_path(self, value: str | os.PathLike[str]) -> Path:
        path = Path(os.path.expandvars(os.path.expanduser(os.fspath(value))))
        if path.is_absolute():
            return path
        if path.parts and path.parts[0] == 'data':
            return Path(resolve_path(path))
        config_path = self.config.get("_config_path") or self.config_path
        if config_path:
            return (Path(config_path).resolve().parent / path).resolve()
        return (REPO_ROOT / path).resolve()

    def _get_device(self):
        if self._device is None:
            import torch

            configured = self.config.get("device")
            if configured is None:
                configured = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._device = torch.device(str(configured))
        return self._device

    def _get_clip(self):
        if self._clip is None:
            from multimodalmodels.loaders import load_clip_model

            model_id = str(self.config.get("clip_model", OFFICIAL_CLIP_MODEL))
            if model_id != OFFICIAL_CLIP_MODEL:
                raise ValueError(
                    f"Official VLMShield uses {OFFICIAL_CLIP_MODEL}; got {model_id!r}."
                )
            self._clip, self._processor = load_clip_model(
                model_id,
                revision=self.config.get("clip_revision"),
                device=self._get_device(),
            )
        return self._clip, self._processor

    def _get_classifier(self):
        asset = self._check_classifier()
        if self._classifier is not None:
            return self._classifier
        import torch

        classifier = ThreeLayerClassifier(dim=EMBEDDING_DIM)
        state = torch.load(asset['path'], map_location="cpu", weights_only=True)
        classifier.load_state_dict(state)
        self._classifier = classifier.to(self._get_device()).eval()
        self._classifier_digest = asset['sha256']
        return self._classifier

    # ------------------------------------------------------------ detection
    def _predict(self, text: str, image_path: str | None):
        """Return (prediction, confidence, probs, has_image, has_text, embedding)."""
        from PIL import Image

        clip_model, processor = self._get_clip()
        classifier = self._get_classifier()
        image = None
        if image_path:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"VLMShield input image not found: {image_path}")
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
        embedding, has_image, has_text = multimodal_embedding(
            clip_model, processor, text, image, self._get_device()
        )
        prediction, confidence, probs = classify(classifier, embedding)
        return prediction, confidence, probs, has_image, has_text, embedding

    def _apply_pair(self, text: str, image_path: str | None) -> dict[str, Any]:
        prediction, confidence, probs, has_image, has_text, embedding = self._predict(
            text, image_path
        )
        if (prediction not in (0, 1) or not math.isfinite(confidence)
                or len(probs) != 2 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs)):
            raise RuntimeError('VLMShield returned invalid classifier probabilities')
        blocked = prediction == 0
        refusal = str(self.config.get("refusal_response", DEFAULT_REFUSAL))
        meta: dict[str, Any] = {
            "defense": "VLMShield",
            "detected_adversarial": blocked,
            "prediction": prediction,
            "prediction_label": LABEL_MEANING[prediction],
            "confidence": confidence,
            "probabilities": {"unsafe": probs[0], "safe": probs[1]},
            "has_image": has_image,
            "has_text": has_text,
            "clip_model": OFFICIAL_CLIP_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "classifier_sha256": self._classifier_digest,
            "query_budget": 0,
            "detector_score": probs[0],
            "detector_score_meaning": "unsafe softmax probability; decision uses argmax",
            "source_image_sha256": _sha256_file(Path(image_path)) if image_path else None,
            "source_text_sha256": hashlib.sha256(text.encode('utf-8')).hexdigest(),
            "official_repo": OFFICIAL_REPO,
            "official_commit": OFFICIAL_COMMIT,
            "algorithm_fidelity": "faithful_port",
        }
        if self.config.get("store_embedding", False):
            meta["embedding"] = [float(v) for v in embedding.detach().cpu().tolist()]
        return {
            "blocked": blocked,
            "text": text,
            "images": [image_path] if image_path else [],
            "default_response": refusal if blocked else None,
            "meta": meta,
        }

    # ------------------------------------------------------------ interfaces
    def apply(self, image_filename, prompt, model=None, image_dir=None, **kwargs):
        image_path = None
        if image_filename:
            image_path = (
                image_filename
                if os.path.isabs(image_filename) or image_dir is None
                else os.path.join(image_dir, image_filename)
            )
        result = self._apply_pair(prompt, image_path)
        return {
            "blocked": result["blocked"],
            "image_filename": image_filename,
            "prompt": result["text"],
            "default_response": result["default_response"],
            "meta": result["meta"],
        }

    def apply_core(self, test_case, target=None) -> dict[str, Any]:
        if len(test_case.images) > 1:
            raise ValueError(
                "VLMShield's adapter scores one image per query (zero images is "
                f"the text-only path); got {len(test_case.images)}."
            )
        image_path = None
        if test_case.images:
            image_path = resolve_path(test_case.images[0])
        result = self._apply_pair(test_case.text, image_path)
        # A passing pair reaches the downstream target unchanged.
        result["images"] = list(test_case.images)
        return result


__all__ = [
    "VLMShieldDefense",
    "OFFICIAL_REPO",
    "OFFICIAL_COMMIT",
    "OFFICIAL_CLIP_MODEL",
    "OFFICIAL_CLASSIFIER_SHA256",
    "DEFAULT_REFUSAL",
    "CLIP_EMBED_DIM",
]
