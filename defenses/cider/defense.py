"""Unified CIDER detector adapter.

Detection follows PandragonXIII/CIDER: compare harmful-query/image cosine
similarity at diffusion checkpoints 0, 50, ..., 350 and reject whenever a
drop is below the calibrated threshold.
"""

import hashlib
import math
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from ..base import BaseDefense
from .detection import Defender
from .encoder import ReusedLlavaEncoder
from .denoising import generate_denoised_img_diffusion, get_denoiser
from data_paths import REPO_ROOT, resolve_path


OFFICIAL_REPO = "https://github.com/PandragonXIII/CIDER"
OFFICIAL_COMMIT = "3c08d7e2d52e27b47d1e428a030e8ffda5efae3e"
OFFICIAL_REFUSAL = "I'm sorry, but I can not assist."
DEFAULT_DENOISER_PATH = 'data/method_assets/CIDER/checkpoints/256x256_diffusion_uncond.pt'
OFFICIAL_DENOISER_SIZE = 2_211_383_297
OFFICIAL_DENOISER_MD5 = "fd9dd2335b8736d521de0aed54bd90ca"


class CIDERDefense(BaseDefense):
    name = 'CIDER'

    def __init__(self, config=None):
        defaults = {'experiment_profile': 'official', 'default_threshold': -0.003936767578125,
                    'cpnum': 8, 'step': 50, 'cuda': 0,
                    'denoiser_path': DEFAULT_DENOISER_PATH,
                    'expected_denoiser_size': OFFICIAL_DENOISER_SIZE,
                    'verify_denoiser_md5': False}
        super().__init__({**defaults, **(config or {})})
        if type(self.config['cpnum']) is not int or self.config['cpnum'] != 8:
            raise ValueError('Official CIDER requires cpnum=8')
        if type(self.config['step']) is not int or self.config['step'] != 50:
            raise ValueError('Official CIDER requires step=50')
        if not math.isfinite(float(self.config['default_threshold'])):
            raise ValueError('CIDER threshold must be finite')
        if type(self.config['expected_denoiser_size']) is not int or self.config['expected_denoiser_size'] < 0:
            raise ValueError('CIDER expected_denoiser_size must be a nonnegative integer')
        if type(self.config['verify_denoiser_md5']) is not bool:
            raise ValueError('CIDER verify_denoiser_md5 must be boolean')
        device = self.config['cuda']
        if type(device) is int:
            if device < 0:
                raise ValueError('CIDER CUDA index must be nonnegative')
        elif not isinstance(device, str) or not (device == 'cpu' or device == 'cuda' or
                device.startswith('cuda:') and device[5:].isdigit()):
            raise ValueError('CIDER cuda must be a CUDA index, cuda[:index], or cpu')
        self.config['denoiser_path'] = resolve_path(self.config['denoiser_path'])
        self._encoder = None
        self._denoiser = None
        self._checkpoint_info = None
        self._checkpoint_stamp = None

    def validate_target(self, model_config):
        from multimodalmodels.registry import LLAVA_15_HF_ID, resolve_model_id
        if resolve_model_id(model_config['model_name_or_path']) != LLAVA_15_HF_ID:
            raise ValueError('CIDER currently requires llava_v1_5 (LLaVA-1.5-7B)')

    def preflight(self, text, images):
        super().preflight(text, images)
        if len(images) != 1:
            raise ValueError(f'CIDER requires exactly one image; got {len(images)}')
        with Image.open(images[0]) as image:
            image.verify()
        if self.config.get('temp_dir') and not os.path.isdir(self.config['temp_dir']):
            raise FileNotFoundError('CIDER temp_dir must be an existing directory')

    def prepare(self):
        self._verify_checkpoint()

    def metadata(self):
        checkpoint = self._verify_checkpoint()
        files = ['defenses/__init__.py', 'defenses/base.py', 'defenses/inference.py',
                 'generate_completions.py', 'multimodalmodels/loaders.py',
                 'multimodalmodels/targets/llava.py', 'multimodalmodels/llava/llava_model.py']
        files += [str(p.relative_to(REPO_ROOT)) for p in sorted(Path(__file__).parent.rglob('*.py'))]
        return {'name': self.name, 'effective_config': self.config,
                'official_repo': OFFICIAL_REPO, 'official_commit': OFFICIAL_COMMIT,
                'algorithm_fidelity': 'faithful_port', 'paper_reproduction': False,
                'encoder': 'reuse_registered_llava_1_5_7b',
                'implementation_sha256': {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
                                          for name in files},
                'denoiser_checkpoint': checkpoint}

    def apply_input(self, text, images, target):
        self.preflight(text, images)
        return self._apply_path(images[0], text, target)

    def _device(self) -> str:
        value = self.config.get("cuda", 0)
        if isinstance(value, int):
            return f"cuda:{value}"
        return str(value)

    def _denoiser_path(self) -> Path:
        return Path(self.config['denoiser_path'])

    def _get_encoder(self, model):
        if self._encoder is None:
            self._encoder = ReusedLlavaEncoder(model)
        return self._encoder

    def _verify_checkpoint(self):
        checkpoint = self._denoiser_path()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "CIDER diffusion checkpoint is missing: "
                f"{checkpoint}. Run: python -m defenses.cider.setup"
            )
        expected_size = int(
            self.config.get("expected_denoiser_size", OFFICIAL_DENOISER_SIZE)
        )
        if expected_size and checkpoint.stat().st_size != expected_size:
            raise ValueError(
                f"CIDER checkpoint size mismatch: expected {expected_size}, "
                f"got {checkpoint.stat().st_size} ({checkpoint})"
            )
        stat = checkpoint.stat()
        stamp = (str(checkpoint), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._checkpoint_info is None or stamp != self._checkpoint_stamp:
            if self._checkpoint_info is not None and self._denoiser is not None:
                raise RuntimeError('CIDER checkpoint changed after loading the denoiser')
            hasher = hashlib.md5()
            sha = hashlib.sha256()
            with checkpoint.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    hasher.update(chunk)
                    sha.update(chunk)
            digest = hasher.hexdigest()
            self._checkpoint_info = {'path': str(checkpoint), 'size_bytes': stat.st_size,
                                     'md5': digest, 'sha256': sha.hexdigest(),
                                     'matches_official': stat.st_size == OFFICIAL_DENOISER_SIZE and digest == OFFICIAL_DENOISER_MD5}
            self._checkpoint_stamp = stamp
        if self.config.get("verify_denoiser_md5", False):
            digest = self._checkpoint_info['md5']
            if digest != OFFICIAL_DENOISER_MD5:
                raise ValueError(
                    "CIDER checkpoint MD5 mismatch: "
                    f"expected {OFFICIAL_DENOISER_MD5}, got {digest}"
                )
        return dict(self._checkpoint_info)

    def _get_denoiser(self):
        if self._denoiser is not None:
            return self._denoiser
        self._verify_checkpoint()
        self._denoiser = get_denoiser(
            device=self._device(), model_path=str(self._denoiser_path())
        )
        return self._denoiser

    def _apply_path(self, image_path: str, prompt: str, model) -> dict:
        threshold = float(
            self.config.get("default_threshold", -0.003936767578125)
        )
        cpnum = int(self.config.get("cpnum", 8))
        step = int(self.config.get("step", 50))
        if cpnum != 8 or step != 50:
            raise ValueError(
                "Official CIDER uses cpnum=8 and step=50 "
                "(checkpoints 0, 50, ..., 350)."
            )
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"CIDER input image not found: {image_path}")

        encoder = self._get_encoder(model)
        denoiser = self._get_denoiser()
        tmp_parent = self.config.get("temp_dir")
        tmp_dir = tempfile.mkdtemp(prefix="cider_", dir=tmp_parent)
        try:
            denoised_dir = os.path.join(tmp_dir, "denoised_imgs")
            os.makedirs(denoised_dir, exist_ok=True)
            denoised_pths = generate_denoised_img_diffusion(
                imgpth=[image_path],
                save_dir=denoised_dir,
                cps=cpnum,
                step=step,
                device=self._device(),
                batch_size=50,
                denoiser=denoiser,
            )
            denoised_pths.sort()
            cossims = [float(v) for v in encoder.calc_cossim(
                [(prompt, path) for path in denoised_pths]
            )]
            if len(cossims) != cpnum:
                raise RuntimeError(
                    f"CIDER expected {cpnum} similarity checkpoints, "
                    f"got {len(cossims)}."
                )
            if not all(math.isfinite(value) for value in cossims):
                raise RuntimeError('CIDER cosine similarities must be finite')
            detector = Defender(threshold=threshold)
            low_idx = int(detector.get_lowest_idx(cossims))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        deltas = [value - cossims[0] for value in cossims]
        is_adversarial = low_idx != 0
        return {
            "blocked": is_adversarial,
            "text": prompt,
            "images": [image_path],
            "default_response": OFFICIAL_REFUSAL if is_adversarial else None,
            "meta": {
                "defense": "CIDER",
                "detected_adversarial": is_adversarial,
                "threshold": threshold,
                "cpnum": cpnum,
                "step": step,
                "denoise_checkpoints": [i * step for i in range(cpnum)],
                "cosine_similarities": cossims,
                "cosine_deltas": deltas,
                "minimum_cosine_delta": min(deltas),
                "lowest_index": low_idx,
                "denoiser_evaluations": cpnum - 1,
                "image_encoder_forwards": cpnum,
                "text_encoder_forwards": cpnum,
                "detection_reasons": ['cosine_similarity_drop'] if is_adversarial else [],
                "official_repo": OFFICIAL_REPO,
                "official_commit": OFFICIAL_COMMIT,
                "algorithm_fidelity": "faithful_port",
            },
        }

    def apply(self, image_filename, prompt, model=None, image_dir=None, **kwargs):
        if model is None or image_dir is None:
            raise ValueError(
                "CIDERDefense.apply() requires `model`(LLaVA) and `image_dir` "
                "(denoise된 이미지를 인코딩하는 데 필요)."
            )

        src_image_path = (
            image_filename
            if os.path.isabs(image_filename)
            else os.path.join(image_dir, image_filename)
        )
        result = self._apply_path(src_image_path, prompt, model)
        return {
            "blocked": result["blocked"],
            "image_filename": image_filename,
            "prompt": result["text"],
            "default_response": result["default_response"],
            "meta": result["meta"],
        }

    def apply_core(self, test_case, target=None) -> dict:
        if target is None:
            raise ValueError("CIDER.apply_core() requires the loaded LLaVA target.")
        if len(test_case.images) != 1:
            raise ValueError(
                f"CIDER's injection-mode adapter requires exactly one image; "
                f"got {len(test_case.images)}."
            )
        image_path = resolve_path(test_case.images[0])
        result = self._apply_path(image_path, test_case.text, target)
        # The paper specifies that a passing pair reaches the MLLM unchanged.
        result["images"] = list(test_case.images)
        return result
