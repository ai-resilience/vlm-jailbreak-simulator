"""Differentiable CE model interfaces ported from our shared surrogate code.

The public input is raw RGB in [0, 1]; processors perform model-specific
normalization and patchification without detaching the optimized pixels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class _LlavaCE:
    """White-box backend: wraps the multimodalmodels LLaVA surrogate."""

    backend = "llava"

    def __init__(self, model_name: str = "LLaVA_v1_5", revision=None):
        import multimodalmodels
        model_cls = getattr(multimodalmodels, model_name)
        self.model = model_cls(revision=revision) if model_name == "LLaVA_v1_5" else model_cls()
        self.device = self.model.device
        self.model.model.requires_grad_(False)

    def _mean_std(self, device):
        mean = torch.tensor(_CLIP_MEAN, device=device)[None, :, None, None]
        std = torch.tensor(_CLIP_STD, device=device)[None, :, None, None]
        return mean, std

    def to_unit01(self, pil: Image.Image) -> torch.Tensor:
        pv = self.model.vis_processor(pil, return_tensors="pt")["pixel_values"].to(self.device)
        mean, std = self._mean_std(self.device)
        return (pv * std + mean).clamp(0, 1)  # [1,3,H,W] in [0,1]

    def compute_loss_batch(self, img01: torch.Tensor, targets, pils=None,
                           prompts=None) -> torch.Tensor:
        if prompts is not None and len(prompts) != len(targets):
            raise ValueError("prompts and targets must have the same length")
        mean, std = self._mean_std(img01.device)
        x_adv = (img01 - mean) / std  # LLaVA wrapper expects CLIP-normalized pixels
        # The wrapper needs a PIL per target only for tokenization; content is irrelevant.
        if pils is None:
            raise ValueError("LLaVA backend needs a PIL (or list) for tokenization")
        return self.model.compute_loss_batch(x_adv, targets, pils, prompts=prompts)

    def zero_grad(self):
        self.model.model.zero_grad()


class _NativeHFCE:
    """Teacher-forced CE over a modern native-HF multimodal chat model.

    One raw RGB tensor is optimized and passed through the model's *fast torch*
    image processor on every step.  Consequently every Qwen patch, InternVL
    tile, or LLaVA-NeXT AnyRes crop remains a coherent differentiable function
    of the PNG that the attack eventually saves.  Metadata and token counts are
    produced from an identically shaped detached tensor and checked against the
    live differentiable preprocessing result.
    """

    _SIZES = {
        "llava-next": (336, 336),
        "qwen2": (336, 336),
        "qwen3-vl": (384, 384),  # divisible by Qwen3's patch*merge factor (32)
        "internvl": (448, 448),
    }

    def __init__(self, backend: str, model_id: str, revision=None):
        from multimodalmodels.loaders import load_native_vlm

        self.backend = backend
        self.model_id = model_id
        self.revision = revision
        self.image_size = self._SIZES[backend]
        if backend == "qwen2":
            from multimodalmodels.loaders import load_qwen2vl
            self.model, self.processor = load_qwen2vl(
                model_id, revision=revision, torch_dtype="auto", device_map="auto",
                processor_kwargs={"use_fast": True},
            )
        else:
            self.model, self.processor = load_native_vlm(
                model_id, revision=revision, torch_dtype="auto", device_map="auto",
                eval_mode=True, processor_kwargs={"use_fast": True},
            )
        self.device = self.model.device
        self.model.requires_grad_(False)
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        try:
            self._dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self._dtype = torch.float32

    def to_unit01(self, pil: Image.Image) -> torch.Tensor:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import pil_to_tensor, resize

        image = pil_to_tensor(pil.convert("RGB")).float().div_(255.0)
        image = resize(
            image,
            list(self.image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return image.unsqueeze(0).to(self.device)

    @staticmethod
    def _messages(prompt: str, image, target: str | None = None) -> list[dict]:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        if target is not None:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": target}],
            })
        return messages

    def _processor_inputs(self, text: str, image: torch.Tensor):
        """Tokenize with a detached image of the exact optimized geometry."""
        return self.processor(
            text=[text],
            images=[image.detach().cpu()],
            padding=False,
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)

    def _live_pixel_values(self, image: torch.Tensor) -> torch.Tensor:
        vision = self.processor.image_processor(
            images=[image],
            return_tensors="pt",
            do_rescale=False,
        )
        pixel_values = vision["pixel_values"].to(
            device=self.device, dtype=self._dtype
        )
        if image.requires_grad and not pixel_values.requires_grad:
            raise RuntimeError(
                f"{self.backend} image processor detached the optimized image. "
                "Install transformers>=4.57.0 and ensure AutoProcessor was "
                "created with use_fast=True."
            )
        return pixel_values

    def _ce_single(
        self, image: torch.Tensor, prompt: str, target: str
    ) -> torch.Tensor:
        detached = image.detach().cpu()
        prefix_messages = self._messages(prompt, detached)
        full_messages = self._messages(prompt, detached, target)
        prefix_text = self.processor.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        prefix = self._processor_inputs(prefix_text, image)
        full = self._processor_inputs(full_text, image)

        prefix_ids = prefix["input_ids"][0]
        full_ids = full["input_ids"][0]
        common = 0
        for left, right in zip(prefix_ids.tolist(), full_ids.tolist()):
            if left != right:
                break
            common += 1
        if common == 0 or common >= full_ids.numel():
            raise RuntimeError(
                f"{self.backend} chat template did not expose assistant target tokens"
            )
        labels = full["input_ids"].clone()
        labels[:, :common] = -100

        live_pixels = self._live_pixel_values(image)
        if "pixel_values" not in full:
            raise RuntimeError(f"{self.backend} processor returned no pixel_values")
        if tuple(live_pixels.shape) != tuple(full["pixel_values"].shape):
            raise RuntimeError(
                f"{self.backend} live/detached preprocessing shape mismatch: "
                f"{tuple(live_pixels.shape)} != {tuple(full['pixel_values'].shape)}"
            )
        model_inputs = dict(full)
        model_inputs["pixel_values"] = live_pixels
        model_inputs["labels"] = labels
        model_inputs["use_cache"] = False
        output = self.model(**model_inputs)
        if getattr(output, "loss", None) is not None:
            return output.loss
        logits = output.logits.float()
        return F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    def compute_loss_batch(self, img01: torch.Tensor, targets, pils=None,
                           prompts=None) -> torch.Tensor:
        n = len(targets)
        if n == 0:
            raise ValueError("targets must not be empty")
        if img01.size(0) not in {1, n}:
            raise ValueError(
                f"image batch must contain 1 or {n} images; got {img01.size(0)}"
            )
        prompts = prompts if prompts is not None else [""] * n
        if len(prompts) != n:
            raise ValueError("prompts and targets must have the same length")
        images = [img01[0] if img01.size(0) == 1 else img01[i] for i in range(n)]
        losses = [
            self._ce_single(images[i], prompts[i], targets[i]) for i in range(n)
        ]
        return torch.stack(losses).mean()

    def zero_grad(self):
        self.model.zero_grad()
