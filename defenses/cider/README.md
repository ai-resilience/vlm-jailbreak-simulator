# CIDER

CIDER compares text/image cosine similarity before and after diffusion-based
image denoising. This adapter reuses the registered **LLaVA-1.5-7B** target's
vision tower, multimodal projector and input embeddings. It accepts exactly
one image. Other targets are rejected before loading weights.

```bash
python generate_completions.py \
  --model_name llava_v1_5 \
  --test_cases_path test_cases/FigStep_SafeBench/FigStep_test_cases.json \
  --save_path completions/FigStep_CIDER.json \
  --defense_name CIDER
```

Generate the undefended baseline with the same input and another output path,
then evaluate the pair:

```bash
python evaluate_completions.py \
  --method_name FigStep \
  --completions_path completions/FigStep_CIDER.json \
  --baseline_results_path completions/FigStep_baseline.json \
  --save_path results/FigStep_CIDER.json
```

The settings live in `configs/defense_configs/CIDER.yaml`. Override them with
`--defense_config PATH`. The released profile uses eight checkpoints
`0, 50, 100, 150, 200, 250, 300, 350` and threshold `-0.003936767578125`.
Checkpoint zero is a 224×224 resized JPEG; each later checkpoint independently
starts from the resized source image. Each of those seven checkpoints uses
the reference denoiser's single predicted-x-start operation, with its diffusion
timestep as the noise level.

For similarities `c[0], ..., c[7]`, the detector computes `d[i] = c[i] - c[0]`.
It returns the index of the lowest delta if `min(d) < threshold`, otherwise
zero. A nonzero index blocks the input. Equality with the threshold passes.
An allowed pair reaches the target **with the original image and question**.

The detector generates no target response. A blocked input has zero target
generation calls; an allowed input has one final generation call. Detection
still requires seven diffusion-model evaluations, eight image-encoder forwards
and eight text-embedding forwards. Generation counts alone do not describe
its full computation cost.

## Checkpoint setup

The official checkpoint is stored outside the source package at
`data/method_assets/CIDER/checkpoints/256x256_diffusion_uncond.pt`.
`VLM_JAILBREAK_DATA_ROOT` redirects the data root. The local port installed it
from the reference simulator and verified its size and MD5.

For a fresh installation, download and verify the checkpoint:

```bash
python -m defenses.cider.setup
```

Or copy an existing checkpoint:

```bash
python -m defenses.cider.setup --source /path/to/256x256_diffusion_uncond.pt
```

Expected size: `2211383297` bytes. Expected MD5:
`fd9dd2335b8736d521de0aed54bd90ca`. Installation uses a temporary file and
replaces the destination after verification. Generation checks size and
records SHA-256/MD5; `verify_denoiser_md5: true` also enforces the official
MD5 at runtime. The denoiser is loaded once per defense instance and reused.
The auxiliary ViT classifier is not loaded for detection.

## Results and scope

Records retain `blocked`, `detected_adversarial`, similarities/deltas, minimum
delta, checkpoint index, model-work counts, target generation counts and timing.
Metadata binds configuration, code and the diffusion checkpoint hash. Use
separate paths for baselines and defended runs. Incremental reuse requires
matching provenance, including shared defense code.

The standard completion pipeline supports CIDER. `mml_official` composition
is unsupported. This port uses the released inference threshold; offline
threshold recalibration is outside the adapter's scope. Denoising samples
random noise, so unseeded runs can produce different decisions. No full
benchmark or exact paper reproduction is claimed.

See [source provenance](UPSTREAM.md).
