# CIDER source provenance

Official repository: https://github.com/PandragonXIII/CIDER

Pinned commit: `3c08d7e2d52e27b47d1e428a030e8ffda5efae3e`.
The local upstream checkout is `/home/jovyan/jailbreak/reference_repos/CIDER`.
Its Apache-2.0 license is retained in `LICENSE`.

The immediate source for this port is
`/home/jovyan/jailbreak/vlm-jailbreak-simulator`:

- `defenses/cider.py`: adapter and reused LLaVA encoder, now `defense.py` and `encoder.py`.
- `defenses/cider_utils.py`: cosine-drop rule and checkpoint-image generation,
  now `detection.py`, `encoder.py` and `denoising.py`.
- `defenses/cider_models/diffusion_denoiser/imagenet/DRM.py`: now `denoiser.py`.
- Nine inference dependencies from `guided_diffusion/`: copied without byte changes.
- `configs/defense_configs/CIDER.yaml`: inference settings retained, checkpoint path relocated.

The shared defense interface, early target/input validation, checkpoint
fingerprints, metadata, setup module and registry/CLI integration are local
adaptations. Image opening in the encoder uses a context manager. The detector
rule and denoising/image conversion operations are retained. Only the inference
rule is extracted from the reference analysis/calibration helper. Caching is
scoped to each defense instance instead of a process-global checkpoint cache.
DRM's import and installation-help paths follow the new package layout.

The diffusion implementation derives from
https://github.com/openai/guided-diffusion. Its OpenAI MIT notice is retained in
`guided_diffusion/LICENSE`. The diffusion checkpoint URL is
https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt;
the expected byte length and MD5 follow the reference setup script.

The reference simulator also reuses LLaVA-1.5 for encoding. This integration
keeps that support restriction and records `paper_reproduction=false`.
