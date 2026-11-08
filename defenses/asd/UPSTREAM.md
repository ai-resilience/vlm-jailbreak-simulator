# AsD source provenance

Official repository: https://github.com/AngelAlita/AsD

Pinned commit: `ba71efe0972c92e38da156ec76911f9122bbe12f`.
The upstream MIT license text is retained unchanged in `LICENSE`.

The immediate source is the reference simulator at
`/home/jovyan/jailbreak/vlm-jailbreak-simulator`:

- `defenses/asd.py` → `defenses/asd/defense.py`.
- `configs/defense_configs/AsD.yaml`: same inference parameters, relocated asset path.
- `defenses/asd_assets/L2_noNorm_clipgrad_tips_336_LR0_1-1_1.png`: copied into the data root.

The original prompt and released PNG are hash-checked. The patch also matches
the pinned upstream checkout's `images/` asset. Area/aspect geometry, Pillow
resize and alpha/paste operations, local seeded RNG and prompt composition
are preserved.

Local adaptations include the common preflight/prepare/metadata/apply_input
interface, data/config-relative paths, asset setup and explicit temporary-image
cleanup for library users. Preflight validates geometry without consuming RNG.
Persistent outputs have paths and hashes recorded alongside the original
test case. All transformations remain model-independent and perform no target
generation themselves.

The upstream target is `liuhaotian/llava-v1.5-7b`; validation uses the shared
`llava-hf/llava-1.5-7b-hf` adapter and its pinned revision. Other target
applications are transfers. `paper_reproduction=false` is recorded.
The port uses the released patch rather than rerunning its optimizer.

