# JailGuard source provenance

Source repository: https://github.com/shiningrain/JailGuard
Pinned commit: `419e1d3d08f4d02998a11a687f925518d2d5dd0c`.
The local pinned upstream tree has no software LICENSE file; its authors'
provenance is retained without inventing a license notice.

This port copies the reference simulator's `defenses/jailguard.py` and
image-only `JailGuard_utils/image_mutations.py`, `utils.py`, `similarity.py`.
In ai-resilience these live together under `defenses/jailguard/`: the adapter
is `defense.py`, and the three helpers retain their original filenames.
The image mutations and divergence helpers are unchanged. The adapter's core
target/path imports are replaced with the shared ai-resilience equivalents;
finite-value guards and early config validation are added.

The generic registry, result routing, call accounting, cache isolation and
CLI wiring are integration code. The original target was MiniGPT-4; validation
uses registered LLaVA and Qwen3 targets. The original en_core_web_md 3.7.1
resource ran in the initial port's spaCy 3.8.18 environment rather than 3.7.2.
The package layout was also checked in `jb_env` with spaCy 3.7.2.
Provenance records both substitutions and keeps paper_reproduction false.

The shared package structure is documented in [defenses/README.md](../README.md).
