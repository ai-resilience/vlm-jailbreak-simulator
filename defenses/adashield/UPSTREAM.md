# AdaShield source provenance

Official repository: https://github.com/SaFo-Lab/AdaShield

Pinned commit: `b6d02c6747bd6984a99ee2e1db1e52736a6a05cc`.
The pinned checkout has no root software LICENSE file. Its provenance is
retained without inventing a license notice.

The immediate source is
`/home/jovyan/jailbreak/vlm-jailbreak-simulator/defenses/adashield.py` and the
static/example YAML configurations. The exact static prompt has SHA-256
`42652154f7926583845e074919d023a7403c8fd8237142e4a1b2f5816c0e963c`, also matching
the pinned upstream `prompts/static_defense_prompt.txt`.

The port changes base/path imports, adds the common `apply_input`/preflight/
prepare/metadata boundary, verifies local CLIP weights and hashes resources.
Its OpenAI CLIP loader is in `multimodalmodels/loaders.py`. Retrieval retains
joint image/text normalization, maximum similarity, seeded random selection
and tie breaking. Non-finite embeddings fail explicitly. Pillow inputs are
closed after conversion. Configuration-relative and data-root paths follow
the shared project conventions.

OpenAI CLIP is pinned to `a1d071733d7111c9c014f024669f959182114e33`.
Its installed MIT license and package metadata remain with that dependency.
The verified checkpoint SHA-256 is
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.

The imported adaptive assets come from the local
`runs/adashield_official_repro/{figstep,qr}/tables/` refinement run, trained
against `liuhaotian/llava-v1.5-13b`. They are not trained pools distributed by
the upstream repository. Raw tables, path-normalized copies, all referenced
images, source-run README and manifests are kept under
`data/method_assets/AdaShield/`, outside tracked source. Inference on the
registered 7B target is recorded as a transfer check with
`paper_reproduction=false`.

