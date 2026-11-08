# MML provenance

Reference: `/home/jovyan/jailbreak/vlm-jailbreak-simulator/multi_attacks/mml/`,
`core/protocols/mml.py`, `core/judges/mml_clas.py` and
`core/targets/openai_vision.py`.

Upstream: https://github.com/wangyu-ovo/MML
Pinned commit: `5966fca529011e8b5ccff61d70365997d6466abc`.
The upstream repository has no explicit software license. Its prompt literals
are not copied into the integration source: `official_source.py` parses the
user's local `const.py` after checking SHA-256
`89d7480ff98a4872b50f16c8bf4271f37074127fa0c1cfdd46a1f50b2cdce9a7`.
It does not import or execute that file.

The reference's generated renderer and clean-room prompt templates remain
adaptations because the upstream image generator was not published. Released
images are copied without modifying their bytes. The official CLAS gates and
stop-on-score-5 loop retain the reference implementation's control flow.

The CLI/schema bridge and image fingerprints are local integration code.
Twenty-one SafeBench title differences use a reviewed mapping bound to stable
category/task IDs, exact benchmark/released-title hashes and source metadata
hashes. No runtime fuzzy matching is performed, and original questions remain
unchanged. The pinned map is under `data/method_assets/MML/`.

No full-paper reproduction is claimed.
