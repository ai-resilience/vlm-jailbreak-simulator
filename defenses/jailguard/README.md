# JailGuard

```bash
python generate_completions.py --model_name llava_v1_5 --test_cases_path test_cases/FigStep_SafeBench/FigStep_test_cases.json --save_path completions/FigStep_JailGuard.json --defense_name JailGuard
```

Use `--defense_config PATH` for a custom YAML. The default
`configs/defense_configs/JailGuard.yaml` uses eight PL image mutations,
variant responses capped at 300 tokens, first-100-character spaCy similarity and KL
threshold 0.025. The input question is unchanged. An all-variants-refused
decision also blocks, as in the reference. A pass adds one original-pair query
at the normal target token budget: eight or nine target calls per input.

JailGuard accepts exactly one image. Text-only and multi-image cases are
rejected before loading a target. `mml_official` cannot compose with defenses;
MML's standard fixed-budget pipeline can use JailGuard.

JailGuard requires spaCy and `en_core_web_md`. The port uses the original
3.7.1 model package without changing the existing spaCy dependencies:

```bash
python -m pip install --no-deps https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl
```

The initial port was tested with spaCy 3.8.18; the model was trained with
3.7.2, so that environment emits a compatibility warning. The package layout
was also verified in the existing `jb_env` with spaCy 3.7.2 and the same 3.7.1
model. Runtime/model versions and resource hashes are stored in completion
metadata. No exact paper reproduction is claimed.

Completion records retain `blocked`, `meta.detected_adversarial`, maximum
divergence, similarity/divergence matrices, response hashes, query counts,
token budgets and elapsed times. The metadata sidecar binds settings, code and
NLP assets. Incremental reuse must match this provenance. Use separate output
paths for baselines and defended runs; mismatched defense settings are rejected.

See [source provenance](UPSTREAM.md).
Use a new output path after changes to the implementation or its layout.
