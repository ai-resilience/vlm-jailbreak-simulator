# AdaShield

AdaShield adds a shield prompt to the target's input. **AdaShield-S** uses the
exact upstream static prompt. **AdaShield-A** retrieves a prompt from successful
refinement rows using OpenAI CLIP ViT-B/32. Both use the existing completion and
evaluation CLIs.

Static S is the default:

```bash
python generate_completions.py --model_name llava_v1_5 \
  --test_cases_path test_cases/FigStep_SafeBench/FigStep_test_cases.json \
  --save_path completions/FigStep_AdaShield-S.json --defense_name AdaShield
```

Adaptive A with the imported FigStep pool:

```bash
python generate_completions.py --model_name llava_v1_5 \
  --test_cases_path test_cases/FigStep_SafeBench/FigStep_test_cases.json \
  --save_path completions/FigStep_AdaShield-A.json --defense_name AdaShield \
  --defense_config configs/defense_configs/AdaShield-A.yaml
```

Use `AdaShield-A-QueryRelated.yaml` for the imported QueryRelated pool.
`AdaShield-A.example.yaml` shows how to provide another refinement pool.
Relative custom pool paths resolve beside the YAML file; `data/...` paths
follow `VLM_JAILBREAK_DATA_ROOT`.

Generate an undefended baseline into another path, then compare it normally:

```bash
python evaluate_completions.py --method_name FigStep \
  --completions_path completions/FigStep_AdaShield-A.json \
  --baseline_results_path completions/FigStep_baseline.json \
  --save_path results/FigStep_AdaShield-A.json
```

## Preserved algorithm

The default LLaVA placement is `query + shield_prompt + query`, with the
original whitespace retained. `append` and `quoted_sandwich` placements are
also supported by the reference adapter. Choose the placement required by
the target's upstream protocol; applying the LLaVA default to another target
is not an exact paper reproduction.

A keeps CSV rows with `final_judge_scores == 1` and a nonempty defense prompt.
The FigStep configuration also excludes the literal `['rephrase prompt']`
sentinel. QueryRelated does not apply that optional filter.

Sample-wise retrieval concatenates raw CLIP image and text features, normalizes
the combined vector, and selects the highest float32 dot product against the
normalized pool. Equal maxima use random tie breaking. Random retrieval is
available with `retrieval_type: random`. A `seed` controls that RNG, whose
sequence depends on selection order. Pool embeddings are cached per instance.
CLIP's original 77-token limit is preserved; text is not silently truncated.

The original images reach the target unchanged. S supports the selected
target's image-count limits; A requires exactly one image. Neither variant
generates a response during preprocessing. Each input receives one final
target generation at its normal token budget.

AdaShield supplies no malicious-input detector decision. `blocked=false`
records that preprocessing forwarded the input, even when the target's answer
refuses the request. Detection Success Rate and detector F1 remain unavailable.
ASR reduction and benign utility compare actual outputs; over-defense and
harmfulness metrics require their respective refusal/severity annotations.

## Local pool assets

Assets live under `data/method_assets/AdaShield/`:

| Pool | CSV tables | Raw rows | Retained prompts | Copied images |
| --- | ---: | ---: | ---: | ---: |
| FigStep | 10 | 50 | 40 | 50 |
| QueryRelated | 13 | 160 | 104 | 160 |

These are **locally refined pools from the reference simulator's LLaVA-1.5-13B
run**, rather than author-distributed trained pools. Applying them to the
registered LLaVA-1.5-7B is a transfer check. Original CSV copies are retained;
portable CSVs change only the image paths. Manifests preserve source hashes
and provenance. A verified local CLIP checkpoint is copied into `checkpoints/`.

To import the same assets on another checkout:

```bash
python -m defenses.adashield.setup \
  --reference_root /path/to/reference/vlm-jailbreak-simulator \
  --clip_checkpoint /path/to/ViT-B-32.pt
```

The importer performs local copies and hash verification. It does not run
refinement training or call an API. Static S needs none of these assets.
The OpenAI CLIP package is pinned in `requirements.txt`; the original
ViT-B/32 checkpoint SHA-256 is enforced before loading.

Metadata binds code, pool tables, selected anchor resources, the CLIP package,
tokenizer vocabulary and checkpoint. Records include the selected prompt,
pool index, source row, similarity and hashes of the original/shielded text.
Use separate output paths for variants and matching provenance for incremental
reuse. A resumed run advances the selection RNG only for pending inputs.

Support covers inference in the standard pipeline. `mml_official` composition,
new refinement training, paper-specific GPT rephrasing and benign-query bypass
are outside this adapter. See [source provenance](UPSTREAM.md).
