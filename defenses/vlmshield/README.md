# VLMShield

VLMShield is an input detector with a separate CLIP ViT-L/14 encoder and the
released 1536→1024→512→2 MLP classifier. Class 0 is unsafe (block); class 1 is
safe (forward). The decision is softmax argmax, with no extra threshold.
Equal probabilities select class 0. Probabilities are classifier scores rather
than calibrated safety guarantees.

## Setup

```bash
python -m defenses.vlmshield.setup
# Or copy an existing released classifier:
python -m defenses.vlmshield.setup --source /path/to/vlmshield_classifier.pt
```

The setup command verifies the published classifier size and SHA-256, writes
atomically and keeps weights outside source under
`data/method_assets/VLMShield/checkpoints/`. Existing incorrect files cause an
error. `VLM_JAILBREAK_DATA_ROOT` redirects the data tree.

The existing shared `load_clip_model()` loads the pinned Hugging Face
`openai/clip-vit-large-patch14` snapshot in float32. Its first use needs an
available local cache or network access. This migration copied the existing
local classifier and reused the cached CLIP model; no model downloads or
package installations were needed.

## Run and evaluate

```bash
python generate_completions.py --model_name llava_v1_5 --test_cases_path test_cases/FigStep_SafeBench/FigStep_test_cases.json --save_path completions/FigStep_VLMShield.json --defense_name VLMShield
python evaluate_completions.py --method_name FigStep --completions_path completions/FigStep_VLMShield.json --baseline_results_path completions/FigStep_baseline.json --save_path results/FigStep_VLMShield.json
```

The baseline must be generated from the same inputs, target and generation
settings, without a defense. `--defense_config PATH` selects custom YAML.
Relative classifier paths resolve beside that YAML, while `data/...` paths
use the configured data root. Image paths follow the standard CLI rules.

The detector accepts zero or one image. Text-only and image-only inputs are
supported; a missing modality contributes 768 zeros. The selected target's
own image limits also apply. Multiple images are rejected before model
loading. Valid images, classifier integrity and configuration are checked
before loading target weights. Detection failure raises an error.

A block returns the simulator's fixed refusal with zero target generation
calls. A pass forwards the original image and text unchanged and generates
once at the requested target budget. The shared runner currently creates a
target instance even for a cohort that will all be blocked; zero calls means
zero target generations, not zero target loading cost. No target queries are
used internally by the detector.

## Features and records

Image features preserve the upstream projected **last hidden-state CLS**
vector; they do not substitute the usual CLIP pooled image feature. Text uses
EOS features, a 75-token chunk size and 10-token overlap. Long text preserves
the upstream final-chunk handling, including its 73-token final-tail cap and
similarity-weighted aggregation. Image and text features are concatenated
without an added normalization step.

Outputs retain the original `test_case` for paired evaluation and include
`detected_adversarial`, safe/unsafe probabilities, confidence, input hashes,
classifier hash, target call counts and timing. `store_embedding: true` also
stores the 1536-dimensional vector. Run metadata fingerprints implementation
files, the classifier, the actual CLIP snapshot resources, library versions
and numerical settings. Incremental reuse checks this provenance.

Unlike input modifiers, VLMShield provides an explicit detection decision.
Detection success rate can use attack results; F1 and over-defense rate also
need benign results. Benign task accuracy needs task annotations and harmfulness
reduction needs severity annotations. The upstream evaluation script calls
unsafe inputs *passed by its filter* “ASR”; this repository's attack ASR uses
the downstream response judge and therefore measures a different outcome.

No changes are required in `generate_test_cases.py` or `evaluate_completions.py`.
The standard MML pipeline can use this defense; `mml_official` remains separate.
See [provenance](UPSTREAM.md).
