# Attack compositions

Compositions connect existing attack operations while keeping the original
question for evaluation. All outputs are `adapted_composition` with
`paper_reproduction: false`.

## Existing three-command workflow

From the repository root, generate the 54-case preview (18 recipes × three
benchmarks), then pass its JSON to the existing completion and evaluation CLIs:

```bash
python generate_test_cases.py --composition_config configs/composition_configs/five_methods_3000.yaml --save_dir test_cases/composed
python generate_completions.py --model_name qwen3_vl_2b --test_cases_path test_cases/composed/preview/test_cases.json --save_path completions/composed.json
python evaluate_completions.py --completions_path completions/composed.json --save_path results/composed.json
```

`--composition_config` and `--method_name` select separate generation modes.
Existing single-attack commands retain their behavior. The composition YAML
controls sources, recipes, counts and seed. `--composition_phase dry-run`
prints the plan; `preflight` checks prerequisites. The default phase is
`preview`; full production is explicit. Outputs are separated by phase.

Choose a target supporting the output's image count. FigStep/MML/CS-DJ recipes
that also include JPS can produce two images. Qwen3-VL supports them; LLaVA-1.5
accepts at most one. Defenses use the existing `--defense_name` option and retain
their own input/model limits; a single-image defense does not automatically
support two-image compositions. Unsupported inputs fail before target loading.
The standard protocol supports compositions; MML's separate `mml_official`
response protocol is not the composition protocol.

## Prepare local assets

The installed assets were copied and verified from the original simulator:

```bash
python -m compositions.setup --reference_root /path/to/original/vlm-jailbreak-simulator
```

This copies the hash-pinned Noto font and OFL notice, plus the exact JPS image,
steering text and training CSV authenticated by the original result manifest.
It does not retrain JPS. JOOD's harmless images and CS-DJ's library use their
existing method asset directories.

To reuse verified original CS-DJ decomposition/retrieval results:

```bash
python -m compositions.setup --reference_root /path/to/original/vlm-jailbreak-simulator --cs_cache_run /path/to/completed/five_method_run
```

The importer checks the completed run and original parent/cache bindings,
verifies contrast images against local assets, and writes a hash manifest for
the imported cache. Runtime reuse requires matching original records and
retrieval/model settings. Uncached questions use the existing live CS-DJ
splitter/retriever. The default configuration points to this optional cache;
remove `cs_dj.precomputed_dir` to compute all results live.

## Five-method production plan

[five_methods_3000.yaml](../configs/composition_configs/five_methods_3000.yaml)
preserves all 18 recipes: nine pairs, seven triples and two quadruples using
FigStep, CS-DJ, MML, JOOD and JPS. It covers all 2,930 benchmark records once,
then adds 70 draws for 3,000 cases. Dataset proportions, original evaluation
types, duplicate-record handling, JPS training exclusions and shared-question
splits retain the source policy.

The measured renderer preserves text instead of clipping it. JOOD uses
native-resolution MixUp with the original image weighted from 0.05 to 0.95;
JPS uses the authenticated auxiliary image/steer pair. These are explicit
adaptations of method components. The five-method plan is not equivalent to
executing each complete standalone attack consecutively.

Inspect the preview images/contact sheets, record observations, then request
full generation. The source implementation's preview-review requirement is
preserved and bound to the actual preview hashes:

```bash
python generate_test_cases.py --composition_config configs/composition_configs/five_methods_3000.yaml --save_dir test_cases/composed --composition_phase review-preview --composition_review_note 'Observations from inspecting all preview sheets'
python generate_test_cases.py --composition_config configs/composition_configs/five_methods_3000.yaml --save_dir test_cases/composed --composition_phase full
python generate_test_cases.py --composition_config configs/composition_configs/five_methods_3000.yaml --save_dir test_cases/composed --composition_phase validate --composition_validate_phase full
```

Full outputs are under `test_cases/composed/full/`. Repeating a generation
command resumes verified existing cases. Configuration, code, font, JPS and
source-plan changes require a new output directory. Changed cached payloads
or image artifacts cause errors.

## General stage recipes

[stages_ready.yaml](../configs/composition_configs/stages_ready.yaml) provides
three source recipes using attacks already ported here:

- P01: CS-DJ split → FigStep rendering.
- P02: CS-DJ split → MML generated layout.
- P03: FigStep rendering → JOOD RandAug.

```bash
python generate_test_cases.py --composition_config configs/composition_configs/stages_ready.yaml --composition_per_recipe 1 --save_dir test_cases/stage_pilot
python generate_test_cases.py --composition_config configs/composition_configs/stages_ready.yaml --composition_per_recipe 1 --save_dir test_cases/stage_pilot --composition_phase validate
```

`--composition_recipes P01,P03` selects recipes, and
`--composition_per_recipe N` overrides final counts for a pilot. Without the
override, preview produces up to three per recipe; full uses YAML quotas.
The full ready profile targets 600 final cases from 720 candidates.
Each completed candidate is checkpointed. Selection uses structural validity,
content deduplication and deterministic ordering, without response judging or
ASR-based selection.

Supported adapters are `cs_dj_split`, `cs_dj_full`, `figstep_render`,
`mml_layout`, `jood_aug` and `jps_joint`. Full JPS optimization uses the existing
ported attack once per component configuration; the five-method profile uses
frozen transfer assets instead. Rendering/image-construction stages cannot
follow another image-producing stage and silently replace its result.

The original eight-recipe
[jailbreak_dataset_v3.yaml](../configs/composition_configs/jailbreak_dataset_v3.yaml)
is retained. Recipes requiring unported stages (such as UltraBreak or JIP)
are explicitly rejected. Select P01/P02/P03 or use the ready configuration.
No unsupported recipe is silently omitted.

## Data, metadata and evaluation

Native intermediate/checkpoint files remain available. Both modes additionally
export `test_cases.json` and `test_cases.json.behaviors.json`, compatible with
the current CLIs. Cases retain recipe IDs, method names, unique case IDs,
original source IDs, image hashes, split, and provenance. General recipes also
record stage input/output hashes and continuity. The sidecar retains original
questions and binds them to the exact exported payload and images.

Completions retain the full original case, including recipe metadata. Evaluation
adds `by_recipe` reports alongside the existing twelve metrics. Each recipe's
population contains only its own questions; aggregate ASR uses Pass@k per
original question across the selected recipes. These two summaries answer
different questions. Advice/opinion records remain in the dataset with their
original types and receive no harmful-instruction binary judge label; headline
ASR includes harmful/copyright questions. Benign controls and quality/severity
annotations follow the existing metric requirements.

## Package layout

`engine.py` holds the source stage contracts; `components.py` adapts existing
attacks; `generator.py` runs general recipes. `five_methods.py`,
`five_runner.py` and `typography.py` preserve the source production generator.
`datasets.py`/`records.py` adapt shared benchmark rows; `export.py`,
`validation.py` and `evaluation.py` connect the existing CLI formats.
No `core` package or imports from the original checkout are required.

