# Data layout

```text
data/
├── original/                         # Existing ai-resilience data
│   ├── HarmBench/
│   │   ├── behaviors/                # Text 400, val 80, test 320, multimodal 110; 2-example CSV
│   │   └── images/                   # 110 original benchmark images
│   ├── advbench/                     # Existing raw CSVs and the 520-question BehaviorID CSV
│   ├── TDC2023/                      # 50-question CSV
│   ├── adv_training/                 # 100-question CSV
│   ├── optimizer_targets/
│   ├── classifier_val_sets/
│   ├── copyright_classifier_hashes/
│   ├── clean.jpeg
│   └── derogatory_corpus.csv
├── benchmarks/                       # Imported benchmark inputs
│   ├── advbench/
│   ├── AdvBenchM/
│   ├── HADES/
│   ├── MM-SafetyBench/
│   └── SafeBench/
└── method_assets/                    # Imported attack resources
    ├── CS-DJ/
    ├── JOOD/
    ├── JailbreakInPieces/
    ├── MML/
    ├── VisualAdv/
    ├── fonts/
    ├── nltk_data/
    └── provenance/
```

`original/` preserves the existing ai-resilience file contents. HarmBench's
question CSVs and source images are grouped together; the additional AdvBench,
TDC2023 and adversarial-training CSVs are stored in their respective directories.
HarmBench validation and test questions are subsets of its 400 text questions.

`benchmarks/` and `method_assets/` were copied from
`/home/jovyan/jailbreak/vlm-jailbreak-simulator/data/`, including its MML assets.
The import preserves file contents and upstream metadata. Original AdvBench-M
data is in `benchmarks/AdvBenchM/`; JOOD's image pool and question subset are in
`method_assets/JOOD/`.

`--dataset_name` selects `AdvBench`, `AdvBenchM`, `HADES`, `MM-SafetyBench` or
`SafeBench` through `benchmark_datasets.py`. `--dataset_name JOOD` separately
selects 170 aligned questions and prompt annotations from `method_assets/JOOD/`;
it does not replace the 246-question original AdvBench-M release.
Each loader keeps the source IDs,
original questions, categories and image associations in a common record format.
Question order and IDs are assigned before applying `[behavior_start_idx,
behavior_end_idx)`. Omitting one bound selects from the start or through the end.
Image variants and category pools stay attached to their questions.

`--behaviors_path` accepts a custom CSV and cannot be combined with
`--dataset_name`. With neither selector, JOOD defaults to its annotated subset,
JPS to AdvBench, and MML to SafeBench. Other methods use
`data/original/HarmBench/behaviors/harmbench_behaviors_text_all.csv`.
The shell launchers explicitly select their own per-method defaults, documented
in [scripts/README.md](scripts/README.md).
Attack YAMLs select training images and target corpora independently of the
evaluation question set. DeltaJP's default training split is from AdvBenchM;
its evaluation images come from the selected questions. VisualAdv and ImgJP
use their configured optimization corpora. JOOD uses `method_assets/JOOD/`.
CS-DJ uses `method_assets/CS-DJ/llava_images/` and `Super Moods.ttf` while taking
its questions from the selected benchmark. Generated collages, embedding caches
and per-question checkpoints are stored under the selected `--save_dir`.
Benchmark CSV, JSON and JSONL files remain unchanged on disk.

JOOD's default CutMix creates 45 attempts per question (five harmless images ×
nine harmful-image weights). Other benchmark questions require an enriched CSV
with a corresponding image and masked multimodal prompt, or a harmful-keyword
annotation for text mixing. Generated mixes stay under `--save_dir`; source
images and prompt annotations remain in `method_assets/JOOD/`.

Each generated test-case JSON has a `.behaviors.json` companion containing the
selected original records, source and selection settings. Its hashes bind the
records to the generated JSON. Completion generation carries the companion to
its output; evaluation automatically uses those original questions and retains
the companion with its results. An explicit evaluation selector must agree with
the saved original questions. Incremental generation rejects mismatched saved
questions or generation settings before loading a target model.

`data_paths.py` resolves `data/...` against `VLM_JAILBREAK_DATA_ROOT`, which
defaults to `<repo>/data`. It also resolves the former ai-resilience relative
paths to their archived locations. External absolute paths remain unchanged.

Generated attack inputs, responses and evaluations remain under `test_cases/`,
`completions/` and `results/`. Imported benchmark and method-asset directories
are excluded from Git; the existing data is relocated under `original/`.
