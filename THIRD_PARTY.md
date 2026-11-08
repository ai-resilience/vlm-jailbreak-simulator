# Third-Party Software, Models, Datasets, and Assets

The shared model layer is implemented in `multimodalmodels/registry.py`,
`loaders.py`, `surrogates.py`, and `targets/`. The target adapters were ported
from our unified simulator's `core/targets/`; model IDs and default HF revisions
are recorded in the registry. Model selection is connected through
`configs/model_configs/models.yaml` and the existing completion CLI.

This document inventories the third-party components that the
**VLM-Jailbreak-Simulator** actually relies on **at runtime** — that is,
components that are imported, vendored in this repository, downloaded as
model weights, read as datasets, or called as remote APIs when the
attacks defined in `simulator_test.sh` (and the supporting
`generate_test_cases.py` / `generate_completions.py` /
`evaluate_completions.py` entry points) are executed.

These components are **not** covered by the MIT License in
[LICENSE](LICENSE). Each remains subject to its **own upstream license**,
notices, and use restrictions.

> ⚠️ Some entries below are **non-commercial only** (e.g., the Vicuna
> weights that MiniGPT-4 builds on), some are under **community licenses
> with their own restrictions** (e.g., the Llama 2 Community License),
> some are **closed/commercial APIs** governed by the provider's Terms
> of Use (e.g., OpenAI), some upstream repositories have **no explicit
> license attached**, and some assets are subject to **separate copyright**
> (e.g., the Arial font). Users and redistributors are solely responsible
> for confirming and complying with the applicable upstream terms before
> using, fine-tuning, or redistributing each component.

The lists below are **not exhaustive**; transitive dependencies of the
listed packages are governed by their own licenses and should be reviewed
independently.

---

## 1. Vendored / Adapted Attack Implementations

The following attack implementations are adapted or copied (in whole or
in part) from third-party research repositories and live inside this
repository. Original copyright and license terms remain with the upstream
authors.

| Component (path in this repo)        | Upstream                                                                                                                                                       | Upstream License (as observed) |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| `multi_attacks/figstep/`             | [thuccslab/figstep](https://github.com/thuccslab/figstep)                                                                                                       | **MIT**                        |
| `multi_attacks/cs_dj/` | [TeamPigeonLab/CS-DJ](https://github.com/TeamPigeonLab/CS-DJ), commit `4577b05fe93e86030ba2e4c3339e3aa59cdbb5ff`; upstream notice retained in `multi_attacks/cs_dj/LICENSE` | **MIT** |
| `image_attacks/visual_adv/`          | [Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models), reference commit `5fc9883cfa9b4183cf05c6eceef49e449e45e9c9`   | No explicit license            |
| `image_attacks/imgjp/`, `image_attacks/deltajp/`, `image_attacks/jp_pgd.py` | [abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model](https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model), commit `fd98910ae4075fe587a1944095c00660aa24cc54` | No explicit license            |
| `multi_attacks/hades/`               | [AoiDragon/HADES](https://github.com/AoiDragon/HADES)                                                                                                           | **MIT**                        |
| `multi_attacks/best_of_n/`           | [jplhughes/bon-jailbreaking](https://github.com/jplhughes/bon-jailbreaking) — at runtime it also imports the upstream `bon` Python package                       | **MIT**                        |
| `multi_attacks/JOOD/`                | [naver-ai/JOOD](https://github.com/naver-ai/JOOD) at `4ca1368` — compact prompt/mixing port; full upstream `LICENSE` file is preserved at `multi_attacks/JOOD/LICENSE`                                    | **Apache-2.0**                 |
| `multi_attacks/jps/` | [thu-coai/JPS](https://github.com/thu-coai/JPS), commit `beacbc58a359005e00a7e7f7b0b485eae64bb163`; reference simulator's JPS logic and dedicated Qwen2 adapter | MIT declared in upstream README; linked root LICENSE absent at the pinned commit |
| `multi_attacks/mml/` | [wangyu-ovo/MML](https://github.com/wangyu-ovo/MML), commit `5966fca529011e8b5ccff61d70365997d6466abc`; released-image replay plus reference simulator renderer/CLAS integration | Upstream has no explicit license; prompt literals read at runtime from the hash-pinned local checkout |

VisualAdv's pixel-space PGD is copied unchanged from our reference simulator's
`image_attacks/visual_adv/pgd.py`. The BehaviorID adapter, shared CE surrogate
integration and artifact checks are local integration code. The shared LLaVA
and Qwen surrogates substitute for the original paper models; their use does
not establish reproduction of the paper's ASR. The integration is implemented
in `image_attacks/visual_adv/visual_adv.py`.

ImgJP adapts the upstream `v1_mprompt.py`, `torchattacks/attack.py` normalization
and `torchattacks/attacks/pgd_uap_v1.py` sequential updates, read from pinned
local Git objects. It substitutes the shared surrogate and target-token CE
for the upstream MiniGPT/full-conversation objective. DeltaJP ports our
reference simulator's cross-class adaptation and its first-category-image
policy. Split filtering, artifact checks and CLI metadata are integration code.

JPS uses pinned `Qwen/Qwen2-VL-7B-Instruct` and
`Qwen/Qwen2.5-14B-Instruct` through the shared loaders. Its training subset is
the [RICommunity/TAP CSV](https://github.com/RICommunity/TAP/blob/main/data/advbench_subset.csv)
linked by the JPS README. The exact CSV Git blob SHA-1, SHA-256 and TAP MIT
notice are retained under `data/method_assets/JPS/`. The unpublished original
base image is replaced explicitly by an existing local image. See
`multi_attacks/jps/UPSTREAM.md`.

MML's SafeBench released images, source metadata and local upstream checkout
are under `data/method_assets/MML/`, outside tracked integration source. The
renderer and clean-room prompts are reference simulator integration code;
the upstream image generator is not released. NLTK's
`averaged_perceptron_tagger` resource and a hash-bound SafeBench alignment map
support generated word replacement and released-data matching respectively.
Optional `mml_official` execution uses OpenAI targets and a CLAS chat judge;
standard GPU verification uses the existing local targets and HarmBench.
See `multi_attacks/mml/UPSTREAM.md`.

CS-DJ generation uses `sentence-transformers/clip-ViT-L-14` and
`Qwen/Qwen2.5-3B-Instruct` through the shared loaders. Their exact revisions
are recorded in the model registry and method config. The CC3M image sample,
Super Moods font and upstream asset metadata are under `data/method_assets/CS-DJ/`;
their original asset terms remain applicable.

The defense implementation `defenses/jailguard/defense.py` and its image mutation/
similarity helpers follow [shiningrain/JailGuard](https://github.com/shiningrain/JailGuard)
at `419e1d3d08f4d02998a11a687f925518d2d5dd0c` through the reference simulator.
The pinned upstream tree has no software LICENSE file. The copied helper bytes,
source hashes and integration changes are documented in
`defenses/jailguard/UPSTREAM.md`.
JailGuard loads spaCy's `en_core_web_md` 3.7.1 model package; its installed
resource metadata and license notices remain with that package. The local
initial spaCy runtime was 3.8.18; the package layout was also checked with
spaCy 3.7.2. No additional NLP weights are redistributed in source.

The `defenses/cider/` package follows
[PandragonXIII/CIDER](https://github.com/PandragonXIII/CIDER) at
`3c08d7e2d52e27b47d1e428a030e8ffda5efae3e` through the reference simulator.
The upstream Apache-2.0 license is retained in `defenses/cider/LICENSE`.
Its guided-diffusion inference files retain the OpenAI MIT notice in
`defenses/cider/guided_diffusion/LICENSE`. The official 256×256 unconditional
diffusion checkpoint is stored locally under `data/method_assets/CIDER/`,
outside tracked source. Adaptations and hashes are documented in
`defenses/cider/UPSTREAM.md`.

The `defenses/adashield/` adapter follows
[SaFo-Lab/AdaShield](https://github.com/SaFo-Lab/AdaShield) at
`b6d02c6747bd6984a99ee2e1db1e52736a6a05cc` through the reference simulator.
The pinned upstream tree has no root software LICENSE file.
AdaShield-A uses the MIT-licensed OpenAI CLIP package pinned to
`a1d071733d7111c9c014f024669f959182114e33`; its installed license remains
with that dependency. The static prompt, locally refined 13B prompt pools,
copied anchor images and verified CLIP checkpoint have provenance and hashes
under `data/method_assets/AdaShield/`. These local pools are not author-released
trained artifacts. Changes and verification are described in
`defenses/adashield/UPSTREAM.md`.

The `defenses/asd/` adapter follows
[AngelAlita/AsD](https://github.com/AngelAlita/AsD) at
`ba71efe0972c92e38da156ec76911f9122bbe12f` through the reference simulator.
Its upstream MIT license text is preserved in `defenses/asd/LICENSE`.
The released protective image is verified by SHA-256 and stored under
`data/method_assets/AsD/patches/`, outside tracked source. Input transformation,
asset provenance and adaptation details are documented in
`defenses/asd/UPSTREAM.md`.

The `defenses/vlmshield/` adapter and feature/classifier helpers follow
[pgqihere/VLMShield](https://github.com/pgqihere/VLMShield) at
`344ca14778383aef57315229118f7669454fb4bb` through the reference simulator.
Its Apache-2.0 license is preserved in `defenses/vlmshield/LICENSE`.
The released classifier is SHA-256 verified and stored under
`data/method_assets/VLMShield/checkpoints/`, outside tracked source. The
existing Transformers loader uses the pinned OpenAI CLIP ViT-L/14 snapshot.
Integration changes and resource provenance are documented in
`defenses/vlmshield/UPSTREAM.md`.

The `compositions/` package adapts the original simulator's composition
contracts, deterministic sampling, source-record schema and measured renderer.
It invokes the already documented attack implementations and labels all
composed samples `adapted_composition`, without claiming paper reproduction.
The copied Noto Sans CJK font is verified against the original pinned
`notofonts/noto-cjk` revision `f8d157532fbfaeda587e826d4cd5b21a49186f7c`.
Its SIL Open Font License notice is retained as
`data/method_assets/fonts/NotoSansCJK-OFL.txt`. JPS transfer assets retain their
original result-manifest hashes; imported CS-DJ results record the source run
and local contrast hashes. Composition details are documented in
`compositions/README.md`.

Upstream licenses verified against each project's `LICENSE` file (and
README, for projects that ship no `LICENSE`) on GitHub at the time of
writing.

> ℹ️ "No explicit license" means that, at the time of review, the upstream
> repository did not contain a `LICENSE` file or equivalent notice.
> Absence of a license does **not** grant any rights to use, copy,
> modify, or redistribute the work. Please contact the upstream authors
> if you need rights beyond what they explicitly permit.

---

## 2. Vendored / Adapted Target-Model Code

The simulator can be configured to load one of several Vision-Language
Models. The wrapper or model-loading code for each is vendored in this
repository.

| Component (path in this repo)              | Upstream                                                                              | Upstream License (as observed) | Used by `simulator_test.sh`? |
| ------------------------------------------ | ------------------------------------------------------------------------------------- | ------------------------------ | ---------------------------- |
| `multimodalmodels/minigpt4/`               | [Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)                      | BSD-3-Clause                   | Yes (target for FigStep, Visual_Adv) |
| `multimodalmodels/llava/`                  | [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) (via Hugging Face `llava-hf/llava-1.5-7b-hf`) | Apache-2.0 (code)              | Yes (target for ImgJP, DeltaJP, Hades, JOOD, BestOfN) |
| `multimodalmodels/qwen/`                   | [QwenLM/Qwen-VL](https://github.com/QwenLM/Qwen-VL)                                    | **Tongyi Qianwen License Agreement** | Not used by default; loaded by `import multimodalmodels` |
| `multimodalmodels/instructblip/`           | [salesforce/LAVIS](https://github.com/salesforce/LAVIS)                                | BSD-3-Clause                   | Not used by default; loaded by `import multimodalmodels` |

> ℹ️ Because `multimodalmodels/__init__.py` imports every wrapper module,
> the vendored Qwen-VL and InstructBLIP code is loaded into Python
> whenever any simulator entry point runs, even though `simulator_test.sh`
> only exercises MiniGPT-4 and LLaVA v1.5. Their upstream licenses
> therefore still apply to this repository.

---

## 3. Pre-Trained Model Weights Downloaded or Loaded at Runtime

The simulator downloads or loads the following third-party model weights
during a default run. Weights are **not** redistributed in this
repository; users must obtain them directly from the upstream provider
and accept the applicable terms.

| Component                                    | Upstream                                                                                                                  | Upstream License (as observed)                                                                                | Used by                                  |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| HarmBench classifier (Llama-2-13b-based)     | [https://huggingface.co/cais/HarmBench-Llama-2-13b-cls](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls)             | **Llama 2 Community License** (Llama 2 derivative); HarmBench framework: MIT                                  | `evaluate_completions.py` (every attack) |
| LLaVA v1.5 (7B) — HF release                 | [https://huggingface.co/llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)                         | LLaVA code: Apache-2.0; weights inherit upstream restrictions (Vicuna / **Llama 2 Community License** family) | ImgJP, DeltaJP, Hades, JOOD, BestOfN     |
| MiniGPT-4 checkpoint (`checkpoints/pretrained_minigpt4.pth`) and the underlying **Vicuna v0** language-model weights | [Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) (user must download separately)                          | MiniGPT-4: BSD-3-Clause; Vicuna weights are based on the original LLaMA and inherit the **LLaMA Research License (non-commercial)** and the **Vicuna delta weights non-commercial terms**. | FigStep, Visual_Adv                      |
| PixArt-α XL-2-1024-MS                        | [https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS)         | Refer to the upstream model card (PixArt-α release terms; built on the OpenRAIL family)                       | Hades (text-to-image generation step)    |

The Llama 2 Community License contains specific restrictions (including
acceptable-use and large-scale-deployment clauses); review the full text
before any use or redistribution. The Vicuna-derived MiniGPT-4 pipeline
inherits LLaMA / Vicuna **non-commercial** terms. Other model families
may impose their own commercial-use limitations.

---

## 4. Closed-Source APIs Called at Runtime

Hades and optional response grading make remote calls to closed-source LLM
APIs. JOOD input generation runs locally. API use is governed by the provider's Terms of Service
and pricing, not by this repository's MIT License.

| Provider | Model identifier used in code                            | Used by              |
| -------- | --------------------------------------------------------- | -------------------- |
| OpenAI   | `gpt-3.5-turbo` (caption / keyword / category generation) | Hades attack         |
| OpenAI   | `gpt-4o-mini` (configurable quality judge) | Optional response grading |

Hades reads API keys from `configs/api_keys.yaml`; response grading uses
`OPENAI_API_KEY`. The
provider's content, rate-limit, and acceptable-use policies apply
independently of this repository's license.

---

## 5. Datasets

The simulator reads the following datasets at runtime. Where data files
are vendored in this repository, they were copied or adapted from the
upstream sources listed.

| Component                                                  | Upstream                                                                                                | Upstream License (as observed) | Vendored here? |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------ | -------------- |
| HarmBench inputs and evaluation assets (`data/original/HarmBench/`, `data/original/optimizer_targets/`, `data/original/classifier_val_sets/`, `data/original/copyright_classifier_hashes/`) | [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench) | MIT | Yes |
| AdvBench harmful-behavior CSVs (`data/original/advbench/`, `data/benchmarks/advbench/`) | [llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks) | MIT | Yes |
| `data/original/derogatory_corpus.csv` (used by Visual_Adv / ImgJP) | Derived from the Visual Adversarial Examples upstream repo (see §1) | No explicit license | Yes |
| `data/original/clean.jpeg` (Visual_Adv base image) | Per the Visual Adversarial Examples upstream repo | Refer to upstream | Yes |

Imported benchmark inputs are stored in `data/benchmarks/`, and reusable attack
resources in `data/method_assets/`. JOOD's configuration reads
`data/method_assets/JOOD/`; the original public AdvBench-M release is stored
separately in `data/benchmarks/AdvBenchM/`. These local data directories retain
the source manifests, license notices and provenance records and are excluded
from Git. See [DATA_LAYOUT.md](DATA_LAYOUT.md).

---

## 6. Python Packages (selected, actually imported at runtime)

The simulator imports the following major third-party Python packages
during the default test pipeline. Consult each package's own
distribution metadata for the authoritative license text and complete
dependency tree.

| Component                                  | Upstream                                                                                                       | License (typical) |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------- | ----------------- |
| `torch` / `torchvision` / `torchaudio`     | [pytorch/pytorch](https://github.com/pytorch/pytorch)                                                          | BSD-3-Clause      |
| `transformers`                             | [huggingface/transformers](https://github.com/huggingface/transformers)                                        | Apache-2.0        |
| `accelerate`                               | [huggingface/accelerate](https://github.com/huggingface/accelerate)                                            | Apache-2.0        |
| `diffusers` (used by Hades for PixArt-α)   | [huggingface/diffusers](https://github.com/huggingface/diffusers)                                              | Apache-2.0        |
| `spacy` (used by `eval_utils.py`)          | [explosion/spaCy](https://github.com/explosion/spaCy)                                                          | MIT               |
| `datasketch` (used by `eval_utils.py`)     | [ekzhu/datasketch](https://github.com/ekzhu/datasketch)                                                        | MIT               |
| `opencv-python-headless` (used by BestOfN) | [opencv/opencv-python](https://github.com/opencv/opencv-python)                                                | Apache-2.0 (with MIT components) |
| `Pillow`                                   | [python-pillow/Pillow](https://github.com/python-pillow/Pillow)                                                | MIT-CMU / HPND    |
| `PyYAML`                                   | [yaml/pyyaml](https://github.com/yaml/pyyaml)                                                                  | MIT               |
| `httpx`                                    | [encode/httpx](https://github.com/encode/httpx)                                                                | BSD-3-Clause      |
| `openai` (Python SDK; Hades + optional response grading)        | [openai/openai-python](https://github.com/openai/openai-python)                                                | Apache-2.0        |
| `tqdm`, `numpy`, `pandas`, `matplotlib`, `scikit-image` | (standard scientific Python stack)                                                                 | BSD / PSF / MIT (per upstream) |

Other packages listed in `requirements.txt` (e.g., `peft`,
`webdataset`, `wandb`, `decord`, `sentencepiece`, `protobuf`,
`beautifulsoup4`, `anthropic`, `visual_genome`, `git+https://github.com/openai/CLIP.git`)
are installed by the environment setup but are not exercised by the
default attack pipelines in `simulator_test.sh`. They remain governed by
their own upstream licenses if you choose to use them.

---

## 7. External Assets

| Component   | Source / Distribution                                                                                                            | Notes                                                                                  |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `Arial.ttf` | Required by the Hades attack (image text rendering); downloaded by the user from a third-party mirror at install time (see README). | Arial is a proprietary typeface owned by Monotype / Microsoft. Confirm that you have the right to use the font in your environment, or substitute a font you are licensed to use, before running the Hades pipeline. |

---

The StrongREJECT rubric and system prompt in `evaluation/evaluation_prompts/`
are copied unchanged from [dsbowen/strong_reject](https://github.com/dsbowen/strong_reject)
at revision `7a551d5b440ec7b75d4f6f5bb7c1719965b76b47` (MIT, Dillon Bowen).
The directory includes the upstream license, source provenance and prompt hashes.
Other quality rubrics and the JSON adapter are simulator code.

## 8. Notes for Users and Redistributors

- This repository does **not** vendor most of the listed third-party
  packages, model weights, or fonts. Users obtain them directly from the
  upstream sources.
- "No explicit license" entries above do **not** grant any rights;
  absent an explicit license, default copyright applies. Contact the
  upstream authors if you need to use such material.
- **Non-commercial** terms (e.g., the LLaMA Research License /
  Vicuna delta-weights terms that flow into MiniGPT-4) and **community
  licenses** (e.g., the Llama 2 Community License covering the HarmBench
  classifier and the LLaVA v1.5 weights) impose restrictions beyond
  permissive open-source norms. Do not assume commercial use is
  permitted.
- The OpenAI APIs used by Hades and optional response grading are subject to OpenAI's Terms
  of Use, content policy, and pricing, none of which are granted by this
  repository's MIT License.
- If you redistribute binaries, containers, fine-tuned weights,
  evaluation outputs, or packaged environments built from this
  repository, you are responsible for satisfying the notice, attribution,
  and license obligations of every bundled or referenced third-party
  component.
- The MIT License in [LICENSE](LICENSE) covers only the original code
  and documentation directly developed by ai.resilience as part of the
  VLM-Jailbreak-Simulator project.
