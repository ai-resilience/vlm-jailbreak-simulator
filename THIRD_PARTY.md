# Third-Party Software, Models, Datasets, and Assets

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
| `image_attacks/visual_adv/`          | [Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models)   | No explicit license            |
| `image_attacks/imgjp/`, `image_attacks/deltajp/` | [abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model](https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model) | No explicit license            |
| `multi_attacks/hades/`               | [AoiDragon/HADES](https://github.com/AoiDragon/HADES)                                                                                                           | **MIT**                        |
| `multi_attacks/best_of_n/`           | [jplhughes/bon-jailbreaking](https://github.com/jplhughes/bon-jailbreaking) — at runtime it also imports the upstream `bon` Python package                       | **MIT**                        |
| `multi_attacks/JOOD/`                | [naver-ai/JOOD](https://github.com/naver-ai/JOOD) — full upstream `LICENSE` file is preserved at `multi_attacks/JOOD/LICENSE`                                    | **Apache-2.0**                 |

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

The Hades and JOOD pipelines make remote calls to closed-source LLM
APIs. Use of these APIs is governed by the provider's Terms of Service
and pricing, not by this repository's MIT License.

| Provider | Model identifier used in code                            | Used by              |
| -------- | --------------------------------------------------------- | -------------------- |
| OpenAI   | `gpt-3.5-turbo` (caption / keyword / category generation) | Hades attack         |
| OpenAI   | `gpt-4-turbo-2024-04-09` (default JOOD target / proposer) | JOOD attack          |

You must supply your own API key via `configs/api_keys.yaml`. The
provider's content, rate-limit, and acceptable-use policies apply
independently of this repository's license.

---

## 5. Datasets

The simulator reads the following datasets at runtime. Where data files
are vendored in this repository, they were copied or adapted from the
upstream sources listed.

| Component                                                  | Upstream                                                                                                | Upstream License (as observed) | Vendored here? |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------ | -------------- |
| HarmBench behavior datasets (`data/behavior_datasets/*.csv`, `data/multimodal_behavior_images/`, `data/optimizer_targets/`, `data/classifier_val_sets/`, `data/copyright_classifier_hashes/`) | [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench) | MIT                            | Yes            |
| AdvBench harmful-behavior CSVs (`data/advbench/`)          | [llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks)                                    | MIT                            | Yes            |
| `data/derogatory_corpus.csv` (used by Visual_Adv / ImgJP)  | Derived from the Visual Adversarial Examples upstream repo (see §1)                                     | No explicit license            | Yes            |
| `data/clean.jpeg` (Visual_Adv base image)                   | Per the Visual Adversarial Examples upstream repo                                                       | Refer to upstream              | Yes            |

> ℹ️ JOOD's `configs/method_configs/JOOD.yaml` additionally references
> an external **AdvBenchM** image / prompt set (`datasets/AdvBenchM/...`).
> AdvBenchM is **not** vendored in this repository; users must obtain it
> from the JOOD upstream and comply with its terms before running the
> JOOD pipeline.

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
| `openai` (Python SDK; Hades + JOOD)        | [openai/openai-python](https://github.com/openai/openai-python)                                                | Apache-2.0        |
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
- The OpenAI APIs used by Hades and JOOD are subject to OpenAI's Terms
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
