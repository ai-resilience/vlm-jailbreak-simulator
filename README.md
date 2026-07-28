
# VLM-Jailbreak-Simulator

This official open-source repository serves as a collaborative space for the participating institutions of the research project entitled *"Development of Resilient AI Technology Against Risk Factors,"* dedicated to disclosing and sharing key research outcomes and core software (SW).

**VLM-Jailbreak-Simulator** is a simulation framework for evaluation of multi-modal jailbreak attacks on vision-language models (VLMs). The framework allows users to configure attack methods, target models, and datasets to conduct controlled simulations and comparative analyses.

Jailbreak attacks are categorized into two types:

- **Image-based attacks** — manipulations relying solely on visual perturbations.  
- **Multi-modal attacks** — methods that jointly manipulate both visual and textual modalities through combined image and prompt-based strategies.

Each category includes recent state-of-the-art jailbreak implementations, enabling reproducible and extensible evaluations under a unified experimental setting.

> **License notice (please read before use).** The MIT License in
> [LICENSE](LICENSE) covers **only** the source code and documentation
> directly developed by ai.resilience as part of this project. When the
> simulator runs, it also vendors copies of third-party attack code,
> downloads third-party model weights, reads third-party datasets,
> renders text with a third-party font, and calls closed-source LLM
> APIs. Those components are **NOT** covered by the MIT License — they
> remain governed by their **original upstream terms**, which include
> the **Llama 2 Community License** (HarmBench classifier, LLaVA v1.5),
> the **LLaMA Research License / Vicuna non-commercial terms** (used
> via MiniGPT-4), the **Tongyi Qianwen License** (vendored Qwen-VL
> code), **Apache-2.0 / BSD-3-Clause / MIT** (most other vendored
> attack and model code), the **OpenAI Terms of Use** (Hades / JOOD
> API calls), the proprietary copyright of the **Arial** typeface
> (Hades font asset), and the upstream copyrights of a few attack
> repositories that carry **no explicit license at all** (notably
> Visual Adversarial Examples and Jailbreaking-Attack-against-MLLM).
> Users and redistributors are responsible for reviewing and complying
> with each upstream license. See [THIRD_PARTY.md](THIRD_PARTY.md) for
> a runtime-oriented inventory.


## Multi-model jailbreak attack
|                                                    **Method**                                                   |       **Source**      | **Key Properties**                                         | 
|:---------------------------------------------------------------------------------------------------------------:|:---------------------:|------------------------------------------------------------|
| FigStep | [FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts](https://arxiv.org/abs/2311.05608)  | Generation-based|
| VisualAdv | [Visual Adversarial Examples Jailbreak Aligned Large Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/30150) | Optimization-based|
| ImgJP | [Jailbreaking Attack against Multimodal Large Language Model](https://arxiv.org/abs/2402.02309) | Optimization-based |
| DeltaJP | [Jailbreaking Attack against Multimodal Large Language Model](https://arxiv.org/abs/2402.02309) | Optimization-based |
| Hades | [Images are achilles’ heel of alignment: Exploiting visual vulnerabilities for jailbreaking multimodal large language models](https://arxiv.org/abs/2403.09792) | Generation-based|
| BEST-OF-N | [BEST-OF-N JAILBREAKING](https://arxiv.org/abs/2412.03556)|Generation-based|
| JOOD | [Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy](https://arxiv.org/abs/2503.20823)|Generation-based|


## Usage
### Installation
```
conda create -n MMJ-env python=3.10
conda activate MMJ-env
pip install -r requirements.txt
python -m spacy download en_core_web_sm
curl -L -o Arial.ttf "https://github.com/JotJunior/PHP-Boleto-ZF2/raw/master/public/assets/fonts/arial.ttf"  (requirement for Hades)
```

> ⚠️ **Note on `Arial.ttf`.** Arial is a proprietary typeface (Monotype /
> Microsoft) and is **not** distributed as part of this repository. The
> download command above fetches the font from a third-party mirror for
> convenience only. Confirm that you have the right to use the Arial font
> in your environment, or substitute a font you are licensed to use,
> before running the HADES attack pipeline.


### Step 1 - Generate Test Cases
In the first step, jailbreak attack techniques are used to generate test cases with `generate_test_cases.py`. You can change the arguments below.
```
python generate_test_cases.py --method_name FigStep --save_dir test_cases/FigStep_test_cases
```

### Step 2 - Generate Completions
After generating test cases，we can generate completions for a target model.
```
python generate_completions.py --test_cases_path test_cases/FigStep_test_cases/FigStep_test_cases.json --save_path completions/FigStep_completions.json --model_name minigpt4
```

### Step 3 - Evaluate Completions
After generate completions from a `target_model` from Step 2, We will utilize the classifier provided by HarmBench to label whether each completion is an example of its corresponding behavior. Check ASR.txt
```
python evaluate_completions.py --method_name FigStep --cls_path cais/HarmBench-Llama-2-13b-cls --behaviors_path data/behavior_datasets/harmbench_behaviors_text_all.csv --completions_path completions/FigStep_completions.json --save_path results/FigStep_results.json --include_advbench_metric
```

### To evaluate all availbale models, run the commands below:
```
./simulator_test.sh
```

## Acknowledgement and citation
We thank the following open-source reposities. Each remains the property of
its upstream authors and is governed by its own license (see
[THIRD_PARTY.md](THIRD_PARTY.md) for license details).

    [1]  https://github.com/thuccslab/figstep
    [2]  https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models
    [3]  https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model
    [4]  https://github.com/AoiDragon/HADES
    [5]  https://github.com/jplhughes/bon-jailbreaking
    [6]  https://github.com/naver-ai/JOOD
    [7]  https://github.com/isXinLiu/MM-SafetyBench
    [8]  https://github.com/centerforaisafety/HarmBench


## License

This project is released under a **dual-scope** licensing arrangement:

- The original source code and documentation directly developed by
  ai.resilience as part of the VLM-Jailbreak-Simulator project are
  released under the **MIT License** (see [LICENSE](LICENSE)).
- All third-party source code (vendored or imported), pre-trained model
  weights, datasets, closed-source APIs, and assets that the simulator
  exercises at runtime are **NOT** covered by the MIT License. They
  remain subject to their **original upstream licenses**, notices, and
  use restrictions, including (without limitation):

  - the **Llama 2 Community License** — governs the
    `cais/HarmBench-Llama-2-13b-cls` classifier used by
    `evaluate_completions.py` and the `llava-hf/llava-1.5-7b-hf`
    weights used as a target VLM;
  - the **LLaMA Research License** and **Vicuna delta-weights terms**
    (*non-commercial*) — flow into the MiniGPT-4 target pipeline via
    Vicuna v0 weights;
  - the **Tongyi Qianwen License Agreement** — governs the vendored
    Qwen-VL wrapper code under `multimodalmodels/qwen/`;
  - **Apache-2.0** / **BSD-3-Clause** terms — govern vendored
    MiniGPT-4, LLaVA, InstructBLIP / LAVIS, JOOD, HarmBench
    components, and most Python dependencies;
  - **MIT** — governs the FigStep, HADES, BoN, and HarmBench upstream
    repositories that the simulator vendors or imports from;
  - the **OpenAI Terms of Use** — govern the `gpt-3.5-turbo` and
    `gpt-4-turbo-2024-04-09` API calls made by the Hades and JOOD
    pipelines;
  - the original copyright of the **Arial** typeface (Monotype /
    Microsoft) — applies to the `Arial.ttf` font asset required by
    Hades;
  - **No explicit license** — applies to the Visual Adversarial
    Examples and Jailbreaking-Attack-against-MLLM upstream
    repositories. Absence of a license does **not** grant any rights;
    default copyright applies.

Users and redistributors are solely responsible for reviewing and
complying with the upstream license of every third-party component they
download, use, fine-tune, or redistribute through this repository. See
[THIRD_PARTY.md](THIRD_PARTY.md) for a runtime-oriented inventory of
the third-party components actually exercised by this project.

