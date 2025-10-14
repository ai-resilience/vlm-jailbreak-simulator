
# VLM-Jailbreak-Simulator
**VLM-Jailbreak-Simulator** is a simulation framework for evaluation of multi-modal jailbreak attacks on vision-language models (VLMs). The framework allows users to configure attack methods, target models, and datasets to conduct controlled simulations and comparative analyses.

Jailbreak attacks are categorized into two types:

- **Image-based attacks** — manipulations relying solely on visual perturbations.  
- **Multi-modal attacks** — methods that jointly manipulate both visual and textual modalities through combined image and prompt-based strategies.

Each category includes recent state-of-the-art jailbreak implementations, enabling reproducible and extensible evaluations under a unified experimental setting.


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
We thank the following open-source reposities.

    [1]  https://github.com/thuccslab/figstep
    [2]  https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models
    [3]  https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model
    [4]  https://github.com/AoiDragon/HADES
    [5]  https://github.com/jplhughes/bon-jailbreaking
    [6]  https://github.com/naver-ai/JOOD
    [7]  https://github.com/isXinLiu/MM-SafetyBench
    [8]  https://github.com/centerforaisafety/HarmBench


