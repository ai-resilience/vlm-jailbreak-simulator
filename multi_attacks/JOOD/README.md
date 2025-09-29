
# 🤡 JOOD (CVPR 2025)
Official implementation for "Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy"

<p align="center"> <img src='.assets/concept_figure.png' align="center" width="600px"> </p>

[Joonhyun Jeong](https://bestdeveloper691.github.io/)<sup>1,2</sup>, Seyun Bae<sup>2</sup>, [Yeonsung Jung](https://yeonsungjung.github.io)<sup>2</sup>, Jaeryong Hwang<sup>3</sup>, [Eunho Yang](https://scholar.google.com/citations?user=UWO1mloAAAAJ&hl=en)<sup>2,4</sup><br>

<sup>1</sup> <sub>NAVER Cloud, ImageVision</sub><br />
<sup>2</sup> <sub>KAIST</sub><br />
<sup>3</sup> <sub>Republic of Korea Naval Academy</sub><br />
<sup>4</sup> <sub>AITRICS</sub><br />

[![](https://img.shields.io/badge/CVPR-2025-blue)](https://cvpr.thecvf.com)
[![Paper](https://img.shields.io/badge/Paper-arxiv.2503.20823-red)](https://arxiv.org/abs/2503.20823)
[![YouTube](https://img.shields.io/badge/YouTube-video-green)](https://www.youtube.com/watch?v=9-dNyCYBAqI&t=7s)

## 🛠️ Install

- Python >= 3.12.7
- Required libraries are listed in `requirements.txt`

## 📂 Dataset

Download [AdvBench-M](https://arxiv.org/abs/2402.02309) dataset from [[Google Drive](https://drive.google.com/drive/folders/1Mgx3jRd9slB4FhwCDrwuMTTifB6NtNAO?usp=sharing)].
<br>
Format the dataset directory structure as below:<br>

```plaintext
datasets/
└── AdvBenchM/
    ├── images/
    │   ├── harmful/
    │   ├── harmless/
    │   └── harmless_text/
    ├── prompts/
    │   ├── all_instructions
    │   ├── all_instructions_harmful_annotated
    │   └── eval_all_instructions
    ├── scenario_def.json
    └── scenario_repr.json
```

## 🚀 How to Attack?

### Text Attacks

```shell
bash scripts/text_attacks/attack_gpt4.sh
```

### Multimodal Attacks

```shell
bash scripts/multimodal_attacks/attack_gpt4.sh
```

- for attack with Typography images, modify ```--harmless_image_dir``` to ```datasets/AdvBenchM/images/harmless_text```

### Supported Target Models
We currently support the following target attack models. You can set the ```target_model``` in the script as shown below:<br>

- ```gpt-4-turbo-2024-04-09```
- ```gpt-4o-2024-08-06```
- ```o1-2024-12-17```
- ```qwenvl2``` (Qwen/Qwen2-VL-7B-Instruct)

💡 Note: For OpenAI models, ensure that you set the correct ```openai_key``` in all the scripts.

## 📊 How to evaluate?

### Inference for ASR

```shell
bash scripts/evaluation/eval_llama_guard.sh
```

- Make sure to modify ```eval_datetime``` and ```aug``` to match the settings you used for the attack.<br>
- You need to set your huggingface access token ```HF_TOKEN``` in the script.
- To run the ASR evaluation using the Meta-Llama-Guard-2-8B model, you must first agree to its license terms provided by Meta:

  👉 [Meta Llama Guard 2 License Agreement](https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B)
  - And then, set ```model_id``` to ```meta-llama/Meta-Llama-Guard-2-8B``` in the [models/llm_guard.py](./models/llm_guard.py#L28)

### Inference for Harmfulness

```shell
bash scripts/evaluation/eval_gpt4.sh
```

- Make sure to modify ```eval_datetime``` and ```aug``` to match the settings you used for the attack.<br>
- You need to set your openai key ```openai_key``` in the script.

### Report Metrics

```
python3 evaluate_metrics.py --eval_dir [YOUR_RESULT_DIR]
```

- Set ```--eval_dir``` with your result directory (e.g., ```datasets/AdvBenchM/outputs/2024_08_28_06_47_30_mixup```)

## 🗞️ News
- **`2025/02/27`**: got accepted in CVPR'25 :partying_face:
- **`2025/06/11`**: open JOOD code

## 🔗 Citation

If you find that this project helps your research, please consider citing as below:

```
@inproceedings{jeong2025playing,
  title={Playing the fool: Jailbreaking llms and multimodal llms with out-of-distribution strategy},
  author={Jeong, Joonhyun and Bae, Seyun and Jung, Yeonsung and Hwang, Jaeryong and Yang, Eunho},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={29937--29946},
  year={2025}
}
```

## 🙏 Acknowledgements

We gratefully acknowledge the following projects and datasets, which our work builds upon:

- [AdvBench](https://github.com/llm-attacks/llm-attacks) – for the design of harmful instruction scenarios.
- [AdvBench-M](https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model) – for the image-based multimodal jailbreak evaluation data.

## License
```
JOOD
Copyright (c) 2025-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
