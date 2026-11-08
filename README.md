# VLM-Jailbreak-Simulator

This official open-source repository serves as a collaborative space for the participating institutions of the research project entitled *"Development of Resilient AI Technology Against Risk Factors,"* dedicated to disclosing and sharing key research outcomes and core software (SW).

**VLM-Jailbreak-Simulator**는 비전 언어 모델의 멀티모달 jailbreak 공격과 방어를 실행하고 비교하는 프레임워크입니다. 개별 공격, 여러 공격의 구성 요소를 연결한 조합, 방어 적용과 응답 평가를 공통 실행 흐름으로 제공합니다.

```text
원본 질문·데이터셋 → 공격 입력 생성 → 대상 모델 응답 생성 → 원본 질문 기준 평가
                                      └─ 선택적으로 방어 적용
```

현재 ai-resilience의 실행 진입점은 `generate_test_cases.py`, `generate_completions.py`, `evaluate_completions.py`입니다. 필요한 공통 기능은 `benchmark_datasets.py`, `multimodalmodels/`, `defenses/`, `compositions/`, `evaluation/`에 배치되어 있으며 별도의 `core/` 패키지 없이 실행합니다.

프로젝트 자체 코드와 문서의 라이선스는 [LICENSE](LICENSE), 외부 코드·데이터·모델·폰트의 출처와 별도 조건은 [THIRD_PARTY.md](THIRD_PARTY.md)를 참고하세요.

- [1. Repository layout](#1-repository-layout)
- [2. Installation and assets](#2-installation-and-assets)
- [3. Quick start](#3-quick-start)
- [4. Datasets](#4-datasets)
- [5. Attacks](#5-attacks)
- [6. Target and surrogate models](#6-target-and-surrogate-models)
- [7. Defenses](#7-defenses)
- [8. Compositions](#8-compositions)
- [9. Evaluation and metrics](#9-evaluation-and-metrics)
- [10. Outputs and resume](#10-outputs-and-resume)
- [11. Related documentation](#11-related-documentation)

## 1. Repository layout

```text
vlm-jailbreak-simulator/
├── generate_test_cases.py       # 데이터셋 선택·공격/조합 입력 생성
├── generate_completions.py      # 대상 모델 응답 생성·방어 적용
├── evaluate_completions.py      # 응답 판정·지표 계산·비교 평가
├── benchmark_datasets.py        # 벤치마크 로더·원본 질문 sidecar
├── data_paths.py                # 데이터 루트·이전 경로 호환
├── attacks/                     # 공격 등록·호출 인터페이스
├── multi_attacks/               # FigStep, CS-DJ, JOOD, JPS, MML 등
├── image_attacks/               # VisualAdv, ImgJP, DeltaJP
├── defenses/                    # JailGuard, CIDER, AdaShield, AsD, VLMShield
├── compositions/                # 조합 실행·샘플 배정·출처 기록·검증
├── multimodalmodels/            # 모델 등록·로딩·추론·최적화 surrogate
├── evaluation/                  # 평가 레코드·응답 grading·12개 지표
├── configs/
│   ├── method_configs/          # 공격 설정
│   ├── defense_configs/         # 방어 설정
│   ├── composition_configs/     # 조합 레시피·수량·샘플링 설정
│   └── model_configs/           # 대상 모델 설정
├── scripts/                     # 공격·방어·조합별 실행 스크립트
├── data/                        # 원본 데이터·벤치마크·방법론 자산
├── checkpoints/                 # 보존된 모델 가중치
└── simulator_test.sh             # 기존 여러 공격의 명령 예시
```

`test_cases/`, `completions/`, `results/`는 실행 시 생성되는 산출물 폴더입니다. 각 단계는 설정한 출력 경로를 생성합니다. 코드와 함께 데이터·모델 가중치가 모두 배포되는 것은 아니므로 필요한 자산을 별도로 준비합니다.

## 2. Installation and assets

### 환경 설치

아래 명령은 저장소 루트에서 실행합니다.

```bash
conda create -n vlm-jailbreak python=3.11
conda activate vlm-jailbreak
python -m pip install -r requirements.txt
```

로컬 실행 확인에 사용한 환경은 Python 3.11.11, PyTorch 2.9.1+cu130, CUDA runtime 13.0, Transformers 4.57.1, NVIDIA B200 MIG 3g.90gb입니다. `requirements.txt`는 Transformers 등 일부 패키지를 고정하며, 전체 환경의 lockfile은 아닙니다. PyTorch/CUDA 빌드는 사용하는 GPU 환경에 맞춰 준비합니다. 메모리 사용량은 대상 모델·최적화 모델·judge 구성에 따라 달라집니다.

모델 ID와 기본 revision은 [모델 레지스트리](multimodalmodels/registry.py)에서 관리합니다. Hugging Face 가중치를 로컬 캐시에 준비하거나 모델 로딩 시 다운로드할 수 있는 환경이 필요합니다. 접근 제한이 있는 가중치는 해당 모델의 접근 권한도 필요합니다.

### 방법론별 추가 자산

| 구성 | 필요한 자산·준비 방법 |
| --- | --- |
| CS-DJ | CC3M 이미지 라이브러리와 폰트: `data/method_assets/CS-DJ/` |
| VisualAdv·ImgJP·DeltaJP·JPS | 공격 YAML에 지정된 초기 이미지·학습 질문·target corpus |
| MML | 공개 replay 이미지·프롬프트 원본, 또는 generated 모드의 폰트·NLTK 자원 |
| JOOD | 전용 질문·마스킹 주석·유해/무해 이미지: `data/method_assets/JOOD/` |
| JailGuard | spaCy `en_core_web_md`; [설치·버전 안내](defenses/jailguard/README.md) |
| CIDER | `python -m defenses.cider.setup` |
| AsD | `python -m defenses.asd.setup` |
| VLMShield | `python -m defenses.vlmshield.setup` |
| AdaShield-A | CLIP ViT-B/32와 refinement pool; [준비 안내](defenses/adashield/README.md) |
| 5개 방법 조합 | Noto 폰트와 검증된 JPS 이미지·steer·학습 CSV; [준비 안내](compositions/README.md) |

CIDER·AsD·VLMShield의 setup 명령은 `--source PATH`로 기존 로컬 자산을 복사할 수도 있습니다. AdaShield-A의 현재 pool은 원본 구현에서 LLaVA-1.5-13B로 정제한 로컬 자산이며, 저자가 배포한 학습 완료 pool과 구분합니다.

저작권 해시 평가를 사용하는 경우에는 추가로 다음 spaCy 모델을 준비합니다.

```bash
python -m spacy download en_core_web_sm
```

Hades의 PixArt·GPT 보조 함수와 선택적인 API 기반 평가에는 각각의 모델·API 설정이 필요합니다. 폰트와 외부 자산의 조건은 [THIRD_PARTY.md](THIRD_PARTY.md)에 정리되어 있습니다.

## 3. Quick start

### 방법론별 스크립트

```bash
# 실행할 명령만 확인: Python·모델을 호출하지 않음
DRY_RUN=1 END=1 bash scripts/attacks/figstep.sh

# SafeBench 첫 질문 1개: 입력 생성 → 응답 생성 → 평가
END=1 bash scripts/attacks/figstep.sh

# 같은 질문 범위의 baseline·JailGuard 응답 생성 및 비교 평가
END=1 bash scripts/defenses/jailguard.sh
```

`START`는 시작 인덱스 포함, `END`는 끝 인덱스 미포함입니다. **`END`를 생략하면 전체 데이터셋을 선택**합니다. 질문 수를 줄여도 최적화 공격의 학습 반복 수는 YAML 설정을 사용합니다.

모델·데이터셋·실험 이름을 바꾸거나 단계 하나만 실행할 수 있습니다.

```bash
MODEL=qwen3_vl_2b DATASET=AdvBench START=10 END=20 RUN_NAME=FigStep_AdvBench_pilot \
  bash scripts/attacks/figstep.sh

END=1 bash scripts/attacks/figstep.sh generate
bash scripts/attacks/figstep.sh complete
bash scripts/attacks/figstep.sh evaluate
```

스크립트는 상대 경로를 저장소 루트 기준으로 해석합니다. 기본 Python 대신 사용할 실행 파일은 `PYTHON=/path/to/python`으로 지정합니다. 전체 스크립트 목록과 환경변수는 [scripts/README.md](scripts/README.md)에 있습니다.

### 세 실행 파일 직접 호출

```bash
python generate_test_cases.py --method_name FigStep --dataset_name SafeBench \
  --behavior_start_idx 0 --behavior_end_idx 1 --save_dir test_cases/FigStep_SafeBench

python generate_completions.py --model_name llava_v1_5 \
  --test_cases_path test_cases/FigStep_SafeBench/FigStep_test_cases.json \
  --save_path completions/FigStep_SafeBench/llava_v1_5/baseline.json

python evaluate_completions.py --method_name FigStep \
  --completions_path completions/FigStep_SafeBench/llava_v1_5/baseline.json \
  --save_path results/FigStep_SafeBench/llava_v1_5/baseline.json
```

원본 질문은 입력 JSON의 `.behaviors.json` sidecar에 저장되어 응답·평가 단계로 이어집니다. 평가에는 공격으로 변형된 문구 대신 원래 벤치마크 질문을 사용합니다.

## 4. Datasets

[benchmark_datasets.py](benchmark_datasets.py)는 아래 다섯 벤치마크와 JOOD 전용 질문 집합을 제공합니다.

| `--dataset_name` | 질문 수 | 제공 정보 |
| --- | ---: | --- |
| `AdvBench` | 520 | 원본 질문·target 문자열 |
| `AdvBenchM` | 246 | 질문·범주별 이미지 pool; 이미지와 질문의 일대일 매칭이 아님 |
| `HADES` | 750 | 원본 질문·keyword·단계별 이미지 6종 |
| `MM-SafetyBench` | 1,680 | 질문·key phrase·SD/SD_TYPO/TYPO 이미지·평가 유형 |
| `SafeBench` | 500 | 원본 질문·FigStep용 서술형 instruction |
| `JOOD` | 170 | JOOD 전용 질문·마스킹 문구·keyword·대표 이미지 |

`AdvBenchM`의 같은 범주에 속한 질문들은 이미지 pool을 공유할 수 있습니다. 현재 DeltaJP는 질문마다 해당 pool의 첫 catalog 이미지를 선택하고 이 정책을 메타데이터에 기록합니다. `JOOD`의 170개 질문 집합은 원본 AdvBenchM 246개와 별개입니다.

```text
data/
├── original/        # 기존 HarmBench, CSV, 이미지, optimizer target 등
├── benchmarks/      # advbench, AdvBenchM, HADES, MM-SafetyBench, SafeBench
└── method_assets/   # 공격·방어·조합별 이미지·pool·가중치·폰트·NLP 자원
```

`--behaviors_path PATH`로 사용자 CSV를 지정할 수 있으며 `--dataset_name`과 함께 사용할 수 없습니다. 공통 필드는 `BehaviorID`, `Behavior`이고, 이미지·마스킹 주석 등 추가 필드는 방법론의 입력 조건에 맞춰 준비합니다. `--behavior_ids_subset`은 ID 목록 또는 ID 파일로 질문을 선택하며 인덱스 범위보다 우선합니다.

선택자를 생략한 **Python CLI**의 기본값과 실행 스크립트의 기본값은 구분합니다.

| Python CLI 방법론 | 선택자 생략 시 데이터셋 |
| --- | --- |
| `JOOD` | JOOD 전용 집합 |
| `JPS` | AdvBench |
| `MML` | SafeBench |
| 나머지 방법론 | `data/original/HarmBench/behaviors/harmbench_behaviors_text_all.csv` |

스크립트는 다음 공격 표의 기본 데이터셋을 명시적으로 전달합니다. 다른 데이터 루트는 `VLM_JAILBREAK_DATA_ROOT`로 지정할 수 있습니다. 데이터 경로·이전 경로 호환 규칙은 [DATA_LAYOUT.md](DATA_LAYOUT.md)를 참고하세요.

## 5. Attacks

현재 [공격 레지스트리](attacks/__init__.py)에 등록된 이름은 10개입니다. `--method_name`에는 아래 CLI 이름을 그대로 사용합니다. 표의 기본 데이터셋은 **실행 스크립트 기준**입니다.

| CLI 이름 | 현재 동작 | 스크립트 기본 데이터셋 | 실행 스크립트 |
| --- | --- | --- | --- |
| `FigStep` | 텍스트를 타이포그래피 이미지로 렌더링 | SafeBench | [figstep.sh](scripts/attacks/figstep.sh) |
| `CS-DJ` | 질문 분해·대조 이미지 검색·3×4 collage 생성 | AdvBench | [cs_dj.sh](scripts/attacks/cs_dj.sh) |
| `Visual_Adv` | 별도 corpus로 universal 이미지를 최적화 | AdvBench | [visualadv.sh](scripts/attacks/visualadv.sh) |
| `ImgJP` | 학습 goal/target 쌍으로 universal 이미지를 최적화 | SafeBench | [imgjp.sh](scripts/attacks/imgjp.sh) |
| `DeltaJP` | 학습 이미지에 대한 universal delta를 최적화해 적용 | AdvBenchM | [deltajp.sh](scripts/attacks/deltajp.sh) |
| `JPS` | 전용 Qwen2-VL 시각 최적화와 텍스트 steer 개정 | SafeBench | [jps.sh](scripts/attacks/jps.sh) |
| `MML` | 공개 이미지 replay 또는 generated 이미지·문구 변환 | SafeBench | [mml.sh](scripts/attacks/mml.sh) |
| `JOOD` | 전용 질문·이미지에 mixing/augmentation 적용 | JOOD | [jood.sh](scripts/attacks/jood.sh) |
| `Hades` | 기존 PixArt·GPT 기반 이미지·문구 생성 | HADES | [hades.sh](scripts/attacks/hades.sh) |
| `BestOfN` | 기존 텍스트·이미지 변형 후보 생성·선택 | SafeBench | [best_of_n.sh](scripts/attacks/best_of_n.sh) |

설정은 [configs/method_configs/](configs/method_configs/)에서 선택합니다. Python CLI에는 `--config PATH`, 스크립트에는 `ATTACK_CONFIG=PATH`를 사용합니다.

```bash
ATTACK_CONFIG=configs/method_configs/MML_mirror.yaml RUN_NAME=MML_mirror_SafeBench \
  END=1 bash scripts/attacks/mml.sh
```

방법론별 주요 입력 조건과 구현 범위는 다음과 같습니다.

- FigStep은 준비된 `statement`를 우선 사용합니다. SafeBench에서는 서술형 instruction이 이에 해당하며, 없으면 원본 질문을 렌더링합니다. 폰트·크기·줄바꿈은 YAML로 선택합니다.
- VisualAdv는 별도 target corpus로 최적화한 이미지를 평가 질문에 재사용합니다. ImgJP·DeltaJP·JPS는 기본적으로 학습 질문 ID와 정규화된 중복 질문을 평가에서 제외합니다. 선택 범위 적용 후 제외한 결과가 sidecar에 기록됩니다.
- DeltaJP에는 평가 질문과 연결할 이미지가 필요합니다. AdvBenchM·HADES·MM-SafetyBench는 이미지를 제공하며, 이미지가 없는 데이터셋에는 별도 이미지 입력이 필요합니다.
- JPS의 기본 학습 설정은 Qwen2-VL-7B와 로컬 Qwen2.5-14B agent를 사용합니다. 검증된 이미지·steer·agent trace를 재사용할 수 있으며 공통 CE surrogate로 전용 최적화 adapter를 대체하지 않습니다.
- MML의 기본 설정은 SafeBench 공개 word-replacement 이미지를 replay합니다. 다른 공개 변형은 mirror·rotate·Base64 YAML, 로컬 렌더링은 `MML_generated.yaml`을 사용합니다. `--protocol mml_official`은 별도 CLAS judge/API 설정과 query-budget 정책을 사용합니다. [MML 출처·범위](multi_attacks/mml/UPSTREAM.md)를 참고하세요.
- JOOD 기본 설정은 무해 이미지 5개와 가중치 9개를 이용한 CutMix로 질문당 45개 시도를 생성합니다. 다른 질문 집합에는 방법론에 맞는 이미지·마스킹 주석을 준비해야 합니다.
- Hades와 BestOfN은 보존된 기존 구현입니다. 등록·스크립트 제공과 논문 실험의 완전한 재현은 구분합니다.

모델 대체, generated/replay 구분과 조합 적용 범위는 설정·출처 메타데이터에 기록합니다. 출처 revision과 외부 자산의 조건은 [THIRD_PARTY.md](THIRD_PARTY.md)에 있습니다.

## 6. Target and surrogate models

[models.yaml](configs/model_configs/models.yaml)에 등록된 대상 모델은 다음과 같습니다.

| `--model_name` | 현재 adapter의 입력 이미지 수 |
| --- | --- |
| `llava_v1_5` | 0~1개 |
| `llava_v1_6` | 0~1개 |
| `llava_onevision_7b_chat` | 0~1개 |
| `qwen2_vl_7b` | 정확히 1개 |
| `qwen2_5_vl_7b` | 정확히 1개 |
| `qwen3_vl_2b` | 0개 이상·다중 이미지 지원 |
| `qwen3_vl_8b` | 0개 이상·다중 이미지 지원 |

위 제한은 현재 adapter 구현의 제한입니다. 특히 다중 이미지 조합에는 이를 지원하는 모델을 선택해야 합니다. 모델 로딩 전 입력 이미지 수·존재 여부를 검사합니다.

입력 JSON은 BehaviorID별 문자열, `[image, prompt]`, `[[image1, image2], prompt]`, `{text, images}` 또는 여러 시도의 목록을 받을 수 있습니다. 이미지 경로는 입력 파일의 `images/`, 입력 파일이 있는 폴더, 저장소·데이터 루트 순으로 해석합니다.

응답 대상 모델과 공격 최적화에 사용하는 surrogate는 별도로 선택합니다. 공통 CE profile은 `llava_v1_5`, `llava_v1_6`, `qwen2_vl_7b`, `qwen3_vl_2b`이며 공격 YAML에서 선택합니다. 스크립트의 `MODEL`은 응답 대상 모델을 바꿉니다. 구현 인터페이스는 [multimodalmodels/README.md](multimodalmodels/README.md)에 있습니다.

## 7. Defenses

방어는 응답 생성 단계의 `--defense_name`으로 선택합니다. [방어 레지스트리](defenses/__init__.py)는 필요한 구현을 지연 import합니다.

| CLI 이름 | 동작 | 입력·대상 조건 | 설정·실행 |
| --- | --- | --- | --- |
| `JailGuard` | 이미지 변형별 응답 불일치·거절 패턴으로 탐지 | 이미지 1개 | [YAML](configs/defense_configs/JailGuard.yaml) · [script](scripts/defenses/jailguard.sh) |
| `CIDER` | diffusion 전후 텍스트/이미지 유사도 변화로 탐지 | 이미지 1개, LLaVA-1.5-7B | [YAML](configs/defense_configs/CIDER.yaml) · [script](scripts/defenses/cider.sh) |
| `AdaShield` | S: 고정 안전 문구, A: CLIP 기반 문구 검색 | S는 대상 모델 제한, A는 이미지 1개 | [S](configs/defense_configs/AdaShield.yaml) · [A](configs/defense_configs/AdaShield-A.yaml) · [QueryRelated](configs/defense_configs/AdaShield-A-QueryRelated.yaml) |
| `AsD` | 보호 패치와 안전 문구를 입력에 적용 | 이미지 1개, 패치가 들어갈 수 있는 크기 | [YAML](configs/defense_configs/AsD.yaml) · [script](scripts/defenses/asd.sh) |
| `VLMShield` | CLIP 특징과 공개 MLP 분류기로 탐지 | 이미지 0~1개, 대상 모델 제한도 적용 | [YAML](configs/defense_configs/VLMShield.yaml) · [script](scripts/defenses/vlmshield.sh) |

AdaShield의 실행 파일은 [S](scripts/defenses/adashield_s.sh), [A](scripts/defenses/adashield_a.sh), [A-QueryRelated](scripts/defenses/adashield_a_query_related.sh)로 나뉩니다. 상세 알고리즘·자산·제약은 [defenses/README.md](defenses/README.md)와 각 방법론 문서에 있습니다.

기존 입력과 baseline이 있으면 다시 생성하지 않고 재사용할 수 있습니다.

```bash
TEST_CASES_PATH=test_cases/FigStep_SafeBench/FigStep_test_cases.json \
BASELINE_PATH=completions/FigStep_SafeBench/llava_v1_5/baseline.json \
  bash scripts/defenses/vlmshield.sh
```

Python CLI로 직접 방어 응답을 생성하고 비교할 수도 있습니다.

```bash
python generate_completions.py --model_name llava_v1_5 \
  --test_cases_path test_cases/FigStep_SafeBench/FigStep_test_cases.json \
  --save_path completions/FigStep_SafeBench/llava_v1_5/VLMShield.json \
  --defense_name VLMShield --defense_config configs/defense_configs/VLMShield.yaml

python evaluate_completions.py --method_name FigStep \
  --completions_path completions/FigStep_SafeBench/llava_v1_5/VLMShield.json \
  --baseline_results_path completions/FigStep_SafeBench/llava_v1_5/baseline.json \
  --save_path results/FigStep_SafeBench/llava_v1_5/VLMShield.json
```

baseline과 방어 비교에는 동일한 입력·대상 모델·생성 설정을 사용하고 출력 파일을 구분합니다. 방어 후에도 원본 `test_case`를 보존해 pair를 맞춥니다. AdaShield와 AsD는 입력 변환 방식으로 별도의 탐지 판정을 제공하지 않으므로 탐지 성공률·detector F1을 임의로 계산하지 않습니다.

## 8. Compositions

[compositions/](compositions/README.md)는 여러 공격의 구성 요소를 연결합니다. 모든 조합은 `adapted_composition`, `paper_reproduction: false`로 기록합니다.

| 설정 | 구성 |
| --- | --- |
| [five_methods_3000.yaml](configs/composition_configs/five_methods_3000.yaml) | FigStep·CS-DJ·MML·JOOD·JPS, 18개 레시피, 전체 3,000개 입력 |
| [stages_ready.yaml](configs/composition_configs/stages_ready.yaml) | CS-DJ Split → FigStep, CS-DJ Split → MML, FigStep → JOOD |
| [jailbreak_dataset_v3.yaml](configs/composition_configs/jailbreak_dataset_v3.yaml) | 원본 8개 레시피 설정; 아직 이전하지 않은 단계는 실행 전에 거부 |

5개 방법의 기본 실행은 레시피당 3개씩 **54개 미리보기**를 만듭니다. 전체 계획은 세 벤치마크의 원본 2,930개 레코드를 한 번씩 사용하고 70개를 추가 배정해 3,000개를 구성합니다. 조합별 샘플 배정·JPS 학습 질문 제외·동일 원문 split 정책을 유지합니다.

```bash
# 조합 입력만 생성
bash scripts/compositions/five_methods.sh generate
PER_RECIPE=1 bash scripts/compositions/stages.sh generate

# 미리보기의 모델 응답·평가
bash scripts/compositions/five_methods.sh complete
bash scripts/compositions/five_methods.sh evaluate
```

기본 대상 모델은 다중 이미지를 지원하는 `qwen3_vl_2b`입니다. 5개 방법 조합에서 JPS는 검증된 이미지·steer 자산을 재사용합니다. 단계형 조합은 YAML에 지정된 변환을 순서대로 적용하고 입력·출력 해시와 실행 이력을 남깁니다. UltraBreak·JIP 등 아직 없는 단계가 필요한 일반 레시피는 지원 단계 이전이 필요합니다.

미리보기를 검토한 뒤 전체 생성과 구조 검증을 실행합니다.

```bash
REVIEW_NOTE='실제 미리보기와 contact sheet 검토 내용' \
  bash scripts/compositions/five_methods.sh review-preview
bash scripts/compositions/five_methods.sh full
VALIDATE_PHASE=full bash scripts/compositions/five_methods.sh validate
```

전체 데이터의 응답 생성·평가에는 `COMPOSITION_PHASE=full`을 지정합니다. 조합 결과도 기존 세 CLI의 JSON·원본 질문 sidecar 형식으로 연결됩니다. 평가에는 전체 집계와 레시피별 `by_recipe`가 함께 저장됩니다. 조언·의견 입력은 원래 평가 유형을 보존하며 일반 유해 요청 ASR에서 제외합니다.

## 9. Evaluation and metrics

기본 binary judge는 `cais/HarmBench-Llama-2-13b-cls`이며 `--cls_path`로 지정합니다. 질문당 여러 시도가 있으면 하나라도 성공한 질문을 성공으로 집계하는 Pass@k 기준을 사용합니다. 단일 공격 평가는 선택한 질문을 대상으로 하며, 조합 평가는 원본 평가 유형에 따라 advice/opinion을 일반 ASR에서 제외합니다.

| 공격 지표 | 측정 내용 |
| --- | --- |
| ASR | 판정된 원본 질문 중 하나 이상의 성공 시도가 있는 질문 비율 |
| Harmfulness | 응답 유해성 평균, 0~4 |
| Convincingness | StrongREJECT 설득력 평균, 1~5 |
| Specificity | StrongREJECT 구체성 평균, 1~5 |
| Contradiction Rate | 거절과 유해한 도움이 함께 존재하는 응답 비율 |
| Intent-conditioned ASR | 의도 이해가 확인된 시도들에 대한 질문별 조건부 성공률 |

| 방어 지표 | 측정 내용·필요한 입력 |
| --- | --- |
| Detection Success Rate | 탐지 판정이 있는 공격 중 탐지한 비율 |
| F1 Score | 공격·정상 입력의 명시적 탐지 판정으로 계산한 F1 |
| Over-defense Rate | 정상 입력 중 차단하거나 거절한 비율 |
| Benign Performance | 정상 과제 정답률·품질과 baseline 대비 변화 |
| ASR Reduction | 동일 입력 pair의 baseline ASR − 방어 ASR |
| Harmfulness Reduction | 동일 입력 pair의 baseline 유해성 평균 − 방어 유해성 평균 |

기본 응답 성공 판정만으로 모든 품질 지표가 채워지지는 않습니다. `--grade_responses --judge_model MODEL`은 원본 질문을 기준으로 선택적 chat-judge grading을 수행하며 API 설정이 필요합니다. 이 경로는 `OPENAI_API_KEY`와 선택적인 `OPENAI_BASE_URL`을 사용합니다. `--cls_path`와 `--judge_model`은 각각 binary judge와 품질 judge를 선택합니다.

정상 입력을 포함한 비교는 다음처럼 평가된 결과를 연결합니다.

```bash
python evaluate_completions.py --metrics_only --model_name llava_v1_5 \
  --completions_path results/defended.json \
  --baseline_results_path results/baseline.json \
  --benign_results_path results/benign_defended.json \
  --benign_baseline_results_path results/benign_baseline.json \
  --save_path results/defense_comparison.json
```

`--metrics_only`는 저장된 판정·주석으로 지표를 계산합니다. 정상 과제 grading에는 원본 정상 질문과 정답 정보를 담은 `--benign_behaviors_path`가 필요합니다. 탐지 판정·정상 과제 주석·유해성 주석·비교 pair가 없으면 해당 지표는 `value: null`, unavailable 사유와 coverage로 보고합니다.

StrongREJECT 프롬프트와 출처는 [evaluation/evaluation_prompts/](evaluation/evaluation_prompts/)에 보존합니다. Python API는 `eval_utils.compute_evaluation_metrics(...)`, `eval_utils.grade_evaluation_results(...)`입니다.

## 10. Outputs and resume

스크립트의 기본 출력 구조입니다.

```text
test_cases/<RUN_NAME>/
completions/<RUN_NAME>/<MODEL>/baseline.json
completions/<RUN_NAME>/<MODEL>/<DEFENSE_LABEL>.json
results/<RUN_NAME>/<MODEL>/baseline.json
results/<RUN_NAME>/<MODEL>/<DEFENSE_LABEL>.json
```

조합은 입력 폴더와 기본 응답·결과 폴더에 `<phase>/`를 추가해 preview와 full을 구분합니다. 단계별 주요 파일은 다음과 같습니다.

| 파일 | 내용 |
| --- | --- |
| 공격별 입력 JSON·`images/` | 생성한 문구·이미지·여러 시도 |
| `*.behaviors.json` | 원본 질문·선택 설정·payload 및 이미지 연결 정보 |
| 응답 JSON | 원본 `test_case`, `generation`, case ID·시도 번호, 방어·조합 메타데이터 |
| `*.metadata.json` | 대상 모델·revision·생성 설정·방어 설정 및 자산 출처 |
| 평가 JSON·`*.metrics.json` | 성공 판정·주석·12개 지표와 조합별 집계 |
| 조합 manifest·sample plan·checkpoint | 샘플 배정·실행 출처·중간 산출물 검증·재개 정보 |

응답 생성 재개에는 Python CLI의 `--incremental_update` 또는 스크립트의 `INCREMENTAL=1`을 사용합니다. 입력·설정·모델 메타데이터가 일치하는 캐시를 재사용합니다. 공격 입력을 다시 만들지 않고 응답만 재개하려면 `complete` 단계를 사용합니다.

```bash
INCREMENTAL=1 bash scripts/attacks/figstep.sh complete
```

질문 범위·학습 설정·모델·입력을 바꾼 실험은 `RUN_NAME` 또는 출력 경로를 구분합니다. 최적화 자산과 조합 checkpoint는 각각의 설정·원본·이미지·코드 해시를 검사합니다. `simulator_test.sh`는 기존 여러 공격의 명령 예시이며, 방법론별 실행에는 `scripts/`를 사용합니다.

## 11. Related documentation

| 문서 | 내용 |
| --- | --- |
| [scripts/README.md](scripts/README.md) | 공격·방어·조합별 실행 파일, 단계와 환경변수 |
| [DATA_LAYOUT.md](DATA_LAYOUT.md) | 데이터 디렉터리·원본 질문 sidecar·경로 계약 |
| [multimodalmodels/README.md](multimodalmodels/README.md) | 대상 모델·surrogate·모델 로딩 인터페이스 |
| [defenses/README.md](defenses/README.md) | 방어 패키지 구조·공통 인터페이스 |
| [compositions/README.md](compositions/README.md) | 조합 레시피·생산 단계·검증·재개 |
| [multi_attacks/jps/UPSTREAM.md](multi_attacks/jps/UPSTREAM.md) | JPS 출처·모델 대체·구현 범위 |
| [multi_attacks/mml/UPSTREAM.md](multi_attacks/mml/UPSTREAM.md) | MML replay·generated·공식 평가 프로토콜 구분 |
| [THIRD_PARTY.md](THIRD_PARTY.md) | 외부 코드·데이터·모델·폰트·API의 출처와 조건 |

## Acknowledgement and license

FigStep, CS-DJ, Visual Adversarial Examples, Jailbreaking Attack against MLLM, HADES, Best-of-N, JOOD, JPS, MML, JailGuard, CIDER, AdaShield, AsD, VLMShield 및 벤치마크·모델 프로젝트의 공개 구현과 자산을 참고합니다. 구성 요소별 출처와 revision은 [THIRD_PARTY.md](THIRD_PARTY.md)에 정리되어 있습니다.

ai.resilience가 직접 개발한 코드와 문서는 [MIT License](LICENSE)를 따릅니다. 외부에서 가져오거나 실행 시 사용하는 코드·모델 가중치·데이터셋·폰트·API에는 해당 upstream의 라이선스와 이용 조건이 적용됩니다. 라이선스 원문과 각 구성 요소의 출처 표기를 함께 확인하세요.
