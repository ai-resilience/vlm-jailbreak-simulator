# 방법론별 실행 스크립트

각 스크립트는 `simulator_test.sh`처럼 기존 세 Python 실행 파일을 순서대로
호출합니다. 방법론별 파일 상단에서 기본 모델·데이터셋·설정을 확인하고
수정할 수 있습니다. 공통 실행·인자 처리는 `_common.sh`에 모았습니다.

```text
scripts/
├── attacks/         # 공격별 입력 생성 → 응답 생성 → 평가
├── defenses/        # 동일 입력의 baseline·방어 응답 생성 → 비교 평가
├── compositions/    # 조합 입력 생성 → 응답 생성 → 레시피별 평가
└── _common.sh
```

프로젝트 환경을 활성화한 뒤 실행합니다. 저장소 밖에서도 실행할 수 있으며,
상대 경로는 항상 저장소 루트를 기준으로 해석합니다. `PYTHON`으로 실행할
Python 파일을 지정할 수도 있습니다.

## 가장 간단한 실행

```bash
# 실제 실행 전 명령 확인: GPU·모델·API를 호출하지 않음
DRY_RUN=1 END=1 bash scripts/attacks/figstep.sh

# 첫 질문 1개: 입력 생성 → 대상 모델 응답 → HarmBench 평가
END=1 bash scripts/attacks/figstep.sh

# 같은 범위의 FigStep 입력에 JailGuard 적용 및 baseline 비교
END=1 bash scripts/defenses/jailguard.sh

# 모델과 질문 범위 변경
MODEL=qwen3_vl_2b START=10 END=20 RUN_NAME=FigStep_Qwen_pilot \
  bash scripts/attacks/figstep.sh
```

`END`는 끝 인덱스 미포함입니다. **생략하면 전체 데이터셋**을 선택합니다.
질문 수를 줄여도 VisualAdv·ImgJP·DeltaJP·JPS의 최적화 예산은 YAML에 지정된
값을 사용합니다. 짧은 최적화 실험은 별도 YAML을 `ATTACK_CONFIG`로 지정합니다.
스크립트가 논문·기본 설정의 학습 횟수를 자동으로 줄이지 않습니다.

## 공격 스크립트

| 스크립트 | CLI 방법론 | 기본 데이터셋 | 기본 응답 모델 |
| --- | --- | --- | --- |
| [figstep.sh](attacks/figstep.sh) | FigStep | SafeBench | llava_v1_5 |
| [cs_dj.sh](attacks/cs_dj.sh) | CS-DJ | AdvBench | llava_v1_5 |
| [visualadv.sh](attacks/visualadv.sh) | Visual_Adv | AdvBench | llava_v1_5 |
| [imgjp.sh](attacks/imgjp.sh) | ImgJP | SafeBench | llava_v1_5 |
| [deltajp.sh](attacks/deltajp.sh) | DeltaJP | AdvBenchM | llava_v1_5 |
| [jps.sh](attacks/jps.sh) | JPS | SafeBench | qwen2_vl_7b |
| [mml.sh](attacks/mml.sh) | MML | SafeBench | llava_v1_5 |
| [jood.sh](attacks/jood.sh) | JOOD | JOOD 전용 170개 | llava_v1_5 |
| [hades.sh](attacks/hades.sh) | Hades | HADES | llava_v1_5 |
| [best_of_n.sh](attacks/best_of_n.sh) | BestOfN | SafeBench | llava_v1_5 |

`Hades`와 `BestOfN`은 현재 레지스트리의 기존 구현을 호출합니다. 특히 Hades는
기존 PixArt 및 GPT 보조 함수의 모델·API 설정을 필요로 합니다. 이 스크립트
추가는 해당 레거시 알고리즘의 재구현이나 GPU 실행 검증을 의미하지 않습니다.
MML은 기본적으로 공개 이미지 replay와 일반 HarmBench 평가를 사용합니다.
별도 `mml_official` API 프로토콜은 기존 MML 문서의 실행법을 사용합니다.

단계 하나만 실행할 수 있습니다.

```bash
END=1 bash scripts/attacks/figstep.sh generate
bash scripts/attacks/figstep.sh complete
bash scripts/attacks/figstep.sh evaluate
```

설정 변형은 같은 방법론 스크립트에 다른 YAML을 지정합니다.

```bash
ATTACK_CONFIG=configs/method_configs/MML_mirror.yaml RUN_NAME=MML_mirror_SafeBench \
  END=1 bash scripts/attacks/mml.sh

ATTACK_CONFIG=configs/method_configs/Visual_Adv_unconstrained.yaml RUN_NAME=VisualAdv_unconstrained \
  END=1 bash scripts/attacks/visualadv.sh
```

## 방어 스크립트

| 스크립트 | 방어 / 설정 |
| --- | --- |
| [jailguard.sh](defenses/jailguard.sh) | JailGuard |
| [cider.sh](defenses/cider.sh) | CIDER, LLaVA-1.5-7B 필요 |
| [adashield_s.sh](defenses/adashield_s.sh) | AdaShield-S |
| [adashield_a.sh](defenses/adashield_a.sh) | AdaShield-A, FigStep pool |
| [adashield_a_query_related.sh](defenses/adashield_a_query_related.sh) | AdaShield-A, QueryRelated pool |
| [asd.sh](defenses/asd.sh) | AsD |
| [vlmshield.sh](defenses/vlmshield.sh) | VLMShield |

기본 실행은 FigStep·SafeBench 입력을 생성한 뒤, **같은 입력·모델·토큰 예산**으로
baseline과 방어 응답을 만들고 비교 평가합니다. `ATTACK`으로 공격을 바꿀 수 있습니다.
이미 생성한 입력·baseline을 주면 각각 재사용하고 해당 생성 단계는 생략합니다.

```bash
TEST_CASES_PATH=test_cases/JOOD_native/JOOD_test_cases.json \
  ATTACK=JOOD bash scripts/defenses/vlmshield.sh

TEST_CASES_PATH=test_cases/FigStep_SafeBench/FigStep_test_cases.json \
  BASELINE_PATH=completions/FigStep_SafeBench/llava_v1_5/baseline.json \
  bash scripts/defenses/adashield_s.sh
```

지원 단계는 `all`, `generate`(공격 입력), `baseline`, `complete`/`defend`,
`evaluate`입니다. `defend`와 `evaluate`는 기존 입력·결과를 사용합니다.
baseline과 방어 결과는 서로 다른 파일에 저장합니다.

방어별 가중치·입력 수·대상 모델 조건은 기존 방어 모듈에서 검사합니다.
정상 입력 결과가 있으면 아래 환경변수로 평가에 연결할 수 있습니다.

```bash
BENIGN_RESULTS_PATH=completions/benign_defended.json \
BENIGN_BASELINE_RESULTS_PATH=completions/benign_baseline.json \
BENIGN_BEHAVIORS_PATH=data/custom/benign.csv \
  bash scripts/defenses/vlmshield.sh evaluate
```

탐지 F1에는 공격·정상 입력의 명시적인 탐지 판정이 필요합니다. 정상 정답률과
유해성 감소에는 해당 주석도 필요하며, 없는 지표는 기존 평가기에서 unavailable로
보고합니다. 스크립트가 주석이나 탐지 판정을 임의로 만들지 않습니다.

## 조합 스크립트

```bash
# 18개 레시피의 미리보기 54개만 생성
bash scripts/compositions/five_methods.sh generate

# 일반 단계 조합에서 레시피당 최종 입력 1개 생성
PER_RECIPE=1 bash scripts/compositions/stages.sh generate

# 미리보기 입력의 응답 생성·평가
bash scripts/compositions/five_methods.sh complete
bash scripts/compositions/five_methods.sh evaluate

# 미리보기 구조 검증
bash scripts/compositions/five_methods.sh validate
```

기본 모델은 다중 이미지를 지원하는 `qwen3_vl_2b`, 기본 phase는 `preview`입니다.
`all`은 생성·응답·평가까지 실행합니다. `END` 대신 YAML 또는 일반 단계의
`RECIPES`, `PER_RECIPE`로 수량을 설정합니다.

5개 방법 조합의 전체 생성은 기존 미리보기 검토 조건을 유지합니다.

```bash
REVIEW_NOTE='54개 미리보기와 contact sheet 검토 내용' \
  bash scripts/compositions/five_methods.sh review-preview
bash scripts/compositions/five_methods.sh full
COMPOSITION_PHASE=full bash scripts/compositions/five_methods.sh complete
COMPOSITION_PHASE=full bash scripts/compositions/five_methods.sh evaluate
VALIDATE_PHASE=full bash scripts/compositions/five_methods.sh validate
```

`preflight`는 준비 상태를, `dry-run` 단계는 Python 생성기의 조합 계획을 확인합니다.
`DRY_RUN=1`은 Python도 실행하지 않고 shell 명령만 표시하는 옵션입니다.
자세한 조합 정책은 [조합 문서](../compositions/README.md)를 참고하세요.

## 경로와 공통 설정

기본 출력은 다음과 같습니다.

```text
test_cases/<RUN_NAME>/
completions/<RUN_NAME>/<MODEL>/baseline.json
completions/<RUN_NAME>/<MODEL>/<DEFENSE_LABEL>.json
results/<RUN_NAME>/<MODEL>/baseline.json
results/<RUN_NAME>/<MODEL>/<DEFENSE_LABEL>.json
```

기본 `RUN_NAME`은 공격의 `방법론_데이터셋`, 조합은 `composed_five_methods` 또는
`composed_stages`입니다. 조합 입력 JSON은 생성 폴더의 `<phase>/test_cases.json`에
있으며, 조합의 기본 응답·결과 폴더에도 `<phase>/`를 추가해 preview와 full을
구분합니다. JSON의 원본 질문 sidecar가 평가까지 이어지므로, 평가 단계에서
HarmBench 질문 CSV를 모든 데이터셋에 강제로 지정하지 않습니다.

설정·범위가 다른 실험은 `RUN_NAME`을 구분하세요. 기존 CLI의 코드·설정·입력
검증을 그대로 사용하므로, 일치하지 않는 캐시를 강제로 재사용하지 않습니다.

| 변수 | 용도 |
| --- | --- |
| `PYTHON`, `MODEL`, `DATASET` | 실행 환경·응답 모델·데이터셋 |
| `BEHAVIORS_PATH` | DATASET 대신 사용할 사용자 CSV |
| `START`, `END`, `BEHAVIOR_IDS` | 질문 선택 |
| `ATTACK_CONFIG`, `DEFENSE_CONFIG`, `COMPOSITION_CONFIG` | YAML 선택 |
| `RUN_NAME`, `CASES_DIR` | 출력 이름과 입력 생성 폴더 |
| `TEST_CASES_PATH` | 기존 입력 재사용; 입력 생성 생략 |
| `COMPLETIONS_PATH`, `RESULTS_PATH` | 개별 출력 JSON 경로 |
| `COMPLETIONS_DIR`, `RESULTS_DIR` | 응답·평가 출력 폴더 |
| `BASELINE_PATH` | 방어 평가에 재사용할 기존 baseline |
| `MAX_NEW_TOKENS` | 응답 토큰 예산, 기본 256 |
| `CLS_PATH` | 평가 judge, 기본 HarmBench |
| `DRY_RUN=1` | 명령 표시만 수행 |
| `INCREMENTAL=1` | 일치하는 응답 캐시 재사용 |
| `METRICS_ONLY=1` | 기존 라벨로 지표 계산 |
| `INCLUDE_ADVBENCH_METRIC=1` | 기존 AdvBench 보조 지표 추가 |
| `GRADE_RESPONSES=1`, `JUDGE_MODEL` | 선택적인 응답 품질 grading; API 설정 필요 |

재개 시 입력을 다시 만들지 않으려면 `complete` 단계를 사용합니다.
각 스크립트의 `--help`로 공통 사용법을 확인할 수 있습니다.
