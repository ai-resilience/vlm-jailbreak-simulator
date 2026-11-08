#!/usr/bin/env bash
# Shared CLI calls. Method scripts only choose the method/config/default model.
set -euo pipefail

SIMULATOR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$SIMULATOR_ROOT"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
PYTHON="${PYTHON:-python}"

sim_run() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        "$@"
    fi
}

sim_require_file() {
    if [[ "${DRY_RUN:-0}" != 1 && ! -f "$1" ]]; then
        printf 'Missing input: %s\nRun the generate/all stage or set TEST_CASES_PATH.\n' "$1" >&2
        return 1
    fi
}

sim_help() {
    cat <<'HELP'
Usage: bash scripts/<attacks|defenses|compositions>/<method>.sh [STAGE]

Attack stages:       all (default), generate, complete, evaluate
Defense stages:      all (default), generate, baseline, complete/defend, evaluate
Composition stages:  all (default), generate, complete, evaluate,
                     preview, full, review-preview, validate, preflight, dry-run

Common settings (environment variables):
  PYTHON            Python executable from the configured environment
  MODEL             Completion target (per-script default)
  DATASET           Named benchmark (per-method default)
  BEHAVIORS_PATH    Custom CSV instead of DATASET
  START / END       Source question slice; END is exclusive; omit END for all
  ATTACK_CONFIG     Attack YAML override
  RUN_NAME          Output namespace (use a new name for changed settings)
  TEST_CASES_PATH   Reuse existing inputs; skips input generation
  CASES_DIR         Input-generation directory
  COMPLETIONS_PATH  Output completions JSON
  RESULTS_PATH      Output evaluation JSON
  MAX_NEW_TOKENS    Completion token budget (default: 256)
  CLS_PATH          Binary judge (default: cais/HarmBench-Llama-2-13b-cls)
  DRY_RUN=1         Print commands without executing them
  INCREMENTAL=1     Reuse matching completion records
  METRICS_ONLY=1    Evaluate existing labels without loading the binary judge

Defense settings:
  ATTACK            Source attack (default: FigStep)
  DEFENSE_CONFIG    Defense YAML override
  BASELINE_PATH     Reuse an existing same-input/model baseline
  BENIGN_RESULTS_PATH / BENIGN_BASELINE_RESULTS_PATH / BENIGN_BEHAVIORS_PATH
                    Optional benign comparison inputs for defense metrics

Composition settings:
  COMPOSITION_CONFIG / COMPOSITION_PHASE (default: preview)
  RECIPES / PER_RECIPE       General-stage recipe selection and pilot count
  REVIEW_NOTE               Observations for the five-method preview review
  VALIDATE_PHASE            Phase to validate (default: COMPOSITION_PHASE)

All relative paths are resolved from the repository root.
Examples:
  END=1 bash scripts/attacks/figstep.sh
  DRY_RUN=1 bash scripts/defenses/jailguard.sh
  TEST_CASES_PATH=test_cases/FigStep_SafeBench/FigStep_test_cases.json \
    bash scripts/defenses/vlmshield.sh
HELP
}

sim_stage() {
    if (( $# > 1 )); then
        printf 'Expected at most one stage argument. Use --help.\n' >&2
        return 2
    fi
    SIM_STAGE="${1:-${STAGE:-all}}"
    if [[ "$SIM_STAGE" == --help || "$SIM_STAGE" == -h || "$SIM_STAGE" == help ]]; then
        sim_help
        exit 0
    fi
    for sim_number in START END MAX_NEW_TOKENS PER_RECIPE; do
        if [[ -n "${!sim_number:-}" && ! "${!sim_number}" =~ ^[0-9]+$ ]]; then
            printf '%s must be a nonnegative integer.\n' "$sim_number" >&2
            return 2
        fi
    done
}

sim_paths() {
    case "$METHOD" in
        FigStep|ImgJP|MML|BestOfN|JPS) SIM_DEFAULT_DATASET=SafeBench ;;
        CS-DJ|Visual_Adv) SIM_DEFAULT_DATASET=AdvBench ;;
        DeltaJP) SIM_DEFAULT_DATASET=AdvBenchM ;;
        Hades) SIM_DEFAULT_DATASET=HADES ;;
        JOOD) SIM_DEFAULT_DATASET=JOOD ;;
        Composed) SIM_DEFAULT_DATASET=composed ;;
        *) printf 'Unknown attack: %s\n' "$METHOD" >&2; return 2 ;;
    esac
    DATASET="${DATASET:-$SIM_DEFAULT_DATASET}"
    MODEL="${MODEL:-llava_v1_5}"
    SIM_DATA_TAG="$DATASET"
    [[ -z "${BEHAVIORS_PATH:-}" ]] || SIM_DATA_TAG=custom
    RUN_NAME="${RUN_NAME:-${METHOD}_${SIM_DATA_TAG}}"
    CASES_DIR="${CASES_DIR:-test_cases/$RUN_NAME}"
    SIM_MODEL_TAG="${MODEL//\//__}"
    COMPLETIONS_DIR="${COMPLETIONS_DIR:-completions/$RUN_NAME/$SIM_MODEL_TAG}"
    RESULTS_DIR="${RESULTS_DIR:-results/$RUN_NAME/$SIM_MODEL_TAG}"
    SIM_EXPLICIT_CASES="${TEST_CASES_PATH:+1}"
    SIM_EXPLICIT_BASELINE="${BASELINE_PATH:+1}"
    case "$METHOD" in
        Visual_Adv) SIM_CASE_FILENAME=Visual_Adv.json ;;
        Hades) SIM_CASE_FILENAME=hades_test_cases.json ;;
        *) SIM_CASE_FILENAME="${METHOD}_test_cases.json" ;;
    esac
    TEST_CASES_PATH="${TEST_CASES_PATH:-$CASES_DIR/$SIM_CASE_FILENAME}"
    BASELINE_PATH="${BASELINE_PATH:-$COMPLETIONS_DIR/baseline.json}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
    CLS_PATH="${CLS_PATH:-cais/HarmBench-Llama-2-13b-cls}"
}

sim_generate_attack() {
    if [[ "$SIM_EXPLICIT_CASES" == 1 ]]; then
        sim_require_file "$TEST_CASES_PATH"
        printf 'Using existing test cases: %s\n' "$TEST_CASES_PATH"
        return
    fi
    local -a args=(generate_test_cases.py --method_name "$METHOD" --save_dir "$CASES_DIR")
    if [[ -n "${BEHAVIORS_PATH:-}" ]]; then
        args+=(--behaviors_path "$BEHAVIORS_PATH")
    else
        args+=(--dataset_name "$DATASET")
    fi
    args+=(--behavior_start_idx "${START:-0}")
    [[ -z "${END:-}" ]] || args+=(--behavior_end_idx "$END")
    [[ -z "${ATTACK_CONFIG:-}" ]] || args+=(--config "$ATTACK_CONFIG")
    [[ -z "${BEHAVIOR_IDS:-}" ]] || args+=(--behavior_ids_subset "$BEHAVIOR_IDS")
    sim_run "$PYTHON" "${args[@]}"
}

sim_complete() {
    local destination="$1"
    shift
    sim_require_file "$TEST_CASES_PATH"
    local -a args=(generate_completions.py --model_name "$MODEL"
        --test_cases_path "$TEST_CASES_PATH" --save_path "$destination"
        --max_new_tokens "$MAX_NEW_TOKENS" "$@")
    [[ "${INCREMENTAL:-0}" != 1 ]] || args+=(--incremental_update)
    sim_run "$PYTHON" "${args[@]}"
}

sim_evaluate() {
    sim_require_file "$COMPLETIONS_PATH"
    local -a args=(evaluate_completions.py --method_name "$METHOD" --model_name "$MODEL"
        --cls_path "$CLS_PATH" --completions_path "$COMPLETIONS_PATH"
        --save_path "$RESULTS_PATH" --asr_results_file "${ASR_RESULTS_FILE:-$RESULTS_DIR/asr_${SIM_RESULT_TAG}.txt}" "$@")
    [[ "${METRICS_ONLY:-0}" != 1 ]] || args+=(--metrics_only)
    [[ "${INCLUDE_ADVBENCH_METRIC:-0}" != 1 ]] || args+=(--include_advbench_metric)
    [[ "${GRADE_RESPONSES:-0}" != 1 ]] || args+=(--grade_responses --judge_model "${JUDGE_MODEL:-gpt-4o-mini}")
    [[ -z "${BENIGN_RESULTS_PATH:-}" ]] || args+=(--benign_results_path "$BENIGN_RESULTS_PATH")
    [[ -z "${BENIGN_BASELINE_RESULTS_PATH:-}" ]] || args+=(--benign_baseline_results_path "$BENIGN_BASELINE_RESULTS_PATH")
    [[ -z "${BENIGN_BEHAVIORS_PATH:-}" ]] || args+=(--benign_behaviors_path "$BENIGN_BEHAVIORS_PATH")
    [[ "${BENIGN_LABEL_MEANS_UTILITY:-0}" != 1 ]] || args+=(--benign_label_means_utility)
    sim_run "$PYTHON" "${args[@]}"
}

sim_attack_pipeline() {
    sim_stage "$@"
    case "$SIM_STAGE" in all|generate|complete|evaluate) ;; *) printf 'Unknown attack stage: %s\n' "$SIM_STAGE" >&2; return 2 ;; esac
    sim_paths
    COMPLETIONS_PATH="${COMPLETIONS_PATH:-$BASELINE_PATH}"
    RESULTS_PATH="${RESULTS_PATH:-$RESULTS_DIR/baseline.json}"
    SIM_RESULT_TAG=baseline
    case "$SIM_STAGE" in
        all) sim_generate_attack; sim_complete "$COMPLETIONS_PATH"; sim_evaluate ;;
        generate) sim_generate_attack ;;
        complete) sim_complete "$COMPLETIONS_PATH" ;;
        evaluate) sim_evaluate ;;
    esac
}

sim_defense_pipeline() {
    sim_stage "$@"
    case "$SIM_STAGE" in all|generate|baseline|complete|defend|evaluate) ;; *) printf 'Unknown defense stage: %s\n' "$SIM_STAGE" >&2; return 2 ;; esac
    METHOD="${ATTACK:-FigStep}"
    sim_paths
    COMPLETIONS_PATH="${COMPLETIONS_PATH:-$COMPLETIONS_DIR/$DEFENSE_LABEL.json}"
    RESULTS_PATH="${RESULTS_PATH:-$RESULTS_DIR/$DEFENSE_LABEL.json}"
    SIM_RESULT_TAG="$DEFENSE_LABEL"
    local -a defense_args=(--defense_name "$DEFENSE" --defense_config "$DEFENSE_CONFIG")
    case "$SIM_STAGE" in all|generate) sim_generate_attack ;; esac
    case "$SIM_STAGE" in
        all|baseline)
            sim_require_file "$TEST_CASES_PATH"
            if [[ "$SIM_EXPLICIT_BASELINE" == 1 ]]; then
                sim_require_file "$BASELINE_PATH"
                printf 'Using existing baseline: %s\n' "$BASELINE_PATH"
            else
                sim_complete "$BASELINE_PATH"
            fi
            ;;
    esac
    case "$SIM_STAGE" in all|complete|defend) sim_complete "$COMPLETIONS_PATH" "${defense_args[@]}" ;; esac
    case "$SIM_STAGE" in
        all|evaluate)
            sim_require_file "$BASELINE_PATH"
            sim_evaluate --baseline_results_path "$BASELINE_PATH"
            ;;
    esac
}

sim_composition_pipeline() {
    sim_stage "$@"
    case "$SIM_STAGE" in
        all|generate|complete|evaluate|preview|full|review-preview|validate|preflight|dry-run) ;;
        *) printf 'Unknown composition stage: %s\n' "$SIM_STAGE" >&2; return 2 ;;
    esac
    METHOD=Composed
    COMPOSITION_PHASE="${COMPOSITION_PHASE:-preview}"
    case "$SIM_STAGE" in preview|full) COMPOSITION_PHASE="$SIM_STAGE" ;; esac
    RUN_NAME="${RUN_NAME:-$COMPOSITION_LABEL}"
    local custom_completions_dir="${COMPLETIONS_DIR:+1}"
    local custom_results_dir="${RESULTS_DIR:+1}"
    sim_paths
    # Keep preview and full responses/results in different default directories.
    [[ "$custom_completions_dir" == 1 ]] || COMPLETIONS_DIR="$COMPLETIONS_DIR/$COMPOSITION_PHASE"
    [[ "$custom_results_dir" == 1 ]] || RESULTS_DIR="$RESULTS_DIR/$COMPOSITION_PHASE"
    [[ "$SIM_EXPLICIT_BASELINE" == 1 ]] || BASELINE_PATH="$COMPLETIONS_DIR/baseline.json"
    if [[ "$SIM_EXPLICIT_CASES" != 1 ]]; then
        TEST_CASES_PATH="$CASES_DIR/$COMPOSITION_PHASE/test_cases.json"
    fi
    COMPLETIONS_PATH="${COMPLETIONS_PATH:-$BASELINE_PATH}"
    RESULTS_PATH="${RESULTS_PATH:-$RESULTS_DIR/baseline.json}"
    SIM_RESULT_TAG=baseline
    if [[ "$SIM_EXPLICIT_CASES" == 1 && ( "$SIM_STAGE" == all || "$SIM_STAGE" == generate ) ]]; then
        sim_require_file "$TEST_CASES_PATH"
        printf 'Using existing test cases: %s\n' "$TEST_CASES_PATH"
    elif [[ "$SIM_STAGE" != complete && "$SIM_STAGE" != evaluate ]]; then
        local phase="$COMPOSITION_PHASE"
        case "$SIM_STAGE" in review-preview|validate|preflight|dry-run) phase="$SIM_STAGE" ;; esac
        local -a args=(generate_test_cases.py --composition_config "$COMPOSITION_CONFIG"
            --composition_phase "$phase" --save_dir "$CASES_DIR")
        [[ -z "${RECIPES:-}" ]] || args+=(--composition_recipes "$RECIPES")
        [[ -z "${PER_RECIPE:-}" ]] || args+=(--composition_per_recipe "$PER_RECIPE")
        [[ -z "${REVIEW_NOTE:-}" ]] || args+=(--composition_review_note "$REVIEW_NOTE")
        [[ "$phase" != validate ]] || args+=(--composition_validate_phase "${VALIDATE_PHASE:-$COMPOSITION_PHASE}")
        sim_run "$PYTHON" "${args[@]}"
    fi
    case "$SIM_STAGE" in all|complete) sim_complete "$COMPLETIONS_PATH" ;; esac
    case "$SIM_STAGE" in all|evaluate) sim_evaluate ;; esac
}
