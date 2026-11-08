#!/usr/bin/env bash
# FigStep: generate attack inputs -> target completions -> evaluation.
# Example: END=1 bash scripts/attacks/figstep.sh
set -euo pipefail

METHOD=FigStep
MODEL="${MODEL:-llava_v1_5}"
DATASET="${DATASET:-SafeBench}"
ATTACK_CONFIG="${ATTACK_CONFIG:-configs/method_configs/FigStep.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_attack_pipeline "$@"
