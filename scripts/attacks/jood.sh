#!/usr/bin/env bash
# JOOD: generate attack inputs -> target completions -> evaluation.
# Example: END=1 bash scripts/attacks/jood.sh
set -euo pipefail

METHOD=JOOD
MODEL="${MODEL:-llava_v1_5}"
DATASET="${DATASET:-JOOD}"
ATTACK_CONFIG="${ATTACK_CONFIG:-configs/method_configs/JOOD.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_attack_pipeline "$@"
