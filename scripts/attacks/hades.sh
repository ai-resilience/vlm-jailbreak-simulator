#!/usr/bin/env bash
# Hades: generate attack inputs -> target completions -> evaluation.
# Example: END=1 bash scripts/attacks/hades.sh
set -euo pipefail

METHOD=Hades
MODEL="${MODEL:-llava_v1_5}"
DATASET="${DATASET:-HADES}"
ATTACK_CONFIG="${ATTACK_CONFIG:-configs/method_configs/Hades.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_attack_pipeline "$@"
