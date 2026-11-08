#!/usr/bin/env bash
# BestOfN: generate attack inputs -> target completions -> evaluation.
# Example: END=1 bash scripts/attacks/best_of_n.sh
set -euo pipefail

METHOD=BestOfN
MODEL="${MODEL:-llava_v1_5}"
DATASET="${DATASET:-SafeBench}"
ATTACK_CONFIG="${ATTACK_CONFIG:-configs/method_configs/BestOfN.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_attack_pipeline "$@"
