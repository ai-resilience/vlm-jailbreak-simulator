#!/usr/bin/env bash
# JPS: generate attack inputs -> target completions -> evaluation.
# Example: END=1 bash scripts/attacks/jps.sh
set -euo pipefail

METHOD=JPS
MODEL="${MODEL:-qwen2_vl_7b}"
DATASET="${DATASET:-SafeBench}"
ATTACK_CONFIG="${ATTACK_CONFIG:-configs/method_configs/JPS.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_attack_pipeline "$@"
