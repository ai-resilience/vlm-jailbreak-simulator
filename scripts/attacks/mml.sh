#!/usr/bin/env bash
# MML: generate attack inputs -> target completions -> evaluation.
# Example: END=1 bash scripts/attacks/mml.sh
set -euo pipefail

METHOD=MML
MODEL="${MODEL:-llava_v1_5}"
DATASET="${DATASET:-SafeBench}"
ATTACK_CONFIG="${ATTACK_CONFIG:-configs/method_configs/MML.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_attack_pipeline "$@"
