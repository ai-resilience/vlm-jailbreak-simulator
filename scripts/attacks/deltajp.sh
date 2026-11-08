#!/usr/bin/env bash
# DeltaJP: generate attack inputs -> target completions -> evaluation.
# Example: END=1 bash scripts/attacks/deltajp.sh
set -euo pipefail

METHOD=DeltaJP
MODEL="${MODEL:-llava_v1_5}"
DATASET="${DATASET:-AdvBenchM}"
ATTACK_CONFIG="${ATTACK_CONFIG:-configs/method_configs/DeltaJP.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_attack_pipeline "$@"
