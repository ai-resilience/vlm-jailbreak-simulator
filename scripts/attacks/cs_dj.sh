#!/usr/bin/env bash
# CS-DJ: generate attack inputs -> target completions -> evaluation.
# Example: END=1 bash scripts/attacks/cs_dj.sh
set -euo pipefail

METHOD=CS-DJ
MODEL="${MODEL:-llava_v1_5}"
DATASET="${DATASET:-AdvBench}"
ATTACK_CONFIG="${ATTACK_CONFIG:-configs/method_configs/CS-DJ.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_attack_pipeline "$@"
