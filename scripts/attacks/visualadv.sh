#!/usr/bin/env bash
# Visual_Adv: generate attack inputs -> target completions -> evaluation.
# Example: END=1 bash scripts/attacks/visualadv.sh
set -euo pipefail

METHOD=Visual_Adv
MODEL="${MODEL:-llava_v1_5}"
DATASET="${DATASET:-AdvBench}"
ATTACK_CONFIG="${ATTACK_CONFIG:-configs/method_configs/Visual_Adv.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_attack_pipeline "$@"
