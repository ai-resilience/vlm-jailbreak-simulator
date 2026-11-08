#!/usr/bin/env bash
# ImgJP: generate attack inputs -> target completions -> evaluation.
# Example: END=1 bash scripts/attacks/imgjp.sh
set -euo pipefail

METHOD=ImgJP
MODEL="${MODEL:-llava_v1_5}"
DATASET="${DATASET:-SafeBench}"
ATTACK_CONFIG="${ATTACK_CONFIG:-configs/method_configs/ImgJP.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_attack_pipeline "$@"
