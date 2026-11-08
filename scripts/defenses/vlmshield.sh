#!/usr/bin/env bash
# VLMShield: same attack inputs -> baseline + defended completions -> paired evaluation.
# Example: END=1 bash scripts/defenses/vlmshield.sh
set -euo pipefail

DEFENSE=VLMShield
DEFENSE_LABEL=VLMShield
MODEL="${MODEL:-llava_v1_5}"
DEFENSE_CONFIG="${DEFENSE_CONFIG:-configs/defense_configs/VLMShield.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_defense_pipeline "$@"
