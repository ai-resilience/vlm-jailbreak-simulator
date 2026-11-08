#!/usr/bin/env bash
# AdaShield-S: same attack inputs -> baseline + defended completions -> paired evaluation.
# Example: END=1 bash scripts/defenses/adashield_s.sh
set -euo pipefail

DEFENSE=AdaShield
DEFENSE_LABEL=AdaShield-S
MODEL="${MODEL:-llava_v1_5}"
DEFENSE_CONFIG="${DEFENSE_CONFIG:-configs/defense_configs/AdaShield.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_defense_pipeline "$@"
