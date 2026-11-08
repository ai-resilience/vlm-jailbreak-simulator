#!/usr/bin/env bash
# AdaShield-A: same attack inputs -> baseline + defended completions -> paired evaluation.
# Example: END=1 bash scripts/defenses/adashield_a.sh
set -euo pipefail

DEFENSE=AdaShield
DEFENSE_LABEL=AdaShield-A
MODEL="${MODEL:-llava_v1_5}"
DEFENSE_CONFIG="${DEFENSE_CONFIG:-configs/defense_configs/AdaShield-A.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_defense_pipeline "$@"
