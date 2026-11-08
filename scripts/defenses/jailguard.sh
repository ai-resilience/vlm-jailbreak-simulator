#!/usr/bin/env bash
# JailGuard: same attack inputs -> baseline + defended completions -> paired evaluation.
# Example: END=1 bash scripts/defenses/jailguard.sh
set -euo pipefail

DEFENSE=JailGuard
DEFENSE_LABEL=JailGuard
MODEL="${MODEL:-llava_v1_5}"
DEFENSE_CONFIG="${DEFENSE_CONFIG:-configs/defense_configs/JailGuard.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_defense_pipeline "$@"
