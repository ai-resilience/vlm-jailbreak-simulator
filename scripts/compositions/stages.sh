#!/usr/bin/env bash
# Generate composed inputs, then use the normal completion/evaluation CLIs.
# Example: bash scripts/compositions/stages.sh generate
set -euo pipefail

COMPOSITION_LABEL=composed_stages
MODEL="${MODEL:-qwen3_vl_2b}"
COMPOSITION_CONFIG="${COMPOSITION_CONFIG:-configs/composition_configs/stages_ready.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_composition_pipeline "$@"
