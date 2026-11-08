#!/usr/bin/env bash
# Generate composed inputs, then use the normal completion/evaluation CLIs.
# Example: bash scripts/compositions/five_methods.sh generate
set -euo pipefail

COMPOSITION_LABEL=composed_five_methods
MODEL="${MODEL:-qwen3_vl_2b}"
COMPOSITION_CONFIG="${COMPOSITION_CONFIG:-configs/composition_configs/five_methods_3000.yaml}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
sim_composition_pipeline "$@"
