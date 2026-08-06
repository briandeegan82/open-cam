#!/usr/bin/env bash
set -euo pipefail

# ColorChecker + D65 artifact-injection dataset.
#
# Thin wrapper over tools-free driver scripts/generate_cc_artifacts.py, which renders
# the ColorChecker once under D65 (iphone_8_rggb) and injects dead pixels, uneven
# illumination and row/column FPN as independent strength ladders (plus a 'combined'
# ladder that stacks all three).  See the driver's --help for the full artifact list.
#
# Usage:
#   scripts/generate_cc_artifacts.sh [artifacts...]
#
#   artifacts: subset of {dead_pixels uneven_illum row_col_noise combined}
#              (default: all four; applies to the chart mode only)
#
# Optional environment overrides:
#   CC_ART_OUT_DIR=out/cc_artifacts      output dataset root
#   CC_ART_SPP=4096                      pixel samples per render
#   CC_ART_SEED=0                        base noise seed
#   CC_ART_GPU=true                      render on GPU (OptiX); false = CPU
#   CC_ART_MODE=both                     chart | patches | both
#                                          chart:   D65 full-chart artifact ladders
#                                          patches: 20 lux/8x per-patch uniformity mode
#                                                   (row noise a factor; RAW mono domain)
#   CC_ART_MAX_PATCHES=                   limit per-patch mode to first N patches

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PY="${REPO_DIR}/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Missing python venv at ${PY}"
  exit 2
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '3,24p' "${BASH_SOURCE[0]}"
  echo
  exec "${PY}" "${REPO_DIR}/scripts/generate_cc_artifacts.py" --help
fi

OUT_DIR="${CC_ART_OUT_DIR:-out/cc_artifacts}"
SPP="${CC_ART_SPP:-4096}"
SEED="${CC_ART_SEED:-0}"
GPU="${CC_ART_GPU:-true}"
MODE="${CC_ART_MODE:-both}"

CMD=("${PY}" "${REPO_DIR}/scripts/generate_cc_artifacts.py"
     --out-dir "${REPO_DIR}/${OUT_DIR}"
     --spp "${SPP}"
     --seed "${SEED}"
     --gpu "${GPU}"
     --mode "${MODE}")
if [[ -n "${CC_ART_MAX_PATCHES:-}" ]]; then
  CMD+=(--max-patches "${CC_ART_MAX_PATCHES}")
fi
if [[ "$#" -gt 0 ]]; then
  CMD+=(--artifacts "$@")
fi

echo "== ColorChecker artifact dataset =="
echo "   out=${OUT_DIR} spp=${SPP} seed=${SEED} gpu=${GPU} mode=${MODE} artifacts=${*:-<all>}"
exec "${CMD[@]}"
