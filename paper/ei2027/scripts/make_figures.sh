#!/usr/bin/env bash
# Run the open-cam pipeline stages on the renders from render_scenes.sh and
# regenerate every figure/number used in the paper.
# Usage (from repo root): paper/ei2027/scripts/make_figures.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"
PY="${REPO}/venv/bin/python"
B="paper/ei2027/build"
S="paper/ei2027/scripts"
T=0.12   # exposure_time_override_s from config/pipeline.yaml
LUX=500  # target_illuminance_lux from config/pipeline.yaml

for ILL in D65 A; do
  for R in default iphone_8; do
    "${PY}" tools/pbrt_spectral_exr_to_electrons.py --exr "${B}/cc_${ILL}.exr" \
      --camera-model-config "config/camera_recipes/${R}.yaml" \
      --scene-manifest-json "${B}/scenes/cc_${ILL}/colorchecker_manifest.json" \
      --out "${B}/cc_${ILL}_${R}_e.npz" --target-illuminance-lux "${LUX}" --integration-time-s "${T}"
  done
  # Same flags as tools/run_pipeline.py uses for the noise stage (except preview normalisation).
  "${PY}" tools/apply_emva_noise.py --camera-model-config config/camera_recipes/iphone_8.yaml --seed 0 \
    --linear-exr "${B}/cc_${ILL}.exr" --electrons-npz "${B}/cc_${ILL}_iphone_8_e.npz" \
    --preview-percentile 99.5 --preview-white-balance-enabled true \
    --preview-color-correction-enabled false --integration-time-s "${T}"
  rm -rf "${B}/pipeline_${ILL}" && mkdir -p "${B}/pipeline_${ILL}"
  cp out/colorchecker_noisy_png/* out/colorchecker_noisy.raw16 "${B}/pipeline_${ILL}/"
done

"${PY}" tools/validate_demosaic_linear.py --camera-model-config config/camera_recipes/iphone_8.yaml \
  --json-out "${B}/demosaic_linear_metrics.json" --crop 2 --electrons-npz "${B}/cc_D65_iphone_8_e.npz"
"${PY}" tools/validate_emva_model.py --camera-model-config config/camera_recipes/iphone_8.yaml \
  --json-out "${B}/emva_validation_iphone_8.json"

"${PY}" "${S}/fig_colorchecker.py"
"${PY}" "${S}/fig_ptc.py"
"${PY}" "${S}/fig_mtf.py"
"${PY}" "${S}/fig_cfa.py"
echo "figures written to paper/ei2027/figures"
