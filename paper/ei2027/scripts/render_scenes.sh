#!/usr/bin/env bash
# Render the spectral pbrt-v4 scenes used in the EI 2027 paper (CPU build).
# Usage (from repo root): paper/ei2027/scripts/render_scenes.sh
# Outputs: paper/ei2027/build/*.exr (git-ignored). SPP can be lowered via SPP_CC / SPP_EDGE.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"
PY="${REPO}/venv/bin/python"
PBRT="${REPO}/third_party/pbrt-v4/build/pbrt"
B="paper/ei2027/build"
SPP_CC="${SPP_CC:-1024}"
SPP_EDGE="${SPP_EDGE:-512}"
mkdir -p "${B}/scenes"

# ColorChecker: same geometry/lens as config/pipeline.yaml (iphone_8 recipe, realistic
# wide_22mm.dat at 8.756 mm aperture, focus 4 m), 64 buckets over 360-830 nm.
for ILL in D65 A; do
  "${PY}" tools/build_colorchecker_scene.py --out-dir "${B}/scenes/cc_${ILL}" \
    --illuminant "spectra/illuminant/interpolated/${ILL}.csv" --light-scale 1.0 \
    --xres 960 --yres 640 --pixelsamples "${SPP_CC}" --film spectral --film-output "${B}/cc_${ILL}.exr" \
    --spectral-nbuckets 64 --spectral-lambda-min 360 --spectral-lambda-max 830 \
    --cam-dist 4.0 --camera realistic --lensfile config/lenses/wide_22mm.dat \
    --aperture-diameter-mm 8.756 --focus-distance 4.0
  "${PBRT}" --quiet --seed 1 "${B}/scenes/cc_${ILL}/colorchecker.pbrt"
done

# Slanted edge through the traced double-Gauss 50 mm lens (config/lenses/dgauss.50mm.dat).
# name aperture_mm focus_m
while read -r NAME AP FOCUS; do
  "${PY}" tools/build_image_quality_targets.py --out-dir "${B}/scenes/edge_${NAME}" --target slanted_edge \
    --camera realistic --lensfile config/lenses/dgauss.50mm.dat --aperture-diameter-mm "${AP}" \
    --focus-distance "${FOCUS}" --cam-dist 3.2 --xres 960 --yres 640 --pixelsamples "${SPP_EDGE}" \
    --film spectral --spectral-nbuckets 32 --spectral-lambda-min 400 --spectral-lambda-max 700
  "${PBRT}" --quiet --seed 1 "${B}/scenes/edge_${NAME}/slanted_edge.pbrt"
  mv out/slanted_edge_spectral.exr "${B}/edge_${NAME}.exr"
done <<'LIST'
f2_focused 25.0 3.2
f2_defocus 25.0 2.6
f56_defocus 8.93 2.6
LIST
echo "renders done"
