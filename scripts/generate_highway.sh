#!/usr/bin/env bash
set -euo pipefail

# End-to-end highway scene run:
#   1) build the scene (tools/build_highway_scene.py)
#   2) render with pbrt (GPU by default; PBRT_GPU=0 for CPU)
#   3) spectral EXR -> electrons (mean-photopic-lux autocalibration)
#   4) EMVA noise + Bayer + demosaic previews
#
# Usage:
#   scripts/generate_highway.sh [name] [extra build_highway_scene.py args...]
#
# Environment overrides:
#   CAMERA_MODEL=config/camera_models/research_highway_wide22.yaml
#   TARGET_LUX=125          # mean SENSOR-PLANE photopic lux. With sun-dominant
#                           # illumination (5:1 direct:diffuse) 125 puts the road
#                           # at ~30% of the 4800 e- full well with a mostly
#                           # unclipped blue sky
#   EXPOSURE_S=0.005        # integration time
#   PBRT_GPU=1              # 0 = CPU render
#   GLARE=0                 # 1 = apply the lens stray-light stage (veiling,
#                           # halo, ghost, starburst) to the EXR before the
#                           # sensor; pair with --visible-sun-disk for low-sun
#                           # glare scenarios
#
# Examples:
#   scripts/generate_highway.sh demo --preset smoke
#   scripts/generate_highway.sh rush --num-cars 8 --sun-elevation 25 --sun-azimuth 210
#   TARGET_LUX=3000 scripts/generate_highway.sh dusk --sun-elevation 8 --turbidity 5

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

NAME="${1:-default}"
shift || true

PY="${REPO_DIR}/venv/bin/python"
PBRT="${REPO_DIR}/third_party/pbrt-v4/build/pbrt"
CAMERA_MODEL="${CAMERA_MODEL:-config/camera_models/research_highway_wide22.yaml}"
TARGET_LUX="${TARGET_LUX:-125}"
EXPOSURE_S="${EXPOSURE_S:-0.005}"
PBRT_GPU="${PBRT_GPU:-1}"

[[ -x "${PY}" ]] || { echo "Missing venv python at ${PY}"; exit 2; }
[[ -x "${PBRT}" ]] || { echo "Missing pbrt at ${PBRT} (see docs/guides/building_pbrt.md)"; exit 2; }

echo "== 1/4 build scene '${NAME}'"
"${PY}" tools/build_highway_scene.py --name "${NAME}" "$@"

SCENE="scenes/generated/highway/${NAME}/highway_${NAME}.pbrt"
MANIFEST="scenes/generated/highway/${NAME}/highway_${NAME}_manifest.json"
EXR="$("${PY}" -c "import json;print(json.load(open('${MANIFEST}'))['film']['filename'])")"

echo "== 2/4 render (${SCENE} -> ${EXR})"
if [[ "${PBRT_GPU}" == "1" ]]; then
  "${PBRT}" --gpu "${SCENE}"
else
  "${PBRT}" "${SCENE}"
fi

GLARE="${GLARE:-0}"
if [[ "${GLARE}" == "1" ]]; then
  GLARE_EXR="${EXR%.exr}_glare.exr"
  echo "== 2b/4 lens stray light (veiling/halo/ghost/starburst) -> ${GLARE_EXR}"
  "${PY}" tools/apply_spectral_psf.py \
    --repo-root "${REPO_DIR}" \
    --camera-model-config "${CAMERA_MODEL}" \
    --exr-in "${EXR}" \
    --exr-out "${GLARE_EXR}"
  EXR="${GLARE_EXR}"
fi

NPZ="out/highway_${NAME}_electrons.npz"
echo "== 3/4 electrons (${CAMERA_MODEL}, target ${TARGET_LUX} lux, ${EXPOSURE_S}s)"
"${PY}" tools/pbrt_spectral_exr_to_electrons.py \
  --repo-root "${REPO_DIR}" \
  --exr "${EXR}" \
  --camera-model-config "${CAMERA_MODEL}" \
  --scene-manifest-json "${MANIFEST}" \
  --target-illuminance-lux "${TARGET_LUX}" \
  --integration-time-s "${EXPOSURE_S}" \
  --out "${NPZ}"

echo "== 4/4 EMVA noise + previews"
"${PY}" tools/apply_emva_noise.py \
  --repo-root "${REPO_DIR}" \
  --camera-model-config "${CAMERA_MODEL}" \
  --seed 0 \
  --linear-exr "${EXR}" \
  --electrons-npz "${NPZ}" \
  --integration-time-s "${EXPOSURE_S}" \
  --preview-percentile 99.5 \
  --preview-white-balance-enabled true \
  --preview-color-correction-enabled false

echo "done: previews in out/colorchecker_noisy_png/, electrons in ${NPZ}"
