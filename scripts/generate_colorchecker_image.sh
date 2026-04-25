#!/usr/bin/env bash
set -euo pipefail

# Full end-to-end run (defaults from config/pipeline.yaml — multispectral PBRT by default):
# 1) build ColorChecker scene
# 2) render with pbrt
# 3) validate render (if validate.enabled)
# 4) optional EMVA parameter validation (if validate_emva.enabled)
# 5) spectral -> electrons forward model (if sensor_forward.enabled)
# 6) EMVA noise + Bayer + demosaic previews (if noise.enabled)
# 7) demosaic linear validation (if validate_demosaic.enabled and electrons NPZ exists)
#
# Usage:
#   scripts/generate_colorchecker_image.sh [seed]
#
# Configuration:
#   PIPELINE_CONFIG=config/pipeline.yaml   # default; edit YAML for film, resolution, paths, seeds,
#                                            render.camera (perspective|realistic), lens / focus, etc.
#   EMVA_FROM_EXR=1                        # drive EMVA from EXR only (omit --electrons-npz)
#
# Example:
#   scripts/generate_colorchecker_image.sh 0

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PIPELINE_CONFIG="${PIPELINE_CONFIG:-config/pipeline.yaml}"
EMVA_FROM_EXR="${EMVA_FROM_EXR:-0}"
PY="${REPO_DIR}/venv/bin/python"

if [[ ! -x "${PY}" ]]; then
  echo "Missing python venv at ${PY}"
  exit 2
fi

while IFS= read -r -d '' kv; do
  export "${kv}"
done < <("${PY}" "${REPO_DIR}/tools/pipeline_shell_env.py" "${REPO_DIR}" "${PIPELINE_CONFIG}" --format env0)

PBRT="${REPO_DIR}/${PBRT_REL}"
SEED_USER="${1-}"
if [[ -n "${SEED_USER}" ]]; then
  SEED="${SEED_USER}"
else
  SEED="${DEFAULT_NOISE_SEED}"
fi

if [[ ! -x "${PBRT}" ]]; then
  echo "Missing pbrt binary at ${PBRT}"
  exit 2
fi

echo "== 1/7 Build scene (config: ${PIPELINE_CONFIG_RESOLVED}, film: ${FILM}) =="
echo "   render illuminant: ${RENDER_ILLUMINANT_REL}"
BUILD_ARGS=(
  "${REPO_DIR}/${SCENE_BUILDER_REL}"
  --repo-root "${REPO_DIR}"
  --illuminant "${RENDER_ILLUMINANT_REL}"
  --light-scale "${LIGHT_SCALE}"
  --xres "${XRES}"
  --yres "${YRES}"
  --pixelsamples "${PIXELSAMPLES}"
  --film "${FILM}"
)
if [[ "${FILM}" == spectral ]]; then
  BUILD_ARGS+=(
    --spectral-nbuckets "${SPECTRAL_NBUCKETS}"
    --spectral-lambda-min "${SPECTRAL_LAMBDA_MIN}"
    --spectral-lambda-max "${SPECTRAL_LAMBDA_MAX}"
  )
fi
if [[ -n "${FILM_OUTPUT_REL}" ]]; then
  BUILD_ARGS+=(--film-output "${FILM_OUTPUT_REL}")
fi
# Match tools/run_pipeline.py: camera / lens flags come from render.* in PIPELINE_CONFIG.
BUILD_ARGS+=(--cam-dist "${CAM_DIST}" --camera "${CAMERA}")
if [[ "${CAMERA}" == realistic ]]; then
  BUILD_ARGS+=(
    --lensfile "${REALISTIC_LENSFILE_REL}"
    --aperture-diameter-mm "${REALISTIC_APERTURE_MM}"
  )
  if [[ -n "${REALISTIC_FOCUS_DISTANCE:-}" ]]; then
    BUILD_ARGS+=(--focus-distance "${REALISTIC_FOCUS_DISTANCE}")
  fi
fi
mapfile -t EXTRA_BUILD_ARGS < <(
  "${PY}" - <<'PY' "${BUILDER_EXTRA_ARGS}"
import json
import sys
for tok in json.loads(sys.argv[1]):
    print(tok)
PY
)
if [[ "${#EXTRA_BUILD_ARGS[@]}" -gt 0 ]]; then
  BUILD_ARGS+=( "${EXTRA_BUILD_ARGS[@]}" )
fi
"${PY}" "${BUILD_ARGS[@]}"

echo "== 2/7 Render with pbrt =="
"${PBRT}" "${REPO_DIR}/${SCENE_FILE_REL}"

if [[ "${POST_PSF_ENABLED}" == 1 ]]; then
  echo "== 2b/7 Post PSF on EXR (camera model lens.post_psf.enabled) =="
  "${PY}" "${REPO_DIR}/${PSF_TOOL_REL}" \
    --repo-root "${REPO_DIR}" \
    --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}" \
    --exr-in "${REPO_DIR}/${EXR_OUT_REL}"
fi

if [[ "${VALIDATE_RENDER}" == 1 ]]; then
  echo "== 3/7 Validate render =="
  "${PY}" "${REPO_DIR}/${VALIDATE_TOOL_REL}" \
    --repo-root "${REPO_DIR}" \
    --exr "${REPO_DIR}/${EXR_OUT_REL}"
else
  echo "== 3/7 Validate render (skipped: validate.enabled false) =="
fi

if [[ "${VALIDATE_EMVA}" == 1 ]]; then
  echo "== 4/7 Validate EMVA model parameters =="
  "${PY}" "${REPO_DIR}/${VALIDATE_EMVA_TOOL_REL}" \
    --repo-root "${REPO_DIR}" \
    --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}" \
    --json-out "${REPO_DIR}/${EMVA_VALIDATION_REPORT_REL}"
else
  echo "== 4/7 Validate EMVA model parameters (skipped: validate_emva.enabled false) =="
fi

if [[ "${SENSOR_FORWARD_ENABLED}" == 1 ]]; then
  if [[ "${SENSOR_FORWARD_MODE}" == pbrt_exr ]]; then
    echo "== 5/7 Electrons from rendered spectral EXR (PBRT → NPZ) =="
    SF_CMD=("${PY}" "${REPO_DIR}/${PBRT_EXR_TO_ELECTRONS_TOOL_REL}" \
      --repo-root "${REPO_DIR}" \
      --exr "${REPO_DIR}/${EXR_OUT_REL}" \
      --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}" \
      --out "${REPO_DIR}/${SENSOR_FORWARD_NPZ_REL}")
    if [[ -n "${SENSOR_FORWARD_TARGET_LUX}" ]]; then
      SF_CMD+=(--target-illuminance-lux "${SENSOR_FORWARD_TARGET_LUX}")
    fi
    if [[ -n "${EXPOSURE_TIME_OVERRIDE_S:-}" ]]; then
      SF_CMD+=(--integration-time-s "${EXPOSURE_TIME_OVERRIDE_S}")
    fi
    "${SF_CMD[@]}"
  else
    echo "== 5/7 Spectral forward (analytic chart → electrons NPZ) =="
    SF_CMD=("${PY}" "${REPO_DIR}/${SENSOR_FORWARD_TOOL_REL}" \
      --repo-root "${REPO_DIR}" \
      --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}")
    if [[ -n "${SENSOR_FORWARD_TARGET_LUX}" ]]; then
      SF_CMD+=(--target-illuminance-lux "${SENSOR_FORWARD_TARGET_LUX}")
    fi
    if [[ -n "${EXPOSURE_TIME_OVERRIDE_S:-}" ]]; then
      SF_CMD+=(--integration-time-s "${EXPOSURE_TIME_OVERRIDE_S}")
    fi
    "${SF_CMD[@]}"
  fi
else
  echo "== 5/7 Spectral forward (skipped: sensor_forward.enabled false) =="
fi

if [[ "${NOISE_ENABLED}" == 1 ]]; then
  echo "== 6/7 EMVA + Bayer + demosaic previews =="
  NOISE_CMD=(
    "${REPO_DIR}/${NOISE_TOOL_REL}"
    --repo-root "${REPO_DIR}"
    --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}"
    --seed "${SEED}"
    --linear-exr "${REPO_DIR}/${EXR_OUT_REL}"
    --preview-percentile "${NOISE_PREVIEW_PERCENTILE}"
  )
  if [[ -n "${NOISE_EXPOSURE_SCALE}" ]]; then
    NOISE_CMD+=(--exposure-scale "${NOISE_EXPOSURE_SCALE}")
  fi
  if [[ "${NOISE_PREVIEW_NO_NORMALIZE:-0}" == 1 ]]; then
    NOISE_CMD+=(--preview-no-normalize)
  fi
  if [[ -n "${NOISE_PREVIEW_WB_ENABLED:-}" ]]; then
    NOISE_CMD+=(--preview-white-balance-enabled "${NOISE_PREVIEW_WB_ENABLED}")
  fi
  if [[ -n "${NOISE_PREVIEW_CC_ENABLED:-}" ]]; then
    NOISE_CMD+=(--preview-color-correction-enabled "${NOISE_PREVIEW_CC_ENABLED}")
  fi
  if [[ -n "${EXPOSURE_TIME_OVERRIDE_S:-}" ]]; then
    NOISE_CMD+=(--integration-time-s "${EXPOSURE_TIME_OVERRIDE_S}")
  fi
  USE_NPZ=0
  if [[ "${EMVA_FROM_EXR}" == 1 ]]; then
    USE_NPZ=0
  elif [[ "${SENSOR_FORWARD_ENABLED}" == 1 ]]; then
    USE_NPZ=1
  else
    USE_NPZ=0
  fi
  if [[ "${USE_NPZ}" == 1 ]]; then
    NOISE_CMD+=(--electrons-npz "${REPO_DIR}/${SENSOR_FORWARD_NPZ_REL}")
  fi
  "${PY}" "${NOISE_CMD[@]}"
else
  echo "== 6/7 EMVA (skipped: noise.enabled false) =="
fi

if [[ "${VALIDATE_DEMOSAIC}" == 1 ]]; then
  if [[ ! -f "${REPO_DIR}/${SENSOR_FORWARD_NPZ_REL}" ]]; then
    echo "== 7/7 Demosaic linear validation (skipped: missing ${SENSOR_FORWARD_NPZ_REL}) =="
  else
    echo "== 7/7 Demosaic linear validation =="
    "${PY}" "${REPO_DIR}/${VALIDATE_DEMOSAIC_TOOL_REL}" \
      --repo-root "${REPO_DIR}" \
      --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}" \
      --electrons-npz "${REPO_DIR}/${SENSOR_FORWARD_NPZ_REL}" \
      --crop "${DEMOSAIC_CROP}" \
      --json-out "${REPO_DIR}/${DEMOSAIC_METRICS_JSON_REL}"
  fi
else
  echo "== 7/7 Demosaic linear validation (skipped: validate_demosaic.enabled false) =="
fi

echo
echo "Done. Pipeline config: ${PIPELINE_CONFIG_RESOLVED}"
echo "Final image outputs (when EMVA ran):"
echo "  out/colorchecker_noisy_png/noisy_demosaic_rgb8.png"
echo "  out/colorchecker_noisy_png/clean_demosaic_rgb8.png"
echo "Other artifacts:"
echo "  ${EXR_OUT_REL}"
echo "  out/colorchecker_noisy.raw16"
echo "  out/colorchecker_noisy_png/run_stats.json"
echo "  ${DEMOSAIC_METRICS_JSON_REL}"
