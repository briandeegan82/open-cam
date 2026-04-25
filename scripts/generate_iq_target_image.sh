#!/usr/bin/env bash
set -euo pipefail

# Full end-to-end run for one IQ target:
# 1) build target scene
# 2) render with pbrt
# 3) optional post-PSF
# 4) optional spectral -> electrons forward model
# 5) optional EMVA noise + preview generation
# 6) optional demosaic linear validation
#
# Usage:
#   scripts/generate_iq_target_image.sh <target> [seed]
#
# target:
#   slanted_edge | iso_noise | siemens_star
#
# Optional environment overrides:
#   PIPELINE_CONFIG=config/pipeline.yaml
#   EMVA_FROM_EXR=1
#   IQ_TARGET_OUT_DIR=scenes/generated/iq_targets
#   IQ_STAGE_OUT_DIR=out/iq_targets
#   IQ_TARGET_BUILDER_ARGS="--slanted-angle-deg 4.0"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

TARGET="${1-}"
if [[ -z "${TARGET}" ]]; then
  echo "Usage: scripts/generate_iq_target_image.sh <target> [seed]"
  echo "target: slanted_edge | iso_noise | siemens_star"
  exit 2
fi
if [[ "${TARGET}" == "-h" || "${TARGET}" == "--help" ]]; then
  sed -n '1,24p' "${BASH_SOURCE[0]}"
  exit 0
fi
case "${TARGET}" in
  slanted_edge|iso_noise|siemens_star) ;;
  *)
    echo "Unsupported target: ${TARGET}"
    echo "target: slanted_edge | iso_noise | siemens_star"
    exit 2
    ;;
esac

PIPELINE_CONFIG="${PIPELINE_CONFIG:-config/pipeline.yaml}"
EMVA_FROM_EXR="${EMVA_FROM_EXR:-0}"
IQ_TARGET_OUT_DIR="${IQ_TARGET_OUT_DIR:-scenes/generated/iq_targets}"
IQ_STAGE_OUT_DIR="${IQ_STAGE_OUT_DIR:-out/iq_targets}"
IQ_TARGET_BUILDER_ARGS="${IQ_TARGET_BUILDER_ARGS:-}"

PY="${REPO_DIR}/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Missing python venv at ${PY}"
  exit 2
fi

while IFS= read -r -d '' kv; do
  export "${kv}"
done < <("${PY}" "${REPO_DIR}/tools/pipeline_shell_env.py" "${REPO_DIR}" "${PIPELINE_CONFIG}" --format env0)

PBRT="${REPO_DIR}/${PBRT_REL}"
if [[ ! -x "${PBRT}" ]]; then
  echo "Missing pbrt binary at ${PBRT}"
  exit 2
fi

SEED_USER="${2-}"
if [[ "${SEED_USER}" == "-h" || "${SEED_USER}" == "--help" ]]; then
  sed -n '1,24p' "${BASH_SOURCE[0]}"
  exit 0
fi
if [[ -n "${SEED_USER}" ]]; then
  SEED="${SEED_USER}"
else
  SEED="${DEFAULT_NOISE_SEED}"
fi

TARGET_SCENE_REL="${IQ_TARGET_OUT_DIR}/${TARGET}.pbrt"
TARGET_SCENE_ABS="${REPO_DIR}/${TARGET_SCENE_REL}"
TARGET_EXR_REL="out/${TARGET}_${FILM}.exr"
TARGET_EXR_ABS="${REPO_DIR}/${TARGET_EXR_REL}"
TARGET_NPZ_ABS="${REPO_DIR}/${IQ_STAGE_OUT_DIR}/${TARGET}_electrons.npz"
TARGET_NOISE_DIR_ABS="${REPO_DIR}/${IQ_STAGE_OUT_DIR}/${TARGET}_noisy_png"
TARGET_NOISE_RAW_ABS="${REPO_DIR}/${IQ_STAGE_OUT_DIR}/${TARGET}_noisy.raw16"
TARGET_DEMOSAIC_JSON_ABS="${REPO_DIR}/${IQ_STAGE_OUT_DIR}/${TARGET}_demosaic_metrics.json"

mkdir -p "${REPO_DIR}/${IQ_STAGE_OUT_DIR}"

echo "== 1/6 Build IQ target scene (${TARGET}) =="
BUILD_CMD=(
  "${PY}" "${REPO_DIR}/tools/build_image_quality_targets.py"
  --repo-root "${REPO_DIR}"
  --out-dir "${REPO_DIR}/${IQ_TARGET_OUT_DIR}"
  --target "${TARGET}"
  --film "${FILM}"
  --xres "${XRES}"
  --yres "${YRES}"
  --pixelsamples "${PIXELSAMPLES}"
  --spectral-nbuckets "${SPECTRAL_NBUCKETS}"
  --spectral-lambda-min "${SPECTRAL_LAMBDA_MIN}"
  --spectral-lambda-max "${SPECTRAL_LAMBDA_MAX}"
  --light-scale "${LIGHT_SCALE}"
  --cam-dist "${CAM_DIST}"
  --camera "${CAMERA}"
  --lensfile "${REALISTIC_LENSFILE_REL}"
  --aperture-diameter-mm "${REALISTIC_APERTURE_MM}"
)
if [[ -n "${REALISTIC_FOCUS_DISTANCE:-}" ]]; then
  BUILD_CMD+=(--focus-distance "${REALISTIC_FOCUS_DISTANCE}")
fi
if [[ -n "${IQ_TARGET_BUILDER_ARGS}" ]]; then
  read -r -a EXTRA_BUILD_ARGS <<< "${IQ_TARGET_BUILDER_ARGS}"
  BUILD_CMD+=( "${EXTRA_BUILD_ARGS[@]}" )
fi
"${BUILD_CMD[@]}"

if [[ ! -f "${TARGET_SCENE_ABS}" ]]; then
  echo "Missing generated scene: ${TARGET_SCENE_ABS}"
  exit 2
fi

echo "== 2/6 Render with pbrt (${TARGET}) =="
"${PBRT}" "${TARGET_SCENE_ABS}"

if [[ "${POST_PSF_ENABLED}" == 1 ]]; then
  echo "== 3/6 Post PSF on EXR =="
  "${PY}" "${REPO_DIR}/${PSF_TOOL_REL}" \
    --repo-root "${REPO_DIR}" \
    --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}" \
    --exr-in "${TARGET_EXR_ABS}"
else
  echo "== 3/6 Post PSF (skipped: camera model post_psf disabled) =="
fi

if [[ "${SENSOR_FORWARD_ENABLED}" == 1 ]]; then
  if [[ "${SENSOR_FORWARD_MODE}" == pbrt_exr ]]; then
    echo "== 4/6 Electrons from rendered spectral EXR =="
    SF_CMD=(
      "${PY}" "${REPO_DIR}/${PBRT_EXR_TO_ELECTRONS_TOOL_REL}"
      --repo-root "${REPO_DIR}"
      --exr "${TARGET_EXR_ABS}"
      --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}"
      --out "${TARGET_NPZ_ABS}"
    )
    if [[ -n "${SENSOR_FORWARD_TARGET_LUX}" ]]; then
      SF_CMD+=(--target-illuminance-lux "${SENSOR_FORWARD_TARGET_LUX}")
    fi
    if [[ -n "${EXPOSURE_TIME_OVERRIDE_S:-}" ]]; then
      SF_CMD+=(--integration-time-s "${EXPOSURE_TIME_OVERRIDE_S}")
    fi
    "${SF_CMD[@]}"
  else
    echo "== 4/6 Sensor forward analytic mode (skipped for scene target) =="
    echo "          set sensor_forward.mode: pbrt_exr for per-target electrons."
  fi
else
  echo "== 4/6 Electrons generation (skipped: sensor_forward.enabled false) =="
fi

if [[ "${NOISE_ENABLED}" == 1 ]]; then
  echo "== 5/6 EMVA + Bayer + demosaic previews =="
  NOISE_CMD=(
    "${REPO_DIR}/${NOISE_TOOL_REL}"
    --repo-root "${REPO_DIR}"
    --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}"
    --seed "${SEED}"
    --linear-exr "${TARGET_EXR_ABS}"
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
  elif [[ -f "${TARGET_NPZ_ABS}" ]]; then
    USE_NPZ=1
  fi
  if [[ "${USE_NPZ}" == 1 ]]; then
    NOISE_CMD+=(--electrons-npz "${TARGET_NPZ_ABS}")
  fi
  "${PY}" "${NOISE_CMD[@]}"

  # apply_emva_noise currently writes fixed colorchecker output names.
  # Preserve per-target outputs so repeated target runs do not overwrite each other.
  cp -f "${REPO_DIR}/out/colorchecker_noisy.raw16" "${TARGET_NOISE_RAW_ABS}"
  rm -rf "${TARGET_NOISE_DIR_ABS}"
  cp -R "${REPO_DIR}/out/colorchecker_noisy_png" "${TARGET_NOISE_DIR_ABS}"
else
  echo "== 5/6 EMVA (skipped: noise.enabled false) =="
fi

if [[ "${VALIDATE_DEMOSAIC}" == 1 ]]; then
  if [[ -f "${TARGET_NPZ_ABS}" ]]; then
    echo "== 6/6 Demosaic linear validation =="
    "${PY}" "${REPO_DIR}/${VALIDATE_DEMOSAIC_TOOL_REL}" \
      --repo-root "${REPO_DIR}" \
      --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}" \
      --electrons-npz "${TARGET_NPZ_ABS}" \
      --crop "${DEMOSAIC_CROP}" \
      --json-out "${TARGET_DEMOSAIC_JSON_ABS}"
  else
    echo "== 6/6 Demosaic linear validation (skipped: missing ${TARGET_NPZ_ABS}) =="
  fi
else
  echo "== 6/6 Demosaic linear validation (skipped: validate_demosaic.enabled false) =="
fi

echo
echo "Done. Pipeline config: ${PIPELINE_CONFIG_RESOLVED}"
echo "Scene: ${TARGET_SCENE_REL}"
echo "Rendered EXR: ${TARGET_EXR_REL}"
echo "Target outputs:"
echo "  ${IQ_STAGE_OUT_DIR}/${TARGET}_noisy.raw16"
echo "  ${IQ_STAGE_OUT_DIR}/${TARGET}_noisy_png/"
echo "  ${IQ_STAGE_OUT_DIR}/${TARGET}_electrons.npz"
echo "  ${IQ_STAGE_OUT_DIR}/${TARGET}_demosaic_metrics.json"
