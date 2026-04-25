#!/usr/bin/env bash
set -euo pipefail

# Batch Munsell generation pipeline (aligned with generate_colorchecker_image.sh config flow):
# 1) build grouped Munsell scenes from MAT spectra (forced spectral film)
# 2) render each generated hue-family scene with pbrt
# 3) optional post-PSF (config/optics.yaml)
# 4) optional sensor forward (from pipeline config)
# 5) optional EMVA noise model per rendered EXR
# 6) optional demosaic linear validation
#
# Usage:
#   scripts/generate_munsell_batch.sh [seed] [hue]
#
# Optional environment overrides:
#   PIPELINE_CONFIG=config/pipeline.yaml
#   EMVA_FROM_EXR=1
#   MUNSELL_OUT_DIR=scenes/generated/munsell
#   MUNSELL_BUILD_ARGS="--hues R,YR,Y --columns 10"
#   MUNSELL_SINGLE_HUE=<hue>
#   PBRT_BIN=third_party/pbrt-v4/build/pbrt
#   PREVIEW_PERCENTILE=99.5
#   EXPOSURE_SCALE=<float>

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

SEED="${1:-0}"
MUNSELL_SINGLE_HUE="${2:-${MUNSELL_SINGLE_HUE:-}}"
MUNSELL_OUT_DIR="${MUNSELL_OUT_DIR:-scenes/generated/munsell}"
PBRT_BIN_REL="${PBRT_BIN:-${PBRT_REL}}"
PBRT_BIN_ABS="${REPO_DIR}/${PBRT_BIN_REL}"
PREVIEW_PERCENTILE="${PREVIEW_PERCENTILE:-${NOISE_PREVIEW_PERCENTILE}}"
MUNSELL_BUILD_ARGS="${MUNSELL_BUILD_ARGS:-}"
SENSOR_FORWARD_OUT_DIR="${REPO_DIR}/out/munsell_sensor_forward"
DEMOSAIC_OUT_DIR="${REPO_DIR}/out/munsell_demosaic_metrics"

if [[ ! -x "${PBRT_BIN_ABS}" ]]; then
  echo "Missing pbrt binary at ${PBRT_BIN_ABS}"
  exit 2
fi
if [[ ! -f "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}" ]]; then
  echo "Missing camera model config at ${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}"
  exit 2
fi

build_scene_index() {
  "${PY}" - <<'PY' "${REPO_DIR}" "${MUNSELL_OUT_DIR}"
from pathlib import Path
import json
import sys
repo = Path(sys.argv[1]).resolve()
out_dir = (repo / sys.argv[2]).resolve()
index_path = out_dir / "index.json"
if index_path.is_file():
    idx = json.loads(index_path.read_text())
    for key in idx.get("hues_generated", []):
        p = out_dir / key / f"munsell_{key}.pbrt"
        if p.is_file():
            print(str(p))
else:
    for p in sorted(out_dir.glob("*/munsell_*.pbrt")):
        print(str(p))
PY
}

select_scenes_by_hue_set() {
  local hue_raw="$1"
  local hue_norm scene key
  hue_norm="$(printf '%s' "${hue_raw}" | tr '[:lower:]' '[:upper:]')"
  while IFS= read -r scene; do
    [[ -z "${scene}" ]] && continue
    key="$(basename "$(dirname "${scene}")")"
    key="$(printf '%s' "${key}" | tr '[:lower:]' '[:upper:]')"
    if [[ "${key}" == "${hue_norm}" || "${key}" == "${hue_norm}_"* ]]; then
      printf '%s\n' "${scene}"
    fi
  done < <(build_scene_index)
}

echo "== 1/6 Build Munsell scenes into ${MUNSELL_OUT_DIR} =="
BUILD_CMD=(
  "${PY}" tools/build_munsell_scenes.py
  --repo-root "${REPO_DIR}"
  --out-dir "${MUNSELL_OUT_DIR}"
  --illuminant "${RENDER_ILLUMINANT_REL}"
)
if [[ -n "${MUNSELL_BUILD_ARGS}" ]]; then
  read -r -a EXTRA_ARGS <<< "${MUNSELL_BUILD_ARGS}"
  BUILD_CMD+=( "${EXTRA_ARGS[@]}" )
fi
if [[ -n "${MUNSELL_SINGLE_HUE}" ]]; then
  echo "Single hue mode: ${MUNSELL_SINGLE_HUE}"
  BUILD_CMD+=( --hues "${MUNSELL_SINGLE_HUE}" )
fi
# Force spectral film so EXRs include S0.*nm channels for integrate_qe.
# Keep this flag last so any accidental --film in MUNSELL_BUILD_ARGS is overridden.
BUILD_CMD+=( --film spectral )
"${BUILD_CMD[@]}"

echo "== 2/6 Render generated Munsell scenes =="
if [[ -n "${MUNSELL_SINGLE_HUE}" ]]; then
  mapfile -t SCENES < <(select_scenes_by_hue_set "${MUNSELL_SINGLE_HUE}")
  if [[ "${#SCENES[@]}" -eq 0 ]]; then
    echo "Requested hue '${MUNSELL_SINGLE_HUE}' was not generated under ${MUNSELL_OUT_DIR}"
    exit 2
  fi
else
  mapfile -t SCENES < <(build_scene_index)
fi

if [[ "${#SCENES[@]}" -eq 0 ]]; then
  echo "No Munsell scenes found under ${MUNSELL_OUT_DIR}"
  exit 2
fi

for scene in "${SCENES[@]}"; do
  echo "Rendering: ${scene}"
  "${PBRT_BIN_ABS}" "${scene}"
done

echo "== 3-6/6 Post stages per Munsell render =="
for scene in "${SCENES[@]}"; do
  hue="$(basename "$(dirname "${scene}")")"
  scene_stem="$(basename "${scene}" .pbrt)"
  munsell_manifest_abs="$(dirname "${scene}")/${scene_stem}_manifest.json"
  noisy_raw_rel="out/munsell_noise/${hue}/munsell_${hue}_noisy.raw16"
  noisy_png_rel="out/munsell_noise/${hue}/munsell_${hue}_noisy_png"
  exr_rel="out/munsell_${hue}_spectral.exr"
  exr_abs="${REPO_DIR}/${exr_rel}"

  if [[ ! -f "${exr_abs}" ]]; then
    alt_exr_rel="out/munsell_${hue}.exr"
    alt_exr_abs="${REPO_DIR}/${alt_exr_rel}"
    if [[ -f "${alt_exr_abs}" ]]; then
      exr_rel="${alt_exr_rel}"
      exr_abs="${alt_exr_abs}"
    fi
  fi
  if [[ ! -f "${exr_abs}" ]]; then
    echo "Skipping ${hue}: rendered EXR not found (checked ${exr_rel} and out/munsell_${hue}.exr)"
    continue
  fi

  if [[ "${POST_PSF_ENABLED}" == 1 ]]; then
    echo "Post-PSF: hue=${hue}, exr=${exr_rel}"
    "${PY}" "${REPO_DIR}/${PSF_TOOL_REL}" \
      --repo-root "${REPO_DIR}" \
        --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}" \
      --exr-in "${exr_abs}"
  fi

  munsell_npz_abs="${SENSOR_FORWARD_OUT_DIR}/munsell_${hue}_electrons.npz"
  if [[ "${SENSOR_FORWARD_ENABLED}" == 1 ]]; then
    if [[ "${SENSOR_FORWARD_MODE}" == pbrt_exr ]]; then
      echo "Sensor-forward pbrt_exr: hue=${hue}, exr=${exr_rel}"
      mkdir -p "${SENSOR_FORWARD_OUT_DIR}"
      "${PY}" "${REPO_DIR}/${PBRT_EXR_TO_ELECTRONS_TOOL_REL}" \
        --repo-root "${REPO_DIR}" \
        --exr "${exr_abs}" \
        --scene-manifest-json "${munsell_manifest_abs}" \
        --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}" \
        --out "${munsell_npz_abs}"
    else
      echo "Sensor-forward analytic mode is not scene-specific; running once from pipeline config."
      "${PY}" "${REPO_DIR}/tools/spectral_sensor_forward.py" \
        --repo-root "${REPO_DIR}" \
        --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}"
    fi
  fi

  if [[ "${NOISE_ENABLED}" == 1 ]]; then
    echo "Noise model: hue=${hue}, exr=${exr_rel}"
    NOISE_CMD=(
      "${PY}" "${REPO_DIR}/tools/apply_emva_noise.py"
      --repo-root "${REPO_DIR}"
      --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}"
      --seed "${SEED}"
      --linear-exr "${exr_abs}"
      --preview-percentile "${PREVIEW_PERCENTILE}"
    )
    if [[ -n "${EXPOSURE_SCALE:-}" ]]; then
      NOISE_CMD+=( --exposure-scale "${EXPOSURE_SCALE}" )
    elif [[ -n "${NOISE_EXPOSURE_SCALE:-}" ]]; then
      NOISE_CMD+=( --exposure-scale "${NOISE_EXPOSURE_SCALE}" )
    fi
    USE_NPZ=0
    if [[ "${EMVA_FROM_EXR}" == 1 ]]; then
      USE_NPZ=0
    elif [[ "${SENSOR_FORWARD_ENABLED}" == 1 && "${SENSOR_FORWARD_MODE}" == pbrt_exr && -f "${munsell_npz_abs}" ]]; then
      USE_NPZ=1
    fi
    if [[ "${USE_NPZ}" == 1 ]]; then
      NOISE_CMD+=( --electrons-npz "${munsell_npz_abs}" )
    fi
    "${NOISE_CMD[@]}"
  else
    echo "Noise disabled in pipeline config; skipping hue=${hue}"
    continue
  fi

  mkdir -p "$(dirname "${REPO_DIR}/${noisy_raw_rel}")"
  cp -f "${REPO_DIR}/out/colorchecker_noisy.raw16" "${REPO_DIR}/${noisy_raw_rel}"
  rm -rf "${REPO_DIR}/${noisy_png_rel}"
  cp -R "${REPO_DIR}/out/colorchecker_noisy_png" "${REPO_DIR}/${noisy_png_rel}"

  if [[ "${VALIDATE_DEMOSAIC}" == 1 ]]; then
    mkdir -p "${DEMOSAIC_OUT_DIR}"
    munsell_metrics_abs="${DEMOSAIC_OUT_DIR}/munsell_${hue}_demosaic_metrics.json"
    DEMOSAIC_CMD=(
      "${PY}" "${REPO_DIR}/tools/validate_demosaic_linear.py"
      --repo-root "${REPO_DIR}"
      --camera-model-config "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}"
      --crop "${DEMOSAIC_CROP}"
      --json-out "${munsell_metrics_abs}"
    )
    if [[ -f "${munsell_npz_abs}" ]]; then
      DEMOSAIC_CMD+=( --electrons-npz "${munsell_npz_abs}" )
    fi
    "${DEMOSAIC_CMD[@]}"
  fi
done

echo
echo "Done."
echo "Pipeline config: ${PIPELINE_CONFIG_RESOLVED}"
echo "Munsell scenes: ${MUNSELL_OUT_DIR}"
echo "Noise outputs: out/munsell_noise/<hue>/"
