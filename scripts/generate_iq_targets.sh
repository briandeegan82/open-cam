#!/usr/bin/env bash
set -euo pipefail

# Build and render image-quality targets (slanted edge / ISO noise / Siemens star).
#
# Usage:
#   scripts/generate_iq_targets.sh [target]
#
# target:
#   slanted_edge | iso_noise | siemens_star | all   (default: all)
#
# Optional environment overrides:
#   PIPELINE_CONFIG=config/pipeline.yaml
#   IQ_TARGET_OUT_DIR=scenes/generated/iq_targets
#   IQ_TARGET_FILM=spectral
#   IQ_TARGET_PIXELSAMPLES=256

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PIPELINE_CONFIG="${PIPELINE_CONFIG:-config/pipeline.yaml}"
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

TARGET="${1:-all}"
if [[ "${TARGET}" == "-h" || "${TARGET}" == "--help" ]]; then
  sed -n '1,18p' "${BASH_SOURCE[0]}"
  exit 0
fi
IQ_TARGET_OUT_DIR="${IQ_TARGET_OUT_DIR:-scenes/generated/iq_targets}"
IQ_TARGET_FILM="${IQ_TARGET_FILM:-${FILM}}"
IQ_TARGET_PIXELSAMPLES="${IQ_TARGET_PIXELSAMPLES:-${PIXELSAMPLES}}"

echo "== 1/2 Build IQ target scenes (${TARGET}) =="
"${PY}" "${REPO_DIR}/tools/build_image_quality_targets.py" \
  --repo-root "${REPO_DIR}" \
  --out-dir "${REPO_DIR}/${IQ_TARGET_OUT_DIR}" \
  --target "${TARGET}" \
  --film "${IQ_TARGET_FILM}" \
  --xres "${XRES}" \
  --yres "${YRES}" \
  --pixelsamples "${IQ_TARGET_PIXELSAMPLES}" \
  --spectral-nbuckets "${SPECTRAL_NBUCKETS}" \
  --spectral-lambda-min "${SPECTRAL_LAMBDA_MIN}" \
  --spectral-lambda-max "${SPECTRAL_LAMBDA_MAX}" \
  --light-scale "${LIGHT_SCALE}" \
  --cam-dist "${CAM_DIST}" \
  --camera "${CAMERA}" \
  --lensfile "${REALISTIC_LENSFILE_REL}" \
  --aperture-diameter-mm "${REALISTIC_APERTURE_MM}" \
  --focus-distance "${REALISTIC_FOCUS_DISTANCE}"

echo "== 2/2 Render generated target scenes =="
shopt -s nullglob
for scene in "${REPO_DIR}/${IQ_TARGET_OUT_DIR}"/*.pbrt; do
  echo "Rendering: ${scene}"
  "${PBRT}" "${scene}"
done

echo
echo "Done."
echo "Generated scenes: ${IQ_TARGET_OUT_DIR}"
echo "Rendered EXRs:"
echo "  out/slanted_edge_${IQ_TARGET_FILM}.exr"
echo "  out/iso_noise_${IQ_TARGET_FILM}.exr"
echo "  out/siemens_star_${IQ_TARGET_FILM}.exr"
