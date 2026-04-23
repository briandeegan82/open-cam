#!/usr/bin/env bash
set -euo pipefail

# Build and render a dedicated stray-light stress scene, then produce OFF/ON
# post-PSF outputs from the same baseline EXR for fair A/B comparison.
#
# Usage:
#   scripts/test_straylight.sh
#
# Optional env overrides:
#   PIPELINE_CONFIG=config/pipeline.yaml
#   STRAY_VEILING=0.02
#   STRAY_HALO_SIGMA=16
#   STRAY_HALO_STRENGTH=0.08
#   STRAY_BRIGHT_RADIANCE=250

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

SCENE_REL="scenes/generated/straylight_test.pbrt"
BASE_EXR_REL="out/straylight_test_base.exr"
OFF_EXR_REL="out/straylight_test_psf_straylight_off.exr"
ON_EXR_REL="out/straylight_test_psf_straylight_on.exr"

STRAY_VEILING="${STRAY_VEILING:-0.02}"
STRAY_HALO_SIGMA="${STRAY_HALO_SIGMA:-16.0}"
STRAY_HALO_STRENGTH="${STRAY_HALO_STRENGTH:-0.08}"
STRAY_BRIGHT_RADIANCE="${STRAY_BRIGHT_RADIANCE:-250.0}"

OFF_CFG="${REPO_DIR}/out/tmp_camera_straylight_off.yaml"
ON_CFG="${REPO_DIR}/out/tmp_camera_straylight_on.yaml"
cleanup() {
  rm -f "${OFF_CFG}" "${ON_CFG}"
}
trap cleanup EXIT

echo "== 1/5 Build stray-light test scene =="
"${PY}" tools/build_straylight_test_scene.py \
  --repo-root "${REPO_DIR}" \
  --out-scene "${REPO_DIR}/${SCENE_REL}" \
  --film spectral \
  --film-output "${BASE_EXR_REL}" \
  --xres "${XRES}" \
  --yres "${YRES}" \
  --pixelsamples "${PIXELSAMPLES}" \
  --spectral-nbuckets "${SPECTRAL_NBUCKETS}" \
  --spectral-lambda-min "${SPECTRAL_LAMBDA_MIN}" \
  --spectral-lambda-max "${SPECTRAL_LAMBDA_MAX}" \
  --cam-dist "${CAM_DIST}" \
  --camera "${CAMERA}" \
  --lensfile "${REALISTIC_LENSFILE_REL}" \
  --aperture-diameter-mm "${REALISTIC_APERTURE_MM}" \
  --focus-distance "${REALISTIC_FOCUS_DISTANCE}" \
  --bright-radiance "${STRAY_BRIGHT_RADIANCE}"

echo "== 2/5 Render baseline EXR =="
"${PBRT}" "${REPO_DIR}/${SCENE_REL}"

echo "== 3/5 Prepare OFF/ON camera configs =="
"${PY}" - <<'PY' "${REPO_DIR}/${CAMERA_MODEL_CONFIG_REL}" "${OFF_CFG}" "${ON_CFG}" "${STRAY_VEILING}" "${STRAY_HALO_SIGMA}" "${STRAY_HALO_STRENGTH}"
import sys
from pathlib import Path
import copy
import yaml

src = Path(sys.argv[1])
off_out = Path(sys.argv[2])
on_out = Path(sys.argv[3])
veiling = float(sys.argv[4])
halo_sigma = float(sys.argv[5])
halo_strength = float(sys.argv[6])

cfg = yaml.safe_load(src.read_text()) or {}
lens = cfg.setdefault("lens", {})
post = lens.setdefault("post_psf", {})
post["enabled"] = True
post.setdefault("mode", "gaussian")
post.setdefault("sigma_pixels", 0.75)
stray = post.setdefault("stray_light", {})
stray.setdefault("veiling_glare_fraction", veiling)
stray.setdefault("halo_sigma_pixels", halo_sigma)
stray.setdefault("halo_strength", halo_strength)

cfg_off = copy.deepcopy(cfg)
cfg_on = copy.deepcopy(cfg)
cfg_off["lens"]["post_psf"]["stray_light"]["enabled"] = False
cfg_on["lens"]["post_psf"]["stray_light"]["enabled"] = True
cfg_on["lens"]["post_psf"]["stray_light"]["veiling_glare_fraction"] = veiling
cfg_on["lens"]["post_psf"]["stray_light"]["halo_sigma_pixels"] = halo_sigma
cfg_on["lens"]["post_psf"]["stray_light"]["halo_strength"] = halo_strength

off_out.parent.mkdir(parents=True, exist_ok=True)
off_out.write_text(yaml.safe_dump(cfg_off, sort_keys=False))
on_out.write_text(yaml.safe_dump(cfg_on, sort_keys=False))
PY

echo "== 4/5 Apply post-PSF with stray-light OFF =="
cp "${REPO_DIR}/${BASE_EXR_REL}" "${REPO_DIR}/${OFF_EXR_REL}"
"${PY}" "${REPO_DIR}/${PSF_TOOL_REL}" \
  --repo-root "${REPO_DIR}" \
  --camera-model-config "${OFF_CFG}" \
  --exr-in "${REPO_DIR}/${OFF_EXR_REL}"

echo "== 5/5 Apply post-PSF with stray-light ON =="
cp "${REPO_DIR}/${BASE_EXR_REL}" "${REPO_DIR}/${ON_EXR_REL}"
"${PY}" "${REPO_DIR}/${PSF_TOOL_REL}" \
  --repo-root "${REPO_DIR}" \
  --camera-model-config "${ON_CFG}" \
  --exr-in "${REPO_DIR}/${ON_EXR_REL}"

echo
echo "Done. Outputs:"
echo "  Baseline EXR: ${BASE_EXR_REL}"
echo "  Stray OFF:    ${OFF_EXR_REL}"
echo "  Stray ON:     ${ON_EXR_REL}"
echo "  Temp configs: out/tmp_camera_straylight_off.yaml, out/tmp_camera_straylight_on.yaml"
