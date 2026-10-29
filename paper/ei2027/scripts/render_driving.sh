#!/usr/bin/env bash
# Driving-scene figure renders (pbrt-v4 spectral film, unbiased path tracer, builder defaults otherwise; the SWIR panel uses the 900-1600 nm film passband).
# Writes paper/ei2027/build/driving/<name>/highway_spectral.exr + manifest. Run from the repo root.
# The night render is also denoised with Open Image Denoise (docs/DENOISING.md; venv/bin/python tools/fetch_oidn.py)
# into out/highway_spectral_denoised.exr, which the figure uses for panel (d); DENOISE_NIGHT=0 skips it.
set -euo pipefail
cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"
PY="${PY:-venv/bin/python}"
PBRT="${PBRT:-third_party/pbrt-v4/build/pbrt}"
B="${B:-paper/ei2027/build/driving}"
XRES="${XRES:-960}"
YRES="${YRES:-540}"
SPP_DAY="${SPP_DAY:-256}"
SPP_NIGHT="${SPP_NIGHT:-1024}"
# SWIR panel needs the extended-range build (docs/SWIR.md):
#   PBRT_NATIVE=OFF PBRT_LAMBDA_MAX_NM=2500 PBRT_BUILD_DIR=third_party/pbrt-v4/build-swir tools/build_pbrt.sh
PBRT_SWIR="${PBRT_SWIR:-third_party/pbrt-v4/build-swir/pbrt}"
DENOISE_NIGHT="${DENOISE_NIGHT:-1}"

render() {
  local name="$1" spp="$2"; shift 2
  local pbrt="${PBRT}"
  if [[ "${1:-}" == "--swir" ]]; then pbrt="${PBRT_SWIR}"; shift; fi
  local d="${B}/${name}"
  mkdir -p "${d}/out"
  "${PY}" tools/build_highway_scene.py --xres "${XRES}" --yres "${YRES}" --pixelsamples "${spp}" \
    --out-dir "${d}" "$@"
  (cd "${d}" && "${ROOT}/${pbrt}" --quiet highway.pbrt)
}

render day_dry   "${SPP_DAY}"
render day_wet   "${SPP_DAY}"   --road-wetness wet --puddles --spray
render swir_900_1600 "${SPP_DAY}" --swir --spectral-library usgs --spectral-lambda-max 1600 \
  --film-lambda-min 900 --spectral-nbuckets 36
render night_wet "${SPP_NIGHT}" --time-of-day night --road-wetness wet --puddles
if [[ "${DENOISE_NIGHT}" == "1" ]]; then
  "${PY}" tools/denoise_spectral_exr.py --exr-in "${B}/night_wet/out/highway_spectral.exr" \
    --scene "${B}/night_wet/highway.pbrt" --pbrt "${PBRT}"
fi
