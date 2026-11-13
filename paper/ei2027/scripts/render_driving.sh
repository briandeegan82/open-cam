#!/usr/bin/env bash
# Driving-scene figure renders (pbrt-v4 spectral film, path tracer, builder defaults; the SWIR panel uses the 900-1600 nm film passband).
# The day panels are unbiased. The night panel has the builder's night default --lamp-regularize 0.1 (biased, docs/HIGHWAY_SCENES.md)
# and is rendered as NIGHT_SEEDS independent renders of SPP_NIGHT / NIGHT_SEEDS samples, merged with tools/merge_seeds.py
# (speckle becomes a local glow, the image total is kept; docs/DENOISING.md). NIGHT_SEEDS=1 gives the single plain render.
# Writes paper/ei2027/build/driving/<name>/highway_spectral.exr + manifest. Run from the repo root.
# The wet renders are also denoised with Open Image Denoise (docs/DENOISING.md; venv/bin/python tools/fetch_oidn.py)
# into out/highway_spectral_denoised.exr, which the figure uses for panels (b) and (d); DENOISE_WET=0 skips it.
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
NIGHT_SEEDS="${NIGHT_SEEDS:-4}"
# SWIR panel needs the extended-range build (docs/SWIR.md):
#   PBRT_NATIVE=OFF PBRT_LAMBDA_MAX_NM=2500 PBRT_BUILD_DIR=third_party/pbrt-v4/build-swir tools/build_pbrt.sh
PBRT_SWIR="${PBRT_SWIR:-third_party/pbrt-v4/build-swir/pbrt}"
DENOISE_WET="${DENOISE_WET:-1}"

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

# render_seeds <name> <total spp> <seeds> <builder args...>: <seeds> renders of spp / seeds, merged
render_seeds() {
  local name="$1" spp="$2" n="$3"; shift 3
  local d="${B}/${name}" files=() s
  mkdir -p "${d}/out"
  "${PY}" tools/build_highway_scene.py --xres "${XRES}" --yres "${YRES}" --pixelsamples "$((spp / n))" \
    --out-dir "${d}" "$@"
  for s in $(seq "${n}"); do
    (cd "${d}" && "${ROOT}/${PBRT}" --quiet --seed "${s}" --outfile "out/highway_spectral_s${s}.exr" highway.pbrt)
    files+=("${d}/out/highway_spectral_s${s}.exr")
  done
  "${PY}" tools/merge_seeds.py --exr-in "${files[@]}" --exr-out "${d}/out/highway_spectral.exr" > /dev/null
}

render day_dry   "${SPP_DAY}"
render day_wet   "${SPP_DAY}"   --road-wetness wet --puddles --spray
render swir_900_1600 "${SPP_DAY}" --swir --spectral-library usgs --spectral-lambda-max 1600 \
  --film-lambda-min 900 --spectral-nbuckets 36
if [[ "${NIGHT_SEEDS}" -ge 3 ]]; then
  render_seeds night_wet "${SPP_NIGHT}" "${NIGHT_SEEDS}" --time-of-day night --road-wetness wet --puddles
else
  render night_wet "${SPP_NIGHT}" --time-of-day night --road-wetness wet --puddles
fi
if [[ "${DENOISE_WET}" == "1" ]]; then
  for name in day_wet night_wet; do
    "${PY}" tools/denoise_spectral_exr.py --exr-in "${B}/${name}/out/highway_spectral.exr" \
      --scene "${B}/${name}/highway.pbrt" --pbrt "${PBRT}"
  done
fi
