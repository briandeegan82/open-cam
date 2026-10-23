#!/usr/bin/env bash
# Driving-scene figure renders (pbrt-v4 spectral film, unbiased path tracer, builder defaults otherwise).
# Writes paper/ei2027/build/driving/<name>/highway_spectral.exr + manifest. Run from the repo root.
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
# 80 m fog: multiple scattering in the medium leaves ~34 % per-pixel render noise at 256 spp; 4096 gives ~8 %.
SPP_FOG="${SPP_FOG:-4096}"

render() {
  local name="$1" spp="$2"; shift 2
  local d="${B}/${name}"
  mkdir -p "${d}/out"
  "${PY}" tools/build_highway_scene.py --xres "${XRES}" --yres "${YRES}" --pixelsamples "${spp}" \
    --out-dir "${d}" "$@"
  (cd "${d}" && "${ROOT}/${PBRT}" --quiet highway.pbrt)
}

render day_dry   "${SPP_DAY}"
render day_wet   "${SPP_DAY}"   --road-wetness wet --puddles --spray
render fog_80m   "${SPP_FOG}"   --haze fog --visibility-m 80
render night_wet "${SPP_NIGHT}" --time-of-day night --road-wetness wet --puddles
