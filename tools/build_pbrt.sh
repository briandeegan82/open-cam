#!/usr/bin/env bash
# Build the CPU pbrt-v4 binary from the third_party/pbrt-v4 submodule with the open-cam
# patches in third_party/patches/*.patch applied (idempotent; the submodule commit is
# never changed). Usage: tools/build_pbrt.sh [--apply-only|--patch-only] [extra cmake --build args]
# NIR build (spectral range 360..PBRT_LAMBDA_MAX_NM, patch 0002):
#   PBRT_LAMBDA_MAX_NM=1100 PBRT_BUILD_DIR=third_party/pbrt-v4/build-nir tools/build_pbrt.sh
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$repo/third_party/pbrt-v4"
build="${PBRT_BUILD_DIR:-$src/build}"

for p in "$repo"/third_party/patches/*.patch; do
  [ -e "$p" ] || continue
  if git -C "$src" apply --reverse --check "$p" 2>/dev/null; then
    echo "already applied: $(basename "$p")"
  else
    git -C "$src" apply "$p"
    echo "applied: $(basename "$p")"
  fi
done
case "${1:-}" in --apply-only | --patch-only) exit 0 ;; esac

env -u PBRT_OPTIX_PATH cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DPBRT_BUILD_NATIVE_EXECUTABLE="${PBRT_NATIVE:-ON}" \
  -DCMAKE_CXX_FLAGS="${PBRT_LAMBDA_MAX_NM:+-DPBRT_LAMBDA_MAX_NM=${PBRT_LAMBDA_MAX_NM}}"
env -u PBRT_OPTIX_PATH cmake --build "$build" -j"$(nproc)" --target pbrt_exe "$@"
echo "pbrt: $build/pbrt"
