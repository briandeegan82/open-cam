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

patches=("$repo"/third_party/patches/*.patch)
[ -e "${patches[0]}" ] || patches=()

# True when the submodule tree is exactly HEAD + the whole patch stack (patches may share
# context lines, so per-patch reverse checks are not enough once later patches are on).
stack_applied() {
  local idx rc=0 i
  idx="$(mktemp)"
  GIT_INDEX_FILE="$idx" git -C "$src" read-tree HEAD
  GIT_INDEX_FILE="$idx" git -C "$src" add -u
  for ((i = ${#patches[@]} - 1; i >= 0; i--)); do
    GIT_INDEX_FILE="$idx" git -C "$src" apply --cached --reverse "${patches[i]}" 2>/dev/null || rc=1
    [ "$rc" = 0 ] || break
  done
  [ "$rc" = 0 ] && [ "$(GIT_INDEX_FILE="$idx" git -C "$src" write-tree)" = "$(git -C "$src" rev-parse 'HEAD^{tree}')" ] || rc=1
  rm -f "$idx"
  return "$rc"
}

if [ "${#patches[@]}" -gt 0 ] && stack_applied; then
  echo "already applied: all ${#patches[@]} patches"
  patches=()
fi
for p in "${patches[@]}"; do
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
