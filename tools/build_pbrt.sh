#!/usr/bin/env bash
# Apply open-cam's pbrt-v4 patches (third_party/patches/*.patch) to the submodule
# checkout and build the CPU pbrt executable. Idempotent: already-applied patches
# are skipped. Patches are never committed into the submodule.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
pbrt="$root/third_party/pbrt-v4"
for p in "$root"/third_party/patches/*.patch; do
  [ -e "$p" ] || continue
  if git -C "$pbrt" apply --reverse --check "$p" 2>/dev/null; then
    echo "already applied: $(basename "$p")"
  else
    git -C "$pbrt" apply "$p"
    echo "applied: $(basename "$p")"
  fi
done
if [ "${1:-}" = "--patch-only" ]; then exit 0; fi
env -u PBRT_OPTIX_PATH cmake -S "$pbrt" -B "$pbrt/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$pbrt/build" -j"$(nproc)" --target pbrt_exe
