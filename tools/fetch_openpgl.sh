#!/usr/bin/env bash
# Fetch and build Intel Open PGL (path guiding library, Apache-2.0, https://github.com/RenderKit/openpgl)
# into third_party/openpgl for pbrt's "guidedpath" integrator (patch 0031, docs/PATH_GUIDING.md).
# Pinned by tag and commit; static library; needs CMake, a C++ compiler and TBB (Ubuntu: libtbb-dev).
# Usage: tools/fetch_openpgl.sh   (idempotent; prints the openpgl_DIR for CMake)
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG=v0.7.1
COMMIT=5c50cd7a331647a69e28cc8a19ff204039cbe78d
dest="$repo/third_party/openpgl"
src="$dest/src"
prefix="$dest/install"
cfg="$prefix/lib/cmake/openpgl-${TAG#v}"

if [ -f "$cfg/openpglConfig.cmake" ]; then
  echo "$cfg"
  exit 0
fi
if [ ! -d "$src/.git" ]; then
  git clone -q --depth 1 --branch "$TAG" https://github.com/RenderKit/openpgl "$src" >&2
fi
got="$(git -C "$src" rev-parse HEAD)"
if [ "$got" != "$COMMIT" ]; then
  echo "fetch_openpgl.sh: $TAG is $got, expected $COMMIT; refusing to build" >&2
  exit 1
fi
cmake -S "$src" -B "$dest/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DOPENPGL_BUILD_STATIC=ON >&2
cmake --build "$dest/build" -j"$(nproc)" >&2
cmake --install "$dest/build" >&2
echo "$cfg"
