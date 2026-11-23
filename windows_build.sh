#!/usr/bin/env bash
# Builds ocl-simsand on Windows (run from Git Bash).
# Usage: ./windows_build.sh [Debug|Release]   (default: Debug)
set -euo pipefail

CONFIG="${1:-Debug}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build"

# Locate CMake: PATH first, then the default install location.
if command -v cmake >/dev/null 2>&1; then
    CMAKE="cmake"
elif [[ -x "/c/Program Files/CMake/bin/cmake.exe" ]]; then
    CMAKE="/c/Program Files/CMake/bin/cmake.exe"
else
    echo "error: CMake not found. Install it or add it to PATH." >&2
    exit 1
fi

# Configure once; CMake picks the newest installed Visual Studio by default.
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    "$CMAKE" -S "$ROOT" -B "$BUILD_DIR" -A x64
fi

"$CMAKE" --build "$BUILD_DIR" --config "$CONFIG" --parallel

echo
echo "Built: $BUILD_DIR/$CONFIG/ocl-simsand.exe"
