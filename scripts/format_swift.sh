#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/.swift-format"
PATHS=(
  "$ROOT/Package.swift"
  "$ROOT/swift/OpenMedKit/Package.swift"
  "$ROOT/swift/OpenMedKit/Sources"
  "$ROOT/swift/OpenMedKit/Tests"
)

swift format format --in-place --recursive --parallel --configuration "$CONFIG" "${PATHS[@]}"
