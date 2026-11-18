#!/usr/bin/env bash
# Build a synthetic SwiftPM consumer that requires the repository root package by
# version, the way documented installs do, and compiles references to the
# MLX-gated Maple and Compass APIs (swift/RootPackageConsumer).
#
# The root package is snapshotted into a throwaway local git repository with a
# synthetic tag that is never pushed. The build fails when the root manifest
# stops exposing the mlx-swift-lm products, or when it gains a dependency that
# SwiftPM rejects inside a version-required package (for example a revision pin).
#
# Usage: scripts/check_swift_root_consumer.sh [macos] [ios]   (default: macos)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONSUMER_SOURCES="$ROOT/swift/RootPackageConsumer/Sources"
SNAPSHOT_VERSION="99.0.0"

if [ "$#" -eq 0 ]; then
  set -- macos
fi
for platform in "$@"; do
  case "$platform" in
    macos | ios) ;;
    *)
      echo "unsupported platform: $platform (expected macos or ios)" >&2
      exit 2
      ;;
  esac
done

WORK="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/openmed-root-consumer.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
SNAPSHOT="$WORK/openmed"
CONSUMER="$WORK/consumer"
mkdir -p "$SNAPSHOT" "$CONSUMER"

# Copy the working-tree contents of every non-ignored file SwiftPM reads from the
# root package, so uncommitted manifest edits are exercised too.
git -C "$ROOT" ls-files -z --cached --others --exclude-standard -- \
  Package.swift Package.resolved LICENSE swift/OpenMedKit |
  while IFS= read -r -d '' path; do
    if [ -f "$ROOT/$path" ]; then
      mkdir -p "$SNAPSHOT/$(dirname "$path")"
      cp "$ROOT/$path" "$SNAPSHOT/$path"
    fi
  done

git -C "$SNAPSHOT" init -q
git -C "$SNAPSHOT" add -A
git -C "$SNAPSHOT" \
  -c user.name="OpenMed consumer check" \
  -c user.email="consumer-check@invalid" \
  -c commit.gpgsign=false \
  commit -q -m "Root package snapshot"
git -C "$SNAPSHOT" -c tag.gpgsign=false tag "$SNAPSHOT_VERSION"

cp -R "$CONSUMER_SOURCES" "$CONSUMER/Sources"
cat >"$CONSUMER/Package.swift" <<EOF
// swift-tools-version: 5.10

import PackageDescription

let package = Package(
    name: "RootPackageConsumer",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "RootPackageConsumer", targets: ["RootPackageConsumer"])
    ],
    dependencies: [
        .package(url: "file://$SNAPSHOT", exact: "$SNAPSHOT_VERSION")
    ],
    targets: [
        .target(
            name: "RootPackageConsumer",
            dependencies: [.product(name: "OpenMedKit", package: "openmed")]
        )
    ]
)
EOF

swift package --package-path "$CONSUMER" resolve

for platform in "$@"; do
  case "$platform" in
    macos)
      echo "Building the versioned root-package consumer for macOS"
      swift build --package-path "$CONSUMER"
      ;;
    ios)
      echo "Building the versioned root-package consumer for iOS"
      (
        cd "$CONSUMER"
        xcodebuild build \
          -scheme RootPackageConsumer \
          -destination 'generic/platform=iOS Simulator' \
          -derivedDataPath "$WORK/DerivedData" \
          -skipPackagePluginValidation
      )
      ;;
  esac
done
