#!/usr/bin/env bash
# Fail fast unless the OpenMedKitRN bridge can be autolinked into a host app.
#
# The bridge package on master ships only src/index.ts and the two native
# module sources. Autolinking needs the packaging from #4478: a podspec, the
# Objective-C RCT_EXTERN_MODULE export, an Android Gradle module with a
# ReactPackage, react-native.config.js, and a CommonJS package.json. Until
# that lands, the simulator and emulator parity jobs cannot build.
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
missing=()
for path in \
  openmedkit-react-native.podspec \
  ios/OpenMedKitRN.m \
  android/build.gradle \
  react-native.config.js; do
  [[ -e "$PACKAGE_DIR/$path" ]] || missing+=("$path")
done
if ! find "$PACKAGE_DIR/android" -name 'OpenMedKitRnPackage.kt' -print -quit | grep -q .; then
  missing+=("android/**/OpenMedKitRnPackage.kt")
fi
if node -e 'process.exit(require(process.argv[1]).type === "module" ? 0 : 1)' \
  "$PACKAGE_DIR/package.json"; then
  missing+=("package.json without \"type\": \"module\"")
fi

if ((${#missing[@]} > 0)); then
  echo "::error::React Native parity needs the OpenMedKitRN packaging from #4478. Missing: ${missing[*]}" >&2
  exit 1
fi
echo "OpenMedKitRN packaging present"
