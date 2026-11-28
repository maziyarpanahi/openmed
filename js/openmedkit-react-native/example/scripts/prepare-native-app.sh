#!/usr/bin/env bash
# Generate the iOS or Android host project for the parity app.
#
#   scripts/prepare-native-app.sh ios|android
#
# The host project comes from the React Native community template pinned by
# this app's package.json (react-native and @react-native-community/cli), so
# no generated Xcode or Gradle project is committed. Autolinking then wires in
# the OpenMedKitRN bridge (`file:..`) and the example-only direct probe
# (`file:./probe`). Run `npm ci` in this directory first.
set -euo pipefail

platform="${1:?usage: prepare-native-app.sh ios|android}"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_ID="org.openmed.openmedkit.parity"
APP_NAME="$(node -p 'require(process.argv[1]).name' "$EXAMPLE_DIR/app.json")"
RN_VERSION="$(node -p 'require(process.argv[1]).dependencies["react-native"]' "$EXAMPLE_DIR/package.json")"

"$EXAMPLE_DIR/scripts/check-packaging.sh"

if [[ ! -x "$EXAMPLE_DIR/node_modules/.bin/rnc-cli" ]]; then
  echo "::error::run npm ci in $EXAMPLE_DIR before preparing the host project" >&2
  exit 1
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
(
  cd "$work"
  "$EXAMPLE_DIR/node_modules/.bin/rnc-cli" init "$APP_NAME" \
    --version "$RN_VERSION" \
    --package-name "$BUNDLE_ID" \
    --skip-install \
    --skip-git-init \
    --install-pods false
)
template="$work/$APP_NAME"

case "$platform" in
  ios)
    rm -rf "$EXAMPLE_DIR/ios"
    cp -R "$template/ios" "$EXAMPLE_DIR/ios"
    # OpenMedKit's Swift package floor is iOS 17.
    perl -0pi -e "s/platform :ios, min_ios_version_supported/platform :ios, '17.0'/" \
      "$EXAMPLE_DIR/ios/Podfile"
    perl -pi -e 's/IPHONEOS_DEPLOYMENT_TARGET = [0-9.]+;/IPHONEOS_DEPLOYMENT_TARGET = 17.0;/g' \
      "$EXAMPLE_DIR/ios/$APP_NAME.xcodeproj/project.pbxproj"
    ;;
  android)
    rm -rf "$EXAMPLE_DIR/android"
    cp -R "$template/android" "$EXAMPLE_DIR/android"
    # OpenMedKit Android requires API 26.
    perl -pi -e 's/minSdkVersion = \d+/minSdkVersion = 26/' "$EXAMPLE_DIR/android/build.gradle"
    ;;
  *)
    echo "usage: prepare-native-app.sh ios|android" >&2
    exit 2
    ;;
esac

echo "prepared $platform host project for $APP_NAME ($BUNDLE_ID, react-native $RN_VERSION)"
