#!/usr/bin/env bash
# Install the release parity app, stage the verified fixture into its sandbox,
# launch it offline, and pull the value-free report it writes.
#
#   scripts/run-on-device.sh ios APP_PATH FIXTURE_DIR REPORT_OUT
#   scripts/run-on-device.sh android APK_PATH FIXTURE_DIR REPORT_OUT
#
# The fixture is copied to local app storage before launch; the app itself
# never downloads anything. On Android the emulator is put in airplane mode
# before launch so native code cannot reach the network either.
set -euo pipefail

platform="${1:?platform}"
artifact="${2:?app or apk path}"
fixture="${3:?fixture directory}"
report_out="${4:?report output path}"
BUNDLE_ID="org.openmed.openmedkit.parity"
FIXTURE_NAME="openmed-parity-fixture"
REPORT_NAME="openmed-rn-parity-report.json"
ATTEMPTS="${OPENMED_PARITY_POLL_ATTEMPTS:-360}"

case "$platform" in
  ios)
    xcrun simctl install booted "$artifact"
    data="$(xcrun simctl get_app_container booted "$BUNDLE_ID" data)"
    rm -rf "$data/Documents/$FIXTURE_NAME" "$data/Documents/$REPORT_NAME"
    mkdir -p "$data/Documents"
    cp -R "$fixture" "$data/Documents/$FIXTURE_NAME"
    xcrun simctl launch booted "$BUNDLE_ID"
    report="$data/Documents/$REPORT_NAME"
    for _ in $(seq 1 "$ATTEMPTS"); do
      [[ -f "$report" ]] && break
      sleep 5
    done
    if [[ ! -f "$report" ]]; then
      echo "::error::the parity app did not write a report" >&2
      exit 1
    fi
    cp "$report" "$report_out"
    ;;
  android)
    files="/sdcard/Android/data/$BUNDLE_ID/files"
    adb install -r "$artifact"
    adb shell rm -rf "$files/$FIXTURE_NAME" "$files/$REPORT_NAME"
    adb shell mkdir -p "$files"
    adb push "$fixture" "$files/$FIXTURE_NAME" >/dev/null
    adb shell cmd connectivity airplane-mode enable || true
    adb shell svc wifi disable || true
    adb shell svc data disable || true
    adb logcat -c || true
    adb shell am start -W -n "$BUNDLE_ID/.MainActivity"
    for _ in $(seq 1 "$ATTEMPTS"); do
      if adb shell ls "$files/$REPORT_NAME" >/dev/null 2>&1; then
        break
      fi
      sleep 5
    done
    if ! adb pull "$files/$REPORT_NAME" "$report_out" >/dev/null; then
      echo "::error::the parity app did not write a report" >&2
      adb logcat -d -s ReactNativeJS:V | grep OPENMED_RN_PARITY || true
      exit 1
    fi
    ;;
  *)
    echo "usage: run-on-device.sh ios|android ARTIFACT FIXTURE_DIR REPORT_OUT" >&2
    exit 2
    ;;
esac

echo "pulled parity report for $platform"
