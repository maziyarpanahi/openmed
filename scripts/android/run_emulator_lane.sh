#!/usr/bin/env bash
# Run the OpenMedKit emulator lane against the connected emulator or device.
#
# Usage:
#   scripts/android/run_emulator_lane.sh --lane NAME --page-size-kb 4|16 \
#       [--known-unloadable LIBRARY[,LIBRARY]] [--negative-control]
#
# 1. Runs every instrumented test in android/openmedkit/src/androidTest with
#    the lane's instrumentation arguments, then collects the value-free
#    OpenMedEmulatorLane records (case IDs, labels, offsets, timings) from
#    logcat.
# 2. With --negative-control, rebuilds the test APK without its ONNX Runtime
#    and tokenizer native libraries and requires NativeRuntimePackagingTest to
#    fail with native_library_unloadable for both of them.
#
# Reports land in android/openmedkit/build/reports/emulator-lane/<lane>/.
set -euo pipefail

lane=""
page_size_kb=""
known_unloadable=""
negative_control=false

while (($#)); do
  case "$1" in
    --lane) lane="${2:-}"; shift 2 ;;
    --page-size-kb) page_size_kb="${2:-}"; shift 2 ;;
    --known-unloadable) known_unloadable="${2:-}"; shift 2 ;;
    --negative-control) negative_control=true; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! "$lane" =~ ^[a-z0-9][a-z0-9_-]{0,47}$ ]]; then
  echo "--lane must be a short lowercase name" >&2
  exit 2
fi
if [[ "$page_size_kb" != "4" && "$page_size_kb" != "16" ]]; then
  echo "--page-size-kb must be 4 or 16" >&2
  exit 2
fi

# `adb get-state` fails at once when no device or more than one is connected;
# `adb logcat` would wait for a device instead.
if [[ "$(adb get-state 2>/dev/null || true)" != "device" ]]; then
  echo "::error title=No target::Connect exactly one emulator or device first." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
android_dir="$repo_root/android"
module_build="$android_dir/openmedkit/build"
results_dir="$module_build/outputs/androidTest-results/connected"
report_dir="$module_build/reports/emulator-lane/$lane"
summary="${GITHUB_STEP_SUMMARY:-/dev/null}"
packaging_test="com.openmed.openmedkit.emulator.NativeRuntimePackagingTest"

runner_args=(
  "-Pandroid.testInstrumentationRunnerArguments.openmedLane=$lane"
  "-Pandroid.testInstrumentationRunnerArguments.openmedExpectedPageSizeKb=$page_size_kb"
)
if [[ -n "$known_unloadable" ]]; then
  runner_args+=(
    "-Pandroid.testInstrumentationRunnerArguments.openmedKnownUnloadableNativeLibraries=$known_unloadable"
  )
fi

rm -rf "$report_dir" "$results_dir"
mkdir -p "$report_dir"

collect_records() {
  # Raw logcat lines from the lane tag are single JSON objects.
  adb logcat -d -v raw -s OpenMedEmulatorLane:I | grep '^{' > "$report_dir/$1" || true
}

# A larger ring buffer keeps early records while the whole suite runs.
adb logcat -G 16M || true
adb logcat -c || true
lane_status=0
(cd "$android_dir" && ./gradlew :openmedkit:connectedDebugAndroidTest "${runner_args[@]}" --stacktrace) \
  || lane_status=$?
collect_records records.jsonl
if [[ -d "$results_dir" ]]; then
  cp -R "$results_dir" "$report_dir/results"
fi

passed="$(grep -c '"outcome":"pass"' "$report_dir/records.jsonl" || true)"
failed="$(grep -c '"outcome":"fail"' "$report_dir/records.jsonl" || true)"
{
  echo "### OpenMedKit emulator lane \`$lane\`"
  echo
  echo "- Environment: \`$(grep '"record":"environment"' "$report_dir/records.jsonl" | head -n 1)\`"
  echo "- Case checks passed: ${passed}; failed: ${failed}"
  grep '"record":"native_library"' "$report_dir/records.jsonl" | sed 's/^/- Native library: `/; s/$/`/' || true
} >> "$summary"

if ((lane_status != 0)); then
  echo "::error title=Emulator lane failed::Instrumented tests failed on lane $lane."
  exit "$lane_status"
fi

if [[ "$negative_control" == true ]]; then
  rm -rf "$results_dir"
  fault_status=0
  (cd "$android_dir" && ./gradlew :openmedkit:connectedDebugAndroidTest \
    -PopenmedNativePackagingFault=missing-native-libraries \
    "-Pandroid.testInstrumentationRunnerArguments.class=$packaging_test" \
    "${runner_args[@]}") || fault_status=$?
  if [[ -d "$results_dir" ]]; then
    cp -R "$results_dir" "$report_dir/negative-control-results"
  fi
  if ((fault_status == 0)); then
    echo "::error title=Negative control passed::The lane accepted an APK without native libraries."
    exit 1
  fi
  for library in onnxruntime djl_tokenizer; do
    if ! grep -R -q -- "native_library_unloadable:$library" "$results_dir"; then
      echo "::error title=Negative control failed for another reason::Expected native_library_unloadable:$library."
      exit 1
    fi
  done
  echo "- Negative control (APK without native libraries): failed as expected with \`native_library_unloadable\`" >> "$summary"
  echo "Negative control failed as expected: native_library_unloadable for onnxruntime and djl_tokenizer."
fi
