#!/usr/bin/env bash
# Real MLX integration gate. No model download or remote inference is implicit.
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIRECTORY="${1:-${OPENMED_LFM_MODEL_DIRECTORY:-}}"
# A developer shell may run under Rosetta; Xcode still builds arm64 below.
if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || true)" != 1 || -z "$MODEL_DIRECTORY" || ! -d "$MODEL_DIRECTORY" ]]; then
  echo "Usage: bash scripts/test_medical_reasoning_mac.sh /absolute/path/to/pinned/model" >&2
  echo "Requires Apple silicon, Xcode, and the complete OpenMedLFM pinned artifact." >&2
  exit 2
fi
MODEL_DIRECTORY="$(cd "$MODEL_DIRECTORY" && pwd)"
for file in chat_template.jinja config.json generation_config.json model.safetensors model.safetensors.index.json tokenizer.json tokenizer_config.json; do
  if [[ ! -s "$MODEL_DIRECTORY/$file" ]]; then
    echo "Missing model file: $file" >&2
    exit 2
  fi
done

RESULT_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/openmed-medical-reasoning-tests.XXXXXX")"
MAC_DERIVED_DATA="${OPENMED_MAC_DERIVED_DATA:-${TMPDIR:-/tmp}/openmed-medical-reasoning-derived}"
PACKAGE_FLAGS=(-skipPackageUpdates)
if [[ -n "${OPENMED_SOURCE_PACKAGES:-}" ]]; then
  PACKAGE_FLAGS+=(-clonedSourcePackagesDirPath "$OPENMED_SOURCE_PACKAGES")
fi
echo "Real-model test results: $RESULT_DIRECTORY"
trap 'echo "Logs and XCTest bundles retained at: $RESULT_DIRECTORY"' EXIT

assert_real_tests() {
  local bundle="$1" minimum="$2" summary="$3"
  xcrun xcresulttool get test-results summary --path "$bundle" --format json > "$summary"
  local result passed failed skipped
  result="$(plutil -extract result raw -o - "$summary")"
  passed="$(plutil -extract passedTests raw -o - "$summary")"
  failed="$(plutil -extract failedTests raw -o - "$summary")"
  skipped="$(plutil -extract skippedTests raw -o - "$summary")"
  if [[ "$result" != Passed || "$failed" != 0 || "$skipped" != 0 || "$passed" -lt "$minimum" ]]; then
    echo "Real-model gate failed: $passed passed, $failed failed, $skipped skipped ($result)." >&2
    return 1
  fi
  echo "Verified: $passed tests passed, zero failures, zero skips."
}

(
  cd "$TASK_ROOT/swift/OpenMedKit"
  TEST_RUNNER_OPENMED_LFM_TOKENIZER_ARTIFACT="$MODEL_DIRECTORY" \
    xcodebuild test -scheme OpenMedKit -destination 'platform=macOS,arch=arm64' \
    -parallel-testing-enabled NO -test-timeouts-enabled YES \
    -maximum-test-execution-time-allowance 120 \
    -only-testing:OpenMedKitTests/OpenMedLFMTests \
    -resultBundlePath "$RESULT_DIRECTORY/package.xcresult" -quiet \
    > "$RESULT_DIRECTORY/package.log" 2>&1
)
assert_real_tests "$RESULT_DIRECTORY/package.xcresult" 9 "$RESULT_DIRECTORY/package-summary.json"

TEST_RUNNER_OPENMED_LFM_MODEL_DIRECTORY="$MODEL_DIRECTORY" \
  xcodebuild test \
  -project "$TASK_ROOT/swift/OpenMedMedicalReasoningDemo/OpenMedMedicalReasoningDemo.xcodeproj" \
  -scheme OpenMedMedicalReasoningMac -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$MAC_DERIVED_DATA" "${PACKAGE_FLAGS[@]}" \
  -parallel-testing-enabled NO -test-timeouts-enabled YES \
  -maximum-test-execution-time-allowance 120 \
  -resultBundlePath "$RESULT_DIRECTORY/demo.xcresult" -quiet \
  > "$RESULT_DIRECTORY/demo.log" 2>&1
assert_real_tests "$RESULT_DIRECTORY/demo.xcresult" 5 "$RESULT_DIRECTORY/demo-summary.json"

echo "OpenMedKit and the shared SwiftUI demo passed native MLX integration."
