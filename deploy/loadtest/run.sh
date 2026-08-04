#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKER_BIN="${DOCKER_BIN-docker}"
K6_BIN="${K6_BIN-k6}"
SERVICE_IMAGE="${LOADTEST_SERVICE_IMAGE-openmed:loadtest}"
SERVICE_PORT="${LOADTEST_SERVICE_PORT-18080}"
CONTAINER_NAME="${LOADTEST_CONTAINER_NAME-openmed-loadtest-${RANDOM}-${BASHPID}}"
BASE_URL="${LOADTEST_BASE_URL-http://127.0.0.1:${SERVICE_PORT}}"
RESULT_FILE="${LOADTEST_RESULT_FILE-${RUNNER_TEMP-${TMPDIR-/tmp}}/openmed-loadtest/summary.json}"
STARTUP_TIMEOUT="${LOADTEST_STARTUP_TIMEOUT_SECONDS-900}"
WARMUP="${LOADTEST_WARMUP-1}"
SERVICE_PROFILE="${OPENMED_PROFILE-prod}"
PRELOAD_MODELS="${OPENMED_SERVICE_PRELOAD_MODELS-disease_detection_superclinical,OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1}"
KEEP_ALIVE="${OPENMED_SERVICE_KEEP_ALIVE-10m}"
CONTAINER_STARTED=0

die() {
  echo "loadtest: $*" >&2
  exit 1
}

cleanup() {
  if [[ "$CONTAINER_STARTED" == "1" ]]; then
    "$DOCKER_BIN" rm --force "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

command -v "$DOCKER_BIN" >/dev/null 2>&1 || die "Docker CLI is required"
command -v "$K6_BIN" >/dev/null 2>&1 || die "k6 is required"
command -v curl >/dev/null 2>&1 || die "curl is required"

if [[ ! "$SERVICE_PORT" =~ ^[0-9]+$ ]] || (( SERVICE_PORT < 1024 || SERVICE_PORT > 65535 )); then
  die "LOADTEST_SERVICE_PORT must be an unprivileged port between 1024 and 65535"
fi
if [[ ! "$STARTUP_TIMEOUT" =~ ^[0-9]+$ ]] || (( STARTUP_TIMEOUT < 1 )); then
  die "LOADTEST_STARTUP_TIMEOUT_SECONDS must be a positive integer"
fi

BASE_URL="${BASE_URL%/}"
case "$BASE_URL" in
  http://127.0.0.1:*|http://localhost:*) ;;
  *) die "LOADTEST_BASE_URL must point to a local loopback service" ;;
esac

if [[ "$RESULT_FILE" != /* ]]; then
  RESULT_FILE="$ROOT_DIR/$RESULT_FILE"
fi
mkdir -p "$(dirname -- "$RESULT_FILE")"

if [[ "${LOADTEST_SKIP_BUILD-0}" != "1" ]]; then
  "$DOCKER_BIN" build \
    --file "$ROOT_DIR/deploy/docker/Dockerfile" \
    --tag "$SERVICE_IMAGE" \
    "$ROOT_DIR"
fi

container_args=(
  run
  --detach
  --rm
  --name "$CONTAINER_NAME"
  --publish "127.0.0.1:${SERVICE_PORT}:8080"
  --env "OPENMED_PROFILE=$SERVICE_PROFILE"
  --env "OPENMED_SERVICE_PRELOAD_MODELS=$PRELOAD_MODELS"
  --env "OPENMED_SERVICE_KEEP_ALIVE=$KEEP_ALIVE"
  --env HF_HOME=/root/.cache/huggingface
)

if [[ -n "${OPENMED_OFFLINE+x}" ]]; then
  container_args+=(--env "OPENMED_OFFLINE=${OPENMED_OFFLINE-}")
fi

container_args+=("$SERVICE_IMAGE")

"$DOCKER_BIN" "${container_args[@]}" >/dev/null
CONTAINER_STARTED=1

wait_for_ready() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl --fail --silent --show-error --max-time 5 \
      "$BASE_URL/readyz" >/dev/null; then
      return 0
    fi
    sleep 2
  done

  echo "loadtest: service did not become ready within ${STARTUP_TIMEOUT}s" >&2
  "$DOCKER_BIN" logs --tail 80 "$CONTAINER_NAME" >&2 2>/dev/null || true
  return 1
}

wait_for_ready

if [[ "$WARMUP" == "1" ]]; then
  synthetic_payload='{"text":"Taylor Reed, born 1981-02-03, visited Example Clinic for a routine follow-up. Call the fictional records desk at 555-0100.","method":"mask","model_name":"OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1","confidence_threshold":0}'
  stream_payload='{"text":"Taylor Reed, born 1981-02-03, visited Example Clinic for a routine follow-up. Call the fictional records desk at 555-0100.","confidence_threshold":0,"chunk_size":128,"window_chars":256,"tokenizer_context_chars":64,"max_entity_chars":128,"include_text":false}'

  curl --fail --silent --show-error --max-time 300 \
    -H 'Content-Type: application/json' \
    --data "{\"text\":\"Taylor Reed, born 1981-02-03, visited Example Clinic for a routine follow-up. Call the fictional records desk at 555-0100.\",\"model_name\":\"disease_detection_superclinical\",\"confidence_threshold\":0}" \
    "$BASE_URL/analyze" >/dev/null
  curl --fail --silent --show-error --max-time 300 \
    -H 'Content-Type: application/json' \
    --data "$synthetic_payload" \
    "$BASE_URL/pii/deidentify" >/dev/null
  curl --fail --silent --show-error --max-time 300 \
    -H 'Content-Type: application/json' \
    --data "$stream_payload" \
    "$BASE_URL/pii/extract/stream" >/dev/null
fi

export BASE_URL
export LOADTEST_RESULT_FILE="$RESULT_FILE"
"$K6_BIN" run "$ROOT_DIR/deploy/loadtest/scenario.js"

echo "loadtest: report written to $RESULT_FILE"
