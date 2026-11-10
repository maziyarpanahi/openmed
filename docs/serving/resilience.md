# Serving Resilience

OpenMed REST model-backed endpoints guard model load and inference work with
bounded retry and an in-process circuit breaker. The protection applies to
`POST /analyze`, `POST /pii/extract`, and `POST /pii/deidentify`.

The circuit breaker is keyed per resolved model/backend inside the service
process. State is not shared across workers or machines. Metrics expose only
aggregate breaker counts, never model names, request text, entities, or other
PHI-derived values.

## Retry Policy

Retries are enabled by default. A failing model load or inference operation is
retried up to `OPENMED_SERVICE_RETRY_MAX_ATTEMPTS` with exponential backoff and
jitter:

```bash
OPENMED_SERVICE_RETRY_MAX_ATTEMPTS=3
OPENMED_SERVICE_RETRY_BACKOFF_INITIAL_SECONDS=0.05
OPENMED_SERVICE_RETRY_BACKOFF_MULTIPLIER=2
OPENMED_SERVICE_RETRY_BACKOFF_MAX_SECONDS=1
OPENMED_SERVICE_RETRY_BACKOFF_JITTER_SECONDS=0.01
```

Set `OPENMED_SERVICE_RESILIENCE_ENABLED=false` to disable both retry and circuit
breaking for local debugging.

## Circuit Breaker

After `OPENMED_SERVICE_CIRCUIT_BREAKER_FAILURE_THRESHOLD` failed model-backed
requests for the same resolved model/backend, the breaker opens. While open,
new calls fail fast with HTTP `503`, error code `circuit_breaker_open`, and a
`Retry-After` header.

After `OPENMED_SERVICE_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS`, one half-open
probe is allowed. A successful probe closes the breaker. A failed probe opens it
again and restarts the cooldown.

```bash
OPENMED_SERVICE_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
OPENMED_SERVICE_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS=30
```

The service profile timeout still bounds blocking work. Configure it through the
active OpenMed profile; the breaker and retry policy do not remove that timeout.

## Metrics

When `OPENMED_SERVICE_METRICS_ENABLED=true`, `GET /metrics` includes aggregate
breaker gauges:

```text
openmed_service_circuit_breaker_closed 1
openmed_service_circuit_breaker_open 0
openmed_service_circuit_breaker_half_open 0
```

These gauges count in-process breakers by state. They intentionally have no
labels so they cannot leak model names, backend ids, request text, or PHI.
