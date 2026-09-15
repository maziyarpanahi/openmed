# Service Load Testing and Latency SLOs

The checked-in load-test harness exercises a local OpenMed service container
with synthetic text. It sends a deterministic mixed workload to:

- `POST /analyze` (40% of requests)
- `POST /pii/deidentify` (40% of requests)
- `POST /pii/extract/stream` (20% of requests)

The harness never reads production traffic or accepts a remote target. The
wrapper binds the temporary container to loopback and refuses a non-loopback
`LOADTEST_BASE_URL`. Do not replace the synthetic fixture with patient text,
credentials, or production URLs.

## Run locally

Install Docker, `curl`, and [k6](https://grafana.com/docs/k6/latest/). From the
repository root, run:

```bash
deploy/loadtest/run.sh
```

The wrapper builds `deploy/docker/Dockerfile`, starts an ephemeral container on
`127.0.0.1:18080`, waits for `/readyz`, warms the three routes, runs k6, and
removes the container on exit. Model downloads are performed by the service
only when the image's local cache does not already contain the preloaded
models. To use a prebuilt image or skip warmup:

```bash
LOADTEST_SERVICE_IMAGE=openmed:local \
LOADTEST_SKIP_BUILD=1 \
LOADTEST_WARMUP=0 \
deploy/loadtest/run.sh
```

To run the scenario against an already-running loopback service, bypass the
wrapper and keep the target local:

```bash
BASE_URL=http://127.0.0.1:8080 \
LOADTEST_RESULT_FILE=/tmp/openmed-loadtest-summary.json \
k6 run deploy/loadtest/scenario.js
```

## SLO gates and reports

k6 exits non-zero when any configured threshold is breached. The console and
the JSON report contain p95 latency, p99 latency, error rate, achieved
throughput, request count, and the configured limits. Set
`LOADTEST_RESULT_FILE` to choose the report location; the default is a temporary
directory so a local run does not create repository files.

| Variable | Default | Meaning |
| --- | ---: | --- |
| `LOADTEST_DURATION_SECONDS` | `30` | Test duration |
| `LOADTEST_RATE` | `2` | Target requests per second |
| `LOADTEST_CONCURRENCY` | `4` | Pre-allocated virtual users |
| `LOADTEST_MAX_VUS` | `2 × concurrency` | Upper virtual-user bound |
| `LOADTEST_SLO_P95_MS` | `30000` | Strict p95 latency ceiling |
| `LOADTEST_SLO_P99_MS` | `60000` | Strict p99 latency ceiling |
| `LOADTEST_SLO_ERROR_RATE` | `0.05` | Strict failed-request ceiling |
| `LOADTEST_SLO_MIN_THROUGHPUT_RPS` | `0.5` | Minimum achieved throughput |

The scenario also accepts the shorter `SLO_P95_MS`, `SLO_P99_MS`,
`SLO_ERROR_RATE`, and `SLO_MIN_THROUGHPUT_RPS` aliases. For example:

```bash
LOADTEST_SLO_P95_MS=10000 \
LOADTEST_SLO_P99_MS=20000 \
LOADTEST_SLO_ERROR_RATE=0.01 \
LOADTEST_SLO_MIN_THROUGHPUT_RPS=1 \
LOADTEST_RESULT_FILE=/tmp/openmed-laptop-slo.json \
deploy/loadtest/run.sh
```

Use a stable, warmed-up service and repeat a profile before changing a gate.
These starting points help tune thresholds by device tier; they are not
performance guarantees:

| Device tier | p95 | p99 | Error rate | Minimum throughput |
| --- | ---: | ---: | ---: | ---: |
| Nano / constrained CPU | 20,000 ms | 30,000 ms | 5% | 0.1 req/s |
| Phone / small laptop | 10,000 ms | 15,000 ms | 2% | 0.25 req/s |
| Laptop | 5,000 ms | 10,000 ms | 1% | 0.5 req/s |
| Server | 3,000 ms | 6,000 ms | 1% | 1 req/s |

The nightly workflow is intentionally separate from pull-request CI. It builds
the service image, runs the same loopback wrapper with synthetic input, and
uploads `loadtest-results/` as the `loadtest-slo-<run-id>` artifact, including
when the SLO gate fails. Use **Run workflow** to tune a profile explicitly;
scheduled runs use the defaults above.

## Failure interpretation

The gate measures service behavior, not model quality. A failure means at least
one of the following occurred:

- p95 or p99 request latency exceeded its configured ceiling;
- a response was not a successful 2xx response;
- achieved request throughput fell below the configured floor; or
- the service did not become ready or a warmup route failed.

Inspect the archived JSON report and container startup logs in the workflow
run. The workload is single-container and single-node by design; it is not a
production capacity test or a replacement for the unit concurrency harness.
