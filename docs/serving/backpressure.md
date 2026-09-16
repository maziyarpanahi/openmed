# Serving Backpressure

OpenMed can put a bounded, in-process admission queue in front of each dynamic
batching path. Admission control protects tail latency and memory during bursts
by rejecting excess work while already-admitted requests continue to run.

Backpressure is active only when dynamic batching is enabled:

```bash
OPENMED_SERVICE_BATCHING_ENABLED=true \
OPENMED_SERVICE_BATCH_HIGH_WATERMARK=256 \
OPENMED_SERVICE_BATCH_LOW_WATERMARK=128 \
OPENMED_SERVICE_BATCH_MAX_QUEUE_WAIT_MS=1000 \
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

The admission depth counts both requests waiting for batch dispatch and batches
currently executing. When depth reaches the high watermark, the queue enters
shedding mode. It stays in that mode until outstanding depth drains to the low
watermark; this hysteresis prevents rapid accept/reject oscillation near the
limit. The `/analyze` and `/pii/extract` paths have independent queues.

Requests rejected by the high watermark receive `503` with error code
`backpressure` and a `Retry-After` header. A request that remains queued longer
than `OPENMED_SERVICE_BATCH_MAX_QUEUE_WAIT_MS` receives the same status and
header with `error.details.reason` set to `max_wait`. Error responses contain
only aggregate queue state and configuration, never request text or other PHI.

## Configuration

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `OPENMED_SERVICE_BATCH_HIGH_WATERMARK` | `OPENMED_SERVICE_BATCH_MAX_QUEUE_SIZE` (default `256`) | Maximum outstanding requests admitted to one batching path. |
| `OPENMED_SERVICE_BATCH_LOW_WATERMARK` | Half of the high watermark (default `128`) | Outstanding depth at or below which shedding stops. Must be lower than the high watermark. |
| `OPENMED_SERVICE_BATCH_MAX_QUEUE_WAIT_MS` | `1000` | Maximum pre-dispatch wait. `0` allows only work that immediately fills and dispatches a batch. |
| `OPENMED_SERVICE_BATCH_MAX_QUEUE_SIZE` | `256` | Hard capacity for each priority-specific pending queue. |
| `OPENMED_SERVICE_BATCH_MAX_WAIT_MS` | `5` | Batch-formation window; this is separate from the maximum queue wait. |

The high watermark is aggregate across priority classes. The existing
per-priority hard capacity remains a final safety bound, so operators should
normally keep `BATCH_MAX_QUEUE_SIZE` at or above the expected per-class share
of the high watermark.

## Metrics

Enable the pull-only metrics endpoint with
`OPENMED_SERVICE_METRICS_ENABLED=true`. Admission control exports only static
queue names and aggregate values:

- `openmed_service_admission_queue_depth{queue=...}`: outstanding admitted work.
- `openmed_service_admission_queue_shedding{queue=...}`: `1` while shedding,
  otherwise `0`.
- `openmed_service_admission_queue_wait_seconds{queue=...}`: latest observed
  pre-dispatch wait.
- `openmed_service_admission_shed_total{queue=...}`: cumulative high-watermark
  rejections and max-wait expirations.

The existing `openmed_service_batch_queue_*` and
`openmed_service_batch_shed_total` families remain available for per-priority
pending-queue visibility.

## Tuning

Start with a high watermark no larger than the request volume that one replica
can finish within its latency SLO. Set the low watermark between one-third and
one-half of the high watermark, then load-test with production-like document
sizes. Reduce the high watermark or maximum queue wait if p99 latency rises
before shedding begins. Clients should honor `Retry-After` and retry with
bounded exponential backoff and jitter.
