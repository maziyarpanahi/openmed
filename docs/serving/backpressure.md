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
python -m openmed.service.logging --host 127.0.0.1 --port 8080
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

## Client Identity and Limiter State

The request throttle (`OPENMED_SERVICE_RATE_LIMIT_*`) and the authentication
failure limiter key their token buckets by client identity.
`OPENMED_SERVICE_THROTTLE_KEY` selects `global`, `peer`, or `x-forwarded-for`.

`x-forwarded-for` trusts the header only when the direct peer is listed in
`OPENMED_SERVICE_TRUSTED_PROXIES`. The service reads entries from right to
left, skips at most `OPENMED_SERVICE_TRUSTED_PROXY_HOPS` trusted proxies (the
direct peer counts as the first), and keys the request by the first untrusted
address. Values a client places at the left of the header are never reached,
so rotating them cannot create new buckets. Malformed entries fall back to the
last trusted hop. When no trusted proxies are configured, or a request arrives
from an untrusted peer, forwarding headers are ignored and the direct peer is
used, exactly as in `peer` mode.

```bash
OPENMED_SERVICE_RATE_LIMIT_RPS=10 \
OPENMED_SERVICE_RATE_LIMIT_BURST=20 \
OPENMED_SERVICE_THROTTLE_KEY=x-forwarded-for \
OPENMED_SERVICE_TRUSTED_PROXIES=198.51.100.0/24,2001:db8:ffff::/48 \
OPENMED_SERVICE_TRUSTED_PROXY_HOPS=2 \
python -m openmed.service.logging --host 0.0.0.0 --port 8080
```

Set the hop count to the number of proxy tiers that append to
`X-Forwarded-For`, for example `2` for a cloud load balancer in front of an
ingress controller. A value that is too low keys requests by a proxy address;
a value that is too high lets a client inside a trusted network choose its
identity.

The authentication-failure limiter defaults to the same `x-forwarded-for`
resolution and the same trusted proxies, so a client that fails
authentication behind a proxy is limited on its own instead of locking out
every client behind that proxy. Set
`OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_KEY=peer` to key it by the direct
peer instead.

Each limiter tracks at most `OPENMED_SERVICE_THROTTLE_MAX_TRACKED_KEYS`
identities (default `10000`). A bucket is dropped only after it has refilled
to the full burst, so forgetting it cannot change an admission decision; a
depleted bucket is never reset. When the bound is reached and no tracked
bucket is idle, new identities share one overflow bucket until capacity frees
up. Concurrency semaphores are retained only while a request holds or waits
for a slot. Rejections and metrics report only the configured key mode and
aggregate limits, never client addresses or header values.

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `OPENMED_SERVICE_THROTTLE_KEY` | `global` | `global`, `peer`, or `x-forwarded-for`. |
| `OPENMED_SERVICE_TRUSTED_PROXIES` | empty | Comma-separated proxy addresses or CIDR networks whose `X-Forwarded-For` entries are honored. |
| `OPENMED_SERVICE_TRUSTED_PROXY_HOPS` | `1` | Maximum trusted proxies skipped, counting the direct peer. |
| `OPENMED_SERVICE_THROTTLE_MAX_TRACKED_KEYS` | `10000` | Per-limiter bound on tracked client identities. |

When Uvicorn's own proxy-header handling rewrites the peer address
(`--forwarded-allow-ips`), configure only one of the two mechanisms so the
header is not interpreted twice.

## Tuning

Start with a high watermark no larger than the request volume that one replica
can finish within its latency SLO. Set the low watermark between one-third and
one-half of the high watermark, then load-test with production-like document
sizes. Reduce the high watermark or maximum queue wait if p99 latency rises
before shedding begins. Clients should honor `Retry-After` and retry with
bounded exponential backoff and jitter.
