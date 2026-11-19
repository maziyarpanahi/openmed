# Sustained resource recovery

`run_resource_soak` extends the existing local ASGI load-test request seam with
bounded cycles, periodic lifecycle actions and independent resource probes.
Unlike throughput measurements, successful requests do not excuse unrecovered
permits, leases, handles, queued jobs or memory growth.

Supply a `SoakProfile`, an aggregate `ResourceSnapshot` probe and an asynchronous
`lifecycle(action, phase)` adapter. Actions cycle through load, cancellation,
retry, drain and model swap. Phases are `before` and `after`; implement them using
the actual runtime's lifecycle seams. The harness itself cancels one request per
group during cancellation cycles, and uses the existing ASGI seam to require
complete streamed responses. Drain-cycle HTTP 503 responses are counted separately
as expected admission refusals. Other request failures, unplanned cancellation,
worker timeouts, lifecycle/cleanup failures and resource residue fail the profile.

The default CI profile is ten cycles, eight requests per cycle, concurrency four,
a one-second operation deadline and a sixty-second overall duration limit. It
requires all post-cycle resource counters to return to their initial baseline.
Set explicit tolerances for a runtime with justified residual allocations. Counts
are bounded, total work is at most 100,000 requests and concurrency at most 64.
Report elapsed duration, completed-request throughput and post-cycle resource
peaks separately. The profile and tolerance values are bound to a SHA-256 digest.

Fixtures use actual `RequestWork`/`ConcurrencyGate` ownership, `ResilienceManager`
retry behavior and `WarmPool` request/model disposal with synthetic runtime
doubles. Deliberate resource leaks, stuck workers and unplanned cancellation
fail independently of throughput. Synthetic clocks or probes are declared
evidence, not a measurement of device memory or file descriptors. Set
`evidence="real_device"` only when supplying genuine measurement adapters and
retain those device measurements separately. No real-model benchmark is bundled.

This is an in-process trusted-runtime harness. Adapters must cooperate with
cancellation, and synchronous probe calls must be bounded by their implementation.
It cannot terminate a hostile thread or a coroutine suppressing cancellation;
use process isolation for such workloads. It opens no network connection and
includes no payloads, model names, source paths or callback exception messages in
its aggregate report.

```sh
.venv/bin/python -m pytest tests/unit/eval/test_resource_soak.py -q
```
