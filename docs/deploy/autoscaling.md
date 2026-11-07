# Kubernetes autoscaling

OpenMed exposes aggregate queue and request gauges that a Kubernetes
HorizontalPodAutoscaler (HPA) can use as backpressure signals. The reference
manifest combines those custom metrics with CPU utilization, keeps two warm
replicas, scales up quickly, and waits five minutes before scaling down.

The signals contain counts and a bounded queue label only: `analyze`,
`pii_extract`, `batch`, or `other`. Unknown queue names collapse to `other`.
They never contain request text, entities, model output, client identity, or
other PHI.

## Enable and scrape metrics

Enable the pull-only metrics endpoint in the service deployment:

```yaml
env:
  - name: OPENMED_SERVICE_METRICS_ENABLED
    value: "true"
```

Configure Prometheus to scrape each OpenMed pod on `/metrics`. Confirm these
series are present before applying the HPA:

- `openmed_service_admission_queue_depth{queue="analyze"}` and
  `openmed_service_admission_queue_depth{queue="pii_extract"}`
- `openmed_service_inflight_requests`

The queue gauge tracks admitted work that has not completed. The in-flight
gauge tracks active HTTP requests. Both are useful earlier saturation signals
than CPU alone when a model or accelerator becomes the bottleneck. A scrape of
`/metrics` is excluded from the in-flight gauge so monitoring does not create a
scaling feedback signal.

## Wire prometheus-adapter

The cluster needs a custom-metrics adapter such as `prometheus-adapter`. Add
rules equivalent to the following to its configuration. Prometheus scrape
discovery must attach `namespace` and `pod` labels to each series.

```yaml
rules:
  custom:
    - seriesQuery: 'openmed_service_admission_queue_depth{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: namespace}
          pod: {resource: pod}
      name:
        matches: '^openmed_service_admission_queue_depth$'
        as: openmed_service_admission_queue_depth
      metricsQuery: 'sum(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
    - seriesQuery: 'openmed_service_inflight_requests{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: namespace}
          pod: {resource: pod}
      name:
        matches: '^openmed_service_inflight_requests$'
        as: openmed_service_inflight_requests
      metricsQuery: 'sum(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
```

Verify the custom metrics API before enabling automatic scaling:

```bash
kubectl get --raw \
  '/apis/custom.metrics.k8s.io/v1beta1/namespaces/default/pods/*/openmed_service_admission_queue_depth'
kubectl get --raw \
  '/apis/custom.metrics.k8s.io/v1beta1/namespaces/default/pods/*/openmed_service_inflight_requests'
```

## Apply the reference HPA

The checked-in manifest targets a Deployment named `openmed-service`. Change
`spec.scaleTargetRef.name` when the deployed name differs, including when a
Helm release prefixes the chart fullname.

The HPA keeps at least two replicas, so the service must run in multi-replica
mode. When the Deployment comes from the Helm chart, set
`autoscaling.enabled=true` and choose how each replica-local component behaves
before applying the HPA. See
[Replica-local state](#replica-local-state-and-safe-topologies).

```bash
kubectl apply -f deploy/k8s/hpa.yaml
kubectl describe hpa openmed-service
```

The HPA uses these starting targets:

| Signal | Per-pod target |
| --- | ---: |
| Admitted queue depth | 4 |
| In-flight requests | 8 |
| CPU utilization | 65% |

Kubernetes calculates a desired replica count for each signal and selects the
largest. CPU remains a general load signal when queueing is low, while either
custom metric can request an earlier scale-up under backpressure. If the custom
metrics API is unavailable, investigate the adapter; do not treat missing data
as a zero queue.

## Reproduce a threshold-to-replicas mapping

For aggregate queue depth `Q` and in-flight requests `F`, the reference targets
map load to `ceil(max(queue_depth / 4, in_flight / 8, 2))`, capped at 10
replicas. The dependency-free helper mirrors this custom-metric calculation:

```python
from openmed.service.scaling_metrics import recommend_replicas

decision = recommend_replicas(queue_depth=9, inflight_requests=8)
assert decision.recommended_replicas == 3
```

| Queue depth | In flight | Queue replicas | In-flight replicas | Final replicas |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 2 |
| 8 | 8 | 2 | 1 | 2 |
| 9 | 8 | 3 | 1 | 3 |
| 4 | 33 | 1 | 5 | 5 |
| 1000 | 1000 | 250 | 125 | 10 |

Tune targets from synthetic load tests, keep resource requests accurate for
the CPU metric, and use a disruption budget when maintaining more than one
replica. Do not expose `/metrics` outside the cluster or add user-controlled
label values. The helper and table show the raw custom-metric recommendation;
the Kubernetes controller also considers CPU, its tolerance, missing or
not-yet-ready pods, and the stabilization policies in the manifest.

## Replica-local state and safe topologies

Some service components keep state in one process or on one node's disk. One
replica is the supported topology for that state. Behind several replicas, a
status read can reach a replica that never saw the job, and every limit is
enforced once per replica. The service cannot count its own replicas, so the
deployment declares the topology:

| Setting | Values | Meaning |
| --- | --- | --- |
| `OPENMED_SERVICE_REPLICA_MODE` | `single` (default), `multi` | Declared replica topology. |
| `OPENMED_SERVICE_REPLICA_ACCEPT_LOCAL_STATE` | Comma-separated categories | Accept per-replica semantics for these components. |
| `OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE` | `async_jobs`, `fhir_bulk_jobs`, `smart_backend_jobs` | Turn these job components off. |

Unknown categories, an unsupported mode, a limiter in the disable list, or a
category that is both accepted and disabled stop the service at startup.

The Helm chart renders `multi` when `replicaCount` is greater than 1 or
`autoscaling.enabled` is true, and `single` otherwise. It renders
`replicaState.accept` and `replicaState.disable` into the two lists. The Compose
bundles publish one fixed host port and set `single`.

### Inventory

| Category | Scope | Multi-replica safety | Enabled when |
| --- | --- | --- | --- |
| `async_jobs` | `node` | `unsafe` | Always, unless disabled |
| `fhir_bulk_jobs` | `process` | `unsafe` | Always, unless disabled |
| `smart_backend_jobs` | `process` | `unsafe` | Always, unless disabled |
| `rate_limiter` | `process` | `per_replica` | `OPENMED_SERVICE_RATE_LIMIT_RPS` and `OPENMED_SERVICE_RATE_LIMIT_BURST` are above 0 |
| `concurrency_limiter` | `process` | `per_replica` | `OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY` is above 0 |
| `auth_failure_limiter` | `process` | `per_replica` | Authentication is on and its failure limit is above 0 |
| `request_coalescer` | `process` | `safe` | `OPENMED_SERVICE_COALESCING_ENABLED` is true |

`async_jobs` covers the job metadata file and the in-memory work queue.
Unsafe components hold state that later requests must read back. Per-replica
components work inside each replica, but the effective limit is the configured
limit times the replica count. The request coalescer only merges identical
concurrent requests that reach the same replica, so it never blocks readiness.
No component has `shared` scope yet; a shared backend would report it.

### Readiness and route behavior

In `single` mode nothing changes. In `multi` mode, `/readyz` returns `503`
with `error.code` set to `replica_unsafe_state` while any enabled component
that is not safe is neither accepted nor disabled. `/livez` stays `200`, so
Kubernetes holds traffic back without restarting the pod. The envelope names
categories only:

```json
{
  "error": {
    "code": "replica_unsafe_state",
    "message": "Replica-local service state is not safe behind multiple replicas",
    "details": {"replica_mode": "multi", "components": ["async_jobs"]}
  }
}
```

The `/jobs`, `/fhir/bulk/...`, and `/fhir/smart-backend/ingestions` routes of
an unaccepted job component also return `503 replica_unsafe_state`, with
`details` naming the component, so a request sent straight to a pod is refused
too. A disabled job component returns `503 component_disabled` in every mode and
never loads the local job store. Authentication still runs first.

### Choosing a topology

- **One replica.** The Helm default and the Compose bundles. Every component is
  available.
- **Several replicas for stateless inference.** Disable the three job
  components. Accept the limiters you configure, or turn them off. For an
  aggregate rate of `R` requests per second across `N` replicas, configure
  about `R / N` per replica and remember that a client pinned to one replica
  gets only that share.
- **Separate releases.** Run autoscaled inference in one release with the job
  components disabled, and run jobs, Bulk Data, and SMART ingestion in a
  second single-replica release.
- **Accepted job components.** Accept `async_jobs`, `fhir_bulk_jobs`, or
  `smart_backend_jobs` only when routing pins every client to one pod for the
  whole job. Status reads on another replica return `404`, and scale-down or a
  restart loses queued work and in-memory job status. This is a documented
  risk, not a supported shared backend.

Helm values for the reference HPA with stateless inference only:

```yaml
autoscaling:
  enabled: true
replicaState:
  accept:
    - concurrency_limiter
  disable:
    - async_jobs
    - fhir_bulk_jobs
    - smart_backend_jobs
config:
  throttle:
    maxConcurrency: 4
```

### Value-free report

Readiness uses the same report that preflight tooling can read before the
service starts:

```python
from openmed.service.app import create_app, replica_state_report

report = replica_state_report(create_app())
print(report.to_dict())
```

The report has a versioned `schema`, the `replica_mode`, `ready`,
`reason_code`, and one entry per category with its `scope`, `safety`, and
`state` (`disabled`, `safe`, `accepted`, or `unsafe`). It never contains paths,
hosts, job or client identifiers, request content, or configuration values.
