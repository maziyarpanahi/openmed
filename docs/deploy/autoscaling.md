# Kubernetes Autoscaling

OpenMed exposes two label-free gauges for scaling model-backed service work:

- `openmed_service_scaling_queue_depth`: pending requests across the pod's
  dynamic batch queues
- `openmed_service_scaling_inflight_requests`: active requests on model-backed
  routes

Both are emitted by the opt-in [`GET /metrics`](../rest-service.md#prometheus-metrics)
surface. They contain no request, client, model, document, or PHI labels.
Prometheus adds the Kubernetes `namespace` and `pod` resource labels while
scraping; the application does not export them itself.

The sample
[`deploy/k8s/hpa.yaml`](https://github.com/maziyarpanahi/openmed/blob/master/deploy/k8s/hpa.yaml)
targets an `openmed-service` Deployment with 2 to 20 replicas. It asks
Kubernetes to keep average queue depth at 8 per pod, average in-flight work at 4
per pod, and average CPU utilization at 60%. The HPA uses the largest
recommendation, so CPU remains an independent scaling floor when the custom
metrics are quiet.

## Prerequisites

The manifest is intentionally cluster-neutral. Supply these components in the
cluster rather than bundling a cloud-provider-specific metrics service:

1. Kubernetes resource metrics for the CPU target.
2. Prometheus scraping OpenMed pods.
3. A Prometheus Adapter serving `custom.metrics.k8s.io`.

Enable metrics and batching in the OpenMed Helm release. The HPA CPU target also
requires a CPU request, which the chart sets by default:

```bash
helm upgrade --install openmed-service deploy/helm/openmed-service \
  --namespace openmed \
  --create-namespace \
  --set config.metrics.enabled=true \
  --set config.batching.enabled=true \
  --set replicaCount=2
```

## Scrape the service

For Prometheus Operator, this `ServiceMonitor` selects the labels produced by
the release command above. Store it in your cluster configuration and apply it
in the same namespace:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: openmed-service
  namespace: openmed
spec:
  namespaceSelector:
    matchNames:
      - openmed
  selector:
    matchLabels:
      app.kubernetes.io/instance: openmed-service
      app.kubernetes.io/name: openmed-service
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

If Prometheus is configured without the Operator, use Kubernetes pod discovery
and retain `namespace` and `pod` as target labels. The adapter rules below need
both labels to associate each sample with a Kubernetes Pod.

## Configure Prometheus Adapter

Merge these entries into the Prometheus Adapter chart's `rules.custom` values.
The queries preserve the adapter's requested pod grouping and expose the exact
metric names referenced by the HPA:

```yaml
rules:
  custom:
    - seriesQuery: >-
        openmed_service_scaling_queue_depth{namespace!="",pod!=""}
      resources:
        overrides:
          namespace:
            resource: namespace
          pod:
            resource: pod
      name:
        matches: ^openmed_service_scaling_queue_depth$
        as: openmed_service_scaling_queue_depth
      metricsQuery: >-
        sum(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)
    - seriesQuery: >-
        openmed_service_scaling_inflight_requests{namespace!="",pod!=""}
      resources:
        overrides:
          namespace:
            resource: namespace
          pod:
            resource: pod
      name:
        matches: ^openmed_service_scaling_inflight_requests$
        as: openmed_service_scaling_inflight_requests
      metricsQuery: >-
        sum(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)
```

After the adapter reloads, verify both Pod metrics before applying the HPA:

```bash
kubectl get --raw \
  '/apis/custom.metrics.k8s.io/v1beta2/namespaces/openmed/pods/%2A/openmed_service_scaling_queue_depth'
kubectl get --raw \
  '/apis/custom.metrics.k8s.io/v1beta2/namespaces/openmed/pods/%2A/openmed_service_scaling_inflight_requests'
kubectl apply --namespace openmed --filename deploy/k8s/hpa.yaml
kubectl describe hpa openmed-service --namespace openmed
```

If either custom metric is missing, check the OpenMed metrics flag, Prometheus
target health, retained target labels, and adapter discovery before tuning the
HPA. Kubernetes avoids a scale-down when one of the configured metrics cannot
be read.

## Reproduce the replica mapping

For a Pods `AverageValue` target, the raw recommendation is equivalent to
`ceil(total metric value / target average)`. CPU uses
`ceil(current replicas * current average CPU / target CPU)`. The HPA selects the
largest signal, then enforces the 2-to-20 replica bounds.

| Load shape | Current replicas | Queue | In flight | CPU | Queue recommendation | In-flight recommendation | CPU recommendation | Desired replicas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Idle | 2 | 0 | 2 | 25% | 0 | 1 | 1 | 2 |
| Steady | 2 | 12 | 9 | 45% | 2 | 3 | 2 | 3 |
| Burst | 3 | 37 | 18 | 75% | 5 | 5 | 4 | 5 |
| Queue-bound | 5 | 96 | 28 | 85% | 12 | 7 | 8 | 12 |
| CPU-bound | 4 | 4 | 3 | 95% | 1 | 1 | 7 | 7 |

Run this dependency-free snippet to reproduce the table:

```python
from math import ceil

cases = [
    ("Idle", 2, 0, 2, 25),
    ("Steady", 2, 12, 9, 45),
    ("Burst", 3, 37, 18, 75),
    ("Queue-bound", 5, 96, 28, 85),
    ("CPU-bound", 4, 4, 3, 95),
]

for name, current, queue, inflight, cpu in cases:
    signals = (
        ceil(queue / 8),
        ceil(inflight / 4),
        ceil(current * cpu / 60),
    )
    desired = max(2, min(20, max(signals)))
    print(name, *signals, desired)
```

These are raw recommendations. Kubernetes tolerance, readiness filtering, and
scale-down stabilization can delay an actual replica change. Tune targets from
representative synthetic traffic and service-level objectives; never add
request-derived or PHI-bearing labels to make the metrics more granular.
