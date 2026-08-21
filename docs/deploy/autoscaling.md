# Kubernetes autoscaling

OpenMed exposes aggregate queue and request gauges that a Kubernetes
HorizontalPodAutoscaler (HPA) can use as backpressure signals. The reference
manifest combines those custom metrics with CPU utilization, keeps two warm
replicas, scales up quickly, and waits five minutes before scaling down.

The signals contain counts and fixed queue labels only. They never contain
request text, entities, model output, client identity, or other PHI.

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
than CPU alone when a model or accelerator becomes the bottleneck.

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
label values.
