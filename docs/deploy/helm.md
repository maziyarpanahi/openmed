# Helm Deployment

The `deploy/helm/openmed-service` chart deploys the OpenMed de-identification
REST service on Kubernetes with separate liveness and readiness probes, a
persistent model cache volume, non-root read-only security contexts, and
configurable service runtime settings.
Its optional production Journey profile adds a pre-install/pre-upgrade
migration gate, a worker Deployment, durable artifact storage, dependency-aware
readiness, and a default-deny NetworkPolicy.

## Install

Build and publish an OpenMed service image to a registry your cluster can pull,
then install the chart with the matching image repository and tag:

```bash
helm upgrade --install openmed-service deploy/helm/openmed-service \
  --namespace openmed \
  --create-namespace \
  --set image.repository=ghcr.io/maziyarpanahi/openmed \
  --set image.tag=v3.2.0
```

The default chart creates:

- a `Deployment` running the REST service on port `8080`
- a `Service` exposing the HTTP port inside the cluster
- a `ConfigMap` for non-secret OpenMed service settings
- a `PersistentVolumeClaim` mounted at `/home/nonroot/.cache`

The service runs as the non-root image user, so its cache directory is
`/home/nonroot/.cache`. That directory holds the OpenMed model cache
(`~/.cache/openmed`), the service job store, and the Hugging Face cache
(`HF_HOME=/home/nonroot/.cache/huggingface`), so downloads survive pod restarts
when the PVC is retained.

## Security context

Every API pod runs hardened by default, with or without the Journey profile:

| Setting | Default |
| --- | --- |
| `podSecurityContext.runAsNonRoot` / `securityContext.runAsNonRoot` | `true` |
| `runAsUser` / `runAsGroup` (pod and container) | `65532` |
| `podSecurityContext.fsGroup` | `65532`, with `fsGroupChangePolicy: OnRootMismatch` |
| `securityContext.readOnlyRootFilesystem` | `true` |
| `securityContext.allowPrivilegeEscalation` | `false` |
| `securityContext.privileged` | `false` |
| `securityContext.capabilities.drop` | `[ALL]` |
| `seccompProfile.type` (pod and container) | `RuntimeDefault` |

UID/GID `65532` is the `nonroot` user baked into the published service image
(`deploy/docker/Dockerfile`), the distroless image, and the Compose profile.
Because the root filesystem is read-only, the API container writes only to
mounted volumes:

| Path | Volume | Purpose |
| --- | --- | --- |
| `/home/nonroot` | `home` emptyDir (`writableVolumes.home.sizeLimit`, `128Mi`) | `HOME` for small library config files. |
| `/tmp` | `tmp` emptyDir (`writableVolumes.tmp.sizeLimit`, `512Mi`) | Scratch space exported as `TMPDIR`. |
| `/home/nonroot/.cache` | `model-cache` PVC, or an emptyDir (`writableVolumes.modelCache.sizeLimit`, `20Gi`) when `persistence.enabled=false` | OpenMed and Hugging Face caches. |
| `journey.artifacts.mountPath` | `journey-artifacts` PVC | Journey artifact store, only with `journey.enabled=true`. |

`fsGroup` makes the mounted volumes group-writable for the non-root user.
The rendered manifests contain no Secrets, `hostPath` volumes, host ports, or
host namespaces.

Helm merges your values with the defaults key by key, so `securityContext: {}`
or `podSecurityContext: {}` keeps every hardened default. Weakening a setting
needs an explicit value for that key, for example:

```yaml
securityContext:
  # Only for an image that must write to its root filesystem.
  readOnlyRootFilesystem: false
```

Set a key to `null` to remove it from the rendered manifest. The chart tests in
`tests/unit/deploy/test_helm_chart.py` assert the hardened posture for the
default, `ci-values.yaml`, Journey, and emptyDir-cache renders, and flag each
weakening override by the rule it breaks. If you build your own image, give it
the same numeric non-root user or override `runAsUser`, `runAsGroup`, and
`fsGroup` together.

### Upgrading from the root-based layout

Earlier chart versions ran the API container as root, left both security
contexts empty unless Journey was enabled, and mounted the model cache at
`/root/.cache/huggingface`. When you upgrade an existing release:

- Remove `extraVolumes` or `extraVolumeMounts` entries named `home` or `tmp`,
  or mounted at `/tmp`; the chart now provides those volumes itself.
- An existing model-cache PVC keeps its files, but they were written for the
  old `/root/.cache/huggingface` mount, so the new layout does not find them
  and models download again on first use. To reuse them, move the volume's
  old top-level entries into a `huggingface/` directory on the volume before
  upgrading. `fsGroup` makes the old root-owned files writable on volume
  types that support ownership management.
- Images that need a writable root filesystem or a root user must set those
  keys explicitly as shown above.

## Production Journey profile

Create a PostgreSQL database and a least-privilege application role, then put
its DSN in an existing Secret. The chart deliberately never accepts the DSN as
a plain Helm value:

```bash
kubectl -n openmed create secret generic openmed-journey-database \
  --from-literal=dsn='postgresql://USER:PASSWORD@postgres.openmed.svc:5432/openmed'
```

Install with Journey enabled and explicitly identify the in-cluster database
pods allowed by the default-deny egress policy:

```bash
helm upgrade --install openmed-service deploy/helm/openmed-service \
  --namespace openmed \
  --create-namespace \
  --set journey.enabled=true \
  --set journey.networkPolicy.databasePodSelector.app=postgres
```

For an external database, keep `databasePodSelector` empty and supply a
reviewed `journey.networkPolicy.additionalEgress` IP block. Do not disable the
whole policy merely to make an unknown destination reachable. DNS is the only
default egress other than an explicitly selected database.

The Helm hook job runs `openmed-journey migrate` before install and upgrade.
It uses an advisory migration lock and checksum verification, so concurrent
releases cannot race the schema. The API readiness command then distinguishes
migration, metadata-store, artifact-store, worker, and local model state. The
worker has separate liveness and readiness probes and no service-account token.

## Upgrade

Upgrade by changing values and running the same release name:

```bash
helm upgrade openmed-service deploy/helm/openmed-service \
  --namespace openmed \
  --set image.repository=ghcr.io/maziyarpanahi/openmed \
  --set image.tag=v3.2.0
```

The chart does not create an Ingress or autoscaling object. Add those in
environment-specific overlays so cluster ingress classes, certificates, and HPA
policy stay outside the reusable chart.

For Journey releases, take a paired PostgreSQL and artifact-volume recovery
point before `helm upgrade`. Wait for the migration hook, both Deployments,
and the value-free golden journey before declaring success. Migrations are
additive; a Helm rollback never reverses them. Roll back only to an image whose
documented compatibility major accepts the current schema, or restore the
paired recovery point into new storage.

## Probes

The chart wires Kubernetes probes to the service endpoints added for
orchestrated deployments:

- liveness probe: `GET /livez`
- readiness probe: `GET /readyz`

`/livez` reports that the process is running. `/readyz` returns success only
after configured model preload completes and flips back to not ready during
graceful shutdown. Probe requests set `Host: localhost` so the service trusted
host middleware accepts kubelet checks without weakening the application host
allowlist.

## Secrets

Do not put tokens or other secrets in `values.yaml`. Reference an existing
Kubernetes Secret through `extraEnv`:

```yaml
extraEnv:
  - name: HF_TOKEN
    valueFrom:
      secretKeyRef:
        name: openmed-hf-token
        key: token
```

Rotate the database credential with overlapping roles: create the successor
role, update the referenced Secret, roll migration/worker/API in order, verify
health, and revoke the old role after the rollback window. Kubernetes Secret
objects are not backups; use the cluster's encrypted secret manager.

## Values Reference

| Value | Default | Description |
| --- | --- | --- |
| `replicaCount` | `1` | Number of service pods. |
| `image.repository` | `openmed` | Container image repository. |
| `image.tag` | `3.2.0` | Container image tag. Empty uses `Chart.appVersion`. |
| `image.pullPolicy` | `IfNotPresent` | Kubernetes image pull policy. |
| `imagePullSecrets` | `[]` | Pull secrets for private image registries. |
| `nameOverride` | `""` | Short name override. |
| `fullnameOverride` | `""` | Full resource name override. |
| `podLabels` | `{}` | Additional pod labels. |
| `podAnnotations` | `{}` | Additional pod annotations. |
| `podSecurityContext.runAsNonRoot` | `true` | Reject a root UID for every container in the API pod. |
| `podSecurityContext.runAsUser` | `65532` | Fixed non-root pod UID. |
| `podSecurityContext.runAsGroup` | `65532` | Fixed non-root pod GID. |
| `podSecurityContext.fsGroup` | `65532` | Group that owns mounted volumes. |
| `podSecurityContext.fsGroupChangePolicy` | `OnRootMismatch` | Re-own volumes only when their root does not match. |
| `podSecurityContext.seccompProfile.type` | `RuntimeDefault` | Pod seccomp profile. |
| `securityContext.runAsNonRoot` | `true` | Reject a root UID for the API container. |
| `securityContext.runAsUser` | `65532` | Fixed non-root container UID. |
| `securityContext.runAsGroup` | `65532` | Fixed non-root container GID. |
| `securityContext.privileged` | `false` | Never run the API container privileged. |
| `securityContext.allowPrivilegeEscalation` | `false` | Block setuid and similar privilege gains. |
| `securityContext.readOnlyRootFilesystem` | `true` | Mount the image root filesystem read-only. |
| `securityContext.capabilities.drop` | `[ALL]` | Drop every Linux capability. |
| `securityContext.seccompProfile.type` | `RuntimeDefault` | Container seccomp profile. |
| `writableVolumes.home.mountPath` | `/home/nonroot` | emptyDir mounted as `HOME`. |
| `writableVolumes.home.sizeLimit` | `128Mi` | `HOME` emptyDir size limit. |
| `writableVolumes.tmp.sizeLimit` | `512Mi` | `/tmp` emptyDir size limit; `/tmp` is exported as `TMPDIR`. |
| `writableVolumes.modelCache.sizeLimit` | `20Gi` | Model-cache emptyDir size limit when `persistence.enabled=false`; empty means unbounded. |
| `terminationGracePeriodSeconds` | `45` | Pod shutdown grace period. |
| `config.profile` | `prod` | `OPENMED_PROFILE`. |
| `config.cacheDir` | `/home/nonroot/.cache/openmed` | `OPENMED_CACHE_DIR`. |
| `config.hfHome` | `/home/nonroot/.cache/huggingface` | `HF_HOME`. |
| `config.preloadModels` | `[]` | Models joined into `OPENMED_SERVICE_PRELOAD_MODELS`. |
| `config.maxResidentModels` | `""` | `OPENMED_SERVICE_MAX_RESIDENT_MODELS`; empty means unbounded. |
| `config.keepAlive` | `10m` | `OPENMED_SERVICE_KEEP_ALIVE`. |
| `config.maxTextLength` | `1000000` | `OPENMED_SERVICE_MAX_TEXT_LENGTH`. |
| `config.corsOrigins` | `[]` | Exact origins joined into `OPENMED_SERVICE_CORS_ORIGINS`. |
| `config.trustedHosts` | `[]` | Extra hosts appended to generated loopback and service DNS names. |
| `config.batching.enabled` | `false` | `OPENMED_SERVICE_BATCHING_ENABLED`. |
| `config.batching.maxSize` | `8` | `OPENMED_SERVICE_BATCH_MAX_SIZE`. |
| `config.batching.maxWaitMs` | `5` | `OPENMED_SERVICE_BATCH_MAX_WAIT_MS`. |
| `config.batching.maxQueueSize` | `256` | `OPENMED_SERVICE_BATCH_MAX_QUEUE_SIZE`. |
| `config.batching.highWatermark` | `256` | `OPENMED_SERVICE_BATCH_HIGH_WATERMARK`. |
| `config.batching.lowWatermark` | `128` | `OPENMED_SERVICE_BATCH_LOW_WATERMARK`. |
| `config.batching.maxQueueWaitMs` | `1000` | `OPENMED_SERVICE_BATCH_MAX_QUEUE_WAIT_MS`. |
| `config.coalescing.enabled` | `false` | `OPENMED_SERVICE_COALESCING_ENABLED`. |
| `config.shutdownDrainSeconds` | `30` | `OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS`. |
| `config.metrics.enabled` | `false` | `OPENMED_SERVICE_METRICS_ENABLED`. |
| `config.throttle.rateLimitRps` | `0` | `OPENMED_SERVICE_RATE_LIMIT_RPS`; `0` disables rate limiting. |
| `config.throttle.rateLimitBurst` | `0` | `OPENMED_SERVICE_RATE_LIMIT_BURST`. |
| `config.throttle.maxConcurrency` | `0` | `OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY`; `0` disables concurrency limiting. |
| `config.throttle.concurrencyWaitSeconds` | `0.05` | `OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS`. |
| `config.throttle.keyBy` | `global` | `OPENMED_SERVICE_THROTTLE_KEY`. |
| `extraEnv` | `[]` | Extra container env entries, typically secret references. |
| `service.type` | `ClusterIP` | Kubernetes Service type. |
| `service.port` | `8080` | Service port. |
| `service.targetPort` | `8080` | Container port. |
| `service.annotations` | `{}` | Service annotations. |
| `probes.liveness.path` | `/livez` | Liveness probe path. |
| `probes.liveness.initialDelaySeconds` | `10` | Liveness initial delay. |
| `probes.liveness.periodSeconds` | `10` | Liveness period. |
| `probes.liveness.timeoutSeconds` | `3` | Liveness timeout. |
| `probes.liveness.failureThreshold` | `3` | Liveness failure threshold. |
| `probes.readiness.path` | `/readyz` | Readiness probe path. |
| `probes.readiness.initialDelaySeconds` | `5` | Readiness initial delay. |
| `probes.readiness.periodSeconds` | `5` | Readiness period. |
| `probes.readiness.timeoutSeconds` | `3` | Readiness timeout. |
| `probes.readiness.failureThreshold` | `3` | Readiness failure threshold. |
| `resources.requests.cpu` | `500m` | Requested CPU. |
| `resources.requests.memory` | `1Gi` | Requested memory. |
| `resources.limits.cpu` | `2` | CPU limit. |
| `resources.limits.memory` | `4Gi` | Memory limit. |
| `persistence.enabled` | `true` | Mount a model-cache PVC; `false` uses a bounded emptyDir. |
| `persistence.existingClaim` | `""` | Existing PVC to mount instead of creating one. |
| `persistence.storageClassName` | `""` | StorageClass for generated PVC. Empty uses the cluster default. |
| `persistence.accessModes` | `[ReadWriteOnce]` | PVC access modes. |
| `persistence.size` | `20Gi` | PVC size. |
| `persistence.mountPath` | `/home/nonroot/.cache` | Model cache mount path. |
| `persistence.subPath` | `""` | Optional PVC subPath. |
| `persistence.annotations` | `{}` | PVC annotations. |
| `nodeSelector` | `{}` | Node selector. |
| `tolerations` | `[]` | Pod tolerations. |
| `affinity` | `{}` | Pod affinity. |
| `extraVolumes` | `[]` | Extra pod volumes. |
| `extraVolumeMounts` | `[]` | Extra container volume mounts. |
| `journey.enabled` | `false` | Enable migrations, worker, artifact PVC, dependency health, and NetworkPolicy. |
| `journey.schema` | `openmed_journey` | Controlled PostgreSQL schema name. |
| `journey.database.secretName` | `openmed-journey-database` | Existing Secret containing the DSN. |
| `journey.database.secretKey` | `dsn` | Key inside the database Secret. |
| `journey.artifacts.existingClaim` | `""` | Existing artifact PVC; empty creates one. |
| `journey.artifacts.size` | `20Gi` | Requested content-addressed artifact capacity. |
| `journey.worker.replicas` | `1` | Operational worker replicas. |
| `journey.worker.port` | `8091` | Internal worker health port. |
| `journey.migration.backoffLimit` | `3` | Bounded migration Job retries. |
| `journey.networkPolicy.enabled` | `true` | Apply default-deny ingress and egress to chart pods. |
| `journey.networkPolicy.allowDns` | `true` | Allow cluster DNS only. |
| `journey.networkPolicy.databasePodSelector` | `{}` | Explicit in-cluster PostgreSQL pod selector. |
| `journey.networkPolicy.additionalEgress` | `[]` | Reviewed extra Kubernetes egress rules. |
