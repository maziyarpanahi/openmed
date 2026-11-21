# Helm Deployment

The `deploy/helm/openmed-service` chart deploys the OpenMed de-identification
REST service on Kubernetes with separate liveness and readiness probes, a
persistent model cache volume, non-root read-only security contexts, and
configurable service runtime settings.
Its optional production Journey profile adds a pre-install/pre-upgrade
migration gate, a worker Deployment, durable artifact storage, dependency-aware
readiness, and a default-deny NetworkPolicy.

The default `config.profile: prod` selects the production
[security profile](../serving/security-profile.md): pods stay unready with
`security_profile_invalid` until `extraEnv` configures authentication. Run
`openmed service preflight` against the same environment before installing.
`deploy/posture/helm/production.yaml` is a validated production overlay, and
the [deployment posture lane](security-posture.md)
(`python -m openmed.deploy.posture`) checks rendered values against the
production contract offline. The chart disables Hugging Face and library
telemetry (`HF_HUB_DISABLE_TELEMETRY`, `DISABLE_TELEMETRY`, `DO_NOT_TRACK`) in
every workload.

## Install

Build and publish an OpenMed service image to a registry your cluster can pull,
then install the chart with the matching image repository and tag:

```bash
helm upgrade --install openmed-service deploy/helm/openmed-service \
  --namespace openmed \
  --create-namespace \
  --set image.repository=ghcr.io/maziyarpanahi/openmed \
  --set image.tag=v3.3.0
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
  --from-literal=dsn='postgresql://USER:PASSWORD@postgres.openmed.svc:5432/openmed?sslmode=verify-full'
```

The Journey pods run in the `prod` profile, which requires verified TLS: a
plaintext or non-verifying DSN is refused before any socket is opened and the
probes report `postgres_tls_required`. Put the CA bundle that issued the
PostgreSQL server certificate, and optionally a client certificate and key, in
an existing Secret. The server certificate must match the DSN host name:

```bash
kubectl -n openmed create secret generic openmed-journey-postgres-tls \
  --from-file=ca.crt=./postgres-ca.crt
```

Install with Journey enabled, reference the TLS Secret, and explicitly identify
the in-cluster database pods allowed by the default-deny egress policy:

```bash
helm upgrade --install openmed-service deploy/helm/openmed-service \
  --namespace openmed \
  --create-namespace \
  --set journey.enabled=true \
  --set journey.database.tls.secretName=openmed-journey-postgres-tls \
  --set journey.networkPolicy.databasePodSelector.app=postgres
```

The chart mounts the Secret read-only at
`journey.database.tls.mountPath` in the migration Job, the worker, and the API
pod, and sets `OPENMED_JOURNEY_POSTGRES_SSLROOTCERT` (plus
`OPENMED_JOURNEY_POSTGRES_SSLCERT` and `OPENMED_JOURNEY_POSTGRES_SSLKEY` when
`clientCertKey` and `clientKeyKey` are set) to file paths only. Leave
`journey.database.tls.secretName` empty only when the database certificate
chains to the image trust store. Server-side `pg_hba.conf` enforcement, role
management, and encryption at rest remain operator responsibilities.

The chart mounts the `dsn` key read-only at
`/var/run/secrets/openmed/journey-database/dsn` in the migration Job, worker,
and API pods and sets `OPENMED_JOURNEY_POSTGRES_DSN_FILE`, so the DSN never
appears in a container environment. Kubernetes presents the file as root-owned
`0440` with the pod `fsGroup` (`65532`), which the service accepts under the
`prod` profile. Set `journey.database.source=env` only for images that predate
`_FILE` support; it restores the `secretKeyRef` mapping of
`OPENMED_JOURNEY_POSTGRES_DSN`.

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
  --set image.tag=v3.3.0
```

The chart does not create an Ingress or autoscaling object. Add those in
environment-specific overlays so cluster ingress classes, certificates, and HPA
policy stay outside the reusable chart. When an HPA such as
`deploy/k8s/hpa.yaml` scales the release, set `autoscaling.enabled=true`.

The chart renders `OPENMED_SERVICE_REPLICA_MODE=multi` when `replicaCount` is
greater than 1 or `autoscaling.enabled` is true. In multi-replica mode the API
stays not ready until each replica-local component is disabled or accepted
through `replicaState`; see
[Replica-local state](autoscaling.md#replica-local-state-and-safe-topologies).

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
graceful shutdown. In multi-replica mode it also returns
`replica_unsafe_state` while replica-local state is neither disabled nor
accepted. Probe requests set `Host: localhost` so the service trusted
host middleware accepts kubelet checks without weakening the application host
allowlist.

## Graceful shutdown

On SIGTERM the service stops accepting work and drains in-flight requests for
up to `config.shutdownDrainSeconds`. Kubernetes kills the pod once
`terminationGracePeriodSeconds` has passed, so the chart refuses to render
unless the grace period is greater than the drain timeout plus 5 seconds. With
the defaults (45 and 30) the check passes. A value for
`OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS` in `extraEnv` overrides the ConfigMap,
so the check uses it instead; setting that variable through `valueFrom` is
refused because the chart cannot read it. If you raise the drain timeout,
raise the grace period with it:

```bash
helm upgrade --install openmed-service deploy/helm/openmed-service \
  --namespace openmed \
  --set config.shutdownDrainSeconds=60 \
  --set terminationGracePeriodSeconds=75
```

Setting only `config.shutdownDrainSeconds=60` fails `helm template` and
`helm install` with a message that names both values.

## Secrets

Do not put tokens or other secrets in `values.yaml`. Mount OpenMed
credentials from an existing Kubernetes Secret with `secretFiles`; each entry
mounts one key read-only and sets the setting's `_FILE` companion, so the value
stays out of the container environment and `kubectl describe`:

```yaml
secretFiles:
  - env: OPENMED_SERVICE_AUTH_API_KEYS
    secretName: openmed-service-auth
    secretKey: api-keys.json
  - env: OPENMED_SERVICE_OTLP_HEADERS
    secretName: openmed-otlp
    secretKey: headers
```

This renders `OPENMED_SERVICE_AUTH_API_KEYS_FILE` pointing at
`/var/run/secrets/openmed/openmed-service-auth-api-keys/value`. Supported
settings are `OPENMED_SERVICE_AUTH_API_KEYS`, `OPENMED_SERVICE_AUTH_JWKS`,
`OPENMED_SERVICE_OTLP_HEADERS`, and `OPENMED_OPENHIM_PASSWORD`; do not also set
the inline variable. The files are root-owned `0440`, so outside the Journey
profile set `podSecurityContext.fsGroup` to the container's group unless the
container runs as root. See
[mounted secret files](../security/key-management.md#service-credentials-from-mounted-secret-files)
for the file rules and error codes.

Third-party variables without a `_FILE` form can still reference a Secret
through `extraEnv`:

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
| `replicaCount` | `1` | Number of service pods. More than 1 renders `OPENMED_SERVICE_REPLICA_MODE=multi`. |
| `autoscaling.enabled` | `false` | Marks the release as scaled by an external HPA and renders multi-replica mode. The chart does not create the HPA. |
| `replicaState.accept` | `[]` | Categories joined into `OPENMED_SERVICE_REPLICA_ACCEPT_LOCAL_STATE`. |
| `replicaState.disable` | `[]` | Job categories joined into `OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE`. |
| `image.repository` | `openmed` | Container image repository. |
| `image.tag` | `3.3.0` | Container image tag. Empty uses `Chart.appVersion`. |
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
| `terminationGracePeriodSeconds` | `45` | Pod shutdown grace period. Must be greater than `config.shutdownDrainSeconds` plus 5 seconds. |
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
| `config.transport.limitConcurrency` | `128` | `OPENMED_SERVICE_LIMIT_CONCURRENCY`; connections or tasks per worker before `503`. |
| `config.transport.timeoutKeepAliveSeconds` | `5` | `OPENMED_SERVICE_TIMEOUT_KEEP_ALIVE`; idle HTTP keep-alive seconds. |
| `config.transport.timeoutGracefulShutdownSeconds` | `30` | `OPENMED_SERVICE_TIMEOUT_GRACEFUL_SHUTDOWN`; keep it below `terminationGracePeriodSeconds`. |
| `config.transport.requestBodyTimeoutSeconds` | `60` | `OPENMED_SERVICE_REQUEST_BODY_TIMEOUT_SECONDS`; total body receive time. |
| `config.transport.requestBodyGraceSeconds` | `10` | `OPENMED_SERVICE_REQUEST_BODY_GRACE_SECONDS`; time before the minimum rate applies. |
| `config.transport.requestBodyMinBytesPerSecond` | `1024` | `OPENMED_SERVICE_REQUEST_BODY_MIN_BYTES_PER_SECOND`; `0` keeps only the total deadline. |
| `config.metrics.enabled` | `false` | `OPENMED_SERVICE_METRICS_ENABLED`. |
| `config.throttle.rateLimitRps` | `0` | `OPENMED_SERVICE_RATE_LIMIT_RPS`; `0` disables rate limiting. |
| `config.throttle.rateLimitBurst` | `0` | `OPENMED_SERVICE_RATE_LIMIT_BURST`. |
| `config.throttle.maxConcurrency` | `0` | `OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY`; `0` disables concurrency limiting. |
| `config.throttle.concurrencyWaitSeconds` | `0.05` | `OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS`. |
| `config.throttle.keyBy` | `global` | `OPENMED_SERVICE_THROTTLE_KEY`; `x-forwarded-for` requires `trustedProxies`. |
| `config.throttle.trustedProxies` | `[]` | Proxy addresses or CIDRs joined into `OPENMED_SERVICE_TRUSTED_PROXIES`; empty ignores `X-Forwarded-For`. |
| `config.throttle.trustedProxyHops` | `1` | `OPENMED_SERVICE_TRUSTED_PROXY_HOPS`; trusted proxy tiers in front of the pod. |
| `config.throttle.maxTrackedKeys` | `10000` | `OPENMED_SERVICE_THROTTLE_MAX_TRACKED_KEYS`; per-limiter client identity bound. |
| `extraEnv` | `[]` | Extra container env entries, such as third-party secret references. |
| `secretFiles` | `[]` | OpenMed credential settings mounted from Secret keys as `<env>_FILE`. |
| `secretFilesMountPath` | `/var/run/secrets/openmed` | Parent directory for `secretFiles` mounts. |
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
| `journey.database.tls.secretName` | `""` | Existing Secret with the PostgreSQL CA bundle and optional client certificate. |
| `journey.database.tls.caKey` | `ca.crt` | CA bundle key; exported as `OPENMED_JOURNEY_POSTGRES_SSLROOTCERT`. |
| `journey.database.tls.clientCertKey` | `""` | Optional client certificate key; exported as `OPENMED_JOURNEY_POSTGRES_SSLCERT`. |
| `journey.database.tls.clientKeyKey` | `""` | Optional client key; exported as `OPENMED_JOURNEY_POSTGRES_SSLKEY`. |
| `journey.database.tls.mountPath` | `/etc/openmed/postgres-tls` | Read-only mount path for the TLS Secret. |
| `journey.database.source` | `file` | `file` mounts the DSN and sets `OPENMED_JOURNEY_POSTGRES_DSN_FILE`; `env` uses a `secretKeyRef` value. |
| `journey.database.mountPath` | `/var/run/secrets/openmed/journey-database` | Read-only DSN Secret mount for `source: file`. |
| `journey.artifacts.existingClaim` | `""` | Existing artifact PVC; empty creates one. |
| `journey.artifacts.size` | `20Gi` | Requested content-addressed artifact capacity. |
| `journey.worker.replicas` | `1` | Operational worker replicas. |
| `journey.worker.port` | `8091` | Internal worker health port. |
| `journey.migration.backoffLimit` | `3` | Bounded migration Job retries. |
| `journey.networkPolicy.enabled` | `true` | Apply default-deny ingress and egress to chart pods. |
| `journey.networkPolicy.allowDns` | `true` | Allow cluster DNS only. |
| `journey.networkPolicy.databasePodSelector` | `{}` | Explicit in-cluster PostgreSQL pod selector. |
| `journey.networkPolicy.additionalEgress` | `[]` | Reviewed extra Kubernetes egress rules. |
