# Kubernetes model operator

The OpenMed Kubernetes operator makes the model version served by an existing
OpenMed REST Deployment declarative. An `OpenMedModel` resource selects one
family, version pointer, tier, replica count, and rollout policy. The operator
then:

1. writes an owned ConfigMap containing the active model manifest pointer;
2. points the target container's `OPENMED_SERVICE_PRELOAD_MODELS` variable at
   that ConfigMap;
3. hashes the manifest into the Deployment pod template so Kubernetes replaces
   the pods and each new process warms the selected model before `/readyz`
   succeeds;
4. reports rollout state through standard conditions and Kubernetes Events;
5. restores the last successful pointer when a rollout exceeds the Deployment
   progress deadline and automatic rollback is enabled.

The operator does not orchestrate training, change cluster autoscaling, ship
model weights, or contact a model registry. Its only network dependency is the
Kubernetes API. The service remains responsible for resolving the configured
model pointer from its local cache or an explicitly permitted model source.

## Resource contract

`OpenMedModel` is namespaced under `openmed.ai/v1alpha1`. Its required fields
are:

| Field | Meaning |
| --- | --- |
| `spec.family` | Logical model family, such as `PII`. |
| `spec.version` | Exact model name, immutable repository id, or local path accepted by the OpenMed service. |
| `spec.tier` | One of `Tiny`, `Small`, `Medium`, `Base`, `Large`, `XLarge`, or `Accurate-XLarge`. |
| `spec.replicas` | Target Deployment replica count, from 1 through 1000. |
| `spec.rolloutStrategy` | `RollingUpdate` or `Recreate`, plus deadline and rollback controls. |

`spec.targetRef.name` defaults to the resource name and
`spec.targetRef.containerName` defaults to `openmed-service`. The target must be
an existing Deployment in the same namespace. One `OpenMedModel` owns one
target Deployment; a second resource targeting an already owned Deployment is
held with a `TargetConflict` condition rather than racing the first resource.

The CRD rejects unknown spec fields and tiers before reconciliation. The
operator repeats the same validation internally so malformed resources cannot
bypass the contract through an older API server or direct test invocation.

## Build and install

Build and publish the small operator image to a registry available to the
cluster. Kopf is the only operator-specific dependency and is pinned in the
image; it is also available through the `operator` Python extra.

```bash
docker build \
  -f deploy/operator/Dockerfile \
  -t registry.example/openmed-operator:v2.3.0 \
  .
docker push registry.example/openmed-operator:v2.3.0
```

Set the image in `deploy/operator/deployment.yaml` (or with a Kustomize image
override), then install the CRD, RBAC, namespace, and single-replica operator:

```bash
kubectl apply -k deploy/operator
kubectl -n openmed-system rollout status deployment/openmed-operator
```

The bundled Deployment runs one replica with a `Recreate` strategy. Running
multiple standalone Kopf replicas against the same resources can cause
duplicate reconciliation, so do not scale it horizontally. The operator has a
liveness endpoint but no readiness endpoint because it does not serve workload
traffic.

For local development without building the image:

```bash
uv sync --extra operator
uv run kopf run --standalone --all-namespaces \
  deploy/operator/openmed_operator.py
```

## Prepare the service Deployment

Install the OpenMed REST service first. Its Deployment must expose the standard
`openmed-service` container and `/readyz` probe. For the Helm chart, use a
stable name matching the custom resource and leave initial preload selection to
the operator:

```bash
helm upgrade --install openmed-service deploy/helm/openmed-service \
  --namespace openmed \
  --create-namespace \
  --set fullnameOverride=openmed-service \
  --set-json 'config.preloadModels=[]'
```

Production clusters should retain the chart's persistent model cache. Ensure
the model named by `spec.version` is already cached for an air-gapped rollout,
or explicitly configure the service pod with the credentials and egress needed
for its one-time download. The operator never reads those credentials and its
RBAC grants no access to Secrets.

## Apply and observe a model

The synthetic example targets the Helm Deployment above:

```bash
kubectl apply -f deploy/operator/example-openmedmodel.yaml
kubectl -n openmed get openmedmodel openmed-service -w
kubectl -n openmed wait \
  --for=condition=Ready \
  --timeout=10m \
  openmedmodel/openmed-service
```

Inspect the value-free lifecycle evidence with:

```bash
kubectl -n openmed describe openmedmodel openmed-service
kubectl -n openmed get events \
  --field-selector involvedObject.kind=OpenMedModel
kubectl -n openmed get configmap openmed-service-model-manifest -o yaml
```

The ConfigMap contains only family, tier, version pointer, and the preload
environment value. The status stores those coordinates, hashes, Deployment
generations, and retained successful specs. It never contains request text,
detected entities, model output, patient identifiers, or credentials.

## Roll out a new version

Change `spec.version` to an immutable challenger pointer:

```bash
kubectl -n openmed patch openmedmodel openmed-service \
  --type=merge \
  -p '{"spec":{"version":"OpenMed/synthetic-pii-v2"}}'
```

The first reconcile writes the new pointer and sets `Progressing=True` with
phase `RollingOut`. New pods preload the model before their service readiness
probe passes. Once the Deployment has observed its new generation and all
desired replicas are updated, ready, and available, the operator sets
`Ready=True`, records the version as `lastSuccessfulSpec`, and emits a
`RolloutSucceeded` Event.

The operator continuously reconciles every 15 seconds. If Helm or another
controller removes the manifest reference, replica count, or rollout settings,
the operator reports `DriftCorrected` and restores the custom resource's desired
state.

## Automatic and manual rollback

With `rollbackOnFailure: true`, a Deployment condition of
`ProgressDeadlineExceeded` or `ReplicaFailure=True` causes an automatic pointer
flip to `lastSuccessfulSpec`. The resource remains explicit about the mismatch:

- `status.phase` is `RolledBack`;
- `status.desiredVersion` remains the failed version;
- `status.activeVersion` is the restored version;
- `Degraded=True` and, once the restored pods are available,
  `RolledBack=True` with reason `RollbackSucceeded`.

The operator does not retry the same failed spec forever. Change the spec to a
new version to start another rollout, or set it to the restored version to make
desired and active state agree.

After two versions have succeeded, request a deliberate rollback to either of
the retained successful versions with an annotation:

```bash
kubectl -n openmed annotate openmedmodel openmed-service \
  openmed.ai/rollback-to=OpenMed/synthetic-pii-v1 --overwrite
```

Only `lastSuccessfulSpec` and `previousSuccessfulSpec` are accepted manual
targets. This prevents an annotation from loading an unreviewed model. After
the rollback is available, update `spec.version` to the restored pointer and
remove the annotation:

```bash
kubectl -n openmed patch openmedmodel openmed-service \
  --type=merge \
  -p '{"spec":{"version":"OpenMed/synthetic-pii-v1"}}'
kubectl -n openmed annotate openmedmodel openmed-service \
  openmed.ai/rollback-to-
```

## Conditions and Events

| Condition | Interpretation |
| --- | --- |
| `Ready` | The requested version is available on every desired replica. |
| `Progressing` | Kubernetes is rolling out either the desired or rollback pointer. |
| `Degraded` | The target is missing/conflicted, the rollout failed, or desired state is held at a rollback. |
| `RolledBack` | A retained successful version has been fully restored. |

The operator emits idempotently named Events for rollout start/success/failure,
rollback start/success/rejection, missing/conflicting targets, and deletion.
Event messages contain lifecycle state only and are safe for cluster-level
operational logs.

## Deletion lifecycle

Deleting an `OpenMedModel` removes its ConfigMap pointer and changes the target
container's preload value to an empty string. The pod-template hash changes, so
the target Deployment replaces its pods and shuts down the old warm pool. The
operator never deletes model weights or the target Deployment. The ConfigMap
also has an owner reference as a garbage-collection backstop.

## RBAC and namespace scope

The bundled deployment watches all namespaces. Its ClusterRole can:

- watch and patch `OpenMedModel` resources, status, and finalizers;
- read and patch Deployments;
- manage only ConfigMaps and Events used for reconciliation;
- discover the CRD for Kopf.

It cannot read Secrets, create workloads, mutate Services, change autoscalers,
or access nodes. Clusters that need strict tenant isolation can copy these same
rules into a namespaced Role, replace `--all-namespaces` with one or more
`--namespace` arguments, and bind the service account only in those namespaces.

## Offline synthetic validation

The unit suite starts an in-process fake Kubernetes HTTP API, applies a
synthetic `OpenMedModel`, observes ConfigMap and Deployment patches, advances a
healthy rollout, forces a progress-deadline failure, and verifies restoration
of the last successful pointer. It also validates the CRD and deployment/RBAC
manifests:

```bash
python -m pytest tests/unit/deploy/test_operator_reconcile.py -q
```

No cluster, model download, restricted vocabulary, real patient data, or
external network call is required by these tests.

## Troubleshooting

- `TargetNotFound`: the target Deployment name or namespace does not match the
  custom resource. Set `spec.targetRef.name` explicitly.
- `ContainerNotFound`: set `spec.targetRef.containerName` to the OpenMed REST
  container name.
- `TargetConflict`: another `OpenMedModel` already owns the target Deployment.
  Give each resource a separate target.
- `ProgressDeadlineExceeded`: inspect pod events, cache capacity, model pointer,
  credentials, memory limits, and `/readyz`. Automatic rollback starts only
  when a last successful spec exists.
- `RollbackTargetUnavailable`: manual rollback retains only the current and
  immediately previous successful specs.

Model extraction is assistive software, not a source of clinical truth. A
model rollout must not automatically trigger diagnosis, treatment, billing,
data release, or another clinical decision.
