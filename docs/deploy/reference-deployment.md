# v3.4 reference deployment declaration

This page declares the deployment contract for the v3.4 release line: which
topology is the reference, which transports are supported and with what
security posture, how each process may be replicated, which components hold
durable state and what recovery point objective (RPO) and recovery time
objective (RTO) each one carries, and which deployments are unsupported.
Reference workflows, recovery drills, and data-integrity evidence measure
against this contract; they do not redefine it.

The machine-readable source of truth is
`deploy/reference/v3.4-reference-deployment.json`, validated by
`deploy/reference/reference-deployment.schema.json`. The tables below render
that file row for row. A unit test fails when the two disagree, when the
declaration fails its schema, or when it names a REST route, worker route,
MCP path, gRPC method, Python module or symbol, Compose service or volume,
Helm value, manifest, or docs page that does not exist in the repository.

The declaration describes current behavior. It contains only controlled codes
and repository-relative references: no hostnames, addresses, credentials,
connection strings, or patient data.

!!! warning "Objectives are declared targets, not measured results"

    `objective_evidence` is `unmeasured_targets`. The RPO and RTO values below
    are the targets for the reference topology when an operator follows the
    documented backup and restore procedure. They are not guarantees, and no
    drill result is recorded here. Operators may declare tighter objectives
    for their own deployment, but must not assume OpenMed provides
    replication, automated backup, or failover.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| reference | The single configuration the release is validated against. |
| supported | Shipped and documented; may be used in production within its constraints. |
| restricted | Shipped, but only safe inside the listed constraints, usually an internal network. |
| unsupported | Present in the code base or repository, but not a supported production deployment. |

Constraint codes are listed exactly as they appear in the declaration.

## Topologies

The reference topology is `compose_single_host`: the offline Compose bundle on
one host, with bundled PostgreSQL, a one-shot migration, the Journey worker,
the REST API, and the optional `smoke` and `mcp` profiles on an internal
network. Only the REST port is published, and it binds to loopback by default.
See [Self-hosted Compose deployment](self-hosted-compose.md) and
[Helm deployment](helm.md) for the operating procedures.

| Topology | Status | Manifests | Services |
| --- | --- | --- | --- |
| `compose_single_host` | reference | `deploy/openmed-compose.yaml`, `deploy/docker/Dockerfile.distroless` | `postgres`, `journey_migrate`, `journey_worker`, `rest_api`, `journey_golden`, `mcp_server` |
| `helm_kubernetes` | supported | `deploy/helm/openmed-service/Chart.yaml`, `deploy/helm/openmed-service/values.yaml`, `deploy/helm/openmed-service/templates/deployment.yaml`, `deploy/helm/openmed-service/templates/network-policy.yaml` | `rest_api`, `journey_migrate`, `journey_worker`, `postgres` |
| `kubernetes_autoscaled` | supported | `deploy/k8s/hpa.yaml` | `rest_api` |
| `kubernetes_model_operator` | supported | `deploy/operator/kustomization.yaml`, `deploy/operator/crd/openmedmodel.yaml` | `rest_api` |
| `development_compose` | unsupported | `docker-compose.yml`, `Dockerfile`, `Dockerfile.mcp` | `rest_api`, `mcp_server` |

The repository-root Compose file is a development convenience. It publishes
the REST port on every interface without authentication and has no Journey
store, so it is declared `unsupported` for production.

## Services and replica modes

| Service | Replica mode | Status | Constraints |
| --- | --- | --- | --- |
| `rest_api` | `single` | reference | none |
| `rest_api` | `horizontal` | restricted | `synchronous_routes_only`, `per_replica_warm_pool`, `per_replica_rate_limits`, `per_replica_job_state` |
| `journey_worker` | `single` | reference | `health_probe_only` |
| `journey_migrate` | `one_shot` | reference | `single_migration_runner`, `completes_before_api_and_worker` |
| `journey_golden` | `one_shot` | supported | `smoke_profile_only`, `synthetic_content_only` |
| `mcp_server` | `single` | supported | `optional_profile` |
| `postgres` | `single` | reference | `bundled_single_instance` |
| `postgres` | `external` | supported | `operator_managed_database`, `replication_not_managed_by_openmed` |
| `grpc_server` | `single` | unsupported | `no_deployment_wiring` |

The REST API can run as more than one replica, for example under the
[Kubernetes autoscaling](autoscaling.md) manifest, only for synchronous
routes. Warm model pools, in-process rate limits, async job metadata, FHIR
Bulk Data jobs, and SMART backend ingestion jobs are all per process. A status
poll that lands on a different replica does not see the job.

The gRPC server is a library factory only. No console entry point, container
image, Compose service, chart template, or health service starts it, so it has
no supported replica mode.

## Transports and security posture

| Transport | Status | Encryption | Authentication | Default exposure | Constraints |
| --- | --- | --- | --- | --- | --- |
| `rest_http` | supported | `optional_tls` | `optional` | `loopback` | `authentication_disabled_by_default`, `non_loopback_requires_authentication`, `non_loopback_requires_tls`, `non_loopback_requires_trusted_hosts`, `cluster_ip_service_in_helm` |
| `graphql_http` | supported | `optional_tls` | `optional` | `loopback` | `shares_rest_listener`, `authentication_disabled_by_default` |
| `mcp_stdio` | supported | `not_applicable` | `not_applicable` | `local_process` | `client_spawned_process` |
| `mcp_streamable_http` | restricted | `none` | `optional` | `internal_network` | `internal_network_only`, `no_published_port`, `authorization_disabled_by_default`, `insecure_loopback_mode_enabled` |
| `journey_worker_health_http` | restricted | `none` | `none` | `internal_network` | `value_free_health_only`, `internal_network_only` |
| `postgresql_wire` | restricted | `none` | `required` | `internal_network` | `tls_not_enforced_by_manifests`, `internal_network_only`, `credentials_from_operator_secret` |
| `job_webhook_outbound` | restricted | `optional_tls` | `required` | `outbound_only` | `hmac_signed_payloads`, `https_not_enforced`, `no_egress_on_reference_network` |
| `grpc` | unsupported | `none` | `none` | `not_wired` | `no_deployment_wiring`, `insecure_port_only`, `binds_all_interfaces_by_default`, `no_health_service` |

- REST and GraphQL share one listener. TLS comes from the launcher's TLS
  flags or from a terminating proxy; API key, JWT, and mTLS authentication are
  opt-in and disabled by default. The Helm chart exposes it through a
  cluster-internal Service. Exposing the listener beyond loopback or the
  cluster requires authentication, TLS, and an explicit trusted-host policy.
  See [REST authentication](../serving/authentication.md) and
  [mutual TLS](../serving/mtls.md).
- MCP over standard input and output opens no listener. The Compose `mcp`
  profile serves stateless streamable HTTP in plaintext on the internal
  network only, with no published port.
- The Journey worker health listener and the PostgreSQL connection are
  plaintext inside the internal network. The shipped manifests do not require
  verified database TLS.
- Job webhooks are signed, but either plaintext or TLS targets are accepted,
  and the reference network has no egress.
- gRPC is declared `unsupported` as a network transport: the default server
  opens an insecure port on every interface with no TLS, authentication, or
  health service, and nothing in the deployment manifests wires it. See
  [gRPC service](../serving/grpc.md) for the library contract.

## Durable components and recovery objectives

Every durable component carries both an RPO and an RTO. Each objective is
either `declared`, with a target and a basis code, or `unsupported`, with a
reason code. The schema rejects a component that omits either objective.

| Component | Storage | Replica mode | RPO | RTO |
| --- | --- | --- | --- | --- |
| `journey_database` | `postgresql` | `single_primary` | 24 h (`daily_application_consistent_backup`) | 4 h (`isolated_restore_then_migrate_and_golden_journey`) |
| `journey_artifacts` | `filesystem_object_store` | `single_writer_volume` | 24 h (`paired_with_journey_database_recovery_point`) | 4 h (`isolated_restore_then_migrate_and_golden_journey`) |
| `model_cache` | `filesystem_cache` | `single_writer_volume` | 0 s (`derived_from_verified_model_artifacts`) | 4 h (`reprovision_from_offline_model_bundle`) |
| `service_job_metadata` | `local_json_file` | `single_process` | unsupported (`payloads_held_in_process_memory`) | unsupported (`single_process_job_store`) |
| `service_bulk_job_state` | `process_memory` | `per_replica_memory` | unsupported (`in_memory_job_state`) | unsupported (`in_memory_job_state`) |
| `database_credentials` | `operator_secret_store` | `external_operator_owned` | unsupported (`operator_owned_secret_store`) | unsupported (`operator_owned_secret_store`) |

- `journey_database` and `journey_artifacts` form one recovery set. Back them
  up together as an application-consistent recovery point at least daily, and
  restore into an isolated namespace, run the migration gate, then run the
  synthetic golden journey before admitting traffic. The procedure is in
  [Self-hosted Compose deployment](self-hosted-compose.md) and the
  [PostgreSQL and object Journey store](../structured/postgres-object-journey-store.md)
  guide.
- `model_cache` holds only artifacts derived from verified model bundles. Loss
  is recovered by re-provisioning from the offline bundle, so there is no
  recovery point to lose.
- `service_job_metadata` is a local JSON file owned by one process. Submitted
  payloads stay in memory, so queued or running jobs do not survive a restart
  and the file must not be shared between replicas. Recovery objectives are
  `unsupported`; resubmit interrupted jobs.
- `service_bulk_job_state` lives in process memory and is lost on restart.
- `database_credentials` are owned by the operator's secret source. OpenMed
  neither renders nor persists them, so their recovery is out of scope.

## Unsupported deployments

| Deployment | Reason code | Summary |
| --- | --- | --- |
| `grpc_network_service` | `no_deployment_wiring` | Serving gRPC as a network service. Nothing in the images, Compose bundle or chart starts it, and its default listener is plaintext and unauthenticated. |
| `multi_replica_async_jobs` | `single_process_job_store` | Running async jobs, FHIR Bulk Data or SMART ingestion behind more than one REST replica. Job state is process-local, so status polls can miss the owning replica. |
| `unauthenticated_non_loopback_rest` | `authentication_and_tls_required_off_loopback` | Publishing the REST or GraphQL listener beyond loopback without authentication, TLS and an explicit trusted-host policy. |
| `public_mcp_streamable_http` | `insecure_loopback_mode` | Publishing the Compose MCP streamable HTTP listener. The profile enables the insecure-loopback authorization mode and has no TLS. |
| `openmed_managed_database_replication` | `replication_not_managed_by_openmed` | Relying on OpenMed for PostgreSQL replication, failover or multi-primary writes. High availability of the database is an operator responsibility. |
| `development_compose_in_production` | `all_interface_publish_without_authentication` | Using the repository-root development Compose file for production. It publishes the REST port on every interface without authentication. |

## Changing the declaration

Edit `deploy/reference/v3.4-reference-deployment.json` and this page together,
then run:

```bash
python -m pytest tests/unit/deploy/test_reference_deployment.py -q
```

The test validates the schema, resolves every reference, checks that each
durable component declares both recovery objectives, compares every table row
on this page with the declaration, and scans both files for hostnames,
addresses, credentials, and patient-data patterns. Its negative controls prove
that a missing route, module, symbol, gRPC method, manifest, Compose service or
volume, Helm value, or docs page fails the check.
