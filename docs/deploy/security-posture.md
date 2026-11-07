# Deployment Security Posture Lane

The [service security profile](../serving/security-profile.md) checks a running
service. The deployment posture lane checks what the Helm chart and the Compose
bundle would hand that service, before anything is deployed. It renders the
chart offline with `helm template`, parses the Compose file, and evaluates every
OpenMed workload (API, Journey worker, migration and smoke jobs, MCP server)
against one contract derived from the `prod` profile.

```bash
python -m openmed.deploy.posture                 # human report
python -m openmed.deploy.posture --json          # versioned JSON report
python -m openmed.deploy.posture --platform compose
```

| Exit code | Meaning |
|---|---|
| `0` | Every target produced exactly its recorded findings. |
| `1` | A target produced a new finding, or a recorded finding no longer occurs. |
| `2` | The lane file could not be read, `helm` is missing, or the `service` extra is not installed. |

The lane makes no network calls and needs no cluster or Docker daemon. The
`Helm Chart` workflow runs it on every change to the chart, the Compose bundle,
the Dockerfiles, the lane fixtures or the security-profile module.

## One contract, two rule sets

**Runtime checks.** The REST service workload's rendered environment is passed
to `run_service_preflight(environ=...)` under the `prod` profile, with the
listener its launch command binds. This runs the registered checks the service
runs at startup (`auth.enabled`, `auth.deny_by_default`, `http.trusted_hosts`,
`http.cors`, `limits.request_body`, `journey.store_tls`,
`egress.configured_adapters` and the rest), so chart values and the runtime
self-check cannot drift apart. A configuration the service would refuse to
parse is reported as `deploy.service_config` / `service_config_invalid`.

**Deployment rules.** These cover what the service cannot see from inside:

| Check | Reasons | Applies to |
|---|---|---|
| `deploy.profile` | `development_profile_selected`, `security_profile_value_invalid`, `security_profile_unverified` | every workload |
| `deploy.overrides` | `insecure_override_enabled` (`OPENMED_MCP_ALLOW_INSECURE_LOCALHOST`, `OPENMED_OPENHIM_ALLOW_INSECURE_HTTP`, `OPENMED_SKIP_MODEL_VERIFY`) | every workload |
| `deploy.environment` | `setting_contradictory` (one setting defined twice with different values, for example an `extraEnv` entry overriding the ConfigMap), `environment_source_unverified` (`envFrom` a Secret or an unrendered ConfigMap, Compose `env_file`) | every workload |
| `deploy.security_context` | `runs_as_root`, `privilege_escalation_allowed`, `root_filesystem_writable`, `capabilities_not_dropped`, `capabilities_added`, `privileged_container`, `seccomp_not_confined`, `host_namespace_shared`, `host_path_mounted` | every workload |
| `deploy.resources` | `cpu_limit_missing`, `memory_limit_missing` | every workload |
| `deploy.readiness` | `readiness_probe_missing`, `readiness_not_wired` (the probe does not reach `/readyz`) | API and worker |
| `deploy.egress` | `telemetry_not_disabled`, `telemetry_contradictory`, `offline_mode_contradictory`, `offline_mode_required` (offline bundles only) | every workload |
| `deploy.inheritance` | `security_profile_mismatch`, `inherited_setting_missing`, `inherited_setting_mismatch` (Journey DSN, schema, TLS files and artifact root) | worker and jobs, against the API |
| `deploy.listener` | `exposed_without_tls` (NodePort or LoadBalancer Service, or a Compose port published beyond loopback, without in-process TLS) | every workload |
| `deploy.workload` | `workload_unclassified`, `service_workload_missing` | every rendering |

Security contexts are evaluated as Kubernetes applies them: a container key
wins over the pod key of the same name. `deploy.listener` takes its level from
the runtime `listener.transport` check, which is advisory in profile v1 because
platforms commonly terminate TLS in front of the service; it becomes blocking in
the profile version that makes that check required. Every other rule is
required.

## Secrets stay unread

Settings sourced from a Secret (`secretKeyRef`, or a `*_FILE` path on a Secret
volume or under `/run/secrets`) and Compose values the operator must supply
(`${VAR:?...}`) are never read. For the runtime checks they are replaced in
memory by synthetic, well-formed stand-ins, so presence is checked but content
is not. A check that needs the secret content itself is reported as deferred
(`secret_sourced_runtime_check`) instead: for example `journey.store_tls` when
the PostgreSQL DSN, and with it `sslmode`, lives in a Secret. The service runs
that check on the real value at startup, and the readiness rule makes sure a
failure keeps the pod unready. Workers and jobs must read the same Secret key as
the API (`deploy.inheritance`).

Reports contain target ids, workload ids, check ids, reason codes, levels, the
names of OpenMed settings, issue and pull-request numbers and digests. They never
contain setting values, Secret names, hosts, paths or file contents.

## Lane targets and recorded findings

`deploy/posture/lane.yaml` lists every rendering and the blocking findings it is
expected to produce. Paths are relative to the lane file and every value is
synthetic.

| Target kind | Renderings |
|---|---|
| shipped | chart defaults, `ci-values.yaml`, the Journey profile, the Compose bundle |
| example | `deploy/posture/helm/production.yaml`, alone and with the Journey profile |
| negative control | one deliberate defect each: development profile, authentication off, wildcard trusted host, plaintext CORS origin, contradictory and insecure overrides, no resources, readiness on `/livez`, a root privileged container, a Journey API that diverges from its worker, and the Compose bundle with `OPENMED_PROFILE=dev` or a wildcard host |

Each recorded finding says why it is expected:

* `fail_closed`: the shipped defaults cannot carry credentials, so
  `auth.enabled` fails and the service stays unready until the operator
  configures authentication.
* `known`: a gap owned by an open pull request (`owner_pr`, `issue`). Today
  these are the API security contexts of the default chart (#4443), Journey
  principal binding and encryption at rest (#4437, #4441) and plaintext
  PostgreSQL in the Compose bundle (#4146).
* `negative_control`: the defect a negative-control fixture introduces. A
  negative-control target must record at least one.

A new finding fails the lane. So does a recorded finding that no longer occurs:
when the owning change lands, remove its entries in the same pull request.

## Production overlay

`deploy/posture/helm/production.yaml` is the validated example for a production
Helm install: it configures authentication from a Secret reference, an explicit
trusted host, an admission limit, an explicit served-model allowlist and
hardened security contexts. It produces no blocking findings:

```bash
helm upgrade --install openmed-service deploy/helm/openmed-service \
  --namespace openmed \
  --values deploy/posture/helm/production.yaml
```

Create the referenced `openmed-service-auth` Secret with the API-key hashes
first. With `journey.enabled`, Journey principal binding and encryption at rest
still block until #3549 and #3550 ship.

## Extending the lane

A change that alters the chart, the Compose bundle or a security check updates
the lane in the same pull request: add or remove recorded findings, add a
negative control for a new production-critical value, or register a new
deployment rule in `openmed.deploy.posture.DEPLOY_CHECKS`. New runtime checks
registered with `register_security_check()` are picked up automatically. Bump
`POSTURE_CONTRACT_ID` when a rule changes what it accepts.

The build-from-source `docker-compose.yml` at the repository root is a local
quickstart, not a production deployment profile, and is not part of the lane.
