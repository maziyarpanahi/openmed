# V4.0 Secure Service Threat Model

This STRIDE-style model covers the OpenMed service boundaries that the v4.0
release must defend: the REST, GraphQL and gRPC transports, asynchronous jobs
and webhooks, Journey storage, model artifacts, deployment, and the
operational and release evidence those surfaces produce. It maps each threat
to the enforcing module and offline test on `master`, or to the open issue and
pull request that owns the missing control.

The secure exchange and recoverable service scope was planned as v3.4. It was
folded into v4.0 ([milestone](https://github.com/maziyarpanahi/openmed/milestone/19),
release plan [#4253](https://github.com/maziyarpanahi/openmed/issues/4253),
workstream [#3543](https://github.com/maziyarpanahi/openmed/issues/3543)), so
this page, the recovery drill and the release packet carry the v4.0 name. The
release evidence issue is
[#3603](https://github.com/maziyarpanahi/openmed/issues/3603).

This is a first-party, design-level review of the repository. It is not a
penetration test, certification, clinical validation, or a claim that any
deployment is safe. It must be re-reviewed independently before a production
security sign-off. Report a suspected bypass through the private
[security disclosure policy](./disclosure-policy.md) and never include real
patient data.

## Scope and method

In scope: the service package (`openmed/service/`), the Journey stores
(`openmed/structured/store/`), model integrity and audit primitives in
`openmed/core/`, the reference Compose and Helm deployment under `deploy/`,
and the v4.0 release gate (`openmed/eval/service_release.py`) and recovery
drill (`openmed/eval/service_recovery_drill.py`). Out of scope: an operator's
identity provider, network perimeter, object-store and database policy,
host hardening, and every clinical decision taken downstream.

Each row of the matrix below names one threat at one boundary. A **Control**
cites a module and the offline test that enforces it on `master`. A **Gap**
cites the issue and pull request that own the missing or partial control.
Many rows carry both: a limited control exists and the complete one is open
work. A gap is not a mitigation. The v4.0 release gate fails while any P0 gap
in the committed checklist is open.

## Assets

| Asset | Why it matters | Required property |
| --- | --- | --- |
| Request and response bodies | They carry clinical text, identifiers and redaction results. | Confidentiality in transit; never logged, traced, cached or copied into evidence. |
| Principals and credentials | API keys, JWT verifiers, mTLS identities and webhook secrets decide who may act. | Authenticity, rotation without restart, file-based custody, no echo in errors. |
| Journey artifacts and metadata | Longitudinal records, evidence locators and reversible surrogate mappings. | Integrity on every read, confidentiality at rest, purpose and consent binding. |
| Async job records and payloads | Queued work, progress, terminal events and webhook deliveries. | Owner binding, no raw payload persistence, exactly-once completion, replay safety. |
| Model artifacts | Weights, tokenizers and configuration the service activates. | Pinned digests, verified before activation, atomic activation and rollback. |
| Deployment configuration | Bind addresses, TLS material, security contexts, egress rules. | Fail-closed production profile; drift detected before admission. |
| Audit, metrics, traces, diagnostics | Operators rely on them to detect and reconstruct incidents. | Value-free content, tamper evidence, continuity across key rotation and retention. |
| Release evidence | The signed packet decides whether a candidate may be promoted. | Exact-candidate binding, signature, completeness, no prohibited values. |

## Adversaries

- **Network caller:** reaches a listening port without valid credentials, or
  replays and modifies captured requests.
- **Authenticated principal:** holds a valid credential and tries to read
  another principal's jobs or Journey records, or exceed purpose and consent.
- **Resource exhauster:** sends oversized, deeply nested, slow or numerous
  requests to deny service to others.
- **Callback target or relay:** receives, forges or replays webhook
  deliveries, or uses a configured URL to reach internal hosts.
- **Storage holder:** reads or alters disks, backups, database rows or object
  stores outside the service.
- **Artifact supplier:** substitutes model weights or configuration upstream
  or in the cache.
- **Misconfigured operator or automation:** starts production with
  development defaults, leaks secrets through the environment, or upgrades
  without a rehearsed migration.
- **Release forger:** edits, reorders, replays or omits release evidence to
  promote an unready candidate.

## Trust boundaries

| Boundary | What crosses it | Required separation |
| --- | --- | --- |
| SB1: client to REST | HTTP requests with clinical text and credentials. | Authenticate before routing, bound bodies and rate, never reflect values. |
| SB2: client to GraphQL | Query documents, variables and batches. | POST-only for PHI, document budgets, authorization of the whole document. |
| SB3: client to gRPC | Unary and streaming RPCs. | Loopback by default, TLS or mTLS, per-RPC authorization and budgets. |
| SB4: service to jobs and webhooks | Queued work, durable job state, outbound callbacks. | Owner binding, payload lifecycle, signed deliveries, egress policy. |
| SB5: service to Journey storage | SQLite, PostgreSQL and object-store records. | Verified reads, encryption at rest, principal and purpose binding, migrations. |
| SB6: artifact source to runtime | Model downloads, cache and activation. | Digest verification, transactional activation and rollback. |
| SB7: deployment to host and cluster | Bind addresses, secrets, containers, network egress. | Fail-closed production profile, non-root read-only containers, no egress by default. |
| SB8: service to operations and release | Logs, metrics, traces, audit, diagnostics, backups, release packets. | Value-free content, tamper evidence, continuity, signed exact-candidate evidence. |

## Threat and control matrix

STRIDE categories are Spoofing, Tampering, Repudiation, Information
disclosure, Denial of service and Elevation of privilege. SV identifiers are
stable for review. Residual risks R1 to R7 are listed after the matrix.

| ID | STRIDE | Threat and boundary | Control or tracked gap; limits |
| --- | --- | --- | --- |
| SV-01 | Spoofing | An unauthenticated caller reaches PHI-bearing REST routes at SB1. | Control — API-key and JWT middleware with route scopes, optional deny-by-default and an auth-failure rate limit: `openmed/service/auth.py`, `tests/unit/service/test_auth_middleware.py`. Control — mTLS client identities: `openmed/service/mtls.py`, `tests/unit/service/test_mtls_auth.py`. Gap — authentication is off unless configured; the fail-closed production profile is [#3552](https://github.com/maziyarpanahi/openmed/issues/3552) ([PR #4439](https://github.com/maziyarpanahi/openmed/pull/4439)); atomic verifier reload is [#3699](https://github.com/maziyarpanahi/openmed/issues/3699) ([PR #4364](https://github.com/maziyarpanahi/openmed/pull/4364)). |
| SV-02 | Tampering / Repudiation | A captured request is modified or replayed at SB1. | Control — HMAC-SHA256 request signatures with a bounded local replay window: `openmed/service/signing.py`, `tests/unit/service/test_hmac_signing_replay.py`. Gap — inbound enforcement on configured routes is [#3924](https://github.com/maziyarpanahi/openmed/issues/3924) ([PR #4200](https://github.com/maziyarpanahi/openmed/pull/4200)); the replay window is replica-local until [#3908](https://github.com/maziyarpanahi/openmed/issues/3908) ([PR #5338](https://github.com/maziyarpanahi/openmed/pull/5338)) (R2). |
| SV-03 | Denial of service | Oversized bodies, floods or bursts exhaust workers at SB1 and SB2. | Control — request-size enforcement for every REST and GraphQL route: `openmed/service/request_limits.py`, `tests/unit/service/test_request_limits.py`. Control — token-bucket throttling: `openmed/service/throttle.py`, `tests/unit/service/test_throttle.py`. Control — bounded admission and hysteretic shedding: `openmed/service/backpressure.py`, `tests/unit/service/test_backpressure_shedding.py`. Gap — body read time and connection limits are [#3905](https://github.com/maziyarpanahi/openmed/issues/3905) ([PR #5133](https://github.com/maziyarpanahi/openmed/pull/5133)); fair admission across principals is [#3701](https://github.com/maziyarpanahi/openmed/issues/3701) ([PR #5447](https://github.com/maziyarpanahi/openmed/pull/5447)). |
| SV-04 | Information disclosure | A browser on another origin, a spoofed Host header or a cached response exposes data at SB1. | Control — CORS and trusted-host policy: `openmed/service/security_headers.py`, `tests/unit/service/test_cors_trusted_host.py`. Gap — no-store, nosniff, referrer and framing headers are [#3912](https://github.com/maziyarpanahi/openmed/issues/3912) ([PR #4324](https://github.com/maziyarpanahi/openmed/pull/4324)); hiding docs and version metadata outside development is [#3911](https://github.com/maziyarpanahi/openmed/issues/3911) ([PR #4198](https://github.com/maziyarpanahi/openmed/pull/4198)). |
| SV-05 | Information disclosure | GraphQL GET places PHI-bearing documents in URLs, proxies and logs, and the IDE is reachable in production at SB2. | Control — the whole document is authorized before any resolver runs: `openmed/service/graphql_app.py`, `tests/unit/service/test_graphql_app.py`. Gap — POST-only operations and IDE hardening are P0 [#3544](https://github.com/maziyarpanahi/openmed/issues/3544) ([PR #3614](https://github.com/maziyarpanahi/openmed/pull/3614)). |
| SV-06 | Denial of service | Deep, aliased, fragment-cyclic or batched GraphQL documents exhaust resolvers at SB2. | Gap — document complexity, variable and execution budgets are [#3545](https://github.com/maziyarpanahi/openmed/issues/3545) ([PR #4438](https://github.com/maziyarpanahi/openmed/pull/4438)). |
| SV-07 | Spoofing / Information disclosure | gRPC listens on every interface without TLS (`[::]:50051`, insecure port) at SB3. | Control — request-selected model paths are denied before loading: `openmed/service/grpc_server.py`, `tests/unit/service/test_grpc_server.py`. Gap — loopback default and explicit TLS and mTLS credentials are P0 [#3546](https://github.com/maziyarpanahi/openmed/issues/3546) ([PR #4436](https://github.com/maziyarpanahi/openmed/pull/4436)); credential reload without dropping work is [#3580](https://github.com/maziyarpanahi/openmed/issues/3580) ([PR #5385](https://github.com/maziyarpanahi/openmed/pull/5385)). |
| SV-08 | Elevation of privilege | gRPC RPCs run without authentication, per-RPC authorization or budgets at SB3. | Gap — P0 [#3547](https://github.com/maziyarpanahi/openmed/issues/3547) ([PR #4822](https://github.com/maziyarpanahi/openmed/pull/4822)). Until it lands, the reference deployment must not expose gRPC beyond loopback (R1). |
| SV-09 | Tampering / Information disclosure | One transport returns fields, states or errors that another redacts or rejects, across SB1 to SB3. | Control — the offline REST contract gate keeps routes, required fields and stable error categories: `openmed/service/api_compatibility.py`, `tests/unit/service/test_api_compatibility.py`. Gap — the REST, GraphQL and gRPC protocol map and parity certification are [#3548](https://github.com/maziyarpanahi/openmed/issues/3548) ([PR #5457](https://github.com/maziyarpanahi/openmed/pull/5457)); breaking SDL and protobuf classification is [#3921](https://github.com/maziyarpanahi/openmed/issues/3921) ([PR #4444](https://github.com/maziyarpanahi/openmed/pull/4444)). |
| SV-10 | Information disclosure | Raw submitted text or redacted output persists in job state or webhook payloads at SB4. | Control — the local job store persists metadata only and rejects damaged files without overwriting them: `openmed/service/jobs.py`, `tests/unit/service/test_jobs_api.py`. Gap — the payload envelope and retention lifecycle are [#3572](https://github.com/maziyarpanahi/openmed/issues/3572) ([PR #4918](https://github.com/maziyarpanahi/openmed/pull/4918)); keyed span and document digests are [#3918](https://github.com/maziyarpanahi/openmed/issues/3918) ([PR #5213](https://github.com/maziyarpanahi/openmed/pull/5213)). |
| SV-11 | Elevation of privilege | Another principal reads a job, bulk export or ingestion record by guessing its identifier at SB4. | Gap — owner binding is [#3907](https://github.com/maziyarpanahi/openmed/issues/3907) ([PR #5088](https://github.com/maziyarpanahi/openmed/pull/5088)); submission bounds are [#3910](https://github.com/maziyarpanahi/openmed/issues/3910) ([PR #5206](https://github.com/maziyarpanahi/openmed/pull/5206)). |
| SV-12 | Tampering / Denial of service | A worker crash loses, duplicates or endlessly retries jobs at SB4. | Control — committed job metadata survives an abrupt worker stop and a reopen, and failed writes roll back; the recovery drill exercises both: `openmed/eval/service_recovery_drill.py`, `tests/unit/eval/test_service_recovery_drill.py`. Gap — leases, checkpoints, poison quarantine and replay are P0 [#3574](https://github.com/maziyarpanahi/openmed/issues/3574) ([PR #5446](https://github.com/maziyarpanahi/openmed/pull/5446)); the drill records them as `not_available`. Draining is [#3576](https://github.com/maziyarpanahi/openmed/issues/3576) ([PR #5451](https://github.com/maziyarpanahi/openmed/pull/5451)). |
| SV-13 | Spoofing / Tampering | A receiver accepts a forged or replayed webhook delivery at SB4. | Control — terminal-event deliveries are HMAC-signed: `openmed/service/webhooks.py`, `tests/unit/service/test_webhooks.py`. Gap — delivery and attempt identifiers, key rotation and a replay ledger are [#3573](https://github.com/maziyarpanahi/openmed/issues/3573) ([PR #5444](https://github.com/maziyarpanahi/openmed/pull/5444)). |
| SV-14 | Information disclosure / Elevation of privilege | A webhook URL, remote inference endpoint or connector reaches internal hosts or exfiltrates data at SB4 and SB7. | Gap — a deny-by-default egress policy is [#3581](https://github.com/maziyarpanahi/openmed/issues/3581) ([PR #5194](https://github.com/maziyarpanahi/openmed/pull/5194)); the OS-enforced no-egress gate and attestation are [#3551](https://github.com/maziyarpanahi/openmed/issues/3551) ([PR #5455](https://github.com/maziyarpanahi/openmed/pull/5455)). The release gate requires the attestation digest. |
| SV-15 | Elevation of privilege / Information disclosure | An authenticated principal reads Journey resources outside its role, purpose, consent or export policy at SB5. | Control — role, purpose, consent, export and field policy, bounded pages and query-bound cursors: `openmed/service/journey_resources.py`, `tests/unit/service/test_journey_resources.py`. Gap — binding the authenticated principal to that policy is P0 [#3549](https://github.com/maziyarpanahi/openmed/issues/3549) ([PR #4437](https://github.com/maziyarpanahi/openmed/pull/4437)). |
| SV-16 | Information disclosure | A stolen disk, backup or database copy exposes Journey artifacts and metadata at SB5. | Control — owner-only file modes and content-addressed, atomically written blobs: `openmed/structured/store/local.py`, `tests/unit/structured/store/test_local.py`. Gap — application-layer encryption is P0 [#3550](https://github.com/maziyarpanahi/openmed/issues/3550) ([PR #4441](https://github.com/maziyarpanahi/openmed/pull/4441)); verified PostgreSQL TLS is [#3915](https://github.com/maziyarpanahi/openmed/issues/3915) ([PR #4146](https://github.com/maziyarpanahi/openmed/pull/4146)). |
| SV-17 | Tampering | A corrupted or substituted Journey blob is served or silently replaced at SB5. | Control — every read revalidates the content digest and a corrupted blob is fenced rather than overwritten: `openmed/structured/store/local.py`, `tests/unit/structured/store/test_local.py`; the drill's corrupted-state scenario checks it: `openmed/eval/service_recovery_drill.py`, `tests/unit/eval/test_service_recovery_drill.py`. Gap — encrypted backup and verified restore to an inactive target are [#3577](https://github.com/maziyarpanahi/openmed/issues/3577) ([PR #5452](https://github.com/maziyarpanahi/openmed/pull/5452)). |
| SV-18 | Tampering / Denial of service | A failed or mixed-version migration corrupts durable state at SB5. | Control — deterministic, checksummed migrations and refusal of newer or drifted schemas: `openmed/structured/store/migrations.py`, `tests/unit/structured/store/test_local.py`. Gap — the durable-schema registry and rehearsal matrix are [#3578](https://github.com/maziyarpanahi/openmed/issues/3578) ([PR #5458](https://github.com/maziyarpanahi/openmed/pull/5458)); mixed-version upgrades are [#3579](https://github.com/maziyarpanahi/openmed/issues/3579) ([PR #5456](https://github.com/maziyarpanahi/openmed/pull/5456)). |
| SV-19 | Tampering / Elevation of privilege | Substituted model weights or configuration are activated at SB6. | Control — offline integrity verification of cached artifacts against pinned digests: `openmed/core/model_integrity.py`, `tests/unit/core/test_model_integrity.py`. Gap — transactional download, verification, activation and rollback are P0 [#3575](https://github.com/maziyarpanahi/openmed/issues/3575) ([PR #5066](https://github.com/maziyarpanahi/openmed/pull/5066)). |
| SV-20 | Elevation of privilege | Production starts with development defaults (authentication off, wildcard origins, docs exposed) at SB7. | Gap — the fail-closed production security-profile self-check is P0 [#3552](https://github.com/maziyarpanahi/openmed/issues/3552) ([PR #4439](https://github.com/maziyarpanahi/openmed/pull/4439)); Compose and Helm posture verification is [#3706](https://github.com/maziyarpanahi/openmed/issues/3706) ([PR #5461](https://github.com/maziyarpanahi/openmed/pull/5461)). The release gate requires the security-profile report digest. |
| SV-21 | Information disclosure | Credentials leak through environment variables, process listings or crash dumps at SB7. | Gap — mounted secret files are [#3914](https://github.com/maziyarpanahi/openmed/issues/3914) ([PR #4440](https://github.com/maziyarpanahi/openmed/pull/4440)); verifier and TLS reload are [#3699](https://github.com/maziyarpanahi/openmed/issues/3699) ([PR #4364](https://github.com/maziyarpanahi/openmed/pull/4364)) and [#3580](https://github.com/maziyarpanahi/openmed/issues/3580) ([PR #5385](https://github.com/maziyarpanahi/openmed/pull/5385)). |
| SV-22 | Elevation of privilege | A container runs as root with a writable root filesystem at SB7. | Control — the chart renders and is tested offline: `deploy/helm/openmed-service/values.yaml`, `tests/unit/deploy/test_helm_chart.py`; its `securityContext` is empty on `master`. Gap — non-root, read-only security contexts are [#3916](https://github.com/maziyarpanahi/openmed/issues/3916) ([PR #4443](https://github.com/maziyarpanahi/openmed/pull/4443)). |
| SV-23 | Information disclosure | Logs, traces, metrics or alerts carry PHI or request bodies at SB8. | Control — structured access logs without bodies: `openmed/service/logging.py`, `tests/unit/service/test_request_logging.py`. Control — span attributes without PHI: `openmed/service/tracing.py`, `tests/unit/service/test_tracing_no_phi.py`. Control — closed-vocabulary operational telemetry: `openmed/service/operational.py`, `tests/unit/service/test_operational_observability.py`. Gap — the bounded no-PHI diagnostic bundle is [#3600](https://github.com/maziyarpanahi/openmed/issues/3600) ([PR #5460](https://github.com/maziyarpanahi/openmed/pull/5460)). |
| SV-24 | Repudiation / Tampering | Audit evidence is truncated, reordered or broken by key rotation or retention at SB8. | Control — tamper-evident, value-free audit chains: `openmed/core/audit_chain.py`, `tests/unit/core/test_audit_chain.py`. Control — caller-owned audit key rotation: `openmed/core/audit_key_rotation.py`, `tests/unit/core/test_audit_key_rotation.py`. Gap — the continuity manifest, signer epochs and tombstones are [#3599](https://github.com/maziyarpanahi/openmed/issues/3599) ([PR #5459](https://github.com/maziyarpanahi/openmed/pull/5459)); the drill records continuity verification as `not_available`. |
| SV-25 | Denial of service | A model-backend or network outage cascades into unbounded retries at SB4 and SB6. | Control — bounded retries and a circuit breaker with a single recovery probe: `openmed/service/resilience.py`, `tests/unit/service/test_circuit_breaker.py`; the drill's network scenario opens, sheds, probes and closes it: `openmed/eval/service_recovery_drill.py`, `tests/unit/eval/test_service_recovery_drill.py`. Gap — the seeded cross-component fault matrix is [#3601](https://github.com/maziyarpanahi/openmed/issues/3601) ([PR #4448](https://github.com/maziyarpanahi/openmed/pull/4448)). |
| SV-26 | Denial of service / Tampering | Shutdown, drain or upgrade drops in-flight requests, streams or jobs across SB1 to SB4. | Control — readiness and liveness probes report draining: `openmed/service/app.py`, `tests/unit/service/test_readiness_liveness.py`. Gap — one drain coordinator for every transport and worker is [#3576](https://github.com/maziyarpanahi/openmed/issues/3576) ([PR #5451](https://github.com/maziyarpanahi/openmed/pull/5451)); zero-downtime upgrades are [#3579](https://github.com/maziyarpanahi/openmed/issues/3579) ([PR #5456](https://github.com/maziyarpanahi/openmed/pull/5456)). |
| SV-27 | Denial of service | An air-gapped site cannot reinstall or recover the service without network access at SB7. | Control — portable air-gapped installation kits are built and verified offline: `openmed/cli/airgap.py`, `tests/unit/cli/test_airgap_cli.py`. Gap — the air-gapped recovery rehearsal is [#3602](https://github.com/maziyarpanahi/openmed/issues/3602). |
| SV-28 | Tampering / Repudiation | Release evidence is changed, omitted, reordered, replayed for another candidate or left unsigned at SB8. | Control — the v4.0 gate binds the exact candidate commit, verifies every evidence digest, fails on any open or under-evidenced P0, and signs a canonical packet; the offline verifier returns stable reason codes: `openmed/eval/service_release.py`, `tests/unit/eval/test_service_release.py`. Limits: the signature is symmetric HMAC with a caller-held key (R6). |
| SV-29 | Information disclosure | Release evidence or documentation carries paths, identifiers, credentials or planted sentinels at SB8. | Control — the gate scans every retained evidence file and the packet for prohibited value classes and planted sentinels and records only class names: `openmed/eval/service_release.py`, `tests/unit/eval/test_service_release.py`; this page is scanned too: `tests/unit/test_service_threat_model.py`. |

## Assumptions

- The reference deployment terminates public TLS at an operator-controlled
  proxy or uses the service's own TLS once
  [#3546](https://github.com/maziyarpanahi/openmed/issues/3546) lands.
- Secrets, signing keys and surrogate-vault keys come from an operator secret
  manager and never from committed files.
- Hosts, container runtimes and databases are patched and access-controlled
  by the operator.
- Model artifacts are downloaded explicitly; there is no mandatory network
  call after download and no telemetry by default.
- Apple Foundation Models paths stay on-device and are not part of this
  service boundary.

## Unsupported deployments

- gRPC exposed beyond loopback before
  [#3546](https://github.com/maziyarpanahi/openmed/issues/3546) and
  [#3547](https://github.com/maziyarpanahi/openmed/issues/3547) land.
- More than one replica sharing replica-local replay, throttle or job state
  ([#3908](https://github.com/maziyarpanahi/openmed/issues/3908)).
- Any cloud LLM fallback for PHI workflows.
- Production with authentication disabled, wildcard origins or the GraphQL IDE
  enabled.
- Release promotion from a packet that is not `READY` or does not verify
  offline against the exact candidate.

## Residual risks

| ID | Residual risk | Owner of the condition |
| --- | --- | --- |
| R1 | Transports without completed TLS and authorization rely on network placement. | Operator, until [#3546](https://github.com/maziyarpanahi/openmed/issues/3546) and [#3547](https://github.com/maziyarpanahi/openmed/issues/3547) close. |
| R2 | Replay windows, throttles and job state are replica-local. | Operator; single-replica deployments only until [#3908](https://github.com/maziyarpanahi/openmed/issues/3908) closes. |
| R3 | Recovery of durable state without encrypted backup and verified restore is manual. | Operator, until [#3577](https://github.com/maziyarpanahi/openmed/issues/3577) closes. |
| R4 | Data at rest relies on host and volume encryption. | Operator, until [#3550](https://github.com/maziyarpanahi/openmed/issues/3550) closes. |
| R5 | Outbound callbacks and connectors are not yet egress-restricted by the service. | Operator network policy, until [#3581](https://github.com/maziyarpanahi/openmed/issues/3581) and [#3551](https://github.com/maziyarpanahi/openmed/issues/3551) close. |
| R6 | The release signature is symmetric HMAC; anyone holding the key can sign. | Release operator key custody. |
| R7 | This review is first-party and has not been independently reviewed. | Maintainers, before production sign-off. |

## Release evidence

The [v4.0 service release gate](../release/v4.0-service-release-gate.md)
binds this page by digest (`threat_model` evidence) together with the
security-profile report, no-egress attestation, recovery-drill report,
migration rehearsal matrix, protocol map, synthetic fixtures, the controlled
failure-code inventory and the P0 status. The committed checklist
`openmed/eval/data/service_release_p0.json` lists every v4.0 service P0 and
the evidence kinds it requires; the gate is `NOT_READY` while any of them is
open, moved, or missing evidence.

## Related guidance

- [Redactor threat model](./threat-model.md)
- [V3 Journey threat model](./journey-threat-model.md)
- [V3 Journey release gate](../release/v3.0-journey-release-gate.md)
- [Security and disclosure policy](./disclosure-policy.md)
