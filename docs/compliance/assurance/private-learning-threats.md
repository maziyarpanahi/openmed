# Private-learning threat reports

Federated and synthetic-data deployments face different threats depending on
coordinator trust, client custody, secure aggregation, differential privacy,
cryptography, network topology, data flows, and artifact retention.
`openmed.compliance.assurance.private_learning_threats` turns a closed
deployment declaration into a structured threat report. A versioned rule
table maps the declaration to findings with stable threat identifiers,
controlled status and reason codes, and the OpenMed modules and pages that
support each control.

> This report maps declared deployment choices to private-learning threats
> with fixed rules. It is not a certification, audit opinion, or legal
> advice, and it does not verify that any declared control is implemented
> or configured correctly.

Every report carries that sentence in its `disclaimer` field. A report never
states that a deployment is cleared for use; it states which threats remain
open under the declared assumptions.

## Generate a report

```python
from openmed.compliance.assurance.private_learning_threats import (
    UNRESOLVED,
    build_private_learning_threat_report,
)

# Synthetic declaration: a cross-silo federation with a curious coordinator.
report = build_private_learning_threat_report(
    {
        "topology": {"training_mode": "federated", "federation_scope": "cross_silo"},
        "trust": {"coordinator": "honest_but_curious"},
        "secure_aggregation": {"protocol": "none"},
    }
)

open_threats = report.to_dict()["unresolved_threat_ids"]
assert all(
    item.status == UNRESOLVED
    for item in report.findings
    if item.threat_id in open_threats
)
json_text = report.to_json()
markdown_text = report.to_markdown()
```

Generation is local, deterministic, and makes no network call. The same
declaration always produces byte-identical JSON and Markdown regardless of
key order. `declaration_digest` is the SHA-256 digest of the canonical
declaration, including every `undeclared` field, and `report_digest` covers
every report field except itself.

## Declaration schema

The declaration is a nested mapping of sections to fields. Every value is one
of the enumerated strings below or `undeclared`. An omitted section or field
is recorded as `undeclared`. An optional top-level `schema_version` must equal
`openmed.compliance.private_learning_deployment.v1`.

| Field | Allowed values |
| --- | --- |
| `topology.training_mode` | `centralized`, `federated`, `hybrid` |
| `topology.federation_scope` | `not_federated`, `cross_silo`, `cross_device` |
| `trust.coordinator` | `trusted`, `honest_but_curious`, `malicious` |
| `trust.clients` | `trusted`, `honest_but_curious`, `malicious` |
| `trust.client_custody` | `site_operated`, `shared_infrastructure`, `third_party_hosted` |
| `secure_aggregation.protocol` | `none`, `pairwise_masking`, `threshold_secret_sharing`, `trusted_execution_environment` |
| `secure_aggregation.collusion_assumption` | `no_collusion_assumed`, `below_threshold`, `may_reach_threshold` |
| `aggregation.minimum_cohort` | `none`, `enforced` |
| `aggregation.update_clipping` | `none`, `enforced` |
| `aggregation.update_validation` | `none`, `schema_allowlist` |
| `aggregation.robust_aggregation` | `none`, `norm_bounding`, `byzantine_robust` |
| `differential_privacy.noise_location` | `none`, `client_local`, `distributed`, `central` |
| `differential_privacy.privacy_unit` | `record_level`, `client_level` |
| `differential_privacy.accountant` | `none`, `basic_composition`, `advanced_composition` |
| `cryptography.transport` | `none`, `tls`, `mutual_tls` |
| `cryptography.update_signing` | `none`, `per_client_signatures` |
| `cryptography.replay_protection` | `none`, `round_bound_nonces` |
| `cryptography.at_rest_encryption` | `none`, `encrypted` |
| `network.topology` | `air_gapped`, `private_network`, `vpn`, `public_internet` |
| `data_flows.raw_records_leave_site` | `never`, `pseudonymized_only`, `raw` |
| `data_flows.model_release` | `internal_only`, `external_partners`, `public` |
| `data_flows.metric_release` | `none`, `aggregate_envelopes`, `per_client` |
| `retention.client_updates` | `not_retained`, `round_scoped`, `retained` |
| `retention.checkpoints` | `not_retained`, `retained` |
| `retention.synthetic_data` | `not_retained`, `retained` |
| `retention.logs` | `none`, `aggregate_only`, `content_bearing` |
| `retention.schedule` | `none`, `enforced` |
| `synthetic_data.release` | `none`, `internal`, `external` |
| `synthetic_data.generator_privacy` | `none`, `differentially_private` |
| `audits.membership_inference` | `none`, `self_test` |
| `audits.synthetic_memorization` | `none`, `release_gate` |

A `centralized` deployment must declare `not_federated` (or leave the scope
undeclared), and a `federated` or `hybrid` deployment must not declare
`not_federated`. Inconsistent topology is rejected.

## Threat catalog

Rules version `openmed.compliance.private_learning_threat_rules.v1` evaluates
these threats in a fixed order. Threat identifiers are stable across rule
versions; a rule change ships under a new rules version.

| ID | Threat | Applies when | Main inputs |
| --- | --- | --- | --- |
| `PLT-001` | `update_inversion` | Federated or hybrid training | Coordinator trust, client custody, secure aggregation, minimum cohort, clipping, noise location and accounting |
| `PLT-002` | `poisoning_backdoor` | Federated or hybrid training | Client trust, client custody, federation scope, robust aggregation, update validation, update signing |
| `PLT-003` | `collusion` | Federated or hybrid training | Secure aggregation and its collusion assumption, coordinator trust, federation scope, minimum cohort, noise |
| `PLT-004` | `membership_inference` | Always | Model release audience, noise and accounting, membership audit |
| `PLT-005` | `synthetic_memorization` | A synthetic release is declared | Generator privacy, memorization release gate |
| `PLT-006` | `retention_exposure` | Any artifact or log is retained | Retained updates, checkpoints, synthetic data, logs, retention schedule, at-rest encryption |
| `PLT-007` | `transport_interception` | Updates or records cross a site boundary | Transport, network topology, record egress |
| `PLT-008` | `update_replay` | Federated or hybrid training | Round-bound nonces, update signing |
| `PLT-009` | `metric_disclosure` | Federated metrics are released | Metric release shape, minimum cohort |

Representative rules:

- A `malicious` coordinator without secure aggregation or client-side noise
  leaves `update_inversion` `unresolved` at `critical` severity with
  `individual_updates_exposed_to_coordinator`. Secure aggregation alone only
  partially mitigates it, because a malicious coordinator can manipulate the
  cohort to isolate one update.
- Noise added by the coordinator (`central`) does not protect updates from a
  coordinator that is not trusted.
- Secure aggregation requires a declared collusion assumption. `below_threshold`
  is reported as an unverified assumption (`partially_mitigated`);
  `may_reach_threshold`, or a malicious coordinator that can enroll sybil
  clients in a `cross_device` federation, leaves `collusion` `unresolved`.
- Robust aggregation never fully mitigates targeted backdoors, and secure
  aggregation limits inspection of individual updates.
- A synthetic release without generator privacy or a memorization release
  gate leaves `synthetic_memorization` `unresolved`.
- Content-bearing logs leave `retention_exposure` `unresolved` at `critical`
  severity. Client updates retained beyond their round stay at best
  `partially_mitigated`.

Mitigations that rely on differential privacy also carry
`privacy_parameters_not_evaluated`: the declaration records where noise is
added and how it is accounted, not epsilon, delta, or clipping bounds.

## Findings

Each finding contains:

| Field | Meaning |
| --- | --- |
| `threat_id`, `threat` | Stable identifier and name from the catalog |
| `status` | `mitigated_by_declared_control`, `partially_mitigated`, `unresolved`, or `not_applicable` |
| `severity` | `critical`, `high`, `medium`, `low`, or `none` for a threat that does not apply |
| `reason_codes` | Sorted codes from the closed `REASON_CODES` vocabulary |
| `undeclared_inputs` | Schema paths whose absence left the finding `unresolved` |
| `controls` | Each relevant control, whether the declaration declares it, and its OpenMed references |

Undeclared inputs never count as a control. When a rule needs an undeclared
input, the finding is `unresolved` with the `undeclared_input` reason code and
lists the missing paths. An empty declaration therefore produces only
`unresolved` findings. The report also lists `unresolved_threat_ids` and a
`status_counts` summary.

## Control references

`openmed_support` is `module_reference` when OpenMed ships a supporting module
or page, and `deployment_provided` when the deployment must supply the control
itself. A reference identifies supporting code; it does not show that the
deployment uses that code correctly.

| Control | OpenMed references |
| --- | --- |
| `aggregate_metric_envelopes` | `openmed/training/federated_metrics.py`, [federated metrics](../../training/federated-metrics.md) |
| `aggregate_only_logging` | [no raw PHI logging](../../security/no-raw-phi-logging.md) |
| `at_rest_encryption` | Deployment provided |
| `differential_privacy` | `openmed/risk/differential_privacy.py`, [structured risk lab](../../privacy/structured-risk-lab.md) |
| `generator_privacy` | Deployment provided |
| `membership_audit` | `openmed/risk/membership_inference.py`, `openmed/risk/membership.py`, [structured risk lab](../../privacy/structured-risk-lab.md) |
| `memorization_audit` | `openmed/eval/synthetic_memorization.py`, [synthetic memorization](../../evaluation/synthetic-memorization.md) |
| `minimum_cohort` | `openmed/training/federated_status.py`, `openmed/training/federated_metrics.py`, [federated round status](../../training/federated-round-status.md) |
| `network_isolation` | Deployment provided |
| `privacy_accounting` | `openmed/risk/differential_privacy.py`, `openmed/risk/privacy_budget.py`, [privacy budget](../../security/privacy-budget.md) |
| `replay_protection` | `openmed/training/federated_round.py`, `openmed/training/federated_schedule.py`, [round lifecycle](../../training/federated-round-lifecycle.md), [scheduling](../../training/federated-scheduling.md) |
| `retention_schedule` | `openmed/risk/audit_retention.py`, `openmed/risk/deletion_plan.py`, [audit retention](../../security/audit-retention.md), [deletion plans](../../security/deletion-plans.md) |
| `robust_aggregation` | Deployment provided |
| `secure_aggregation` | Deployment provided |
| `transport_security` | Deployment provided |
| `update_clipping` | `openmed/training/federated_update_metadata.py`, [update metadata](../../training/federated-update-metadata.md) |
| `update_schema_validation` | `openmed/training/federated_update_metadata.py`, `openmed/training/federated_schema_fingerprint.py`, [update metadata](../../training/federated-update-metadata.md), [schema fingerprints](../../training/federated-schema-fingerprints.md) |
| `update_signing` | `openmed/core/key_custody.py`, [key custody metadata](../../security/key-custody-metadata.md) |

OpenMed's differential-privacy helpers calibrate noise and account for
aggregate releases; they are not a private training optimizer. Federated
update metadata records a clipping declaration but does not inspect tensors.
A test checks that every referenced path exists in the repository, so a moved
or deleted control module fails the suite instead of leaving a dangling
reference.

## Privacy boundary

The declaration has no field for site names, hostnames, paths, endpoints,
participant counts, or free text. Unknown keys at any level are rejected, and
values outside the enumerated vocabulary are rejected. Errors are fixed
`field: code` messages such as `declaration: unknown_field` or
`network.topology: invalid_value`; they never echo the submitted key or value
and are raised without a chained cause. `repr()` of a declaration or report
shows only its digest.

## Limitations

- The report reflects declared choices only. It does not inspect code,
  configuration, keys, traffic, or model weights.
- Rules are coarse: they do not weigh privacy budgets, cohort sizes, or
  attacker capabilities beyond the declared trust levels.
- The report is technical evidence for a site review. Site governance,
  security review, and legal review remain the deployer's responsibility.

See also [federated update metadata](../../training/federated-update-metadata.md),
[federated aggregate metrics](../../training/federated-metrics.md), and
[synthetic memorization](../../evaluation/synthetic-memorization.md).
