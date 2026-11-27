# Private-learning stability

Private learning is an **experimental** capability. OpenMed ships its
metadata contracts as stable APIs, while the federated execution, differential
privacy, secure aggregation and federation governance modules keep the
experimental label until independent evidence closes the general-availability
(GA) criteria below.

!!! warning "A version tag does not graduate the API"
    Outside parties gate the stability label. A release number, a milestone or a
    shipped example never promotes these modules to stable on its own.

## Current level

| Surface | Level | Scope |
| --- | --- | --- |
| Private-learning metadata contracts | Stable | Seven metadata-only modules released between v2.3.0 and v3.1.0 |
| Private-learning execution, DP, secure aggregation and federation governance | Experimental | Federated and differentially private adapter training, secure aggregation, federation identity and governance, independent evaluation and reproducible promotion evidence, tracked by the [v4.2 private-learning release plan](https://github.com/maziyarpanahi/openmed/issues/4252) as an optional experimental capability |

Both rows are local-first. Neither row makes a network call part of its
defaults, and neither row stores participant identifiers, local metrics, model
updates or raw identifiers in the artifacts it releases.

## What stays stable

The metadata contracts keep their normal SemVer guarantees: a name exported
from these modules changes only through the
[deprecation policy](../release/semver-and-channels.md) and a migration guide.
They were released as follows.

| Module | Contract | First released | Guide |
| --- | --- | --- | --- |
| `openmed/training/federated_round.py` | Round lifecycle state machine, schema `openmed.training.federated_round.v1` | v2.3.0 | [Federated round lifecycle](federated-round-lifecycle.md) |
| `openmed/training/federated_schedule.py` | Ordered schedule windows, schema `openmed.training.federated_schedule.v1` | v2.5.0 | [Federated round scheduling](federated-scheduling.md) |
| `openmed/training/federated_status.py` | Operator status summary with suppression, schema `openmed.training.federated_status.v1` | v2.5.0 | [Federated round status](federated-round-status.md) |
| `openmed/training/federated_update_metadata.py` | Dense update metadata envelope, schema `openmed.training.federated_update_metadata.v1` | v2.5.0 | [Federated update metadata](federated-update-metadata.md) |
| `openmed/training/federated_metrics.py` | Aggregate metric envelopes, schema `openmed.training.federated_metric.v1` | v2.5.0 | [Federated aggregate metrics](federated-metrics.md) |
| `openmed/training/federated_schema_fingerprint.py` | Domain-separated update schema fingerprints, `openmed.training.federated_schema_fingerprint.v1` | v3.0.0 | [Federated update schema fingerprints](federated-schema-fingerprints.md) |
| `openmed/training/private_training_schemas.py` | Catalog of self-contained Draft 2020-12 schemas with `urn:openmed:training:` identifiers | v3.1.0 | [Private-training JSON schemas](private-training-schemas.md) |

These modules validate and serialize metadata. They do not decide whether a
privacy or quality gate passes, and they do not run training.

## GA criteria

Three outside gates decide when private learning becomes stable. All three are
open, and none of them can be closed from inside this repository alone.

| Criterion | Issue | What closes it |
| --- | --- | --- |
| Sealed evaluation at independent clinical sites | [#2876](https://github.com/maziyarpanahi/openmed/issues/2876) | A signed immutable model evaluated against local holdouts with no network egress, returning signed aggregates, uncertainty and environment digests while cases stay at the evaluator |
| Reviewed threat model and protocol profile | [#3209](https://github.com/maziyarpanahi/openmed/issues/3209) | Explicit adversaries, trust assumptions, approved cryptographic primitives, key lifecycle and review criteria, with residual risks and validation evidence named for every security claim |
| Pre-registered DP utility study | [#3212](https://github.com/maziyarpanahi/openmed/issues/3212) | Pre-registered epsilon and delta bands, accounting assumptions, tasks, datasets, seeds, utility and leakage metrics, language and fairness slices, stopping rules and non-compensable gates |

## v7.0 tracking

The three criteria and the evidence that verifies them are tracked to the
**v7.0** release milestone:

- [#4455](https://github.com/maziyarpanahi/openmed/issues/4455) declares the
  evidence each GA item needs and how it is verified.
- [#4456](https://github.com/maziyarpanahi/openmed/issues/4456) runs that
  evidence through its verifiers before the gate may report a stable label.
- [#4465](https://github.com/maziyarpanahi/openmed/issues/4465) blocks a stable
  marker in releases that lack re-verified GA evidence for the exact commit.
- [#4471](https://github.com/maziyarpanahi/openmed/issues/4471) promotes the
  federated API from experimental to stable in the public-API baseline.

The label follows the evidence: it changes after the verifiers accept evidence
for the release commit, not when the version number changes. A release that
still carries the experimental marker passes those checks; a release that
claims stable without the evidence fails them.

## What the label does not claim

- It is not clinical validation. The
  [clinical validation protocol](../clinical-validation-protocol.md) governs
  studies run against user-supplied labeled data, and no private-learning
  artifact has completed one.
- It is not a compliance certification. OpenMed does not self-certify; see the
  [compliance posture](../compliance.md). A threat model or a privacy budget
  estimate is evidence for your own review, not a substitute for it.
- It is not a production federated-system guarantee. Secure aggregation,
  collusion resistance, dropout handling and federation governance are under
  review in [#3209](https://github.com/maziyarpanahi/openmed/issues/3209) and
  carry no formal-verification claim.
- It is not a model-quality claim. Federated utility numbers stay qualified by
  their site, slice and budget, and the DP utility study in
  [#3212](https://github.com/maziyarpanahi/openmed/issues/3212) is the gate for
  publishing them as stable.

## Using an experimental surface

Pin the package, keep private learning out of a clinical production path until
your own validation covers the deployment, and read the metadata contract
guides above before relying on an artifact. Report a leak or a bypass privately
through the [disclosure policy](../security/disclosure-policy.md), never in a
public issue.
