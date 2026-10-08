# Private learning overview

OpenMed's private-learning contracts describe a federated training round as
offline, metadata-only records. Each contract validates one part of the round
without accepting participant identities, local examples, gradients, tensor
values, or network endpoints. This page explains how the existing pages fit
together and what they deliberately do not do.

## Round flow

A round moves through the states defined by the
[round lifecycle](../training/federated-round-lifecycle.md). The other contracts
attach to those states:

| Lifecycle state | Contract | What it validates |
| --- | --- | --- |
| `planned` | [Round scheduling](../training/federated-scheduling.md) | Ordered UTC windows for enrollment, update submission, aggregation, and evaluation |
| `collecting` | [Update metadata](../training/federated-update-metadata.md) | The declared structure of a dense adapter update against a coordinator-supplied policy |
| `collecting` | [Schema fingerprints](../training/federated-schema-fingerprints.md) | Whether two validated updates declare the same parameter schema |
| `evaluating` | [Aggregate metrics](../training/federated-metrics.md) | One aggregate measurement with clipping bounds, minimum-group suppression, and coarse participant bands |
| Any state | [Round status](../training/federated-round-status.md) | A deterministic operator summary with suppressed small counts and stable reason codes |

The lifecycle validates transitions only. It does not decide whether a quality
or privacy gate passes; `held` rounds return to `evaluating` for a new
decision, and `promoted` and `aborted` are terminal.

## Shared properties

- Records are immutable and serialize to canonical JSON, so equivalent inputs
  produce byte-identical artifacts.
- Parsers reject unknown fields, unsupported schema versions, and invalid
  cross-field combinations.
- Errors are categorical and do not echo submitted keys or values.
- No contract reads the system clock, performs network calls, or writes files.

## Supported surfaces

The Python library and CLI are the required trainer, client, coordinator, and
evaluation surfaces for the v3.3 release. The contracts on these pages are
available today through the `openmed.training` Python package. Their JSON
records are designed to stay interoperable with air-gapped and out-of-process
implementations.

Swift, Android, and REST training surfaces are outside v3.3 unless a dedicated
issue adds compatibility, privacy, and release-gate evidence.

## What these contracts do not claim

These pages describe validation of declared metadata. They do not:

- transport updates, aggregate tensors, or verify tensor bytes against a
  declared digest;
- add differential-privacy noise, choose privacy budgets, or run a privacy
  accountant;
- certify that an output is anonymous, de-identified, or compliant with any
  regulation or standard;
- decide whether a model is promoted.

Minimum-group suppression is an output control, not differential privacy. A
valid record means the metadata is well formed; it does not establish that a
round was private, safe, or useful.

## Adding private-learning pages

New private-learning pages live under `docs/private-learning/`, for example
`docs/private-learning/update-clipping.md`. When adding a page:

1. Add it to the **Private Learning** section in `mkdocs.yml`.
2. Add the same path, in the same position, to the `navigated` list in
   `docs/brand/system/publication.yml`.
3. Link it from the round-flow table above when it attaches to a lifecycle
   state.

The pages under `docs/training/federated-*.md` keep their existing paths so
inbound links continue to resolve.
