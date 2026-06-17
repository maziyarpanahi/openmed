# Release Streams, SemVer, and Channels

OpenMed uses two release streams with different blast radii.

## Stream A: Model Artifacts

Model artifacts are data. A bad checkpoint affects one model entry and can be
rolled back by repointing the manifest to the last green artifact.

- Versioning: repository suffix or artifact revision plus a reproducibility hash.
- Cadence: daily-capable once the release gates and manifest automation are in place.
- Gate owner: release engineering plus the evaluation gate.
- Rollback: manifest pointer flip, regenerated cards, and a tracking issue.

## Stream B: Library and SDK

The `openmed` wheel and source distribution are code. A bad library release can
break every downstream install, so it follows SemVer and moves more slowly.

- Versioning: `MAJOR.MINOR.PATCH`.
- Cadence: patch daily-to-weekly, minor monthly, major by milestone.
- Gate owner: release engineering with maintainer sign-off for minor and major changes.
- Rollback: forward-fix with the next patch; yanking remains a manual registry action.

## Channels

| Channel | Selector | Contents | Cadence | Audience |
|---|---|---|---|---|
| Nightly / edge | `pip install openmed --pre` | Latest green code and gated model pins | every green merge or daily | early adopters and internal evaluation |
| Stable | `pip install openmed` | Full golden-suite pass and canary-ready pins | patch daily-to-weekly, minor monthly | default users |
| LTS | `pip install "openmed==1.6.*"` | Security and recall-backstop fixes only | as needed for 12 months | regulated deployments |

Nightly builds use PEP 440 development releases such as `1.6.0.devN`.
Release candidates such as `1.6.0rc1` are reserved for pre-stable cuts.

## SemVer Rules

- PATCH: registry or manifest refresh, new gated model surfaced, bug fix, or
  documentation sync without API change.
- MINOR: additive public API, new optional argument, new optional extra, or new
  capability package.
- MAJOR: breaking API change, label-schema break, or migration that requires a
  downstream code change.

Public APIs are deprecated for two minor releases before removal. Deprecations
must emit `DeprecationWarning`, name the replacement, and appear in
`CHANGELOG.md` under `Deprecated`.

## Release Gates

The release gates, not aggregate F1 alone, decide whether a model artifact is
releasable. A candidate must satisfy critical-leakage, recall, quantization
delta, device-tier, span-integrity, and regression checks before it can move to
stable. Library releases must also pass the repository policy, dependency
license policy, and test suite.
