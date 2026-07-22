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
| LTS | `pip install "openmed==1.8.*"` | Security and recall-backstop fixes only | as needed for 12 months | regulated deployments |

Nightly builds use PEP 440 development releases such as `1.9.0.devN`.
Release candidates such as `1.9.0rc1` are reserved for pre-stable cuts.

## SemVer Rules

- PATCH: registry or manifest refresh, new gated model surfaced, bug fix, or
  documentation sync without API change.
- MINOR: additive public API, new optional argument, new optional extra, or new
  capability package.
- MAJOR: breaking API change, label-schema break, or migration that requires a
  downstream code change.

## Deprecation Policy

Public APIs receive at least one full minor-version warning window before
removal. A deprecation introduced in `1.9` remains usable throughout the `1.9`
line; its scheduled removal must still respect SemVer, so this notice window is
not permission to make a breaking removal in a later `1.x` release.

New deprecations use `openmed.utils.deprecated`, emit `DeprecationWarning`, name
the replacement when one exists, and appear in `CHANGELOG.md` under
`Deprecated`. The decorator records the introduction and planned-removal
versions so the static API-surface differ can classify the symbol as an
intentional deprecation:

```python
from openmed.utils import deprecated


@deprecated(since="1.9", remove_in="2.0", replacement="openmed.new_api")
def legacy_api() -> None:
    """Compatibility entry point retained during the warning window."""
```

Every breaking or deprecated change must also have a before/after entry in the
relevant migration guide. Version-scoped tag builds compare the candidate
against the release baseline and fail when the guide omits either kind of
required change.

## Release Gates

The release gates, not aggregate F1 alone, decide whether a model artifact is
releasable. A candidate must satisfy critical-leakage, recall, quantization
delta, device-tier, span-integrity, and regression checks before it can move to
stable. Library releases must also pass the repository policy, dependency
license policy, and test suite.
