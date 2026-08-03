# Detector plugin SDK stability policy

OpenMed detector plugins add span-producing privacy detectors without adding a
runtime dependency to OpenMed core. This page defines the supported plugin
contract, its compatibility guarantees, and the rules third-party packages
must follow.

The public implementation lives in `openmed.core.detector_plugins`. APIs not
documented on this page remain implementation details.

## Supported contract

Plugins publish an entry point in the stable `openmed.detectors` group. The
loaded value may be:

- one `DetectorSpec` instance;
- a callable that returns a `DetectorSpec`, an iterable of specs, or `None`;
- an iterable containing supported values from the previous two forms.

Discovery is lazy and process-local. OpenMed loads entry points on the first
detector lookup, registers valid specs, and logs and skips a plugin that fails
to load. Plugin packages must not depend on import order or mutate OpenMed's
internal registry directly.

### Stable `DetectorSpec` fields

| Field | Contract |
|---|---|
| `name` | Non-empty identifier unique within its stage. A colon is not allowed. |
| `stage` | One of `deterministic`, `fast_pii`, or `clinical_phi`. |
| `languages` | A language tag, sequence of tags, or wildcard (`*`, `all`, `any`). Tags are normalized to lowercase BCP 47-style hyphenated values. |
| `detect` | Callable invoked with normalized text and keyword context including `lang`. It returns a sequence of `OpenMedSpan` records. |
| `provenance_prefix` | Non-empty prefix used with `name` to produce the stable detector identifier. Defaults to `plugin`. |
| `covered_labels` | Optional canonical-label declaration used by policy capability checks. Wildcards expand to all canonical labels. |

`DetectorSpec.capability()` and the resulting `DetectorCapability` record are
also public. A capability is emitted only when `covered_labels` is non-empty.

## Component kinds and execution stages

The current SDK supports detector plugins only. A detector must select the
stage that matches its cost and purpose:

- `deterministic` for patterns, validators, and dictionary checks;
- `fast_pii` for broad PII/PHI detection suitable for the fast privacy pass;
- `clinical_phi` for clinically specialized PHI detection.

Other OpenMed extension surfaces are not implicitly detector plugins. Do not
register model loaders, exporters, anonymizers, service middleware, or network
clients in `openmed.detectors`.

## Span and label requirements

Detector callables must return `OpenMedSpan` values whose offsets refer to the
normalized text supplied to the plugin. Spans must satisfy these rules:

- `start` and `end` are valid character offsets and `start < end`;
- `entity_type` and `canonical_label` normalize to an OpenMed canonical label;
- `score` is a finite confidence value appropriate for arbitration;
- metadata and evidence never contain raw PHI or PII;
- the detector does not retain input text after the call completes.

OpenMed rewrites document identity, text hashes, and detector provenance before
arbitration. Plugins must not rely on placeholder `doc_id` or `text_hash` values
surviving pipeline execution.

## Minimal package

Declare the entry point in `pyproject.toml`:

```toml
[project]
name = "example-openmed-detector"
dependencies = ["openmed>=1.9,<2"]

[project.entry-points."openmed.detectors"]
example_mrn = "example_openmed_detector:detector"
```

Return a `DetectorSpec` from the referenced object:

```python
import re

from openmed.core.detector_plugins import DetectorSpec
from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash

_MRN_PATTERN = re.compile(r"\bMRN:\s*([A-Za-z0-9-]+)\b")


def detect(text: str, *, lang: str, context=None):
    match = _MRN_PATTERN.search(text)
    if match is None:
        return ()

    start, end = match.span(1)
    return (
        OpenMedSpan(
            doc_id="plugin-placeholder",
            start=start,
            end=end,
            text_hash=hmac_text_hash(text[start:end], "plugin-placeholder"),
            entity_type="ID_NUM",
            canonical_label="ID_NUM",
            score=0.95,
        ),
    )


def detector() -> DetectorSpec:
    return DetectorSpec(
        name="example_mrn",
        stage="deterministic",
        languages=("en",),
        detect=detect,
        covered_labels=("ID_NUM",),
    )
```

A typical source layout is:

```text
example-openmed-detector/
  pyproject.toml
  src/example_openmed_detector/__init__.py
  tests/test_detector.py
  LICENSE
  README.md
```

## Local-first and opt-in rules

Plugin installation is an explicit user opt-in. After installation, plugins
must preserve OpenMed's local-first guarantees:

- no telemetry or background network calls by default;
- no automatic model or dataset download during import or discovery;
- no raw PHI in logs, exceptions, caches, temporary files, or traces;
- remote services require explicit configuration and must be disabled by
  default;
- restricted datasets and credentials remain user-supplied and are never
  bundled;
- core-compatible dependencies use permissive licenses.

Heavy runtimes, remote adapters, and license-restricted integrations belong in
optional extras. A missing optional dependency should produce a clear setup
error only when the related detector is selected, not when OpenMed discovers
the package.

## Semantic-versioning policy

The following changes to the documented plugin protocol require an OpenMed
major release unless an earlier deprecation path preserves compatibility:

- renaming or removing the `openmed.detectors` entry-point group;
- removing or renaming a stable `DetectorSpec` field;
- making an optional field required or changing its accepted value shape;
- removing an execution stage or changing its established meaning;
- changing the detector callable's required arguments or return type;
- changing offset interpretation away from normalized-text character offsets;
- removing a canonical label without a compatibility alias;
- changing discovery so a previously valid entry-point return shape is rejected;
- weakening the local-first, no-telemetry, or no-raw-PHI defaults.

Additive optional fields with backward-compatible defaults, new canonical
labels, new stages, and new entry-point return conveniences may ship in a minor
release. Documentation clarifications and stricter rejection of inputs that
were already invalid may ship in a patch release.

Deprecated fields or values remain available for at least two minor releases,
emit a `DeprecationWarning`, identify their replacement, and appear in the
changelog before removal.

## Upgrade and conformance checks

Before widening the supported OpenMed version range, plugin authors should:

1. construct every exported `DetectorSpec` under the oldest and newest
   supported OpenMed versions;
2. exercise discovery through installed entry-point metadata, not only direct
   registration;
3. run synthetic positive, negative, overlap, Unicode, and invalid-span cases;
4. verify every declared `covered_labels` value normalizes successfully;
5. confirm logs and exceptions contain no input surface text;
6. test with network access disabled unless the user explicitly enables a
   remote integration.

The repository tests in `tests/unit/core/test_detector_plugins.py` demonstrate
the current discovery and validation contract. A standalone example plugin and
published conformance kit are planned; until they are available, use those
tests and this page as the compatibility baseline.
