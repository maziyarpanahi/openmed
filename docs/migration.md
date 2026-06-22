# v1 to v2 Migration Guide

This guide tracks the compatibility promise for applications moving from the
OpenMed 1.5.x line toward the v2 roadmap. It should be read together with the
[changelog](https://github.com/maziyarpanahi/openmed/blob/master/CHANGELOG.md),
which remains the release-by-release source of truth for shipped behavior.

## Import Stability

OpenMed keeps the existing top-level import surface stable while v2 packages are
introduced. Code that works today with imports like these should continue to
work through the v2 transition:

```python
from openmed import analyze_text, extract_pii, deidentify, reidentify
from openmed import OpenMedConfig, ModelLoader
from openmed import BatchProcessor, process_batch
```

The v2 direction is additive first: new packages may appear under `openmed.*`,
but existing top-level names are not removed without a documented deprecation
window and changelog entry.

## New Package Surfaces

These top-level packages are part of the v2 organization work. Import them
directly when you need their specialized APIs; keep using top-level imports for
the stable one-line workflows.

| Package | Status | Use it for |
| --- | --- | --- |
| `openmed.clinical` | Available since 1.5.x as a namespace package | Future clinical document sections, context, grounding, relations, SDOH, and FHIR/OMOP exporters. |
| `openmed.eval` | Available since 1.5.x | Benchmark fixtures, metrics, report generation, and release-gate evaluation helpers. |
| `openmed.structured` | Available since 1.5.x as a namespace package | Future structured-data privacy capabilities such as column classification and k-anonymity workflows. |
| `openmed.risk` | Available since 1.5.x | Re-identification risk reporting via `risk_report`. |
| `openmed.interop` | Available since 1.5.x as a namespace package | Future optional adapters and subprocess-isolated integrations. |
| `openmed.ner` | Available since 1.5.x | Zero-shot NER indexing, inference, label maps, and token-classification adapters. |
| `openmed.mcp` | Available since 1.5.x | MCP server creation and command entry points. |

Namespace packages marked as future-facing may not expose a broad public API yet.
Check the package `__all__` before depending on a symbol.

## Changed and Typed Results

The core one-line APIs already return typed result objects when you request the
default structured output:

- `analyze_text(...)` returns `PredictionResult` for the default `dict` output
  path, or a formatted `json`, `html`, or `csv` payload when requested.
- `deidentify(...)` returns `DeidentificationResult`.
- Batch workflows return `BatchResult`, with per-item `BatchItemResult` entries.
- Re-identification risk helpers expose `risk_report` from `openmed.risk`.

When upgrading, prefer consuming documented attributes on these result objects
instead of depending on incidental dictionary shape. If you need a serialized
payload, request an explicit output format or call the documented formatter.

## Service and Async Boundaries

The Python SDK APIs remain synchronous. Async behavior exists at the REST service
boundary, where FastAPI handlers manage request timeouts, model lifecycle, and
model unloading. For service deployments, review:

- [REST Service](rest-service.md)
- `GET /models/loaded`
- `POST /models/unload`
- `OPENMED_SERVICE_KEEP_ALIVE`

Future async SDK helpers, if added, should be documented here with an "available
since" note before users rely on them.

## Upgrade Checklist

1. Pin and read the target release notes in
   [CHANGELOG.md](https://github.com/maziyarpanahi/openmed/blob/master/CHANGELOG.md).
2. Keep existing imports from `openmed` for `analyze_text`, `extract_pii`,
   `deidentify`, `reidentify`, `OpenMedConfig`, `ModelLoader`, and batch helpers.
3. Move only specialized code to subpackages such as `openmed.eval`,
   `openmed.risk`, or `openmed.ner`.
4. Treat namespace-only packages such as `openmed.clinical`,
   `openmed.structured`, and `openmed.interop` as placeholders until their
   public APIs are documented.
5. Run the test suite against the new version, including any model-loading,
   PII, REST service, and batch-processing paths your application uses.
6. If you consume REST endpoints, validate request schemas and language/model
   values against the target release.

## Deprecations

There are currently no v1.5.x top-level API removals scheduled by this guide.

| API or behavior | Status | Replacement | Earliest removal |
| --- | --- | --- | --- |
| None | No current deprecation | n/a | n/a |

## Deprecation Policy

OpenMed follows Semantic Versioning and records release changes in
[CHANGELOG.md](https://github.com/maziyarpanahi/openmed/blob/master/CHANGELOG.md).
Deprecations should include:

- a changelog entry in the release where the deprecation starts
- an alternative API or migration path
- an "earliest removal" version or major-version window
- tests or docs that keep old behavior visible until the removal window

Compatibility work should prefer aliases, adapters, and warning periods over
silent behavior changes. Major package re-homes belong in a major-version
migration and should not remove the stable top-level imports without an explicit
deprecation cycle.

## Future Change Tracking

When roadmap items land, update this guide with concrete "available since"
notes before adding upgrade instructions. In particular, track:

- policy-style configuration arguments if they become part of public APIs
- async SDK helpers if they are introduced outside the REST service boundary
- new typed result classes or renamed fields
- any moved package surface that still needs a top-level compatibility alias
