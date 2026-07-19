# Migrating from OpenMed v1 to v2

OpenMed v2 has not been released yet. This page is the living upgrade contract
for applications moving from the latest v1 release to v2. Update it alongside
the [changelog](https://github.com/maziyarpanahi/openmed/blob/master/CHANGELOG.md)
whenever a v2 change affects downstream code.

## Compatibility contract

The v2 migration is not a package re-home. Documented imports from the
top-level `openmed` module remain available unless they complete the project's
deprecation process. Existing code such as the following should continue to
work:

```python
from openmed import OpenMedConfig, analyze_text, deidentify, extract_pii
```

New capability areas use explicit package imports instead of expanding the
root namespace indefinitely. These additions do not require existing v1 code
to change.

OpenMed follows [Semantic Versioning](release/semver-and-channels.md). Public
APIs are deprecated for at least two minor releases before removal, emit a
`DeprecationWarning`, name their replacement, and appear under `Deprecated` in
the changelog.

## Upgrade checklist

1. Upgrade to the latest v1 release and run the application test suite before
   testing a v2 pre-release.
2. Review the
   [changelog](https://github.com/maziyarpanahi/openmed/blob/master/CHANGELOG.md)
   from the installed version through the target v2 version.
3. Run tests with Python warnings enabled and replace any deprecated calls.
4. Keep established `from openmed import ...` imports unless a release note
   explicitly documents a completed deprecation.
5. Update type annotations that assume `analyze_text(...,
   output_format="dict")` returns a plain `dict`.
6. Re-run configuration, privacy, direct-identifier recall, and leakage tests
   with synthetic fixtures before deployment.

## Additive package surface

The following namespaces were added during v1 and remain available for v2.
They are imported explicitly and do not replace the stable root API.

| Namespace | Available since | Purpose | Upgrade action |
|---|---:|---|---|
| `openmed.structured` | 1.6.0 | Reserved structured-data privacy namespace. It has no public helpers in 1.9.1. | Do not depend on private modules; adopt public exports as they are documented. |
| `openmed.risk` | 1.6.0 | Re-identification risk reports, privacy budgets, k-anonymity, and audit diffs. | Import risk helpers from `openmed.risk`. |
| `openmed.interop` | 1.6.0 | Lazy optional adapters for FHIR, HL7 v2, dataframes, orchestration tools, and third-party libraries. | Use `available_adapters()` and install the adapter's optional extra. |
| `openmed.clinical` | 1.7.0 | Clinical context, relations, timelines, normalization, and extraction helpers. | Import clinical helpers from `openmed.clinical`. |
| `openmed.eval` | 1.9.0 | Evaluation suites, calibration, leakage, fairness, and release-gate evidence. | Import evaluation helpers from `openmed.eval`. |

## Changed and expanded surfaces

### Typed analysis results

Since 1.7.0, `analyze_text(..., output_format="dict")` returns an immutable
`AnalyzeResult` rather than a plain dictionary. Mapping access and `to_dict()`
preserve the v1 payload shape, so most callers can migrate incrementally:

```python
from openmed import AnalyzeResult, analyze_text

result: AnalyzeResult = analyze_text("Synthetic clinical text")
entities = result["entities"]       # legacy mapping access
payload = result.to_dict()          # explicit plain-dict boundary
```

See [Analyze Text Helper](analyze-text.md) for the current return contract.

### Policy-aware de-identification

The optional `policy=` argument has been available since 1.6.0. Prefer named
policy profiles over application-side combinations of redaction flags so that
arbitration and mandatory safety-sweep behavior stay versioned together:

```python
from openmed import deidentify

result = deidentify(text, policy="hipaa_safe_harbor")
```

See [De-identification API](api/deidentification.md) and [Configuration
Profiles](profiles.md).

### Async and streaming helpers

Async APIs are additive; synchronous v1 entry points remain supported.

- `openmed.interop.fhir_bulk.deidentify_ndjson_async` is available since
  1.7.0 for FHIR Bulk NDJSON workflows.
- `openmed.processing.advanced_ner.stream_token_classifier` is available since
  1.8.0 for asynchronous token-classification streams.
- Service async jobs and webhooks are documented in [Async REST Jobs &
  Webhooks](serving/async-jobs.md).

### Configuration validation

Keep validation at process boundaries. `OpenMedConfig`, `validate_input`, and
`validate_model_name` remain available from the root module. Review
[Configuration & Validation](configuration.md) when moving to a new major
version, especially for offline mode, model paths, devices, and optional
accelerated attention backends.

## Deprecations and compatibility aliases

| v1 surface | Status in 1.9.1 | Replacement | v2 action |
|---|---|---|---|
| `deidentify(..., shift_dates=True)` | Deprecated alias, still accepted | `deidentify(..., method="shift_dates")` | Replace now; check the v2 changelog before relying on the alias. |

The following compatibility behaviors are not currently deprecated:

- `AnalyzeResult` mapping access and `to_dict()` preserve the legacy result
  payload.
- REST `GET /health` remains an alias for deployments that have not moved to
  `/livez` and `/readyz`.

If a future v2 release adds or removes a deprecation, the changelog is the
authoritative record and this table must be updated in the same change.

## Verify an upgrade

```bash
uv sync --extra dev --extra docs
uv run pytest tests/ -q
uv run mkdocs build --strict
```

For privacy-sensitive deployments, also run the applicable direct-identifier
recall, critical-leakage, span-integrity, and policy-profile gates before
promoting the new version.

## References

- [Changelog](https://github.com/maziyarpanahi/openmed/blob/master/CHANGELOG.md)
- [Release Streams, SemVer, and Channels](release/semver-and-channels.md)
- [OpenMed 1.9.1 Release Notes](release/v1.9.1.md)
- [OpenMed 1.9.0 Release Notes](release/v1.9.0.md)
- [Configuration & Validation](configuration.md)
