# Migration guides

For the current release candidate, see [OpenMed 3.2 to 3.3](migration/3.2-to-3.3.md) for multimodal intake, local providers, wire versions and review controls.
The [3.1 to 3.2 guide](migration/3.1-to-3.2.md) covers the Python 3.11 minimum, synthetic rendering, multilingual privacy and local decision backends.
For the earlier governed-workflow migration, see
[OpenMed 3.0 to 3.1](migration/3.0-to-3.1.md): direct FHIR/OpenMRS and OMOP
writers, permission contracts, staged changes and reviewed execution.
The [2.5 to 3.0 guide](migration/2.5-to-3.0.md) covers the Journey migration.
The v1-to-v2 contract below remains available for older installations.

OpenMed v2 begins with the `2.0.0` release. This page is the durable upgrade
contract for applications moving from the final v1 line to v2. Update it
alongside the
[changelog](https://github.com/maziyarpanahi/openmed/blob/master/CHANGELOG.md)
whenever a v2 change affects downstream code. For the concrete first-major
upgrade checklist, see [Migrating from 1.9 to 2.0](migration/1.9-to-2.0.md).

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
APIs receive at least one full minor-version warning window, emit a
`DeprecationWarning`, name their replacement, and appear under `Deprecated` in
the changelog. Breaking removal must wait for a major release; the warning
window does not permit a breaking change in a later minor release. See the
[public API deprecation policy](compliance/api-deprecation-policy.md).

## Upgrade checklist

1. Upgrade to the latest v1 release and run the application test suite before
   upgrading to v2.
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
| `openmed.structured` | 1.6.0 | Structured-data privacy namespace with public flowsheet and lab-panel structures in v2. | Existing v1 code is unaffected; adopt new public helpers only when the target release documents them. |
| `openmed.risk` | 1.6.0 | Re-identification risk reports, privacy budgets, k-anonymity, and audit diffs. | Import risk helpers from `openmed.risk`. |
| `openmed.interop` | 1.6.0 | Lazy optional adapters for FHIR, HL7 v2, dataframes, orchestration tools, and third-party libraries. | Use `available_adapters()` and install the adapter's optional extra. |
| `openmed.clinical` | 1.6.0 | Clinical context, relations, timelines, normalization, and extraction helpers. | Import clinical helpers from `openmed.clinical`. |
| `openmed.eval` | 1.6.0 | Evaluation suites, calibration, leakage, fairness, and release-gate evidence. | Import evaluation helpers from `openmed.eval`. |

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

- `openmed.interop.fhir.bulk.deidentify_ndjson_async` is available since
  1.8.0 for FHIR Bulk NDJSON workflows.
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
| `deidentify(..., shift_dates=True)` | Deprecated alias, still accepted | `deidentify(..., method="shift_dates")` | Use the method argument before 5.0. |

The following compatibility behaviors are not currently deprecated:

- `AnalyzeResult` mapping access and `to_dict()` preserve the legacy result
  payload.
- REST `GET /health` remains an alias for deployments that have not moved to
  `/livez` and `/readyz`.

If a later v2 release adds or removes a deprecation, the changelog is the
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
- [Migrating from 1.9 to 2.0](migration/1.9-to-2.0.md)
- [Migrating from 2.0 to 2.1](migration/2.0-to-2.1.md)
- [Migrating from 2.1 to 2.2](migration/2.1-to-2.2.md)
- [Migrating from 2.2 to 2.3](migration/2.2-to-2.3.md)
- [Migrate from 2.3 to 2.5](migration/2.3-to-2.5.md)
- [Migrating from 2.5 to 3.0](migration/2.5-to-3.0.md)
- [OpenMed 2.5.0 Release Notes](release/v2.5.0.md)
- [OpenMed 2.3.0 Release Notes](release/v2.3.0.md)
- [OpenMed 2.2.0 Release Notes](release/v2.2.0.md)
- [OpenMed 2.1.0 Release Notes](release/v2.1.0.md)
- [OpenMed 2.0.0 Release Notes](release/v2.0.0.md)
- [OpenMed 1.9.1 Release Notes](release/v1.9.1.md)
- [OpenMed 1.9.0 Release Notes](release/v1.9.0.md)
- [Configuration & Validation](configuration.md)

## Planned 5.0 changes

The [removals ledger](migration/5.0-removals.json) tracks the warning status and
replacement for each planned change. Run
`python scripts/release/check_removals_ledger.py` from the repository root to
check the entries against source.

### Typed span evidence

Free-form span evidence is planned to move to typed evidence references.

### Audit chain

Direct construction of `HashChainAuditLog` is deprecated in 4.3 for removal
in 5.0. Pass `CoreAuditSink(chain)` from `openmed.compliance` to
`assemble_dsar_package(..., audit_sink=sink)` or `Part11AuditEmitter(audit_sink=sink)`,
where `chain` is an `openmed.core.audit_chain.AuditChain`. The core ledger
holds only commitments. Existing Part 11 exports keep their wire format.
Persist the core chain separately with `chain.write(path)` for independent
verification with `openmed audit verify`.

### Policy receipts

De-identification and decision responses are planned to carry policy receipts.

### Generated clients

The TypeScript and Go clients are planned to move to generated OpenAPI clients.

### Service settings

Service configuration is planned to use a single validated loader.

### Android model acquisition

Android apps using MLX-named cache and download methods will migrate to the
format-aware model acquisition API.

### MCP non-loopback HTTP binds

The PHI-capable MCP server warns on non-loopback HTTP startup in 4.3.
For 5.0 bind the server to loopback behind an authenticated reverse proxy,
use stdio, or choose the PHI-free server. Docker deployments can colocate
the proxy and the MCP server in the same network namespace.
[Transport and container examples](mcp-clients.md#streamable-http-connections)

### FHIR Bulk import path

Change the import module from `openmed.interop.fhir_bulk` to
`openmed.interop.fhir.bulk` and keep the same exported names. Importing the
legacy module warns in 4.3 and is scheduled for removal in 5.0.

```python
from openmed.interop.fhir.bulk import deidentify_ndjson
```


### Date shift boolean alias

| Before 4.3 | Preferred form |
| --- | --- |
| `deidentify(text, shift_dates=True)` | `deidentify(text, method="shift_dates")` |

The boolean remains supported through 4.x. Python warns at the call site.
REST exposes the fixed notice in the `OpenMed-Deprecation` response header
and MCP in `metadata.deprecation_notice`. Use the existing method argument
alone to select date shifting.
