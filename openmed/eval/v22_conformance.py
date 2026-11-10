"""Offline V2.2 exchange conformance and release-evidence helpers.

The reference flow is intentionally synthetic and narrow. It proves that the
declared OpenMed form, grounding, FHIR R4/IPS, and FHIR-to-OMOP subsets compose
without a network connection. It does not claim standards certification.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from openmed.clinical.exporters.fhir.exchange import build_ips_patient_summary
from openmed.clinical.exporters.fhir.grounded import to_fhir
from openmed.clinical.grounding import VocabLoader, VocabSource, ground
from openmed.interop.fhir.bulk_checkpoint import (
    BulkCheckpointCompatibilityError,
    create_checkpoint,
    validate_resume,
)
from openmed.interop.fhir.reference_integrity import reference_integrity_report
from openmed.interop.fhir.versions import (
    FHIRVersion,
    UnsupportedFHIRFieldError,
    convert_resource,
)
from openmed.interop.fhir_omop import (
    FHIR_RESOURCE_TYPES as FHIR_OMOP_RESOURCE_TYPES,
)
from openmed.interop.fhir_omop import load_fhir_bundle
from openmed.interop.omop import validate_omop_tables
from openmed.mcp.authorization_conformance import (
    ConformanceManifest,
    run_conformance,
)
from openmed.structured.forms import extract_form_fields
from openmed.structured.privacy_lab import (
    StructuredPrivacyPolicy,
    run_structured_privacy_lab,
)

V22_COVERAGE_SCHEMA_VERSION: Final = "openmed.v22.coverage-matrix.v1"
V22_EVIDENCE_SCHEMA_VERSION: Final = "openmed.v22.conformance-evidence.v1"
V22_FOCUSED_TEST_COMMAND: Final = (
    ".venv/bin/python -m pytest "
    "tests/integration/test_v22_exchange_conformance.py "
    "tests/unit/release/test_v22_coverage_matrix.py -q"
)

_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_MATRIX_REQUIRED_FIELDS: Final = frozenset(
    {
        "schema_version",
        "matrix_version",
        "release",
        "synthetic_only",
        "test_command",
        "entries",
    }
)
_ENTRY_REQUIRED_FIELDS: Final = frozenset(
    {
        "id",
        "standard_profile",
        "version",
        "supported_subset",
        "known_gaps",
        "fixture_sha256",
        "test_nodes",
        "source_modules",
    }
)


class V22ConformanceError(ValueError):
    """Raised when the focused V2.2 proof fails closed."""


class V22CoverageMatrixError(V22ConformanceError):
    """Raised when the versioned V2.2 coverage matrix drifts."""


@dataclass(frozen=True)
class V22ReferenceResult:
    """PHI-free artifacts and hashes from the synthetic reference flow."""

    review_artifact: Mapping[str, Any]
    grounding_audit: Mapping[str, Any]
    fhir_bundle: Mapping[str, Any]
    omop_report: Mapping[str, Any]
    evidence: Mapping[str, Any]
    artifact_paths: Mapping[str, Path]
    fhir_sha256: str
    omop_sha256: str
    evidence_sha256: str

    @property
    def hashes(self) -> dict[str, str]:
        """Return the three release-pinned deterministic hashes."""

        return {
            "fhir_sha256": self.fhir_sha256,
            "omop_sha256": self.omop_sha256,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True)
class V22NegativeBoundaryResult:
    """One sanitized expected-failure result from a focused fixture."""

    case_id: str
    category: str
    boundary: str
    safe_error: str

    def to_dict(self) -> dict[str, str]:
        """Return a deterministic public result without fixture values."""

        return {
            "case_id": self.case_id,
            "status": "expected_failure",
            "category": self.category,
            "boundary": self.boundary,
            "safe_error": self.safe_error,
            "error_sha256": _sha256_text(self.safe_error),
        }


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest of canonical JSON."""

    return _sha256_text(canonical_json(value))


def critical_leakage_counts(
    direct_identifiers: Sequence[str],
    surfaces: Mapping[str, Any],
) -> dict[str, int]:
    """Count exact, case-insensitive identifier occurrences by surface."""

    identifiers = tuple(
        identifier.casefold()
        for identifier in direct_identifiers
        if isinstance(identifier, str) and identifier
    )
    counts: dict[str, int] = {}
    for name, value in surfaces.items():
        text = _surface_text(value).casefold()
        counts[str(name)] = sum(text.count(identifier) for identifier in identifiers)
    return dict(sorted(counts.items()))


def assert_zero_critical_leakage(
    direct_identifiers: Sequence[str],
    surfaces: Mapping[str, Any],
) -> dict[str, int]:
    """Return per-surface zero counts or fail without echoing an identifier."""

    counts = critical_leakage_counts(direct_identifiers, surfaces)
    leaking = sorted(name for name, count in counts.items() if count)
    if leaking:
        raise V22ConformanceError(
            "critical identifier leakage detected in surfaces: " + ", ".join(leaking)
        )
    return counts


def run_v22_reference_flow(
    fixture_root: str | Path,
    work_dir: str | Path,
) -> V22ReferenceResult:
    """Run the offline synthetic form-to-FHIR-to-OMOP reference flow.

    Only redacted review data, grounding hashes/codes, pseudonymous FHIR, OMOP
    rows with redacted note text, and counts/hashes are written to ``work_dir``.
    """

    fixtures = Path(fixture_root)
    output_dir = Path(work_dir)
    form_fixture = _read_json(fixtures / "reference_form.json")
    direct_identifiers = _string_sequence(
        form_fixture.get("synthetic_direct_identifiers"),
        "synthetic_direct_identifiers",
    )

    form_result = extract_form_fields(
        form_fixture["source"],
        schema=form_fixture["schema"],
    )
    fields = {field.link_id: field for field in form_result.fields}
    missing_fields = sorted(
        {"patient_name", "record_id", "contact_phone", "condition", "a1c_value"}
        - fields.keys()
    )
    if missing_fields:
        raise V22ConformanceError(
            "reference form is missing declared fields: " + ", ".join(missing_fields)
        )
    review_artifact = form_result.to_dict()

    vocabulary_fixture = fixtures / "grounding_vocabulary.jsonl"
    vocabulary_path = output_dir / "fixtures" / vocabulary_fixture.name
    vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
    with vocabulary_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(vocabulary_fixture.read_text(encoding="utf-8"))
    vocabulary_digest = _sha256_bytes(vocabulary_path.read_bytes())
    loader = VocabLoader(
        cache_dir=output_dir / "vocabulary-cache",
        local_only=True,
        registry={
            "icd10cm": VocabSource(
                system="icd10cm",
                path=vocabulary_path,
                sha256=vocabulary_digest.removeprefix("sha256:"),
                version="v22-synthetic-2026-08",
                license_note="Synthetic test vocabulary; not a distributable code set.",
            )
        },
    )
    condition = fields["condition"]
    grounded = ground(
        [
            {
                "text": condition.value,
                "start": condition.start,
                "end": condition.end,
                "label": "condition",
                "assertion": {
                    "temporality": "recent",
                    "certainty": "certain",
                    "negation": "affirmed",
                    "experiencer": "patient",
                },
            }
        ],
        systems=("icd10cm",),
        loader=loader,
        offline=True,
    )
    if len(grounded) != 1 or grounded[0].codes != {"icd10cm": "E11.9"}:
        raise V22ConformanceError(
            "reference condition did not ground to the pinned code"
        )
    grounding_audit = grounded[0].to_audit_dict()

    patient_seed = canonical_json(sorted(direct_identifiers))
    patient_id = f"patient-{hashlib.sha256(patient_seed.encode()).hexdigest()[:24]}"
    document_id = f"v22-{sha256_json(review_artifact)[7:23]}"
    patient = {"resourceType": "Patient", "id": patient_id}
    subject_reference = f"Patient/{patient_id}"
    condition_resource = to_fhir(
        grounded[0],
        resource="Condition",
        subject_reference=subject_reference,
        document_id=document_id,
    )
    if not isinstance(condition_resource, Mapping):
        raise V22ConformanceError("reference FHIR Condition export was empty")
    fhir_bundle = build_ips_patient_summary(
        [condition_resource],
        patient=patient,
        document_id=document_id,
        author_reference=subject_reference,
        validate_output=True,
    )
    integrity = reference_integrity_report(fhir_bundle, fhir_version="R4")
    if not integrity.valid:
        raise V22ConformanceError("reference FHIR Bundle has unresolved references")

    omop_resources = [
        entry["resource"]
        for entry in fhir_bundle.get("entry", ())
        if isinstance(entry, Mapping)
        and isinstance(entry.get("resource"), Mapping)
        and entry["resource"].get("resourceType") in FHIR_OMOP_RESOURCE_TYPES
    ]
    omop_vocabulary = _read_json(fixtures / "omop_vocabulary.json")
    loaded = load_fhir_bundle(
        omop_resources,
        vocabulary=omop_vocabulary,
        vocabulary_snapshot="v22-synthetic-2026-08",
        source_system="OpenMed V2.2 synthetic FHIR",
        source_version="FHIR R4 4.0.1",
    )
    violations = validate_omop_tables(loaded.omop)
    if violations:
        raise V22ConformanceError("reference OMOP output violates CDM constraints")
    omop_report = loaded.to_dict()

    review_sha256 = sha256_json(review_artifact)
    grounding_sha256 = sha256_json(grounding_audit)
    fhir_sha256 = sha256_json(fhir_bundle)
    omop_sha256 = sha256_json(omop_report)
    evidence: dict[str, Any] = {
        "schema_version": V22_EVIDENCE_SCHEMA_VERSION,
        "release": "2.2",
        "synthetic": True,
        "offline": True,
        "reference_flow": [
            "form_intake_and_deidentification",
            "local_icd10cm_grounding",
            "fhir_r4_ips_patient_summary",
            "fhir_r4_to_omop_cdm_5_4",
        ],
        "artifact_hashes": {
            "form_review_sha256": review_sha256,
            "grounding_audit_sha256": grounding_sha256,
            "fhir_sha256": fhir_sha256,
            "omop_sha256": omop_sha256,
        },
        "grounding": {
            "code": grounding_audit["code"],
            "system_uri": grounding_audit["system_uri"],
            "start": grounding_audit["start"],
            "end": grounding_audit["end"],
            "vocabulary_sha256": vocabulary_digest,
        },
        "fhir": {
            "core_version": "4.0.1",
            "ips_profile_version": "2.0.1",
            "resource_counts": _fhir_resource_counts(fhir_bundle),
            "reference_integrity": integrity.to_dict(),
        },
        "omop": {
            "cdm_version": "5.4",
            "summary": loaded.summary.to_dict(),
        },
        "critical_leakage": {
            "maximum_allowed": 0,
            "observed": 0,
            "surfaces": [
                "form review",
                "grounding audit",
                "FHIR serialization",
                "OMOP report",
                "evidence artifact",
                "temporary paths",
            ],
        },
        "qualification": (
            "Tested synthetic subsets only; no HL7, OHDSI, regulatory, or "
            "national-authority certification is claimed."
        ),
    }
    evidence_sha256 = sha256_json(evidence)

    artifact_paths = {
        "form_review": output_dir / "form-review.json",
        "grounding_audit": output_dir / "grounding-audit.json",
        "fhir": output_dir / "fhir-r4-patient-summary.json",
        "omop": output_dir / "omop-cdm-5.4-report.json",
        "evidence": output_dir / "v2.2-conformance-evidence.json",
    }
    assert_zero_critical_leakage(
        direct_identifiers,
        {
            "form_review": review_artifact,
            "grounding_audit": grounding_audit,
            "fhir": fhir_bundle,
            "omop": omop_report,
            "evidence": evidence,
            "temporary_paths": [str(path) for path in artifact_paths.values()],
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("form_review", review_artifact),
        ("grounding_audit", grounding_audit),
        ("fhir", fhir_bundle),
        ("omop", omop_report),
        ("evidence", evidence),
    ):
        _write_json(artifact_paths[name], payload)

    return V22ReferenceResult(
        review_artifact=review_artifact,
        grounding_audit=grounding_audit,
        fhir_bundle=fhir_bundle,
        omop_report=omop_report,
        evidence=evidence,
        artifact_paths=artifact_paths,
        fhir_sha256=fhir_sha256,
        omop_sha256=omop_sha256,
        evidence_sha256=evidence_sha256,
    )


def run_v22_negative_checks(
    fixture_root: str | Path,
) -> tuple[V22NegativeBoundaryResult, ...]:
    """Run four independent offline expected-failure fixtures."""

    fixtures = Path(fixture_root)
    return (
        _run_fhir_r5_negative(fixtures / "fhir_r5_negative.json"),
        _run_bulk_resume_negative(fixtures / "bulk_resume_negative.json"),
        _run_structured_privacy_negative(fixtures / "structured_privacy_negative.json"),
        _run_mcp_authorization_negative(fixtures / "mcp_authorization_negative.json"),
    )


def load_coverage_matrix(path: str | Path) -> dict[str, Any]:
    """Load the versioned JSON coverage matrix and reject duplicate keys."""

    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, ValueError, TypeError) as exc:
        raise V22CoverageMatrixError("coverage matrix is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise V22CoverageMatrixError("coverage matrix must be an object")
    return payload


def validate_coverage_matrix(
    repo_root: str | Path,
    matrix: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Validate versions, fixture hashes, test nodes, and source modules."""

    root = Path(repo_root).resolve()
    payload = (
        load_coverage_matrix(matrix)
        if isinstance(matrix, (str, Path))
        else dict(matrix)
    )
    missing = sorted(_MATRIX_REQUIRED_FIELDS - payload.keys())
    if missing:
        raise V22CoverageMatrixError(
            "coverage matrix is missing fields: " + ", ".join(missing)
        )
    if payload["schema_version"] != V22_COVERAGE_SCHEMA_VERSION:
        raise V22CoverageMatrixError("coverage matrix schema_version is unsupported")
    if payload["matrix_version"] != 1 or payload["release"] != "2.2":
        raise V22CoverageMatrixError("coverage matrix release version is invalid")
    if payload["synthetic_only"] is not True:
        raise V22CoverageMatrixError("coverage matrix must declare synthetic_only")
    if payload["test_command"] != V22_FOCUSED_TEST_COMMAND:
        raise V22CoverageMatrixError("coverage matrix test command drifted")
    entries = payload["entries"]
    if not isinstance(entries, list) or not entries:
        raise V22CoverageMatrixError("coverage matrix entries must be a non-empty list")

    seen_ids: set[str] = set()
    for index, entry_value in enumerate(entries):
        if not isinstance(entry_value, Mapping):
            raise V22CoverageMatrixError(f"coverage entry {index} must be an object")
        entry = dict(entry_value)
        missing_entry = sorted(_ENTRY_REQUIRED_FIELDS - entry.keys())
        if missing_entry:
            raise V22CoverageMatrixError(
                f"coverage entry {index} is missing fields: " + ", ".join(missing_entry)
            )
        entry_id = _required_text(entry["id"], f"entries[{index}].id")
        if entry_id in seen_ids:
            raise V22CoverageMatrixError("coverage entry ids must be unique")
        seen_ids.add(entry_id)
        _required_text(entry["standard_profile"], f"entries[{index}].standard_profile")
        _required_text(entry["version"], f"entries[{index}].version")
        _nonempty_text_list(
            entry["supported_subset"], f"entries[{index}].supported_subset"
        )
        _nonempty_text_list(entry["known_gaps"], f"entries[{index}].known_gaps")
        _validate_fixture_hashes(root, entry["fixture_sha256"], index)
        _validate_test_nodes(root, entry["test_nodes"], index)
        _validate_source_modules(root, entry["source_modules"], index)
    return payload


def render_coverage_matrix_markdown(matrix: Mapping[str, Any]) -> str:
    """Render the complete, deterministic V2.2 standards coverage document."""

    entries = matrix.get("entries", ())
    lines = [
        "<!-- Generated from tests/fixtures/v22/coverage_matrix.json. -->",
        "# OpenMed v2.2 tested standards matrix",
        "",
        "This release gate is an offline, synthetic integration proof. It records",
        "only the subsets exercised by the focused suite and is not certification",
        "by HL7, OHDSI, the EU, or any national authority.",
        "",
        f"Matrix schema: `{matrix.get('schema_version', '')}`",
        f"Matrix version: `{matrix.get('matrix_version', '')}`",
        f"Release: `{matrix.get('release', '')}`",
        "",
        "## Focused command",
        "",
        "```shell",
        str(matrix.get("test_command", "")),
        "```",
        "",
        "The command is intentionally small. GitHub Actions runs the repository-wide",
        "suite separately.",
        "",
        "## Declared coverage",
        "",
        "| ID | Standard or profile | Version | Tested subset | Known gaps | Fixtures and SHA-256 | Source modules | Test nodes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry_value in entries if isinstance(entries, Sequence) else ():
        entry = dict(entry_value)
        fixtures = entry.get("fixture_sha256", {})
        fixture_cell = "<br>".join(
            f"`{_markdown(path)}`<br>`{_markdown(digest)}`"
            for path, digest in sorted(dict(fixtures).items())
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown(entry.get("id", "")),
                    _markdown(entry.get("standard_profile", "")),
                    _markdown(entry.get("version", "")),
                    "<br>".join(
                        _markdown(item) for item in entry.get("supported_subset", ())
                    ),
                    "<br>".join(
                        _markdown(item) for item in entry.get("known_gaps", ())
                    ),
                    fixture_cell,
                    "<br>".join(
                        f"`{_markdown(item)}`"
                        for item in entry.get("source_modules", ())
                    ),
                    "<br>".join(
                        f"`{_markdown(item)}`" for item in entry.get("test_nodes", ())
                    ),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Leakage and evidence boundary",
            "",
            "The reference test requires zero exact occurrences of every synthetic",
            "direct identifier across review JSON, grounding audit output, serialized",
            "FHIR, the redacted OMOP report, captured errors and logs, generated evidence,",
            "and temporary artifact paths. Fixture data is synthetic and committed; no",
            "real-patient or DUA data is used.",
            "",
            "FHIR Composition is deliberately excluded from the FHIR-to-OMOP input because",
            "the bridge declares only Patient, Encounter, Condition, Observation, Procedure,",
            "MedicationStatement, and MedicationRequest. Every other limitation is stated",
            "in the table rather than inferred as support.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_fhir_r5_negative(path: Path) -> V22NegativeBoundaryResult:
    fixture = _read_json(path)
    sensitive = _string_sequence(fixture["sensitive_values"], "sensitive_values")
    try:
        convert_resource(fixture["resource"], FHIRVersion.R5, FHIRVersion.R4)
    except UnsupportedFHIRFieldError as exc:
        if exc.path != "Observation.unsupportedCrossVersionField":
            raise V22ConformanceError(
                "FHIR R5 fixture failed at the wrong path"
            ) from exc
        safe_error = str(exc)
    else:
        raise V22ConformanceError("FHIR R5 negative fixture unexpectedly converted")
    result = V22NegativeBoundaryResult(
        case_id="fhir-r5-unsupported-field",
        category="unsupported_cross_version_field",
        boundary="Observation.unsupportedCrossVersionField",
        safe_error=safe_error,
    )
    assert_zero_critical_leakage(
        sensitive, {"error": safe_error, "result": result.to_dict()}
    )
    return result


def _run_bulk_resume_negative(path: Path) -> V22NegativeBoundaryResult:
    fixture = _read_json(path)
    sensitive = _string_sequence(fixture["sensitive_values"], "sensitive_values")
    checkpoint_context = dict(fixture["checkpoint_context"])
    progress = dict(fixture["progress"])
    checkpoint = create_checkpoint(
        fixture["resource_type"],
        **checkpoint_context,
        **progress,
    )
    validate_resume(
        checkpoint,
        resource_type=fixture["resource_type"],
        **checkpoint_context,
    )
    try:
        validate_resume(
            checkpoint,
            resource_type=fixture["resource_type"],
            **dict(fixture["resume_context"]),
        )
    except BulkCheckpointCompatibilityError as exc:
        safe_error = str(exc)
        if "endpoint scope" not in safe_error:
            raise V22ConformanceError(
                "Bulk Data resume fixture failed at the wrong boundary"
            ) from exc
    else:
        raise V22ConformanceError("Bulk Data resume fixture unexpectedly matched")
    result = V22NegativeBoundaryResult(
        case_id="bulk-data-resume-scope-mismatch",
        category="incompatible_checkpoint",
        boundary="endpoint_scope",
        safe_error=safe_error,
    )
    assert_zero_critical_leakage(
        sensitive,
        {
            "checkpoint": checkpoint.to_dict(),
            "error": safe_error,
            "result": result.to_dict(),
        },
    )
    return result


def _run_structured_privacy_negative(path: Path) -> V22NegativeBoundaryResult:
    fixture = _read_json(path)
    sensitive = _string_sequence(fixture["sensitive_values"], "sensitive_values")
    rows = fixture["records"]
    result_payload = run_structured_privacy_lab(
        rows,
        StructuredPrivacyPolicy(**fixture["policy"]),
        population_assumptions=fixture["population_assumptions"],
        membership_candidates=rows,
    )
    if result_payload.meets_policy:
        raise V22ConformanceError(
            "structured privacy negative fixture unexpectedly met policy"
        )
    safe_error = "structured privacy policy rejected at membership inference boundary"
    result = V22NegativeBoundaryResult(
        case_id="structured-privacy-membership-threshold",
        category="membership_inference_threshold",
        boundary="structured_privacy_policy",
        safe_error=safe_error,
    )
    assert_zero_critical_leakage(
        sensitive,
        {
            "privacy_report": result_payload.to_dict(),
            "error": safe_error,
            "result": result.to_dict(),
        },
    )
    return result


def _run_mcp_authorization_negative(path: Path) -> V22NegativeBoundaryResult:
    fixture = _read_json(path)
    sensitive = _string_sequence(fixture["sensitive_values"], "sensitive_values")
    manifest_payload = {
        key: fixture[key]
        for key in (
            "schema_version",
            "synthetic",
            "protected_resource",
            "authorization_server",
            "policy",
            "cases",
        )
    }
    manifest = ConformanceManifest.from_mapping(manifest_payload)
    case_id = "negative-unapproved-state-change"
    report = run_conformance(manifest, covered_case_ids=(case_id,))
    if not report.ok or len(report.results) != 1:
        raise V22ConformanceError("MCP authorization negative fixture did not conform")
    case = report.results[0]
    if (
        case.failure_category != "unapproved_state_change"
        or case.failure_boundary != "state_change_policy"
    ):
        raise V22ConformanceError(
            "MCP authorization fixture failed at the wrong boundary"
        )
    safe_error = "unapproved_state_change at state_change_policy"
    result = V22NegativeBoundaryResult(
        case_id=case_id,
        category="unapproved_state_change",
        boundary="state_change_policy",
        safe_error=safe_error,
    )
    assert_zero_critical_leakage(
        sensitive,
        {
            "mcp_report": report.to_dict(),
            "error": safe_error,
            "result": result.to_dict(),
        },
    )
    return result


def _validate_fixture_hashes(root: Path, value: Any, index: int) -> None:
    if not isinstance(value, Mapping) or not value:
        raise V22CoverageMatrixError(
            f"entries[{index}].fixture_sha256 must be a non-empty object"
        )
    for relative, expected in value.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise V22CoverageMatrixError("fixture paths and hashes must be strings")
        if _DIGEST_RE.fullmatch(expected) is None:
            raise V22CoverageMatrixError(f"fixture hash is invalid: {relative}")
        path = _resolve_inside(root, relative, "fixture")
        if not path.is_file():
            raise V22CoverageMatrixError(f"declared fixture is missing: {relative}")
        if _sha256_normalized_text_file(path) != expected:
            raise V22CoverageMatrixError(f"declared fixture hash drifted: {relative}")


def _validate_test_nodes(root: Path, value: Any, index: int) -> None:
    nodes = _nonempty_text_list(value, f"entries[{index}].test_nodes")
    for node in nodes:
        relative, *selectors = node.split("::")
        path = _resolve_inside(root, relative, "test")
        if not path.is_file():
            raise V22CoverageMatrixError(f"declared test is missing: {node}")
        if selectors:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError) as exc:
                raise V22CoverageMatrixError(
                    f"declared test cannot be parsed: {node}"
                ) from exc
            names = {
                item.name
                for item in ast.walk(tree)
                if isinstance(
                    item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            }
            if selectors[-1] not in names:
                raise V22CoverageMatrixError(f"declared test node is missing: {node}")


def _validate_source_modules(root: Path, value: Any, index: int) -> None:
    modules = _nonempty_text_list(value, f"entries[{index}].source_modules")
    for module in modules:
        base = root.joinpath(*module.split("."))
        candidates = (base.with_suffix(".py"), base / "__init__.py")
        if not any(candidate.is_file() for candidate in candidates):
            raise V22CoverageMatrixError(f"declared source module is missing: {module}")


def _resolve_inside(root: Path, relative: str, kind: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise V22CoverageMatrixError(
            f"declared {kind} path must be repository-relative"
        )
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise V22CoverageMatrixError(f"declared {kind} path escapes the repository")
    return resolved


def _fhir_resource_counts(bundle: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in bundle.get("entry", ()):
        if not isinstance(entry, Mapping) or not isinstance(
            entry.get("resource"), Mapping
        ):
            continue
        resource_type = str(entry["resource"].get("resourceType", ""))
        counts[resource_type] = counts.get(resource_type, 0) + 1
    return dict(sorted(counts.items()))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise V22ConformanceError(f"invalid V2.2 fixture: {path.name}") from exc
    if not isinstance(payload, dict):
        raise V22ConformanceError(f"V2.2 fixture must be an object: {path.name}")
    return payload


def _write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_normalized_text_file(path: Path) -> str:
    """Hash UTF-8 fixture text after normalizing platform line endings."""

    return _sha256_text(path.read_text(encoding="utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _surface_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise V22ConformanceError(f"{name} must be a sequence")
    result = tuple(item for item in value if isinstance(item, str) and item)
    if len(result) != len(value) or not result:
        raise V22ConformanceError(f"{name} must contain non-empty strings")
    return result


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V22CoverageMatrixError(f"{name} must be non-empty text")
    return value


def _nonempty_text_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise V22CoverageMatrixError(f"{name} must be a non-empty list")
    result = tuple(_required_text(item, name) for item in value)
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


__all__ = [
    "V22_COVERAGE_SCHEMA_VERSION",
    "V22_EVIDENCE_SCHEMA_VERSION",
    "V22_FOCUSED_TEST_COMMAND",
    "V22ConformanceError",
    "V22CoverageMatrixError",
    "V22NegativeBoundaryResult",
    "V22ReferenceResult",
    "assert_zero_critical_leakage",
    "canonical_json",
    "critical_leakage_counts",
    "load_coverage_matrix",
    "render_coverage_matrix_markdown",
    "run_v22_negative_checks",
    "run_v22_reference_flow",
    "sha256_json",
    "validate_coverage_matrix",
]
