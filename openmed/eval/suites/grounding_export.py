"""Synthetic grounding-to-FHIR/OMOP round-trip validation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.clinical.exporters import achilles_smoke_check, to_fhir, to_omop
from openmed.clinical.grounding import Candidate, GroundedSpan
from openmed.eval.report import BenchmarkReport

__all__ = [
    "DEFAULT_GROUNDING_EXPORT_FIXTURE",
    "FHIR_R4_VERSION",
    "GROUNDING_EXPORT",
    "FhirValidatorResult",
    "load_grounding_export_fixture",
    "run_grounding_export_suite",
    "validate_fhir_r4_shape",
    "validate_with_hl7_validator",
]

GROUNDING_EXPORT = "grounding_export_roundtrip"
FHIR_R4_VERSION = "4.0.1"
# The official validator rejects unresolved custom extensions unless their
# exact URL domain is allowlisted. Keep that exception scoped to OpenMed.
_OPENMED_FHIR_EXTENSION_BASE_URL = "https://openmed.ai/fhir/StructureDefinition/"
DEFAULT_GROUNDING_EXPORT_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "golden"
    / "fixtures"
    / "grounding_export.jsonl"
)


@dataclass(frozen=True)
class FhirValidatorResult:
    """Aggregate, PHI-free result from an out-of-process FHIR validator run.

    Args:
        errors: Fatal and error findings.
        warnings: Warning findings.
        information: Informational findings.
        official_validator_executed: Whether the HL7 validator CLI was run.
        output_hash: SHA-256 evidence fingerprint without validator prose.
    """

    errors: int
    warnings: int
    information: int
    official_validator_executed: bool
    output_hash: str


def load_grounding_export_fixture(
    path: str | Path = DEFAULT_GROUNDING_EXPORT_FIXTURE,
) -> tuple[Mapping[str, Any], tuple[GroundedSpan, ...]]:
    """Load the committed synthetic document and its grounded spans.

    Args:
        path: Synthetic JSONL fixture path.

    Returns:
        Document metadata and deterministic grounded spans.
    """

    source = Path(path)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ValueError("grounding export fixture must contain one JSON object")
    document = rows[0]
    text = str(document.get("text") or "")
    spans: list[GroundedSpan] = []
    for payload in document.get("spans") or ():
        mention = str(payload["text"])
        start = int(payload.get("start", text.index(mention)))
        candidate_payload = payload["candidate"]
        spans.append(
            GroundedSpan(
                text=mention,
                start=start,
                end=start + len(mention),
                canonical_label=str(payload["canonical_label"]),
                candidates=(
                    Candidate(
                        system=str(candidate_payload["system"]),
                        code=str(candidate_payload["code"]),
                        display=str(candidate_payload["display"]),
                        score=float(candidate_payload["score"]),
                        source="sparse",
                        matched_alias=str(candidate_payload["display"]),
                        match_kind="exact",
                        vocab_version="sha256:synthetic-grounding-export-v1",
                    ),
                ),
                metadata={
                    "omop_concept_id": int(payload["omop_concept_id"]),
                    **({"value": payload["value"]} if "value" in payload else {}),
                    **({"unit": payload["unit"]} if "unit" in payload else {}),
                },
            )
        )
    return document, tuple(spans)


def run_grounding_export_suite(
    *,
    fixture_path: str | Path = DEFAULT_GROUNDING_EXPORT_FIXTURE,
    validator_jar: str | Path | None = None,
    java: str = "java",
) -> BenchmarkReport:
    """Export the synthetic fixture and score FHIR/OMOP conformance.

    Args:
        fixture_path: Synthetic grounding/export JSONL fixture.
        validator_jar: Optional local official validator CLI JAR.
        java: Java executable used only when ``validator_jar`` is supplied.

    Returns:
        Aggregate synthetic conformance report.
    """

    document, spans = load_grounding_export_fixture(fixture_path)
    bundle = to_fhir(
        spans,
        subject_reference="Patient/synthetic-subject",
        document_id=str(document["document_id"]),
    )
    assert bundle is not None
    structural_errors = validate_fhir_r4_shape(bundle)
    if validator_jar is None:
        fhir_result = FhirValidatorResult(
            errors=len(structural_errors),
            warnings=0,
            information=0,
            official_validator_executed=False,
            output_hash=_stable_hash(structural_errors),
        )
    else:
        fhir_result = validate_with_hl7_validator(
            bundle,
            validator_jar=validator_jar,
            java=java,
        )

    concept_map = {
        (span.candidates[0].system, span.candidates[0].code): int(
            span.metadata["omop_concept_id"]
        )
        for span in spans
    }
    omop = to_omop(
        spans,
        document_text=str(document["text"]),
        document_id=str(document["document_id"]),
        person_id=str(document["person_id"]),
        concept_resolver=concept_map,
        vocabulary_version="synthetic-v1",
    )
    omop_violations = achilles_smoke_check(omop)
    passed = fhir_result.errors == 0 and not omop_violations
    return BenchmarkReport(
        suite=GROUNDING_EXPORT,
        model_name="deterministic-grounding-exporters",
        device="cpu",
        fixture_count=len(spans),
        metrics={
            "passed": passed,
            "fhir": {
                "errors": fhir_result.errors,
                "warnings": fhir_result.warnings,
                "information": fhir_result.information,
                "official_validator_executed": (
                    fhir_result.official_validator_executed
                ),
            },
            "omop": {
                "achilles_smoke_passed": not omop_violations,
                "violations": len(omop_violations),
            },
        },
        metadata={
            "synthetic": True,
            "phi": False,
            "fhir_version": FHIR_R4_VERSION,
            "validator_output_hash": fhir_result.output_hash,
            "omop_smoke_scope": (
                "core table/column, concept-id, person-id, concept-reference, "
                "NOTE_NLP offset and reachability checks"
            ),
        },
    )


def validate_with_hl7_validator(
    bundle: Mapping[str, Any],
    *,
    validator_jar: str | Path,
    java: str = "java",
    timeout: float = 300.0,
) -> FhirValidatorResult:
    """Validate a synthetic Bundle with the official out-of-process R4 CLI.

    Args:
        bundle: Synthetic FHIR Bundle to validate.
        validator_jar: Local validator CLI JAR path.
        java: Java executable or path.
        timeout: Maximum validator runtime in seconds.

    Returns:
        Aggregate severity counts and an evidence hash.
    """

    jar = Path(validator_jar).expanduser()
    if not jar.is_file():
        raise FileNotFoundError(f"FHIR validator jar does not exist: {jar}")
    with tempfile.TemporaryDirectory(prefix="openmed-fhir-validator-") as directory:
        root = Path(directory)
        source = root / "bundle.json"
        output = root / "operation-outcome.json"
        source.write_text(
            json.dumps(bundle, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                java,
                "-jar",
                str(jar),
                str(source),
                "-version",
                FHIR_R4_VERSION,
                "-tx",
                "n/a",
                "-extension",
                _OPENMED_FHIR_EXTENSION_BASE_URL,
                "-output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output_text = output.read_text(encoding="utf-8") if output.exists() else ""
        severities = _operation_outcome_severities(output_text)
        errors = severities["fatal"] + severities["error"]
        if completed.returncode != 0 and errors == 0:
            errors = 1
        digest_source = "\n".join(
            (str(completed.returncode), completed.stdout, completed.stderr, output_text)
        )
        return FhirValidatorResult(
            errors=errors,
            warnings=severities["warning"],
            information=severities["information"],
            official_validator_executed=True,
            output_hash=hashlib.sha256(digest_source.encode("utf-8")).hexdigest(),
        )


def validate_fhir_r4_shape(resource: Mapping[str, Any]) -> tuple[str, ...]:
    """Return deterministic structural errors for the emitted R4 subset.

    Args:
        resource: FHIR resource or Bundle mapping.

    Returns:
        Human-readable structural error messages without source clinical text.
    """

    errors: list[str] = []
    resource_type = resource.get("resourceType")
    if resource_type == "Bundle":
        if resource.get("type") not in {"transaction", "batch", "collection"}:
            errors.append("Bundle.type is missing or unsupported")
        entries = resource.get("entry")
        if not isinstance(entries, list):
            return (*errors, "Bundle.entry must be a list")
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or not isinstance(
                entry.get("resource"), Mapping
            ):
                errors.append(f"Bundle.entry[{index}].resource is missing")
                continue
            errors.extend(validate_fhir_r4_shape(entry["resource"]))
        return tuple(errors)

    required = {
        "Condition": ("subject", "code", "verificationStatus"),
        "MedicationStatement": ("status", "medicationCodeableConcept", "subject"),
        "Observation": ("status", "code", "subject"),
        "Procedure": ("status", "code", "subject"),
    }
    if resource_type not in required:
        return (f"unsupported resourceType {resource_type!r}",)
    for field in required[resource_type]:
        if field not in resource:
            errors.append(f"{resource_type}.{field} is missing")
    return tuple(errors)


def _operation_outcome_severities(payload: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not payload:
        return counts
    try:
        outcome = json.loads(payload)
    except json.JSONDecodeError:
        return counts
    for issue in outcome.get("issue") or ():
        if isinstance(issue, Mapping):
            severity = str(issue.get("severity") or "information")
            counts[severity] += 1
    return counts


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path, default=DEFAULT_GROUNDING_EXPORT_FIXTURE
    )
    parser.add_argument("--validator-jar", type=Path)
    parser.add_argument("--java", default="java")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the suite from the command line."""

    args = _parse_args(argv)
    report = run_grounding_export_suite(
        fixture_path=args.fixture,
        validator_jar=args.validator_jar,
        java=args.java,
    )
    if args.output is not None:
        report.write_json(args.output)
    else:
        print(report.to_json())
    return 0 if report.metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
