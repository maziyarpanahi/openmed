"""Synthetic FHIR R4 Bundle round-trip fidelity evaluation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openmed.clinical.exporters.codeable_concept_simple import system_uri
from openmed.clinical.exporters.fhir import to_bundle
from openmed.eval.report import BenchmarkReport

__all__ = [
    "DEFAULT_FHIR_ROUNDTRIP_FIXTURE",
    "FHIR_R4_VERSION",
    "FHIR_ROUNDTRIP",
    "FHIR_ROUNDTRIP_FIXTURE_PATH",
    "FHIR_ROUNDTRIP_SCHEMA_VERSION",
    "FhirRoundTripCode",
    "FhirRoundTripFixture",
    "FhirRoundTripReference",
    "FhirRoundTripScore",
    "FhirRoundTripSpan",
    "assert_fhir_roundtrip_fidelity",
    "build_fhir_bundle",
    "build_fhir_resources",
    "load_fhir_roundtrip_fixtures",
    "parse_fhir_bundle",
    "run_fhir_roundtrip",
    "run_fhir_roundtrip_suite",
    "score_fhir_roundtrip_fixture",
    "score_fhir_roundtrip_fixtures",
]

FHIR_ROUNDTRIP = "fhir_roundtrip_fidelity"
FHIR_R4_VERSION = "4.0.1"
FHIR_ROUNDTRIP_SCHEMA_VERSION = 1
FHIR_ROUNDTRIP_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "golden" / "fixtures" / "fhir_roundtrip.jsonl"
)
DEFAULT_FHIR_ROUNDTRIP_FIXTURE = FHIR_ROUNDTRIP_FIXTURE_PATH

_CODE_FIELD_BY_RESOURCE = {
    "AllergyIntolerance": "code",
    "Condition": "code",
    "DiagnosticReport": "code",
    "Immunization": "vaccineCode",
    "MedicationRequest": "medicationCodeableConcept",
    "MedicationStatement": "medicationCodeableConcept",
    "Observation": "code",
    "Procedure": "code",
    "ServiceRequest": "code",
}
_SYSTEM_ALIASES = {
    "HPO": "hpo",
    "ICD10CM": "icd-10-cm",
    "ICD11": "icd-11-mms",
    "LOINC": "loinc",
    "MESH": "mesh",
    "RXNORM": "rxnorm",
    "SNOMED": "snomed",
}


@dataclass(frozen=True)
class FhirRoundTripCode:
    """One synthetic terminology code attached to a source span."""

    system: str
    code: str
    display: str | None = None

    def __post_init__(self) -> None:
        if not self.system.strip():
            raise ValueError("FHIR round-trip code system must be non-empty")
        if not self.code.strip():
            raise ValueError("FHIR round-trip code must be non-empty")

    def to_dict(self) -> dict[str, str]:
        """Return the FHIR Coding-shaped representation."""

        result = {"system": self.system, "code": self.code}
        if self.display is not None:
            result["display"] = self.display
        return result


@dataclass(frozen=True)
class FhirRoundTripReference:
    """A source span reference to another span or a direct FHIR reference."""

    path: str
    target: str
    many: bool = False

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("FHIR round-trip reference path must be non-empty")
        if not self.target.strip():
            raise ValueError("FHIR round-trip reference target must be non-empty")


@dataclass(frozen=True)
class FhirRoundTripSpan:
    """A synthetic text span that becomes one FHIR resource."""

    span_id: str
    resource_type: str
    resource_id: str
    text: str = ""
    codes: tuple[FhirRoundTripCode, ...] = ()
    references: tuple[FhirRoundTripReference, ...] = ()

    def __post_init__(self) -> None:
        for value, label in (
            (self.span_id, "span id"),
            (self.resource_type, "resource type"),
            (self.resource_id, "resource id"),
        ):
            if not value.strip():
                raise ValueError(f"FHIR round-trip {label} must be non-empty")
        if not isinstance(self.text, str):
            raise TypeError("FHIR round-trip span text must be a string")
        if any(not isinstance(code, FhirRoundTripCode) for code in self.codes):
            raise TypeError("FHIR round-trip codes must be FhirRoundTripCode values")
        if any(
            not isinstance(reference, FhirRoundTripReference)
            for reference in self.references
        ):
            raise TypeError(
                "FHIR round-trip references must be FhirRoundTripReference values"
            )


@dataclass(frozen=True)
class FhirRoundTripFixture:
    """One validated synthetic Bundle round-trip fixture."""

    fixture_id: str
    bundle_type: str
    doc_id: str
    spans: tuple[FhirRoundTripSpan, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fixture_id.strip():
            raise ValueError("FHIR round-trip fixture id must be non-empty")
        if self.bundle_type not in {"transaction", "batch"}:
            raise ValueError(
                "FHIR round-trip bundle type must be 'transaction' or 'batch'"
            )
        if not self.doc_id.strip():
            raise ValueError("FHIR round-trip document id must be non-empty")
        if not self.spans:
            raise ValueError("FHIR round-trip fixture must contain spans")
        span_ids = [span.span_id for span in self.spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("FHIR round-trip span ids must be unique")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("FHIR round-trip fixture metadata must be a mapping")
        if self.metadata.get("synthetic", True) is not True:
            raise ValueError("FHIR round-trip fixtures must be synthetic")
        if self.metadata.get("contains_real_phi", False) is not False:
            raise ValueError("FHIR round-trip fixtures must not contain real PHI")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class FhirRoundTripScore:
    """PHI-safe fidelity measurements for one synthetic fixture."""

    fixture_id: str
    bundle_type: str
    resource_count: int
    parsed_resource_count: int
    matched_resource_count: int
    input_code_count: int
    preserved_code_count: int
    internal_reference_count: int
    resolved_reference_count: int
    dangling_reference_count: int
    resource_match_rate: float
    code_preservation_rate: float
    internal_reference_resolution_rate: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, JSON-compatible score fields."""

        return {
            "bundle_type": self.bundle_type,
            "code_preservation_rate": self.code_preservation_rate,
            "dangling_reference_count": self.dangling_reference_count,
            "fixture_id": self.fixture_id,
            "input_code_count": self.input_code_count,
            "internal_reference_count": self.internal_reference_count,
            "internal_reference_resolution_rate": (
                self.internal_reference_resolution_rate
            ),
            "matched_resource_count": self.matched_resource_count,
            "parsed_resource_count": self.parsed_resource_count,
            "passed": self.passed,
            "preserved_code_count": self.preserved_code_count,
            "resolved_reference_count": self.resolved_reference_count,
            "resource_count": self.resource_count,
            "resource_match_rate": self.resource_match_rate,
        }


def load_fhir_roundtrip_fixtures(
    path: str | Path | None = None,
) -> tuple[FhirRoundTripFixture, ...]:
    """Load and validate committed or caller-supplied synthetic JSONL fixtures.

    Args:
        path: JSONL path. The committed synthetic fixture is used when omitted.

    Returns:
        Fixtures in file order.

    Raises:
        ValueError: If the JSONL shape or safety metadata is invalid.
    """

    fixture_path = Path(path) if path is not None else FHIR_ROUNDTRIP_FIXTURE_PATH
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fixtures = tuple(
        _fixture_from_mapping(row, row_index=index) for index, row in enumerate(rows)
    )
    if not fixtures:
        raise ValueError("FHIR round-trip fixture file must contain at least one row")
    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("FHIR round-trip fixture ids must be unique")
    return fixtures


def build_fhir_resources(
    fixture: FhirRoundTripFixture,
) -> tuple[dict[str, Any], ...]:
    """Build standalone FHIR resources from one span/code fixture."""

    spans_by_id = {span.span_id: span for span in fixture.spans}
    resources: list[dict[str, Any]] = []
    for span in fixture.spans:
        resource: dict[str, Any] = {
            "resourceType": span.resource_type,
            "id": span.resource_id,
        }
        if span.codes:
            concept: dict[str, Any] = {
                "coding": [code.to_dict() for code in span.codes]
            }
            if span.text:
                concept["text"] = span.text
            resource[_CODE_FIELD_BY_RESOURCE.get(span.resource_type, "code")] = concept
        for reference in span.references:
            _set_reference(
                resource,
                reference.path,
                {"reference": _resolve_target(reference.target, spans_by_id)},
                many=reference.many,
            )
        resources.append(resource)
    return tuple(resources)


def build_fhir_bundle(fixture: FhirRoundTripFixture) -> dict[str, Any]:
    """Assemble fixture resources into a deterministic R4 Bundle."""

    return to_bundle(
        build_fhir_resources(fixture),
        doc_id=fixture.doc_id,
        bundle_type=fixture.bundle_type,
    )


def parse_fhir_bundle(bundle: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Parse Bundle entries back into standalone resource mappings.

    This deliberately performs only the local Bundle parse needed by the
    fidelity suite. It does not contact a FHIR server or run an HL7 validator.
    """

    if not isinstance(bundle, Mapping) or bundle.get("resourceType") != "Bundle":
        raise ValueError("FHIR round-trip parser requires a resourceType=Bundle")
    entries = bundle.get("entry", ())
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("FHIR Bundle entry must be a sequence")
    resources: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"FHIR Bundle entry at index {index} must be a mapping")
        resource = entry.get("resource")
        if not isinstance(resource, Mapping):
            raise ValueError(
                f"FHIR Bundle entry at index {index} is missing a resource mapping"
            )
        resources.append(dict(resource))
    return tuple(resources)


def score_fhir_roundtrip_fixture(
    fixture: FhirRoundTripFixture,
) -> FhirRoundTripScore:
    """Build, parse, and score one synthetic Bundle fixture."""

    source_resources = build_fhir_resources(fixture)
    bundle = build_fhir_bundle(fixture)
    parsed_resources = parse_fhir_bundle(bundle)

    source_keys = [
        (resource.get("resourceType"), resource.get("id"))
        for resource in source_resources
    ]
    parsed_keys = [
        (resource.get("resourceType"), resource.get("id"))
        for resource in parsed_resources
    ]
    matched_resource_count = sum(
        source_key == parsed_key
        for source_key, parsed_key in zip(source_keys, parsed_keys)
    )
    resource_denominator = max(len(source_resources), len(parsed_resources), 1)
    resource_match_rate = matched_resource_count / resource_denominator

    expected_codes = Counter(
        (code.system, code.code) for span in fixture.spans for code in span.codes
    )
    parsed_codes = Counter(_iter_codings(parsed_resources))
    preserved_code_count = sum(
        min(count, parsed_codes[code_key]) for code_key, count in expected_codes.items()
    )
    input_code_count = sum(expected_codes.values())
    code_preservation_rate = (
        preserved_code_count / input_code_count if input_code_count else 1.0
    )

    full_urls = {
        entry.get("fullUrl")
        for entry in bundle.get("entry", ())
        if isinstance(entry, Mapping) and isinstance(entry.get("fullUrl"), str)
    }
    references = tuple(_iter_references(parsed_resources))
    internal_references = tuple(
        reference for reference in references if not _is_external_reference(reference)
    )
    resolved_reference_count = sum(
        reference in full_urls for reference in internal_references
    )
    dangling_reference_count = len(internal_references) - resolved_reference_count
    internal_reference_resolution_rate = (
        resolved_reference_count / len(internal_references)
        if internal_references
        else 1.0
    )
    passed = (
        resource_match_rate == 1.0
        and code_preservation_rate == 1.0
        and dangling_reference_count == 0
    )
    return FhirRoundTripScore(
        fixture_id=fixture.fixture_id,
        bundle_type=fixture.bundle_type,
        resource_count=len(source_resources),
        parsed_resource_count=len(parsed_resources),
        matched_resource_count=matched_resource_count,
        input_code_count=input_code_count,
        preserved_code_count=preserved_code_count,
        internal_reference_count=len(internal_references),
        resolved_reference_count=resolved_reference_count,
        dangling_reference_count=dangling_reference_count,
        resource_match_rate=resource_match_rate,
        code_preservation_rate=code_preservation_rate,
        internal_reference_resolution_rate=internal_reference_resolution_rate,
        passed=passed,
    )


def score_fhir_roundtrip_fixtures(
    fixtures: Sequence[FhirRoundTripFixture] | None = None,
) -> dict[str, Any]:
    """Aggregate round-trip scores into report-ready metrics and metadata."""

    resolved_fixtures = (
        tuple(fixtures) if fixtures is not None else load_fhir_roundtrip_fixtures()
    )
    if not resolved_fixtures:
        raise ValueError("FHIR round-trip evaluation requires at least one fixture")
    scores = tuple(
        score_fhir_roundtrip_fixture(fixture) for fixture in resolved_fixtures
    )

    resource_count = sum(score.resource_count for score in scores)
    parsed_resource_count = sum(score.parsed_resource_count for score in scores)
    matched_resource_count = sum(score.matched_resource_count for score in scores)
    input_code_count = sum(score.input_code_count for score in scores)
    preserved_code_count = sum(score.preserved_code_count for score in scores)
    internal_reference_count = sum(score.internal_reference_count for score in scores)
    resolved_reference_count = sum(score.resolved_reference_count for score in scores)
    dangling_reference_count = sum(score.dangling_reference_count for score in scores)
    resource_match_rate = matched_resource_count / max(
        resource_count, parsed_resource_count, 1
    )
    code_preservation_rate = (
        preserved_code_count / input_code_count if input_code_count else 1.0
    )
    internal_reference_resolution_rate = (
        resolved_reference_count / internal_reference_count
        if internal_reference_count
        else 1.0
    )
    bundle_type_counts = Counter(score.bundle_type for score in scores)
    passed = (
        all(score.passed for score in scores)
        and resource_match_rate == 1.0
        and code_preservation_rate == 1.0
        and dangling_reference_count == 0
    )
    return {
        "metrics": {
            "bundle_type_counts": dict(sorted(bundle_type_counts.items())),
            "code_preservation_rate": code_preservation_rate,
            "dangling_reference_count": dangling_reference_count,
            "input_code_count": input_code_count,
            "internal_reference_count": internal_reference_count,
            "internal_reference_resolution_rate": (internal_reference_resolution_rate),
            "matched_resource_count": matched_resource_count,
            "parsed_resource_count": parsed_resource_count,
            "passed": passed,
            "preserved_code_count": preserved_code_count,
            "resolved_reference_count": resolved_reference_count,
            "resource_count": resource_count,
            "resource_match_rate": resource_match_rate,
            "per_fixture": {score.fixture_id: score.to_dict() for score in scores},
        },
        "metadata": {
            "bundle_types": sorted(bundle_type_counts),
            "contains_real_phi": False,
            "fhir_version": FHIR_R4_VERSION,
            "fixture_ids": [score.fixture_id for score in scores],
            "reference_policy": (
                "internal references must resolve to a Bundle fullUrl; absolute "
                "external references are excluded"
            ),
            "synthetic": True,
        },
        "scores": [score.to_dict() for score in scores],
    }


def run_fhir_roundtrip_suite(
    *,
    fixture_path: str | Path | None = None,
    fixtures: Sequence[FhirRoundTripFixture] | None = None,
    model_name: str = "deterministic-fhir-roundtrip",
    device: str = "cpu",
    generated_at: str | None = None,
    raise_on_failure: bool = False,
) -> BenchmarkReport:
    """Run the offline synthetic FHIR round-trip fidelity suite.

    Args:
        fixture_path: Optional JSONL fixture path.
        fixtures: Optional already-loaded fixtures. Cannot be combined with
            ``fixture_path``.
        model_name: Report model label; no model inference is performed.
        device: Report device label.
        generated_at: Optional caller-supplied report timestamp.
        raise_on_failure: Raise from :func:`assert_fhir_roundtrip_fidelity` when
            the report does not pass. By default, failures are reported in the
            metrics so callers can inspect the evidence.
    """

    if fixture_path is not None and fixtures is not None:
        raise ValueError("provide fixture_path or fixtures, not both")
    resolved_fixtures = (
        tuple(fixtures)
        if fixtures is not None
        else load_fhir_roundtrip_fixtures(fixture_path)
    )
    scored = score_fhir_roundtrip_fixtures(resolved_fixtures)
    report = BenchmarkReport(
        suite=FHIR_ROUNDTRIP,
        model_name=model_name,
        device=device,
        fixture_count=len(resolved_fixtures),
        metrics=scored["metrics"],
        generated_at=generated_at,
        metadata=scored["metadata"],
    )
    if raise_on_failure:
        assert_fhir_roundtrip_fidelity(report)
    return report


def run_fhir_roundtrip(**kwargs: Any) -> BenchmarkReport:
    """Compatibility alias for :func:`run_fhir_roundtrip_suite`."""

    return run_fhir_roundtrip_suite(**kwargs)


def assert_fhir_roundtrip_fidelity(report: BenchmarkReport) -> None:
    """Raise when a round-trip report fails its fidelity gate."""

    metrics = report.metrics
    failures: list[str] = []
    if metrics.get("resource_match_rate") != 1.0:
        failures.append("resource_match_rate")
    if metrics.get("code_preservation_rate") != 1.0:
        failures.append("code_preservation_rate")
    if metrics.get("dangling_reference_count", 0) != 0:
        failures.append("dangling_reference_count")
    if failures:
        raise AssertionError(
            "FHIR round-trip fidelity check failed: " + ", ".join(failures)
        )


def _fixture_from_mapping(
    row: Mapping[str, Any],
    *,
    row_index: int,
) -> FhirRoundTripFixture:
    if not isinstance(row, Mapping):
        raise ValueError(f"FHIR round-trip fixture row {row_index} must be an object")
    schema_version = int(row.get("schema_version", FHIR_ROUNDTRIP_SCHEMA_VERSION))
    if schema_version != FHIR_ROUNDTRIP_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported FHIR round-trip fixture schema: {schema_version}"
        )
    fixture_id = _required_string(row, "fixture_id", fallback_key="id")
    bundle_type = _required_string(row, "bundle_type").casefold()
    doc_id = _required_string(row, "doc_id", fallback_value=fixture_id)
    raw_spans = row.get("spans")
    if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, (str, bytes)):
        raise ValueError(f"FHIR round-trip fixture {fixture_id} spans must be a list")
    spans = tuple(
        _span_from_mapping(payload, fixture_id=fixture_id, span_index=index)
        for index, payload in enumerate(raw_spans)
    )
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"FHIR round-trip fixture {fixture_id} metadata must be an object"
        )
    return FhirRoundTripFixture(
        fixture_id=fixture_id,
        bundle_type=bundle_type,
        doc_id=doc_id,
        spans=spans,
        metadata=dict(metadata),
    )


def _span_from_mapping(
    payload: Mapping[str, Any],
    *,
    fixture_id: str,
    span_index: int,
) -> FhirRoundTripSpan:
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"FHIR round-trip fixture {fixture_id} span {span_index} must be an object"
        )
    span_id = _required_string(payload, "span_id", fallback_key="spanId")
    resource_type = _required_string(
        payload,
        "resource_type",
        fallback_key="resourceType",
    )
    resource_id = _required_string(
        payload,
        "resource_id",
        fallback_key="resourceId",
        fallback_value=payload.get("id"),
    )
    text = payload.get("text", "")
    if not isinstance(text, str):
        raise ValueError(
            f"FHIR round-trip fixture {fixture_id} span {span_index} text must be a string"
        )
    raw_codes = payload.get("codes", payload.get("codings", ()))
    codes = tuple(_code_from_value(value) for value in _code_values(raw_codes))
    raw_references = payload.get("references", ())
    references = tuple(
        _reference_from_value(value) for value in _reference_values(raw_references)
    )
    return FhirRoundTripSpan(
        span_id=span_id,
        resource_type=resource_type,
        resource_id=resource_id,
        text=text,
        codes=codes,
        references=references,
    )


def _code_values(raw_codes: Any) -> tuple[Mapping[str, Any], ...]:
    if raw_codes is None:
        return ()
    if isinstance(raw_codes, Mapping):
        if "system" in raw_codes or "code" in raw_codes:
            return (raw_codes,)
        return tuple(
            {"system": system, "code": code} for system, code in raw_codes.items()
        )
    if not isinstance(raw_codes, Sequence) or isinstance(raw_codes, (str, bytes)):
        raise ValueError("FHIR round-trip codes must be a list or object")
    if any(not isinstance(value, Mapping) for value in raw_codes):
        raise ValueError("FHIR round-trip code entries must be objects")
    return tuple(raw_codes)


def _code_from_value(value: Mapping[str, Any]) -> FhirRoundTripCode:
    raw_system = value.get("system")
    raw_code = value.get("code")
    if not isinstance(raw_system, str) or not isinstance(raw_code, str):
        raise ValueError("FHIR round-trip codes require string system and code")
    system_key = _SYSTEM_ALIASES.get(raw_system.strip().upper(), raw_system.strip())
    try:
        canonical_system = system_uri(system_key)
    except ValueError:
        if not system_key.startswith(("http://", "https://")):
            raise ValueError(
                "FHIR round-trip code systems must be known vocabulary ids or URIs"
            ) from None
        canonical_system = system_key
    display = value.get("display")
    if display is not None and not isinstance(display, str):
        raise ValueError("FHIR round-trip code display must be a string")
    return FhirRoundTripCode(canonical_system, raw_code, display)


def _reference_values(raw_references: Any) -> tuple[Any, ...]:
    if raw_references is None:
        return ()
    if isinstance(raw_references, Mapping):
        if "path" in raw_references or "target" in raw_references:
            return (raw_references,)
        return tuple(
            {"path": path, "target": target} for path, target in raw_references.items()
        )
    if not isinstance(raw_references, Sequence) or isinstance(
        raw_references, (str, bytes)
    ):
        raise ValueError("FHIR round-trip references must be a list or object")
    return tuple(raw_references)


def _reference_from_value(value: Any) -> FhirRoundTripReference:
    if not isinstance(value, Mapping):
        raise ValueError("FHIR round-trip reference entries must be objects")
    path = value.get("path", value.get("field"))
    target = value.get("target", value.get("target_span_id"))
    if target is None:
        target = value.get("reference")
    if not isinstance(path, str) or not isinstance(target, str):
        raise ValueError("FHIR round-trip references require string path and target")
    return FhirRoundTripReference(
        path=path,
        target=target,
        many=bool(value.get("many", False)),
    )


def _required_string(
    payload: Mapping[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
    fallback_value: Any = None,
) -> str:
    value = payload.get(key)
    if value is None and fallback_key is not None:
        value = payload.get(fallback_key)
    if value is None:
        value = fallback_value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"FHIR round-trip fixture field {key!r} must be non-empty")
    return value.strip()


def _resolve_target(
    target: str,
    spans_by_id: Mapping[str, FhirRoundTripSpan],
) -> str:
    span = spans_by_id.get(target)
    if span is not None:
        return f"{span.resource_type}/{span.resource_id}"
    return target


def _set_reference(
    resource: dict[str, Any],
    path: str,
    reference: dict[str, str],
    *,
    many: bool,
) -> None:
    parts = tuple(part for part in path.split(".") if part)
    if not parts or len(parts) != len(path.split(".")):
        raise ValueError("FHIR round-trip reference paths must not contain empty parts")
    current: dict[str, Any] = resource
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            nested: dict[str, Any] = {}
            current[part] = nested
            current = nested
        elif isinstance(existing, dict):
            current = existing
        else:
            raise ValueError("FHIR round-trip reference path collides with a value")
    field_name = parts[-1]
    if many:
        existing = current.get(field_name)
        if existing is None:
            current[field_name] = [reference]
        elif isinstance(existing, list):
            existing.append(reference)
        else:
            current[field_name] = [existing, reference]
    else:
        current[field_name] = reference


def _iter_codings(node: Any):
    if isinstance(node, Mapping):
        if isinstance(node.get("system"), str) and isinstance(node.get("code"), str):
            yield node["system"], node["code"]
        for value in node.values():
            yield from _iter_codings(value)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for value in node:
            yield from _iter_codings(value)


def _iter_references(node: Any):
    if isinstance(node, Mapping):
        reference = node.get("reference")
        if isinstance(reference, str) and reference:
            yield reference
        for value in node.values():
            yield from _iter_references(value)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for value in node:
            yield from _iter_references(value)


def _is_external_reference(reference: str) -> bool:
    return reference.startswith(("http://", "https://", "mailto:", "#"))
