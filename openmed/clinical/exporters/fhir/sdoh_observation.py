"""Passive, bounded FHIR R4 projection of value-free SDOH evidence.

Only HL7 category codes are bundled. Clinical terminology and opaque reference
mappings belong to the trusted caller. No extraction, server contact, credential
resolution, approval verification or storage occurs here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from itertools import islice
from typing import Any
from urllib.parse import urlsplit

from openmed.core.iso_temporal import parse_iso_date, parse_iso_datetime

from ...sdoh_evidence import (
    ASSERTION_STATUSES,
    EVIDENCE_TYPES,
    REVIEW_STATUSES,
    SOURCE_SECTIONS,
    SDOHEvidence,
)
from ...sdoh_experiencer import SDOHExperiencerEvidence
from ...sdoh_sensitive_use import (
    ProhibitedAutomatedUse,
    SDOHPurpose,
    SDOHSensitiveUseLabel,
)
from ...sdoh_temporal import SDOHTemporalEvidence
from .validate import validate_resource

SDOH_FHIR_IG_VERSION = "2.3.0"
SDOH_FHIR_US_CORE_VERSION = "7.0.0"
_DOMAIN_SYSTEM = "http://hl7.org/fhir/us/sdoh-clinicalcare/CodeSystem/SDOHCC-CodeSystemTemporaryCodes"
_SECURITY_SYSTEM = "https://openmed.dev/fhir/CodeSystem/sdoh-sensitive-use"
_EXTENSION = "https://openmed.dev/fhir/StructureDefinition/sdoh-evidence"
_DOMAIN_CODES = {
    "food_insecurity": "food-insecurity",
    "housing": "housing-instability",
    "housing_insecurity": "housing-instability",
    "employment": "employment-status",
    "employment_status": "employment-status",
    "financial_strain": "financial-insecurity",
    "transportation": "transportation-insecurity",
    "education": "educational-attainment",
    "insurance": "health-insurance-coverage-status",
    "social_support": "social-connection",
    "utilities": "utility-insecurity",
}
_HOUSING_DOMAINS = {"housing-instability", "homelessness", "inadequate-housing"}
_REFERENCE = re.compile(
    r"urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}\Z")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_DATETIME = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])\Z"
)


class SDOHFHIRExportError(ValueError):
    """Controlled export configuration error with no input values.

    Args:
        code: A fixed export error code; unknown codes become invalid_input.
    """

    def __init__(self, code: str) -> None:
        allowed = {
            "invalid_input",
            "invalid_code",
            "invalid_record",
            "offset_mismatch",
            "invalid_effective_time",
            "invalid_reference",
            "invalid_terminology",
            "invalid_purpose",
            "invalid_clock",
            "clock_failed",
            "invalid_resource",
        }
        self.code = code if type(code) is str and code in allowed else "invalid_input"
        super().__init__(self.code)


@dataclass(frozen=True, slots=True)
class SDOHFHIRCode:
    """Caller-approved terminology binding without display or source surfaces.

    Args:
        system: Trusted canonical HTTP(S)/URN code-system URI, never dereferenced.
        code: Approved ASCII code token. Syntax does not prove terminology validity
            or absence of identifiers; never construct bindings from source text.
    """

    system: str
    code: str

    def __post_init__(self) -> None:
        valid = False
        if type(self.system) is str and 0 < len(self.system) <= 256:
            try:
                parsed = urlsplit(self.system)
                valid = (
                    self.system.isascii()
                    and all(33 <= ord(c) <= 126 for c in self.system)
                    and not any(c in self.system for c in "%\\?#")
                    and parsed.scheme in {"http", "https", "urn"}
                    and not parsed.username
                    and not parsed.password
                    and (
                        bool(parsed.netloc)
                        if parsed.scheme != "urn"
                        else bool(parsed.path)
                    )
                )
            except ValueError:
                pass
        if not valid or type(self.code) is not str or not _CODE.fullmatch(self.code):
            raise SDOHFHIRExportError("invalid_code")


@dataclass(frozen=True, slots=True)
class SDOHObservationBinding:
    """Explicit observation concept and optional answer from trusted configuration.

    Args:
        code: Type of observation, not inferred from determinant text.
        value: Optional approved answer concept. Missing answers produce a
            preliminary Observation with dataAbsentReason and an explicit loss.
        domain_category: Optional pinned HL7 domain code. Housing requires an
            explicit selection because generic housing evidence does not identify
            instability, homelessness or inadequate housing by itself.
    """

    code: SDOHFHIRCode
    value: SDOHFHIRCode | None = None
    domain_category: str | None = None

    def __post_init__(self) -> None:
        if type(self.code) is not SDOHFHIRCode or (
            self.value is not None and type(self.value) is not SDOHFHIRCode
        ):
            raise SDOHFHIRExportError("invalid_terminology")
        if self.domain_category is not None and (
            type(self.domain_category) is not str
            or self.domain_category
            not in set(_DOMAIN_CODES.values()) | _HOUSING_DOMAINS
        ):
            raise SDOHFHIRExportError("invalid_terminology")


def _effective(value: Any) -> tuple[str, date | datetime]:
    normalized = None
    parsed: date | datetime | None = None
    if type(value) is str:
        try:
            if _DATE.fullmatch(value):
                parsed = parse_iso_date(value)
                normalized = parsed.isoformat()
            elif _DATETIME.fullmatch(value):
                timestamp: datetime = parse_iso_datetime(value)
                parsed = timestamp.astimezone(timezone.utc)
                normalized = parsed.isoformat().replace("+00:00", "Z")
        except (ValueError, OverflowError):
            pass
    if normalized is None or parsed is None:
        raise SDOHFHIRExportError("invalid_effective_time")
    return normalized, parsed


@dataclass(frozen=True, slots=True)
class SDOHFHIRRecord:
    """Aligned evidence and explicit qualifiers for one passive export candidate.

    Args:
        evidence: Value-free SDOH evidence, including assertion and review state.
        experiencer: Explicit upstream subject attribution for the same offsets.
        temporal: Upstream temporal qualifier for the same offsets.
        effective_start: Optional approved ISO date or aware datetime. Qualifier
            classes alone supply no calendar date and never use the wall clock.
        effective_end: Optional matching-precision end, no earlier than start.
        use_label: Restrictive sensitive-use policy for the entire SDOH record.

    The caller owns review authenticity and approved temporal disclosure,
    including date shifting where needed. No source text or raw finding value
    is accepted. Unknown/historical/future temporal states cannot become final.
    """

    evidence: SDOHEvidence
    experiencer: SDOHExperiencerEvidence
    temporal: SDOHTemporalEvidence
    effective_start: str | None = None
    effective_end: str | None = None
    use_label: SDOHSensitiveUseLabel = field(
        default_factory=lambda: SDOHSensitiveUseLabel("sdoh")
    )

    def __post_init__(self) -> None:
        if (
            type(self.evidence) is not SDOHEvidence
            or type(self.experiencer) is not SDOHExperiencerEvidence
            or type(self.temporal) is not SDOHTemporalEvidence
            or type(self.use_label) is not SDOHSensitiveUseLabel
            or self.use_label.field_name != "sdoh"
        ):
            raise SDOHFHIRExportError("invalid_record")
        closed = (
            (self.evidence.evidence_type, EVIDENCE_TYPES),
            (self.evidence.assertion, ASSERTION_STATUSES),
            (self.evidence.source_section, SOURCE_SECTIONS),
            (self.evidence.review_status, REVIEW_STATUSES),
            (
                self.experiencer.experiencer,
                ("patient", "household", "family", "unknown"),
            ),
            (
                self.temporal.temporal_class,
                ("current", "historical", "future", "unknown"),
            ),
        )
        if any(
            type(value) is not str or value not in allowed for value, allowed in closed
        ):
            raise SDOHFHIRExportError("invalid_record")
        spans = (
            self.evidence.span,
            self.experiencer.source_offsets,
            self.temporal.source_offsets,
        )
        if len(set(spans)) != 1 or any(
            type(n) is not int or not 0 <= n <= 2**31 - 1
            for span in spans
            for n in span
        ):
            raise SDOHFHIRExportError("offset_mismatch")
        if self.effective_start is None:
            if self.effective_end is not None:
                raise SDOHFHIRExportError("invalid_effective_time")
            return
        start, parsed_start = _effective(self.effective_start)
        if self.temporal.temporal_class == "unknown" or self.temporal.has_conflict:
            raise SDOHFHIRExportError("invalid_effective_time")
        object.__setattr__(self, "effective_start", start)
        if self.effective_end is not None:
            end, parsed_end = _effective(self.effective_end)
            if type(parsed_start) is not type(parsed_end) or parsed_end < parsed_start:
                raise SDOHFHIRExportError("invalid_effective_time")
            object.__setattr__(self, "effective_end", end)


@dataclass(frozen=True, slots=True)
class SDOHFHIRExportResult:
    """Local resources and controlled losses, without transport or storage.

    Args:
        observations: Fresh passive FHIR resources.
        provenance: Matching evidence-offset Provenance resources.
        losses: Input indices and controlled loss codes. excluded=False denotes
            partial answer loss on a retained preliminary Observation.
    """

    observations: tuple[dict[str, Any], ...]
    provenance: tuple[dict[str, Any], ...]
    losses: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible projection and explicit counts."""
        return copy.deepcopy(
            {
                "ig_version": SDOH_FHIR_IG_VERSION,
                "observations": list(self.observations),
                "provenance": list(self.provenance),
                "losses": list(self.losses),
                "exported_count": len(self.observations),
                "excluded_count": sum(loss["excluded"] for loss in self.losses),
            }
        )


def _coding(system: str, code: str, version: str | None = None) -> dict[str, str]:
    coding = {"system": system, "code": code}
    if version is not None:
        coding["version"] = version
    return coding


def _concept(code: SDOHFHIRCode) -> dict[str, Any]:
    return {"coding": [_coding(code.system, code.code)]}


def _loss(record: SDOHFHIRRecord, purpose: SDOHPurpose) -> str | None:
    label = record.use_label
    if (
        purpose not in label.allowed_purposes
        or not label.human_review_required
        or set(label.prohibited_automated_uses) != set(ProhibitedAutomatedUse)
    ):
        return "sensitive_use_refused"
    if not record.experiencer.patient_record_eligible:
        return "non_patient_or_unresolved"
    evidence = record.evidence
    if evidence.assertion == "absent":
        return "negated_need"
    if evidence.assertion != "present":
        return "assertion_unconfirmed"
    if evidence.review_status in {"rejected", "unknown", "refused"}:
        return "review_refused"
    if evidence.evidence_type in {"unknown", "refused"}:
        return "evidence_unconfirmed"
    if evidence.determinant not in _DOMAIN_CODES:
        return "domain_unmapped"
    return None


def _recorded(clock: Callable[[], datetime]) -> str:
    value: Any = None
    failed = False
    try:
        value = clock()
    except Exception:
        failed = True
    if failed:
        raise SDOHFHIRExportError("clock_failed")
    result = None
    try:
        if (
            type(value) is datetime
            and value.tzinfo is not None
            and value.utcoffset() is not None
        ):
            result = (
                value.astimezone(timezone.utc)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z")
            )
    except Exception:
        pass
    if result is None:
        raise SDOHFHIRExportError("invalid_clock")
    return result


def to_sdoh_observations(
    records: Sequence[SDOHFHIRRecord],
    *,
    terminology: Mapping[int, SDOHObservationBinding],
    subject_reference: str,
    source_reference: str,
    software_reference: str,
    purpose: SDOHPurpose,
    clock: Callable[[], datetime],
) -> SDOHFHIRExportResult:
    """Project bounded evidence into local R4 Observations and Provenance.

    Args:
        records: At most 512 aligned typed records, not raw findings or excerpts.
        terminology: Caller-approved bindings keyed by the input record's integer
            index, at most 512. Answers may differ within one determinant.
            Missing codes are explicit exclusions; missing answers are partial losses.
        subject_reference: Opaque UUIDv4 URN for the Patient.
        source_reference: Opaque UUIDv4 URN for the local DocumentReference.
        software_reference: Opaque UUIDv4 URN for the projecting Device.
        purpose: Explicit allowed sensitive-use purpose; no automated decisions.
        clock: Injected technical recorded clock, called once only if exporting.

    Returns:
        Passive resources and controlled indexed losses. Reviewed, current,
        affirmative patient evidence with an answer can be final; every other
        retained candidate remains preliminary. Categories pin SDOHCC 2.3.0
        and US Core 7.0.0; no full IG conformance declaration is made.

    Raises:
        SDOHFHIRExportError: With a fixed code for invalid configuration.
    """
    if (
        type(records) not in {tuple, list}
        or len(records) > 512
        or any(type(record) is not SDOHFHIRRecord for record in records)
    ):
        raise SDOHFHIRExportError("invalid_input")
    if type(purpose) is not SDOHPurpose or not callable(clock):
        raise SDOHFHIRExportError(
            "invalid_purpose" if type(purpose) is not SDOHPurpose else "invalid_clock"
        )
    for reference in (subject_reference, source_reference, software_reference):
        if type(reference) is not str or not _REFERENCE.fullmatch(reference):
            raise SDOHFHIRExportError("invalid_reference")
    if len({subject_reference, source_reference, software_reference}) != 3:
        raise SDOHFHIRExportError("invalid_reference")
    copied: dict[int, SDOHObservationBinding] = {}
    failed = False
    try:
        if not isinstance(terminology, Mapping) or len(terminology) > 512:
            raise ValueError
        for key, value in islice(terminology.items(), 513):
            if (
                type(key) is not int
                or not 0 <= key < len(records)
                or key in copied
                or type(value) is not SDOHObservationBinding
                or len(copied) >= 512
            ):
                raise ValueError
            copied[key] = value
    except Exception:
        failed = True
    if failed:
        raise SDOHFHIRExportError("invalid_terminology")
    terminology = copied
    losses: list[dict[str, Any]] = []
    selected = []
    for index, record in enumerate(records):
        reason = _loss(record, purpose)
        binding = terminology.get(index)
        if reason is None and binding is None:
            reason = "terminology_unmapped"
        domain = None
        if reason is None and binding is not None:
            determinant = record.evidence.determinant or ""
            allowed_domains = (
                _HOUSING_DOMAINS
                if determinant in {"housing", "housing_insecurity"}
                else {_DOMAIN_CODES[determinant]}
            )
            domain = binding.domain_category
            if domain is None:
                if len(allowed_domains) == 1:
                    domain = next(iter(allowed_domains))
                else:
                    reason = "domain_ambiguous"
            elif domain not in allowed_domains:
                reason = "domain_binding_mismatch"
        if reason is not None:
            losses.append({"input_index": index, "code": reason, "excluded": True})
        else:
            selected.append((index, record, binding, domain))
    if not selected:
        return SDOHFHIRExportResult((), (), tuple(losses))
    recorded = _recorded(clock)
    observations: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record, binding, domain in selected:
        assert binding is not None and domain is not None
        evidence, temporal = record.evidence, record.temporal
        final = (
            evidence.review_status == "reviewed"
            and temporal.temporal_class == "current"
            and not temporal.review_required
            and not temporal.has_conflict
            and binding.value is not None
        )
        security = [
            _coding(_SECURITY_SYSTEM, "sensitive-sdoh"),
            _coding(_SECURITY_SYSTEM, "human-review-required"),
        ]
        security.extend(
            _coding(_SECURITY_SYSTEM, "allowed-" + p.value)
            for p in record.use_label.allowed_purposes
        )
        security.extend(
            _coding(_SECURITY_SYSTEM, "prohibited-" + p.value)
            for p in record.use_label.prohibited_automated_uses
        )
        extensions = [
            {"url": "start", "valueUnsignedInt": evidence.span[0]},
            {"url": "end", "valueUnsignedInt": evidence.span[1]},
            {"url": "evidence-type", "valueCode": evidence.evidence_type},
            {"url": "review-status", "valueCode": evidence.review_status},
            {"url": "temporal-class", "valueCode": temporal.temporal_class},
            {"url": "assertion", "valueCode": evidence.assertion},
            {"url": "source-section", "valueCode": evidence.source_section},
            {"url": "experiencer", "valueCode": record.experiencer.experiencer},
        ]
        resource: dict[str, Any] = {
            "resourceType": "Observation",
            "status": "final" if final else "preliminary",
            "meta": {"security": copy.deepcopy(security)},
            "category": [
                {
                    "coding": [
                        _coding(
                            "http://terminology.hl7.org/CodeSystem/observation-category",
                            "social-history",
                        )
                    ]
                },
                {
                    "coding": [
                        _coding(
                            "http://hl7.org/fhir/us/core/CodeSystem/us-core-category",
                            "sdoh",
                            SDOH_FHIR_US_CORE_VERSION,
                        )
                    ]
                },
                {
                    "coding": [
                        _coding(
                            _DOMAIN_SYSTEM,
                            domain,
                            SDOH_FHIR_IG_VERSION,
                        )
                    ]
                },
            ],
            "code": _concept(binding.code),
            "subject": {"reference": subject_reference, "type": "Patient"},
            "derivedFrom": [
                {"reference": source_reference, "type": "DocumentReference"}
            ],
            "extension": [{"url": _EXTENSION, "extension": copy.deepcopy(extensions)}],
        }
        if binding.value is not None:
            resource["valueCodeableConcept"] = _concept(binding.value)
        else:
            resource["dataAbsentReason"] = {
                "coding": [
                    _coding(
                        "http://terminology.hl7.org/CodeSystem/data-absent-reason",
                        "unknown",
                    )
                ]
            }
        if record.effective_start is not None:
            if record.effective_end is None:
                resource["effectiveDateTime"] = record.effective_start
            else:
                resource["effectivePeriod"] = {
                    "start": record.effective_start,
                    "end": record.effective_end,
                }
        digest = hashlib.sha256(
            json.dumps(
                {"observation": resource, "software": software_reference},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:40]
        resource["id"] = "openmed-sdoh-" + digest
        if digest in seen:
            losses.append(
                {"input_index": index, "code": "duplicate_input", "excluded": True}
            )
            continue
        seen.add(digest)
        if binding.value is None:
            losses.append(
                {"input_index": index, "code": "answer_unmapped", "excluded": False}
            )
        if not validate_resource(resource).is_valid:
            raise SDOHFHIRExportError("invalid_resource")
        observations.append(resource)
        prov: dict[str, Any] = {
            "resourceType": "Provenance",
            "meta": {"security": copy.deepcopy(security)},
            "target": [{"reference": "Observation/" + resource["id"]}],
            "recorded": recorded,
            "agent": [{"who": {"reference": software_reference, "type": "Device"}}],
            "entity": [
                {
                    "role": "source",
                    "what": {
                        "reference": source_reference,
                        "type": "DocumentReference",
                        "extension": [{"url": _EXTENSION, "extension": extensions}],
                    },
                }
            ],
        }
        prov["id"] = (
            "openmed-sdoh-prov-"
            + hashlib.sha256(
                json.dumps(prov, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:40]
        )
        provenance.append(prov)
    losses.sort(key=lambda loss: loss["input_index"])
    return SDOHFHIRExportResult(tuple(observations), tuple(provenance), tuple(losses))


__all__ = [
    "SDOH_FHIR_IG_VERSION",
    "SDOH_FHIR_US_CORE_VERSION",
    "SDOHFHIRCode",
    "SDOHFHIRExportError",
    "SDOHFHIRExportResult",
    "SDOHFHIRRecord",
    "SDOHObservationBinding",
    "to_sdoh_observations",
]
