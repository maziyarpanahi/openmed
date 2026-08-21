"""Fail-closed, local FHIR Bulk Data NDJSON privacy gateway.

This module implements the local processing half of FHIR Bulk Data 3.0.0.
Resources are handled one at a time, output files are committed atomically,
and checkpoints contain only hashes, counts, and structural metadata.  The
module never logs or includes raw resource content in an error or report.

Network authentication and asynchronous SMART export polling live in
``openmed.service.smart_backend``.  That service uses the streaming functions
here so the network boundary cannot bypass the same safety checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, TextIO

from ..fhir_operations import Deidentifier, de_identify_resource

__all__ = [
    "BULK_DATA_VERSION",
    "DEFAULT_MAX_BUFFERED_RESOURCES",
    "SUPPORTED_R4_RESOURCE_TYPES",
    "BulkDataGateway",
    "FHIRBulkGateway",
    "BulkExportSummary",
    "BulkGatewayConfig",
    "FHIRBulkConfig",
    "BulkJobCancelled",
    "BulkRejection",
    "FHIRBulkJobReport",
    "FHIRNDJSONLineError",
    "NDJSONFileSummary",
    "NDJSONLineError",
    "RejectionRecord",
    "deidentify_export",
    "deidentify_ndjson",
    "deidentify_ndjson_async",
    "deidentify_ndjson_stream",
    "iter_ndjson",
]

BULK_DATA_VERSION = "3.0.0"
DEFAULT_MAX_BUFFERED_RESOURCES = 1
_DEFAULT_POLICY = "hipaa_safe_harbor"
_DEFAULT_METHOD = "replace"
_CHECKPOINT_VERSION = 1

# The gateway deliberately uses an allow-list.  Unknown resource types are
# rejected rather than copied through as an apparently de-identified export.
# Binary is intentionally absent even though it is a valid R4 resource: the
# gateway cannot inspect arbitrary base64 payloads safely.
SUPPORTED_R4_RESOURCE_TYPES = frozenset(
    {
        "Account",
        "ActivityDefinition",
        "AdverseEvent",
        "AllergyIntolerance",
        "Appointment",
        "AppointmentResponse",
        "AuditEvent",
        "Basic",
        "BodyStructure",
        "CarePlan",
        "CareTeam",
        "CatalogEntry",
        "ChargeItem",
        "ChargeItemDefinition",
        "Claim",
        "ClaimResponse",
        "ClinicalImpression",
        "CodeSystem",
        "Communication",
        "CommunicationRequest",
        "CompartmentDefinition",
        "Composition",
        "ConceptMap",
        "Condition",
        "Consent",
        "Contract",
        "Coverage",
        "DetectedIssue",
        "Device",
        "DeviceDefinition",
        "DeviceMetric",
        "DeviceRequest",
        "DeviceUseStatement",
        "DiagnosticReport",
        "DocumentManifest",
        "DocumentReference",
        "DomainResource",
        "EffectEvidenceSynthesis",
        "Encounter",
        "Endpoint",
        "EnrollmentRequest",
        "EnrollmentResponse",
        "EpisodeOfCare",
        "EventDefinition",
        "Evidence",
        "EvidenceVariable",
        "ExampleScenario",
        "ExplanationOfBenefit",
        "FamilyMemberHistory",
        "Flag",
        "Goal",
        "GraphDefinition",
        "Group",
        "GuidanceResponse",
        "HealthcareService",
        "ImagingStudy",
        "Immunization",
        "ImmunizationEvaluation",
        "ImmunizationRecommendation",
        "ImplementationGuide",
        "InsurancePlan",
        "Invoice",
        "Library",
        "Linkage",
        "List",
        "Location",
        "Measure",
        "MeasureReport",
        "Media",
        "Medication",
        "MedicationAdministration",
        "MedicationDispense",
        "MedicationKnowledge",
        "MedicationRequest",
        "MedicationStatement",
        "MedicinalProduct",
        "MedicinalProductAuthorization",
        "MedicinalProductContraindication",
        "MedicinalProductIndication",
        "MedicinalProductIngredient",
        "MedicinalProductInteraction",
        "MedicinalProductManufactured",
        "MedicinalProductPackaged",
        "MedicinalProductPharmaceutical",
        "MedicinalProductUndesirableEffect",
        "MessageDefinition",
        "MessageHeader",
        "MolecularSequence",
        "NamingSystem",
        "NutritionOrder",
        "Observation",
        "OperationDefinition",
        "OperationOutcome",
        "Organization",
        "OrganizationAffiliation",
        "Patient",
        "PaymentNotice",
        "PaymentReconciliation",
        "Person",
        "PlanDefinition",
        "Practitioner",
        "PractitionerRole",
        "Procedure",
        "Provenance",
        "Questionnaire",
        "QuestionnaireResponse",
        "RelatedPerson",
        "RequestGroup",
        "ResearchDefinition",
        "ResearchElementDefinition",
        "ResearchStudy",
        "ResearchSubject",
        "RiskAssessment",
        "RiskEvidenceSynthesis",
        "Schedule",
        "SearchParameter",
        "ServiceRequest",
        "Slot",
        "Specimen",
        "StructureDefinition",
        "StructureMap",
        "Subscription",
        "Substance",
        "SupplyDelivery",
        "SupplyRequest",
        "Task",
        "TerminologyCapabilities",
        "TestReport",
        "TestScript",
        "ValueSet",
        "VerificationResult",
        "VisionPrescription",
        "Bundle",
        "Parameters",
    }
)

_SAFE_NARRATIVE_TAGS = frozenset(
    {
        "a",
        "abbr",
        "b",
        "blockquote",
        "br",
        "caption",
        "cite",
        "code",
        "col",
        "colgroup",
        "dd",
        "del",
        "div",
        "dl",
        "dt",
        "em",
        "i",
        "ins",
        "kbd",
        "li",
        "ol",
        "p",
        "pre",
        "q",
        "samp",
        "small",
        "span",
        "strike",
        "strong",
        "sub",
        "sup",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "tt",
        "ul",
        "var",
    }
)
_UNSAFE_NARRATIVE_TAGS = frozenset(
    {"embed", "form", "iframe", "object", "script", "style"}
)


class BulkJobCancelled(RuntimeError):
    """Raised when a bulk job is cancelled before its next atomic commit."""


@dataclass(frozen=True)
class BulkRejection:
    """A PHI-free record for a resource that was not written."""

    line_number: int
    reason: str
    resource_sha256: str
    resource_type: str | None = None
    path: str | None = None

    @property
    def reason_code(self) -> str:
        """Return a stable machine-readable rejection reason."""

        return self.reason

    def to_dict(self) -> dict[str, Any]:
        """Return a report-safe rejection mapping."""

        payload: dict[str, Any] = {
            "line_number": self.line_number,
            "reason": self.reason,
            "resource_sha256": self.resource_sha256,
        }
        if self.resource_type is not None:
            payload["resource_type"] = _safe_resource_type(self.resource_type)
        if self.path is not None:
            payload["path"] = _safe_structural_path(self.path)
        return payload


RejectionRecord = BulkRejection


@dataclass(frozen=True)
class NDJSONLineError:
    """PHI-free description of malformed or failed NDJSON input."""

    line_number: int
    message: str
    resource_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a report-safe error mapping."""

        payload: dict[str, Any] = {
            "line_number": self.line_number,
            "message": self.message,
        }
        if self.resource_sha256 is not None:
            payload["resource_sha256"] = self.resource_sha256
        return payload


class FHIRNDJSONLineError(ValueError):
    """Raised by :func:`iter_ndjson` when a line cannot be parsed."""

    def __init__(self, path: Path, line_number: int, message: str) -> None:
        self.path = path
        self.line_number = line_number
        self.message = message
        super().__init__(f"{path}: line {line_number}: {message}")

    def to_summary_error(
        self, *, resource_sha256: str | None = None
    ) -> NDJSONLineError:
        """Return a PHI-free summary error for this parse failure."""

        return NDJSONLineError(
            line_number=self.line_number,
            message=self.message,
            resource_sha256=resource_sha256,
        )


@dataclass(frozen=True)
class NDJSONFileSummary:
    """Processing summary for one NDJSON input file."""

    source: str
    destination: str
    lines_processed: int = 0
    resources_deidentified: int = 0
    blank_lines: int = 0
    errors: tuple[NDJSONLineError, ...] = ()
    output_sha256: str = ""
    rejections: tuple[BulkRejection, ...] = ()
    peak_buffered_resources: int = 0
    resumed: bool = False

    @property
    def error_count(self) -> int:
        """Return malformed, failed, and fail-closed resource counts."""

        return len(self.errors) + len(self.rejections)

    @property
    def rejection_count(self) -> int:
        """Return the number of resources rejected without output."""

        return len(self.rejections)

    @property
    def ok(self) -> bool:
        """Return whether no input was malformed or rejected."""

        return not self.errors and not self.rejections

    def to_dict(self) -> dict[str, Any]:
        """Return a report-safe file summary without local paths."""

        return {
            "source": _safe_label(self.source),
            "destination": _safe_label(self.destination),
            "lines_processed": self.lines_processed,
            "resources_deidentified": self.resources_deidentified,
            "blank_lines": self.blank_lines,
            "error_count": self.error_count,
            "rejection_count": self.rejection_count,
            "errors": [error.to_dict() for error in self.errors],
            "rejections": [rejection.to_dict() for rejection in self.rejections],
            "output_sha256": self.output_sha256,
            "peak_buffered_resources": self.peak_buffered_resources,
            "resumed": self.resumed,
        }


@dataclass(frozen=True)
class BulkExportSummary:
    """Aggregate summary for a directory-level bulk export pass."""

    input_dir: str
    output_dir: str
    files: tuple[NDJSONFileSummary, ...] = field(default_factory=tuple)
    policy: str = _DEFAULT_POLICY
    policy_version: str = "v1"
    gateway_version: str = "openmed-fhir-bulk/1"
    started_at: float = 0.0
    finished_at: float = 0.0
    resumed_files: int = 0

    @property
    def file_count(self) -> int:
        """Return the number of NDJSON files processed."""

        return len(self.files)

    @property
    def lines_processed(self) -> int:
        """Return the total number of physical lines read."""

        return sum(file.lines_processed for file in self.files)

    @property
    def resources_deidentified(self) -> int:
        """Return the total number of resources written."""

        return sum(file.resources_deidentified for file in self.files)

    @property
    def errors(self) -> tuple[NDJSONLineError, ...]:
        """Return all malformed-line errors across files."""

        return tuple(error for file in self.files for error in file.errors)

    @property
    def rejections(self) -> tuple[BulkRejection, ...]:
        """Return all fail-closed resource rejections across files."""

        return tuple(rejection for file in self.files for rejection in file.rejections)

    @property
    def error_count(self) -> int:
        """Return all malformed and rejected records."""

        return sum(file.error_count for file in self.files)

    @property
    def rejection_count(self) -> int:
        """Return all fail-closed resource rejections."""

        return sum(file.rejection_count for file in self.files)

    @property
    def peak_buffered_resources(self) -> int:
        """Return the highest per-file resource buffer observed."""

        return max((file.peak_buffered_resources for file in self.files), default=0)

    @property
    def output_sha256(self) -> str:
        """Return a deterministic digest over ordered output file digests."""

        digest = hashlib.sha256()
        for file in self.files:
            digest.update(file.output_sha256.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-free job report."""

        resource_types: dict[str, int] = {}
        for file in self.files:
            resource_type = _resource_type_from_label(file.source)
            resource_types[resource_type] = (
                resource_types.get(resource_type, 0) + file.resources_deidentified
            )
        return {
            "bulk_data_version": BULK_DATA_VERSION,
            "gateway_version": self.gateway_version,
            "policy": _safe_metadata(self.policy),
            "policy_version": _safe_metadata(self.policy_version),
            "files_total": self.file_count,
            "files_completed": self.file_count,
            "resumed_files": self.resumed_files,
            "lines_processed": self.lines_processed,
            "resources_deidentified": self.resources_deidentified,
            "resource_types": resource_types,
            "error_count": self.error_count,
            "rejection_count": self.rejection_count,
            "output_sha256": self.output_sha256,
            "peak_buffered_resources": self.peak_buffered_resources,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": max(0.0, self.finished_at - self.started_at),
            "files": [file.to_dict() for file in self.files],
            "rejections": [rejection.to_dict() for rejection in self.rejections],
            "provenance": {
                "gateway": "openmed.interop.fhir.bulk",
                "bulk_data_version": BULK_DATA_VERSION,
                "policy": _safe_metadata(self.policy),
                "policy_version": _safe_metadata(self.policy_version),
            },
        }


@dataclass(frozen=True)
class FHIRBulkJobReport:
    """Stable report envelope used by asynchronous service jobs."""

    job_id: str
    status: str
    summary: BulkExportSummary
    cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-free report with the job identifier and status."""

        payload = self.summary.to_dict()
        payload.update(
            {
                "job_id": self.job_id,
                "status": self.status,
                "cancelled": self.cancelled,
            }
        )
        return payload


@dataclass(frozen=True)
class BulkGatewayConfig:
    """Configuration for a local directory export/import pass."""

    input_dir: str | Path
    output_dir: str | Path
    checkpoint_path: str | Path | None = None
    policy: str = _DEFAULT_POLICY
    method: str = _DEFAULT_METHOD
    max_buffered_resources: int = DEFAULT_MAX_BUFFERED_RESOURCES

    @property
    def input_path(self) -> Path:
        """Return the input directory as a path."""

        return Path(self.input_dir)

    @property
    def output_path(self) -> Path:
        """Return the output directory as a path."""

        return Path(self.output_dir)

    @property
    def checkpoint_file(self) -> Path:
        """Return the durable checkpoint location."""

        if self.checkpoint_path is not None:
            return Path(self.checkpoint_path)
        return self.output_path / ".openmed-fhir-bulk-checkpoint.json"

    def __post_init__(self) -> None:
        if self.max_buffered_resources < 1:
            raise ValueError("max_buffered_resources must be at least 1")
        if not str(self.input_dir).strip():
            raise ValueError("input_dir must not be blank")
        if not str(self.output_dir).strip():
            raise ValueError("output_dir must not be blank")
        if not str(self.policy).strip():
            raise ValueError("policy must not be blank")
        if not str(self.method).strip():
            raise ValueError("method must not be blank")


class BulkDataGateway:
    """Process a local multi-file FHIR export with atomic resume semantics."""

    def __init__(
        self,
        config: BulkGatewayConfig,
        *,
        deidentifier: Deidentifier | None = None,
        supported_resource_types: Iterable[str] | None = None,
    ) -> None:
        self.config = config
        self.deidentifier = deidentifier
        self.supported_resource_types = (
            frozenset(supported_resource_types)
            if supported_resource_types is not None
            else SUPPORTED_R4_RESOURCE_TYPES
        )

    def export(
        self,
        *,
        cancel_event: Event | None = None,
        job_id: str = "local",
    ) -> FHIRBulkJobReport:
        """Run the local export and return a PHI-free asynchronous-style report."""

        started_at = time.time()
        source_root = self.config.input_path
        destination_root = self.config.output_path
        if not source_root.is_dir():
            raise ValueError("input export path is not a directory")
        destination_root.mkdir(parents=True, exist_ok=True)
        checkpoint = _LocalCheckpoint.load(
            self.config.checkpoint_file,
            configuration=_local_checkpoint_configuration(
                self.config,
                self.supported_resource_types,
            ),
        )
        summaries: list[NDJSONFileSummary] = []

        try:
            for source in sorted(source_root.rglob("*.ndjson")):
                if not source.is_file():
                    continue
                _raise_if_cancelled(cancel_event)
                relative = source.relative_to(source_root)
                destination = destination_root / relative
                input_sha256 = _sha256_file(source)
                record = checkpoint.completed.get(_relative_key(relative))
                if _checkpoint_matches(record, input_sha256, destination):
                    summaries.append(
                        _summary_from_record(
                            record["summary"],
                            source=source,
                            destination=destination,
                            resumed=True,
                        )
                    )
                    continue

                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_name(f"{destination.name}.part")
                partial.unlink(missing_ok=True)
                try:
                    summary = deidentify_ndjson(
                        source,
                        partial,
                        policy=self.config.policy,
                        method=self.config.method,
                        deidentifier=self.deidentifier,
                        supported_resource_types=self.supported_resource_types,
                        max_buffered_resources=self.config.max_buffered_resources,
                        cancel_event=cancel_event,
                    )
                    _raise_if_cancelled(cancel_event)
                    os.replace(partial, destination)
                except BaseException:
                    partial.unlink(missing_ok=True)
                    raise

                summaries.append(summary)
                checkpoint.mark_completed(
                    relative,
                    input_sha256=input_sha256,
                    summary=summary,
                )
        except BulkJobCancelled:
            raise

        finished_at = time.time()
        ordered = tuple(summaries)
        resumed_files = sum(1 for summary in ordered if summary.resumed)
        aggregate = BulkExportSummary(
            input_dir=str(source_root),
            output_dir=str(destination_root),
            files=ordered,
            policy=self.config.policy,
            policy_version="v1",
            started_at=started_at,
            finished_at=finished_at,
            resumed_files=resumed_files,
        )
        return FHIRBulkJobReport(job_id=job_id, status="succeeded", summary=aggregate)

    run = export


def iter_ndjson(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield parsed FHIR resources lazily from an NDJSON file."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            yield _parse_resource_line(source, line, line_number)


def deidentify_ndjson(
    in_path: str | Path,
    out_path: str | Path,
    *,
    policy: str = _DEFAULT_POLICY,
    method: str = _DEFAULT_METHOD,
    deidentifier: Deidentifier | None = None,
    supported_resource_types: Iterable[str] | None = None,
    max_buffered_resources: int = DEFAULT_MAX_BUFFERED_RESOURCES,
    cancel_event: Event | None = None,
) -> NDJSONFileSummary:
    """Stream one NDJSON file through the fail-closed privacy path."""

    source = Path(in_path)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part")
    temporary.unlink(missing_ok=True)
    try:
        with (
            source.open("r", encoding="utf-8") as input_stream,
            temporary.open("w", encoding="utf-8", newline="") as output_stream,
        ):
            summary = deidentify_ndjson_stream(
                input_stream,
                output_stream,
                source=source,
                destination=destination,
                policy=policy,
                method=method,
                deidentifier=deidentifier,
                supported_resource_types=supported_resource_types,
                max_buffered_resources=max_buffered_resources,
                cancel_event=cancel_event,
            )
        os.replace(temporary, destination)
        return summary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def deidentify_ndjson_stream(
    lines: Iterable[str],
    output_stream: TextIO,
    *,
    source: str | Path = "<stream>",
    destination: str | Path = "<stream>",
    policy: str = _DEFAULT_POLICY,
    method: str = _DEFAULT_METHOD,
    deidentifier: Deidentifier | None = None,
    supported_resource_types: Iterable[str] | None = None,
    max_buffered_resources: int = DEFAULT_MAX_BUFFERED_RESOURCES,
    cancel_event: Event | None = None,
) -> NDJSONFileSummary:
    """Stream NDJSON text through one-resource-at-a-time de-identification."""

    if max_buffered_resources < 1:
        raise ValueError("max_buffered_resources must be at least 1")
    processor = _NDJSONStreamProcessor(
        source=source,
        destination=destination,
        output_stream=output_stream,
        policy=policy,
        method=method,
        deidentifier=deidentifier,
        supported_resource_types=(
            frozenset(supported_resource_types)
            if supported_resource_types is not None
            else SUPPORTED_R4_RESOURCE_TYPES
        ),
        max_buffered_resources=max_buffered_resources,
        cancel_event=cancel_event,
    )
    for line_number, line in enumerate(lines, start=1):
        processor.process_line(line, line_number)
    return processor.summary()


async def deidentify_ndjson_async(
    lines: AsyncIterable[str],
    out_path: str | Path,
    *,
    source: str | Path = "<stream>",
    destination: str | Path | None = None,
    policy: str = _DEFAULT_POLICY,
    method: str = _DEFAULT_METHOD,
    deidentifier: Deidentifier | None = None,
    supported_resource_types: Iterable[str] | None = None,
    max_buffered_resources: int = DEFAULT_MAX_BUFFERED_RESOURCES,
    cancel_event: Event | None = None,
) -> NDJSONFileSummary:
    """Stream async NDJSON lines to a temporary output file."""

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_destination = output_path if destination is None else destination
    temporary = output_path.with_name(f"{output_path.name}.part")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", encoding="utf-8") as output_stream:
            processor = _NDJSONStreamProcessor(
                source=source,
                destination=summary_destination,
                output_stream=output_stream,
                policy=policy,
                method=method,
                deidentifier=deidentifier,
                supported_resource_types=(
                    frozenset(supported_resource_types)
                    if supported_resource_types is not None
                    else SUPPORTED_R4_RESOURCE_TYPES
                ),
                max_buffered_resources=max_buffered_resources,
                cancel_event=cancel_event,
            )
            line_number = 0
            async for line in lines:
                line_number += 1
                processor.process_line(line, line_number)
            summary = processor.summary()
        os.replace(temporary, output_path)
        return summary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass
class _NDJSONStreamProcessor:
    source: str | Path
    destination: str | Path
    output_stream: TextIO
    policy: str
    method: str
    deidentifier: Deidentifier | None
    supported_resource_types: frozenset[str]
    max_buffered_resources: int
    cancel_event: Event | None
    lines_processed: int = 0
    resources_deidentified: int = 0
    blank_lines: int = 0
    errors: list[NDJSONLineError] = field(default_factory=list)
    rejections: list[BulkRejection] = field(default_factory=list)
    peak_buffered_resources: int = 0
    _digest: Any = field(default_factory=hashlib.sha256)

    def process_line(self, line: str, line_number: int) -> None:
        """Process one physical line without ever writing an unsafe resource."""

        _raise_if_cancelled(self.cancel_event)
        self.lines_processed += 1
        raw_hash = _sha256_bytes(line.encode("utf-8"))
        if not line.strip():
            self.blank_lines += 1
            return

        try:
            resource = _parse_resource_line(Path(str(self.source)), line, line_number)
        except FHIRNDJSONLineError as exc:
            self.errors.append(exc.to_summary_error(resource_sha256=raw_hash))
            return

        try:
            resource_type, _ = _validate_resource_for_bulk(
                resource,
                supported_resource_types=self.supported_resource_types,
            )
        except _BulkResourceRejected as exc:
            self.rejections.append(
                BulkRejection(
                    line_number=line_number,
                    reason=exc.reason,
                    resource_sha256=_canonical_resource_hash(resource),
                    resource_type=exc.resource_type,
                    path=exc.path,
                )
            )
            return
        except ValueError as exc:
            message = str(exc)
            if message not in {"resource is missing 'resourceType'"}:
                message = "resource validation failed"
            self.errors.append(
                NDJSONLineError(
                    line_number=line_number,
                    message=message,
                    resource_sha256=_canonical_resource_hash(resource),
                )
            )
            return

        self.peak_buffered_resources = max(self.peak_buffered_resources, 1)
        if self.peak_buffered_resources > self.max_buffered_resources:
            raise RuntimeError("configured resource buffer bound was exceeded")
        try:
            transformed = de_identify_resource(
                resource,
                policy=self.policy,
                method=self.method,
                deidentifier=self.deidentifier,
            )
            if not isinstance(transformed, dict):
                raise TypeError("privacy pipeline returned an invalid resource")
            if transformed.get("resourceType") != resource_type:
                raise TypeError("privacy pipeline changed resource type")
            encoded = json.dumps(
                transformed,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError):
            # Do not reuse exception text: a user-supplied de-identifier can
            # accidentally include a raw value in its exception message.
            self.errors.append(
                NDJSONLineError(
                    line_number=line_number,
                    message="resource transformation failed",
                    resource_sha256=_canonical_resource_hash(resource),
                )
            )
            return
        finally:
            # The implementation processes exactly one resource at a time.
            pass

        self.output_stream.write(encoded.decode("utf-8"))
        self.output_stream.write("\n")
        self._digest.update(encoded)
        self._digest.update(b"\n")
        self.resources_deidentified += 1

    def summary(self) -> NDJSONFileSummary:
        """Return the PHI-free summary for all processed lines."""

        return NDJSONFileSummary(
            source=str(self.source),
            destination=str(self.destination),
            lines_processed=self.lines_processed,
            resources_deidentified=self.resources_deidentified,
            blank_lines=self.blank_lines,
            errors=tuple(self.errors),
            output_sha256=self._digest.hexdigest(),
            rejections=tuple(self.rejections),
            peak_buffered_resources=self.peak_buffered_resources,
        )


def deidentify_export(
    in_dir: str | Path,
    out_dir: str | Path,
    *,
    policy: str = _DEFAULT_POLICY,
    method: str = _DEFAULT_METHOD,
    deidentifier: Deidentifier | None = None,
    supported_resource_types: Iterable[str] | None = None,
    max_buffered_resources: int = DEFAULT_MAX_BUFFERED_RESOURCES,
    checkpoint_path: str | Path | None = None,
    cancel_event: Event | None = None,
) -> BulkExportSummary:
    """De-identify every NDJSON file using atomic output commits and resume."""

    report = BulkDataGateway(
        BulkGatewayConfig(
            input_dir=in_dir,
            output_dir=out_dir,
            checkpoint_path=checkpoint_path,
            policy=policy,
            method=method,
            max_buffered_resources=max_buffered_resources,
        ),
        deidentifier=deidentifier,
        supported_resource_types=supported_resource_types,
    ).export(cancel_event=cancel_event)
    return report.summary


# Descriptive aliases keep the public API discoverable for callers that use
# the issue's "FHIR Bulk" terminology rather than the generic data gateway
# name.
FHIRBulkGateway = BulkDataGateway
FHIRBulkConfig = BulkGatewayConfig


@dataclass
class _LocalCheckpoint:
    path: Path
    completed: dict[str, dict[str, Any]]
    configuration: str = ""

    @classmethod
    def load(cls, path: Path, *, configuration: str = "") -> "_LocalCheckpoint":
        if not path.exists():
            return cls(path=path, completed={}, configuration=configuration)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("bulk checkpoint is not valid JSON") from exc
        completed = payload.get("completed", {})
        if payload.get("version") != _CHECKPOINT_VERSION or not isinstance(
            completed, dict
        ):
            raise ValueError("bulk checkpoint has invalid version or state")
        stored_configuration = payload.get("configuration", "")
        if stored_configuration != configuration:
            completed = {}
        return cls(
            path=path,
            completed=dict(completed),
            configuration=configuration,
        )

    def mark_completed(
        self,
        relative: Path,
        *,
        input_sha256: str,
        summary: NDJSONFileSummary,
    ) -> None:
        self.completed[_relative_key(relative)] = {
            "input_sha256": input_sha256,
            "summary": summary.to_dict(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": _CHECKPOINT_VERSION,
                    "configuration": self.configuration,
                    "completed": self.completed,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


def _summary_from_record(
    record: Mapping[str, Any],
    *,
    source: Path,
    destination: Path,
    resumed: bool,
) -> NDJSONFileSummary:
    """Reconstruct a safe summary from a validated checkpoint record."""

    errors = tuple(
        NDJSONLineError(
            line_number=int(item.get("line_number", 0)),
            message=str(item.get("message", "checkpoint error")),
            resource_sha256=item.get("resource_sha256"),
        )
        for item in record.get("errors", [])
        if isinstance(item, Mapping)
    )
    rejections = tuple(
        BulkRejection(
            line_number=int(item.get("line_number", 0)),
            reason=str(item.get("reason", "rejected")),
            resource_sha256=str(item.get("resource_sha256", "")),
            resource_type=item.get("resource_type"),
            path=item.get("path"),
        )
        for item in record.get("rejections", [])
        if isinstance(item, Mapping)
    )
    return NDJSONFileSummary(
        source=str(source),
        destination=str(destination),
        lines_processed=int(record.get("lines_processed", 0)),
        resources_deidentified=int(record.get("resources_deidentified", 0)),
        blank_lines=int(record.get("blank_lines", 0)),
        errors=errors,
        output_sha256=str(record.get("output_sha256", "")),
        rejections=rejections,
        peak_buffered_resources=int(record.get("peak_buffered_resources", 0)),
        resumed=resumed,
    )


def _checkpoint_matches(
    record: Mapping[str, Any] | None,
    input_sha256: str,
    destination: Path,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    summary = record.get("summary")
    if not isinstance(summary, Mapping) or record.get("input_sha256") != input_sha256:
        return False
    expected = summary.get("output_sha256")
    return (
        isinstance(expected, str)
        and bool(expected)
        and destination.is_file()
        and _sha256_file(destination) == expected
    )


def _local_checkpoint_configuration(
    config: BulkGatewayConfig,
    supported_resource_types: frozenset[str],
) -> str:
    payload = {
        "gateway": "openmed-fhir-bulk/1",
        "policy": config.policy,
        "method": config.method,
        "max_buffered_resources": config.max_buffered_resources,
        "supported_resource_types": sorted(supported_resource_types),
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _parse_resource_line(path: Path, line: str, line_number: int) -> dict[str, Any]:
    try:
        raw = json.loads(line.lstrip("\ufeff") if line_number == 1 else line)
    except json.JSONDecodeError as exc:
        raise FHIRNDJSONLineError(
            path, line_number, f"malformed JSON: {exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise FHIRNDJSONLineError(
            path,
            line_number,
            "line must contain a JSON object",
        )
    return raw


def _validate_resource_for_bulk(
    resource: Mapping[str, Any],
    *,
    supported_resource_types: frozenset[str],
) -> tuple[str, str | None]:
    resource_type = resource.get("resourceType")
    if not isinstance(resource_type, str) or not resource_type:
        raise ValueError("resource is missing 'resourceType'")
    if resource_type == "Binary":
        raise _BulkResourceRejected(
            "unsupported_binary",
            resource_type=resource_type,
        )
    if resource_type not in supported_resource_types:
        raise _BulkResourceRejected(
            "unsupported_resource_type",
            resource_type=resource_type,
        )
    _validate_nested_paths(
        resource,
        path=resource_type,
        supported_resource_types=supported_resource_types,
        seen=set(),
    )
    return resource_type, None


def _validate_nested_paths(
    node: Any,
    *,
    path: str,
    supported_resource_types: frozenset[str],
    seen: set[int],
) -> None:
    if isinstance(node, (dict, list)):
        identity = id(node)
        if identity in seen:
            raise _BulkResourceRejected("cyclic_resource_path", path=path)
        seen.add(identity)

    if isinstance(node, dict):
        nested_type = node.get("resourceType")
        if nested_type == "Binary":
            raise _BulkResourceRejected(
                "unsupported_binary",
                resource_type="Binary",
                path=path,
            )
        if (
            nested_type is not None
            and isinstance(nested_type, str)
            and nested_type not in supported_resource_types
        ):
            raise _BulkResourceRejected(
                "unsupported_resource_type",
                resource_type=nested_type,
                path=path,
            )
        if nested_type is not None and not isinstance(nested_type, str):
            raise _BulkResourceRejected("unsafe_resource_path", path=path)
        for key, value in node.items():
            child_path = f"{path}.{key}"
            if key == "text" and isinstance(value, dict):
                if "div" not in value:
                    raise _BulkResourceRejected("unsafe_narrative", path=child_path)
                _validate_narrative(value["div"], path=f"{child_path}.div")
            elif (
                key == "resource" and value is not None and not isinstance(value, dict)
            ):
                raise _BulkResourceRejected("unsafe_resource_path", path=child_path)
            elif (
                key == "resource"
                and isinstance(value, dict)
                and not value.get("resourceType")
            ):
                raise _BulkResourceRejected("unsafe_resource_path", path=child_path)
            elif key == "contained" and isinstance(value, list):
                for index, item in enumerate(value):
                    if not isinstance(item, dict) or not item.get("resourceType"):
                        raise _BulkResourceRejected(
                            "unsafe_resource_path",
                            path=f"{child_path}[{index}]",
                        )
            _validate_nested_paths(
                value,
                path=child_path,
                supported_resource_types=supported_resource_types,
                seen=seen,
            )
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _validate_nested_paths(
                value,
                path=f"{path}[{index}]",
                supported_resource_types=supported_resource_types,
                seen=seen,
            )

    if isinstance(node, (dict, list)):
        seen.discard(id(node))


def _validate_narrative(div: Any, *, path: str) -> None:
    if not isinstance(div, str) or not div.strip():
        raise _BulkResourceRejected("unsafe_narrative", path=path)
    upper = div.upper()
    if (
        "<!DOCTYPE" in upper
        or "<!ENTITY" in upper
        or "<!--" in div
        or "<?" in div
        or "\x00" in div
    ):
        raise _BulkResourceRejected("unsafe_narrative", path=path)
    if re.search(r"&(?:#\d+|#x[0-9a-f]+);", div, flags=re.IGNORECASE):
        raise _BulkResourceRejected("unsafe_narrative", path=path)
    try:
        root = ET.fromstring(div)
    except (ET.ParseError, ValueError):
        raise _BulkResourceRejected("unsafe_narrative", path=path) from None
    if _local_name(root.tag) != "div":
        raise _BulkResourceRejected("unsafe_narrative", path=path)
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag in _UNSAFE_NARRATIVE_TAGS or tag not in _SAFE_NARRATIVE_TAGS:
            raise _BulkResourceRejected("unsafe_narrative", path=path)
        for attribute, value in element.attrib.items():
            name = _local_name(attribute).lower()
            if name.startswith("on") or name in {"src", "style"}:
                raise _BulkResourceRejected("unsafe_narrative", path=path)
            if name == "href" and not str(value).startswith("#"):
                raise _BulkResourceRejected("unsafe_narrative", path=path)


@dataclass(frozen=True)
class _BulkResourceRejected(ValueError):
    reason: str
    resource_type: str | None = None
    path: str | None = None


def _raise_if_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise BulkJobCancelled("bulk job cancelled")


def _canonical_resource_hash(resource: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        resource,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_key(path: Path) -> str:
    return path.as_posix()


def _local_name(value: Any) -> str:
    return str(value).rsplit("}", 1)[-1]


def _resource_type_from_label(value: str | Path) -> str:
    stem = Path(str(value)).stem
    return _safe_resource_type(stem)


def _safe_resource_type(value: Any) -> str:
    if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Za-z0-9]{0,63}", value):
        return value
    return "unknown"


def _safe_structural_path(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,256}", value):
        return value
    return f"path-{_sha256_bytes(value.encode('utf-8'))[:16]}"


def _safe_metadata(value: Any) -> str:
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        return value
    return "redacted"


def _safe_label(value: str | Path) -> str:
    # Reports carry stable labels, never caller-controlled absolute paths.
    name = Path(str(value)).name
    if not name:
        return "stream"
    if _resource_type_from_label(name) != "unknown":
        return name
    return f"file-{_sha256_bytes(name.encode('utf-8'))[:16]}"
