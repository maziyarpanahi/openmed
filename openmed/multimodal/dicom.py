"""DICOM PS3.15 header de-identification.

The module imports pydicom lazily so the multimodal package remains importable
without optional imaging dependencies. Header provenance records tags and
actions only; raw PHI and original UIDs are intentionally omitted.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from .base import ExtractedDocument, register_handler
from .exceptions import MissingDependencyError

_DICOM_INSTALL_HINT = 'Install with: pip install "openmed[multimodal]".'
_DEFAULT_UID_SALT = "openmed-dicom-uid-v1"
_PROFILE_NAME = "DICOM PS3.15 Basic Application Level Confidentiality Profile"
_DEID_METHOD = "PS3.15 Basic Profile; dates modified; UIDs remapped"


@dataclass(frozen=True)
class DicomHeaderDeidPolicy:
    """Policy knobs for DICOM header de-identification."""

    output_path: str | Path | None = None
    date_shift_days: int | None = None
    patient_key: str | bytes | None = None
    date_shift_max_days: int | None = None
    date_shift_secret: str | bytes | None = None
    uid_salt: str | bytes = _DEFAULT_UID_SALT
    keep_year: bool = False


@dataclass(frozen=True)
class DicomHeaderAction:
    """Audit-safe description of one DICOM header action."""

    tag: str
    keyword: str
    vr: str
    action: str
    ps315_action: str
    location: str = "Dataset"
    value_sha256: str | None = None
    value_length: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable audit-safe action summary."""

        payload: dict[str, Any] = {
            "tag": self.tag,
            "keyword": self.keyword,
            "vr": self.vr,
            "action": self.action,
            "ps315_action": self.ps315_action,
            "location": self.location,
        }
        if self.value_sha256 is not None:
            payload["value_sha256"] = self.value_sha256
        if self.value_length is not None:
            payload["value_length"] = self.value_length
        return payload


@dataclass(frozen=True)
class DicomHeaderDeidResult:
    """Result returned by :func:`deidentify_dicom_headers`."""

    source_path: Path
    output_path: Path
    date_shift_days: int
    actions: tuple[DicomHeaderAction, ...]
    uid_remap_count: int
    private_tag_removed_count: int

    @property
    def action_count(self) -> int:
        """Number of header actions performed."""

        return len(self.actions)

    def to_audit_report(self) -> dict[str, Any]:
        """Return an audit-safe provenance summary."""

        action_counts = Counter(action.action for action in self.actions)
        return {
            "type": "dicom_header_deidentification",
            "profile": _PROFILE_NAME,
            "source_path_sha256": _hash_value(str(self.source_path)),
            "output_path_sha256": _hash_value(str(self.output_path)),
            "source_suffix": self.source_path.suffix.lower(),
            "output_suffix": self.output_path.suffix.lower(),
            "date_shift_days": self.date_shift_days,
            "longitudinal_temporal_information_modified": "MODIFIED",
            "action_count": self.action_count,
            "action_counts": dict(sorted(action_counts.items())),
            "uid_remap_count": self.uid_remap_count,
            "private_tag_removed_count": self.private_tag_removed_count,
            "actions": [action.to_dict() for action in self.actions],
        }


@dataclass(frozen=True)
class DicomPixelRedactionPolicy:
    """Policy knobs for DICOM burned-in pixel-text redaction."""

    output_path: str | Path | None = None
    ocr_engine: Any = None
    model_name: str | None = None
    confidence_threshold: float = 0.5
    bbox_padding: int = 1
    verify_residual: bool = True
    fail_on_residual: bool = True
    custom_recognizer: Any = None


@dataclass(frozen=True)
class DicomPixelFinding:
    """Audit-safe description of one OCR-projected pixel redaction."""

    frame_index: int
    bbox: tuple[int, int, int, int]
    label: str
    confidence: float
    text_sha256: str
    text_length: int
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return an audit-safe finding summary without raw OCR text."""

        payload: dict[str, Any] = {
            "frame_index": self.frame_index,
            "bbox": list(self.bbox),
            "label": self.label,
            "confidence": self.confidence,
            "text_sha256": self.text_sha256,
            "text_length": self.text_length,
        }
        if self.sources:
            payload["sources"] = list(self.sources)
        return payload


@dataclass(frozen=True)
class DicomResidualTextReport:
    """Residual OCR PHI verification report for redacted DICOM pixels."""

    frame_count: int
    residuals: tuple[DicomPixelFinding, ...] = ()

    @property
    def residual_entity_count(self) -> int:
        """Number of residual OCR-projected PHI findings."""

        return len(self.residuals)

    @property
    def passed(self) -> bool:
        """Whether residual OCR found zero PHI."""

        return self.residual_entity_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable residual report."""

        return {
            "frame_count": self.frame_count,
            "passed": self.passed,
            "residual_entity_count": self.residual_entity_count,
            "residuals": [finding.to_dict() for finding in self.residuals],
        }


@dataclass(frozen=True)
class DicomPixelRedactionResult:
    """Result returned by :func:`redact_dicom_pixels`."""

    source_path: Path
    output_path: Path
    frames_processed: int
    findings: tuple[DicomPixelFinding, ...]
    residual_report: DicomResidualTextReport

    @property
    def redaction_count(self) -> int:
        """Number of projected pixel boxes blacked out."""

        return len(self.findings)

    def to_audit_report(self) -> dict[str, Any]:
        """Return an audit-safe provenance summary."""

        label_counts = Counter(finding.label for finding in self.findings)
        frame_counts = Counter(finding.frame_index for finding in self.findings)
        return {
            "type": "dicom_pixel_ocr_redaction",
            "source_path_sha256": _hash_value(str(self.source_path)),
            "output_path_sha256": _hash_value(str(self.output_path)),
            "source_suffix": self.source_path.suffix.lower(),
            "output_suffix": self.output_path.suffix.lower(),
            "frames_processed": self.frames_processed,
            "redaction_count": self.redaction_count,
            "redaction_counts_by_label": dict(sorted(label_counts.items())),
            "redaction_counts_by_frame": {
                str(frame): count for frame, count in sorted(frame_counts.items())
            },
            "findings": [finding.to_dict() for finding in self.findings],
            "residual_report": self.residual_report.to_dict(),
        }


@dataclass
class _Context:
    date_shift_days: int
    keep_year: bool
    uid_salt: bytes
    actions: list[DicomHeaderAction] = field(default_factory=list)
    uid_map: dict[str, str] = field(default_factory=dict)
    private_tag_removed_count: int = 0


# PS3.15 action X: remove the Attribute.
_REMOVE_TAGS = {
    0x00080081,  # Institution Address
    0x00081040,  # Institutional Department Name
    0x00081050,  # Performing Physician's Name
    0x00081060,  # Name of Physician(s) Reading Study
    0x00081070,  # Operators' Name
    0x00081080,  # Admitting Diagnoses Description
    0x00081140,  # Referenced Image Sequence
    0x00100021,  # Issuer of Patient ID
    0x00100050,  # Patient's Insurance Plan Code Sequence
    0x00101000,  # Other Patient IDs
    0x00101001,  # Other Patient Names
    0x00101002,  # Other Patient IDs Sequence
    0x00101005,  # Patient's Birth Name
    0x00101060,  # Patient's Mother's Birth Name
    0x00101080,  # Military Rank
    0x00101081,  # Branch of Service
    0x00101090,  # Medical Record Locator
    0x00102110,  # Allergies
    0x00102150,  # Country of Residence
    0x00102152,  # Region of Residence
    0x00102154,  # Patient's Telephone Numbers
    0x00102155,  # Patient's Telecom Information
    0x00102160,  # Ethnic Group
    0x00102180,  # Occupation
    0x001021B0,  # Additional Patient History
    0x001021D0,  # Last Menstrual Date
    0x001021F0,  # Patient's Religious Preference
    0x00104000,  # Patient Comments
    0x00380010,  # Admission ID
    0x00380300,  # Current Patient Location
    0x00380400,  # Patient's Institution Residence
    0x00400006,  # Scheduled Performing Physician's Name
    0x0040000B,  # Scheduled Performing Physician Identification Sequence
    0x00400009,  # Scheduled Procedure Step ID
    0x00401001,  # Requested Procedure ID
    0x00401010,  # Names of Intended Recipients of Results
    0x00401101,  # Person Identification Code Sequence
    0x00401102,  # Person's Address
    0x00401103,  # Person's Telephone Numbers
    0x00401104,  # Person's Telecom Information
    0x00402016,  # Placer Order Number / Imaging Service Request
    0x00402017,  # Filler Order Number / Imaging Service Request
    0x00402411,  # Performed Station AE Title
    0x00402412,  # Performed Station Name
    0x00402413,  # Performed Location
    0x00400253,  # Performed Procedure Step ID
}

# PS3.15 action Z or Z/D: clear the Attribute value while retaining the tag.
_ZERO_TAGS = {
    0x00080050,  # Accession Number
    0x00080080,  # Institution Name
    0x00080090,  # Referring Physician's Name
    0x00080092,  # Referring Physician's Address
    0x00080094,  # Referring Physician's Telephone Numbers
    0x00100010,  # Patient's Name
    0x00100020,  # Patient ID
    0x00100030,  # Patient's Birth Date
    0x00100032,  # Patient's Birth Time
    0x00100040,  # Patient's Sex
    0x00101010,  # Patient's Age
}

_UID_KEEP_KEYWORDS = {
    "AffectedSOPClassUID",
    "ImplementationClassUID",
    "MediaStorageSOPClassUID",
    "RequestedSOPClassUID",
    "SOPClassUID",
    "TransferSyntaxUID",
}

_UID_REMAP_KEYWORDS = {
    "AcquisitionUID",
    "ConcatenationUID",
    "DimensionOrganizationUID",
    "FailedSOPInstanceUIDList",
    "FiducialUID",
    "FrameOfReferenceUID",
    "IrradiationEventUID",
    "MediaStorageSOPInstanceUID",
    "ObservationUID",
    "PaletteColorLookupTableUID",
    "ReferencedFrameOfReferenceUID",
    "ReferencedSOPInstanceUID",
    "RequestedSOPInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SpecimenUID",
    "StorageMediaFileSetUID",
    "StudyInstanceUID",
    "SynchronizationFrameOfReferenceUID",
    "TargetUID",
    "TrackingUID",
    "TransactionUID",
}

_DT_DATE_RE = re.compile(r"^(?P<date>\d{8})(?P<rest>.*)$")
_SEEDED_PHI_SCRUB_VRS = frozenset(
    {"AE", "AS", "CS", "LO", "LT", "SH", "ST", "UC", "UR", "UT"}
)
_SEEDED_PHI_SOURCE_VRS = _SEEDED_PHI_SCRUB_VRS | {"DA", "DT", "PN", "TM", "UI"}


def deidentify_dicom_headers(
    path: str | Path,
    *,
    policy: Any | None = None,
) -> DicomHeaderDeidResult:
    """De-identify DICOM headers and write a de-identified DICOM file.

    ``policy`` may be a :class:`DicomHeaderDeidPolicy`, a mapping, or any object
    exposing equivalent attributes. When no ``output_path`` is supplied, the
    source file is rewritten in place.
    """

    pydicom = _import_pydicom()
    source = Path(path)
    resolved_policy = _coerce_policy(policy)
    output_path = (
        Path(resolved_policy.output_path)
        if resolved_policy.output_path is not None
        else source
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shift_days = _resolve_shift_days(resolved_policy)
    context = _Context(
        date_shift_days=shift_days,
        keep_year=resolved_policy.keep_year,
        uid_salt=_bytes_value(resolved_policy.uid_salt, name="uid_salt"),
    )

    dataset = pydicom.dcmread(source, force=True)
    seeded_phi_terms = _seeded_phi_scrub_terms(dataset)
    _deidentify_dataset(dataset, context, location="Dataset")
    _clear_seeded_phi_copies(
        dataset,
        seeded_phi_terms,
        context,
        location="Dataset",
    )
    _set_standard_deid_markers(dataset, context)
    _deidentify_file_meta(dataset, context)
    if hasattr(dataset, "preamble"):
        dataset.preamble = b"\0" * 128

    _save_dataset(dataset, output_path)
    return DicomHeaderDeidResult(
        source_path=source,
        output_path=output_path,
        date_shift_days=shift_days,
        actions=tuple(context.actions),
        uid_remap_count=len(context.uid_map),
        private_tag_removed_count=context.private_tag_removed_count,
    )


def redact_dicom_pixels(
    path: str | Path,
    *,
    policy: Any | None = None,
    output_path: str | Path | None = None,
    ocr_engine: Any = None,
    models: Any = None,
    model_name: str | None = None,
    confidence_threshold: float | None = None,
    bbox_padding: int | None = None,
    verify_residual: bool | None = None,
    fail_on_residual: bool | None = None,
    custom_recognizer: Any = None,
    lang: str | None = None,
) -> DicomPixelRedactionResult:
    """Redact burned-in PHI text from DICOM pixels using OCR bboxes.

    Header values from the source DICOM seed a per-image custom recognizer
    before header de-identification clears them. Returned reports intentionally
    exclude raw OCR/header text and carry hashes, labels, bboxes, and counts.
    """

    pydicom = _import_pydicom()
    source = Path(path)
    resolved_policy = _override_pixel_policy(
        _coerce_pixel_policy(policy),
        output_path=output_path,
        ocr_engine=ocr_engine,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        bbox_padding=bbox_padding,
        verify_residual=verify_residual,
        fail_on_residual=fail_on_residual,
        custom_recognizer=custom_recognizer,
    )
    destination = (
        Path(resolved_policy.output_path)
        if resolved_policy.output_path is not None
        else source
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    dataset = pydicom.dcmread(source, force=True)
    header_recognizer = _header_seed_recognizer(dataset)
    model = _resolve_pixel_model_name(models, resolved_policy.model_name)

    if "PixelData" not in dataset:
        _save_dataset(dataset, destination)
        residual_report = DicomResidualTextReport(frame_count=0)
        return DicomPixelRedactionResult(
            source_path=source,
            output_path=destination,
            frames_processed=0,
            findings=(),
            residual_report=residual_report,
        )

    _decompress_pixel_data(dataset)
    pixel_array = _copy_pixel_array(dataset)
    frame_views = tuple(_iter_pixel_frames(pixel_array, dataset))
    findings: list[DicomPixelFinding] = []

    for frame_index, frame in enumerate(frame_views):
        frame_findings = _detect_frame_pixel_findings(
            frame,
            dataset=dataset,
            frame_index=frame_index,
            policy=resolved_policy,
            model_name=model,
            header_recognizer=header_recognizer,
            lang=lang,
        )
        for finding in frame_findings:
            _blackout_bbox(frame, finding.bbox, dataset)
        findings.extend(frame_findings)

    dataset.PixelData = _pixel_bytes(pixel_array)

    residual_report = DicomResidualTextReport(frame_count=len(frame_views))
    if resolved_policy.verify_residual:
        residual_report = _residual_text_report(
            frame_views,
            dataset=dataset,
            policy=resolved_policy,
            model_name=model,
            header_recognizer=header_recognizer,
            lang=lang,
        )
        if resolved_policy.fail_on_residual and not residual_report.passed:
            raise ValueError(
                "DICOM residual OCR PHI verification failed: "
                f"{residual_report.residual_entity_count} residual findings"
            )

    _save_dataset(dataset, destination)
    return DicomPixelRedactionResult(
        source_path=source,
        output_path=destination,
        frames_processed=len(frame_views),
        findings=tuple(findings),
        residual_report=residual_report,
    )


def _import_pydicom() -> Any:
    try:
        return importlib.import_module("pydicom")
    except ImportError as exc:  # pragma: no cover - exercised without extra.
        raise MissingDependencyError(
            dependency="pydicom", instruction=_DICOM_INSTALL_HINT
        ) from exc


def _import_numpy() -> Any:
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:  # pragma: no cover - exercised without extra.
        raise MissingDependencyError(
            dependency="numpy", instruction=_DICOM_INSTALL_HINT
        ) from exc


def _coerce_pixel_policy(policy: Any | None) -> DicomPixelRedactionPolicy:
    if policy is None:
        return DicomPixelRedactionPolicy()
    if isinstance(policy, DicomPixelRedactionPolicy):
        return policy
    if isinstance(policy, Mapping):
        return DicomPixelRedactionPolicy(
            output_path=policy.get("pixel_output_path", policy.get("output_path")),
            ocr_engine=policy.get("ocr_engine"),
            model_name=(
                policy.get("pii_model_name")
                or policy.get("pii_model")
                or policy.get("model_name")
            ),
            confidence_threshold=_optional_float(
                policy.get("confidence_threshold"), default=0.5
            ),
            bbox_padding=_optional_int(policy.get("bbox_padding")) or 1,
            verify_residual=bool(policy.get("verify_residual", True)),
            fail_on_residual=bool(policy.get("fail_on_residual", True)),
            custom_recognizer=policy.get("custom_recognizer"),
        )
    return DicomPixelRedactionPolicy(
        output_path=getattr(
            policy,
            "pixel_output_path",
            getattr(policy, "output_path", None),
        ),
        ocr_engine=getattr(policy, "ocr_engine", None),
        model_name=(
            getattr(policy, "pii_model_name", None)
            or getattr(policy, "pii_model", None)
            or getattr(policy, "model_name", None)
        ),
        confidence_threshold=_optional_float(
            getattr(policy, "confidence_threshold", None), default=0.5
        ),
        bbox_padding=_optional_int(getattr(policy, "bbox_padding", None)) or 1,
        verify_residual=bool(getattr(policy, "verify_residual", True)),
        fail_on_residual=bool(getattr(policy, "fail_on_residual", True)),
        custom_recognizer=getattr(policy, "custom_recognizer", None),
    )


def _override_pixel_policy(
    policy: DicomPixelRedactionPolicy,
    *,
    output_path: str | Path | None,
    ocr_engine: Any,
    model_name: str | None,
    confidence_threshold: float | None,
    bbox_padding: int | None,
    verify_residual: bool | None,
    fail_on_residual: bool | None,
    custom_recognizer: Any,
) -> DicomPixelRedactionPolicy:
    return DicomPixelRedactionPolicy(
        output_path=output_path if output_path is not None else policy.output_path,
        ocr_engine=ocr_engine if ocr_engine is not None else policy.ocr_engine,
        model_name=model_name if model_name is not None else policy.model_name,
        confidence_threshold=(
            float(confidence_threshold)
            if confidence_threshold is not None
            else policy.confidence_threshold
        ),
        bbox_padding=(
            int(bbox_padding) if bbox_padding is not None else policy.bbox_padding
        ),
        verify_residual=(
            bool(verify_residual)
            if verify_residual is not None
            else policy.verify_residual
        ),
        fail_on_residual=(
            bool(fail_on_residual)
            if fail_on_residual is not None
            else policy.fail_on_residual
        ),
        custom_recognizer=(
            custom_recognizer
            if custom_recognizer is not None
            else policy.custom_recognizer
        ),
    )


def _coerce_policy(policy: Any | None) -> DicomHeaderDeidPolicy:
    if policy is None:
        return DicomHeaderDeidPolicy()
    if isinstance(policy, DicomHeaderDeidPolicy):
        return policy
    if isinstance(policy, Mapping):
        return DicomHeaderDeidPolicy(
            output_path=policy.get("output_path"),
            date_shift_days=_optional_int(policy.get("date_shift_days")),
            patient_key=policy.get("patient_key"),
            date_shift_max_days=_optional_int(policy.get("date_shift_max_days")),
            date_shift_secret=policy.get("date_shift_secret"),
            uid_salt=policy.get("uid_salt", _DEFAULT_UID_SALT),
            keep_year=bool(policy.get("keep_year", False)),
        )
    return DicomHeaderDeidPolicy(
        output_path=getattr(policy, "output_path", None),
        date_shift_days=_optional_int(getattr(policy, "date_shift_days", None)),
        patient_key=getattr(policy, "patient_key", None),
        date_shift_max_days=_optional_int(getattr(policy, "date_shift_max_days", None)),
        date_shift_secret=getattr(policy, "date_shift_secret", None),
        uid_salt=getattr(policy, "uid_salt", _DEFAULT_UID_SALT),
        keep_year=bool(getattr(policy, "keep_year", False)),
    )


def _optional_int(value: Any | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any | None, *, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _resolve_shift_days(policy: DicomHeaderDeidPolicy) -> int:
    from openmed.core.pii import _resolve_date_shift_days

    shift_days = _resolve_date_shift_days(
        date_shift_days=policy.date_shift_days,
        patient_key=policy.patient_key,
        date_shift_max_days=policy.date_shift_max_days,
        date_shift_secret=policy.date_shift_secret,
    )
    if shift_days == 0:
        raise ValueError("date_shift_days must be non-zero for DICOM de-id")
    return shift_days


def _resolve_pixel_model_name(models: Any, policy_model_name: str | None) -> str | None:
    if policy_model_name:
        return str(policy_model_name)
    if models is None:
        return None
    if isinstance(models, str):
        return models
    if isinstance(models, Mapping):
        for key in ("pii_model_name", "pii_model", "model_name", "pii"):
            value = models.get(key)
            if value:
                return str(value)
        return None
    for attr in ("pii_model_name", "pii_model", "model_name"):
        value = getattr(models, attr, None)
        if value:
            return str(value)
    return None


def _copy_pixel_array(dataset: Any) -> Any:
    numpy = _import_numpy()
    return numpy.array(dataset.pixel_array, copy=True)


def _decompress_pixel_data(dataset: Any) -> None:
    """Normalize compressed Pixel Data before editing and writing raw bytes."""
    file_meta = getattr(dataset, "file_meta", None)
    transfer_syntax = getattr(file_meta, "TransferSyntaxUID", None)
    if transfer_syntax is None or not bool(
        getattr(transfer_syntax, "is_compressed", False)
    ):
        return
    try:
        dataset.decompress(generate_instance_uid=False)
    except Exception:
        raise ValueError(
            "Compressed DICOM Pixel Data could not be decoded with the installed "
            "pixel-data codecs"
        ) from None


def _iter_pixel_frames(pixel_array: Any, dataset: Any) -> Sequence[Any]:
    number_of_frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    if number_of_frames > 1 and getattr(pixel_array, "ndim", 0) >= 3:
        frame_count = min(number_of_frames, int(pixel_array.shape[0]))
        return tuple(pixel_array[index] for index in range(frame_count))
    return (pixel_array,)


def _pixel_bytes(pixel_array: Any) -> bytes:
    numpy = _import_numpy()
    return numpy.ascontiguousarray(pixel_array).tobytes()


def _detect_frame_pixel_findings(
    frame: Any,
    *,
    dataset: Any,
    frame_index: int,
    policy: DicomPixelRedactionPolicy,
    model_name: str | None,
    header_recognizer: Any,
    lang: str | None,
) -> tuple[DicomPixelFinding, ...]:
    from . import ocr as ocr_mod

    languages = [lang] if lang else None
    ocr_result = ocr_mod.ocr(
        _frame_for_ocr(frame, dataset),
        engine=policy.ocr_engine,
        languages=languages,
    )
    document = ocr_result.to_document()
    entities = _detect_pixel_entities(
        document.text,
        model_name=model_name,
        confidence_threshold=policy.confidence_threshold,
        lang=lang,
        header_recognizer=header_recognizer,
        custom_recognizer=policy.custom_recognizer,
    )
    return _project_entities_to_findings(
        document,
        entities,
        frame_index=frame_index,
        frame_shape=frame.shape,
        padding=policy.bbox_padding,
    )


def _detect_pixel_entities(
    text: str,
    *,
    model_name: str | None,
    confidence_threshold: float,
    lang: str | None,
    header_recognizer: Any,
    custom_recognizer: Any,
) -> tuple[Any, ...]:
    if not text.strip():
        return ()

    result = _extract_dicom_pixel_phi(
        text,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        lang=lang,
        custom_recognizer=custom_recognizer,
    )
    entities = list(getattr(result, "entities", ()) or ())
    if header_recognizer is not None:
        entities.extend(header_recognizer.detect_entities(text))
    return _dedupe_entities(text, entities)


def _extract_dicom_pixel_phi(
    text: str,
    *,
    model_name: str | None,
    confidence_threshold: float,
    lang: str | None,
    custom_recognizer: Any,
) -> Any:
    from openmed.core.pii import extract_pii

    kwargs: dict[str, Any] = {
        "confidence_threshold": confidence_threshold,
        "lang": lang or "en",
        "custom_recognizer": custom_recognizer,
    }
    if model_name is not None:
        kwargs["model_name"] = model_name
    return extract_pii(text, **kwargs)


def _dedupe_entities(text: str, entities: Sequence[Any]) -> tuple[Any, ...]:
    deduped: list[Any] = []
    seen: set[tuple[int, int, str]] = set()
    for entity in entities:
        bounds = _entity_bounds(entity, text)
        if bounds is None:
            continue
        label = _entity_label(entity)
        key = (bounds[0], bounds[1], label)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return tuple(deduped)


def _project_entities_to_findings(
    document: ExtractedDocument,
    entities: Sequence[Any],
    *,
    frame_index: int,
    frame_shape: Sequence[int],
    padding: int,
) -> tuple[DicomPixelFinding, ...]:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    findings: list[DicomPixelFinding] = []
    seen_bboxes: set[tuple[int, int, int, int]] = set()
    for entity in entities:
        bounds = _entity_bounds(entity, document.text)
        if bounds is None:
            continue
        surface = document.text[bounds[0] : bounds[1]]
        for span in document.spans:
            if span.bbox is None or not _intervals_overlap(
                span.start, span.end, bounds[0], bounds[1]
            ):
                continue
            bbox = _clamp_bbox(span.bbox, width=width, height=height, padding=padding)
            if bbox is None or bbox in seen_bboxes:
                continue
            seen_bboxes.add(bbox)
            findings.append(
                DicomPixelFinding(
                    frame_index=frame_index,
                    bbox=bbox,
                    label=_entity_label(entity),
                    confidence=_entity_confidence(entity),
                    text_sha256=_hash_value(surface),
                    text_length=len(surface),
                    sources=_entity_sources(entity),
                )
            )
    return tuple(findings)


def _residual_text_report(
    frame_views: Sequence[Any],
    *,
    dataset: Any,
    policy: DicomPixelRedactionPolicy,
    model_name: str | None,
    header_recognizer: Any,
    lang: str | None,
) -> DicomResidualTextReport:
    residuals: list[DicomPixelFinding] = []
    for frame_index, frame in enumerate(frame_views):
        residuals.extend(
            _detect_frame_pixel_findings(
                frame,
                dataset=dataset,
                frame_index=frame_index,
                policy=policy,
                model_name=model_name,
                header_recognizer=header_recognizer,
                lang=lang,
            )
        )
    return DicomResidualTextReport(
        frame_count=len(frame_views),
        residuals=tuple(residuals),
    )


def _frame_for_ocr(frame: Any, dataset: Any) -> Any:
    numpy = _import_numpy()
    array = numpy.asarray(frame)
    if _is_color_frame(array, dataset):
        return _normalize_color_frame(array)

    image = _normalize_uint8(array)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = 255 - image
    return image


def _is_color_frame(frame: Any, dataset: Any) -> bool:
    samples = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    return samples > 1 or photometric in {"RGB", "YBR_FULL", "YBR_FULL_422"}


def _normalize_color_frame(frame: Any) -> Any:
    numpy = _import_numpy()
    array = numpy.asarray(frame)
    if array.dtype == numpy.uint8:
        return numpy.array(array, copy=True)
    return _normalize_uint8(array)


def _normalize_uint8(array: Any) -> Any:
    numpy = _import_numpy()
    values = numpy.asarray(array)
    if values.dtype == numpy.uint8:
        return numpy.array(values, copy=True)
    values = values.astype("float32", copy=False)
    minimum = float(numpy.nanmin(values))
    maximum = float(numpy.nanmax(values))
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum == maximum:
        return numpy.zeros(values.shape, dtype=numpy.uint8)
    scaled = (values - minimum) * (255.0 / (maximum - minimum))
    return numpy.clip(scaled, 0, 255).astype(numpy.uint8)


def _blackout_bbox(
    frame: Any,
    bbox: tuple[int, int, int, int],
    dataset: Any,
) -> None:
    x0, y0, x1, y1 = bbox
    value = _pixel_black_value(frame, dataset)
    if getattr(frame, "ndim", 0) >= 3:
        frame[y0:y1, x0:x1, ...] = value
    else:
        frame[y0:y1, x0:x1] = value


def _pixel_black_value(frame: Any, dataset: Any) -> int:
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() != "MONOCHROME1":
        return 0
    bits_stored = int(
        getattr(dataset, "BitsStored", getattr(dataset, "BitsAllocated", 8)) or 8
    )
    return int((2**bits_stored) - 1)


def _clamp_bbox(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    x0 = max(0, math.floor(float(bbox[0])) - padding)
    y0 = max(0, math.floor(float(bbox[1])) - padding)
    x1 = min(width, math.ceil(float(bbox[2])) + padding)
    y1 = min(height, math.ceil(float(bbox[3])) + padding)
    if x0 >= x1 or y0 >= y1:
        return None
    return (x0, y0, x1, y1)


def _intervals_overlap(
    start: int,
    end: int,
    other_start: int,
    other_end: int,
) -> bool:
    return start < other_end and end > other_start


def _entity_bounds(entity: Any, text: str) -> tuple[int, int] | None:
    start = getattr(entity, "start", None)
    end = getattr(entity, "end", None)
    if (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(text)
    ):
        return start, end

    surface = str(getattr(entity, "text", "") or "")
    if not surface:
        return None
    found = text.find(surface)
    if found < 0:
        return None
    return found, found + len(surface)


def _entity_label(entity: Any) -> str:
    return str(
        getattr(entity, "canonical_label", None)
        or getattr(entity, "entity_type", None)
        or getattr(entity, "label", None)
        or "UNKNOWN"
    )


def _entity_confidence(entity: Any) -> float:
    try:
        return float(getattr(entity, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _entity_sources(entity: Any) -> tuple[str, ...]:
    sources = getattr(entity, "sources", None)
    if sources:
        return tuple(str(source) for source in sources)
    metadata = getattr(entity, "metadata", None) or {}
    if isinstance(metadata, Mapping):
        source = metadata.get("detector") or metadata.get("source")
        if source:
            return (str(source),)
        custom = metadata.get("custom_recognizer")
        if isinstance(custom, Mapping) and custom.get("detector"):
            return (str(custom["detector"]),)
    return ("model",)


def _header_seed_recognizer(dataset: Any) -> Any:
    terms = _header_seed_terms(dataset)
    if not terms:
        return None
    from openmed.core.custom_recognizer import CustomRecognizer

    return CustomRecognizer.from_config(
        {
            "case_sensitive": False,
            "deny_terms": [
                {
                    "term": term,
                    "label": label,
                    "confidence": 1.0,
                    "id": _header_seed_rule_id(keyword, term),
                }
                for keyword, label, term in terms
            ],
        }
    )


def _header_seed_terms(dataset: Any) -> tuple[tuple[str, str, str], ...]:
    specs = (
        ("PatientName", "NAME"),
        ("OtherPatientNames", "NAME"),
        ("PatientBirthName", "NAME"),
        ("PatientMothersBirthName", "NAME"),
        ("PatientID", "ID_NUM"),
        ("OtherPatientIDs", "ID_NUM"),
        ("AccessionNumber", "ID_NUM"),
        ("AdmissionID", "ID_NUM"),
        ("StudyID", "ID_NUM"),
        ("PatientBirthDate", "DATE"),
        ("StudyDate", "DATE"),
        ("SeriesDate", "DATE"),
        ("ContentDate", "DATE"),
        ("AcquisitionDate", "DATE"),
    )
    terms: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for keyword, label in specs:
        value = getattr(dataset, keyword, None)
        for raw in _dicom_value_strings(value):
            for term in _header_value_variants(raw, keyword=keyword):
                normalized = " ".join(term.split())
                if not normalized:
                    continue
                dedupe_key = normalized.casefold()
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                terms.append((keyword, label, normalized))
    return tuple(terms)


def _dicom_value_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, bytes):
        return (value.decode("utf-8", errors="ignore"),)
    if isinstance(value, list | tuple):
        return tuple(item for child in value for item in _dicom_value_strings(child))
    return (str(value),)


def _header_value_variants(value: str, *, keyword: str) -> tuple[str, ...]:
    text = value.strip()
    if not text:
        return ()

    variants = [text]
    if "^" in text:
        parts = [part.strip() for part in text.split("^") if part.strip()]
        if parts:
            variants.append(" ".join(parts))
        if len(parts) >= 2:
            variants.append(f"{parts[1]} {parts[0]}")
            variants.append(f"{parts[0]} {parts[1]}")

    if keyword.lower().endswith("date"):
        variants.extend(_date_variants(text))
    return tuple(variants)


def _date_variants(value: str) -> tuple[str, ...]:
    text = value.strip()
    if not re.fullmatch(r"\d{8}", text):
        return ()
    year, month, day = text[:4], text[4:6], text[6:8]
    return (
        f"{year}-{month}-{day}",
        f"{month}/{day}/{year}",
        f"{day}/{month}/{year}",
    )


def _header_seed_rule_id(keyword: str, term: str) -> str:
    digest = _hash_value(f"{keyword}:{term}")[:12]
    return f"dicom_header_{keyword}_{digest}"


def _deidentify_dataset(dataset: Any, context: _Context, *, location: str) -> None:
    for tag in list(dataset.keys()):
        element = dataset[tag]
        tag_int = int(element.tag)

        if element.tag.is_private:
            _record_action(
                element,
                context,
                action="remove",
                ps315_action="X",
                location=location,
            )
            del dataset[tag]
            context.private_tag_removed_count += 1
            continue

        if tag_int in _REMOVE_TAGS:
            _record_action(
                element,
                context,
                action="remove",
                ps315_action="X",
                location=location,
            )
            del dataset[tag]
            continue

        if tag_int in _ZERO_TAGS:
            _clear_element(element, context, ps315_action="Z", location=location)
            continue

        if element.VR == "SQ":
            _deidentify_sequence(element, context, location=location)
            continue

        if _should_remap_uid(element):
            _remap_uid_element(element, context, location=location)
        elif element.VR in {"DA", "DT"}:
            _shift_date_element(element, context, location=location)
        elif element.VR == "TM":
            _clear_element(element, context, ps315_action="Z", location=location)
        elif element.VR == "PN":
            _clear_element(element, context, ps315_action="Z", location=location)


def _deidentify_sequence(element: Any, context: _Context, *, location: str) -> None:
    for index, item in enumerate(element.value or ()):
        child_location = f"{location}.{_keyword(element)}[{index}]"
        _deidentify_dataset(item, context, location=child_location)


def _seeded_phi_scrub_terms(dataset: Any) -> tuple[str, ...]:
    terms = {
        _normalize_seeded_phi_text(term)
        for _keyword_name, _label, term in _header_seed_terms(dataset)
    }

    def collect(source: Any) -> None:
        for tag in list(source.keys()):
            element = source[tag]
            if element.VR == "SQ":
                for item in element.value or ():
                    collect(item)
                continue
            if str(element.VR) not in _SEEDED_PHI_SOURCE_VRS:
                continue
            tag_int = int(element.tag)
            sensitive_source = (
                element.tag.is_private
                or tag_int in _REMOVE_TAGS
                or tag_int in _ZERO_TAGS
                or element.VR in {"DA", "DT", "PN", "TM"}
                or _should_remap_uid(element)
            )
            if not sensitive_source:
                continue
            for raw in _dicom_value_strings(element.value):
                for variant in _header_value_variants(raw, keyword=_keyword(element)):
                    terms.add(_normalize_seeded_phi_text(variant))

    collect(dataset)
    return tuple(
        sorted((term for term in terms if len(term) >= 3), key=len, reverse=True)
    )


def _clear_seeded_phi_copies(
    dataset: Any,
    terms: Sequence[str],
    context: _Context,
    *,
    location: str,
) -> None:
    if not terms:
        return
    for tag in list(dataset.keys()):
        element = dataset[tag]
        if element.VR == "SQ":
            for index, item in enumerate(element.value or ()):
                child_location = f"{location}.{_keyword(element)}[{index}]"
                _clear_seeded_phi_copies(
                    item,
                    terms,
                    context,
                    location=child_location,
                )
            continue
        if str(element.VR) not in _SEEDED_PHI_SCRUB_VRS:
            continue
        values = _dicom_value_strings(element.value)
        if any(
            term in _normalize_seeded_phi_text(value)
            for value in values
            for term in terms
        ):
            _clear_element(element, context, ps315_action="Z", location=location)


def _normalize_seeded_phi_text(value: Any) -> str:
    text = re.sub(r"[\^=,_]+", " ", str(value))
    return " ".join(text.casefold().split())


def _set_standard_deid_markers(dataset: Any, context: _Context) -> None:
    dataset.PatientIdentityRemoved = "YES"
    dataset.DeidentificationMethod = _DEID_METHOD
    dataset.LongitudinalTemporalInformationModified = "MODIFIED"

    for tag_int, keyword, vr, ps315_action in (
        (0x00120062, "PatientIdentityRemoved", "CS", "D"),
        (0x00120063, "DeidentificationMethod", "LO", "D"),
        (0x00280303, "LongitudinalTemporalInformationModified", "CS", "D"),
    ):
        context.actions.append(
            DicomHeaderAction(
                tag=_format_tag_int(tag_int),
                keyword=keyword,
                vr=vr,
                action="replace",
                ps315_action=ps315_action,
                location="Dataset",
            )
        )


def _deidentify_file_meta(dataset: Any, context: _Context) -> None:
    file_meta = getattr(dataset, "file_meta", None)
    if file_meta is None:
        return

    media_uid = file_meta.get((0x0002, 0x0003))
    if media_uid is not None:
        _remap_uid_element(media_uid, context, location="FileMetaDataset")

    source_ae_title = file_meta.get((0x0002, 0x0016))
    if source_ae_title is not None:
        _record_action(
            source_ae_title,
            context,
            action="remove",
            ps315_action="X",
            location="FileMetaDataset",
        )
        del file_meta[(0x0002, 0x0016)]

    file_meta.ImplementationVersionName = "OPENMED_DEID"
    context.actions.append(
        DicomHeaderAction(
            tag=_format_tag_int(0x00020013),
            keyword="ImplementationVersionName",
            vr="SH",
            action="replace",
            ps315_action="D",
            location="FileMetaDataset",
        )
    )


def _clear_element(
    element: Any,
    context: _Context,
    *,
    ps315_action: str,
    location: str,
) -> None:
    _record_action(
        element,
        context,
        action="clear",
        ps315_action=ps315_action,
        location=location,
    )
    element.value = ""


def _should_remap_uid(element: Any) -> bool:
    if element.VR != "UI":
        return False
    keyword = _keyword(element)
    if keyword in _UID_KEEP_KEYWORDS:
        return False
    return keyword in _UID_REMAP_KEYWORDS or keyword.endswith("UID")


def _remap_uid_element(element: Any, context: _Context, *, location: str) -> None:
    _record_action(
        element,
        context,
        action="replace_uid",
        ps315_action="U",
        location=location,
    )
    element.value = _map_dicom_values(
        element.value, lambda value: _remap_uid(value, context)
    )


def _remap_uid(value: Any, context: _Context) -> str:
    text = str(value).strip()
    if not text:
        return text
    replacement = context.uid_map.get(text)
    if replacement is None:
        digest = hashlib.sha256(context.uid_salt + text.encode("utf-8")).digest()
        replacement = f"2.25.{uuid.UUID(bytes=digest[:16]).int}"
        context.uid_map[text] = replacement
    return replacement


def _shift_date_element(element: Any, context: _Context, *, location: str) -> None:
    _record_action(
        element,
        context,
        action="shift_date",
        ps315_action="C",
        location=location,
    )
    if element.VR == "DA":
        element.value = _map_dicom_values(
            element.value,
            lambda value: _shift_dicom_date(value, context),
        )
    else:
        element.value = _map_dicom_values(
            element.value,
            lambda value: _shift_dicom_datetime(value, context),
        )


def _map_dicom_values(value: Any, mapper: Any) -> Any:
    if isinstance(value, list | tuple):
        return [mapper(item) for item in value]
    return mapper(value)


def _shift_dicom_date(value: Any, context: _Context) -> str:
    text = str(value).strip()
    if not text:
        return text
    try:
        shifted = datetime.strptime(text, "%Y%m%d") + timedelta(
            days=context.date_shift_days
        )
        if context.keep_year:
            shifted = _replace_year_safe(shifted, int(text[:4]))
    except (TypeError, ValueError, OverflowError):
        return ""
    return shifted.strftime("%Y%m%d")


def _shift_dicom_datetime(value: Any, context: _Context) -> str:
    text = str(value).strip()
    if not text:
        return text
    match = _DT_DATE_RE.match(text)
    if match is None:
        return ""
    shifted = _shift_dicom_date(match.group("date"), context)
    return f"{shifted}{match.group('rest')}" if shifted else ""


def _replace_year_safe(date_value: datetime, year: int) -> datetime:
    try:
        return date_value.replace(year=year)
    except ValueError:
        return date_value.replace(year=year, month=2, day=28)


def _record_action(
    element: Any,
    context: _Context,
    *,
    action: str,
    ps315_action: str,
    location: str,
) -> None:
    context.actions.append(
        DicomHeaderAction(
            tag=_format_tag(element.tag),
            keyword=_keyword(element),
            vr=str(element.VR),
            action=action,
            ps315_action=ps315_action,
            location=location,
            value_sha256=_hash_value(element.value),
            value_length=len(str(element.value)),
        )
    )


def _keyword(element: Any) -> str:
    keyword = getattr(element, "keyword", "")
    return str(keyword) if keyword else _format_tag(element.tag)


def _format_tag(tag: Any) -> str:
    return f"({int(tag) >> 16:04X},{int(tag) & 0xFFFF:04X})"


def _format_tag_int(tag: int) -> str:
    return f"({tag >> 16:04X},{tag & 0xFFFF:04X})"


def _hash_value(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def _bytes_value(value: str | bytes, *, name: str) -> bytes:
    if isinstance(value, bytes):
        if not value:
            raise ValueError(f"{name} must be non-empty")
        return value
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        if not encoded:
            raise ValueError(f"{name} must be non-empty")
        return encoded
    raise TypeError(f"{name} must be str or bytes")


def _save_dataset(dataset: Any, output_path: Path) -> None:
    try:
        dataset.save_as(output_path, enforce_file_format=True)
    except TypeError:  # pragma: no cover - older pydicom compatibility.
        dataset.save_as(output_path, write_like_original=False)


def _dicom_handler(
    path: str | Path,
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    pixel_result = redact_dicom_pixels(path, policy=policy, models=models, lang=lang)
    header_policy = _coerce_policy(policy)
    header_policy = DicomHeaderDeidPolicy(
        output_path=pixel_result.output_path,
        date_shift_days=header_policy.date_shift_days,
        patient_key=header_policy.patient_key,
        date_shift_max_days=header_policy.date_shift_max_days,
        date_shift_secret=header_policy.date_shift_secret,
        uid_salt=header_policy.uid_salt,
        keep_year=header_policy.keep_year,
    )
    result = deidentify_dicom_headers(pixel_result.output_path, policy=header_policy)
    return ExtractedDocument(
        text="",
        metadata={
            "format": "dicom",
            "dicom_header_deid": result.to_audit_report(),
            "dicom_pixel_redaction": pixel_result.to_audit_report(),
        },
    )


register_handler(".dcm", _dicom_handler, requires_multimodal=False)


__all__ = [
    "DicomHeaderAction",
    "DicomHeaderDeidPolicy",
    "DicomHeaderDeidResult",
    "deidentify_dicom_headers",
]
