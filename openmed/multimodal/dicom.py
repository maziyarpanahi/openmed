"""DICOM PS3.15 header de-identification.

The module imports pydicom lazily so the multimodal package remains importable
without optional imaging dependencies. Header provenance records tags and
actions only; raw PHI and original UIDs are intentionally omitted.
"""

from __future__ import annotations

import hashlib
import importlib
import re
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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
    _deidentify_dataset(dataset, context, location="Dataset")
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


def _import_pydicom() -> Any:
    try:
        return importlib.import_module("pydicom")
    except ImportError as exc:  # pragma: no cover - exercised without extra.
        raise MissingDependencyError(
            dependency="pydicom", instruction=_DICOM_INSTALL_HINT
        ) from exc


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
    del models, lang
    result = deidentify_dicom_headers(path, policy=policy)
    return ExtractedDocument(
        text="",
        metadata={
            "format": "dicom",
            "dicom_header_deid": result.to_audit_report(),
        },
    )


register_handler(".dcm", _dicom_handler, requires_multimodal=False)


__all__ = [
    "DicomHeaderAction",
    "DicomHeaderDeidPolicy",
    "DicomHeaderDeidResult",
    "deidentify_dicom_headers",
]
