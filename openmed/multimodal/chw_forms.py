"""De-identification for ODK, CommCare, and KoBoToolbox form exports.

The XForms ecosystem represents the same submission in several shapes: nested
JSON, flat CSV with slash-delimited paths, and long-format repeat CSVs.  This
module keeps those structures intact while applying field-level privacy rules.
It uses only the standard library at import time; the regular OpenMed text
de-identification pipeline is imported lazily only for narrative answers.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.core.labels import (
    DATE,
    DATE_OF_BIRTH,
    DIRECT_IDENTIFIER,
    GPS_COORDINATES,
    ID_NUM,
    LABEL_METADATA,
    LOCATION,
    OTHER,
    PERSON,
    PHONE,
    QUASI_IDENTIFIER,
    STREET_ADDRESS,
    normalize_label,
)

from .base import ExtractedDocument, register_handler
from .tabular_csv import (
    ACTION_DATE_SHIFT,
    ACTION_DROP,
    ACTION_FREE_TEXT_REDACT,
    ACTION_HASH,
    ACTION_KEEP,
    ACTION_MASK,
    DIRECT_ID,
    QUASI_ID,
    SAFE,
    derive_date_shift_days,
)

ACTION_GENERALIZE_GEO = "generalize_geo"

SUPPORTED_CHW_ACTIONS = frozenset(
    {
        ACTION_DATE_SHIFT,
        ACTION_DROP,
        ACTION_FREE_TEXT_REDACT,
        ACTION_GENERALIZE_GEO,
        ACTION_HASH,
        ACTION_KEEP,
        ACTION_MASK,
    }
)

TextRedactor = Callable[[str], str]


@dataclass(frozen=True)
class ChwFieldDecision:
    """Classification and action for one logical XForm field path."""

    field_path: str
    path_parts: tuple[str, ...]
    assigned_class: str
    action: str
    canonical_label: str | None = None
    policy_label: str | None = None
    detection_source: str = "default"

    def to_manifest(
        self,
        *,
        platform: str,
        row_count: int,
        value_count: int,
        value_count_affected: int,
    ) -> dict[str, Any]:
        """Return a PHI-free manifest entry containing policy and counts only."""
        return {
            "platform": platform,
            "column_name": self.field_path,
            "field_path": self.field_path,
            "path_parts": list(self.path_parts),
            "assigned_class": self.assigned_class,
            "canonical_label": self.canonical_label,
            "policy_label": self.policy_label,
            "action": self.action,
            "detection_source": self.detection_source,
            "row_count": row_count,
            "value_count": value_count,
            "value_count_affected": value_count_affected,
        }


@dataclass(frozen=True)
class RedactedChwForm:
    """A structure-preserving CHW form export and its PHI-safe manifest."""

    text: str
    format: str
    platform: str
    row_count: int
    manifest: tuple[dict[str, Any], ...]
    data: Any
    delimiter: str | None = None

    def to_document(self) -> ExtractedDocument:
        """Bridge the export into the shared multimodal document contract."""
        metadata: dict[str, Any] = {
            "format": f"chw_form_{self.format}",
            "platform": self.platform,
            "row_count": self.row_count,
            "redaction_manifest": list(self.manifest),
        }
        if self.delimiter is not None:
            metadata["delimiter"] = self.delimiter
        return ExtractedDocument(text=self.text, metadata=metadata)


@dataclass(frozen=True)
class _PolicyOptions:
    metadata_action: str = ACTION_HASH
    geopoint_action: str = ACTION_GENERALIZE_GEO
    geo_precision: int = 2
    date_shift_days: int | None = None
    header_heuristics: Mapping[str, str] | None = None
    action_overrides: Mapping[str, str] | None = None


@dataclass
class _FieldStats:
    decision: ChwFieldDecision
    value_count: int = 0
    affected_count: int = 0


_DROP_VALUE = object()
_WRAPPER_KEYS = frozenset({"data", "rows", "submissions", "value"})

_METADATA_KEYS = frozenset(
    {
        "caseid",
        "deviceid",
        "instanceid",
        "parentkey",
        "submissiontime",
        "uuid",
    }
)
_METADATA_RAW_KEYS = frozenset(
    {
        "__id",
        "__version__",
        "_id",
        "_index",
        "_parent_index",
        "_parent_table_name",
        "_status",
        "_submission__uuid",
        "_submission_time",
        "_submitted_by",
        "_validation_status",
        "_uuid",
        "@case_id",
        "case_id",
        "deviceid",
        "instanceid",
        "key",
        "parent_key",
    }
)

_PERSON_KEYS = frozenset(
    {
        "beneficiaryname",
        "childname",
        "clientname",
        "firstname",
        "fullname",
        "householdhead",
        "lastname",
        "membername",
        "mothername",
        "name",
        "patientname",
        "respondentname",
    }
)
_PHONE_KEYS = frozenset(
    {
        "contactnumber",
        "mobile",
        "mobilenumber",
        "msisdn",
        "phone",
        "phonenumber",
        "telephone",
    }
)
_ID_KEYS = frozenset(
    {
        "beneficiaryid",
        "birthcertificatenumber",
        "clientid",
        "householdid",
        "idnumber",
        "nationalid",
        "nationalidnumber",
        "nhifnumber",
        "passportnumber",
        "patientid",
        "recordid",
    }
)
_DOB_KEYS = frozenset({"birthdate", "dateofbirth", "dob"})
_DATE_KEYS = frozenset(
    {
        "admissiondate",
        "encounterdate",
        "eventdate",
        "followupdate",
        "servicedate",
        "visitdate",
    }
)
_ADDRESS_KEYS = frozenset(
    {
        "address",
        "homeaddress",
        "physicaladdress",
        "residentialaddress",
        "streetaddress",
    }
)
_LOCATION_KEYS = frozenset(
    {"county", "district", "location", "province", "subcounty", "village", "ward"}
)
_GEO_KEYS = frozenset(
    {
        "coordinate",
        "coordinates",
        "geo",
        "geolocation",
        "geopoint",
        "geoshape",
        "geotrace",
        "gps",
        "gpscoordinates",
        "latitude",
        "longitude",
    }
)
_FREE_TEXT_MARKERS = (
    "comment",
    "counsellingnote",
    "describe",
    "description",
    "freetext",
    "narrative",
    "note",
    "otherspecify",
    "visitdetail",
)

_JSON_MARKERS = frozenset(
    {
        "__id",
        "_submission_time",
        "_uuid",
        "@case_id",
        "case_id",
        "deviceid",
        "instanceid",
    }
)
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def parse_xform_path(header: str) -> tuple[str, ...]:
    """Split slash/dot-delimited XForm group and repeat paths."""
    return tuple(part.strip() for part in re.split(r"[/.]", header) if part.strip())


def classify_chw_field(
    field_path: str,
    *,
    sample_values: Sequence[str] = (),
    policy: Mapping[str, Any] | None = None,
) -> ChwFieldDecision:
    """Classify one XForm path using platform metadata and canonical labels."""
    options = _policy_options(policy)
    parts = parse_xform_path(field_path) or (field_path,)
    raw_leaf = parts[-1].strip()
    leaf = _field_key(raw_leaf)
    normalized_parts = tuple(_field_key(part) for part in parts)

    if _is_metadata_path(parts):
        decision = _decision(
            field_path,
            parts,
            ID_NUM,
            options.metadata_action,
            "platform_metadata",
        )
    elif _is_geo_path(normalized_parts) or any(
        _looks_like_geopoint(value) for value in sample_values
    ):
        decision = ChwFieldDecision(
            field_path=field_path,
            path_parts=parts,
            assigned_class=QUASI_ID,
            action=options.geopoint_action,
            canonical_label=GPS_COORDINATES,
            policy_label=QUASI_IDENTIFIER,
            detection_source="xform_geopoint",
        )
    else:
        configured_label = _configured_label(field_path, leaf, options)
        label = configured_label or _canonical_label_for_key(leaf)
        if label is not None:
            source = "policy_header" if configured_label else "xform_header"
            decision = _decision(
                field_path,
                parts,
                label,
                _default_action(label),
                source,
            )
        elif any(marker in leaf for marker in _FREE_TEXT_MARKERS):
            decision = ChwFieldDecision(
                field_path=field_path,
                path_parts=parts,
                assigned_class=SAFE,
                action=ACTION_FREE_TEXT_REDACT,
                detection_source="narrative_header",
            )
        else:
            decision = ChwFieldDecision(
                field_path=field_path,
                path_parts=parts,
                assigned_class=SAFE,
                action=ACTION_FREE_TEXT_REDACT,
                detection_source="unclassified_text_safety",
            )

    override = _action_override(decision, options.action_overrides)
    if override is None:
        return decision
    return ChwFieldDecision(
        field_path=decision.field_path,
        path_parts=decision.path_parts,
        assigned_class=decision.assigned_class,
        action=override,
        canonical_label=decision.canonical_label,
        policy_label=decision.policy_label,
        detection_source=decision.detection_source,
    )


def redact_chw_form(
    source: str | os.PathLike[str] | Any,
    *,
    platform: str | None = None,
    policy: Mapping[str, Any] | None = None,
    models: Any | None = None,
    lang: str = "en",
    text_redactor: TextRedactor | None = None,
) -> RedactedChwForm:
    """De-identify a JSON or CSV submission export without changing its shape.

    ``policy`` accepts ``metadata_action`` (``hash`` or ``drop``),
    ``geopoint_action`` (``generalize_geo`` or ``drop``), ``geo_precision``,
    ``date_shift_days``, ``header_heuristics``, and ``action_overrides``.
    """
    text = _read_text(source)
    export_format = _export_format(source, text)
    options = _policy_options(policy)
    redactor = text_redactor or _text_redactor_from_models(models)

    if export_format == "json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid CHW form JSON export") from exc
        resolved_platform = _normalize_platform(platform) or _detect_json_platform(data)
        return _redact_json_export(
            data,
            platform=resolved_platform,
            options=options,
            lang=lang,
            text_redactor=redactor,
        )

    return _redact_csv_export(
        text,
        platform=_normalize_platform(platform),
        options=options,
        lang=lang,
        text_redactor=redactor,
    )


def _redact_json_export(
    data: Any,
    *,
    platform: str,
    options: _PolicyOptions,
    lang: str,
    text_redactor: TextRedactor | None,
) -> RedactedChwForm:
    values: dict[tuple[str, ...], list[str]] = {}
    _collect_json_values(data, (), values)
    decisions = {
        path: classify_chw_field(
            "/".join(path),
            sample_values=samples,
            policy=_options_mapping(options),
        )
        for path, samples in values.items()
        if path
    }
    stats = {path: _FieldStats(decision) for path, decision in decisions.items()}
    transformed = _transform_json(
        data,
        (),
        decisions=decisions,
        stats=stats,
        options=options,
        lang=lang,
        text_redactor=text_redactor,
        record_date_shift_days=_record_date_shift_days(data, 0, options),
    )
    output_text = (
        json.dumps(
            transformed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    row_count = _json_row_count(data)
    manifest = _manifest(stats, platform=platform, row_count=row_count)
    return RedactedChwForm(
        text=output_text,
        format="json",
        platform=platform,
        row_count=row_count,
        manifest=manifest,
        data=transformed,
    )


def _redact_csv_export(
    text: str,
    *,
    platform: str | None,
    options: _PolicyOptions,
    lang: str,
    text_redactor: TextRedactor | None,
) -> RedactedChwForm:
    delimiter = _sniff_delimiter(text)
    records = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
    if not records:
        raise ValueError("CHW form CSV export is empty")
    headers = tuple(records[0])
    width = len(headers)
    if any(len(row) != width for row in records[1:]):
        raise ValueError("CHW form CSV rows must match the header width")
    rows = [tuple(row) for row in records[1:]]
    resolved_platform = platform or _detect_csv_platform(headers)

    decisions: list[ChwFieldDecision] = []
    for index, header in enumerate(headers):
        samples = tuple(row[index] for row in rows if row[index])[:50]
        decisions.append(
            classify_chw_field(
                header,
                sample_values=samples,
                policy=_options_mapping(options),
            )
        )
    stats = {(decision.field_path,): _FieldStats(decision) for decision in decisions}
    output_indices = [
        index
        for index, decision in enumerate(decisions)
        if decision.action != ACTION_DROP
    ]
    output_rows: list[tuple[str, ...]] = []
    for row_index, row in enumerate(rows):
        record_date_shift_days = derive_date_shift_days(
            row,
            record_index=row_index,
            fixed_days=options.date_shift_days,
            seed="openmed-chw-csv-v1",
        )
        output_row: list[str] = []
        for index, decision in enumerate(decisions):
            stat = stats[(decision.field_path,)]
            value = row[index]
            if value:
                stat.value_count += 1
            redacted, changed = _apply_action(
                value,
                decision,
                options=options,
                lang=lang,
                text_redactor=text_redactor,
                record_date_shift_days=record_date_shift_days,
            )
            if changed:
                stat.affected_count += 1
            if index in output_indices:
                output_row.append(str(redacted))
        output_rows.append(tuple(output_row))

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
    writer.writerow([headers[index] for index in output_indices])
    writer.writerows(output_rows)
    output_text = stream.getvalue()
    manifest = _manifest(stats, platform=resolved_platform, row_count=len(rows))
    return RedactedChwForm(
        text=output_text,
        format="csv" if delimiter == "," else "tsv",
        platform=resolved_platform,
        row_count=len(rows),
        manifest=manifest,
        data=tuple(output_rows),
        delimiter=delimiter,
    )


def _collect_json_values(
    value: Any,
    path: tuple[str, ...],
    fields: dict[tuple[str, ...], list[str]],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = _json_child_path(path, str(key), child)
            _collect_json_values(child, child_path, fields)
        return
    if isinstance(value, list):
        for child in value:
            _collect_json_values(child, path, fields)
        return
    if path and value is not None:
        fields.setdefault(path, []).append(str(value))


def _transform_json(
    value: Any,
    path: tuple[str, ...],
    *,
    decisions: Mapping[tuple[str, ...], ChwFieldDecision],
    stats: Mapping[tuple[str, ...], _FieldStats],
    options: _PolicyOptions,
    lang: str,
    text_redactor: TextRedactor | None,
    record_date_shift_days: int,
) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, child in value.items():
            child_path = _json_child_path(path, str(key), child)
            transformed = _transform_json(
                child,
                child_path,
                decisions=decisions,
                stats=stats,
                options=options,
                lang=lang,
                text_redactor=text_redactor,
                record_date_shift_days=record_date_shift_days,
            )
            if transformed is not _DROP_VALUE:
                output[str(key)] = transformed
        return output
    if isinstance(value, list):
        output_list: list[Any] = []
        for index, child in enumerate(value):
            child_date_shift_days = record_date_shift_days
            if not path:
                child_date_shift_days = _record_date_shift_days(
                    child,
                    index,
                    options,
                )
            transformed = _transform_json(
                child,
                path,
                decisions=decisions,
                stats=stats,
                options=options,
                lang=lang,
                text_redactor=text_redactor,
                record_date_shift_days=child_date_shift_days,
            )
            output_list.append(None if transformed is _DROP_VALUE else transformed)
        return output_list
    if value is None or not path:
        return value

    decision = decisions[path]
    stat = stats[path]
    stat.value_count += 1
    transformed, changed = _apply_action(
        value,
        decision,
        options=options,
        lang=lang,
        text_redactor=text_redactor,
        record_date_shift_days=record_date_shift_days,
    )
    if changed:
        stat.affected_count += 1
    return transformed


def _apply_action(
    value: Any,
    decision: ChwFieldDecision,
    *,
    options: _PolicyOptions,
    lang: str,
    text_redactor: TextRedactor | None,
    record_date_shift_days: int,
) -> tuple[Any, bool]:
    if decision.action == ACTION_KEEP or value is None:
        return value, False
    if decision.action == ACTION_DROP:
        return _DROP_VALUE, bool(value)
    if value == "":
        return value, False
    if decision.action == ACTION_HASH:
        redacted = _hash_value(value, decision.canonical_label or ID_NUM)
    elif decision.action == ACTION_MASK:
        redacted = f"[{decision.canonical_label or 'PHI'}]"
    elif decision.action == ACTION_GENERALIZE_GEO:
        redacted = _generalize_geo(value, precision=options.geo_precision)
    elif decision.action == ACTION_FREE_TEXT_REDACT:
        if not isinstance(value, str):
            return value, False
        redacted = _redact_free_text(
            str(value),
            text_redactor=text_redactor,
            lang=lang,
        )
    elif decision.action == ACTION_DATE_SHIFT:
        redacted = _shift_date_value(
            str(value),
            decision.canonical_label or DATE,
            shift_days=record_date_shift_days,
            lang=lang,
        )
    else:  # pragma: no cover - policy validation prevents this branch
        raise ValueError(f"unsupported CHW form action: {decision.action!r}")
    return redacted, redacted != value


def _decision(
    field_path: str,
    parts: tuple[str, ...],
    label: str,
    action: str,
    source: str,
) -> ChwFieldDecision:
    assigned_class, policy_label = _class_for_label(label)
    return ChwFieldDecision(
        field_path=field_path,
        path_parts=parts,
        assigned_class=assigned_class,
        action=action,
        canonical_label=label,
        policy_label=policy_label,
        detection_source=source,
    )


def _class_for_label(label: str) -> tuple[str, str | None]:
    metadata = LABEL_METADATA.get(label)
    policy_label = str(metadata["policy_label"]) if metadata else None
    if policy_label == DIRECT_IDENTIFIER:
        return DIRECT_ID, policy_label
    if policy_label == QUASI_IDENTIFIER:
        return QUASI_ID, policy_label
    return SAFE, policy_label


def _default_action(label: str) -> str:
    if label == DATE:
        return ACTION_DATE_SHIFT
    if label == ID_NUM:
        return ACTION_HASH
    assigned_class, _ = _class_for_label(label)
    if assigned_class in {DIRECT_ID, QUASI_ID}:
        return ACTION_MASK
    return ACTION_KEEP


def _canonical_label_for_key(key: str) -> str | None:
    if key in _PERSON_KEYS:
        return PERSON
    if key in _PHONE_KEYS:
        return PHONE
    if key in _ID_KEYS or key.endswith(("idnumber", "nationalidentifier")):
        return ID_NUM
    if key in _DOB_KEYS:
        return DATE_OF_BIRTH
    if key in _DATE_KEYS:
        return DATE
    if key in _ADDRESS_KEYS:
        return STREET_ADDRESS
    if key in _LOCATION_KEYS:
        return LOCATION
    canonical = normalize_label(key)
    return canonical if canonical != OTHER else None


def _configured_label(
    field_path: str,
    leaf: str,
    options: _PolicyOptions,
) -> str | None:
    if not options.header_heuristics:
        return None
    normalized: dict[str, str] = {}
    for key, value in options.header_heuristics.items():
        normalized[str(key).lower()] = str(value)
        normalized[_field_key(str(key))] = str(value)
    candidates = (field_path.lower(), _field_key(field_path), leaf)
    for candidate in candidates:
        raw_label = normalized.get(candidate)
        if raw_label is not None:
            label = normalize_label(raw_label)
            return label if label != OTHER else None
    return None


def _action_override(
    decision: ChwFieldDecision,
    overrides: Mapping[str, str] | None,
) -> str | None:
    if not overrides:
        return None
    normalized = {str(key).lower(): str(value) for key, value in overrides.items()}
    leaf = _field_key(decision.path_parts[-1])
    candidates = (
        decision.field_path.lower(),
        _field_key(decision.field_path),
        leaf,
        (decision.canonical_label or "").lower(),
        decision.assigned_class.lower(),
    )
    for candidate in candidates:
        if candidate and candidate in normalized:
            action = normalized[candidate]
            if action not in SUPPORTED_CHW_ACTIONS:
                raise ValueError(f"unsupported CHW form redaction action: {action!r}")
            return action
    return None


def _is_metadata_path(parts: Sequence[str]) -> bool:
    raw_leaf = parts[-1].strip().lower()
    leaf = _field_key(raw_leaf)
    if raw_leaf in _METADATA_RAW_KEYS or leaf in _METADATA_KEYS:
        return True
    normalized_parts = tuple(_field_key(part) for part in parts)
    if {"meta", "system"} & set(normalized_parts[:-1]) and leaf in {
        "end",
        "receivedon",
        "start",
        "submissiondate",
        "timeend",
        "timestart",
        "username",
    }:
        return True
    return bool("case" in normalized_parts and leaf in {"caseid", "id"})


def _is_geo_path(parts: Sequence[str]) -> bool:
    return any(part in _GEO_KEYS for part in parts)


def _looks_like_geopoint(value: str) -> bool:
    first_point = value.split(";", 1)[0].strip().split()
    if len(first_point) < 2:
        return False
    if not all(_NUMBER_RE.fullmatch(item) for item in first_point[:2]):
        return False
    latitude, longitude = (float(item) for item in first_point[:2])
    return -90 <= latitude <= 90 and -180 <= longitude <= 180


def _generalize_geo(value: Any, *, precision: int) -> Any:
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return _rounded_number(value, precision)
    if isinstance(value, list):
        return [_generalize_geo(item, precision=precision) for item in value]
    if isinstance(value, tuple):
        return tuple(_generalize_geo(item, precision=precision) for item in value)
    if isinstance(value, Mapping):
        return {
            str(key): _generalize_geo(item, precision=precision)
            for key, item in value.items()
        }

    points: list[str] = []
    for point in str(value).split(";"):
        components = point.strip().split()
        if len(components) < 2 or not all(
            _NUMBER_RE.fullmatch(item) for item in components[:2]
        ):
            return ""
        points.append(
            " ".join(
                _format_coordinate(float(item), precision) for item in components[:2]
            )
        )
    return ";".join(points)


def _rounded_number(value: int | float, precision: int) -> float:
    rounded = round(float(value), precision)
    return 0.0 if rounded == 0 else rounded


def _format_coordinate(value: float, precision: int) -> str:
    rounded = _rounded_number(value, precision)
    return f"{rounded:.{precision}f}"


def _hash_value(value: Any, label: str) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
    return f"{label}_{digest}"


def _shift_date_value(value: str, label: str, *, shift_days: int, lang: str) -> str:
    from .tabular_csv import _redact_with_core

    return _redact_with_core(
        value,
        label=label,
        method="shift_dates",
        date_shift_days=shift_days,
        keep_year=True,
        lang=lang,
    )


def _record_date_shift_days(
    value: Any,
    record_index: int,
    options: _PolicyOptions,
) -> int:
    material = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return derive_date_shift_days(
        (material,),
        record_index=record_index,
        fixed_days=options.date_shift_days,
        seed="openmed-chw-json-v1",
    )


def _redact_free_text(
    value: str,
    *,
    text_redactor: TextRedactor | None,
    lang: str,
) -> str:
    if text_redactor is not None:
        return str(text_redactor(value))
    from openmed.core.pii import deidentify

    return deidentify(value, method="mask", lang=lang).deidentified_text


def _text_redactor_from_models(models: Any | None) -> TextRedactor | None:
    if callable(models):
        return lambda text: str(models(text))
    if isinstance(models, Mapping):
        candidate = models.get("text_redactor")
        if callable(candidate):
            return lambda text: str(candidate(text))
    candidate = getattr(models, "text_redactor", None)
    if callable(candidate):
        return lambda text: str(candidate(text))
    return None


def _policy_options(policy: Mapping[str, Any] | None) -> _PolicyOptions:
    values = policy if isinstance(policy, Mapping) else {}
    metadata_action = str(values.get("metadata_action", ACTION_HASH))
    geopoint_action = str(values.get("geopoint_action", ACTION_GENERALIZE_GEO))
    if metadata_action not in {ACTION_HASH, ACTION_DROP}:
        raise ValueError("metadata_action must be 'hash' or 'drop'")
    if geopoint_action not in {ACTION_GENERALIZE_GEO, ACTION_DROP}:
        raise ValueError("geopoint_action must be 'generalize_geo' or 'drop'")
    geo_precision = int(values.get("geo_precision", 2))
    if not 0 <= geo_precision <= 4:
        raise ValueError("geo_precision must be between 0 and 4")
    date_shift_days_value = values.get("date_shift_days")
    date_shift_days = (
        int(date_shift_days_value) if date_shift_days_value is not None else None
    )
    header_heuristics = values.get("header_heuristics")
    action_overrides = values.get("action_overrides")
    return _PolicyOptions(
        metadata_action=metadata_action,
        geopoint_action=geopoint_action,
        geo_precision=geo_precision,
        date_shift_days=date_shift_days,
        header_heuristics=(
            header_heuristics if isinstance(header_heuristics, Mapping) else None
        ),
        action_overrides=(
            action_overrides if isinstance(action_overrides, Mapping) else None
        ),
    )


def _options_mapping(options: _PolicyOptions) -> dict[str, Any]:
    return {
        "metadata_action": options.metadata_action,
        "geopoint_action": options.geopoint_action,
        "geo_precision": options.geo_precision,
        "date_shift_days": options.date_shift_days,
        "header_heuristics": options.header_heuristics,
        "action_overrides": options.action_overrides,
    }


def _manifest(
    stats: Mapping[Any, _FieldStats],
    *,
    platform: str,
    row_count: int,
) -> tuple[dict[str, Any], ...]:
    ordered = sorted(stats.values(), key=lambda stat: stat.decision.field_path)
    return tuple(
        stat.decision.to_manifest(
            platform=platform,
            row_count=row_count,
            value_count=stat.value_count,
            value_count_affected=stat.affected_count,
        )
        for stat in ordered
    )


def _json_child_path(path: tuple[str, ...], key: str, value: Any) -> tuple[str, ...]:
    if not path and key.lower() in _WRAPPER_KEYS and isinstance(value, list):
        return path
    parts = parse_xform_path(key) or (key,)
    return path + parts


def _json_row_count(data: Any) -> int:
    if isinstance(data, list):
        return len(data)
    if isinstance(data, Mapping):
        for key, value in data.items():
            if key.lower() in _WRAPPER_KEYS and isinstance(value, list):
                return len(value)
        return 1
    return 0


def _read_text(source: str | os.PathLike[str] | Any) -> str:
    if hasattr(source, "read"):
        content = source.read()
        if not isinstance(content, str):
            raise TypeError("source file-like object must return text")
        return content.removeprefix("\ufeff")
    if isinstance(source, os.PathLike):
        return Path(source).read_text(encoding="utf-8-sig")
    if isinstance(source, str):
        content = source.removeprefix("\ufeff")
        if (
            "\n" in content
            or "\r" in content
            or content.lstrip().startswith(("{", "["))
        ):
            return content
        path = Path(content)
        try:
            if path.exists():
                return path.read_text(encoding="utf-8-sig")
        except OSError:
            pass
        return content
    raise TypeError("source must be a path, text content, or text file-like object")


def _export_format(source: Any, text: str) -> str:
    suffix = (
        Path(str(source)).suffix.lower()
        if isinstance(source, (str, os.PathLike))
        else ""
    )
    if suffix == ".json" or text.lstrip().startswith(("{", "[")):
        return "json"
    return "csv"


def _sniff_delimiter(text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",\t")
        return str(dialect.delimiter)
    except csv.Error:
        first_line = text.splitlines()[0] if text.splitlines() else ""
        return "\t" if "\t" in first_line and "," not in first_line else ","


def _normalize_platform(platform: str | None) -> str | None:
    if platform is None:
        return None
    normalized = platform.strip().lower().replace("toolbox", "").strip()
    aliases = {"odk central": "odk", "commcare hq": "commcare", "kobo": "kobo"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"odk", "commcare", "kobo"}:
        raise ValueError("platform must be 'odk', 'commcare', or 'kobo'")
    return normalized


def _detect_json_platform(data: Any) -> str:
    paths: list[str] = []
    _collect_json_paths(data, (), paths)
    lowered = {path.lower() for path in paths}
    if any("_submission_time" in path or "_uuid" in path for path in lowered):
        return "kobo"
    if any(path.startswith("form/") or "/case/" in path for path in lowered):
        return "commcare"
    return "odk"


def _detect_csv_platform(headers: Sequence[str]) -> str:
    lowered = tuple(header.lower() for header in headers)
    if any("_submission_time" in header or "_uuid" in header for header in lowered):
        return "kobo"
    if any(
        header.startswith("form.") or header.startswith("case.") for header in lowered
    ):
        return "commcare"
    return "odk"


def _collect_json_paths(value: Any, path: tuple[str, ...], paths: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = _json_child_path(path, str(key), child)
            _collect_json_paths(child, child_path, paths)
    elif isinstance(value, list):
        for child in value[:3]:
            _collect_json_paths(child, path, paths)
    elif path:
        paths.append("/".join(path))


def _looks_like_chw_form(path: str | Path) -> bool:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return False
    if source.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return False
        paths: list[str] = []
        _collect_json_paths(data, (), paths)
        raw = {parse_xform_path(item)[-1].lower() for item in paths if item}
        return bool(raw & _JSON_MARKERS)

    try:
        delimiter = _sniff_delimiter(text)
        headers = next(csv.reader(io.StringIO(text), delimiter=delimiter))
    except (csv.Error, StopIteration):
        return False
    metadata_headers = [
        header
        for header in headers
        if _is_metadata_path(parse_xform_path(header) or (header,))
    ]
    if any(
        header.strip().lower() not in {"key", "parent_key"}
        for header in metadata_headers
    ):
        return True
    return bool(
        metadata_headers and any("/" in header or "." in header for header in headers)
    )


def _chw_form_handler(
    path: str | os.PathLike[str],
    *,
    policy: Mapping[str, Any] | None = None,
    models: Any | None = None,
    lang: str | None = None,
) -> ExtractedDocument:
    return redact_chw_form(
        path,
        policy=policy,
        models=models,
        lang=lang or "en",
    ).to_document()


def _field_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


register_handler(
    (".json", ".csv", ".tsv"),
    _chw_form_handler,
    detector=_looks_like_chw_form,
    requires_multimodal=False,
)


__all__ = [
    "ACTION_GENERALIZE_GEO",
    "ChwFieldDecision",
    "RedactedChwForm",
    "classify_chw_field",
    "parse_xform_path",
    "redact_chw_form",
]
