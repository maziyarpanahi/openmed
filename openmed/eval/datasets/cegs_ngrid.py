"""Eval-only loader for the credentialed CEGS N-GRID de-id corpus."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

from openmed.core.labels import (
    AGE,
    CANONICAL_LABELS,
    DATE,
    EMAIL,
    ID_NUM,
    IP_ADDRESS,
    LOCATION,
    OCCUPATION,
    ORGANIZATION,
    OTHER,
    PERSON,
    PHONE,
    SSN,
    STREET_ADDRESS,
    TIME,
    URL,
    USERNAME,
    VEHICLE_REGISTRATION,
    ZIPCODE,
    normalize_label,
)
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import EvalSpan

from ._dua import (
    fixture_id,
    load_json_rows,
    require_credentialed_path,
    source_path_hash,
)
from .dua_stubs import DUACredentialRequired
from .licenses import license_for

CEGS_NGRID = "cegs-ngrid"
CEGS_NGRID_PATH_ENV = "OPENMED_CEGS_NGRID_PATH"
CEGS_NGRID_AUTHORITY = "DBMI"
CEGS_NGRID_DUA = "DBMI / CEGS N-GRID access terms"
CEGS_NGRID_TASK = "deid_ner"

CEGS_NGRID_LABEL_TO_CANONICAL: Mapping[str, str] = {
    "account": ID_NUM,
    "account_number": ID_NUM,
    "age": AGE,
    "city": LOCATION,
    "contact": OTHER,
    "contact_email": EMAIL,
    "contact_fax": PHONE,
    "contact_ipaddress": IP_ADDRESS,
    "contact_phone": PHONE,
    "contact_url": URL,
    "country": LOCATION,
    "date": DATE,
    "date_of_birth": DATE,
    "department": ORGANIZATION,
    "device": ID_NUM,
    "doctor": PERSON,
    "email": EMAIL,
    "fax": PHONE,
    "healthplan": ID_NUM,
    "health_plan": ID_NUM,
    "hospital": ORGANIZATION,
    "id": ID_NUM,
    "id_ssn": SSN,
    "idnum": ID_NUM,
    "id_num": ID_NUM,
    "ipaddress": IP_ADDRESS,
    "ip_address": IP_ADDRESS,
    "license": ID_NUM,
    "location": LOCATION,
    "location_city": LOCATION,
    "location_country": LOCATION,
    "location_department": ORGANIZATION,
    "location_hospital": ORGANIZATION,
    "location_location_other": LOCATION,
    "location_organization": ORGANIZATION,
    "location_room": LOCATION,
    "location_state": LOCATION,
    "location_street": STREET_ADDRESS,
    "location_zip": ZIPCODE,
    "medicalrecord": ID_NUM,
    "medical_record": ID_NUM,
    "medical_record_number": ID_NUM,
    "mrn": ID_NUM,
    "name": PERSON,
    "name_doctor": PERSON,
    "name_patient": PERSON,
    "name_username": USERNAME,
    "organization": ORGANIZATION,
    "patient": PERSON,
    "patient_name": PERSON,
    "person": PERSON,
    "phone": PHONE,
    "profession": OCCUPATION,
    "phi": OTHER,
    "room": LOCATION,
    "ssn": SSN,
    "state": LOCATION,
    "street": STREET_ADDRESS,
    "street_address": STREET_ADDRESS,
    "time": TIME,
    "url": URL,
    "username": USERNAME,
    "vehicle": VEHICLE_REGISTRATION,
    "vehicle_id": VEHICLE_REGISTRATION,
    "zip": ZIPCODE,
    "zipcode": ZIPCODE,
    "zip_code": ZIPCODE,
}

_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})
_XML_SUFFIXES = frozenset({".xml"})
_REPO_LABEL_PREFIX_RE = re.compile(r"^(?:B|I|E|S|U)[-_]", re.IGNORECASE)


def load_cegs_ngrid(path: str | Path | None = None) -> list[BenchmarkFixture]:
    """Load CEGS N-GRID rows from an authorized local export.

    JSON/JSONL rows, i2b2-style XML, and paired BRAT files are accepted. No
    default path is used: callers must provide a credentialed path or set
    ``OPENMED_CEGS_NGRID_PATH``.
    """

    root = require_credentialed_path(
        path,
        dataset=CEGS_NGRID,
        authority=CEGS_NGRID_AUTHORITY,
        env_var=CEGS_NGRID_PATH_ENV,
    )
    fixtures: list[BenchmarkFixture] = []
    json_files = _files(root, _JSON_SUFFIXES)
    xml_files = _files(root, _XML_SUFFIXES)
    ann_files = _paired_annotation_files(root)

    for source in json_files:
        for row_number, row in enumerate(
            load_json_rows(
                source,
                dataset=CEGS_NGRID,
                authority=CEGS_NGRID_AUTHORITY,
            ),
            start=1,
        ):
            fixtures.append(
                _fixture_from_row(
                    row,
                    source=source,
                    root=root,
                    row_number=row_number,
                )
            )
    for source in xml_files:
        fixtures.append(_fixture_from_xml(source, root=root))
    for annotation_path in ann_files:
        text_path = annotation_path.with_suffix(".txt")
        if not text_path.exists():
            raise ValueError(
                f"CEGS N-GRID BRAT annotation {annotation_path.name} requires "
                "a paired .txt file"
            )
        fixtures.append(
            _fixture_from_brat(
                text_path,
                annotation_path,
                root=root,
            )
        )

    if not fixtures:
        raise DUACredentialRequired(
            f"{CEGS_NGRID_AUTHORITY} credentialed {CEGS_NGRID} path contains no "
            "supported fixtures; no corpus rows were loaded"
        )
    _validate_unique_fixture_ids(fixtures)
    return fixtures


load_cegs_ngrid_fixtures = load_cegs_ngrid


def map_cegs_ngrid_label(label: str) -> str:
    """Map one CEGS N-GRID PHI label to ``CANONICAL_LABELS``."""

    key = _mapping_key(label)
    canonical = CEGS_NGRID_LABEL_TO_CANONICAL.get(key)
    if canonical is None:
        normalized = normalize_label(str(label).strip())
        if normalized in CANONICAL_LABELS:
            canonical = normalized
    if canonical is None:
        allowed = ", ".join(sorted(CEGS_NGRID_LABEL_TO_CANONICAL))
        raise ValueError(
            f"unknown CEGS N-GRID PHI label {label!r}; expected one of: {allowed}"
        )
    if canonical not in CANONICAL_LABELS:
        raise RuntimeError(f"CEGS N-GRID mapping is not canonical: {canonical!r}")
    return canonical


def cegs_ngrid_suite_metadata() -> dict[str, Any]:
    """Return row-free metadata for the CEGS N-GRID eval view."""

    return {
        "access": (
            f"credentialed local path only; pass path=... or set {CEGS_NGRID_PATH_ENV}"
        ),
        "dataset": CEGS_NGRID,
        "dua": CEGS_NGRID_DUA,
        "eval_only": True,
        "label_mapping": dict(sorted(CEGS_NGRID_LABEL_TO_CANONICAL.items())),
        "license": license_for(CEGS_NGRID).to_dict(),
        "network_fetch": False,
        "redistribution": "never; read-only from user-supplied credentialed path",
        "suite": CEGS_NGRID,
        "task": "ner",
        "task_view": CEGS_NGRID_TASK,
    }


def _fixture_from_row(
    row: Mapping[str, Any],
    *,
    source: Path,
    root: Path,
    row_number: int,
) -> BenchmarkFixture:
    text = _row_text(row)
    record_id = _record_id(row, fallback=f"row-{row_number}")
    raw_spans = row.get("entities") or row.get("spans") or row.get("annotations")
    spans = tuple(
        _span_from_mapping(item, text=text, source_name=source.name)
        for item in _span_rows(raw_spans)
    )
    return BenchmarkFixture(
        fixture_id=fixture_id(CEGS_NGRID, source, root, record_id),
        text=text,
        gold_spans=spans,
        language=str(row.get("language") or row.get("lang") or "en"),
        metadata=_metadata(source, root, suite=CEGS_NGRID),
    )


def _fixture_from_brat(
    text_path: Path,
    annotation_path: Path,
    *,
    root: Path,
) -> BenchmarkFixture:
    with text_path.open("r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    spans: list[EvalSpan] = []
    for line_number, line in enumerate(
        annotation_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.startswith("T"):
            continue
        columns = line.split("\t", 2)
        if len(columns) != 3:
            raise ValueError(
                f"malformed CEGS N-GRID BRAT entity at {annotation_path.name}:"
                f"{line_number}"
            )
        label, offsets = _brat_offsets(columns[1], text=text)
        spans.append(
            _span_from_values(
                start=offsets[0],
                end=offsets[1],
                label=label,
                text=text,
                supplied_text=columns[2],
                source_name=annotation_path.name,
            )
        )
    return BenchmarkFixture(
        fixture_id=fixture_id(CEGS_NGRID, text_path, root, text_path.stem),
        text=text,
        gold_spans=tuple(spans),
        metadata=_metadata(text_path, root, suite=CEGS_NGRID),
    )


def _fixture_from_xml(path: Path, *, root: Path) -> BenchmarkFixture:
    try:
        document = ET.parse(path)
    except ET.ParseError as exc:
        raise ValueError(f"failed to parse CEGS N-GRID XML {path.name}: {exc}") from exc
    xml_root = document.getroot()
    text_element = next(
        (
            element
            for element in xml_root.iter()
            if _local_name(element.tag).upper() in {"TEXT", "TEXTBODY", "NOTE"}
        ),
        None,
    )
    if text_element is None:
        raise ValueError(f"CEGS N-GRID XML {path.name} is missing a TEXT element")
    text = "".join(text_element.itertext())
    spans: list[EvalSpan] = []
    for element in xml_root.iter():
        local_name = _local_name(element.tag).upper()
        if local_name in {"TEXT", "TEXTBODY", "NOTE", "TAGS", "DOCUMENT"}:
            continue
        attributes = {
            str(key).casefold(): value for key, value in element.attrib.items()
        }
        if not _has_offsets(attributes):
            continue
        label = (
            attributes.get("type")
            or attributes.get("label")
            or attributes.get("category")
            or local_name
        )
        start, end = _xml_offsets(attributes)
        spans.append(
            _span_from_values(
                start=start,
                end=end,
                label=str(label),
                text=text,
                supplied_text=attributes.get("text") or "",
                source_name=path.name,
            )
        )
    return BenchmarkFixture(
        fixture_id=fixture_id(CEGS_NGRID, path, root, path.stem),
        text=text,
        gold_spans=tuple(spans),
        metadata=_metadata(path, root, suite=CEGS_NGRID),
    )


def _span_from_mapping(
    item: Any,
    *,
    text: str,
    source_name: str,
) -> EvalSpan:
    if not isinstance(item, Mapping):
        raise ValueError(f"CEGS N-GRID span in {source_name} must be an object")
    start, end = _mapping_offsets(item)
    return _span_from_values(
        start=start,
        end=end,
        label=str(
            item.get("label")
            or item.get("type")
            or item.get("entity_type")
            or item.get("tag")
            or ""
        ),
        text=text,
        supplied_text=str(item.get("text") or item.get("surface") or ""),
        source_name=source_name,
    )


def _span_from_values(
    *,
    start: int,
    end: int,
    label: str,
    text: str,
    supplied_text: str,
    source_name: str,
) -> EvalSpan:
    if start < 0 or end <= start or end > len(text):
        raise ValueError(
            f"invalid CEGS N-GRID span offsets {start}:{end} in {source_name}"
        )
    span_text = text[start:end]
    if supplied_text and supplied_text != span_text:
        raise ValueError(f"CEGS N-GRID span text mismatch in {source_name}")
    source_label = _REPO_LABEL_PREFIX_RE.sub("", label.strip())
    canonical = map_cegs_ngrid_label(source_label)
    return EvalSpan(
        start=start,
        end=end,
        label=canonical,
        text=span_text,
        metadata={"source_label": source_label},
    )


def _metadata(source: Path, root: Path, *, suite: str) -> dict[str, Any]:
    return {
        "cache_corpus_rows": False,
        "dataset": suite,
        "dua": CEGS_NGRID_DUA,
        "eval_only": True,
        "license": license_for(suite).to_dict(),
        "network_fetch": False,
        "redistribution": "never; read-only from user-supplied credentialed path",
        "source_path_hash": source_path_hash(source, root),
        "suite": suite,
        "task": "ner",
        "task_view": CEGS_NGRID_TASK,
    }


def _row_text(row: Mapping[str, Any]) -> str:
    for key in ("text", "note", "document"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("CEGS N-GRID rows require non-empty text")


def _record_id(row: Mapping[str, Any], *, fallback: str) -> str:
    for key in ("id", "record_id", "document_id", "doc_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return fallback


def _span_rows(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [
            {"id": key, **dict(item)}
            for key, item in value.items()
            if isinstance(item, Mapping)
        ]
    if not isinstance(value, list):
        raise ValueError("CEGS N-GRID spans must be a list or mapping")
    return [item for item in value if isinstance(item, Mapping)]


def _mapping_offsets(item: Mapping[str, Any]) -> tuple[int, int]:
    if item.get("start") is not None and item.get("end") is not None:
        return _integer(item["start"], "span start"), _integer(item["end"], "span end")
    if item.get("offset") is not None and item.get("length") is not None:
        start = _integer(item["offset"], "span offset")
        return start, start + _integer(item["length"], "span length")
    raise ValueError("CEGS N-GRID spans require start/end or offset/length")


def _brat_offsets(specification: str, *, text: str) -> tuple[str, tuple[int, int]]:
    values = specification.split(maxsplit=1)
    if len(values) != 2:
        raise ValueError("malformed CEGS N-GRID BRAT span offsets")
    label, offset_spec = values
    spans = []
    for segment in offset_spec.split(";"):
        offsets = segment.split()
        if len(offsets) != 2:
            raise ValueError("malformed CEGS N-GRID discontinuous BRAT span")
        spans.append(
            (_integer(offsets[0], "BRAT start"), _integer(offsets[1], "BRAT end"))
        )
    start = min(item[0] for item in spans)
    end = max(item[1] for item in spans)
    if end > len(text):
        raise ValueError("CEGS N-GRID BRAT span is outside document text")
    return label, (start, end)


def _xml_offsets(attributes: Mapping[str, Any]) -> tuple[int, int]:
    if attributes.get("start") is not None and attributes.get("end") is not None:
        return _integer(attributes["start"], "XML span start"), _integer(
            attributes["end"], "XML span end"
        )
    if attributes.get("begin") is not None and attributes.get("end") is not None:
        return _integer(attributes["begin"], "XML span begin"), _integer(
            attributes["end"], "XML span end"
        )
    if attributes.get("offset") is not None and attributes.get("length") is not None:
        start = _integer(attributes["offset"], "XML span offset")
        return start, start + _integer(attributes["length"], "XML span length")
    raise ValueError("CEGS N-GRID XML span requires offsets")


def _has_offsets(attributes: Mapping[str, Any]) -> bool:
    return (
        (attributes.get("start") is not None and attributes.get("end") is not None)
        or (attributes.get("begin") is not None and attributes.get("end") is not None)
        or (
            attributes.get("offset") is not None
            and attributes.get("length") is not None
        )
    )


def _files(root: Path, suffixes: set[str] | frozenset[str]) -> tuple[Path, ...]:
    wanted = {suffix.casefold() for suffix in suffixes}
    if root.is_file():
        return (root,) if root.suffix.casefold() in wanted else tuple()
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.casefold() in wanted
    )


def _paired_annotation_files(root: Path) -> tuple[Path, ...]:
    if root.is_file() and root.suffix.casefold() == ".txt":
        annotation_path = root.with_suffix(".ann")
        return (annotation_path,) if annotation_path.exists() else tuple()
    return _files(root, {".ann"})


def _mapping_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _local_name(value: str) -> str:
    return value.rsplit("}", maxsplit=1)[-1]


def _validate_unique_fixture_ids(fixtures: list[BenchmarkFixture]) -> None:
    ids = [fixture.fixture_id for fixture in fixtures]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {CEGS_NGRID} fixture ids")


__all__ = [
    "CEGS_NGRID",
    "CEGS_NGRID_AUTHORITY",
    "CEGS_NGRID_DUA",
    "CEGS_NGRID_LABEL_TO_CANONICAL",
    "CEGS_NGRID_PATH_ENV",
    "CEGS_NGRID_TASK",
    "cegs_ngrid_suite_metadata",
    "load_cegs_ngrid",
    "load_cegs_ngrid_fixtures",
    "map_cegs_ngrid_label",
]
