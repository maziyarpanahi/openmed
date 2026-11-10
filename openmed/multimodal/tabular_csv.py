"""CSV/TSV PHI column detection and column-scoped redaction.

The tabular path treats columns as the unit of policy. Header names and sampled
cell values are mapped to the canonical label taxonomy in
``openmed.core.labels``; the resulting policy class drives a per-column action
such as masking direct identifiers, hashing record IDs, or date-shifting
quasi-identifier dates. The emitted manifest intentionally contains column
metadata and counts only, never raw cell values.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.core.labels import (
    ACCOUNT_NUMBER,
    API_KEY,
    DATE,
    DATE_OF_BIRTH,
    DIRECT_IDENTIFIER,
    EMAIL,
    ID_NUM,
    IP_ADDRESS,
    LABEL_METADATA,
    MAC_ADDRESS,
    OTHER,
    PASSWORD,
    PERSON,
    PHONE,
    PIN,
    QUASI_IDENTIFIER,
    SSN,
    STREET_ADDRESS,
    USERNAME,
    normalize_label,
)

from .base import ExtractedDocument, register_handler

DIRECT_ID = "DIRECT_ID"
QUASI_ID = "QUASI_ID"
SAFE = "SAFE"

ACTION_DROP = "drop"
ACTION_HASH = "hash"
ACTION_MASK = "mask"
ACTION_DATE_SHIFT = "date_shift"
ACTION_FREE_TEXT_REDACT = "free_text_redact"
ACTION_KEEP = "keep"

SUPPORTED_ACTIONS = frozenset(
    {
        ACTION_DROP,
        ACTION_HASH,
        ACTION_MASK,
        ACTION_DATE_SHIFT,
        ACTION_FREE_TEXT_REDACT,
        ACTION_KEEP,
    }
)

TextRedactor = Callable[[str], str]


@dataclass(frozen=True)
class ColumnDecision:
    """Classification and redaction action for one source column."""

    index: int
    name: str
    assigned_class: str
    action: str
    canonical_label: str | None = None
    policy_label: str | None = None
    detection_source: str = "default"
    confidence: float = 0.0
    sampled_values: int = 0

    def to_manifest(self, *, row_count: int, row_count_affected: int) -> dict[str, Any]:
        """Return a PHI-safe manifest row for audit/reporting."""
        return {
            "column_index": self.index,
            "column_name": self.name,
            "assigned_class": self.assigned_class,
            "canonical_label": self.canonical_label,
            "policy_label": self.policy_label,
            "action": self.action,
            "detection_source": self.detection_source,
            "confidence": round(float(self.confidence), 6),
            "sampled_values": self.sampled_values,
            "row_count": row_count,
            "row_count_affected": row_count_affected,
        }


@dataclass(frozen=True)
class TableView:
    """Parsed delimited table plus column classifications."""

    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    delimiter: str
    has_header: bool
    columns: tuple[ColumnDecision, ...]


@dataclass(frozen=True)
class RedactedTable:
    """Column-redacted table output and its per-column manifest."""

    text: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    delimiter: str
    has_header: bool
    manifest: tuple[dict[str, Any], ...]
    columns: tuple[ColumnDecision, ...]

    def to_document(self) -> ExtractedDocument:
        """Bridge tabular output into the multimodal document contract."""
        return ExtractedDocument(
            text=self.text,
            metadata={
                "format": "csv" if self.delimiter == "," else "tsv",
                "delimiter": self.delimiter,
                "has_header": self.has_header,
                "row_count": len(self.rows),
                "redaction_manifest": list(self.manifest),
            },
        )


_HEADER_LABELS = {
    "name": PERSON,
    "fullname": PERSON,
    "patient": PERSON,
    "patientname": PERSON,
    "membername": PERSON,
    "subscribername": PERSON,
    "subjectname": PERSON,
    "mrn": ID_NUM,
    "medicalrecordnumber": ID_NUM,
    "medrecnumber": ID_NUM,
    "patientid": ID_NUM,
    "memberid": ID_NUM,
    "recordid": ID_NUM,
    "subjectid": ID_NUM,
    "identifier": ID_NUM,
    "ssn": SSN,
    "socialsecuritynumber": SSN,
    "email": EMAIL,
    "emailaddress": EMAIL,
    "phone": PHONE,
    "telephone": PHONE,
    "phonenumber": PHONE,
    "mobile": PHONE,
    "address": STREET_ADDRESS,
    "streetaddress": STREET_ADDRESS,
    "dob": DATE_OF_BIRTH,
    "dateofbirth": DATE_OF_BIRTH,
    "birthdate": DATE_OF_BIRTH,
    "admitdate": DATE,
    "admissiondate": DATE,
    "dischargedate": DATE,
    "encounterdate": DATE,
    "servicedate": DATE,
    "visitdate": DATE,
    "appointmentdate": DATE,
    "eventdate": DATE,
    "date": DATE,
    "username": USERNAME,
    "accountnumber": ACCOUNT_NUMBER,
    "password": PASSWORD,
    "pin": PIN,
    "apikey": API_KEY,
    "ipaddress": IP_ADDRESS,
    "macaddress": MAC_ADDRESS,
}

_NOTE_HEADER_KEYS = frozenset(
    {
        "note",
        "notes",
        "clinicalnote",
        "clinicalnotes",
        "comment",
        "comments",
        "description",
        "narrative",
        "freetext",
        "text",
    }
)

_HASH_LABELS = frozenset({ID_NUM, ACCOUNT_NUMBER, USERNAME})
_DATE_SHIFT_LABELS = frozenset({DATE})

_SSN_RE = re.compile(r"^\s*\d{3}-\d{2}-\d{4}\s*$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(
    r"^\s*(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\s*$"
)
_MRN_RE = re.compile(r"^\s*(?:MRN|MEDREC|MR)[\s:._-]*[A-Z0-9-]{3,}\s*$", re.I)
_DATE_RE = re.compile(
    r"^\s*(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*$"
)
_PERSON_NAME_RE = re.compile(
    r"^\s*[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?\s+"
    r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?\s*$"
)


def read_table(
    source: str | os.PathLike[str] | Any,
    *,
    delimiter: str | None = None,
    has_header: bool | None = None,
    header_heuristics: Mapping[str, str] | None = None,
    action_overrides: Mapping[str, str] | None = None,
    sample_size: int = 50,
) -> TableView:
    """Parse CSV/TSV input and classify each column.

    Args:
        source: A filesystem path, text content, or text file-like object.
        delimiter: Optional delimiter override. By default comma/tab sniffing is
            used.
        has_header: Optional header override. By default ``csv.Sniffer`` plus
            OpenMed header heuristics are used.
        header_heuristics: Extra ``header -> canonical label`` mappings. Header
            keys are normalized by stripping non-alphanumerics and lowercasing.
        action_overrides: Optional ``header/canonical/class -> action`` mapping.
        sample_size: Number of non-empty cell values sampled per column.

    Returns:
        Parsed table view with per-column decisions attached.
    """
    text = _read_text(source)
    resolved_delimiter = _sniff_delimiter(text, delimiter)
    records = _read_records(text, resolved_delimiter)
    width = max((len(row) for row in records), default=0)
    padded = tuple(_pad_row(row, width) for row in records)

    if width == 0:
        return TableView((), (), resolved_delimiter, False, ())

    resolved_has_header = _resolve_has_header(
        text,
        padded,
        resolved_delimiter,
        has_header=has_header,
        header_heuristics=header_heuristics,
    )
    if resolved_has_header:
        headers = _normalize_headers(padded[0])
        rows = padded[1:]
    else:
        headers = tuple(f"column_{index + 1}" for index in range(width))
        rows = padded

    columns = classify_columns(
        headers,
        rows,
        header_heuristics=header_heuristics,
        action_overrides=action_overrides,
        sample_size=sample_size,
    )
    return TableView(headers, rows, resolved_delimiter, resolved_has_header, columns)


def classify_columns(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    header_heuristics: Mapping[str, str] | None = None,
    action_overrides: Mapping[str, str] | None = None,
    sample_size: int = 50,
) -> tuple[ColumnDecision, ...]:
    """Classify columns as direct identifier, quasi-identifier, or safe."""
    decisions: list[ColumnDecision] = []
    for index, header in enumerate(headers):
        values = _column_values(rows, index, sample_size)
        decision = _classify_column(
            index,
            header,
            values,
            header_heuristics=header_heuristics,
            action_overrides=action_overrides,
        )
        decisions.append(decision)
    return tuple(decisions)


def redact_table(
    source: str | os.PathLike[str] | Any,
    *,
    policy: Any | None = None,
    models: Any | None = None,
    delimiter: str | None = None,
    has_header: bool | None = None,
    header_heuristics: Mapping[str, str] | None = None,
    action_overrides: Mapping[str, str] | None = None,
    date_shift_days: int | None = None,
    date_shift_seed: str = "openmed-tabular-csv-v1",
    keep_year: bool = True,
    lang: str = "en",
    sample_size: int = 50,
    text_redactor: TextRedactor | None = None,
) -> RedactedTable:
    """Redact a CSV/TSV table using column-scoped policy decisions.

    Args:
        source: A filesystem path, text content, or text file-like object.
        policy: Reserved for future policy objects; mapping values may include
            ``action_overrides`` and ``header_heuristics``.
        models: Optional text-redaction callable or mapping with a
            ``text_redactor`` callable for note-like columns.
        delimiter: Optional delimiter override.
        has_header: Optional header override.
        header_heuristics: Extra ``header -> canonical label`` mappings.
        action_overrides: Optional ``header/canonical/class -> action`` mapping.
        date_shift_days: Fixed date shift. When omitted, a deterministic
            per-record non-zero shift is derived and reused across date columns.
        date_shift_seed: Seed material for derived per-record shifts.
        keep_year: Preserve source year when shifting dates.
        lang: Language hint for OpenMed date parsing.
        sample_size: Number of non-empty cell values sampled per column.
        text_redactor: Optional callable used for note-like free-text columns.

    Returns:
        Redacted table text, parsed rows, decisions, and a PHI-safe manifest.
    """
    merged_header_heuristics, merged_action_overrides = _merge_policy_options(
        policy,
        header_heuristics=header_heuristics,
        action_overrides=action_overrides,
    )
    table = read_table(
        source,
        delimiter=delimiter,
        has_header=has_header,
        header_heuristics=merged_header_heuristics,
        action_overrides=merged_action_overrides,
        sample_size=sample_size,
    )
    resolved_redactor = text_redactor or _text_redactor_from_models(models)

    affected = {decision.index: 0 for decision in table.columns}
    output_indices = [
        decision.index for decision in table.columns if decision.action != ACTION_DROP
    ]
    redacted_rows: list[tuple[str, ...]] = []

    for row_index, row in enumerate(table.rows):
        row_shift_days: int | None = None
        redacted_row: list[str] = []
        for decision in table.columns:
            value = row[decision.index] if decision.index < len(row) else ""
            if decision.action == ACTION_DATE_SHIFT and row_shift_days is None:
                row_shift_days = derive_date_shift_days(
                    row,
                    record_index=row_index,
                    fixed_days=date_shift_days,
                    seed=date_shift_seed,
                )
            redacted_value, changed = _redact_cell(
                value,
                decision,
                date_shift_days=row_shift_days,
                keep_year=keep_year,
                lang=lang,
                text_redactor=resolved_redactor,
            )
            if changed:
                affected[decision.index] += 1
            if decision.index in output_indices:
                redacted_row.append(redacted_value)
        redacted_rows.append(tuple(redacted_row))

    output_headers = tuple(table.headers[index] for index in output_indices)
    output_text = _write_records(
        output_headers,
        redacted_rows,
        delimiter=table.delimiter,
        has_header=table.has_header,
    )
    manifest = tuple(
        decision.to_manifest(
            row_count=len(table.rows),
            row_count_affected=affected[decision.index],
        )
        for decision in table.columns
    )
    return RedactedTable(
        text=output_text,
        headers=output_headers,
        rows=tuple(redacted_rows),
        delimiter=table.delimiter,
        has_header=table.has_header,
        manifest=manifest,
        columns=table.columns,
    )


def _tabular_csv_handler(
    path: str | os.PathLike[str],
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    # ``lang`` is accepted for the shared handler contract; tabular text has no
    # OCR step, so language selection does not apply here.
    return redact_table(path, policy=policy, models=models).to_document()


def _read_text(source: str | os.PathLike[str] | Any) -> str:
    if hasattr(source, "read"):
        return str(source.read())
    if isinstance(source, os.PathLike):
        return _read_path_text(Path(source))
    if isinstance(source, str):
        if "\n" in source or "\r" in source:
            return source
        path = Path(source)
        if path.exists():
            return _read_path_text(path)
        return source
    raise TypeError("source must be a path, text content, or text file-like object")


def _read_path_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _sniff_delimiter(text: str, delimiter: str | None) -> str:
    if delimiter is not None:
        if delimiter not in {",", "\t"}:
            raise ValueError("delimiter must be ',' or '\\t'")
        return delimiter

    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        if dialect.delimiter in {",", "\t"}:
            return str(dialect.delimiter)
    except csv.Error:
        pass

    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    return "\t" if "\t" in first_line and "," not in first_line else ","


def _read_records(text: str, delimiter: str) -> tuple[tuple[str, ...], ...]:
    stream = io.StringIO(text, newline="")
    reader = csv.reader(stream, delimiter=delimiter)
    return tuple(tuple(cell for cell in row) for row in reader)


def _pad_row(row: Sequence[str], width: int) -> tuple[str, ...]:
    return tuple(row) + ("",) * max(0, width - len(row))


def _resolve_has_header(
    text: str,
    records: Sequence[Sequence[str]],
    delimiter: str,
    *,
    has_header: bool | None,
    header_heuristics: Mapping[str, str] | None,
) -> bool:
    if has_header is not None:
        return has_header
    if not records:
        return False

    first = records[0]
    header_keys = _merged_header_labels(header_heuristics)
    if any(_header_key(cell) in header_keys for cell in first):
        return True
    if any(_header_key(cell) in _NOTE_HEADER_KEYS for cell in first):
        return True

    try:
        return bool(csv.Sniffer().has_header(text[:8192]))
    except (TypeError, csv.Error):
        return _looks_like_header_row(records)


def _looks_like_header_row(records: Sequence[Sequence[str]]) -> bool:
    if len(records) < 2:
        return False
    first = records[0]
    second = records[1]
    if not first or len(first) != len(second):
        return False
    first_text = sum(1 for cell in first if cell and not _looks_like_scalar(cell))
    second_scalar = sum(1 for cell in second if _looks_like_scalar(cell))
    return first_text >= max(1, len(first) // 2) and second_scalar > 0


def _looks_like_scalar(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if _DATE_RE.fullmatch(stripped):
        return True
    try:
        float(stripped)
    except ValueError:
        return False
    return True


def _normalize_headers(headers: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, int] = {}
    normalized: list[str] = []
    for index, header in enumerate(headers):
        name = header.strip() or f"column_{index + 1}"
        count = seen.get(name, 0) + 1
        seen[name] = count
        normalized.append(name if count == 1 else f"{name}_{count}")
    return tuple(normalized)


def _column_values(
    rows: Sequence[Sequence[str]],
    index: int,
    sample_size: int,
) -> tuple[str, ...]:
    values: list[str] = []
    for row in rows:
        value = row[index].strip() if index < len(row) else ""
        if value:
            values.append(value)
        if len(values) >= sample_size:
            break
    return tuple(values)


def _classify_column(
    index: int,
    header: str,
    values: Sequence[str],
    *,
    header_heuristics: Mapping[str, str] | None,
    action_overrides: Mapping[str, str] | None,
) -> ColumnDecision:
    header_key = _header_key(header)
    label = _label_from_header(header, header_heuristics=header_heuristics)
    source = "header_name"
    confidence = 1.0

    if label is None:
        sampled = _label_from_values(values)
        if sampled is not None:
            label, confidence = sampled
            source = "value_sample"

    note_like = header_key in _NOTE_HEADER_KEYS
    if label is None:
        assigned_class = SAFE
        policy_label = None
        action = ACTION_FREE_TEXT_REDACT if note_like else ACTION_KEEP
        source = "header_name" if note_like else "default"
        confidence = 0.8 if note_like else 1.0
    else:
        assigned_class, policy_label = _class_from_label(label)
        action = _default_action(label, assigned_class)

    action = _apply_action_override(
        action,
        header=header,
        canonical_label=label,
        assigned_class=assigned_class,
        action_overrides=action_overrides,
    )
    return ColumnDecision(
        index=index,
        name=header,
        assigned_class=assigned_class,
        action=action,
        canonical_label=label,
        policy_label=policy_label,
        detection_source=source,
        confidence=confidence,
        sampled_values=len(values),
    )


def _merged_header_labels(
    header_heuristics: Mapping[str, str] | None,
) -> dict[str, str]:
    merged = dict(_HEADER_LABELS)
    for key, label in (header_heuristics or {}).items():
        canonical = normalize_label(str(label))
        if canonical != OTHER:
            merged[_header_key(str(key))] = canonical
    return merged


def _label_from_header(
    header: str,
    *,
    header_heuristics: Mapping[str, str] | None,
) -> str | None:
    key = _header_key(header)
    header_labels = _merged_header_labels(header_heuristics)
    if key in header_labels:
        return header_labels[key]

    canonical = normalize_label(header)
    if canonical != OTHER:
        return canonical
    return None


def _label_from_values(values: Sequence[str]) -> tuple[str, float] | None:
    if not values:
        return None

    checks = (
        (SSN, _SSN_RE),
        (EMAIL, _EMAIL_RE),
        (PHONE, _PHONE_RE),
        (ID_NUM, _MRN_RE),
        (DATE, _DATE_RE),
        (PERSON, _PERSON_NAME_RE),
    )
    best_label: str | None = None
    best_score = 0.0
    for label, pattern in checks:
        matches = sum(1 for value in values if pattern.fullmatch(value.strip()))
        score = matches / len(values)
        threshold = 0.8 if label == PERSON else 0.6
        if matches and score >= threshold and score > best_score:
            best_label = label
            best_score = score

    if best_label is None:
        return None
    return best_label, best_score


def _class_from_label(canonical_label: str) -> tuple[str, str | None]:
    metadata = LABEL_METADATA.get(canonical_label)
    policy_label = str(metadata["policy_label"]) if metadata else None
    if policy_label == DIRECT_IDENTIFIER:
        return DIRECT_ID, policy_label
    if policy_label == QUASI_IDENTIFIER:
        return QUASI_ID, policy_label
    return SAFE, policy_label


def _default_action(canonical_label: str, assigned_class: str) -> str:
    if canonical_label in _DATE_SHIFT_LABELS:
        return ACTION_DATE_SHIFT
    if canonical_label in _HASH_LABELS:
        return ACTION_HASH
    if assigned_class in {DIRECT_ID, QUASI_ID}:
        return ACTION_MASK
    return ACTION_KEEP


def _apply_action_override(
    action: str,
    *,
    header: str,
    canonical_label: str | None,
    assigned_class: str,
    action_overrides: Mapping[str, str] | None,
) -> str:
    if not action_overrides:
        return action

    candidates = [
        header,
        _header_key(header),
        canonical_label or "",
        assigned_class,
    ]
    lowered = {str(key).lower(): value for key, value in action_overrides.items()}
    for candidate in candidates:
        if not candidate:
            continue
        override = lowered.get(str(candidate).lower())
        if override is not None:
            normalized = str(override)
            if normalized not in SUPPORTED_ACTIONS:
                raise ValueError(f"unsupported tabular redaction action: {override!r}")
            return normalized
    return action


def _merge_policy_options(
    policy: Any | None,
    *,
    header_heuristics: Mapping[str, str] | None,
    action_overrides: Mapping[str, str] | None,
) -> tuple[Mapping[str, str] | None, Mapping[str, str] | None]:
    if not isinstance(policy, Mapping):
        return header_heuristics, action_overrides

    policy_headers = policy.get("header_heuristics")
    policy_actions = policy.get("action_overrides")
    merged_headers = _merge_mapping(header_heuristics, policy_headers)
    merged_actions = _merge_mapping(action_overrides, policy_actions)
    return merged_headers, merged_actions


def _merge_mapping(
    explicit: Mapping[str, str] | None,
    policy_value: Any,
) -> Mapping[str, str] | None:
    merged: dict[str, str] = {}
    if isinstance(policy_value, Mapping):
        merged.update({str(key): str(value) for key, value in policy_value.items()})
    if explicit:
        merged.update({str(key): str(value) for key, value in explicit.items()})
    return merged or None


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


def _redact_cell(
    value: str,
    decision: ColumnDecision,
    *,
    date_shift_days: int | None,
    keep_year: bool,
    lang: str,
    text_redactor: TextRedactor | None,
) -> tuple[str, bool]:
    if decision.action == ACTION_KEEP:
        return value, False
    if decision.action == ACTION_DROP:
        return "", bool(value)
    if not value:
        return value, False
    if decision.action == ACTION_FREE_TEXT_REDACT:
        redacted = _redact_free_text(value, text_redactor=text_redactor)
        return redacted, redacted != value

    label = decision.canonical_label or "PHI"
    method = "shift_dates" if decision.action == ACTION_DATE_SHIFT else decision.action
    redacted = _redact_with_core(
        value,
        label=label,
        method=method,
        date_shift_days=date_shift_days,
        keep_year=keep_year,
        lang=lang,
    )
    return redacted, redacted != value


def _redact_free_text(
    value: str,
    *,
    text_redactor: TextRedactor | None,
) -> str:
    if text_redactor is not None:
        return text_redactor(value)

    from openmed.core.pii import deidentify

    return deidentify(value, method="mask").deidentified_text


def _redact_with_core(
    value: str,
    *,
    label: str,
    method: str,
    date_shift_days: int | None,
    keep_year: bool,
    lang: str,
) -> str:
    from openmed.core.pii import PIIEntity, _redact_entity

    entity = PIIEntity(
        text=value,
        label=label,
        entity_type=label,
        start=0,
        end=len(value),
        confidence=1.0,
        original_text=value,
    )
    return _redact_entity(
        entity,
        method,  # type: ignore[arg-type]
        keep_year=keep_year,
        date_shift_days=date_shift_days,
        lang=lang,
    )


def derive_date_shift_days(
    values: Sequence[str],
    *,
    record_index: int,
    fixed_days: int | None = None,
    seed: str = "openmed-tabular-csv-v1",
) -> int:
    """Return a deterministic non-zero date shift for one logical record.

    The helper is shared by structured exporters so date quasi-identifiers use
    the same per-record policy as CSV/TSV redaction. Source values are hashed in
    memory and are never returned or persisted.

    Args:
        values: Stable string representation of the record fields.
        record_index: Zero-based position used to distinguish duplicate rows.
        fixed_days: Optional explicit non-zero shift.
        seed: Caller-controlled deterministic seed material.

    Returns:
        A non-zero offset between -365 and 364 days when no fixed shift is set.
    """
    if fixed_days is not None:
        if fixed_days == 0:
            raise ValueError("date_shift_days must be non-zero")
        return fixed_days

    material = "\x1f".join([seed, str(record_index), *values]).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    shift = int.from_bytes(digest[:2], "big") % 730 - 365
    return shift if shift != 0 else 1


def shift_quasi_identifier_date(
    value: str,
    *,
    shift_days: int,
    keep_year: bool = True,
    lang: str = "en",
) -> str:
    """Shift one date value with the tabular quasi-identifier policy.

    Args:
        value: Date text accepted by OpenMed's date redactor.
        shift_days: Required non-zero day offset.
        keep_year: Preserve the source year after shifting.
        lang: OpenMed language hint for localized date parsing.

    Returns:
        Shifted date text in the source format where supported.
    """
    if shift_days == 0:
        raise ValueError("shift_days must be non-zero")
    return _redact_with_core(
        value,
        label=DATE,
        method="shift_dates",
        date_shift_days=shift_days,
        keep_year=keep_year,
        lang=lang,
    )


def _write_records(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    delimiter: str,
    has_header: bool,
) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
    if has_header:
        writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue()


def _header_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


register_handler((".csv", ".tsv"), _tabular_csv_handler, requires_multimodal=False)


__all__ = [
    "ACTION_DATE_SHIFT",
    "ACTION_DROP",
    "ACTION_FREE_TEXT_REDACT",
    "ACTION_HASH",
    "ACTION_KEEP",
    "ACTION_MASK",
    "DIRECT_ID",
    "QUASI_ID",
    "SAFE",
    "ColumnDecision",
    "RedactedTable",
    "TableView",
    "classify_columns",
    "derive_date_shift_days",
    "read_table",
    "redact_table",
    "shift_quasi_identifier_date",
]
