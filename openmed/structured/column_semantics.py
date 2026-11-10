"""Local, review-first semantic classification for clinical table columns.

The classifier combines schema names, shared identifier validators, value
shapes, and aggregate distribution cues. It returns an editable policy
manifest and never applies that policy; callers must review it and invoke a
separate application API.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from openmed.core.anonymizer.providers.clinical_ids import (
    validate_mrn,
    validate_phone_us,
    validate_ssn,
    validate_uk_nhs_number,
)
from openmed.core.labels import canonical_label_for_column_semantic

from .qi_detect import DEFAULT_BATCH_SIZE, DEFAULT_MAX_ROWS, _read_table_sample

AUTO_POLICY_SCHEMA_VERSION: Final = "1.0"
DEFAULT_CONFIDENCE_THRESHOLD: Final = 0.70

ACTION_ROUTE_TO_DEIDENTIFY: Final = "route-to-deidentify"
ACTION_GENERALIZE: Final = "generalize"
ACTION_SUPPRESS: Final = "suppress"
ACTION_KEEP: Final = "keep"
ACTION_MANUAL_REVIEW: Final = "manual-review"

ROLE_DIRECT_ID: Final = "direct-id"
ROLE_QUASI_ID: Final = "quasi-id"
ROLE_SENSITIVE: Final = "sensitive"
ROLE_SAFE: Final = "safe"
ROLE_FREE_TEXT: Final = "free-text"
ROLE_MANUAL_REVIEW: Final = "manual-review"

_DIRECT_SEMANTICS = frozenset(
    {
        "person_name",
        "medical_record_number",
        "nhs_number",
        "social_security_number",
        "record_identifier",
        "email_address",
        "phone_number",
        "street_address",
        "date_of_birth",
    }
)
_QUASI_SEMANTICS = frozenset(
    {
        "date",
        "age",
        "postal_code",
        "location",
        "gender",
        "organization",
        "clinical_code",
        "diagnosis_code",
        "procedure_code",
        "medication_code",
        "lab_code",
    }
)
_SENSITIVE_SEMANTICS = frozenset(
    {
        "clinical_condition",
        "medication",
        "procedure",
        "lab_test",
        "lab_value",
        "reference_range",
    }
)
_CODE_SEMANTICS = frozenset(
    {
        "clinical_code",
        "diagnosis_code",
        "procedure_code",
        "medication_code",
        "lab_code",
    }
)

_HEADER_SEMANTICS: Final[Mapping[str, tuple[str, float]]] = {
    "name": ("person_name", 0.94),
    "fullname": ("person_name", 0.97),
    "patientname": ("person_name", 0.98),
    "membername": ("person_name", 0.98),
    "firstname": ("person_name", 0.96),
    "givenname": ("person_name", 0.96),
    "lastname": ("person_name", 0.96),
    "familyname": ("person_name", 0.96),
    "surname": ("person_name", 0.96),
    "mrn": ("medical_record_number", 0.99),
    "medicalrecordnumber": ("medical_record_number", 0.99),
    "nhs": ("nhs_number", 0.99),
    "nhsnumber": ("nhs_number", 0.99),
    "ssn": ("social_security_number", 0.99),
    "socialsecuritynumber": ("social_security_number", 0.99),
    "patientid": ("record_identifier", 0.96),
    "memberid": ("record_identifier", 0.96),
    "recordid": ("record_identifier", 0.96),
    "subjectid": ("record_identifier", 0.96),
    "identifier": ("record_identifier", 0.90),
    "email": ("email_address", 0.98),
    "emailaddress": ("email_address", 0.98),
    "phone": ("phone_number", 0.97),
    "phonenumber": ("phone_number", 0.98),
    "telephone": ("phone_number", 0.97),
    "address": ("street_address", 0.92),
    "streetaddress": ("street_address", 0.98),
    "dob": ("date_of_birth", 0.99),
    "dateofbirth": ("date_of_birth", 0.99),
    "birthdate": ("date_of_birth", 0.99),
    "age": ("age", 0.98),
    "patientage": ("age", 0.98),
    "zip": ("postal_code", 0.97),
    "zipcode": ("postal_code", 0.98),
    "postalcode": ("postal_code", 0.98),
    "postcode": ("postal_code", 0.98),
    "city": ("location", 0.94),
    "county": ("location", 0.94),
    "state": ("location", 0.92),
    "region": ("location", 0.92),
    "country": ("location", 0.92),
    "location": ("location", 0.94),
    "gender": ("gender", 0.97),
    "sex": ("gender", 0.97),
    "facility": ("organization", 0.90),
    "hospital": ("organization", 0.94),
    "clinic": ("organization", 0.92),
    "provider": ("organization", 0.86),
    "organization": ("organization", 0.94),
    "admissiondate": ("date", 0.98),
    "admitdate": ("date", 0.98),
    "dischargedate": ("date", 0.98),
    "encounterdate": ("date", 0.98),
    "servicedate": ("date", 0.98),
    "visitdate": ("date", 0.98),
    "eventdate": ("date", 0.96),
    "date": ("date", 0.90),
    "diagnosiscode": ("diagnosis_code", 0.98),
    "dxcode": ("diagnosis_code", 0.98),
    "icd10": ("diagnosis_code", 0.98),
    "icd10code": ("diagnosis_code", 0.98),
    "procedurecode": ("procedure_code", 0.98),
    "cptcode": ("procedure_code", 0.98),
    "medicationcode": ("medication_code", 0.98),
    "drugcode": ("medication_code", 0.96),
    "rxnormcode": ("medication_code", 0.98),
    "ndccode": ("medication_code", 0.98),
    "labcode": ("lab_code", 0.98),
    "loinc": ("lab_code", 0.98),
    "loinccode": ("lab_code", 0.98),
    "loincode": ("lab_code", 0.98),
    "diagnosis": ("clinical_condition", 0.94),
    "condition": ("clinical_condition", 0.94),
    "disease": ("clinical_condition", 0.94),
    "medication": ("medication", 0.94),
    "drug": ("medication", 0.90),
    "procedure": ("procedure", 0.94),
    "labtest": ("lab_test", 0.94),
    "testname": ("lab_test", 0.90),
    "labvalue": ("lab_value", 0.96),
    "resultvalue": ("lab_value", 0.94),
    "measurement": ("lab_value", 0.88),
    "unit": ("unit", 0.94),
    "resultunit": ("unit", 0.94),
    "referencerange": ("reference_range", 0.96),
    "clinicalnote": ("free_text", 0.99),
    "note": ("free_text", 0.96),
    "notes": ("free_text", 0.96),
    "comment": ("free_text", 0.94),
    "comments": ("free_text", 0.94),
    "narrative": ("free_text", 0.97),
    "summary": ("free_text", 0.94),
    "dischargesummary": ("free_text", 0.99),
    "status": ("categorical", 0.82),
    "category": ("categorical", 0.82),
    "active": ("boolean", 0.84),
    "isactive": ("boolean", 0.88),
}

_LAB_VALUE_HEADERS = frozenset(
    {
        "bmi",
        "creatinine",
        "glucose",
        "heartrate",
        "height",
        "hemoglobin",
        "systolicbp",
        "temperature",
        "weight",
    }
)
_DATE_HEADER_TOKENS = ("date", "datetime", "timestamp", "time")
_FREE_TEXT_HEADER_TOKENS = (
    "description",
    "freeform",
    "freetext",
    "narrative",
    "note",
    "summary",
)

_SSN_SHAPE = re.compile(r"^\s*\d{3}[- ]\d{2}[- ]\d{4}\s*$")
_EMAIL_SHAPE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_SHAPE = re.compile(
    r"^\s*(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)"
    r"\d{3}[\s.-]?\d{4}\s*$"
)
_ICD_SHAPE = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?$", re.I)
_LOINC_SHAPE = re.compile(r"^\d{1,5}-\d$")
_POSTAL_SHAPE = re.compile(r"^(?:\d{5}(?:-\d{4})?|[A-Z]\d[A-Z]\s?\d[A-Z]\d)$", re.I)
_INTEGER_SHAPE = re.compile(r"^[+-]?\d+$")
_NUMBER_SHAPE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def classify_columns(
    path: str | Path,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_rows: int = DEFAULT_MAX_ROWS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Classify a local table and return an editable, unapplied auto-policy.

    Args:
        path: CSV, TSV, JSONL/NDJSON, or Parquet table path.
        confidence_threshold: Minimum confidence required for an automatic
            recommendation. Lower-confidence columns abstain to manual review.
        max_rows: Maximum number of rows sampled locally.
        batch_size: Maximum Parquet batch size.

    Returns:
        A mutable JSON-compatible policy containing a semantic type, canonical
        label, role, confidence, evidence, and recommended action per column.
        The artifact is marked pending review and is never auto-applied.
    """

    _validate_configuration(confidence_threshold, max_rows, batch_size)
    sample = _read_table_sample(
        Path(path),
        max_rows=max_rows,
        batch_size=batch_size,
        full_scan=False,
    )
    return classify_records(
        sample.rows,
        columns=sample.columns,
        confidence_threshold=confidence_threshold,
        source_format=sample.format,
        sample_complete=sample.complete,
        max_rows=max_rows,
    )


def classify_records(
    records: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    source_format: str = "records",
    sample_complete: bool = True,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Classify in-memory records using the same review-first policy schema."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    rows = tuple(records)
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError("records must contain mappings")
    resolved_columns = _resolve_columns(rows, columns)
    decisions = {
        column: _classify_column(
            column,
            [row.get(column) for row in rows],
            confidence_threshold=confidence_threshold,
        )
        for column in resolved_columns
    }
    abstained = [name for name, decision in decisions.items() if decision["abstained"]]
    action_counts = Counter(
        str(decision["recommended_action"]) for decision in decisions.values()
    )
    return {
        "schema_version": AUTO_POLICY_SCHEMA_VERSION,
        "kind": "openmed-column-auto-policy",
        "review_status": "pending",
        "review_required": True,
        "applied": False,
        "confidence_threshold": round(confidence_threshold, 6),
        "sample": {
            "format": source_format,
            "sampled_rows": len(rows),
            "max_rows": max_rows,
            "complete": bool(sample_complete),
        },
        "columns": decisions,
        "summary": {
            "column_count": len(resolved_columns),
            "abstained_column_count": len(abstained),
            "abstained_columns": abstained,
            "recommended_action_counts": dict(sorted(action_counts.items())),
        },
    }


def write_auto_policy(
    policy: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically persist a reviewable auto-policy as formatted JSON."""

    destination = Path(path)
    if (destination.exists() or destination.is_symlink()) and not overwrite:
        raise FileExistsError(f"Policy output already exists: {destination}")
    if not destination.parent.exists():
        raise FileNotFoundError(
            f"Policy output directory does not exist: {destination.parent}"
        )
    payload = json.dumps(policy, indent=2, ensure_ascii=False, sort_keys=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def _validate_configuration(
    confidence_threshold: float,
    max_rows: int,
    batch_size: int,
) -> None:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _resolve_columns(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None,
) -> tuple[str, ...]:
    if columns is not None:
        resolved = tuple(columns)
        if any(type(column) is not str or not column for column in resolved):
            raise TypeError("column names must be non-empty strings")
        if len(resolved) != len(set(resolved)):
            raise ValueError("column names must be unique")
        return resolved

    resolved_list: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("records must contain mappings")
        for column in row:
            if type(column) is not str or not column:
                raise TypeError("column names must be non-empty strings")
            if column not in seen:
                seen.add(column)
                resolved_list.append(column)
    if not resolved_list:
        raise ValueError("table must contain at least one column")
    return tuple(resolved_list)


def _classify_column(
    column: str,
    values: Sequence[Any],
    *,
    confidence_threshold: float,
) -> dict[str, Any]:
    rendered = [_render_value(value) for value in values]
    non_empty = [value for value in rendered if value]
    normalized_header = _name_key(column)
    candidates: dict[str, tuple[float, list[str]]] = {}

    header_candidate = _header_candidate(normalized_header)
    if header_candidate is not None:
        semantic_type, score = header_candidate
        _offer(candidates, semantic_type, score, "header-name")

    _offer_value_candidates(candidates, normalized_header, non_empty)
    distribution = _distribution_cues(values, non_empty)
    for semantic_type, score, evidence in distribution["candidates"]:
        _offer(candidates, semantic_type, score, evidence)

    if candidates:
        inferred_type, (confidence, evidence) = max(
            candidates.items(),
            key=lambda item: (item[1][0], _semantic_priority(item[0])),
        )
    else:
        inferred_type, confidence, evidence = "unknown", 0.0, ["no-signal"]

    abstained = confidence < confidence_threshold
    semantic_type = "unknown" if abstained else inferred_type
    canonical_label = canonical_label_for_column_semantic(semantic_type)
    column_role = ROLE_MANUAL_REVIEW if abstained else _role_for(semantic_type)
    recommended_action = (
        ACTION_MANUAL_REVIEW if abstained else _action_for(semantic_type)
    )
    evidence.extend(
        [
            f"non_null_count={len(non_empty)}",
            f"null_count={len(values) - len(non_empty)}",
            f"cardinality={distribution['cardinality']}",
            f"uniqueness_ratio={distribution['uniqueness_ratio']:.6f}",
        ]
    )
    return {
        "semantic_type": semantic_type,
        "inferred_semantic_type": inferred_type,
        "canonical_label": canonical_label,
        "column_role": column_role,
        "recommended_action": recommended_action,
        "confidence": round(confidence, 6),
        "abstained": abstained,
        "evidence": list(dict.fromkeys(evidence)),
        "profile": {
            "sampled_rows": len(values),
            "non_null_count": len(non_empty),
            "null_count": len(values) - len(non_empty),
            "cardinality": distribution["cardinality"],
            "uniqueness_ratio": round(distribution["uniqueness_ratio"], 6),
        },
    }


def _header_candidate(header: str) -> tuple[str, float] | None:
    exact = _HEADER_SEMANTICS.get(header)
    if exact is not None:
        return exact
    if header in _LAB_VALUE_HEADERS:
        return "lab_value", 0.90
    if any(token in header for token in _FREE_TEXT_HEADER_TOKENS):
        return "free_text", 0.91
    if header.endswith("id"):
        return "record_identifier", 0.84
    if header.endswith("code"):
        return "clinical_code", 0.58
    if any(token in header for token in _DATE_HEADER_TOKENS):
        return "date", 0.86
    return None


def _offer_value_candidates(
    candidates: dict[str, tuple[float, list[str]]],
    header: str,
    values: Sequence[str],
) -> None:
    if not values:
        return
    mrn_ratio = _match_ratio(values, validate_mrn)
    nhs_ratio = _match_ratio(values, validate_uk_nhs_number)
    ssn_ratio = _match_ratio(
        values,
        lambda value: bool(_SSN_SHAPE.fullmatch(value)) and validate_ssn(value),
    )
    email_ratio = _match_ratio(
        values, lambda value: bool(_EMAIL_SHAPE.fullmatch(value))
    )
    phone_ratio = _match_ratio(
        values,
        lambda value: bool(_PHONE_SHAPE.fullmatch(value)) and validate_phone_us(value),
    )
    date_ratio = _match_ratio(values, _is_date_text)
    postal_ratio = _match_ratio(
        values, lambda value: bool(_POSTAL_SHAPE.fullmatch(value))
    )
    code_ratio = _match_ratio(
        values,
        lambda value: bool(
            _ICD_SHAPE.fullmatch(value) or _LOINC_SHAPE.fullmatch(value)
        ),
    )

    for semantic_type, ratio, minimum, base, evidence in (
        ("medical_record_number", mrn_ratio, 0.60, 0.98, "validated-mrn-shape"),
        ("nhs_number", nhs_ratio, 0.70, 0.98, "validated-nhs-checksum"),
        ("social_security_number", ssn_ratio, 0.70, 0.98, "validated-ssn-shape"),
        ("email_address", email_ratio, 0.70, 0.96, "email-value-shape"),
        ("phone_number", phone_ratio, 0.70, 0.93, "validated-phone-shape"),
        ("date", date_ratio, 0.70, 0.90, "date-value-shape"),
        ("postal_code", postal_ratio, 0.80, 0.84, "postal-value-shape"),
        ("clinical_code", code_ratio, 0.70, 0.92, "clinical-code-value-shape"),
    ):
        if ratio >= minimum:
            score = min(0.995, base + 0.015 * ratio)
            _offer(candidates, semantic_type, score, f"{evidence}-ratio={ratio:.6f}")

    if header in {"code", "value", "col", "col1", "column"} and code_ratio:
        _offer(
            candidates,
            "clinical_code",
            min(0.96, 0.66 + code_ratio * 0.3),
            f"ambiguous-header-code-shape-ratio={code_ratio:.6f}",
        )
    if header == "value":
        decimal_ratio = _match_ratio(
            values,
            lambda value: bool(_NUMBER_SHAPE.fullmatch(value)) and "." in value,
        )
        if decimal_ratio >= 0.70:
            _offer(
                candidates,
                "lab_value",
                min(0.90, 0.75 + decimal_ratio * 0.15),
                f"ambiguous-header-decimal-value-ratio={decimal_ratio:.6f}",
            )


def _distribution_cues(
    original_values: Sequence[Any],
    values: Sequence[str],
) -> dict[str, Any]:
    counts = Counter(values)
    cardinality = len(counts)
    uniqueness_ratio = _rate(cardinality, len(values))
    candidates: list[tuple[str, float, str]] = []
    if not values:
        return {
            "cardinality": 0,
            "uniqueness_ratio": 0.0,
            "candidates": candidates,
        }

    boolean_ratio = _match_ratio(values, _is_boolean_text)
    numeric_ratio = _match_ratio(values, _is_number_text)
    average_length = sum(len(value) for value in values) / len(values)
    wordy_ratio = _match_ratio(
        values,
        lambda value: len(value.split()) >= 6 or len(value) >= 48,
    )

    if boolean_ratio >= 0.9 and cardinality <= 3:
        candidates.append(("boolean", 0.88, "boolean-distribution"))
    if wordy_ratio >= 0.6 or average_length >= 64:
        candidates.append(("free_text", 0.90, "long-text-distribution"))
    if numeric_ratio >= 0.9:
        is_integral = _match_ratio(
            values, lambda value: bool(_INTEGER_SHAPE.fullmatch(value))
        )
        semantic_type = "numeric"
        score = 0.72 if is_integral >= 0.9 else 0.78
        candidates.append((semantic_type, score, "numeric-distribution"))
    if (
        numeric_ratio < 0.9
        and cardinality <= max(3, min(12, math.ceil(math.sqrt(len(values)))))
        and uniqueness_ratio <= 0.5
    ):
        candidates.append(("categorical", 0.76, "low-cardinality-distribution"))

    typed_non_missing = [
        value
        for value in original_values
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]
    if typed_non_missing and all(
        isinstance(value, bool) for value in typed_non_missing
    ):
        candidates.append(("boolean", 0.96, "boolean-native-type"))
    return {
        "cardinality": cardinality,
        "uniqueness_ratio": uniqueness_ratio,
        "candidates": candidates,
    }


def _offer(
    candidates: dict[str, tuple[float, list[str]]],
    semantic_type: str,
    score: float,
    evidence: str,
) -> None:
    current = candidates.get(semantic_type)
    if current is None:
        candidates[semantic_type] = (score, [evidence])
        return
    current_score, current_evidence = current
    combined = min(0.995, max(score, current_score) + 0.025)
    candidates[semantic_type] = (combined, [*current_evidence, evidence])


def _semantic_priority(semantic_type: str) -> tuple[int, float]:
    if semantic_type in _DIRECT_SEMANTICS:
        return 4, 0.0
    if semantic_type in _CODE_SEMANTICS:
        return 3, 0.0
    if semantic_type in _QUASI_SEMANTICS:
        return 2, 0.0
    return 1, 0.0


def _role_for(semantic_type: str) -> str:
    if semantic_type in _DIRECT_SEMANTICS:
        return ROLE_DIRECT_ID
    if semantic_type in _QUASI_SEMANTICS:
        return ROLE_QUASI_ID
    if semantic_type == "free_text":
        return ROLE_FREE_TEXT
    if semantic_type in _SENSITIVE_SEMANTICS:
        return ROLE_SENSITIVE
    return ROLE_SAFE


def _action_for(semantic_type: str) -> str:
    if semantic_type == "free_text":
        return ACTION_ROUTE_TO_DEIDENTIFY
    if semantic_type in _DIRECT_SEMANTICS - {"date_of_birth"}:
        return ACTION_SUPPRESS
    if semantic_type in _QUASI_SEMANTICS or semantic_type == "date_of_birth":
        return ACTION_GENERALIZE
    return ACTION_KEEP


def _render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value.strip())
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return "" if not value.is_finite() else str(value)
    if isinstance(value, float):
        return "" if not math.isfinite(value) else repr(value)
    if isinstance(value, (bool, int)):
        return str(value)
    return ""


def _is_date_text(value: str) -> bool:
    for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            datetime.strptime(value, pattern)
        except ValueError:
            continue
        return True
    return False


def _is_boolean_text(value: str) -> bool:
    return value.casefold() in {"0", "1", "false", "no", "true", "yes"}


def _is_number_text(value: str) -> bool:
    return bool(_NUMBER_SHAPE.fullmatch(value))


def _match_ratio(values: Sequence[str], predicate: Any) -> float:
    matches = 0
    for value in values:
        try:
            matches += int(bool(predicate(value)))
        except (TypeError, ValueError):
            continue
    return _rate(matches, len(values))


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


__all__ = [
    "ACTION_GENERALIZE",
    "ACTION_KEEP",
    "ACTION_MANUAL_REVIEW",
    "ACTION_ROUTE_TO_DEIDENTIFY",
    "ACTION_SUPPRESS",
    "AUTO_POLICY_SCHEMA_VERSION",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "ROLE_DIRECT_ID",
    "ROLE_FREE_TEXT",
    "ROLE_MANUAL_REVIEW",
    "ROLE_QUASI_ID",
    "ROLE_SAFE",
    "ROLE_SENSITIVE",
    "classify_columns",
    "classify_records",
    "write_auto_policy",
]
