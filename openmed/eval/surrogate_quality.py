"""Offline surrogate-quality evaluation for multilingual PHI replacements."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

from openmed.core.anonymizer.providers import clinical_ids
from openmed.core.locale_formats import format_hint, parse_date
from openmed.core.pii_i18n import (
    validate_aadhaar,
    validate_chinese_resident_identity_card,
    validate_french_nir,
    validate_german_steuer_id,
    validate_spanish_dni,
    validate_spanish_nie,
)

DEFAULT_SURROGATE_QUALITY_FIXTURE: Final = (
    Path(__file__).resolve().parent
    / "golden"
    / "fixtures"
    / "surrogate_multilingual.jsonl"
)
DEFAULT_SURROGATE_QUALITY_LOCALES: Final = ("en", "fr", "de", "es", "hi", "zh")
DEFAULT_SURROGATE_QUALITY_PASS_RATE: Final = 0.90
SURROGATE_QUALITY_DIMENSIONS: Final = (
    "script",
    "format",
    "checksum",
    "consistency",
    "naturalness",
)

_SCRIPT_BY_LANGUAGE: Final = {
    "en": "latin",
    "fr": "latin",
    "de": "latin",
    "es": "latin",
    "hi": "devanagari",
    "zh": "han",
}
_MIN_SCRIPT_RATIO: Final = {
    "latin": 0.80,
    "devanagari": 0.70,
    "han": 0.70,
}
_NATIONAL_ID_VALIDATORS: Final[Mapping[str, tuple[Callable[[str], bool], ...]]] = {
    "en": (clinical_ids.validate_ssn,),
    "fr": (validate_french_nir,),
    "de": (validate_german_steuer_id,),
    "es": (validate_spanish_dni, validate_spanish_nie),
    "hi": (validate_aadhaar,),
    "zh": (validate_chinese_resident_identity_card,),
}
_PLACEHOLDER_RE = re.compile(
    r"\b(?:"
    r"anon|anonymous|dummy|example|fake|john\s+doe|jane\s+doe|patient|"
    r"placeholder|sample|test|unknown|xxx+"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SurrogateQualityRecord:
    """One synthetic surrogate set for a locale-specific document identity."""

    record_id: str
    language: str
    locale: str
    surrogates: Mapping[str, str]
    expected: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SurrogateQualityRecord":
        """Build a record from a JSON-compatible mapping."""

        metadata = dict(payload.get("metadata") or {})
        if metadata.get("contains_real_phi"):
            raise ValueError("surrogate-quality fixtures must not contain real PHI")
        surrogates = {
            str(key): str(value)
            for key, value in (payload.get("surrogates") or {}).items()
        }
        if not surrogates:
            raise ValueError("surrogate-quality record must contain surrogates")
        return cls(
            record_id=str(payload["record_id"]),
            language=str(payload["language"]).strip().lower(),
            locale=str(payload.get("locale") or payload["language"]),
            surrogates=surrogates,
            expected=dict(payload.get("expected") or {}),
            metadata=metadata,
        )

    def field(self, name: str) -> str:
        """Return a surrogate field value, or an empty string."""

        return str(self.surrogates.get(name, "")).strip()


@dataclass(frozen=True)
class SurrogateQualityCheck:
    """Result for one quality dimension on one surrogate record."""

    dimension: str
    passed: bool
    score: float
    reason: str = "ok"
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "dimension": self.dimension,
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class SurrogateRecordResult:
    """Quality result for one surrogate fixture record."""

    record_id: str
    language: str
    checks: tuple[SurrogateQualityCheck, ...]

    @property
    def passed(self) -> bool:
        """Return whether every dimension passed."""

        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "record_id": self.record_id,
            "language": self.language,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class LocaleSurrogateQualityReport:
    """Aggregated surrogate-quality scores for one language."""

    language: str
    record_results: tuple[SurrogateRecordResult, ...]

    @property
    def total_records(self) -> int:
        """Return the number of records evaluated for this language."""

        return len(self.record_results)

    @property
    def passed_records(self) -> int:
        """Return the number of fully passing records."""

        return sum(1 for result in self.record_results if result.passed)

    @property
    def pass_rate(self) -> float:
        """Return the fully passing record rate."""

        if not self.record_results:
            return 0.0
        return self.passed_records / self.total_records

    @property
    def dimension_scores(self) -> Mapping[str, float]:
        """Return per-dimension pass rates for this language."""

        scores: dict[str, float] = {}
        for dimension in SURROGATE_QUALITY_DIMENSIONS:
            checks = [
                check
                for result in self.record_results
                for check in result.checks
                if check.dimension == dimension
            ]
            scores[dimension] = (
                sum(1 for check in checks if check.passed) / len(checks)
                if checks
                else 0.0
            )
        return scores

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "language": self.language,
            "total_records": self.total_records,
            "passed_records": self.passed_records,
            "pass_rate": self.pass_rate,
            "dimension_scores": dict(self.dimension_scores),
            "record_results": [result.to_dict() for result in self.record_results],
        }


@dataclass(frozen=True)
class SurrogateQualityReport:
    """Per-locale surrogate-quality gate report."""

    locale_reports: Mapping[str, LocaleSurrogateQualityReport]
    required_locales: tuple[str, ...] = DEFAULT_SURROGATE_QUALITY_LOCALES
    min_pass_rate: float = DEFAULT_SURROGATE_QUALITY_PASS_RATE

    @property
    def missing_locales(self) -> tuple[str, ...]:
        """Return required locales that had no evaluated records."""

        return tuple(
            lang
            for lang in self.required_locales
            if lang not in self.locale_reports
            or self.locale_reports[lang].total_records == 0
        )

    @property
    def passed(self) -> bool:
        """Return whether the report satisfies the configured gate."""

        if self.missing_locales:
            return False
        return all(
            self.locale_reports[lang].pass_rate >= self.min_pass_rate
            for lang in self.required_locales
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "passed": self.passed,
            "min_pass_rate": self.min_pass_rate,
            "required_locales": list(self.required_locales),
            "missing_locales": list(self.missing_locales),
            "locales": {
                lang: report.to_dict()
                for lang, report in sorted(self.locale_reports.items())
            },
        }


def load_surrogate_quality_records(
    path: str | Path = DEFAULT_SURROGATE_QUALITY_FIXTURE,
) -> tuple[SurrogateQualityRecord, ...]:
    """Load synthetic surrogate-quality records from JSONL."""

    records: list[SurrogateQualityRecord] = []
    fixture_path = Path(path)
    with fixture_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(SurrogateQualityRecord.from_mapping(payload))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid surrogate-quality fixture "
                    f"{fixture_path}:{line_number}: {exc}"
                ) from exc
    return tuple(records)


def evaluate_surrogate_quality(
    records: Iterable[SurrogateQualityRecord | Mapping[str, Any]] | None = None,
    *,
    fixture_path: str | Path = DEFAULT_SURROGATE_QUALITY_FIXTURE,
    required_locales: Sequence[str] = DEFAULT_SURROGATE_QUALITY_LOCALES,
    min_pass_rate: float = DEFAULT_SURROGATE_QUALITY_PASS_RATE,
) -> SurrogateQualityReport:
    """Evaluate multilingual surrogate naturalness and consistency offline."""

    if records is None:
        normalized_records = load_surrogate_quality_records(fixture_path)
    else:
        normalized_records = tuple(
            record
            if isinstance(record, SurrogateQualityRecord)
            else SurrogateQualityRecord.from_mapping(record)
            for record in records
        )

    by_language: dict[str, list[SurrogateRecordResult]] = {}
    for record in normalized_records:
        result = evaluate_surrogate_record(record)
        by_language.setdefault(result.language, []).append(result)

    locale_reports = {
        language: LocaleSurrogateQualityReport(language, tuple(results))
        for language, results in by_language.items()
    }
    return SurrogateQualityReport(
        locale_reports=locale_reports,
        required_locales=tuple(str(lang).lower() for lang in required_locales),
        min_pass_rate=float(min_pass_rate),
    )


def evaluate_surrogate_record(record: SurrogateQualityRecord) -> SurrogateRecordResult:
    """Evaluate all quality dimensions for one surrogate record."""

    checks = (
        _script_check(record),
        _format_check(record),
        _checksum_check(record),
        _consistency_check(record),
        _naturalness_check(record),
    )
    return SurrogateRecordResult(
        record_id=record.record_id,
        language=record.language,
        checks=checks,
    )


def _script_check(record: SurrogateQualityRecord) -> SurrogateQualityCheck:
    script = _SCRIPT_BY_LANGUAGE.get(record.language)
    if script is None:
        return SurrogateQualityCheck(
            "script",
            False,
            0.0,
            reason="unsupported_script_language",
            details={"language": record.language},
        )
    name = record.field("name")
    ratio = _script_ratio(name, script)
    floor = _MIN_SCRIPT_RATIO[script]
    return SurrogateQualityCheck(
        "script",
        ratio >= floor,
        ratio,
        reason="ok" if ratio >= floor else "script_ratio_below_floor",
        details={"script": script, "floor": floor},
    )


def _format_check(record: SurrogateQualityRecord) -> SurrogateQualityCheck:
    value = record.field("date_of_birth")
    if not value:
        return SurrogateQualityCheck("format", False, 0.0, reason="missing_dob")

    try:
        hint = format_hint(record.language)
    except ValueError as exc:
        return SurrogateQualityCheck("format", False, 0.0, reason=str(exc))

    parsed = parse_date(value, record.language)
    expected = _expected_birth_date(record)
    expected_match = expected is None or parsed.normalized == expected
    order_match = parsed.order == hint.date_order
    passed = parsed.normalized is not None and expected_match and order_match
    details = {
        "date_order": hint.date_order,
        "parsed": parsed.normalized,
        "parsed_order": parsed.order,
        "expected_birth_date": expected,
    }
    if parsed.normalized is None:
        reason = parsed.reason or "unparsed_date"
    elif not order_match:
        reason = "locale_date_order_mismatch"
    elif not expected_match:
        reason = "birth_date_mismatch"
    else:
        reason = "ok"
    return SurrogateQualityCheck(
        "format",
        passed,
        1.0 if passed else 0.0,
        reason=reason,
        details=details,
    )


def _checksum_check(record: SurrogateQualityRecord) -> SurrogateQualityCheck:
    value = record.field("national_id")
    validators = _NATIONAL_ID_VALIDATORS.get(record.language, ())
    if not value:
        return SurrogateQualityCheck("checksum", False, 0.0, reason="missing_id")
    if not validators:
        return SurrogateQualityCheck(
            "checksum",
            False,
            0.0,
            reason="unsupported_id_language",
            details={"language": record.language},
        )

    passed_validator = next(
        (validator.__name__ for validator in validators if validator(value)),
        "",
    )
    return SurrogateQualityCheck(
        "checksum",
        bool(passed_validator),
        1.0 if passed_validator else 0.0,
        reason="ok" if passed_validator else "national_id_checksum_failed",
        details={
            "validators": [validator.__name__ for validator in validators],
            "passed_validator": passed_validator,
        },
    )


def _consistency_check(record: SurrogateQualityRecord) -> SurrogateQualityCheck:
    missing = [
        field_name
        for field_name in ("name", "date_of_birth", "national_id")
        if not record.field(field_name)
    ]
    if missing:
        return SurrogateQualityCheck(
            "consistency",
            False,
            0.0,
            reason="missing_identity_fields",
            details={"missing": missing},
        )

    issues: list[str] = []
    expected_birth = _expected_birth_date(record)
    parsed_birth = parse_date(record.field("date_of_birth"), record.language).normalized
    if expected_birth is not None and parsed_birth != expected_birth:
        issues.append("date_of_birth_expected_mismatch")

    if not _checksum_check(record).passed:
        issues.append("national_id_checksum_failed")

    id_facts = _id_facts(record.language, record.field("national_id"))
    expected_gender = _normalize_gender(record.expected.get("gender"))
    id_gender = id_facts.get("gender")
    if expected_gender and id_gender and id_gender != expected_gender:
        issues.append("gender_mismatch")

    expected_region = str(record.expected.get("region_code") or "")
    id_region = id_facts.get("region_code")
    if expected_region and id_region and id_region != expected_region:
        issues.append("region_mismatch")

    if expected_birth is not None:
        if record.language == "fr" and id_facts:
            if (
                id_facts.get("birth_year_suffix") != expected_birth[0] % 100
                or id_facts.get("birth_month") != expected_birth[1]
            ):
                issues.append("national_id_birth_month_mismatch")
        elif id_facts.get("birth_date") and id_facts["birth_date"] != expected_birth:
            issues.append("national_id_birth_date_mismatch")

    passed = not issues
    return SurrogateQualityCheck(
        "consistency",
        passed,
        1.0 if passed else 0.0,
        reason="ok" if passed else "identity_inconsistent",
        details={"issues": issues, "id_facts": id_facts},
    )


def _naturalness_check(record: SurrogateQualityRecord) -> SurrogateQualityCheck:
    name = record.field("name")
    if not name:
        return SurrogateQualityCheck("naturalness", False, 0.0, reason="missing_name")
    if _PLACEHOLDER_RE.search(_ascii_fold(name)):
        return SurrogateQualityCheck(
            "naturalness",
            False,
            0.0,
            reason="placeholder_name",
        )
    if any(char.isdigit() for char in name):
        return SurrogateQualityCheck(
            "naturalness",
            False,
            0.0,
            reason="name_contains_digit",
        )

    if record.language == "zh":
        han_chars = [char for char in name if _char_in_script(char, "han")]
        passed = 2 <= len(han_chars) <= 6 and _script_ratio(name, "han") >= 0.70
    elif record.language == "hi":
        tokens = [token for token in name.split() if token]
        passed = len(tokens) >= 2 and _script_ratio(name, "devanagari") >= 0.70
    else:
        tokens = re.split(r"[\s'-]+", name.strip())
        passed = len([token for token in tokens if token]) >= 2 and all(
            _has_letter(token) and len(token) >= 2 for token in tokens if token
        )

    return SurrogateQualityCheck(
        "naturalness",
        passed,
        1.0 if passed else 0.0,
        reason="ok" if passed else "implausible_name_shape",
        details={"language": record.language},
    )


def _script_ratio(text: str, script: str) -> float:
    chars = [char for char in text if unicodedata.category(char)[0] in {"L", "M"}]
    if not chars:
        return 0.0
    matched = sum(1 for char in chars if _char_in_script(char, script))
    return matched / len(chars)


def _char_in_script(char: str, script: str) -> bool:
    codepoint = ord(char)
    if script == "latin":
        return "LATIN" in unicodedata.name(char, "")
    if script == "devanagari":
        return 0x0900 <= codepoint <= 0x097F
    if script == "han":
        return (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        )
    return False


def _has_letter(text: str) -> bool:
    return any(unicodedata.category(char).startswith("L") for char in text)


def _ascii_fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text)
    return "".join(char for char in folded if not unicodedata.combining(char))


def _expected_birth_date(record: SurrogateQualityRecord) -> tuple[int, int, int] | None:
    raw = record.expected.get("birth_date")
    if raw is None:
        return None
    try:
        year, month, day = (int(part) for part in str(raw).split("-", 2))
    except ValueError:
        return None
    return year, month, day


def _normalize_gender(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"f", "female", "woman"}:
        return "female"
    if raw in {"m", "male", "man"}:
        return "male"
    return raw


def _id_facts(language: str, value: str) -> Mapping[str, Any]:
    if language == "fr" and validate_french_nir(value):
        cleaned = re.sub(r"[\s.-]", "", value).upper()
        return {
            "gender": "male" if cleaned[0] == "1" else "female",
            "birth_year_suffix": int(cleaned[1:3]),
            "birth_month": int(cleaned[3:5]),
            "region_code": cleaned[5:7],
        }
    if language == "zh" and validate_chinese_resident_identity_card(value):
        cleaned = re.sub(r"[\s-]", "", value).upper()
        sequence = int(cleaned[14:17])
        return {
            "birth_date": (
                int(cleaned[6:10]),
                int(cleaned[10:12]),
                int(cleaned[12:14]),
            ),
            "gender": "male" if sequence % 2 else "female",
            "region_code": cleaned[:6],
        }
    return {}


__all__ = [
    "DEFAULT_SURROGATE_QUALITY_FIXTURE",
    "DEFAULT_SURROGATE_QUALITY_LOCALES",
    "DEFAULT_SURROGATE_QUALITY_PASS_RATE",
    "SURROGATE_QUALITY_DIMENSIONS",
    "LocaleSurrogateQualityReport",
    "SurrogateQualityCheck",
    "SurrogateQualityRecord",
    "SurrogateQualityReport",
    "SurrogateRecordResult",
    "evaluate_surrogate_quality",
    "evaluate_surrogate_record",
    "load_surrogate_quality_records",
]
