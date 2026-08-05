"""Synthetic recognizer gate for the Indian multi-identifier pack.

The suite is fully offline and contains no government registry data. Failure
records expose only offsets, identifier types, and HMAC hashes; raw identifier
surfaces never appear in reports or logs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.detector_plugins import detect_indian_identifiers
from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash

INDIAN_MULTI_ID = "indian_multi_id"
INDIAN_MULTI_ID_FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "golden"
    / "fixtures"
    / "indian"
    / "indian_multi_ids.json"
)
_REPORT_HASH_KEY = "openmed-indian-multi-id-eval"


@dataclass(frozen=True)
class IndianIdExpectedSpan:
    """One expected validator-backed identifier span."""

    identifier_type: str
    entity_type: str
    canonical_label: str
    start: int
    end: int
    text: str = field(repr=False)


@dataclass(frozen=True)
class IndianIdHardNegative:
    """One structurally invalid identifier that must not be accepted."""

    identifier_type: str
    reason: str
    text: str = field(repr=False)


@dataclass(frozen=True)
class IndianIdFixture:
    """Synthetic note with valid spans and structurally invalid negatives."""

    fixture_id: str
    language: str
    text: str = field(repr=False)
    expected: tuple[IndianIdExpectedSpan, ...] = ()
    hard_negatives: tuple[IndianIdHardNegative, ...] = ()
    expected_redacted_text: str = field(default="", repr=False)


@dataclass(frozen=True)
class IndianIdEvalFailure:
    """Privacy-safe failure record for one expected or rejected span."""

    fixture_id: str
    identifier_type: str
    reason: str
    start: int
    end: int
    span_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible failure without raw identifier text."""

        return {
            "end": self.end,
            "fixture_id": self.fixture_id,
            "identifier_type": self.identifier_type,
            "reason": self.reason,
            "span_hash": self.span_hash,
            "start": self.start,
        }


@dataclass(frozen=True)
class IndianIdEvalResult:
    """Aggregate acceptance-gate result for the recognizer pack."""

    fixture_count: int
    expected_span_count: int
    detected_span_count: int
    entity_leakage_count: int
    false_accept_count: int
    failures: tuple[IndianIdEvalFailure, ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether every identifier was redacted with no false accepts."""

        return (
            self.entity_leakage_count == 0
            and self.false_accept_count == 0
            and not self.failures
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, raw-identifier-free report."""

        return {
            "detected_span_count": self.detected_span_count,
            "entity_leakage_count": self.entity_leakage_count,
            "expected_span_count": self.expected_span_count,
            "failure_count": len(self.failures),
            "failures": [failure.to_dict() for failure in self.failures],
            "false_accept_count": self.false_accept_count,
            "fixture_count": self.fixture_count,
            "passed": self.passed,
            "suite": INDIAN_MULTI_ID,
        }


def load_indian_id_fixtures(
    path: str | Path | None = None,
) -> list[IndianIdFixture]:
    """Load and validate the bundled synthetic Indian identifier fixtures."""

    fixture_path = Path(path) if path is not None else INDIAN_MULTI_ID_FIXTURE_PATH
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if payload.get("synthetic") is not True:
        raise ValueError(f"{fixture_path} must declare synthetic=true")

    fixtures = [
        _fixture_from_mapping(item, fixture_path=fixture_path)
        for item in payload.get("fixtures", [])
    ]
    if not fixtures:
        raise ValueError(f"{fixture_path} does not contain fixtures")
    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if len(set(fixture_ids)) != len(fixture_ids):
        raise ValueError(f"{fixture_path} contains duplicate fixture ids")
    return fixtures


def redact_indian_identifiers(
    text: str,
    spans: Sequence[OpenMedSpan],
) -> str:
    """Mask detected identifiers while preserving all non-identifier text."""

    redacted = text
    for span in sorted(spans, key=lambda item: item.start, reverse=True):
        placeholder = f"[{span.canonical_label}]"
        redacted = redacted[: span.start] + placeholder + redacted[span.end :]
    return redacted


def evaluate_indian_id_recognizer(
    fixtures: Sequence[IndianIdFixture],
) -> IndianIdEvalResult:
    """Run exact-span, hard-negative, and entity-leakage acceptance gates."""

    failures: list[IndianIdEvalFailure] = []
    expected_span_count = 0
    detected_span_count = 0
    entity_leakage_count = 0
    false_accept_count = 0

    for fixture in fixtures:
        spans = detect_indian_identifiers(fixture.text, lang=fixture.language)
        detected_span_count += len(spans)
        observed = {
            (span.start, span.end, span.entity_type, span.canonical_label)
            for span in spans
        }
        redacted = redact_indian_identifiers(fixture.text, spans)

        for expected in fixture.expected:
            expected_span_count += 1
            key = (
                expected.start,
                expected.end,
                expected.entity_type,
                expected.canonical_label,
            )
            if key not in observed:
                failures.append(
                    _failure(
                        fixture.fixture_id,
                        expected.identifier_type,
                        expected.text,
                        start=expected.start,
                        end=expected.end,
                        reason="expected_span_missing",
                    )
                )
            if expected.text in redacted:
                entity_leakage_count += 1
                failures.append(
                    _failure(
                        fixture.fixture_id,
                        expected.identifier_type,
                        expected.text,
                        start=expected.start,
                        end=expected.end,
                        reason="entity_leakage",
                    )
                )

        for negative in fixture.hard_negatives:
            for start in _surface_offsets(fixture.text, negative.text):
                end = start + len(negative.text)
                if any(start < span.end and end > span.start for span in spans):
                    false_accept_count += 1
                    failures.append(
                        _failure(
                            fixture.fixture_id,
                            negative.identifier_type,
                            negative.text,
                            start=start,
                            end=end,
                            reason="hard_negative_accepted",
                        )
                    )

        if redacted != fixture.expected_redacted_text:
            failures.append(
                IndianIdEvalFailure(
                    fixture_id=fixture.fixture_id,
                    identifier_type="suite",
                    reason="redacted_text_mismatch",
                    start=0,
                    end=len(fixture.text),
                    span_hash=hmac_text_hash(fixture.text, _REPORT_HASH_KEY),
                )
            )

    return IndianIdEvalResult(
        fixture_count=len(fixtures),
        expected_span_count=expected_span_count,
        detected_span_count=detected_span_count,
        entity_leakage_count=entity_leakage_count,
        false_accept_count=false_accept_count,
        failures=tuple(failures),
    )


def run_indian_id_evaluation(
    path: str | Path | None = None,
) -> IndianIdEvalResult:
    """Load the bundled fixtures and run the Indian identifier gate."""

    return evaluate_indian_id_recognizer(load_indian_id_fixtures(path))


def indian_id_suite_metadata(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Return discoverable metadata for the offline synthetic suite."""

    fixture_path = Path(path) if path is not None else INDIAN_MULTI_ID_FIXTURE_PATH
    return {
        "bundles_registry_data": False,
        "fixture_path": str(fixture_path),
        "report_fields": ["offsets", "hmac_hashes", "identifier_types"],
        "suite": INDIAN_MULTI_ID,
        "synthetic": True,
    }


def _fixture_from_mapping(
    payload: Mapping[str, Any],
    *,
    fixture_path: Path,
) -> IndianIdFixture:
    metadata = payload.get("metadata") or {}
    if metadata.get("synthetic") is not True:
        raise ValueError(f"{fixture_path} contains a non-synthetic fixture")
    if metadata.get("contains_real_phi") is not False:
        raise ValueError(f"{fixture_path} must declare contains_real_phi=false")

    text = str(payload["text"])
    expected = tuple(
        IndianIdExpectedSpan(
            identifier_type=str(item["identifier_type"]),
            entity_type=str(item["entity_type"]),
            canonical_label=str(item["canonical_label"]),
            start=int(item["start"]),
            end=int(item["end"]),
            text=str(item["text"]),
        )
        for item in payload.get("valid", [])
    )
    for span in expected:
        if text[span.start : span.end] != span.text:
            raise ValueError(
                f"{fixture_path}: offset mismatch in {payload.get('id', '<unknown>')}"
            )

    hard_negatives = tuple(
        IndianIdHardNegative(
            identifier_type=str(item["identifier_type"]),
            reason=str(item["reason"]),
            text=str(item["text"]),
        )
        for item in payload.get("hard_negatives", [])
    )
    if not expected or not hard_negatives:
        raise ValueError(
            f"{fixture_path}: every fixture needs valid and invalid identifiers"
        )
    return IndianIdFixture(
        fixture_id=str(payload["id"]),
        language=str(payload.get("language") or "en"),
        text=text,
        expected=expected,
        hard_negatives=hard_negatives,
        expected_redacted_text=str(payload["expected_output"]["text"]),
    )


def _surface_offsets(text: str, surface: str) -> tuple[int, ...]:
    offsets: list[int] = []
    start = 0
    while True:
        found = text.find(surface, start)
        if found < 0:
            return tuple(offsets)
        offsets.append(found)
        start = found + len(surface)


def _failure(
    fixture_id: str,
    identifier_type: str,
    surface: str,
    *,
    start: int,
    end: int,
    reason: str,
) -> IndianIdEvalFailure:
    return IndianIdEvalFailure(
        fixture_id=fixture_id,
        identifier_type=identifier_type,
        reason=reason,
        start=start,
        end=end,
        span_hash=hmac_text_hash(surface, _REPORT_HASH_KEY),
    )


__all__ = [
    "INDIAN_MULTI_ID",
    "INDIAN_MULTI_ID_FIXTURE_PATH",
    "IndianIdEvalFailure",
    "IndianIdEvalResult",
    "IndianIdExpectedSpan",
    "IndianIdFixture",
    "IndianIdHardNegative",
    "evaluate_indian_id_recognizer",
    "indian_id_suite_metadata",
    "load_indian_id_fixtures",
    "redact_indian_identifiers",
    "run_indian_id_evaluation",
]
